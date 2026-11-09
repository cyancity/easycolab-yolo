#!/usr/bin/env python3
"""
Local Headless Google Colab Bridge for EasyColab macOS App.
Powered by google-colab-cli (inspired by FireTable/comfy-colab-kit).
Supports:
  - auth_status : Check OAuth login state, real CU balance, hourly burn rate, and active VMs
  - auth_url    : Generate Google OAuth2 authorization URL for 1-click browser login
  - auth_code   : Exchange Google authorization code and save ~/.config/colab-cli/token.json
  - wake        : Provision/reuse Colab GPU VM (G4/H100/A100/L4/T4), deploy app.py, and return gradio.live URL
  - keepalive   : Refresh Colab runtime proxy & kernel activity while active
  - stop_all    : Unassign and terminate all active Colab VMs on the account (verify 0 CU/hr burn)
"""
import os
import re
import sys
import json
import time
import uuid
import argparse
import tempfile
import urllib.request
from importlib import resources
from datetime import datetime, timezone

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


def _ensure_utf8_stdio():
    """确保 Windows / macOS 子进程管道输出始终采用 UTF-8 编码，避免 emoji 在 GBK 代码页下抛出 UnicodeEncodeError"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_ensure_utf8_stdio()


def _sync_macos_system_proxies_to_env():
    """
    macOS 原生 GUI 应用启动子进程时默认不包含终端的 HTTP_PROXY / HTTPS_PROXY 环境变量，
    而 websocket-client (Jupyter 内核 WebSocket 通道) 仅从 os.environ 读取代理配置。
    此处自动读取 macOS / Windows 系统级代理 (scutil --proxy / Internet Settings / urllib.request.getproxies) 并注入环境变量。
    """
    try:
        proxies = urllib.request.getproxies()
        for k in ("http", "https"):
            if k in proxies and proxies[k]:
                os.environ.setdefault(f"{k}_proxy", proxies[k])
                os.environ.setdefault(f"{k.upper()}_PROXY", proxies[k])
    except Exception:
        pass


_sync_macos_system_proxies_to_env()

from colab_cli.auth import PUBLIC_SCOPES, TOKEN_CONFIG_PATH, REMOTE_REDIRECT_URI
from colab_cli.common import state
from colab_cli.client import Accelerator, Variant, Shape, resolve_assign_shape, PostAssignmentResponse, ColabRequestError, TooManyAssignmentsError
from colab_cli.state import SessionState
from colab_cli.runtime import ColabRuntime
from colab_cli.contents import ContentsClient


def _load_client_config() -> dict:
    cfg_path = os.path.expanduser("~/.colab-cli-oauth-config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    res = resources.files("colab_cli").joinpath("oauth_config.json")
    return json.loads(res.read_text(encoding="utf-8"))


def _has_valid_token() -> bool:
    if not os.path.exists(TOKEN_CONFIG_PATH):
        return False
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH, PUBLIC_SCOPES)
        if creds and creds.valid:
            return True
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            return True
    except Exception:
        return False
    return False


PKCE_VERIFIER_PATH = os.path.expanduser("~/.config/colab-cli/pkce_verifier.txt")
ACCOUNT_PROFILE_CACHE_PATH = os.path.expanduser("~/.config/colab-cli/account_profile.json")
# 多账号凭据库：保存每个登录过的账号 OAuth 凭据，实现「切号不丢号」
ACCOUNTS_STORE_PATH = os.path.expanduser("~/.config/colab-cli/accounts.json")


def _load_accounts_store() -> dict:
    """读取多账号凭据库；文件缺失或损坏时返回空库，保证调用方逻辑不中断"""
    store = {"version": 1, "accounts": []}
    if not os.path.exists(ACCOUNTS_STORE_PATH):
        return store
    try:
        with open(ACCOUNTS_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("accounts"), list):
            store["version"] = int(data.get("version") or 1)
            store["accounts"] = [a for a in data["accounts"] if isinstance(a, dict)]
    except Exception:
        pass
    return store


def _save_accounts_store(store: dict):
    """多账号凭据库落盘：同目录临时文件 + os.replace 原子替换，权限 0600 尽力而为"""
    dir_path = os.path.dirname(ACCOUNTS_STORE_PATH)
    os.makedirs(dir_path, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="accounts_", suffix=".tmp", dir=dir_path)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False)
        try:
            os.chmod(tmp_path, 0o600)
        except Exception:
            pass
        os.replace(tmp_path, ACCOUNTS_STORE_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def _upsert_account(email: str, token_json: str):
    """以邮箱为唯一键把账号凭据 upsert 进多账号库，不影响其他已保存账号"""
    email = (email or "").strip()
    if not email or not token_json:
        return
    store = _load_accounts_store()
    accounts = store.get("accounts", [])
    now_iso = datetime.now(timezone.utc).isoformat()
    for a in accounts:
        if str(a.get("email") or "").strip().lower() == email.lower():
            a["token_json"] = token_json
            a["updated_at"] = now_iso
            break
    else:
        accounts.append({"email": email, "token_json": token_json, "updated_at": now_iso})
    store["accounts"] = accounts
    _save_accounts_store(store)


def _remove_account_from_store(email: str) -> bool:
    """从多账号库中移除指定邮箱条目，返回是否确有删除（其余账号一律保留）"""
    email = (email or "").strip().lower()
    if not email:
        return False
    store = _load_accounts_store()
    accounts = store.get("accounts", [])
    kept = [a for a in accounts if str(a.get("email") or "").strip().lower() != email]
    if len(kept) == len(accounts):
        return False
    store["accounts"] = kept
    _save_accounts_store(store)
    return True


def _clear_local_session_store():
    """切换或退出 Google 账号时，清理本地缓存的旧账号 Colab 会话记录"""
    try:
        existing = state.store.list()
        if isinstance(existing, dict):
            for name in list(existing.keys()):
                state.store.remove(name)
    except Exception:
        pass
    state._client = None
    state._sessions = None


def _get_account_email() -> str:
    """获取当前绑定 Google 账号邮箱（按 refresh_token 指纹本地缓存，避免重复请求）"""
    if not os.path.exists(TOKEN_CONFIG_PATH):
        return ""
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH, PUBLIC_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        tok_fp = (creds.refresh_token or creds.token or "")[-20:]
        if tok_fp and os.path.exists(ACCOUNT_PROFILE_CACHE_PATH):
            try:
                with open(ACCOUNT_PROFILE_CACHE_PATH, "r", encoding="utf-8") as cf:
                    cached = json.load(cf)
                if cached.get("fp") == tok_fp and cached.get("email"):
                    return str(cached["email"])
            except Exception:
                pass
        if creds and creds.token:
            r = requests.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {creds.token}"},
                timeout=5,
            )
            if r.status_code == 200:
                email = str(r.json().get("email") or "").strip()
                if email and tok_fp:
                    os.makedirs(os.path.dirname(ACCOUNT_PROFILE_CACHE_PATH), exist_ok=True)
                    with open(ACCOUNT_PROFILE_CACHE_PATH, "w", encoding="utf-8") as cf:
                        json.dump({"fp": tok_fp, "email": email}, cf)
                return email
    except Exception:
        pass
    return ""


def cmd_auth_url():
    client_config = _load_client_config()
    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    # 显式传入 select_account consent，强制弹出 Google 多账号选择器以支持自由切换账号
    auth_url, _ = flow.authorization_url(prompt="select_account consent", token_usage="remote")
    os.makedirs(os.path.dirname(PKCE_VERIFIER_PATH), exist_ok=True)
    if getattr(flow, "code_verifier", None):
        with open(PKCE_VERIFIER_PATH, "w", encoding="utf-8") as f:
            f.write(flow.code_verifier)
    print(json.dumps({"ok": True, "auth_url": auth_url}))


def cmd_auth_code(code: str):
    try:
        client_config = _load_client_config()
        flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
        flow.redirect_uri = REMOTE_REDIRECT_URI
        if os.path.exists(PKCE_VERIFIER_PATH):
            with open(PKCE_VERIFIER_PATH, "r", encoding="utf-8") as f:
                flow.code_verifier = f.read().strip()
        flow.fetch_token(code=code.strip())
        creds = flow.credentials
        _clear_local_session_store()
        for p in (ACCOUNT_PROFILE_CACHE_PATH, PKCE_VERIFIER_PATH):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
        with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        # 新登录账号收进多账号库（upsert，不覆盖其他账号），实现切号不丢号
        try:
            email = _get_account_email()
            if email:
                # _get_account_email 内部可能刷新凭据并回写 token.json，取磁盘上最新的一份入库
                with open(TOKEN_CONFIG_PATH, "r", encoding="utf-8") as f:
                    _upsert_account(email, f.read())
        except Exception:
            pass
        return cmd_auth_status()
    except Exception as e:
        print(json.dumps({"ok": False, "logged_in": False, "error": str(e)}))


def cmd_auth_logout():
    # 删除 token 前先识别当前账号，只把该账号从多账号库移除，其余已保存账号保留
    try:
        email = _get_account_email()
        if email:
            _remove_account_from_store(email)
    except Exception:
        pass
    _clear_local_session_store()
    for p in (TOKEN_CONFIG_PATH, PKCE_VERIFIER_PATH, ACCOUNT_PROFILE_CACHE_PATH):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    print(json.dumps({"ok": True, "logged_in": False, "email": ""}))


def _safe_sync_sessions():
    for attempt in range(3):
        try:
            return state.sync_sessions()
        except Exception:
            if attempt == 2:
                # store.list() 返回 {name: SessionState}，直接复制即可
                return dict(state.store.list()), []
            time.sleep(1.2)


def cmd_auth_status():
    if not _has_valid_token():
        print(json.dumps({"ok": True, "logged_in": False}))
        return

    email = _get_account_email()
    # 升级前就已登录的账号不在多账号库里，首次换号会丢；这里补录一次
    if email and not any(
        str(a.get("email") or "").strip().lower() == email.lower()
        for a in _load_accounts_store().get("accounts", [])
    ):
        try:
            with open(TOKEN_CONFIG_PATH, "r", encoding="utf-8") as f:
                _upsert_account(email, f.read())
        except Exception:
            pass
    # CU 用量查询单独容错：免费账号无 CU 计量时会异常，降级为 null 并附 usage_error，不拖垮整个状态输出
    usage_error = None
    paid_balance = burn_rate_hourly = active_assignments = None
    eligible_gpus = []
    try:
        info = _fetch_ccu_info_raw()
        paid_balance = round(float(info.get("currentBalance") or 0), 2)
        burn_rate_hourly = round(float(info.get("consumptionRateHourly") or 0), 2)
        active_assignments = int(info.get("assignmentsCount") or 0)
        eligible_gpus = [str(g).upper() for g in info.get("eligibleGpus") or []]
    except Exception as e:
        usage_error = str(e)[:120]

    local_sessions, assignments = _safe_sync_sessions()
    name_by_ep = {s.endpoint: s.name for s in local_sessions.values()}
    sess_list = []
    for a in assignments:
        sess_list.append({
            "name": name_by_ep.get(a.endpoint, "colab-vm"),
            "endpoint": a.endpoint,
            "accelerator": a.accelerator.value,
            "variant": a.variant.name,
            "shape": a.machine_shape.name,
        })

    payload = {
        "ok": True,
        "logged_in": True,
        "email": email,
        "paid_balance": paid_balance,
        "burn_rate_hourly": burn_rate_hourly,
        "active_assignments": active_assignments,
        "eligible_gpus": eligible_gpus,
        "sessions": sess_list,
    }
    if usage_error:
        payload["usage_error"] = usage_error
    print(json.dumps(payload))


def cmd_accounts_list():
    """列出多账号库中已保存的账号与当前活跃账号（只输出邮箱与 active 标记，绝不输出任何 token 内容）"""
    store = _load_accounts_store()
    active_email = _get_account_email()
    accounts = [
        {
            "email": str(a.get("email") or ""),
            "active": str(a.get("email") or "").strip().lower() == active_email.strip().lower(),
        }
        for a in store.get("accounts", [])
        if a.get("email")
    ]
    print(json.dumps({"ok": True, "active_email": active_email, "accounts": accounts}))


def cmd_accounts_switch(email: str):
    """切换到多账号库中的指定账号：凭据写入 token.json 并清理旧会话缓存，随后输出完整登录状态"""
    email = (email or "").strip()
    store = _load_accounts_store()
    entry = None
    for a in store.get("accounts", []):
        if str(a.get("email") or "").strip().lower() == email.lower():
            entry = a
            break
    if not entry:
        print(json.dumps({"ok": False, "error": f"account not found: {email}"}))
        return
    try:
        creds = Credentials.from_authorized_user_info(json.loads(entry.get("token_json") or ""), PUBLIC_SCOPES)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)[:200]}))
        return
    try:
        # 库里凭据可能已过期，先刷新；刷新失败就不动 token.json，保持当前账号可用
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"该账号凭据已失效，请重新登录：{str(e)[:120]}"}))
        return
    token_json = creds.to_json()
    os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
    with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(token_json)
    try:
        _upsert_account(entry.get("email"), token_json)
    except Exception:
        pass
    _clear_local_session_store()
    cmd_auth_status()


def cmd_accounts_remove(email: str):
    """从多账号库移除指定账号；若移除的是当前活跃账号，同时清理本地 token 与账号缓存"""
    email = (email or "").strip()
    target = ""
    for a in _load_accounts_store().get("accounts", []):
        if str(a.get("email") or "").strip().lower() == email.lower():
            target = str(a.get("email") or "")
            break
    if not target:
        print(json.dumps({"ok": False, "error": f"account not found: {email}"}))
        return
    active_email = _get_account_email()
    was_active = bool(active_email) and active_email.strip().lower() == email.lower()
    _remove_account_from_store(target)
    if was_active:
        # 当前活跃账号被移除后，active_email 失去对应 token，直接置空避免再查一次网络
        for p in (TOKEN_CONFIG_PATH, ACCOUNT_PROFILE_CACHE_PATH):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        active_email = ""
    print(json.dumps({
        "ok": True,
        "removed_email": target,
        "active_email": active_email,
    }))


def _build_rclone_drive_conf() -> str:
    """
    复用本地已授权的 Google Colab CLI OAuth 凭据 (含 drive.file 权限域)，
    生成云端 rclone Google Drive 配置，使无浏览器唤醒的 G4 实例也能自动挂载/同步 Google Drive，
    持久化保存 ComfyUI 自定义节点、模型与工作流。
    """
    if not os.path.exists(TOKEN_CONFIG_PATH):
        return ""
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH, PUBLIC_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        raw = json.loads(creds.to_json())
        scopes = raw.get("scopes") or []
        drive_scope = "drive" if any(s.endswith("/drive") for s in scopes) else (
            "drive.file" if any("drive.file" in s for s in scopes) else ""
        )
        if not drive_scope or not raw.get("token") or not raw.get("refresh_token"):
            return ""
        tok_obj = {
            "access_token": raw["token"],
            "token_type": "Bearer",
            "refresh_token": raw["refresh_token"],
        }
        if raw.get("expiry"):
            tok_obj["expiry"] = raw["expiry"]
        lines = [
            "[easycolab_gdrive]",
            "type = drive",
            f"client_id = {raw.get('client_id', '')}",
            f"client_secret = {raw.get('client_secret', '')}",
            f"scope = {drive_scope}",
            f"token = {json.dumps(tok_obj, separators=(',', ':'))}",
            "",
        ]
        return "\n".join(lines)
    except Exception:
        return ""


def cmd_stop_all():
    if not _has_valid_token():
        print(json.dumps({"ok": False, "error": "尚未登录 Google Colab CLI"}))
        return

    try:
        local_sessions, assignments = state.sync_sessions()
        stopped_count = 0
        for s in list(local_sessions.values()):
            try:
                rt = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id)
                try:
                    rt.execute_code(
                        "import urllib.request\n"
                        "try:\n"
                        # API 挂在 /k/<密钥> 下，裸路径 403；密钥由 app.py 启动时写入该文件
                        "    _k = open('/content/.easycolab_runtime_secret').read().strip()\n"
                        "    urllib.request.urlopen(urllib.request.Request("
                        "'http://127.0.0.1:7860/k/' + _k + '/minimax_api/comfyui/sync_drive', "
                        "data=b'{}', headers={'Content-Type':'application/json'}, method='POST'), timeout=18)\n"
                        "except Exception:\n"
                        "    pass\n",
                        allow_stdin=False,
                        timeout=22,
                    )
                except Exception:
                    pass
                rt.stop(shutdown_kernel=True)
            except Exception:
                pass
            try:
                state.client.unassign(s.endpoint)
                stopped_count += 1
            except Exception:
                pass
            state.store.remove(s.name)

        # 同时清理所有未在本地登记的云端孤儿实例 (Orphan VMs)
        for a in assignments:
            try:
                state.client.unassign(a.endpoint)
                stopped_count += 1
            except Exception:
                pass

        info = state.client.get_consumption_user_info()
        print(json.dumps({
            "ok": True,
            "stopped_count": stopped_count,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": int(info.assignments_count),
            "message": f"🛑 已彻底释放全部云端实例！当前计费速率: {info.consumption_rate_hourly:.2f} CU/小时",
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


def _wait_for_runtime_ready(session: SessionState, timeout_s: int = 45) -> bool:
    """
    等待新分配的 Colab 容器内部 Jupyter Server (/api/kernels) 完成启动。
    规避容器刚启动前几秒出现的 SSLEOFError / ConnectionReset / 502 / 503。
    若检测到 404 (实例如已被销毁)，立即返回 False 以便重新分配新实例。
    """
    base_url = session.url.rstrip("/")
    probe_url = f"{base_url}/api/kernels"
    headers = {
        "Accept": "application/json",
        "X-Colab-Client-Agent": "colab-cli",
        "X-Colab-Runtime-Proxy-Token": session.token,
        "Authorization": f"Bearer {session.token}",
    }
    params = {"authuser": "0", "colab-runtime-proxy-token": session.token}
    deadline = time.time() + timeout_s
    not_found_count = 0

    while time.time() < deadline:
        try:
            r = requests.get(probe_url, headers=headers, params=params, timeout=8)
            if r.status_code == 200:
                return True
            if r.status_code in (404, 410):
                not_found_count += 1
                if not_found_count >= 2:
                    return False
        except requests.exceptions.RequestException:
            pass
        time.sleep(2.0)
    return False


GPU_ACCEL_MAP = {
    "h100": Accelerator.H100,
    "g4": Accelerator.G4,
    "a100": Accelerator.A100,
    "l4": Accelerator.L4,
    "t4": Accelerator.T4,
}

# G4 (RTX PRO 6000 Blackwell 96GB)、H100、L4 在 Colab 后端均为固定单规格大内存实例，传 shape=hm 会报 400 Bad Request
SINGLE_SHAPE_GPUS = frozenset({Accelerator.G4, Accelerator.H100, Accelerator.L4})


def _resolve_gpu_accel(gpu_type: str) -> Accelerator:
    return GPU_ACCEL_MAP.get((gpu_type or "g4").strip().lower(), Accelerator.G4)


class AllocationError(Exception):
    """分配失败，消息已转成用户能看懂的中文说明"""


def _fetch_ccu_info_raw() -> dict:
    """GET /tun/m/ccu-info 原始 JSON。colab_cli 的 CcuInfo 会丢掉 eligibleGpus/ineligibleGpus，这里自己解析"""
    r = state.client.session.request(
        "GET",
        "https://colab.research.google.com/tun/m/ccu-info",
        params={"authuser": "0"},
        headers={"Accept": "application/json", "X-Colab-Client-Agent": "colab-cli"},
        timeout=15,
    )
    r.raise_for_status()
    body = r.text[len(")]}'\n"):] if r.text.startswith(")]}'\n") else r.text
    return json.loads(body or "{}")


def _check_gpu_eligible(accel: Accelerator):
    """分配前查账号资格：不在 eligibleGpus 里的卡直接拒绝，不发 assign 请求（避免无效请求累积被风控）"""
    try:
        info = _fetch_ccu_info_raw()
    except Exception:
        return  # 查询失败不阻断分配，交给 assign 自己报错
    eligible = [str(g).upper() for g in info.get("eligibleGpus") or []]
    if eligible and accel.value.upper() not in eligible:
        raise AllocationError(
            f"当前账号没有 {accel.value} 的使用资格，可用：{', '.join(eligible)}。请换卡或切换到有该资格的账号。"
        )


def _explain_assign_error(accel: Accelerator, err: Exception) -> str:
    status = getattr(getattr(err, "response", None), "status_code", None)
    if status == 503:
        return (
            f"Colab 暂时分不出 {accel.value}（HTTP 503，已自动重试 3 次）。"
            "通常是该型号容量紧张；若所有付费卡都 503 而 T4 正常，也可能是账号被临时限制，"
            "可在 Colab 网页里连一次该 GPU 查看提示。请稍后再试或换卡，不要短时间反复点。"
        )
    if status == 400:
        return f"Colab 拒绝了 {accel.value}（HTTP 400）：账号没有该型号的资格或参数不支持，请换卡。"
    return f"申请 {accel.value} 失败：{err}"


def _allocate_fresh_session(session_name: str, gpu_type: str) -> SessionState:
    accel = _resolve_gpu_accel(gpu_type)
    _check_gpu_eligible(accel)
    shape = None if accel in SINGLE_SHAPE_GPUS else resolve_assign_shape(accel, high_mem=True)
    gpu_desc = "G4 (RTX PRO 6000 · 96GB)" if accel == Accelerator.G4 else accel.value
    print(f"WAKE_LOG: 🚀 正在向 Google Colab 申请分配 {gpu_desc} 实例...", flush=True)
    try:
        try:
            res = state.client.assign(uuid.uuid4(), variant=Variant.GPU, accelerator=accel, shape=shape)
        except ColabRequestError as req_err:
            if shape is not None and getattr(getattr(req_err, "response", None), "status_code", None) == 400:
                shape = None
                res = state.client.assign(uuid.uuid4(), variant=Variant.GPU, accelerator=accel, shape=None)
            else:
                raise
    except TooManyAssignmentsError:
        raise AllocationError("Colab 拒绝分配（HTTP 412）：账号下活动实例过多或触发了临时用量限制，先释放其他实例再试。")
    except ColabRequestError as req_err:
        raise AllocationError(_explain_assign_error(accel, req_err))
    if isinstance(res, PostAssignmentResponse):
        token = res.runtime_proxy_info.token
        url = res.runtime_proxy_info.url
        endpoint = res.endpoint
        exp = res.runtime_proxy_info.expires_at()
    else:
        token = res.runtime_proxy_info.token if hasattr(res, "runtime_proxy_info") else getattr(res, "runtime_proxy_token", "")
        url = res.runtime_proxy_info.url if hasattr(res, "runtime_proxy_info") else ""
        endpoint = res.endpoint
        exp = res.runtime_proxy_info.expires_at() if hasattr(res, "runtime_proxy_info") else None

    s = SessionState(
        name=session_name,
        token=token,
        url=url,
        endpoint=endpoint,
        token_expires_at=exp,
        variant=Variant.GPU.value,
        accelerator=accel.value,
        machine_shape=Shape.HIGH_RAM.name if shape == Shape.HIGH_RAM else Shape.STANDARD.name,
    )
    state.store.add(s)
    print(f"WAKE_LOG: ✅ 成功分配 {accel.value} 云端虚拟机！正在等待容器内核就绪...", flush=True)
    return s


def cmd_wake(gpu_type: str, app_py_path: str, session: str = ""):
    """
    无浏览器一键唤醒 Colab GPU 并自动部署 EasyColab 多模型服务端：
    1. 幂等检查：若已有运行中的会话且探活成功则直接复用，否则自动清理失效会话并分配新实例
    2. 等待容器 Jupyter 端点 (/api/kernels) TLS/HTTP 握手就绪，规避 SSLEOFError
    3. 上传本地最新 app.py 到 /content/app.py
    4. 在远端内核安装依赖并后台拉起 app.py，提取 https://xxxx.gradio.live 隧道链接返回给 Mac 客户端
    """
    if not _has_valid_token():
        print("WAKE_ERROR: 尚未授权登录 Google 账号，请先点击「🔑 绑定账号」", flush=True)
        return

    try:
        session_name = session or "easycolab"
        print("WAKE_LOG: 🔍 正在检查账号下已有的 Colab 实例...", flush=True)
        local_sessions, assignments = _safe_sync_sessions()
        if session:
            # 工作台按会话名精确管理同账号下的多个实例，不接管其他会话或云端孤儿实例
            local_sessions = {k: v for k, v in local_sessions.items() if k == session}
            assignments = []

        target_session = (
            local_sessions.get(session_name)
            or local_sessions.get("aevix")
            or local_sessions.get("minimax")
        )
        if target_session:
            session_name = target_session.name
        elif local_sessions:
            target_session = list(local_sessions.values())[0]
            session_name = target_session.name

        # 若本地未记录但云端已有活跃分配，直接纳管该云端实例
        if not target_session and assignments:
            a = assignments[0]
            target_session = SessionState(
                name=session_name,
                token=a.runtime_proxy_info.token,
                url=a.runtime_proxy_info.url,
                endpoint=a.endpoint,
                token_expires_at=a.runtime_proxy_info.expires_at(),
                variant=Variant.GPU.value,
                accelerator=a.accelerator.value,
                machine_shape=Shape.HIGH_RAM.name if a.machine_shape == Shape.HIGH_RAM else Shape.STANDARD.name,
            )
            state.store.add(target_session)

        requested_accel = _resolve_gpu_accel(gpu_type)
        # 先确认资格再动旧实例，避免"旧卡已释放、新卡又申请不到"
        _check_gpu_eligible(requested_accel)
        if target_session:
            cur_acc = (target_session.accelerator or "").strip().upper()
            if cur_acc != requested_accel.value.upper():
                print(
                    f"WAKE_LOG: 🔄 当前已分配实例型号为 [{cur_acc or 'CPU'}]，正在释放并切换为您选择的 [{requested_accel.value}]...",
                    flush=True,
                )
                try:
                    state.client.unassign(target_session.endpoint)
                except Exception:
                    pass
                state.prune_session(target_session.name)
                target_session = None
            else:
                print(f"WAKE_LOG: ♻️ 正在检查已有实例 [{target_session.name}] ({target_session.accelerator}) 连通性...", flush=True)
                if not _wait_for_runtime_ready(target_session, timeout_s=15):
                    print("WAKE_LOG: ⚠️ 旧实例已失效或正在销毁，正在自动清理并重新分配新实例...", flush=True)
                    try:
                        state.client.unassign(target_session.endpoint)
                    except Exception:
                        pass
                    state.prune_session(target_session.name)
                    target_session = None

        if not target_session:
            target_session = _allocate_fresh_session(session_name, gpu_type)
            if not _wait_for_runtime_ready(target_session, timeout_s=50):
                print("WAKE_LOG: ⏳ 容器仍在初始化，继续尝试连接内核...", flush=True)

        print("WAKE_LOG: 正在上传最新多模型开源服务端代码 (app.py)...", flush=True)
        contents = ContentsClient(target_session)
        for up_attempt in range(4):
            try:
                contents.upload(app_py_path, "content/app.py")
                break
            except Exception as up_err:
                if up_attempt == 3:
                    raise up_err
                time.sleep(2.5)
        # runtime/ 下的附加模块（GPU 采集、Flash-Next 运行时）与 app.py 放在同一目录，app.py 按需 import
        rt_dir = os.path.join(os.path.dirname(os.path.abspath(app_py_path)), "runtime")
        for name in sorted(os.listdir(rt_dir)) if os.path.isdir(rt_dir) else []:
            if name.endswith(".py"):
                contents.upload(os.path.join(rt_dir, name), "content/" + name)

        # app.py 是薄入口，逻辑在同目录的 colab_server/ 包里。Jupyter contents 接口不会自动建目录，
        # 所以包文件先以平铺名 colab_server__<模块>.py 传到 /content，启动脚本再装回 /content/colab_server/
        pkg_local = os.path.join(os.path.dirname(os.path.abspath(app_py_path)), "colab_server")
        pkg_files = sorted(n for n in os.listdir(pkg_local) if n.endswith(".py")) if os.path.isdir(pkg_local) else []
        with open(app_py_path, "r", encoding="utf-8") as _af:
            uses_pkg = re.search(r"^from colab_server\b", _af.read(), re.M) is not None
        if uses_pkg and not pkg_files:
            raise RuntimeError("app.py 依赖 colab_server/ 包，但 %s 不存在或为空" % pkg_local)
        for name in pkg_files:
            contents.upload(os.path.join(pkg_local, name), "content/colab_server__" + name)

        rclone_conf_text = _build_rclone_drive_conf()
        if rclone_conf_text:
            import tempfile
            tmp_rc_fd, tmp_rc_path = tempfile.mkstemp(prefix="easycolab_rc_", suffix=".conf")
            try:
                with os.fdopen(tmp_rc_fd, "w", encoding="utf-8") as rcf:
                    rcf.write(rclone_conf_text)
                os.chmod(tmp_rc_path, 0o600)
                contents.upload(tmp_rc_path, "content/.easycolab_rclone.conf")
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_rc_path):
                    try:
                        os.remove(tmp_rc_path)
                    except Exception:
                        pass

        print("WAKE_LOG: 正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)...", flush=True)
        boot_code = r"""
import os, sys, time, re, shutil, subprocess

# 0. 配置无网页模式下的 Google Drive 云盘自动挂载/高速同步 (供模型库与 ComfyUI 节点持久化)
rc_stage = "/content/.easycolab_rclone.conf"
if os.path.exists(rc_stage):
    rc_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(rc_dir, mode=0o700, exist_ok=True)
    rc_dst = os.path.join(rc_dir, "rclone.conf")
    shutil.move(rc_stage, rc_dst)
    os.chmod(rc_dst, 0o600)

if os.path.exists(os.path.expanduser("~/.config/rclone/rclone.conf")):
    if not shutil.which("rclone"):
        subprocess.run(
            ["bash", "-lc", "curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip && unzip -q -o /tmp/rclone.zip -d /tmp && cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone && chmod +x /usr/local/bin/rclone && rm -rf /tmp/rclone*"],
            check=False,
        )
    if shutil.which("rclone") and not os.path.ismount("/content/drive/MyDrive"):
        subprocess.run(["rclone", "mkdir", "easycolab_gdrive:EasyColab_Drive"], check=False)
        if os.path.exists("/dev/fuse") and (shutil.which("fusermount3") or shutil.which("fusermount")):
            os.makedirs("/content/drive/MyDrive", exist_ok=True)
            subprocess.Popen(
                [
                    "rclone", "mount", "easycolab_gdrive:EasyColab_Drive", "/content/drive/MyDrive",
                    "--vfs-cache-mode", "writes",
                    "--vfs-write-back", "3s",
                    "--dir-cache-time", "15s",
                    "--drive-chunk-size", "64M",
                    "--buffer-size", "32M",
                    "--allow-non-empty",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(10):
                if os.path.ismount("/content/drive/MyDrive"):
                    print("✅ 已通过 Google OAuth 自动挂载 Google Drive (/content/drive/MyDrive)！", flush=True)
                    break
                time.sleep(0.5)
            else:
                # 容器未开放 FUSE 挂载权限时清理空目录，由 app.py 自动启用 rclone 云盘高速双向同步
                try:
                    if not os.listdir("/content/drive/MyDrive"):
                        os.rmdir("/content/drive/MyDrive")
                except Exception:
                    pass

# 1. 如果尚未安装 DiffSynth-Studio，自动快速安装 (含最新 transformers 以支持 Qwen3.8-27B)
if not os.path.exists("/content/DiffSynth-Studio"):
    print("正在克隆 DiffSynth-Studio...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/modelscope/DiffSynth-Studio.git", "/content/DiffSynth-Studio"], check=True)
    # 大包安装改为后台 + 心跳输出：内核长时间静默会触发桥接侧 iopub 超时（300s），
    # 超时后桥接会换新内核重跑 boot_code，而旧内核仍在装包，两个 pip 并发改坏 site-packages
    _pip = subprocess.Popen([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes", "huggingface_hub", "hf_transfer", "qwen-vl-utils", "einops", "kernels>=0.17.2,<0.18.0", "av", "gradio", "modelscope", "imageio", "imageio-ffmpeg", "scipy", "soundfile", "librosa"])
    while _pip.poll() is None:
        time.sleep(30)
        print("⏳ 依赖安装中，请稍候...", flush=True)
    if _pip.returncode != 0:
        raise RuntimeError("pip install failed with exit code %s" % _pip.returncode)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "fla-core", "flash-linear-attention"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--only-binary=:all:", "causal-conv1d"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/DiffSynth-Studio"], check=True)
else:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        import kernels  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "kernels>=0.17.2,<0.18.0"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "fla-core", "flash-linear-attention"], check=False)

# 2. 检查是否已有运行中的 app.py 或 llama-server，先清理旧进程释放 GPU 显存
subprocess.run(["pkill", "-f", "python.*/content/app.py"], check=False)
subprocess.run(["pkill", "-f", "llama-server"], check=False)
# 等旧进程真正退出再拉起新实例：app.py 退出时有 Drive 同步等清理动作，固定 sleep(1) 可能不够，
# 新进程会因 7860 端口被占而启动失败
for _ in range(10):
    _probe = subprocess.run(["pgrep", "-f", "python.*/content/app.py"], capture_output=True)
    if _probe.returncode != 0:
        break
    time.sleep(1)

# 2.5 把平铺上传的包文件装回 /content/colab_server/（先清掉旧版本，避免残留已删除的模块）
_pkg_files = sorted(f for f in os.listdir("/content") if f.startswith("colab_server__") and f.endswith(".py"))
if _pkg_files:
    shutil.rmtree("/content/colab_server", ignore_errors=True)
    os.makedirs("/content/colab_server")
    for _f in _pkg_files:
        shutil.move("/content/" + _f, "/content/colab_server/" + _f[len("colab_server__"):])

# 3. 后台启动 /content/app.py 并等待 gradio.live 链接生成
log_path = "/content/easycolab_server.log"
if os.path.exists(log_path):
    os.remove(log_path)

with open(log_path, "w") as out_f:
    subprocess.Popen([sys.executable, "-u", "/content/app.py"], cwd="/content", stdout=out_f, stderr=subprocess.STDOUT)

# 服务地址含 /k/<密钥> 路径段（API 全部挂载其下，裸路径 403），提取时必须保留
url_re = re.compile(r"https://[a-zA-Z0-9.-]+\.gradio\.live(?:/k/[A-Za-z0-9_-]+)?")
found_url = None
for _ in range(90):
    time.sleep(2)
    if os.path.exists(log_path):
        txt = open(log_path, "r", errors="replace").read()
        # Gradio 会先打印不带密钥段的 "Running on public URL"，只认带 /k/ 的完整地址
        keyed = [u for u in url_re.findall(txt) if "/k/" in u]
        if keyed:
            found_url = keyed[-1]
            break

if found_url:
    print("WAKE_URL:" + found_url, flush=True)
else:
    tail = open(log_path, "r", errors="replace").read()[-600:] if os.path.exists(log_path) else "no log"
    print("WAKE_ERROR: 启动超时，日志尾部: " + tail, flush=True)
"""
        def _on_kid(kid):
            target_session.kernel_id = kid
            state.store.add(target_session)

        def _on_sid(sid):
            target_session.session_id = sid
            state.store.add(target_session)

        def _run_boot_on_runtime(use_saved_kernel: bool):
            rt = ColabRuntime(
                target_session.url,
                target_session.token,
                session_name=target_session.name,
                history=state.history,
                kernel_id=target_session.kernel_id if use_saved_kernel else None,
                session_id=target_session.session_id if use_saved_kernel else None,
                on_kernel_started=_on_kid,
                on_session_started=_on_sid,
            )
            try:
                return rt.execute_code(boot_code, allow_stdin=False, timeout=300)
            finally:
                try:
                    rt.stop()
                except Exception:
                    pass

        try:
            outputs = _run_boot_on_runtime(use_saved_kernel=True)
        except Exception:
            target_session.kernel_id = None
            target_session.session_id = None
            outputs = _run_boot_on_runtime(use_saved_kernel=False)

        for out in outputs:
            if "text" in out:
                for line in out["text"].splitlines():
                    print(line, flush=True)
            elif out.get("output_type") == "error":
                print("WAKE_ERROR: " + out.get("evalue", "Execution error"), flush=True)

    except Exception as e:
        print(f"WAKE_ERROR: {str(e)}", flush=True)


def cmd_keepalive():
    """
    向活跃的 Colab Headless 实例发送官方 Runtime Proxy 探活与轻量 Jupyter 内核脉冲：
    由于 EasyColab 推理流量走 *.gradio.live 隧道，若长时间无 *.prod.r.colab.dev 代理流量且内核 last_activity 静止，
    Colab 后台闲置回收器可能在连续生成第 2 条视频中途误判实例闲置并强制回收。
    本命令仅在客户端处于活跃窗口或正在执行生成任务时由 Mac 端定期调用。
    """
    if not _has_valid_token():
        print(json.dumps({"ok": True, "logged_in": False}))
        return

    try:
        local_sessions, assignments = _safe_sync_sessions()
        ep_map = {a.endpoint: a for a in assignments}
        touched = 0

        # 若云端有分配但本地无会话记录，自动纳管以便续期
        if not local_sessions and assignments:
            a0 = assignments[0]
            s_auto = SessionState(
                name="easycolab",
                token=a0.runtime_proxy_info.token,
                url=a0.runtime_proxy_info.url,
                endpoint=a0.endpoint,
                token_expires_at=a0.runtime_proxy_info.expires_at(),
                variant=Variant.GPU.value,
                accelerator=a0.accelerator.value,
                machine_shape=Shape.HIGH_RAM.name if a0.machine_shape == Shape.HIGH_RAM else Shape.STANDARD.name,
            )
            state.store.add(s_auto)
            local_sessions = {"easycolab": s_auto}

        for s in list(local_sessions.values()):
            if s.endpoint in ep_map:
                info_proxy = ep_map[s.endpoint].runtime_proxy_info
                s.token = info_proxy.token
                s.url = info_proxy.url
                s.token_expires_at = info_proxy.expires_at()
                state.store.add(s)

            if not s.url or not s.token:
                continue

            base_url = s.url.rstrip("/")
            headers = {
                "Accept": "application/json",
                "X-Colab-Client-Agent": "colab-cli",
                "X-Colab-Runtime-Proxy-Token": s.token,
                "Authorization": f"Bearer {s.token}",
            }
            params = {"authuser": "0", "colab-runtime-proxy-token": s.token}
            try:
                r = requests.get(f"{base_url}/api/kernels", headers=headers, params=params, timeout=10)
                if r.status_code == 200:
                    kernels = r.json() if r.text else []
                    active_kid = None
                    if isinstance(kernels, list) and kernels:
                        known_ids = [k.get("id") for k in kernels if k.get("id")]
                        active_kid = s.kernel_id if (s.kernel_id in known_ids) else known_ids[0]
                        # 清理多余的历史空转内核，仅保留主内核
                        for extra_id in known_ids:
                            if extra_id != active_kid:
                                try:
                                    requests.delete(
                                        f"{base_url}/api/kernels/{extra_id}",
                                        headers=headers,
                                        params=params,
                                        timeout=5,
                                    )
                                except Exception:
                                    pass

                    def _save_kid(kid, _s=s):
                        _s.kernel_id = kid
                        state.store.add(_s)

                    def _save_sid(sid, _s=s):
                        _s.session_id = sid
                        state.store.add(_s)

                    rt = ColabRuntime(
                        s.url,
                        s.token,
                        session_name=s.name,
                        kernel_id=active_kid,
                        session_id=s.session_id,
                        on_kernel_started=_save_kid,
                        on_session_started=_save_sid,
                    )
                    try:
                        rt.execute_code("_easycolab_hb = 1", allow_stdin=False, timeout=12)
                        touched += 1
                    finally:
                        try:
                            rt.stop(shutdown_kernel=False)
                        except Exception:
                            pass
            except Exception:
                pass

        info = state.client.get_consumption_user_info()
        print(json.dumps({
            "ok": True,
            "logged_in": True,
            "touched_kernels": touched,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": int(info.assignments_count),
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "logged_in": True, "error": str(e)}))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth_status")
    sub.add_parser("auth_url")
    sub.add_parser("auth_logout")

    p_code = sub.add_parser("auth_code")
    p_code.add_argument("code", type=str)

    sub.add_parser("accounts_list")
    p_switch = sub.add_parser("accounts_switch")
    p_switch.add_argument("email", type=str)
    p_remove = sub.add_parser("accounts_remove")
    p_remove.add_argument("email", type=str)

    p_wake = sub.add_parser("wake")
    p_wake.add_argument("--gpu", type=str, default="G4")
    p_wake.add_argument("--session", type=str, default="")
    p_wake.add_argument(
        "--app-py",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
    )

    sub.add_parser("keepalive")
    sub.add_parser("stop_all")

    args = parser.parse_args(argv)
    if args.cmd == "auth_status":
        cmd_auth_status()
    elif args.cmd == "auth_url":
        cmd_auth_url()
    elif args.cmd == "auth_logout":
        cmd_auth_logout()
    elif args.cmd == "auth_code":
        cmd_auth_code(args.code)
    elif args.cmd == "accounts_list":
        cmd_accounts_list()
    elif args.cmd == "accounts_switch":
        cmd_accounts_switch(args.email)
    elif args.cmd == "accounts_remove":
        cmd_accounts_remove(args.email)
    elif args.cmd == "wake":
        cmd_wake(args.gpu, args.app_py, args.session)
    elif args.cmd == "keepalive":
        cmd_keepalive()
    elif args.cmd == "stop_all":
        cmd_stop_all()


def run_cli(argv=None):
    """供独立 CLI 或打包后的 Windows exe (--bridge-cli) 子进程模式统一调用"""
    try:
        main(argv)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    run_cli()
