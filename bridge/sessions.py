"""云端会话相关命令：auth_status / keepalive / stop_all，以及会话同步与 CU 用量查询。"""
import json
import time

import requests

from colab_cli.common import state
from colab_cli.client import Shape, Variant
from colab_cli.state import SessionState
from colab_cli.runtime import ColabRuntime
from colab_cli.auth import TOKEN_CONFIG_PATH

from .credentials import _get_account_email, _has_valid_token
from .store import _load_accounts_store, _upsert_account


def _safe_sync_sessions():
    for attempt in range(3):
        try:
            return state.sync_sessions()
        except Exception:
            if attempt == 2:
                # store.list() 返回 {name: SessionState}，直接复制即可
                return dict(state.store.list()), []
            time.sleep(1.2)


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
