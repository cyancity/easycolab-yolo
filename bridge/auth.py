"""账号命令：auth_url / auth_code / auth_logout / accounts_list / accounts_switch / accounts_remove。"""
import os
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from colab_cli.auth import PUBLIC_SCOPES, TOKEN_CONFIG_PATH, REMOTE_REDIRECT_URI

from .credentials import (
    ACCOUNT_PROFILE_CACHE_PATH,
    PKCE_VERIFIER_PATH,
    _clear_local_session_store,
    _get_account_email,
    _load_client_config,
)
from .sessions import cmd_auth_status
from .store import _load_accounts_store, _remove_account_from_store, _upsert_account


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
