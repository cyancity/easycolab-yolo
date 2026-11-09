"""本地 OAuth 凭据：token 校验与刷新、当前账号邮箱、本地会话缓存清理、云端 rclone 配置。"""
import os
import json
from importlib import resources

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from colab_cli.auth import PUBLIC_SCOPES, TOKEN_CONFIG_PATH
from colab_cli.common import state


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
