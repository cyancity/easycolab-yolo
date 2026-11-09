"""多账号凭据库：保存每个登录过的账号 OAuth 凭据，实现「切号不丢号」。"""
import os
import json
import tempfile
from datetime import datetime, timezone

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
