"""多账号：每个账号一个独立的「家目录」，colab-cli 的 token、sessions 都落在里面。

colab_cli 在 import 时用 expanduser("~/.config/colab-cli/...") 固定路径，
所以桥接子进程靠改 USERPROFILE/HOME 隔离账号；本进程内的查询直接按路径读 token。
"""
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials

from colab_cli.auth import PUBLIC_SCOPES
from colab_cli.client import Client, Prod
from colab_cli.state import StateStore

from .keys import write_private
from .paths import ACCOUNTS_DIR, BRIDGE, LEGACY_ACCOUNTS

COLAB = "https://colab.research.google.com"
XSSI = ")]}'\n"
HEADERS = {"Accept": "application/json", "X-Colab-Client-Agent": "colab-cli"}
_locks: Dict[str, threading.Lock] = {}


def slug(email: str) -> str:
    """邮箱转目录名：只留 [a-z0-9._-]，去掉首尾的点和空格，结果永远是单个安全路径段（挡掉 .. 穿越）"""
    return re.sub(r"[^a-z0-9._-]", "_", email.strip().lower()).strip(". ") or "_"


def mask(email: str) -> str:
    """取 @ 前的前 3 后 3 位：同域名账号（都是 gmail）也能分得清"""
    local = email.split("@")[0]
    return f"{local[:3]}•••{local[-3:]}" if len(local) > 6 else local[:2] + "•••"


def is_known(email: str) -> bool:
    """email 是否是已登录的账号；接口入参先过这一关再拼路径"""
    return isinstance(email, str) and email in emails()


def home_of(email: str) -> Path:
    return ACCOUNTS_DIR / slug(email) / "home"


def cli_dir(home: Path) -> Path:
    return home / ".config" / "colab-cli"


def token_path(email: str) -> Path:
    return cli_dir(home_of(email)) / "token.json"


def sessions_store(email: str) -> StateStore:
    return StateStore(str(cli_dir(home_of(email)) / "sessions.json"))


def _copy_oauth(home: Path):
    """用户自备 OAuth 客户端配置放在真实家目录；桥接子进程用的是假家目录，要一并带过去"""
    src = Path.home() / ".colab-cli-oauth-config.json"
    try:
        if src.exists():
            home.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, home / src.name)
    except OSError:
        pass


def bridge_env(home: Path) -> dict:
    env = dict(os.environ)
    env["USERPROFILE"] = str(home)
    env["HOME"] = str(home)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_bridge(home: Path, args: List[str], timeout: int = 90) -> dict:
    """跑一次桥接命令，取最后一行 JSON"""
    p = subprocess.run(
        [sys.executable, str(BRIDGE), *args], env=bridge_env(home), capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    for line in reversed(p.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError((p.stderr or p.stdout or "bridge 无输出").strip()[-300:])


# ---------- 账号列表 ----------

def import_legacy() -> int:
    """把旧客户端 accounts.json 里的账号拆到各自家目录，已存在的不覆盖"""
    if not LEGACY_ACCOUNTS.exists():
        return 0
    try:
        data = json.loads(LEGACY_ACCOUNTS.read_text(encoding="utf-8"))
    except Exception:
        return 0
    n = 0
    for a in data.get("accounts", []):
        email, tok = str(a.get("email") or ""), a.get("token_json")
        if not email or not tok:
            continue
        tp = token_path(email)
        if tp.exists():
            continue
        tp.parent.mkdir(parents=True, exist_ok=True)
        _copy_oauth(home_of(email))
        write_private(tp, tok)
        _write_meta(email)
        n += 1
    return n


def _write_meta(email: str):
    meta = ACCOUNTS_DIR / slug(email) / "meta.json"
    if not meta.exists():
        meta.parent.mkdir(parents=True, exist_ok=True)
        write_private(meta, json.dumps({"email": email, "added_at": int(time.time())}))


def emails() -> List[str]:
    out = []
    for d in sorted(ACCOUNTS_DIR.iterdir()):
        meta = d / "meta.json"
        if d.is_dir() and meta.exists() and (d / "home" / ".config" / "colab-cli" / "token.json").exists():
            try:
                out.append(json.loads(meta.read_text(encoding="utf-8"))["email"])
            except Exception:  # noqa: BLE001 meta 损坏只跳过这个账号，不拖垮整个列表
                continue
    return out


def remove(email: str):
    shutil.rmtree(ACCOUNTS_DIR / slug(email), ignore_errors=True)


# ---------- 凭据与 Colab 接口 ----------

def credentials(email: str) -> Credentials:
    lock = _locks.setdefault(email, threading.Lock())
    with lock:
        tp = token_path(email)
        creds = Credentials.from_authorized_user_file(str(tp), PUBLIC_SCOPES)
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
            write_private(tp, creds.to_json())
        return creds


def session(email: str) -> AuthorizedSession:
    return AuthorizedSession(credentials(email))


def client(email: str) -> Client:
    return Client(Prod(), session(email))


def _get_json(email: str, path: str) -> dict:
    r = session(email).get(COLAB + path, params={"authuser": "0"}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    body = r.text[len(XSSI):] if r.text.startswith(XSSI) else r.text
    return json.loads(body or "{}")


def ccu_info(email: str) -> dict:
    """余额、消耗速率、活动实例数、可用/不可用显卡"""
    return _get_json(email, "/tun/m/ccu-info")


def assignments(email: str) -> List[dict]:
    return _get_json(email, "/tun/m/assignments").get("assignments", [])


def unassign(email: str, endpoint: str):
    client(email).unassign(endpoint)


def tier(eligible: List[str]) -> str:
    if "H100" in eligible:
        return "Ultra 档"
    if any(g in eligible for g in ("A100", "L4", "G4")):
        return "Pro 档"
    return "免费档"


# ---------- 登录 ----------

_pending: Dict[str, Path] = {}


def login_start() -> dict:
    pid = secrets.token_hex(6)
    home = ACCOUNTS_DIR / f"_pending_{pid}" / "home"
    home.mkdir(parents=True, exist_ok=True)
    _copy_oauth(home)
    res = run_bridge(home, ["auth_url"], timeout=30)
    _pending[pid] = home
    return {"pending_id": pid, "auth_url": res.get("auth_url", "")}


def login_finish(pending_id: str, code: str) -> dict:
    home = _pending.pop(pending_id, None)
    if not home:
        raise ValueError("登录会话已过期，请重新点登录")
    res = run_bridge(home, ["auth_code", code.strip()], timeout=60)
    email = str(res.get("email") or "")
    if not res.get("logged_in") or not email:
        shutil.rmtree(home.parent, ignore_errors=True)
        raise ValueError(res.get("error") or "授权失败，请确认授权码完整")
    tok = (cli_dir(home) / "token.json").read_text(encoding="utf-8")
    tp = token_path(email)
    tp.parent.mkdir(parents=True, exist_ok=True)
    _copy_oauth(home_of(email))
    write_private(tp, tok)
    _write_meta(email)
    shutil.rmtree(home.parent, ignore_errors=True)
    return {"email": email}
