"""后台采集：账号余额与云端实例（30 s）、实例运行状态（5 s）、实例保活（80 s）、释放失败重试。"""
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import requests

from . import accounts, deployments
from .keys import redact
from .paths import BRIDGE

ACCOUNT_CACHE: Dict[str, dict] = {}
_pool = ThreadPoolExecutor(max_workers=6)
_wake_accounts = threading.Event()


def refresh_accounts_soon():
    _wake_accounts.set()


def _log_exc(where: str):
    print(f"[monitor] {where} 出错，继续下一轮：\n{redact(traceback.format_exc())}", file=sys.stderr, flush=True)


def _num(v, default: float = 0.0) -> float:
    """遥测字段可能是 None、字符串或 NaN，统一转成有限的 float"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f and abs(f) != float("inf") else default


def _refresh_one(email: str):
    entry = {"at": time.time(), "error": ""}
    try:
        entry["ccu"] = accounts.ccu_info(email)
        entry["assignments"] = accounts.assignments(email)
    except Exception as e:  # noqa: BLE001 单个账号失败不影响其他账号
        old = ACCOUNT_CACHE.get(email, {})
        entry.update({k: old.get(k) for k in ("ccu", "assignments") if old.get(k) is not None})
        entry["error"] = redact(str(e))[:160]
    ACCOUNT_CACHE[email] = entry


def _accounts_loop():
    while True:
        try:
            emails = accounts.emails()
            for gone in set(ACCOUNT_CACHE) - set(emails):
                ACCOUNT_CACHE.pop(gone, None)
            list(_pool.map(_safe(_refresh_one, "账号刷新"), emails))
        except Exception:  # noqa: BLE001
            _log_exc("账号刷新循环")
        _wake_accounts.wait(30)
        _wake_accounts.clear()


def _safe(fn, where: str):
    """包一层 try/except：线程池里单个任务出错不应连累整轮"""
    def run(*a):
        try:
            return fn(*a)
        except Exception:  # noqa: BLE001
            _log_exc(where)
    return run


def _poll(d: dict):
    try:
        r = requests.get(d["url"].rstrip("/") + "/minimax_api/status", timeout=8)
        r.raise_for_status()
        st = r.json()
        if not isinstance(st, dict):
            raise ValueError("状态不是 JSON 对象")
    except Exception:  # noqa: BLE001
        d["fails"] = d.get("fails", 0) + 1
        if d["fails"] == 4:
            deployments.event(d, "warn", "实例 20 秒无响应")
        return
    if d.get("fails", 0) >= 4:
        deployments.event(d, "ok", "实例恢复响应")
    d["fails"] = 0
    with deployments._lock:  # noqa: SLF001 与落盘快照互斥，避免保存时字段被改
        d["tele"] = st
        total = _num(st.get("vram_total_gb"))
        if total > 0:
            h = d.setdefault("hist", {"vram": [], "ram": []})
            h["vram"] = (h.get("vram", []) + [round(100 * _num(st.get("vram_used_gb")) / total)])[-60:]
            rt = _num(st.get("ram_total_gb"))
            if rt > 0:
                h["ram"] = (h.get("ram", []) + [round(100 * _num(st.get("ram_used_gb")) / rt)])[-60:]


def _reclaimed(d: dict) -> bool:
    """实例长时间无响应时，确认 Colab 是否已经回收"""
    ep = deployments.endpoint_of(d)
    cache = ACCOUNT_CACHE.get(d["email"], {})
    if not ep or cache.get("assignments") is None:
        return False
    return all(a.get("endpoint") != ep for a in cache["assignments"])


def _keepalive(email: str):
    home = accounts.home_of(email)
    try:
        subprocess.run([sys.executable, str(BRIDGE), "keepalive"], env=accounts.bridge_env(home),
                       capture_output=True, timeout=120)
    except Exception:  # noqa: BLE001
        pass


def _instances_loop():
    last_keep: Dict[str, float] = {}
    poll, retry = _safe(_poll, "实例轮询"), _safe(deployments.retry_release, "释放重试")
    while True:
        try:
            deps = [deployments.get(x["id"]) for x in deployments.all_deps()]
            deps = [d for d in deps if d]
            live = [d for d in deps if d["status"] in ("starting", "running") and d.get("url")]
            list(_pool.map(poll, live))
            for d in live:
                if d["status"] == "running" and d.get("fails", 0) >= 12 and _reclaimed(d):
                    d["status"], d["error"] = "stopped", "Colab 已回收该实例"
                    deployments.event(d, "warn", d["error"])
            for d in deps:  # 释放失败的实例：到点就重试（间隔和次数由 deployments 控制）
                if d["status"] == "release_failed" and time.time() >= d.get("_next", 0):
                    _pool.submit(retry, d)
            for email in {d["email"] for d in live}:
                if time.time() - last_keep.get(email, 0) > 80:
                    last_keep[email] = time.time()
                    _pool.submit(_keepalive, email)
        except Exception:  # noqa: BLE001
            _log_exc("实例循环")
        time.sleep(5)


def start():
    threading.Thread(target=_accounts_loop, daemon=True).start()
    threading.Thread(target=_instances_loop, daemon=True).start()
