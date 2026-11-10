"""桥接子进程与云端实例的小工具：结束进程树、确认云端实例是否还在、查部署对应的实例地址。"""
import os
import signal
import subprocess
import sys
import traceback

from . import accounts
from .paths import NO_WINDOW


def log(msg: str):
    print(f"[deployments] {msg}", file=sys.stderr, flush=True)


def kill_proc(p):
    """结束桥接子进程及其子进程树，防止取消后仍在申请 GPU"""
    if p is None or p.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, timeout=15,
                           creationflags=NO_WINDOW)
        else:
            os.killpg(p.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        log("结束子进程树失败：\n" + traceback.format_exc())
    try:
        p.kill()
    except Exception:  # noqa: BLE001 进程已退出
        pass


def gone(d: dict, endpoint: str) -> bool:
    """unassign 报错时再确认一次：云端已没有该实例就等同释放成功"""
    try:
        return all(a.get("endpoint") != endpoint for a in accounts.assignments(d["email"]))
    except Exception:  # noqa: BLE001
        return False


def endpoint_of(d: dict) -> str:
    try:
        s = accounts.sessions_store(d["email"]).get(d["session"])
    except Exception:  # noqa: BLE001
        return ""
    return s.endpoint if s else ""
