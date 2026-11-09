"""排队等卡：分不到卡（503/容量不足）的部署进入 waiting，隔一段时间自动再申请一次。

enqueue 排队；monitor 每轮调用 tick，到点就让 deployments.retry_waiting 申请一次；申请不到由
deployments._run 调 rearm/report 回到 waiting，次数用完才算失败；cancel 取消排队。
全局任意两次排队申请之间至少隔 GAP 秒，Colab 的 503 冷却没结束也不申请，避免被判定为滥用。
d["wait"] = {tries 已申请次数, max 最多次数, next_at 下次申请的时间戳}
"""
import time

from . import deployments as dep
from . import failures

EVERY = 15 * 60  # 两次尝试的间隔（秒）
TRIES = 8
GAP = 120        # 全局任意两次排队尝试之间的最短间隔（秒）
FIRST_MIN = 60   # 排队后第一次申请至少等这么久
_last = [0.0]    # 全局最近一次排队尝试的时间


def _hhmm(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def enqueue(dep_id: str) -> dict:
    with dep._lock:  # noqa: SLF001
        d = dep.get(dep_id)
        if not d or d["status"] != "failed" or failures.kind_of(d) != "capacity":
            raise ValueError("只有分不到卡的部署可以排队")
        lk = d.get("_rel_lock")
        if lk is not None and lk.locked():
            raise ValueError("正在释放上一次的实例，稍后再排队")
        d["status"] = "waiting"
        d["wait"] = {"tries": 0, "max": TRIES,
                     "next_at": time.time() + max(dep.cooldown_left(d["email"], d["gpu"]), FIRST_MIN)}
        d["params"] = {**d.get("params", {}), "probe": True}  # 排队时每次只发 1 次分配请求
        d.pop("stopped_at", None)
    dep.event(d, "info", f"排队等卡：每 {EVERY // 60} 分钟申请一次，最多 {TRIES} 次")
    return d


def tick():
    """monitor 每轮调用：最多挑一个到点的排队部署申请一次"""
    now = time.time()
    if now - _last[0] < GAP:
        return
    due = []
    for x in dep.all_deps():
        # cooldown_left 已含全局冷却：任一账号刚被拒，所有账号一起等
        if x["status"] == "waiting" and x.get("wait") and x["wait"]["next_at"] <= now \
                and not dep.cooldown_left(x["email"], x["gpu"]):
            due.append(x)
    if not due:
        return
    d = dep.get(min(due, key=lambda x: x["wait"]["next_at"])["id"])
    if d and dep.retry_waiting(d):
        _last[0] = now


def cancel(dep_id: str):
    with dep._lock:  # noqa: SLF001
        d = dep.get(dep_id)
        if not d or d["status"] != "waiting":
            return
        d["status"], d["stopped_at"] = "stopped", time.time()
    dep.event(d, "info", "已取消排队")


def rearm(d: dict) -> bool:
    """部署失败时在 dep._lock 内调用：排队中分配又被拒且次数没用完，改回 waiting 并定下次时间"""
    w = d.get("wait")
    if not w or w["tries"] >= w["max"] or failures.kind_of(d) != "capacity":
        return False
    w["next_at"] = time.time() + max(EVERY, dep.cooldown_left(d["email"], d["gpu"]))
    return True


def report(d: dict, queued: bool):
    w = d["wait"]
    if queued:
        dep.event(d, "warn", f"仍分不出 {d['gpu']}，第 {w['tries']}/{w['max']} 次；{_hhmm(w['next_at'])} 再试")
        return
    dep.event(d, "err", d["error"])
    if failures.kind_of(d) == "capacity":
        dep.event(d, "err", f"排队 {w['max']} 次仍分不出，已停止")
