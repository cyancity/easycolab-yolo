"""部署编排：唤醒 → 推送密钥 → 下载权重 → 载入显存 → 自测 → 就绪；停止与自动释放。"""
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional

import requests

from . import accounts, catalog, failures, selftest, waitlist
from .cloud import endpoint_of, gone as _gone, kill_proc as _kill_proc, log as _log  # noqa: F401 endpoint_of 对外保留
from .keys import gateway_key, redact, write_private
from .paths import APP_PY, BRIDGE, DEPLOYMENTS_FILE

STEPS = ["分配实例", "部署服务", "下载权重", "载入显存", "自测"]
# release_failed：实例可能还活着，监控线程会按 RELEASE_EVERY 间隔重试释放，所以也算活动状态
# waiting：排队等卡，没有实例，但部署还要继续
ACTIVE = ("starting", "running", "stopping", "release_failed", "waiting")
_EMOJI = re.compile("[\U0001F000-\U0001FFFF☀-➿️‍]")
# 分配被拒（503/412/超时/分配失败）都要冷却，避免连续重试被判定为滥用
_REFUSED = re.compile(r"\b(503|412)\b|timed?[ -]?out|超时|allocation|分配", re.I)
_lock = threading.RLock()  # 保护 _deps 和冷却表；持有它时不要调用 _save/event，避免和 _write_lock 死锁
_write_lock = threading.Lock()
_deps: Dict[str, dict] = {}
_cooldown: Dict[tuple, float] = {}  # (email, gpu) -> 可再次申请的时间戳
COOLDOWN_S = 600
RELEASE_EVERY = 60  # 释放失败后的重试间隔（秒）
RELEASE_TRIES = 10
# 部署参数：probe=true 只发 1 次分配请求（503 不重试）；high_mem=true 给 T4/A100 申请大内存规格
PARAM_ENV = {"probe": {"COLAB_CLI_ASSIGN_RETRIES": "0"}, "high_mem": {"COLAB_FLEET_HIGH_MEM": "1"}}
# Colab 限流可能按出口 IP 计：任一账号被拒后，所有账号一起暂停分配，避免被判定为滥用
_global = {"until": 0.0, "why": ""}
# 不落盘的运行时字段；以下划线开头的键（子进程句柄、重试计数等）同样不落盘
_TRANSIENT = {"pid", "fails"}
_KEEP_RAW = {"url", "comfy_url"}  # 服务地址本身含访问密钥，落盘必须保持可用


def _now() -> str:
    return time.strftime("%H:%M")


# ---------- 持久化 ----------

def _plain(v, key: str = ""):
    """递归复制可序列化字段，字符串过 redact；字典被别的线程改动时抛 RuntimeError，由调用方重试"""
    if isinstance(v, dict):
        return {k: _plain(x, k) for k, x in list(v.items()) if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(v, (list, tuple)):
        return [_plain(x, key) for x in list(v)]
    if isinstance(v, str):
        return v if key in _KEEP_RAW else redact(v)
    return v if isinstance(v, (int, float, bool)) or v is None else str(v)


def _snapshot() -> dict:
    last = None
    for _ in range(20):
        try:
            with _lock:
                now = time.time()
                deps = [{k: _plain(x, k) for k, x in list(d.items()) if not k.startswith("_") and k not in _TRANSIENT}
                        for d in list(_deps.values())]
                cool = [[e, g, t] for (e, g), t in list(_cooldown.items()) if t > now]
                return {"deps": deps, "cooldown": cool, "global": dict(_global)}
        except RuntimeError as e:  # 其他线程正在改字典，稍后重试
            last = e
            time.sleep(0.005)
    raise RuntimeError(f"快照多次失败：{last}")


def _save():
    """原子写盘；任何错误只记日志，不影响部署流程"""
    try:
        with _write_lock:  # 快照在写盘锁内取，保证后写的永远是更新的状态
            write_private(DEPLOYMENTS_FILE, json.dumps(_snapshot(), ensure_ascii=False))
    except Exception:  # noqa: BLE001
        _log("保存部署状态失败：" + traceback.format_exc())


def load():
    if not DEPLOYMENTS_FILE.exists():
        return
    try:
        raw = json.loads(DEPLOYMENTS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 文件损坏：留一份备份再继续，避免下次保存直接覆盖
        _log("部署状态文件损坏，已备份为 .bad\n" + traceback.format_exc())
        try:
            os.replace(DEPLOYMENTS_FILE, DEPLOYMENTS_FILE.with_name(DEPLOYMENTS_FILE.name + ".bad"))
        except OSError:
            pass
        return
    items = raw if isinstance(raw, list) else (raw.get("deps") or [])
    if isinstance(raw, dict):
        for e, g, t in raw.get("cooldown") or []:
            _cooldown[(e, g)] = float(t)
        gl = raw.get("global") or {}
        _global.update(until=float(gl.get("until") or 0), why=str(gl.get("why") or ""))
    pending = []
    for d in items:
        if not isinstance(d, dict) or not d.get("id") or not d.get("status"):
            continue
        for k, v in (("events", []), ("tele", {}), ("hist", {"vram": [], "ram": []}), ("bench", {}), ("params", {})):
            d.setdefault(k, v)
        if d["status"] == "starting":  # 重启时子进程已丢，实例可能已分配：必须释放
            d["status"], d["error"] = "failed", "工作台重启，部署中断"
            pending.append((d, "工作台重启，已释放未完成的部署实例"))
        elif d["status"] == "stopping":
            pending.append((d, "已释放实例"))
        elif d["status"] == "release_failed":
            d["_tries"], d["_next"] = 0, time.time() + 5
        _deps[d["id"]] = d
    for d, why in pending:
        threading.Thread(target=_release, args=(d, why), daemon=True).start()


def event(d: dict, kind: str, msg: str):
    msg = redact(_EMOJI.sub("", str(msg or ""))).strip()
    if not msg:
        return
    with _lock:
        d["events"] = (list(d.get("events") or []) + [[_now(), kind, msg[:300]]])[-60:]
    _save()


def _step(d: dict, n: int, text: str = ""):
    d["step"], d["step_text"] = n, redact(text or STEPS[n - 1])
    _save()


def all_deps() -> List[dict]:
    with _lock:
        return [{k: v for k, v in list(d.items()) if not k.startswith("_")} for d in _deps.values()]


def get(dep_id: str) -> Optional[dict]:
    return _deps.get(dep_id)


def cooldown_left(email: str, gpu: str) -> int:
    return max(0, int(max(_cooldown.get((email, gpu), 0), _global["until"]) - time.time()))


def global_cooldown() -> dict:
    left = max(0, int(_global["until"] - time.time()))
    return {"left": left, "why": _global["why"]} if left else {}


def _cool(d: dict, why: str):
    until = time.time() + COOLDOWN_S
    with _lock:
        _cooldown[(d["email"], d["gpu"])] = until
        _global.update(until=until, why=redact(f"{accounts.mask(d['email'])} 申请 {d['gpu']} 被拒：{why}")[:120])
    _save()


# ---------- 实例 HTTP ----------

def _api(d: dict, path: str, body: Optional[dict] = None, timeout: int = 20) -> dict:
    url = d["url"].rstrip("/") + path
    r = requests.post(url, json=body or {}, timeout=timeout) if body is not None else requests.get(url, timeout=timeout)
    r.raise_for_status()  # 4xx/5xx 一律按失败处理
    return r.json()


def _check_alive(d: dict):
    if d["status"] != "starting":
        raise RuntimeError("部署已取消")


def _wait(d: dict, check, timeout_s: int, every: float = 3.0):
    """轮询实例状态直到 check(status) 返回 True；连续 10 次请求失败说明实例大概率已死，直接放弃"""
    end = time.time() + timeout_s
    fails = 0
    while time.time() < end:
        _check_alive(d)
        try:
            st = _api(d, "/minimax_api/status")
            fails = 0
            d["tele"] = st
            if st.get("task_message"):
                d["step_text"] = redact(_EMOJI.sub("", str(st["task_message"])))[:120]
            if check(st):
                return st
        except (requests.RequestException, ValueError):
            fails += 1
            if fails >= 10:
                raise RuntimeError("实例连续 10 次无响应，可能已被 Colab 回收")
        time.sleep(every)
    raise TimeoutError(f"{STEPS[d['step'] - 1]}超时")


# ---------- 部署流程 ----------

def start(email: str, model_id: str, gpu: str, params: Optional[dict] = None) -> dict:
    m = catalog.BY_ID.get(model_id)
    if not m or not m.get("ready"):
        raise ValueError("该模型暂不能部署")
    left = cooldown_left(email, gpu)
    if left:
        raise ValueError(f"Colab 刚拒绝分配，{left // 60 + 1} 分钟后再试，避免被判定为滥用")
    dep_id = "d" + secrets.token_hex(3)
    d = {
        "id": dep_id, "email": email, "model": model_id, "gpu": gpu, "params": params or {},
        "session": "wb-" + dep_id, "status": "starting", "step": 1, "step_text": STEPS[0],
        "url": "", "comfy_url": "", "created_at": time.time(), "ready_at": None, "error": "",
        "events": [], "tele": {}, "hist": {"vram": [], "ram": []}, "bench": {},
    }
    with _lock:
        _deps[dep_id] = d
    event(d, "ok", f"开始部署 {m['name']} · {gpu} · {accounts.mask(email)}")
    threading.Thread(target=_run, args=(d,), daemon=True).start()
    return d


def retry_waiting(d: dict) -> bool:
    """waitlist.tick 调用：排队的部署到点再申请一次，申请到了就接着往下部署"""
    with _lock:
        w = d.get("wait")
        if d["status"] != "waiting" or not w:
            return False
        w["tries"] += 1
        d.update(status="starting", step=1, step_text=STEPS[0], error="", url="")
        d.pop("fail_log", None)
    event(d, "info", f"排队第 {w['tries']}/{w['max']} 次：申请 {d['gpu']}")
    threading.Thread(target=_run, args=(d,), daemon=True).start()
    return True


def _run(d: dict):
    m = catalog.BY_ID[d["model"]]
    try:
        _wake(d)
        _push_key(d)
        if m.get("app_id"):
            _download(d, m)
            _load(d, m)
        elif d["model"] == "comfy":
            _comfy(d)
        if m["kind"] == "语言":
            _step(d, 5)
            bench(d)  # 自测失败只记警告，不影响就绪
        with _lock:  # 期间被取消（stopping/stopped）就绝不能写回 running
            ok = d["status"] == "starting"
            if ok:
                d["status"], d["ready_at"], d["step_text"] = "running", time.time(), ""
        if ok:
            event(d, "ok", "就绪")
    except Exception as e:  # noqa: BLE001 失败要落到界面上，不再往外抛
        with _lock:
            was_starting = d["status"] == "starting"
            if was_starting:
                d["error"] = redact(str(e))[:300]
                queued = waitlist.rearm(d)  # 排队中又被拒且次数没用完：回到 waiting，不算失败
                d["status"] = "waiting" if queued else "failed"
        if was_starting:
            if d.get("wait"):
                waitlist.report(d, queued)
            else:
                event(d, "err", d["error"])
            _keep_evidence(d)
            _release(d, reason="本次申请失败，已释放实例" if queued else "部署失败，已自动释放实例，避免空耗 CU")
    finally:
        _kill_proc(d.get("_proc"))
        if d["status"] in ("stopping", "stopped"):
            # 取消时桥接子进程可能在 stop 之后才写入 session，这里补一次释放
            _release(d, reason="已释放实例（取消后补释放）")
        _save()


def _keep_evidence(d: dict):
    """失败后释放前留一份实例日志（释放后实例就没了）；取不到不影响释放，随后的 _release 会落盘"""
    try:
        lines = failures.capture_log(d)
    except Exception:  # noqa: BLE001
        return _log("取失败前日志出错：" + traceback.format_exc())
    with _lock:
        d["fail_log"] = lines


def _wake(d: dict):
    home = accounts.home_of(d["email"])
    env = accounts.bridge_env(home)
    for k, extra in PARAM_ENV.items():
        if d["params"].get(k) is True:
            env.update(extra)
    cmd = [sys.executable, str(BRIDGE), "wake", "--gpu", d["gpu"], "--session", d["session"], "--app-py", str(APP_PY)]
    extra = {} if sys.platform == "win32" else {"start_new_session": True}
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", **extra)
    with _lock:  # 与 stop() 互相可见：要么 stop 看到句柄，要么这里看到已取消
        d["_proc"], d["pid"] = p, p.pid
        cancelled = d["status"] != "starting"
    if cancelled:
        _kill_proc(p)
    err = ""
    for raw in p.stdout:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("WAKE_URL:"):
            d["url"] = line[len("WAKE_URL:"):].strip()
        elif line.startswith("WAKE_ERROR:"):
            err = line[len("WAKE_ERROR:"):].strip()
        else:
            msg = line[len("WAKE_LOG:"):] if line.startswith("WAKE_LOG:") else line
            if "上传" in msg or "Google Drive" in msg:
                _step(d, 2)
            event(d, "info", msg)
    p.wait()
    _check_alive(d)
    if err and _REFUSED.search(err):
        _cool(d, redact(err)[:60])
    if err or not d["url"]:
        raise RuntimeError(err or "唤醒结束但没有拿到服务地址")
    event(d, "ok", "服务已启动")


def _push_key(d: dict):
    for _ in range(10):
        _check_alive(d)
        try:
            _api(d, "/minimax_api/config_api_key", {"api_key": gateway_key()})
            return
        except requests.RequestException:
            time.sleep(3)
    raise RuntimeError("服务没有响应，无法写入 API Key")


def _download(d: dict, m: dict):
    _step(d, 3)
    st = _api(d, "/minimax_api/download_model", {"model_id": m["app_id"]}, timeout=60)
    if st.get("all_downloaded"):
        event(d, "ok", "权重已在缓存中，跳过下载")
        return
    _wait(d, lambda s: s.get("is_downloading") or s.get("all_downloaded"), 60)
    st = _wait(d, lambda s: not s.get("is_downloading"), 4 * 3600, every=5)
    if not st.get("all_downloaded"):
        raise RuntimeError("下载未完成：" + str(st.get("task_message") or ""))
    event(d, "ok", "权重下载完成")


def _load(d: dict, m: dict):
    _step(d, 4)
    body = {"model_id": m["app_id"], **m.get("load", {}), **d["params"].get("load", {})}
    _api(d, "/minimax_api/load_model", body, timeout=60)
    # 显存预检这类秒级失败来不及置 is_loading_vram，只留下「加载 … 失败: …」，不认它会空等 60 秒再报超时
    st = _wait(d, lambda s: s.get("is_loading_vram") or m["app_id"] in (s.get("loaded_model_ids") or [])
               or "失败" in str(s.get("task_message") or ""), 60)
    if not st.get("is_loading_vram") and m["app_id"] not in (st.get("loaded_model_ids") or []):
        raise RuntimeError(str(st.get("task_message") or "载入失败"))
    st = _wait(d, lambda s: not s.get("is_loading_vram"), 3600, every=4)
    if m["app_id"] not in (st.get("loaded_model_ids") or []):
        raise RuntimeError("载入失败：" + str(st.get("task_message") or ""))
    event(d, "ok", f"已载入显存 {st.get('vram_used_gb', 0)} / {st.get('vram_total_gb', 0)} GB")


def _comfy(d: dict):
    _step(d, 4, "启动 ComfyUI")
    _api(d, "/minimax_api/comfyui/start", {}, timeout=60)
    end, fails = time.time() + 1800, 0
    while time.time() < end:
        _check_alive(d)
        try:
            c = _api(d, "/minimax_api/comfyui/status").get("comfyui", {})
            fails = 0
        except (requests.RequestException, ValueError):
            fails += 1
            if fails >= 10:
                raise RuntimeError("实例连续 10 次无响应，可能已被 Colab 回收")
            time.sleep(5)
            continue
        d["step_text"] = redact(_EMOJI.sub("", str(c.get("status_message") or "启动 ComfyUI")))[:120]
        if c.get("is_running") and c.get("public_url"):
            d["comfy_url"] = c["public_url"]
            return
        time.sleep(5)
    raise TimeoutError("ComfyUI 启动超时")


def bench(d: dict):
    """真实请求一次，测单流解码速度和首字延迟；失败只记警告，绝不让已载入的实例失败或被释放"""
    try:
        res = selftest.measure(d)
    except Exception as e:  # noqa: BLE001
        event(d, "warn", f"自测失败：{e}")
        return
    if isinstance(res, str):
        event(d, "warn", res)
        return
    d["bench"] = res
    ok = res["tps"] >= catalog.SLO["tpsMin"]
    srv = f"（服务端 {res['srv_tps']} tok/s）" if res.get("srv_tps") else ""
    event(d, "ok" if ok else "warn", f"自测：单流 {res['tps']} tok/s{srv}，首字 {res['ttft']} s")


# ---------- 停止与释放 ----------

def stop(dep_id: str):
    d = _deps.get(dep_id)
    if d and d["status"] == "waiting":  # 排队中没有实例，只取消排队
        return waitlist.cancel(dep_id)
    with _lock:
        d = _deps.get(dep_id)
        if not d or d["status"] in ("stopped", "stopping"):
            return
        d["status"] = "stopping"
        d["_tries"] = 0
    _kill_proc(d.get("_proc"))  # 唤醒还在进行时先掐掉子进程，否则它会继续申请 GPU
    _save()
    threading.Thread(target=_stop_run, args=(d,), daemon=True).start()


def _stop_run(d: dict):
    if d.get("url") and d["model"] == "comfy":
        try:
            _api(d, "/minimax_api/comfyui/sync_drive", {}, timeout=30)
            event(d, "ok", "ComfyUI 已同步到 Drive")
        except Exception:  # noqa: BLE001
            pass
    _release(d, reason="已释放实例")


def _release(d: dict, reason: str):
    """释放实例。unassign 失败时保留 session 和部署记录，转入 release_failed 由监控线程重试"""
    with _lock:
        lk = d.setdefault("_rel_lock", threading.Lock())
    with lk:
        ok = True
        try:
            s = accounts.sessions_store(d["email"]).get(d["session"])  # 每次重新读盘，取消后才写入的也能看到
            if s:
                try:
                    accounts.unassign(d["email"], s.endpoint)
                except Exception:
                    if not _gone(d, s.endpoint):
                        raise
                accounts.sessions_store(d["email"]).remove(d["session"])  # 成功后才从 store 移除
                event(d, "ok", reason)
        except Exception as e:  # noqa: BLE001
            ok = False
            with _lock:
                d["_tries"] = n = d.get("_tries", 0) + 1
                d["_next"] = time.time() + RELEASE_EVERY if n < RELEASE_TRIES else float("inf")
                if d["status"] != "release_failed":
                    d["after"] = d["status"] if d["status"] in ("failed", "waiting") else "stopped"
                d["status"] = "release_failed"
            tail = f"{RELEASE_EVERY} 秒后重试" if n < RELEASE_TRIES else "已停止自动重试，请在账号页手动释放"
            event(d, "err", f"释放失败（第 {n}/{RELEASE_TRIES} 次），{tail}：{e}")
        if ok:
            with _lock:
                if d["status"] == "release_failed":
                    d["status"] = d.get("after") or "stopped"
                elif d["status"] == "stopping":
                    d["status"] = "stopped"
                d["_tries"] = 0
                if d["status"] != "waiting":  # 排队中的部署还没结束
                    d.setdefault("stopped_at", time.time())
    _save()


def retry_release(d: dict):
    """监控线程调用：release_failed 到点后重试释放（每 60 s 一次，最多 10 次）"""
    with _lock:
        if d["status"] != "release_failed" or time.time() < d.get("_next", 0) or d.get("_tries", 0) >= RELEASE_TRIES:
            return
        d["_next"] = time.time() + RELEASE_EVERY  # 先占位，避免上一次还没返回就被再次提交
    _release(d, reason="已释放实例（重试成功）")
