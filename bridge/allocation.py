"""GPU 型号解析、资格预检、分配实例与容器就绪探测。"""
import json
import os
import time
import uuid

import requests

from colab_cli.common import state
from colab_cli.client import Accelerator, Variant, Shape, resolve_assign_shape, PostAssignmentResponse, ColabRequestError, TooManyAssignmentsError
from colab_cli.state import SessionState

from .sessions import _fetch_ccu_info_raw


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


def _record_assign_error(accel: Accelerator, shape, err: Exception):
    """Colab 拒绝分配时，把原始响应追加到本账号的 assign-diag.jsonl，供工作台排查；不改命令行输出"""
    try:
        resp = getattr(err, "response", None)
        body = getattr(err, "response_body", None)
        if body is None and resp is not None:
            body = getattr(resp, "text", "")
        headers = {k: v for k, v in (getattr(resp, "headers", None) or {}).items()
                   if k.lower() in ("content-type", "retry-after") or (k.lower().startswith("x-")
                   and not any(w in k.lower() for w in ("token", "cookie", "auth")))}
        rec = {"at": int(time.time()), "accelerator": accel.value, "shape": getattr(shape, "name", None),
               "status": getattr(resp, "status_code", None), "headers": headers, "body": str(body or "")[:2000]}
        path = os.path.join(os.path.expanduser("~"), ".config", "colab-cli", "assign-diag.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _allocate_fresh_session(session_name: str, gpu_type: str) -> SessionState:
    accel = _resolve_gpu_accel(gpu_type)
    _check_gpu_eligible(accel)
    # 默认不带 shape=hm，与官方 CLI 一致：2026-11-10 实测带 hm 的 T4 在免费档和 Ultra 账号都返回 503，
    # 不带 hm 连续两次分配成功。需要大内存的部署由工作台设 COLAB_FLEET_HIGH_MEM=1
    high_mem = os.environ.get("COLAB_FLEET_HIGH_MEM", "0") == "1"
    shape = None if accel in SINGLE_SHAPE_GPUS or not high_mem else resolve_assign_shape(accel, high_mem=True)
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
        _record_assign_error(accel, shape, req_err)
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
