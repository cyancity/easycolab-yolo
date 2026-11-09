import os
import time
import threading
from typing import Optional, Tuple, Dict, Any
import shutil
import torch
from . import state
from .bootstrap import get_live_console_Lines
from .common import BOOT_TIME, LOADED_RUNTIMES
from .config import (
    MODEL_DOWNLOAD_STATE,
    _get_system_ram_gb,
    DRIVE_MOUNTED,
    _has_rclone_drive_remote,
    MODEL_CATALOG,
)
from .idle import estimate_cu_burn_rate
from .model_files import check_model_files_status


_TELEMETRY_LOCK = threading.Lock()
_CACHED_DRIVE_METRICS: Dict[str, Any] = {
    "ts": 0.0,
    "drive_total_gb": 0.0,
    "drive_used_gb": 0.0,
    "drive_free_gb": 0.0,
}
_CACHED_FILE_STATUS_MAP: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHED_LIVE_TELEMETRY: Dict[str, Any] = {}


def _get_cached_model_files_status(target_id: str) -> Dict[str, Any]:
    """对磁盘/云盘 FUSE 权重扫描做轻量缓存，避免高频状态轮询在视频生成期间阻塞 I/O"""
    now = time.time()
    ttl = 1.8 if MODEL_DOWNLOAD_STATE["is_downloading"] else 12.0
    with _TELEMETRY_LOCK:
        cached = _CACHED_FILE_STATUS_MAP.get(target_id)
        if cached and (now - cached[0]) < ttl:
            return dict(cached[1])
    fresh = check_model_files_status(target_id)
    with _TELEMETRY_LOCK:
        _CACHED_FILE_STATUS_MAP[target_id] = (now, dict(fresh))
    return fresh


def _get_storage_metrics_dict(refresh_drive: bool = False) -> Dict[str, Any]:
    """采集 Colab 实例系统内存 (RAM)、本地磁盘、/dev/shm 内存加速盘及 Google Drive 云盘指标 (单位 GB)"""
    ram_total_gb, ram_free_gb = _get_system_ram_gb()
    ram_used_gb = round(max(0.0, ram_total_gb - ram_free_gb), 1) if ram_total_gb > 0 else 0.0

    disk_path = "/content" if os.path.exists("/content") else os.getcwd()
    disk_total_gb, disk_used_gb, disk_free_gb = 0.0, 0.0, 0.0
    try:
        du = shutil.disk_usage(disk_path)
        disk_total_gb = round(du.total / (1024 ** 3), 1)
        disk_free_gb = round(du.free / (1024 ** 3), 1)
        disk_used_gb = round(max(0.0, (du.total - du.free) / (1024 ** 3)), 1)
    except Exception:
        pass

    shm_total_gb, shm_used_gb, shm_free_gb = 0.0, 0.0, 0.0
    if os.path.exists("/dev/shm"):
        try:
            su = shutil.disk_usage("/dev/shm")
            shm_total_gb = round(su.total / (1024 ** 3), 1)
            shm_free_gb = round(su.free / (1024 ** 3), 1)
            shm_used_gb = round(max(0.0, (su.total - su.free) / (1024 ** 3)), 1)
        except Exception:
            pass

    now = time.time()
    if DRIVE_MOUNTED and (refresh_drive or (now - _CACHED_DRIVE_METRICS["ts"]) >= 15.0):
        if os.path.exists("/content/drive/MyDrive"):
            try:
                gu = shutil.disk_usage("/content/drive/MyDrive")
                _CACHED_DRIVE_METRICS["drive_total_gb"] = round(gu.total / (1024 ** 3), 1)
                _CACHED_DRIVE_METRICS["drive_free_gb"] = round(gu.free / (1024 ** 3), 1)
                _CACHED_DRIVE_METRICS["drive_used_gb"] = round(max(0.0, (gu.total - gu.free) / (1024 ** 3)), 1)
                _CACHED_DRIVE_METRICS["ts"] = now
            except Exception:
                pass

    return {
        "ram_total_gb": ram_total_gb,
        "ram_used_gb": ram_used_gb,
        "ram_free_gb": ram_free_gb,
        "disk_total_gb": disk_total_gb,
        "disk_used_gb": disk_used_gb,
        "disk_free_gb": disk_free_gb,
        "shm_total_gb": shm_total_gb,
        "shm_used_gb": shm_used_gb,
        "shm_free_gb": shm_free_gb,
        "drive_total_gb": _CACHED_DRIVE_METRICS["drive_total_gb"],
        "drive_used_gb": _CACHED_DRIVE_METRICS["drive_used_gb"],
        "drive_free_gb": _CACHED_DRIVE_METRICS["drive_free_gb"],
    }


def _sample_live_hardware_telemetry_once() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        gpu_name = "CPU Mode (未检测到 GPU)"
        cuda_avail = False
        free_gb, total_gb, used_gb = 0.0, 0.0, 0.0
        cu_rate = 0.10
    else:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_avail = True
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_gb = round(free_mem / (1024 ** 3), 2)
        total_gb = round(total_mem / (1024 ** 3), 2)
        used_gb = round(total_gb - free_gb, 2)
        cu_rate = estimate_cu_burn_rate(gpu_name)

    snap = {
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_free_gb": free_gb,
        "vram_total_gb": total_gb,
        "vram_used_gb": used_gb,
        "cu_burn_rate_per_hour": cu_rate,
    }
    snap.update(_get_storage_metrics_dict(refresh_drive=False))
    if "_poll_local_comfyui_runtime_quiet" in globals():
        _poll_local_comfyui_runtime_quiet()
    with _TELEMETRY_LOCK:
        _CACHED_LIVE_TELEMETRY.clear()
        _CACHED_LIVE_TELEMETRY.update(snap)
    return snap


def _telemetry_sampler_loop():
    """独立后台线程以 2s 周期采样 GPU 显存/内存/磁盘指标，保证 HTTP 轮询非阻塞秒回"""
    while True:
        try:
            _sample_live_hardware_telemetry_once()
        except Exception:
            pass
        time.sleep(2.0)


threading.Thread(target=_telemetry_sampler_loop, daemon=True).start()


def get_live_telemetry_snapshot() -> Dict[str, Any]:
    with _TELEMETRY_LOCK:
        if _CACHED_LIVE_TELEMETRY:
            return dict(_CACHED_LIVE_TELEMETRY)
    return _sample_live_hardware_telemetry_once()


def get_gpu_status_dict(model_id: Optional[str] = None) -> Dict[str, Any]:
    """返回结构化的 GPU 状态、实例磁盘空间、多模型注册信息、CU 算力消耗账本、闲置倒计时与实时终端进度流"""
    if model_id and model_id in MODEL_CATALOG:
        state.SELECTED_MODEL_ID = model_id
    target_id = state.SELECTED_MODEL_ID

    file_status = _get_cached_model_files_status(target_id)
    hw = get_live_telemetry_snapshot()
    uptime_seconds = int(time.time() - BOOT_TIME)
    idle_seconds = int(max(0.0, time.time() - state.LAST_ACTIVE_TIME))
    idle_remaining_seconds = (
        max(0, state.IDLE_TIMEOUT_MINUTES * 60 - idle_seconds) if state.IDLE_TIMEOUT_MINUTES > 0 else -1
    )

    gpu_name = hw["gpu_name"]
    cuda_avail = hw["cuda_available"]
    free_gb = hw["vram_free_gb"]
    total_gb = hw["vram_total_gb"]
    used_gb = hw["vram_used_gb"]
    cu_rate = hw["cu_burn_rate_per_hour"]

    session_cu_used = round((uptime_seconds / 3600.0) * cu_rate, 3)
    loaded_ids = list(LOADED_RUNTIMES.keys())
    loaded_details = [
        {
            "id": mid,
            "name": rt.get("name", mid),
            "mode": rt.get("mode_label", ""),
            "vram_gb": rt.get("vram_gb", 0.0),
            "context_length": int((rt.get("llm_bundle") or {}).get("context_length") or 0),
        }
        for mid, rt in LOADED_RUNTIMES.items()
    ]
    loaded_ctx = 0
    if target_id in LOADED_RUNTIMES and LOADED_RUNTIMES[target_id].get("llm_bundle"):
        loaded_ctx = int(LOADED_RUNTIMES[target_id]["llm_bundle"].get("context_length") or 0)
    else:
        for rt in LOADED_RUNTIMES.values():
            if rt.get("llm_bundle") and rt["llm_bundle"].get("context_length"):
                loaded_ctx = int(rt["llm_bundle"]["context_length"])
                break

    any_loaded = bool(LOADED_RUNTIMES) or (state.PIPELINE is not None) or (state.LLM_BUNDLE is not None)
    loaded_for_selected = (target_id in LOADED_RUNTIMES) or (any_loaded and state.LOADED_MODEL_ID == target_id)
    summary_mode = (
        LOADED_RUNTIMES[target_id]["mode_label"]
        if target_id in LOADED_RUNTIMES
        else (" + ".join(rt["name"] for rt in LOADED_RUNTIMES.values()) if LOADED_RUNTIMES else (state.CURRENT_MODEL_MODE or "未加载"))
    )

    base = {
        "ok": True,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_free_gb": free_gb,
        "vram_total_gb": total_gb,
        "vram_used_gb": used_gb,
        "model_loaded": loaded_for_selected,
        "any_model_loaded": any_loaded,
        "loaded_model_id": target_id if target_id in LOADED_RUNTIMES else state.LOADED_MODEL_ID,
        "loaded_model_ids": loaded_ids,
        "loaded_models_detail": loaded_details,
        "current_model": summary_mode,
        "context_length": state.QWEN_CONTEXT_LENGTH,
        "loaded_context_length": loaded_ctx,
        "is_downloading": MODEL_DOWNLOAD_STATE["is_downloading"],
        "is_loading_vram": MODEL_DOWNLOAD_STATE["is_loading_vram"],
        "task_progress": MODEL_DOWNLOAD_STATE["progress"],
        "task_message": MODEL_DOWNLOAD_STATE["message"],
        "uptime_seconds": uptime_seconds,
        "cu_burn_rate_per_hour": cu_rate,
        "session_cu_used": session_cu_used,
        "idle_timeout_minutes": state.IDLE_TIMEOUT_MINUTES,
        "idle_remaining_seconds": idle_remaining_seconds,
        "api_key_enabled": bool(state.API_AUTH_KEY),
        "api_request_count": state.API_REQUEST_COUNT,
        "console_lines": get_live_console_Lines(),
    }
    for k in (
        "ram_total_gb", "ram_used_gb", "ram_free_gb",
        "disk_total_gb", "disk_used_gb", "disk_free_gb",
        "shm_total_gb", "shm_used_gb", "shm_free_gb",
        "drive_total_gb", "drive_used_gb", "drive_free_gb",
    ):
        base[k] = hw.get(k, 0.0)
    base.update(file_status)
    if _has_rclone_drive_remote():
        base["drive_persisted"] = True
    base["comfyui"] = get_comfyui_status_dict()
    try:
        import gpu_stats  # 工作台唤醒时随 app.py 上传；笔记本模式没有这个文件就跳过
        gpu_stats.start_sampler()
        base["gpu"] = gpu_stats.snapshot()
    except Exception:
        pass
    llama_id = next(iter(_loaded_llama_ids()), None)
    if llama_id:
        try:
            import gpu_stats
            base["llm_metrics"] = gpu_stats.llama_metrics(f"http://127.0.0.1:{LOADED_RUNTIMES[llama_id]['port']}", timeout=2)
        except Exception:
            pass
    return base


def get_gpu_and_model_summary_text() -> str:
    info = get_gpu_status_dict()
    lines = []
    if not info["cuda_available"]:
        lines.append("GPU 状态: 未检测到 GPU（请在 Colab 中切换为 A100 / L4 GPU）")
    else:
        lines.append(
            f"GPU: {info['gpu_name']} | 显存可用: {info['vram_free_gb']:.1f} / {info['vram_total_gb']:.1f} GB"
        )
    if info.get("ram_total_gb", 0) > 0:
        lines.append(
            f"实例内存 (RAM): 已用 {info['ram_used_gb']:.1f} GB | 可用 {info['ram_free_gb']:.1f} / {info['ram_total_gb']:.1f} GB"
        )
    if info.get("disk_total_gb", 0) > 0:
        shm_part = (
            f" | /dev/shm 内存盘可用: {info['shm_free_gb']:.1f} / {info['shm_total_gb']:.1f} GB"
            if info.get("shm_total_gb", 0) >= 1.0
            else ""
        )
        lines.append(
            f"实例磁盘: 已用 {info['disk_used_gb']:.1f} GB | 可用 {info['disk_free_gb']:.1f} / {info['disk_total_gb']:.1f} GB{shm_part}"
        )
    lines.append(
        f"算力计费: 费率 ≈ {info['cu_burn_rate_per_hour']:.2f} CU/小时 | 本次开机已用: {info['session_cu_used']:.2f} CU (运行 {info['uptime_seconds']//60} 分钟)"
    )
    ram_tag = " + /dev/shm 内存盘(2.4GB/s)" if info.get("ramdisk_enabled") else ""
    storage_tag = ("Google Drive 云盘持久化" if info["drive_persisted"] else "Colab 高速磁盘") + ram_tag
    lines.append(
        f"当前模型 [{info['selected_model_name']}] 权重 [{storage_tag}]: 已就绪 {info['downloaded_count']}/{info['total_count']} 个文件 ({info['total_size_gb']:.2f} GB)"
    )
    for f in info["files"][:8]:
        mark = "[已就绪]" if f["exists"] else "[未下载]"
        sz = f"({f['size_gb']:.2f} GB)" if f["exists"] else ""
        lines.append(f"   {mark} {f['name']} {sz}")
    if len(info["files"]) > 8:
        lines.append(f"   ... 其余 {len(info['files']) - 8} 个分片文件")

    vram_state = f"已加载进显存 ({info['current_model']})" if info["any_model_loaded"] else "尚未加载进显存"
    lines.append(f"显存引擎状态: {vram_state}")
    return "\n".join(lines)


def _link_forward():
    # 前向引用：定义在后面的模块，函数体运行时才用；入口在全部模块加载完后统一回填，保证严格按切片顺序加载
    global _loaded_llama_ids, _poll_local_comfyui_runtime_quiet, get_comfyui_status_dict
    from .flash_next_glue import _loaded_llama_ids
    from .comfy_runtime import _poll_local_comfyui_runtime_quiet, get_comfyui_status_dict
