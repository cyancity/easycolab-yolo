import gc
import sys
from typing import Optional, List, Tuple
import torch
from . import state
from .bootstrap import CONSOLE_LOCK, ACTIVE_TQDM_BARS
from .common import (
    LOADED_RUNTIMES,
    GPU_LOCK,
    ABORT_GENERATION_EVENT,
    JOBS,
    _JOB_CANCEL_EVENTS,
    _is_llm_model_id,
    _stop_gguf_server_bundle,
)
from .config import MODEL_DOWNLOAD_STATE
from .idle import touch_activity


def _sync_legacy_globals():
    """将多模型显存驻留表 LOADED_RUNTIMES 同步至兼容全局变量 PIPELINE / LLM_BUNDLE"""
    state.LLM_BUNDLE = None
    if _is_llm_model_id(state.SELECTED_MODEL_ID) and state.SELECTED_MODEL_ID in LOADED_RUNTIMES:
        state.LLM_BUNDLE = LOADED_RUNTIMES[state.SELECTED_MODEL_ID].get("llm_bundle")
    if state.LLM_BUNDLE is None:
        for llm_id in ("qwen38_27b_gguf", "qwen38_27b"):
            rt = LOADED_RUNTIMES.get(llm_id)
            if rt and rt.get("llm_bundle") is not None:
                state.LLM_BUNDLE = rt["llm_bundle"]
                break

    sel_rt = LOADED_RUNTIMES.get(state.SELECTED_MODEL_ID)
    if sel_rt and sel_rt.get("pipeline") is not None:
        state.PIPELINE = sel_rt["pipeline"]
    else:
        state.PIPELINE = None
        for rt in reversed(list(LOADED_RUNTIMES.values())):
            if rt.get("pipeline") is not None:
                state.PIPELINE = rt["pipeline"]
                break

    if state.SELECTED_MODEL_ID in LOADED_RUNTIMES:
        state.LOADED_MODEL_ID = state.SELECTED_MODEL_ID
    elif LOADED_RUNTIMES:
        state.LOADED_MODEL_ID = next(reversed(LOADED_RUNTIMES))
    else:
        state.LOADED_MODEL_ID = None

    if LOADED_RUNTIMES:
        state.CURRENT_MODEL_MODE = " + ".join(rt["mode_label"] for rt in LOADED_RUNTIMES.values())
    else:
        state.CURRENT_MODEL_MODE = None


def _trim_linux_cpu_heap():
    """在释放超大规模 CPU BF16 权重张量后主动触发 glibc malloc_trim(0)，将物理页立即归还给系统 MemAvailable"""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def _unload_model_locked(model_id: Optional[str] = None) -> List[str]:
    """在已持有 GPU_LOCK 的前提下卸载指定模型（或全部模型），返回被卸载的模型名称列表"""
    removed_names: List[str] = []
    if not model_id or model_id == "all":
        for mid in list(LOADED_RUNTIMES.keys()):
            entry = LOADED_RUNTIMES.pop(mid, None)
            if entry:
                _stop_gguf_server_bundle(entry.get("llm_bundle"))
                removed_names.append(entry.get("name", mid))
                entry.clear()
        _stop_gguf_server_bundle(state.LLM_BUNDLE)
        state.PIPELINE = None
        state.LLM_BUNDLE = None
        state.LOADED_MODEL_ID = None
        state.CURRENT_MODEL_MODE = None
    elif model_id in LOADED_RUNTIMES:
        entry = LOADED_RUNTIMES.pop(model_id, None)
        if entry:
            _stop_gguf_server_bundle(entry.get("llm_bundle"))
            removed_names.append(entry.get("name", model_id))
            if state.PIPELINE is entry.get("pipeline"):
                state.PIPELINE = None
            if state.LLM_BUNDLE is entry.get("llm_bundle"):
                state.LLM_BUNDLE = None
            entry.clear()
        _sync_legacy_globals()

    _trim_linux_cpu_heap()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return removed_names


def _unload_all_gpu_models():
    """彻底释放全部已加载模型的 GPU 显存"""
    _unload_model_locked("all")


def unload_model_from_vram_core(model_id: Optional[str] = None) -> Tuple[bool, str]:
    """供客户端按需独立卸载单个或全部显存驻留模型（立即中断正在进行的推理任务以避免死锁）"""
    touch_activity()
    ABORT_GENERATION_EVENT.set()
    for jid, j in list(JOBS.items()):
        if j.get("status") in ("queued", "running") and (not model_id or model_id == "all" or j.get("model_id") == model_id):
            ev = _JOB_CANCEL_EVENTS.get(jid)
            if ev is not None:
                ev.set()
            j["status"] = "failed"
            j["progress"] = 0.0
            j["message"] = "已终止推理并从显存卸载该模型"
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()
    with GPU_LOCK:
        try:
            removed = _unload_model_locked(model_id)
            free_gb = (torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)) if torch.cuda.is_available() else 0.0
            if not removed:
                msg = "当前指定模型未驻留显存，无需卸载。"
            else:
                remain_cnt = len(LOADED_RUNTIMES)
                remain_str = f"（显存中仍保留 {remain_cnt} 个常驻模型）" if remain_cnt > 0 else "（显存已完全清空）"
                msg = f"已从显存卸载 {', '.join(removed)}，当前可用显存 {free_gb:.1f} GB {remain_str}"
            MODEL_DOWNLOAD_STATE["message"] = msg
            print(f"[显存管理] {msg}")
            return True, msg
        finally:
            ABORT_GENERATION_EVENT.clear()


def _ensure_vram_for_new_model_locked(target_id: str, required_free_gb: float) -> List[str]:
    """
    多模型共驻智能显存守卫 (LRU Eviction)：
    - 若目标模型自身已在显存中，先卸载旧实例以便重载；
    - 若剩余可用显存 >= required_free_gb，保留所有已加载模型实现多模型同时常驻！
    - 仅当剩余显存不足以容纳新模型时，按最早加载顺序 (LRU) 自动腾退旧模型，防止 CUDA OOM。
    """
    evicted: List[str] = []
    if target_id in LOADED_RUNTIMES:
        _unload_model_locked(target_id)

    if not torch.cuda.is_available():
        return evicted

    free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
    while free_gb < required_free_gb and LOADED_RUNTIMES:
        oldest_id = next(iter(LOADED_RUNTIMES))
        names = _unload_model_locked(oldest_id)
        evicted.extend(names)
        free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
    return evicted
