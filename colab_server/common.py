import os
import time
import atexit
import secrets
import subprocess
import threading
from typing import Optional, Dict, Any
from . import state


from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

# ==============================================================================
# 全局状态、算力计费账本与 /dev/shm 内存盘加速 (借鉴 comfy-colab-kit)
# ==============================================================================
BOOT_TIME = time.time()

import gc

LOADED_RUNTIMES: Dict[str, Dict[str, Any]] = {}  # 多模型显存共驻表: model_id -> runtime dict
# 每次启动随机生成的路径密钥：全部 /minimax_api 与 /v1 路由挂载在 /k/<RUNTIME_SECRET>/ 之下，
# 仅启动日志输出的完整地址可访问；裸路径只返回 403，防止 gradio.live URL 泄露后 API 被直接调用
RUNTIME_SECRET = secrets.token_urlsafe(16)
GPU_LOCK = threading.Lock()
ABORT_GENERATION_EVENT = threading.Event()
JOBS: Dict[str, Dict[str, Any]] = {}
_JOB_CANCEL_EVENTS: Dict[str, threading.Event] = {}


class GenerationAbortedError(RuntimeError):
    """用户主动点击停止推理时立即中断 DiffSynth / Transformers GPU 循环的受控异常"""
    pass


def _clamp_context_length(val: Optional[int]) -> int:
    try:
        v = int(val) if val is not None else state.QWEN_CONTEXT_LENGTH
    except Exception:
        v = state.QWEN_CONTEXT_LENGTH
    if v <= 0:
        v = state.QWEN_CONTEXT_LENGTH
    return max(4096, min(262144, v))


def _is_llm_model_id(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    return MODEL_CATALOG.get(model_id, {}).get("modality") == "llm_vlm"


def _stop_gguf_server_bundle(bundle: Optional[Dict[str, Any]]):
    if not bundle or bundle.get("backend") not in ("gguf", "flash_next"):
        return
    proc = bundle.pop("server_proc", None)
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except Exception:
            pass
        time.sleep(0.25)


def _cleanup_all_gguf_servers():
    for rt in list(LOADED_RUNTIMES.values()):
        _stop_gguf_server_bundle(rt.get("llm_bundle"))
    _stop_gguf_server_bundle(state.LLM_BUNDLE)


atexit.register(_cleanup_all_gguf_servers)
if os.path.exists("/content"):
    subprocess.run(["pkill", "-f", "llama-server.*Qwen3\\.8-(27B|Flash)"], check=False)


def _make_abort_stopping_criteria(local_cancel: Optional[threading.Event] = None):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _AbortCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if ABORT_GENERATION_EVENT.is_set():
                return True
            if local_cancel is not None and local_cancel.is_set():
                return True
            return False

    return StoppingCriteriaList([_AbortCriteria()])


def _make_timed_token_streamer(tokenizer, timeout: float = 120.0):
    from transformers import TextIteratorStreamer

    class _TimedStreamer(TextIteratorStreamer):
        def __init__(self, tok, **kw):
            super().__init__(tok, **kw)
            self.generated_tokens = 0
            self.first_token_ts: Optional[float] = None
            self._saw_prompt = False

        def put(self, value):
            now = time.time()
            if self.skip_prompt and not self._saw_prompt:
                self._saw_prompt = True
            else:
                n = int(value.numel()) if hasattr(value, "numel") else 1
                self.generated_tokens += max(1, n)
                if self.first_token_ts is None:
                    self.first_token_ts = now
            super().put(value)

        @property
        def tok_per_sec(self) -> float:
            if self.first_token_ts is None or self.generated_tokens <= 1:
                return 0.0
            dt = max(0.05, time.time() - self.first_token_ts)
            return round((self.generated_tokens - 1) / dt, 1)

    return _TimedStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=timeout,
    )


def _link_forward():
    # 前向引用：定义在后面的模块，函数体运行时才用；入口在全部模块加载完后统一回填，保证严格按切片顺序加载
    global MODEL_CATALOG
    from .config import MODEL_CATALOG
