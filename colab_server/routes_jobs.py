"""任务与模型生命周期接口：download / load / unload / cancel / generate / job / media / disconnect。"""
import os
import uuid
import time
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from . import state
from .bootstrap import CONSOLE_LOCK, ACTIVE_TQDM_BARS, get_live_console_Lines
from .common import (
    LOADED_RUNTIMES,
    GPU_LOCK,
    ABORT_GENERATION_EVENT,
    JOBS,
    _JOB_CANCEL_EVENTS,
    _clamp_context_length,
    _is_llm_model_id,
)
from .config import (
    MODEL_DOWNLOAD_STATE,
    DRIVE_MOUNTED,
    MODEL_CATALOG,
    DEFAULT_OUTPUT_DIR,
    LEGACY_DRIVE_OUTPUT_DIRS,
)
from .idle import touch_activity
from .telemetry import get_live_telemetry_snapshot, get_gpu_status_dict
from .weights import download_model_weights_core
from .flash_next_glue import _is_llama_model_id, flash_next_load_core
from .vram import unload_model_from_vram_core
from .loader import load_model_to_vram_core
from .comfy_packs import disconnect_colab_runtime
from .schemas import DownloadModelRequest, LoadModelRequest, GenerateVideoRequest
from .llm_jobs import _get_qwen_llm_bundle
from .jobs import _run_async_generation_job
from .api_auth import UnloadModelRequest
from .routes_common import api_base


def attach(app: FastAPI):
    API_BASE = api_base()

    @app.post(API_BASE + "/minimax_api/download_model")
    def api_download_model(req: DownloadModelRequest):
        touch_activity()
        target_id = req.model_id if (req.model_id and req.model_id in MODEL_CATALOG) else state.SELECTED_MODEL_ID
        if not MODEL_DOWNLOAD_STATE["is_downloading"]:
            t = threading.Thread(
                target=download_model_weights_core,
                args=(target_id, req.hf_token),
                daemon=True,
            )
            t.start()
        return get_gpu_status_dict(model_id=target_id)

    @app.post(API_BASE + "/minimax_api/load_model")
    def api_load_model(req: LoadModelRequest):
        touch_activity()
        target_id = req.model_id if (req.model_id and req.model_id in MODEL_CATALOG) else state.SELECTED_MODEL_ID
        if _is_llama_model_id(target_id):  # 上下文上限由 llama_specs 决定，不走 256K 截断的全局设置
            if not MODEL_DOWNLOAD_STATE["is_loading_vram"]:
                threading.Thread(target=flash_next_load_core, args=(req.options, req.context_length, target_id), daemon=True).start()
            return get_gpu_status_dict(model_id=target_id)
        if req.context_length is not None:
            state.QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)
        if not MODEL_DOWNLOAD_STATE["is_loading_vram"]:
            t = threading.Thread(
                target=load_model_to_vram_core,
                args=(target_id, req.precision_mode, req.offload_mode, req.hf_token, req.context_length),
                daemon=True,
            )
            t.start()
        return get_gpu_status_dict(model_id=target_id)

    @app.post(API_BASE + "/minimax_api/unload_model")
    def api_unload_model(req: UnloadModelRequest):
        touch_activity()
        unload_model_from_vram_core(req.model_id or None)
        return get_gpu_status_dict(model_id=state.SELECTED_MODEL_ID)

    @app.post(API_BASE + "/minimax_api/cancel")
    def api_cancel_generation():
        touch_activity()
        ABORT_GENERATION_EVENT.set()
        for ev in list(_JOB_CANCEL_EVENTS.values()):
            ev.set()
        for jid, j in list(JOBS.items()):
            if j.get("status") in ("queued", "running"):
                j["status"] = "failed"
                j["progress"] = 0.0
                j["message"] = "已手动停止当前推理任务"
        state.CURRENT_STAGE_BAR_LABEL = None
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        print("[任务控制] 已收到停止指令，正在中断当前 GPU 推理并清空进度缓存...")

        def _clear_abort_after_release():
            with GPU_LOCK:
                ABORT_GENERATION_EVENT.clear()
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()

        threading.Thread(target=_clear_abort_after_release, daemon=True).start()
        return get_gpu_status_dict(model_id=state.SELECTED_MODEL_ID)

    @app.post(API_BASE + "/minimax_api/generate")
    def api_generate(req: GenerateVideoRequest):
        touch_activity()
        target_model_id = (
            req.model_id
            if (req.model_id and req.model_id in MODEL_CATALOG)
            else state.SELECTED_MODEL_ID
        )
        if not LOADED_RUNTIMES and state.PIPELINE is None and state.LLM_BUNDLE is None:
            return {
                "job_id": "none",
                "status": "failed",
                "progress": 0.0,
                "message": "尚未有任何模型加载至 GPU 显存！请先在左侧点击「下载权重」与「加载到显存」。",
                "seed": req.seed,
                "elapsed": 0.0,
                "filename": None,
                "video_url": None,
            }

        # 若请求的是 LLM/VLM 但当前显存中驻留的是另一变体（如选中了 BF16 但已加载 GGUF，或反之），自动无缝路由至已就绪变体
        if _is_llm_model_id(target_model_id) and target_model_id not in LOADED_RUNTIMES:
            alt_bundle = _get_qwen_llm_bundle(preferred_model_id=target_model_id)
            if alt_bundle and alt_bundle.get("model_id") in LOADED_RUNTIMES:
                target_model_id = str(alt_bundle["model_id"])
                req.model_id = target_model_id

        if target_model_id not in LOADED_RUNTIMES and state.LOADED_MODEL_ID != target_model_id:
            resident_names = "、".join(rt["name"] for rt in LOADED_RUNTIMES.values()) or "无"
            req_name = MODEL_CATALOG.get(target_model_id, {}).get("name", target_model_id)
            return {
                "job_id": "none",
                "status": "failed",
                "progress": 0.0,
                "message": f"{req_name} 尚未加载到显存（当前已常驻：{resident_names}）。请点击左侧「加载到显存」，显存充裕时可同时常驻无需卸载现有模型。",
                "seed": req.seed,
                "elapsed": 0.0,
                "filename": None,
                "video_url": None,
            }

        # 若此前有残留未结束的任务，先置位其专属中断信号并清空进度条，防止旧任务继续占用 GPU 或污染新任务进度
        for old_jid, old_j in list(JOBS.items()):
            if old_j.get("status") in ("queued", "running"):
                old_ev = _JOB_CANCEL_EVENTS.get(old_jid)
                if old_ev is not None:
                    old_ev.set()
                old_j["status"] = "failed"
                old_j["progress"] = 0.0
                old_j["message"] = "已被新提交任务接替中断"
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()

        is_llm_req = _is_llm_model_id(target_model_id)
        batch_count = 1 if is_llm_req else max(1, min(8, int(getattr(req, "batch_count", 1) or 1)))
        job_id = uuid.uuid4().hex[:12]
        _JOB_CANCEL_EVENTS[job_id] = threading.Event()
        JOBS[job_id] = {
            "job_id": job_id,
            "model_id": target_model_id,
            "status": "queued",
            "progress": 0.0,
            "message": f"任务已提交至 Colab 队列 (共 {batch_count} 项)..." if batch_count > 1 else "任务已提交至 Colab 队列...",
            "seed": req.seed,
            "started_at": time.time(),
            "elapsed": 0.0,
            "filename": None,
            "video_url": None,
            "media_type": "text" if is_llm_req else (
                "image" if MODEL_CATALOG.get(target_model_id, {}).get("modality") == "image" else "video"
            ),
            "text_output": "",
            "thinking_output": "",
            "batch_count": batch_count,
            "completed_count": 0,
            "completed_videos": [],
        }
        t = threading.Thread(target=_run_async_generation_job, args=(job_id, req), daemon=True)
        t.start()
        return JOBS[job_id]

    @app.get(API_BASE + "/minimax_api/job/{job_id}")
    def api_job_status(job_id: str):
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
        res = dict(JOBS[job_id])
        if res.get("status") in ("queued", "running"):
            touch_activity()
            if res.get("started_at"):
                res["elapsed"] = round(max(0.0, time.time() - float(res["started_at"])), 1)
        res["console_lines"] = get_live_console_Lines()
        res["telemetry"] = get_live_telemetry_snapshot()
        return res

    @app.get(API_BASE + "/minimax_api/video/{filename}")
    @app.get(API_BASE + "/minimax_api/media/{filename}")
    def api_download_media(filename: str):
        touch_activity()
        safe_name = os.path.basename(filename)
        file_path = os.path.join(DEFAULT_OUTPUT_DIR, safe_name)
        if not os.path.exists(file_path) and DRIVE_MOUNTED:
            for leg_out in LEGACY_DRIVE_OUTPUT_DIRS:
                alt_path = os.path.join(leg_out, safe_name)
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Media file not found")
        ext = os.path.splitext(safe_name)[1].lower()
        mime = (
            "image/png" if ext == ".png"
            else "image/jpeg" if ext in (".jpg", ".jpeg")
            else "image/webp" if ext == ".webp"
            else "video/mp4"
        )
        return FileResponse(file_path, media_type=mime, filename=safe_name)

    @app.post(API_BASE + "/minimax_api/disconnect")
    def api_disconnect():
        msg = disconnect_colab_runtime()
        return {"ok": True, "message": msg}
