"""ComfyUI 云端节点工作台与 Google Drive 持久化接口（/minimax_api/comfyui/*）。"""
import os
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from . import state
from .config import COMFYUI_STATE
from .idle import touch_activity
from .telemetry import get_gpu_status_dict
from .comfy_runtime import _resolve_comfyui_persistence_paths, get_comfyui_status_dict, free_comfyui_vram_core
from .comfy_sync import sync_comfyui_to_drive_core, stop_comfyui_core
from .comfy_start import start_comfyui_core, download_comfyui_model_core
from .comfy_packs import install_comfyui_preset_pack_core
from .schemas import ComfyStartRequest, ComfyWeightDownloadRequest, ComfyInstallPackRequest
from .routes_common import api_base


def attach(app: FastAPI):
    API_BASE = api_base()

    # --------------------------------------------------------------------------
    # ComfyUI 云端节点工作台与 Google Drive 持久化专用接口 (/minimax_api/comfyui/*)
    # --------------------------------------------------------------------------
    @app.get(API_BASE + "/minimax_api/comfyui/status")
    def api_comfyui_status():
        return {"ok": True, "comfyui": get_comfyui_status_dict()}

    @app.post(API_BASE + "/minimax_api/comfyui/start")
    def api_comfyui_start(req: ComfyStartRequest):
        touch_activity()
        if not COMFYUI_STATE["is_starting"]:
            threading.Thread(
                target=start_comfyui_core,
                args=(req.unload_existing, req.force_restart),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=state.SELECTED_MODEL_ID)

    @app.post(API_BASE + "/minimax_api/comfyui/stop")
    def api_comfyui_stop():
        touch_activity()
        stop_comfyui_core(sync_before_stop=True)
        return get_gpu_status_dict(model_id=state.SELECTED_MODEL_ID)

    @app.post(API_BASE + "/minimax_api/comfyui/free_vram")
    def api_comfyui_free_vram():
        touch_activity()
        ok, msg = free_comfyui_vram_core()
        res = get_gpu_status_dict(model_id=state.SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        return res

    @app.post(API_BASE + "/minimax_api/comfyui/sync_drive")
    def api_comfyui_sync_drive():
        touch_activity()
        ok, msg = sync_comfyui_to_drive_core(quiet=False)
        res = get_gpu_status_dict(model_id=state.SELECTED_MODEL_ID)
        res["action_ok"] = ok
        res["action_message"] = msg
        return res

    @app.post(API_BASE + "/minimax_api/comfyui/download_weight")
    def api_comfyui_download_weight(req: ComfyWeightDownloadRequest):
        touch_activity()
        if not COMFYUI_STATE["is_downloading_weight"]:
            threading.Thread(
                target=download_comfyui_model_core,
                args=(req.url_or_repo, req.filename, req.subfolder, req.hf_token),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=state.SELECTED_MODEL_ID)

    @app.post(API_BASE + "/minimax_api/comfyui/install_pack")
    def api_comfyui_install_pack(req: ComfyInstallPackRequest):
        touch_activity()
        if not COMFYUI_STATE["is_downloading_weight"] and not COMFYUI_STATE.get("installing_pack_id"):
            threading.Thread(
                target=install_comfyui_preset_pack_core,
                args=(req.pack_id, req.auto_start, req.hf_token),
                daemon=True,
            ).start()
        return get_gpu_status_dict(model_id=state.SELECTED_MODEL_ID)

    @app.get(API_BASE + "/minimax_api/comfyui/output_media/{rel_path:path}")
    def api_comfyui_output_media(rel_path: str):
        touch_activity()
        comfy_root, persist_root, _, _, _ = _resolve_comfyui_persistence_paths()
        for base_out in (os.path.join(comfy_root, "output"), os.path.join(persist_root, "output")):
            if not os.path.isdir(base_out):
                continue
            cand = os.path.normpath(os.path.join(base_out, rel_path))
            # commonpath 严格限制在 base_out 内；startswith 会放行 outputX 这类同级前缀目录
            if os.path.commonpath([os.path.abspath(cand), os.path.abspath(base_out)]) != os.path.abspath(base_out):
                raise HTTPException(status_code=400, detail="Invalid output path")
            if os.path.isfile(cand):
                safe_name = os.path.basename(cand)
                ext = os.path.splitext(safe_name)[1].lower()
                mime = (
                    "image/png" if ext == ".png"
                    else "image/jpeg" if ext in (".jpg", ".jpeg")
                    else "image/webp" if ext == ".webp"
                    else "image/gif" if ext == ".gif"
                    else "video/webm" if ext == ".webm"
                    else "video/mp4"
                )
                return FileResponse(cand, media_type=mime, filename=safe_name)
        raise HTTPException(status_code=404, detail="ComfyUI output file not found")
