"""状态与配置接口：status / select_model / config_idle / config_context / config_api_key。"""
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from . import state
from .common import LOADED_RUNTIMES, _clamp_context_length
from .idle import touch_activity
from .telemetry import get_gpu_status_dict
from .vram import _sync_legacy_globals
from .schemas import SelectModelRequest, ContextConfigRequest, IdleConfigRequest
from .api_auth import ApiKeyConfigRequest
from .routes_common import api_base


def attach(app: FastAPI):
    API_BASE = api_base()

    @app.get(API_BASE + "/minimax_api/status")
    def api_status(model_id: Optional[str] = None):
        return get_gpu_status_dict(model_id=model_id)

    @app.post(API_BASE + "/minimax_api/select_model")
    def api_select_model(req: SelectModelRequest):
        touch_activity()
        return get_gpu_status_dict(model_id=req.model_id)

    @app.post(API_BASE + "/minimax_api/config_idle")
    def api_config_idle(req: IdleConfigRequest):
        state.IDLE_TIMEOUT_MINUTES = req.idle_timeout_minutes
        touch_activity()
        return get_gpu_status_dict()

    @app.post(API_BASE + "/minimax_api/config_context")
    def api_config_context(req: ContextConfigRequest):
        state.QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)
        # BF16 Transformers 运行时支持直接热更新动态上下文上限；GGUF 需重载以重分配 KV 槽位
        bf16_rt = LOADED_RUNTIMES.get("qwen38_27b")
        if bf16_rt and bf16_rt.get("llm_bundle"):
            bf16_rt["llm_bundle"]["context_length"] = state.QWEN_CONTEXT_LENGTH
            bf16_rt["mode_label"] = f"Qwen3.8-27B · BF16 原生 · {state.QWEN_CONTEXT_LENGTH // 1024}K"
            _sync_legacy_globals()
        touch_activity()
        print(f"[上下文配置] 目标最大上下文长度已设为 {state.QWEN_CONTEXT_LENGTH // 1024}K ({state.QWEN_CONTEXT_LENGTH} tokens)")
        return get_gpu_status_dict(model_id=state.SELECTED_MODEL_ID)

    @app.post(API_BASE + "/minimax_api/config_api_key")
    def api_config_api_key(req: ApiKeyConfigRequest):
        new_key = (req.api_key or "").strip()
        if not new_key:
            # 免密开放模式已移除：空密钥一律拒绝（见 _verify_external_api_auth），只允许换新值
            return JSONResponse(
                status_code=400,
                content={"error": "鉴权已强制开启，不允许清空 API Key（免密开放模式已移除）。如需更换密钥请传入新值。"},
            )
        state.API_AUTH_KEY = new_key
        touch_activity()
        print("[API 网关] 对外协议网关 (已开启密钥鉴权)")
        return get_gpu_status_dict()
