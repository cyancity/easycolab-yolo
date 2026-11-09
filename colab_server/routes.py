import os
import io
import uuid
import time
import base64
import threading
from typing import Optional, List, Dict, Any
from starlette.concurrency import run_in_threadpool
import json
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from . import state
from .bootstrap import CONSOLE_LOCK, ACTIVE_TQDM_BARS, get_live_console_Lines
from .common import (
    LOADED_RUNTIMES,
    RUNTIME_SECRET,
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
    COMFYUI_STATE,
    MODEL_CATALOG,
    DEFAULT_OUTPUT_DIR,
    LEGACY_DRIVE_OUTPUT_DIRS,
)
from .idle import touch_activity
from .telemetry import get_live_telemetry_snapshot, get_gpu_status_dict
from .weights import download_model_weights_core
from .flash_next_glue import (
    _is_llama_model_id,
    flash_next_load_core,
    _flash_next_port,
    _flash_next_proxy,
)
from .vram import _sync_legacy_globals, unload_model_from_vram_core
from .loader import load_model_to_vram_core
from .comfy_runtime import (
    _resolve_comfyui_persistence_paths,
    get_comfyui_status_dict,
    free_comfyui_vram_core,
)
from .comfy_sync import sync_comfyui_to_drive_core, stop_comfyui_core
from .comfy_start import start_comfyui_core, download_comfyui_model_core
from .comfy_packs import install_comfyui_preset_pack_core, disconnect_colab_runtime
from .schemas import (
    SelectModelRequest,
    DownloadModelRequest,
    LoadModelRequest,
    ContextConfigRequest,
    IdleConfigRequest,
    ComfyStartRequest,
    ComfyWeightDownloadRequest,
    ComfyInstallPackRequest,
    GenerateVideoRequest,
)
from .llm_jobs import _get_qwen_llm_bundle, _count_bundle_tokens
from .jobs import _run_async_generation_job
from .api_auth import ApiKeyConfigRequest, UnloadModelRequest, _verify_external_api_auth
from .qwen_stream import (
    _extract_qwen_tool_calls,
    _IncrementalQwenStreamParser,
    _normalize_to_hf_messages_and_images,
    _normalize_hf_tools,
)
from .qwen_prepare import _prepare_qwen_streamer_generation


def attach_macos_api_routes(app: FastAPI):
    """在 Gradio 底层的 FastAPI 实例上挂载原生 macOS 客户端接口 + OpenAI / Anthropic 双协议 API 网关"""
    API_BASE = "/k/" + RUNTIME_SECRET  # 启动日志输出的完整地址包含本段

    try:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except Exception:
        # Gradio 启动后的 FastAPI 实例默认已内置全局 CORS 中间件
        pass

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

    # --------------------------------------------------------------------------
    # 1. OpenAI 兼容协议: GET /v1/models, POST /v1/chat/completions, POST /v1/images/generations
    # --------------------------------------------------------------------------
    @app.get(API_BASE + "/v1/models")
    @app.get(API_BASE + "/models")
    def openai_list_models(request: Request):
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        now_ts = int(time.time())
        any_qwen_loaded = ("qwen38_27b" in LOADED_RUNTIMES) or ("qwen38_27b_gguf" in LOADED_RUNTIMES)
        models_data = [
            {"id": "unsloth/Qwen3.8-27B-GGUF", "object": "model", "created": now_ts, "owned_by": "unsloth", "loaded": "qwen38_27b_gguf" in LOADED_RUNTIMES},
            {"id": "Qwen3.8-27B-UD-Q4_K_M", "object": "model", "created": now_ts, "owned_by": "unsloth", "loaded": "qwen38_27b_gguf" in LOADED_RUNTIMES},
            {"id": "Qwen/Qwen3.8-27B", "object": "model", "created": now_ts, "owned_by": "qwen", "loaded": any_qwen_loaded},
            {"id": "qwen3.8-27b", "object": "model", "created": now_ts, "owned_by": "qwen", "loaded": any_qwen_loaded},
            {"id": "qwen3.8-flash-next", "object": "model", "created": now_ts, "owned_by": "unsloth", "loaded": "qwen38_flash_next" in LOADED_RUNTIMES},
            {"id": "qwen3.5-4b-smoke", "object": "model", "created": now_ts, "owned_by": "unsloth", "loaded": "qwen35_4b_smoke" in LOADED_RUNTIMES},
            {"id": "Qwen/Qwen-Image-2.1", "object": "model", "created": now_ts, "owned_by": "qwen", "loaded": "qwen_image_21" in LOADED_RUNTIMES},
            {"id": "Tongyi-MAI/Z-Image-Turbo", "object": "model", "created": now_ts, "owned_by": "tongyi", "loaded": "z_image_turbo" in LOADED_RUNTIMES},
            {"id": "MiniMaxAI/MiniMax-H3", "object": "model", "created": now_ts, "owned_by": "minimax", "loaded": "minimax_h3" in LOADED_RUNTIMES},
        ]
        return {"object": "list", "data": models_data}

    @app.post(API_BASE + "/v1/chat/completions")
    @app.post(API_BASE + "/chat/completions")
    async def openai_chat_completions(request: Request):
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()

        body = await request.json()
        fn_port = _flash_next_port(str(body.get("model") or ""))
        if fn_port:
            state.API_REQUEST_COUNT += 1
            return await _flash_next_proxy(fn_port, "/v1/chat/completions", await request.body())
        req_model = body.get("model") or "Qwen/Qwen3.8-27B"
        bundle = _get_qwen_llm_bundle(preferred_model_id=str(req_model))
        if bundle is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Qwen3.8-27B / Qwen3.8-27B-GGUF 尚未加载至 Colab GPU 显存，请先在 Mac 客户端点击「加载到显存」。",
                        "type": "server_error",
                        "code": "model_not_loaded",
                    }
                },
            )

        state.API_REQUEST_COUNT += 1
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 16384)

        # 支持通过 enable_thinking / chat_template_kwargs / reasoning_effort 控制思考模式与思考深度
        enable_thinking = True
        effort_val: Optional[str] = None
        ctk = body.get("chat_template_kwargs") if isinstance(body.get("chat_template_kwargs"), dict) else {}
        if "enable_thinking" in body:
            enable_thinking = bool(body["enable_thinking"])
        elif "enable_thinking" in ctk:
            enable_thinking = bool(ctk["enable_thinking"])
        raw_eff = str(body.get("reasoning_effort") or ctk.get("reasoning_effort") or "").strip().lower()
        if raw_eff in ("none", "disabled", "off"):
            enable_thinking = False
        elif raw_eff:
            effort_val = raw_eff

        temperature = float(body["temperature"]) if body.get("temperature") is not None else (1.0 if enable_thinking else 0.7)
        top_p = float(body["top_p"]) if body.get("top_p") is not None else (0.95 if enable_thinking else 0.80)
        top_k = int(body["top_k"]) if body.get("top_k") is not None else 20
        min_p = float(body["min_p"]) if body.get("min_p") is not None else 0.0
        presence_penalty = float(body["presence_penalty"]) if body.get("presence_penalty") is not None else (0.0 if enable_thinking else 1.5)
        repeat_penalty = float(body["repeat_penalty"]) if body.get("repeat_penalty") is not None else 1.0

        print(f"[OpenAI API] POST /v1/chat/completions | model={req_model} | backend={bundle.get('backend')} | stream={stream} | thinking={enable_thinking}")

        hf_messages, pil_images = _normalize_to_hf_messages_and_images(body.get("messages", []), protocol="openai")
        hf_tools = _normalize_hf_tools(body.get("tools"), protocol="openai")

        streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
            hf_messages=hf_messages,
            pil_images=pil_images,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            preferred_model_id=str(req_model),
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            reasoning_effort=effort_val,
        )

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        created_ts = int(time.time())

        if not stream:
            # 非流式生成可能持续数分钟，同步 drain 必须放入线程池，避免阻塞事件循环
            def _drain_nonstream():
                parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
                think_parts: List[str] = []
                text_parts: List[str] = []
                out_chunks = 0
                try:
                    for piece in streamer:
                        out_chunks += 1
                        for kind, val in parser.feed(piece):
                            if kind == "thinking":
                                think_parts.append(val)
                            elif kind == "text":
                                text_parts.append(val)
                finally:
                    cancel_event.set()
                tail_events, tool_calls = parser.finish()
                for kind, val in tail_events:
                    if kind == "thinking":
                        think_parts.append(val)
                    elif kind == "text":
                        text_parts.append(val)
                gen_thread.join(timeout=5.0)
                if err_box[0] is not None:
                    return JSONResponse(status_code=500, content={"error": {"message": str(err_box[0]), "type": "server_error"}})

                full_thinking = "".join(think_parts).strip()
                full_text, extra_tools = _extract_qwen_tool_calls("".join(text_parts).strip())
                all_tools = tool_calls + extra_tools

                msg_obj: Dict[str, Any] = {"role": "assistant", "content": full_text if (full_text or not all_tools) else None}
                if full_thinking:
                    msg_obj["reasoning_content"] = full_thinking
                if all_tools:
                    msg_obj["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                        }
                        for tc in all_tools
                    ]

                final_prompt_tokens = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
                comp_tokens = int(getattr(streamer, "generated_tokens", 0)) or (
                    _count_bundle_tokens(bundle, full_thinking + full_text) if (full_thinking or full_text) else max(1, out_chunks)
                )
                finish_reason = "tool_calls" if all_tools else "stop"

                return {
                    "id": cmpl_id,
                    "object": "chat.completion",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [{"index": 0, "message": msg_obj, "finish_reason": finish_reason}],
                    "usage": {
                        "prompt_tokens": final_prompt_tokens,
                        "completion_tokens": comp_tokens,
                        "total_tokens": final_prompt_tokens + comp_tokens,
                    },
                }

            return await run_in_threadpool(_drain_nonstream)

        def _openai_sse_stream():
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            comp_chars = 0
            try:
                first_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

                for piece in streamer:
                    comp_chars += len(piece)
                    for kind, val in parser.feed(piece):
                        if not val:
                            continue
                        delta_payload = {"reasoning_content": val} if kind == "thinking" else {"content": val}
                        chunk_obj = {
                            "id": cmpl_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": req_model,
                            "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n"

                tail_events, tool_calls = parser.finish()
                for kind, val in tail_events:
                    if not val:
                        continue
                    delta_payload = {"reasoning_content": val} if kind == "thinking" else {"content": val}
                    chunk_obj = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req_model,
                        "choices": [{"index": 0, "delta": delta_payload, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n"

                gen_thread.join(timeout=5.0)
                if getattr(streamer, "generated_tokens", 0) > 0:
                    print(f"[OpenAI API] 流式完成 | {streamer.generated_tokens} tokens | {streamer.tok_per_sec} tok/s")
            finally:
                cancel_event.set()

            if tool_calls:
                tc_deltas = [
                    {
                        "index": idx,
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                    }
                    for idx, tc in enumerate(tool_calls)
                ]
                tc_chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {"tool_calls": tc_deltas}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"

            final_prompt_toks = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
            est_out_toks = int(getattr(streamer, "generated_tokens", 0)) or max(1, comp_chars // 3)
            final_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "usage": {
                    "prompt_tokens": final_prompt_toks,
                    "completion_tokens": est_out_toks,
                    "total_tokens": final_prompt_toks + est_out_toks,
                },
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _openai_sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.post(API_BASE + "/v1/images/generations")
    @app.post(API_BASE + "/images/generations")
    async def openai_images_generations(request: Request):
        """当显存中驻留了 Qwen-Image-2.1 或 Z-Image-Turbo（可与 Qwen3.8-27B 同时共驻）时，提供标准 OpenAI 生图接口"""
        auth_err = _verify_external_api_auth(request, protocol="openai")
        if auth_err:
            return auth_err
        touch_activity()

        body = await request.json()
        req_model_str = str(body.get("model") or "").lower()
        img_target_id: Optional[str] = None
        if ("z-image" in req_model_str or "z_image" in req_model_str) and "z_image_turbo" in LOADED_RUNTIMES:
            img_target_id = "z_image_turbo"
        elif "qwen" in req_model_str and "qwen_image_21" in LOADED_RUNTIMES:
            img_target_id = "qwen_image_21"
        else:
            for cand in ("z_image_turbo", "qwen_image_21"):
                if cand in LOADED_RUNTIMES:
                    img_target_id = cand
                    break

        if not img_target_id:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "请先在客户端加载 Qwen-Image-2.1 或 Z-Image-Turbo 至显存（可与 Qwen3.8-27B 同时常驻）。", "type": "server_error"}},
            )

        img_pipe = LOADED_RUNTIMES[img_target_id]["pipeline"]
        state.API_REQUEST_COUNT += 1
        prompt = body.get("prompt", "")
        size_str = str(body.get("size", "1024x1024")).lower()
        try:
            w, h = [int(x) for x in size_str.split("x", 1)]
        except Exception:
            w, h = 1024, 1024
        n = max(1, min(4, int(body.get("n", 1))))
        steps = int(body.get("steps") or (8 if img_target_id == "z_image_turbo" else 40))

        # 生图全程持有 GPU_LOCK 且逐张推理耗时较长，放入线程池避免阻塞事件循环
        def _gen_images():
            results = []
            with GPU_LOCK:
                for _ in range(n):
                    cur_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                    if img_target_id == "z_image_turbo":
                        img = img_pipe(prompt=prompt, width=w, height=h, num_inference_steps=steps, seed=cur_seed, rand_device="cuda")
                    else:
                        img = img_pipe(prompt=prompt, negative_prompt=" ", width=w, height=h, num_inference_steps=steps, seed=cur_seed)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    results.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})
            return results

        results = await run_in_threadpool(_gen_images)
        return {"created": int(time.time()), "data": results}

    # --------------------------------------------------------------------------
    # 2. Anthropic Messages 协议: POST /v1/messages & POST /v1/messages/count_tokens
    #    完美兼容 Anthropic Python/TS SDK、Claude Code (ANTHROPIC_BASE_URL) 等工具
    # --------------------------------------------------------------------------
    @app.post(API_BASE + "/v1/messages/count_tokens")
    @app.post(API_BASE + "/messages/count_tokens")
    @app.post(API_BASE + "/v1/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        auth_err = _verify_external_api_auth(request, protocol="anthropic")
        if auth_err:
            return auth_err
        body = await request.json()
        raw_txt = json.dumps(body.get("messages", []), ensure_ascii=False)
        bundle = _get_qwen_llm_bundle(preferred_model_id=str(body.get("model") or ""))
        toks = _count_bundle_tokens(bundle, raw_txt)
        return {"input_tokens": toks}

    @app.post(API_BASE + "/v1/messages")
    @app.post(API_BASE + "/messages")
    @app.post(API_BASE + "/v1/v1/messages")
    async def anthropic_messages(request: Request):
        auth_err = _verify_external_api_auth(request, protocol="anthropic")
        if auth_err:
            return auth_err
        touch_activity()

        body = await request.json()
        fn_port = _flash_next_port(str(body.get("model") or ""))
        if fn_port:
            state.API_REQUEST_COUNT += 1
            return await _flash_next_proxy(fn_port, "/v1/messages", await request.body())
        req_model = body.get("model") or "Qwen/Qwen3.8-27B"
        bundle = _get_qwen_llm_bundle(preferred_model_id=str(req_model))
        if bundle is None:
            return JSONResponse(
                status_code=503,
                content={
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "Qwen3.8-27B / Qwen3.8-27B-GGUF 尚未加载至 Colab GPU 显存，请先在 Mac 客户端点击「加载到显存」。",
                    },
                },
            )

        state.API_REQUEST_COUNT += 1
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens") or 16384)

        thinking_cfg = body.get("thinking")
        if isinstance(thinking_cfg, dict):
            enable_thinking = thinking_cfg.get("type") != "disabled"
        else:
            enable_thinking = True

        temperature = float(body["temperature"]) if body.get("temperature") is not None else (1.0 if enable_thinking else 0.7)
        top_p = float(body["top_p"]) if body.get("top_p") is not None else (0.95 if enable_thinking else 0.80)
        top_k = int(body["top_k"]) if body.get("top_k") is not None else 20

        print(f"[Anthropic API] POST /v1/messages | model={req_model} | backend={bundle.get('backend')} | stream={stream} | thinking={enable_thinking}")

        hf_messages, pil_images = _normalize_to_hf_messages_and_images(
            body.get("messages", []),
            system_prompt=body.get("system"),
            protocol="anthropic",
        )
        hf_tools = _normalize_hf_tools(body.get("tools"), protocol="anthropic")

        streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
            hf_messages=hf_messages,
            pil_images=pil_images,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            preferred_model_id=str(req_model),
            top_k=top_k,
            min_p=0.0,
            presence_penalty=0.0 if enable_thinking else 1.5,
            repeat_penalty=1.0,
        )

        msg_id = f"msg_{uuid.uuid4().hex[:22]}"

        if not stream:
            # 非流式生成可能持续数分钟，同步 drain 必须放入线程池，避免阻塞事件循环
            def _drain_anthropic_nonstream():
                parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
                think_parts: List[str] = []
                text_parts: List[str] = []
                try:
                    for piece in streamer:
                        for kind, val in parser.feed(piece):
                            if kind == "thinking":
                                think_parts.append(val)
                            elif kind == "text":
                                text_parts.append(val)
                finally:
                    cancel_event.set()
                tail_events, tool_calls = parser.finish()
                for kind, val in tail_events:
                    if kind == "thinking":
                        think_parts.append(val)
                    elif kind == "text":
                        text_parts.append(val)
                gen_thread.join(timeout=5.0)
                if err_box[0] is not None:
                    return JSONResponse(
                        status_code=500,
                        content={"type": "error", "error": {"type": "api_error", "message": str(err_box[0])}},
                    )

                full_thinking = "".join(think_parts).strip()
                full_text, extra_tools = _extract_qwen_tool_calls("".join(text_parts).strip())
                all_tools = tool_calls + extra_tools

                content_blocks: List[Dict[str, Any]] = []
                if full_thinking:
                    content_blocks.append({"type": "thinking", "thinking": full_thinking, "signature": "qwen38-sig"})
                if full_text or not all_tools:
                    content_blocks.append({"type": "text", "text": full_text})
                for tc in all_tools:
                    content_blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})

                final_prompt_tokens = int(getattr(streamer, "prompt_tokens", prompt_tokens) or prompt_tokens)
                out_tokens = int(getattr(streamer, "generated_tokens", 0)) or (
                    _count_bundle_tokens(bundle, full_thinking + full_text) if (full_thinking or full_text) else 1
                )
                stop_reason = "tool_use" if all_tools else "end_turn"

                return {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": req_model,
                    "content": content_blocks,
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "usage": {"input_tokens": final_prompt_tokens, "output_tokens": out_tokens},
                }

            return await run_in_threadpool(_drain_anthropic_nonstream)

        def _anthropic_sse_event(ev_name: str, payload: Dict[str, Any]) -> str:
            return f"event: {ev_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def _anthropic_sse_stream():
            parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
            yield _anthropic_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": req_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": prompt_tokens, "output_tokens": 1},
                    },
                },
            )

            next_block_idx = 0
            active_block_type: Optional[str] = None  # "thinking" | "text" | None
            active_block_idx = 0
            comp_chars = 0

            def _ensure_block(target_type: str):
                nonlocal next_block_idx, active_block_type, active_block_idx
                out_strs: List[str] = []
                if active_block_type == target_type:
                    return out_strs
                if active_block_type is not None:
                    if active_block_type == "thinking":
                        out_strs.append(
                            _anthropic_sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": active_block_idx,
                                    "delta": {"type": "signature_delta", "signature": "qwen38-sig"},
                                },
                            )
                        )
                    out_strs.append(
                        _anthropic_sse_event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": active_block_idx},
                        )
                    )
                active_block_idx = next_block_idx
                next_block_idx += 1
                active_block_type = target_type
                block_init = (
                    {"type": "thinking", "thinking": ""}
                    if target_type == "thinking"
                    else {"type": "text", "text": ""}
                )
                out_strs.append(
                    _anthropic_sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": active_block_idx, "content_block": block_init},
                    )
                )
                return out_strs

            try:
                for piece in streamer:
                    comp_chars += len(piece)
                    for kind, val in parser.feed(piece):
                        if not val:
                            continue
                        for s in _ensure_block(kind):
                            yield s
                        delta_obj = (
                            {"type": "thinking_delta", "thinking": val}
                            if kind == "thinking"
                            else {"type": "text_delta", "text": val}
                        )
                        yield _anthropic_sse_event(
                            "content_block_delta",
                            {"type": "content_block_delta", "index": active_block_idx, "delta": delta_obj},
                        )

                tail_events, tool_calls = parser.finish()
                for kind, val in tail_events:
                    if not val:
                        continue
                    for s in _ensure_block(kind):
                        yield s
                    delta_obj = (
                        {"type": "thinking_delta", "thinking": val}
                        if kind == "thinking"
                        else {"type": "text_delta", "text": val}
                    )
                    yield _anthropic_sse_event(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": active_block_idx, "delta": delta_obj},
                    )

                gen_thread.join(timeout=5.0)
                if getattr(streamer, "generated_tokens", 0) > 0:
                    print(f"[Anthropic API] 流式完成 | {streamer.generated_tokens} tokens | {streamer.tok_per_sec} tok/s")
            finally:
                cancel_event.set()

            if active_block_type is not None:
                if active_block_type == "thinking":
                    yield _anthropic_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": active_block_idx,
                            "delta": {"type": "signature_delta", "signature": "qwen38-sig"},
                        },
                    )
                yield _anthropic_sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": active_block_idx},
                )
                active_block_type = None
            elif not tool_calls:
                # 若全程无输出块，保底发送一个空文本块符合 Anthropic 协议规范
                yield _anthropic_sse_event(
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                )
                yield _anthropic_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
                next_block_idx = 1

            for tc in tool_calls:
                t_idx = next_block_idx
                next_block_idx += 1
                yield _anthropic_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": t_idx,
                        "content_block": {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": {}},
                    },
                )
                yield _anthropic_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": t_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    },
                )
                yield _anthropic_sse_event("content_block_stop", {"type": "content_block_stop", "index": t_idx})

            est_out_toks = int(getattr(streamer, "generated_tokens", 0)) or max(1, comp_chars // 3)
            yield _anthropic_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use" if tool_calls else "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": est_out_toks},
                },
            )
            yield _anthropic_sse_event("message_stop", {"type": "message_stop"})

        return StreamingResponse(
            _anthropic_sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

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

    # ---- 裸路径兼容层：未带 /k/<密钥> 前缀的旧路径一律 403，防止 gradio.live URL 泄露后 API 被直接调用 ----
    def _legacy_denied(request: Request):
        return JSONResponse(
            status_code=403,
            content={"error": "本实例已启用路径密钥：请使用启动日志输出的完整服务地址（含 /k/<token> 段）访问 API。"},
        )

    @app.api_route("/minimax_api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def _legacy_minimax_api(request: Request, rest: str = ""):
        return _legacy_denied(request)

    @app.api_route("/v1/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def _legacy_v1(request: Request, rest: str = ""):
        return _legacy_denied(request)

    for _bare_path in ("/models", "/chat/completions", "/images/generations", "/messages", "/messages/count_tokens"):
        app.add_api_route(_bare_path, _legacy_denied, methods=["GET", "POST"])
