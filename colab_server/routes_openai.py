"""OpenAI 兼容协议：/v1/models、/v1/chat/completions、/v1/images/generations。"""
import io
import uuid
import time
import base64
from typing import Optional, List, Dict, Any
from starlette.concurrency import run_in_threadpool
import json
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from . import state
from .common import LOADED_RUNTIMES, GPU_LOCK
from .idle import touch_activity
from .flash_next_glue import _flash_next_port, _flash_next_proxy
from .llm_jobs import _get_qwen_llm_bundle, _count_bundle_tokens
from .api_auth import _verify_external_api_auth
from .qwen_stream import (
    _extract_qwen_tool_calls,
    _IncrementalQwenStreamParser,
    _normalize_to_hf_messages_and_images,
    _normalize_hf_tools,
)
from .qwen_prepare import _prepare_qwen_streamer_generation
from .routes_common import api_base


def attach(app: FastAPI):
    API_BASE = api_base()

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
