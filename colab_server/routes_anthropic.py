"""Anthropic Messages 协议：/v1/messages、/v1/messages/count_tokens。"""
import uuid
from typing import Optional, List, Dict, Any
from starlette.concurrency import run_in_threadpool
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from . import state
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
