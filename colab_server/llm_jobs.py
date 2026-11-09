import io
import time
import base64
import threading
from typing import Optional, List, Tuple, Dict, Any
import json
import urllib.parse
import urllib.request
from PIL import Image
from . import state
from .common import (
    LOADED_RUNTIMES,
    ABORT_GENERATION_EVENT,
    JOBS,
    _JOB_CANCEL_EVENTS,
    GenerationAbortedError,
    _clamp_context_length,
    _is_llm_model_id,
)
from .idle import touch_activity
from .gguf_server import _make_noproxy_opener
from .schemas import GenerateVideoRequest


def decode_base64_image(b64_str: Optional[str]) -> Optional[Image.Image]:
    if not b64_str:
        return None
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _split_thinking_and_answer(raw_text: str) -> Tuple[str, str]:
    """解析 Qwen3.8 <think>...</think> 推理链与最终回复"""
    text = raw_text or ""
    if "</think>" in text:
        think_part, ans_part = text.split("</think>", 1)
        think_part = think_part.replace("<think>", "").strip()
        return think_part, ans_part.strip()
    if text.lstrip().startswith("<think>"):
        return text.replace("<think>", "", 1).strip(), ""
    return "", text.strip()


def _is_bundle_alive(bundle: Optional[Dict[str, Any]]) -> bool:
    if not bundle:
        return False
    if bundle.get("backend") == "gguf":
        proc = bundle.get("server_proc")
        return proc is not None and proc.poll() is None
    return bundle.get("model") is not None


def _get_qwen_llm_bundle(preferred_model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []
    if preferred_model_id:
        low = preferred_model_id.lower()
        if preferred_model_id in ("qwen38_27b", "qwen38_27b_gguf"):
            candidates.append(preferred_model_id)
        elif "gguf" in low or "q4_k" in low or "unsloth" in low:
            candidates.append("qwen38_27b_gguf")
        elif "qwen3.8" in low or "qwen38" in low:
            candidates.extend(["qwen38_27b_gguf", "qwen38_27b"])

    if _is_llm_model_id(state.SELECTED_MODEL_ID) and state.SELECTED_MODEL_ID not in candidates:
        candidates.append(state.SELECTED_MODEL_ID)

    for fallback_id in ("qwen38_27b_gguf", "qwen38_27b"):
        if fallback_id not in candidates:
            candidates.append(fallback_id)

    for cid in candidates:
        rt = LOADED_RUNTIMES.get(cid)
        if rt:
            b = rt.get("llm_bundle")
            if _is_bundle_alive(b):
                return b
            # GGUF 子进程意外退出时清理陈旧表项
            if b and b.get("backend") == "gguf":
                LOADED_RUNTIMES.pop(cid, None)

    if _is_bundle_alive(state.LLM_BUNDLE):
        return state.LLM_BUNDLE
    return None


def _count_bundle_tokens(bundle: Optional[Dict[str, Any]], text: str) -> int:
    if not text:
        return 1
    if bundle is not None:
        if bundle.get("backend") == "gguf":
            try:
                opener = _make_noproxy_opener()
                tok_req = urllib.request.Request(
                    f"{bundle['base_url']}/tokenize",
                    data=json.dumps({"content": text}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with opener.open(tok_req, timeout=4.0) as resp:
                    obj = json.loads(resp.read().decode("utf-8"))
                    toks = obj.get("tokens")
                    if isinstance(toks, list):
                        return max(1, len(toks))
            except Exception:
                pass
        elif bundle.get("processor") is not None:
            try:
                return max(1, len(bundle["processor"].tokenizer.encode(text)))
            except Exception:
                pass
    return max(1, len(text) // 3)


def _run_qwen_llm_job(job_id: str, req: GenerateVideoRequest, target_llm_id: Optional[str] = None):
    """运行 Qwen3.8-27B (BF16 原生 或 Unsloth UD-Q4_K_M GGUF) 深度推理对话与多模态视觉理解"""
    start_time = time.time()
    if req.context_length is not None:
        state.QWEN_CONTEXT_LENGTH = _clamp_context_length(req.context_length)

    bundle = _get_qwen_llm_bundle(preferred_model_id=target_llm_id or req.model_id)
    if bundle is None:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["message"] = "Qwen3.8-27B 尚未加载至 GPU 显存！请先在左侧点击「加载到显存」。"
        return

    model_label = "Qwen3.8-27B-GGUF" if bundle.get("backend") == "gguf" else "Qwen3.8-27B"
    first_img = decode_base64_image(req.first_frame_base64)
    last_img = decode_base64_image(req.last_frame_base64)
    attached_images = [im for im in (first_img, last_img) if im is not None]

    messages: List[Dict[str, Any]] = []
    if req.chat_messages:
        for m in req.chat_messages[:-1]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content:
                messages.append({"role": role, "content": [{"type": "text", "text": content}]})

    user_content: List[Dict[str, Any]] = []
    for im in attached_images:
        user_content.append({"type": "image", "image": im})
    user_content.append({"type": "text", "text": req.visual_prompt})
    messages.append({"role": "user", "content": user_content})

    JOBS[job_id]["media_type"] = "text"
    JOBS[job_id]["progress"] = 0.12
    JOBS[job_id]["message"] = f"{model_label} 正在编码多模态上下文并启动推理..."

    job_cancel_event = _JOB_CANCEL_EVENTS.setdefault(job_id, threading.Event())
    if job_cancel_event.is_set():
        raise GenerationAbortedError("已手动停止当前推理任务")
    ABORT_GENERATION_EVENT.clear()
    bundle_ctx = int(bundle.get("context_length") or state.QWEN_CONTEXT_LENGTH)
    max_toks = max(64, min(65536, int(req.max_new_tokens or 16384), max(256, bundle_ctx - 256)))
    think_on = bool(req.enable_thinking)
    # 遵循 Unsloth Qwen3.8 官方部署采样表：Thinking (temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence=0.0)；Non-Thinking (temp=0.7, top_p=0.80, top_k=20, min_p=0.0, presence=1.5)
    default_temp = 1.0 if think_on else 0.7
    default_top_p = 0.95 if think_on else 0.80
    default_presence = 0.0 if think_on else 1.5
    temp = max(0.0, min(2.0, float(req.temperature if req.temperature is not None else default_temp)))
    eff_top_p = max(0.05, min(1.0, float(req.top_p if req.top_p is not None else default_top_p)))
    eff_presence = max(0.0, min(2.0, float(req.presence_penalty if req.presence_penalty is not None else default_presence)))
    eff_top_k = max(0, int(req.top_k if req.top_k is not None else 20))
    eff_min_p = max(0.0, min(1.0, float(req.min_p if req.min_p is not None else 0.0)))
    eff_repeat = max(0.5, min(2.0, float(req.repeat_penalty if req.repeat_penalty is not None else 1.0)))
    eff_effort = str(req.reasoning_effort or "medium").strip().lower()

    streamer, _, start_in_thinking, gen_thread, err_box, cancel_event = _prepare_qwen_streamer_generation(
        hf_messages=messages,
        pil_images=attached_images,
        hf_tools=None,
        enable_thinking=think_on,
        max_new_tokens=max_toks,
        temperature=temp,
        top_p=eff_top_p,
        preferred_model_id=bundle.get("model_id"),
        top_k=eff_top_k,
        min_p=eff_min_p,
        presence_penalty=eff_presence,
        repeat_penalty=eff_repeat,
        reasoning_effort=eff_effort,
    )

    parser = _IncrementalQwenStreamParser(start_in_thinking=start_in_thinking)
    think_parts: List[str] = []
    ans_parts: List[str] = []
    token_chunks = 0
    last_ui_sync = 0.0

    try:
        for piece in streamer:
            if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                cancel_event.set()
                break
            token_chunks += 1
            for kind, val in parser.feed(piece):
                if kind == "thinking":
                    think_parts.append(val)
                elif kind == "text":
                    ans_parts.append(val)
            now_ts = time.time()
            # 按 ~10Hz 节流拼接流式字符串快照，避免 100+ tok/s 高速解码下逐 token O(N^2) 字符串拷贝争抢 GIL
            if token_chunks <= 2 or (now_ts - last_ui_sync) >= 0.10:
                last_ui_sync = now_ts
                think_txt = "".join(think_parts).strip()
                ans_txt = "".join(ans_parts).strip()
                JOBS[job_id]["thinking_output"] = think_txt
                JOBS[job_id]["text_output"] = ans_txt
                JOBS[job_id]["progress"] = min(0.95, round(0.15 + 0.80 * min(1.0, token_chunks / float(max(128.0, max_toks * 0.5))), 3))
                elapsed_now = round(now_ts - start_time, 1)
                JOBS[job_id]["elapsed"] = elapsed_now
                tps = streamer.tok_per_sec
                speed_tag = f" · {tps} tok/s" if tps > 0 else ""
                JOBS[job_id]["message"] = (
                    f"{model_label} 正在深度思考 ({elapsed_now}s{speed_tag})..."
                    if (think_txt and not ans_txt)
                    else f"{model_label} 正在流式输出回复 ({elapsed_now}s{speed_tag})..."
                )
                touch_activity()
    except Exception:
        cancel_event.set()
    finally:
        cancel_event.set()

    tail_events, _ = parser.finish()
    for kind, val in tail_events:
        if kind == "thinking":
            think_parts.append(val)
        elif kind == "text":
            ans_parts.append(val)

    gen_thread.join(timeout=10.0)
    if err_box[0] is not None:
        raise err_box[0]

    think_final = "".join(think_parts).strip()
    ans_final = "".join(ans_parts).strip()
    if req.enable_thinking and not think_final and "</think>" in ans_final:
        think_final, ans_final = _split_thinking_and_answer("<think>\n" + ans_final)

    if job_cancel_event.is_set() or (ABORT_GENERATION_EVENT.is_set() and not (think_final or ans_final)):
        raise GenerationAbortedError("已手动停止当前推理任务")

    elapsed = round(time.time() - start_time, 1)
    final_tps = streamer.tok_per_sec
    gen_toks = streamer.generated_tokens
    tps_summary = f" · 共 {gen_toks} tokens · {final_tps} tok/s" if gen_toks > 0 and final_tps > 0 else ""
    JOBS[job_id]["thinking_output"] = think_final
    JOBS[job_id]["text_output"] = ans_final
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["progress"] = 1.0
    JOBS[job_id]["elapsed"] = elapsed
    JOBS[job_id]["message"] = f"{model_label} 推理完成！耗时 {elapsed}s{tps_summary}"
    print(f"[{model_label}] 推理完成 | {gen_toks} tokens | {final_tps} tok/s | 耗时 {elapsed}s")


def _fmt_mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


def _link_forward():
    # 前向引用：定义在后面的模块，函数体运行时才用；入口在全部模块加载完后统一回填，保证严格按切片顺序加载
    global _IncrementalQwenStreamParser, _prepare_qwen_streamer_generation
    from .qwen_stream import _IncrementalQwenStreamParser
    from .qwen_prepare import _prepare_qwen_streamer_generation
