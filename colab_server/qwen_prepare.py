import gc
import socket
import threading
from typing import Optional, List, Dict, Any
import json
import urllib.parse
import urllib.request
import torch
from PIL import Image
from . import state
from .common import (
    GPU_LOCK,
    ABORT_GENERATION_EVENT,
    _make_abort_stopping_criteria,
    _make_timed_token_streamer,
)
from .idle import touch_activity
from .gguf_server import _make_noproxy_opener
from .llm_jobs import _get_qwen_llm_bundle
from .qwen_stream import _GGUFStreamIterator, _hf_messages_to_gguf_openai_messages


def _prepare_gguf_streamer_generation(
    bundle: Dict[str, Any],
    hf_messages: List[Dict[str, Any]],
    hf_tools: Optional[List[Dict[str, Any]]],
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: Optional[float] = None,
    repeat_penalty: float = 1.0,
    reasoning_effort: Optional[str] = None,
):
    """通过驻留 GPU 的 llama-server 流式执行 Qwen3.8-27B-UD-Q4_K_M 推理 (遵循 Unsloth 官方 GGUF 部署参数规范，支持最高 256K 上下文与视觉输入)"""
    oai_msgs, est_prompt_tokens = _hf_messages_to_gguf_openai_messages(hf_messages)
    bundle_ctx = int(bundle.get("context_length") or state.QWEN_CONTEXT_LENGTH)
    max_toks = max(16, min(65536, int(max_new_tokens or 16384), max(256, bundle_ctx - min(est_prompt_tokens, bundle_ctx - 256))))

    streamer = _GGUFStreamIterator(est_prompt_tokens=est_prompt_tokens)
    cancel_event = threading.Event()
    err_box: List[Optional[Exception]] = [None]

    eff_presence = (
        float(presence_penalty)
        if presence_penalty is not None
        else (0.0 if enable_thinking else 1.5)
    )
    tpl_kwargs: Dict[str, Any] = {"enable_thinking": bool(enable_thinking)}
    if enable_thinking and reasoning_effort and reasoning_effort not in ("default", "none"):
        tpl_kwargs["reasoning_effort"] = str(reasoning_effort)

    eff_temp = max(0.0, min(2.0, float(temperature)))
    eff_top_p_val = max(0.05, min(1.0, float(top_p)))
    eff_top_k_val = max(0, int(top_k))
    eff_min_p_val = max(0.0, min(1.0, float(min_p)))
    eff_pres_val = max(0.0, min(2.0, eff_presence))
    eff_rep_val = max(0.5, min(2.0, float(repeat_penalty)))

    # 针对 Qwen3.8 超大词表 (248,320) + MTP 多词元投机解码定制采样管线：
    # - 无惩罚项时 (Thinking 默认) 仅保留 GPU 原生采样器，100% 在 GPU 端完成采样，零 PCIe Logits 拷贝；
    # - 启用惩罚项时 (Non-Thinking presence_penalty=1.5) 先在 GPU 执行 top_k 将 248K 词表筛至 20 项，再计算 penalties。
    has_penalties = (eff_pres_val > 1e-5) or (abs(eff_rep_val - 1.0) > 1e-4)
    req_samplers: List[str] = []
    if eff_top_k_val > 0:
        req_samplers.append("top_k")
    if has_penalties:
        req_samplers.append("penalties")
    if eff_top_p_val < 0.9999:
        req_samplers.append("top_p")
    if eff_min_p_val > 1e-6:
        req_samplers.append("min_p")
    req_samplers.append("temperature")

    payload: Dict[str, Any] = {
        "model": "Qwen3.8-27B-UD-Q4_K_M",
        "messages": oai_msgs,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_toks,
        "temperature": eff_temp,
        "top_p": eff_top_p_val,
        "top_k": eff_top_k_val,
        "min_p": eff_min_p_val,
        "presence_penalty": eff_pres_val,
        "repeat_penalty": eff_rep_val,
        "samplers": req_samplers,
        "chat_template_kwargs": tpl_kwargs,
    }
    if hf_tools:
        payload["tools"] = hf_tools

    def _locked_gguf_generate():
        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            streamer.end()
            return

        with GPU_LOCK:
            if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                streamer.end()
                return

            srv_proc = bundle.get("server_proc")
            if srv_proc is not None and srv_proc.poll() is not None:
                err_box[0] = RuntimeError(
                    f"llama-server 子进程已意外退出 (code={srv_proc.returncode})，请在左侧重新点击「加载到显存」。"
                )
                streamer.end()
                return

            opener = _make_noproxy_opener()
            url = f"{bundle['base_url']}/v1/chat/completions"
            req_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            http_req = urllib.request.Request(
                url,
                data=req_bytes,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                method="POST",
            )

            in_reasoning = False
            tc_acc: Dict[int, Dict[str, str]] = {}
            resp = None
            raw_sock = None
            stop_watch = threading.Event()

            try:
                # 严禁将 raw_sock.settimeout 设为短秒数：Python 3.13 SocketIO 一旦触发单次超时会将 _timeout_occurred 永久置 True，
                # 导致后续 resp.readline() 立即抛出 OSError("cannot read from timed out object")。
                # 此处保持长超时 900s 以覆盖 256K 超长上下文预填充，并由独立守护线程监听中断信号关闭套接字。
                resp = opener.open(http_req, timeout=900.0)
                raw_sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)

                def _abort_watcher():
                    while not stop_watch.wait(0.25):
                        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            if raw_sock is not None:
                                try:
                                    raw_sock.shutdown(socket.SHUT_RDWR)
                                except Exception:
                                    pass
                            if resp is not None:
                                try:
                                    resp.close()
                                except Exception:
                                    pass
                            break

                threading.Thread(target=_abort_watcher, daemon=True).start()

                while True:
                    if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                        break
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue

                    if isinstance(chunk.get("error"), dict):
                        raise RuntimeError(str(chunk["error"].get("message") or chunk["error"]))

                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        if usage.get("prompt_tokens"):
                            streamer.prompt_tokens = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens"):
                            streamer.generated_tokens = max(streamer.generated_tokens, int(usage["completion_tokens"]))

                    timings = chunk.get("timings")
                    if isinstance(timings, dict) and timings.get("predicted_per_second"):
                        streamer._server_tps = float(timings["predicted_per_second"])

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    r_txt = delta.get("reasoning_content")
                    if r_txt:
                        if not in_reasoning:
                            streamer.put_text("<think>\n" + str(r_txt), token_inc=1)
                            in_reasoning = True
                        else:
                            streamer.put_text(str(r_txt), token_inc=1)

                    c_txt = delta.get("content")
                    if c_txt:
                        if in_reasoning:
                            streamer.put_text("</think>\n" + str(c_txt), token_inc=1)
                            in_reasoning = False
                        else:
                            streamer.put_text(str(c_txt), token_inc=1)

                    d_tcs = delta.get("tool_calls")
                    if isinstance(d_tcs, list):
                        for tc_item in d_tcs:
                            if not isinstance(tc_item, dict):
                                continue
                            idx = int(tc_item.get("index", 0))
                            slot = tc_acc.setdefault(idx, {"name": "", "arguments": ""})
                            fn_obj = tc_item.get("function") or {}
                            if fn_obj.get("name"):
                                slot["name"] += str(fn_obj["name"])
                            if fn_obj.get("arguments"):
                                slot["arguments"] += str(fn_obj["arguments"])

                if in_reasoning:
                    streamer.put_text("</think>\n", token_inc=0)
                    in_reasoning = False

                if tc_acc:
                    for idx in sorted(tc_acc.keys()):
                        slot = tc_acc[idx]
                        if not slot["name"]:
                            continue
                        try:
                            arg_val = json.loads(slot["arguments"]) if slot["arguments"] else {}
                        except Exception:
                            arg_val = {"raw": slot["arguments"]}
                        tc_str = json.dumps({"name": slot["name"], "arguments": arg_val}, ensure_ascii=False)
                        streamer.put_text(f"<tool_call>\n{tc_str}\n</tool_call>", token_inc=0)

            except urllib.error.HTTPError as he:
                err_body = ""
                try:
                    err_body = he.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if he.code == 400 and ("context" in err_body.lower() or "exceed" in err_body.lower()):
                    err_box[0] = RuntimeError(
                        f"输入长度超出当前 GGUF 上下文窗口 ({bundle_ctx // 1024}K = {bundle_ctx} tokens)！"
                        f"请在左侧调大「上下文长度」（最高 256K）后点击重新加载模型。({err_body[:180]})"
                    )
                else:
                    err_box[0] = RuntimeError(f"GGUF 推理服务返回 HTTP {he.code}: {err_body[:240]}")
            except Exception as ex:
                if not (cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set()):
                    err_box[0] = ex
            finally:
                stop_watch.set()
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                streamer.end()
                touch_activity()

    t = threading.Thread(target=_locked_gguf_generate, daemon=True)
    t.start()
    return streamer, streamer.prompt_tokens, False, t, err_box, cancel_event


def _prepare_qwen_streamer_generation(
    hf_messages: List[Dict[str, Any]],
    pil_images: List[Image.Image],
    hf_tools: Optional[List[Dict[str, Any]]],
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    preferred_model_id: Optional[str] = None,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: Optional[float] = None,
    repeat_penalty: float = 1.0,
    reasoning_effort: Optional[str] = None,
):
    """构建 Qwen3.8 (GGUF 或 BF16 Transformers) 后台流式推理任务并返回 (streamer, prompt_tokens, start_in_thinking, gen_thread, err_box, cancel_event)"""
    bundle = _get_qwen_llm_bundle(preferred_model_id=preferred_model_id)
    if bundle is not None and bundle.get("backend") == "gguf":
        return _prepare_gguf_streamer_generation(
            bundle=bundle,
            hf_messages=hf_messages,
            hf_tools=hf_tools,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            reasoning_effort=reasoning_effort,
        )

    model = bundle["model"]
    processor = bundle["processor"]
    bundle_ctx = int(bundle.get("context_length") or state.QWEN_CONTEXT_LENGTH)

    tpl_kwargs: Dict[str, Any] = dict(
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if enable_thinking and reasoning_effort and reasoning_effort not in ("default", "none"):
        tpl_kwargs["reasoning_effort"] = str(reasoning_effort)
    if hf_tools:
        tpl_kwargs["tools"] = hf_tools

    try:
        prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)
    except TypeError:
        tpl_kwargs.pop("reasoning_effort", None)
        try:
            prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)
        except TypeError:
            tpl_kwargs.pop("enable_thinking", None)
            prompt_text = processor.apply_chat_template(hf_messages, **tpl_kwargs)

    start_in_thinking = enable_thinking and (
        prompt_text.rstrip().endswith("<think>") or "<think>" in prompt_text[-24:]
    ) and not prompt_text.rstrip().endswith("</think>")

    prompt_tokens = max(1, len(processor.tokenizer.encode(prompt_text)))
    streamer = _make_timed_token_streamer(processor.tokenizer, timeout=180.0)

    max_toks = max(16, min(65536, int(max_new_tokens or 16384), max(128, bundle_ctx - prompt_tokens)))
    do_sample = temperature > 0.01
    cancel_event = threading.Event()
    err_box: List[Optional[Exception]] = [None]

    def _locked_generate():
        if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            try:
                streamer.end()
            except Exception:
                pass
            return

        with GPU_LOCK:
            if cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                try:
                    streamer.end()
                except Exception:
                    pass
                return

            inputs = None
            try:
                if pil_images:
                    inputs = processor(text=[prompt_text], images=pil_images, padding=True, return_tensors="pt").to(model.device)
                else:
                    inputs = processor(text=[prompt_text], padding=True, return_tensors="pt").to(model.device)

                seq_in_len = int(inputs["input_ids"].shape[-1])
                if seq_in_len >= bundle_ctx:
                    raise RuntimeError(
                        f"输入长度 ({seq_in_len} tokens) 超出当前设定的上下文上限 ({bundle_ctx // 1024}K = {bundle_ctx} tokens)，"
                        f"请在左侧调大「上下文长度」（最高支持 256K）。"
                    )
                eff_max_toks = max(16, min(max_toks, bundle_ctx - seq_in_len))

                gen_kwargs: Dict[str, Any] = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=eff_max_toks,
                    do_sample=do_sample,
                    stopping_criteria=_make_abort_stopping_criteria(cancel_event),
                )
                if do_sample:
                    gen_kwargs["temperature"] = max(0.05, min(2.0, float(temperature)))
                    gen_kwargs["top_p"] = max(0.05, min(1.0, float(top_p)))
                    if int(top_k) > 0:
                        gen_kwargs["top_k"] = int(top_k)

                with torch.inference_mode():
                    model.generate(**gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                err_box[0] = RuntimeError(
                    "GPU 显存不足 (OOM)：BF16 原生精度长上下文峰值显存较高，建议切换使用独立的「Qwen3.8-27B-GGUF (UD-Q4_K_M)」量化模型，或适当调低上下文长度。"
                )
                try:
                    streamer.end()
                except Exception:
                    pass
            except Exception as ex:
                err_box[0] = ex
                try:
                    streamer.end()
                except Exception:
                    pass
            finally:
                if inputs is not None:
                    del inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                touch_activity()

    t = threading.Thread(target=_locked_generate, daemon=True)
    t.start()
    return streamer, prompt_tokens, start_in_thinking, t, err_box, cancel_event
