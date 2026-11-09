import io
import re
import uuid
import time
import queue
import base64
from typing import Optional, List, Tuple, Dict, Any
import json
from PIL import Image
from .llm_jobs import decode_base64_image
from .api_auth import _load_pil_from_url_or_b64


def _extract_qwen_tool_calls(raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """解析 Qwen3 / Qwen3.8 输出中的 <tool_call>...</tool_call> (兼容 JSON 与 XML 参数格式)"""
    if "<tool_call>" not in raw_text:
        return raw_text, []

    tool_calls: List[Dict[str, Any]] = []
    tc_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    for match in tc_pattern.finditer(raw_text):
        inner = match.group(1).strip()
        call_id = f"toolu_{uuid.uuid4().hex[:16]}"
        parsed_name = None
        parsed_args: Dict[str, Any] = {}

        # 格式 1: JSON {"name": "...", "arguments": {...}}
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict) and "name" in obj:
                parsed_name = str(obj["name"])
                raw_args = obj.get("arguments", obj.get("parameters", {}))
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception:
                        parsed_args = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
        except Exception:
            pass

        # 格式 2: Qwen3.5/3.8 XML <function=fn_name><parameter=k>v</parameter></function>
        if not parsed_name:
            fn_m = re.search(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", inner, re.DOTALL)
            if fn_m:
                parsed_name = fn_m.group(1).strip()
                body = fn_m.group(2)
                for pm in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
                    p_key = pm.group(1).strip()
                    p_val_raw = pm.group(2).strip()
                    try:
                        parsed_args[p_key] = json.loads(p_val_raw)
                    except Exception:
                        parsed_args[p_key] = p_val_raw

        if parsed_name:
            tool_calls.append({"id": call_id, "name": parsed_name, "arguments": parsed_args})

    cleaned_text = tc_pattern.sub("", raw_text).strip()
    return cleaned_text, tool_calls


class _IncrementalQwenStreamParser:
    """流式增量拆分 <think>...</think>、正文与 <tool_call>...</tool_call>，杜绝半截标签外泄"""

    def __init__(self, start_in_thinking: bool):
        self.in_thinking = start_in_thinking
        self.in_tool_call = False
        self.buf = ""
        self.tool_raw = ""

    def feed(self, piece: str) -> List[Tuple[str, str]]:
        events: List[Tuple[str, str]] = []
        self.buf += piece

        while self.buf:
            if self.in_tool_call:
                self.tool_raw += self.buf
                self.buf = ""
                break

            if self.in_thinking:
                # 忽略开头的显式 <think>
                if self.buf.lstrip().startswith("<think>"):
                    idx = self.buf.find("<think>")
                    self.buf = self.buf[idx + len("<think>"):]
                    continue
                end_idx = self.buf.find("</think>")
                if end_idx != -1:
                    before = self.buf[:end_idx]
                    if before:
                        events.append(("thinking", before))
                    self.buf = self.buf[end_idx + len("</think>"):].lstrip("\n")
                    self.in_thinking = False
                    continue
                # 保留末尾可能未闭合的 '<' 前缀
                lt = self.buf.rfind("<")
                if lt != -1 and len(self.buf) - lt < 12 and "</think>".startswith(self.buf[lt:]):
                    safe = self.buf[:lt]
                    if safe:
                        events.append(("thinking", safe))
                    self.buf = self.buf[lt:]
                    break
                events.append(("thinking", self.buf))
                self.buf = ""
            else:
                if self.buf.lstrip().startswith("<think>"):
                    idx = self.buf.find("<think>")
                    before = self.buf[:idx]
                    if before:
                        events.append(("text", before))
                    self.buf = self.buf[idx + len("<think>"):]
                    self.in_thinking = True
                    continue

                tc_idx = self.buf.find("<tool_call>")
                if tc_idx != -1:
                    before = self.buf[:tc_idx]
                    if before:
                        events.append(("text", before))
                    self.tool_raw += self.buf[tc_idx:]
                    self.buf = ""
                    self.in_tool_call = True
                    break

                lt = self.buf.rfind("<")
                if lt != -1 and len(self.buf) - lt < 14 and (
                    "<tool_call>".startswith(self.buf[lt:]) or "<think>".startswith(self.buf[lt:])
                ):
                    safe = self.buf[:lt]
                    if safe:
                        events.append(("text", safe))
                    self.buf = self.buf[lt:]
                    break

                events.append(("text", self.buf))
                self.buf = ""

        return events

    def finish(self) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        events: List[Tuple[str, str]] = []
        if self.buf:
            if self.in_tool_call:
                self.tool_raw += self.buf
            elif self.in_thinking:
                events.append(("thinking", self.buf))
            else:
                events.append(("text", self.buf))
            self.buf = ""

        _, tool_calls = _extract_qwen_tool_calls(self.tool_raw) if self.tool_raw else ("", [])
        return events, tool_calls


def _normalize_to_hf_messages_and_images(
    raw_messages: List[Dict[str, Any]],
    system_prompt: Any = None,
    protocol: str = "openai",
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """将 OpenAI 或 Anthropic 消息数组统一转换为 Qwen3.8 Processor 标准多模态结构"""
    hf_messages: List[Dict[str, Any]] = []
    pil_images: List[Image.Image] = []

    if system_prompt:
        if isinstance(system_prompt, str) and system_prompt.strip():
            hf_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
        elif isinstance(system_prompt, list):
            sys_txt = "\n".join(
                b.get("text", "") for b in system_prompt if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if sys_txt:
                hf_messages.append({"role": "system", "content": [{"type": "text", "text": sys_txt}]})

    for msg in raw_messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "tool":
            txt = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            hf_messages.append({"role": "tool", "content": txt})
            continue

        blocks: List[Dict[str, Any]] = []
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "text":
                    blocks.append({"type": "text", "text": str(part.get("text", ""))})
                elif ptype == "image_url":
                    img_obj = part.get("image_url", {})
                    url_str = img_obj.get("url", "") if isinstance(img_obj, dict) else str(img_obj)
                    im = _load_pil_from_url_or_b64(url_str)
                    if im is not None:
                        pil_images.append(im)
                        blocks.append({"type": "image", "image": im})
                elif ptype == "image":
                    src = part.get("source", {})
                    im = None
                    if isinstance(src, dict):
                        if src.get("type") == "base64":
                            im = decode_base64_image(src.get("data", ""))
                        elif src.get("type") == "url":
                            im = _load_pil_from_url_or_b64(src.get("url", ""))
                    if im is not None:
                        pil_images.append(im)
                        blocks.append({"type": "image", "image": im})
                elif ptype == "tool_use":
                    tc_str = json.dumps({"name": part.get("name", ""), "arguments": part.get("input", {})}, ensure_ascii=False)
                    blocks.append({"type": "text", "text": f"<tool_call>\n{tc_str}\n</tool_call>"})
                elif ptype == "tool_result":
                    tr_content = part.get("content", "")
                    if isinstance(tr_content, list):
                        tr_txt = "\n".join(x.get("text", "") for x in tr_content if isinstance(x, dict))
                    else:
                        tr_txt = str(tr_content)
                    hf_messages.append({"role": "tool", "content": tr_txt})

        # OpenAI assistant historical tool_calls
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                raw_a = fn.get("arguments", "{}")
                try:
                    a_obj = json.loads(raw_a) if isinstance(raw_a, str) else raw_a
                except Exception:
                    a_obj = {}
                tc_json = json.dumps({"name": fn.get("name", ""), "arguments": a_obj}, ensure_ascii=False)
                blocks.append({"type": "text", "text": f"<tool_call>\n{tc_json}\n</tool_call>"})

        if blocks:
            hf_messages.append({"role": role, "content": blocks})

    return hf_messages, pil_images


def _normalize_hf_tools(raw_tools: Optional[List[Dict[str, Any]]], protocol: str) -> Optional[List[Dict[str, Any]]]:
    if not raw_tools:
        return None
    hf_tools: List[Dict[str, Any]] = []
    for t in raw_tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            hf_tools.append(t)
        elif "name" in t:
            # Anthropic tool schema -> HF OpenAI function tool schema
            hf_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
    return hf_tools if hf_tools else None


class _GGUFStreamIterator:
    """与 _TimedStreamer 接口一致的 GGUF 流式迭代器，统一供原生客户端与 OpenAI/Anthropic 网关消费"""

    def __init__(self, est_prompt_tokens: int = 1):
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self.prompt_tokens: int = max(1, int(est_prompt_tokens))
        self.generated_tokens: int = 0
        self.first_token_ts: Optional[float] = None
        self._server_tps: float = 0.0

    def put_text(self, text: str, token_inc: int = 1):
        if not text:
            return
        now = time.time()
        if self.first_token_ts is None:
            self.first_token_ts = now
        if token_inc > 0:
            self.generated_tokens += token_inc
        self._q.put(text)

    def end(self):
        self._q.put(None)

    @property
    def tok_per_sec(self) -> float:
        if self._server_tps > 0:
            return round(self._server_tps, 1)
        if self.first_token_ts is None or self.generated_tokens <= 1:
            return 0.0
        dt = max(0.05, time.time() - self.first_token_ts)
        return round((self.generated_tokens - 1) / dt, 1)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item


def _pil_to_jpeg_data_url(im: Image.Image) -> str:
    rgb = im.convert("RGB")
    max_side = max(rgb.size)
    if max_side > 2048:
        scale = 2048.0 / float(max_side)
        rgb = rgb.resize(
            (max(64, int(rgb.size[0] * scale)), max(64, int(rgb.size[1] * scale))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _hf_messages_to_gguf_openai_messages(hf_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    oai_msgs: List[Dict[str, Any]] = []
    est_chars = 0
    est_imgs = 0

    for msg in hf_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "tool":
            s_txt = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            est_chars += len(s_txt)
            oai_msgs.append({"role": "tool", "content": s_txt, "tool_call_id": msg.get("tool_call_id", "call_0")})
            continue

        if isinstance(content, str):
            est_chars += len(content)
            oai_msgs.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            has_img = any(isinstance(b, dict) and b.get("type") == "image" and b.get("image") is not None for b in content)
            if not has_img:
                txt = "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
                est_chars += len(txt)
                oai_msgs.append({"role": role, "content": txt})
            else:
                parts: List[Dict[str, Any]] = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "image" and b.get("image") is not None:
                        est_imgs += 1
                        parts.append({"type": "image_url", "image_url": {"url": _pil_to_jpeg_data_url(b["image"])}})
                    elif b.get("type") == "text":
                        t_str = str(b.get("text", ""))
                        est_chars += len(t_str)
                        parts.append({"type": "text", "text": t_str})
                oai_msgs.append({"role": role, "content": parts})

    est_tokens = max(1, est_chars // 3 + est_imgs * 512)
    return oai_msgs, est_tokens
