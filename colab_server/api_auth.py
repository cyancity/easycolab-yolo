import io
import hmac
from typing import Optional
import urllib.parse
import urllib.request
from PIL import Image
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from . import state
from .llm_jobs import decode_base64_image


class ApiKeyConfigRequest(BaseModel):
    api_key: str = ""


class UnloadModelRequest(BaseModel):
    model_id: str = ""


# ==============================================================================
# 标准对外网关：OpenAI (/v1/chat/completions, /v1/models, /v1/images/generations)
#            与 Anthropic (/v1/messages, /v1/messages/count_tokens) 双协议兼容引擎
# ==============================================================================
def _verify_external_api_auth(request: Request, protocol: str = "openai") -> Optional[JSONResponse]:
    """同时支持 Authorization: Bearer <key> 与 x-api-key: <key> 鉴权。

    密钥为空时一律拒绝（__main__ 启动时已自动生成随机密钥，这里兜底防御）。"""
    expected = (state.API_AUTH_KEY or "").strip()
    auth_hdr = (request.headers.get("authorization") or "").strip()
    bearer_tok = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else ""
    x_api_tok = (request.headers.get("x-api-key") or "").strip()
    provided = bearer_tok or x_api_tok
    if expected and hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        return None
    if protocol == "anthropic":
        return JSONResponse(
            status_code=401,
            content={"type": "error", "error": {"type": "authentication_error", "message": "Invalid API Key"}},
        )
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "Invalid API Key", "type": "invalid_request_error", "code": "invalid_api_key"}},
    )


def _load_pil_from_url_or_b64(src: str) -> Optional[Image.Image]:
    if not src:
        return None
    src = src.strip()
    if src.startswith("http://") or src.startswith("https://"):
        req = urllib.request.Request(src, headers={"User-Agent": "Qwen3.8-API-Gateway/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return decode_base64_image(src)
