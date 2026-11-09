"""本机固定 API 网关：/v1/* 按请求里的 model 转发到对应实例，外部 agent 配置不随实例变化。"""
import json
import time
from typing import Dict, List, Optional
from urllib.parse import quote

import anyio
import httpx
import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import catalog, deployments
from .keys import gateway_key

router = APIRouter()
_models_cache: Dict[str, tuple] = {}  # dep_id -> (time, models)


def instance_models(d: dict, max_age: float = 30) -> List[dict]:
    hit = _models_cache.get(d["id"])
    if hit and time.time() - hit[0] < max_age:
        return hit[1]
    try:
        r = requests.get(d["url"].rstrip("/") + "/v1/models", timeout=10,
                         headers={"Authorization": "Bearer " + gateway_key()})
        r.raise_for_status()
        models = r.json().get("data", [])
    except Exception:
        models = hit[1] if hit else []
    _models_cache[d["id"]] = (time.time(), models)
    return models


def _running() -> List[dict]:
    return [d for d in deployments.all_deps() if d["status"] == "running" and d.get("url")]


def route(model: str) -> Optional[dict]:
    running = _running()
    for d in running:
        if any(m.get("id") == model for m in instance_models(d)):
            return d
    # 名字对不上时，只有一个语言模型实例就直接用它
    llm = [d for d in running if catalog.BY_ID[d["model"]]["kind"] == "语言"]
    return llm[0] if len(llm) == 1 else None


def _authorized(req: Request) -> bool:
    k = gateway_key()
    auth = req.headers.get("authorization", "")
    return auth == "Bearer " + k or req.headers.get("x-api-key") == k


def _deny():
    return JSONResponse({"error": {"message": "API Key 不对，请用工作台里显示的网关密钥", "type": "auth"}}, status_code=401)


def _all_models() -> List[dict]:
    data, seen = [], set()
    for d in _running():
        for m in instance_models(d):
            if m.get("loaded", True) and m.get("id") not in seen:
                seen.add(m["id"])
                data.append({"id": m["id"], "object": "model", "owned_by": catalog.BY_ID[d["model"]]["name"]})
    return data


def _available_names() -> List[str]:
    return sorted({m["id"] for x in _running() for m in instance_models(x)})


def _bad_gateway(msg: str):
    return JSONResponse({"error": {"message": msg, "type": "upstream"}}, status_code=502)


async def _relay(method: str, url: str, headers: dict, body: Optional[bytes] = None):
    """把请求转给实例，响应流式回传；连不上返回 502"""
    client = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=20))
    try:
        upstream = await client.send(client.build_request(method, url, content=body, headers=headers), stream=True)
    except httpx.HTTPError as e:
        await client.aclose()
        return _bad_gateway(f"实例连接失败：{type(e).__name__}")
    except BaseException:
        await client.aclose()
        raise

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        except httpx.HTTPError:
            pass  # 上游中途断开：结束流，调用方自会发现不完整
        finally:
            # 客户端断开会取消本协程，清理必须屏蔽取消才能真正关掉上游
            with anyio.CancelScope(shield=True):
                await upstream.aclose()
                await client.aclose()

    return StreamingResponse(relay(), status_code=upstream.status_code,
                             media_type=upstream.headers.get("content-type", "application/json"))


@router.get("/v1/models")
async def list_models(request: Request):
    if not _authorized(request):
        return _deny()
    # requests 是阻塞调用，放线程里跑，实例失联时才不会卡住事件循环
    return {"object": "list", "data": await anyio.to_thread.run_sync(_all_models)}


# ---- MiniMax-H3 视频：实例自带接口不在 /v1 下，这里换成固定地址转发 ----
def _h3_instance() -> Optional[dict]:
    return next((d for d in _running() if d["model"] == "h3"), None)


def _no_h3():
    return JSONResponse({"error": {"message": "没有运行中的 MiniMax-H3 实例，请先在工作台部署"}}, status_code=404)


async def _h3_forward(method: str, path: str, request: Request, body: Optional[bytes] = None):
    if not _authorized(request):
        return _deny()
    d = await anyio.to_thread.run_sync(_h3_instance)
    if not d:
        return _no_h3()
    headers = {"Accept-Encoding": "identity"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    return await _relay(method, d["url"].rstrip("/") + "/minimax_api/" + path, headers, body)


@router.post("/v1/h3/generate")
async def h3_generate(request: Request):
    body = await request.body()
    try:
        parsed = json.loads(body or b"{}")
        if isinstance(parsed, dict) and not parsed.get("model_id"):
            body = json.dumps({**parsed, "model_id": "minimax_h3"}).encode()  # 实例可能同时驻留别的模型，显式指定
    except ValueError:
        pass  # 不是 JSON 就原样转，由实例返回 422
    return await _h3_forward("POST", "generate", request, body)


@router.get("/v1/h3/jobs/{job_id}")
async def h3_job(job_id: str, request: Request):
    return await _h3_forward("GET", "job/" + quote(job_id, safe=""), request)


@router.get("/v1/h3/media/{name}")
async def h3_media(name: str, request: Request):
    if name in (".", "..") or "\\" in name:
        return JSONResponse({"error": {"message": "文件名不合法"}}, status_code=400)
    return await _h3_forward("GET", "media/" + quote(name, safe=""), request)


@router.post("/v1/{path:path}")
async def forward(path: str, request: Request):
    if not _authorized(request):
        return _deny()
    body = await request.body()
    try:
        parsed = json.loads(body or b"{}")
        model = str(parsed.get("model") or "") if isinstance(parsed, dict) else ""
    except ValueError:
        model = ""
    d = await anyio.to_thread.run_sync(route, model)
    if not d:
        names = await anyio.to_thread.run_sync(_available_names)
        return JSONResponse({"error": {"message": f"没有运行中的实例提供模型 {model!r}，可用：{names}"}}, status_code=404)
    # identity：不让上游压缩，原样转发的字节才能被调用方直接解析
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + gateway_key(),
               "x-api-key": gateway_key(), "Accept-Encoding": "identity"}
    for h in ("anthropic-version", "anthropic-beta", "accept"):
        if h in request.headers:
            headers[h] = request.headers[h]
    return await _relay("POST", d["url"].rstrip("/") + "/v1/" + path, headers, body)
