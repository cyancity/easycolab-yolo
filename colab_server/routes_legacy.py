"""裸路径兼容层与兜底路由：必须最后注册：catch-all 先注册会吞掉之后注册的路由。"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def attach(app: FastAPI):
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
