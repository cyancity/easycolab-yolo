from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import routes_admin, routes_openai, routes_anthropic, routes_jobs, routes_comfy, routes_legacy


def attach_macos_api_routes(app: FastAPI):
    """在 Gradio 底层的 FastAPI 实例上挂载原生 macOS 客户端接口 + OpenAI / Anthropic 双协议 API 网关

    路由分组在 routes_*.py 里，按原来的注册顺序依次挂载；routes_legacy 的 catch-all 必须最后。
    """
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

    routes_admin.attach(app)
    routes_openai.attach(app)
    routes_anthropic.attach(app)
    routes_jobs.attach(app)
    routes_comfy.attach(app)
    routes_legacy.attach(app)
