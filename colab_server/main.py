import os
import secrets
from . import state
from .common import RUNTIME_SECRET
from .routes import attach_macos_api_routes
from .ui import CUSTOM_CSS, create_ui


def main():
    # /v1 网关必须始终有密钥：未配置时启动即自动生成随机密钥（见下方输出），避免空密钥裸奔
    if not (state.API_AUTH_KEY or "").strip():
        state.API_AUTH_KEY = secrets.token_urlsafe(18)
    ui = create_ui()
    ui.queue()
    app, local_url, share_url = ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inline=False,
        prevent_thread_lock=True,
        css=CUSTOM_CSS,
    )
    attach_macos_api_routes(app)
    # 供同机进程（桥接的 stop_all 云盘兜底同步）拼出带密钥的本地 API 路径
    try:
        with open("/content/.easycolab_runtime_secret", "w") as _sf:
            _sf.write(RUNTIME_SECRET)
        os.chmod("/content/.easycolab_runtime_secret", 0o600)
    except Exception:
        pass
    full_api_url = ((share_url or local_url) or "").rstrip("/") + "/k/" + RUNTIME_SECRET
    print("\n" + "=" * 72)
    print("🍎 【EasyColab 服务地址】请复制下方完整链接（含 /k/ 密钥段）粘贴到客户端：")
    print(f"👉  {full_api_url}")
    print(f"🔑 OpenAI / Anthropic 协议 API Key: {state.API_AUTH_KEY}")
    print("（客户端与 Claude Code / Cursor 等工具均需使用此地址与密钥；裸域名仅开放网页界面）")
    print("=" * 72 + "\n")
    ui.block_thread()
