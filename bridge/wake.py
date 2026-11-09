"""wake 命令：复用或分配 Colab GPU 实例，上传服务端代码，远端启动并回传 gradio.live 地址。"""
from colab_cli.common import state
from colab_cli.client import Variant, Shape
from colab_cli.state import SessionState
from colab_cli.contents import ContentsClient

from .allocation import _allocate_fresh_session, _check_gpu_eligible, _resolve_gpu_accel, _wait_for_runtime_ready
from .boot import run_boot_script
from .credentials import _has_valid_token
from .sessions import _safe_sync_sessions
from .upload import upload_server_files


def cmd_wake(gpu_type: str, app_py_path: str, session: str = ""):
    """
    无浏览器一键唤醒 Colab GPU 并自动部署 EasyColab 多模型服务端：
    1. 幂等检查：若已有运行中的会话且探活成功则直接复用，否则自动清理失效会话并分配新实例
    2. 等待容器 Jupyter 端点 (/api/kernels) TLS/HTTP 握手就绪，规避 SSLEOFError
    3. 上传本地最新 app.py 到 /content/app.py
    4. 在远端内核安装依赖并后台拉起 app.py，提取 https://xxxx.gradio.live 隧道链接返回给 Mac 客户端
    """
    if not _has_valid_token():
        print("WAKE_ERROR: 尚未授权登录 Google 账号，请先点击「🔑 绑定账号」", flush=True)
        return

    try:
        session_name = session or "easycolab"
        print("WAKE_LOG: 🔍 正在检查账号下已有的 Colab 实例...", flush=True)
        local_sessions, assignments = _safe_sync_sessions()
        if session:
            # 工作台按会话名精确管理同账号下的多个实例，不接管其他会话或云端孤儿实例
            local_sessions = {k: v for k, v in local_sessions.items() if k == session}
            assignments = []

        target_session = (
            local_sessions.get(session_name)
            or local_sessions.get("aevix")
            or local_sessions.get("minimax")
        )
        if target_session:
            session_name = target_session.name
        elif local_sessions:
            target_session = list(local_sessions.values())[0]
            session_name = target_session.name

        # 若本地未记录但云端已有活跃分配，直接纳管该云端实例
        if not target_session and assignments:
            a = assignments[0]
            target_session = SessionState(
                name=session_name,
                token=a.runtime_proxy_info.token,
                url=a.runtime_proxy_info.url,
                endpoint=a.endpoint,
                token_expires_at=a.runtime_proxy_info.expires_at(),
                variant=Variant.GPU.value,
                accelerator=a.accelerator.value,
                machine_shape=Shape.HIGH_RAM.name if a.machine_shape == Shape.HIGH_RAM else Shape.STANDARD.name,
            )
            state.store.add(target_session)

        requested_accel = _resolve_gpu_accel(gpu_type)
        # 先确认资格再动旧实例，避免"旧卡已释放、新卡又申请不到"
        _check_gpu_eligible(requested_accel)
        if target_session:
            cur_acc = (target_session.accelerator or "").strip().upper()
            if cur_acc != requested_accel.value.upper():
                print(
                    f"WAKE_LOG: 🔄 当前已分配实例型号为 [{cur_acc or 'CPU'}]，正在释放并切换为您选择的 [{requested_accel.value}]...",
                    flush=True,
                )
                try:
                    state.client.unassign(target_session.endpoint)
                except Exception:
                    pass
                state.prune_session(target_session.name)
                target_session = None
            else:
                print(f"WAKE_LOG: ♻️ 正在检查已有实例 [{target_session.name}] ({target_session.accelerator}) 连通性...", flush=True)
                if not _wait_for_runtime_ready(target_session, timeout_s=15):
                    print("WAKE_LOG: ⚠️ 旧实例已失效或正在销毁，正在自动清理并重新分配新实例...", flush=True)
                    try:
                        state.client.unassign(target_session.endpoint)
                    except Exception:
                        pass
                    state.prune_session(target_session.name)
                    target_session = None

        if not target_session:
            target_session = _allocate_fresh_session(session_name, gpu_type)
            if not _wait_for_runtime_ready(target_session, timeout_s=50):
                print("WAKE_LOG: ⏳ 容器仍在初始化，继续尝试连接内核...", flush=True)

        print("WAKE_LOG: 正在上传最新多模型开源服务端代码 (app.py)...", flush=True)
        contents = ContentsClient(target_session)
        upload_server_files(contents, app_py_path)

        print("WAKE_LOG: 正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)...", flush=True)
        outputs = run_boot_script(target_session)

        for out in outputs:
            if "text" in out:
                for line in out["text"].splitlines():
                    print(line, flush=True)
            elif out.get("output_type") == "error":
                print("WAKE_ERROR: " + out.get("evalue", "Execution error"), flush=True)

    except Exception as e:
        print(f"WAKE_ERROR: {str(e)}", flush=True)
