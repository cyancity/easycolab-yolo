import time
import threading
import urllib.parse
import urllib.request
from . import state
from .common import GPU_LOCK
from .config import MODEL_DOWNLOAD_STATE, COMFYUI_STATE


def _ping_local_colab_jupyter(force: bool = False):
    """
    在活跃推理/下载期间定期轻触本地 Colab Jupyter Server (127.0.0.1:8080/api/kernels)，
    避免无网页模式下推理流量全走 gradio.live 导致 Colab 误判容器闲置而在生成第 2 条视频中途回收实例。
    """
    now = time.time()
    if not force and (now - state._LAST_JUPYTER_HEARTBEAT_TS) < 40.0:
        return
    state._LAST_JUPYTER_HEARTBEAT_TS = now

    def _do_ping():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for ep in ("http://127.0.0.1:8080/api/kernels", "http://127.0.0.1:8080/api/status"):
            try:
                req = urllib.request.Request(ep, headers={"X-Colab-Client-Agent": "colab-cli"})
                with opener.open(req, timeout=2.0) as resp:
                    resp.read(256)
            except Exception:
                pass

    threading.Thread(target=_do_ping, daemon=True).start()


def touch_activity():
    state.LAST_ACTIVE_TIME = time.time()
    _ping_local_colab_jupyter(force=False)


def estimate_cu_burn_rate(gpu_name: str) -> float:
    """根据 Colab GPU 型号估算每小时消耗的计算单元 (CU/hr)"""
    name_up = (gpu_name or "").upper()
    if "H100" in name_up:
        return 11.78
    elif "PRO 6000" in name_up or "BLACKWELL" in name_up or "G4" in name_up:
        return 10.05
    elif "A100" in name_up:
        return 5.37
    elif "L4" in name_up:
        return 1.71
    elif "T4" in name_up:
        return 1.17
    elif "V100" in name_up:
        return 3.80
    return 1.50


def _idle_watchdog_loop():
    """借鉴 comfy-colab-kit 的 stopper.py：当超过设定闲置时间无任务时自动释放 Colab 虚拟机"""
    while True:
        time.sleep(15)
        # 若正在下载、加载模型、生成视频/LLM推理或 ComfyUI 正在安装/执行节点队列，持续刷新活跃时间并向本地 Jupyter 发送保活脉冲
        if (
            MODEL_DOWNLOAD_STATE["is_downloading"]
            or MODEL_DOWNLOAD_STATE["is_loading_vram"]
            or GPU_LOCK.locked()
            or COMFYUI_STATE["is_starting"]
            or COMFYUI_STATE["is_downloading_weight"]
            or COMFYUI_STATE["queue_running"] > 0
            or COMFYUI_STATE["queue_pending"] > 0
        ):
            touch_activity()
            continue

        if state.IDLE_TIMEOUT_MINUTES <= 0:
            _ping_local_colab_jupyter(force=False)
            continue

        idle_sec = time.time() - state.LAST_ACTIVE_TIME
        if idle_sec >= state.IDLE_TIMEOUT_MINUTES * 60:
            print(f"🛑 [自动省额度看门狗] 已连续闲置 {state.IDLE_TIMEOUT_MINUTES} 分钟，正在自动释放 Colab 虚拟机...")
            disconnect_colab_runtime()
            break
        else:
            # 尚在用户设定的活跃保活窗口内，轻触本地 Jupyter 防止 Colab 提前强杀
            _ping_local_colab_jupyter(force=False)


threading.Thread(target=_idle_watchdog_loop, daemon=True).start()


def _link_forward():
    # 前向引用：定义在后面的模块，函数体运行时才用；入口在全部模块加载完后统一回填，保证严格按切片顺序加载
    global disconnect_colab_runtime
    from .comfy_packs import disconnect_colab_runtime
