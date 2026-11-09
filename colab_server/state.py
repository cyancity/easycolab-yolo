"""可被重绑定的模块级全局（原 app.py 里靠 `global` 改写或顶层多次赋值的名字）。

其它模块一律用 `state.NAME` 读写。就地修改的 dict/list/锁不在这里。
"""
import time
from typing import Optional

CURRENT_STAGE_BAR_LABEL: Optional[str] = None

LAST_ACTIVE_TIME = time.time()

IDLE_TIMEOUT_MINUTES = 30  # 默认闲置 30 分钟自动释放 Colab 虚拟机，可在 Mac 端调节

PIPELINE = None

LLM_BUNDLE = None  # {"backend": "transformers"|"gguf", ...}

SELECTED_MODEL_ID = "minimax_h3"

LOADED_MODEL_ID = None

CURRENT_MODEL_MODE = None

QWEN_CONTEXT_LENGTH = 32768  # 用户可自定义上下文长度 (4K ~ 256K = 262144 tokens)

API_AUTH_KEY = ""

API_REQUEST_COUNT = 0

_LAST_JUPYTER_HEARTBEAT_TS: float = 0.0

_COMFY_BG_SYNC_STARTED = False

_COMFY_ADOPT_CHECKED = False
