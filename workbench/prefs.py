"""工作台偏好：目前只有首选显卡。存在用户数据目录，不进仓库。"""
import json
import threading

from . import catalog
from .keys import write_private
from .paths import PREFS_FILE

_lock = threading.Lock()


def get() -> dict:
    """文件不存在、读不出或显卡已不在目录里时，退回默认值"""
    gpu = catalog.PREFER_GPU_DEFAULT
    with _lock:
        try:
            v = json.loads(PREFS_FILE.read_text(encoding="utf-8")).get("gpu")
        except (OSError, ValueError, AttributeError):
            v = None
    if isinstance(v, str) and v in {g["id"] for g in catalog.GPUS}:
        gpu = v
    return {"gpu": gpu}


def set_gpu(gpu: str) -> dict:
    """gpu 必须在目录里，调用方先校验"""
    with _lock:
        write_private(PREFS_FILE, json.dumps({"gpu": gpu}))
    return {"gpu": gpu}
