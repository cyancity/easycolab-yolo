from . import state


import os
import io
import gc
import re
import sys
import uuid
import time
import glob
import queue
import atexit
import socket
import base64
import hmac
import secrets
import subprocess
import threading
from collections import deque
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

from starlette.concurrency import run_in_threadpool

# ==============================================================================
# 实时终端进度与 tqdm 捕获器 (将 Colab 网页端的实时下载进度同步给 macOS App)
# ==============================================================================
class TerminalProgressMirror:
    def __init__(self, orig_stream, buffer_deque: deque, active_bars: dict, lock: threading.Lock):
        self.orig_stream = orig_stream
        self.buffer_deque = buffer_deque
        self.active_bars = active_bars
        self.lock = lock
        self._tls = threading.local()
        self._ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def write(self, data: str):
        try:
            self.orig_stream.write(data)
        except Exception:
            pass
        if not data:
            return
        clean = self._ansi_re.sub("", data)
        partial = getattr(self._tls, "partial", "")
        with self.lock:
            for ch in clean:
                if ch in ("\r", "\n"):
                    line = partial.strip()
                    if line:
                        # 识别类似 "model-00001-of-00014.safetensors: 22%..." 或无前缀采样/VAE进度条 "54%|#####4 | 19/35 [08:31<07:11...]"
                        if "%" in line and ("it/s" in line or "B/s" in line or "step" in line or "|" in line):
                            pct_idx = line.find("%")
                            colon_idx = line.find(":")
                            if 0 < colon_idx < pct_idx:
                                key = line[:colon_idx].strip()
                                self.active_bars[key] = line
                            else:
                                stage_lbl = state.CURRENT_STAGE_BAR_LABEL
                                if stage_lbl:
                                    # 若在 VAE 解码等子阶段内部触发 tqdm，标注阶段名避免子阶段 100% 被误认为整条视频已生成完毕
                                    if line.startswith("100%"):
                                        self.active_bars["sampling"] = f"{stage_lbl}: 子阶段已完成，正在继续后续合成封装..."
                                    else:
                                        self.active_bars["sampling"] = f"{stage_lbl}: {line}"
                                else:
                                    self.active_bars["sampling"] = line
                        else:
                            self.buffer_deque.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
                    partial = ""
                else:
                    partial += ch
        self._tls.partial = partial

    def flush(self):
        try:
            self.orig_stream.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.orig_stream, "isatty", lambda: False)()


CONSOLE_LOCK = threading.Lock()
CONSOLE_LINES: deque = deque(maxlen=120)
ACTIVE_TQDM_BARS: Dict[str, str] = {}

if not isinstance(sys.stdout, TerminalProgressMirror):
    sys.stdout = TerminalProgressMirror(sys.__stdout__, CONSOLE_LINES, ACTIVE_TQDM_BARS, CONSOLE_LOCK)
if not isinstance(sys.stderr, TerminalProgressMirror):
    sys.stderr = TerminalProgressMirror(sys.__stderr__, CONSOLE_LINES, ACTIVE_TQDM_BARS, CONSOLE_LOCK)


def get_live_console_Lines() -> List[str]:
    with CONSOLE_LOCK:
        bars = list(ACTIVE_TQDM_BARS.values())[-10:]
        logs = list(CONSOLE_LINES)[-60:]
        return logs + (["─── 实时下载/采样进度流 ───"] + bars if bars else [])


# ==============================================================================
# 自动补齐音视频依赖 (防止出现 No module named 'av' 等错误)
# ==============================================================================
def ensure_runtime_dependencies():
    missing = []
    for pkg, mod in [
        ("av", "av"),
        ("hf_transfer", "hf_transfer"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"🔧 正在自动安装缺失的音视频依赖: {', '.join(missing)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("✅ 缺失依赖自动安装完成！")


ensure_runtime_dependencies()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# 强制 DiffSynth-Studio 所有内置下载走 HuggingFace 国际高速源 (彻底杜绝回退到国内 ModelScope 导致 1~2MB/s)
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
# 将 DiskMap 的句柄缓冲阈值扩大至 1TB，避免加载 33B 模型时频繁关闭/重开 safetensors 映射
os.environ["DIFFSYNTH_DISK_MAP_BUFFER_SIZE"] = str(10**12)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import gradio as gr
from PIL import Image
