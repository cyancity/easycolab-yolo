"""网关密钥：外部 agent 调本机网关、网关转发到实例，用同一把 key。"""
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

from .paths import GATEWAY_FILE

_key_lock = threading.Lock()


def write_private(path: Path, text: str):
    """原子写入含凭据的文件：先写临时文件再 os.replace，权限尽量收紧到 0600"""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        for i in range(4):  # Windows 上目标被杀软或读取方短暂占用时会拒绝替换
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if i == 3:
                    raise
                time.sleep(0.05)
    finally:
        tmp.unlink(missing_ok=True)


def gateway_key() -> str:
    """读网关 key；文件不存在才生成。文件存在但读不出 key 时报错，不静默换 key（会让所有已配置的客户端失效）"""
    with _key_lock:
        if GATEWAY_FILE.exists():
            try:
                k = json.loads(GATEWAY_FILE.read_text(encoding="utf-8")).get("key")
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"网关密钥文件无法读取，请检查或删除后重启：{GATEWAY_FILE.name}（{type(e).__name__}）") from e
            if not k or not isinstance(k, str):
                raise RuntimeError(f"网关密钥文件里没有有效的 key，请检查或删除后重启：{GATEWAY_FILE.name}")
            return k
        k = "sk-ec-" + secrets.token_hex(16)
        write_private(GATEWAY_FILE, json.dumps({"key": k}))
        return k


def mask_key(k: str) -> str:
    return k[:6] + "••••••" + k[-4:]


# sk- 前面不能紧跟字母数字，避免误伤 task-xxxx 之类的普通词
_SECRET = re.compile(r"/k/[A-Za-z0-9_-]+|(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{6,}|(API Key:\s*)\S+|([?&]token=)[^\s&\"']+")


def _mask_match(m) -> str:
    if m.group(1) or m.group(2):
        return (m.group(1) or m.group(2)) + "••••••"
    return "/k/••••••" if m.group(0).startswith("/k/") else "sk-••••••"


def redact(text: str) -> str:
    """打掉路径密钥、sk- 密钥和日志里的 API Key，用于事件、错误信息和日志"""
    return _SECRET.sub(_mask_match, str(text or ""))
