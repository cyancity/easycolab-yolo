"""与本机其他软件的对接：目前只有 Magpie（本地模型网关，注册了 magpie:// 链接）。"""
import os
import shutil
import subprocess
import sys
import time
import urllib.parse

from . import catalog

_CACHE_S = 60
_cache = {"at": 0.0, "value": False}


def _detect_magpie() -> bool:
    if sys.platform == "win32":
        import winreg
        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_CLASSES_ROOT):
            sub = r"Software\Classes\magpie" if root == winreg.HKEY_CURRENT_USER else "magpie"
            try:
                winreg.CloseKey(winreg.OpenKey(root, sub))
                return True
            except OSError:
                continue
        return False
    return shutil.which("magpie") is not None


def magpie_installed() -> bool:
    """是否装了 Magpie（能处理 magpie:// 链接）。结果缓存 60 秒，任何异常都当作没装"""
    now = time.monotonic()
    if now - _cache["at"] < _CACHE_S and _cache["at"]:
        return _cache["value"]
    try:
        found = _detect_magpie()
    except Exception:  # noqa: BLE001 探测失败不能影响 /api/state
        found = False
    _cache.update(at=now, value=found)
    return found


def language_models() -> list:
    """网关 /v1/models 对外提供的语言模型 id，按目录顺序"""
    return [m["app_id"] for m in catalog.MODELS if m.get("kind") == "语言" and m.get("app_id")]


def magpie_link(key: str, models: list) -> str:
    """Magpie 的供应商导入链接；Magpie 打开后会先弹确认框，用户同意才保存"""
    from .server import PORT  # 放在函数里：server 导入本模块，顶层导入会循环
    query = urllib.parse.urlencode({
        "name": "Colab Fleet", "id": "colab-fleet", "chat": f"http://127.0.0.1:{PORT}/v1",
        "key": key, "models": ",".join(models),
    }, quote_via=urllib.parse.quote)
    return f"magpie://import?{query}"


def open_link(link: str) -> None:
    """让系统用默认程序打开链接。链接里带完整密钥：不要记录异常对象或链接本身，报错只给固定文案"""
    try:
        if sys.platform == "win32":
            os.startfile(link)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", link], check=True, timeout=10)
        else:
            subprocess.run(["xdg-open", link], check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError("打开 Magpie 失败，请确认它已安装") from e
