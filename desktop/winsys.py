"""与系统打交道的部分：单实例、独立窗口、开机自启、提示框。

只有 Windows 做完整实现；其他系统退回浏览器打开和标准错误输出。
"""
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

APP = "ColabFleet"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
WIN = sys.platform == "win32"
_mutex = None


def claim_single_instance() -> bool:
    """拿到全局互斥量返回 True；已有一个客户端在运行返回 False"""
    global _mutex
    if not WIN:
        return True
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _mutex = k32.CreateMutexW(None, False, "Local\\ColabFleetDesktop")
    return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS


def _app_browser():
    """能开独立应用窗口（--app）的浏览器：优先 Edge（Windows 自带），其次 Chrome"""
    if not WIN:
        return None
    import winreg
    for exe in ("msedge.exe", "chrome.exe"):
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}") as k:
                    path = winreg.QueryValue(k, None)
                if path and Path(path).exists():
                    return path
            except OSError:
                pass
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base and (Path(base) / "Microsoft/Edge/Application/msedge.exe").exists():
            return str(Path(base) / "Microsoft/Edge/Application/msedge.exe")
    return None


def open_window(url: str):
    exe = _app_browser()
    if exe:
        try:
            subprocess.Popen([exe, f"--app={url}"], close_fds=True)
            return
        except OSError:
            pass
    webbrowser.open(url)


def _autostart_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --tray'
    pyw = Path(sys.executable).with_name("pythonw.exe")
    entry = Path(__file__).resolve().parents[1] / "packaging" / "windows" / "entry.py"
    return f'"{pyw if pyw.exists() else sys.executable}" "{entry}" --tray'


def autostart_enabled() -> bool:
    if not WIN:
        return False
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, APP)
        return True
    except OSError:
        return False


def set_autostart(on: bool):
    """与安装包写的是同一个值（HKCU Run 下的 ColabFleet），托盘开关和安装选项互相认得"""
    if not WIN:
        return
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if on:
            winreg.SetValueEx(k, APP, 0, winreg.REG_SZ, _autostart_command())
        else:
            try:
                winreg.DeleteValue(k, APP)
            except FileNotFoundError:
                pass


def _box(text: str, flags: int) -> int:
    import ctypes
    return ctypes.windll.user32.MessageBoxW(None, text, "Colab Fleet", flags | 0x40000)  # MB_TOPMOST


def alert(text: str):
    if WIN:
        _box(text, 0x10)  # MB_ICONERROR
    else:
        print(text, file=sys.stderr)


def confirm(text: str) -> bool:
    if not WIN:
        return True
    return _box(text, 0x4 | 0x30) == 6  # MB_YESNO | MB_ICONWARNING → IDYES
