#!/usr/bin/env python3
"""
EasyColab Windows Native Desktop Client — Main Entry Point.

Supports two execution modes:
  1. Standard GUI Studio mode: `EasyColab.exe` or `python windows_app/easycolab_win.py`
  2. Headless Bridge CLI mode (used internally by packaged EasyColab.exe):
     `EasyColab.exe --bridge-cli <auth_status|auth_url|auth_code|auth_logout|wake|keepalive|stop_all> ...`
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_gui_stdio() -> None:
    """In Windows SUBSYSTEM:WINDOWS GUI mode, stdout/stderr are None unless piped."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _setup_import_paths() -> None:
    internal_path: Path | None = None

    # 1. Check explicit env var if present
    env_internal = os.environ.get("EASYCOLAB_INTERNAL_DIR", "").strip()
    if env_internal and Path(env_internal).is_dir():
        internal_path = Path(env_internal).resolve()

    # 2. Direct filesystem layout detection (immune to CRT/_pth environment isolation)
    if internal_path is None:
        script_dir = Path(__file__).resolve().parent
        if (script_dir / "runtime").is_dir() and (script_dir / "site-packages").is_dir():
            internal_path = script_dir
        else:
            exe_dir = Path(sys.executable).resolve().parent
            if (exe_dir / "_internal" / "runtime").is_dir():
                internal_path = (exe_dir / "_internal").resolve()

    if internal_path is not None:
        if not getattr(sys, "frozen", False):
            sys.frozen = True  # type: ignore[attr-defined]
        if not hasattr(sys, "_MEIPASS"):
            sys._MEIPASS = str(internal_path)  # type: ignore[attr-defined]
        if hasattr(os, "add_dll_directory"):
            for dll_dir in (
                internal_path / "runtime",
                internal_path / "site-packages" / "PySide6",
                internal_path / "site-packages" / "shiboken6",
            ):
                if dll_dir.is_dir():
                    try:
                        os.add_dll_directory(str(dll_dir))
                    except Exception:
                        pass

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        for p in (
            exe_dir,
            exe_dir / "_internal",
            exe_dir / "_internal" / "site-packages",
            internal_path,
            (internal_path / "site-packages") if internal_path else None,
            getattr(sys, "_MEIPASS", None),
        ):
            if p and Path(p).exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
    else:
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))


_ensure_gui_stdio()
_setup_import_paths()

# Fast-path headless bridge mode (skips Qt initialization completely)
if len(sys.argv) >= 2 and sys.argv[1] == "--bridge-cli":
    import colab_bridge

    colab_bridge.run_cli(sys.argv[2:])


def _apply_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("app.easycolab.studio.windows")
    except Exception:
        pass


def _sync_windows_titlebar_dark_mode(hwnd: int, dark: bool) -> None:
    """Enable native Windows 10/11 immersive dark/light titlebar via DWM."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import ctypes.wintypes  # noqa: F401  # 不导入则 ctypes.wintypes.HWND 抛 AttributeError，暗色标题栏静默失效

        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        val = ctypes.c_int(1 if dark else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.wintypes.HWND(hwnd),
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(val),
            ctypes.sizeof(val),
        )
    except Exception:
        pass


def main() -> int:
    _apply_windows_app_id()

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("EasyColab")
    app.setOrganizationName("EasyColab")

    from windows_app.main_window import EasyColabMainWindow
    from windows_app.theme import build_app_icon
    from windows_app.viewmodel import EasyColabViewModel

    app.setWindowIcon(build_app_icon())

    vm = EasyColabViewModel()
    win = EasyColabMainWindow(vm)
    win.show()

    _sync_windows_titlebar_dark_mode(int(win.winId()), vm.dark_mode)
    vm.theme_changed.connect(lambda: _sync_windows_titlebar_dark_mode(int(win.winId()), vm.dark_mode))

    vm.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
