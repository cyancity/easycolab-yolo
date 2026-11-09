# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for EasyColab Windows Desktop Client (EasyColab.exe).
Bundles the Qt6 Studio UI, QtMultimedia MP4 audio/video plugins, colab_bridge.py,
google-colab-cli OAuth config, app.py, and EasyColab_Server.ipynb into a standalone
distribution requiring ZERO Python installation on user machines.
"""
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

SPEC_DIR = Path(SPECPATH).resolve()
PROJECT_ROOT = SPEC_DIR.parent
ICON_PATH = SPEC_DIR / "assets" / "AppIcon.ico"

datas = []
for res_name in ("app.py", "colab_bridge.py", "EasyColab_Server.ipynb", "requirements.txt"):
    src = PROJECT_ROOT / res_name
    if src.exists():
        datas.append((str(src), "."))

# colab_bridge.py 的实现包，随脚本一起放进 _internal/bridge
for py_file in sorted((PROJECT_ROOT / "bridge").glob("*.py")):
    datas.append((str(py_file), "bridge"))

if ICON_PATH.exists():
    datas.append((str(ICON_PATH), "assets"))
png_path = SPEC_DIR / "assets" / "AppIcon.png"
if png_path.exists():
    datas.append((str(png_path), "assets"))

# Include colab_cli oauth_config.json and package metadata
datas += collect_data_files("colab_cli")
datas += collect_data_files("certifi")
for pkg in ("google-colab-cli", "google-auth", "google-auth-oauthlib", "requests"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

hiddenimports = [
    "colab_bridge",
    "google.auth.transport.requests",
    "google.oauth2.credentials",
    "google_auth_oauthlib.flow",
    "requests_oauthlib",
    "websocket",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
]
hiddenimports += collect_submodules("colab_cli")
try:
    hiddenimports += collect_submodules("jupyter_kernel_client")
except Exception:
    pass

excludes = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQml",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "tkinter",
    "matplotlib",
    "scipy",
    "torch",
]

a = Analysis(
    [str(SPEC_DIR / "easycolab_win.py")],
    pathex=[str(PROJECT_ROOT), str(SPEC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EasyColab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(ICON_PATH) if ICON_PATH.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="EasyColab",
)
