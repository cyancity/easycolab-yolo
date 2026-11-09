"""工作台的文件位置。仓库内路径按本文件推算，用户数据放系统应用数据目录。"""
import os
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
BRIDGE = REPO_ROOT / "colab_bridge.py"
APP_PY = REPO_ROOT / "app.py"
STATIC_DIR = PKG_DIR / "static"
MODEL_DOCS_DIR = PKG_DIR / "model_docs"

# 旧版桌面客户端的多账号库，首次启动时导入
LEGACY_ACCOUNTS = Path.home() / ".config" / "colab-cli" / "accounts.json"


def data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".config"
    d = base / "EasyColab" / "workbench"
    d.mkdir(parents=True, exist_ok=True)
    return d


DATA_DIR = data_dir()
ACCOUNTS_DIR = DATA_DIR / "accounts"
ACCOUNTS_DIR.mkdir(exist_ok=True)
DEPLOYMENTS_FILE = DATA_DIR / "deployments.json"
GATEWAY_FILE = DATA_DIR / "gateway.json"
