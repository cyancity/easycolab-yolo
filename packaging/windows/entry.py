"""PyInstaller 打包入口；开发时也可以直接 python packaging/windows/entry.py 运行"""
import sys
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from desktop.app import main  # noqa: E402

main()
