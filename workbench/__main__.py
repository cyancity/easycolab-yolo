"""启动工作台：python -m workbench [--no-browser]"""
import argparse
import sys
import threading
import webbrowser

import uvicorn

from .server import PORT, app, boot


def main():
    ap = argparse.ArgumentParser(prog="python -m workbench")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    boot()
    url = f"http://127.0.0.1:{PORT}/"
    print(f"EasyColab 工作台：{url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
