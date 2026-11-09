"""命令行入口：参数解析与分发。"""
import os
import sys
import argparse

from .auth import (
    cmd_accounts_list,
    cmd_accounts_remove,
    cmd_accounts_switch,
    cmd_auth_code,
    cmd_auth_logout,
    cmd_auth_url,
)
from .sessions import cmd_auth_status, cmd_keepalive, cmd_stop_all
from .wake import cmd_wake

# 仓库根（colab_bridge.py 与 app.py 所在目录）：本文件在 bridge/ 下，上两级
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth_status")
    sub.add_parser("auth_url")
    sub.add_parser("auth_logout")

    p_code = sub.add_parser("auth_code")
    p_code.add_argument("code", type=str)

    sub.add_parser("accounts_list")
    p_switch = sub.add_parser("accounts_switch")
    p_switch.add_argument("email", type=str)
    p_remove = sub.add_parser("accounts_remove")
    p_remove.add_argument("email", type=str)

    p_wake = sub.add_parser("wake")
    p_wake.add_argument("--gpu", type=str, default="G4")
    p_wake.add_argument("--session", type=str, default="")
    p_wake.add_argument(
        "--app-py",
        type=str,
        default=os.path.join(ROOT_DIR, "app.py"),
    )

    sub.add_parser("keepalive")
    sub.add_parser("stop_all")

    args = parser.parse_args(argv)
    if args.cmd == "auth_status":
        cmd_auth_status()
    elif args.cmd == "auth_url":
        cmd_auth_url()
    elif args.cmd == "auth_logout":
        cmd_auth_logout()
    elif args.cmd == "auth_code":
        cmd_auth_code(args.code)
    elif args.cmd == "accounts_list":
        cmd_accounts_list()
    elif args.cmd == "accounts_switch":
        cmd_accounts_switch(args.email)
    elif args.cmd == "accounts_remove":
        cmd_accounts_remove(args.email)
    elif args.cmd == "wake":
        cmd_wake(args.gpu, args.app_py, args.session)
    elif args.cmd == "keepalive":
        cmd_keepalive()
    elif args.cmd == "stop_all":
        cmd_stop_all()


def run_cli(argv=None):
    """供独立 CLI 或打包后的 Windows exe (--bridge-cli) 子进程模式统一调用"""
    try:
        main(argv)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
