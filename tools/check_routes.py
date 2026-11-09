#!/usr/bin/env python3
"""路由表等价检查：拆分前后的 attach_macos_api_routes(app) 注册出的路由必须逐项一致。

用法：python tools/check_routes.py [--root <含 app.py 和 colab_server/ 的目录>] [--ref HEAD]

做法：两个子进程，都在 smoke_import.py 同款的桩环境里（重型库用假模块，线程/atexit/subprocess/makedirs 只记录
不执行）导入包，然后建一个真 FastAPI 实例调用 attach_macos_api_routes：
  orig：把 colab_server.routes 换成 `git show <ref>:colab_server/routes.py` 的原文（拆分前的单函数版本）
  new ：用当前工作区的 colab_server/routes.py（及其 routes_*.py）
比较有序列表 (路由类型, 路径, 排序后的方法, endpoint 名, endpoint 参数签名)。顺序也必须一致：
FastAPI 取第一个匹配，catch-all 与裸路径 403 必须排在最后。
不能证明：路由处理函数体的行为（只比较注册结果，函数体靠逐字搬运保证）。
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_import import HEAVY, OPTIONAL  # noqa: E402


def child(kind: str, root: str, orig_src_path: str):
    import atexit
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import inspect
    import runpy
    import tempfile
    import threading
    import types
    from unittest.mock import MagicMock

    class Stub(types.ModuleType):
        __path__ = []

        def __getattr__(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            v = MagicMock(name=f"{self.__name__}.{n}")
            setattr(self, n, v)
            return v

    class StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def __init__(self, names):
            self.names = names

        def find_spec(self, name, p=None, target=None):
            if name.split(".")[0] in self.names:
                return importlib.machinery.ModuleSpec(name, self, is_package=True)

        def create_module(self, spec):
            m = Stub(spec.name)
            if spec.name == "pydantic":
                m.BaseModel = type("BaseModel", (), {})
            return m

        def exec_module(self, module):
            pass

    class OrigRoutes(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """colab_server.routes 换成拆分前的原文。"""

        def find_spec(self, name, p=None, target=None):
            if name == "colab_server.routes":
                return importlib.util.spec_from_loader(name, self)

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            with open(orig_src_path, encoding="utf-8") as f:
                code = compile(f.read(), "<routes.py@orig>", "exec")
            exec(code, module.__dict__)

    if kind == "orig":
        sys.meta_path.insert(0, OrigRoutes())
    sys.meta_path.insert(0, StubFinder(HEAVY))
    sys.meta_path.append(StubFinder(OPTIONAL))
    os.chdir(tempfile.mkdtemp())
    threading.Thread.start = lambda self: None
    atexit.register = lambda f, *a, **k: f
    for fname in ("run", "call", "check_call", "check_output", "Popen"):
        setattr(subprocess, fname, lambda *a, **k: MagicMock(returncode=0))
    os.makedirs = lambda *a, **k: None
    sys.path.insert(0, root)
    out, err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, "w")
    runpy.run_path(os.path.join(root, "app.py"), run_name="app_check_routes")
    from fastapi import FastAPI
    from colab_server.common import RUNTIME_SECRET
    from colab_server.routes import attach_macos_api_routes

    app = FastAPI()
    before = len(app.routes)
    attach_macos_api_routes(app)
    sys.stdout, sys.stderr = out, err
    table = []
    for r in app.routes[before:]:
        ep = getattr(r, "endpoint", None)
        table.append([
            type(r).__name__,
            (getattr(r, "path", None) or "").replace(RUNTIME_SECRET, "<SECRET>"),  # 每个进程的密钥随机
            sorted(getattr(r, "methods", None) or []),
            getattr(ep, "__name__", None),
            str(inspect.signature(ep)) if ep else None,
            inspect.iscoroutinefunction(ep) if ep else None,
        ])
    sys.__stdout__.write("\n@@RESULT@@" + json.dumps({"table": table, "n_middleware": len(app.user_middleware)}))


def run_child(kind, root, orig_src_path):
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", kind, root, orig_src_path],
                       capture_output=True, text=True, encoding="utf-8")
    if "@@RESULT@@" not in p.stdout:
        sys.exit(f"[{kind}] 失败:\n{p.stdout[-1500:]}\n{p.stderr[-3000:]}")
    return json.loads(p.stdout.split("@@RESULT@@", 1)[1])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(*sys.argv[2:5])
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--ref", default="HEAD", help="拆分前 routes.py 所在的 git 版本")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    import tempfile
    src = subprocess.run(["git", "show", f"{a.ref}:colab_server/routes.py"], cwd=root, capture_output=True, check=True).stdout
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        orig = run_child("orig", root, f.name)
        new = run_child("new", root, f.name)
    finally:
        os.unlink(f.name)
    print(f"原版 {len(orig['table'])} 条路由，拆分后 {len(new['table'])} 条；中间件 {orig['n_middleware']} / {new['n_middleware']}")
    ok = orig == new
    if not ok:
        import difflib
        fmt = lambda t: [" | ".join(map(str, r)) for r in t["table"]]
        for line in difflib.unified_diff(fmt(orig), fmt(new), "orig", "new", lineterm="", n=2):
            print("  " + line)
        if orig["n_middleware"] != new["n_middleware"]:
            print("  中间件数量不同")
        print("路由表不一致")
    else:
        print("路由表逐项一致（类型、路径、方法、endpoint 名、签名、是否 async、顺序）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
