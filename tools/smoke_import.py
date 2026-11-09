#!/usr/bin/env python3
"""导入冒烟：在假的重型依赖下，分别导入原单文件和拆分后的包，对比导入时的副作用序列。

用法：python tools/smoke_import.py --mono <原 app.py> --root <含 app.py 和 colab_server/ 的目录>

桩（sys.modules 里的假模块）只替换这些东西：torch / torchaudio / gradio / transformers / diffusers /
av / librosa / soundfile / hf_transfer 等重型库（本机多半没装；装了也不加载）。桩接受任意属性和调用，所以
它们藏起了：这些库自己的 import 报错、版本不兼容、模块顶层对它们 API 的真实调用结果（装饰器会被桩吃掉）。
fastapi / pydantic / starlette / PIL / requests 本机有就用真的，没有才用桩。
被中和的副作用（只记录不执行）：线程启动、atexit 注册、subprocess（含 pip install、pkill）、os.makedirs。
不能证明：gradio 界面、路由、模型加载在真环境里能跑——那些只在函数体里，导入时不执行。
"""
import argparse
import difflib
import json
import os
import subprocess
import sys

HEAVY = ("torch", "torchaudio", "gradio", "transformers", "diffusers", "av", "librosa", "soundfile",
         "hf_transfer", "accelerate", "bitsandbytes", "diffsynth", "safetensors", "numpy", "cv2", "imageio")
OPTIONAL = ("fastapi", "pydantic", "starlette", "PIL", "requests", "huggingface_hub", "uvicorn", "psutil")
ENV_KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "HF_HUB_ENABLE_HF_TRANSFER", "DIFFSYNTH_DOWNLOAD_SOURCE")


def child(kind: str, path: str, root: str):
    import atexit
    import importlib.abc
    import importlib.machinery
    import runpy
    import tempfile
    import threading
    import types
    from unittest.mock import MagicMock

    events = []
    loads = []

    class Stub(types.ModuleType):
        __path__ = []

        def __getattr__(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            v = MagicMock(name=f"{self.__name__}.{n}")
            setattr(self, n, v)
            return v

    class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def __init__(self, names):
            self.names = names

        def find_spec(self, name, p=None, target=None):
            if name.split(".")[0] in self.names:
                return importlib.machinery.ModuleSpec(name, self, is_package=True)

        def create_module(self, spec):
            m = Stub(spec.name)
            if spec.name == "pydantic":
                m.BaseModel = type("BaseModel", (), {})
            events.append(["stub-import", spec.name] + [k + "=" + ("set" if os.environ.get(k) else "unset") for k in ENV_KEYS])
            return m

        def exec_module(self, module):
            pass

    class Rec(importlib.abc.MetaPathFinder):
        def find_spec(self, name, p=None, target=None):
            if name.startswith("colab_server.") and name not in loads:
                loads.append(name)

    sys.meta_path.insert(0, Rec())
    sys.meta_path.insert(0, Finder(HEAVY))
    sys.meta_path.append(Finder(OPTIONAL))
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)

    def norm(x):
        return str(x).replace(tmp, "<cwd>").replace("\\", "/")

    def fn_name(f):
        return getattr(f, "__qualname__", None) or repr(f)

    def fake_thread_start(self):
        events.append(["thread", fn_name(self._target), "daemon=" + str(self.daemon)])

    threading.Thread.start = fake_thread_start
    atexit.register = lambda f, *a, **k: events.append(["atexit", fn_name(f)]) or f
    for fname in ("run", "call", "check_call", "check_output", "Popen"):
        setattr(subprocess, fname, lambda cmd, *a, _f=fname, **k: events.append(
            ["subprocess." + _f, norm(" ".join(cmd) if isinstance(cmd, (list, tuple)) else cmd)[:90]]) or MagicMock(returncode=0))
    os.makedirs = lambda p, *a, **k: events.append(["makedirs", norm(os.path.abspath(p))])
    env0 = dict(os.environ)
    out, err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, "w")
    if kind == "pkg":
        sys.path.insert(0, root)
    ns = runpy.run_path(path, run_name="app_smoke")
    sys.stdout, sys.stderr = out, err
    names = set(ns)
    if kind == "pkg":
        for mod in [m for m in sys.modules if m.startswith("colab_server")]:
            names |= set(vars(sys.modules[mod]))
    events.append(["env-diff"] + sorted(k for k in os.environ if os.environ[k] != env0.get(k)))
    res = {"events": events, "names": sorted(n for n in names if not n.startswith("__")), "loads": loads}
    sys.__stdout__.write("\n@@RESULT@@" + json.dumps(res))


def run_child(kind, path, root):
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", kind, path, root],
                       capture_output=True, text=True, encoding="utf-8")
    if "@@RESULT@@" not in p.stdout:
        sys.exit(f"[{kind}] 导入失败:\n{p.stdout[-1500:]}\n{p.stderr[-3000:]}")
    return json.loads(p.stdout.split("@@RESULT@@", 1)[1])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(*sys.argv[2:5])
    ap = argparse.ArgumentParser()
    ap.add_argument("--mono", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--order", default="", help="期望的模块顺序（逗号分隔）；默认读 root/app.py 的导入列表")
    a = ap.parse_args()
    mono = run_child("mono", os.path.abspath(a.mono), "")
    pkg = run_child("pkg", os.path.join(os.path.abspath(a.root), "app.py"), os.path.abspath(a.root))
    ok = True
    print(f"副作用事件：原单文件 {len(mono['events'])} 条，拆分后 {len(pkg['events'])} 条")
    if mono["events"] != pkg["events"]:
        ok = False
        print("副作用序列不一致：")
        fmt = lambda evs: [" | ".join(e) for e in evs]
        for line in difflib.unified_diff(fmt(mono["events"]), fmt(pkg["events"]), "mono", "pkg", lineterm="", n=1):
            print("  " + line)
    else:
        print("副作用序列（线程/atexit/subprocess/makedirs/桩导入时机/环境变量）与原单文件完全一致")
    missing = sorted(set(mono["names"]) - set(pkg["names"]))
    if missing:
        ok = False
        print("原顶层名字在包里找不到：", missing)
    else:
        print(f"原单文件 {len(mono['names'])} 个顶层名字，包里全部能找到")
    order = [x for x in a.order.split(",") if x] or None
    if order is None:
        import re
        with open(os.path.join(a.root, "app.py"), encoding="utf-8") as f:
            order = re.findall(r"^    (\w+),$", f.read(), re.M)
    # 子模块（routes_*、qwen_optimize_* 等）由父模块导入，只比较入口列出的顶层模块的先后
    got = [m for m in (x.split(".", 1)[1] for x in pkg["loads"]) if m in order]
    if got != order:
        print("警告：模块开始加载的顺序和切片顺序不同（底部前向 import 提前加载了后面的模块）：")
        print("  期望", order)
        print("  实际", got)
    else:
        print("模块加载顺序与切片顺序一致")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
