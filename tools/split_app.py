#!/usr/bin/env python3
"""把单文件 app.py 机械拆成 colab_server/ 包 + 一个薄入口 app.py。

用法：python tools/split_app.py --src <单文件 app.py> --out <输出根目录>
输出：<out>/app.py（入口）、<out>/colab_server/*.py。同一个输入永远产生同一个输出。

模块是 app.py 的连续切片，边界由下面 MODULE_MAP 里“第一个顶层名字”决定，不依赖行号。
改写只动名字位置（按 AST 列偏移），其余源码（注释、字符串、格式）原样搬运：
  1. 被 global 重绑定或顶层多次赋值的名字搬进 state.py，引用改成 state.NAME；
  2. 其它跨模块引用：定义在前面的模块 -> 顶部 from .mod import；定义在后面的 -> 底部 import；
  3. 原顶层 import 留在原位置，其它模块按需带自己的副本；
  4. if __name__ == "__main__": 块变成 colab_server/main.py 的 main()。
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_model import Model, SplitError, alias_text  # noqa: E402

PKG = "colab_server"

# (模块名, 该模块第一条顶层语句)：第一项可以是顶层名字，也可以是语句首行的前缀
MODULE_MAP = [
    ("bootstrap", None),                                 # 标准库 import、终端镜像、依赖自检、环境变量
    ("kernels", "_ensure_torchaudio_compatible"),        # torchaudio / triton / 快速内核补丁
    ("qwen_optimize", "_optimize_qwen35_for_fast_inference"),
    ("common", "from fastapi import FastAPI"),           # fastapi import、全局锁与计数、GGUF 清理、流式器
    ("config", "MODEL_DOWNLOAD_STATE"),                  # 内存盘、模型根目录、MODEL_CATALOG、输出目录
    ("hf_download", "_StripAuthRedirectHandler"),
    ("idle", "_LAST_JUPYTER_HEARTBEAT_TS"),              # 闲置回收、算力计费
    ("model_files", "_get_model_dirs"),
    ("telemetry", "_TELEMETRY_LOCK"),
    ("weights", "build_structured_prompt"),              # 权重下载与落盘
    ("gguf_server", "_make_noproxy_opener"),
    ("flash_next_glue", "FLASH_NEXT_ID"),
    ("vram", "_sync_legacy_globals"),
    ("loader", "load_model_to_vram_core"),
    ("pipeline_opt", "_enable_high_vram_optimizations"),
    ("video", "RESOLUTION_MAP"),
    ("comfy_presets", "COMFY_STANDARD_MODEL_SUBDIRS"),
    ("comfy_workflows", "_prune_workflow_dangling_links"),
    ("comfy_runtime", "_resolve_comfyui_persistence_paths"),
    ("comfy_sync", "sync_comfyui_to_drive_core"),
    ("comfy_start", "start_comfyui_core"),
    ("comfy_packs", "install_comfyui_preset_pack_core"),
    ("schemas", "SelectModelRequest"),
    ("llm_jobs", "decode_base64_image"),
    ("jobs", "from contextlib import contextmanager"),
    ("api_auth", "ApiKeyConfigRequest"),
    ("qwen_stream", "_extract_qwen_tool_calls"),
    ("qwen_prepare", "_prepare_gguf_streamer_generation"),
    ("routes", "attach_macos_api_routes"),               # 984 行单函数，之后手工拆
    ("ui", "PRESET_EXAMPLES"),
    ("main", "if __name__"),
]

STATE_DOC = '"""可被重绑定的模块级全局（原 app.py 里靠 `global` 改写或顶层多次赋值的名字）。\n\n其它模块一律用 `state.NAME` 读写。就地修改的 dict/list/锁不在这里。\n"""'
WIDTH = 100


def char_col(line: str, byte_col: int) -> int:
    return len(line.encode("utf-8")[:byte_col].decode("utf-8"))


def fmt_from(mod: str, names) -> str:
    one = f"from {mod} import {', '.join(names)}"
    if len(one) <= WIDTH:
        return one
    return f"from {mod} import (\n" + "".join(f"    {n},\n" for n in names) + ")"


class Emitter:
    def __init__(self, model: Model, link: bool = False):
        self.m = model
        self.link = link
        self.deps = model.collect()
        if self.deps.eager_forward:
            raise SplitError("模块顶层使用了后面模块的名字：\n  " + "\n  ".join(self.deps.eager_forward))
        self._plan_edits()

    # --- 行级改写计划 ---
    def _plan_edits(self):
        m = self.m
        self.removed = set()
        for i in m.state_inits:
            st = m.stmts[i]
            self.removed.update(range(m.attached_start(st), st.end + 1))
        self.line_repl = {}  # 行号 -> 新文本；None 表示删行
        self.cols = {}  # 行号 -> [(c0, c1, 新文本)]
        for n in self.deps.state_edits:
            ln = m.lines[n.lineno - 1]
            if re.search(r"\{\s*" + re.escape(n.id) + r"\s*=\s*[}!:]", ln):
                raise SplitError(f"第 {n.lineno} 行：f-string 的 {{{n.id}=}} 改成 state.X 后输出会变，需人工处理")
            self.cols.setdefault(n.lineno, []).append(
                (char_col(ln, n.col_offset), char_col(ln, n.end_col_offset), "state." + n.id))
        for g in m.an.global_decls:
            names = g.node.names
            kept = [x for x in names if x not in m.state]
            if len(kept) == len(names):
                continue
            ln = m.lines[g.node.lineno - 1]
            mt = re.match(r"^(\s*)global\s+[\w\s,]+?(\s*#.*)?$", ln)
            if not mt or g.node.end_lineno != g.node.lineno:
                raise SplitError(f"第 {g.node.lineno} 行 global 语句格式不支持：{ln!r}")
            self.line_repl[g.node.lineno] = (mt.group(1) + "global " + ", ".join(kept) + (mt.group(2) or "")) if kept else None
        if m.main_idx is not None:
            self.line_repl[m.stmts[m.main_idx].start] = "def main():"

    def _line(self, no: int):
        if no in self.line_repl:
            return self.line_repl[no]
        s = self.m.lines[no - 1]
        for c0, c1, txt in sorted(self.cols.get(no, []), reverse=True):
            s = s[:c0] + txt + s[c1:]
        return s

    def _body(self, lo: int, hi: int):
        out, gap = [], False
        for no in range(lo, hi + 1):
            if no in self.removed or (no in self.line_repl and self.line_repl[no] is None):
                gap = True
                continue
            s = self._line(no)
            if gap and s.strip() == "" and out and out[-1].strip() == "":
                continue
            gap = False
            out.append(s)
        while out and out[0].strip() == "":
            out.pop(0)
        while out and out[-1].strip() == "":
            out.pop()
        return out

    # --- 头尾 import ---
    def _imp_lines(self, binds):
        out = []
        for si, k in sorted(binds):
            t = alias_text(self.m.stmts[si].node, k)
            if t not in out:
                out.append(t)
        return self._merge_from(out)

    @staticmethod
    def _merge_from(texts):
        """同一个来源的 from-import 合并成一行，保持首次出现的顺序。"""
        order, names = [], {}
        for t in texts:
            mt = re.match(r"^from (\S+) import (.+)$", t)
            if not mt:
                order.append(t)
                continue
            if mt.group(1) not in names:
                names[mt.group(1)] = []
                order.append(("from", mt.group(1)))
            names[mt.group(1)].append(mt.group(2))
        return [fmt_from(o[1], names[o[1]]) if isinstance(o, tuple) else o for o in order]

    def _mod_imports(self, table):
        out = []
        for src in sorted(table, key=lambda s: self.m.order[s]):
            names = sorted(table[src], key=lambda n: self.m.defs[n][0])
            out.append(fmt_from("." + src, names))
        return out

    def render(self, mod: str) -> str:
        m, d = self.m, self.deps
        if mod == "state":
            body = []
            for i in sorted(m.state_inits):
                st = m.stmts[i]
                body += [""] + [m.lines[n - 1] for n in range(m.attached_start(st), st.end + 1)]
            head = [STATE_DOC] + self._imp_lines(d.imp_back.get("state", set()))
            return "\n".join(head + body) + "\n"
        k = m.mod_names.index(mod)
        lo, hi = m.line_range(k)
        body = self._body(lo, hi)
        head = self._imp_lines(d.imp_back.get(mod, set()))
        if mod in d.uses_state:
            head.append("from . import state")
        head += self._mod_imports(d.back.get(mod, {}))
        tail = self._imp_lines(d.imp_fwd.get(mod, set())) + self._mod_imports(d.fwd.get(mod, {}))
        out = head + (["", ""] if head else []) + body
        if tail and not self.link:
            out += ["", "", "# 前向引用：定义在后面的模块，只在函数体里用，放在底部避免循环导入"] + tail
        elif tail:
            bound = [nm for si, kk in sorted(d.imp_fwd.get(mod, set())) for nm, k2 in m.stmts[si].imports if k2 == kk]
            for src in sorted(d.fwd.get(mod, {}), key=lambda s: m.order[s]):
                bound += sorted(d.fwd[mod][src], key=lambda n: m.defs[n][0])
            out += ["", "", "def _link_forward():",
                    "    # 前向引用：定义在后面的模块，函数体运行时才用；入口在全部模块加载完后统一回填，保证严格按切片顺序加载",
                    "    global " + ", ".join(dict.fromkeys(bound))]
            out += ["    " + ln for t in tail for ln in t.split("\n")]
        return "\n".join(out) + "\n"

    def first_name(self, mod: str) -> str:
        if mod == "state":
            return "(mutable globals)"
        a, _ = self.m.mod_range(self.m.mod_names.index(mod))
        st = self.m.stmts[a]
        if st.binds:
            return st.binds[0]
        return self.m.lines[st.start - 1].strip()[:50]


def entry_source(mods, link: bool = False) -> str:
    names = ["state"] + [x for x in mods]
    body = "".join(f"    {n},\n" for n in names)
    linker = ""
    if link:
        linker = "\nfor _m in (" + ", ".join(names) + "):\n    getattr(_m, \"_link_forward\", lambda: None)()\n"
    return (
        '"""EasyColab Colab 服务端入口。\n\n逻辑都在 colab_server/ 包里；这里按原单文件的执行顺序导入各模块，再启动。\n'
        '模块顺序即 app.py 原来的语句顺序，导入时的副作用（线程、atexit、环境变量）依赖它，不要调换。\n"""\n'
        f"from colab_server import (  # noqa: F401\n{body})\n"
        f"{linker}from colab_server.main import main\n\n"
        'if __name__ == "__main__":\n    main()\n'
    )


def split(src_text: str, out_dir: str, quiet: bool = False, link: bool = False):
    m = Model(src_text, MODULE_MAP)
    em = Emitter(m, link)
    pkg_dir = os.path.join(out_dir, PKG)
    os.makedirs(pkg_dir, exist_ok=True)
    files = {"__init__.py": '"""EasyColab Colab 服务端包，由 tools/split_app.py 从单文件 app.py 拆出。"""\n'}
    for mod in ["state"] + m.mod_names:
        files[mod + ".py"] = em.render(mod)
    for name, text in files.items():
        with open(os.path.join(pkg_dir, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    with open(os.path.join(out_dir, "app.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(entry_source(m.mod_names, link))
    stale = sorted(set(os.listdir(pkg_dir)) - set(files) - {"__pycache__"})
    rows = [(mod, files[mod + ".py"].count("\n"), em.first_name(mod)) for mod in ["state"] + m.mod_names]
    if not quiet:
        print(f"{'module':<18}{'lines':>6}  first top-level name")
        for name, n, first in rows:
            print(f"{name:<18}{n:>6}{'  >500!' if n > 500 else ''}  {first}")
        print(f"{'TOTAL':<18}{sum(r[1] for r in rows):>6}")
        print("mutable globals ->", ", ".join(sorted(m.state)))
        for s in stale:
            print("warning: 包里有本脚本不生成的文件（未删除）：", s)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="单文件 app.py")
    ap.add_argument("--out", required=True, help="输出根目录（写 app.py 和 colab_server/）")
    ap.add_argument("--forward", choices=("bottom", "link"), default="bottom",
                    help="前向引用的处理：bottom=模块底部 import（默认）；link=_link_forward() 由入口回填，严格按切片顺序加载")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    with open(a.src, encoding="utf-8", newline="") as f:
        text = f.read()
    if "\r\n" in text:
        text = text.replace("\r\n", "\n")
    if "from colab_server import" in text and "def create_ui" not in text:
        sys.exit("--src 看起来已经是拆分后的入口，请传原单文件")
    try:
        split(text, a.out, a.quiet, a.forward == "link")
    except SplitError as e:
        sys.exit(f"split_app: {e}")


if __name__ == "__main__":
    main()
