"""split_app.py 的数据层：把单文件拆成顶层语句，算出每个模块的依赖和需要改写的位置。"""
import ast
import bisect
import builtins
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from split_scope import analyze

BANNER = re.compile(r"#\s*[=\-]{5,}")


class SplitError(Exception):
    pass


@dataclass
class Stmt:
    idx: int
    node: ast.stmt
    start: int  # 含装饰器
    end: int
    chunk_start: int  # 含前面的空行和注释
    binds: List[str] = field(default_factory=list)  # def/class/赋值绑定的名字
    imports: List[Tuple[str, int]] = field(default_factory=list)  # (绑定名, alias 序号)


def target_names(t: ast.AST) -> List[str]:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        return [n for e in t.elts for n in target_names(e)]
    if isinstance(t, ast.Starred):
        return target_names(t.value)
    return []


def is_main_guard(n: ast.stmt) -> bool:
    return (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__")


def build_stmts(tree: ast.Module) -> List[Stmt]:
    out, prev_end = [], 0
    for i, n in enumerate(tree.body):
        s = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        if s <= prev_end:
            raise SplitError(f"第 {s} 行有多条顶层语句同行，无法按行切片")
        st = Stmt(i, n, s, n.end_lineno, prev_end + 1)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            st.binds = [n.name]
        elif isinstance(n, ast.Assign):
            st.binds = [x for t in n.targets for x in target_names(t)]
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            st.binds = target_names(n.target)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for k, al in enumerate(n.names):
                if al.name == "*":
                    raise SplitError(f"第 {n.lineno} 行 import *，无法分析")
                st.imports.append(((al.asname or al.name).split(".")[0], k))
        out.append(st)
        prev_end = n.end_lineno
    return out


def alias_text(n: ast.stmt, k: int) -> str:
    al = n.names[k]
    nm = al.name + (f" as {al.asname}" if al.asname else "")
    if isinstance(n, ast.Import):
        return f"import {nm}"
    return f"from {'.' * n.level}{n.module or ''} import {nm}"


class Model:
    """parse + 全局名分析。mods: [(模块名, 起始语句序号)]，state 之外按顺序。"""

    def __init__(self, src: str, module_map: List[Tuple[str, Optional[str]]]):
        self.lines = src.split("\n")
        if self.lines and self.lines[-1] == "":
            self.lines.pop()
        self.tree = ast.parse(src)
        self.stmts = build_stmts(self.tree)
        self.an = analyze(self.tree)
        self._starts = [s.start for s in self.stmts]
        self.main_idx = next((s.idx for s in self.stmts if is_main_guard(s.node)), None)
        self._bind_tables()
        self._find_state()
        self._assign_modules(module_map)

    def stmt_of(self, lineno: int) -> int:
        return bisect.bisect_right(self._starts, lineno) - 1

    # --- 绑定表 ---
    def _bind_tables(self):
        self.defs: Dict[str, List[int]] = {}
        self.imps: Dict[str, List[Tuple[int, int]]] = {}
        for st in self.stmts:
            for nm in st.binds:
                self.defs.setdefault(nm, []).append(st.idx)
            for nm, k in st.imports:
                self.imps.setdefault(nm, []).append((st.idx, k))
        self.main_bound: Dict[str, int] = {}  # main 块里赋值的名字
        for r in self.an.refs:
            if (self.main_idx is not None and not r.in_func and isinstance(r.node.ctx, ast.Store)
                    and self.stmt_of(r.node.lineno) == self.main_idx):
                self.main_bound.setdefault(r.name, r.node.lineno)
        for nm in set(self.defs) & set(self.imps):
            raise SplitError(f"{nm} 同时被 import 和赋值/定义绑定")

    def _find_state(self):
        st: Set[str] = set(self.an.rebound_in_func)
        st |= {n for n, v in self.defs.items() if len(v) > 1}
        for nm in self.main_bound:
            outside = any(r.name == nm and self.stmt_of(r.node.lineno) != self.main_idx for r in self.an.refs)
            if nm in self.defs or outside:
                st.add(nm)
        for n in ast.walk(self.tree):  # 这两种绑定写不成 state.X
            if (isinstance(n, ast.ExceptHandler) and n.name in st) or                     (isinstance(n, ast.NamedExpr) and n.target.id in st):
                raise SplitError(f"第 {n.lineno} 行：可变全局被 except-as / := 绑定，需人工处理")
        self.state: Set[str] = st
        self.main_locals = {n for n in self.main_bound if n not in st}
        self.state_inits: Dict[int, str] = {}  # 语句序号 -> 名字（搬进 state.py 的初始化语句）
        for nm in sorted(st):
            if nm in self.imps:
                raise SplitError(f"可变全局 {nm} 是 import 绑定，需人工处理")
            idxs = self.defs.get(nm, [])
            if not idxs:
                raise SplitError(f"可变全局 {nm} 在顶层没有初始值，需人工处理")
            for i in idxs:
                n = self.stmts[i].node
                simple = (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)) \
                    or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None)
                if i == self.main_idx or not simple:
                    if i != self.main_idx:
                        raise SplitError(f"可变全局 {nm} 的顶层语句（第 {n.lineno} 行）不是简单赋值")
                    continue
                self.state_inits[i] = nm

    # --- 模块切片 ---
    def find_first(self, first: str) -> int:
        if first in self.defs:
            return self.defs[first][0]
        if first in self.imps:
            return self.imps[first][0][0]
        for st in self.stmts:
            if self.lines[st.start - 1].strip().startswith(first):
                return st.idx
        raise SplitError(f"模块边界 {first!r} 在 app.py 里找不到")

    def _assign_modules(self, module_map):
        self.mod_names = [m for m, _ in module_map]
        bounds = []
        for k, (name, first) in enumerate(module_map):
            bounds.append(0 if k == 0 else self.find_first(first))
        if bounds != sorted(set(bounds)) or bounds[0] != 0:
            raise SplitError("模块边界必须严格递增（按 app.py 原顺序）：" + str(list(zip(self.mod_names, bounds))))
        self.bounds = bounds
        self.mod_of: List[str] = []
        for st in self.stmts:
            k = bisect.bisect_right(bounds, st.idx) - 1
            self.mod_of.append(self.mod_names[k])
        for i in self.state_inits:
            self.mod_of[i] = "state"
        self.order = {m: k for k, m in enumerate(["state"] + self.mod_names)}

    def mod_range(self, k: int) -> Tuple[int, int]:
        """第 k 个模块的语句区间 [a, b)。"""
        b = self.bounds[k + 1] if k + 1 < len(self.bounds) else len(self.stmts)
        return self.bounds[k], b

    def line_range(self, k: int) -> Tuple[int, int]:
        a, b = self.mod_range(k)
        hi = len(self.lines) if b >= len(self.stmts) else self.stmts[b - 1].end
        return self.stmts[a].chunk_start, hi

    def attached_start(self, st: Stmt) -> int:
        """搬走状态初始化语句时，紧贴在它上面的注释一起带走（横幅不带）。"""
        ln = st.start
        while ln - 1 >= st.chunk_start:
            t = self.lines[ln - 2].strip()
            if not t.startswith("#") or BANNER.match(t):
                break
            ln -= 1
        return ln

    # --- 依赖 ---
    def collect(self) -> "Deps":
        d = Deps(self.mod_names)
        for r in self.an.refs:
            i = self.stmt_of(r.node.lineno)
            in_main = i == self.main_idx
            if in_main and r.name in self.main_locals:
                continue
            if r.name == "__name__" and in_main:
                continue
            mod = self.mod_of[i]
            lazy = r.lazy or in_main
            if i in self.state_inits and r.name == self.state_inits[i]:
                continue
            if r.name in self.state:
                if i in self.state_inits:
                    raise SplitError(f"状态初始化语句（第 {r.node.lineno} 行）引用了可变全局 {r.name}")
                d.uses_state.add(mod)
                d.state_edits.append(r.node)
                continue
            if r.name in self.defs:
                j = self.mod_of[self.defs[r.name][0]]
                if i in self.state_inits and j != "state":
                    raise SplitError(f"状态初始化语句（第 {r.node.lineno} 行）引用了 {r.name}（{j}），需人工处理")
                d.add_def(mod, j, r.name, lazy, self.order, r.node.lineno)
            elif r.name in self.imps:
                d.add_imp(mod, r.name, lazy, self, r.node.lineno)
            elif not hasattr(builtins, r.name):
                raise SplitError(f"第 {r.node.lineno} 行：{r.name} 没有定义")
        return d


class Deps:
    def __init__(self, mod_names):
        self.uses_state: Set[str] = set()
        self.state_edits: List[ast.Name] = []
        self.back: Dict[str, Dict[str, Set[str]]] = {}  # 模块 -> 来源模块 -> 名字（顶部 import）
        self.fwd: Dict[str, Dict[str, Set[str]]] = {}  # 底部 import
        self.imp_back: Dict[str, Set[Tuple[int, int]]] = {}
        self.imp_fwd: Dict[str, Set[Tuple[int, int]]] = {}
        self.eager_forward: List[str] = []

    def add_def(self, mod, src, name, lazy, order, lineno):
        if src == mod:
            return
        if order[src] < order[mod]:
            self.back.setdefault(mod, {}).setdefault(src, set()).add(name)
        elif lazy:
            self.fwd.setdefault(mod, {}).setdefault(src, set()).add(name)
        else:
            self.eager_forward.append(f"{mod}（第 {lineno} 行）在导入时就用了后面模块 {src} 的 {name}")

    def add_imp(self, mod, name, lazy, model: "Model", lineno):
        binds = model.imps[name]
        mine = [b for b in binds if model.mod_of[b[0]] == mod]
        if mine:
            return
        if mod == "state":  # state.py 最先加载，只放简单初始值，import 一律自带
            self.imp_back.setdefault(mod, set()).update(binds)
            return
        before = {b for b in binds if model.order[model.mod_of[b[0]]] < model.order[mod]}
        if before:
            self.imp_back.setdefault(mod, set()).update(before)
        elif lazy:
            self.imp_fwd.setdefault(mod, set()).update(binds)
        else:
            self.eager_forward.append(f"{mod}（第 {lineno} 行）在导入时就用了后面才 import 的 {name}")
