"""split_app.py / check_split.py 共用的作用域分析：找出每个 Name 是局部还是模块全局。

不用 symtable：它的子表和 AST 节点对不上号，这里自己走一遍 AST，规则和 Python 一致
（global/nonlocal、类作用域对方法不可见、推导式独立作用域、海象运算符绑定到外层函数）。
"""
import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


@dataclass
class Ref:
    """一次对模块全局名的引用。lazy=True 表示出现在函数体内，导入时不会执行。"""
    name: str
    node: ast.Name
    lazy: bool
    in_func: bool  # 声明在函数里（用于判断是否真的“重绑定”了全局）


@dataclass
class GlobalDecl:
    node: ast.Global
    func: ast.AST


class Scope:
    def __init__(self, kind: str, parent: Optional["Scope"], node: ast.AST):
        self.kind, self.parent, self.node = kind, parent, node
        self.bound: Set[str] = set()
        self.globals_decl: Set[str] = set()
        self.nonlocals: Set[str] = set()


def _arg_names(a: ast.arguments) -> List[str]:
    out = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    for x in (a.vararg, a.kwarg):
        if x:
            out.append(x.arg)
    return out


def _collect(scope: Scope, body_nodes: List[ast.AST]):
    """收集本作用域绑定的名字，不进入嵌套的函数/类/lambda 体。"""

    def walk(n: ast.AST, in_comp: bool):
        if isinstance(n, ast.Global):
            scope.globals_decl.update(n.names)
        elif isinstance(n, ast.Nonlocal):
            scope.nonlocals.update(n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope.bound.add(n.name)
            for d in n.decorator_list:
                walk(d, in_comp)
            if not isinstance(n, ast.ClassDef):
                _walk_defaults(n.args)
            return
        elif isinstance(n, ast.Lambda):
            _walk_defaults(n.args)
            return
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                scope.bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            if not in_comp:
                scope.bound.add(n.id)
        elif isinstance(n, ast.NamedExpr):
            scope.bound.add(n.target.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            scope.bound.add(n.name)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            scope.bound.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            scope.bound.add(n.rest)
        comp = in_comp or isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
        for c in ast.iter_child_nodes(n):
            walk(c, comp)

    def _walk_defaults(a: ast.arguments):
        for d in a.defaults + [x for x in a.kw_defaults if x]:
            walk(d, False)

    for n in body_nodes:
        walk(n, False)


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.scope = Scope("module", None, None)
        self.refs: List[Ref] = []
        self.global_decls: List[GlobalDecl] = []
        # 在函数里被重绑定（Store/Del/AugAssign）的全局名
        self.rebound_in_func: Dict[str, List[ast.Name]] = {}

    # --- 作用域解析 ---
    def _resolve(self, name: str) -> str:
        s = self.scope
        if name in s.globals_decl or s.kind == "module":
            return "global"
        if name in s.bound or name in s.nonlocals:
            return "local"
        p = s.parent
        while p is not None:
            if p.kind == "class":
                p = p.parent
                continue
            if p.kind == "module":
                return "global"
            if name in p.globals_decl:
                return "global"
            if name in p.bound or name in p.nonlocals:
                return "local"
            p = p.parent
        return "global"

    def _in_func(self) -> bool:
        s = self.scope
        while s is not None:
            if s.kind in ("func", "lambda"):
                return True
            s = s.parent
        return False

    def visit_Name(self, node: ast.Name):
        if self._resolve(node.id) == "global":
            in_func = self._in_func()
            self.refs.append(Ref(node.id, node, in_func, in_func))
            if in_func and isinstance(node.ctx, (ast.Store, ast.Del)):
                self.rebound_in_func.setdefault(node.id, []).append(node)

    def visit_Global(self, node: ast.Global):
        func = self.scope.node
        self.global_decls.append(GlobalDecl(node, func))

    # --- 嵌套作用域 ---
    def _enter(self, kind: str, node: ast.AST, body: List[ast.AST], params: List[str] = ()):
        sc = Scope(kind, self.scope, node)
        sc.bound.update(params)
        _collect(sc, body)
        self.scope = sc
        for n in body:
            self.visit(n)
        self.scope = sc.parent

    def _visit_args_outer(self, a: ast.arguments):
        for d in a.defaults + [x for x in a.kw_defaults if x]:
            self.visit(d)
        for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
            if x is not None and x.annotation is not None:
                self.visit(x.annotation)

    def _visit_func(self, node):
        for d in node.decorator_list:
            self.visit(d)
        self._visit_args_outer(node.args)
        if node.returns:
            self.visit(node.returns)
        self._enter("func", node, node.body, _arg_names(node.args))

    visit_FunctionDef = visit_AsyncFunctionDef = _visit_func

    def visit_Lambda(self, node: ast.Lambda):
        self._visit_args_outer(node.args)
        self._enter("lambda", node, [node.body], _arg_names(node.args))

    def visit_ClassDef(self, node: ast.ClassDef):
        for d in node.decorator_list:
            self.visit(d)
        for b in node.bases:
            self.visit(b)
        for k in node.keywords:
            self.visit(k.value)
        self._enter("class", node, node.body)

    def _visit_comp(self, node):
        gens = node.generators
        self.visit(gens[0].iter)  # 第一个可迭代对象在外层作用域求值
        sc = Scope("comp", self.scope, node)
        for g in gens:
            _collect_targets(sc, g.target)
        for n in ast.walk(node):
            if isinstance(n, ast.NamedExpr):  # 绑定到外层，不是推导式局部
                pass
        self.scope = sc
        for i, g in enumerate(gens):
            if i:
                self.visit(g.iter)
            self.visit(g.target)
            for c in g.ifs:
                self.visit(c)
        for attr in ("elt", "key", "value"):
            if hasattr(node, attr):
                self.visit(getattr(node, attr))
        self.scope = sc.parent

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _visit_comp


def _collect_targets(sc: Scope, t: ast.AST):
    for n in ast.walk(t):
        if isinstance(n, ast.Name):
            sc.bound.add(n.id)


def analyze(tree: ast.Module) -> Analyzer:
    an = Analyzer()
    _collect(an.scope, tree.body)
    for n in tree.body:
        an.visit(n)
    return an
