#!/usr/bin/env python3
"""静态检查拆分结果：python tools/check_split.py --mono <原 app.py> --root <含 colab_server/ 的目录>

1. 原单文件的每个顶层名字在包里恰好定义一次（可变全局只在 state.py）；
2. 每个模块里每个自由名字都能解析（内置、import、本模块定义、state.X）；
3. 没有模块顶层语句使用了后面才定义的名字（含装饰器、默认值、类体、顶层调用）；
4. 每个顶层语句的 AST 与原文件一致（把可变全局换成 state.X、去掉对应 global、if __name__ 换成 def main 之后），
   所以搬运中没有丢语句、没有改错位置；
5. 每个模块的行数（超过 500 行标出）。
"""
import argparse
import ast
import builtins
import bisect
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_app import MODULE_MAP  # noqa: E402
from split_model import Model, build_stmts, is_main_guard  # noqa: E402
from split_scope import analyze  # noqa: E402

DUNDER = {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__builtins__"}


def load_pkg(root: str):
    d = os.path.join(root, "colab_server")
    mods = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".py") and fn != "__init__.py":
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                text = f.read()
            mods[fn[:-3]] = (text, ast.parse(text))
    return mods


def module_binds(tree):
    """每个顶层语句绑定的名字（含 import 和 if/for 里的模块级赋值）。"""
    stmts = build_stmts(tree)
    an = analyze(tree)
    per = [set(s.binds) | {n for n, _ in s.imports} for s in stmts]
    starts = [s.start for s in stmts]
    for r in an.refs:
        if not r.in_func and isinstance(r.node.ctx, (ast.Store, ast.Del)):
            per[bisect.bisect_right(starts, r.node.lineno) - 1].add(r.name)
    return stmts, an, per, starts


def is_link_fn(n) -> bool:
    return isinstance(n, ast.FunctionDef) and n.name == "_link_forward"


def link_names(tree) -> set:
    """--forward link 模式下 _link_forward() 里 global 声明并回填的名字。"""
    return {nm for n in tree.body if is_link_fn(n) for g in ast.walk(n) if isinstance(g, ast.Global) for nm in g.names}


def expected_ast(mono: Model):
    deps = mono.collect()
    ids = {id(n) for n in deps.state_edits}
    state = mono.state

    class T(ast.NodeTransformer):
        def visit_Name(self, node):
            if id(node) in ids:
                return ast.copy_location(ast.Attribute(ast.Name("state", ast.Load()), node.id, node.ctx), node)
            return node

        def visit_Global(self, node):
            kept = [n for n in node.names if n not in state]
            if not kept:
                return None
            node.names = kept
            return node

    out = []
    for st in mono.stmts:
        if isinstance(st.node, (ast.Import, ast.ImportFrom)):
            continue
        n = st.node
        if is_main_guard(n):
            fn = ast.parse("def main():\n    pass").body[0]
            fn.body = n.body
            n = fn
        out.append(ast.dump(T().visit(n)))
    return Counter(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mono", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    with open(a.mono, encoding="utf-8") as f:
        mono = Model(f.read(), MODULE_MAP)
    pkg = load_pkg(a.root)
    errors = []

    # 1. 名字只定义一次
    where = {}
    info = {}
    for name, (text, tree) in pkg.items():
        info[name] = module_binds(tree)
        stmts, _, per, _ = info[name]
        for st, names in zip(stmts, per):
            if st.imports or is_link_fn(st.node):
                continue
            for nm in names:
                where.setdefault(nm, []).append(name)
    expected = set(mono.defs) | {"main"}
    for nm in sorted(expected):
        w = where.get(nm, [])
        if len(w) != 1:
            errors.append(f"名字 {nm} 在包里定义了 {len(w)} 次：{w}")
        elif nm in mono.state and w != ["state"]:
            errors.append(f"可变全局 {nm} 不在 state.py：{w}")
    for nm in sorted(set(mono.imps)):
        if not any(nm in per_i for _, _, per, _ in info.values() for per_i in per):
            errors.append(f"import 名字 {nm} 在包里没有任何绑定")
    for nm, w in where.items():
        if nm not in expected and nm not in mono.main_locals and nm != "state":
            errors.append(f"包里多出原文件没有的顶层名字 {nm}：{w}")

    # 2/3. 名字解析 + 顶层不用前向名字
    state_names = {n for n, w in where.items() if "state" in w}
    for name, (text, tree) in pkg.items():
        stmts, an, per, starts = info[name]
        bound_at = {}
        for i, names in enumerate(per + [link_names(tree)]):  # 回填的名字等价于绑定在模块最后
            for nm in names:
                bound_at.setdefault(nm, []).append(i)
        for r in an.refs:
            if isinstance(r.node.ctx, (ast.Store, ast.Del)) and not r.in_func:
                continue
            nm = r.name
            if nm in DUNDER or hasattr(builtins, nm):
                continue
            if nm not in bound_at:
                errors.append(f"{name}.py 第 {r.node.lineno} 行：{nm} 没有定义也没有 import")
                continue
            if not r.lazy:
                i = bisect.bisect_right(starts, r.node.lineno) - 1
                if min(bound_at[nm]) >= i:
                    errors.append(f"{name}.py 第 {r.node.lineno} 行：顶层语句使用了后面才绑定的 {nm}")
        for n in ast.walk(tree):
            if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "state"
                    and n.attr not in state_names and "state" in bound_at):
                errors.append(f"{name}.py 第 {n.lineno} 行：state.{n.attr} 在 state.py 里不存在")
        for g in an.global_decls:
            for nm in g.node.names:
                if nm in mono.state:
                    errors.append(f"{name}.py 第 {g.node.lineno} 行：global {nm} 应该已经删掉")

    # 4. 语句级 AST 等价
    got = Counter()
    for name, (text, tree) in pkg.items():
        for i, n in enumerate(tree.body):
            if isinstance(n, (ast.Import, ast.ImportFrom)) or is_link_fn(n):
                continue
            if i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and name == "state":
                continue
            got[ast.dump(n)] += 1
    want = expected_ast(mono)
    for k, v in (want - got).items():
        errors.append(f"原文件有、包里没有（或改动了）的语句 ×{v}：{k[:110]}...")
    for k, v in (got - want).items():
        errors.append(f"包里有、原文件没有的语句 ×{v}：{k[:110]}...")

    # 5. 行数
    print(f"{'module':<18}{'lines':>6}")
    over = []
    for name, (text, _) in pkg.items():
        n = text.count("\n")
        print(f"{name:<18}{n:>6}{'  >500!' if n > 500 else ''}")
        if n > 500:
            over.append(f"{name}.py {n} 行")
    print("超过 500 行：", over or "无")
    print(f"顶层语句：原文件 {sum(want.values())} 条（不含 import），包里 {sum(got.values())} 条")
    if errors:
        print(f"\n发现 {len(errors)} 个问题：")
        for e in errors[:60]:
            print("  -", e)
        sys.exit(1)
    print("静态检查通过：名字唯一、自由名字都可解析、顶层无前向引用、语句 AST 与原文件一致")


if __name__ == "__main__":
    main()
