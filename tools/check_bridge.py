#!/usr/bin/env python3
"""检查 colab_bridge.py 拆成 bridge/ 包后零行为变化：python tools/check_bridge.py [--rev 拆分前的提交]

静态：
1. 旧文件每个顶层 def/class/赋值在 bridge/ 里恰好定义一次，且 AST 与旧版一致（cmd_wake 因抽出 upload/boot 单独检查）；
2. 每个模块的自由名字都能解析（内置、本模块绑定、import）；
3. 旧文件里的每个字符串常量（含远端启动脚本）在新包里出现次数不少于旧版；
4. 每个源文件行数 <= 500。
动态（不联网）：新旧入口在假 HOME 下逐个命令对比 stdout / stderr / 退出码 / HOME 落盘文件。
"""
import argparse
import ast
import builtins
import os
import re
import subprocess
import sys
import symtable
import tempfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE_SPLIT_REV = "967d23f"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def bridge_modules():
    d = os.path.join(ROOT, "bridge")
    return {fn[:-3]: read(os.path.join(d, fn)) for fn in sorted(os.listdir(d)) if fn.endswith(".py")}


def top_defs(tree):
    """顶层 def / class / 赋值：名字 -> AST dump（import 不算）。"""
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out[n.name] = ast.dump(n)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                out[t.id] = ast.dump(n.value)
    return out


def string_counter(tree):
    return Counter(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))


def free_names(text, path):
    """symtable 求每个模块里解析不到的全局名字。"""
    top = symtable.symtable(text, path, "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    missing = set()

    def walk(t):
        for s in t.get_symbols():
            nm = s.get_name()
            if t is not top and s.is_referenced() and s.is_global() and nm not in defined and not hasattr(builtins, nm):
                missing.add(nm)
        for c in t.get_children():
            walk(c)

    for s in top.get_symbols():
        if s.is_referenced() and s.get_name() not in defined and not hasattr(builtins, s.get_name()):
            if s.get_name() not in ("__file__", "__name__"):
                missing.add(s.get_name())
    walk(top)
    return missing


def find_func(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def static_check(old_text, errors):
    old = ast.parse(old_text)
    mods = bridge_modules()
    trees = {k: ast.parse(v) for k, v in mods.items()}
    where, new_defs = {}, {}
    for m, t in trees.items():
        for nm, dump in top_defs(t).items():
            where.setdefault(nm, []).append(m)
            new_defs[nm] = dump
    old_defs = top_defs(old)
    for nm, dump in old_defs.items():
        if len(where.get(nm, [])) != 1:
            errors.append(f"名字 {nm} 在 bridge/ 里定义了 {len(where.get(nm, []))} 次：{where.get(nm)}")
        elif nm not in ("cmd_wake", "main") and new_defs[nm] != dump:
            errors.append(f"{nm} 的 AST 与旧版不一致")
    for nm, ms in where.items():
        if len(ms) > 1:
            errors.append(f"名字 {nm} 重复定义：{ms}")
    # main：只允许 --app-py 的默认值表达式不同（__file__ 所在目录 -> ROOT_DIR）
    def main_dump(fn):
        for n in ast.walk(fn):
            if isinstance(n, ast.keyword) and n.arg == "default" and isinstance(n.value, ast.Call):
                n.value = ast.Constant("<default>")
        return ast.dump(fn)

    if main_dump(find_func(trees["cli"], "main")) != main_dump(find_func(old, "main")):
        errors.append("main 的 AST 与旧版不一致")
    # cmd_wake：旧版 try 体里 upload 一段 / boot 一段抽走后，其余语句顺序与内容一致
    old_try = next(n for n in find_func(old, "cmd_wake").body if isinstance(n, ast.Try))
    new_try = next(n for n in find_func(trees["wake"], "cmd_wake").body if isinstance(n, ast.Try))
    up = find_func(trees["upload"], "upload_server_files").body
    old_body = [ast.dump(s) for s in old_try.body]
    new_body = [ast.dump(s) for s in new_try.body]
    n_up = len(up)
    i = next(k for k, s in enumerate(old_try.body) if isinstance(s, ast.For) and "up_attempt" in ast.dump(s))
    # 旧 upload 段从 For(up_attempt) 起，顶层语句数与 upload_server_files 相同（import tempfile 在 if 体内）
    old_up = old_try.body[i:i + n_up]
    for a, b in zip(old_up, up):
        da, db = ast.dump(a), ast.dump(b)
        da = re.sub(r"Import\(names=\[alias\(name='tempfile'\)\]\), ", "", da)
        if da != db:
            errors.append("cmd_wake 抽到 upload.py 的语句与旧版不一致")
            break
    head_old, head_new = old_body[:i], new_body[:i]
    if head_old != head_new:
        errors.append("cmd_wake 上传前的语句与旧版不一致")
    # 旧版 boot_code 常量与 BOOT_CODE 相同
    boot_old = next(n.value.value for n in ast.walk(find_func(old, "cmd_wake")) if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", "") == "boot_code")
    if trees["boot"].body and top_defs(trees["boot"]).get("BOOT_CODE") is None:
        errors.append("boot.py 缺 BOOT_CODE")
    boot_new = next(n.value.value for n in trees["boot"].body if isinstance(n, ast.Assign))
    if boot_old != boot_new:
        errors.append("BOOT_CODE 与旧版 boot_code 不一致")
    # 字符串多重集
    new_cnt = Counter()
    for t in list(trees.values()) + [ast.parse(read(os.path.join(ROOT, "colab_bridge.py")))]:
        new_cnt += string_counter(t)
    for s, c in (string_counter(old) - new_cnt).items():
        errors.append(f"字符串在新包里少了：{s[:60]!r} x{c}")
    # 自由名字、行数
    for m, text in mods.items():
        miss = free_names(text, m + ".py")
        if miss:
            errors.append(f"bridge/{m}.py 有解析不到的名字：{sorted(miss)}")
    for path in [os.path.join(ROOT, "colab_bridge.py")] + [os.path.join(ROOT, "bridge", f"{m}.py") for m in mods]:
        n = len(read(path).splitlines())
        print(f"  {os.path.relpath(path, ROOT)}: {n} 行")
        if n > 500:
            errors.append(f"{path} 超过 500 行")
    miss = free_names(read(os.path.join(ROOT, "colab_bridge.py")), "colab_bridge.py")
    if miss:
        errors.append(f"colab_bridge.py 有解析不到的名字：{sorted(miss)}")


ACCOUNTS_SEED = '{"version": 1, "accounts": [{"email": "a@b.c", "token_json": "not json", "updated_at": "x"}]}'
CASES = [  # (名称, 参数, HOME 里预置的 {相对路径: 内容})
    ("no_args", [], {}),
    ("help", ["--help"], {}),
    ("unknown_cmd", ["bogus"], {}),
    ("wake_help", ["wake", "--help"], {}),
    ("auth_code_missing_arg", ["auth_code"], {}),
    ("wake_bad_option", ["wake", "--nope"], {}),
    ("auth_status_no_token", ["auth_status"], {}),
    ("auth_status_bad_token", ["auth_status"], {".config/colab-cli/token.json": "garbage"}),
    ("auth_logout", ["auth_logout"], {".config/colab-cli/token.json": "garbage"}),
    ("auth_url", ["auth_url"], {}),
    ("accounts_list_empty", ["accounts_list"], {}),
    ("accounts_list_seeded", ["accounts_list"], {".config/colab-cli/accounts.json": ACCOUNTS_SEED}),
    ("accounts_switch_missing", ["accounts_switch", "x@y.z"], {}),
    ("accounts_switch_bad_json", ["accounts_switch", "A@B.c"], {".config/colab-cli/accounts.json": ACCOUNTS_SEED}),
    ("accounts_remove_missing", ["accounts_remove", "x@y.z"], {}),
    ("accounts_remove_seeded", ["accounts_remove", "a@b.c"], {".config/colab-cli/accounts.json": ACCOUNTS_SEED}),
    ("keepalive_no_token", ["keepalive"], {}),
    ("stop_all_no_token", ["stop_all"], {}),
    ("wake_no_token", ["wake", "--gpu", "T4", "--session", "s1"], {}),
]


def run_case(script, args, seed):
    with tempfile.TemporaryDirectory() as home:
        for rel, content in seed.items():
            p = os.path.join(home, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
        env = dict(os.environ, HOME=home, USERPROFILE=home, PYTHONDONTWRITEBYTECODE="1")
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env.pop(k, None)
        r = subprocess.run([sys.executable, script, *args], env=env, capture_output=True, timeout=120)
        files = {}
        for dp, _, fns in os.walk(home):
            for fn in fns:
                rel = os.path.relpath(os.path.join(dp, fn), home).replace(os.sep, "/")
                files[rel] = read(os.path.join(dp, fn)) if rel.endswith("accounts.json") else "<{} bytes>".format(
                    os.path.getsize(os.path.join(dp, fn)))
        norm = lambda b: re.sub(rb"(state|code_challenge)=[A-Za-z0-9_\-]+", rb"\1=X", b)  # noqa: E731
        return r.returncode, norm(r.stdout), norm(r.stderr), files


def dynamic_check(old_text, errors):
    with tempfile.TemporaryDirectory() as d:
        old_script = os.path.join(d, "colab_bridge.py")  # 同名，argparse 的 prog 才一致
        with open(old_script, "w", encoding="utf-8", newline="\n") as f:
            f.write(old_text)
        new_script = os.path.join(ROOT, "colab_bridge.py")
        for name, args, seed in CASES:
            a, b = run_case(old_script, args, seed), run_case(new_script, args, seed)
            ok = a == b
            first = a[1].decode("utf-8", "replace").strip().splitlines()[:1] or a[2].decode("utf-8", "replace").strip().splitlines()[-1:]
            print(f"  {'OK  ' if ok else 'DIFF'} {name:28s} exit={a[0]} {first[0][:70] if first else ''}")
            if not ok:
                errors.append(f"命令 {name} 新旧输出不一致：\n  old={a}\n  new={b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default=PRE_SPLIT_REV, help="拆分前 colab_bridge.py 所在提交")
    ap.add_argument("--static-only", action="store_true")
    a = ap.parse_args()
    old_text = subprocess.run(["git", "show", f"{a.rev}:colab_bridge.py"], cwd=ROOT, capture_output=True,
                              check=True).stdout.decode("utf-8")
    errors = []
    print("静态检查")
    static_check(old_text, errors)
    if not a.static_only:
        print("CLI 对比（假 HOME，不联网）")
        dynamic_check(old_text, errors)
    for e in errors:
        print("错误：" + e)
    print("通过" if not errors else f"失败：{len(errors)} 项")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
