#!/usr/bin/env python3
"""用 app.py 和 colab_server/*.py 重新生成 EasyColab_Server.ipynb 里写文件的代码格。

用法：python tools/build_notebook.py [--check]
  默认写回笔记本；--check 只检查笔记本是否已是最新（不一致时退出码 1）。

笔记本里原来的 `%%writefile app.py` 格被替换成一组格：先 `mkdir -p colab_server`，再每个包文件一格
`%%writefile colab_server/<模块>.py`，最后是入口 `%%writefile app.py`。其它格原样保留。
包的文件顺序取自 app.py 入口里 `from colab_server import (...)` 的列表（即加载顺序）。
笔记本模式不上传 runtime/*.py（flash_next / gpu_stats 只在工作台部署时才有）。
"""
import argparse
import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, "EasyColab_Server.ipynb")
PKG = "colab_server"
MARK = "%%writefile "


def read(path: str) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        return f.read().replace("\r\n", "\n")


def package_order(entry_text: str):
    """入口 app.py 里 from colab_server import (...) 的名字，没有则说明还是单文件。"""
    for n in ast.parse(entry_text).body:
        if isinstance(n, ast.ImportFrom) and n.module == PKG:
            return [a.name for a in n.names]
    return []


def code_cell(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": cell_id},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_cells(entry_text: str):
    order = package_order(entry_text)
    if not order:  # 单文件模式：和拆分前一样
        return [code_cell("write_webui", MARK + "app.py\n" + entry_text)]
    cells = [code_cell("make_pkg_dir", f"#@title 创建服务端包目录\n!mkdir -p {PKG}")]
    for mod in ["__init__"] + order:
        path = os.path.join(ROOT, PKG, mod + ".py")
        if not os.path.isfile(path):
            sys.exit(f"build_notebook: 缺少 {PKG}/{mod}.py，先跑 tools/split_app.py")
        cells.append(code_cell(f"write_pkg_{mod}", f"{MARK}{PKG}/{mod}.py\n" + read(path)))
    cells.append(code_cell("write_webui", MARK + "app.py\n" + entry_text))
    return cells


def is_generated(cell: dict) -> bool:
    src = "".join(cell.get("source", []))
    return (cell.get("metadata", {}).get("id") == "make_pkg_dir"
            or src.startswith(MARK + "app.py\n") or src.startswith(MARK + PKG + "/"))


def build(nb: dict, entry_text: str) -> dict:
    cells = nb["cells"]
    idx = [i for i, c in enumerate(cells) if is_generated(c)]
    if idx:
        at = idx[0]
    else:  # 没有写文件的格：放在“启动服务端”格之前
        at = next((i for i, c in enumerate(cells) if "python app.py" in "".join(c.get("source", []))), len(cells))
    kept = [c for i, c in enumerate(cells) if i not in set(idx)]
    before = len([i for i in range(at) if i not in set(idx)])
    nb["cells"] = kept[:before] + write_cells(entry_text) + kept[before:]
    return nb


def dump(nb: dict, crlf: bool = True) -> str:
    # 与现有笔记本一致：2 空格缩进、不转义中文、换行风格跟随原文件
    text = json.dumps(nb, indent=2, ensure_ascii=False) + "\n"
    return text.replace("\n", "\r\n") if crlf else text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    with open(NOTEBOOK, encoding="utf-8", newline="") as f:
        old = f.read()
    new = dump(build(json.loads(old), read(os.path.join(ROOT, "app.py"))), "\r\n" in old)
    if a.check:
        print("笔记本已是最新" if new == old else "笔记本和 app.py / colab_server 不一致，请跑 tools/build_notebook.py")
        sys.exit(0 if new == old else 1)
    with open(NOTEBOOK, "w", encoding="utf-8", newline="") as f:
        f.write(new)
    nb = json.loads(new)
    print(f"已写入 {os.path.basename(NOTEBOOK)}：{len(nb['cells'])} 个格，其中写文件的 {sum(is_generated(c) for c in nb['cells'])} 个")


if __name__ == "__main__":
    main()
