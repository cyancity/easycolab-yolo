# PyInstaller 配置，由 packaging/windows/build.py 调用
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parents[1]


def tree(src, dest):
    """目录里除字节码缓存外的文件原样放进包内。Colab 端代码要按源文件上传，不能只留在 PYZ 里"""
    base = ROOT / src
    return [(str(p), str(Path(dest) / p.relative_to(base).parent))
            for p in base.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]


datas = (
    tree("workbench/static", "workbench/static")
    + tree("workbench/model_docs", "workbench/model_docs")
    + tree("colab_server", "colab_server")
    + tree("runtime", "runtime")
    + tree("bridge", "bridge")
    + [(str(ROOT / "app.py"), "."), (str(ROOT / "colab_bridge.py"), ".")]
    + collect_data_files("colab_cli")
)
hidden = (
    collect_submodules("uvicorn") + collect_submodules("workbench") + collect_submodules("bridge")
    + collect_submodules("colab_cli") + collect_submodules("desktop") + ["pystray._win32"]
)

a = Analysis(
    [str(ROOT / "packaging" / "windows" / "entry.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "pyarrow", "pandas"],  # pyarrow 只在 jupyter_mimetypes 序列化表格时按需导入，这里用不到
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ColabFleet",
    console=False,
    icon=str(ROOT / "build" / "app.ico"),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ColabFleet", upx=False)
