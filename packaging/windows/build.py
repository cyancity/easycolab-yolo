"""打包 Windows 桌面客户端：python packaging/windows/build.py [版本号]

1. 用 workbench/static/icon-512.png 生成 build/app.ico
2. PyInstaller 打出 dist/ColabFleet/ColabFleet.exe
3. 找得到 Inno Setup 6 的 ISCC.exe 时，再打出 dist/ColabFleet-Setup-<版本>.exe

依赖：pip install -r requirements.txt -r requirements-desktop.txt
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def make_icon() -> Path:
    from PIL import Image
    out = ROOT / "build" / "app.ico"
    out.parent.mkdir(exist_ok=True)
    sizes = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
    Image.open(ROOT / "workbench" / "static" / "icon-512.png").save(out, sizes=sizes)
    return out


def find_iscc():
    found = shutil.which("iscc")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 str(Path(local) / "Programs") if local else None):
        if base and (Path(base) / "Inno Setup 6" / "ISCC.exe").exists():
            return str(Path(base) / "Inno Setup 6" / "ISCC.exe")
    return None


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    version = (arg or os.environ.get("COLAB_FLEET_VERSION") or "0.1.0").lstrip("v")
    make_icon()
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build" / "pyinstaller"),
                    str(HERE / "ColabFleet.spec")], check=True, cwd=ROOT)
    iscc = find_iscc()
    if not iscc:
        print("没有找到 Inno Setup 6（ISCC.exe），只生成了 dist/ColabFleet/。装好后重跑可得到安装包。")
        return
    subprocess.run([iscc, f"/DAppVersion={version}", str(HERE / "ColabFleet.iss")], check=True, cwd=HERE)
    print("安装包：", ROOT / "dist" / f"ColabFleet-Setup-{version}.exe")


if __name__ == "__main__":
    main()
