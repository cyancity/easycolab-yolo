#!/usr/bin/env python3
"""
Cross-compile & assemble a standalone, zero-dependency Windows x64 release
(`dist/EasyColab/EasyColab.exe` + `EasyColab-Windows-x64.zip`) directly from macOS/Linux.

Architecture:
  - Native x86_64 Win32 GUI bootloader (`EasyColab.exe`, SUBSYSTEM:WINDOWS, AppIcon.ico,
    Windows 11 PerMonitorV2 High-DPI & UTF-8 manifest) compiled via mingw-w64.
  - Official PSF-signed Python 3.12.10 Windows x64 Embeddable Runtime (`_internal/runtime/`).
  - Official `cp312-win_amd64` binary wheels (`PySide6` + `QtMultimedia` + FFmpeg + `colab_bridge` deps).
  - Transitive PE DLL import graph analyzer (`x86_64-w64-mingw32-objdump -p`) that prunes unused
    QML / 3D / PDF / Designer bloat while guaranteeing 100% of required Qt6 + FFmpeg DLLs remain intact.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
WIN_APP_DIR = REPO_ROOT / "windows_app"
LAUNCHER_SRC_DIR = WIN_APP_DIR / "win_launcher"
DIST_ROOT = REPO_ROOT / "dist"
BUNDLE_DIR = DIST_ROOT / "EasyColab"
INTERNAL_DIR = BUNDLE_DIR / "_internal"
RUNTIME_DIR = INTERNAL_DIR / "runtime"
SITE_PACKAGES_DIR = INTERNAL_DIR / "site-packages"
OUTPUT_ZIP = REPO_ROOT / "EasyColab-Windows-x64.zip"

PY_VERSION = "3.12.10"
PY_EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
CACHE_DIR = Path("/tmp/easycolab_win_build_cache")

REQUIRED_PYSIDE_MODULES = {
    "__init__.py",
    "_config.py",
    "_git_pyside_version.py",
    "qt.conf",
    "QtCore.pyd",
    "QtGui.pyd",
    "QtWidgets.pyd",
    "QtNetwork.pyd",
    "QtMultimedia.pyd",
    "QtMultimediaWidgets.pyd",
    "QtSvg.pyd",
    "QtSvgWidgets.pyd",
    "QtOpenGL.pyd",
    "QtOpenGLWidgets.pyd",
}

REQUIRED_QT_PLUGIN_DIRS = {
    "platforms",
    "styles",
    "imageformats",
    "iconengines",
    "multimedia",
    "tls",
    "networkinformation",
    "generic",
}

# Standard Windows system DLLs provided by Windows 10/11 OS
WINDOWS_SYSTEM_DLL_PREFIXES = (
    "api-ms-win-",
    "ext-ms-win-",
)
WINDOWS_SYSTEM_DLLS = {
    "kernel32.dll", "user32.dll", "gdi32.dll", "shell32.dll", "advapi32.dll",
    "ole32.dll", "oleaut32.dll", "uuid.dll", "ws2_32.dll", "winmm.dll",
    "imm32.dll", "dwmapi.dll", "uxtheme.dll", "shlwapi.dll", "comdlg32.dll",
    "comctl32.dll", "version.dll", "setupapi.dll", "cfgmgr32.dll", "propsys.dll",
    "ntdll.dll", "crypt32.dll", "bcrypt.dll", "ncrypt.dll", "secur32.dll",
    "iphlpapi.dll", "dnsapi.dll", "winhttp.dll", "wininet.dll", "netapi32.dll",
    "userenv.dll", "wtsapi32.dll", "powrprof.dll", "d3d11.dll", "d3d12.dll",
    "dxgi.dll", "dwrite.dll", "d2d1.dll", "dcomp.dll", "mf.dll", "mfplat.dll",
    "mfreadwrite.dll", "mfuuid.dll", "evr.dll", "wmVCORE.DLL".lower(),
    "opengl32.dll", "glu32.dll", "avrt.dll", "mmdevapi.dll", "ksuser.dll",
    "msvcrt.dll", "ucrtbase.dll", "rpcrt4.dll", "dbghelp.dll", "psapi.dll",
    "mpr.dll", "credui.dll", "wintrust.dll", "hid.dll", "dxva2.dll",
    "d3d9.dll", "uxtheme.dll", "shcore.dll", "uiautomationcore.dll",
    "authz.dll", "icuuc.dll", "icuin.dll", "mscms.dll", "cabinet.dll", "msi.dll",
}


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def get_pe_imports(pe_path: Path, objdump_bin: str) -> List[str]:
    proc = subprocess.run(
        [objdump_bin, "-p", str(pe_path)],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    return [m.group(1).strip() for m in re.finditer(r"DLL Name:\s+(\S+)", proc.stdout)]


def is_windows_os_dll(dll_name: str) -> bool:
    low = dll_name.lower()
    if low in WINDOWS_SYSTEM_DLLS:
        return True
    return any(low.startswith(pref) for pref in WINDOWS_SYSTEM_DLL_PREFIXES)


def build_native_exe() -> None:
    step("1/6 Compiling native x86_64 Windows GUI launcher (EasyColab.exe)...")
    gcc = shutil.which("x86_64-w64-mingw32-gcc")
    windres = shutil.which("x86_64-w64-mingw32-windres")
    if not gcc or not windres:
        raise RuntimeError("mingw-w64 not found. Install via `brew install mingw-w64`.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    res_obj = CACHE_DIR / "launcher_res.o"
    exe_out = BUNDLE_DIR / "EasyColab.exe"

    subprocess.run(
        [windres, "launcher.rc", "-O", "coff", "-o", str(res_obj)],
        cwd=str(LAUNCHER_SRC_DIR),
        check=True,
    )
    subprocess.run(
        [
            gcc,
            "-O2",
            "-s",
            "-municode",
            "-mwindows",
            "-static-libgcc",
            str(LAUNCHER_SRC_DIR / "launcher.c"),
            str(res_obj),
            "-lshell32",
            "-luser32",
            "-lkernel32",
            "-o",
            str(exe_out),
        ],
        check=True,
    )
    print(f"    Compiled {exe_out} ({exe_out.stat().st_size:,} bytes)")


def setup_python_embed_runtime() -> None:
    step(f"2/6 Extracting official Python {PY_VERSION} Windows x64 embeddable runtime...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_zip = CACHE_DIR / f"python-{PY_VERSION}-embed-amd64.zip"
    tmp_dl = Path(f"/tmp/python-{PY_VERSION}-embed-amd64.zip")
    if tmp_dl.exists() and not cached_zip.exists():
        shutil.copy2(tmp_dl, cached_zip)
    if not cached_zip.exists():
        print(f"    Downloading {PY_EMBED_URL} ...")
        urllib.request.urlretrieve(PY_EMBED_URL, cached_zip)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cached_zip, "r") as zf:
        zf.extractall(RUNTIME_DIR)

    # Configure isolated _pth so system PYTHONPATH/Registry never pollutes EasyColab
    pth_file = RUNTIME_DIR / "python312._pth"
    pth_file.write_text(
        "python312.zip\r\n"
        ".\r\n"
        "..\r\n"
        "..\\site-packages\r\n"
        "import site\r\n",
        encoding="utf-8",
    )
    print(f"    Configured {pth_file}")


def install_win_amd64_wheels() -> None:
    step("3/6 Installing cp312-win_amd64 binary wheels into _internal/site-packages...")
    SITE_PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    whl_cache = CACHE_DIR / "wheels_cp312_win_amd64"
    whl_cache.mkdir(parents=True, exist_ok=True)

    reqs = [
        "PySide6-Essentials>=6.6.0",
        "PySide6-Addons>=6.6.0",
        "requests>=2.31.0",
        "google-colab-cli>=0.7.5",
        "google-auth>=2.28.0",
        "google-auth-oauthlib>=1.2.0",
        "requests-oauthlib>=1.3.1",
        "websocket-client>=1.7.0",
    ]
    if len(list(whl_cache.glob("*.whl"))) < 50:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--dest",
                str(whl_cache),
                "--platform",
                "win_amd64",
                "--python-version",
                "3.12",
                "--implementation",
                "cp",
                "--abi",
                "cp312",
                "--only-binary=:all:",
                *reqs,
            ],
            check=True,
        )
    else:
        print(f"    Using cached cp312-win_amd64 wheels in {whl_cache}")

    wheel_files = sorted(whl_cache.glob("*.whl"))
    print(f"    Extracting {len(wheel_files)} verified wheels...")
    for whl in wheel_files:
        with zipfile.ZipFile(whl, "r") as zf:
            zf.extractall(SITE_PACKAGES_DIR)


def prune_unused_qt_bloat() -> None:
    step("4/6 Pruning unused Qt QML/3D/PDF/Designer binaries via PE import graph...")
    objdump = shutil.which("x86_64-w64-mingw32-objdump")
    assert objdump is not None

    pyside_dir = SITE_PACKAGES_DIR / "PySide6"
    shiboken_dir = SITE_PACKAGES_DIR / "shiboken6"

    # 1. Remove dev folders & CLI .exe tools inside PySide6
    for folder_name in ("qml", "include", "glue", "typesystems", "examples", "doc", "scripts"):
        p = pyside_dir / folder_name
        if p.is_dir():
            shutil.rmtree(p)
    for exe in pyside_dir.glob("*.exe"):
        exe.unlink()

    # 2. Keep only required plugin categories (and drop qpdf.dll imageformat plugin)
    plugins_dir = pyside_dir / "plugins"
    if plugins_dir.is_dir():
        for sub in list(plugins_dir.iterdir()):
            if sub.is_dir() and sub.name not in REQUIRED_QT_PLUGIN_DIRS:
                shutil.rmtree(sub)
        qpdf_plugin = plugins_dir / "imageformats" / "qpdf.dll"
        if qpdf_plugin.exists():
            qpdf_plugin.unlink()

    # 3. Remove unused PySide6 Python bindings (.pyd / .pyi)
    for pyi in pyside_dir.rglob("*.pyi"):
        pyi.unlink()
    for pyi in shiboken_dir.rglob("*.pyi"):
        pyi.unlink()
    for pyd in list(pyside_dir.glob("*.pyd")):
        if pyd.name not in REQUIRED_PYSIDE_MODULES:
            pyd.unlink()

    # 4. Walk transitive PE DLL dependency graph from all kept .pyd and plugin .dll files
    dll_lookup: Dict[str, Path] = {}
    for search_dir in (pyside_dir, shiboken_dir, RUNTIME_DIR):
        for f in search_dir.glob("*.dll"):
            dll_lookup[f.name.lower()] = f

    roots: List[Path] = list(pyside_dir.glob("*.pyd")) + list(shiboken_dir.glob("*.pyd"))
    if plugins_dir.is_dir():
        roots.extend(plugins_dir.rglob("*.dll"))
    # Always keep software OpenGL fallback & Direct3D shader compiler for Windows VMs / RDP
    for extra in ("opengl32sw.dll", "d3dcompiler_47.dll"):
        if extra in dll_lookup:
            roots.append(dll_lookup[extra])

    needed_dlls: Set[str] = set()
    queue: List[Path] = list(roots)
    visited: Set[Path] = set()

    while queue:
        cur = queue.pop()
        if cur in visited or not cur.exists():
            continue
        visited.add(cur)
        for imp in get_pe_imports(cur, objdump):
            low = imp.lower()
            if low not in needed_dlls:
                needed_dlls.add(low)
                if low in dll_lookup:
                    queue.append(dll_lookup[low])

    removed_count = 0
    removed_bytes = 0
    for dll in list(pyside_dir.glob("*.dll")):
        if dll.name.lower() not in needed_dlls:
            removed_bytes += dll.stat().st_size
            dll.unlink()
            removed_count += 1

    # Also remove bulky unused test suites inside pyarrow/tests if present
    for test_dir in (
        SITE_PACKAGES_DIR / "pyarrow" / "tests",
        SITE_PACKAGES_DIR / "pyarrow" / "include",
    ):
        if test_dir.is_dir():
            shutil.rmtree(test_dir)

    print(
        f"    Pruned {removed_count} unused Qt DLLs ({removed_bytes / (1024 * 1024):.1f} MB saved), "
        f"kept {len(list(pyside_dir.glob('*.dll')))} transitive Qt/FFmpeg DLLs."
    )


def copy_app_sources_and_resources() -> None:
    step("5/6 Copying EasyColab application modules, icons & Colab server resources...")
    # Entry script & core resources in _internal/
    shutil.copy2(WIN_APP_DIR / "easycolab_win.py", INTERNAL_DIR / "easycolab_win.py")
    for res_name in ("app.py", "colab_bridge.py", "EasyColab_Server.ipynb"):
        shutil.copy2(REPO_ROOT / res_name, INTERNAL_DIR / res_name)
    # Colab 服务端包与运行时模块：桥接唤醒时会随 app.py 一起上传
    for pkg in ("colab_server", "runtime"):
        dst = INTERNAL_DIR / pkg
        dst.mkdir(parents=True, exist_ok=True)
        for py_file in (REPO_ROOT / pkg).glob("*.py"):
            shutil.copy2(py_file, dst / py_file.name)

    # Copy windows_app package
    dst_pkg = INTERNAL_DIR / "windows_app"
    dst_pkg.mkdir(parents=True, exist_ok=True)
    for py_file in WIN_APP_DIR.glob("*.py"):
        if py_file.name == "build_windows_from_mac.py":
            continue
        shutil.copy2(py_file, dst_pkg / py_file.name)

    dst_assets = dst_pkg / "assets"
    dst_assets.mkdir(parents=True, exist_ok=True)
    for asset in (WIN_APP_DIR / "assets").glob("*"):
        if asset.is_file():
            shutil.copy2(asset, dst_assets / asset.name)

    # User quick-start note in root EasyColab/ next to EasyColab.exe (pure ASCII filename for legacy zip archivers)
    readme_txt = BUNDLE_DIR / "README-Windows.txt"
    readme_txt.write_text(
        "EasyColab Windows 免安装绿色版 (x64) / Portable Edition\r\n"
        "========================================================\r\n\r\n"
        "1. 请确保已将压缩包完整解压（保留 EasyColab.exe 与同级 _internal 文件夹）。\r\n"
        "   Please extract the entire archive (keep EasyColab.exe alongside the _internal folder).\r\n"
        "2. 直接双击 EasyColab.exe 即可启动工作台（已内置完整运行环境与音视频解码器，无需安装 Python）。\r\n"
        "   Double-click EasyColab.exe to launch (Python runtime, Qt6 & codecs included).\r\n"
        "3. 若首次运行遇到 Windows SmartScreen 提示“Windows 已保护你的电脑”，\r\n"
        "   请点击「更多信息」->「仍要运行」 (More info -> Run anyway)。\r\n"
        "4. 生成的视频与图片默认自动保存在：%USERPROFILE%\\Videos\\EasyColab\r\n",
        encoding="utf-8-sig",
    )


def verify_and_zip() -> None:
    step("6/6 Auditing PE64 binary dependencies & creating EasyColab-Windows-x64.zip...")
    objdump = shutil.which("x86_64-w64-mingw32-objdump")
    assert objdump is not None

    exe_path = BUNDLE_DIR / "EasyColab.exe"
    exe_imports = get_pe_imports(exe_path, objdump)
    for imp in exe_imports:
        if not is_windows_os_dll(imp):
            raise RuntimeError(f"EasyColab.exe has unexpected non-OS DLL dependency: {imp}")
    print(f"    EasyColab.exe imports verified clean OS-only DLLs: {exe_imports}")

    # Verify all PySide6 + runtime DLLs/PYDs resolve every non-OS dependency
    available_dlls: Set[str] = {
        p.name.lower()
        for d in (RUNTIME_DIR, SITE_PACKAGES_DIR / "PySide6", SITE_PACKAGES_DIR / "shiboken6")
        for p in d.glob("*.dll")
    }
    check_targets = (
        list((SITE_PACKAGES_DIR / "PySide6").glob("*.pyd"))
        + list((SITE_PACKAGES_DIR / "PySide6").glob("*.dll"))
        + list((SITE_PACKAGES_DIR / "PySide6" / "plugins").rglob("*.dll"))
        + list(RUNTIME_DIR.glob("*.pyd"))
    )
    missing: Dict[str, List[str]] = {}
    for bin_file in check_targets:
        for imp in get_pe_imports(bin_file, objdump):
            low = imp.lower()
            if low not in available_dlls and not is_windows_os_dll(low):
                missing.setdefault(bin_file.name, []).append(imp)

    if missing:
        raise RuntimeError(f"Missing transitive DLLs detected: {missing}")
    print(f"    Verified {len(check_targets)} PE64 binaries (.pyd/.dll): 0 missing dependencies!")

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    file_count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for item in sorted(BUNDLE_DIR.rglob("*")):
            if item.is_file():
                arcname = Path("EasyColab") / item.relative_to(BUNDLE_DIR)
                zf.write(item, str(arcname).replace("\\", "/"))
                file_count += 1

    zip_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\nSUCCESS! Packaged {file_count} files -> {OUTPUT_ZIP} ({zip_mb:.1f} MB)")


def main() -> None:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    build_native_exe()
    setup_python_embed_runtime()
    install_win_amd64_wheels()
    prune_unused_qt_bloat()
    copy_app_sources_and_resources()
    verify_and_zip()


if __name__ == "__main__":
    main()
