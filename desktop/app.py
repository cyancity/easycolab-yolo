"""桌面客户端入口。

ColabFleet.exe            启动后台（已在运行就不重复启动），打开窗口，托盘常驻
ColabFleet.exe --tray     开机自启用：只启动后台和托盘，不开窗口
ColabFleet.exe --bridge … 打包后没有独立 python，工作台调用 colab_bridge.py 时走这个入口
"""
import os
import socket
import sys
import threading
import time
import urllib.request

PORT = 8790  # 与 workbench.server.PORT 一致；这里不导入 server，--bridge 子进程不必加载整套后台
URL = f"http://127.0.0.1:{PORT}/"
LOG_MAX = 5 * 1024 * 1024


def _attach_stdio():
    """无控制台的 exe 里 sys.stdout 可能是 None；桥接子进程的输出要经管道交回工作台"""
    for name, fd in (("stdout", 1), ("stderr", 2)):
        if getattr(sys, name) is None:
            try:
                setattr(sys, name, open(fd, "w", encoding="utf-8", errors="replace", closefd=False, buffering=1))
            except OSError:
                setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _run_bridge(argv):
    _attach_stdio()
    sys.argv = ["colab_bridge.py", *argv]
    from bridge.cli import run_cli  # 导入 bridge 时先完成 UTF-8 输出与系统代理注入
    run_cli(argv)


def _log_to_file():
    """打包后没有控制台：后台日志写进数据目录，出问题时能查"""
    from workbench.paths import DATA_DIR, FROZEN
    if not FROZEN and sys.stdout is not None:
        return None
    path = DATA_DIR / "desktop.log"
    if path.exists() and path.stat().st_size > LOG_MAX:
        path.replace(path.with_suffix(".log.1"))
    f = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    sys.stdout = sys.stderr = f
    print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 {' '.join(sys.argv[1:])}")
    return path


def _port_busy() -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def _page() -> bytes:
    try:
        with urllib.request.urlopen(URL, timeout=2) as r:
            return r.read() if r.status == 200 else b""
    except OSError:
        return b""


def _wait_ready(seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if _page():
            return True
        time.sleep(0.5)
    return False


def _start_server():
    import uvicorn
    from workbench.server import app, boot
    boot()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    thread = threading.Thread(target=server.run, name="workbench", daemon=True)
    thread.start()
    return server, thread


def _active_count() -> int:
    from workbench import deployments
    return sum(1 for d in deployments.all_deps() if d.get("status") in deployments.ACTIVE)


def _run_tray(server):
    import pystray
    from PIL import Image
    from desktop import winsys
    from workbench.paths import STATIC_DIR

    def quit_app(icon, _item):
        n = _active_count()
        if n and not winsys.confirm(
                f"还有 {n} 台实例在运行。\n\n退出后本机网关停止，agent 调不到模型。"
                "实例仍在 Colab 上消耗计算单元，直到闲置自动释放或你在工作台停止它。\n\n确定退出？"):
            return
        server.should_exit = True
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("打开工作台", lambda *_: winsys.open_window(URL), default=True),
        pystray.MenuItem("开机自启", lambda *_: winsys.set_autostart(not winsys.autostart_enabled()),
                         checked=lambda _item: winsys.autostart_enabled()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("退出", quit_app),
    )
    pystray.Icon("ColabFleet", Image.open(STATIC_DIR / "icon-192.png"), "Colab Fleet", menu).run()


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["--bridge"]:
        return _run_bridge(argv[1:])
    tray_only = "--tray" in argv
    log = _log_to_file()
    from desktop import winsys

    if not winsys.claim_single_instance():
        # 已有客户端在跑：等它的后台就绪后开窗口，自己退出
        if not tray_only and _wait_ready(30):
            winsys.open_window(URL)
        return
    if _port_busy():
        if b"EasyColab" in _page():  # 开发时 python -m workbench 起的后台，直接用
            if not tray_only:
                winsys.open_window(URL)
            return
        winsys.alert(f"端口 {PORT} 被其他程序占用，Colab Fleet 无法启动。")
        return

    server, thread = _start_server()
    if not _wait_ready(60):
        winsys.alert("工作台后台没有启动成功。" + (f"\n日志：{log}" if log else ""))
        os._exit(1)
    if not tray_only:
        winsys.open_window(URL)
    _run_tray(server)
    thread.join(timeout=10)
    os._exit(0)  # 监控线程不必等，进程直接结束
