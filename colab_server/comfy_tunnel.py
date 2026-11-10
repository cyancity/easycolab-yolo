"""ComfyUI 画布的公网隧道。隧道只指向访问保护端口（ComfyUI 端口 + 1，见 comfy_guard.ensure_guard），不直接指向 ComfyUI。

优先 gradio 分享隧道（*.gradio.live），建不起来再退回 cloudflared 快速隧道（*.trycloudflare.com）。
2026-11-10 在 G4 同一实例实测：gradio.live 约 449 KB/s、空请求约 1.0 s；trycloudflare 约 67 KB/s、约 2.4 s。
ComfyUI 前端约 10 MB、400 多个请求，经 trycloudflare 第一次打开要约 5 分钟。

gradio 没有公开的隧道接口。这里调用 gradio.networking.setup_tunnel：Blocks.launch(share=True) 自己用的就是它，
4.x 到 6.31 参数不变。参数名对不上、建隧道报错或超时，都退回 cloudflared。
"""
import atexit
import inspect
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.request
from typing import Optional, Tuple
from urllib.parse import urlparse

_GRADIO_PARAMS = ["local_host", "local_port", "share_token", "share_server_address", "share_server_tls_certificate"]
_GRADIO_TIMEOUT = 75.0  # setup_tunnel 里取分享服务器最多 30 s，等 frpc 报地址最多 30 s
_CF_WAIT = 25.0
_CF_CMD = "cloudflared tunnel --url http://127.0.0.1:"
_CF_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def _frpc_pattern(port: int) -> str:
    # gradio 的 frpc 命令行是 <frpc> http -n <分享令牌> -l <端口> -i <主机> ...；app.py 自己的分享隧道是 -l 7860
    return f"frpc.* -l {int(port)} -i 127\\.0\\.0\\.1"


def _pgrep(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, check=False).returncode == 0


def kill_tunnels(guard_port: int) -> None:
    """停掉画布隧道（两种都停），不碰 app.py 自己的分享隧道"""
    subprocess.run(["pkill", "-f", _CF_CMD], check=False)
    subprocess.run(["pkill", "-f", _frpc_pattern(guard_port)], check=False)


def tunnel_alive(guard_port: int) -> bool:
    """指向访问保护端口的隧道进程还在不在（app.py 热更新重启后用来判断旧地址能否沿用）"""
    return _pgrep(_frpc_pattern(guard_port)) or _pgrep(f"{_CF_CMD}{int(guard_port)}")


def _first_line(e: BaseException) -> str:
    return f"{type(e).__name__}: {(str(e).strip().splitlines() or [''])[0][:160]}"


def _drain(proc: Optional[subprocess.Popen]) -> None:
    """gradio 拿到地址后不再读 frpc 的输出，管道写满 frpc 会卡住，这里接着读掉"""
    if proc is None:
        return
    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            threading.Thread(target=lambda s=stream: [None for _ in iter(s.readline, b"")], daemon=True).start()


def _open_gradio(guard_port: int) -> Tuple[str, Optional[subprocess.Popen]]:
    from gradio import networking, tunneling

    fn = getattr(networking, "setup_tunnel", None)
    if fn is None or list(inspect.signature(fn).parameters)[:5] != _GRADIO_PARAMS:
        raise RuntimeError("这个 gradio 版本没有可用的 setup_tunnel")
    box = {}

    def run():
        try:
            box["url"] = fn("127.0.0.1", guard_port, secrets.token_urlsafe(32), None, None)
        except BaseException as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(_GRADIO_TIMEOUT)
    host = urlparse(str(box.get("url") or "").strip()).netloc
    if not host:
        subprocess.run(["pkill", "-f", _frpc_pattern(guard_port)], check=False)
        raise RuntimeError(_first_line(box["err"]) if "err" in box else "超时")
    tn = next((tn for tn in reversed(getattr(tunneling, "CURRENT_TUNNELS", []))
               if getattr(tn, "local_port", None) == guard_port and getattr(tn, "proc", None)), None)
    if tn is None:
        return f"https://{host}", None
    # gradio 在 app.py 退出时会停掉隧道并打印地址；画布隧道要和 cloudflared 一样活过 app.py 热更新重启
    atexit.unregister(tn.kill)
    _drain(tn.proc)
    return f"https://{host}", tn.proc


def _cloudflared_bin() -> str:
    cf_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if os.path.isfile(cf_bin) and os.access(cf_bin, os.X_OK):
        return cf_bin
    dl_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    os.makedirs(os.path.dirname(cf_bin), exist_ok=True)
    urllib.request.urlretrieve(dl_url, cf_bin)
    os.chmod(cf_bin, 0o755)
    return cf_bin


def _open_cloudflared(guard_port: int) -> Tuple[str, subprocess.Popen]:
    proc = subprocess.Popen(
        [_cloudflared_bin(), "tunnel", "--url", f"http://127.0.0.1:{guard_port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    box = [""]

    def watch():  # 一直读到进程退出，顺带防止管道写满
        try:
            for ln in proc.stdout or []:
                m = _CF_RE.search(ln)
                if m and not box[0]:
                    box[0] = m.group(0)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=watch, daemon=True).start()
    t_end = time.time() + _CF_WAIT
    while not box[0] and time.time() < t_end and proc.poll() is None:
        time.sleep(0.5)
    if not box[0]:
        proc.terminate()
        raise RuntimeError("未能获取 Cloudflare 公网隧道地址，请重试。")
    return box[0], proc


def open_tunnel(guard_port: int) -> Tuple[str, Optional[subprocess.Popen]]:
    """先停掉旧的画布隧道，再开新的，返回 (公网地址, 隧道进程)。两种都失败时抛 RuntimeError"""
    kill_tunnels(guard_port)
    try:
        url, proc = _open_gradio(guard_port)
        print("[ComfyUI] 画布隧道：gradio.live")
        return url, proc
    except Exception as e:  # noqa: BLE001
        print(f"[ComfyUI] gradio 隧道不可用，改用 Cloudflare：{_first_line(e)}")
    url, proc = _open_cloudflared(guard_port)
    print("[ComfyUI] 画布隧道：trycloudflare")
    return url, proc
