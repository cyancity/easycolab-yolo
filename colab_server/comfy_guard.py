"""ComfyUI 前面的访问保护。画布公网隧道（comfy_tunnel）指向这里（ComfyUI 端口 + 1），不直接指向 ComfyUI。

ComfyUI 自己没有登录，隧道地址谁拿到谁就能用这张显卡。这里要求一个随机令牌：
  GET /__colab_fleet_auth?token=<令牌>   令牌对就写 HttpOnly Cookie 并跳回 /
  其余请求（含 WebSocket）带 Cookie 或 Authorization: Bearer <令牌> 才转发，否则 401
令牌只经实例的密钥路径（/k/<secret>/minimax_api/comfyui/status）交给本机工作台。

作为子进程运行：python -m colab_server.comfy_guard --listen-port 8189 --upstream-port 8188 --token-file <文件>
令牌从文件读，不出现在命令行里。aiohttp 随 ComfyUI 的依赖装好，这里不另装。
"""
import argparse
import asyncio
import hmac
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

COOKIE = "colab_fleet_comfy"
AUTH_PATH = "/__colab_fleet_auth"
TOKEN_FILE = "/tmp/.easycolab_comfy_token"
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
        "transfer-encoding", "upgrade", "host", "content-length"}
_PAGE_401 = "需要从 Colab Fleet 打开画布。".encode("utf-8")
# 前端打包产物文件名带内容哈希，内容不会变。ComfyUI 一律回 no-store，经隧道每次打开都要重下约 10 MB，
# 这里改成浏览器可长期缓存（private：只存在用户自己的浏览器里）
_HASHED = re.compile(r"^/assets/[^/]+-[A-Za-z0-9_-]{8}\.(?:js|css)$")
_IMMUTABLE = "private, max-age=31536000, immutable"


# ---------- 在 app.py 进程里调用：令牌和保护进程 ----------

def token() -> str:
    """本机这台实例的令牌；第一次调用时生成并写入只有本用户可读的文件"""
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            t = f.read().strip()
        if t:
            return t
    except OSError:
        pass
    t = secrets.token_urlsafe(24)
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(t)
    return t


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _guard_alive(port: int) -> bool:
    """带当前令牌访问登录口，回 302 才算在跑且令牌一致；令牌文件重建过的旧进程会回 401"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        opener.open(f"http://127.0.0.1:{port}{AUTH_PATH}?token={urllib.parse.quote(token())}", timeout=1.0)
    except urllib.error.HTTPError as e:
        return e.code == 302
    except OSError:
        return False
    return False


def ensure_guard(upstream_port: int) -> int:
    """保证保护进程在跑，返回它监听的端口（隧道应指向这个端口）"""
    port = upstream_port + 1
    token()
    if _guard_alive(port):
        return port
    subprocess.run(["pkill", "-f", "colab_server.comfy_guard"], check=False)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.Popen(
        [sys.executable, "-m", "colab_server.comfy_guard", "--listen-port", str(port),
         "--upstream-port", str(upstream_port), "--token-file", TOKEN_FILE],
        cwd=here, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(40):
        if _guard_alive(port):
            return port
        time.sleep(0.25)
    raise RuntimeError("ComfyUI 访问保护没有启动成功")


# ---------- 保护进程本身 ----------

def make_app(upstream: str, secret: str):
    from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web
    from multidict import CIMultiDict

    expected = secret.encode()

    def bad_gateway():
        return web.Response(status=502, body="ComfyUI 没有响应，可能还在启动。".encode("utf-8"),
                            content_type="text/plain", charset="utf-8")

    def _ok(value: str) -> bool:
        return bool(value) and hmac.compare_digest(value.encode(), expected)

    def authorized(req) -> bool:
        auth = req.headers.get("Authorization", "")
        return _ok(req.cookies.get(COOKIE, "")) or (auth.startswith("Bearer ") and _ok(auth[7:]))

    def forward_headers(req) -> CIMultiDict:
        # 头名大小写不定，一律按小写比较；我们的令牌（Authorization 和自己的 Cookie）不交给 ComfyUI 和它的插件
        out = CIMultiDict()
        for k, v in req.headers.items():
            lk = k.lower()
            if lk in _HOP or lk.startswith("sec-websocket") or lk in ("authorization", "cookie"):
                continue
            out.add(k, v)
        cookies = [c.strip() for h in req.headers.getall("Cookie", []) for c in h.split(";")]
        cookies = [c for c in cookies if c and not c.startswith(COOKIE + "=")]
        if cookies:
            out["Cookie"] = "; ".join(cookies)
        return out

    async def login(req):
        if not _ok(req.query.get("token", "")):
            return web.Response(status=401, body=_PAGE_401, content_type="text/plain", charset="utf-8")
        resp = web.Response(status=302, headers={"Location": "/"})
        https = req.headers.get("X-Forwarded-Proto", "").lower() == "https" or '"https"' in req.headers.get("Cf-Visitor", "")
        resp.set_cookie(COOKIE, secret, httponly=True, secure=https, samesite="Lax", max_age=30 * 86400, path="/")
        return resp

    async def websocket(req, url: str):
        # 先连上游：连不上就回 502，不先给浏览器 101 再断开
        try:
            up = await req.app["session"].ws_connect(url, headers=forward_headers(req), max_msg_size=0)
        except (ClientError, OSError):
            return bad_gateway()
        down = web.WebSocketResponse(max_msg_size=0)
        try:
            await down.prepare(req)

            async def pump(src, dst):
                async for msg in src:
                    if msg.type == WSMsgType.TEXT:
                        await dst.send_str(msg.data)
                    elif msg.type == WSMsgType.BINARY:
                        await dst.send_bytes(msg.data)
                    else:
                        break
            tasks = [asyncio.ensure_future(pump(down, up)), asyncio.ensure_future(pump(up, down))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in tasks:
                t.cancel()
        finally:
            await up.close()
        await down.close()
        return down

    async def proxy(req):
        if req.path == AUTH_PATH:
            return await login(req)
        if not authorized(req):
            return web.Response(status=401, body=_PAGE_401, content_type="text/plain", charset="utf-8")
        url = upstream + str(req.rel_url)
        if req.headers.get("Upgrade", "").lower() == "websocket":
            return await websocket(req, "ws" + url[4:])
        body = await req.read() if req.can_read_body else None
        resp = None
        try:
            async with req.app["session"].request(req.method, url, headers=forward_headers(req), data=body,
                                                  allow_redirects=False) as up:
                resp = web.StreamResponse(status=up.status, reason=up.reason)
                for k, v in up.headers.items():
                    if k.lower() not in _HOP:
                        resp.headers.add(k, v)
                if up.status == 200 and req.method == "GET" and _HASHED.match(req.path):
                    resp.headers["Cache-Control"] = _IMMUTABLE
                await resp.prepare(req)
                async for chunk in up.content.iter_chunked(64 * 1024):
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except (ClientError, OSError):
            if resp is not None and resp.prepared:
                raise  # 已经开始回传，只能断开
            return bad_gateway()

    async def on_startup(app):
        # 不自动解压：原样转发 Content-Encoding 和字节
        app["session"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10), auto_decompress=False)

    async def on_cleanup(app):
        await app["session"].close()

    app = web.Application(client_max_size=1024 ** 3)  # 允许上传大图和视频首帧
    app.router.add_route("*", "/{tail:.*}", proxy)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main(argv=None):
    from aiohttp import web
    ap = argparse.ArgumentParser(prog="python -m colab_server.comfy_guard")
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--upstream-port", type=int, required=True)
    ap.add_argument("--token-file", required=True)
    args = ap.parse_args(argv)
    with open(args.token_file, "r", encoding="utf-8") as f:
        secret = f.read().strip()
    if not secret:
        raise SystemExit("令牌文件为空")
    web.run_app(make_app(f"http://127.0.0.1:{args.upstream_port}", secret),
                host="127.0.0.1", port=args.listen_port, access_log=None, print=None)


if __name__ == "__main__":
    main()
