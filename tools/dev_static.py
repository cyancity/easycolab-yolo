"""前端开发服务：python tools/dev_static.py [端口] [后端地址]

从本仓库提供 workbench/static/，/api 和 /v1 转发给已在运行的工作台后端（默认 127.0.0.1:8790）。
用于在不重启后端的情况下看前端改动。转发的写操作会真实生效，点部署、停止前先确认；
只想看界面时在 URL 上加 ?mock=daily 用样例数据。
"""
import http.server
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent / "workbench" / "static"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
BACKEND = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8790"
FORWARD_HEADERS = ("Content-Type", "Authorization", "Accept")


class Handler(http.server.SimpleHTTPRequestHandler):
    # Windows 注册表可能把 .js 映射成 text/plain，ES module 会拒绝加载
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".svg": "image/svg+xml",
        ".webmanifest": "application/manifest+json",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _proxied(self):
        return self.path.startswith("/api/") or self.path.startswith("/v1/")

    def _forward(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(BACKEND + self.path, data=body, method=self.command)
        for h in FORWARD_HEADERS:
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                status, headers, data = r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            status, headers, data = e.code, e.headers, e.read()
        except OSError as e:
            status, headers, data = 502, {}, f'{{"detail": "后端未响应：{e}"}}'.encode()
        self.send_response(status)
        self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self._proxied():
            return self._forward()
        if self.path.split("?")[0] == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self._proxied():
            return self._forward()
        self.send_error(405)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
