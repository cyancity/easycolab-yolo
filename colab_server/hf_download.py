import os
import sys
import time
import threading
from typing import Tuple
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


# ==============================================================================
# 借鉴 comfy-colab-kit 的 8 线程 HTTP Range 并行极速下载器 (400~700 MB/s)
# ==============================================================================
class _StripAuthRedirectHandler(urllib.request.HTTPRedirectHandler):
    """重定向到 HuggingFace CDN (CloudFront/Xet) 时剥离 Authorization 头，防止 400/403 拒绝"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None and "Authorization" in new_req.unredirected_hdrs:
            del new_req.unredirected_hdrs["Authorization"]
        if new_req is not None and "Authorization" in new_req.headers:
            del new_req.headers["Authorization"]
        return new_req


def fast_parallel_hf_download(
    repo_id: str,
    filename: str,
    dest_dir: str,
    hf_token: str = "",
    num_connections: int = 8,
) -> str:
    """
    使用 8 路并发 HTTP Range 连接直接从 HuggingFace CDN 下载大权重文件至内存盘/本地盘，
    规避单连接 ~60MB/s 限速，在 Colab 上可达 400~700 MB/s。
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"

    url = f"https://huggingface.co/{repo_id}/resolve/main/{urllib.parse.quote(filename, safe='/')}"
    opener = urllib.request.build_opener(_StripAuthRedirectHandler())
    headers = {"User-Agent": "EasyColab/2.0"}
    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        # Probe final CDN URL & Content-Length via Range: bytes=0-0
        probe_req = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-0"})
        with opener.open(probe_req, timeout=20) as resp:
            cdn_url = resp.geturl()
            content_range = resp.headers.get("Content-Range", "")
            if "/" in content_range:
                total_size = int(content_range.split("/")[-1])
            else:
                total_size = int(resp.headers.get("Content-Length", 0))

        # 若文件较小 (<32MB) 或未返回支持分片的大小，直接走 hf_hub_download
        if total_size < 32 * 1024 * 1024:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)

        with open(temp_path, "wb") as f:
            f.truncate(total_size)

        downloaded_bytes = 0
        dl_lock = threading.Lock()
        start_ts = time.time()
        last_print_ts = [0.0]

        part_size = (total_size + num_connections - 1) // num_connections
        ranges = [
            (i * part_size, min((i + 1) * part_size - 1, total_size - 1))
            for i in range(num_connections)
            if i * part_size < total_size
        ]

        def _download_chunk(byte_range: Tuple[int, int]):
            nonlocal downloaded_bytes
            start_b, end_b = byte_range
            chunk_req = urllib.request.Request(
                cdn_url,
                headers={"User-Agent": "EasyColab/2.0", "Range": f"bytes={start_b}-{end_b}"},
            )
            with urllib.request.urlopen(chunk_req, timeout=60) as r, open(temp_path, "r+b") as wf:
                wf.seek(start_b)
                while True:
                    buf = r.read(4 * 1024 * 1024)  # 4MB buffer
                    if not buf:
                        break
                    wf.write(buf)
                    now = time.time()
                    with dl_lock:
                        downloaded_bytes += len(buf)
                        if now - last_print_ts[0] >= 0.35 or downloaded_bytes >= total_size:
                            last_print_ts[0] = now
                            elapsed = max(0.001, now - start_ts)
                            speed_mbs = (downloaded_bytes / (1024 ** 2)) / elapsed
                            pct = int(downloaded_bytes * 100 / total_size)
                            cur_gb = downloaded_bytes / (1024 ** 3)
                            tot_gb = total_size / (1024 ** 3)
                            eta_s = int(((total_size - downloaded_bytes) / (1024 ** 2)) / max(0.1, speed_mbs))
                            sys.stdout.write(
                                f"\r{filename}: {pct}%|{'█' * (pct // 10):<10}| "
                                f"{cur_gb:.2f}G/{tot_gb:.2f}G [{int(elapsed):02d}s<{eta_s:02d}s, {speed_mbs:.1f}MB/s]\n"
                            )
                            sys.stdout.flush()

        with ThreadPoolExecutor(max_workers=num_connections) as pool:
            futures = [pool.submit(_download_chunk, r) for r in ranges]
            for fut in as_completed(futures):
                fut.result()

        os.replace(temp_path, dest_path)
        return dest_path

    except Exception as e:
        print(f"ℹ️ 多线程分片直连回退至标准 HF 下载 ({e})...")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)
