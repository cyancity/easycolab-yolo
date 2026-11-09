import os
import time
import socket
import subprocess
import threading
from typing import Optional, List, Tuple, Dict, Any
import json
import shutil
import urllib.parse
import urllib.request
import torch
from .config import MODEL_DOWNLOAD_STATE


def _make_noproxy_opener():
    """本地回环请求 (127.0.0.1 llama-server) 强制绕过环境变量代理"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _build_llama_ld_env(install_dir: str) -> Dict[str, str]:
    """收集 llama.cpp 动态库目录 + PyTorch 内置 NVIDIA CUDA Runtime/cuBLAS 目录"""
    env = os.environ.copy()
    ld_dirs: List[str] = []
    if os.path.isdir(install_dir):
        for root, _, files in os.walk(install_dir):
            if any(f == "llama-server" or ".so" in f for f in files):
                if root not in ld_dirs:
                    ld_dirs.append(root)

    try:
        import site
        site_roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:
        site_roots = []
    site_roots.append(os.path.dirname(os.path.dirname(torch.__file__)))

    for sp in site_roots:
        nv_root = os.path.join(sp, "nvidia")
        if os.path.isdir(nv_root):
            for sub in os.listdir(nv_root):
                lib_d = os.path.join(nv_root, sub, "lib")
                if os.path.isdir(lib_d) and lib_d not in ld_dirs:
                    ld_dirs.append(lib_d)

    for sys_lib in ("/usr/lib64-nvidia", "/usr/local/nvidia/lib64", "/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"):
        if os.path.isdir(sys_lib) and sys_lib not in ld_dirs:
            ld_dirs.append(sys_lib)

    prev_ld = env.get("LD_LIBRARY_PATH", "")
    if prev_ld:
        ld_dirs.append(prev_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_dirs)
    return env


def _download_file_to_path(url: str, dst_path: str, desc: str = ""):
    req = urllib.request.Request(url, headers={"User-Agent": "EasyColab-Engine/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst_path, "wb") as wf:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            wf.write(chunk)
            done += len(chunk)
            if desc and total > 0:
                pct = int(done * 100 / total)
                MODEL_DOWNLOAD_STATE["message"] = f"{desc} ({done/(1024**2):.0f}/{total/(1024**2):.0f} MB · {pct}%)"


def _ensure_llama_server_binary() -> Tuple[str, Dict[str, str]]:
    """
    自动安装/校验官方 ggml-org/llama.cpp Linux x64 CUDA 12.8 预编译 llama-server + cudart 二进制。
    固定使用 ubuntu-cuda-12.x（NVIDIA 驱动向下兼容 CUDA 12.x/13.0/13.5+ 全系驱动，避免 CUDA 13.5 PTX 在 Colab 13.0 驱动上报 unsupported toolchain）。
    """
    install_dir = (
        "/content/llama_cpp_cuda"
        if os.path.exists("/content")
        else os.path.join(os.getcwd(), "llama_cpp_cuda")
    )
    os.makedirs(install_dir, exist_ok=True)

    def _find_server_bin() -> Optional[str]:
        for root, _, files in os.walk(install_dir):
            if "llama-server" in files:
                cand = os.path.join(root, "llama-server")
                try:
                    os.chmod(cand, 0o755)
                except Exception:
                    pass
                return cand
        return None

    def _has_cudart12() -> bool:
        for _, _, files in os.walk(install_dir):
            if any(f.startswith("libcudart.so.12") or f.startswith("libcublas.so.12") for f in files):
                return True
        return False

    def _verify_server_bin(bin_p: str, run_env: Dict[str, str]) -> Tuple[bool, str]:
        try:
            proc = subprocess.run(
                [bin_p, "--version"],
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=15,
            )
            if proc.returncode != 0:
                return False, (proc.stdout or "").strip()
            ver_txt = (proc.stdout or "").strip()
            if torch.cuda.is_available():
                dev_proc = subprocess.run(
                    [bin_p, "--list-devices"],
                    env=run_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=15,
                )
                dev_txt = (dev_proc.stdout or "").strip()
                if dev_proc.returncode != 0 or "CUDA0" not in dev_txt:
                    return False, f"CUDA 设备未挂载 ({dev_txt[:200]})"
            return True, ver_txt
        except Exception as ex:
            return False, str(ex)

    existing_bin = _find_server_bin()
    if existing_bin and _has_cudart12():
        env = _build_llama_ld_env(install_dir)
        ok, _ = _verify_server_bin(existing_bin, env)
        if ok:
            return existing_bin, env

    shutil.rmtree(install_dir, ignore_errors=True)
    os.makedirs(install_dir, exist_ok=True)

    cuda_prefix = "ubuntu-cuda-12."
    fallback_tag = "b12180"
    main_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"
    cudart_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/cudart-llama-{fallback_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"

    try:
        api_req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=8",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "EasyColab-Engine/1.0"},
        )
        with urllib.request.urlopen(api_req, timeout=10) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
        for rel in releases:
            tag = str(rel.get("tag_name") or "")
            if not tag.startswith("b"):
                continue
            found_main, found_rt = None, None
            for asset in rel.get("assets") or []:
                aname = str(asset.get("name") or "")
                aurl = str(asset.get("browser_download_url") or "")
                if not aurl or not aname.endswith("-x64.tar.gz"):
                    continue
                if aname.startswith("llama-") and cuda_prefix in aname:
                    found_main = aurl
                elif aname.startswith("cudart-llama-") and cuda_prefix in aname:
                    found_rt = aurl
            if found_main:
                main_url = found_main
                if found_rt:
                    cudart_url = found_rt
                break
    except Exception as api_err:
        print(f"[GGUF 引擎] GitHub Releases 查询跳过，使用稳定镜像版本 {fallback_tag} ({api_err})")

    print(f"[GGUF 引擎] 正在下载官方预编译 llama.cpp CUDA 12 推理后端: {os.path.basename(main_url)} ...")
    tar_path = os.path.join(install_dir, "llama_cuda_bin.tar.gz")
    _download_file_to_path(main_url, tar_path, desc="正在下载 llama.cpp CUDA 引擎")
    subprocess.check_call(["tar", "-xzf", tar_path, "-C", install_dir])
    try:
        os.remove(tar_path)
    except Exception:
        pass

    if cudart_url:
        print(f"[GGUF 引擎] 正在下载配套 CUDA 12 Runtime 动态库 ({os.path.basename(cudart_url)})...")
        rt_tar = os.path.join(install_dir, "cudart_bin.tar.gz")
        _download_file_to_path(cudart_url, rt_tar, desc="正在下载 CUDA Runtime 动态库")
        subprocess.check_call(["tar", "-xzf", rt_tar, "-C", install_dir])
        try:
            os.remove(rt_tar)
        except Exception:
            pass

    bin_path = _find_server_bin()
    if not bin_path:
        raise RuntimeError("解压 llama.cpp CUDA 归档后未找到 llama-server 可执行文件")

    env = _build_llama_ld_env(install_dir)
    ok, ver_out = _verify_server_bin(bin_path, env)
    if not ok:
        raise RuntimeError(f"llama-server CUDA 启动自检失败: {ver_out[:300]}")

    first_line = ver_out.splitlines()[0] if ver_out else "ready"
    print(f"[GGUF 引擎] llama-server CUDA 引擎校验通过 ({first_line})")
    return bin_path, env


def _pick_free_local_port() -> int:
    """在非临时端口段 (18080~18999) 选取空闲本地监听端口，规避 Colab 容器 ephemeral 端口段 (32768+) 抢占冲突。"""
    for cand in range(18080, 18999):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", cand))
                return cand
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _start_qwen_gguf_server(
    gguf_path: str,
    mmproj_path: Optional[str],
    ctx_len: int,
    cur_free_gb: float,
    mtp_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    启动驻留 GPU 的 llama-server 实例加载 Qwen3.8-27B-UD-Q4_K_M.gguf + mmproj-F16.gguf (+ 可选 MTP 投机解码头)。
    集成 Unsloth Native MTP 投机解码 (--spec-type draft-mtp)、GPU 端采样 (--backend-sampling) 与 q8_0/q4_0 高带宽 KV Cache。
    """
    bin_path, env = _ensure_llama_server_binary()

    # Qwen3.8-27B 仅 16 层 Attention 产生 KV Cache，256K 满长下: q8_0≈8.8GB (无损困惑度，较 f16 省 7.2GB 显存且带宽翻倍), q4_0≈4.6GB
    q8_kv_gb = (ctx_len / 262144.0) * 8.8
    q4_kv_gb = (ctx_len / 262144.0) * 4.6

    if cur_free_gb >= 19.2 + q8_kv_gb:
        primary_kv = "q8_0"
        primary_kv_gb = q8_kv_gb
    else:
        primary_kv = "q4_0"
        primary_kv_gb = q4_kv_gb

    has_mtp_file = bool(mtp_path and os.path.exists(mtp_path))

    # 计算精确 -ngl 层数并显式 --fit off，规避 llama.cpp 在同时传入 --mmproj 与大 -c 时 --fit on 的异常
    if cur_free_gb >= (19.0 + primary_kv_gb):
        primary_ngl = 999
    else:
        avail_for_weights = max(4.0, cur_free_gb - primary_kv_gb - 2.4)
        primary_ngl = max(12, min(64, int(avail_for_weights / 0.25)))

    primary_mtp = has_mtp_file and primary_ngl == 999 and (cur_free_gb >= 20.6 + primary_kv_gb)
    attempts: List[Tuple[str, int, bool]] = [(primary_kv, primary_ngl, primary_mtp)]

    # 若 q8_0 下显存刚好差一点装 MTP，自动尝试 q4_0 KV + MTP 全量 GPU 驻留
    if has_mtp_file and not primary_mtp and (cur_free_gb >= 20.6 + q4_kv_gb):
        attempts.insert(0, ("q4_0", 999, True))

    if primary_ngl == 999:
        if primary_mtp:
            attempts.append((primary_kv, 999, False))
        fallback_ngl = max(16, min(64, int(max(4.0, cur_free_gb - q4_kv_gb - 2.8) / 0.25)))
        attempts.append(("q4_0", fallback_ngl, False))

    opener = _make_noproxy_opener()
    last_err = ""

    for attempt_idx, (kv_type, ngl_val, use_mtp) in enumerate(attempts):
        port = _pick_free_local_port()
        cmd = [
            bin_path,
            "-m", gguf_path,
            "--fit", "off",
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(ctx_len),
            "--kv-unified",
            "--parallel", "1",
            "--flash-attn", "on",
            "--cache-type-k", kv_type,
            "--cache-type-v", kv_type,
            "-b", "2048",
            "-ub", "1024",
            "--backend-sampling",
            "--samplers", "top_k;top_p;min_p;temperature",
            "--temp", "1.0",
            "--top-p", "0.95",
            "--top-k", "20",
            "--min-p", "0.0",
            "--repeat-penalty", "1.0",
            "--presence-penalty", "0.0",
            "--jinja",
            "--reasoning-format", "deepseek",
            "-ngl", str(ngl_val),
        ]
        if use_mtp and mtp_path:
            cmd.extend([
                "--model-draft", mtp_path,
                "-ngld", "999",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "4",
                "--spec-draft-p-min", "0.45",
                "--cache-type-k-draft", kv_type,
                "--cache-type-v-draft", kv_type,
            ])
        if mmproj_path and os.path.exists(mmproj_path):
            cmd.extend(["--mmproj", mmproj_path, "--image-min-tokens", "1024"])

        fit_desc = "全量 GPU 驻留" if ngl_val >= 99 else f"混合驻留 ({ngl_val}/65 层 GPU)"
        mtp_desc = " + MTP 多词元投机解码" if use_mtp else ""
        print(
            f"[GGUF 引擎] 正在启动 Qwen3.8-27B-UD-Q4_K_M{mtp_desc} 服务 "
            f"(上下文 {ctx_len // 1024}K={ctx_len}, KV={kv_type}, GPU采样, {fit_desc}, port={port})..."
        )
        MODEL_DOWNLOAD_STATE["progress"] = 0.55
        MODEL_DOWNLOAD_STATE["message"] = (
            f"正在将 Qwen3.8-27B-UD-Q4_K_M{mtp_desc} 装载至 GPU ({ctx_len // 1024}K 上下文 · KV {kv_type})..."
        )

        recent_logs: List[str] = []
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _drain_stdout(p: subprocess.Popen, log_buf: List[str]):
            try:
                assert p.stdout is not None
                for raw_line in p.stdout:
                    ln = raw_line.rstrip("\r\n")
                    if not ln:
                        continue
                    log_buf.append(ln)
                    if len(log_buf) > 100:
                        del log_buf[:25]
            except Exception:
                pass

        threading.Thread(target=_drain_stdout, args=(proc, recent_logs), daemon=True).start()

        health_url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + 210.0
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with opener.open(health_url, timeout=2.0) as h_resp:
                    if h_resp.status == 200:
                        ready = True
                        break
            except urllib.error.HTTPError as he:
                # 503 表示 llama-server 正在加载权重到显存
                if he.code == 503:
                    MODEL_DOWNLOAD_STATE["progress"] = min(0.95, round(MODEL_DOWNLOAD_STATE["progress"] + 0.01, 2))
            except Exception:
                pass
            time.sleep(0.5)

        if ready:
            ctx_k_str = f"{ctx_len // 1024}K"
            quant_tag = "UD-Q4_K_M+MTP" if use_mtp else "UD-Q4_K_M"
            mode_label = f"Qwen3.8-27B · {quant_tag} · {ctx_k_str} (KV:{kv_type})"
            print(f"[GGUF 引擎] Qwen3.8-27B-UD-Q4_K_M 已就绪 ({mode_label})")
            return {
                "backend": "gguf",
                "model_id": "qwen38_27b_gguf",
                "port": port,
                "base_url": f"http://127.0.0.1:{port}",
                "server_proc": proc,
                "context_length": ctx_len,
                "kv_type": kv_type,
                "mtp_enabled": use_mtp,
                "mode_label": mode_label,
                "gguf_path": gguf_path,
                "mmproj_path": mmproj_path,
                "mtp_path": mtp_path if use_mtp else None,
            }

        # 未就绪：清理子进程并记录日志
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        tail_txt = "\n".join(recent_logs[-12:])
        last_err = tail_txt or "llama-server 启动超时"
        if attempt_idx + 1 < len(attempts):
            print(f"[GGUF 引擎] 当前配置 (KV={kv_type}, ngl={ngl_val}, MTP={use_mtp}) 显存触顶，自动降级重试...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.5)

    raise RuntimeError(f"Qwen3.8-27B-GGUF 服务启动失败: {last_err[-400:]}")
