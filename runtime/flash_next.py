#!/usr/bin/env python3
"""Serve Qwen3.8-Flash-Next on one Colab GPU through llama.cpp `llama-server`.

Goal: ctx >= 300K (target 512K), single-stream decode >= 100 tok/s (target 150),
VRAM use >= 75%.  Targets: Colab G4 (RTX PRO 6000 Blackwell 96GB), H100 80GB, A100 80GB.
Only the standard library and huggingface_hub are used.  Units: GiB = 2**30 bytes;
pass nvidia-smi memory.total / 1024 as the GPU size.  "K" in ctx_k = 1024 tokens.

VERIFIED (read on 2026-11-10)
  * HF repo `unsloth/Qwen3.8-Flash-Next-GGUF`, file tree via
    https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main?recursive=true
    - shards `<Q>/Qwen3.8-Flash-Next-<Q>-0000i-of-0000n.gguf`; sizes in QUANTS below are from this tree.
    - the engram table is one whole shard (IQ4_NL 28.8 GB for the UD-IQ3_XXS..UD-Q4_K_XL quants,
      Q8_0 54.4 GB for UD-Q5_K_XL/UD-Q6_K_XL/Q8_0); other tables: BF16/Q8_0/Q5_1/IQ4_NL
      in `engram/` plus `engram/set_split.py` (not used here).
    - MTP heads in `MTP/`: `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (4,137,429,120 B, full head) and
      `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2,786,568,256 B, reuses main embeddings).
  * llama.cpp: https://github.com/danielhanchen/llama.cpp branch `qwen4exp/mtp` exists
    (head 6fcaa16f4b36 on 2026-09-21, api.github.com/repos/danielhanchen/llama.cpp/branches/qwen4exp/mtp).
    Upstream https://github.com/ggml-org/llama.cpp merged "Qwen4Exp: add MTP" (PR #29761, 2026-10-01).
    Upstream does NOT load the `shared` MTP head (missing token_embd.weight), so the full head is the default.
  * Flags (https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): -fa [on|off|auto],
    -ctk/-ctv (q8_0 allowed), --rope-scaling yarn, --rope-scale N, --yarn-orig-ctx N, -ot/--override-tensor
    PATTERN=BUFTYPE, --jinja, --metrics, -a/--alias, -np/--parallel, -md/--model-draft,
    --spec-type draft-mtp, --spec-draft-n-max.
  * Architecture (https://huggingface.co/Qwen/Qwen3.8-Flash-Next): 48 layers = 12 x (3 Gated DeltaNet + 1 QSA
    full-attention); QSA has 2 KV heads, head dim 256, indexer dim 128; native ctx 262,144;
    YaRN factor 4.0 example, factor 2.0 suggested for 524,288 tokens.
  * Unsloth guide https://unsloth.ai/docs/models/qwen3.8-next : ~100 tok/s baseline, ~170 with MTP on 1x RTX
    PRO 6000; sampling temp 1.0 / top-p 0.95 / top-k 20 / min-p 0.
    https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/raw/main/MTP/README.md recommends n-max 2
    (guide example uses 5); MTP gains drop with concurrency, so --parallel 1.
  * `-ot per_layer_token_embd.weight=CPU` appears in user commands on ggml-org/llama.cpp PR #28243.

UNVERIFIED (assumed, check on first real run)
  1. That `per_layer_token_embd.weight` is the 51B engram/n-gram table.  If the pattern matches nothing
     llama.cpp loads it on the GPU (OOM).  Check the log line `CPU model buffer size` is ~28 GB.
  2. That `-fa on` with `-ctk/-ctv q8_0` works on the QSA sparse-attention path, and that the CLI YaRN
     flags override the GGUF rope settings for arch qwen4exp.
  3. Whether the danielhanchen branch (Sep 21) is newer than what the Unsloth GGUFs need, vs upstream
     master (`--llama-source upstream`).  Full MTP head working with the unsloth main GGUFs is assumed.
  4. KV bytes per token: the indexer key cache (128 dim per QSA layer), MTP draft KV and compute buffers
     are estimates (see constants).  Real VRAM must be read from nvidia-smi.
  5. nvcc / cmake being present on the Colab G4 image; CMAKE_CUDA_ARCHITECTURES=120 for Blackwell
     (CUDA >= 12.8 needed).  `-DLLAMA_CURL=OFF` is only a hint to older trees.
  6. Decode speed (>=100 / 150 tok/s) is not modelled; only measured numbers count.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import io
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request

LOG = "FN_LOG:"
GIB = 2 ** 30
HF_REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
MODEL_DIRNAME = "unsloth/Qwen3.8-Flash-Next-GGUF"  # = <models root>/<repo id>, the layout app.py uses for MODEL_CATALOG
SOURCES = {
    "fork": ("https://github.com/danielhanchen/llama.cpp", "qwen4exp/mtp"),
    "upstream": ("https://github.com/ggml-org/llama.cpp", "master"),
}
ENGRAM_TENSOR_RE = r"per_layer_token_embd\.weight=CPU"  # UNVERIFIED #1
NATIVE_CTX = 262144
ALIAS = "qwen3.8-flash-next"

# (shard sizes in bytes in file order, 1-based index of the engram shard). Source: HF tree (see docstring).
_H = 10946618
QUANTS = {
    "UD-IQ1_S": ((_H, 37568095648, 28800138432), 3),
    "UD-IQ1_M": ((_H, 39196009888, 28800138432), 3),
    "UD-IQ2_XS": ((_H, 47021008288, 28800138432), 3),
    "UD-Q2_0": ((_H, 41738806688, 28800138432), 3),
    "UD-Q2_K_XL": ((_H, 49732486272, 1037181344, 28800138432), 4),
    "UD-IQ3_XXS": ((_H, 49946027488, 4869325888, 28800138432), 4),
    "UD-Q3_K_XL": ((_H, 49742117248, 10911746208, 28800138432), 4),
    "UD-IQ4_XS": ((_H, 49604491008, 15256783680, 28800138432), 4),
    "UD-Q4_K_XL": ((_H, 49720539552, 32803030176, 28800138432), 4),
    "UD-Q5_K_XL": ((_H, 682434912, 54400261312, 49990245824, 49882416192, 3320101792), 3),
    "UD-Q6_K_XL": ((10946624, 682434912, 54400261312, 49814108512, 49419317344, 14838313984), 3),
    "Q8_0": ((10946624, 682434912, 54400261312, 49446841216, 49668930400, 34015618784), 3),
}
# MTP head files: variant -> (repo path, bytes).
MTP_FILES = {
    "full": ("MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf", 4137429120),
    "shared": ("MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", 2786568256),
}

# --- KV / runtime model (all ESTIMATES except the architecture numbers) ---------------------------------
# Source: Qwen model card https://huggingface.co/Qwen/Qwen3.8-Flash-Next (48 layers = 12 x (3 DeltaNet + 1 QSA)).
FULL_ATTN_LAYERS = 12
KV_HEADS = 2            # QSA KV heads (model card)
HEAD_DIM = 256          # QSA head dim (model card)
INDEXER_DIM = 128       # QSA indexer dim (model card); ASSUMED to be cached once per token per layer
# K + V elements per token per full-attention layer, plus the assumed indexer key cache.
KV_ELEMS_PER_TOKEN = FULL_ATTN_LAYERS * (2 * KV_HEADS * HEAD_DIM + INDEXER_DIM)
MTP_KV_ELEMS_PER_TOKEN = 2 * KV_HEADS * HEAD_DIM  # draft block: 1 dense attention layer (PR #27836); ASSUMED
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q4_0": 18 / 32}  # ggml block sizes
DELTANET_LAYERS = 36
DELTANET_STATE_GIB = DELTANET_LAYERS * 48 * 128 * 128 * 4 / GIB  # 48 V heads x 128 x 128 fp32 per layer; ASSUMED
RUNTIME_BUFFERS_GIB = 3.0   # CUDA context + compute buffers at -ub 1024; ASSUMED
MTP_BUFFERS_GIB = 0.5       # draft compute buffers; ASSUMED
HEADROOM_FRAC = 0.05        # keep 5% of VRAM free
VRAM_TARGET = 0.75
EST = "ESTIMATE"


_LOG_HOOK = None


def set_log_hook(fn) -> None:
    """Call fn(msg) for every FN_LOG line (app.py shows it as the task message). Pass None to clear."""
    global _LOG_HOOK
    _LOG_HOOK = fn


def log(msg: str) -> None:
    print(f"{LOG} {msg}", flush=True)
    if _LOG_HOOK is not None:
        try:
            _LOG_HOOK(msg)
        except Exception:
            pass


def _shards(quant: str) -> list[tuple[str, int]]:
    if quant not in QUANTS:
        raise ValueError(f"unknown quant {quant!r}; choose from {sorted(QUANTS)}")
    sizes, _ = QUANTS[quant]
    n = len(sizes)
    return [(f"{quant}/Qwen3.8-Flash-Next-{quant}-{i + 1:05d}-of-{n:05d}.gguf", s) for i, s in enumerate(sizes)]


def weight_files(quant: str, mtp: bool, mtp_variant: str = "full") -> list[tuple[str, int]]:
    """Repo path and byte size of every file ensure_weights needs."""
    files = _shards(quant)
    if mtp:
        files.append(MTP_FILES[mtp_variant])
    return files


def plan(gpu_vram_gb: float, ctx_k: int, quant: str, kv: str, mtp: bool, mtp_variant: str = "full") -> dict:
    """VRAM budget in GiB.  Weights are file sizes (exact); everything else is an estimate."""
    if kv not in KV_BYTES:
        raise ValueError(f"unknown kv type {kv!r}; choose from {sorted(KV_BYTES)}")
    shards = _shards(quant)
    engram = shards[QUANTS[quant][1] - 1][1]
    weights_gpu = (sum(s for _, s in shards) - engram) / GIB
    tokens = ctx_k * 1024
    kv_gib = tokens * KV_ELEMS_PER_TOKEN * KV_BYTES[kv] / GIB
    mtp_w = MTP_FILES[mtp_variant][1] / GIB if mtp else 0.0
    mtp_kv = tokens * MTP_KV_ELEMS_PER_TOKEN * KV_BYTES[kv] / GIB if mtp else 0.0
    mtp_buf = MTP_BUFFERS_GIB if mtp else 0.0
    parts = {
        "weights_gpu_exact": weights_gpu,
        f"kv_cache_{EST}": kv_gib,
        "mtp_weights_exact": mtp_w,
        f"mtp_kv_{EST}": mtp_kv,
        f"mtp_buffers_{EST}": mtp_buf,
        f"deltanet_state_{EST}": DELTANET_STATE_GIB,
        f"runtime_buffers_{EST}": RUNTIME_BUFFERS_GIB,
    }
    used = sum(parts.values())
    fill = used / gpu_vram_gb
    notes = [
        f"engram shard {engram / GIB:.1f} GiB is kept in CPU RAM (-ot), needs that much host RAM/page cache",
        "KV counts the 12 full-attention layers only; DeltaNet state is constant per slot",
        "numbers marked ESTIMATE are not measured; check nvidia-smi after start",
    ]
    if tokens > NATIVE_CTX:
        notes.append(f"ctx {tokens} > native {NATIVE_CTX}: static YaRN factor {tokens / NATIVE_CTX:.2f} "
                     "(may slightly hurt short-text quality)")
    if tokens < 300 * 1024:
        notes.append("ctx below the 300K requirement")
    if fill < VRAM_TARGET:
        notes.append(f"fill {fill:.0%} < {VRAM_TARGET:.0%} target; a larger quant or KV type would use more VRAM")
    if used > gpu_vram_gb * (1 - HEADROOM_FRAC):
        notes.append("does not fit with 5% headroom: use a smaller quant, kv q8_0/q4_0 or lower ctx")
    free = gpu_vram_gb * (1 - HEADROOM_FRAC) - used
    if free > engram / GIB:
        notes.append("enough spare VRAM to put the engram table on the GPU as well (drop the -ot flag)")
    return {
        "gpu_vram_gib": gpu_vram_gb, "ctx_k": ctx_k, "ctx_tokens": tokens, "quant": quant, "kv": kv, "mtp": mtp,
        "components_gib": {k: round(v, 2) for k, v in parts.items()},
        "used_gib": round(used, 2), "fill_pct": round(fill * 100, 1),
        "fits": used <= gpu_vram_gb * (1 - HEADROOM_FRAC), "meets_vram_target": fill >= VRAM_TARGET,
        "engram_cpu_gib": round(engram / GIB, 2), "notes": notes,
    }


# --- llama.cpp build --------------------------------------------------------------------------------------
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kw)


def detect_sm() -> str:
    """Compute capability from nvidia-smi, e.g. '12.0' -> '120'."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.split()[0]
    return out.replace(".", "")


def detect_vram_gib() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split()[0]
    return float(out) / 1024


def _remote_commit(url: str, branch: str) -> str | None:
    try:
        out = subprocess.run(["git", "ls-remote", url, f"refs/heads/{branch}"], capture_output=True,
                             text=True, check=True, timeout=60).stdout.split()
        return out[0] if out else None
    except (subprocess.SubprocessError, OSError):
        return None


def _extract(tarball: str, name: str) -> str:
    """Extract to a local dir; Drive mounts do not keep exec bits."""
    dest = os.path.join("/tmp" if os.path.isdir("/tmp") else os.path.dirname(tarball), "fn_bin", name)
    exe = os.path.join(dest, "llama-server")
    if not os.path.isfile(exe):
        os.makedirs(dest, exist_ok=True)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(dest)
        os.chmod(exe, 0o755)
    return exe


def ensure_llama_server(cache_dir: str, source: str = "fork", sm: str | None = None) -> str:
    """Return a path to llama-server; build once, then reuse the tarball under cache_dir/llama.cpp/."""
    url, branch = SOURCES[source]
    sm = sm or detect_sm()
    cdir = os.path.join(cache_dir, "llama.cpp")
    os.makedirs(cdir, exist_ok=True)
    commit = _remote_commit(url, branch)
    if commit is None:  # offline: newest cached build for this arch
        cached = sorted((f for f in os.listdir(cdir) if f.endswith(f"-sm{sm}.tar.gz")),
                        key=lambda f: os.path.getmtime(os.path.join(cdir, f)))
        if not cached:
            raise RuntimeError(f"cannot reach {url} and no cached build for sm{sm}")
        log(f"git unreachable, using cached build {cached[-1]}")
        return _extract(os.path.join(cdir, cached[-1]), cached[-1][:-7])
    name = f"{commit[:12]}-sm{sm}"
    tarball = os.path.join(cdir, f"{name}.tar.gz")
    if os.path.isfile(tarball):
        log(f"llama-server cache hit {name}")
        return _extract(tarball, name)
    for tool in ("git", "cmake", "nvcc"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found; needed to build llama-server (UNVERIFIED #5)")
    src = f"/tmp/fn_build/{name}"
    shutil.rmtree(src, ignore_errors=True)
    log(f"cloning {url} {branch}")
    _run(["git", "clone", "--depth", "1", "--branch", branch, url, src])
    log(f"cmake configure sm{sm}")
    _run(["cmake", "-S", src, "-B", f"{src}/build", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={sm}",
          "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"])
    log("building llama-server (about 10-25 min, first start only)")
    proc = subprocess.Popen(["cmake", "--build", f"{src}/build", "--target", "llama-server",
                             "-j", str(os.cpu_count() or 4)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last, tail = -10, []
    for line in proc.stdout:
        tail = (tail + [line.rstrip()])[-30:]
        m = re.match(r"\[\s*(\d+)%\]", line)
        if m and int(m.group(1)) >= last + 10:
            last = int(m.group(1))
            log(f"build {last}%")
    if proc.wait() != 0:
        raise RuntimeError("llama-server build failed:\n" + "\n".join(tail))
    exe = f"{src}/build/bin/llama-server"
    tmp = tarball + ".part"
    with tarfile.open(tmp, "w:gz") as tf:
        tf.add(exe, arcname="llama-server")
    os.replace(tmp, tarball)
    log(f"cached {tarball}")
    return _extract(tarball, name)


# --- weights ----------------------------------------------------------------------------------------------
def _downloaded_bytes(root: str, files: list[tuple[str, int]]) -> int:
    """Finished files plus huggingface_hub's `.incomplete` partials (local_dir/.cache/huggingface/download/)."""
    done = 0
    for rel, size in files:
        dest = os.path.join(root, *rel.split("/"))
        if os.path.isfile(dest):
            done += min(os.path.getsize(dest), size)
            continue
        part = os.path.join(root, ".cache", "huggingface", "download", *rel.split("/"))
        done += max([min(os.path.getsize(f), size) for f in glob.glob(glob.escape(part) + ".*.incomplete")] or [0])
    return done


def ensure_weights(cache_dir: str, quant: str, mtp: bool, mtp_variant: str = "full", progress=None) -> dict:
    """Download missing files into cache_dir/<MODEL_DIRNAME>/ and return their paths.

    progress(fraction, text) is called every ~2 s from a helper thread while files download."""
    root = os.path.join(cache_dir, MODEL_DIRNAME)
    os.makedirs(root, exist_ok=True)
    stop = threading.Event()
    if progress is not None:
        files = weight_files(quant, mtp, mtp_variant)
        total = sum(s for _, s in files)

        def _poll():
            while not stop.wait(2.0):
                done = _downloaded_bytes(root, files)
                progress(done / total, f"{done / 1e9:.1f} / {total / 1e9:.1f} GB")
        threading.Thread(target=_poll, daemon=True).start()
    try:
        return _ensure_weights(root, quant, mtp, mtp_variant)
    finally:
        stop.set()


def _ensure_weights(root: str, quant: str, mtp: bool, mtp_variant: str) -> dict:
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # set before huggingface_hub is imported
    from huggingface_hub import hf_hub_download
    paths = {}
    for rel, size in weight_files(quant, mtp, mtp_variant):
        dest = os.path.join(root, *rel.split("/"))
        if os.path.isfile(dest) and os.path.getsize(dest) == size:
            log(f"have {rel}")
        else:
            log(f"downloading {rel} ({size / 1e9:.1f} GB)")
            hf_hub_download(HF_REPO, rel, local_dir=root)
            if os.path.getsize(dest) != size:
                raise RuntimeError(f"{rel}: size {os.path.getsize(dest)} != expected {size}")
        paths[rel] = dest
    shard_paths = [paths[r] for r, _ in _shards(quant)]
    return {"root": root, "model": shard_paths[0], "shards": shard_paths,
            "mtp": paths[MTP_FILES[mtp_variant][0]] if mtp else None}


# --- command / process ------------------------------------------------------------------------------------
def build_command(paths: dict, ctx_k: int, kv: str, mtp: bool, port: int, draft_n: int = 2,
                  host: str = "127.0.0.1", binary: str = "llama-server", threads: int | None = None) -> list[str]:
    tokens = ctx_k * 1024
    cmd = [binary, "-m", paths["model"], "--alias", ALIAS, "--host", host, "--port", str(port),
           "-c", str(tokens), "-np", "1", "-ngl", "999", "-ot", ENGRAM_TENSOR_RE,
           "-fa", "on", "-ctk", kv, "-ctv", kv, "-b", "2048", "-ub", "1024",
           "--jinja", "--metrics",
           "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0"]
    if threads:
        cmd += ["-t", str(threads)]
    if tokens > NATIVE_CTX:
        cmd += ["--rope-scaling", "yarn", "--rope-scale", f"{tokens / NATIVE_CTX:g}",
                "--yarn-orig-ctx", str(NATIVE_CTX)]
    if mtp:
        cmd += ["-md", paths["mtp"], "--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_n)]
    return cmd


def launch(cmd: list[str], log_path: str = "/tmp/llama-server.log") -> subprocess.Popen:
    log(f"starting llama-server, log {log_path}")
    f = open(log_path, "ab")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)


def wait_ready(port: int, timeout_s: float = 1800, proc: subprocess.Popen | None = None, on_tick=None) -> bool:
    """Poll /health: 503 while loading, 200 when ready.  False on timeout or if the process died.
    on_tick() is called after every failed poll (about every 3 s)."""
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            log(f"llama-server exited with {proc.returncode}")
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        if on_tick is not None:
            on_tick()
        time.sleep(3)
    return False


def free_port(preferred: int = 8091) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
        except OSError:
            s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def log_tail(path: str, n: int = 400) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace").strip()
    except OSError:
        return ""


def forward(port: int, path: str, raw: bytes, timeout: float = 900.0):
    """POST raw bytes to llama-server (loopback, env proxies ignored).  Returns (status, content_type, file-like
    response the caller must close); never raises: connection errors become a 502 JSON body."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=raw, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)
        return r.status, r.headers.get_content_type(), r
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get_content_type(), e
    except Exception as e:
        body = json.dumps({"error": {"message": f"Flash-Next server unreachable: {e}", "type": "server_error"}})
        return 502, "application/json", io.BytesIO(body.encode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gpu-vram", type=float, help="GiB; default from nvidia-smi")
    ap.add_argument("--ctx", type=int, default=512, help="context in K tokens (1K = 1024)")
    ap.add_argument("--quant", default="UD-IQ4_XS", choices=sorted(QUANTS))
    ap.add_argument("--kv", default="q8_0", choices=sorted(KV_BYTES))
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--mtp-variant", default="full", choices=sorted(MTP_FILES))
    ap.add_argument("--draft-n", type=int, default=2)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--llama-source", default="fork", choices=sorted(SOURCES))
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--dry-run", action="store_true", help="print plan, file list and command; no network/GPU/git")
    a = ap.parse_args()

    vram = a.gpu_vram if a.gpu_vram else detect_vram_gib()
    p = plan(vram, a.ctx, a.quant, a.kv, a.mtp, a.mtp_variant)
    if a.dry_run:
        root = os.path.join(a.cache, MODEL_DIRNAME)
        files = weight_files(a.quant, a.mtp, a.mtp_variant)
        paths = {"model": os.path.join(root, *files[0][0].split("/")),
                 "mtp": os.path.join(root, *MTP_FILES[a.mtp_variant][0].split("/")) if a.mtp else None}
        url, branch = SOURCES[a.llama_source]
        print(json.dumps(p, indent=2))
        print(json.dumps({"hf_repo": HF_REPO, "llama_cpp": {"url": url, "branch": branch},
                          "files": [{"path": r, "bytes": s} for r, s in files],
                          "total_gb": round(sum(s for _, s in files) / 1e9, 1)}, indent=2))
        print(" ".join(build_command(paths, a.ctx, a.kv, a.mtp, a.port, a.draft_n)))
        return 0
    if not p["fits"]:
        log("plan says it does not fit: " + "; ".join(p["notes"]))
        return 2
    exe = ensure_llama_server(a.cache, a.llama_source)
    paths = ensure_weights(a.cache, a.quant, a.mtp, a.mtp_variant)
    proc = launch(build_command(paths, a.ctx, a.kv, a.mtp, a.port, a.draft_n, binary=exe, threads=os.cpu_count()))
    if not wait_ready(a.port, a.timeout, proc):
        log("not ready; see /tmp/llama-server.log")
        proc.terminate()
        return 1
    log(f"READY http://127.0.0.1:{a.port}/v1 model={ALIAS}")
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
