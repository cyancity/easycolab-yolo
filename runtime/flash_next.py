#!/usr/bin/env python3
"""Serve Qwen GGUF models on one Colab GPU through llama.cpp `llama-server` (Flash-Next + a small smoke model).

Model data (repo, files, MTP, ctx, extra args, architecture) is in llama_specs.SPECS, keyed by app model id; every
function takes `spec=<id>`, default Flash-Next.  The small model (qwen35_4b_smoke, fits a T4) uses the same
build / download / launch / ready / forward path, so a T4 run validates what Flash-Next needs on G4 / H100 / A100.

Flash-Next goal: ctx >= 300K (target 512K), single-stream decode >= 100 tok/s (target 150),
VRAM use >= 75%.  Targets: Colab G4 (RTX PRO 6000 Blackwell 96GB), H100 80GB, A100 80GB.
Only the standard library and huggingface_hub are used.  Units: GiB = 2**30 bytes;
pass nvidia-smi memory.total / 1024 as the GPU size.  "K" in ctx_k = 1024 tokens.

VERIFIED (read on 2026-11-10)
  * HF repo `unsloth/Qwen3.8-Flash-Next-GGUF`, file tree via
    https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main?recursive=true
    - shard layout, sizes, engram shard and MTP heads: see llama_specs.py (other engram tables in `engram/`
      plus `engram/set_split.py` are not used here).
  * llama.cpp: https://github.com/danielhanchen/llama.cpp branch `qwen4exp/mtp` exists
    (head 6fcaa16f4b36 on 2026-09-21, api.github.com/repos/danielhanchen/llama.cpp/branches/qwen4exp/mtp).
    Upstream https://github.com/ggml-org/llama.cpp merged "Qwen4Exp: add MTP" (PR #29761, 2026-10-01).
    Upstream does NOT load the `shared` MTP head (missing token_embd.weight), so the full head is the default.
  * Flags: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md (-fa, -ctk/-ctv, --rope-scaling
    yarn, -ot, --jinja, --metrics, -a, -np, -md, --spec-type draft-mtp, --spec-draft-n-max).
  * Architecture and YaRN factor (2.0 for 524,288 tokens): see llama_specs.py.
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
  4. KV bytes per token, MTP draft KV and compute buffers are estimates.  Read real VRAM from nvidia-smi.
  5. nvcc / cmake being present on the Colab G4 image; CMAKE_CUDA_ARCHITECTURES=120 for Blackwell
     (CUDA >= 12.8 needed).  `-DLLAMA_CURL=OFF` is only a hint to older trees.
  6. Decode speed (>=100 / 150 tok/s) is not modelled; only measured numbers count.
  7. The fork supports Qwen3.5 MTP (smoke model, see llama_specs.py).
  8. Build recipe (-DGGML_NATIVE=OFF + AVX2, cuda libs linked dynamically) has not been built on any GPU yet.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import io
import json
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

import llama_specs
from llama_specs import SPECS

LOG = "FN_LOG:"
GIB = 2 ** 30
SOURCES = {
    "fork": ("https://github.com/danielhanchen/llama.cpp", "qwen4exp/mtp"),
    "upstream": ("https://github.com/ggml-org/llama.cpp", "master"),
}
DEFAULT_SPEC = llama_specs.FLASH_NEXT_ID


def _spec(spec: str) -> dict:
    if spec not in SPECS:
        raise ValueError(f"unknown llama model {spec!r}; choose from {sorted(SPECS)}")
    return SPECS[spec]


# --- KV / runtime model (all ESTIMATES except the architecture numbers; per-model values in llama_specs.arch) ---
# Flash-Next source: Qwen model card https://huggingface.co/Qwen/Qwen3.8-Flash-Next (48 layers = 12 x (3 DeltaNet + 1 QSA)).
# indexer_dim: QSA indexer key cache, ASSUMED to be cached once per token per full-attention layer.
# MTP draft block: 1 dense attention layer (PR #27836); ASSUMED.  DeltaNet state: fp32 v_heads x dim x dim per layer; ASSUMED.
# runtime/mtp buffers: CUDA context + compute buffers at -ub 1024 / draft compute buffers; ASSUMED.
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q4_0": 18 / 32}  # ggml block sizes
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


def _shards(quant: str, spec: str = DEFAULT_SPEC) -> list[tuple[str, int]]:
    sp = _spec(spec)
    if quant not in sp["quants"]:
        raise ValueError(f"unknown quant {quant!r} for {spec}; choose from {sorted(sp['quants'])}")
    sizes, _ = sp["quants"][quant]
    n = len(sizes)
    return [(sp["file_fmt"].format(q=quant, i=i + 1, n=n), s) for i, s in enumerate(sizes)]


def _mtp_file(spec: str, mtp_variant: str) -> tuple[str, int] | None:
    """Separate MTP head file, or None when the model has no such file (head embedded in the main GGUF)."""
    files = _spec(spec)["mtp_files"]
    if not files:
        return None
    if mtp_variant not in files:
        raise ValueError(f"unknown mtp variant {mtp_variant!r} for {spec}; choose from {sorted(files)}")
    return files[mtp_variant]


def weight_files(quant: str, mtp: bool, mtp_variant: str = "full", spec: str = DEFAULT_SPEC) -> list[tuple[str, int]]:
    """Repo path and byte size of every file ensure_weights needs."""
    files = _shards(quant, spec)
    head = _mtp_file(spec, mtp_variant) if mtp else None
    if head:
        files.append(head)
    return files


def plan(gpu_vram_gb: float, ctx_k: int, quant: str, kv: str, mtp: bool, mtp_variant: str = "full",
         spec: str = DEFAULT_SPEC) -> dict:
    """VRAM budget in GiB.  Weights are file sizes (exact); everything else is an estimate."""
    if kv not in KV_BYTES:
        raise ValueError(f"unknown kv type {kv!r}; choose from {sorted(KV_BYTES)}")
    sp = _spec(spec)
    a = sp["arch"]
    shards = _shards(quant, spec)
    engram_idx = sp["quants"][quant][1]
    engram = shards[engram_idx - 1][1] if engram_idx else 0
    weights_gpu = (sum(s for _, s in shards) - engram) / GIB
    tokens = ctx_k * 1024
    # K + V elements per token per full-attention layer, plus the assumed indexer key cache
    kv_elems = a["full_attn_layers"] * (2 * a["kv_heads"] * a["head_dim"] + a["indexer_dim"])
    kv_gib = tokens * kv_elems * KV_BYTES[kv] / GIB
    head = _mtp_file(spec, mtp_variant) if mtp else None
    mtp_w = head[1] / GIB if head else 0.0
    mtp_kv = tokens * (2 * a["kv_heads"] * a["head_dim"]) * KV_BYTES[kv] / GIB if mtp else 0.0
    mtp_buf = a["mtp_buffers_gib"] if mtp else 0.0
    deltanet_state = a["deltanet_layers"] * a["deltanet_v_heads"] * a["deltanet_dim"] * a["deltanet_dim"] * 4 / GIB
    parts = {
        "weights_gpu_exact": weights_gpu,
        f"kv_cache_{EST}": kv_gib,
        "mtp_weights_exact": mtp_w,
        f"mtp_kv_{EST}": mtp_kv,
        f"mtp_buffers_{EST}": mtp_buf,
        f"deltanet_state_{EST}": deltanet_state,
        f"runtime_buffers_{EST}": a["runtime_buffers_gib"],
    }
    used = sum(parts.values())
    fill = used / gpu_vram_gb
    notes = []
    if engram:
        notes.append(f"engram shard {engram / GIB:.1f} GiB is kept in CPU RAM (-ot), needs that much host RAM/page cache")
    notes += [
        f"KV counts the {a['full_attn_layers']} full-attention layers only; DeltaNet state is constant per slot",
        "numbers marked ESTIMATE are not measured; check nvidia-smi after start",
    ]
    native = sp["native_ctx"]
    if tokens > native:
        notes.append(f"ctx {tokens} > native {native}: static YaRN factor {tokens / native:.2f} "
                     "(may slightly hurt short-text quality)")
    if tokens < 300 * 1024:
        notes.append("ctx below the 300K requirement")
    if fill < VRAM_TARGET:
        notes.append(f"fill {fill:.0%} < {VRAM_TARGET:.0%} target; a larger quant or KV type would use more VRAM")
    if used > gpu_vram_gb * (1 - HEADROOM_FRAC):
        notes.append("does not fit with 5% headroom: use a smaller quant, kv q8_0/q4_0 or lower ctx")
    free = gpu_vram_gb * (1 - HEADROOM_FRAC) - used
    if engram and free > engram / GIB:
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
    """Compute capability of GPU 0 from nvidia-smi, e.g. '7.5' -> '75' (T4), '12.0' -> '120' (G4)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.split()[0]
    return out.replace(".", "")


def cmake_args(src: str, sm: str) -> list[str]:
    """cmake configure args.  The binary is built for exactly one GPU arch (sm) and, because the tarball is cached
    on a shared drive, for a baseline x86 CPU (AVX2) instead of -march=native: the next session may land on a
    different host CPU than the one that built it."""
    if not re.fullmatch(r"\d{2,3}[a-z]?", sm):
        raise ValueError(f"bad compute capability {sm!r}")
    return ["cmake", "-S", src, "-B", f"{src}/build", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={sm}",
            "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_BMI2=ON",
            "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"]


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
    """Return a path to llama-server; build once, then reuse the tarball under cache_dir/llama.cpp/.

    Tarball, extraction dir and build dir are all named `<commit12>-sm<cc>` (cc = compute capability of this GPU),
    so a T4 (sm75) build is never picked up on an A100 (sm80) / L4 (sm89) / H100 (sm90) / G4 (sm120) or vice versa."""
    url, branch = SOURCES[source]
    sm = sm or detect_sm()
    cmake_args("", sm)  # validate sm before it goes into a file name
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
    log(f"cmake configure sm{sm} ({llama_specs.GPU_ARCHS.get(sm, 'unlisted GPU')})")
    _run(cmake_args(src, sm))
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


def ensure_weights(cache_dir: str, quant: str, mtp: bool, mtp_variant: str = "full", progress=None,
                   spec: str = DEFAULT_SPEC) -> dict:
    """Download missing files into cache_dir/<repo id>/ and return their paths (layout app.py uses for MODEL_CATALOG).

    progress(fraction, text) is called every ~2 s from a helper thread while files download."""
    root = os.path.join(cache_dir, _spec(spec)["repo"])
    os.makedirs(root, exist_ok=True)
    stop = threading.Event()
    if progress is not None:
        files = weight_files(quant, mtp, mtp_variant, spec)
        total = sum(s for _, s in files)

        def _poll():
            while not stop.wait(2.0):
                done = _downloaded_bytes(root, files)
                progress(done / total, f"{done / 1e9:.1f} / {total / 1e9:.1f} GB")
        threading.Thread(target=_poll, daemon=True).start()
    try:
        return _ensure_weights(root, quant, mtp, mtp_variant, spec)
    finally:
        stop.set()


def _ensure_weights(root: str, quant: str, mtp: bool, mtp_variant: str, spec: str) -> dict:
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # set before huggingface_hub is imported
    from huggingface_hub import hf_hub_download
    paths = {}
    for rel, size in weight_files(quant, mtp, mtp_variant, spec):
        dest = os.path.join(root, *rel.split("/"))
        if os.path.isfile(dest) and os.path.getsize(dest) == size:
            log(f"have {rel}")
        else:
            log(f"downloading {rel} ({size / 1e9:.1f} GB)")
            hf_hub_download(_spec(spec)["repo"], rel, local_dir=root)
            if os.path.getsize(dest) != size:
                raise RuntimeError(f"{rel}: size {os.path.getsize(dest)} != expected {size}")
        paths[rel] = dest
    shard_paths = [paths[r] for r, _ in _shards(quant, spec)]
    head = _mtp_file(spec, mtp_variant) if mtp else None
    return {"root": root, "model": shard_paths[0], "shards": shard_paths, "mtp": paths[head[0]] if head else None}


# --- command / process ------------------------------------------------------------------------------------
def build_command(paths: dict, ctx_k: int, kv: str, mtp: bool, port: int, draft_n: int | None = None,
                  host: str = "127.0.0.1", binary: str = "llama-server", threads: int | None = None,
                  spec: str = DEFAULT_SPEC, parallel: int | None = None, cache_ram_mb: int | None = None) -> list[str]:
    """parallel: request slots; >1 adds --kv-unified so the slots share one KV pool of `ctx_k` and a single request
    can still use all of it.  cache_ram_mb: host-RAM prompt cache, so a long prefix (an agent's system prompt and
    history) evicted from the slot is restored instead of prefilled again; 0 disables."""
    sp = _spec(spec)
    tokens = ctx_k * 1024
    np_ = max(1, int(parallel or sp.get("parallel", 1)))
    cmd = [binary, "-m", paths["model"], "--alias", sp["alias"], "--host", host, "--port", str(port),
           "-c", str(tokens), "-np", str(np_), "-ngl", "999", *sp["extra_args"],
           "-fa", "on", "-ctk", kv, "-ctv", kv, "-b", "2048", "-ub", "1024",
           "--jinja", "--metrics", *sp["sampling"]]
    if np_ > 1:
        cmd += ["--kv-unified"]
    if cache_ram_mb:
        cmd += ["--cache-ram", str(int(cache_ram_mb))]
    if threads:
        cmd += ["-t", str(threads)]
    if tokens > sp["native_ctx"]:
        cmd += ["--rope-scaling", "yarn", "--rope-scale", f"{tokens / sp['native_ctx']:g}",
                "--yarn-orig-ctx", str(sp["native_ctx"])]
    if mtp:
        if paths.get("mtp"):  # models whose MTP head is inside the main GGUF have no -md file
            cmd += ["-md", paths["mtp"]]
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_n or sp["draft_n"])]
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


def forward(port: int, path: str, raw: bytes, timeout: float = 900.0, label: str = "Flash-Next"):
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
        body = json.dumps({"error": {"message": f"{label} server unreachable: {e}", "type": "server_error"}})
        return 502, "application/json", io.BytesIO(body.encode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_SPEC, choices=sorted(SPECS), help="app model id (llama_specs.SPECS)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gpu-vram", type=float, help="GiB; default from nvidia-smi")
    ap.add_argument("--sm", help="compute capability without the dot (T4 75, A100 80, L4 89, H100 90, G4 120); "
                                 "default from nvidia-smi; with --dry-run it adds the build recipe to the output")
    ap.add_argument("--ctx", type=int, help="context in K tokens (1K = 1024); default per model")
    ap.add_argument("--quant", help="default per model")
    ap.add_argument("--kv", default=None, choices=sorted(KV_BYTES), help="default per model")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--mtp-variant", default="full")
    ap.add_argument("--draft-n", type=int, help="default per model")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--llama-source", default="fork", choices=sorted(SOURCES))
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--dry-run", action="store_true", help="print plan, file list and command; no network/GPU/git")
    a = ap.parse_args()
    sp = _spec(a.model)
    quant, kv, ctx = a.quant or sp["default_quant"], a.kv or sp["default_kv"], a.ctx or sp["default_ctx_k"]
    if quant not in sp["quants"]:
        ap.error(f"--quant must be one of {sorted(sp['quants'])} for {a.model}")

    vram = a.gpu_vram if a.gpu_vram else detect_vram_gib()
    p = plan(vram, ctx, quant, kv, a.mtp, a.mtp_variant, a.model)
    if a.dry_run:
        root = os.path.join(a.cache, sp["repo"])
        files = weight_files(quant, a.mtp, a.mtp_variant, a.model)
        head = _mtp_file(a.model, a.mtp_variant) if a.mtp else None
        paths = {"model": os.path.join(root, *files[0][0].split("/")),
                 "mtp": os.path.join(root, *head[0].split("/")) if head else None}
        url, branch = SOURCES[a.llama_source]
        build = {"url": url, "branch": branch}
        if a.sm:
            build.update(sm=a.sm, gpu=llama_specs.GPU_ARCHS.get(a.sm), cache_file=f"llama.cpp/<commit12>-sm{a.sm}.tar.gz",
                         cmake=" ".join(cmake_args("<src>", a.sm)))
        print(json.dumps(p, indent=2))
        print(json.dumps({"hf_repo": sp["repo"], "llama_cpp": build,
                          "files": [{"path": r, "bytes": s} for r, s in files],
                          "total_gb": round(sum(s for _, s in files) / 1e9, 1)}, indent=2))
        print(" ".join(build_command(paths, ctx, kv, a.mtp, a.port, a.draft_n, spec=a.model)))
        return 0
    if not p["fits"]:
        log("plan says it does not fit: " + "; ".join(p["notes"]))
        return 2
    exe = ensure_llama_server(a.cache, a.llama_source, a.sm)
    paths = ensure_weights(a.cache, quant, a.mtp, a.mtp_variant, spec=a.model)
    proc = launch(build_command(paths, ctx, kv, a.mtp, a.port, a.draft_n, binary=exe, threads=os.cpu_count(), spec=a.model))
    if not wait_ready(a.port, a.timeout, proc):
        log("not ready; see /tmp/llama-server.log")
        proc.terminate()
        return 1
    log(f"READY http://127.0.0.1:{a.port}/v1 model={sp['alias']}")
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
