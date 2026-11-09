import os
import time
from typing import Optional, List, Tuple, Dict, Any
from starlette.concurrency import run_in_threadpool
import shutil
import torch
from fastapi.responses import Response, StreamingResponse
from . import state
from .common import LOADED_RUNTIMES, GPU_LOCK
from .config import (
    MODEL_DOWNLOAD_STATE,
    DRIVE_MOUNTED,
    COMFYUI_STATE,
    MODEL_BASE_ROOT,
    MODEL_CATALOG,
)
from .idle import touch_activity
from .telemetry import _TELEMETRY_LOCK, _CACHED_FILE_STATUS_MAP


# ==============================================================================
# 独立 llama-server 模型 (Qwen3.8-Flash-Next、Qwen3.5-4B 链路冒烟)：MODEL_CATALOG 里 runtime == "llama" 的条目，
# runtime/flash_next.py 负责编译、下载、命令行，规格在 runtime/llama_specs.py，网关原样转发
# ==============================================================================
FLASH_NEXT_ID = "qwen38_flash_next"
FLASH_NEXT_LOG = "/tmp/llama-server-flash-next.log"


def _is_llama_model_id(model_id: Optional[str]) -> bool:
    return bool(model_id) and MODEL_CATALOG.get(model_id, {}).get("runtime") == "llama"


def _loaded_llama_ids() -> List[str]:
    return [mid for mid in LOADED_RUNTIMES if _is_llama_model_id(mid)]


def _flash_next_mod():
    try:
        import flash_next  # 与 app.py 同目录上传；笔记本模式没有这个文件
    except ImportError:
        raise RuntimeError("缺少 flash_next.py，请通过工作台部署 (runtime/ 会随 app.py 一起上传)")
    flash_next.set_log_hook(lambda m: MODEL_DOWNLOAD_STATE.update(message=f"llama.cpp · {m}"))
    return flash_next


def _flash_next_weights_dir(model_id: str = FLASH_NEXT_ID) -> str:
    """权重根目录：优先已有文件所在处，其次云盘（空间够时），否则本地 models/"""
    spec = MODEL_CATALOG[model_id]
    local = os.path.join(os.getcwd(), "models")
    for root in (MODEL_BASE_ROOT, local):
        if os.path.exists(os.path.join(root, spec["repo_id"], spec["files"][0][0])):
            return root
    if DRIVE_MOUNTED:
        try:
            if shutil.disk_usage("/content/drive/MyDrive").free / (1024 ** 3) < spec["approx_size_gb"] + 2.0:
                return local
        except Exception:
            pass
    return MODEL_BASE_ROOT


def _flash_next_progress(frac: float, text: str):
    MODEL_DOWNLOAD_STATE["progress"] = round(0.02 + 0.96 * min(1.0, frac), 3)
    MODEL_DOWNLOAD_STATE["message"] = f"正在下载 llama.cpp 模型权重 {text} · {frac:.0%}"


def flash_next_download_core(hf_token: str = "", model_id: str = FLASH_NEXT_ID) -> Tuple[bool, str]:
    touch_activity()
    if MODEL_DOWNLOAD_STATE["is_downloading"]:
        return False, "模型正在下载中，请稍候..."
    spec = MODEL_CATALOG[model_id]
    state.SELECTED_MODEL_ID = model_id
    MODEL_DOWNLOAD_STATE.update(is_downloading=True, progress=0.02, message=f"正在下载 {spec['name']} 权重...")
    try:
        if hf_token and hf_token.strip():
            os.environ["HF_TOKEN"] = hf_token.strip()
        fn = _flash_next_mod()
        fn.ensure_weights(_flash_next_weights_dir(model_id), fn.SPECS[model_id]["default_quant"], True,
                          progress=_flash_next_progress, spec=model_id)
        msg = f"{spec['name']} 权重已就绪，现在可点击「加载到显存」！"
        MODEL_DOWNLOAD_STATE.update(progress=1.0, message=msg)
        return True, msg
    except Exception as e:
        msg = f"下载 {spec['name']} 权重出错: {e}"
        MODEL_DOWNLOAD_STATE["message"] = msg
        print(msg)
        return False, msg
    finally:
        with _TELEMETRY_LOCK:
            _CACHED_FILE_STATUS_MAP.clear()
        MODEL_DOWNLOAD_STATE["is_downloading"] = False


def _cache_ram_mb(want) -> int:
    """提示词内存缓存上限：不超过本机内存的 35%（N-gram 表已常驻 26.8 GiB，G4 机器 176 GB）"""
    try:
        total_mb = int(open("/proc/meminfo").readline().split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return 0
    return max(0, min(int(want or 0), int(total_mb * 0.35)))


def flash_next_load_core(options: Optional[Dict[str, Any]] = None, context_length: Optional[int] = None,
                         model_id: str = FLASH_NEXT_ID) -> Tuple[bool, str]:
    """options: quant / kv / mtp (True) / draft_n / parallel / cache_ram_mb，默认值与上下文默认/上限见 llama_specs.SPECS；不写入 QWEN_CONTEXT_LENGTH"""
    touch_activity()
    if MODEL_DOWNLOAD_STATE["is_loading_vram"]:
        return False, "模型正在加载进显存中，请稍候..."
    if not torch.cuda.is_available():
        return False, "未检测到 CUDA GPU！请在 Colab 中切换为 G4 / H100 / A100 GPU。"
    o = options or {}
    spec = MODEL_CATALOG[model_id]
    state.SELECTED_MODEL_ID = model_id
    t_start, proc = time.time(), None
    MODEL_DOWNLOAD_STATE.update(is_loading_vram=True, progress=0.05, message=f"正在准备 {spec['name']} (llama.cpp 首次编译约 10-25 分钟)...")
    try:
        fn = _flash_next_mod()
        ls = fn.SPECS[model_id]
        quant, kv, mtp = str(o.get("quant") or ls["default_quant"]), str(o.get("kv") or ls["default_kv"]), bool(o.get("mtp", True))
        ctx = max(8192, min(ls["max_ctx_k"] * 1024, int(context_length or o.get("context_length") or ls["default_ctx_k"] * 1024)))
        total_gb = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
        plan = fn.plan(total_gb, ctx // 1024, quant, kv, mtp, spec=model_id)
        if plan["used_gib"] > total_gb:
            raise RuntimeError(f"显存不足: 预估需 {plan['used_gib']} GiB > GPU {total_gb:.0f} GiB，请降低上下文或换小量化")
        exe = fn.ensure_llama_server(MODEL_BASE_ROOT)
        MODEL_DOWNLOAD_STATE["progress"] = 0.3
        paths = fn.ensure_weights(_flash_next_weights_dir(model_id), quant, mtp, spec=model_id)
        MODEL_DOWNLOAD_STATE.update(progress=0.45, message="正在腾退显存并启动 llama-server...")
        with GPU_LOCK:
            if torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3) < plan["used_gib"] and COMFYUI_STATE.get("is_running"):
                free_comfyui_vram_core()
            evicted = _ensure_vram_for_new_model_locked(model_id, min(plan["used_gib"], total_gb - 1.2))
            free_before = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            port = fn.free_port(8091)
            cmd = fn.build_command(paths, ctx // 1024, kv, mtp, port, int(o.get("draft_n") or 0) or None, binary=exe,
                                   threads=os.cpu_count(), spec=model_id, parallel=o.get("parallel"),
                                   cache_ram_mb=_cache_ram_mb(o.get("cache_ram_mb", ls.get("cache_ram_mb", 0))))
            proc = fn.launch(cmd, FLASH_NEXT_LOG)
            MODEL_DOWNLOAD_STATE["progress"] = 0.55

            def _tick():
                MODEL_DOWNLOAD_STATE["progress"] = min(0.95, MODEL_DOWNLOAD_STATE["progress"] + 0.004)
                MODEL_DOWNLOAD_STATE["message"] = f"正在载入权重至显存 ({int(time.time() - t_start)}s)..."

            if not fn.wait_ready(port, 1800, proc, on_tick=_tick):
                raise RuntimeError("llama-server 未能就绪")
            vram_gb = round(max(0.5, free_before - torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)), 1)
            mode_label = f"{ls['label']} · {quant} · {ctx // 1024}K" + (" · MTP" if mtp else "")
            LOADED_RUNTIMES[model_id] = {
                "model_id": model_id, "name": spec["name"], "modality": spec["modality"], "mode_label": mode_label,
                "pipeline": None, "vram_gb": vram_gb, "loaded_at": time.time(), "port": port, "proc": proc,
                "match": ls["match"], "label": ls["label"],  # 网关按 match 在请求的模型名里选目标、按 label 报错
                # server_proc 让 _stop_gguf_server_bundle 在卸载/退出时终止进程
                "llm_bundle": {"backend": "flash_next", "context_length": ctx, "server_proc": proc},
            }
            _sync_legacy_globals()
        msg = f"{spec['name']} 已装载 ({mode_label}，约 {vram_gb}GB，耗时 {round(time.time() - t_start)}s)" + (f"，已腾退 {', '.join(evicted)}" if evicted else "")
        MODEL_DOWNLOAD_STATE.update(progress=1.0, message=msg)
        return True, msg
    except Exception as e:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        tail = _flash_next_mod().log_tail(FLASH_NEXT_LOG) if proc is not None else ""
        err = f"加载 {spec['name']} 失败: {e}" + (f" | {tail}" if tail else "")
        MODEL_DOWNLOAD_STATE["message"] = err
        print(err)
        return False, err
    finally:
        MODEL_DOWNLOAD_STATE["is_loading_vram"] = False
        touch_activity()


def _flash_next_port(model_name: str) -> Optional[int]:
    """请求应转给 llama 运行时模型时返回其端口：模型名含该模型的 match，或有 llama 模型已加载且没有 27B 在线"""
    alive = []
    for mid in _loaded_llama_ids():
        rt = LOADED_RUNTIMES[mid]
        if rt["proc"].poll() is not None:
            LOADED_RUNTIMES.pop(mid, None)
        else:
            alive.append(rt)
    if not alive:
        return None
    low = model_name.lower().replace("_", "-")
    for rt in alive:
        if rt["match"] in low:
            return rt["port"]
    if not ("qwen38_27b" in LOADED_RUNTIMES or "qwen38_27b_gguf" in LOADED_RUNTIMES):
        return alive[0]["port"]
    return None


async def _flash_next_proxy(port: int, path: str, raw: bytes):
    """原样转发请求体到本机 llama-server；event-stream 响应逐块透传"""
    label = next((rt["label"] for rt in LOADED_RUNTIMES.values() if rt.get("port") == port and rt.get("label")), "llama-server")
    status, ctype, resp = await run_in_threadpool(_flash_next_mod().forward, port, path, raw, 900.0, label)
    if ctype != "text/event-stream":
        data = await run_in_threadpool(resp.read)
        resp.close()
        return Response(content=data, status_code=status, media_type=ctype)

    def _chunks():
        try:
            while True:
                chunk = resp.read1(8192)
                if not chunk:
                    break
                yield chunk
        finally:
            resp.close()

    return StreamingResponse(_chunks(), media_type=ctype, headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _link_forward():
    # 前向引用：定义在后面的模块，函数体运行时才用；入口在全部模块加载完后统一回填，保证严格按切片顺序加载
    global _sync_legacy_globals, _ensure_vram_for_new_model_locked, free_comfyui_vram_core
    from .vram import _sync_legacy_globals, _ensure_vram_for_new_model_locked
    from .comfy_runtime import free_comfyui_vram_core
