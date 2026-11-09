import os
import time
import threading
from typing import Optional, List, Tuple
import shutil
from . import state
from .bootstrap import CONSOLE_LOCK, ACTIVE_TQDM_BARS
from .config import MODEL_DOWNLOAD_STATE, DRIVE_MOUNTED, MODEL_CATALOG
from .hf_download import fast_parallel_hf_download
from .idle import touch_activity
from .model_files import _get_model_dirs, _find_existing_model_file, check_model_files_status
from .telemetry import _TELEMETRY_LOCK, _CACHED_FILE_STATUS_MAP


def build_structured_prompt(
    visual_prompt: str,
    soundscape_prompt: str,
    music_prompt: str,
) -> str:
    visual_prompt = (visual_prompt or "").strip()
    soundscape_prompt = (soundscape_prompt or "").strip()
    music_prompt = (music_prompt or "").strip()

    if not soundscape_prompt and not music_prompt:
        return visual_prompt

    parts = [visual_prompt]
    if soundscape_prompt:
        parts.append(f"Soundscape & Dialogue: {soundscape_prompt}")
    if music_prompt:
        parts.append(f"Background Music: {music_prompt}")
    return "\n".join(parts)


# ==============================================================================
# 独立功能 1：多模型通用极速下载器 (8路并发 HTTP Range + 后台 Google Drive 备份)
# ==============================================================================
def _async_backup_to_drive(src_file: str, dst_dir: str, fname: str):
    """在后台线程中将内存盘/NVMe下载好的权重同步到 Google Drive，不阻塞用户加载模型"""
    if not DRIVE_MOUNTED:
        return
    try:
        dst_file = os.path.join(dst_dir, fname)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        if os.path.exists(dst_file) and os.path.getsize(dst_file) == os.path.getsize(src_file):
            return
        tmp_dst = dst_file + ".syncing"
        with open(src_file, "rb") as rf, open(tmp_dst, "wb") as wf:
            shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
        os.replace(tmp_dst, dst_file)
        print(f"[后台云盘同步完成] {fname} 已持久化保存至 Google Drive")
    except Exception as e:
        print(f"[后台云盘同步跳过] {fname}: {e}")


def download_model_weights_core(model_id: Optional[str] = None, hf_token: str = "") -> Tuple[bool, str]:
    if _is_llama_model_id(model_id):
        return flash_next_download_core(hf_token, model_id)
    touch_activity()
    if MODEL_DOWNLOAD_STATE["is_downloading"]:
        return False, "模型正在下载中，请稍候..."

    target_id = model_id if (model_id and model_id in MODEL_CATALOG) else state.SELECTED_MODEL_ID
    state.SELECTED_MODEL_ID = target_id
    spec = MODEL_CATALOG[target_id]
    repo_id = spec["repo_id"]
    req_files = spec["files"]

    MODEL_DOWNLOAD_STATE["is_downloading"] = True
    MODEL_DOWNLOAD_STATE["progress"] = 0.02
    MODEL_DOWNLOAD_STATE["message"] = f"正在为 {spec['name']} 启动 8 路并发 HTTP Range 极速下载通道..."
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()

    try:
        if hf_token and hf_token.strip():
            os.environ["HF_TOKEN"] = hf_token.strip()
            from huggingface_hub import login
            login(token=hf_token.strip(), add_to_git_credential=False)

        ram_dir, nvme_dir, drive_dir = _get_model_dirs(target_id)
        drive_has_space = False
        if DRIVE_MOUNTED and os.path.exists("/content/drive/MyDrive"):
            try:
                drive_free_gb = shutil.disk_usage("/content/drive/MyDrive").free / (1024 ** 3)
                drive_has_space = drive_free_gb >= spec["approx_size_gb"] + 2.0
                if not drive_has_space:
                    print(
                        f"ℹ️ Google Drive 剩余空间 ({drive_free_gb:.1f} GB) 小于 {spec['name']} 所需容量 "
                        f"({spec['approx_size_gb']:.1f} GB)，自动改用实例本地高速 NVMe 磁盘 ({nvme_dir}) 存储！"
                    )
            except Exception:
                drive_has_space = False

        primary_dl_dir = ram_dir if ram_dir else (drive_dir if (DRIVE_MOUNTED and drive_has_space) else nvme_dir)
        os.makedirs(primary_dl_dir, exist_ok=True)
        total_files = len(req_files)

        for idx, (fname, desc) in enumerate(req_files):
            touch_activity()
            existing = _find_existing_model_file(target_id, fname)
            base_prog = idx / float(total_files + 1)
            MODEL_DOWNLOAD_STATE["progress"] = round(base_prog + 0.04, 2)

            if existing:
                # 若使用 Qwen3.8-27B，确保同一目录内有软链接以便 AutoModel 直接按目录加载
                if target_id == "qwen38_27b":
                    target_link = os.path.join(primary_dl_dir, fname)
                    os.makedirs(os.path.dirname(target_link), exist_ok=True)
                    if os.path.abspath(existing) != os.path.abspath(target_link) and not os.path.exists(target_link):
                        try:
                            os.symlink(existing, target_link)
                        except Exception:
                            pass
                size_gb = os.path.getsize(existing) / (1024 ** 3)
                MODEL_DOWNLOAD_STATE["message"] = f"[{idx+1}/{total_files}] {fname} 已就绪 ({size_gb:.2f} GB)，跳过下载"
                print(MODEL_DOWNLOAD_STATE["message"])
                continue

            MODEL_DOWNLOAD_STATE["message"] = f"[{idx+1}/{total_files}] 正在 8 线程并发下载 {fname} ({desc})..."
            print(MODEL_DOWNLOAD_STATE["message"])
            dl_path = fast_parallel_hf_download(
                repo_id=repo_id,
                filename=fname,
                dest_dir=primary_dl_dir,
                hf_token=hf_token,
                num_connections=8,
            )
            if DRIVE_MOUNTED and drive_has_space and primary_dl_dir != drive_dir:
                threading.Thread(
                    target=_async_backup_to_drive,
                    args=(dl_path, drive_dir, fname),
                    daemon=True,
                ).start()

        MODEL_DOWNLOAD_STATE["progress"] = 0.92
        if target_id == "qwen38_27b_gguf":
            MODEL_DOWNLOAD_STATE["message"] = "正在预装并校验 llama.cpp CUDA 高速推理引擎..."
            _ensure_llama_server_binary()
        else:
            MODEL_DOWNLOAD_STATE["message"] = f"正在获取 {spec['name']} 的分词器与 Processor 配置文件..."
            from huggingface_hub import snapshot_download

            if target_id == "minimax_h3":
                from diffsynth.pipelines.minimax_h3_audio_video import ModelConfig
                proc_cfg = ModelConfig(
                    model_id="MiniMaxAI/MiniMax-H3",
                    origin_file_pattern="FL2VA/processor/",
                    download_source="huggingface",
                )
                proc_cfg.download_if_necessary()
            elif target_id == "qwen38_27b":
                snapshot_download(
                    repo_id="Qwen/Qwen3.8-27B",
                    local_dir=primary_dl_dir,
                    ignore_patterns=["*.safetensors", "*.bin", "*.pth"],
                )
            elif target_id == "qwen_image_21":
                from diffsynth.pipelines.qwen_image_21 import ModelConfig
                proc_cfg = ModelConfig(
                    model_id="Qwen/Qwen-Image-2.1",
                    origin_file_pattern="processor/",
                    download_source="huggingface",
                )
                proc_cfg.download_if_necessary()
            elif target_id == "z_image_turbo":
                from diffsynth.pipelines.z_image import ModelConfig
                tok_cfg = ModelConfig(
                    model_id="Tongyi-MAI/Z-Image-Turbo",
                    origin_file_pattern="tokenizer/",
                    download_source="huggingface",
                )
                tok_cfg.download_if_necessary()

        MODEL_DOWNLOAD_STATE["progress"] = 1.0
        status = check_model_files_status(target_id)
        msg = f"{spec['name']} 全部 {status['downloaded_count']}/{status['total_count']} 个权重文件已就绪（共 {status['total_size_gb']:.2f} GB），现在可点击「加载到显存」！"
        MODEL_DOWNLOAD_STATE["message"] = msg
        print(msg)
        return True, msg

    except Exception as e:
        err_msg = f"下载 {spec['name']} 权重出错: {str(e)}"
        MODEL_DOWNLOAD_STATE["message"] = err_msg
        print(err_msg)
        return False, err_msg
    finally:
        with _TELEMETRY_LOCK:
            _CACHED_FILE_STATUS_MAP.clear()
        MODEL_DOWNLOAD_STATE["is_downloading"] = False


def download_minimax_h3_weights_core(hf_token: str = "") -> Tuple[bool, str]:
    return download_model_weights_core("minimax_h3", hf_token)


# ==============================================================================
# 独立功能 2：多模型通用显存加载引擎 (自动卸载旧模型释放显存 + 零拷贝极速加载)
# ==============================================================================
def _resolve_and_stage_model_file(model_id: str, fname: str) -> str:
    """解析模型权重路径；若位于较慢的云盘且 /dev/shm 内存盘充裕，自动预热至内存盘"""
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(model_id)
    existing = _find_existing_model_file(model_id, fname)
    if not existing:
        return os.path.join(drive_dir, fname)

    if ram_dir and not existing.startswith("/dev/shm/"):
        ram_target = os.path.join(ram_dir, fname)
        if os.path.exists(ram_target) and os.path.getsize(ram_target) == os.path.getsize(existing):
            return ram_target
        try:
            os.makedirs(os.path.dirname(ram_target), exist_ok=True)
            free_shm = shutil.disk_usage("/dev/shm").free
            file_sz = os.path.getsize(existing)
            if free_shm > file_sz + (2 * 1024 ** 3):
                t0 = time.time()
                print(f"正在将 {fname} ({file_sz/(1024**3):.2f} GB) 高速预热至 /dev/shm 内存盘...")
                tmp_ram = ram_target + ".staging"
                with open(existing, "rb") as rf, open(tmp_ram, "wb") as wf:
                    shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
                os.replace(tmp_ram, ram_target)
                dt = max(0.01, time.time() - t0)
                print(f"{fname} 内存盘预热完成 ({file_sz/(1024**2)/dt:.0f} MB/s)")
                return ram_target
        except Exception as e:
            print(f"内存盘预热跳过 ({e})，直接读取原路径")
    return existing


def _stage_model_files_to_local_nvme(
    model_id: str,
    paths: List[str],
    rel_names: List[str],
    group_label: str,
) -> List[str]:
    """当分片仍需走磁盘映射且原文件位于慢速云盘 FUSE (/content/drive) 时，若本地 NVMe 空间充裕则整组顺序缓存至本地高速 SSD"""
    if not paths or len(paths) != len(rel_names):
        return paths
    _, nvme_dir, _ = _get_model_dirs(model_id)
    if not any(p.startswith("/content/drive/") for p in paths):
        return paths

    try:
        os.makedirs(nvme_dir, exist_ok=True)
        free_nvme_bytes = shutil.disk_usage(nvme_dir).free
        needed_bytes = 0
        stage_plan: List[Tuple[int, str, str, int]] = []
        result_paths = list(paths)

        for idx, (src_p, rel_name) in enumerate(zip(paths, rel_names)):
            if not src_p.startswith("/content/drive/") or not os.path.exists(src_p):
                continue
            src_sz = os.path.getsize(src_p)
            dst_p = os.path.join(nvme_dir, rel_name)
            if os.path.exists(dst_p) and os.path.getsize(dst_p) == src_sz:
                result_paths[idx] = dst_p
                continue
            needed_bytes += src_sz
            stage_plan.append((idx, src_p, dst_p, src_sz))

        if not stage_plan:
            return result_paths

        # 保留至少 22GB 本地磁盘安全余量，避免挤占视频导出与系统空间
        if free_nvme_bytes <= needed_bytes + (22 * 1024 ** 3):
            return result_paths

        total_gb = needed_bytes / (1024 ** 3)
        print(f"⚡ [NVMe 高速缓存] 正在将 {group_label} ({len(stage_plan)} 个分片 / {total_gb:.1f} GB) 从云盘顺序预热至本地 NVMe SSD...")
        t_all = time.time()
        for step_i, (idx, src_p, dst_p, src_sz) in enumerate(stage_plan, start=1):
            os.makedirs(os.path.dirname(dst_p), exist_ok=True)
            MODEL_DOWNLOAD_STATE["message"] = (
                f"正在将 {group_label} 高速缓存至本地 SSD ({step_i}/{len(stage_plan)})，消除后续换入延迟..."
            )
            tmp_dst = dst_p + ".staging"
            with open(src_p, "rb") as rf, open(tmp_dst, "wb") as wf:
                shutil.copyfileobj(rf, wf, length=32 * 1024 * 1024)
            os.replace(tmp_dst, dst_p)
            result_paths[idx] = dst_p
        dt_all = max(0.05, time.time() - t_all)
        print(f"⚡ [NVMe 高速缓存] {group_label} 本地 SSD 预热完成 ({total_gb / dt_all * 1024:.0f} MB/s)")
        return result_paths
    except Exception as e:
        print(f"ℹ️ 本地 NVMe 预热跳过 ({e})，继续读取原路径")
        return paths


def _link_forward():
    # 前向引用：定义在后面的模块，函数体运行时才用；入口在全部模块加载完后统一回填，保证严格按切片顺序加载
    global _ensure_llama_server_binary, _is_llama_model_id, flash_next_download_core
    from .gguf_server import _ensure_llama_server_binary
    from .flash_next_glue import _is_llama_model_id, flash_next_download_core
