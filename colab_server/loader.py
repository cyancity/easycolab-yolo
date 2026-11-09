import os
import sys
import time
import subprocess
import threading
from typing import Optional, Tuple
import torch
from . import state
from .bootstrap import CONSOLE_LOCK, ACTIVE_TQDM_BARS
from .kernels import _ensure_torchaudio_compatible, _ensure_qwen_fast_kernels
from .qwen_optimize import _optimize_qwen35_for_fast_inference
from .common import LOADED_RUNTIMES, GPU_LOCK, _clamp_context_length
from .config import (
    MODEL_DOWNLOAD_STATE,
    _get_system_ram_gb,
    DRIVE_MOUNTED,
    COMFYUI_STATE,
    MODEL_CATALOG,
)
from .hf_download import fast_parallel_hf_download
from .idle import touch_activity
from .model_files import _get_model_dirs, _find_existing_model_file, check_model_files_status
from .weights import (
    _async_backup_to_drive,
    _resolve_and_stage_model_file,
    _stage_model_files_to_local_nvme,
)
from .gguf_server import _start_qwen_gguf_server
from .flash_next_glue import _is_llama_model_id, flash_next_load_core
from .vram import _sync_legacy_globals, _trim_linux_cpu_heap, _ensure_vram_for_new_model_locked


def load_model_to_vram_core(
    model_id: Optional[str] = None,
    precision_mode: str = "bf16",
    offload_mode: str = "disk",
    hf_token: str = "",
    context_length: Optional[int] = None,
) -> Tuple[bool, str]:
    if _is_llama_model_id(model_id):
        return flash_next_load_core(None, context_length, model_id)
    touch_activity()

    if context_length is not None:
        state.QWEN_CONTEXT_LENGTH = _clamp_context_length(context_length)
    ctx_len = state.QWEN_CONTEXT_LENGTH

    if MODEL_DOWNLOAD_STATE["is_loading_vram"]:
        return False, "模型正在加载进显存中，请稍候..."

    if not torch.cuda.is_available():
        return False, "未检测到 CUDA GPU！请在 Colab 中切换为 G4 / A100 / L4 GPU。"

    target_id = model_id if (model_id and model_id in MODEL_CATALOG) else state.SELECTED_MODEL_ID
    state.SELECTED_MODEL_ID = target_id
    spec = MODEL_CATALOG[target_id]
    file_status = check_model_files_status(target_id)
    auto_switched_notice = ""

    if target_id == "qwen38_27b_gguf":
        if not (
            _find_existing_model_file("qwen38_27b_gguf", "Qwen3.8-27B-UD-Q4_K_M.gguf")
            and _find_existing_model_file("qwen38_27b_gguf", "mmproj-F16.gguf")
        ):
            return False, f"{spec['name']} 权重尚未下载完整 ({file_status['downloaded_count']}/{file_status['total_count']})！请先点击「下载权重」。"
    else:
        if not file_status["all_downloaded"]:
            return False, f"{spec['name']} 权重尚未下载完整 ({file_status['downloaded_count']}/{file_status['total_count']})！请先点击「下载权重」。"

    t_start = time.time()
    MODEL_DOWNLOAD_STATE["is_loading_vram"] = True
    MODEL_DOWNLOAD_STATE["progress"] = 0.12
    MODEL_DOWNLOAD_STATE["message"] = f"正在评估显存余量并装载 {spec['name']} 至 GPU..."
    with CONSOLE_LOCK:
        ACTIVE_TQDM_BARS.clear()

    with GPU_LOCK:
        try:
            total_vram_gb = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
            vram_limit = max(8.0, total_vram_gb - 2.5)

            # 计算新模型所需最小空闲显存阈值：余量充足时自动与已加载模型共驻显存！
            if target_id == "qwen38_27b":
                required_free_gb = 53.0
            elif target_id == "qwen38_27b_gguf":
                q4_kv_gb = (ctx_len / 262144.0) * 4.6
                required_free_gb = min(max(20.8, 20.6 + q4_kv_gb), max(12.0, total_vram_gb - 1.2))
            elif target_id == "minimax_h3":
                # BF16 单阶段峰值约 62.1GB；在 G4 (96GB) 上可与 Qwen3.8-GGUF (~21GB) 同时共驻显存
                required_free_gb = min(64.5, max(24.0, total_vram_gb - 5.0))
            else:
                required_free_gb = 24.0

            if torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3) < required_free_gb and COMFYUI_STATE.get("is_running"):
                free_comfyui_vram_core()
            evicted_names = _ensure_vram_for_new_model_locked(target_id, required_free_gb)
            free_before_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)

            vram_config = {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": "disk",
                "onload_device": "disk",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }

            new_pipe = None
            new_llm = None
            mode_label = spec["name"]

            if target_id == "minimax_h3":
                from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

                _trim_linux_cpu_heap()
                total_ram_gb, avail_ram_gb = _get_system_ram_gb()
                cpu_fast_vram_config = {
                    "offload_dtype": torch.bfloat16,
                    "offload_device": "cpu",
                    "onload_dtype": torch.bfloat16,
                    "onload_device": "cpu",
                    "preparing_dtype": torch.bfloat16,
                    "preparing_device": "cuda",
                    "computation_dtype": torch.bfloat16,
                    "computation_device": "cuda",
                }

                te_rel_names = [f"FL2VA/text_encoder/model-{i:05d}-of-00014.safetensors" for i in range(1, 15)]
                dit_rel_names = [f"FL2VA/transformer/model-{i:05d}-of-00013.safetensors" for i in range(1, 14)]

                MODEL_DOWNLOAD_STATE["progress"] = 0.22
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 MiniMax-H3 BF16 多模态编码器权重 (14 分片)..."
                te_paths = [_resolve_and_stage_model_file("minimax_h3", fn) for fn in te_rel_names]

                MODEL_DOWNLOAD_STATE["progress"] = 0.42
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 MiniMax-H3 BF16 33B DiT 主干权重 (13 分片)..."
                dit_paths = [_resolve_and_stage_model_file("minimax_h3", fn) for fn in dit_rel_names]

                MODEL_DOWNLOAD_STATE["progress"] = 0.58
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 3D 视频 VAE 与 32kHz 立体声 VAE 权重..."
                vvae_path = _resolve_and_stage_model_file(
                    "minimax_h3", "FL2VA/video_vae/source/model.safetensors"
                )
                avae_path = _resolve_and_stage_model_file(
                    "minimax_h3", "FL2VA/audio_vae/model.safetensors"
                )

                # 基于实例可用系统内存自适应选择热备策略（严守 >=24GB CPU 内存安全余量，杜绝 OOM）：
                # - Tier 1 (如 G4 176GB RAM, 可用 >=148GB)：text_encoder(62.1GB) + dit(61.7GB) 双主干常驻系统内存，PCIe 极速互换 (~4-6s)！
                # - Tier 2 (可用 88~148GB RAM)：text_encoder(62.1GB) 常驻系统内存 (~4-6s 极速解析提示词)，dit 顺序缓存至本地高速 NVMe！
                # - Tier 3 (低内存机型 <88GB RAM)：保持磁盘按需映射，若本地 NVMe 空间充裕则脱离慢速云盘 FUSE。
                ram_residency_tag = ""
                if avail_ram_gb >= 148.0:
                    te_cfg = cpu_fast_vram_config
                    dit_cfg = cpu_fast_vram_config
                    vae_cfg = cpu_fast_vram_config
                    ram_residency_tag = "双主干内存热备"
                    print(
                        f"⚡ [G4 大内存热备] 检测到充裕系统内存 ({avail_ram_gb:.1f}/{total_ram_gb:.1f} GB)，"
                        "已启用 32B 文本编码器 + 33B DiT 双主干 DDR5 内存常驻热备（阶段切换仅需 ~5 秒）！"
                    )
                    MODEL_DOWNLOAD_STATE["progress"] = 0.68
                    MODEL_DOWNLOAD_STATE["message"] = (
                        f"正在将 MiniMax-H3 预装载至 {total_ram_gb:.0f}GB 高速系统内存热备池（首次载入后换提示词仅需 ~5 秒）..."
                    )
                elif avail_ram_gb >= 88.0:
                    te_cfg = cpu_fast_vram_config
                    dit_paths = _stage_model_files_to_local_nvme(
                        "minimax_h3", dit_paths, dit_rel_names, "MiniMax-H3 33B DiT 主干"
                    )
                    dit_cfg = vram_config
                    vae_cfg = cpu_fast_vram_config
                    ram_residency_tag = "编码器内存热备"
                    print(
                        f"⚡ [混合内存热备] 系统可用内存 {avail_ram_gb:.1f}/{total_ram_gb:.1f} GB，"
                        "已将 32B 文本编码器锁入系统内存热备（换提示词 ~5 秒极速换入），DiT 走本地高速存储！"
                    )
                    MODEL_DOWNLOAD_STATE["progress"] = 0.68
                    MODEL_DOWNLOAD_STATE["message"] = (
                        "正在将 32B 文本编码器装载至系统内存热备池（后续解析新提示词仅需 ~5 秒）..."
                    )
                else:
                    te_paths = _stage_model_files_to_local_nvme(
                        "minimax_h3", te_paths, te_rel_names, "MiniMax-H3 多模态编码器"
                    )
                    dit_paths = _stage_model_files_to_local_nvme(
                        "minimax_h3", dit_paths, dit_rel_names, "MiniMax-H3 33B DiT 主干"
                    )
                    te_cfg = vram_config
                    dit_cfg = vram_config
                    vae_cfg = vram_config
                    MODEL_DOWNLOAD_STATE["progress"] = 0.72
                    MODEL_DOWNLOAD_STATE["message"] = "正在构建 MiniMax-H3 BF16 原生 GPU 计算图与 Processor..."

                new_pipe = MiniMaxH3Pipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=te_paths, **te_cfg),
                        ModelConfig(path=dit_paths, **dit_cfg),
                        ModelConfig(path=vvae_path, **vae_cfg),
                        ModelConfig(path=avae_path, **vae_cfg),
                    ],
                    processor_config=ModelConfig(
                        model_id="MiniMaxAI/MiniMax-H3",
                        origin_file_pattern="FL2VA/processor/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                mode_label = "MiniMax-H3 · BF16 原生"
                residency_tag = _enable_high_vram_optimizations(
                    new_pipe,
                    total_vram_gb=total_vram_gb,
                    preload_vram=True,
                    skip_preload_attrs=("dit",) if free_before_gb >= 76.0 else ("text_encoder", "dit"),
                )
                tags = [t for t in (ram_residency_tag, residency_tag) if t]
                if tags:
                    mode_label = f"{mode_label} · {' / '.join(tags)}"

            elif target_id == "qwen_image_21":
                from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
                MODEL_DOWNLOAD_STATE["progress"] = 0.45
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 Qwen-Image-2.1 权重并构建 GPU 管线..."
                dit_paths = [
                    _resolve_and_stage_model_file(target_id, f"transformer/diffusion_pytorch_model-{i:05d}-of-00002.safetensors")
                    for i in range(1, 3)
                ]
                te_paths = [
                    _resolve_and_stage_model_file(target_id, f"text_encoder/model-{i:05d}-of-00004.safetensors")
                    for i in range(1, 5)
                ]
                vae_path = _resolve_and_stage_model_file(target_id, "vae/diffusion_pytorch_model.safetensors")
                new_pipe = QwenImage21Pipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=dit_paths, **vram_config),
                        ModelConfig(path=te_paths, **vram_config),
                        ModelConfig(path=vae_path, **vram_config),
                    ],
                    processor_config=ModelConfig(
                        model_id="Qwen/Qwen-Image-2.1",
                        origin_file_pattern="processor/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                _enable_high_vram_optimizations(new_pipe, total_vram_gb=total_vram_gb, preload_vram=True)
                mode_label = "Qwen-Image-2.1 · BF16"

            elif target_id == "z_image_turbo":
                from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
                MODEL_DOWNLOAD_STATE["progress"] = 0.45
                MODEL_DOWNLOAD_STATE["message"] = "正在映射 Z-Image-Turbo 权重并构建 8 步极速生图管线..."
                dit_paths = [
                    _resolve_and_stage_model_file(target_id, f"transformer/diffusion_pytorch_model-{i:05d}-of-00003.safetensors")
                    for i in range(1, 4)
                ]
                te_paths = [
                    _resolve_and_stage_model_file(target_id, f"text_encoder/model-{i:05d}-of-00003.safetensors")
                    for i in range(1, 4)
                ]
                vae_path = _resolve_and_stage_model_file(target_id, "vae/diffusion_pytorch_model.safetensors")
                new_pipe = ZImagePipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device="cuda",
                    model_configs=[
                        ModelConfig(path=dit_paths, **vram_config),
                        ModelConfig(path=te_paths, **vram_config),
                        ModelConfig(path=vae_path, **vram_config),
                    ],
                    tokenizer_config=ModelConfig(
                        model_id="Tongyi-MAI/Z-Image-Turbo",
                        origin_file_pattern="tokenizer/",
                        download_source="huggingface",
                    ),
                    vram_limit=vram_limit,
                )
                _enable_high_vram_optimizations(new_pipe, total_vram_gb=total_vram_gb, preload_vram=True)
                mode_label = "Z-Image-Turbo · 8步极速 BF16"

            elif target_id == "qwen38_27b":
                if free_before_gb < 49.0:
                    raise RuntimeError(
                        f"当前 GPU 可用显存 ({free_before_gb:.1f} GB / 总计 {total_vram_gb:.1f} GB) 不足以加载 "
                        f"Qwen3.8-27B BF16 原生权重（需约 52 GB 显存，推荐 A100 80GB）。"
                        f"若在 L4 (24GB) 或中低显存运行，请在模型中心选择独立的「Qwen3.8-27B-GGUF (UD-Q4_K_M)」量化版！"
                    )

                MODEL_DOWNLOAD_STATE["progress"] = 0.30
                MODEL_DOWNLOAD_STATE["message"] = (
                    f"正在装载 Qwen3.8-27B BF16 原生满血权重 (上下文上限 {ctx_len // 1024}K)..."
                )
                first_shard = _find_existing_model_file("qwen38_27b", "model-00001-of-00018.safetensors")
                model_dir = os.path.dirname(first_shard) if first_shard else _get_model_dirs("qwen38_27b")[2]

                # 确保配置文件在同一目录中完备
                if not os.path.exists(os.path.join(model_dir, "config.json")):
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id="Qwen/Qwen3.8-27B",
                        local_dir=model_dir,
                        ignore_patterns=["*.safetensors", "*.bin", "*.pth"],
                    )

                def _load_qwen_vl_bf16():
                    _ensure_torchaudio_compatible()
                    _ensure_qwen_fast_kernels()
                    from transformers import AutoProcessor, AutoModelForImageTextToText
                    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
                    mdl = AutoModelForImageTextToText.from_pretrained(
                        model_dir,
                        dtype=torch.bfloat16,
                        attn_implementation="sdpa",
                        device_map="cuda",
                        trust_remote_code=True,
                    )
                    mode_tag = f"Qwen3.8-27B · BF16 原生 · {ctx_len // 1024}K"
                    print("[加速引擎] 已启用 FLA Triton 线性注意力 + Triton 融合 CausalConv1d + PyTorch SDPA！")
                    mdl.eval()
                    _optimize_qwen35_for_fast_inference(mdl, max_context_len=ctx_len)
                    return mdl, processor, mode_tag

                try:
                    mdl, proc, mode_tag = _load_qwen_vl_bf16()
                except (KeyError, ValueError, ImportError) as up_e:
                    print(f"检测到旧版 transformers ({up_e})，正在自动升级最新 transformers...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "qwen-vl-utils"]
                    )
                    import importlib
                    import transformers
                    importlib.reload(transformers)
                    mdl, proc, mode_tag = _load_qwen_vl_bf16()

                new_llm = {
                    "backend": "transformers",
                    "model": mdl,
                    "processor": proc,
                    "model_id": "qwen38_27b",
                    "context_length": ctx_len,
                }
                mode_label = mode_tag

            elif target_id == "qwen38_27b_gguf":
                MODEL_DOWNLOAD_STATE["progress"] = 0.25
                MODEL_DOWNLOAD_STATE["message"] = "正在预热 Qwen3.8-27B-UD-Q4_K_M.gguf 与 mmproj 视觉权重..."
                gguf_path = _resolve_and_stage_model_file("qwen38_27b_gguf", "Qwen3.8-27B-UD-Q4_K_M.gguf")
                mmproj_path = _resolve_and_stage_model_file("qwen38_27b_gguf", "mmproj-F16.gguf")

                mtp_rel = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
                mtp_path: Optional[str] = None
                try:
                    if not _find_existing_model_file("qwen38_27b_gguf", mtp_rel):
                        MODEL_DOWNLOAD_STATE["progress"] = 0.38
                        MODEL_DOWNLOAD_STATE["message"] = "正在自动补齐 Unsloth MTP 投机解码加速头 (1.31 GB)..."
                        print("[GGUF 引擎] 检测到尚未缓存 MTP 投机解码权重，正在 8 路并发极速补齐...")
                        ram_d, nvme_d, drive_d = _get_model_dirs("qwen38_27b_gguf")
                        dl_dir = ram_d if ram_d else (drive_d if DRIVE_MOUNTED else nvme_d)
                        os.makedirs(dl_dir, exist_ok=True)
                        dl_mtp = fast_parallel_hf_download(
                            repo_id="unsloth/Qwen3.8-27B-GGUF",
                            filename=mtp_rel,
                            dest_dir=dl_dir,
                            hf_token=os.environ.get("HF_TOKEN", ""),
                            num_connections=8,
                        )
                        if DRIVE_MOUNTED and dl_dir != drive_d:
                            threading.Thread(
                                target=_async_backup_to_drive,
                                args=(dl_mtp, drive_d, mtp_rel),
                                daemon=True,
                            ).start()
                    mtp_path = _resolve_and_stage_model_file("qwen38_27b_gguf", mtp_rel)
                except Exception as mtp_e:
                    print(f"[GGUF 引擎] MTP 投机权重获取跳过，将以标准模式启动: {mtp_e}")
                    mtp_path = None

                new_llm = _start_qwen_gguf_server(
                    gguf_path=gguf_path,
                    mmproj_path=mmproj_path,
                    ctx_len=ctx_len,
                    cur_free_gb=free_before_gb,
                    mtp_path=mtp_path,
                )
                mode_label = new_llm["mode_label"]

            free_after_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            vram_delta_gb = round(max(0.5, free_before_gb - free_after_gb), 1)

            LOADED_RUNTIMES[target_id] = {
                "model_id": target_id,
                "name": spec["name"],
                "modality": spec["modality"],
                "mode_label": mode_label,
                "pipeline": new_pipe,
                "llm_bundle": new_llm,
                "vram_gb": vram_delta_gb,
                "loaded_at": time.time(),
            }
            _sync_legacy_globals()

            elapsed_s = round(time.time() - t_start, 1)
            MODEL_DOWNLOAD_STATE["progress"] = 1.0
            co_count = len(LOADED_RUNTIMES)
            evict_tip = f"（因显存上限已自动腾退 {', '.join(evicted_names)}）" if evicted_names else ""
            co_tip = f" | 当前共 {co_count} 个模型同时常驻显存，切换 0 等待" if co_count > 1 else ""
            msg = f"{spec['name']} 已装载至显存 ({mode_label}，约 {vram_delta_gb}GB，耗时 {elapsed_s}s){co_tip}！{evict_tip}{auto_switched_notice}"
            MODEL_DOWNLOAD_STATE["message"] = msg
            print(msg)
            return True, msg

        except Exception as e:
            _sync_legacy_globals()
            err = f"加载 {spec['name']} 至显存失败: {str(e)}"
            MODEL_DOWNLOAD_STATE["message"] = err
            print(err)
            return False, err
        finally:
            MODEL_DOWNLOAD_STATE["is_loading_vram"] = False
            touch_activity()


def load_minimax_h3_to_vram_core(
    precision_mode: str = "bf16",
    offload_mode: str = "fast_mmap",
    hf_token: str = "",
) -> Tuple[bool, str]:
    return load_model_to_vram_core("minimax_h3", "bf16", offload_mode, hf_token)


def _link_forward():
    # 前向引用：定义在后面的模块，函数体运行时才用；入口在全部模块加载完后统一回填，保证严格按切片顺序加载
    global _enable_high_vram_optimizations, free_comfyui_vram_core
    from .pipeline_opt import _enable_high_vram_optimizations
    from .comfy_runtime import free_comfyui_vram_core
