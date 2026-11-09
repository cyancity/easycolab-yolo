import os
import subprocess
import threading
from typing import Optional, Tuple, Dict, Any
import shutil


MODEL_DOWNLOAD_STATE: Dict[str, Any] = {
    "is_downloading": False,
    "is_loading_vram": False,
    "progress": 0.0,
    "message": "等待检查模型状态...",
}


def _get_system_ram_gb() -> Tuple[float, float]:
    """返回 (系统总内存 GB, 当前可用内存 GB)"""
    try:
        mem_total, mem_avail = 0.0, 0.0
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) / (1024 ** 2)
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) / (1024 ** 2)
        return round(mem_total, 1), round(mem_avail, 1)
    except Exception:
        return 0.0, 0.0


def setup_colab_ramdisk() -> Optional[str]:
    """
    借鉴 comfy-colab-kit/packages/colab/setup.py：
    在 Colab 大内存实例 (G4 176GB / A100 83GB / L4 53GB RAM) 上自动扩容 /dev/shm (tmpfs 内存盘，读写 2.4GB/s+)，
    为适配容量的模型提供零延迟 mmap 预热加速。
    """
    if not os.path.exists("/dev/shm"):
        return None
    total_ram_gb, avail_ram_gb = _get_system_ram_gb()
    if total_ram_gb >= 48.0:
        # 避免 DiffSynth DiskMap 默认每 2GB (10^9 参数) 反复关闭并重开全部 14 个 safetensors 分片句柄
        os.environ.setdefault("DIFFSYNTH_DISK_MAP_BUFFER_SIZE", str(80 * 10**9))
    if total_ram_gb < 40.0 or avail_ram_gb < 25.0:
        return None
    try:
        cur_shm_total_gb = shutil.disk_usage("/dev/shm").total / (1024 ** 3)
        target_gb = min(88, max(30, int(total_ram_gb * 0.50)))
        if cur_shm_total_gb + 1.0 < target_gb:
            subprocess.run(
                ["mount", "-o", f"remount,size={target_gb}G", "/dev/shm"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        shm_stat = shutil.disk_usage("/dev/shm")
        shm_free_gb = shm_stat.free / (1024 ** 3)
        if shm_free_gb >= 22.5:
            ram_root = "/dev/shm/EasyColab_Models"
            os.makedirs(ram_root, exist_ok=True)
            print(f"⚡ [RAMDisk 内存盘加速已开启] /dev/shm 可用空间 {shm_free_gb:.1f} GB (2.4GB/s 零延迟读写)")
            return ram_root
    except Exception:
        pass
    return None


RAM_ROOT_DIR = setup_colab_ramdisk()

# 如果挂载了 Google Drive，优先将模型持久化保存在云盘 EasyColab_Models（并自动兼容读取历史 Aevix_Models / MiniMax_H3_Models 目录）
DRIVE_MOUNTED = os.path.exists("/content/drive/MyDrive")


def _has_rclone_drive_remote() -> bool:
    rc_path = os.path.expanduser("~/.config/rclone/rclone.conf")
    if not os.path.exists(rc_path) or not shutil.which("rclone"):
        return False
    try:
        with open(rc_path, "r", encoding="utf-8", errors="ignore") as f:
            return "[easycolab_gdrive]" in f.read()
    except Exception:
        return False


COMFYUI_LOCK = threading.Lock()
COMFYUI_STATE: Dict[str, Any] = {
    "is_running": False,
    "is_starting": False,
    "is_syncing_drive": False,
    "is_downloading_weight": False,
    "installing_pack_id": "",
    "progress": 0.0,
    "status_message": "尚未启动（支持一键拉起 ComfyUI + Google Drive 自动软链持久化）",
    "public_url": "",
    "local_port": 8188,
    "queue_running": 0,
    "queue_pending": 0,
    "persistence_mode": "drive_symlink" if DRIVE_MOUNTED else ("rclone_drive_sync" if _has_rclone_drive_remote() else "local_only"),
    "persistence_label": "Google Drive 实时软链挂载" if DRIVE_MOUNTED else ("Google Drive 云盘自动持久化" if _has_rclone_drive_remote() else "实例本地高速存储"),
    "drive_persisted": bool(DRIVE_MOUNTED or _has_rclone_drive_remote()),
    "custom_nodes_count": 0,
    "models_count": 0,
    "workflows_count": 0,
    "last_sync_ts": 0.0,
    "last_history_count": 0,
    "comfy_proc": None,
    "tunnel_proc": None,
}
MODEL_BASE_ROOT = (
    "/content/drive/MyDrive/EasyColab_Models"
    if DRIVE_MOUNTED
    else os.path.join(os.getcwd(), "models")
)
LEGACY_DRIVE_MODEL_ROOT = "/content/drive/MyDrive/MiniMax_H3_Models"
LEGACY_DRIVE_MODEL_ROOTS = [
    "/content/drive/MyDrive/Aevix_Models",
    "/content/drive/MyDrive/MiniMax_H3_Models",
]
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = MODEL_BASE_ROOT

MINIMAX_H3_BF16_FILES = (
    [
        (f"FL2VA/transformer/model-{i:05d}-of-00013.safetensors", f"DiT 音视频主干 BF16 {i}/13")
        for i in range(1, 14)
    ]
    + [
        (f"FL2VA/text_encoder/model-{i:05d}-of-00014.safetensors", f"多模态文本编码器 BF16 {i}/14")
        for i in range(1, 15)
    ]
    + [
        ("FL2VA/video_vae/source/model.safetensors", "3D 视频 VAE 解码器 BF16 (约 9.7 GB)"),
        ("FL2VA/audio_vae/model.safetensors", "32kHz 立体声 VAE 解码器 BF16 (约 0.56 GB)"),
    ]
)

# ==============================================================================
# 开源旗舰多模型注册中心 (Hugging Face Top Models 2026 · 全量免审核 Gated=False)
# ==============================================================================
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "minimax_h3": {
        "id": "minimax_h3",
        "name": "MiniMax-H3 (33B)",
        "repo_id": "MiniMaxAI/MiniMax-H3",
        "modality": "video_audio",
        "category": "全模态音视频生成 · BF16 原生满血 · 32kHz 立体声",
        "approx_size_gb": 134.1,
        "files": MINIMAX_H3_BF16_FILES,
    },
    "qwen38_27b": {
        "id": "qwen38_27b",
        "name": "Qwen3.8-27B",
        "repo_id": "Qwen/Qwen3.8-27B",
        "modality": "llm_vlm",
        "category": "视觉理解与深度推理大模型 · BF16 原生满血",
        "approx_size_gb": 51.8,
        "files": [
            (f"model-{i:05d}-of-00018.safetensors", f"Qwen3.8-27B BF16 权重分片 {i}/18")
            for i in range(1, 19)
        ],
    },
    "qwen38_27b_gguf": {
        "id": "qwen38_27b_gguf",
        "name": "Qwen3.8-27B-GGUF",
        "repo_id": "unsloth/Qwen3.8-27B-GGUF",
        "modality": "llm_vlm",
        "category": "Unsloth UD-Q4_K_M + MTP 投机加速 · 支持 256K 上下文",
        "approx_size_gb": 18.7,
        "files": [
            ("Qwen3.8-27B-UD-Q4_K_M.gguf", "Unsloth UD-Q4_K_M 主模型权重 (约 16.5 GB)"),
            ("mmproj-F16.gguf", "多模态视觉投影器 mmproj-F16 (约 0.93 GB)"),
            ("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "Unsloth MTP 多词元并行投机解码头 (约 1.31 GB)"),
        ],
    },
    "qwen38_flash_next": {
        "id": "qwen38_flash_next",
        "name": "Qwen3.8-Flash-Next",
        "repo_id": "unsloth/Qwen3.8-Flash-Next-GGUF",
        "modality": "llm",
        "runtime": "llama",  # 独立 llama-server，规格见 runtime/llama_specs.py
        "category": "180B MoE · UD-IQ4_XS + MTP 投机加速 · 512K 上下文 (llama.cpp 独立服务)",
        "approx_size_gb": 91.1,
        # 与 runtime/flash_next.py 默认量化 UD-IQ4_XS + MTP 完整头一致；其他量化由加载时补下载，不计入这里的状态
        "files": [
            ("UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf", "UD-IQ4_XS 元数据分片 1/4 (约 11 MB)"),
            ("UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf", "UD-IQ4_XS 主权重分片 2/4 (约 49.6 GB)"),
            ("UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf", "UD-IQ4_XS 主权重分片 3/4 (约 15.3 GB)"),
            ("UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf", "Engram 词表分片 4/4 (约 28.8 GB，驻 CPU 内存)"),
            ("MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "MTP 多词元并行投机解码头 (约 4.1 GB)"),
        ],
    },
    "qwen35_4b_smoke": {
        "id": "qwen35_4b_smoke",
        "name": "Qwen3.5-4B 链路冒烟",
        "repo_id": "unsloth/Qwen3.5-4B-MTP-GGUF",
        "modality": "llm",
        "runtime": "llama",
        "category": "4B · UD-Q4_K_XL + MTP · 与 Flash-Next 同一套 llama-server 链路，T4 可跑，仅用于验证链路",
        "approx_size_gb": 3.0,
        "files": [
            ("Qwen3.5-4B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL 主权重 (约 3.0 GB，MTP 头已内置)"),
        ],
    },
    "qwen_image_21": {
        "id": "qwen_image_21",
        "name": "Qwen-Image-2.1",
        "repo_id": "Qwen/Qwen-Image-2.1",
        "modality": "image",
        "category": "旗舰文生图与多图指令修图",
        "approx_size_gb": 30.9,
        "files": [
            ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "DiT 图像主干 1/2"),
            ("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "DiT 图像主干 2/2"),
            ("text_encoder/model-00001-of-00004.safetensors", "多模态编码器 1/4"),
            ("text_encoder/model-00002-of-00004.safetensors", "多模态编码器 2/4"),
            ("text_encoder/model-00003-of-00004.safetensors", "多模态编码器 3/4"),
            ("text_encoder/model-00004-of-00004.safetensors", "多模态编码器 4/4"),
            ("vae/diffusion_pytorch_model.safetensors", "高保真图像 VAE"),
        ],
    },
    "z_image_turbo": {
        "id": "z_image_turbo",
        "name": "Z-Image-Turbo",
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "modality": "image",
        "category": "8 步极速超清写实与中英文字生图",
        "approx_size_gb": 30.6,
        "files": [
            ("transformer/diffusion_pytorch_model-00001-of-00003.safetensors", "Z-Image DiT 主干 1/3"),
            ("transformer/diffusion_pytorch_model-00002-of-00003.safetensors", "Z-Image DiT 主干 2/3"),
            ("transformer/diffusion_pytorch_model-00003-of-00003.safetensors", "Z-Image DiT 主干 3/3"),
            ("text_encoder/model-00001-of-00003.safetensors", "文本编码器 1/3"),
            ("text_encoder/model-00002-of-00003.safetensors", "文本编码器 2/3"),
            ("text_encoder/model-00003-of-00003.safetensors", "文本编码器 3/3"),
            ("vae/diffusion_pytorch_model.safetensors", "图像 VAE 解码器"),
        ],
    },
}

DEFAULT_OUTPUT_DIR = (
    "/content/drive/MyDrive/EasyColab_Outputs"
    if DRIVE_MOUNTED
    else os.path.join(os.getcwd(), "outputs")
)
LEGACY_DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/MiniMax_H3_Outputs"
LEGACY_DRIVE_OUTPUT_DIRS = [
    "/content/drive/MyDrive/Aevix_Outputs",
    "/content/drive/MyDrive/MiniMax_H3_Outputs",
]
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "models"), exist_ok=True)
