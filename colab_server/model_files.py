import os
from typing import Optional, Tuple, Dict, Any
import shutil
from . import state
from .config import DRIVE_MOUNTED, MODEL_BASE_ROOT, LEGACY_DRIVE_MODEL_ROOTS, MODEL_CATALOG


def _get_model_dirs(model_id: str) -> Tuple[Optional[str], str, str]:
    """返回指定模型的 (RAM内存盘目录或None, 本地高速NVMe目录, Google Drive/持久盘目录)"""
    spec = MODEL_CATALOG.get(model_id, MODEL_CATALOG["minimax_h3"])
    repo_sub = spec["repo_id"]
    ram_dir = None
    if os.path.exists("/dev/shm"):
        try:
            shm_free_gb = shutil.disk_usage("/dev/shm").free / (1024 ** 3)
            # 仅当 /dev/shm 可用空间足以容纳模型时才把新模型放入内存盘
            if shm_free_gb >= spec["approx_size_gb"] + 3.0:
                ram_dir = f"/dev/shm/EasyColab_Models/{model_id}"
            elif spec["approx_size_gb"] <= 60.0 and (
                os.path.exists(f"/dev/shm/EasyColab_Models/{model_id}")
                or os.path.exists(f"/dev/shm/Aevix_Models/{model_id}")
                or os.path.exists(f"/dev/shm/OpenModels/{model_id}")
            ):
                ram_dir = f"/dev/shm/EasyColab_Models/{model_id}"
        except Exception:
            pass
    nvme_dir = os.path.join(os.getcwd(), "models", repo_sub)
    drive_dir = os.path.join(MODEL_BASE_ROOT, repo_sub)
    return ram_dir, nvme_dir, drive_dir


def _find_existing_model_file(model_id: str, fname: str) -> Optional[str]:
    """按速度优先级查找已存在的模型权重文件：1. /dev/shm 内存盘 -> 2. 本地 NVMe -> 3. Google Drive (含历史目录与 HuggingFace 缓存兼容)"""
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(model_id)
    spec = MODEL_CATALOG.get(model_id, MODEL_CATALOG["minimax_h3"])
    repo_sub = spec["repo_id"]
    candidates = []
    if ram_dir:
        candidates.append(os.path.join(ram_dir, fname))
    candidates.append(os.path.join(f"/dev/shm/EasyColab_Models/{model_id}", fname))
    candidates.append(os.path.join(f"/dev/shm/Aevix_Models/{model_id}", fname))
    candidates.append(os.path.join(f"/dev/shm/OpenModels/{model_id}", fname))
    candidates.append(os.path.join(nvme_dir, fname))
    candidates.append(os.path.join(drive_dir, fname))
    if DRIVE_MOUNTED:
        for leg_root in LEGACY_DRIVE_MODEL_ROOTS:
            if os.path.isdir(leg_root):
                candidates.append(os.path.join(leg_root, repo_sub, fname))
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 64 * 1024:
            return p
    return None


def check_model_files_status(model_id: Optional[str] = None) -> Dict[str, Any]:
    """检查指定开源模型在内存盘/磁盘/云盘上的权重文件完整性与大小"""
    target_id = model_id if (model_id and model_id in MODEL_CATALOG) else state.SELECTED_MODEL_ID
    spec = MODEL_CATALOG[target_id]
    req_files = spec["files"]
    ram_dir, nvme_dir, drive_dir = _get_model_dirs(target_id)

    files_info = []
    downloaded_count = 0
    total_size_gb = 0.0
    ram_cached_count = 0

    for fname, desc in req_files:
        actual_path = _find_existing_model_file(target_id, fname)
        if actual_path:
            size_gb = round(os.path.getsize(actual_path) / (1024 ** 3), 2)
            total_size_gb += size_gb
            downloaded_count += 1
            in_ram = actual_path.startswith("/dev/shm/")
            if in_ram:
                ram_cached_count += 1
            files_info.append({
                "name": fname,
                "desc": desc + (" [RAM内存盘]" if in_ram else ""),
                "exists": True,
                "size_gb": size_gb,
            })
        else:
            files_info.append({
                "name": fname,
                "desc": desc,
                "exists": False,
                "size_gb": 0.0,
            })

    all_downloaded = downloaded_count == len(req_files) and len(req_files) > 0
    return {
        "selected_model_id": target_id,
        "selected_model_name": spec["name"],
        "selected_model_repo": spec["repo_id"],
        "selected_model_modality": spec["modality"],
        "files": files_info,
        "downloaded_count": downloaded_count,
        "total_count": len(req_files),
        "all_downloaded": all_downloaded,
        "total_size_gb": round(total_size_gb, 2),
        "drive_persisted": DRIVE_MOUNTED,
        "ramdisk_enabled": ram_dir is not None,
        "ram_cached_count": ram_cached_count,
        "model_storage_path": ram_dir or drive_dir,
        "bf16_shards_count": downloaded_count if target_id == "minimax_h3" else 0,
    }
