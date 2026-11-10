import os
import time
import subprocess
import threading
from typing import List, Tuple, Dict, Any
import json
import shutil
import urllib.parse
import urllib.request
from . import state
from .config import _has_rclone_drive_remote, COMFYUI_STATE
from .idle import touch_activity
from .pipeline_opt import _reclaim_transient_gpu_memory
from . import comfy_guard, comfy_tunnel
from .comfy_presets import _COMFY_STATE_FILE, _COMFY_STATUS_CACHE, _ensure_comfyui_beginner_assets
from .comfy_workflows import _get_comfy_preset_packs_status


def _resolve_comfyui_persistence_paths() -> Tuple[str, str, str, str, bool]:
    comfy_root = "/content/ComfyUI" if os.path.exists("/content") else os.path.join(os.getcwd(), "ComfyUI")
    if os.path.exists("/content/drive/MyDrive"):
        persist_root = "/content/drive/MyDrive/EasyColab_ComfyUI"
        return comfy_root, persist_root, "drive_symlink", "Google Drive 实时软链直连", True
    if _has_rclone_drive_remote():
        persist_root = "/content/drive_mirror/EasyColab_ComfyUI"
        return comfy_root, persist_root, "rclone_drive_sync", "Google Drive 自动软链持久化", True
    persist_root = (
        "/content/EasyColab_ComfyUI_Data"
        if os.path.exists("/content")
        else os.path.join(os.getcwd(), "EasyColab_ComfyUI_Data")
    )
    return comfy_root, persist_root, "local_only", "实例本地高速存储", False


def _safe_symlink_dir(target_dir: str, link_path: str):
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    if os.path.islink(link_path):
        cur_target = os.readlink(link_path)
        if os.path.abspath(cur_target) == os.path.abspath(target_dir):
            return
        os.unlink(link_path)
    elif os.path.isdir(link_path):
        for entry in os.listdir(link_path):
            src_item = os.path.join(link_path, entry)
            dst_item = os.path.join(target_dir, entry)
            if not os.path.exists(dst_item):
                try:
                    shutil.move(src_item, dst_item)
                except Exception:
                    pass
        shutil.rmtree(link_path, ignore_errors=True)
    elif os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(target_dir, link_path)


def _scan_comfyui_counts_and_outputs() -> Tuple[Tuple[int, int, int], List[Dict[str, Any]]]:
    now = time.time()
    if (now - _COMFY_STATUS_CACHE["ts"]) < 6.0:
        return _COMFY_STATUS_CACHE["counts"], list(_COMFY_STATUS_CACHE["outputs"])

    comfy_root, persist_root, _, _, _ = _resolve_comfyui_persistence_paths()
    cn_count = 0
    cn_base = os.path.join(comfy_root, "custom_nodes")
    if not os.path.isdir(cn_base):
        cn_base = os.path.join(persist_root, "custom_nodes")
    if os.path.isdir(cn_base):
        try:
            for d in os.listdir(cn_base):
                if d.startswith(".") or d == "__pycache__":
                    continue
                if os.path.isdir(os.path.join(cn_base, d)):
                    cn_count += 1
        except Exception:
            pass

    model_exts = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
    m_count = 0
    m_base = os.path.join(persist_root, "models")
    if os.path.isdir(m_base):
        try:
            for root_d, _, fnames in os.walk(m_base):
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in model_exts:
                        m_count += 1
        except Exception:
            pass

    wf_count = 0
    wf_dir = os.path.join(persist_root, "user", "default", "workflows")
    if os.path.isdir(wf_dir):
        try:
            wf_count = sum(1 for fn in os.listdir(wf_dir) if fn.lower().endswith(".json"))
        except Exception:
            pass

    recent_outputs: List[Dict[str, Any]] = []
    out_dir = os.path.join(comfy_root, "output")
    if not os.path.isdir(out_dir):
        out_dir = os.path.join(persist_root, "output")
    valid_media = {".mp4", ".webm", ".mov", ".webp", ".gif", ".png", ".jpg", ".jpeg"}
    if os.path.isdir(out_dir):
        collected = []
        try:
            for root_d, _, fnames in os.walk(out_dir):
                for fn in fnames:
                    if fn.startswith("."):
                        continue
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in valid_media:
                        continue
                    full_p = os.path.join(root_d, fn)
                    try:
                        st = os.stat(full_p)
                        if st.st_size < 512:
                            continue
                        rel_p = os.path.relpath(full_p, out_dir).replace("\\", "/")
                        mtype = "video" if ext in {".mp4", ".webm", ".mov"} else "image"
                        collected.append((st.st_mtime, {
                            "rel_path": rel_p,
                            "filename": fn,
                            "media_type": mtype,
                            "size_mb": round(st.st_size / (1024 ** 2), 2),
                            "mtime": int(st.st_mtime),
                        }))
                    except Exception:
                        pass
            collected.sort(key=lambda x: x[0], reverse=True)
            recent_outputs = [item for _, item in collected[:16]]
        except Exception:
            pass

    _COMFY_STATUS_CACHE["ts"] = now
    _COMFY_STATUS_CACHE["counts"] = (cn_count, m_count, wf_count)
    _COMFY_STATUS_CACHE["outputs"] = recent_outputs
    return (cn_count, m_count, wf_count), recent_outputs


def _try_adopt_running_comfyui():
    """若 app.py 热更新重启前实例上已有在线 ComfyUI 与画布隧道，自动无缝接管，不断开用户已打开的画布"""
    if state._COMFY_ADOPT_CHECKED or COMFYUI_STATE.get("is_running") or COMFYUI_STATE.get("is_starting"):
        return
    state._COMFY_ADOPT_CHECKED = True

    port = int(COMFYUI_STATE.get("local_port") or 8188)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/system_stats", timeout=1.0) as resp:
            if resp.status != 200:
                return
    except Exception:
        return

    try:
        guard_port = comfy_guard.ensure_guard(port)
    except Exception as e:  # noqa: BLE001
        print(f"[ComfyUI] 访问保护启动失败，不接管：{e}")
        # 旧版隧道直连 ComfyUI，没有保护就不能留着
        comfy_tunnel.kill_tunnels(port + 1)
        return

    pub_url = ""
    if os.path.isfile(_COMFY_STATE_FILE) and comfy_tunnel.tunnel_alive(guard_port):
        try:
            st_obj = json.load(open(_COMFY_STATE_FILE, "r", encoding="utf-8"))
            if int(st_obj.get("guard_port") or 0) == guard_port:  # 旧版隧道直连 ComfyUI，不能沿用
                pub_url = str(st_obj.get("public_url") or "").strip()
        except Exception:
            pub_url = ""

    if not pub_url:
        try:
            pub_url, COMFYUI_STATE["tunnel_proc"] = comfy_tunnel.open_tunnel(guard_port)
        except Exception as e:  # noqa: BLE001
            print(f"[ComfyUI] 画布隧道没建起来：{e}")
            pub_url = ""

    comfy_root, persist_root, _, label, _ = _resolve_comfyui_persistence_paths()
    _ensure_comfyui_beginner_assets(comfy_root, persist_root)
    COMFYUI_STATE["is_running"] = True
    COMFYUI_STATE["public_url"] = pub_url
    COMFYUI_STATE["progress"] = 1.0
    COMFYUI_STATE["status_message"] = f"已无缝接管在线 ComfyUI 画布（{label}）"
    if pub_url:
        try:
            with open(_COMFY_STATE_FILE, "w", encoding="utf-8") as sf:
                json.dump({"public_url": pub_url, "port": port, "guard_port": guard_port}, sf)
        except Exception:
            pass
    if not state._COMFY_BG_SYNC_STARTED:
        state._COMFY_BG_SYNC_STARTED = True
        threading.Thread(target=_comfyui_periodic_drive_sync_loop, daemon=True).start()
    print(f"[ComfyUI] 已自动接管实例上正在运行的 ComfyUI 服务（{'画布隧道在线' if pub_url else '画布隧道没建起来'}）")


def _poll_local_comfyui_runtime_quiet():
    proc = COMFYUI_STATE.get("comfy_proc")
    if proc is not None and proc.poll() is not None:
        COMFYUI_STATE["is_running"] = False
        COMFYUI_STATE["queue_running"] = 0
        COMFYUI_STATE["queue_pending"] = 0
        if not COMFYUI_STATE["is_starting"]:
            COMFYUI_STATE["status_message"] = f"ComfyUI 进程已退出 (code={proc.returncode})，可点击重新启动"
        COMFYUI_STATE["comfy_proc"] = None
        return

    if not COMFYUI_STATE.get("is_running"):
        _try_adopt_running_comfyui()
        if not COMFYUI_STATE.get("is_running"):
            return

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    port = int(COMFYUI_STATE.get("local_port") or 8188)
    try:
        with opener.open(f"http://127.0.0.1:{port}/queue", timeout=1.2) as r:
            q_data = json.loads(r.read().decode("utf-8", errors="replace"))
        q_run = len(q_data.get("queue_running") or [])
        q_pend = len(q_data.get("queue_pending") or [])
        COMFYUI_STATE["queue_running"] = q_run
        COMFYUI_STATE["queue_pending"] = q_pend
        if q_run > 0 or q_pend > 0:
            touch_activity()
    except Exception:
        pass

    try:
        with opener.open(f"http://127.0.0.1:{port}/history?max_items=5", timeout=1.2) as r:
            h_data = json.loads(r.read().decode("utf-8", errors="replace"))
        if isinstance(h_data, dict):
            h_sig = hash(tuple(sorted(h_data.keys())))
            if h_sig != COMFYUI_STATE.get("last_history_count", 0) and COMFYUI_STATE.get("last_history_count", 0) != 0:
                touch_activity()
                _COMFY_STATUS_CACHE["ts"] = 0.0
            COMFYUI_STATE["last_history_count"] = h_sig
    except Exception:
        pass


def get_comfyui_status_dict() -> Dict[str, Any]:
    _, persist_root, mode, label, is_drive = _resolve_comfyui_persistence_paths()
    (cn_count, m_count, wf_count), recent_outputs = _scan_comfyui_counts_and_outputs()
    preset_packs = _get_comfy_preset_packs_status(persist_root)
    COMFYUI_STATE["persistence_mode"] = mode
    COMFYUI_STATE["persistence_label"] = label
    COMFYUI_STATE["drive_persisted"] = is_drive
    COMFYUI_STATE["custom_nodes_count"] = cn_count
    COMFYUI_STATE["models_count"] = m_count
    COMFYUI_STATE["workflows_count"] = wf_count
    return {
        "is_running": bool(COMFYUI_STATE["is_running"]),
        "is_starting": bool(COMFYUI_STATE["is_starting"]),
        "is_syncing_drive": bool(COMFYUI_STATE["is_syncing_drive"]),
        "is_downloading_weight": bool(COMFYUI_STATE["is_downloading_weight"]),
        "installing_pack_id": str(COMFYUI_STATE.get("installing_pack_id") or ""),
        "last_pack_result": dict(COMFYUI_STATE.get("last_pack_result") or {}),
        "progress": float(COMFYUI_STATE["progress"]),
        "status_message": str(COMFYUI_STATE["status_message"]),
        "public_url": str(COMFYUI_STATE["public_url"] or ""),
        # 访问保护的令牌：这个接口在实例密钥路径下，只有本机工作台读得到
        "token": comfy_guard.token() if COMFYUI_STATE["public_url"] else "",
        "local_port": int(COMFYUI_STATE["local_port"]),
        "queue_running": int(COMFYUI_STATE["queue_running"]),
        "queue_pending": int(COMFYUI_STATE["queue_pending"]),
        "persistence_mode": mode,
        "persistence_label": label,
        "drive_persisted": is_drive,
        "persist_root": persist_root,
        "custom_nodes_count": cn_count,
        "models_count": m_count,
        "workflows_count": wf_count,
        "last_sync_ts": int(COMFYUI_STATE["last_sync_ts"]),
        "recent_outputs": recent_outputs,
        "preset_packs": preset_packs,
    }


def free_comfyui_vram_core() -> Tuple[bool, str]:
    if not COMFYUI_STATE.get("is_running"):
        return False, "ComfyUI 当前未运行"
    port = int(COMFYUI_STATE.get("local_port") or 8188)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(req, timeout=8.0) as _:
            pass
        _reclaim_transient_gpu_memory()
        msg = "已释放 ComfyUI 驻留模型与显存缓存（画布保持在线）"
        COMFYUI_STATE["status_message"] = msg
        print(f"[ComfyUI] {msg}")
        return True, msg
    except Exception as e:
        return False, f"释放 ComfyUI 显存失败: {e}"


def _link_forward():
    # 前向引用：定义在后面的模块，函数体运行时才用；入口在全部模块加载完后统一回填，保证严格按切片顺序加载
    global _comfyui_periodic_drive_sync_loop
    from .comfy_sync import _comfyui_periodic_drive_sync_loop
