import os
import re
import sys
import time
import subprocess
from typing import List, Tuple, Dict, Any
import json
import shutil
import urllib.parse
import urllib.request
from .config import _has_rclone_drive_remote, COMFYUI_LOCK, COMFYUI_STATE
from .pipeline_opt import _reclaim_transient_gpu_memory
from .comfy_presets import _COMFY_PROTECTED_PIP_PKGS, _COMFY_STATE_FILE, _COMFY_STATUS_CACHE
from .comfy_runtime import (
    _resolve_comfyui_persistence_paths,
    _safe_symlink_dir,
    _scan_comfyui_counts_and_outputs,
)


def sync_comfyui_to_drive_core(quiet: bool = False) -> Tuple[bool, str]:
    if COMFYUI_STATE["is_syncing_drive"]:
        return True, "正在后台同步 ComfyUI 数据到 Google Drive..."
    comfy_root, persist_root, mode, label, is_drive = _resolve_comfyui_persistence_paths()
    if not os.path.isdir(comfy_root) and not os.path.isdir(persist_root):
        return False, "ComfyUI 尚未初始化"

    COMFYUI_STATE["is_syncing_drive"] = True
    try:
        os.makedirs(os.path.join(persist_root, "models"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "custom_nodes"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "user"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "output"), exist_ok=True)
        os.makedirs(os.path.join(persist_root, "input"), exist_ok=True)

        # 1. 自动接管自定义节点在 models/ 下新建的非软链目录，迁移至持久化目录并建立软链
        models_dir = os.path.join(comfy_root, "models")
        if os.path.isdir(models_dir):
            for entry in os.listdir(models_dir):
                if entry in ("configs", "__pycache__") or entry.startswith("."):
                    continue
                local_sub = os.path.join(models_dir, entry)
                if os.path.isdir(local_sub) and not os.path.islink(local_sub):
                    persist_sub = os.path.join(persist_root, "models", entry)
                    _safe_symlink_dir(persist_sub, local_sub)

        # 2. 将本地高速 NVMe 上的 custom_nodes 增量镜像同步到持久化目录，并记录 Git 源清单
        cn_local = os.path.join(comfy_root, "custom_nodes")
        cn_persist = os.path.join(persist_root, "custom_nodes")
        manifest: Dict[str, Any] = {}
        manifest_path = os.path.join(persist_root, "custom_nodes_manifest.json")
        if os.path.exists(manifest_path):
            try:
                manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
            except Exception:
                manifest = {}

        if os.path.isdir(cn_local):
            for entry in sorted(os.listdir(cn_local)):
                if entry.startswith(".") or entry == "__pycache__":
                    continue
                src_node = os.path.join(cn_local, entry)
                if not os.path.isdir(src_node):
                    continue
                dst_node = os.path.join(cn_persist, entry)
                if shutil.which("rsync"):
                    subprocess.run(
                        ["rsync", "-a", "--delete", "--exclude=__pycache__", f"{src_node}/", f"{dst_node}/"],
                        check=False,
                    )
                else:
                    shutil.copytree(
                        src_node,
                        dst_node,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    )
                git_cfg = os.path.join(src_node, ".git", "config")
                if os.path.exists(git_cfg):
                    try:
                        cfg_txt = open(git_cfg, "r", encoding="utf-8", errors="ignore").read()
                        m_url = re.search(r"url\s*=\s*(\S+)", cfg_txt)
                        if m_url:
                            manifest[entry] = {"git_url": m_url.group(1).strip()}
                    except Exception:
                        pass

            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, ensure_ascii=False, indent=2)

        # 3. 若处于 rclone 云盘镜像模式，通过 Google Cloud 内网 8 并发同步至 Google Drive
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote():
            if not quiet:
                print("[ComfyUI 云盘同步] 正在通过高速通道将自定义节点、模型与工作流同步至 Google Drive...")
            subprocess.run(
                [
                    "rclone", "copy",
                    persist_root,
                    "easycolab_gdrive:EasyColab_Drive/EasyColab_ComfyUI",
                    "--transfers", "8",
                    "--checkers", "16",
                    "--drive-chunk-size", "64M",
                    "--exclude", "**/__pycache__/**",
                    "--exclude", "**/*.part",
                    "--exclude", "**/*.tmp",
                ],
                check=False,
            )

        COMFYUI_STATE["last_sync_ts"] = time.time()
        _COMFY_STATUS_CACHE["ts"] = 0.0
        (cn_count, m_count, wf_count), _ = _scan_comfyui_counts_and_outputs()
        msg = f"已持久化至 {label}（节点 {cn_count} 个 · 权重 {m_count} 个 · 工作流 {wf_count} 个）"
        if not quiet:
            COMFYUI_STATE["status_message"] = msg
            print(f"[ComfyUI 云盘持久化] {msg}")
        return True, msg
    except Exception as e:
        err = f"ComfyUI 云盘同步异常: {e}"
        if not quiet:
            print(f"[ComfyUI] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_syncing_drive"] = False


def _comfyui_periodic_drive_sync_loop():
    while True:
        time.sleep(60.0)
        if COMFYUI_STATE.get("is_running") and not COMFYUI_STATE.get("is_starting"):
            if COMFYUI_STATE.get("queue_running", 0) == 0:
                sync_comfyui_to_drive_core(quiet=True)


def _ensure_comfyui_manager_tunnel_config(comfy_root: str, persist_root: str):
    """允许通过 trycloudflare 公网隧道在 ComfyUI-Manager 内直接安装 Git 节点与模型，不受默认 normal 安全锁拦截"""
    cfg_targets = [
        os.path.join(persist_root, "user", "default", "ComfyUI-Manager", "config.ini"),
        os.path.join(persist_root, "user", "__manager", "config.ini"),
        os.path.join(comfy_root, "custom_nodes", "ComfyUI-Manager", "config.ini"),
    ]
    for cfg_p in cfg_targets:
        try:
            os.makedirs(os.path.dirname(cfg_p), exist_ok=True)
            if os.path.exists(cfg_p):
                txt = open(cfg_p, "r", encoding="utf-8", errors="ignore").read()
                if "security_level" in txt:
                    # 归位到 Manager 默认的 normal 档：weak 档允许经 Web 界面直接安装任意 Git
                    # 自定义节点并执行其代码，等于把隧道 URL 变成未认证 RCE 入口
                    new_txt = re.sub(r"security_level\s*=\s*\S+", "security_level = normal", txt)
                    if new_txt != txt:
                        with open(cfg_p, "w", encoding="utf-8") as wf:
                            wf.write(new_txt)
                    continue
            with open(cfg_p, "w", encoding="utf-8") as wf:
                wf.write("[default]\nsecurity_level = normal\npreview_method = auto\n")
        except Exception:
            pass


def _install_persisted_custom_node_requirements(comfy_root: str):
    """扫描已从 Google Drive 恢复的所有自定义节点 requirements.txt，过滤掉 torch/xformers 后批量补齐依赖"""
    import hashlib
    cn_dir = os.path.join(comfy_root, "custom_nodes")
    if not os.path.isdir(cn_dir):
        return
    collected_reqs: List[str] = []
    for entry in sorted(os.listdir(cn_dir)):
        req_file = os.path.join(cn_dir, entry, "requirements.txt")
        if not os.path.isfile(req_file):
            continue
        try:
            for raw_line in open(req_file, "r", encoding="utf-8", errors="ignore"):
                line = raw_line.split("#", 1)[0].strip()
                if not line or line.startswith("-"):
                    continue
                if _COMFY_PROTECTED_PIP_PKGS.match(line):
                    continue
                if line not in collected_reqs:
                    collected_reqs.append(line)
        except Exception:
            pass

    if not collected_reqs:
        return
    digest = hashlib.sha256("\n".join(collected_reqs).encode("utf-8")).hexdigest()
    stamp_file = "/tmp/.easycolab_comfy_node_reqs.sha256"
    if os.path.exists(stamp_file):
        try:
            if open(stamp_file, "r").read().strip() == digest:
                return
        except Exception:
            pass

    print(f"[ComfyUI] 正在为已持久化的自定义节点自动恢复 {len(collected_reqs)} 项 Python 依赖...")
    tmp_req = "/tmp/easycolab_comfy_merged_reqs.txt"
    with open(tmp_req, "w", encoding="utf-8") as wf:
        wf.write("\n".join(collected_reqs) + "\n")
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q", "-r", tmp_req])
    if rc == 0:
        with open(stamp_file, "w") as sf:
            sf.write(digest)


def _ensure_cloudflared_binary() -> str:
    cf_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if os.path.isfile(cf_bin) and os.access(cf_bin, os.X_OK):
        return cf_bin
    dl_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    os.makedirs(os.path.dirname(cf_bin), exist_ok=True)
    urllib.request.urlretrieve(dl_url, cf_bin)
    os.chmod(cf_bin, 0o755)
    return cf_bin


def stop_comfyui_core(sync_before_stop: bool = True) -> Tuple[bool, str]:
    with COMFYUI_LOCK:
        if sync_before_stop and (COMFYUI_STATE.get("is_running") or os.path.isdir("/content/ComfyUI")):
            sync_comfyui_to_drive_core(quiet=True)
        for key in ("tunnel_proc", "comfy_proc"):
            proc = COMFYUI_STATE.get(key)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                COMFYUI_STATE[key] = None
        subprocess.run(["pkill", "-f", "ComfyUI/main.py"], check=False)
        subprocess.run(["pkill", "-f", "cloudflared tunnel --url http://127.0.0.1:"], check=False)
        subprocess.run(["pkill", "-f", "colab_server.comfy_guard"], check=False)
        if os.path.exists(_COMFY_STATE_FILE):
            try:
                os.remove(_COMFY_STATE_FILE)
            except Exception:
                pass
        _reclaim_transient_gpu_memory()
        COMFYUI_STATE["is_running"] = False
        COMFYUI_STATE["is_starting"] = False
        COMFYUI_STATE["public_url"] = ""
        COMFYUI_STATE["queue_running"] = 0
        COMFYUI_STATE["queue_pending"] = 0
        COMFYUI_STATE["progress"] = 0.0
        COMFYUI_STATE["status_message"] = "ComfyUI 已停止（节点、模型与工作流已持久化保存至 Google Drive）"
        print("[ComfyUI] 服务与公网隧道已停止，GPU 显存已回收。")
        return True, COMFYUI_STATE["status_message"]
