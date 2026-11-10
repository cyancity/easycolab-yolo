import os
import re
import sys
import time
import subprocess
import threading
from typing import Tuple
import json
import shutil
import urllib.parse
import urllib.request
from . import state
from .config import _has_rclone_drive_remote, COMFYUI_STATE
from .hf_download import fast_parallel_hf_download
from .idle import touch_activity
from .vram import unload_model_from_vram_core
from . import comfy_guard, comfy_tunnel
from .comfy_presets import (
    COMFY_STANDARD_MODEL_SUBDIRS,
    _COMFY_PROTECTED_PIP_PKGS,
    _COMFY_STATE_FILE,
    _COMFY_STATUS_CACHE,
    COMFY_PRESET_WORKFLOW_PACKS,
    _ensure_comfyui_beginner_assets,
    _ensure_easycolab_comfy_preset_helper,
)
from .comfy_workflows import _build_and_install_comfy_preset_workflows
from .comfy_runtime import (
    _resolve_comfyui_persistence_paths,
    _safe_symlink_dir,
    _scan_comfyui_counts_and_outputs,
)
from .comfy_sync import (
    sync_comfyui_to_drive_core,
    _comfyui_periodic_drive_sync_loop,
    _ensure_comfyui_manager_tunnel_config,
    _install_persisted_custom_node_requirements,
    stop_comfyui_core,
)


def start_comfyui_core(unload_existing: bool = False, force_restart: bool = False) -> Tuple[bool, str]:
    touch_activity()
    if COMFYUI_STATE["is_starting"]:
        return True, "ComfyUI 正在部署启动中，请稍候..."
    if COMFYUI_STATE["is_running"] and COMFYUI_STATE.get("public_url") and not force_restart:
        return True, f"ComfyUI 已在运行: {COMFYUI_STATE['public_url']}"

    if force_restart and COMFYUI_STATE["is_running"]:
        stop_comfyui_core(sync_before_stop=True)

    COMFYUI_STATE["is_starting"] = True
    COMFYUI_STATE["progress"] = 0.06
    COMFYUI_STATE["status_message"] = "正在初始化 ComfyUI 环境与 Google Drive 持久化目录..."

    try:
        if unload_existing:
            COMFYUI_STATE["status_message"] = "正在释放现有常驻模型显存，为 ComfyUI 腾出完整 96GB 空间..."
            unload_model_from_vram_core("all")

        comfy_root, persist_root, mode, label, is_drive = _resolve_comfyui_persistence_paths()

        # 1. 若为 rclone 云盘同步模式且本实例首次拉起，先从 Google Drive 高速拉取已持久化的节点、模型与工作流
        pull_flag = "/tmp/.easycolab_comfy_drive_pulled"
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote() and not os.path.exists(pull_flag):
            COMFYUI_STATE["progress"] = 0.14
            COMFYUI_STATE["status_message"] = "正在从 Google Drive 恢复历史持久化的 ComfyUI 节点、模型与工作流..."
            print("[ComfyUI] 正在从 Google Drive 同步已保存的节点、模型与工作流...")
            os.makedirs(persist_root, exist_ok=True)
            subprocess.run(
                [
                    "rclone", "copy",
                    "easycolab_gdrive:EasyColab_Drive/EasyColab_ComfyUI",
                    persist_root,
                    "--transfers", "8",
                    "--checkers", "16",
                    "--drive-chunk-size", "64M",
                ],
                check=False,
            )
            with open(pull_flag, "w") as pf:
                pf.write("1")

        # 2. 克隆或检查 ComfyUI 主程序 (置于本地高速 NVMe，保证秒级导入)
        if not os.path.exists(os.path.join(comfy_root, "main.py")):
            COMFYUI_STATE["progress"] = 0.24
            COMFYUI_STATE["status_message"] = "正在克隆官方最新 ComfyUI 核心引擎..."
            print("[ComfyUI] 正在克隆官方最新 ComfyUI 仓库...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", comfy_root],
                check=True,
            )
            COMFYUI_STATE["progress"] = 0.36
            COMFYUI_STATE["status_message"] = "正在安装 ComfyUI 核心依赖（保留当前 G4 Blackwell CUDA PyTorch）..."
            raw_reqs = open(os.path.join(comfy_root, "requirements.txt"), "r", encoding="utf-8").readlines()
            safe_reqs = [
                ln.strip() for ln in raw_reqs
                if ln.strip() and not ln.strip().startswith("#") and not _COMFY_PROTECTED_PIP_PKGS.match(ln)
            ]
            tmp_core_req = "/tmp/comfy_core_safe_reqs.txt"
            with open(tmp_core_req, "w", encoding="utf-8") as wf:
                wf.write("\n".join(safe_reqs) + "\n")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", tmp_core_req], check=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "gguf", "opencv-python-headless", "imageio-ffmpeg", "gitpython", "toml"],
                check=False,
            )

        # 3. 建立 Google Drive 软链体系：models/* 各子目录、user 工作流配置、output 成片输出全部自动软链至持久化根目录
        COMFYUI_STATE["progress"] = 0.48
        COMFYUI_STATE["status_message"] = f"正在建立 {label} 目录软链 (models / workflows / output)..."
        os.makedirs(os.path.join(persist_root, "models"), exist_ok=True)
        all_model_subdirs = set(COMFY_STANDARD_MODEL_SUBDIRS)
        for existing_sub in os.listdir(os.path.join(persist_root, "models")):
            if not existing_sub.startswith(".") and os.path.isdir(os.path.join(persist_root, "models", existing_sub)):
                all_model_subdirs.add(existing_sub)
        for sub in sorted(all_model_subdirs):
            _safe_symlink_dir(
                os.path.join(persist_root, "models", sub),
                os.path.join(comfy_root, "models", sub),
            )
        _safe_symlink_dir(os.path.join(persist_root, "user"), os.path.join(comfy_root, "user"))
        _safe_symlink_dir(os.path.join(persist_root, "output"), os.path.join(comfy_root, "output"))
        _ensure_comfyui_beginner_assets(comfy_root, persist_root)

        # 若云盘中已有任何预置包的全部权重，但缺少中文工作流文件或工作流版本尚未升级至 v2 (修复 widgets_values_named 兼容)，自动重新生成
        wf_dir_chk = os.path.join(persist_root, "user", "default", "workflows")
        schema_v2_ok = os.path.isfile(os.path.join(wf_dir_chk, ".easycolab_wf_schema_v2"))
        for pk in COMFY_PRESET_WORKFLOW_PACKS:
            if all(
                os.path.isfile(os.path.join(persist_root, "models", sub, fn))
                for sub, fn, _, _, _ in pk["models"]
            ):
                if (not schema_v2_ok) or (not all(os.path.isfile(os.path.join(wf_dir_chk, w)) for w in pk["workflows"])):
                    _build_and_install_comfy_preset_workflows(persist_root, [pk["id"]])

        # 4. 恢复已持久化的自定义节点至本地高速 NVMe，并预装 ComfyUI-Manager 与 VideoHelperSuite 视频核心节点
        COMFYUI_STATE["progress"] = 0.58
        COMFYUI_STATE["status_message"] = "正在恢复已持久化的自定义节点并配置 ComfyUI-Manager..."
        cn_local = os.path.join(comfy_root, "custom_nodes")
        cn_persist = os.path.join(persist_root, "custom_nodes")
        os.makedirs(cn_local, exist_ok=True)
        os.makedirs(cn_persist, exist_ok=True)

        for entry in sorted(os.listdir(cn_persist)):
            if entry.startswith(".") or entry in ("__pycache__", "easycolab_preset_helper"):
                continue
            src_p = os.path.join(cn_persist, entry)
            dst_p = os.path.join(cn_local, entry)
            if os.path.isdir(src_p) and not os.path.exists(dst_p):
                if shutil.which("rsync"):
                    subprocess.run(["rsync", "-a", f"{src_p}/", f"{dst_p}/"], check=False)
                else:
                    shutil.copytree(src_p, dst_p, dirs_exist_ok=True)

        # 若 manifest 中记录了节点但目录缺失，自动按 Git URL 补齐
        manifest_path = os.path.join(persist_root, "custom_nodes_manifest.json")
        if os.path.exists(manifest_path):
            try:
                manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
                for node_name, meta in manifest.items():
                    git_url = (meta or {}).get("git_url", "")
                    dst_p = os.path.join(cn_local, node_name)
                    if git_url and not os.path.exists(dst_p):
                        subprocess.run(["git", "clone", "--depth", "1", git_url, dst_p], check=False)
            except Exception:
                pass

        default_nodes = [
            ("ComfyUI-Manager", "https://github.com/ltdrdata/ComfyUI-Manager.git"),
            ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
        ]
        for node_dir_name, repo_url in default_nodes:
            target_node_dir = os.path.join(cn_local, node_dir_name)
            if not os.path.exists(target_node_dir):
                print(f"[ComfyUI] 正在预装核心插件 {node_dir_name}...")
                subprocess.run(["git", "clone", "--depth", "1", repo_url, target_node_dir], check=False)

        _ensure_easycolab_comfy_preset_helper(comfy_root, persist_root)
        _ensure_comfyui_manager_tunnel_config(comfy_root, persist_root)

        COMFYUI_STATE["progress"] = 0.70
        COMFYUI_STATE["status_message"] = "正在检查并安装已保存自定义节点的 Python 依赖包..."
        _install_persisted_custom_node_requirements(comfy_root)
        sync_comfyui_to_drive_core(quiet=True)

        # 5. 启动 ComfyUI 服务进程 (127.0.0.1:8188) 并实时镜像输出至 EasyColab 日志面板
        COMFYUI_STATE["progress"] = 0.80
        COMFYUI_STATE["status_message"] = "正在拉起 ComfyUI 节点推理引擎 (端口 8188)..."
        port = int(COMFYUI_STATE.get("local_port") or 8188)
        subprocess.run(["pkill", "-f", "ComfyUI/main.py"], check=False)
        comfy_tunnel.kill_tunnels(port + 1)
        subprocess.run(["pkill", "-f", "colab_server.comfy_guard"], check=False)
        time.sleep(0.6)

        comfy_cmd = [
            sys.executable, "-u", "main.py",
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--enable-cors-header", "*",
            "--preview-method", "auto",
        ]
        comfy_proc = subprocess.Popen(
            comfy_cmd,
            cwd=comfy_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        COMFYUI_STATE["comfy_proc"] = comfy_proc

        def _forward_comfy_logs(proc: subprocess.Popen):
            try:
                if proc.stdout is None:
                    return
                for raw_ln in proc.stdout:
                    ln = raw_ln.rstrip()
                    if ln:
                        print(f"[ComfyUI] {ln}")
            except Exception:
                pass

        threading.Thread(target=_forward_comfy_logs, args=(comfy_proc,), daemon=True).start()

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        ready = False
        for _ in range(90):
            if comfy_proc.poll() is not None:
                raise RuntimeError(f"ComfyUI 启动失败提前退出 (code={comfy_proc.returncode})，请查看右侧运行日志。")
            try:
                with opener.open(f"http://127.0.0.1:{port}/system_stats", timeout=1.5) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(1.0)

        if not ready:
            raise RuntimeError("等待 ComfyUI 本地端口 8188 就绪超时，请查看右侧运行日志。")

        # 6. 公网隧道：优先 gradio.live，失败退回 Cloudflare（见 comfy_tunnel）
        COMFYUI_STATE["progress"] = 0.92
        COMFYUI_STATE["status_message"] = "ComfyUI 内核已就绪，正在建立全屏画布公网隧道..."
        guard_port = comfy_guard.ensure_guard(port)  # 隧道指向访问保护，不直接暴露 ComfyUI
        public_url, tunnel_proc = comfy_tunnel.open_tunnel(guard_port)
        COMFYUI_STATE["tunnel_proc"] = tunnel_proc

        COMFYUI_STATE["public_url"] = public_url
        COMFYUI_STATE["is_running"] = True
        COMFYUI_STATE["progress"] = 1.0
        try:
            with open(_COMFY_STATE_FILE, "w", encoding="utf-8") as sf:
                json.dump({"public_url": public_url, "port": port, "guard_port": guard_port}, sf)
        except Exception:
            pass
        _COMFY_STATUS_CACHE["ts"] = 0.0
        (cn_count, m_count, _), _ = _scan_comfyui_counts_and_outputs()
        ok_msg = f"ComfyUI 已就绪（{label} · 节点 {cn_count} 个 · 权重 {m_count} 个）"
        COMFYUI_STATE["status_message"] = ok_msg
        print("[ComfyUI] 画布公网隧道已开通")

        if not state._COMFY_BG_SYNC_STARTED:
            state._COMFY_BG_SYNC_STARTED = True
            threading.Thread(target=_comfyui_periodic_drive_sync_loop, daemon=True).start()

        return True, ok_msg
    except Exception as e:
        err_msg = f"ComfyUI 启动出错: {e}"
        COMFYUI_STATE["status_message"] = err_msg
        COMFYUI_STATE["progress"] = 0.0
        print(f"[ComfyUI] {err_msg}")
        return False, err_msg
    finally:
        COMFYUI_STATE["is_starting"] = False
        touch_activity()


def download_comfyui_model_core(
    url_or_repo: str,
    filename: str = "",
    subfolder: str = "diffusion_models",
    hf_token: str = "",
) -> Tuple[bool, str]:
    """8 线程并发下载模型权重直达持久化的 ComfyUI/models/<subfolder>，下次重开 G4 免重复下载"""
    touch_activity()
    raw_src = (url_or_repo or "").strip()
    if not raw_src:
        return False, "请输入 HuggingFace 仓库/直链或模型下载 URL"
    if COMFYUI_STATE["is_downloading_weight"]:
        return False, "当前已有 ComfyUI 权重正在下载中，请稍候"

    clean_sub = re.sub(r"[^a-zA-Z0-9_.-]", "", (subfolder or "diffusion_models").strip()) or "diffusion_models"
    comfy_root, persist_root, mode, label, _ = _resolve_comfyui_persistence_paths()
    dest_dir = os.path.join(persist_root, "models", clean_sub)
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.isdir(comfy_root):
        _safe_symlink_dir(dest_dir, os.path.join(comfy_root, "models", clean_sub))

    COMFYUI_STATE["is_downloading_weight"] = True
    COMFYUI_STATE["progress"] = 0.05
    try:
        token = (hf_token or "").strip() or None
        hf_m = re.match(r"^https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/[^/]+/(.+?)(\?.*)?$", raw_src)
        if hf_m:
            repo_id = hf_m.group(1)
            rel_file = hf_m.group(2)
            base_name = os.path.basename(filename.strip() or rel_file)
            COMFYUI_STATE["status_message"] = f"正在 8 线程极速下载 {base_name} 至 models/{clean_sub}..."
            print(f"[ComfyUI 极速下载] {repo_id} -> {rel_file} -> models/{clean_sub}/{base_name}")
            dl_path = fast_parallel_hf_download(repo_id=repo_id, filename=rel_file, dest_dir=dest_dir, hf_token=token)
            final_path = os.path.join(dest_dir, base_name)
            if os.path.abspath(dl_path) != os.path.abspath(final_path):
                shutil.move(dl_path, final_path)
        elif raw_src.startswith("http://") or raw_src.startswith("https://"):
            url_path = urllib.parse.urlparse(raw_src).path
            base_name = os.path.basename(filename.strip() or url_path) or f"model_{int(time.time())}.safetensors"
            final_path = os.path.join(dest_dir, base_name)
            COMFYUI_STATE["status_message"] = f"正在下载 {base_name} 至 models/{clean_sub}..."
            print(f"[ComfyUI 直链下载] {raw_src} -> models/{clean_sub}/{base_name}")
            headers = {"User-Agent": "EasyColab/2.0"}
            if token and "huggingface.co" in raw_src:
                headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(raw_src, headers=headers)
            tmp_p = final_path + ".part"
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_p, "wb") as wf:
                total_sz = int(resp.headers.get("Content-Length") or 0)
                done_sz = 0
                t0 = time.time()
                last_log = 0.0
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    wf.write(chunk)
                    done_sz += len(chunk)
                    now_t = time.time()
                    if total_sz > 0 and (now_t - last_log) >= 0.6:
                        last_log = now_t
                        pct = done_sz / float(total_sz)
                        COMFYUI_STATE["progress"] = round(min(0.98, pct), 3)
                        spd = (done_sz / (1024 ** 2)) / max(0.01, now_t - t0)
                        print(f"{base_name}: {int(pct * 100)}%| {done_sz / (1024**3):.2f}G/{total_sz / (1024**3):.2f}G [{spd:.1f}MB/s]")
            os.replace(tmp_p, final_path)
        else:
            if not filename.strip():
                return False, "填写 HuggingFace 仓库 ID (如 Comfy-Org/Wan_2.1_ComfyUI_repackaged) 时请同时填写文件路径"
            rel_file = filename.strip()
            base_name = os.path.basename(rel_file)
            COMFYUI_STATE["status_message"] = f"正在 8 线程极速下载 {base_name} 至 models/{clean_sub}..."
            dl_path = fast_parallel_hf_download(repo_id=raw_src, filename=rel_file, dest_dir=dest_dir, hf_token=token)
            final_path = os.path.join(dest_dir, base_name)
            if os.path.abspath(dl_path) != os.path.abspath(final_path):
                shutil.move(dl_path, final_path)

        if mode == "rclone_drive_sync":
            sync_comfyui_to_drive_core(quiet=True)
        _COMFY_STATUS_CACHE["ts"] = 0.0
        sz_gb = os.path.getsize(final_path) / (1024 ** 3)
        msg = f"权重 {base_name} ({sz_gb:.2f} GB) 已下载至 models/{clean_sub} 并持久化到 {label}（在画布按 R 键即可刷新选用）"
        COMFYUI_STATE["progress"] = 1.0
        COMFYUI_STATE["status_message"] = msg
        print(f"[ComfyUI] {msg}")
        return True, msg
    except Exception as e:
        err = f"下载 ComfyUI 权重失败: {e}"
        COMFYUI_STATE["status_message"] = err
        print(f"[ComfyUI] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_downloading_weight"] = False
        touch_activity()
