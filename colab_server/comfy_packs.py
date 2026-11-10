import os
import subprocess
import threading
from typing import Tuple
import shutil
import urllib.parse
import urllib.request
from .config import _has_rclone_drive_remote, COMFYUI_STATE
from .hf_download import fast_parallel_hf_download
from .idle import touch_activity
from .comfy_presets import (
    _COMFY_STATUS_CACHE,
    COMFY_PRESET_WORKFLOW_PACKS,
    _ensure_comfyui_beginner_assets,
)
from .comfy_workflows import _build_and_install_comfy_preset_workflows
from .comfy_runtime import _resolve_comfyui_persistence_paths, _safe_symlink_dir
from .comfy_sync import sync_comfyui_to_drive_core
from .comfy_start import start_comfyui_core
from .comfy_templates import template_models


_PACK_LOCK = threading.Lock()


def _pack_result(pack_id: str, ok: bool, message: str):
    """给本机工作台看的结果：它按 id 等这一条出现，再决定继续还是告警"""
    COMFYUI_STATE["last_pack_result"] = {"id": pack_id, "ok": ok, "message": message}


def claim_pack(pack_id: str) -> bool:
    """占住下载位并清掉上一次的结果。路由在起线程前调用，工作台轮询时立刻能看到 installing_pack_id"""
    with _PACK_LOCK:
        if COMFYUI_STATE["is_downloading_weight"] or COMFYUI_STATE.get("installing_pack_id"):
            return False
        COMFYUI_STATE["installing_pack_id"] = pack_id
        COMFYUI_STATE.pop("last_pack_result", None)
        return True


def _fail_early(pack_id: str, msg: str) -> Tuple[bool, str]:
    COMFYUI_STATE["installing_pack_id"] = ""
    _pack_result(pack_id, False, msg)
    return False, msg


def install_comfyui_preset_pack_core(
    pack_id: str,
    auto_start: bool = True,
    hf_token: str = "",
    claimed: bool = False,
) -> Tuple[bool, str]:
    """一键配齐预置中文工作流包：8 线程下载缺失模型权重 + 写入中文傻瓜注释工作流 + 生成默认参考图 + 自动拉起/刷新画布"""
    touch_activity()
    if not claimed and not claim_pack(pack_id):
        return False, "当前已有模型或工作流包正在高速下载配齐中，请稍候..."
    if pack_id.startswith("tpl:"):
        # 官方模板：pack_id 形如 tpl:<模板id>,<模板id>，只下载模板声明的模型，不写我们自己的工作流
        ids = [x.strip() for x in pack_id[4:].split(",") if x.strip()]
        COMFYUI_STATE["status_message"] = "正在读取官方模板..."
        try:
            files = template_models(ids)
        except Exception as e:  # noqa: BLE001
            return _fail_early(pack_id, f"读取官方模板失败: {e}")
        target_pack = {"title": "官方模板模型", "official": True, "workflows": [],
                       "models": [(sub, name, repo, path, 0.0) for sub, name, repo, path in files]}
    else:
        target_pack = next((p for p in COMFY_PRESET_WORKFLOW_PACKS if p["id"] == pack_id), None)
    if not target_pack:
        return _fail_early(pack_id, f"未找到预置工作流包: {pack_id}")

    comfy_root, persist_root, mode, label, _ = _resolve_comfyui_persistence_paths()
    models_list = target_pack["models"]
    total_m = len(models_list)

    COMFYUI_STATE["is_downloading_weight"] = True
    COMFYUI_STATE["installing_pack_id"] = pack_id
    COMFYUI_STATE["progress"] = 0.03
    COMFYUI_STATE["status_message"] = f"正在准备配齐「{target_pack['title']}」（共 {total_m} 个官方权重）..."

    try:
        token = (hf_token or "").strip() or ""
        # 若为 rclone 模式且尚未拉取过云盘历史权重，先快速检查恢复，避免重复下载已持久化的权重
        pull_flag = "/tmp/.easycolab_comfy_drive_pulled"
        if mode == "rclone_drive_sync" and _has_rclone_drive_remote() and not os.path.exists(pull_flag):
            COMFYUI_STATE["status_message"] = "正在先从 Google Drive 检查已持久化的历史模型权重..."
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

        _ensure_comfyui_beginner_assets(comfy_root, persist_root)

        for idx, (sub, fname, repo_id, repo_rel_path, approx_gb) in enumerate(models_list, start=1):
            touch_activity()
            dest_dir = os.path.join(persist_root, "models", sub)
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.isdir(comfy_root):
                _safe_symlink_dir(dest_dir, os.path.join(comfy_root, "models", sub))

            final_path = os.path.join(dest_dir, fname)
            base_prog = 0.05 + 0.80 * ((idx - 1) / float(max(1, total_m)))
            COMFYUI_STATE["progress"] = round(base_prog, 3)

            if os.path.isfile(final_path) and os.path.getsize(final_path) > 1024 * 1024:
                print(f"[ComfyUI 预置包] ({idx}/{total_m}) 已存在跳过: models/{sub}/{fname}")
                continue

            size_txt = f" (约 {approx_gb:.1f} GB)" if approx_gb else ""
            COMFYUI_STATE["status_message"] = f"[{idx}/{total_m}] 正在 8 线程极速下载 {fname}{size_txt}..."
            print(f"[ComfyUI 预置包] ({idx}/{total_m}) 开始 8 线程下载 {repo_id}/{repo_rel_path} -> models/{sub}/{fname}")
            dl_path = fast_parallel_hf_download(
                repo_id=repo_id,
                filename=repo_rel_path,
                dest_dir=dest_dir,
                hf_token=token,
                num_connections=8,
            )
            if os.path.abspath(dl_path) != os.path.abspath(final_path):
                shutil.move(dl_path, final_path)
                # 清理 fast_parallel_hf_download 因嵌套路径创建的空子目录 (如 split_files/*)
                top_nested = repo_rel_path.split("/", 1)[0]
                nested_dir = os.path.join(dest_dir, top_nested)
                real_dest, real_nested = os.path.realpath(dest_dir), os.path.realpath(nested_dir)
                inside = real_nested.startswith(real_dest + os.sep) and real_nested != real_dest
                if top_nested and inside and os.path.isdir(nested_dir):
                    try:
                        shutil.rmtree(nested_dir, ignore_errors=True)
                    except Exception:
                        pass

        COMFYUI_STATE["progress"] = 0.88
        if target_pack.get("official"):
            installed_wfs = []
            _pack_result(pack_id, True, f"已下载 {total_m} 个模型文件")  # 云盘同步在后面慢慢做，不挡就绪
        else:
            COMFYUI_STATE["status_message"] = f"正在生成「{target_pack['title']}」全中文三步傻瓜注释工作流..."
            installed_wfs = _build_and_install_comfy_preset_workflows(persist_root, [pack_id])
        if os.path.isdir(comfy_root):
            _safe_symlink_dir(os.path.join(persist_root, "user"), os.path.join(comfy_root, "user"))
            _safe_symlink_dir(os.path.join(persist_root, "output"), os.path.join(comfy_root, "output"))
            _ensure_comfyui_beginner_assets(comfy_root, persist_root)

        if mode == "rclone_drive_sync":
            COMFYUI_STATE["progress"] = 0.93
            COMFYUI_STATE["status_message"] = "正在将配齐的模型与中文工作流持久化同步至 Google Drive..."
            sync_comfyui_to_drive_core(quiet=True)

        _COMFY_STATUS_CACHE["ts"] = 0.0
        wf_names_str = "、".join(installed_wfs) if installed_wfs else "、".join(target_pack["workflows"])

        # 若 ComfyUI 正在运行，主动轻触 /object_info 刷新模型下拉缓存；若尚未启动且开启了 auto_start，直接自动拉起画布
        if COMFYUI_STATE.get("is_running"):
            try:
                port = int(COMFYUI_STATE.get("local_port") or 8188)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f"http://127.0.0.1:{port}/object_info/UNETLoader", timeout=3.0) as _:
                    pass
            except Exception:
                pass
            COMFYUI_STATE["progress"] = 1.0
            ok_msg = f"「{target_pack['title']}」已全部配齐！打开画布左侧「工作流」点击 {wf_names_str} 即可直接传图运行"
            COMFYUI_STATE["status_message"] = ok_msg
            print(f"[ComfyUI 预置包] {ok_msg}")
            _pack_result(pack_id, True, ok_msg)
            return True, ok_msg
    except Exception as e:
        err = f"配齐预置工作流包失败: {e}"
        _pack_result(pack_id, False, err)
        COMFYUI_STATE["progress"] = 0.0
        COMFYUI_STATE["status_message"] = err
        print(f"[ComfyUI 预置包] {err}")
        return False, err
    finally:
        COMFYUI_STATE["is_downloading_weight"] = False
        COMFYUI_STATE["installing_pack_id"] = ""
        touch_activity()

    # 若尚未拉起 ComfyUI，在释放下载锁后自动拉起云端画布隧道
    if auto_start and not COMFYUI_STATE.get("is_running") and not COMFYUI_STATE.get("is_starting"):
        print("[ComfyUI 预置包] 权重与工作流已就绪，正在自动拉起 ComfyUI 全屏画布...")
        ok_s, msg_s = start_comfyui_core(unload_existing=False, force_restart=False)
        if ok_s:
            final_msg = f"「{target_pack['title']}」已配齐且画布已启动！在左侧「工作流」打开中文模板即可直接出片"
            COMFYUI_STATE["status_message"] = final_msg
            return True, final_msg
        return False, msg_s

    return True, f"「{target_pack['title']}」模型与中文工作流已配齐！"


def disconnect_colab_runtime() -> str:
    try:
        if os.path.isdir("/content/ComfyUI") or COMFYUI_STATE.get("is_running"):
            sync_comfyui_to_drive_core(quiet=True)
    except Exception:
        pass
    try:
        from google.colab import runtime
        runtime.unassign()
        return "🛑 已向 Colab 发送释放运行时指令，实例即将关闭并停止扣除计算单元。"
    except ImportError:
        return "ℹ️ 当前不在 Google Colab 环境中，无需释放云端运行时。"
    except Exception as e:
        return f"⚠️ 释放运行时失败: {e}"
