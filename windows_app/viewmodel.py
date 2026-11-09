"""
Core ViewModel (EasyColabViewModel) for the EasyColab Windows Client.
Handles settings persistence, asynchronous HTTP calls to Colab (/minimax_api/*),
headless colab_bridge.py subprocess orchestration (with CREATE_NO_WINDOW on Windows),
2.5s telemetry & keepalive watchdog loop, job polling, and local media management.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from PySide6.QtCore import QObject, QTimer, Signal

from windows_app.models import (
    IMAGE_RESOLUTIONS,
    MODEL_SPEC_MAP,
    OPEN_MODEL_CATALOG,
    PRESET_TEMPLATES,
    VIDEO_RESOLUTIONS,
    ChatTurnItem,
    ComfyOutputMediaItem,
    ComfyPresetPackItem,
    LoadedModelSummary,
    LocalMediaItem,
    ModelFileItem,
    ModelModality,
    OpenModelSpec,
    ResolutionOption,
    build_default_comfy_preset_packs,
    format_duration,
    get_app_config_dir,
    get_lang,
    get_legacy_media_dirs,
    get_local_save_dir,
    is_en,
    localize_server_text,
    mask_email_display,
    mask_url_display,
    open_path_in_os,
    resolve_resource_path,
    set_lang,
    snap_video_frames,
    strip_emoji,
    tr,
)

WIN_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class EasyColabViewModel(QObject):
    """Central state & async controller emitting Qt signals onto the GUI thread."""

    # Signals emitted exclusively on the Qt main thread via _dispatch_ui
    _ui_callback_sig = Signal(object)

    state_changed = Signal()
    telemetry_changed = Signal()
    console_changed = Signal()
    chat_changed = Signal()
    media_list_changed = Signal()
    media_selected = Signal(object, object, bool)  # (Path | None, elapsed: float | None, auto_play: bool)
    theme_changed = Signal()
    language_changed = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._ui_callback_sig.connect(self._execute_ui_callback)

        self._config_path = get_app_config_dir() / "settings.json"
        self._settings = self._load_settings_file()

        # ── Language Setting (Default MUST be Simplified Chinese "zh") ──
        self.language: str = "en" if str(self._settings.get("language", "zh")) == "en" else "zh"
        set_lang(self.language)

        # ── Persisted Settings (Windows defaults to Light Mode) ──
        self.dark_mode: bool = bool(self._settings.get("darkMode", False))
        self.mask_server_url: bool = bool(self._settings.get("maskServerURL", False))
        self.colab_server_url: str = str(self._settings.get("colabServerURL", ""))
        self.wake_gpu_choice: str = str(self._settings.get("wakeGPUChoiceV2", "G4"))
        if self.wake_gpu_choice not in {"G4", "H100", "A100", "L4", "T4"}:
            self.wake_gpu_choice = "G4"
        self.total_budget_cu: float = float(self._settings.get("totalBudgetCU", 2000.0))
        self.idle_timeout_minutes: int = int(self._settings.get("idleTimeoutMinutes", 30))
        if self.idle_timeout_minutes not in {0, 15, 30, 60}:
            self.idle_timeout_minutes = 30
        self.context_length_k: int = max(4, min(256, int(self._settings.get("qwenContextLengthK", 32))))
        self.reasoning_effort: str = str(self._settings.get("reasoningEffort", "medium"))
        if self.reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            self.reasoning_effort = "medium"
        self.api_key_enabled: bool = bool(self._settings.get("apiKeyEnabled", True))
        self.api_access_key: str = str(self._settings.get("apiAccessKey", "")).strip()
        if not self.api_access_key:
            self.api_access_key = "sk-colab-" + secrets.token_hex(10)
            self._save_setting("apiAccessKey", self.api_access_key)
        self.media_elapsed_map: Dict[str, float] = {
            str(k): float(v)
            for k, v in (self._settings.get("mediaElapsedSecondsMap") or {}).items()
            if isinstance(v, (int, float))
        }
        self.has_seen_onboarding: bool = bool(self._settings.get("hasSeenOnboardingGuide_v1", False))
        self.combined_chart_mode: bool = bool(self._settings.get("combinedTelemetryChartMode", False))
        self.hf_token: str = str(self._settings.get("hfToken", ""))

        # ── Account & Headless Colab Bridge State ──
        self.cli_logged_in: bool = False
        self.account_email: str = ""
        self.official_cu_balance: Optional[float] = None
        self.official_burn_rate: Optional[float] = None
        self.active_vm_count: int = 0
        self.show_oauth_code_input: bool = False
        self.is_waking_colab: bool = False
        self.is_releasing_colab: bool = False
        self._auth_busy: bool = False
        # 多账号切换：bridge store 里已保存的账号与当前账号用量摘要
        self.saved_accounts: List[Dict] = []
        self.usage_note: str = ""
        self.active_gpu_summary: str = ""

        # ── Remote Runtime & Telemetry State ──
        self.is_connected: bool = False
        self.gpu_name: str = tr("尚未连接云端 GPU", "Cloud GPU Not Connected")
        self.vram_used_gb: float = 0.0
        self.vram_total_gb: float = 0.0
        self.vram_free_gb: float = 0.0
        self.ram_used_gb: float = 0.0
        self.ram_total_gb: float = 0.0
        self.ram_free_gb: float = 0.0
        self.disk_used_gb: float = 0.0
        self.disk_total_gb: float = 0.0
        self.disk_free_gb: float = 0.0
        self.shm_total_gb: float = 0.0
        self.shm_free_gb: float = 0.0
        self.drive_total_gb: float = 0.0
        self.drive_free_gb: float = 0.0
        self.drive_persisted: bool = False

        self.vram_history_pct: List[float] = []
        self.ram_history_pct: List[float] = []
        self.disk_history_pct: List[float] = []
        self._last_telemetry_sample_ts: float = 0.0

        self.uptime_seconds: int = 0
        self.cu_burn_rate_per_hour: float = 0.0
        self.session_cu_used: float = 0.0
        self.idle_remaining_seconds: int = 1800
        self.api_request_count: int = 0

        self._consecutive_status_failures: int = 0
        self._status_check_inflight: bool = False
        self._remote_context_synced: bool = False
        self._remote_idle_synced: bool = False
        self._remote_api_key_synced: bool = False

        # ── Model Selection, Weights & VRAM State ──
        self.selected_model_id: str = "minimax_h3"
        self.task_mode_id: str = "t2va"
        self.selected_precision: str = "bf16"
        self.selected_offload: str = "disk"

        self.model_files: List[ModelFileItem] = [f.clone() for f in MODEL_SPEC_MAP["minimax_h3"].default_files]
        self.downloaded_count: int = 0
        self.total_count: int = len(self.model_files)
        self.all_downloaded: bool = False
        self.total_size_gb: float = 0.0
        self.is_downloading_weights: bool = False
        self.is_loading_model: bool = False
        self.model_task_progress: float = 0.0

        self.loaded_model_ids: List[str] = []
        self.loaded_models_detail: List[LoadedModelSummary] = []
        self.loaded_context_length: int = 32768
        self.current_model_name: str = ""

        # ── Generation & Sampling State ──
        self.visual_prompt: str = PRESET_TEMPLATES[0].visual
        self.soundscape_prompt: str = PRESET_TEMPLATES[0].soundscape
        self.music_prompt: str = PRESET_TEMPLATES[0].music
        self.negative_prompt: str = ""

        self.selected_video_res: ResolutionOption = VIDEO_RESOLUTIONS[0]
        self.selected_image_res: ResolutionOption = IMAGE_RESOLUTIONS[0]
        self.num_frames: int = 124
        self.inference_steps: int = 35
        self.seed: int = 42
        self.randomize_seed: bool = True
        self.batch_count: int = 1
        self.denoising_strength: float = 0.70

        self.enable_thinking: bool = True
        self.max_new_tokens: int = 16384
        self.temperature: float = 1.0
        self.top_p: float = 0.95
        self.presence_penalty: float = 0.0

        self.is_generating: bool = False
        self.progress_value: float = 0.0
        self.current_job_elapsed: float = 0.0
        self._job_start_local_ts: float = 0.0
        self._active_generation_token: str = ""

        # ── ComfyUI State ──
        self.comfy_running: bool = False
        self.comfy_starting: bool = False
        self.comfy_syncing_drive: bool = False
        self.comfy_downloading_weight: bool = False
        self.comfy_installing_pack_id: str = ""
        self.comfy_progress: float = 0.0
        self.comfy_status_message: str = tr("云端 ComfyUI 尚未启动", "Cloud ComfyUI not started")
        self.comfy_public_url: str = ""
        self.comfy_queue_running: int = 0
        self.comfy_queue_pending: int = 0
        self.comfy_persistence_label: str = tr("Google Drive 自动软链持久化", "Google Drive Auto-Symlink Persistence")
        self.comfy_custom_nodes_count: int = 0
        self.comfy_models_count: int = 0
        self.comfy_workflows_count: int = 0
        self.comfy_unload_others_on_start: bool = False
        self.comfy_recent_outputs: List[ComfyOutputMediaItem] = []
        self.comfy_preset_packs: List[ComfyPresetPackItem] = build_default_comfy_preset_packs()

        # ── Chat, Logs & Local Gallery ──
        self.chat_history: List[ChatTurnItem] = []
        self.status_message: str = tr(
            "就绪：请先绑定 Google 账号并唤醒 G4 实例，或粘贴已有隧道链接连接。",
            "Ready: Bind your Google account and wake a G4 runtime, or paste an existing tunnel URL.",
        )
        self.live_console_lines: List[str] = [
            tr("EasyColab Windows 工作台已就绪。", "EasyColab Windows Studio is ready."),
            tr(
                f"本地成片自动保存目录: {get_local_save_dir()}",
                f"Local media auto-save folder: {get_local_save_dir()}",
            ),
        ]
        self.local_media_items: List[LocalMediaItem] = []
        self.current_media_path: Optional[Path] = None

        # ── Timers ──
        self._tick_counter: int = 0
        self._bg_timer = QTimer(self)
        self._bg_timer.setInterval(2500)
        self._bg_timer.timeout.connect(self._on_background_tick)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(200)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)

    # ─── Initialization & Lifecycle ──────────────────────────────────────────

    def start(self) -> None:
        self.refresh_local_media_gallery(auto_select_first=True)
        self.refresh_official_colab_account_usage()
        self.load_saved_accounts()
        if self.normalized_base_url:
            self.check_colab_status(silent=True)
        self._bg_timer.start()

    def _dispatch_ui(self, fn: Callable[[], None]) -> None:
        self._ui_callback_sig.emit(fn)

    def _execute_ui_callback(self, fn: object) -> None:
        if callable(fn):
            fn()

    # ─── Settings Helpers ────────────────────────────────────────────────────

    def _load_settings_file(self) -> dict:
        if self._config_path.exists():
            try:
                return json.loads(self._config_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_setting(self, key: str, value: Any) -> None:
        self._settings[key] = value
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                json.dumps(self._settings, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def toggle_dark_mode(self) -> None:
        self.dark_mode = not self.dark_mode
        self._save_setting("darkMode", self.dark_mode)
        self.theme_changed.emit()

    def set_language(self, lang: str) -> None:
        target = "en" if str(lang).strip().lower().startswith("en") else "zh"
        if target == self.language and get_lang() == target:
            return
        self.language = target
        set_lang(self.language)
        self._save_setting("language", self.language)
        self._translate_default_runtime_strings()
        self.language_changed.emit()

    def toggle_language(self) -> None:
        self.set_language("zh" if self.language == "en" else "en")

    def _translate_default_runtime_strings(self) -> None:
        to_en = is_en()

        def swap(val: str, pairs: List[tuple[str, str]]) -> str:
            for zh_txt, en_txt in pairs:
                if to_en and val == zh_txt:
                    return en_txt
                if (not to_en) and val == en_txt:
                    return zh_txt
            return localize_server_text(val) if to_en else val

        self.gpu_name = swap(
            self.gpu_name,
            [("尚未连接云端 GPU", "Cloud GPU Not Connected")],
        )
        self.comfy_status_message = swap(
            self.comfy_status_message,
            [
                ("云端 ComfyUI 尚未启动", "Cloud ComfyUI not started"),
                (
                    "尚未启动（支持一键拉起 ComfyUI + Google Drive 自动软链持久化）",
                    "Not started (1-click ComfyUI + Google Drive symlink persistence)",
                ),
            ],
        )
        self.comfy_persistence_label = swap(
            self.comfy_persistence_label,
            [
                ("Google Drive 自动软链持久化", "Google Drive Auto-Symlink Persistence"),
                ("Google Drive 实时软链挂载", "Google Drive Live Symlink Mounted"),
                ("Google Drive 云盘自动持久化", "Google Drive Cloud Auto-Persistence"),
                ("实例本地高速存储", "Instance Local High-Speed Storage"),
            ],
        )
        self.status_message = swap(
            self.status_message,
            [
                (
                    "就绪：请先绑定 Google 账号并唤醒 G4 实例，或粘贴已有隧道链接连接。",
                    "Ready: Bind your Google account and wake a G4 runtime, or paste an existing tunnel URL.",
                ),
                (
                    "请先输入 https://xxxx.gradio.live 服务地址，或点击「唤醒实例并部署」。",
                    "Please enter a https://xxxx.gradio.live server URL or click 'Wake & Deploy'.",
                ),
                ("请先连接或唤醒 Colab 云端实例。", "Please connect or wake a Colab cloud runtime first."),
                (
                    "尚未连接云端 Colab 服务，请先唤醒或连接实例。",
                    "Not connected to Colab server. Please wake or connect a runtime first.",
                ),
                ("已退出当前 Google 账号。", "Signed out of Google account."),
                ("已释放全部云端实例。", "All cloud runtimes released."),
                ("云端实例已释放，本地状态已同步归零。", "Cloud runtime released; local state reset."),
                ("已生成新的随机 API Key。", "Generated a new random API Key."),
                ("请先输入提示词或对话内容。", "Please enter a prompt or message first."),
            ],
        )

        save_dir = get_local_save_dir()
        line_pairs = [
            ("EasyColab Windows 工作台已就绪。", "EasyColab Windows Studio is ready."),
            (f"本地成片自动保存目录: {save_dir}", f"Local media auto-save folder: {save_dir}"),
            ("已清空历史输出，正在监听云端实时日志...", "Console cleared. Listening for live cloud logs..."),
        ]
        self.live_console_lines = [swap(ln, line_pairs) for ln in self.live_console_lines]

        for d in self.loaded_models_detail:
            if d.id in MODEL_SPEC_MAP:
                d.name = MODEL_SPEC_MAP[d.id].name

        for tpl in PRESET_TEMPLATES:
            if tpl.visual_en and tpl.visual_en != tpl.visual_zh:
                if to_en and self.visual_prompt == tpl.visual_zh:
                    self.visual_prompt = tpl.visual_en
                    break
                if (not to_en) and self.visual_prompt == tpl.visual_en:
                    self.visual_prompt = tpl.visual_zh
                    break

    def set_mask_server_url(self, enabled: bool) -> None:
        self.mask_server_url = bool(enabled)
        self._save_setting("maskServerURL", self.mask_server_url)
        self.state_changed.emit()
        self.console_changed.emit()

    def set_colab_server_url(self, url: str) -> None:
        cleaned = (url or "").strip()
        if cleaned != self.colab_server_url:
            self.colab_server_url = cleaned
            self._save_setting("colabServerURL", cleaned)
            self.state_changed.emit()

    def set_wake_gpu_choice(self, gpu: str) -> None:
        if gpu in {"G4", "H100", "A100", "L4", "T4"}:
            self.wake_gpu_choice = gpu
            self._save_setting("wakeGPUChoiceV2", gpu)
            self.state_changed.emit()

    def set_combined_chart_mode(self, combined: bool) -> None:
        self.combined_chart_mode = bool(combined)
        self._save_setting("combinedTelemetryChartMode", self.combined_chart_mode)
        self.telemetry_changed.emit()

    def mark_onboarding_seen(self) -> None:
        if not self.has_seen_onboarding:
            self.has_seen_onboarding = True
            self._save_setting("hasSeenOnboardingGuide_v1", True)

    def set_enable_thinking(self, enabled: bool) -> None:
        self.enable_thinking = bool(enabled)
        if self.enable_thinking:
            self.temperature = 1.0
            self.top_p = 0.95
            self.presence_penalty = 0.0
        else:
            self.temperature = 0.7
            self.top_p = 0.80
            self.presence_penalty = 1.5
        self.state_changed.emit()

    def set_reasoning_effort(self, effort: str) -> None:
        if effort in {"low", "medium", "high", "xhigh"}:
            self.reasoning_effort = effort
            self._save_setting("reasoningEffort", effort)
            self.state_changed.emit()

    def set_context_length_k(self, k_val: int, sync_remote: bool = True) -> None:
        clamped = max(4, min(256, int(k_val)))
        self.context_length_k = clamped
        self._save_setting("qwenContextLengthK", clamped)
        self.state_changed.emit()
        if sync_remote and self.is_connected:
            self.sync_context_length_to_colab()

    # ─── Computed Properties ─────────────────────────────────────────────────

    @property
    def normalized_base_url(self) -> str:
        raw = self.colab_server_url.strip()
        while raw.endswith("/"):
            raw = raw[:-1]
        return raw

    @property
    def active_model_spec(self) -> OpenModelSpec:
        return MODEL_SPEC_MAP.get(self.selected_model_id, OPEN_MODEL_CATALOG[0])

    @property
    def context_length_tokens(self) -> int:
        return self.context_length_k * 1024

    @property
    def is_selected_model_loaded_in_vram(self) -> bool:
        return self.selected_model_id in self.loaded_model_ids

    @property
    def is_qwen_loaded_in_vram(self) -> bool:
        return ("qwen38_27b_gguf" in self.loaded_model_ids) or ("qwen38_27b" in self.loaded_model_ids)

    @property
    def display_account_label(self) -> str:
        if not self.cli_logged_in:
            return tr("未绑定", "Not bound")
        if self.account_email:
            return mask_email_display(self.account_email, self.mask_server_url)
        return tr("已连接", "Connected")

    @property
    def display_remaining_cu(self) -> float:
        if self.official_cu_balance is not None:
            return max(0.0, self.official_cu_balance)
        return max(0.0, self.total_budget_cu - self.session_cu_used)

    @property
    def display_burn_rate(self) -> float:
        if self.official_burn_rate is not None:
            return max(0.0, self.official_burn_rate)
        return max(0.0, self.cu_burn_rate_per_hour)

    def masked_url(self, text: str) -> str:
        return mask_url_display(text, self.mask_server_url)

    def append_console_line(self, line: str) -> None:
        clean = strip_emoji(self.masked_url(line))
        if not clean:
            return
        self.live_console_lines.append(clean)
        if len(self.live_console_lines) > 400:
            self.live_console_lines = self.live_console_lines[-400:]
        self.console_changed.emit()

    def clear_console_lines(self) -> None:
        self.live_console_lines = [
            tr("已清空历史输出，正在监听云端实时日志...", "Console cleared. Listening for live cloud logs...")
        ]
        self.console_changed.emit()

    # ─── Local Media Gallery Management ──────────────────────────────────────

    def refresh_local_media_gallery(self, auto_select_first: bool = False) -> None:
        exts = {".mp4", ".png", ".jpg", ".jpeg", ".webp"}
        seen_names = set()
        found: List[LocalMediaItem] = []

        for folder in get_legacy_media_dirs():
            if not folder.exists():
                continue
            try:
                for entry in folder.iterdir():
                    if not entry.is_file() or entry.suffix.lower() not in exts:
                        continue
                    if entry.name in seen_names:
                        continue
                    seen_names.add(entry.name)
                    try:
                        stat = entry.stat()
                        ctime = stat.st_mtime
                    except Exception:
                        ctime = 0.0
                    found.append(
                        LocalMediaItem(
                            path=entry,
                            filename=entry.name,
                            is_image=(entry.suffix.lower() != ".mp4"),
                            created_ts=ctime,
                            elapsed_seconds=self.media_elapsed_map.get(entry.name),
                        )
                    )
            except Exception:
                pass

        found.sort(key=lambda x: x.created_ts, reverse=True)
        self.local_media_items = found
        self.media_list_changed.emit()

        if auto_select_first and self.current_media_path is None and found:
            self.select_local_media(found[0].path, auto_play=False)

    def select_local_media(self, path: Optional[Path], auto_play: bool = True) -> None:
        self.current_media_path = path
        elapsed = self.media_elapsed_map.get(path.name) if path else None
        self.media_selected.emit(path, elapsed, auto_play)

    def export_or_reveal_notebook(self) -> None:
        nb_src = resolve_resource_path("EasyColab_Server.ipynb")
        if not nb_src.exists():
            self.status_message = tr(
                "未找到内置 EasyColab_Server.ipynb 文件。",
                "Built-in EasyColab_Server.ipynb file not found.",
            )
            self.state_changed.emit()
            return
        # If running inside a frozen bundle (_internal), copy to Downloads so user can easily drag into browser
        if getattr(sys, "frozen", False):
            dl_dir = Path.home() / "Downloads"
            dl_dir.mkdir(parents=True, exist_ok=True)
            dst = dl_dir / "EasyColab_Server.ipynb"
            try:
                shutil.copy2(nb_src, dst)
                nb_src = dst
            except Exception:
                pass
        self.status_message = tr(f"已定位笔记本文件：{nb_src}", f"Located notebook file: {nb_src}")
        self.state_changed.emit()
        open_path_in_os(nb_src, reveal=True)

    # ─── Headless colab_bridge.py Execution ──────────────────────────────────

    def _resolve_python_for_bridge(self, install_if_missing: bool = False) -> List[str]:
        """
        Return base command prefix to execute colab_bridge CLI:
        - Portable Windows bundle: `[_internal/runtime/python.exe, "-X", "utf8", colab_bridge.py]`
        - PyInstaller Windows .exe: `[EasyColab.exe, "--bridge-cli"]`
        - Dev Python source: `[python, colab_bridge.py]` (with automatic managed venv fallback if needed).
        """
        bridge_py = resolve_resource_path("colab_bridge.py")

        # 1. Direct filesystem check for bundled embeddable runtime (works regardless of sys.frozen)
        exe_dir = Path(sys.executable).resolve().parent
        pkg_parent = Path(__file__).resolve().parent.parent
        env_internal = os.environ.get("EASYCOLAB_INTERNAL_DIR", "").strip()
        embedded_candidates = [
            pkg_parent / "runtime" / "python.exe",
            exe_dir / "_internal" / "runtime" / "python.exe",
            exe_dir / "runtime" / "python.exe",
        ]
        if env_internal:
            embedded_candidates.insert(0, Path(env_internal) / "runtime" / "python.exe")

        for emb_py in embedded_candidates:
            if emb_py.exists() and bridge_py.exists():
                return [str(emb_py), "-X", "utf8", str(bridge_py)]

        # 2. PyInstaller frozen executable fallback
        if getattr(sys, "frozen", False):
            return [sys.executable, "--bridge-cli"]

        # 3. Dev Python source mode (never treat a non-Python GUI .exe as a Python interpreter)
        candidates: List[Path] = []
        cur_exe = Path(sys.executable)
        if cur_exe.stem.lower().startswith("python"):
            candidates.append(cur_exe)
        repo_root = Path(__file__).resolve().parent.parent
        if sys.platform == "win32":
            candidates.append(repo_root / ".venv" / "Scripts" / "python.exe")
            managed_dir = get_app_config_dir() / "venv"
            managed_py = managed_dir / "Scripts" / "python.exe"
        else:
            candidates.append(repo_root / ".venv" / "bin" / "python")
            managed_dir = get_app_config_dir() / "venv"
            managed_py = managed_dir / "bin" / "python"
        candidates.append(managed_py)

        for py in candidates:
            if py.exists():
                try:
                    chk = subprocess.run(
                        [str(py), "-c", "import colab_cli, google.auth, websocket"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=8,
                        creationflags=WIN_NO_WINDOW,
                    )
                    if chk.returncode == 0:
                        return [str(py), str(bridge_py)]
                except Exception:
                    pass

        if install_if_missing:
            self._dispatch_ui(
                lambda: self._set_status_and_log(
                    tr(
                        "首次使用正在初始化本地 Colab 连接组件 (约需 15~30 秒)...",
                        "First-run setup: initializing local Colab bridge components (~15-30s)...",
                    )
                )
            )
            if not managed_py.exists():
                subprocess.run(
                    [sys.executable, "-m", "venv", str(managed_dir)],
                    check=True,
                    creationflags=WIN_NO_WINDOW,
                )
            subprocess.run(
                [
                    str(managed_py),
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "google-colab-cli>=0.7.5",
                    "google-auth>=2.28.0",
                    "google-auth-oauthlib>=1.2.0",
                    "requests>=2.31.0",
                    "requests-oauthlib>=1.3.1",
                    "websocket-client>=1.7.0",
                ],
                check=True,
                creationflags=WIN_NO_WINDOW,
            )
            return [str(managed_py), str(bridge_py)]

        if not cur_exe.stem.lower().startswith("python"):
            return [sys.executable, "--bridge-cli"]
        return [sys.executable, str(bridge_py)]

    def _set_status_and_log(self, msg: str) -> None:
        self.status_message = strip_emoji(self.masked_url(msg))
        self.append_console_line(msg)
        self.state_changed.emit()

    def _bridge_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _run_bridge_json(self, subargs: List[str], install_if_missing: bool = False, timeout: int = 45) -> dict:
        cmd = self._resolve_python_for_bridge(install_if_missing=install_if_missing) + subargs
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=self._bridge_env(),
            timeout=timeout,
            creationflags=WIN_NO_WINDOW,
        )
        out = (proc.stdout or "").strip()
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        if proc.stderr:
            return {"ok": False, "error": proc.stderr.strip()[-300:]}
        return {"ok": False, "error": out or tr("空响应", "Empty response")}

    # ─── Google OAuth & Headless Wake / Keepalive / Stop ─────────────────────

    def refresh_official_colab_account_usage(self) -> None:
        if self._auth_busy:
            return
        self._auth_busy = True

        def worker() -> None:
            try:
                data = self._run_bridge_json(["auth_status"], install_if_missing=False, timeout=25)
            except Exception as exc:
                data = {"ok": False, "error": str(exc)}

            def apply_ui() -> None:
                self._auth_busy = False
                logged = bool(data.get("logged_in", False))
                self.cli_logged_in = logged
                if logged:
                    if data.get("email"):
                        self.account_email = str(data["email"])
                    if data.get("paid_balance") is not None:
                        bal = float(data["paid_balance"])
                        self.official_cu_balance = bal
                        self.total_budget_cu = bal
                        self._save_setting("totalBudgetCU", bal)
                        self.usage_note = ""
                    else:
                        self.usage_note = tr(
                            "用量信息不可用（免费账号无 CU 计量）",
                            "Usage info unavailable (free accounts have no CU metering)",
                        )
                    if data.get("burn_rate_hourly") is not None:
                        self.official_burn_rate = float(data["burn_rate_hourly"])
                    if data.get("active_assignments") is not None:
                        self.active_vm_count = int(data["active_assignments"])
                        if self.active_vm_count == 0 and not self.is_connected and self.loaded_model_ids:
                            self.reset_stale_remote_runtime_state(
                                tr("云端实例已释放，本地状态已同步归零。", "Cloud runtime released; local state reset.")
                            )
                    # 从会话列表按 accelerator 字符串去重计数，拼出 "GPU:T4 ×1, H100 ×1"
                    gpu_counts: Dict[str, int] = {}
                    for sess in data.get("sessions") or []:
                        if isinstance(sess, dict):
                            acc = str(sess.get("accelerator") or "").strip()
                            if acc:
                                gpu_counts[acc] = gpu_counts.get(acc, 0) + 1
                    self.active_gpu_summary = ", ".join(f"{k} ×{v}" for k, v in gpu_counts.items())
                else:
                    self.usage_note = ""
                    self.active_gpu_summary = ""
                self.state_changed.emit()

            self._dispatch_ui(apply_ui)

        threading.Thread(target=worker, daemon=True).start()

    def start_google_oauth_login(self, switching_account: bool = False) -> None:
        self.status_message = tr("正在生成 Google OAuth 安全授权链接...", "Generating Google OAuth authorization URL...")
        self.state_changed.emit()

        def worker() -> None:
            try:
                data = self._run_bridge_json(["auth_url"], install_if_missing=True, timeout=120)
            except Exception as exc:
                data = {"ok": False, "error": str(exc)}

            def apply_ui() -> None:
                auth_url = str(data.get("auth_url") or "").strip()
                if auth_url:
                    self.show_oauth_code_input = True
                    tip = (
                        tr(
                            "请在弹出的浏览器中选择要切换的 Google 账号，并将授权码粘贴回下方输入框。",
                            "Select the Google account in your browser and paste the authorization code below.",
                        )
                        if switching_account
                        else tr(
                            "已在浏览器打开 Google 授权页，请登录并复制授权码粘贴至下方输入框。",
                            "Opened Google sign-in in browser. Authorize and paste the code below.",
                        )
                    )
                    self.status_message = tip
                    self.append_console_line(tip)
                    webbrowser.open(auth_url)
                else:
                    err = str(data.get("error") or tr("无法生成授权链接", "Failed to generate OAuth URL"))
                    self.status_message = tr(f"获取授权链接失败：{err}", f"Failed to get OAuth URL: {err}")
                    self.append_console_line(self.status_message)
                self.state_changed.emit()

            self._dispatch_ui(apply_ui)

        threading.Thread(target=worker, daemon=True).start()

    def submit_google_oauth_code(self, code: str) -> None:
        trimmed = (code or "").strip()
        if not trimmed:
            self.status_message = tr(
                "请先粘贴浏览器中显示的 Google 授权码。",
                "Please paste the Google authorization code from your browser first.",
            )
            self.state_changed.emit()
            return

        was_logged_in = self.cli_logged_in
        prev_email = self.account_email
        self.status_message = tr(
            "正在验证授权码并同步 Google Colab 算力账户...",
            "Verifying authorization code & syncing Google Colab account...",
        )
        self.state_changed.emit()

        def worker() -> None:
            try:
                data = self._run_bridge_json(["auth_code", trimmed], install_if_missing=True, timeout=45)
            except Exception as exc:
                data = {"ok": False, "error": str(exc)}

            def apply_ui() -> None:
                if data.get("logged_in"):
                    new_email = str(data.get("email") or "")
                    if was_logged_in and (new_email != prev_email or self.colab_server_url):
                        self.set_colab_server_url("")
                        self.reset_stale_remote_runtime_state(
                            tr(
                                "已切换 Google 账号，请点击「唤醒实例并部署」启动新实例。",
                                "Switched Google account. Click 'Wake & Deploy' to launch a new runtime.",
                            )
                        )
                    self.cli_logged_in = True
                    self.account_email = new_email
                    self.show_oauth_code_input = False
                    # 新账号用量未知，先清掉上一账号残留，等用量刷新回填
                    self.usage_note = ""
                    self.active_gpu_summary = ""
                    if data.get("paid_balance") is not None:
                        bal = float(data["paid_balance"])
                        self.official_cu_balance = bal
                        self.total_budget_cu = bal
                        self._save_setting("totalBudgetCU", bal)
                    if data.get("burn_rate_hourly") is not None:
                        self.official_burn_rate = float(data["burn_rate_hourly"])
                    if data.get("active_assignments") is not None:
                        self.active_vm_count = int(data["active_assignments"])
                    self.status_message = tr(
                        f"Google 账号绑定成功 ({self.display_account_label})！现在可一键唤醒 {self.wake_gpu_choice} 实例。",
                        f"Google account bound ({self.display_account_label})! Ready to wake {self.wake_gpu_choice} runtime in 1 click.",
                    )
                    self.append_console_line(self.status_message)
                    self.load_saved_accounts()
                else:
                    err = str(data.get("error") or tr("授权码无效或已过期", "Invalid or expired authorization code"))
                    self.status_message = tr(f"绑定失败：{err}", f"Account binding failed: {err}")
                    self.append_console_line(self.status_message)
                self.state_changed.emit()

            self._dispatch_ui(apply_ui)

        threading.Thread(target=worker, daemon=True).start()

    def logout_google_account(self) -> None:
        def worker() -> None:
            try:
                self._run_bridge_json(["auth_logout"], install_if_missing=False, timeout=20)
            except Exception:
                pass

            def apply_ui() -> None:
                self.cli_logged_in = False
                self.account_email = ""
                self.official_cu_balance = None
                self.official_burn_rate = None
                self.active_vm_count = 0
                self.usage_note = ""
                self.active_gpu_summary = ""
                self.show_oauth_code_input = False
                self.set_colab_server_url("")
                self.reset_stale_remote_runtime_state(tr("已退出当前 Google 账号。", "Signed out of Google account."))
                self.load_saved_accounts()
                self.state_changed.emit()

            self._dispatch_ui(apply_ui)

        threading.Thread(target=worker, daemon=True).start()

    def load_saved_accounts(self) -> None:
        """后台拉取 bridge store 中已保存的多账号列表；失败静默置空。"""

        def worker() -> None:
            try:
                data = self._run_bridge_json(["accounts_list"], install_if_missing=False, timeout=20)
            except Exception:
                data = {"ok": False}

            def apply_ui() -> None:
                active_email = str(data.get("active_email") or "")
                accounts: List[Dict] = []
                for item in data.get("accounts") or []:
                    if isinstance(item, dict) and item.get("email"):
                        email = str(item["email"])
                        # item 未带 active 时退回用顶层 active_email 判断
                        is_active = bool(item.get("active")) or (email == active_email and bool(active_email))
                        accounts.append({"email": email, "active": is_active})
                self.saved_accounts = accounts
                self.state_changed.emit()

            self._dispatch_ui(apply_ui)

        threading.Thread(target=worker, daemon=True).start()

    def switch_saved_account(self, email: str) -> None:
        """切换到 store 中已保存的账号；全程复用 _auth_busy 防重入。"""
        target = (email or "").strip()
        if not target or self._auth_busy:
            return
        self._auth_busy = True
        self.status_message = tr("正在切换 Google 账号...", "Switching Google account...")
        self.state_changed.emit()

        def worker() -> None:
            try:
                data = self._run_bridge_json(["accounts_switch", target], install_if_missing=False, timeout=45)
            except Exception as exc:
                data = {"ok": False, "error": str(exc)}

            def apply_ui() -> None:
                self._auth_busy = False
                if data.get("ok") and data.get("logged_in"):
                    new_email = str(data.get("email") or target)
                    masked = mask_email_display(new_email, self.mask_server_url)
                    self.cli_logged_in = True
                    self.account_email = new_email
                    self.show_oauth_code_input = False
                    # 上一账号的用量与云端残留先清零，等用量刷新回填
                    self.official_cu_balance = None
                    self.official_burn_rate = None
                    self.active_vm_count = 0
                    self.usage_note = ""
                    self.active_gpu_summary = ""
                    self.reset_stale_remote_runtime_state(
                        tr(
                            f"已切换到 {masked}，云端状态已复位。",
                            f"Switched to {masked}; cloud state reset.",
                        )
                    )
                    self.refresh_official_colab_account_usage()
                    self.load_saved_accounts()
                else:
                    err = str(data.get("error") or tr("切换失败", "Account switch failed"))
                    self.status_message = tr(f"切换账号失败：{err}", f"Account switch failed: {err}")
                    self.append_console_line(self.status_message)
                self.state_changed.emit()

            self._dispatch_ui(apply_ui)

        threading.Thread(target=worker, daemon=True).start()

    def wake_colab_headless(self) -> None:
        if self.is_waking_colab:
            return
        if not self.cli_logged_in:
            self.status_message = tr(
                "请先点击「绑定」授权登录 Google 账号。",
                "Please click 'Bind' to sign in with your Google account first.",
            )
            self.state_changed.emit()
            return

        app_py = resolve_resource_path("app.py")
        if not app_py.exists():
            self.status_message = tr(f"未找到服务端脚本 app.py ({app_py})", f"Server script app.py not found ({app_py})")
            self.state_changed.emit()
            return

        gpu = self.wake_gpu_choice
        self.is_waking_colab = True
        self.status_message = tr(
            f"正在唤醒 Google Colab {gpu} 实例并自动部署多模型服务端...",
            f"Waking Google Colab {gpu} runtime & deploying multi-model server...",
        )
        self.append_console_line(
            tr(
                f"─── 开始无网页唤醒 Colab {gpu} 实例 ───",
                f"─── Starting Headless Wake for Colab {gpu} Runtime ───",
            )
        )
        self.state_changed.emit()

        def worker() -> None:
            try:
                cmd = self._resolve_python_for_bridge(install_if_missing=True) + [
                    "wake",
                    "--gpu",
                    gpu,
                    "--app-py",
                    str(app_py),
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=self._bridge_env(),
                    creationflags=WIN_NO_WINDOW,
                )
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("WAKE_URL:"):
                        url = line[len("WAKE_URL:") :].strip()

                        def on_url(u: str = url) -> None:
                            self.set_colab_server_url(u)
                            self.append_console_line(
                                tr(
                                    f"成功获取云端隧道地址: {self.masked_url(u)}",
                                    f"Tunnel URL acquired: {self.masked_url(u)}",
                                )
                            )
                            self.status_message = tr(
                                f"云端 {gpu} 实例已唤醒就绪，正在同步模型状态...",
                                f"Cloud {gpu} runtime is online! Syncing model status...",
                            )
                            self.state_changed.emit()

                        self._dispatch_ui(on_url)
                    elif line.startswith("WAKE_ERROR:"):
                        err_txt = strip_emoji(line[len("WAKE_ERROR:") :].strip())

                        def on_err(e: str = err_txt) -> None:
                            loc_e = localize_server_text(e)
                            msg_e = tr(f"唤醒失败: {loc_e}", f"Wake failed: {loc_e}")
                            self.append_console_line(msg_e)
                            self.status_message = msg_e
                            self.state_changed.emit()

                        self._dispatch_ui(on_err)
                    else:
                        clean = strip_emoji(line.replace("WAKE_LOG:", "").strip())
                        if clean:

                            def on_log(c: str = clean) -> None:
                                loc_c = localize_server_text(c)
                                self.append_console_line(loc_c)
                                self.status_message = loc_c
                                self.state_changed.emit()

                            self._dispatch_ui(on_log)
                proc.wait(timeout=30)
            except Exception as exc:
                err_msg = str(exc)
                self._dispatch_ui(
                    lambda: self._set_status_and_log(tr(f"唤醒进程异常: {err_msg}", f"Wake process error: {err_msg}"))
                )

            def finish_ui() -> None:
                self.is_waking_colab = False
                self.state_changed.emit()
                self.refresh_official_colab_account_usage()
                if self.normalized_base_url:
                    self.check_colab_status(silent=False)

            self._dispatch_ui(finish_ui)

        threading.Thread(target=worker, daemon=True).start()

    def keepalive_colab_runtime(self) -> None:
        def worker() -> None:
            try:
                data = self._run_bridge_json(["keepalive"], install_if_missing=False, timeout=35)
            except Exception:
                return

            def apply_ui() -> None:
                if data.get("paid_balance") is not None:
                    bal = float(data["paid_balance"])
                    self.official_cu_balance = bal
                    self.total_budget_cu = bal
                if data.get("burn_rate_hourly") is not None:
                    self.official_burn_rate = float(data["burn_rate_hourly"])
                if data.get("active_assignments") is not None:
                    self.active_vm_count = int(data["active_assignments"])
                self.state_changed.emit()

            self._dispatch_ui(apply_ui)

        threading.Thread(target=worker, daemon=True).start()

    def disconnect_remote_colab(self) -> None:
        if self.is_releasing_colab:
            return
        self.is_releasing_colab = True
        self._active_generation_token = str(uuid.uuid4())
        self.is_generating = False
        self.status_message = tr(
            "正在同步云盘数据并彻底销毁所有运行中的 Colab GPU 实例...",
            "Syncing Google Drive & terminating all active Colab GPU runtimes...",
        )
        self.append_console_line(
            tr(
                "─── 正在断开并释放全部云端 GPU 实例 ───",
                "─── Disconnecting & Releasing All Cloud GPU Runtimes ───",
            )
        )
        self.state_changed.emit()

        base = self.normalized_base_url

        def worker() -> None:
            if base:
                try:
                    requests.post(f"{base}/minimax_api/disconnect", json={}, timeout=8)
                except Exception:
                    pass
            try:
                data = self._run_bridge_json(["stop_all"], install_if_missing=False, timeout=60)
            except Exception as exc:
                data = {"ok": False, "error": str(exc)}

            def apply_ui() -> None:
                self.is_releasing_colab = False
                self.set_colab_server_url("")
                raw_msg = str(data.get("message") or data.get("error") or tr("已释放全部云端实例。", "All cloud runtimes released."))
                msg = localize_server_text(strip_emoji(raw_msg))
                self.reset_stale_remote_runtime_state(msg)
                if data.get("paid_balance") is not None:
                    self.official_cu_balance = float(data["paid_balance"])
                self.official_burn_rate = float(data.get("burn_rate_hourly") or 0.0)
                self.active_vm_count = int(data.get("active_assignments") or 0)
                self.append_console_line(msg)
                self.state_changed.emit()

            self._dispatch_ui(apply_ui)

        threading.Thread(target=worker, daemon=True).start()

    # ─── Background 2.5s Telemetry & Watchdog Loop ───────────────────────────

    def _on_elapsed_tick(self) -> None:
        if self.is_generating and self._job_start_local_ts > 0:
            local_el = time.time() - self._job_start_local_ts
            if local_el > self.current_job_elapsed:
                self.current_job_elapsed = local_el
                self.state_changed.emit()
        elif not self.is_generating:
            self._elapsed_timer.stop()

    def _on_background_tick(self) -> None:
        self._tick_counter += 1
        tick = self._tick_counter

        if self.is_connected:
            if not self.is_generating:
                self.check_colab_status(silent=True)
            elif tick % 6 == 0:
                self.check_colab_status(silent=True)
        elif self.normalized_base_url and (tick % 4 == 0):
            self.check_colab_status(silent=True)

        # Smart Keepalive & Official CU Ledger Sync
        if tick % 32 == 0 and self.cli_logged_in and not self.is_waking_colab:
            should_keepalive = self.is_connected and (
                self.is_generating
                or self.is_downloading_weights
                or self.is_loading_model
                or self.comfy_starting
                or self.comfy_downloading_weight
                or self.comfy_queue_running > 0
                or self.idle_timeout_minutes <= 0
                or self.idle_remaining_seconds > 75
            )
            if should_keepalive:
                self.keepalive_colab_runtime()
            else:
                self.refresh_official_colab_account_usage()
        elif tick % 16 == 0 and self.cli_logged_in and not self.is_connected:
            self.refresh_official_colab_account_usage()

    # ─── Remote Status & Telemetry Handling ──────────────────────────────────

    def reset_stale_remote_runtime_state(self, message: str = "") -> None:
        self.is_connected = False
        self.gpu_name = tr("尚未连接云端 GPU", "Cloud GPU Not Connected")
        self.vram_used_gb = 0.0
        self.vram_total_gb = 0.0
        self.vram_free_gb = 0.0
        self.ram_used_gb = 0.0
        self.ram_total_gb = 0.0
        self.ram_free_gb = 0.0
        self.disk_used_gb = 0.0
        self.disk_total_gb = 0.0
        self.disk_free_gb = 0.0
        self.shm_total_gb = 0.0
        self.shm_free_gb = 0.0
        self.drive_total_gb = 0.0
        self.drive_free_gb = 0.0
        self.drive_persisted = False
        self.vram_history_pct.clear()
        self.ram_history_pct.clear()
        self.disk_history_pct.clear()
        self.loaded_model_ids.clear()
        self.loaded_models_detail.clear()
        self.is_downloading_weights = False
        self.is_loading_model = False
        self.is_generating = False
        self.comfy_running = False
        self.comfy_starting = False
        self.comfy_public_url = ""
        self.cu_burn_rate_per_hour = 0.0
        self._remote_context_synced = False
        self._remote_idle_synced = False
        self._remote_api_key_synced = False
        if message:
            self.status_message = strip_emoji(self.masked_url(message))
        self.state_changed.emit()
        self.telemetry_changed.emit()

    def _record_telemetry_sample(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_telemetry_sample_ts) < 0.85:
            return
        self._last_telemetry_sample_ts = now

        def push(arr: List[float], used: float, total: float) -> None:
            if total <= 0:
                return
            pct = max(0.0, min(1.0, used / total))
            if not arr:
                arr.extend([pct] * 14)
            else:
                arr.append(pct)
                if len(arr) > 48:
                    del arr[: len(arr) - 48]

        push(self.vram_history_pct, self.vram_used_gb, self.vram_total_gb)
        push(self.ram_history_pct, self.ram_used_gb, self.ram_total_gb)
        push(self.disk_history_pct, self.disk_used_gb, self.disk_total_gb)

    def _apply_live_telemetry_snapshot(self, tel: dict) -> None:
        if not isinstance(tel, dict):
            return
        if tel.get("gpu_name"):
            self.gpu_name = strip_emoji(str(tel["gpu_name"]))
        for attr, key in (
            ("vram_free_gb", "vram_free_gb"),
            ("vram_total_gb", "vram_total_gb"),
            ("vram_used_gb", "vram_used_gb"),
            ("ram_total_gb", "ram_total_gb"),
            ("ram_used_gb", "ram_used_gb"),
            ("ram_free_gb", "ram_free_gb"),
            ("disk_total_gb", "disk_total_gb"),
            ("disk_used_gb", "disk_used_gb"),
            ("disk_free_gb", "disk_free_gb"),
            ("shm_total_gb", "shm_total_gb"),
            ("shm_free_gb", "shm_free_gb"),
            ("drive_total_gb", "drive_total_gb"),
            ("drive_free_gb", "drive_free_gb"),
            ("cu_burn_rate_per_hour", "cu_burn_rate_per_hour"),
        ):
            if tel.get(key) is not None:
                setattr(self, attr, float(tel[key]))
        self._record_telemetry_sample(force=False)
        self.telemetry_changed.emit()

    def _apply_status_response(self, d: dict, update_status_msg: bool = False) -> None:
        self._consecutive_status_failures = 0
        was_connected = self.is_connected
        self.is_connected = True

        if d.get("gpu_name"):
            self.gpu_name = strip_emoji(str(d["gpu_name"]))
        for attr in (
            "vram_free_gb",
            "vram_total_gb",
            "vram_used_gb",
            "ram_total_gb",
            "ram_used_gb",
            "ram_free_gb",
            "disk_total_gb",
            "disk_used_gb",
            "disk_free_gb",
            "shm_total_gb",
            "shm_free_gb",
            "drive_total_gb",
            "drive_free_gb",
            "cu_burn_rate_per_hour",
            "session_cu_used",
            "total_size_gb",
            "task_progress",
        ):
            if d.get(attr) is not None:
                val = float(d[attr])
                if attr == "task_progress":
                    self.model_task_progress = val
                else:
                    setattr(self, attr, val)

        if d.get("uptime_seconds") is not None:
            self.uptime_seconds = int(d["uptime_seconds"])
        if d.get("idle_remaining_seconds") is not None:
            self.idle_remaining_seconds = int(d["idle_remaining_seconds"])
        if d.get("api_request_count") is not None:
            self.api_request_count = int(d["api_request_count"])
        if d.get("drive_persisted") is not None:
            self.drive_persisted = bool(d["drive_persisted"])
        if d.get("loaded_context_length") is not None:
            self.loaded_context_length = int(d["loaded_context_length"])

        if isinstance(d.get("loaded_model_ids"), list):
            self.loaded_model_ids = [str(x) for x in d["loaded_model_ids"]]
        elif d.get("loaded_model_id"):
            self.loaded_model_ids = [str(d["loaded_model_id"])]
        elif d.get("any_model_loaded") is False:
            self.loaded_model_ids = []

        if isinstance(d.get("loaded_models_detail"), list):
            self.loaded_models_detail = [LoadedModelSummary.from_dict(x) for x in d["loaded_models_detail"]]

        if d.get("current_model"):
            self.current_model_name = strip_emoji(str(d["current_model"]))

        # Update files list only if response matches currently selected model
        resp_model = d.get("selected_model_id")
        if (not resp_model or resp_model == self.selected_model_id) and isinstance(d.get("files"), list):
            self.model_files = [ModelFileItem.from_dict(x) for x in d["files"]]
            self.downloaded_count = int(d.get("downloaded_count", sum(1 for f in self.model_files if f.exists)))
            self.total_count = int(d.get("total_count", len(self.model_files)))
            self.all_downloaded = bool(d.get("all_downloaded", self.total_count > 0 and self.downloaded_count >= self.total_count))

        if d.get("is_downloading") is not None:
            self.is_downloading_weights = bool(d["is_downloading"])
        if d.get("is_loading_vram") is not None:
            self.is_loading_model = bool(d["is_loading_vram"])

        # Console lines from server
        if isinstance(d.get("console_lines"), list) and d["console_lines"]:
            cleaned_lines = [
                strip_emoji(self.masked_url(str(ln)))
                for ln in d["console_lines"]
                if str(ln).strip()
            ]
            if cleaned_lines:
                self.live_console_lines = cleaned_lines[-400:]
                self.console_changed.emit()

        # ComfyUI status object
        comfy = d.get("comfyui")
        if isinstance(comfy, dict):
            self.comfy_running = bool(comfy.get("is_running", False))
            self.comfy_starting = bool(comfy.get("is_starting", False))
            self.comfy_syncing_drive = bool(comfy.get("is_syncing_drive", False))
            self.comfy_downloading_weight = bool(comfy.get("is_downloading_weight", False))
            self.comfy_installing_pack_id = str(comfy.get("installing_pack_id") or "")
            self.comfy_progress = float(comfy.get("progress") or 0.0)
            if comfy.get("status_message"):
                self.comfy_status_message = localize_server_text(
                    strip_emoji(self.masked_url(str(comfy["status_message"])))
                )
            if comfy.get("public_url") is not None:
                self.comfy_public_url = str(comfy.get("public_url") or "")
            self.comfy_queue_running = int(comfy.get("queue_running") or 0)
            self.comfy_queue_pending = int(comfy.get("queue_pending") or 0)
            if comfy.get("persistence_label"):
                self.comfy_persistence_label = localize_server_text(strip_emoji(str(comfy["persistence_label"])))
            if comfy.get("drive_persisted") is not None:
                self.drive_persisted = self.drive_persisted or bool(comfy["drive_persisted"])
            self.comfy_custom_nodes_count = int(comfy.get("custom_nodes_count") or 0)
            self.comfy_models_count = int(comfy.get("models_count") or 0)
            self.comfy_workflows_count = int(comfy.get("workflows_count") or 0)
            if isinstance(comfy.get("recent_outputs"), list):
                self.comfy_recent_outputs = [ComfyOutputMediaItem.from_dict(x) for x in comfy["recent_outputs"]]
            if isinstance(comfy.get("preset_packs"), list) and comfy["preset_packs"]:
                self.comfy_preset_packs = [ComfyPresetPackItem.from_dict(x) for x in comfy["preset_packs"]]

        if d.get("action_message"):
            self.status_message = localize_server_text(strip_emoji(self.masked_url(str(d["action_message"]))))
        elif d.get("task_message") and (self.is_downloading_weights or self.is_loading_model):
            self.status_message = localize_server_text(strip_emoji(self.masked_url(str(d["task_message"]))))
        elif update_status_msg and d.get("message"):
            self.status_message = localize_server_text(strip_emoji(self.masked_url(str(d["message"]))))
        elif not was_connected:
            self.status_message = tr(f"已连接云端 {self.gpu_name}！", f"Connected to cloud {self.gpu_name}!")

        # One-time per-connection remote config syncs
        if not self._remote_context_synced:
            self._remote_context_synced = True
            if d.get("context_length") != self.context_length_tokens:
                self.sync_context_length_to_colab()
        if not self._remote_idle_synced:
            self._remote_idle_synced = True
            if d.get("idle_timeout_minutes") != self.idle_timeout_minutes:
                self.update_idle_timeout_config(self.idle_timeout_minutes)
        if self.api_key_enabled and not bool(d.get("api_key_enabled", False)) and not self._remote_api_key_synced:
            self._remote_api_key_synced = True
            self.sync_api_key_to_colab()

        self._record_telemetry_sample(force=not was_connected)
        self.state_changed.emit()
        self.telemetry_changed.emit()

    def check_colab_status(self, silent: bool = False) -> None:
        base = self.normalized_base_url
        if not base:
            if not silent:
                self.status_message = tr(
                    "请先输入 https://xxxx.gradio.live 服务地址，或点击「唤醒实例并部署」。",
                    "Please enter a https://xxxx.gradio.live server URL or click 'Wake & Deploy'.",
                )
                self.state_changed.emit()
            return

        if silent and self._status_check_inflight:
            return
        self._status_check_inflight = True
        target_model = self.selected_model_id
        timeout_s = 28 if self.is_generating else (20 if silent else 22)

        def worker() -> None:
            try:
                r = requests.get(
                    f"{base}/minimax_api/status",
                    params={"model_id": target_model},
                    timeout=timeout_s,
                )
                if r.status_code == 200:
                    payload = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(payload, update_status_msg=not silent))
                elif r.status_code == 404:

                    def on_404() -> None:
                        self._consecutive_status_failures += 1 if self.is_generating else 2
                        if not silent or self._consecutive_status_failures >= 6:
                            self.reset_stale_remote_runtime_state(
                                tr(
                                    "云端隧道地址已失效或实例已释放，请重新唤醒部署。",
                                    "Cloud tunnel expired or runtime terminated. Please wake & deploy again.",
                                )
                            )

                    self._dispatch_ui(on_404)
                else:
                    code = r.status_code

                    def on_http_err() -> None:
                        self._consecutive_status_failures += 1
                        limit = 8 if self.is_generating else 5
                        if not silent or self._consecutive_status_failures >= limit:
                            self.reset_stale_remote_runtime_state(
                                tr(f"连接云端服务异常 (HTTP {code})。", f"Cloud server error (HTTP {code}).")
                            )

                    self._dispatch_ui(on_http_err)
            except Exception as exc:
                err_str = str(exc)

                def on_net_err() -> None:
                    self._consecutive_status_failures += 1
                    limit = 8 if self.is_generating else 5
                    if not silent or self._consecutive_status_failures >= limit:
                        self.reset_stale_remote_runtime_state(
                            tr(f"无法连接云端隧道：{err_str[:120]}", f"Cannot reach cloud tunnel: {err_str[:120]}")
                        )

                self._dispatch_ui(on_net_err)
            finally:
                self._status_check_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    # ─── Model Selection, Weight Download, Load & Unload ─────────────────────

    def select_model(self, model_id: str) -> None:
        if model_id not in MODEL_SPEC_MAP:
            return
        spec = MODEL_SPEC_MAP[model_id]
        self.selected_model_id = model_id
        if spec.modes:
            self.task_mode_id = spec.modes[0].id
        self.inference_steps = spec.default_steps
        self.model_files = [f.clone() for f in spec.default_files]
        self.downloaded_count = sum(1 for f in self.model_files if f.exists)
        self.total_count = len(self.model_files)
        self.all_downloaded = False
        self.state_changed.emit()

        base = self.normalized_base_url
        if self.is_connected and base and spec.modality != ModelModality.COMFY_UI:

            def worker() -> None:
                try:
                    r = requests.post(
                        f"{base}/minimax_api/select_model",
                        json={"model_id": model_id},
                        timeout=30,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
                except Exception:
                    pass

            threading.Thread(target=worker, daemon=True).start()
        elif self.is_connected and base:
            self.check_colab_status(silent=True)

    def set_task_mode(self, mode_id: str) -> None:
        self.task_mode_id = mode_id
        self.state_changed.emit()

    def download_model_weights(self) -> None:
        base = self.normalized_base_url
        if not self.is_connected or not base:
            self.status_message = tr("请先连接或唤醒 Colab 云端实例。", "Please connect or wake a Colab cloud runtime first.")
            self.state_changed.emit()
            return

        target_model = self.selected_model_id
        self.is_downloading_weights = True
        self.model_task_progress = 0.01
        self.status_message = tr(
            f"正在云端启动 8 线程极速下载 {self.active_model_spec.name} 权重...",
            f"Starting 8-thread cloud download for {self.active_model_spec.name} weights...",
        )
        self.state_changed.emit()

        def worker() -> None:
            try:
                r = requests.post(
                    f"{base}/minimax_api/download_model",
                    json={"model_id": target_model, "hf_token": self.hf_token.strip()},
                    timeout=60,
                )
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
                while True:
                    time.sleep(1.5)
                    sr = requests.get(
                        f"{base}/minimax_api/status",
                        params={"model_id": target_model},
                        timeout=25,
                    )
                    if sr.status_code != 200:
                        break
                    sdata = sr.json()
                    self._dispatch_ui(lambda sd=sdata: self._apply_status_response(sd, update_status_msg=True))
                    if not sdata.get("is_downloading", False):
                        break
            except Exception as exc:
                msg = tr(f"下载权重请求异常：{exc}", f"Weight download request error: {exc}")
                self._dispatch_ui(lambda: self._set_status_and_log(msg))
            finally:

                def done() -> None:
                    self.is_downloading_weights = False
                    self.state_changed.emit()

                self._dispatch_ui(done)

        threading.Thread(target=worker, daemon=True).start()

    def load_model_to_vram(self) -> None:
        base = self.normalized_base_url
        if not self.is_connected or not base:
            self.status_message = tr("请先连接或唤醒 Colab 云端实例。", "Please connect or wake a Colab cloud runtime first.")
            self.state_changed.emit()
            return
        if not self.all_downloaded:
            self.status_message = tr(
                "请先点击「下载权重」将模型分片拉取到云盘/实例磁盘。",
                "Please click 'Download Weights' to fetch model shards first.",
            )
            self.state_changed.emit()
            return

        target_model = self.selected_model_id
        self.is_loading_model = True
        self.model_task_progress = 0.05
        self.status_message = tr(
            f"正在将 {self.active_model_spec.name} 装载至云端 GPU 显存...",
            f"Loading {self.active_model_spec.name} into cloud GPU VRAM...",
        )
        self.state_changed.emit()

        payload = {
            "model_id": target_model,
            "precision_mode": self.selected_precision,
            "offload_mode": self.selected_offload,
            "context_length": self.context_length_tokens,
            "hf_token": self.hf_token.strip(),
        }

        def worker() -> None:
            try:
                r = requests.post(f"{base}/minimax_api/load_model", json=payload, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
                while True:
                    time.sleep(1.5)
                    sr = requests.get(
                        f"{base}/minimax_api/status",
                        params={"model_id": target_model},
                        timeout=25,
                    )
                    if sr.status_code != 200:
                        break
                    sdata = sr.json()
                    self._dispatch_ui(lambda sd=sdata: self._apply_status_response(sd, update_status_msg=True))
                    if not sdata.get("is_loading_vram", False):
                        break
            except Exception as exc:
                msg = tr(f"装载显存请求异常：{exc}", f"Load to VRAM request error: {exc}")
                self._dispatch_ui(lambda: self._set_status_and_log(msg))
            finally:

                def done() -> None:
                    self.is_loading_model = False
                    self.state_changed.emit()

                self._dispatch_ui(done)

        threading.Thread(target=worker, daemon=True).start()

    def _clear_local_loaded_record(self, model_id: str) -> None:
        if model_id == "all":
            self.loaded_model_ids.clear()
            self.loaded_models_detail.clear()
        else:
            self.loaded_model_ids = [m for m in self.loaded_model_ids if m != model_id]
            self.loaded_models_detail = [m for m in self.loaded_models_detail if m.id != model_id]

    def unload_model_from_vram(self, model_id: Optional[str] = None) -> None:
        target = model_id or self.selected_model_id
        base = self.normalized_base_url
        if not base:
            self._clear_local_loaded_record(target)
            self.state_changed.emit()
            return

        self.status_message = tr("正在从 GPU 显存卸载模型并回收缓存...", "Unloading model from GPU VRAM & freeing cache...")
        self.state_changed.emit()

        def worker() -> None:
            try:
                r = requests.post(
                    f"{base}/minimax_api/unload_model",
                    json={"model_id": target},
                    timeout=20,
                )
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
                    return
            except Exception:
                pass

            def fallback() -> None:
                self._clear_local_loaded_record(target)
                self.status_message = tr("已清除本地模型显存驻留记录。", "Cleared local VRAM residency record.")
                self.state_changed.emit()

            self._dispatch_ui(fallback)

        threading.Thread(target=worker, daemon=True).start()

    # ─── Remote Config Endpoints (Idle, API Key, Context) ────────────────────

    def update_idle_timeout_config(self, minutes: int) -> None:
        self.idle_timeout_minutes = minutes
        self._save_setting("idleTimeoutMinutes", minutes)
        self.state_changed.emit()

        base = self.normalized_base_url
        if not self.is_connected or not base:
            return

        def worker() -> None:
            try:
                r = requests.post(
                    f"{base}/minimax_api/config_idle",
                    json={"idle_timeout_minutes": minutes},
                    timeout=20,
                )
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=False))
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def set_api_key_enabled(self, enabled: bool) -> None:
        self.api_key_enabled = bool(enabled)
        self._save_setting("apiKeyEnabled", self.api_key_enabled)
        self.state_changed.emit()
        self.sync_api_key_to_colab()

    def regenerate_api_key(self) -> None:
        self.api_access_key = "sk-colab-" + secrets.token_hex(10)
        self._save_setting("apiAccessKey", self.api_access_key)
        self.status_message = tr("已生成新的随机 API Key。", "Generated a new random API Key.")
        self.state_changed.emit()
        if self.is_connected:
            self.sync_api_key_to_colab()

    def sync_api_key_to_colab(self, custom_key: Optional[str] = None) -> None:
        if custom_key is not None:
            self.api_access_key = custom_key.strip()
            self._save_setting("apiAccessKey", self.api_access_key)
        base = self.normalized_base_url
        if not self.is_connected or not base:
            return
        payload_key = self.api_access_key.strip() if self.api_key_enabled else ""

        def worker() -> None:
            try:
                r = requests.post(
                    f"{base}/minimax_api/config_api_key",
                    json={"api_key": payload_key},
                    timeout=20,
                )
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=False))
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def sync_context_length_to_colab(self) -> None:
        base = self.normalized_base_url
        if not self.is_connected or not base:
            return
        ctx_tokens = self.context_length_tokens

        def worker() -> None:
            try:
                r = requests.post(
                    f"{base}/minimax_api/config_context",
                    json={"context_length": ctx_tokens},
                    timeout=20,
                )
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=False))
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    # ─── Generation & Job Polling State Machine ──────────────────────────────

    def start_generation(
        self,
        first_frame_b64: Optional[str] = None,
        last_frame_b64: Optional[str] = None,
    ) -> None:
        if self.is_generating:
            return

        spec = self.active_model_spec
        modality = spec.modality
        prompt = self.visual_prompt.strip()
        if not prompt:
            self.status_message = tr("请先输入提示词或对话内容。", "Please enter a prompt or message first.")
            self.state_changed.emit()
            return

        if self.task_mode_id == "fl2va" and not first_frame_b64 and not last_frame_b64:
            self.status_message = tr(
                "「首尾帧生视频」模式请至少上传起始首帧或结束尾帧参考图。",
                "First/Last-Frame Video mode requires at least a start or end reference frame.",
            )
            self.state_changed.emit()
            return
        if self.task_mode_id == "i2i_edit" and not first_frame_b64 and not last_frame_b64:
            self.status_message = tr(
                "指令修图/重绘模式请至少上传一张参考原图。",
                "Instruction Edit / Img2Img mode requires at least one reference image.",
            )
            self.state_changed.emit()
            return

        base = self.normalized_base_url
        if not self.is_connected or not base:
            self.status_message = tr(
                "尚未连接云端 Colab 服务，请先唤醒或连接实例。",
                "Not connected to Colab server. Please wake or connect a runtime first.",
            )
            self.state_changed.emit()
            return
        if not self.is_selected_model_loaded_in_vram:
            self.status_message = tr(
                f"请先在左侧模型中心点击「加载到显存」装载 {spec.name}。",
                f"Please click 'Load to VRAM' in the Model Center to load {spec.name} first.",
            )
            self.state_changed.emit()
            return

        token = str(uuid.uuid4())
        self._active_generation_token = token
        self.is_generating = True
        self.progress_value = 0.01
        self.current_job_elapsed = 0.0
        self._job_start_local_ts = time.time()
        self._elapsed_timer.start()

        width, height = (
            (self.selected_image_res.width, self.selected_image_res.height)
            if modality == ModelModality.IMAGE
            else (self.selected_video_res.width, self.selected_video_res.height)
        )

        assistant_idx: Optional[int] = None
        chat_messages_payload: Optional[List[Dict[str, str]]] = None
        if modality == ModelModality.LLM_VLM:
            has_img = bool(first_frame_b64 or last_frame_b64) and self.task_mode_id == "vlm"
            self.chat_history.append(ChatTurnItem(role="user", content=prompt, has_image=has_img))
            chat_messages_payload = [{"role": t.role, "content": t.content} for t in self.chat_history]
            self.chat_history.append(ChatTurnItem(role="assistant", content="", thinking=""))
            assistant_idx = len(self.chat_history) - 1
            self.visual_prompt = ""
            self.chat_changed.emit()

        payload: Dict[str, Any] = {
            "model_id": self.selected_model_id,
            "task_mode": self.task_mode_id,
            "visual_prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "soundscape_prompt": self.soundscape_prompt,
            "music_prompt": self.music_prompt,
            "width": width,
            "height": height,
            "num_frames": snap_video_frames(self.num_frames),
            "num_inference_steps": int(self.inference_steps),
            "fps": 24,
            "seed": int(self.seed),
            "randomize_seed": bool(self.randomize_seed),
            "batch_count": 1 if modality == ModelModality.LLM_VLM else int(self.batch_count),
            "denoising_strength": float(self.denoising_strength),
            "enable_thinking": bool(self.enable_thinking),
            "max_new_tokens": int(self.max_new_tokens),
            "context_length": self.context_length_tokens,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": float(self.presence_penalty),
            "repeat_penalty": 1.0,
            "reasoning_effort": self.reasoning_effort,
        }
        if chat_messages_payload is not None:
            payload["chat_messages"] = chat_messages_payload
        if self.task_mode_id in {"fl2va", "i2v", "i2i_edit", "vlm"}:
            if first_frame_b64:
                payload["first_frame_base64"] = first_frame_b64
            if last_frame_b64:
                payload["last_frame_base64"] = last_frame_b64

        self.status_message = tr(
            f"正在向云端提交 {spec.name} 推理任务...",
            f"Submitting {spec.name} inference job to cloud...",
        )
        self.state_changed.emit()

        def worker() -> None:
            try:
                r = requests.post(f"{base}/minimax_api/generate", json=payload, timeout=30)
                if r.status_code != 200:
                    err_txt = r.text[:200]
                    raise RuntimeError(f"HTTP {r.status_code}: {err_txt}")
                job = r.json()
                if job.get("status") == "failed":
                    msg = localize_server_text(
                        strip_emoji(str(job.get("message") or tr("任务提交失败", "Failed to submit job")))
                    )

                    def on_fail() -> None:
                        if self._active_generation_token != token:
                            return
                        if assistant_idx is not None and assistant_idx < len(self.chat_history):
                            self.chat_history[assistant_idx].content = msg
                            self.chat_changed.emit()
                        self.is_generating = False
                        self.progress_value = 0.0
                        self.status_message = msg
                        self.state_changed.emit()

                    self._dispatch_ui(on_fail)
                    return

                job_id = str(job.get("job_id") or "")
                self._poll_colab_job(base, job_id, token, assistant_idx)
            except Exception as exc:
                err_msg = tr(f"提交推理任务失败：{exc}", f"Failed to submit inference job: {exc}")

                def on_exc() -> None:
                    if self._active_generation_token != token:
                        return
                    if assistant_idx is not None and assistant_idx < len(self.chat_history):
                        self.chat_history[assistant_idx].content = err_msg
                        self.chat_changed.emit()
                    self.is_generating = False
                    self.progress_value = 0.0
                    self.status_message = err_msg
                    self.state_changed.emit()

                self._dispatch_ui(on_exc)

        threading.Thread(target=worker, daemon=True).start()

    def _download_media_to_local(self, base_url: str, rel_or_abs_url: str, filename: str, elapsed: Optional[float]) -> Optional[Path]:
        full_url = rel_or_abs_url if rel_or_abs_url.startswith("http") else f"{base_url}{rel_or_abs_url}"
        save_dir = get_local_save_dir()
        safe_name = Path(filename).name
        dst = save_dir / safe_name
        r = requests.get(full_url, timeout=180)
        if r.status_code == 200 and r.content:
            dst.write_bytes(r.content)
            if elapsed is not None and elapsed > 0:
                self.media_elapsed_map[safe_name] = float(elapsed)
                self._save_setting("mediaElapsedSecondsMap", self.media_elapsed_map)
            return dst
        return None

    def _poll_colab_job(
        self,
        base: str,
        job_id: str,
        token: str,
        assistant_idx: Optional[int],
    ) -> None:
        poll_interval = 0.6 if assistant_idx is not None else 1.0
        consecutive_404 = 0
        consecutive_errors = 0
        downloaded_filenames: set[str] = set()

        while self._active_generation_token == token:
            time.sleep(poll_interval)
            if self._active_generation_token != token:
                return

            try:
                r = requests.get(f"{base}/minimax_api/job/{job_id}", timeout=30)
                if r.status_code == 404:
                    consecutive_404 += 1
                    if consecutive_404 >= 5:
                        self._dispatch_ui(
                            lambda: self.reset_stale_remote_runtime_state(
                                tr(
                                    "云端推理进程已中断（可能因显存溢出或实例释放），请重新唤醒实例。",
                                    "Cloud inference process interrupted (possible OOM or runtime release). Please wake the runtime again.",
                                )
                            )
                        )
                        return
                    continue
                if r.status_code != 200:
                    consecutive_errors += 1
                    if consecutive_errors >= 12:

                        def on_abort() -> None:
                            if self._active_generation_token == token:
                                self.is_generating = False
                                self.status_message = tr(
                                    "轮询云端任务多次失败，请检查网络连接。",
                                    "Polling cloud job failed repeatedly. Please check your network connection.",
                                )
                                self.state_changed.emit()

                        self._dispatch_ui(on_abort)
                        return
                    continue

                consecutive_404 = 0
                consecutive_errors = 0
                job = r.json()
            except Exception:
                consecutive_errors += 1
                if consecutive_errors >= 12:

                    def on_net_abort() -> None:
                        if self._active_generation_token == token:
                            self.is_generating = False
                            self.status_message = tr(
                                "网络连接连续超时，已停止轮询。",
                                "Network connection timed out repeatedly. Stopped polling.",
                            )
                            self.state_changed.emit()

                    self._dispatch_ui(on_net_abort)
                    return
                continue

            # Progressive media download on worker thread before final UI update
            newly_saved: List[tuple[Path, Optional[float]]] = []
            for item in job.get("completed_videos") or []:
                fname = str(item.get("filename") or "")
                vurl = str(item.get("video_url") or "")
                el = float(item["elapsed"]) if item.get("elapsed") is not None else None
                if fname and vurl and fname not in downloaded_filenames:
                    try:
                        saved = self._download_media_to_local(base, vurl, fname, el)
                        if saved:
                            downloaded_filenames.add(fname)
                            newly_saved.append((saved, el))
                    except Exception:
                        pass

            status = str(job.get("status") or "")
            if status == "completed":
                fname = str(job.get("filename") or "")
                vurl = str(job.get("video_url") or "")
                el = float(job["elapsed"]) if job.get("elapsed") is not None else None
                if fname and vurl and fname not in downloaded_filenames:
                    try:
                        saved = self._download_media_to_local(base, vurl, fname, el)
                        if saved:
                            downloaded_filenames.add(fname)
                            newly_saved.append((saved, el))
                    except Exception:
                        pass

            def update_ui(
                j: dict = job,
                st: str = status,
                saved_files: List[tuple[Path, Optional[float]]] = newly_saved,
            ) -> None:
                if self._active_generation_token != token:
                    return

                if isinstance(j.get("telemetry"), dict):
                    self._apply_live_telemetry_snapshot(j["telemetry"])

                raw_prog = float(j.get("progress") or 0.0)
                if st != "completed":
                    self.progress_value = min(0.98, max(0.0, raw_prog))
                if j.get("message"):
                    self.status_message = localize_server_text(strip_emoji(self.masked_url(str(j["message"]))))
                if j.get("elapsed") is not None:
                    self.current_job_elapsed = max(self.current_job_elapsed, float(j["elapsed"]))
                if j.get("seed") is not None:
                    self.seed = int(j["seed"])

                if isinstance(j.get("console_lines"), list) and j["console_lines"]:
                    self.live_console_lines = [
                        strip_emoji(self.masked_url(str(ln)))
                        for ln in j["console_lines"]
                        if str(ln).strip()
                    ][-400:]
                    self.console_changed.emit()

                if assistant_idx is not None and assistant_idx < len(self.chat_history):
                    turn = self.chat_history[assistant_idx]
                    if j.get("thinking_output") is not None:
                        turn.thinking = str(j["thinking_output"])
                    if j.get("text_output") is not None:
                        turn.content = str(j["text_output"])
                    self.chat_changed.emit()

                if saved_files:
                    self.refresh_local_media_gallery(auto_select_first=False)
                    latest_path, latest_el = saved_files[-1]
                    self.select_local_media(latest_path, auto_play=True)

                if st == "completed":
                    self.is_generating = False
                    self.progress_value = 1.0
                    b_cnt = int(j.get("batch_count") or 1)
                    if b_cnt > 1 and saved_files:
                        self.status_message = tr(
                            f"批量生成全部完成（共 {b_cnt} 项），已自动回传保存至本地目录！",
                            f"Batch generation complete ({b_cnt} items) and auto-saved locally!",
                        )
                    self.state_changed.emit()
                    self.check_colab_status(silent=True)
                elif st == "failed":
                    msg = localize_server_text(
                        strip_emoji(str(j.get("message") or tr("云端推理失败", "Cloud inference failed")))
                    )
                    if assistant_idx is not None and assistant_idx < len(self.chat_history):
                        if not self.chat_history[assistant_idx].content:
                            self.chat_history[assistant_idx].content = msg
                            self.chat_changed.emit()
                    self.is_generating = False
                    self.progress_value = 0.0
                    self.status_message = msg
                    self.state_changed.emit()
                else:
                    self.state_changed.emit()

            self._dispatch_ui(update_ui)
            if status in {"completed", "failed"}:
                return

    def cancel_current_generation(self) -> None:
        self._active_generation_token = str(uuid.uuid4())
        self.is_generating = False
        self.progress_value = 0.0
        self.status_message = tr(
            "已发送停止信号，正在中断云端 GPU 当前采样任务...",
            "Stop signal sent. Interrupting active cloud GPU sampling job...",
        )
        self.live_console_lines = [
            ln for ln in self.live_console_lines if "实时下载/采样进度流" not in ln
        ]
        self.state_changed.emit()
        self.console_changed.emit()

        base = self.normalized_base_url
        if base:
            threading.Thread(
                target=lambda: self._safe_post_empty(f"{base}/minimax_api/cancel", timeout=12),
                daemon=True,
            ).start()

    @staticmethod
    def _safe_post_empty(url: str, timeout: int = 12) -> None:
        try:
            requests.post(url, json={}, timeout=timeout)
        except Exception:
            pass

    # ─── ComfyUI Studio Actions ──────────────────────────────────────────────

    def start_comfyui(self, force_restart: bool = False, auto_open_browser: bool = True) -> None:
        base = self.normalized_base_url
        if not self.is_connected or not base:
            self.status_message = tr(
                "请先连接或唤醒 Colab 云端实例。",
                "Please connect or wake a Colab cloud runtime first.",
            )
            self.state_changed.emit()
            return

        self.comfy_starting = True
        self.status_message = tr(
            "正在云端部署启动 ComfyUI 节点画布与 Cloudflare 加密隧道...",
            "Deploying ComfyUI node canvas & Cloudflare HTTPS tunnel on cloud...",
        )
        self.state_changed.emit()

        payload = {
            "unload_existing": bool(self.comfy_unload_others_on_start),
            "force_restart": bool(force_restart),
        }

        def worker() -> None:
            try:
                r = requests.post(f"{base}/minimax_api/comfyui/start", json=payload, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
                for _ in range(180):
                    time.sleep(1.5)
                    sr = requests.get(
                        f"{base}/minimax_api/status",
                        params={"model_id": self.selected_model_id},
                        timeout=22,
                    )
                    if sr.status_code != 200:
                        break
                    sdata = sr.json()
                    self._dispatch_ui(lambda sd=sdata: self._apply_status_response(sd, update_status_msg=False))
                    comfy = sdata.get("comfyui") or {}
                    pub_url = str(comfy.get("public_url") or "")
                    if comfy.get("is_running") and pub_url and not comfy.get("is_starting"):
                        if auto_open_browser:
                            webbrowser.open(pub_url)
                        break
            except Exception as exc:
                msg = tr(f"启动 ComfyUI 异常：{exc}", f"Error starting ComfyUI: {exc}")
                self._dispatch_ui(lambda: self._set_status_and_log(msg))
            finally:

                def done() -> None:
                    self.comfy_starting = False
                    self.state_changed.emit()

                self._dispatch_ui(done)

        threading.Thread(target=worker, daemon=True).start()

    def stop_comfyui(self) -> None:
        base = self.normalized_base_url
        if not base:
            return
        self.status_message = tr(
            "正在同步工作流至云盘并停止 ComfyUI 服务...",
            "Syncing workflows to Google Drive and stopping ComfyUI...",
        )
        self.state_changed.emit()

        def worker() -> None:
            try:
                r = requests.post(f"{base}/minimax_api/comfyui/stop", json={}, timeout=35)
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
            except Exception as exc:
                self._dispatch_ui(
                    lambda: self._set_status_and_log(
                        tr(f"停止 ComfyUI 异常：{exc}", f"Error stopping ComfyUI: {exc}")
                    )
                )

        threading.Thread(target=worker, daemon=True).start()

    def free_comfy_vram(self) -> None:
        base = self.normalized_base_url
        if not base:
            return

        def worker() -> None:
            try:
                r = requests.post(f"{base}/minimax_api/comfyui/free_vram", json={}, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
            except Exception as exc:
                self._dispatch_ui(
                    lambda: self._set_status_and_log(
                        tr(f"清理显存异常：{exc}", f"Error freeing VRAM: {exc}")
                    )
                )

        threading.Thread(target=worker, daemon=True).start()

    def sync_comfy_drive(self) -> None:
        base = self.normalized_base_url
        if not base:
            return
        self.comfy_syncing_drive = True
        self.status_message = tr(
            "正在将 ComfyUI 自定义节点、模型与工作流双向同步至 Google Drive...",
            "Syncing ComfyUI custom nodes, models & workflows to Google Drive...",
        )
        self.state_changed.emit()

        def worker() -> None:
            try:
                r = requests.post(f"{base}/minimax_api/comfyui/sync_drive", json={}, timeout=90)
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
            except Exception as exc:
                self._dispatch_ui(
                    lambda: self._set_status_and_log(
                        tr(f"同步云盘异常：{exc}", f"Error syncing to Google Drive: {exc}")
                    )
                )
            finally:

                def done() -> None:
                    self.comfy_syncing_drive = False
                    self.state_changed.emit()

                self._dispatch_ui(done)

        threading.Thread(target=worker, daemon=True).start()

    def download_comfy_weight(self, url_or_repo: str, filename: str, subfolder: str) -> None:
        src = (url_or_repo or "").strip()
        if not src:
            self.status_message = tr(
                "请先填写模型直链 URL 或 HuggingFace Repo ID。",
                "Please enter a direct model URL or Hugging Face Repo ID first.",
            )
            self.state_changed.emit()
            return
        base = self.normalized_base_url
        if not self.is_connected or not base:
            self.status_message = tr(
                "请先连接或唤醒 Colab 云端实例。",
                "Please connect or wake a Colab cloud runtime first.",
            )
            self.state_changed.emit()
            return

        self.comfy_downloading_weight = True
        self.status_message = tr(
            f"正在 8 线程极速下载权重至 ComfyUI/models/{subfolder}...",
            f"8-thread downloading weights to ComfyUI/models/{subfolder}...",
        )
        self.state_changed.emit()

        payload = {
            "url_or_repo": src,
            "filename": (filename or "").strip(),
            "subfolder": subfolder,
            "hf_token": self.hf_token.strip(),
        }

        def worker() -> None:
            try:
                r = requests.post(f"{base}/minimax_api/comfyui/download_weight", json=payload, timeout=25)
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
                while True:
                    time.sleep(1.5)
                    sr = requests.get(
                        f"{base}/minimax_api/status",
                        params={"model_id": self.selected_model_id},
                        timeout=22,
                    )
                    if sr.status_code != 200:
                        break
                    sdata = sr.json()
                    self._dispatch_ui(lambda sd=sdata: self._apply_status_response(sd, update_status_msg=False))
                    comfy = sdata.get("comfyui") or {}
                    if not comfy.get("is_downloading_weight", False):
                        break
            except Exception as exc:
                self._dispatch_ui(
                    lambda: self._set_status_and_log(
                        tr(f"下载 ComfyUI 权重异常：{exc}", f"Error downloading ComfyUI weight: {exc}")
                    )
                )
            finally:

                def done() -> None:
                    self.comfy_downloading_weight = False
                    self.state_changed.emit()

                self._dispatch_ui(done)

        threading.Thread(target=worker, daemon=True).start()

    def install_comfy_pack(self, pack_id: str, open_when_ready: bool = True) -> None:
        base = self.normalized_base_url
        if not self.is_connected or not base:
            self.status_message = tr(
                "请先连接或唤醒 Colab 云端实例。",
                "Please connect or wake a Colab cloud runtime first.",
            )
            self.state_changed.emit()
            return

        self.comfy_installing_pack_id = pack_id
        self.status_message = tr(
            "正在云端一键配齐预置工作流模型包并写入全中文节点模板...",
            "Installing preset workflow pack & generating guided node templates on cloud...",
        )
        self.state_changed.emit()

        payload = {
            "pack_id": pack_id,
            "auto_start": True,
            "hf_token": self.hf_token.strip(),
        }

        def worker() -> None:
            try:
                r = requests.post(f"{base}/minimax_api/comfyui/install_pack", json=payload, timeout=30)
                if r.status_code == 404:
                    self._dispatch_ui(
                        lambda: self._set_status_and_log(
                            tr(
                                "当前云端服务端版本较旧，请点击左侧「唤醒实例并部署」热更新服务端代码。",
                                "Cloud server version is outdated. Click 'Wake & Deploy' on the left to hot-update the server.",
                            )
                        )
                    )
                    return
                if r.status_code == 200:
                    data = r.json()
                    self._dispatch_ui(lambda: self._apply_status_response(data, update_status_msg=True))
                for _ in range(600):
                    time.sleep(1.5)
                    sr = requests.get(
                        f"{base}/minimax_api/status",
                        params={"model_id": self.selected_model_id},
                        timeout=22,
                    )
                    if sr.status_code != 200:
                        break
                    sdata = sr.json()
                    self._dispatch_ui(lambda sd=sdata: self._apply_status_response(sd, update_status_msg=False))
                    comfy = sdata.get("comfyui") or {}
                    busy = (
                        bool(comfy.get("is_downloading_weight"))
                        or bool(comfy.get("installing_pack_id"))
                        or bool(comfy.get("is_starting"))
                    )
                    if not busy:
                        pub_url = str(comfy.get("public_url") or "")
                        if open_when_ready and comfy.get("is_running") and pub_url:
                            webbrowser.open(pub_url)
                        break
            except Exception as exc:
                self._dispatch_ui(
                    lambda: self._set_status_and_log(
                        tr(f"安装预置工作流包异常：{exc}", f"Error installing preset workflow pack: {exc}")
                    )
                )
            finally:

                def done() -> None:
                    self.comfy_installing_pack_id = ""
                    self.state_changed.emit()

                self._dispatch_ui(done)

        threading.Thread(target=worker, daemon=True).start()

    def pull_comfy_output_media(self, item: ComfyOutputMediaItem) -> None:
        base = self.normalized_base_url
        if not base:
            return
        encoded_rel = "/".join(urllib.parse.quote(seg, safe="") for seg in item.rel_path.split("/"))
        url = f"{base}/minimax_api/comfyui/output_media/{encoded_rel}"
        save_name = "comfy_" + Path(item.filename).name
        dst = get_local_save_dir() / save_name
        self.status_message = tr(
            f"正在从云端 ComfyUI 回传成片 {item.filename}...",
            f"Pulling output media {item.filename} from cloud ComfyUI...",
        )
        self.state_changed.emit()

        def worker() -> None:
            try:
                r = requests.get(url, timeout=180)
                if r.status_code == 200 and r.content:
                    dst.write_bytes(r.content)

                    def on_saved() -> None:
                        self.status_message = tr(
                            f"已回传保存至本地：{dst.name}",
                            f"Saved locally: {dst.name}",
                        )
                        self.refresh_local_media_gallery(auto_select_first=False)
                        self.select_local_media(dst, auto_play=True)
                        self.state_changed.emit()

                    self._dispatch_ui(on_saved)
                else:
                    code = r.status_code
                    self._dispatch_ui(
                        lambda: self._set_status_and_log(
                            tr(f"回传成片失败 (HTTP {code})", f"Failed to pull output media (HTTP {code})")
                        )
                    )
            except Exception as exc:
                self._dispatch_ui(
                    lambda: self._set_status_and_log(
                        tr(f"回传成片异常：{exc}", f"Error pulling output media: {exc}")
                    )
                )

        threading.Thread(target=worker, daemon=True).start()
