"""
Main 3-Column Studio Window (1480x900) for the EasyColab Windows Client.
Implements 1:1 feature & layout parity with macOS EasyColabStudio.swift:
  - Left Sidebar (356px): Colab Instance + Telemetry Spline Charts, CU & Idle Sleep,
    Open Model Center (6 models + co-residency + weight shard manager), Adaptive Params.
  - Center Studio Workspace: Mode Switcher + MiniMax-H3 Video/Audio Studio, ComfyUI
    4-Pack Node Studio, Qwen3.8 Chat/VLM/Dual-Protocol API Gateway, Qwen-Image & Z-Image.
  - Right Collapsible Run Log Panel (352px): Live task progress + color-coded dark terminal.
"""
from __future__ import annotations

import html
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QFontMetrics, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QMainWindow,
)

from windows_app.models import (
    COMFY_WEIGHT_SUBFOLDERS,
    GPU_CHOICES,
    IMAGE_RESOLUTIONS,
    MODEL_SPEC_MAP,
    OPEN_MODEL_CATALOG,
    PRESET_TEMPLATES,
    VIDEO_DURATION_PRESETS,
    VIDEO_RESOLUTIONS,
    ComfyOutputMediaItem,
    ComfyPresetPackItem,
    ModelModality,
    OpenModelSpec,
    format_duration,
    format_uptime_hms,
    get_idle_timeout_choices,
    get_lang,
    get_local_save_dir,
    is_en,
    mask_email_display,
    open_path_in_os,
    snap_video_frames,
    tr,
)
from windows_app.onboarding_dialog import ColabGuideDialog
from windows_app.theme import (
    DARK_PALETTE,
    LIGHT_PALETTE,
    StudioPalette,
    build_app_icon,
    build_stylesheet,
    draw_easycolab_icon_pixmap,
    icon_glyph,
    metric_threshold_color,
)
from windows_app.viewmodel import EasyColabViewModel
from windows_app.widgets import (
    AutoHeightStackedWidget,
    CollapsibleBox,
    CombinedTelemetryChartWidget,
    FrameDropWell,
    MediaPreviewWidget,
    ModeRadioCard,
    PillBadge,
    SidebarCard,
    SingleMetricChartWidget,
    StatusDot,
    WorkspaceCard,
    set_btn_role,
)


class EasyColabMainWindow(QMainWindow):
    def __init__(self, vm: EasyColabViewModel):
        super().__init__()
        self.vm = vm
        self._syncing_ui = False
        self._show_right_log = True
        self._last_coresident_sig: object = None
        self._last_shards_sig: object = None
        self._last_modes_sig: object = None
        self._last_packs_sig: object = None
        self._last_cfy_out_sig: object = None

        self.setWindowTitle(
            tr(
                "EasyColab — 云端大显存多模态 AI 创作工作台",
                "EasyColab — Cloud High-VRAM Multimodal AI Studio",
            )
        )
        self.setWindowIcon(build_app_icon())
        self.resize(1480, 900)
        self.setMinimumSize(1280, 820)

        central = QWidget()
        self.setCentralWidget(central)
        main_hbox = QHBoxLayout(central)
        main_hbox.setContentsMargins(0, 0, 0, 0)
        main_hbox.setSpacing(0)

        # 1. Left Sidebar (356px)
        self.left_sidebar = self._build_left_sidebar()
        main_hbox.addWidget(self.left_sidebar)

        # 2. Center Studio Workspace (expanding)
        self.center_workspace = self._build_center_workspace()
        main_hbox.addWidget(self.center_workspace, 1)

        # 3. Right Collapsible Run Log Panel (352px)
        self.right_log_panel = self._build_right_log_panel()
        main_hbox.addWidget(self.right_log_panel)

        # Shortcuts
        sc_gen1 = QShortcut(QKeySequence("Ctrl+Return"), self)
        sc_gen1.activated.connect(self._trigger_generate)
        sc_gen2 = QShortcut(QKeySequence("Ctrl+Enter"), self)
        sc_gen2.activated.connect(self._trigger_generate)
        sc_help = QShortcut(QKeySequence("Ctrl+Shift+/"), self)
        sc_help.activated.connect(lambda: self.open_onboarding_guide(0))
        sc_f1 = QShortcut(QKeySequence("F1"), self)
        sc_f1.activated.connect(lambda: self.open_onboarding_guide(0))

        # Wire ViewModel signals
        self.vm.state_changed.connect(self._refresh_all_state)
        self.vm.telemetry_changed.connect(self._refresh_telemetry)
        self.vm.console_changed.connect(self._refresh_console)
        self.vm.chat_changed.connect(self._refresh_chat_conversation)
        self.vm.media_list_changed.connect(self._refresh_media_gallery)
        self.vm.media_selected.connect(self._on_media_selected)
        self.vm.theme_changed.connect(self._apply_theme)
        self.vm.language_changed.connect(self._on_language_changed)

        self._apply_theme()
        self._refresh_all_state()
        self._refresh_telemetry()
        self._refresh_console()
        self._refresh_media_gallery()

        # First-run onboarding guide after 350ms
        if not self.vm.has_seen_onboarding:
            QTimer.singleShot(350, lambda: self.open_onboarding_guide(0))

    @property
    def palette_obj(self) -> StudioPalette:
        return DARK_PALETTE if self.vm.dark_mode else LIGHT_PALETTE

    def _on_language_changed(self) -> None:
        """Rebuild the 3-column studio central widget live while preserving transient UI state."""
        # 1. Snapshot transient widget states not stored on self.vm
        well_paths = {
            "video_first_well": getattr(self.video_first_well, "_file_path", None),
            "video_last_well": getattr(self.video_last_well, "_file_path", None),
            "vlm_img1_well": getattr(self.vlm_img1_well, "_file_path", None),
            "vlm_img2_well": getattr(self.vlm_img2_well, "_file_path", None),
            "img_ref1_well": getattr(self.img_ref1_well, "_file_path", None),
            "img_ref2_well": getattr(self.img_ref2_well, "_file_path", None),
        }
        show_log = self._show_right_log
        oauth_txt = self.oauth_input.text()
        hf_tok_txt = self.hf_token_input.text()
        cfy_src_txt = self.cfy_src_input.text()
        cfy_fn_txt = self.cfy_fname_input.text()
        cfy_sub_val = self.cfy_subfolder_combo.currentData() or self.cfy_subfolder_combo.currentText()
        llm_txt = (
            self.vm.visual_prompt
            if self.vm.active_model_spec.modality == ModelModality.LLM_VLM
            else self.llm_prompt_edit.toPlainText()
        )
        api_key_txt = self.api_key_input.text()
        shards_expanded = bool(getattr(self.shards_box, "_expanded", False))
        cur_media = self.vm.current_media_path

        # 2. Cleanly stop video playback on old preview widget before teardown
        self.media_preview.stop_video()

        # 3. Reset cached layout signatures
        self._last_coresident_sig = None
        self._last_shards_sig = None
        self._last_modes_sig = None
        self._last_packs_sig = None
        self._last_cfy_out_sig = None

        # 4. Update window title
        self.setWindowTitle(
            tr(
                "EasyColab — 云端大显存多模态 AI 创作工作台",
                "EasyColab — Cloud High-VRAM Multimodal AI Studio",
            )
        )

        # 5. Rebuild fresh central widget
        central = QWidget()
        main_hbox = QHBoxLayout(central)
        main_hbox.setContentsMargins(0, 0, 0, 0)
        main_hbox.setSpacing(0)

        self.left_sidebar = self._build_left_sidebar()
        main_hbox.addWidget(self.left_sidebar)

        self.center_workspace = self._build_center_workspace()
        main_hbox.addWidget(self.center_workspace, 1)

        self.right_log_panel = self._build_right_log_panel()
        self.right_log_panel.setVisible(show_log)
        main_hbox.addWidget(self.right_log_panel)

        old_central = self.takeCentralWidget()
        self.setCentralWidget(central)
        if old_central is not None:
            old_central.deleteLater()

        # 6. Restore snapshotted transient inputs & drop wells
        self._syncing_ui = True
        self.oauth_input.setText(oauth_txt)
        self.hf_token_input.setText(hf_tok_txt)
        self.cfy_src_input.setText(cfy_src_txt)
        self.cfy_fname_input.setText(cfy_fn_txt)
        sub_idx = self.cfy_subfolder_combo.findData(cfy_sub_val)
        if sub_idx >= 0:
            self.cfy_subfolder_combo.setCurrentIndex(sub_idx)
        self.llm_prompt_edit.setPlainText(llm_txt)
        if api_key_txt:
            self.api_key_input.setText(api_key_txt)
        self.shards_box.set_expanded(shards_expanded)
        self._syncing_ui = False

        for attr_name, fpath in well_paths.items():
            if fpath is not None and Path(fpath).exists():
                getattr(self, attr_name).load_from_file(Path(fpath))

        self._apply_theme()
        self._refresh_all_state()
        self._refresh_telemetry()
        self._refresh_console()
        self._refresh_chat_conversation()
        self._refresh_media_gallery()
        if cur_media is not None and cur_media.exists():
            el = self.vm.media_elapsed_map.get(cur_media.name)
            self.media_preview.load_media(cur_media, elapsed_seconds=el, auto_play=False)

    def _apply_theme(self) -> None:
        pal = self.palette_obj
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet(pal))
        self.theme_btn.setText(tr("亮色", "Light") if pal.is_dark else tr("深色", "Dark"))
        self.theme_btn.setToolTip(tr("切换亮色 / 深色工作台主题", "Toggle Light / Dark studio theme"))
        for w in (self.chart_vram, self.chart_ram, self.chart_disk, self.chart_combined):
            w.set_palette_theme(pal)
        for well in (
            self.video_first_well,
            self.video_last_well,
            self.vlm_img1_well,
            self.vlm_img2_well,
            self.img_ref1_well,
            self.img_ref2_well,
        ):
            well.set_palette_theme(pal)
        self.media_preview.set_palette_theme(pal)
        self._refresh_all_state()
        self._refresh_chat_conversation()
        self._refresh_console()

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. LEFT SIDEBAR (356px, 4 Cards)
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_left_sidebar(self) -> QWidget:
        container = QWidget()
        container.setObjectName("LeftSidebarContainer")
        container.setFixedWidth(356)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Top 40px header
        hdr = QFrame()
        hdr.setObjectName("TopHeaderBar")
        hdr.setFixedHeight(42)
        h_lay = QHBoxLayout(hdr)
        h_lay.setContentsMargins(14, 0, 12, 0)
        h_lay.setSpacing(6)

        logo = QLabel()
        logo.setPixmap(draw_easycolab_icon_pixmap(22))
        h_lay.addWidget(logo)

        title = QLabel("EasyColab")
        title.setStyleSheet("font-size: 14px; font-weight: 800; letter-spacing: 0.2px;")
        h_lay.addWidget(title)
        h_lay.addStretch(1)

        self.lang_btn = set_btn_role(QPushButton("EN" if not is_en() else "中文"), "pill")
        self.lang_btn.setFixedHeight(26)
        self.lang_btn.setToolTip(tr("切换界面为英文 (Switch to English)", "Switch UI to Chinese (切换为中文)"))
        self.lang_btn.clicked.connect(self.vm.toggle_language)
        h_lay.addWidget(self.lang_btn)

        self.theme_btn = set_btn_role(QPushButton(tr("深色", "Dark")), "pill")
        self.theme_btn.setFixedHeight(26)
        self.theme_btn.clicked.connect(self.vm.toggle_dark_mode)
        h_lay.addWidget(self.theme_btn)

        guide_pill = set_btn_role(QPushButton(tr("新手引导", "Guide")), "pill")
        guide_pill.setFixedHeight(26)
        guide_pill.clicked.connect(lambda: self.open_onboarding_guide(0))
        h_lay.addWidget(guide_pill)
        vbox.addWidget(hdr)

        # Scrollable cards stack
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body_w = QWidget()
        body_lay = QVBoxLayout(body_w)
        body_lay.setContentsMargins(12, 12, 12, 16)
        body_lay.setSpacing(12)

        body_lay.addWidget(self._build_sidebar_card1_instance())
        body_lay.addWidget(self._build_sidebar_card2_cu_idle())
        body_lay.addWidget(self._build_sidebar_card3_models())
        body_lay.addWidget(self._build_sidebar_card4_params())
        body_lay.addStretch(1)

        scroll.setWidget(body_w)
        vbox.addWidget(scroll, 1)
        return container

    def _rebuild_account_menu(self) -> None:
        """每次弹出时重建换号菜单：已存账号置顶（活跃账号打勾禁用），后接 OAuth 换号与退出。"""
        menu = self.acc_menu
        menu.clear()
        for acc in self.vm.saved_accounts:
            email = str(acc.get("email") or "")
            if not email:
                continue
            is_active = bool(acc.get("active"))
            act = QAction(
                ("✓ " if is_active else "") + mask_email_display(email, self.vm.mask_server_url),
                self,
            )
            if is_active:
                act.setEnabled(False)
            else:
                act.triggered.connect(lambda checked=False, addr=email: self.vm.switch_saved_account(addr))
            menu.addAction(act)
        if menu.actions():
            menu.addSeparator()
        act_switch = QAction(tr("切换 Google 账号...", "Switch Google Account..."), self)
        act_switch.triggered.connect(lambda: self.vm.start_google_oauth_login(True))
        menu.addAction(act_switch)
        act_logout = QAction(tr("退出当前账号", "Sign Out"), self)
        act_logout.triggered.connect(self.vm.logout_google_account)
        menu.addAction(act_logout)

    def _build_sidebar_card1_instance(self) -> SidebarCard:
        self.conn_badge = PillBadge(tr("离线", "Offline"), fg="#64748B", bg="rgba(148,163,184,0.16)")
        card = SidebarCard(tr("Colab 算力实例", "Colab Cloud Runtime"), trailing=self.conn_badge)

        # Row 1: Google Account
        acc_row = QHBoxLayout()
        acc_row.setSpacing(6)
        acc_lbl = QLabel(tr("Google 账号", "Google Account"))
        acc_lbl.setStyleSheet("color: #64748B; font-size: 11.5px;")
        acc_row.addWidget(acc_lbl)

        self.account_val_lbl = QLabel(tr("未绑定", "Not Linked"))
        self.account_val_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.account_val_lbl.setStyleSheet("font-size: 11.5px; font-weight: 600;")
        acc_row.addWidget(self.account_val_lbl, 1)

        self.bind_btn = set_btn_role(QPushButton(tr("绑定", "Bind")), "primary")
        self.bind_btn.clicked.connect(lambda: self.vm.start_google_oauth_login(False))
        acc_row.addWidget(self.bind_btn)

        self.acc_refresh_btn = QPushButton(tr("刷新", "Refresh"))
        self.acc_refresh_btn.clicked.connect(self.vm.refresh_official_colab_account_usage)
        acc_row.addWidget(self.acc_refresh_btn)

        self.acc_menu_btn = QPushButton(tr("换号", "Account"))
        self.acc_menu = QMenu(self)
        self.acc_menu.aboutToShow.connect(self._rebuild_account_menu)
        self.acc_menu_btn.setMenu(self.acc_menu)
        acc_row.addWidget(self.acc_menu_btn)
        card.body.addLayout(acc_row)

        # Row 1a: 账号用量紧凑提示（免费账号无 CU 计量等）
        self.acc_usage_lbl = QLabel("")
        self.acc_usage_lbl.setWordWrap(True)
        self.acc_usage_lbl.setStyleSheet("font-size: 11px; color: #64748B;")
        self.acc_usage_lbl.setVisible(False)
        card.body.addWidget(self.acc_usage_lbl)

        # Row 1b: OAuth Code input (hidden by default)
        self.oauth_box = QWidget()
        oa_lay = QHBoxLayout(self.oauth_box)
        oa_lay.setContentsMargins(0, 0, 0, 0)
        oa_lay.setSpacing(5)
        self.oauth_input = QLineEdit()
        self.oauth_input.setPlaceholderText(tr("粘贴浏览器显示的授权码", "Paste OAuth code from browser"))
        self.oauth_input.returnPressed.connect(self._submit_oauth_code)
        oa_lay.addWidget(self.oauth_input, 1)
        oa_ok = set_btn_role(QPushButton(tr("确认", "Submit")), "primary")
        oa_ok.clicked.connect(self._submit_oauth_code)
        oa_cancel = QPushButton(tr("取消", "Cancel"))
        oa_cancel.clicked.connect(self._cancel_oauth_code)
        oa_lay.addWidget(oa_ok)
        oa_lay.addWidget(oa_cancel)
        self.oauth_box.setVisible(False)
        card.body.addWidget(self.oauth_box)

        # Row 2: GPU Picker + 1-Click Wake
        self.wake_row_widget = QWidget()
        wk_lay = QHBoxLayout(self.wake_row_widget)
        wk_lay.setContentsMargins(0, 0, 0, 0)
        wk_lay.setSpacing(6)
        self.gpu_combo = QComboBox()
        self.gpu_combo.setFixedWidth(134)
        for val, label in GPU_CHOICES:
            self.gpu_combo.addItem(label, val)
        self.gpu_combo.currentIndexChanged.connect(self._on_gpu_combo_changed)
        wk_lay.addWidget(self.gpu_combo)

        self.wake_btn = set_btn_role(QPushButton(tr("唤醒实例并部署", "Wake & Deploy")), "primary")
        self.wake_btn.clicked.connect(self.vm.wake_colab_headless)
        wk_lay.addWidget(self.wake_btn, 1)
        card.body.addWidget(self.wake_row_widget)

        # Row 3: Server URL + Privacy Eye + Connect + ?
        url_row = QHBoxLayout()
        url_row.setSpacing(5)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://xxxx.gradio.live")
        self.url_input.setStyleSheet("font-family: Consolas, monospace; font-size: 11.5px;")
        self.url_input.editingFinished.connect(lambda: self.vm.set_colab_server_url(self.url_input.text()))
        self.url_input.returnPressed.connect(lambda: self._on_connect_clicked())
        url_row.addWidget(self.url_input, 1)

        self.mask_eye_btn = QPushButton(tr("打码", "Mask"))
        self.mask_eye_btn.setFixedWidth(58)
        self.mask_eye_btn.setToolTip(
            tr(
                "录屏/截图防泄密打码开关：隐藏公网隧道域名与账号邮箱",
                "Stream/screenshot privacy mask: hide public tunnel URLs and account email",
            )
        )
        self.mask_eye_btn.clicked.connect(lambda: self.vm.set_mask_server_url(not self.vm.mask_server_url))
        url_row.addWidget(self.mask_eye_btn)

        self.connect_btn = QPushButton(tr("连接", "Connect"))
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        url_row.addWidget(self.connect_btn)

        help_mini = set_btn_role(QPushButton("?"), "ghost")
        help_mini.setFixedWidth(24)
        help_mini.clicked.connect(lambda: self.open_onboarding_guide(0))
        url_row.addWidget(help_mini)
        card.body.addLayout(url_row)

        # Row 4: Hardware Status + Chart Mode Toggle
        hw_hdr = QHBoxLayout()
        hw_hdr.setSpacing(5)
        self.hw_dot = StatusDot("#64748B", size=8)
        hw_hdr.addWidget(self.hw_dot)
        self.hw_gpu_lbl = QLabel(tr("尚未连接云端 GPU", "Cloud GPU Not Connected"))
        self.hw_gpu_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.hw_gpu_lbl.setStyleSheet("font-size: 11px; font-weight: 600;")
        hw_hdr.addWidget(self.hw_gpu_lbl, 1)

        self.drive_pill = PillBadge(tr("云盘", "Drive"), fg="#10B981", bg="rgba(16,185,129,0.16)")
        self.drive_pill.setToolTip(tr("Google Drive 云盘持久化已激活", "Google Drive cloud persistence active"))
        self.drive_pill.setVisible(False)
        hw_hdr.addWidget(self.drive_pill)

        self.chart_mode_btn = set_btn_role(QPushButton(tr("同屏", "Overlay")), "pill")
        self.chart_mode_btn.setToolTip(
            tr("切换「分项折线图」与「三指标同屏对比图」", "Toggle between separate metric charts and combined overlay chart")
        )
        self.chart_mode_btn.clicked.connect(lambda: self.vm.set_combined_chart_mode(not self.vm.combined_chart_mode))
        hw_hdr.addWidget(self.chart_mode_btn)
        card.body.addLayout(hw_hdr)

        # Telemetry Charts Stack (Page 0: 3 separate charts, Page 1: combined chart)
        self.charts_stack = AutoHeightStackedWidget()
        sep_w = QWidget()
        sep_lay = QVBoxLayout(sep_w)
        sep_lay.setContentsMargins(0, 0, 0, 0)
        sep_lay.setSpacing(6)
        self.chart_vram = SingleMetricChartWidget(tr("GPU 显存", "GPU VRAM"), "#5487FA")
        self.chart_ram = SingleMetricChartWidget(tr("系统内存", "System RAM"), "#1AC287")
        self.chart_disk = SingleMetricChartWidget(tr("实例磁盘", "Runtime Disk"), "#0DBAE0")
        sep_lay.addWidget(self.chart_vram)
        sep_lay.addWidget(self.chart_ram)
        sep_lay.addWidget(self.chart_disk)
        self.charts_stack.addWidget(sep_w)

        self.chart_combined = CombinedTelemetryChartWidget()
        self.charts_stack.addWidget(self.chart_combined)
        card.body.addWidget(self.charts_stack)

        # Footer for /dev/shm and Google Drive free space
        self.storage_footer_lbl = QLabel("")
        self.storage_footer_lbl.setStyleSheet("font-size: 10.5px; color: #64748B; font-family: Consolas, monospace;")
        self.storage_footer_lbl.setVisible(False)
        card.body.addWidget(self.storage_footer_lbl)

        return card

    def _build_sidebar_card2_cu_idle(self) -> SidebarCard:
        card = SidebarCard(tr("算力额度与自动休眠", "Compute Units & Auto-Sleep"))

        # 3-Column Metric Bar
        metrics_frame = QFrame()
        metrics_frame.setObjectName("SubtlePanel")
        m_grid = QGridLayout(metrics_frame)
        m_grid.setContentsMargins(10, 8, 10, 8)
        m_grid.setHorizontalSpacing(8)
        m_grid.setVerticalSpacing(2)

        self.cu_title_lbl = QLabel(tr("账户余额", "CU Balance"))
        self.cu_title_lbl.setStyleSheet("font-size: 10.5px; color: #64748B;")
        burn_title = QLabel(tr("扣费速率", "Burn Rate"))
        burn_title.setStyleSheet("font-size: 10.5px; color: #64748B;")
        up_title = QLabel(tr("本次运行", "Session Uptime"))
        up_title.setStyleSheet("font-size: 10.5px; color: #64748B;")

        self.cu_val_lbl = QLabel("-- CU")
        self.cu_val_lbl.setStyleSheet("font-size: 12.5px; font-weight: 700; font-family: Consolas, monospace;")
        self.burn_val_lbl = QLabel("0.00 CU/h")
        self.burn_val_lbl.setStyleSheet("font-size: 12.5px; font-weight: 700; font-family: Consolas, monospace;")
        self.uptime_val_lbl = QLabel("00:00:00")
        self.uptime_val_lbl.setStyleSheet("font-size: 12.5px; font-weight: 700; font-family: Consolas, monospace;")

        m_grid.addWidget(self.cu_title_lbl, 0, 0)
        m_grid.addWidget(burn_title, 0, 1)
        m_grid.addWidget(up_title, 0, 2)
        m_grid.addWidget(self.cu_val_lbl, 1, 0)
        m_grid.addWidget(self.burn_val_lbl, 1, 1)
        m_grid.addWidget(self.uptime_val_lbl, 1, 2)
        card.body.addWidget(metrics_frame)

        # Idle Auto-Release Row
        idle_row = QHBoxLayout()
        idle_row.setSpacing(8)
        idle_lbl = QLabel(tr("闲置自动释放", "Idle Auto-Release"))
        idle_lbl.setStyleSheet("font-size: 11.5px; color: #64748B;")
        idle_row.addWidget(idle_lbl)
        idle_row.addStretch(1)
        self.idle_combo = QComboBox()
        for mins, txt in get_idle_timeout_choices():
            self.idle_combo.addItem(txt, mins)
        self.idle_combo.currentIndexChanged.connect(self._on_idle_combo_changed)
        idle_row.addWidget(self.idle_combo)
        card.body.addLayout(idle_row)

        self.idle_countdown_lbl = QLabel("")
        self.idle_countdown_lbl.setWordWrap(True)
        self.idle_countdown_lbl.setStyleSheet("font-size: 11px; color: #64748B;")
        card.body.addWidget(self.idle_countdown_lbl)

        # Destructive Release All VMs Button
        self.release_all_btn = set_btn_role(QPushButton(tr("释放全部实例", "Release All Runtimes")), "danger")
        self.release_all_btn.clicked.connect(self._confirm_release_all_vms)
        card.body.addWidget(self.release_all_btn)

        return card

    def _build_sidebar_card3_models(self) -> SidebarCard:
        self.model_center_badge = PillBadge("HF Top 6", fg="#60A5FA", bg="rgba(59,130,246,0.16)")
        card = SidebarCard(tr("开源模型中心", "Open Model Center"), trailing=self.model_center_badge)

        # 6 Model Selector Buttons
        self._model_card_btns: Dict[str, QPushButton] = {}
        for spec in OPEN_MODEL_CATALOG:
            btn = QPushButton()
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setMinimumHeight(46)
            btn.clicked.connect(lambda _c=False, mid=spec.id: self.vm.select_model(mid))
            card.body.addWidget(btn)
            self._model_card_btns[spec.id] = btn

        # Co-residency strip (visible when >= 1 model loaded in VRAM)
        self.coresident_box = QFrame()
        self.coresident_box.setObjectName("SubtlePanel")
        self.coresident_lay = QVBoxLayout(self.coresident_box)
        self.coresident_lay.setContentsMargins(8, 6, 8, 6)
        self.coresident_lay.setSpacing(4)
        self.coresident_box.setVisible(False)
        card.body.addWidget(self.coresident_box)

        # Selected Model Control Stack (Page 0: Standard Model Weights/VRAM, Page 1: ComfyUI mini controls)
        self.model_ctrl_stack = AutoHeightStackedWidget()

        # Page 0: Standard Model Weight & VRAM Manager
        std_w = QFrame()
        std_w.setObjectName("SubtlePanel")
        std_lay = QVBoxLayout(std_w)
        std_lay.setContentsMargins(10, 8, 10, 9)
        std_lay.setSpacing(6)

        repo_row = QHBoxLayout()
        repo_row.addWidget(QLabel(tr("HF 仓库", "HF Repo")))
        self.repo_val_lbl = QLabel()
        self.repo_val_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color: #94A3B8;")
        repo_row.addWidget(self.repo_val_lbl, 1, Qt.AlignmentFlag.AlignRight)
        std_lay.addLayout(repo_row)

        wt_row = QHBoxLayout()
        wt_row.addWidget(QLabel(tr("磁盘权重", "Disk Weights")))
        self.weight_status_lbl = QLabel()
        self.weight_status_lbl.setStyleSheet("font-size: 11.5px; font-weight: 600;")
        wt_row.addWidget(self.weight_status_lbl, 1, Qt.AlignmentFlag.AlignRight)
        std_lay.addLayout(wt_row)

        # Collapsible shard list
        self.shards_box = CollapsibleBox(tr("查看权重分片 (0)", "View Weight Shards (0)"), expanded=False)
        self.shards_scroll = QScrollArea()
        self.shards_scroll.setFixedHeight(112)
        self.shards_scroll.setWidgetResizable(True)
        self.shards_list_w = QWidget()
        self.shards_list_lay = QVBoxLayout(self.shards_list_w)
        self.shards_list_lay.setContentsMargins(2, 2, 2, 2)
        self.shards_list_lay.setSpacing(2)
        self.shards_scroll.setWidget(self.shards_list_w)
        self.shards_box.content_layout.addWidget(self.shards_scroll)
        std_lay.addWidget(self.shards_box)

        # Optional HF Token for gated downloads
        hf_row = QHBoxLayout()
        hf_lbl = QLabel("HF Token")
        hf_lbl.setStyleSheet("font-size: 11px; color: #94A3B8;")
        hf_row.addWidget(hf_lbl)
        self.hf_token_input = QLineEdit(self.vm.hf_token)
        self.hf_token_input.setPlaceholderText(tr("可选 (下载受限仓库时填写)", "Optional (for gated HF repos)"))
        self.hf_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_input.editingFinished.connect(self._on_hf_token_edited)
        hf_row.addWidget(self.hf_token_input, 1)
        std_lay.addLayout(hf_row)

        # KV context mismatch hint for Qwen GGUF
        self.kv_mismatch_lbl = QLabel("")
        self.kv_mismatch_lbl.setWordWrap(True)
        self.kv_mismatch_lbl.setStyleSheet("font-size: 11px; color: #F59E0B;")
        self.kv_mismatch_lbl.setVisible(False)
        std_lay.addWidget(self.kv_mismatch_lbl)

        # Offline hint banner
        self.offline_banner = QWidget()
        ob_lay = QHBoxLayout(self.offline_banner)
        ob_lay.setContentsMargins(0, 0, 0, 0)
        ob_lay.setSpacing(6)
        ob_txt = QLabel(tr("尚未连接云端 GPU", "Cloud GPU Not Connected"))
        ob_txt.setStyleSheet("color: #F59E0B; font-size: 11px;")
        ob_lay.addWidget(ob_txt, 1)
        self.offline_quick_btn = QPushButton(tr("一键唤醒", "1-Click Wake"))
        self.offline_quick_btn.clicked.connect(self._on_offline_quick_action)
        ob_lay.addWidget(self.offline_quick_btn)
        std_lay.addWidget(self.offline_banner)

        # Progress bar when downloading/loading weights
        self.weight_prog_bar = QProgressBar()
        self.weight_prog_bar.setRange(0, 100)
        self.weight_prog_bar.setVisible(False)
        std_lay.addWidget(self.weight_prog_bar)

        # Download + Load VRAM + Eject buttons
        act_row = QHBoxLayout()
        act_row.setSpacing(6)
        self.dl_weights_btn = QPushButton(tr("下载权重", "Download"))
        self.dl_weights_btn.clicked.connect(self.vm.download_model_weights)
        act_row.addWidget(self.dl_weights_btn)

        self.load_vram_btn = set_btn_role(QPushButton(tr("加载到显存", "Load to VRAM")), "primary")
        self.load_vram_btn.clicked.connect(self.vm.load_model_to_vram)
        act_row.addWidget(self.load_vram_btn, 1)

        self.eject_vram_btn = set_btn_role(QPushButton(tr("卸载", "Unload")), "danger")
        self.eject_vram_btn.setToolTip(tr("从 GPU 显存卸载当前模型", "Unload current model from GPU VRAM"))
        self.eject_vram_btn.clicked.connect(lambda: self.vm.unload_model_from_vram(self.vm.selected_model_id))
        act_row.addWidget(self.eject_vram_btn)
        std_lay.addLayout(act_row)
        self.model_ctrl_stack.addWidget(std_w)

        # Page 1: ComfyUI Sidebar Mini Box
        cfy_w = QFrame()
        cfy_w.setObjectName("SubtlePanel")
        cfy_lay = QVBoxLayout(cfy_w)
        cfy_lay.setContentsMargins(10, 8, 10, 9)
        cfy_lay.setSpacing(6)
        st_row = QHBoxLayout()
        st_row.addWidget(QLabel(tr("画布状态", "Canvas Status")))
        self.cfy_side_status_lbl = QLabel(tr("未启动", "Stopped"))
        self.cfy_side_status_lbl.setStyleSheet("font-weight: 700; font-size: 11.5px;")
        st_row.addWidget(self.cfy_side_status_lbl, 1, Qt.AlignmentFlag.AlignRight)
        cfy_lay.addLayout(st_row)

        self.cfy_side_prog = QProgressBar()
        self.cfy_side_prog.setRange(0, 100)
        self.cfy_side_prog.setVisible(False)
        cfy_lay.addWidget(self.cfy_side_prog)

        cfy_btns = QHBoxLayout()
        cfy_btns.setSpacing(6)
        self.cfy_side_start_btn = set_btn_role(QPushButton(tr("启动 ComfyUI 画布", "Launch ComfyUI")), "primary")
        self.cfy_side_start_btn.clicked.connect(self._on_comfy_side_primary)
        cfy_btns.addWidget(self.cfy_side_start_btn, 1)
        self.cfy_side_stop_btn = set_btn_role(QPushButton(tr("停止", "Stop")), "danger")
        self.cfy_side_stop_btn.clicked.connect(self.vm.stop_comfyui)
        cfy_btns.addWidget(self.cfy_side_stop_btn)
        cfy_lay.addLayout(cfy_btns)
        self.model_ctrl_stack.addWidget(cfy_w)

        card.body.addWidget(self.model_ctrl_stack)
        return card

    def _build_sidebar_card4_params(self) -> SidebarCard:
        self.params_card = SidebarCard(tr("规格与采样参数", "Resolution & Sampling"))
        self.params_stack = AutoHeightStackedWidget()

        # Page 0: Video / VideoAudio Params
        vid_p = QWidget()
        vp_lay = QVBoxLayout(vid_p)
        vp_lay.setContentsMargins(0, 0, 0, 0)
        vp_lay.setSpacing(8)
        vp_lay.addWidget(QLabel(tr("视频分辨率", "Video Resolution")))
        self.vid_res_combo = QComboBox()
        for r in VIDEO_RESOLUTIONS:
            self.vid_res_combo.addItem(r.label, r.id)
        vr_idx = self.vid_res_combo.findData(self.vm.selected_video_res.id)
        if vr_idx >= 0:
            self.vid_res_combo.setCurrentIndex(vr_idx)
        self.vid_res_combo.currentIndexChanged.connect(self._on_vid_res_changed)
        vp_lay.addWidget(self.vid_res_combo)

        init_fr = snap_video_frames(self.vm.num_frames)
        fr_hdr = QHBoxLayout()
        fr_hdr.addWidget(QLabel(tr("视频总帧数 (4~15 秒)", "Total Frames (4-15s)")))
        self.vid_frames_lbl = QLabel(
            tr(f"{init_fr} 帧 · 约 {init_fr / 24.0:.1f} 秒", f"{init_fr} frames · ~{init_fr / 24.0:.1f}s")
        )
        self.vid_frames_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; color: #60A5FA;")
        fr_hdr.addWidget(self.vid_frames_lbl, 0, Qt.AlignmentFlag.AlignRight)
        vp_lay.addLayout(fr_hdr)

        self.vid_frames_slider = QSlider(Qt.Orientation.Horizontal)
        self.vid_frames_slider.setRange(90, 362)
        self.vid_frames_slider.setValue(init_fr)
        self.vid_frames_slider.valueChanged.connect(self._on_vid_frames_changed)
        vp_lay.addWidget(self.vid_frames_slider)

        dur_chips = QHBoxLayout()
        dur_chips.setSpacing(4)
        for fr_val, d_txt in VIDEO_DURATION_PRESETS:
            b = QPushButton(d_txt)
            b.setFixedHeight(24)
            b.clicked.connect(lambda _c=False, f=fr_val: self.vid_frames_slider.setValue(f))
            dur_chips.addWidget(b)
        vp_lay.addLayout(dur_chips)

        vs_hdr = QHBoxLayout()
        vs_hdr.addWidget(QLabel(tr("采样步数", "Sampling Steps")))
        self.vid_steps_lbl = QLabel(tr(f"{self.vm.inference_steps} 步", f"{self.vm.inference_steps} steps"))
        self.vid_steps_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        vs_hdr.addWidget(self.vid_steps_lbl, 0, Qt.AlignmentFlag.AlignRight)
        vp_lay.addLayout(vs_hdr)
        self.vid_steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.vid_steps_slider.setRange(10, 60)
        self.vid_steps_slider.setSingleStep(5)
        self.vid_steps_slider.setValue(max(10, min(60, self.vm.inference_steps)))
        self.vid_steps_slider.valueChanged.connect(self._on_vid_steps_changed)
        vp_lay.addWidget(self.vid_steps_slider)

        vp_lay.addLayout(self._build_seed_and_batch_controls(is_video=True))
        self.params_stack.addWidget(vid_p)

        # Page 1: Image Params
        img_p = QWidget()
        ip_lay = QVBoxLayout(img_p)
        ip_lay.setContentsMargins(0, 0, 0, 0)
        ip_lay.setSpacing(8)
        ip_lay.addWidget(QLabel(tr("输出画幅", "Aspect Ratio & Size")))
        self.img_res_combo = QComboBox()
        for r in IMAGE_RESOLUTIONS:
            self.img_res_combo.addItem(r.label, r.id)
        ir_idx = self.img_res_combo.findData(self.vm.selected_image_res.id)
        if ir_idx >= 0:
            self.img_res_combo.setCurrentIndex(ir_idx)
        self.img_res_combo.currentIndexChanged.connect(self._on_img_res_changed)
        ip_lay.addWidget(self.img_res_combo)

        is_hdr = QHBoxLayout()
        is_hdr.addWidget(QLabel(tr("采样步数", "Sampling Steps")))
        self.img_steps_lbl = QLabel(tr(f"{self.vm.inference_steps} 步", f"{self.vm.inference_steps} steps"))
        self.img_steps_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        is_hdr.addWidget(self.img_steps_lbl, 0, Qt.AlignmentFlag.AlignRight)
        ip_lay.addLayout(is_hdr)
        self.img_steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.img_steps_slider.setRange(4, 50)
        self.img_steps_slider.setValue(max(4, min(50, self.vm.inference_steps)))
        self.img_steps_slider.valueChanged.connect(self._on_img_steps_changed)
        ip_lay.addWidget(self.img_steps_slider)

        self.denoise_box = QWidget()
        dn_lay = QVBoxLayout(self.denoise_box)
        dn_lay.setContentsMargins(0, 0, 0, 0)
        dn_lay.setSpacing(4)
        dn_hdr = QHBoxLayout()
        dn_hdr.addWidget(QLabel(tr("重绘幅度 (Denoising)", "Denoising Strength")))
        self.denoise_lbl = QLabel(f"{self.vm.denoising_strength:.2f}")
        self.denoise_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        dn_hdr.addWidget(self.denoise_lbl, 0, Qt.AlignmentFlag.AlignRight)
        dn_lay.addLayout(dn_hdr)
        self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_slider.setRange(20, 100)
        self.denoise_slider.setValue(int(round(self.vm.denoising_strength * 100)))
        self.denoise_slider.valueChanged.connect(self._on_denoise_changed)
        dn_lay.addWidget(self.denoise_slider)
        ip_lay.addWidget(self.denoise_box)

        ip_lay.addLayout(self._build_seed_and_batch_controls(is_video=False))
        self.params_stack.addWidget(img_p)

        # Page 2: LLM / VLM Params
        llm_p = QWidget()
        lp_lay = QVBoxLayout(llm_p)
        lp_lay.setContentsMargins(0, 0, 0, 0)
        lp_lay.setSpacing(8)
        self.thinking_chk = QCheckBox(tr("启用深度思考链 (Thinking Mode)", "Enable Thinking Mode (CoT)"))
        self.thinking_chk.setChecked(self.vm.enable_thinking)
        self.thinking_chk.toggled.connect(self._on_thinking_toggled)
        lp_lay.addWidget(self.thinking_chk)

        self.effort_box = QWidget()
        ef_lay = QVBoxLayout(self.effort_box)
        ef_lay.setContentsMargins(0, 0, 0, 0)
        ef_lay.setSpacing(4)
        ef_lay.addWidget(QLabel(tr("思考深度 (Reasoning Effort)", "Reasoning Effort")))
        ef_btns = QHBoxLayout()
        ef_btns.setSpacing(4)
        self._effort_btns: Dict[str, QPushButton] = {}
        for ef_id, ef_lbl in (
            ("low", tr("精简", "Low")),
            ("medium", tr("均衡", "Med")),
            ("high", tr("深入", "High")),
            ("xhigh", tr("极深", "XHigh")),
        ):
            b = QPushButton(ef_lbl)
            b.setFixedHeight(26)
            b.clicked.connect(lambda _c=False, eid=ef_id: self.vm.set_reasoning_effort(eid))
            ef_btns.addWidget(b)
            self._effort_btns[ef_id] = b
        ef_lay.addLayout(ef_btns)
        lp_lay.addWidget(self.effort_box)

        ctx_hdr = QHBoxLayout()
        ctx_hdr.addWidget(QLabel(tr("上下文窗口 (最高 256K)", "Context Window (up to 256K)")))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(4, 256)
        self.ctx_spin.setSingleStep(4)
        self.ctx_spin.setSuffix(" K")
        self.ctx_spin.setValue(self.vm.context_length_k)
        self.ctx_spin.valueChanged.connect(lambda v: self._on_ctx_changed(v, sync_remote=True))
        ctx_hdr.addWidget(self.ctx_spin)
        lp_lay.addLayout(ctx_hdr)

        self.ctx_slider = QSlider(Qt.Orientation.Horizontal)
        self.ctx_slider.setRange(4, 256)
        self.ctx_slider.setSingleStep(4)
        self.ctx_slider.setValue(self.vm.context_length_k)
        self.ctx_slider.sliderReleased.connect(lambda: self.vm.set_context_length_k(self.ctx_slider.value(), True))
        self.ctx_slider.valueChanged.connect(lambda v: self._on_ctx_slider_preview(v))
        lp_lay.addWidget(self.ctx_slider)

        ctx_chips = QHBoxLayout()
        ctx_chips.setSpacing(4)
        for kv in (8, 16, 32, 64, 128, 256):
            cb = QPushButton(f"{kv}K")
            cb.setFixedHeight(23)
            cb.clicked.connect(lambda _c=False, k=kv: self.vm.set_context_length_k(k, True))
            ctx_chips.addWidget(cb)
        lp_lay.addLayout(ctx_chips)

        tok_hdr = QHBoxLayout()
        tok_hdr.addWidget(QLabel(tr("最大输出长度 (含思考链)", "Max Output Tokens (incl. CoT)")))
        self.max_tok_lbl = QLabel(str(self.vm.max_new_tokens))
        self.max_tok_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        tok_hdr.addWidget(self.max_tok_lbl, 0, Qt.AlignmentFlag.AlignRight)
        lp_lay.addLayout(tok_hdr)
        self.max_tok_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_tok_slider.setRange(1024, 65536)
        self.max_tok_slider.setSingleStep(512)
        self.max_tok_slider.setValue(self.vm.max_new_tokens)
        self.max_tok_slider.valueChanged.connect(self._on_max_tokens_changed)
        lp_lay.addWidget(self.max_tok_slider)

        tmp_hdr = QHBoxLayout()
        tmp_hdr.addWidget(QLabel(tr("采样温度 (Temperature)", "Temperature")))
        self.temp_lbl = QLabel(f"{self.vm.temperature:.2f} · Top-P {self.vm.top_p:.2f} · Top-K 20")
        self.temp_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 10.5px; color: #94A3B8;")
        tmp_hdr.addWidget(self.temp_lbl, 0, Qt.AlignmentFlag.AlignRight)
        lp_lay.addLayout(tmp_hdr)
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(10, 150)
        self.temp_slider.setValue(int(round(self.vm.temperature * 100)))
        self.temp_slider.valueChanged.connect(self._on_temp_changed)
        lp_lay.addWidget(self.temp_slider)
        self.params_stack.addWidget(llm_p)

        # Page 3: ComfyUI Sidebar Persistence & Queue Params
        cfy_p = QWidget()
        cp_lay = QVBoxLayout(cfy_p)
        cp_lay.setContentsMargins(0, 0, 0, 0)
        cp_lay.setSpacing(8)
        self.cfy_persist_lbl = QLabel(
            tr("架构：Google Drive 自动软链持久化", "Storage: Google Drive Auto-Symlink Persistence")
        )
        self.cfy_persist_lbl.setWordWrap(True)
        self.cfy_persist_lbl.setStyleSheet("font-size: 11.5px; color: #94A3B8;")
        cp_lay.addWidget(self.cfy_persist_lbl)

        self.cfy_counts_lbl = QLabel(tr("已存节点 / 权重：0 个节点 · 0 个权重", "Saved Nodes / Weights: 0 nodes · 0 weights"))
        self.cfy_counts_lbl.setStyleSheet("font-size: 11.5px;")
        cp_lay.addWidget(self.cfy_counts_lbl)

        self.cfy_queue_lbl = QLabel(tr("画布执行队列：运行 0 · 排队 0", "Canvas Queue: 0 running · 0 queued"))
        self.cfy_queue_lbl.setStyleSheet("font-size: 11.5px; font-family: Consolas, monospace;")
        cp_lay.addWidget(self.cfy_queue_lbl)

        self.cfy_unload_chk = QCheckBox(tr("启动前释放其它常驻模型显存", "Free co-resident models VRAM before start"))
        self.cfy_unload_chk.setChecked(self.vm.comfy_unload_others_on_start)
        self.cfy_unload_chk.toggled.connect(lambda v: setattr(self.vm, "comfy_unload_others_on_start", v))
        cp_lay.addWidget(self.cfy_unload_chk)

        cfy_act = QHBoxLayout()
        cfy_act.setSpacing(6)
        sync_b = QPushButton(tr("立即同步云盘", "Sync Drive Now"))
        sync_b.clicked.connect(self.vm.sync_comfy_drive)
        free_b = QPushButton(tr("清缓存显存", "Free VRAM Cache"))
        free_b.clicked.connect(self.vm.free_comfy_vram)
        cfy_act.addWidget(sync_b)
        cfy_act.addWidget(free_b)
        cp_lay.addLayout(cfy_act)
        self.params_stack.addWidget(cfy_p)

        self.params_card.body.addWidget(self.params_stack)
        return self.params_card

    def _build_seed_and_batch_controls(self, is_video: bool) -> QVBoxLayout:
        lay = QVBoxLayout()
        lay.setSpacing(6)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel(tr("随机种子", "Seed")))
        spin = QSpinBox()
        spin.setRange(0, 2147483647)
        spin.setValue(self.vm.seed)
        spin.valueChanged.connect(lambda v: setattr(self.vm, "seed", int(v)))
        seed_row.addWidget(spin, 1)
        chk = QCheckBox(tr("随机", "Random"))
        chk.setChecked(self.vm.randomize_seed)
        chk.toggled.connect(lambda v: self._sync_rand_seed(v))
        seed_row.addWidget(chk)
        lay.addLayout(seed_row)

        lay.addWidget(QLabel(tr("批量连抽", "Batch Count")))
        b_row = QHBoxLayout()
        b_row.setSpacing(4)
        btns: Dict[int, QPushButton] = {}
        for n in (1, 2, 3, 4):
            if is_en():
                btn_lbl = f"{n}x"
            else:
                unit = "条" if is_video else "张"
                btn_lbl = f"{n} {unit}"
            b = QPushButton(btn_lbl)
            b.setFixedHeight(26)
            b.clicked.connect(lambda _c=False, cnt=n: self._set_batch_count(cnt))
            b_row.addWidget(b)
            btns[n] = b
        lay.addLayout(b_row)

        if is_video:
            self.vid_seed_spin = spin
            self.vid_rand_chk = chk
            self._vid_batch_btns = btns
        else:
            self.img_seed_spin = spin
            self.img_rand_chk = chk
            self._img_batch_btns = btns
        return lay

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. CENTER STUDIO WORKSPACE (Top Bar + Mode Switcher + 4 Studios)
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_center_workspace(self) -> QWidget:
        container = QWidget()
        container.setObjectName("CenterWorkspaceContainer")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Top 42px Workspace Bar
        top_bar = QFrame()
        top_bar.setObjectName("WorkspaceTopBar")
        top_bar.setFixedHeight(42)
        tb_lay = QHBoxLayout(top_bar)
        tb_lay.setContentsMargins(16, 0, 14, 0)
        tb_lay.setSpacing(8)

        self.ws_top_dot = StatusDot("#64748B", size=8)
        tb_lay.addWidget(self.ws_top_dot)
        self.ws_top_summary = QLabel(tr("开源多模型云端工作台", "Open Multi-Model Cloud AI Studio"))
        self.ws_top_summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.ws_top_summary.setStyleSheet("font-size: 12px; font-weight: 600;")
        tb_lay.addWidget(self.ws_top_summary, 1)

        self.preset_btn = QPushButton(tr("示例模板", "Presets"))
        self.preset_menu = QMenu(self)
        self.preset_btn.setMenu(self.preset_menu)
        tb_lay.addWidget(self.preset_btn)

        open_out_btn = QPushButton(tr("本地输出目录", "Output Folder"))
        open_out_btn.clicked.connect(lambda: open_path_in_os(get_local_save_dir()))
        tb_lay.addWidget(open_out_btn)

        guide_top_btn = QPushButton(tr("使用指南", "User Guide"))
        guide_top_btn.clicked.connect(lambda: self.open_onboarding_guide(0))
        tb_lay.addWidget(guide_top_btn)

        self.toggle_log_btn = set_btn_role(QPushButton(tr("日志栏", "Logs")), "ghost")
        self.toggle_log_btn.setToolTip(tr("折叠 / 展开右侧实时运行日志面板", "Collapse / expand right live run log panel"))
        self.toggle_log_btn.clicked.connect(self._toggle_right_log_panel)
        tb_lay.addWidget(self.toggle_log_btn)
        vbox.addWidget(top_bar)

        # Scrollable Studio Canvas
        self.ws_scroll = QScrollArea()
        self.ws_scroll.setWidgetResizable(True)
        self.ws_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        canvas = QWidget()
        self.ws_inner_vbox = QVBoxLayout(canvas)
        self.ws_inner_vbox.setContentsMargins(20, 16, 20, 22)
        self.ws_inner_vbox.setSpacing(14)

        # Universal Top Mode Switcher Card
        self.ws_inner_vbox.addWidget(self._build_mode_switcher_card())

        # Modality Studio Stack:
        # 0 = MiniMax-H3 Video/Audio, 1 = ComfyUI Studio, 2 = Qwen LLM/VLM/API, 3 = Image Studio
        self.studio_stack = AutoHeightStackedWidget()
        self.studio_stack.addWidget(self._build_video_audio_studio())
        self.studio_stack.addWidget(self._build_comfyui_studio())
        self.studio_stack.addWidget(self._build_qwen_llm_studio())
        self.studio_stack.addWidget(self._build_image_studio())
        self.ws_inner_vbox.addWidget(self.studio_stack)

        # Shared Media Preview + History Bottom Split (shown for Video, Image & ComfyUI)
        self.shared_media_split = self._build_shared_media_bottom_split()
        self.ws_inner_vbox.addWidget(self.shared_media_split)
        self.ws_inner_vbox.addStretch(1)

        self.ws_scroll.setWidget(canvas)
        vbox.addWidget(self.ws_scroll, 1)
        return container

    def _build_mode_switcher_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("WorkspaceCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(8)
        self.ms_title_lbl = QLabel()
        self.ms_title_lbl.setStyleSheet("font-size: 16px; font-weight: 800;")
        top.addWidget(self.ms_title_lbl)

        self.ms_repo_pill = PillBadge("", fg="#64748B", bg="rgba(148,163,184,0.16)", mono=True)
        top.addWidget(self.ms_repo_pill)
        top.addStretch(1)

        self.ms_coresident_pill = PillBadge("", fg="#2563EB", bg="rgba(59,130,246,0.14)")
        self.ms_coresident_pill.setVisible(False)
        top.addWidget(self.ms_coresident_pill)

        self.ms_vram_pill = PillBadge(tr("待加载至显存", "Not Loaded in VRAM"), fg="#D97706", bg="rgba(245,158,11,0.14)")
        top.addWidget(self.ms_vram_pill)
        lay.addLayout(top)

        self.ms_summary_lbl = QLabel()
        self.ms_summary_lbl.setWordWrap(True)
        self.ms_summary_lbl.setStyleSheet("font-size: 12px; color: #64748B;")
        lay.addWidget(self.ms_summary_lbl)

        self.mode_buttons_row = QHBoxLayout()
        self.mode_buttons_row.setSpacing(10)
        lay.addLayout(self.mode_buttons_row)
        return card

    # ─── Studio A: MiniMax-H3 Video + Audio Studio ───────────────────────────

    def _build_video_audio_studio(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        # Reference First/Last Frame Drop Wells (fl2va only)
        self.video_frames_card = WorkspaceCard(
            tr("首尾关键帧参考图", "Start & End Keyframe References"),
            tr(
                "上传起始首帧或结束尾帧（至少 1 张），精准控制画面构图与镜头过渡",
                "Upload a start frame and/or end frame (at least 1) for precise composition & camera transition",
            ),
        )
        wells_row = QHBoxLayout()
        wells_row.setSpacing(12)
        self.video_first_well = FrameDropWell(
            tr("起始首帧 (Index 0)", "Start Frame (Index 0)"),
            tr("拖拽或点击选择首帧图像", "Drop or click to select start frame"),
        )
        self.video_last_well = FrameDropWell(
            tr("结束尾帧 (可选 · Index -1)", "End Frame (Optional · Index -1)"),
            tr("拖拽或点击选择尾帧图像", "Drop or click to select end frame"),
        )
        wells_row.addWidget(self.video_first_well, 1)
        wells_row.addWidget(self.video_last_well, 1)
        self.video_frames_card.body.addLayout(wells_row)
        lay.addWidget(self.video_frames_card)

        # Multimodal Prompt Composer
        p_card = WorkspaceCard(
            tr("镜头与多模态提示词", "Camera & Multimodal Audio-Video Prompt"),
            tr(
                "描述画面主体、镜头运动、光影风格，并可同步指定环境音效/台词对白与背景配乐",
                "Describe visual subjects, camera motion, and lighting, plus synced soundscape/dialogue & background score",
            ),
        )
        p_card.body.addWidget(
            QLabel(tr("视觉画面与运镜描述 (Visual & Camera Prompt)", "Visual & Camera Motion Prompt"))
        )
        self.vid_visual_edit = QPlainTextEdit(self.vm.visual_prompt)
        self.vid_visual_edit.setPlaceholderText(
            tr("输入主体、构图、运镜与光影风格描述...", "Describe subject, framing, camera movement, and lighting...")
        )
        self.vid_visual_edit.setFixedHeight(92)
        self.vid_visual_edit.textChanged.connect(self._on_vid_visual_edited)
        p_card.body.addWidget(self.vid_visual_edit)

        aud_row = QHBoxLayout()
        aud_row.setSpacing(12)

        sc_col = QVBoxLayout()
        sc_col.setSpacing(4)
        sc_col.addWidget(QLabel(tr("环境音效与台词对白（可选）", "Soundscape & Dialogue (Optional)")))
        self.vid_sound_edit = QPlainTextEdit(self.vm.soundscape_prompt)
        self.vid_sound_edit.setPlaceholderText(
            tr(
                "如：列车驶过铁轨撞击声、风吹树叶沙沙声、人物轻声感叹...",
                "e.g. Rhythmic train clatter over steel tracks, rustling leaves, soft spoken dialogue...",
            )
        )
        self.vid_sound_edit.setFixedHeight(64)
        self.vid_sound_edit.textChanged.connect(
            lambda: setattr(self.vm, "soundscape_prompt", self.vid_sound_edit.toPlainText())
        )
        sc_col.addWidget(self.vid_sound_edit)
        aud_row.addLayout(sc_col, 1)

        mu_col = QVBoxLayout()
        mu_col.setSpacing(4)
        mu_col.addWidget(QLabel(tr("背景音乐风格（可选）", "Background Music Style (Optional)")))
        self.vid_music_edit = QPlainTextEdit(self.vm.music_prompt)
        self.vid_music_edit.setPlaceholderText(
            tr(
                "如：温柔怀旧的独奏钢琴与温暖木吉他弦乐...",
                "e.g. Gentle nostalgic solo piano with warm acoustic guitar and ambient strings...",
            )
        )
        self.vid_music_edit.setFixedHeight(64)
        self.vid_music_edit.textChanged.connect(
            lambda: setattr(self.vm, "music_prompt", self.vid_music_edit.toPlainText())
        )
        mu_col.addWidget(self.vid_music_edit)
        aud_row.addLayout(mu_col, 1)

        p_card.body.addLayout(aud_row)
        p_card.body.addLayout(self._build_generate_action_bar("vid"))
        lay.addWidget(p_card)
        return page

    # ─── Studio B: ComfyUI 4-Card Node Studio ────────────────────────────────

    def _build_comfyui_studio(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        # Card 1: Fullscreen Node Canvas
        self.cfy_status_badge = PillBadge(tr("未启动", "Stopped"), fg="#64748B", bg="rgba(148,163,184,0.16)")
        c1 = WorkspaceCard(
            tr("云端 ComfyUI 全屏节点画布", "Cloud ComfyUI Fullscreen Node Canvas"),
            tr(
                "一键部署 ComfyUI + Manager 并自动开通 Cloudflare HTTPS 加密隧道，复用客户端保活与自动休眠",
                "1-click deploy ComfyUI + Manager over an encrypted Cloudflare HTTPS tunnel with desktop keepalive & auto-sleep",
            ),
            trailing=self.cfy_status_badge,
        )
        self.cfy_launch_btn = set_btn_role(
            QPushButton(
                tr(
                    "一键拉起云端 ComfyUI 并在浏览器打开全屏画布",
                    "1-Click Launch Cloud ComfyUI & Open Fullscreen Canvas in Browser",
                )
            ),
            "primary",
        )
        self.cfy_launch_btn.setFixedHeight(38)
        self.cfy_launch_btn.clicked.connect(lambda: self.vm.start_comfyui(force_restart=False, auto_open_browser=True))
        c1.body.addWidget(self.cfy_launch_btn)

        self.cfy_running_bar = QWidget()
        rb_lay = QHBoxLayout(self.cfy_running_bar)
        rb_lay.setContentsMargins(0, 0, 0, 0)
        rb_lay.setSpacing(8)
        open_cfy_btn = set_btn_role(
            QPushButton(tr("在浏览器打开全屏节点画布", "Open Fullscreen Canvas in Browser")), "success"
        )
        open_cfy_btn.setFixedHeight(34)
        open_cfy_btn.clicked.connect(self._open_comfy_browser)
        copy_cfy_btn = QPushButton(tr("复制画布链接", "Copy Canvas URL"))
        copy_cfy_btn.clicked.connect(self._copy_comfy_url)
        restart_cfy_btn = QPushButton(tr("重启生效", "Restart ComfyUI"))
        restart_cfy_btn.clicked.connect(lambda: self.vm.start_comfyui(force_restart=True, auto_open_browser=False))
        rb_lay.addWidget(open_cfy_btn, 1)
        rb_lay.addWidget(copy_cfy_btn)
        rb_lay.addWidget(restart_cfy_btn)
        c1.body.addWidget(self.cfy_running_bar)

        self.cfy_main_prog = QProgressBar()
        self.cfy_main_prog.setRange(0, 100)
        self.cfy_main_prog.setVisible(False)
        c1.body.addWidget(self.cfy_main_prog)

        self.cfy_tunnel_lbl = QLabel(tr("云端 ComfyUI 尚未启动", "Cloud ComfyUI not started"))
        self.cfy_tunnel_lbl.setWordWrap(True)
        self.cfy_tunnel_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 11.5px; color: #94A3B8;")
        c1.body.addWidget(self.cfy_tunnel_lbl)
        lay.addWidget(c1)

        # Card 2: 4 Preset Workflow Packs
        self.cfy_packs_badge = PillBadge(
            tr("已就绪 0 / 4 套", "0 / 4 Packs Ready"), fg="#2563EB", bg="rgba(59,130,246,0.14)"
        )
        c2 = WorkspaceCard(
            tr(
                "预置图生视频 / 首尾帧运镜工作流包（一键配齐模型与中文画布）",
                "Preset I2V & First/Last-Frame Workflow Packs (1-Click Models + Guided Templates)",
            ),
            tr(
                "零基础免找权重、免配节点：一键高速拉取全套模型并自动生成中文指引工作流模板",
                "Zero manual setup: 1-click high-speed download of official weights + ready-to-run guided workflow templates",
            ),
            trailing=self.cfy_packs_badge,
        )
        steps_row = QHBoxLayout()
        steps_row.setSpacing(8)
        for st_txt in (
            tr("1. 点击下方「一键配齐模型与工作流」", "1. Click '1-Click Install Models & Workflows' below"),
            tr("2. 在画布最左侧点「工作流」选中文模板", "2. Open 'Workflows' sidebar in ComfyUI & pick a template"),
            tr("3. 上传 1~2 张图片点右上角「运行」出片", "3. Upload 1-2 images & click 'Run' to generate"),
        ):
            chip = QFrame()
            chip.setObjectName("SubtlePanel")
            chip.setMinimumHeight(42)
            cl = QHBoxLayout(chip)
            cl.setContentsMargins(10, 6, 10, 6)
            lbl = QLabel(st_txt)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 11px; color: #64748B;")
            cl.addWidget(lbl)
            steps_row.addWidget(chip, 1)
        c2.body.addLayout(steps_row)

        self.cfy_packs_widget = QWidget()
        self.cfy_packs_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.cfy_packs_container = QVBoxLayout(self.cfy_packs_widget)
        self.cfy_packs_container.setContentsMargins(0, 0, 0, 0)
        self.cfy_packs_container.setSpacing(8)
        for init_pack in self.vm.comfy_preset_packs:
            self.cfy_packs_container.addWidget(self._build_comfy_pack_row(init_pack))
        c2.body.addWidget(self.cfy_packs_widget)
        lay.addWidget(c2)

        # Card 3: Google Drive Symlink & Persistence
        sync_drv_btn = QPushButton(tr("立即同步至云盘", "Sync to Drive Now"))
        sync_drv_btn.clicked.connect(self.vm.sync_comfy_drive)
        c3 = WorkspaceCard(
            tr("Google Drive 自动软链与跨实例持久化", "Google Drive Auto-Symlink & Cross-Runtime Persistence"),
            tr(
                "云端路径: /content/drive/MyDrive/EasyColab_ComfyUI（销毁换卡后自动挂载复用）",
                "Cloud path: /content/drive/MyDrive/EasyColab_ComfyUI (auto-reused across fresh GPU runtimes)",
            ),
            trailing=sync_drv_btn,
        )
        tiles_row = QHBoxLayout()
        tiles_row.setSpacing(10)
        self.cfy_tile_models = self._make_metric_tile(
            tr("models/* 模型权重库", "models/* Weight Library"), tr("0 个文件", "0 files")
        )
        self.cfy_tile_nodes = self._make_metric_tile(
            tr("custom_nodes 自定义节点", "custom_nodes Extensions"), tr("0 个插件", "0 nodes")
        )
        self.cfy_tile_wfs = self._make_metric_tile(
            tr("workflows 工作流存档", "workflows Saved Templates"), tr("0 套模板", "0 templates")
        )
        tiles_row.addWidget(self.cfy_tile_models[0], 1)
        tiles_row.addWidget(self.cfy_tile_nodes[0], 1)
        tiles_row.addWidget(self.cfy_tile_wfs[0], 1)
        c3.body.addLayout(tiles_row)
        lay.addWidget(c3)

        # Card 4: 8-Thread Custom Weight Downloader
        c4 = WorkspaceCard(
            tr("8 线程极速下载权重至 ComfyUI 云盘目录", "8-Thread High-Speed Weight Downloader to ComfyUI Drive"),
            tr(
                "支持粘贴 Hugging Face / Civitai 直链 URL，或填写 HF Repo ID + 文件子路径直接拉取至对应子目录",
                "Paste a direct Hugging Face / Civitai URL, or enter an HF Repo ID + filename to pull directly into ComfyUI",
            ),
        )
        dl_row1 = QHBoxLayout()
        dl_row1.setSpacing(8)
        dl_row1.addWidget(QLabel(tr("目标目录", "Target Folder")))
        self.cfy_subfolder_combo = QComboBox()
        for sf in COMFY_WEIGHT_SUBFOLDERS:
            self.cfy_subfolder_combo.addItem(f"models/{sf}", sf)
        dl_row1.addWidget(self.cfy_subfolder_combo)
        self.cfy_src_input = QLineEdit()
        self.cfy_src_input.setPlaceholderText(
            tr(
                "模型直链 https://... 或 HF 仓库如 Kijai/WanVideo_comfy",
                "Direct URL https://... or HF Repo ID e.g. Kijai/WanVideo_comfy",
            )
        )
        dl_row1.addWidget(self.cfy_src_input, 1)
        c4.body.addLayout(dl_row1)

        dl_row2 = QHBoxLayout()
        dl_row2.setSpacing(8)
        self.cfy_fname_input = QLineEdit()
        self.cfy_fname_input.setPlaceholderText(
            tr(
                "可选：仓库内文件子路径或保存文件名 (如 Wan2_2-I2V-14B.safetensors)",
                "Optional: subpath in repo or target filename (e.g. Wan2_2-I2V-14B.safetensors)",
            )
        )
        dl_row2.addWidget(self.cfy_fname_input, 1)
        self.cfy_dl_weight_btn = set_btn_role(
            QPushButton(tr("8 线程下载至云盘", "8-Thread Download")), "primary"
        )
        self.cfy_dl_weight_btn.clicked.connect(self._on_comfy_custom_dl)
        dl_row2.addWidget(self.cfy_dl_weight_btn)
        c4.body.addLayout(dl_row2)
        lay.addWidget(c4)
        return page

    def _make_metric_tile(self, label: str, val: str) -> tuple[QFrame, QLabel]:
        f = QFrame()
        f.setObjectName("SubtlePanel")
        l = QVBoxLayout(f)
        l.setContentsMargins(12, 8, 12, 8)
        l.setSpacing(2)
        t = QLabel(label)
        t.setStyleSheet("font-size: 11px; color: #94A3B8;")
        v = QLabel(val)
        v.setStyleSheet("font-size: 14px; font-weight: 700; font-family: Consolas, monospace;")
        l.addWidget(t)
        l.addWidget(v)
        return f, v

    # ─── Studio C: Qwen3.8 Chat, VLM & Dual-Protocol API Gateway ─────────────

    def _build_qwen_llm_studio(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        self.qwen_mode_stack = AutoHeightStackedWidget()

        # Subpage 0: Chat & VLM Interactive Studio
        chat_page = QWidget()
        cp_lay = QVBoxLayout(chat_page)
        cp_lay.setContentsMargins(0, 0, 0, 0)
        cp_lay.setSpacing(14)

        # VLM Image Drop Wells (shown only when task_mode_id == "vlm")
        self.vlm_images_card = WorkspaceCard(
            tr("待分析视觉图像", "Vision Input Images"),
            tr(
                "拖入 1~2 张图片进行构图光影解析、OCR 识别或反推专业生图/生视频 Prompt",
                "Drop 1-2 images for composition analysis, OCR extraction, or reverse-engineering T2I/T2V prompts",
            ),
        )
        vlm_row = QHBoxLayout()
        vlm_row.setSpacing(12)
        self.vlm_img1_well = FrameDropWell(
            tr("输入图像 1", "Input Image 1"),
            tr("拖拽或点击选择图片", "Drop or click to select image"),
        )
        self.vlm_img2_well = FrameDropWell(
            tr("对比图像 2 (可选)", "Comparison Image 2 (Optional)"),
            tr("拖拽或点击选择第二张图片", "Drop or click to select second image"),
        )
        vlm_row.addWidget(self.vlm_img1_well, 1)
        vlm_row.addWidget(self.vlm_img2_well, 1)
        self.vlm_images_card.body.addLayout(vlm_row)
        cp_lay.addWidget(self.vlm_images_card)

        # Prompt Composer
        q_card = WorkspaceCard(
            tr("问题与指令输入", "Prompt & Instruction Input"),
            tr(
                "支持最高 256K 超长上下文、深度思考链推理、代码架构设计与分镜脚本创作 (Ctrl+Enter 发送)",
                "Supports up to 256K context, deep chain-of-thought reasoning, architecture design & storyboards (Ctrl+Enter)",
            ),
        )
        self.llm_prompt_edit = QPlainTextEdit()
        self.llm_prompt_edit.setPlaceholderText(
            tr("输入问题、代码需求或视觉分析指令...", "Enter your question, coding spec, or visual analysis prompt...")
        )
        self.llm_prompt_edit.setFixedHeight(86)
        self.llm_prompt_edit.textChanged.connect(self._on_llm_prompt_edited)
        q_card.body.addWidget(self.llm_prompt_edit)
        q_card.body.addLayout(self._build_generate_action_bar("llm"))
        cp_lay.addWidget(q_card)

        # Multi-turn Conversation Card
        conv_hdr_btns = QWidget()
        ch_lay = QHBoxLayout(conv_hdr_btns)
        ch_lay.setContentsMargins(0, 0, 0, 0)
        ch_lay.setSpacing(6)
        to_api_btn = QPushButton(tr("API 接入配置", "API Gateway Setup"))
        to_api_btn.clicked.connect(lambda: self.vm.set_task_mode("api"))
        clr_chat_btn = QPushButton(tr("清空对话", "Clear Chat"))
        clr_chat_btn.clicked.connect(self._clear_chat_history)
        ch_lay.addWidget(to_api_btn)
        ch_lay.addWidget(clr_chat_btn)

        conv_card = WorkspaceCard(
            tr("对话与深度推理记录", "Conversation & Deep Reasoning Transcript"),
            tr(
                "实时流式输出长思维链 (Thinking Process) 与最终回复",
                "Live streaming of chain-of-thought (Thinking Process) and final response",
            ),
            trailing=conv_hdr_btns,
        )
        self.chat_turns_container = QVBoxLayout()
        self.chat_turns_container.setSpacing(10)
        conv_card.body.addLayout(self.chat_turns_container)
        cp_lay.addWidget(conv_card)
        self.qwen_mode_stack.addWidget(chat_page)

        # Subpage 1: OpenAI + Anthropic Dual-Protocol API Gateway
        self.qwen_mode_stack.addWidget(self._build_api_gateway_page())

        lay.addWidget(self.qwen_mode_stack)
        return page

    def _build_api_gateway_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        self.api_req_count_lbl = QLabel(tr("累计 API 请求: 0 次", "Total API Requests: 0"))
        self.api_req_count_lbl.setStyleSheet(
            "font-size: 11.5px; font-weight: 600; color: #64748B; font-family: Consolas, monospace;"
        )
        auth_card = WorkspaceCard(
            tr("网关安全鉴权与运行状态", "API Gateway Authentication & Status"),
            tr(
                "公网隧道开启期间，建议保持密钥鉴权开启以保护你的 Colab 算力单元 (CU)",
                "Keep API Key authentication enabled while the public tunnel is live to protect your Colab Compute Units (CU)",
            ),
            trailing=self.api_req_count_lbl,
        )
        top_r = QHBoxLayout()
        self.api_auth_chk = QCheckBox(
            tr("启用 API Key 鉴权（同时校验 Bearer 与 x-api-key）", "Require API Key Auth (validates Bearer & x-api-key)")
        )
        self.api_auth_chk.setChecked(self.vm.api_key_enabled)
        self.api_auth_chk.toggled.connect(self.vm.set_api_key_enabled)
        top_r.addWidget(self.api_auth_chk)
        top_r.addStretch(1)
        self.api_coresident_hint = QLabel("")
        self.api_coresident_hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.api_coresident_hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.api_coresident_hint.setStyleSheet("font-size: 11.5px;")
        top_r.addWidget(self.api_coresident_hint, 1)
        auth_card.body.addLayout(top_r)

        self.api_key_row_w = QWidget()
        kr = QHBoxLayout(self.api_key_row_w)
        kr.setContentsMargins(0, 0, 0, 0)
        kr.setSpacing(8)
        kr.addWidget(QLabel(tr("访问密钥", "API Key")))
        self.api_key_input = QLineEdit(self.vm.api_access_key)
        self.api_key_input.setStyleSheet("font-family: Consolas, monospace;")
        self.api_key_input.returnPressed.connect(lambda: self._sync_api_key_clicked())
        kr.addWidget(self.api_key_input, 1)

        sync_k_btn = QPushButton(tr("同步到云端", "Sync to Cloud"))
        sync_k_btn.clicked.connect(self._sync_api_key_clicked)
        regen_k_btn = QPushButton(tr("重新生成", "Regenerate"))
        regen_k_btn.clicked.connect(self._regen_api_key_clicked)
        copy_k_btn = set_btn_role(QPushButton(tr("复制密钥", "Copy Key")), "primary")
        copy_k_btn.clicked.connect(
            lambda: self._copy_text(
                self.vm.api_access_key,
                tr("已复制 API Key 到剪贴板。", "Copied API Key to clipboard."),
            )
        )
        kr.addWidget(sync_k_btn)
        kr.addWidget(regen_k_btn)
        kr.addWidget(copy_k_btn)
        auth_card.body.addWidget(self.api_key_row_w)
        lay.addWidget(auth_card)

        # Side-by-side Protocol Cards
        proto_row = QHBoxLayout()
        proto_row.setSpacing(14)

        # Left: OpenAI Compatible Protocol
        oai_card = WorkspaceCard(
            tr("OpenAI 兼容协议", "OpenAI-Compatible Protocol"),
            tr(
                "兼容 OpenAI SDK、Cursor、Cherry Studio、Chatbox 等客户端",
                "Compatible with OpenAI SDK, Cursor, Cherry Studio, Chatbox & standard tools",
            ),
        )
        self.oai_base_lbl = self._add_endpoint_copy_row(
            oai_card.body, "Base URL", lambda: self._api_snippets()["oai_base"]
        )
        self.oai_chat_lbl = self._add_endpoint_copy_row(
            oai_card.body, tr("Chat 接口", "Chat URL"), lambda: self._api_snippets()["oai_chat"]
        )
        self.oai_model_lbl = self._add_endpoint_copy_row(
            oai_card.body, tr("模型名称", "Model ID"), lambda: self._api_snippets()["repo"]
        )
        oai_desc = QLabel(
            tr(
                "支持 stream 流式输出、reasoning_content 深度思考链、image_url 视觉输入与 Function Calling；加载生图模型时亦支持 /v1/images/generations。",
                "Supports SSE streaming, reasoning_content CoT, image_url vision inputs & Function Calling; also serves /v1/images/generations when an image model is loaded.",
            )
        )
        oai_desc.setWordWrap(True)
        oai_desc.setStyleSheet("font-size: 11px; color: #64748B;")
        oai_card.body.addWidget(oai_desc)
        self.oai_code_box = self._make_readonly_code_view()
        oai_card.body.addWidget(self.oai_code_box)
        oai_btns = QHBoxLayout()
        b_py = QPushButton(tr("复制 Python 代码", "Copy Python SDK"))
        b_py.clicked.connect(
            lambda: self._copy_text(
                self._api_snippets()["oai_py"],
                tr("已复制 OpenAI Python 调用代码。", "Copied OpenAI Python snippet."),
            )
        )
        b_ocurl = QPushButton(tr("复制 curl 命令", "Copy curl"))
        b_ocurl.clicked.connect(
            lambda: self._copy_text(
                self._api_snippets()["oai_curl"],
                tr("已复制 OpenAI curl 测试命令。", "Copied OpenAI curl command."),
            )
        )
        oai_btns.addWidget(b_py)
        oai_btns.addWidget(b_ocurl)
        oai_btns.addStretch(1)
        oai_card.body.addLayout(oai_btns)
        proto_row.addWidget(oai_card, 1)

        # Right: Anthropic Messages Protocol
        ant_card = WorkspaceCard(
            tr("Anthropic Messages 协议", "Anthropic Messages Protocol"),
            tr(
                "原生实现 /v1/messages，兼容 Anthropic SDK 与 Claude Code",
                "Native /v1/messages endpoint for Anthropic SDK & Claude Code CLI",
            ),
        )
        self.ant_base_lbl = self._add_endpoint_copy_row(
            ant_card.body, "Base URL", lambda: self._api_snippets()["ant_base"]
        )
        self.ant_msg_lbl = self._add_endpoint_copy_row(
            ant_card.body, "Messages", lambda: self._api_snippets()["ant_msg"]
        )
        self._add_endpoint_copy_row(
            ant_card.body,
            tr("鉴权请求头", "Auth Header"),
            lambda: tr("x-api-key 或 Authorization: Bearer", "x-api-key or Authorization: Bearer"),
        )
        ant_desc = QLabel(
            tr(
                "完整支持 Anthropic SSE 事件流（thinking_delta 思考块、text_delta 文本块、tool_use 工具块）及 /v1/messages/count_tokens 接口。",
                "Full Anthropic SSE streaming (thinking_delta, text_delta, tool_use blocks) plus /v1/messages/count_tokens.",
            )
        )
        ant_desc.setWordWrap(True)
        ant_desc.setStyleSheet("font-size: 11px; color: #64748B;")
        ant_card.body.addWidget(ant_desc)
        self.ant_code_box = self._make_readonly_code_view()
        ant_card.body.addWidget(self.ant_code_box)
        ant_btns = QHBoxLayout()
        b_ps = QPushButton(tr("复制 PowerShell 配置", "Copy PowerShell Env"))
        b_ps.clicked.connect(
            lambda: self._copy_text(
                self._api_snippets()["claude_ps"],
                tr("已复制 Windows PowerShell Claude Code 配置。", "Copied Windows PowerShell Claude Code env."),
            )
        )
        b_sh = QPushButton(tr("复制 Bash 配置", "Copy Bash Env"))
        b_sh.clicked.connect(
            lambda: self._copy_text(
                self._api_snippets()["claude_sh"],
                tr("已复制 Bash ANTHROPIC_BASE_URL 配置。", "Copied Bash ANTHROPIC_BASE_URL env."),
            )
        )
        b_acurl = QPushButton(tr("复制 Anthropic curl", "Copy Anthropic curl"))
        b_acurl.clicked.connect(
            lambda: self._copy_text(
                self._api_snippets()["ant_curl"],
                tr("已复制 Anthropic /v1/messages curl 命令。", "Copied Anthropic /v1/messages curl command."),
            )
        )
        ant_btns.addWidget(b_ps)
        ant_btns.addWidget(b_sh)
        ant_btns.addWidget(b_acurl)
        ant_btns.addStretch(1)
        ant_card.body.addLayout(ant_btns)
        proto_row.addWidget(ant_card, 1)

        lay.addLayout(proto_row)
        return page

    def _add_endpoint_copy_row(self, parent_layout: QVBoxLayout, label: str, val_getter) -> QLabel:
        row = QHBoxLayout()
        row.setSpacing(6)
        k = QLabel(label)
        k.setFixedWidth(74)
        k.setStyleSheet("font-size: 11px; color: #64748B;")
        row.addWidget(k)
        v = QLabel(val_getter())
        v.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        v.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(v, 1)
        cp = QPushButton(tr("复制", "Copy"))
        cp.setFixedHeight(24)
        cp.clicked.connect(
            lambda: self._copy_text(
                val_getter(),
                tr(f"已复制 {label} 到剪贴板。", f"Copied {label} to clipboard."),
            )
        )
        row.addWidget(cp)
        parent_layout.addLayout(row)
        return v

    @staticmethod
    def _make_readonly_code_view() -> QPlainTextEdit:
        ed = QPlainTextEdit()
        ed.setObjectName("CodeSnippetBox")
        ed.setReadOnly(True)
        ed.setFixedHeight(168)
        return ed

    # ─── Studio D: Qwen-Image-2.1 & Z-Image-Turbo Studio ─────────────────────

    def _build_image_studio(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        self.img_refs_card = WorkspaceCard(
            tr("指令修图 / 风格重绘参考图", "Instruction Edit / Img2Img Reference Images"),
            tr(
                "上传原图后输入自然语言修改指令（Qwen-Image-2.1 支持双图主体融合）",
                "Upload source image(s) and enter natural-language edit instructions (Qwen-Image-2.1 supports dual-image fusion)",
            ),
        )
        r_row = QHBoxLayout()
        r_row.setSpacing(12)
        self.img_ref1_well = FrameDropWell(
            tr("主参考图 1 (必选)", "Primary Reference 1 (Required)"),
            tr("拖拽或点击上传主参考图", "Drop or click to upload primary image"),
        )
        self.img_ref2_well = FrameDropWell(
            tr("融合参考图 2 (可选)", "Fusion Reference 2 (Optional)"),
            tr("拖拽或点击上传第二张融合参考图", "Drop or click to upload second fusion image"),
        )
        r_row.addWidget(self.img_ref1_well, 1)
        r_row.addWidget(self.img_ref2_well, 1)
        self.img_refs_card.body.addLayout(r_row)
        lay.addWidget(self.img_refs_card)

        p_card = WorkspaceCard(
            tr("画面与海报排版提示词", "Image & Typography Generation Prompt"),
            tr(
                "支持复杂构图、高质感商业摄影及精准中英文海报文字直出",
                "Supports complex scene composition, commercial photography & accurate Chinese/English poster typography",
            ),
        )
        p_card.body.addWidget(QLabel(tr("正向提示词 (Prompt)", "Positive Prompt")))
        self.img_visual_edit = QPlainTextEdit(self.vm.visual_prompt)
        self.img_visual_edit.setPlaceholderText(
            tr(
                "描述画面主体、光影质感、镜头语言或需要排版的双引号文字...",
                "Describe subject, lighting, camera lens, or quoted poster text to render...",
            )
        )
        self.img_visual_edit.setFixedHeight(92)
        self.img_visual_edit.textChanged.connect(self._on_img_visual_edited)
        p_card.body.addWidget(self.img_visual_edit)

        p_card.body.addWidget(
            QLabel(
                tr(
                    "反向提示词 Negative Prompt（可选，排除低质或多余元素）",
                    "Negative Prompt (Optional — exclude artifacts or unwanted elements)",
                )
            )
        )
        self.img_neg_edit = QPlainTextEdit(self.vm.negative_prompt)
        self.img_neg_edit.setPlaceholderText(
            tr(
                "如：blurry, low quality, watermark, deformed hands...",
                "e.g. blurry, low quality, watermark, deformed hands...",
            )
        )
        self.img_neg_edit.setFixedHeight(48)
        self.img_neg_edit.textChanged.connect(
            lambda: setattr(self.vm, "negative_prompt", self.img_neg_edit.toPlainText())
        )
        p_card.body.addWidget(self.img_neg_edit)

        p_card.body.addLayout(self._build_generate_action_bar("img"))
        lay.addWidget(p_card)
        return page

    def _build_generate_action_bar(self, prefix: str) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(10)

        gen_btn = set_btn_role(QPushButton(tr("开始生成", "Generate")), "primary")
        gen_btn.setFixedHeight(38)
        gen_btn.clicked.connect(self._trigger_generate)
        row.addWidget(gen_btn, 1)

        stop_btn = set_btn_role(QPushButton(tr("停止推理", "Stop")), "danger")
        stop_btn.setFixedHeight(38)
        stop_btn.setFixedWidth(96)
        stop_btn.clicked.connect(self.vm.cancel_current_generation)
        row.addWidget(stop_btn)
        box.addLayout(row)

        prog = QProgressBar()
        prog.setRange(0, 100)
        prog.setVisible(False)
        box.addWidget(prog)

        setattr(self, f"{prefix}_gen_btn", gen_btn)
        setattr(self, f"{prefix}_stop_btn", stop_btn)
        setattr(self, f"{prefix}_prog_bar", prog)
        return box

    def _build_shared_media_bottom_split(self) -> QWidget:
        wrap = QWidget()
        hbox = QHBoxLayout(wrap)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(14)

        self.media_preview = MediaPreviewWidget()
        hbox.addWidget(self.media_preview, 1)

        # Right side stack: Page 0 = Local Media Gallery (254px), Page 1 = ComfyUI Cloud Outputs + Local (290px)
        self.right_gallery_stack = QStackedWidget()
        self.right_gallery_stack.setFixedWidth(268)

        # Page 0: Local History Gallery
        loc_card = WorkspaceCard(
            tr("本机成片库", "Local Media Library"),
            tr("自动保存至 Videos/EasyColab", "Auto-saved to Videos/EasyColab"),
        )
        loc_scroll = QScrollArea()
        loc_scroll.setWidgetResizable(True)
        loc_scroll.setMinimumHeight(250)
        loc_w = QWidget()
        self.local_gallery_lay = QVBoxLayout(loc_w)
        self.local_gallery_lay.setContentsMargins(0, 0, 4, 0)
        self.local_gallery_lay.setSpacing(6)
        loc_scroll.setWidget(loc_w)
        loc_card.body.addWidget(loc_scroll)
        self.right_gallery_stack.addWidget(loc_card)

        # Page 1: ComfyUI Cloud Outputs
        cfy_out_card = WorkspaceCard(
            tr("ComfyUI 云端输出成片", "ComfyUI Cloud Outputs"),
            tr("一键拉回本机播放与永久保存", "1-click pull to local library & preview"),
        )
        cfy_scroll = QScrollArea()
        cfy_scroll.setWidgetResizable(True)
        cfy_scroll.setMinimumHeight(250)
        cfy_w = QWidget()
        self.comfy_outputs_lay = QVBoxLayout(cfy_w)
        self.comfy_outputs_lay.setContentsMargins(0, 0, 4, 0)
        self.comfy_outputs_lay.setSpacing(6)
        cfy_scroll.setWidget(cfy_w)
        cfy_out_card.body.addWidget(cfy_scroll)
        self.right_gallery_stack.addWidget(cfy_out_card)

        hbox.addWidget(self.right_gallery_stack)
        return wrap

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. RIGHT COLLAPSIBLE RUN LOG PANEL (352px)
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_right_log_panel(self) -> QWidget:
        container = QWidget()
        container.setObjectName("RightLogContainer")
        container.setFixedWidth(352)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        hdr = QFrame()
        hdr.setObjectName("LogTopBar")
        hdr.setFixedHeight(42)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(12, 0, 10, 0)
        hl.setSpacing(6)
        log_title = QLabel(tr("运行日志", "Run Logs"))
        log_title.setStyleSheet("font-weight: 700; font-size: 12.5px;")
        hl.addWidget(log_title)
        hl.addStretch(1)

        self.log_pill = PillBadge(tr("就绪", "Ready"), fg="#64748B", bg="rgba(148,163,184,0.16)", mono=True)
        hl.addWidget(self.log_pill)

        cp_btn = set_btn_role(QPushButton(tr("复制", "Copy")), "ghost")
        cp_btn.clicked.connect(
            lambda: self._copy_text(
                "\n".join(self.vm.live_console_lines),
                tr("已复制全部运行日志到剪贴板。", "Copied all run logs to clipboard."),
            )
        )
        clr_btn = set_btn_role(QPushButton(tr("清空", "Clear")), "ghost")
        clr_btn.clicked.connect(self.vm.clear_console_lines)
        hl.addWidget(cp_btn)
        hl.addWidget(clr_btn)
        vbox.addWidget(hdr)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(12, 12, 12, 12)
        bl.setSpacing(10)

        # Status Summary Box
        st_box = QFrame()
        st_box.setObjectName("SidebarCard")
        st_lay = QVBoxLayout(st_box)
        st_lay.setContentsMargins(10, 8, 10, 8)
        st_lay.setSpacing(6)
        self.log_status_lbl = QLabel(self.vm.status_message)
        self.log_status_lbl.setWordWrap(True)
        self.log_status_lbl.setStyleSheet("font-size: 11.5px; line-height: 1.35;")
        st_lay.addWidget(self.log_status_lbl)
        self.log_task_prog = QProgressBar()
        self.log_task_prog.setRange(0, 100)
        self.log_task_prog.setVisible(False)
        st_lay.addWidget(self.log_task_prog)
        bl.addWidget(st_box)

        # Theme-Aware Monospace Terminal
        self.terminal_view = QTextBrowser()
        self.terminal_view.setObjectName("LogTerminalView")
        self.terminal_view.setOpenExternalLinks(False)
        bl.addWidget(self.terminal_view, 1)

        vbox.addWidget(body, 1)
        return container

    # ═══════════════════════════════════════════════════════════════════════════
    # UI REFRESH & SIGNAL HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════

    def open_onboarding_guide(self, step: int = 0) -> None:
        dlg = ColabGuideDialog(self.vm, initial_step=step, parent=self)
        dlg.exec()

    def _toggle_right_log_panel(self) -> None:
        self._show_right_log = not self._show_right_log
        self.right_log_panel.setVisible(self._show_right_log)

    def _submit_oauth_code(self) -> None:
        self.vm.submit_google_oauth_code(self.oauth_input.text())
        self.oauth_input.clear()

    def _cancel_oauth_code(self) -> None:
        self.vm.show_oauth_code_input = False
        self.oauth_input.clear()
        self._refresh_all_state()

    def _on_gpu_combo_changed(self, idx: int) -> None:
        if self._syncing_ui:
            return
        val = self.gpu_combo.itemData(idx)
        if val:
            self.vm.set_wake_gpu_choice(str(val))

    def _on_connect_clicked(self) -> None:
        self.vm.set_colab_server_url(self.url_input.text())
        self.vm.check_colab_status(silent=False)

    def _on_idle_combo_changed(self, idx: int) -> None:
        if self._syncing_ui:
            return
        mins = self.idle_combo.itemData(idx)
        if mins is not None:
            self.vm.update_idle_timeout_config(int(mins))

    def _confirm_release_all_vms(self) -> None:
        ans = QMessageBox.question(
            self,
            tr("释放所有 Colab 实例？", "Release All Colab Runtimes?"),
            tr(
                "账号下所有运行中的 Colab GPU 实例将被销毁，计费立即归零。\n确认立即释放吗？",
                "All active Colab GPU runtimes on your account will be terminated and billing will stop immediately.\nConfirm release now?",
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.vm.disconnect_remote_colab()

    def _on_hf_token_edited(self) -> None:
        self.vm.hf_token = self.hf_token_input.text().strip()
        self.vm._save_setting("hfToken", self.vm.hf_token)

    def _on_offline_quick_action(self) -> None:
        if self.vm.cli_logged_in:
            self.vm.wake_colab_headless()
        elif self.vm.normalized_base_url:
            self.vm.check_colab_status(silent=False)
        else:
            self.open_onboarding_guide(0)

    def _on_comfy_side_primary(self) -> None:
        if self.vm.comfy_running and self.vm.comfy_public_url:
            webbrowser.open(self.vm.comfy_public_url)
        else:
            self.vm.start_comfyui(force_restart=False, auto_open_browser=True)

    def _open_comfy_browser(self) -> None:
        if self.vm.comfy_public_url:
            webbrowser.open(self.vm.comfy_public_url)

    def _copy_comfy_url(self) -> None:
        if self.vm.comfy_public_url:
            self._copy_text(
                self.vm.comfy_public_url,
                tr("已复制 ComfyUI 公网画布地址到剪贴板。", "Copied ComfyUI public canvas URL to clipboard."),
            )

    def _on_comfy_custom_dl(self) -> None:
        sf = str(self.cfy_subfolder_combo.currentData() or "diffusion_models")
        self.vm.download_comfy_weight(self.cfy_src_input.text(), self.cfy_fname_input.text(), sf)

    def _on_vid_res_changed(self, idx: int) -> None:
        if 0 <= idx < len(VIDEO_RESOLUTIONS):
            self.vm.selected_video_res = VIDEO_RESOLUTIONS[idx]

    def _on_vid_frames_changed(self, val: int) -> None:
        snapped = snap_video_frames(val)
        self.vm.num_frames = snapped
        self.vid_frames_lbl.setText(
            tr(f"{snapped} 帧 · 约 {snapped / 24.0:.1f} 秒", f"{snapped} frames · ~{snapped / 24.0:.1f}s")
        )

    def _on_vid_steps_changed(self, val: int) -> None:
        snapped = max(10, min(60, int(round(val / 5.0)) * 5))
        self.vm.inference_steps = snapped
        self.vid_steps_lbl.setText(tr(f"{snapped} 步", f"{snapped} steps"))

    def _on_img_res_changed(self, idx: int) -> None:
        if 0 <= idx < len(IMAGE_RESOLUTIONS):
            self.vm.selected_image_res = IMAGE_RESOLUTIONS[idx]

    def _on_img_steps_changed(self, val: int) -> None:
        self.vm.inference_steps = int(val)
        self.img_steps_lbl.setText(tr(f"{val} 步", f"{val} steps"))

    def _on_denoise_changed(self, val: int) -> None:
        d = val / 100.0
        self.vm.denoising_strength = d
        self.denoise_lbl.setText(f"{d:.2f}")

    def _sync_rand_seed(self, checked: bool) -> None:
        self.vm.randomize_seed = checked
        if not self._syncing_ui:
            self._syncing_ui = True
            self.vid_rand_chk.setChecked(checked)
            self.img_rand_chk.setChecked(checked)
            self._syncing_ui = False

    def _set_batch_count(self, cnt: int) -> None:
        self.vm.batch_count = cnt
        self._refresh_batch_buttons()

    def _on_thinking_toggled(self, checked: bool) -> None:
        if self._syncing_ui:
            return
        self.vm.set_enable_thinking(checked)

    def _on_ctx_slider_preview(self, val: int) -> None:
        if self._syncing_ui:
            return
        snapped = max(4, min(256, int(round(val / 4.0)) * 4))
        self._syncing_ui = True
        self.ctx_spin.setValue(snapped)
        self._syncing_ui = False
        self.vm.set_context_length_k(snapped, sync_remote=False)

    def _on_ctx_changed(self, val: int, sync_remote: bool) -> None:
        if self._syncing_ui:
            return
        self._syncing_ui = True
        self.ctx_slider.setValue(val)
        self._syncing_ui = False
        self.vm.set_context_length_k(val, sync_remote=sync_remote)

    def _on_max_tokens_changed(self, val: int) -> None:
        snapped = max(1024, min(65536, int(round(val / 512.0)) * 512))
        self.vm.max_new_tokens = snapped
        self.max_tok_lbl.setText(str(snapped))

    def _on_temp_changed(self, val: int) -> None:
        t = val / 100.0
        self.vm.temperature = t
        self.temp_lbl.setText(f"{t:.2f} · Top-P {self.vm.top_p:.2f} · Top-K 20")

    def _on_vid_visual_edited(self) -> None:
        if not self._syncing_ui:
            self.vm.visual_prompt = self.vid_visual_edit.toPlainText()

    def _on_img_visual_edited(self) -> None:
        if not self._syncing_ui:
            self.vm.visual_prompt = self.img_visual_edit.toPlainText()

    def _on_llm_prompt_edited(self) -> None:
        if not self._syncing_ui:
            self.vm.visual_prompt = self.llm_prompt_edit.toPlainText()

    def _clear_chat_history(self) -> None:
        self.vm.chat_history.clear()
        self._refresh_chat_conversation()

    def _sync_api_key_clicked(self) -> None:
        self.vm.sync_api_key_to_colab(self.api_key_input.text())
        self.vm.status_message = tr(
            "API 密钥已同步更新至 Colab 云端网关。",
            "API Key synced to cloud Colab gateway.",
        )
        self._refresh_all_state()

    def _regen_api_key_clicked(self) -> None:
        self.vm.regenerate_api_key()
        self.api_key_input.setText(self.vm.api_access_key)
        self._refresh_all_state()

    def _copy_text(self, text: str, tip: str) -> None:
        cb = QGuiApplication.clipboard()
        if cb is not None:
            cb.setText(text)
        self.vm.status_message = tip
        self.log_status_lbl.setText(tip)

    def _trigger_generate(self) -> None:
        spec = self.vm.active_model_spec
        if spec.modality == ModelModality.COMFY_UI:
            return
        if spec.modality == ModelModality.LLM_VLM and self.vm.task_mode_id == "api":
            return

        first_b64: Optional[str] = None
        last_b64: Optional[str] = None
        if spec.modality in (ModelModality.VIDEO_AUDIO, ModelModality.VIDEO):
            self.vm.visual_prompt = self.vid_visual_edit.toPlainText()
            if self.vm.task_mode_id == "fl2va":
                first_b64 = self.video_first_well.to_jpeg_base64()
                last_b64 = self.video_last_well.to_jpeg_base64()
        elif spec.modality == ModelModality.LLM_VLM:
            self.vm.visual_prompt = self.llm_prompt_edit.toPlainText()
            if self.vm.task_mode_id == "vlm":
                first_b64 = self.vlm_img1_well.to_jpeg_base64()
                last_b64 = self.vlm_img2_well.to_jpeg_base64()
        elif spec.modality == ModelModality.IMAGE:
            self.vm.visual_prompt = self.img_visual_edit.toPlainText()
            if self.vm.task_mode_id == "i2i_edit":
                first_b64 = self.img_ref1_well.to_jpeg_base64()
                if spec.id == "qwen_image_21":
                    last_b64 = self.img_ref2_well.to_jpeg_base64()

        self.vm.start_generation(first_frame_b64=first_b64, last_frame_b64=last_b64)
        if spec.modality == ModelModality.LLM_VLM and not self.vm.visual_prompt:
            self._syncing_ui = True
            self.llm_prompt_edit.clear()
            self._syncing_ui = False

    # ─── Master State Refresh ────────────────────────────────────────────────

    def _refresh_all_state(self) -> None:
        self._syncing_ui = True
        pal = self.palette_obj
        spec = self.vm.active_model_spec

        # 1. Connection & Account
        if self.vm.is_connected:
            self.conn_badge.setText(tr("在线", "Online"))
            self.conn_badge.set_badge_style("#10B981", "rgba(16,185,129,0.16)")
            self.hw_dot.set_color("#10B981")
            self.ws_top_dot.set_color("#10B981")
            self.connect_btn.setText(tr("刷新", "Refresh"))
            top_text = tr(
                f"{self.vm.gpu_name} · 显存 {self.vm.vram_used_gb:.1f}/{self.vm.vram_total_gb:.0f}G · "
                f"内存 {self.vm.ram_used_gb:.1f}/{self.vm.ram_total_gb:.0f}G · "
                f"磁盘可用 {self.vm.disk_free_gb:.0f}/{self.vm.disk_total_gb:.0f}G",
                f"{self.vm.gpu_name} · VRAM {self.vm.vram_used_gb:.1f}/{self.vm.vram_total_gb:.0f}G · "
                f"RAM {self.vm.ram_used_gb:.1f}/{self.vm.ram_total_gb:.0f}G · "
                f"Disk Free {self.vm.disk_free_gb:.0f}/{self.vm.disk_total_gb:.0f}G",
            )
            self.ws_top_summary.setToolTip(top_text)
            top_w = max(220, self.ws_top_summary.width() or 360)
            self.ws_top_summary.setText(
                QFontMetrics(self.ws_top_summary.font()).elidedText(top_text, Qt.TextElideMode.ElideRight, top_w)
            )
        else:
            self.conn_badge.setText(tr("离线", "Offline"))
            self.conn_badge.set_badge_style("#94A3B8", "rgba(148,163,184,0.16)")
            self.hw_dot.set_color("#64748B")
            self.ws_top_dot.set_color("#64748B")
            self.connect_btn.setText(tr("连接", "Connect"))
            self.ws_top_summary.setToolTip("")
            self.ws_top_summary.setText(tr("开源多模型云端工作台", "Open Multi-Model Cloud AI Studio"))

        acct_txt = self.vm.display_account_label
        self.account_val_lbl.setToolTip(acct_txt)
        acct_w = max(90, (self.account_val_lbl.width() or 120) - 4)
        self.account_val_lbl.setText(
            QFontMetrics(self.account_val_lbl.font()).elidedText(acct_txt, Qt.TextElideMode.ElideMiddle, acct_w)
        )
        self.bind_btn.setVisible(not self.vm.cli_logged_in)
        self.acc_refresh_btn.setVisible(self.vm.cli_logged_in)
        self.acc_menu_btn.setVisible(self.vm.cli_logged_in)
        acc_note = self.vm.usage_note if self.vm.cli_logged_in else ""
        self.acc_usage_lbl.setText(acc_note)
        self.acc_usage_lbl.setVisible(bool(acc_note))
        self.oauth_box.setVisible(self.vm.show_oauth_code_input)
        self.wake_row_widget.setVisible(self.vm.cli_logged_in)

        g_idx = self.gpu_combo.findData(self.vm.wake_gpu_choice)
        if g_idx >= 0:
            self.gpu_combo.setCurrentIndex(g_idx)
        self.wake_btn.setEnabled(not self.vm.is_waking_colab)
        self.wake_btn.setText(
            tr("唤醒部署中...", "Deploying...") if self.vm.is_waking_colab else tr("唤醒实例并部署", "Wake & Deploy")
        )

        if not self.url_input.hasFocus():
            self.url_input.setText(self.vm.colab_server_url)
        self.url_input.setEchoMode(
            QLineEdit.EchoMode.Password if self.vm.mask_server_url else QLineEdit.EchoMode.Normal
        )
        self.url_input.setPlaceholderText(
            tr("已开启打码保护", "Privacy mask enabled") if self.vm.mask_server_url else "https://xxxx.gradio.live"
        )
        self.mask_eye_btn.setText(tr("已打码", "Masked") if self.vm.mask_server_url else tr("打码", "Mask"))

        self.hw_gpu_lbl.setText(self.vm.gpu_name)
        self.drive_pill.setVisible(self.vm.drive_persisted)

        # 2. CU Ledger & Idle Timeout
        gpu_summary = self.vm.active_gpu_summary
        if gpu_summary:
            has_official = self.vm.official_cu_balance is not None
            self.cu_title_lbl.setText(
                tr(
                    f"账户余额 · 实例：{gpu_summary}",
                    f"CU Balance · Instances: {gpu_summary}",
                )
                if has_official
                else tr(
                    f"预计余额 · 实例：{gpu_summary}",
                    f"Est. Balance · Instances: {gpu_summary}",
                )
            )
        else:
            self.cu_title_lbl.setText(
                tr("账户余额", "CU Balance") if self.vm.official_cu_balance is not None else tr("预计余额", "Est. Balance")
            )
        self.cu_val_lbl.setText(f"{self.vm.display_remaining_cu:.2f} CU")
        burn = self.vm.display_burn_rate
        self.burn_val_lbl.setText(f"{burn:.2f} CU/h")
        self.burn_val_lbl.setStyleSheet(
            f"font-size: 12.5px; font-weight: 700; font-family: Consolas, monospace; "
            f"color: {'#F59E0B' if burn > 0 else pal.text_primary};"
        )
        self.uptime_val_lbl.setText(format_uptime_hms(self.vm.uptime_seconds))

        idle_idx = self.idle_combo.findData(self.vm.idle_timeout_minutes)
        if idle_idx >= 0:
            self.idle_combo.setCurrentIndex(idle_idx)
        if self.vm.idle_timeout_minutes <= 0:
            self.idle_countdown_lbl.setText(
                tr(
                    "已关闭闲置自动休眠（请记得用完手动释放实例）",
                    "Idle auto-release is Off (remember to release runtime manually)",
                )
            )
        elif self.vm.is_connected:
            rem = max(0, self.vm.idle_remaining_seconds)
            self.idle_countdown_lbl.setText(
                tr(
                    f"{rem // 60}分{rem % 60:02d}秒无操作后自动释放实例",
                    f"Auto-release in {rem // 60}m {rem % 60:02d}s if idle",
                )
            )
        else:
            self.idle_countdown_lbl.setText(
                tr(
                    f"连接后超过 {self.vm.idle_timeout_minutes} 分钟闲置将自动释放实例",
                    f"Auto-release after {self.vm.idle_timeout_minutes} min of inactivity once connected",
                )
            )

        if self.vm.active_vm_count > 0:
            self.release_all_btn.setText(
                tr(
                    f"释放全部实例 ({self.vm.active_vm_count} 台运行中)",
                    f"Release All Runtimes ({self.vm.active_vm_count} Active)",
                )
            )
        else:
            self.release_all_btn.setText(tr("释放全部实例", "Release All Runtimes"))
        self.release_all_btn.setEnabled(not self.vm.is_releasing_colab)

        # 3. Model Center Cards & Co-Residency
        blue_badge_fg = "#60A5FA" if pal.is_dark else "#2563EB"
        if len(self.vm.loaded_model_ids) > 1:
            self.model_center_badge.setText(
                tr(f"共驻 {len(self.vm.loaded_model_ids)} 模型", f"{len(self.vm.loaded_model_ids)} Co-Resident")
            )
            self.model_center_badge.set_badge_style("#059669", "rgba(16,185,129,0.14)")
        else:
            self.model_center_badge.setText("HF Top 6")
            self.model_center_badge.set_badge_style(blue_badge_fg, "rgba(59,130,246,0.14)")

        for m_spec in OPEN_MODEL_CATALOG:
            btn = self._model_card_btns[m_spec.id]
            is_sel = m_spec.id == self.vm.selected_model_id
            is_loaded = m_spec.id in self.vm.loaded_model_ids or (
                m_spec.id == "comfyui_studio" and self.vm.comfy_running
            )
            status_tag = tr("  [显存就绪]", "  [VRAM Ready]") if is_loaded else ""
            btn.setText(f"{m_spec.name}{status_tag}\n{m_spec.badge}")
            if is_sel:
                btn.setStyleSheet(
                    f"QPushButton {{ text-align: left; padding: 7px 10px; font-size: 11.5px; font-weight: 700; "
                    f"background-color: {pal.selected_card_bg}; border: 1.5px solid {pal.selected_card_border}; border-radius: 8px; }}"
                )
            else:
                btn.setStyleSheet(
                    f"QPushButton {{ text-align: left; padding: 7px 10px; font-size: 11.5px; "
                    f"background-color: {pal.input_fill}; border: 1px solid {pal.card_border}; border-radius: 8px; }}"
                    f"QPushButton:hover {{ border-color: {pal.accent_blue}; }}"
                )

        self._rebuild_coresident_strip()

        if spec.modality == ModelModality.COMFY_UI:
            self.model_ctrl_stack.setCurrentIndex(1)
            if self.vm.comfy_running:
                self.cfy_side_status_lbl.setText(tr("在线 (端口 8188)", "Online (Port 8188)"))
                self.cfy_side_status_lbl.setStyleSheet("color: #10B981; font-weight: 700; font-size: 11.5px;")
                self.cfy_side_start_btn.setText(tr("打开画布", "Open Canvas"))
            elif self.vm.comfy_starting:
                self.cfy_side_status_lbl.setText(tr("部署启动中...", "Deploying..."))
                self.cfy_side_status_lbl.setStyleSheet("color: #F59E0B; font-weight: 700; font-size: 11.5px;")
                self.cfy_side_start_btn.setText(tr("正在启动...", "Starting..."))
            else:
                self.cfy_side_status_lbl.setText(tr("未启动", "Stopped"))
                self.cfy_side_status_lbl.setStyleSheet("color: #94A3B8; font-weight: 700; font-size: 11.5px;")
                self.cfy_side_start_btn.setText(tr("启动 ComfyUI 画布", "Launch ComfyUI"))
            self.cfy_side_stop_btn.setEnabled(self.vm.comfy_running)
            cfy_busy = (
                self.vm.comfy_starting or self.vm.comfy_downloading_weight or bool(self.vm.comfy_installing_pack_id)
            )
            self.cfy_side_prog.setVisible(cfy_busy)
            self.cfy_side_prog.setValue(int(self.vm.comfy_progress * 100))
        else:
            self.model_ctrl_stack.setCurrentIndex(0)
            self.repo_val_lbl.setText(spec.repo_id)
            if self.vm.all_downloaded:
                sz = self.vm.total_size_gb if self.vm.total_size_gb > 0 else spec.approx_size_gb
                self.weight_status_lbl.setText(
                    tr(
                        f"已就绪 {self.vm.downloaded_count}/{self.vm.total_count} ({sz:.1f} GB)",
                        f"Ready {self.vm.downloaded_count}/{self.vm.total_count} ({sz:.1f} GB)",
                    )
                )
                self.weight_status_lbl.setStyleSheet("color: #10B981; font-size: 11.5px; font-weight: 700;")
            else:
                self.weight_status_lbl.setText(
                    tr(
                        f"{self.vm.downloaded_count}/{self.vm.total_count} 个分片",
                        f"{self.vm.downloaded_count}/{self.vm.total_count} shards",
                    )
                )
                self.weight_status_lbl.setStyleSheet("color: #F59E0B; font-size: 11.5px; font-weight: 600;")

            self._rebuild_shards_list()

            # KV mismatch check for Qwen GGUF
            loaded_k = self.vm.loaded_context_length // 1024
            if (
                spec.id == "qwen38_27b_gguf"
                and self.vm.is_selected_model_loaded_in_vram
                and loaded_k > 0
                and loaded_k != self.vm.context_length_k
            ):
                self.kv_mismatch_lbl.setText(
                    tr(
                        f"当前显存 KV 为 {loaded_k}K，点「重新装载」切换至 {self.vm.context_length_k}K",
                        f"Loaded KV is {loaded_k}K — click 'Reload VRAM' to apply {self.vm.context_length_k}K",
                    )
                )
                self.kv_mismatch_lbl.setVisible(True)
            else:
                self.kv_mismatch_lbl.setVisible(False)

            self.offline_banner.setVisible(not self.vm.is_connected)
            self.offline_quick_btn.setText(
                tr("一键唤醒", "1-Click Wake") if self.vm.cli_logged_in else tr("重连", "Reconnect")
            )

            busy_w = self.vm.is_downloading_weights or self.vm.is_loading_model
            self.weight_prog_bar.setVisible(busy_w)
            self.weight_prog_bar.setValue(int(self.vm.model_task_progress * 100))

            self.dl_weights_btn.setText(
                tr("校验权重", "Verify") if self.vm.all_downloaded else tr("下载权重", "Download")
            )
            self.dl_weights_btn.setEnabled(self.vm.is_connected and not busy_w)

            self.load_vram_btn.setText(
                tr("重新装载", "Reload VRAM")
                if self.vm.is_selected_model_loaded_in_vram
                else tr("加载到显存", "Load to VRAM")
            )
            self.load_vram_btn.setEnabled(self.vm.is_connected and self.vm.all_downloaded and not busy_w)
            self.eject_vram_btn.setVisible(self.vm.is_selected_model_loaded_in_vram)

        # 4. Adaptive Params Card
        if spec.modality in (ModelModality.VIDEO_AUDIO, ModelModality.VIDEO):
            self.params_card.title_label.setText(tr("规格与采样参数", "Resolution & Sampling"))
            self.params_stack.setCurrentIndex(0)
            self.vid_steps_slider.setValue(self.vm.inference_steps)
            self.vid_steps_lbl.setText(tr(f"{self.vm.inference_steps} 步", f"{self.vm.inference_steps} steps"))
        elif spec.modality == ModelModality.IMAGE:
            self.params_card.title_label.setText(tr("规格与采样参数", "Resolution & Sampling"))
            self.params_stack.setCurrentIndex(1)
            self.img_steps_slider.setValue(self.vm.inference_steps)
            self.img_steps_lbl.setText(tr(f"{self.vm.inference_steps} 步", f"{self.vm.inference_steps} steps"))
            self.denoise_box.setVisible(spec.id == "z_image_turbo" and self.vm.task_mode_id == "i2i_edit")
        elif spec.modality == ModelModality.LLM_VLM:
            self.params_card.title_label.setText(tr("推理与对话参数", "Reasoning & Sampling"))
            self.params_stack.setCurrentIndex(2)
            self.thinking_chk.setChecked(self.vm.enable_thinking)
            self.effort_box.setVisible(self.vm.enable_thinking)
            for ef_id, b in self._effort_btns.items():
                if ef_id == self.vm.reasoning_effort:
                    b.setStyleSheet(
                        f"background-color: {pal.accent_blue}; color: #FFFFFF; font-weight: 700; border-radius: 5px;"
                    )
                else:
                    b.setStyleSheet("")
            self.ctx_spin.setValue(self.vm.context_length_k)
            self.ctx_slider.setValue(self.vm.context_length_k)
            self.temp_slider.setValue(int(round(self.vm.temperature * 100)))
            self.temp_lbl.setText(f"{self.vm.temperature:.2f} · Top-P {self.vm.top_p:.2f} · Top-K 20")
        elif spec.modality == ModelModality.COMFY_UI:
            self.params_card.title_label.setText(tr("ComfyUI 云盘持久化与控制", "ComfyUI Persistence & Control"))
            self.params_stack.setCurrentIndex(3)
            self.cfy_persist_lbl.setText(
                tr(f"架构：{self.vm.comfy_persistence_label}", f"Storage: {self.vm.comfy_persistence_label}")
            )
            self.cfy_counts_lbl.setText(
                tr(
                    f"已存节点 / 权重：{self.vm.comfy_custom_nodes_count} 个节点 · {self.vm.comfy_models_count} 个权重",
                    f"Saved Nodes / Weights: {self.vm.comfy_custom_nodes_count} nodes · {self.vm.comfy_models_count} weights",
                )
            )
            self.cfy_queue_lbl.setText(
                tr(
                    f"画布执行队列：运行 {self.vm.comfy_queue_running} · 排队 {self.vm.comfy_queue_pending}",
                    f"Canvas Queue: {self.vm.comfy_queue_running} running · {self.vm.comfy_queue_pending} queued",
                )
            )

        self._refresh_batch_buttons()

        # 5. Center Mode Switcher & Preset Menu
        self._rebuild_preset_menu()
        self.ms_title_lbl.setText(spec.name)
        self.ms_repo_pill.setText(spec.repo_id)
        self.ms_summary_lbl.setText(spec.summary)

        if len(self.vm.loaded_model_ids) > 1:
            self.ms_coresident_pill.setText(
                tr(
                    f"{len(self.vm.loaded_model_ids)} 模型显存共驻",
                    f"{len(self.vm.loaded_model_ids)} Models Co-Resident",
                )
            )
            self.ms_coresident_pill.setVisible(True)
        else:
            self.ms_coresident_pill.setVisible(False)

        ready_vram = self.vm.is_selected_model_loaded_in_vram or (
            spec.modality == ModelModality.COMFY_UI and self.vm.comfy_running
        )
        if ready_vram:
            self.ms_vram_pill.setText(tr("显存已就绪", "VRAM Ready"))
            self.ms_vram_pill.set_badge_style("#10B981", "rgba(16,185,129,0.16)")
        else:
            self.ms_vram_pill.setText(tr("待加载至显存", "Not Loaded in VRAM"))
            self.ms_vram_pill.set_badge_style("#F59E0B", "rgba(245,158,11,0.16)")

        self._rebuild_mode_radio_cards()

        # 6. Active Modality Studio Routing
        if spec.modality in (ModelModality.VIDEO_AUDIO, ModelModality.VIDEO):
            self.studio_stack.setCurrentIndex(0)
            self.video_frames_card.setVisible(self.vm.task_mode_id == "fl2va")
            self.shared_media_split.setVisible(True)
            self.right_gallery_stack.setCurrentIndex(0)
            self._update_gen_bar(
                "vid",
                tr(
                    f"连续生成 {self.vm.batch_count} 条全模态音视频 (Ctrl+Enter)",
                    f"Generate {self.vm.batch_count}x Audio-Video Clips (Ctrl+Enter)",
                )
                if self.vm.batch_count > 1
                else tr("开始生成音视频 (Ctrl+Enter)", "Generate Audio-Video (Ctrl+Enter)"),
            )
        elif spec.modality == ModelModality.COMFY_UI:
            self.studio_stack.setCurrentIndex(1)
            self.shared_media_split.setVisible(True)
            self.right_gallery_stack.setCurrentIndex(1)
            self._refresh_comfyui_studio_cards()
        elif spec.modality == ModelModality.LLM_VLM:
            self.studio_stack.setCurrentIndex(2)
            self.shared_media_split.setVisible(False)
            if self.vm.task_mode_id == "api":
                self.qwen_mode_stack.setCurrentIndex(1)
                self._refresh_api_gateway_cards()
            else:
                self.qwen_mode_stack.setCurrentIndex(0)
                self.vlm_images_card.setVisible(self.vm.task_mode_id == "vlm")
                self._update_gen_bar("llm", tr("发送并开始深度推理 (Ctrl+Enter)", "Send & Start Deep Reasoning (Ctrl+Enter)"))
        elif spec.modality == ModelModality.IMAGE:
            self.studio_stack.setCurrentIndex(3)
            self.img_refs_card.setVisible(self.vm.task_mode_id == "i2i_edit")
            self.img_ref2_well.setVisible(spec.id == "qwen_image_21")
            self.shared_media_split.setVisible(True)
            self.right_gallery_stack.setCurrentIndex(0)
            self._update_gen_bar(
                "img",
                tr(
                    f"连续生成 {self.vm.batch_count} 张高清图像 (Ctrl+Enter)",
                    f"Generate {self.vm.batch_count}x HD Images (Ctrl+Enter)",
                )
                if self.vm.batch_count > 1
                else tr("开始生成图像 (Ctrl+Enter)", "Generate Image (Ctrl+Enter)"),
            )

        # 7. Right Log Panel Header & Status
        self.log_status_lbl.setText(self.vm.status_message)
        any_busy = (
            self.vm.is_generating
            or self.vm.is_downloading_weights
            or self.vm.is_loading_model
            or self.vm.is_waking_colab
            or self.vm.comfy_starting
            or self.vm.comfy_downloading_weight
            or bool(self.vm.comfy_installing_pack_id)
        )
        if any_busy:
            self.log_pill.setText(tr("实时传输", "Streaming"))
            self.log_pill.set_badge_style("#059669", "rgba(16,185,129,0.14)")
        else:
            self.log_pill.setText(tr(f"{len(self.vm.live_console_lines)} 行", f"{len(self.vm.live_console_lines)} lines"))
            self.log_pill.set_badge_style("#64748B", "rgba(148,163,184,0.16)")

        if self.vm.is_generating:
            self.log_task_prog.setVisible(True)
            self.log_task_prog.setValue(int(self.vm.progress_value * 100))
        elif self.vm.is_downloading_weights or self.vm.is_loading_model:
            self.log_task_prog.setVisible(True)
            self.log_task_prog.setValue(int(self.vm.model_task_progress * 100))
        else:
            self.log_task_prog.setVisible(False)

        self._syncing_ui = False

    def _update_gen_bar(self, prefix: str, idle_label: str) -> None:
        gen_btn: QPushButton = getattr(self, f"{prefix}_gen_btn")
        stop_btn: QPushButton = getattr(self, f"{prefix}_stop_btn")
        prog: QProgressBar = getattr(self, f"{prefix}_prog_bar")
        if self.vm.is_generating:
            pct = int(self.vm.progress_value * 100)
            gen_btn.setText(
                tr(
                    f"正在云端推理中... {pct}% · 已用时 {format_duration(self.vm.current_job_elapsed)}",
                    f"Inferring on Cloud... {pct}% · Elapsed {format_duration(self.vm.current_job_elapsed)}",
                )
            )
            gen_btn.setEnabled(False)
            stop_btn.setEnabled(True)
            prog.setVisible(True)
            prog.setValue(pct)
        else:
            gen_btn.setText(idle_label)
            gen_btn.setEnabled(True)
            stop_btn.setEnabled(False)
            prog.setVisible(False)

    def _refresh_batch_buttons(self) -> None:
        pal = self.palette_obj
        for btns in (self._vid_batch_btns, self._img_batch_btns):
            for n, b in btns.items():
                if n == self.vm.batch_count:
                    b.setStyleSheet(
                        f"background-color: {pal.accent_blue}; color: #FFFFFF; font-weight: 700; border-radius: 5px;"
                    )
                else:
                    b.setStyleSheet("")

    def _rebuild_preset_menu(self) -> None:
        self.preset_menu.clear()
        mod = self.vm.active_model_spec.modality
        matching = [t for t in PRESET_TEMPLATES if t.modality == mod]
        if not matching:
            matching = PRESET_TEMPLATES
        for tpl in matching:
            act = QAction(tpl.title, self)
            act.triggered.connect(lambda _c=False, tp=tpl: self._apply_preset_template(tp))
            self.preset_menu.addAction(act)

    def _apply_preset_template(self, tpl) -> None:
        self._syncing_ui = True
        self.vm.visual_prompt = tpl.visual
        self.vm.soundscape_prompt = tpl.soundscape
        self.vm.music_prompt = tpl.music
        self.vid_visual_edit.setPlainText(tpl.visual)
        self.vid_sound_edit.setPlainText(tpl.soundscape)
        self.vid_music_edit.setPlainText(tpl.music)
        self.img_visual_edit.setPlainText(tpl.visual)
        self.llm_prompt_edit.setPlainText(tpl.visual)
        self._syncing_ui = False
        self.vm.status_message = tr(f"已填入示例提示词：{tpl.title}", f"Applied preset template: {tpl.title}")
        self.log_status_lbl.setText(self.vm.status_message)

    def _rebuild_coresident_strip(self) -> None:
        sig = (
            get_lang(),
            tuple(self.vm.loaded_model_ids),
            tuple((d.id, d.name, d.vram_gb) for d in self.vm.loaded_models_detail),
        )
        if sig == self._last_coresident_sig:
            return
        self._last_coresident_sig = sig
        self._clear_layout(self.coresident_lay)

        details = self.vm.loaded_models_detail
        if not details and not self.vm.loaded_model_ids:
            self.coresident_box.setVisible(False)
            return

        self.coresident_box.setVisible(True)
        hdr = QHBoxLayout()
        lbl = QLabel(
            tr(
                f"显存常驻模型 ({len(self.vm.loaded_model_ids)})",
                f"Co-Resident in VRAM ({len(self.vm.loaded_model_ids)})",
            )
        )
        lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #10B981;")
        hdr.addWidget(lbl, 1)
        if len(self.vm.loaded_model_ids) >= 2:
            ua = set_btn_role(QPushButton(tr("全部卸载", "Unload All")), "ghost")
            ua.setStyleSheet("font-size: 10.5px; color: #F87171;")
            ua.clicked.connect(lambda: self.vm.unload_model_from_vram("all"))
            hdr.addWidget(ua)
        self.coresident_lay.addLayout(hdr)

        items_to_show = (
            [(d.id, d.name, f"{d.vram_gb:.1f}G" if d.vram_gb else "") for d in details]
            if details
            else [(mid, MODEL_SPEC_MAP[mid].name if mid in MODEL_SPEC_MAP else mid, "") for mid in self.vm.loaded_model_ids]
        )
        for mid, mname, vtxt in items_to_show:
            r = QHBoxLayout()
            r.setSpacing(4)
            info = f"{mname}" + (f" ({vtxt})" if vtxt else "")
            il = QLabel(info)
            il.setStyleSheet("font-size: 11px;")
            r.addWidget(il, 1)
            xb = set_btn_role(QPushButton(tr("卸载", "Unload")), "ghost")
            xb.setFixedHeight(22)
            xb.clicked.connect(lambda _c=False, m=mid: self.vm.unload_model_from_vram(m))
            r.addWidget(xb)
            self.coresident_lay.addLayout(r)

    def _rebuild_shards_list(self) -> None:
        self.shards_box.set_title(
            tr(f"查看权重分片 ({len(self.vm.model_files)})", f"View Weight Shards ({len(self.vm.model_files)})")
        )
        sig = (
            get_lang(),
            self.vm.selected_model_id,
            tuple((f.name, f.exists, round(f.size_gb, 2), f.desc) for f in self.vm.model_files),
        )
        if sig == self._last_shards_sig:
            return
        self._last_shards_sig = sig
        self._clear_layout(self.shards_list_lay)
        for f in self.vm.model_files:
            mark = tr("[已就绪]", "[Ready]") if f.exists else tr("[未下载]", "[Missing]")
            color = "#059669" if f.exists else "#64748B"
            sz = (
                f"{f.size_gb:.1f}G"
                if f.exists and f.size_gb > 0
                else (tr("已就绪", "Ready") if f.exists else tr("未下", "Missing"))
            )
            short_name = Path(f.name).name
            lbl = QLabel(f"{mark} {short_name}  ({sz})")
            if f.desc:
                lbl.setToolTip(f.desc)
            lbl.setStyleSheet(f"font-family: Consolas, monospace; font-size: 10.5px; color: {color};")
            self.shards_list_lay.addWidget(lbl)
        self.shards_list_lay.addStretch(1)

    def _rebuild_mode_radio_cards(self) -> None:
        sig = (get_lang(), self.vm.selected_model_id, self.vm.task_mode_id, self.vm.dark_mode)
        if sig == self._last_modes_sig:
            return
        self._last_modes_sig = sig
        self._clear_layout(self.mode_buttons_row)
        pal = self.palette_obj
        for m in self.vm.active_model_spec.modes:
            is_cur = m.id == self.vm.task_mode_id
            card = ModeRadioCard(
                glyph=icon_glyph(m.icon),
                title=m.title,
                subtitle=m.subtitle,
                selected=is_cur,
                pal=pal,
            )
            card.clicked.connect(lambda mid=m.id: self.vm.set_task_mode(mid))
            self.mode_buttons_row.addWidget(card, 1)

    def _refresh_comfyui_studio_cards(self) -> None:
        if self.vm.comfy_running:
            if self.vm.comfy_queue_running > 0 or self.vm.comfy_queue_pending > 0:
                self.cfy_status_badge.setText(
                    tr(
                        f"队列渲染中 ({self.vm.comfy_queue_running} 运行 / {self.vm.comfy_queue_pending} 排队)",
                        f"Rendering ({self.vm.comfy_queue_running} running / {self.vm.comfy_queue_pending} queued)",
                    )
                )
                self.cfy_status_badge.set_badge_style("#60A5FA", "rgba(59,130,246,0.16)")
            else:
                self.cfy_status_badge.setText(tr("画布在线待命", "Canvas Online"))
                self.cfy_status_badge.set_badge_style("#10B981", "rgba(16,185,129,0.16)")
        elif self.vm.comfy_starting:
            self.cfy_status_badge.setText(tr("正在部署启动...", "Deploying..."))
            self.cfy_status_badge.set_badge_style("#F59E0B", "rgba(245,158,11,0.16)")
        else:
            self.cfy_status_badge.setText(tr("未启动", "Stopped"))
            self.cfy_status_badge.set_badge_style("#94A3B8", "rgba(148,163,184,0.16)")

        self.cfy_launch_btn.setVisible(not self.vm.comfy_running)
        self.cfy_launch_btn.setEnabled(not self.vm.comfy_starting)
        self.cfy_running_bar.setVisible(self.vm.comfy_running)

        busy = self.vm.comfy_starting or self.vm.comfy_downloading_weight or bool(self.vm.comfy_installing_pack_id)
        self.cfy_main_prog.setVisible(busy)
        self.cfy_main_prog.setValue(int(self.vm.comfy_progress * 100))

        tunnel_str = self.vm.masked_url(self.vm.comfy_public_url) if self.vm.comfy_public_url else ""
        tunnel_tag = tr(f"  |  画布隧道: {tunnel_str}", f"  |  Tunnel: {tunnel_str}") if tunnel_str else ""
        self.cfy_tunnel_lbl.setText(f"{self.vm.comfy_status_message}{tunnel_tag}")

        # Preset Packs
        ready_cnt = sum(1 for p in self.vm.comfy_preset_packs if p.ready)
        total_packs = len(self.vm.comfy_preset_packs)
        self.cfy_packs_badge.setText(
            tr(f"已就绪 {ready_cnt} / {total_packs} 套", f"{ready_cnt} / {total_packs} Packs Ready")
        )
        packs_sig = (
            get_lang(),
            tuple(
                (p.id, p.ready, p.installed_models, p.total_models, p.workflows_installed)
                for p in self.vm.comfy_preset_packs
            ),
            self.vm.comfy_installing_pack_id,
            int(self.vm.comfy_progress * 100) if self.vm.comfy_installing_pack_id else 0,
            self.vm.comfy_downloading_weight,
            self.vm.dark_mode,
        )
        if packs_sig != self._last_packs_sig:
            self._last_packs_sig = packs_sig
            self._clear_layout(self.cfy_packs_container)
            for pack in self.vm.comfy_preset_packs:
                self.cfy_packs_container.addWidget(self._build_comfy_pack_row(pack))
            self.cfy_packs_widget.updateGeometry()

        self.cfy_tile_models[1].setText(
            tr(f"{self.vm.comfy_models_count} 个文件", f"{self.vm.comfy_models_count} files")
        )
        self.cfy_tile_nodes[1].setText(
            tr(f"{self.vm.comfy_custom_nodes_count} 个插件", f"{self.vm.comfy_custom_nodes_count} nodes")
        )
        self.cfy_tile_wfs[1].setText(
            tr(f"{self.vm.comfy_workflows_count} 套模板", f"{self.vm.comfy_workflows_count} templates")
        )

        # Cloud outputs list
        out_sig = (get_lang(), tuple((o.rel_path, o.mtime) for o in self.vm.comfy_recent_outputs))
        if out_sig != self._last_cfy_out_sig:
            self._last_cfy_out_sig = out_sig
            self._clear_layout(self.comfy_outputs_lay)
            if not self.vm.comfy_recent_outputs:
                empty = QLabel(tr("云端 output 目录暂无新成片", "No new outputs in cloud output/ folder"))
                empty.setStyleSheet("color: #64748B; font-size: 11.5px; padding: 12px;")
                self.comfy_outputs_lay.addWidget(empty)
            else:
                for item in self.vm.comfy_recent_outputs:
                    self.comfy_outputs_lay.addWidget(self._build_comfy_output_row(item))
            self.comfy_outputs_lay.addStretch(1)

    def _build_comfy_pack_row(self, pack: ComfyPresetPackItem) -> QFrame:
        f = QFrame()
        f.setObjectName("SubtlePanel")
        f.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        f.setMinimumHeight(88)
        hl = QHBoxLayout(f)
        hl.setContentsMargins(12, 10, 12, 10)
        hl.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(4)
        t_row = QHBoxLayout()
        t_row.setSpacing(6)
        tl = QLabel(pack.title)
        tl.setStyleSheet("font-size: 12.5px; font-weight: 700;")
        t_row.addWidget(tl)
        is_dark = self.vm.dark_mode
        bg_col = "rgba(16,185,129,0.14)" if pack.is_primary_recommended else "rgba(59,130,246,0.14)"
        if pack.is_primary_recommended:
            fg_col = "#10B981" if is_dark else "#059669"
        else:
            fg_col = "#60A5FA" if is_dark else "#2563EB"
        t_row.addWidget(PillBadge(pack.badge, fg=fg_col, bg=bg_col))
        t_row.addStretch(1)
        left.addLayout(t_row)

        sc = QLabel(pack.scenario)
        sc.setWordWrap(True)
        sc.setStyleSheet("font-size: 11.5px; color: #64748B;")
        left.addWidget(sc)

        meta_prefix = tr(
            f"权重 {pack.installed_models}/{pack.total_models} · 约 {pack.approx_size_gb:.1f} GB  |  ",
            f"Weights {pack.installed_models}/{pack.total_models} · ~{pack.approx_size_gb:.1f} GB  |  ",
        )
        meta = QLabel(meta_prefix + " · ".join(pack.workflows))
        meta.setWordWrap(True)
        meta.setStyleSheet("font-family: Consolas, monospace; font-size: 10.5px; color: #64748B;")
        left.addWidget(meta)
        hl.addLayout(left, 1)

        right = QVBoxLayout()
        right.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        is_installing = self.vm.comfy_installing_pack_id == pack.id
        if pack.ready and not is_installing:
            open_b = set_btn_role(QPushButton(tr("打开画布直接使用", "Open in Canvas")), "success")
            open_b.clicked.connect(self._on_comfy_side_primary)
            rew_b = QPushButton(tr("重写中文模板", "Rewrite Templates"))
            rew_b.clicked.connect(lambda _c=False, pid=pack.id: self.vm.install_comfy_pack(pid, open_when_ready=False))
            right.addWidget(open_b)
            right.addWidget(rew_b)
        else:
            pct = int(self.vm.comfy_progress * 100)
            lbl = (
                tr(f"正在配齐 ({pct}%)...", f"Installing ({pct}%)...")
                if is_installing
                else tr("一键配齐模型与工作流", "1-Click Install Models & Workflows")
            )
            inst_b = set_btn_role(QPushButton(lbl), "primary")
            inst_b.setEnabled(not is_installing and not self.vm.comfy_downloading_weight)
            inst_b.clicked.connect(lambda _c=False, pid=pack.id: self.vm.install_comfy_pack(pid, open_when_ready=True))
            right.addWidget(inst_b)
        hl.addLayout(right)
        return f

    def _build_comfy_output_row(self, item: ComfyOutputMediaItem) -> QFrame:
        f = QFrame()
        f.setObjectName("SubtlePanel")
        hl = QHBoxLayout(f)
        hl.setContentsMargins(8, 6, 8, 6)
        hl.setSpacing(6)
        ic = tr("[视频]", "[Vid]") if item.media_type == "video" else tr("[图片]", "[Img]")
        lbl = QLabel(f"{ic} {item.filename}\n{item.size_mb:.1f} MB")
        lbl.setStyleSheet("font-size: 11px;")
        hl.addWidget(lbl, 1)
        pull_b = set_btn_role(QPushButton(tr("拉回", "Pull")), "primary")
        pull_b.setFixedWidth(56)
        pull_b.clicked.connect(lambda _c=False, it=item: self.vm.pull_comfy_output_media(it))
        hl.addWidget(pull_b)
        return f

    def _api_snippets(self) -> Dict[str, str]:
        base_host = self.vm.normalized_base_url or "https://xxxx.gradio.live"
        shown_host = self.vm.masked_url(base_host)
        oai_base_real = f"{base_host}/v1"
        oai_base_shown = f"{shown_host}/v1"
        key = self.vm.api_access_key if self.vm.api_key_enabled else "sk-no-key-required"

        if self.vm.selected_model_id == "qwen38_27b_gguf":
            repo = "unsloth/Qwen3.8-27B-GGUF"
        elif self.vm.selected_model_id == "qwen38_27b":
            repo = "Qwen/Qwen3.8-27B"
        elif "qwen38_27b_gguf" in self.vm.loaded_model_ids and "qwen38_27b" not in self.vm.loaded_model_ids:
            repo = "unsloth/Qwen3.8-27B-GGUF"
        else:
            repo = "Qwen/Qwen3.8-27B"

        py_msg = tr("帮我设计一个高并发网关架构", "Design a high-concurrency API gateway architecture")
        oai_curl_msg = tr("用三句话解释量子纠缠", "Explain quantum entanglement in three sentences")
        ant_curl_msg = tr("分析 Rust 所有权机制的核心优势", "Explain the core advantages of Rust ownership")
        ps_comment = tr("# PowerShell (Windows 推荐)", "# PowerShell (Recommended on Windows)")

        def make_py(b: str) -> str:
            return (
                "from openai import OpenAI\n\n"
                f'client = OpenAI(base_url="{b}", api_key="{key}")\n'
                "stream = client.chat.completions.create(\n"
                f'    model="{repo}",\n'
                f'    messages=[{{"role": "user", "content": "{py_msg}"}}],\n'
                "    stream=True,\n"
                ")\n"
                "for chunk in stream:\n"
                "    delta = chunk.choices[0].delta\n"
                '    if getattr(delta, "reasoning_content", None):\n'
                '        print(delta.reasoning_content, end="", flush=True)\n'
                "    if delta.content:\n"
                '        print(delta.content, end="", flush=True)'
            )

        oai_curl = (
            f"curl {oai_base_real}/chat/completions \\\n"
            '  -H "Content-Type: application/json" \\\n'
            f'  -H "Authorization: Bearer {key}" \\\n'
            f'  -d \'{{"model": "{repo}", "stream": true, "messages": [{{"role": "user", "content": "{oai_curl_msg}"}}]}}\''
        )
        ant_curl = (
            f"curl {base_host}/v1/messages \\\n"
            '  -H "Content-Type: application/json" \\\n'
            f'  -H "x-api-key: {key}" \\\n'
            '  -H "anthropic-version: 2023-06-01" \\\n'
            f'  -d \'{{"model": "{repo}", "max_tokens": 2048, "stream": true, "messages": [{{"role": "user", "content": "{ant_curl_msg}"}}]}}\''
        )
        claude_ps = (
            f'$env:ANTHROPIC_BASE_URL="{base_host}"\n'
            f'$env:ANTHROPIC_API_KEY="{key}"\n'
            f'$env:ANTHROPIC_MODEL="{repo}"\n'
            "claude"
        )
        claude_ps_shown = (
            f'{ps_comment}\n$env:ANTHROPIC_BASE_URL="{shown_host}"\n'
            f'$env:ANTHROPIC_API_KEY="{key}"\n'
            f'$env:ANTHROPIC_MODEL="{repo}"\n\n'
            f'# Git Bash / WSL\nexport ANTHROPIC_BASE_URL="{shown_host}"\n'
            f'export ANTHROPIC_API_KEY="{key}"\n'
            f'export ANTHROPIC_MODEL="{repo}"'
        )
        claude_sh = (
            f'export ANTHROPIC_BASE_URL="{base_host}"\n'
            f'export ANTHROPIC_API_KEY="{key}"\n'
            f'export ANTHROPIC_MODEL="{repo}"'
        )
        return {
            "repo": repo,
            "oai_base": oai_base_real,
            "oai_base_shown": oai_base_shown,
            "oai_chat": f"{oai_base_real}/chat/completions",
            "oai_chat_shown": f"{oai_base_shown}/chat/completions",
            "oai_py": make_py(oai_base_real),
            "oai_py_shown": make_py(oai_base_shown),
            "oai_curl": oai_curl,
            "ant_base": base_host,
            "ant_base_shown": shown_host,
            "ant_msg": f"{base_host}/v1/messages",
            "ant_msg_shown": f"{shown_host}/v1/messages",
            "ant_curl": ant_curl,
            "claude_ps": claude_ps,
            "claude_ps_shown": claude_ps_shown,
            "claude_sh": claude_sh,
        }

    def _refresh_api_gateway_cards(self) -> None:
        sn = self._api_snippets()
        self.api_req_count_lbl.setText(
            tr(f"累计 API 请求: {self.vm.api_request_count} 次", f"Total API Requests: {self.vm.api_request_count}")
        )
        self.api_auth_chk.setChecked(self.vm.api_key_enabled)
        self.api_key_row_w.setVisible(self.vm.api_key_enabled)
        if not self.api_key_input.hasFocus():
            self.api_key_input.setText(self.vm.api_access_key)

        disp_name = "Qwen3.8-27B-GGUF" if "GGUF" in sn["repo"] else "Qwen3.8-27B"
        if not self.vm.is_qwen_loaded_in_vram:
            self.api_coresident_hint.setText(
                tr(
                    f"提示：请先在左侧点击「加载到显存」装载 {disp_name}",
                    f"Tip: Click 'Load to VRAM' on the left to load {disp_name} first",
                )
            )
            self.api_coresident_hint.setStyleSheet("color: #F59E0B; font-size: 11.5px;")
        elif len(self.vm.loaded_model_ids) > 1:
            self.api_coresident_hint.setText(
                tr(
                    f"多模型共驻已激活：外部 API 可随时调用 {disp_name}",
                    f"Multi-model co-residency active: external APIs can call {disp_name} anytime",
                )
            )
            self.api_coresident_hint.setStyleSheet("color: #10B981; font-size: 11.5px;")
        else:
            self.api_coresident_hint.setText(
                tr(f"{disp_name} 网关已就绪", f"{disp_name} API Gateway Ready")
            )
            self.api_coresident_hint.setStyleSheet("color: #10B981; font-size: 11.5px;")

        self.oai_base_lbl.setText(sn["oai_base_shown"])
        self.oai_chat_lbl.setText(sn["oai_chat_shown"])
        self.oai_model_lbl.setText(sn["repo"])
        self.oai_code_box.setPlainText(sn["oai_py_shown"])

        self.ant_base_lbl.setText(sn["ant_base_shown"])
        self.ant_msg_lbl.setText(sn["ant_msg_shown"])
        self.ant_code_box.setPlainText(sn["claude_ps_shown"])

    # ─── Telemetry, Console, Chat & Local Gallery Refreshers ─────────────────

    def _refresh_telemetry(self) -> None:
        pal = self.palette_obj
        v_col = metric_threshold_color("vram", self.vm.vram_free_gb, pal.metric_vram)
        r_col = metric_threshold_color("ram", self.vm.ram_free_gb, pal.metric_ram)
        d_col = metric_threshold_color("disk", self.vm.disk_free_gb, pal.metric_disk)

        if self.vm.combined_chart_mode:
            self.chart_mode_btn.setText(tr("分项", "Split"))
            self.charts_stack.setCurrentIndex(1)
            self.chart_combined.update_all(
                [
                    (tr("显存", "VRAM"), self.vm.vram_history_pct, self.vm.vram_used_gb, self.vm.vram_total_gb, v_col),
                    (tr("内存", "RAM"), self.vm.ram_history_pct, self.vm.ram_used_gb, self.vm.ram_total_gb, r_col),
                    (tr("磁盘", "Disk"), self.vm.disk_history_pct, self.vm.disk_used_gb, self.vm.disk_total_gb, d_col),
                ]
            )
        else:
            self.chart_mode_btn.setText(tr("同屏", "Overlay"))
            self.charts_stack.setCurrentIndex(0)
            self.chart_vram.update_metric(self.vm.vram_history_pct, self.vm.vram_used_gb, self.vm.vram_total_gb, v_col)
            self.chart_ram.update_metric(self.vm.ram_history_pct, self.vm.ram_used_gb, self.vm.ram_total_gb, r_col)
            self.chart_disk.update_metric(self.vm.disk_history_pct, self.vm.disk_used_gb, self.vm.disk_total_gb, d_col)

        parts = []
        if self.vm.shm_total_gb >= 1.0:
            parts.append(
                tr(
                    f"内存盘可用 {self.vm.shm_free_gb:.1f}/{self.vm.shm_total_gb:.0f}G",
                    f"RAM Disk Free {self.vm.shm_free_gb:.1f}/{self.vm.shm_total_gb:.0f}G",
                )
            )
        if self.vm.drive_total_gb > 0:
            parts.append(
                tr(
                    f"云盘可用 {self.vm.drive_free_gb:.1f}G",
                    f"Drive Free {self.vm.drive_free_gb:.1f}G",
                )
            )
        if parts:
            self.storage_footer_lbl.setText(" · ".join(parts))
            self.storage_footer_lbl.setVisible(True)
        else:
            self.storage_footer_lbl.setVisible(False)

    def _refresh_console(self) -> None:
        is_dark = self.vm.dark_mode
        html_lines = []
        for ln in self.vm.live_console_lines:
            esc = html.escape(ln)
            if any(k in ln for k in ("失败", "错误", "异常", "ERROR", "Error", "error:", "Failed", "failed")):
                col = "#FA7575" if is_dark else "#DC2626"
            elif "───" in ln:
                col = "#75CCFA" if is_dark else "#0284C7"
            elif any(k in ln for k in ("%", "已装载", "成功", "完成", "Loaded", "Success", "Complete", "complete", "ready.")):
                col = "#59EB8C" if is_dark else "#059669"
            else:
                col = "#DEDEDE" if is_dark else "#1E293B"
            html_lines.append(f'<div style="color:{col}; margin-bottom:2px; white-space:pre-wrap;">{esc}</div>')
        self.terminal_view.setHtml("".join(html_lines))
        sb = self.terminal_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _refresh_chat_conversation(self) -> None:
        self._clear_layout(self.chat_turns_container)
        if not self.vm.chat_history:
            empty = QLabel(
                tr(
                    "尚无对话记录。输入问题或拖入图片后按 Ctrl+Enter 即可开始推理。",
                    "No conversation yet. Enter a prompt or drop an image and press Ctrl+Enter to start reasoning.",
                )
            )
            empty.setStyleSheet("color: #64748B; font-size: 12px; padding: 14px;")
            self.chat_turns_container.addWidget(empty)
            return

        is_dark = self.vm.dark_mode
        user_col = "#60A5FA" if is_dark else "#2563EB"
        asst_col = "#34D399" if is_dark else "#059669"
        for turn in self.vm.chat_history:
            box = QFrame()
            box.setObjectName("SubtlePanel")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(12, 10, 12, 10)
            bl.setSpacing(6)

            hdr = QHBoxLayout()
            if turn.role == "user":
                rl = QLabel(tr("提问", "You"))
                rl.setStyleSheet(f"font-weight: 700; color: {user_col}; font-size: 12px;")
                hdr.addWidget(rl)
                if turn.has_image:
                    hdr.addWidget(
                        PillBadge(tr("附带视觉图像", "With Image"), fg="#059669", bg="rgba(16,185,129,0.14)")
                    )
                hdr.addStretch(1)
            else:
                rl = QLabel(f"{self.vm.active_model_spec.name}")
                rl.setStyleSheet(f"font-weight: 700; color: {asst_col}; font-size: 12px;")
                hdr.addWidget(rl)
                hdr.addStretch(1)
                cp = set_btn_role(QPushButton(tr("复制回复", "Copy Reply")), "ghost")
                cp.setStyleSheet("font-size: 11px;")
                cp.clicked.connect(
                    lambda _c=False, txt=turn.content: self._copy_text(
                        txt,
                        tr("已复制模型回复到剪贴板。", "Copied model reply to clipboard."),
                    )
                )
                hdr.addWidget(cp)
            bl.addLayout(hdr)

            if turn.role == "assistant" and turn.thinking:
                think_box = CollapsibleBox(
                    tr(f"深度思考过程 ({len(turn.thinking)} 字)", f"Thinking Process ({len(turn.thinking)} chars)"),
                    expanded=(not bool(turn.content)),
                )
                th_lbl = QLabel(turn.thinking)
                th_lbl.setWordWrap(True)
                th_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                th_lbl.setStyleSheet("font-size: 11.5px; color: #64748B; padding: 4px;")
                think_box.content_layout.addWidget(th_lbl)
                bl.addWidget(think_box)

            content_txt = (
                turn.content
                if turn.content
                else (tr("正在思考与生成回复...", "Thinking & generating reply...") if turn.role == "assistant" else "")
            )
            c_lbl = QLabel(content_txt)
            c_lbl.setWordWrap(True)
            c_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            c_lbl.setStyleSheet("font-size: 12.5px; line-height: 1.45;")
            bl.addWidget(c_lbl)
            self.chat_turns_container.addWidget(box)

    def _refresh_media_gallery(self) -> None:
        self._clear_layout(self.local_gallery_lay)
        if not self.vm.local_media_items:
            empty = QLabel(tr("本地目录暂无历史作品", "No local outputs yet"))
            empty.setStyleSheet("color: #64748B; font-size: 11.5px; padding: 12px;")
            self.local_gallery_lay.addWidget(empty)
        else:
            for item in self.vm.local_media_items[:60]:
                ic = tr("[图片]", "[Img]") if item.is_image else tr("[视频]", "[Vid]")
                ts_str = datetime.fromtimestamp(item.created_ts).strftime("%m-%d %H:%M") if item.created_ts > 0 else ""
                el_str = f" · {format_duration(item.elapsed_seconds, compact=True)}" if item.elapsed_seconds else ""
                btn = QPushButton(f"{ic} {item.filename[:22]}\n{ts_str}{el_str}")
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setStyleSheet("QPushButton { text-align: left; padding: 6px 8px; font-size: 11px; }")
                btn.clicked.connect(lambda _c=False, p=item.path: self.vm.select_local_media(p, auto_play=True))
                self.local_gallery_lay.addWidget(btn)
        self.local_gallery_lay.addStretch(1)

    def _on_media_selected(self, path_obj: object, elapsed_obj: object, auto_play: bool) -> None:
        p = path_obj if isinstance(path_obj, Path) else None
        el = float(elapsed_obj) if isinstance(elapsed_obj, (int, float)) else None
        self.media_preview.load_media(p, elapsed_seconds=el, auto_play=auto_play)

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                EasyColabMainWindow._clear_layout(item.layout())
