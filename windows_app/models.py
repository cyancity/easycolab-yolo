"""
Data models, static catalogs, resolution presets, ComfyUI workflow packs,
i18n translation helpers, and path/string utilities for the EasyColab Windows Client
(1:1 parity with macOS, defaulting to Simplified Chinese "zh").
"""
from __future__ import annotations

import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── Core i18n State & Translation Helpers (Default: Simplified Chinese "zh") ─

_CURRENT_LANG: str = "zh"


def get_lang() -> str:
    return _CURRENT_LANG


def is_en() -> bool:
    return _CURRENT_LANG == "en"


def set_lang(lang: str) -> None:
    global _CURRENT_LANG
    _CURRENT_LANG = "en" if str(lang).strip().lower().startswith("en") else "zh"


def tr(zh: str, en: str) -> str:
    return en if _CURRENT_LANG == "en" else zh


_SHARD_DESC_EXACT: Dict[str, str] = {
    "3D 视频 VAE BF16 (9.7 GB)": "3D Video VAE BF16 (9.7 GB)",
    "32kHz 立体声 VAE BF16 (0.6 GB)": "32kHz Stereo Audio VAE BF16 (0.6 GB)",
    "Unsloth UD-Q4_K_M 主模型 (16.5 GB)": "Unsloth UD-Q4_K_M Main Weights (16.5 GB)",
    "多模态视觉投影器 (0.93 GB)": "Multimodal Vision Projector (0.93 GB)",
    "Unsloth MTP 多词元投机解码头 (1.31 GB)": "Unsloth MTP Speculative Head (1.31 GB)",
    "图像 VAE": "Image VAE",
}

_SHARD_DESC_PREFIXES: List[Tuple[str, str]] = [
    ("DiT 音视频主干 BF16 ", "DiT A/V Backbone BF16 "),
    ("多模态编码器 BF16 ", "Multimodal Encoder BF16 "),
    ("权重分片 ", "Weight Shard "),
    ("编码器 ", "Text Encoder "),
]


def localize_shard_desc(desc: Optional[str]) -> Optional[str]:
    if desc is None or not is_en() or not desc:
        return desc
    if desc in _SHARD_DESC_EXACT:
        return _SHARD_DESC_EXACT[desc]
    out = desc
    for zh_p, en_p in _SHARD_DESC_PREFIXES:
        if out.startswith(zh_p):
            return en_p + out[len(zh_p) :]
    return out


_SERVER_EXACT_MAP: Dict[str, str] = {
    "尚未连接云端 GPU": "Cloud GPU Not Connected",
    "云端 ComfyUI 尚未启动": "Cloud ComfyUI not started",
    "Google Drive 自动软链持久化": "Google Drive Auto-Symlink Persistence",
    "Google Drive 实时软链挂载": "Google Drive Live Symlink Mounted",
    "Google Drive 云盘自动持久化": "Google Drive Cloud Auto-Persistence",
    "实例本地高速存储": "Instance Local High-Speed Storage",
    "尚未启动（支持一键拉起 ComfyUI + Google Drive 自动软链持久化）": (
        "Not started (1-click ComfyUI + Google Drive symlink persistence)"
    ),
    "ComfyUI 已停止（节点、模型与工作流已持久化保存至 Google Drive）": (
        "ComfyUI stopped (nodes, models & workflows persisted to Google Drive)"
    ),
    "正在初始化 ComfyUI 环境与 Google Drive 持久化目录...": (
        "Initializing ComfyUI environment & Google Drive persistence directories..."
    ),
    "正在释放现有常驻模型显存，为 ComfyUI 腾出完整 96GB 空间...": (
        "Releasing co-resident VRAM models to free full 96GB for ComfyUI..."
    ),
    "正在从 Google Drive 恢复历史持久化的 ComfyUI 节点、模型与工作流...": (
        "Restoring persisted ComfyUI custom nodes, models & workflows from Google Drive..."
    ),
    "正在克隆官方最新 ComfyUI 核心引擎...": "Cloning official ComfyUI core engine...",
    "正在安装 ComfyUI 核心依赖（保留当前 G4 Blackwell CUDA PyTorch）...": (
        "Installing ComfyUI core dependencies (preserving CUDA PyTorch)..."
    ),
    "正在恢复已持久化的自定义节点并配置 ComfyUI-Manager...": (
        "Restoring persisted custom nodes & configuring ComfyUI-Manager..."
    ),
    "正在检查并安装已保存自定义节点的 Python 依赖包...": (
        "Checking & installing Python dependencies for custom nodes..."
    ),
    "正在拉起 ComfyUI 节点推理引擎 (端口 8188)...": "Starting ComfyUI node engine (port 8188)...",
    "ComfyUI 内核已就绪，正在建立 Cloudflare 高速全屏画布隧道...": (
        "ComfyUI engine ready, opening Cloudflare fullscreen canvas tunnel..."
    ),
    "正在先从 Google Drive 检查已持久化的历史模型权重...": (
        "Checking persisted model weights in Google Drive..."
    ),
    "正在将配齐的模型与中文工作流持久化同步至 Google Drive...": (
        "Syncing installed models & workflows to Google Drive..."
    ),
    "正在预装并校验 llama.cpp CUDA 高速推理引擎...": "Verifying llama.cpp CUDA inference engine...",
    "正在映射 MiniMax-H3 BF16 多模态编码器权重 (14 分片)...": (
        "Mapping MiniMax-H3 BF16 multimodal encoder weights (14 shards)..."
    ),
    "正在映射 MiniMax-H3 BF16 33B DiT 主干权重 (13 分片)...": (
        "Mapping MiniMax-H3 BF16 33B DiT backbone weights (13 shards)..."
    ),
    "正在映射 3D 视频 VAE 与 32kHz 立体声 VAE 权重...": (
        "Mapping 3D Video VAE & 32kHz Stereo Audio VAE weights..."
    ),
    "正在构建 MiniMax-H3 BF16 原生 GPU 计算图与 Processor...": (
        "Building MiniMax-H3 native BF16 GPU graph & Processor..."
    ),
    "正在映射 Qwen-Image-2.1 权重并构建 GPU 管线...": "Mapping Qwen-Image-2.1 weights & building GPU pipeline...",
    "正在映射 Z-Image-Turbo 权重并构建 8 步极速生图管线...": (
        "Mapping Z-Image-Turbo weights & building 8-step turbo pipeline..."
    ),
    "正在预热 Qwen3.8-27B-UD-Q4_K_M.gguf 与 mmproj 视觉权重...": (
        "Warming up Qwen3.8-27B-UD-Q4_K_M.gguf & mmproj vision weights..."
    ),
    "正在自动补齐 Unsloth MTP 投机解码加速头 (1.31 GB)...": (
        "Fetching Unsloth MTP speculative decoding head (1.31 GB)..."
    ),
    "正在检查账号下已有的 Colab 实例...": "Checking existing Colab runtimes on your account...",
    "旧实例已失效或正在销毁，正在自动清理并重新分配新实例...": (
        "Previous runtime expired; cleaning up and allocating a fresh runtime..."
    ),
    "容器仍在初始化，继续尝试连接内核...": "Container still initializing, retrying kernel connection...",
    "正在上传最新多模型开源服务端代码 (app.py)...": "Uploading multi-model server engine (app.py)...",
    "正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)...": (
        "Configuring Google Drive persistence, engines & tunnel (~40-90s)..."
    ),
    "尚未授权登录 Google 账号，请先点击「绑定账号」": "Google account not authorized. Please click Bind first.",
    "已释放全部云端实例": "All cloud runtimes released",
    "已释放全部云端实例。": "All cloud runtimes released.",
}

_SERVER_REGEX_RULES: List[Tuple[re.Pattern[str], object]] = [
    (
        re.compile(r"^已无缝接管在线 ComfyUI 画布（(.+?)）$"),
        lambda m: f"Reconnected to active ComfyUI canvas ({localize_server_text(m.group(1))})",
    ),
    (
        re.compile(r"^ComfyUI 进程已退出 \(code=([^)]+)\)，可点击重新启动$"),
        r"ComfyUI process exited (code=\1). Click to restart.",
    ),
    (
        re.compile(r"^正在建立 (.+?) 目录软链 \(models / workflows / output\)\.\.\.$"),
        lambda m: f"Linking {localize_server_text(m.group(1))} directories (models / workflows / output)...",
    ),
    (
        re.compile(r"^正在 8 线程极速下载 (.+?) 至 (.+?)\.\.\.$"),
        r"8-thread downloading \1 to \2...",
    ),
    (
        re.compile(r"^正在下载 (.+?) 至 (.+?)\.\.\.$"),
        r"Downloading \1 to \2...",
    ),
    (
        re.compile(r"^正在准备配齐「(.+?)」（共 (\d+) 个官方权重）\.\.\.$"),
        r'Preparing workflow pack "\1" (\2 official weights)...',
    ),
    (
        re.compile(r"^正在生成「(.+?)」全中文三步傻瓜注释工作流\.\.\.$"),
        r'Generating guided workflow templates for "\1"...',
    ),
    (
        re.compile(r"^正在为 (.+?) 启动 8 路并发 HTTP Range 极速下载通道\.\.\.$"),
        r"Starting 8-thread HTTP Range downloader for \1...",
    ),
    (
        re.compile(r"^\[(\d+/\d+)\] (.+?) 已就绪 \((.+?)\)，跳过下载$"),
        r"[\1] \2 ready (\3), skipping",
    ),
    (
        re.compile(r"^\[(\d+/\d+)\] 正在 8 线程并发下载 (.+?) \((.+?)\)\.\.\.$"),
        lambda m: f"[{m.group(1)}] 8-thread downloading {m.group(2)} ({localize_shard_desc(m.group(3))})...",
    ),
    (
        re.compile(r"^正在获取 (.+?) 的分词器与 Processor 配置文件\.\.\.$"),
        r"Fetching tokenizer & processor configs for \1...",
    ),
    (
        re.compile(r"^正在评估显存余量并装载 (.+?) 至 GPU\.\.\.$"),
        r"Checking free VRAM and loading \1 onto GPU...",
    ),
    (
        re.compile(r"^\[显存常驻加速\] 正在将子模块 (.+?) 预热锁入 GPU 显存\.\.\.$"),
        r"[VRAM Pinning] Locking submodule \1 into GPU VRAM...",
    ),
    (
        re.compile(r"^正在向 Google Colab 申请分配 (.+?) 实例\.\.\.$"),
        r"Requesting Google Colab \1 runtime...",
    ),
    (
        re.compile(r"^成功分配 (.+?) 云端虚拟机！正在等待容器内核就绪\.\.\.$"),
        r"Allocated \1 cloud VM! Waiting for kernel startup...",
    ),
    (
        re.compile(r"^当前已分配实例型号为 \[(.+?)\]，正在释放并切换为您选择的 \[(.+?)\]\.\.\.$"),
        r"Current runtime is [\1], releasing and switching to [\2]...",
    ),
    (
        re.compile(r"^正在检查已有实例 \[(.+?)\] \((.+?)\) 连通性\.\.\.$"),
        r"Checking connectivity of existing runtime [\1] (\2)...",
    ),
    (
        re.compile(r"^启动超时，日志尾部:\s*(.*)$"),
        r"Startup timed out. Recent logs: \1",
    ),
]


def localize_server_text(text: str) -> str:
    """Translate common backend / bridge status strings when UI is in English mode."""
    if not text or not is_en():
        return text
    stripped = text.strip()
    if stripped in _SERVER_EXACT_MAP:
        return _SERVER_EXACT_MAP[stripped]
    for pattern, repl in _SERVER_REGEX_RULES:
        if pattern.search(stripped):
            return pattern.sub(repl, stripped)  # type: ignore[arg-type]
    out = stripped
    for zh_p, en_p in _SHARD_DESC_PREFIXES:
        if zh_p in out:
            out = out.replace(zh_p, en_p)
    for zh_exact, en_exact in _SHARD_DESC_EXACT.items():
        if zh_exact in out:
            out = out.replace(zh_exact, en_exact)
    return out


# ─── Data Models & Dynamic Bilingual Catalogs ────────────────────────────────


class ModelModality(str, Enum):
    VIDEO_AUDIO = "video_audio"
    VIDEO = "video"
    IMAGE = "image"
    LLM_VLM = "llm_vlm"
    COMFY_UI = "comfyui"


@dataclass(frozen=True)
class TaskModeItem:
    id: str
    title_zh: str
    title_en: str
    subtitle_zh: str
    subtitle_en: str
    icon: str

    @property
    def title(self) -> str:
        return tr(self.title_zh, self.title_en)

    @property
    def subtitle(self) -> str:
        return tr(self.subtitle_zh, self.subtitle_en)


class ModelFileItem:
    def __init__(
        self,
        name: str,
        desc: Optional[str] = None,
        exists: bool = False,
        size_gb: float = 0.0,
        desc_en: Optional[str] = None,
    ) -> None:
        self.name: str = name
        self._desc_zh: Optional[str] = desc
        self._desc_en: Optional[str] = desc_en
        self.exists: bool = exists
        self.size_gb: float = size_gb

    @property
    def desc(self) -> Optional[str]:
        if is_en():
            return self._desc_en if self._desc_en else localize_shard_desc(self._desc_zh)
        return self._desc_zh

    def clone(self) -> "ModelFileItem":
        return ModelFileItem(
            name=self.name,
            desc=self._desc_zh,
            exists=self.exists,
            size_gb=self.size_gb,
            desc_en=self._desc_en,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ModelFileItem":
        return cls(
            name=str(d.get("name", "")),
            desc=d.get("desc"),
            exists=bool(d.get("exists", False)),
            size_gb=float(d.get("size_gb", 0.0) or 0.0),
        )


@dataclass
class OpenModelSpec:
    id: str
    name_zh: str
    name_en: str
    repo_id: str
    badge_zh: str
    badge_en: str
    summary_zh: str
    summary_en: str
    approx_size_gb: float
    icon: str
    modality: ModelModality
    default_steps: int
    modes: List[TaskModeItem]
    default_files: List[ModelFileItem]

    @property
    def name(self) -> str:
        return tr(self.name_zh, self.name_en)

    @property
    def badge(self) -> str:
        return tr(self.badge_zh, self.badge_en)

    @property
    def summary(self) -> str:
        return tr(self.summary_zh, self.summary_en)


@dataclass(frozen=True)
class ResolutionOption:
    id: str
    label_zh: str
    label_en: str
    width: int
    height: int

    @property
    def label(self) -> str:
        return tr(self.label_zh, self.label_en)


@dataclass(frozen=True)
class PresetTemplate:
    modality: ModelModality
    title_zh: str
    title_en: str
    visual_zh: str
    visual_en: str = ""
    soundscape: str = ""
    music: str = ""

    @property
    def title(self) -> str:
        return tr(self.title_zh, self.title_en)

    @property
    def visual(self) -> str:
        return tr(self.visual_zh, self.visual_en or self.visual_zh)


@dataclass
class LoadedModelSummary:
    id: str
    name: str
    mode: Optional[str] = None
    vram_gb: Optional[float] = None
    context_length: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "LoadedModelSummary":
        raw_id = str(d.get("id", ""))
        raw_name = strip_emoji(str(d.get("name", "")))
        if raw_id in MODEL_SPEC_MAP:
            raw_name = MODEL_SPEC_MAP[raw_id].name
        return cls(
            id=raw_id,
            name=raw_name,
            mode=strip_emoji(str(d["mode"])) if d.get("mode") else None,
            vram_gb=float(d["vram_gb"]) if d.get("vram_gb") is not None else None,
            context_length=int(d["context_length"]) if d.get("context_length") is not None else None,
        )


@dataclass
class ComfyOutputMediaItem:
    rel_path: str
    filename: str
    media_type: str = "video"
    size_mb: float = 0.0
    mtime: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "ComfyOutputMediaItem":
        return cls(
            rel_path=str(d.get("rel_path", "")),
            filename=str(d.get("filename", "")),
            media_type=str(d.get("media_type") or "video"),
            size_mb=float(d.get("size_mb") or 0.0),
            mtime=int(d.get("mtime") or 0),
        )


_COMFY_PACK_I18N: Dict[str, Tuple[str, str, str, str, str, str]] = {
    "wan22_i2v_flf2v": (
        "万相 Wan2.2 14B · 图生视频 + 首尾帧双工作流",
        "Wan2.2 14B · Image-to-Video + First/Last Frame Dual Workflows",
        "新手首选 · 4步极速",
        "Beginner Pick · 4-Step Fast",
        "含「单图动起来」与「首尾双图丝滑过渡」2 套中文画布，G4 约 25 秒出片，纯官方内置节点零红框报错",
        "Includes Single-Image I2V & First/Last-Frame Smooth Transition workflows (~25s on G4, pure built-in nodes, zero missing-node errors)",
    ),
    "minimax_h3_i2v_flf2v": (
        "MiniMax-H3 Turbo · 图生有声视频 + 首尾帧双工作流",
        "MiniMax-H3 Turbo · Audio-Video I2V + First/Last Frame Workflows",
        "自带立体声音效 · 8步加速",
        "Synced Stereo Audio · 8-Step",
        "含「单图生有声视频」与「首尾双图有声过渡」2 套极简画布，一次生成高清动态画面 + 同步环境音效/配乐/对白",
        "Includes Single-Image & First/Last-Frame Audio-Video workflows generating HD motion with synced ambient SFX, music & dialogue",
    ),
    "ltx23_flf2v": (
        "LTX-2.3 (22B) · 首尾帧电影级运镜有声工作流",
        "LTX-2.3 (22B) · Cinematic First/Last-Frame Camera Control + Audio",
        "电影推拉摇移 · 仅 2 个权重",
        "Cinematic Camera · Only 2 Weights",
        "专为首尾关键帧大幅度镜头运动优化（如高空俯拍平滑推镜至低角度特写 + 自动合成配乐音轨）",
        "Optimized for dramatic camera motion between keyframes (e.g., aerial crane dive to low-angle close-up with synced soundtrack)",
    ),
    "hunyuan15_i2v_1080p": (
        "混元 HunyuanVideo 1.5 · 单图直出 1080P 超分工作流",
        "HunyuanVideo 1.5 · Single-Image Direct 1080P Super-Res Workflow",
        "720P + 1080P 双路超清",
        "720P + 1080P Dual Output",
        "已默认激活潜空间超分管线，传 1 张图一键自动渲染 720P 原片与 1920×1080 超高清成片",
        "Latent super-resolution pipeline enabled by default — upload 1 image to render both 720P base and crisp 1920×1080 HD video",
    ),
}


@dataclass
class ComfyPresetPackItem:
    id: str
    _title: str
    _badge: str
    _scenario: str
    approx_size_gb: float
    workflows: List[str]
    installed_models: int = 0
    total_models: int = 0
    workflows_installed: bool = False
    ready: bool = False

    @property
    def title(self) -> str:
        entry = _COMFY_PACK_I18N.get(self.id)
        if entry:
            return tr(entry[0], entry[1])
        return localize_server_text(self._title)

    @property
    def badge(self) -> str:
        entry = _COMFY_PACK_I18N.get(self.id)
        if entry:
            return tr(entry[2], entry[3])
        return localize_server_text(self._badge)

    @property
    def scenario(self) -> str:
        entry = _COMFY_PACK_I18N.get(self.id)
        if entry:
            return tr(entry[4], entry[5])
        return localize_server_text(self._scenario)

    @property
    def is_primary_recommended(self) -> bool:
        return self.id == "wan22_i2v_flf2v"

    @classmethod
    def from_dict(cls, d: dict) -> "ComfyPresetPackItem":
        return cls(
            id=str(d.get("id", "")),
            _title=str(d.get("title", "")),
            _badge=str(d.get("badge", "")),
            _scenario=str(d.get("scenario", "")),
            approx_size_gb=float(d.get("approx_size_gb") or 0.0),
            workflows=list(d.get("workflows") or []),
            installed_models=int(d.get("installed_models") or 0),
            total_models=int(d.get("total_models") or 0),
            workflows_installed=bool(d.get("workflows_installed", False)),
            ready=bool(d.get("ready", False)),
        )


@dataclass
class LocalMediaItem:
    path: Path
    filename: str
    is_image: bool
    created_ts: float
    elapsed_seconds: Optional[float] = None


@dataclass
class ChatTurnItem:
    role: str  # "user" | "assistant"
    content: str
    thinking: str = ""
    has_image: bool = False
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ─── Static Catalogs (1:1 Parity with macOS EasyColabStudio.swift) ────────────

OPEN_MODEL_CATALOG: List[OpenModelSpec] = [
    OpenModelSpec(
        id="minimax_h3",
        name_zh="MiniMax-H3 (33B)",
        name_en="MiniMax-H3 (33B)",
        repo_id="MiniMaxAI/MiniMax-H3",
        badge_zh="BF16 原生 · 134 GB",
        badge_en="Native BF16 · 134 GB",
        summary_zh="全模态 BF16 原生满血精度同步生成高清视频与 32kHz 立体声音效、人声对白及配乐",
        summary_en="Full-modality native BF16 generation of HD video with synced 32kHz stereo SFX, dialogue & music",
        approx_size_gb=134.1,
        icon="film.stack",
        modality=ModelModality.VIDEO_AUDIO,
        default_steps=35,
        modes=[
            TaskModeItem(
                id="t2va",
                title_zh="文生音视频",
                title_en="Text to Video+Audio",
                subtitle_zh="输入镜头与声音描述，同步生成视频与立体声音轨",
                subtitle_en="Describe visuals and sound to generate video with synced stereo audio",
                icon="film",
            ),
            TaskModeItem(
                id="fl2va",
                title_zh="首尾帧生视频",
                title_en="First/Last Frame to Video",
                subtitle_zh="上传首帧或尾帧参考图，精准控制起止画面与运镜过渡",
                subtitle_en="Upload start/end reference frames for precise composition & camera transitions",
                icon="photo.on.rectangle.angled",
            ),
        ],
        default_files=(
            [
                ModelFileItem(
                    name=f"FL2VA/transformer/model-{i:05d}-of-00013.safetensors",
                    desc=f"DiT 音视频主干 BF16 {i}/13",
                    desc_en=f"DiT A/V Backbone BF16 {i}/13",
                )
                for i in range(1, 14)
            ]
            + [
                ModelFileItem(
                    name=f"FL2VA/text_encoder/model-{i:05d}-of-00014.safetensors",
                    desc=f"多模态编码器 BF16 {i}/14",
                    desc_en=f"Multimodal Encoder BF16 {i}/14",
                )
                for i in range(1, 15)
            ]
            + [
                ModelFileItem(
                    name="FL2VA/video_vae/source/model.safetensors",
                    desc="3D 视频 VAE BF16 (9.7 GB)",
                    desc_en="3D Video VAE BF16 (9.7 GB)",
                ),
                ModelFileItem(
                    name="FL2VA/audio_vae/model.safetensors",
                    desc="32kHz 立体声 VAE BF16 (0.6 GB)",
                    desc_en="32kHz Stereo Audio VAE BF16 (0.6 GB)",
                ),
            ]
        ),
    ),
    OpenModelSpec(
        id="comfyui_studio",
        name_zh="ComfyUI 节点工作台",
        name_en="ComfyUI Node Studio",
        repo_id="comfyanonymous/ComfyUI",
        badge_zh="节点视频创作 · 云盘软链持久化",
        badge_en="Node Workflows · Drive Symlink",
        summary_zh="一键拉起云端 ComfyUI + Manager 全屏节点画布，自定义节点、视频权重与工作流自动软链持久化至 Google Drive",
        summary_en="One-click cloud ComfyUI + Manager fullscreen node canvas with auto Google Drive symlink persistence for nodes, models & workflows",
        approx_size_gb=0.0,
        icon="point.3.connected.trianglepath.dotted",
        modality=ModelModality.COMFY_UI,
        default_steps=25,
        modes=[
            TaskModeItem(
                id="comfy_canvas",
                title_zh="全屏节点画布",
                title_en="Fullscreen Node Canvas",
                subtitle_zh="一键部署并跳转浏览器全屏节点画布，复用 G4 保活、折线监控与自动休眠",
                subtitle_en="Deploy & open fullscreen node canvas in browser with G4 keepalive, telemetry & auto-sleep",
                icon="macwindow.badge.plus",
            ),
            TaskModeItem(
                id="comfy_assets",
                title_zh="极速下权重与回传成片",
                title_en="Fast Weight DL & Output Pull",
                subtitle_zh="8 线程极速下载视频模型至云盘 models 目录，并将 output 成片一键回传本机",
                subtitle_en="8-thread high-speed model download to Drive and 1-click output media sync to local PC",
                icon="externaldrive.badge.icloud",
            ),
        ],
        default_files=[],
    ),
    OpenModelSpec(
        id="qwen38_27b",
        name_zh="Qwen3.8-27B",
        name_en="Qwen3.8-27B",
        repo_id="Qwen/Qwen3.8-27B",
        badge_zh="BF16 原生 · 51.8 GB",
        badge_en="Native BF16 · 51.8 GB",
        summary_zh="Hugging Face 榜首开源多模态大模型（BF16 原生满血精度，支持最高 256K 上下文，需大显存实例）",
        summary_en="Top open-source multimodal LLM in full native BF16 precision with up to 256K context (high-VRAM GPU)",
        approx_size_gb=51.8,
        icon="brain",
        modality=ModelModality.LLM_VLM,
        default_steps=20,
        modes=[
            TaskModeItem(
                id="chat",
                title_zh="深度推理对话",
                title_en="Deep Reasoning Chat",
                subtitle_zh="支持长思维链推理、代码架构、复杂逻辑与分镜脚本创作",
                subtitle_en="Long chain-of-thought reasoning, architecture design, logic & storyboard writing",
                icon="bubble.left.and.bubble.right",
            ),
            TaskModeItem(
                id="vlm",
                title_zh="图像视觉理解",
                title_en="Vision Understanding",
                subtitle_zh="拖入图片进行画面解析、OCR 识别或反推专业生图/生视频提示词",
                subtitle_en="Drop images for scene analysis, OCR, or reverse-engineering image/video prompts",
                icon="eye",
            ),
            TaskModeItem(
                id="api",
                title_zh="API 协议网关",
                title_en="API Protocol Gateway",
                subtitle_zh="提供 OpenAI (/v1) 与 Anthropic (/v1/messages) 双协议公网接口",
                subtitle_en="Dual-protocol public gateway supporting OpenAI (/v1) & Anthropic (/v1/messages)",
                icon="network",
            ),
        ],
        default_files=[
            ModelFileItem(
                name=f"model-{i:05d}-of-00018.safetensors",
                desc=f"权重分片 {i}/18",
                desc_en=f"Weight Shard {i}/18",
            )
            for i in range(1, 19)
        ],
    ),
    OpenModelSpec(
        id="qwen38_27b_gguf",
        name_zh="Qwen3.8-27B-GGUF",
        name_en="Qwen3.8-27B-GGUF",
        repo_id="unsloth/Qwen3.8-27B-GGUF",
        badge_zh="UD-Q4_K_M + MTP · 18.7 GB",
        badge_en="UD-Q4_K_M + MTP · 18.7 GB",
        summary_zh="Unsloth 动态 4-bit 量化 + 原生 MTP 多词元投机解码加速，支持最高 256K 上下文与多模态视觉理解",
        summary_en="Unsloth dynamic 4-bit quantization + native MTP speculative decoding (~100 tok/s), up to 256K context & vision",
        approx_size_gb=18.7,
        icon="cpu",
        modality=ModelModality.LLM_VLM,
        default_steps=20,
        modes=[
            TaskModeItem(
                id="chat",
                title_zh="深度推理对话",
                title_en="Deep Reasoning Chat",
                subtitle_zh="UD-Q4_K_M + MTP 投机解码极速推理，支持长思维链、代码架构与超长上下文分析",
                subtitle_en="Ultra-fast UD-Q4_K_M + MTP speculative decoding for CoT reasoning, coding & long-context analysis",
                icon="bubble.left.and.bubble.right",
            ),
            TaskModeItem(
                id="vlm",
                title_zh="图像视觉理解",
                title_en="Vision Understanding",
                subtitle_zh="内置 mmproj-F16 视觉投影器，支持图片解析、OCR 与生图/生视频提示词反推",
                subtitle_en="Built-in mmproj-F16 vision projector for image analysis, OCR & prompt reverse-engineering",
                icon="eye",
            ),
            TaskModeItem(
                id="api",
                title_zh="API 协议网关",
                title_en="API Protocol Gateway",
                subtitle_zh="提供 OpenAI (/v1) 与 Anthropic (/v1/messages) 双协议公网接口",
                subtitle_en="Dual-protocol public gateway supporting OpenAI (/v1) & Anthropic (/v1/messages)",
                icon="network",
            ),
        ],
        default_files=[
            ModelFileItem(
                name="Qwen3.8-27B-UD-Q4_K_M.gguf",
                desc="Unsloth UD-Q4_K_M 主模型 (16.5 GB)",
                desc_en="Unsloth UD-Q4_K_M Main Weights (16.5 GB)",
            ),
            ModelFileItem(
                name="mmproj-F16.gguf",
                desc="多模态视觉投影器 (0.93 GB)",
                desc_en="Multimodal Vision Projector (0.93 GB)",
            ),
            ModelFileItem(
                name="MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
                desc="Unsloth MTP 多词元投机解码头 (1.31 GB)",
                desc_en="Unsloth MTP Speculative Head (1.31 GB)",
            ),
        ],
    ),
    OpenModelSpec(
        id="qwen_image_21",
        name_zh="Qwen-Image-2.1",
        name_en="Qwen-Image-2.1",
        repo_id="Qwen/Qwen-Image-2.1",
        badge_zh="文生图与修图 · 30.9 GB",
        badge_en="T2I & Image Edit · 30.9 GB",
        summary_zh="通义千问新一代旗舰图像生成与自然语言多图指令编辑模型",
        summary_en="Flagship Qwen text-to-image generation and natural-language multi-image editing model",
        approx_size_gb=30.9,
        icon="paintbrush.pointed",
        modality=ModelModality.IMAGE,
        default_steps=40,
        modes=[
            TaskModeItem(
                id="t2i",
                title_zh="高清文生图",
                title_en="HD Text to Image",
                subtitle_zh="复杂构图、精准中英文排版与高质感摄影/插画生成",
                subtitle_en="Complex composition, accurate bilingual typography & studio-grade photography/illustration",
                icon="photo.artframe",
            ),
            TaskModeItem(
                id="i2i_edit",
                title_zh="多图指令修图",
                title_en="Multi-Image Instruction Edit",
                subtitle_zh="上传 1~2 张原图，用自然语言指令改局部、换背景或融合主体",
                subtitle_en="Upload 1-2 reference images to edit regions, swap backgrounds, or fuse subjects via natural language",
                icon="wand.and.rays",
            ),
        ],
        default_files=[
            ModelFileItem(name="transformer/diffusion_pytorch_model-00001-of-00002.safetensors", desc="DiT 1/2", desc_en="DiT 1/2"),
            ModelFileItem(name="transformer/diffusion_pytorch_model-00002-of-00002.safetensors", desc="DiT 2/2", desc_en="DiT 2/2"),
            ModelFileItem(name="text_encoder/model-00001-of-00004.safetensors", desc="编码器 1/4", desc_en="Text Encoder 1/4"),
            ModelFileItem(name="text_encoder/model-00002-of-00004.safetensors", desc="编码器 2/4", desc_en="Text Encoder 2/4"),
            ModelFileItem(name="text_encoder/model-00003-of-00004.safetensors", desc="编码器 3/4", desc_en="Text Encoder 3/4"),
            ModelFileItem(name="text_encoder/model-00004-of-00004.safetensors", desc="编码器 4/4", desc_en="Text Encoder 4/4"),
            ModelFileItem(name="vae/diffusion_pytorch_model.safetensors", desc="图像 VAE", desc_en="Image VAE"),
        ],
    ),
    OpenModelSpec(
        id="z_image_turbo",
        name_zh="Z-Image-Turbo",
        name_en="Z-Image-Turbo",
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        badge_zh="8步极速生图 · 30.6 GB",
        badge_en="8-Step Turbo T2I · 30.6 GB",
        summary_zh="Hugging Face 高人气 8 步蒸馏极速写实摄影与中英文海报生图模型",
        summary_en="Popular 8-step distilled photorealistic & bilingual poster image generation model",
        approx_size_gb=30.6,
        icon="bolt.square",
        modality=ModelModality.IMAGE,
        default_steps=8,
        modes=[
            TaskModeItem(
                id="t2i",
                title_zh="8 步极速文生图",
                title_en="8-Step Turbo T2I",
                subtitle_zh="仅需 8 步采样，秒级输出商业级人像摄影与中英文排版画面",
                subtitle_en="Commercial-grade portraits & bilingual poster typography in just 8 sampling steps",
                icon="bolt",
            ),
            TaskModeItem(
                id="i2i_edit",
                title_zh="图生图风格重绘",
                title_en="Image-to-Image Restyle",
                subtitle_zh="上传参考底图并调节重绘幅度，保留构图生成全新细节",
                subtitle_en="Upload a reference image and adjust denoising strength to restyle while preserving composition",
                icon="square.on.square",
            ),
        ],
        default_files=[
            ModelFileItem(name="transformer/diffusion_pytorch_model-00001-of-00003.safetensors", desc="DiT 1/3", desc_en="DiT 1/3"),
            ModelFileItem(name="transformer/diffusion_pytorch_model-00002-of-00003.safetensors", desc="DiT 2/3", desc_en="DiT 2/3"),
            ModelFileItem(name="transformer/diffusion_pytorch_model-00003-of-00003.safetensors", desc="DiT 3/3", desc_en="DiT 3/3"),
            ModelFileItem(name="text_encoder/model-00001-of-00003.safetensors", desc="编码器 1/3", desc_en="Text Encoder 1/3"),
            ModelFileItem(name="text_encoder/model-00002-of-00003.safetensors", desc="编码器 2/3", desc_en="Text Encoder 2/3"),
            ModelFileItem(name="text_encoder/model-00003-of-00003.safetensors", desc="编码器 3/3", desc_en="Text Encoder 3/3"),
            ModelFileItem(name="vae/diffusion_pytorch_model.safetensors", desc="图像 VAE", desc_en="Image VAE"),
        ],
    ),
]

MODEL_SPEC_MAP: Dict[str, OpenModelSpec] = {m.id: m for m in OPEN_MODEL_CATALOG}

VIDEO_RESOLUTIONS: List[ResolutionOption] = [
    ResolutionOption(
        "v_832_480",
        "832 × 480 (16:9 横屏标准 · 推荐)",
        "832 × 480 (16:9 Landscape · Recommended)",
        832,
        480,
    ),
    ResolutionOption(
        "v_480_832",
        "480 × 832 (9:16 竖屏短视频)",
        "480 × 832 (9:16 Vertical Short)",
        480,
        832,
    ),
    ResolutionOption(
        "v_1280_720",
        "1280 × 720 (16:9 720P 高清)",
        "1280 × 720 (16:9 720P HD)",
        1280,
        720,
    ),
    ResolutionOption(
        "v_720_1280",
        "720 × 1280 (9:16 720P 竖屏)",
        "720 × 1280 (9:16 720P Vertical)",
        720,
        1280,
    ),
    ResolutionOption(
        "v_640_640",
        "640 × 640 (1:1 方形)",
        "640 × 640 (1:1 Square)",
        640,
        640,
    ),
]

IMAGE_RESOLUTIONS: List[ResolutionOption] = [
    ResolutionOption(
        "i_1024_1024",
        "1024 × 1024 (1:1 超清方图 · 推荐)",
        "1024 × 1024 (1:1 Square HD · Recommended)",
        1024,
        1024,
    ),
    ResolutionOption(
        "i_1344_768",
        "1344 × 768 (16:9 电影横幅)",
        "1344 × 768 (16:9 Cinematic Wide)",
        1344,
        768,
    ),
    ResolutionOption(
        "i_768_1344",
        "768 × 1344 (9:16 竖版海报)",
        "768 × 1344 (9:16 Vertical Poster)",
        768,
        1344,
    ),
    ResolutionOption(
        "i_1536_1024",
        "1536 × 1024 (3:2 单反摄影)",
        "1536 × 1024 (3:2 DSLR Photo)",
        1536,
        1024,
    ),
]

PRESET_TEMPLATES: List[PresetTemplate] = [
    PresetTemplate(
        modality=ModelModality.VIDEO_AUDIO,
        title_zh="[动漫] 新海诚风 · 盛夏樱花铁道口",
        title_en="[Anime] Makoto Shinkai · Cherry Blossom Railway Crossing",
        visual_zh=(
            "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor "
            "school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past "
            "in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink "
            "cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, "
            "vibrant cinematic lighting, smooth 24fps keyframe animation."
        ),
        soundscape=(
            "Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning "
            "bell, warm spring wind rustling through leaves, a soft gasp from the girl."
        ),
        music="Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings.",
    ),
    PresetTemplate(
        modality=ModelModality.VIDEO_AUDIO,
        title_zh="[动漫] 飞碟社风 · 苍蓝雷光剑术对决",
        title_en="[Anime] ufotable Sakuga · Blue Lightning Sword Duel",
        visual_zh=(
            "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime "
            "swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, "
            "swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and "
            "glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur."
        ),
        soundscape=(
            "Sharp metallic sword unsheathing and clashing clang, roaring thunder and crackling electric arcs, "
            "heavy stone debris crumbling, fierce battle cry."
        ),
        music="Epic dark orchestral battle music with fast taiko drums, intense choir, and distorted electric guitar riffs.",
    ),
    PresetTemplate(
        modality=ModelModality.VIDEO_AUDIO,
        title_zh="[写实] 冰岛巨型瀑布 FPV 穿越机航拍",
        title_en="[Realistic] Icelandic Waterfall FPV Drone Dive",
        visual_zh=(
            "FPV drone shot diving down a majestic Icelandic waterfall surrounded by lush green moss and basalt "
            "columns under golden hour sunlight, mist rising into a rainbow."
        ),
        soundscape="Roaring waterfall rushing water, wind whooshing past the camera, distant seabirds calling.",
        music="Epic cinematic acoustic guitar and soaring cello.",
    ),
    PresetTemplate(
        modality=ModelModality.IMAGE,
        title_zh="[生图] 东方赛博霓虹商业海报（含中文排版）",
        title_en="[Image] Cyberpunk Shanghai Neon Poster (Typography)",
        visual_zh=(
            "超清商业电影海报摄影，雨夜霓虹闪烁的未来上海外滩街道，一位身着机能风银色夹克的少女回眸凝视镜头，"
            "景深虚化光斑，顶部醒目发光现代中文字体排版写着“未来纪元 2026”，极致光影细节，8K画质。"
        ),
        visual_en=(
            "Ultra-HD commercial movie poster photography, futuristic rain-slicked Shanghai Bund street glowing with "
            "neon reflections at night, a young woman in a sleek silver techwear jacket looking back over her shoulder "
            'at the camera, creamy bokeh light orbs, bold glowing modern title typography at the top reading "FUTURE ERA 2026", '
            "cinematic lighting, 8K detail."
        ),
    ),
    PresetTemplate(
        modality=ModelModality.IMAGE,
        title_zh="[生图] 胶片质感午后咖啡馆肖像",
        title_en="[Image] Medium Format Film Afternoon Tokyo Cafe",
        visual_zh=(
            "Hasselblad medium format film photography, soft golden afternoon sunlight streaming through wooden "
            "blinds onto a cozy minimalist Tokyo cafe table, a ceramic pour-over coffee dripper with rising steam "
            "beside an open sketchbook, natural film grain, ultra-detailed textures."
        ),
    ),
    PresetTemplate(
        modality=ModelModality.LLM_VLM,
        title_zh="[推理] 为 AI 视频导演生成 3 组新海诚风分镜提示词",
        title_en="[Reasoning] Write 3 Makoto Shinkai Anime Storyboard Prompts",
        visual_zh=(
            "请帮我构思 3 组极具画面感的新海诚风格动画短片分镜，每组包含："
            "1) 英文 Visual & Camera 镜头提示词；2) Soundscape 环境音描述；3) Background Music 配乐描述。"
        ),
        visual_en=(
            "Please design 3 vivid Makoto Shinkai-style anime short film storyboards. For each shot include: "
            "1) Visual & Camera prompt; 2) Soundscape description; 3) Background Music description."
        ),
    ),
    PresetTemplate(
        modality=ModelModality.LLM_VLM,
        title_zh="[视觉] 深度解析图片构图并反推生图/生视频 Prompt",
        title_en="[Vision] Analyze Composition & Reverse-Engineer T2I/T2V Prompt",
        visual_zh="请仔细观察这张图片的主体、光影色调、镜头景别与艺术风格，先做简要中文分析，然后反推出一段高质量的英文生图/生视频 Prompt。",
        visual_en=(
            "Carefully examine the subject, lighting, color grading, camera framing, and artistic style of this image. "
            "Provide a brief breakdown first, then reverse-engineer a detailed English T2I / T2V generation prompt."
        ),
    ),
]


def build_default_comfy_preset_packs() -> List[ComfyPresetPackItem]:
    return [
        ComfyPresetPackItem(
            id="wan22_i2v_flf2v",
            _title="万相 Wan2.2 14B · 图生视频 + 首尾帧双工作流",
            _badge="新手首选 · 4步极速",
            _scenario="含「单图动起来」与「首尾双图丝滑过渡」2 套中文画布，G4 约 25 秒出片，纯官方内置节点零红框报错",
            approx_size_gb=35.4,
            workflows=[
                "01_Wan2.2_极速单图生视频(4步傻瓜版).json",
                "02_Wan2.2_首尾帧丝滑过渡(双图4步极速).json",
            ],
            installed_models=0,
            total_models=6,
        ),
        ComfyPresetPackItem(
            id="minimax_h3_i2v_flf2v",
            _title="MiniMax-H3 Turbo · 图生有声视频 + 首尾帧双工作流",
            _badge="自带立体声音效 · 8步加速",
            _scenario="含「单图生有声视频」与「首尾双图有声过渡」2 套极简画布，一次生成高清动态画面 + 同步环境音效/配乐/对白",
            approx_size_gb=39.1,
            workflows=[
                "03_MiniMax-H3_极速图生有声视频(单图带立体声).json",
                "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json",
            ],
            installed_models=0,
            total_models=5,
        ),
        ComfyPresetPackItem(
            id="ltx23_flf2v",
            _title="LTX-2.3 (22B) · 首尾帧电影级运镜有声工作流",
            _badge="电影推拉摇移 · 仅 2 个权重",
            _scenario="专为首尾关键帧大幅度镜头运动优化（如高空俯拍平滑推镜至低角度特写 + 自动合成配乐音轨）",
            approx_size_gb=36.3,
            workflows=["05_LTX-2.3_首尾帧电影运镜(双图极速带音效).json"],
            installed_models=0,
            total_models=2,
        ),
        ComfyPresetPackItem(
            id="hunyuan15_i2v_1080p",
            _title="混元 HunyuanVideo 1.5 · 单图直出 1080P 超分工作流",
            _badge="720P + 1080P 双路超清",
            _scenario="已默认激活潜空间超分管线，传 1 张图一键自动渲染 720P 原片与 1920×1080 超高清成片",
            approx_size_gb=43.5,
            workflows=["06_Hunyuan1.5_单图直出1080P超清视频.json"],
            installed_models=0,
            total_models=7,
        ),
    ]


COMFY_WEIGHT_SUBFOLDERS: List[str] = [
    "diffusion_models",
    "checkpoints",
    "loras",
    "vae",
    "text_encoders",
    "clip_vision",
    "controlnet",
    "upscale_models",
    "audio_encoders",
    "gguf",
]

GPU_CHOICES = [
    ("G4", "G4 · PRO 6000"),
    ("H100", "H100 · 80GB"),
    ("A100", "A100 · 80GB"),
    ("L4", "L4 · 24GB"),
    ("T4", "T4 · 16GB"),
]


def get_idle_timeout_choices() -> List[Tuple[int, str]]:
    return [
        (15, tr("15 分钟", "15 min")),
        (30, tr("30 分钟", "30 min")),
        (60, tr("60 分钟", "60 min")),
        (0, tr("关闭", "Off")),
    ]


IDLE_TIMEOUT_CHOICES = [
    (15, "15 分钟"),
    (30, "30 分钟"),
    (60, "60 分钟"),
    (0, "关闭"),
]

VIDEO_DURATION_PRESETS = [
    (90, "4s"),
    (124, "5s"),
    (192, "8s"),
    (243, "10s"),
    (294, "12s"),
    (362, "15s"),
]


# ─── String & Path Helpers ───────────────────────────────────────────────────

_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0000FE0F"
    "\U0000200D"
    "\U000020E3"
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    if not text:
        return ""
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def snap_video_frames(val: int) -> int:
    snapped = int(round((val - 5) / 17.0)) * 17 + 5
    return max(90, min(362, snapped))


def format_duration(sec: float, compact: bool = False) -> str:
    s = max(0.0, float(sec))
    if s < 60.0:
        return f"{s:.1f}s" if (compact or is_en()) else f"{s:.1f} 秒"
    mins = int(s) // 60
    rem = s - mins * 60
    if compact:
        return f"{mins}m{rem:.0f}s"
    if is_en():
        return f"{mins}m {rem:.1f}s"
    return f"{mins}分{rem:.1f}秒"


def format_uptime_hms(seconds: int) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    rem = s % 60
    return f"{h:02d}:{m:02d}:{rem:02d}"


def mask_url_display(text: str, mask_enabled: bool) -> str:
    if not mask_enabled or not text:
        return text
    out = re.sub(r"https?://[a-zA-Z0-9.-]+\.gradio\.live", "https://••••••••••••.gradio.live", text)
    out = re.sub(r"https?://[a-zA-Z0-9.-]+\.trycloudflare\.com", "https://••••••••••••.trycloudflare.com", out)
    # 路径密钥和服务端打印的 API Key 也会出现在日志里，一并打码
    out = re.sub(r"/k/[A-Za-z0-9_-]+", "/k/••••••", out)
    out = re.sub(r"(API Key:\s*)\S+", r"\g<1>••••••", out)
    return out


def mask_email_display(email: str, mask_enabled: bool) -> str:
    if not email:
        return ""
    if not mask_enabled or "@" not in email:
        return email
    # 打码只保留整串前 3 个与后 3 个字符（含域名），中间隐藏，避免域名泄露邮箱服务商
    if len(email) <= 6:
        return f"{email[:2]}•••"
    return f"{email[:3]}•••{email[-3:]}"


def mask_secret_display(secret: str, mask_enabled: bool) -> str:
    """API Key 等密钥的展示打码：保留前 3 后 3 字符，中间隐藏；关闭打码时原样返回"""
    if not secret:
        return ""
    if not mask_enabled:
        return secret
    if len(secret) <= 6:
        return f"{secret[:2]}•••"
    return f"{secret[:3]}•••{secret[-3:]}"


def get_project_or_bundle_root() -> Path:
    """Locate repository root (in source mode) or PyInstaller bundle directory (in .exe mode)."""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resolve_resource_path(filename: str) -> Path:
    """Find bundled resource (app.py, colab_bridge.py, EasyColab_Server.ipynb, AppIcon.ico)."""
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / filename)
        candidates.append(exe_dir / "_internal" / filename)
        if hasattr(sys, "_MEIPASS"):
            candidates.append(Path(sys._MEIPASS) / filename)
    win_dir = Path(__file__).resolve().parent
    repo_root = win_dir.parent
    candidates.extend([
        repo_root / filename,
        win_dir / filename,
        win_dir / "assets" / filename,
    ])
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def get_app_config_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) / "EasyColab" if base else (Path.home() / "AppData" / "Local" / "EasyColab")
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "EasyColab"
    else:
        root = Path.home() / ".config" / "EasyColab"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_local_save_dir() -> Path:
    """Default local media output directory: %USERPROFILE%\\Videos\\EasyColab on Windows, ~/Movies/EasyColab on Mac."""
    if sys.platform == "win32":
        videos_root = Path.home() / "Videos"
        if not videos_root.exists():
            # Fallback if Windows Videos folder was redirected or missing
            videos_root = Path(os.path.expanduser("~")) / "Videos"
        save_dir = videos_root / "EasyColab"
    else:
        save_dir = Path.home() / "Movies" / "EasyColab"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_legacy_media_dirs() -> List[Path]:
    parent = get_local_save_dir().parent
    dirs = [get_local_save_dir()]
    for name in ("Aevix", "Colab_AI_Studio", "MiniMax_H3_Videos"):
        d = parent / name
        if d.is_dir():
            dirs.append(d)
    return dirs


def open_path_in_os(path: Path, reveal: bool = False) -> None:
    """Open file/folder in Explorer (Windows) or Finder (macOS)."""
    import subprocess

    target = Path(path).resolve()
    try:
        if sys.platform == "win32":
            if reveal and target.is_file():
                subprocess.Popen(["explorer", f"/select,{str(target)}"])
            else:
                os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            if reveal and target.is_file():
                subprocess.Popen(["open", "-R", str(target)])
            else:
                subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target.parent if reveal and target.is_file() else target)])
    except Exception:
        pass
