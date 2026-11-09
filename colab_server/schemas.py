from typing import Optional, List, Dict, Any
from pydantic import BaseModel


# ==============================================================================
# 供 macOS 原生客户端调用的异步 REST API 接口 (/minimax_api/* · 多模型通用)
# ==============================================================================
class SelectModelRequest(BaseModel):
    model_id: str = "minimax_h3"


class DownloadModelRequest(BaseModel):
    model_id: Optional[str] = None
    hf_token: str = ""


class LoadModelRequest(BaseModel):
    model_id: Optional[str] = None
    precision_mode: str = "bf16"
    offload_mode: str = "disk"
    hf_token: str = ""
    context_length: Optional[int] = None
    options: Dict[str, Any] = {}  # 模型专属选项，如 llama 运行时模型 (Flash-Next 等) 的 quant / kv / mtp / draft_n


class ContextConfigRequest(BaseModel):
    context_length: int = 32768


class IdleConfigRequest(BaseModel):
    idle_timeout_minutes: int = 30


class ComfyStartRequest(BaseModel):
    unload_existing: bool = False
    force_restart: bool = False


class ComfyWeightDownloadRequest(BaseModel):
    url_or_repo: str
    filename: str = ""
    subfolder: str = "diffusion_models"
    hf_token: str = ""


class ComfyInstallPackRequest(BaseModel):
    pack_id: str
    auto_start: bool = True
    hf_token: str = ""


class GenerateVideoRequest(BaseModel):
    model_id: Optional[str] = None
    task_mode: str = "t2va"
    visual_prompt: str
    negative_prompt: str = ""
    soundscape_prompt: str = ""
    music_prompt: str = ""
    first_frame_base64: Optional[str] = None
    last_frame_base64: Optional[str] = None
    width: int = 832
    height: int = 480
    num_frames: int = 124
    num_inference_steps: int = 35
    fps: int = 24
    seed: int = 42
    randomize_seed: bool = True
    batch_count: int = 1
    denoising_strength: float = 0.70
    enable_thinking: bool = True
    max_new_tokens: int = 16384
    context_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: Optional[float] = None
    repeat_penalty: float = 1.0
    reasoning_effort: str = "medium"
    chat_messages: Optional[List[Dict[str, str]]] = None
