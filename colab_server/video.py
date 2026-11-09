import os
import time
import glob
from datetime import datetime
from typing import Optional, List, Tuple
import torch
import gradio as gr
from PIL import Image
from . import state
from .common import GPU_LOCK
from .idle import touch_activity
from .weights import build_structured_prompt
from .pipeline_opt import _reclaim_transient_gpu_memory, _run_pipeline_with_smart_tiling


# ==============================================================================
# 视频生成核心函数
# ==============================================================================
RESOLUTION_MAP = {
    "832 × 480 (16:9 横屏标准 - 推荐)": (832, 480),
    "480 × 832 (9:16 竖屏短视频)": (480, 832),
    "1280 × 720 (16:9 720P 高清)": (1280, 720),
    "720 × 1280 (9:16 720P 竖屏高清)": (720, 1280),
    "640 × 640 (1:1 方形)": (640, 640),
}


def generate_video(
    task_mode: str,
    visual_prompt: str,
    soundscape_prompt: str,
    music_prompt: str,
    first_frame_img: Optional[Image.Image],
    last_frame_img: Optional[Image.Image],
    resolution_label: str,
    num_frames: int,
    num_inference_steps: int,
    fps: int,
    seed: int,
    randomize_seed: bool,
    save_dir: str,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], str, int, List[str]]:
    touch_activity()

    if not visual_prompt or not visual_prompt.strip():
        return None, "⚠️ 请输入画面描述提示词 (Prompt)！", seed, get_history_videos(save_dir)

    if state.PIPELINE is None:
        return (
            None,
            "⚠️ 模型尚未加载到 GPU 显存！请先在左侧点击「1️⃣ 下载模型权重」与「2️⃣ 加载模型至显存」。",
            seed,
            get_history_videos(save_dir),
        )

    if randomize_seed or seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    width, height = RESOLUTION_MAP.get(resolution_label, (832, 480))
    full_prompt = build_structured_prompt(visual_prompt, soundscape_prompt, music_prompt)

    keyframes = []
    keyframe_indices = []
    if task_mode == "fl2va":
        if first_frame_img is not None:
            keyframes.append(first_frame_img.convert("RGB"))
            keyframe_indices.append(0)
        if last_frame_img is not None:
            keyframes.append(last_frame_img.convert("RGB"))
            keyframe_indices.append(-1)
        if not keyframes:
            return (
                None,
                "⚠️ 在「首尾帧图生视频」模式下，请至少上传一张【首帧图片】或【尾帧图片】！",
                seed,
                get_history_videos(save_dir),
            )

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"minimax_h3_{task_mode}_{timestamp}_seed{seed}.mp4"
    output_path = os.path.join(save_dir, filename)

    start_time = time.time()
    with GPU_LOCK:
        try:
            from diffsynth.utils.data.audio_video import write_video_audio

            if progress is not None:
                progress(0.05, desc="正在准备潜空间与多模态编码...")

            call_kwargs = dict(
                prompt=full_prompt,
                height=height,
                width=width,
                num_frames=int(num_frames),
                num_inference_steps=int(num_inference_steps),
                seed=int(seed),
            )
            if keyframes:
                call_kwargs["keyframes"] = keyframes
                call_kwargs["keyframe_indices"] = keyframe_indices

            _reclaim_transient_gpu_memory()
            video, audio = _run_pipeline_with_smart_tiling(state.PIPELINE, call_kwargs)

            if progress is not None:
                progress(0.92, desc="正在合成 32kHz 立体声音轨与 MP4 视频...")

            write_video_audio(
                video=video,
                audio=audio,
                output_path=output_path,
                fps=int(fps),
                audio_sample_rate=32000,
            )
            video = None
            audio = None
            _reclaim_transient_gpu_memory()

            elapsed = time.time() - start_time
            status_msg = (
                f"✅ 生成完成！耗时: {elapsed:.1f} 秒 | 分辨率: {width}×{height} | "
                f"帧数: {num_frames} ({num_frames/fps:.1f}s) | Seed: {seed}\n"
                f"📁 已保存至: {output_path}"
            )
            return output_path, status_msg, seed, get_history_videos(save_dir)

        except Exception as e:
            _reclaim_transient_gpu_memory()
            return None, f"❌ 生成过程中出错: {str(e)}", seed, get_history_videos(save_dir)
        finally:
            touch_activity()


def get_history_videos(save_dir: str) -> List[str]:
    if not os.path.exists(save_dir):
        return []
    files = glob.glob(os.path.join(save_dir, "*.mp4"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:12]
