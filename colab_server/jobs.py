import os
import time
import threading
from datetime import datetime
import torch
from . import state
from .bootstrap import CONSOLE_LOCK, ACTIVE_TQDM_BARS
from .common import (
    LOADED_RUNTIMES,
    GPU_LOCK,
    ABORT_GENERATION_EVENT,
    JOBS,
    _JOB_CANCEL_EVENTS,
    GenerationAbortedError,
    _is_llm_model_id,
)
from .config import MODEL_CATALOG, DEFAULT_OUTPUT_DIR
from .idle import touch_activity
from .weights import build_structured_prompt
from .pipeline_opt import _reclaim_transient_gpu_memory, _run_pipeline_with_smart_tiling
from .schemas import GenerateVideoRequest
from .llm_jobs import decode_base64_image, _run_qwen_llm_job, _fmt_mmss


from contextlib import contextmanager


@contextmanager
def _pipeline_job_execution_guard(
    pipe,
    job_id: str,
    job_cancel_event: threading.Event,
    start_time: float,
    base_prog: float,
    span_prog: float,
    prefix: str,
    model_name: str,
    media_type: str,
):
    """
    为所有 DiffSynth 视频/生图管线注入全链路真实进度与毫秒级 GPU 中断守卫：
    1. 挂载层块级 forward_pre_hook：点击停止推理后在当前 Transformer Block 结束时 (<0.1s) 立即抛出 GenerationAbortedError，
       瞬时释放 GPU_LOCK，彻底杜绝僵尸任务阻塞新任务。
    2. 接管 progress_bar_cmd 与阶段切换 load_models_to_device：
       - 0.04 ~ 0.08: 多模态提示词编码与潜空间初始化
       - 0.08 ~ 0.82 (图像 0.90): DiT 逐步去噪采样（每步实时刷新百分比与剩余时间，单行刷新不堆叠）
       - 0.84 ~ 0.93: 3D 视频 VAE 与 32kHz 立体声 VAE 解码（明确区分采样完成与后处理解码，绝不提前报 100%）
       - 0.94 ~ 0.99: MP4 音轨合成与落盘
    """
    import types

    dit_end = 0.90 if media_type == "image" else 0.82
    hook_handles = []
    orig_load_fn = getattr(pipe, "load_models_to_device", None)

    def _check_abort():
        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

    def _abort_pre_hook(_mod, _inp):
        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

    # 为管线主干及内部 Transformer Blocks 挂载轻量级中断钩子，使长耗时单步也能在 ~100ms 内立刻中止
    try:
        seen_mods = set()
        for _, child in pipe.named_children():
            if child is None or not isinstance(child, torch.nn.Module):
                continue
            if id(child) not in seen_mods:
                seen_mods.add(id(child))
                hook_handles.append(child.register_forward_pre_hook(_abort_pre_hook))
            for sub_name, sub_mod in child.named_modules():
                if isinstance(sub_mod, torch.nn.ModuleList) and any(
                    k in sub_name for k in ("blocks", "layers")
                ):
                    for blk in sub_mod:
                        if isinstance(blk, torch.nn.Module) and id(blk) not in seen_mods:
                            seen_mods.add(id(blk))
                            hook_handles.append(blk.register_forward_pre_hook(_abort_pre_hook))
    except Exception:
        pass

    if orig_load_fn is not None:
        def _stage_aware_load_models(self_pipe, model_names):
            _check_abort()
            names_set = set(model_names or ())
            if "text_encoder" in names_set:
                state.CURRENT_STAGE_BAR_LABEL = "多模态提示词编码"
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + 0.05 * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}{model_name} 正在使用多模态编码器解析提示词 (5%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = "提示词编码:   5%|#---------| 正在解析画面、运镜与声音提示词..."
            elif "video_vae" in names_set or "vae" in names_set:
                v_frac = 0.92 if media_type == "image" else 0.85
                lbl = "高清图像 VAE 解码" if media_type == "image" else "3D 视频 VAE 解码"
                state.CURRENT_STAGE_BAR_LABEL = lbl
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + v_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}DiT 采样已完成，正在执行{lbl} ({int(v_frac * 100)}%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = f"{lbl}: {int(v_frac * 100):3d}%|########--| 正在将潜空间张量解码为高清像素..."
            elif "audio_vae" in names_set:
                state.CURRENT_STAGE_BAR_LABEL = "32kHz 立体声解码"
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + 0.92 * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = f"{prefix}视频帧解码完成，正在执行 32kHz 立体声音轨解码 (92%)..."
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = "32kHz 立体声解码:  92%|#########-| 正在解码双声道环境音与配乐波形..."
            res = orig_load_fn(model_names)
            _check_abort()
            touch_activity()
            return res

        pipe.load_models_to_device = types.MethodType(_stage_aware_load_models, pipe)

    def _job_progress_bar(iterable, *_args, **_kwargs):
        _check_abort()
        items = list(iterable)
        total = max(1, len(items))
        loop_t0 = time.time()
        state.CURRENT_STAGE_BAR_LABEL = None

        if job_id in JOBS:
            JOBS[job_id]["progress"] = round(base_prog + 0.08 * span_prog, 3)
            JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
            JOBS[job_id]["message"] = f"{prefix}{model_name} 正在执行去噪采样第 1/{total} 步 (8%)..."
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS["sampling"] = f"DiT 去噪采样:   0%|----------| 0/{total} [正在执行第 1/{total} 步...]"

        for idx, item in enumerate(items):
            _check_abort()
            yield item
            _check_abort()
            done = idx + 1
            step_frac = done / float(total)
            dt = max(0.01, time.time() - loop_t0)
            avg_s = dt / float(done)
            eta_s = max(0.0, avg_s * (total - done))
            overall_frac = 0.08 + (dit_end - 0.08) * step_frac
            fill = max(0, min(10, int(round(10 * step_frac))))
            touch_activity()

            if done < total:
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + overall_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = (
                        f"{prefix}{model_name} 正在采样 {done}/{total} 步 "
                        f"(总进度 {int(overall_frac * 100)}% · {avg_s:.1f}s/步 · 采样剩余约 {int(round(eta_s))}s)..."
                    )
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = (
                        f"DiT 去噪采样: {int(step_frac * 100):3d}%|{'#' * fill}{'-' * (10 - fill)}| "
                        f"{done}/{total} [{_fmt_mmss(dt)}<{_fmt_mmss(eta_s)}, {avg_s:.2f}s/step] · 总进度 {int(overall_frac * 100)}%"
                    )
            else:
                post_frac = dit_end + 0.02
                next_lbl = "高清图像 VAE 解码" if media_type == "image" else "3D 视频与音频 VAE 解码"
                state.CURRENT_STAGE_BAR_LABEL = next_lbl
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = round(base_prog + post_frac * span_prog, 3)
                    JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                    JOBS[job_id]["message"] = (
                        f"{prefix}DiT 采样已完成 ({total}/{total})，正在进入{next_lbl} ({int(post_frac * 100)}%)..."
                    )
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS["sampling"] = (
                        f"{next_lbl}: {int(post_frac * 100):3d}%|########--| "
                        f"DiT 采样已完成 ({total}/{total}，耗时 {_fmt_mmss(dt)}) -> 正在解码输出..."
                    )
                print(f"[{model_name}] DiT 采样 {total}/{total} 步完成 ({dt:.1f}s)，开始执行{next_lbl}...")

    try:
        yield _job_progress_bar
    finally:
        state.CURRENT_STAGE_BAR_LABEL = None
        if orig_load_fn is not None:
            pipe.load_models_to_device = orig_load_fn
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass


def _run_async_generation_job(job_id: str, req: GenerateVideoRequest):
    touch_activity()
    job_cancel_event = _JOB_CANCEL_EVENTS.setdefault(job_id, threading.Event())
    active_model_id = (
        req.model_id
        if (req.model_id and req.model_id in MODEL_CATALOG)
        else (state.SELECTED_MODEL_ID if state.SELECTED_MODEL_ID in LOADED_RUNTIMES else (state.LOADED_MODEL_ID or state.SELECTED_MODEL_ID))
    )
    batch_count = max(1, min(8, int(getattr(req, "batch_count", 1) or 1)))

    JOBS[job_id]["model_id"] = active_model_id
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["progress"] = 0.01
    JOBS[job_id]["batch_count"] = batch_count
    JOBS[job_id]["completed_count"] = 0
    JOBS[job_id]["completed_videos"] = []
    start_time = time.time()
    JOBS[job_id]["started_at"] = start_time

    try:
        if job_cancel_event.is_set():
            raise GenerationAbortedError("已手动停止当前推理任务")

        # 分支 A：Qwen3.8-27B / Qwen3.8-27B-GGUF 视觉理解与深度推理对话
        if _is_llm_model_id(active_model_id):
            _run_qwen_llm_job(job_id, req, target_llm_id=active_model_id)
            return

        rt = LOADED_RUNTIMES.get(active_model_id)
        active_pipe = rt.get("pipeline") if rt else (state.PIPELINE if state.LOADED_MODEL_ID == active_model_id else None)
        if active_pipe is None:
            req_name = MODEL_CATALOG.get(active_model_id, {}).get("name", active_model_id)
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = f"{req_name} 尚未加载到 GPU 显存！请先在左侧点击「加载到显存」（显存充裕时可与现有模型同时常驻）。"
            return

        spec = MODEL_CATALOG.get(active_model_id, MODEL_CATALOG["minimax_h3"])
        modality = spec["modality"]
        media_type = "image" if modality == "image" else "video"
        JOBS[job_id]["media_type"] = media_type
        JOBS[job_id]["progress"] = 0.02
        JOBS[job_id]["message"] = (
            f"正在准备 {spec['name']} 推理管线 (共 {batch_count} 项)..."
            if batch_count > 1
            else f"正在准备 {spec['name']} 推理管线..."
        )

        base_seed = req.seed
        if req.randomize_seed or base_seed < 0:
            base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        JOBS[job_id]["seed"] = base_seed

        first_img = decode_base64_image(req.first_frame_base64)
        last_img = decode_base64_image(req.last_frame_base64)

        keyframes = []
        keyframe_indices = []
        if active_model_id == "minimax_h3" and req.task_mode == "fl2va":
            if first_img is not None:
                keyframes.append(first_img)
                keyframe_indices.append(0)
            if last_img is not None:
                keyframes.append(last_img)
                keyframe_indices.append(-1)
            if not keyframes:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["message"] = "首尾帧模式下请至少提供首帧或尾帧图片"
                return
        elif modality == "image" and req.task_mode == "i2i_edit" and first_img is None and last_img is None:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = "图像编辑/重绘模式下请至少拖入一张输入图片"
            return

        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        last_filename = None
        last_seed = base_seed

        for batch_idx in range(batch_count):
            if job_cancel_event.is_set():
                raise GenerationAbortedError("已手动停止当前推理任务")

            if batch_idx == 0:
                current_seed = base_seed
            elif req.randomize_seed:
                current_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            else:
                current_seed = base_seed + batch_idx

            last_seed = current_seed
            JOBS[job_id]["seed"] = current_seed
            prefix = f"[{batch_idx + 1}/{batch_count}] " if batch_count > 1 else ""
            base_prog = batch_idx / float(batch_count)
            span_prog = 1.0 / float(batch_count)

            if GPU_LOCK.locked():
                JOBS[job_id]["progress"] = round(base_prog + 0.01 * span_prog, 3)
                JOBS[job_id]["message"] = f"{prefix}正在等待上一任务释放 GPU 显存锁..."

            with GPU_LOCK:
                if job_cancel_event.is_set():
                    raise GenerationAbortedError("已手动停止当前推理任务")
                ABORT_GENERATION_EVENT.clear()

                # 获取 GPU 锁后重置本轮起始计时并清空所有旧进度条，确保新任务严格从 0% 与 0.0s 起步
                sub_start = time.time()
                if batch_idx == 0:
                    start_time = sub_start
                    JOBS[job_id]["started_at"] = start_time
                    JOBS[job_id]["elapsed"] = 0.0
                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()

                JOBS[job_id]["progress"] = round(base_prog + 0.04 * span_prog, 3)
                JOBS[job_id]["message"] = (
                    f"{prefix}{spec['name']} 正在初始化潜空间并准备 {req.num_inference_steps} 步推理 (Seed: {current_seed})..."
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                touch_activity()
                _reclaim_transient_gpu_memory()

                with _pipeline_job_execution_guard(
                    pipe=active_pipe,
                    job_id=job_id,
                    job_cancel_event=job_cancel_event,
                    start_time=start_time,
                    base_prog=base_prog,
                    span_prog=span_prog,
                    prefix=prefix,
                    model_name=spec["name"],
                    media_type=media_type,
                ) as step_progress_bar:
                    if active_model_id == "minimax_h3":
                        from diffsynth.utils.data.audio_video import write_video_audio
                        full_prompt = build_structured_prompt(
                            req.visual_prompt, req.soundscape_prompt, req.music_prompt
                        )
                        filename = f"minimax_h3_{req.task_mode}_{timestamp}_seed{current_seed}.mp4"
                        output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
                        call_kwargs = dict(
                            prompt=full_prompt,
                            height=int(req.height),
                            width=int(req.width),
                            num_frames=int(req.num_frames),
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            progress_bar_cmd=step_progress_bar,
                        )
                        if keyframes:
                            call_kwargs["keyframes"] = keyframes
                            call_kwargs["keyframe_indices"] = keyframe_indices
                        video, audio = _run_pipeline_with_smart_tiling(active_pipe, call_kwargs)
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.95 * span_prog, 3)
                        JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
                        JOBS[job_id]["message"] = f"{prefix}VAE 解码完成，正在合成 32kHz 立体声音轨并封装 MP4 (95%)..."
                        with CONSOLE_LOCK:
                            ACTIVE_TQDM_BARS["sampling"] = "音视频封装:  95%|#########-| 正在编码 H.264 视频与 32kHz 立体声 MP4..."
                        write_video_audio(
                            video=video,
                            audio=audio,
                            output_path=output_path,
                            fps=int(req.fps),
                            audio_sample_rate=32000,
                        )
                        video = None
                        audio = None

                    elif active_model_id == "qwen_image_21":
                        filename = f"qwen_image21_{req.task_mode}_{timestamp}_seed{current_seed}.png"
                        output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
                        edit_imgs = [im for im in (first_img, last_img) if im is not None] if req.task_mode == "i2i_edit" else None
                        img_out = active_pipe(
                            prompt=req.visual_prompt,
                            negative_prompt=req.negative_prompt.strip() or " ",
                            edit_image=edit_imgs if edit_imgs else None,
                            height=int(req.height),
                            width=int(req.width),
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            progress_bar_cmd=step_progress_bar,
                        )
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.96 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在写入无损 PNG 图像文件 (96%)..."
                        img_out.save(output_path)
                        img_out = None
                        edit_imgs = None

                    elif active_model_id == "z_image_turbo":
                        filename = f"z_image_turbo_{req.task_mode}_{timestamp}_seed{current_seed}.png"
                        output_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
                        w, h = int(req.width), int(req.height)
                        call_kw = dict(
                            prompt=req.visual_prompt,
                            negative_prompt=req.negative_prompt or "",
                            height=h,
                            width=w,
                            num_inference_steps=int(req.num_inference_steps),
                            seed=int(current_seed),
                            rand_device="cuda",
                            progress_bar_cmd=step_progress_bar,
                        )
                        if req.task_mode == "i2i_edit" and first_img is not None:
                            call_kw["input_image"] = first_img.resize((w, h))
                            call_kw["denoising_strength"] = max(0.15, min(1.0, float(req.denoising_strength or 0.70)))
                        img_out = active_pipe(**call_kw)
                        if job_cancel_event.is_set() or ABORT_GENERATION_EVENT.is_set():
                            raise GenerationAbortedError("已手动停止当前推理任务")
                        JOBS[job_id]["progress"] = round(base_prog + 0.96 * span_prog, 3)
                        JOBS[job_id]["message"] = f"{prefix}正在写入无损 PNG 图像文件 (96%)..."
                        img_out.save(output_path)
                        img_out = None
                        call_kw.clear()

                    else:
                        raise ValueError(f"未知模型 ID: {active_model_id}")

                if not os.path.exists(output_path) or os.path.getsize(output_path) < 256:
                    raise RuntimeError("输出媒体文件未成功生成或写入为空")

                with CONSOLE_LOCK:
                    ACTIVE_TQDM_BARS.clear()
                _reclaim_transient_gpu_memory()

            sub_elapsed = round(time.time() - sub_start, 1)
            last_filename = filename
            kind_cn = "图像" if media_type == "image" else "视频"
            print(f"[{spec['name']}] {kind_cn}生成完成 | 耗时 {sub_elapsed}s | Seed: {current_seed} | 文件: {filename}")
            media_item = {
                "index": batch_idx + 1,
                "seed": current_seed,
                "filename": filename,
                "video_url": f"/minimax_api/video/{filename}",
                "media_type": media_type,
                "elapsed": sub_elapsed,
            }
            JOBS[job_id]["completed_videos"] = list(JOBS[job_id]["completed_videos"]) + [media_item]
            JOBS[job_id]["completed_count"] = batch_idx + 1
            JOBS[job_id]["filename"] = filename
            JOBS[job_id]["video_url"] = f"/minimax_api/video/{filename}"
            JOBS[job_id]["elapsed"] = round(time.time() - start_time, 1)
            if batch_count > 1 and batch_idx + 1 < batch_count:
                JOBS[job_id]["progress"] = round((batch_idx + 1) / float(batch_count), 3)
                JOBS[job_id]["message"] = (
                    f"第 {batch_idx + 1}/{batch_count} 项已完成 ({sub_elapsed}s)，已同步回传 Mac！继续生成第 {batch_idx + 2} 项..."
                )
            touch_activity()

        elapsed = round(time.time() - start_time, 1)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 1.0
        JOBS[job_id]["elapsed"] = elapsed
        JOBS[job_id]["filename"] = last_filename
        JOBS[job_id]["video_url"] = f"/minimax_api/video/{last_filename}" if last_filename else None
        if batch_count > 1:
            JOBS[job_id]["message"] = (
                f"批量生成 {batch_count} 项全部完成！总耗时 {elapsed}s (平均 {round(elapsed / batch_count, 1)}s/项)"
            )
        else:
            JOBS[job_id]["message"] = f"生成完成！耗时 {elapsed}s (Seed: {last_seed})"

    except GenerationAbortedError as abort_err:
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        _reclaim_transient_gpu_memory()
        if job_id in JOBS:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["progress"] = 0.0
            JOBS[job_id]["message"] = str(abort_err) or "已手动停止当前推理任务"
        print(f"[任务控制] 任务 {job_id} 已中断并释放 GPU 显存锁。")
    except Exception as e:
        with CONSOLE_LOCK:
            ACTIVE_TQDM_BARS.clear()
        _reclaim_transient_gpu_memory()
        if job_id in JOBS:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = f"生成出错: {str(e)}"
    finally:
        _JOB_CANCEL_EVENTS.pop(job_id, None)
        touch_activity()
