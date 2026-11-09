import gc
import time
from typing import Tuple
import torch
from .config import MODEL_DOWNLOAD_STATE
from .vram import _trim_linux_cpu_heap


def _enable_high_vram_optimizations(
    pipe,
    total_vram_gb: float,
    preload_vram: bool = True,
    skip_preload_attrs: Tuple[str, ...] = (),
) -> str:
    """
    针对 G4 (96GB) / A100 (80GB) 等大显存实例 (>45GB VRAM) 的深度加速（完美兼容多模型显存共驻与 BF16 超大模型）：
    1. 拦截 DiffSynth 默认在每个阶段切换时强制触发的 module.offload()，基于实时剩余显存智能判断：
       - 对于 MiniMax-H3 BF16 (text_encoder 62.1GB + dit 61.7GB > 96GB)，在 text_encoder 与 dit 互斥切换时精准腾退，
         而在 dit -> video_vae (9.7GB) -> audio_vae (0.56GB) 阶段只要余量 >= 25GB 即全量常驻，连抽换种子 0 重载！
    2. 为 Qwen3-VL 32B PromptEmbedder 挂载零触发 LRU 缓存：命中缓存时连 load_models_to_device(["text_encoder"]) 亦直接跳过，
       保证 61.7GB DiT 主干在 G4 96GB 显存中全程锁死不卸载！
    """
    import types

    is_minimax_bf16 = pipe.__class__.__name__ == "MiniMaxH3Pipeline"

    # 1. 挂载 PromptEmbedder 编码缓存（连抽多条或同 Prompt 换 Seed 时 0ms 瞬时跳过 32B Qwen3-VL 推理及显存换入）
    try:
        for unit in getattr(pipe, "units", []):
            if unit.__class__.__name__ == "MiniMaxH3Unit_PromptEmbedder" and not hasattr(
                unit, "_orig_process"
            ):
                unit._orig_process = unit.process
                unit._orig_onload_names = getattr(unit, "onload_model_names", ("text_encoder",))
                # 置空外层自动 onload，仅在缓存未命中时按需加载 text_encoder，避免同提示词连抽时把常驻 DiT 挤出显存
                unit.onload_model_names = ()
                unit._embed_cache = {}

                def _cached_prompt_process(
                    self_unit,
                    pipe_obj,
                    prompt,
                    keyframes=None,
                    ref_blocks=None,
                    height=None,
                    width=None,
                    text_embedding=None,
                ):
                    if ref_blocks is None and text_embedding is None and prompt is not None:
                        kf_sig = None
                        if keyframes:
                            try:
                                kf_sig = tuple(
                                    (img.size, img.mode, img.tobytes()[:512]) for img in keyframes
                                )
                            except Exception:
                                kf_sig = id(keyframes)
                        cache_key = (prompt, height, width, kf_sig)
                        cached = self_unit._embed_cache.get(cache_key)
                        if cached is not None:
                            print("⚡ [PromptCache] 命中 Qwen3-VL 32B 多模态编码缓存，0ms 瞬时跳过文本编码阶段！")
                            return {
                                "prompt_embeds": cached["prompt_embeds"].to(
                                    pipe_obj.device, pipe_obj.torch_dtype
                                ),
                                "text_token_tags": cached["text_token_tags"].to(
                                    pipe_obj.device, torch.long
                                ),
                            }
                        t_pe = time.time()
                        pipe_obj.load_models_to_device(self_unit._orig_onload_names)
                        res = self_unit._orig_process(
                            pipe_obj,
                            prompt,
                            keyframes=keyframes,
                            ref_blocks=ref_blocks,
                            height=height,
                            width=width,
                            text_embedding=text_embedding,
                        )
                        self_unit._embed_cache.clear()
                        self_unit._embed_cache[cache_key] = {
                            "prompt_embeds": res["prompt_embeds"].detach().clone(),
                            "text_token_tags": res["text_token_tags"].detach().clone(),
                        }
                        print(
                            f"⚡ [PromptEmbedder] 多模态提示词解析完成，总耗时 {time.time() - t_pe:.1f}s"
                            "（已写入缓存，同提示词换种子后续 0ms）"
                        )
                        return res
                    t_pe = time.time()
                    pipe_obj.load_models_to_device(self_unit._orig_onload_names)
                    res = self_unit._orig_process(
                        pipe_obj,
                        prompt,
                        keyframes=keyframes,
                        ref_blocks=ref_blocks,
                        height=height,
                        width=width,
                        text_embedding=text_embedding,
                    )
                    print(f"⚡ [PromptEmbedder] 多模态提示词解析完成，总耗时 {time.time() - t_pe:.1f}s")
                    return res

                unit.process = types.MethodType(_cached_prompt_process, unit)
    except Exception as e:
        print(f"ℹ️ Prompt 缓存挂载跳过: {e}")

    if total_vram_gb < 45.0:
        return ""

    def _is_submodel_fully_prepared_on_cuda(mdl) -> bool:
        if mdl is None:
            return False
        found_stateful = False
        for mod in mdl.modules():
            if hasattr(mod, "state"):
                found_stateful = True
                if getattr(mod, "state", 0) != 2:
                    return False
        if found_stateful:
            return True
        p = next(mdl.parameters(), None)
        return bool(p is not None and p.device.type == "cuda")

    def _offload_submodel_to_host(mdl) -> bool:
        if mdl is None or not getattr(mdl, "vram_management_enabled", False):
            return False
        has_active = any(
            hasattr(mod, "state") and getattr(mod, "state", 0) != 0
            for mod in mdl.modules()
        )
        if not has_active:
            return False
        if hasattr(mdl, "offload"):
            mdl.offload()
        else:
            for mod in mdl.modules():
                if hasattr(mod, "offload"):
                    mod.offload()
        return True

    # 2. 替换 load_models_to_device：智能阶段显存驻留 + 整块 PCIe 极速换入
    def _smart_load_models_to_device(self_pipe, model_names):
        if not getattr(self_pipe, "vram_management_enabled", False):
            return
        target_set = set(model_names or ())
        if not target_set:
            return

        # 快速路径：若请求的所有目标子模块均已 100% 处于 GPU 显存 (state=2)，0ms 直接放行，绝不误卸其他常驻模块
        target_children = [
            (name, mdl)
            for name, mdl in self_pipe.named_children()
            if name in target_set and mdl is not None
        ]
        if target_children and all(_is_submodel_fully_prepared_on_cuda(mdl) for _, mdl in target_children):
            return

        def _get_free_vram_gb() -> float:
            try:
                return torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            except Exception:
                return 0.0

        free_gb = _get_free_vram_gb()
        need_heavy_backbone = is_minimax_bf16 and (("dit" in target_set) or ("text_encoder" in target_set))
        req_free_gb = 66.5 if need_heavy_backbone else 24.0

        # 第一级腾退：切换 MiniMax-H3 两大 62GB 主干时，优先仅腾退对侧闲置大主干，让轻量的 video_vae / audio_vae 在 G4 96GB 中全程钉死显存
        if need_heavy_backbone and free_gb < req_free_gb:
            offloaded_heavy = False
            for name, model in self_pipe.named_children():
                if name not in target_set and name in ("dit", "text_encoder") and model is not None:
                    if _offload_submodel_to_host(model):
                        offloaded_heavy = True
            if offloaded_heavy:
                gc.collect()
                torch.cuda.empty_cache()
                free_gb = _get_free_vram_gb()

        # 第二级腾退：仅当腾退大主干后显存仍不足 (例如与 21GB Qwen-GGUF 共驻显存时)，再按需腾退其余非目标子模块
        if free_gb < req_free_gb:
            offloaded_rest = False
            for name, model in self_pipe.named_children():
                if name not in target_set and model is not None:
                    if _offload_submodel_to_host(model):
                        offloaded_rest = True
            if offloaded_rest:
                gc.collect()
                torch.cuda.empty_cache()
                free_gb = _get_free_vram_gb()

        # 目标模块整块换入：显存充裕时直接一次性执行 preparing() 锁入 GPU (state=2)，彻底消灭推理时 800+ 层逐层同步钩子开销
        for name, model in target_children:
            if _is_submodel_fully_prepared_on_cuda(model):
                continue
            if getattr(model, "vram_management_enabled", False):
                t_sw = time.time()
                if hasattr(model, "onload"):
                    model.onload()
                else:
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()

                cur_free_gb = _get_free_vram_gb()
                bulk_need_gb = 66.0 if (is_minimax_bf16 and name in ("dit", "text_encoder")) else 22.0
                if cur_free_gb >= bulk_need_gb:
                    for module in model.modules():
                        if hasattr(module, "preparing") and getattr(module, "state", 2) != 2:
                            try:
                                module.preparing()
                            except Exception:
                                pass
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    dt_sw = time.time() - t_sw
                    if dt_sw >= 0.3:
                        print(f"⚡ [极速显存换入] 子模块 {name} 已整块载入 GPU 显存，耗时 {dt_sw:.1f}s")

    pipe.load_models_to_device = types.MethodType(_smart_load_models_to_device, pipe)
    pipe._high_vram_fast_mode = True

    # 3. 预热指定子模块至 GPU 显存 (state=2)，保留 >=22GB 安全激活余量
    if preload_vram:
        children = [
            (name, mdl)
            for name, mdl in pipe.named_children()
            if mdl is not None and name not in skip_preload_attrs
        ]
        for idx, (attr_name, model) in enumerate(children):
            free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
            if free_gb < 22.0:
                break
            prog = round(0.80 + 0.18 * ((idx + 1) / max(1, len(children))), 2)
            MODEL_DOWNLOAD_STATE["progress"] = prog
            MODEL_DOWNLOAD_STATE["message"] = f"[显存常驻加速] 正在将子模块 {attr_name} 预热锁入 GPU 显存..."
            for module in model.modules():
                if hasattr(module, "preparing") and getattr(module, "state", 2) != 2:
                    try:
                        module.preparing()
                    except Exception:
                        pass
        _trim_linux_cpu_heap()
        used_gb = (
            torch.cuda.mem_get_info("cuda")[1] - torch.cuda.mem_get_info("cuda")[0]
        ) / (1024 ** 3)
        print(
            f"[大显存常驻已激活] 管线子模块已预热至显存 (当前总驻留 {used_gb:.1f} GB / {total_vram_gb:.1f} GB)"
        )
        return "大显存常驻"

    return "大显存常驻加速"


def _reclaim_transient_gpu_memory():
    """在连续视频/图像生成前后主动回收 Python 引用与 CUDA 缓存碎片，杜绝第 2 条视频中途 OOM 崩溃"""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _run_pipeline_with_smart_tiling(pipe, call_kwargs: dict):
    """仅当实时空闲显存 >= 46GB 时启用整帧 VAE 解码 (tiled=False)；超长帧序列 (如 15s / 362帧) 触顶时自动开启梯度检查点兜底杜绝 OOM"""
    # 若生成高分辨率超长视频 (如 720P + >=260帧/11~15秒)，采样前若 video_vae 驻留显存且总空闲偏紧，先释放 VAE 留足 33GB+ 激活空间
    num_f = int(call_kwargs.get("num_frames") or 124)
    pix = int(call_kwargs.get("height") or 480) * int(call_kwargs.get("width") or 832)
    if pipe.__class__.__name__ == "MiniMaxH3Pipeline" and num_f >= 260 and pix >= 1280 * 700:
        try:
            vvae = getattr(pipe, "video_vae", None)
            if vvae is not None and hasattr(vvae, "offload"):
                vvae.offload()
                _reclaim_transient_gpu_memory()
            elif vvae is not None:
                for mod in vvae.modules():
                    if hasattr(mod, "offload"):
                        mod.offload()
                _reclaim_transient_gpu_memory()
        except Exception:
            pass

    if getattr(pipe, "_high_vram_fast_mode", False) and "tiled" not in call_kwargs:
        try:
            free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
        except Exception:
            free_gb = 0.0
        if free_gb >= 46.0:
            fast_kwargs = dict(call_kwargs)
            fast_kwargs["tiled"] = False
            try:
                return pipe(**fast_kwargs)
            except torch.cuda.OutOfMemoryError:
                print("⚠️ 整帧 VAE 解码显存触顶，正在清理缓存并自动切换为 Tiled 分块安全解码...")
                _reclaim_transient_gpu_memory()
    safe_kwargs = dict(call_kwargs)
    safe_kwargs.setdefault("tiled", True)
    try:
        return pipe(**safe_kwargs)
    except torch.cuda.OutOfMemoryError:
        if pipe.__class__.__name__ == "MiniMaxH3Pipeline" and not safe_kwargs.get("use_gradient_checkpointing"):
            print("⚠️ 超长序列注意力显存触顶，正在自动释放闲置子模块并启用梯度检查点安全重试...")
            for sub_name in ("video_vae", "audio_vae", "text_encoder"):
                sub_m = getattr(pipe, sub_name, None)
                if sub_m is not None:
                    if hasattr(sub_m, "offload"):
                        sub_m.offload()
                    else:
                        for mod in sub_m.modules():
                            if hasattr(mod, "offload"):
                                mod.offload()
            _reclaim_transient_gpu_memory()
            gc_kwargs = dict(safe_kwargs)
            gc_kwargs["use_gradient_checkpointing"] = True
            return pipe(**gc_kwargs)
        raise
