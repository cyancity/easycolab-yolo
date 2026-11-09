import gc
import torch

from .qwen_optimize_fuse import build_kernels, fuse_modules
from .qwen_optimize_graph import install_cuda_graph


def _optimize_qwen35_for_fast_inference(mdl, max_context_len: int = 32768):
    """
    针对 Qwen3.8-27B BF16 (64层混合 GatedDeltaNet + Attention) 的深度算子融合与全模型 CUDAGraph 解码引擎：
    1. 融合 Qwen3_5RMSNorm -> 原生 C++ F.rms_norm (消除每 token 1,700+ 次 Python 标量/张量调度)
    2. 融合 Qwen3_5MLP [gate_proj; up_proj] 单次 GEMV + FLA Triton SwiGLU
    3. 融合 Qwen3_5Attention [q_proj; k_proj; v_proj] 单次 GEMV + PyTorch GQA SDPA
    4. 融合 Qwen3_5GatedDeltaNet [qkv; z; b; a] 单次 GEMV + Triton g/beta + 原生 GQA FLA + Triton RMSNormGated
    5. 预分配持久化 StaticCache 捕获 CUDAGraph 极速解码，同时兼容超长上下文 (最高 256K) 动态混合缓存
    """
    try:
        import torch.nn as nn
        import torch.nn.functional as F
        import triton
        import triton.language as tl
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq
        from transformers import StaticCache
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        from fla.modules.activations import swiglu

        text_model = getattr(getattr(mdl, "model", None), "language_model", None)
        if text_model is None:
            return

        # 单卡权重已经在 GPU 上。device_map 留下的 AlignDevicesHook 会在每层
        # forward 里递归扫描混合缓存，NF4 比 BF16 慢的一大截来自这里。
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(mdl)
        except Exception as hook_err:
            print(f"[加速引擎] accelerate hook 拆除跳过: {hook_err}")

        kernels = build_kernels(triton, tl)

        fuse_modules(
            text_model, mq, nn, F, swiglu, triton, kernels, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as fuse_err:
        print(f"[加速引擎] 算子融合未完成 ({type(fuse_err).__name__}: {fuse_err})")
        return

    # 图捕获失败不能把已经装上的融合层一起丢掉。
    try:
        from transformers.modeling_outputs import BaseModelOutputWithPast

        install_cuda_graph(mdl, text_model, mq, F, StaticCache, BaseModelOutputWithPast, max_context_len)
    except Exception as opt_err:
        print(f"[加速引擎] CUDAGraph 未捕获，融合算子仍保留 ({type(opt_err).__name__}: {opt_err})")
