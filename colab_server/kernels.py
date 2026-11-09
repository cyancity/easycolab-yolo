import sys
import subprocess
import torch


def _ensure_torchaudio_compatible():
    """防止 pip -U 升级覆盖 Colab 预装的 CUDA 定制版 torchaudio 导致 libtorchaudio.so ABI 不兼容崩溃。"""
    try:
        import torchaudio  # noqa: F401
        return
    except (OSError, ImportError) as e:
        for k in [m for m in list(sys.modules.keys()) if m == "torchaudio" or m.startswith("torchaudio.")]:
            sys.modules.pop(k, None)
        ver_raw = getattr(torch, "__version__", "")
        base_ver, plus, local_tag = ver_raw.partition("+")
        cu_tag = local_tag if plus and local_tag.startswith("cu") else ""
        print(f"检测到 torchaudio 动态库异常 ({e})，正在自动对齐 PyTorch ({ver_raw}) ABI...")
        repaired = False
        if base_ver and cu_tag:
            rc = subprocess.call([
                sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
                "--index-url", f"https://download.pytorch.org/whl/{cu_tag}",
                f"torchaudio=={base_ver}",
            ])
            if rc == 0:
                try:
                    import torchaudio  # noqa: F401
                    print(f"torchaudio 已自动修复对齐至 {torchaudio.__version__}！")
                    repaired = True
                except Exception:
                    for k in [m for m in list(sys.modules.keys()) if m == "torchaudio" or m.startswith("torchaudio.")]:
                        sys.modules.pop(k, None)
        if not repaired:
            try:
                import transformers.utils.import_utils as _iu
                _iu._torchaudio_available = False
            except Exception:
                pass


_ensure_torchaudio_compatible()


def _register_triton_causal_conv1d_shim():
    """当环境无预编译 causal_conv1d wheel (如 Python 3.13 + CUDA 13) 时，注册原生 Triton 融合算子以消除逐 token 多算子回退开销。"""
    if "causal_conv1d" in sys.modules:
        return
    try:
        import types
        import torch.nn.functional as F
        import triton
        import triton.language as tl

        @triton.jit
        def _triton_conv1d_update_k4(
            x_ptr, state_ptr, w_ptr, b_ptr, out_ptr,
            D,
            STATE_LEN: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            APPLY_SILU: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            pid_b = tl.program_id(0)
            pid_d = tl.program_id(1)
            offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = offs_d < D

            x_idx = pid_b * D + offs_d
            x_raw = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
            x0 = x_raw.to(tl.float32)

            st_base = (pid_b * D + offs_d) * STATE_LEN
            st_proto = tl.load(state_ptr + st_base, mask=mask, other=0.0)
            s0 = tl.load(state_ptr + st_base + (STATE_LEN - 3), mask=mask, other=0.0).to(tl.float32)
            s1 = tl.load(state_ptr + st_base + (STATE_LEN - 2), mask=mask, other=0.0).to(tl.float32)
            s2 = tl.load(state_ptr + st_base + (STATE_LEN - 1), mask=mask, other=0.0).to(tl.float32)

            if STATE_LEN == 4:
                tl.store(state_ptr + st_base + 0, s0.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 1, s1.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 2, s2.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 3, x0.to(st_proto.dtype), mask=mask)
            elif STATE_LEN == 3:
                tl.store(state_ptr + st_base + 0, s1.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 1, s2.to(st_proto.dtype), mask=mask)
                tl.store(state_ptr + st_base + 2, x0.to(st_proto.dtype), mask=mask)

            w_base = offs_d * 4
            w0 = tl.load(w_ptr + w_base + 0, mask=mask, other=0.0).to(tl.float32)
            w1 = tl.load(w_ptr + w_base + 1, mask=mask, other=0.0).to(tl.float32)
            w2 = tl.load(w_ptr + w_base + 2, mask=mask, other=0.0).to(tl.float32)
            w3 = tl.load(w_ptr + w_base + 3, mask=mask, other=0.0).to(tl.float32)

            acc = s0 * w0 + s1 * w1 + s2 * w2 + x0 * w3
            if HAS_BIAS:
                b = tl.load(b_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)
                acc += b
            if APPLY_SILU:
                acc = acc * tl.sigmoid(acc)
            tl.store(out_ptr + x_idx, acc.to(x_raw.dtype), mask=mask)

        def _shim_causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None, conv_state_indices=None):
            act_silu = activation in ("silu", "swish")
            if (
                x.is_cuda
                and cache_seqlens is None
                and conv_state_indices is None
                and x.ndim == 3
                and x.shape[2] == 1
                and weight.ndim == 2
                and weight.shape[1] == 4
                and conv_state.ndim == 3
                and conv_state.shape[2] in (3, 4)
                and conv_state.is_contiguous()
                and (activation is None or act_silu)
            ):
                x_c = x if x.is_contiguous() else x.contiguous()
                w_c = weight if weight.is_contiguous() else weight.contiguous()
                b_c = (bias if bias.is_contiguous() else bias.contiguous()) if bias is not None else w_c
                out = torch.empty_like(x_c)
                B, D, _ = x_c.shape
                grid = (B, triton.cdiv(D, 256))
                _triton_conv1d_update_k4[grid](
                    x_c, conv_state, w_c, b_c, out,
                    D,
                    STATE_LEN=conv_state.shape[2],
                    HAS_BIAS=bias is not None,
                    APPLY_SILU=act_silu,
                    BLOCK_D=256,
                )
                return out

            _, hidden_size, seq_len = x.shape
            state_len = conv_state.shape[-1]
            h_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
            conv_state.copy_(h_new[:, :, -state_len:])
            out = F.conv1d(h_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)[:, :, -seq_len:]
            if act_silu:
                out = F.silu(out)
            return out.to(x.dtype)

        def _shim_causal_conv1d_fn(x, weight, bias=None, activation=None, **kwargs):
            _, hidden_size, seq_len = x.shape
            padding = weight.shape[-1] - 1
            out = F.conv1d(
                x.to(weight.dtype),
                weight=weight.unsqueeze(1),
                bias=bias,
                padding=padding,
                groups=hidden_size,
            )[:, :, :seq_len]
            if activation in ("silu", "swish"):
                out = F.silu(out)
            return out.to(x.dtype)

        mod = types.ModuleType("causal_conv1d")
        mod.__version__ = "1.5.0+triton"
        mod.causal_conv1d_update = _shim_causal_conv1d_update
        mod.causal_conv1d_fn = _shim_causal_conv1d_fn
        sys.modules["causal_conv1d"] = mod
    except Exception as e:
        print(f"[加速引擎] Triton causal_conv1d 注册跳过: {e}")


def _ensure_qwen_fast_kernels():
    """确保 Qwen3.8 线性注意力 Triton 算子 (fla-core) 与融合 causal_conv1d_update 算子真正生效。"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    need_rebind = False
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule  # noqa: F401
    except Exception:
        print("正在安装 Qwen3.8 GatedDeltaNet 原生 Triton 算子 (fla-core + flash-linear-attention)...")
        for k in [m for m in list(sys.modules.keys()) if m == "fla" or m.startswith("fla.")]:
            sys.modules.pop(k, None)
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "einops"])
        subprocess.call([
            sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
            "fla-core", "flash-linear-attention",
        ])
        for k in [m for m in list(sys.modules.keys()) if m == "fla" or m.startswith("fla.")]:
            sys.modules.pop(k, None)
        need_rebind = True

    try:
        import causal_conv1d  # noqa: F401
    except ImportError:
        if sys.version_info >= (3, 13):
            _register_triton_causal_conv1d_shim()
            need_rebind = True
        else:
            rc = subprocess.call([
                sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                "--only-binary=:all:", "causal-conv1d",
            ])
            if rc != 0:
                _register_triton_causal_conv1d_shim()
                need_rebind = True

    if need_rebind and "transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules:
        import importlib
        try:
            if "transformers.integrations.hub_kernels" in sys.modules:
                importlib.reload(sys.modules["transformers.integrations.hub_kernels"])
            importlib.reload(sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"])
        except Exception:
            pass
