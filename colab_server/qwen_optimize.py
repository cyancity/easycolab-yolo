import gc
import torch


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

        @triton.jit
        def _qwen35_rmsnorm_gated_silu(x_ptr, z_ptr, w_ptr, out_ptr, D, BLOCK_D: tl.constexpr, eps):
            row = tl.program_id(0)
            offs = tl.arange(0, BLOCK_D)
            mask = offs < D
            x_raw = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
            x = x_raw.to(tl.float32)
            z = tl.load(z_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            var = tl.sum(x * x, axis=0) / D
            rstd = tl.rsqrt(var + eps)
            y = (x * rstd * w) * (z * tl.sigmoid(z))
            tl.store(out_ptr + row * D + offs, y.to(x_raw.dtype), mask=mask)

        @triton.jit
        def _qwen35_g_beta_fused(ba_ptr, neg_exp_A_ptr, dt_bias_ptr, g_ptr, beta_ptr, H, BLOCK_H: tl.constexpr):
            row = tl.program_id(0)
            offs = tl.arange(0, BLOCK_H)
            mask = offs < H
            b_raw = tl.load(ba_ptr + row * (2 * H) + offs, mask=mask, other=0.0)
            b = b_raw.to(tl.float32)
            a = tl.load(ba_ptr + row * (2 * H) + H + offs, mask=mask, other=0.0).to(tl.float32)
            neg_exp_A = tl.load(neg_exp_A_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            dt_b = tl.load(dt_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            beta = tl.sigmoid(b)
            x = a + dt_b
            sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
            g = neg_exp_A * sp
            tl.store(beta_ptr + row * H + offs, beta.to(b_raw.dtype), mask=mask)
            tl.store(g_ptr + row * H + offs, g, mask=mask)

        # head_dim=256 时把整段 D 放进寄存器会溢出，解码被延迟绑死。
        # 按 BLOCK_D=32 分块，fp32 累加写到捕获前就分配好的全局缓冲。
        _decode_acc_holder = [None]

        def _decode_acc_buf(q):
            hq, hd = q.shape[1], q.shape[-1]
            buf = _decode_acc_holder[0]
            if buf is None or buf.shape != (hq, hd) or buf.device != q.device:
                buf = torch.empty((hq, hd), device=q.device, dtype=torch.float32)
                _decode_acc_holder[0] = buf
            return buf

        @triton.jit
        def _qwen35_gqa_decode_attn(
            q_ptr, k_ptr, v_ptr, cum_ptr, acc_ptr, out_ptr, scale, max_seq,
            D: tl.constexpr, GQA_GROUPS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        ):
            pid_hq = tl.program_id(0)
            pid_hkv = pid_hq // GQA_GROUPS
            seq_len = tl.load(cum_ptr).to(tl.int32)
            acc_base = acc_ptr + pid_hq * D
            q_base = q_ptr + pid_hq * D
            k_base = k_ptr + pid_hkv * max_seq * D
            v_base = v_ptr + pid_hkv * max_seq * D
            q_dtype = tl.load(q_base).dtype
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                tl.store(acc_base + offs_d, tl.zeros([BLOCK_D], dtype=tl.float32), mask=offs_d < D)
            m_i = -float("inf")
            l_i = 0.0
            for start_n in range(0, seq_len, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < seq_len
                qk = tl.zeros([BLOCK_N], dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D
                    q_tile = tl.load(q_base + offs_d, mask=mask_d, other=0.0).to(tl.float32)
                    ptrs = offs_n[:, None] * D + offs_d[None, :]
                    mask_2d = mask_n[:, None] & mask_d[None, :]
                    k_blk = tl.load(k_base + ptrs, mask=mask_2d, other=0.0).to(tl.float32)
                    qk += tl.sum(k_blk * q_tile[None, :], axis=1)
                qk = qk * scale
                qk = tl.where(mask_n, qk, -float("inf"))
                m_ij = tl.max(qk, axis=0)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new)
                l_i = l_i * alpha + tl.sum(tl.where(mask_n, p, 0.0), axis=0)
                for d0 in range(0, D, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    mask_d = offs_d < D
                    ptrs = offs_n[:, None] * D + offs_d[None, :]
                    mask_2d = mask_n[:, None] & mask_d[None, :]
                    v_blk = tl.load(v_base + ptrs, mask=mask_2d, other=0.0).to(tl.float32)
                    acc_tile = tl.load(acc_base + offs_d, mask=mask_d, other=0.0)
                    acc_tile = acc_tile * alpha + tl.sum(p[:, None] * v_blk, axis=0)
                    tl.store(acc_base + offs_d, acc_tile, mask=mask_d)
                m_i = m_new
            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                acc_tile = tl.load(acc_base + offs_d, mask=mask_d, other=0.0)
                tl.store(out_ptr + pid_hq * D + offs_d, (acc_tile / l_i).to(q_dtype), mask=mask_d)

        for mod in text_model.modules():
            if isinstance(mod, mq.Qwen3_5RMSNorm):
                mod._w1 = (1.0 + mod.weight.detach().float()).to(mod.weight.dtype).contiguous()
                mod._norm_shape = (mod.weight.shape[0],)

                def _fast_rmsnorm(self, x):
                    return F.rms_norm(x, self._norm_shape, self._w1, self.eps)

                mod.forward = _fast_rmsnorm.__get__(mod, type(mod))

            elif isinstance(mod, mq.Qwen3_5MLP):
                if (
                    isinstance(mod.gate_proj, nn.Linear)
                    and mod.gate_proj.bias is None
                    and mod.up_proj.bias is None
                    and mod.gate_proj.weight.dtype in (torch.bfloat16, torch.float16)
                ):
                    inter = mod.gate_proj.out_features
                    mod._gate_up_w = torch.cat([mod.gate_proj.weight.detach(), mod.up_proj.weight.detach()], dim=0).contiguous()
                    mod.gate_proj = None
                    mod.up_proj = None
                    mod._inter = inter

                    def _fast_mlp(self, x):
                        gu = F.linear(x, self._gate_up_w)
                        g, u = gu.split(self._inter, dim=-1)
                        return self.down_proj(swiglu(g, u))

                    mod.forward = _fast_mlp.__get__(mod, type(mod))

            elif isinstance(mod, mq.Qwen3_5Attention):
                # NF4 权重是 uint8，拼不成一份 BF16 QKV，但解码仍要走静态 Triton 核。
                # 预填继续用 SDPA，因为静态缓冲里有 padding，必须靠官方 mask。
                can_fuse_qkv = (
                    isinstance(mod.q_proj, nn.Linear)
                    and mod.q_proj.bias is None
                    and mod.k_proj.bias is None
                    and mod.v_proj.bias is None
                    and getattr(getattr(mod.q_proj, "weight", None), "dtype", None) in (torch.bfloat16, torch.float16)
                )
                if can_fuse_qkv:
                    qd, kd, vd = mod.q_proj.out_features, mod.k_proj.out_features, mod.v_proj.out_features
                    mod._qkv_w = torch.cat(
                        [mod.q_proj.weight.detach(), mod.k_proj.weight.detach(), mod.v_proj.weight.detach()], dim=0
                    ).contiguous()
                    mod._qkv_splits = (qd, kd, vd)
                    mod.q_proj = mod.k_proj = mod.v_proj = None

                def _fast_attn(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, self.head_dim)
                    if self.q_proj is None:
                        qkv = F.linear(hidden_states, self._qkv_w)
                        q_raw, k_raw, v_raw = torch.split(qkv, self._qkv_splits, dim=-1)
                    else:
                        q_raw = self.q_proj(hidden_states)
                        k_raw = self.k_proj(hidden_states)
                        v_raw = self.v_proj(hidden_states)
                    query_states, gate = torch.chunk(q_raw.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
                    gate = gate.reshape(*input_shape, -1)
                    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                    key_states = self.k_norm(k_raw.view(hidden_shape)).transpose(1, 2)
                    value_states = v_raw.view(hidden_shape).transpose(1, 2)
                    cos, sin = position_embeddings
                    query_states, key_states = mq.apply_rotary_pos_emb(query_states, key_states, cos, sin)
                    if past_key_values is not None:
                        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
                    cache_lyr = (
                        past_key_values.layers[self.layer_idx]
                        if (past_key_values is not None and hasattr(past_key_values, "layers"))
                        else None
                    )
                    use_static = (
                        input_shape == (1, 1)
                        and cache_lyr is not None
                        and hasattr(cache_lyr, "cumulative_length")
                        and torch.is_tensor(cache_lyr.cumulative_length)
                        and getattr(cache_lyr, "keys", None) is not None
                        and getattr(past_key_values, "_no_left_pad", True)
                    )
                    if use_static:
                        q_c = query_states.contiguous()
                        attn_output = torch.empty_like(q_c)
                        k_buf = cache_lyr.keys
                        v_buf = cache_lyr.values
                        hq = q_c.shape[1]
                        hkv = k_buf.shape[1]
                        _qwen35_gqa_decode_attn[(hq,)](
                            q_c,
                            k_buf,
                            v_buf,
                            cache_lyr.cumulative_length,
                            _decode_acc_buf(q_c),
                            attn_output,
                            self.scaling,
                            k_buf.shape[2],
                            D=self.head_dim,
                            GQA_GROUPS=hq // hkv,
                            BLOCK_N=32,
                            BLOCK_D=32,
                            num_warps=4,
                        )
                    else:
                        attn_output = F.scaled_dot_product_attention(
                            query_states,
                            key_states,
                            value_states,
                            attn_mask=attention_mask,
                            dropout_p=0.0,
                            scale=self.scaling,
                            enable_gqa=True,
                        )
                    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
                    return self.o_proj(attn_output * torch.sigmoid(gate)), None

                mod.forward = _fast_attn.__get__(mod, type(mod))

            elif isinstance(mod, mq.Qwen3_5GatedDeltaNet):
                if (
                    isinstance(mod.in_proj_qkv, nn.Linear)
                    and mod.in_proj_qkv.bias is None
                    and mod.in_proj_z.bias is None
                    and mod.in_proj_b.bias is None
                    and mod.in_proj_a.bias is None
                    and mod.in_proj_qkv.weight.dtype in (torch.bfloat16, torch.float16)
                ):
                    qkv_d = mod.in_proj_qkv.out_features
                    z_d = mod.in_proj_z.out_features
                    b_d = mod.in_proj_b.out_features
                    a_d = mod.in_proj_a.out_features
                    mod._qkvzba_w = torch.cat(
                        [
                            mod.in_proj_qkv.weight.detach(),
                            mod.in_proj_z.weight.detach(),
                            mod.in_proj_b.weight.detach(),
                            mod.in_proj_a.weight.detach(),
                        ],
                        dim=0,
                    ).contiguous()
                    mod.in_proj_qkv = mod.in_proj_z = mod.in_proj_b = mod.in_proj_a = None
                    mod._splits = (qkv_d, z_d, b_d + a_d)
                    mod._neg_exp_A = (-mod.A_log.detach().float().exp()).contiguous()
                    mod._dt_bias_f32 = mod.dt_bias.detach().float().contiguous()
                    mod._block_h = triton.next_power_of_2(b_d)
                    mod._block_d = triton.next_power_of_2(mod.head_v_dim)

                    def _fast_gdn(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
                        if attention_mask is not None:
                            hidden_states = mq.apply_mask_to_padding_states(hidden_states, attention_mask)
                        B, L, _ = hidden_states.shape
                        conv_state = None
                        rec_state = None
                        recording_past = False
                        if cache_params is not None:
                            if hasattr(cache_params, "get_conv_state"):
                                conv_state = cache_params.get_conv_state(self.layer_idx, state_idx=0)
                                rec_state = cache_params.get_recurrent_state(self.layer_idx, state_idx=0)
                                recording_past = cache_params.is_recording_past(self.layer_idx)
                            elif getattr(cache_params, "has_previous_state", False):
                                cs_list = getattr(cache_params, "conv_states", None)
                                rs_list = getattr(cache_params, "recurrent_states", None)
                                if isinstance(cs_list, (list, tuple)) and self.layer_idx < len(cs_list):
                                    conv_state = cs_list[self.layer_idx]
                                if isinstance(rs_list, (list, tuple)) and self.layer_idx < len(rs_list):
                                    rec_state = rs_list[self.layer_idx]

                        proj = F.linear(hidden_states, self._qkvzba_w)
                        mixed_qkv, z, ba = torch.split(proj, self._splits, dim=-1)
                        mixed_qkv = mixed_qkv.transpose(1, 2)

                        if conv_state is not None and L == 1 and not recording_past:
                            mixed_qkv = mq.causal_conv1d_update(
                                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
                            )
                        else:
                            if cache_params is not None:
                                if hasattr(cache_params, "update_conv_state"):
                                    mixed_qkv = cache_params.update_conv_state(
                                        mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                                    )
                                else:
                                    cs_list = getattr(cache_params, "conv_states", None)
                                    if isinstance(cs_list, list) and self.layer_idx < len(cs_list):
                                        ksize = self.conv_kernel_size
                                        padded = F.pad(mixed_qkv, (ksize - mixed_qkv.shape[-1], 0)) if mixed_qkv.shape[-1] < ksize else mixed_qkv
                                        cs_list[self.layer_idx] = padded[..., -ksize:].contiguous()
                            mixed_qkv = mq.causal_conv1d_fn(
                                mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation
                            )
                            if cache_params is not None:
                                mixed_qkv = mixed_qkv[:, :, -L:]

                        mixed_qkv = mixed_qkv.transpose(1, 2)
                        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
                        q = q.view(B, L, -1, self.head_k_dim)
                        k = k.view(B, L, -1, self.head_k_dim)
                        v = v.view(B, L, -1, self.head_v_dim)

                        H = self.num_v_heads
                        g_t = torch.empty((B, L, H), device=hidden_states.device, dtype=torch.float32)
                        beta_t = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)
                        _qwen35_g_beta_fused[(B * L,)](
                            ba.contiguous(), self._neg_exp_A, self._dt_bias_f32, g_t, beta_t, H, BLOCK_H=self._block_h
                        )

                        if self.num_v_heads // self.num_k_heads > 1:
                            rep = self.num_v_heads // self.num_k_heads
                            q = q.repeat_interleave(rep, dim=2)
                            k = k.repeat_interleave(rep, dim=2)
                        if rec_state is not None and L == 1:
                            core_out, last_state = fused_recurrent_gated_delta_rule(
                                q, k, v, g=g_t, beta=beta_t, initial_state=rec_state, output_final_state=True, use_qk_l2norm_in_kernel=True
                            )
                        else:
                            core_out, last_state = chunk_gated_delta_rule(
                                q, k, v, g=g_t, beta=beta_t, initial_state=rec_state, output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True
                            )
                        if cache_params is not None:
                            if hasattr(cache_params, "update_recurrent_state"):
                                cache_params.update_recurrent_state(last_state, self.layer_idx)
                            else:
                                rs_list = getattr(cache_params, "recurrent_states", None)
                                if isinstance(rs_list, list) and self.layer_idx < len(rs_list):
                                    rs_list[self.layer_idx] = last_state

                        rows = B * L * H
                        normed = torch.empty((rows, self.head_v_dim), device=hidden_states.device, dtype=hidden_states.dtype)
                        _qwen35_rmsnorm_gated_silu[(rows,)](
                            core_out.contiguous(),
                            z.contiguous(),
                            self.norm.weight,
                            normed,
                            self.head_v_dim,
                            BLOCK_D=self._block_d,
                            eps=self.norm.variance_epsilon,
                        )
                        return self.out_proj(normed.view(B, L, -1))

                    mod.forward = _fast_gdn.__get__(mod, type(mod))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as fuse_err:
        print(f"[加速引擎] 算子融合未完成 ({type(fuse_err).__name__}: {fuse_err})")
        return

    # 图捕获失败不能把已经装上的融合层一起丢掉。
    try:
        from transformers.modeling_outputs import BaseModelOutputWithPast

        free_gb_now = torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3)
        static_cap = 32768 if free_gb_now >= 12.0 else (16384 if free_gb_now >= 8.0 else 4096)
        max_static_seq = max(2048, min(int(max_context_len or 32768), static_cap))
        t_cfg = text_model.config
        embed_dtype = text_model.embed_tokens.weight.dtype
        dev = text_model.embed_tokens.weight.device
        try:
            sc = StaticCache(config=text_model.config, max_cache_len=max_static_seq)
        except Exception:
            sc = StaticCache(config=mdl.config, max_cache_len=max_static_seq)
        for _lyr in sc.layers:
            _lyr.is_compileable = False
        try:
            sc.is_compileable = False
        except Exception:
            pass
        sc._no_left_pad = True

        # 旧版 StaticCache 把 conv 状态长度锁成第一次 prefill 的长度，而且没有 get_conv_state。
        # 这里改成按卷积核宽度保存，这样不同长度的 prefill 都能对上已捕获的 decode 图。
        def _stabilize_linear_cache(cache):
            for lyr in cache.layers:
                if isinstance(getattr(lyr, "conv_states", None), dict) or isinstance(getattr(lyr, "has_previous_state", None), dict):
                    continue
                if not hasattr(lyr, "update_conv_state"):
                    continue
                box = {"conv": None, "rec": None, "has_prev": False, "ksize": None, "batch_shape": None}
                lyr._fast_conv = box
                orig_reset = lyr.reset

                def update_conv_state(conv_states, state_idx=0, conv_kernel_size=None, _box=box, **kwargs):
                    ksize = int(conv_kernel_size or conv_states.shape[-1])
                    batch_shape = tuple(conv_states.shape[:-1])
                    if (
                        _box["conv"] is None
                        or _box["ksize"] != ksize
                        or _box["batch_shape"] != batch_shape
                        or _box["conv"].device != conv_states.device
                    ):
                        _box["ksize"] = ksize
                        _box["batch_shape"] = batch_shape
                        _box["conv"] = torch.zeros(
                            *batch_shape, ksize, dtype=conv_states.dtype, device=conv_states.device
                        )
                        _box["has_prev"] = False
                    if not _box["has_prev"]:
                        full = conv_states
                        if full.shape[-1] < ksize:
                            full = F.pad(full, (ksize - full.shape[-1], 0))
                        _box["has_prev"] = True
                    else:
                        full = torch.cat([_box["conv"], conv_states], dim=-1)
                    _box["conv"].copy_(full[..., -ksize:])
                    return full

                def update_recurrent_state(recurrent_states, state_idx=0, _box=box, **kwargs):
                    if (
                        _box["rec"] is None
                        or _box["rec"].shape != recurrent_states.shape
                        or _box["rec"].device != recurrent_states.device
                    ):
                        _box["rec"] = torch.empty_like(recurrent_states)
                    _box["rec"].copy_(recurrent_states)
                    return _box["rec"]

                def reset(_box=box, _orig=orig_reset):
                    try:
                        _orig()
                    except Exception:
                        pass
                    _box["has_prev"] = False
                    if _box["conv"] is not None:
                        _box["conv"].zero_()
                    if _box["rec"] is not None:
                        _box["rec"].zero_()

                lyr.update_conv_state = update_conv_state
                lyr.update_recurrent_state = update_recurrent_state
                lyr.reset = reset

        def _install_cache_api(cache):
            if not hasattr(cache, "get_conv_state"):
                def get_conv_state(layer_idx, state_idx=0):
                    lyr = cache.layers[layer_idx]
                    box = getattr(lyr, "_fast_conv", None)
                    if box is not None:
                        if not box["has_prev"] or box["conv"] is None:
                            return None
                        return box["conv"]
                    hps = getattr(lyr, "has_previous_state", False)
                    if isinstance(hps, dict):
                        if not hps.get(state_idx, False):
                            return None
                    elif not hps:
                        return None
                    st = getattr(lyr, "conv_states", None)
                    if isinstance(st, dict):
                        return st.get(state_idx)
                    if st is None or not getattr(lyr, "is_conv_states_initialized", False):
                        return None
                    return st

                cache.get_conv_state = get_conv_state
            if not hasattr(cache, "get_recurrent_state"):
                def get_recurrent_state(layer_idx, state_idx=0):
                    lyr = cache.layers[layer_idx]
                    box = getattr(lyr, "_fast_conv", None)
                    if box is not None:
                        if not box["has_prev"] or box["rec"] is None:
                            return None
                        return box["rec"]
                    hps = getattr(lyr, "has_previous_state", False)
                    if isinstance(hps, dict):
                        if not hps.get(state_idx, False):
                            return None
                    elif not hps:
                        return None
                    st = getattr(lyr, "recurrent_states", None)
                    if isinstance(st, dict):
                        return st.get(state_idx)
                    if st is None or not getattr(lyr, "is_recurrent_states_initialized", False):
                        return None
                    return st

                cache.get_recurrent_state = get_recurrent_state
            if not hasattr(cache, "is_recording_past"):
                def is_recording_past(layer_idx):
                    return bool(getattr(cache.layers[layer_idx], "record_past", False))

                cache.is_recording_past = is_recording_past
            orig_len = cache.get_seq_length

            def get_seq_length(layer_idx=0):
                n = orig_len(layer_idx)
                if torch.is_tensor(n):
                    return int(n.item())
                return n

            cache.get_seq_length = get_seq_length

        _stabilize_linear_cache(sc)
        _install_cache_api(sc)

        min_val = torch.finfo(embed_dtype).min
        mask_buf = torch.full((1, 1, 1, max_static_seq), min_val, device=dev, dtype=embed_dtype)
        fast_out_cls = getattr(mq, "Qwen3_5ModelOutputWithPast", None) or BaseModelOutputWithPast

        def _reset_sc_inplace():
            try:
                sc.reset()
                return
            except Exception:
                pass
            for lyr in sc.layers:
                box = getattr(lyr, "_fast_conv", None)
                if box is not None:
                    box["has_prev"] = False
                    if box["conv"] is not None:
                        box["conv"].zero_()
                    if box["rec"] is not None:
                        box["rec"].zero_()
                conv = getattr(lyr, "conv_states", None)
                if isinstance(conv, dict):
                    for st in conv.values():
                        if torch.is_tensor(st):
                            st.zero_()
                    rec = getattr(lyr, "recurrent_states", None)
                    if isinstance(rec, dict):
                        for st in rec.values():
                            if torch.is_tensor(st):
                                st.zero_()
                    hps = getattr(lyr, "has_previous_state", None)
                    if isinstance(hps, dict):
                        for k in list(hps):
                            hps[k] = False
                elif torch.is_tensor(conv):
                    conv.zero_()
                    rec = getattr(lyr, "recurrent_states", None)
                    if torch.is_tensor(rec):
                        rec.zero_()
                    if isinstance(getattr(lyr, "has_previous_state", None), bool):
                        lyr.has_previous_state = False
                if getattr(lyr, "is_initialized", False) and torch.is_tensor(getattr(lyr, "keys", None)):
                    lyr.keys.zero_()
                    lyr.values.zero_()
                cum = getattr(lyr, "cumulative_length", None)
                if torch.is_tensor(cum):
                    cum.zero_()
                elif isinstance(cum, int):
                    lyr.cumulative_length = 0

        orig_tm_forward = text_model.forward
        orig_lm_head_forward = mdl.lm_head.forward

        with torch.inference_mode():
            dummy_emb = torch.zeros((1, 4, t_cfg.hidden_size), device=dev, dtype=embed_dtype)
            orig_tm_forward(inputs_embeds=dummy_emb, past_key_values=sc, use_cache=True)

            static_attn_lyr = next(lyr for lyr in sc.layers if hasattr(lyr, "cumulative_length"))
            s_emb = torch.zeros((1, 1, t_cfg.hidden_size), device=dev, dtype=embed_dtype)
            s_pos3 = torch.zeros((3, 1, 1), device=dev, dtype=torch.long)
            mask_dict = {"full_attention": mask_buf, "linear_attention": None}

            def _decode_graph_step():
                pos_emb = text_model.rotary_emb(s_emb, s_pos3)
                h = s_emb
                for i, dec in enumerate(text_model.layers[: t_cfg.num_hidden_layers]):
                    h = dec(
                        h,
                        position_embeddings=pos_emb,
                        attention_mask=mask_dict[t_cfg.layer_types[i]],
                        past_key_values=sc,
                        use_cache=True,
                    )
                h_norm = text_model.norm(h)
                logits_out = orig_lm_head_forward(h_norm)
                return h_norm, logits_out

            side_stream = torch.cuda.Stream(device=dev)
            side_stream.wait_stream(torch.cuda.current_stream(device=dev))
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    s_out, s_logits = _decode_graph_step()
            torch.cuda.current_stream(device=dev).wait_stream(side_stream)

            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg, stream=side_stream):
                s_out, s_logits = _decode_graph_step()
            torch.cuda.synchronize(device=dev)
            _reset_sc_inplace()

        s_out_ptr = s_out.data_ptr()
        s_out_numel = s_out.numel()

        def _fast_lm_head_forward(x):
            if x.is_cuda and x.data_ptr() == s_out_ptr and x.numel() == s_out_numel:
                return s_logits
            return orig_lm_head_forward(x)

        def _hooked_tm_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            **kwargs,
        ):
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = text_model.embed_tokens(input_ids)
                input_ids = None
            if past_key_values is sc and use_cache and inputs_embeds is not None and inputs_embeds.shape[0] == 1:
                L = inputs_embeds.shape[1]
                if L > 1:
                    mask_buf.fill_(min_val)
                    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2 and attention_mask.shape[1] == L:
                        mask_buf[:, :, :, :L] = torch.where(attention_mask.view(1, 1, 1, L).bool(), 0.0, min_val)
                    else:
                        mask_buf[:, :, :, :L] = 0.0
                    return orig_tm_forward(
                        input_ids=None,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=sc,
                        inputs_embeds=inputs_embeds,
                        use_cache=True,
                        **kwargs,
                    )
                elif L == 1:
                    cur_len = static_attn_lyr.cumulative_length
                    if position_ids is None:
                        p3 = cur_len.reshape(1, 1, 1).expand(3, 1, 1)
                    else:
                        pos = position_ids
                        if pos.ndim == 1:
                            pos = pos.reshape(1, 1, -1)
                        elif pos.ndim == 2:
                            pos = pos.reshape(1, pos.shape[0], pos.shape[-1])
                        if pos.shape[0] == 4:
                            pos = pos[1:]
                        elif pos.shape[0] == 1:
                            pos = pos.expand(3, -1, -1)
                        elif pos.shape[0] != 3:
                            pos = pos.reshape(3, 1, -1)
                        p3 = pos[:, :1, -1:].reshape(3, 1, 1)
                    if p3.dtype != s_pos3.dtype or p3.device != s_pos3.device:
                        p3 = p3.to(device=s_pos3.device, dtype=s_pos3.dtype)
                    s_emb.copy_(inputs_embeds)
                    s_pos3.copy_(p3)
                    mask_buf.scatter_(3, cur_len.reshape(1, 1, 1, 1), 0.0)
                    cg.replay()
                    return fast_out_cls(last_hidden_state=s_out, past_key_values=sc)

            return orig_tm_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )

        text_model.forward = _hooked_tm_forward
        mdl.lm_head.forward = _fast_lm_head_forward

        mdl._cg_refs = (cg, sc, side_stream, s_emb, s_pos3, mask_buf, s_out, s_logits)
        if getattr(mdl, "generation_config", None) is not None:
            mdl.generation_config.disable_compile = True

        orig_generate = mdl.generate

        def _fast_generate(*args, **kwargs):
            kwargs.setdefault("disable_compile", True)
            inp_ids = kwargs.get("input_ids")
            if inp_ids is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
                inp_ids = args[0]
            amask = kwargs.get("attention_mask")
            no_pad = True if not isinstance(amask, torch.Tensor) else bool(amask.all().item())
            max_new = int(kwargs.get("max_new_tokens") or 2048)
            if (
                no_pad
                and kwargs.get("past_key_values") is None
                and inp_ids is not None
                and inp_ids.ndim == 2
                and inp_ids.shape[0] == 1
                and (inp_ids.shape[1] + max_new) <= (max_static_seq - 8)
            ):
                _reset_sc_inplace()
                kwargs["past_key_values"] = sc
            return orig_generate(*args, **kwargs)

        mdl.generate = _fast_generate
        print(
            f"[加速引擎] 全 64 层融合算子 + CUDAGraph 静态图引擎已就绪 "
            f"(CUDAGraph 静态加速 {max_static_seq} tokens / 动态最大上下文 {int(max_context_len or 32768)} tokens)！"
        )
    except Exception as opt_err:
        print(f"[加速引擎] CUDAGraph 未捕获，融合算子仍保留 ({type(opt_err).__name__}: {opt_err})")
