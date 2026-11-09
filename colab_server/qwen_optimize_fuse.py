"""Qwen3.5 快速推理：Triton 核与算子融合（由 qwen_optimize.py 编排）。

两个入口都原样搬自 _optimize_qwen35_for_fast_inference：
build_kernels 在融合阶段开头定义 Triton 核（triton / tl 由调用方在 try 内导入后传入），
fuse_modules 遍历文本模型，把各层 forward 换成融合版本。
"""
import torch


def build_kernels(triton, tl):
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

    return _qwen35_rmsnorm_gated_silu, _qwen35_g_beta_fused, _qwen35_gqa_decode_attn, _decode_acc_buf


def fuse_modules(
    text_model, mq, nn, F, swiglu, triton, kernels, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
):
    _qwen35_rmsnorm_gated_silu, _qwen35_g_beta_fused, _qwen35_gqa_decode_attn, _decode_acc_buf = kernels

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
