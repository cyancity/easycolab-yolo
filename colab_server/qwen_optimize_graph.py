"""Qwen3.5 快速推理：静态缓存与全模型 CUDAGraph 解码（由 qwen_optimize.py 编排）。

install_cuda_graph 原样搬自 _optimize_qwen35_for_fast_inference 的第二阶段，
失败时由调用方的 try/except 打印并保留已装上的融合层。
"""
import torch


def install_cuda_graph(mdl, text_model, mq, F, StaticCache, BaseModelOutputWithPast, max_context_len):
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
