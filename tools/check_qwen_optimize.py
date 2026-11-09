#!/usr/bin/env python3
"""行为对比：拆分后的 qwen_optimize 与原单文件函数，在同一批记录型假对象上产生的事件序列必须逐条一致。

用法：python tools/check_qwen_optimize.py [--ref HEAD]   （--ref 是原单文件所在的 git 版本；提交拆分后用 HEAD~1）
桩沿用 smoke_import 的做法：torch / triton / fla / transformers / accelerate 都是假模块，不加载真库。
Rec 记录属性读写、调用、下标、迭代；函数和绑定方法按 __name__ 记录，所以搬家不影响比较。
每个场景先跑优化函数，再把被换上的 forward、缓存方法、钩子真正调用一遍，事件进同一条 LOG。
不能证明：真 torch 的数值、Triton 核的编译与运行、CUDAGraph 捕获与回放、真实 transformers 的类和缓存 API。
"""
import contextlib, importlib, io, os, subprocess, sys, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG, CFG, ABSENT = [], {}, object()

CREATED = []


def desc(v):
    if isinstance(v, types.MethodType):
        return "bound:" + v.__func__.__name__
    if isinstance(v, types.FunctionType):
        return "fn:" + v.__name__
    if isinstance(v, (list, tuple)):
        return type(v).__name__ + "[" + ",".join(map(desc, v)) + "]"
    if isinstance(v, dict):
        return "{" + ",".join("%s:%s" % (desc(k), desc(x)) for k, x in v.items()) + "}"
    return repr(v)

class Rec:
    def __init__(self, p, **preset):
        object.__setattr__(self, "_p", p)
        object.__setattr__(self, "_kids", dict(preset))
    def _kid(self, k, p):
        return self._kids.setdefault(k, Rec(p))
    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        LOG.append(("get", self._p, n))
        v = self._kid(n, "%s.%s" % (self._p, n))
        if v is ABSENT:
            raise AttributeError(n)
        return v
    def __setattr__(self, n, v):
        LOG.append(("set", self._p, n, desc(v)))
        self._kids[n] = v
    def __call__(self, *a, **k):
        LOG.append(("call", self._p, desc(a), desc(k)))
        if self._p.endswith((".split", ".update")):
            return (Rec(self._p + "()0"), Rec(self._p + "()1"))
        return Rec(self._p + "()")
    def __getitem__(self, k):
        LOG.append(("item", self._p, desc(k)))
        return self._kid("[%s]" % desc(k), "%s[%s]" % (self._p, desc(k)))
    def __setitem__(self, k, v):
        LOG.append(("setitem", self._p, desc(k), desc(v)))
    def __iter__(self):
        LOG.append(("iter", self._p))
        return iter([self._kid("[%d]" % i, "%s[%d]" % (self._p, i)) for i in range(3)])
    def __enter__(self):
        LOG.append(("enter", self._p))
        return self
    def __exit__(self, *a):
        return LOG.append(("exit", self._p))

    __bool__, __int__ = lambda s: True, lambda s: 1
    __repr__ = lambda s: "<%s>" % s._p
    __lt__ = __le__ = __gt__ = __ge__ = lambda s, o: False
    __neg__ = __invert__ = lambda s: Rec("(~%s)" % s._p)

for _n in "add sub mul truediv floordiv radd rsub rmul rtruediv matmul".split():
    setattr(Rec, "__%s__" % _n, lambda s, o, _n=_n: Rec("(%s %s %s)" % (s._p, _n, desc(o))))

def mkfn(name, ret=None, fail=None):
    def f(*a, **k):
        LOG.append(("call", name, desc(a), desc(k)))
        if fail and CFG.get(fail):
            raise RuntimeError(name + " boom")
        return ret() if ret else Rec(name + "()")
    f.__name__ = name
    return f

class FakeMod(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        setattr(self, n, Rec(self.__name__ + "." + n))
        return getattr(self, n)

def mk(name, **attrs):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        n = ".".join(parts[:i])
        if n not in sys.modules:
            sys.modules[n] = FakeMod(n)
            i > 1 and setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], sys.modules[n])
    sys.modules[name].__dict__.update(attrs)
    return sys.modules[name]

Linear, Qwen3_5RMSNorm, Qwen3_5MLP, Qwen3_5Attention, Qwen3_5GatedDeltaNet = (
    type(n, (Rec,), {}) for n in ("Linear", "Qwen3_5RMSNorm", "Qwen3_5MLP", "Qwen3_5Attention", "Qwen3_5GatedDeltaNet"))

def mem_info(_dev):
    LOG.append(("call", "mem_get_info", _dev))
    if CFG.get("mem_raises"):
        raise RuntimeError("no cuda")
    return (int(CFG["free_gb"] * 1024 ** 3), 0)

def static_cache(**kw):
    LOG.append(("call", "StaticCache", desc(kw)))
    if CFG.get("sc_first_fails") and "tm.config" in desc(kw):
        raise RuntimeError("sc boom")
    A = ABSENT
    lays = [Rec("sc.L0", cumulative_length=Rec("cum"), keys=Rec("keys"), values=Rec("vals"), update_conv_state=A, _fast_conv=A, has_previous_state=A),
            Rec("sc.L1", update_conv_state=Rec("ucs1"), reset=Rec("reset1"), cumulative_length=A),
            Rec("sc.L2", conv_states={}, has_previous_state={0: True}, _fast_conv=A, cumulative_length=A)]
    sc = Rec("sc", layers=lays, get_conv_state=A, get_recurrent_state=A, is_recording_past=A, _no_left_pad=A,
             get_seq_length=mkfn("orig_get_seq_length"), reset=mkfn("sc_reset", fail="reset_raises"))
    CREATED.append(sc)
    return sc

def install_fakes():
    t = mk("torch", bfloat16="bf16", float16="fp16", long="long", float32="f32", Tensor=Rec,
           is_tensor=lambda x: isinstance(x, Rec))
    t.split = lambda x, sizes, dim=-1: tuple(Rec("%s.split%d" % (x._p, i)) for i in range(len(sizes)))
    t.chunk = lambda x, n, dim=-1: tuple(Rec("%s.chunk%d" % (x._p, i)) for i in range(n))
    t.cuda = Rec("torch.cuda", is_available=lambda: CFG.get("cuda", True), mem_get_info=mem_info)
    t.nn = mk("torch.nn", Linear=Linear, functional=mk("torch.nn.functional"))
    mk("triton", jit=lambda f: Rec("kernel:" + f.__name__), next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    mk("triton.language")
    pair = lambda: (Rec("o"), Rec("state"))
    mk("fla.ops.gated_delta_rule", chunk_gated_delta_rule=mkfn("chunk_gdr", pair), fused_recurrent_gated_delta_rule=mkfn("fused_rec_gdr", pair))
    mk("fla.modules.activations", swiglu=mkfn("swiglu"))
    mk("accelerate.hooks", remove_hook_from_submodules=mkfn("remove_hook", fail="hook_raises"))
    mk("transformers", StaticCache=static_cache)
    mk("transformers.modeling_outputs", BaseModelOutputWithPast=mkfn("BaseModelOutputWithPast"))
    mk("transformers.models.qwen3_5.modeling_qwen3_5", Qwen3_5RMSNorm=Qwen3_5RMSNorm, Qwen3_5MLP=Qwen3_5MLP,
       Qwen3_5Attention=Qwen3_5Attention, Qwen3_5GatedDeltaNet=Qwen3_5GatedDeltaNet, apply_rotary_pos_emb=mkfn("rope", lambda: (Rec("q"), Rec("k"))),
       apply_mask_to_padding_states=mkfn("pad_mask"), causal_conv1d_update=mkfn("conv_update"), causal_conv1d_fn=mkfn("conv_fn"))

def lin(name, ok):
    return Linear(name, bias=None, out_features=8, weight=Rec(name + ".w", dtype="bf16" if ok else "u8"))

def build(ok):  # ok=False 走各层不可融合分支（NF4 / 非 Linear / 带 bias）
    norm = Qwen3_5RMSNorm("norm", weight=Rec("norm.w", dtype="bf16"), eps=1e-6)
    mlp = Qwen3_5MLP("mlp", gate_proj=lin("g", ok), up_proj=lin("u", ok), down_proj=Rec("down"))
    attn = Qwen3_5Attention("attn", q_proj=lin("q", ok) if ok else Rec("q"), k_proj=lin("k", ok), v_proj=lin("v", ok), head_dim=4,
                            scaling=0.5, layer_idx=0, o_proj=Rec("o_proj"))
    gdn = Qwen3_5GatedDeltaNet("gdn", in_proj_qkv=lin("qkv", ok), in_proj_z=lin("z", ok), in_proj_b=lin("b", ok), in_proj_a=lin("a", ok),
                               A_log=Rec("A"), dt_bias=Rec("dt"), head_v_dim=6, head_k_dim=4, num_v_heads=4, num_k_heads=2, layer_idx=1,
                               conv_kernel_size=4, key_dim=8, value_dim=12, activation="silu", conv1d=Rec("conv1d"),
                               norm=Rec("gnorm", variance_epsilon=1e-6), out_proj=Rec("out_proj"))
    if not ok:
        gdn.in_proj_z.bias = Rec("bias")
    cfg = Rec("tm.config", hidden_size=8, num_hidden_layers=3, layer_types=["full_attention", "linear_attention", "linear_attention"])
    tm = Rec("tm", config=cfg, embed_tokens=Rec("emb", weight=Rec("emb.w", dtype="bf16", device="cuda:0")),
             layers=[Rec("dec%d" % i) for i in range(3)], modules=lambda: 1 / 0 if CFG.get("fuse_fails") else [norm, mlp, attn, gdn, Rec("other")])
    mdl = Rec("mdl", model=Rec("model", language_model=tm if CFG.get("has_tm", True) else None), lm_head=Rec("lm_head"),
              generate=mkfn("orig_generate"), config=Rec("mdl.config"), generation_config=Rec("gen_cfg") if ok else None)
    return mdl, tm, (norm, mlp, attn, gdn)

def exercise(mdl, tm, mods):
    """把被换上的东西真正调用一遍。"""
    x = lambda n, shape: Rec(n, shape=shape, ndim=len(shape), dtype="bf16", device="cuda:0")
    norm, mlp, attn, gdn = mods
    [m.forward(x("x", (1, 1, 8))) for m in (norm, mlp) if "forward" in m._kids]
    if "forward" in attn._kids:
        for shape in ((1, 1, 8), (1, 5, 8)):
            for pkv in (None, Rec("pkv", layers=[Rec("pl", cumulative_length=Rec("c"), keys=Rec("kk"), values=Rec("vv"))])):
                attn.forward(x("h", shape), (Rec("cos"), Rec("sin")), Rec("mask"), past_key_values=pkv)
    if "forward" in gdn._kids:
        new_api = Rec("cache_new", get_conv_state=mkfn("get_conv"), get_recurrent_state=mkfn("get_rec"), is_recording_past=mkfn("is_rec", lambda: False),
                      update_conv_state=mkfn("upd_conv"), update_recurrent_state=mkfn("upd_rec"))
        old_api = Rec("cache_old", get_conv_state=ABSENT, update_conv_state=ABSENT, update_recurrent_state=ABSENT, has_previous_state=True,
                      conv_states=[Rec("cs0"), Rec("cs1")], recurrent_states=[Rec("rs0"), Rec("rs1")])
        for shape in ((1, 1, 8), (1, 7, 8)):
            for cache in (None, new_api, old_api):
                gdn.forward(x("h", shape), cache_params=cache, attention_mask=None if shape[1] == 1 else Rec("am"))
    sc = CREATED[-1] if CREATED else None
    if sc is None:
        return
    [(sc.get_conv_state(i), sc.get_recurrent_state(i), sc.is_recording_past(i)) for i in range(3)]
    l1 = sc.layers[1]
    cs, rs = x("cs", (2, 3, 2)), x("rs", (2, 3, 2))
    for _ in range(2):
        l1.update_conv_state(cs, 0, 4)
        l1.update_recurrent_state(rs)
        sc.get_conv_state(1), sc.get_recurrent_state(1)
    l1.reset()
    sc.get_seq_length(0)
    if "forward" not in tm._kids or desc(tm._kids["forward"]) != "fn:_hooked_tm_forward":
        return
    emb, am = (lambda L: x("emb_in", (1, L, 8))), Rec("am", ndim=2, shape=(1, 4))
    tm.forward(input_ids=Rec("ids"), past_key_values=sc, use_cache=True, attention_mask=am, inputs_embeds=None)
    tm.forward(inputs_embeds=emb(4), past_key_values=sc, use_cache=True, attention_mask=am)
    tm.forward(inputs_embeds=emb(4), past_key_values=sc, use_cache=True)
    for pos in (None, x("p1", (5,)), x("p2", (1, 5)), x("p4", (4, 1, 1))):
        tm.forward(inputs_embeds=emb(1), past_key_values=sc, use_cache=True, position_ids=pos)
    tm.forward(inputs_embeds=emb(1), past_key_values=Rec("other_cache"), use_cache=True)
    mdl.lm_head.forward(x("h_norm", (1, 1, 8)))
    mdl.generate(x("ids", (1, 5)), max_new_tokens=16)
    mdl.generate(input_ids=x("ids", (1, 5)), attention_mask=x("amask", (1, 5)), max_new_tokens=99999)
    mdl.generate(input_ids=x("ids2", (2, 5)))
SCENARIOS = {
    "S1 全部可融合，显存 20G": (True, dict(free_gb=20.0)),
    "S2 不可融合分支，显存 9G，StaticCache 首次失败，reset 失败，无 generation_config": (False, dict(free_gb=9.0, sc_first_fails=True, reset_raises=True)),
    "S3 显存 3G": (True, dict(free_gb=3.0)),
    "S4 无 language_model 提前返回": (True, dict(has_tm=False, free_gb=20.0)),
    "S5 accelerate hook 拆除失败": (True, dict(hook_raises=True, free_gb=20.0)),
    "S6 mem_get_info 失败（图阶段异常，融合保留）": (True, dict(mem_raises=True)),
    "S7 无 CUDA": (True, dict(cuda=False, free_gb=20.0)),
    "S8 融合阶段异常，直接返回": (True, dict(fuse_fails=True, free_gb=20.0)),
}

def run(fn, ok, cfg):
    LOG.clear(), CREATED.clear(), CFG.clear(), CFG.update(cfg)
    mdl, tm, mods = build(ok)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(mdl, 4096 if cfg.get("free_gb") == 3.0 else 32768)
        LOG.append(("--exercise",))
        try:
            exercise(mdl, tm, mods)
        except Exception as e:
            LOG.append(("exercise-error", type(e).__name__, str(e)))
    return list(LOG), buf.getvalue()

def main():
    ref = (sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "HEAD")
    install_fakes()
    src = subprocess.run(["git", "show", ref + ":colab_server/qwen_optimize.py"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", check=True).stdout
    assert "build_kernels" not in src, "--ref 指向的版本已经是拆分后的，换 HEAD~1"
    orig = types.ModuleType("qwen_optimize_orig")
    exec(compile(src, "qwen_optimize_orig", "exec"), orig.__dict__)
    sys.path.insert(0, ROOT)
    new, ok = importlib.import_module("colab_server.qwen_optimize"), True
    for name, (good, cfg) in SCENARIOS.items():
        (a, pa), (b, pb) = (run(m._optimize_qwen35_for_fast_inference, good, cfg) for m in (orig, new))
        same = a == b and pa == pb
        ok &= same
        print("%s %s：%d 条事件；打印：%s" % ("一致" if same else "不一致", name, len(a), pa.strip().replace("\n", " / ")[:60]))
        [print("  exercise 提前结束（两边相同）：", e[1:]) for e in a if e[0] == "exercise-error"]
        if not same:
            i = next((k for k, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))
            print("  首个差异 #%d\n    原: %s\n    新: %s" % (i, a[i:i + 1], b[i:i + 1]))
            if pa != pb:
                print("  打印不同:\n", pa, "\n--\n", pb)
    sys.exit(0 if ok else 1)
if __name__ == "__main__":
    main()
