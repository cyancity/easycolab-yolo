"""可部署的模型、显卡和达标线。fit 里的速度与显存占用是预估，部署后以自测为准。"""

SLO = {"ctxMin": 300, "tpsMin": 100, "tpsGoal": 150, "fillGoal": 75}

GPUS = [
    {"id": "T4", "vram": 16},
    {"id": "L4", "vram": 24},
    {"id": "A100", "vram": 80},
    {"id": "H100", "vram": 80},
    {"id": "G4", "vram": 96},
]


def _f(tps, fill, cls, **kw):
    return {"tps": tps, "fill": fill, "cls": cls, **kw}


MODELS = [
    {
        "id": "flash-next", "app_id": "qwen38_flash_next", "name": "Qwen3.8-Flash-Next", "kind": "语言",
        "size": "UD-IQ4_XS · 93.7 GB", "spec": "180B MoE · 6B 激活", "ctx": 512, "tps": "100–170", "slo": "ok",
        "ready": True, "load": {"precision_mode": "bf16", "context_length": 524288},
        "fit": {"A100": _f("≈100–150", 95, "low", note="仅 80GB 版"), "H100": _f("≈150–220", 92, "good", note="KV q8"),
                "G4": _f("100 / 170", 86, "good", note="MTP 实测", rec=True)},
    },
    {
        "id": "q27gguf", "app_id": "qwen38_27b_gguf", "name": "Qwen3.8-27B GGUF", "kind": "语言",
        "size": "UD-Q4_K_M · 18.7 GB", "spec": "27B Dense", "ctx": 256, "tps": "≈55–80", "slo": "bad",
        "ready": True, "load": {"precision_mode": "bf16", "context_length": 131072},
        "fit": {"L4": _f("≈30", 92, "low"), "A100": _f("≈60", 41, "low", rec=True),
                "H100": _f("≈80", 41, "low"), "G4": _f("≈55", 34, "low")},
    },
    {
        "id": "q27", "app_id": "qwen38_27b", "name": "Qwen3.8-27B", "kind": "语言",
        "size": "BF16 · 51.8 GB", "spec": "27B Dense", "ctx": 256, "tps": "≈30–40", "slo": "bad",
        "ready": True, "load": {"precision_mode": "bf16", "context_length": 131072},
        "fit": {"A100": _f("≈35", 80, "low"), "H100": _f("≈45", 80, "low", rec=True), "G4": _f("≈32", 70, "low")},
    },
    {
        "id": "qimg", "app_id": "qwen_image_21", "name": "Qwen-Image-2.1", "kind": "图像",
        "size": "BF16 · 30.9 GB", "spec": "≈4 s / 张", "ready": True, "load": {"precision_mode": "bf16"},
        "fit": {"A100": _f("4 s/张", 48, "low"), "H100": _f("2.5 s/张", 48, "low"), "G4": _f("3 s/张", 40, "low", rec=True)},
    },
    {
        "id": "zimg", "app_id": "z_image_turbo", "name": "Z-Image-Turbo", "kind": "图像",
        "size": "BF16 · 30.6 GB", "spec": "8 步出图", "ready": True, "load": {"precision_mode": "bf16"},
        "fit": {"L4": _f("3 s/张", 90, "good", rec=True), "A100": _f("1.2 s/张", 45, "low"),
                "H100": _f("0.8 s/张", 45, "low"), "G4": _f("1 s/张", 38, "low")},
    },
    {
        "id": "h3", "app_id": "minimax_h3", "name": "MiniMax-H3", "kind": "视频",
        "size": "134 GB", "spec": "文生视频", "ready": True, "load": {"precision_mode": "bf16"},
        "fit": {"G4": _f("≈6 min / 5 s", 94, "good", note="需内存卸载", rec=True)},
    },
    {
        "id": "comfy", "app_id": None, "name": "ComfyUI", "kind": "工作流",
        "size": "Wan2.2 · LTX 2.3 · Hunyuan 1.5", "spec": "网页操作", "ready": True,
        "fit": {"L4": _f("—", 70, "low"), "A100": _f("—", 80, "good"), "H100": _f("—", 80, "good"),
                "G4": _f("—", 85, "good", rec=True)},
    },
    # 只起服务、不加载模型；不在界面列出，用于冒烟测试
    {"id": "runtime", "app_id": None, "name": "空服务", "kind": "服务", "hidden": True, "ready": True,
     "fit": {g["id"]: _f("—", 0, "low") for g in GPUS}},
]

BY_ID = {m["id"]: m for m in MODELS}


def public_models():
    return [m for m in MODELS if not m.get("hidden")]
