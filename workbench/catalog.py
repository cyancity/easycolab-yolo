"""可部署的模型、显卡和达标线。fit 里的速度与显存占用是预估，部署后以自测为准。"""

# 上下文线 256K = Flash-Next 原生 262,144，不开 YaRN；负责人 2026-11-10 定
SLO = {"ctxMin": 256, "tpsMin": 100, "tpsGoal": 150, "fillGoal": 75}

# 偏好的显卡：部署推荐会给它加分，可在部署弹层里改，存在用户数据目录的 prefs.json
PREFER_GPU_DEFAULT = "G4"

GPUS = [
    {"id": "T4", "vram": 16, "name": "Tesla T4"},
    {"id": "L4", "vram": 24, "name": "NVIDIA L4"},
    # Colab 的 A100 是 40GB 版：2026-11-10 Ultra 账号实测 GPU 39 GiB，Flash-Next 预检需 75.58 GiB 载入失败
    {"id": "A100", "vram": 40, "name": "A100 SXM4 40GB"},
    {"id": "H100", "vram": 80, "name": "H100 80GB"},
    {"id": "G4", "vram": 96, "name": "RTX PRO 6000 Blackwell"},
]


def _f(tps, fill, cls, **kw):
    return {"tps": tps, "fill": fill, "cls": cls, **kw}


# 图片与视频交给 ComfyUI 的官方模板，这里只登记模板 id。部署时 Colab 端读模板声明的模型并下载
# （colab_server/comfy_templates.py），我们不维护工作流。size 取自官方 templates/index.json，同包模板共用
# 大部分文件，标「估算」的是按共用文件推的，只用于界面提示
COMFY_PACKS = [
    {"id": "h3", "title": "MiniMax-H3", "kind": "视频", "size": "约 45–57 GB", "default": True,
     "templates": ["video_minimax_h3_t2v", "video_minimax_h3_i2v"]},
    {"id": "z_image", "title": "Z-Image-Turbo", "kind": "图片", "size": "约 21 GB", "default": True,
     "templates": ["image_z_image_turbo"]},
    {"id": "qwen_image", "title": "Qwen-Image 2.1", "kind": "图片", "size": "约 27 GB",
     "templates": ["image_qwen_image_2_1_t2i", "image_qwen_image_2_1_image_edit"]},
    {"id": "wan22", "title": "Wan 2.2 14B", "kind": "视频", "size": "约 69 GB（估算）",
     "templates": ["video_wan2_2_14B_t2v", "video_wan2_2_14B_i2v"]},
    {"id": "ltx25", "title": "LTX 2.5", "kind": "视频", "size": "约 53–57 GB",
     "templates": ["video_ltx2_5_t2v", "video_ltx2_5_i2v"]},
    {"id": "hunyuan15", "title": "HunyuanVideo 1.5", "kind": "视频", "size": "约 45–60 GB（估算）",
     "templates": ["video_hunyuan_video_1.5_720p_t2v", "video_hunyuan_video_1.5_720p_i2v"]},
]
COMFY_PACK_BY_ID = {p["id"]: p for p in COMFY_PACKS}

MODELS = [
    {
        "id": "flash-next", "app_id": "qwen38_flash_next", "name": "Qwen3.8-Flash-Next", "kind": "语言",
        "runtime": "llama",
        "size": "UD-IQ4_XS · 93.7 GB", "spec": "180B MoE · 6B 激活", "ctx": 256, "ctx_max": 512, "tps": "108–136", "slo": "ok",
        "ready": True, "load": {"precision_mode": "bf16", "context_length": 262144},
        "fit": {"H100": _f("≈150–220", 92, "good", note="KV q8"),
                "G4": _f("108–136", 85, "good", note="2026-11-10 实测 · MTP", rec=True)},
    },
    # 链路冒烟：与 Flash-Next 同一套 llama-server 链路 (编译、下载、启动参数、就绪、转发、指标、卸载)，T4 上跑通后再上 A100/H100 测速。
    # fit 的速度和占用是预估 (4B Q4 约 2.8 GB 权重 + 128K q8 KV + MTP)，只用于链路验证，不对应达标线。
    {
        "id": "q35-4b-smoke", "app_id": "qwen35_4b_smoke", "name": "Qwen3.5-4B · 链路冒烟", "kind": "语言",
        "runtime": "llama",
        "size": "UD-Q4_K_XL · 3.0 GB", "spec": "4B Dense · DeltaNet 混合注意力 + MTP", "ctx": 128, "tps": "≈55–90", "slo": "bad",
        "ready": True, "load": {"precision_mode": "bf16", "context_length": 131072},
        "fit": {"T4": _f("≈55–90", 55, "low", note="链路验证，非达标线", rec=True),
                "L4": _f("≈50–80", 36, "low", note="链路验证，非达标线"),
                "A100": _f("≈150–220", 22, "low", note="链路验证，非达标线"),
                "H100": _f("≈200–300", 11, "low", note="链路验证，非达标线"),
                "G4": _f("≈200–280", 9, "low", note="链路验证，非达标线")},
    },
    {
        "id": "q27gguf", "app_id": "qwen38_27b_gguf", "name": "Qwen3.8-27B GGUF", "kind": "语言",
        "size": "UD-Q4_K_M · 18.7 GB", "spec": "27B Dense", "ctx": 256, "tps": "≈60–87", "slo": "bad",
        "ready": True, "load": {"precision_mode": "bf16", "context_length": 131072},
        "fit": {"L4": _f("≈30", 92, "low"), "A100": _f("≈60", 82, "low", rec=True),
                "H100": _f("≈80", 41, "low"), "G4": _f("87", 32, "low", note="2026-11-10 实测 · MTP · 256K")},
    },
    {
        "id": "q27", "app_id": "qwen38_27b", "name": "Qwen3.8-27B", "kind": "语言",
        "size": "BF16 · 51.8 GB", "spec": "27B Dense", "ctx": 256, "tps": "≈30–40", "slo": "bad",
        "ready": True, "load": {"precision_mode": "bf16", "context_length": 131072},
        "fit": {"H100": _f("≈45", 80, "low", rec=True), "G4": _f("≈32", 70, "low")},
    },
    {
        # 图片和视频改走 ComfyUI 官方模板（见 COMFY_PACKS）；这三个自写管线的部署项先隐藏，验证后删除
        "id": "qimg", "app_id": "qwen_image_21", "name": "Qwen-Image-2.1", "kind": "图像", "hidden": True,
        "size": "BF16 · 30.9 GB", "spec": "≈4 s / 张", "ready": True, "load": {"precision_mode": "bf16"},
        "fit": {"A100": _f("4 s/张", 96, "low", note="40GB 偏紧，未实测"), "H100": _f("2.5 s/张", 48, "low"), "G4": _f("3 s/张", 40, "low", rec=True)},
    },
    {
        "id": "zimg", "app_id": "z_image_turbo", "name": "Z-Image-Turbo", "kind": "图像", "hidden": True,
        "size": "BF16 · 30.6 GB", "spec": "8 步出图", "ready": True, "load": {"precision_mode": "bf16"},
        "fit": {"L4": _f("3 s/张", 90, "good", rec=True), "A100": _f("1.2 s/张", 90, "low", note="40GB 偏紧，未实测"),
                "H100": _f("0.8 s/张", 45, "low"), "G4": _f("1 s/张", 38, "low")},
    },
    {
        "id": "h3", "app_id": "minimax_h3", "name": "MiniMax-H3", "kind": "视频", "hidden": True,
        "size": "134 GB", "spec": "文生视频", "ready": True, "load": {"precision_mode": "bf16"},
        "fit": {"G4": _f("≈6 min / 5 s", 94, "good", note="需内存卸载", rec=True)},
    },
    {
        "id": "comfy", "app_id": None, "name": "图片与视频（ComfyUI）", "kind": "图片与视频",
        "size": "按所选模型下载", "spec": "画布 · 官方模板", "ready": True, "packs": COMFY_PACKS,
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
