// 模型目录、显卡和达标线：取自 2026-11-10 本机 /api/state 快照，只含公开目录信息
export const CATALOG = {
  "slo": {
    "ctxMin": 256,
    "tpsMin": 100,
    "tpsGoal": 150,
    "fillGoal": 75
  },
  "gpus": [
    {
      "id": "T4",
      "vram": 16,
      "name": "Tesla T4"
    },
    {
      "id": "L4",
      "vram": 24,
      "name": "NVIDIA L4"
    },
    {
      "id": "A100",
      "vram": 40,
      "name": "A100 SXM4 40GB"
    },
    {
      "id": "H100",
      "vram": 80,
      "name": "H100 80GB"
    },
    {
      "id": "G4",
      "vram": 96,
      "name": "RTX PRO 6000 Blackwell"
    }
  ],
  "models": [
    {
      "id": "flash-next",
      "app_id": "qwen38_flash_next",
      "name": "Qwen3.8-Flash-Next",
      "kind": "语言",
      "runtime": "llama",
      "size": "UD-IQ4_XS · 93.7 GB",
      "spec": "180B MoE · 6B 激活",
      "ctx": 256,
      "ctx_max": 512,
      "tps": "108–136",
      "slo": "ok",
      "ready": true,
      "load": {
        "precision_mode": "bf16",
        "context_length": 262144
      },
      "fit": {
        "H100": {
          "tps": "≈150–220",
          "fill": 92,
          "cls": "good",
          "note": "KV q8"
        },
        "G4": {
          "tps": "108–136",
          "fill": 85,
          "cls": "good",
          "note": "2026-11-10 实测 · MTP",
          "rec": true
        }
      }
    },
    {
      "id": "q35-4b-smoke",
      "app_id": "qwen35_4b_smoke",
      "name": "Qwen3.5-4B · 链路冒烟",
      "kind": "语言",
      "runtime": "llama",
      "size": "UD-Q4_K_XL · 3.0 GB",
      "spec": "4B Dense · DeltaNet 混合注意力 + MTP",
      "ctx": 128,
      "tps": "≈55–90",
      "slo": "bad",
      "ready": true,
      "load": {
        "precision_mode": "bf16",
        "context_length": 131072
      },
      "fit": {
        "T4": {
          "tps": "≈55–90",
          "fill": 55,
          "cls": "low",
          "note": "链路验证，非达标线",
          "rec": true
        },
        "L4": {
          "tps": "≈50–80",
          "fill": 36,
          "cls": "low",
          "note": "链路验证，非达标线"
        },
        "A100": {
          "tps": "≈150–220",
          "fill": 22,
          "cls": "low",
          "note": "链路验证，非达标线"
        },
        "H100": {
          "tps": "≈200–300",
          "fill": 11,
          "cls": "low",
          "note": "链路验证，非达标线"
        },
        "G4": {
          "tps": "≈200–280",
          "fill": 9,
          "cls": "low",
          "note": "链路验证，非达标线"
        }
      }
    },
    {
      "id": "q27gguf",
      "app_id": "qwen38_27b_gguf",
      "name": "Qwen3.8-27B GGUF",
      "kind": "语言",
      "size": "UD-Q4_K_M · 18.7 GB",
      "spec": "27B Dense",
      "ctx": 256,
      "tps": "≈60–87",
      "slo": "bad",
      "ready": true,
      "load": {
        "precision_mode": "bf16",
        "context_length": 131072
      },
      "fit": {
        "L4": {
          "tps": "≈30",
          "fill": 92,
          "cls": "low"
        },
        "A100": {
          "tps": "≈60",
          "fill": 82,
          "cls": "low",
          "rec": true
        },
        "H100": {
          "tps": "≈80",
          "fill": 41,
          "cls": "low"
        },
        "G4": {
          "tps": "87",
          "fill": 32,
          "cls": "low",
          "note": "2026-11-10 实测 · MTP · 256K"
        }
      }
    },
    {
      "id": "q27",
      "app_id": "qwen38_27b",
      "name": "Qwen3.8-27B",
      "kind": "语言",
      "size": "BF16 · 51.8 GB",
      "spec": "27B Dense",
      "ctx": 256,
      "tps": "≈30–40",
      "slo": "bad",
      "ready": true,
      "load": {
        "precision_mode": "bf16",
        "context_length": 131072
      },
      "fit": {
        "H100": {
          "tps": "≈45",
          "fill": 80,
          "cls": "low",
          "rec": true
        },
        "G4": {
          "tps": "≈32",
          "fill": 70,
          "cls": "low"
        }
      }
    },
    {
      "id": "qimg",
      "hidden": true,
      "app_id": "qwen_image_21",
      "name": "Qwen-Image-2.1",
      "kind": "图像",
      "size": "BF16 · 30.9 GB",
      "spec": "≈4 s / 张",
      "ready": true,
      "load": {
        "precision_mode": "bf16"
      },
      "fit": {
        "A100": {
          "tps": "4 s/张",
          "fill": 96,
          "cls": "low",
          "note": "40GB 偏紧，未实测"
        },
        "H100": {
          "tps": "2.5 s/张",
          "fill": 48,
          "cls": "low"
        },
        "G4": {
          "tps": "3 s/张",
          "fill": 40,
          "cls": "low",
          "rec": true
        }
      }
    },
    {
      "id": "zimg",
      "hidden": true,
      "app_id": "z_image_turbo",
      "name": "Z-Image-Turbo",
      "kind": "图像",
      "size": "BF16 · 30.6 GB",
      "spec": "8 步出图",
      "ready": true,
      "load": {
        "precision_mode": "bf16"
      },
      "fit": {
        "L4": {
          "tps": "3 s/张",
          "fill": 90,
          "cls": "good",
          "rec": true
        },
        "A100": {
          "tps": "1.2 s/张",
          "fill": 90,
          "cls": "low",
          "note": "40GB 偏紧，未实测"
        },
        "H100": {
          "tps": "0.8 s/张",
          "fill": 45,
          "cls": "low"
        },
        "G4": {
          "tps": "1 s/张",
          "fill": 38,
          "cls": "low"
        }
      }
    },
    {
      "id": "h3",
      "hidden": true,
      "app_id": "minimax_h3",
      "name": "MiniMax-H3",
      "kind": "视频",
      "size": "134 GB",
      "spec": "文生视频",
      "ready": true,
      "load": {
        "precision_mode": "bf16"
      },
      "fit": {
        "G4": {
          "tps": "≈6 min / 5 s",
          "fill": 94,
          "cls": "good",
          "note": "需内存卸载",
          "rec": true
        }
      }
    },
    {
      "id": "comfy",
      "app_id": null,
      "name": "图片与视频（ComfyUI）",
      "kind": "图片与视频",
      "size": "按所选模型下载",
      "spec": "画布 · 官方模板",
      "ready": true,
      "packs": [
        { "id": "h3", "title": "MiniMax-H3", "kind": "视频", "size": "约 45–57 GB", "default": true },
        { "id": "z_image", "title": "Z-Image-Turbo", "kind": "图片", "size": "约 21 GB", "default": true },
        { "id": "qwen_image", "title": "Qwen-Image 2.1", "kind": "图片", "size": "约 27 GB", "default": false },
        { "id": "wan22", "title": "Wan 2.2 14B", "kind": "视频", "size": "约 69 GB（估算）", "default": false },
        { "id": "ltx25", "title": "LTX 2.5", "kind": "视频", "size": "约 53–57 GB", "default": false },
        { "id": "hunyuan15", "title": "HunyuanVideo 1.5", "kind": "视频", "size": "约 45–60 GB（估算）", "default": false }
      ],
      "fit": {
        "L4": {
          "tps": "—",
          "fill": 70,
          "cls": "low"
        },
        "A100": {
          "tps": "—",
          "fill": 80,
          "cls": "good"
        },
        "H100": {
          "tps": "—",
          "fill": 80,
          "cls": "good"
        },
        "G4": {
          "tps": "—",
          "fill": 85,
          "cls": "good",
          "rec": true
        }
      }
    }
  ]
};
