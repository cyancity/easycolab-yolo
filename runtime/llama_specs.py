"""Model specs for the shared llama-server runtime (runtime/flash_next.py).

One entry per app model id (the key of MODEL_CATALOG in app.py).  flash_next.py reads everything
model-specific from here: HF repo, file layout and sizes, MTP head, context limits, extra launch
args, and the architecture numbers behind the VRAM estimate.  Sizes are bytes from the Hugging Face
tree API (https://huggingface.co/api/models/<repo>/tree/main?recursive=true, read 2026-11-10).

qwen38_flash_next
  unsloth/Qwen3.8-Flash-Next-GGUF.  Shards `<Q>/Qwen3.8-Flash-Next-<Q>-0000i-of-0000n.gguf`; the engram
  table is one whole shard (IQ4_NL 28.8 GB for UD-IQ3_XXS..UD-Q4_K_XL, Q8_0 54.4 GB for UD-Q5_K_XL and up),
  kept in CPU RAM with `-ot`.  MTP heads in `MTP/`: `full` (4,137,429,120 B) and `shared` (2,786,568,256 B,
  reuses main embeddings; upstream llama.cpp does not load it, so `full` is the default).
  Architecture (https://huggingface.co/Qwen/Qwen3.8-Flash-Next): 48 layers = 12 x (3 Gated DeltaNet + 1 QSA);
  QSA has 2 KV heads, head dim 256, indexer dim 128; native ctx 262,144; YaRN factor 2.0 for 524,288.

qwen35_4b_smoke
  Chain-validation model for small GPUs (T4 16 GB).  There is NO Qwen3.8 GGUF below 27B from unsloth or Qwen
  (unsloth/Qwen3.8-* only has 27B, Flash-Next and 2.4T; the small Qwen3.8 GGUFs are third-party distills),
  so it uses the closest family member: unsloth/Qwen3.5-4B-MTP-GGUF, the same hybrid design (Gated DeltaNet +
  gated attention, 3:1) with an MTP head.  The MTP head is inside the main GGUF (one file per quant, no `MTP/`
  folder), so only `--spec-type draft-mtp` is needed and `-md` is not passed.
  Architecture (https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json): 32 layers = 8 x (3 linear_attention
  + 1 full_attention); 16 Q heads, 4 KV heads, head_dim 256; linear attn 32 V heads x 128 x 128; native ctx 262,144.
  UD-Q4_K_XL file: 2,990,664,000 B.  Needs a llama.cpp that has the Qwen3.5 MTP code (upstream PR #22673, merged
  2026-05-16); the fork branch is assumed to contain it (UNVERIFIED, check on the first T4 run).

KV / buffer numbers in `arch` are estimates (see flash_next.plan); only the weight sizes are exact.
"""
from __future__ import annotations

FLASH_NEXT_ID = "qwen38_flash_next"
SMOKE_ID = "qwen35_4b_smoke"

_H = 10946618
# quant -> (shard sizes in bytes in file order, 1-based index of the engram shard or None)
_FLASH_QUANTS = {
    "UD-IQ1_S": ((_H, 37568095648, 28800138432), 3),
    "UD-IQ1_M": ((_H, 39196009888, 28800138432), 3),
    "UD-IQ2_XS": ((_H, 47021008288, 28800138432), 3),
    "UD-Q2_0": ((_H, 41738806688, 28800138432), 3),
    "UD-Q2_K_XL": ((_H, 49732486272, 1037181344, 28800138432), 4),
    "UD-IQ3_XXS": ((_H, 49946027488, 4869325888, 28800138432), 4),
    "UD-Q3_K_XL": ((_H, 49742117248, 10911746208, 28800138432), 4),
    "UD-IQ4_XS": ((_H, 49604491008, 15256783680, 28800138432), 4),
    "UD-Q4_K_XL": ((_H, 49720539552, 32803030176, 28800138432), 4),
    "UD-Q5_K_XL": ((_H, 682434912, 54400261312, 49990245824, 49882416192, 3320101792), 3),
    "UD-Q6_K_XL": ((10946624, 682434912, 54400261312, 49814108512, 49419317344, 14838313984), 3),
    "Q8_0": ((10946624, 682434912, 54400261312, 49446841216, 49668930400, 34015618784), 3),
}
_SMOKE_QUANTS = {
    "UD-Q4_K_XL": ((2990664000,), None),
    "Q4_K_M": ((2834975040,), None),
    "Q6_K": ((3639654720,), None),
    "Q8_0": ((4610580800,), None),
}
_QWEN_SAMPLING = ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0"]

SPECS = {
    FLASH_NEXT_ID: {
        "label": "Flash-Next",
        "alias": "qwen3.8-flash-next",          # llama-server --alias = the model name clients send
        "match": "flash-next",                   # substring of a normalised request model name that selects it
        "repo": "unsloth/Qwen3.8-Flash-Next-GGUF",   # also the directory under the models root
        "file_fmt": "{q}/Qwen3.8-Flash-Next-{q}-{i:05d}-of-{n:05d}.gguf",
        "quants": _FLASH_QUANTS,
        "default_quant": "UD-IQ4_XS",
        "mtp_files": {
            "full": ("MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf", 4137429120),
            "shared": ("MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", 2786568256),
        },
        "default_kv": "q8_0",
        "default_ctx_k": 512,
        "max_ctx_k": 1024,
        "native_ctx": 262144,                    # above this the command adds static YaRN (factor = ctx / native)
        "draft_n": 2,
        # the 51B engram table lives in CPU RAM.  2026-11-10 on G4 without this flag the engram shard was still
        # CPU-mapped (26.8 GiB mapped, 0.6 GiB resident) and VRAM was 59.5 GB: llama.cpp keeps per-layer token
        # embeddings on the CPU by itself, so the flag only documents the intent
        # -lzm off: this fork's --lazy-mode defaults to auto, which reads per-layer embedding tables over 4 GiB (the
        # engram) row by row from disk on demand.  Once the page cache drops them (another quant loaded, a download)
        # decode fell from 112-144 to 57-76 tok/s on G4 (2026-11-10); off keeps the table resident (+100 s load).
        "extra_args": ["-ot", r"per_layer_token_embd\.weight=CPU", "-lzm", "off"],
        "sampling": _QWEN_SAMPLING,
        "arch": {"full_attn_layers": 12, "kv_heads": 2, "head_dim": 256, "indexer_dim": 128,
                 "deltanet_layers": 36, "deltanet_v_heads": 48, "deltanet_dim": 128,
                 "runtime_buffers_gib": 3.0, "mtp_buffers_gib": 0.5},
    },
    SMOKE_ID: {
        "label": "Qwen3.5-4B 冒烟",
        "alias": "qwen3.5-4b-smoke",
        "match": "qwen3.5-4b",
        "repo": "unsloth/Qwen3.5-4B-MTP-GGUF",
        "file_fmt": "Qwen3.5-4B-{q}.gguf",
        "quants": _SMOKE_QUANTS,
        "default_quant": "UD-Q4_K_XL",
        "mtp_files": {},                         # MTP head is embedded in the main GGUF
        "default_kv": "q8_0",
        "default_ctx_k": 128,
        "max_ctx_k": 256,
        "native_ctx": 262144,
        "draft_n": 3,                            # ASSUMED; the repo README example uses 6
        "extra_args": [],
        "sampling": _QWEN_SAMPLING,
        "arch": {"full_attn_layers": 8, "kv_heads": 4, "head_dim": 256, "indexer_dim": 0,
                 "deltanet_layers": 24, "deltanet_v_heads": 32, "deltanet_dim": 128,
                 "runtime_buffers_gib": 3.0, "mtp_buffers_gib": 0.5},
    },
}

# Compute capability ("7.5" -> "75") -> CMAKE_CUDA_ARCHITECTURES value and the Colab GPU it is for.
GPU_ARCHS = {
    "75": "T4", "80": "A100", "86": "A10/RTX30", "89": "L4", "90": "H100", "100": "B200", "120": "G4 (RTX PRO 6000 Blackwell)",
}
