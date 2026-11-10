// /api/model_docs 的退路数据（真实后端不可用时返回）：形状同真实接口 { docs: { [id]: doc }, issues: [] }
// flash-next 一条完整，其余最小。内容为原型占位，以后端为准
const HF = "https://huggingface.co";
const line = (text, source) => ({ text, source });
const mk = (id, name, hf, summary) => ({ id, name, hf, summary, facts: [], strengths: [], weaknesses: [], usage: [] });

const flash = {
  id: "flash-next", name: "Qwen3.8-Flash-Next", hf: `${HF}/Qwen/Qwen3.8-Flash-Next`,
  summary: "180B 总参数的 MoE，每个 token 只激活 6B。在 G4 上带 MTP 推测解码，单流速度能过 100 tok/s 达标线，是目前唯一同时满足速度、上下文和显存三项的语言模型。",
  facts: [
    line("规格：180B MoE，6B 激活", `${HF}/Qwen/Qwen3.8-Flash-Next`),
    line("量化：UD-IQ4_XS，93.7 GB", `${HF}/unsloth/Qwen3.8-Flash-Next-GGUF`),
    line("上下文：原生 256K，YaRN ×2 到 512K", `${HF}/Qwen/Qwen3.8-Flash-Next`),
    line("实测速度：G4 单流 100 到 170 tok/s（开 MTP）", `${HF}/Qwen/Qwen3.8-Flash-Next`),
  ],
  strengths: [
    line("MoE 每步只算 6B，单流速度远高于同体量的稠密模型", `${HF}/Qwen/Qwen3.8-Flash-Next`),
    line("512K 上下文，显存占用约 86%，G4 上不浪费", `${HF}/Qwen/Qwen3.8-Flash-Next`),
  ],
  weaknesses: [
    line("权重 93.7 GB，只放得下 A100 80GB 以上的卡，A100 上已是极限", `${HF}/unsloth/Qwen3.8-Flash-Next-GGUF`),
    line("首次部署要下载近 100 GB，约需 10 分钟", `${HF}/unsloth/Qwen3.8-Flash-Next-GGUF`),
  ],
  usage: [
    line("OpenAI 兼容接口，模型名 qwen3.8-flash-next，直接填工作台的接入地址即可", `${HF}/Qwen/Qwen3.8-Flash-Next`),
    line("长文档问答把上下文拉满 512K；日常对话 256K 就够，速度更稳", `${HF}/Qwen/Qwen3.8-Flash-Next`),
  ],
};

const rest = [
  mk("q35-4b-smoke", "Qwen3.5-4B · 链路冒烟", `${HF}/unsloth/Qwen3.5-4B-GGUF`, "4B 稠密小模型，只用来验证部署链路是否通，速度达不到达标线。"),
  mk("q27gguf", "Qwen3.8-27B GGUF", `${HF}/unsloth/Qwen3.8-27B-GGUF`, "27B 稠密模型的 Q4 量化版，A100 上约 60 tok/s，低于单流 100 的达标线。"),
  mk("q27", "Qwen3.8-27B", `${HF}/Qwen/Qwen3.8-27B`, "27B 稠密模型的 BF16 原版，质量最高，速度最慢。"),
  mk("comfy", "图片与视频（ComfyUI）", "https://github.com/comfyanonymous/ComfyUI", "在 ComfyUI 画布里用官方模板出图出视频。部署时勾选要用的模型包，只下载选中的权重。"),
];

export const DOCS = { docs: Object.fromEntries([flash, ...rest].map((d) => [d.id, d])), issues: [] };
