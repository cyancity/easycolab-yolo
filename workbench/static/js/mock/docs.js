// /api/model_docs 的退路数据（真实后端不可用时返回）：形状同真实接口 { docs: { [id]: doc }, issues: [] }
// flash-next、qimg、h3 三条完整，其余最小。内容为原型占位，以后端为准
const HF = "https://huggingface.co";
const line = (text, source) => ({ text, source });
const mk = (id, name, hf, summary) => ({ id, name, hf, summary, facts: [], strengths: [], weaknesses: [], usage: [], templates: [] });

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
  templates: [
    { title: "长文档总结", scene: "阅读", prompt: "请阅读下面的文档，用五条要点总结核心结论，再列出三个需要进一步确认的问题。\n\n{文档正文}", params: { temperature: 0.3 }, source: `${HF}/Qwen/Qwen3.8-Flash-Next`, source_note: "官方推荐的低温度设置" },
    { title: "代码评审", scene: "编程", prompt: "你是资深工程师。评审下面的改动，先说是否可以合并，再按严重程度列出问题，每条给出修改建议。\n\n{diff}", params: { temperature: 0.2 }, source: `${HF}/Qwen/Qwen3.8-Flash-Next`, source_note: "官方推荐的低温度设置" },
  ],
};

const qimg = {
  id: "qimg", name: "Qwen-Image-2.1", hf: `${HF}/Qwen/Qwen-Image-2.1`,
  summary: "文生图模型，中文文字渲染稳定，适合海报、封面和带字的图。A100 上约 4 秒一张。",
  facts: [
    line("权重：BF16，30.9 GB", `${HF}/Qwen/Qwen-Image-2.1`),
    line("速度：A100 约 4 s / 张，G4 约 3 s / 张", `${HF}/Qwen/Qwen-Image-2.1`),
    line("接口：POST /v1/images/generations", `${HF}/Qwen/Qwen-Image-2.1`),
  ],
  strengths: [
    line("图中的中文标题和短句基本不出错", `${HF}/Qwen/Qwen-Image-2.1`),
    line("提示词越具体，构图越可控", `${HF}/Qwen/Qwen-Image-2.1`),
  ],
  weaknesses: [line("显存紧张的卡（L4 24GB）要卸载部分权重，速度会慢", `${HF}/Qwen/Qwen-Image-2.1`)],
  usage: [line("size 支持 1024x1024、1328x1328、1664x928；参考图通过 image 字段传入", `${HF}/Qwen/Qwen-Image-2.1`)],
  templates: [
    { title: "活动海报", scene: "海报", prompt: "秋季咖啡节海报，暖色调，居中大标题「秋日咖啡节」，下方一行小字「10 月 18 日 · 城市公园」，扁平插画风格", negative: "模糊，错别字，多余文字", params: { size: "1328x1328" }, source: `${HF}/Qwen/Qwen-Image-2.1`, source_note: "按官方示例改写" },
    { title: "产品白底图", scene: "电商", prompt: "一只银色保温杯，纯白背景，柔和顶光，产品摄影，细节清晰", negative: "阴影过重，反光过曝", params: { size: "1024x1024" }, source: `${HF}/Qwen/Qwen-Image-2.1`, source_note: "按官方示例改写" },
  ],
};

const h3 = {
  id: "h3", name: "MiniMax-H3", hf: `${HF}/MiniMaxAI/MiniMax-H3`,
  summary: "文生视频模型，可同时生成画面、音效和配乐。体量大，需要 G4 并把部分权重卸载到内存，5 秒视频约 6 分钟。",
  facts: [
    line("权重：134 GB", `${HF}/MiniMaxAI/MiniMax-H3`),
    line("速度：G4 约 6 min / 5 s 视频", `${HF}/MiniMaxAI/MiniMax-H3`),
    line("接口：POST /v1/h3/generate，轮询 /v1/h3/jobs/{id}", `${HF}/MiniMaxAI/MiniMax-H3`),
  ],
  strengths: [
    line("画面、音效、配乐一次出，不用后期合成", `${HF}/MiniMaxAI/MiniMax-H3`),
    line("镜头运动描述写得越清楚，运镜越稳", `${HF}/MiniMaxAI/MiniMax-H3`),
  ],
  weaknesses: [
    line("只有 G4 放得下，且依赖内存卸载，一次只能跑一个任务", `${HF}/MiniMaxAI/MiniMax-H3`),
    line("部署要下载 134 GB 权重，耗时最长", `${HF}/MiniMaxAI/MiniMax-H3`),
  ],
  usage: [line("先小步数、低分辨率试镜头，满意后再放大出片", `${HF}/MiniMaxAI/MiniMax-H3`)],
  templates: [
    { title: "产品特写", scene: "广告", prompt: "镜头缓慢推近一台银色咖啡机，蒸汽升起，晨光从左侧窗户照入，浅景深", sound: "蒸汽声，轻微的金属碰撞声", music: "轻柔的钢琴，节奏舒缓", params: { resolution: "1280x720", frames: 121, steps: 30 }, source: `${HF}/MiniMaxAI/MiniMax-H3`, source_note: "按官方示例改写" },
    { title: "城市延时", scene: "空镜", prompt: "黄昏的城市天际线延时摄影，车流灯光拉出光轨，云层缓慢流动", sound: "远处的车流声", music: "氛围电子乐", params: { resolution: "1280x720", frames: 121, steps: 30 }, source: `${HF}/MiniMaxAI/MiniMax-H3`, source_note: "按官方示例改写" },
  ],
};

const rest = [
  mk("q35-4b-smoke", "Qwen3.5-4B · 链路冒烟", `${HF}/unsloth/Qwen3.5-4B-GGUF`, "4B 稠密小模型，只用来验证部署链路是否通，速度达不到达标线。"),
  mk("q27gguf", "Qwen3.8-27B GGUF", `${HF}/unsloth/Qwen3.8-27B-GGUF`, "27B 稠密模型的 Q4 量化版，A100 上约 60 tok/s，低于单流 100 的达标线。"),
  mk("q27", "Qwen3.8-27B", `${HF}/Qwen/Qwen3.8-27B`, "27B 稠密模型的 BF16 原版，质量最高，速度最慢。"),
  mk("zimg", "Z-Image-Turbo", `${HF}/Tongyi-MAI/Z-Image-Turbo`, "8 步出图的蒸馏模型，L4 上约 3 秒一张。"),
  mk("comfy", "ComfyUI", "https://github.com/comfyanonymous/ComfyUI", "节点式工作流界面，在浏览器里操作 Wan、LTX、Hunyuan 等视频模型。"),
];

export const DOCS = { docs: Object.fromEntries([flash, qimg, h3, ...rest].map((d) => [d.id, d])), issues: [] };
