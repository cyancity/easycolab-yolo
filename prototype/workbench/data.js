// 原型假数据。账号名是虚构的打码样式；带「预估」的数字未实测。
window.SLO = { ctxMin: 300, tpsMin: 100, tpsGoal: 150, fillGoal: 75 };

window.GPUS = [
  { id: "T4", vram: 16 },
  { id: "L4", vram: 24 },
  { id: "A100", vram: 80 },
  { id: "H100", vram: 80 },
  { id: "G4", vram: 96 },
];

window.ACCOUNTS = [
  { id: "ult", label: "ult•••com", plan: "AI Ultra", cu: 2500, burn: 22.6, running: 2,
    gpus: ["T4", "L4", "A100", "H100", "G4"], cache: "Flash-Next · Qwen-Image · Z-Image", cacheGB: 205, state: "ok" },
  { id: "pro", label: "pro•••com", plan: "AI Pro", cu: 191.5, burn: 4.8, running: 1,
    gpus: ["T4", "L4", "A100", "G4"], cache: "Z-Image", cacheGB: 31, state: "warn", note: "付费卡 503，8 分钟后可再试" },
  { id: "ji1", label: "ji1•••com", plan: "AI Pro · Jio", cu: 0, burn: 0, running: 0,
    gpus: ["T4"], cache: "", cacheGB: 0, state: "idle", note: "无 Colab 计算单元" },
  { id: "ji2", label: "ji2•••com", plan: "AI Pro · Jio", cu: 0, burn: 0, running: 0,
    gpus: ["T4"], cache: "", cacheGB: 0, state: "idle", note: "无 Colab 计算单元" },
  { id: "ji3", label: "ji3•••com", plan: "未登录", cu: null, burn: 0, running: 0,
    gpus: [], cache: "", cacheGB: 0, state: "login" },
];

window.MODELS = [
  { id: "flash-next", name: "Qwen3.8-Flash-Next", kind: "语言", size: "UD-IQ4_XS · 93.7 GB", spec: "180B MoE · 6B 激活",
    ctx: 512, tps: "100–170", slo: "ok", cachedOn: ["ult"],
    fit: { T4: null, L4: null,
      A100: { tps: "≈100–150", fill: 95, ctx: 384, note: "仅 80GB 版", cls: "low" },
      H100: { tps: "≈150–220", fill: 92, ctx: 512, note: "KV q8", cls: "good" },
      G4: { tps: "100 / 170", fill: 86, ctx: 512, note: "MTP 实测", cls: "good", rec: true } } },
  { id: "q27", name: "Qwen3.8-27B", kind: "语言", size: "Q4_K_M · 18.7 GB", spec: "27B Dense",
    ctx: 256, tps: "≈55–80", slo: "bad", cachedOn: ["ult", "pro"],
    fit: { T4: null, L4: { tps: "≈30", fill: 92, ctx: 64, cls: "low" },
      A100: { tps: "≈60", fill: 41, ctx: 256, cls: "low" }, H100: { tps: "≈80", fill: 41, ctx: 256, cls: "low" },
      G4: { tps: "≈55", fill: 34, ctx: 256, cls: "low" } } },
  { id: "qimg", name: "Qwen-Image-2.1", kind: "图像", size: "BF16 · 30.9 GB", spec: "≈4 s / 张 预估",
    cachedOn: ["ult"], fit: { T4: null, L4: null,
      A100: { tps: "4 s/张", fill: 48, cls: "low" }, H100: { tps: "2.5 s/张", fill: 48, cls: "low" },
      G4: { tps: "3 s/张", fill: 40, cls: "low" } } },
  { id: "zimg", name: "Z-Image-Turbo", kind: "图像", size: "BF16 · 30.6 GB", spec: "8 步出图",
    cachedOn: ["ult", "pro"], fit: { T4: null, L4: { tps: "3 s/张", fill: 90, cls: "good", rec: true },
      A100: { tps: "1.2 s/张", fill: 45, cls: "low" }, H100: { tps: "0.8 s/张", fill: 45, cls: "low" },
      G4: { tps: "1 s/张", fill: 38, cls: "low" } } },
  { id: "h3", name: "MiniMax-H3", kind: "视频", size: "134 GB", spec: "文生视频",
    cachedOn: [], fit: { T4: null, L4: null, A100: null, H100: null,
      G4: { tps: "≈6 min / 5 s", fill: 94, cls: "good", rec: true, note: "需内存卸载" } } },
  { id: "comfy", name: "ComfyUI", kind: "工作流", size: "Wan2.2 · LTX 2.3 · Hunyuan 1.5", spec: "网页操作",
    cachedOn: ["ult"], fit: { T4: null, L4: { tps: "—", fill: 70, cls: "low" },
      A100: { tps: "—", fill: 80, cls: "good" }, H100: { tps: "—", fill: 80, cls: "good" },
      G4: { tps: "—", fill: 85, cls: "good", rec: true } } },
];


window.INSTANCES = [
  { id: "i2", model: "q27", name: "Qwen3.8-27B", kind: "语言", gpu: "A100 80GB", acct: "ult", status: "running",
    uptime: "1 h 12 m", cuh: 11.8, used: 14.2, ctx: 128, tps: 58, ttft: 0.9, util: 38, vram: 33, vramTotal: 80,
    reqs: 312, model_id: "qwen3.8-27b",
    utilHist: ([30, 42, 35, 48, 39, 33, 44, 38, 41, 36, 40, 38]),
    tpsHist: ([61, 57, 59, 55, 60, 58, 56, 59, 58, 57, 59, 58]),
    events: [
      ["10:02", "ok", "分配 A100 80GB · ult•••com"],
      ["10:03", "ok", "挂载 Drive，命中权重缓存，跳过 18.7 GB 下载"],
      ["10:04", "ok", "llama.cpp 加载完成 41 s"],
      ["10:04", "ok", "自测：单流 58 tok/s，上下文 128K"],
      ["10:04", "warn", "未达标：TPS 低于 100，上下文低于 300K"],
    ] },
  { id: "i1", model: "flash-next", name: "Qwen3.8-Flash-Next", kind: "语言", gpu: "G4 96GB", acct: "ult", status: "running",
    uptime: "3 h 40 m", cuh: 10.8, used: 39.6, ctx: 512, tps: 164, ttft: 0.6, util: 93, vram: 83, vramTotal: 96,
    reqs: 2140, model_id: "qwen3.8-flash-next",
    utilHist: ([88, 92, 95, 90, 94, 93, 91, 96, 92, 93, 94, 93]),
    tpsHist: ([158, 166, 170, 161, 165, 163, 168, 160, 166, 164, 162, 164]),
    events: [
      ["06:24", "ok", "分配 G4 96GB · ult•••com"],
      ["06:25", "ok", "命中 Drive 缓存：权重 93.7 GB、llama.cpp 编译产物"],
      ["06:27", "ok", "n-gram 表放内存 28.8 GB，主体上显存 65 GB"],
      ["06:27", "ok", "自测：单流 164 tok/s（MTP 开），上下文 512K"],
    ] },
  { id: "i3", model: "zimg", name: "Z-Image-Turbo", kind: "图像", gpu: "L4 24GB", acct: "pro", status: "starting",
    step: 3, stepText: "从 Drive 缓存载入权重 62%", uptime: "2 m", cuh: 4.8, used: 0.2,
    events: [
      ["10:11", "ok", "分配 L4 · pro•••com"],
      ["10:12", "ok", "挂载 Drive"],
      ["10:12", "busy", "载入权重 62%"],
    ] },
  { id: "i4", model: "comfy", name: "ComfyUI · Wan2.2", kind: "工作流", gpu: "A100", acct: "pro", status: "failed",
    uptime: "—", cuh: 0, used: 0,
    error: "Colab 暂时分不出 A100（HTTP 503，已重试 3 次）",
    events: [
      ["09:51", "ok", "资格检查通过：A100"],
      ["09:51", "err", "申请 A100：503 ×3"],
    ] },
];
