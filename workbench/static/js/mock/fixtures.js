// 原型样例数据：四个场景的初始状态。字段名和类型对齐真实 /api/state，邮箱和密钥都是假数据
import { CATALOG } from "./catalog.js";

export const SCENARIOS = [["daily", "日常"], ["trouble", "出问题"], ["fresh", "全新"], ["offline", "离线"]];
export const CUH = { T4: 1.8, L4: 4.8, A100: 11.8, H100: 15, G4: 11.6 };
export const VRAM = { T4: 15, L4: 22.5, A100: 80, H100: 80, G4: 95.6 };
export const SERVED = {
  "flash-next": "qwen3.8-flash-next", "q35-4b-smoke": "qwen3.5-4b", q27gguf: "qwen3.8-27b-gguf", q27: "qwen3.8-27b",
  qimg: "qwen-image-2.1", zimg: "z-image-turbo", h3: "minimax-h3",
};
const ALL = ["T4", "L4", "A100", "H100", "G4"];

export const rnd = (a, b) => a + Math.random() * (b - a);
export const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
export const hhmm = (agoMin) => new Date(Date.now() - agoMin * 60000).toTimeString().slice(0, 5);
export const ev = (ago, kind, text) => [typeof ago === "string" ? ago : hhmm(ago), kind, text];
export const series = (base, spread, n = 30, lo = 0, hi = 1e9) =>
  Array.from({ length: n }, (_, i) => +clamp(base + Math.sin(i / 3) * spread * 0.5 + (Math.random() - 0.5) * spread, lo, hi).toFixed(1));
export const mask = (email) => { const u = email.split("@")[0]; return `${u.slice(0, 3)}•••${u.slice(-3)}`; };
export const modelOf = (id) => CATALOG.models.find((m) => m.id === id);

export function mkAccount(email, tier, cu, eligible, extra = {}) {
  return { email, label: mask(email), tier, cu, eligible, state: "ok", note: "", cooldown: {}, ...extra };
}

// 实例骨架：字段齐全，带下划线的是模拟器内部状态，不会出现在响应里
export function mkInstance(acct, modelId, gpu, o = {}) {
  const m = modelOf(modelId);
  return {
    id: "i-" + Math.random().toString(16).slice(2, 8), email: acct.email, label: acct.label,
    model: modelId, name: m?.name || "", kind: m?.kind || "", gpu, gpu_req: gpu,
    status: "starting", step: 1, step_text: "", error: "", uptime_s: 0, cuh: CUH[gpu], used: 0,
    vram_used: null, vram_total: VRAM[gpu], ram_used: null, ram_total: gpu === "T4" ? 12.7 : gpu === "L4" ? 53 : gpu === "G4" ? 167 : 83.5,
    ctx: null, reqs: 0, util: null, util_hist: [], bench: null, hist: { vram: [], ram: [] }, events: [],
    comfy_url: "", has_url: false, models: [], endpoint: "", _t0: Date.now(), _stepT0: Date.now(), ...o,
  };
}

// 已在运行的实例：给出显存、利用率、折线历史
export function mkRunning(acct, modelId, gpu, o) {
  const { up = 3600, vram, ram, util, tps, ttft, ctx, reqs = 0, ...rest } = o;
  const i = mkInstance(acct, modelId, gpu, {
    status: "running", step: 5, step_text: "自测通过", _t0: Date.now() - up * 1000,
    vram_used: vram, ram_used: ram, util, ctx: ctx ?? null, reqs,
    util_hist: series(util, 14, 30, 5, 99), hist: { vram: series(vram, 0.5, 30), ram: series(ram, 1.2, 30) },
    bench: tps ? { tps, ttft } : null, models: [SERVED[modelId]], has_url: true,
    endpoint: `https://${modelId}-${Math.random().toString(16).slice(2, 6)}.tunnel.example.com`, ...rest,
  });
  if (modelId === "comfy") i.comfy_url = i.endpoint;
  return i;
}

function flashNext(ultra) {
  return mkRunning(ultra, "flash-next", "G4", {
    id: "i-flash1", up: 7980, vram: 82.6, ram: 38.4, util: 88, tps: 168, ttft: 0.4, ctx: 512, reqs: 342,
    events: [ev(133, "info", "向 Colab 申请 G4"), ev(132, "ok", "已分配 G4，显存 95.6 GB"), ev(131, "info", "部署推理服务"),
      ev(127, "info", "下载权重 93.7 GB"), ev(118, "ok", "权重就绪"), ev(116, "info", "载入显存，上下文 512K"),
      ev(113, "ok", "自测通过：单流 168 tok/s，首字 0.4 s"), ev(21, "info", "累计请求 300 次")],
  });
}

const GATEWAY = { base: "http://127.0.0.1:8790/v1", key_masked: "sk-ec-••••••93d6" };

function daily() {
  const ultra = mkAccount("ultra001@example.com", "Ultra", 2380.5, ALL);
  const pro = mkAccount("prodev007@example.com", "Pro", 191.5, ["T4", "L4", "A100", "G4"]);
  const jio1 = mkAccount("jiofarm001@example.com", "Jio AI Pro", 0, ["T4"], { note: "无 Colab 计算单元" });
  const jio2 = mkAccount("jiofarm002@example.com", "Jio AI Pro", 0, ["T4"], { note: "无 Colab 计算单元" });
  const jio3 = mkAccount("jiofarm003@example.com", "Jio AI Pro", null, [], { state: "warn", note: "未登录" });
  const q27 = mkRunning(ultra, "q27gguf", "A100", {
    id: "i-q27g2", up: 3960, vram: 33, ram: 22.4, util: 55, tps: 61, ttft: 0.9, ctx: 128, reqs: 57,
    events: [ev(67, "info", "向 Colab 申请 A100"), ev(66, "ok", "已分配 A100，显存 80 GB"), ev(61, "info", "下载权重 18.7 GB"),
      ev(55, "ok", "权重就绪"), ev(53, "info", "载入显存，上下文 128K"), ev(51, "warn", "自测完成：单流 61 tok/s，低于 100 tok/s 达标线")],
  });
  const qimg = mkRunning(pro, "qimg", "L4", {
    id: "i-qimg3", up: 2400, vram: 19.8, ram: 9.6, util: 35, reqs: 12,
    events: [ev(41, "info", "向 Colab 申请 L4"), ev(40, "ok", "已分配 L4，显存 22.5 GB"), ev(36, "info", "下载权重 30.9 GB"),
      ev(28, "ok", "权重就绪，载入显存"), ev(26, "ok", "出图自测通过：约 4 s / 张"), ev(9, "info", "累计出图 12 张")],
  });
  const h3 = mkInstance(pro, "h3", "G4", {
    id: "i-h3x4", step: 3, step_text: "下载权重 42%", _t0: Date.now() - 70000, _stepT0: Date.now() - 2100,
    events: [ev(1, "info", "向 Colab 申请 G4"), ev(1, "ok", "已分配 G4，显存 95.6 GB"), ev(1, "info", "部署推理服务"), ev(0, "info", "开始下载权重 134 GB")],
  });
  const old = mkInstance(pro, "zimg", "L4", {
    id: "i-zimg5", status: "stopped", step: 5, step_text: "已释放", _up: 5400, used: 7.2, cuh: 4.8,
    events: [ev("昨 20:41", "info", "向 Colab 申请 L4"), ev("昨 20:42", "ok", "已分配 L4"), ev("昨 20:49", "ok", "出图自测通过：约 3 s / 张"),
      ev("昨 22:10", "info", "收到释放请求"), ev("昨 22:11", "ok", "已释放，累计 7.2 CU")],
  });
  return { accounts: [ultra, pro, jio1, jio2, jio3], instances: [flashNext(ultra), q27, qimg, h3, old], cooldown: null };
}

function trouble() {
  const ultra = mkAccount("ultra001@example.com", "Ultra", 2380.5, ALL, { cooldown: { H100: 540 } });
  const pro = mkAccount("prodev007@example.com", "Pro", 191.5, ["T4", "L4", "A100", "G4"]);
  const jio1 = mkAccount("jiofarm001@example.com", "Jio AI Pro", 0, ["T4"], { note: "无 Colab 计算单元" });
  const jio2 = mkAccount("jiofarm002@example.com", "Jio AI Pro", 0, ["T4"], { note: "无 Colab 计算单元" });
  const jio3 = mkAccount("jiofarm003@example.com", "Jio AI Pro", null, ["T4"], { state: "error", note: "凭据失效" });
  const rel = mkInstance(pro, "zimg", "L4", {
    id: "i-zimg6", status: "release_failed", step: 5, step_text: "释放失败", _t0: Date.now() - 3000 * 1000, cuh: 4.8,
    events: [ev(50, "info", "向 Colab 申请 L4"), ev(49, "ok", "已分配 L4"), ev(8, "info", "收到释放请求"),
      ev(8, "err", "Colab 没有确认释放（HTTP 500）"), ev(7, "warn", "1 分钟后自动重试"), ev(1, "err", "第 2 次释放仍然失败，实例可能还在扣 CU")],
  });
  const orphan = mkInstance(jio1, "", "T4", {
    id: "orphan-9c2e", name: "未纳管 T4 实例", status: "orphan", model: null, kind: "", step: 0, cuh: 1.8, ram_total: 12.7,
    endpoint: "https://orphan-9c2e.tunnel.example.com", _t0: Date.now() - 5400 * 1000,
    events: [ev(0, "warn", "在 Colab 上发现一台不是工作台启动的实例")],
  });
  const failed = mkInstance(ultra, "q27", "H100", {
    id: "i-q27f7", status: "failed", step: 1, step_text: "分配失败", _up: 180, _t0: Date.now() - 900 * 1000,
    error: "Colab 暂时分不出 H100（HTTP 503，已自动重试 3 次）。通常是该型号容量紧张，过几分钟再试，或者先选 A100 或 G4，它们同样放得下这个模型。",
    events: [ev(15, "info", "向 Colab 申请 H100"), ev(15, "warn", "H100 返回 503，30 秒后重试"), ev(14, "warn", "第 2 次重试仍是 503"),
      ev(13, "err", "第 3 次重试失败，放弃分配"), ev(13, "ok", "已自动释放，不会继续扣 CU")],
  });
  return {
    accounts: [ultra, pro, jio1, jio2, jio3], instances: [flashNext(ultra), rel, orphan, failed],
    cooldown: { left: 540, why: "H100 连续 503" },
  };
}

function fresh() {
  return {
    accounts: [mkAccount("ultra001@example.com", "Ultra", 2380.5, ALL), mkAccount("prodev007@example.com", "Pro", 191.5, ["T4", "L4", "A100", "G4"])],
    instances: [], cooldown: null,
  };
}

export function buildScenario(name) {
  const s = { daily, trouble, fresh, offline: fresh }[name]?.() || daily();
  return { ...s, gateway: { ...GATEWAY }, name };
}
