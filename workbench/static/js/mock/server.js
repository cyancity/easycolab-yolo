// 原型用的假后端：劫持 window.fetch 处理 /api/* 和 /v1/*，其余请求透传
import { CATALOG } from "./catalog.js";
import { DOCS } from "./docs.js";
import { CUH, SCENARIOS, SERVED, buildScenario, clamp, ev, hhmm, mkAccount, mkInstance, modelOf, rnd } from "./fixtures.js";

const realFetch = window.fetch.bind(window);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// 场景只由网址的 ?mock= 决定，不写 localStorage
const scenario = () => {
  const s = new URLSearchParams(location.search).get("mock");
  return SCENARIOS.some(([k]) => k === s) ? s : "daily";
};

class HttpError extends Error {
  constructor(status, detail) { super(detail); this.status = status; this.detail = detail; }
}

let S = buildScenario(scenario());
const jobs = new Map(), pending = new Map();
let jobN = 0, acctN = 0;

/* ---------- 生命周期 ---------- */
const STEP_TEXT = [(i) => `向 Colab 申请 ${i.gpu}`, () => "安装推理服务", (i, p) => `下载权重 ${p}%`, () => "载入显存", () => "自测中"];
const STEP_DONE = [(i) => `已分配 ${i.gpu}，显存 ${i.vram_total} GB`, "推理服务就绪", "权重就绪", "载入完成"];
const maxNum = (s) => Math.max(...(String(s).match(/\d+(\.\d+)?/g) || [0]).map(Number));
const push = (i, kind, text) => i.events.push(ev(0, kind, text));

function goRunning(i) {
  const m = modelOf(i.model), fit = m?.fit?.[i.gpu];
  i.status = "running";
  i.step = 5;
  i.step_text = "自测通过";
  i.vram_used = +(i.vram_total * ((fit?.fill ?? 60) / 100)).toFixed(1);
  i.ram_used = +(i.ram_total * rnd(0.3, 0.5)).toFixed(1);
  i.util = Math.round(rnd(40, 85));
  i.util_hist = Array(30).fill(i.util);
  i.hist = { vram: Array(30).fill(i.vram_used), ram: Array(30).fill(i.ram_used) };
  i.models = [SERVED[i.model] || i.model];
  i.has_url = true;
  i.endpoint = `https://${i.model}-${Math.random().toString(16).slice(2, 6)}.tunnel.example.com`;
  if (i.model === "comfy") i.comfy_url = i.endpoint;
  if (m?.kind === "语言") {
    i.ctx = i._ctx ?? m.ctx ?? 128;
    i.bench = { tps: Math.round(maxNum(fit?.tps) * rnd(0.92, 1)), ttft: +rnd(0.3, 0.9).toFixed(1) };
    push(i, i.bench.tps >= CATALOG.slo.tpsMin ? "ok" : "warn", `自测完成：单流 ${i.bench.tps} tok/s，首字 ${i.bench.ttft} s`);
  } else push(i, "ok", "自测通过，服务就绪");
}

function tick() {
  const now = Date.now();
  if (S.cooldown) {
    S.cooldown.left -= 1;
    for (const a of S.accounts) for (const g of Object.keys(a.cooldown)) { a.cooldown[g] -= 1; if (a.cooldown[g] <= 0) delete a.cooldown[g]; }
    if (S.cooldown.left <= 0) S.cooldown = null;
  }
  for (const i of S.instances) {
    if (i.status === "starting") {
      const el = now - i._stepT0;
      if (el >= 5000) {
        const done = STEP_DONE[i.step - 1];
        if (done) push(i, "ok", typeof done === "function" ? done(i) : done);
        i.step++;
        i._stepT0 = now;
        if (i.step > 5) { goRunning(i); continue; }
        if (i.step === 3) push(i, "info", `开始下载权重 ${modelOf(i.model)?.size?.split("·").pop().trim() || ""}`);
      }
      const p = Math.min(99, Math.round(((now - i._stepT0) / 5000) * 100));
      i.step_text = STEP_TEXT[i.step - 1](i, p);
    } else if (i.status === "stopping" && now - i._stopT >= 3000) {
      i._up = Math.floor((now - i._t0) / 1000);
      i.used = +((i.cuh * i._up) / 3600).toFixed(1);
      if (i._orphan) { S.instances = S.instances.filter((x) => x !== i); continue; }
      i.status = "stopped";
      i.step_text = "已释放";
      push(i, "ok", `已释放，累计 ${i.used} CU`);
    }
    if (i._benchAt && now >= i._benchAt) {
      i._benchAt = 0;
      i.bench = { ...i.bench, tps: Math.round((i.bench?.tps || 100) * rnd(0.94, 1.06)) };
      push(i, "info", `重新自测：单流 ${i.bench.tps} tok/s`);
    }
  }
}
setInterval(tick, 1000);

// 每次轮询时，运行中实例的指标小幅抖动
function jitter(i) {
  const roll = (arr, v) => { arr.push(v); if (arr.length > 30) arr.shift(); };
  i.util = Math.round(clamp(i.util + rnd(-6, 6), 8, 99));
  i.vram_used = +clamp(i.vram_used + rnd(-0.2, 0.2), i.vram_total * 0.1, i.vram_total * 0.99).toFixed(1);
  i.ram_used = +clamp(i.ram_used + rnd(-0.4, 0.4), i.ram_total * 0.1, i.ram_total * 0.95).toFixed(1);
  roll(i.util_hist, i.util);
  roll(i.hist.vram, i.vram_used);
  roll(i.hist.ram, i.ram_used);
  if (i.kind === "语言" && Math.random() < 0.6) i.reqs += Math.floor(rnd(0, 3));
}

function stateBody() {
  const now = Date.now();
  const live = (i) => ["running", "starting", "stopping"].includes(i.status);
  const instances = S.instances.map((i) => {
    if (i.status === "running") jitter(i);
    const up = live(i) ? Math.floor((now - i._t0) / 1000) : i._up ?? 0;
    const out = Object.fromEntries(Object.entries(i).filter(([k]) => !k.startsWith("_")));
    return { ...out, uptime_s: up, used: live(i) ? +((i.cuh * up) / 3600).toFixed(2) : i.used, ctx: i.ctx, events: [...i.events] };
  });
  const accounts = S.accounts.map((a) => {
    const mine = S.instances.filter((i) => i.email === a.email);
    return {
      ...a, cooldown: { ...a.cooldown },
      burn: +mine.filter((i) => i.status === "running" || i.status === "stopping").reduce((s, i) => s + i.cuh, 0).toFixed(1),
      count: mine.filter((i) => i.status !== "stopped" && i.status !== "failed").length,
    };
  });
  return {
    accounts, instances, gateway: S.gateway, cooldown: S.cooldown, slo: CATALOG.slo, gpus: CATALOG.gpus, models: CATALOG.models,
  };
}

/* ---------- 生成结果的占位图 ---------- */
const xml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[c]));
const head = (text) => [...String(text || "")].slice(0, 12).join("");
function posterSvg(text, w, h) {
  const hue = Math.floor(rnd(0, 300));
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">` +
    `<stop offset="0" stop-color="hsl(${hue} 65% 48%)"/><stop offset="1" stop-color="hsl(${hue + 60} 60% 26%)"/></linearGradient></defs>` +
    `<rect width="${w}" height="${h}" fill="url(#g)"/><path d="M${w / 2 - h / 14} ${h / 2 - h / 10}L${w / 2 + h / 8} ${h / 2}L${w / 2 - h / 14} ${h / 2 + h / 10}Z" fill="white" fill-opacity=".85"/>` +
    `<text x="50%" y="78%" fill="white" font-size="${Math.round(h / 14)}" font-family="sans-serif" text-anchor="middle" dominant-baseline="middle">${xml(head(text))}</text></svg>`;
}
// 真 PNG 的 base64（不含前缀），图上写提示词前 12 个字
function pngB64(text, w, h) {
  const c = Object.assign(document.createElement("canvas"), { width: w, height: h }), g = c.getContext("2d"), hue = Math.floor(rnd(0, 300));
  const gr = g.createLinearGradient(0, 0, w, h);
  gr.addColorStop(0, `hsl(${hue} 65% 48%)`);
  gr.addColorStop(1, `hsl(${hue + 60} 60% 26%)`);
  g.fillStyle = gr;
  g.fillRect(0, 0, w, h);
  g.fillStyle = "white";
  g.font = `${Math.round(h / 14)}px sans-serif`;
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(head(text), w / 2, h / 2);
  return c.toDataURL("image/png").split(",")[1];
}

/* ---------- 路由 ---------- */
const find = (id) => {
  const i = S.instances.find((x) => x.id === id);
  if (!i) throw new HttpError(404, "实例不存在");
  return i;
};

function deploy({ model, email, gpu, params }) {
  const m = modelOf(model), a = S.accounts.find((x) => x.email === email);
  if (!m || !a) throw new HttpError(400, "模型或账号不存在");
  if (a.state === "error") throw new HttpError(400, "账号登录已失效，请先重新登录");
  if (!a.eligible.includes(gpu) || !m.fit?.[gpu]) throw new HttpError(400, `${a.label} 不能在 ${gpu} 上部署 ${m.name}`);
  if (a.cooldown[gpu]) throw new HttpError(400, `${gpu} 正在冷却，约 ${Math.ceil(a.cooldown[gpu] / 60)} 分钟后再试`);
  if (gpu !== "T4" && !(a.cu > 0)) throw new HttpError(400, `${a.label} 没有 CU 余额`);
  const i = mkInstance(a, model, gpu, { cuh: CUH[gpu], _ctx: params?.load?.context_length ? params.load.context_length / 1024 : null });
  i.step_text = `向 Colab 申请 ${gpu}`;
  i.events.push(ev(0, "info", `向 Colab 申请 ${gpu}`));
  S.instances.unshift(i);
  return { id: i.id };
}

function stop(id) {
  const i = find(id);
  if (i.status === "waiting") { i.status = "stopped"; push(i, "info", "已取消排队"); return { ok: true }; }
  if (!["running", "starting", "release_failed"].includes(i.status)) throw new HttpError(400, "实例当前状态不能停止");
  i.status = "stopping";
  i.step_text = "正在释放";
  i._stopT = Date.now();
  push(i, "info", "收到释放请求");
  return { ok: true };
}

function wait(id) {
  const i = find(id);
  if (i.status !== "failed" || i.fail?.kind !== "capacity") throw new HttpError(400, "只有分不到卡的部署可以排队");
  Object.assign(i, { status: "waiting", wait: { tries: 0, max: 8, next_at: Math.floor(Date.now() / 1000) + 10 * 60 } });
  delete i.fail;
  push(i, "info", "排队等卡：每 15 分钟申请一次，最多 8 次");
  return { ok: true };
}

function loginFinish({ pending_id, code }) {
  if (code === "bad") throw new HttpError(400, "授权码无效或已过期，请重新打开授权页获取");
  const p = pending.get(pending_id);
  const fix = p?.email && S.accounts.find((a) => a.email === p.email);
  if (fix) { fix.state = "ok"; fix.note = ""; if (fix.cu == null) fix.cu = 0; return { email: fix.email }; }
  acctN++;
  const a = mkAccount(`newuser${String(acctN).padStart(3, "0")}@example.com`, "Pro", 100, ["T4", "L4", "A100", "G4"]);
  S.accounts.push(a);
  return { email: a.email };
}

const ROUTES = [
  ["GET", /^\/api\/state$/, () => stateBody()],
  ["POST", /^\/api\/deploy$/, (m, b) => deploy(b)],
  ["POST", /^\/api\/instances\/([^/]+)\/stop$/, (m) => stop(m[1])],
  ["POST", /^\/api\/instances\/([^/]+)\/wait$/, (m) => wait(m[1])],
  ["POST", /^\/api\/instances\/([^/]+)\/bench$/, (m) => { find(m[1])._benchAt = Date.now() + 1500; return { ok: true }; }],
  ["POST", /^\/api\/orphans\/stop$/, (m, b) => {
    const i = S.instances.find((x) => x.status === "orphan" && x.email === b.email && x.endpoint === b.endpoint);
    if (!i) throw new HttpError(404, "没有找到这台未纳管实例");
    Object.assign(i, { status: "stopping", _orphan: true, _stopT: Date.now() });
    return { ok: true };
  }],
  ["POST", /^\/api\/accounts\/login\/start$/, (m, b) => {
    const id = "p-" + Math.random().toString(16).slice(2, 8);
    pending.set(id, { email: b.email });
    return { pending_id: id, auth_url: "https://example.com/oauth-demo?id=" + id };
  }],
  ["POST", /^\/api\/accounts\/login\/finish$/, (m, b) => loginFinish(b)],
  ["GET", /^\/api\/gateway\/key$/, () => ({ key: "sk-ec-demo-local-0000" })],
  ["POST", /^\/v1\/h3\/generate$/, (m, b) => {
    const id = `job-${++jobN}`;
    jobs.set(id, { id, progress: 0, prompt: String(b.prompt || "") });
    return { job_id: id, status: "queued" };
  }],
  // 每查一次进度 +20：排队、生成中、完成
  ["GET", /^\/v1\/h3\/jobs\/([^/]+)$/, (m) => {
    const j = jobs.get(decodeURIComponent(m[1]));
    if (!j) throw new HttpError(404, "任务不存在");
    j.progress = Math.min(100, j.progress + 20);
    if (j.progress >= 100) return { status: "completed", message: "已完成", filename: `${j.id}.mp4` };
    return { status: j.progress <= 20 ? "queued" : "running", message: j.progress <= 20 ? "排队中" : `生成中 ${j.progress}%`, filename: "" };
  }],
  ["POST", /^\/v1\/images\/generations$/, async (m, b) => {
    await sleep(2000);
    return { data: [{ b64_json: pngB64(String(b.prompt || "图像"), 512, 512) }] };
  }],
];

window.fetch = async (input, init = {}) => {
  const url = new URL(typeof input === "string" ? input : input.url, location.href);
  if (url.origin !== location.origin || !/^\/(api|v1)\//.test(url.pathname)) return realFetch(input, init);
  const method = (init.method || (typeof input === "object" && input.method) || "GET").toUpperCase();
  const json = (status, body) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  // 模型资料优先走真实后端，不可用时退回样例
  if (method === "GET" && url.pathname === "/api/model_docs") {
    try { const r = await realFetch(input, init); if (r.ok && (r.headers.get("content-type") || "").includes("json")) return r; } catch { /* 退回样例 */ }
    await sleep(rnd(150, 400));
    return json(200, DOCS);
  }
  await sleep(rnd(150, 400));
  if (S.name === "offline") throw new TypeError("Failed to fetch");
  const media = url.pathname.match(/^\/v1\/h3\/media\/([^/]+)$/);
  if (method === "GET" && media) return new Response(posterSvg(decodeURIComponent(media[1]), 1280, 720), { headers: { "Content-Type": "image/svg+xml" } });
  const route = ROUTES.find(([m, re]) => m === method && re.test(url.pathname));
  if (!route) return json(404, { detail: "接口不存在" });
  try {
    const body = init.body ? JSON.parse(init.body) : {};
    return json(200, await route[2](url.pathname.match(route[1]), body));
  } catch (e) {
    if (e instanceof HttpError) return json(e.status, { detail: e.detail });
    return json(500, { detail: "模拟服务出错：" + e.message });
  }
};
