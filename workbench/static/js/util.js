// 公共小工具与界面状态
import { fmtNum, readMs } from "./motion.js";

export const $ = (s) => document.querySelector(s);
export { icon } from "./icons.js";

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
// 实例日志、错误信息来自远端，一律转义后再进 innerHTML
export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

export function dur(s) {
  if (!s || s < 60) return s ? `${Math.round(s)} s` : "—";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h} h ${m} m` : `${m} m`;
}

// 原样保留
export const num = (v, d = 1) => (v == null ? "—" : Number(v).toFixed(d).replace(/\.0+$/, ""));
export const pct = (a, b) => (a != null && b ? Math.round((100 * a) / b) : null);

// 会跳变的数字：变化时由 morph 补间。sep 为 true 时带千分位
export const numTag = (v, d = 0, sep = false) =>
  v == null || !isFinite(Number(v)) ? "—"
    : `<span data-num="${Number(v)}" data-dec="${d}"${sep ? " data-sep" : ""}>${fmtNum(v, d, sep)}</span>`;

export function sparkline(pts, warn) {
  if (!pts || pts.length < 2) return "";
  const w = 120, h = 32, max = Math.max(...pts, 1) * 1.1, min = Math.min(...pts) * 0.8;
  const xy = pts.map((v, i) => [(i / (pts.length - 1)) * w, h - ((v - min) / (max - min || 1)) * h]);
  const line = xy.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  return `<svg class="spark ${warn ? "warn" : ""}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
    <path class="area" d="${line} L${w} ${h} L0 ${h} Z"/><path class="line" d="${line}" vector-effect="non-scaling-stroke"/></svg>`;
}

export const meter = (p, cls = "") => `<span class="meter ${cls}"><i style="width:${Math.min(p || 0, 100)}%"></i></span>`;
// 同 meter，宽度走 CSS 变量，换值时由 CSS 过渡
export const bar = (p, cls = "") => `<span class="meter ${cls}" style="--p:${Math.min(Math.max(p || 0, 0), 100)}%"><i></i></span>`;

// 样例模式的「知道了」单独存，不和真实实例 id 混在一起
export const DISMISS_KEY = new URLSearchParams(location.search).get("mock") ? "proto.dismissed" : "dismissed";

// 外链只放行 https，其余返回空串
export function safeUrl(u) {
  try { const x = new URL(String(u)); return x.protocol === "https:" ? x.href : ""; } catch { return ""; }
}

const stored = (k) => { try { return JSON.parse(localStorage.getItem(k) || "[]"); } catch { return []; } };

export const ui = {
  data: null, view: "overview", sel: null, apiTab: "openai",
  launch: null, login: null, toast: "", toastKind: "", toastId: 0,
  offline: false, loading: true,
  collapsed: new Set(["ended"]),
  dismissed: new Set(stored(DISMISS_KEY)),
  confirm: null, logOpen: new Set(), msel: null, docs: null, gen: {}, tplOpen: {}, resetScroll: false,
};

export function toast(msg, kind = "") {
  ui.toast = msg;
  ui.toastKind = kind;
  ui.toastId++;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { ui.toast = ""; document.dispatchEvent(new Event("rerender")); }, readMs("--duration-toast") || 4000);
  document.dispatchEvent(new Event("rerender"));
}

/* ---------- 实例状态与达标判断 ---------- */
export const isLLM = (i) => i.kind === "语言";

// 模型的达标标签：语言模型显示上下文和预估速度，其他显示类型
export const modelChip = (m) => m.kind !== "语言" ? `<span class="chip">${esc(m.kind)}</span>`
  : `<span class="chip ${m.slo === "ok" ? "ok" : "warn"}">${num(m.ctx, 0)}K · ${esc(m.tps)} tok/s</span>`;

export const STATUS = {
  running: ["ok", "运行中"], starting: ["busy", "部署中"], stopping: ["busy", "释放中"],
  failed: ["err", "失败"], stopped: ["", "已释放"], orphan: ["warn", "未纳管"],
  release_failed: ["err", "释放失败"],
};
export const STEPS = ["分配实例", "部署服务", "下载权重", "载入显存", "自测"];

// 语言模型三项，其他模型只有显存；数据缺失时 ok 为 null
export function sloChecks(i) {
  const s = ui.data.slo, out = [];
  const vp = pct(i.vram_used, i.vram_total);
  if (isLLM(i)) {
    const tps = i.bench?.tps;
    out.push({ k: "tps", label: "单流速度", value: tps ? `${num(tps, 0)} tok/s` : "—", line: `${num(s.tpsMin, 0)} tok/s`, ok: tps ? tps >= s.tpsMin : null });
    out.push({ k: "ctx", label: "上下文", value: i.ctx ? `${num(i.ctx, 0)}K` : "—", line: `${num(s.ctxMin, 0)}K`, ok: i.ctx ? i.ctx >= s.ctxMin : null });
  }
  out.push({ k: "vram", label: "显存", value: vp == null ? "—" : `${vp}%`, line: `${num(s.fillGoal, 0)}%`, ok: vp == null ? null : vp >= s.fillGoal });
  return out;
}

// 失败实例能做的事，第一个是主按钮。容量、资格类失败里「重新部署」就是换卡或换账号，所以只留一个按钮
export function failActions(i) {
  const f = i.fail, acts = f?.actions || ["redeploy"], out = [];
  if (acts.includes("relogin")) out.push({ label: "重新登录", attr: "data-relogin", value: i.email });
  if (acts.includes("release_orphans")) out.push({ label: `释放 ${f.orphans} 台未纳管实例`, attr: "data-release-orphans", value: i.email });
  if (i.model) out.push({ label: acts.includes("switch") ? "换卡或换账号" : "重新部署", attr: "data-redeploy", value: i.id });
  if (acts.includes("logs")) out.push({ label: ui.logOpen.has(i.id) ? "收起日志" : "查看失败前日志", attr: "data-fail-log", value: i.id });
  return out.map((a, n) => ({ ...a, tone: n === 0 && a.attr !== "data-fail-log" ? "primary" : "" }));
}

// 需要处理的事，固定顺序：释放失败、未纳管、部署失败、登录失效、冷却、没达标
export function attention(data, dismissed = new Set()) {
  const g = { release: [], orphan: [], failed: [], login: [], cooldown: [], slo: [] };
  const where = (i) => `${i.label} · ${i.gpu_req || i.gpu}`;
  for (const i of data.instances) {
    if (i.status === "release_failed") {
      g.release.push({ key: `release-${i.id}`, sev: "err", kind: "release", id: i.id, title: `${i.name} 没释放成功，可能还在扣 CU`, sub: where(i),
        actions: [{ label: "重试释放", attr: "data-stop", value: i.id, tone: "danger" }] });
    } else if (i.status === "orphan") {
      g.orphan.push({ key: `orphan-${i.id}`, sev: "err", kind: "orphan", id: i.id, title: `有一台 ${i.gpu} 不是工作台启动的，正在扣 CU`, sub: where(i),
        actions: [{ label: "释放", attr: "data-orphan", value: `${i.email}|${i.endpoint}`, tone: "danger" }] });
    } else if (i.status === "failed" && !dismissed.has(i.id)) {
      const why = i.fail?.title || String(i.error || "").split("。")[0];
      g.failed.push({ key: `failed-${i.id}`, sev: "warn", kind: "failed", id: i.id, title: `${i.name} 部署失败`, sub: `${where(i)}${why ? " · " + why : ""}`,
        hint: i.fail?.hint || "", links: i.fail?.links || [], log: ui.logOpen.has(i.id) ? i.fail?.log : null,
        actions: [...failActions(i), { label: "知道了", attr: "data-dismiss", value: i.id, tone: "" }] });
    } else if (i.status === "running") {
      const bad = sloChecks(i).filter((c) => c.ok === false);
      if (bad.length) {
        g.slo.push({ key: `slo-${i.id}`, sev: "warn", kind: "slo", id: i.id, title: `${i.name} 没达标`, sub: `${where(i)} · ${bad.map((c) => c.label).join("、")}没达标`,
          actions: [{ label: "看方案", attr: "data-view-inst", value: i.id, tone: "" }] });
      }
    }
  }
  for (const a of data.accounts) {
    if (a.state === "error") {
      g.login.push({ key: `login-${a.email}`, sev: "err", kind: "login", id: a.email, title: `${a.label} 登录失效`, sub: `${a.label}${a.note ? " · " + a.note : ""}`,
        actions: [{ label: "重新登录", attr: "data-relogin", value: a.email, tone: "primary" }] });
    }
  }
  if (data.cooldown?.left > 0) {
    g.cooldown.push({ key: "cooldown", sev: "warn", kind: "cooldown", id: "", title: `分配暂停 ${Math.ceil(data.cooldown.left / 60)} 分钟`, sub: data.cooldown.why || "",
      actions: [{ label: "换一张卡部署", attr: "data-new", value: "", tone: "" }] });
  }
  return [...g.release, ...g.orphan, ...g.failed, ...g.login, ...g.cooldown, ...g.slo];
}

export const hoursLeft = (cu, burn) => (burn > 0 ? (cu || 0) / burn : null);

// OpenAI 是 curl，Anthropic 是环境变量；key 传打码或真实密钥
export function snippet(i, key, tab) {
  const base = ui.data.gateway.base;
  const model = i.models?.[0] || "（载入后显示）";
  return tab === "anthropic"
    ? `ANTHROPIC_BASE_URL=${base.replace(/\/v1$/, "")}\nANTHROPIC_AUTH_TOKEN=${key}\nANTHROPIC_MODEL=${model}`
    : `curl ${base}/chat/completions \\\n  -H "Authorization: Bearer ${key}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model":"${model}","messages":[{"role":"user","content":"你好"}]}'`;
}
