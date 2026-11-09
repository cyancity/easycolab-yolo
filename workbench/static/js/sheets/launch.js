// 新建部署弹层：固定三步（模型 → 账号 × 显卡矩阵 → 确认）。打分与不可选原因沿用定稿
import { esc, icon, modelChip, num, ui } from "../util.js";

const modelOf = (id) => ui.data.models.find((m) => m.id === id);
const acctOf = (email) => ui.data.accounts.find((a) => a.email === email);
const CTX = [[131072, "128K"], [262144, "256K"], [393216, "384K"], [524288, "512K"]];
const KINDS = ["语言", "图像", "视频", "工作流"];
const STEP_NAMES = ["模型", "机器", "确认"];

// icon() 不带额外 class，这里补一个（建议并入内核：icon(name, cls)）
const ic = icon;

export function newLaunch(model = null) {
  // 传入的模型不存在或没就绪时退回第一个可部署的，免得按不存在的模型算矩阵
  const ok = ui.data.models.some((m) => m.id === model && m.ready);
  ui.launch = { step: 1, dir: 1, model: ok ? model : ui.data.models.find((m) => m.ready)?.id, email: null, gpu: null, ctx: null, rec: null, busy: false };
}

function cellState(m, a, g) {
  if (!a.eligible?.includes(g.id)) return ["no", "无资格"];
  if (!m.fit?.[g.id]) return ["no", "放不下"];
  if (g.id !== "T4" && !(a.cu > 0)) return ["no", "无余额"];
  if (a.cooldown?.[g.id]) return ["no", `冷却 ${Math.ceil(a.cooldown[g.id] / 60)} 分`];
  return null;
}

// 打分：达标 +100，推荐 +50，显存占用 /10，余额（上限 500）/100
function ranked(modelId) {
  const m = modelOf(modelId), cands = [];
  if (!m) return cands;
  for (const a of ui.data.accounts) {
    if (a.state === "loading" || a.state === "error") continue;
    for (const g of ui.data.gpus) {
      if (cellState(m, a, g)) continue;
      const f = m.fit[g.id];
      cands.push({ a, g, m, score: (f.cls === "good" ? 100 : 0) + (f.rec ? 50 : 0) + f.fill / 10 + Math.min(a.cu || 0, 500) / 100 });
    }
  }
  return cands.sort((x, y) => y.score - x.score);
}

export function recommend() {
  const best = ranked(ui.launch.model)[0];
  if (best) { ui.launch.email = best.a.email; ui.launch.gpu = best.g.id; ui.launch.rec = `${best.a.email}|${best.g.id}`; }
}

// 不改 ui，空状态用
export function bestCell(modelId) {
  const b = ranked(modelId)[0];
  return b ? { email: b.a.email, gpu: b.g.id, label: `${b.m.name} · ${b.g.id} · ${b.a.label}` } : null;
}

/* ---------- 步骤条 ---------- */
function stepper(cur) {
  return `<ol class="stepper sh-steps" aria-label="部署步骤">${STEP_NAMES.map((n, k) => {
    const done = k + 1 < cur;
    return `<li class="sh-st ${done ? "done is-checked" : ""}" data-key="st-${k + 1}" ${k + 1 === cur ? 'aria-current="step"' : ""}>
      <span class="sh-dot"><span class="sh-n">${k + 1}</span>${ic("check", "check-in")}</span><span class="sh-st-n">${n}</span>
      ${k < STEP_NAMES.length - 1 ? `<i class="sh-line" aria-hidden="true"><b style="--f:${done ? 1 : 0}"></b></i>` : ""}</li>`;
  }).join("")}</ol>`;
}

/* ---------- 第 1 步：模型 ---------- */
function card(m, picked, recId) {
  const attrs = m.ready ? `role="radio" tabindex="0" data-pick="${esc(m.id)}" aria-checked="${picked}"` : `aria-disabled="true"`;
  return `<div class="sh-model ${m.ready ? "" : "is-off"}" data-key="pick-${esc(m.id)}" ${attrs}>
    <div class="sh-model-h"><b class="ellipsis">${esc(m.name)}</b>${recId === m.id ? `<span class="chip brand">推荐</span>` : ""}
      <span class="sh-tick">${ic("check", "check-in")}</span></div>
    <div class="sh-model-m">${modelChip(m)}<span class="muted ellipsis">${esc(m.spec)}</span></div>
    <div class="sh-model-s subtle">${m.ready ? esc(m.size) : `<span class="chip brand">${esc(m.pending || "整理中")}</span>`}</div></div>`;
}

function step1() {
  const L = ui.launch, list = ui.data.models;
  const rec = list.find((m) => m.ready && m.kind === "语言" && m.slo === "ok")?.id;
  const kinds = [...KINDS, ...new Set(list.map((m) => m.kind).filter((k) => !KINDS.includes(k)))];
  const groups = kinds.map((k) => ({ k, items: list.filter((m) => m.kind === k) })).filter((g) => g.items.length);
  return `<div class="sh-groups" role="radiogroup" aria-label="模型">${groups.map((g) =>
    `<section class="sh-group" data-key="grp-${esc(g.k)}"><h3 class="sh-gh">${esc(g.k)}</h3>
      <div class="sh-models">${g.items.map((m) => card(m, L.model === m.id, rec)).join("")}</div></section>`).join("")}</div>`;
}

/* ---------- 第 2 步：矩阵 ---------- */
function cell(m, a, g) {
  const L = ui.launch, s = cellState(m, a, g);
  if (s) return `<td><div class="sh-cell no" aria-disabled="true">${esc(s[1])}</div></td>`;
  const f = m.fit[g.id], key = `${a.email}|${g.id}`, on = L.email === a.email && L.gpu === g.id;
  const label = `${a.label}，${g.id}，预估 ${f.tps}，显存 ${num(f.fill, 0)}%${L.rec === key ? "，推荐" : ""}`;
  return `<td><button class="sh-cell ${f.cls === "good" ? "good" : "low"}" role="radio" aria-checked="${on}" aria-label="${esc(label)}" data-cell="${esc(key)}">
    <span class="sh-tick">${ic("check", "check-in")}</span>
    ${L.rec === key ? `<span class="sh-rec">推荐</span>` : ""}
    <b>${esc(f.tps)}</b><span>显存 ${num(f.fill, 0)}%</span>${f.note ? `<small class="ellipsis">${esc(f.note)}</small>` : ""}</button></td>`;
}

function step2() {
  const m = modelOf(ui.launch.model);
  const accts = ui.data.accounts.filter((a) => a.state !== "loading" && a.state !== "error");
  if (!accts.length) return `<div class="empty">${icon("users")}<p>没有可用账号，先添加一个</p></div>`;
  const head = `<tr><th></th>${ui.data.gpus.map((g) => `<th scope="col" data-key="gh-${esc(g.id)}"><b>${esc(g.id)}</b> <span class="muted">${num(g.vram, 0)}GB</span></th>`).join("")}</tr>`;
  const rows = accts.map((a) => `<tr data-key="row-${esc(a.email)}">
    <th scope="row" class="sh-rowh"><div class="sh-who"><b class="ellipsis">${esc(a.label)}</b><span class="chip">${esc(a.tier || "")}</span></div>
      <span class="muted">${a.cu == null ? "—" : `${Math.round(a.cu).toLocaleString("en-US")} CU`}</span></th>
    ${ui.data.gpus.map((g) => cell(m, a, g)).join("")}</tr>`).join("");
  const s = ui.data.slo;
  return `<div class="sh-scroll"><table class="sh-matrix" role="presentation"><thead>${head}</thead><tbody>${rows}</tbody></table></div>
    <div class="sh-legend"><span><i class="sw good"></i>达标且显存 ≥ ${num(s.fillGoal, 0)}%</span><span><i class="sw low"></i>不达标或浪费</span>
      <span><i class="sw no"></i>不可选</span><span class="grow"></span><span>数字为预估，部署后自测替换</span></div>`;
}

/* ---------- 第 3 步：确认 ---------- */
function step3() {
  const L = ui.launch, m = modelOf(L.model), a = acctOf(L.email), f = m.fit[L.gpu];
  const llm = m.kind === "语言", s = ui.data.slo;
  const ctx = L.ctx ?? m.load?.context_length;
  const seg = CTX.filter(([v]) => v <= (m.ctx || 256) * 1024)
    .map(([v, l]) => `<button data-ctx="${v}" aria-pressed="${ctx === v}">${l}</button>`).join("");
  const lowCtx = llm && ctx != null && ctx / 1024 < s.ctxMin;
  return `<div class="sh-sum panel"><div class="kv">
      <span class="k">模型</span><span>${esc(m.name)} · ${esc(m.size)}</span><span></span>
      <span class="k">机器</span><span>${esc(L.gpu)} · ${esc(a.label)}（${esc(a.tier)}）</span><span></span>
      <span class="k">预估</span><span>${esc(f.tps)} · 显存 ${num(f.fill, 0)}%</span><span></span></div></div>
    ${llm ? `<div class="sh-ctx" data-key="ctx"><span class="muted">上下文</span><span class="seg" role="group" aria-label="上下文">${seg}</span>
      ${lowCtx ? `<span class="chip warn" data-key="ctx-low">低于 ${num(s.ctxMin, 0)}K</span>` : ""}</div>` : ""}
    ${llm && m.slo !== "ok" ? `<div class="banner warn" data-key="slo-warn">${icon("triangle-alert")}<span class="grow">这个模型预估单流低于 ${num(s.tpsMin, 0)} tok/s。</span></div>` : ""}`;
}

/* ---------- 弹层 ---------- */
export function renderLaunch() {
  const L = ui.launch;
  if (!L) return "";
  const body = L.step === 1 ? step1() : L.step === 2 ? step2() : step3();
  const m = modelOf(L.model), a = acctOf(L.email);
  const can = L.step === 1 ? !!m?.ready : L.step === 2 ? !!L.gpu : !L.busy;
  const summary = [m?.name, a && L.gpu && L.step > 1 ? `${L.gpu} · ${a.label}` : ""].filter(Boolean).map(esc).join(" · ");
  return `<div class="scrim" data-key="sheet-launch" data-close><div class="sheet sh-launch" role="dialog" aria-modal="true" aria-label="新建部署">
    <div class="sheet-h"><h2>新建部署</h2>${stepper(L.step)}
      <button class="btn ghost icon-btn push" data-close aria-label="关闭">${icon("x")}</button></div>
    <div class="sheet-b" data-key="step-${L.step}" data-step="${L.step}" style="--dir:${L.dir || 1}">${body}</div>
    <div class="sheet-f"><span class="grow muted ellipsis">${summary}</span>
      ${L.step > 1 ? `<button class="btn" data-key="f-back" data-back ${L.busy ? "disabled" : ""}>上一步</button>` : ""}
      ${L.step < 3 ? `<button class="btn primary" data-key="f-next" data-next ${can ? "" : "disabled"}>下一步</button>`
        : `<button class="btn primary" data-key="f-start" data-start ${can ? "" : "disabled"}>${L.busy ? ic("loader", "spin") : icon("rocket")}${L.busy ? "提交中" : "启动"}</button>`}
    </div></div></div>`;
}
