// 占位：新建部署弹层（三步），由视图写手整体替换。逻辑沿用现行 sheets.js
import { esc, icon, num, ui } from "../util.js";

const modelOf = (id) => ui.data.models.find((m) => m.id === id);
const acctOf = (email) => ui.data.accounts.find((a) => a.email === email);
const CTX = [[131072, "128K"], [262144, "256K"], [393216, "384K"], [524288, "512K"]];

export function newLaunch(model = null) {
  ui.launch = { step: 1, dir: 1, model: model || ui.data.models.find((m) => m.ready)?.id, email: null, gpu: null, ctx: null, rec: null, busy: false };
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

function card(m, picked) {
  const attrs = m.ready ? `role="radio" tabindex="0" data-pick="${esc(m.id)}" aria-checked="${picked}"` : `aria-disabled="true"`;
  return `<div class="panel ph-card" data-key="pick-${esc(m.id)}" ${attrs}><b>${esc(m.name)}</b>
    <span class="muted">${esc(m.kind)} · ${esc(m.spec)} · ${esc(m.size)}</span></div>`;
}

function step2() {
  const L = ui.launch, m = modelOf(L.model);
  const accts = ui.data.accounts.filter((a) => a.state !== "loading" && a.state !== "error");
  const head = `<tr><th></th>${ui.data.gpus.map((g) => `<th>${esc(g.id)} · ${num(g.vram, 0)}GB</th>`).join("")}</tr>`;
  const rows = accts.map((a) => `<tr><th class="ph-rowh">${esc(a.label)} <span class="muted">${a.cu == null ? "" : Math.round(a.cu) + " CU"}</span></th>
    ${ui.data.gpus.map((g) => {
      const s = cellState(m, a, g);
      if (s) return `<td><div class="ph-cell no">${s[1]}</div></td>`;
      const f = m.fit[g.id], on = L.email === a.email && L.gpu === g.id, key = `${a.email}|${g.id}`;
      return `<td><button class="ph-cell ${esc(f.cls)}" role="radio" aria-checked="${on}" data-cell="${esc(key)}">
        <b>${esc(f.tps)}</b><span>显存 ${num(f.fill, 0)}%${L.rec === key ? " · 推荐" : ""}</span></button></td>`;
    }).join("")}</tr>`).join("");
  return `<table class="ph-matrix">${head}${rows}</table>`;
}

function step3() {
  const L = ui.launch, m = modelOf(L.model), a = acctOf(L.email), f = m.fit[L.gpu];
  const llm = m.kind === "语言";
  const ctx = L.ctx ?? m.load?.context_length;
  const seg = CTX.filter(([v]) => v <= (m.ctx || 256) * 1024).map(([v, l]) => `<button data-ctx="${v}" aria-pressed="${ctx === v}">${l}</button>`).join("");
  return `<div class="kv"><span class="k">模型</span><span>${esc(m.name)} · ${esc(m.size)}</span><span></span>
      <span class="k">机器</span><span>${esc(L.gpu)} · ${esc(a.label)}（${esc(a.tier)}）</span><span></span>
      <span class="k">预估</span><span>${esc(f.tps)} · 显存 ${num(f.fill, 0)}%</span><span></span></div>
    ${llm ? `<div class="field"><label>上下文</label><span class="seg">${seg}</span></div>` : ""}
    ${llm && m.slo !== "ok" ? `<div class="banner warn">${icon("triangle-alert")}<span class="grow">这个模型预估单流低于 ${num(ui.data.slo.tpsMin, 0)} tok/s。</span></div>` : ""}`;
}

export function renderLaunch() {
  const L = ui.launch;
  if (!L) return "";
  const names = ["模型", "机器", "确认"];
  const body = L.step === 1 ? `<div class="ph-grid">${ui.data.models.map((m) => card(m, L.model === m.id)).join("")}</div>` : L.step === 2 ? step2() : step3();
  const m = modelOf(L.model), a = acctOf(L.email);
  const can = L.step === 1 ? !!m?.ready : L.step === 2 ? !!L.gpu : !L.busy;
  const summary = [m?.name, a && L.gpu ? `${L.gpu} · ${a.label}` : ""].filter(Boolean).map(esc).join(" · ");
  return `<div class="scrim" data-key="sheet-launch" data-close><div class="sheet" role="dialog" aria-modal="true" aria-label="新建部署">
    <div class="sheet-h"><h2>新建部署</h2><div class="stepper">${names.map((n, k) =>
      `<span class="${k + 1 < L.step ? "done" : ""}" ${k + 1 === L.step ? 'aria-current="step"' : ""}>${k + 1} ${n}</span>`).join("")}</div>
      <button class="btn ghost icon-btn push" data-close aria-label="关闭">${icon("x")}</button></div>
    <div class="sheet-b"><div class="step-body" data-key="step-${L.step}" data-step="${L.step}" style="--dir:${L.dir || 1}">${body}</div></div>
    <div class="sheet-f"><span class="grow muted">${summary}</span>
      ${L.step > 1 ? `<button class="btn" data-back>上一步</button>` : ""}
      ${L.step < 3 ? `<button class="btn primary" data-next ${can ? "" : "disabled"}>下一步</button>`
        : `<button class="btn primary" data-start ${can ? "" : "disabled"}>${icon("rocket")}${L.busy ? "提交中" : "启动"}</button>`}
    </div></div></div>`;
}
