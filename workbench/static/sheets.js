// 新建部署（三步）与添加账号两个弹层
import { esc, icon, num, ui } from "./util.js";
import { modelCard } from "./views.js";

const modelOf = (id) => ui.data.models.find((m) => m.id === id);
const acctOf = (email) => ui.data.accounts.find((a) => a.email === email);
const CTX = [[131072, "128K"], [262144, "256K"], [393216, "384K"], [524288, "512K"]];

export function newLaunch(model = null) {
  ui.launch = { step: 1, model: model || ui.data.models.find((m) => m.ready)?.id, email: null, gpu: null, ctx: null, busy: false };
}

function cellState(m, a, g) {
  if (!a.eligible?.includes(g.id)) return ["no", "无资格"];
  if (!m.fit?.[g.id]) return ["no", "放不下"];
  if (g.id !== "T4" && !(a.cu > 0)) return ["no", "无余额"];
  if (a.cooldown?.[g.id]) return ["no", `冷却 ${Math.ceil(a.cooldown[g.id] / 60)} 分`];
  return null;
}

export function recommend() {
  const L = ui.launch, m = modelOf(L.model);
  const cands = [];
  for (const a of ui.data.accounts) for (const g of ui.data.gpus) {
    if (cellState(m, a, g)) continue;
    const f = m.fit[g.id];
    cands.push({ a, g, score: (f.cls === "good" ? 100 : 0) + (f.rec ? 50 : 0) + f.fill / 10 + Math.min(a.cu || 0, 500) / 100 });
  }
  cands.sort((x, y) => y.score - x.score);
  if (cands[0]) { L.email = cands[0].a.email; L.gpu = cands[0].g.id; }
}

function step1() {
  return `<div class="pick">${ui.data.models.map((m) => modelCard(m, true, ui.launch.model === m.id)).join("")}</div>`;
}

function step2() {
  const L = ui.launch, m = modelOf(L.model);
  const accts = ui.data.accounts.filter((a) => a.state !== "loading" && a.state !== "error");
  const head = `<tr><th></th>${ui.data.gpus.map((g) => `<th>${esc(g.id)} · ${num(g.vram, 0)}GB</th>`).join("")}</tr>`;
  const rows = accts.map((a) => `<tr><th class="rowh"><div class="acct"><b>${esc(a.label)}</b>
      <span class="muted">${a.cu == null ? "" : Math.round(a.cu) + " CU · "}${esc(a.tier || "")}</span></div></th>
    ${ui.data.gpus.map((g) => {
      const s = cellState(m, a, g);
      if (s) return `<td><div class="cell no">${s[1]}</div></td>`;
      const f = m.fit[g.id], on = L.email === a.email && L.gpu === g.id;
      return `<td><button class="cell ${esc(f.cls)}" role="radio" aria-checked="${on}" data-cell="${esc(a.email)}|${esc(g.id)}">
        <b>${esc(f.tps)}</b><span>显存 ${num(f.fill, 0)}%${f.note ? " · " + esc(f.note) : ""}</span></button></td>`;
    }).join("")}</tr>`).join("");
  return `<table class="matrix">${head}${rows}</table>
    <div class="legend"><span><span class="sw ok"></span>达标且显存 ≥ ${num(ui.data.slo.fillGoal, 0)}%</span>
    <span><span class="sw warn"></span>不达标或浪费</span><span>数字为预估，部署后自测替换</span></div>`;
}

function step3() {
  const L = ui.launch, m = modelOf(L.model), a = acctOf(L.email), f = m.fit[L.gpu];
  const isLLM = m.kind === "语言";
  const ctx = L.ctx ?? m.load?.context_length;
  const seg = CTX.filter(([v]) => v <= (m.ctx || 256) * 1024).map(([v, l]) => `<button data-ctx="${v}" aria-pressed="${ctx === v}">${l}</button>`).join("");
  return `<div class="kv summary">
      <span class="k">模型</span><span>${esc(m.name)} · ${esc(m.size)}</span><span></span>
      <span class="k">机器</span><span>${esc(L.gpu)} · ${esc(a.label)}（${esc(a.tier)}）</span><span></span>
      <span class="k">预估</span><span>${esc(f.tps)} · 显存 ${num(f.fill, 0)}%</span><span></span>
    </div>
    ${isLLM ? `<div class="field"><label>上下文</label><span class="seg">${seg}</span>
      <span class="${ctx / 1024 >= ui.data.slo.ctxMin ? "muted" : "chip warn"}">${ctx / 1024 >= ui.data.slo.ctxMin ? "" : "低于 300K"}</span></div>` : ""}
    ${isLLM && m.slo !== "ok" ? `<div class="banner warn">${icon("triangle-alert")}<span class="grow">这个模型预估单流低于 ${num(ui.data.slo.tpsMin, 0)} tok/s。</span></div>` : ""}`;
}

export function renderLaunch() {
  const L = ui.launch;
  if (!L) return "";
  const names = ["模型", "机器", "确认"];
  const body = [step1, step2, step3][L.step - 1]();
  const m = modelOf(L.model), a = acctOf(L.email);
  const can = L.step === 1 ? !!m?.ready : L.step === 2 ? !!L.gpu : !L.busy;
  const summary = [m?.name, a && L.gpu ? `${L.gpu} · ${a.label}` : ""].filter(Boolean).map(esc).join(" · ");
  return `<div class="scrim" data-close><div class="sheet" role="dialog" aria-modal="true" aria-label="新建部署">
    <div class="sheet-h"><h2>新建部署</h2><div class="stepper">${names.map((n, k) =>
      `<span class="${k + 1 < L.step ? "done" : ""}" ${k + 1 === L.step ? 'aria-current="step"' : ""}>${k + 1} ${n}</span>`).join("")}</div>
      <button class="btn ghost icon-btn push" data-close aria-label="关闭">${icon("x")}</button></div>
    <div class="sheet-b">${body}</div>
    <div class="sheet-f"><span class="grow muted">${summary}</span>
      ${L.step > 1 ? `<button class="btn" data-back>上一步</button>` : ""}
      ${L.step < 3 ? `<button class="btn primary" data-next ${can ? "" : "disabled"}>下一步</button>`
        : `<button class="btn primary" data-start ${can ? "" : "disabled"}>${icon("rocket")}${L.busy ? "提交中" : "启动"}</button>`}
    </div></div></div>`;
}

export function renderLogin() {
  const g = ui.login;
  if (!g) return "";
  const body = g.step === 1
    ? `<p>在打开的页面里选择要添加的 Google 账号并同意授权，复制页面上的授权码。</p>
       <button class="btn primary" data-login-open ${g.busy ? "disabled" : ""}>${icon("external-link")}打开 Google 授权页</button>`
    : `<label class="stack"><span>授权码</span><input id="code" class="input" autocomplete="off" spellcheck="false" value="${esc(g.code || "")}"></label>
       <div class="row-actions"><a class="btn ghost" href="${esc(g.auth_url)}" target="_blank" rel="noopener">${icon("external-link")}重新打开授权页</a>
       <button class="btn primary" data-login-finish ${g.busy ? "disabled" : ""}>${g.busy ? "验证中" : "添加账号"}</button></div>`;
  return `<div class="scrim" data-close-login><div class="sheet narrow" role="dialog" aria-modal="true" aria-label="添加账号">
    <div class="sheet-h"><h2>添加账号</h2><button class="btn ghost icon-btn push" data-close-login aria-label="关闭">${icon("x")}</button></div>
    <div class="sheet-b">${body}${g.error ? `<div class="banner err">${icon("circle-x")}<span class="grow">${esc(g.error)}</span></div>` : ""}</div></div></div>`;
}
