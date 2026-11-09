// 工作台原型：纯前端，假数据见 data.js
const $ = (s) => document.querySelector(s);
const acctOf = (id) => ACCOUNTS.find((a) => a.id === id);
const modelOf = (id) => MODELS.find((m) => m.id === id);
const icon = (n) => `<i data-lucide="${n}"></i>`;
const state = { view: "instances", sel: "i2", apiTab: "openai", launch: null };

const STATUS = {
  running: ["ok", "运行中"], starting: ["busy", "启动中"], failed: ["err", "失败"], action: ["warn", "需要处理"],
};

function sparkPath(pts, w = 120, h = 32) {
  const max = Math.max(...pts) * 1.1, min = Math.min(...pts) * 0.8;
  const xy = pts.map((v, i) => [(i / (pts.length - 1)) * w, h - ((v - min) / (max - min || 1)) * h]);
  const line = xy.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  return { line, area: `${line} L${w} ${h} L0 ${h} Z` };
}
const spark = (pts, warn) => {
  const p = sparkPath(pts);
  return `<svg class="spark ${warn ? "warn" : ""}" viewBox="0 0 120 32" preserveAspectRatio="none" aria-hidden="true">
    <path class="area" d="${p.area}"/><path class="line" d="${p.line}" vector-effect="non-scaling-stroke"/></svg>`;
};
const meter = (pct, cls = "") => `<span class="meter ${cls}"><i style="width:${pct}%"></i></span>`;

/* ---------- 侧栏 ---------- */
function renderNav() {
  const totalCu = ACCOUNTS.reduce((s, a) => s + (a.cu || 0), 0);
  const burn = ACCOUNTS.reduce((s, a) => s + a.burn, 0);
  const run = INSTANCES.filter((i) => i.status === "running").length;
  const item = (v, ic, label, count) =>
    `<button class="nav-item" data-nav="${v}" ${state.view === v ? 'aria-current="page"' : ""}>${icon(ic)}${label}<span class="count">${count ?? ""}</span></button>`;
  $("#nav").innerHTML = `
    <div class="brand"><span class="brand-mark">${icon("zap")}</span>EasyColab</div>
    ${item("instances", "server", "实例", INSTANCES.length)}
    ${item("accounts", "users", "账号", ACCOUNTS.length)}
    ${item("models", "boxes", "模型", MODELS.length)}
    <div class="sidebar-foot">
      <div class="fleet">
        <div class="fleet-row"><span class="muted">余额</span><b>${totalCu.toFixed(0)} CU</b></div>
        <div class="fleet-row"><span class="muted">消耗</span><b>${burn.toFixed(1)} CU/h</b></div>
        <div class="fleet-row"><span class="muted">运行</span><b>${run} 台</b></div>
      </div>
      <button class="nav-item" id="theme">${icon("sun-moon")}深浅色</button>
    </div>`;
}

/* ---------- 实例 ---------- */
function instRow(i) {
  const [dot, label] = STATUS[i.status];
  const a = acctOf(i.acct);
  let right = "";
  if (i.status === "running" && i.kind === "语言") {
    const ok = i.tps >= SLO.tpsMin && i.ctx >= SLO.ctxMin;
    right = `<span class="chip ${ok ? "ok" : "warn"}">${i.tps} tok/s</span>`;
  } else right = `<span class="r">${label}</span>`;
  const sub = i.status === "running"
    ? `<span>${i.gpu}</span>${meter(i.util, i.util < SLO.fillGoal ? "warn" : "ok")}<span>${i.util}%</span>`
    : i.status === "starting" ? `<span>${i.gpu}</span><span>${i.stepText}</span>` : `<span>${i.gpu}</span><span>申请失败 · 503</span>`;
  return `<div class="row" role="option" tabindex="0" data-inst="${i.id}" aria-selected="${state.sel === i.id}">
    <span class="dot ${dot}"></span><span class="t">${i.name}</span>${right}
    <div class="s">${sub}</div>
    <div class="s subtle" style="grid-column:2/4">${a.label} · ${i.uptime}</div></div>`;
}

function sloStat(k, v, unit, ok, hist, extra = "") {
  return `<div class="stat"><span class="k">${k}</span>
    <span class="v ${ok ? "" : "low"}">${v}<small>${unit}</small></span>${hist ? spark(hist, !ok) : ""}${extra}</div>`;
}

function headroom(i) {
  const tpsOk = i.tps >= SLO.tpsMin, ctxOk = i.ctx >= SLO.ctxMin, fillOk = i.util >= SLO.fillGoal;
  if (tpsOk && ctxOk && fillOk) return "";
  const vramPct = Math.round((i.vram / i.vramTotal) * 100);
  const reasons = [];
  if (!tpsOk) reasons.push(`单流 ${i.tps} tok/s 低于 ${SLO.tpsMin}`);
  if (!ctxOk) reasons.push(`上下文 ${i.ctx}K 低于 ${SLO.ctxMin}K`);
  if (!fillOk) reasons.push(`GPU ${i.util}%、显存 ${vramPct}%，近一半闲置`);
  const opt = (what, why, fill, tps, btn, primary) => `<div class="opt">
      <div><div class="what">${what}</div><div class="why">${why}</div></div>
      <div class="fill">${meter(fill, fill >= SLO.fillGoal ? "ok" : "warn")}<span>显存 ${fill}%</span><span class="chip ${tps >= SLO.tpsMin ? "ok" : "warn"}">≈${tps} tok/s</span></div>
      <button class="btn sm ${primary ? "primary" : ""}">${btn}</button></div>`;
  return `<section class="panel headroom">
    <div class="panel-h">${icon("gauge")}未达标，还有空间<span class="right muted">${reasons.join(" · ")}</span></div>
    <div class="opts">
      ${opt("在这台 A100 上改跑 Qwen3.8-Flash-Next", "MoE 每个 token 只算 6B 参数；n-gram 表放内存，主体 65 GB 上显存，上下文 384K", 95, 120, "替换并重启", true)}
      ${opt("把请求转到 G4 上已运行的 Flash-Next，释放本机", "实例 i1 达标且有余量；省下 11.8 CU/h", 86, 164, "转移并释放")}
      ${opt("保留 27B，开启投机解码并把上下文拉到 256K", "仍达不到 300K 和 100 tok/s，只作过渡", 58, 75, "调整参数")}
    </div></section>`;
}

function apiPanel(i) {
  const snippets = {
    openai: `curl http://127.0.0.1:8790/v1/chat/completions \\\n  -H "Authorization: Bearer sk-••••••" \\\n  -d '{"model":"${i.model_id}","messages":[...]}'`,
    anthropic: `ANTHROPIC_BASE_URL=http://127.0.0.1:8790\nANTHROPIC_AUTH_TOKEN=sk-••••••\nANTHROPIC_MODEL=${i.model_id}`,
  };
  const tab = (k, l) => `<button class="tab" role="tab" data-api="${k}" aria-selected="${state.apiTab === k}">${l}</button>`;
  return `<section class="panel"><div class="panel-h">${icon("plug")}API
      <span class="right tabs" role="tablist">${tab("openai", "OpenAI")}${tab("anthropic", "Anthropic")}</span></div>
    <div class="panel-b"><div class="kv">
      <span class="k">地址</span><span class="mono">http://127.0.0.1:8790/v1</span><button class="btn sm ghost icon-btn" aria-label="复制地址">${icon("copy")}</button>
      <span class="k">模型</span><span class="mono">${i.model_id}</span><button class="btn sm ghost icon-btn" aria-label="复制模型名">${icon("copy")}</button>
      <span class="k">密钥</span><span class="mono">sk-••••••••••••</span><button class="btn sm ghost icon-btn" aria-label="复制密钥">${icon("copy")}</button>
      <span class="k">近 1 小时</span><span>${i.reqs} 次请求 · 首字 ${i.ttft} s</span><span></span>
    </div><div class="code">${snippets[state.apiTab]}</div></div></section>`;
}

function events(i) {
  const ic = { ok: "check", warn: "triangle-alert", err: "circle-x", busy: "loader" };
  return `<section class="panel"><div class="panel-h">${icon("list")}过程</div><div class="panel-b timeline">
    ${i.events.map(([t, k, m]) => `<div class="ev ${k}"><time>${t}</time>${icon(ic[k])}<span class="msg">${m}</span><span></span></div>`).join("")}
    </div></section>`;
}

function detail(i) {
  const a = acctOf(i.acct);
  const [dot, label] = STATUS[i.status];
  const isComfy = i.model === "comfy";
  const head = `<div class="d-head"><div>
      <h2>${i.name}</h2>
      <div class="d-sub"><span class="chip ${dot === "busy" ? "brand" : dot}">${label}</span><span>${i.gpu}</span><span>${a.label} · ${a.plan}</span>
      ${i.status === "running" ? `<span>${i.uptime}</span><span>${i.cuh} CU/h · 已用 ${i.used} CU</span>` : ""}</div></div>
    <div class="d-actions">
      ${isComfy ? `<button class="btn">${icon("workflow")}打开 ComfyUI</button>` : ""}
      <button class="btn">${icon("external-link")}在 Colab 查看</button>
      ${i.status === "failed" ? "" : `<button class="btn danger">${icon("square")}停止</button>`}
    </div></div>`;
  if (i.status === "failed") {
    return head + `<div class="banner err">${icon("circle-x")}<span class="grow">${i.error}</span>
      <button class="btn sm">换 L4 重试</button><button class="btn sm" disabled>8 分钟后可重试</button></div>` + events(i);
  }
  if (i.status === "starting") {
    const steps = ["分配", "挂载 Drive", "载入权重", "自测", "API 就绪"];
    return head + `<div class="banner busy">${icon("loader")}<div class="grow">${steps[i.step - 1]}：${i.stepText}
      <div class="steps">${steps.map((_, n) => `<i class="${n + 1 < i.step ? "done" : n + 1 === i.step ? "now" : ""}"></i>`).join("")}</div></div></div>` + events(i);
  }
  const vramPct = Math.round((i.vram / i.vramTotal) * 100);
  const llm = i.kind === "语言";
  const stats = `<section class="panel stats">
    ${llm ? sloStat("单流速度", i.tps, `tok/s · 线 ${SLO.tpsMin}/${SLO.tpsGoal}`, i.tps >= SLO.tpsMin, i.tpsHist) : ""}
    ${llm ? sloStat("上下文", i.ctx + "K", `· 线 ${SLO.ctxMin}K`, i.ctx >= SLO.ctxMin, null, `<span class="muted">${i.ctx >= 512 ? "YaRN ×2" : "原生"}</span>`) : ""}
    ${sloStat("GPU 利用率", i.util, "%", i.util >= SLO.fillGoal, i.utilHist)}
    ${sloStat("显存", i.vram, `/ ${i.vramTotal} GB`, vramPct >= SLO.fillGoal, null, meter(vramPct, vramPct >= SLO.fillGoal ? "ok" : "warn"))}
  </section>`;
  return head + stats + headroom(i) + `<div class="two">${apiPanel(i)}${events(i)}</div>`;
}

function viewInstances() {
  const i = INSTANCES.find((x) => x.id === state.sel) || INSTANCES[0];
  return `<div class="split"><div class="list" role="listbox" aria-label="实例">
      <div class="list-head"><span>全部账号</span><span>${INSTANCES.length}</span></div>
      ${INSTANCES.map(instRow).join("")}</div>
    <div class="detail">${detail(i)}</div></div>`;
}

/* ---------- 账号 ---------- */
function viewAccounts() {
  const rows = ACCOUNTS.map((a) => {
    const gpus = GPUS.map((g) => `<span class="chip ${a.gpus.includes(g.id) ? "ok" : "off"}">${g.id}</span>`).join("");
    const st = a.state === "login" ? `<button class="btn sm primary">${icon("log-in")}登录</button>`
      : a.state === "warn" ? `<span class="chip warn">${a.note}</span>`
      : a.note ? `<span class="chip">${a.note}</span>` : `<span class="chip ok">正常</span>`;
    return `<tr><td><div class="acct"><b>${a.label}</b><span class="muted">${a.plan}</span></div></td>
      <td class="num">${a.cu == null ? "—" : a.cu.toLocaleString() + " CU"}</td>
      <td class="num">${a.burn ? a.burn + " CU/h" : "—"}</td>
      <td class="num">${a.running || "—"}</td>
      <td><div class="gpus">${a.gpus.length ? gpus : "—"}</div></td>
      <td>${a.cacheGB ? `${a.cacheGB} GB <span class="muted">· ${a.cache}</span>` : "—"}</td>
      <td>${st}</td></tr>`;
  }).join("");
  return `<div class="page"><section class="panel"><table class="grid">
    <thead><tr><th>账号</th><th>余额</th><th>消耗</th><th>实例</th><th>可用显卡</th><th>Drive 缓存</th><th>状态</th></tr></thead>
    <tbody>${rows}</tbody></table></section></div>`;
}

/* ---------- 模型 ---------- */
function sloChip(m) {
  if (m.kind !== "语言") return `<span class="chip">${m.kind}</span>`;
  return m.slo === "ok" ? `<span class="chip ok">${m.ctx}K · ${m.tps} tok/s</span>`
    : `<span class="chip warn">${m.ctx}K · ${m.tps} tok/s</span>`;
}
function modelCard(m, pickable) {
  const cached = m.cachedOn.map((id) => acctOf(id).label).join("、");
  return `<div class="panel mcard" ${pickable ? `role="radio" tabindex="0" data-pick="${m.id}" aria-checked="${state.launch?.model === m.id}"` : ""}>
    <h3>${m.name}</h3><div class="meta">${sloChip(m)}<span class="chip">${m.spec}</span></div>
    <span class="muted">${m.size}</span>
    <div class="foot"><span class="${cached ? "" : "subtle"}">${cached ? icon("hard-drive") + " " + cached : "未缓存"}</span></div></div>`;
}
function viewModels() {
  return `<div class="page"><div class="cards">${MODELS.map((m) => modelCard(m, false)).join("")}</div></div>`;
}

/* ---------- 新建部署 ---------- */
function launchStep1() { return `<div class="pick">${MODELS.map((m) => modelCard(m, true)).join("")}</div>`; }

function launchStep2() {
  const m = modelOf(state.launch.model);
  const accts = ACCOUNTS.filter((a) => a.state !== "login");
  const head = `<tr><th></th>${GPUS.map((g) => `<th>${g.id} · ${g.vram}GB</th>`).join("")}</tr>`;
  const rows = accts.map((a) => `<tr><th class="rowh"><div class="acct"><b>${a.label}</b><span class="muted">${a.cu} CU${m.cachedOn.includes(a.id) ? " · 已缓存" : ""}</span>${a.state === "warn" ? `<span class="chip warn">503 冷却中</span>` : ""}</div></th>
    ${GPUS.map((g) => {
      const f = m.fit[g.id];
      if (!a.gpus.includes(g.id)) return `<td><div class="cell no">无资格</div></td>`;
      if (!f) return `<td><div class="cell no">放不下</div></td>`;
      if (a.cu <= 0) return `<td><div class="cell no">无余额</div></td>`;
      const on = state.launch.acct === a.id && state.launch.gpu === g.id;
      return `<td><button class="cell ${f.cls}" role="radio" aria-checked="${on}" data-cell="${a.id}|${g.id}">
        <b>${f.tps}</b><span>显存 ${f.fill}%${f.note ? " · " + f.note : ""}</span></button></td>`;
    }).join("")}</tr>`).join("");
  return `<table class="matrix">${head}${rows}</table>
    <div class="legend"><span><span class="sw" style="background:var(--success)"></span>达标且显存 ≥ ${SLO.fillGoal}%</span>
    <span><span class="sw" style="background:var(--warning)"></span>不达标或浪费</span><span>数字为预估，启动后自测替换</span></div>`;
}

function launchStep3() {
  const L = state.launch;
  const seg = (key, vals) => `<span class="seg">${vals.map(([v, l]) =>
    `<button data-seg="${key}|${v}" aria-pressed="${String(L[key]) === String(v)}">${l}</button>`).join("")}</span>`;
  const total = L.gpu === "G4" ? 96 : 80;
  const w = { iq3: 53, iq4: 65, q4: 82 }[L.quant];
  const kv = Math.round((L.ctx / 512) * (L.kvq ? 6.4 : 12.9) * 10) / 10;
  const rt = L.mtp ? 5 : 3;
  const used = w + kv + rt, fill = Math.round((used / total) * 100), fits = used <= total;
  const tps = Math.round((L.gpu === "H100" ? 150 : 100) * (L.mtp ? 1.65 : 1) * (L.quant === "iq3" ? 1.1 : L.quant === "q4" ? 0.9 : 1));
  const pct = (v) => ((v / total) * 100).toFixed(1);
  return `<div class="field"><label>上下文</label>${seg("ctx", [[256, "256K"], [384, "384K"], [512, "512K"]])}<span class="${L.ctx >= SLO.ctxMin ? "muted" : "chip warn"}">${L.ctx > 256 ? "YaRN" : "低于 300K"}</span></div>
    <div class="field"><label>量化</label>${seg("quant", [["iq3", "IQ3_XXS"], ["iq4", "IQ4_XS"], ["q4", "Q4_K_XL"]])}<span></span></div>
    <div class="field"><label>KV 缓存</label>${seg("kvq", [[false, "f16"], [true, "q8"]])}<span></span></div>
    <div class="field"><label>MTP 投机解码</label>${seg("mtp", [[true, "开"], [false, "关"]])}<span></span></div>
    <div class="budget">
      <div class="row2"><b>${L.gpu} 显存 ${used.toFixed(1)} / ${total} GB</b><span class="${fits ? (fill >= SLO.fillGoal ? "" : "chip warn") : "chip err"}">${fits ? fill + "%" : "超出 " + (used - total).toFixed(1) + " GB"}</span></div>
      <div class="bar"><i class="w" style="width:${pct(w)}%"></i><i class="kv" style="width:${pct(kv)}%"></i><i class="rt" style="width:${pct(rt)}%"></i></div>
      <div class="row2 muted"><span><span class="sw" style="background:var(--chart-1)"></span>权重 ${w} GB　<span class="sw" style="background:var(--chart-2)"></span>KV ${kv} GB　<span class="sw" style="background:var(--chart-4)"></span>运行时 ${rt} GB</span><span>n-gram 表 28.8 GB 放内存</span></div>
      <div class="row2"><span>预估单流</span><span class="chip ${tps >= SLO.tpsGoal ? "ok" : tps >= SLO.tpsMin ? "brand" : "warn"}">≈${tps} tok/s</span></div>
    </div>`;
}

function renderLaunch() {
  const L = state.launch;
  if (!L) { $("#sheet").innerHTML = ""; return; }
  const names = ["模型", "机器", "参数"];
  const body = [launchStep1, launchStep2, launchStep3][L.step - 1]();
  const m = modelOf(L.model), a = acctOf(L.acct);
  const can = L.step === 1 ? !!L.model : L.step === 2 ? !!L.gpu : true;
  const summary = [m?.name, a && L.gpu ? `${L.gpu} · ${a.label}` : ""].filter(Boolean).join(" · ");
  $("#sheet").innerHTML = `<div class="scrim" data-close><div class="sheet" role="dialog" aria-modal="true" aria-label="新建部署">
    <div class="sheet-h"><h2>新建部署</h2><div class="stepper">${names.map((n, k) =>
      `<span class="${k + 1 < L.step ? "done" : ""}" ${k + 1 === L.step ? 'aria-current="step"' : ""}>${k + 1} ${n}</span>`).join("")}</div>
      <button class="btn ghost icon-btn" style="margin-left:auto" data-close aria-label="关闭">${icon("x")}</button></div>
    <div class="sheet-b">${body}</div>
    <div class="sheet-f"><span class="grow muted">${summary}</span>
      ${L.step > 1 ? `<button class="btn" data-back>上一步</button>` : ""}
      ${L.step < 3 ? `<button class="btn primary" data-next ${can ? "" : "disabled"}>下一步</button>`
        : `<button class="btn primary" data-start>${icon("rocket")}启动${m?.cachedOn.includes(L.acct) ? " · Drive 已缓存" : ""}</button>`}
    </div></div></div>`;
  lucide.createIcons();
}

/* ---------- 主渲染与事件 ---------- */
function render() {
  renderNav();
  const titles = { instances: "实例", accounts: "账号", models: "模型" };
  $("#title").textContent = titles[state.view];
  $("#view").innerHTML = { instances: viewInstances, accounts: viewAccounts, models: viewModels }[state.view]();
  renderLaunch();
  lucide.createIcons();
}

document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-nav],[data-inst],[data-api],[data-pick],[data-cell],[data-seg],[data-next],[data-back],[data-start],[data-close],#theme,#new");
  if (!t) return;
  const L = state.launch;
  if (t.dataset.nav) state.view = t.dataset.nav;
  else if (t.dataset.inst) state.sel = t.dataset.inst;
  else if (t.dataset.api) state.apiTab = t.dataset.api;
  else if (t.id === "theme") document.documentElement.classList.toggle("dark");
  else if (t.id === "new") state.launch = { step: 1, model: "flash-next", acct: null, gpu: null, ctx: 512, quant: "iq4", kvq: false, mtp: true };
  else if (t.dataset.pick) { L.model = t.dataset.pick; L.acct = L.gpu = null; }
  else if (t.dataset.cell) { [L.acct, L.gpu] = t.dataset.cell.split("|"); L.kvq = L.gpu !== "G4"; L.ctx = L.gpu === "A100" ? 384 : 512; }
  else if (t.dataset.seg) { const [k, v] = t.dataset.seg.split("|"); L[k] = v === "true" ? true : v === "false" ? false : isNaN(v) ? v : Number(v); }
  else if (t.dataset.next !== undefined) {
    L.step++;
    if (L.step === 2 && !L.gpu) {
      const m = modelOf(L.model);
      const recGpu = Object.keys(m.fit).find((g) => m.fit[g]?.rec);
      const acct = ACCOUNTS.find((a) => a.gpus.includes(recGpu) && a.cu > 0 && m.cachedOn.includes(a.id))
        || ACCOUNTS.find((a) => a.gpus.includes(recGpu) && a.cu > 0);
      if (acct) { L.acct = acct.id; L.gpu = recGpu; }
    }
  }
  else if (t.dataset.back !== undefined) L.step--;
  else if (t.dataset.start !== undefined) { state.launch = null; state.view = "instances"; state.sel = "i3"; }
  else if (t.dataset.close !== undefined && (e.target === t || t.tagName === "BUTTON")) state.launch = null;
  else return;
  render();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.launch) { state.launch = null; render(); }
  if ((e.key === "Enter" || e.key === " ") && e.target.matches(".row,[data-pick]")) { e.preventDefault(); e.target.click(); }
});
render();
