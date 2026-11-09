// 概览：需要你处理、四个指标、实例卡片、账号条
import { STATUS, STEPS, attention, bar, dur, esc, hoursLeft, icon, isLLM, numTag, pct, sloChecks, sparkline, ui } from "../util.js";
import { bestCell } from "../sheets/launch.js";
import { failButton, failLinks, failLog } from "./failure.js";

const LIVE = ["running", "starting", "stopping"];
const sum = (xs, f) => xs.reduce((s, x) => s + (Number(f(x)) || 0), 0);
const pctAt = (v, max) => `${Math.min(Math.max((v / max) * 100, 0), 100).toFixed(1)}%`;

/* ---------- 需要你处理 ---------- */
function attn(list) {
  if (!list.length) return "";
  const sev = list.some((x) => x.sev === "err") ? "err" : "warn";
  const rows = list.map((a) => `<div class="ov-att-row ${a.sev}" data-key="${esc(a.key)}">
    <span class="ov-ic">${icon(a.sev === "err" ? "circle-x" : "triangle-alert")}</span>
    <div class="grow"><div class="ov-att-t">${esc(a.title)}</div><div class="ov-att-s">${esc(a.sub)}</div>${a.hint ? `<div class="ov-att-h">${esc(a.hint)}</div>` : ""}</div>
    <div class="ov-acts">${failLinks(a.links)}${a.actions.map(failButton).join("")}</div>${failLog(a.id, a.log)}</div>`).join("");
  return `<section class="panel ov-att ${sev}" data-key="sec-attn" aria-label="需要你处理">
    <div class="panel-h">${icon(sev === "err" ? "circle-x" : "triangle-alert")}需要你处理<span class="badge ${sev}">${list.length}</span></div>
    <div data-list>${rows}</div></section>`;
}

/* ---------- 指标 ---------- */
const stat = (k, v, sub = "", cls = "") =>
  `<div class="stat"><span class="k">${k}</span><span class="v ${cls}">${v}</span>${sub ? `<span class="ov-sub">${sub}</span>` : ""}</div>`;

function stats(d) {
  const run = d.instances.filter((i) => i.status === "running");
  const boot = d.instances.filter((i) => i.status === "starting").length;
  const met = run.filter((i) => sloChecks(i).every((c) => c.ok === true)).length;
  const burn = sum(d.accounts, (a) => a.burn), cu = sum(d.accounts, (a) => a.cu);
  const left = hoursLeft(cu, burn);
  return `<section class="panel stats" data-key="sec-stats">
    ${stat("运行", `${numTag(run.length)}<small>台</small>`, boot ? `部署中 ${numTag(boot)}` : "")}
    ${stat("达标", run.length ? `${numTag(met)} / ${numTag(run.length)}<small>台</small>` : "—", "", run.length ? (met === run.length ? "ov-ok" : "ov-low") : "")}
    ${stat("消耗", `${numTag(burn, 1)}<small>CU/h</small>`)}
    ${stat("可撑", left == null ? "—" : `约 ${numTag(left, 0)}<small>小时</small>`, "", left != null && left < 24 ? "ov-low" : "")}</section>`;
}

/* ---------- 实例卡片 ---------- */
const util = (i) => i.util == null ? "" : `<div class="ov-util"><span class="ov-cap">GPU ${numTag(i.util)}%</span>${sparkline(i.util_hist)}</div>`;

function llmBody(i) {
  const s = ui.data.slo, tps = i.bench?.tps, vp = pct(i.vram_used, i.vram_total);
  const cls = tps ? (tps >= s.tpsMin ? "ok" : "warn") : "";
  const ctxOk = sloChecks(i).find((c) => c.k === "ctx")?.ok;
  const scale = s.tpsGoal * 1.5;
  const big = tps
    ? `<span class="ov-big ${cls === "ok" ? "ov-ok" : "ov-low"}">${numTag(tps, 0)}<small>tok/s 单流</small></span>`
    : `<span class="ov-big">—</span>${i.status === "running" ? `<button class="btn sm" data-bench="${esc(i.id)}">${icon("refresh-cw")}重新自测</button>` : ""}`;
  return `<div class="ov-main">${big}${util(i)}</div>
    <div class="ov-rail ${cls}" style="--p:${tps ? pctAt(tps, scale) : "0%"}"><div class="ov-rail-t"><i></i></div>
      <span class="ov-tick" style="--at:${pctAt(s.tpsMin, scale)}"><b>${s.tpsMin}</b></span><span class="ov-tick" style="--at:${pctAt(s.tpsGoal, scale)}"><b>${s.tpsGoal}</b></span></div>
    <div class="ov-meta"><span>上下文 <b class="${ctxOk === false ? "ov-low" : ""}">${i.ctx ? `${numTag(i.ctx, 0)}K` : "—"}</b></span>
      <span class="ov-vram">显存 ${bar(vp, vp != null && vp < s.fillGoal ? "warn" : "ok")}<b>${vp == null ? "—" : `${numTag(vp)}%`}</b></span></div>`;
}

function mediaBody(i) {
  const vp = pct(i.vram_used, i.vram_total), low = vp != null && vp < ui.data.slo.fillGoal;
  return `<div class="ov-main"><span class="ov-big ${vp == null ? "" : low ? "ov-low" : "ov-ok"}">${vp == null ? "—" : numTag(vp)}<small>% 显存</small></span>${util(i)}</div>
    ${bar(vp, low ? "warn" : "ok")}`;
}

function bootBody(i) {
  const at = Math.min(Math.max(i.step || 1, 1), STEPS.length);
  const segs = STEPS.map((_, n) => `<i class="${n + 1 < at ? "done" : n + 1 === at ? "cur" : ""}"></i>`).join("");
  return `<div class="ov-steps" role="img" aria-label="第 ${at} 步，共 ${STEPS.length} 步">${segs}</div>
    <div class="ov-boot"><span class="ellipsis">${esc(i.step_text || STEPS[at - 1])}</span><span class="muted num">${dur(i.uptime_s)}</span></div>`;
}

const stopBody = (i) =>
  `<div class="ov-boot"><span class="ov-wait"><span class="spin">${icon("loader")}</span>正在释放</span><span class="muted num">已用 ${dur(i.uptime_s)}</span></div>`;

function card(i) {
  const [dot, label] = STATUS[i.status];
  const run = i.status === "running", llm = isLLM(i);
  const tag = !run ? `<span class="chip ${dot === "busy" ? "brand" : dot}">${esc(label)}</span>` : llm ? "" : `<span class="chip">${esc(i.kind)}</span>`;
  const body = i.status === "starting" ? bootBody(i) : i.status === "stopping" ? stopBody(i) : llm ? llmBody(i) : mediaBody(i);
  const foot = run
    ? `<button class="btn sm" data-copy-snippet="${esc(i.id)}">${icon("copy")}复制调用</button>${llm || !i.model ? "" : `<button class="btn sm" data-open-model="${esc(i.model)}">${icon("sparkles")}模板</button>`}` : "";
  return `<article class="panel ov-card" data-key="card-${esc(i.id)}" data-view-inst="${esc(i.id)}">
    <div class="ov-card-h"><span class="dot ${dot}"></span>
      <div class="grow"><div class="ov-card-t ellipsis">${esc(i.name || i.gpu)}</div>
        <div class="ov-card-s ellipsis">${esc(i.label)} · ${esc(i.gpu_req || i.gpu)}${run && i.uptime_s ? ` · ${dur(i.uptime_s)}` : ""}</div></div>${tag}</div>
    ${body}
    <div class="ov-card-f">${foot}<button class="btn sm ghost push" data-view-inst="${esc(i.id)}">详情${icon("arrow-right")}</button></div></article>`;
}

/* ---------- 空状态（实例页共用） ---------- */
export function emptyState() {
  const m = ui.data.models.find((x) => x.ready && x.kind === "语言" && x.slo === "ok");
  const best = m && bestCell(m.id);
  return `<div class="empty ov-empty" data-key="empty">${icon("server")}<p>还没有运行的模型</p>
    <div class="ov-empty-act"><button class="btn primary" data-new>${icon("plus")}新建部署</button>
    ${best ? `<button class="btn ov-quick" data-quick="${esc(m.id)}">${icon("rocket")}<span><b>一键部署推荐</b><small>${esc(best.label)}</small></span></button>` : ""}</div></div>`;
}

/* ---------- 账号条 ---------- */
const ACCT = { ok: ["ok", "正常"], warn: ["warn", "需注意"], error: ["err", "登录失效"], loading: ["", "加载中"] };

function accounts(d) {
  const items = d.accounts.map((a) => {
    const [dot, why] = ACCT[a.state] || ["", ""];
    const gpus = (a.eligible || []).map((g) => `<span class="chip">${esc(g)}</span>`).join("");
    return `<button class="ov-acct" data-key="acct-${esc(a.email)}" data-nav="accounts">
      <span class="ov-acct-1"><span class="dot ${dot}" role="img" aria-label="${why}"></span><b class="ellipsis">${esc(a.label)}</b><span class="ov-cap ellipsis">${esc(a.tier)}</span></span>
      <span class="ov-acct-cu">${a.cu == null ? "—" : `${numTag(a.cu, 0, true)} CU`}</span>
      <span class="ov-gpus">${gpus || `<span class="ov-cap">无可用显卡</span>`}</span></button>`;
  }).join("");
  return `<section class="ov-sec" data-key="sec-accts"><h2 class="ov-h">账号</h2><div class="ov-accts" data-list>${items}</div></section>`;
}

export function viewOverview() {
  const d = ui.data, live = d.instances.filter((i) => LIVE.includes(i.status));
  return `<div class="page" data-page="overview">${attn(attention(d, ui.dismissed))}${stats(d)}
    ${live.length ? `<section class="ov-sec" data-key="sec-insts"><h2 class="ov-h">实例</h2><div class="ov-grid" data-list data-flip>${live.map(card).join("")}</div></section>` : emptyState()}
    ${accounts(d)}</div>`;
}
