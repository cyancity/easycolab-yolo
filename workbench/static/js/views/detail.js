// 实例详情：instances.js 提供 .detail 容器，这里返回它的内部 HTML。纯函数，只读 ui
import { STATUS, STEPS, dur, esc, failActions, icon, isLLM, num, numTag, pct, snippet, sloChecks, sparkline, ui, waitInfo } from "../util.js";
import { genPanel } from "./generate.js";
import { failButton, failLinks, failLog } from "./failure.js";

const spin = (name) => icon(name, "spin");
const slo = () => ui.data.slo;
const hasComfy = (i) => typeof i.comfy_url === "string" && i.comfy_url.startsWith("https://");

/* ---------- 头部 ---------- */
function actions(i) {
  const id = esc(i.id), out = [];
  if (i.status === "running") {
    if (!isLLM(i) && ui.docs?.[i.model]?.templates?.length) out.push(`<button class="btn" data-key="a-tpl" data-open-model="${esc(i.model)}">${icon("sparkles")}模板</button>`);
    if (hasComfy(i)) out.push(`<a class="btn" data-key="a-comfy" href="${esc(i.comfy_url)}" target="_blank" rel="noopener">${icon("workflow")}打开 ComfyUI</a>`);
    if (isLLM(i)) out.push(`<button class="btn" data-key="a-bench" data-bench="${id}">${icon("timer")}重新自测</button>`);
    const arm = ui.confirm === i.id;
    out.push(`<button class="btn danger ${arm ? "armed" : ""}" data-key="a-stop" data-stop="${id}">${icon("square")}${arm ? "确认停止" : "停止"}</button>`);
  } else if (i.status === "starting") {
    out.push(`<button class="btn danger" data-key="a-cancel" data-stop="${id}">${icon("square")}取消并释放</button>`);
  } else if (i.status === "waiting") {
    out.push(`<button class="btn" data-key="a-cancel" data-stop="${id}">${icon("x")}取消排队</button>`);
  } else if (i.status === "stopped" && i.model) {
    out.push(`<button class="btn" data-key="a-redeploy" data-redeploy="${id}">${icon("rotate-ccw")}重新部署</button>`);
  }
  return out.join("");
}

function head(i) {
  const [dot, label] = STATUS[i.status] || ["", i.status];
  const run = i.status === "running";
  return `<div class="d-head dt-head" data-key="d-head"><div class="dt-title"><h2>${esc(i.name)}</h2>
      <div class="d-sub"><span class="chip ${dot === "busy" ? "brand" : dot}">${esc(label)}</span><span>${esc(i.gpu)}</span><span>${esc(i.label)}</span>
      ${run ? `<span>${dur(i.uptime_s)}</span><span>${numTag(i.cuh, 1)} CU/h · 已用 ${numTag(i.used, 2)} CU</span>` : ""}</div></div>
    <div class="d-actions">${actions(i)}</div></div>`;
}

/* ---------- 运行中：达标条、指标、余量 ---------- */
function sloBar(i) {
  const cs = sloChecks(i), bad = cs.filter((c) => c.ok === false).length;
  const [tone, ic, text] = bad ? ["warn", "triangle-alert", `${bad} 项没达标`]
    : cs.every((c) => c.ok) ? ["ok", "check-circle", "达标"] : ["", "info", "暂无数据"];
  return `<section class="panel dt-slo" data-key="d-slo"><span class="dt-sum ${tone}">${icon(ic)}${text}</span>
    <div class="dt-checks">${cs.map((c) => `<span class="chip dt-chk ${c.ok === false ? "warn" : c.ok ? "ok" : ""}" data-key="chk-${c.k}">
      ${icon(c.ok === false ? "x" : c.ok ? "check" : "dot")}<span>${esc(c.label)}</span><b>${esc(c.value)}</b><span class="dt-line">线 ${esc(c.line)}</span></span>`).join("")}</div></section>`;
}

function stat(key, k, v, unit, ok, extra = "") {
  return `<div class="stat" data-key="st-${key}"><span class="k">${k}</span><span class="v ${ok === false ? "low" : ""}">${v}<small>${unit}</small></span>${extra}</div>`;
}

function stats(i) {
  const s = slo(), vp = pct(i.vram_used, i.vram_total), tps = i.bench?.tps;
  const vLow = vp != null && vp < s.fillGoal;
  return `<section class="panel stats dt-stats" data-key="d-stats">
    ${isLLM(i) ? stat("tps", "单流速度", numTag(tps, 0), `tok/s · 线 ${num(s.tpsMin, 0)}/${num(s.tpsGoal, 0)}`, tps ? tps >= s.tpsMin : undefined) : ""}
    ${isLLM(i) ? stat("ctx", "上下文", i.ctx ? `${numTag(i.ctx, 0)}K` : "—", `线 ${num(s.ctxMin, 0)}K`, i.ctx ? i.ctx >= s.ctxMin : undefined) : ""}
    ${i.util != null ? stat("util", "GPU 利用率", `${numTag(i.util, 0)}%`, "近 60 s · 线 " + num(s.fillGoal, 0) + "%", i.util >= s.fillGoal, sparkline(i.util_hist, i.util < s.fillGoal)) : ""}
    ${stat("vram", "显存", numTag(i.vram_used, 1), `/ ${num(i.vram_total, 0)} GB`, vp == null ? undefined : !vLow, sparkline(i.hist?.vram, vLow))}
    ${stat("ram", "内存", numTag(i.ram_used, 1), `/ ${num(i.ram_total, 0)} GB`, undefined, sparkline(i.hist?.ram))}
  </section>`;
}

// 余量：哪项没达标、怎么办。能直接执行的建议带 act 按钮
function headroom(i) {
  const s = slo(), vp = pct(i.vram_used, i.vram_total), reasons = [], tips = [];
  if (isLLM(i) && i.bench?.tps && i.bench.tps < s.tpsMin) {
    reasons.push(`单流 ${num(i.bench.tps, 0)} tok/s 低于 ${num(s.tpsMin, 0)}`);
    if (i.model !== "flash-next" && ui.data.models.some((m) => m.id === "flash-next" && m.ready)) {
      tips.push({ what: "换 Qwen3.8-Flash-Next", why: "MoE 每个 token 只算 6B 参数，G4 实测 108–136 tok/s（MTP）",
        act: `<button class="btn primary sm" data-quick="flash-next">${icon("rocket")}换 Qwen3.8-Flash-Next</button>` });
    }
  }
  if (isLLM(i) && i.ctx && i.ctx < s.ctxMin) {
    reasons.push(`上下文 ${num(i.ctx, 0)}K 低于 ${num(s.ctxMin, 0)}K`);
    tips.push({ what: "拉长上下文", why: "Flash-Next 原生 256K，开 YaRN ×2 到 512K" });
  }
  if (vp != null && vp < s.fillGoal) {
    reasons.push(`显存只用了 ${vp}%`);
    tips.push({ what: "吃满显存", why: "换更大的量化或更大的模型，或在同卡上并发加载第二个模型；也可以换显存更小的卡省 CU" });
  }
  if (i.util != null && i.util < s.fillGoal) {
    reasons.push(`GPU 利用率 ${Math.round(i.util)}%`);
    if (isLLM(i)) tips.push({ what: "提高并发或换更大的模型", why: "单流请求喂不满算力时，调高并发槽位，或换参数更多的模型把算力用上" });
  }
  if (!reasons.length) return "";
  return `<section class="panel dt-room" data-key="d-room"><div class="panel-h">${icon("gauge")}未达标，还有空间
      <span class="right muted">${reasons.join(" · ")}</span></div>
    ${tips.length ? `<div class="dt-opts">${tips.map((t) => `<div class="dt-opt"><div class="dt-opt-t"><div class="dt-what">${t.what}</div><div class="dt-why">${t.why}</div></div>${t.act || ""}</div>`).join("")}</div>` : ""}</section>`;
}

/* ---------- 接入 ---------- */
function apiPanel(i) {
  const g = ui.data.gateway, model = i.models?.[0] || "";
  const tab = (k, l) => `<button class="tab" role="tab" data-api="${k}" aria-selected="${ui.apiTab === k}">${l}</button>`;
  const row = (k, v, attr, label) => `<span class="k">${k}</span><span class="mono ellipsis">${v}</span>
      <button class="btn sm ghost icon-btn" ${attr} aria-label="${label}" ${attr.includes('=""') ? "disabled" : ""}>${icon("copy")}</button>`;
  return `<section class="panel dt-api" data-key="d-api"><div class="panel-h">${icon("plug")}接入
      <span class="right dt-tools"><span class="tabs" role="tablist">${tab("openai", "OpenAI")}${tab("anthropic", "Anthropic")}</span>
        <button class="btn sm" data-copy-snippet="${esc(i.id)}">${icon("copy")}复制调用</button></span></div>
    <div class="panel-b dt-api-b"><div class="kv">
      ${row("地址", esc(g.base), `data-copy="${esc(g.base)}"`, "复制地址")}
      ${row("模型", esc(model || "（载入后显示）"), `data-copy="${esc(model)}"`, "复制模型名")}
      ${row("密钥", esc(g.key_masked), "data-copy-key", "复制密钥")}
      <span class="k">请求</span><span>${numTag(i.reqs ?? 0, 0)} 次${i.bench?.ttft ? ` · 首字 ${num(i.bench.ttft)} s` : ""}</span><span></span>
    </div><div class="code">${esc(snippet(i, g.key_masked, ui.apiTab))}</div></div></section>`;
}

/* ---------- 过程时间线：新的在上 ---------- */
// 后端只回最近 30 条，下标会随截断整体错位；key 取时间和内容，相同的再按出现次序区分
function evKeys(list) {
  const seen = new Map();
  return list.map(([t, k, m]) => {
    let h = 0;
    for (const c of `${t}|${k}|${m}`) h = (h * 31 + c.charCodeAt(0)) | 0;
    const base = (h >>> 0).toString(36), n = (seen.get(base) || 0) + 1;
    seen.set(base, n);
    return `ev-${base}-${n}`;
  });
}
const EV_ICON = { ok: "check", info: "dot", warn: "triangle-alert", err: "circle-x" };

function events(i) {
  const keys = evKeys(i.events || []);
  const evs = (i.events || []).map((e, idx) => [keys[idx], ...e]).reverse();
  return `<section class="panel dt-events" data-key="d-events"><div class="panel-h">${icon("list")}过程</div>
    <div class="panel-b dt-tl" data-list>${evs.length ? evs.map(([idx, t, k, m]) =>
      `<div class="dt-ev ${esc(k)}" data-key="${idx}"><time>${esc(t)}</time>${icon(EV_ICON[k] || "dot")}<span class="dt-msg">${esc(m)}</span></div>`).join("")
      : `<span class="muted">暂无记录</span>`}</div></section>`;
}

/* ---------- 部署中：五段进度 ---------- */
function progress(i) {
  const step = i.step || 1;
  const seg = STEPS.map((name, n) => {
    const st = n + 1 < step ? "done" : n + 1 === step ? "now" : "todo";
    const ic = st === "done" ? icon("check") : st === "now" ? spin("loader") : "";
    return `<li class="dt-step ${st}" data-key="seg-${n}">${n ? `<span class="dt-link" style="--p:${n < step ? 100 : 0}%"><i></i></span>` : ""}<span class="dt-node">${ic}</span><span class="dt-lbl">${name}</span></li>`;
  }).join("");
  return `<section class="panel dt-prog" data-key="d-prog"><ol class="dt-steps" aria-label="部署进度">${seg}</ol>
    <div class="dt-now"><span class="dt-now-t">${esc(i.step_text || STEPS[step - 1])}</span><span class="muted">已用 ${dur(i.uptime_s)}</span></div></section>`;
}

/* ---------- 横幅 ---------- */
const banner = (tone, ic, text, btn) =>
  `<div class="banner dt-banner ${tone}" data-key="d-banner" role="${tone === "err" ? "alert" : "status"}">${icon(ic)}<span class="grow dt-banner-t">${text}</span>${btn ? `<span class="dt-banner-acts">${btn}</span>` : ""}</div>`;

// 失败横幅：归类标题、原始错误、下一步提示，右侧是外链和按钮；日志展开后另起一块
function failBanner(i) {
  const f = i.fail, id = esc(i.id);
  if (!f) return banner("err", "circle-x", esc(i.error), i.model ? `<button class="btn" data-redeploy="${id}">${icon("rotate-ccw")}重新部署</button>` : "");
  const text = `<b>${esc(f.title)}</b>${i.error ? `<span class="dt-err">${esc(i.error)}</span>` : ""}<span class="dt-hint">${esc(f.hint)}</span>`;
  return banner("err", "circle-x", text, failLinks(f.links) + failActions(i).map(failButton).join(""))
    + failLog(i.id, ui.logOpen.has(i.id) ? f.log : null);
}

export function detail(i) {
  const id = esc(i.id), top = head(i);
  switch (i.status) {
    case "running":
      return top + sloBar(i) + stats(i) + headroom(i) + genPanel(i) + `<div class="two" data-key="d-two">${apiPanel(i)}${events(i)}</div>`;
    case "starting": return top + progress(i) + events(i);
    case "waiting": return top + banner("warn", "clock", `等卡中${i.wait ? " · " + esc(waitInfo(i.wait)) : ""}`) + events(i);
    case "failed":
      return top + failBanner(i) + events(i);
    case "release_failed":
      return top + banner("err", "circle-x", "Colab 没有确认释放，实例可能仍在消耗 CU。工作台每分钟自动重试，也可以手动重试或去 Colab 网页确认。",
        failLinks(i.fail?.links) + `<button class="btn danger" data-stop="${id}">${icon("rotate-ccw")}重试释放</button>`) + events(i);
    case "orphan":
      return top + banner("warn", "info", `这台 ${esc(i.gpu)} 不是工作台部署的（可能来自旧客户端），仍在消耗 CU。`,
        `<button class="btn danger" data-orphan="${esc(i.email)}|${esc(i.endpoint)}">${icon("square")}释放</button>`);
    default: return top + events(i);
  }
}
