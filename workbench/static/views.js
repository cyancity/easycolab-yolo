// 实例、账号两个页面和模型卡片（模型页见 models.js）
import { dur, esc, icon, meter, num, pct, sparkline, ui } from "./util.js";

const STATUS = {
  running: ["ok", "运行中"], starting: ["busy", "部署中"], stopping: ["busy", "释放中"],
  failed: ["err", "失败"], stopped: ["", "已释放"], orphan: ["warn", "未纳管"],
  release_failed: ["err", "释放失败"],
};
const STEPS = ["分配实例", "部署服务", "下载权重", "载入显存", "自测"];
const slo = () => ui.data.slo;
const isLLM = (i) => i.kind === "语言";

/* ---------- 实例 ---------- */
function row(i) {
  const [dot, label] = STATUS[i.status] || ["", i.status];
  const vp = pct(i.vram_used, i.vram_total);
  let right = `<span class="r">${esc(label)}</span>`;
  if (i.status === "running" && isLLM(i) && i.bench?.tps) {
    right = `<span class="chip ${i.bench.tps >= slo().tpsMin ? "ok" : "warn"}">${num(i.bench.tps, 0)} tok/s</span>`;
  }
  const sub = i.status === "running" && vp != null
    ? `<span>${esc(i.gpu_req)}</span>${meter(vp, vp < slo().fillGoal ? "warn" : "ok")}<span>显存 ${vp}%</span>`
    : i.status === "starting" ? `<span>${esc(i.gpu_req)}</span><span class="ellipsis">${esc(i.step_text || STEPS[i.step - 1])}</span>`
    : i.status === "failed" ? `<span>${esc(i.gpu_req)}</span><span class="ellipsis">${esc(i.error)}</span>`
    : i.status === "release_failed" ? `<span>${esc(i.gpu_req || i.gpu)}</span><span class="ellipsis">可能仍在消耗 CU，自动重试中</span>`
    : `<span>${esc(i.gpu_req || i.gpu)}</span>`;
  return `<div class="row" role="option" tabindex="0" data-inst="${esc(i.id)}" aria-selected="${ui.sel === i.id}">
    <span class="dot ${dot}"></span><span class="t">${esc(i.name)}</span>${right}
    <div class="s">${sub}</div>
    <div class="s subtle">${esc(i.label)}${i.uptime_s ? " · " + dur(i.uptime_s) : ""}</div></div>`;
}

function stat(k, v, unit, ok, extra = "") {
  return `<div class="stat"><span class="k">${k}</span><span class="v ${ok === false ? "low" : ""}">${v}<small>${unit}</small></span>${extra}</div>`;
}

function headroom(i) {
  if (i.status !== "running") return "";
  const vp = pct(i.vram_used, i.vram_total);
  const reasons = [], tips = [];
  if (isLLM(i) && i.bench?.tps && i.bench.tps < slo().tpsMin) {
    reasons.push(`单流 ${num(i.bench.tps, 0)} tok/s 低于 ${num(slo().tpsMin, 0)}`);
    tips.push(["换 Qwen3.8-Flash-Next", "MoE 每个 token 只算 6B 参数，G4 实测 100 / 170（MTP）tok/s；运行时下一期上线"]);
  }
  if (isLLM(i) && i.ctx && i.ctx < slo().ctxMin) {
    reasons.push(`上下文 ${num(i.ctx, 0)}K 低于 ${num(slo().ctxMin, 0)}K`);
    tips.push(["拉长上下文", "Flash-Next 原生 256K，开 YaRN ×2 到 512K"]);
  }
  if (vp != null && vp < slo().fillGoal) {
    reasons.push(`显存只用了 ${vp}%`);
    tips.push(["吃满显存", "换更大的量化或更大的模型，或在同卡上并发加载第二个模型；也可以换显存更小的卡省 CU"]);
  }
  if (i.util != null && i.util < slo().fillGoal) {
    reasons.push(`GPU 利用率 ${Math.round(i.util)}%`);
    if (isLLM(i)) tips.push(["提高并发或换更大的模型", "单流请求喂不满算力时，调高并发槽位，或换参数更多的模型把算力用上"]);
  }
  if (!reasons.length) return "";
  return `<section class="panel headroom"><div class="panel-h">${icon("gauge")}未达标，还有空间
      <span class="right muted">${reasons.join(" · ")}</span></div>
    <div class="opts">${tips.map(([w, y]) => `<div class="opt"><div><div class="what">${w}</div><div class="why">${y}</div></div></div>`).join("")}</div></section>`;
}

function apiPanel(i) {
  const g = ui.data.gateway;
  const model = i.models[0] || "（载入后显示）";
  const snip = {
    openai: `curl ${g.base}/chat/completions \\\n  -H "Authorization: Bearer ${g.key_masked}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model":"${model}","messages":[{"role":"user","content":"你好"}]}'`,
    anthropic: `ANTHROPIC_BASE_URL=${g.base.replace(/\/v1$/, "")}\nANTHROPIC_AUTH_TOKEN=${g.key_masked}\nANTHROPIC_MODEL=${model}`,
  };
  const tab = (k, l) => `<button class="tab" role="tab" data-api="${k}" aria-selected="${ui.apiTab === k}">${l}</button>`;
  return `<section class="panel"><div class="panel-h">${icon("plug")}API
      <span class="right tabs" role="tablist">${tab("openai", "OpenAI")}${tab("anthropic", "Anthropic")}</span></div>
    <div class="panel-b"><div class="kv">
      <span class="k">地址</span><span class="mono">${esc(g.base)}</span><button class="btn sm ghost icon-btn" data-copy="${esc(g.base)}" aria-label="复制地址">${icon("copy")}</button>
      <span class="k">模型</span><span class="mono">${esc(model)}</span><button class="btn sm ghost icon-btn" data-copy="${esc(i.models[0] || "")}" aria-label="复制模型名">${icon("copy")}</button>
      <span class="k">密钥</span><span class="mono">${esc(g.key_masked)}</span><button class="btn sm ghost icon-btn" data-copy-key aria-label="复制密钥">${icon("copy")}</button>
      <span class="k">请求</span><span>${num(i.reqs ?? 0, 0)} 次${i.bench?.ttft ? ` · 首字 ${num(i.bench.ttft)} s` : ""}</span><span></span>
    </div><div class="code">${esc(snip[ui.apiTab])}</div></div></section>`;
}

function events(i) {
  const ic = { ok: "check", info: "dot", warn: "triangle-alert", err: "circle-x" };
  const evs = [...(i.events || [])].reverse();
  return `<section class="panel"><div class="panel-h">${icon("list")}过程</div><div class="panel-b timeline">
    ${evs.length ? evs.map(([t, k, m]) => `<div class="ev ${esc(k)}"><time>${esc(t)}</time>${icon(ic[k] || "dot")}<span class="msg">${esc(m)}</span><span></span></div>`).join("") : `<span class="muted">暂无记录</span>`}
    </div></section>`;
}

function detail(i) {
  const [dot, label] = STATUS[i.status] || ["", i.status];
  const act = [];
  if (typeof i.comfy_url === "string" && i.comfy_url.startsWith("https://")) act.push(`<a class="btn" href="${esc(i.comfy_url)}" target="_blank" rel="noopener">${icon("workflow")}打开 ComfyUI</a>`);
  if (i.status === "running" && isLLM(i)) act.push(`<button class="btn" data-bench="${esc(i.id)}">${icon("timer")}重新自测</button>`);
  if (["running", "starting"].includes(i.status)) act.push(`<button class="btn danger" data-stop="${esc(i.id)}">${icon("square")}${i.status === "starting" ? "取消并释放" : "停止"}</button>`);
  if (i.status === "release_failed") act.push(`<button class="btn danger" data-stop="${esc(i.id)}">${icon("rotate-ccw")}重试释放</button>`);
  if (i.status === "orphan") act.push(`<button class="btn danger" data-orphan="${esc(i.email)}|${esc(i.endpoint)}">${icon("square")}释放</button>`);
  if (["failed", "stopped"].includes(i.status) && i.model) act.push(`<button class="btn" data-redeploy="${esc(i.id)}">${icon("rotate-ccw")}重新部署</button>`);
  const head = `<div class="d-head"><div><h2>${esc(i.name)}</h2>
      <div class="d-sub"><span class="chip ${dot === "busy" ? "brand" : dot}">${esc(label)}</span><span>${esc(i.gpu)}</span><span>${esc(i.label)}</span>
      ${i.status === "running" ? `<span>${dur(i.uptime_s)}</span><span>${num(i.cuh)} CU/h · 已用 ${num(i.used, 2)} CU</span>` : ""}</div></div>
    <div class="d-actions">${act.join("")}</div></div>`;
  if (i.status === "orphan") {
    return head + `<div class="banner warn">${icon("info")}<span class="grow">这台 ${esc(i.gpu)} 不是工作台部署的（可能来自旧客户端），仍在消耗 CU。</span></div>`;
  }
  if (i.status === "release_failed") {
    return head + `<div class="banner err">${icon("circle-x")}<span class="grow">Colab 没有确认释放，实例可能仍在消耗 CU。工作台每分钟自动重试，也可以手动重试或去 Colab 网页确认。</span></div>` + events(i);
  }
  if (i.status === "failed") return head + `<div class="banner err">${icon("circle-x")}<span class="grow">${esc(i.error)}</span></div>` + events(i);
  if (i.status === "starting") {
    return head + `<div class="banner busy">${icon("loader")}<div class="grow">${esc(STEPS[i.step - 1])}：${esc(i.step_text)}
      <div class="steps">${STEPS.map((_, n) => `<i class="${n + 1 < i.step ? "done" : n + 1 === i.step ? "now" : ""}"></i>`).join("")}</div></div></div>` + events(i);
  }
  if (i.status !== "running") return head + events(i);
  const vp = pct(i.vram_used, i.vram_total), rp = pct(i.ram_used, i.ram_total);
  const tps = i.bench?.tps;
  const stats = `<section class="panel stats">
    ${isLLM(i) ? stat("单流速度", tps ? num(tps, 0) : "—", `tok/s · 线 ${num(slo().tpsMin, 0)}/${num(slo().tpsGoal, 0)}`, tps ? tps >= slo().tpsMin : undefined) : ""}
    ${isLLM(i) ? stat("上下文", i.ctx ? num(i.ctx, 0) + "K" : "—", `· 线 ${num(slo().ctxMin, 0)}K`, i.ctx ? i.ctx >= slo().ctxMin : undefined) : ""}
    ${i.util != null ? stat("GPU 利用率", Math.round(i.util), "% · 近 60 s", i.util >= slo().fillGoal, sparkline(i.util_hist, i.util < slo().fillGoal)) : ""}
    ${stat("显存", num(i.vram_used), `/ ${num(i.vram_total, 0)} GB`, vp == null ? undefined : vp >= slo().fillGoal, sparkline(i.hist?.vram, vp < slo().fillGoal))}
    ${stat("内存", num(i.ram_used), `/ ${num(i.ram_total, 0)} GB`, undefined, sparkline(i.hist?.ram))}
  </section>`;
  return head + stats + headroom(i) + `<div class="two">${apiPanel(i)}${events(i)}</div>`;
}

export function viewInstances() {
  const list = ui.data.instances;
  if (!list.length) {
    return `<div class="empty">${icon("server")}<p>还没有实例</p><button class="btn primary" data-new>${icon("plus")}新建部署</button></div>`;
  }
  if (!list.some((i) => i.id === ui.sel)) ui.sel = list[0].id;
  const i = list.find((x) => x.id === ui.sel);
  return `<div class="split"><div class="list" role="listbox" aria-label="实例">
      <div class="list-head"><span>全部账号</span><span>${list.length}</span></div>${list.map(row).join("")}</div>
    <div class="detail">${detail(i)}</div></div>`;
}

/* ---------- 账号 ---------- */
export function viewAccounts() {
  const rows = ui.data.accounts.map((a) => {
    const gpus = ui.data.gpus.map((g) => `<span class="chip ${a.eligible?.includes(g.id) ? "ok" : "off"}">${esc(g.id)}</span>`).join("");
    const st = a.state === "loading" ? `<span class="chip">读取中</span>`
      : a.state === "error" ? `<span class="chip err" title="${esc(a.note)}">凭据失效</span>`
      : a.state === "warn" ? `<span class="chip warn">${esc(a.note)}</span>`
      : a.note ? `<span class="chip">${esc(a.note)}</span>` : `<span class="chip ok">正常</span>`;
    return `<tr><td><div class="acct"><b>${esc(a.label)}</b><span class="muted">${esc(a.tier || "")}</span></div></td>
      <td class="num">${a.cu == null ? "—" : Number(a.cu).toLocaleString(undefined, { maximumFractionDigits: 1 }) + " CU"}</td>
      <td class="num">${a.burn ? num(a.burn) + " CU/h" : "—"}</td>
      <td class="num">${a.count ? num(a.count, 0) : "—"}</td>
      <td><div class="gpus">${a.eligible?.length ? gpus : "—"}</div></td><td>${st}</td></tr>`;
  }).join("");
  return `<div class="page"><div class="page-h"><span class="grow"></span><button class="btn" data-login>${icon("user-plus")}添加账号</button></div>
    <section class="panel"><table class="grid">
    <thead><tr><th>账号</th><th>余额</th><th>消耗</th><th>实例</th><th>可用显卡</th><th>状态</th></tr></thead>
    <tbody>${rows || `<tr><td colspan="6" class="muted">还没有账号</td></tr>`}</tbody></table></section></div>`;
}

/* ---------- 模型 ---------- */
export const modelChip = (m) => m.kind !== "语言" ? `<span class="chip">${esc(m.kind)}</span>`
  : `<span class="chip ${m.slo === "ok" ? "ok" : "warn"}">${num(m.ctx, 0)}K · ${esc(m.tps)} tok/s</span>`;

export function modelCard(m, pickable, picked) {
  const attrs = pickable && m.ready ? `role="radio" tabindex="0" data-pick="${esc(m.id)}" aria-checked="${picked}"` : pickable ? `aria-disabled="true"` : "";
  return `<div class="panel mcard ${m.ready ? "" : "disabled"}" ${attrs}>
    <h3>${esc(m.name)}</h3><div class="meta">${modelChip(m)}<span class="chip">${esc(m.spec)}</span></div>
    <span class="muted">${esc(m.size)}</span>
    ${m.ready ? "" : `<div class="foot"><span class="chip brand">${esc(m.pending)}</span></div>`}</div>`;
}
