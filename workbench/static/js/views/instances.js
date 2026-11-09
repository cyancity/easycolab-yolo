// 实例页：左栏分组列表，右栏详情
import { STATUS, STEPS, bar, dur, esc, icon, isLLM, num, pct, ui } from "../util.js";
import { detail } from "./detail.js";
import { emptyState } from "./overview.js";

const GROUPS = [
  ["attention", "需要处理", ["release_failed", "orphan"]],
  ["running", "运行中", ["running", "stopping"]],
  ["starting", "部署中", ["starting"]],
  ["ended", "已结束", ["failed", "stopped"]],
];
const inGroup = (list, st) => list.filter((i) => st.includes(i.status));

function row(i) {
  const s = ui.data.slo, [dot, label] = STATUS[i.status] || ["", i.status];
  const vp = pct(i.vram_used, i.vram_total), tps = i.bench?.tps, gpu = esc(i.gpu_req || i.gpu);
  const right = i.status === "running" && isLLM(i) && tps
    ? `<span class="chip ${tps >= s.tpsMin ? "ok" : "warn"}">${num(tps, 0)} tok/s</span>`
    : `<span class="r ${dot === "err" || dot === "warn" ? dot : ""}">${esc(label)}</span>`;
  const why = String(i.error || "").split("。")[0];
  let sub = `<span>${gpu}</span>`;
  if (i.status === "running" && vp != null) sub += `${bar(vp, vp < s.fillGoal ? "warn" : "ok")}<span>${vp}%</span>`;
  else if (i.status === "starting") {
    const at = Math.min(Math.max(i.step || 1, 1), STEPS.length);
    sub = `<span class="ellipsis">${gpu} · ${esc(i.step_text || STEPS[at - 1])}</span>${bar(((at - 0.5) / STEPS.length) * 100)}`;
  } else if (i.status === "failed") sub += `<span class="ellipsis">${esc(why)}</span>`;
  else if (i.status === "release_failed") sub += `<span class="ellipsis">${esc(why || "可能还在扣 CU")}</span>`;
  else if (i.status === "orphan") sub += `<span class="ellipsis">不是工作台启动的</span>`;
  return `<div class="row" role="option" tabindex="0" data-key="row-${esc(i.id)}" data-inst="${esc(i.id)}" aria-selected="${ui.sel === i.id}">
    <span class="dot ${dot}"></span><span class="t">${esc(i.name || i.gpu)}</span>${right}
    <div class="s">${sub}</div>
    <div class="s subtle"><span class="ellipsis">${esc(i.label)}${i.uptime_s ? ` · ${dur(i.uptime_s)}` : ""}</span></div></div>`;
}

function group([k, name, st], list) {
  const rows = inGroup(list, st);
  if (!rows.length) return "";
  const shut = ui.collapsed.has(k);
  return `<div class="ov-group" data-key="group-${k}" role="group" aria-label="${name}">
    <button class="list-head ov-gh ${k === "attention" ? "attn" : ""}" data-group="${k}" aria-expanded="${!shut}">${icon("chevron-right")}<span>${name}</span><span class="n">${rows.length}</span></button>
    <div class="ov-rows" data-list data-flip>${(shut ? rows.filter((i) => i.id === ui.sel) : rows).map(row).join("")}</div></div>`;
}

// ui.sel 不在列表里时，选可见分组（没折叠的）的第一项。折叠的组仍露出当前选中项，免得右栏的实例在左栏找不到
function pick(list) {
  if (list.some((i) => i.id === ui.sel)) return;
  const seen = GROUPS.filter((g) => !ui.collapsed.has(g[0])).flatMap((g) => inGroup(list, g[2]));
  ui.sel = (seen[0] || list[0]).id;
}

export function viewInstances() {
  const list = ui.data.instances;
  if (!list.length) return `<div class="page" data-page="instances">${emptyState()}</div>`;
  pick(list);
  const cur = list.find((i) => i.id === ui.sel);
  return `<div class="page split" data-page="instances">
    <div class="list" role="listbox" aria-label="实例" data-list>${GROUPS.map((g) => group(g, list)).join("")}</div>
    <div class="detail" data-key="detail-${esc(cur.id)}">${detail(cur)}</div></div>`;
}
