// 占位：实例页，由视图写手整体替换
import { STATUS, esc, icon, ui } from "../util.js";
import { detail } from "./detail.js";

const GROUPS = [
  ["attn", "需要处理", ["release_failed", "orphan"]], ["run", "运行中", ["running", "stopping"]],
  ["boot", "部署中", ["starting"]], ["ended", "已结束", ["failed", "stopped"]],
];

const row = (i) => `<div class="ph-item" data-key="inst-${esc(i.id)}" data-inst="${esc(i.id)}" role="option" tabindex="0" aria-selected="${ui.sel === i.id}">
  <span class="dot ${STATUS[i.status]?.[0] || ""}"></span><div class="grow"><b>${esc(i.name)}</b><div class="muted">${esc(i.label)} · ${esc(i.gpu_req || i.gpu)}</div></div></div>`;

export function viewInstances() {
  const list = ui.data.instances;
  if (!list.length) return `<div class="page" data-page="instances"><div class="empty">${icon("server")}<p>还没有实例</p><button class="btn primary" data-new>${icon("plus")}新建部署</button></div></div>`;
  if (!list.some((i) => i.id === ui.sel)) ui.sel = list[0].id;
  const groups = GROUPS.map(([k, label, st]) => {
    const rows = list.filter((i) => st.includes(i.status));
    if (!rows.length) return "";
    const shut = ui.collapsed.has(k);
    return `<div data-key="group-${k}"><button class="ph-group" data-group="${k}" aria-expanded="${!shut}">${icon(shut ? "chevron-right" : "chevron-down")}${label} ${rows.length}</button>
      <div class="ph-list" data-list ${shut ? "hidden" : ""}>${rows.map(row).join("")}</div></div>`;
  }).join("");
  return `<div class="page ph-split" data-page="instances"><div class="ph-side" role="listbox" aria-label="实例">${groups}</div>
    <div class="ph-main">${detail(list.find((x) => x.id === ui.sel))}</div></div>`;
}
