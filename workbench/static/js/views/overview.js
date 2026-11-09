// 占位：概览，由视图写手整体替换
import { STATUS, attention, esc, icon, ui } from "../util.js";
import { bestCell } from "../sheets/launch.js";

const action = (a) => `<button class="btn sm ${esc(a.tone)}" ${esc(a.attr)}="${esc(a.value)}">${esc(a.label)}</button>`;

export function viewOverview() {
  const d = ui.data, att = attention(d, ui.dismissed);
  const live = d.instances.filter((i) => ["running", "starting", "stopping"].includes(i.status));
  const best = bestCell("flash-next");
  const empty = `<div class="empty">${icon("server")}<p>还没有实例</p><div class="ph-row"><button class="btn" data-new>${icon("plus")}新建部署</button>
    ${best ? `<button class="btn primary" data-quick="flash-next">${icon("rocket")}一键部署推荐 · ${esc(best.label)}</button>` : ""}</div></div>`;
  return `<div class="page" data-page="overview">
    ${att.length ? `<section class="panel"><div class="panel-h">${icon("circle-alert")}需要你处理</div><div class="ph-list" data-list>
      ${att.map((a) => `<div class="ph-item" data-key="${esc(a.key)}"><span class="chip ${a.sev}">${a.sev === "err" ? "出错" : "注意"}</span>
        <div class="grow"><b>${esc(a.title)}</b><div class="muted">${esc(a.sub)}</div></div>${a.actions.map(action).join("")}</div>`).join("")}</div></section>` : ""}
    <section class="panel"><div class="panel-h">${icon("server")}实例</div>
      ${live.length ? `<div class="ph-list" data-list data-flip>${live.map((i) => `<div class="ph-item" data-key="inst-${esc(i.id)}" data-view-inst="${esc(i.id)}" role="button" tabindex="0">
        <span class="dot ${STATUS[i.status][0]}"></span><div class="grow"><b>${esc(i.name)}</b><div class="muted">${esc(i.label)} · ${esc(i.gpu_req)} · ${STATUS[i.status][1]}</div></div></div>`).join("")}</div>` : empty}</section>
    <section class="panel"><div class="panel-h">${icon("users")}账号</div><div class="ph-list">
      ${d.accounts.map((a) => `<div class="ph-item" data-key="acct-${esc(a.email)}" data-nav="accounts" role="button" tabindex="0"><span class="dot ${a.state === "error" ? "err" : a.state === "warn" ? "warn" : "ok"}"></span>
        <div class="grow"><b>${esc(a.label)}</b> <span class="muted">${esc(a.tier || "")}</span></div><span class="muted">${a.cu == null ? "—" : Math.round(a.cu) + " CU"}</span></div>`).join("")}</div></section></div>`;
}
