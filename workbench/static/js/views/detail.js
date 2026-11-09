// 占位：实例详情，由视图写手整体替换
import { STATUS, STEPS, dur, esc, icon, num, sloChecks, ui } from "../util.js";
import { genPanel } from "./generate.js";

export function detail(i) {
  const [dot, label] = STATUS[i.status] || ["", i.status];
  const act = [];
  if (i.status === "running") {
    const arm = ui.confirm === i.id;
    act.push(`<button class="btn danger ${arm ? "armed" : ""}" data-stop="${esc(i.id)}">${icon("square")}${arm ? "再点一次确认停止" : "停止"}</button>`);
  }
  if (i.status === "starting") act.push(`<button class="btn danger" data-stop="${esc(i.id)}">${icon("square")}取消并释放</button>`);
  if (i.status === "release_failed") act.push(`<button class="btn danger" data-stop="${esc(i.id)}">${icon("rotate-ccw")}重试释放</button>`);
  if (i.status === "orphan") act.push(`<button class="btn danger" data-orphan="${esc(i.email)}|${esc(i.endpoint)}">${icon("square")}释放</button>`);
  if (["failed", "stopped"].includes(i.status) && i.model) act.push(`<button class="btn" data-redeploy="${esc(i.id)}">${icon("rotate-ccw")}重新部署</button>`);
  const checks = i.status === "running" ? `<section class="panel"><div class="panel-b ph-row">${sloChecks(i).map((c) =>
    `<span class="chip ${c.ok === false ? "warn" : c.ok ? "ok" : ""}">${esc(c.label)} ${esc(c.value)} / ${esc(c.line)}</span>`).join("")}</div></section>` : "";
  const steps = i.status === "starting" ? `<div class="banner busy">${icon("loader")}<span class="grow">${esc(STEPS[i.step - 1])}：${esc(i.step_text)}</span></div>` : "";
  const err = i.status === "failed" ? `<div class="banner err">${icon("circle-x")}<span class="grow">${esc(i.error)}</span></div>` : "";
  const evs = [...(i.events || [])].reverse().map(([t, k, m]) => `<div class="ph-ev ${esc(k)}"><time>${esc(t)}</time><span>${esc(m)}</span></div>`).join("");
  return `<div class="ph-row"><div class="grow"><h2 class="ph-h2">${esc(i.name)}</h2>
      <div class="muted"><span class="chip ${dot === "busy" ? "brand" : dot}">${esc(label)}</span> ${esc(i.gpu)} · ${esc(i.label)}${i.status === "running" ? ` · ${dur(i.uptime_s)} · ${num(i.cuh)} CU/h` : ""}</div></div>${act.join("")}</div>
    ${checks}${steps}${err}${i.status === "running" && ["图像", "视频"].includes(i.kind) ? genPanel(i) : ""}
    <section class="panel"><div class="panel-h">${icon("list")}过程</div><div class="panel-b">${evs || `<span class="muted">暂无记录</span>`}</div></section>`;
}
