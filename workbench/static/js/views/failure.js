// 部署失败的处理入口：外链、按钮、失败前日志。概览的待办和实例详情共用
import { esc, icon, safeUrl } from "../util.js";

// 外链只放行 https，新标签打开
export const failLinks = (links) => (links || []).map((l) => {
  const url = safeUrl(l.url);
  return url ? `<a class="btn sm" href="${esc(url)}" target="_blank" rel="noopener noreferrer">${icon("external-link")}${esc(l.label)}</a>` : "";
}).join("");

export const failButton = (a) =>
  `<button class="btn sm ${esc(a.tone)}" ${esc(a.attr)}="${esc(a.value)}">${esc(a.label)}</button>`;

// 日志来自远端，esc 后放进 pre；按钮展开时才有 lines
export const failLog = (id, lines) => lines?.length
  ? `<pre class="code fail-log" data-key="log-${esc(id)}" tabindex="0" aria-label="失败前日志">${esc(lines.join("\n"))}</pre>` : "";
