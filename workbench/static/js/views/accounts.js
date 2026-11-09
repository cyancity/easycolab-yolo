// 账号页：余额、消耗、实例、可用显卡和状态，失效的账号行里直接给「重新登录」
import { esc, icon, numTag, ui } from "../util.js";

function status(a) {
  if (a.state === "loading") return `<span class="chip">读取中</span>`;
  if (a.state === "error") return `<span class="chip err" title="${esc(a.note)}">凭据失效</span>`;
  if (a.state === "warn") return `<span class="chip warn">${esc(a.note || "需要留意")}</span>`;
  return a.note ? `<span class="chip">${esc(a.note)}</span>` : `<span class="chip ok">正常</span>`;
}

const dash = `<span class="subtle">—</span>`;

function row(a) {
  const gpus = ui.data.gpus.map((g) => `<span class="chip ${a.eligible?.includes(g.id) ? "ok" : "off"}">${esc(g.id)}</span>`).join("");
  const cu = a.cu == null ? dash : `<span class="${a.cu > 0 ? "" : "subtle"}">${numTag(a.cu, Number.isInteger(a.cu) ? 0 : 1, true)} CU</span>`;
  return `<tr class="pg-acct-row" data-key="acct-${esc(a.email)}">
    <td><div class="pg-who"><b class="ellipsis" title="${esc(a.label)}">${esc(a.label)}</b>${a.tier ? `<span class="chip">${esc(a.tier)}</span>` : ""}</div></td>
    <td class="num">${cu}</td>
    <td class="num">${a.burn ? `${numTag(a.burn, 1)} CU/h` : dash}</td>
    <td class="num">${a.count ? numTag(a.count, 0) : dash}</td>
    <td><div class="pg-chips">${gpus}</div></td>
    <td>${status(a)}</td>
    <td class="pg-act">${a.state === "error" ? `<button class="btn sm" data-relogin="${esc(a.email)}">重新登录</button>` : ""}</td></tr>`;
}

export function viewAccounts() {
  const list = ui.data.accounts;
  return `<div class="page" data-page="accounts">
    <div class="page-h"><span class="grow"></span><button class="btn" data-login>${icon("user-plus")}添加账号</button></div>
    <section class="panel pg-scroll"><table class="grid pg-acct">
      <thead><tr><th scope="col">账号</th><th scope="col">余额</th><th scope="col">消耗</th><th scope="col">实例</th><th scope="col">可用显卡</th><th scope="col">状态</th><th scope="col"><span class="sr-only">操作</span></th></tr></thead>
      <tbody>${list.map(row).join("") || `<tr data-key="empty"><td colspan="7" class="muted pg-empty">还没有账号</td></tr>`}</tbody></table></section></div>`;
}
