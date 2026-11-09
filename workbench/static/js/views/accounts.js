// 占位：账号页，由视图写手整体替换
import { esc, icon, num, ui } from "../util.js";

export function viewAccounts() {
  const rows = ui.data.accounts.map((a) => {
    const gpus = ui.data.gpus.map((g) => `<span class="chip ${a.eligible?.includes(g.id) ? "ok" : "off"}">${esc(g.id)}</span>`).join("");
    const st = a.state === "loading" ? `<span class="chip">读取中</span>`
      : a.state === "error" ? `<span class="chip err">${esc(a.note || "凭据失效")}</span><button class="btn sm" data-relogin="${esc(a.email)}">重新登录</button>`
      : a.state === "warn" ? `<span class="chip warn">${esc(a.note)}</span>`
      : a.note ? `<span class="chip">${esc(a.note)}</span>` : `<span class="chip ok">正常</span>`;
    return `<tr data-key="acct-${esc(a.email)}"><td><b>${esc(a.label)}</b> <span class="muted">${esc(a.tier || "")}</span></td>
      <td class="num">${a.cu == null ? "—" : Number(a.cu).toLocaleString(undefined, { maximumFractionDigits: 1 }) + " CU"}</td>
      <td class="num">${a.burn ? num(a.burn) + " CU/h" : "—"}</td>
      <td class="num">${a.count ? num(a.count, 0) : "—"}</td>
      <td><div class="ph-row">${a.eligible?.length ? gpus : "—"}</div></td><td><div class="ph-row">${st}</div></td></tr>`;
  }).join("");
  return `<div class="page" data-page="accounts"><div class="ph-row"><span class="grow"></span><button class="btn" data-login>${icon("user-plus")}添加账号</button></div>
    <section class="panel"><table class="grid"><thead><tr><th>账号</th><th>余额</th><th>消耗</th><th>实例</th><th>可用显卡</th><th>状态</th></tr></thead>
    <tbody>${rows || `<tr><td colspan="6" class="muted">还没有账号</td></tr>`}</tbody></table></section></div>`;
}
