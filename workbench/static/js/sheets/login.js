// 添加账号弹层：两步（打开 Google 授权页 → 粘贴授权码）。输入框用 data-bind 写回 ui.login.code
import { esc, icon, ui } from "../util.js";

const ic = icon;
const NAMES = ["打开授权页", "粘贴授权码"];

function stepper(cur) {
  return `<ol class="stepper sh-steps" aria-label="添加步骤">${NAMES.map((n, k) => {
    const done = k + 1 < cur;
    return `<li class="sh-st ${done ? "done is-checked" : ""}" data-key="st-${k + 1}" ${k + 1 === cur ? 'aria-current="step"' : ""}>
      <span class="sh-dot"><span class="sh-n">${k + 1}</span>${ic("check", "check-in")}</span><span class="sh-st-n">${n}</span>
      ${k < NAMES.length - 1 ? `<i class="sh-line" aria-hidden="true"><b style="--f:${done ? 1 : 0}"></b></i>` : ""}</li>`;
  }).join("")}</ol>`;
}

export function renderLogin() {
  const g = ui.login;
  if (!g) return "";
  const acct = g.email ? ui.data.accounts.find((a) => a.email === g.email) : null;
  const title = g.email ? `重新登录${acct ? ` ${acct.label}` : ""}` : "添加账号";
  const step = g.step === 2 ? 2 : 1;
  const body = step === 1
    ? `<p>在打开的页面里选择${acct ? ` ${esc(acct.label)}` : "要添加的 Google 账号"}并同意授权，复制页面上的授权码。</p>
       <div class="sh-actions"><button class="btn primary" data-login-open ${g.busy ? "disabled" : ""}>${g.busy ? ic("loader", "spin") : icon("external-link")}打开 Google 授权页</button></div>`
    : `<label class="stack"><span>授权码</span><input id="code" class="input" data-bind="login.code" autocomplete="off" spellcheck="false" value="${esc(g.code || "")}"></label>
       <div class="row-actions"><a class="btn ghost" href="${esc(g.auth_url)}" target="_blank" rel="noopener">${icon("external-link")}重新打开授权页</a>
       <button class="btn primary" data-login-finish ${g.busy ? "disabled" : ""}>${g.busy ? ic("loader", "spin") : ""}${g.busy ? "验证中" : "添加账号"}</button></div>`;
  return `<div class="scrim" data-key="sheet-login" data-close-login><div class="sheet narrow" role="dialog" aria-modal="true" aria-label="${esc(title)}">
    <div class="sheet-h"><h2>${esc(title)}</h2><button class="btn ghost icon-btn push" data-close-login aria-label="关闭">${icon("x")}</button></div>
    <div class="sheet-b">${stepper(step)}
      <div class="sh-login" data-key="login-${step}" ${step === 2 ? 'data-step="2" style="--dir:1"' : ""}>${body}</div>
      ${g.error ? `<div class="banner err" data-key="login-err" role="alert">${icon("circle-x")}<span class="grow">${esc(g.error)}</span></div>` : ""}</div></div></div>`;
}
