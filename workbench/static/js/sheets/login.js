// 占位：添加账号弹层，由视图写手整体替换。输入框用 data-bind 写回 ui.login.code
import { esc, icon, ui } from "../util.js";

export function renderLogin() {
  const g = ui.login;
  if (!g) return "";
  const acct = g.email && ui.data.accounts.find((a) => a.email === g.email);
  const title = acct ? "重新登录" : "添加账号";
  const body = g.step === 1
    ? `<p>在打开的页面里选择${acct ? ` ${esc(acct.label)}` : "要添加的 Google 账号"}并同意授权，复制页面上的授权码。</p>
       <button class="btn primary" data-login-open ${g.busy ? "disabled" : ""}>${icon("external-link")}打开 Google 授权页</button>`
    : `<label class="stack"><span>授权码</span><input id="code" class="input" data-bind="login.code" autocomplete="off" spellcheck="false" value="${esc(g.code || "")}"></label>
       <div class="row-actions"><a class="btn ghost" href="${esc(g.auth_url)}" target="_blank" rel="noopener">${icon("external-link")}重新打开授权页</a>
       <button class="btn primary" data-login-finish ${g.busy ? "disabled" : ""}>${g.busy ? "验证中" : "添加账号"}</button></div>`;
  return `<div class="scrim" data-key="sheet-login" data-close-login><div class="sheet narrow" role="dialog" aria-modal="true" aria-label="${title}">
    <div class="sheet-h"><h2>${title}</h2><button class="btn ghost icon-btn push" data-close-login aria-label="关闭">${icon("x")}</button></div>
    <div class="sheet-b">${body}${g.error ? `<div class="banner err">${icon("circle-x")}<span class="grow">${esc(g.error)}</span></div>` : ""}</div></div></div>`;
}
