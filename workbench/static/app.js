// 工作台入口：渲染、轮询、事件
import * as api from "./api.js";
import { $, esc, icon, toast, ui } from "./util.js";
import { viewAccounts, viewInstances, viewModels } from "./views.js";
import { newLaunch, recommend, renderLaunch, renderLogin } from "./sheets.js";

const TITLES = { instances: "实例", accounts: "账号", models: "模型" };

function nav() {
  const d = ui.data;
  const cu = d.accounts.reduce((s, a) => s + (Number(a.cu) || 0), 0);
  const burn = d.accounts.reduce((s, a) => s + (Number(a.burn) || 0), 0);
  const run = d.instances.filter((i) => i.status === "running").length;
  const item = (v, ic, n) => `<button class="nav-item" data-nav="${v}" ${ui.view === v ? 'aria-current="page"' : ""}>${icon(ic)}${TITLES[v]}<span class="count">${n}</span></button>`;
  return `<div class="brand"><span class="brand-mark">${icon("zap")}</span>EasyColab</div>
    ${item("instances", "server", d.instances.length)}${item("accounts", "users", d.accounts.length)}${item("models", "boxes", d.models.length)}
    <div class="sidebar-foot"><div class="fleet">
      <div class="fleet-row"><span class="muted">余额</span><b>${Math.round(cu).toLocaleString()} CU</b></div>
      <div class="fleet-row"><span class="muted">消耗</span><b>${burn.toFixed(1)} CU/h</b></div>
      <div class="fleet-row"><span class="muted">运行</span><b>${run} 台</b></div></div>
      <button class="nav-item" id="theme">${icon("sun-moon")}深浅色</button></div>`;
}

function render() {
  if (!ui.data) return;
  // 弹层里的输入框在重绘时保留内容和焦点
  const code = $("#code");
  if (code && ui.login) ui.login.code = code.value;
  const focused = document.activeElement?.id;
  $("#nav").innerHTML = nav();
  $("#title").textContent = TITLES[ui.view];
  $("#offline").hidden = !ui.offline;
  const cd = ui.data.cooldown?.left;
  $("#cooldown").hidden = !cd;
  if (cd) $("#cooldown").textContent = `分配暂停 ${Math.ceil(cd / 60)} 分钟 · ${ui.data.cooldown.why}`;
  $("#view").innerHTML = { instances: viewInstances, accounts: viewAccounts, models: viewModels }[ui.view]();
  $("#sheet").innerHTML = renderLaunch() + renderLogin();
  $("#toast").innerHTML = ui.toast ? `<div class="toast ${ui.toastKind}" role="status">${esc(ui.toast)}</div>` : "";
  if (focused === "code") $("#code")?.focus();
}
document.addEventListener("rerender", render);

async function poll() {
  try { await api.refresh(); } catch { ui.offline = true; }
  if (!document.querySelector(".sheet")) render();
  setTimeout(poll, document.hidden ? 10000 : 3000);
}

async function copy(text) {
  await navigator.clipboard.writeText(text);
  toast("已复制");
}

async function act(fn, ok) {
  try { await fn(); if (ok) toast(ok, "ok"); await api.refresh(); } catch (e) { toast(e.message, "err"); }
  render();
}

const handlers = {
  nav: (t) => { ui.view = t.dataset.nav; },
  inst: (t) => { ui.sel = t.dataset.inst; },
  api: (t) => { ui.apiTab = t.dataset.api; },
  new: () => newLaunch(),
  pick: (t) => { ui.launch.model = t.dataset.pick; ui.launch.email = ui.launch.gpu = null; },
  cell: (t) => { [ui.launch.email, ui.launch.gpu] = t.dataset.cell.split("|"); },
  ctx: (t) => { ui.launch.ctx = Number(t.dataset.ctx); },
  next: () => { ui.launch.step++; if (ui.launch.step === 2 && !ui.launch.gpu) recommend(); },
  back: () => { ui.launch.step--; },
  close: (t, e) => { if (e.target === t || t.tagName === "BUTTON") ui.launch = null; else return false; },
  closeLogin: (t, e) => { if (e.target === t || t.tagName === "BUTTON") ui.login = null; else return false; },
  start: () => {
    const L = ui.launch, m = ui.data.models.find((x) => x.id === L.model);
    const params = L.ctx && m.kind === "语言" ? { load: { context_length: L.ctx } } : {};
    L.busy = true;
    // 失败时恢复按钮并保留弹层，错误由 act 以 toast 提示
    act(async () => {
      try { const r = await api.deploy(L.model, L.email, L.gpu, params); ui.launch = null; ui.view = "instances"; ui.sel = r.id; }
      catch (e) { L.busy = false; throw e; }
    }, "已开始部署");
  },
  stop: (t) => act(() => api.stopInstance(t.dataset.stop), "正在释放"),
  orphan: (t) => { const [e, ep] = t.dataset.orphan.split("|"); act(() => api.stopOrphan(e, ep), "已释放"); },
  bench: (t) => act(() => api.rebench(t.dataset.bench), "自测完成"),
  redeploy: (t) => {
    const i = ui.data.instances.find((x) => x.id === t.dataset.redeploy);
    newLaunch(i.model); ui.launch.step = 2; recommend();
  },
  copy: (t) => copy(t.dataset.copy),
  copyKey: async () => copy(await api.gatewayKey()),
  login: () => { ui.login = { step: 1 }; },
  loginOpen: () => {
    // 先同步开窗口，避免被浏览器当成弹窗拦截
    const w = window.open("about:blank", "_blank");
    ui.login.busy = true;
    act(async () => {
      const r = await api.loginStart();
      Object.assign(ui.login, { step: 2, pending_id: r.pending_id, auth_url: r.auth_url, busy: false, error: "" });
      if (w) w.location = r.auth_url;
    });
  },
  loginFinish: () => {
    const g = ui.login;
    g.code = $("#code").value.trim();
    if (!g.code) return;
    g.busy = true;
    act(async () => {
      try { const r = await api.loginFinish(g.pending_id, g.code); ui.login = null; toast(`已添加 ${r.email.slice(0, 3)}•••`, "ok"); }
      catch (e) { Object.assign(g, { busy: false, error: e.message, step: 1 }); }
    });
  },
};

const KEYS = Object.keys(handlers);
document.addEventListener("click", (e) => {
  if (e.target.closest("#theme")) { document.documentElement.classList.toggle("dark"); localStorage.theme = document.documentElement.classList.contains("dark") ? "dark" : "light"; return; }
  const t = e.target.closest(KEYS.map((k) => `[data-${k.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase())}]`).join(",") + ",#new");
  if (!t || t.disabled) return;
  const key = t.id === "new" ? "new" : KEYS.find((k) => t.dataset[k] !== undefined);
  if (handlers[key](t, e) === false) return;
  render();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && (ui.launch || ui.login)) { ui.launch = ui.login = null; render(); }
  if ((e.key === "Enter" || e.key === " ") && e.target.matches(".row,[data-pick]")) { e.preventDefault(); e.target.click(); }
  if (e.key === "Enter" && e.target.id === "code") handlers.loginFinish();
});

if (localStorage.theme === "light") document.documentElement.classList.remove("dark");
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js");
poll();
