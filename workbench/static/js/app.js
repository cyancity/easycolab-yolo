// 工作台入口：渲染、轮询、事件
import * as api from "./api.js";
import { $, DISMISS_KEY, attention, esc, hoursLeft, icon, numTag, safeUrl, snippet, toast, ui } from "./util.js";
import { morph } from "./morph.js";
import { readMs } from "./motion.js";
import { viewOverview } from "./views/overview.js";
import { viewInstances } from "./views/instances.js";
import { viewAccounts } from "./views/accounts.js";
import { viewModels } from "./views/models.js";
import { genHandlers } from "./views/generate.js";
import { newLaunch, packsOf, recommend, redeploy, renderLaunch } from "./sheets/launch.js";
import { renderLogin } from "./sheets/login.js";

const TITLES = { overview: "概览", instances: "实例", accounts: "账号", models: "模型" };
const VIEWS = { overview: viewOverview, instances: viewInstances, accounts: viewAccounts, models: viewModels };
const FOCUSABLE = 'button:not([disabled]),[href],input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
const inst = (id) => ui.data?.instances.find((x) => x.id === id);

/* ---------- 外壳 ---------- */
function nav() {
  const d = ui.data;
  const att = d ? attention(d, ui.dismissed) : [];
  const badge = att.length ? `<span class="badge ${att.some((x) => x.sev === "err") ? "err" : "warn"}">${att.length}</span>` : "";
  const item = (v, ic, extra = "") =>
    `<button class="nav-item" data-nav="${v}" ${ui.view === v ? 'aria-current="page"' : ""}>${icon(ic)}<span>${TITLES[v]}</span>${extra}</button>`;
  let fleet = "";
  if (d) {
    const cu = d.accounts.reduce((s, a) => s + (Number(a.cu) || 0), 0);
    const burn = d.accounts.reduce((s, a) => s + (Number(a.burn) || 0), 0);
    const run = d.instances.filter((i) => i.status === "running").length;
    const h = hoursLeft(cu, burn);
    fleet = `<div class="fleet" data-key="fleet">
      <div class="fleet-row"><span class="muted">余额</span><b>${numTag(cu, 0, true)} CU</b></div>
      <div class="fleet-row"><span class="muted">消耗</span><b>${numTag(burn, 1)} CU/h</b></div>
      <div class="fleet-row"><span class="muted">可撑</span><b>${h == null ? "—" : `约 ${numTag(h, 0)} 小时`}</b></div>
      <div class="fleet-row"><span class="muted">运行</span><b>${numTag(run, 0)} 台</b></div></div>`;
  }
  return `<div class="brand"><span class="brand-mark">${icon("zap")}</span>EasyColab</div>
    ${item("overview", "layout-dashboard")}${item("instances", "server", badge)}${item("accounts", "users")}${item("models", "boxes")}
    <div class="sidebar-foot">${fleet}<button class="nav-item" id="theme">${icon("sun-moon")}<span>深浅色</span></button></div>`;
}

function topbar() {
  const d = ui.data, cd = d?.cooldown?.left, g = d?.gateway;
  return `<h1>${TITLES[ui.view]}</h1>
    ${ui.offline ? `<span class="chip err" data-key="chip-offline">工作台服务未响应</span>` : ""}
    ${cd ? `<span class="chip warn" data-key="chip-cooldown">分配暂停 ${Math.ceil(cd / 60)} 分钟 · ${esc(d.cooldown.why)}</span>` : ""}
    <span class="spacer"></span>
    ${g ? `<button class="pill" data-copy="${esc(g.base)}" aria-label="复制接入地址"><span class="mono">${esc(g.base.replace(/^https?:\/\//, ""))}</span>${icon("copy")}</button>` : ""}
    <button class="btn primary" id="new" ${d ? "" : "disabled"}>${icon("plus")}新建部署</button>`;
}

function body() {
  if (ui.data) return VIEWS[ui.view]();
  if (ui.loading) {
    return `<div class="page" data-page="loading"><div class="skeleton-card"><div class="skeleton w-40"></div><div class="skeleton w-72"></div><div class="skeleton w-56"></div></div></div>`;
  }
  return `<div class="page" data-page="offline"><div class="empty">${icon("circle-alert")}<p>连不上工作台服务</p>
    <button class="btn" data-retry>${icon("refresh-cw")}重试</button></div></div>`;
}

let shownView = null, prevOpen = false, trigger = null, lastClick = null;

function render() {
  morph($("#nav"), nav());
  morph($("#topbar"), topbar());
  morph($("#view"), body());
  if (ui.data) morph($("#sheet"), renderLaunch() + renderLogin());
  morph($("#toast"), ui.toast
    ? `<div class="toast ${esc(ui.toastKind)}" data-key="toast-${ui.toastId}" role="status"><span>${esc(ui.toast)}</span><i class="toast-timer"></i></div>` : "");
  if (shownView !== ui.view) {
    shownView = ui.view;
    const p = $("#view .page");
    if (p) { p.classList.add("page-enter"); setTimeout(() => p.classList.remove("page-enter"), readMs("--duration-base") + 60); }
  }
  // 切页、换选中项时详情回到顶部；左侧列表的滚动位置不动
  if (ui.resetScroll) { ui.resetScroll = false; $("#view").scrollTop = 0; const d = $("#view .detail"); if (d) d.scrollTop = 0; }
  sheetFocus();
}
document.addEventListener("rerender", render);

// 弹层开合时的焦点：打开移进第一个可聚焦元素，关闭还给触发元素
function sheetFocus() {
  const open = !!(ui.launch || ui.login);
  const root = $("#sheet > [data-key]:not(.is-leaving)");
  if (open) {
    if (!prevOpen) trigger = lastClick || document.activeElement;
    if (root && !root.contains(document.activeElement)) root.querySelector(FOCUSABLE)?.focus({ preventScroll: true });
  } else if (prevOpen) {
    if (trigger?.isConnected) trigger.focus({ preventScroll: true });
    trigger = null;
  }
  prevOpen = open;
}

/* ---------- 数据 ---------- */
// 模型资料失败不影响主流程；模型页或还没有资料时，每次轮询重拉一次
let docsBusy = false;
function loadDocs() {
  if (docsBusy) return;
  docsBusy = true;
  api.modelDocs().then((r) => { ui.docs = r.docs || {}; render(); }).catch(() => {}).finally(() => { docsBusy = false; });
}

let pollT;
async function poll() {
  clearTimeout(pollT);
  if (ui.view === "models" || !ui.docs) loadDocs();
  try { await api.refresh(); } catch { ui.offline = true; }
  ui.loading = false;
  // 先排下一次，渲染出错也不会让轮询停掉
  pollT = setTimeout(poll, document.hidden ? 10000 : 3000);
  render();
}

async function copy(text) {
  try { await navigator.clipboard.writeText(text); } catch {
    const ta = Object.assign(document.createElement("textarea"), { value: text });
    document.body.append(ta); ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    if (!ok) throw new Error("复制失败，浏览器没有给剪贴板权限");
  }
  toast("已复制");
}

// 先拿到新状态再提示，提示触发的重绘才不会用旧数据把选中项重置掉
async function act(fn, ok) {
  try { await fn(); } catch (e) { toast(e.message, "err"); render(); return; }
  try { await api.refresh(); } catch { ui.offline = true; }
  if (ok) toast(ok, "ok");
  render();
}

let confirmT;
const closeSheets = () => { ui.launch = ui.login = null; };
const direct = (step, dir = 1) => { ui.launch.step = step; ui.launch.dir = dir; };

const handlers = {
  nav: (t) => { ui.view = t.dataset.nav; ui.resetScroll = true; },
  inst: (t) => { ui.sel = t.dataset.inst; ui.resetScroll = true; },
  viewInst: (t) => { ui.view = "instances"; ui.sel = t.dataset.viewInst; ui.resetScroll = true; },
  api: (t) => { ui.apiTab = t.dataset.api; },
  new: () => newLaunch(),
  quick: (t) => { newLaunch(t.dataset.quick); recommend(); direct(ui.launch.gpu ? 3 : 2); },
  modelDeploy: (t) => { newLaunch(t.dataset.modelDeploy); recommend(); direct(2); },
  model: (t) => { ui.msel = t.dataset.model; ui.resetScroll = true; },
  pick: (t) => { ui.launch.model = t.dataset.pick; ui.launch.packs = undefined; ui.launch.email = ui.launch.gpu = ui.launch.rec = ui.launch.avoid = null; },
  cell: (t) => { [ui.launch.email, ui.launch.gpu] = t.dataset.cell.split("|"); },
  ctx: (t) => { ui.launch.ctx = Number(t.dataset.ctx); },
  packToggle: (t) => {
    const L = ui.launch, id = t.dataset.packToggle, cur = packsOf(ui.data.models.find((x) => x.id === L.model));
    L.packs = cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id];
  },
  preferGpu: (t) => {
    const gpu = t.dataset.preferGpu, prev = ui.data.prefs;
    if (prev?.gpu === gpu) return;
    ui.data.prefs = { gpu };
    recommend();
    act(async () => { try { await api.setPrefs(gpu); } catch (e) { ui.data.prefs = prev; recommend(); throw e; } });
  },
  next: () => { direct(ui.launch.step + 1); if (ui.launch.step === 2 && !ui.launch.gpu) recommend(); },
  back: () => direct(ui.launch.step - 1, -1),
  close: (t, e) => { if (e.target === t || t.tagName === "BUTTON") closeSheets(); else return false; },
  closeLogin: (t, e) => { if (e.target === t || t.tagName === "BUTTON") ui.login = null; else return false; },
  start: () => {
    const L = ui.launch, m = ui.data.models.find((x) => x.id === L.model);
    const params = m.packs ? { comfy_packs: packsOf(m) }
      : L.ctx && m.kind === "语言" ? { load: { context_length: L.ctx } } : {};
    L.busy = true;
    // 失败时恢复按钮并保留弹层，错误由 act 以 toast 提示
    act(async () => {
      let r;
      try { r = await api.deploy(L.model, L.email, L.gpu, params); } catch (e) { L.busy = false; throw e; }
      // 已提交就关弹层，之后的刷新失败也不能让人以为没部署而再点一次
      ui.launch = null; ui.view = "instances"; ui.sel = r.id; ui.resetScroll = true;
    }, "已开始部署");
  },
  stop: (t) => {
    const id = t.dataset.stop;
    // 运行中的实例先原地确认，3 秒内不再点就取消
    if (inst(id)?.status === "running" && ui.confirm !== id) {
      ui.confirm = id;
      clearTimeout(confirmT);
      confirmT = setTimeout(() => { ui.confirm = null; render(); }, 3000);
      return;
    }
    ui.confirm = null;
    clearTimeout(confirmT);
    act(() => api.stopInstance(id), inst(id)?.status === "waiting" ? "已取消排队" : "正在释放");
  },
  wait: (t) => act(() => api.waitInstance(t.dataset.wait), "已排队，每 15 分钟申请一次，最多 8 次"),
  orphan: (t) => { const [e, ep] = t.dataset.orphan.split("|"); act(() => api.stopOrphan(e, ep), "已释放"); },
  bench: (t) => act(() => api.rebench(t.dataset.bench), "自测完成"),
  redeploy: (t) => redeploy(inst(t.dataset.redeploy)),
  failLog: (t) => { const id = t.dataset.failLog; ui.logOpen.has(id) ? ui.logOpen.delete(id) : ui.logOpen.add(id); },
  releaseOrphans: (t) => {
    const list = ui.data.instances.filter((x) => x.status === "orphan" && x.email === t.dataset.releaseOrphans);
    act(async () => { for (const o of list) await api.stopOrphan(o.email, o.endpoint); }, "已释放");
  },
  dismiss: (t) => {
    ui.dismissed.add(t.dataset.dismiss);
    localStorage.setItem(DISMISS_KEY, JSON.stringify([...ui.dismissed]));
  },
  group: (t) => { const g = t.dataset.group; ui.collapsed.has(g) ? ui.collapsed.delete(g) : ui.collapsed.add(g); },
  copy: (t) => copy(t.dataset.copy),
  copyKey: async () => { try { await copy(await api.gatewayKey()); } catch (e) { toast(e.message, "err"); } },
  copySnippet: async (t) => {
    try { await copy(snippet(inst(t.dataset.copySnippet), await api.gatewayKey(), ui.apiTab)); } catch (e) { toast(e.message, "err"); }
  },
  openCanvas: (t) => {
    // 先同步开窗口，避免被当成弹窗拦截；地址带访问令牌，不写进页面
    const w = window.open("about:blank", "_blank");
    act(async () => {
      try {
        const url = safeUrl(await api.canvasUrl(t.dataset.openCanvas));
        if (!url) throw new Error("画布地址无效");
        if (w) w.location = url;
      } catch (e) { w?.close(); throw e; }
    });
  },
  importMagpie: () => act(() => api.importMagpie(), "已打开 Magpie，在它的窗口里确认"),
  retry: () => { poll(); },
  login: () => { ui.login = { step: 1 }; },
  relogin: (t) => { ui.login = { step: 1, email: t.dataset.relogin }; },
  loginOpen: () => {
    // 先同步开窗口，避免被浏览器当成弹窗拦截
    const w = window.open("about:blank", "_blank"), g = ui.login;
    g.busy = true;
    act(async () => {
      try {
        const r = await api.loginStart(g.email);
        const url = safeUrl(r.auth_url);
        if (!url) throw new Error("授权地址无效");
        if (ui.login !== g) { w?.close(); return; }   // 等待期间弹层已关
        Object.assign(g, { step: 2, pending_id: r.pending_id, auth_url: url, busy: false, error: "" });
        if (w) w.location = url;
      } catch (e) { g.busy = false; w?.close(); throw e; }
    });
  },
  loginFinish: () => {
    const g = ui.login;
    g.code = (g.code ?? $("#code")?.value ?? "").trim();
    if (!g.code) return;
    g.busy = true;
    act(async () => {
      try { const r = await api.loginFinish(g.pending_id, g.code); ui.login = null; toast(`已添加 ${r.email.slice(0, 3)}•••`, "ok"); }
      catch (e) { Object.assign(g, { busy: false, error: e.message, step: 1 }); }
    });
  },
  ...genHandlers,
};

/* ---------- 事件 ---------- */
const KEYS = Object.keys(handlers);
const SEL = KEYS.map((k) => `[data-${k.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase())}]`).join(",") + ",#new";

document.addEventListener("click", (e) => {
  if (e.target.closest("#theme")) {
    document.documentElement.classList.toggle("dark");
    localStorage.theme = document.documentElement.classList.contains("dark") ? "dark" : "light";
    return;
  }
  const t = e.target.closest(SEL);
  if (!t || t.disabled) return;
  const key = t.id === "new" ? "new" : KEYS.find((k) => t.dataset[k] !== undefined);
  lastClick = t;
  if (handlers[key](t, e) === false) return;
  render();
});

// data-bind="a.b.c" 写回 ui 对应路径，输入过程不重绘
function bind(e) {
  const t = e.target.closest?.("[data-bind]");
  if (!t) return null;
  const path = t.dataset.bind.split(".");
  let o = ui;
  for (const k of path.slice(0, -1)) o = o[k] ??= {};
  o[path.at(-1)] = t.type === "checkbox" ? t.checked : t.value;
  return t;
}
document.addEventListener("input", bind);
document.addEventListener("change", (e) => {
  const t = bind(e);
  // 选择类控件提交后重绘一次，让依赖它的显示跟上
  if (t && (t.tagName === "SELECT" || ["checkbox", "radio", "range"].includes(t.type))) render();
});

document.addEventListener("keydown", (e) => {
  const open = ui.launch || ui.login;
  if (e.key === "Escape") {
    if (open) { closeSheets(); render(); } else if (ui.confirm) { ui.confirm = null; render(); }
    return;
  }
  if ((e.key === "Enter" || e.key === " ") && e.target.matches('[role="option"],[role="radio"]') && !e.target.matches("button,input,a,select,textarea")) {
    e.preventDefault();
    e.target.click();
    return;
  }
  if (e.key === "Enter" && e.target.matches('#code,[data-bind="login.code"]') && ui.login?.step === 2) { handlers.loginFinish(); render(); return; }
  if (e.key === "Tab" && open) {
    const root = $("#sheet > [data-key]:not(.is-leaving)");
    const list = root ? [...root.querySelectorAll(FOCUSABLE)] : [];
    if (!list.length) return;
    const first = list[0], last = list.at(-1);
    if (e.shiftKey && (e.target === first || !list.includes(e.target))) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && (e.target === last || !list.includes(e.target))) { e.preventDefault(); first.focus(); }
  }
});

/* ---------- 启动 ---------- */
// 只有网址带 ?mock=daily|trouble|fresh|offline 才加载假后端和原型工具条，否则直连真实后端
async function boot() {
  if (localStorage.theme === "light") document.documentElement.classList.remove("dark");
  const scn = new URLSearchParams(location.search).get("mock");
  if (["daily", "trouble", "fresh", "offline"].includes(scn)) {
    await import("./mock/server.js");
    await import("./devbar.js");
  } else if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/sw.js");
  }
  render();
  poll();
}
boot();
