// 动效工具：时长读 CSS 变量，减弱动效时全部跳过
// 减弱动效或页面在后台（rAF 会暂停）时都不播放补间
export const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches || document.hidden;

export const readVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// 读 "260ms" 或 "0.26s" 形式的时长变量，返回毫秒
export function readMs(name) {
  const v = readVar(name);
  const n = parseFloat(v);
  if (!isFinite(n)) return 0;
  return /ms$/.test(v) ? n : /s$/.test(v) ? n * 1000 : n;
}

export function fmtNum(v, dec = 0, sep = false) {
  const n = Number(v);
  if (!isFinite(n)) return "—";
  return sep ? n.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec }) : n.toFixed(dec);
}

// 读出元素当前显示的数字（补间中以 _cur 为准）
export function shownNum(el) {
  if (el._cur != null) return el._cur;
  const n = parseFloat(el.textContent.replace(/,/g, "").replace(/[^\d.\-]/g, ""));
  return isFinite(n) ? n : null;
}

export function stopCount(el) {
  cancelAnimationFrame(el._raf);
  clearTimeout(el._tt);
  el._cur = null;
}

// rAF 被节流时用定时器兜底，保证最终一定落在目标值
export function nextFrame(fn) {
  let done = false;
  const run = () => { if (!done) { done = true; fn(); } };
  requestAnimationFrame(run);
  setTimeout(run, 50);
}

export function countUp(el, from, to, dec = 0) {
  stopCount(el);
  const sep = el.dataset.sep != null;
  const ms = readMs("--duration-number");
  const final = () => { stopCount(el); el.textContent = fmtNum(to, dec, sep); };
  if (reduced() || !ms || from == null || from === to) { final(); return; }
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min((now - t0) / ms, 1);
    if (k >= 1) { final(); return; }
    el._cur = from + (to - from) * (1 - Math.pow(1 - k, 3));
    el.textContent = fmtNum(el._cur, dec, sep);
    el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
  el._tt = setTimeout(final, ms + 120);
}

// 状态点变绿时外扩一次光圈
export function ping(el) {
  if (reduced()) return;
  el.classList.remove("is-ping");
  void el.offsetWidth;
  el.classList.add("is-ping");
  el.addEventListener("animationend", () => el.classList.remove("is-ping"), { once: true });
}
