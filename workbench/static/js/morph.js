// 把新 HTML 就地合并进容器：保留焦点、输入、视频，并给增删换位加动效
import { countUp, nextFrame, ping, readMs, readVar, reduced, shownNum, stopCount } from "./motion.js";

const KEEP = new Set(["is-entering", "is-leaving", "is-ping", "page-enter"]);
const FORM = new Set(["INPUT", "TEXTAREA", "SELECT"]);

const keyOf = (n) => {
  if (n.nodeType !== 1) return null;
  const k = n.getAttribute("data-key");
  if (k != null) return k;
  const p = n.getAttribute("data-page");
  return p != null ? "page:" + p : null;
};
const isLeaving = (n) => n.nodeType === 1 && n.classList.contains("is-leaving");
const nextLive = (n) => { while (n && isLeaving(n)) n = n.nextSibling; return n; };
const sameKind = (a, b) => a.nodeType === b.nodeType && (a.nodeType !== 1 || a.tagName === b.tagName);
const keepClasses = (el) => [...el.classList].filter((c) => KEEP.has(c));

function syncAttrs(el, nc) {
  const focused = el === document.activeElement && FORM.has(el.tagName);
  for (const { name, value } of [...nc.attributes]) {
    if (name === "style" && el._anim) continue;
    if (name === "value" && focused) continue;
    if (name === "class") {
      const next = [...new Set([...value.split(/\s+/).filter(Boolean), ...keepClasses(el)])].join(" ");
      if (el.getAttribute("class") !== next) el.setAttribute("class", next);
    } else if (el.getAttribute(name) !== value) el.setAttribute(name, value);
  }
  for (const { name } of [...el.attributes]) {
    if (nc.hasAttribute(name) || (name === "style" && el._anim) || (name === "value" && focused)) continue;
    if (name === "class") el.setAttribute("class", keepClasses(el).join(" "));
    else el.removeAttribute(name);
  }
}

// 焦点在输入框上时不动它的值和选区
function syncControl(el, nc) {
  if (el === document.activeElement) return;
  if (el.tagName === "TEXTAREA") { if (el.value !== nc.value) el.value = nc.value; return; }
  if (el.tagName === "SELECT") {
    const sel = [...nc.options].findIndex((o) => o.hasAttribute("selected"));
    if (sel >= 0 && el.selectedIndex !== sel) el.selectedIndex = sel;
    return;
  }
  if (el.type === "checkbox" || el.type === "radio") { el.checked = nc.hasAttribute("checked"); return; }
  const v = nc.getAttribute("value") ?? "";
  if (el.value !== v) el.value = v;
}

function patch(el, nc, ctx) {
  if (el.nodeType !== 1) { if (el.nodeValue !== nc.nodeValue) el.nodeValue = nc.nodeValue; return; }
  if (el.hasAttribute("data-static") && nc.hasAttribute("data-static")) return;
  const wasOk = el.classList.contains("dot") && el.classList.contains("ok");
  const prevNum = el.dataset?.num;
  syncAttrs(el, nc);
  if (ctx.live && !wasOk && el.classList.contains("dot") && el.classList.contains("ok")) ctx.pings.push(el);
  if (prevNum != null && el.hasAttribute("data-num")) {
    const to = Number(nc.dataset.num);
    if (prevNum !== nc.dataset.num && ctx.live && isFinite(to)) {
      const from = shownNum(el);
      if (from == null) el.textContent = nc.textContent;
      else ctx.nums.push({ el, from, to, dec: Number(nc.dataset.dec) || 0 });
      return;
    }
    if (el._cur != null && prevNum === nc.dataset.num) return;
    stopCount(el);
  } else if (prevNum != null) stopCount(el);   // 数字节点被复用成普通文本时停掉补间
  if (el.tagName === "TEXTAREA") { syncControl(el, nc); return; }
  patchChildren(el, nc, ctx);
  if (FORM.has(el.tagName)) syncControl(el, nc);
}

function cancelLeave(el) {
  clearTimeout(el._lt);
  if (el._te) { el.removeEventListener("transitionend", el._te); el._te = null; }
  el._anim = false;
  el.inert = false;
  el.classList.remove("is-leaving");
  el.removeAttribute("aria-hidden");
  el.style.pointerEvents = "";
  el.style.height = "";
  el.style.overflow = "";
}

function leave(el) {
  const fixed = getComputedStyle(el).position === "fixed";
  el._anim = true;
  el.setAttribute("aria-hidden", "true");
  el.inert = true;   // 离场中不能再被聚焦或触发
  el.style.pointerEvents = "none";
  if (!fixed) { el.style.height = el.offsetHeight + "px"; el.style.overflow = "hidden"; void el.offsetHeight; }
  el.classList.add("is-leaving");
  const done = () => { clearTimeout(el._lt); el.remove(); };
  el._te = (e) => { if (e.target === el && e.propertyName === "height") done(); };
  el.addEventListener("transitionend", el._te);
  el._lt = setTimeout(done, readMs("--duration-slow") + 60);
}

function enter(el) {
  const fixed = getComputedStyle(el).position === "fixed";
  const h = fixed ? 0 : el.offsetHeight;
  el._anim = true;
  if (!fixed) { el.style.height = "0px"; el.style.overflow = "hidden"; }
  el.classList.add("is-entering");
  void el.offsetHeight;
  nextFrame(() => {
    el.classList.remove("is-entering");
    if (!fixed) el.style.height = h + "px";
    setTimeout(() => { el._anim = false; el.style.height = ""; el.style.overflow = ""; }, readMs("--duration-slow") + 60);
  });
}

function patchChildren(parent, np, ctx) {
  const isList = parent.hasAttribute("data-list");
  const keyed = new Map();
  for (const n of parent.childNodes) { const k = keyOf(n); if (k != null) keyed.set(k, n); }
  const kept = new Set();
  let cur = nextLive(parent.firstChild);
  for (const nc of [...np.childNodes]) {
    if (nc.nodeType === 8) continue;
    const k = keyOf(nc);
    let target = k != null ? keyed.get(k) : null;
    if (target && target.tagName !== nc.tagName) {
      if (target === cur) cur = nextLive(cur.nextSibling);
      keyed.delete(k);
      parent.insertBefore(nc, cur);   // 放到当前位置，不留在旧节点原处，否则伴随换位时顺序会错
      target.remove();
      kept.add(nc);
      continue;
    }
    if (k == null) {
      target = cur;
      while (target && (keyOf(target) != null || !sameKind(target, nc) || isLeaving(target) || kept.has(target))) target = target.nextSibling;
    }
    if (target) {
      if (k != null) keyed.delete(k);
      if (isLeaving(target)) cancelLeave(target);
      if (target === cur) cur = nextLive(cur.nextSibling);
      else parent.insertBefore(target, cur);
      kept.add(target);
      patch(target, nc, ctx);
    } else {
      parent.insertBefore(nc, cur);
      kept.add(nc);
      if (isList && ctx.live && k != null) ctx.entering.push(nc);
    }
  }
  for (const n of [...parent.childNodes]) {
    if (kept.has(n) || isLeaving(n)) continue;
    if (isList && ctx.live && keyOf(n) != null) leave(n); else n.remove();
  }
}

function snapshot(root) {
  const lists = [...root.querySelectorAll("[data-list][data-flip]")];
  if (root.matches("[data-list][data-flip]")) lists.push(root);
  return lists.map((list) => {
    const rects = new Map();
    for (const c of list.children) { const k = keyOf(c); if (k != null && !isLeaving(c)) rects.set(k, c.getBoundingClientRect()); }
    return { list, rects };
  });
}

function flip(snaps) {
  const ms = readMs("--duration-base"), easing = readVar("--ease-out") || "ease-out";
  for (const { list, rects } of snaps) {
    if (!list.isConnected) continue;
    for (const c of list.children) {
      const old = rects.get(keyOf(c));
      if (!old || isLeaving(c)) continue;
      const now = c.getBoundingClientRect();
      const dx = old.left - now.left, dy = old.top - now.top;
      if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) continue;
      c.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }], { duration: ms, easing });
    }
  }
}

export function morph(container, html, { animate = true } = {}) {
  const tpl = document.createElement("template");
  tpl.innerHTML = html;
  const ctx = { live: animate && !reduced(), entering: [], nums: [], pings: [] };
  const active = document.activeElement;
  const snaps = ctx.live ? snapshot(container) : [];
  patchChildren(container, tpl.content, ctx);
  if (active && active !== document.body && active.isConnected && document.activeElement !== active) active.focus({ preventScroll: true });
  if (!ctx.live) return;
  for (const el of ctx.entering) enter(el);
  for (const el of ctx.pings) ping(el);
  for (const n of ctx.nums) countUp(n.el, n.from, n.to, n.dec);
  flip(snaps);
}
