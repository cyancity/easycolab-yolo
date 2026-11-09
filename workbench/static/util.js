// 公共小工具与界面状态
export const $ = (s) => document.querySelector(s);
export { icon } from "./icons.js";

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
// 实例日志、错误信息来自远端，一律转义后再进 innerHTML
export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

export function dur(s) {
  if (!s || s < 60) return s ? `${Math.round(s)} s` : "—";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h} h ${m} m` : `${m} m`;
}

export const num = (v, d = 1) => (v == null ? "—" : Number(v).toFixed(d).replace(/\.0+$/, ""));
export const pct = (a, b) => (a != null && b ? Math.round((100 * a) / b) : null);

export function sparkline(pts, warn) {
  if (!pts || pts.length < 2) return "";
  const w = 120, h = 32, max = Math.max(...pts, 1) * 1.1, min = Math.min(...pts) * 0.8;
  const xy = pts.map((v, i) => [(i / (pts.length - 1)) * w, h - ((v - min) / (max - min || 1)) * h]);
  const line = xy.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  return `<svg class="spark ${warn ? "warn" : ""}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
    <path class="area" d="${line} L${w} ${h} L0 ${h} Z"/><path class="line" d="${line}" vector-effect="non-scaling-stroke"/></svg>`;
}

export const meter = (p, cls = "") => `<span class="meter ${cls}"><i style="width:${Math.min(p || 0, 100)}%"></i></span>`;

export const ui = {
  data: null, view: "instances", sel: null, apiTab: "openai",
  launch: null, login: null, toast: "", toastKind: "",
  docs: null, msel: null, tplOpen: {}, resetScroll: false,
};

export function toast(msg, kind = "") {
  ui.toast = msg;
  ui.toastKind = kind;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { ui.toast = ""; document.dispatchEvent(new Event("rerender")); }, 4000);
}
