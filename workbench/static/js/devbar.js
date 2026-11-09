// 原型工具条：右下角切换场景。切换后重置状态并刷新，只在原型里存在
import { SCENARIOS } from "./mock/fixtures.js";

// 场景由网址的 ?mock= 决定，切换就是改网址并刷新
const current = () => {
  const s = new URLSearchParams(location.search).get("mock");
  return SCENARIOS.some(([k]) => k === s) ? s : "daily";
};
const go = (mock) => {
  const q = new URLSearchParams(location.search);
  if (mock) q.set("mock", mock); else q.delete("mock");
  localStorage.removeItem("proto.dismissed");
  location.search = q.toString();
};

const bar = document.createElement("div");
bar.className = "devbar";
const draw = () => {
  const open = localStorage.getItem("proto.devbar") !== "closed";
  bar.dataset.open = open;
  bar.innerHTML = `<button class="devbar-toggle" data-devbar-toggle aria-expanded="${open}" aria-label="${open ? "收起" : "展开"}原型工具条">原型</button>
    <div class="devbar-body" role="group" aria-label="场景" ${open ? "" : "hidden"}>${SCENARIOS.map(([k, l]) =>
      `<button class="devbar-btn" data-scn="${k}" aria-pressed="${current() === k}">${l}</button>`).join("")}
      <button class="devbar-btn" data-scn="off" aria-pressed="false">真实数据</button></div>`;
};
bar.addEventListener("click", (e) => {
  const t = e.target.closest("button");
  if (!t) return;
  if (t.dataset.scn) {
    if (t.dataset.scn !== current()) go(t.dataset.scn === "off" ? "" : t.dataset.scn);
  } else {
    localStorage.setItem("proto.devbar", bar.dataset.open === "true" ? "closed" : "open");
    draw();
  }
});
draw();
document.body.append(bar);
