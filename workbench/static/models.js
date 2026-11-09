// 模型页：左侧模型列表，右侧说明、优缺点、用法和模板
import { esc, icon, ui } from "./util.js";
import { modelChip } from "./views.js";
import { hasCode } from "./callcode.js";

// 来源和外链只放行 https，其余按纯文字显示
function safeUrl(u) {
  try { const x = new URL(String(u)); return x.protocol === "https:" ? x.href : ""; } catch { return ""; }
}

function source(s) {
  const text = String(s || "").trim();
  if (!text) return "";
  const href = safeUrl(text);
  return href
    ? `<a class="src" href="${esc(href)}" target="_blank" rel="noopener" aria-label="来源" title="${esc(href)}">${icon("link")}</a>`
    : `<span class="src-text">${esc(text)}</span>`;
}

const items = (list) => list.map((x) => `<li><span class="grow">${esc(x.text)}</span>${source(x.source)}</li>`).join("");

function section(title, ic, list, cls = "", ordered = false) {
  if (!list.length) return "";
  const tag = ordered ? "ol" : "ul";
  return `<section class="panel"><div class="panel-h ${cls}">${icon(ic)}${title}</div><${tag} class="rows">${items(list)}</${tag}></section>`;
}

const chipOf = (k, v) => `<span class="chip">${esc(k)} ${esc(typeof v === "object" ? JSON.stringify(v) : v)}</span>`;

function template(m, t, idx) {
  const key = `${m.id}:${idx}`, open = !!ui.tplOpen[key];
  const extras = [["反向", t.negative], ["音效", t.sound], ["音乐", t.music]].filter(([, v]) => v);
  const expandable = extras.length > 0 || t.prompt.length > 180;
  const note = [t.source_note, safeUrl(t.source) ? "" : t.source].filter(Boolean).join(" · ");
  return `<article class="panel tpl">
    <div class="tpl-h"><b>${esc(t.title)}</b>${t.scene ? `<span class="muted">${esc(t.scene)}</span>` : ""}</div>
    <div class="prompt ${open ? "" : "clamp"}">${esc(t.prompt)}</div>
    ${open && extras.length ? `<div class="tpl-extra">${extras.map(([k, v]) => `<span class="muted">${k}</span><div class="prompt">${esc(v)}</div>`).join("")}</div>` : ""}
    <div class="tpl-f">
      ${Object.entries(t.params || {}).map(([k, v]) => chipOf(k, v)).join("")}
      ${note ? `<span class="src-text">${esc(note)}</span>` : ""}${safeUrl(t.source) ? source(t.source) : ""}
      <span class="grow"></span>
      ${expandable ? `<button class="btn sm ghost" data-tpl-open="${esc(key)}" aria-expanded="${open}">${open ? "收起" : "展开"}</button>` : ""}
      <button class="btn sm" data-copy-prompt="${esc(key)}">${icon("copy")}复制提示词</button>
      ${hasCode(m) ? `<button class="btn sm" data-copy-code="${esc(key)}">${icon("copy")}复制调用代码</button>` : ""}
    </div></article>`;
}

function detail(m, doc) {
  const href = safeUrl(doc?.hf);
  const hfLabel = href && new URL(href).hostname === "github.com" ? "GitHub" : "Hugging Face";
  const pair = section("优点", "check", doc?.strengths || [], "pro") + section("缺点", "triangle-alert", doc?.weaknesses || [], "con");
  const tpls = doc?.templates || [];
  const body = [
    doc?.summary ? `<p class="lead">${esc(doc.summary)}</p>` : "",
    section("事实", "info", doc?.facts || []),
    pair ? `<div class="pair">${pair}</div>` : "",
    section("用法", "list", doc?.usage || [], "", true),
    tpls.length ? `<div class="tpls">${tpls.map((t, i) => template(m, t, i)).join("")}</div>` : "",
  ].join("");
  return `<div class="d-head"><div><h2>${esc(m.name)}</h2>
      <div class="d-sub">${modelChip(m)}<span>${esc(m.spec)}</span><span>${esc(m.size)}</span></div></div>
    <div class="d-actions">
      ${href ? `<a class="btn" href="${esc(href)}" target="_blank" rel="noopener">${icon("external-link")}${hfLabel}</a>` : ""}
      <button class="btn primary" data-model-deploy="${esc(m.id)}" ${m.ready ? "" : "disabled"}>${icon("rocket")}部署</button></div></div>
    ${body.replace(/\s/g, "") ? body : `<p class="muted">说明整理中</p>`}`;
}

// 点击时按「模型id:序号」找回模板
export function tplRef(key) {
  const [id, idx] = key.split(":");
  const model = ui.data.models.find((x) => x.id === id);
  const tpl = ui.docs?.[id]?.templates?.[Number(idx)];
  return model && tpl ? { model, tpl } : null;
}

const row = (m) => `<div class="row m" role="option" tabindex="0" data-model="${esc(m.id)}" aria-selected="${ui.msel === m.id}">
  <span class="t">${esc(m.name)}</span>${modelChip(m)}
  <div class="s subtle">${esc(m.size)}</div></div>`;

export function viewModels() {
  const list = ui.data.models;
  if (!list.some((m) => m.id === ui.msel)) ui.msel = list[0]?.id;
  const m = list.find((x) => x.id === ui.msel);
  return `<div class="split"><div class="list" role="listbox" aria-label="模型">${list.map(row).join("")}</div>
    <div class="detail">${m ? detail(m, ui.docs?.[m.id]) : ""}</div></div>`;
}
