// 模型页：左侧模型列表，右侧各卡适配、说明、优缺点、用法和模板
import { bar, esc, icon, modelChip, safeUrl, ui } from "../util.js";
import { hasCode } from "../callcode.js";


function source(s) {
  const text = String(s || "").trim();
  if (!text) return "";
  const href = safeUrl(text);
  return href
    ? `<a class="pg-src" href="${esc(href)}" target="_blank" rel="noopener" aria-label="来源" title="${esc(href)}">${icon("link")}</a>`
    : `<span class="pg-src-text">${esc(text)}</span>`;
}

const items = (list, id) => list.map((x, i) => `<li data-key="${id}-${i}"><span class="grow">${esc(x.text)}</span>${source(x.source)}</li>`).join("");

function section(id, title, ic, list, cls = "", ordered = false) {
  if (!list.length) return "";
  const tag = ordered ? "ol" : "ul";
  return `<section class="panel" data-key="${id}"><div class="panel-h ${cls}">${icon(ic)}${title}</div><${tag} class="pg-rows">${items(list, id)}</${tag}></section>`;
}

const chipOf = (k, v) => `<span class="chip">${esc(k)} ${esc(typeof v === "object" ? JSON.stringify(v) : v)}</span>`;

// 各卡适配：显卡、预估速度、显存占用、推荐；这个模型放不下的卡灰显
function fitTable(m) {
  const fit = m.fit || {}, llm = m.kind === "语言";
  const rows = ui.data.gpus.map((g) => {
    const f = fit[g.id];
    const gpu = `<th scope="row"><b>${esc(g.id)}</b> <span class="muted">${esc(g.vram)}GB</span></th>`;
    if (!f) return `<tr class="is-off" data-key="fit-${esc(g.id)}">${gpu}<td colspan="3" class="subtle">放不下</td></tr>`;
    const good = f.cls === "good";
    return `<tr data-key="fit-${esc(g.id)}">${gpu}
      <td><span class="pg-tps ${good ? "good" : "low"}">${esc(f.tps)}${llm ? ` <small>tok/s</small>` : ""}</span>${f.note ? `<small class="muted pg-note">${esc(f.note)}</small>` : ""}</td>
      <td><div class="pg-fill">${bar(f.fill, good ? "ok" : "warn")}<span class="num">${esc(Math.round(f.fill))}%</span></div></td>
      <td>${f.rec ? `<span class="chip brand">推荐</span>` : ""}</td></tr>`;
  }).join("");
  return `<section class="panel" data-key="fit"><div class="panel-h">${icon("gauge")}各卡适配</div>
    <div class="pg-scroll"><table class="grid pg-fit"><thead><tr><th scope="col">显卡</th><th scope="col">预估速度</th><th scope="col">显存占用</th><th scope="col"><span class="sr-only">推荐</span></th></tr></thead>
    <tbody>${rows}</tbody></table></div></section>`;
}

function template(m, t, idx) {
  const key = `${m.id}:${idx}`, open = !!ui.tplOpen[key];
  const extras = [["反向", t.negative], ["音效", t.sound], ["音乐", t.music]].filter(([, v]) => v);
  const expandable = extras.length > 0 || t.prompt.length > 180;
  const note = [t.source_note, safeUrl(t.source) ? "" : t.source].filter(Boolean).join(" · ");
  const params = Object.entries(t.params || {}).map(([k, v]) => chipOf(k, v)).join("");
  return `<article class="panel pg-tpl" data-key="tpl-${idx}">
    <div class="pg-tpl-h"><b>${esc(t.title)}</b>${t.scene ? `<span class="muted">${esc(t.scene)}</span>` : ""}</div>
    <div class="pg-prompt ${open ? "" : "clamp"}">${esc(t.prompt)}</div>
    ${open && extras.length ? `<div class="pg-tpl-extra" data-key="extra">${extras.map(([k, v]) => `<span class="muted">${k}</span><div class="pg-prompt">${esc(v)}</div>`).join("")}</div>` : ""}
    <div class="pg-tpl-f">
      <div class="pg-tpl-meta">${params}${note ? `<span class="pg-src-text">${esc(note)}</span>` : ""}${safeUrl(t.source) ? source(t.source) : ""}</div>
      <div class="pg-tpl-act">
        ${expandable ? `<button class="btn sm ghost" data-tpl-open="${esc(key)}" aria-expanded="${open}">${open ? "收起" : "展开"}</button>` : ""}
        <button class="btn sm" data-copy-prompt="${esc(key)}">${icon("copy")}复制提示词</button>
        ${hasCode(m) ? `<button class="btn sm" data-copy-code="${esc(key)}">${icon("copy")}复制调用代码</button>` : ""}
      </div></div></article>`;
}

function detail(m, doc) {
  const href = safeUrl(doc?.hf);
  const hfLabel = href && new URL(href).hostname === "github.com" ? "GitHub" : "Hugging Face";
  const pair = section("strengths", "优点", "check", doc?.strengths || [], "pg-pro") + section("weaknesses", "缺点", "triangle-alert", doc?.weaknesses || [], "pg-con");
  const tpls = doc?.templates || [];
  const empty = !(doc?.summary || ["facts", "strengths", "weaknesses", "usage", "templates"].some((k) => doc?.[k]?.length));
  const body = [
    doc?.summary ? `<p class="pg-lead" data-key="lead">${esc(doc.summary)}</p>` : "",
    fitTable(m),
    section("facts", "事实", "info", doc?.facts || []),
    pair ? `<div class="pg-pair" data-key="pair">${pair}</div>` : "",
    section("usage", "用法", "list", doc?.usage || [], "", true),
    tpls.length ? `<div class="pg-tpls" data-key="tpls">${tpls.map((t, i) => template(m, t, i)).join("")}</div>` : "",
  ].join("");
  return `<div class="d-head" data-key="head"><div><h2>${esc(m.name)}</h2>
      <div class="d-sub">${modelChip(m)}<span>${esc(m.spec)}</span><span>${esc(m.size)}</span></div></div>
    <div class="d-actions">
      ${href ? `<a class="btn" href="${esc(href)}" target="_blank" rel="noopener">${icon("external-link")}${hfLabel}</a>` : ""}
      <button class="btn primary" data-model-deploy="${esc(m.id)}" ${m.ready ? "" : "disabled"}>${icon("rocket")}部署</button></div></div>
    ${body}${empty ? `<p class="muted" data-key="pending">说明整理中</p>` : ""}`;
}

// 点击时按「模型id:序号」找回模板
export function tplRef(key) {
  const [id, idx] = key.split(":");
  const model = ui.data.models.find((x) => x.id === id);
  const tpl = ui.docs?.[id]?.templates?.[Number(idx)];
  return model && tpl ? { model, tpl } : null;
}

const row = (m) => `<div class="row m" role="option" tabindex="0" data-key="m-${esc(m.id)}" data-model="${esc(m.id)}" aria-selected="${ui.msel === m.id}">
  <span class="t">${esc(m.name)}</span>
  <div class="s">${modelChip(m)}<span class="subtle ellipsis">${esc(m.size)}</span>${m.ready ? "" : `<span class="chip brand">${esc(m.pending || "整理中")}</span>`}</div></div>`;

export function viewModels() {
  const list = ui.data.models;
  if (!list.some((m) => m.id === ui.msel)) ui.msel = list[0]?.id;
  const m = list.find((x) => x.id === ui.msel);
  return `<div class="page split" data-page="models"><div class="list" role="listbox" aria-label="模型">${list.map(row).join("")}</div>
    <div class="detail" data-key="detail-${esc(m?.id)}">${m ? detail(m, ui.docs?.[m.id]) : ""}</div></div>`;
}
