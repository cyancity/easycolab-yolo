"""模型说明库：读 model_docs/*.json，校验后给界面用。文件可由调研流程随时放入或替换，坏数据只跳过、不拖垮服务。

文件格式（每个模型一个 JSON，id 取自文件内的 id 字段）：
{id, name, hf, summary, facts[{text,source}], strengths[...], weaknesses[...], usage[...],
 templates[{title, scene, prompt, negative?, sound?, music?, params{}, source, source_note}]}
同一个 id 出现在多个文件里时，标量字段取先读到的非空值（<id>.json 优先），列表合并去重。
"""
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple

from .paths import MODEL_DOCS_DIR

log = logging.getLogger("workbench.model_docs")
_lock = threading.Lock()
_cache: Dict[str, object] = {"sig": None, "docs": {}, "issues": []}

_LISTS = ("facts", "strengths", "weaknesses", "usage")
_TEXT_FIELDS = ("name", "hf", "summary")
_TPL_TEXT = ("title", "scene", "prompt", "negative", "sound", "music", "source", "source_note")


def _str(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def _item(raw, where: str, issues: List[str]) -> Optional[dict]:
    if isinstance(raw, str):  # 容忍只写了文字的条目
        raw = {"text": raw}
    if not isinstance(raw, dict) or not _str(raw.get("text")):
        issues.append(f"{where}：条目缺少 text，已跳过")
        return None
    return {"text": _str(raw["text"]), "source": _str(raw.get("source"))}


def _template(raw, where: str, issues: List[str]) -> Optional[dict]:
    if not isinstance(raw, dict) or not _str(raw.get("title")) or not _str(raw.get("prompt")):
        issues.append(f"{where}：模板缺少 title 或 prompt，已跳过")
        return None
    params = raw.get("params")
    if params is not None and not isinstance(params, dict):
        issues.append(f"{where}：params 不是对象，已忽略")
        params = None
    return {**{k: _str(raw.get(k)) for k in _TPL_TEXT}, "params": params or {}}


def _normalize(raw, fname: str, issues: List[str]) -> Optional[dict]:
    if not isinstance(raw, dict) or not _str(raw.get("id")):
        issues.append(f"{fname}：顶层不是对象或缺少 id，已跳过整个文件")
        return None
    doc = {"id": _str(raw["id"]), **{k: _str(raw.get(k)) for k in _TEXT_FIELDS}}
    for key in _LISTS:
        items = raw.get(key, [])
        if not isinstance(items, list):
            issues.append(f"{fname}：{key} 不是数组，已忽略")
            items = []
        doc[key] = [x for i, r in enumerate(items) if (x := _item(r, f"{fname} {key}[{i}]", issues))]
    tpls = raw.get("templates", [])
    if not isinstance(tpls, list):
        issues.append(f"{fname}：templates 不是数组，已忽略")
        tpls = []
    doc["templates"] = [x for i, r in enumerate(tpls) if (x := _template(r, f"{fname} templates[{i}]", issues))]
    return doc


def _merge(into: dict, doc: dict):
    for k in _TEXT_FIELDS:
        into[k] = into.get(k) or doc[k]
    for k in (*_LISTS, "templates"):
        key = "title" if k == "templates" else "text"
        have = {x[key] for x in into[k]}
        into[k] += [x for x in doc[k] if x[key] not in have]


def _scan() -> Tuple[Dict[str, dict], List[str]]:
    docs: Dict[str, dict] = {}
    issues: List[str] = []
    for f in sorted(MODEL_DOCS_DIR.glob("*.json"), key=lambda p: (p.stem != p.stem.split(".")[0], p.name)):
        try:
            raw = json.loads(f.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            issues.append(f"{f.name}：读取失败（{type(e).__name__}），已跳过整个文件")
            continue
        doc = _normalize(raw, f.name, issues)
        if not doc:
            continue
        if doc["id"] in docs:
            _merge(docs[doc["id"]], doc)
        else:
            docs[doc["id"]] = doc
    return docs, issues


def _signature() -> tuple:
    try:
        return tuple((f.name, f.stat().st_mtime_ns) for f in sorted(MODEL_DOCS_DIR.glob("*.json")))
    except OSError:
        return ()


def get_docs() -> dict:
    """返回 {"docs": {id: doc}, "issues": [...]}；目录里文件有增删或修改时重新读取"""
    sig = _signature()
    with _lock:
        if sig != _cache["sig"]:
            docs, issues = _scan()
            for msg in issues:
                log.warning("模型说明：%s", msg)
            _cache.update(sig=sig, docs=docs, issues=issues)
        return {"docs": _cache["docs"], "issues": _cache["issues"]}
