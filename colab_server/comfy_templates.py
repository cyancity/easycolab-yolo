"""按 ComfyUI 官方模板 id 列出它要的模型文件，部署时预先下载到服务器。

官方模板在加载节点的 properties.models 里写明 name / url / directory（子图里的节点也要看）。
模板 JSON 随 ComfyUI 的依赖 comfyui-workflow-templates 装在本机；读不到时退回 GitHub 上的同名文件。
我们不维护任何工作流，模型清单跟着官方模板走。网页版 ComfyUI 的「缺模型」弹窗只会下载到浏览器那台电脑，
所以必须在这里替用户下好。
"""
import json
import re
import urllib.parse
import urllib.request
from typing import List, Tuple

_RAW = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/{}.json"
_HF = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/main/(.+)$")
_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _load(tid: str) -> dict:
    if not _ID.match(tid):
        raise ValueError(f"模板 id 不合法: {tid}")
    try:
        from importlib import resources
        path = resources.files("comfyui_workflow_templates_json") / "templates" / f"{tid}.json"
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  包没装或版本里没有这个模板
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(_RAW.format(tid), timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))


def _safe(part: str) -> bool:
    """相对路径，逐段不能是空、. 或 ..，不能带反斜杠；先解码再查，%2E%2E 也拦得住"""
    return bool(part) and "\\" not in part and not part.startswith("/") \
        and all(seg not in ("", ".", "..") for seg in part.split("/"))


def template_models(ids: List[str]) -> List[Tuple[str, str, str, str]]:
    """[(models 下的目录, 文件名, HF 仓库, 仓库内路径)]，按目录加文件名去重"""
    found = {}
    for tid in ids:
        data = _load(tid)
        nodes = list(data.get("nodes") or [])
        for sg in (data.get("definitions") or {}).get("subgraphs") or []:
            nodes += sg.get("nodes") or []
        for node in nodes:
            for m in (node.get("properties") or {}).get("models") or []:
                sub, name, url = m.get("directory"), m.get("name"), m.get("url")
                if not (sub and name and url):
                    continue
                hit = _HF.match(url.split("?", 1)[0])
                if not hit:
                    raise ValueError(f"{tid} 的模型 {name} 不是 Hugging Face main 分支上的文件: {url}")
                repo, path = urllib.parse.unquote(hit.group(1)), urllib.parse.unquote(hit.group(2))
                if not all(_safe(x) for x in (sub, name, repo, path)):
                    raise ValueError(f"{tid} 的模型路径不安全: {sub}/{name} <- {repo}/{path}")
                found[(sub, name)] = (repo, path)
    if not found:
        raise ValueError("这些模板没有声明任何模型: " + ", ".join(ids))
    return [(sub, name, repo, path) for (sub, name), (repo, path) in found.items()]
