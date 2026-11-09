"""部署失败归类：把失败部署的错误文本、失败步骤归成一类，给界面一条能做的下一步。

fail = {kind, title, hint, links:[{label,url}], actions:[...], log:[...]}
actions 取值：redeploy 重新部署、switch 换卡或换账号、release_orphans 释放未纳管实例、relogin 重新登录、logs 查看失败前日志。
链接只放行 https，URL 里除账号邮箱外不带任何东西；邮箱只用于让 Colab 打开对应的 Google 账号。
"""
import re
from typing import List, Optional
from urllib.parse import quote

import requests

from .keys import redact

COLAB = "https://colab.research.google.com/"
LOG_LINES = 60      # 留存失败前日志的行数
LOG_LINE_CHARS = 240
LOG_CHARS = 12000   # 单次部署日志总字符上限，防止落盘文件膨胀

_RE = {
    "auth": re.compile(r"尚未授权|未登录|invalid_grant|refresh.{0,20}(fail|error)|token.{0,30}(expired|revoked|invalid)|凭据失效", re.I),
    "http401": re.compile(r"\b401\b|Unauthorized", re.I),  # 下载和载入阶段的 401 多半来自 Hugging Face，不算 Google 登录失效
    "too_many": re.compile(r"HTTP 412|TooManyAssignments|活动实例过多"),
    "capacity": re.compile(r"分不出|HTTP 503"),
    "no_balance": re.compile(r"余额|计算单元|\b0 CU\b|insufficient|not enough|quota", re.I),
    "not_eligible": re.compile(r"使用资格|没有该型号的资格|HTTP 400|无资格|not eligible", re.I),
    "reclaimed": re.compile(r"回收|连续 10 次无响应|服务没有响应|reclaimed", re.I),
    "download": re.compile(r"下载|download|huggingface", re.I),
    "load": re.compile(r"载入失败|out of memory|\bOOM\b|CUDA|编译|compile|显存不足", re.I),
    "timeout": re.compile(r"超时|timed?[ -]?out", re.I),
    "interrupted": re.compile(r"工作台重启"),
}

_TEXT = {
    # kind: (标题, 提示)
    "capacity": ("Colab 暂时没有这张卡", "多半是该型号容量紧张。过几分钟再试，或者换卡、换账号。若所有付费卡都分不出而 T4 正常，可能是账号被临时限制，去 Colab 网页连一次该 GPU 能看到提示。"),
    "too_many": ("账号下活动实例太多", "Colab 限制同一账号的并发实例数。先释放这个账号下不用的实例，再重新部署。"),
    "not_eligible": ("账号没有这张卡的资格", "这个账号当前不能使用所选显卡。换一张卡、换一个账号，或到 Colab 套餐页确认订阅。"),
    "no_balance": ("账号没有计算单元", "付费显卡需要 Colab 计算单元（CU）。余额为 0 时只能用 T4；换有余额的账号，或到套餐页购买。"),
    "auth": ("账号登录已失效", "授权过期或被撤销。重新登录这个账号后再部署。"),
    "download": ("权重下载失败", "权重从 Hugging Face 下载。检查模型页是否仍可访问、是否需要同意协议或 Hugging Face 令牌；也可能只是网络波动，重新部署会续传。"),
    "load": ("模型载入失败", "多为显存不足、编译失败或模型文件不完整。先看失败前日志，再决定换更大的卡还是调低上下文。"),
    "reclaimed": ("实例被 Colab 回收或没有响应", "Colab 可能已回收这台实例（空闲、额度用尽或被抢占）。到 Colab 网页确认运行状态，再重新部署。"),
    "timeout": ("等待超时", "某一步超过了等待上限。Colab 当前可能较慢，可以稍后重新部署，或到网页确认实例状态。"),
    "interrupted": ("部署被中断", "工作台重启时部署还没完成，实例已自动释放。重新部署即可。"),
    "release_failed": ("Colab 没有确认释放", "实例可能仍在消耗计算单元。工作台每分钟自动重试；也可以到 Colab 网页确认并手动结束。"),
    "unknown": ("部署失败", "错误原因没有匹配到已知类型。先看过程记录和失败前日志，再到 Colab 网页确认账号和实例状态。"),
}


def colab_url(email: str, path: str = "") -> str:
    return f"{COLAB}{path}?authuser={quote(email or '', safe='@')}" if email else COLAB + path


def _https(url) -> str:
    return url if isinstance(url, str) and url.startswith("https://") and " " not in url else ""


def kind_of(d: dict, cu: Optional[float] = None) -> str:
    """cu 是账号当前余额（未知传 None），用来区分「没资格」和「没余额」"""
    err, step = str(d.get("error") or ""), int(d.get("step") or 0)
    if _RE["auth"].search(err) or (step <= 2 and _RE["http401"].search(err) and not _RE["download"].search(err)):
        return "auth"
    for k in ("too_many", "capacity"):
        if _RE[k].search(err):
            return k
    if step <= 1 and re.search(r"\b503\b", err):
        return "capacity"
    if _RE["no_balance"].search(err):
        return "no_balance"
    if _RE["not_eligible"].search(err):
        return "no_balance" if cu is not None and cu <= 0 and str(d.get("gpu") or "").upper() != "T4" else "not_eligible"
    if _RE["reclaimed"].search(err):
        return "reclaimed"
    if step == 3 or _RE["download"].search(err):
        return "download"
    if step == 4 or _RE["load"].search(err):
        return "load"
    if _RE["timeout"].search(err):
        return "timeout"
    return "interrupted" if _RE["interrupted"].search(err) else "unknown"


def _model_link(hf: str) -> List[dict]:
    url = _https(hf)
    if not url:
        return []
    return [{"label": "Hugging Face 模型页" if "huggingface.co/" in url else "模型主页", "url": url}]


def classify(d: dict, hf: str = "", orphans: int = 0, cu: Optional[float] = None) -> dict:
    """d 是失败或释放失败的部署；hf 是模型主页；orphans 是该账号下未纳管实例数；cu 是账号余额"""
    email, log = d.get("email") or "", [str(x) for x in d.get("fail_log") or []]
    if d.get("status") == "release_failed":
        kind = "release_failed"
    else:
        kind = kind_of(d, cu)
    title, hint = _TEXT[kind]
    colab = [{"label": "打开 Colab", "url": colab_url(email)}]
    plans = [{"label": "Colab 套餐页", "url": colab_url(email, "signup")}]
    model = _model_link(hf)
    links = {"not_eligible": plans, "no_balance": plans, "auth": [], "download": model + colab, "load": model + colab}.get(kind, colab)
    actions = ["redeploy"] if kind != "release_failed" else []
    if kind in ("capacity", "not_eligible", "no_balance"):
        actions.insert(0, "switch")
    if kind == "too_many" and orphans > 0:
        actions.append("release_orphans")
    if kind == "auth":
        actions.insert(0, "relogin")
    if log:
        actions.append("logs")
    return {"kind": kind, "title": title, "hint": hint, "links": links, "actions": actions,
            "log": [redact(x) for x in log], "orphans": orphans if kind == "too_many" else 0}


def capture_log(d: dict) -> List[str]:
    """失败后、释放实例前取一份实例日志。先问实例，取不到就用最近一次轮询缓存的；最长约 8 s"""
    lines = None
    if d.get("url"):
        try:
            r = requests.get(d["url"].rstrip("/") + "/minimax_api/status", timeout=(3, 5))
            r.raise_for_status()
            lines = r.json().get("console_lines")
        except Exception:  # noqa: BLE001 实例可能已死，退回缓存
            lines = None
    if not isinstance(lines, list):
        lines = (d.get("tele") or {}).get("console_lines")
    out: List[str] = []
    for x in (lines if isinstance(lines, list) else [])[-LOG_LINES:]:
        s = redact(str(x)).strip()[:LOG_LINE_CHARS]
        if s:
            out.append(s)
    while out and sum(len(x) for x in out) > LOG_CHARS:
        out.pop(0)
    return out
