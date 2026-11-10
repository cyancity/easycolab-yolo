"""部署 ComfyUI：启动画布，再按勾选的模型包让 Colab 端下载官方模板声明的模型。
从 deployments.py 拆出；用到的部署工具函数在调用时从 deployments 取，避免循环导入问题"""
import time

import requests

from . import catalog
from . import deployments as dep
from .keys import redact


def start(d: dict):
    dep._step(d, 4, "启动 ComfyUI")
    dep._api(d, "/minimax_api/comfyui/start", {}, timeout=60)
    end, fails = time.time() + 1800, 0
    while time.time() < end:
        dep._check_alive(d)
        try:
            c = dep._api(d, "/minimax_api/comfyui/status").get("comfyui", {})
            fails = 0
        except (requests.RequestException, ValueError):
            fails += 1
            if fails >= 10:
                raise RuntimeError("实例连续 10 次无响应，可能已被 Colab 回收")
            time.sleep(5)
            continue
        d["step_text"] = redact(dep._EMOJI.sub("", str(c.get("status_message") or "启动 ComfyUI")))[:120]
        if c.get("is_running") and c.get("public_url"):
            d["comfy_url"] = c["public_url"]
            d["comfy_token"] = str(c.get("token") or "")  # 画布访问保护的令牌，旧版 Colab 端没有
            return
        time.sleep(5)
    raise TimeoutError("ComfyUI 启动超时")


def models(d: dict):
    """按部署时勾选的模型包，让 Colab 端下载官方模板声明的模型文件。失败只告警：画布照样能用"""
    packs = [catalog.COMFY_PACK_BY_ID[p] for p in d["params"].get("comfy_packs") or [] if p in catalog.COMFY_PACK_BY_ID]
    if not packs:
        return
    titles = "、".join(p["title"] for p in packs)
    pack_id = "tpl:" + ",".join(dict.fromkeys(t for p in packs for t in p["templates"]))
    d["step_text"] = f"下载模型：{titles}"
    dep._api(d, "/minimax_api/comfyui/install_pack", {"pack_id": pack_id, "auto_start": False}, timeout=60)
    start, fails = time.time(), 0
    while time.time() - start < 3600:
        dep._check_alive(d)
        try:
            c = dep._api(d, "/minimax_api/comfyui/status").get("comfyui", {})
            fails = 0
        except (requests.RequestException, ValueError):
            fails += 1
            if fails >= 10:
                raise RuntimeError("实例连续 10 次无响应，可能已被 Colab 回收")
            time.sleep(5)
            continue
        res = c.get("last_pack_result") or {}
        if res.get("id") == pack_id:
            if res.get("ok"):
                dep.event(d, "ok", f"模型已就绪：{titles}")
            else:
                dep.event(d, "warn", "模型下载失败：" + redact(str(res.get("message") or ""))[:200])
            return
        if c.get("installing_pack_id") != pack_id and time.time() - start > 90:
            dep.event(d, "warn", "模型下载没有开始，请在画布里检查")
            return
        d["step_text"] = redact(dep._EMOJI.sub("", str(c.get("status_message") or "下载模型")))[:120]
        time.sleep(5)
    dep.event(d, "warn", "模型下载超过 60 分钟，没有等它完成")
