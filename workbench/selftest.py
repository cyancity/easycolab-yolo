"""部署自测：真实请求一次，测单流解码速度和首字延迟。"""
import json
import time
from typing import Union

import requests

from .keys import gateway_key


def measure(d: dict) -> Union[dict, str]:
    """返回 {tps, ttft, tokens, at, ctx, srv_tps}；测不了时返回给用户看的原因。
    tps 是本机经隧道收到的速度；srv_tps 是 llama-server 末块 timings 里的解码速度 (有才写)，不含隧道开销"""
    from .gateway import instance_models  # 避免循环引用
    ids = [x["id"] for x in instance_models(d) if x.get("loaded")]
    if not ids:
        return "没有已载入的语言模型，跳过自测"
    body = {"model": ids[0], "stream": True, "max_tokens": 400,
            "messages": [{"role": "user", "content": "用中文详细介绍一下 GPU 显存带宽对大模型推理速度的影响，至少 400 字。"}]}
    t0, first, n, srv = time.time(), None, 0, None
    with requests.post(d["url"].rstrip("/") + "/v1/chat/completions", json=body, stream=True, timeout=300,
                       headers={"Authorization": "Bearer " + gateway_key()}) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[5:])
                srv = (chunk.get("timings") or {}).get("predicted_per_second") or srv
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            except Exception:  # noqa: BLE001
                continue
            if delta.get("content") or delta.get("reasoning_content"):
                n += 1
                first = first or time.time()
    end = time.time()
    if not first or n < 2:
        return "自测没有收到输出"
    tps = round((n - 1) / max(end - first, 1e-3), 1)
    res = {"tps": tps, "ttft": round(first - t0, 2), "tokens": n, "at": time.time(),
           "ctx": int((d.get("tele") or {}).get("loaded_context_length") or 0)}
    if srv:
        res["srv_tps"] = round(float(srv), 1)
    return res
