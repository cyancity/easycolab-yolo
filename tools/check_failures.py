#!/usr/bin/env python3
"""部署失败归类检查：向工作台注入各类失败部署，断言 /api/state 返回的 fail 类型、链接和动作。

用法：python tools/check_failures.py
全程不联网：用临时数据目录，不读真实账号，不发 Colab 请求；失败日志取自部署里缓存的遥测。
"""
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

TMP = tempfile.mkdtemp(prefix="ec-check-")
os.environ["LOCALAPPDATA"] = TMP  # Windows 的数据目录
os.environ["HOME"] = TMP          # 其他系统的数据目录
os.environ["USERPROFILE"] = TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from workbench import deployments, failures, monitor, server  # noqa: E402

EMAIL, EMAIL2 = "tester.one+a@gmail.com", "other@gmail.com"
SECRET = "sk-ec-0123456789abcdef0123456789abcdef"
deployments._save = lambda: None  # noqa: SLF001 不落盘
server.accounts.emails = lambda: [EMAIL, EMAIL2]
monitor.ACCOUNT_CACHE.update({
    EMAIL: {"ccu": {"currentBalance": 120, "eligibleGpus": ["T4", "L4", "A100"]},
            "assignments": [{"endpoint": "ep-orphan-0001", "accelerator": "T4"}], "error": ""},
    EMAIL2: {"ccu": {"currentBalance": 0, "eligibleGpus": ["T4"]}, "assignments": [], "error": ""},
})

# (期望类型, 错误文本, 步骤, 账号, 显卡, 模型, 期望链接片段, 期望动作)
CASES = [
    ("capacity", "Colab 暂时分不出 H100（HTTP 503，已自动重试 3 次）。通常是该型号容量紧张。", 1, EMAIL, "A100", "q27",
     ["colab.research.google.com/?authuser="], ["switch", "redeploy", "wait"]),
    ("too_many", "Colab 拒绝分配（HTTP 412）：账号下活动实例过多或触发了临时用量限制，先释放其他实例再试。", 1, EMAIL, "L4", "q27",
     ["colab.research.google.com/?authuser="], ["redeploy", "release_orphans"]),
    ("not_eligible", "当前账号没有 H100 的使用资格，可用：T4, L4, A100。请换卡或切换到有该资格的账号。", 1, EMAIL, "L4", "q27",
     ["colab.research.google.com/signup?authuser="], ["switch", "redeploy"]),
    ("not_eligible", "Colab 拒绝了 A100（HTTP 400）：账号没有该型号的资格或参数不支持，请换卡。", 1, EMAIL, "A100", "q27",
     ["colab.research.google.com/signup?authuser="], ["switch", "redeploy"]),
    ("no_balance", "当前账号没有 A100 的使用资格，可用：T4。请换卡或切换到有该资格的账号。", 1, EMAIL2, "A100", "q27",
     ["colab.research.google.com/signup?authuser="], ["switch", "redeploy"]),
    ("auth", "尚未授权登录 Google 账号，请先点击「绑定账号」", 1, EMAIL, "A100", "q27", [], ["relogin", "redeploy"]),
    ("auth", "invalid_grant: Token has been expired or revoked.", 1, EMAIL, "A100", "q27", [], ["relogin", "redeploy"]),
    ("download", "下载未完成：Hugging Face 401 Unauthorized", 3, EMAIL, "A100", "q27",
     ["huggingface.co/Qwen/Qwen3.8-27B", "colab.research.google.com/?authuser="], ["redeploy"]),
    ("download", "下载权重超时", 3, EMAIL, "A100", "q27gguf", ["huggingface.co/unsloth/"], ["redeploy"]),
    ("load", "载入失败：CUDA out of memory", 4, EMAIL, "A100", "q27",
     ["huggingface.co/Qwen/Qwen3.8-27B", "colab.research.google.com/?authuser="], ["redeploy", "logs"]),
    ("reclaimed", "实例连续 10 次无响应，可能已被 Colab 回收", 3, EMAIL, "A100", "q27",
     ["colab.research.google.com/?authuser="], ["redeploy"]),
    ("timeout", "分配实例超时", 1, EMAIL, "A100", "q27", ["colab.research.google.com/?authuser="], ["redeploy"]),
    ("interrupted", "工作台重启，部署中断", 1, EMAIL, "A100", "q27", ["colab.research.google.com/?authuser="], ["redeploy"]),
    ("unknown", "唤醒结束但没有拿到服务地址", 2, EMAIL, "A100", "q27", ["colab.research.google.com/?authuser="], ["redeploy"]),
]

LOG = [f"[10:0{i}:00] line {i}" for i in range(9)] + [f"[10:09:00] API Key: {SECRET}", f"[10:09:01] url https://x.gradio.live/k/{SECRET}/v1"]


def inject(n: int, case) -> str:
    kind, err, step, email, gpu, model, _, acts = case
    did = f"t{n:02d}"
    log = "logs" in acts
    deployments._deps[did] = {  # noqa: SLF001
        "id": did, "email": email, "model": model, "gpu": gpu, "params": {}, "session": "wb-" + did,
        "status": "failed", "step": step, "step_text": "", "url": "", "comfy_url": "", "created_at": time.time(),
        "ready_at": None, "error": err, "events": [], "tele": {"console_lines": LOG} if log else {}, "hist": {"vram": [], "ram": []},
        "bench": {}, "fail_log": failures.capture_log({"tele": {"console_lines": LOG}}) if log else [],
    }
    return did


def main() -> int:
    client = TestClient(server.app, base_url="http://127.0.0.1:8790")
    ids = {inject(n, c): c for n, c in enumerate(CASES)}
    # release_failed：实例可能还活着，也要给出口
    deployments._deps["rf"] = {**deployments._deps["t00"], "id": "rf", "status": "release_failed", "after": "failed"}  # noqa: SLF001
    st = client.get("/api/state")
    assert st.status_code == 200, st.text
    views = {i["id"]: i for i in st.json()["instances"]}
    bad = 0

    def check(ok, msg):
        nonlocal bad
        if not ok:
            bad += 1
        print(("通过  " if ok else "失败  ") + msg)

    for did, (kind, err, step, email, gpu, model, link_bits, acts) in ids.items():
        f = views[did].get("fail") or {}
        got = f.get("kind")
        check(got == kind, f"{did} 期望 {kind} 实得 {got}  ← {err[:36]}")
        urls = [x["url"] for x in f.get("links", [])]
        check(all(any(b in u for u in urls) for b in link_bits) and (bool(urls) or not link_bits),
              f"{did} 链接 {urls}")
        check(f.get("actions") == acts, f"{did} 动作 {f.get('actions')}")
        check(bool(f.get("title")) and bool(f.get("hint")), f"{did} 有 title 和 hint")
        for u in urls:
            p = urlparse(u)
            check(p.scheme == "https", f"{did} https {u}")
            if "colab.research" in u:
                check(set(re.findall(r"[?&](\w+)=", u)) == {"authuser"} and u.endswith(email.replace("+", "%2B")), f"{did} 只带邮箱 {u}")
        if "logs" in acts:
            blob = "\n".join(f["log"])
            check(len(f["log"]) == 11 and SECRET not in blob and "line 8" in blob, f"{did} 日志 {len(f['log'])} 行且已打码")
    rf = views["rf"]["fail"]
    check(rf["kind"] == "release_failed" and rf["links"] and rf["links"][0]["url"].startswith("https://"), "release_failed 有 fail 和链接")
    # 旧字段不变
    for k in ("status", "error", "step", "events", "label", "gpu_req", "has_url"):
        check(k in views["t00"], f"旧字段 {k} 仍在")
    blob = st.text
    check(SECRET not in blob and EMAIL.replace("+", "%2B") in blob, "整个响应不含密钥")
    # 没有未纳管实例时不给释放入口
    monitor.ACCOUNT_CACHE[EMAIL]["assignments"] = []
    v = {i["id"]: i for i in client.get("/api/state").json()["instances"]}
    check("release_orphans" not in v["t01"]["fail"]["actions"], "无未纳管实例时不给 release_orphans")
    # 非失败实例不带 fail
    deployments._deps["ok"] = {**deployments._deps["t00"], "id": "ok", "status": "running"}  # noqa: SLF001
    check("fail" not in {i["id"]: i for i in client.get("/api/state").json()["instances"]}["ok"], "运行中实例不带 fail")
    # 实例连不上时退回遥测缓存；超长日志被截到行数和总量上限
    t0 = time.time()
    got = failures.capture_log({"url": "https://127.0.0.1:9", "tele": {"console_lines": LOG}})
    check(len(got) == 11 and SECRET not in "".join(got) and time.time() - t0 < 10, f"实例不可达时用缓存日志（{time.time() - t0:.1f} s）")
    big = failures.capture_log({"tele": {"console_lines": ["x" * 500] * 200}})
    check(len(big) <= failures.LOG_LINES and sum(map(len, big)) <= failures.LOG_CHARS, f"日志上限 {len(big)} 行 {sum(map(len, big))} 字符")
    print(f"\n{'全部通过' if not bad else f'{bad} 项失败'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
