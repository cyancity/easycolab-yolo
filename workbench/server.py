"""工作台 HTTP 服务：/ 网页（PWA）、/api 管理接口、/v1 模型网关。只监听 127.0.0.1。"""
import time
import urllib.parse
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import accounts, catalog, deployments, failures, gateway, integrations, model_docs, monitor, prefs, waitlist
from .keys import gateway_key, mask_key, redact
from .paths import STATIC_DIR

PORT = 8790
app = FastAPI(title="EasyColab 工作台", docs_url=None, redoc_url=None)
app.include_router(gateway.router)

# 只认本机地址，挡 DNS rebinding；写接口再挡跨站请求
_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"[::1]:{PORT}"}
_ORIGINS = {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"}


def _forbid(msg: str, code: int = 403):
    return JSONResponse({"detail": msg}, status_code=code)


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.headers.get("host", "").lower() not in _HOSTS:
        return _forbid("Host 不合法", 400)
    path = request.url.path
    if path.startswith("/api/"):
        origin = request.headers.get("origin")
        if origin is not None and origin not in _ORIGINS:
            return _forbid("跨站请求被拒绝")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if not request.headers.get("content-type", "").lower().startswith("application/json"):
                return _forbid("写接口必须用 application/json")
            if request.headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
                return _forbid("跨站请求被拒绝")
        if path == "/api/gateway/key" and not (
                origin in _ORIGINS or request.headers.get("sec-fetch-site") == "same-origin"):
            return _forbid("完整密钥只给本页面读取")
    return await call_next(request)


def _num(v, default=None):
    """遥测字段可能是字符串或 None，转不了数就用默认值"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f and f not in (float("inf"), float("-inf")) else default


def _int(v, default=0) -> int:
    f = _num(v)
    return default if f is None else int(f)


def _account_view(email: str) -> dict:
    c = monitor.ACCOUNT_CACHE.get(email)
    view = {"email": email, "label": accounts.mask(email), "state": "loading", "note": ""}
    if not c:
        return view
    ccu = c.get("ccu") or {}
    eligible = [str(g).upper() for g in ccu.get("eligibleGpus") or []]
    view.update({
        "tier": accounts.tier(eligible) if ccu else "",
        "cu": _num(ccu.get("currentBalance")), "burn": _num(ccu.get("consumptionRateHourly"), 0),
        "count": _int(ccu.get("assignmentsCount")), "eligible": eligible,
        "cooldown": {g["id"]: deployments.cooldown_left(email, g["id"]) for g in catalog.GPUS
                     if deployments.cooldown_left(email, g["id"])},
        "state": "error" if c.get("error") and not ccu else "ok",
        "note": "凭据失效或网络异常：" + redact(c["error"]) if c.get("error") else "",
    })
    if view["cooldown"]:
        view["state"], view["note"] = "warn", "、".join(view["cooldown"]) + " 刚返回 503，冷却中"
    elif ccu and not view["cu"] and eligible == ["T4"]:
        view["note"] = "无 Colab 计算单元"
    return view


def _fail_view(d: dict, orphans: dict) -> dict:
    """失败实例的归类：余额和资格取自账号缓存，未纳管实例数决定要不要给「释放」入口"""
    ccu = (monitor.ACCOUNT_CACHE.get(d["email"]) or {}).get("ccu") or {}
    hf = ((model_docs.get_docs()["docs"].get(d["model"])) or {}).get("hf", "")
    return failures.classify(d, hf=hf, orphans=orphans.get(d["email"], 0), cu=_num(ccu.get("currentBalance")))


def _instance_view(d: dict, orphans: Optional[dict] = None) -> dict:
    m = catalog.BY_ID[d["model"]]
    t = d.get("tele") or {}
    gpu_t = t.get("gpu") if isinstance(t.get("gpu"), dict) else {}
    up = _int(t.get("uptime_seconds") or ((d.get("stopped_at") or time.time()) - d["created_at"]))
    comfy = d.get("comfy_url") or ""
    view = {
        "id": d["id"], "email": d["email"], "label": accounts.mask(d["email"]), "model": d["model"],
        "name": m["name"], "kind": m["kind"], "gpu": t.get("gpu_name") or d["gpu"], "gpu_req": d["gpu"],
        "status": d["status"], "step": d["step"], "step_text": redact(d.get("step_text", "")), "error": redact(d.get("error", "")),
        "uptime_s": up, "cuh": _num(t.get("cu_burn_rate_per_hour")), "used": _num(t.get("session_cu_used")),
        "vram_used": _num(t.get("vram_used_gb")), "vram_total": _num(t.get("vram_total_gb")),
        "ram_used": _num(t.get("ram_used_gb")), "ram_total": _num(t.get("ram_total_gb")),
        "ctx": _int(t.get("loaded_context_length")) // 1024, "reqs": _int(t.get("api_request_count"), None),
        "util": _num(gpu_t.get("gpu_util_avg_60s")),
        "util_hist": [v for v in (_num(x) for x in gpu_t.get("history") or []) if v is not None],
        "bench": d.get("bench") or {}, "hist": d.get("hist") or {},
        "events": [[e[0], e[1], redact(e[2])] for e in d["events"][-30:]],
        "comfy_url": comfy if isinstance(comfy, str) and comfy.startswith("https://") else "",
        "has_url": bool(d.get("url")),
        "packs": [p for p in (d.get("params") or {}).get("comfy_packs") or [] if p in catalog.COMFY_PACK_BY_ID],
        "models": [x["id"] for x in gateway.instance_models(d) if x.get("loaded")] if d["status"] == "running" else [],
    }
    if d["status"] in ("failed", "release_failed"):
        view["fail"] = _fail_view(d, orphans or {})
    if d["status"] == "waiting" and d.get("wait"):
        view["wait"] = {k: d["wait"].get(k) for k in ("tries", "max", "next_at")}
    return view


def _orphans(deps: list) -> list:
    """云端存在但不是工作台部署的实例（旧客户端开的或残留的），照样显示并可释放"""
    mine = {deployments.endpoint_of(d) for d in deps if d["status"] in deployments.ACTIVE}
    out = []
    for email, c in monitor.ACCOUNT_CACHE.items():
        for a in c.get("assignments") or []:
            if a.get("endpoint") not in mine:
                out.append({"id": "o-" + a["endpoint"][-10:], "email": email, "label": accounts.mask(email),
                            "endpoint": a["endpoint"], "gpu": a.get("accelerator", ""), "status": "orphan",
                            "name": "未纳管实例", "kind": "其他", "events": []})
    return out


@app.get("/api/state")
def state():
    deps = deployments.all_deps()
    recent = [d for d in deps if d["status"] in deployments.ACTIVE or time.time() - (d.get("stopped_at") or d["created_at"]) < 6 * 3600]
    recent.sort(key=lambda d: (d["status"] not in deployments.ACTIVE, -d["created_at"]))
    orphans = _orphans(deps)
    by_email: dict = {}
    for o in orphans:
        by_email[o["email"]] = by_email.get(o["email"], 0) + 1
    return {
        "slo": catalog.SLO, "gpus": catalog.GPUS, "prefs": prefs.get(), "models": catalog.public_models(),
        "accounts": [_account_view(e) for e in accounts.emails()],
        "instances": [_instance_view(d, by_email) for d in recent] + orphans,
        "gateway": {"base": f"http://127.0.0.1:{PORT}/v1", "key_masked": mask_key(gateway_key())},
        "cooldown": deployments.global_cooldown(),
        "integrations": {"magpie": integrations.magpie_installed()},
    }


class PrefsReq(BaseModel):
    gpu: str


@app.post("/api/prefs")
def set_prefs(req: PrefsReq):
    gpu = req.gpu.strip().upper()
    if gpu not in {g["id"] for g in catalog.GPUS}:
        raise HTTPException(400, f"不支持的显卡 {req.gpu}")
    return prefs.set_gpu(gpu)


@app.get("/api/model_docs")
def get_model_docs():
    return model_docs.get_docs()


class DeployReq(BaseModel):
    model: str
    email: str
    gpu: str
    params: dict = {}


@app.post("/api/deploy")
def deploy(req: DeployReq):
    if req.email not in accounts.emails():
        raise HTTPException(404, "账号不存在")
    gpu = req.gpu.upper()
    if gpu not in {g["id"] for g in catalog.GPUS}:
        raise HTTPException(400, f"不支持的显卡 {req.gpu}")
    m = catalog.BY_ID.get(req.model)
    if not m:
        raise HTTPException(400, "模型不存在")
    if not m.get("ready"):
        raise HTTPException(400, "该模型暂不能部署")
    if gpu not in (m.get("fit") or {}):
        raise HTTPException(400, f"{m['name']} 不支持 {gpu}")
    eligible = [str(g).upper() for g in ((monitor.ACCOUNT_CACHE.get(req.email) or {}).get("ccu") or {}).get("eligibleGpus") or []]
    if eligible and gpu not in eligible:
        raise HTTPException(400, f"该账号当前不能使用 {gpu}，可用：{'、'.join(eligible)}")
    if m.get("packs") is not None:
        packs = req.params.get("comfy_packs")
        if not isinstance(packs, list) or not packs or any(p not in catalog.COMFY_PACK_BY_ID for p in packs):
            raise HTTPException(400, "至少选一个模型包")
    try:
        d = deployments.start(req.email, req.model, gpu, req.params)
    except ValueError as e:
        raise HTTPException(400, str(e))
    monitor.refresh_accounts_soon()
    return {"id": d["id"]}


@app.post("/api/instances/{dep_id}/stop")
def stop(dep_id: str):
    if not deployments.get(dep_id):
        raise HTTPException(404, "实例不存在")
    deployments.stop(dep_id)
    monitor.refresh_accounts_soon()
    return {"ok": True}


@app.post("/api/instances/{dep_id}/wait")
def wait(dep_id: str):
    if not deployments.get(dep_id):
        raise HTTPException(404, "实例不存在")
    try:
        waitlist.enqueue(dep_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/instances/{dep_id}/bench")
def rebench(dep_id: str):
    d = deployments.get(dep_id)
    if not d or d["status"] != "running":
        raise HTTPException(400, "实例未运行")
    deployments.bench(d)
    return d.get("bench") or {}


@app.post("/api/instances/{dep_id}/canvas")
def canvas(dep_id: str):
    """打开 ComfyUI 画布的地址。带访问令牌，只经这个接口给本页面（中间件挡住跨站请求），不放进 /api/state"""
    d = deployments.get(dep_id)
    url = (d or {}).get("comfy_url") or ""
    if not d or d["status"] != "running" or not url.startswith("https://"):
        raise HTTPException(400, "画布还没就绪")
    token = d.get("comfy_token") or ""
    return {"url": f"{url}/__colab_fleet_auth?token={urllib.parse.quote(token)}" if token else url}


class OrphanReq(BaseModel):
    email: str
    endpoint: str


@app.post("/api/orphans/stop")
def stop_orphan(req: OrphanReq):
    if req.email not in accounts.emails():
        raise HTTPException(404, "账号不存在")
    live = (monitor.ACCOUNT_CACHE.get(req.email) or {}).get("assignments") or []
    if req.endpoint not in {a.get("endpoint") for a in live}:
        raise HTTPException(400, "该实例不在此账号的当前分配里")
    accounts.unassign(req.email, req.endpoint)
    monitor.refresh_accounts_soon()
    return {"ok": True}


class CodeReq(BaseModel):
    pending_id: str
    code: str


@app.post("/api/accounts/login/start")
def login_start():
    return accounts.login_start()


@app.post("/api/accounts/login/finish")
def login_finish(req: CodeReq):
    try:
        res = accounts.login_finish(req.pending_id, req.code)
    except Exception as e:  # noqa: BLE001 授权失败原因要原样给到界面
        raise HTTPException(400, str(e)[:300])
    monitor.refresh_accounts_soon()
    return res


@app.get("/api/gateway/key")
def full_key():
    return {"key": gateway_key()}


@app.post("/api/integrations/magpie")
def import_magpie():
    if not integrations.magpie_installed():
        raise HTTPException(400, "没有检测到 Magpie")
    try:
        integrations.open_link(integrations.magpie_link(gateway_key(), integrations.language_models()))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/sw.js")
def service_worker():
    # Service Worker 必须从根路径提供，作用域才覆盖整个应用
    return FileResponse(STATIC_DIR / "sw.js", media_type="text/javascript")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


def boot():
    imported = accounts.import_legacy()
    if imported:
        print(f"已从旧客户端导入 {imported} 个账号")
    model_docs.get_docs()  # 启动时读一遍，坏文件的告警直接出现在启动日志里
    deployments.load()
    monitor.start()
