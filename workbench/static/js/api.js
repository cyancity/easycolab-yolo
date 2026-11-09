// 与本机工作台服务通信
import { ui } from "./util.js";

async function call(method, path, body) {
  const r = await fetch(path, {
    method,
    // 后端要求非 GET 一律声明 JSON，无请求体时也发 {}
    headers: method === "GET" ? {} : { "Content-Type": "application/json" },
    body: method === "GET" ? undefined : JSON.stringify(body ?? {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || data.error?.message || `请求失败 ${r.status}`);
  return data;
}

export async function refresh() {
  ui.data = await call("GET", "/api/state");
  ui.offline = false;
  return ui.data;
}

export const deploy = (model, email, gpu, params) => call("POST", "/api/deploy", { model, email, gpu, params });
export const stopInstance = (id) => call("POST", `/api/instances/${id}/stop`);
export const rebench = (id) => call("POST", `/api/instances/${id}/bench`);
export const stopOrphan = (email, endpoint) => call("POST", "/api/orphans/stop", { email, endpoint });
// email 只在重新登录时传，后端据此修复对应账号
export const loginStart = (email) => call("POST", "/api/accounts/login/start", email ? { email } : undefined);
export const loginFinish = (pending_id, code) => call("POST", "/api/accounts/login/finish", { pending_id, code });
export const gatewayKey = () => call("GET", "/api/gateway/key").then((r) => r.key);

// 模型资料：{ docs: { [id]: doc }, issues: [] }，失败由调用方忽略
export const modelDocs = () => call("GET", "/api/model_docs");
// 生成走网关：H3 视频、出图
export const h3Generate = (body) => call("POST", "/v1/h3/generate", body);
export const h3Job = (id) => call("GET", `/v1/h3/jobs/${encodeURIComponent(id)}`);
export const imageGenerate = (body) => call("POST", "/v1/images/generations", body);
