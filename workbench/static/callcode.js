// 由模板生成经本机网关调用的 Python 示例。图像走 OpenAI 风格接口，H3 走网关的 /h3 转发路由
const IMAGE_MODEL = { qimg: "Qwen/Qwen-Image-2.1", zimg: "Tongyi-MAI/Z-Image-Turbo" };
const SIZE = /(\d+)\s*[x×X*]\s*(\d+)/;

export const hasCode = (m) => m.id in IMAGE_MODEL || m.id === "h3";

// 缩进对象字面量的后续行，让它嵌进调用里
const obj = (o) => JSON.stringify(o, null, 4).replace(/\n/g, "\n    ");

function imageBody(m, t) {
  const p = t.params || {};
  const wh = p.width && p.height ? `${p.width}x${p.height}` : "";
  const fromRes = SIZE.exec(String(p.size ?? p.resolution ?? ""));
  const body = { model: IMAGE_MODEL[m.id], prompt: t.prompt,
    size: fromRes ? `${fromRes[1]}x${fromRes[2]}` : wh || "1024x1024", n: Number(p.n) || 1 };
  const steps = Number(p.steps ?? p.num_inference_steps);
  if (steps) body.steps = steps;
  return body; // 实例的生图接口不读 negative_prompt，不放进请求
}

function videoBody(t) {
  const body = { task_mode: "t2va", visual_prompt: t.prompt };
  if (t.negative) body.negative_prompt = t.negative;
  if (t.sound) body.soundscape_prompt = t.sound;
  if (t.music) body.music_prompt = t.music;
  const p = { ...(t.params || {}) };
  const res = SIZE.exec(String(p.resolution ?? ""));
  if (res && !p.width) [p.width, p.height] = [Number(res[1]), Number(res[2])];
  for (const [from, to] of [["frames", "num_frames"], ["steps", "num_inference_steps"]]) {
    if (from in p && !(to in p)) p[to] = p[from];
    delete p[from];
  }
  delete p.resolution;
  return { ...body, ...p };
}

export function callCode(m, t, base, key) {
  const auth = JSON.stringify("Bearer " + key);
  if (m.id in IMAGE_MODEL) {
    return `import base64, requests

r = requests.post(
    "${base}/images/generations",
    headers={"Authorization": ${auth}},
    json=${obj(imageBody(m, t))},
    timeout=600,
)
r.raise_for_status()
for i, im in enumerate(r.json()["data"]):
    open(f"image_{i}.png", "wb").write(base64.b64decode(im["b64_json"]))
`;
  }
  return `import time, requests

BASE = "${base}"
H = {"Authorization": ${auth}}

job = requests.post(
    f"{BASE}/h3/generate",
    headers=H,
    json=${obj(videoBody(t))},
).json()
while job["status"] in ("queued", "running"):
    time.sleep(10)
    job = requests.get(f"{BASE}/h3/jobs/{job['job_id']}", headers=H).json()
if job["status"] != "completed":
    raise SystemExit(job["message"])
video = requests.get(f"{BASE}/h3/media/{job['filename']}", headers=H)
open(job["filename"], "wb").write(video.content)
`;
}
