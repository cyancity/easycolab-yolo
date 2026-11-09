// 失败样例：字段和归类对齐后端 workbench/failures.py 的输出（标题和提示逐字取自该文件），邮箱是假数据
const TEXT = {
  "capacity": [
    "Colab 暂时没有这张卡",
    "多半是该型号容量紧张。过几分钟再试，或者换卡、换账号。若所有付费卡都分不出而 T4 正常，可能是账号被临时限制，去 Colab 网页连一次该 GPU 能看到提示。"
  ],
  "too_many": [
    "账号下活动实例太多",
    "Colab 限制同一账号的并发实例数。先释放这个账号下不用的实例，再重新部署。"
  ],
  "not_eligible": [
    "账号没有这张卡的资格",
    "这个账号当前不能使用所选显卡。换一张卡、换一个账号，或到 Colab 套餐页确认订阅。"
  ],
  "no_balance": [
    "账号没有计算单元",
    "付费显卡需要 Colab 计算单元（CU）。余额为 0 时只能用 T4；换有余额的账号，或到套餐页购买。"
  ],
  "auth": [
    "账号登录已失效",
    "授权过期或被撤销。重新登录这个账号后再部署。"
  ],
  "download": [
    "权重下载失败",
    "权重从 Hugging Face 下载。检查模型页是否仍可访问、是否需要同意协议或 Hugging Face 令牌；也可能只是网络波动，重新部署会续传。"
  ],
  "load": [
    "模型载入失败",
    "多为显存不足、编译失败或模型文件不完整。先看失败前日志，再决定换更大的卡还是调低上下文。"
  ],
  "reclaimed": [
    "实例被 Colab 回收或没有响应",
    "Colab 可能已回收这台实例（空闲、额度用尽或被抢占）。到 Colab 网页确认运行状态，再重新部署。"
  ],
  "timeout": [
    "等待超时",
    "某一步超过了等待上限。Colab 当前可能较慢，可以稍后重新部署，或到网页确认实例状态。"
  ],
  "interrupted": [
    "部署被中断",
    "工作台重启时部署还没完成，实例已自动释放。重新部署即可。"
  ],
  "release_failed": [
    "Colab 没有确认释放",
    "实例可能仍在消耗计算单元。工作台每分钟自动重试；也可以到 Colab 网页确认并手动结束。"
  ],
  "unknown": [
    "部署失败",
    "错误原因没有匹配到已知类型。先看过程记录和失败前日志，再到 Colab 网页确认账号和实例状态。"
  ]
};

const colab = (email, path = "") => `https://colab.research.google.com/${path}?authuser=${encodeURIComponent(email).replace("%40", "@")}`;
const HF = {
  q27: "https://huggingface.co/Qwen/Qwen3.8-27B", q27gguf: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF",
  "flash-next": "https://huggingface.co/Qwen/Qwen3.8-Flash-Next", zimg: "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
};

// kind 对应的动作；log 有内容才给「查看失败前日志」；orphans 是该账号下未纳管实例数
export function mkFail(kind, email, { model, log = [], orphans = 0 } = {}) {
  const [title, hint] = TEXT[kind], open = { label: "打开 Colab", url: colab(email) };
  const plans = { label: "Colab 套餐页", url: colab(email, "signup") };
  const hf = HF[model] ? [{ label: "Hugging Face 模型页", url: HF[model] }] : [];
  const links = { not_eligible: [plans], no_balance: [plans], auth: [], download: [...hf, open], load: [...hf, open] }[kind] || [open];
  const actions = ["redeploy"];
  if (["capacity", "not_eligible", "no_balance"].includes(kind)) actions.unshift("switch");
  if (kind === "capacity") actions.push("wait");
  if (kind === "too_many" && orphans) actions.push("release_orphans");
  if (kind === "auth") actions.unshift("relogin");
  if (log.length) actions.push("logs");
  return { kind, title, hint, links, actions, log, orphans: kind === "too_many" ? orphans : 0 };
}

export const LOAD_LOG = [
  "[11:42:03] 下载完成：Qwen3.8-27B 共 12 个分片，51.8 GB",
  "[11:42:41] 开始载入显存：bf16，上下文 128K",
  "[11:43:10] 载入第 9/12 个分片，显存 71.2 / 80 GB",
  "[11:43:26] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.50 GiB. GPU 0 has a total capacity of 79.14 GiB of which 3.12 GiB is free.",
  "[11:43:26] 载入失败：显存不足，已回滚已载入的分片",
  "[11:43:27] 当前显存 71.2 GB 已用，无其他模型占用",
];
export const UNKNOWN_LOG = [
  "[11:50:11] 启动推理服务，等待隧道地址",
  "[11:50:42] cloudflared: failed to connect to edge, retrying in 8s",
  "[11:51:20] 隧道进程退出，退出码 1",
];
