# Colab Fleet

本机多账号 Colab GPU 工作台。把开源模型一键部署到多个 Google 账号的 Colab GPU 上，经本机固定的 OpenAI / Anthropic 网关给外部 agent 调用，并检查速度、上下文和显存有没有吃满。

## 组成

| 目录 | 运行位置 | 作用 |
|---|---|---|
| `workbench/` | 本机 | FastAPI 服务 + PWA 界面：账号、部署、实例监控、固定网关 `/v1`、模型资料 |
| `bridge/`（入口 `colab_bridge.py`） | 本机子进程 | 用 google-colab-cli 申请、保活、释放 Colab 实例，上传服务端 |
| `colab_server/`（入口 `app.py`） | Colab 虚拟机 | 模型下载、加载、推理接口、ComfyUI |
| `runtime/` | Colab 虚拟机 | llama-server 编译与启动（Qwen3.8-Flash-Next 等） |
| `tools/` | 本机 | 笔记本生成和各项等价检查 |

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS / Linux 用 .venv/bin/python
.venv/Scripts/python -m workbench
```

浏览器打开 `http://127.0.0.1:8790/`，在 Chrome 地址栏点「安装」即成桌面应用。在账号页添加 Google 账号，新建部署时按「账号 × 显卡」矩阵选机器。外部 agent 统一调用 `http://127.0.0.1:8790/v1`，密钥在界面里复制。

申请实例默认不带 `shape=hm`（与官方 colab-cli 一致；2026-11-10 实测带 hm 的 T4 在免费档和 Ultra 账号上都返回 503）。`POST /api/deploy` 的 `params` 可以加两个开关：`high_mem: true` 给 T4/A100 申请大内存规格，`probe: true` 只发一次分配请求、503 不重试。llama.cpp 模型还可以传 `load.options`，例如 `{"load": {"options": {"quant": "UD-Q2_0", "draft_n": 2}}}`；可选量化见 `runtime/llama_specs.py`。

Flash-Next 在 G4 上的实测（2026-11-10，IQ4_XS，MTP 草稿 2）：

- 默认 256K、4 个请求槽（`-np 4 --kv-unified`）：部署自测单流 121.4 tok/s，显存 74.1 / 95.0 GB。
- 并发总吞吐：1 路 124、2 路 168、4 路 190 tok/s（每路 52–62）、8 路仍是 190。关掉 MTP 时单流 103、4 路 212。
- 内存提示词缓存（`--cache-ram`，上限取 60 GB 与内存 35% 的较小值）：同一段 9.6 万 token 前缀隔两个别的请求再发，预填充从 33.5 s 降到 0.2 s。命中来自内存缓存还是请求槽里残留的 KV，未确认。
- 33.6 万 token 输入（512K 部署）：预填充 1,376 tok/s，解码 30.6 tok/s。

Qwen3.8-27B GGUF（UD-Q4_K_M，256K，MTP）在同一台 G4：部署自测 87.2 tok/s，首字 1.32 s，显存 30.9 GB。Colab 分到的 A100 是 40GB 版，放不下 Flash-Next。

界面开发可以用样例数据，不碰真实 Colab：`http://127.0.0.1:8790/?mock=daily`（另有 `trouble`、`fresh`、`offline`）。

## 修改 Colab 端代码后

```bash
.venv/Scripts/python tools/build_notebook.py        # 重新生成 EasyColab_Server.ipynb
.venv/Scripts/python tools/check_routes.py          # 路由表回归
.venv/Scripts/python tools/check_bridge.py          # 桥接命令行回归
```

## 来源与许可

本项目源自 [wy51ai/easycolab](https://github.com/wy51ai/easycolab)（Apache License 2.0），保留原许可证，见 [LICENSE](./LICENSE) 与 [NOTICE](./NOTICE)。`colab_server/` 与 `bridge/` 由原项目的 `app.py`、`colab_bridge.py` 拆分并修改而来；`workbench/`、`runtime/` 为新增。

感谢 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)、[ComfyUI](https://github.com/comfyanonymous/ComfyUI)、[MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)、[Qwen](https://github.com/QwenLM)、[Unsloth](https://github.com/unslothai/unsloth)、[llama.cpp](https://github.com/ggml-org/llama.cpp)、[google-colab-cli](https://pypi.org/project/google-colab-cli/)。
