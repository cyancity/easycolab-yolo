# EasyColab

**简体中文** | [English](./README_EN.md)

**EasyColab** 是一款专为 **macOS 与 Windows** 原生打造的 Google Colab 云端大显存多模态 AI 创作工作台（macOS SwiftUI 原生客户端 + Windows PySide6/Qt6 桌面客户端 + 云端多模型加速引擎）。

默认适配 **Google Colab G4（NVIDIA RTX PRO 6000 Blackwell · `96 GB` 显存 + `176 GB` 系统内存）** 以及 `H100 / A100 / L4` 实例，支持免开网页一键唤醒部署、Google Drive 云盘权重永久复用、多模型显存共驻、实时算力折线图监控、中英文双语界面一键切换与闲置自动释放省额度。

![EasyColab 工作台界面预览](./shot.png)

---

## 核心特性

- **双平台桌面级创作体验 + 中英文双语切换 + 新手交互引导（macOS & Windows 1:1 全功能对齐）**
  - **macOS 端**采用纯原生 SwiftUI + AppKit 构建，**Windows 端**采用 PySide6 (Qt6 + FFmpeg 多媒体管线) 构建并适配 Windows 11 沉浸式深浅色标题栏与高分屏缩放；首次启动均自带 **4 步交互式新手引导**，零配置自动初始化连接组件。
  - 支持 **中英文（简体中文 / English）界面实时一键切换**（默认显示中文），状态与偏好自动记忆，无需重启应用。
  - 实时绘制云端 **GPU 显存 / 系统内存 / 磁盘空间** 平滑曲线走势图（支持分项与同屏叠加模式），同步显示官方真实剩余 **CU 计算单元** 与每小时消耗速率。
  - 内置 **30 分钟闲置自动休眠看门狗** 与一键「断开并释放实例」，杜绝忘记关机空耗算力额度。

- **四大旗舰开源多模态引擎开箱即用**
  1. **`MiniMax-H3 (BF16 原生满血)` · 4～15 秒电影级有声视频**
     - 支持文生有声视频、首帧图生视频、首尾帧过渡控镜，原生输出 `24 FPS` 画面 + `32 kHz` 立体声音轨。
     - **G4 176GB 双主干内存热备加速**：自动将 `32B 文本编码器` 与 `33B DiT 主干` 常驻高速 DDR5 内存，同提示词连抽 `0 ms` 瞬回，修改新提示词整块 PCIe 换入解析仅需约 `8～12 秒`。
  2. **`ComfyUI 节点工作台` · 云盘持久化 + 4 套零基础预置工作流包**
     - 一键部署 ComfyUI + Manager 并开通免费 Cloudflare HTTPS 加密隧道，所有 `custom_nodes`、`models` 与 `workflows` 自动软链/同步至 Google Drive。
     - 内置 **万相 Wan2.2 14B**、**MiniMax-H3 Turbo**、**LTX-2.3 (22B)**、**混元 HunyuanVideo 1.5** 四套「一键配齐模型 + 节点指引」的图生视频与首尾帧运镜工作流。
  3. **`Qwen3.8-27B (UD-Q4_K_M + MTP 投机解码 / BF16)` · 100 tok/s 视觉对话与双协议网关**
     - 支持最高 `256K` 上下文、长思维链推理、多模态看图反推提示词。
     - 内置兼容 **OpenAI (`/v1/chat/completions`)** 与 **Anthropic (`/v1/messages`)** 的公网 API 网关（支持 Bearer / `x-api-key` 鉴权），可直接接入 **Claude Code**、**Cursor** 等开发工具。
  4. **`Qwen-Image-2.1` & `Z-Image-Turbo` · 2K 海报生图与多图指令修图**
     - 支持高质感中英文排版直出、双图自然语言指令修图与 8 步极速出图；所有生成的 MP4 / PNG 自动回传至本地目录（macOS: `~/Movies/EasyColab`，Windows: `%USERPROFILE%\Videos\EasyColab`）。

---

## 快速开始

### 本机多账号工作台（推荐）

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS / Linux 用 .venv/bin/python
.venv/Scripts/python -m workbench
```

浏览器打开 `http://127.0.0.1:8790/`，在 Chrome 地址栏点「安装」即成桌面应用。账号页添加 Google 账号（旧客户端登录过的账号会自动导入），新建部署时按「账号 × 显卡」矩阵选机器。外部 agent 统一调用 `http://127.0.0.1:8790/v1`（OpenAI / Anthropic 协议），密钥在实例页复制。


### 系统要求
- **操作系统**：
  - **macOS**：macOS 13.0 (Ventura) 或更高版本（推荐 Apple Silicon / macOS Sonoma & Sequoia）
  - **Windows**：Windows 10 (1903+) 或 Windows 11 64 位 (x64)
- **云端账号**：开通了 **Google Colab Pro / Pro+ / Google AI Ultra**（拥有可用 CU 计算单元）的 Google 账号；免费版用户请使用下方「Google Colab 官方网页端运行 `.ipynb`」。

---

### macOS 客户端

#### 方式一：本地一行命令编译运行（推荐，无 Gatekeeper 拦截）
只需系统装有 Xcode Command Line Tools（若未安装可执行 `xcode-select --install`）：

```bash
git clone https://github.com/<your-username>/EasyColab.git
cd EasyColab
bash macos_app/build_macos_app.sh
open EasyColab.app
```

> 编译脚本会自动将 `app.py`、`colab_bridge.py`、`EasyColab_Server.ipynb` 打包进 `EasyColab.app/Contents/Resources/`。你可以将生成的 `EasyColab.app` 直接拖入 `/Applications` 独立使用；首次点击「绑定」时，应用会在 `~/Library/Application Support/EasyColab/venv` 自动创建隔离 Python 环境并安装连接组件。

#### 方式二：从 GitHub Releases 下载预编译 `.app`
若下载的是未经过 Apple Developer ID 公证的预编译 `.app`，首次打开若遇到 macOS 提示 *“无法验证开发者 / 已损坏”*，请在终端执行一次解除隔离命令：

```bash
xattr -cr /Applications/EasyColab.app
```

---

### Google Colab 官方网页端运行（适合 Colab 免费版 / 浏览器用户）
1. 打开 [Google Colab](https://colab.research.google.com/)，上传本仓库根目录下的 `EasyColab_Server.ipynb`（或在客户端引导页点击「导出 / 定位 `.ipynb` 笔记本」）。
2. 在 Colab 菜单栏选择 `代码执行程序 -> 更改运行时类型`，选择 `G4` / `A100` / `L4` GPU 并按顺序运行单元格。
3. 将输出的 `https://xxxx.gradio.live` 链接粘贴至桌面客户端左侧服务地址栏点击「连接」即可。

---

## 隐私与安全建议

1. **录屏 / 截图防泄密开关**：在左侧「服务地址」输入框旁提供了 **打码保护开关**（macOS 为眼睛图标，Windows 为「打码」按钮）。在录制教程、直播或截图分享前，请开启打码隐藏你的 `*.gradio.live` 与 `*.trycloudflare.com` 临时公网隧道域名及账号邮箱，防止他人通过链接蹭用你的云端 GPU。
2. **开启 API Key 鉴权**：对外使用 Qwen API 网关时，请保持「启用 API Key 鉴权」处于开启状态。
3. **本地凭证与数据目录**：
   - Google OAuth 授权凭证仅保存在本机 `~/.config/colab-cli/token.json`（Windows 为 `%USERPROFILE%\.config\colab-cli\token.json`），不会上传至任何第三方服务器。
   - Windows 客户端本地配置文件位于 `%LOCALAPPDATA%\EasyColab\settings.json`，成片默认保存于 `%USERPROFILE%\Videos\EasyColab`。

---

## 免责声明与合规说明 (Disclaimer)

- **非官方独立项目**：EasyColab 是社区开发的独立开源客户端与辅助工具，**与 Google LLC、Google Colab、MiniMax、阿里巴巴通义千问（Qwen）等官方机构无任何隶属、赞助或背书关系**。
- **平台服务条款（ToS）提示**：
  - 客户端内的「免网页一键唤醒」功能底层基于开源社区库 [`google-colab-cli`](https://pypi.org/project/google-colab-cli/) 与自购付费计算单元（Paid Compute Units）交互，**强烈建议仅在已订阅 Colab Pro / Pro+ / Google AI Ultra 的自有付费账号上合理自用**。
  - 请勿在 Google Colab 免费层（Free Tier）上滥用无网页后台连接，严禁用于加密货币挖矿、批量多号滥用或违规算力转售。使用第三方客户端连接云端运行时可能受平台策略变更影响，相关账号风险由使用者自行评估并承担；若追求 100% 官方网页交互体验，请使用仓库内置的 `EasyColab_Server.ipynb`。
- **模型与上游组件许可证**：
  - 本仓库**不包含也不分发任何模型权重或 GPL 源码**。所有模型权重均由用户在运行时从 Hugging Face 官方仓库自行下载，须遵守各模型原始许可证（如 *MiniMax-H3 Community License Agreement*、*Apache-2.0*、*Tongyi Qianwen License* 等）。
  - 云端 ComfyUI 工作台在用户个人虚拟机内按需拉取官方仓库并通过独立进程 HTTP API 通信，ComfyUI 本体遵循其 **GPL-3.0** 协议。

---

## 致谢 (Acknowledgements)

感谢以下优秀的开源项目与模型团队：
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) & [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)
- [Qwen](https://github.com/QwenLM) & [Unsloth](https://github.com/unslothai/unsloth)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [google-colab-cli](https://pypi.org/project/google-colab-cli/)

---

## License

本项目源代码基于 [Apache License 2.0](./LICENSE) 开源发布。
