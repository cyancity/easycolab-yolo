# Colab Fleet

## Why
- 为什么：一个人管 5 个 Google 账号的 Colab GPU，开源模型要能一键部署成外部 agent 直接调用的 API；相信算力只有被吃满才不算浪费。
- 怎么做：只做部署、监控和压榨性能，不做模型试玩；外部只认一个本机固定地址；每个结论带实测数据（tok/s、上下文、显存）。
- 做什么：本机 PWA 工作台 + Colab 端服务；不做多用户、远程访问、网页内对话。

阶段：快速迭代

高风险路径：`workbench/accounts.py`、`workbench/keys.py`（凭据）；`workbench/gateway.py`（对外接口）；`bridge/`（Colab 分配与释放，涉及 CU 消耗和封号风险）。

## 导航
- 产品目标、硬指标、完成标准：`PRODUCT.md`
- 界面规则：`DESIGN.md`
- 运行与开发：`README.md`

## 项目约定
- Colab 分配请求必须克制：同一账号或显卡 503 后先冷却，不得短时间连续重试（有封号风险）。
- 改 `colab_server/` 或 `app.py` 后必须跑 `tools/build_notebook.py`，并跑 `tools/check_routes.py`。
- 改 `bridge/` 后必须跑 `tools/check_bridge.py`，命令行输出与协议前缀（`WAKE_LOG:` 等）不得改变。
- 模型资料 `workbench/model_docs/*.json` 每条必须带可打开的来源，追不到来源的不收。
- 账号数据在 `%LOCALAPPDATA%\EasyColab\workbench`，不进仓库。
