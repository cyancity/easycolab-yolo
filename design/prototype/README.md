# EasyColab 工作台改版原型

改版直接落在 `workbench/static/`（基于前端定稿 967d23f）。内核先在本目录写成，再整体搬过去；搬完后本目录删除，规则并入 `DESIGN.md`。

开发模式：URL 带 `?mock=daily|trouble|fresh|offline` 时用样例数据（不碰真实 Colab），只认 URL 参数、不持久化，免得把样例当成真实账号状态。没有参数就直连真实后端。右下角工具条只在开发模式出现。

## 目的

PRODUCT.md 的使用者是产品负责人本人，按「第一次打开也能上手」设计。七条原则，每条都能用来否决设计：

1. 打开先回答「有没有事要我管」。需要处理的事排在最前，没有就不出现。
2. 每个数字带判断：达标绿、不达标黄、出错红，旁边能看到达标线（单流 100/150 tok/s、上下文 300K、显存 75%）。
3. 每个问题旁边就是处理按钮，不让人去别处找。
4. 默认值就是推荐值：部署矩阵默认选中打分最高的格子；空状态给「一键部署推荐」。
5. 会花钱或丢状态的操作要能认出来：停止运行中的实例要原地二次确认；释放没确认的实例不用确认（它在省钱）。
6. 接入地址永远一步可达：顶栏常驻「接入地址」，点一下复制。
7. 动效只说明变化：出现、消失、换位置、状态切换、数值变化、步骤推进。不做装饰性动画。

## 不能改的交互决定（来自前端负责会话）

- 部署固定三步：模型 → 账号 × 显卡矩阵 → 确认。不可选格子写原因（无资格、放不下、无余额、冷却 N 分）。默认选中打分：达标 +100，推荐 +50，显存占用 /10，余额（上限 500）/100。
- 部署失败的实例已自动释放，给「重新部署」；释放失败给「重试释放」并醒目提示可能还在扣 CU；不是工作台部署的实例给「释放」。
- 运行中的实例没达标，必须显示余量面板：哪项没达标、建议怎么做。
- 网关只有一个地址一把密钥。页面只显示打码密钥，复制时才调 `/api/gateway/key`。
- 账号显示 `label`（@ 前前 3 后 3）。顶栏保留离线、冷却两个状态标签。
- 全中文，不加多余的二级标签和说明文字。`api.js` 已有导出函数和 `ui` 已有字段不改名、不改语义，只新增。
- 3 秒轮询期间，焦点、输入框内容、弹层步骤、选中格子都保持。

## 文件与归属

同一时间一个文件只有一个写入者。

| 文件 | 归属 |
|---|---|
| `index.html`、`tokens.css`、`css/base.css`、`css/components.css`、`css/motion.css`、`js/app.js`、`js/api.js`、`js/util.js`、`js/icons.js`、`js/morph.js`、`js/motion.js`、`js/mock/*`、`js/devbar.js` | 内核 |
| `js/views/overview.js`、`js/views/instances.js`、`css/overview.css` | 概览与实例列表 |
| `js/views/detail.js`、`js/views/generate.js`、`css/detail.css` | 实例详情与生成 |
| `js/sheets/launch.js`、`js/sheets/login.js`、`js/views/accounts.js`、`js/views/models.js`（由定稿 `models.js` 迁来，另导出 `tplRef`）、`js/callcode.js`（定稿原样）、`css/sheets.css`、`css/pages.css` | 弹层、账号、模型 |

视图写手需要新的公共工具或图标时，写在自己文件里并在交付说明里列出，由负责人合并进内核文件。

## 模块接口

所有视图是纯函数：读 `ui`，返回 HTML 字符串，不碰 DOM、不发请求。远端来的文本一律 `esc()`。

```js
// js/views/overview.js
export function viewOverview(): string
// js/views/instances.js     （右侧调用 detail(i)）
export function viewInstances(): string
// js/views/detail.js        （运行中的图像、视频实例调用 genPanel(i)）
export function detail(i): string
// js/views/generate.js
export function genPanel(i): string                 // 不适用时返回 ""
export const genHandlers: Record<string, (t, e) => any>   // app.js 合并进 handlers
// js/views/accounts.js
export function viewAccounts(): string
// js/views/models.js
export function viewModels(): string
// js/sheets/launch.js
export function newLaunch(model?): void
export function recommend(): void                   // 写 ui.launch.email/gpu/rec
export function bestCell(modelId): { email, gpu, label } | null   // 不改 ui，供空状态用
export function renderLaunch(): string
// js/sheets/login.js
export function renderLogin(): string
```

`js/util.js` 导出（保留原有 `$ esc dur num pct sparkline meter ui toast icon`，新增）：

```js
export const isLLM = (i) => i.kind === "语言"
export const STATUS   // { running: ["ok","运行中"], starting: ["busy","部署中"], stopping: ["busy","释放中"],
                      //   failed: ["err","失败"], stopped: ["","已释放"], orphan: ["warn","未纳管"], release_failed: ["err","释放失败"] }
export const STEPS    // ["分配实例","部署服务","下载权重","载入显存","自测"]
export function sloChecks(i): Array<{ k: "tps"|"ctx"|"vram", label, value, line, ok: boolean|null }>
     // 语言模型三项，其他模型只有显存；数据缺失时 ok 为 null
export function attention(data, dismissed): Array<AttentionItem>
export function hoursLeft(cu, burn): number|null    // burn 为 0 返回 null
export function snippet(i, key, tab): string        // OpenAI 是 curl，Anthropic 是环境变量；key 传打码或真实密钥
export function num(v, d)                           // 原样保留
export function bar(p, cls)                         // 同 meter，但用 style="--p:42%"，宽度走 CSS 过渡
```

`AttentionItem = { key, sev: "err"|"warn", kind, id, title, sub, actions: [{ label, attr, value, tone: "primary"|"danger"|"" }] }`。按下表顺序排列（是否还在花钱优先）：

| kind | 条件 | 标题 | 动作 |
|---|---|---|---|
| release | `status === "release_failed"` | `{name} 没释放成功，可能还在扣 CU` | 重试释放 `data-stop` danger |
| orphan | `status === "orphan"` | `有一台 {gpu} 不是工作台启动的，正在扣 CU` | 释放 `data-orphan="{email}|{endpoint}"` danger |
| failed | `status === "failed"` 且 id 不在 dismissed | `{name} 部署失败` | 重新部署 `data-redeploy` primary；知道了 `data-dismiss` |
| login | 账号 `state === "error"` | `{label} 登录失效` | 重新登录 `data-relogin="{email}"` primary |
| cooldown | `data.cooldown.left > 0` | `分配暂停 {N} 分钟` | 换一张卡部署 `data-new` |
| slo | 运行中，单流 < tpsMin 或上下文 < ctxMin 或显存 < fillGoal | `{name} 没达标` | 看方案 `data-view-inst` |

`sub`：账号 label · GPU，加上原因（slo 写哪几项没达标，failed 写 `error` 首句）。

## ui 状态

原有字段：`data view sel apiTab launch login toast toastKind`。`view` 取值新增 `"overview"`，默认 `"overview"`。

新增字段：

```js
collapsed: Set(["ended"])     // 实例列表折叠的分组
dismissed: Set                // 「知道了」的失败实例 id，存 localStorage["dismissed"]
confirm: null                 // 正在二次确认停止的实例 id，3 秒后自动清空
msel: null                    // 模型页选中的模型 id
docs: null                    // /api/model_docs 缓存（数组）
gen: {}                       // { [instId]: { form: {...}, jobs: [{ id, status, progress, prompt, media: [{ kind, url, name }] }] } }
launch.dir: 1 | -1            // 步骤前进或后退，决定切换方向
launch.rec: "email|gpu"       // recommend() 选中的格子，矩阵里标「推荐」
login.email                   // 重新登录时预填的账号
```

## 事件约定

点击：`data-<name>` 的元素触发 `handlers.<camelName>(el, event)`，返回 `false` 时不重绘。值放在属性里。

| 属性 | 行为 |
|---|---|
| `data-nav="overview|instances|accounts|models"` | 切页 |
| `data-inst="{id}"` | 实例页选中 |
| `data-view-inst="{id}"` | 跳到实例页并选中 |
| `data-api="openai|anthropic"` | 切接入示例 |
| `data-new` / `#new` | 打开部署第 1 步 |
| `data-quick="{modelId}"` | 推荐组合直达第 3 步 |
| `data-deploy-model="{modelId}"` | 指定模型直达第 2 步 |
| `data-pick` `data-cell` `data-ctx` `data-next` `data-back` `data-start` `data-close` `data-close-login` | 部署弹层，语义同原版 |
| `data-stop="{id}"` | running 先进入二次确认（`ui.confirm`），再点才停；其他状态直接停 |
| `data-orphan` `data-bench` `data-redeploy` `data-copy` `data-copy-key` | 同原版 |
| `data-copy-snippet="{id}"` | 取真实密钥，复制当前标签页的完整调用示例 |
| `data-login` / `data-relogin="{email}"` / `data-login-open` / `data-login-finish` | 添加账号、重新登录 |
| `data-dismiss="{id}"` | 失败实例移出待处理 |
| `data-group="{name}"` | 折叠或展开实例列表分组 |
| `data-model="{modelId}"` | 模型页选中（定稿名称，`ui.msel`） |
| `data-model-deploy="{modelId}"` | 指定模型直达部署第 2 步（定稿名称） |
| `data-tpl-open="{modelId}:{idx}"` `data-copy-prompt` `data-copy-code` | 模板展开、复制提示词、复制调用代码（取完整密钥，提示「含完整密钥」），定稿逻辑不改 |
| `data-gen` `data-tpl` | 生成面板，本期不做 |

输入：`<input data-bind="gen.{instId}.form.prompt">` 在 input 事件里写回 `ui` 对应路径，渲染时从 `ui` 读值。

键盘：Esc 关弹层；`role=option`、`role=radio` 上 Enter 或空格等于点击；弹层打开时焦点移进弹层，关闭后回到触发按钮。

## 渲染与动效约定

`render()` 用 `morph(container, html)` 局部更新，不再整块替换 innerHTML。视图要配合：

- 重复项必须带父级内唯一的 `data-key`（如 `data-key="inst-{id}"`）。
- 子项会增删的容器加 `data-list`：新项带 `is-entering` 进场，删掉的项先加 `is-leaving` 播完再移除。要动画换位的再加 `data-flip`。
- 会跳变的数字写成 `<span data-num="168.4" data-dec="0">168</span>`，变化时补间。
- 进度和占比条用 `style="--p:42%"`，宽度由 CSS 过渡。
- 不想被重绘打断的元素（视频播放器）加 `data-static`。
- 页面根节点 `<div class="page" data-page="{view}">`，切页时进场。
- 弹层根节点带 `data-key="sheet-launch"` 或 `data-key="sheet-login"`，放在 `#sheet[data-list]` 里，开合都有动画。部署步骤体带 `data-step="{n}"`，按 `launch.dir` 左右滑入。

动效规格（时长和曲线只用 tokens.css 变量）：

| 场景 | 动画 | 时长 |
|---|---|---|
| 切页 | 淡入 + 上移 4px | base，ease-out |
| 列表项进场 / 离场 | 淡入 + 高度展开 / 淡出 + 收起 | slow，ease-out / fast，ease-in |
| 列表换位 | FLIP 位移 | base，ease-out |
| 状态点变绿 | 一次外扩光圈 | slow |
| 数字变化 | 从旧值补间 | slow |
| 进度、显存条 | 宽度过渡 | slow，ease-out |
| 弹层 | 遮罩淡入；面板 0.98 → 1 并上移 8px；关闭反向 | slow 开，fast 关 |
| 部署步骤 | 内容沿前进方向滑入 12px 并淡入 | base |
| 选中格子 | 边框和底色过渡，勾选图标缩放进入 | fast |
| toast | 从底部升起，自动消失前有收缩的时间线 | base |
| 待处理项被解决 | 收起离场 | slow |
| 按钮按下 | 缩到 0.98 | fast |

`prefers-reduced-motion` 下 tokens 的时长归零，JS 补间和 FLIP 也要跳过。

## 页面

视口 1440×900（侧栏 15rem），深色默认，浅色可切。

外壳：侧栏依次是品牌、导航（概览、实例、账号、模型，实例上有待处理数徽标，红或黄）、底部车队卡（余额 CU、消耗 CU/h、可撑约 N 小时、运行 N 台）、深浅色切换。顶栏：页面标题、离线和冷却标签、空白、接入地址胶囊（`127.0.0.1:8790/v1` + 复制）、新建部署。

概览（默认页）：
1. 需要你处理：`attention()` 的卡片列表，每行图标、标题、副行、右侧按钮。为空时整块不渲染。
2. 四个指标：运行（N 台，部署中 M）、达标（x / y 台）、消耗（CU/h）、可撑（约 N 小时）。
3. 实例卡片网格：运行中、部署中、释放中的实例。语言模型卡片主数字是单流 tok/s，下面一条带 100 和 150 刻度的达标轨；另有上下文、显存条、GPU 利用率小折线。图像视频模型卡片显示显存条和「生成」入口。部署中显示五段进度、当前步骤文字和已用时间。卡片底部：复制调用、详情。点卡片进详情。没有实例时显示空状态：新建部署、一键部署推荐（`bestCell` 给出的模型 · 显卡 · 账号）。
4. 账号条：每个账号一块，显示 label、档位、余额、可用显卡、状态点；点击去账号页。

实例页：左栏分组列表（需要处理：释放失败、未纳管；运行中：运行、释放中；部署中；已结束：失败、已释放，默认折叠），右栏详情：
- 头部：名称、状态、显卡、账号、运行时长、CU/h 和已用；右侧按状态给动作。停止运行中实例原地二次确认。
- 运行中：达标条（速度、上下文、显存三项勾叉，总结「达标」或「N 项没达标」）→ 指标面板（数字补间、折线）→ 余量面板（没达标时）→ 生成面板（图像、视频）→ 接入面板和过程时间线两栏。余量建议能直接执行的做成按钮（换 Flash-Next = `data-quick="flash-next"`）。
- 部署中：大号五段进度（完成打勾、当前转圈、文字、已用时间）+ 过程。
- 失败、释放失败、未纳管：横幅里直接放处理按钮 + 过程。

部署弹层：步骤条、按步骤左右滑动；模型按类型分组，第一个达标语言模型标「推荐」；矩阵的推荐格子带「推荐」角标，选中格子出勾；确认页摘要、上下文分段、未达标警告；启动后关弹层，新实例在列表进场并选中。

账号页：表格（账号、余额、消耗、实例、可用显卡、状态），失效行带「重新登录」，右上「添加账号」。

模型页：沿用定稿的 `models.js`、`callcode.js` 逻辑（左列表右详情；HF 和部署按钮；事实、优点、缺点、用法、模板，每条带来源链接，无障碍名称「来源」），只改样式、局部更新接线和动效。`GET /api/model_docs` 返回 `{ docs: { [id]: doc }, issues: [] }`，`ui.docs = r.docs`；停在模型页或 `ui.docs` 为空时每次轮询重拉。

```js
doc = { id, name, hf: "https://…", summary: "…",
  facts | strengths | weaknesses | usage: [{ text, source }],
  templates: [{ title, scene, prompt, negative?, sound?, music?, params: {…}, source, source_note }] }
```

生成面板：本期不做，`genPanel(i)` 返回 `""`，等 T4 链路跑通后由前端负责会话确认范围。图像、视频实例的详情里放「模板」入口，跳到模型页对应模型。网关真实形状：`POST /v1/h3/generate` → `{ job_id, status }`；`GET /v1/h3/jobs/{id}` → `{ status: queued|running|completed|failed, message, filename }`；`POST /v1/images/generations` → `{ data: [{ b64_json }] }`。

## 底线

- 颜色、尺寸、阴影、时长只引用 tokens.css 变量；字面量只出现在 tokens.css。1~3px 线宽和焦点环除外。
- 不引入外部依赖和 CDN；图标在 `icons.js` 内联（Lucide 图形）。
- 源文件不超过 500 行（CSS 和本文件不计）。不写绝对路径。
- 状态齐全：空、加载、错误、长文本、键盘焦点、浅色和深色。
