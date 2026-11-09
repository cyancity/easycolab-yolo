---
version: alpha
name: EasyColab 工作台
description: 一个人管理多个 Google 账号的 Colab GPU，把开源模型部署成本机 API；按第一次打开也能上手设计。
colors:
  primary: "#1b1b19"
  on-primary: "#fafaf8"
  brand: "#2f5bd3"
  on-brand: "#ffffff"
  brand-text: "#2a55cc"
  brand-soft: "#eaf0fd"
  on-brand-soft: "#1f44a6"
  background: "#f7f7f5"
  surface: "#ffffff"
  surface-muted: "#f0f0ed"
  surface-hover: "#ebebe7"
  on-surface: "#1b1b19"
  on-surface-muted: "#62625c"
  on-surface-subtle: "#8b8b84"
  border: "#e3e3de"
  border-strong: "#cfcfc8"
  success: "#1f8a4c"
  success-soft: "#e7f4ec"
  on-success-soft: "#17693a"
  warning: "#a35f00"
  warning-soft: "#fbf1e1"
  on-warning-soft: "#7f4a00"
  error: "#c8312b"
  on-error: "#ffffff"
  error-soft: "#fbeaea"
  on-error-soft: "#9e2420"
typography:
  display:
    fontFamily: system-ui
    fontSize: 32px
    fontWeight: 600
    lineHeight: 40px
    letterSpacing: -0.01em
  headline:
    fontFamily: system-ui
    fontSize: 24px
    fontWeight: 600
    lineHeight: 32px
  title-lg:
    fontFamily: system-ui
    fontSize: 20px
    fontWeight: 600
    lineHeight: 28px
  title:
    fontFamily: system-ui
    fontSize: 16px
    fontWeight: 600
    lineHeight: 24px
  body-lg:
    fontFamily: system-ui
    fontSize: 16px
    fontWeight: 400
    lineHeight: 26px
  body:
    fontFamily: system-ui
    fontSize: 14px
    fontWeight: 400
    lineHeight: 22px
  body-sm:
    fontFamily: system-ui
    fontSize: 13px
    fontWeight: 400
    lineHeight: 20px
  label:
    fontFamily: system-ui
    fontSize: 13px
    fontWeight: 500
    lineHeight: 18px
  caption:
    fontFamily: system-ui
    fontSize: 12px
    fontWeight: 400
    lineHeight: 16px
  code:
    fontFamily: ui-monospace
    fontSize: 13px
    fontWeight: 400
    lineHeight: 20px
rounded:
  none: 0px
  sm: 4px
  md: 6px
  lg: 8px
  xl: 12px
  full: 9999px
spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
  2xl: 32px
  3xl: 48px
  control-sm: 28px
  control: 32px
  control-lg: 40px
  row: 40px
  sidebar: 240px
  topbar: 48px
  content-max: 1200px
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    height: "{spacing.control}"
    padding: 12px
  button-primary-hover:
    backgroundColor: "#383835"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    height: "{spacing.control}"
    padding: 12px
  button-secondary-hover:
    backgroundColor: "{colors.surface-hover}"
  button-ghost:
    textColor: "{colors.on-surface-muted}"
    rounded: "{rounded.md}"
    height: "{spacing.control}"
  button-ghost-hover:
    backgroundColor: "{colors.surface-hover}"
    textColor: "{colors.on-surface}"
  button-danger:
    backgroundColor: "{colors.error}"
    textColor: "{colors.on-error}"
    rounded: "{rounded.md}"
    height: "{spacing.control}"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    height: "{spacing.control}"
    padding: 10px
  button-brand:
    backgroundColor: "{colors.brand}"
    textColor: "{colors.on-brand}"
    rounded: "{rounded.md}"
    height: "{spacing.control}"
  link:
    textColor: "{colors.brand-text}"
  input-placeholder:
    textColor: "{colors.on-surface-subtle}"
  divider:
    backgroundColor: "{colors.border}"
    height: 1px
  switch-track:
    backgroundColor: "{colors.border-strong}"
    rounded: "{rounded.full}"
    height: 20px
    width: 36px
  switch-track-on:
    backgroundColor: "{colors.brand}"
  icon-success:
    textColor: "{colors.success}"
  icon-warning:
    textColor: "{colors.warning}"
  card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
    rounded: "{rounded.xl}"
    padding: 16px
  badge:
    backgroundColor: "{colors.surface-hover}"
    textColor: "{colors.on-surface-muted}"
    typography: "{typography.caption}"
    rounded: "{rounded.full}"
    height: 20px
    padding: 8px
  badge-success:
    backgroundColor: "{colors.success-soft}"
    textColor: "{colors.on-success-soft}"
  badge-warning:
    backgroundColor: "{colors.warning-soft}"
    textColor: "{colors.on-warning-soft}"
  badge-error:
    backgroundColor: "{colors.error-soft}"
    textColor: "{colors.on-error-soft}"
  table-header:
    backgroundColor: "{colors.surface-muted}"
    textColor: "{colors.on-surface-muted}"
    typography: "{typography.label}"
    height: "{spacing.control}"
  table-row:
    height: "{spacing.row}"
  table-row-hover:
    backgroundColor: "{colors.surface-hover}"
  table-row-selected:
    backgroundColor: "{colors.brand-soft}"
  nav-item:
    textColor: "{colors.on-surface-muted}"
    rounded: "{rounded.md}"
    height: "{spacing.control}"
  nav-item-active:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.on-surface}"
  dialog:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.xl}"
    padding: 20px
    width: 384px
---

# EasyColab 工作台

代码里实际生效的值在 `workbench/static/tokens.css`；本文件的 YAML 与它的浅色值一一对应，深色、阴影、动效、密度只写在 `tokens.css`。改值时两边同一提交改。产品定位和场景见 `PRODUCT.md`。

## Overview

安静、精确、内容优先。界面退到后面，数据和操作在前面，像一件顺手的工具而不是一张海报。层次靠留白、字重和底色区分，颜色只在需要传达含义时出现。

基于工作台预设：钴蓝品牌色、中性色温、默认密度、8px 圆角，深色默认、浅色可切，主视口 1440×900。本地单用户工具，由 Python 服务直接提供静态文件，不用构建链。

## Colors

- 中性色 `background`、`surface`、`surface-muted` 三层：画布最暗，承载内容的面板最亮，侧栏和表头介于两者之间。
- `on-surface-muted` 是次要文字，可以用于正文；`on-surface-subtle` 只用于占位、禁用和时间戳。
- 品牌色 `brand` 只做这几件事：选中的开关和复选框、进度、焦点环、选中行和选中导航的底（`brand-soft`）、链接（`brand-text`）、图表第一序列。不大面积铺底，不做渐变。
- 状态色成套使用：实心色用于图标和危险按钮，`*-soft` 做徽章和提示条的底，`on-*-soft` 做上面的文字。状态永远同时有文字或图标，不只靠颜色。
- 正文与背景对比度不低于 4.5:1，大字和图标不低于 3:1。

## Typography

- 默认用系统字体：macOS 和 iOS 是苹方加 SF，Windows 是微软雅黑 UI 加 Segoe UI，Android 是 Noto Sans SC 加 Roboto。不下载网络字体，桌面和移动端与系统观感一致。
- 界面正文 14px（`body`），表单、表格、列表都用它；长文阅读用 `body-lg`。一个页面最多三种字号加粗细组合承担层级。
- 字重只用 400、500、600。微软雅黑没有 500 和 600 的独立字形，Windows 上会落到常规或粗体，设计时不要依赖 500 和 600 的细微差别。
- 中文不加负字距，不用斜体；`display` 的负字距只对拉丁字母生效。
- 数字列用等宽数字（`font-variant-numeric: tabular-nums`）并右对齐。

## Layout

- 4px 网格。间距从 `spacing` 取值，与 Tailwind 默认 spacing 一致，Tailwind 项目直接用 `p-4`、`gap-2` 等类，不写任意值。
- 应用外壳：左侧栏 240px，顶栏 48px，内容区内边距 24px，窄屏 16px。阅读型页面内容最宽 1200px，工作台类页面铺满。
- 控件高度：小 28px、默认 32px、大 40px；表格行 40px。触屏设备自动放大到 40px 和 44px，触控目标不小于 44px。
- 断点沿用 Tailwind：640、768、1024、1280、1536。768 以下侧栏收起，表格隐藏次要列。
- 对齐靠父级 flex、grid 和 gap，不靠负 margin 或绝对定位。

## Elevation & Depth

- 先用底色分层，再用 1px 边框，阴影只给浮起来的东西：菜单、弹窗、toast、拖拽中的元素。
- 卡片默认只有边框没有阴影；当前选中的导航项和分段按钮用最轻的阴影。
- 深色模式不靠阴影分层，靠更亮的表面色和边框；浮层额外加一圈极淡的亮边。
- 弹窗遮罩用 `--overlay`，不做毛玻璃。

## Shapes

- 圆角基准 8px：按钮和输入框 6px，卡片和弹窗 12px，徽章和开关全圆。
- 嵌套时内层圆角不大于外层。
- 图标用 lucide，线宽 2，正文旁 16px，空态等大图标 20px 起。

## Components

- 按钮：主按钮每屏最多一个；其余用次按钮（白底加边框）或幽灵按钮（透明底，只在 surface 上用）。`button-brand` 留给需要品牌感的主操作。危险操作用 `button-danger`。停止运行中的实例在按钮上原地二次确认（3 秒内再点才执行），不弹窗；释放没确认或未纳管的实例不用确认，它在省钱。图标按钮必须有无障碍名称。
- 输入：边框用 `border-strong`，聚焦时换成品牌色加一圈浅色光晕。标签在上，提示和错误在下；不用占位符代替标签。错误时边框和文字都变成危险色，并说明怎么改。
- 表格：表头用 `surface-muted`，行悬停 `surface-hover`，选中 `brand-soft`。批量操作出现在表格上方工具栏。
- 导航：当前项用白底加最轻阴影，不用品牌色铺底；数量用 `caption` 放在右侧。
- 弹窗：标题写清后果，正文说明能否恢复，按钮写具体动作（删除、发布），不写「确定」。
- 反馈：成功用 toast 并给撤销；失败写原因和下一步（重试、已保存在本地）。
- 每个界面同时设计空态、加载、错误和长文本，不只设计有数据的样子。

## Do's and Don'ts

- Do 组件里只用语义 token；十六进制、具体 px 只出现在 token 文件。
- Do 所有可交互元素有悬停、键盘焦点（2px 品牌色焦点环）和禁用三种状态。
- Do 动效只动 transform 和 opacity，进入 180ms 左右，退出更快；尊重减少动态效果设置。
- Don't 用紫粉渐变、发光边框、毛玻璃、无意义的装饰图形。
- Don't 卡片里再套卡片，一层容器能解决就不用两层。
- Don't 用颜色作为状态的唯一信号。
- Don't 用 Tailwind 任意值（`p-[13px]`、`bg-[#3a5ccc]`）或 `!important` 修布局。

## 交互

七条原则，每条都能用来否决设计：

1. 打开先回答「有没有事要我管」。概览页最上面是待处理事项，没有就不出现。
2. 每个数字带判断：达标绿、不达标黄、出错红，旁边能看到达标线（单流 100/150 tok/s、上下文 256K、显存 75%）。
3. 每个问题旁边就是处理按钮。
4. 默认值就是推荐值：部署矩阵默认选中打分最高的格子；没有实例时给「一键部署推荐」。
5. 会花钱或丢状态的操作要能认出来，见 Components 的危险操作。
6. 接入地址永远一步可达：顶栏常驻，点一下复制。
7. 动效只说明变化，见下一节。

页面：概览（默认）、实例（左分组列表、右详情）、账号、模型（左列表、右资料）。部署固定三步：模型 → 账号 × 显卡矩阵 → 确认，不可选格子写原因。

待处理事项按「是否还在花钱」排序：释放失败 → 未纳管 → 部署失败 → 登录失效 → 冷却中 → 没达标。

渲染：`render()` 用 `morph()` 按 `data-key` 局部更新，3 秒轮询不打断焦点、输入、弹层步骤和滚动。视图是纯函数，只读 `ui`，远端文本一律 `esc()`。

开发模式：URL 带 `?mock=daily|trouble|fresh|offline` 时用样例数据，只认 URL 参数、不持久化。`python tools/dev_static.py` 从本仓库提供前端并把 `/api`、`/v1` 转发给运行中的后端。

## 动效

时长和曲线只用 `tokens.css` 的变量，`prefers-reduced-motion` 下全部归零，JS 补间和 FLIP 也跳过。

| 场景 | 动画 |
|---|---|
| 切页 | 淡入并上移，base |
| 列表项进场、离场 | 淡入并展开高度 / 淡出并收起，slow / fast。列表增删必须带动重排，这里是「只动 transform 和 opacity」的例外 |
| 列表换位 | FLIP 位移，base |
| 状态点变绿 | 一次外扩光圈 |
| 数字变化 | 从旧值补间，`data-num` |
| 进度、显存条 | 宽度过渡，`--p` |
| 弹层 | 遮罩淡入，面板从 0.98 放大并上移；关闭反向且更快 |
| 部署步骤 | 沿前进方向滑入 |
| 选中格子 | 边框和底色过渡，勾选图标缩放进入 |
| toast | 从底部升起，带收缩的时间线 |
| 按钮按下 | 缩到 0.98 |
