# Codex reasoning 投影与通知漂移修复

更新时间：2026-08-29

## 目标与状态

- 主目标：对齐 Codex 的 reasoning 可见性边界，raw reasoning 不进入用户可见执行过程；已实现的 `model/list/updated` 不再误报协议漂移。
- 当前阶段：完成。
- 状态：完成。

## 窄写集

- `src/lib/api/agentRuntime/appServerNotificationDrift.ts`
- `src/lib/api/agentRuntime/appServerNotificationDrift.test.ts`
- `src/components/agent/chat/components/timeline-utils/displayTextResolvers.ts`
- `src/components/agent/chat/components/timeline-utils/displayTextResolvers.test.ts`
- `src/components/agent/chat/components/AgentThreadTimeline.reasoning.test.tsx`
- `src/components/agent/chat/utils/reasoningDisplayText.ts`
- `src/components/agent/chat/utils/agentThreadGroupingItemSummary.ts`
- `internal/exec-plans/codex-reasoning-projection-and-notification-drift-fix.md`

工作树中已有的 Runtime、Provider、Scheduled Tasks、AppSidebar 与其他 Agent GUI 改动均为外部脏热区，本次不修改、不回退。

## 根因与边界

1. `model/list/updated` 已存在 typed protocol、App Server producer 和 model registry consumer，但 renderer drift catalog 漏登记，因此被错误分类为 `unknown`。
2. canonical reasoning Item 正确区分 `summary` 与 raw `content`，但 timeline display resolver 将两者合并显示，违背 Codex 默认 `show_raw_agent_reasoning = false` 的产品边界。
3. 现场 Turn `turn_84b50790d9ba4188a318dc24f88ab420` 于 2026-08-29 09:19:21 开始、09:21:00 正常 completed。`terminalRecoveryPoll` 是等待真实终态的恢复兜底，不是本次错误根因。

分类：

- `current`：App Server typed `model/list/updated`、canonical reasoning summary/content、Turn/Item lifecycle、renderer summary projection。
- `compat`：无新增；仅保留旧 reasoning `text` 作为没有 typed summary 时的可见 summary fallback。
- `deprecated`：无。
- `dead / deleted`：无；不新增 mock 或终态合成路径。

## Agent Verification Contract

```text
改动名称：Codex reasoning 投影与通知漂移修复
执行计划文件：internal/exec-plans/codex-reasoning-projection-and-notification-drift-fix.md
负责人：Codex
预算标签：budget:normal
风险等级：P1
影响模块：App Server notification drift、Agent timeline reasoning projection
不做范围：不改 provider wire、canonical 持久化、Turn 终态状态机、Electron/App Server command
```

Current 主链：

```text
前端入口：AgentThreadTimeline / model registry listener
前端网关：App Server notification adapter
Electron Desktop Host bridge：既有 app_server_handle_json_lines，本次不改
App Server method：model/list/updated、item/reasoning/*、turn/completed
RuntimeCore / service owner：model registry 与 RuntimeCore，本次不改
read model：canonical Thread/Turn/Item
runtime event：reasoning.summary、reasoning.delta、turn.completed
Evidence Pack 字段：protocol method、Turn/Item status、reasoning summary/content
GUI surface：Agent chat 执行过程
```

Happy Path：模型目录刷新只触发独立 registry 更新，不生成 conversation warning；reasoning summary 可见，raw content 保留在协议/read model 但不显示；Turn 只由真实 completed/failed/canceled 收口。

Evidence Layers：

| Layer               | 需要 | 计划证据                                           |
| ------------------- | ---- | -------------------------------------------------- |
| deterministic-smoke | 是   | 定向 Vitest、contracts、current fixture            |
| gui-trace           | 是   | GUI smoke 与 current fixture 真实 Electron         |
| runtime-transcript  | 是   | 现场 rollout Turn/Item 终态与 summary/content 分类 |
| release-artifact    | 否   | 非发版任务                                         |

必跑命令：

```bash
npx vitest run "src/lib/api/agentRuntime/appServerNotificationDrift.test.ts" "src/components/agent/chat/components/timeline-utils/displayTextResolvers.test.ts" "src/components/agent/chat/components/AgentThreadTimeline.reasoning.test.tsx"
npm run test:contracts
npm run smoke:agent-runtime-current-fixture
npm run typecheck
npm run verify:gui-smoke -- --reuse-running
```

Agent QC 场景映射：P1 Agent chat reasoning projection 与 terminal lifecycle；不运行 live Provider 或 full qcloop。

架构影响：非重大架构变更。只修正既有 renderer 投影策略与通知 catalog，不改变协议、跨层 owner、数据流、crate 或 package。

## 退出条件

1. `model/list/updated` 分类为 `known_diagnostic_only` 且不产生 conversation warning payload。
2. reasoning summary 继续显示，raw `content` 在折叠和展开状态均不进入 GUI。
3. 定向测试、contracts、current fixture、typecheck 与 GUI smoke 通过；notification 分类由 deterministic regression 锁定，真实 Electron 主链无 console/page error。
4. 完成记录包含验证结果、未验证原因与治理分类。

## 验证记录

- 定向 Vitest：通过，`3 files / 29 tests`；覆盖 `model/list/updated` diagnostic-only、summary 可见、raw `text/content` 在折叠与展开状态均不可见。
- ESLint：通过，窄写集 `0 warnings / 0 errors`。
- TypeScript：`npm run typecheck` 通过。
- Contracts：`npm run test:contracts` 通过；App Server client `299 checks`，frontend commands `22`，Electron host commands `85`，mock priority commands `0`，DevBridge truth commands `13`。
- Current fixture：`npm run smoke:agent-runtime-current-fixture` 通过；覆盖 completed/failed/canceled、cancel-then-continue、unknown Item、approval、Skills、MCP、media 和真实 Electron GUI/read model，`liveProviderUsed=false`。
- GUI smoke：`npm run verify:gui-smoke -- --reuse-running` 通过；证据 `.lime/qc/project-gates/standalone-shell-01-20260829014406-8049/shell-01-electron-smoke/summary.json`，`21/21` assertions，console/page/invoke/preload/IPC/legacy/mock fallback error 均为 `0`。
- 现场 runtime transcript：用户 session 的 Turn `turn_84b50790d9ba4188a318dc24f88ab420` 最终状态为 `completed`；reasoning Item 为 `summary=[]`、raw `content` 非空，随后 final-answer Item 与 Turn 依次 completed，证明终态状态机无需合成事件，问题位于 renderer 可见性边界。
- 未运行 live Provider：本次不改 provider wire 或 runtime 状态机，现场 transcript 已提供真实 provider 行为证据，额外调用会产生无关成本。
- 完成度：`100%`。current 投影和 diagnostic catalog 已修复；无 compat/deprecated/dead 新增，无 mock fallback。

## 追加修复：Codex Markdown 块级渲染对齐

更新时间：2026-08-29

### 目标与窄写集

现场 final-answer Item 保留了模型原始压缩 Markdown，其中标题、表格、列表、代码围栏和正文缺少块级换行。目标是在 Renderer current owner 中恢复语义边界，再交给既有 `react-markdown + remark-gfm` AST 渲染；不修改 canonical Item、provider wire、App Server read model 或终态恢复。

窄写集：

- `src/components/agent/chat/components/MarkdownRendererMarkdownModel.ts`
- `src/components/agent/chat/utils/markdownLooseSyntaxNormalizer.ts`
- `src/components/agent/chat/components/MarkdownRenderer.normalization.test.tsx`
- 本执行计划

### 根因与实现

1. `---###标题`、`标题**正文`、`标题|紧凑表格` 和 `**编号标题**-**列表项**` 缺少块级边界，GFM parser 只能按普通文本解析。
2. 压缩目录树把 `MemoryHub` 误当 fenced-code language，且树分支没有换行；当前修复将该形态恢复成 `text` 围栏和逐行树结构。
3. 原强强调修复正则允许跨行空白，会把相邻 `**` 错误配对并吞掉段落；已改为同一行内的状态化修复，并让普通松散语法只接受空格/Tab。
4. 正文中的 `A + B`、行内代码和表格续行会被列表启发式误拆；当前实现保护行内代码、限制 `+` 的压缩列表推断，并只在未闭合表格行内合并续行。
5. 对齐 Codex `markdown_render.rs` 的核心原则：保留 raw Markdown 作为事实源，先恢复块级结构，再由 Markdown parser/AST 处理标题、表格、列表、代码和行内样式；不建立第二套 HTML 字符串渲染器。

### 验证记录

- 真实 session `01a04b19-8ed6-7710-8606-caa08f699b09` final-answer 全文进入生产预处理流水线后得到 `7` 个标题、`4` 行表格数据、`14` 个列表项和逐行目录树；不再残留 `四大核心模块|模块`、`核心设计理念**`，`sqlite-vec`、`BM25 + 向量 + RRF`、`OpenTelemetry + Langfuse/Opik` 保持行内语义。
- Markdown/Streaming 定向 Vitest：通过，`4 files / 63 tests`；其中完整现场回答 DOM 回归直接断言标题层级、表格行、列表项、行内代码归属和纯文本代码树。
- TypeScript：`npm run typecheck` 通过。
- ESLint/Prettier：窄写集通过；`git diff --check` 通过。
- current fixture：`npm run smoke:agent-runtime-current-fixture` 通过；覆盖 history/cache hydration、真实 `turn.completed` 收尾、Claw 终态、approval、Plan、Skills、MCP、media、coding workbench、内容工厂和多入口 GUI fixture，`liveProviderUsed=false`。
- GUI smoke：`npm run verify:gui-smoke -- --reuse-running` 通过；Electron Shell-01 `21/21` assertions，`app-server` 初始化成功，证据目录为 `.lime/qc/project-gates/standalone-shell-01-20260829033754-99028/shell-01-electron-smoke/`。

分类：

- `current`：Renderer Markdown normalization、`react-markdown + remark-gfm` AST、canonical final-answer Item。
- `compat`：无新增。
- `deprecated`：无。
- `dead / deleted`：无；未增加 HTML fallback、mock backend 或第二套消息事实源。

## 追加修复：Turn start/accept 窗口的历史终态竞态

更新时间：2026-08-29

### 根因与实现

真实 Electron `gui-coding-input` 复现表明，第二回合在 listener 已绑定、后端尚未发出第一个带真实 `turnId` 的 `item/started` 时，React effect 仍可能执行上一帧的闭包。该闭包看到上一回合的 completed read model，提前调用 `settleActiveRuntimeStream`，清掉当前 listener；后端随后仍完成第二回合，但 GUI 只能保留失败前的旧投影。

修复收归 `useAgentRuntimeSyncEffects` current owner：

- 接入已有 `currentStreamingEventNameRef`，读取最新 active listener，避免旧 effect 闭包把 start/accept 窗口误判为空闲。
- listener 已存在且 stream 尚未绑定真实 `turnId` 时，runtime-sync 不消费历史 completed/failed/done read model；提交失败与静默终态恢复仍由对应 stream 生命周期负责。
- 保留本地 pending running turn 保护，确保 Codex `turn/started -> item/* -> turn/completed` 事件链完整投影，不用 `turn/completed` 替代 Item 列表。

### 验证

- 定向 Vitest：`useAgentRuntimeSyncEffects.terminalReadModel.test.tsx` `13/13`、`useAgentRuntimeSyncEffects.test.tsx` `24/24`、`useAgentStreamController.test.tsx` `6/6`，合计 `43/43` 通过。
- renderer 构建：`npm run build:renderer:electron:smoke` 通过。
- 真实 Electron Coding Workbench fixture：`gui-coding-input` 通过；第二回合保持 `isSending=true` 直到真实 `item_started/turnId` 到达，最终 `consoleErrors=[]`、`pageErrors=[]`，Workbench 与 artifact snapshot 均可见。
- Agent Runtime current fixture：`npm run smoke:agent-runtime-current-fixture` 通过，包含 history/cache、真实 `turn.completed`、cancel-then-continue、active steer、approval、Skills、MCP、media、Coding Workbench 与 Article Editor，`liveProviderUsed=false`。
- Contracts：`npm run test:contracts` 通过，App Server client `299 checks`，命令/脚本/模态/发布/文档边界均通过。
- GUI smoke：`npm run verify:gui-smoke` 通过，Electron Shell-01 `21/21` assertions，app-server 初始化成功，无 console/page/invoke/preload/IPC/legacy/mock fallback 错误。

分类保持不变：`current` 为 runtime-sync 与 Codex 对齐的 Turn/Item 生命周期保护；无新增 `compat`、`deprecated`、`dead` 或 mock fallback。
