# Lime 核心用户主流程 E2E 审计

状态：本轮范围完成；Windows packaged updater 待实机验收
日期：2026-07-31

## 目标与用户闭环

普通桌面用户从启动 Lime 开始，完成 Provider / 模型确认，创建对话并完成一轮响应，再验证停止后继续、历史恢复、关键设置与更新状态。完成标准不是页面可打开，而是 GUI 状态、Electron IPC、App Server current JSON-RPC 与 read model identity 一致，生产 mock fallback 命中为零。

## 范围

- 启动、首页加载、主导航、控制台基线。
- `设置 -> AI 服务商` 的 Provider 列表、显式模型可见性与返回聊天后的模型选择。
- Claw 新会话、用户消息、流式 assistant 输出与真实 terminal。
- 停止、输入框立即恢复、同一会话继续下一轮。
- 最近对话 / 历史详情恢复与 Electron 重启后的 read model。
- `设置 -> 关于` 的当前版本与更新状态语义。

不做范围：不修改真实用户 Provider 凭证；不保存配置、API key、完整 prompt 或 provider 响应；不把 external fixture 当作 live Provider；不宣称 macOS Gate B 能替代 Windows Squirrel packaged 验收；不覆盖社媒内容、素材、浏览器自动化、MCP、Agent Apps 和所有 `@` 命令的产品矩阵。

## 写集

- 本执行记录。
- `src/components/api-key-provider/ModelAddPanel.tsx`。
- `src/components/api-key-provider/ApiKeyProviderSection.ui.test.tsx`。
- `src/RootRouter.tsx`、`src/RootRouter.test.tsx`。
- `src/components/ui/sonner.test.tsx`。
- `src/components/agent/chat/utils/agentRuntimeErrorPresentation.ts` 及其测试。
- `src/components/agent/chat/hooks/agentStreamSubmitFailure.ts`。
- `src/components/agent/chat/workspace/useWorkspaceSendActions.ts` 及其测试。
- `src/i18n/resources/{zh-CN,zh-TW,en-US,ja-JP,ko-KR}/settings.json`。
- 发现其它缺陷后，只扩展到最接近根因的 current owner、定向测试与五语言资源，并在修改前登记。

避让：`internal/aiprompts/architecture.md`、`internal/exec-plans/refactor-v2-implementation.md`、App Server protocol / Skills runtime 并行脏区，以及当前未归属本任务的工作树改动。

## Agent Verification Contract

### 基本信息

```text
改动名称：Lime 核心用户主流程 E2E 审计
执行计划文件：internal/exec-plans/core-user-flow-e2e-audit-2026-07-31.md
负责人：Codex
预算标签：budget:normal
风险等级：P0
影响模块：Electron Desktop Host、App Server current session、Claw GUI、Provider / model GUI、历史恢复、Updater GUI
不做范围：live Provider、真实用户配置写入、Windows packaged installer
```

### Current 主链

```text
前端入口：Claw 首页、模型选择器、设置页、最近对话
前端网关：src/lib/api/* typed gateway / AppServerClient
Electron Desktop Host bridge：preload contextBridge -> electron-ipc -> app_server_handle_json_lines
App Server method：agentSession/* current session surface 及对应 read/list；宿主更新能力不进入业务 JSON-RPC
RuntimeCore / service owner：RuntimeCore external fixture backend；Updater 归 electron/updateHost.ts
read model：Thread / Turn / Item projection 与 current session read model
runtime event：message.delta、turn.completed、turn.canceled 及 current terminal projection
Evidence Pack 字段：method、transport、status、sessionId、turnId、scenario marker、mock/console error 计数
GUI surface：Claw 输入框、消息列表、停止按钮、最近对话、AI 服务商、关于
```

### Happy Path

```text
用户输入 / Agent 输入：隔离 fixture 中发送无敏感信息的场景 marker
预期 runtime events：turn started -> message delta -> turn completed；取消场景为 turn canceled
预期 tool calls：基础聊天无需工具；不以工具调用作为完成条件
预期 approval / sandbox：无审批；fixture 不写用户工作区
预期 artifact：无
预期 evidence：Electron/preload/IPC/App Server/runtime/read model/GUI identity 一致
预期 GUI 状态：消息可见、terminal 后输入框恢复、历史可重新打开、Provider 显式模型可见、更新状态不误报
失败时应停在哪一层：按 Renderer、Electron bridge、App Server、runtime/read model、平台 updater 精确分类
```

### Evidence Layers

| Layer | 本次是否需要 | 证据路径 / 计划路径 | 不需要的原因 |
| --- | --- | --- | --- |
| deterministic-smoke | 是 | `smoke:agent-runtime-current-fixture` 与 Claw fixture summaries | - |
| gui-trace | 是 | 真实 Electron fixture / CDP，记录脱敏 trace 摘要 | - |
| runtime-transcript | 是 | fixture summary 中的 terminal 与 read model identity | - |
| release-artifact | 否 | - | 本轮不制作或安装发行包 |

### 必跑命令

```bash
# C0
npm run test:contracts

# C1
npm run smoke:agent-runtime-current-fixture
npm run smoke:claw-chat-current-fixture
npm run smoke:claw-chat-current-fixture -- --scenario cancel
npm run smoke:claw-chat-current-fixture -- --scenario cancel-then-continue
npm run smoke:agent-session-history-electron-fixture

# C2
npm run verify:gui-smoke
LIME_ELECTRON_REMOTE_DEBUGGING_PORT=9223 npm run electron:dev
npm run bridge:health -- --timeout-ms 120000
```

未跑：C3 / C4 qcloop、live Provider、Windows packaged updater。
原因：用户未要求发送数据到真实 Provider，本机不是 Windows packaged 环境。
风险：不能证明具体线上 Provider 的回答质量、网络稳定性或 Windows 安装器闭环。
后续触发条件：明确授权 live Provider，或提供 Windows N-1 packaged 测试机。

### Agent QC 场景映射

```text
P0: 启动与主链可用、消息可见、真实 terminal、取消、取消后继续、历史恢复
P1: Provider / 模型可见性、设置导航、更新状态语义
P2: 非主流程页面视觉与扩展能力
```

选择依据：本轮验证用户最基本的可用闭环；允许 deterministic sidecar / external fixture，但 official evidence 只声明到对应 Gate，不冒充 live Provider 或 packaged Windows。

### Supervisor Rubric

本轮不使用 LLM Supervisor。确定性断言只判断：主流程能否完成、identity 是否一致、mock / console / invoke error 是否为零。

### 回写规则

```text
失败类型：产品阻塞、体验误导、信息泄露、桥接缺口、测试缺口、环境噪音
回写资产：最接近根因的单元 / 组件测试、fixture smoke、Playwright trace 或 GUI flow
关闭条件：修复后同一场景复测通过，且不恢复 compat / mock fallback
```

### 完成标准

```text
主线目标是否完成：是；已完成 controlled current 主链与 macOS Electron Gate B，未扩张到 live Provider 或 Windows packaged claim
已跑验证：Provider / updater / typed error 定向回归、contracts、App Server protocol check、current runtime 聚合 fixture、typed error success/failure Gate B、GUI smoke、版本一致性
未跑验证及原因：live Provider 与 Windows packaged，见上文
是否存在 token / Provider / GUI owner 风险：不调用 live Provider；真实 Electron CDP 独占本轮窗口
是否可进入 release evidence：否，缺 Windows packaged 与 live Provider claim
下一刀：在 Windows N-1 Squirrel 安装包上完成发现、下载、重启安装与最终版本确认
```

### 架构图确认

```text
架构影响：当前为只读 E2E 审计，尚未改变 owner、协议、read model 或交付门禁
架构图已更新：不适用；若修复触达架构敏感路径，将重新分类
责任开发者确认：不适用；当前无重大架构变更
```

## 验收矩阵

| 场景 | Proof level | 状态 | 证据 / 问题 |
| --- | --- | --- | --- |
| 启动、首页与导航 | Gate B | 通过 | `verify:gui-smoke` 21/21；首页长短输入 current Electron fixture 通过 |
| Provider 与显式模型 | Gate A + Gate B bridge | 通过，Windows 平台待补 | Provider / ModelSelector 68/68；Provider hooks / storage / context / integration 26/26；不声明 live Provider |
| 新会话、stream、terminal | Current fixture + Gate B | 通过 | 聚合 current fixture 通过；typed error success/failure 均为 Gate B controlled fixture |
| 停止后继续 | Current fixture + Gate B | 通过 | cancel、cancel-then-continue 与 active steer 聚合场景通过 |
| 历史恢复与重启 | Gate B fixture | 通过 | history/cache hydration、Plan history hydrate 与 Coding Workbench 恢复通过 |
| 关于与更新状态 | Gate A + host unit | 通过，Windows packaged 待补 | updater / About / UpdateNotification / Sidebar 51/51；版本 `1.117.0`；Squirrel N-1 实机未验收 |

## 问题台账

| 严重度 | 分类 | 场景 | 状态 | 根因 / 修复 |
| --- | --- | --- | --- | --- |
| P0 | 桥接缺口 / 构建阻塞 | 任何需要 source 重建 App Server sidecar 的 Electron 主流程 | 已解决 | typed `error` notification 的 Rust 关联项歧义已消除；`cargo check -p app-server-protocol`、host 重建与 GUI smoke 通过 |
| P0 | 桥接缺口 / contract 漂移 | Electron host fixture 重建 | 已解决 | generated TypeScript notification union、client 与 fixture 已同步；`test:contracts` 和聚合 Electron fixture 通过 |
| P0 | 产品阻塞 / 体验误导 | 自定义 Provider 添加 | 已修复 | 未保存态改为“取消”；保存成功但连接测试失败时，“完成添加”仍激活已持久化 Provider |
| P0 | 产品阻塞 | 发送失败反馈 | 已修复并回归 | 恢复全局 Toaster；typed error success/failure Electron fixture 均无 console / invoke error |
| P0 | 信息泄露 / 体验误导 | 不可执行模型路由发送 | 已修复并回归 | 用户只看到五语言 Provider 不可用提示，内部路由错误保留在诊断边界 |
| P0 | 测试缺口 | typed retry GUI 等待 | 已修复 | fixture 原先只在用户消息分组内查 assistant 状态且错误要求输入框禁用；改用全局 `data-status` 语义状态并验证 active steer 输入可用 |
| P0 | 测试缺口 | typed retry trace 聚合 | 已修复 | external backend 在 retry 后漏发 `provider.first_text_delta.received`；补齐 Provider checkpoint，不放宽 Provider / App Server 分离断言 |
| P0 | 测试缺口 | typed retry read model | 已修复 | 删除过时的 `detail.thread_read` 私有状态读取，统一使用 canonical `turns[]` owner；completed 等待同时要求真实 terminal |
| P0 | 测试缺口 | typed retry failure 中间态 | 已修复 | 不再等待不会进入 `thread/read` 的内部错误 detail；改为证明 prompt 保留且 read model 在 `turn.failed` 前仍为非终态 |
| P1 | 可维护性 | Electron fixture 编排文件超过 1000 行 | 已登记退出条件 | `backend-script.mjs`、`smoke.mjs` 与 `smoke.test.mjs` 不再继续承接下一种 backend 场景；新增场景前先把 typed error backend 生成片段和 assertion guard 拆到独立领域模块，聚合入口只保留编排 |

## 执行日志

- 2026-07-31：建立 `budget:normal / P0` 审计合同；本机 `1420 / 3030 / 9223` 均无监听，从干净环境开始。
- 2026-07-31：`npm run smoke:agent-runtime-current-fixture` 通过；覆盖首页首发、stream terminal、停止后继续、历史 hydrate、审批、计划、Skills / MCP、媒体与工作台，明确 `liveProviderUsed=false`。
- 2026-07-31：随后单独运行 `npm run smoke:claw-chat-current-fixture` 时，检测到 13:14 更新的并行 App Server protocol source 比 13:03 的 sidecar 新，重建在 `envelopes.rs:753` 失败；错误为 `ambiguous associated item`。旧聚合证据有效但不能替代当前 source build，新的 Gate B 暂停。
- 2026-07-31：测试进程临时设置 `RUSTFLAGS=-Aambiguous-associated-items` 后，`cargo check -p app-server-protocol` 通过，但继续重建 host 时 `packages/app-server-client` 因 `error` notification 未进入生成 method union 而失败。临时 lint 参数只用于继续诊断，不关闭默认构建阻塞。
- 2026-07-31：隔离 Electron CDP 复现 Provider 静默丢失：填写 `E2E Local Provider / e2e-explicit-model` 后点击“完成添加”，页面回到“还没有启用模型”，console / invoke error 均为 0；源码确认该按钮直接绑定 `onCancel`。
- 2026-07-31：继续复现保存失败分支：本机不可达 endpoint 的配置成功持久化，设置列表显示 `e2e-explicit-model` 且标为默认，但返回首页仍只显示 `gpt-5.2-pro`；根因为完成动作未调用 `onActivated`，导致 persisted provider 与 active provider 分裂。
- 2026-07-31：沿不可执行 Provider 真实发送：`agentSession/turn/start` 经 `electron-ipc -> app_server_handle_json_lines` 返回 `runtime model route is not executable`；输入恢复，但无可见 toast。源码确认生产唯一 Toaster 挂载在发布改动中被删除。
- 2026-07-31：恢复 `RootRouter` 的 Toaster 挂载，覆盖主应用、Browser Runtime、Resource Manager 和开发 smoke 窗口；新增真实 Sonner DOM 回归。发送提交与流式提交失败统一使用 `resolveAgentRuntimeSubmitErrorMessage`，将不可执行路由隐藏为 Provider 暂不可用五语言文案。
- 2026-07-31：定向 Vitest 通过：`RootRouter.test.tsx` 4/4、`sonner.test.tsx` 1/1、`agentRuntimeErrorPresentation.test.ts` 14/14、`useWorkspaceSendActions.test.tsx` 155/155。
- 2026-07-31：修复 typed error fixture 的 DOM 作用域、`data-status`、Provider checkpoint、canonical read model 状态读取与 pre-terminal 等待合同；fixture guard 81/81 通过。
- 2026-07-31：typed error / runtime status 最终定向回归 93/93 通过；全工作树 `git diff --check` 通过，fixture 进程无残留。
- 2026-07-31：`typed-error-retry-success` 与 `typed-error-retry-failure` 真实 Electron controlled Gate B 均通过；两者均命中 Electron IPC、App Server current JSON-RPC、Provider/App Server trace 分离、同一 turn identity，console / invoke error 为零。
- 2026-07-31：`npm run smoke:agent-runtime-current-fixture` 完整通过；覆盖首页、历史恢复、停止继续、审批、active steer、Plan、Skills、MCP、媒体、代码工作台、typed error success/failure 与内容工厂，`liveProviderUsed=false`。
- 2026-07-31：`npm run verify:gui-smoke` 通过，21/21 assertions；evidence：`.lime/qc/project-gates/standalone-shell-01-20260731103342-78558/shell-01-electron-smoke/summary.json`。
- 2026-07-31：本轮 controlled 主流程完成。Windows packaged Squirrel 从 N-1 到新版本的发现、下载、重启安装与最终版本确认仍是唯一发布平台阻塞，完成度 92%。
