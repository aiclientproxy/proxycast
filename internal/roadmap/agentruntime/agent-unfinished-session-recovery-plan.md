# Agent 未完成会话自动恢复计划

> 状态：implemented-in-progress
> 日期：2026-07-07
> 主链：AgentRuntime current session / thread / turn read model
> 关联入口：[AgentRuntime Profile 路线图](./README.md)、[状态 / 历史 / 遥测主链](../../aiprompts/state-history-telemetry.md)、[Clawstream S1 P0 执行计划](../../exec-plans/clawstream-s1-p0-implementation-plan.md)

## 1. 背景

用户在 Claw 首页或侧栏看到一个未完成会话时，当前体验还不够产品化：

1. 意外退出或刷新后，用户回到首页时不知道后台是否仍在跑。
2. 侧栏能显示最近会话，但运行态、终态和输入框状态仍容易被局部状态误判。
3. 已修复的 `running / terminal` 投影仍需要上升为统一产品模型，避免后续在首页、侧栏、输入框、工作区继续各自判断。

这不是单个按钮或单个入口问题，而是 AgentRuntime current 主链里 `Session -> Thread -> Turn -> ReadModel -> GUI projection` 的恢复产品化问题。

## 2. Codex 参考口径

参考 `/Users/coso/Documents/dev/rust/codex` 时只采纳通用架构原则，不照搬文件结构：

- Codex 的 rollout/session 以可发现的持久事实源组织：`sessions`、`ThreadItem`、`ThreadsPage`、`recency_at / updated_at`、session meta 和 rollout JSONL。
- 恢复不是 UI 猜测“最近按钮”，而是从 session / thread / turn 事实源读取当前状态，再决定是否 resume / reconnect。
- running、interrupted、terminal 和 queued user input 都是一等状态，不能靠固定 timeout 合成完成态。

Lime 对应 current 主链必须继续走：

```text
React / Agent UI
  -> src/lib/api/agentRuntime/*
  -> AppServerClient / safeInvoke
  -> Electron IPC app_server_handle_json_lines
  -> App Server JSON-RPC agentSession/*
  -> RuntimeCore / backend / read model
```

禁止为了恢复体验新增 `agent_runtime_*` production truth、renderer mock fallback、App Server mock backend 或旧 Tauri wrapper。

## 3. 产品原则

### 3.1 默认体验

采用“自动恢复但留首页”：

1. App 启动、刷新、从后台恢复、回到首页时自动扫描未完成 Agent 会话。
2. 如果存在最近一个可恢复会话，后台自动 reconnect / resume。
3. 不强制跳转到会话详情；首页保留当前首屏，但显示轻量恢复卡。
4. 用户点击首页恢复卡或侧栏会话后进入详情观察流式输出。

不采用“启动后自动跳回会话”。强制跳转会让首页成为不稳定入口，也会在多个未完成任务时制造抢焦点问题。

### 3.2 状态事实源

状态只来自 App Server read model：

- 可恢复：`running`、`queued`、`waitingAction`
- 终态：`completed`、`failed`、`canceled`
- 终态必须 fail closed：即使存在残留 `active_turn_id`，也不能再判定为 running。

前端本地 `isSending / activeStream` 只能作为瞬时交互态，不允许成为重启后的持久事实源。

### 3.3 V1 范围

V1 只覆盖 Agent session：

- Claw 普通会话
- 首页首发后 materialized 的 Agent session
- 侧栏最近对话中的 current session
- queued turn 和 waiting action 的可见状态

V1 不覆盖：

- 图片 / 视频 / 媒体 task file 全生命周期
- 内容工厂 workflow 全生命周期
- right surface pending request 的统一异步收件箱
- live Provider 专项联网证明

这些后续可以挂到同一异步恢复模型，但不阻塞本轮未完成 Agent 会话体验。

## 4. 设计

### 4.1 未完成会话投影

新增一个纯投影层，输入为 `agentSession/list` 的 session overview 和必要时的 `agentSession/read`：

```text
session overview
  -> candidate filter
  -> read detail for active candidate
  -> unfinished session projection
  -> home card / sidebar status / inputbar state
```

投影输出至少包含：

- `sessionId`
- `title`
- `preview`
- `status`
- `latestTurnStatus`
- `activeTurnId`
- `updatedAt`
- `actionLabel`

候选排序使用 `updatedAt` 或 read model 等价 recency 字段；没有稳定时间时 fail closed，不自动恢复。

### 4.1.1 运行态事实源收敛

截至 2026-07-07，运行态不再允许由 list / read / evidence / GUI 各自猜测：

- App Server current 事实源为 `runtime/status.rs::SessionRuntimeState`。
- `read_model.rs`、`projection_store.rs`、`session_lifecycle.rs` 均消费同一个 `resolve_agent_session_runtime_state(...)` / `resolve_session_runtime_state(...)`。
- `runtime/exports/*` 也必须消费同一投影，避免 handoff / replay / analysis / review export 把 stale running 继续判成 `in_progress`。
- 前端 `threadReadActivity.ts`、`unfinishedSessionProjection.ts`、`agentStreamResumeBinding.ts`、`AgentChatWorkspace.tsx` 只消费 read model / overview 投影；`isSending / activeStream` 只能作为当前 renderer 交互态。

固定规则：

- terminal session fail closed：即使存在残留 `active_turn_id`，也清空 active / queued 投影。
- stale running 降级为 `idle`，但 `diagnostics.latest_turn_status` / `latest_turn_status` 保留原始 `running`，用于诊断而不是恢复。
- queued / waitingAction 仍是一等 active 状态。

### 4.2 自动恢复调度器

调度器只做三件事：

1. 扫描最近 Agent 会话。
2. 自动恢复最近一个可恢复会话。
3. 发布只读投影给首页和侧栏。

约束：

- 同一 `sessionId / activeTurnId` 只恢复一次，避免重复 `runtimeGetSession` 轮询。
- 最多自动恢复 1 个 active session；其余未完成会话只显示状态。
- 如果 read model 返回终态，立即清理本地 active stream。
- 如果恢复失败，记录诊断但不把首页卡死；侧栏显示失败或可重试状态。

### 4.2.1 前台 / 后台 presentation

本轮采用窄接口拆开两种恢复 presentation：

- `foreground`：默认行为，用于会话详情、URL 带 `initialSessionId`、用户显式点击最近会话或任务 tab。允许应用本地 / cached snapshot，允许 `switchTopic(...)` 绑定详情输出。
- `background`：仅用于新任务首页且无 `contentId / initialSessionId`。允许扫描会话和调用 `agentSession/read` 恢复后台运行态，但不写前台 `sessionId / messages / threadRead`，不自动 `switchTopic(...)`，只更新 topics / 最近会话 / 侧栏投影。

这把“恢复后台任务”和“打开会话详情”拆成两步：系统可以继续跑未完成任务，用户仍停留在首页；只有用户显式打开会话后，输入框按钮和流式输出才绑定到该 session。

### 4.3 首页与侧栏 UI

首页：

- 在输入框下方或推荐区上方显示轻量卡：`正在继续：{title}`。
- 卡片只提供进入会话，不直接暴露内部 `sessionId / turnId / JSON-RPC`。
- 如果恢复中的会话输出已完成，卡片变为“已完成，查看结果”或自动消失，具体由 read model 决定。

侧栏：

- 最近对话使用同一投影显示状态图标。
- `running / queued` 显示运行态；`waitingAction` 显示等待用户操作；终态不显示运行图标。
- 侧栏点击会话时只打开该 session，不重新创建 session。

输入框：

- 当前 active session 正在输出时，发送按钮显示“正在输出”态，停止按钮可用。
- 终态刷新后输入框立即恢复可输入。
- 首页首发 pending preview 只能作为临时视觉态，一旦 materialized session 可读，必须切到 read model 投影。

## 5. 实施切片

### P0：文档与事实源锁定

- 本文档作为 current roadmap 入口。
- `internal/roadmap/agentruntime/README.md` 增加导航。
- 不修改 reliability compat 文档作为 current 主入口。

### P1：纯投影与单测

- 已新增 `unfinishedSessionProjection` 和侧栏 / topics 映射。
- 已覆盖：
  - running / queued / waitingAction 被识别为未完成
  - completed / failed / canceled 不进入恢复队列
  - terminal + stale `active_turn_id` 不误判
  - 多个候选只选择最近一个自动恢复

### P2：首页恢复调度器

- 已在 Agent Chat workspace / session hook 层接入 `sessionRestorePresentation`。
- 新任务首页进入 `background` presentation，后台 `runtime.getSession(... source: "homeBackgroundRecovery")` 后只合入 topics projection，不强制跳转详情。
- `foreground` presentation 保留已有详情页、显式会话恢复、action_required、压缩和 runtime hydrate 行为。

### P3：侧栏与输入框状态统一

- 侧栏最近会话状态、任务列表 topic 状态和输入框 running 态已消费 App Server read model / overview 投影。
- 新任务首页旧 task tab fallback 已在 `home-background-recovery` 场景跳过，避免旧 tab 抢回详情。
- 已新增必要 i18n key，覆盖 `zh-CN / zh-TW / en-US / ja-JP / ko-KR`。
- 已把 App Server `read / list / session overview / evidence / handoff export` 的运行态收敛到共享 `SessionRuntimeState` projector，避免侧栏、主区、输入框、证据包继续分裂。

### P4：Electron / CDP 验证

- 已扩展 `reopen-running-turn` Electron fixture：运行中刷新 / 重启后回首页，自动恢复同一 session，且 background presentation 不抢回详情。
- 保留 `cancel-then-continue`、`inputbar-rich-restore` 和 terminal stale 回归。
- CDP Gate B 只保存 method、transport、status、sessionId、turnId 和必要 marker，不保存 prompt、token 或 provider secret。
- 本轮已用真实 Electron CDP 复跑 background reload / restart，证明 `window.__LIME_ELECTRON__`、`window.electronAPI.invoke`、`electron-ipc` trace 与用户可见状态同时成立。证据为 `.lime/cdp-evidence/reopen-running-turn-cdp-gate-background-reload-summary.json` 与 `.lime/cdp-evidence/reopen-running-turn-cdp-gate-background-restart-summary.json`。
- 2026-07-07 继续补齐首页恢复卡 Gate B：reload / restart 均使用 `--presentation-mode background` 验证首页 `home-unfinished-session-card[data-status=running]`、侧栏同 session `data-status=running`、点击恢复卡后进入同一详情并显示 inputbar 停止态，cancel 后侧栏 / inputbar 回 idle。证据为 `.lime/cdp-evidence/reopen-running-turn-cdp-gate-background-reload-home-card-summary.json` 与 `.lime/cdp-evidence/reopen-running-turn-cdp-gate-background-restart-home-card-summary.json`，两者 `completedGateB=true`。
- 2026-07-07 首页首发 Enter Gate B：真实 packaged Electron 通过 `chromium.connectOverCDP("http://127.0.0.1:9223")` 接入 `file://.../dist/index.html?nativeStartup=1`，断言 `window.__LIME_ELECTRON__`、`window.electronAPI.invoke` 与 `supportsCommand("app_server_handle_json_lines")` 成立；在 `home-start-surface` 的 `textarea[name="agent-chat-message"]` 输入后按 Enter，trace 捕获 `agentSession/turn/start`，`transport=electron-ipc`、`status=success`、`sessionId=sess_57db605682c940be8d4f50b20ce6bac6`、`turnId=02070273-0538-47a1-801c-dd544f9be1ab`，GUI 显示用户消息、assistant 摘要、message list，停止按钮消失且输入框恢复可用。证据为 `.lime/cdp-evidence/home-enter-claw-cdp-gate-summary.json` 与 `.lime/cdp-evidence/home-enter-claw-cdp-gate-screenshot.png`；`lime_invoke_error_buffer_v1` 为 0，console error 仅为 packaged fixture `net::ERR_FILE_NOT_FOUND` 资源噪音。该证据只证明首页首发 Enter -> Claw current turn 主路径，不证明 live Provider 质量、多会话并发或网络中断恢复。
- 2026-07-08 继续验证：`NODE_OPTIONS="--max-old-space-size=8192" npx tsc --noEmit --project "tsconfig.renderer.json" --pretty false` 通过；`npx vitest run "src/components/agent/chat/hooks/agentChatShared.test.ts" "src/components/agent/chat/hooks/agentSessionTopicViewModel.unit.test.ts" "src/components/agent/chat/hooks/agentSessionState.test.ts" "src/components/agent/chat/home/HomeStartSurface.test.tsx" "src/components/AppSidebar.conversations.test.tsx" "src/components/agent/chat/components/TaskCenterTabStrip.test.tsx"` 6 files / 134 tests 通过；`npx vitest run "src/components/agent/chat/AgentChatWorkspace.inputRestoreGuard.test.ts"` 1 test 通过；`npm run test:contracts` 完整通过。该补充把 CDP Gate B 结果提升为 renderer 类型图、状态投影回归和 App Server client/command contract 同步通过。
- 2026-07-08 queued CDP 补充：`claw-chat-current-fixture-smoke.mjs` 增加可选 `--cdp-port`，默认行为不变；传入端口时通过 `chromium.connectOverCDP(...)` 接入真实 Electron renderer 后继续执行原 current fixture。已执行 `npm run smoke:claw-chat-current-fixture -- --scenario inputbar-pending-steer-multi-queue --timeout-ms 240000 --cdp-port 9261 --prefix claw-chat-current-fixture-inputbar-pending-steer-multi-queue-cdp --evidence-dir ".lime/qc/gui-evidence/claw-chat-current-fixture"`，summary 为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-inputbar-pending-steer-multi-queue-cdp-summary.json`，截图为同目录 `*-chat.png`。关键结果：proofLevel=`Gate B CDP controlled fixture`，CDP page 为 packaged `file://.../dist/index.html?nativeStartup=1`；read model 中两个 queued turn 均为 `status=queued`，位置 `0 / 1`，`orderPreserved=true`；scenario assertions 包含 active prompt 到达 backend、active output visible、rich input deferred、multiple queued、queue order preserved、second text queued、rich prompt 未在 cancel 前启动；common assertions 包含 Electron preload bridge、App Server JSON-RPC、current `agentSession/*` methods、external fixture backend、no invoke errors、no actionable console errors。该证据关闭 queued 多队列 read model 与 GUI running/inputbar 状态的 CDP Gate B 缺口；不证明 Electron/App Server 进程级 restart、live Provider 或网络中断恢复。
- 2026-07-08 queued pop-front resume CDP 补充：已执行 `npm run smoke:claw-chat-current-fixture -- --scenario inputbar-pending-steer-pop-front-resume --timeout-ms 240000 --cdp-port 9262 --prefix claw-chat-current-fixture-inputbar-pending-steer-pop-front-resume-cdp --evidence-dir ".lime/qc/gui-evidence/claw-chat-current-fixture"`，summary 为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-inputbar-pending-steer-pop-front-resume-cdp-summary.json`，截图为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-inputbar-pending-steer-pop-front-resume-cdp-chat.png`。关键结果：proofLevel=`Gate B CDP controlled fixture`，CDP page 为 packaged `file://.../dist/index.html?nativeStartup=1`；current methods 覆盖 `agentSession/queuedTurn/promote`、`agentSession/turn/cancel`、`agentSession/thread/resume`、`agentSession/read`；scenario assertions 证明 GUI promote 已点击、active turn 走 current cancel 且 backend 收到 cancel、rich queued turn 通过 current resume 启动并从队列移除、第二个 queued turn reindex 到 position `0`、renderer reload 后队列面板水合第二条 queued turn，且 textarea 可用；common assertions 继续证明 Electron preload bridge、App Server JSON-RPC、external fixture backend、无 invoke errors、无可行动 console errors。该证据关闭 queued pop-front resume / 剩余队列 hydrate / inputbar ready 的 CDP Gate B 缺口；不声明 live Provider、网络中断、多会话并发或 Electron/App Server 进程级 restart。
- 2026-07-08 cancel-then-continue CDP 补充：已执行 `npm run smoke:claw-chat-current-fixture -- --scenario cancel-then-continue --timeout-ms 240000 --cdp-port 9263 --prefix claw-chat-current-fixture-cancel-then-continue-cdp --evidence-dir ".lime/qc/gui-evidence/claw-chat-current-fixture"`，summary 为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-cancel-then-continue-cdp-summary.json`，截图为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-cancel-then-continue-cdp-chat.png`。关键结果：proofLevel=`Gate B CDP controlled fixture`，CDP page 为 packaged `file://.../dist/index.html?nativeStartup=1`；scenario assertions 证明 current `agentSession/turn/cancel` 被调用、external backend 收到 `turnCancel`、read model 进入 `canceled`、GUI 停止后输入框恢复可用；随后同一 session 从 GUI 输入 `继续输出`，backend ledger 记录第二个 `turnStart`，GUI 与 read model 均完成第二轮，且无 invoke errors / 无可行动 console errors。该证据关闭“停止后无法继续输出 / 左侧仍转但主区已停”的同一 session CDP Gate B 回归缺口；仍不声明 live Provider、多会话并发或网络中断。
- 2026-07-08 模块化后 CDP 复跑：拆分 `claw-chat-current-fixture-*` 后，先用 `--cdp-port 9264` 做 packaged full rebuild 复跑；该次在进入 Electron 场景前被 smoke timeout 终止，日志显示 `electron:build:app-server-assets Terminated: 15` / exit code `143`，归类为外部 Cargo/package cache 冷构建与锁等待耗尽 `240000ms`，不是 GUI 断言失败，也没有生成 Gate B 反证。随后复用已构建产物执行 `LIME_ELECTRON_FIXTURE_BUILD_READY=1 npm run smoke:claw-chat-current-fixture -- --scenario inputbar-pending-steer-pop-front-resume --timeout-ms 240000 --cdp-port 9265 --prefix claw-chat-current-fixture-inputbar-pending-steer-pop-front-resume-cdp-rerun-skipbuild --evidence-dir ".lime/qc/gui-evidence/claw-chat-current-fixture"`，summary 为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-inputbar-pending-steer-pop-front-resume-cdp-rerun-skipbuild-summary.json`，截图和 backend ledger 为同名前缀文件。关键结果：`ok=true`、proofLevel=`Gate B CDP controlled fixture`，CDP endpoint UA 为 `Lime/1.94.0 ... Electron/42.3.3`，page 为 packaged `file://.../dist/index.html?nativeStartup=1`；43 个断言全 true，无 false assertion，`actionableConsoleErrors=[]`，继续证明 Electron preload bridge、App Server JSON-RPC、current `agentSession/*` methods、queued promote/cancel/resume/read、队列 reindex、reload 后队列 hydrate 和 inputbar ready。该复跑证明 fixture 模块化没有破坏 CDP Gate B 主链；仍不把 skipbuild 产物声明为 clean full rebuild，也不声明 live Provider、网络中断、多会话并发或 Electron/App Server 进程级 restart。
- 2026-07-08 停止后继续输出 CDP 复跑：为防止模块化后“主区停了但侧栏还转 / 输入框不可继续”回归，复用已构建产物执行 `LIME_ELECTRON_FIXTURE_BUILD_READY=1 npm run smoke:claw-chat-current-fixture -- --scenario cancel-then-continue --timeout-ms 240000 --cdp-port 9266 --prefix claw-chat-current-fixture-cancel-then-continue-cdp-rerun-skipbuild --evidence-dir ".lime/qc/gui-evidence/claw-chat-current-fixture"`，summary 为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-cancel-then-continue-cdp-rerun-skipbuild-summary.json`。关键结果：`ok=true`、proofLevel=`Gate B CDP controlled fixture`，CDP endpoint UA 为 `Lime/1.94.0 ... Electron/42.3.3`，page 为 packaged `file://.../dist/index.html?nativeStartup=1`；断言包含 Electron preload bridge、App Server JSON-RPC、current `agentSession/turn/start`、current `agentSession/turn/cancel`、external backend 收到 cancel、read model canceled、GUI stop clicked、GUI 同 session 提交 `继续输出`、backend 记录第二个 turn、GUI/read model 均 completed、inputbar ready、not stuck streaming、no invoke errors、no actionable console errors，且无 false assertion。该复跑把“停止后继续输出”作为模块化后的最新 CDP Gate B 证据；仍不声明 live Provider 或进程级重启恢复。
- 2026-07-08 多未完成会话 CDP 骨架：`reopen-running-turn-cdp-gate.mjs` 新增可选 `--multi-running-sessions`，默认不影响原 Gate；开启时在主会话 running 前额外创建第二个 App Server current session，并通过 `agentSession/turn/start` + external backend 让第二会话保持 running。已执行 `LIME_ELECTRON_FIXTURE_BUILD_READY=1 npm run smoke:reopen-running-turn-cdp-gate -- --presentation-mode background --reopen-mode reload --multi-running-sessions --timeout-ms 240000 --cdp-port 9267 --prefix reopen-running-turn-cdp-gate-background-reload-multi-running --evidence-dir ".lime/cdp-evidence"`，summary 为 `.lime/cdp-evidence/reopen-running-turn-cdp-gate-background-reload-multi-running-summary.json`，trace summary 为同名前缀 `*-trace-summary.json`。关键结果：`ok=true`、`completedGateB=true`、proofLevel=`Gate B controlled fixture`，CDP endpoint UA 为 `Lime/1.94.0 ... Electron/42.3.3`，page 为 packaged `file://.../dist/index.html?nativeStartup=1`；assertions 全 true 且无 false assertion，包含 `multiRunningSecondaryStarted`、`multiRunningPrimaryAndSecondarySidebarBeforeReopen`、`multiRunningPrimaryAndSecondarySidebarAfterReopen`、`multiRunningHomeKeepsPrimaryRecoveryCard`、`multiRunningSecondaryStillRunningAfterPrimaryCancel`、`multiRunningSecondaryCleanupCanceled`、`homeBackgroundBeforeReopen`、`homeBackgroundAfterReopen`、`threadResumeSeen`、`turnCancelSeen`。该证据证明多 running session 时首页恢复卡仍绑定主会话、不强制跳转详情，侧栏两条会话均显示 running，主会话停止收口后第二会话仍独立 running，随后第二会话显式 cancel 并进入 read model canceled；仍不声明 live Provider、网络中断、多窗口并发或进程级 restart。
- 2026-07-08 首页首发 dev 回归修复：`handleNonMaterializedTaskCenterSessionReady(...)` 在真实 session materialized 后同时执行 `markTaskCenterLocalSessionOverride(readySessionId)` 与 `persistTaskCenterMaterializedSessionNavigation(readySessionId)`，避免首页 direct dispatch 只更新本地 task tab 而不把 URL / `initialSessionId` 同步到 Claw 详情。`scripts/electron/run-dev.mjs` 在 initial 和 app-server rebuild 后启动 Electron 前统一等待 renderer `http://127.0.0.1:1420` ready，避免 Electron 先打开 `chrome-error://chromewebdata/`；`scripts/lib/electron-dev-sidecar.mjs` 的 watcher 改为按 path segment 忽略 `target / node_modules / dist-electron`，避免 Cargo `target/debug/build/...rs` 反复触发 sidecar rebuild。真实 dev Electron CDP 已执行 `LIME_ELECTRON_REMOTE_DEBUGGING_PORT=9269 npm run electron:dev` 后连接 `http://127.0.0.1:9269`，summary 为 `.lime/cdp-evidence/home-enter-claw-after-fix-cdp-clean-summary.json`，截图为 `.lime/cdp-evidence/home-enter-claw-after-fix-cdp-clean.png`；关键结果：`ok=true`、proofLevel=`Gate B real Electron CDP dev`、`runtime.electron=true`、`runtime.hasInvoke=true`、`runtime.isChromeError=false`、首页发送前 `homeVisible=true / textareaSessionId=null / textareaDisabled=false`，发送后 `homeVisible=false / textareaSessionId=sess_6a2a63ef57c34084914fdf0e6f892845 / textareaDisabled=false / bodyTextIncludesNihao=true`，trace 捕获 `agentSession/turn/start`、`transport=electron-ipc`、`status=success`，且 `invokeErrors=[]`。contract retarget 后又用 external fixture backend 复跑 dev Electron CDP，summary 为 `.lime/cdp-evidence/home-enter-claw-after-current-contract-fix-cdp-final-summary.json`，截图为 `.lime/cdp-evidence/home-enter-claw-after-current-contract-fix-cdp-final.png`；关键结果：`ok=true`、proofLevel=`Gate B real Electron CDP dev external fixture`、发送前仍为首页无 session，发送后 `homeVisible=false`、`textareaSessionId=sess_2719200a9a734e73842f549e8c4a70bf`、`textareaDisabled=false`、用户消息与 assistant 输出均可见、停止按钮消失、`invokeErrorsCount=0`，trace 捕获 current `agentSession/turn/start` 且 `transport=electron-ipc / status=success`。该证据关闭“首页输入回车没有进入 Claw 对话页”的 dev 主路径回归；不声明 live Provider、多窗口并发、网络中断或进程级 external backend continuation。

## 6. 验收标准

V1 完成必须同时满足：

1. 回到首页后，最近一个未完成 Agent session 自动恢复，但页面不强制跳走。
2. 侧栏对应会话显示运行态；终态后运行图标消失。
3. 输入框按钮在输出中显示正在输出；终态后恢复可输入。
4. stale terminal / failed / canceled session 不再触发无限 `runtimeGetSession` 轮询。
5. 停止后同会话继续输出仍能完成。
6. 生产路径不出现 renderer mock fallback、App Server mock backend 或 legacy runtime command。

## 7. 验证入口

定向测试优先：

```bash
npm test -- --run src/components/agent/chat/hooks/agentSessionState.test.ts src/components/agent/chat/hooks/agentStreamResumeBinding.test.ts src/components/agent/chat/hooks/useAgentRuntimeSyncEffects.test.tsx
npm test -- --run src/components/agent/chat/workspace/workspaceSceneSessionProjection.unit.test.ts
```

Agent Runtime 聚合门禁：

```bash
npm run smoke:agent-runtime-current-fixture
```

GUI 主路径门禁：

```bash
npm run verify:gui-smoke
```

需要证明真实 Electron CDP 时，证据等级必须标为 Gate B，并证明：

- `window.__LIME_ELECTRON__ === true`
- `window.electronAPI.invoke` 存在
- trace 中存在 `transport: "electron-ipc"`
- JSON-RPC method 包含 `agentSession/read` 和对应恢复动作

CDP 骨架入口：

```bash
node scripts/agent-runtime/agent-session-recovery-cdp-gate.mjs --cdp-port 9239 --timeout-ms 180000 --prefix agent-session-recovery-cdp-gate-skeleton
npm run smoke:reopen-running-turn-cdp-gate -- --cdp-port 9239 --timeout-ms 180000 --prefix reopen-running-turn-cdp-gate-skeleton
```

`agent-session-recovery-cdp-gate` 只证明真实 Electron CDP / preload / `app_server_handle_json_lines` / `agentSession/list|read` 壳层，默认 `APP_SERVER_BACKEND_MODE=unavailable`，不触发 `agentSession/turn/start`。完整 Gate B 必须新增或扩展 `reopen-running-turn` 场景，使用 `APP_SERVER_BACKEND_MODE=external`，并证明 reload/reopen 后同一 `sessionId / turnId` 仍处于 running UI 与 event resume 链路。

`reopen-running-turn-cdp-gate` 当前是 skeleton 聚合入口：先跑 CDP shell gate，再跑 `cancel-then-continue` current fixture，并在 summary 中标记 `completedGateB=false`。它用于防止后续继续口头描述 Gate B，但仍不能替代真正同一 renderer reload/reopen 的完整证明。

## 8. 当前证据

截至 2026-07-07，已有相关证据：

- `.lime/cdp-evidence/reopen-running-turn-cdp-summary.json`：历史证据，只能作为参考；当前仍需用新的 `reopen-running-turn` Gate B 复跑并确认 `agentSession/turn/start`、`agentSession/thread/resume`、reload/reopen 后的同一 `turnId` 事件续接和输入框 running 态。
- `.lime/cdp-evidence/agent-terminal-rich-restore-cdp-summary.json`：证明 direct preload IPC 到 `app_server_handle_json_lines`、App Server current method 与 read model 链路可用。
- `.lime/cdp-evidence/agent-session-recovery-cdp-gate-skeleton-summary.json`：证明真实 Electron CDP attach 到 packaged `file://.../dist/index.html?nativeStartup=1`，`window.__LIME_ELECTRON__`、`window.electronAPI.invoke`、`supportsCommand("app_server_handle_json_lines")` 均成立；renderer `safeInvoke` trace 包含 `transport: "electron-ipc"`、`app_server_handle_json_lines`、`agentSession/list`、`agentSession/read`，并从侧栏打开同一 `sessionId`。该证据使用 `APP_SERVER_BACKEND_MODE=unavailable`，不触发 `agentSession/turn/start`，不证明 live Provider 或运行中 turn 输出。
- `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-cancel-then-continue-regression-summary.json`：证明停止后同会话继续输出。
- `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-inputbar-rich-restore-regression-summary.json`：证明 output-free cancel 后 inputbar 富草稿恢复。

这些证据证明底层 current 链路可用，但还不等于首页级产品体验已经完成；本文目标就是把这些底层能力收敛成统一首页 / 侧栏 / 输入框恢复体验。

本轮新增 / 复跑的代码级证据：

- `npx vitest run "src/components/agent/chat/projection/unfinishedSessionProjection.test.ts" "src/components/agent/chat/hooks/agentChatShared.test.ts" "src/components/app-sidebar/AppSidebarConversationRow.test.tsx" "src/components/agent/chat/utils/taskCenterTabs.test.ts"`：通过，64 tests。
- `npx vitest run "src/components/agent/chat/hooks/useAgentChat.test.tsx" -t "首页后台恢复运行候选"`：通过，1 focused test，证明 `background` presentation 只触发 `homeBackgroundRecovery`，不写前台 `sessionId / messages / threadRead`。
- `npx vitest run "src/components/agent/chat/hooks/useAgentChat.test.tsx"`：通过，185 tests；新增 focused test 后该文件为 186 tests，需在最终验证中复跑完整文件。
- `npx vitest run "scripts/agent-runtime/agent-session-recovery-cdp-gate.test.mjs"`：通过，3 tests，守住 CDP Gate B 骨架必须使用真实 Electron、`chromium.connectOverCDP`、`app_server_handle_json_lines`、`agentSession/list/read`，且不启用 live Provider 或 mock backend。
- `node "scripts/agent-runtime/agent-session-recovery-cdp-gate.mjs" --cdp-port 9239 --timeout-ms 180000 --prefix agent-session-recovery-cdp-gate-skeleton`：通过，Gate B skeleton；见 `.lime/cdp-evidence/agent-session-recovery-cdp-gate-skeleton-summary.json`。
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server session_list_projection --lib`：通过，8 tests；证明 list overview 对 running / stale running / terminal 的投影同源。
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server thread_read --lib`：通过，5 tests；证明 read model stale running 降级为 `idle`，但保留 `diagnostics.latest_turn_status=running`。
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server basic_evidence_pack_downgrades_stale_running_turn --lib`：通过，1 test；证明 evidence pack 不再把 stale running 判成 `in_progress`。
- `npx vitest run "scripts/agent-runtime/reopen-running-turn-cdp-gate.test.mjs"`：通过，3 tests；守住 `reopen-running-turn` skeleton 入口、claim boundary、current runtime signals 和 mock fallback 禁止项。
- `npm run smoke:reopen-running-turn-cdp-gate -- --cdp-port 9239 --timeout-ms 180000 --prefix reopen-running-turn-cdp-gate-systematic`：通过，生成 `.lime/cdp-evidence/reopen-running-turn-cdp-gate-systematic-summary.json`。该 summary 明确 `status=passed_skeleton`、`proofLevel=Gate B skeleton`、`completedGateB=false`，只证明 CDP shell gate 与 `cancel-then-continue` current fixture 分别通过，不证明 reload/reopen 后同一 running `turnId` 续接。

剩余 Gate B 细节边界：

- running turn / queued turn 的 controlled fixture CDP Gate B 已有；后续只在触碰恢复链路时复跑，避免退化。
- 仍未声明完成：Electron/App Server 进程级 restart 后 queued turn 自动继续、live Provider 长输出、多 running sessions、网络中断恢复矩阵。
- 如后续改动触碰终态收口，仍需复核 terminal 后运行图标消失、输入框恢复可输入、不再触发 stale `runtimeGetSession` 轮询。

## 9. 后续扩展

V1 完成后，再考虑把下列异步任务接入同一模型：

- media task artifact：图片、视频、封面、转写、链接解析
- content factory workflow：文章、PPT、表单等产物型 workflow
- right surface pending：对象画布、浏览器、工作台 pending request
- managed objective：长目标自动继续与人工暂停 / 恢复

扩展前必须先回答：

1. 该任务是否有 current owner read model。
2. 是否能稳定提供 `taskId / sessionId / turnId / updatedAt / status`。
3. GUI 是否只读投影，而不是自己猜生命周期。
4. 是否有 Electron fixture 或 CDP Gate B 证据。
