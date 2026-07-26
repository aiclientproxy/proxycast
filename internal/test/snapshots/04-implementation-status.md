# P0 实现与证据状态

日期：2026-07-26

本表只记录精确场景证据。`npm run verify:gui-smoke` 或 current Agent fixture 整体通过，不能替代
某个场景要求的故障注入、DOM geometry 或 public JSON-RPC 断言。

覆盖面：本表覆盖 [01-frontend-test-plan.md](01-frontend-test-plan.md) 的全部 `40` 个 GUI 场景，
以及 [02-runtime-contract-test-plan.md](02-runtime-contract-test-plan.md) 的 `4` 个多模型 contract
与 `6` 个在 `03-source-to-scenario-map.md` 中独立成列的 `runtime-*` contract。

02 余下 `8` 个 `runtime-*` ID 在 source map 中已并入对应 GUI 场景，不单独建行，避免同一证据被
计两次：`runtime-approval` -> `approval-four-states`；`runtime-tool-output` -> `tool-running-terminal`；
`runtime-history-fork` -> `history-fork-lineage`；`runtime-mcp-elicitation` -> `mcp-inventory-elicitation`；
`runtime-multi-agent` -> `multi-agent-roster`；`runtime-provider-lowering` -> `provider-circuit-breaker`
与 `provider-readiness`；`runtime-safety-errors` -> `error-safety-sandbox`；
`runtime-diagnostics` -> `error-mcp-hook`。

状态定义：

- `covered`：已有最贴 owner 的确定性测试。
- `partial`：已有相邻能力或部分状态，但尚未完整表达场景语义。
- `gate-missing`：owner 测试已存在，目标 Gate A/B 精确证据尚缺。
- `owner-blocked`：current protocol/read model 无法表达该状态，不能在测试层造字段。
- `owner-unwired`：owner 代码存在且有单元测试，但没有任何 current 主链调用点。不算 `partial`，
  因为没有可被 GUI 观察的行为；必须先接线才谈 Gate。

## Agent 主路径

| ID                               | Owner 状态 | 现有 evidence                                                                                                      | 精确 Gate                          | 下一刀                                                                                                      |
| -------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `turn-stream-complete`           | covered    | `packages/agent-runtime-projection/tests/clawstreamP0.test.mjs`、current Agent fixture                             | Gate B covered                     | 保持 `turn.completed` 单终态守卫                                                                            |
| `turn-stream-repair`             | covered    | projection/adapter 回归；`claw-chat-current-fixture-turn-stream-repair-summary.json`                               | Gate B covered                     | 保持丢中段 delta、canonical `item.completed` 修复、单一终态和 identity 精确断言                             |
| `turn-interrupt-resume`          | covered    | owner 回归；`claw-chat-current-fixture-turn-interrupt-resume-summary.json`                                         | Gate B covered                     | 保持 output-free interrupt、rich draft 完整恢复、同一 turn identity 和无伪 assistant 输出断言               |
| `turn-budget-limit`              | partial    | `goalLifecycleHydrate.test.mjs`、budget projection tests                                                           | Gate A/B missing                   | 补 typed budget terminal 到 composer/status 的同源断言                                                      |
| `tool-running-terminal`          | covered    | `tool-runtime/src/execution_process/tests.rs`、`ToolCallDisplay.commandOutput.test.tsx`                            | Gate B needs exact rerun           | 复跑 start/output/interrupt/terminal current fixture                                                        |
| `tool-unified-exec-wait`         | covered    | `tool-runtime::unified_exec` typed observation/streak tests、`clawstreamP0.test.mjs` wait + final message 顺序回归 | Gate B missing                     | Electron fixture 运行 active process，连续空 poll 后注入输出并完成，核对同一 tool/session identity 与可见态 |
| `approval-four-states`           | covered    | `permissionEvents.test.mjs`、App Server action projection tests                                                    | Gate B covered by approval fixture | 补齐各 decision 的稳定 scenario id                                                                          |
| `approval-parallel-aggregate`    | covered    | `clawstreamP0.test.mjs`：一项 resolved 后仍 waiting，另一项 expired 后收敛                                         | Gate B missing                     | external fixture 同时发出两个 request，按不同终态 settle                                                    |
| `history-replay-isomorphic`      | covered    | `historyReplayVisual.test.mjs`、`threadResumeRunningStream.test.mjs`                                               | Gate B needs exact rerun           | 复跑 cold resume，核对 active MCP 不被改写 completed                                                        |
| `history-fork-lineage`           | covered    | `threadForkLineage.test.mjs`                                                                                       | Gate B missing                     | public JSON-RPC fork/resume 后核对 parent/child 与 Item prefix                                              |
| `history-compaction-replacement` | covered    | `contextCompactionItem.test.mjs`                                                                                   | Gate B missing                     | manual/auto/mid-turn 各补一条 current fixture replacement lineage                                           |

## 输入和布局

| ID                       | Owner 状态 | 现有 evidence                                                                                                                        | 精确 Gate                                                | 下一刀                                                                                                                             |
| ------------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `composer-submit`        | covered    | `agentStreamUserInputSubmission.test.ts`、current Agent fixture                                                                      | Gate B covered                                           | 保持 optimistic user item 与 canonical identity 收敛                                                                               |
| `composer-queue-steer`   | covered    | submit/adapter/thread client steer tests、queued-turn current boundary、dead-surface contract guard、`inputbar-active-steer` fixture | exact Gate B covered                                     | 保持同一 Turn steer、provider second step、无第二次 `turn/start` 与无 public queue/PendingTurn GUI 断言                            |
| `composer-mention-slash` | covered    | `skillMentionInputRestore.test.mjs` 及 composer mention tests                                                                        | Gate A missing                                           | 固定 locale/viewport 验证弹窗和歧义选择                                                                                            |
| `composer-parent-owned`  | covered    | App Server projection/public JSON-RPC direct-input policy、canonical Thread adapter、workspace/Inputbar component tests              | Gate A + exact Electron Gate B missing (fixture blocker) | 冷重启 fixture 显式传入 `fixture.provider.modelPreference` 后创建 spawned child，断言 Thread fact、禁用态、服务端拒绝和无 IPC 下沉 |
| `layout-resize-reflow`   | covered    | `LayoutTransition.test.tsx`、`claw-chat-current-fixture-layout-narrow-overflow-summary.json`                                         | Gate B covered                                           | 保持 1280x820 -> 880x720 -> 1280x820 的页头、模式条、scroll anchor、Files surface 与无重叠断言                                     |
| `layout-narrow-overflow` | partial    | Markdown 横向滚动、toolbar wrap、inline input 防裁切 component tests；resize Gate B 已覆盖表格、页头和输入栏                         | resize subset Gate B covered                             | 固定最长文案/路径/审批按钮，补窄宽截图与无重叠 geometry                                                                            |
| `layout-markdown-rich`   | covered    | `MarkdownRenderer.normalization.test.tsx` 与 code block tests                                                                        | Gate A missing                                           | 补 CJK、表格、代码、file path/line 的稳定页面证据                                                                                  |

## 当前 blocker 边界

1. `composer-parent-owned` 的 owner blocker 已解除：App Server 从 durable canonical spawn lineage
   投影 `canAcceptDirectInput`，同一 policy 同时约束 `turn/start`、`turn/steer`、compact、settings、
   memory mode 和 shell command；Renderer 不从 `parentThreadId` 猜测。当前 cold restart Gate B fixture
   在创建 thread 前报 `thread/start requires a non-empty model`；测试夹具必须显式传入
   `fixture.provider.modelPreference`，不得恢复生产隐式 model fallback。
2. `tool-unified-exec-wait` 的 owner blocker 已解除：existing drain/status 是唯一 process 事实源，
   `tool-runtime::unified_exec` 现在投影 `output | waiting | terminal` observation 和连续空 poll 计数；
   后续输出清零 streak，terminal 与最后输出共存，未知 session fail closed。剩余缺口仅是精确 Gate B。
3. `layout-resize-reflow` 已有真实 Electron Gate B：compact 页头与模式条无相交，三段 viewport
   `distanceToBottom=0`，表格尾行、完成标记和 turn group 均未被输入栏遮挡，Files surface 在 compact
   隐藏时仍保留 `activeSurface=files`。`layout-narrow-overflow` 仍需补最长文案、路径和审批按钮矩阵，
   不因这条 resize 子集证据升级为完整覆盖。
4. `composer-queue-steer` 的精确 Gate B 已完成：Renderer 从 canonical `thread/read` 取得 active Turn，
   通过真实 textarea Enter 提交 `turn/steer(expectedTurnId)` 并保持同一 Turn identity。Codex TUI
   `ChatWidget::input_queue` 的 message queue/pending preview 是本地 UI scheduler，不属于 Codex v2
   App Server public surface；Lime 不复制该 scheduler，相关 snapshot 为 `defer / TUI-only`。public
   promote/remove、Renderer detailed queued snapshot 与 pending-steer fixture 已物理删除并禁止回流。

## Queue / Steer 分类

- `current`：`thread/read -> turn/steer(expectedTurnId)`；active steer 保持同一 Turn identity。
- `current`：RuntimeCore/session-loop 内部 FIFO、durable recovery、`queued_turn_count` 与 evidence。
- `dead / deleted / forbidden-to-restore`：public v0 `agentSession/queuedTurn/promote|remove`、v2
  `turn/queue/promote`、Renderer `QueuedTurnSnapshot/queued_turns`、PendingTurn GUI 与三个
  `inputbar-pending-steer-*` Electron 场景。
- `compat` / `deprecated`：无；仓库没有外部用户，不保留双轨或退出期包装层。

## Queue / Steer Gate B 证据

- 场景：`inputbar-active-steer`；三次独立运行均为 `Gate B controlled fixture`，真实经过 Electron、
  preload/IPC、`app_server_handle_json_lines` 与 App Server current JSON-RPC。
- current 调用顺序为 `thread/read -> turn/steer(expectedTurnId)`；`turn/steer` transport 为
  `electron-ipc`，expected Turn id 与 active Turn id 相同。
- 三次运行均为 initial `turn/start = 1`、post-baseline `turn/start = 0`、public queue method hits
  `= 0`、provider chat requests `= 2`；GUI 与 read model 各自只有一个 unique Turn id，且都等于
  active Turn id。10/10 active-steer assertions 全部为 true。
- Gate B 暴露 optimistic user message 可能永久保留 `pending-turn:<uuid>` 的真实缺陷。根因是
  canonical `turn_started`/`item_started` 只绑定 assistant identity；现由
  `bindSubmissionMessagesToRuntimeTurn` 按精确 `pendingTurnKey` 同时收敛 user 与 assistant identity，
  并覆盖缺失 `turn_started`、首个 canonical item 到达的回归。
- 证据：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-active-steer-run-1-summary.json`、
  `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-active-steer-run-2-summary.json`、
  `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-active-steer-run-3-summary.json`。
- 第三次运行基于当前完整源码 fresh build：Renderer `7091` modules、Electron host/preload 与 App
  Server sidecar 均成功构建；`verify:gui-smoke` 21/21 assertions 通过，evidence run id 为
  `standalone-shell-01-20260726061317-51571`。先前 Guardian 并行编译漂移已自行收敛，本轮未修改该热区。
- 当前聚合 Agent fixture 仍在无关 Right Surface visual matrix source-string guard 停止（89/90）；
  active-steer 四项 guard、场景过滤 3/3 与 owner 定向测试均通过。

## Resize Gate B 证据

- 命令：`node "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs" --scenario
"electron-resize-reflow" --prefix "claw-chat-current-fixture-layout-narrow-overflow" --timeout-ms 60000`。
- 证据等级：`Gate B controlled fixture`。真实 Electron/preload/IPC 命中
  `app_server_handle_json_lines`，App Server current methods 包含 `thread/start`、`turn/start`、
  `thread/read` 与 `workspaceRightSurface/request`；thread/turn/item identity 一致，最终状态为
  `turn.completed`，mock fallback、invoke error、console error、page error 均为 0。
- 固定 viewport：`1280x820 -> 880x720 -> 1280x820`。compact 页头为 `y=8..60`，模式条为
  `y=60..102`；三段 `distanceToBottom=0`，表格只渲染一次且首尾行各出现一次，document 无横向溢出。
- 截图已人工复核：wide/compact/restored 的页头、表格尾行、完成标记、输入栏和 Files surface
  可见状态与结构化 geometry 一致。

## Interrupt Gate B 证据

- 命令：`node "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs" --scenario
"inputbar-rich-restore" --prefix "claw-chat-current-fixture-turn-interrupt-resume" --timeout-ms 180000`。
- 证据等级：`Gate B controlled fixture`。真实 Electron/preload/IPC 命中 current `turn/start` 与
  `turn/interrupt`，同一 thread/turn/item identity 一致；backend 终态为 `turn.canceled`，read model
  与 GUI 状态为 `interrupted`。
- output-free interrupt 后文本、图片、路径和 skill 草稿全部恢复，停止按钮消失，输入框重新可提交，
  页面没有伪 assistant 正文；mock fallback、legacy command、invoke error、console error、page error
  均为 0。

## 多模型状态

多模型控制面以 `grok-build@6e386420825bd44ae648c63e7c8cba12fcec9401` 为 oracle。Codex
`model-picker` snapshot 只提供弹窗、列表、键盘和布局场景；不能裁决 provider/model 的业务语义。

| ID                         | Owner 状态 | 现有 evidence                                                                                                                                                                                                                                   | 精确 Gate                                    | 下一刀                                                                                        |
| -------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `model-effort-remap`       | covered    | `modelReasoningPolicy.test.ts`、`ModelSelector.test.tsx`、`modelRegistry.test.ts`；Rust catalog conversion 与 App Server `ModelInfo` projection tests                                                                                           | Gate A/B missing                             | 固定 viewport 展示结构化菜单；Electron trace 断言 `deep` 只显示、wire 只发送 `xhigh`          |
| `model-switch-atomic`      | covered    | Renderer hook/adapter/thread client：同一 UI 事件合并 model/provider/effort，只调用 `thread/settings/update`，成功后提交 UI，失败保留原状态；App Server public JSON-RPC 证明 active turn 保留旧设置、后续 turn 使用新设置且 restart/resume 一致 | exact Gate B missing                         | Electron fixture 捕获一次 `thread/settings/update`，再注入 typed failure 验证 UI rollback     |
| `provider-readiness`       | covered    | `runtime_backend::model_route_resolver`：capability snapshot 缺失、未知模型、disabled provider、capability gap 均 fail closed；ready fallback payload 有结构化事实                                                                              | Gate B/live evidence missing                 | 用 current catalog 选择未就绪 route，断言 provider call 为零且 GUI 展示 typed readiness error |
| `provider-circuit-breaker` | covered    | `model-provider::current_client::health/transport`：closed/open/half-open、单 probe、route key 隔离、bounded window、最多 5 次 stream request、Retry-After cap 与 jitter backoff                                                                | live/provider observability evidence missing | 补结构化 health/retry evidence，证明用户锁定模型不会被静默 fallback 改写                      |

`model-switch-atomic` 的 Renderer owner blocker 已解除：Chat setter 在同一事件循环内合并目标
provider/model/effort，current adapter 从 canonical session detail 解析 `threadId`，只调用一次
`thread/settings/update`；服务端成功后才更新 UI、workspace/session preference 与 synced fact，失败显示
五语言 typed error 并保留原状态。

## P1：设置、状态和工具面板

`model-picker` 是 01 的 UI 场景 ID，其业务语义由上一节的 `model-effort-remap` 与
`model-switch-atomic` 裁决；本行只跟踪弹窗、列表、键盘与布局证据，不重复记录控制平面结论。

| ID                          | Owner 状态    | 现有 evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 精确 Gate                                      | 下一刀                                                                                                                                                                                                              |
| --------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model-picker`              | covered       | `ChatModelSelector.integration.test.tsx`、`input-kit/ModelSelector.test.tsx`、`modelPickerPolicy.test.ts`、`modelPickerPolicyBoundary.test.ts`、`codexModelPickerPolicyOrigin.test.ts`                                                                                                                                                                                                                                                                                                                                                      | Gate A/B missing                               | 固定 viewport/locale 验证弹窗滚动、键盘选择与窄宽，不再由 TUI 文案推导业务断言                                                                                                                                      |
| `permissions-picker`        | partial       | `permissionEvents.test.mjs`、`InputbarApprovalPrompt.test.tsx`、`agentApprovalServerRequest.unit.test.ts`、`legacyToolPermissionGuard.test.ts`                                                                                                                                                                                                                                                                                                                                                                                              | Gate B missing                                 | 证明 access profile 选择回写当前 thread，且不混用 workspace 默认值                                                                                                                                                  |
| `status-surface-matrix`     | covered       | `statusSurfaceMatrix.test.mjs`、`harnessStatusPanelViewModel.unit.test.ts`、六个 `HarnessStatusPanel.*.test.tsx`、`threadWorkspaceHeaderViewModel.unit.test.ts`                                                                                                                                                                                                                                                                                                                                                                             | Gate A missing                                 | 固定 viewport 证明 footer/header/title/model/reasoning/goal/rate-limit 同源                                                                                                                                         |
| `status-running-completed`  | covered       | `MessageListRuntimeStatus.test.tsx`、`InputbarRuntimeStatusLine.test.tsx`、`agentRuntimeStatus.unit.test.ts`、`agentStreamRuntimeStatusController.test.ts`、`threadStatusRuntimeUpdate.test.mjs`、`tokenUsageReplay.test.mjs`                                                                                                                                                                                                                                                                                                               | Gate A/B missing                               | 补 usage-limited/blocked 终态与完成耗时的同源断言                                                                                                                                                                   |
| `mcp-inventory-elicitation` | covered       | `mcpStartupStatus.test.mjs`、`mcpInventoryStatus.test.mjs`、`mcpElicitation.test.mjs`、`mcpThreadScope.test.mjs`、`McpServerElicitationDialog.test.tsx`、`mcpServerElicitation.unit.test.ts`、`mcp.failClosed.test.ts`、`lime-mcp` 151/151                                                                                                                                                                                                                                                                                                  | `smoke:mcp-current` 已过；exact Gate B missing | 补 elicitation 表单 typed 响应与 outer request id 匹配的 Electron 证据                                                                                                                                              |
| `hooks-lifecycle`           | owner-unwired | Owner 已收敛到 `tool-runtime`：`hook_lifecycle.rs` 负责裁决（block/abort/rewrite/inject、`MissingDecision`/未实现 handler/不可信来源/非法 matcher/非 Sync 模式全部 fail closed、稳定 `run_id`、按 display order 聚合）；`hook_runtime.rs` 负责 discovery 与 `Command` 执行（单一 Codex 事件分组格式、跨平台 shell、stdin 上下文、超时、blocking 语义、stdout 结构化结果）；`hook_gated_executor.rs` 装饰任何 `RuntimeToolExecutor`，`PreToolUse` 阻断/改写（非法改写 fail closed）、`PostToolUse` 评估，阻断前标记 `before_handler` 不进 handler、已执行不标成未执行。三模块定向 34/34，crate 全量 295/295。旧 `agent/src/hooks.rs` 已物理删除，回流守卫 `rust-retired-agent-hook-manager`（dead，实际引用 0） | 全部 Gate missing                              | 仍缺 sampling step 接线、canonical Item 投影、App Server notification、Renderer 与 Gate B，因此保持 `owner-unwired`。旧扁平 `{hooks:[...]}`、已退役事件名与 `async_exec` 为 `dead / deleted / forbidden-to-restore` |
| `multi-agent-roster`        | covered       | `multiAgentItemTaxonomy.test.mjs`、`multiAgentToolSchema.test.mjs`、`multiAgentVisualSnapshot.test.mjs`、`subagents.failed.test.mjs`、Rust `runtime/tests/agent_control/{concurrent,effective_route,fork,restart}.rs`、`agent_control_gateway/wait.rs`、`subagentClient.current-boundary.test.ts`                                                                                                                                                                                                                                           | Gate B missing                                 | spawn/wait/followup/interrupt 全链 Electron 证据与 worker notification identity                                                                                                                                     |
| `plugin-marketplace`        | covered       | `pluginMarketplaceViewModel.unit.test.ts`、`pluginMarketplace.unit.test.ts`、`PluginsPageViewModel.unit.test.ts`、`pluginCapabilityRuntime.test.mjs`、`pluginContract.unit.test.ts`                                                                                                                                                                                                                                                                                                                                                         | Gate A missing                                 | install/remove/detail/error 恢复路径的稳定页面证据                                                                                                                                                                  |
| `apps-and-capabilities`     | covered       | `agent-capability-catalog/tests/catalog.test.mjs`、`selectedCapabilityStack.test.mjs`、`skillsCatalogScope.test.mjs`、`skill-package-current.test.mjs`、`skillBindingsCurrentBoundary.test.ts`、`useLimeSkills.test.tsx`                                                                                                                                                                                                                                                                                                                    | Gate A/B missing                               | watcher/readiness 归 V1-07；未完成前不按现有 catalog 测试标完整覆盖                                                                                                                                                 |
| `plan-goal`                 | covered       | `AgentPlanBlock.test.tsx`、`planState.unit.test.ts`、`ThreadGoalPanel.test.tsx`、`useAgentSessionThreadGoal.{unit,component}.test`、`goalLifecycleHydrate.test.mjs`、`threadGoalClient.unit.test.ts`、`InputbarComposerSection.planStatus.test.tsx`                                                                                                                                                                                                                                                                                         | Gate A/B missing                               | 与 `turn-budget-limit` 共用 budget terminal 证据，避免两处各造一套状态                                                                                                                                              |
| `feedback-update-usage`     | partial       | `usageStats.test.ts`、`usageStats.current-boundary.test.ts`、`TokenUsageDisplay.test.tsx`、`appUpdate.test.ts`、`UpdateCheckSettings.test.tsx`、`update-notification.test.tsx`、`reviewFeedbackProjection.test.ts`                                                                                                                                                                                                                                                                                                                          | Gate A missing                                 | feedback consent 二次确认与登录/限额错误的五语言稳定文案                                                                                                                                                            |

## P1：媒体、产物和错误

| ID                       | Owner 状态 | 现有 evidence                                                                                                                                                                                                                          | 精确 Gate                                | 下一刀                                                                |
| ------------------------ | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------- |
| `media-image-attachment` | covered    | `localRemoteImageInput.test.mjs`、`MessageImageAttachments.test.tsx`、`useImageAttachments.test.tsx`、`imageAttachments.test.ts`、`InputbarVisionCapabilityNotice.test.tsx`、`modelInputModalityPolicy.test.ts`                        | Gate A/B missing                         | 历史恢复后的图片引用/占位，以及不支持媒体时的可见拒绝态               |
| `media-image-generation` | covered    | `ImageTaskViewer.test.tsx`、`imageWorkbench{Command,Presentation,StatusText,MessageDisplay}` 系列、`MessageList.imageTasks.test.tsx`、`agentChatHistoryLocalMerge.imageTasks.test.ts`                                                  | Gate B missing                           | begin/success/failure 三态 Electron 证据；失败不伪造 artifact         |
| `artifact-diff-preview`  | covered    | `diffArtifactSnapshot.test.mjs`、`artifactEvents.test.mjs`、`artifact-write.test.mjs`、`FileChangesSummaryCard.test.tsx`、`StreamingRenderer.fileChanges.test.tsx`、`messageListTimelineContentParts.fileChangesTerminal.unit.test.ts` | Gate B covered（typed `artifact/write`） | 补 diff hunk、path/line 与 preview action 的 Gate A geometry          |
| `error-config-provider`  | covered    | `agentRuntimeErrorPresentation.test.ts`、`agentStreamErrorController.test.ts`、`providerConnectionError.test.ts`、`agentSessionDetailHydrationError.test.ts`、`runtime_backend::model_route_resolver` fail-closed                      | Gate B missing                           | 与 `provider-readiness` 共用未就绪 route 的同一条 Electron 证据       |
| `error-safety-sandbox`   | partial    | `tool-runtime/src/sandbox.rs`(4)、`sandbox/command.rs`(3)、`agent_tools/execution/sandbox_backend_tests.rs`(7)                                                                                                                         | Gate A/B missing                         | deny/enable/fallback 提示与 retry 路径的 GUI 证据；Windows 语义未闭合 |
| `error-mcp-hook`         | partial    | MCP 侧 `mcp.failClosed.test.ts` 与 `lime-mcp` 151/151；hook 侧无 current 接线                                                                                                                                                          | Gate B missing                           | MCP 半边可先推进；hook 半边被 `hooks-lifecycle` 阻塞，不合并计数      |

## P2：产品壳和边界

| ID                            | Owner 状态    | 现有 evidence                                                                                                                                                                       | 精确 Gate         | 下一刀                                                                                                           |
| ----------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------- |
| `onboarding-setup`            | partial       | `lib/base-setup/{bootstrap,compiler,validator,storage,rolloutGate}.test.ts`、`setupStateStore.test.ts`                                                                              | Gate A missing    | Lime onboarding 与 Codex TUI onboarding 不同构，只验证现役字段、错误与恢复路径                                   |
| `settings-memory-personality` | covered       | `settings-v2/general/memory/index.test.tsx`、`MemoryStoreStatusPanel.test.tsx`、`StyleProfileSelector.test.tsx`、`rolloutCandidates.unit.test.ts`、`account/profile/index.test.tsx` | Gate A missing    | 只验证 Lime 现役设置；Codex ChatGPT 专属项保持排除                                                               |
| `search-resume-picker`        | covered       | `index.sidebar.test.tsx`、`threadResumeRunningStream.test.mjs`、`sessionHistoryPaginationController.test.ts`；V1-18 侧边栏 Thread 查询 Gate B                                       | Gate A missing    | 搜索、排序、分页、窄宽与加载错误的稳定页面证据                                                                   |
| `platform-windows`            | owner-blocked | `electron/windowsSquirrelStartup.test.ts`、`windows-squirrel-rc-smoke.test.mjs`、`hotkeys/platform.test.ts`、`contentPostPlatform.unit.test.ts`                                     | 全部 Gate missing | 仅覆盖安装器与快捷键；Windows transport/sandbox 语义仍是 08 register 的 P1 阻塞，无 Windows 真机证据前不得标覆盖 |
| `cli-doctor`                  | defer         | 无，且按计划不建立 Lime GUI owner                                                                                                                                                   | 不适用            | 保持 `defer`，只在 03 map 中作为负向分类留存                                                                     |

## Runtime contract 场景

这些是 contract/integration 层 owner，Gate A 不适用；只有在结果被 GUI 消费时才要求 Gate B，
因此不并入下一节的 Gate 统计。

| ID                                 | Owner 状态 | 现有 evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 缺口                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `runtime-compaction`（23）         | covered    | `context_compaction.rs` 9 个 owner 测试，含 `compaction_windows_preserve_replacement_history_lineage`、`replacement_history_keeps_compacted_user_boundaries_before_summary`、`malformed_compaction_lineage_fails_closed_instead_of_resetting_ids`、两个 fork seed nested-marker 拒绝；public JSON-RPC `thread_fork_compaction_jsonrpc.rs::compacted_thread_fork_replays_replacement_and_surviving_tail_after_restart`（冷重启 replay）、`thread_compact_jsonrpc.rs`；`thread_fork/tests.rs`、`runtime/tests/sessions.rs`、`agent_control/fork.rs`、`thread_item_projection/typed_tests.rs`；`contextCompactionItem.test.mjs` 6 例 | 实现与 contract 证据已闭环：`replacementHistory` 与 `windowNumber/firstWindowId/previousWindowId/windowId` 链已落地并 fail closed。**08 register 第 3 条「仍以 summary/tail 为主、缺 replacement_history 与 window lineage」已过期，需按本行证据更正**（08 当前被并行车道占用，未在本轮改写）。剩余缺口只是 `history-compaction-replacement` 的 manual/auto/mid-turn 三条 Gate B fixture |
| `runtime-pending-input`（3）       | covered    | `agent-runtime/src/session_loop/input_queue.rs`（830 行）与 `session_loop/tests.rs`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 无 public 面缺口；按 `composer-queue-steer` 裁决，内部 FIFO 为 current，public queue 写平面禁止回流。`input_queue.rs` 已超 AGENTS.md 的 800 行拆分阈值，需登记退出条件                                                                                                                                                                                                                   |
| `runtime-model-layout`（6）        | covered    | `agent-runtime/src/session_config.rs` 与多模型章节的 model switch owner 测试                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 与 `model-switch-atomic` 共用 Gate B 缺口，不重复计数                                                                                                                                                                                                                                                                                                                                    |
| `runtime-context-budget`（2）      | partial    | `agent-protocol/src/model_context.rs`、`agent/src/protocol_context_projection.rs`、`agent/src/model_request_policy.rs`、`tool-runtime/src/tool_io.rs`                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 预算/截断需表达为 typed 状态而非字符串推断；与 `turn-budget-limit` 共用终态证据                                                                                                                                                                                                                                                                                                          |
| `runtime-mcp-exposure`（3）        | partial    | `tool-runtime/src/tool_lifecycle.rs`、`tool_extension.rs`、`mcp_connection/step_snapshot.rs`、`turn_snapshot.rs`（骨架）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | deferred tool expose/recover/resume 不漂移的证据依赖 V1-05 snapshot 接线                                                                                                                                                                                                                                                                                                                 |
| `runtime-environment-context`（9） | partial    | `agent-protocol::{world_state::RuntimeWorldState,MultiAgentMode}`、App Server `request_context::turn_context` typed projection、Agent Runtime `provider_turn` typed resolver/provider request capture、v2 schema/generated TS、public JSON-RPC 与 protocol render tests                                                                                                                                                                                                                                                                                                                                                           | Environment、permissions、collaboration 与 `ultra -> proactive` effective multi-agent mode 已进入真实 provider request；AGENTS/apps/plugins/environment instructions、realtime 与 Codex durable full/patch history仍缺。禁止回到 prompt 层逐项拼字符串或恢复 arbitrary JSON mode                                                                                                         |

## 场景覆盖统计

以 01/02 计划的场景 ID 为分母，按上表实际状态汇总：

| 分组              | 场景数 | 精确 Gate 已闭环 | 待复跑 | Gate 缺口 | 非纯 Gate 阻塞                                            |
| ----------------- | -----: | ---------------: | -----: | --------: | --------------------------------------------------------- |
| P0 Agent 主路径   |     11 |                4 |      2 |         4 | 1（`turn-budget-limit` partial contract）                 |
| P0 输入和布局     |      7 |                3 |      0 |         4 | 0                                                         |
| 多模型控制平面    |      4 |                0 |      0 |         4 | 0                                                         |
| P1 设置/状态/面板 |     11 |                0 |      0 |        10 | 1（`hooks-lifecycle` owner-unwired）                      |
| P1 媒体/产物/错误 |      6 |                1 |      0 |         5 | 0                                                         |
| P2 产品壳和边界   |      5 |                0 |      0 |         3 | 2（`platform-windows` owner-blocked、`cli-doctor` defer） |
| 合计              |     44 |                8 |      2 |        30 | 4                                                         |

精确 Gate 闭环率为 `8 / 44 = 18.2%`；把两条待复跑计入则为 `10 / 44 = 22.7%`。
owner 层面已有确定性测试的比例远高于此，但 owner 测试不能替代场景要求的 Gate 证据，
因此本表不用 owner 覆盖率对外表述完成度。

Runtime contract 6 项另计：`covered` 3（`runtime-compaction`、`runtime-pending-input`、
`runtime-model-layout`）、`partial` 3（`runtime-context-budget`、`runtime-mcp-exposure`、
`runtime-environment-context`），`missing` 0。

因此本方案的场景总数为 `44 + 6 = 50`；当前已无 `missing`。world-state 已进入 provider request，
但 typed producer 与 durable full/patch history 尚未闭合，因此仍不能计为 covered。

### 三条最短路径

按“解除下游依赖”排序，而非按缺口数量排序：

1. **完成 world state producer 与历史链**。`RuntimeWorldState` 已从 App Server 投影进入真实 provider
   request，effective multi-agent mode 也已按 `ultra -> proactive` 接入；下一步补
   AGENTS/apps/plugins/environment instructions 与 realtime，再建立 Codex full/patch durable history。Codex 的 9 个快照仍同时供给
   `error-config-provider`、`apps-and-capabilities` 与 `permissions-picker`，禁止三处各自拼字符串。
2. **完成 V1-05 Hook 接线**。`hooks-lifecycle` 是唯一的 `owner-unwired`，同时卡住
   `error-mcp-hook` 的一半与 `runtime-mcp-exposure` 的 deferred tool 证据。
3. **补 `history-compaction-replacement` 的三条 Gate B fixture**。lineage 本身不是缺口：
   `replacementHistory` 与 window 链已实现、已 fail closed、且有冷重启 public JSON-RPC replay 证据。
   `runtime-compaction` 的 23 个快照是 P0 权重最大的一组，剩下要做的只是 manual、auto 与 mid-turn
   各一条 current fixture，证明 replacement 只注入一次且旧摘要不重复。

其余 Gate 缺口多为“owner 已 covered、只差固定 viewport 或一次 Electron fixture 复跑”，
可在上述三条推进的同时按 owner 批量补齐，不构成串行阻塞。

## Session mutation 清理

旧 `agentSession/update` 已物理删除并归类为 `dead / deleted / forbidden-to-restore`，不再存在
compat、fallback 或待迁出的 metadata consumer。原字段按单一职责迁到 current owner：

- 模型、provider、reasoning 与工具偏好：`thread/settings/update`。
- 标题：`thread/name/set`。
- 归档、恢复与删除：`thread/archive`、`thread/unarchive`、`thread/delete`。
- Article Workspace 草稿和持久化选择：`artifact/write` 的完整 `workspacePatch` snapshot。

回流守卫覆盖 Rust protocol/catalog/schema export、App Server dispatch/runtime、生成 schema、typed
client、Renderer gateway、DevBridge policy 与 command catalog；旧 method/DTO/helper/schema 只能出现
在 `FORBIDDEN_METHODS`、`not.toContain` 等负向测试或历史 evidence 中。

本次定向证据：

- `npm exec vitest run "src/lib/dev-bridge/commandPolicy.test.ts" "src/lib/dev-bridge/http-client.test.ts" "src/lib/governance/legacySurfaceCatalog.test.ts"`：264/264 通过。
- `npm --prefix "packages/app-server-client" test`：83/83 通过。
- `node "scripts/check-app-server-client-contract.mjs"`：284 项检查通过。
- public queue 清理定向验证：App Server TypeScript client 90/90、Renderer queue/current projection
  107/107、fixture source guards 91/91；Rust related（含 `app-server` 1528/1528、protocol 80/80）
  与 `npm run test:contracts` 均通过。schema/generated TS 已重建为 794 个协议类型，失败 0。
- governance legacy report 为零引用候选 0、分类漂移候选 0、边界违规 0；current Agent fixture
  聚合通过且 `liveProviderUsed=false`；GUI smoke evidence run id 为
  `standalone-shell-01-20260726043817-91929`，结果 pass。
- active-steer 实施完成时定向 Vitest 为 5 files / 115 tests 通过；当前并行工作树复跑为 114/115，
  唯一失败是无关 Right Surface visual matrix 的旧 source-string guard。active-steer 场景过滤 3/3、
  其余四个 owner/fixture 文件 37/37 通过。当前完整源码 Electron smoke fresh build 通过，Renderer
  `7091` modules transformed；`inputbar-active-steer` 三次独立 Gate B 均通过，10 项场景断言全为
  true，mock fallback、invoke error、console error 与 page error 均为 0。
- 当前 `npm run test:contracts` 通过：802 个协议类型无生成漂移，App Server client 284 checks，
  command/harness/modality/scripts/Electron release/docs boundary 全部通过。
- `turn-stream-repair` 精确 Electron Gate B 通过：真实 Electron/preload/IPC、App Server JSON-RPC、
  runtime/read model 与 GUI 使用同一 thread/turn/item identity；fixture 在首段 `message.delta` 后故意
  丢弃中段 delta，再由完整 canonical `item.completed` 修复，并由 `turn.completed` 收敛。GUI 中首段、
  overflow marker、表格首尾行和完成标记均只出现一次，真实渲染 `<table>` 数量为 1；read model
  canonical text 长度为 `1803`，SHA-256 为
  `7c1f81f089abd3fe5c732aece5074a99e8a5874cf21b2aaf94369a92626f308b`；mock fallback、console
  error、page error 均为 0。
- `npx vitest run "scripts/agent-runtime/claw-chat-current-fixture-smoke.test.mjs" --silent=passed-only --disableConsoleIntercept`：74/74 通过。
- `content-factory-article-workspace` 精确 Electron Gate B 通过：真实 Electron/preload/IPC、
  App Server JSON-RPC、runtime/read model 与 Article Editor 使用同一 thread identity；
  `artifact/write` 命中 3 次，`agentSession/update` 命中 0 次，mock fallback、console error、
  page error 均为 0。刷新和从侧边栏重新打开后仍恢复 `metadata.editedDraft` 对应正文。
- `npm run smoke:agent-runtime-current-fixture`：聚合通过，`liveProviderUsed=false`。
- `npm run verify:gui-smoke`：21/21 Electron shell assertions 通过，evidence run id 为
  `standalone-shell-01-20260725182656-23369`。

Article 编辑事实只由 `artifact.metadata.editedDraft` 与完整 `workspacePatch` 表达；
`source.edited` 不属于 current 写入契约，fixture 不保留该重复布尔事实或正向断言。
