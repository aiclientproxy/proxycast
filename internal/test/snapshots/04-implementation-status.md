# P0 实现与证据状态

日期：2026-07-26

本表只记录精确场景证据。`npm run verify:gui-smoke` 或 current Agent fixture 整体通过，不能替代
某个场景要求的故障注入、DOM geometry 或 public JSON-RPC 断言。

状态定义：

- `covered`：已有最贴 owner 的确定性测试。
- `partial`：已有相邻能力或部分状态，但尚未完整表达场景语义。
- `gate-missing`：owner 测试已存在，目标 Gate A/B 精确证据尚缺。
- `owner-blocked`：current protocol/read model 无法表达该状态，不能在测试层造字段。

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
