# v2 渲染对齐执行计划

状态：in-progress

主目标：以 Codex 的 Thread / Turn / Item 生命周期和渲染语义替换 Lime 当前把 canonical Item 二次压缩为 Message、tool_call、extension 的对话渲染路径，同时保持 model-provider 的 Grok 多模型控制面和 OpenCode 多模态/多协议 lowering 不变。

当前阶段：V2-01 唯一 ConversationProjection contract（首切片已完成）

下一刀：收紧 `readCanonicalThreadItem` 的 typed ItemProjection 返回边界，并让 live/cold/replay 进入同一 reducer fixture

## 1. 约束与非目标

必须遵守：

- Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI 是唯一产品链。
- Codex 是 runtime、Thread/Turn/Item、App Server、工具生命周期、恢复和 GUI 投影的主参考。
- Grok Build 继续是 model catalog、route、capability、switch、retry/circuit breaker 的主参考。
- OpenCode 继续是 provider wire、canonical content、媒体、endpoint 和多协议 lowering 的辅参考。
- Renderer 不直接读取 provider raw payload、文件系统、PTY、credential 或 raw JSON-RPC request id。
- 无外部兼容需求：调用应直接迁移，旧生产入口在替代路径验证后物理删除，不引入长期 wrapper。

明确不做：

- 不把 Codex 的 ChatGPT account、单一 provider、TUI 外观或 provider wire 搬进 Lime。
- 不把多模型/多模态能力降级为文本或单模型 fallback。
- 不让 component、Electron 或 App Server processor 重新实现 model selection、capability 或 provider lowering。
- 不把通知字符串、raw response 或旧 Message cache 作为第二个 history owner。

## 2. 分阶段计划

### V2-00：冻结事实源与覆盖库存

状态：completed（2026-07-28；18/72/11 coverage fixture 与治理守卫通过）

目标：把 v2 的上游集合、当前 schema 和渲染缺口固化成可回归的 fixture，避免实施期间凭印象补类型。

写集：

- internal/refactor/v2/\*\*
- app-server-protocol 的 schema/fixture 目录，仅在上游字段缺口有明确 producer 时修改
- 前端 coverage fixture 与 contract test

动作：

1. 从 Codex 4c434651 的 item.rs、envelopes.rs、thread_history.rs 与 TUI history_cell/streaming 建立 18 Item、72 notification、11 reverse request manifest。
2. 为每个 entry 标记 current、planned、product-scope-excluded 或 deprecated，并把来源 revision 固化。
3. 记录 Lime 当前的 18 Item、28 typed notification、4 typed reverse request 和 14 个前端直连 notification。
4. 把 model/list/updated 等 Lime-owned 多模型/多模态 surface 单列为 model-provider ownership；它们不能被当作 Renderer 的本地补丁，也不能因 Codex 无同名 method 被删除。

退出条件：

- ITEM-PROJECTIONS 与 EVENT-PROJECTIONS 的每个计数均能由 fixture 验证。
- upstream revision 变化时，CI 报出新增/删除/字段漂移，不能只改文档 hash。

### V2-01：唯一 ConversationProjection contract

状态：in-progress（首切片 UnknownItemProjection + ContextCompaction 可见性已完成）

目标：在 Renderer gateway 之后建立 discriminated projection，取代以 AgentThreadItem + Message content part 混合驱动场景的路径。

建议写集：

- src/lib/api/agentRuntime/conversationProjection/\*\*
- src/lib/api/agentRuntime/appServerCanonicalItemReader.ts
- src/lib/api/agentRuntime/appServerEventTimelineReaders.ts
- src/lib/api/agentRuntime/appServerV2Notification.ts
- src/lib/api/agentProtocolCoreTypes.ts 与对应 d.ts/generated consumer
- packages/app-server-client/\*\*，仅在 exact protocol shape 已变更时

动作：

1. 将接近或超过体量阈值的 canonical reader、event timeline reader 拆为 contract、item reader、notification reader、reducer、sanitizer 和 test fixture 模块；不继续向单文件堆状态机。
2. 定义 ConversationProjection、TurnProjection、ItemProjection、PendingInteractionProjection、NoticeProjection 和 UnknownItemProjection。
3. type + id、threadId + turnId、首次 sequence、started/completed timestamp 是唯一 identity 规则。
4. 建立 started、delta、progress、completed、turn completed、server request resolved、transport disconnect 的幂等 reducer。
5. 建立有界 orphan buffer；started 前 delta 可回放，terminal 后 late delta 拒绝并记录诊断。
6. 不再把 declined lower 成 failed，不再把 unknown type return null。

删除目标：

- 删除 readCanonicalThreadItem 中对合法未知 Item 的静默 null 分支，替换为 UnknownItemProjection。
- 删除以 extension 作为多个 Codex Item 的长期默认渲染出口；只允许在单一、已知的临时迁移 test fixture 内存在。
- 删除 Renderer 依赖 event timestamp 伪造 sequence 的路径，改为使用 canonical sequence 或显式 provenance。

退出条件：

- live notification、thread/read、replay fixture 进入同一 reducer，得到相同的 Item identity、顺序和 terminal 状态。
- 同一 Item 的 completed snapshot 可覆盖任何 delta 草稿；Plan、Reasoning、Patch、MCP progress 都有负向测试。

### V2-02：协议与安全 display shape 收口

状态：pending

目标：补齐对渲染不可缺少、但目前被 String 或 opaque Value 模糊化的 Item 字段，保持 provider 细节留在 model-provider。

建议写集：

- lime-rs/crates/app-server-protocol/src/protocol/v2/item.rs
- lime-rs/crates/app-server-protocol/src/protocol/v2/envelopes.rs
- schema/json、generated client、packages/app-server-client
- lime-rs/crates/app-server/src/runtime/thread_item_projection/\*\*
- Electron host/preload 的 media/path/open semantic gateway

动作：

1. 将 MCP result/error、媒体 reference、DynamicToolCall output、UserInput audio/localAudio 形成 typed、size-bounded、脱敏的 current contract。
2. 将 cwd、path、plugin/script、memory citation 和资源链接 lower 为 host-safe display/action reference，不向 Renderer 透传权限或绝对路径。
3. 仅在 Codex current protocol 已有且 Lime 属于产品范围时补 notification/reverse request；每项同步 schema、handler、client、fixture。
4. 模型/媒体 capability 继续取 ResolvedModelRoute 和 model-provider canonical content；协议不根据 provider name 假设 input/output modality。

删除目标：

- 删除以 opaque Value 或 display text 猜测 MCP 成败、工具输出类型和媒体类型的 Renderer 分支。
- 删除 local path 直接进入 React props 的现有历史通路。

退出条件：

- npm run test:contracts 通过。
- Rust producer、JSON schema、typed client、Electron semantic gateway、renderer consumer 对同一字段使用同一名称和空值语义。
- 图片/音频不可用、无权限、格式不支持、结果过大均 fail visible 且不泄露原始路径/凭证。

### V2-03：原序 TurnTimeline 与核心 ItemRenderer

状态：pending（ContextCompaction 的前置删除已在 V2-01 首切片落地，完整原序时间线尚未开始）

目标：从历史和直播中直接渲染 Item，而不是先合成为 assistant Message。

建议写集：

- src/components/agent/chat/projection/\*\*
- src/components/agent/chat/components/AgentThreadTimeline.tsx
- src/components/agent/chat/components/item-renderers/\*\*
- src/components/agent/chat/components/MessageList.tsx
- src/components/agent/chat/hooks/agentChatHistoryThreadItems.ts
- src/components/agent/chat/hooks/agentStreamAgentMessageContentSync.ts
- src/components/agent/chat/utils/threadTimelineView.ts

动作：

1. 新建 TurnTimeline，以 Turn 为虚拟化第一层、长输出/Diff 为第二层；只按 canonical sequence 排序。
2. 先接 UserMessage、AgentMessage、Plan、Reasoning、CommandExecution、FileChange、WebSearch、ImageView、ImageGeneration、ContextCompaction。
3. 复用成熟 Markdown、Diff、文件变更、媒体和工具展示子组件，但输入必须改为 ItemProjection，不再读取 legacy Message 合成状态。
4. streaming 文本按 animation frame 或 30-50ms 合批；长表格/Markdown 采用稳定区与可变尾部的语义，避免 token 级整页重排。
5. ContextCompaction 改为低干扰信息行，不能继续被 filterConversationThreadItems 静默隐藏。

删除目标：

- 删除 agentChatHistoryThreadItems 以 assistantDraft 拼接完整 Item 时间线的主生产路径。
- 删除 agentStreamAgentMessageContentSync 对 canonical Item 进行第二次 Message content part 合成的主生产路径。
- 删除 threadTimelineView 中 context_compaction 的隐藏条件。
- 删除 MessageList 中依赖上述旧聚合结果的只读 compatibility branch。

退出条件：

- Message -> Tool -> Message -> Tool 的交错 sequence 在 live、cold read、replay 一致。
- Markdown、Plan、Reasoning、Shell output、Diff 和 ContextCompaction 均可独立定位、更新、折叠和恢复。
- 没有新 UI 卡片套卡，主对象 Thread、当前状态和唯一下一步在桌面界面中清楚。

### V2-04：工具、阻塞交互与多 Agent

目标：补 MCP、DynamicTool、collaboration、Hook 与所有 current reverse request 的唯一交互层。

建议写集：

- src/components/agent/chat/components/item-renderers/\*\*
- src/components/agent/chat/components/PendingInteractionLayer/\*\*
- src/lib/api/appServerServerRequest.ts 及对应 event bus
- electron/appServerHost.ts、preload/gateway 与 contract tests
- lime-rs/crates/app-server/src/processor/\*\*，仅当前 request producer 接线

动作：

1. 渲染 MCP arguments/progress/content/structuredContent/error，DynamicTool text/image/audio 输出，CollabAgentToolCall 与 SubAgentActivity。
2. 将 command/file approval、requestUserInput、MCP elicitation 统一为 PendingInteractionProjection；已有 DecisionPanel/McpServerElicitationDialog 作为可复用表单内容，不保持第二 pending store。
3. pending anchor 只定位到 Item；表单始终位于 Composer 上方且跨 Thread 排队。
4. 对 permission approval 与 item/tool/call 先完成 product-scope 和 runtime producer，再连接 UI；不得由 Renderer 伪造。
5. serverRequest/resolved、turn completed、thread closed、disconnect 均执行同一终结 reducer。

删除目标：

- 删除按 ActionRequired 文本或旧 request_id 在 Message 内构造重复审批卡的路径。
- 删除一条 request 同时在时间线、弹窗、任务栏出现多个可提交表面的路径。
- 删除 raw request id 进入 React state、日志或可见 DOM 的路径。

退出条件：

- 同一 pending request 只允许提交一次，跨客户端 resolved 后所有表面立即只读。
- command/file/MCP/用户输入在真实 Electron 中走原始 App Server request/response，不使用 mock backend。
- DynamicTool 和多 Agent 行不推断业务完成状态；只显示 canonical item lifecycle。

### V2-05：高级通知、恢复和治理删除

目标：完成剩余产品范围内 notification、全面 recovery 与旧投影物理删除。

动作：

1. 按 EVENT-PROJECTIONS 逐项补 Hook、Turn plan/diff、environment、warning、fuzzy search、process、realtime、Windows sandbox 等 planned surface。
2. 每项先补 App Server typed protocol 和 producer，再补 Electron gateway、projection、renderer 和 Gate B；不能由前端先造状态。
3. 加入 unknown notification drift recorder：known-but-excluded 发 DX，unknown notification fail visible；unknown reverse request fail closed。
4. 移除已被 direct ItemRenderer 替代的旧 Message hydration、stream merge、legacy event parser、compat DTO 和测试夹具。
5. 在同一变更集中更新 internal/aiprompts/architecture.md，记录 Renderer read path 从 Message synthesis 收敛到 ConversationProjection，并完成架构图确认。

删除完成标准：

- rg 不再找到旧 canonical Item -> Message 合成入口被生产调用。
- 无 protocol/v0、agentSession/\*、renderer mock fallback 或第二 timeline store 回流。
- governance catalog 将移除路径标为 dead / deleted / forbidden-to-restore，并有负向扫描测试。

## 3. 验证与门禁

每个阶段先跑受影响定向测试；跨协议、bridge 或 GUI 的切片至少执行：

    npm run test:contracts
    npm run test:rust:related -- <changed paths>
    npm run verify:gui-smoke
    npm run smoke:agent-runtime-current-fixture
    npm run governance:legacy-report

整合前执行：

    npm run verify:local

如需全量前端回归，使用：

    npm run test:resume

Gate A 证明 typed schema、projection reducer、冷读、分页、序列、终态和覆盖 map。Gate B 必须证明：

    Renderer
      -> preload/IPC
      -> app_server_handle_json_lines
      -> App Server JSON-RPC
      -> runtime/read model
      -> GUI

Gate B 最少场景：Agent Markdown、Reasoning、Plan、Search、Shell output、File Diff、MCP progress/result、dynamic tool、approval、requestUserInput、图片、音频、interrupt、disconnect/resume 和未知 Item。

## 4. 风险与护栏

| 风险                           | 护栏                                                                                                             |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| 为赶 UI 先在 Renderer 造字段   | protocol producer、schema、consumer 同批完成；否则保留 gap                                                       |
| Codex 对齐误伤多模型/多模态    | model-provider 写集独立；每次删除先确认 ResolvedModelRoute、canonical content、lowering、capability 测试未受影响 |
| 新旧时间线长期并存             | 每个 V2 slice 都有明确迁移调用点和删除目标；无双轨 release                                                       |
| 直播与恢复不一致               | 同 reducer + live/cold/replay 参数化 fixture                                                                     |
| 大日志/Diff 造成性能回退       | Turn/输出二级虚拟化、有界 buffer、按需抽屉、流式合批                                                             |
| 路径、secret、raw request 泄漏 | host semantic action、脱敏 display projection、raw id 只留 Electron main                                         |
| 协议升级静默丢字段             | coverage map、unknown drift、schema revision fixture、fail visible                                               |

## 5. 分类与完成度

| 分类           | v2 裁决                                                                                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| current        | App Server v2 主链、Thread/Turn/Item durable identity、model-provider 多模型/多模态 current owner、现有成熟 Markdown/Diff/media 子组件                            |
| compat         | 不保留长期生产 compat；仅可存在一次性迁移测试夹具                                                                                                                 |
| deprecated     | fileChange outputDelta、thread/compacted、legacy patch/exec approval 的去重处理                                                                                   |
| dead / deleted | unknown Item null drop、静默 hide ContextCompaction（本轮已删除）；Item -> Message 二次合成主链、raw request id 进入 Renderer、重复 timeline store 仍是待删除目标 |

计划完成度：12%。V2-00 已完成；V2-01 完成未知 Item fail-visible 与 ContextCompaction 可见性的首切片，但 typed ConversationProjection、统一 reducer、unknown drift recorder 和完整 ItemRenderer 仍未完成。

## 6. 执行台账

### 2026-07-28：V2-00 完成，V2-01 首切片完成

状态：已完成本轮切片；总体仍为 `in-progress`。

本轮实际写集：

- v2 事实与计划：`internal/refactor/v2/README.md`、`ITEM-PROJECTIONS.md`、`EVENT-PROJECTIONS.md`、`IMPLEMENTATION-PLAN.md`、`fixtures/render-projection-coverage.v0.1.json`。
- 覆盖守卫：`src/lib/governance/renderProjectionCoverageBoundary.test.ts`。
- canonical Item：`src/lib/api/agentRuntime/appServerCanonicalItemReader.ts` 与对应测试。
- Item 类型合同：`src/lib/api/agentProtocolCoreTypes.ts`、`src/lib/api/agentProtocol.d.ts`。
- 时间线可见性与 renderer 回归：`threadTimelineView.ts`、`threadTimelineView.test.ts`、`AgentThreadTimelineItemRenderers.tsx`、`AgentThreadTimeline.test.tsx`。
- 五语言文案：`src/i18n/resources/{zh-CN,zh-TW,en-US,ja-JP,ko-KR}/agent.json`。

本轮完成：

- coverage fixture 固化 Codex revision `4c43465133428898aa84f0bfc02c306ed65fb66a` 的 18 Item、72 notification、11 reverse request；`model/list/updated` 保持 Lime model-provider 扩展。
- 合法未知 camelCase Item 投影为 `unknown_item`，只保留 `upstream_type`、有界脱敏 `field_names` 和 lifecycle identity；raw metadata 与原始值不进入投影。
- 旧 snake_case/已知内部 Item 继续 fail closed；未知 Item 使用既有 unsupported fallback，不展开 raw JSON。
- 删除 `threadTimelineView.ts` 对 `context_compaction` 的静默过滤；保留既有 `ContextCompactionCard` 信息行，并补组件 DOM 回归。

实际验证：

- `npm test -- ...` 定向集合：5 个文件、67 tests passed。
- `npm run typecheck`：通过。
- `npm run test:contracts`：通过（protocol/client、command、modality、scripts、release、docs boundary）。
- `npm run governance:legacy-report`：通过，边界违规 0。
- `npm run smoke:agent-runtime-current-fixture`：通过，真实 Electron/App Server/IPC current fixture 全套通过，`liveProviderUsed=false`。
- `npm run verify:gui-smoke`：通过，Electron shell/app-server smoke pass。
- `git diff --check`：通过。

未通过或未完成：

- `npm run test:related -- ...` 在两次单独尝试中均被仓库 runner 的 `EISDIR: read electron` 退出错误阻断；精确 Vitest 集合已通过，related runner 问题未修改生产代码。
- 尚未有 unknown Item 专项 Gate B fixture；当前 Gate B 证明的是既有真实 runtime/rendering 主链未回归，不等价于未知协议 drift 的跨进程证据。
- 尚未建立 protocol revision/method/type drift recorder、统一 ConversationProjection reducer、live/cold/replay 等价 fixture。

治理分类：

- `current`：Codex-aligned `unknown_item` safety projection、既有 ContextCompaction renderer、五语言文案与 current Electron/App Server 主链。
- `compat`：无新增长期生产 compat；旧输入仅保留测试边界的 fail-closed 断言。
- `deprecated`：未新增；现有 `thread/compacted`、legacy approval/output surface 仍按 EVENT-PROJECTIONS 裁决。
- `dead / deleted`：本轮删除 unknown Item `null` drop 与 ContextCompaction hide 分支；Item -> Message 二次合成等重复主链仍待后续切片删除。

下一刀：新建唯一 `ConversationProjection`/`ItemProjection` contract owner，先把 live notification、thread/read、replay 三个入口统一到同一 reducer，再处理 declined/interrupted 状态和 orphan/late delta 语义。
