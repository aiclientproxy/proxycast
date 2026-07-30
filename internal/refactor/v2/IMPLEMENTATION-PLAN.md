# v2 渲染对齐执行计划

状态：in-progress

主目标：以 Codex 的 Thread / Turn / Item 生命周期和渲染语义替换 Lime 当前把 canonical Item 二次压缩为 Message、tool_call、extension 的对话渲染路径，同时保持 model-provider 的 Grok 多模型控制面和 OpenCode 多模态/多协议 lowering 不变。

当前阶段：V2-04 剩余产品范围交互与 V2-05 notification/host capability/recovery；V2-00、V2-01、V2-02、V2-03 已关闭，direct TurnTimeline、production replay、session history、长列表性能、MCP elicitation 与 media read v2 Gate B 已通过

下一刀：完成 V2-04 剩余 product-scope reverse request / multi-agent 场景；随后按 EVENT-PROJECTIONS 逐项推进 V2-05 尚未实现的 notification、host capability 与 recovery，不恢复 Message synthesis、extension fallback、v0 media read 或第二 pending store。

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

状态：completed（2026-07-29；18/72/11 coverage baseline、pinned upstream schema/hash/method drift gate 与运行时 notification drift recorder 已落地）

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

状态：completed（2026-07-29；live、cold read 与 production `thread/resume` replay 复用同一 reducer，canonical Item -> Message 合成已删除）

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

状态：completed（2026-07-29；declined/interrupted、MCP typed result/error、DynamicTool `inputAudio`、路径安全展示与 thread-scoped media read 异常态门禁已落地）

目标：补齐对渲染不可缺少、但目前被 String 或 opaque Value 模糊化的 Item 字段，保持 provider 细节留在 model-provider。

建议写集：

- lime-rs/crates/app-server-protocol/src/protocol/v2/item.rs
- lime-rs/crates/app-server-protocol/src/protocol/v2/envelopes.rs
- schema/json、generated client、packages/app-server-client
- lime-rs/crates/app-server/src/runtime/thread_item_projection/\*\*
- Electron host/preload 的 media/path/open semantic gateway

动作：

1. 将 MCP result/error、媒体 reference 与 DynamicToolCall text/image/audio output 形成 typed、size-bounded、脱敏的 current contract。Lime 当前 UserMessage 产品协议不包含 audio/localAudio，reader 必须 fail closed，不能保留半实现前端类型。
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

状态：completed（2026-07-29；direct canonical Turn render projection、User/Agent/Media/Process segment、生产 MessageList 接管、session history Gate B、240 Turn / 720 Item 长列表性能 Gate B 与整合门禁已完成）

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
- 受控 Electron Gate B 证明首帧 Turn/Item 挂载有界、canonical 长正文默认不进入完整 DOM，且 long task、console error、page error 为 0。

### V2-04：工具、阻塞交互与多 Agent

状态：in-progress（四类 reverse request 已统一到单一 PendingInteractionController 与 Composer 上方交互层，MCP elicitation Gate B 已关闭；剩余 product-scope reverse request 与 multi-agent 场景仍待逐项证明）

目标：补 MCP、DynamicTool、collaboration、Hook 与所有 current reverse request 的唯一交互层。

建议写集：

- src/components/agent/chat/components/item-renderers/\*\*
- src/components/agent/chat/components/PendingInteractionLayer/\*\*
- src/lib/api/appServerServerRequest.ts 及对应 event bus
- electron/appServerHost.ts、preload/gateway 与 contract tests
- lime-rs/crates/app-server/src/processor/\*\*，仅当前 request producer 接线

动作：

1. 渲染 MCP arguments/progress/content/structuredContent/error，DynamicTool text/image/audio 输出，CollabAgentToolCall 与 SubAgentActivity。
2. 将 command/file approval、requestUserInput、MCP elicitation 统一为 PendingInteractionProjection；复用 DecisionPanel/McpServerElicitationForm 作为纯表单内容，不保持第二 pending store。
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
3. 维护已落地的 unknown notification drift recorder：known-but-excluded 发 DX，unknown notification fail visible；unknown reverse request fail closed。
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

| 分类           | v2 裁决                                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| current        | App Server v2 主链、ConversationProjection、direct TurnTimeline、bounded restored Turn window、canonical long-message preview、production replay、统一 PendingInteraction、thread-scoped media read、model-provider 多模型/多模态 owner |
| compat         | 不保留长期生产 compat；仅可存在一次性迁移测试夹具                                                                                                |
| deprecated     | fileChange outputDelta、thread/compacted、`media.read.chunk/completed` transient bypass 与尚未完成 current producer 的旧 notification 裁决；只允许迁出 |
| dead / deleted | canonical Item -> Message tool/agent/reasoning 合成、首帧无界历史挂载、canonical 长正文绕过 preview、unknown Item null drop、ContextCompaction hide、通用 extension fallback、重复 pending store、v0 media read |

计划完成度：保守估算 82%。该数字按 V2-00 至 V2-05 六个阶段退出条件等权折算：V2-00、V2-01、V2-02、V2-03 已关闭；V2-04 已关闭 MCP elicitation Gate B 但剩余产品范围交互与 multi-agent 证据未齐；V2-05 的大量 planned notification、host capability 与全面 recovery 仍未完成。该估算不表示 v2 已可整体交付或 release-ready。

## 6. 执行台账

### 2026-07-28：V2-00 coverage baseline 与 V2-01 首切片

状态：已完成 coverage baseline 切片；V2-00 的真实 upstream revision drift 退出条件仍为 `in-progress`。

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

下一刀：把 `ConversationProjection` live adapter 接入会话状态 owner，迁移并删除首个 `Item -> Message` 合成生产入口；随后补 protocol revision/method/type drift recorder 与 declined/interrupted 语义。

### 2026-07-28：V2-01 reducer 首切片

状态：已完成本轮 reducer 切片；总体仍为 `in-progress`。

本轮实际写集：

- `src/lib/api/agentRuntime/conversationProjection/{contracts,reducer,adapters,index}.ts` 与 reducer 回归。
- `src/lib/api/agentRuntime/appServerCanonicalThreadProjection.ts`。
- 本计划台账。

本轮完成：

- 新建唯一 typed `ConversationProjection` contract：Thread、Turn、Item、PendingInteraction、Notice、diagnostic 与 source 均有明确类型。
- reducer 以 `thread_id + turn_id + item.id` 为 Item identity，按 canonical sequence 保序；重复 event id 不改变 projection。
- started 前 delta 进入有界 orphan buffer；started 后回放；completed snapshot 覆盖草稿；terminal 后 late delta fail visible 并记录 diagnostic。
- `thread/read` 的 canonical Item 已实际通过 reducer 产出，不能再在该入口依赖循环 push 的临时顺序。
- live notification payload 与 replay fixture 均通过同一个 typed adapter 进入同一 reducer；Unit/typed fixture 证明 live、thread/read、replay 样本的 Item 投影和 completed Turn 等价，不构成 Gate A 或 production replay consumer 证据。

实际验证：

- `npx vitest run src/lib/api/agentRuntime/conversationProjection/reducer.test.ts src/lib/api/agentRuntime/appServerCanonicalThreadProjection.test.ts src/lib/api/agentRuntime/appServerV2Notification.test.ts`：38 tests passed。
- `npm run typecheck`：通过。
- `git diff --check`：通过（执行时）。
- `npm run verify:local`：该切片未取得最终结果；当前没有可复用的运行中进程，不得标记为通过。

未完成：

- 当前 live adapter 仍由 fixture/调用方显式使用，尚未接管 `useAgentStream` 的会话状态 owner；因此 Renderer 主路径仍保留旧 `AgentThreadItem -> Message` 合成。
- 仍缺 unknown Item 专项真实 Gate B fixture、protocol revision/method/type drift recorder、declined/interrupted 终态和 pending interaction 收口。

治理分类：

- `current`：ConversationProjection contract/reducer、canonical thread/read reducer 接线、既有 App Server v2 notification 投影与 model-provider 多模型/多模态 owner。
- `compat`：无新增生产 compat；replay adapter 仅是 source 标记，不提供旧 API wrapper。
- `deprecated`：旧 Message synthesis 仅作为 V2-03 已记录的迁出目标，本轮未扩展它。
- `dead / deleted`：本轮删除 thread/read 内 Item 的直接 append 顺序 owner；尚未达到删除 Renderer Message synthesis 的条件。

### 2026-07-29：V2-01 live owner、输出边界与首个 Message synthesis 删除

状态：本轮实现和 Unit/typed fixture 定向验证完成；V2-01 与总体计划仍为 `in-progress`。

本轮实际写集：

- `src/lib/api/agentRuntime/conversationProjection/**`、`appServerV2Notification.ts`、`appServerCanonicalThreadProjection.ts`。
- `src/lib/api/agentRuntime/canonicalThreadHistoryWindow.ts`、`appServerSessionClient.ts`、`clientFactory.test.ts` 与分页/恢复回归。
- `src/lib/api/agentProtocol{.d.ts,EventTypes.ts,ParserUtils.ts}` 与 protocol envelope 回归。
- `src/components/agent/chat/hooks/agentStream{ConversationProjection,RuntimeHandler,RuntimeHandlerTypes,RuntimeLifecycleEvents}*`。
- `src/components/agent/chat/hooks/agentRuntimeAdapter*`、`sessionHydrationController.ts` 与历史窗口合并回归。
- `src/components/settings-v2/system/developer/index.tsx` 与既有设置页 timer lifecycle 回归。
- `scripts/check-app-server-client-contract.mjs`、`src/lib/governance/legacySurfaceCatalog.{json,test.ts}`。
- `internal/refactor/v2/**`、`internal/aiprompts/architecture.md` 与旧研究计划导航纠偏。

本轮完成：

- 每个 stream request 持有一个 ConversationProjection reducer owner；live Item snapshot/delta 回写 canonical ThreadItem state。direct v2 无 event id 时，在 router 后分配 request 内到达序号，不做内容 hash 去重。
- live owner 不再以 Renderer session id 预绑定 canonical thread id；existing canonical Item 或首个带明确 direct-v2 `protocol_method` 的事件才能建立 owner，跨 Thread 事件 fail closed。无 `protocol_method` 的 compat 事件不能抢占 owner，仍由 legacy lifecycle 写入原有 ThreadItem 状态。
- direct `item/started -> item/commandExecution/outputDelta* -> item/completed` 经过 production-shape typed notification fixture 与 typed adapter；多个 delta 累积到同一 Item，completed snapshot 权威覆盖草稿。该证据不冒充 App Server/Electron 跨进程证据。
- CommandExecution、Tool、WebSearch、Patch stdout/stderr 的 projection 输出限制为 256 KiB，使用尾部保留和显式截断标记。
- unknown Item 记录一次 revision/method/type/identity/脱敏字段名 drift，并写入现有 conversation diagnostics store；该切片当时尚未接 notification recorder，后续已由本计划末尾的整合切片关闭。
- cold read history window 改由 current `thread/items/list` 与 `thread/turns/list` 分页读取；opaque canonical Item cursor 优先于旧数字 message cursor 和 offset，`historyMode=paginated` 的 embedded turns 不再冒充完整事实。Turn 分页扫描到 EOF，后续页 active/failed/interrupted 等无 Item Turn 仍保留。
- 未到 Item EOF 时 cursor 以 `has_more=true` 表达继续加载，`messages_count` 保持未知且不生成 `start_index`；只有 EOF 后才返回精确总数和绝对索引。GUI 对未知总数显示“仍有更早历史”，不把观察下界冒充总数。
- `AgentEvent` parser、公开类型与 `.d.ts` 保留 `protocol_method` / `protocol_revision`，使 current/compat 分流和 drift diagnostic 不依赖 raw payload 旁路。
- session detail 缓存 canonical thread identity；并发详情只在已包含 `thread_id` 时复用给 read-model 解析，不含 identity 的详情不得吞掉后续分页响应。App Server client 与 Renderer adapter 均要求 requested id 等于响应 `sessionId` 或 `threadId`，错配响应 fail closed 且不得污染 identity cache。
- 物理删除 `agentStreamToolItemMessageSync.ts` 与专属测试；Message 不再接收 canonical Tool Item 的 `toolCalls/contentParts` 二次合成，工具顺序由 ThreadItem canonical sequence 证明。
- contracts 和 legacy catalog 已从旧正向存在性断言改为 current projection 接线与 `dead / deleted / forbidden-to-restore` 负向守卫。
- 全量前端测试暴露的 Developer Settings 2.5 秒卸载后 state 回写已修复：所有短暂消息共享单一 timer owner，新消息取消旧 timer，组件卸载时清理。

架构确认：

- 架构影响：major；Renderer live Item state owner 改为 request-scoped ConversationProjection，Electron、App Server、ThreadStore、model-provider 与 tool-runtime 的 owner 边界保持不变。
- 架构图更新：已同步 `internal/aiprompts/architecture.md` 的 Renderer ConversationProjection 数据流和 replay fixture 边界。
- 责任人：root，2026-07-29。
- 确认内容：已核对 live/read 生产依赖方向、replay 仅为 fixture、completed snapshot 权威性、256 KiB UTF-8 输出边界、unknown drift 脱敏和旧 tool-specific Item -> Message 回流守卫；真实 PR 仍需在 PR body 复述并由 committed diff 门禁复核。

已通过：

- `npx vitest run src/components/agent/chat/hooks/useAgentChat.test.tsx`：185 tests passed，覆盖历史窗口合并、live timeline、terminal refresh 与 session identity 缓存。
- `npx vitest run src/components/agent/chat/hooks/agentRuntimeAdapter.test.ts src/components/agent/chat/hooks/agentStreamRuntimeLifecycleEvents.unit.test.ts src/components/agent/chat/hooks/agentStreamRuntimeHandler.unit.test.ts src/components/agent/chat/hooks/agentStreamRuntimeHandler.test.ts src/components/agent/chat/hooks/agentStreamTurnEventBinding.test.ts src/lib/api/agentRuntime/conversationProjection/reducer.test.ts src/lib/api/agentRuntime/appServerCanonicalThreadProjection.test.ts src/lib/api/agentRuntime/appServerV2Notification.test.ts`：8 files / 152 tests passed，覆盖 current/compat Item 分流、跨 Thread fail-closed、direct command bounded output、projection 与 terminal binding。
- `npx vitest run src/lib/api/agentRuntime/appServerSessionClient.test.ts src/components/agent/chat/hooks/sessionHistoryPaginationController.test.ts src/components/agent/chat/components/MessageList.historyWindow.test.tsx src/components/agent/chat/hooks/agentSessionRestoreViewModel.unit.test.ts src/i18n/__tests__/loadNamespace.test.ts src/components/settings-v2/system/developer/index.test.tsx`：6 files / 79 tests passed，覆盖 opaque Item cursor、Turn 后续页空终态、未知总数、双层 identity fail-closed、GUI 文案与 timer lifecycle。
- `npm run typecheck`。
- `node scripts/check-app-server-client-contract.mjs`：284 checks passed。
- `src/lib/governance/legacySurfaceCatalog.test.ts`：223 tests passed。
- `npm run test:contracts`。
- `npm run governance:legacy-report`：边界违规 0；仅报告既有分类漂移 `rust-app-paths-root-fetch-duplication`。

已补充通过：

- `npm run test:resume`：状态文件 `.lime/test/vitest-smart-last-run.json` 为 `passed`，112/112 批全部通过。
- `npx vitest run src/lib/api/agentRuntime/appServerCanonicalItemReader.test.ts src/lib/api/agentRuntime/appServerCanonicalThreadProjection.test.ts src/lib/api/agentRuntime/appServerV2Notification.test.ts src/lib/api/agentRuntime/appServerEventTimelineReaders.unit.test.ts src/components/agent/chat/components/timeline-utils/itemConverters.unit.test.ts`：5 files / 80 tests passed。
- `npm run typecheck`。
- `npm run test:contracts`：通过，包含 generated protocol 无漂移、App Server client 284 checks、command/modality/scripts/release/docs boundary。
- `npm run governance:legacy-report`：边界违规 0；仅报告既有分类漂移 `rust-app-paths-root-fetch-duplication`。
- `git diff --check`。

本轮状态语义收口：

- direct v2 与 compat Turn reader 均保留 canonical `interrupted`，仍使用 `turn_canceled` 事件类型触发现有终止副作用，不再把业务状态降为 `canceled`。
- CommandExecution/FileChange 的 `declined` 生命周期投影为 `completed`；业务结果分别保留在 typed `command_status: declined` 与 `file_status: declined`，不得伪装成执行失败。
- declined command/patch 的 display result 保持 `success: false`，同时保留业务状态 metadata，避免 completed lifecycle 被误显示成成功。

待本轮后续门禁：

- fresh `npm run verify:local`。此前 fresh 运行只通过静态阶段，在全量测试开始后被主动中断，不能记为本切片成功；112/112 resume 只证明前端全量批次。
- `npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke`。
- unknown Item 专项真实 Electron Gate B；当前通用 smoke 不能替代该证据。

治理分类：

- `current`：ConversationProjection contract/reducer/sanitizer、live request owner、thread/read 生产接线、replay adapter、App Server v2 notification 主链；该切片当时未接 production replay，后续已由 `thread/resume` 接线关闭。
- `compat`：无新增生产 compat。
- `deprecated`：该切片当时仍存在的 `canonicalItemsToMessages`、正文/Reasoning Message content part 同步和 MessageList compatibility 分支；后续 direct TurnTimeline 接管后均已删除。
- `dead / deleted`：tool-specific canonical Item -> Message toolCalls/contentParts synthesis 模块与正向专属测试，已有物理缺失和生产符号负向守卫。

V2-03 历史只读审计结论（后续 direct TurnTimeline 切片已关闭）：

- 当时 `MessageList`、render window 与 pagination 仍以 Message 为顶层锚点，直接删除 `canonicalItemsToMessages` 会导致冷历史空白、loaded count 错误或重复分页。
- 最小垂直切片是从现有 `threadTurns/threadItems` 纯派生 canonical Turn render entries：User/Agent Message Item 独立成段，连续非消息 Item 作为 process 段复用 `AgentThreadTimeline`；尚无 canonical Item 覆盖的 optimistic/imported/local Message 保留 residual Message 路径。
- 后续已按该顺序让 direct Turn 生效，并物理删除 cold process/reasoning、live AgentMessage/Reasoning 同步与 `canonicalItemsToMessages`。
- 本轮不把迁移期 direct/residual 选择做成第二个 store；它只能是 ConversationProjection state 上的纯派生 render projection。

### 2026-07-29：direct timeline、production replay、pending interaction 与特殊 Item 收口

状态：实现与定向验证完成；当时整合门禁进行中，后续已由下一台账关闭；v2 总体仍为 `in-progress`。

本轮完成：

- `MessageList` 通过纯派生 `turnTimelineRenderProjection` 直接渲染 canonical Turn；UserMessage、AgentMessage、Media 与连续 Process segment 保持 canonical sequence，未被 canonical Item 覆盖的 optimistic/imported/local surface 才保留 residual Message。
- 物理删除 canonical Item 到 tool/agent/reasoning Message content part 的生产合成入口；`canonicalItemsToMessages`、旧 agent/reasoning sync 与 tool-specific synthesis 不再是生产符号。
- `thread/resume` 使用 canonical reader 创建 replay reducer，安装到 active stream state；后续 live notification 继续复用同一 reducer，production replay 不再只是 fixture。
- command approval、file approval、requestUserInput 与 MCP elicitation 统一到 `PendingInteractionController`；JSON-RPC action token 只由 dispatcher 请求闭包持有，GUI 只消费 semantic interaction identity，唯一可操作表面位于 Composer 上方。
- HookPrompt、Sleep、entered/exited Review 从通用 extension 改为 typed `hook_prompt`、`sleep`、`review_boundary`，并有低干扰时间线 renderer；Hook 不向 DOM 暴露 `hookRunId`。
- MCP result/error 使用 typed shape，路径采用安全展示，DynamicTool 保留真实协议 `inputAudio`。UserMessage audio/localAudio 因 Lime Rust `UserInput/AgentInput` 无产品 owner，删除前端半实现并补 fail-closed 负向回归。
- V2-00 pinned upstream schema/hash/method drift gate 与 V2-05 unknown notification drift recorder 已存在；后者只证明未知/未投影 method 可诊断，不能把 72 个 notification 的 planned surface 视为完成。

架构确认：

- 架构影响：major；Renderer read/live/resume 与 pending interaction owner 均发生收敛，但 Electron 仍只做 Desktop Host，App Server/RuntimeCore/ThreadStore 的事实源不变。
- 架构图更新：已同步 `internal/aiprompts/architecture.md` 的 live/read/resume reducer、direct TurnTimeline 与 PendingInteraction 数据流。
- 责任人确认：root，2026-07-29。确认未引入第二 store、compat wrapper、生产 mock fallback 或 raw request id Renderer owner。

定向验证：

- canonical reader + timeline：2 files / 57 tests passed。
- production replay：4 files / 45 tests passed；stream projection/runtime handler 扩圈 2 files / 63 tests passed。
- pending interaction：7 files / 227 tests passed；controller 修复后 5/5 再次通过。
- `npm run typecheck` 与 `git diff --check` 通过（各 owner 交接时）。

当时剩余退出条件（后续已由下一台账关闭）：

- 重建 Renderer 后完成 contracts、current runtime fixture、session history Gate B、MCP elicitation Gate B、GUI smoke、fresh `verify:local` 与 `test:resume`。
- V2-05 的 planned notification/host capability 仍须逐项从 producer、protocol、gateway、projection 到 Gate B 实现；不得以 drift warning 冒充产品能力。

### 2026-07-29：Renderer 整合 Gate B 与完整本地门禁关闭

状态：本轮整合门禁已关闭；v2 总体仍为 `in-progress`，不能标记整体完成或 release-ready。

本轮事实收口：

- session detail 必须先请求 canonical `thread/read`；`thread/list` 只提供摘要，不能冒充详情。详情响应继续直接消费 canonical `items`，兼容字段保持 `messages: []`，不恢复 Item -> Message 合成。
- AgentMessage terminal phase 使用 current `final_answer`；旧 `phase: "final"` 不再作为正向 fixture 或生产语义。
- session history Electron fixture 证明真实 preload、archive/unarchive、`thread/read`、turn/item 分页、`thread/resume` 与 GUI 使用同一 thread/turn/item identity；三组 Turn、九个 Item 顺序稳定，图片附件去重，console/page errors 为空。
- MCP elicitation Gate B 证明真实 Electron preload、`app_server_handle_json_lines`、runtime capability advertisement、Renderer 表单提交/关闭、MCP ledger 接受和 provider final text；未命中旧 MCP Desktop command，console errors 为空。
- MCP/session-history 守卫改为格式无关匹配，避免格式化工具展开数组时产生误报；相关守卫 10/10 通过。

整合验证：

- `npm run test:contracts`：通过。
- `npm run smoke:mcp-elicitation-gate-b`：通过；证据为 `.lime/qc/gui-evidence/mcp-elicitation-gate-b/mcp-elicitation-gate-b-summary.json`，`ok=true`、`proofLevel=Gate B`。
- session history Electron 证据为 `.lime/qc/gui-evidence/agent-session-history-electron-fixture/agent-session-history-electron-fixture-summary.json`，`ok=true`、`electronPreloadBridge=true`、console/page errors 为空。
- `npm run verify:gui-smoke`：通过。
- `npm run test:resume`：112/112 批完成，状态为 `passed`。
- fresh `npm run verify:local`：通过 i18n、lint、typecheck、112/112 Vitest、Rust changed-scope、Electron renderer/host build 与真实 GUI smoke。
- `src/lib/governance/legacySurfaceCatalog.test.ts`：223/223 通过；`npm run governance:legacy-report` 为零引用候选 0、分类漂移 0、边界违规 0。
- `npm run governance:scripts` 与 `git diff --check`：通过。

治理分类：

- `current`：canonical session detail、direct Item timeline、production replay、PendingInteraction MCP elicitation 与真实 Electron/App Server Gate B。
- `compat`：无。
- `deprecated`：fileChange outputDelta、thread/compacted、未迁完的旧 notification 裁决，以及尚未完成 producer/Gate B 的 product-scope reverse request。
- `dead / deleted / forbidden-to-restore`：canonical Item -> Message 合成、旧 `phase: "final"`、重复 pending/dialog owner、旧 MCP Desktop commands、app_paths 重复 root 获取样板。

OPEN_REF：V2-02 media read v2 与异常态门禁已关闭；下一刀回到 V2-03 长列表性能证据、V2-04 剩余 product-scope reverse request / multi-agent 场景，并按 EVENT-PROJECTIONS 推进 V2-05 planned notification、host capability 与 recovery。unknown notification drift recorder 只保留诊断职责，不能作为上述能力已实现的证据。

### 2026-07-29：V2-02 media read v2 与异常态门禁关闭

状态：completed；v2 总体仍为 `in-progress`，不得标记整体完成或 release-ready。

本轮完成：

- GUI sidecar 读取从 v0 `agentSession/media/read` 直接迁到 Lime-owned v2 `media/read`；Rust、schema、generated/package client 与 Renderer 统一使用 canonical `threadId`，未增加 compat、第二 reader 或生产 mock fallback。
- v0 method/type/client/build symbols 已从生产面物理删除，contract guard 保留唯一负向引用；生产目录扫描命中 0。
- SidecarStore 继续约束 Thread scope、range、max-bytes 与 digest。读取缺失或权限拒绝保留脱敏 metadata fallback；图片、音频、视频浏览器解码失败切换到共享 `unsupported` 兜底面；object URL 按替换、卸载和预算淘汰释放。
- controlled Electron Gate B 读取 471-byte PNG，两次 `media/read` 只带 `threadId`、无 `sessionId`；sidecar 置空后 GUI 展示 Markdown metadata fallback。legacy/mock 命中 0，console/page error 0，截图与 summary/ledger 同目录。
- Gate B 只证明图片成功与 sidecar 不可用；权限拒绝、图片/音频格式失败、过大、range 与 digest 由最近边界的 Rust/TS 定向测试覆盖，不扩张为所有音频、视频或平台异常的真实 Electron 证据。

验证：

- `npm run smoke:agent-runtime-current-fixture`：通过；媒体 Gate B evidence 为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-media-read-v2-gate-b-summary.json` 与同目录 PNG。
- fresh `npm run verify:local`：通过 i18n、lint、typecheck、112/112 Vitest、contracts、Rust changed-scope、Electron renderer/host build 与真实 GUI smoke。
- 最终异常态定向回归：2 files / 29 tests passed；定向 ESLint 与 `npm run typecheck` 通过。
- `npm run governance:legacy-report`：零引用候选 0、分类漂移 0、边界违规 0；`npm run governance:scripts` 与 `git diff --check` 通过。

治理分类：`media/read`、App Server/SidecarStore、typed client 与 fail-visible Renderer 为 current；无 compat；`media.read.chunk` / `media.read.completed` 继续为 deprecated/V2-05 transient bypass；v0 media read method/type/client symbols 为 dead/deleted/forbidden-to-restore。

### 2026-07-29：V2-03 bounded TurnTimeline 与长列表 Gate B 关闭

状态：completed；v2 总体仍为 `in-progress`，不得标记整体完成或 release-ready。

本轮完成：

- canonical 历史继续直接进入 `ConversationTurnTimeline`，首帧由现有 Turn render window 限定；240 Turn / 720 Item fixture 最终只扫描 30 个 Item、挂载 10 个 canonical Turn，residual Message 为 0。
- 已完成的历史 assistant 纯文本超过 900 字进入 compact preview，超过 24,000 字使用 2,000 字 long preview；streaming、A2UI 与非文本 part 不折叠，用户可显式展开全文。
- history chrome 补稳定 evidence attributes；长正文 tail marker 在展开前不进入 DOM，避免 canonical 正文恢复后立即触发完整 Markdown 挂载。
- 受控 Gate B 的首次 MessageList paint 为 37ms、稳定 paint 为 200ms，long task、console error、page error 均为 0；历史读取沿用 current `thread/read`、`thread/items/list`、`thread/turns/list`。

验证：

- direct timeline 与历史预览扩圈：5 files / 61 tests passed；定向 ESLint、`npm run typecheck` 通过。
- `npm run smoke:agent-session-history-electron-fixture`：通过；证据为 `.lime/qc/gui-evidence/agent-session-history-electron-fixture/agent-session-history-electron-fixture-summary.json` 与同目录 long-list PNG。
- `npm run smoke:agent-runtime-current-fixture`：完整通过，`liveProviderUsed=false`。
- 独立 `npm run verify:gui-smoke`：通过；证据为 `.lime/qc/project-gates/standalone-shell-01-20260729161109-71996/shell-01-electron-smoke/summary.json`。
- fresh `npm run verify:local`：通过 i18n、lint、typecheck、112/112 Vitest、contracts、scripts/docs governance、Rust changed-scope 与第二次真实 Electron GUI smoke；后者证据为 `.lime/qc/project-gates/standalone-shell-01-20260729163104-57139/shell-01-electron-smoke/summary.json`。
- 最终 `npm run governance:legacy-report`：零引用候选 0、分类漂移 0、边界违规 0；`npm run test:contracts` 与 `git diff --check` 再次通过。

证据边界：长列表 Gate B 使用 `APP_SERVER_BACKEND_MODE=unavailable` 的 controlled fixture，不调用 live Provider，也不冒充真实用户历史或所有平台的性能结论。

治理分类：direct canonical TurnTimeline、bounded restored Turn window、canonical long-message preview 与 Electron long-list fixture 为 `current`；无 `compat`；V2-05 planned notification/transient bypass 继续为 `deprecated`；canonical Item -> Message 合成、首帧无界历史挂载和 canonical 长正文绕过 preview 为 `dead / deleted / forbidden-to-restore`。

下一刀：进入 V2-04 剩余 product-scope reverse request / multi-agent 场景，再推进 V2-05 notification、host capability 与 recovery。总体完成度保守估算约 82%。
