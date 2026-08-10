# v2 渲染对齐执行计划

状态：in-progress

主目标：以 Codex 的 Thread / Turn / Item 生命周期和渲染语义替换 Lime 当前把 canonical Item 二次压缩为 Message、tool_call、extension 的对话渲染路径，同时保持 model-provider 的 Grok 多模型控制面和 OpenCode 多模态/多协议 lowering 不变。

当前阶段：V2-05 notification、host capability 与 recovery；V2-00、V2-01、V2-02、V2-03、V2-04 已关闭，direct TurnTimeline、production replay、session history、长列表性能、MCP elicitation、Multi-Agent、三项 product-scope reverse request、media read v2、unknown Item fail-visible recovery、`skills/changed` catalog invalidation、typed `error` retry/terminal、`turn/plan/updated` checklist、`mcpServer/oauthLogin/completed`、`mcpServer/startupStatus/updated` 与 unified exec terminal interaction Gate B 已通过

下一刀：V2-05 已关闭 media transient bypass、unknown Item fail-visible recovery、`configWarning` owner 迁移、thread-scoped `warning` typed/recovery、`skills/changed` catalog invalidation、typed `error` retry/terminal、`turn/plan/updated` checklist、MCP OAuth completion、MCP startup status、unified exec terminal interaction、Hook lifecycle、`turn/diff/updated` typed/recovery、Guardian auto-approval review typed/recovery 与 Guardian denial circuit breaker `guardianWarning` 主链；standalone `process/*` 仍为 product-scope-excluded。继续审计具备真实 producer/consumer 的 remaining planned notification、host capability 或 recovery。不重复改写已关闭 owner，也不恢复 raw unified diff、unsandboxed process/spawn、Message synthesis、unknown null drop、extension fallback、v0 media/config owner、旧 MCP Desktop lifecycle event、旧 Team 工具、第二 pending store、第二 Skill catalog owner、Plan ThreadItem、独立 `write_stdin` Tool Item 或由 error 抢占 Turn terminal 的旁路状态机。

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

1. 从 Codex c4f42d16 的 item.rs、envelopes.rs、thread_history.rs 建立 18 Item、72 notification、11 reverse request manifest；TUI history_cell/streaming 只用于语义取证，不作为 Desktop UI 实现目标。
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

状态：completed（全部 product-scope reverse request 已通过定向回归；DynamicTool、MCP elicitation 与 Multi-Agent AgentControl 另有真实 Electron Gate B）

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
4. permission approval 与 item/tool/call 已由 runtime producer 和 Electron Host binding 承接；Renderer 只消费统一 pending 与 canonical Item，不得伪造请求或执行结果。
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

1. 按 EVENT-PROJECTIONS 逐项补 environment、warning、fuzzy search、realtime、Windows sandbox 等仍在产品范围内的 planned surface；Hook 与 `turn/diff/updated` 已有 current owner，standalone `process/*` 仍为 product-scope-excluded，不得恢复为第二套 current。
2. 每项先补 App Server typed protocol 和 producer，再补 Electron gateway、projection、renderer 和 Gate B；不能由前端先造状态。
3. 维护已落地的 unknown notification drift recorder：known-but-excluded 发 DX，unknown notification fail visible；standalone `process/*` 只保留脱敏字段名诊断，不进入 Renderer；current `turn/diff/updated` 走 canonical Turn/Changes，unknown reverse request fail closed。
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

| 分类                   | v2 裁决                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| current                | App Server v2 主链、ConversationProjection、direct TurnTimeline、bounded restored Turn window、canonical long-message preview、production replay、统一 PendingInteraction、thread-scoped media read、typed unknown Item live/cold fail-visible 与专项 Gate B、typed `configWarning` producer/GUI、thread-scoped typed `warning` 与 durable recovery、typed `skills/changed` catalog invalidation 与 GUI 自动重读、typed `error` live/durable recovery 与 Turn terminal ownership、`turn/plan/updated` canonical checklist 与 cold recovery、typed `mcpServer/oauthLogin/completed` 与 GUI 自动刷新、typed `mcpServer/startupStatus/updated` 与 MCP 连接态投影、Hook transient lifecycle、`turn/diff/updated` canonical Turn/Changes、unified exec terminal interaction typed/cold recovery、Guardian auto-approval review lifecycle/pending projection、六工具 AgentControl、canonical SubAgent activity、parent-owned child direct-input policy、`currentTime/read`、`item/permissions/requestApproval`、`item/tool/call`、typed DynamicToolCall、model-provider 多模型/多模态 owner |
| compat                 | 不保留长期生产 compat；仅可存在一次性迁移测试夹具                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| product-scope-excluded | standalone unsandboxed `process/outputDelta` / `process/exited`；只保留 upstream inventory 与脱敏 drift 诊断，不进入 current protocol 或 Renderer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| deprecated             | fileChange outputDelta、thread/compacted 与尚未完成 current producer 的旧 notification 裁决；只允许迁出                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| dead / deleted         | canonical Item -> Message tool/agent/reasoning 合成、首帧无界历史挂载、canonical 长正文绕过 preview、unknown Item null drop、ContextCompaction hide、通用 extension fallback、重复 pending store、v0 media/config notification owner、media transient notification/raw subscription/live-drain 旁路、旧 MCP OAuth/start/stop/error Desktop event、裸旧 Team 工具、raw output 状态推断与生产 mock fallback                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

计划完成度：保守估算 97%。该数字按 V2-00 至 V2-05 六个阶段退出条件与剩余风险折算：V2-00、V2-01、V2-02、V2-03、V2-04 已关闭，V2-05 的 media transient bypass、unknown Item fail-visible recovery、`configWarning` typed owner、thread-scoped `warning` typed/recovery、`skills/changed` catalog invalidation、typed `error` retry/terminal、`turn/plan/updated` canonical checklist、MCP OAuth completion、MCP startup status、Hook lifecycle 与 unified exec terminal interaction、`turn/diff/updated` canonical Turn/Changes 链已关闭；standalone `process/*` 已明确排除出 Lime 产品范围，其余 planned notification、host capability 与全面 recovery 仍未完成。该估算不表示 v2 已可整体交付或 release-ready。

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

- coverage fixture 固化 Codex revision `c4f42d161ae44a8d696ee9fb595709661979d187` 的 18 Item、72 notification、11 reverse request；`model/list/updated` 保持 Lime model-provider 扩展。
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

治理分类：`media/read`、App Server/SidecarStore、typed client 与 fail-visible Renderer 为 current；无 compat；在此历史节点 `media.read.chunk` / `media.read.completed` 仍为 deprecated，已于 2026-07-31 的 V2-05 media transient retirement 切片物理删除；v0 media read method/type/client symbols 为 dead/deleted/forbidden-to-restore。

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

### 2026-07-30：V2-04 Multi-Agent AgentControl Gate B 关闭

状态：Multi-Agent 子切片 `completed`；V2-04 与 v2 总体仍为 `in-progress`，不得标记整体完成或 release-ready。

本轮完成：

- `spawn_agent`、`list_agents`、`send_message`、`followup_task`、`interrupt_agent`、`wait_agent` 六个 canonical AgentControl 工具在 current inventory、provider request、read model 与可见 DOM 中一致；每个工具行只出现一次且均为 `completed`。
- canonical wait 工具投影固定为 `wait_agent`，不再从 raw output 或 `subagent_activity(kind=wait)` 反推工具完成态。Started、Interacted、Interrupted 继续只读 canonical SubAgent Item identity。
- parent-owned child 只读 canonical Thread fact `canAcceptDirectInput=false`。Composer、发送、access mode、model selector 与 task mode 均禁用；五语言专用 placeholder 有稳定断言，UI 尝试不会产生 `turn/start`。
- App Server 在真实冷重启后仍拒绝 parent-owned child 直接输入，返回 `-32600 / direct app-server input is not allowed for parent-owned threads`；该规则不只依赖 Renderer 禁用态。
- Gate B 真实替换 Electron/App Server 进程，旧进程树完全退出；Tool Item、SubAgent activity、child Thread 与 `wait_agent` state identity 跨重启保持一致。

Gate B 证据：

- `.lime/qc/agent-runtime-tool-execution-smoke.json`：`status=pass`、39/39 assertions 全真、`failedAssertions=[]`，proof level 为 Gate B。
- Electron PID `40827 -> 42257`，旧进程树退出后 `remainingPids=[]`；重启后恢复 6 个 Tool row、4 个 SubAgent activity row 和最终 assistant 文本。
- parent-owned child 截图：`.lime/qc/agent-runtime-tool-execution-smoke-parent-owned-child.png`；重启前后截图：`.lime/qc/agent-runtime-tool-execution-smoke-pre-restart-visible-dom.png`、`.lime/qc/agent-runtime-tool-execution-smoke-cold-restart-visible-dom.png`。
- invoke error 与 console error 均为 0。场景使用 localhost OpenAI-compatible provider fixture，证明真实 Electron/preload/IPC/App Server/runtime/read model/GUI 主链，不证明 live Provider。

验证：

- focused Vitest：3 files / 88 tests passed；AgentControl Gate assertion 回归 13/13 passed。
- Rust App Server：1623 个单测及相关 integration targets 通过。
- `npm run build`、`npm run test:contracts`、`npm run governance:legacy-report`、`npm run smoke:agent-control-cold-restart-gate-b`、`npm run smoke:agent-runtime-current-fixture` 与 `npm run verify:gui-smoke` 通过；最新 GUI smoke 证据为 `.lime/qc/project-gates/standalone-shell-01-20260730095250-16113/shell-01-electron-smoke/summary.json`。
- `npm run test:related` 因 runner 把工作树中的 `electron/` 目录误当文件而报 `EISDIR`；随后使用精确 Vitest 文件模式完成 3 files / 88 tests 验证，该 runner 缺陷不冒充通过。

证据文档：`internal/research/refactor/v2/13-evidence/2026-07-30-v2-04-multi-agent-agent-control-gate-b.md`。

治理分类：六工具 AgentControl、typed wait states、canonical SubAgent activity、parent-owned direct-input policy、cold-restart identity 与真实 Electron GUI 为 `current`；无 `compat`；`currentTime/read`、`item/permissions/requestApproval`、`item/tool/call` 仍为 `deprecated / producer-and-Gate-B-pending`；裸旧 Team 工具、raw output 推断与生产 mock fallback 为 `dead / deleted / forbidden-to-restore`。

下一刀：只关闭 V2-04 剩余三项 product-scope reverse request 的真实 producer、PendingInteraction 与 Gate B，不再改写已关闭的 Multi-Agent owner；随后进入 V2-05。总体完成度保守估算约 84%。

### 2026-07-30：V2-04 Host Capabilities 与 Product-Scope Reverse Request 关闭

状态：V2-04 `completed`；v2 总体仍为 `in-progress`，不得标记整体完成或 release-ready。

本轮完成：

- `currentTime/read` 只由 Electron Desktop Host 读取系统时钟并通过唯一 App Server JSON-RPC client 返回；请求绑定 canonical `threadId`，超时、非整数、越界和重复 waiter fail closed，不进入 Thread 时间线，Renderer 无裸时钟 API。
- `item/permissions/requestApproval` 通过 `tool-runtime -> agent-runtime -> App Server server-request -> PendingInteractionController` 传递 typed request/grant；cwd、reason、environment、permission profile diff、session/thread/turn identity 均校验，权限提升、相对路径和重复/迟到响应拒绝。
- `item/tool/call` 由 Electron Host 冻结 `desktop.appInfo` binding，Renderer 无法注入 dynamicTools、伪造 namespace/tool/schema/arguments 或观察执行请求；AppInfo 只读返回 name/version/locale/platform。
- DynamicTool 从可信 session metadata 建立 exact route snapshot，namespace flatten 为 `desktop__appInfo`，与 native/MCP/gateway 重名、deferLoading、非法 schema 和保留名均拒绝；executor 先注册 exact response waiter 再发 `dynamic_tool.requested`。
- canonical `ThreadItemPayload::DynamicToolCall` 显式保存 callId、namespace、tool、原始 JSON arguments、有序 text/image/audio content、success、duration；App Server projection、provider history、read model 与 GUI 均读取 typed payload，不从 metadata 猜测核心字段。

真实 Gate B 证据：

- `.lime/qc/gui-evidence/mcp-elicitation-gate-b/v2-04-host-capabilities-final-summary.json`
- `.lime/qc/gui-evidence/mcp-elicitation-gate-b/v2-04-host-capabilities-final-raw.json`
- `.lime/qc/gui-evidence/mcp-elicitation-gate-b/v2-04-host-capabilities-final.png`
- `ok=true`，proof level `Gate B`；真实 Electron/preload/IPC/`app_server_handle_json_lines`/App Server/runtime/read model/GUI 链路可见，dynamic tool provider result 与 canonical started/completed 可见，request 对 Renderer 隐藏，provider request count=3，console errors、missing methods、legacy MCP commands 均为 0。

验证：

- `cargo check --manifest-path lime-rs/Cargo.toml -p agent-protocol -p lime-agent -p app-server`：通过。
- Dynamic typed 回归：agent-protocol 1/1、lime-agent 3/3、app-server 5/5；permission 回归 agent-runtime 3/3、lime-agent 1/1、tool-runtime 2/2、app-server 1/1；current time app-server 5/5、tool-runtime 2/2。
- `npm run typecheck:electron`：通过；Electron Host Vitest 3 files / 39 tests 通过。
- `npm run test:contracts`：通过（protocol types 无漂移、app-server-client 286 checks、command/harness/modality/scripts/docs/release guards 全部通过）。
- `npm run verify:gui-smoke`：通过；最新证据 `.lime/qc/project-gates/standalone-shell-01-20260730133414-77042/shell-01-electron-smoke/summary.json`。

治理分类：三项 reverse request、Electron host capability、typed DynamicToolCall、permission waiter 与 current-time host 为 `current`；无 `compat`；V2-05 尚未实现的 notification/transient bypass、host capability 与 recovery 继续为 `deprecated / 迁出中`；旧 MCP Desktop command、Renderer 伪造 binding、metadata 核心字段猜测、生产 mock fallback 为 `dead / deleted / forbidden-to-restore`。

架构确认：本轮明确落实既有 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI` 产品链，并新增 canonical DynamicToolCall typed owner；`internal/aiprompts/architecture.md` 已同步第 25 节。责任开发者确认：root，2026-07-30。

下一刀：进入 V2-05 notification、host capability 与 recovery，保持 V2-04 owner 冻结。v2 总体完成度保守估算约 88%。

### 2026-07-31：V2-05 `configWarning` typed notification owner 迁移

状态：该垂直切片 `completed`；V2-05 与 v2 总体仍为 `in-progress`，不得标记 release-ready。

本轮完成：

- 对齐 Codex v2 `TextPosition`、`TextRange`、`ConfigWarningNotification`，把 `configWarning` 注册到 v2 method catalog、`ServerNotification` 双向 JSON-RPC、schema type/registry 与 generated TypeScript。
- initialize 与 turn/start producer 显式改用 v2 notification；现有 Renderer response projection、dedupe toast 和五语言文案保持唯一消费链。
- 物理删除 v0 method constant、DTO、notification variant、catalog entry、schema files 与正向测试；v0 decoder 和 catalog 对该 method fail closed。
- wire method、producer timing 与用户行为未改变，不新增 Electron 业务命令、compat、第二 notification channel 或 mock fallback。

验证：协议 owner 2/2、App Server producer/integration 6/6、schema fixture 1/1、Renderer/client/toast 45/45、`npm run test:contracts` 287 checks 及附属 guards、`npm run governance:legacy-report` 0/0/0 均通过。Rust related 与 GUI smoke 的最终结果记录在主执行台账。

治理分类：v2 typed notification、App Server producer、generated client 与 Renderer toast 为 `current`；无 `compat`；其余 V2-05 planned notification/recovery 仍为迁移开放项；v0 `configWarning` owner 为 `dead / deleted / forbidden-to-restore`。

### 2026-07-31：V2-05 thread-scoped `warning` typed notification 与恢复

状态：该垂直切片 `completed`；V2-05 与 v2 总体仍为 `in-progress`，不得标记 release-ready。

本轮实现：

- Codex 基础 wire 保留 `threadId?: string | null` 与必填 `message`；Lime 以可选 `code` 作为最小产品扩展，承接 `skill_not_available`、`skill_load_failed`、`mention_not_available` 五语言 toast，缺失时仍按 message 展示。
- current `AgentEvent::Warning` 经 durable `runtime.warning`、App Server projector 与 v2 `warning` notification 进入 Renderer `{ type: "warning" }`，不再通过 `agentSession/event`。
- 当前 Agent Chat producer 与 Renderer route 都要求非空 canonical `threadId`；通用 protocol 仍可解码 `threadId: null`，但该全局形态没有伪造 GUI owner 或完成证据。
- full canonical 冷读与 history-limit projection summary 都从同一 durable warning event 派生历史 warning item，并保留 `code`；没有新增第二 persistence 或 canonical Item lifecycle 变体。
- 实时与恢复都只接受精确 message/code shape；畸形 message/code、旧 alias、缺失或不匹配的 thread/turn identity 均 fail closed。raw `agentSession/event` 包装的 `runtime.warning` 被拒绝。`guardianWarning` 因没有 Guardian runtime producer 继续保持 planned。

验证：protocol v2 50/50、schema fixture 1/1、App Server warning 12/12、projection-summary recovery 1/1、Renderer 3 files / 61 tests、generated TypeScript drift check 均通过。`npm run test:contracts` 通过 289 项 app-server-client checks 及全部附属 guards；Rust related 覆盖 19 个 scoped packages、18 个 lib 目标，共 3597 项（3593 passed、4 个既有环境/live 测试 ignored、0 failed）。`npm run smoke:agent-runtime-current-fixture` 完整通过且 `liveProviderUsed=false`；该聚合 smoke 没有普通 warning 专项 DOM 断言，不冒充 warning 专项 Gate B。fresh `npm run verify:local` 通过 i18n、lint、typecheck、112/112 Vitest、contracts、Rust changed-scope、Electron renderer/host build 与真实 GUI smoke；后者 21/21 assertions 通过，证据为 `.lime/qc/project-gates/standalone-shell-01-20260730194114-24572/shell-01-electron-smoke/summary.json`。`cargo fmt --all --check`、定向 Prettier、`git diff --check` 与 `npm run governance:legacy-report` 通过，治理结果为零引用候选 0、分类漂移 0、边界违规 0。

治理分类：v2 `warning` protocol/projector、thread-scoped Renderer toast 与 durable recovery 为 `current`；无 `compat`；raw warning wrapper 为 `dead / forbidden-to-restore`；`guardianWarning` 为 `planned`。`v2_notifications.rs` 已超过 1000 行且主要由 inline tests 构成，本刀只增加路由并把业务 projector 放入独立 `warning.rs`；后续触碰 inline test 区前应迁出专用测试模块。

架构确认：typed notification/recovery owner 已同步 `internal/aiprompts/architecture.md` 第 27 节；唯一产品链保持不变。责任开发者确认：root，2026-07-31。

下一刀：审计 `skills/changed` 与其真实 Skill catalog producer/consumer；只有能形成 typed producer -> GUI refresh -> Gate B 的垂直切片才改为 current。v2 总体完成度保守估算约 91%。

### 2026-07-31：V2-05 `skills/changed` catalog invalidation

状态：该垂直切片 `completed`；V2-05 与 v2 总体仍为 `in-progress`，不得标记 release-ready。

本轮实现：

- v2 protocol、schema、generated client 与 typed decoder 只接受 `skills/changed {}`；默认 Skill roots 的创建、修改、删除以及启动后首次出现都经 10 秒节流 watcher 失效 `lime-skills` snapshot cache 并广播通知。
- 成功的 Lime catalog mutation 在 App Server processor 边界失效同一 cache 并附带 typed notification；失败 mutation 和其他 app 的 mutation 不发通知。
- Composer `useLimeSkills` 的自动加载与 notification refresh 统一读取 current `skill/list`；管理中心 `skillManagement/list` 语义不变，`skillsApi.getRuntimeCatalog()` 复用既有 executable Skill decoder 与纯 catalog projection，不新增第二 catalog owner。
- Renderer 只订阅 typed event bus；命中后记录 console-only `skillsChanged.received` marker 并自动重读，重连/重新挂载仍主动 list。该 notification 是进程级瞬时失效，不写入 Thread/Turn/Item、持久化或 recovery replay。

验证：纯 projection、API、Hook 与 fixture guard 共 4 files / 99 tests 通过；`npm run typecheck`、`npm run test:contracts`（840 schema definitions、832 generated types、0 generation failures、290 app-server-client checks）、`npm run governance:legacy-report`、Prettier、MJS `node --check`、`cargo fmt --all --check` 与 `git diff --check` 通过。Rust workspace lib 汇总为 32 个 suite、4670 passed、5 ignored、0 failed。`npm run test:related` 因仓库既有 runner 把 `electron/` 目录当文件读取而报 `EISDIR`，未记为通过，已由精确 Vitest 覆盖本轮前端回归。

产品证据：`npm run smoke:claw-chat-current-fixture -- --scenario skills-runtime` 通过，proof level 为 `Gate B controlled fixture`。真实 Electron/preload/IPC/App Server 链先产生初始 `skill/list`，随后隔离 HOME 默认 root 新建 `notification-refresh/SKILL.md`，typed `skills/changed` marker 为 1，`skill/list` 自动由 1 次增至 2 次且 transport 为 `electron-ipc`，GUI 新 Skill 可见，手动刷新点击为 0；console/page/actionable errors 与 failed assertions 均为 0。证据：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-summary.json`。

整合复验：聚合 `npm run smoke:agent-runtime-current-fixture` 完整通过且 `liveProviderUsed=false`，包含 `skills/changed`、Inputbar rich restore、Skills Runtime 三入口及其余 current Agent/Electron fixture。首次聚合运行暴露的 Inputbar 失败属于 fixture DOM 定位合同：夹具用展示名 `Capability Report` 大小写敏感匹配 current catalog 的 `capability-report`，因此候选已可见但未点击；生产 CharacterMention 与 catalog identity 无缺陷。夹具现统一折叠大小写、空格、连字符和下划线，并用 source guard 锁定候选与 `@capability-report` badge 的同一等价规则。独立 `inputbar-rich-restore` Gate B 随后通过，text/image/path/skill 在 output-free cancel 后全部恢复，console/page error 与 mock fallback 为 0；证据为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-inputbar-rich-restore-regression-summary.json`。

通用 `npm run verify:gui-smoke` 通过，真实 Electron renderer、App Server sidecar、Claw shell reload 与 memory settings 均就绪；证据为 `.lime/qc/project-gates/standalone-shell-01-20260731043417-63564/shell-01-electron-smoke/summary.json`。fresh `npm run verify:local` 通过版本一致性和 i18n unused 后，在全仓 lint 被非本切片脏文件 `src/components/input-kit/ModelSelector.tsx:432` 的既有 `react-hooks/exhaustive-deps` warning 阻断，因此未记为通过；本切片未改该文件或越界修复。后续独立 `npm run typecheck`、contracts、99 项精确测试、GUI smoke、Rust fmt 与 Gate B 均已取得明确通过结果。

治理分类：v2 `skills/changed` protocol/schema、App Server watcher/mutation producer、`lime-skills` cache invalidation、typed Renderer event bus 与 current `skill/list` GUI refresh 为 `current`；无 `compat`；`skillManagement/list` 仅保留管理中心原语义，不是 Composer catalog owner；持久化/replay、`skills/extraRoots/set`、第二 catalog owner 与生产 mock fallback 均未新增。

架构确认：catalog invalidation owner 已同步 `internal/aiprompts/architecture.md` 第 28 节；唯一产品链保持不变。责任开发者确认：root，2026-07-31。

下一刀：回到 `EVENT-PROJECTIONS.md` 审计下一项具备真实 producer/consumer 的 planned notification、host capability 或 recovery；`guardianWarning` 在 Guardian runtime producer 落地前保持 planned。v2 总体完成度保守估算约 92%。

### 2026-07-31：V2-05 typed `error` retry/terminal closure

状态：该垂直切片 `completed`；V2-05 与 v2 总体仍为 `in-progress`，不得标记 release-ready。

本轮实现：

- v2 protocol、schema、generated client 与 strict Renderer decoder 承接 typed `error`；App Server projector 将 durable `runtime.error` 投影到同一 thread/turn identity，raw `agentSession/event` 不再是 GUI owner。
- `willRetry=true` 只进入可见重试状态，后续权威 `turn/completed(completed)` 清理提示并完成 Turn；一个或多个 retry error 后的 `willRetry=false` 只显示 terminal error，最终失败仍由权威 `turn/completed(failed)` 决定。
- canonical full/limited read model 统一合入 durable runtime warning 与 error。恢复出的 runtime error 保留可见错误证据，但 Turn 保持 running，直到 durable Turn terminal 到达；没有新增第二 persistence、第二 terminal reducer 或 error 合成终态。
- SDK 将 typed error 放在独立 signal 通道，不送入只接受 lifecycle sequence 的 verifier。Gate B assertion 把 provider first-text trace 与 App Server message delta、Renderer output/paint、current trace method、W3C 和 identity 证据拆分；typed-error 场景只豁免本来不会发生的 provider 首文本证据，不能豁免 current 主链证据。

验证：typed-error Renderer/fixture 共 6 files / 94 tests、`packages/app-server-client` 8 files / 98 tests、`packages/agent-runtime-client` 22 tests 通过；Rust canonical runtime error recovery 与 `runtime_error_does_not_preempt_a_later_turn_failed_terminal` 两项定向回归分别通过。`npm run test:contracts` 通过，app-server-client contract 为 292 checks；`npm run smoke:agent-runtime-current-fixture` 完整通过且 `liveProviderUsed=false`；`npm run verify:gui-smoke` 21/21 assertions 通过，证据为 `.lime/qc/project-gates/standalone-shell-01-20260731104641-30858/shell-01-electron-smoke/summary.json`。

产品证据：success 场景经真实 Electron、preload/IPC、`app_server_handle_json_lines`、App Server、RuntimeCore、read model 与 GUI 观察 `error(willRetry=true) -> turn/completed(completed)`，同一 identity、零生产 mock fallback，证据为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-typed-error-retry-success-gateb-02-summary.json`。failure 场景经同一链观察 `error(true)* -> error(false) -> turn/completed(failed)`，pending read model、GUI failed 与最终 read-model/后台 terminal 一致，证据为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-typed-error-retry-failure-gateb-03-summary.json`。两者 `ok=true`、proof level 均为 `Gate B controlled fixture`；聚合回归证据分别为 `claw-chat-current-fixture-typed-error-retry-success-regression-summary.json` 与 `claw-chat-current-fixture-typed-error-retry-failure-regression-summary.json`。

治理分类：v2 typed `error` protocol/projector、durable runtime error recovery、strict Renderer signal 与权威 Turn terminal ownership 为 `current`；无 `compat`；raw error wrapper、provider trace 对 current 主链证据的错误替代和由 error 抢占 Turn terminal 的第二状态机为 `dead / forbidden-to-restore`；其余 V2-05 planned notification、host capability 与全面 recovery 仍为开放项。

架构确认：本切片没有改变 public owner、唯一产品链或依赖方向，只补齐既有 notification/projector/read-model/Renderer 边界的 typed contract 与恢复语义；无需改写 `internal/aiprompts/architecture.md`。责任开发者确认：root，2026-07-31。

下一刀：回到 `EVENT-PROJECTIONS.md` 审计下一项具备真实 producer/consumer 的 planned notification、host capability 或 recovery；`guardianWarning` 在 Guardian runtime producer 落地前保持 planned。v2 总体完成度保守估算约 93%。

### 2026-07-31：V2-05 `turn/plan/updated` canonical checklist 与恢复

状态：该垂直切片 `completed`；V2-05 与 v2 总体仍为 `in-progress`，不得标记 release-ready。

本轮实现：

- App Server 从 canonical `update_plan` `ToolOutput.structured_content` 派生 typed `turn.plan.updated`；Renderer 严格解析该通知并把实时快照投影到唯一 run-control checklist owner。
- canonical cold read 从成功的 `update_plan` 工具项恢复最新有效计划，兼容 App Server 真实 wire 的 `[ { name: "plan", value: "[...]" } ]` 参数形状；空数组可以清空旧 checklist，失败或非法快照不会覆盖上一份有效计划。
- read model 保留 canonical `update_plan` 工具项以支持恢复，但不生成 `ThreadItem.plan`，不从 Message/tool card 合成 checklist，也不展示 Plan UI、decision panel 或 `update_plan` 工具行。

验证：`appServerCanonicalThreadProjection.test.ts` 10/10；`npm run test:contracts` 通过，app-server-client contract 为 292 checks；`npm run typecheck`、`npm run smoke:agent-runtime-current-fixture` 与 `npm run verify:gui-smoke` 均通过。Gate B 证据为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-summary.json`：`scenario=turn-plan-update`、`proofLevel=Gate B controlled fixture`、`methodCount=1`、通知位于事件序列 `planIndex=16` 且 `turn/completed` 位于 `terminalIndex=27`；实时与 reload 后均可见 2 项 checklist，canonical read model `updatePlanToolCount=1`、`planItemCount=0`，Plan block/decision/update_plan tool row 均为 0，Electron IPC、App Server、read model 与 GUI identity 一致，invoke/page/console error 为 0。

治理分类：`turn/plan/updated` typed producer、strict Renderer parser、canonical `update_plan` cold recovery 与 run-control checklist 为 `current`；无 `compat`；`update_plan` 场景下的 Plan ThreadItem、Message/tool card 合成、Plan UI/decision panel/tool card 和 renderer mock fallback 均为 `dead / forbidden-to-restore`，独立 Plan mode 的现役 UI 不在本切片范围内。本切片没有改变 public owner、唯一产品链或依赖方向，无需改写 `internal/aiprompts/architecture.md`。

下一刀：回到 `EVENT-PROJECTIONS.md` 审计下一项具备真实 producer/consumer 的 planned notification、host capability 或 recovery；`guardianWarning` 在 Guardian runtime producer 落地前保持 planned。v2 总体完成度保守估算约 94%。

### 2026-07-31：V2-05 MCP OAuth typed completion notification

状态：该垂直切片 `completed`；V2-05 与 v2 总体仍为 `in-progress`，不得标记 release-ready。

本轮实现：

- Rust MCP OAuth 返回 `McpOAuthLoginHandle` completion future；App Server 等待 callback 成功或失败后发布 `mcpServer/oauthLogin/completed`，payload 固定为必填 `name`、必填可空 `threadId`、`success` 与可选 `error`。
- protocol v2、schema、generated TypeScript 与 app-server-client strict decoder 使用同一 typed contract。Renderer 在任何异步 Desktop listener 注册前同步订阅 App Server event bus，避免快速 callback 被排空，并在完成后自动刷新 `mcpServerStatus/list` 与 `mcpTool/list`。
- 生产路径已删除 `mcp:oauth_completed`；contract guard 禁止该字符串回流 `electron`、`lime-rs`、`packages` 与 `src`。没有新增 compat wrapper、第二 OAuth 状态 owner或生产 mock fallback。

验证：`lime-mcp` OAuth 22 passed、app-server-protocol v2 59 passed、App Server MCP processor 4 passed、app-server-client 101 passed、`src/hooks/useMcp.test.tsx` 9 passed；`npm run test:contracts` 通过，包含 847 schema definitions、839 generated protocol types、0 generation failures 与 292 app-server-client checks。

产品证据：`npm run smoke:mcp-oauth-notification-electron-fixture` 通过，proof level 为 `Gate B`。真实 Electron/preload/IPC、`app_server_handle_json_lines`、App Server OAuth login/event drain、本地 OAuth provider callback 与 Renderer GUI 链全部命中；授权前后状态和 completion toast 均可见，`mcpServer/oauth/login`、`mcpServerStatus/list`、`mcpTool/list` 自动刷新成立，`openExternalUrlHitCount=1`、`electronIpcHitCount=8`、`mockFallbackHitCount=0`、`failedInvokeCount=0`，console/page/invoke errors 均为 0。证据为 `.lime/qc/mcp-oauth-notification/mcp-oauth-notification-fixture-summary.json` 与同目录 PNG。

治理分类：typed protocol/schema、App Server completion producer、strict client decoder、Renderer typed event bus 与自动刷新为 `current`；无 `compat`；`mcp:oauth_completed` 为 `dead / deleted / forbidden-to-restore`。本切片没有改变 public owner、唯一产品链或依赖方向，无需改写 `internal/aiprompts/architecture.md`。责任开发者确认：root，2026-07-31。

下一刀：回到 `EVENT-PROJECTIONS.md` 继续审计具备真实 producer/consumer 的 planned notification、host capability 或 recovery；`guardianWarning` 在 Guardian runtime producer 落地前保持 planned。v2 总体完成度保守估算约 95%，仍非 release-ready。

### 2026-07-31：V2-05 MCP startup status typed notification

状态：该垂直切片 `completed`；V2-05 与 v2 总体仍为 `in-progress`，不得标记 release-ready。

本轮实现：

- protocol v2、schema 与 generated TypeScript 新增 Codex 对齐的 `mcpServer/startupStatus/updated` contract，payload 固定为必填 `threadId`、`name`、`status`、`error`、`failureReason`；status 为 `starting | ready | failed | cancelled`，unknown/missing field fail closed。
- `mcpServer/start` 在调用 RuntimeCore 前发布 `starting`，成功发布 `ready`，失败发布带 error 的 `failed` 后保留原 JSON-RPC error。Renderer 同步订阅 typed event bus：`starting` 投影连接态，终态刷新 `mcpServerStatus/list` 与 `mcpTool/list`，失败同步更新连接错误和全局错误。
- 生产路径删除 `mcp:server_started`、`mcp:server_stopped`、`mcp:server_error`；contract guard 与测试禁止四条 legacy MCP lifecycle event（含已删除的 `mcp:oauth_completed`）回流 `electron`、`lime-rs`、`packages` 与 `src`。资源与工具更新事件保留在各自 current owner，本切片不扩张。

验证：`lime-mcp` 151/151、App Server protocol MCP 6/6、App Server MCP processor 6/6、Hook 10/10、app-server-client 102/102、MCP workspace fixture 3/3；`npm run test:contracts` 通过，包含 850 schema definitions、842 generated protocol types、0 generation failures 与 292 app-server-client checks；`npm run test:rust:related -- <MCP paths>`、`npm run governance:legacy-report`、`npm run governance:scripts` 与 `npm run verify:gui-smoke` 均通过。

产品证据：`npm run smoke:mcp-startup-notification-electron-fixture` 通过，proof level 为 `Gate B`。真实 Electron/preload/IPC、`app_server_handle_json_lines`、App Server event drain、runtime MCP server 与 Settings GUI 全链命中；成功服务器可见 `starting -> ready`，失败服务器可见 `starting -> failed`，自动刷新 `mcpServerStatus/list` 与 `mcpTool/list`。证据中 `appServerHandleJsonLinesHitCount=4`、`appServerDrainEventsHitCount=7`、`electronIpcHitCount=11`、`mockFallbackHitCount=0`、`failedInvokeCount=0`，legacy MCP commands、console/page/invoke errors 均为 0。证据为 `.lime/qc/mcp-startup-notification/mcp-startup-notification-fixture-summary.json` 与同目录 PNG。

治理分类：typed protocol/schema、App Server startup producer、strict client decoder、Renderer typed event bus、GUI 连接态与专用 Gate B 为 `current`；无 `compat`；`mcp:server_started`、`mcp:server_stopped`、`mcp:server_error` 为 `dead / deleted / forbidden-to-restore`。本切片没有改变 public owner、唯一产品链或依赖方向，无需改写 `internal/aiprompts/architecture.md`。责任开发者确认：root，2026-07-31。

下一刀：回到 `EVENT-PROJECTIONS.md` 继续审计具备真实 producer/consumer 的 planned notification、host capability 或 recovery；`guardianWarning` 在 Guardian runtime producer 落地前保持 planned。v2 总体完成度保守估算约 96%，仍非 release-ready。

### 2026-07-31：V2-05 unknown Item fail-visible recovery Gate B

状态：该垂直切片 `completed`；V2-05 与 v2 总体仍为 `in-progress`，不得标记 release-ready。

本轮实现：

- canonical `ThreadItemPayload::Unknown` 与 v2 `ThreadItem::UnknownItem` 只保留 upstream type 和排序、限量、敏感名脱敏后的字段名；raw values 与 event metadata 不进入持久化、read model 或 GUI，unknown 数据不得退回 `Extension` escape hatch。
- App Server materializer、canonical notification、thread projection 与 read model 使用同一 thread/turn/item identity；`item.started -> item.completed -> turn.completed` 后，`thread/read` 仍恢复同一个 completed Unknown。
- Renderer 终态 read-model snapshot 现会把 canonical `thread_items` 合入 direct timeline。此前 snapshot 只更新 `threadRead`，导致 live Unknown 在 terminal refresh 后消失；完成态历史摘要现持续显示同一安全诊断。

验证：定向 Vitest 66/66，unknown fixture/合并/UI 回归 23/23；`npm run typecheck`、`npm run test:contracts`、`npm run governance:scripts`、`npm run governance:legacy-report` 与 `git diff --check` 通过。`npm run smoke:agent-runtime-current-fixture` 聚合门禁完整通过且 `liveProviderUsed=false`；`npm run verify:gui-smoke` 21/21 assertions 通过，证据为 `.lime/qc/project-gates/standalone-shell-01-20260731194839-48452/shell-01-electron-smoke/summary.json`。`npm run test:rust:related -- <Unknown Item paths>` 按反向依赖扩圈并以退出码 0 完成，关键结果包括 agent-protocol 41/41、app-server 1659/1659、app-server-protocol 99/99、lime-mcp 151/151。

产品证据：`npm run smoke:unknown-item-recovery-electron-fixture` 通过，proof level 为 `Gate B controlled fixture`。真实 Electron/preload/IPC、`app_server_handle_json_lines`、App Server external runtime/read model 与 direct TurnTimeline 全链命中；GUI 只有 1 个 Unknown，`futureCapability` 与 `[redacted] / label / opaquePayload / status` 可见，raw values、secret 与 `unknown_item` 内部名不可见；read model 恢复同一 completed Item，production mock fallback 与 console/page/invoke errors 均为 0。证据为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-unknown-item-regression-summary.json`。

治理分类：typed Unknown protocol/canonical payload、App Server materializer/read model、Renderer terminal merge、GUI fail-visible 与专项 Gate B 为 `current`；无 `compat`；unknown Item null drop、raw payload/metadata 透传、Extension fallback 与生产 mock fallback 为 `dead / deleted / forbidden-to-restore`。

架构确认：本切片没有改变 public owner、唯一产品链或依赖方向，只补齐既有 canonical Item -> v2 read model -> direct TurnTimeline 的 typed 安全投影和终态恢复；无需改写 `internal/aiprompts/architecture.md`。责任开发者确认：root，2026-07-31。

下一刀：回到 `EVENT-PROJECTIONS.md` 继续审计具备真实 producer/consumer 的 planned notification、host capability 或 recovery；`guardianWarning` 在 Guardian runtime producer 落地前保持 planned。v2 总体完成度保守估算约 97%，仍非 release-ready。

### 2026-08-01：V2-05 unified exec terminal interaction

状态：该垂直切片 `completed`；V2-05 与 v2 总体仍为 `in-progress`，不得标记 release-ready。

本轮实现：

- `write_stdin` 复用原始 `exec_command` Command Item identity；其 Started 事件不创建第二个 Tool Item，完成后释放 session 绑定，跨 Thread 写入 fail closed。
- App Server 通过 typed `item/commandExecution/terminalInteraction` 投影仅含 `sent N chars` 的脱敏摘要，并写入 canonical `CommandExecution.terminalInteractions`；原始 stdin 不进入通知、持久化、read model 或 GUI。
- Renderer live timeline、historical direct timeline 与 reload recovery 消费同一 canonical 字段；Electron fixture 先以 current `thread/resume` 建立订阅，再从真实 `app_server_drain_events` 等待后台 typed notification，不修改生产协议。

验证：tool-runtime 跨 Thread/原 Item 回归、Renderer 7 files / 159 tests、fixture guard 6/6、`npm run test:contracts`、相关 Rust/SDK 测试、`npm run smoke:agent-runtime-current-fixture` 与 `npm run verify:gui-smoke` 均通过。专项 `npm run smoke:codex-import-continuation-electron-fixture -- --timeout-ms 240000` 证明导入与普通会话 completed Command Item 同构，均只保存 `sent 9 chars`，实时与 reload GUI 不显示原始 stdin。专项证据为 `.lime/qc/gui-evidence/codex-import-continuation-fixture/codex-import-continuation-fixture-summary.json`；通用 GUI smoke 证据为 `.lime/qc/project-gates/standalone-shell-01-20260801154238-47878/shell-01-electron-smoke/summary.json`。

治理分类：unified exec、canonical terminal interactions、typed notification、Renderer live/historical projection 与 Gate B fixture 为 `current`；无 `compat`；原始 stdin、独立 `write_stdin` Tool Item、retired Bash/PowerShell tools 与生产 mock fallback 为 `dead / forbidden-to-restore`。本切片未改变 public owner、唯一产品链或依赖方向，无需改写 `internal/aiprompts/architecture.md`。责任开发者确认：root，2026-08-01。

下一刀：其余 planned notification 在当前工作树仍缺少完整 producer、consumer、持久化语义与 Gate B，保持 planned；不得为提高完成度新增协议 facade、compat 或生产 mock。v2 总体完成度保守估算仍为 97%，仍非 release-ready。

### 2026-08-09：V2-05 Guardian auto-approval review lifecycle

状态：该垂直切片 `completed`；V2-05 与 v2 总体仍为 `in-progress`，不得标记 release-ready。

本轮实现：

- `strictAutoReview` 不再转成人工审批；当前 shell/`exec_command` tool decision 会复用同一 session 的
  `model-provider` 做无工具结构化 Guardian sampling，30 秒超时、取消、provider 不可用、非法 JSON 或不确定结果全部
  fail closed 为拒绝。没有为未接入的 MCP、patch、network 或 permission 生产假 review producer。
- AgentEvent 新增 `guardian_review_started/completed`，App Server projector 生成 exact
  `item/autoApprovalReview/started|completed` typed v2 notification；协议 schema、manifest、generated TypeScript、client
  lifecycle union 与 strict decoder 同步，completed 严格拒绝 `inProgress`、未知 action、额外字段和非 `agent` decision source。
- Renderer v2 route、drift registry 与 sequence gate 接受同一 thread/turn identity；started 建立
  `pending_interactions[id=reviewId, kind=guardian_review]`，completed 将 approved/denied/timedOut/aborted 映射为
  resolved/declined/cancelled，并保留 action/review 快照。缺失 started 的 completion 只记录诊断，不创建第二份 pending store。
- Electron 只转发 App Server JSONL；没有 TUI detached/background review UI、raw provider JSON、额外 IPC、兼容 wrapper 或
  production mock fallback。多模型 catalog/default/switch/capability/readiness/retry/circuit breaker 与多模态 sampling/media
  lowering 继续归 Grok-aligned `model-provider`。

验证：`cargo test -p app-server-protocol --lib` 112/112；App Server Guardian projector 2/2；app-server-client build 与
strict notification tests 14/14；Renderer Guardian/projection 定向套件 50/50；`npm run typecheck`、`npm run test:contracts`
（301 checks）与 `npm run governance:legacy-report`（2112/1376 文件，0 候选、0 漂移、0 边界违规）通过。当前尚未单独运行
Guardian 专项 Electron Gate B 或 live provider evidence；聚合 `smoke:agent-runtime-current-fixture` 仍是下一步验证，不把
浏览器投影冒充真实桌面证据。

治理分类：Guardian agent/provider/App Server/typed client/ConversationProjection 为 `current`；无 `compat` 或
`deprecated`；旧人工审批冒充 auto review、raw side-channel、TUI detached review、生产 mock fallback 与未接入 producer
的 Guardian 扩展为 `dead / forbidden-to-restore`。`guardianWarning` 不在本条目中冒充完成，随后由独立 denial circuit
breaker 切片收口。

架构确认：已同步 `internal/aiprompts/architecture.md` 第 38 节；责任开发者确认：root，2026-08-09。下一刀进入
Guardian denial circuit breaker 的独立 `guardianWarning` producer/consumer 切片，不恢复旧双轨。

### 2026-08-09：V2-05 Guardian denial circuit breaker warning closure

状态：该垂直切片 `completed`；V2-05 与 v2 总体仍为 `in-progress`，不得标记 release-ready。

本轮实现：

- 同一 session/turn 的 Guardian denial 维护最近 5 次窗口；连续 3 次拒绝只触发一次高优先级
  `guardian_warning`，并中断当前 turn。approved、关闭 session 和新 turn 都清理 circuit-breaker 状态。
- `AgentEvent::GuardianWarning` 经 durable `guardian.warning` 进入 App Server v2 `guardianWarning`，严格要求非空
  `threadId` 与 `message`，不降级为普通 `warning`，不复用 Guardian review completed 或用户审批。
- typed protocol/schema、generated client、strict signal decoder、Renderer sequence/drift guard 与
  `ConversationProjection` 的 `NoticeProjection` 已同步；Electron 继续只转发 App Server JSONL，不新增 IPC 或 TUI
  detached/background review UI。

验证：`cargo test -p lime-agent runtime_state` 18/18、`cargo test -p app-server-protocol --lib` 113/113、App Server
durable mapper 20/20、v2 notification projector 49/49、Renderer 定向 Vitest 49/49、`npm run typecheck`、
`npm run test:contracts`（301 checks）、`npm run governance:legacy-report`（零分类漂移/边界违规）、
`npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke`、产品矩阵守卫 4/4 与 `git diff --check`
均通过。GUI smoke 证据为 `.lime/qc/project-gates/standalone-shell-01-20260809121633-64660/shell-01-electron-smoke/summary.json`。

治理分类：circuit breaker、AgentEvent、durable mapper、App Server v2 projector、typed client 与 Desktop notice 为
`current`；无 `compat` 或 `deprecated`。普通 warning 冒充、raw side-channel、TUI detached UI 与生产 mock fallback
为 `dead / deleted / forbidden-to-restore`。当前产品范围统计为 `131 implemented / 53 planned /
36 product-scope-excluded`，完成度 `131 / 184 = 71.2%`。下一刀继续从 remaining planned 集合中选择有完整
producer/consumer、恢复语义和 Gate B 证据的 owner。
