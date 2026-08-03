# Claw Trace 系统实施全过程计划

> 状态：active / S43 remains in_progress; S46 Workspace Trace adaptive layout completed
> 创建时间：2026-06-27
> 更新时间：2026-06-28
> 关联路线图：`internal/roadmap/trace/README.md`
> 关联图：`internal/roadmap/agentui/images/agentui-stream-latency-map-20260509.svg`

## 1. 主目标

系统性建立 Claw Trace 体系，用结构化 trace 把一次 Claw turn 的延迟拆成 provider/API、App Server、bridge、renderer apply、render flush、first paint 等可诊断阶段。

本计划不是单点日志优化，也不是 Harness 功能扩展。Trace 的采集与定义属于 Claw / App Server runtime / renderer 主链；Harness 只消费、展示或导出 trace evidence。

## 2. 骨架优先原则

本计划按“先骨架，后细节”推进：

1. 先落 trace context、event envelope、Noop recorder、开关、最小 checkpoint 和 summary 投影。
2. 再补 provider phase、Developer UI、support bundle、OTEL / W3C 传播等细节。
3. 每一刀只做能推进体系闭环的改动，不把局部首字优化、临时日志、UI polish 当成阶段完成。
4. 每次修改实现前先更新本计划的状态、写集和下一刀；实现后回写进度、验证结果和剩余缺口。

Trace 诊断必须同时满足两类消费者：

- JSON / JSONL evidence 给 AI、自动化校验、support bundle 和离线分析消费，保持 summary-only、可解析、可复制。
- Developer UI / 工作台 Trace Tab 给人消费，必须提供分段、timeline、慢段、缺失 phase、baseline 对比和归因摘要；不能把 raw JSON dump 当作人类调试体验。

允许并鼓励按 Codex 的实现方式重构 Lime 当前不合理的局部结构：

- 参考 `codex-rs/core/src/turn_timing.rs`，把分散的首字时间点收敛为 turn timing / phase checkpoint，而不是继续追加一次性字段。
- 参考 `codex-rs/rollout-trace/src/raw_event.rs`，所有 trace 事件使用统一 envelope、schema version、seq、wall time、thread / turn context。
- 参考 `codex-rs/rollout-trace/src/protocol_event.rs`，优先从现有 runtime / App Server event 映射 trace vocabulary，避免新增第二套业务事件。
- 参考 `codex-rs/otel/src/trace_context.rs`，W3C trace context 只作为传播层，不能替代 Lime 内部 trace id / run id。
- 如果现有实现为了兼容旧路径导致 trace、timing 或 runtime event 语义混乱，优先小步重构到上述结构，而不是继续包一层适配。

## 3. 非目标

- 不把 Trace 变成第二套 runtime event 或 read model。
- 不用正文内容、展示文案或正则判断 trace stage。
- 不记录完整 prompt、API key、环境变量、工具原始输出等敏感内容。
- 不把 provider/API 等待误归因给 Lime 客户端本地输出。
- 不复用 `workspace_harness_enabled` 作为 Trace 开关。
- 不把 Harness 模块作为 trace 采集事实源。

## 4. 当前阶段总览

| 阶段                                     | 状态        | 目标                                                                                                                                                | 退出条件                                                                                                                                                                |
| ---------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0 计划与边界冻结                        | completed   | 建立全过程计划、冻结 Harness 边界、确定骨架优先顺序                                                                                                 | 本文件创建并纳入 README 导航                                                                                                                                            |
| S1 Trace 合同骨架                        | completed   | 定义 trace config、trace id、span/event envelope、Noop recorder                                                                                     | 默认关闭时无运行开销；开关与 metadata 可单测                                                                                                                            |
| S2 Renderer 最小 checkpoint              | completed   | submit / received / applied / flush / first paint 分段                                                                                              | 本地输出链路能生成 summary，不依赖 provider                                                                                                                             |
| S3 App Server 最小 checkpoint            | completed   | request received / message.delta emitted / terminal checkpoint                                                                                      | 可计算 App Server emit 到 renderer receive 的桥接段                                                                                                                     |
| S4 Latency Map 与 evidence               | completed   | 更新 SVG，把 provider/API 与 Lime 本地输出拆开                                                                                                      | 图与 trace summary 口径一致                                                                                                                                             |
| S5 Provider phase 细节                   | completed   | Agent provider first event / first text delta / failed / canceled                                                                                   | provider_wait_ms 与 client_local_ms 分开                                                                                                                                |
| S6 Developer 调试闭环                    | completed   | 前端 summary projector、Developer & Labs 开关、compact 导出、support bundle、保留策略                                                               | summary 可区分 provider/API 与客户端本地输出；开启后可导出 compact history；默认关闭；清理不影响 session                                                                |
| S7 回归闭环                              | completed   | fixture / GUI smoke / contract guard                                                                                                                | current fixture 生成 trace evidence 并通过                                                                                                                              |
| S8 App Server raw trace store            | completed   | 内部 append-only JSONL、summary-only redaction、session retention、writer 热路径收敛                                                                | 不新增 JSON-RPC；不写 prompt / provider payload / assistant text；同一 trace seq 由 writer 状态递增                                                                     |
| S9 raw trace read/list API               | completed   | 通过 current App Server diagnostics API 读取 summary-only trace 列表和事件                                                                          | 同步 protocol / client / frontend gateway / Developer UI；不引入 mock fallback；不导出敏感 payload                                                                      |
| S10 support bundle summary               | completed   | 支持包默认包含 trace-store summary，不默认导出 raw JSONL 正文                                                                                       | trace store 作为 JSONL schema / parser owner；support bundle 只导出 summary-only 文件清单和计数                                                                         |
| S11 selective trace export               | completed   | 通过显式开发者动作导出单条 summary-only trace zip                                                                                                   | 默认 support bundle 仍只带 summary；zip 重序列化 safe event，不复制原始 JSONL 字节；同步 protocol/client                                                                |
| S12 fixture export evidence              | completed   | 真实 Electron fixture 验证 diagnostics trace list/read/export 闭环                                                                                  | summary 证明 provider/app_server checkpoint、summary-only export 和 current method 均成立                                                                               |
| S13 support bundle trace opt-in          | completed   | 支持包可由开发者显式附带单条 summary-only trace export zip                                                                                          | 默认支持包行为不变；显式参数复用 trace export 语义；manifest/README 清楚声明不包含 raw payload                                                                          |
| S14 span diagnostics                     | completed   | 从 summary-only trace events 投影慢段和缺失 phase，帮助开发者定位客户端/服务端慢点                                                                  | 不新增协议；不读取 raw payload；Developer UI 能展示 slow segments / phase gaps                                                                                          |
| S15 support bundle fixture               | completed   | 真实 Electron fixture 验证 support bundle trace opt-in 使用 current App Server trace root                                                           | `diagnostics/supportBundle/export` 出现在 method list；support bundle 附带 summary-only trace zip 且不含 raw JSONL                                                      |
| S16 W3C trace context carrier            | completed   | 参考 Codex carrier 边界，让 renderer -> App Server -> trace evidence 保留合法 W3C `traceparent`                                                     | 不替代 Lime 内部 trace id；非法 carrier 不传播；summary-only trace metrics 可关联 W3C trace id；后续 OTEL exporter 另起阶段                                             |
| S17 App Server request span              | completed   | 参考 Codex `app_server_tracing::request_span`，在 JSON-RPC request 边界建立可关联 span                                                              | `agentSession/turn/start` 进入 `app_server.request` span；span 记录安全 trace/session/turn/W3C 字段；不新增命令面、不引入 mock                                          |
| S18 OTEL exporter / remote parent        | completed   | 接入真实 OpenTelemetry exporter 与 W3C remote parent                                                                                                | App Server 一等 OTEL 依赖、subscriber/exporter 配置、测试 exporter 证明 trace id / parent span id 继承                                                                  |
| S19 provider W3C header propagation      | completed   | 参考 Codex HTTP trace header 注入，把合法 W3C carrier 从 App Server turn context 传到 provider HTTP 请求                                            | `traceparent/tracestate` 只在合法 carrier 下进入 provider HTTP header；非法 carrier 不注入；不依赖 OTEL exporter 开启；不记录 prompt/provider payload                   |
| S20 provider request id correlation      | completed   | 参考 Codex `upstream_request_id` / response debug context，将 provider response header request id 关联到本地 trace                                  | 只提取 header-safe request id；经 `ProviderTraceEvent` 进入 RuntimeEvent 与 summary-only trace metrics；不记录 provider body、prompt、assistant delta 或 raw JSONL      |
| S21 Developer trace drilldown            | completed   | 在 Developer UI 中提供 summary-only timeline filter 与 selected event detail，帮助定位慢段和缺失 phase                                              | 不新增 App Server method；只消费 `diagnostics/trace/read` summary-only events；五语言文案与 UI 回归同步                                                                 |
| S22 Developer span drilldown             | completed   | 让 Developer UI 的 phase span 成为可点击诊断对象，可直接定位该 span 内的 summary-only events                                                        | 不新增 App Server method；span key / rows helper 落在 `clawTraceTimeline.ts`；React 只维护选择状态；不读取 raw payload；五语言文案与 UI 回归同步                        |
| S23 Trace compare baseline               | completed   | 基于 compact Trace history 建立当前 summary 与最近 baseline 的轻量回归对比                                                                          | 不新增 App Server method；只消费 compact summary/history；compare projector 落在 `src/lib/trace`；不读取 raw entries / prompt / provider payload                        |
| S24 App Server Trace compare             | completed   | 基于 `diagnostics/trace/read` summary-only events 比较最近 Trace 与上一条 Trace 的 provider/App Server 分段                                         | 不新增 App Server method；显式加载 timeline 时才读取最多两条 trace；compare projector 落在 `src/lib/trace`；不读取 raw payload / prompt / assistant delta text          |
| S25 compact long-term baseline           | completed   | 从 compact Trace history 选择 retained window 的长期 baseline，避免 baseline 随最近慢样本漂移                                                       | 不新增 App Server method；只消费 `agentUiPerformanceTraceHistory` compact summary；UI 展示 baseline window；不读取 raw entries / prompt / provider payload              |
| S26 App Server retained baseline         | completed   | 从 App Server retained trace window 选择长期 baseline，避免最近慢 trace 成为 compare baseline                                                       | 显式加载 timeline 时只读取最新 / 最早两条 summary-only trace 详情；不读取中间 trace、raw payload、prompt、assistant delta text 或 `tracestate`                          |
| S27 regression evidence attribution      | completed   | 合并 compact client 分段与 App Server provider/API 分段，输出首字回退归因报告                                                                       | projector 落在 `src/lib/trace`；`rootDurationMs` 不进入归因 totals；打开 Developer 设置页本身不查询 App Server                                                          |
| S28 manual regression trend history      | completed   | 手动保存 / 复制 / 清空 retained regression report，用于跨运行追踪归因变化                                                                           | localStorage retained window 只保存 summary-only report、verdict、owner totals、segment delta 和 window 计数；不做后台自动采集                                          |
| S29 regression alert projection          | completed   | 基于当前 regression report 与手动 retained trend 投影开发者告警状态                                                                                 | 不新增 App Server method；不后台采集；不保存新数据；告警阈值与 repeated owner 判断落在纯 projector；Developer UI 只展示 summary-only alert                              |
| S30 alert channel control                | completed   | 为 regression alert 增加 Developer 显式开关，并把 Claw Trace 顶部配置控件从中心面板拆出                                                             | 不新增 App Server method；不后台采集；`alert_enabled` 默认关闭；中心面板不继续膨胀，抽出的控件只负责配置 UI                                                             |
| S31 local alert channel inbox            | completed   | 为 enabled regression alert 增加本地 summary-only retained channel，可复制和清空                                                                    | 不新增 App Server method；不自动查询 App Server；不上传；不保存 raw entries / raw trace JSONL / prompt / provider payload / assistant delta text                        |
| S32 local notification dispatcher        | completed   | 为本地 alert channel 增加可选桌面通知 dispatcher 骨架                                                                                               | 不新增 App Server / Electron method；`alert_notification_enabled` 默认关闭；只在新写入 summary-only alert 且通知权限已授予时尝试；不自动请求权限；不是最终后台通知      |
| S33 Electron desktop notification bridge | completed   | 将 trace alert 通知接入 Electron Desktop Host 原生 Notification current 命令                                                                        | 不新增 App Server method；新增 `show_desktop_notification` 壳命令与前端 API 网关；Host 只接受 summary-only title/body/tag/silent；仍不是后台常驻轮询                    |
| S34 foreground global alert monitor      | completed   | 离开 Developer 设置页后，主应用窗口仍可基于本地 compact summary 持续评估 alert 并触发通知                                                           | 不新增 App Server method；不调用 `diagnostics/trace/list/read`；不做 OS daemon；只监听本地 Agent UI performance summary-only 事件并 debounce 评估                       |
| S35 notification host test split         | completed   | 参考 Codex 中心文件只做 dispatch 的方式，将 trace alert 桌面通知 Host 逻辑的细节回归从超大 `hostCommands.test.ts` 拆到模块级测试                    | 不新增命令、不改变通知 payload 合同；`hostCommands` 只保留 dispatcher 级 smoke，summary-only 校验与 failure path 落在 `desktopNotificationHost` 模块测试                |
| S36 project shell host split             | completed   | 参考 Codex 中心文件只做 dispatch 的方式，将 `project_shell_session_*` 状态、poller、drain 和 shutdown dispose 下沉到 Project Shell Host 模块        | 不新增命令、不改前端 API 网关、不新增 App Server method；`hostCommands` 只保留 dispatcher，App Server PTY current 细节由 `projectShellHost` 模块测试守住                |
| S37 file shell host split                | completed   | 参考 Codex 中心文件只做 dispatch 的方式，将文件预览、Finder/默认应用打开、文件图标和文件管理器位置下沉到 File Shell Host 模块                       | 不新增命令、不改前端 API 网关、不新增 App Server method；`hostCommands` 只保留 dispatcher，文件壳能力细节由 `fileShellHost` 模块测试守住                                |
| S38 system utility host split            | completed   | 参考 Codex 中心文件只做 dispatch 的方式，将外链、系统设置、快捷键、环境预览、文件关联和浏览器诊断占位下沉到 System Utility Host 模块                | 不新增命令、不改前端 API 网关、不新增 App Server method；`hostCommands` 只保留 dispatcher，系统工具壳能力细节由 `systemUtilityHost` 模块测试守住                        |
| S39 agent app shell host split           | completed   | 参考 Codex 中心文件只做 dispatch 的方式，将 Agent App 目录选择、shell launch 与 UI runtime lifecycle 下沉到 Agent App Shell Host 模块               | 不新增命令、不改前端 API 网关、不新增 App Server method；`hostCommands` 只保留 dispatcher，Agent App shell / UI runtime current 细节由 `agentAppShellHost` 模块测试守住 |
| S40 agent app runtime task host split    | completed   | 参考 Codex 中心文件只做 dispatch 的方式，将 Agent App runtime task start/read/cancel/host response 下沉到 Agent App Runtime Task Host 模块          | 不新增命令、不改前端 API 网关、不新增 App Server method；`hostCommands` 只保留 dispatcher；`runWorker=false` 不查询 UI runtime status；模块级测试守住 current 投影      |
| S41 independent Trace workspace tab      | completed   | 将 Trace 作为与 Harness 并列的一等工作台 Tab，首版展示人类可读的 summary、首字分段、慢段归因、客户端健康和阶段缺口，而不是 Harness 子页或 JSON dump | 不新增协议；只消费现有 compact summary 与 summary-only trace evidence；Harness 不成为 trace schema/采集/开关 owner；Trace Tab 有五语言文案与 UI 回归                    |
| S42 Trace Tab session selection 收口     | completed   | 让 Trace Tab 默认优先选中同 workspace 的真实 Claw turn session，并在只有历史恢复链路时明确提示“当前不是发送链路”                                    | 不新增协议；只消费现有 compact summary 与 summary-only trace evidence；默认选择不再把 history restore 当成主链；五语言文案与定向测试必须同步完成                        |
| S43 electron host voice model split      | in_progress | 继续参考 Codex 中心 dispatcher 只做接线的方式，把 `voice_models_*` 本地模型下载/安装/删除/目录读取下沉到独立 `VoiceModelHost`                       | 不新增命令、不改前端 API 网关、不新增 App Server method；`hostCommands` 只保留 dispatcher，voice model 细节回归由 `voiceModelHost` 模块测试守住                         |
| S44 Trace Tab non-send path UX           | completed   | 将 history restore / unknown 非发送链路从首字发送诊断视图中拆出，只展示可用恢复耗时、session 摘要和 recorded phases，避免整页 `--` 空卡片           | 不新增协议；只消费现有 Agent UI performance compact summary；非发送链路不渲染 first-token split / baseline / regression / coverage 空面板；五语言文案与 UI 回归同步     |
| S45 Trace Tab recorded phase layout      | completed   | 将 recorded phases 从 raw checkpoint chip 墙改成人类可读的阶段组列表，避免界面直接暴露重复 checkpoint 名称                                          | 不新增协议；raw phase 仍保留在 summary / 复制摘要中；UI 按动作归类展示记录次数；五语言文案与 UI 回归同步                                                                |
| S46 Workspace Trace adaptive layout      | completed   | 收口 Workspace 右侧 Trace / Right Surface 打开后的自适应宽度，让 Trace 作为辅助诊断栏，不再压缩主对话；同时统一消息列、助手气泡和输入框阅读宽度    | 不新增协议或文案；Right Surface 默认聊天宽度有稳定回归；显式 chatPanelWidth 不被覆盖；消息正文与输入框使用同一宽度 token；定向 UI 回归与 lint 通过                    |

补充收口：

- 工作区默认挂载不再以 `workspacePluginHistoryRestoreProjection` 作为 installed agent apps 预拉条件，改为仅在 `threadRead` 存在 history restore snapshot 时预拉。
- 普通对话不再默认查询 installed agent apps；保留的查询仅来自显式 `@` 插件激活和历史恢复场景。
- 工作台 Trace Tab 已区分发送链路与非发送链路：只有 `claw_turn` 展开首字分段、baseline 和 regression；history restore 只展示恢复链路耗时，避免把“非发送路径无首字数据”误呈现为不完整 Trace。
- S44 验证：`./node_modules/.bin/vitest run "src/components/agent/chat/workspace/WorkspaceTraceTab.test.tsx"` 通过，覆盖 history restore 不渲染 first-token / baseline / regression 空面板；`./node_modules/.bin/eslint` 覆盖 Trace Tab 写集通过；`git diff --check` 覆盖本轮写集通过。
- S45 收口：Trace Tab 的 recorded phases 不再直接渲染 `messageList.paint` / `agentRuntime.getSession.success` 等 raw checkpoint chip，改为 View Model 投影出的阶段组和记录次数；raw checkpoint 继续保留在 compact summary / 复制摘要中，供 AI 和自动化排查使用。
- S45 验证：`./node_modules/.bin/vitest run "src/components/agent/chat/workspace/WorkspaceTraceTab.test.tsx"` 通过，覆盖阶段组计数与 UI 不泄露 raw checkpoint；`./node_modules/.bin/eslint "src/components/agent/chat/workspace/workspaceTracePanelModel.ts" "src/components/agent/chat/workspace/WorkspaceTraceTab.tsx" "src/components/agent/chat/workspace/WorkspaceTraceTab.test.tsx"` 通过。
- S46 收口：Trace / Harness 这类 Right Surface 属于辅助诊断面板，打开后保留主对话作为工作台主任务画布；消息列、助手气泡、inline 输入框和 floating 输入框使用同一 Codex-style 阅读宽度事实源 `clamp(640px, 68%, 720px)`，避免宽屏下正文与输入框错位，或 Trace 打开后主对话被挤成窄列。
- S46 验证：`./node_modules/.bin/vitest run "src/components/agent/chat/workspace/WorkspaceMainArea.test.tsx" "src/components/agent/chat/components/MessageList.test.tsx" "src/components/agent/chat/components/MessageList.messageActions.test.tsx"` 通过，3 个文件、39 个用例，覆盖 Right Surface 默认聊天宽度、显式宽度不被覆盖、消息列与助手气泡阅读宽度；`./node_modules/.bin/eslint` 覆盖 S46 TS/TSX 写集通过；`git diff --check` 覆盖 S46 写集通过；`npm run verify:gui-smoke` 通过，覆盖 renderer smoke build、Electron host build、App Server sidecar、renderer loaded、app-server initialized、claw workbench shell ready 和 memory settings ready。

## 5. 事实源与拟写集

### 5.1 设计事实源

- `internal/roadmap/trace/README.md`
- `internal/roadmap/trace/prd.md`
- `internal/roadmap/trace/architecture.md`
- `internal/roadmap/trace/diagrams.md`
- `internal/roadmap/trace/code-map.md`
- `internal/roadmap/trace/implementation-plan.md`
- `internal/exec-plans/claw-trace-system-implementation-plan.md`

### 5.2 预计实现写集

前端骨架：

- `src/lib/api/appConfigTypes.ts`
- `src/lib/developerFeatures.ts`
- `src/lib/trace/clawTrace.ts`
- `src/lib/api/desktopNotification.ts`
- `src/lib/api/agentRuntime/appServerEventStream.ts`
- `src/lib/api/agentProtocol.ts`
- `src/hooks/useDeveloperFeatureFlags.ts`
- `src/components/agent/chat/AgentChatWorkspace.tsx`
- `src/components/agent/chat/hooks/agentStreamPerformanceMetrics.ts`
- `src/components/agent/chat/hooks/agentStreamUserInputSendPreparation.ts`
- `src/components/agent/chat/hooks/agentStreamSubmitExecution.ts`
- `src/components/agent/chat/hooks/agentStreamTurnEventBinding.ts`
- `src/components/agent/chat/hooks/agentStreamRuntimeHandler.ts`
- `src/components/agent/chat/hooks/agentStreamTextDeltaController.ts`
- `src/components/agent/chat/hooks/agentStreamTextRenderFlushController.ts`

App Server 骨架：

- `lime-rs/crates/app-server/src/runtime.rs`
- `lime-rs/crates/app-server/src/runtime/event_store.rs`
- `lime-rs/crates/app-server/src/runtime/trace.rs`
- `lime-rs/crates/app-server/src/runtime/trace_store.rs`
- `lime-rs/crates/app-server/src/runtime/trace_store/export.rs`
- `lime-rs/crates/app-server/src/runtime/trace_store/summary.rs`
- `lime-rs/crates/app-server/src/runtime/storage_roots.rs`
- `lime-rs/crates/app-server/src/local_data_source/diagnostics/support_bundle.rs`
- `lime-rs/crates/app-server/src/local_data_source/diagnostics/support_bundle/trace_attachment.rs`
- `lime-rs/crates/app-server/src/runtime_backend/tool_events.rs`
- `lime-rs/crates/app-server/src/runtime/tests/turn_lifecycle.rs`
- `lime-rs/crates/app-server/src/main.rs`
- S5 provider phase 只进入 `runtime_backend/tool_events.rs` 与 Agent reply loop；不向 `runtime_backend.rs` 追加 trace 逻辑。

Agent / agent provider phase：

- `lime-rs/crates/agent-rust/crates/agent/src/agents/agent.rs`
- `lime-rs/crates/agent-rust/crates/agent/src/agents/provider_trace.rs`
- `lime-rs/crates/agent-rust/crates/agent/src/agents/mod.rs`
- `lime-rs/crates/agent-rust/crates/agent/src/agents/subagent_handler.rs`
- `lime-rs/crates/agent/src/protocol.rs`
- `lime-rs/crates/agent/src/event_converter.rs`
- `lime-rs/crates/agent/src/lib.rs`

测试与证据：

- `src/components/settings-v2/system/developer/index.tsx`
- `src/components/settings-v2/system/developer/ClawTraceSettingsPanel.tsx`
- `src/components/settings-v2/system/developer/index.test.tsx`
- `src/components/agent/chat/workspace/WorkspaceTraceTab.tsx`
- `src/components/agent/chat/workspace/WorkspaceTraceTab.test.tsx`
- `src/components/agent/chat/workspace/workspaceTracePanelModel.ts`
- `src/components/agent/chat/components/TaskCenterUtilityToolbar.tsx`
- `src/components/agent/chat/components/TaskCenterUtilityToolbar.integration.test.tsx`
- `src/components/agent/chat/workspace/WorkspaceConversationScene.tsx`
- `src/components/agent/chat/workspace/WorkspaceConversationScene.test.tsx`
- `src/components/agent/chat/workspace/right-surface/*`
- `src/components/agent/chat/workspace/workspaceRightSurfaceRuntimeProjection.ts`
- `src/components/agent/chat/workspace/workspaceRightSurfaceRuntimeProjection.unit.test.ts`
- `src/i18n/resources/{zh-CN,zh-TW,en-US,ja-JP,ko-KR}/agent.json`
- `src/lib/crashDiagnosticAgentUiPerformance.ts`
- `src/lib/agentUiPerformanceTraceHistory.ts`
- `src/lib/crashDiagnostic.ts`
- `src/lib/crashDiagnostic.test.ts`
- `src/i18n/resources/{zh-CN,zh-TW,en-US,ja-JP,ko-KR}/settings.json`
- `src/components/agent/chat/hooks/agentStreamPerformanceMetrics.test.ts`
- `src/lib/developerFeatures.test.ts`
- `src/lib/trace/clawTrace.test.ts`
- `src/lib/trace/clawTraceTimeline.ts`
- `src/lib/trace/clawTraceTimeline.test.ts`
- `src/lib/trace/clawTraceRegressionAlert.ts`
- `src/lib/trace/clawTraceRegressionAlert.test.ts`
- `src/lib/trace/clawTraceRegressionAlertChannel.ts`
- `src/lib/trace/clawTraceRegressionAlertChannel.test.ts`
- `src/lib/trace/clawTraceRegressionAlertDispatcher.ts`
- `src/lib/trace/clawTraceRegressionAlertDispatcher.test.ts`
- `src/lib/trace/clawTraceRegressionAlertNotifier.ts`
- `src/lib/trace/clawTraceRegressionAlertNotifier.test.ts`
- `src/lib/trace/clawTraceRegressionAlertMonitor.ts`
- `src/lib/trace/clawTraceRegressionAlertMonitor.test.ts`
- `src/lib/trace/clawTraceRegressionAlertPresentation.ts`
- `src/hooks/useClawTraceRegressionAlertMonitor.ts`
- `src/hooks/useClawTraceRegressionAlertMonitor.test.tsx`
- `src/App.tsx`
- `src/lib/api/desktopNotification.test.ts`
- `electron/desktopNotificationHost.ts`
- `electron/desktopNotificationHost.test.ts`
- `electron/agentAppRuntimeTaskHost.ts`
- `electron/agentAppRuntimeTaskHost.test.ts`
- `electron/fileShellHost.ts`
- `electron/fileShellHost.test.ts`
- `electron/projectShellHost.ts`
- `electron/projectShellHost.test.ts`
- `electron/electronRuntime.ts`
- `electron/hostCommands.ts`
- `electron/hostCommands.test.ts`
- `electron/ipcChannels.ts`
- `electron/ipcChannels.test.ts`
- `src/lib/api/agentProtocol.test.ts`
- `src/lib/api/agentRuntime/threadClient.test.ts`
- `src/components/agent/chat/hooks/agentStreamUserInputSendPreparation.test.ts`
- `src/components/agent/chat/hooks/agentStreamRuntimeMetricsController.test.ts`
- `src/components/agent/chat/hooks/agentStreamTextDeltaController.test.ts`
- `src/components/agent/chat/hooks/agentStreamTextRenderFlushController.test.ts`
- `internal/roadmap/agentui/images/agentui-stream-latency-map-20260509.svg`

> 注意：当前工作树已有大量未提交改动。实际触碰前必须先读目标文件现状，只做本计划窄写集，不回滚其他改动。

### 5.3 巨型文件触碰策略

- `src/lib/crashDiagnostic.ts` 当前约 `1369` 行，已超过仓库体量边界。本轮 S6 只允许在该文件做参数与 payload 字段接线，不继续追加复杂业务逻辑。
- 新增的 Agent UI performance summary 裁剪逻辑放入 `src/lib/crashDiagnosticAgentUiPerformance.ts`，避免继续膨胀诊断主文件。
- 风险：`crashDiagnostic.ts` 仍承担 clipboard、payload、summary、导出目录等多职责，后续每次接诊断字段都会增加回归成本。
- 下一次拆分入口：把 clipboard text / platform guide / payload builder 按 `crashDiagnosticClipboard.ts`、`crashDiagnosticPayload.ts`、`crashDiagnosticPlatform.ts` 拆出；退出条件是 `crashDiagnostic.ts` 降到 `800` 行以下，现有 `src/lib/crashDiagnostic.test.ts` 仍通过。
- `lime-rs/crates/agent-rust/crates/agent/src/agents/agent.rs` 当前仍约 `9117` 行，远超仓库体量边界。本轮 S20 已先把 provider trace event model 拆到 `agents/provider_trace.rs`，避免继续把 trace schema 堆进中心 reply loop 文件。
- 风险：`agent.rs` 仍混合 reply loop、tool orchestration、session state、native tool hook、trace emission 等职责；继续在这里追加 diagnostics 会放大回归面。
- 下一次拆分入口：优先按 Codex “中心文件只做 dispatch 接线”方式，把 provider loop diagnostics、direct answer response shaping、native tool execution hook 分别拆到 `agents/provider_trace.rs`、`agents/direct_answer.rs`、`agents/native_tool_execution.rs`；退出条件是新增 trace/runtime 逻辑不再直接增长 `agent.rs`。
- `src/lib/api/agentProtocol.ts` 当前约 `2180` 行，已超过仓库体量边界。本轮 S20 只同步 provider trace request id 字段，没有做协议 parser 拆分。
- 风险：AgentEvent parser、envelope normalization、provider trace projection 与多类事件 normalization 混在同一文件，后续每次新增诊断字段都要触碰超大网关。
- 下一次拆分入口：把 provider trace parser / runtime envelope helpers 拆到 `src/lib/api/agentProtocolProviderTrace.ts` 或同目录子模块；退出条件是 `agentProtocol.ts` 降到 `1000` 行以下，现有 `agentProtocol.test.ts` 继续覆盖导出 API。
- `electron/hostCommands.ts` 当前约 `2221` 行，仍超过仓库体量边界。S33 已把通知 payload 校验、限长与发送逻辑拆到 `electron/desktopNotificationHost.ts`；S36 已把 Project Shell PTY 会话状态、poller、drain 和 shutdown dispose 拆到 `electron/projectShellHost.ts`；S37 已把文件预览、Finder/默认应用打开、文件图标和文件管理器位置拆到 `electron/fileShellHost.ts`；S38 已把系统工具能力拆到 `electron/systemUtilityHost.ts`；S39 已把 Agent App shell / UI runtime lifecycle 拆到 `electron/agentAppShellHost.ts`；S40 已把 Agent App runtime task start/read/cancel/host response 拆到 `electron/agentAppRuntimeTaskHost.ts`；S43 正在把 voice model 本地文件壳能力拆到 `electron/voiceModelHost.ts`。
- 风险：Electron Host 本地壳能力仍混在单个大 dispatcher；继续追加壳能力会放大命令边界回归面。
- S35 已先把通知 Host 细节回归拆到 `electron/desktopNotificationHost.test.ts`，`hostCommands.test.ts` 只保留 dispatcher smoke，避免超大 host command 测试继续承接 summary-only payload 校验。
- 下一次拆分入口：继续按本地壳能力领域把 layered design export 等剩余大块分拆到独立 Host 模块，中心文件只保留命令 dispatch；退出条件是新增 Desktop Host 壳能力不再直接增长 `hostCommands.ts`。

## 6. 骨架完成定义

S1-S4 完成后才算 Trace 骨架完成：

- Developer config 有独立 Trace 开关，默认关闭。
- trace id / run id / turn id 可从 renderer 传到 App Server，再回到 renderer event payload 或 summary。
- disabled 路径使用 Noop，不写 trace 文件，不影响首字主路径。
- 至少能记录：
  - `renderer.submit`
  - `app_server.turn.received`
  - `app_server.message_delta.emitted`
  - `renderer.event.received`
  - `renderer.text_delta.applied`
  - `renderer.text.flush`
  - `renderer.text.first_paint`
- summary 明确区分：
  - provider/API wait
  - app server processing
  - bridge delivery
  - renderer apply
  - render flush
  - first paint
- SVG latency map 与 summary 术语一致。

## 7. 细节阶段定义

S5-S7 才处理以下细节：

- Agent provider request / first provider event / first text delta / failed / canceled。
- W3C `traceparent / tracestate`。
- OTEL exporter。
- Developer & Labs trace panel。
- support bundle 导出。
- retention / sampling / verbose redaction UI。
- 更完整的 GUI smoke evidence。

这些不阻塞骨架落地，但不能从最终目标中删除。

## 8. 验证矩阵

骨架阶段优先跑定向验证：

```bash
npx vitest run "src/components/agent/chat/hooks/agentStreamPerformanceMetrics.test.ts" "src/components/agent/chat/hooks/agentStreamTextDeltaController.test.ts" "src/components/agent/chat/hooks/agentStreamTextRenderFlushController.test.ts"
xmllint --noout "internal/roadmap/agentui/images/agentui-stream-latency-map-20260509.svg"
git diff --check -- "internal/exec-plans/claw-trace-system-implementation-plan.md" "internal/roadmap/trace" "internal/roadmap/agentui/images/agentui-stream-latency-map-20260509.svg"
```

触碰 App Server 后补：

```bash
cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace
```

触碰协议 / client / bridge method 后补：

```bash
npm run test:contracts
```

触碰 Claw 主路径后补：

```bash
npm run smoke:agent-runtime-current-fixture
```

## 9. 进度日志

### 2026-06-27

- 创建本全过程计划，作为后续 Trace 实施的进度事实源。
- 决策：先完成 S1-S4 骨架闭环，再回头补 S5-S7 细节。
- 决策：Harness 只作为 Trace evidence 消费方，不作为采集、开关或 schema owner。
- 完成：本计划已加入 `internal/exec-plans/README.md` 导航。
- 完成 S1 Trace 合同骨架：
  - `DeveloperConfig.claw_trace` 独立于 `workspace_harness_enabled`。
  - `normalizeClawTraceConfig / resolveClawTraceEnabled` 支持默认关闭、采样率 clamp 和独立 debug override。
  - `src/lib/trace/clawTrace.ts` 提供 schema version、checkpoint vocabulary、event envelope、Noop recorder 和内存 recorder。
  - `agentUiPerformanceTrace` metadata 扩展 `traceId / runId / turnId / serverEvent* / rendererEventReceivedAt`，并能计算 `bridgeDeliveryDeltaMs`。
- 验证通过：`npx vitest run "src/lib/developerFeatures.test.ts" "src/lib/trace/clawTrace.test.ts" "src/components/agent/chat/hooks/agentStreamPerformanceMetrics.test.ts"`，3 个文件、13 个用例通过。
- 完成 S2 Renderer 最小 checkpoint 骨架：
  - `useDeveloperFeatureFlags` 返回独立 `clawTraceEnabled`，经 `AgentChatWorkspace -> useAgentChat -> useAgentStream -> prepared send env` 下传，不复用 Harness。
  - 发送准备阶段在 Trace 开启时生成 `traceId / runId / requestId / submittedAt`，并随同一份 request metadata 进入 draft、submit op、request state。
  - `agentSession/event` projection 在 envelope 层补 `server_event_emitted_at` 与 `renderer_event_received_at`，parser 保留这些字段。
  - turn event binding 把 envelope 映射到当前 `performanceTrace`，首个 text delta 指标新增 `serverToRendererDeltaMs / rendererEventReceivedDeltaMs / serverEventDeltaMs`。
  - flush / first paint 继续走现有 `recordAgentStreamPerformanceMetric`，能随最新 trace metadata 进入 projection diagnostics。
- 验证通过：`npx vitest run "src/lib/developerFeatures.test.ts" "src/lib/trace/clawTrace.test.ts" "src/lib/api/agentProtocol.test.ts" "src/lib/api/agentRuntime/threadClient.test.ts" "src/components/agent/chat/hooks/agentStreamPerformanceMetrics.test.ts" "src/components/agent/chat/hooks/agentStreamUserInputSendPreparation.test.ts" "src/components/agent/chat/hooks/agentStreamTextDeltaController.test.ts" "src/components/agent/chat/hooks/agentStreamRuntimeMetricsController.test.ts" "src/components/agent/chat/hooks/agentStreamPreparedSendEnv.test.ts"`，9 个文件、102 个用例通过。
- 验证通过：`git diff --check` 覆盖本轮 trace 计划与 S1/S2 写集。
- 验证未完成：`npm run typecheck` 运行超过 6 分钟无输出，已中断，退出码 `130`；后续进入 S3 前优先跑更窄的类型/单测门禁或在合适窗口重跑全量。
- 下一刀：开始 S3 App Server 最小 checkpoint，并同步 S4 latency SVG。
- 完成 S3 App Server 最小 checkpoint：
  - 新增 `lime-rs/crates/app-server/src/runtime/trace.rs`，从 `turn_runtime_options.metadata.agentUiPerformanceTrace` 抽取 `traceId / runId / requestId / submittedAt`。
  - 在 `runtime/event_store.rs` 的 RuntimeEvent -> AgentEvent 统一出口装饰 trace metadata，不在 provider/backend emit 点分散加 hook。
  - `message.delta / message.delta_batch / message.batch` 写入 `app_server.message_delta.emitted` checkpoint；`message.created / turn.started / turn.accepted` 写入 `app_server.turn.received`；terminal event 写入 `app_server.turn.terminal`。
  - AgentEvent payload 增加 `trace_id / run_id / request_id / server_event_emitted_at / trace.checkpoint`，renderer 可计算 server emit -> renderer receive 的桥接段。
- 完成 S4 latency map：
  - `internal/roadmap/agentui/images/agentui-stream-latency-map-20260509.svg` 已拆出 `Provider/API wait` 与 `Lime local output`，不再把 provider TTFT 误归因到客户端。
  - SVG 口径对齐 `server_event_emitted_at -> renderer_event_received_at -> firstTextDelta -> flush -> firstPaint`。
- 完成快速响应路由硬编码收敛：
  - `fastResponseRouting.ts` 将默认 slot、resolver、label、runtime status、reasoning effort 收敛到 `DEFAULT_AGENT_FAST_RESPONSE_ROUTING_PROFILE`。
  - metadata 增加 `profile_id/profileId`，便于 trace/diagnostic 关联路由配置，不再只靠散落 magic string 识别。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace_metadata_is_attached_to_runtime_events`。
- 验证通过：`rustfmt --edition 2021 --check "lime-rs/crates/app-server/src/runtime/trace.rs" "lime-rs/crates/app-server/src/runtime/event_store.rs" "lime-rs/crates/app-server/src/runtime/tests/turn_lifecycle.rs"`。
- 验证通过：`npx vitest run "src/components/agent/chat/utils/fastResponseRouting.test.ts" "src/lib/api/agentProtocol.test.ts" "src/components/agent/chat/hooks/agentStreamUserInputSendPreparation.test.ts" "src/components/agent/chat/hooks/agentStreamPerformanceMetrics.test.ts" "src/components/agent/chat/hooks/agentStreamRuntimeMetricsController.test.ts" "src/components/agent/chat/hooks/agentStreamTextDeltaController.test.ts"`，6 个文件、62 个用例通过。
- 验证通过：`xmllint --noout "internal/roadmap/agentui/images/agentui-stream-latency-map-20260509.svg"`。
- 验证通过：`npm run smoke:agent-runtime-current-fixture`，覆盖 current Agent Runtime / Claw GUI Electron fixture，`liveProviderUsed=false`。
- 验证通过：`npm run test:contracts`。为让既有未提交 App Server smoke 变更通过 contract guard，同步更新了 `scripts/check-app-server-client-contract.mjs` 的当前实现字符串；为让 `docs:boundary` 通过，同步修正了插件路线图中的旧内部文档路径引用。
- 完成 S5 Provider phase trace：
  - Agent reply loop 在 provider stream 消费点发出 `ProviderTraceEvent`，覆盖 `request_started / first_event_received / first_text_delta_received / failed / canceled`。
  - provider request start 放在 MOIM 注入之后，避免把本地 provider 前置准备时间粗暴算进 provider/API wait。
  - `lime_agent::AgentEvent::ProviderTrace` 只携带 provider、model、attempt、elapsed_ms、text_chars、status、failure_category、retryable 等安全 metadata，不记录 prompt、provider payload 或错误全文。
  - `runtime_backend/tool_events.rs` 将 provider trace 映射为 `provider.request.started`、`provider.first_event.received`、`provider.first_text_delta.received`、`provider.failed`、`provider.canceled`。
  - `runtime/trace.rs` 为 `provider.*` event 附加 `trace_id / run_id / request_id / server_event_emitted_at / trace.checkpoint`，与 `message.delta` 的 App Server emit checkpoint 分开。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server provider_trace`，2 个 provider trace 定向用例通过。
- 更新：`internal/roadmap/trace/code-map.md`、`implementation-plan.md`、`architecture.md`、`diagrams.md` 和 latency SVG 已同步 S5 provider event 名称。
- 完成 S6 前端 summary/projector 小闭环：
  - `agentProtocol.ts` 新增 `provider_trace` typed event，支持 `provider.request.started / provider.first_event.received / provider.first_text_delta.received / provider.failed / provider.canceled`。
  - `appServerEventStream.ts` 将 App Server `provider.*` notification 投影为前端 `provider_trace`，避免落入 unknown event 噪声。
  - `agentStreamTurnEventBinding.ts` 消费 provider trace 但不驱动 UI，记录 `agentStream.providerTrace` 诊断指标。
  - `agentStreamPerformanceMetrics.ts` 与 `agentUiPerformanceMetrics.ts` 暴露 `providerWaitMs / serverToRendererFirstTextDeltaMs / rendererApplyFirstTextDeltaMs / clientLocalOutputMs` summary 字段。
  - `clientLocalOutputMs = firstTextPaint - server_event_emitted_at`，只覆盖 App Server 已发出首个 text delta 之后的 Lime 本地输出段；provider/API 慢仍归入 `providerWaitMs`。
- 验证通过：`npx vitest run "src/lib/api/agentProtocol.test.ts" "src/lib/api/agentRuntime/threadClient.test.ts" "src/components/agent/chat/hooks/agentStreamRuntimeMetricsController.test.ts" "src/components/agent/chat/hooks/agentStreamPerformanceMetrics.test.ts" "src/components/agent/chat/hooks/agentStreamTextRenderFlushController.test.ts"`，5 个文件、81 个用例通过。
- 验证通过：`npm run smoke:agent-runtime-current-fixture`，覆盖 current Agent Runtime / Claw GUI Electron fixture，`liveProviderUsed=false`。
- 验证通过：`npm run test:contracts`。
- 验证通过：`xmllint --noout "internal/roadmap/agentui/images/agentui-stream-latency-map-20260509.svg"`。
- 验证通过：`git diff --check` 覆盖本轮 trace summary/projector 写集。
- 完成 S6 Developer 调试闭环第一刀：
  - Developer & Labs / Developer Tools 增加独立 Claw Trace 开关，保存到 `developer.claw_trace.enabled`，不复用 `workspace_harness_enabled`。
  - 拆出 `ClawTraceSettingsPanel`，补齐 `level`、`sample_rate`、复制当前 summary、清空内存 summary。
  - 五语言设置文案覆盖开关、状态 pill 和保存结果。
  - `buildCrashDiagnosticPayload` 默认附加 `agent_ui_performance_summary`，只包含 session 级数值指标、phase 列表和计数。
  - 新增 `src/lib/crashDiagnosticAgentUiPerformance.ts` 做 summary 裁剪，避免继续向超大 `crashDiagnostic.ts` 追加复杂逻辑。
  - `agentUiPerformanceMetrics.d.ts` 补齐 provider/client 分段字段，保持外部类型与实现一致。
- 验证通过：`npx vitest run "src/components/settings-v2/system/developer/index.test.tsx" "src/lib/crashDiagnostic.test.ts" "src/components/agent/chat/hooks/agentStreamPerformanceMetrics.test.ts"`，3 个文件、43 个用例通过。
- 完成 S7 compact fixture evidence 第一刀：
  - `src/lib/trace/clawTrace.ts` 的 checkpoint vocabulary 已补齐 `provider.request.started / provider.first_event.received / provider.first_text_delta.received / provider.failed / provider.canceled`，与 Rust `runtime/trace.rs` 对齐。
  - 新增 `scripts/agent-runtime/claw-chat-current-fixture-agent-ui-trace.mjs`，负责临时启用 Claw Trace debug override、读取 compact `window.__LIME_AGENTUI_PERF__.summary()`、证明不导出 raw entries / raw provider payload。
  - `claw-chat-current-fixture-backend-file.mjs` 在首个 `message.delta` 前发出 provider lifecycle fixture events，走 current App Server event_store 与 renderer event stream。
  - `claw-chat-current-fixture-common-assertions.mjs` 新增 `agentUiPerformanceTraceEvidenceAvailable / agentUiPerformanceTraceSeparatesProviderAndClient / agentUiPerformanceTraceNoRawPayload`。
  - 真实 Electron fixture complete 场景已产出 evidence：`providerWaitMs=90`、`serverToRendererFirstTextDeltaMs=290`、`rendererApplyFirstTextDeltaMs=1`、`clientLocalOutputMs=337`、`rawEntriesExported=false`、`forbiddenFragmentPresent=false`。
- 验证通过：`npx vitest run "src/lib/trace/clawTrace.test.ts" "src/components/agent/chat/hooks/agentStreamPerformanceMetrics.test.ts" "scripts/agent-runtime/claw-chat-current-fixture-smoke.test.mjs"`，3 个文件、29 个用例通过。
- 验证通过：`node --check "scripts/agent-runtime/claw-chat-current-fixture-agent-ui-trace.mjs"`、`node --check "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs"`、`node --check "scripts/agent-runtime/claw-chat-current-fixture-backend-file.mjs"`。
- 验证通过：`npm run electron:build:smoke`，刷新 packaged fixture renderer / Electron host / App Server sidecar。
- 验证通过：`node "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs" --scenario complete --prefix claw-trace-s7-evidence --timeout-ms 180000`。
- 修正：plan history hydrate 会刷新 Agent UI perf 内存窗口，因此 plan 场景在主回合 read model 完成后、history hydrate 前采集 `agentUiPerformanceTrace`；末尾采集结果另存 `agentUiPerformanceTraceLatest`，不覆盖已有 provider/client evidence。
- 修正：Expert Plaza 点击入口不是标准输入框首字流式链路，backend 可产出 provider events，但当前页面没有同一个 stream listener 的 firstEvent/paint 指标；该场景显式豁免 provider/client 分段强制断言，仍保留 raw payload 脱敏断言。
- 验证通过：`node "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs" --scenario plan --prefix claw-trace-plan-evidence --timeout-ms 180000`。
- 验证通过：`node "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs" --scenario expert-plaza-skills-runtime --prefix claw-trace-expert-plaza-evidence --timeout-ms 180000`。
- 验证通过：`npm run smoke:agent-runtime-current-fixture`，覆盖 cancel / plan / skills / MCP / expert / expert plaza / expert panel 多场景，`liveProviderUsed=false`。
- 完成 S6 compact history / export / retention：
  - 新增 `src/lib/agentUiPerformanceTraceHistory.ts`，复用 diagnostic summary 裁剪结果，只保存 compact summary。
  - retention 固定为最近 20 个快照 / 7 天，导出对象显式声明 `compact_summary_only`、`raw_entries=false`、`prompt_text=false`、`provider_payload=false`。
  - `ClawTraceSettingsPanel` 增加保存当前快照、复制 Trace history、清空 Trace history，并显示当前 history 数量与最近保存时间。
  - `window.__LIME_AGENTUI_PERF__` 增加 `saveSnapshot / history / exportHistory / clearHistory`，便于 fixture 和 Playwright 读取 compact evidence。
  - 更新五语言 `settings.json` 文案与 Developer 设置页回归。
- 验证通过：`npx vitest run "src/lib/agentUiPerformanceTraceHistory.test.ts" "src/lib/agentUiPerformanceMetrics.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，3 个文件、24 个用例通过。
- 验证通过：`npx vitest run "src/lib/crashDiagnostic.test.ts" "src/components/agent/chat/hooks/agentStreamPerformanceMetrics.test.ts"`，2 个文件、26 个用例通过。
- 验证通过：`npx eslint "src/lib/agentUiPerformanceTraceHistory.ts" "src/lib/agentUiPerformanceTraceHistory.test.ts" "src/lib/agentUiPerformanceMetrics.ts" "src/lib/agentUiPerformanceMetrics.test.ts" "src/components/settings-v2/system/developer/ClawTraceSettingsPanel.tsx" "src/components/settings-v2/system/developer/index.test.tsx" --max-warnings 0`。
- 验证通过：`npx prettier --check` 覆盖本轮 Trace history / Developer UI / i18n / roadmap 写集。
- 验证通过：`git diff --check` 覆盖本轮 Trace history / Developer UI / i18n / roadmap 写集。
- 验证通过：`npm run smoke:agent-runtime-current-fixture`，覆盖 current Agent Runtime / Claw GUI Electron fixture，`liveProviderUsed=false`。
- 完成 S8 App Server raw trace store 内部骨架：
  - 新增 `runtime/trace_store.rs`，从已装饰的 `AgentEvent.payload.trace` 投影 append-only `RawTraceEvent` envelope，包含 `schema_version / seq / wall_time_unix_ms / trace_id / run_id / request_id / session_id / thread_id / turn_id / event_id / event_sequence / event_type / checkpoint / metrics / redaction`。
  - Trace 文件落到平台数据目录派生的 `runtime/traces/sessions/session_<session>/trace_<trace>.jsonl`，不硬编码用户路径。
  - redaction 固定为 `summary_only`：不保存 raw AgentEvent payload、不保存 prompt、不保存 provider payload、不保存 assistant delta 文本，只保留安全 scalar metrics 和文本长度。
  - 每个 session 保留最近 100 个 trace JSONL 文件；当前只提供内部 writer 和测试读取，不新增 App Server JSON-RPC method，不扩大命令边界。
  - `event_store.rs` 在 current RuntimeEvent -> AgentEvent 统一出口追加 trace candidate events；写 trace 失败只记录 warning，不阻断 turn 或 GUI 流。
  - 参考 Codex `rollout-trace::TraceWriter` 的热路径方式，`TraceEventWriter` 缓存每个 trace 文件的 `next_seq`，同一 trace 多次 append 不再每次读取完整 JSONL 计数；retention 只在新 trace 文件创建后触发，避免 Trace 开启时把 O(n) 文件扫描挂到每个流式 delta。
- 验证通过：`rustfmt --edition 2021 "lime-rs/crates/app-server/src/runtime/trace_store.rs"`。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace_writer -- --nocapture`，2 个 writer 用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace`，9 个 trace 相关用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server storage_roots`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server initialize_database_uses_configured_data_dir`，覆盖 `runtime/traces` 路径派生与初始化接线。
- 验证通过：`npx prettier --check` 覆盖 trace roadmap / exec plan 文档，`git diff --check` 覆盖本轮 Rust 与 roadmap 写集。
- 完成 S9 raw trace read/list current API：
  - App Server protocol 新增 `diagnostics/trace/list`、`diagnostics/trace/read`，同步 method names、catalog、schema registry 与 generated schema/types。
  - `runtime/trace_store.rs` 暴露 summary-only list/read projection，`runtime/diagnostics.rs` 与 `processor/diagnostics.rs` 接入 current diagnostics method；没有 trace writer 时返回 `available=false`。
  - `packages/app-server-client` 与 `src/lib/api/appServer*` 同步 client helper / constants / types；`src/lib/api/serverRuntime.ts` 提供前端网关，不在组件里直接散落 `safeInvoke`。
  - `ClawTraceSettingsPanel` 增加复制 App Server Trace 列表、复制最近 App Server Trace；读取结果仍是 `summary_only` redaction envelope，不导出 raw AgentEvent payload / prompt / provider payload / assistant text。
  - 五语言 `settings.json` 补齐新动作、成功和失败反馈文案。
- 验证通过：`npx vitest run "src/lib/api/serverRuntime.test.ts" "src/lib/api/appServer.test.ts" "src/lib/api/logs.current-boundary.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，4 个文件、57 个用例通过。
- 完成 S9 redaction hardening：
  - `diagnostics/trace/list` 不再返回本机 trace root。
  - `DiagnosticsTraceSummary.path` 从绝对 JSONL 路径收敛为 `sessions/session_<id>/trace_<id>.jsonl` 逻辑相对路径。
  - Rust writer 与前端网关测试已增加该边界夹具，避免 Developer UI 复制结果泄露本机数据目录。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace`，9 个 Trace 相关用例通过。
- 验证通过：`npx vitest run "src/lib/api/serverRuntime.test.ts" "src/lib/api/appServer.test.ts" "src/lib/api/logs.current-boundary.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，4 个文件、57 个用例通过。
- 验证通过：`npm run test:contracts`、`npx prettier --check` 覆盖本轮文档 / 前端写集、`git diff --check`。
- 完成 P3 timeline / phase span 第一刀：
  - 新增 `src/lib/trace/clawTraceTimeline.ts`，从 `diagnostics/trace/read` 的 summary-only events 投影 timeline rows、phase spans、root duration 和 redaction mode。
  - `ClawTraceSettingsPanel` 增加“Load Trace timeline”动作，点击后读取最近一条 App Server Trace；打开设置页本身不自动查询 App Server。
  - Timeline UI 展示 checkpoint、event type、phase、offset/delta 和安全 scalar metrics；nested/raw payload 会被 projector 丢弃。
  - 五语言 `settings.json` 补齐加载动作、成功/失败反馈、timeline overview、span/event 标签。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，2 个文件、21 个用例通过。
- 验证通过：`npx eslint "src/lib/trace/clawTraceTimeline.ts" "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/ClawTraceSettingsPanel.tsx" "src/components/settings-v2/system/developer/index.test.tsx" --max-warnings 0`。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx" "src/lib/api/serverRuntime.test.ts" "src/lib/api/appServer.test.ts" "src/lib/api/logs.current-boundary.test.ts"`，5 个文件、59 个用例通过。
- 验证通过：`npm run test:contracts`。
- 验证通过：`npx prettier --check` 覆盖 timeline / Developer UI / i18n / roadmap 写集，`git diff --check`。
- 验证未完成：`npm run typecheck` 运行约 3 分钟无输出后手动中断，退出码 `130`；本轮以定向 eslint、Vitest 与 contract 作为新增写集证据，后续全量收口前仍需在合适窗口重跑。
- 完成 S10 support bundle trace summary：
  - `StorageRoots::from_data_root` 拆出纯路径派生，support bundle 不再自己 hard code `runtime/traces`，也不会为了导出诊断包创建 runtime 目录。
  - `runtime/trace_store.rs` 成为 JSONL trace 文件摘要 parser owner，新增 `summarize_trace_event_store` 只读投影；support bundle 只把它映射为 `meta/trace-store-summary.json`。
  - `runtime/trace_store.rs` 内联测试迁移到 `runtime/tests/trace_store.rs`，生产文件从 858 行降到 735 行，低于仓库 800 行预警线。
  - 支持包 `manifest`、`README.txt` 与 `included_sections` 默认包含 `meta/trace-store-summary.json`，`omitted_sections` 明确 raw trace event JSONL 正文默认不包含。
  - summary 只包含相对路径、文件大小、event/parse error 计数、session/trace id、首末 wall time 和 modified time；不会导出 prompt、provider payload、assistant delta text 或 raw AgentEvent payload。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server support_bundle -- --nocapture`，2 个 support bundle 用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace -- --nocapture`，11 个 trace 相关用例通过。
- 完成 S11 selective trace export：
  - App Server protocol 新增 `diagnostics/trace/export`，同步 method names、catalog、schema registry、schema fixtures 与 generated TypeScript protocol types。
  - `runtime/trace_store.rs` 新增单条 trace 显式导出能力，输出 zip 到用户 Downloads/Desktop/temp 默认目录；测试通过内部 helper 写入临时目录，不污染真实下载目录。
  - 导出 zip 固定包含 `meta/manifest.json`、`meta/trace-summary.json`、`trace/events.jsonl`、`README.txt`。
  - `trace/events.jsonl` 由 `RawTraceEvent` 重新序列化生成，不复制原始 JSONL 字节；manifest 使用 `summaryOnlyTraceEventsIncluded=true`，避免误称为 raw payload export。
  - missing trace 返回 `available=true/exported=false`，不伪造 bundle path。
  - `src/lib/api/serverRuntime.ts` 增加 `exportDiagnosticsTrace` 网关，Developer UI 增加“Export latest Trace”显式动作；打开设置页本身仍不自动查询 App Server。
  - support bundle 默认策略不变：只包含 `meta/trace-store-summary.json`，不自动包含 raw trace JSONL 正文或 export zip。
  - 参考 Codex “中心文件只做 dispatch 接线”的方式，把 zip export helper 拆到 `runtime/trace_store/export.rs`，把 support bundle summary projector 拆到 `runtime/trace_store/summary.rs`；`runtime/trace_store.rs` 降到 698 行。
- 验证通过：`npx vitest run "src/lib/api/serverRuntime.test.ts" "src/lib/api/appServer.test.ts" "src/lib/api/logs.current-boundary.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，4 个文件、60 个用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol catalog -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol schema_fixtures -- --nocapture`。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace -- --nocapture`，13 个 trace 相关用例通过。
- 验证通过：`npm run test:contracts`。
- 验证未完成：`npm run typecheck` 运行超过 3 分钟仍无输出，手动中断，退出码 `130`；本轮以定向 Vitest、ESLint、Rust test 与 contract 作为新增写集证据。
- 完成 S12 fixture export evidence：
  - `claw-chat-current-fixture-agent-ui-trace.mjs` 新增 `collectAppServerTraceEvidence`，通过现有 Electron preload bridge / App Server JSON-RPC 调用 `diagnostics/trace/list`、`diagnostics/trace/read`、`diagnostics/trace/export`。
  - fixture evidence 只保存 compact trace summary、checkpoint、redaction、zip 文件名和 included/omitted sections，不保存 export 绝对路径。
  - `claw-chat-current-fixture-smoke.mjs` 在读取 invoke trace buffer 前收集 App Server trace evidence，让 assertion report 能看到三个 diagnostics trace method。
  - `claw-chat-current-fixture-common-assertions.mjs` 新增 `appServerTraceEvidenceAvailable / UsesCurrentMethods / SeparatesProviderAndServer / ExportedSummaryOnly / NoRawPayload`。
  - 为避免真实 Electron fixture 污染用户 Downloads/Desktop，`LIME_TRACE_EXPORT_OUTPUT_DIR` 可覆盖 trace export 输出目录；fixture 将其指向一次性 temp root。
  - docs boundary 门禁发现插件路线图仍有旧文档站路径引用，已最小修正为 `internal/roadmap/plugin`。
- 真实 Electron evidence 通过：`node "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs" --scenario complete --prefix claw-trace-p5-export-evidence --timeout-ms 180000`。
  - 证据文件：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-trace-p5-export-evidence-summary.json`。
  - 关键 evidence：`diagnostics/trace/list`、`diagnostics/trace/read`、`diagnostics/trace/export` 均出现在 `appServerRequestMethods`；trace events 包含 `provider.first_text_delta.received` 与 `app_server.message_delta.emitted`；export redaction 为 `summary_only`，包含 `trace/events.jsonl`，omitted 包含 `assistant delta text`。
- 验证通过：`npx vitest run "scripts/agent-runtime/claw-chat-current-fixture-smoke.test.mjs"`，20 个 smoke guard 用例通过。
- 验证通过：`node --check` 覆盖 fixture trace helper、smoke、common assertions 和 constants。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace -- --nocapture`，13 个 trace 相关用例通过。
- 验证通过：`npm run test:contracts`。
- 开始 S13 support bundle trace opt-in：
  - 决策：不新增平行 diagnostics method，扩展现有 `diagnostics/supportBundle/export` 的可选参数。
  - 决策：默认支持包继续只包含 `meta/trace-store-summary.json`，不隐式附带 trace export zip。
  - 决策：显式附带 trace 时复用 `diagnostics/trace/export` 的 summary-only zip 语义；support bundle 只装配 zip，不复制原始 JSONL 字节，也不解释 trace schema。
  - 参考 Codex：保持 writer / export / reducer 或 projector 的职责拆分，中心文件只做接线；本轮不把 support bundle 做成第二套 trace parser。
- 完成 S13 support bundle trace opt-in：
  - App Server protocol 新增 `SupportBundleExportParams.include_trace_export` 与 `SupportBundleTraceExportSelection`，没有新增平行 support bundle method。
  - `diagnostics/supportBundle/export` 可选接收单条 `session_id / trace_id`，默认无参行为仍只包含 `meta/trace-store-summary.json`。
  - `trace_store.rs` 暴露 `export_trace_events_from_store_to_path`，support bundle 复用同一 summary-only zip writer，不复制原始 JSONL 字节，不写 prompt / provider payload / assistant delta text。
  - Developer UI 新增显式动作“Export support bundle with Trace”，只在点击时查询最近 trace；打开设置页或普通对话不会自动查询 trace list。
  - package client、前端 `serverRuntime` 网关、App Server method spec、五语言 settings 文案与 Developer UI 回归已同步。
  - 参考 Codex `rollout-trace` 的职责边界，把 support bundle 的 trace 附件装配拆到 `local_data_source/diagnostics/support_bundle/trace_attachment.rs`；`support_bundle.rs` 从 692 行降到 627 行，主流程只做支持包装配。
- 验证通过：`npx vitest run "src/lib/api/serverRuntime.test.ts" "src/lib/api/appServer.test.ts" "src/lib/api/logs.current-boundary.test.ts" "src/components/settings-v2/system/developer/index.test.tsx" "packages/app-server-client/tests/client.test.mjs"`，5 个文件、115 个用例通过。
- 验证通过：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --package app-server-protocol --package app-server`。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server support_bundle -- --nocapture`，3 个 support bundle 用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace -- --nocapture`，14 个 trace 相关用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol schema_fixtures -- --nocapture`，4 个 schema fixture 用例通过。
- 验证通过：`npm run test:contracts`。
- 完成 S14 span diagnostics：
  - `src/lib/trace/clawTraceTimeline.ts` 在既有 timeline / phase spans 基础上新增 `slow_segments` 与 `phase_gaps` 投影。
  - 慢段阈值与最大条数通过 `projectClawTraceTimeline(..., options)` 可配置，Developer UI 只使用默认展示阈值，不把阈值判断散落到组件。
  - `phase_gaps` 只基于 summary-only checkpoint phase 判断缺失的 provider_api / app_server / terminal，不读取 raw payload，也不把 renderer 本地 summary 强塞进 App Server trace。
  - `ClawTraceSettingsPanel` 在点击 “Load Trace timeline” 后展示 Diagnostics 区块，列出慢段 checkpoint 边界和缺失 phase；打开设置页本身仍不自动查询 App Server。
  - 五语言 settings 文案已补齐，用户可见新增文案全部走 i18n key。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，2 个文件、24 个用例通过。
- 完成 S15 support bundle fixture evidence：
  - 真实 Electron fixture 初次失败暴露根因：`diagnostics/supportBundle/export` 的 trace 附件仍通过 `app_paths::preferred_data_dir()` 推断 trace root，和 App Server 启动时的 `StorageRoots.trace_log_root` 不一定一致。
  - 修复：`RuntimeCore::export_support_bundle` 从当前 `TraceEventWriter` 注入真实 trace root；`LocalAppDataSource` 保留默认本地支持包导出能力，但 current App Server method 不再让 support bundle 自己猜 runtime trace 事实源。
  - `TraceEventWriter` 暴露 crate 内部 `root()`，support bundle helper 增加显式 trace root 入口；默认本地 helper 仍兼容原来的 app data guess，不改变非 App Server 调用方行为。
  - Rust 回归新增 `support_bundle_uses_supplied_trace_root_for_opt_in_trace_export`，覆盖非默认 trace root、summary-only 嵌套 trace zip、`meta/trace-store-summary.json` 不含 assistant 文本。
  - 真实 Electron evidence 文件：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-trace-s15-support-bundle-evidence-summary.json`。
  - 关键 evidence：`diagnostics/supportBundle/export` 出现在根级 `appServerRequestMethods`；`supportBundleWithTrace.traceExportIncluded=true`；`supportBundleWithTrace.rawTraceJsonlOmitted=true`；`includedSections` 含 `trace-export/claw-trace-*.zip`，`omittedSections` 明确 raw trace event JSONL 未包含。
- 验证通过：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --package app-server`。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server support_bundle -- --nocapture`，4 个 support bundle 用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace -- --nocapture`，15 个 trace 相关用例通过。
- 验证通过：`npx vitest run "scripts/agent-runtime/claw-chat-current-fixture-smoke.test.mjs"`，20 个 smoke guard 用例通过。
- 真实 Electron evidence 通过：`node "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs" --scenario complete --prefix claw-trace-s15-support-bundle-evidence --timeout-ms 180000`。
- 验证通过：`npm run test:contracts`。首次执行时 docs boundary 发现插件路线图中仍有 LimeCore 外仓旧内部文档路径文字引用；已最小改为“LimeCore 外仓 plugin roadmap / operations runbook”描述后重跑通过。
- 验证通过：`git diff --check`。
- 开始 S16 W3C trace context carrier：
  - 决策：按 Codex `otel/src/trace_context.rs` 的边界只引入传播 carrier，不把 W3C trace id 替换 Lime 内部 `claw_trace_* / run_id / request_id`。
  - 决策：本轮只做 renderer metadata、App Server 事件装饰、summary-only trace evidence 的最小闭环；真实 OpenTelemetry exporter 与 span parent 接入后续单独阶段处理。
  - 决策：`tracestate` 只在 App Server runtime event payload 中短字符串传播；raw trace store metrics 暂不保存 `tracestate`，避免高基数或潜在敏感上下文进入导出。
- 完成 S16 W3C trace context carrier：
  - `src/lib/trace/clawTrace.ts` 新增 W3C carrier 生成与校验；trace 开启时前端 request metadata 写入合法 `traceparent`，已有合法 carrier 会被保留，trace 关闭时不新增字段。
  - App Server `runtime/trace.rs` 解析 nested `w3cTraceContext`，只传播合法 `traceparent / tracestate / traceId`；非法 `traceparent` 被丢弃，不影响 Lime 内部 trace id。
  - `runtime/trace_store.rs` 在 summary-only metrics 中保存 `w3c_trace_id / w3c_traceparent`，不保存 prompt、provider payload、assistant delta text、raw JSONL 原始字节或 `tracestate`。
  - 真实 Electron evidence 文件：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-trace-s16-w3c-evidence-summary.json`。
  - 关键 evidence：`appServerTraceEvidenceHasW3cCarrier=true`；App Server trace events 的 `w3cTraceId` 一致，且每个 checkpoint 均带合法 `traceparent` carrier。
- 验证通过：`npx vitest run "src/lib/trace/clawTrace.test.ts" "src/components/agent/chat/hooks/agentStreamPerformanceMetrics.test.ts" "src/components/agent/chat/hooks/agentStreamUserInputSendPreparation.test.ts" "scripts/agent-runtime/claw-chat-current-fixture-smoke.test.mjs"`，4 个文件、45 个用例通过。
- 验证通过：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --package app-server`。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace -- --nocapture`，16 个 trace 相关用例通过。
- 验证通过：`npm run test:contracts`。
- 验证通过：`npm run electron:build:smoke`。
- 真实 Electron evidence 通过：`node "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs" --scenario complete --prefix claw-trace-s16-w3c-evidence --timeout-ms 180000`。
- 验证通过：`npx prettier --check` 覆盖本轮 Trace/W3C/fixture/计划写集，`git diff --check`。
- 验证未完成：`npm run typecheck` 运行超过 3 分半仍无输出，手动中断，退出码 `130`；本轮以定向 Vitest、Rust trace test、Electron build smoke、contract guard 和真实 Electron fixture 作为可交付证据。
- 开始 S17 App Server request span boundary：
  - 决策：参考 Codex `app_server_tracing::request_span + request_fut.instrument(...)` 的职责边界，把 request span 放在 App Server JSON-RPC request dispatch 入口，而不是塞进 runtime event 转换或 trace store writer。
  - 决策：Lime App Server 当前只有 `tracing` 是一等依赖；`opentelemetry* / tracing-opentelemetry` 主要来自嵌入的 Agent 依赖锁文件。本阶段不把 OTEL 依赖和全局 subscriber 一次性拉进 App Server，先完成 span 边界和 carrier seam，真实 exporter 拆到 S18。
  - 决策：不新增 JSON-RPC top-level `trace` 字段；先复用现有 `agentSession/turn/start.runtimeOptions.metadata.agentUiPerformanceTrace.w3cTraceContext`，保持 S16 的 carrier 合同不漂移。
- 完成 S17 App Server request span boundary：
  - 新增 `trace_context.rs`，把 W3C `traceparent / tracestate` 校验和归一化从 `runtime/trace.rs` 抽成共用模块；runtime event 装饰与 processor request span 使用同一套合法性规则。
  - 新增 `processor/request_trace.rs`，从 `agentSession/turn/start` params 的 runtime metadata 抽取 `traceId / runId / requestId / sessionId / turnId / w3cTraceContext`，创建 `app_server.request` span。
  - `RequestProcessor::handle_request` 与 `handle_request_streaming` 统一用 `tracing::Instrument` 包住 request future；streaming turn 的 runtime / backend / event callback 也处于同一 request span 上下文。
  - span 只记录安全 scalar 字段：`rpc.method`、`rpc.request_id`、client name/version、`agent.session_id`、`agent.turn_id`、`claw.trace_id`、`claw.run_id`、`claw.request_id`、`w3c.trace_id`、`w3c.parent_span_id`、`w3c.trace_flags`、`w3c.traceparent.valid`；不记录 prompt、assistant delta text、provider payload、raw JSONL 或 `tracestate`。
  - 非 `agentSession/turn/start` method 不解析 Claw trace metadata；非法 traceparent 会记录 `w3c.traceparent.valid=false` 并保留 Lime 内部 trace identity，不阻断 turn。
- 验证通过：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --package app-server`。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace_context -- --nocapture`，4 个用例通过，覆盖共用 W3C parser 与既有 runtime invalid propagation。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server request_trace -- --nocapture`，3 个 request span metadata 用例通过。
- 修复 fixture 稳定性缺口：`claw-chat-current-fixture-agent-ui-trace.mjs` 在 renderer ready 前启用 Trace debug override 时，如果当前页仍是 Electron startup placeholder / `about:blank`，只安装 init script，不再强制 reload；真实 renderer 导航后自动带上 debug override。若已在真实页面，则仅对 Electron reload 竞态 `ERR_ABORTED / frame detached` 做恢复，后续仍由 `waitForRendererReady` 证明页面可用。
- 验证通过：`node --check "scripts/agent-runtime/claw-chat-current-fixture-agent-ui-trace.mjs"`。
- 验证通过：`npx vitest run "scripts/agent-runtime/claw-chat-current-fixture-smoke.test.mjs"`，20 个 fixture guard 用例通过。
- 验证通过：`node "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs" --scenario skills-runtime --prefix claw-chat-current-fixture-skills-runtime-regression-rerun --timeout-ms 180000`。
- 验证通过：`node "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs" --scenario cancel-then-continue --prefix claw-chat-current-fixture-cancel-then-continue-regression-rerun --timeout-ms 180000`。
- 验证通过：`npm run smoke:agent-runtime-current-fixture`，覆盖 history/cache、stream terminal、Electron/App Server fixture guard、Coding Workbench、cancel-then-continue、Plan hydrate、Skills Runtime、MCP structuredContent、Expert Skills、Expert Plaza、Expert Panel；`liveProviderUsed=false`。
- 开始 S18 OTEL exporter / remote parent：
  - 决策：参考 Codex `codex_otel::set_parent_from_w3c_trace_context` 与 app-server tracing 测试，先把 W3C carrier 转成 OpenTelemetry remote parent，而不是继续只在 span attribute 里记录 trace id。
  - 决策：OpenTelemetry exporter 必须默认关闭；生产网络导出只能由开发者显式环境配置开启，测试使用内存 exporter 证明 span parent 语义。
  - 决策：继续复用 `agentSession/turn/start.runtimeOptions.metadata.agentUiPerformanceTrace.w3cTraceContext`，不新增第二个 JSON-RPC trace 字段，也不记录 prompt、assistant delta、provider payload、raw JSONL 或 `tracestate` 到 span attributes。
- 完成 S18 OTEL exporter / remote parent：
  - 新增 `otel_trace.rs`，按 Codex 的职责边界提供 `context_from_w3c_trace_context` 与 `set_parent_from_w3c_trace_context`；W3C carrier 仍只作为传播层，不替代 Lime 内部 `traceId / runId / requestId`。
  - `processor/request_trace.rs` 在 `app_server.request` span 创建后设置 OpenTelemetry remote parent；合法 carrier 导出的 span 继承 renderer trace id 与 parent span id，非法 carrier 只记录 `w3c.traceparent.valid=false`，不阻断 turn。
  - App Server 引入一等 `opentelemetry / opentelemetry-otlp / opentelemetry_sdk / tracing-opentelemetry` 依赖；`opentelemetry_sdk` 不启用 testing feature，request trace 测试使用本地最小 `SpanExporter`，避免为了测试拉大依赖面。
  - `main.rs` 在 stdio transport 前安装 opt-in OTLP tracing guard；默认不安装 subscriber、不产生网络导出、不向 stdout 写 tracing 日志。只有 `APP_SERVER_OTEL_EXPORTER=otlp`、`OTEL_TRACES_EXPORTER=otlp`、`OTEL_EXPORTER_OTLP_ENDPOINT` 或 `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` 显式配置时启用。
  - span attributes 仍只包含安全 scalar：`rpc.*`、client、session/turn、`claw.*`、`w3c.trace_id / parent_span_id / trace_flags / traceparent.valid`；不记录 prompt、assistant delta text、provider payload、raw JSONL 或 `tracestate`。
- 验证通过：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --package app-server`。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server otel_trace -- --nocapture`，1 个 OTEL context 用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server request_trace -- --nocapture`，5 个 request trace 用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace -- --nocapture`，24 个 trace 相关用例通过。
- 验证通过：`npm run test:contracts`。
- 验证通过：`npm run smoke:agent-runtime-current-fixture`，覆盖 current Agent Runtime / Claw GUI Electron fixture；`liveProviderUsed=false`。
- 完成 S19 provider W3C header propagation：
  - `runtime_backend/request_context.rs` 复用 App Server `trace_context.rs` parser，把合法 `agentUiPerformanceTrace.w3cTraceContext` 投影为 Agent `TurnContextOverride.metadata.w3c_trace_context`；非法 `traceparent` 不进入 turn context。
  - Agent `session_context.rs` 扩展现有 `RequestCorrelationContext`，读取并归一化 `w3c_trace_context.traceparent / tracestate`；`tracestate` 只在 `traceparent` 合法时传播。
  - `providers/api_client.rs` 继续复用统一 `send_request` header 注入管道，所有 provider HTTP request 自动携带标准 `traceparent / tracestate` header，避免每个 provider 手写。
  - 参考 Codex `inject_span_w3c_trace_headers` 的统一 HTTP header 注入思路；Lime 没有照搬“只从 current OTEL span 注入”，因为 Lime 的 OTLP/subscriber 默认关闭，主路径必须依赖显式 request carrier 才稳定。
  - 修复 Agent 测试枚举穷尽性：`AgentEvent::ProviderTrace` 在 `tests/agent.rs` 的非业务事件分支中显式忽略，避免 provider trace 事件新增后阻塞测试编译。
- 验证通过：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --package app-server`、`cargo fmt --manifest-path "lime-rs/crates/agent-rust/Cargo.toml" --package agent-core`。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server w3c_trace_context -- --nocapture`，5 个 W3C / trace context 相关用例通过。
- 验证通过：`CARGO_TARGET_DIR="lime-rs/target" cargo test --manifest-path "lime-rs/crates/agent-rust/crates/agent/Cargo.toml" w3c_trace_context -- --nocapture`、`test_request_correlation_headers_injection`、`test_request_correlation_context_rejects_invalid_w3c_traceparent`。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace -- --nocapture`，26 个 trace / OTEL / support bundle / provider phase 用例通过。
- 验证通过：`npm run test:contracts`。
- 验证通过：`npm run smoke:agent-runtime-current-fixture`，覆盖 current Agent Runtime / Claw GUI Electron fixture；`liveProviderUsed=false`。
- 开始 S20 provider request id correlation：
  - 决策：主要参考 Codex `rollout-trace::RawTraceEventPayload::InferenceCompleted/Failed/Cancelled.upstream_request_id` 与 `response-debug-context` 的 header 提取方式；Lime 只接入 provider response header request id，不扩展 cf-ray、auth error、HTTP body 或 provider raw response。
  - 决策：不修改每个 provider trait 返回类型，先在 Agent 统一 `ApiRequestBuilder::response_post/response_get` 捕获 response headers，再通过 turn task-local 交给 reply loop。
  - 决策：`request_started` 阶段尚未收到 response headers，不填 provider request id；`first_event / first_text_delta / failed / canceled` 可携带已捕获的 provider request id。
  - 决策：只接受长度不超过 256 的 header-safe visible ASCII request id；含空格、控制字符或空值的 header 丢弃。
- 完成 S20 provider request id correlation：
  - Agent `session_context.rs` 新增 `ProviderResponseContext` task-local，按优先级提取 `x-request-id / x-oai-request-id / x-openai-request-id / request-id / x-amzn-requestid / x-amz-request-id / x-goog-request-id / x-ms-request-id`。
  - Agent `providers/api_client.rs` 在 provider HTTP response 返回后记录 response headers；每次 provider request 发送前清空旧 context，避免跨请求泄漏。
  - Agent `ProviderTraceEvent` 新增 `provider_request_id / provider_request_id_header`，并在 first event、first text、failed、canceled 事件上挂载。
  - 参考 Codex 的职责拆分方式，把 provider trace stage/event 和耗时计算从超大 `agent.rs` 移到 `agents/provider_trace.rs`；`agent.rs` 只保留 reply loop 中的事件发射接线。
  - `lime_agent::AgentEvent::ProviderTrace`、Agent -> Agent event converter、App Server `runtime_backend/tool_events.rs`、trace store safe metrics whitelist、前端 AgentEvent parser 与 App Server event projection 已同步。
  - App Server fixture backend 与 trace store 测试已断言 request id 进入 RuntimeEvent payload 和 summary-only metrics。
- 验证通过：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --package app-server --package lime-agent`、`cargo fmt --manifest-path "lime-rs/crates/agent-rust/Cargo.toml" --package agent-core`。
- 验证通过：`CARGO_TARGET_DIR="lime-rs/target" cargo test --manifest-path "lime-rs/crates/agent-rust/crates/agent/Cargo.toml" provider_response_context -- --nocapture`，3 个 request id / provider context 用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server provider_trace -- --nocapture`，2 个 provider trace 用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent provider_trace -- --nocapture`，协议 crate provider_trace 过滤编译通过。
- 验证通过：`npx vitest run "src/lib/api/agentProtocol.test.ts" "src/lib/api/agentRuntime/threadClient.test.ts"`，2 个文件、64 个用例通过。
- 验证通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server trace -- --nocapture`，26 个 trace / OTEL / support bundle / provider phase 用例通过。
- 开始 S21 Developer trace drilldown：
  - 决策：不新增 App Server method，不自动查询 trace；继续复用 Developer UI 的显式 “Load Trace timeline” 动作和 `diagnostics/trace/read` summary-only events。
  - 决策：筛选与详情逻辑优先放在 `src/lib/trace/clawTraceTimeline.ts`，React 面板只做展示和用户选择状态，避免把 phase / slow segment 判断散落到组件。
  - 决策：Developer 诊断面板允许出现 Trace、Provider / API、App Server 等技术词；新增用户可见文案仍必须覆盖五语言。
- 完成 S21 Developer trace drilldown：
  - `clawTraceTimeline.ts` 新增 `ClawTraceTimelineFilter`、稳定 row key 和 `filterClawTraceTimelineRows`，支持 all、phase、slow segment 过滤。
  - `ClawTraceSettingsPanel` 在 timeline 内增加 filter chips、可点击事件列表和 selected event detail；详情只显示 checkpoint、phase、seq、event type、offset/delta 和 safe metrics，不读取 raw payload。
  - 五语言 `settings.json` 已补齐 filter / detail / empty state 文案。
  - `index.test.tsx` 增加 UI 回归：加载 timeline 后默认显示 provider request id safe metric，切换 App Server / Slow filter 时事件列表与详情同步变化，且 raw provider payload 不出现。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，2 个文件、24 个用例通过。
- 开始 S22 Developer span drilldown：
  - 决策：参考 Codex `rollout-trace` 的 reducer / projection 边界，span 选择、row 定位和 key 稳定性放到 `src/lib/trace/clawTraceTimeline.ts`，组件不重复实现 trace 区间判断。
  - 决策：点击 phase span 只消费已有 `diagnostics/trace/read` summary-only events；不新增 App Server method，不查询 raw JSONL 原始字节，不导出 prompt、provider payload、assistant delta text 或 `tracestate`。
  - 决策：Developer 诊断面板把 span 当作当前排查对象，展示 duration、range、event count 与 selected event；筛选 chips 仍保留 all / phase / slow，用于快速切换视角。
- 完成 S22 Developer span drilldown：
  - `clawTraceTimeline.ts` 新增稳定 `clawTraceSpanKey`、`findClawTraceSpanByKey` 与 `filterClawTraceTimelineRowsBySpan`，span 内事件定位逻辑集中在 projector 层。
  - `ClawTraceSettingsPanel` 将 phase span 卡片改为可点击 button；点击后设置当前 span、切到对应 phase filter，并选中 span 内第一条事件。
  - 详情区新增 selected span summary，显示 phase、duration、range 与 event count；事件详情仍只展示 summary-only safe metrics。
  - 五语言 `settings.json` 补齐 selected span 标题文案；UI 回归覆盖点击 App Server span 后事件列表与详情同步定位，且 raw provider payload 仍不出现。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，2 个文件、24 个用例通过。
- 验证通过：`npx eslint "src/lib/trace/clawTraceTimeline.ts" "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/ClawTraceSettingsPanel.tsx" "src/components/settings-v2/system/developer/index.test.tsx" --max-warnings 0`。
- 开始 S23 Trace compare baseline：
  - 决策：先做 compact Agent UI performance summary 的 lightweight compare，比较当前内存 summary 与最近一次保存快照，服务首字/本地输出回归排查。
  - 决策：compare 只消费 `agentUiPerformanceTraceHistory` 已保存的 compact summary，不触碰 App Server raw trace store、不新增 diagnostics method、不读取 raw entries / prompt / provider payload。
  - 决策：按 Codex reducer/projector 思路把 baseline 选择、metric delta 和 regression 分类放到 `src/lib/trace` 纯函数；Developer UI 只展示结果。
- 完成 S23 Trace compare baseline：
  - 新增 `src/lib/trace/clawTraceBaseline.ts`，从当前 compact summary 与最近 Trace history record 投影 baseline comparison，比较 `providerWaitMs / serverToRendererFirstTextDeltaMs / rendererApplyFirstTextDeltaMs / clientLocalOutputMs`。
  - 回归判定采用保守阈值：至少 +50ms 且相对 baseline +15% 才标为 `regressed`，避免微小抖动误报；改善同样要求至少 -50ms。
  - 新增 `ClawTraceBaselineComparisonCard`，Developer UI 展示 baseline label、verdict 和 metric delta；仍只展示 compact summary 数值，不展示 raw payload。
  - 按 Codex “中心文件只做接线”方式拆出 `ClawTraceTimelineView` 与 `ClawTraceBaselineComparisonCard`；`ClawTraceSettingsPanel.tsx` 从 1300+ 行降到 727 行，不再继续堆 timeline/baseline 业务 JSX。
  - 五语言 `settings.json` 补齐 baseline compare / verdict / metric 标签；组件回归覆盖 baseline 入口和 raw payload 不出现。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceBaseline.test.ts" "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，3 个文件、27 个用例通过。
- 验证通过：`npx eslint` 覆盖 baseline projector、timeline projector、拆分后的 Developer UI 组件与回归测试。
- 开始 S24 App Server Trace compare：
  - 决策：主要参考 Codex `rollout-trace` reducer/projector 边界，App Server summary-only events 先投影成 timeline，再由纯 compare projector 计算 metric delta；React 只展示 compare 结果。
  - 决策：不新增 JSON-RPC，不把 compare 下推到 App Server；Developer UI 只有在点击 “Load Trace timeline” 后才通过 `diagnostics/trace/list` 读取最多两条 trace，再分别 `diagnostics/trace/read`。
  - 决策：S24 只比较 App Server trace 能证明的 provider/API 与 App Server emit/terminal 分段；renderer apply、render flush、first paint 继续由 S23 compact summary baseline 覆盖，避免把 provider/API 等待误归因给客户端本地输出。
  - 决策：compare 结果只展示 checkpoint、duration、delta、verdict 和 trace id；不读取 raw JSONL 原始字节，不展示 prompt、provider payload、assistant delta text 或 `tracestate`。
- 完成 S24 App Server Trace compare：
  - 新增 `src/lib/trace/clawTraceAppServerComparison.ts`，从最近 trace 与上一条 trace 的 timeline projection 投影 `providerFirstEventMs / providerFirstTextMs / providerToAppServerFirstDeltaMs / appServerFirstDeltaToTerminalMs / rootDurationMs`。
  - 回归判定沿用 S23 的保守阈值：至少 +50ms 且相对 baseline +15% 才标为 `regressed`，避免微小抖动误报；改善要求至少 -50ms。
  - `ClawTraceSettingsPanel` 的 “Load Trace timeline” 显式动作改为 `diagnostics/trace/list { limit: 2 }`，只在有上一条 trace 时读取 baseline trace；打开设置页本身仍不查询 App Server。
  - 新增 `ClawTraceAppServerComparisonCard` 展示最近 trace 与 baseline trace 的 verdict 和 metric delta；五语言 `settings.json` 已补齐文案。
  - UI 回归覆盖读取两条 summary-only trace、展示 App Server Trace compare、provider first text/root duration 回退，并断言 raw provider payload 不出现。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceAppServerComparison.test.ts" "src/lib/trace/clawTraceBaseline.test.ts" "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，4 个文件、30 个用例通过。
- 验证通过：`npx eslint` 覆盖 App Server compare projector、baseline/timeline projector、拆分后的 Developer UI 组件与回归测试。
- 验证通过：`npx prettier --check` 覆盖新增 TS/TSX、五语言 settings 与 trace 文档；`git diff --check` 覆盖本轮 trace 写集。
- 验证通过：`npm run test:contracts`。本轮未新增 App Server method；contract guard 用于确认 diagnostics trace current method / protocol / docs boundary 未漂移。
- 开始 S25 compact long-term baseline：
  - 决策：现有 S23 只拿最近一条 compact history record 做 baseline，适合短期对比，但当最近快照本身已经变慢时会把回归吞掉。
  - 决策：S25 先复用现有 localStorage compact history retention window，不新增 schema、不新增 App Server method、不新增自动采集；projector 从 retained records 中选择最早快照作为长期 baseline，并输出 history window 信息。
  - 决策：比较仍只消费 `AgentUiPerformanceDiagnosticSummary` 的聚合数值，不读取 raw entries、prompt、provider payload 或 assistant delta text；React card 只展示 projector 输出。
- 完成 S25 compact long-term baseline：
  - `clawTraceBaseline.ts` 新增 `oldest_retained_snapshot` baseline strategy，先按 `saved_at_ms` 排序，再选择 retained history window 中最早快照作为长期 baseline。
  - `ClawTraceBaselineComparison` 新增 `baseline_strategy / history_record_count / latest_saved_at`，用于解释当前比较窗口；metric delta 仍只包含聚合数值。
  - `ClawTraceBaselineComparisonCard` 展示 baseline window：快照数量、最早保留快照作为 baseline、最近快照时间。
  - 五语言 `settings.json` 补齐 baseline window 文案；UI 回归覆盖窗口文案，projector 单测覆盖多条 history 时不再拿最近慢样本做 baseline。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceBaseline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，2 个文件、26 个用例通过。
- 验证通过：`npx eslint "src/lib/trace/clawTraceBaseline.ts" "src/lib/trace/clawTraceBaseline.test.ts" "src/components/settings-v2/system/developer/ClawTraceBaselineComparisonCard.tsx" "src/components/settings-v2/system/developer/index.test.tsx" --max-warnings 0`。
- 开始 S26 App Server retained-window baseline：
  - 决策：S24 只拿上一条 App Server trace 做 baseline，适合短期对比，但如果最近几条 trace 已经一起变慢，会把长期回退吞掉。
  - 决策：继续复用现有 `diagnostics/trace/list`，不新增 App Server method；Developer UI 只有点击 “Load Trace timeline” 后才读取 retained summary window。
  - 决策：summary list 可取最多 20 条，但详情只读取最新 trace 和窗口内最早 trace；比较仍只消费 summary-only timeline projection，不读取 raw JSONL 原始字节、prompt、provider payload、assistant delta text 或 `tracestate`。
- 完成 S26 App Server retained-window baseline：
  - `clawTraceAppServerComparison.ts` 新增 `oldest_retained_trace` baseline strategy，并提供 `selectClawTraceAppServerComparisonWindow`，从 retained trace summaries 中选择最新 trace 与最早保留 trace。
  - `ClawTraceAppServerComparison` 新增 `baseline_strategy / trace_window_count / latest_trace_id`，用于解释 App Server compare 的窗口口径；metric delta 仍只包含 checkpoint duration。
  - `ClawTraceSettingsPanel` 的 “Load Trace timeline” 显式动作改为 `diagnostics/trace/list { limit: 20 }`，随后只读取最新 / 最早两条 summary-only trace 详情；打开设置页本身仍不查询 App Server。
  - `ClawTraceAppServerComparisonCard` 展示 trace window：trace 数量、最早保留 trace 作为 baseline、最新 trace id。
  - 五语言 `settings.json` 补齐 App Server compare window 文案；UI 回归覆盖 3 条 trace 时不读取中间 trace、只用最早 trace 做 baseline。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceAppServerComparison.test.ts" "src/lib/trace/clawTraceBaseline.test.ts" "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，4 个文件、32 个用例通过。
- 验证通过：`npx eslint "src/lib/trace/clawTraceAppServerComparison.ts" "src/lib/trace/clawTraceAppServerComparison.test.ts" "src/components/settings-v2/system/developer/ClawTraceSettingsPanel.tsx" "src/components/settings-v2/system/developer/ClawTraceAppServerComparisonCard.tsx" "src/components/settings-v2/system/developer/index.test.tsx" --max-warnings 0`。
- 验证通过：`npx prettier --check` 覆盖 S26 TS/TSX、五语言 settings 与 trace 文档；`git diff --check` 覆盖本轮 trace 写集。
- 验证通过：`npm run test:contracts`。本轮未新增 App Server method；contract guard 用于确认 diagnostics trace current method / protocol / docs boundary 未漂移。
- 开始 S27 regression evidence attribution：
  - 决策：S23/S25 已能比较 compact 客户端分段，S24/S26 已能比较 App Server provider/API 分段，但 Developer UI 还没有把二者合并成“首字回退到底归因到谁”的报告。
  - 决策：继续按 Codex reducer/projector 边界做纯投影；React 只展示归因结果，不重新判断 metric 所属阶段。
  - 决策：报告只消费既有 compare projector 输出，不新增 App Server method、不自动查询 App Server、不读取 raw JSONL 原始字节、prompt、provider payload、assistant delta text 或 `tracestate`。
  - 决策：`rootDurationMs` 只作为 App Server compare 展示，不进入归因 totals，避免总时长与子分段重复计数。
- 完成 S27 regression evidence attribution：
  - 新增 `src/lib/trace/clawTraceRegressionReport.ts`，合并 compact baseline 与 App Server compare 的 summary-only metric delta，归因到 `provider_api / app_server / lime_client`。
  - 新增 `ClawTraceRegressionReportCard`，展示归因焦点、compact/App Server window、owner totals 和 top segments。
  - `ClawTraceSettingsPanel` 接入 regression report 卡片；打开设置页本身仍不查询 App Server，只有加载 timeline 后 report 才包含 App Server trace evidence。
  - 五语言 `settings.json` 补齐 regression evidence 文案；UI 回归覆盖无证据、compact baseline 稳定态、加载 App Server timeline 后 Provider / API 焦点与 raw payload 不泄露。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceRegressionReport.test.ts" "src/lib/trace/clawTraceAppServerComparison.test.ts" "src/lib/trace/clawTraceBaseline.test.ts" "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，5 个文件、34 个用例通过。
- 验证通过：`npx eslint "src/lib/trace/clawTraceRegressionReport.ts" "src/lib/trace/clawTraceRegressionReport.test.ts" "src/lib/trace/clawTraceAppServerComparison.ts" "src/components/settings-v2/system/developer/ClawTraceRegressionReportCard.tsx" "src/components/settings-v2/system/developer/ClawTraceSettingsPanel.tsx" "src/components/settings-v2/system/developer/index.test.tsx" --max-warnings 0`。
- 验证通过：`npx prettier --check` 覆盖 S27 TS/TSX、五语言 settings 与 trace 文档；`git diff --check` 覆盖本轮 trace 写集。
- 验证通过：`npm run test:contracts`。本轮未新增 App Server method；contract guard 用于确认 diagnostics trace current method / protocol / docs boundary 未漂移。
- 开始 S28 manual regression trend history：
  - 决策：S27 已有 regression evidence 归因报告，但跨运行趋势还只能靠开发者现场观察；S28 先做手动保存/复制/清空的 retained history，不做后台自动采集或自动告警。
  - 决策：trend history 只保存 summary-only report、verdict、owner totals、segment delta 和 window 计数；不保存 raw entries、raw trace JSONL、prompt、provider payload、assistant delta text 或 `tracestate`。
  - 决策：trend history 放在 `src/lib/trace/clawTraceRegressionTrend.ts`，复用 20 条 / 7 天 retained window；`ClawTraceSettingsPanel` 继续只做接线，避免中心组件膨胀。
- 完成 S28 manual regression trend history：
  - 新增 `clawTraceRegressionTrend.ts`，提供 save/list/overview/export/clear；损坏历史数据 fail closed。
  - `ClawTraceRegressionReportCard` 增加 Save regression evidence / Copy regression trend / Clear regression trend 显式动作，并展示已保存 report 数量和最近 verdict。
  - `ClawTraceSettingsPanel.tsx` 在 S28 时为 780 行，仍低于 800 行预警线；trend 逻辑没有塞回中心文件。
  - 五语言 `settings.json` 补齐 trend action / toast 文案；UI 回归覆盖保存、复制、清空 trend，并断言导出不包含 raw provider payload。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceRegressionTrend.test.ts" "src/lib/trace/clawTraceRegressionReport.test.ts" "src/lib/trace/clawTraceAppServerComparison.test.ts" "src/lib/trace/clawTraceBaseline.test.ts" "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，6 个文件、37 个用例通过。
- 验证通过：`npx eslint "src/lib/trace/clawTraceRegressionTrend.ts" "src/lib/trace/clawTraceRegressionTrend.test.ts" "src/components/settings-v2/system/developer/ClawTraceRegressionReportCard.tsx" "src/components/settings-v2/system/developer/ClawTraceSettingsPanel.tsx" "src/components/settings-v2/system/developer/index.test.tsx" --max-warnings 0`。
- 验证通过：`npx prettier --check` 覆盖 S28 trace projector / Developer UI / 五语言 settings / trace 文档 / 执行计划写集；`git diff --check` 覆盖本轮 trace 写集；`rg -n "[ \t]+$"` 覆盖未跟踪 trace 新文件和 latency map SVG，未发现尾随空白。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；本轮未新增 App Server method，contract guard 用于确认 current diagnostics trace / docs boundary 未漂移。
- 验证未执行：`npm run verify:gui-smoke`。S28 只改显式 Developer 诊断 projector / UI localStorage trend / 文档，不改 Claw streaming 热路径、Electron bridge 或 App Server runtime 行为；本轮以定向 Vitest、ESLint、Prettier、diff 与 contract 作为最低门槛，后续触达主聊天 GUI 或 bridge 时再补 GUI smoke。
- 开始 S29 regression alert projection：
  - 决策：S29 做 Developer 侧 summary-only alert 投影，不做后台自动采集、不新增 App Server method、不自动保存 trend。
  - 决策：告警只消费当前 regression report 与手动 retained trend records；`none / watch / warning / critical`、delta threshold 和 repeated owner 判断集中在 `src/lib/trace/clawTraceRegressionAlert.ts`。
  - 决策：`ClawTraceSettingsPanel.tsx` 不继续增长，卡片只展示 projector 输出；alert 不保存 raw entries、raw trace JSONL、prompt、provider payload、assistant delta text 或 `tracestate`。
- 完成 S29 regression alert projection：
  - 新增 `clawTraceRegressionAlert.ts`，输出 severity、reason、primary owner、当前回退 delta、重复 owner 次数和 retained report window。
  - 新增 `clawTraceRegressionAlert.test.ts`，覆盖无证据 fail closed、单次 watch、大幅 warning、重复 owner critical，以及当前 report 已保存时不重复计数。
  - `ClawTraceRegressionReportCard` 增加 compact alert 状态行，继续使用既有 Save / Copy / Clear trend 显式动作，不后台读取 App Server。
  - 五语言 `settings.json` 补齐 alert 标题、severity、reason 和 summary 文案；UI 回归覆盖 App Server regression 时显示 Provider / API watch alert，且 raw provider payload 不泄露。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceRegressionAlert.test.ts" "src/lib/trace/clawTraceRegressionTrend.test.ts" "src/lib/trace/clawTraceRegressionReport.test.ts" "src/lib/trace/clawTraceAppServerComparison.test.ts" "src/lib/trace/clawTraceBaseline.test.ts" "src/lib/trace/clawTraceTimeline.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，7 个文件、42 个用例通过。
- 验证通过：`npx eslint "src/lib/trace/clawTraceRegressionAlert.ts" "src/lib/trace/clawTraceRegressionAlert.test.ts" "src/components/settings-v2/system/developer/ClawTraceRegressionReportCard.tsx" "src/components/settings-v2/system/developer/ClawTraceSettingsPanel.tsx" "src/components/settings-v2/system/developer/index.test.tsx" --max-warnings 0`。
- 验证通过：`npx prettier --check` 覆盖 S29 trace projector / Developer UI / 五语言 settings / trace 文档 / 执行计划写集；`git diff --check` 覆盖本轮 trace 写集；`rg -n "[ \t]+$"` 覆盖未跟踪 alert 新文件和文档，未发现尾随空白。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；本轮未新增 App Server method，contract guard 用于确认 current diagnostics trace / docs boundary 未漂移。
- 开始 S30 alert channel control：
  - 决策：S29 的 alert 仍只是 Developer 侧 summary-only projector；S30 增加显式 `developer.claw_trace.alert_enabled`，默认关闭，避免打开 Developer 设置页就评估 watch / warning / critical。
  - 决策：alert channel 只控制当前面板是否计算 alert，不影响 Claw Trace 采集开关、不触发后台采集、不自动保存 trend、不新增 App Server method。
  - 决策：`ClawTraceSettingsPanel.tsx` 已接近 800 行预警线，先把 trace enabled / level / sample_rate / alert channel 配置 UI 拆到独立组件，中心面板继续只做保存接线和数据装配。
- 完成 S30 alert channel control：
  - `ClawTraceConfig` 与 `normalizeDeveloperConfig` 支持 `alert_enabled`，默认值为 `false`，显式开启才允许 regression card 评估 alert。
  - 新增 `ClawTraceConfigControls.tsx`，承接 trace enabled / level / sample_rate / alert channel 配置 UI；保存逻辑仍由 `ClawTraceSettingsPanel` 注入。
  - `ClawTraceRegressionReportCard` 增加 `alertEnabled` 输入；关闭时只展示关闭态，不调用 `projectClawTraceRegressionAlert`。
  - 五语言 `settings.json` 补齐 alert channel 和关闭态文案；Developer UI 回归覆盖 alert channel 保存 `developer.claw_trace.alert_enabled=true`。
  - `ClawTraceSettingsPanel.tsx` 当前 734 行，低于 800 行预警线；`ClawTraceConfigControls.tsx` 117 行，`ClawTraceRegressionReportCard.tsx` 405 行。
- 验证通过：`npx vitest run "src/lib/developerFeatures.test.ts" "src/components/settings-v2/system/developer/index.test.tsx" "src/lib/trace/clawTraceRegressionAlert.test.ts"`，3 个文件、36 个用例通过。
- 验证通过：`npx eslint "src/lib/api/appConfigTypes.ts" "src/lib/developerFeatures.ts" "src/lib/developerFeatures.test.ts" "src/components/settings-v2/system/developer/ClawTraceConfigControls.tsx" "src/components/settings-v2/system/developer/ClawTraceSettingsPanel.tsx" "src/components/settings-v2/system/developer/ClawTraceRegressionReportCard.tsx" "src/components/settings-v2/system/developer/index.test.tsx" --max-warnings 0`。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；本轮未新增 App Server method，contract guard 用于确认 current diagnostics trace / docs boundary 未漂移。
- 验证通过：`npx prettier --check` 覆盖 S30 TS/TSX、五语言 settings、trace 文档与执行计划写集；`git diff --check` 覆盖 S30 写集。
- 验证未执行：`npm run verify:gui-smoke`。S30 只改 Developer 设置页显式配置、前端纯投影开关和文档，不改 Claw streaming 热路径、Electron bridge 或 App Server runtime 行为；本轮以定向 Vitest、ESLint、Prettier、diff 与 contract 作为最低门槛，后续触达主聊天 GUI 或 bridge 时再补 GUI smoke。
- 开始 S31 local alert channel inbox：
  - 决策：S30 只有显式开关，缺少真正可复制 / 可清理的告警通道记录；S31 先补 Developer 本地 retained inbox，不进入 App Server 后台查询或系统通知。
  - 决策：告警通道只在 `developer.claw_trace.alert_enabled=true` 且当前 panel 已有 watch / warning / critical 投影时写入；不保存 `none`，相同 alert/report fingerprint 去重。
  - 决策：通道只保存 alert、report verdict、owner totals、evidence sources 和窗口计数等 summary-only 字段；不保存 segments 以外 raw evidence、不保存 raw trace JSONL、prompt、provider payload、assistant delta text 或 `tracestate`。
- 完成 S31 local alert channel inbox：
  - 新增 `clawTraceRegressionAlertChannel.ts`，提供 list / overview / record / export / clear；retention 固定为最近 20 条 / 7 天。
  - `ClawTraceRegressionReportCard` 在 alert channel 开启且有 actionable alert 时写入本地 channel，并展示 channel count / latest severity / retention policy。
  - Developer UI 增加 Copy alert channel / Clear alert channel 显式动作；复制内容标记 `mode=summary_only_alert`，清空只删除本地 alert channel，不影响 trend/history/session。
  - 五语言 `settings.json` 补齐 alert channel 状态、retention、动作和成功 / 失败反馈文案。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceRegressionAlertChannel.test.ts" "src/lib/trace/clawTraceRegressionAlert.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，3 个文件、33 个用例通过。
- 验证通过：`npx eslint "src/lib/trace/clawTraceRegressionAlertChannel.ts" "src/lib/trace/clawTraceRegressionAlertChannel.test.ts" "src/lib/trace/clawTraceRegressionAlert.ts" "src/components/settings-v2/system/developer/ClawTraceRegressionReportCard.tsx" "src/components/settings-v2/system/developer/index.test.tsx" --max-warnings 0`。
- 验证通过：`npx prettier --check` 覆盖 S31 TS/TSX、五语言 settings、trace roadmap 与执行计划写集；`git diff --check` 覆盖 S31 写集；`rg -n "[ \t]+$"` 覆盖 S31 写集，未发现尾随空白。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；本轮未新增 App Server method，contract guard 用于确认 current diagnostics trace / docs boundary 未漂移。
- 行数复核：`ClawTraceSettingsPanel.tsx` 734 行，`ClawTraceConfigControls.tsx` 117 行，`ClawTraceRegressionReportCard.tsx` 536 行，`clawTraceRegressionAlertChannel.ts` 522 行，均低于 800 行预警线。
- 验证未执行：`npm run verify:gui-smoke`。S31 只改 Developer 设置页诊断卡、本地 summary-only retained channel 和文案，不改 Claw streaming 热路径、Electron bridge 或 App Server runtime 行为；本轮以定向 Vitest、ESLint、Prettier、diff 与 contract 作为最低门槛，后续触达主聊天 GUI 或 bridge 时再补 GUI smoke。
- 开始 S32 local notification dispatcher：
  - 决策：S32 先补本地 notification dispatcher 骨架，不直接接 Electron 原生通知、不新增 App Server method、不做常驻后台轮询。
  - 决策：新增 `developer.claw_trace.alert_notification_enabled`，默认关闭；`alert_enabled` 仍是 alert channel 总闸门，通知开关不能让关闭态告警后台采集或写入。
  - 决策：通知只在新写入 summary-only alert channel record 时尝试；相同 fingerprint 命中 existing record 时不重复通知。
  - 决策：浏览器 Notification adapter 不调用 `requestPermission`；只有既有权限为 `granted` 时才发送，`default / denied / unsupported` 都只返回状态，不弹权限请求。
- 完成 S32 local notification dispatcher：
  - 新增 `clawTraceRegressionAlertDispatcher.ts`，统一处理 alert gate、channel 写入、duplicate 判断、通知尝试和失败兜底，React 组件只负责输入与 overview 刷新。
  - 新增 `clawTraceRegressionAlertNotifier.ts`，提供浏览器 Notification adapter；不接 Electron bridge、不上传、不读取 App Server、不保存 raw trace。
  - `ClawTraceConfig` 与 `normalizeDeveloperConfig` 支持 `alert_notification_enabled=false` 默认值；`ClawTraceConfigControls` 增加桌面告警通知开关，五语言 settings 文案同步。
  - `ClawTraceRegressionReportCard` 改为通过 dispatcher 写入本地 channel，并只对新 record 尝试通知；alert channel export 仍是 `summary_only_alert`。
- 验证通过：`npx vitest run "src/lib/developerFeatures.test.ts" "src/lib/trace/clawTraceRegressionAlertNotifier.test.ts" "src/lib/trace/clawTraceRegressionAlertDispatcher.test.ts" "src/lib/trace/clawTraceRegressionAlertChannel.test.ts" "src/lib/trace/clawTraceRegressionAlert.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，6 个文件、51 个用例通过。
- 验证通过：`npx eslint` 覆盖 app config type、developer config normalizer、dispatcher / notifier、alert channel、Developer 设置页组件与测试写集。
- 验证通过：`npx prettier --check` 覆盖 S32 TS/TSX、五语言 settings、trace roadmap 与执行计划写集；`git diff --check` 覆盖 S32 写集；`rg -n "[ \t]+$"` 覆盖 S32 写集，未发现尾随空白。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；本轮未新增 App Server / Electron method，contract guard 用于确认 current diagnostics trace / docs boundary 未漂移。
- 行数复核：`ClawTraceSettingsPanel.tsx` 748 行，`ClawTraceConfigControls.tsx` 145 行，`ClawTraceRegressionReportCard.tsx` 582 行，`clawTraceRegressionAlertChannel.ts` 522 行，`clawTraceRegressionAlertDispatcher.ts` 140 行，`clawTraceRegressionAlertNotifier.ts` 75 行，均低于 800 行预警线。
- 验证未执行：`npm run verify:gui-smoke`。S32 只改 Developer 设置页配置 UI、本地 summary-only dispatcher / notifier 和文档，不改 Claw streaming 热路径、Electron bridge 或 App Server runtime 行为；本轮以定向 Vitest、ESLint、Prettier、diff 与 contract 作为最低门槛，后续触达主聊天 GUI、Electron bridge 或后台 OS 原生通知时再补 GUI smoke。
- 开始 S33 Electron desktop notification bridge：
  - 决策：桌面系统通知属于 Electron Desktop Host 壳能力，不进入 App Server JSON-RPC，也不恢复 legacy facade。
  - 决策：前端通过 `src/lib/api/desktopNotification.ts -> safeInvoke("show_desktop_notification")` 进入 Host；组件和 trace dispatcher 不直接裸调 IPC。
  - 决策：Host 只接受 `title / body / tag / silent`，并在 Host 侧裁剪 title/body/tag；拒绝 `raw_trace_jsonl` 等未声明字段，避免 notification payload 变成 raw trace 旁路。
  - 决策：S33 只把已有 Developer alert 的通知尝试切到 Electron 原生 Notification；仍不实现后台常驻轮询，不在打开设置页时自动查询 App Server。
- 完成 S33 Electron desktop notification bridge：
  - 新增 `electron/desktopNotificationHost.ts`，封装 Electron `Notification.isSupported()`、summary-only payload 校验、限长、tag -> id 映射和 `sent / unsupported / failed` 结果。
  - `electron/electronRuntime.ts` 导出 `Notification`；`electron/ipcChannels.ts` 白名单增加 `show_desktop_notification`；`ElectronHostCommands` 只做 dispatch 接线。
  - 新增 `src/lib/api/desktopNotification.ts` 作为 renderer API 网关，收到 diagnostic facade 或非预期返回时 fail closed。
  - `clawTraceRegressionAlertNotifier.ts` 增加 `desktopHostClawTraceRegressionAlertNotifier`，`ClawTraceRegressionReportCard` 改用 Desktop Host adapter；browser Notification adapter 仅保留为非 Electron / 单测边界。
  - `agentCommandCatalog.json` 将 `show_desktop_notification` 登记为 `systemUtilityCommands`，不新增 App Server method、不新增生产 mock。
- 验证通过：`npx tsc --noEmit --project "tsconfig.electron.json" --pretty false`。
- 验证通过：`npx vitest run "electron/ipcChannels.test.ts" "electron/hostCommands.test.ts" "src/lib/api/desktopNotification.test.ts" "src/lib/trace/clawTraceRegressionAlertNotifier.test.ts" "src/lib/trace/clawTraceRegressionAlertDispatcher.test.ts" "src/components/settings-v2/system/developer/index.test.tsx"`，6 个文件、134 个用例通过。
- 验证通过：`npx eslint "electron/electronRuntime.ts" "electron/desktopNotificationHost.ts" "electron/hostCommands.ts" "electron/hostCommands.test.ts" "electron/ipcChannels.ts" "electron/ipcChannels.test.ts" "src/lib/api/desktopNotification.ts" "src/lib/api/desktopNotification.test.ts" "src/lib/trace/clawTraceRegressionAlertNotifier.ts" "src/lib/trace/clawTraceRegressionAlertNotifier.test.ts" "src/components/settings-v2/system/developer/ClawTraceRegressionReportCard.tsx" --max-warnings 0`。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；新增 `show_desktop_notification` 已进入 Electron host command 契约，未新增 App Server method 或生产 mock。
- 验证通过：`npx prettier --check` 覆盖 S33 新增 Host / API / notifier / Developer card、治理目录册、trace roadmap 与执行计划写集；为避免格式化 `electron/hostCommands.ts` / `electron/hostCommands.test.ts` 中既有旧行，本轮这两个巨型文件由 ESLint、Vitest 和 `git diff --check` 覆盖。
- 验证未完成：前端全量 `npx tsc --noEmit --pretty false` 运行数分钟无输出后中断，退出码 `130`；本轮已有 Electron typecheck、定向 Vitest、ESLint、Prettier、diff 与 contracts 覆盖实际命令边界风险。
- 开始 S34 foreground global alert monitor：
  - 决策：S33 只解决“怎么发 Electron 原生通知”，没有解决离开 Developer 设置页后谁持续评估 alert；S34 补主窗口 foreground monitor，而不是 OS daemon 或 App Server 后台轮询。
  - 决策：monitor 只消费本地 Agent UI performance compact summary、compact history 和 regression trend；不调用 `diagnostics/trace/list`、`diagnostics/trace/read` 或其它 App Server trace API，避免每个对话/每次打开页面自动查 App Server。
  - 决策：`alert_enabled` 仍是总闸门，`alert_notification_enabled` 只控制新写入 alert record 后是否通过 Desktop Host notifier 尝试系统通知；Trace 关闭或 alert 关闭时不写 channel、不通知。
  - 决策：按 Codex rollout-trace 的分层原则，把触发源、projection、dispatcher 和 UI 接线拆开：metric recorded 事件只广播 summary-only 元信息，monitor service 做 projection，React hook 只做 App 主窗口挂载与 debounce。
- 完成 S34 foreground global alert monitor：
  - `agentUiPerformanceMetrics.ts` 新增 `lime:agent-ui-performance-metric-recorded` 本地事件和订阅 API；事件 detail 只有 `id / phase / sessionId / workspaceId / source`，不携带 raw metrics、prompt、provider payload 或 assistant delta。
  - 新增 `clawTraceRegressionAlertMonitor.ts`，从当前 compact summary 与 retained compact history 投影 baseline comparison、regression report 和 alert，然后复用 dispatcher 写入 summary-only alert channel；显式返回 `app_server_trace_requested=false`。
  - 新增 `clawTraceRegressionAlertPresentation.ts`，统一告警通知 title/body 文案投影，Developer card 和 global monitor 共用，避免通知文案拼装散落到组件。
  - 新增 `useClawTraceRegressionAlertMonitor.ts`，在主应用窗口加载 developer config、监听 app-config 变更和本地 metric recorded 事件，并 debounce 触发 monitor；`App.tsx` 挂载该 hook，独立 RootRouter 子窗口不挂载。
  - `ClawTraceRegressionReportCard` 改用 presentation helper 生成通知文案，保留 Developer 设置页的手动 trend / alert channel 操作。
- 验证通过：`npx vitest run "src/lib/agentUiPerformanceMetrics.test.ts" "src/lib/trace/clawTraceRegressionAlertPresentation.test.ts" "src/lib/trace/clawTraceRegressionAlertMonitor.test.ts" "src/hooks/useClawTraceRegressionAlertMonitor.test.tsx" "src/components/settings-v2/system/developer/index.test.tsx"`，5 个文件、36 个用例通过。
- 验证通过：`npx vitest run "src/lib/trace/clawTraceRegressionAlertNotifier.test.ts" "src/lib/api/desktopNotification.test.ts" "src/lib/trace/clawTraceRegressionAlertDispatcher.test.ts"`，3 个文件、15 个用例通过。
- 验证通过：`npx eslint "src/App.tsx" "src/hooks/useClawTraceRegressionAlertMonitor.ts" "src/hooks/useClawTraceRegressionAlertMonitor.test.tsx" "src/lib/agentUiPerformanceMetrics.ts" "src/lib/agentUiPerformanceMetrics.test.ts" "src/lib/trace/clawTraceRegressionAlertMonitor.ts" "src/lib/trace/clawTraceRegressionAlertMonitor.test.ts" "src/lib/trace/clawTraceRegressionAlertPresentation.ts" "src/lib/trace/clawTraceRegressionAlertPresentation.test.ts" "src/components/settings-v2/system/developer/ClawTraceRegressionReportCard.tsx" --max-warnings 0`。
- 验证通过：`npx tsc --noEmit --project "tsconfig.electron.json" --pretty false`；`npx tsc --noEmit --project "tsconfig.node.json" --pretty false`。
- 验证未完成：前端全量 `npx tsc --noEmit --pretty false` 运行约数分钟后触发 TypeScript 编译器内部错误 `Debug Failure. No error for last overload signature`，没有给出源码文件；这不是普通 TS 类型错误。S34 已用定向 Vitest / ESLint / Electron+Node typecheck 覆盖新增写集，后续全量收口需单独定位 TS 23.4.0 + 当前 workspace 的 compiler crash。
- 开始 S35 notification host test split：
  - 决策：参考 Codex 中心文件只做 dispatch 的方式，继续把 Electron Desktop Host 通知细节留在 `desktopNotificationHost.ts`，`hostCommands.ts` / `hostCommands.test.ts` 只保留命令分发级别的 smoke。
  - 决策：S35 不新增命令、不改变 `show_desktop_notification` payload 合同、不新增 App Server method；目标只是把 summary-only 校验、unsupported 和 failure path 回归从超大 `hostCommands.test.ts` 拆到模块级测试。
- 完成 S35 notification host test split：
  - 新增 `electron/desktopNotificationHost.test.ts`，覆盖 title/body/tag/silent summary-only 校验、unsupported 返回、Notification 构造失败和 show 失败等 Host 细节。
  - `electron/hostCommands.test.ts` 只保留 `show_desktop_notification` dispatcher smoke，避免超大 Host command 测试继续承接 payload 校验细节；dispatcher 测试改为只断言调用 `desktopNotificationHost`，不再直接关注 Electron Notification 构造器。
  - `electron/hostCommands.ts` 仍只做 `show_desktop_notification` dispatch 接线；实际通知 payload 校验、限长和发送结果仍在 `desktopNotificationHost.ts`。
- 进一步收紧：`hostCommands.test.ts` 不再直接 mock Electron `Notification`，只验证 dispatcher 到 `desktopNotificationHost` 的调用链。
- 验证通过：`npx vitest run "electron/desktopNotificationHost.test.ts" "electron/hostCommands.test.ts" "src/lib/api/desktopNotification.test.ts"`，3 个文件、94 个用例通过。
- 验证通过：`npx eslint "electron/desktopNotificationHost.ts" "electron/desktopNotificationHost.test.ts" "electron/hostCommands.ts" "electron/hostCommands.test.ts" "src/lib/api/desktopNotification.ts" "src/lib/api/desktopNotification.test.ts" --max-warnings 0`。
- 验证通过：`npx prettier --check "electron/desktopNotificationHost.test.ts" "electron/hostCommands.test.ts" "internal/exec-plans/claw-trace-system-implementation-plan.md"`。
- 验证通过：`git diff --check -- "electron/desktopNotificationHost.test.ts" "electron/hostCommands.test.ts" "internal/exec-plans/claw-trace-system-implementation-plan.md"`。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；S35 未新增 App Server method 或生产 mock，只验证 Electron Desktop Host 命令契约未漂移。
- 开始 S36 project shell host split：
  - 决策：继续参考 Codex dispatcher-only 方式，把 Project Shell PTY 会话状态、event poller、drain、write 后主动 drain 和 shutdown dispose 从超大 `hostCommands.ts` 下沉到独立 `electron/projectShellHost.ts`。
  - 决策：S36 不新增命令、不改变 `project_shell_session_*` payload 合同、不新增 App Server method、不改前端 `src/lib/api/projectShell.ts` 网关；只重排 Electron Desktop Host 内部职责。
  - 决策：`hostCommands.test.ts` 只保留 `run_project_shell_command` 与 `project_shell_session_*` dispatcher smoke；App Server PTY method 参数、event drain、写入后 drain 和 shutdown kill 细节由 `projectShellHost.test.ts` 覆盖。
- 完成 S36 project shell host split：
  - 新增 `electron/projectShellHost.ts`，承接 `runCommand / startSession / writeSession / resizeSession / killSession / disposeForShutdown`。
  - 新增 `electron/projectShellHost.test.ts`，覆盖 Project Shell current 封装、App Server PTY methods、event drain 转发、写入后主动 drain、shutdown dispose。
  - `electron/hostCommands.ts` 只保留 Project Shell command dispatch，`hostCommands.test.ts` 只验证分发到 `ProjectShellHost`。
  - `electron/hostCommands.ts` 从约 `3914` 行降到约 `3735` 行，但仍超过体量边界；下一刀继续拆 system utility / file shell / agent app shell。
- 验证通过：`npx vitest run "electron/projectShellHost.test.ts" "electron/hostCommands.test.ts"`，2 个文件、89 个用例通过。
- 验证通过：`npx eslint "electron/projectShellHost.ts" "electron/projectShellHost.test.ts" "electron/hostCommands.ts" "electron/hostCommands.test.ts" --max-warnings 0`。
- 验证通过：`npx tsc --noEmit --project "tsconfig.electron.json" --pretty false`。
- 验证通过：`npx prettier --check "electron/projectShellHost.ts" "electron/projectShellHost.test.ts" "electron/hostCommands.test.ts" "internal/exec-plans/claw-trace-system-implementation-plan.md"`。
- 验证通过：`git diff --check -- "electron/projectShellHost.ts" "electron/projectShellHost.test.ts" "electron/hostCommands.ts" "electron/hostCommands.test.ts" "internal/exec-plans/claw-trace-system-implementation-plan.md"`。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；S36 未新增 App Server method、Electron command 或生产 mock，只验证 Electron Desktop Host 命令契约未漂移。
- 验证未通过：`npm run verify:local` 在全仓库 smart 模式的 `npm run i18n:unused -- --check` 失败，报告 `agent: plugin.marketplace.cloudRequired.*` 与 `agent: plugin.marketplace.filter.*` 共 7 个未引用 key；这些文件不在 S36 Project Shell Host 写集内，本轮不顺手修改无关 i18n 脏改。
- 开始 S37 file shell host split：
  - 决策：继续参考 Codex dispatcher-only 方式，把 `open_file_preview_window / reveal_in_finder / open_with_default_app / get_file_icon_data_url / get_home_dir / get_file_manager_locations` 从超大 `hostCommands.ts` 下沉到独立 `electron/fileShellHost.ts`。
  - 决策：S37 不新增命令、不改变 payload 合同、不新增 App Server method、不改前端 API 网关；只重排 Electron Desktop Host 内部职责。
  - 决策：`hostCommands.test.ts` 只保留 File Shell dispatcher smoke；BrowserWindow 预览复用、相对路径拒绝、Electron shell 错误和文件管理器位置去重由 `fileShellHost.test.ts` 覆盖。
- 完成 S37 file shell host split：
  - 新增 `electron/fileShellHost.ts`，承接 `openFilePreviewWindow / revealInFinder / openWithDefaultApp / getFileIconDataUrl / getHomeDir / getFileManagerLocations`。
  - 新增 `electron/fileShellHost.test.ts`，覆盖 BrowserWindow 预览窗口新建与复用、相对路径拒绝、Finder/默认应用打开、文件图标失败隔离、系统主目录 fail closed、文件管理器位置去重。
  - `electron/hostCommands.ts` 只保留 File Shell command dispatch，`hostCommands.test.ts` 只验证分发到 `FileShellHost`。
  - `electron/hostCommands.ts` 从约 `3735` 行降到约 `3506` 行，但仍超过体量边界；下一刀继续拆 system utility / agent app shell。
- 验证通过：`npx vitest run "electron/fileShellHost.test.ts" "electron/hostCommands.test.ts"`，2 个文件、85 个用例通过。
- 验证通过：`npx eslint "electron/fileShellHost.ts" "electron/fileShellHost.test.ts" "electron/hostCommands.ts" "electron/hostCommands.test.ts" --max-warnings 0`。
- 验证通过：`npx tsc --noEmit --project "tsconfig.electron.json" --pretty false`。
- 验证通过：`npx prettier --check "electron/fileShellHost.ts" "electron/fileShellHost.test.ts" "electron/hostCommands.test.ts" "internal/exec-plans/claw-trace-system-implementation-plan.md"`。
- 验证通过：`git diff --check -- "electron/fileShellHost.ts" "electron/fileShellHost.test.ts" "electron/hostCommands.ts" "electron/hostCommands.test.ts" "internal/exec-plans/claw-trace-system-implementation-plan.md"`。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；S37 未新增 App Server method、Electron command 或生产 mock，只验证 Electron Desktop Host 命令契约未漂移。
- 验证未通过：`npm run verify:local` 仍在全仓库 smart 模式的 `npm run i18n:unused -- --check` 失败，报告 `agent: plugin.marketplace.cloudRequired.*` 与 `agent: plugin.marketplace.filter.*` 共 7 个未引用 key；这些文件不在 S37 File Shell Host 写集内，本轮不顺手修改无关 i18n 脏改。
- 开始 S38 system utility host split：
  - 决策：继续参考 Codex dispatcher-only 方式，把 `open_external_url / open_system_settings_url / get_voice_shortcut_runtime_status / validate_shortcut / get_environment_preview` 以及文件关联、浏览器连接器、Chrome bridge degraded diagnostic 占位从超大 `hostCommands.ts` 下沉到独立 `electron/systemUtilityHost.ts`。
  - 决策：S38 不新增命令、不改变 payload 合同、不新增 App Server method、不改前端 API 网关；只重排 Electron Desktop Host 内部职责。
  - 决策：`hostCommands.test.ts` 只保留 System Utility dispatcher smoke；URL scheme 校验、语音快捷键 fallback、环境预览脱敏、文件关联和浏览器 diagnostic 占位由 `systemUtilityHost.test.ts` 覆盖。
- 完成 S38 system utility host split：
  - 新增 `electron/systemUtilityHost.ts`，承接外链/系统设置打开、语音快捷键状态与校验、环境预览、`.skill/.skills` 文件关联 degraded 状态、浏览器连接器/Chrome bridge/backend degraded 状态。
  - 新增 `electron/systemUtilityHost.test.ts`，覆盖 URL scheme fail closed、快捷键解析与保留组合拒绝、无效语音快捷键回退默认值、环境变量预览 API key 脱敏、diagnostic 占位形态和 `get_chrome_profile_sessions` 非枚举 diagnostic metadata。
  - `electron/hostCommands.ts` 只保留 System Utility command dispatch，`hostCommands.test.ts` 只验证分发到 `SystemUtilityHost`。
  - 同步 `scripts/check-command-contracts.mjs`，让契约守卫继续检查 `hostCommands.ts` 的 dispatcher case，同时在 `systemUtilityHost.ts` 中确认浏览器 degraded diagnostic 投影，避免重构后守卫误要求逻辑回流中心文件。
  - `electron/hostCommands.ts` 从约 `3506` 行降到约 `3117` 行，但仍超过体量边界；下一刀继续拆 agent app shell / voice model / layered design export 等剩余大块。
- 验证通过：`npx vitest run "electron/systemUtilityHost.test.ts" "electron/hostCommands.test.ts"`，2 个文件、72 个用例通过。
- 验证通过：`npx eslint "scripts/check-command-contracts.mjs" "electron/systemUtilityHost.ts" "electron/systemUtilityHost.test.ts" "electron/hostCommands.ts" "electron/hostCommands.test.ts" --max-warnings 0`。
- 验证通过：`npx tsc --noEmit --project "tsconfig.electron.json" --pretty false`。
- 验证通过：`npx prettier --check "scripts/check-command-contracts.mjs" "electron/systemUtilityHost.ts" "electron/systemUtilityHost.test.ts" "electron/hostCommands.test.ts" "internal/exec-plans/claw-trace-system-implementation-plan.md"`。
- 验证通过：`git diff --check -- "scripts/check-command-contracts.mjs" "electron/systemUtilityHost.ts" "electron/systemUtilityHost.test.ts" "electron/hostCommands.ts" "electron/hostCommands.test.ts" "internal/exec-plans/claw-trace-system-implementation-plan.md"`。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；S38 未新增 App Server method、Electron command 或生产 mock，只同步 Electron Desktop Host 内部 owner 与命令契约守卫。
- 验证未通过：`npm run verify:local` 仍在全仓库 smart 模式的 `npm run i18n:unused -- --check` 失败，报告 `agent: plugin.marketplace.cloudRequired.*` 与 `agent: plugin.marketplace.filter.*` 共 7 个未引用 key；这些文件不在 S38 System Utility Host 写集内，本轮不顺手修改无关 i18n 脏改。
- 开始 S39 agent app shell host split：
  - 决策：继续参考 Codex dispatcher-only 方式，把 `agent_app_select_directory / agent_app_launch_shell / agent_app_start_ui_runtime / agent_app_get_ui_runtime_status / agent_app_stop_ui_runtime` 从超大 `hostCommands.ts` 下沉到独立 `electron/agentAppShellHost.ts`。
  - 决策：S39 不新增命令、不改变 payload 合同、不新增 App Server method、不改前端 `src/lib/api/agentApps.ts` 网关；只重排 Electron Desktop Host 内部职责。
  - 决策：`hostCommands.test.ts` 只保留 Agent App shell dispatcher smoke；Electron directory picker、App Server `agentAppShell/prepare`、UI runtime `start/status/stop`、BrowserWindow 复用与 fail closed 细节由 `agentAppShellHost.test.ts` 覆盖。
- 完成 S39 agent app shell host split：
  - 新增 `electron/agentAppShellHost.ts`，承接 Agent App 目录选择、shell prepare、UI runtime 启停/状态、BrowserWindow 打开与复用、surface projection。
  - 新增 `electron/agentAppShellHost.test.ts`，覆盖 directory picker、launch success、窗口复用、prepare blocked、descriptor invalid 和 UI runtime lifecycle method 透传。
  - `electron/hostCommands.ts` 只保留 Agent App shell command dispatch，`hostCommands.test.ts` 只验证分发到 `AgentAppShellHost`。
  - 同步 `scripts/check-app-server-client-contract.mjs`，让 App Server client contract 继续检查 `hostCommands.ts` 的 dispatcher case，同时把 `agentAppUiRuntime/*` lifecycle 实现 owner 更新为 `agentAppShellHost.ts`。
  - `electron/hostCommands.ts` 从约 `3117` 行降到约 `2772` 行，但仍超过体量边界；下一刀继续拆 Agent App runtime task / voice model / layered design export 等剩余大块。
- 验证通过：`npx vitest run "electron/agentAppShellHost.test.ts" "electron/hostCommands.test.ts"`，2 个文件、68 个用例通过。
- 验证通过：`npx eslint "scripts/check-app-server-client-contract.mjs" "electron/agentAppShellHost.ts" "electron/agentAppShellHost.test.ts" "electron/hostCommands.ts" "electron/hostCommands.test.ts" --max-warnings 0`。
- 验证通过：`npx tsc --noEmit --project "tsconfig.electron.json" --pretty false`。
- 验证通过：`npx prettier --check "scripts/check-app-server-client-contract.mjs" "electron/agentAppShellHost.ts" "electron/agentAppShellHost.test.ts" "electron/hostCommands.test.ts" "internal/exec-plans/claw-trace-system-implementation-plan.md"`。
- 验证通过：`git diff --check -- "scripts/check-app-server-client-contract.mjs" "electron/agentAppShellHost.ts" "electron/agentAppShellHost.test.ts" "electron/hostCommands.ts" "electron/hostCommands.test.ts" "internal/exec-plans/claw-trace-system-implementation-plan.md"`。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；S39 未新增 App Server method、Electron command 或生产 mock，只同步 Electron Desktop Host 内部 owner 与 App Server client / command 契约守卫。
- 验证未通过：`npm run verify:local` 仍在全仓库 smart 模式的 `npm run i18n:unused -- --check` 失败，报告 `agent: plugin.marketplace.cloudRequired.*` 与 `agent: plugin.marketplace.filter.*` 共 7 个未引用 key；这些文件不在 S39 Agent App Shell Host 写集内，本轮不顺手修改无关 i18n 脏改。
- 开始 S40 agent app runtime task host split：
  - 决策：继续参考 Codex 中心文件只做 dispatch 的方式，把 `agent_app_runtime_start_task / get_task / cancel_task / submit_host_response` 从超大 `hostCommands.ts` 下沉到独立 `electron/agentAppRuntimeTaskHost.ts`。
  - 决策：S40 不新增命令、不改前端 API 网关、不新增 App Server method；`agent_app_runtime_*` 仍是 Agent App 进入 App Server / AgentRuntime current 主链的 Desktop facade。
  - 决策：`hostCommands.test.ts` 只保留 Agent App runtime task dispatcher smoke；`runWorker=false` 不查询 UI runtime status、`turnConfig` 透传、host response / cancel / read 细节由 `agentAppRuntimeTaskHost.test.ts` 覆盖。
- 完成 S40 agent app runtime task host split：
  - 新增 `electron/agentAppRuntimeTaskHost.ts`，承接 Agent App runtime task start/read/cancel/host response，并通过 App Server current `agentSession/turn/start` 等方法收敛。
  - 新增 `electron/agentAppRuntimeTaskHost.test.ts`，覆盖 `turnConfig` -> `RuntimeOptions.hostOptions.agentChatRequest` 透传、`runWorker=false` 不查询 UI runtime status、start/read/cancel/host response current 投影。
  - `electron/hostCommands.ts` 只保留 `agent_app_runtime_*` command dispatch，`hostCommands.test.ts` 只验证分发到 `AgentAppRuntimeTaskHost`。
  - 同步 `scripts/check-app-server-client-contract.mjs`，让 App Server client contract 继续检查 `agent_app_runtime_*` dispatcher case 与新 Host owner。
  - `electron/hostCommands.ts` 从约 `2772` 行降到约 `2221` 行，但仍超过体量边界；下一刀继续拆 voice model / layered design export 等剩余大块。
- 验证通过：`npx vitest run "electron/agentAppRuntimeTaskHost.test.ts" "electron/hostCommands.test.ts"`，2 个文件、63 个用例通过。
- 开始 S41 independent Trace workspace tab：
  - 决策：Trace 是 Workspace 右侧 surface 中与 Harness 同级的一等 tab，不是 Harness 子页；Harness 只消费 evidence，不拥有 trace schema、采集、开关或 UI 主入口。
  - 决策：Trace Tab 首版只消费已有 `summarizeAgentUiPerformanceMetrics()` compact summary 与本地 metric event，不在 mount 时主动查询 App Server trace list/read，不新增协议或生产 mock。
  - 决策：JSON / JSONL evidence 只作为复制 compact diagnostic summary 的 secondary action；主视图面向人展示首字分段、客户端可优化段、慢段归因、客户端健康和阶段覆盖，不展示 raw entries / prompt / provider payload / assistant delta。
- 完成 S41 independent Trace workspace tab：
  - 新增 `src/components/agent/chat/workspace/WorkspaceTraceTab.tsx` 和纯 ViewModel `workspaceTracePanelModel.ts`；组件只渲染人读 summary，ViewModel 负责 client/server/bridge 分段、client actionable 汇总、慢段排序、missing phase 与 health metrics。
  - Trace 接入 `right-surface` registry / type / runtime projection / `RightSurfaceHost` tab label，并由 `AgentChatWorkspace` 作为 `rightSurfaceDefinitions.trace` 渲染；关闭 developer Trace 或隐藏 navbar utility actions 时会清理/禁用 Trace surface。
  - `TaskCenterUtilityToolbar` 增加 icon-only Trace toggle，与 Harness 同级；`WorkspaceConversationScene` / `useWorkspaceConversationSceneRuntime` 透传 Trace toggle 状态与点击回调。
  - 五语言 `agent.json` 补齐 `navbar.trace/openTrace/closeTrace`、`rightSurface.tabs.trace` 和 `tracePanel.*` 文案；本轮补齐此前遗漏的 `ko-KR`。
  - 新增 `WorkspaceTraceTab.test.tsx`，覆盖分段归因、client actionable 只计 client 段、空态、主面板不泄露 raw entries，以及复制 evidence 只输出 compact diagnostic summary。
  - 更新 toolbar / scene / right-surface projection / registry 单测，守住 Trace 作为 Harness 同级 surface、available/disabled/pending 和工具栏入口状态。
  - 修复 `AgentChatWorkspace.tsx` 中 Trace 透传重复 key，消除 smoke build 阶段的 duplicate key warning。
- 验证通过：`npx eslint "src/components/agent/chat/workspace/WorkspaceTraceTab.tsx" "src/components/agent/chat/workspace/workspaceTracePanelModel.ts" "src/components/agent/chat/workspace/WorkspaceTraceTab.test.tsx" "src/components/agent/chat/components/TaskCenterUtilityToolbar.tsx" "src/components/agent/chat/components/TaskCenterUtilityToolbar.integration.test.tsx" "src/components/agent/chat/workspace/WorkspaceConversationScene.tsx" "src/components/agent/chat/workspace/WorkspaceConversationScene.test.tsx" "src/components/agent/chat/workspace/right-surface/rightSurfaceRegistry.unit.test.ts" "src/components/agent/chat/workspace/right-surface/rightSurfaceToolbarProjection.unit.test.ts" "src/components/agent/chat/workspace/workspaceRightSurfaceRuntimeProjection.ts" "src/components/agent/chat/workspace/workspaceRightSurfaceRuntimeProjection.unit.test.ts" --max-warnings 0`。
- 验证通过：`npx eslint "src/components/agent/chat/AgentChatWorkspace.tsx" --max-warnings 0`。
- 验证通过：`npx vitest run "src/components/agent/chat/workspace/WorkspaceTraceTab.test.tsx" "src/components/agent/chat/components/TaskCenterUtilityToolbar.integration.test.tsx" "src/components/agent/chat/workspace/WorkspaceConversationScene.test.tsx" "src/components/agent/chat/workspace/right-surface/rightSurfaceRegistry.unit.test.ts" "src/components/agent/chat/workspace/right-surface/rightSurfaceToolbarProjection.unit.test.ts" "src/components/agent/chat/workspace/workspaceRightSurfaceRuntimeProjection.unit.test.ts"`，6 个文件、73 个用例通过；`WorkspaceConversationScene.test.tsx` 仍输出既有 i18next test-env warning，但退出码为 `0`。
- 验证通过：`npx prettier --check` 覆盖 S41 Trace Tab / right-surface / toolbar / scene 测试、`ko-KR` agent 文案与执行计划写集。
- 验证通过：`git diff --check -- ...` 覆盖 S41 Trace Tab / right-surface / toolbar / scene 测试、`ko-KR` agent 文案与执行计划写集。
- 验证通过：`npm run test:contracts`。命令输出仍提示本地 ignored `scripts/__pycache__` / `.pyc` 存在但不会提交，退出码为 `0`；S40/S41 未新增 App Server method 或生产 mock，只同步 Electron Desktop Host 内部 owner 与 right-surface 前端 UI。
- 验证通过：`npm run verify:gui-smoke`。首次运行通过但 Vite 报告 Trace 透传重复 key warning；修复后第二次重跑通过，未再出现 duplicate key warning。该 smoke 覆盖 renderer smoke build、`packages/app-server-client` build、`typecheck:electron`、Electron host build、App Server sidecar build、renderer loaded、app-server initialized、claw workbench shell ready、memory settings ready。
- 验证未完成：前端全量 `npx tsc --noEmit --project "tsconfig.json" --pretty false` 运行超过 5 分钟无输出后手动中断，退出码 `130`。本轮已有定向 ESLint、Vitest、Electron typecheck（经 `verify:gui-smoke`）、App Server client build、contract guard 和 GUI smoke 覆盖新增写集；全量前端 typecheck 仍需后续单独窗口重跑或定位仓库级耗时问题。
- 开始 S46 Workspace Trace adaptive layout：
  - 决策：Trace / Harness / expert info 等 Right Surface 是辅助诊断或辅助信息面板，打开后不能按普通代码画布比例压缩主对话；没有显式 `chatPanelWidth` 时使用 Right Surface 专用默认聊天宽度。
  - 决策：普通对话态不能把消息正文、助手气泡和输入框分别写死成不同宽度；统一走 Codex-style 自适应对话内容轨道 `clamp(640px, 68%, 720px)`，宽屏保持窄阅读列，中窄屏继续按容器可用宽度自适应。
  - 决策：不新增协议、不新增文案、不改变 Trace 数据采集；本轮只修 Workspace 可见布局。
- 完成 S46 Workspace Trace adaptive layout：
  - 新增 `conversationLayoutTokens.ts`，集中定义对话内容轨道宽度；`MessageList`、assistant bubble、inline input、floating input 和 plan decision inputbar replacement 统一消费该 token。
  - `WorkspaceMainArea` 在 `rightSurfaceContent` 存在且未显式传入 `chatPanelWidth` 时使用 `RIGHT_SURFACE_CHAT_PANEL_WIDTH / MIN_WIDTH`，避免 Trace 打开后主对话被旧 `chat-canvas` 默认值挤窄；显式 code workbench / subagents 宽度不被覆盖。
  - `LayoutTransition` 暴露 `data-chat-panel-min-width`，让右侧辅助栏分栏规则可被稳定断言。
- 验证通过：`./node_modules/.bin/vitest run "src/components/agent/chat/workspace/WorkspaceMainArea.test.tsx" "src/components/agent/chat/components/MessageList.test.tsx" "src/components/agent/chat/components/MessageList.messageActions.test.tsx"`，3 个文件、39 个用例通过。
- 验证通过：`./node_modules/.bin/eslint "src/components/agent/chat/styles/conversationLayoutTokens.ts" "src/components/agent/chat/workspace/WorkspaceMainArea.tsx" "src/components/agent/chat/workspace/WorkspaceStyles.tsx" "src/components/agent/chat/workspace/useWorkspaceInputbarSceneRuntime.tsx" "src/components/workspace/layout/LayoutTransition.tsx" "src/components/agent/chat/styles/index.ts" "src/components/agent/chat/components/MessageList.tsx" "src/components/agent/chat/workspace/WorkspaceMainArea.test.tsx" "src/components/agent/chat/components/MessageList.messageActions.test.tsx" --max-warnings 0`。
- 验证通过：`git diff --check` 覆盖 S46 写集。
- 验证通过：`npm run verify:gui-smoke` 覆盖 renderer smoke build、Electron host build、App Server sidecar、renderer loaded、app-server initialized、claw workbench shell ready 和 memory settings ready。

## 10. 当前下一刀

1. 已完成的诊断闭环：
   - Developer & Labs 可以保存当前 compact Trace summary 快照、复制 compact history JSON、清空 history。
   - history 本地保留最近 20 个快照 / 7 天，并明确不保存 raw entries / prompt / provider payload。
   - App Server 已有 summary-only raw trace JSONL store，并通过 `diagnostics/trace/list`、`diagnostics/trace/read` 提供开发者读面。
   - support bundle 默认包含 `meta/trace-store-summary.json`，但不包含 raw trace JSONL 正文。
   - Developer UI 可显式导出最近单条 summary-only trace zip，默认支持包仍不含 raw JSONL 正文。
   - Developer UI 可显式导出“带最近 Trace 的支持包”，默认支持包和普通对话仍不会查询或附带 trace zip。
   - Developer UI 可显式加载 Trace timeline，并从 summary-only events 展示 phase spans、slow segments、缺失 phase、App Server retained-window trace compare、regression evidence 归因报告、手动 regression trend history 和 summary-only regression alert；alert channel 默认关闭，只有 `developer.claw_trace.alert_enabled=true` 时才评估 watch / warning / critical，并把 actionable alert 写入本地 summary-only channel；`alert_notification_enabled` 默认关闭，只在新写入 alert 时通过 Electron Desktop Host 原生 Notification 尝试桌面通知；主窗口 foreground monitor 已能在离开 Developer 设置页后监听本地 compact metric 事件并 debounce 评估 alert，但不查询 App Server trace list/read。
   - support bundle trace opt-in 已通过真实 Electron fixture 证明走 current App Server method，并使用 RuntimeCore 当前 trace root，不再由 support bundle 自己推断 trace store。
2. 已完成 S18：真实 OTEL exporter / W3C remote parent 接入。App Server request span 已能通过 exporter 继承 renderer carrier 的 trace id / parent span id，生产 OTLP 导出默认关闭、显式环境变量开启。
3. 已完成 S19：合法 W3C carrier 已从 App Server turn context 透传到 Agent provider HTTP request headers；非法 carrier 不注入，`tracestate` 不单独传播。
4. 已完成 S20：provider response header request id 已从 Agent 统一 HTTP builder 进入 `ProviderTraceEvent`、RuntimeEvent 和 summary-only trace metrics，方便把本地 trace 与上游 provider 工单 / 后端日志关联。
5. 已完成 S21：Developer UI 可对 summary-only timeline 做 all / phase / slow filter，并查看选中事件的 safe metrics 详情。
6. 已完成 S22：Developer UI phase span 可点击定位 span 内事件，并在详情区展示当前 span summary。
7. 当前 Trace 主链已覆盖：renderer checkpoint、App Server checkpoint、provider phase、summary-only raw trace store、diagnostics read/export、support bundle opt-in、span diagnostics、W3C carrier、App Server request span、OTEL remote parent、provider HTTP header propagation、provider request id correlation 与 Developer drilldown。
8. 已完成 S23：基于 compact Trace history 的 trace compare / regression baseline，可判断当前本地输出分段是否相对最近快照回退。
9. 已完成 S24：App Server summary-only trace compare，用最近 trace 与上一条 trace 对比 provider/API 与 App Server 分段。
10. 已完成 S25：compact history retained window 的长期 baseline，避免 baseline 随最近慢样本漂移。
11. 已完成 S26：App Server retained trace window 的长期 baseline，避免最近慢 trace 成为新的 App Server compare baseline。
12. 已完成 S27：regression evidence 归因报告，合并 compact client 分段与 App Server provider/API 分段，明确回退焦点是 Provider / API、App Server 还是 Lime 本地输出。
13. 已完成 S28：手动 regression trend history，开发者可保存、复制、清空 retained regression report，用于跨运行追踪归因变化。
14. 已完成 S29：summary-only regression alert 投影，开发者可在当前 report 与 retained trend 基础上看到 watch / warning / critical 状态。
15. 已完成 S30：alert channel 显式开关与配置控件拆分，`alert_enabled` 默认关闭，中心面板降到 800 行预警线以下。
16. 已完成 S31：本地 summary-only alert channel inbox，支持 retained channel 写入、复制与清空。
17. 已完成 S32：本地通知 dispatcher 骨架，支持 `alert_notification_enabled` 显式开关、重复 fingerprint 不重复通知。
18. 已完成 S33：Electron Desktop Host 原生通知桥，`show_desktop_notification` 只接受 summary-only title/body/tag/silent，经前端 API 网关接入 trace notifier。
19. 已完成 S34：主应用窗口 foreground global alert monitor，离开 Developer 设置页后仍可基于本地 compact summary/history 持续评估 alert；不做 OS daemon、不新增 App Server method、不自动读取 App Server trace list/read。
20. 已完成 S35：通知 Host 细节回归已拆到 `electron/desktopNotificationHost.test.ts`，`hostCommands.test.ts` 只保留 `show_desktop_notification` dispatcher smoke；不新增命令、不改变 summary-only payload 合同。
21. 已完成 S36：Project Shell PTY 会话状态、poller、drain 和 shutdown dispose 已拆入 `electron/projectShellHost.ts`；`hostCommands.test.ts` 只保留分发 smoke，不再承接 PTY current 细节。
22. 已完成 S37：File Shell 本地壳能力已拆入 `electron/fileShellHost.ts`；`hostCommands.test.ts` 只保留分发 smoke，不再承接 BrowserWindow / Electron shell / 文件管理器位置细节。
23. 已完成 S38：System Utility 本地壳能力已拆入 `electron/systemUtilityHost.ts`；`hostCommands.test.ts` 只保留分发 smoke，不再承接 URL scheme、快捷键、环境预览、文件关联和浏览器 degraded diagnostic 细节。
24. 已完成 S39：Agent App shell 本地壳能力已拆入 `electron/agentAppShellHost.ts`；`hostCommands.test.ts` 只保留分发 smoke，不再承接 directory picker、shell prepare、UI runtime lifecycle 与 BrowserWindow 细节。
25. 已完成 S40：Agent App runtime task start/read/cancel/host response 已拆入 `electron/agentAppRuntimeTaskHost.ts`；`hostCommands.test.ts` 只保留分发 smoke，不再承接 `turnConfig` 透传、`runWorker=false` 或 task lifecycle current 细节。
26. 已完成 S41：Workspace 右侧 surface 已增加与 Harness 同级的 Trace Tab；主视图展示人读首字分段、慢段归因、客户端健康和阶段缺口，compact JSON evidence 只作为复制摘要动作，不作为主界面。
27. 已完成 S46：Workspace 对话内容轨道已统一到 Codex-style 自适应宽度 `clamp(640px, 68%, 720px)`，消息正文 / 助手气泡 / 输入框统一口径；Right Surface 打开时使用辅助栏专用聊天宽度，不再沿用旧 `chat-canvas` 默认比例挤压主对话。
28. 仍未完成：系统级后台 daemon / 应用完全关闭后的告警、Trace Tab retained baseline / App Server timeline drilldown 的进一步人读化、以及 `agentProtocol.ts` / `agent.rs` / `electron/hostCommands.ts` 等超大文件的后续拆分；`hostCommands.ts` 当前约 `2221` 行，下一刀优先继续拆 voice model / layered design export 等剩余大块。
