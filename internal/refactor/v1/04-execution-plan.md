# 执行计划

> 2026-07-19 快速骨架 checkpoint：先落地 v2 `turn/steer`、canonical history normalize、provider incremental stream 和 Renderer canonical thread projection；后续 current home-hotpath Gate B 已 68/68 通过，但本 checkpoint 仍不关闭 D1/D2，也不替代 workspace/Cargo、V1-15 全场景和完整 Codex 能力验收。

## 总目标

将 Lime 收敛为 Codex-first 的单一 Agent runtime：

```text
Electron Desktop Host
  -> App Server JSON-RPC v2
  -> RuntimeCore / agent-runtime
  -> Thread/Turn/Item + ThreadStore
  -> model-provider / tool-runtime
  -> typed projection / GUI / evidence
```

多模型只在 `model-provider` 内实现：grok-build 负责 catalog/selection/switch/
capability/retry/breaker，OpenCode 只提供 provider-neutral content、endpoint、media
lowering 和多协议 stream 参考。不存在第二 runtime、第二 history、第二 provider owner
或长期兼容层。

## 执行纪律

1. 先迁移真实生产消费者，再删除旧入口；每刀都保持 workspace 可编译。
2. 禁止为“过渡”新增平级 adapter。短期编译映射只能在同一 owner 内，并在该刀退出时删除。
3. 每个条目同时更新 Rust protocol/client、Electron Host/preload、Renderer gateway、catalog、fixture、文档和测试。
4. 未通过 Gate A 的协议/数据契约，不进入 Gate B；没有真实 Electron/read model/ThreadStore 证据，不得标记完成。
5. Codex 已有的非 ChatGPT-only 能力优先直接迁移对应模块、类型和测试语义；只有 Lime credential、桌面宿主或产品范围差异才做薄适配，并在回报中注明 Codex 源路径。

## 本轮进度（2026-07-19）

- `app-server-protocol/protocol/v2` 已建立 Codex-first Thread/Turn/Item wire、核心方法 registry、typed client/server envelope、分页参数和 schema owner；六类 direct v2 lifecycle/delta notification 已接入唯一 `V2NotificationProjector`，v0 lifecycle DTO/schema/fixture 已删除。单一 codegen 当前为 748 个 schema definitions、740 个 TypeScript protocol types、0 生成失败、0 漂移。`thread/start` 缺 model/provider 已 fail closed，并从 durable canonical read model 返回；`thread/resume` 已具备 current cold rejoin/history hydrate 与 actor-ordered active turn snapshot/stale status 归一化骨架，`thread/archive`、`thread/unarchive` 及其 schema 已实现。App Server 现在有 per-thread listener generation：RuntimeEventHub 按 canonical `threadId` demux，start/resume 的 response 与 thread-scoped pending request replay 通过同一 bounded connection writer 排在后续 live event 前；缺失 threadId fail closed，断连移除 subscription。token usage/ThreadGoal、MCP migrated-owner terminal、raw JSONL evidence，以及 fork/delete、其余产品方法和 typed Item/server request 仍是 OPEN，因此不能标记 V1-00/V1-01 整体完成。
- `ThreadStore::append_items` 已提供 canonical Item append 边界，canonical store 拒绝 `item.sequence > outer sequence` 且保持失败原子性；`thread-store` 28/28、App Server canonical 31/31 通过。`ThreadHistoryBuilder` 目前只是 canonical store normalize 骨架，raw Codex RolloutItem、完整 compaction/rollback/fork round-trip 与唯一 reducer 收敛仍是 V1-02 OPEN_REF。
- `runtime-core` 的未知 provider type/name 和 Chat 任务未实现 wire 协议均 fail closed 为 `UnsupportedProtocol`；model route 定向测试 12/12 通过。图片任务继续走专用 lowering。
- AgentControl restart recovery 已把显式 runtime request 与 durable session provider/model defaults 合并，缺 route 时 deferred；restart 定向测试 11/11 通过，App Server startup 对明确缺 selection 的错误只告警。
- image API、server/services/skills 等 provider 消费者已迁入 `model-provider` current owner；`lime-providers` crate、workspace/Cargo.lock 和正向引用已删除，分类为 `dead / deleted / forbidden-to-restore`。真实 route/capability/credential preflight 与 durable route 仍是 provider 主线 OPEN_REF。
- transport 已补齐 WebSocket/Unix socket acceptor、bounded outbound、slow-client disconnect、initialized/ping-pong/close/reconnect；transport tests 17/17，App Server slow-client 单测通过。`optOutNotificationMethods` 仍待 v2 initialize/session state，不能用 transport 层伪造。
- v2 `turn/steer` 已接 processor/runtime 原子入口，steer user message 使用独立 canonical Item identity；App Server steer 7/7。provider_calls 已直接消费 `model-provider::CurrentProviderClient::stream`，provider streaming 7/7；history normalize 9/9。
- 当前单一 codegen 为 748 个 schema definitions、740 个 TypeScript protocol types、0 生成失败、0 漂移；`packages/app-server-client` build 与最新 740 类型状态的完整 `npm run test:contracts` 已通过。此前 Renderer/Electron typecheck、package/Rust 定向测试已通过。Electron Host/Plugin task host 已消费 v2 Thread/Turn identity，不再存在旧字段 typecheck blocker。Agent fixture 的历史/流式/fixture guard 通过；public JSON-RPC 已锁住 `thread/start` v2 envelope，并对旧 `agentSession/start` fail closed。临时 `./--help/json` schema 目录已不存在。

快速骨架曾分为三个互斥车道：v2 lifecycle notification/schema、Electron canonical identity/time、真实 `thread/start` route/ThreadStore owner。三个车道已汇合；后续仍禁止新增 namespace compat、第二套 flat codegen 或 Renderer fallback。

三车道已完成骨架交接并关闭 lifecycle 双轨：v2 direct notification 覆盖 `thread/started`、`turn/started|completed`、`item/started|completed`、`item/agentMessage/delta`；Electron Host/Plugin task host 已消费 v2 Thread/Turn identity；`thread/start` 已删除 `unknown` fallback 并从 durable canonical read model 返回。App Server 只有一个 `V2NotificationProjector`，Rust/TS clients 与 Renderer direct pipeline 已接通；v0 `typedEvent`、`canonicalEvent`、六类 lifecycle DTO/schema/fixture/正向测试已删除。单一 codegen 当前为 748 个 schema definitions、740 个 TypeScript protocol types、0 生成失败、0 漂移，完整 `npm run test:contracts` 已通过。`eventSequenceGate` 只允许 direct lifecycle 与明确 raw side-channel，wrapper lifecycle/action fail closed。

Gate B definitive 记录：共享 Rust 热区的 projector 可见性与 async sink `Send` 编译问题已修复，最终 evidence 使用 source-built App Server sidecar。`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-home-hotpath-v2-definitive-summary.json` 为 `ok=true`、`Gate B CDP controlled fixture`、68/68 assertions；console/page error、legacy command hit、mock fallback hit 均为 0。canonical identity 为 `sessionId=sess_7361160a434846a9841ec0e7bb5bf2fa`、`threadId=thread_f2065f4a31a6470aafad7ff4d3ebc072`、`turnId=turn_8c35081960cd448bbb2a9c024020f6a0`。性能为 pending paint 47ms、submit accepted 247ms、first text paint 344ms、first delta to paint 31ms、client-local output 71ms。client-local 指标已存在；只有 provider/server latency 继续归 App Server diagnostics trace，Renderer 不以 `Date.now()` 等本地时间戳伪造。

diagnostics trace timing 复验已闭环：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-home-hotpath-v2-trace-timing-final-summary.json` 为 `ok=true`、72/72 assertions，source-built sidecar 的 diagnostics trace `traceCount=1/eventCount=8`；provider first-text `providerWaitMs=90`，App Server message delta `serverEventEmittedAt=1784448387417`，均来自 summary-only trace metrics。`appServerIpcHitCount=62`、legacy/mock/page error 均为 0；新增 trace compactor 正/负向测试 2/2 与 fixture guard 63/63 通过。该项关闭 instrumentation blocker；其余 V1-15 能力按下方 OPEN_REF 继续。

Recovery / history Gate B 最新闭环：`.lime/cdp-evidence/agent-session-recovery-cdp-gate-summary.json` 证明真实 Electron、preload/IPC、`app_server_handle_json_lines` 与 `thread/start/read/list/resume` 使用同一 canonical identity；resume 为 metadata-only，不含 legacy fields，也不重发 `thread/started`。`.lime/qc/gui-evidence/agent-session-history-electron-fixture/agent-session-history-electron-fixture-canonical-v2-summary.json` 证明 canonical ThreadStore 的 3 Turns / 9 Items、分页 cursor、DOM 顺序与 archive/unarchive 重启读回。旧 `agentSession/update` 前置、`projected_*` history seed、visual replay helper、queue event projection 与 queued composer restore 均为 `dead / deleted / forbidden-to-restore`。loaded resume 已增加 actor-ordered `activeTurnId` snapshot；本轮 listener 汇合骨架已关闭跨 thread 广播、stale connection subscription 和 response/replay/live producer 分裂。raw JSONL 顺序测试、token usage/ThreadGoal replay 与跨 connection reconnect 仍 OPEN；MCP terminal owner migration 已落 current 终态 owner 骨架，但等待共享 App Server 输入类型迁移汇合后的编译/定向测试证据。

Renderer 旧投影清退已完成：1126 行 `appServerEventPayloadProjection.ts`、零消费者 `canonicalApprovalItemProjection.ts` 及其正向测试已物理删除；`appServerEventStreamProjection.ts` 是唯一 projector，只接受 direct v2 lifecycle 与 provider/runtime/image/media raw side-channel，wrapper lifecycle/action/canonicalEvent/typedEvent fail closed。`agentSession/runtimeEvents/append` 的生产消费者、package/Renderer wrapper、Rust v0 protocol/handler/schema/fixture 和旧正向测试也已删除，typed `artifact/write` 是唯一 Artifact 保存入口；旧 method 只允许出现在负向回流守卫和历史 evidence。`agentSession/action/respond` 的剩余生产 surface 继续迁到 typed reverse server request，不得与已删除 append 绑定成同一 OPEN_REF。provider/server latency 继续由 App Server diagnostics trace 提供，不在 Renderer 建第二套计时事实源。

删除后复验通过历史 31/31、流式 32/32、fixture guard 76/76，以及 `home-hotpath-regression`、`home-hotpath-greeting-regression` 两条真实 Electron Claw 热路径。聚合 fixture 随后在 Coding Workbench 暴露独立 blocker：旧 fixture 使用调用方 session id 调 `agentSession/update`，没有消费 v2 `thread/start` 的 canonical session identity，后续还向 v2 `turn/start` 传旧 `runtimeOptions`。该路径必须随 typed session/route 迁移直接改写，不增加 alias 或 wrapper；它不撤销本轮 projector/Claw 验证，但阻止把完整 `smoke:agent-runtime-current-fixture` 报为全绿。

上述 Coding Workbench source/contract blocker 已收口：fixture 现在消费 `thread/start` 返回的 canonical session/thread identity，并在一次 start 中提交 `model/modelProvider`；后续走 `turn/start` 的 v2 application `additionalContext.metadata`，不再调用 `agentSession/update` 或读取旧 `runtimeOptions.*.metadata`。client contract 反向禁止旧 update，相关静态测试 6/6、完整 `test:contracts` 通过；本轮未重跑 Coding Workbench 独立 Electron Gate B，因此只关闭 source/contract 漂移，不把该产品场景标为 Gate B completed。

`verify:gui-smoke` 的 Renderer、Electron Host 与 source-built App Server 产物均构建成功。首次 shell 运行因 smoke HOME 未隔离而检测到真实用户数据库的活动 `lime.db-wal`，迁移边界正确 fail closed，未执行数据库操作；改用隔离临时 HOME 直接复跑同一 built Electron smoke 后，App Server initialize、Claw shell reload、Memory settings 与结构化 evidence 全部通过。

上述切片均不等于整体 Codex parity。下一刀沿 current `thread/resume` 先完成 raw JSONL `response -> replay -> live` Gate 与 MCP owner 定向验证，再补 canonical token usage owner；ThreadGoal 仍需单独确定 Lime canonical owner，不能伪装成 `ManagedObjective`。随后推进 typed Item/server request、raw rollout、唯一 history reducer 和 compaction lineage；V1-15 再分别补 restart/resume/model switch/approval/MCP/child agent/cold read、live provider 与 Windows 证据。

## 阶段总览

| 阶段 | 目标                                                              | 主要 owner                                                          | 退出条件                                                                   |
| ---- | ----------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| P0   | v2 protocol、ThreadStore/raw rollout、history/compaction/recovery | `app-server-protocol`、`app-server`、`thread-store`、`runtime-core` | 不再有 production `agentSession/*`；cold/live/replay/restart identity 一致 |
| P1   | Item、tool/hooks、sandbox/approval、MCP、Skills/Plugins/Apps      | `tool-runtime`、`agent-runtime`、App Server                         | 每个 item/tool/reverse request 有 typed lifecycle 和恢复证据               |
| P2   | 单一 model catalog/route/capability + grok/OpenCode provider wire | `model-provider`、`runtime-core`                                    | 未知 model/capability/credential fail closed；route/attempt/usage 可追溯   |
| P3   | AgentControl V2、transport、environment/startup、Gate B           | `thread-store`、`agent-runtime`、`app-server-transport`、Electron   | graph/mailbox/transport/restart/真实 Electron 闭环                         |
| P4   | Codex 额外 App Server surface 与 typed client                     | `app-server-protocol`、App Server、`app-server-client`              | 每项实现或明确产品范围排除，current catalog 无静默缺口                     |

## P0：协议与可恢复状态

### V1-00 v2 method registry 迁移

**写集**：
`lime-rs/crates/app-server-protocol/src/protocol/{v0,v2}`、schema/manifest、
App Server dispatch/processor、Electron Host/preload、`src/lib/api` gateway、
command catalog、contract fixture、smoke scripts。

**动作**：

1. 以 Codex v2 `common.rs`、`protocol/v2/**` 和 method registry 为唯一 wire contract。
2. 将 `agentSession/start/read/list/thread/resume/turn/start/turn/cancel` 迁移为
   `thread/start/read/list/resume/fork/...`、`turn/start/interrupt/steer`，同时迁移
   server request、notification、experimental filtering 和 cancellation。
3. 所有生产调用、Host 白名单、Renderer gateway、catalog、fixture 一起切换；不得新增
   `agentSession` wrapper。
4. 删除 Agent 旧 v0 schema/manifest 和 production `agentSession/*` 文案/断言；其余 Lime 产品方法先迁入 v2/current registry，全部迁完后再删除整个 `protocol/v0/**`。

**退出**：`rg` 在 production path 无 `agentSession`；schema round-trip、unknown method、
server request、cancel、pagination、late notification、`npm run test:contracts` 全部通过。

### V1-01 Thread/Turn/Item schema

**写集**：`agent-protocol`、`app-server-protocol/protocol/v2`、read model、projection、
TypeScript generated types。

**动作**：

1. 逐字段对齐 Thread/Turn（ID、fork/source、秒级时间、status、error、path/cwd、name、
   environment、model provider）。
2. 对齐 ordinal、cursor、terminal timestamp、item ID prefix 和 unknown field policy。
3. 只允许 canonical Item 写入；presentation 不得创建 synthetic item。

**退出**：字段 diff、serde/TS round-trip、cold/live/read/replay 同一 identity；rollback 与
active turn snapshot 测试通过。

### V1-02 ThreadStore/raw rollout/history

**写集**：`thread-store`、App Server `ProjectionStore`/EventLog/repair、rollout import、
SQLite/JSONL repository。

**动作**：

1. `ThreadStore::append_items` 只接收已 canonical 的 raw rollout item；metadata 只能走
   `update_thread_metadata` patch，append 不推导 metadata。
2. 让 `ThreadHistoryBuilder` 负责 coalesce、rollback、分页和 active snapshot；EventLog
   保留事件顺序/repair provenance，不再成为第二 transcript owner。
3. Codex RolloutItem/Compacted/rollback marker/fork cut/unknown malformed line 有明确
   retention 和 round-trip；import 只在 source adapter 存在。
4. 删除旧 `session_repository`、第二 transcript DB 和 renderer persistence。

**退出**：crash tail、duplicate sequence、projection failure、repair、cold/live/replay、
Codex rollout import round-trip 全部通过；append 与 metadata patch 事务边界可证明。

### V1-03 Context compaction lineage

**写集**：`runtime-core` context、`agent-runtime` conversation、App Server compaction、
provider history、ThreadStore Item schema。

**动作**：

1. 将 `replacement_history`、`window_number`、`first/previous/window_id` 作为 durable
   compaction lineage；summary 进入 canonical `ContextCompaction`/model-visible item。
2. durable history 永不被 compaction 删除；provider history 只按最新有效窗口重建。
3. resume/fork/rollback、重复 compaction、损坏窗口、无有效 tail 明确 fail-closed 或
   完整历史策略；禁止 summary + 全量旧 history 双发。

**退出**：`compact.rs`/`rollout_reconstruction.rs` 对应测试语义在 Lime replay、restart、
provider history 和 GUI evidence 中一致。

## P1：Item、工具与控制面

### V1-04 Item inventory 与 projection

**写集**：`agent-protocol`、`app-server-protocol` v2 item、ThreadStore、Renderer projection、
schema fixtures。

**动作**：逐项实现 UserMessage、HookPrompt、AgentMessage、Plan、Reasoning、CommandExecution、
FileChange、McpToolCall、DynamicToolCall、CollabAgentToolCall、SubAgentActivity、WebSearch、
ImageView、Sleep、ImageGeneration、Review、ContextCompaction、MemoryCitation 的字段、状态、
started/delta/completed、分页、replay 和 GUI 读取。

**退出**：每个 Item 有字段级 cold/live/replay fixture，terminal 后 late delta 被拒绝。

### V1-05 Tool hooks/exposure/lifecycle

**写集**：`tool-runtime`、native tools、hook runtime、tool inventory、provider stream reducer。

**动作**：

1. 固定 `ToolSnapshot + HookSnapshot -> typed ToolCall -> execute_call -> NormalizedToolOutput`。
2. 加入 Direct/Deferred/Hidden、ToolSearch/LoadableToolSpec、parallel、CodeMode/Direct source、
   argument diff、统一 truncation/outputRef。
3. 实现 SessionStart/UserPromptSubmit/PreToolUse/PermissionRequest/PostToolUse/PreCompact/
   PostCompact/Stop/SubagentStop 的 injection/block/abort/rewrite/permission lifecycle。

**退出**：provider 不再产生 raw tool lifecycle；hook/tool output 在 canonical Item、history、
evidence、GUI 中保持 call identity。

### V1-06 Sandbox/approval/guardian

**写集**：permission/execpolicy/network/guardian、Electron server request、process runtime。

**退出**：`PermissionProfile -> SandboxPolicy -> ApprovalRequest` 单向解析；deny/timeout/cancel、
session approval、network amendment、Windows/macOS/Linux、重启恢复均有 evidence。

### V1-07 MCP/Skills/Plugins/Apps

**写集**：MCP manager/snapshot/elicitation、SkillsService/ watcher、PluginsManager、Apps cache。

**退出**：required/optional + OAuth/dependency + generation replace + elicitation pause/recover；
skill/plugin/app changed/readiness/update notification 可重放；管理面与 sampling 面无交叉事实源。

## P2：单一多模型/provider owner

### V1-08 Catalog/availability/cache

**写集**：`model-provider::canonical`、provider registry、API Key Provider、services registry、
model cache。

**动作**：

1. 删除 `EnhancedModelMetadata`、runtime JSON capability 与 App Server/Renderer 二次推导，保留
   一个 `ModelCatalogEntry`/`ModelCapabilitySnapshot` 和一次 typed conversion。
2. 合并 bundled/configured/remote/cache/ETag，cache key 必须含 provider endpoint、credential
   fingerprint、tenant/account entitlement；credential identity 变化精确 invalidation。
3. provider availability 必须同时检查 enabled、credential/integration、endpoint readiness；
   builtin provider 名称不得直接判 ready。
4. 刷新失败保留旧 catalog；explicit default、small/auxiliary route、release/cost/quota/status
   由 current catalog owner 处理。

**退出**：未知 model、unknown capability、无 credential、非法 allowlist、disabled provider
均 fail closed；catalog refresh/re-auth/tenant 变化不会串身份。

### V1-09 Effective request options/route

**写集**：`model-provider::runtime_provider`、`model_route_resolver`、`runtime-core`、agent runtime。

**动作**：为每次 sampling 固化 `EffectiveTurnOptions`/`ResolvedModelRoute`，覆盖 auth scheme、
headers/body/variant、context window、max output、temperature/top-p、idle timeout、max retries、
stream tool calls、reasoning、backend search、origin/client/deployment/user、compaction/title/
web-search/image-description/prompt-suggestion/subagent auxiliary route。

**退出**：route/attempt/usage/cost/quota/account identity 在 Thread/Turn/read model/evidence 可追溯；
child/restart/replay 保持同一 effective options。

### V1-10 Switch/stream/retry/breaker

**写集**：session switch、provider lowering/current client/stream、transport policy、breaker。

**退出**：active agent compatibility、zero-turn rebuild、watch generation、unknown/partial stream、
429/5xx/timeout retry、breaker open/half-open/close、首个可见 event 后禁止 fallback 重放均通过。

### V1-11 删除 lime-providers

**动作顺序**：

1. 将 `server`、`services`、`skills`、`image_api` 等消费者改用 `model-provider` current client、
   lowering、stream 和统一 credential owner；删除独立 converter/session/signature store 调用。
2. 从 workspace/Cargo 依赖、catalog、文档、测试 fixture 删除 `lime-providers`。
3. 加 crate/import 扫描，禁止新引用。

**退出**：workspace 无 `crates/providers` 成员/依赖；`cargo check --workspace` 与 provider 定向测试通过。

## P3：Multi-Agent、transport、environment、Gate B

### V1-12 AgentControl V2

补 role/config precedence、max depth/width、residency、rollout budget、fork all/last-N/none、
trigger_turn/queue-only、wait priority、child tool subset 和 recovery fuzz；graph/identity/mailbox
只能由 `thread-store` 持久化。

### V1-13 App Server transport

实现或删除 `AppServerTransport` ws/unix/off：stdio、WebSocket、Unix control socket、bounded
ingress/outbound、slow-client disconnect、per-connection initialized/experimental/opt-out
notification、request cancellation 与 reconnect 均按 Codex transport tests 验收。不能以 enum/URL
解析代替 acceptor。

### V1-14 Environment/instructions/startup

补 AGENTS.md discovery/cache/precedence、child environment inheritance、config lock、session
startup prewarm、sticky environment/root、turn timing/usage/OTEL identity；restart/replay 不改变
instruction source 或 effective route。

### V1-15 真实 Electron Gate B

证明 `Renderer -> preload/IPC -> app_server_handle_json_lines -> App Server v2 -> RuntimeCore ->
provider/tool -> ThreadStore -> GUI`，覆盖 restart/resume/compaction/model switch/approval/MCP/child
agent/cold read。mock/localhost provider 只能做 fixture，不得冒充 production proof。

current home-hotpath fixture 已由上述 definitive evidence 68/68 通过；该结果关闭本轮 Electron
主链与 instrumentation blocker，不等于 V1-15 全部完成。restart/resume/compaction/model switch、
approval/MCP/child agent/cold read、live provider 与 Windows cross-platform UDS 仍需分别补证据。

### V1-15 detached 首轮响应式策略与 Soul 复验（2026-07-22）

目标：修复真实首页 detached 首轮仍按退役 session identity 判断响应式策略的问题，并确认
`memory.soul` 的贱兮兮 Style Pack 与 `compact_tools` 可以在同一 current runtime 请求中生效。

已完成：

- `apply_app_server_turn_policy` 的响应式入口只接受 current `app_id=agent-chat` 与
  `business_object_ref.kind=agent.thread`；退役 `desktop + agent.session` 不再启用响应式 profile，
  不新增 alias、compat 或双读。
- current ingress、follow-up、required search、plugin activation、structured mention 与 retired
  desktop 负向守卫已覆盖。相关 App Server 模块测试 50/50 通过，其中 Soul prompt 2/2、
  `session_prompt_context` 15/15、`model_selection` 33/33。
- 用户配置已从旧 `sassy_cute_executor` 一次性保存为 current
  `cheeky_sassy_executor`；resolver 未增加旧 id alias。

真实 Electron / CDP 证据：

- source-built App Server mtime 晚于策略源码，隔离 Electron 连接 `127.0.0.1:1420/?nativeStartup=1`，
  `window.__LIME_ELECTRON__=true`、preload invoke 存在，`electron-ipc` 与 current `turn/start`
  均命中；detached 首页输入“你好”后 GUI terminal，console/page error 均为 0。
- 本地 OpenAI-compatible provider capture 只保存结构化摘要：最终请求的 Soul 五个关键 marker
  全部为 true，system prompt 长度 5387；工具面为 16 个工具，全部属于 `compact_tools`
  allowlist，无 full-surface 工具泄漏。GUI 可见回复为“哟，终于想起找我啦？说吧，今天想搞点什么事？”。
- `soul-style` 仓库 fixture 另行证明四类完整 prompt surface marker 与 GUI/read model 输出，但该
  scenario 主动绑定默认 workspace，并将 usage 固定为 `prompt_tokens=31_000`；这个数字不能用于
  判断 detached `compact_tools` 是否生效。该 fixture 当前 v2 prompt-ingress 解析仍按旧 flat
  `params.input.text/sessionId` 读取，导致产品回合完成后 `fixturePromptReachedBackend` 断言失败；
  归类为共享 fixture evidence parser 漂移，不作为本窄写集产品失败，也不在并行热区夹写。

验证：

- `npm run smoke:agent-runtime-current-fixture`：通过；历史 31/31、流式 32/32、fixture guard
  84/84 及完整 Electron 场景聚合通过，`liveProviderUsed=false`。
- `npm run verify:gui-smoke`：通过；Renderer、Electron Host/preload 与 source-built App Server
  sidecar 构建成功，真实 Electron shell 初始化、Claw reload、Memory settings 与结构化 evidence 均通过。
- `npm run governance:legacy-report`：通过；零引用候选 0、分类漂移 0、边界违规 0。
- `npm run test:rust:related -- <本轮 4 个 App Server 文件>` 扩至 App Server 全量后为
  1456 passed / 6 failed；6 项均位于本轮避让的并行重构热区（ThreadGoal、event-log repair、
  compaction 与 credential UTF-8），不与本写集重叠。
- 当前分类：`current = agent-chat + agent.thread`；`dead = 响应式策略中的 desktop +
  agent.session`；无新增 `compat` / `deprecated`。

### V1-15 首页首发 Workspace 闪烁修复（2026-07-22）

目标：修复首页输入后出现 `首页 -> Claw -> 首页 -> Claw` 的前后台场景竞态，保持
首页首发创建 session 后的 Workspace 前景态连续。

根因与修复：

- 真实 Electron CDP 在修复前记录到 `375ms` 进入 Claw、`576ms` 回到首页、`837ms`
  再次进入 Claw，首页回闪约 `261ms`。`FLICKER-DEBUG2` 证明新 session 已存在、消息和
  `isSending` 均为真，但 `isHomeSessionBackgroundRecovery` 把本次首页首发误判成旧会话后台恢复，
  继而清空 `hasHomeConversationActivity` 并重新选择首页空态。
- `resolveTaskCenterHomeChromeState` 现在先计算 `isTaskCenterDraftSendPending`，存在本次首页
  draft send request 时不再把当前 session 视为后台恢复。判定使用 request identity，不用模糊的
  全局发送态替代；旧会话后台输出的隐藏语义保持不变。
- 修复后同一路径在 `191ms` 进入 Claw、`1254ms` 由 session header 接管，观察窗口内
  `returnedHomeAfterChat=false`，发送后无 `FLICKER-DEBUG2`，console error 为 0；临时
  `FLICKER-DEBUG*` 已从生产源码删除。

验证：

- `taskCenterSurfaceState.unit.test.ts`、`workspaceShellChromeRuntime.unit.test.ts`、
  `chatLayoutVisibility.test.ts` 合计 38/38 通过；新增回归覆盖“首页首发创建 session 后仍带
  background recovery 标记”的精确状态组合。
- `npm run typecheck`、窄写集 `git diff --check` 通过；
  `npm run smoke:agent-runtime-current-fixture` 的 history 31/31、streaming 32/32、fixture guard
  84/84、首页两条 hotpath 与 Electron 聚合场景通过，`liveProviderUsed=false`。
- 本次最新 `npm run verify:gui-smoke` 的版本、Renderer、Electron main/preload 与 source-built
  App Server sidecar 构建均通过；Cargo artifact 锁释放后续跑同一 built shell smoke，App Server
  1.109.0 初始化、Claw shell 首次加载/reload、Memory settings 与结构化 evidence 全部通过，
  `result=pass`。

### V1-04/V1-09/V1-10 Reasoning 可见性与终态偏差取证（2026-07-22）

结论：当前“思考不断重复”不是 Renderer 创建多个 reasoning 节点，也不是累计全文被重复拼接。
原始 `Thinking Process` 文本来自当前模型/provider；Lime 将 raw reasoning 默认展示、合并
summary/raw wire 事件，并在 provider 缺少 terminal 时长期保持 `inProgress`，共同放大成用户可见
的重复与卡住体验。该项在取证时分类为 `current / alignment-open`；summary/raw typed 投影、默认
可见性和 EOF-without-completed fail-closed 已由 V1-20 关闭。

证据：

- 真实 Electron 当前 DOM 只有 1 个 `thinking-block`；App Server `thread/items/list` 只有 1 个
  reasoning item，内容由 `"Thinking"`、`" Process"`、`":"` 等真实增量 token 组成，没有多份
  reasoning item，也没有累计全文重复 append。本轮未捕获 `duplicate_event_id`。
- `model-provider/src/current_client/stream.rs` 当前把
  `response.reasoning_text.delta` 与 `response.reasoning_summary_text.delta` 都降为同一个
  `LlmEvent::ReasoningDelta`；`agent-runtime` 随后只保留单一 `text`，App Server/Renderer 无法再
  判断这是 summary 还是 raw content。
- 对齐基线为 Codex `9fc715c086`：`codex-api/src/sse/responses.rs` 将两类 wire event 分别投影为
  `ReasoningSummaryDelta` 与 `ReasoningContentDelta`；`app-server-protocol/.../thread_history.rs`
  将 summary 和 raw content 分别写入 `ThreadItem::Reasoning.summary/content`；
  `show_raw_agent_reasoning` 默认 `false`，TUI history 只有显式
  `RawReasoningVisibility::Visible` 才显示 raw content。
- 同一模型回合只有 reasoning、没有 assistant message 或 terminal，Turn 长时间停留在
  `inProgress`。Codex Responses SSE 在流关闭但未收到 `response.completed` 时返回
  `stream closed before response.completed`，idle timeout 返回显式 stream error；Lime 当前没有
  等价的 fail-closed 产品终态。

取证后的收口要求必须落在 current owner，不在 Renderer 做字符串过滤：

1. `model-provider` 保留 summary/raw 两类 typed event，`agent-runtime` 与 canonical
   Thread/Turn/Item 分别持久化 `summary/content`，不再压成单一 `text`。
2. GUI 默认只展示 reasoning summary；raw content 默认隐藏，仅由显式诊断/开发开关启用，
   语义对齐 Codex `show_raw_agent_reasoning=false`。
3. provider stream 对 EOF-without-completed、idle timeout 和 transport error 产出唯一 terminal
   error，App Server read model 与 GUI 同步结束 `inProgress`；补 SSE/WS、cold/live/replay 和
   真实 Electron fixture，禁止测试侧合成 terminal。

### V1-04/V1-15 短问候扩题与持续滴流收口（2026-07-23）

根因与改动：

- 异常会话 canonical event 证明用户输入只有“你好”，旧项目长文由同一 Provider turn 直接流入，
  不是 Renderer 串接其他 Turn。22 个新会话中多次精确复现
  “资料回来了，工具这次没掉链子，挺争气。先看能落地的结论。”；该句与默认
  `cheeky_sassy_executor` 的 `after_tool_success` few-shot 原文完全一致。
- `runtime/soul/prompt_context.rs` 继续保留 response/surface/anti-repetition 风格契约，但不再把
  style pack 的示例原文渲染进运行时 system prompt。新增 turn relevance 约束：简单问候只短答，
  不推断未声明任务、不继续无关工作、不从 style/persona 上下文总结项目。
- Codex 对照确认 SSE idle timeout 会被任意持续 frame 刷新，不能拦截 1-3 秒一字的无限滴流。
  `agent-runtime/provider_turn.rs` 在既有 first-visible-output deadline 外新增每个 provider sampling
  step 的不可刷新绝对 deadline，默认 `300s`，测试/受控 harness 可用
  `provider_step_timeout_ms` / `providerStepTimeoutMs` 覆盖。deadline 命中后结束活动输出 item、
  发送 execution failure trace 并返回唯一 terminal error；不在 GUI 合成完成态。
- Chat Completions 请求没有历史消息、workspace AGENTS、`previous_response_id` 或 conversation id；
  旧项目正文剩余来源分类为 Agnes Provider 非预期生成/服务端状态污染。Lime 不增加 provider 专用
  endpoint、identity 或 header 特判。

验证：

- `cargo test -p agent-runtime provider_turn::tests -- --nocapture`：27/27 通过；新增回归证明持续
  reasoning heartbeat 与可见文字滴流都不能刷新绝对 deadline，且活动 reasoning/text item
  分别正常结束为 `ReasoningEnd` / `TextEnd(FinalAnswer)`。
- `cargo test -p app-server runtime_soul_prompt_excludes_few_shot_example_text -- --nocapture`：通过；
  运行时 prompt 保留风格契约与短问候约束，但不包含两条内置示例原文。
- `npm run smoke:agent-runtime-current-fixture`：通过；history 31/31、streaming 32/32、fixture guard
  86/86，source-built App Server sidecar 与首页普通任务/短问候、停止后继续、审批、图片、Skills、
  MCP、Workbench 等 Electron Gate B 聚合场景通过，`liveProviderUsed=false`。
- `cargo build -p app-server --bin app-server` 与 `npm run bridge:health -- --timeout-ms 120000`：通过；
  既有 Electron dev watcher 自动加载新二进制，current Bridge `/health` 返回 `status=ok`。
- 仓库排除依赖、target 与 `.git` 后搜索 `code.ylsagi.com`：零命中；未新增 Agnes endpoint、
  Provider identity 或域名硬编码。
- 未调用正式 Agnes Provider；真实 Provider 输出质量与服务端会话隔离仍需用户显式授权后复验。

## P4：额外 Codex App Server surface

逐项实现或明确产品范围排除并从 current catalog/完成定义删除：`backgroundTerminals`、
`process/*`、`command/exec/*`、`fs/watch`、fuzzy search、realtime audio/text/SDP、review、file
checkpoint、dynamic tools、attestation、environment/current time/memory mode。实现项必须进入
App Server v2 current owner，不能通过旧 WebSocket/HTTP agent protocol 旁路。

## loaded-thread listener owner 收敛（2026-07-20）

目标：让 resume barrier 和 per-thread live event 顺序有唯一 current actor owner，避免继续把 listener 业务堆进 App Server 根文件。

已完成：

- 新增 `lime-rs/crates/app-server/src/thread_listener.rs`，承接 external runtime event、listener generation、resume barrier、subscribe/replay/deferred-live 顺序和唯一 v2 projector。
- `lib.rs` 仅保留 transport writer、connection 生命周期和 request 编排；`thread_state.rs` 继续承接状态与双向 connection index。
- 重复 resume barrier 在准备阶段 fail closed，并补回归；不新增 compat、fallback 或旧 runtime 入口。
- raw JSONL 复跑修正了 fixture 缺 canonical store 和 live connection 重复订阅误判：start/resume 现在允许同一 connection 幂等订阅，双向索引仍由 HashSet 去重。

验证与未完成：

- 当前源码 `cargo check -p app-server --lib` 通过；listener 3/3、thread_state 5/5、server_request 16/16、MCP elicitation 10/10 通过。
- scoped rustfmt、窄写集 `git diff --check` 与 `governance:legacy-report` 通过；全 package rustfmt 仅被并行 `runtime/objectives.rs` import 排序漂移阻塞。
- 当前 MCP reverse request 的 exact owner + reconnect claim 与 Codex thread-scoped fan-out + first terminal 不同，分类为 `current / alignment-open`，不能作为 parity 证据；connection writer 刀必须同时收敛该语义。
- 下一刀：runtime instance/generation owner、connection writer sequencer、unsubscribe/idle unload；随后才接 canonical token usage/ThreadGoal replay。Codex resume 的 path/history/override 进入 typed `ResumeThreadOptions`，不在 transport 层继续堆字段。

### V1-16 Rollout 追加热路径降复杂度（2026-07-23）

目标：消除新回合首发时 App Server 因反复扫描历史 rollout 占满 CPU 的本地延迟，不通过“你好”特判、静默切换模型或放松 canonical identity 来掩盖 Provider TTFT。

根因与改动：

- 真实运行 App Server 持续约 `97-98%` CPU；采样落在
  `drive_backend_to_completion -> append_runtime_events -> apply_history_sync ->
  RolloutStore::append_history -> scan_rollout -> history_content_digest`。
  单个活跃 rollout 约 `9MB / 1311` 行，每个流式事件都从头解析全部 JSONL 并重新计算全部
  history SHA-256，形成 O(n²) 写入路径。
- 对齐 Codex `rollout::recorder` 的单 writer / 追加语义，当前最小修复保留 Lime 现有同步
  durability owner：`append_history` 仅验证首行 `SessionMeta` identity，然后以 64KB 分块从文件尾部
  读取到最近一条 history。它验证该记录和尾部 metadata 的 schema、identity、digest、fingerprint，
  再执行 latest 的幂等、collision 与 stale 判定；不再重算早期 history digest。
- `scan_rollout` 未删除，仍是冷启动、snapshot、archive、repair 与完整 tamper detection 的唯一全量
  校验路径。临时 `/tmp` 性能探针已从 `runtime_backend.rs` 与 `runtime/event_store.rs` 删除。

验证状态：

- `cargo test -p app-server canonical_rollout::tests`：5/5 通过，覆盖早期记录不进入追加热路径、latest
  幂等、collision/stale、尾部 metadata 和最新 history 损坏 fail-closed。
- `cargo test -p app-server runtime::canonical_thread_store_tests`：35/36 通过；唯一失败为
  `empty_projection_rebuilds_active_and_archived_threads_from_rollouts`。失败断言要求 projected user summary
  不含 `input`，但并行中的 `agent-protocol::ThreadItemPayload::UserMessage` 多模态改造已经由
  `projection_rebuild.rs` 正式写入 `input`；本刀未触碰该数据面，失败不归因于 rollout 追加优化。
- 真实 Electron Gate B 已复测：冷重启后首轮 `turn/start` 仍在 `electron-ipc`、`status=success`，约
  `266ms` 进入 Claw，Provider 首字约 `12.072s`；第二轮约 `113ms` 进入 Claw、首字约 `10.997s`。
  两轮均无首页回闪、`thinking-block=0`、invoke error 为 0、attempt=1；完成后的 App Server idle CPU
  为 `0.0%`。这证明本地 rollout 热路径已不再制造持续高 CPU，但 Provider TTFT 仍是独立上游性能问题，
  不能用“你好”特判或静默换模型掩盖。
- `npm run smoke:agent-runtime-current-fixture`：通过，history `31/31`、streaming `32/32`、fixture guard
  `86/86`，包含首页首发普通任务与短问候真实 Electron fixture，`liveProviderUsed=false`。
- `npm run verify:gui-smoke`：通过，真实 Electron Host/preload、App Server `1.109.0`、Claw shell reload、
  Memory settings 与结构化 evidence 均通过。

### V1-17 相同 Prompt 的 canonical Turn hydration 收口（2026-07-23）

目标：修复同一会话连续发送相同文本时，terminal recovery 已拿到第二轮 canonical 完成态，
但 Renderer 把第二轮 user/assistant 错误合并进第一轮、导致第二轮继续显示“生成中”的问题。
该刀只收敛 Thread/Turn/Item identity，不修改 Provider 输出、prompt 或终态协议。

根因与改动：

- `agentChatHistoryLocalMergeMatching.ts` 旧逻辑先按 user 文本、assistant 内容签名匹配，最后才检查
  `runtimeTurnId`。两轮 prompt 都是“你好”时，第二轮 canonical user 会命中第一轮 local user，
  后续顺序 assistant 也命中第一轮 local assistant，造成第二轮正文覆盖第一轮、真正第二轮
  `pending-turn:*` 壳未收口。
- 匹配现在以 canonical `runtimeTurnId` 为第一身份：目标带 canonical turn 时先查 exact turn；文本、
  签名、顺序和 interrupted fallback 可以承接无 turn / `pending-turn:*` 本地壳，但不得跨到另一个
  canonical turn。该语义对齐 Codex 的 Thread/Turn/Item identity，不增加文本兼容旁路。
- 增量 hydration 只覆盖新 turn 时，保留未被 hydrated turn set 覆盖的既有 local canonical 历史；
  第二轮 canonical user/assistant 替换第二轮 pending 壳，第一轮 canonical user/assistant 原样保留。
- 新增回归精确覆盖：本地第一轮 completed、第二轮 pending、两轮 prompt 都是“你好”、hydrated
  仅含第二轮 canonical user/assistant；断言两个 turn 各自保留且第二轮不再残留 generating 状态。

验证：

- 红灯证据：修复前新增回归只返回 2 条消息，第二轮 canonical 结果占用第一轮 local user/assistant
  ID；修复后 local merge、local tail、missing users、runtime sync 共 `48/48` 通过。
- `npm run typecheck`、Prettier、窄写集 `git diff --check` 通过。
  `npm run test:related -- <merge paths>` 被现有 Vitest/Vite 环境以
  `EISDIR: read .../lime/electron` 阻塞；相同受影响测试已通过显式 Vitest 定向入口，未把该基础设施
  错误误报为实现失败。
- `npm run smoke:agent-runtime-current-fixture` 通过：history `31/31`、streaming `32/32`、
  fixture guard `86/86`，以及首页普通首发/短问候、停止后继续、审批、图片、Skills、MCP、
  Workbench 等 Electron 聚合场景全部通过，`liveProviderUsed=false`。
- `npm run smoke:claw-chat-current-fixture` 与 `npm run verify:gui-smoke` 通过；后者证明真实 Electron
  Host/preload、source-built App Server `1.110.0`、Claw shell reload、Memory settings 与结构化
  evidence 均为 `result=pass`。
- 真实 Electron CDP `9231` Gate B 复测使用同一稳定 Host/App Server/Renderer PID，页面确认
  `window.__LIME_ELECTRON__=true`、preload invoke 可用。首页首轮 probe 仅发生 `home=true -> false`，
  第二轮全程 `home=false`；没有再次回首页。
- 当前 read model：session `sess_8079a30ba04f4b48b5bfa783625b69e9`、thread
  `thread_e9ae39de8e624895ba8af2c4c9c2e887` 有两个独立 turn：
  `turn_59793ef8aa4043ae9d83c3e558e54d63` 与
  `turn_15cf1b69dc1d4e79a68c1f2e02bb8ade`，均为 `completed`、各含 1 个 user item 与 1 个
  assistant item、reasoning item 为 0。GUI 同时保留两个独立消息组，第一轮正文未被改写，第二轮
  无“生成中/正在生成回复”占位，`thinkingBlocks=0`、invoke error 为 0；开发日志出现
  `AgentStream.terminalRecoveryPoll.recovered` 后 GUI/read model 一致收口。
- 首轮 live Provider 从发送到 GUI 完成约 `8.5s`；本刀消除了错误归并与持续生成，但没有把
  Provider TTFT 伪装成本地完成。上游首事件等待仍按 V1-16 的独立性能项继续测量。

### V1-18 侧边栏 Thread 查询与首页首发 Gate B 复测（2026-07-23）

目标：收口首页发送后的真实 Electron 复测证据，并消除侧边栏 Thread 查询在多项目场景中的重复
workspace 扫描；本刀不引入 UI 特判，也不把 Provider 延迟伪装成本地完成。

根因与改动：

- App Server 的 `thread/list` / `thread/read` 在做 agent context enrichment 时，为每个 Thread
  重新打开 SQLite 连接并重复 attach / schema 初始化，形成 N+1。新增复用当前 SQLite 连接的
  `read_thread_spawn_parent_with_conn`，让同一请求内的 enrichment 复用现有连接。
- `useAppSidebarSessions` 原先在已有全局查询和 cwd 查询之外，又按 `workspaceId` 做一次全局扫描；
  该扫描不提供新的分页语义，已删除，保留全局查询与每个 cwd 查询，项目归组断言同步更新。
- 相关修改集中在 `lime-rs/crates/app-server/src/runtime/canonical_thread_store.rs`、
  `lime-rs/crates/app-server/src/runtime/canonical_thread_store/agent_graph.rs`、
  `src/components/app-sidebar/useAppSidebarSessions.ts` 与对应组件测试。

性能证据：

- 真实 Electron 直连 `thread/list`：修复前 limit 11 约 `12ms`、limit 21 约 `21-22ms`；修复后
  limit 11 约 `4ms`、limit 21 约 `5-6ms`。
- 真实 Electron CDP `http://127.0.0.1:9231` 确认 `window.__LIME_ELECTRON__ === true`、
  `window.electronAPI.invoke` 可用。两次从首页发送“你好”均只发生一次 `home=true -> false`，
  之后保持 Claw 工作台，没有 `FLICKER-DEBUG` / `FLICKER-DEBUG2`、`duplicate_event_id` 或
  本轮新增控制台 error；`thinking-block=0`，最终只保留一份 assistant 正文。
- 第一次 live 复测：`submitAccepted=90ms`，Provider 首字 trace `providerWaitMs=862ms`，
  Renderer `firstTextDelta=2863ms`，`firstTextPaint=2924ms`；第二次从首页复测：
  `submitAccepted=111ms`，Provider 首字 `providerWaitMs=635ms`，Renderer
  `firstTextDelta=1164ms`，`firstTextPaint=1197ms`。首字阶段仍受 live Provider TTFT 影响，
  但 UI 额外渲染延迟约 `33ms`，未发现重复思考或前端重复输出。

Gate B 证据边界：

- 本轮 trace buffer 命中 `transport=electron-ipc`、`command=app_server_handle_json_lines`、
  `status=success`，现行 JSON-RPC 方法为 `thread/start`、`turn/start`、`thread/read`、
  `thread/turns/list`、`thread/items/list` 和 `agentSession/update`；没有把旧的
  `agentSession/turn/start` 名称当作 current 证据。
- 该证据证明 Electron Desktop Host、preload/IPC、App Server JSON-RPC、read model 与 GUI
  的同轮闭环；不证明其他 Provider、其他平台或所有 live 网络条件下的普遍 TTFT，也不替代
  runtime backend/provider fixture 对最终 system prompt 的证据。

验证：

- AppSidebar 组件测试 `51/51`；App Server related Rust `1485/1485`；`npm run typecheck`；
  `cargo build --manifest-path "lime-rs/Cargo.toml" -p app-server --bin app-server`。
- `npm run smoke:agent-runtime-current-fixture`、`npm run smoke:claw-chat-current-fixture`、
  `npm run test:contracts`（296 checks）、`npm run verify:gui-smoke`、
  `npm run bridge:health -- --timeout-ms 120000` 与窄写集 `git diff --check` 均通过。

### V1-19 EventLog 追加与 Projection schema 热路径收口（2026-07-23）

目标：消除连续 runtime event 追加时重复扫描完整 JSONL，以及每次打开 ProjectionStore 都重复执行
schema/index/column migration 的本地启动阻塞；保持 durable-before-notify 和冷启动全量校验不变。

根因与改动：

- `EventLogWriter` 为每个 session 缓存文件长度、mtime、最后 sequence 和 event id 集合；文件被外部
  修改时自动回退到既有全量 append guard，追加成功后只更新缓存，不重复扫描早期事件。
- `ProjectionStore::open_thread_store()` 只负责打开连接和必要的 attach；完整 schema/index/column
  migration 收敛到显式初始化路径，避免每次 `thread/read`、`thread/list` 连接都重复执行。
- 归档/删除 event log 时同步清除 append state，避免旧文件状态污染新 session；冷启动、repair 和
  tamper detection 仍使用全量扫描 owner。

验证：

- `cargo test -p app-server runtime::event_log`：19/19；
  `cargo test -p app-server canonical_thread_store`：77/77；`cargo check -p app-server --lib` 通过。
- `npm run bridge:health -- --timeout-ms 120000` 通过；真实 Electron CDP Gate B 连续 3 轮首页输入
  “你好”均只发生一次 `home -> Claw`，未回首页，`assistantCount=1`、`thinkingBlocks=0`。
  首页提交到 `turn/start` 约 `155-313ms`，观测到的 live Provider 首字等待约 `568ms`；剩余首字
  延迟属于 Provider/运行时事件链，不再归因于 rollout/schema 重复扫描。
- 临时采样报告和 CDP 探针已移入系统废纸篓，生产源码无 `FLICKER-DEBUG*` 或临时探针残留。

### V1-20 Reasoning 可见性与 Skills 初始化热路径收口（2026-07-23）

目标：完成 V1-04 reasoning summary/raw 的 Codex typed 对齐，并删除首轮初始化中与
`AgentSkillSnapshot` 重复的全量 Skill registry reload；不在 Renderer 做内容字符串过滤，也不以
短问候特判掩盖运行时延迟。

根因与改动：

- `model-provider -> runtime-core -> agent-runtime -> agent protocol -> App Server -> Renderer` 现在分别
  保留 `ReasoningSummaryDelta` 与 `ReasoningContentDelta`。summary 和 raw content 共享 canonical
  reasoning item identity，但只有 raw content 进入下一次 provider history；Responses stream 在未收到
  completed 时 EOF 会 fail closed。
- GUI 的活动流和恢复流默认 `surfaceThinkingDeltas=false`，summary 作为可见过程分段展示，raw content
  默认隐藏。`thinking` 仅表达本轮推理能力偏好，不再隐式解锁 raw chain-of-thought；Skill 明确保留的
  process timeline 继续按既有 typed metadata 展示。
- 首段正文到达时，只有实际收到过隐藏 raw reasoning 才执行对应清理；纯正文不会产生额外消息刷新，
  summary 也不会被误清。完成态继续由 canonical item snapshot 单次接管，不生成第二份 assistant 正文。
- `AgentRuntimeState::initialize()` 删除 `reload_skills()`。普通回合的 current owner 是
  `AgentSkillSnapshot`：只扫描 metadata/locator、按 root 与 `SKILL.md` mtime 自动失效缓存；明确选择后
  才读取 Skill 正文。workspace runtime enable 仍按精确目录注册，安装/删除后的显式 reload API 保留。
- 临时 `FLICKER-DEBUG*`、runtime/provider preflight probe 和 `/tmp/lime-runtime-preflight-probe.jsonl`
  均已删除；执行计划中的旧日志名称只作为历史 evidence 保留。

验证：

- reasoning 前端相关 11 个测试文件 `189/189` 通过，覆盖活动流、恢复流、typed event projection、
  summary 分段、raw 隐藏、消息持久化与时间线；`npm run typecheck` 通过。
- `model-provider current_client::stream_tests` `9/9`、`agent-runtime provider_turn::tests` `31/31`
  通过，覆盖 summary/raw 分型、provider history、EOF fail-closed、可见输出 deadline 与 token usage。
- `lime-skills agent_snapshot` `6/6`、`skill_loader` `5/5`、`lime-agent runtime_state_support` `5/5`、
  App Server 主回合初始化 `1/1` 通过，证明 metadata-only snapshot、缓存自动失效、按需 registry 与
  不带全量 reload 的 runtime 初始化均可用。
- 最新真实 Electron CDP warm 回归：`home -> Claw=1054ms`、`submitAccepted=64ms`、
  `firstEvent=199ms`、`firstTextDelta=1261ms`、`firstTextPaint=1318ms`、点击到正文 DOM `2406ms`、
  点击到终态 `2439ms`；`assistantCount=1`、`thinkingBlocks=0`、invoke error 为 0，页面只发生一次
  `home -> Claw`。第一次冷启动采样把“正在准备回复”误识别为正文，因此不作为精确 TTFT 证据。

分类：`current = typed reasoning summary/raw + AgentSkillSnapshot + 真实 Electron 主链`；
`dead / deleted = 深度思考开关解锁 raw 显示、首轮全量 Skills reload、临时性能/闪烁 probe`；本刀未新增
`compat` 或 `deprecated`。

### V1-21 Reasoning lifecycle 重复分段收口（2026-07-24）

目标：修复同一 Codex reasoning item 在 canonical lifecycle snapshot 前后被重复投影的问题；继续保持
summary 可见、raw reasoning 默认隐藏，不新增 Renderer 内容特判或第二套事件 owner。

根因与改动：

- `item_started` / `item_updated` / `item_completed` 在 runtime handler 与 lifecycle handler 两层都会清空
  `streamedReasoningSourceItemId`、`streamedReasoningSummaryIndex` 和当前 segment。于是同一
  `itemId + summaryIndex` 的下一条 delta 被误判为新段，产生重复 thinking part。
- canonical item lifecycle 现在只负责 ThreadItem 与消息快照接管，不再拥有 streamed summary reset。
  segment 只在新 `itemId + summaryIndex`、`reasoning_summary_part_added`、显式过程边界或 turn 终态切换，
  与 Codex 的 item identity / summary index 语义一致。
- 回归覆盖 `summary delta -> 同 item_updated -> 同 summary delta -> item_completed`：增量阶段只有一个
  thinking part，完成态由 canonical reasoning snapshot 单次接管。

验证与清理：

- reasoning/runtime 定向 `28/28`，canonical reader/v2 notification/turn binding/resume binding
  `67/67`；`npm run typecheck`、Prettier 与窄写集 `git diff --check` 通过。
- `npm run test:contracts` 通过（759 generated types、296 client checks）；
  `npm run governance:legacy-report` 为 0 零引用候选、0 分类漂移、0 边界违规。
- `npm run verify:gui-smoke` 与修改后 `npm run smoke:agent-runtime-current-fixture` 通过；后者覆盖首页首发、
  短问候、停止续写、审批、Skills/MCP、历史恢复和工作台 Electron 场景。
- 真实 Electron CDP 确认 Electron/preload 主链；本轮首页短问候 `firstEvent=158ms`、
  `providerWaitMs=550ms`、`firstTextDelta=1286ms`、`firstTextPaint=1338ms`。该样本未出现 14 秒级
  Renderer 阻塞；历史 14.5 秒首事件等待继续归 Provider/模型链路诊断。
- 生产源码无 `FLICKER-DEBUG*`、视频拆帧、临时 Swift/CDP probe 或截图残留；执行计划中的关键词仅为
  历史 evidence。`current = canonical reasoning lifecycle + streamed summary identity`；被删除的 lifecycle
  双 reset 为 `dead / deleted / forbidden-to-restore`，无新增 `compat/deprecated`。

### V1-22 Plan typed delta 生命周期收口（2026-07-24）

目标：按 Codex v2 对齐 Plan 的 public typed delta，删除此前把每个内部 `plan.delta` 错误投影成
`item/started` 的旧 wire 语义，并让 Renderer 复用同一 canonical Plan item identity。

写集与唯一 owner：

- `app-server-protocol` 的 `PlanDeltaNotification`、v2 method/schema/codegen 与 serde 回归；
- App Server `V2NotificationProjector` 与同 owner 的 `plan.rs`，以及 `runtime_backend/plan_events.rs`；
- `packages/app-server-client`、Renderer v2 notification route、Plan projection/controller 与定向测试；
- `internal/refactor/v1` inventory/fixture 与 public JSON-RPC integration test。

当前链路：内部 event log 仍使用 `plan.delta` 作为 App Server canonical event/projection 表达；生产
public wire 只输出一次 `item/started`，每个增量输出 `item/plan/delta(threadId, turnId, itemId, delta)`，
最后输出一次 `item/completed`。Plan delta 保留原始换行，不复用会 `trim()` 的通用 payload reader；
terminal 后的 late delta fail closed。

验证：

- `cargo test -p app-server-protocol plan_delta_notification_round_trips_codex_shape`：通过；
- `cargo test -p app-server v2_notifications::tests::maps_plan` 与
  `v2_notifications::tests::rejects_plan_delta`：通过；
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server --test thread_v2_jsonrpc plan_delta_uses_one_typed_item_lifecycle_in_public_jsonrpc_messages`：通过；
- Renderer Plan projection、`packages/app-server-client`、`packages/agent-runtime-client` 定向测试：分别
  `35/35`、`78/78`、`23/23` 通过。
- `npm run typecheck`、`npm run test:contracts`（760 generated types、296 client checks）与
  `npm run governance:legacy-report`（0 零引用候选、0 分类漂移、0 边界违规）通过；
  `npm run smoke:agent-runtime-current-fixture` 与 `npm run verify:gui-smoke` 均通过。后两者证明真实
  Electron Host/preload、App Server、Plan history hydrate 与 GUI 的 current 链路可用，fixture 未使用
  live Provider。

分类：`current = item/plan/delta + 单一 Plan item lifecycle + canonical itemId`；旧的每 delta
`item/started` 投影与重复生命周期属于 `dead / deleted / forbidden-to-restore`；未新增
`compat/deprecated`。生产源码无 `FLICKER-DEBUG*`、视频拆帧、临时 Swift/CDP probe、截图或其他本轮
调试垃圾残留。下一刀回到 `CommandExecution` typed increment，不在 Plan 上保留第二套 wire。

### V1-23 CommandExecution outputDelta 生命周期收口（2026-07-24）

目标：按 Codex v2 对齐 `item/commandExecution/outputDelta`，让 Runtime 原始输出、App Server
public notification、Renderer tool item 使用同一个 canonical `itemId`，并清理 metadata-only
事件伪造增量的风险。

写集与唯一 owner：

- `app-server-protocol` 的 `CommandExecutionOutputDeltaNotification`、v2 method/schema/codegen 与
  serde 回归；`item/commandExecution/terminalInteraction` 暂不接入，因为当前 Runtime 没有可靠的
  `stdin` source event，不伪造 terminal interaction。
- Runtime `coding_events.rs` 的 `command.output` 增加完整原始 `delta`；流式 delta 保留换行，
  无流式增量的 result fallback 使用完整 `result.output`，metadata-only 事件不写 delta。
- App Server `V2NotificationProjector` 新增 command owner：只接受已 started 且未 completed 的
  `commandExecution` item，身份不一致、缺 delta、late delta 均 fail closed。
- Renderer v2 route、sequence gate 和既有 `tool_output_delta` projection 复用 canonical source
  item，不新增第二种 GUI item。

当前链路：`command.started -> item/started -> command.output -> item/commandExecution/outputDelta*`
`-> command.exited -> item/completed`。内部 `command.output` 仍是 event-log/projection 表达，
public wire 只暴露 Codex typed notification。

验证：

- `cargo check -p app-server-protocol -p app-server` 通过；protocol serde、App Server projector、
  Runtime coding event 定向测试通过。
- Renderer v2 notification / sequence gate 定向测试 `35/35` 通过。
- public JSON-RPC integration 覆盖 started、同 itemId 的 output delta、completed，以及 late output
  fail-closed；schema fixtures 与 TypeScript generated types 已同步。
- `npm run test:contracts`、`npm run governance:legacy-report`、
  `npm run smoke:agent-runtime-current-fixture` 与 `npm run verify:gui-smoke` 通过；其中 Electron fixture
  覆盖首页短问候、Claw 工作台、App Server current read model 与 GUI 终态，未使用 live Provider。
- `npm run verify:local` 在既有 `i18n:unused -- --check` 门禁处停止：当前仓库有 `38` 个未引用 key
  候选，且本轮未改 `src/i18n/**`。该跨域基线问题未通过删除翻译或新增白名单掩盖；本轮其余定向、
  contract、runtime fixture 与 Electron GUI 门禁均已独立通过。

分类：`current = item/commandExecution/outputDelta + 单一 CommandExecution lifecycle + canonical
itemId`；未接入的 `terminalInteraction` 为 `gap`，不是兼容实现；旧的 metadata-only 伪造 delta
属于 `dead / deleted / forbidden-to-restore`。下一刀进入 `item/fileChange/patchUpdated`，不恢复
Codex 已 deprecated 的 `item/fileChange/outputDelta`。

### V1-24 FileChange patchUpdated 生命周期收口（2026-07-24）

目标：按 Codex v2 对齐 `item/fileChange/patchUpdated`，让 Runtime patch 快照、App Server public
notification、typed client 与 Renderer patch item 使用同一个 canonical `itemId`；不恢复 Codex 已
deprecated 且服务端不再发送的 `item/fileChange/outputDelta`。

写集与唯一 owner：

- `app-server-protocol` 新增 `FileChangePatchUpdatedNotification`、method/schema/codegen 与 serde 回归；
- App Server `V2NotificationProjector` 的 `file_change.rs` 直接投影既有 `patch.started/applied/failed/declined`，
  不新增第二套 Runtime event；
- `packages/app-server-client`、`packages/agent-runtime-client`、Renderer v2 route 与 sequence gate 只消费
  typed `patchUpdated`，并复用现有 patch GUI item；
- `internal/refactor/v1` inventory 与 public JSON-RPC integration 记录 canonical identity 和禁止回流项。

当前链路：MCP manager 的真实 `ProgressNotification` 由 Runtime 转成带
`notification_kind=mcp_progress` 与精确 route identity 的 `AgentEvent::ToolProgress`；App Server 只在
canonical `McpToolCall` 已 started 且未 completed 时输出 `item/mcpToolCall/progress`。typed client 与
Renderer 继续使用 notification 的 canonical `itemId` 更新同一工具 item，不按工具名推断 MCP 类型。

当前链路：`patch.started -> item/started -> item/fileChange/patchUpdated -> patch terminal ->
item/completed`。`patchUpdated` 携带完整结构化 `changes[]` 快照，canonical item identity 来自
Thread/Turn/Item projector，实测 `patch-1` 归一为 `item_patch-1`；空 changes 不发更新。重复 start、
terminal-before-start、重复 terminal、started 前/terminal 后的 Renderer update 和 item type drift 均
fail closed。

验证：

- `cargo check -p app-server-protocol -p app-server` 通过；protocol serde 与 App Server projector
  `21/21` 通过；
- public JSON-RPC integration 通过，锁定 `item/started -> item/fileChange/patchUpdated ->
  item/completed` 及同一 canonical `itemId`；
- Renderer v2 notification / sequence gate 与 inventory `42/42`、`packages/app-server-client` `79/79`、
  `packages/agent-runtime-client` `23/23` 通过；schema fixture、`npm run check:protocol-types` 与 generated
  TypeScript 无漂移；`npm run test:contracts` 通过 `296` 项 contract checks；
- `npm run smoke:agent-runtime-current-fixture` 通过，覆盖真实 Electron 首页首发、Claw 终态、停止后
  同会话继续、approval、Plan、MCP、Skills 与工作台分支，`liveProviderUsed=false`；
- `npm run verify:gui-smoke` 通过，真实 Electron Desktop Host、preload/IPC、App Server 初始化、Claw
  工作台 reload 与记忆设置均完成；Gate B evidence 为
  `.lime/qc/project-gates/standalone-shell-01-20260724025631-60196/shell-01-electron-smoke/summary.json`；
- `npm run governance:legacy-report`、`git diff --check` 与生产源码残留扫描通过；无 legacy 分类漂移、
  边界违规、`FLICKER-DEBUG*`、`console.trace`、`FileChangeOutputDelta`、视频拆帧或临时 probe 残留。

分类：`current = item/fileChange/patchUpdated + 单一 FileChange lifecycle + canonical itemId`；
`item/fileChange/outputDelta` 为 `deprecated / excluded / forbidden-to-restore`；文本 patch delta、重复
GUI item 与临时调试路径为 `dead / deleted`，未新增 `compat`。下一刀进入
`item/mcpToolCall/progress`。

### V1-25 MCP Tool Call progress 生命周期收口（2026-07-24，进行中）

目标：按 Codex v2 对齐 `item/mcpToolCall/progress`，让真实 MCP
`ProgressNotification`、Runtime `McpToolCall` item、App Server public notification、typed client 与
Renderer 使用同一个 canonical `itemId`。不允许 Renderer 伪造 progress，也不允许 App Server
按工具名猜测 MCP 类型。

写集与唯一 owner：

- `tool-runtime::McpStepSnapshot` 暴露已冻结的 server、raw tool 与 model-visible tool route identity；
- `agent/current_provider_turn` 在同一 sampling step 共享 route map，把 MCP started/completed 投影为
  `ThreadItemPayload::McpToolCall`，并仅把真实 `ProgressNotification` 转为 `AgentEvent::ToolProgress`；
- `app-server-protocol`、App Server `V2NotificationProjector`、两个 typed client、Renderer v2 route 与
  sequence gate 接入 `item/mcpToolCall/progress`；
- `internal/refactor/v1` inventory 与 public JSON-RPC integration 记录 canonical identity 和禁止回流项。

退出条件：事件顺序固定为 `item/started -> item/mcpToolCall/progress* -> item/completed`，三者复用
同一个 canonical `itemId`；started 前、terminal 后、generic Tool item、空 message 与非
`notification_kind=mcp_progress` 的 progress 均 fail closed。验证至少覆盖 Rust related、public
JSON-RPC、`npm run test:contracts`、`npm run smoke:mcp-current`、
`npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke`、治理扫描和残留清理。

当前实现已将 RMCP `RequestHandle.progress_token` 保留到调用级 subscriber，并从
`McpConnectionCall` 向 Agent executor 传递调用级 notification stream。旧 connection-wide
subscription、广播 handler 与空订阅测试已退出生产路径；`ProgressToken` 保持 number/string wire
类型，不再转成 Debug 字符串。当前分类为 `current = request-token correlation + canonical itemId`；
`dead = connection-wide progress subscription`，未新增 `compat/deprecated`。

已验证：`lime-mcp` 151/151、`lime-agent --lib` 271/271、`tool-runtime` 271/271；Rust related
扩展到 `agent-runtime`、App Server、Agent、MCP、Scheduler、Server、Tool Runtime 七个 crate 后
退出码为 0；public JSON-RPC MCP progress 2/2、protocol round-trip 1/1、Renderer sequence/projection
42/42、App Server client 80/80、Agent Runtime client 23/23 通过。`npm run test:contracts`、
`npm run smoke:mcp-current`、`npm run governance:legacy-report` 与全树 `git diff --check` 通过；MCP
smoke 复用已运行的 Desktop Host/DevBridge，没有启动第二套 Electron。

仍未满足退出条件：完整 `lime-agent` 测试会额外编译已脱离生产模块树的
`tests/legacy_permission_surfaces.rs`，其中两个历史断言与同样已退出编译图的旧权限实现不一致；
该 surface 已证明为 `dead-candidate`，应在明确删除确认后连同旧守卫/文档改为
`dead / deleted / forbidden-to-restore`，不得恢复旧 `bash` alias 或虚构默认权限。现有
`electron:dev` 仍由并行进程持有，本轮没有重复启动 `smoke:agent-runtime-current-fixture` 或
`verify:gui-smoke`；这两项重跑前 V1-25 继续保持“进行中”。另外，RMCP 官方 subscribe-after-send
模式仍存在极快首帧在 subscriber 注册前丢失的理论窗口；本切片只声明 token 隔离，不声明严格
lossless，也不把 elicitation owner gate 下的串行调用写成真正 parallel MCP evidence。

### typed `artifact/write` 与旧 runtime append 删除收口（2026-07-24）

目标：让 Artifact 保存只经过 `Renderer typed gateway -> App Server JSON-RPC v2 -> RuntimeCore ->
ThreadStore/artifact read`，删除公共任意 RuntimeEvent 注入入口，不保留 compat。

完成结果：

- `artifact/write` protocol、processor、typed package client、Renderer gateway 与 Workspace 保存链已接通；
  Content Factory current fixture 使用 typed writer，不再追加 arbitrary workflow/error event。
- `agentSession/runtimeEvents/append` 的 package/Renderer wrapper、Rust v0 method/DTO/catalog/schema/
  fixture/handler/dispatch 和旧正向测试已物理删除。public JSON-RPC 对旧 method 返回
  `METHOD_NOT_FOUND`，package export 与 governance catalog 均有禁止回流守卫。
- generic append 不再按 `"artifact.snapshot"` 字符串放行终态 turn。私有
  `TerminalTurnAppendPolicy` 只允许 durable recovery 和 typed `append_artifact_snapshot` 显式写入；
  `artifact/write` processor 不能提交任意 `RuntimeEvent`。

验证：processor 终态 typed 保存 1/1、public `artifact/write -> artifact/read` 1/1、package artifact
3/3、Renderer/App Server artifact 56/56、fixture guards 91/91、protocol codegen 零漂移、
`npm run test:contracts` 与治理扫描通过。

独立真实 Electron Gate B 已关闭：`direct-session` backend 不发 `artifact.snapshot`，Turn terminal 后
由 Renderer current `AppServerClient.writeArtifact` 经 production `safeInvoke` 发出 typed write。
summary 精确记录 `app_server_handle_json_lines / electron-ipc / artifact/write`，并校验 trace、typed
response、read model 的 `threadId/turnId/artifactRef` 一致；GUI 完成 hydrate 并打开 Workbench。
证据为 `.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/typed-artifact-write-v2-summary.json`、
`typed-artifact-write-v2-workbench.png` 与 `typed-artifact-write-v2-backend-ledger.json`；`ok=true`、
backend Artifact 注入为零、console/page error 为零。新增 fixture guard 7/7 通过；最终 contracts 为
764 protocol types 零漂移 / 296 client checks，scripts governance、legacy report 0/0/0 和全树 diff
check 通过。分类：typed writer 与领域 artifact append 为 `current`；旧 append 为
`dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。本切片状态为 `closed`。

## 完成定义

只有同时满足以下条件才能称为“完全对齐 Codex”：

1. Agent `protocol/v0`/production `agentSession/*`、其余 v0 方法迁入 v2 后的旧 module、`lime-providers`、旧 repository 和第二 history 已物理删除；
2. Thread/Turn/Item、raw rollout、ThreadStore metadata patch、compaction lineage、rollback/fork/recovery 通过；
3. Tool/hooks/sandbox/approval/MCP/Skills/Plugins/Apps/AgentControl/transport 只有一个 current owner；
4. 每个 Turn 的 effective route、capability、attempt、usage、cost/quota、provider identity 可追溯；
5. Gate A/B 证明真实 Electron 与 cold/live/replay/restart 同一 canonical identity；
6. `governance:legacy-report`、`test:contracts`、Rust workspace/受影响 crate 测试通过，且扫描守卫阻止双轨回流。
