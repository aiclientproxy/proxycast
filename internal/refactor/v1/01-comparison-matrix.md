# Codex 与 Lime 逐项对照

本表比较能力语义、字段、状态、恢复和真实消费证据，不比较文件数量。`partial`
表示已有实现但尚未证明 Codex 等价；`wrong owner` 表示能力存在但事实源错误。
本仓库没有外部客户和历史兼容负担，`agentSession/*`、`lime-providers` 和旧
repository 不按长期兼容处理。

## 主链矩阵

| # | Codex current 能力与证据 | Lime 当前 owner/事实 | 状态 | 关键缺口 | 下一刀/退出条件 |
| --- | --- | --- | --- | --- | --- |
| 1 | v2 method registry：`thread/start|resume|fork|archive|delete|unarchive|read|list|search|loaded/list|turns/list|items/list|inject_items`、`turn/start|steer|interrupt`，以及 typed server request/notification；`codex-rs/app-server-protocol/src/protocol/common.rs`、`protocol/v2/**` | `app-server-protocol/src/protocol/v0/**`，Electron/App Server/Renderer 生产主路径仍大量使用 `agentSession/*` | **missing / wrong owner P0** | 不是简单命名差异：method、request/response、server request、notification、实验 API 和 cancellation 都是另一套协议；`agentSession` 不能继续作为 current write path | 迁移 protocol/client/host/renderer/catalog/fixture；生成 v2 schema round-trip；删除 v0 与所有 production `agentSession/*` 引用 |
| 2 | Thread/Turn 字段级 contract：Thread id/sessionId/forkedFromId/parentThreadId/preview/ephemeral/historyMode/modelProvider/status/path/cwd/source/threadSource/gitInfo/name；Turn error、duration、started/completed 秒级时间；`protocol/v2/thread_data.rs` | `agent-protocol/src/thread.rs` 与 App Server read model 使用 session_id、毫秒时间、archived/admission/queue 等 Lime 字段 | **partial P0** | 缺 v2 必填字段、时间单位、错误形状、fork lineage、source/environment；额外字段不能替代 Codex 字段 | 建立字段级 schema diff、时间/ID/ordinal 约束和 unknown/late update 测试；冷读/live/evidence 同一 identity |
| 3 | `ThreadHistoryBuilder` 按 raw rollout change coalesce repeated item/turn snapshot，rollback 删除 turn changes 并保留 active snapshot；`protocol/thread_history.rs`、`thread_history_projection_tests.rs` | `event_log*`、`projection_store*`、`thread_item_projection*`、`conversation_import` | **partial / wrong owner P0** | 尚未证明 coalesce、rollback、分页和 malformed tail 语义；import/event DTO 可能形成第二 history | 以 raw canonical append + 独立 metadata patch 为 ThreadStore contract；history builder round-trip、rollback/fork boundary 通过 |
| 4 | `ThreadStore::append_items` 只追加 canonical rollout item；`update_thread_metadata` 独立 patch；`LiveThread` 拥有 rollout persistence；Local store 为 JSONL history + SQLite metadata | App Server `ProjectionStore`、EventLog、ThreadStore、旧 `session_repository` 并存，当前文档曾把 EventLog-first 误写成唯一 storage 语义 | **partial / wrong owner P0** | append 不得从 item 推断 metadata；EventLog repair、ThreadStore materialize、metadata patch 的事务边界未完全锁定；旧 repository 必须删除 | `ThreadStore` trait + App Server store adapter 只有一条 canonical 写路径；EventLog 仅顺序/repair provenance；删除旧 repository/第二 transcript DB |
| 5 | Context manager、`compact.rs`、`rollout_reconstruction.rs` 使用 `replacement_history`、`window_number`、`first/previous/window_id`，resume/fork/rollback 重建窗口链 | `context_compaction.rs`、`provider_history.rs` 主要依赖 summary + `tailStartTurnId` | **missing / partial P0** | summary 是否成为 canonical/model-visible item、窗口 lineage 和 durable replacement history 未证明；可能出现 summary + 全量旧 history 双发 | 补 `ContextCompaction` 字段与窗口链；durable history 不删除，只重建 provider history；重复压缩、损坏尾部、resume/fork/rollback 回归通过 |
| 6 | Session/Turn task、input queue、cancel/steer、terminal lifecycle、token budget；`core/src/session/**`、`tasks/**` | `agent-runtime` session loop、reply loop、runtime queue 与 App Server runtime | **partial** | 多套 queue/reply/timeout 状态机和 restart/provider-selection 异常路径未证明等价 | 用 Codex lifecycle state table 和 single queue owner；queued/in-progress/completed/failed/interrupted/cancelled 全状态恢复测试 |
| 7 | `ThreadItem` tagged union：UserMessage、HookPrompt、AgentMessage、Plan、Reasoning、CommandExecution、FileChange、McpToolCall、DynamicToolCall、CollabAgentToolCall、SubAgentActivity、WebSearch、ImageView、Sleep、ImageGeneration、Entered/ExitedReview、ContextCompaction、MemoryCitation；`protocol/v2/item.rs` | `thread_item_projection*`、`agent-protocol` payload broad union，字段与 GUI evidence 未逐项锁定 | **partial P0** | 缺 Hook fragments、command action/process/duration、MCP app context/result、dynamic content items、memory citation、review/compaction/media 字段及 projection/replay 证据 | 建立字段级 Item inventory；每个变体验证 started/delta/completed、cold/live/replay、provider history 和 GUI projection |
| 8 | Tool registry/exposure：Direct/Deferred/Hidden、ToolSearch/LoadableToolSpec、parallel capability、argument diff、CodeMode/Direct source、统一 output truncation/call-output pairing；`core/src/tools/registry.rs` | `tool-runtime`、`runtime_backend/native_tools`、`tool_inventory`、provider stream reducer | **partial / wrong owner candidate P1** | 只有 executor/side effect 维度；deferred discovery、hook lifecycle（SessionStart/UserPromptSubmit/PreToolUse/PermissionRequest/PostToolUse/PreCompact/PostCompact/Stop/SubagentStop）和 output pairing 缺 contract | `ToolSnapshot + HookSnapshot + RuntimeTool::execute_call` 唯一链；provider 只产 typed call；hook block/abort/rewrite 和 output sidecar 回归通过 |
| 9 | sandbox、permission profile、execpolicy/network amendment、guardian review/source、session approval cache、Windows sandbox；`core/src/sandboxing`、`exec_policy`、`guardian` | `permission_preflight.rs`、`approval_decision_contract.rs`、Electron approval gateway | **partial P1** | permission profile、network/exec amendment、guardian source、unsandboxed shell、跨平台恢复字段不完整；approval 与 capability 跨层重复 | 单向 `PermissionProfile -> SandboxPolicy -> ApprovalRequest`；deny/timeout/cancel/allow-for-session、macOS/Windows/Linux 和重启恢复通过 |
| 10 | MCP manager 每 step 生成 immutable `McpRuntimeSnapshot`，含 config/overlay、connection manager、environment、required/optional、OAuth/scopes、tool/resource/prompt/subscribe、elicitation pause/timeout | `app-server/runtime/mcp.rs`、`McpStepSnapshot`、`mcp_elicitation.rs`、local data source | **partial P1** | snapshot 只有 tools/routes/caller；缺 generation/auth/environment provenance、refresh-inflight、OAuth/retry 和 elicitation lifecycle 计数 | 管理面与 sampling 面分离；旧 step 保持 `Arc` snapshot；elicitation 只为 typed reverse waiter，不写 durable history |
| 11 | Skills watcher + scope/precedence/injection/implicit invocation；Plugins trust/manifest/selected package；Apps async cache/auth/readiness/update notification | `skill_registry`、`runtime/skills.rs`、`plugin_packages*`、`local_data_source/{skills,plugins}` | **partial P1** | CRUD/catalog 存在但 watcher、dependency-driven MCP/OAuth、plugin host lifecycle、app readiness/cache/update 未证明 | service owner 生成 typed projection/notification；GUI 不读 DB、不维护第二 catalog；changed/restart/replay 回归通过 |
| 12 | Agent control V2：root-scoped registry、role/config precedence、max depth/width、residency/execution limiter/rollout budget、fork all/last-N/none、trigger_turn/queue-only、wait mailbox/steer/timeout；`core/src/agent/control*`、`tools/handlers/multi_agents_v2` | `thread-store/{agent_graph,agent_identity,agent_mailbox}`、App Server agent control、agent-runtime | **partial P3** | graph/identity/mailbox 已有，但 role/limits/residency/budget、V1/V2 split、wait priority 和 child tool scope 未全对齐 | durable graph/identity/mailbox 唯一 owner；fork lineage、concurrent wait、recovery fuzz、Gate B 通过；不得恢复 Team/旧 alias |
| 13 | App Server transport：stdio、WebSocket、Unix control socket、bounded ingress/outbound、slow-client disconnect、per-connection initialized/experimental/opt-out notification；`app-server-transport/src/transport/{stdio,websocket,unix_socket}.rs` | `app-server-transport` 只有 `stdio.rs`；`mod.rs` 仅声明/解析 ws/unix/off，`app-server/main.rs` 对非 stdio 直接 unsupported；广播存在 unbounded channel | **partial / misleading P1/P3** | enum/URL 解析不是实现；缺 ws/unix acceptor、逐连接状态、slow-client、notification filtering；并发输出可能无界 | 要么实现 Codex acceptor/backpressure/filtering 并补 transport tests，要么删除未实现 variant；禁止声明式兼容 |
| 14 | Provider client 只输出 normalized event；provider wire 不进入 Thread/Turn/Item；Responses/WebSocket/SSE reducer 保留 unknown/partial/error | `model-provider/{current_client,lowering,provider_stream,runtime_provider}`，另有 `lime-providers` | **partial / wrong owner P0** | provider raw tool/event 仍可能穿透；重复 crate 维护第二 trait/stream/session | `model-provider` 唯一 network owner；canonical content/lowering/stream reducer 统一；删除 `lime-providers` |
| 15 | Model catalog/route：bundled+remote/cache/ETag/auth visibility、explicit allow/disabled/hidden、credential readiness、capability matrix、auxiliary model slots | `model-provider::CanonicalModel`、`core::EnhancedModelMetadata`、runtime JSON `CapabilitySnapshot`、services registry 三套结构 | **wrong owner / partial P0** | 多次推导；unknown capability 默认 true；builtin provider 可能无 key 即 ready；model cache 不含 credential/tenant identity；variant/auxiliary route 缺失 | 一个 `ModelCatalogEntry`/`ModelCapabilitySnapshot`；route 必须 match+credential+capability；失败保留旧 catalog，identity 变更精确 invalidation；未知 fail closed |
| 16 | Session model switch 检查 active agent/harness compatibility、zero-turn rebuild、reasoning gate、watch generation；retry/breaker 按 transport attempt | `model_route_resolver`、`runtime-core/model_route`、provider retry | **partial P2** | route 可能只按 readiness 放行；per-model endpoint/header/auth、variant、idle timeout/max retries、compaction/title/search/image/subagent auxiliary route 未固化 | 每个 Turn 持久化 `ResolvedModelRoute`/attempt；可见 event 后禁止 fallback；switch/restart/child/auxiliary model 回归通过 |
| 17 | Rollout/replay/evidence：canonical raw rollout、Compacted replacement、rollback marker/fork cut、unknown line policy、usage/trace provenance | EventLog JSONL、canonical Thread/Turn/Item read model、conversation import | **partial P4** | Lime AgentEvent JSONL 未证明可 round-trip Codex RolloutItem；derived facts 可能从 raw event 补 GUI 状态；import provenance 边界需锁定 | raw rollout replay 与 ThreadHistoryBuilder 互测；repair 先于 materialize；derived facts 只能读 canonical store/route provenance |
| 18 | AGENTS.md/environment selection、config lock、startup prewarm、sticky roots、turn timing/memory/usage metadata | Lime context packets、request_context、diagnostics、settings | **partial P2** | instruction source/precedence、child environment inheritance、config replay/prewarm 和 usage/OTEL 与 Thread/Turn identity 未纳入恢复 | environment/config/startup snapshot 在 Turn 固化；restart/replay 不改变 instruction source 或 route；缺失显式 fail |
| 19 | `backgroundTerminals`、`process/*`、`command/exec/*`、`fs/watch`、`fuzzy file search`、`realtime/*`、review、file checkpoint、dynamic tools、attestation、current time、memory mode | Lime 有若干独立 process/files/media/voice/guardian 模块与旧 WebSocket backend | **missing or explicit-scope decision** | 代码存在不等于 App Server v2 contract；旧 WebSocket/HTTP handler 可能是第二业务协议 | 每项登记“实现/产品范围排除”；实现必须进入 App Server v2/current owner，排除项从 current catalog 和完成定义删除 |
| 20 | 多客户端只消费 App Server；schema/codegen/experimental filtering 统一 | Electron/Renderer current，但仍有 legacy bridge、mock、fixture 和 agentSession method assertions | **partial** | Gate B 尚未证明 v2 wire、real IPC、read model、provider route、restart 全链；mock 不能冒充 live provider | 迁移 Host/preload/gateway/catalog/fixture；Gate A/B 证明真实 Electron -> App Server -> RuntimeCore -> ThreadStore -> GUI |

## 分类结论

### current（保留并继续演进）

`agent-runtime`、`runtime-core`、`thread-store`、`tool-runtime`、`model-provider`、
App Server、Electron Desktop Host 是目标 owner。`ProjectionStore` 只能作为 App Server
对 ThreadStore contract 的实现，不另立 history owner。

### wrong owner / deprecated（迁移后删除）

- `app-server-protocol/src/protocol/v0/**` 与所有 production `agentSession/*`；
- `lime-rs/crates/providers/**`（crate `lime-providers`）及其直接消费者；
- `core`/`services`/runtime JSON 的重复 model catalog/capability 推导；
- provider raw tool mapper、旧 session repository、第二 transcript/event store；
- 只声明但未实现 ws/unix/off 的 transport variant。

### compat（仅短期边界，不承接新逻辑）

本项目不保留长期 compat。只有一次性 Codex rollout import/source adapter 可以在输入
边界存在，导入后必须立即写入 canonical ThreadStore，并带 provenance；迁移完成即删除
import-only runtime DTO。

### dead / deleted / forbidden-to-restore

旧 Team/`agent_runtime_*` 第二后端、Renderer synthetic roster、production mock/fallback、
provider-specific GUI state、旧 WebSocket agent scheduler 和任何“从 raw event 猜 UI 状态”
路径均属于 dead；新代码或 catalog 不得恢复。
