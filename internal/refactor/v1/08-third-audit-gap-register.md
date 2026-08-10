# 第三轮对比查缺与范围校正

状态：P0/P1 阻塞登记（2026-07-18）

本轮是对 [01-comparison-matrix.md](01-comparison-matrix.md) 和
[07-second-audit-gap-register.md](07-second-audit-gap-register.md) 的再次审计。审计对象仍是
Codex current runtime 与 Lime current owner；Grok Build 和 OpenCode 只用于
`model-provider` 的控制面、wire/lowering 和多协议 stream 对照。结论不是文件数量比较，
而是比较状态、身份、恢复、协议和证据是否可证明。

## 1. 总结结论

当前不能称为“完全对齐 Codex”。至少有七个阻塞面：

1. Lime Agent 生产写路径仍在 `app-server-protocol/src/protocol/v0/**` 的
   `agentSession/*`，尚未完成 Codex v2 method registry 和 Thread/Turn/Item lifecycle。
2. Thread 字段、时间单位、history mode、fork/metadata 生命周期没有逐字段闭合；
   Lime 把 admission/queue/approval 扩展放在 Turn，必须明确为 Lime extension，而不是
   误称 Codex contract。
3. Lime 的 history/compaction 仍以 summary/tail 为主，缺少 Codex rollout 的
   `replacement_history` 和 context-window lineage，resume/fork/rollback 不能证明确定性。
4. App Server 重启恢复存在真实故障：缺少 provider/model selection 时，control spawn
   recovery 不能让 warmup 失败。恢复必须持久化或显式传递完整 route，并在缺失时 defer
   mailbox，不得让 App Server 进程退出。
5. Lime `ProtocolKind` 宣布的部分 provider（Gemini、Bedrock、Fal、Vertex 等）在
   `model-provider` 中会落入 `Custom`，当前 client 再按 Chat Completions 发送；这是
   协议错配和 fail-open，不是 partial capability。
6. transport enum 已声明 WebSocket/Unix/off，但实际 App Server 入口只启动 stdio；
   URL parser 和单元测试不能替代真实 acceptor、health probe、逐连接能力和慢客户端行为。
7. hooks、AGENTS.md/world-state、tool exposure/repair、MCP snapshot、Multi-Agent
   role/limit/residency、rollout/evidence provenance 尚未达到 Codex 的可恢复契约。

## 2. 事实源与分类校正

### 2.1 唯一 owner

| 能力                                                               | current owner                      | 禁止路径                                                |
| ------------------------------------------------------------------ | ---------------------------------- | ------------------------------------------------------- |
| Thread/Turn/Item、method/notification、server request              | `app-server-protocol` + App Server | `agentSession/*` 新写入、Renderer transcript 拼接       |
| session/turn/queue/steer/cancel/recovery                           | `agent-runtime` + App Server       | Electron loop、第二 session state machine               |
| canonical model context、compaction、world state                   | `runtime-core` + `thread-store`    | provider history tail 直接充当 durable truth            |
| catalog、route、capability、credential readiness、lowering、stream | `model-provider`                   | `lime-providers`、provider adapter 直接生成 Item        |
| tool、approval、sandbox、MCP dispatch                              | `tool-runtime`                     | App Server 拼 provider body 或 GUI 作为 policy owner    |
| durable Thread/Turn/Item、graph、identity、mailbox                 | `thread-store`                     | EventLog、Renderer cache 或 session metadata 第二事实源 |

### 2.2 不能把整个 `protocol/v0` 直接判 dead

现有 `protocol/v0` 同时包含 Agent 旧入口和 Lime 自有的 workspace、browser、media、voice、
channel、MCP 管理等产品方法。应按方法组迁移和分类：

- `agentSession/*`、旧 Agent event DTO、旧 Agent action/respond/replay：
  `deprecated -> deleted`，完成 v2 迁移后由负向 guard 防回流。
- `thread/read/list/turns/list/items/list` 等已存在的读取能力：迁移到 v2 schema，不能
  继续以 v0 DTO 作为 current contract。
- 非 Agent 产品方法：逐组写入产品范围表，确认 current owner 后保留或迁移；没有产品
  决策前不得因为 Codex 对齐而静默删除。

因此，删除条件是“Agent 旧 surface 和重复 DTO 已无生产引用”，不是删除整个协议目录。

## 3. P0 新增和细化缺口

### P0-01：协议 registry、握手和方法范围

Codex 的 `app-server-protocol/src/protocol/common.rs` 已把请求、通知、server request、
serialization scope 和 v2 schema 绑定；README 还要求每条连接执行
`initialize -> initialized`，支持 `experimentalApi`、MCP form elicitation、精确的
`optOutNotificationMethods` 和 `-32001` overload。Lime 当前只有 v0 catalog，
`InitializeParams` 只有 `eventMethods/experimental`，Processor 只保存 `clientInfo`，
没有逐连接 notification opt-out 或 v2 experimental gating。

已补 machine-readable method matrix：
`internal/refactor/v1/fixtures/codex-method-product-scope.v0.1.json`。矩阵覆盖 Codex
注册表的 221 个方向化 method，每个只能标记为 `implemented`、`product-scope-excluded`
或 `planned`，并由
`src/lib/governance/codexMethodProductScopeBoundary.test.ts` 固定方向计数、唯一性、
owner/evidence、planned gap、excluded rationale 和 Lime generated manifest 同方向契约。
这一步不把“模块存在”当成实现，未闭合的范围继续保留在下表和矩阵的 `planned`。

`thread/increment_elicitation` 与 `thread/decrement_elicitation` 已补 exact v2 schema、Thread
serialization/exclusive access、loaded-thread `i64` 引用计数、溢出/下溢、archive/delete/unload
清理、typed clients 与 public JSON-RPC evidence，因此 method boundary 已从 `planned` 迁入
`implemented`。剩余生命周期缺口是把该 Thread-level pause state 接入 `agent-runtime` provider
active-time budget；现有 MCP connection-local pause state 不得冒充这个 consumer。

`thread/approveGuardianDeniedAction` 已补 exact opaque-event v2 boundary、Thread serialization、loaded
Thread 校验、typed Guardian action parse、Codex exact-action developer marker、session actor active-turn
delivery、durable provider-only history 与 public JSON-RPC evidence，因此 method boundary 已从 `planned`
迁入 `implemented`。Lime 仍未生产 Guardian assessment/review lifecycle；该 producer gap 继续留在 MCP/tool
review 工作，不得把手工 continuation method 冒充完整 Guardian reviewer。

`thread/inject_items` 已补 exact opaque-item v2 boundary、Codex current `ResponseItem` validation union、
remote image guard、active regular Turn session actor delivery、durable `response_item.injected` provider-only
history、restart/cold hydrate、archived fail-closed、Responses 原样 lowering、非 Responses route fail-closed 与
Rust/TypeScript typed client/public JSON-RPC evidence，因此 method boundary 已从 `planned` 迁入
`implemented`。raw item 不生成用户可见 Thread Item，也不经过 `ThreadItemPayload::Extension`。P0-03 仍保留
全局 canonical history/rollout、rollback/fork/replay 与未知记录一致性的更大退出条件；不得用本 method
boundary 的完成冒充整个 history parity 已关闭。

最低分组如下：

| 分组               | Codex surface                                                                                                                              | Lime 退出条件                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- | ---- | ----------- | ---------- | ---------- | ------- | ------ | --------- | ----------- | -------- | --------------- | --------------- | -------- | ------------- | ------------------------------------------------------------------------------------------------------------- |
| Thread lifecycle   | `thread/start                                                                                                                              | resume                                                                     | fork                                                                                       | read                                                                                                | list | loaded/list | turns/list | items/list | archive | delete | unarchive | unsubscribe | name/set | metadata/update | settings/update | rollback | inject_items` | v2 request/response/notification、cursor、subscription、status 和 fork boundary 全部有 schema/handler/fixture |
| Turn               | `turn/start                                                                                                                                | steer                                                                      | interrupt`                                                                                 | accepted 与 started 分离；steer/interrupt 的 active-turn、review/compact 拒绝和 terminal 状态可恢复 |
| Item/approval      | `item/started                                                                                                                              | completed`、delta、approval、dynamic tool、MCP progress                    | server request 精确关联 connection/request id；unknown method/late delta/deny 不产生副作用 |
| Process/command/fs | `command/exec/*`、`process/*`、`fs/*`                                                                                                      | 真实 owner 或明确产品排除；不能用 `executionProcess/*` 同名映射冒充 parity |
| Runtime controls   | `model/list`、`permissionProfile/list`、`environment/*`、`hooks/list`、`skills/*`、`plugin/*`、`app/list`、`review/*`、`thread/realtime/*` | 每个组写明 current owner、实验 gate、通知和 Gate B 证据                    |

`modelProvider/capabilities/read` 已从 Runtime controls 退出：Codex 的空参数方法读取单一全局 Provider，
不能表达 Lime 的 per-thread provider/model route；旧静态实现无产品消费者，已归入
`product-scope-excluded / dead / deleted / forbidden-to-restore`。能力事实继续绑定 executable model
catalog、resolved route 与 provider lowering，不新增全局兼容读。

### P0-02：Thread/Turn/Item 逐字段契约

Codex v2 `Thread` 还包含 `ephemeral`、`historyMode`、`path`、`cwd`、`cliVersion`、
`source`、`threadSource`、`forkedFromId`、`gitInfo`、`name`、`recencyAt`、
`agentNickname`/`agentRole` 和秒级时间戳；`Turn` 只有 canonical status、error、
started/completed/duration 和 `itemsView`。Lime `agent-protocol/src/thread.rs` 目前使用
毫秒时间戳、`archived`/metadata 扩展，并把 admission/queue/approval 放在 Turn。

退出条件：

- 建立字段对照表：名称、serde shape、单位、nullable、生成 owner、cold/live/replay 来源；
- Codex 字段保持 v2 语义，Lime 扩展使用明确的 `limeExtension`/版本化字段，不覆盖
  canonical status；
- `ThreadTurnsView`、`TurnItemsView`、`status`、`error`、ordinal、item id prefix 和
  terminal timestamp 在 schema fixture、read model、live notification、replay 中一致；
- `thread/start/resume/fork/read` 的 `ephemeral`、`path/cwd`、`forkedFromId`、
  `historyMode` 和 `includeTurns/itemsView` 有逐字段 round-trip 测试。

### P0-03：canonical history、ThreadHistoryBuilder 与 rollout

Codex `thread_history.rs` 和 rollout store 的 contract 包括：started/completed item 合并、
change accumulator 去重、rollback 删除 turn/item、late exec completion 归属原 turn、
unknown turn item 丢弃、review item、active-turn snapshot、item id/ordinal 保持，以及
`RolloutItem` 的 raw `ResponseItem`、compaction、rollback marker、world-state、
inter-agent metadata 和未知行处理。Lime materializer 已有类似测试，但不能证明它就是
model-visible history 或 rollout truth。

退出条件：

- `ThreadStore::append_items` 只接受 canonical rollout item；metadata patch 走独立 API；
- full materialization、incremental materialization、cold read、live read、replay/export
  得到同一 `(thread, turn, item, sequence, ordinal)`；
- coalesce、late completion、unknown turn、review mode、rollback、损坏尾部修复和未知
  event 保留均有定向测试；
- evidence 记录 raw canonical model context、source provenance、attempt/usage/route，
  不从 `AgentEvent` 文本反推 GUI item。

**2026-07-19 候选实现审阅（未接入）**：`thread-store/src/history.rs` 已被 `lib.rs` 暴露，
但没有生产消费者，App Server `thread_item_projection` 仍是实际 reducer。该候选尚未校验
outer sequence 与 item sequence，same-sequence 新 turn、跨 turn item identity、Turn 内嵌 items
与 flat items、exact retry fingerprint 均与现有 store contract 不一致。因此当前只能标记为
`current-candidate / rejected-for-integration`，不能关闭 P0-03；必须先收敛到唯一 reducer 并补
builder -> store -> cold/live/replay round-trip。

### P0-04：compaction window lineage

Codex `CompactedItem` 持久化 `replacement_history`、`window_number`、`first_window_id`、
`previous_window_id`、`window_id`；model-context scanner 用这些字段决定 bounded replay，
遇到旧形状、rollback 或缺字段会 fail-closed 扫描到头。Codex 还覆盖 pre-turn、inline/mid-turn、
remote compaction、token accounting 和 compaction hooks。Lime 目前的 summary、
`tailStartTurnId` 和 provider transcript tail 不能替代这些字段。

退出条件：

- compaction item 和 replacement history 是 durable canonical data，不删除旧 history；
- resume/fork/rollback/重复压缩/损坏尾部都能确定性重建 model context；
- window identity 进入 ThreadStore、provider request、trace/evidence 和 GUI read model；
- pre-turn、mid-turn、manual/remote compaction、预算耗尽和 hook 前后均有测试。

**2026-07-19 进度（partial）**：Lime 已持久化窗口字段，新增 UUIDv7 window id；replacement
history 现在保留压缩前用户边界并追加最终摘要 user message，provider history 仅消费最新
有效窗口和 tail，nested/top-level marker 缺字段时 fail-safe 回退完整事件历史。`thread/inject_items`
所需的 Codex current `ResponseItem` validation union 已进入 current owner；仍未完成
rollback/fork/replay 的窗口链确定性和 ThreadStore durable canonical compaction item 端到端证据；
不得关闭本 P0。

### P0-05：重启恢复不能因缺 route 退出 App Server

已观察到 Electron host 日志：

```text
failed to recover agent control spawns: ...
App Server runtime backend requires provider/model selection
```

`agent_control.rs` 已尝试持久化 child session defaults，`agent_mailbox_delivery.rs` 也有
defer 分支，但恢复入口仍需证明所有调用方都把 defer 当作可恢复状态，而不是 warmup fatal。

**2026-07-19 进度（partial）**：缺 route 现在由 `RuntimeCoreError::PendingRoute` 表达，
JSON-RPC 标记为 retryable；mailbox admission、graph edge 和 restart 测试保留等待状态，
不会 ack 或合成 terminal。providerConfig、route protocol、credential reference/effective
generation 尚未进入 durable defaults，已有 route 但 provider disabled/unknown model 时的
fail-closed 与 catalog 变更重试仍是 OPEN_REF。

退出条件：

1. 每个 child spawn 在 commit 前持久化完整 `providerPreference`、`modelPreference`、
   `providerConfig`、route protocol、credential reference 和 effective generation；
2. restart/recovery 优先使用 durable defaults，其次使用显式 `runtimeOptions.runtimeRequest`；
3. 两者都缺失时返回 typed `PendingRoute`，保留 mailbox/graph，不关闭 App Server、不 ack
   消息、不伪造 turn terminal；
4. provider catalog/credential 恢复后只触发一次 deterministic retry，重复重启、并发 wait、
   partial child cleanup 和错误尾部都有测试；
5. Electron warmup、App Server restart、Gate B 都验证进程存活和下一次恢复成功。

### P0-06：provider protocol 必须 fail closed

`app-server-protocol::ProtocolKind` 声明的多种协议不能自动成为可用能力；当前
`runtime_backend/model_route_contract.rs` 对未实现协议映射为 `ModelProviderProtocol::Custom`，
而 `model-provider/current_client.rs` 对 `Custom` 会按 Chat Completions/SSE 处理。结果是
Gemini、Bedrock、Fal、Vertex 等可能“路由成功、wire 错误”。

退出条件：

- `ProtocolKind -> ProviderWire` 是穷举映射；未实现协议返回
  `RouteFailure(unsupported_protocol)`，联网前失败；
- 每个已支持协议有独立 request lowering、content/media/tool result、stream reducer、
  usage/finish/error 和 retry fixture；
- `Custom` 只有注册了 explicit adapter、schema、capability 和 health check 才能使用；
- trace/read model/evidence 记录 protocol、endpoint、route source 和 failure，而不伪造成功。

## 4. P1 新增和细化缺口

### P1-01：Tool lifecycle 与 provider tool repair

Codex tools 还包含 `ToolExposure::{Direct, Deferred, Hidden}`、ToolSearch/loadable spec、
pre/post/permission hook、CodeMode、parallel orchestration、argument diff、cancellation/
teardown、output truncation、call/output pairing、extension tool 和 background terminal。
OpenCode 的 `experimental_repairToolCall` 会修正常见参数/大小写错误，无法修复时生成
`invalid` tool call；Lime 当前无同等 canonical 语义，invalid JSON 直接变成 provider error。

退出条件：tool definition、executor、hook snapshot 来自同一 sampling step；malformed args、
unknown tool、repair success/failure、cancel/timeout、truncated output 和 late completion
都只产生一组可重放的 typed lifecycle，不执行未授权副作用。

### P1-02：Approval、sandbox、guardian

逐项覆盖 permission profile、sandbox policy、exec policy amendment、network proxy approval、
per-key approval cache（尤其 apply_patch 多文件）、`AcceptForSession`、user/guardian reviewer、
hook-before-review 顺序、`additionalPermissions`、timeout/cancel/deny 无副作用，以及
macOS/Linux/Windows/unsandboxed `thread/shellCommand`。当前 preflight/decision/cache 模块
存在，但字段、恢复和跨平台证据未闭合。

### P1-03：MCP immutable snapshot

每个 sampling step 固化 config、plugin availability、connection manager、runtime context、
environment id、auth/scopes、generation、required/optional server、startup/tool timeout、
enabled/disabled tools。还需覆盖 OAuth、ordered plugin/connector overlay、tool catalog cache、
resource/prompt/subscription、elicitation pause counter、auto-deny、guardian review 和
replace-inflight 行为。管理面 live read 不能替换 in-flight snapshot。

### P1-04：Hooks、instructions、world state、environment

Codex 独立拥有 AGENTS.md discovery/cache/precedence、instruction source projection、
world-state snapshot/diff、SessionStart/UserPromptSubmit/PreToolUse/PermissionRequest/
PostToolUse/PreCompact/PostCompact/Stop hooks 及 started/completed telemetry。Lime plugin/workflow
hook 事件不能自动证明这些 Codex hooks 已实现；需要明确 source、阻断/改写语义、子环境继承、
config lock、startup prewarm、sticky environment 和 turn timing/usage provenance。

### P1-05：transport 与连接语义

Lime 当前 `AppServerTransport` 只在 parser/enum 层声明 stdio、WebSocket、UnixSocket、off，
App Server 实际入口仍调用 `start_stdio_connection`。必须二选一：

- 实现真实 WebSocket listener、Unix socket HTTP Upgrade、`/readyz`、`/healthz`、Origin 拒绝、
  connection cleanup、bounded ingress/request/outbound queue、`-32001` overload、逐连接
  initialized/notification opt-out；或
- 在产品范围明确只支持 stdio，并删除未实现的 ws/unix/off enum、URL 和正向 fixture，防止
  能力声明回流。

### P1-06：evidence/replay provenance

EventLog/trace/evidence 丰富不等于 Codex rollout 对齐。补 raw canonical context、compaction
replacement round-trip、malformed tail repair、unknown event preservation、attempt/usage/route/
provider credential fingerprint、source provenance 和 replay import/export 一致性。evidence
只能消费 current read model，不能反向驱动 runtime。

## 5. P2/P3 明确不能静默跳过

| 域              | 必须逐项决定的 Codex contract                                                                                                                                                                                                                                                            | 允许的 Lime 结果                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Multi-Agent     | root-shared AgentControl、role/model/reasoning/service-tier 约束、max width/depth、execution limiter、rollout budget、nickname/path、V2 residency restore、full/last-N fork、过滤 tool/reasoning/inter-agent side-channel、mailbox `trigger_turn`、delivery phase、steer 优先、wait 三态 | `implemented` 或 `product-scope-excluded`，不能以 graph/identity/mailbox 模块存在代替完成 |
| Process/admin   | `command/exec/*`、`process/spawn                                                                                                                                                                                                                                                         | writeStdin                                                                                | resizePty | kill`、`fs/\*`、file checkpoint、rollback、inject items、config lock、current time、memory mode、attestation | 有真实 App Server handler/read model/evidence，或写出明确 exclusion 和删除未实现声明的条件 |
| Realtime/review | `thread/realtime/*`、review start/notification、audio/text/speech lifecycle                                                                                                                                                                                                              | 产品需要则迁入 current；不需要则在 method matrix 标记排除并移除 catalog/fixture           |
| Clients         | CLI/TUI/SDK/VS Code 都只消费 App Server schema                                                                                                                                                                                                                                           | Lime 可仅实现 Electron/typed client；不得为“对齐 Codex”把客户端逻辑塞进 runtime           |

## 6. 多模型最终取舍（本轮修正）

| 维度                                                           | primary               | secondary | 本轮新增验收                                                                              |
| -------------------------------------------------------------- | --------------------- | --------- | ----------------------------------------------------------------------------------------- |
| catalog/default/selection/switch/child subset/retry-breaker    | Grok Build            | -         | route readiness、credential identity、effective options、unsupported protocol fail closed |
| endpoint/auth/query/header/variant/body/content/media/lowering | Lime `model-provider` | OpenCode  | variant/header/body merge、media capability、repairToolCall/invalid tool、协议穷举        |
| runtime/session/history/Thread/Turn/Item                       | Codex                 | -         | 不把 OpenCode session store 或 Grok MvpAgent 带入 Lime                                    |

Grok 的 `SamplerConfig` 与 Codex `ModelProviderInfo` 提醒我们：provider/model 选择还包括
auth scheme、headers/query、variant、limits、timeouts、stream/request retry、service tier、
tenant/account 和 compaction knobs。OpenCode 的 `Catalog`/`SessionRunnerModel` 补充了
credential metadata、available/default、variant overlay、API/endpoint fail-closed 和
`provider.use` policy。它们都必须落到一个 Rust `model-provider` owner，不能把 AI SDK、
Effect runtime、OpenCode session 或 Grok 独立 agent loop 引进来。

## 7. 最小验证和回流守卫

### P0/P1 验收

```bash
npm run test:contracts
npm run test:rust:related -- lime-rs/crates/app-server-protocol lime-rs/crates/app-server lime-rs/crates/agent-protocol lime-rs/crates/thread-store lime-rs/crates/model-provider
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
npm run bridge:health -- --timeout-ms 120000
```

定向测试必须包含：schema/method matrix、initialize capability、server request correlation、
overload、Thread field round-trip、history builder、rollout repair、compaction lineage、
provider unsupported protocol、provider default recovery、tool repair/approval、MCP snapshot、
transport listener/health，以及真实 Electron restart。

### 回流扫描

```bash
rg -n "agentSession|lime-providers|Custom.*Chat|start_stdio_connection|WebSocket|UnixSocket|replacement_history|tailStartTurnId" electron src scripts lime-rs internal/refactor/v1
npm run governance:legacy-report
npm run test:contracts
```

扫描命中只能出现在本目录历史证据、负向 guard 或明确的产品范围表中。生产路径不得
出现 Agent 旧命令、第二 provider owner、未实现 transport 声明、provider raw-to-Item
映射或 mock fallback。

## 8. 下一刀

下一刀不是扩展 GUI，而是先完成 **P0-01 + P0-05 + P0-06**：把 Agent method registry
迁到 v2、让缺 provider/model 的恢复变成可持久化的 pending route、让未实现 provider
协议在联网前 fail closed。只有这三项通过，compaction/history 和多模型切换的后续证据
才有可信基础。
