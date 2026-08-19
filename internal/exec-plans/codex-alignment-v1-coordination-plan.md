# Codex 对齐 v1 并行协调计划

状态：快速骨架阶段完成（2026-07-28）

完成口径：Codex Agent runtime 主链与 grok-build 多模型 / 多模态控制面骨架已完成；
Fal/Bedrock adapter、真实云 Provider/地区/凭证矩阵，以及
audio/video/file 的逐 Provider chat wire 属于后续细节，不计入本阶段完成度。

## 2026-08-11 P1-01 provider tool-call repair 接线

状态：`completed / current`（provider sampling + JSON Schema repair + 组合竞态 + 跨协议 capture，以及 CodeMode planning、custom-tool contract、standalone process host、CodeCell trace/evidence 与专项 Electron Gate B 均已关闭；P1-01 总项完成）。

主目标：把已经存在于 `tool-runtime` 的 typed repair outcome 接入 current provider sampling
step，确保 provider 返回的 malformed/scalar/unknown tool call 不再提前中止整步，也绝不进入
真实 handler；大小写、current native alias、已知参数别名和 JSON Schema 校验只按本 step 冻结的
definitions 执行。

窄写集：`runtime-core/src/llm_protocol/canonical.rs`、
`model-provider/src/current_client{,/stream,/request_capture_tests}.rs`、`agent-runtime/src/provider_turn{,/tests}.rs`、
`tool-runtime/{Cargo.toml,src/tool_call_surface.rs,src/code_mode.rs,src/tool_definition.rs,src/turn_snapshot.rs,src/lib.rs}`、`lime-rs/Cargo.lock`、本计划、
`internal/aiprompts/architecture.md` 与 `internal/refactor/v1/08-third-audit-gap-register.md`。

唯一数据流：

`provider raw tool call -> canonical LlmEvent(raw_arguments) -> frozen RuntimeToolStepSnapshot(definition + schema + executor) -> repair_tool_call -> deterministic numeric coercion -> JSON Schema validation -> executable call | invalid no-handler lifecycle -> tool result -> next sampling step`

退出条件：repair success 使用 canonical name/arguments 且只执行一次；malformed/scalar/unknown/schema mismatch
统一产生 exactly-one started/completed lifecycle 和一个 model-visible failed tool result，
`handler_executed=false`；坏 schema fail closed；provider error 不再吞掉可恢复调用；repair 后的 cancel/timeout/
late completion 组合不覆盖或重放 canonical terminal。最低验证为四个 Rust owner 的 related/integration、
`smoke:agent-runtime-current-fixture`、治理报告和 diff check。本刀不改 GUI/bridge；Electron Gate B
只作为 Desktop 主链无回归证据，不冒充 schema repair 语义或 live provider 证据。

完成结果：`runtime-core` canonical `ToolCall` 显式保留 `raw_arguments`；`model-provider` 对 malformed
JSON、scalar arguments 和空工具名只保留 raw call，不再把整个 provider step 提前终止。`agent-runtime`
在写入 transcript 前用冻结的 `RuntimeToolStepSnapshot` 调用 `tool-runtime::repair_tool_call`：大小写、current
native/workspace alias 和已知参数别名修复为 canonical call；unknown、空名称、malformed 或 scalar 参数统一
转为带 typed repair metadata 的 `invalid` call。专用 `before_handler` executor 保证 invalid call 不触达真实
handler，同时仍只产生一组 started/completed lifecycle 和一个 model-visible failed result；下一 sampling step
只看到 canonical call/result transcript。

第二刀把 `repair_tool_call` 的 API 直接切换为接收冻结的 `RuntimeToolDefinition`，不保留 names-only
兼容入口。alias normalization 后，只对 schema 明确声明为 `integer`/`number` 的合法 JSON 数值字符串做
递归 deterministic coercion，再使用同一 `input_schema` 完整校验；schema mismatch 与 schema 编译失败都进入
typed `invalid` no-handler lifecycle。validation error 使用 masked instance value，model-visible arguments 和
下一 sampling transcript 都不包含原始参数值；没有新增第二份 schema owner 或自然语言猜测修复。

第三刀在 `agent-runtime::provider_turn` 补齐组合竞态：repair 后的 canonical handler 启动后取消 turn，canonical
executor 立即生成唯一 `aborted` completed 并释放 turn；测试随后显式释放 handler 自有后台工作，证明迟到 success
无法覆盖终态、追加 lifecycle 或触发第二次 provider sampling。另一场景让 repaired tool 正常完成后挂起下一 provider
step 并触发 absolute timeout，证明 handler 只执行一次，canonical call/result transcript 只进入该 request，
lifecycle 不重放。该刀只增加 current owner 跨层回归，不新增生产分支或第二终态 owner。

第四刀补齐全部 current chat transport 的 request-capture 矩阵。既有 Gemini、Vertex、Azure Responses 与 Ollama
Responses 已从真实 loopback 捕获 path/auth/body 并消费 native SSE；Responses WebSocket 已捕获 handshake、
`response.create` 与 HTTP replay。新增独立 `request_capture_tests`，从 `CurrentProviderClient::stream` 分别进入
OpenAI Chat Completions、OpenAI Responses HTTP 与 Anthropic Messages，联合断言 canonical system/user/image、
repaired tool call/result transcript、tool schema、generation lowering、native auth/header、exact endpoint 与 SSE
terminal。Anthropic 只断言服务端真实返回的 input/output usage，不伪造 `total_tokens`。该刀不新增 transport、
protocol alias 或生产 capture owner；Bedrock/Fal 继续在发网前 fail closed。

第五刀完成 CodeMode 的 planning foundation，但不伪造可执行能力。`tool-runtime` 的 exposure 直接切换为 Codex
六态 `Direct / Deferred / DeferredModelOnly / DirectModelOnly / CodeModeOnly / Hidden`，并新增唯一
`RuntimeToolMode::{Direct, CodeMode, CodeModeOnly}` 决策和 frozen tool plan。规划器精确区分 direct model surface、
searchable surface 与 nested CodeMode surface，复用 Codex namespace 拼接、JavaScript identifier normalization、
`exec`/`wait` 保留名和 normalization collision first-winner 语义；不可用的普通 CodeMode 只在明确允许时回退
Direct，CodeModeOnly 或禁用 fallback 时直接 fail closed。当前 `model-provider` 仍只有跨协议 function-tool canonical
contract，仓库也没有 CodeMode session/runtime host，因此本刀不向 provider 曝光假的 `exec`/`wait`，不新增 V8/JS
依赖，也不复制 TUI warning 或远程 host。

验证结果：第一刀 `tool-runtime tool_call_surface` `12/12`、`model-provider --lib` `236/236`、
`agent-runtime provider_turn::tests` `43/43` 通过；第二刀 `tool-runtime tool_call_surface` `15/15`，完整
`tool-runtime 320/320`、`agent-runtime 193/193` 与 provider-turn `43/43` 通过。第三刀两个新增组合回归
定向 `3/3`、完整 provider-turn `45/45` 与完整 `agent-runtime 195/195` 通过；并行首轮曾由测试自身 `1s`
外层防死锁超时先于业务 `20ms` deadline 被调度，放宽为不改变生产语义的 `5s` 测试护栏后完整回归稳定通过。
`npm run test:rust:related -- lime-rs/crates/agent-runtime/src/provider_turn/tests.rs` 扩展到 `agent-runtime`、
`app-server`、`lime-agent`、`lime-scheduler`、`lime-server` 五个 current/反向依赖 crate 并退出 `0`；仅有一条
既有 `app-server` 测试 helper 的 dead-code warning，与本刀无关。
第四刀 request capture 定向 `3/3`、完整 `model-provider 239/239` 通过，覆盖新增 Chat/Responses HTTP/Anthropic
与既有 Gemini/Vertex/Azure/Ollama/Responses WebSocket/reducer 全矩阵；未调用 live provider 或读取真实凭证。
`npm run test:rust:related -- lime-rs/crates/model-provider/src/current_client.rs lime-rs/crates/model-provider/src/current_client/request_capture_tests.rs`
扩展到 `model-provider` 与 12 个反向依赖 crate 并退出 `0`；其中 `agent-runtime 195/195`、
`model-provider 239/239`、`tool-runtime 320/320` 全绿，仍只有一条既有 `app-server` 测试 helper dead-code warning。
第五刀新增 CodeMode planning 定向测试 `3/3`、snapshot 回归 `1/1` 与完整 `tool-runtime 323/323`
通过。`npm run test:rust:related -- lime-rs/crates/tool-runtime/src/{code_mode.rs,tool_definition.rs,turn_snapshot.rs,lib.rs}`
扩展到 `agent-runtime`、`app-server`、`lime-agent`、`lime-mcp`、`lime-scheduler`、`lime-server`、
`tool-runtime` 七个 current/反向依赖 crate 并退出 `0`，其中 `agent-runtime 195/195`、`tool-runtime 323/323`；
仅有一条既有 `app-server` 测试 helper dead-code warning，与本刀无关。
`npm run test:rust:related -- lime-rs/crates/{runtime-core,model-provider,tool-runtime,agent-runtime}` 对扩展后的
14 个 current/反向依赖 crate 退出码为 `0`，其中 `agent-runtime 193/193`、`tool-runtime 317/317`。
第二刀 `npm run test:rust:related -- lime-rs/crates/{tool-runtime,agent-runtime}` 扩展到 7 个 current/反向依赖
crate 并退出 `0`；其中新增依赖后的 `agent-runtime 193/193`、`tool-runtime 320/320` 全绿。
`npm run test:contracts` 通过，生成 `959` 个 protocol types 且零漂移，App Server client contract
`301` checks 全绿。`npm run governance:legacy-report` 扫描 `2120` 个 current 文件，零引用候选、分类漂移和
边界违规均为 `0`；`cargo fmt --all -- --check` 与 `git diff --check` 通过。

`npm run smoke:agent-runtime-current-fixture` 最终完整退出 `0`，覆盖真实 Electron Desktop Host、preload/IPC、
`app_server_handle_json_lines`、App Server/runtime/read model 和 GUI，且 `liveProviderUsed=false`。首轮在
Expert Panel Skills Runtime 场景的失败已确认是另一聚合进程使用同一静态 prefix/backend ledger 造成证据串线：
失败摘要的外层 session 与被轮询 read-model thread identity 不同；使用唯一 prefix 的专项复跑退出 `0`，待并发
进程结束后的完整聚合复跑同样退出 `0`。因此 Gate B 只作为 Desktop 主链无回归证据，不冒充 repair 语义或 live
provider 证据。

第二刀在无同类聚合进程并发的条件下再次完整运行该 Gate B 并直接退出 `0`；unknown Item、首页热路径、
Coding Workbench、图片命令、cancel/continue、approval、Plan、Skills Runtime、MCP structuredContent、media
reference、Expert Plaza/Panel、typed error success/failure 与 Content Factory 均通过，`liveProviderUsed=false`。

分类：canonical raw call、冻结 schema repair、typed invalid no-handler lifecycle 与回归证据均为 `current`；此前
provider malformed call 直接终止整个 step、以及未按冻结 schema 校验便执行 handler 的行为均为
`dead / deleted / forbidden-to-restore`；无新增 `compat` 或 `deprecated`。前四刀关闭 provider sampling、
JSON Schema repair、repair 后 cancel/timeout/late completion 的组合竞态与全部 current chat transport 的
request capture；CodeMode exposure/mode/plan、provider freeform/custom-tool canonical contract、
catalog/readiness gate、transport-neutral session contract 与测试注入下的 Agent loop `exec/wait` boundary 为
`current foundation`；canonical thread-owned lazy service、actor identity/interrupt/shutdown owner 同属
`current foundation`；per-cell nested dispatch、outer `exec/wait` canonical Tool lifecycle 与 notify Desktop event
projection 也同属 `current foundation`。该阶段保留的 standalone CodeMode host OS 进程隔离、thread-owned
CodeCell trace/evidence owner、CodeMode 专项 Electron Gate B 与 terminal 组合证据，已分别由 2026-08-12
standalone process host 和 2026-08-18 CodeCell trace/evidence 切片关闭；P1-01 不再保留 alignment-open 项。

本文件是并行执行的协调面，不定义新的 runtime owner。目标是让多个进程同时推进
Codex v2 对齐时保持窄写集、可编译、可回滚和可验证。所有实现仍服从：

```text
Electron Desktop Host
  -> App Server JSON-RPC v2
  -> RuntimeCore / agent-runtime
  -> Thread/Turn/Item + ThreadStore
  -> model-provider / tool-runtime
```

多模型只归 `model-provider`；grok-build 是控制平面主参考，OpenCode 只补 provider wire；
`agentSession/*`、`protocol/v0`、`lime-providers` 和第二 history owner 不允许成为新代码落点。

## 0. Codex 迁移原则

### 2026-08-10 Config Control Plane 收口

主目标：将 Codex `config/read`、`config/value/write`、`config/batchWrite` 接入 Lime Desktop 唯一 App Server
配置主链，并把 Settings/Claw fixture/evidence 从旧 Electron `get_config/save_config` 迁出；`configRequirements/read`
按无 MDM/requirements owner 的产品范围裁决为 excluded。

窄写集：`scripts/electron/mcp-config-fixture-smoke.mjs`、Settings/Claw fixture 与 evidence、
`scripts/README.md`、`internal/aiprompts/{commands,architecture}.md`、产品范围矩阵/fixture。

唯一数据流：

`Desktop Settings/fixture -> app_server_handle_json_lines -> config/read|config/value/write|config/batchWrite -> lime_core config.yaml`

退出条件：旧配置命令不再出现在正向生产/fixture调用；证据只记录 App Server method；单一用户层、版本冲突、未知 key、
project-local/非当前 filePath fail closed；矩阵统计同步为 `139 implemented / 42 planned / 39 product-scope-excluded`，产品范围完成度为
`139 / 181 = 76.8%`。

验证：八组 Settings evidence 定向 Vitest `32/32`，ASR provider 配置迁移测试 `7/7`，产品矩阵守卫 `4/4`，所有迁移脚本
`node --check`，`npm run test:contracts`（301 checks）与 `npm run governance:legacy-report`（0 分类漂移、0 边界违规）通过；
`npm run smoke:agent-runtime-current-fixture` 已通过且 `liveProviderUsed=false`。剩余旁路：Electron crash diagnostics 与
desktop-host mock/retired tests 仍保留历史 `get_config/save_config` 文本，只能作为负向 retired guard，不能作为 current surface。

凡是 Codex 已有且不依赖 ChatGPT-only 产品的能力，优先直接复制对应 Rust 模块、类型、
状态机和测试，再做 Lime 的 credential、Electron、产品范围适配；不先手写一个“相似版本”。
参考路径必须记录到车道回报中，例如：

- 协议/Item/history：`codex-rs/app-server-protocol/src/protocol/v2/**`、
  `protocol/thread_history.rs`、`thread-store/README.md`；
- transport：`codex-rs/app-server-transport/src/transport/{stdio,websocket,unix_socket}.rs`
  及其 tests；
- compaction/recovery：`core/src/compact.rs`、`core/src/session/rollout_reconstruction.rs`；
- tools/hooks/MCP/AgentControl：Codex 对应 registry、hook runtime、MCP snapshot、
  `agent/control*` 和 multi-agent V2 tests；
- model control：grok-build；provider wire：OpenCode，均必须翻译到 Lime 唯一
  `model-provider` owner，不能复制其 session/UI/runtime。

“已参考”只有在 Lime 对应定向测试通过后才算有效；文件名或 enum 相似不算 parity。

## 1. 并行车道

每个进程只能修改自己的窄写集。跨车道文件先发协调消息，由 `/root` 统一合并；不要直接
覆盖其他车道未提交的工作树改动。

| 车道   | 当前进程/代理                              | 窄写集                                                                                                                                  | 目标                                                                                                                                         | 依赖                                                                                                                                                                                                                                                  | 状态                         |
| ------ | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| A 协议 | `/root`（协议车道已汇合）                  | `lime-rs/crates/app-server-protocol/src/protocol/**`、协议 schema/fixture、Electron `appServerHost` 与生产 gateway 中的核心 method 常量 | `thread/start/read/list/resume`、`turn/start/interrupt/steer` wire v2；旧核心 `agentSession` 不再接收                                        | 六类 direct v2 lifecycle/delta 与唯一 projector 已接，v0 lifecycle wrapper 已删除；当前为 748 schema definitions / 740 TS types / 0 漂移，archive/unarchive 已实现；fork/delete、typed Item/server request 与 loaded-thread live recovery 仍 OPEN_REF | merged-with-open-refs        |
| B 路由 | `/root/provider_failclosed_slice` -> B2/B3 | Rust route contract + TypeScript current client/gateway/Electron Plugin facade（不改 Rust 协议/transport）                              | unknown capability/model/direct config 缺 snapshot 返回 typed `RouteFailure`；production gateway/Plugin facade/contract guard 改用 v2 method | server/services/skills/image 已迁到 `model-provider`；旧 crate、workspace/Cargo.lock 和正向引用已删除；owner guard 2/2、治理 0 违规，source-built App Server 已用于 Gate B；workspace 全量门禁仍待独立收尾证据                                        | done-with-workspace-open-ref |
| C 传输 | `/root/transport_codex_slice`              | `lime-rs/crates/app-server-transport/**`、App Server transport wiring/transport tests                                                   | 对照 Codex 实现 ws/unix/stdio bounded/slow-client/filtering                                                                                  | WS/Unix/stdio lifecycle、bounded/slow-client、filtering 17/17；Unix parent 0700 与 canonical startup-lock 单层路径已修，source-built App Server sidecar 已用于 current home-hotpath Gate B 68/68；Windows cross-platform UDS 仍 OPEN_REF              | done-with-open-refs          |
| D 协调 | `/root`                                    | `internal/refactor/v1/**`、`internal/exec-plans/**`、`internal/aiprompts/**`、跨车道集成/验证                                           | 更新方案、消除冲突、运行全局扫描、组织删除顺序和 Gate A/B                                                                                    | A/B/C 骨架已汇合，current home-hotpath Gate B 68/68；继续收敛 typed Item/server request、durable route、history/recovery 与全量门禁                                                                                                                   | 进行中                       |

### 1.1 loaded resume 下一并行切片

先完成三个互斥骨架，再由 `/root` 单点汇合；不得让多个进程同时修改 `app-server/src/lib.rs`：

| 子车道                   | 窄写集                                                              | 本轮退出条件                                                                                                                        | 依赖/禁止项                                             |
| ------------------------ | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| R1 connection context    | `app-server/src/lib.rs`、`processor/{mod,dispatch}.rs` 及其定向测试 | 非 initialize transport request 在 spawn 后仍携带 `ConnectionId + RequestId`；response/follow-up 使用同一 connection outgoing owner | 不实现 thread subscription，不改 `server_request.rs`    |
| R2 pending replay source | `app-server/src/server_request.rs` 及其单测                         | pending route 保存原始 request、thread identity 与 owner；提供按 thread/owner 稳定排序的只读 snapshot，resolve/drop 仍 exactly-once | 不发送 replay，不改 transport 或 processor              |
| R3 thread listener state | 新增 App Server `thread_state.rs`/对应单测                          | per-thread listener command channel、connection set、generation 与旧 listener 退出保护可独立验证                                    | 不接 event pump，不改 `lib.rs`、RuntimeCore 或 Renderer |

汇合切片由 `/root` 唯一修改 event pump/outgoing：将 live `AgentEvent` 路由到 per-thread
listener，并在同一 listener 内执行 `active snapshot -> subscribe -> resume response -> token
usage -> goal -> pending requests -> live event`。完成前，全局 broadcast、Electron recent
buffer 和 `RpcDispatch.notifications` 都不得被当作 resume 排序 owner。最终 Gate 必须逐条读取
raw JSONL，禁止用会跳过并缓存消息的 helper，严格断言 response/replay/live 顺序、disconnect
清理和 listener generation。

### 写集硬规则

1. 车道 A 不修改 `model-provider`、ThreadStore、runtime route；车道 B 不修改协议/transport；车道 C 不修改协议方法或 provider route。
2. 车道 D 不直接重写 A/B/C 的实现文件；只更新协调文档、架构事实源和验证记录。
3. 任何需要跨写集的删除（`protocol/v0`、`lime-providers`、旧 schema、workspace member）必须先完成引用扫描，再由 `/root` 发起单独删除刀。
4. 生成文件只能由拥有对应 schema/生成脚本的车道更新；禁止多车道同时运行生成器。
5. 不执行 `git commit`、`git push`、`git reset` 或创建分支；共享工作树中的用户既有改动一律保留。

## 2. 状态协议

每个车道完成一个切片后向 `/root` 回报以下固定格式：

```text
LANE: A|B|C|D
SLICE: V1-xx
STATUS: done|blocked|needs-integration
WRITE_SET: <实际修改路径>
REMOVED: <删除的 surface；无则 none>
CHECKS: <执行的命令与结果>
OPEN_REFS: <仍需迁移的引用；无则 none>
CONFLICTS: <与其他车道冲突；无则 none>
NEXT: <下一步>
```

`done` 只表示本车道定向测试通过，不代表整体 Codex 对齐完成；整体完成由 Gate A/B 和
第 6 节的全局守卫决定。

## 3. 依赖与交接顺序

```text
A: v2 wire core
  ├──> A-integrate: Host/preload/gateway/catalog/schema fixtures
  ├──> D: 删除 agentSession/v0 的生产引用
  └──> Gate A

B: route fail-closed
  ├──> agent control restart/recovery
  ├──> provider catalog/auxiliary route
  └──> Gate B model/restart evidence

C: transport bounded/connection semantics
  └──> Gate A/B initialize, server request, overload, reconnect evidence

A + B + C
  └──> D: 删除双轨 owner -> workspace compile -> full contracts -> GUI/Gate B
```

交接前检查：

1. 车道只提交（在共享工作树中表现为修改）自己的写集，先运行定向 `cargo fmt --check` 或等价格式检查。
2. 回报中列出未迁移引用；`/root` 不得在未读清单前删除目录。
3. `/root` 合并时按文件逐个检查 `git diff`，不使用 destructive checkout/reset 覆盖其他改动。
4. 任何编译失败先归类为车道回归、共享基线失败或用户既有脏改动，不通过猜测修改其他车道。

## 4. 删除协调闸门

### Gate D1：核心协议切换

满足以下条件才能删除 `agentSession` 核心写路径：

- A 回报 v2 method/notification/server request schema 与 Rust 定向测试通过；
- Electron Host、preload、Renderer gateway、catalog、fixture 已切换；
- `rg -n "agentSession/(start|read|list|thread/resume|turn/start|turn/cancel)"` 只剩删除 guard、历史 evidence 或待迁移清单；
- `npm run test:contracts` 通过。

这只关闭 Agent 核心写路径，不等于可以删除包含 Lime 自有 workspace/browser/media/voice
方法的整个 v0 module。其他方法必须先登记产品范围并迁移到 v2/current registry；全部迁完
后才能执行最终 v0 module 删除刀。

### Gate D2：provider owner 收敛

满足以下条件才能删除 `lime-providers`：

- server/services/skills/image 等所有直接消费者已使用 `model-provider` current client/lowering/stream；
- workspace/Cargo.lock、catalog、内部文档和测试无 `lime-providers` 正向引用；
- model route unknown capability/model、unsupported protocol 和 credential readiness 均 fail closed；
- `cargo check --workspace` 与 provider 相关 crate 测试通过。

### Gate D3：transport 声明与实现一致

- ws/unix/off 有真实 acceptor、逐连接 initialized/experimental/opt-out、bounded outbound 和 slow-client 处理；或
- 产品范围明确只支持 stdio，并同步删除 enum/URL/fixture/文档中的未实现声明；
- transport tests 覆盖 overload、disconnect、server request、reconnect。

## 5. 统一验证门禁

每个 checkpoint 先跑最近边界的定向检查；所有车道汇合后按顺序运行：

```bash
git diff --check
npm run governance:legacy-report
npm run test:contracts
npm run test:rust:related -- lime-rs/crates/app-server-protocol lime-rs/crates/app-server lime-rs/crates/agent-protocol lime-rs/crates/thread-store lime-rs/crates/model-provider
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
npm run bridge:health -- --timeout-ms 120000
```

回流扫描：

```bash
rg -n "agentSession|lime-providers|crate = \"providers\"|Custom.*Chat|protocol/v0" electron src scripts lime-rs internal
```

扫描命中只能出现在本协调计划、v1 缺口登记、历史 research/evidence 或明确负向 guard；
production current path 命中即阻塞，不得以“兼容”关闭告警。

## 6. 当前协调记录

| 时间       | 事件                                         | 决策/动作                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-07-18 | 第二轮审计完成                               | 新增 `internal/refactor/v1/07-second-audit-gap-register.md`，将 v2 protocol、ThreadStore、compaction、transport、provider duplicate owner 升级为 P0/P1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-07-18 | 第三轮审计文件出现                           | 以 `internal/refactor/v1/08-third-audit-gap-register.md` 作为补充证据；不得把它当新 owner                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-07-18 | 用户授权删除和无兼容重构                     | 允许在 D1/D2/D3 通过后物理删除 `protocol/v0`、核心 `agentSession`、`lime-providers` 和未实现 transport 声明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2026-07-18 | 三条代码车道启动                             | A 协议、B 路由、C 传输互不覆盖写集；D 负责协调和最终验证                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-07-19 | D 车道 V1-03 preflight                       | 只读对照 Codex `thread-store`/`ThreadHistoryBuilder` 后确认 Lime 当前 `ThreadStore` 仍只有 `apply_history`，compaction 仍是 `summary + tailStartTurnId`；未抢占 A/B/C 热区，等待联合 handoff                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-07-19 | D 车道 compaction lineage slice              | `context_compaction.rs` 写入 `replacementHistory`/window lineage，`provider_history.rs` 消费 typed summary replacement 并对 legacy marker fail-safe；定向 `context_compaction` 4/4、`provider_history` 14/14、`context_auto_compaction` 4/4、`sessions` 20/20；状态为 `partial`，仍缺完整 canonical `ResponseItem` history、UUIDv7 window state、rollback/fork/replay evidence                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 2026-07-19 | D 车道统一回查                               | `git diff --check`、定向 rustfmt、`governance:legacy-report` 通过；`npm run test:contracts` 在 `check:protocol-types` 因 A 车道生成文件尚未汇合而阻塞，需 A handoff 后重跑                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 2026-07-19 | D 车道 V1-04 evidence/provider + dead DAO    | `evidence_provider` 导出 `canonical-provenance.v1`（canonical item、provider attempt/route、usage/retry、credential fingerprint、replay digest）；`model-provider`/`lime-agent` 对 unsupported protocol、credential/readiness 继续 fail closed；删除零引用 `core::database::dao::A2UIFormDao` 及 export，保留 retired schema cleanup。Evidence 1+7、model-provider 30、lime-agent 6、lime-core 675 全部通过，治理/格式/差异扫描通过。该 checkpoint 当时的 OPEN_REF 为 A 的 v2 schema/client/gateway、B 的 PendingRoute/ThreadStore/compaction/lime-providers 收敛、C 的 transport Gate、Gate A/B；后续 current home-hotpath Gate B 已关闭，但仍不宣称 v1 完成                                                                                                                                             |
| 2026-07-18 | D 车道 V1-05 route fail-closed               | `runtime-core` 未知 provider type/name 不再默认映射 `OpenaiChat`，改为 `ProtocolKind::Unknown`；Chat 任务的 Gemini/Ollama/Bedrock/Fal/Vertex 等非当前 wire adapter 协议也生成 `RouteFailureCategory::UnsupportedProtocol`，图片任务仍交给专用 lowering；`runtime-core` model_route 12/12 通过。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-07-18 | D 车道 V1-06 restart recovery                | AgentControl recovery 现在按“显式 runtime request -> durable session provider/model defaults”补全 route；`runtime_options: Some` 但缺 provider/model 也会 deferred。新增 session-default merge 测试，restart recovery 11/11 通过；App Server warmup 不再因缺 route 直接失败。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-07-18 | D 车道 V1-06 startup guard                   | App Server main 对明确的 provider/model selection 缺失错误只记录 deferred warning，其余 recovery error 仍为 fatal；`RuntimeCoreError::is_provider_selection_required` 提供稳定分类。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-07-18 | B 车道 V1-08/V1-11 endpoint slice            | provider agent 将 image API 的 Responses endpoint builder 迁入 `model-provider::current_client::responses_endpoint`，移除该调用点对 `lime-providers` 的依赖；server image tests 45/45、owner boundary 1/1 通过。剩余 server/services/skills consumers 继续 OPEN_REF。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-07-18 | C 车道 V1-13 transport follow-up             | WebSocket/Unix socket 广播改为 per-connection bounded `try_send`，慢客户端队满断开；增加 initialized、ping/pong、close/reconnect 和 App Server slow-client 单测。Transport 17/17 通过；`optOutNotificationMethods` 协议字段仍未伪造，交给 A/D。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-07-19 | 用户再次确认“该从 Codex 复制就复制”          | 协议车道不得把 Codex v2 类型继续扩展到 `protocol/v0`；current owner 必须建立 `app-server-protocol/src/protocol/v2/**`，按 Codex v2 的请求、响应、Thread/Turn/Item tagged union、server request/notification 和测试复制，再做 Lime 依赖边界适配。无法直接引入的 Codex core 类型只能在 v2 边界用等价 wire 类型，不得改回 v0 或新增第二套 envelope。                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-07-19 | P0-05 重启恢复切片启动                       | App Server 恢复必须优先使用持久化 provider/model/config/route；显式 runtime options 次之；两者缺失时返回 typed `PendingRoute`，保留 graph/mailbox、进程继续存活且不 ack/terminal。恢复切片独占 `lime-rs/crates/app-server/src/runtime/**`，不修改协议车道和 provider lowering。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-07-19 | D 车道 V1-P0-05 typed PendingRoute handoff   | `RuntimeCoreError::PendingRoute`、retryable JSON-RPC data、mailbox deferred admission 和 restart regression 已落入 runtime owner；状态为 `needs-integration`。`rustfmt --check`、`git diff --check` 通过；定向 Cargo 被邻接 provider owner 清理残留的 `lime-services` `lime-providers` 测试引用阻塞。OPEN_REF：providerConfig/协议/credential generation 持久化、已有 route 不可执行时的 fail-closed、catalog 变更重试。                                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-07-19 | D 车道 compaction lineage follow-up          | 按 Codex `compact.rs`/`rollout_reconstruction.rs` 继续收敛：新窗口使用 UUIDv7；replacement history 保留压缩前用户边界并追加最终摘要 user message；nested/top-level compaction marker 字段回退，durable event history 仍不删除。补充窗口 UUID、replacement boundary 和 lineage 测试；`rustfmt`/`git diff --check` 通过。`cargo test -p app-server context_compaction --lib` 尚未到目标测试，因邻接 `lime-services` 残留 `lime-providers` 引用/删除中的 provider compile errors 阻塞。                                                                                                                                                                                                                                                                                                                      |
| 2026-07-19 | D 车道 V1-02 canonical append slice          | `thread-store::ThreadStore::append_items` 新增 canonical `ThreadItem` append params；ProjectionStore 复用 history sequence/fingerprint 事务但跳过 thread metadata/turn snapshot refresh。补 exact retry、sequence collision、metadata unchanged、identity/FK 原子失败测试；`cargo test -p thread-store --lib` 19/19。该 slice 仍不是 Codex raw `RolloutItem` writer，live rollout/replay materializer 继续 OPEN_REF。                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-07-19 | D 车道 P0-05 route snapshot follow-up        | child route metadata 记录 providerConfig 非敏感字段、`apiKeyPresent` 和已有 `credentialRef`/`routeProtocol`/`effectiveGeneration`；恢复时显式 request 优先，持久化快照不回填 api_key。`agent_control.rs` 已拆出 `runtime/agent_control/route.rs`（主文件约 697 行）；待 app-server 依赖汇合后验证。OPEN_REF 仍是 typed route/credential owner、不可执行 route fail-closed 和 catalog/credential deterministic retry。                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-07-19 | D 车道协议/transport 缓存验证                | 当前无 `cargo`/`rustc` 进程；最新测试二进制晚于对应源码，直接执行 `app_server_protocol-1954511f58102ba1` 57/57、`app_server_transport-b1234f8fc8e19120` 17/17 通过。该证据只证明最近一次已构建产物，不替代 workspace 恢复后的源码重编译。v2 仍缺 typed client/server envelope、完整 catalog/dispatch 和 TS/schema 收敛；`turn/steer` 尚未接入生产 dispatcher，D1 保持未通过。根目录误生成的未跟踪 `--help/` schema tree 不计入 fixture evidence。                                                                                                                                                                                                                                                                                                                                                         |
| 2026-07-19 | D 车道 ThreadHistoryBuilder 候选审阅         | 未跟踪 `thread-store/src/history.rs`/`history_tests.rs` 虽被 `lib.rs` 接入，但生产消费者为 0，当前仍由 App Server `thread_item_projection` reducer 承接；候选实现还存在 Turn/flat items 分叉、same-sequence 新 turn 放行、item identity 与 store PK 不一致、merge 后相等误判 exact retry、outer/item sequence 未校验等问题。分类为 `current-candidate / rejected-for-integration`，不得作为 V1-02/P0-03 完成证据；后续必须选定唯一 reducer，完成 builder -> store -> cold/live/replay round-trip 后再迁出重复实现。                                                                                                                                                                                                                                                                                       |
| 2026-07-19 | D2 workspace 中间态阻塞                      | `lime-providers` 源码/manifest 已删除且消费者已迁到 `model-provider`，但磁盘仍有空 `crates/providers/` 目录，`members = ["crates/*"]` 因而加载缺失 manifest 失败；`Cargo.lock` 另有孤立 package block。分类为 `dead / deletion-incomplete / forbidden-to-restore`。不得补 dummy manifest 或 compat crate；清空目录和 prune lock 后必须重跑 Cargo/owner guard。删除空目录等待危险操作明确确认。                                                                                                                                                                                                                                                                                                                                                                                                            |
| 2026-07-19 | A 车道 schema options slice                  | `schema_export.rs` 将 v2 schema 文件导出绑定到 `include_protocol_types`，不再错误跟随 `include_envelopes`；新增 FF/FT/TF/TT 四组合 bundle/tree 回归。`cargo test -p app-server-protocol --lib schema_export` 4/4、定向 rustfmt、`git diff --check` 通过；未运行 codegen。OPEN_REF：typed v2 envelopes/catalog/dispatch、TS/schema fixture 全量 contract。                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-07-19 | A 车道 schema 名称冲突审计                   | v0/v2 共享 8 个 `Thread*List/Read/Turns/Items Params/Response` 名称；bundle `$defs` 是平面 map，当前 v2 插入会覆盖 v0（例如 v0 `turnsView` 与 v2 `includeTurns`），而独立 `json/v0`/`json/v2` fixture 仍各自不同。分类为协议迁移阻塞，交 A owner 决定 namespace 或先完成 v0 production 切换；D 不擅自改名、删 v0 或重跑 codegen。                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-07-19 | D 车道 append sequence 语义审计              | `Materializer` 将每个 item 的 `sequence` 设为原始 event 序号，而批次 `ThreadHistoryChangeSet.sequence` 取最新事件，因此合法批次允许 item sequence 小于 outer sequence；当前 `validate_change_set`/`validate_append_items` 未拒绝 item sequence 晚于 outer sequence。候选 history 测试已构造该非法形状（item=3, outer=2），后续应在 canonical store validator 加 fail-closed 回归；当前避让 App Server 热区，未修改。                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-07-19 | D2 workspace 清理汇合                        | 隔壁 D2 车道已物理清空 `crates/providers/` 残留空目录并从 `Cargo.lock` 移除孤立 `lime-providers` package；未恢复 dead crate、未新增 compat。随后 `cargo test -p thread-store --lib` 源码重编译 26/26 通过。仍需 `cargo metadata --locked`、workspace/provider owner guard 和全量 related 验证。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-07-19 | D2 交接门禁复核                              | 本进程只读运行 `cargo metadata --locked --no-deps --format-version 1` 成功；`npx vitest run scripts/lib/model-provider-owner-boundary.test.ts` 2/2 通过。D2 结构残留可进入 workspace related 验证；该 checkpoint 当时协议 v2 typed envelope/catalog/dispatch、TS contract、Gate A/B 仍为 OPEN_REF，后续 current home-hotpath Gate B 见 68/68 definitive 记录。                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 2026-07-19 | D 车道 canonical projection fixture 复核     | 隔壁新增的 `appServerCanonicalThreadProjection.test.ts` 定向 Vitest 2/2 通过，覆盖 Codex Unix 秒、`notLoaded`、Turn 运行/失败、Item lifecycle、image/localImage 与 MCP 结构化结果。该 checkpoint 证据仅属于 Renderer projection，当时 Rust v2 handler、schema/client 和真实 Electron Gate A/B 未闭合；后续 current home-hotpath Gate B 见 68/68 definitive 记录。                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-07-19 | D2 provider owner 删除完成                   | server/services/skills/image 消费者统一到 `model-provider::CurrentProviderClient`；`lime-providers` crate、workspace 依赖、Cargo.lock package 和正向引用已删除，导出文件名改为 `provider-config-*`。UI/catalog 218/218、owner guard 2/2、治理 0 违规、metadata/ESLint/Prettier 通过；active 架构文档改为 `dead / deleted / forbidden-to-restore`。全量 workspace 编译仍需等待无关 `agent-runtime/session_loop.rs` 热区汇合。                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-07-19 | D1 `turn/steer` 原子切片                     | 对照 Codex `turn_steer_inner`：RuntimeCore 新增 thread-owned 原子 steer API，session loop 校验 expected active turn，失败不创建回合；typed processor 严格降低 v2 UserInput，成功写入带 `clientId` 的 `message.created` 且不发 `turn/started`。隔离 Cargo 已越过损坏全局 registry，但源码重编译先被隔壁 `agent-runtime/session_loop.rs` 生命周期定义缺失阻塞，尚不记 done。                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 2026-07-19 | D3 private Unix/control-lock follow-up       | transport 17/17 基线通过后补 Codex owner-only 0700 parent 语义，并修默认 socket 导致 `app-server-control/app-server-control` 双层 startup lock 的路径错误；该 checkpoint 当时的 Electron Gate B OPEN_REF 已由后续 source-built sidecar 的 current home-hotpath 68/68 关闭。Windows cross-platform UDS 继续 OPEN_REF。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-07-19 | D1 `steerTurn` typed client skeleton         | `packages/app-server-client` 已把生成的 v2 `TurnSteerParams/TurnSteerResponse` 接入 request client、Connection 和 AgentRuntime facade；typecheck、package 67/67、`check:protocol-types`、client contract 291 checks 与 diff check 通过。App Server 仍是 v0 raw envelope，thread/turn 其余 v2 handler、Renderer/plugin consumer 和 direct notification 继续 OPEN_REF；不关闭 D1。                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-07-19 | D 车道 canonical session projection skeleton | Renderer session client 统一走 `thread/list`、`thread/read`；paginated history 按 Codex 约束走 `thread/turns/list(itemsView: summary)` + `thread/items/list` 分页合并，补 thread/turn/item identity、legacy 空 turns 回读、Unix 秒和 URI 图片投影回归；定向 Vitest 15/15、diff check 通过。Projection 文件接近 900 行，hydrator 对相对/UNC local path 仍 OPEN_REF。                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-07-19 | D 车道 canonical alias 漂移审计              | 发现 `appServerClientMethodSpecs` 已把 `startSession/readSession` 映射到 `thread/start/read`，但 payload/response 类型和 `appServerReadModelClient`、`threadClient`、`executionRun`、`themeContextSearch` 仍使用旧字段；`threadClient` 仍发送 `turnsView`。分类为 D1 P0 OPEN_REF，必须同步 canonical v2 method、类型、消费者和 fixture 后再删旧 alias，不以局部 session client 测试关闭。                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-07-19 | V1 快速骨架汇合（history/steer/provider）    | `ThreadHistoryBuilder` 已接入 canonical store 的 replay/persisted normalize 边界，`thread-store` `cargo check` 通过、history 专项 9/9；v2 `turn/steer` 已接 processor dispatch，Text/Image/LocalImage lower、unsupported input fail-closed、active-turn 原子校验和 `message.created` 通知，定向 rustfmt/diff-check 通过但 Cargo 仍待共享 App Server 依赖完成；provider_calls 已切到 `model-provider::CurrentProviderClient::stream`，OpenAI/Anthropic canonical events 采用增量 SSE lowering，provider Cargo 正在独立 target 编译。状态：needs-integration，不宣称 D1/D2 完成。                                                                                                                                                                                                                           |
| 2026-07-19 | D2 provider streaming skeleton 验证          | 独立 target `cargo test -p lime-server handlers::provider_calls --lib` 7/7 通过；覆盖 OpenAI/Anthropic 增量、tool block 顺序、terminal/error、客户端 drop 传播和 Gemini fail-closed。`rustfmt --check`、窄写集 `git diff --check`、model-provider owner boundary 2/2 通过；完整 server/websocket 旧调用点仍 OPEN_REF。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-07-19 | D1 `turn/steer` skeleton 最终验证            | 修复 steer user message 复用初始 completed Item 的 lifecycle 冲突：每次 steer 以 `clientUserMessageId` 或 UUID 建立独立 canonical `itemId`，保留 `item.started -> message.created -> item.completed`。完整缓存下 `cargo test -p app-server steer_ --lib` 7/7 通过；默认 Cargo cache 的 `tracing-subscriber` 缺源文件属于本机缓存问题。D1 仍因其余 thread/turn v2 handler 与 Renderer alias OPEN_REF 未关闭。                                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-07-19 | D 车道 Renderer canonical consumer skeleton  | `executionRun` 改用 `thread/list` + `thread/read(includeTurns:true)`，按 `Thread.id`、Unix 秒、`Turn.id/status` 投影；`appServerReadModelClient` 改用 canonical Thread projection 并对非法 Turn fail-closed；`themeContextSearch` 改为 `thread/start -> turn/start(threadId,input[]) -> thread/read`；AgentOp lowering 改为 v2 `threadId + UserInput[]`；创建入口只接受 `thread/start.result.thread`。定向 Vitest：agentProtocol 28/28、executionRun 7/7、themeContextSearch 4/4、read-model/projection 6/6、session client 13/13；Prettier 与 `git diff --check` 通过。该 checkpoint 当时的 OPEN_REF 包含 Rust handler、server notification、旧 session consumer、package aliases 与 Gate A/B；后续 direct lifecycle 与 current home-hotpath Gate B 已收敛，`D1` 仍因其余产品方法/typed surface 未关闭。 |
| 2026-07-19 | D 车道 Renderer typecheck 复核               | `npx tsc --noEmit --project tsconfig.renderer.json --pretty false` 退出码 2。已修本切片的 `UserInput` literal 与 `executionRun` reducer 类型；剩余诊断集中在 `packages/agent-runtime-client` 旧 event verifier/缺 `steerTurn`、GUI child/queue projection、plugin runtime 旧 Thread/Turn/Item 字段与独立 workspace optional-prop 问题。分类为 canonical consumer OPEN_REF，不允许通过恢复 `threadId/turnId/*AtMs/payload/kind/turnsView` 到生成 v2 类型来消除。                                                                                                                                                                                                                                                                                                                                           |

| 2026-07-19 | A/D 快速骨架源码汇合 | 新增 v2 `ClientRequest`、标准 `ClientResponse` payload、lossless `ServerRequest` 和 fail-closed `ServerNotification`，四类已进入 schema/TS；canonical append 统一拒绝晚于 outer batch 的 Item sequence，缺失 Turn 由 builder typed fail-closed。隔离源码验证：protocol 61/61、thread-store 28/28、App Server canonical 31/31、metadata/格式/diff、治理 0 违规与 command contracts 通过。`test:contracts` 的 protocol-types 752/752 通过，随后被共享 Plugin/session-gateway 旧断言阻塞；不得为过门禁恢复 compat。 |
| 2026-07-19 | D 快速骨架最终门禁 | contract 守卫已改到 canonical v2 facade（含 `steerTurn`、Thread.id/Turn.id），不恢复 session compat；当前 `npm run test:contracts` 完整通过，protocol types 752/752、client 292 checks。Renderer session/plugin 定向 Vitest 17/17、Rust steer 7/7、provider streaming 7/7、history 9/9、rustfmt 与 `git diff --check` 通过。全量 typecheck 仍被并行 Workspace optional-prop 与旧 canonical approval projection 阻塞；`./--help/json` 临时 schema 目录当前已不存在。状态：骨架完成、整体 Codex parity 未完成。 |
| 2026-07-19 | A 车道 v2 dispatcher/thread-start 骨架继续推进 | `dispatch.rs` 已将 `thread/start`、`turn/start`、`turn/interrupt`、`turn/steer` 接到 v2 handler，`mod.rs` 增加 `ThreadStartResponse` 投影，turn admission 改用 `turn/start`。该切片仍是 needs-integration：`thread/start` 当前以旧 `runtime.start_session` 创建并对缺失 model/provider 默认 `unknown`，必须迁到 canonical ThreadStore/route fail-closed；`thread/resume` 仍委托旧 queued-turn handler，v2 standalone notifications 与完整 public JSON-RPC 测试未闭合。 |
| 2026-07-19 | D 车道 typecheck ownership 复核 | `canonicalApprovalItemProjection.ts` 是唯一 tracked 的旧 `ThreadItem.payload` 读取点；v2 `ThreadItem` 没有 approval variant，Rust projection 对 approval 明确 fail-closed，因此该 projection 只能迁为 v2 tagged-union guard 或删除为 dead surface，禁止恢复旧 payload 类型。`useAgentChatWorkspaceCommandRuntime.ts`、`useAgentChatWorkspaceSceneRuntime.tsx` 是隔壁进程未跟踪热改，D 车道不接管、不夹写；完整 renderer typecheck 等其 owner 汇合后重跑。 |
| 2026-07-19 | D 车道 approval guard 收敛 | `canonicalApprovalItemProjection.ts` 保留为 retired guard：仅识别 v2 tagged-union identity，始终拒绝 approval；旧 payload/itemId/threadId 输入不会生成 GUI item。审批 current owner 仍是 typed server request -> `action.required/action.resolved`。定向 Vitest 9/9 通过；未删除负向守卫，避免旧协议回流。 |
| 2026-07-19 | D 车道 Renderer typecheck 收敛 | 接管隔壁进程已停止写入的两个 Workspace 未跟踪 owner 文件，在 props 解构边界统一生成 boolean/entry/chrome 的 resolved defaults，未改公共 contract、未使用断言或 compat。`npx tsc --noEmit --project tsconfig.renderer.json --pretty false` 完整通过，Prettier 与 `git diff --check` 通过。 |
| 2026-07-19 | D 车道 thread v2 public JSON-RPC guard | 新增 `app-server/tests/thread_v2_jsonrpc.rs`，从 public JSONL 验证显式 model/provider 的 `thread/start` 返回 v2 `{thread,...}` 且无 `{session}`，并锁住 `agentSession/start` 为 METHOD_NOT_FOUND。源码定向测试 2/2、rustfmt/diff check 通过。该证据只锁定当前 public wire；`thread/start` 的 route/canonical store fail-closed 与其他 lifecycle method 仍为 OPEN_REF。 |
| 2026-07-19 | D 车道旧 turn handler 去重 | 引用扫描确认 `processor/mod.rs::handle_turn_start/handle_turn_cancel` 只有定义、无调用；v2 `processor/turn.rs` 已是唯一 dispatcher owner，因此直接删除两段旧实现和失效 import，不保留 wrapper。rustfmt、零引用扫描、窄写集 diff check 通过；App Server bin/rlib 产物晚于源码，证明共享 build 已编译该删除。 |
| 2026-07-19 | A/D schema 全平面冲突复核 | 对现有 v0/v2 fixture 递归审计得到 22 个同名交集，其中 16 个异构；当前 9 项顶层 allowlist 没有隔离内联 `$defs`，会把仍在生产发送的 v0 `CanonicalThreadEventNotification` 错生成成 v2 `ThreadItem` wire。禁止扩大 allowlist 或新增 `Legacy*` rename；先迁 live `item/started|completed` 与 delta producer/consumer，再删除 v0 canonical event/DTO/fixture，最后恢复 schema 异构重复 fail-closed。若不能同轮完成，只允许按 Codex 的 `definitions.v2` namespace + 独立 flat-v2 codegen 做有退出条件的短期过渡。 |
| 2026-07-19 | 下一轮写集锁 | A-live-wire 独占 v2 item/notification DTO、schema registry/export 与生成物；D-producer 仅在 A handoff 后修改 App Server event mapper/tests；Renderer/client 车道只迁 current event pipeline。typed approval server-request 等 live item wire 稳定后再接入，避免同时改 `ServerNotification`/`ServerRequest`/生成物。所有车道禁止触碰 history/provider/storage 并行热区。 |
| 2026-07-19 | Agent fixture Gate 预检 | `npm run smoke:agent-runtime-current-fixture` 的历史恢复 31/31、流式收尾 32/32、Electron fixture guard 75/75 通过；重建 Electron host 在 `typecheck:electron` 失败，`electron/appServerHost.ts` 与 `electron/pluginRuntimeTaskHost.ts` 仍读取 v0 `sessionId/turnId/threadId/*AtMs`，而生成 v2 类型只提供 `Thread.id/Turn.id/*At`。该热区交给 Electron/A-live-wire owner，禁止通过给 v2 类型恢复旧字段或在 host 增加 compat。 |
| 2026-07-19 | D 车道收尾 contracts 复核 | `check:protocol-types` 仍为 752/752 且无漂移；随后 `app-server-client-contract` 被并行 Plugin runtime 迁移阻塞：`agentRuntimeAppServerClient.ts`、其测试、`PluginRuntimePage.tsx`、两个 fixture 缺少要求的 `pluginRuntime:` current 标记。该失败与本轮写集无关，Plugin owner 需补 current marker/契约，不得恢复旧 runtime facade。 |
| 2026-07-19 | D 车道 Plugin 契约事实校正 | 在当前共享工作树直接运行 `node scripts/check-app-server-client-contract.mjs`，292/292 通过；`pluginRuntime:` 已由 current Plugin host options 消费，上一条记录是并行修改中的瞬时失败，不再是现时 blocker。未为通过守卫恢复旧 facade。 |
| 2026-07-19 | D 车道并行写集重排 | A-live-wire 独占 `app-server-protocol` v2 lifecycle notification/schema/generated types；Electron-v2 独占 `appServerHost` 与 `pluginRuntimeTaskHost` 的 canonical identity/time 迁移；Thread-start 独占 App Server `thread/start`、RuntimeCore/ThreadStore 接线与 public JSON-RPC 测试。D 车道只做协调、集成与门禁，三车道不得交叉修改协议生成物。 |
| 2026-07-19 | 临时 schema 目录复核 | 仓库根 `./--help` 与 `./--help/json` 均不存在、无 tracked/untracked 状态；该目录是此前 CLI 参数误落盘的临时产物，不属于目标目录结构。后续 schema 生成只允许写入 `lime-rs/crates/app-server-protocol/schema/json/**`。 |
| 2026-07-19 | A-live-wire v2 lifecycle 骨架交接 | v2 `ServerNotification` 已包含 `thread/started`、`turn/started|completed`、`item/started|completed`、`item/agentMessage/delta`，payload 直接复用 canonical Thread/Turn/Item identity；v2 Rust 11/11、独立 schema fixture 通过。单一 TS codegen 在写文件前检测 v0/v2 nested `$defs`，当前对 `CollabAgentStatus/Thread/ThreadItem/ThreadStatus/Turn/TurnError/TurnStatus` 七个异构冲突 fail-closed；未新增 namespace、compat 或第二 codegen。状态 `needs-integration`。 |
| 2026-07-19 | Electron-v2 identity 交接 | Plugin task host 的 start/read/cancel 已切 `thread/start`、`turn/start`、`thread/read`、`turn/interrupt` v2 shape，删除 session-already-exists facade 和 legacy thread projection。Rust immediate admission 与 strict Gate B 稳定后，`appServerHost` 已删除 turn/start 特殊 streaming ACK、thread/read identity fallback 与 recent-turn inference，统一走 generic JSON-RPC request；bounded recent notification replay 继续作为 current 诊断/第二观察者。Electron Host 22/22、tsc、contracts 292 checks、diff check 通过。剩余 `agentSession/runtimeEvents/append` 与 `agentSession/action/respond` 归 item/approval cutover。 |
| 2026-07-19 | D 车道真实 `thread/start` 最小 owner | 删除 `unknown` model/provider fallback；model/provider 缺失或空白返回 INVALID_PARAMS。创建前将 provider/model/cwd/history/source/runtime selections 写入 BusinessObjectRef metadata，使 `start_session -> create_empty_canonical_thread` 首次 durable row 即完整；响应从 canonical `thread/read` 回读后投影，不再手工拼 Thread。public JSON-RPC 3/3，覆盖 v2 envelope、durable read-back、缺 selection fail-closed 与旧 `agentSession/start` METHOD_NOT_FOUND。真实 route/capability/credential preflight 仍是下一刀 OPEN_REF。 |
| 2026-07-19 | D 车道统一门禁复核 | v2 protocol 11/11、thread v2 JSON-RPC 3/3、Electron 38/38、Electron tsc、App Server client contract 292、`governance:legacy-report` 0 边界违规、`git diff --check` 通过。`npm run test:contracts` 预期停在上述 7 个 heterogeneous nested collision；这是主动 fail-closed，不是允许长期保留的兼容状态。 |
| 2026-07-19 | Gate B Claw 热路径复核 | `npm run smoke:agent-runtime-current-fixture` 的前置历史 31/31、流式 32/32、fixture guard 75/75、Renderer/Electron/sidecar 重建均通过；早期 `thread/read` 缺 `threadId` 已迁为 canonical `threadId + includeTurns`。后续 complete strict Gate B 已通过，见本表“Gate B direct-v2 strict 闭环”；home-hotpath 的独立 backend ledger 空结果不再作为 direct lifecycle blocker。 |
| 2026-07-19 | direct lifecycle producer/consumer 收敛 | App Server 统一到唯一 `V2NotificationProjector`；Rust/TS clients、Electron 与 Renderer 直接消费六类 v2 lifecycle/delta。删除 v0 `typedEvent`、`canonicalEvent`、六类 lifecycle DTO/schema/fixture/正向测试；单一 codegen 731 类型、0 失败、0 漂移，`npm run test:contracts`、Renderer typecheck、Rust protocol/client/projector 定向验证通过。`agentSession/event` 只保留明确 raw side-channel，wrapper lifecycle/action fail closed。 |
| 2026-07-19 | Renderer canonical wrapper 清理 | production projection 已收敛为 `appServerEventStreamProjection.ts` 的 direct v2 + provider/runtime/image/media raw side-channel；1126 行 `appServerEventPayloadProjection.ts`、test-only raw lifecycle/canonical/action projector 与正向 fixture 已物理删除。wrapper lifecycle/action/canonicalEvent/typedEvent 只保留负向 fail-closed 断言，旧路径进入 client contract forbidden guard。Renderer typecheck、相关 projection/thread client 62/62、stream/sync component 39/39、contracts 292 checks 与 diff check 通过。 |
| 2026-07-19 | Gate B identity 修复与证据推进 | Renderer adapter 先由 canonical session detail 解析 `thread_id` 再读取；fixture 已改为 `thread/list` 按新回合 `sessionId` 解析 canonical `threadId`，guard 63/63。早期 Electron 快速终态 `clientUserMessageId -> thread/read` fallback 只用于 Rust admission 迁移窗口，strict Gate B 后已物理删除，不再是 current identity owner。 |
| 2026-07-19 | Gate B direct-v2 strict 闭环 | 共享 Rust 热区的 projector 可见性、sink ownership 与 spawned future `Send` 问题已由对应 owner 修复，最新 App Server binary 完成重建。Claw fixture 删除 `agentSession/event` 正向订阅/parser 和三处 deferred 放行，只接受六类 direct v2 notification；guard 63/63。前置 direct-v2 证据 `claw-chat-current-fixture-direct-v2-strict` 为 `ok=true`、52/52 assertions、console/invoke error 0；动态 probe 同一 turn 观察 delta、Item started/completed、`turn/completed`，canonical read model 的 WebFetch completed/output/turnId 全部对齐。52/52 只保留为历史里程碑，current home-hotpath Gate B 的 definitive 结论见本表末尾 68/68 记录。client-local firstTextDelta/firstTextPaint/clientLocalOutput 已存在；只有 provider/server latency 继续归 App Server diagnostics trace，Renderer 不伪造。 |
| 2026-07-19 | Electron synthetic admission 删除复验 | Rust immediate admission 稳定后物理删除 Electron `requestStreamingTurnStart`、250ms ACK grace、thread/read identity retry、recent-turn inference、synthetic accepted response 与 8 组正向 fallback tests；generic JSON-RPC、bounded recent replay、取消和 stale-sidecar recovery 保持 current。client contract 已锁住 8 个 deleted 符号。Electron Host 22/22、Electron tsc、contracts 292 checks 通过；重建 Host 后前置 `claw-chat-current-fixture-direct-v2-no-host-fallback` Gate B 为 ok=true、52/52 assertions、app_server_handle_json_lines hit 45、console/invoke error 0，dynamic probe direct lifecycle 与 read model 同 turn 对齐；最终 current 结论由后续 home-hotpath 68/68 definitive 证据承担。 |
| 2026-07-19 | Renderer dead projection 收口 | 零消费者且恒返回 `null` 的 `canonicalApprovalItemProjection.ts` 与同名 test 已物理删除；审批 current owner 继续是 typed server request/action-required 路径。旧 `appServerEventPayloadProjection.ts` 也已由独立 current stream projection 完整替代并删除。三条物理路径统一归类 `dead / deleted / forbidden-to-restore`，未新增 compat/deprecated wrapper；remaining raw side-channel 只存在于 current stream projection，后续按 provider/runtime/image/media typed owner 迁出。 |
| 2026-07-19 | home-hotpath Gate B definitive 闭环 | definitive 证据为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-home-hotpath-v2-definitive-summary.json`：`ok=true`、`Gate B CDP controlled fixture`、68/68 assertions，console/page error、legacy command hit、mock fallback hit 均为 0。canonical identity 为 `sessionId=sess_7361160a434846a9841ec0e7bb5bf2fa`、`threadId=thread_f2065f4a31a6470aafad7ff4d3ebc072`、`turnId=turn_8c35081960cd448bbb2a9c024020f6a0`；pending paint 47ms、submit accepted 247ms、first text paint 344ms、first delta to paint 31ms、client-local output 71ms。current home-hotpath Gate B 与 instrumentation blocker 均已关闭；client-local 指标已存在，只有 provider/server latency 继续归 App Server diagnostics trace，Renderer 不以 `Date.now()` 等本地时间戳伪造。Windows cross-platform UDS、live provider 及 V1-15 的 restart/resume/compaction/model-switch 等完整覆盖仍分别 OPEN_REF，不得据此宣称整体 Codex parity 完成。 |
| 2026-07-19 | Renderer projector 删除后 Electron 复验 | `smoke:agent-runtime-current-fixture` 在 source rebuild 后通过历史 31/31、流式 32/32、fixture guard 76/76，并通过 `home-hotpath-regression` 与 `home-hotpath-greeting-regression` 两条真实 Electron Claw 热路径；App Server 缺 provider/model 的启动崩溃未复现。聚合流程随后在 Coding Workbench 的 `agentSession/update` 失败：fixture 用调用方生成的 session id 更新由 v2 `thread/start` 创建的 canonical session，且后续仍向 v2 `turn/start` 传旧 `runtimeOptions`。该项归 typed session identity/route 迁移 OPEN_REF，不恢复 session alias 或 wrapper；不影响本轮 direct v2 projector 与 Claw 主路径通过结论。`verify:gui-smoke` 的 Renderer/Host/App Server build 通过；首次 shell 因测试 HOME 未隔离、检测到真实用户 `lime.db-wal` 而正确拒绝迁移，未触碰用户数据；用隔离临时 HOME 直接复跑同一 built Electron smoke 后 App Server initialize、Claw reload、Memory settings 与结构化 evidence 全部通过。 |
| 2026-07-19 | P0-01 v2 method/serverRequest 单一 owner 完成 | 对照 Codex `server_request_definitions!`，`protocol/v2` 现独占 9 个 request、6 个 notification、`mcpServer/elicitation/request` serverRequest 及 `McpServerElicitation*` DTO；中央 `app_server_method_catalog()` 聚合 v0 产品子集与三类 v2 registry，并对重复 wire method fail closed。`ServerRequest` 是仅接受 canonical elicitation params 的闭合 tagged union，未知 method fail closed；response 使用 Codex 名 `McpServerElicitationRequestResponse`。已物理删除 v0 elicitation method 常量、catalog 项、DTO/tests 和 3 个 v0 schema，不保留 alias/compat；Rust/TS consumers 与三处 transport fixture 已迁 canonical form。单 owner 重生成 schema/manifest/v2 fixtures 与 TS：731 类型、0 失败、0 漂移，生成 TS `ServerRequest.method` 为 literal 且 params typed。验证：protocol 56/56 + schema fixture 1/1、App Server elicitation 7/7、Rust client 29/29、TS client 70/70、Renderer/Electron MCP 47/47、`npm run test:contracts` 全通过、`governance:legacy-report` 0 边界违规、rustfmt/Prettier/diff check 通过。P0-01 当前切片 100%；后续扩展 approval/tool server request 直接在 v2 registry 增加 typed arm，不得恢复 v0 owner。 |
| 2026-07-19 | diagnostics trace v2 admission skeleton | 根因确认是 v2 `additionalContext.metadata` 未进入 RuntimeRequest 顶层，导致 `trace_context_for_turn` 返回 None；provider `elapsed_ms` 与 `server_event_emitted_at` 已是 Rust current facts。`processor/turn.rs` 现仅提升 application kind 下的 `agentUiPerformanceTrace`/snake alias，对 untrusted/非法/非对象值 fail closed，并保留原 additionalContext。Rust 定向 2/2、rustfmt/diff check 通过；未触碰 protocol/Renderer 热区。OPEN_REF：重建 sidecar 后复跑一次 home-hotpath Gate B，必须证明 `diagnostics/trace/list traceCount>0`，不得在 Renderer 伪造 providerWait。 |
| 2026-07-19 | P0-05 多 recipient 恢复骨架闭合 | `recover_agent_control_spawns` 不再因首个 child 的 typed `PendingRoute` 提前终止整批恢复：保留首个 pending 错误并继续扫描，已有持久 route 的后续 child 仍会启动 deterministic mailbox turn、完成 canonical 投影并 ack；循环结束再返回首错。新增双 recipient restart 回归，证明缺 route mailbox 保持 pending、可执行 mailbox 被消费。App Server 定向 1/1、restart 模块 12/12、rustfmt 与 diff check 通过。仅修改 RuntimeCore recovery owner，未进入 provider/runtime backend 热区。OPEN_REF：schedule in-flight 去重、已有 route 不可执行时统一 typed fail-closed、catalog/credential generation 变更后的 deterministic retry。 |
| 2026-07-19 | diagnostics trace v2 admission + timing Gate B 闭环 | v2 application metadata lowering 已接入既有 RuntimeRequest trace owner；source-built sidecar 重建后真实 Electron home-hotpath 证据 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-home-hotpath-v2-trace-timing-final-summary.json` 为 `ok=true`、72/72 assertions。Gate B 覆盖 Electron/preload/IPC/App Server/read model/GUI，`appServerIpcHitCount=62`、trace `eventCount=8`、`traceCount=1`，`provider.first_text_delta.received` 的 `providerWaitMs=90`、`app_server.message_delta.emitted` 的 `serverEventEmittedAt=1784448387417` 均来自 summary-only diagnostics trace；legacy/mock/page error 均为 0，W3C carrier、trace export/support bundle 通过。新增压缩器安全标量投影与 2 条正/负向单测，定向脚本测试 65/65。该 instrumentation blocker 关闭；V1-15 其余 restart/resume/compaction/model-switch/approval/MCP/child-agent/cold-read、live provider 和 Windows UDS 仍保持 OPEN_REF。 |
| 2026-07-19 | current fixture/contract stale guard 收口 | `test:contracts` 揭示两个守卫仍要求已迁出的事实：Coding Workbench fixture 的 `agentSession/update`，以及 runtime 缺 route 的旧 session-default 文案。守卫现要求 `thread/start -> turn/start -> thread/read/list`，并禁止 Coding Workbench 回流 `agentSession/update`；runtime fail-closed 文案锁定 start/resume canonical thread 的完整 `modelProvider/model` route。Coding Workbench metadata 静态测试同步从 `runtimeOptions.runtimeRequest.metadata` 迁到 v2 application `additionalContext.metadata`，并禁止两条旧 metadata 入口。验证：`npm run test:contracts` 全通过（731 types、292 client checks）、相关 Vitest 71/71、Prettier/diff check 通过。该项是 source/contract 收口；Coding Workbench 独立 Electron Gate B 本轮未复跑，仍需后续真实证据。 |
| 2026-07-19 | Codex turn route / canonical fixture 收敛 | 对照 Codex current 确认 `turn/start` 不包含 `queueIfBusy`、`skipPreSubmitResume`、`runtimeOptions`、`providerPreference/providerConfig`、`searchMode/webSearch`；same-turn input 由 Core session loop steer，provider 属于 thread sticky config，turn 只允许 typed model 等 override。Renderer `additionalContext` 已只保留 `rendererEventName` 与 application metadata/trace，业务 route/queue/system prompt 不再偷渡。RuntimeCore 将 imported-only defaults owner 替换为 `session_runtime_defaults`，从 canonical session metadata 合并 provider/model/cwd/workspace root/approval/sandbox，再由 turn typed request 覆盖；model-only override 保留 durable provider，缺完整 route 继续 fail closed，旧错误不再指导调用方提交 `runtimeOptions`。Coding Workbench fixture 已切 `thread/start -> turn/start -> thread/read/list` 与 canonical session/thread/turn identity，旧 `runtimeOptions/runtimeRequest/agentSession/update` 归零；`agentSession/event` 仅作为尚未迁出的 raw artifact side-channel 保留。验证：session defaults 3/3、model selection 32/32、agentProtocol 28/28、Workbench guard 6/6、Renderer typecheck/格式/diff check 通过。OPEN_REF：Workbench 独立 Electron Gate B、v2 thread/resume/settings、typed approval/artifact Item 到位后删除余下 raw side-channel 与旧 action/queue methods。 |
| 2026-07-19 | Renderer typed TurnStart contract 收口 | `threadClient` 测试 helper 已直接构造 v2 `TurnStartParams`，只正向保留 `threadId/input/model/effort/approvalPolicy/sandboxPolicy/outputSchema` 与 application `metadata`；旧 `runtimeRequest`、provider/search、queue/skip、system prompt 和 workspace route 不再作为 `turn/start` 测试事实。client contract 同步改为 typed fields 正向守卫与 dead submit 字段负向守卫。验证：`threadClient.test.ts` 29/29、Renderer `tsc --noEmit`、`npm run test:contracts`（731 types、292 client checks）和 `governance:legacy-report`（0 边界违规）通过。下一刀是让 submit builder 直接产出 typed TurnStart + renderer event route，并删除 `AgentUserInputOp.preferences` 中仍无 wire 消费的 provider/search/thinking/execution/auto-continue 胖 DTO 字段；该刀继续避让 App Server Rust/session defaults 与 `internal/refactor/v1/**` 并行热区。 |

| 2026-07-19 | Renderer typed submit op 骨架完成 | `AgentUserInputOp` 已从 `text/session/preferences/provider/search/queue/systemPrompt/workspace` 胖 DTO 收缩为精确 `{ type, eventName, turn: TurnStartParams }`；submit builder 直接生成 typed `threadId/input/model/effort/approvalPolicy/sandboxPolicy/additionalContext`，mapper 只把 renderer event route 合入 application context。`AgentUserPreferences` 已从 source 与声明文件删除，provider/search 等只允许留在各自 durable thread/tool-policy owner，不再成为 Turn submit surface。新增类型级与 client contract 负向守卫；相关 Renderer Vitest 6 文件 99/99、governance legacy 0 边界违规、Prettier/diff check 通过。完整 client contract 在本刀前曾通过 731 types/292 checks；本刀后 typed-op 新守卫无失败，但当前全脚本被隔壁 Rust data/runtime backend 正在迁移的 4 个 stale snippet 检查阻塞，未越界修改其热区。下一刀是把 `submitOpRuntimeCompaction` 提炼为只负责 metadata sanitizer 与 typed model 决策，删除已无消费者的 provider/search/thinking/execution/auto-continue 输出和 `submitOpToolPreferenceCompaction` 平级实现。 |

| 2026-07-19 | P0-05 mailbox recovery 单飞骨架 | RuntimeCore 增加仅用于进程内协调的 per-session keyed async gate；`schedule_pending_agent_mailbox_triggers` 的同 session recovery 串行执行，不丢弃 follower，且每次调用继续等待自身 admission/完成。leader drain 期间追加的 durable TriggerTurn 会在同一轮或后继 flight 被再次读取，关闭 `TurnAlreadyActive` 提前返回和尾部丢唤醒；不同 session 不共享全局锁。durable mailbox/ThreadStore 仍是唯一 pending 事实源，gate 不持久化业务状态。阻塞 backend 并发回归 1/1、mailbox 模块 11/11、restart 模块 12/12、窄写集 rustfmt/diff check 全部通过；证明两个 follower 不提前返回、两条 deterministic mailbox turn 各执行一次且最终 pending 为空。OPEN_REF：已有 route 不可执行时统一 typed fail-closed、catalog/credential generation 变更后的 deterministic retry。 |
| 2026-07-19 | P0-05 typed route failure + committed wakeup 骨架 | 缺 provider/model selection 不再依赖字符串 Backend sentinel，统一由 typed `PendingRoute` 保留 provider-only/model-only hint；已有 route 的 failure 按 `(category, reasonCode)` allowlist 分类：provider setup/disabled、credential missing、model catalog unavailable 和明确 no-candidate 才可 Pending，capability gap/unsupported protocol/endpoint 返回 typed `RouteRejected`（JSON-RPC `retryable=false`），internal/未知 readiness 保持 Backend fail-closed。provider create/update/import、catalog fetch、credential create/update 成功提交后 fire-and-forget 唤醒 durable AgentControl/mailbox recovery，Pending 仅 debug、其它恢复错误 warn，不反向污染已成功配置 RPC。最新 source build 精确 rejection 1/1；同一测试二进制晚于窄写集源码，model-selection 33/33、initialization 11/11、restart 12/12；rustfmt、scoped diff check、`governance:scripts` 通过。client contract 中本轮 typed route 守卫已通过，整体仍被隔壁 Plugin data surface 缺 `resolve_plugin_runtime_dir` / `list_plugin_installed_state` 两项阻塞。全部为 `current`，无 compat/deprecated；OPEN_REF：committed generation 去重、queued turn/thread settings 唤醒、providerConfig/credential generation 持久化 owner 与真实配置变更端到端重试证据。 |
| 2026-07-19 | Coding canonical Item projection + Gate B 闭环 | canonical `command_execution`/`patch` ThreadItem 已投影为 coding changes/outputs/logs，Workspace 只消费 `projectedThreadItems`；fixture 收敛为 message delta/completed、canonical tool/file/command `item/started|completed`、`artifact.snapshot` 与 `turn/completed`，raw command/test/file/patch 双轨及其正向 fixture 已删除，不保留 compat。真实 `gui-coding-input` Gate B 已通过：session/thread identity 一致，recovery 严格 turn-scoped，changes/outputs/logs 在 GUI 可见，每个 turn 恰好 1 条 assistant message，无 duplicate ID，terminal 后无 `inProgress`，console/page/invoke error 均为 0。该切片归 `current`，被替代 raw surface 归 `dead / deleted / forbidden-to-restore`；整体 Codex parity 仍未完成。下一刀依次收敛 `thread/resume`、typed approval/server request、typed artifact/test Item，并在 typed owner 到位后删除余下 `artifact.snapshot` raw side-channel。 |

| 2026-07-19 | P0-05 durable route generation 骨架 | `core::database::dao::RouteStateDao` 以 SQLite `settings` 表的 `model_route_generation` key 成为唯一 committed generation 事实源；system/custom provider、credential readiness 与 provider model cache 的实际写入/删除在同一事务推进 generation，alias/sort/UI/usage/error、重复 toggle/import、cache hit/空 delete 不推进。RuntimeCore 删除 process-local committed counter，只保留 single-flight gate 与 `last_attempted_generation`，每次 commit signal 从 Local App DataSource 读取 durable generation，同 generation 只尝试一次，provider/key delete 也发布 signal；启动仍由 durable mailbox recovery 承接。验证：DAO 5/5、provider readiness 2/2、catalog cache 7/7、coordinator 2/2、Local provider 2/2、scoped rustfmt/diff check 通过。全部为 `current`，无 compat/deprecated。OPEN_REF：统一 queued user turn + mailbox pending-work gate、cold hydrate 的 per-turn options 持久化、真实 provider/catalog/credential 变更后的 Gate B 重试证据；整体 Codex v1 对齐仍约 38%，不宣称 P0-05 或 v1 完成。 |
| 2026-07-19 | Renderer typed submit compaction + fail-closed queue | `submitOpRuntimeCompaction` 仅保留 metadata sanitizer 与 typed model 决策；`AgentUserInputOp` 当前唯一形状为 `{ type, eventName, turn: TurnStartParams }`，provider/search/thinking/execution/auto-continue submit 字段及 `submitOpToolPreferenceCompaction` 平级实现归 `dead / deleted / forbidden-to-restore`。图片输入统一由 `MessageImage.data + mediaType` 构造完整 `data:image/*;base64,...` URL；跨 target session 时仅消费 `executionRuntime.session_id === resolvedActiveSessionId`，避免旧会话裁剪 metadata。v2 尚无 `queueIfBusy`，queued submit 直接 fail closed 并提示迁移 `turn/steer`，删除错误的 accepted/dispose/refresh 假队列路径。验证：相关 Renderer 9 文件 102/102、legacy catalog 215/215、Renderer `tsc --noEmit`、Prettier、`git diff --check` 全通过；`governance:legacy-report` 0 边界违规。`node scripts/check-app-server-client-contract.mjs` 仍仅被隔壁 Plugin data surface 缺 `resolve_plugin_runtime_dir(&state)` / `list_plugin_installed_state().map_err(data_error)` 两条 stale snippet 阻塞，未越界修改。`smoke:agent-runtime-current-fixture` 前置历史 31/31、流式 32/32、fixture guard 76/76、Renderer/Host/App Server 重建通过；真实 Claw Electron 热路径因 fixture 预期两段新闻正文未投影而失败，输入框已恢复可用。全部为 `current` 或 `dead`，无 compat/deprecated。OPEN_REF：实现 Codex `turn/steer` 的 active-turn 输入 owner 并删除剩余 queue projection；继续清理 execution facade 未使用的 `searchMode` / `explicitToolPreferences` / `requestTurnId`；修复 Claw fixture 的 canonical assistant body assertion。 |
| 2026-07-19 | Renderer submit facade dead-option cleanup | preparation -> execution 接线删除 `searchMode`、`explicitToolPreferences` 与 execution 层 `requestTurnId`；`requestTurnId` 仅继续由 lifecycle 保留为本地 optimistic identity，不再作为 submit facade 参数。owner boundary 清单同步移除已无 `RuntimeSearchMode`/session DTO 消费的 preparation/builder/compaction 项。验证：`agentStreamSend` 3/3、`agentStreamUserInputSendPreparation` 23/23、core owner boundary 94/94、Renderer `tsc --noEmit`、Prettier/diff check 通过；`agentStreamUserInputSubmission`/`agentStreamSubmitExecution` 复跑被隔壁物理删除 `packages/agent-runtime-projection/src/threadRollbackProjection.ts` 阻塞，未恢复该 dead surface。分类为 current typed facade + dead option，未新增 compat。OPEN_REF：待 projection 车道稳定后补 submission/execution fixture 回归，并继续实现 `turn/steer` active-turn owner。 |
| 2026-07-19 | D 车道 v2 `thread/resume` 冷恢复骨架 | `METHOD_THREAD_RESUME` 不再路由到 `thread_resume_unavailable`；`RuntimeCore::resume_thread` 以 canonical ThreadStore 读取 thread/session identity 后执行当前 session hydration，handler 返回同一 Thread/Turn/Item 投影。支持 `excludeTurns` 与 `initialTurnsPage`（复用 `thread/turns/list` page owner），paginated history 强制 metadata-only，resume 不发 `thread/started`。history/path、runtime config overrides、permissions+sandbox 冲突和 archived thread 均 fail-closed；v2 ingress 与 request serialization scope 拒绝/忽略 legacy `sessionId`，避免旧 session resolver 在 dispatcher 前抢占请求。公共 JSON-RPC 6/6、request serialization 14/14、治理 0 分类漂移/0 边界违规、窄写集 rustfmt/diff check 通过。`cargo fmt --all --check` 被其他并行 `agent-runtime` 文件未格式化阻塞；Rust related 被并行 `canonical_rollout.rs` 缺 digest/file helper 与旧 queue audit 字段阻塞；contracts 仍被既有 Plugin data surface 两条 stale snippet 阻塞，均未归本刀。分类：v2 handler/read/projection/serialization 为 `current`；queued session resume 与 `agentSession` facade 继续 `deprecated`，legacy sessionId 入口 `dead / forbidden-to-restore`。下一刀：同一 listener 的 loaded-thread rejoin、pending typed server request/token usage replay 与 in-progress interruption；仍不得恢复 queued-turn resume。 |

| 2026-07-19 | P0-05 queued turn + mailbox 统一恢复骨架 | `queue.added` 以 `queuedTurnIntent` 持久化 per-turn runtime options，并在写入前清除 direct API key；cold hydrate 只恢复仍为 `Queued` 的同一 `event.turn_id`，损坏/未知 schema fail closed。既有 `MailboxTriggerFlights` 提升为 session pending-work 唯一进程内 gate：gate 内先按 turns 顺序连续恢复 queued user turn，`Blocked`/typed `PendingRoute` 保留 lease 且禁止 TriggerTurn 越过，只有 queue `Empty` 后才 drain mailbox；公共 queued resume 复用同一 gate。`recover_agent_control_spawns` 现合并内存 queued session 与 Projection DB `projected_turns.status='queued'` 的无上限稳定扫描，多 session 保留首个 PendingRoute 并继续。验证：queued intent/schema/脱敏 3/3、新增内存 queued recovery 1/1、mailbox 11/11、queue resume 5/5、restart 15/15、scoped rustfmt/diff check、治理扫描 0 分类漂移/0 边界违规。全部为 `current`，无 compat/deprecated。OPEN_REF：real EventLog + ProjectionStore cold restart round-trip、queued + mailbox 同 session 顺序、PendingRoute lease exactly-once、多 session 新回归、真实 provider/catalog/credential generation Gate B；P0-05 与整体 Codex v1 仍未完成，整体约 38%。 |
| 2026-07-19 | Resume contract dead surface 第一刀 | `packages/agent-ui-contracts` 已物理删除 `AgentRuntimeResumeMode` / `AgentRuntimeResumeActionDecision` / `AgentRuntimeResumeContract`、validator、schema constant、checked-in Resume schema 与正向测试；capability manifest、`hitl.actions`、`action.required/action.resolved` 与 action projection 保留。包内测试 29/29，dist 扫描旧 Resume 符号 0，scoped diff check 通过。分类：保留的 action/capability 面为 `current`；Resume contract 为 `dead / deleted / forbidden-to-restore`。 |
| 2026-07-19 | Claw queued-resume fixture 收口 | 删除 `resumeQueuedTurnForPromptIfNeeded`、`waitForBackendTurnStartWithCurrentQueueResume`、Skills `*QueueResume` 断言与 RPC `{resumed, turnCount, turns}` summary；Skills flow 直接等待 backend ledger turn start，pending-steer 只以 backend start + current read-model dequeue 作证据，`thread/resume` 仅保留负向 guard。六文件 `node --check`、Prettier、scoped diff check 通过，fixture test 63/63；本刀未运行 Electron smoke。分类：backend ledger/read model 为 `current`，queued-resume helper/summary 为 `dead / deleted`。 |

| 2026-07-19 | Renderer active-turn steer 骨架 | submit gate 现在无条件读取 canonical `thread/read` 的 active turn；active 时构造最小 typed `turn/steer`（`threadId/expectedTurnId/input/clientUserMessageId`），idle 时才构造 `turn/start`。lifecycle 改为惰性 optimistic：canonical 判定前不写本地 Turn/Item，steer 不创建新 listener/Turn，接受后刷新 canonical read model；本地 busy/queued flag 不再投影“已加入排队列表”假草稿。adapter 增加 typed steer 委托，canonical control API 改名 `getThreadTurnControl`，client declaration 与 contract guard 同步。架构图确认：`internal/aiprompts/architecture.md` §8.1 已记录 canonical start/steer gate、identity 与 fail-closed 边界。验证：adapter 14/14、execution 8/8、submission 1/1、lifecycle 8/8、preparation 23/23、flow-control 16/16、typed builder 20/20；`git diff --check`、Prettier 与 `governance:legacy-report`（0 分类漂移、0 边界违规）通过。`npm run test:contracts` 已越过本刀 typed guard，但被隔壁 Rust App Server data/runtime 迁移的 `resolve_plugin_runtime_dir` 与 `list_plugin_installed_state` stale snippets 阻塞。全量 Renderer `tsc` 因共享工作树并行编译持续 13 分钟无输出后中止，未作为通过证据。OPEN_REF：删除余下 queued snapshot/queue event GUI projection；补真实 Electron Gate B `turn/steer` 证据；Rust steer metadata/input lowering 与 non-Regular error parity；继续全局 Codex v1 缺口审计。 |
| 2026-07-19 | Renderer queued-turn 写平面退场 | typed thread client、adapter、`useAgentStream/useAgentChat`、workspace scene、Inputbar、MessageList、Harness 与 Reliability 已删除 queued user turn 的 promote/remove action；`QueuedTurnsPanel` 与五语言 action 文案物理删除，stop 只恢复当前 submitted draft，不再删除/恢复 queued draft。`scripts/check-app-server-client-contract.mjs` 删除正向片段并新增生产 Renderer 负向守卫。架构图确认：`internal/aiprompts/architecture.md` §8.1 已明确 durable pending-work 只归 RuntimeCore/session loop，GUI 只读 canonical status/count/evidence。保留 Rust durable recovery、session/evidence queued count；详细 snapshot projection 作为下一刀 dead candidate；App Server v0/generated queued methods 由协议车道继续退场，本刀不夹写。验证：FlowControl 10/10、queue hydration 2/2、Inputbar 20/20、Workbench 5/5、Conversation 10/10、Reliability/Harness 15/15；client/adapter/i18n 与治理门禁见本 checkpoint 后续验证记录。整体 Codex v1 仍约 38%。下一刀：删除详细 `QueuedTurnSnapshot` GUI state、raw queue event 到本地 projection 和无消费者的 user-turn queue helper；不得混删 Plugin/task 通用 `queue.changed`。 |
| 2026-07-19 | Codex `thread/resume` 语义复核 | 直接对照 Codex `protocol/v2/thread.rs::ThreadResumeParams/Response` 与 App Server `thread_resume_inner`：该 method 只负责 running-thread rejoin 或 durable history rehydrate，不承载 action decisions，不启动 queued turn。Content Factory workflow evidence 中 `thread/resume + resumeContract.decisions` 因此是 `dead`；其四个目标文件已存在其他进程的 v2 method 改动，D 车道本轮只读并避让，精确退出点是删除 resume-contract projector/parser/filter 与正向 test，保留 `agentSession/action/respond` workflow audit 直到 typed approval/server-request owner 接管。 |

| 2026-07-19 | P0-05 durable pending-work 冷恢复闭环 | RuntimeCore 在任何 mailbox canonical Item/ack 与 lifecycle event 前调用 `ExecutionBackend::preflight_turn`；RuntimeBackend preflight 与实际执行复用完整 provider/catalog/credential/capability route resolution，PendingRoute 只回滚内存 accepted turn并恢复 queued lease，实际执行仍二次 resolve 和写 routing evidence。spawn/followup 在 durable append 后 detached wake，普通 completed/failed/canceled terminal 再次唤醒同 session pending work，统一 per-session gate 继续保证 queued FIFO 优先且 PendingRoute 不自旋。`queuedTurnIntent` 升级 typed v2 allowlist：只保留 durable provider/model/policy/workspace/search identity 与 bounded `clientUserMessageId/collaborationMode`，direct API key/base URL、system prompt、output contracts、provider capability override、未知 metadata 与派生 prompt 均不持久化；direct credential 无 repository provider identity 时 fail closed。cold discovery 合并 EventLog 安全尾部 repair、ProjectionStore 与内存状态，单 session 非路由错误记录后继续扫描。最终同一 App Server test binary：pending-work 5/5、route-generation 1/1、mailbox 14/14、intent 5/5、queue-resume 5/5、restart 12/12、RuntimeBackend initialization 11/11、model-selection 33/33、turn-flows 5/5、EventLog 19/19；contracts 298 checks、legacy report 0 分类漂移/0 边界违规、rustfmt/diff check 通过。current fixture 前置历史 31/31、流式 32/32、guard 76/76 后，fixture build 被并行协议车道已删除 `thread/archive|unarchive` 但 TS client consumer 尚未同步阻塞；未获得真实 Electron/preload/IPC/live provider generation Gate B，不关闭 P0-05，整体 Codex v1 对齐仍约 38%。 |
| 2026-07-19 | Renderer queued-turn 写平面 Gate B 收尾 | 首次 `smoke:agent-runtime-current-fixture` 在真实 Electron 根渲染发现 `InputbarCore` 遗留 `queuedTurns.length` 导致 `ReferenceError`；删除该读取和旧正向 component test，并补 `queuedTurnCurrentBoundary` 物理删除/不回流守卫。最终 FlowControl/Inputbar/i18n/boundary 58/58、adapter 14/14、Conversation 10/10、Reliability/Harness 15/15、queue hydration 2/2、Workbench 5/5，Renderer 9304 modules 与 Electron host/preload 构建通过；`governance:legacy-report` 为 0 分类漂移/0 边界违规，生产 Renderer promote/remove 名称扫描为 0，Prettier、`node --check` 与 `git diff --check` 通过。复用已构建 host/sidecar 的真实 Electron `home-hotpath` Gate B 通过，summary 为 `.lime/qc/gui-evidence/claw-chat-current-fixture/renderer-queued-write-plane-retired-summary.json`。聚合 current fixture 与 `verify:gui-smoke` 的后续 host 重建被并行协议车道 `ThreadArchive*` / `METHOD_THREAD_SHELL_COMMAND` 导出不完整阻塞；`test:contracts` 被 `agentSession/archiveMany` manifest method constant 缺失阻塞，均不归本刀。分类：start/steer/read/status/count/evidence 与 Rust durable recovery 为 `current`；Renderer promote/remove/UI/action 文案为 `dead / deleted / forbidden-to-restore`；详细 queued snapshot/projection 为下一刀 `dead candidate`。 |
| 2026-07-19 | Renderer queued-turn raw/snapshot projection 收口 | 按 Codex v2 `turn/start`/`turn/steer` 单一主链，物理删除 `queue_added`/`queue_removed`/`queue_started`/`queue_cleared` 的 TS protocol union、parser、Renderer projection、stream lifecycle callback 与 `queueProjection` package；删除 rich `QueuedTurnSnapshot` 输入草稿恢复，仅保留 submitted draft 的 cancel/error/stop 恢复和 canonical turn/item 活动。Inputbar、runtime card、MessageList timeline 与 scene 直传不再读取 queued snapshot/count；Canvas/Coding、Reliability/Harness 非 MessageList projection 仍保留上游 numeric/diagnostic owner，Plugin/task 通用 `queue.changed` 与 Rust durable pending-work 不变。contract guard 改为 retired-file/raw-event 负向守卫并补 Plugin 正向守卫。验证：package 289/289；raw protocol/projection/runtime 197/197；restore 15/15；Inputbar/MessageList/Scene 41/41 + 18/18；Prettier、`git diff --check` 通过。`node scripts/check-app-server-client-contract.mjs` 仅被并行 session-history fixture 的 stale archive snippets 阻塞，未出现本刀新增 guard 失败。分类：raw queue event、queueProjection、rich queued draft restore 为 `dead / deleted / forbidden-to-restore`；Plugin/task queue、RuntimeCore pending-work、canonical status/count/evidence 为 `current`；剩余 Canvas/Coding/Reliability/Harness queued snapshot owner 继续下一刀。
| 2026-07-19 | P0-05 非 chat turn route preflight 骨架 | image command preflight 复用现有 typed intent parser，识别成功或缺业务参数时跳过 chat route，malformed intent 继续 fail closed；plugin worker preflight 复用 `Run/Reject/Ignore` resolver，已安装 Run 与 Reject 跳过 chat route，未安装 Run 和 Ignore/plugin activation 继续 preflight，避免 mailbox/lifecycle 后才发现 fallback route 不可用。RuntimeCore image 回归、plugin Run/Reject/PassThrough spy 回归已落地；scoped rustfmt 与 diff check 通过。App Server 定向构建已进入本体后，被并行未跟踪 `runtime/session_shell.rs` 对私有 `RuntimeSessionTaskContext.turn_id` 字段的访问阻塞（应由该车道迁到 `turn_id()`），本轮按窄写集规则未夹写；未重跑 current fixture、GUI smoke 或 Gate B。全部新增面为 `current`，无 compat/deprecated/dead；P0-05 仍不关闭，下一步待 runtime/protocol owner 汇合后重跑 4 条新回归与 current fixture。 |
| 2026-07-19 | Resume / queue 双轨删除与 canonical Gate B 闭环 | Renderer input restore 只消费当前 submitted draft，删除 queued draft normalizer/sort/read-model refresh、旧 reason/plan fields 和 queue raw-event dispatcher；相关 projection/restore 42/42。协议为 748 schema definitions / 740 TS types / 0 漂移，完整 `test:contracts` 与 `governance:legacy-report` 通过。session-history fixture 删除 `agentSession/update`、`projected_*` seed 与 visual replay 双文件，真实 Gate B 证明 canonical ThreadStore 3 Turns / 9 Items、分页、DOM 顺序及 archive/unarchive 重启读回。Recovery CDP Gate 修复 route generation 读取的 `Connection` 借用与 sidebar 漏传 canonical threadId 后通过，证明 Electron/preload/IPC/`app_server_handle_json_lines`/`thread/start/read/list/resume` identity 一致、metadata-only resume、无 legacy fields、不重发 `thread/started`。分类：ThreadStore/v2 recovery/read model 为 `current`；旧 resume contract、queue event/composer restore、legacy history seed 为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。整体 Codex v1 仍约 38%，下一刀回到 loaded-thread live listener rejoin、resume response/event 原子排序、active turn snapshot、pending server request/token usage replay 与 stale `inProgress -> interrupted`。 |
| 2026-07-19 | loaded-thread active snapshot 骨架 | `agent-runtime` session actor 增加有序 `RuntimeSessionSnapshot { active_turn_id }` command/handle/registry；App Server `thread/resume` 只通过 canonical ThreadStore 的 thread/session identity 查询现有 actor，cold hydrate 不伪造 active loop。resume 投影让 live turn 保持 `inProgress`、其它 stale `inProgress -> interrupted`，Thread 归一为 `active`/`idle` 并保留 `systemError`。验证：actor snapshot 1/1、resume normalize 3/3、public JSON-RPC live actor resume 定向 1/1、两 package rustfmt、scoped diff check 与 `governance:legacy-report`（0 分类漂移、0 边界违规）通过；后续复跑被并行 `canonical_thread_store`/history 多库拆分连续暴露的 ATTACH 未完成、database lock 与 `E0597` statement borrow 编译错误阻塞，新旧测试均在本刀断言前失败，不归本写集且未夹写。架构图确认：`internal/aiprompts/architecture.md` §7 已记录唯一 snapshot owner 与 listener 未完成边界。分类全部为 `current`，无 compat/deprecated/dead 新增；整体仍约 38%。OPEN*REF：per-thread listener identity/generation、connection-scoped subscription、response -> usage/goal/pending request -> live event 原子排序与 raw JSONL Gate。 |
| 2026-07-19 | loaded resume R2/R3 listener/replay source 骨架 | R2 在 `server_request.rs` 让 PendingRoute 保存原始 request、canonical thread scope、owner 与稳定生成序，`snapshot_for_owner_thread` 13/13；R3 新增 `thread_state.rs` current manager，提供 per-thread listener command channel、strict generation、旧 listener cancel、双向 connection index、disconnect/unsubscribe cleanup 2/2。两车道 rustfmt、diff check、定向测试通过；尚未接 event pump/processor，断连后的 callback owner 迁移、response/replay/live 原子序仍 OPEN。分类全部为 `current`，无 compat/deprecated/dead。R1 connection context plumbing 进行中。 |
| 2026-07-19 | loaded resume listener 汇合骨架（执行中） | App Server 已把 RuntimeEventHub 从全连接广播收窄为 canonical `AgentEvent.threadId -> per-thread listener` demux；listener generation 持有唯一 v2 projector，并在同一 command actor 内执行 exact connection subscribe、response、thread-scoped pending server-request replay、后续 live event 入队。`thread/start`/`thread/resume` 都复用既有 per-connection bounded writer；缺 threadId fail closed，external runtime append 不再保留 raw transport publish 旁路。R2 增加 detached route + atomic owner claim：断连只取消 unscoped request，thread-scoped callback 保留并在 resume 迁移；R2 独立 target `server_request` 15/15。R3 增加 live-connection guard，断连清双向 subscription。R1 exact connection/request context 2/2，`cargo check -p app-server --lib` 已通过，rustfmt/diff check 通过。raw JSONL `response -> replay -> live` 测试已落，但最新 test binary 被并行 `AgentInput -> RuntimeReplyInput/TurnStartRequest` 迁移的既有编译错误阻塞，暂不标 completed。全部为 `current`，无 compat/deprecated/dead 新增。OPEN_REF：token usage/ThreadGoal canonical replay、raw JSONL Gate 待并行类型迁移汇合后复跑、跨 connection reconnect 集成测试；`app-server/src/lib.rs` 已超过 3600 行，继续接 usage/goal 前必须把 listener actor 与 resume transport sequencer 抽到独立 current module，禁止继续在根文件叠业务逻辑。整体 Codex v1 仍约 38%。 |
| 2026-07-19 | loaded resume MCP migrated-owner 终态骨架（执行中） | `server_request` 不再让 MCP 缓存 reverse request 的初始 connection owner；resolve/error/cancel 现在把实际终态 owner 与结果一起交给 waiter，`cancel_with_owner` 在 RMCP domain close 时原子移除 route 并返回当时 current owner。`mcp_elicitation` 只向 migrated owner 发送 `serverRequest/resolved`；route 仍 detached 且没有 current owner 时 fail closed，不回退旧连接。新增 claimed owner route-removal、resume 后 resolved 精确投递与 domain close 精确投递测试。scoped rustfmt/diff check 已通过；App Server lib check 已越过本刀唯一 `mut pending` 编译修正，仍被并行输入类型迁移错误阻塞，定向测试待该车道汇合后复跑。全部为 `current`，无 compat/deprecated/dead；若未来要求 detached 期间 terminal 在下次 reconnect replay，需要独立 canonical tombstone 设计，本骨架不伪造。 |
| 2026-07-19 | loaded resume token usage 协议/读取研究骨架（OPEN） | 并行审计对照 Codex 增加 typed `thread/tokenUsage/updated`、`ThreadTokenUsage` 与 `TokenUsageBreakdown` 协议/round-trip/schema 注册；`runtime/thread_usage.rs` 仅作为 `cfg(test)` 严格读取 nested total/last/context-window 的研究 helper，不进入生产 binary。审计确认 Lime 现有 `turn.completed.payload.usage` 多为 flat counters，尚无可靠 canonical cumulative total/last lowering；因此未把 usage notification 插入 resume listener，`excludeTurns` 也未新增无效 transport tuple，当前不计生产交付。scoped rustfmt/diff check、协议 v2 定向 18/18、`npm run test:contracts`（740 types 无漂移、299 checks）与治理扫描（0 分类漂移/0 边界违规）通过。下一刀必须先确定 canonical usage producer/持久化形状，再接 `response -> usage -> pending replay -> live`；ThreadGoal 仍独立 OPEN，禁止拿 `ManagedObjective` 冒充。 |
| 2026-07-19 | Renderer queue fixture 双轨清理 | 生产 Renderer 已无 `expectingQueue`、`QueuedTurnSnapshot`、slash-skill preflight 或 queue snapshot callback owner；删除已退役 `agentStreamSlashSkillPreflight` 正向测试，并批量清理 hook 测试夹具中的旧 queue 属性/callback，保留 boundary guard 的负向字符串断言。定向 submit/session 19 文件首轮 146/148，修正两条 stale queue 文案/telemetry 断言后复跑 148/148；listener/readiness/submit-failure/tail-recovery 14/14。`git diff --check`、Prettier 与既有 `governance:legacy-report`（0 分类漂移、0 边界违规）通过。`smoke:agent-runtime-current-fixture` 前置历史 31/31、流式 32/32、fixture guard 74/74、Renderer/Electron 构建通过；sidecar Cargo 重建因共享 `lime-rs/target` 争用导致 `libtokio-*.rmeta`缺失而失败，未形成新的 Gate B 结论。分类：queue 预测/丰富 snapshot/旧 slash preflight 测试为`dead / deleted / forbidden-to-restore`；canonical start/steer、status/count、RuntimeCore pending-work 为 `current`。 |
| 2026-07-20 | loaded resume listener barrier 接入（执行中） | `thread/resume` transport 在请求进入 handler 前准备 per-thread/per-connection barrier；listener 在 barrier 存在时延迟同 thread live event，`CompleteResume`严格按`resume response -> thread-scoped pending server request replay -> deferred live event`发送，并在成功 resume 后原子 claim detached owner。失败路径释放 barrier 且不订阅 thread；shutdown 不重新创建 listener。新增真实`run_json_lines -> initialize -> thread/start -> thread/resume` raw JSONL 回归与 barrier 幂等测试；rustfmt/diff check、`npm run test:contracts`（748/740 schema/types、299 client checks）和治理报告（0 分类漂移、0 边界违规）通过。当前独立 App Server 编译仍被隔壁 `AgentInput -> RuntimeReplyInput/TurnStartRequest`迁移及 runtime backend stale field 诊断阻塞，未宣称定向测试通过。token usage、ThreadGoal、跨连接 reconnect replay 仍为`OPEN_REF`，整体 Codex v1 对齐约 38%。分类：barrier/listener/replay 为 `current`；无 compat/deprecated 新增。 |
| 2026-07-20 | P0-05 provider generation 真实 Gate B 闭环 | 删除自定义 OpenAI-compatible provider 最后一把 key 后，runtime readiness 不再复用允许 keyless model discovery 的判断；`ModelRegistryService::requires_api_key_for_runtime`仅豁免 Ollama、固定本地 keyless provider 与明确 Lime tenant managed host，RuntimeBackend 对缺 key route 返回`missing_enabled_api_key`。真实 Electron Gate B 使用 source-built App Server，完成 parent provider 暂停、Electron IPC key delete、spawn release、child PendingRoute、冷重启、幂等 provider update、key recreate 与 child recovery；最终 25/25 assertions 全部通过，provider generation `5 -> 6 -> 6 -> 7 -> 7`，child provider request 1 次，canonical mailbox Turn/Item 各 exactly-once，console/page/invoke/mock 均为 0，证据为 `.lime/qc/provider-generation-pending-route-gate-b.json`。Gate harness 同步校正 canonical `ItemId`的`item\*` 前缀，并仅从成功 direct request log 投影 method/transport/status，证据不保存 key 参数；Vitest 8/8、`cargo check -p app-server --lib`、Prettier、node check、diff check 通过。定向 Rust test 因并行 `AgentInput -> RuntimeReplyInput` 旧测试 fixture 编译错误未执行到目标断言。`npm run verify:gui-smoke` 已完成 renderer/host/assets 和 21/21 pass evidence，但首次因 Electron child 在 evidence 后未退出而被 launcher watchdog 判 exit 1；`exitElectronSmoke`增加 10 秒有界 App Server stop 后，Electron typecheck、host rebuild 与直接`node scripts/electron/smoke.mjs`exit 0，summary 为`.lime/qc/project-gates/standalone-shell-01-20260719201546-87478/shell-01-electron-smoke/summary.json`。本轮 provider route/recovery 与 smoke shutdown 为 `current`；无 compat/deprecated/dead 新增。P0-05 的本轮 credential generation claim 已关闭，但 loaded resume token usage/ThreadGoal/reconnect 与整体 Codex v1 仍为 OPEN_REF，整体完成度仍约 38%。 |
| 2026-07-20 | ThreadGoal host interrupt drain + response barrier | RuntimeCore 普通 cancel 与 approval cancel 在 active session-loop 下先捕获 per-turn completion 并 interrupt actor；driver 收到 `Interrupted`/`Replaced`completion 后先 drain provider/tool event FIFO，再以同一`AppendingRuntimeEventSink`append/publish`turn.canceled`，最后 signal barrier。response 等待 signal 后返回，但不重复返回已由 hub 发布的 terminal。无 active driver 保留 direct fast-cancel fallback，detached backend cancel 不拥有 durable terminal。受控 active admitted 回归 1/1，证明响应返回后 read model 立即为 Canceled、durable `provider.usage < turn.canceled`、Goal 结算 25 个非缓存 token、terminal replay 不重复增量且 outbox 仅 1 条；session replacement/live action 2/2、permission preflight 7/7、fast cancel 1/1、turn lifecycle 27/27、agent-runtime session loop 37/37、`cargo check -p app-server --tests`、scoped rustfmt/diff check、治理报告 0/0/0 通过。真实 Electron host cancel Gate B 也通过：`turn/interrupt` 后 GUI/read model interrupted，同会话继续 Turn completed，legacy/mock/console/page error 为 0。聚合 fixture 随后在 approval-resume 的 action.required 出现前超时，该场景未进入本轮 cancel 分支；approval-cancel 独立 Gate B 与相邻 approval fixture/projection 仍 OPEN，不宣称整体 Codex v1 完成。 |

### 2026-07-20 loaded-thread listener owner 收敛

- `app-server/src/thread_listener.rs` 成为 per-thread listener actor 的唯一 current owner；外部 runtime event 入队、resume barrier、同一 projector 下的 subscribe/replay/deferred-live 顺序已从 `lib.rs` 抽离。`lib.rs` 从 3603 行降至约 3297 行，未引入第二套 transport writer。
- `prepare_thread_resume` 对重复 `(connectionId, requestId)` barrier fail closed，并补重复准备回归。该修复只收紧状态机，不改变 wire payload。
- raw JSONL Gate 首次复跑暴露两项真实缺口：fixture 未提供 canonical `ProjectionStore`，导致 `thread/start` durable read-back 返回 Error；同一 live connection 在 start 后 resume 的幂等重复订阅被误判为 unavailable。fixture 已改用临时 canonical store，`subscribe_connection` 现在仅在连接不 live 时失败，并保持双向 HashSet 去重。
- 本轮新建/修改全部归 `current`；没有新增 `compat`、`deprecated` 或恢复任何 dead runtime。`thread_state.rs` 继续只负责状态、generation 和 connection index。
- Codex 对照确认的下一刀不是继续堆 transport 字段：先补 runtime instance owner、explicit unsubscribe/idle unload、统一 connection writer sequencer，再接 `tokenUsage/goal snapshot -> pending replay -> live`；resume path/history/override 需在 typed `ResumeThreadOptions` owner 中实现。
- 语义偏差：当前 MCP reverse request 使用 exact owner + reconnect claim，Codex 使用 thread-scoped fan-out + first terminal。该骨架暂列 `current / alignment-open`，不计完全对齐；connection writer 刀必须迁到 Codex thread scope 或给出明确产品排除，禁止两种语义长期并存。
- 验证：当前源码 `cargo check -p app-server --lib` 通过；同一 current test binary 的 listener 3/3、thread_state 5/5、server_request 16/16、MCP elicitation 10/10 通过；scoped rustfmt、`git diff --check` 与 `governance:legacy-report` 通过。全 package `cargo fmt --check` 被并行 `runtime/objectives.rs` import 排序漂移阻塞，本窄写集无格式漂移。
- token usage replay 已进入 production owner：`runtime/thread_usage.rs` 从测试研究 helper 升级为 `RuntimeCore` 的严格 typed snapshot 读取；只有 canonical event 同时具备 nested `total`、`last`、完整非负 counters 与 context window 时才生成 `thread/tokenUsage/updated`，flat/partial usage 继续 fail closed，不伪造累计值。成功 `thread/resume` 现在由同一 listener barrier 严格发送 `response -> token usage -> thread-scoped pending server request -> deferred live event`，历史 usage 仅投递给本次恢复连接，不重新写入 event log。
- token usage 当前验证：`runtime::thread_usage` 4/4、listener 四段顺序 1/1、`resume_barrier` 3/3、`server_request::tests` 16/16；`cargo check -p app-server --lib` 通过（仅既有 dead-code warnings），`npm run test:contracts` 通过（748 schema definitions / 740 TS types / 299 client checks），`npm run governance:legacy-report` 为 0 分类漂移 / 0 边界违规，`rustfmt --check --config skip_children=true` 与 scoped `git diff --check` 通过。`ThreadGoal`、跨 connection raw JSONL reconnect、Codex thread-scoped server-request first-terminal 语义仍为 `OPEN_REF`；整体 Codex v1 对齐仍约 38%。
- reconnect skeleton 已补齐到 current transport owner：`run_transport_events(..., single_client_mode=false)` 覆盖连接 A 断开后的 thread-scoped owner detach、连接 B 的独立 transport initialize、resume 原子 claim、旧 A response `ClientMismatch`、B response exactly-once，以及 `response -> pending replay -> deferred live` 顺序。为使 per-connection transport session 与 process-global runtime 基线不冲突，重复 connection initialize 允许复用已建立的 process baseline，同一 connection 重复 initialize 与未初始化 business request 均 fail closed；server-request JSONL 夹具同步完成真实 initialize handshake。该测试当前是多连接 TransportEvent Gate，尚未宣称跨 socket/raw JSONL Gate。
- 本轮 reconnect 验证：`thread_listener_tests` 3/3、`server_request::tests` 16/16、`runtime::thread_usage` 4/4、dispatch context 2/2、`cargo check -p app-server --lib`、contracts、治理、scoped rustfmt/diff check 全部通过。`ThreadGoal` 审计确认必须新建独立 durable GoalStore、processor 与 listener snapshot/update/clear command；v0 `ManagedObjective` 语义不同，不能作为 ThreadGoal owner，也不做包装或字段映射。剩余下一刀为独立 ThreadGoal current owner 与 raw JSONL reconnect Gate。
- ThreadGoal durable skeleton 已接入 current 主链：canonical ThreadStore 的独立 `thread_goals` 表保存 goal id、objective、status、budget、usage 和时间戳，`runtime/thread_goal.rs` 是 RuntimeCore owner，`processor/thread_goal.rs` 只负责 v2 `thread/goal/set|get|clear`、response 与 `updated/cleared` notification；未复用或映射 `ManagedObjective`。首次 set 必须提供非空 objective，编辑保留 createdAt/usage，clear 幂等，冷重启从同一 state DB 读回。
- resume barrier 顺序已扩为 `response -> token usage -> ThreadGoal updated/cleared snapshot -> pending server request -> deferred live`；无 goal 也显式 replay `thread/goal/cleared`，避免 GUI 保留旧状态。真实 stdio reconnect 场景验证已有 goal 在新连接返回 updated snapshot 后才 replay pending request，并继续保持旧连接 `ClientMismatch` 与 live event 后置。
- ThreadGoal 骨架验证：public JSON-RPC lifecycle/cold restart `1/1`，listener resume/reconnect `4/4`，`cargo check -p app-server --lib` 通过；完整 `npm run test:contracts` 通过（764 schema definitions / 756 generated protocol types / 299 client checks，含 command/harness/modality/scripts/Electron/docs guards），`governance:legacy-report` 为 0 分类漂移 / 0 边界违规，scoped rustfmt 与 `git diff --check` 通过。额外单条内部顺序测试因会再次触发约 28 分钟 lib test 重链接而中止，已有四条 listener 测试覆盖同一断言。分类全部为 `current`，无 compat/deprecated/dead 新增；goal usage accounting、自动 continuation、analytics/GUI structured owner 仍为 `OPEN_REF`，整体 Codex v1 对齐仍约 38%。
- ThreadGoal live mutation notification 已进入同一 per-thread listener FIFO：transport 从 processor 结果中分离 `updated/cleared`，先向 origin 写 response，再由 listener 向 thread subscribers fan-out 并补未订阅 origin；origin 已订阅时按 connection id 去重。goal notification 与 runtime event 共用 resume deferred queue，最后一个 barrier 完成后按入队顺序 flush；`clear=false` 不生成通知。单个 stale subscriber 投递失败现在只记录并移除该 thread subscription，不再在 mutation 已持久化、response 已发送后通过 `streamed_tx Err` 终止全局 transport。actor 与 raw stdio listener 回归 `6/6`，`cargo check -p app-server --lib`、contracts（764 schema / 756 generated types / 299 client checks）、治理（0 零引用候选 / 0 分类漂移 / 0 边界违规）、scoped rustfmt/diff check 全部通过。分类全部为 `current`，无 compat/deprecated/dead；整体 Codex v1 对齐仍约 38%。
- ThreadGoal accounting durable primitive 已落在 current GoalStore：复制 Codex 的四种 status mode、negative delta clamp、双零 unchanged、exact internal goal id 条件、SQLite immediate transaction 原子增量，以及达到 token budget 时同一 UPDATE 切换 `budgetLimited`；同一 in-flight goal 在 budgetLimited 后仍可继续累计。store 回归拆开验证 paused filter、旧 goal id 防串账、预算跨越和持续累计，RuntimeCore 只暴露窄 accounting/identity 接口。当前真实 `turn.completed.usage` 仍是 flat provider counters，不满足 `runtime/thread_usage.rs` 的严格 cumulative total/last/context-window contract，因此未把不可信 usage 接入 canonical projection，也未伪造 live notification；下一刀固定为 canonical usage producer -> durable per-turn baseline/watermark -> `ProjectionStore::apply_canonical_events` 幂等接线 -> listener 内 `turn/completed -> thread/goal/updated`。`cargo check -p app-server --tests` 已在并行 let-chain 修正汇合后通过，编译包含本轮 inline 回归；因 App Server lib test 单次重链接约 28 分钟，本刀未重复链接执行。scoped rustfmt/diff check 与 SQLite RETURNING 语义检查通过。分类全部为 `current`，无 compat/deprecated/dead；整体 Codex v1 对齐仍约 38%。
- ThreadGoal terminal accounting 已接入 current canonical 主链：复用隔壁已产出的 `runtime/thread_usage.rs` cumulative lowering 与 `canonical_thread_store/goal_accounting.rs` durable primitive，没有重写 `event_store.rs` 或 accounting owner；新增薄 `goal_projection.rs` 以生产稳定的 `turn.accepted` 绑定 exact goal/Plan/cumulative-token/wall baseline，`turn.completed|failed|canceled` 在存在完整 cumulative snapshot 时原子推进 source watermark、usage、`budgetLimited` 与 durable update outbox。Plan mode 同时读取 typed runtime metadata 和 current `additionalContext.metadata` 应用元数据；partial/untrusted usage、无 admission baseline、goal identity 已变化均 fail closed。per-thread listener 现在严格发送 `terminal -> thread/goal/updated`，成功 fan-out 后确认 outbox，exact terminal replay 不重复发 goal update；旧的无 watermark accounting primitive 与重复测试已删除，不保留双轨。最终共享树复验：`cargo check -p app-server --tests`、goal projection `2/2`（含 failed/canceled exactly-once）、accounting `7/7`、listener `7/7` 通过；此前 state table owner inventory `1/1`、contracts（764 schema / 756 generated types / 299 client checks）、治理（0 零引用候选 / 0 分类漂移 / 0 边界违规）、scoped rustfmt/diff check 亦通过。架构图确认：`internal/aiprompts/architecture.md` 已更新，责任开发者 root，2026-07-20。分类：baseline/watermark/outbox/projection/listener 为 `current`；旧 primitive 为 `dead / deleted`；无 compat/deprecated。生产 failed/canceled usage producer 完整覆盖、turn 中途 goal rebind、tool-finish/abort flush、idle accounting、outbox crash-drain、自动 continuation 与 GUI structured owner 仍为 `OPEN_REF`，整体 Codex v1 对齐仍约 38%。

- Codex typed `CollaborationMode` 已从 admission 骨架收口到 thread settings 和 Renderer submit current owner：`agent-protocol` 的 `ModeKind/CollaborationMode/CollaborationModeSettings` 贯通 v2 `turn/start`、`thread/settings/update`、`RuntimeRequest`、`TurnContextOverride`、queued-turn intent、Goal Plan admission、`tool-runtime` Plan gate 与 scheduler fixture；thread settings 持久化、冷重启恢复和普通 `model/effort` 局部更新均按 Codex `with_updates` 语义刷新当前 mode，损坏 persisted mode fail closed，developer instructions 由 typed settings 直接承接。Renderer 的 `HandleSendOptions.collaborationMode` 是唯一 mode 输入，Inputbar、EmptyState 与 plan implementation adjustment 直接发送 typed `"plan"`，submit preparation/submission/execution 构造完整 Codex `mode + settings` wire；业务 metadata 只走 typed `additionalContext`，生产代码不再读写 `requestMetadata.harness.collaboration_mode`，implementation accept 回到 default mode。Renderer `AppServerClient` 同步公开 `updateThreadSettings` 与 `setThreadMemoryMode`，不再由 Plugin 绕过 typed gateway。验证：agent-protocol `1/1`、App Server typed unit `6/6`、`thread_control_jsonrpc` `5/5`、Renderer typed send chain `187/187`、EmptyState `62/62`、thread settings projection `7/7`、Claw current fixture Vitest `65/65`、App Server gateway + Plugin `43/43`、`cargo check -p app-server --tests`、schema/protocol types `759/759`、`npm run typecheck`、完整 `npm run test:contracts`（含 `299` client checks 和 docs boundary）、`npm run governance:legacy-report`（`0/0/0`）与 `git diff --check` 均通过；`rg "collaboration_mode" src packages scripts` 只剩 contract absent snippet 和两条负向断言。为汇合隔壁 `RuntimeSessionTaskFailure.reason_code` 最小补齐 3 个 `None` 初始化点。分类：typed mode、settings persistence、Renderer typed submit 与 typed gateway 为 `current`；旧字符串 mode、metadata 双轨和协议外顶层 metadata 为 `dead / deleted / forbidden-to-restore`；整体 Codex v1 对齐仍约 38%，下一刀回到 provider abort/cancel usage、tool-finish/abort flush、idle accounting、outbox crash-drain 与自动 continuation。

- 2026-07-20 并行 current fixture 复用结果：隔壁进程运行 `claw-chat-current-fixture-smoke.mjs --scenario image-command --keep-temp`，本进程未重复启动、未终止且未夹写 harness。Gate B 已真实经过 Electron、App Server runtime、图片生成、read model 和 GUI，图片卡片、预览与完成文案均可见，但 summary 明确为 `ok=false`，终态断言仅因 `hasTokenUsage=false` 失败；证据为 `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-summary.json`。该失败归入 usage/GUI structured owner 的 `current / alignment-open` blocker，不回退 mock、不为通过断言补兼容；typed collaboration mode 的定向收口不受影响，但当前共享树尚未达到 Agent 主链可交付门槛。

- 2026-07-20 provider usage 中间 source 已收敛：`CurrentProviderTurnEvent::Usage` 携带 `attempt`，由 `AgentEvent::ProviderUsage` lowering 为 durable `provider.usage`；同 attempt snapshot 替换、跨 attempt 累加，`provider.step` 不再进入 canonical usage 计费，正常 completion 不双计。生产 EventLog 回归确认 `turn.accepted` 持久化 `goalAccountingMode=plan`。验证：agent-runtime 18/18、lime-agent 21/21、thread usage 8/8、tool lowering 15/15、goal projection 2/2、goal accounting 8/8、listener 8/8、`cargo check -p app-server --tests`、全 workspace rustfmt check、diff check、治理 0/0/0 通过。current fixture 前置 31/31、32/32、76/76 通过，真实 Electron 热路径完成 backend、GUI/read model 与 Gate B trace，但性能 trace 未捕获 pre-turn `turnStartAt`，在 `homeHotpathPreTurnTraceWindowAvailable` 退出 1；console/invoke/page error 为 0，本轮未夹写共享 GUI harness。新增 provider usage / outbox 顺序测试均属于 `current`；原 flat/provider-step usage source 为 `dead / deleted`，无 compat/deprecated。仍 OPEN：provider abort/cancel flush 的真实 integration、tool-finish/abort flush、idle time、turn 内 goal rebind、自动 continuation、启动期 outbox drain。

- 2026-07-20 structured failed goal terminal 已接入 current admission owner：`RuntimeReplyAttemptErrorKind::UsageLimitExceeded` 经 session loop `reason_code` 保留到 `turn.failed.payload.reason=usage_limit_exceeded`，普通执行错误稳定为 `turn_error`；GoalStore 同事务将 Active 普通失败转 `blocked`，将 Active/BudgetLimited usage limit 转 `usage_limited`，Plan turn、cancel 与 BudgetLimited 普通失败保持原状态。端到端测试进一步发现 synchronous `RuntimeCore::start_turn` 的 Collecting 分支在 backend 已发 `turn.started` 时缺少 durable `turn.accepted`，现由 RuntimeCore 在首个 Turn lifecycle 进展前补齐且抑制 backend 重复 accepted；零事件 `UnavailableBackend` 仍回滚，不持久化假 Turn。验证：read-model/failed 7/7、turn lifecycle 27/27、goal projection 3/3、goal accounting 10/10、`cargo check -p app-server --tests`、scoped rustfmt/diff check 与治理 0/0/0 通过。分类全部为 `current`，没有新增 compat/deprecated；structured failed/usage-limit accounting 缺口已关闭。仍 OPEN：provider abort/cancel 完整 usage flush、tool-finish/abort flush、idle time、turn 内 goal rebind、自动 continuation、启动期 outbox drain 与 GUI structured owner；整体仍不能宣称完整 Codex v1 对齐。

- 2026-07-20 ThreadGoal late creation 已接入 current RuntimeCore/GoalStore owner：Turn accepted 时没有 Goal 不再永久失去 accounting；运行中首次创建 Active Goal 会在 RuntimeCore state 锁内读取 executing Turn、Plan mode、canonical cumulative usage、source watermark 和 mutation 时间，并与 Goal 写入同一 SQLite Immediate 事务完成 late-bind。已有 binding 不被 accepted replay 或重复 set 重置，Queued Turn 不得抢占 executing Turn binding；真实 session-loop 集成覆盖 `accepted -> provider usage -> set -> provider usage -> terminal`，只计 set 后 15 tokens。验证：RuntimeCore late-bind/queued-turn selection 2/2、goal projection 4/4、goal accounting 10/10、`cargo check -p app-server --tests` 与 scoped diff check 通过。分类全部为 `current`，未新增 compat/deprecated/dead；仍 OPEN：已有 Goal pause/resume、clear/replacement 前的旧增量 flush/rebind、provider/tool abort flush、idle time、自动 continuation、启动期 outbox drain和 GUI structured owner。

- 2026-07-20 已有 ThreadGoal external mutation 已完成 Codex 顺序收敛：RuntimeCore set/clear 都以 state mutex 与 event append 线性化，GoalStore 在单一 SQLite Immediate 事务内执行 exact old-goal progress flush、set/clear、active baseline reset 或旧 binding 删除；active objective patch 立即返回 mutation 前 usage，pause/resume 不计 paused 区间，clear/recreate 将同一 Turn 改绑新 goal id。accounting 核心与 terminal outbox wrapper 已拆开复用，mutation flush 不写旧 snapshot outbox，由现有 set/clear response + listener notification 发送唯一最终状态；同 sequence 只允许 external mutation 推进 wall time，stale/terminal rebind fail closed，后半事务失败会回滚 goal usage、metadata 与 baseline。验证：goal projection 8/8、goal accounting 10/10、RuntimeCore Goal mutation 3/3、GoalStore 2/2、thread listener 7/7、`cargo check -p app-server --tests`、scoped rustfmt/diff check 与治理 0/0/0 通过。分类全部为 `current`，无 compat/deprecated/dead 新增；仍 OPEN：provider/tool abort 完整 usage flush、idle accounting、自动 continuation、启动期 outbox drain 与 GUI structured owner。

- 2026-07-20 ThreadGoal 成功 tool-finish 与 provider cancel 竞态已按 Codex current 语义补齐：canonical `item.completed` 仅对成功的 Tool/MCP/Collab/Command 结算当前 turn 已 durable 的最新 usage/time，复用 exact goal/source watermark 与 outbox transaction；per-thread listener 严格发送 `item/completed -> thread/goal/updated` 后做 server-enqueue ack，accounting/source replay 不双计。provider stream 若同一次 poll 同时返回 Usage 并观察到 cancel，会先发 attempt-scoped `provider.usage`，随后按 canceled 终止，不继续处理文本/工具且不伪造 `ProviderStep`。失败工具在 canonical Item 具备 typed `handler_executed` 前 fail closed，不把 blocked/denied/aborted 误计为进展。Codex ext/goal 没有 durable outbox 或 startup drain；其 crash recovery 是每次 `thread/resume` 从 SQLite 重发 latest Goal/Cleared snapshot。Lime 已用真实 stdio reconnect 证明等价且更强的 `resume response -> latest Goal snapshot -> captured watermark ack -> pending request -> deferred live`，不新增无订阅 startup broadcast 或历史逐条 replay；DB/source accounting exactly-once，live notification 仅保证 server enqueue，resume 是允许重复的 latest-state convergence。验证：provider turn 20/20、goal projection 10/10、thread listener 8/8、新竞态 1/1、scoped rustfmt 与 diff check 通过。分类全部为 `current`；成功 tool-finish、Usage/cancel same-poll、resume crash recovery 已关闭，仍 OPEN：failed tool typed execution fact、idle accounting、自动 continuation与 GUI structured owner。

- 2026-07-20 图片消息 usage live/reload blocker 已关闭：`thread/tokenUsage/updated` 由 `useAgentSession` 的 session-scoped turn usage owner 消费，随 session 切换清空，并按 thread/turn 精确过滤；消息替换只通过纯 `previous -> snapshot` merge 与 runtime turn identity 继承 usage，模块级全局 turn/task Map 已删除。`agentStreamRuntimeHandler` 在 token notification 落消息时同时绑定精确 `runtimeTurnId`，图片 snapshot 继续由既有 image preview owner 合并。定向验证：usage merge 6/6、stream handler 56/56、image preview 9/9、snapshot stability 3/3、图片 footer 14/14、Renderer typecheck 与 diff check 通过。Gate B 证据 `.lime/qc/gui-evidence/claw-chat-current-fixture/release-v1.108.0-gate-b-session-usage-summary.json` 同时证明 live 终态和 reload 后均显示 `31.0K Tokens`，Electron/preload/IPC/App Server/runtime/read model/GUI identity 对齐；整场仍因独立断言 `imageCommandWorkflowAuditReadModelProjected` 为 `ok=false`，不得标记 image-command 全场通过。`smoke:agent-runtime-current-fixture` 另因既有 `homeHotpathPreTurnTraceWindowAvailable` 性能窗口失败。分类：session usage owner、turn binding和纯 merge 为 `current`；模块级全局缓存为 `dead / deleted`；无 compat/deprecated。下一刀固定为收敛 image workflow audit read-model projection 与 home-hotpath pre-turn trace，整体 Codex v1 对齐仍约 38%。

- 2026-07-20 tool abort / host cancel 复核收窄了上一条 accounting 结论：provider poll-level Usage/cancel 竞态已关闭，但不等于 App Server host abort 全链关闭。Codex 以独立 `ToolCallOutcome::Aborted` 仲裁 host cancel，Goal 不计 Aborted；Lime 已在 `RuntimeToolExecutionOutcome::Aborted` 建立同一赢家语义，并以 `tool_outcome=aborted` + `ItemStatus::Cancelled` 投影到 canonical Item。Goal 因此恢复计 genuine `Failed { handler_executed: true }`，但不计 Aborted/blocked/denied。RuntimeCore host cancel 仍需在 fast-cancel 合同下建立 `cancel signal -> drain current provider/tool accounting events -> durable turn.canceled -> listener terminal/goal update` 顺序。Codex 没有 startup outbox drain；Lime crash convergence owner 固定为 resume latest snapshot + captured-watermark ACK，startup drain 从 parity blocker 移除，仅保留 retention/GC 运维项。验证：tool lifecycle 3/3、dispatch 8/8、tool executor 7/7、canonical Aborted Item 1/1、provider turn 20/20、Lime current provider turn 21/21、Goal projection 11/11、listener 8/8、四 crate `cargo check --tests`、scoped rustfmt、diff check 与治理 0/0/0 全部通过。当前 OPEN 统一为 host abort/interrupt ordering、idle accounting、自动 continuation和 GUI structured Goal/terminal owner。

- 2026-07-20 typed approval cancel Gate B 已按 Codex reverse server request 骨架闭环：durable `action.required` 先进入 canonical pending descriptor，App Server 再以 `item/commandExecution/requestApproval` 向唯一 Electron client 发 typed reverse request；Renderer 不再消费已退役 raw action side-channel。用户选择 Cancel 后，App Server 将 typed response 映射回 RuntimeCore `AgentSessionApprovalDecision::Cancel`，等待既有 actor interrupt/driver barrier 完成，再把已持久化的 `action.resolved -> item.completed -> turn.canceled` 按原 FIFO 交给 thread listener 发布，避免 GUI 依赖 recovery poll。canonical Approval 继续作为 durable control item，但因 Codex v2 `ThreadItem` 没有 Approval variant，`thread/read` 与 `thread/items/list` 仅过滤这一已知 out-of-band item；未知 Extension 仍 fail closed，pending 详情由 `agentSession/action/replay` 读取。fixture 同步改用真实 canonical `threadId`，pending 断言组合 `thread/read waitingOnApproval` 与 durable replay descriptor，terminal read 只验证 interrupted Turn、tool identity、pending 清空和无 tool result，requestId/decision 由 reverse response及 backend ledger 独立证明。验证：Approval v2 filter `1/1`、Renderer reverse request controller `2/2`、current fixture unit `69/69`、`cargo check -p app-server --tests`、脚本语法/rustfmt/scoped diff check 与治理 `0/0/0` 通过；真实 Electron Gate B `.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-approval-request-cancel-server-request-root-7-summary.json` 为 `ok=true`，全部 assertion 通过，Electron/preload/IPC/App Server/runtime/read model/GUI identity 一致，console/page/invoke error 与 legacy/mock 命中均为 0。`npm run test:contracts` 的 protocol generation 为 `762 types / 0 failures / no drift`，随后被并行 v2 重构的 6 条 stale snippet 阻塞：3 条旧 capability deny 断言、2 条旧 stdio event helper 和 1 条生成 TS `ServerRequest` 文本形状；本刀不恢复已迁出的旧实现。分类：typed reverse request、durable pending descriptor、Approval out-of-band projection和 cancel barrier 为 `current`；raw `action.required` Renderer side-channel 为 `dead / retired / forbidden-to-restore`；无 compat/deprecated。OPEN_REF：将 browser/tool confirmation 从 command approval skeleton 拆到 Codex 对应 typed tool/permissions server request、host interrupt 主动 abort pending server request并发送 `serverRequest/resolved`、approval resume/decline/reconnect Gate B；整体 Codex v1 对齐仍约 38%。

- 2026-07-21 typed interaction server request 已从 command 单路扩为 Codex 三路：新增 `item/fileChange/requestApproval` 与 `item/tool/requestUserInput`，command/file decision wire 统一为 camelCase `accept|acceptForSession|decline|cancel`，不保留 PascalCase 兼容。`apply_patch` 绑定既有 FileChange tool call item；`request_user_input` 从错误的 MCP elicitation response kind 改为 AskUser，保留 `autoResolutionMs` 与原 tool call item id。Renderer 三路共用唯一 dispatcher，command/file/AskUser 找不到 typed pending 时 fail closed，不再回退 `agentSession/action/respond`。schema/client 当前为 778 definitions / 770 generated protocol types。验证：workspace `cargo check -p app-server --tests`、v2 protocol 15/15、schema fixture 1/1、tool request-user-input 8/8、agent-runtime request-user-input 5/5、Agent bridge 2/2、Renderer dispatcher/controllers 15/15、Renderer/Node typecheck、protocol drift、command/harness contracts及 legacy governance 0/0/0 均通过；完整 `test:contracts` 只剩共享 `lib.rs` 拆分造成的 5 条既有 stale snippet，本轮多 variant `ServerRequest` 守卫已修正。分类：三路 typed protocol/producer/dispatcher 为 `current`；App Server 内部 `respond_action` session response boundary 与剩余 legacy elicitation 为 `deprecated`；PascalCase decision、Renderer command/AskUser fallback 为 `dead / deleted / forbidden-to-restore`。FileChange 与 AskUser 尚未跑独立 Gate B；下一刀固定为 v2 `serverRequest/resolved`、FileChange Declined/Cancelled terminal 和多文件 Add/Delete/Move canonical projection，再删除内部 v0 action DTO。整体 Codex v1 对齐仍约 38%。

## 7. 2026-07-21 typed approval server-request lifecycle 骨架

- App Server typed command/file/AskUser reverse request 统一保留 `PendingServerRequest` 的 outer id 与 owner，收到 response 后先通过 thread listener FIFO 发布并等待 `serverRequest/resolved`，再交给 RuntimeCore `respond_action`。转场错误（`REQUEST_CANCELLED`）发布 resolved 后直接结束，不产生 late `action/respond`；未知 Extension 继续 fail closed。
- Renderer 仅在已有 Claw debug override 开启时写入脱敏 lifecycle buffer：`request -> response -> resolved -> terminal`，只保留 outer id、method、thread/turn/item/approval identity 和 decision，不写入 command、cwd、prompt 或响应正文。fixture 以 canonical thread/turn identity 选择 resolved 后的 terminal，避免把 approval 前已完成的用户 item 误判为 continuation terminal。
- Gate B 真实证据已通过：
  - resume：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-approval-request-resume-server-request-resolved-root-v6-summary.json`；允许后第二 Turn session cache 命中且 GUI 不再弹审批。
  - decline：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-approval-request-decline-server-request-resolved-root-summary.json`；拒绝后 Turn 正常 completed，不产生 canceled。
  - cancel：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-approval-request-cancel-server-request-resolved-root-v2-summary.json`；取消后 Turn interrupted/canceled 语义、pending 清零且无 tool result。
- 定向验证：`cargo test -p app-server approval_server_request::tests::`（6/6）；`src/lib/api/appServerServerRequest.unit.test.ts` + current fixture smoke（79/79）；current fixture smoke 脚本语法、Prettier、Renderer lifecycle trace 单测通过；`npm run typecheck` 通过。`npm run test:contracts` 的协议生成通过 `770 types / 0 failures / no drift`，后续仍被并行 v2 已迁出实现的 5 条 stale snippet 守卫阻断（capability deny 3、stdio helper 2），不恢复旧实现。
- 分类：typed reverse request、resolved FIFO barrier、脱敏 debug evidence、thread/turn scope assertions 为 `current`；raw action side-channel 为 `dead / retired / forbidden-to-restore`；无新增 compat/deprecated。OPEN_REF：把 host `turn/interrupt` 的 pending route 精确 abort 与 durable action terminal 完整接入；将 `ServerRequestResolvedNotification` 从现有 `requestId` 形状补齐 Codex `threadId + requestId`，并为 FileChange/AskUser 独立 Gate B。

- 2026-07-21 host interrupt pending-route skeleton：`ServerRequestRouter` 现从 typed reverse request params 保留 `threadId + turnId`，并提供精确 `cancel_for_thread_turn`。它会 detach 目标 route、向原 waiter 传递 `REQUEST_CANCELLED`、拒绝 late response，且不影响同 thread 的其它 Turn 或其它 Thread；定向 router/typed approval 测试 23/23 通过。该 helper 仍未接入共享 `lib.rs` terminal event pump：接线必须在发布 `turn.canceled` 前等待相同 route 的 `serverRequest/resolved` FIFO completion，并持久化 `action.canceled`，否则不得标为 host interrupt 完成，也不得以无序 cancel 代替该屏障。分类为 `current / alignment-open`，无 compat/deprecated。

- 2026-07-21 host interrupt current wiring：`RequestProcessor` 的 v2 `turn/interrupt` 现在先调用 App Server interrupt hook；hook 按 canonical `threadId + turnId` detach reverse routes，经 thread listener FIFO 等待 `serverRequest/resolved` 完成后才唤醒 approval waiter，因此 transition 不再重复发布 resolved，也不会被 reconnect replay。RuntimeCore 在 terminal 前持久化 `action.canceled`；若存在 active tool item，再按 `action.canceled -> item.completed(status=cancelled) -> turn.canceled` 写入 canonical history，`agentSession/action/replay` 不再返回 pending action。验证：router 23/23、pending descriptor 2/2、new waiting-action cancel 1/1、既有 cancel 2/2、App Server `cargo check --lib` 与 diff check 通过。当前仍缺真实 Electron Gate B `turn/interrupt` evidence，以及 protocol `ServerRequestResolvedNotification` 的 `threadId + requestId` schema/client 收口；分类为 `current / alignment-open`，无 compat/deprecated。

- 2026-07-21 resolved notification protocol source 收口：current v0 wire DTO、typed approval、host interrupt router 与 MCP elicitation producer 已全部补齐 Codex `threadId + requestId`，MCP terminal 场景 10/10、protocol crate 61/61、App Server compile 通过。`schema/json/**`、manifest 与 generated TS 处于并行脏热区，schema writer 会先删除并重建整个目录；本进程按协作约束未重写他人产物。OPEN_REF：由持有 protocol/schema 写集的进程执行 fixture regeneration、`npm run generate:protocol-types`、contract/typecheck，并让 Renderer resolved parser 使用 mandatory `threadId` 绑定 scope；在此之前不把 schema/client 视为完成。分类为 `current / alignment-open`，无 compat/deprecated。

- 2026-07-21 Renderer resolved scope 收口：`appServerServerRequest` dispatcher 与共享 `AppServerEventBus` 均要求 `serverRequest/resolved.params.threadId` 为非空字符串；缺 scope 的通知 fail closed，不中止 in-flight handler，也不写 resolved tombstone。脱敏 lifecycle trace 的 resolved entry 记录 `threadId`。定向 Vitest：dispatcher/event bus 21/21；Renderer/Node typecheck、Prettier 与 `git diff --check` 通过。`check:protocol-types` 对当前生成物无漂移，但 schema/manifest/generated TS 仍保留旧的 requestId-only 形状，因并行热区未夹写；完整 `test:contracts` 继续被并行 v2 已迁出实现的 5 条 stale snippet 阻塞，真实 Electron host-interrupt Gate B 也仍 OPEN。分类为 `current / alignment-open`，无 compat/deprecated。
- 2026-07-21 临时 schema 生成证据：使用当前 `app-server-protocol` source 写入 `/tmp/lime-schema-ITwQmD`，`v0/ServerRequestResolvedNotification.json` 已生成 `threadId` 属性并将 `requestId`、`threadId` 同时列为 required；仓库 `schema/json/**` 与 generated TS 未被本进程重写。该证据确认剩余缺口只是共享生成物合并窗口，不是 Rust source 语义缺失。
- 2026-07-21 Renderer resolved 复合身份收口：dispatcher 的 in-flight/settled key 与共享 EventBus 的 pending/seen/resolved tombstone key 均从 requestId-only 改为 canonical `(threadId, requestId)`；错误 thread 的 resolved 不得 abort handler，也不得吞掉同 outer id 的另一条 request。缺 threadId 继续 fail closed。验证：dispatcher/EventBus、command/file approval、AskUser、MCP elicitation 五组单测 37/37，Renderer/Node typecheck、Prettier、scoped diff check 通过。分类为 `current`，无 compat/deprecated。
- OPEN_REF：v2 `turn/interrupt` 当前先通过 `thread/read` 解析 session 并校验 turnId 非空，随后立即执行 pending-route abort hook；RuntimeCore 对 turn 是否存在的检查发生在 hook 之后，且没有独立的“属于该 thread 且仍可中断”preflight token。真实 Gate B 前应由 Rust owner 把 preflight 与 cancel 绑定到同一 thread-serialized current 边界，保证无效/已终态 turn 不会提前发布 resolved 或 detach route；本进程因 processor/runtime 处于并行大改热区未夹写。
- 2026-07-21 当前树验证：`cargo check -p app-server --lib`、App Server router/approval 23/23、waiting-action cancel 1/1、`cargo test -p app-server-protocol --lib` 61/61 通过；`cargo test -p app-server-protocol --test schema_fixtures` 明确失败，提示落盘 `json/app_server_protocol.schemas.json` 与当前 Rust source 生成结果不一致。该失败归属共享 schema/generated 写集，不是本轮 Renderer 或 Rust source 改动引入。
- 2026-07-21 真实 Electron approval-cancel 复验：`npm run smoke:claw-chat-current-fixture -- --scenario approval-request-cancel --prefix resolved-scope-cancel-v1 --timeout-ms 180000` 通过，summary 为 `.lime/qc/gui-evidence/claw-chat-current-fixture/resolved-scope-cancel-v1-summary.json`；Electron/preload/IPC/App Server/runtime/read model/GUI 全链 assertions 为 true，`serverRequest/resolved -> item.completed` 顺序、pending 清零、interrupted Turn、无 tool result、legacy/mock/console/page/invoke error 均通过。脚本单测 68/68 通过。该证据仍只把 resolved 投影为 method/requestId，未保留 `threadId`，因此 Gate B 功能闭环通过，scope 字段证据仍 OPEN，脚本 owner 需补投影和断言。
- 2026-07-21 真实 Electron approval-resume 复验：`npm run smoke:claw-chat-current-fixture -- --scenario approval-request-resume --prefix resolved-scope-resume-v1 --timeout-ms 180000` 通过，summary 为 `.lime/qc/gui-evidence/claw-chat-current-fixture/resolved-scope-resume-v1-summary.json`；首轮 response/resolved/terminal 顺序、pending 清零、第二 Turn session-cache 命中、无第二审批 prompt、GUI/read model 完成及无 legacy/mock/console/page/invoke error 全部为 true。summary 同样只投影 resolved method/requestId，scope 字段证据继续 OPEN；当前复验确认复合 `(threadId, requestId)` renderer key 未破坏正常审批和 reconnect/session-cache 主链。
- 2026-07-21 Gate B scope evidence 收口：`approval-trace` summary 现保留 resolved `threadId`，`approval-assertions` 强制它与 request/response/current thread 一致，smoke unit fixture 已同步。三条独立真实 Electron 证据均通过：`.lime/qc/gui-evidence/claw-chat-current-fixture/resolved-scope-cancel-v2-summary.json`、`resolved-scope-resume-v2-summary.json`、`resolved-scope-decline-v2-summary.json`。三份 summary 均证明 `request.threadId == response.threadId == resolved.threadId`；cancel 为 interrupted，resume 第二 Turn session-cache 命中，decline 为 completed 且非 canceled。脚本单测 68/68、node check、Prettier、scoped diff check 通过。此前“resolved scope 字段证据 OPEN”已关闭。
- scope evidence 收尾验证：Renderer dispatcher/EventBus、command/file approval、AskUser、MCP elicitation 与 current fixture smoke 六组共 105/105 通过；`npm run test:contracts` 的 protocol type generation 继续为 770 types / 0 failures / no drift，整体仅保留共享 Rust 重构的 5 条既有 stale snippet（capability deny 3、stdio event helper 2），本刀未新增 contract failure。

- 2026-07-21 ThreadGoal automatic continuation current 骨架：删除旧 `ManagedObjective` 最多 8 次同步自动循环及其 owner，续跑唯一收敛到 `runtime/thread_goal_continuation.rs`。Codex `InternalModelContextFragment(source="goal")` 对应为 durable `thread.goal.continuation`，当前 Turn agent-only 输入不重复进入 provider history，未来 Turn 可恢复；canonical Item 和 v2 notification 均不投影用户消息或 raw side-channel。continuation 改为 admission-only，driver 挂在现有 Tokio runtime，Completed 后异步触发下一轮；per-session single-flight 防重复 admission。terminal idle scheduler 先等待 queued 用户 Turn / TriggerTurn mailbox admission-or-empty，再启动 Goal；pending-owned Turn 使用独立 input kind，避免重入同一 mailbox gate。idle `thread/goal/set(active)` 与无 live Turn 的 `thread/resume` 已接 generic idle gate；set 严格在 response 与 `thread/goal/updated` listener completion 后 admission，resume 复用 barrier 保证 response/snapshot/live 顺序。Plan 跳过，failed 由既有 projection 转 blocked/usageLimited，显式 cancel 不立即续跑。验证：ThreadGoal 8/8、queue 15/15、pending recovery 5/5、Turn lifecycle 29/29、thread listener 8/8、thread v2 JSON-RPC 9/9、`cargo check -p app-server --tests` 通过。隔壁未跟踪 `thread_goal_continuation_jsonrpc.rs` 仍断言旧的“set 后先跑用户 Turn”语义，本进程按窄写集未修改；该测试需由 owner 更新为 idle set immediate continuation 后复跑。分类：ThreadGoal continuation/event/provider history/pending priority 为 `current`；旧 ManagedObjective auto loop 为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated 新增。OPEN_REF：fork deferral、完整 resume/reconnect 自动续跑 evidence；idle wall-clock accounting 已由下一条关闭。

- 2026-07-21 ThreadGoal idle wall-time accounting current 骨架：新增 `canonical_thread_store/goal_idle.rs` 作为唯一进程内 baseline/permit owner，`ProjectionStore` clone 共享 `active_goal_id + last_accounted_at`，durable usage 继续只写 canonical `thread_goals`。set/pause/clear 在 mutation 前 flush；非 Plan `turn.accepted` 在同一 SQLite Immediate 事务内完成 idle seconds、typed outbox 与 exact Goal Turn bind，token delta 固定为零，失败注入证明三者整体回滚；snapshot 按 accepted timestamp 截断，2 秒投影延迟回归证明 idle/per-turn 不重叠；Plan admission 丢弃 prior idle。accepted replay 不重扣、不清当前 baseline，也不覆盖 turn 中途 late-bind baseline。terminal 只有 persisted Goal 仍 active 且非 Plan 时 re-arm；paused/blocked/usageLimited/budgetLimited/complete/Plan 清 baseline。同进程重复 resume 保留同 Goal baseline，cold `ProjectionStore` resume 才从当前时刻重置，进程离线时间不计入。单 permit 覆盖 snapshot -> SQLite write -> mark accounted，并发 clone-store mutation 回归证明只结算一次。验证：`idle_` filter 当前树 11/11；canonical Goal accounting/projection 28/28；continuation 5/5；RuntimeCore Goal 4/4；`cargo check -p app-server --lib`、scoped rustfmt、`git diff --check`、治理扫描（0 零引用/分类漂移/边界违规）通过。完整 `cargo check -p app-server --tests` 当前被隔壁 thread-delete 收口中的旧 `session_archive_jsonrpc.rs` 引用已删除 `METHOD_AGENT_SESSION_DELETE` 阻塞，本进程不恢复 legacy 常量、不夹写其测试；在该并行变化前的合并工作树完整 check 已通过。并行协调：测试期间隔壁 thread-delete owner 先落 `mod` 再落文件、随后修正 sibling visibility，产生两次瞬时编译失败；本进程未补其文件、未改其权限。隔壁 `thread_goal_continuation_jsonrpc.rs` 与 `interrupted` 仍未触碰。分类：idle accounting/baseline/outbox/resume hook 为 `current`；无 compat/deprecated/dead 新 surface。ThreadGoal idle accounting OPEN_REF 已关闭；下一刀是 public fork 后从 canonical Thread/Turn/Item 低层重建跨重启 provider history，继续避开 raw event 复制与并行 runtime 热区。整体 Codex v1 对齐仍约 40%。

- 2026-07-21 public `thread/fork` current 骨架：v2 `ThreadForkParams/Response`、method/envelope/schema registry、独立 `processor/thread_fork.rs`、`RuntimeCore::fork_thread` 与 typed client/gateway catalog 已接为唯一 public Thread fork 链；AgentControl fork 保持独立的 Multi-Agent topology owner，不复用、不做 compat。fork 从 canonical Thread/Turn/Item 复制 full、`lastTurnId` 或 `beforeTurnId` 的 terminal prefix，重写 target thread/session identity，并只生成内存 baseline event，不把 source raw EventLog 复制为第二套 history。普通 fork 不继承 Goal；persistent `deferGoalContinuation` 在 `goal_fork.rs` 持 source idle permit 并于同一 SQLite `IMMEDIATE` 事务 flush source usage、复制 exact Goal snapshot、写 Goal 外键 durable marker。fork/resume 不自动 continuation，首个真实 `turn.accepted` 幂等消费 marker；无 Goal 不造 marker，paused Goal 原样继承，target delete 通过 Goal FK 级联清 marker。`ephemeral + defer`、`lastTurnId + beforeTurnId`、path 和尚无 persistent owner 的 ephemeral 均 fail closed。验证：公共 JSON-RPC `thread_fork_jsonrpc` 2/2（history / Goal / restart / explicit admission；full/last/before、无 Goal、paused Goal、delete cascade、非法参数），App Server tests check、protocol round-trip、schema fixture、scoped rustfmt、`check:protocol-types`（773 generated types）、package client 64/64、`test:contracts`（299 checks）、`governance:legacy-report`（0 分类漂移 / 0 边界违规）和 `git diff --check` 全通过。全 crate `cargo fmt --check` 仍只被隔壁 `session_archive_jsonrpc.rs` 格式漂移阻塞，本切片的三个 fork 文件已经 scoped rustfmt 通过。分类：public `thread/fork`、canonical fork history、Goal snapshot/marker/admission 为 `current`；AgentControl fork 为独立 `current`；无 compat/deprecated 新增。OPEN_REF：fork 后跨重启 provider message history lowering，尤其多模态/tool/compaction，必须从 canonical Thread/Turn/Item 重建，不得回拷 legacy raw event。

- 2026-07-21 `thread/fork` 跨重启 provider history 收口：target metadata 新增内部 `forkSequence` 固化 copied canonical prefix；`thread_fork` 在相同 sequence slot 生成 typed `thread.fork.canonical_item` seed，其余 slot 保持 baseline，因此首个 target event 仍从 `throughSequence + 1` 连续追加。`provider_history/canonical.rs` 成为 User/Agent/Reasoning/Tool/MCP lowering owner，MCP inner/runtime 名通过 `lime_mcp::naming` 去重规范。`thread/read` 在普通 EventLog hydration 成功后仍合并 seed，直接 `turn/start` 与显式 `thread/resume` 都不会漏前缀；ProjectionRepair 修复 canonical history 前也用同一 seed 补齐 target EventLog tail，避免第二次重启删除 copied Turn/Item。source raw EventLog 从未复制。canonical 不能无损表达的 input/assistant media、compaction replacement history、collab arguments、Extension 和无结果 Tool 全部 fail closed。公共 JSON-RPC capture backend 已证明 source user -> tool call -> tool result -> assistant 精确顺序，连续两次 restart 后 target tail 只追加一次且 prefix 不重复；验证 `thread_fork_jsonrpc` 3/3、ProjectionRepair 3/3、MCP 名 1/1、fail-closed 2/2、AgentControl fork 隔离回归 1/1、`cargo check -p app-server --tests`、治理扫描（0 零引用/分类漂移/边界违规）与 scoped diff check 通过。related 全量首轮为 1439/1444：本刀误把 AgentControl fork 当 public fork 的 1 条失败已收紧到 `forked_from_id + forkSequence` 并定向复跑通过；剩余 4 条来自并行 Goal table inventory、repairable EventLog tail 和 compaction 热区，本刀未夹写。架构确认已写入 `internal/aiprompts/architecture.md`；分类：canonical seed/provider lowering/repair merge 为 `current`，raw EventLog copy/有损兼容为 `dead / forbidden-to-restore`，无 compat/deprecated 新增。该 OPEN_REF 以 text/reasoning/tool/MCP 可恢复、不可表达能力 fail-closed 的骨架标准关闭；完整 canonical multimodal 与 compaction payload 扩展仍是下一阶段能力项。整体 Codex v1 仍约 40%。

- 2026-07-21 host interrupt preflight 收口：`RuntimeCore::ensure_turn_interruptible(session_id, turn_id)` 与 `cancel_turn` 现在共享 active-turn 校验；v2 `turn/interrupt` 在 pending-route abort hook 前执行 preflight，`SessionNotFound` / `TurnNotActive` 映射为 `INVALID_REQUEST`，无效或终态 turn 不再提前 detach reverse route。回归覆盖 terminal turn preflight/cancel reject、waiting-action cancel 顺序与 abort hook 调用次数；`cargo check -p app-server --lib`、`cargo test -p app-server --lib processor::tests::turn_steer::`（4/4）、terminal cancel/preflight（1/1）、waiting-action cancel（1/1）、`server_request::tests`（23/23）及本轮 Rust `rustfmt --check` 通过。全 workspace fmt 仍被并行 `app-server-protocol/schema_export/registry.rs` 差异阻塞，未覆盖该写集。该步属于 `current / alignment-open`，尚未证明真实 Electron host-interrupt Gate B，也未关闭 schema/generated client 落盘缺口。

- 2026-07-21 protocol generated write-set audit：Rust `ServerRequestResolvedNotification` 已是必填 `threadId + requestId`，但落盘 `schema/json/v0/ServerRequestResolvedNotification.json`、aggregate schema 与 `packages/app-server-client/src/generated/protocol-types.ts` 仍是 requestId-only；`schema_fixtures` 因此失败，明确为 bundle 与 source 生成结果不一致。`write_schema_fixtures` 会先删除整个 schema root，当前该目录存在大量并行 M/D 与未跟踪 `v2/**`，生成 TS 脚本和产物也在写入中；未获 owner 交接前禁止执行 schema regeneration 或 protocol type generation（包括会临时覆写的 check）。交接后最小顺序为 fixture regeneration、TS generation、schema fixture、contracts；此项保持 `current / alignment-open`，不以旧 schema/client 假装完成。

- 2026-07-21 host interrupt public JSONL integration：新增独立 `app-server/tests/host_interrupt_transport_jsonrpc.rs`，从真实 `run_json_lines` 进入 initialize、`thread/start`、`turn/start`、typed command approval reverse request 与 `turn/interrupt`。测试证明 `serverRequest/resolved(threadId, outer requestId)` 早于 `item/completed` 和 `turn/completed`，canonical durable event 严格为 `action.canceled -> item.completed(status=cancelled) -> turn.canceled`；迟到 outer JSON-RPC response 经后续 `thread/read` FIFO barrier 后仍不调用 backend `respond_action`，read model Turn 为 interrupted/canceled terminal。Codex v2 `dynamicToolCall` wire enum 不含 cancelled，因此该条 `item/completed` 对外按 `failed + success=false` lowering，canonical owner 仍保留 cancelled 事实。`cargo test -p app-server --test host_interrupt_transport_jsonrpc` 1/1 通过。分类为 `current / App Server integration`；真实 Electron/preload/IPC/GUI Gate B 仍 OPEN。

- 2026-07-21 `serverRequest/resolved` v2 owner 收口：按 Codex `app-server-protocol/src/protocol/v2/notification.rs` 复制 `ServerRequestResolvedNotification { threadId, requestId }`，并将 method 常量、v2 `ServerNotification` union、schema registry、manifest 与 generated TypeScript 全部迁到 v2；删除 v0 DTO、constant、notification variant、catalog owner 与 v0 schema 文件。App Server `server_request`、typed approval、MCP elicitation 三个生产消费者直接导入 v2，未新增兼容层。验证：v2 protocol 21/21、resolved 相关 App Server lib tests 22/22、真实 `host_interrupt_transport_jsonrpc` 1/1、schema fixtures 1/1、`npm run check:protocol-types`（770 types、0 drift）、scoped rustfmt 与 diff check 通过。分类：v2 notification/schema/client 与 producer 为 `current`；v0 resolved surface 为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。下一刀回到 FileChange：修正 Declined/Cancelled terminal lowering、canonical 多文件 Add/Delete/Move payload，并补独立 FileChange Gate B；整体 Codex v1 对齐仍约 38%。

- 2026-07-21 host interrupt Gate B 进入实现窗口：协议 schema/generated client 已由 A 车道闭合，公共 JSONL integration 已证明 `turn/interrupt` 的 reverse-request abort、resolved FIFO barrier、canonical `action.canceled -> item.completed(cancelled) -> turn.canceled` 和 late outer response fail-closed。剩余唯一主链缺口是隔离 Electron fixture `approval-request-host-interrupt`：在 GUI pending approval 后经 preload/IPC 调 `turn/interrupt`，断言同一 thread/turn/request identity、无 renderer response、无 backend `actionRespond`、pending 清零、interrupted GUI/read model、无 tool result及 canonical event 顺序。写集预定为 current fixture 的 scenario、trace、assertion 和 smoke unit 文件；这些脚本当前由并行进程持有且均为 dirty，本车道只读审阅、未夹写。`internal/refactor/v1/**` 继续避让。分类：host interrupt Runtime/App Server/协议为 `current`；v0 resolved 为 `dead / deleted / forbidden-to-restore`；Gate B evidence 为 OPEN_REF，未以 JSONL integration 代替。

- 2026-07-21 FileChange batch current owner 收口：canonical `ThreadItemPayload::File` 已从单 `path/diff` 替换为 Codex 对齐的 `changes[] + status`，Add/Delete/Update/Move 按同一 `patchId` 进入单个 Thread Item，Move 强制 `path=source`、`move_path=destination`。`patch.started`、`patch.applied`、`patch.declined`、`patch.failed` 都携带或保留同一 batch；patch 专属逐文件 `file.changed` 旁路已删除。`tool_approval_declined` 经 tool-runtime normalizer 保留为 `reasonCode`，投影为 terminal `PatchApplyStatus::Declined`；取消仍由 turn cancel 主链终止。v2 `PatchChangeKind` 直接复制 Codex `{ type, move_path? }` wire，schema bundle、v2 schemas 与 generated TS 已同步，未新增 compat。验证：protocol tagged wire 1/1、schema fixtures 1/1、`cargo check -p app-server --tests`、coding events 23/23、canonical declined batch 1/1、v2 projection 1/1、`npm run test:contracts`（770 types、299 client checks）均通过。架构图已更新并由 root 确认。分类：FileChange batch/protocol/tool metadata 为 `current`；patch 的逐文件旁路为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。OPEN_REF：补真实 Electron FileChange Gate B（Add/Delete/Update/Move + decline/cancel），验证 GUI/read model 与当前 v2 Item 同一 identity。

- 2026-07-21 FileChange cold-read 同构收口：canonical ThreadStore read 不再把 File Item 投影为内部 `file_change`；它与 v2 live `fileChange` 一样生成 GUI `patch`，保留 `changes[]/paths/text`，并把 `Rejected`/`Failed` 显式映射为 `status=failed + success=false`。现有 Renderer 的 patch timeline 因此无需新增类型或 fallback；历史恢复与 live notification 不再有两套 FileChange 显示语义。迁掉测试夹具中的旧 `file_change` shape。验证：declined cold-read 1/1、canonical coding overlay 1/1、App Server compile 通过。分类为 `current`；内部 `file_change` display shape 为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。

- 2026-07-21 Renderer FileChange batch 消费收口：`AgentThreadPatchItem` 显式持有 Codex tagged `changes[]`，live raw `patch.*` 先统一 lower 为 `{path, kind:{type,move_path?}, diff}`，cold v2 保持同形；timeline 文件审查卡直接按 batch 聚合 Add/Delete/Update/Move 与 diff，不再从 `paths[]` 猜成全 Update、`+0/-0`。Move 在 GUI 显示 `source -> destination`，底层仍保留官方 `move_path` wire。验证：相关 5 个 Vitest 文件 41/41、Renderer/Node typecheck、scoped diff check 通过。真实 `gui-coding-input` Gate B 首跑命中 Electron/preload/App Server/thread/read，但暴露 fixture 仍发送旧单文件 File payload，read model 无 FileChange，失败证据为 `.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/file-change-renderer-current-v1-summary.json`；并行 owner 随后已把该 fixture 改为 `changes[]`，本进程检测到 mtime 变化后未夹写。分类：structured batch lowering/typed shape/GUI 为 `current`；`paths[] -> update +0/-0` 展示语义为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。OPEN_REF：在共享 fixture 合并窗口后复跑真实 Electron，并在单一 `file-change-batch` scenario 内补 Add/Delete/Update/Move exact read-model/DOM、Decline completed 与 Cancel interrupted 三个 case。

- 2026-07-21 host interrupt Electron Gate B 骨架与真实阻塞：新增 `approval-request-host-interrupt` 场景，真实 Electron 已命中 preload/IPC、`app_server_handle_json_lines`、`turn/interrupt`、backend `turnCancel` 与 durable `turn.canceled`；GUI 显示“已停止”，console/page/invoke error 为 0，但审批卡仍显示“待确认 1”，输入框未恢复，证据为 `.lime/qc/gui-evidence/claw-chat-current-fixture/approval-request-host-interrupt-v2-summary.json`。只读复核确认 `AppServerConnection.waitForResponse()` 已把 inline notification 写入 mirror，普通 `nextServerMessage()` 可继续读取；此前尝试的 Electron Host control queue 会重复投递，已完整撤回。当前最窄 Renderer 缺口在脏热区 `useAgentTools.ts`：controller snapshot 同步只 upsert，remote resolved 触发 AbortSignal 并移除 controller pending 后，没有按本 controller 曾投影的 request ids 从 `pendingActions` 与 message `actionRequests/contentParts` 删除旧卡片。下一刀由该文件当前 owner 持有 `Set<requestId>` 做 snapshot reconcile，只删除 server-request controller 自己投影且已消失的 actions，并补 remote resolved component/hook 回归；不得恢复 Host 第二队列或清空其它 runtime action。dispatcher 已增加独立 resolved tombstone，使重复 `(threadId, requestId)` resolved 不重复 trace/abort，定向 Host/dispatcher/approval 测试通过。分类：Electron/AppServerConnection mirror、EventBus、typed controller 为 `current`；Host control queue 为 `dead / reverted / forbidden-to-restore`；真实 Gate B 仍 `OPEN_REF`。

- 2026-07-21 host interrupt Electron Gate B 闭环：Renderer typed controller snapshot 现在只跟踪自身已投影的 request ids，并在 remote resolved/AbortSignal 使 snapshot 缩减时按差集清理 `pendingActions` 与 message `actionRequests/contentParts`；不清空其它 runtime action。dispatcher 以独立 resolved tombstone 保证重复 `(threadId, requestId)` resolved 不重复 trace/abort。fixture 同步使用 E2E Desktop Host 的真实 `<electronUserDataDir>/app-server` root，读取 canonical EventLog，并把 host-interrupt 排除在默认新闻 completion wait 之外。真实 source-built Electron 证据 `.lime/qc/gui-evidence/claw-chat-current-fixture/approval-request-host-interrupt-v5-summary.json` 为 `ok=true`、60/60 assertions：Electron/preload/IPC/App Server/runtime/read model/GUI identity 一致，resolved 命中同 outer request 且早于 terminal，renderer response 为 0、backend `actionRespond` 为 0，pending 清零、输入框恢复、Turn 为 interrupted、无 tool result；canonical EventLog 顺序索引为 `action.canceled=8 < item.completed(cancelled)=9 < turn.canceled=10`，legacy/mock/console/page/invoke error 均为 0。最终验证：相关 Host/EventBus/dispatcher/controllers/action-state/fixture Vitest 138/138、`npm run typecheck`、`npm run test:contracts`（770 generated types、299 client checks）与全树 `git diff --check` 通过。Host control queue 已证伪并撤回，底层 `AppServerConnection` mirror 继续作为唯一 current 上行机制。分类：dispatcher/controller/fixture evidence 为 `current`；Host control queue 为 `dead / reverted / forbidden-to-restore`；host-interrupt Gate B OPEN_REF 已关闭，整体 Codex v1 仍约 38%。

- 2026-07-21 FileChange Cancel/Decline runtime 语义收口：对照 Codex `ReviewDecision::{Denied,Abort}` 后，`RuntimeBackend::respond_action` 不再把 ToolConfirmation `Cancel` 压成 `confirmed=false` 的普通 resolve；Cancel 现在 terminalize pending action 为 `Canceled`，发布 `action.canceled`，RuntimeCore 不向 approval waiter 投递 response，而是关闭 active generic Tool 为 `cancelled` 并经 session loop 中断 Turn。`Decline` 仍投递 false response，由工具层生成 `tool_approval_declined`，允许 Turn 继续并投影 FileChange `Declined`。App Server-owned preflight Cancel 也从 `action.resolved` 收敛为 `action.canceled`。验证：action owner 1/1、live session 3/3、preflight Decline/Cancel 2/2、rustfmt、scoped diff check、`npm run test:contracts`（770 generated types、299 client checks）、`governance:legacy-report`（0 violation）和 `governance:scripts` 全部通过。分类：Cancel/Decline 分支与 turn interrupt 为 `current`；`Cancel -> confirmed=false -> tool_approval_declined/patch.declined` 为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。

- 2026-07-21 FileChange approval Gate B 首跑阻塞：隔壁进程运行 `npm run smoke:code-artifact-workbench-electron-fixture -- --scenario file-change-batch --prefix file-change-batch-approval-gate-b-v1 --timeout-ms 240000`；backend ledger 已证明同一 Turn 发出 `patch.started + action.required(toolName=apply_patch)`，但 Renderer 未显示 `item/fileChange/requestApproval` 卡，失败证据为 `.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/file-change-batch-approval-gate-b-v1-summary.json` 与同前缀截图。当前最窄 OPEN_REF 有两项：其一，持有 `src/lib/api/appServerServerRequest.ts` / `useAgentTools.ts` / approval card 热区的 owner 修复 FileChange reverse request 投影并补 pending/Decline/Cancel DOM 回归；其二，持有 `lime-rs/crates/agent/src/current_provider_turn/tool_executor.rs` 的 owner 将生产 `availableDecisions` 从 `allow_once/decline` 对齐为 FileChange/Command 所需的 `allow_once/allow_for_session/decline/cancel`，否则 RuntimeCore 会按声明正确拒绝 Cancel。当前不得以 backend ledger 或 Rust 测试替代 Gate B，修复后复用同一场景重跑；整体 Codex v1 完成度仍约 40%。

- 2026-07-21 FileChange Decline read-model 根因收口：v3 真实 Electron 已能显示 FileChange 审批卡并点击 Decline，external backend ledger 发出 `action.resolved -> patch.declined -> item.completed -> turn.completed`，但 App Server read model 仍停在 Turn/FileChange `inProgress`；失败证据为 `.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/file-change-batch-approval-gate-b-v3-summary.json`。只读审计确认整批事件在 append 前被 `agent_ui_sequence_verifier` 原子拒绝：validator 只把 `patch.applied/patch.failed` 当 terminal，遗漏 `patch.declined`，最终触发 `patch_unclosed_at_turn_end`。隔壁 owner 已在 `agent_ui_sequence_verifier.rs` 补 `patch.declined` 闭合和 without-start 负向回归；本车道新增独立 public RuntimeCore integration `app-server/tests/file_change_approval_read_model.rs`，证明 Decline 后 Turn `completed`、patch `declined`、Add/Delete/Update/Move batch 保留，并修正 `coding_activity_projection` 把 declined patch 从 running 集合闭合。验证：validator owner 1/1、public integration 1/1、fixture helper 16/16、scoped rustfmt/diff check 通过；`test:rust:related` 跑满 App Server lib 1416 条，本轮相关测试通过，整体仅剩 2 条并行 conversation-import 旧 `file_artifact` display-shape 断言失败。分类为 `current`；把 `patch.declined` 当非终态的 validator/projection 语义为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。OPEN_REF：等当前 approval-cancel Electron 进程退出后复跑 FileChange Gate B；fixture Cancel 仍需精确验证 `action.canceled -> item.completed(cancelled) -> turn.canceled`，不得发 `action.resolved/patch.failed/patch.declined`；生产 `tool_executor` 仍只声明 `allow_once/decline`，真实 RuntimeCore Cancel 尚受声明契约阻断。

- 2026-07-21 FileChange Decline canonical read 与 v6 Gate：public integration 已升级为临时 `ProjectionStore + read_session_current`，直接覆盖 production canonical ThreadStore read，断言 Turn `completed`、File Item 对外 `patch(status=failed,file_status=rejected)`、四种 tagged changes 与 Move `move_path` 保留、`running_patch_count=0`，1/1 通过。`tool_lifecycle` denied parser 同步加入实际协议词值 `decline/declined`，新增回归证明 Decline 后错误 `item.completed(completed)` 被 `tool_result_after_action_denied` 拒绝，而 `item.completed(failed)` 可闭合，1/1 通过。隔壁随后运行 v6 Electron：`.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/file-change-batch-approval-gate-b-v6-summary.json`；backend ledger 已发 `action.resolved -> patch.declined -> item.completed -> message.delta -> turn.completed`，read-model wait 通过，截图可见精确 4 文件卡、拒绝 continuation 与 Turn 已完成，console/page/invoke error 为 0；但唯一 GUI 卡未提供 fixture 期待的 `data-file-status=declined`，最终 `FileChange terminal GUI not reached`，Cancel case 因此未执行。该 Gate 仍为 `OPEN_REF`，不得宣称通过。下一刀归 Renderer/fixture 热区 owner：保留 canonical `file_status` 到 live/terminal card（或按产品 DOM current contract修正 Gate），补同 item identity 的 terminal DOM 断言后再运行 Cancel；Cancel fixture 必须改为 `action.canceled -> item.completed(cancelled) -> turn.canceled`，不能继续发 `action.resolved`。生产 `tool_executor` 仍需先发布 `cancel`；`allow_for_session` 要等 FileChange session-cache owner后才能暴露。普通 reverse-request 失败 fallback 也仍需由 `approval_server_request.rs` owner从 `Cancel` 对齐为 Codex `Decline`，turn transition 分支继续直接 return。

- 2026-07-21 FileChange Cancel validator 与 v9 stale-card：Codex Abort 不产生 patch applied/failed/declined terminal，因此 sequence validator 现只对 `turn.canceled` 允许清理 active patch；`turn.completed/turn.failed` 仍严格报 `patch_unclosed_at_turn_end`。新增 canonical integration 证明 Cancel 事件严格为 `action.canceled -> apply_patch item.completed(cancelled) -> turn.canceled`，Turn 为 Canceled，且不存在 `patch.applied/patch.failed/patch.declined`；Decline/Cancel integration 2/2、cancel validator 1/1、原 strict unclosed guard 1/1、rustfmt/diff check 与治理扫描通过。隔壁 v9 Electron 在 updated fixture 中为 Decline 额外发 canonical File `item.completed(Rejected)`，backend/read model 已终态，但 `.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/file-change-batch-approval-gate-b-v9-summary.json` 仍失败：唯一 `file-changes-summary-card` 与三条已显示 row 的 `data-file-status` 全是 `inProgress`，没有用同 item identity 的 terminal File item刷新；console/page/invoke error 均为 0，Cancel case仍未执行。下一刀明确归 Renderer live item/content-part merge owner：同一 File item terminal 到达时替换/更新既有 inProgress batch card，并稳定展示四条 row；不得在 fixture 添加第二套展示旁路。fixture Cancel 自身仍错误发 `action.resolved`，owner 需同步改为 `action.canceled` 后再跑 Gate。分类：Cancel abandon、Decline denied parser与 canonical integration 为 `current`；Cancel patch terminal及 completed/failed turn放弃 active patch 均为 `dead / forbidden-to-restore`；无 compat/deprecated。整体 Codex v1 对齐仍约 40%。

- 2026-07-21 Renderer FileChange terminal merge 收口：`buildTimelineInlineContentParts` 不再无条件把历史 `file_changes_batch` 追加到 canonical timeline；它从 `metadata.threadItemId/threadItemIds` 读取 identity，同一 File Item 的最新 terminal part 直接替换旧 `inProgress` 卡，无 identity 或不同 identity 的历史卡继续保留。独立回归覆盖 batch-id -> single-id 的 declined 替换与无 identity 保留，timeline 明确测试清单 28/28、Renderer typecheck、scoped ESLint/Prettier/diff check 通过；`test:related` 因共享 Vitest 收集器 `EISDIR .../electron` 未进入断言，已由明确测试清单替代。分类：identity-aware terminal merge 为 `current`；旧快照无条件覆盖 canonical terminal 卡的行为为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。真实 v10 Gate B 暂未启动：fixture helper 在 13:21 被并行 owner继续修改，当前中间态虽已把 Cancel 首事件改成 `action.canceled`，但仍缺 canonical `item.completed(cancelled)` 且相邻 helper 测试未同步；本车道继续避让该热区，等待 owner 交接后再验证 Decline DOM terminal 和完整 Cancel 顺序。整体 Codex v1 对齐仍约 40%。

- 2026-07-21 FileChange approval Electron Gate B 闭环：v11/v12 的唯一失败 `typedFileChangeServerRequests=false` 已确认为 evidence 假阴性，不是 App Server reverse-request 分发缺口；同期 Electron IPC trace 已存在 `app-server-request:*` outer response，但 fixture 未开启 `lime:debug:claw-trace-enabled:v1`，导致脱敏 lifecycle localStorage 为空。`file-change-batch` 场景现于首轮发送前显式开启 trace 并清空旧 lifecycle，不改变生产审批路径。真实 source-built Electron v13 证据 `.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/file-change-batch-approval-gate-b-v13-summary.json` 为 `ok=true`，全部 assertions 为 true：两轮均命中 `item/fileChange/requestApproval`，且相同 outer id 严格满足 `request < response < serverRequest/resolved`；Decline 保留精确 Add/Delete/Update/Move 四项、FileChange `declined`、Turn `completed`；Cancel 保留同批四项、FileChange `inProgress`、Turn `interrupted`，且不合成 patch/FileChange terminal；GUI 四行、pending 清零、Electron preload/IPC/App Server identity 一致，console/page/invoke error 均为 0。脚本定向回归 17/17、Node syntax、Prettier 与 scoped diff check 通过。分类：typed FileChange reverse request、Decline/Cancel read model/GUI 与 Gate B evidence 为 `current`；raw approval UI 仅为运行时 projection，不是第二 owner；无 compat/deprecated，FileChange Gate B `OPEN_REF` 已关闭。整体 Codex v1 对齐仍约 40%。

- 2026-07-21 v1.108.0 Gate A/B 发布候选收口：严格按 `internal/refactor/v1/05-verification-and-guardrails.md` 执行 Gate A，`test:contracts`、`test:rust:related -- app-server agent-protocol thread-store` 与 `governance:legacy-report` 全绿；App Server 1419/1419、agent-protocol 34/34、thread-store 28/28，治理扫描零分类漂移/零边界违规。Rust related 首轮仅有两条 Codex import 测试仍断言已删除的 `file_artifact` display shape，确认 production 已正确投影 canonical File Item 为 current `patch` 后，只迁移测试断言，原失败 2/2 与完整 related 均通过，未恢复兼容 shape。Gate B 的 `verify:gui-smoke` 生成 `standalone-shell-01-20260721063158-62100`，真实 Electron/preload/IPC 命中 `app_server_handle_json_lines` 33 次、21/21 断言、mock/legacy/console/page error 为 0；`smoke:agent-runtime-current-fixture` 全绿，Content Factory v2 Article Editor 70/70，动态 session/thread identity、编辑、reload、artifact/read model 和 workflow 控制闭环。`verify:app-version`、完整 `typecheck`、`cargo check -p app-server --lib` 与 `git diff --check` 通过；此前 3 条 dead-code warning 在 current owner 汇合后已消失。Codex import CCI-001..011 已全部 completed，架构确认写入第 7 节对应执行计划，tracker 进入 `ready-for-gate`；Windows 真实 Electron 仍为平台 follow-up。v1.108.0 release candidate 门禁已通过，但整个 Codex v1 对齐仍约 40%，不得据此宣称 `internal/refactor/v1` 全部完成。

- 2026-07-21 FileChange public JSONL integration 收口：新增独立 `app-server/tests/file_change_transport_jsonrpc.rs`，从真实 `run_json_lines` 进入 initialize、`thread/start`、`turn/start`，由 owned `RuntimeEventHub` 后台发出 `action.required(toolName=apply_patch)`。测试断言只产生一个 `item/fileChange/requestApproval`，Renderer 形态 outer response `{decision:"decline"}` 被 runtime action contract 规范化为 `{confirmed:false}`，且同 outer id 的 `serverRequest/resolved` 严格早于 `turn/completed`；最终 `thread/read` 返回 canonical `item_<patchId>` FileChange、`status=declined`、Turn `completed`，并精确保留 Add/Delete/Update/Move 四项 tagged changes。验证：`cargo test -p app-server --test file_change_transport_jsonrpc` 1/1、`npm run test:rust:related -- lime-rs/crates/app-server/tests/file_change_transport_jsonrpc.rs` 1420/1420、rustfmt check 与未跟踪文件 no-index whitespace check 均通过。分类：public JSON-RPC transport integration 为 `current / App Server integration`，测试 backend 为 `test-only`；无 compat/deprecated/dead 新 surface。该证据补齐 App Server 层，不替代上一条已通过的 Electron Gate B；FileChange transport integration 完成度 100%，整体 Codex v1 对齐仍约 40%。

- 2026-07-21 共享回归与并行写集核对：当前工作树无 Rust 编译进程；本进程未修改隔壁持有的 `tool_executor.rs`、`approval_server_request.rs`、FileChange fixture、ThreadGoal/idle continuation owner 或 ThreadStore raw-rollout owner。验证 `cargo test -p lime-agent tool_approval_exposes_cancel_without_session_grant` 1/1、`cargo test -p app-server approval_server_request --lib` 7/7、`cargo test -p app-server --test file_change_transport_jsonrpc` 1/1；随后执行 `npm run test:rust:related -- lime-rs/crates/app-server-protocol lime-rs/crates/app-server lime-rs/crates/agent-protocol lime-rs/crates/thread-store lime-rs/crates/model-provider`，相关 workspace crate 单测全绿（agent-protocol 34、agent-runtime 159、app-server 1420、thread-store 28、model-provider 156、tool-runtime 249 及其余反向依赖）。只读 Codex 对照确认：`runtime/objectives.rs` 的 ManagedObjective 自动循环不是 ThreadGoal continuation owner；真正下一刀必须由同一 owner 一次性接管 `thread_goal.rs`、`goal_accounting.rs`、`turn_execution.rs`、旧 objective 触发点与独立 JSONL/Gate B 测试，当前这些文件均为并行热区，禁止另起第二 owner。分类：现有审批/FileChange/ThreadStore projection 为 `current`；旧 ManagedObjective continuation 与 snapshot/raw rollout 双轨为 `deprecated/dead` 候选，暂不删除，等待 owner 交接和引用扫描；本轮无新增 compat。`

- 2026-07-21 `thread/delete` current 垂直骨架：v2 protocol/ingress 已接到 App Server dispatch、`RuntimeCore::delete_thread`、RolloutStore 与三库 ProjectionStore；已持久化 spawn descendants 按 deepest-first 停止 session loop/backend，并清 rollout、event log、sidecar、goal/accounting/outbox、Agent identity/mailbox 与 canonical/history/projection rows，响应 `{}` 后按 child-to-root 顺序返回 `thread/deleted`。Renderer/package typed client 已改用 `deleteThread({threadId})`，GUI session facade 先解析 canonical threadId；旧 `agentSession/delete` production catalog/DTO/dispatch/helper、v0 schema 与 generated TS surface 已删除并有负向守卫。公共 `run_json_lines` 集成 1/1 通过，覆盖 parent+child response/notification 顺序、磁盘/三库/identity/mailbox 清理、删除后 read 与冷启动不复活；protocol schema fixture 1/1、package 71/71、Renderer 本刀定向 35/35、完整 typecheck、`test:contracts`（779 schema definitions / 771 TS types / 299 client checks）、`governance:legacy-report`（0 分类漂移 / 0 边界违规）、真实 Electron `verify:gui-smoke` 与非 live Provider `smoke:agent-runtime-current-fixture` 全绿。Rust related 的 App Server 1431 项为 1429 通过、2 项失败，均在并行 ThreadGoal baseline / event-log tail repair 热区；本刀独立 JSONL 与删除 unit 3/3 均绿。分类：`thread/delete` 为 `current`；`agentSession/delete` 为 `dead / deleted / forbidden-to-restore`。尚未闭合：pending-only child、整棵子树单事务、listener/跨连接广播、trace/telemetry 清理；当前仍是骨架阶段，整体 Codex v1 约 40%。

- 2026-07-21 `thread/delete` 完整性收口：`ProjectionStore` 新增 immutable subtree snapshot 与 strict delete，两阶段边界覆盖 persisted、pending-only child 及 `pending_session_id`；`BEGIN IMMEDIATE` 后在同一连接重读 recursive CTE，快照漂移零写入失败，一次 ATTACH transaction 清 state/history/projection、goal/accounting/outbox、spawn graph、Agent identity/mailbox，canonical root 最后删除。RuntimeCore 先停全部 live owner，幂等清 rollout/EventLog/Sidecar/Trace/Telemetry，再单次提交并清内存 session/cache/goal continuation；外部文件已清而最终 SQL 失败时，三库与内存 session 保留，删除可稳定重试，第二次成功删除返回 not found。transport 在 `{}` 响应后经 per-thread listener 向 origin 与订阅连接广播 pending/child/root 通知，随后 `ThreadStateManager::remove_thread` 取消 listener、resume barrier 和双向 connection index；子树 pending server requests 按注册顺序先发 `serverRequest/resolved` 并以 `REQUEST_CANCELLED` 终止。验证：atomic store 2/2、ThreadState 1/1、server-request 1/1、双连接广播 1/1、失败重试/二次删除 1/1、trace 1/1、telemetry 1/1、带 pending-only child 的 public `run_json_lines` integration 1/1、原 RuntimeCore 删除 2/2 均通过；最终复跑 public `thread_delete_jsonrpc` 1/1、ThreadState 双向 connection index 清理 1/1、`npm run test:contracts`（781 schema definitions / 773 generated types / 299 client checks）、`npm run governance:legacy-report`（0 分类漂移 / 0 边界违规）、`npm run smoke:agent-runtime-current-fixture` 与全树 `git diff --check` 均通过，fixture 使用真实 Electron/preload/IPC/App Server/read model 且 `liveProviderUsed=false`。App Server related 1439 项仅有 2 条并行 ThreadGoal/EventLog 热区既有失败；workspace fmt 只被并行 `agent-protocol/src/lib.rs` 与 `app-server/tests/session_archive_jsonrpc.rs` 格式漂移阻塞，本切片 scoped rustfmt 已通过。架构图确认：本轮没有改变 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI` 依赖方向，删除的唯一 storage commit owner 仍是 App Server `ProjectionStore`；确认状态 `confirmed`。分类：`thread/delete` 为 `current`；`agentSession/delete` 继续为 `dead / deleted / forbidden-to-restore`，无 compat/deprecated 新增；本切片实现完成，整体 Codex v1 仍约 40%。

- 2026-07-22 AgentControl fork Codex sanitize 收口：按 Codex `core/src/agent/control/spawn.rs::keep_forked_rollout_item` 对齐 Multi-Agent fork 语义，`fork_turns=none` 在完整 canonical history 读取前短路，只创建 fresh child；`LastNTurns` 先裁剪 Turn 窗口，再读取/校验 durable input。Reasoning、Tool、MCP、Plan、Approval、Command、File、Media、SubAgent、Extension 与 assistant 非 final item 作为内部 rollout 项过滤，不伪造第二套 provider lowering；FullHistory 遇无 replacement history 的 ContextCompaction、final-answer media、非 text user input、重复/未完成 user item 或缺 durable input 时，在 pending edge、child session、mailbox、EventLog 前 fail closed。新增独立 `runtime/tests/agent_control/fork.rs`，覆盖 sanitize 过滤、compaction 零残留、final media 零残留、LastN 选窗与 None fresh child。验证：AgentControl 全族 35/35、fork 专项 4/4、`cargo check -p app-server --lib`、scoped rustfmt/diff check、`npm run governance:legacy-report`（0 零引用候选、0 分类漂移、0 边界违规）通过；`npm run test:rust:related` 为 1445/1449，剩余 4 条来自并行 Goal table inventory、repairable EventLog tail、compaction/session context 热区，本轮未夹写。分类：AgentControl sanitize/fork boundary 为 `current`；未新增 compat/deprecated/dead；public `thread/fork` 的无损 canonical lowering 继续与 AgentControl 拓扑 fork 隔离。下一刀回到未闭合的 Codex Multi-Agent compacted replacement history 或跨重启 Gate B，不在本车道恢复 raw EventLog copy。

- 2026-07-22 ContextCompaction lineage fail-closed：`runtime/context_compaction.rs` 不再把最新但部分缺失的 `replacementHistory/windowNumber/firstWindowId/previousWindowId/windowId` 静默当作新链；完全没有 lineage 字段的 legacy event 才按计数兜底，部分字段必须完整解析，window number 为正数且所有恢复 ID 为 UUIDv7，否则返回带 event identity 的 `RuntimeCoreError::Backend`，阻止 resume/next compaction 生成断裂 lineage。新增 malformed marker 回归并将 imported lineage fixture 改为真实 UUIDv7；验证 `runtime::context_compaction::tests` 7/7、scoped rustfmt 与 `git diff --check` 通过。分类：lineage validation/fail-closed 为 `current`；部分 marker 静默重置 window chain 为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。完整 canonical `ContextCompaction` replacement payload、rollback/fork/window-chain provider history 仍是 P0 OPEN_REF，未宣称 compaction parity。

- 2026-07-22 provider route identity/cache scoped 骨架：model registry 的 API fetch 现在用 `SHA-256(api_key)` 作为不落盘的 credential fingerprint，将带凭证 fetch 的 cache 读写隔离到 scoped key；无凭证读取继续只访问 unscoped namespace，不跨 credential fallback。credential-scoped cache 与 tenant namespace、taxonomy/expiry/clear generation 共享既有 owner，新增隔离回归；验证 `lime-services model_registry_service` 56/56、无新增 production dead-code warning。非主 Turn 的 host-managed/image direct text 路径补齐 `ResolvedModelRoute.auth.credential_ref -> ModelRouteProviderConfiguration -> configure_model_route_provider_for_session_with_provider_and_credential_ref`，不再重新 round-robin；验证 lime-agent provider configuration 8/8、App Server model route contract 6/6、model route resolver 9/9、scoped rustfmt 与 `git diff --check` 通过。分类：credential-scoped cache 与 route identity 传播为 `current`；非主路径丢失 resolved credential 后重新 round-robin 为 `dead / forbidden-to-restore`；无 compat/deprecated。旧 `agentSession/action/{replay,respond}`、`agentSession/runtimeEvents/append` 仍为共享 protocol/dispatch/frontend 热区 OPEN_REF，本车道未夹写。

- 2026-07-22 compaction prompt boundary：`runtime/memory_prompt.rs` 识别完整 `replacementHistory + window lineage` 后不再把同一摘要重复注入 system prompt；无 lineage 的 legacy compaction 仍保留明确 fail-safe。并行 owner 新增两项单测；复核 `memory_prompt` 16/16、`provider_history` 16/16、scoped rustfmt/diff check 通过。related 全量曾为 1438/1442，剩余 2 条旧 `sessions.rs` compaction 断言与 2 条并行 Goal/storage/EventLog 热区失败，未在本车道夹写。分类：replacement-history prompt boundary 为 `current`；有效 replacement history 的重复 system-prompt 注入为 `dead / deleted / forbidden-to-restore`；legacy summary fallback 为显式 `compat`/fail-safe，不承接新业务逻辑。

- 2026-07-22 typed server-request response ingress 骨架：`packages/app-server-client/src/connection.ts` 新增 `respondServerRequest` 与 `rejectServerRequest`，并由标准 `AgentRuntimeClient` 与 browser-safe session gateway 以可选能力透出；只按 reverse JSON-RPC outer `requestId` 回写 result/error，不从 thread/turn/action 元数据推断路由。连接现在跟踪实际收到的 reverse request id，未知或重复 id 回包直接 fail-closed。新增隔离 `server-request-response.test.mjs` 覆盖成功、取消、重复 id 和缺 responder；README 同步记录 current boundary。验证 package client 全量测试（73/73）、build/typecheck、agent-runtime-client typecheck/test（23/23）、`npm run test:contracts`（299 client checks，773 generated types）、Prettier 与 scoped `git diff --check` 通过。该 helper 仅提供 v2 server-request 回包边界，尚未迁移旧 `agentSession/action/{replay,respond}` 与 `agentSession/runtimeEvents/append` 生产消费者，不能关闭 P0 OPEN_REF；分类：typed response ingress 为 `current`，旧 action/runtime-events 仍 `deprecated -> deleted` 候选。

- 2026-07-22 AgentControl compacted replacement fork 收口：FullHistory 从已验证的 compaction marker 提取 `ForkCompactionSeed`，保留 replacement/window lineage，将 session-level `context.compaction.completed` 重写为 child session/thread，并把 source `tailStartTurnId` 映射到稳定的 child fork Turn；marker、replacement payload、UUIDv7 window lineage 或 tail 映射异常均在 pending edge/session/mailbox/EventLog 前 fail closed。LastN 只重建选中 Turn 并丢弃 compacted prefix，fork `none` 仍保持 fresh child；不复制 raw EventLog，不复用 public `thread/fork` strict validator。验证：AgentControl fork 5/5、provider history 14/14、context compaction 7/7、`cargo check -p app-server --lib`、scoped rustfmt、全树 `git diff --check` 与 `npm run governance:legacy-report`（0 零引用候选、0 分类漂移、0 边界违规）通过。分类：compaction seed/child lineage/provider boundary 为 `current`；无损表达不了的 marker/input/media 及 raw history copy 为 `dead / forbidden-to-restore`；无 compat/deprecated 新增。下一项仍是完整 canonical multimodal replacement payload 与跨重启 Gate B，不得用 raw event 回拷补洞。

- 2026-07-22 typed reverse server-request 回包验证补录：在上一条 ingress 骨架基础上，`AppServerConnection.nextMessage` 与 `nextServerMessage` 都登记实际收到的 reverse JSON-RPC outer id；`respondServerRequest`/`rejectServerRequest` 对未知、重复或已取消 id fail closed，`serverRequest/resolved` 终态通知会撤销 pending outer id，标准 `AgentRuntimeClient` 与 browser-safe `sessionGateway` 只按 outer id 透传，不从 action/thread 元数据推断路由。新增回包测试覆盖成功、取消、重复 id、generic read path 与 gateway 缺 responder；app-server-client 定向 3/3、全量 75/75，agent-runtime-client 23/23，`test:contracts` 773 generated types/299 client checks，治理扫描与窄写集 diff check 均通过。分类：typed reverse response ingress 为 `current`；`agentSession/action/{replay,respond}` 与 `agentSession/runtimeEvents/append` 仍为共享协议/dispatch/frontend 热区的 `deprecated -> deleted` 候选，尚未迁移生产消费者，P0 OPEN_REF 保持开启；本轮不触碰隔壁 `internal/refactor/v1/**`、Rust protocol/runtime 热区及未跟踪 `interrupted`。

- 2026-07-22 AgentControl compaction cold-restart evidence：fork 测试关闭 parent RuntimeCore 后重新 hydrate child，并从同一 EventLog/ProjectionStore 重建 provider history，确认 replacement prefix 与 child tail 在重启后各出现一次；新增 nested/top-level marker 字段一致性 fail-closed 回归。验证：fork 5/5、context compaction 8/8、provider history 14/14、scoped rustfmt、`cargo check -p app-server --lib` 与 `git diff --check` 通过。该证据关闭 AgentControl compaction 的跨重启骨架缺口；完整 public `thread/fork` 多模态/compaction payload 仍由其 current owner 继续推进。

- 2026-07-22 typed reverse response checkpoint：重建 `app-server-client` dist 后，server-request 回包定向 3/3、package 全量 75/75、`agent-runtime-client` 23/23；`npm run test:contracts` 通过（773 generated types、299 client checks），`npm run governance:legacy-report` 通过（0 零引用候选、0 分类漂移、0 边界违规），`git diff --check` 通过。`nextMessage`/`nextServerMessage` 的 reverse outer id 登记与 `serverRequest/resolved` tombstone 已有 current 证据；旧 `agentSession/action/{replay,respond}`、`agentSession/runtimeEvents/append` 生产引用仍未迁移，继续标记 `deprecated -> deleted` / P0 OPEN_REF，不以 domain request id 冒充 outer id。

- 2026-07-22 current fixture recheck：历史/缓存 31/31、流式收尾 32/32、Electron fixture guard 84/84、首页热路径、短问候热路径与 Coding Workbench 均通过；完整 smoke 在 `image-command` 停止于 `image_generate` task artifact 数量为 0。失败 summary 显示 GUI session 仍绑定通用 `fixture-model`，未绑定本场景创建的 `Fixture Image Provider`，且 console/page/invoke errors 均为 0；归属 provider/image route 配置并行写集，不能作为 typed server-request 回包回归。P0-05 的图片 Gate B 保持未验证，待 provider owner 交接后复跑。Renderer/Node `npm run typecheck` 随后通过。

- 2026-07-22 旧 action 迁移映射契约（交接前只读约束）：`ServerRequest.id` 是唯一 JSON-RPC outer id，只能用于 `respond/error`；`itemId`、`approvalId`、`requestId` 等 domain 字段只作为 UI pending/settled 语义键，禁止互相替代。`item/commandExecution/requestApproval` 与 `item/fileChange/requestApproval` 只回 `{decision}`，`item/tool/requestUserInput` 只回 `{answers}`，MCP elicitation 只回其 typed response；`thread/resume` 的 pending reverse request 由 listener/event bus replay，删除 `agentSession/action/replay`，不得再扫描 session/turn/action 猜 waiter。`agentSession/action/respond` 只有在所有旧 UI/Plugin 入口改用 typed responder 后才能删除；`agentSession/runtimeEvents/append` 必须等 current typed Item/artifact writer，禁止把旧 runtime-event payload 猜测转换为 Item。

- 2026-07-22 AgentControl compaction marker 双层完整性守卫：`latest_fork_compaction_seed` 现在要求顶层与 `artifact` 同时提供并逐字段一致的 `tailStartTurnId`、replacement history 与 window lineage；任一侧缺字段或值漂移均在 child side effect 前 fail closed，避免半旧半新的 marker 被静默拼成新链。新增 partial nested marker 回归；`context_compaction` 9/9、AgentControl fork 5/5、scoped rustfmt 与 `git diff --check` 通过。分类：双层 marker 验证为 `current`；部分 lineage marker 的静默合并为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。隔壁 public `thread/fork`、协议、ThreadGoal、EventLog 写集未触碰，下一刀仍回到其 owner 的 canonical multimodal/compaction Gate B。

- 2026-07-22 MCP step snapshot generation 骨架：`tool-runtime::McpConnectionRegistry` 为 register/remove/inherit mutation 分配单调 generation，并在持 registry lock 的同一视图中捕获 `McpStepSnapshot.generation`；snapshot 保持 immutable，旧 sampling step 不因 registry 替换而改路由，generation 可用于 tool exposure/execution evidence 关联。新增旧快照隔离与 generation 单调性回归。验证：`tool-runtime` MCP 定向 9/9、全量 250/250，`cargo check -p lime-agent --lib`、`cargo check -p agent-runtime --lib`、scoped rustfmt、`git diff --check` 与 `npm run governance:legacy-report`（0 零引用候选、0 分类漂移、0 边界违规）通过。分类：snapshot generation/immutability 为 `current`；无新增 compat/deprecated/dead。下一刀仍是 MCP auth/scopes/environment provenance 或 public `thread/fork` canonical multimodal Gate B，继续避让隔壁协议/ThreadGoal/EventLog 写集。
- 2026-07-22 MCP generation 调用/elicitation provenance：`McpStepSnapshot::dispatch` 将捕获的 generation 写入 `McpCallScope`，`McpBridgeClient` 保持同一 scope 传递，`ElicitationOwnerGate` 到 `ElicitationRequestRouter` 再保留内部 `snapshot_generation`；该字段不进入 App Server reverse-request wire。新增 dispatch scope 观察回归与真实 scoped elicitation 回归。验证：`tool-runtime` MCP 10/10、`lime-mcp` elicitation 23/23、scoped rustfmt、`git diff --check` 通过。分类：调用/elicitation provenance 为 `current`；无新增 compat/deprecated/dead。MCP auth/scopes/environment provenance 仍为下一刀，继续避让隔壁协议/ThreadGoal/EventLog 写集。
- 2026-07-22 MCP auth-scope provenance 骨架：`McpServerConfig.scopes` 经过 `McpBridgeSnapshot.auth_scopes`、`McpConnectionProvenance` 和 `McpStepSnapshot` 写入 `McpCallScope`，继续贯通 bridge 与内部 `ElicitationRequest`，不进入 App Server reverse-request wire；未从 stdio `cwd` 臆造 environment identity，environment owner 继续保持 OPEN_REF。验证：`tool-runtime` MCP 10/10、`lime-mcp` elicitation 23/23、auth status 6/6、`cargo check -p tool-runtime --lib`、`cargo check -p lime-mcp --lib`、`cargo check -p lime-agent --lib`、scoped rustfmt 通过。分类：auth-scope provenance 为 `current`；无新增 compat/deprecated/dead；environment provenance 仍待显式配置 owner。
- 2026-07-22 MCP environment identity provenance 骨架：按 Codex `DEFAULT_MCP_SERVER_ENVIRONMENT_ID = "local"` 语义，`McpServerConfig` 增加显式 `environment_id`（支持 snake/camel 输入，缺省 local，空值 fail closed），不再从 stdio `cwd` 推导环境。配置身份经 `McpBridgeSnapshot -> McpConnectionProvenance -> McpStepSnapshot -> McpCallScope -> ElicitationRequest` 传播；仍不进入 App Server reverse-request wire。新增默认/显式/空值配置回归、dispatch scope 与 nested elicitation provenance 断言。验证：`cargo test -p tool-runtime --lib` 251/251、`cargo test -p lime-mcp --lib` 142/142、`cargo test -p lime-agent --lib mcp` 12/12、三个 crate 测试目标编译、`git diff --check` 均通过。分类：environment identity/provenance 为 `current`；从 cwd 猜测环境身份为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。尚未实现 Codex `McpRuntimeContext` 的命名 environment registry/remote executor 解析，继续保持 OPEN_REF，不以 cwd 或默认值伪造 unknown environment 的成功解析；下一刀应由 MCP runtime owner 接管解析边界，继续避开隔壁 App Server/protocol/runtime 热区。

| 2026-07-22 | typed server-request / image fixture 交接复核 | 独立 Electron `image-command` fixture 复现：turn 已进入 RuntimeCore 并以 `tool_failure` 完成，但 `image_generate` task artifact 为 0、图片 fixture server 请求数为 0；GUI session 绑定文本 Fixture Provider，且无 console/page/invoke error，归属 provider/image route 并行写集，未在本车道修复。`npm run test:contracts`（773 generated types、299 client checks）、`npm run governance:legacy-report`（0 零引用候选/0 分类漂移/0 边界违规）、`git diff --check` 与 typed server-request 控制器定向 29/29 通过。旧 `agentSession/action/{replay,respond}`、`agentSession/runtimeEvents/append` 仍待 protocol/Renderer/Plugin owner 交接，保持 `deprecated -> deleted` / P0 OPEN_REF。 |

### 2026-07-22 MCP environment registry 启动边界收口

`lime-mcp` 新增唯一运行时 `McpEnvironmentRegistry`，默认只注册 Codex 语义的
`local` 环境；`McpClientManager::start_server` 在 transport 分派、stdio 进程创建和
streamable HTTP 连接前解析 `McpServerConfig.environment_id`。未知显式身份（例如
`remote`）返回 `ConfigError` 并保持 manager 空闲，禁止把配置身份降级成本机执行。
新增 stdio/streamable HTTP 两条 unknown-environment 不启动/不连接回归。远程 executor/backend 尚未接入，继续保持
`OPEN_REF`，不伪造注册或 compat fallback。

验证：`cargo test -p lime-mcp --lib` 144/144、`cargo test -p lime-agent --lib mcp -- --nocapture` 12/12、
`cargo test -p tool-runtime --lib mcp_connection -- --nocapture` 10/10、局部 rustfmt、`git diff --check`。
分类：环境 registry/启动解析为 `current`；从 `cwd` 猜环境或未知环境本机 fallback 为
`dead / deleted / forbidden-to-restore`；无新增 `compat/deprecated`。

同日补齐 Codex startup deadline 语义：stdio transport 删除旧的 `max(configured, 60s)`
隐式放大，startup timeout 现在完全由 `McpServerConfig.startup_timeout` 决定，与
streamable HTTP 一致。该命令特定 fallback 属于 `dead / deleted / forbidden-to-restore`，
无兼容层。stdio 子进程环境也改为 Codex 的平台核心变量 allowlist + 显式 `env` 覆盖，
不再继承宿主全部环境；Windows 白名单同步包含系统盘、Program Files、AppData 与
PowerShell 核心变量。复验：MCP 145/145、Agent MCP 12/12、tool-runtime MCP 10/10。

### 2026-07-22 typed replay gateway 迁移切片

Renderer `replayAgentRuntimeRequest` 现在只从审批/AskUser typed server-request controller
的当前 pending snapshot 重建 `action_required` view，并按 `(session/thread, domain requestId)`
校验作用域；没有当前 typed request 时返回 `null`，不调用或扫描旧
`agentSession/action/replay`，也不把 JSON-RPC outer id 与 domain request id 混用。旧
Renderer `replayAction` 客户端依赖、正向 facade 测试和 contract guard 已改为 typed
pending / fail-closed 断言；协议层 DTO/方法仍由共享 owner 保持，尚未删除协议本身。

验证：typed replay + thread client 定向测试 33/33、Renderer/Node `npm run typecheck`、
`npm run test:contracts`（773 generated types、299 client checks）、
`npm run governance:legacy-report`（0 零引用候选、0 分类漂移、0 边界违规）、Prettier 与
`git diff --check` 通过。`src/lib/api/agent.test.ts` 全文件复跑为 55/68，剩余 13 条是
并行 session/thread canonical DTO 迁移造成的既有 fixture 失败，本切片未夹写。分类：typed
pending replay / fail-closed gateway 为 `current`；Renderer 旧 `action/replay` 调用面为
`deprecated -> deleted`，协议/dispatch 生产消费者仍是 P0 OPEN_REF；无新增 compat。

### 2026-07-22 typed response 短路切片

Renderer `respondAgentRuntimeAction` 现在先按 `(session/thread, domain requestId,
actionType)` 查找审批/AskUser typed server-request pending snapshot；显式传入
`action_scope` 时还会逐字段校验。命中后直接 settle 对应 typed controller，由现有
reverse JSON-RPC dispatcher 使用 outer id 回包，不再重复发送
`agentSession/action/respond`。Renderer Plugin runtime 的 `submitHostResponse` 同步复用该
短路，并保留可注入 responder 方便隔离测试。Renderer chat 与 Plugin 未命中 typed
pending 时均显式 fail closed，已删除 generic client 依赖、旧 DTO 转换、Plugin App
Server gateway 委托与 Page/Agent Run 正向断言；domain `requestId` 不会冒充 JSON-RPC
outer id。Plugin capability catalog 外的 `lime.agent.respondAction` 旧别名也已删除，只有
`submitHostResponse` 能进入 typed responder。MCP elicitation 继续走独立 typed
controller，不复用本短路。

验证：typed replay + thread client 前序定向 44/44；Renderer Plugin client/host/Page
Bridge/Agent Run 本轮 38/38；Plugin dispatcher 当前方法/旧 alias 负向场景 1/1；
Renderer/Node `npm run typecheck`、
`npm run test:contracts`（773 generated types、299 client checks）、
`npm run governance:legacy-report`（0 零引用候选、0 分类漂移、0 边界违规）、Prettier 与
scoped `git diff --check` 通过。分类：typed response 短路为 `current`；旧
`agentSession/action/respond` 的 Renderer chat/Plugin 生产入口为 `dead / retired
guard-only`，Plugin SDK `respondAction` alias 为 `dead / deleted / forbidden-to-restore`。
共享 package/Rust protocol/dispatch、只有 domain requestId 的 Electron
Plugin task host 仍为 `deprecated -> deleted` / P0 OPEN_REF，由对应热区 owner 继续迁移；
无新增 compat。`agentSession/runtimeEvents/append` 继续等待 current typed Item/artifact
writer，未猜测转换旧 runtime-event payload。

### 2026-07-22 MCP stdio process ownership 收口

`lime-mcp` 的 stdio 启动现在为每个本地子进程绑定 current `StdioProcessHandle`；Unix
进程组由 Codex 风格的 TERM -> 延迟 KILL 终止，Windows 走 `taskkill /T /F`。wrapper
持有 process handle 与 stderr reader 生命周期，`stop_server`、初始化失败、startup
timeout 和 wrapper drop 都不再只依赖 rmcp 直接子进程清理。旧未接管的
`McpClientWrapper.process` 字段和 `kill_process` 入口删除，无 compat 包装。
macOS/Unix PATH 补全也收回 `environment.rs` 的最小环境 owner；显式 `PATH` 覆盖保持
优先，Windows 不再误走 Unix 的 glob/冒号拼接分支。

新增 Unix 真实 stdio fixture：完成 MCP initialize 后启动后台孙进程，`stop_server` 验证
整个进程组退出；同时覆盖 stderr 持续排空。验证：`cargo test -p lime-mcp --lib`
147/147、`cargo test -p lime-agent --lib mcp -- --nocapture` 12/12、
`cargo test -p tool-runtime --lib mcp_connection -- --nocapture` 10/10、局部 rustfmt、
`git diff --check`、`npm run governance:legacy-report`（0 零引用候选、0 分类漂移、0
边界违规）。分类：本地 stdio process ownership/stderr lifecycle 为 `current`；未接管
直接子进程的旧 wrapper 字段、只杀 leader 的 stop 语义为 `dead / deleted / forbidden-to-restore`；无
`compat/deprecated`。未运行 GUI/contract 门禁，因为本刀未改 Electron、App Server 或
JSON-RPC protocol。下一刀仍是 Codex `McpRuntimeContext` 的真实 remote executor/backend，
不得注册假 remote 或回退本机执行。

### 2026-07-22 MCP stdio launcher placement 骨架

`McpEnvironmentRegistry` 从环境名集合收敛为 typed placement registry，当前只发布
`McpEnvironment::Local`；`McpClientManager` 在 transport 启动前解析 placement，并将本地
stdio 的 command/args、最小环境、cwd、进程组、stderr reader 与 process handle 创建统一
委托给 `LocalStdioLauncher`。manager lifecycle 只保留环境分派、startup deadline、rmcp
serve 和连接发布/错误事件，不再内联持有本地 spawn 细节。

验证：`cargo test -p lime-mcp --lib` 148/148、
`cargo test -p lime-agent --lib mcp -- --nocapture` 12/12、
`cargo test -p tool-runtime --lib mcp_connection -- --nocapture` 10/10、MCP 相关文件
`rustfmt --check`、scoped `git diff --check` 与 `npm run governance:legacy-report`
（0 零引用候选、0 分类漂移、0 边界违规）通过。分类：typed local placement 与
`LocalStdioLauncher` 为 `current`；manager 内联 spawn/environment/process ownership 为
`dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。真实 remote executor/backend
仍为 `OPEN_REF`，当前 registry 不注册 `remote`，未知环境继续在任何 stdio/HTTP I/O 前
fail closed。未运行 GUI/contract 门禁，因为本刀未触及 Electron、App Server 或 JSON-RPC
边界。

### 2026-07-22 MCP per-server parallel execution 收口

`McpServerConfig.supports_parallel_tool_calls` 不再只停留在管理状态展示：该能力经
`McpBridgeSnapshot -> McpConnectionRegistry -> McpStepSnapshot` 固化到当前 sampling step
的每条 MCP route，registry replacement 后旧 snapshot 的并发策略保持不变。Agent turn
snapshot 将未 opt-in 的 MCP 工具登记为 serial；provider 允许 parallel tool calls 时，
`agent-runtime` 使用 Codex 同语义的读写门，让 opt-in 工具共享执行、serial 工具对同批
其它调用保持独占。未 advertised 的工具继续 fail closed 且不得并发执行；非 MCP 工具的
既有策略本刀不改变。

验证：`cargo test -p tool-runtime --lib mcp_connection -- --nocapture` 11/11、
`cargo test -p agent-runtime provider_turn::tests:: -- --nocapture` 21/21、
`cargo test -p lime-agent --lib mcp -- --nocapture` 12/12、
`cargo test -p lime-mcp --lib` 148/148、scoped `rustfmt --check`、
`git diff --check` 与 `npm run governance:legacy-report`（0 零引用候选、0 分类漂移、
0 边界违规）通过。分类：per-server parallel capability、immutable step policy 与执行门为
`current`；只展示配置但执行阶段按模型全局开关无条件并发的旧语义为
`dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。未运行 GUI/contract 门禁，
因为本刀未改 Electron、App Server 或 JSON-RPC。MCP 真实 remote executor/backend 继续为
`OPEN_REF`，不得用本地执行 fallback 伪造。

### 2026-07-22 MCP tool lifecycle environment identity 消费收口

`McpStepSnapshot` 现在按已发布 runtime tool name 暴露 immutable `environment_id`；Agent
turn capture 将该映射与 serial policy 一起固化到 `RuntimeToolStepSnapshot`，缺少 route
provenance 时 fail closed。`agent-runtime` 构造 canonical `ToolCall` 时使用当前 sampling
step 捕获的 MCP environment identity，started/completed lifecycle evidence 不再把所有已
发布 MCP 工具硬编码成 `local`；native 与未发布工具继续显式使用 `local`。registry 替换
后旧 step 保持原 environment identity。

验证：`cargo test -p tool-runtime --lib mcp_connection -- --nocapture` 11/11、
`cargo test -p agent-runtime provider_turn::tests:: -- --nocapture` 22/22、
`cargo test -p lime-agent --lib mcp -- --nocapture` 12/12、scoped `rustfmt --check`、
`git diff --check` 与 `npm run governance:legacy-report`（0 零引用候选、0 分类漂移、0
边界违规）通过。`npm run smoke:mcp-current` 也通过：命中
`app_server_handle_json_lines` 与 7 个 current MCP read methods，legacy MCP 命令命中为 0。
分类：step environment identity 的 lifecycle 消费为 `current`；已发布 MCP 调用统一写死
`local` 的旧 evidence 语义为 `dead / deleted / forbidden-to-restore`；无
`compat/deprecated`。真实 remote executor/backend 与 foreign cwd 仍为 `OPEN_REF`；当前
lifecycle 保留 host working directory 只作为既有 `PathBuf` 占位，不把它声明为 remote
cwd，也不注册假 remote 或回退本机执行。

### 2026-07-22 Plugin worker current owner 接线骨架

Renderer Plugin task 的 v2 `turn/start` 已把业务 metadata 从不合法的嵌套
`responsesapiClientMetadata` 迁到 application-kind `additionalContext.metadata`；前者只保留
Rust `HashMap<String, String>` 可接收的 scalar trace/task identity。`PluginRuntimePage` 从
已安装 manifest 计算 worker contract，`AgentRuntimeCapabilityHost` 只为 manifest 声明的
task kind 生成受信任 `plugin.paneAction`，contract blocker 或缺 output kind 均 fail closed，
普通 Plugin Agent task 继续进入 provider 主链。

Electron `PluginRuntimeTaskHost` 同步退化为 App Server 委托壳：不再 spawn Node worker，
不再调用 `agentSession/runtimeEvents/append`，而是把同一 application metadata 交给
RuntimeCore 的 current plugin worker owner；`submitHostResponse` 也不再回退 generic
`agentSession/action/respond`。Rust 继续拥有 installed state/signature gate、entrypoint、
process timeout、output contract、`artifact.snapshot` 内部事件和 Thread/Turn/Item read model。

验证：Plugin typed client/Host 23/23、Plugin Runtime Page/Host Bridge 18/18、Electron task
host 8/8，相关 ESLint 与 scoped `git diff --check` 通过。workspace `tsc --noEmit` 被并行
canonical DTO/test fixture 改动的大量既有错误阻塞，本轮目标文件无报错。分类：Rust worker
和 typed `additionalContext` lowering 为 `current`；Electron IPC 壳为待删除的
`deprecated`；零生产引用的 `electron/pluginTaskWorker.ts` 为 `dead / delete-pending`，待取得
文件删除确认后物理删除并补回流守卫。共享协议/dispatch 与编辑器 artifact snapshot 的
`agentSession/runtimeEvents/append` 仍为 P0 OPEN_REF；编辑器必须等待 typed artifact writer，
不得复用本次 worker lowering。`scripts/check-app-server-client-contract.mjs` 正由并行车道修改，
本切片未夹写；其 Plugin Electron 正向守卫需要由脚本 owner 改成 absent guard。

### 2026-07-22 MCP replacement in-flight 快照守卫

新增真实 stdio MCP 受控启动回归：replacement generation 在 initialize 阶段被外部闸门
暂停时，`AgentRuntimeState::mcp_runtime` 仍返回已发布的旧 generation；放行 initialize、
完成 tools discovery 后，才原子发布含新 bridge 的 generation。该测试直接守住现有
`build -> start -> publish` 顺序，避免后续 refresh 把半初始化 manager 或空 snapshot 暴露
给并发 sampling step。required replacement 失败继续保留旧 generation，optional sibling
失败继续允许健康 bridge 发布。

验证：单 case 1/1、`cargo test -p lime-agent --lib mcp -- --nocapture` 13/13、scoped
`rustfmt --check` 与 `git diff --check` 通过。分类：replacement in-flight 旧快照可读与成功
后的原子切换为 `current`；启动中提前清空/替换 current runtime 的语义为
`dead / forbidden-to-restore`；无 `compat/deprecated`。本刀只补 owner 级并发证据，没有修改
生产代码或进入 App Server/protocol/Renderer 热区。Codex configured/runtime/effective
overlay 与真实 remote executor/backend 仍为 `OPEN_REF`。

### 2026-07-22 ThreadGoal Renderer 唯一 owner 收口

聊天 Goal 已按 canonical `thread_id + ThreadGoal` 收敛到
`useAgentSessionThreadGoal -> ThreadGoalPanel`；Harness detail、Inputbar inline 和 TaskRail 共享同一
identity 规则，mismatch fail closed，EmptyState 不再从 session/workspace id 猜测 Goal。旧
`ManagedObjectivePanel`、Inputbar inline wrapper、criteria/audit/continue 组件与模型已物理删除；
Harness 不再读取 `threadRead.managed_objective` 或触发旧 read-model refresh。五语言聊天文案迁到
`agentChat.threadGoal.*`，只为 Automation 暂留其仍动态消费的 9 个
`agentChat.managedObjective.status.*` 键，不建立聊天兼容层。

验证：九个直接相关 Vitest 文件 228/228；扩展 ThreadGoal/Harness 集合 298/298；`npm run
typecheck`、`npm run test:contracts`（781 schema definitions / 773 generated types / 300 client
checks）、`npm run governance:legacy-report`、scoped Prettier 与 `git diff --check` 均通过。`npm run
verify:gui-smoke` 真实 Electron Gate B 通过，证据
`.lime/qc/project-gates/standalone-shell-01-20260722100448-79233/shell-01-electron-smoke/summary.json`。
`smoke:agent-runtime-current-fixture` 的 history/stream/guard 与前六个 Electron 场景通过，随后因并发
GUI smoke 争用导致 approval Electron launch 失败；释放并发后单独重跑
`approval-request-resume` 已通过并生成 current fixture evidence，未把资源争用记为产品失败。

分类：canonical ThreadGoal hook/panel/identity/i18n 为 `current`；聊天 ManagedObjective UI、criteria/
audit/continue 语义为 `dead / deleted / forbidden-to-restore`；聊天无 `compat/deprecated`。Automation
旧 Objective 控制面仍是独立删除写集，下一刀应按前端 -> protocol/runtime -> repository -> 文档/守卫
顺序整体清退，禁止回挂 ThreadGoal。架构图确认：责任开发者 root，2026-07-22，依赖方向仍是
`Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item -> GUI`。整体 Codex
v1 对齐仍约 40%。

### 2026-07-22 ManagedObjective 全链删除

目标：在无客户、无兼容负担前提下，删除 Codex 不存在且实际断链的 v0 `ManagedObjective` 平行系统，
保留 canonical `ThreadGoal`、`thread/goal/set|get|clear`、Goal tools 和 idle automatic continuation。
事实核对确认 Automation 只投影 `request_metadata.harness.managed_objective`，生产没有
`automation_job` owner 的写入；旧 audit 查询因此不是可交付链路。

窄写集分为四条互斥车道：root 负责 Automation/capability/renderer 与架构计划；Rust owner 负责
v0 protocol、App Server runtime/local data source 和 core repository；TS protocol owner 负责
App Server renderer client 与 package client；scripts owner 负责 smoke/helper/i18n 和
forbidden-to-restore 守卫。明确避让并行模型路由、Task Center、Workspace shell 与长期 Electron/Vite/
App Server 进程，不触碰 `internal/refactor/v1`。

架构分类：canonical thread-owned `ThreadGoal` 为 `current`；Automation schedule/run/history 仍为
`current`；`ManagedObjective` DTO、owner kind、criteria/audit/evidence、手动 continue、六个 v0 RPC、
renderer projection、repository/table 和专属 smoke 为 `dead / deleted / forbidden-to-restore`；无
`compat/deprecated`。架构图确认：责任开发者 root，2026-07-22，唯一依赖方向仍是
`Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item -> GUI`。

完成结果：手写 production Rust/TS/Electron 旧 Objective 残留为 0；协议 schema 与 generated TS 共
19 个文件、2173 行纯删除，canonical `ThreadGoal` schema/client、idle continuation、fork deferral 和
per-turn budget/usage accounting 均保留。repo-wide forbidden-to-restore 守卫已覆盖
`ManagedObjective`、`managed_objective`、`managed_objectives`，两份 Agent Workspace roadmap 和两份
active 执行计划已纠正 current owner，dated history 只保留为 evidence。本切片完成度 100%；整体 Codex
v1 对齐仍约 40%。

验证：前端定向 7 文件 101/101；guard 定向 8/8；Rust protocol/client/core/App Server `cargo check`
通过且无 warning；schema fixture 1/1；package client 7 files 75/75；`npm run typecheck`、`npm run
test:contracts`（296 checks）、`npm run governance:legacy-report`（0/0/0）、`npm run governance:scripts`、
`npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke` 全部通过。Gate B evidence：
`.lime/qc/project-gates/standalone-shell-01-20260722111626-84163/shell-01-electron-smoke/summary.json`。
首次 schema writer 编译因隔壁 Cargo 共用 `lime-rs/target` 的 rmeta 争用退出，未进入目录写入；改用
`.lime/target/objective-schema` 隔离 target 后重跑通过，不构成产品 blocker。

### 2026-07-22 canonical ordered UserMessage 冷读与 fork/restart 闭环

目标：把 UserMessage 的唯一 canonical payload 收敛为有序 `AgentInput[]`，贯通
`RuntimeEvent -> ThreadItem -> SQLite snapshot/rebuild -> v2 thread/read -> fork/restart provider history`；
无客户与历史兼容负担，不保留 scalar `content` 双轨或空内容 merge。

完成结果：`ThreadItemPayload::UserMessage.content` 已改为 `Vec<AgentInput>`；materializer 保留
Text/Image/LocalImage/Skill/Mention 顺序、detail 与 text elements；v2 projection、canonical read model、
projection rebuild、ThreadStore cold snapshot、fork provider lowering 与 synthetic terminal event 均输出完整
ordered input。Conversation import 的 current canonical item producer 同步写入 `input`，旧空 content 回填已从
ThreadStore 与 projection merge 删除，并补 fail-closed 非法/空 UserMessage 校验和 merge 回流回归。

验证：`agent-protocol` payload JSON round-trip `1/1`；`thread-store` snapshot rebuild `1/1`；App Server
`user_message` owner 组 `16/16`（含 import duplicate、materializer、projection、cold restart、fork 与空 merge
回流）；`npm run test:rust:related -- ... -- user_message` `16/16`；公共 JSON-RPC
`thread_fork_rebuilds_provider_history_across_restarts_without_duplicate_prefix` `1/1`；
`npm run test:contracts` 通过（756 generated protocol types 无漂移、296 client checks）；`npm run typecheck`、
scoped rustfmt、`git diff --check`、`npm run governance:legacy-report`（0/0/0）均通过。
Electron/GUI 证据复用并行车道当前产物：
`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-home-hotpath-regression-summary.json`
与 `.lime/qc/project-gates/standalone-shell-01-20260722130218-30380/shell-01-electron-smoke/summary.json`；
本轮未重复启动整套 fixture，避免占用隔壁 Electron 进程，且本切片未改 Renderer/Bridge。

分类：ordered UserMessage canonical payload、import producer、ThreadStore/read model、fork/provider lowering
均为 `current`；scalar payload、空 content merge 与图片拒绝旧逻辑为 `dead / deleted / forbidden-to-restore`；
无 `compat/deprecated`。并行避让：`typed_tests/incremental.rs` 的 reasoning repeated-fragment 与
`media_task_worker/route.rs` 未触碰；隔壁对 `change_set.rs` 的 reasoning merge 改动已保留，本切片只删除其
UserMessage 空内容回填分支。架构图仍为 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore ->
Thread/Turn/Item projection -> GUI`。下一刀回到 provider abort/cancel usage、tool-finish/abort flush、
idle accounting、自动 continuation 与 GUI structured owner；整体 Codex v1 对齐约 40%。

### 2026-07-23 ThreadHistoryBuilder Turn/Item 投影一致性

目标：在不触碰并行 App Server/protocol 热区的前提下，修复 Codex-first canonical history
reducer 的 item-first、item update 与 cold snapshot 嵌套投影不一致。raw `ThreadItem` 仍是唯一
item owner，`Turn.items` 只由 reducer 从 raw items 重建，不新增兼容层或第二份持久化事实源。

完成结果：`append_items_at` 在新增和更新后同步已存在的 `Turn.items`；`append_turns_at` 与
`apply_change_set` 在 turn 后到时挂回已有 raw items，并在 rollback 后重建剩余 turn 的嵌套投影；
重复更新不会产生嵌套 item 重复，snapshot 的 `Turn.items` 与 raw item 集合保持一致。新增 item-first
到 turn、item update 去重两项回归测试。

验证：`thread-store` 全量 35/35；App Server cold-resume raw JSONL 回归 1/1；scoped `cargo fmt
--check -p thread-store`、`git diff --check`、`npm run governance:legacy-report`（0 零引用候选 / 0
分类漂移 / 0 边界违规）和 `npm run test:contracts` 全部通过。workspace rustfmt 仍被并行既有的
`agent-protocol/src/lib.rs`、`app-server/src/runtime/agent_mailbox_delivery.rs`、
`app-server/tests/session_archive_jsonrpc.rs` 格式漂移阻塞，未夹写这些文件。

分类：ThreadHistoryBuilder raw/Turn projection 为 `current`；无新增 `compat/deprecated`，旧的
不同步 nested projection 行为为 `dead / deleted / forbidden-to-restore`。架构图未改变，仍为
`Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`。
下一刀回到 provider abort/cancel usage 与 canonical token usage/replay owner；`cache_write_input_tokens`
仍需等待共享 App Server/protocol 热区释放后一次性贯通，不能半截新增 wire 字段。

### 2026-07-23 agent-runtime cache-write usage 累计

目标：先在不修改 App Server/protocol wire 的前提下，把 provider 已解析的 cache-write token
贯通到 Codex-first turn-scoped runtime usage，避免 provider 终止、取消和正常完成路径在 runtime
内部丢失该值。

完成结果：`RuntimeSessionTokenUsage` 新增 `cache_write_input_tokens`；session context/input
handle 的累计 API 使用饱和加法；`run_current_provider_turn` 将 provider
`cache_creation_input_tokens` 映射为 Codex 语义的 cache-write usage。旧三字段累计行为没有保留
第二套兼容 API。

验证：`agent-runtime` 全量 167/167；定向 session usage 与 provider cancellation 回归均通过；
`cargo fmt --check -p agent-runtime`、`git diff --check` 通过。该字段尚未进入 App Server
canonical snapshot、v2 JSON-RPC 或 resume replay，避免在并行 `lib.rs`/protocol 热区半截落 wire。

分类：runtime usage 累计为 `current`；provider cache-write 在 runtime 内被静默丢弃的路径为
`dead / deleted / forbidden-to-restore`；无新增 `compat/deprecated`。下一刀待共享 owner 释放后，
一次性接入 `thread_usage`、v2 `TokenUsage`、resume projection 与持久化回归。

### 2026-07-23 provider failed/canceled usage flush

目标：补齐 provider step 在 usage 已到达、但随后取消、超时或 provider error 时的 runtime
usage flush，避免只有正常 `ProviderStep` 才累计 token。

完成结果：`run_current_provider_turn` 使用单一 `record_session_token_usage` helper；取消分支
会保留当前 usage event 并写入 session runtime，step timeout、stream error、provider error
也会 flush 已观察到的最新 usage。正常完成路径复用同一 helper，cache-write 与 input/output
保持一次性累计。

验证：`agent-runtime` 全量 171/171；新增
`cancellation_flushes_provider_usage_to_the_session_runtime` 与
`provider_error_flushes_prior_usage_to_the_session_runtime`、
`provider_step_timeout_flushes_prior_usage_to_the_session_runtime`、
`stream_error_flushes_prior_usage_to_the_session_runtime` 回归通过；scoped rustfmt、
`git diff --check` 与 `npm run governance:legacy-report`（0 零引用候选 / 0 分类漂移 / 0 边界违规）
通过。App Server canonical snapshot/v2 wire 尚未修改，仍等待并行协议 owner 释放。

分类：provider terminal usage flush 为 `current`；失败/取消后静默丢 usage 为
`dead / deleted / forbidden-to-restore`；无新增 `compat/deprecated`。下一刀仍是共享
`thread_usage`/v2/resume canonical owner 的一次性贯通。

### 2026-07-23 App Server cold-resume replay 顺序证据（并行车道交接）

并行 App Server 车道在未跟踪测试
`lime-rs/crates/app-server/tests/thread_resume_replay_jsonrpc.rs` 中补齐 raw JSONL cold-resume
回归：重启后 `thread/resume` response 必须先于 `thread/tokenUsage/updated`、
`thread/goal/updated`，再允许 live turn event；同时断言 canonical thread identity、token usage
快照和不出现重复 `thread/started`。本车道只读取并运行该产物，未修改其文件或 App Server 热区。

验证：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server --test
thread_resume_replay_jsonrpc -- --nocapture` 1/1 通过。该证据仍未覆盖
`cache_write_input_tokens` 的 v2 wire 字段；生产字段贯通继续等待
`thread_usage`、protocol v2 和 resume projection 的共享 owner 释放。

### 2026-07-23 并行收口编译门禁

在不抢写隔壁 App Server/protocol 热区的前提下，运行
`cargo check --manifest-path "lime-rs/Cargo.toml" --workspace --locked`，workspace 全量通过。
当前唯一输出是既有 `lime-cli/src/video.rs` 未使用 `SharedTaskWriteArgs` warning，不影响编译；
Electron/App Server 长驻进程继续保留运行。

### 2026-07-24 cache-write usage current projection 收口

目标：在不恢复旧 runtime 或新增 compat 层的前提下，将 Codex canonical
`cache_write_input_tokens` 从 provider/runtime usage 一次性贯通到 App Server v2、冷恢复、
`thread/read` 历史投影和 Renderer GUI usage。GUI 既有 `cache_creation_input_tokens` 只作为
边界显示字段，不成为第二份 token usage 事实源。

完成结果：App Server `TokenUsageBreakdown`/schema/generated client 已消费
`cacheWriteInputTokens`；Renderer `thread/tokenUsage/updated` projector 与历史 normalizer
将 canonical 字段映射到既有 GUI prompt-cache 写入展示；`read_model_turn_usage` 的
`thread/read` turn usage 同步输出 canonical snake_case。未新增调试日志、CDP probe、临时脚本、
兼容包装或平行 usage 类型。v2 envelope 稳定名单同步纳入现有的 command output、file patch 与
plan delta 三类 current notification，避免 schema 已扩展但稳定断言仍停留在旧集合。

验证：前端 projection/consumer/history 27/27；App Server `read_model_turn_usage` 2/2；
App Server cold-resume public JSON-RPC 1/1；protocol v2 round-trip 1/1；`npm run test:contracts`
通过（762 generated types、296 client checks）；`npm run test:rust:related --
lime-rs/crates/agent-runtime lime-rs/crates/app-server-protocol lime-rs/crates/app-server` 覆盖 20 个
受影响及反向依赖 crate 并通过；`npm run verify:gui-smoke` 通过并生成 standalone Electron Gate B
evidence；`rustfmt --check` 与全树 `git diff --check` 通过。`npm run governance:legacy-report` 报告
0 零引用候选、0 分类漂移、0 边界违规；
`npm run smoke:agent-runtime-current-fixture` 聚合 Gate B 已完成，所有 summary 均为
`ok=true`、错误数为 0，覆盖首页首发/短问候、取消后继续、审批、图片、Skills/MCP、历史恢复和
coding workbench 场景。

分类：runtime/canonical/v2/resume/read-model/Renderer projection 为 `current`；原先在
runtime、wire、history 或 GUI 边界静默丢弃 cache-write 的行为为 `dead / deleted /
forbidden-to-restore`；无新增 `compat/deprecated`。架构图未改变，仍为
`Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`。
下一刀回到 provider abort/cancel 对齐、重复 reasoning terminal 语义及真实 fixture 的 Gate B 收口，
不要重新引入本轮清除的临时诊断 surface。

### 2026-07-24 V1-25 MCP progress 调用归属与并行协调

目标：关闭 connection-wide progress subscription 导致不同 MCP tool call 串流的缺陷，让
`RequestHandle.progress_token`、调用级 subscriber、Runtime route identity 与 public
`item/mcpToolCall/progress` 使用同一调用事实，不保留双轨。

当前实现与验证：RMCP token A/B dispatcher 与真实 duplex 重叠调用隔离通过；MCP 151/151、Agent
lib 271/271、Tool Runtime 271/271、七个相关/反向依赖 crate 的 Rust related、public JSON-RPC
2/2、Renderer 42/42、两个 typed client 80/80 与 23/23、contracts、MCP current smoke、治理扫描和
全树 diff check 均通过。默认 MCP smoke 直接复用隔壁已经运行的 Desktop Host/DevBridge；检测到
`npm run electron:dev` 和真实 Lime 进程后，没有重复启动 Agent fixture 或 GUI smoke。

并行阻塞：完整 Agent 测试只剩独立 `legacy_permission_surfaces` 25/27；两条失败来自已退出生产编译图
的旧字符串权限表和 shell 启发式，不属于 MCP 回归。按无兼容口径应物理删除
`tool_permissions.rs`、`shell_security.rs`、旧 integration fixture，并把当前正向保活 guard/PRD
改成禁止恢复；文件删除等待明确危险操作确认。V1-25 在删除门禁与 Agent/GUI smoke 补齐前不标记
完成。typed `artifact/write` 与 `agentSession/runtimeEvents/append` 删除已由下一节闭环；MCP owner
不得重复实现该切片，后续只处理本节列出的 MCP 并发、首帧与 Gate B 缺口。

### 2026-07-24 typed `artifact/write` 生产链与旧 append 删除闭环

目标：用 `artifact/write` 的 typed snapshot/response 取代 Artifact Workbench 对
`agentSession/runtimeEvents/append` 和 generic RuntimeEvent response projector 的依赖，保持
`Renderer typed gateway -> App Server JSON-RPC v2 -> RuntimeCore -> ThreadStore/artifact read`
单向主链，不新增 compat wrapper。

完成结果：Rust v2 protocol/processor 继续作为唯一 write owner；`packages/app-server-client` 的
request client 与 `AppServerConnection`、Renderer `AppServerClient` 均新增 typed
`writeArtifact(ArtifactWriteParams)`。Artifact Workbench 保存请求只提交 canonical `threadId`、
可选 `turnId` 与 `ArtifactSnapshot`，不再构造客户端任意 `RuntimeEvent`；保存证据直接消费
`ArtifactWriteResponse.eventId/sequence/persistedAt/sidecar`。`sessionId` 只保留在本地持久化
scope 和尚未迁移的 `artifact/read` 读回边界，不用于 v2 write 路由。public JSONL integration
新增 `initialize -> artifact/write -> artifact/read` 回归，证明 sidecar 正文可读且没有
`agentSession/event` wrapper。

验证：protocol round-trip 1/1、App Server processor artifact write 1/1、public JSON-RPC 1/1；
package client 82/82；Renderer App Server/Artifact client 57/57，Workspace evidence 2/2；
`npm run typecheck`、`npm run check:protocol-types`（775 definitions / 767 generated types / 0 漂移）、
`npm run test:contracts`（296 checks）及 scoped Prettier/rustfmt/diff check 通过。

删除结果：`agentSession/runtimeEvents/append` 的 package/Renderer wrapper、Rust v0
protocol/handler/catalog/schema/fixture 和旧正向测试已物理删除；公共 JSON-RPC 对旧 method 返回
`METHOD_NOT_FOUND`，package 不再导出旧常量或 helper，治理 catalog 已增加 Rust dead surface 回流
守卫。Content Factory fixture 已迁到两次 typed `artifact/write`，不再用 arbitrary runtime event
伪造 workflow/error 状态。

Runtime terminal policy 同刀收口：generic external/internal append 在终态 turn 上统一 `Reject`；durable
recovery 与 typed `append_artifact_snapshot` 才可显式 `Allow`。`artifact/write` processor 只能调用该
领域方法，不能提交任意 `RuntimeEvent`，因此保留终态后编辑器保存 artifact 的产品语义，但关闭了
公共事件注入后门。processor 精确终态保存测试 1/1、public `artifact/write -> artifact/read` 1/1、
package artifact 3/3、Renderer/App Server artifact 56/56、fixture guards 91/91、protocol codegen
零漂移、`npm run test:contracts` 与治理扫描均通过。

分类：typed protocol/client/Renderer writer、领域 artifact append 与 public transport 为 `current`；
`agentSession/runtimeEvents/append` 为 `dead / deleted / forbidden-to-restore`；无新增
`compat/deprecated`。本切片未改变仓库架构方向，架构图确认状态为 `confirmed`。

真实 Electron Gate B 已关闭：`direct-session` external backend 只发 message/tool/file/command/turn
事实，不发 `artifact.snapshot`；canonical Turn 完成后，fixture 经 Renderer current
`AppServerClient.writeArtifact` 与 production `safeInvoke` 进入
`app_server_handle_json_lines / electron-ipc / artifact/write`。trace 中的 `threadId`、`turnId`、
`artifactRef` 与 typed response、read model 完全一致，随后 GUI hydrate 并打开 Workbench。证据为
`.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/typed-artifact-write-v2-summary.json`、
`typed-artifact-write-v2-workbench.png` 与 `typed-artifact-write-v2-backend-ledger.json`；summary
`ok=true`，`typedArtifactWriteElectronIpcTrace=true`，backend 未注入 Artifact，console/page error 均为
0。fixture guard 7/7、protocol 764 types 零漂移、contracts 296 checks、scripts governance、
`governance:legacy-report`（0 零引用候选 / 0 分类漂移 / 0 边界违规）和全树
`git diff --check` 通过。本切片状态为 `closed`，旧 append 删除不再列为 OPEN_REF。

V1-25 状态校正（隔壁 MCP owner 写集，本车道只读）：request-token 生产链已接通，但现有双调用证据仍受
`ElicitationOwnerGate` 全调用持锁影响，未证明服务端 `max_in_flight == 2`；请求发送后才注册
progress subscriber 也保留首帧竞态。V1-25 继续保持“进行中”，退出前必须补真实重叠调用断言、
消除 send-before-subscribe 竞态，并取得 `RMCP -> Agent -> App Server -> Renderer/GUI` Gate B。

### 2026-07-25 `thread/name/set` current 切片

按 Codex `ThreadSetNameParams` / `ThreadSetNameResponse` 补齐 Lime v2 current owner：
`thread/name/set { threadId, name } -> {}`，名称采用 trim 后非空校验，持久化委托
`ThreadStore::update_thread_metadata`，并发出 typed `thread/name/updated`。同步了 v2
method/envelope/notification、schema registry 与 checked-in fixtures、generated TypeScript
client、App Server dispatcher 和 connection/request client method；没有新增兼容层，也没有恢复
`agentSession` 命名入口。

验证：`cargo check -p app-server --tests`、`thread_name_jsonrpc` 1/1、`app-server-protocol`
v2 定向测试 30/30、schema fixture 1/1、`npm --prefix packages/app-server-client test` 82/82、
`npm run test:contracts`（764 generated protocol types、286 checks）和 `git diff --check` 均通过。
分类：v2 name method/notification、RuntimeCore、ThreadStore 和 typed client 为 `current`；无
`compat/deprecated`。下一刀回到 Codex P0 `thread/loaded/list` / `thread/metadata/update` 或
typed `thread/closed`，继续补真实 App Server/GUI 消费证据。隔壁持有的
`internal/refactor/v1/fixtures/codex-method-product-scope.v0.1.json` 当前仍把该 method 列为
`planned`；本车道不夹写，交接时应由矩阵 owner 将其移入 `implemented` 并重算计数。

### 2026-07-25 Codex method 产品范围矩阵骨架

本车道只修改 `internal/refactor/v1/**`、既有治理测试与本协调记录，避让隔壁发布、Cargo、
provider、runtime 和 approval 热区。对照 Codex
`codex-rs/app-server-protocol/src/protocol/common.rs` 注册表，建立
`codex-method-product-scope.v0.1.json`：213 个方向化 method 无遗漏、无重复，分类为
`50 implemented / 130 planned / 33 product-scope-excluded`。`implemented` 只接受 Lime
generated manifest 中同方向、同名契约；Codex account/attestation/remote-control、test-only、
internal raw response 和 deprecated v1 surface 明确排除，不保留 compat。

验证：Codex source registry 逐项比对 `213/213`；矩阵 Vitest `3/3`；Prettier、
`git diff --check`、`npm run test:contracts`（761 types、286 checks）、
`npm run governance:legacy-report`（0 零引用候选、0 分类漂移、0 边界违规）通过。
本切片只完成范围事实和守卫，不宣称 method 语义、字段、恢复或 GUI parity 已完成；下一刀回到
P0 history lineage（compact/rollback/fork/replay）并逐项把真实 current owner 从 `planned` 移入
`implemented`。

### 2026-07-25 `thread/compact/start` current 切片

目标：直接删除旧 `agentSession/compact` 产品契约，以 Codex exact
`thread/compact/start { threadId } -> {}` 取代；请求 ack 与执行状态分离，压缩状态只通过标准
`turn/*`、`item/*` lifecycle 投影，不新增 compat event 或 response payload。

本车道认领 compaction 相关 App Server protocol/runtime、typed client、Renderer gateway、schema、
scope matrix、current 文档与回流守卫；避让 release、Cargo manifest/lock、approval fixture 和其它
并行脏热区。切片已收口：`thread/compact/start` 进入 protocol v2、App Server、RuntimeCore、typed
client、Renderer gateway、schema 与标准 `turn/*`、`item/*` lifecycle；旧
`agentSession/compact` 生产契约和 generated types 已物理删除，并由负向测试禁止回流。Codex 已将
`thread/rollback` 标为 deprecated / will be removed，Lime 将其归为
`product-scope-excluded`，不新增公开 rollback。范围矩阵现为
`51 implemented / 128 planned / 34 product-scope-excluded`。

验证：`cargo check -p app-server-protocol -p app-server --tests`；App Server compaction runtime
`3/3`；public JSON-RPC `3/3`；protocol v2 `24/24`；schema fixture `3/3`；protocol codegen
`761/761`、0 drift；Renderer/API/governance 定向 Vitest `80/80`；typed client `82/82`；scope
matrix guard `4/4`；`npm run test:contracts`（761 types、286 client checks 与完整 command/docs
boundary）；`npm run governance:legacy-report`（0 零引用候选、0 分类漂移、0 边界违规）；
`npm run smoke:agent-runtime-current-fixture`；`npm run verify:gui-smoke`；旧 compact 名称生产源码
零残留。两条 GUI 命令均取得真实 Electron/App Server Gate B，fixture 明确
`liveProviderUsed=false`。

治理分类：`thread/compact/start`、RuntimeCore、typed client 与标准 lifecycle 为 `current`；
`agentSession/compact` 为 `dead / deleted / forbidden-to-restore`；`thread/rollback` 为
`product-scope-excluded`；`compat / deprecated` 为零。架构确认：confirmed；本切片没有改变既定
owner 或依赖方向，继续遵循
`Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item -> GUI`，无需修改
架构图。下一刀回到 P0 fork/replay durable lineage。

### 2026-07-25 `thread/loaded/list` current 切片

目标：按 Codex v2 建立 `thread/loaded/list { cursor?, limit? } -> { data, nextCursor }`，只读取
`RuntimeCoreState.sessions` 这一内存 owner，不把 durable `ThreadStore`、projection 或 renderer cache
变成第二份 loaded 事实源。

完成结果：v2 method/envelope/schema、App Server dispatcher/processor、Rust 与 TypeScript typed client
均已接通；loaded 快照过滤 hidden/internal session，按 thread id 字典序排序并去重，支持 Codex stale
UUID cursor 插入点语义与 `limit.max(1)`。非法 cursor 精确返回 `INVALID_REQUEST (-32600)`。
`thread/start`、public `thread/fork` 与 AgentControl product spawn 已直接替换旧前缀双 ID，统一生成
UUIDv7 且 `sessionId == thread.id`，loaded cursor 因而保持 Codex 的严格 UUID 校验，没有兼容映射。
`thread/delete` 会移除 loaded owner；
`thread/archive` 现在先 shutdown/close 已加载 runtime owner，再从内存 sessions 移除并归档 durable
thread，与 Codex archive-unload 语义一致。

验证：public `thread_loaded_list_jsonrpc` 2/2，覆盖 hidden、空列表、两页分页、stale cursor、
`limit=0`、malformed cursor、delete、archive unload，以及 cold fork `thread/read` 不加载、显式
`thread/resume` 才进入 loaded owner；protocol v2 31/31；schema fixture 1/1；Rust
client 26/26；TypeScript client 83/83；AgentControl graph/restart/fork 36/36；
`npm run test:contracts`（774 definitions / 766 generated types / 286 client checks）通过；
`npm run governance:legacy-report` 为 0 零引用候选、0 分类漂移、0 边界违规；全树
`git diff --check` 通过；`npm run verify:gui-smoke` 通过真实 Electron/App Server sidecar 初始化、
reload、Workbench shell、Memory Settings 与 evidence summary。生成器由 `/root` 单点顺序执行，未与
`internal/refactor/v1/**` owner 夹写。架构确认：confirmed；本切片只补既定 App Server current
主链能力，没有改变架构图方向。

fork/replay 补充收口：`thread/read` 已删除 fork-specific hydration 写副作用；v2 `turn/start`、
`turn/interrupt` 与 `thread/compact/start` 只解析 `RuntimeCoreState.sessions` 中已加载的 thread，冷启动
请求按 Codex 返回 `INVALID_REQUEST`，不再通过 durable read 暗中 resume。`thread/resume` 是唯一 public
fork hydration owner：先从 canonical full Thread 重建 fork seed，再按 sequence 合并 target EventLog
增量，恢复 turn input/runtime options/output references，避免第二次冷启动从 `N+1` 单独回放造成历史
断裂。unloaded ThreadGoal mutation 只持久化，不启动 live continuation；mailbox durable recovery 保留其
内部显式 projection hydration，不扩散 public fork fallback。定向验证：fork/restart provider history
`3/3`，loaded-list `2/2`，manual compact `3/3`，fork compaction `1/1`，ThreadGoal continuation `2/2`。
Rust related 的 App Server lib `1498/1498`；contracts 为 `766` generated protocol types、`286` client
checks 且零漂移；治理扫描为 `0` 零引用候选、`0` 分类漂移、`0` 边界违规；
`smoke:agent-runtime-current-fixture` 与 `verify:gui-smoke` 均通过真实 Electron/App Server sidecar，
`cargo fmt --all --check` 与全树 `git diff --check` 通过。

分类：`thread/loaded/list`、内存 sessions owner、typed clients 与 archive unload 为 `current`；旧
`thread_*` / `sess_*` public start/fork identity 及 `agent-*` / `thread-*` AgentControl product
identity，以及 `thread/read` 隐式 hydration 为 `dead / replaced / forbidden-to-restore`；无新增
`compat/deprecated`。该 OPEN_REF 已关闭；下一刀回到 P0 fork/replay durable lineage 的完整 typed
history/reconnect evidence。

### 2026-07-25 public `thread/fork` usage 与 ThreadGoal replay 顺序收口

目标：按 Codex public fork wire 固化 source canonical prefix 的累计 token usage 与继承 Goal
快照，使 fork response、即时通知和冷重启 resume replay 使用同一 target identity 与顺序；不复制
source raw EventLog，不新增兼容包装。

完成结果：`RuntimeCore::fork_thread` 只在选中的 canonical `turn_ids` 内读取最后一个完整
`thread_usage` snapshot，重写为 target `thread.token_usage` durable event，sequence 接在 canonical
fork history 后，并同时写入 target EventLog 与 loaded `StoredSession.events`。因此 target 冷重启
`thread/resume` 能重建同一 usage，而 `excludeTurns=true` 不产生即时 usage。公共 processor 现在严格
发送 `response -> thread/tokenUsage/updated -> thread/started`；当 `deferGoalContinuation=true`
且 target 确有继承 Goal 时，继续发送 `thread/goal/updated`，无 Goal 不误发 `updated/cleared`。
`thread/started.turns` 保持空列表，resume 不重发 `thread/started`。

验证：public JSONL `thread_resume_replay_jsonrpc` `2/2`（usage 顺序、Goal 顺序、excludeTurns
负向、target 冷重启 replay）；`thread_fork_jsonrpc` `3/3`；App Server lib `1498/1498`；
`npm run test:contracts`（774 definitions / 766 generated protocol types / 286 client checks，
零漂移）；`npm run governance:legacy-report`（0 零引用候选 / 0 分类漂移 / 0 边界违规）；
`cargo fmt --all --check` 与 scoped `git diff --check`；`npm run smoke:agent-runtime-current-fixture`
与 `npm run verify:gui-smoke` 均通过真实 Electron/App Server sidecar，fixture 明确
`liveProviderUsed=false`。架构确认：confirmed；本刀只补既定 App Server current 主链通知与
durable replay，没有改变 owner 或依赖方向。

治理分类：fork usage event、通知 sequencer、ThreadGoal inherited snapshot、冷恢复 replay 与
paginated source metadata-first 拒绝为 `current`；source raw EventLog 回拷与 fork-specific
hydration fallback 继续为 `dead / forbidden-to-restore`；无新增 `compat/deprecated`。仍待下一刀：
mid-turn fork 的 interrupted snapshot、完整 typed reconnect/history evidence，以及 provider request
的 `forked_from_thread_id` provenance。

### 2026-07-25 mid-turn fork 与 Responses provenance 闭环

目标：按 Codex canonical Thread/Turn/Item 语义补齐 active turn fork，不复制 source raw EventLog，
并把 fork lineage 从 canonical `Thread.forked_from_id` 单向传到 OpenAI Responses turn metadata；
Chat Completions 与 Anthropic 不发送该字段。

完成结果：mid-turn fork 从 canonical snapshot 生成 target interrupted Turn，source active Turn 保持
不变；target 冷重启/refork 保留完整 synthetic seed 与连续 sequence，interrupted developer marker
恰好一次。`forked_from_thread_id` 只沿
`ExecutionRequest -> AgentSessionConfig -> CurrentProviderRequestMetadata -> CanonicalRequest.metadata`
传递，Responses 将 identity/provenance 放入 `client_metadata["x-codex-turn-metadata"]` JSON 字符串；
保留键不能被 extra metadata 覆盖，且不把 lineage 写回 Thread/StoredSession/runtime metadata。

验证：App Server mid-turn fork lib `4/4`；public `thread_fork_jsonrpc` `4/4` 与
`thread_resume_replay_jsonrpc` `2/2`；Responses metadata `1/1`；Agent Runtime canonical lineage
`1/1`；public `thread_fork_midturn_jsonrpc` `1/1`；`cargo check -p app-server --tests` 通过。
分类：canonical interrupted snapshot、durable replay 与 Responses provenance 为 `current`；raw
EventLog 拷贝、Thread metadata lineage 副本和非 Responses wire provenance 为
`dead / forbidden-to-restore`；无 `compat/deprecated`。该 P0 缺口已关闭。

### 2026-07-25 provider/model 原子 session route 切换

目标：对齐 Grok model switch 的单一 session route owner，修复 `thread/settings/update` 只能改
model、跨 provider 后仍继承旧 `providerSelector` 的缺口。App Server JSON-RPC、RuntimeCore session
actor、canonical metadata 与下一 Turn 必须观察同一个 provider/model 对。

完成结果：`ThreadSettingsUpdateParams` 新增 typed `modelProvider`；显式 provider 变更必须同时提交
非空 `model`，单独 model 更新仍用于同 provider 切换。session actor 在一次 metadata transaction
中更新 `providerSelector`、`providerName`、`modelName` 与 collaboration mode model；任何 model
route 变化都会删除旧 `agentControlRoute` resolved snapshot，使下一 Turn 重新执行 provider readiness、
credential/capability 与 lowering 解析，而不是复用旧 provider client 事实。active Turn 不被中断，
后续 Turn 和 EventLog 冷恢复后的 Turn 均只读取新 route；没有增加兼容字段或第二套 route store。

已验证：public `thread_control_jsonrpc` `5/5`，覆盖 active/subsequent Turn、零 Item 持久化、显式
resume 冷恢复、provider/model request capture 与 fail-closed；App Server route/collaboration unit
`1/1`；App Server Protocol lib `70/70`；schema fixture `1/1`；protocol codegen `766/766` 零漂移；
TypeScript typed client `83/83`。分类：v2 protocol/schema/client、session actor 与 canonical metadata
为 `current`；旧 resolved route snapshot 在切换时为 `dead / invalidated`；无
`compat/deprecated`。收尾门禁：`npm run test:contracts` 通过（`766` generated types、`286`
client checks、命令/模态/脚本/文档边界全通过）；`npm run governance:legacy-report` 为 `0`
零引用候选、`0` 分类漂移、`0` 边界违规；`npm run smoke:agent-runtime-current-fixture` 与
`npm run verify:gui-smoke` 均通过真实 Electron/App Server sidecar，fixture 明确
`liveProviderUsed=false`；`cargo fmt --all --check` 与全树 `git diff --check` 通过。架构确认：
confirmed；本轮只扩展既有 v2 settings contract 和 RuntimeCore owner，没有改变架构图方向。
下一刀是把 circuit breaker 从 per-session client 提升为按
`provider/model/base-url/protocol/credential-scope` 共享的 route health registry，避免相同 credential
route 的不同 session 各自学习故障状态，同时不让一个 key 的 429/5xx 污染另一个 key。

### 2026-07-25 shared provider route health registry

目标：对齐 Grok 按 upstream route 复用 circuit breaker 的控制面语义，消除同一
`provider/model/base-url/protocol/credential-scope` 在不同 session client 间各自累计故障的偏差；不共享 HTTP
client、WebSocket、HTTP fallback 或 credential，避免把 session transport 状态和凭证边界混入
route health。

完成结果：`model-provider` 新增可注入、无全局静态状态的
`CurrentProviderHealthRegistry`。key 规范化 provider、model、base URL（URL parser 统一 host、
scheme、默认 port 与尾随 slash）、显式 protocol 与 credential scope；持久化凭证以 UUID 分隔，direct
runtime key 只以 SHA-256 指纹分隔，registry 不保留 raw key。空 base URL 仍按 wire protocol 解析为
current default upstream。`CurrentProviderClient::new_with_health_registry` 只从 registry 取得 breaker，
并继续独立创建 HTTP client、WebSocket 与 fallback state。`AgentRuntimeState` 持有并 clone 同一个
registry，session route 不变时复用现有 client；route 改变时替换 client，但切回原 route 会复用该
route 已有的 health entry。

验证：`cargo test -p model-provider` 为 169/169，覆盖规范化同 route 共享，以及 model/base URL/
protocol 三维隔离；`cargo test -p lime-agent provider_session_tests --lib` 为 3/3。后者以真实 local
HTTP fixture 连续 10 次 429 打开 A route，切到同 provider/base URL 的 B model 后仍允许网络请求，
切回 A route 则在网络前被拒绝，fixture 精确收到 11 次而非 12 次请求。credential scope 隔离 health
state；未共享 raw API key、HTTP client、WebSocket 或 fallback state。

扩大验证：`npm run test:rust:related -- lime-rs/crates/model-provider lime-rs/crates/agent` 以退出码 0
通过，覆盖反向依赖图中的 `model-provider`、`lime-agent`、`agent-runtime`、`app-server`、
`tool-runtime` 等 13 个 crate；services 中 4 个既有本地 TCP/联网用例保持显式 ignored。`npm run
smoke:agent-runtime-current-fixture` 通过 history/cache、stream terminal、86 个 fixture guard 以及完整
Electron Gate B，包含 Claw、Coding Workbench、approval、Skills、MCP、media 和 Article Editor current
路径，`liveProviderUsed=false`。`cargo fmt --all --check`、`git diff --check` 和
`npm run governance:legacy-report`（0 零引用候选 / 0 分类漂移 / 0 边界违规）通过。

分类：route health registry、`model-provider` client 注入与 `AgentRuntimeState` registry owner 为
`current`；旧 per-session 独立 breaker 构造为 `dead / replaced`；无 `compat/deprecated`。架构确认：
confirmed；本刀只在既有 `model-provider -> agent runtime` 边界内补 route health state，没有改变
`Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item -> GUI` 主链。
下一刀回到 Codex P0 history/reconnect 的 typed product evidence，provider 侧继续优先补 capability
lowering 和 route readiness 的可观测性，不再引入第二套 provider owner。

### 2026-07-25 typed reasoning capability 与 canonical lowering 闭环

目标：对齐 Codex 的 canonical reasoning effort，并按 Grok/OpenCode 的多模型控制面把服务商展示值与
provider wire value 分离。App Server capability 必须提供 typed menu DTO；GUI 只提交 canonical
`value`，不得把展示 id 或无类型 JSON 传入 session/turn；provider lowering 继续由
`model-provider` 单一 owner 负责。

完成结果：新增 `ModelReasoningEffortOptionInfo` 与 `ModelReasoningEffortSupportInfo`，包含 typed
`id/value/label/description/default`、支持级别与来源；`ModelCapabilitiesInfo.reasoning_effort`、App
Server projection、RuntimeCore capability snapshot 和 Renderer registry/policy 已统一消费该 DTO。
前端删除 opaque capability cast 和旧 `{ effort: ... }` 菜单对象兼容解析；provider 自定义展示项
例如 `Deep` 会提交 canonical `xhigh`，不会提交 UI id `deep`。`options` 为空时只保留 App Server 从
current `levels` 投影的字符串项，不引入第二套前端默认表。

lowering 闭环已确认：canonical value 经 `thread/settings/update` 或 `turn/start` 进入
`RuntimeProviderConfig.reasoning_effort`；OpenAI Responses 发送 `reasoning: { effort: "xhigh" }`，
Chat Completions 发送 `reasoning_effort: "xhigh"`。Anthropic thinking 是预算制，当前不伪造
`reasoning_effort`，继续 fail closed，后续如启用必须新增独立 capability/lowering，而不是复用 OpenAI
字段。契约守卫同步更新为共享 route health 构造签名
`create_configured_reply_provider(config, &self.provider_health)`，旧单参数签名不再作为正向事实源。

验证：`cargo check -p app-server --tests`；App Server projection `2/2`、model capability `10/10`、
RuntimeCore model task `4/4`、protocol schema fixture `1/1`；Renderer 定向 Vitest `53/53`；
`cargo fmt --all --check`、`npm run check:protocol-types`、`npm run test:contracts`（generated client 零
漂移、App Server client `286` checks）、`npm run governance:legacy-report`（0 零引用候选 / 0 分类漂移 /
0 边界违规）与 `git diff --check` 均通过。`npm run smoke:agent-runtime-current-fixture` 完整通过真实
Electron、preload/IPC、`app_server_handle_json_lines`、App Server 与 GUI Gate B；evidence 断言
`externalFixtureBackendUsed=true`、`liveProviderNotUsed=true`、`noMockFallbackHits=true`、
`noLegacyCommandHits=true`，不把受控 external provider 误报成 live provider。`npm run
verify:gui-smoke` 也通过 renderer build、真实 Electron host/preload、App Server sidecar、Claw shell
reload 与 Memory Settings，summary `result=pass`。

全量 `tsc --noEmit` 仍被当前工作树既有 Thread/Inputbar/Skill/AppServer test 类型漂移阻塞，错误未指向
本切片 model registry 文件；本切片不扩散修复无关热区。分类：typed reasoning DTO、canonical menu
value、projection、GUI consumption 与 provider lowering 为 `current`；无类型 capability JSON、旧
`{ effort }` 菜单对象解析和旧 provider guard 签名为 `dead / deleted / forbidden-to-restore`；无新增
`compat/deprecated`。架构确认：confirmed；owner 与依赖方向未变化，无需修改架构图。

### 2026-07-25 Codex model switch reasoning policy 真实消费闭环

目标：关闭前端已有 `resolveModelReasoningEffortForModelSwitch` 但生产选择器未消费的断点，使模型切换
严格按 Codex catalog 顺序归一化 reasoning effort；不再由 GUI 固定选择 `medium`、第一档或仅清空。

完成结果：`ModelSelector` 的初始 catalog reconcile、模型点击和选中态统一复用
`modelReasoningPolicy` owner。新模型支持当前 canonical value 时保留；不支持或当前为空时选择 provider
声明顺序的中位档，再退到模型声明 default；新模型没有 typed reasoning menu 时清空。Grok 风格展示
`id/label` 仍只提交 canonical `value`。组件的 `setModel` 与 `setReasoningEffort` 继续由
`useAgentContext` 的 microtask pending owner 合并为一次原子 `thread/settings/update`，没有新增前端 route
store、provider 名称推断或兼容层。

验证：Renderer model taxonomy/registry/reasoning policy/selector 定向 Vitest `72/72`；
`useAgentContext` 已有 current session 原子 provider/model/effort 回写回归；Renderer TypeScript
`tsc --noEmit`、Prettier 与 scoped `git diff --check` 通过。分类：Codex model switch policy、typed
catalog 顺序和 canonical value 为 `current`；GUI 固定 `medium`、第一档 fallback 及未消费策略为
`dead / replaced`；无 `compat/deprecated`。架构确认：confirmed；本刀未改变 owner 或依赖方向，无需
修改架构图。下一刀继续补 provider readiness/capability lowering 的可观测性与跨 session model switch
真实产品证据，不新增第二套多模型 owner。

### 2026-07-25 thread settings route preflight 收口

目标：关闭 `thread/settings/update` 先持久化未知模型、未就绪 provider 或 unsupported reasoning
effort，直到下一 Turn 才失败的控制面缺口。模型路由变更必须在 RuntimeCore session actor 的同一串行
操作内先通过生产 `RuntimeBackend` readiness/catalog/capability route，再写 canonical thread metadata；
不得用 profile fallback 冒充用户选择成功。

完成结果：session actor 先在内存中生成候选 `AgentSession + ThreadSettings`，仅当 model、provider、
effort 或 collaboration mode 可能改变路由时调用 `ExecutionBackend::preflight_thread_settings`。生产后端
复用下一 Turn 的 `resolve_turn_route`，精确校验 provider/model、credential readiness、model catalog
和 reasoning capability；route fallback、未知模型和 unsupported effort 均返回 typed
`RuntimeCoreError`，失败不会修改内存或 projection metadata。cwd、approval、sandbox、personality 等
非路由设置继续直接更新，不引入额外 provider 请求。并发重命名残留的旧
`preflight_model_switch` 测试调用已删除，统一使用短领域名 `preflight_thread_settings_route`。

验证：`cargo test -p app-server model_switch --lib` 为 `2/2`，覆盖 ready + catalog 已知模型成功、未知
模型失败、unsupported effort 失败，以及失败后旧 provider/model metadata 完全不变；`cargo check -p
app-server --lib`、`cargo fmt --all --check` 和全树 `git diff --check` 通过。`npm run test:contracts`
通过（`774` definitions / `766` generated protocol types / `284` client checks，命令契约零漂移）；
`npm run governance:legacy-report` 为 `0` 零引用候选、`0` 分类漂移、`0` 边界违规。此前同一工作树的
current fixture 已完成 19 个真实 Electron/App Server 场景，均为 `ok=true`、无 legacy/mock fallback，
且 `liveProviderUsed=false`；该 Gate B 证明主链接线，不冒充本失败分支的 live provider 证据。

分类：候选 settings、session actor preflight、生产 route resolver 和 typed failure 为 `current`；
“先持久化、下一 Turn 再失败”与 model/profile fallback 为 `dead / replaced / forbidden-to-restore`；无
`compat/deprecated`。架构确认：confirmed；本刀复用既有 RuntimeCore、model-provider route 与
Thread metadata owner，没有改变架构图方向。该多模型 P0 已关闭；后续 P1 依次是 enabled-provider
过滤后的 `model/list`、`modelProvider/capabilities/read` 产品范围决策，以及 Multi-Agent per-agent
model/reasoning/service-tier 覆盖，不在本刀扩散。

### 2026-07-25 enabled-provider model catalog 收口

目标：按 Grok/OpenCode 的 available provider -> model catalog 关系关闭 `model/list` 的平行旧目录，
保证 Renderer 只能看到当前 enabled provider 声明或实时缓存的模型；disabled provider 与已下线本地
`model_registry` 表均不得成为选择器事实源。

完成结果：App Server local data source 不再先读取 `ModelRegistryService` 的空内存 registry，再追加
provider 模型，而是直接从 `ApiKeyProviderService::get_all_providers` 过滤 enabled provider，按请求的
`providerId/tier` 追加 `custom_models` 与 provider-scoped 实时缓存并去重。`lime-services` 删除从未被
生产填充的 `models_cache` 字段，以及零剩余消费者的 `get_all_models`、
`get_models_by_provider`、`get_models_by_tier`、`search_models` 和本地打分逻辑，共移除 138 行 dead
surface；Provider 实时缓存、typed taxonomy/reasoning capability 与 alias owner 保持不变，没有新增
Renderer filter、compat wrapper 或第二套 catalog。

验证：新增 public `model_list_jsonrpc` 从 `initialize -> model/list -> RuntimeCore ->
LocalAppDataSource` 断言 enabled provider 模型可见、disabled provider 在全量和 provider-scoped 查询中
均不可见，并在旧 `model_registry` 表写入 enabled provider 的陈旧模型验证其不能回流，`1/1` 通过；
App Server inline `list_models_*` 为 `2/2`；`lime-services model_registry_service` 为 `63/63`，同时覆盖
provider cache 与前序 typed reasoning capability。`npm run test:contracts` 通过（`774` definitions /
`766` generated types / `284` client checks，零协议漂移）；`npm run governance:legacy-report` 为 `0`
零引用候选、`0` 分类漂移、`0` 边界违规；`cargo fmt --all --check` 与全树 `git diff --check` 通过。

分类：enabled provider store、declared model、provider-scoped cache 与 public `model/list` 为
`current`；空 `models_cache`、本地 registry catalog 读取和 Rust `search_models` 为
`dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。架构确认：confirmed；本刀删除平行
目录并强化既有 App Server owner，没有改变架构图方向。该 P1 已关闭；剩余多模型 P1 是
`modelProvider/capabilities/read` 产品范围决策与 Multi-Agent per-agent model/reasoning/service-tier
覆盖。

### 2026-07-25 model provider capabilities current contract（首次接线，语义已 superseded）

目标：按 Codex v2 精确实现空参数 `modelProvider/capabilities/read`，并把 capability 映射留在
`model-provider` 唯一 owner。App Server 从 current `routing.default_provider` 取得 provider ID，
精确解析 provider registry 中的实际 provider type；不得按显示名猜测、回退第一个 enabled provider
或为未知类型补兼容默认值。

完成结果：新增 Codex-shaped v2 params/response 与 typed envelope，App Server public JSON-RPC、
RuntimeCore、LocalAppDataSource、Rust/TypeScript clients 和 generated schema 已接通。默认 provider
capability 与 Codex 一致为 `namespaceTools/imageGeneration/webSearch = true/true/true`；AWS Bedrock
为 `true/false/false`。缺少默认 provider、空 ID 或未知存储类型全部 fail closed；读取前会校验
数据库原始 provider type，避免旧 DAO 的 OpenAI fallback 把损坏记录伪装成可用能力。method 只登记
在 protocol v2，由 central catalog 自动合并，并标记为 `SharedRead`；没有恢复 v0 DTO/enum。

验证：`model-provider` capability unit `3/3`；App Server LocalDataSource unit `2/2`；public
`model_provider_capabilities_jsonrpc` `1/1`；Rust `app-server-client` check；protocol schema fixture
`1/1`；TypeScript client `83/83`；scope matrix guard `4/4`。`npm run test:contracts` 通过（`776`
schema definitions、`768` generated protocol types、`284` client checks，以及 command/modality/scripts/
release/docs boundary）；`npm run governance:legacy-report` 为 `0` 零引用候选、`0` 分类漂移、`0`
边界违规；`cargo fmt --all --check` 与全树 `git diff --check` 通过。范围矩阵现为
`52 implemented / 127 planned / 34 product-scope-excluded`，`permissionProfile/list` 仍单独保持
planned。

分类：v2 protocol/schema、provider capability mapping、App Server current owner 与 typed clients 为
`current`；provider 名称猜测、unknown-to-OpenAI capability fallback 和 v0 平行 DTO 为
`dead / forbidden-to-restore`；无 `compat/deprecated`。架构确认：confirmed；本刀只补既有
`App Server -> model-provider` 依赖方向，不改变架构图。该多模型 P1 已关闭；下一刀是 Multi-Agent
per-agent model/reasoning/service-tier 覆盖，并继续以 Codex runtime contract 为主、grok-build 控制面
为参考。

状态校正（2026-07-26，已再次 superseded）：本节仅保留首次 protocol/handler 接线的历史 evidence；
本日曾临时改为所有 `enabled + runtime-ready` chat provider 的 UI capability 上界并集。该 union 语义随后
被“current configured provider capability 语义校正”替换，不再是 current owner。

### 2026-07-25 Multi-Agent per-agent model controls（骨架已关闭）

目标：按 Codex Multi-Agent v2 为 `spawn_agent` 增加可选 `model`、`reasoning_effort` 与
`service_tier`。省略时继承父 Turn 的 effective route；显式覆盖时必须在创建 child session、graph edge
或 mailbox 前，复用 production `ExecutionBackend` 完成 provider/model route 与 reasoning capability
校验。service tier 作为 typed generation config 进入 `model-provider` lowering，不以任意 metadata 或
第二套 Multi-Agent config owner 传递。

窄写集：`tool-runtime/src/agent_control.rs`、App Server protocol/runtime 的 typed runtime request、
`runtime/agent_control_gateway.rs` 及独立 spawn route helper、`lime-agent` provider configuration、
`model-provider` runtime config/lowering、这些 owner 的定向测试与生成 schema。现有
`agent_control_gateway.rs` 已超过 1000 行，本刀不得继续在主文件堆叠 route 业务逻辑；新增校验和覆盖
逻辑必须留在子模块，主文件只保留 command dispatch 接线。退出条件是：三种覆盖可独立或组合生效、
非法 model/reasoning 在 durable 写入前 fail closed、service tier 到达 OpenAI wire、默认继承与 cold
restart 不回退父 route，并通过相关 Rust tests、schema/contracts、legacy guard 和 diff/fmt 检查。

分类预期：per-agent typed overrides、production route preflight、child durable defaults 与 provider
lowering 为 `current`；未校验字符串直传、父 route snapshot 覆盖 child 显式选择、平行 v0 DTO 为
`dead / replaced / forbidden-to-restore`；不新增 `compat/deprecated`。该刀不改变既有 owner 与依赖方向，
只把缺失配置贯通 current 主链；架构图预期无需修改，完成后再次确认。

结果：`spawn_agent` 已支持可选 `model`、`reasoning_effort`、`service_tier`，省略时继承父 Turn
effective route；显式 model/reasoning 在 child session、graph edge 与 mailbox 写入前走 production route
preflight，并拒绝 model fallback 与 reasoning downgrade。service tier 已贯通 queued/session durable defaults、
`lime-agent` provider configuration、`model-provider` identity/lowering，并以 OpenAI Chat/Responses 原生
`service_tier` 字段发送；cold restart 继续恢复 child tier 与 durable route。验证：Rust tests target 编译、
AgentControl 37/37、tool-runtime 7/7、spawn preflight 2/2、provider lowering 1/1、schema fixture 1/1、
768 个 TypeScript protocol types 无漂移、contracts 284 项、legacy report 0 零引用候选/0 分类漂移/0
边界违规、Agent current fixture Gate B 与全树 diff/fmt check 均通过；`liveProviderUsed=false`。分类确认：
上述实现均为 `current`；字符串直传、父 route 覆盖 child 显式选择与平行 v0 DTO 为
`dead / replaced / forbidden-to-restore`；无 `compat/deprecated`。架构确认：confirmed，既有
`App Server -> RuntimeCore -> model-provider` owner 与依赖方向未变。后续细节是用 model catalog
`service_tiers` 对显式 tier 做精确成员校验，并向动态 tool description 注入可用 model/effort/tier；本刀不
宣称这两项 catalog 精度已经完成；这两项由下一节关闭。

### 2026-07-26 Multi-Agent catalog 精度与动态工具描述收口

目标：关闭上一节保留的两项 catalog 精度缺口。`spawn_agent.service_tier` 必须像 Codex 一样只接受
当前 resolved provider/model catalog 明确列出的 tier；模型可选项、reasoning effort/default 和 service
tier 必须进入同一 Turn 的动态 tool definition snapshot。不得按 provider/model 名称猜测 tier，不得在
child session、spawn edge、identity 或 mailbox 写入后才失败，也不得把整份 provider catalog 或 secret
作为 durable route metadata 持久化。

完成结果：`EnhancedModelMetadata` 增加 typed `visibility`、`service_tiers` 和
`default_service_tier`，Provider `/models` ingestion 只保留明确声明并按 id 去重的 tier，未知 default
fail closed。App Server 从当前 resolved provider 的 `model/list` 构造最多 5 个 Codex 风格
`spawn_agent` model override 描述，只列 `visibility=list` 模型，并展示 reasoning levels/default 与
service tier id；gateway 持有每 Turn 不可变 snapshot。显式 tier preflight 同时要求 resolved route 回显
完全一致、`modelRegistry.status=matched` 且 tier 精确属于选中模型；失败 reason code 为
`spawn_agent_service_tier_unsupported`。

durable `agentControlRoute` 不保存原始 `modelRegistry`，只投影 `status=matched`、选中模型的 typed
`service_tiers` 与列表内合法 default；重复/缺 id tier、endpoint、credential 和未知 catalog 字段全部丢弃。
新增集成回归证明 unsupported tier 在 durable child session、spawn edge、identity 和 mailbox 之前拒绝，
并证明显式 model/reasoning/tier 覆盖后的最小 route snapshot 可被 child cold defaults 继续使用。

验证：App Server AgentControl `38/38`，其中 route 安全投影 `2/2`、显式覆盖 `1/1`、unsupported tier
零副作用 `1/1`；spawn runtime/tool options `4/4`；tool-runtime AgentControl `8/8`；services Provider
catalog conversion `15/15`；protocol schema fixture `1/1`；相关五个 Rust crate `cargo check --tests`
通过。schema/TypeScript 生成物为 `777` definitions / `769` protocol types，`npm run test:contracts`
通过（client `284` checks）；`npm run governance:legacy-report` 为 `0` 零引用候选、`0` 分类漂移、`0`
边界违规；`npm run smoke:agent-runtime-current-fixture` 全部通过，`liveProviderUsed=false`。范围矩阵 method
数量在本刀完成时仍为 `52 implemented / 127 planned / 34 product-scope-excluded`；后续
`model/safetyBuffering/updated` 收口后的最新计数见下一节。

分类：typed catalog、动态 tool snapshot、精确 preflight 与最小 durable route 为 `current`；按名称猜测
tier、任意字符串直传、整份 catalog 持久化和失败后补偿式 child 清理为
`dead / replaced / forbidden-to-restore`；无 `compat/deprecated`。架构确认：confirmed；本刀只强化既有
`model-provider/ModelRegistryService -> RuntimeBackend -> tool-runtime AgentControl` 数据流，没有新增
owner 或改变依赖方向，无需修改架构图。该 Multi-Agent model/reasoning/service-tier catalog 精度项已
关闭；剩余工作回到整体 Codex v2 产品范围矩阵的其他 planned methods 和 live provider 多模型证据。

### 2026-07-26 model safety buffering v2 notification 收口

目标：将已有 `provider_safety_buffering` runtime event 从 deprecated `agentSession/event` side-channel
迁入 Codex exact `model/safetyBuffering/updated`。只提升已有可靠 producer 的 safety buffering；没有
runtime 事实源的 `model/rerouted` 与 `model/verification` 继续保持 planned，不复制空壳 method。

窄写集：App Server protocol v2 model DTO、notification envelope/method/schema registry、App Server
v2 notification projector、scope matrix、生成协议和定向测试。投影必须要求非空 thread/turn/model、
严格字符串数组和布尔值；Lime runtime payload 的 `retryModel` 只在 wire lowering 时映射为 Codex
`fasterModel`。任何 malformed payload 都 fail closed，不得回退 `agentSession/event`。

实现结果：已新增 Codex 同形 `ModelSafetyBufferingUpdatedNotification`，typed envelope 与 JSON-RPC
round-trip 已接入；projector 将 current provider event 直接投影为 v2 通知，并保留 provider owner 内部
payload 命名。范围矩阵拆分为 safety buffering current 与 reroute/verification planned，最新计数为
`53 implemented / 126 planned / 34 product-scope-excluded`。

验证：protocol exact round-trip `1/1`、App Server safety buffering 定向测试 `3/3`、protocol v2
`32/32`、v2 notification projector `26/26`、schema fixture `1/1`、scope matrix guard `4/4` 全部通过。
schema/TypeScript 生成物为 `778` definitions / `770` protocol types，`npm run test:contracts` 通过
（client `284` checks）；`npm run governance:legacy-report` 为 `0` 零引用候选、`0` 分类漂移、`0`
边界违规；`cargo fmt --all -- --check` 与全树 `git diff --check` 通过。Agent Runtime current fixture
完成真实 Electron/App Server 主路径回归，全部场景通过，`liveProviderUsed=false`。

分类：typed v2 notification 与现有 provider event lowering 为 `current`；safety buffering 的
`agentSession/event` side-channel 为 `dead / replaced / forbidden-to-restore`；`model/rerouted` 与
`model/verification` 仍为 `planned`，无 `compat`。架构确认：confirmed；本刀只替换既有 App Server
通知投影出口，不改变 `model-provider -> RuntimeCore/App Server -> GUI` 方向，无需修改架构图。

### 2026-07-26 Thread method 范围事实校正

范围矩阵复核发现 `thread/loaded/list`、`thread/name/set` 与 `thread/name/updated` 已具备 exact v2
protocol/schema、App Server handler/projector、typed client 与公开 JSON-RPC 证据，却仍残留在
`planned`。本刀不新增行为，只删除该分类漂移：前两项移入 thread request current，name 通知移入
thread notification current；总 inventory 保持 `213`，最新计数为
`56 implemented / 123 planned / 34 product-scope-excluded`。

验证：`thread_loaded_list_jsonrpc` `2/2`、`thread_name_jsonrpc` `1/1` 通过。分类：上述三个 method
均为 `current`；无 `compat/deprecated`。下一刀实现 Codex exact `thread/status/changed`，状态只从已
成功解析的 lifecycle 与 action event 投影，覆盖 approval/user-input active flags，不用 transport
硬编码字符串冒充产品实现。

### 2026-07-26 `thread/status/changed` current 切片

按 Codex exact `ThreadStatusChangedNotification`、`ThreadStatus` 与 `ThreadActiveFlag` 补齐状态投影。
状态 owner 拆为 `app-server/src/processor/v2_notifications/thread_status.rs`，由 per-thread listener
持有：`thread/started` 静默登记 loaded，`turn.started` 投影 `Active`，terminal turn 投影 `Idle`；
`tool_confirmation` 与 `ask_user` action 以 request id 去重并分别投影 `waitingOnApproval` /
`waitingOnUserInput`，resolved/canceled/expired 释放对应 flag。Malformed action 不改变状态；没有可靠
unload producer 的 `thread/closed` 继续保持 planned。

已接入 v2 method/envelope/schema registry、checked-in JSON schema、generated TypeScript 和
notification projector，通知顺序按 Codex 在 Turn 生命周期之前发 status。范围矩阵最新计数为
`57 implemented / 122 planned / 34 product-scope-excluded`。

验证：protocol v2 `33/33`、v2 notification projector `28/28`、thread status unit `2/2`、公开
`thread_compact_jsonrpc` Active -> Idle `1/1`、loaded/name public tests `3/3`、schema fixture `1/1`；
生成物 `779 definitions / 771 protocol types`。`npm run smoke:agent-runtime-current-fixture` 完整通过
真实 Electron/App Server、approval、Skills、MCP、media 与 Article Editor 场景，
`liveProviderUsed=false`；`npm run verify:gui-smoke` 通过真实 host/preload、App Server sidecar、Claw
shell reload 与 Memory Settings。无 `compat/deprecated`；架构确认：confirmed；本刀只扩展既有
`App Server -> Thread listener -> Thread/Turn/Item -> GUI` 通知投影 owner，没有改变架构图。

### 2026-07-26 `thread/metadata/update` current 切片

按 Codex exact DTO 补齐 `thread/metadata/update`：Git metadata 使用 omission / `null` / string
三态 patch，非空字符串在边界 trim 并拒绝空值；`isPinned` 同步进入 `Thread`、`thread/list`
filter 和持久化投影。实现复用既有 `ThreadStore::update_thread_metadata`，只在写入时把历史
`git_info` 键直接收敛为 current `gitInfo`，保留其余 metadata；归档线程也允许更新，不新增
compat 或第二套 store owner。

已接入 v2 method、typed envelope、schema registry、generated TypeScript、typed client 和公开
JSON-RPC。公开测试覆盖首次写入、partial clear、归档更新、重启冷读、pinned list filter，以及
空 patch / 空 gitInfo / 空字符串 fail closed。范围矩阵最新精确计数为
`58 implemented / 121 planned / 34 product-scope-excluded`，产品范围完成度
`58 / 179 = 32.4%`。

验证：protocol v2 `35/35`、`thread_metadata_jsonrpc` `2/2` 通过；schema/TypeScript 生成物为
`782 definitions / 774 protocol types`。分类：v2 method、ThreadStore metadata patch、Thread
projection 和 typed client 均为 `current`；无 `compat/deprecated`。架构确认：confirmed；本刀只
补全 `App Server JSON-RPC -> RuntimeCore -> ThreadStore -> Thread projection` 既有主链，不改变
架构图。下一刀回到多模型 current owner，按 readiness 过滤修复 `model/list` 并实现
`modelProvider/capabilities/read` 的 enabled + runtime-ready provider 能力并集。

### 2026-07-26 多模型 runtime readiness 事实源收口（capability union 语义已 superseded）

目标：让 `model/list` 与 `modelProvider/capabilities/read` 复用 runtime routing 的 configured-provider
readiness，删除 capability read 对单一 `routing.default_provider` 的依赖。可见模型和能力上界只允许由
`enabled + runtime-ready` 的 chat provider 贡献；需要 key 但没有 enabled key、disabled、Fal 非 chat
provider 与未知 provider type 均 fail closed，Ollama 等显式 keyless provider 继续可用。

实现结果：`runtime_backend::model_routing::configured_provider_readiness` 成为 configured-provider
readiness 唯一 current 判定，`model/list` 与 capability 聚合共同委托该函数。capability read 现在遍历所有
runtime-ready provider，对 namespace tools、image generation、web search 做 OR union，不再读取 default
provider。Provider DAO 删除未知 type 静默降级为 `openai` 的两处 fallback；批量读取跳过非法记录，单条或
凭证联表读取返回类型转换错误，避免 runtime credential 旁路重新放行。

验证：App Server model provider 单元测试 `10/10`、runtime routing `5/5`、公开 `model/list` JSON-RPC
`1/1`、公开 capability JSON-RPC `1/1`、Core provider DAO `10/10` 通过；
`cargo check -p app-server --tests`、`npm run test:contracts`（`782` schema definitions / `774` protocol
types / client `284` checks）与 `npm run governance:legacy-report`（`0` 零引用候选、`0` 分类漂移、`0`
边界违规）通过，定向 rustfmt 与 diff check 通过。范围矩阵 method 数量不变，仍为
`58 implemented / 121 planned / 34 product-scope-excluded`，产品范围完成度 `58 / 179 = 32.4%`。

分类：共享 readiness、runtime-ready model catalog 和多 provider capability union 为 `current`；
default-provider 单模型 capability resolver、enabled-only model 泄漏与未知 type -> OpenAI fallback 为
`dead / replaced / forbidden-to-restore`；无 `compat/deprecated`。架构确认：confirmed；本刀只让既有
`model-provider/ModelRegistryService -> RuntimeBackend/App Server` 多模型控制面收敛到同一 readiness，
没有新增 network/runtime owner 或改变依赖方向，无需修改架构图。下一刀应补 live provider 多模型切换与
capability union 的真实 Gate B 证据，再回到范围矩阵中优先级最高的 planned Codex method。

状态校正（2026-07-26）：本节的 configured-provider readiness 与 `model/list` 过滤仍为 current；
`modelProvider/capabilities/read` 遍历全部 ready provider 做 OR union 的语义已由后文 current-provider
精确读取替换。不得再把全局能力上界当成 Codex capability method 的返回值。

### 2026-07-26 `thread/unsubscribe` current 切片

目标与写集：按 Codex exact connection-scoped 语义补齐 `thread/unsubscribe`，只修改 v2 protocol/schema、
App Server processor 与现有 `ThreadStateManager` 接线、Rust/TypeScript typed client、公开 JSONL transport
测试和 method 范围矩阵。退出条件为三态 response 可验证，unsubscribe 后 thread 仍 loaded，不停止 turn、
不立即卸载 thread、不伪造 `thread/closed`。

实现结果：`RequestProcessor` 复用 `AppServer` 持有的共享 `ThreadStateManager`；handler 只移除当前
transport connection 对目标 thread 的订阅。未加载 thread 清理残留 listener/subscription state 并返回
`notLoaded`，已加载但当前连接未订阅返回 `notSubscribed`，成功移除返回 `unsubscribed`。无 connection
context 的 direct request fail closed；没有新增 compat、第二套订阅表或 idle unload 空壳。

公开 JSONL 测试覆盖 `unsubscribed -> notSubscribed`、冷 UUID `notLoaded`、unsubscribe 后
`thread/loaded/list` 仍包含目标 thread，且不产生 `thread/closed`。范围矩阵只移动 1 个 exact method，
最新精确计数为 `59 implemented / 120 planned / 34 product-scope-excluded`，产品范围完成度
`59 / 179 = 33.0%`。分类：v2 method、共享 subscription owner 和 typed clients 均为 `current`；
`thread/closed` 继续 `planned`，无 `compat/deprecated/dead` 新增。架构确认：confirmed；本刀只补既有
`App Server transport -> RequestProcessor -> ThreadStateManager` 生命周期入口，不改变主链方向，无需修改架构图。

验证：protocol v2 `36/36`、schema/fixture `10/10`、Rust typed client 定向 `1/1`、公开
`thread_unsubscribe_jsonrpc` `1/1`、TypeScript client `85/85`、范围矩阵守卫 `4/4` 通过；
`cargo check -p app-server --tests`、`npm run test:contracts`（`788` schema definitions / `780` protocol
types / client `284 checks`）通过；`npm run governance:legacy-report` 为 `0` 零引用候选、`0` 分类漂移、
`0` 边界违规。

### 2026-07-26 `thread/closed` idle unload lifecycle

目标与写集：按 Codex exact 生命周期补齐 `thread/closed`，只修改 v2 notification protocol/schema、
共享 `ThreadStateManager`、App Server transport lifecycle、RuntimeCore idle unload、公开 JSONL 测试和
method 范围矩阵。默认延迟与 Codex 一致为 `30` 分钟；测试通过 App Server builder 注入短延迟。

实现结果：unsubscribe 与 transport close 只更新 connection-scoped subscription，不直接卸载或发
closed。共享 thread state 以 generation ticket 管理 idle unload；重订阅会使旧 ticket 失效，unloading
期间 resume/listener attach fail closed。只有 thread 连续无订阅且 inactive 满延迟、RuntimeCore 成功关闭
session loop/backend 并移除 loaded session 后，App Server 才按 `thread/status/changed { notLoaded } ->
thread/closed` 顺序广播给所有 initialized 且未 opt-out 的连接。archive/delete/graceful shutdown 不复用该
notification；active turn 不会被 idle unload 中断。

验证：protocol v2 `32/32`；公开 `thread_closed_jsonrpc` `1/1` 与 `thread_unsubscribe_jsonrpc` `1/1`；
重订阅取消 ticket、active turn 拒绝 unload、reconnect replay 定向测试各 `1/1`；`cargo check -p
app-server --lib` 和全 workspace rustfmt check 通过。生成物为 `789` schema definitions / `781` protocol
types。最终契约、治理与全 tests 门禁见本节后续验证记录。

最终验证（当前合并工作树）：`cargo check -p app-server --tests`、protocol v2 `37/37`、schema fixture
`1/1`、TypeScript client `85/85`、Renderer 发送/steer/adapter/build 定向 `120/120`、范围矩阵守卫
`4/4`、`npm run test:contracts`（`786` schema definitions / `778` protocol types / `284` client checks）、
`npm run governance:legacy-report`（`0` 零引用候选、`0` 分类漂移、`0` 边界违规）、workspace rustfmt 与
`git diff --check` 通过。生成计数下降来自同轮删除非 Codex queued-turn public 回流，不是
`thread/closed` contract 丢失。

范围矩阵只移动 exact `thread/closed`，最新计数为 `60 implemented / 119 planned / 34
product-scope-excluded`，产品范围完成度 `60 / 179 = 33.5%`。分类：v2 notification、idle unload owner、
RuntimeCore unload 与 broadcast transport 为 `current`；即时 unsubscribe/transport-close closed、
archive/delete closed 和生产 mock fallback 为 `dead / forbidden-to-restore`；无 `compat/deprecated`。
架构确认：confirmed；本刀补全既有 `App Server transport -> RuntimeCore -> Thread/Turn projection -> GUI`
生命周期，不改变 owner 或依赖方向，无需修改架构图。下一刀回到剩余 P0 history lineage exact methods。

### 2026-07-26 queued-turn public 回流删除

目标与写集：对照 Codex current App Server 注册表与 `TurnStartParams` / `Turn` exact shape，删除并行车道
重新加入的 `turn/queue/promote`、v2 `queueIfBusy`、`Turn.queue`、Renderer “稍后处理/优先执行”写平面及
PendingTurn 列表。RuntimeCore/session loop 内部 FIFO durable pending-work、mailbox、queued count/evidence
保持 `current`，不因 public surface 删除而恢复第二套 runtime。

完成结果：移除 v2 DTO/method/envelope/dispatch/handler、Rust/TypeScript typed client、Renderer gateway、
adapter、发送 preparation/submission 参数链、Inputbar UI、五语言 dead 文案和正向测试；重生成 schema 与
protocol types。`scripts/check-app-server-client-contract.mjs` 新增 public queued-turn 回流守卫，覆盖 v2
protocol、App Server dispatch、typed clients、Renderer gateway、schema bundle/manifest 与已删除独立
schema 文件。active turn 输入继续唯一走 exact `turn/steer`；idle 新回合唯一走 exact `turn/start`。

验证：`cargo check -p app-server --tests`、protocol v2 `37/37`、schema fixture `1/1`、TypeScript client
`85/85`、Renderer 定向 `120/120`、`npm run test:contracts`（`786 / 778 / 284`）与
`npm run governance:legacy-report`（`0 / 0 / 0`）通过。分类：`turn/start|steer|interrupt`、内部 durable
pending-work 与只读 count/evidence 为 `current`；public queue promote、queueIfBusy、Turn.queue、PendingTurn
GUI 为 `dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。架构确认：confirmed；本刀恢复既有
架构约束，没有改变 owner 或依赖方向，无需修改架构图。

下一刀：先修正多模型范围事实错误。当前 Lime `model/list` 仍是 v0 `{ providerId, tier } -> { models }`
DTO，却在范围矩阵中被误标为 Codex exact implemented；应直接迁到 v2
`{ cursor?, limit?, includeHidden? } -> { data, nextCursor }`，复用 runtime readiness，并按 Codex/Grok
产品语义过滤 hidden/non-selectable model，不保留双 DTO。随后回到 history lineage 的
`thread/searchOccurrences`。范围矩阵当前仍为 `60 / 179 = 33.5%`；在 exact `model/list` 与 Codex HEAD
注册表增量重新审计前不得提高完成度。

### 2026-07-26 Codex exact v2 `model/list` 与多 provider catalog 收口

目标与写集：直接删除 v0 `model/list` method、params/response、schema 与 catalog 注册，不保留 compat；
在 `app-server-protocol` v2、App Server LocalDataSource/RuntimeCore、typed client、Renderer gateway、
Electron 恢复投影、模型管理页、fixture/contract guard 和架构事实源内完成一次迁移。内部多 provider
目录参考 Grok/OpenCode 的 provider -> model 分层，但 public JSON-RPC 保持 Codex exact DTO，不扩展
provider 字段。

完成结果：v2 `ModelListParams` 为 `cursor / limit / includeHidden`，响应为 `{ data, nextCursor }`；
Codex `Model` 的 reasoning、modality、service tier 字段使用 typed DTO。LocalDataSource 以
`ProviderModelCatalog` 聚合 configured-provider readiness、provider scope、排序和 provider 内 model id
去重；Spawn Agent 直接消费内部 provider-scoped catalog，不反向依赖 public DTO。public `Model.id` 使用
可逆 opaque route `route:<base64url(providerId)>.<base64url(stableModelId)>`，`Model.model` 保留 provider
wire model id；Renderer 与 Electron 只通过共享 decoder 恢复 route。默认只返回 `List` visibility，
`includeHidden=true` 才包含 `Hide/None` 并标记 `hidden=true`；分页对齐 Codex 的空目录、`limit=0`、
末尾 cursor 与非法 cursor fail-closed 语义。Renderer 聚合全部分页、拒绝重复 cursor，并隔离 visible/
hidden cache；模型管理页和 Electron 恢复显式请求 hidden catalog。无事实源的 `isDefault` 不猜测，
缺失 reasoning effort 使用 Codex `none`，未进入 public DTO 的 capability 不由 Renderer 推断。

验证：`cargo check -p app-server --tests`；`cargo test -p app-server-protocol`（unit `77/77`、schema
fixture `1/1`）；公开 `model_list_jsonrpc` `2/2`；Runtime catalog/pagination `6/6`；LocalDataSource
provider catalog `7/7`；Renderer/Node、Electron 与 package client typecheck；package client `87/87`、
Renderer/Electron 定向 `71/71`；`npm run test:contracts`（client `284 checks`）；
`npm run governance:legacy-report`（`0` 零引用候选、`0` 分类漂移、`0` 边界违规）；三个修改脚本
`node --check` 均通过。最终 Gate B：`npm run smoke:agent-runtime-current-fixture` 完整通过且
`liveProviderUsed=false`，`npm run verify:gui-smoke` 通过；模型管理专用
`node scripts/model-registry-current-smoke.mjs` 观察到 Electron Host、`app_server_handle_json_lines`、
`model/list { includeHidden: true }` 与合法 `{ data, nextCursor }`，无 legacy/forbidden method、无 console
error。仓库 smart related runner 曾因把 `electron/` 目录当文件读取而报 `EISDIR`，已用直接 Vitest
完成等价定向验证，不归因于本实现。

分类：v2 exact DTO、内部 provider-scoped catalog、opaque route owner、typed gateway 和 Electron/GUI
消费链为 `current`；v0 `model/list` DTO/schema/注册、public provider/tier filter、`response.models`、
按 provider 名称猜 route 和 visible/hidden 共享缓存为 `dead / deleted / forbidden-to-restore`；无
`compat/deprecated`。架构影响：重大，替换 public JSON-RPC DTO/owner；架构图已更新
`internal/aiprompts/architecture.md` 6.3；责任开发者确认：`root, 2026-07-26`。范围矩阵仍为
`60 implemented / 119 planned / 34 product-scope-excluded`，产品范围完成度保持
`60 / 179 = 33.5%`，不因修正已计数 method 的 exact 语义而增加。下一刀回到
`thread/searchOccurrences`，并单独执行 Codex HEAD registry 增量复核。

### 2026-07-26 Codex exact v2 `thread/searchOccurrences`

目标与写集：复制 Codex current contract，在 `app-server-protocol` v2、`ThreadStore`、App Server
`ProjectionStore`/RuntimeCore/processor、Rust/TypeScript typed client、schema、产品范围矩阵与架构事实源
内建立单一 occurrence-search owner。Renderer 当前没有 Thread 内查找产品面，首轮不复用侧栏标题搜索，
也不新增没有分页、跳转和高亮闭环的 GUI 空壳。

完成结果：public wire 为
`{ threadId, searchTerm, cursor?, limit? } -> { data, nextCursor }`；默认 `50`、最大 `250`，`limit=0`
按 Codex 提升为 `1`。唯一读取链是
`App Server -> RuntimeCore -> ThreadStore::search_thread_occurrences -> ProjectionStore -> canonical_turns/canonical_items`，
支持 cold、archived 与已物化 fork history；不扫描旧 timeline DAO、deprecated projection 或 `item_json LIKE`。
搜索从 typed payload 提取 UserMessage Text 和每个 Turn canonical ordinal 最后的 `final/final_answer`
AgentMessage，assistant Markdown 先转纯文本，再做大小写不敏感 literal match。snippet range 使用 UTF-16 code
unit；opaque search cursor 绑定 Thread、原始 search term 与 occurrence 位置，返回的 inclusive `turnCursor`
可直接传给 `thread/turns/list`。空 term、非法 UUID、损坏/跨 term cursor、missing Thread 与 unsupported store
都通过结构化 `ThreadStoreErrorKind` fail closed。schema/generated types 与产品范围矩阵已同步，方法从
`planned` 移入 `implemented`。

验证：`cargo check -p app-server --tests`；`cargo test -p app-server-protocol`（unit `78/78`、schema
fixture `1/1`）；公开 `thread_search_occurrences_jsonrpc` `2/2`；App Server canonical-store 过滤测试
`77/77`；Rust typed client exact request `1/1`；TypeScript package client `88/88`；范围矩阵 `4/4`；
`npm run test:contracts`（`796` schema definitions / `788` protocol types / `284` client checks）；
`npm run governance:legacy-report`（`0` 零引用候选、`0` 分类漂移、`0` 边界违规）；scoped
rustfmt/Prettier 与 scoped `git diff --check` 通过。证据覆盖 protocol、domain integration、public App Server
JSON-RPC 与 typed clients；本轮没有 Renderer 产品面，因此未把 GUI smoke 或 Electron Gate B 冒充为本 method
证据，Desktop Host 继续只透明转发 `app_server_handle_json_lines`。

分类：v2 method/DTO/schema、ThreadStore contract、canonical SQLite search、RuntimeCore/processor 与 typed
clients 为 `current`；旧 timeline DAO、`item_json LIKE`、侧栏标题搜索复用和生产 mock fallback 为
`dead / forbidden-to-restore`，本轮无新增 `compat/deprecated`。架构影响：重大，新增 public JSON-RPC
method 与 canonical read owner；`internal/aiprompts/architecture.md` 6.2.1 已补数据流、正文选择、cursor、
UTF-16 与 cold/archived 规则；责任开发者确认：`root, 2026-07-26`。范围矩阵更新为
`61 implemented / 118 planned / 34 product-scope-excluded`，产品范围完成度为 `61 / 179 = 34.1%`。
下一刀单独执行 Codex HEAD method registry 增量审计，再按 P0 回到剩余 history lineage exact methods；
nested fork 独立回归、snippet ellipsis 和 active 未持久化 snapshot 排除可作为 store 精度补测，不阻塞本骨架。

### 2026-07-26 Codex HEAD method registry 增量审计

目标与结果：将产品范围事实源从 Codex `9fc715c0861c956c894a91890b78dc05b304ba29` 更新到
`4c43465133428898aa84f0bfc02c306ed65fb66a`。逐项复核 `protocol/common.rs` 四方向注册表及其生成
schema 变更后，确认唯一 method 增量是 client request
`externalAgentConfig/import/recordHistory`；server request、server notification 与 client notification
没有新增或删除。该方法用于由客户端记录在 App Server 外部完成的 Agent 配置导入历史，归入现有
`external-agent-config-planned` P4 owner；Lime 当前没有 exact config-import lifecycle，不用近似配置能力冒充，
也不为审计新增 compat。

事实源同步：fixture upstream revision、总数、方向计数、状态计数、identity hash、矩阵说明、gap register
和 boundary test 已同轮更新。最新盘点为 `214` 个方向化 identity：`130 clientRequest / 11
serverRequest / 72 serverNotification / 1 clientNotification`；分类为
`61 implemented / 119 planned / 34 product-scope-excluded`。产品范围分母因上游新增 planned method 从
`179` 增到 `180`，所以当前完成度从审计前的 `61 / 179 = 34.1%` 调整为
`61 / 180 = 33.9%`；这是上游范围扩张，不是已实现能力回退。

验证：范围矩阵 boundary test `4/4`、Prettier 与 `git diff --check` 通过。分类：更新后的 registry
fixture/guard 为 `current`；新增 method 为 `planned`；无 `compat/deprecated/dead` 变化。架构确认：不适用，
本轮只更新外部注册表 inventory 和产品裁决，没有新增 Lime protocol、owner 或依赖方向。下一刀按 P0
继续剩余 history lineage exact methods，P4 external config import 不抢占主链。

### 2026-07-26 Codex exact v2 `thread/search`

目标与写集：在 `app-server-protocol` v2、`thread-store`、App Server canonical store/RuntimeCore/processor、
Rust/TypeScript typed client、schema、产品范围矩阵与架构事实源内完成 exact content search；不复用
`thread/list.searchTerm` 或 Renderer 侧栏已加载标题过滤，不接没有 snippet 产品闭环的 GUI 空壳。

完成结果：public wire 为
`{ cursor?, limit?, sortKey?, sortDirection?, sourceKinds?, archived?, searchTerm } ->
{ data: [{ thread, snippet }], nextCursor, backwardsCursor }`。search term trim 后必须非空；默认 limit `25`、
范围 `1..100`，默认 `created_at desc`，默认/空 source kinds 为 `cli + vscode`，active 与 archived 严格二选一。
唯一 owner 是
`RequestProcessor -> RuntimeCore -> ThreadStore::search_threads -> ProjectionStore -> canonical_threads/canonical_items`；
正文从 typed UserMessage/AgentMessage 提取，大小写不敏感匹配，每个 Thread 只返回首个正文 snippet，name/preview
不参与。store-owned opaque cursor 绑定 term/archive/sort key/source kinds，支持同 sort key 下切换方向反向分页；
非法或跨查询 cursor fail closed。结果 Thread 复用 v2 projection，不新增第二套 session DTO。schema/generated
types 与两端 typed client 已同步，`thread/search` 从 `planned` 移入 `implemented`。

验证：`cargo check -p thread-store -p app-server-protocol -p app-server`；
`cargo test -p app-server-protocol`（unit `79/79`、schema fixture `1/1`）；canonical store search `2/2`；
公开 `thread_search_jsonrpc` `2/2`；Rust typed client exact request `1/1`；TypeScript package client `89/89`；
范围矩阵 `4/4`；`npm run test:contracts`（`799` schema definitions / `791` protocol types /
`284` client checks）；`npm run governance:legacy-report`（`0` 零引用候选、`0` 分类漂移、`0` 边界违规）。
证据覆盖 protocol、domain integration、public App Server JSON-RPC 与 typed clients；本轮未改 Renderer，因此
不把 GUI smoke 或 Electron Gate B 冒充 `thread/search` method evidence。

分类：v2 method/DTO/schema、ThreadStore contract、canonical content search、RuntimeCore/processor 与 typed clients
为 `current`；现有 `thread/list.searchTerm` 仍只是 list metadata filter，侧栏标题筛选仍是独立未迁产品入口，
均不得冒充 content search；无新增 `compat/deprecated`，无生产 mock fallback。`app-server-client/src/lib.rs`
已经远超 1000 行，本刀只按既有模式增加薄 typed delegation；退出条件是在下一次 client 结构治理时按 thread/model/
artifact domain 拆分 typed builders，禁止继续向该文件加入 handler 或领域逻辑。

架构影响：重大，新增 public JSON-RPC method 与 canonical cross-thread content read owner。
架构图已更新：`internal/aiprompts/architecture.md` 6.2.1 `thread/search` 数据流与边界。
责任开发者确认：root，2026-07-26。
确认内容：已核对目录归属、数据流、依赖方向、协议边界和验证门禁；Electron 仍只透明转发 JSONL，
provider/tool/model owner 均未改变。

范围矩阵更新为 `62 implemented / 118 planned / 34 product-scope-excluded`，产品范围完成度为
`62 / 180 = 34.4%`。本切片完成度 100%；下一刀仍按 P0 处理剩余 thread history/runtime lifecycle method，
Renderer snippet 搜索产品接入作为独立交互切片，不阻塞本 method boundary。

### 2026-07-26 Codex exact v2 `thread/backgroundTerminals/*`

目标与写集：复制 Codex current `thread/backgroundTerminals/{list,terminate,clean}` contract，在
`app-server-protocol` v2、App Server processor/RuntimeCore/ExecutionProcessServer、`tool-runtime`
supervisor、Rust/TypeScript typed clients、schema、范围矩阵与架构事实源内建立 thread-scoped 唯一
owner；不修改旧 `executionProcess/*` v0 wire，不从 provider metadata 反解 Thread identity，也不新增
compat 或生产 mock fallback。

完成结果：`CurrentTurnToolExecutor` 与 `thread/shellCommand` 显式下传 canonical Thread identity，
`ExecutionProcessServer` 维护 authoritative thread index 和单调数字 public process id。list 按 public id
稳定排序，只返回当前 Thread 的 running process；cursor anchor 消失后继续返回更大 id，默认返回全部，
`limit=0` 提升为 `1`。terminate 在 registry 内原子校验 `threadId + processId`，跨 Thread 返回
`terminated=false`；命中后立即从 list 隐藏并向真实 supervisor 发终止信号。clean 使用同一隐藏/终止
边界，外部控制同步清理 `unified_exec` session mapping；`write_stdin` 拒绝跨 Thread session 操作。
public JSON-RPC 测试真实启动 `sleep 30`，覆盖隔离、terminate、clean 与非法 cursor。v0 queued-turn
协议删除后的 Rust client/schema/generated-types 残留也已同轮删除，未恢复 handler 或包装层。

验证：Rust typed client exact request `1/1`；v2 exact protocol `36/36`；schema fixture `1/1`；
`tool-runtime` unified exec `9/9`；background pagination `2/2`；公共
`thread_background_terminals_jsonrpc` 真实进程 `1/1`；TypeScript package client `90/90`；范围矩阵
`4/4`；`npm run test:contracts`（`802` schema definitions / `794` protocol types / `284` client
checks）；`npm run governance:legacy-report`（`0` 零引用候选、`0` 分类漂移、`0` 边界违规）。证据覆盖
protocol、owner integration、public App Server JSON-RPC、真实 local process supervisor 与 typed clients；
本轮没有 Renderer/Electron 产品面改动，因此未把 GUI smoke 或 Gate B 冒充 method 证据。
scoped rustfmt、Prettier、矩阵 JSON 解析和全树 `git diff HEAD --check` 通过；全树
`cargo fmt --all -- --check` 只报告本切片外 `app-server/src/lib.rs` 与
`tests/model_list_jsonrpc.rs` 的既有 model-list 排版漂移，本轮按窄写集原则未改写。

分类：v2 method/DTO/schema、thread index、RuntimeCore/processor、supervisor ownership 与 typed clients 为
`current`；queued-turn v0 client/schema 残留为 `dead / deleted / forbidden-to-restore`；旧全局
`executionProcess/*` 不计 Codex parity，仍由其现有消费者约束，本轮未新增 `compat/deprecated`。架构影响：
重大，新增 public JSON-RPC method 与 thread-scoped process control boundary；架构图已更新
`internal/aiprompts/architecture.md` 6.2.1。责任开发者确认：`root, 2026-07-26`；已核对 owner、数据流、
锁边界、协议同步与验证门禁，Electron 仍只透明转发 JSONL。

范围矩阵更新为 `65 implemented / 115 planned / 34 product-scope-excluded`，产品范围完成度为
`65 / 180 = 36.1%`。本切片完成度 100%；剩余直接 P0 为
`thread/approveGuardianDeniedAction`、`thread/increment_elicitation`、
`thread/decrement_elicitation` 与 `thread/inject_items`。

### 2026-07-26 Codex exact v2 Thread elicitation accounting skeleton

目标与写集：复制 Codex current `thread/increment_elicitation` 与
`thread/decrement_elicitation` method/DTO/refcount semantics，在 `app-server-protocol` v2、App Server
processor/RuntimeCore、Rust/TypeScript typed clients、schema、范围矩阵与架构事实源内建立唯一
Thread-local owner；不复用 MCP connection-local pause state，不持久化 live registration，不新增 v0/compat
wire 或生产 mock。

完成结果：两个 method 使用 Thread serialization 与 exclusive access，只接受 loaded canonical Thread。
RuntimeCore 以 checked `i64` 维护 process-local refcount；`0 -> 1 -> N` 返回 `paused=true`，归零返回
`paused=false` 并删除 entry。increment overflow、decrement underflow、非法/unknown/cold Thread 全部 fail
closed；archive、delete、idle unload、agent-control child cleanup 与 import compensating cleanup 都移除 stale
entry。public JSON-RPC 覆盖两次 increment/decrement、underflow、unknown Thread 与 archive 后 cold rejection；
RuntimeCore owner 测试直接证明 overflow 与 archive entry cleanup。

本切片没有把 `paused` registry 接入 `agent-runtime` provider first-visible-output/provider-step active-time
budget。现有 `lime-mcp::ElicitationPauseState` 只负责单个 MCP connection 内 active-time accounting，不能成为
Thread owner。统一 pause consumer 与 upstream experimental capability gate 仍是 lifecycle/connection blocker；
因此本切片 100% 完成的是 method control-plane skeleton，不声称完整 timeout parity。

验证：`app-server-protocol` unit `81/81`；schema fixture `1/1`；RuntimeCore elicitation unit `3/3`；
public `thread_elicitation_jsonrpc` `1/1`；Rust typed client exact request `1/1`；TypeScript client + scope
matrix `78/78`；`npm run test:contracts`（`806` schema definitions / `798` protocol types / `284` client
checks）；`npm run governance:legacy-report`（`0` 零引用候选、`0` 分类漂移、`0` 边界违规）。首次
contracts 运行发现守卫仍要求已退役 `mockUpdateAgentRuntimeSession`；同轮把它替换为 current
`mockUpdateAgentRuntimeThreadSettings` 负向断言，没有恢复 `agentSession/update` 测试 mock。

分类：两个 v2 method/DTO/schema、RuntimeCore registry、processor 与 typed clients 为 `current`；旧
`agentSession/update` 守卫字符串为 `dead / deleted / forbidden-to-restore`；未新增 `compat/deprecated`。
架构影响：重大，新增 public JSON-RPC method 与 loaded Thread volatile control boundary；架构图已更新
`internal/aiprompts/architecture.md` 6.2.1。责任开发者确认：`root, 2026-07-26`；已核对 owner、锁与
cleanup 边界、协议同步、非持久化语义和验证门禁，Electron 仍只透明转发 JSONL。

范围矩阵更新为 `67 implemented / 113 planned / 34 product-scope-excluded`，产品范围完成度为
`67 / 180 = 37.2%`。剩余直接 P0 为 `thread/approveGuardianDeniedAction` 与
`thread/inject_items`；provider active-time pause consumer 作为 elicitation lifecycle blocker 单独保留。

### 2026-07-26 Codex exact v2 `thread/approveGuardianDeniedAction` skeleton

目标与写集：复制 Codex current opaque-event wire、typed Guardian action 校验与 exact-action developer
continuation，在 `app-server-protocol` v2、App Server processor/RuntimeCore、`agent-runtime` session input、
canonical provider history、Rust/TypeScript typed clients、schema、范围矩阵与架构事实源内建立唯一 owner；
不复用旧 `agentSession/action/respond`，不新增 v0/compat wire、用户可见 Item 或生产 mock fallback。

完成结果：public wire 保持 `{ threadId, event: JsonValue } -> {}`，RuntimeCore 只接受 loaded canonical
Thread，并将 event fail-closed 解析为 typed status/action union；command/execve/apply-patch 的本地路径必须为
绝对路径，缺字段与 unknown variant 均拒绝。只有 `status=denied` 生成 Codex exact-action developer marker；
其他合法 status 无副作用返回空对象。active regular Turn 通过新增 `RuntimeSessionInput::Developer` 在下一
sampling boundary 消费，idle、finishing race 与 restart 通过 durable
`guardian.denied_action.approved` provider-only event 恢复；同一 canonical provider-history owner 将其 lowering
为 Developer message，且不投影为用户可见 Thread Item。

本切片没有实现 Guardian assessment/reviewer producer、review lifecycle 或用户产品面，因此 100% 完成的是
manual continuation method skeleton，不声称完整 Guardian 产品闭环。`thread/inject_items` 仍因缺完整
Responses API `ResponseItem` union 与 canonical raw rollout owner 保持 `planned`；不得写入
`ThreadItemPayload::Extension` 制造第二事实源。

验证：`app-server-protocol` unit `82/82`、schema fixture `1/1`；RuntimeCore Guardian unit `2/2`；公开
`thread_guardian_jsonrpc` `1/1`；`agent-runtime` developer-role lowering `1/1`；Rust typed client exact request
`1/1`；TypeScript package client `92/92`；范围矩阵 `4/4`；`npm run test:contracts`（`808` schema
definitions / `800` protocol types / `284` client checks）；`npm run governance:legacy-report`（`0` 零引用候选、
`0` 分类漂移、`0` 边界违规）。最终 scoped format 与 `git diff HEAD --check` 见本节后的收尾验证记录。

分类：v2 method/DTO/schema、RuntimeCore typed validation、session developer input、durable provider-only event、
provider-history lowering 与 typed clients 为 `current`；旧 `agentSession/action/respond` 复用、用户可见伪 Item、
宽泛 permission cache 和生产 mock fallback 为 `dead / forbidden-to-restore`；无 `compat/deprecated`。架构影响：
重大，新增 public JSON-RPC method 与 provider-only continuation boundary；架构图已更新
`internal/aiprompts/architecture.md` 6.2.1。责任开发者确认：`root, 2026-07-26`；已核对 owner、数据流、
协议边界、active/idle race、历史恢复与验证门禁，Electron 仍只透明转发 JSONL。

范围矩阵更新为 `68 implemented / 112 planned / 34 product-scope-excluded`，产品范围完成度为
`68 / 180 = 37.8%`。下一刀是 `thread/inject_items` 的 canonical `ResponseItem`/raw rollout owner；在 owner
和 full union 明确前保持 blocker，不用近似 Item 或 extension payload 冒充 Codex parity。

### 2026-07-26 Codex exact v2 `thread/inject_items` skeleton

目标与写集：复制 Codex current opaque-item wire 与 `ResponseItem` validation/lowering semantics，在
`agent-protocol`、`app-server-protocol` v2、App Server processor/RuntimeCore、`agent-runtime` session input、
canonical provider history、`model-provider` lowering、Rust/TypeScript typed clients、schema、范围矩阵与
架构事实源内建立唯一 owner；不使用 `ThreadItemPayload::Extension`，不新增 v0/compat wire、用户可见伪 Item
或生产 mock fallback。

完成结果：public wire 保持 `{ threadId, items: JsonValue[] } -> {}`，RuntimeCore 在副作用前按 Codex current
`ResponseItem` union fail-closed 校验并拒绝远程图片 URL。cold Thread 先从 canonical store hydrate/resume，
archived、unknown、空数组与 malformed item 均拒绝。每个 item 写入 durable
`response_item.injected` provider-only event；active regular Turn 同时经 session actor 投递
`RuntimeSessionInput::RawResponseItem`，idle、finishing race 与 restart 从同一 durable provider-history owner
恢复。Responses provider 原样 lowering，包括 validation union 未消费的 provider 扩展字段；Chat Completions、
Anthropic 与其他非 Responses route 在发网前 fail closed。raw item 不生成用户可见 Thread Item。

本切片没有关闭 P0-03 的全局 canonical history/rollout、rollback/fork/replay、损坏尾部与未知记录一致性；
100% 完成的是 `thread/inject_items` method boundary skeleton，不声称整个 history parity 已完成。

验证：`agent-runtime` raw input lowering `2/2`；RuntimeCore inject owner `4/4`；public
`thread_inject_items_jsonrpc` restart/archive `2/2`；`app-server-protocol` unit `83/83` 与 schema fixture
`1/1`；`model-provider` raw lowering `2/2`；Rust typed client exact request `1/1`；TypeScript package client
`93/93`；`npm run test:contracts`（`810` schema definitions / `802` protocol types / `284` client checks，
含 `governance:scripts`）；`npm run governance:legacy-report`（`0` 零引用候选、`0` 分类漂移、`0` 边界违规）；
scoped Prettier、矩阵 JSON 解析与 `git diff HEAD --check` 通过；
`npm run smoke:agent-runtime-current-fixture` 通过（含 current Electron fixture，`liveProviderUsed=false`）。
本轮没有 `thread/inject_items` 专用 Renderer/Electron 产品面，因此不把该聚合 Gate B 当成 method 专项 GUI
evidence。

分类：v2 method/DTO/schema、`ResponseItem` validation、RuntimeCore durable event、session raw input、
provider-history projection、Responses lowering 与 typed clients 为 `current`；extension 伪 Item、非 Responses
近似转换、v0/compat wire 和生产 mock fallback 为 `dead / forbidden-to-restore`；无 `compat/deprecated`。
架构影响：重大，新增 public JSON-RPC method 与 provider-only raw history boundary；架构图已更新
`internal/aiprompts/architecture.md` 6.2.1。责任开发者确认：`root, 2026-07-26`；已核对 owner、数据流、
协议边界、active/cold/archive/race 语义、provider lowering 和验证门禁，Electron 仍只透明转发 JSONL。

范围矩阵更新为 `69 implemented / 111 planned / 34 product-scope-excluded`，产品范围完成度为
`69 / 180 = 38.3%`。下一刀按用户要求回到 Codex + Grok/OpenCode 多模型控制平面，优先补
`model/rerouted`、`model/verification`、provider readiness、retry/circuit breaker；P0-03 更大的 canonical
history/rollout parity 继续保留，不能被本 method boundary 关闭。

### 2026-07-26 provider breaker credential-scope 隔离

目标与窄写集：补正 `model-provider` shared route health 的身份边界；只修改
`model-provider` health key、crate manifest、全局架构事实源与本计划。breaker key 固定为
`provider/model/base-url/protocol/credential-scope`：持久化凭证使用 UUID，direct runtime credential
使用 SHA-256 指纹，绝不保存 raw key。保持 `CurrentProviderHealthRegistry` 为唯一 owner，HTTP client、
WebSocket 与 HTTP fallback 仍为 session-local；不新增 App Server method、renderer state、compat 或 mock。

退出条件：同 route + 同一 direct key 复用 breaker；不同 stored UUID 与不同 direct key 必须互相隔离；
crate 定向测试、`test:contracts`、治理扫描、format/diff 全部通过。此 transport 诊断不得伪造 Codex
`model/rerouted` 或 `model/verification`，两者继续保持 `planned`，直到存在对应 cyber safety runtime
producer 或被明确列为产品范围排除。

完成结果与分类：health key 已加入 credential scope；同一 direct key 稳定复用 SHA-256 指纹 entry，
不同 stored UUID、stored/direct 与不同 direct key 均隔离。该实现、manifest/lock 与架构规则为
`current`；旧的无 credential scope route key 为 `dead / replaced`；无 `compat/deprecated`。没有新增
method 或产品面，范围矩阵不变，当前产品范围仍为 `69 / 180 = 38.3%`。

验证：独立 target 下 health 定向测试 `10/10`；`npm run test:rust:related --
lime-rs/crates/model-provider` 热缓存复跑退出码 `0`，覆盖 `model-provider` 与 12 个反向依赖 crate，
其中 `model-provider` 为 `176/176`。首次 related 运行在 App Server 既有 external backend JSONL 用例
出现一次 `output line timeout`（`1536/1537`），该用例单独复跑 `1/1`，随后完整 related 复跑通过。
`npm run test:contracts`、`npm run governance:legacy-report`（`0` 零引用候选 / `0` 分类漂移 /
`0` 边界违规）、`cargo fmt --all -- --check` 与 `git diff HEAD --check` 通过。未运行 GUI/Gate B：本刀
没有 Renderer、Electron、App Server protocol 或用户可见产品面变更。架构确认：`root, 2026-07-26`；
owner 和依赖方向未改变，只补正既有 breaker 身份边界。

### 2026-07-26 provider breaker observer

目标与窄写集：参考 Grok `xai-circuit-breaker` 的 observer，但保持 Lime current owner 不变。在
`model-provider::current_client::health` 为 shared breaker 增加 transition、half-open probe admission、
request rejection 与 failure 的结构化 tracing；只修改该 owner、全局架构事实源与本计划。日志只能使用
provider/model/protocol、credential kind 和 SHA-256 route hash，禁止输出 base URL、credential UUID、API key、
prompt 或 Thread/Turn payload。

退出条件：observer 回调必须发生在 breaker mutex 释放后；Closed/Open/HalfOpen transition、probe 接纳/拒绝
和 rejection retry-after 都有稳定字段；同一测试证明 telemetry 无 endpoint/credential 泄漏。该诊断只服务
provider retry/health evidence，不产生 Codex `model/rerouted` 或 `model/verification`，也不新增 App Server
method、GUI state、compat 或 mock。

完成结果与分类：shared breaker 现在通过 `provider_health` target 发出 Closed/Open/HalfOpen transition、
half-open probe admission/rejection、request rejection `retry_after_ms` 与 failure outcome；observer 回调均在
breaker mutex 释放后执行。route telemetry 只保留 provider/model/protocol、credential kind 和不可逆 SHA-256
route hash；不记录 endpoint、credential UUID、API key 或请求正文。已有 half-open probe 时只返回最多 `50ms`
短退避，不重复宣告完整 open cooldown。实现与架构约束为 `current`；普通 transport fallback/retry 伪造
`model/rerouted`、`model/verification` 的路径为 `dead / forbidden-to-restore`；无 `compat/deprecated`。范围矩阵
不变，当前产品范围仍为 `69 / 180 = 38.3%`。

验证：独立 target 下 health 定向测试 `13/13`；`cargo test -p model-provider` 为 `179/179`；
`CARGO_TARGET_DIR=/tmp/lime-model-provider-health npm run test:rust:related --
lime-rs/crates/model-provider/src/current_client/health.rs` 覆盖 `model-provider` 与 12 个反向依赖 crate，退出码
`0`。`npm run test:contracts` 通过（`810` schema definitions / `802` protocol types / `284` client checks）；
`npm run governance:legacy-report` 通过（`0` 零引用候选 / `0` 分类漂移 / `0` 边界违规）；workspace rustfmt、
scoped Prettier 与 `git diff HEAD --check` 均通过。未运行 GUI/Gate B：本刀没有 Renderer、Electron、App Server
protocol 或用户可见产品面变更。架构确认：`root, 2026-07-26`；observer 仍在既有 `model-provider` network
owner 内，未改变依赖方向。下一刀继续 provider readiness/retry 的结构化 evidence，不为普通 provider 故障
生产 cyber safety notification。

### 2026-07-26 provider transport retry observer

目标与窄写集：把 `CurrentProviderClient` 内部真实 HTTP/WebSocket retry 变成可核验的结构化 evidence，复用
既有 shared health route identity；只修改 `model-provider::current_client` 的 health telemetry、transport
retry helper 和三个真实 retry 点，不新增 public protocol、App Server method、GUI state、compat 或 mock。
`health.rs` 已拆出 `current_client/health/telemetry.rs`，从 `1033` 行降到 `941` 行；telemetry owner 同时承接
`provider_health` 与 `provider_retry` 两个 tracing target。retry event 只记录 transport、稳定 reason、
failed/next/max attempt、delay、delay source 和可选 status code，继续禁止 endpoint、credential、请求正文
和错误正文。Codex request policy 保持只重试 5xx；429、认证和内容拒绝不被普通 transport retry 放大。

完成结果与分类：HTTP request transport error、HTTP 5xx 和 WebSocket connect retry 都在 sleep 前通过
`TransportRetryEvent` 发出 observer 回调，回调不持有 breaker mutex；`Retry-After` 与指数退避来源进入稳定
`delay_source`，retry route 仍使用不可逆 route hash。`health.rs` 大小守卫已满足；`current_client.rs` 只保留
薄接线，下一次 transport 结构治理必须拆分 `send_stream_request` / WebSocket connect，禁止继续向该 `2560`
行文件增加领域逻辑。实现与拆分为 `current`；未产生的普通 fallback/safety notification 为
`dead / forbidden-to-restore`；无 `compat/deprecated`。范围矩阵不变，仍为 `69 / 180 = 38.3%`。

验证：独立 target 下 `cargo test -p model-provider` 为 `180/180`，覆盖 health observer、retry delay source
和现有 HTTP/WebSocket retry 场景；`npm run test:rust:related --` 三个 current client 路径覆盖
`model-provider` 与 12 个反向依赖 crate，退出码 `0`。`npm run test:contracts` 通过（`810` schema definitions /
`802` protocol types / `284` client checks）；`npm run governance:legacy-report` 通过（`0` 零引用候选 / `0`
分类漂移 / `0` 边界违规）；workspace rustfmt、scoped Prettier 与 `git diff HEAD --check` 均通过。未运行
GUI/Gate B：本刀没有 Renderer、Electron、App Server protocol 或用户可见产品面。架构确认：`root,
2026-07-26`；只补既有 `model-provider` 诊断边界，不改变 provider/readiness owner 或依赖方向。下一刀回到
provider readiness 的 live multi-model switch evidence，普通 transport failure 仍不得生产 Codex cyber
safety 通知。

### 2026-07-26 resolved no-auth provider transport

目标与窄写集：修复已解析的 `AuthKind::NoAuth` route 在 App Server/Agent config 投影后被
`CurrentProviderClient` 错判为“缺 API key”的 readiness 缺口。写集限定为 `model-provider` runtime config
与 HTTP/WebSocket request boundary、`agent` provider configuration、App Server route contract、架构事实源和本
计划；不实现尚无完整 adapter 的 `OllamaChat`，不新增协议、compat、mock 或按 provider 名称猜认证。

完成结果与分类：`ResolvedModelRoute.auth.kind` 现在精确投影为 `RuntimeProviderAuth`：`NoAuth` 创建 direct
config 并清除误带 key，HTTP 与 Responses WebSocket 都省略认证头；`ApiKey` 保持发网前 fail-closed；
`OemManaged` 没有 current `model-provider` adapter，因此 admission 明确拒绝。direct config 不再伪造
`manual:<session>` credential UUID；API-key direct route 可按 key 指纹隔离 breaker，`NoAuth` 以稳定
`no-auth` health scope 共享。当前 route/auth projection、provider request boundary、health identity 为
`current`；由空 key、session synthetic UUID 或 provider 名称隐式猜认证的行为为 `dead / replaced`；无
`compat/deprecated`。范围矩阵不变，仍为 `69 / 180 = 38.3%`。

退出条件与验证：本地 HTTP 抓包证明 `NoAuth` 不发送 `Authorization` 或 `x-api-key`；Responses WebSocket
handshake 不发送 `Authorization`；API-key route 缺 key 在网络前拒绝；route contract、Agent projection 与
NoAuth breaker identity 皆有回归。`cargo test -p model-provider` 为 `184/184`；
`npm run test:rust:related --` 五个受影响路径覆盖 `model-provider` 与 12 个反向依赖 crate，退出码 `0`；
`npm run test:contracts` 通过（`810` schema definitions / `802` protocol types / `284` client checks）；
`npm run governance:legacy-report` 通过（`0` 零引用候选 / `0` 分类漂移 / `0` 边界违规）；workspace rustfmt、
scoped Prettier 与 `git diff HEAD --check` 均通过。`lime-services` 的 4 个 live/本地监听测试维持既有
`ignored`，不是失败。未运行 GUI/Gate B：本刀没有 Renderer、Electron、App Server public protocol 或
用户可见产品面变更；未运行 live provider，因为需要显式凭证授权。架构确认：`root, 2026-07-26`；依赖方向
仍为 App Server -> agent -> model-provider，Electron 未参与。

### 2026-07-26 provider readiness/admission consistency

目标与窄写集：消除多模型控制面的假 ready。此前 configured-provider readiness 除 Fal 外只检查
enabled/key，Gemini、Vertex、Bedrock、Ollama 会进入 selectable catalog，并会在 profile fallback 中提前
截断后续可执行 provider；Azure OpenAI 还会因 OpenAI-shaped body 被错误当作普通 Bearer Chat adapter。
写集限定为 `runtime-core` provider protocol inference、`model-provider` current adapter availability、App
Server readiness/catalog、Agent admission 复用、provider connection/chat probe、架构事实源和本计划。

完成结果与分类：`RuntimeProviderProtocol` 现在集中声明 route protocol 与 effective provider type 的 current
adapter availability；App Server configured readiness、direct readiness、Agent admission 与模型管理 probe
共用该事实。Gemini、Azure OpenAI、Vertex、Bedrock、Ollama、Fal 在完整 adapter 落地前统一
`unsupported_protocol`，不进入 `model/list` runtime-ready catalog 或 capability union，也不触网伪装成
OpenAI Chat。Store 未命中时删除 provider-name builtin ready；显式 direct config 只评估绑定 selection 一次，
不再把同一 endpoint/credential 复用于 profile fallback。stored provider 的 unsupported coding slot 会被记录
为 blocked attempt，并继续选择后续 ready slot。adapter/readiness/catalog/probe 为 `current`；builtin-name
fail-open、key-only readiness、keyless Ollama selectable catalog、Azure Bearer Chat 与 direct-config fallback
复用为 `dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。范围矩阵不变，仍为
`69 / 180 = 38.3%`。

验证：adapter availability `2/2`、App Server routing/readiness `10/10`、model catalog/capability union
`11/11`、provider connection/chat preflight `1/1` 通过，且 probe 回归使用不可连接地址证明未发生网络访问；
`cargo test -p model-provider` 为 `186/186`。`npm run test:rust:related --` 九个受影响路径覆盖
`model-provider`、`runtime-core`、App Server、Agent 与反向依赖共 `14` 个 crate，退出码 `0`；
`lime-services` 为 `209` passed / `4` ignored，ignored 项仍是既有 live/本地监听测试。
`npm run test:contracts` 通过（`810` schema definitions / `802` protocol types / `284` client checks）；
`npm run governance:legacy-report` 通过（`0` 零引用候选 / `0` 分类漂移 / `0` 边界违规）；workspace rustfmt、
scoped Prettier 与 `git diff HEAD --check` 均通过。未运行 live provider：需要显式凭证授权；未运行 GUI/Gate B：
本刀没有 Renderer、Electron、App Server public protocol 或用户可见产品面变更。架构影响：重大，收紧多模型
control plane 的 ready 定义并删除 builtin fail-open；架构图文字约束已更新，责任开发者确认：
`root, 2026-07-26`。本 provider readiness/admission P0 切片完成；Codex v1 总范围仍为
`69 / 180 = 38.3%`，不把局部完成度计入 method 覆盖率。

### 2026-07-26 current configured provider capability 语义校正

目标与窄写集：按 Codex `modelProvider/capabilities/read` 的 current contract，删除 Lime 对所有 ready
provider 做 OR union 的平行语义。method 继续保持空参数，每次调用读取 `LIME_CONFIG_PATH` 指向的最新
`config.yaml`，使用当前产品实际写入的顶层 `default_provider` 精确查 Store；只评估该 provider 的 runtime
readiness 与 capability snapshot，不回退第一个 ready provider、不按名称猜测，也不修改 protocol、typed
client、Electron 或 Renderer。

完成结果与分类：当前 provider ready 且存在 current adapter 时返回其三项 capability；Store 未命中、空 ID、
disabled、缺 key、非 chat 或 unsupported adapter 均返回全 false。精确 current-provider read、共享 readiness
与 `model-provider` capability mapping 为 `current`；跨 provider OR union、首个 ready fallback 和 provider
名称猜测为 `dead / replaced / forbidden-to-restore`；无新增 compat。Core 同时保留顶层
`default_provider` 与 `routing.default_provider` 是独立配置治理缺口：本刀以 Electron/AppConfig 实际读写的
顶层字段为产品事实，不读取已脱离产品写链的 routing 副本；下一刀应直接收敛 Config schema 与消费者，
不再维持双字段。该缺口已由下节同轮关闭。

验证：App Server current-provider selection unit `2/2`、公开 v2 JSON-RPC 空参数合同 `1/1` 在后续 hosted
capability fail-closed 校正前通过；最终 capability 值与复验状态以后文“provider capability 与 adapter
availability 同源”为准。
`npm run test:rust:related --` 两个受影响路径覆盖 `agent-runtime`、App Server、`lime-agent`、`lime-cli`、
`lime-mcp`、`lime-media-runtime`、`lime-processor`、`lime-scheduler`、`lime-server`、`lime-services`、
`lime-skills`、`model-provider` 与 `tool-runtime` 共 13 个 crate，退出码 `0`，其中 `model-provider`
`186/186`、`lime-services` `209 passed / 4 ignored`（既有 live/本地监听用例）。`npm run test:contracts`
通过（`810` schema definitions / `802` protocol types / `284` client checks）；`npm run governance:legacy-report`
通过（`0` 零引用候选 / `0` 分类漂移 / `0` 边界违规）；本写集 scoped rustfmt、scoped Prettier 与
`git diff HEAD --check` 通过。后续接管已关闭进程遗留的 world-state 骨架后，全树格式与扩大 Rust
验证结果见本节后的最终收尾记录。未运行 GUI/Gate B 或 live provider：本刀没有 Renderer、Electron、
公开协议或用户可见产品面变更，且 live provider 需要显式凭证授权。范围矩阵 method 数量不变，仍为
`69 / 180 = 38.3%`。架构确认：`root, 2026-07-26`；本刀修正既有 App Server -> model-provider 读取语义，
没有新增 owner、公开 boundary 或依赖方向。

### 2026-07-26 default provider 配置单一事实源

目标与写集：紧接 current-provider capability 校正，删除 Core Config 中顶层 `default_provider` 与
`routing.default_provider` 双事实源。产品写链已由 Electron/AppConfig 固定写顶层字段，因此本刀直接修改
Core Config schema/YAML merge、`lime-config` observer、`lime-server` 初始化/热更新消费者及测试；不保留
双读、字段优先级或旧 YAML fallback。

完成结果与分类：`Config.default_provider` 现在是唯一 `current` 产品字段；`RoutingConfig` 只包含
`model_aliases`，并用 `deny_unknown_fields` 明确拒绝旧 `routing.default_provider`。Config merge、import、
observer、Server router 初始化/热更新与 App Server capability read 统一消费顶层字段。旧嵌套字段、双写与
优先级合并为 `dead / deleted / rejected / forbidden-to-restore`；无 `compat/deprecated`。该 schema 收敛已
同步架构事实源，责任开发者确认：`root, 2026-07-26`。

验证：`cargo test -p lime-core` 为 `691/691`，doc tests `2 passed / 8 ignored`；`cargo test -p lime-config`
为 `11/11`；`cargo test -p lime-server --lib` 为 `117/117`。接管并补齐同期 world-state 骨架后，使用隔离的
`LIME_CONFIG_PATH` 重跑五个受影响路径的 `npm run test:rust:related --`，覆盖 17 个直接与反向依赖 crate，
退出码 `0`；其中 App Server `1543/1543`、Core `691/691`、config `11/11`、server `117/117`。首次未隔离
复验的 App Server 为 `1532/1542`，10 个失败均由本机真实配置仍含已拒绝的
`routing.default_provider`、额外产生预期 `configWarning` 所致；最终门禁不修改用户配置，也不为该旧字段
恢复兼容。`npm run test:contracts` 通过（`810` schema definitions / `802` protocol types / `284` client checks）；
`npm run governance:legacy-report` 通过（`0` 零引用候选 / `0` 分类漂移 / `0` 边界违规）；本写集 scoped
rustfmt、scoped Prettier 与 `git diff HEAD --check` 通过。未运行 GUI/Gate B 或 live provider：本刀没有
Renderer、Electron IPC、公开协议或用户可见产品面变更，且 live provider 需要显式凭证授权。范围矩阵
method 数量不变，仍为 `69 / 180 = 38.3%`。

### 2026-07-26 provider capability 与 adapter availability 同源

目标与写集：继续收敛多模型控制面中 provider-type 识别的重复事实源，只修改
`model-provider/src/provider_capabilities.rs`、架构事实源和本计划。此前 readiness 使用
`RuntimeProviderProtocol::from_provider_type`，capability mapping 另维护一份字符串 match；两者对
`openai_responses`、`responses`、`anthropic_compatible` 等规范别名会产生“ready 但 capability 缺失”的漂移。

完成结果与分类：`ProviderCapabilities::from_provider_type` 现在直接委托 current adapter availability；当前
OpenAI Chat/Responses、Codex、Anthropic Messages adapter 及其规范别名返回明确的全 false snapshot，因为
Lime 尚未完成 Codex namespace/hosted image/hosted web 的 request + reducer 闭环。未实现的 Gemini、Azure
OpenAI、Vertex、Bedrock、Ollama、Fal 与未知类型统一返回 `None`。共享 adapter availability、fail-closed
hosted capability 与 capability projection 为 `current`；重复 provider alias 白名单、普通 function tool
冒充 hosted capability、未实现 adapter 的乐观 capability 和 ready/capability 漂移为
`dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。

验证：provider capability 定向矩阵 `3/3`，`cargo test -p model-provider` 为 `186/186`。接管并补齐同期
world-state 骨架后，使用隔离的 `LIME_CONFIG_PATH` 重跑
`npm run test:rust:related -- lime-rs/crates/model-provider/src/provider_capabilities.rs`，覆盖 13 个 crate，
退出码 `0`；其中 Agent Runtime `174/174`、App Server `1543/1543`、model-provider `186/186`，
`lime-services` 为 `209 passed / 4 ignored`，ignored 项仍是既有 live/本地监听测试。
`npm run test:contracts` 通过（`810` schema definitions / `802` protocol types / `284` client checks）；
`npm run governance:legacy-report` 通过（`0` 零引用候选 / `0` 分类漂移 / `0` 边界违规）；scoped rustfmt、
scoped Prettier 与 `git diff HEAD --check` 通过。未运行 GUI/Gate B 或 live provider：本刀没有 Renderer、
Electron IPC、公开协议或用户可见产品面变更，且 live provider 需要显式凭证授权。架构影响：不改变 owner
或依赖方向，只删除同一 `model-provider` owner 内的重复判断并收紧 capability gate。范围矩阵 method 数量
不变，仍为 `69 / 180 = 38.3%`。

### 2026-07-26 world-state 未完成骨架接管与收尾

目标与写集：其他进程已关闭后，接管其留在 `agent-protocol::world_state` 与 App Server turn context 的未完成
骨架，只关闭 `turn_context_from_request` 引用不存在 helper 的编译 blocker。写集限定为 typed world-state
投影、既有 workspace scope 回归和本计划；这不是新的 Codex method，不修改 multi-model readiness、public
protocol、Renderer 或 Electron。

完成结果与分类：App Server 从当前 typed request 投影 environment（cwd/project root/workspace/thread/turn、
provider/model/reasoning）、permissions（approval/sandbox/web search）和 collaboration mode，并写入
`AgentTurnContext.metadata["world_state"]`。当前请求没有事实源的 multi-agent mode 与 instruction sections
保持缺失，不猜测、不复制 prompt 文本。typed world-state 与 App Server 投影为 `current skeleton`；provider
request 尚未消费该 snapshot，因此 `runtime-environment-context` 只能从 `missing` 升为 `partial`，不能标记
covered。缺失 helper 的半成品引用为 `dead / completed`；无 `compat/deprecated`。该接管不计入 method 覆盖率，总范围仍为
`69 / 180 = 38.3%`。

验证：新增边界回归直接断言 environment/permissions/collaboration 及未伪造字段，定向测试 `1/1`；
`agent-protocol` 为 `37/37`；隔离配置后的 App Server 全量为 `1543/1543`。上述 default-provider 五路径
related 与 capability 专项 related 均退出码 `0`；workspace `cargo fmt --all -- --check`、执行计划 Prettier
与全树 `git diff HEAD --check` 均通过。架构影响：重大；本刀建立唯一
`agent-protocol world-state` DTO owner 与 `App Server -> AgentTurnContext.metadata` 投影边界，并同步全局
架构图。provider request consumer 与其余 typed producer 仍是下一阶段退出条件。责任开发者确认：
`root, 2026-07-26`。

### 2026-07-26 world-state provider-visible consumption

目标与写集：继续完成上一刀的第一条 consumer 链，只修改 `agent-runtime::provider_turn`、provider request
capture 回归、typed world-state provenance、架构和状态文档。App Server snapshot 存在时只消费
`AgentTurnContext.metadata["world_state"]`；没有 snapshot 的非 App Server 调用者也必须经
`RuntimeWorldState::from_cwd`，不保留手写 cwd XML 平行路径。

完成结果与分类：Agent Runtime 现在反序列化 typed snapshot，并通过 `RuntimeWorldState` 的 XML-safe
renderer 在当前 user input 前注入一次 provider-visible contextual user message；environment、permissions
与 collaboration 均进入真实 provider request，缺失的 multi-agent/instruction sections 不会被伪造。损坏的
metadata 明确 fail closed，不用 cwd 隐藏 contract drift；cwd-only state 不再冒充 App Server provenance。
typed DTO、App Server producer、Agent Runtime consumer 与 provider request capture 为 `current`；旧的
runtime cwd XML 拼接与重复 escape owner 为 `dead / deleted / forbidden-to-restore`；无
`compat/deprecated`。`runtime-environment-context` 仍为 `partial`，因为 AGENTS/apps/plugins/environment
instructions、realtime、multi-agent typed producer及 Codex durable full/patch history 尚未完成。method 范围
不变，总进度仍为 `69 / 180 = 38.3%`。

验证：provider request 定向回归 `7/7`，坏 metadata fail-closed `1/1`，`agent-protocol` world-state
`3/3`。使用隔离 `LIME_CONFIG_PATH` 的 `npm run test:rust:related --` 覆盖 27 个直接与反向依赖 crate，
退出码 `0`；其中 Agent Protocol `38/38`、Agent Runtime `176/176`、App Server `1543/1543`、
model-provider `186/186`。首次未隔离运行在 App Server 出现 10 个失败：9 个由真实用户配置中的已拒绝旧
字段产生额外 `configWarning`，1 个为同批时序断言；隔离重跑全部通过。未修改用户配置，也未恢复旧字段
兼容。workspace rustfmt、全树 diff check 与 `npm run governance:legacy-report` 通过，治理结果为 `0` 零引用
候选 / `0` 分类漂移 / `0` 边界违规。`npm run smoke:agent-runtime-current-fixture` 的历史恢复 `31/31`、
流式收尾 `32 passed / 50 skipped`、Electron/App Server fixture guard `90/90`、renderer 与 Electron host build
均通过；首次 App Server sidecar rebuild 因本机磁盘只剩约 `1.8 GiB`、Cargo 报
`No space left on device` 而中止。其他进程释放临时输出后，本轮未删除约 `143 GiB` 的
`lime-rs/target`，磁盘恢复约 `75 GiB` 可用并完整重跑成功，命令退出码 `0`。Gate B 覆盖真实 Electron、
preload/IPC、`app_server_handle_json_lines`、App Server、runtime/read model 与 GUI；关键场景包括首页/问候
热路径、Coding Workbench、图片与普通画图意图、停止后继续、四类 approval、rich draft 恢复、active
steer、plan/history hydrate、Skills、MCP structured content、media reference、Expert Plaza/Panel 与文章编辑器，
`liveProviderUsed=false`，未命中生产 mock fallback。当前证据等级为 Unit + Domain integration + App Server
integration + Gate B；本刀不修改 Renderer/Electron。架构确认：`root, 2026-07-26`；owner 与依赖方向
不变，只把已有 typed snapshot 接入 current consumer。

### 2026-07-26 effective multi-agent world-state producer

目标与写集：继续关闭 world-state 的一个 typed producer，并按 Codex current
`MultiAgentMode`/`effective_multi_agent_mode` 语义删除 Lime 的 arbitrary JSON 控制面。写集限定为
`agent-protocol` typed enum/world-state renderer、App Server v2 protocol/producer、schema/generated TypeScript、
public JSON-RPC/provider request 回归、架构与状态文档；不扩张 Multi-Agent topology、GUI 或 Electron owner。

完成结果与分类：新增 exact `MultiAgentMode = explicitRequestOnly | proactive | custom(String)` typed owner；
App Server 只从 resolved reasoning effort 生成 effective state，`ultra -> proactive`，其余为
`explicitRequestOnly`。deprecated `thread/start.multiAgentMode` 与 `turn/start.multiAgentMode` 仍按 Codex typed
wire 接受，但 runtime 明确忽略，不写 durable metadata、不覆盖 effective mode；Thread start/resume/fork 响应
固定返回 `explicitRequestOnly`。provider-visible world-state 使用同一 typed enum 渲染 Codex 对齐的
`<multi_agent_mode>` instruction。typed enum、v2 schema/client、effective producer 与 consumer 为 `current`；
任意 JSON mode、请求覆盖 effective policy 与 durable `multiAgentMode` 副本为
`dead / deleted / forbidden-to-restore`；无 `compat/deprecated` runtime 路径。AGENTS/apps/plugins/environment
instructions、realtime 和 Codex durable full/patch world-state history 仍缺，因此
`runtime-environment-context` 保持 `partial`。method 范围不变，总进度仍为 `69 / 180 = 38.3%`。

架构影响：重大，公开 v2 字段从 `Value/unknown` 收紧为 Codex typed union，并建立 effective mode 唯一
producer；全局架构图已同步。责任开发者确认：`root, 2026-07-26`；已核对 owner、数据流、deprecated
wire、durable metadata、provider visibility、schema/client 与验证门禁。

### 2026-07-26 Product DB dead migration 清退与多模型 Gate B 收尾

目标与写集：在无历史兼容负担前提下，删除 Product DB 整库搬迁双轨，并把 Settings provider Gate B
收敛到 current CRUD/model/auth 场景；同时接管 provider 审计留下的 Codex/OpenCode lowering 骨架，完成
fail-closed capability 与图片 provider route 的真实 Electron 复验。写集限定为 Core `app_paths`、治理守卫、
Settings Gate B 聚合器、`model-provider`/`runtime-core` request lowering、App Server media route lowering及本计划；
不恢复 migration wrapper，不扩张未实现 provider adapter，不把 fixture 结果冒充 live provider。

完成结果与分类：Product DB 只创建 current AgentRoot 路径；旧整库复制、migration manifest、cleanup 模块和
Provider migration Electron fixture 已物理删除，SETTINGS Gate B 从 `19` 个场景降为 `18` 个，只保留 current
`provider-crud-model-auth`。当前 adapter capability 继续 fail closed；Responses Lite input prefix、
reasoning summary/text verbosity、parallel tool 三态和 selected-model trace 已补入 current provider lowering。
未知 capability 现在 fail closed；由此暴露的图片路由错误已在 App Server lowering 修正：顶层 workflow
contract 继续记录 `text_generation / image_generation / vision_input`，传给图片 provider 的 exact route 只要求
`image_generation`，图片输入能力继续由 input modalities 判断。AgentRoot Product DB、model-control importer、
provider capability/lowering 和 media resolved route 为 `current`；整库复制、manifest、cleanup、migration
fixture 与 SETTINGS recovery 场景为 `dead / deleted / forbidden-to-restore`；无新增 `compat/deprecated`。
method 范围不变，总进度仍为 `69 / 180 = 38.3%`。

验证：Core `app_paths` `18/18`、Core 全量 `670/670`，删除/SETTINGS 治理定向 Vitest `23/23`；provider
capability `3/3`、model-provider `191/191`、runtime-core `50/50`、Agent model request policy `17/17`。
本次图片 route lowering追加 App Server `media_task_payload` `8/8` 与 runtime-core `model_task` `6/6`。
`npm run test:contracts`、`npm run governance:scripts`、`npm run governance:legacy-report` 均通过，治理结果为
`0` 零引用候选 / `0` 分类漂移 / `0` 边界违规。同期 Hook 骨架的机械格式漂移与未使用 import 已清理，
定向回归 `8/8`；workspace rustfmt、scoped Prettier 与 `git diff HEAD --check` 均通过。首次聚合 Gate B 在
真实 media worker 暴露
`image_worker_start_failed` 后按上述边界修复；图片命令定向 Electron fixture 与完整
`npm run smoke:agent-runtime-current-fixture` 随后均退出码 `0`。完整 Gate B 覆盖真实 Electron、preload/IPC、
`app_server_handle_json_lines`、App Server、runtime/read model、media worker 与 GUI，包含图片/普通画图、
approval、active steer、Plan 历史恢复、Skills、MCP、media reference、Expert 与内容工厂场景，
`liveProviderUsed=false`，未使用生产 mock fallback。架构确认：`root, 2026-07-26`；provider/network owner、
App Server route owner 与 Electron Desktop Host 依赖方向不变。

剩余多模型 P1：运行期 breaker open 后尚不会回到 resolver 选择备用 route；服务端实际生效模型与
verification 尚未进入 canonical event；Gemini/Vertex/Bedrock/Ollama/Azure/Fal adapter 仍明确 unavailable；
hosted web/image request + reducer + Item 闭环尚未实现。以上均维持 fail closed，不以名称、默认值或普通
function tool 猜测 capability。下一刀优先 health-aware runtime reroute 与 model/rerouted evidence，再按完整
wire fixture 成组扩展 adapter/hosted tool。

### 2026-07-26 health-aware runtime route fallback

目标与写集：完成上一节的最高优先级多模型 P1，让 provider 的可安全重试运行期失败回到 RuntimeCore
resolver 选择备用 profile route。写集限定为 `agent-runtime` provider failure contract、RuntimeCore model
routing exclusion/evidence、App Server runtime backend/resolver、对应测试、全局架构与本计划；不修改
App Server method/protocol/schema、Renderer、Electron，不新增 `model/rerouted` 平行事件或生产 mock。

完成结果与分类：`model-provider` 产生的 classification/retryable 经 `agent-runtime` 保留到单 route
`ReplyAttemptError`；只有未输出、未消费 pending steer、provider 调用前未发 structured-input warning 的
rate-limit/provider-internal/transport retryable 失败允许重路由。已消费 steer 的 route 会关闭 reroute safety，
避免第二次执行从原始 history 重建时静默丢输入；已发 warning 的 route 同样关闭 reroute safety，避免重复
用户可见诊断。
RuntimeCore 接收明确的 provider/model exclusion，将失败 route 与结构化 runtime failure 保留在
`routingAttempts`，并继续解析下一条 ready candidate。App Server 重新解析 selection、credential 和 provider
config 后执行备用 route，复用既有 `routing.fallback.applied` 并标记
`fallbackReason=runtime_provider_failure`。direct provider config、auth/permission/quota/request/context/content
policy/unknown、非 retryable 及已经产生输出的失败均不 fallback；无备用 candidate 时返回最近一次真实
provider error。evidence 不包含 endpoint、credential ref、API key 或 provider 错误正文。以上均为
`current`；无新增 `compat/deprecated/dead` surface，第二 resolver、direct-config profile 偷换及输出、steer、
warning 后重放为 `dead / forbidden-to-restore`。method 范围没有变化，总进度仍为
`69 / 180 = 38.3%`。

验证与退出条件：RuntimeCore exclusion/evidence 单测、Agent Runtime canonical provider error 传播单测、
App Server direct/unsafe failure policy 单测与真实双 provider HTTP fixture 已通过；fixture 证明 primary 503
在当前 provider retry 预算耗尽后，RuntimeCore 排除 primary，backup OpenAI SSE 成功闭合 Turn，并产生备用
provider identity 与 `runtime_provider_failure` evidence。2026-07-27 安全收尾另补已消费 steer、提前 warning
禁止重路由的回归，以及 HTTP transport URL/error chain、HTTP error response body、route endpoint/credential
reference 不进入 durable failure/evidence 的去敏断言。最终定向验证为 provider transport `6/6`、steer
reroute guard `1/1`、Lime Agent current provider turn `24/24`、App Server runtime reroute `2/2`；workspace
rustfmt、全树 diff check 与 legacy governance 通过，治理结果为 `0` 零引用候选 / `1` 个既有 deprecated
分类漂移 / `0` 边界违规。`npm run smoke:agent-runtime-current-fixture` 完整通过，覆盖真实 Electron、
preload/IPC、App Server、runtime/read model 与 GUI，`liveProviderUsed=false`，未命中生产 mock fallback。
首次扩大 Rust related 在 Agent Runtime `177/178` 停于既有 world-state/trace 断言
`backend_start_exposes_trace_snapshot`（`Some(false)` 对 `None`），未为本刀回退该并行改动；其余本刀 owner
均由上述定向测试和 current fixture 覆盖。live provider 未执行，因为本刀不需要真实凭证且禁止在 evidence
中接触 secret。架构影响：重大，明确运行期失败跨 owner 的控制流与 evidence owner；责任开发者确认：
`root, 2026-07-27`。

下一优先级：服务端实际生效模型与 verification canonical evidence；随后按完整 request/stream fixture 成组
扩展 Gemini/Vertex/Bedrock/Ollama/Azure/Fal adapter，hosted web/image 继续保持 fail closed，直至 request、
reducer 与 Item projection 同时闭环。

### 2026-07-27 server model evidence 与 model/verification

目标与写集：完成 health-aware fallback 后的下一条多模型 P1，写集限定为 RuntimeCore canonical event、
`model-provider` Responses SSE/WS reducer、Agent Runtime/Lime Agent 事件贯通、App Server runtime event 与 v2
notification projector、协议/schema/generated TypeScript、范围矩阵、架构和本计划。不实现普通 provider
fallback 到 `model/rerouted` 的错误映射，不扩张 Renderer/Electron、未实现 adapter 或 hosted tool。

完成结果与分类：Responses HTTP/WS handshake 读取 `openai-model`/`x-openai-model`，event 按
`response.headers -> headers` 优先级读取同名 header，明确忽略 `response.model`；事实经 provider-neutral
`ServerModel` 进入 `model.server_reported` durable diagnostic evidence，并按 Turn 去重。verification 只在可信
Codex route，或指向 `api.openai.com` 的 first-party OpenAI Responses route 启用；只解析
`response.metadata.metadata.openai_verification_recommendation[]` 中的
`trusted_access_for_cyber`，未知/非数组/错误 event/header-only/第三方 compatible route 均 fail closed。
Agent Runtime 在 transport retry 与 tool-loop sampling 间每 Turn 最多发一次 verification；App Server 将
`model.verification` 直接投影为 exact camelCase v2 `model/verification`，重复 event 忽略，缺 identity 或未知 enum
拒绝且不回退 deprecated `agentSession/event`。server model 诊断不发 v2 side-channel，也不进入 resume item
replay。canonical producer、runtime fact、v2 DTO/method/envelope/schema/client/projector 为 `current`；普通
fallback 伪造 `model/rerouted`、信任 `response.model`、第三方伪造 cyber metadata 和 notification wrapper
fallback 为 `dead / forbidden-to-restore`；无 `compat/deprecated`。`model/rerouted` 仍为 `planned`，只允许未来
可信 requested/server mismatch 的 `highRiskCyberActivity` producer。范围矩阵把 verification 单独转为
implemented，当前真实进度为 `70 / 180 = 38.9%`。

验证：Responses 定向 `32/32`，覆盖 HTTP header、event header precedence、`response.model` 负向、verification
去重/fail-closed、第三方信任门、WS handshake/event 及 metadata 后仍允许无可见输出 transport fallback；Agent
Runtime 跨 sampling 去重 `1/1`；Lime Agent typed serialization `1/1`；App Server runtime fact mapping `1/1`、
v2 direct/fail-closed projector `2/2`；App Server protocol exact round-trip 与 envelope schema 各 `1/1`。
`cargo check -p model-provider -p agent-runtime -p lime-agent -p app-server-protocol -p app-server --tests` 通过；
schema fixture 与 generated TypeScript 已由唯一生成入口重建。最终只读安全复核发现 provider selector
`codex` 可绕过第三方 endpoint 信任门；已改为由 resolved runtime provider、Responses protocol 与 exact
`api.openai.com` host 共同判定，custom provider 的展示型 id 不再造成官方 endpoint 漏报，第三方
Codex-compatible WS 负向守卫已补齐，verification 定向回归 `4/4`。`npm run test:contracts` 完整通过：
`813` 个 schema definition、`805` 个 generated protocol type、`284` 个 client contract check；workspace
rustfmt、scoped Prettier 与全树 diff check 通过。legacy governance 通过，结果为 `0` 零引用候选 / `1` 个
既有 deprecated 分类漂移 / `0` 边界违规。`npm run smoke:agent-runtime-current-fixture` 完成 current Gate B，
19 份 Electron 场景 summary 均为 `ok: true`，覆盖 Electron、preload/IPC、App Server、runtime/read model 与
GUI，`liveProviderUsed=false`，未把 controlled fixture 冒充 live provider。架构影响：重大，新增 provider
wire fact 到 exact v2 notification 的 current 数据流；责任开发者确认：`root, 2026-07-27`。

下一刀先让 `model.server_reported` evidence 自描述 provider、selected model 与 route attempt，并建立 Turn +
route + reported model 去重键；随后才实现可信 requested/server mismatch 的 `model/rerouted` producer。该诊断
增强不改变本节 `model/verification` method 的 implemented 分类，也不得把普通 provider fallback 改名为
`model/rerouted`。

### 2026-07-27 model/rerouted 与 route-aware server evidence

目标与写集：完成上一节声明的多模型 P1 下一刀。写集限定为 RuntimeCore canonical event、
`model-provider` trust boundary、Agent Runtime/Lime Agent 传播、App Server route evidence/transient sink/v2
projector、协议/schema/generated client、范围矩阵、架构与本计划；不修改 Renderer/Electron，不增加 warning
文案，不实现缺失 adapter/hosted tool。

完成结果与分类：`CurrentProviderClient` 只在 Responses、resolved provider 为 `openai|codex` 且 endpoint host
为 exact `api.openai.com` 时信任 server-model metadata。requested/server model 按 ASCII 大小写不敏感比较；
首次 mismatch 产生 canonical `ModelReroute { from_model, to_model,
HighRiskCyberActivity }`，相同或仅大小写不同不产生。transport replay 与 tool-loop sampling 由 provider stream、
Agent Runtime 和 App Server 三层收敛为每 Turn 最多一次；跨 provider route 继续只保留第一次 cyber reroute。
App Server 通过 transient sink 发布 exact camelCase v2 `model/rerouted`，不追加 state/EventLog，不进入 cold
resume item replay；缺 identity、空 model、未知 reason 均 fail closed，不回退 `agentSession/event`。
`model.server_reported` durable evidence 同步增加 provider、requestedModel、selectedModel、routeAttempt，并以
Turn 内 route + reported model 去重。普通 retryable 503 route fallback 仍只有
`routing.fallback.applied`，没有 `model/rerouted`。以上 producer、canonical event、transient boundary、v2
DTO/method/envelope/schema/client/projector 均为 `current`；selector/展示名建立信任、第三方 endpoint 伪造、
普通 fallback 冒充 cyber reroute、durable/resume reroute 为 `dead / forbidden-to-restore`；无
`compat/deprecated`。

验证与进度：`cargo check -p app-server --tests` 通过；model-provider tracked stream `4/4`、Agent Runtime
sampling 去重 `1/1`、App Server protocol exact round-trip `1/1`、App Server reroute/route evidence/fallback
隔离 `4/4` 通过。schema fixture 与 generated TypeScript 已由唯一生成入口重建，为 `815` definitions / `807`
protocol types。范围矩阵将 exact `model/rerouted` 从 planned 移入 implemented，计数为
`71 implemented / 109 planned / 34 product-scope-excluded`，产品范围完成度 `71 / 180 = 39.4%`。架构影响：
重大，新增既有 runtime event pipeline 的明确 transient 分支；责任开发者确认：`root, 2026-07-27`。
最终门禁：`npm run test:contracts` 通过（`284` 个 client contract checks，schema/generated client 零漂移）；
Codex method scope 守卫直接 Vitest `4/4` 通过（smart related 入口先遇到 Vite `EISDIR .../electron`，未将该
工具入口错误冒充测试结果）；`npm run governance:legacy-report` 为 `0` 零引用候选 / `1` 个既有 deprecated
分类漂移 / `0` 边界违规；`npm run smoke:agent-runtime-current-fixture` 完整通过，覆盖真实 Electron、
preload/IPC、App Server、runtime/read model 与 GUI，`liveProviderUsed=false`；workspace rustfmt 与全树 diff
check 通过。live provider 未执行，本刀信任/去重语义由受控 provider stream 与 HTTP fixture 验证，不接触
真实凭证。

下一刀：优先补 Grok/OpenCode 参考下仍 fail closed 的 provider adapter 与 hosted web/image request + reducer +
Item 闭环；或按产品范围优先级实现 Skills/Plugins/Apps watcher/readiness。不得继续用近似 model fallback、
function tool 或 provider 名称冒充 capability/runtime parity。

### 2026-07-27 Gemini GenerateContent transport 与工具历史闭环

目标与写集：完成上一节多模型 P1 的第一个新增 transport。provider/model control plane 继续参考
`grok-build`，Gemini endpoint、canonical lowering、tool schema 和 SSE reducer 参考 OpenCode；写集限定为
`model-provider`、RuntimeCore route、Agent Runtime/tool lifecycle、App Server readiness/catalog/provider history、
services/skills current client 接线、架构与本计划。未修改 Renderer/Electron，不新增 provider crate、compat
wrapper 或生产 mock。

完成结果与分类：新增 dedicated `GeminiGenerateContent` protocol，从 provider store、catalog、enabled API key
readiness、RuntimeCore Chat admission、App Server route lowering 到 `CurrentProviderClient` 使用同一 current
事实源。wire 固定为 Google `streamGenerateContent?alt=sse` 和 `x-goog-api-key`，禁止 Bearer；request lowering
覆盖 system instruction、text、inline base64 image、assistant function call、function response、generation config
与 Gemini tool-schema projection。SSE reducer 覆盖 text/reasoning/tool lifecycle、usage trailer、finish reason、
prompt blocked、malformed part 和 truncated EOF。`thoughtSignature` 经 canonical `provider_metadata`、Agent tool
call、通用 `ToolLifecycleEvent`、`ThreadItem.metadata` 与 provider history lowering 跨 Turn 保留，不新增 Gemini
专属持久化 schema。Gemini 为 `current`；Vertex/Azure/Bedrock/Ollama/Fal chat 仍 fail closed；旧的“Gemini
unsupported/non-chat”测试和 Skills 测试专用 mapper 为 `dead / deleted`；无 `compat/deprecated`。

验证：跨 crate `cargo check` 覆盖 `tool-runtime/model-provider/agent-runtime/lime-agent/lime-services/lime-skills/
app-server --tests` 并通过；Gemini request-capture/reducer `4/4`、Agent Runtime metadata propagation `1/1`、
Lime Agent item projection `1/1`、App Server routing `11/11`、Gemini catalog `1/1`、provider-history restore
`1/1`、services unsupported fail-before-network `1/1` 均通过。`npm run test:rust:related -- ...` 完整通过，
其中 Agent Runtime `180`、App Server `1560`、model-provider `203`、RuntimeCore `51`、tool-runtime `306`，并
覆盖全部推导出的反向依赖 crate。`npm run smoke:agent-runtime-current-fixture` 完整通过，覆盖真实 Electron、
preload/IPC、App Server、runtime/read model 与 GUI，最终闭合到内容工厂 Article Editor / `articleDraft` 场景，
`liveProviderUsed=false`。`npm run test:contracts` 通过（`815` 个 schema definitions、`807` 个 generated
protocol types、`284` 个 client contract checks，生成文件零漂移）；`npm run governance:legacy-report` 为 `0`
零引用候选 / `1` 个既有 deprecated 分类漂移 / `0` 边界违规；workspace rustfmt、三份范围/架构 Markdown
Prettier 与全树 diff check 均通过。live provider 未执行，不读取真实凭证；wire 证据来自隔离 loopback
fixture，未把受控 fixture 冒充真实 Gemini 网络调用。

进度口径：本刀没有新增 Codex App Server method，因此 method 产品范围仍为
`71 implemented / 109 planned / 34 product-scope-excluded`，即 `71 / 180 = 39.4%`；该百分比只衡量 exact
method boundary，不代表多模型 transport 总完成度。多模型 transport 本刀新增了 Gemini request/stream/tool/
history 的完整切片。

下一刀：按控制面优先级实现 hosted web/image 的 request + reducer + Item 闭环，或选择下一条 provider
transport；不得把 Vertex Gemini、Azure、Bedrock、Ollama 或 Fal 通过 Gemini alias 放行。Ollama 后续仅允许
按 Codex current 的 Responses transport 独立收敛，不恢复 Ollama Chat。

### 2026-07-27 Ollama Chat 删除与 Responses 收敛

目标与写集：Codex HEAD 已删除 `wire_api = "chat"` 与 `ollama-chat`，内建 Ollama 使用 keyless Responses。
本刀删除 Lime 的 `OllamaChat` protocol surface，把 provider route/readiness/catalog/config/test probe 收敛到
现有 `OpenaiResponses` + `NoAuth` owner；`/api/tags` 继续只做模型发现。写集限定为 protocol/schema/generated
client、RuntimeCore、model-provider、Agent provider config、App Server route/catalog、services、contract guard、
架构与本计划；未修改 Renderer/Electron，不新增 compat、provider crate 或生产 mock。

完成结果与分类：Ollama provider type/name 解析为 `OpenaiResponses`，framing 为 SSE，base host
`http://127.0.0.1:11434` 由统一 endpoint builder 生成 `/v1/responses`。stored/direct keyless route 均 ready，
catalog 保留 Ollama 模型；provider config 保留实际 resolved provider identity，认证显式为 `NoAuth`，connection/
chat probe 不再强制选择 API key。专用 loopback fixture 捕获 canonical system/user/tool call/tool result/tool
definition request，证明 POST `/v1/responses`、无 Authorization、Responses SSE text/usage/finish 闭环。
`OpenaiResponses` adapter、Ollama keyless readiness/catalog/probe 为 `current`；`ProtocolKind::OllamaChat`、
`ollama_chat`、NDJSON agent turn、专用 lowering 与 Chat Completions fallback 为
`dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。

验证：Ollama model-provider loopback `1/1`、RuntimeCore route `1/1`、Agent no-auth config `1/1`、App Server
Ollama route/readiness/catalog `4/4`、services keyless credential/protocol `1/1`、既有 unsupported provider 负向
集合 `1/1` 均通过。`npm run test:rust:related -- ...` 完整通过，其中 Agent Runtime `180`、App Server
`1561`、model-provider `204`、RuntimeCore `52`、tool-runtime `306`，并覆盖全部推导出的反向依赖 crate。
`npm run test:contracts` 通过（`815` 个 schema definitions、`807` 个 generated protocol types、`284` 个 client
contract checks，生成文件零漂移）；`npm run governance:legacy-report` 为 `0` 零引用候选 / `1` 个既有
deprecated 分类漂移 / `0` 边界违规。`npm run smoke:agent-runtime-current-fixture` 完整通过，覆盖真实
Electron、preload/IPC、`app_server_handle_json_lines`、App Server、runtime/read model 与 GUI，最终闭合到内容
工厂 Article Editor / `articleDraft` 场景，`liveProviderUsed=false`。live provider 未执行，不读取真实凭证；
Ollama wire 证据来自隔离 loopback HTTP fixture，未把受控 fixture 冒充真实 Ollama 网络调用。

进度口径：本刀不新增 Codex App Server method，因此 method 产品范围仍为
`71 implemented / 109 planned / 34 product-scope-excluded`，即 `71 / 180 = 39.4%`；该百分比只衡量 exact
method boundary，不代表多模型 transport 总完成度。本刀推进的是 Ollama Responses transport 和 keyless
control-plane readiness。

下一刀：优先实现 hosted web/image 的 request + reducer + Item 闭环，或选择 Vertex/Bedrock/Azure/Fal 中一条
完整 adapter；不得以 provider alias、Chat Completions 或恢复 `OllamaChat` 冒充支持。

### 2026-07-27 Official Responses hosted web search 闭环

目标与写集：按 Codex hosted `WebSearch` tool spec 与 `web_search_call` item 语义，完成官方 Responses 的
request lowering、stream reducer、provider-executed Item lifecycle 与 capability read；Grok/OpenCode 继续只作
多模型分层参考。写集限定为 `model-provider`、Agent Runtime provider turn、App Server provider capability、
架构与本计划；未修改公共 JSON-RPC/schema、Renderer/Electron，不新增 provider crate、compat wrapper 或生产
mock。

完成结果与分类：resolved provider 必须是 OpenAI/Codex Responses 且 endpoint host exact
`api.openai.com`，canonical `WebSearch` 才 lower 为 `{ type: "web_search", external_web_access: true }`。
Responses reducer 将 `web_search_call` 收敛为 `provider_executed=true` 的 ToolCall/ToolResult，原始 response item
进入 provider metadata/history；Agent Runtime 只发出 environment=`provider` 的 started/completed lifecycle，
不调用本地 `WebSearch` executor，也不把 Finish 错判为本地 ToolCall。第三方 Responses、Ollama、Chat
Completions、未知 route 与非 canonical 别名均不获得 hosted capability/promotion。官方 request/reducer/
provider-executed lifecycle/capability projection 为 `current`；provider 名称猜测、第三方 hosted promotion、
`WebSearchTool`/`mcp.system.WebSearch` 别名提升和 provider-executed 搜索回落本地执行为
`dead / forbidden-to-restore`；无 `compat/deprecated`。

验证：`cargo check -p model-provider -p agent-runtime -p app-server --tests` 通过；model-provider hosted search
request/capability/reducer `5/5`、Agent Runtime provider-executed no-local-execution `1/1`、App Server capability
read `1/1` 通过；`npm run test:rust:related -- ...` 完整通过，覆盖 Agent Runtime、App Server、
model-provider、tool-runtime 及全部推导出的反向依赖 crate；`npm run governance:legacy-report` 为 `0` 零引用
候选 / `1` 个既有 deprecated 分类漂移 / `0` 边界违规。`npm run smoke:agent-runtime-current-fixture`
完整通过，覆盖真实 Electron、preload/IPC、App Server、runtime/read model 与 GUI，最终闭合到内容工厂
Article Editor / `articleDraft` 场景，`liveProviderUsed=false`。workspace rustfmt、两份 Markdown Prettier 与
全树 diff check 通过。live provider 未执行，不读取真实凭证；hosted wire 语义由受控 reducer/request fixture
验证，未把 current fixture 冒充 live OpenAI 证据。

进度口径：本刀不新增 Codex App Server method，method 产品范围仍为
`71 implemented / 109 planned / 34 product-scope-excluded`，即 `71 / 180 = 39.4%`；该百分比只衡量 exact
method boundary，不代表多模型 transport 总完成度。本刀推进的是 hosted Responses tool 与 Item 生命周期。

下一刀：优先收敛 model capability provenance，把 `canonical/provider_explicit/inferred_hint` 变成 route
admission 的显式事实；启发式 catalog hint 不得授权执行，Renderer 不得继续生成假 capability。随后再实现
hosted image 或下一条完整 provider adapter。

### 2026-07-27 model capability provenance 与 admission 收口

目标与窄写集：按 Codex fail-closed runtime admission 与 Grok/OpenCode 多模型 capability 分层，把 catalog hint
和 executable snapshot 拆成显式 provenance。写集限定为 Core/Services model registry、App Server route
metadata、RuntimeCore admission、Renderer `model/list` projection、架构与本计划；不修改公共 JSON-RPC method、
Electron bridge、Gemini/provider streaming、verification/reroute 或生产 mock。

完成结果与分类：`EnhancedModelMetadata` 新增 `canonical/provider_explicit/inferred_hint`；canonical registry、API
明确 capability 字段和 typed direct config 属于权威快照，App Server 将 provenance 投影到 route metadata，
RuntimeCore 只允许前两者授权 route。裸 `custom_models`、仅名称推断和无 capability 字段的 provider model list
保留为 catalog hint，但统一返回 `capability_snapshot_missing`，即使其推断对象非空也不触网。Renderer Codex
`model/list` 投影保留真实 picker/reasoning-effort/input-modality 字段并标记 `inferred_hint`，删除硬编码
`tools/streaming/json_mode/function_calling=false`、空 execution/context/tool-call/reasoning-output/Responses/
truncation/native-tool policy 和空 runtime feature。权威 provenance 与 admission 为 `current`；名称启发式 route
授权、对象存在即放行和 Renderer 假能力为 `dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。

行为影响：Ollama Responses transport、keyless readiness 和协议解析仍为 `current`，但 `/api/tags` 与裸字符串
模型不构成 capability authority，因此当前会 fail closed。下一刀必须直接把 `custom_models: Vec<String>` 替换为
携带显式 capability snapshot 的 typed model config，并让 Ollama/自定义 provider 的真实 model discovery 产出
provider-explicit snapshot；不得重新放行名称启发式或增加双轨配置。

架构影响：重大；模型发现、App Server route metadata 与 RuntimeCore admission 之间新增显式信任边界，但 owner
与依赖方向不变。架构图确认：已核对 `architecture.md` 第 19 节的 provenance 数据流、Renderer 非授权边界与
fail-closed 退出条件；责任开发者确认：`root, 2026-07-27`。本刀不新增 Codex App Server method，method 产品范围
仍为 `71 / 180 = 39.4%`；该比例不代表多模型 transport 完成度。

验证与 Gate B 收尾：App Server model route 聚合 `23/23`、metadata provenance `2/2`、全量单元
`1561/1561` 通过；`npm run test:rust:related -- <本刀 Rust 写集>` 完整覆盖 20 个反向依赖 crate 并通过，
workspace rustfmt 与 `git diff --check` 通过。Renderer 定向测试 `35/35`、TypeScript typecheck、Prettier、
`npm run test:contracts`（`815` schema definitions / `807` generated protocol types / `284` client checks，生成物
零漂移）和 `npm run governance:legacy-report`（零引用候选 `0`、既有 deprecated 分类漂移 `1`、边界违规 `0`）
通过。聚合 `npm run smoke:agent-runtime-current-fixture` 已通过历史恢复、stream terminal、Electron guards、
Renderer/sidecar build、Claw 首页与短问候、Coding Workbench，随后在图片场景发现 fixture 仍以裸
`customModels` 充当 route authority；该旧 fixture 已直接替换为带显式 capability 的 `/v1/models` discovery，
不恢复 inferred hint 放行。fixture 定向测试 `83/83`、scoped Prettier 和单场景真实 Electron
`npm run smoke:claw-chat-current-fixture -- --scenario image-command` 通过；evidence 显示带 Authorization 的
model discovery 后，同一 provider/model 进入 `/v1/images/generations`，worker task 终态 `succeeded`、结果图
`1`，`imageCommandWorkerUsedFixtureProviderAndModel` 与 `imageCommandTaskArtifactTerminal` 均为 `true`。该 Gate B
覆盖 Electron、preload/IPC、`app_server_handle_json_lines`、App Server、provider discovery、resolved route、
media worker、read model 与 GUI；使用受控 external provider，不冒充 live provider 证据。

### 2026-07-27 Provider typed models[] 配置闭环

目标与写集：完成上一节的退出条件，直接把 Provider 模型配置从裸字符串替换为唯一 typed 形态
`Provider.models[] = { id, displayName?, capability? }`。写集限定为 Core provider DAO/schema/system defaults、
Services model registry、App Server v0 protocol/local data source、图片/视频 provider model 读取、generated
client、Renderer provider gateway/设置与媒体消费者、current smoke fixture、数据库/架构事实源和本计划；不保留
旧字段、不新增 compat wrapper，不改变 Codex method 产品范围。

完成结果与分类：Provider 持久化、App Server `modelProvider/list|read|update` 和 Renderer 公共 gateway 统一消费
typed records。带 `capability` 的配置进入 registry 时标记 `provider_explicit`；id-only 配置只生成
`inferred_hint` 并在 RuntimeCore admission fail closed。App Server 双向转换保留 task/input/output/runtime、基础
capabilities 与 reasoning-effort menu/source；非法 taxonomy 直接返回结构化错误。Renderer 编辑模型 ID 时保留
已有 display name/capability，新 ID 只创建 `{ id }`，图片、视频和脚本只在明确需要 ID 投影时读取 `.id`。
`Provider.models[]`、`canonical/provider_explicit` authority 和 typed JSON-RPC/client 为 `current`；无新增
`compat/deprecated`；`custom_models/customModels`、字符串数组持久化合同、名称启发式执行授权和 Renderer 假
capability 为 `dead / deleted / forbidden-to-restore`。生产源码、Renderer、packages 与 scripts 对旧字段扫描为
零匹配。

验证：跨 Core/Services/App Server/Server 的 `cargo check --tests` 通过；Core system provider `3/3`、Services
model registry `59/59`、App Server provider/projection `17/17`、media route `10/10`、Server image provider
`45/45`、公共 `model/list` JSON-RPC `2/2` 通过。Renderer/provider/media 显式集合 `120/120` 通过，包含
capability 保留和 UI 添加/聊天试跑边界；TypeScript typecheck 通过。`npm run test:contracts` 通过（`817`
schema definitions、`809` generated protocol types、`284` client checks，生成物零漂移）；scripts governance
通过；legacy governance 为零引用候选 `0`、既有 deprecated 分类漂移 `1`、边界违规 `0`；workspace rustfmt
与 `git diff --check` 通过。图片场景 Gate B 已在上一节完成，本节收尾只增加 typed conversion/UI 回归和文档，
不重复启动 Electron。

进度口径：本次 Provider typed config 切片完成度为 `100%`。Codex method 产品范围没有新增 method，仍为
`71 implemented / 109 planned / 34 product-scope-excluded`，即 `71 / 180 = 39.4%`；该百分比只衡量 method
边界，不能代表本切片或整体多模型 transport 完成度。架构影响：重大，Provider capability 从配置到 route
admission 建立了单一 typed 事实链；架构图确认：已更新 `architecture.md` 第 19 节；责任开发者确认：`root,
2026-07-27`。

### 2026-07-27 Catalog refresh 与 Turn 选择协调闭环

目标与写集：按 Grok 多模型控制面的 catalog refresh/current selection 语义，在 Codex 单一 Turn admission owner
内完成 provider/model 自动协调。写集限定为 App Server model catalog、session settings、Turn admission、direct
route snapshot、v2 notification projector、公共 JSON-RPC 回归、架构与本计划；不新增 selection store、私有
JSON-RPC method、compat wrapper 或生产 mock。

完成结果与分类：所有生产 `start_turn_inner` 入口在 provider execution 前消费同一 catalog generation。当前选择
仍为 visible、authoritative、chat-capable 且 route preflight 通过时保持不变；失效时先选同 Provider，再按 catalog
顺序选择其他 ready Provider，并通过既有 `thread/settings/update` actor preflight 后持久化模型、默认 effort 与
service tier。generation 最多重试三次，持续变化或无可执行模型 fail closed。显式 direct provider config 与带
`routeSource=direct_provider_config` 的 durable route 不参与 catalog 替换；选择变化会清除旧 provider config/
`agentControlRoute`，阻止内部 continuation、queued resume、workflow retry 和 mailbox 复用旧 route。前台
`turn/start` 在同一 dispatch 中恰好发送一次 exact `thread/settings/updated`；后台入口通过 transient runtime event
进入同一 v2 projector，不写 EventLog、不参与 resume replay。该 generation/reconcile/actor/notification 链为
`current`；silent fallback、`inferred_hint` 执行授权、direct route catalog 替换、旧 route 继续执行和第二套 selection
store 为 `dead / forbidden-to-restore`；无 `compat/deprecated`。

执行边界：catalog reconciliation 只在 `ExecutionBackend::requires_provider_selection()` 为 `true` 时进入。
provider-backed RuntimeBackend 必须经过同一 selection/admission；受控 external fixture backend 不承担生产
provider 选择，因此不会为了测试数据再造第二套 catalog。该 guard 同时适用于前台 `turn/start`、内部
continuation、queued resume、workflow retry 与 mailbox 恢复，避免不同入口各自猜测模型。

Gate B 收尾时发现 discovery cache 的事实源断裂：`/v1/models` 已按 API Key 指纹写入 credential-scoped cache，
但 `model_catalog()` 只读 unscoped cache；fixture provider 的静态 `models[]` 为空时，Turn admission 因此得到
`model_catalog_has_no_executable_selection`。Services 现提供统一的 scoped cache read，App Server catalog 遍历
Provider 的 enabled keys，通过 durable credential ref 读取对应 cache，并在同一 Provider 内合并去重；只有无
enabled key 且运行时不要求 API Key 的 keyless Provider 才读取 keyless cache。未把 discovery 结果回写静态
`models[]`，也未放宽 `inferred_hint` admission。无 scope cache 冒充 credential authority 的旧读取方式归
`dead / forbidden-to-restore`。

已完成验证：`cargo check -p app-server --tests` 通过；catalog owner 回归 `4/4`、统一 RuntimeCore Turn 入口
`1/1`、direct route snapshot `1/1`、后台 exact settings notification projector `1/1`、公共
`model_selection_refresh_jsonrpc` `1/1` 通过；公共回归同时证明 selection notification 恰好一次、持久化设置与
`thread/read` 一致。workspace rustfmt 与 `git diff --check` 通过；旧 `custom_models/customModels` 及字符串模型
序列化扫描为零匹配。credential-scoped cache 回归为 Services `4/4`、App Server provider/catalog `14/14`，
catalog refresh/reconcile `5/5`、公共 JSON-RPC `1/1`；`npm run test:rust:related -- <本刀 Rust 写集>` 完整通过。
通用 OpenAI-compatible fixture 的 `/v1/models` 现返回它实际支持的显式 chat/text/stream/tool capability，fixture
单测 `13/13`，不再靠模型名推断授权执行。

真实 Electron Gate B 已闭合两条 provider-backed 路径。图片命令 evidence
`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-image-command-regression-summary.json`
证明带 Authorization 的 `GET /v1/models` 后进入同 Provider/model 的 `POST /v1/images/generations`，task
终态 `succeeded`、结果图 `1`、legacy/mock 命中 `0`。Content Factory 原失败场景定向复跑通过，session
`019fa3de-6518-7ed2-b2e8-75ac380b2673`；最终完整 `npm run smoke:agent-runtime-current-fixture` 退出 `0`，末场景
session `019fa3e4-47da-78b2-8fa1-cdaf26bdabf8`，覆盖 Electron/preload/IPC、App Server、credential-scoped
discovery、provider-backed `turn/start`、read model 与 Article Editor 可见终态。两条证据均使用受控 fixture，
`liveProviderUsed=false`，不冒充 live provider。架构影响：重大，已更新 `architecture.md` 第 20 节；责任开发者
确认：`root, 2026-07-27`。

进度口径：catalog refresh/current selection 切片完成度为 `100%`。本刀不新增 Codex App Server method，method
产品范围仍为 `71 implemented / 109 planned / 34 product-scope-excluded`，即 `71 / 180 = 39.4%`；该百分比只
衡量 exact method boundary，不代表本切片或整体多模型 transport 完成度。

### 2026-07-27 Official Responses hosted image generation 闭环

目标与写集：沿用 hosted web 的 provider-executed lifecycle，完成 official Responses
`image_generation` request、stream reducer、terminal history、App Server exact Item 和 Renderer read model。
写集限定为 `model-provider`、Agent Runtime provider turn、App Server protocol/projection、generated schema/client、
Renderer canonical item reader、架构与本计划；不把本地 `lime_create_image_generation_task` 提升为 hosted tool，
不新增 Electron command、provider backend 或 compat DTO。

完成结果与分类：仅 official OpenAI/Codex Responses route 且 endpoint host 精确为 `api.openai.com` 时暴露 hosted
image capability，并且只把 canonical `ImageGeneration` lowering 为 `{ type: "image_generation" }`。第三方
Responses gateway、Ollama、Chat Completions、别名和本地 media task tool 保持普通 function。Responses reducer
消费 `image_generation_call` 的 added/done/completed，按 item identity exactly-once 发出 provider-executed
ToolCall/ToolResult，completed 缺字符串 `result` 时 fail closed，最终 finish reason 保持 `Stop`。Agent Runtime
不走本地 executor，并按 `type + id` 用 terminal raw item 覆盖 history 中的 `in_progress` item。

App Server 将 terminal provider metadata 投影为 Codex exact `ImageGenerationItem`：`id/status/result` required，
`revisedPrompt/savedPath` optional；Renderer 投影为 dedicated `image_generation` read-model item，缺 required 字段
直接拒绝，不再降级 generic extension。以上 request/reducer/lifecycle/history/protocol/read model 为 `current`；
loose `result?: Value/status?: String` DTO、alias/第三方 hosted promotion、本地工具重复执行与 generic extension
降级为 `dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。

验证：model-provider image generation `5/5`、Agent Runtime provider-executed 定向 `2/2`、App Server image
projection `6/6`、Renderer canonical item reader `23/23` 通过。`npm run test:rust:related -- <本刀 Rust 写集>`
扩展到 19 个 current owner/反向依赖 crate，全部 lib 测试通过，其中 Agent Runtime `182/182`、App Server
`1574/1574`、App Server protocol `86/86`、model-provider `214/214`、RuntimeCore `52/52`、tool-runtime
`306/306`。协议 schema 在独立临时目录生成后机械同步；`npm run test:contracts` 通过（`817` schema
definitions、`809` generated protocol types、`284` client checks，生成物零漂移）。

`npm run smoke:agent-runtime-current-fixture` 完整通过，覆盖真实 Electron、preload/IPC、App Server sidecar、
history/terminal/read model 与 GUI current fixture，`liveProviderUsed=false`；它证明现有产品主链无回归，不冒充
official hosted image 的 live provider 证据。`npm run governance:legacy-report` 通过（零引用候选 `0`、既有
deprecated 分类漂移 `1`、边界违规 `0`），workspace rustfmt 与 `git diff --check` 通过。`npm run test:related`
因 Vite 将仓库 `electron/` 目录当文件读取而报 `EISDIR`，精确 Vitest 文件入口已替代并通过。未读取真实凭证、
未执行 live provider。

进度口径：本次 request/reducer/lifecycle/history/protocol/read-model 骨架切片完成度为 `100%`。尚未计入本切片的
产品细节是把 hosted `result` base64 真正渲染为聊天图片，以及 official OpenAI live request/stream 证据；两项
保持后续工作，不能用 current fixture 替代。本刀不新增 Codex App Server method，method 产品范围仍为
`71 / 180 = 39.4%`；该百分比只衡量 exact method boundary，不能代表 hosted image 或多模型 transport 完成度。
架构影响：重大，已更新 `architecture.md` 第 21 节；责任开发者确认：`root, 2026-07-27`。

### 2026-07-28 Hosted image Agent Chat 消费闭环

目标与写集：完成上一节明确保留的用户可见消费链，把 App Server dedicated `image_generation` read-model item
接入 Agent Chat 的实时流、历史恢复和图片展示。写集限定为 Renderer 既有 ThreadItem 内容投影、历史 hydrate、
实时 content-part upsert、媒体引用展示、定向测试和本计划；不修改 provider request/reducer、App Server 协议、
Electron command、多模型 catalog 或发布文件。

完成结果与分类：`completed + generation_status=completed + 非空 result` 按 Codex exact 语义投影为
`data:image/png;base64,<result>`，复用唯一 `media_reference` 内容类型，并透传 `revised_prompt` 与 `saved_path`。
历史只有图片、没有 assistant final text 时仍生成 assistant 图片消息；实时 `item.completed` 原位 upsert，重复事件按
`threadItemId` exactly once。`in_progress`、`failed` 与空 result 均不生成破图。hosted data URI 在聊天内直接渲染
`img`，不把 base64 展开为可见文本或复制进调试属性；普通 sidecar/file media reference 继续使用原卡片与 Workbench
预览。该 dedicated item -> existing content part -> existing renderer 链为 `current`；新建 hosted-image DTO/卡片体系、
从 prompt/model/saved path 猜图、失败态占位图和刷新后才可见的历史-only 消费为 `dead / forbidden-to-restore`；
无 `compat/deprecated`。

验证：内容投影、历史恢复、实时同步与 StreamingRenderer 精确 Vitest `44/44` 通过；Renderer/Node TypeScript
typecheck 通过；scoped Prettier 与 `git diff --check` 通过。`npm run test:related -- <本刀源码写集>` 复现既有 Vite
`EISDIR: read .../electron` 基础设施错误，未出现断言失败，精确 Vitest 作为定向证据。完整
`npm run smoke:agent-runtime-current-fixture` 退出 `0`，覆盖真实 Electron/preload/App Server、历史恢复、图片命令、
media reference 与 Workbench，`liveProviderUsed=false`；`npm run verify:gui-smoke` 退出 `0`，完成 renderer/host/
App Server sidecar 构建和真实 Electron shell smoke。未读取真实凭证、未执行 official OpenAI live provider，现有
fixture 证据不冒充 hosted image live request/stream。

进度口径：本次 hosted image 用户可见消费切片完成度为 `100%`；上一节剩余项现只保留 official OpenAI live
request/stream 证据。本刀不新增 Codex App Server method，method 产品范围仍为 `71 / 180 = 39.4%`，不能把该
method 百分比解释为本切片或多模型 transport 完成度。架构影响：非重大；复用既有 ThreadItem、`media_reference`
与 StreamingRenderer owner，依赖方向和第 21 节架构图未变化。

### 2026-07-28 Azure OpenAI Responses adapter

目标与写集：参考 OpenCode Azure provider 的默认语义，在既有 `model-provider` 网络 owner 内实现 provider-aware
Responses adapter：Azure route 默认进入 Responses，使用 resource base URL 下的 `/openai/v1/responses`、
`api-key` header 和 `api-version=v1` query。写集限定为 RuntimeCore route protocol、Agent provider 配置、
model-provider adapter/request capture、Services/App Server readiness/catalog 回归和本计划；不新增 provider
crate、兼容 wrapper、Electron command 或 GUI surface，不修改 hosted image 写集与发布文件。

退出条件：Azure provider identity 必须保留到网络边界，禁止退化为普通 OpenAI Bearer adapter；Responses
request/history/tool/SSE 继续复用 current canonical lowering/reducer；hosted web/image capability 保持 fail closed；
loopback request-capture 必须证明 exact path/query/header/body/terminal stream；configured provider readiness、
direct route admission 与 connection test 改为正向。最低验证为 model-provider/Agent/RuntimeCore/Services/App Server
定向测试、`npm run test:rust:related -- <写集>`、current runtime fixture、rustfmt 和 `git diff --check`。本切片
已完成。

完成结果与分类：Azure OpenAI Responses 为 `current`，provider identity 从 RuntimeCore resolved route 经 App Server
与 Agent typed config 保留到 `model-provider`，最终使用 `/openai/v1/responses`、`api-key` 与 typed
`api-version`；缺省版本和 system provider 存量均收敛到 `v1`。Responses lowering、function tool history、SSE
reducer 与 health/circuit owner 继续复用 current algebra，health scope 增加 normalized API version。声明式
`Provider.models[]` 可进入 catalog，自动 `/models` discovery 仍 fail closed。Azure Chat Completions、Bearer/NoAuth、
deployment URL、WebSocket、hosted web/image capability 与 OpenAI server metadata trust 为
`dead / forbidden-to-restore`；无 `compat/deprecated`。Vertex、Bedrock、Fal 的 unsupported 边界未放宽。

验证：`model-provider --lib azure_` `5/5`、RuntimeCore `1/1`、Agent `1/1`、App Server `4/4`、Services
`3/3` 通过；旧 unsupported provider 集合和 non-chat catalog 负向测试通过。跨 crate
`cargo check --tests` 覆盖 `model-provider/lime-agent/runtime-core/lime-services/app-server/lime-server/lime-skills`
并通过；模型能力推断精确 Vitest `17/17` 通过；`cargo fmt --check` 与 `git diff --check` 通过。完整
`npm run smoke:agent-runtime-current-fixture` 通过，覆盖真实 Electron/preload/App Server current fixture，
`liveProviderUsed=false`。`npm run test:related -- <模型能力文件>` 仍复现既有 Vite
`EISDIR: read .../electron` 基础设施错误，已用精确 Vitest 替代定向证据。未读取真实 Azure 凭证、未执行 live
provider，因此 fixture 不作为 Azure live 证据。

进度口径：本 Azure Responses transport 切片完成度 `100%`；不代表 Vertex/Bedrock/Fal 或全部 Codex App Server
method 已完成。架构影响：重大；扩展 current provider transport union、typed endpoint/auth 投影和 health scope，
未改变 owner 方向。架构图/文字已同步 `internal/aiprompts/architecture.md`。责任开发者确认：root，2026-07-28。

### 2026-07-28 Vertex Gemini adapter

目标与写集：参考 OpenCode 的 Google Vertex project/location 与 Bearer 认证语义，在唯一 `model-provider` 网络
owner 内实现 dedicated Vertex Gemini adapter。写集限定为 RuntimeCore route admission、Core runtime credential、
Agent/App Server provider 投影、Services readiness/catalog/connection probe、model-provider endpoint/auth/request
capture、架构与本计划；不修改 Renderer、Electron、版本、Forge、manifest、锁文件或发布脚本。

退出条件：Vertex provider identity 必须保留为 `VertexGemini`，不得通过普通 Gemini API-key 或 Custom protocol
放行；typed project/location 必须生成 regional/global Vertex project endpoint；Bearer header、Gemini canonical
body、SSE terminal/usage 必须由 loopback request capture 证明；缺 context、NoAuth、带 path/query/fragment 的 origin
必须发网前拒绝；configured readiness 和声明式 catalog 改为正向，Bedrock/Fal 继续 fail closed。自动 Vertex
模型发现不在本切片范围，不能被声明模型替代或冒充。

完成结果与分类：`RuntimeProviderProtocol::VertexGemini` 为 `current`，RuntimeCore、Agent、App Server、Services、
Server 与 Skills 都委托同一 `CurrentProviderClient`；`RuntimeCredentialData::VertexKey` 承载 typed
`project/location` 和已解析 endpoint。regional/global host、publisher path、Bearer auth、Gemini lowering/SSE reducer、
health scope、configured readiness 与 declared model catalog 均已接通。普通 Gemini alias、`x-goog-api-key`、
OpenAI-compatible body、NoAuth、WebSocket、缺 context 和 origin 自带 path/query/fragment 为
`dead / forbidden-to-restore`；无 `compat/deprecated`。Bedrock/Fal 未放宽。

验证：跨 `model-provider/runtime-core/lime-agent/lime-services/app-server/lime-server/lime-skills` 的
`cargo check --tests` 通过；`vertex` 定向测试 `7/7` 通过，其中 model-provider request/endpoint/fail-closed `3/3`、
RuntimeCore admission `1/1`、Services credential `1/1`、App Server readiness/catalog `2/2`。`npm run
test:rust:related -- <Vertex 写集>` 以退出码 `0` 完成，覆盖 `20` 个 current crate 与反向依赖。`cargo fmt --check`、
`git diff --check`、`npm run governance:legacy-report` 通过；治理结果为零引用候选 `0`、边界违规 `0`，唯一已有
`deprecated` 分类漂移与本切片无关。

current fixture 的前端终态、fixture guard、两个 Claw 首页 Gate B 与 Coding Workbench Gate B 已通过；图片命令
Gate B 的单场景重跑也通过。聚合 `npm run smoke:agent-runtime-current-fixture` 两次未能完整退出：第一次在 Coding
Workbench workspace navigation 停留首页，第二次在 hosted-image 的非关键 `modelProvider/list` 发生 `30s` IPC 超时。
两个失败场景均按聚合器同参数单独重跑通过，故记录为共享 Electron/App Server fixture 的时序不稳定，而非 Vertex
provider 回归；不得将单场景通过冒充完整聚合全绿。未读取真实凭证、未执行 live Vertex。

进度口径：Vertex Gemini adapter 的实现及 provider/current-owner 验证完成度为 `100%`；本协调计划的完整 Gate B
聚合证据仍待消除上述 fixture 时序不稳定后才能标记全绿。架构影响：重大，已更新 `architecture.md` 第 22 节；责任
开发者确认：root，2026-07-28。

### 2026-07-28 `model/list` 与 Multi-Agent 可执行模型单事实源

目标与写集：修复 Codex `model/list` picker DTO、RuntimeCore catalog recovery 和 Multi-Agent `spawn_agent`
模型说明之间的准入漂移。写集限定为 App Server model control owner、`spawn_agent` catalog projection、公共
JSON-RPC 集成测试与本计划；不扩展 Codex `Model` wire，不修改 Renderer、Electron、provider transport、版本、
manifest、锁文件或发布脚本。

完成结果与分类：App Server 现在由唯一 `is_executable_chat_model` 谓词判定 capability provenance 与 task family。
`canonical/provider_explicit` 且 task family 为空或包含 `chat/reasoning` 的模型才可进入执行控制面；
`inferred_hint` 和 image/embedding 等非 chat 模型即使 picker 可见、或请求 `includeHidden=true`，也不会进入
`model/list`。Codex `includeHidden` 语义继续只影响已具备执行资格模型的 picker visibility。RuntimeCore 自动恢复、
Codex picker DTO 与当前 Provider 下的 `spawn_agent` model/reasoning/service-tier 说明复用同一能力准入，其中自动恢复
和 spawn override 继续只消费 `visibility=list`。该统一谓词及三条消费者为 `current`；visibility-only 授权、
Renderer 二次猜测 chat 能力、inferred/media 模型进入 chat picker 或 Multi-Agent override 为
`dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。

验证：App Server `model_list` 定向回归为单元 `5/5`、公共 JSON-RPC `2/2`，覆盖 exact Codex v2 shape、分页、
hidden、provider readiness、inferred hint 与 image generation 排除；Multi-Agent catalog projection `1/1` 通过。
`npm run test:rust:related -- <本刀 App Server 写集>` 运行 App Server 全量 lib `1581/1581` 通过；workspace rustfmt
与 `git diff --check` 通过。`npm run test:contracts` 通过（`817` schema definitions、`809` generated protocol
types、`284` client checks，生成物零漂移）；`npm run governance:legacy-report` 结果为零引用候选 `0`、既有
deprecated 分类漂移 `1`、边界违规 `0`。本刀不改 GUI、Electron 或 bridge，故未重复运行 Gate A/B；Vertex 节记录的
聚合 Electron fixture 时序不稳定仍是全计划 Gate B blocker，本结果不将定向 Rust/JSON-RPC 证据冒充 GUI 或 live
provider 证据。

进度口径：本次 control-plane admission 切片完成度为 `100%`；Codex method 产品范围未新增 method，仍为
`71 / 180 = 39.4%`。后者只衡量 method boundary，不能代表本切片、多模型 transport 或全计划完成度。架构影响：
非重大；未改变第 19/20/22 节既有 owner 与依赖方向，只消除了同一 catalog 的重复准入判断。下一刀回到 Grok
control plane，审计 provider health/retry/circuit state 是否仍只有执行期内部状态、而没有 current App Server 可读
状态；不得用静态 provider readiness 冒充运行时健康度。

### 2026-07-28 静态 Provider capability 孤立命令删除

目标与裁决：完成 Grok control-plane health/retry/circuit 审计后，确认
`modelProvider/capabilities/read` 只读取全局默认 Provider，返回静态
`namespaceTools/imageGeneration/webSearch`，既不绑定当前 Thread route，也不读取共享 circuit/runtime health，
且 Renderer、Electron 与产品网关均无消费者。Codex 的同名方法成立于单一全局 `model_provider` 配置；Lime
已经按 Thread 选择 provider/model，直接复制会把全局静态值冒充当前 route truth。因此该上游 identity 保留在
method 产品范围矩阵，但从 `implemented` 改为 `product-scope-excluded`。

完成结果与分类：已删除 v2 DTO/method/envelope/schema registry、App Server ingress/handler/RuntimeCore/
LocalDataSource、Rust/TypeScript typed clients、公共 JSON-RPC 正向测试及 checked-in/generated protocol surface，
生产源码与生成物零引用。真实 provider/model capability、configured readiness、provider lowering 与共享
route health registry 继续为 `current`；`modelProvider/capabilities/read` 为
`dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。治理 catalog 新增 Rust 与前端回流守卫，禁止
恢复 method 字符串、DTO 或 helper。method 矩阵当前为
`70 implemented / 109 planned / 35 product-scope-excluded`，产品范围完成度 `70 / 179 = 39.1%`。

验证：Rust schema 与 TypeScript 生成物由唯一生成入口重建，为 `815` schema definitions、`807` generated
protocol types；schema fixture `1/1`、`npm run test:contracts` 全绿（`284` client checks，生成物零漂移）。
`npm run test:rust:related -- <本刀 Rust 写集>` 覆盖 `19` 个 current owner/反向依赖 crate 并全部通过，
其中 App Server lib `1579/1579`。`npm run governance:legacy-report` 通过（零引用候选 `0`、既有 deprecated
分类漂移 `1`、边界违规 `0`）；workspace rustfmt 与全树 `git diff --check` 通过。本刀是纯协议/App Server dead
method 删除，未改 GUI、Electron 或 bridge，故未重复运行 Gate A/B；Vertex 节已有的聚合 Electron fixture 时序
不稳定仍是全计划 Gate B blocker。

下一刀：回到多模型主链 P0，优先收口 dedicated Gemini route admission 与旧负向断言，再补 Gemini request/
stream/tool/usage request-capture；随后处理 runtime health 的 App Server 可读投影时，必须绑定 exact route scope，
不得恢复本次删除的全局静态 capability 方法。

### 2026-07-28 Exact-route provider health snapshot

目标与写集：在既有 `model-provider -> AgentRuntimeState` current owner 内补全 Grok circuit control-plane 的
只读事实，允许已解析 route 查询同一共享 breaker 的状态，而不把静态 readiness 或全局 provider metadata 冒充
runtime health。写集限定为 health registry、Agent Runtime 委托、架构事实源与本计划；不新增 App Server JSON-RPC、
Renderer 状态、compat 或 mock。

完成结果与分类：`CurrentProviderHealthRegistry::snapshot_for(exact RuntimeProviderConfig)` 对未执行 route 返回
unknown（`None`）且绝不创建 synthetic closed entry；已知 route 只返回脱敏 state、closed-window sample/failure count、
half-open probe 状态和 open/half-open retry timing。breaker 打开后当前内部实现会丢弃统计窗口，因此 counts 明确为
unknown，不伪造旧样本。`AgentRuntimeState::provider_health_snapshot` 仅委托其 clone 共享的同一 registry；session
client、HTTP/WebSocket/fallback 均未被改为共享。snapshot、registry 与 Agent Runtime 委托为 `current`；全局
provider health、静态 capability/readiness 代替 route health、unknown route 自动 closed 和无消费者 JSON-RPC
method 为 `dead / forbidden-to-restore`；无 `compat/deprecated`。

验证：model-provider health `18/18` 通过，覆盖 unknown/read-only、exact model 隔离、closed count、open/half-open
state/retry、API version/credential scope 和敏感 endpoint/key 不泄漏；Agent provider-session `4/4` 通过，覆盖 clone
共享 registry、session 切换仍保留原 route open 状态且不污染其他 model。`npm run test:rust:related -- <本刀写集>`
扩展到 `13` 个 current owner/反向依赖 crate 并全部通过，其中 App Server `1579/1579`、model-provider
`224/224`、agent-runtime `182/182`、tool-runtime `306/306`。workspace rustfmt、`git diff --check` 与
`npm run governance:legacy-report` 通过（零引用候选 `0`、既有分类漂移 `1`、边界违规 `0`）。无
GUI/bridge/protocol 改动，Gate A/B 与 contracts 不适用。

进度口径：本 internal control-plane slice 的骨架与定向回归已完成；App Server/GUI 只有出现 exact Thread route
消费者时才接入，并且必须传递解析后的 `RuntimeProviderConfig`，不能根据 provider name 猜 route。架构影响：非重大；
未改变 owner 或主链方向，只补齐现有 registry 的可读事实。责任开发者确认：root，2026-07-28。

### 2026-07-28 Exact credential reroute

目标与写集：继续 Grok 多凭证控制面 P0，把原先只按 provider/model 排除整条 route 的粗粒度运行期失败处理，
替换为 exact route 内的 credential-specific reroute。写集限定为 RuntimeCore exclusion、Services key selection、
App Server route/credential 编排、Agent Runtime 失败安全判定、架构事实源和定向测试；不新增 App Server method、
Renderer/GUI、Electron bridge、兼容 wrapper、版本、Forge、manifest、锁文件或发布脚本。

完成结果与分类：`ModelRouteExclusion` 现在区分 route 与内部 credential scope，credential ref 不进入 payload 或
Debug。RuntimeCore 只对 route scope 跳过 provider/model；credential scope 保持同一 candidate ready。App Server
对未产生输出、未消费 pending input 的 401/403、quota、429、5xx/transport 只排除当前 repository credential，
durable ref 命中失败 key 时从同一 provider key pool 选择下一把；direct request 不换 key。Services 的 exact
provider selection 会过滤失败/不可解密 key，指定 provider 存在时禁止按 provider type 跨到其他 provider。
key 池耗尽返回原 provider error，不转 backup model；已配置且非 keyless provider 没有 credential 时 fail closed，
禁止降成 `NoAuth`。上述 RuntimeCore、Agent Runtime、Services 与 App Server owner 均为 `current`；整条 route
排除 repository credential 失败、跨 provider/type 借 key、credential exhaustion 后 profile fallback 和隐式
NoAuth 为 `dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。

验证：RuntimeCore credential scope `1/1`、Agent Runtime credential-safe failure `1/1`、Services exact key
selection/exhaustion `1/1`、App Server credential 定向 `14/14` 与真实 loopback 401/503 `3/3` 通过。loopback
证明两把 key 使用同一 provider/model，事件 payload 不含 credential ref；单 key 耗尽保留 provider error且不请求
backup model。`npm run test:rust:related -- <本刀写集>` 扩展到 `14` 个 current crate/反向依赖并全部通过，
其中 App Server `1581/1581`、Agent Runtime `183/183`、tool-runtime `306/306`。workspace rustfmt 与全树
`git diff --check` 通过；`npm run governance:legacy-report` 通过（零引用候选 `0`、既有 deprecated 分类漂移
`1`、边界违规 `0`）。本刀未改公共协议、Bridge 或 GUI，故 contracts、Gate A/B 不适用。

进度口径：本次 credential-specific reroute 骨架切片完成度为 `100%`；跨 Turn 的真实 `Retry-After`、quota reset
与 credential cooldown 尚未实现，后续必须由 provider error metadata 驱动，禁止添加固定假 cooldown。架构影响：
重大；更新第 8 节运行期重路由 scope 与 fail-closed 规则。责任开发者确认：root，2026-07-28。

### 2026-07-28 Cross-Turn credential cooldown

目标与写集：在 exact credential reroute 基础上补齐 Grok 多凭证控制面的跨 Turn 恢复窗口。参考 Codex 对
`Retry-After` 秒值/HTTP-date 的解析，以及 Grok 把服务端 retry hint 保留为结构化 sampling error metadata 的做法，
写集限定为 `model-provider` HTTP error metadata、`agent-runtime` error propagation、Services credential selector、
App Server reroute 编排、架构事实源与定向测试。没有公共协议、Renderer、Electron、GUI 或 compat surface。

完成结果与分类：`CurrentProviderError` 现在携带可选非零 `retry_after`，来源只允许真实 `Retry-After` 或 exhausted
request/token quota reset header；request-layer 5xx sleep 继续按 Codex 上限裁剪，跨 Turn metadata 保留服务端完整
窗口。Agent Runtime 原样贯穿该 duration；App Server 仅在未产生可见输出、未消费 pending input、非 direct request
且已绑定 repository credential 的安全 reroute 上登记 cooldown。`ApiKeyProviderService` 使用进程内 deadline 跳过
cooldown key，durable preferred ref 同样改选同 provider 下一把，过期自动清理；exact credential read 不改变语义。
该 error/selector/reroute 链为 `current`。固定假 cooldown、把 credential ref/deadline/header 暴露到 payload/Debug/
tracing/evidence、direct request 自动换 key、跨 provider 借 key为 `dead / forbidden-to-restore`；无
`compat/deprecated`。

验证：model-provider retry metadata parser `8/8`（含 request/token 同时耗尽时取较晚 reset）、Agent Runtime error
contract `10/10`、Services cooldown selector `1/1`、App Server 真实两 Turn loopback `1/1` 通过。loopback 精确
证明第一 Turn 收到 `429 + Retry-After` 后从失败 key 切换，第二 Turn 仍跳过该 key；没有 model route fallback。
`npm run test:rust:related -- <本刀写集>` 以退出码 `0` 覆盖 13 个 current owner/反向依赖 crate，其中
Agent Runtime `184/184`、App Server `1582/1582`、tool-runtime `306/306`。`cargo check` 覆盖
model-provider/agent-runtime/lime-services/app-server/lime-server 通过；workspace rustfmt、`git diff --check` 与
`npm run governance:legacy-report` 通过（零引用候选 `0`、既有 deprecated 分类漂移 `1`、边界违规 `0`）。crate
manifest 只复用 workspace 现有 `chrono` 和锁文件已有 `httpdate`，Cargo.lock 已同步。本刀无公共命令/协议/GUI
变化，contracts 与 Gate A/B 不适用。

进度口径：跨 Turn credential cooldown 骨架完成度 `100%`；当前为进程内 control-plane health，不承诺跨 App
重启持久化。下一刀回到 provider adapter 收口，优先修 Gemini dedicated admission 与旧负向断言，再补 Gemini
request/stream/tool/usage capture。架构影响：重大；扩展第 8 节 provider error metadata 与 credential selector
状态，未改变 current owner 或主链方向。责任开发者确认：root，2026-07-28。

### 2026-07-28 Gemini dedicated adapter 收口复核

目标与写集：按当前工作树复核普通 Gemini API Key 从 catalog/readiness、RuntimeCore admission、App Server route
projection、Agent/Server/Skills credential mapping 到 `model-provider` request/SSE reducer 的完整 current 链，避免按
旧审计结论重复实现。写集只新增 App Server dedicated protocol 投影守卫、Gemini/Vertex route health 隔离守卫，
并清理正向 readiness 测试中的误导局部命名；不新增协议、兼容层、Renderer、Electron 或第二套 provider backend。

完成结果与分类：Gemini 已由 `ProtocolKind::GeminiGenerateContent` 正向准入并投影到唯一
`ModelProviderProtocol::GeminiGenerateContent`，configured provider 只在 enabled key 存在时 ready，声明模型进入
current catalog。普通 Gemini 继续使用 `x-goog-api-key` 与 Google GenerateContent endpoint；Vertex 继续使用独立
`VertexGemini` identity、typed project/location endpoint 与 Bearer token。exact route health key 的 protocol scope
保证两者即使 provider/model/gateway origin 相同也不共享 breaker。上述 control-plane、wire adapter 与隔离守卫为
`current`；Gemini/Vertex alias 互借 identity/auth/endpoint、OpenAI-compatible lowering、旧 unsupported 正向断言为
`dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。

专项验证：model-provider Gemini/Vertex request、endpoint、lowering、SSE、usage、tool、fail-closed `7/7`；RuntimeCore
Gemini/Vertex admission `3/3`；Agent provider projection `10/10`；App Server dedicated projection、Gemini readiness
各 `1/1`；Services Vertex credential projection `1/1`；Server Gemini credential mapper `1/1`；model-provider
Gemini/Vertex health isolation `1/1`；Skills current provider crate test `1/1`。workspace rustfmt 与 `git diff --check`
通过。`npm run test:rust:related -- <Gemini/Vertex 写集>` 以退出码 `0` 覆盖 13 个 current crate 与反向依赖，
其中 Agent Runtime `184/184`、App Server `1583/1583`、tool-runtime `306/306`；
`npm run governance:legacy-report` 结果为零引用候选 `0`、既有 deprecated 分类漂移 `1`、边界违规 `0`。
本刀没有公共 protocol、GUI、Electron 或 bridge 改动，contracts 与 Gate A/B 不适用。

进度口径：Gemini dedicated adapter 当前收口切片完成度为 `100%`；该百分比只描述本 adapter，不代表 Codex 全部
method 或产品计划完成度。架构影响：非重大；复用第 16/22 节既有 owner 与依赖方向，只补充隔离守卫和事实证据。
下一刀回到多模型 control plane 的 model switch/fallback policy 或尚未实现的 provider adapter，不再重复扩展 Gemini。
责任开发者确认：root，2026-07-28。

### 2026-07-28 Codex model switch provider-history 骨架

目标与写集：继续多模型 control plane P0，复制 Codex
`core/src/context/model_switch_instructions.rs` 的 `<model_switch>` developer-context 语义，并适配 Lime
唯一的 RuntimeCore provider-history owner。写集限定为 App Server provider-history、ExecutionBackend 内部
history contract、RuntimeBackend sampling 接线、对应定向测试、本计划与架构事实源；不新增 App Server
method、协议字段、pending state、GUI 分支、provider adapter 或兼容层。

完成结果与分类：provider history 从 `StoredSession.events` 选择最近一个已完成 Turn 的最后一条
`routing.decision.made`，以其 selected provider/model 作为 previous truth；当前 route 只读取 Turn preflight
后保存的 schema v2 `agentControlRoute.providerPreference/modelPreference`，RuntimeBackend sampling 则通过内部
typed `ProviderTurnHistory` 对每次实际 route selection 重新比较。二者不同时，在历史末尾、当前 user input
之前追加一次 Codex preamble 的 `<model_switch>` developer message；route 字段做 XML text escaping。
首 Turn、相同 route、仅 effort/service tier 变化不注入；failed Turn 不成为 previous truth；切换 Turn 完成后
下一 Turn 不重复注入；同一 Turn health-aware reroute 不复用初始 route marker，而按 fallback selection 生成或
移除。该临时投影为 `current`；持久化 `pendingModelSwitch`、候选设置提前成为历史事实、
provider-specific marker 与旧无切换边界行为为 `dead / replaced / forbidden-to-restore`；无
`compat/deprecated`。

专项验证：`cargo test --manifest-path lime-rs/Cargo.toml -p app-server provider_history --lib`
通过 `24/24`，覆盖 durable cold-history 推导、marker 顺序、一次性注入、同 route、effort/tier 变化、失败
Turn 隔离、首 Turn、XML escaping 与实际 reroute retarget；完整 App Server lib 回归通过 `1589/1589`；
`npm run test:rust:related -- <model-switch 写集>` 识别 App Server 并以退出码 `0` 再次通过 `1589/1589`。
App Server 包级 rustfmt check 与 scoped `git diff --check` 通过；`npm run governance:legacy-report` 为零引用
候选 `0`、既有 deprecated 分类漂移 `1`、边界违规 `0`。本刀没有公共 protocol、GUI、Electron 或 bridge
改动，contracts 与 Gate A/B 不适用。

进度口径：model-switch provider-history 骨架实现完成度 `100%`；该百分比只描述本切片，不代表 Codex
model-specific instructions catalog 或全部多模型计划完成。架构影响：非重大；复用既有 RuntimeCore
provider-history、durable routing event 与 RuntimeBackend route attempt，不改变 owner 或依赖方向。责任开发者
确认：root，2026-07-28。

### 2026-07-28 `model/list` authoritative capability 投影

目标与窄写集：继续 Grok 多模型 catalog/control-plane 主线，让公开 `model/list` 与 RuntimeCore route 直接复用同一
typed capability snapshot。写集限定为 v2 `Model` DTO/schema/generated client、App Server catalog projection、
Renderer registry projection、定向测试、架构事实源与本计划；不新增 provider network owner、Electron 后端、
mock fallback 或 compat。

完成结果与分类：Codex 的 `cursor/limit/includeHidden`、分页、picker、reasoning effort 与 input modality 字段保持
不变；Lime `Model` 追加 `providerId`、`capabilitySnapshot`、`contextWindow` 与 `maxOutputTokens`。snapshot 从
`EnhancedModelMetadata` 的 `canonical/provider_explicit/inferred_hint` provenance 和 capability 原样投影；目前
public executable catalog 已在 RuntimeCore 排除 `inferred_hint`，故 Renderer 只接受权威 snapshot。`video/file`
同时进入 Codex picker input modalities 与完整 snapshot；task family、input/output modality、runtime feature、
tools/streaming/JSON/function calling/reasoning 和 limits 不再被前端重新推断。Renderer 校验 opaque route 解出的
provider 必须等于 `providerId`，source 缺失或 provider 不一致直接失败。该 App Server catalog、typed client 与
Renderer projection 为 `current`；`model/list` 内把 capability 固定标为 `inferred_hint`、从输入模态猜 vision/
task family、固定 text output 或清空 limits 为 `dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。

验证：协议 round-trip `1/1`、App Server catalog unit `6/6`、公开 JSON-RPC `model_list_jsonrpc` `2/2`、
Renderer registry Vitest `16/16` 通过。schema fixture 与 TypeScript client 通过唯一生成入口重建（`815` schema
definitions、`807` generated protocol types）；`npm run test:contracts` 通过（`284` client checks、生成物零漂移）。
`npm run test:rust:related -- <本刀 Rust 写集>` 以退出码 `0` 覆盖 20 个 current owner/反向依赖 crate，其中
App Server lib `1590/1590`、Agent Runtime `184/184`、tool-runtime `306/306`。`npm run governance:legacy-report`
为零引用候选 `0`、既有 deprecated 分类漂移 `1`、边界违规 `0`。smart-related Vitest runner 在加载环境时触发既有
`EISDIR .../electron`，同一配置下直接 `vitest run src/lib/api/modelRegistry.test.ts` 已通过；本刀未改 GUI layout、
Electron/preload 或实时 provider，未把以上协议/JSON-RPC 证据冒充 Gate A/B。

进度口径：本切片完成度 `100%`，证明模型选择、能力显示和 route admission 共享一份 catalog 事实；不代表所有
provider adapter、runtime reroute 或 GUI Gate B 已完成。架构影响：重大；更新第 6.3 与第 19 节 public catalog
boundary，owner 与依赖方向不变。责任开发者确认：root，2026-07-28。

### 2026-07-28 多模型 / 多模态骨架最终收口

目标与参考事实源：本轮按用户最终口径完成快速骨架收口。Agent loop、Thread/Turn/Item 与 provider history
继续对齐 Codex；model catalog、thread model switch、provider readiness、credential cooldown 与 route-scoped
circuit breaker 对照 `grok-build`；canonical media parts、Gemini/Vertex/Azure/Ollama wire lowering 对照 OpenCode。
三者只提供语义参考，Lime current owner 仍是 App Server、RuntimeCore、`model-provider` 与 Renderer typed gateway。

完成结果与分类：公开 `model/list` 已投影 authoritative typed capability snapshot；
`thread/settings/update` 在持久化候选前执行 route/catalog/capability preflight，并在 Turn admission 时协调 durable
thread settings；跨 provider/model 的下一次 sampling 追加一次 Codex `<model_switch>` developer context，实际 reroute
按最终 route 重新计算。Gemini GenerateContent、Vertex Gemini、Azure Responses、Ollama Responses 为 dedicated
`current` adapter，OpenAI/Anthropic 继续复用既有 current wire；provider route health 以 protocol/model/base URL/
credential scope 隔离，credential cooldown 可跨 Turn 跳过失败 key。输入 image/audio/video/file capability 与输出
media reference 进入同一 typed catalog；Agent 对话的 current 可执行输入按 `grok-build` sampling 主链限定为
`text + image`，图片使用 canonical part、Thread/Turn/Item projection 和 GUI card/workbench preview。audio/video/file
在 dedicated adapter 完成前只保留 typed capability 表达并 fail closed，不把 schema 骨架冒充 provider wire。
Azure Chat、Gemini/Vertex alias 互借、前端 capability 猜测、独立 capability read method 与旧 Vec provider-history
contract 为 `dead / deleted / forbidden-to-restore`；Fal/Bedrock chat adapter 尚未实现，继续在发网前 fail closed，
不建 compat；无新增 `deprecated`。

验证：ProviderTurnHistory 五个 App Server integration target 共 `8/8`；Services provider/admission `23/23`、
RuntimeCore route `15/15`、App Server readiness `15/15`、model-provider current transport `123/123`；多模型/
多模态 Renderer 定向测试 `77/77`；queued durable input 与 public `turn/steer` sidecar 回归分别 `1/1`，证明
持久化 event/read model 不含 inline base64，resume 后 provider 仍获得可读 data URL。
`npm run test:rust:related -- <本轮 provider/control-plane/multimodal 写集>` 以退出码 `0` 覆盖 current owner
与反向依赖 crate，其中 App Server `1593/1593`、Agent Runtime `184/184`、model-provider `227/227`、
tool-runtime `306/306`。TypeScript typecheck、workspace rustfmt、`git diff --check`、
`npm run test:contracts`（807 generated types、284 client checks、modality/scripts/docs guards）与 production
`cargo check -p app-server` 均通过且无 history dead-code warning。

真实产品证据：`npm run smoke:agent-runtime-current-fixture` 通过真实 Electron Desktop Host、preload/IPC、
`app_server_handle_json_lines`、App Server sidecar、read model 与 GUI；其中明确覆盖 model binding、图片命令、
普通画图意图以及 `media item / imageView -> Agent Chat card -> Workbench preview`，`liveProviderUsed=false`。
`multimodal-provider-capture-smoke` 另证明首轮和历史图片均到达 provider、durable/read-model/evidence 只保留
canonical `sidecar://media/input-<sha256>.<ext>`，seed 为
`.lime/qc/agent-runtime-multimodal-sidecar-v2/lime.db`。
`npm run verify:gui-smoke` 另以独立 Electron smoke 退出码 `0` 通过。`npm run governance:legacy-report` 为零引用
候选 `0`、边界违规 `0`，仅保留既有 deprecated 分类漂移 `1`。

进度口径：用户要求的“先快速实现骨架”切片完成度 `100%`，可以结束本轮。该数字不表示 Fal/Bedrock 等
未实现 adapter 已支持，也不表示每个真实云 provider、地区、凭证组合都完成 live 验证；这些属于后续细节扩展，
也不表示 audio/video/file 已进入 Agent 对话 provider wire；这些能力必须继续在 current owner 内按 capability 和
exact wire 逐个实现。架构影响：重大；第 6.3、8、16、19、22 节
已经记录唯一 owner、grok-build/OpenCode 分工和 Gate B 边界。责任开发者确认：root，2026-07-28。

### 2026-07-28 Grok 多模型骨架最终验证与媒体模型回归修复

参考口径：多模型 catalog、默认选择、model switch、provider readiness、service tier、retry/circuit breaker 与
多模态 sampling 行为以 `grok-build` 为主参考；Codex 继续负责 Agent runtime 与 Thread/Turn/Item 语义；OpenCode
只用于补足具体 provider wire 和 canonical media lowering，不成为模型控制面事实源。

完成结果与分类：`thread/settings/update` 已把 `serviceTier` 纳入原子 route preflight，并按 catalog 中的精确 tier
ID fail closed；公开 `model/list` 在分页前为首个可见 executable model 标记唯一稳定 `isDefault`；Provider HTTP
transport 尊重 Grok `x-should-retry: false`，该 header 会压制 5xx 重试并把最终错误标为不可重试。全包验证同时发现
chat catalog reconciliation 会错误删除权威 image-only 模型，导致 public 图片命令在媒体专用 admission 前失败；
当前 owner 已改为保留 catalog 中权威、可见的专用模型，普通 chat 与 media route 仍分别在各自 admission 边界校验。
以上均为 `current`；把专用模型强制改选为 chat 模型、服务端明确禁止后仍重试、持久化未校验 tier 为
`dead / replaced / forbidden-to-restore`；无 `compat/deprecated`。

验证：`model-provider 229/229` 通过；首次 App Server 全包发现 `media_task_jsonrpc` 两条回归后没有掩盖失败，根因
修复后该 public JSON-RPC target `6/6`、专用 model reconciliation 回归 `1/1` 通过。最终
`npm run test:rust:related -- <本刀写集>` 以退出码 `0` 覆盖 13 个 current owner/反向依赖 crate，其中 App Server
`1596/1596`、Agent Runtime `184/184`、model-provider `229/229`、tool-runtime `306/306`。`npm run test:contracts`、
modality/scripts/docs guards、workspace rustfmt、`git diff --check` 与 `npm run governance:legacy-report` 通过；治理结果
为零引用候选 `0`、边界违规 `0`，仅既有 deprecated 分类漂移 `1`。此前同一骨架变更集的真实 Electron Gate B、
GUI smoke 与 multimodal provider capture 证据继续有效；本刀没有修改 Renderer/Electron 或 provider media lowering。

完成口径：用户要求的快速多模型/多模态骨架为 `100%`，可以结束本轮；这不等于完整复制 Grok。该切片当时记录的
catalog refresh、typed update notification、GUI cache 主动失效与 App Server default owner 已在下方最终收口切片完成；
当前后续细节以最终收口清单为准。架构影响：非重大；修复既有 catalog 与 admission 职责边界，没有新增 owner、
兼容层或生产 mock。责任开发者确认：root，2026-07-28。

### 2026-07-28 App Server 默认模型与 canonical media 最终收口

目标与参考边界：按用户最终确认把多模型 catalog/default/switch、provider readiness、service tier、retry/circuit
breaker 和多模态 sampling 以 `/Users/coso/Documents/dev/rust/grok-build` 为主参考；Codex 继续负责 Agent runtime、
Thread/Turn/Item、fork/history 和 GUI 护栏；OpenCode 只辅助 provider wire、canonical content 与媒体 lowering。

完成结果与分类：App Server ready catalog 顺序现在同时驱动 `model/list.isDefault` 和缺省 `thread/start`，无显式
provider/model 时由 RuntimeCore 选择首个 visible + authoritative + executable chat model，半截或空白显式 route
fail closed，精确 route preflight 仍发生在 session/thread 持久化之前。Renderer Agent warmup 不再读取第二份
`get_default_provider`，session gateway 允许缺省 route 直接交给 App Server；显式工作区选择继续原样传递。
Provider credential create/enable/delete 会进入同一个 provider-scoped catalog refresh coordinator；成功提交 cache 后
发布 typed `model/list/updated`，Renderer 失效缓存并强制重读 `model/list`，不轮询或本地推断 generation/default。
`thread/settings/update` 在 model/collaboration model 变化且未显式给 tier 时，从目标 catalog 安装受支持的
`default_service_tier`，目标无默认时清空旧 tier。公共 v2 chat `InputModality` 收窄为 `text | image`；
audio/video/file 只保留在通用 provider capability taxonomy，未实现 dedicated chat ingress/sidecar/wire 前不能作为
Agent chat 能力。上述均为 `current`。Renderer 本地默认 Provider 仲裁、跨模型残留旧 service tier、把通用媒体
taxonomy 暴露成 chat input schema 为 `dead / deleted / forbidden-to-restore`；无新增 `compat/deprecated`。

同一收口包含 canonical media 恢复：fork 会校验并复制 source session 的 image sidecar 到 target session，保持
URI/digest/media type，创建或持久化失败会回滚 target sidecar；不恢复绝对路径，不持久化 inline base64。
Renderer Message 图片不再把 `sidecar://` 直接交给 Chromium，而通过 `agentSession/media/read` 读取临时 data URL。
这些路径为 `current`；直接浏览器 sidecar scheme、durable base64 与绝对路径恢复为 `dead / forbidden-to-restore`。

验证：新增 service-tier 重建 lib 回归 `1/1`、公共 `model/list + thread/start` JSON-RPC `3/3`、Renderer session/default
定向回归 `200/200`、`cargo check -p app-server --tests` 通过。fork JSON-RPC `1/1`、Message image attachments `2/2`、
App Server 全包和 GUI smoke 已在同一变更集的前序阶段通过；最终静态收口再次通过 `npm run typecheck`、
`npm run test:contracts`（`808` generated protocol types、`284` client checks，含 modality/scripts/Electron/docs guards）、
`npm run governance:legacy-report`（零引用候选 `0`、既有 deprecated 分类漂移 `1`、边界违规 `0`）、workspace
`cargo fmt --check` 与 `git diff --check`。最终当前树 `npm run verify:gui-smoke` 退出码 `0`，证据为
`.lime/qc/project-gates/standalone-shell-01-20260728123252-79288/shell-01-electron-smoke/summary.json`。真实 Electron
`inputbar-rich-restore` 修复后定向 Gate B 通过，实际命中
`agentSession/media/read`，legacy command/mock fallback/console error/page error 均为 `0`，证据：
`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-inputbar-rich-restore-regression-summary.json`。
修复前聚合 `npm run smoke:agent-runtime-current-fixture` 在该场景失败，修复后只重跑失败场景，尚未从头全量复跑；
不得把定向通过描述为修复后聚合门禁全量通过。

最终文档收口时再次复跑 catalog refresh 公共 JSON-RPC `2/2`、Renderer registry/cache Hook `18/18`、typed
notification parser `7/7` 与完整 `npm run test:contracts`，均通过；协议仍为 `808` generated types、`284` client
checks，生成物零漂移。

文件规模退出条件：`runtime/thread_fork.rs` 当前 `1182` 行，下一次 fork/media 修改前先把 sidecar copy/rollback 拆到
`runtime/thread_fork_media.rs`；model selection/default/reconciliation 已拆到
`runtime/model_providers/selection.rs`。本轮 service tier 收尾又把目标模型默认值重建抽到
`runtime/session_operations/model_defaults.rs`，`runtime/session_operations.rs` 已降到 `928` 行；下一次 settings patch、
validation 或持久化行为修改前继续拆出 typed settings mutation owner，不再向主文件堆业务逻辑。

完成口径：用户要求的快速多模型/多模态骨架切片为 `100%`，不是“完整复制 Grok/Codex/OpenCode”。真实云
provider/地区/凭证矩阵、Fal/Bedrock adapter 及 audio/video/file 的逐 provider
chat wire 属后续细节；在 exact ingress、durability、capability admission 和 lowering 全部完成前继续 fail closed。
架构影响：重大；第 19/20 节已确认唯一 default owner 和 chat modality 边界。责任开发者确认：root，2026-07-28。

### 2026-07-28 空响应重采样与 chat/media admission 终态修正

目标与参考：Agent sampling loop 继续服从 Codex owner，并参考 `grok-build` 的 `AttemptOutcome::Empty`、
`EmptyReason::{ReasoningOnly,NoVisibleContent}` 与 content-filter 终态分类；多模型 catalog/default/switch 和
Text/Image chat 能力继续以 `grok-build` 为主参考。OpenCode 只用于 provider wire，不参与本刀控制面裁决。

完成结果与分类：`agent-runtime::provider_turn` 对 reasoning-only、完全空 `stop` 和工具后的空 final 使用独立两次
语义重采样预算；重采样不消耗 `max_turns`，复用同一 tool/hook snapshot，不把空尝试的 assistant reasoning 写回
Provider transcript，并在下一次请求前检查 Provider token budget。`content_filter` 空终态合法结束，`length/error`
空终态直接失败。上述为 `current`；reasoning-only 直接失败、工具后空终答成功和重采样占用工具回合为
`dead / deleted / forbidden-to-restore`。

模型选择统一使用 canonical `ModelTaskRequest + route_capability_gap` 验证 `chat + text input + text output + streaming`；
`model/list.isDefault` 与缺省 `thread/start` 继续消费同一 ready catalog 顺序。图片生成等专用模型仍属于 media task
catalog/admission，但不再作为 Agent chat Thread 的保留选择，也不能绕过 `thread/start` preflight。model-only 切换
继续从目标模型重建或清空 `serviceTier`。这些路径为 `current`；专用媒体模型绕过 chat admission 为
`dead / deleted / forbidden-to-restore`，无新增 `compat/deprecated`。

结构收口：按上一节退出条件，将 selection/default/reconciliation 与 picker projection 拆到
`runtime/model_providers/selection.rs`；`runtime/model_providers.rs` 从 `1273` 行降到 `429` 行，新 owner 为 `854` 行。
`agent-runtime/src/provider_turn.rs` 当前 `1887` 行，下一次 sampling-attempt 行为修改前必须拆出独立 attempt owner；
`provider_turn/tests.rs` 的大体量测试也应随 owner 拆分，不得继续堆入实现文件。

验证：`agent-runtime --lib` `189/189`；App Server model selection `11/11`；最终 session operations 定向
`16/16`，App Server related `1606/1606`。同一变更集的 App Server 全包、lime-agent、scheduler、server 分别通过
`1604/1604`、`257/257`、`24/24`、`118/118`；`npm run test:contracts` 通过 `808` generated protocol types、
`284` client checks 及 modality/scripts/docs guards。最终 `npm run governance:scripts` 通过；
`npm run governance:legacy-report` 为零引用候选 `0`、既有 deprecated 分类漂移 `1`、边界违规 `0`；workspace
rustfmt 与 `git diff --check` 通过。

收尾只读复核发现并修复一个 P1：同模型的 `collaborationMode` 更新不再被误判成模型切换并覆盖用户已选
`serviceTier`；只有最终 `(provider, model)` 真正变化时才从目标 catalog 重建或清空 tier。请求同时给出不一致的
`model` 与 `collaborationMode.settings.model` 现在 fail closed，且拒绝后不持久化候选设置。新增回归覆盖同模型
mode-only 保留 tier、collaboration 模型切换重建默认 tier 和冲突双模型字段；复核未发现其他 P0/P1、凭证泄漏或
owner 越界。

`npm run smoke:agent-runtime-current-fixture` 首轮通过 history/cache、stream terminal、Electron guards、Claw 首页、
代码工作台、图片命令、停止后继续、审批、Plan、Skills、MCP 与 media reference 等场景，最后在 Content Factory
预建 thread 处因未注册文本 fixture provider 被新的 production preflight 以 `provider_not_configured` 正确拒绝，
因此该次聚合退出码为 `1`，不得记作全量通过。修复只扩充 test-only provider 准备步骤，没有放宽 production
preflight；当前拆分后 sidecar 已定向复跑该失败场景并通过 Gate B，证据为
`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-content-factory-article-workspace-regression-summary.json`，
其中 Electron/preload/IPC/App Server/read model identity 一致，legacy command 与 mock fallback 命中均为 `0`。
受时间优先级约束未从聚合第一项再次全量复跑。架构影响：重大；更新架构第 6.3、19、20 节，责任开发者确认：
root，2026-07-28。

### 2026-07-28 xAI 视频 current 骨架收口

目标与 owner：视频生成保持 dedicated media task，不进入 Agent chat picker。catalog/capability 与异步任务语义参考
`grok-build`，具体 xAI wire/lowering 归 Lime `model-provider`；App Server 负责任务调度和 exact credential，
`media-runtime` 只负责 durable progress/artifact，不再直接发 Provider HTTP。

完成结果与分类：RuntimeCore 使用独立 `xai_video` 协议，只允许 Fal/xAI 视频 route；`model-provider` 实现 Fal
同步 POST 与 xAI `POST /videos/generations -> request_id -> GET /videos/{id}`，覆盖
done/failed/expired/timeout/cancelled，并严格使用 resolved auth header/prefix。任务持久化
`provider_task.protocol/request_id/status`；App Server scheduler 只恢复 stale、已有 request id 的 xAI running task，
恢复只 poll、不重复 POST。以上为 `current`；media-runtime 直发视频 HTTP、把 xAI 诊断标为 Fal、普通 OpenAI
视频 fallback 为 `dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。

验证：media-runtime route `9/9`；public App Server JSON-RPC xAI start/poll `1/1`、Fal 回归 `1/1`；此前同一
变更集的 model-provider xAI lowering/URL/result `4/4`、media-runtime xAI start/poll/resume `2/2`、Fal worker
`1/1`、App Server media worker `19/19` 均通过。最终 `npm run test:contracts` 通过 `808` generated protocol
types、`284` client checks 及 modality/scripts/Electron release/docs guards；workspace rustfmt、`git diff --check`
与 legacy governance 通过，治理摘要为零引用候选 `0`、边界违规 `0`、既有 deprecated 分类漂移 `1`。
架构影响：重大；更新架构第 23 节与多模型执行计划，责任开发者确认：root，2026-07-28。

### 2026-08-06 raw side-channel allowlist 与未支持能力复核

目标与范围：继续按 Codex Desktop/App Server runtime contract 复核 Lime 的通知边界；TUI UI 不作为实现目标，
多模型、多模态控制面仍以 `grok-build` 为主参考。写集限定为 v2 notification projector、事件回归、package/Hook
事实源文档与本计划；不新增协议 method、Electron 后端或第二套 runtime。

完成结果与分类：Rust `V2NotificationProjector` 不再把所有未知 `AgentEvent` 自动包装为
`agentSession/event`。raw envelope 只允许 `message.created` durable input、五个 provider trace、
`image_task.created/parameters.required/presentation.generated`（含既有 underscore alias）与 `runtime.status`；
`approval.session_cache.hit`、`provider.step` 是显式 audit-only，不发送客户端。未知前缀事件、lifecycle alias
（例如 `warning`、`turn.interrupted`）和其他未知事件在 App Server 边界直接返回 `RUNTIME_ERROR`。Rust/TypeScript
client 使用相同 exact allowlist，Renderer 继续只消费 provider/media/runtime side-channel，`message.created` 由
canonical read model 恢复而不伪造第二套 Item lifecycle。direct v2 Thread/Turn/Item 主链不变；raw side-channel
仍为 `compat/deprecated` 迁出面，不得新增 lifecycle 依赖。thread resume/reconnect 的 raw 顺序回归已改用
allowlist 内的 `provider.request.started`；`provider.step` 不再作为客户端正向 fixture。

权限 profile 复核结论：`ThreadSettings.activePermissionProfile`、`ThreadSettingsUpdateParams.permissions`、
`GrantedPermissionProfile` 和工具审批权限对象属于不同语义。当前 Lime 只有 approval/sandbox/web-search
world-state 与 per-request granted permissions；`thread/settings/update.permissions` 没有 named profile
resolver、`permissionProfile/list` producer 或 GUI producer，继续在 RuntimeCore fail closed，不把 Codex 配置表
强行复制到 App Server。若未来进入产品范围，resolver 必须归 `tool-runtime`/permission policy owner，设置持久化
仍归 App Server，不能由 provider 或 Electron 承接。

辅助模型复核结论：Grok 的 `session_summary_model`、`image_description_model`、`prompt_suggest_model_pin`
是实际 auxiliary sampler；Lime `topic`、`generation_topic`、`history_compress`、`agent_meta` 配置仍主要停留在
设置/兼容 auxiliary 观测面，`AISummaryService` 仍是未接 current provider 的 stub，主链构造也不注入它。当前不
伪造“已接入”证据，也不新增第二套调用链；后续若实现，应由 RuntimeCore/App Server auxiliary task owner 统一
接入 `model-provider` catalog/readiness/retry，并补真实 Desktop Gate B。

验证：App Server v2 notification 定向 `39/39`、Rust client event `6/6`、TypeScript client `104/104`、Codex
method scope `4/4`、render projection coverage `5/5` 通过；`cargo test --quiet --manifest-path lime-rs/Cargo.toml
-p app-server --lib` 全量 `1689/1689` 通过。`npm run test:contracts` 全通过（含 App Server client contract
`292` checks），`npm run governance:legacy-report` 为 `0` 分类漂移、`0` 边界违规，`npm run
harness:doc-freshness` 为 `0` issues；rustfmt、Prettier 与 `git diff --check` 通过。包/Hook/计划文档已同步 raw
side-channel 语义。`npm run verify:local` smart 全量通过：前端 `113/113` 批次、Rust changed scope 的 App Server
`1689/1689`、Rust client `33/33`、test client `10/10`，以及 generic Electron GUI smoke/App Server initialize 均
通过；该 generic smoke 是当前工作树的补充证据，不冒充本刀专属 Gate B 场景。本刀未改 Electron、Renderer 主链或
用户可见交互，因此未新增场景化 Gate B。架构影响：非重大，复用既有 explicit raw allowlist 和 current owner；
责任开发者：root，2026-08-06。

### 2026-08-06 Durable Ordered Thread Section current 收口

目标与参考：按 Codex HEAD `c4f42d161ae44a8d696ee9fb595709661979d187` 对齐 durable ordered Thread Section；
Desktop 只复用 Lime 紧凑侧栏分组，不复制 Codex TUI 布局。多模型、多模态控制面继续以 `grok-build` 为主参考，
本刀不触碰 provider/model/media owner。

完成结果：`threadSection/list/create/update/delete` 与 `thread/section/move` 已具备 current v2 protocol、
JSON Schema、generated TypeScript、typed client、App Server handler、SQLite section/membership/order、内置
Pinned section、冷启动恢复和公开 JSON-RPC 测试。`ThreadSectionMoveParams.sectionId` 修正为必填 nullable，
公共 schema helper 收到 v2 owner。Renderer session gateway 先读 section catalog，再按 `section_position` 逐 section
读取 `thread/list`，最后读取 `sectionId: null`；去掉最终 `updatedAt` 排序，session/topic/sidebar 均保持服务端顺序。

Desktop 侧栏删除 `lime.app-sidebar.favorite-session-ids`、favorite state 和 `isPinned` UI 投影；conversation menu
通过 current `thread/section/move` 进入或离开内置 Pinned section，Pinned/custom section 作为首组，未分组会话才进入
已有项目/独立对话分组。项目置顶、FileManager pin、插件 pinned tabs 不属于本刀删除范围。

验证与修复：Renderer/API/Sidebar/Topic 定向 Vitest `107/107`、Agent API `30/30`、client factory `10/10`、
App Server runtime boundary `25/25` 与公开 Thread Section JSON-RPC `1/1` 通过；协议生成物已重建并显示
`ThreadSectionMoveParams.sectionId: null | string`。首次真实 Electron Gate B 暴露未分组请求错误携带
`sectionId: null + sortKey: section_position`；修复后只对非空 section ID 使用 `section_position`，未分组列表显式发送
`sectionId: null` 且沿用默认顺序。`npm run smoke:agent-runtime-current-fixture` 随后完整通过，覆盖真实 Electron、
preload/IPC、`app_server_handle_json_lines`、App Server、read model、会话恢复、代码工作台、图片命令、approval、Plan、
Skills 与 typed error，production mock/legacy fallback 为 `0`。`npm run verify:local` 当前树完整通过：lint、typecheck、
前端 `113/113` 批次、`test:contracts`、Rust changed scope 和 `verify:gui-smoke` 均退出码 `0`；Electron smoke 证据为
`.lime/qc/project-gates/standalone-shell-01-20260806054730-23273/shell-01-electron-smoke/summary.json`，其中
legacy command、mock fallback、console/page/invoke error 均为 `0`。`npm run governance:legacy-report`、workspace
rustfmt 与 `git diff --check` 通过。为满足非生成文件 `800` 行治理护栏，Thread Section 分发改用短 v2 owner 引用并将
私有 handler 收敛到 `thread_sections.rs`，`dispatch.rs` 当前为 `798` 行；没有新增兼容包装层。

分类：Thread Section protocol/store/typed gateway/sidebar projection 为 `current`；旧 `isPinned`、localStorage
favorite、侧栏时间二次排序为 `dead / deleted / forbidden-to-restore`；无 `compat/deprecated`。产品范围矩阵更新为
`75 implemented / 110 planned / 35 product-scope-excluded`，完成度 `75 / 185 = 40.5%`。架构影响：重大；已更新
`internal/aiprompts/architecture.md` 第 29 节，责任开发者确认：root，2026-08-06。下一刀回到 Codex planned
method 或补 custom section 的 Desktop 管理入口，不恢复 TUI 或第二套导航事实源。

### 2026-08-06 Desktop custom Thread Section 管理入口

目标与阶段：完成上一节已经进入 current protocol/store 的 custom Thread Section Desktop 产品面。主对象是侧栏
会话分组；本切片覆盖目录管理和会话归属变更。交互遵循 Lime 紧凑侧栏，不复制 Codex TUI，不触碰
`grok-build` 所属 provider/model/media owner。

窄写集：`src/lib/api/threadSections*`、`src/components/app-sidebar/AppSidebarConversationShelf*`、
`AppSidebarConversationMenus*`、`sidebarConversationGroups*`、`useAppSidebarConversationActions*`、必要的
`AppSidebar.tsx` 接线与测试 fixture、五份 `navigation.json`、本计划。退出条件：typed gateway 只有一个 owner；空 custom
section 也按 catalog 顺序可见；Pinned 不可重命名/删除；会话可在 custom section 与未分组之间移动；无 localStorage、
mock backend 或兼容包装；五语言覆盖、定向测试、typecheck/contracts、GUI smoke 和 current fixture 通过。架构影响：
非重大，只补既有第 29 节 Desktop projection/command producer，不改变 owner 或依赖方向。

完成结果：`threadSections.ts` 是唯一 typed gateway，侧栏从 durable catalog 读取并保留空 custom section；新增、重命名、
删除 custom section 均写入 App Server JSON-RPC/SQLite，删除后成员回到未分组。会话移动统一走
`thread/section/move`，Pinned 只是内置 section 快捷入口，不再存在 favorite/localStorage 双轨。菜单支持 custom
section 选择和“不分组”，五种 locale 均有稳定文案与 tooltip。

验证结果：Thread Section、分组投影与边界定向测试 `15/15`，`AppSidebar.conversations.test.tsx` `53/53`；
`npm run typecheck`、`npm run i18n:check`（coverage `100%`，missing/extra `0`）、`npm run i18n:unused -- --check`
（unused `0`）、`npm run test:contracts`（generated protocol types `873`、App Server client checks `292`）、scoped
Prettier 与 `git diff --check` 通过。`npm run verify:gui-smoke` 通过真实 Electron/App Server 链路，证据为
`.lime/qc/project-gates/standalone-shell-01-20260806095438-66588/shell-01-electron-smoke/summary.json`；
`npm run smoke:agent-runtime-current-fixture` 完整通过，覆盖历史恢复、首页、会话、代码工作台、图片、approval、Plan、
Skills、MCP、media 与 typed error，`liveProviderUsed=false`。最终 `npm run verify:local` 退出码 `0`，包含前端
`113/113` 批次、changed Rust scope、contracts 与 GUI smoke；`npm run governance:legacy-report` 为零引用候选 `0`、
分类漂移候选 `0`、边界违规 `0`，`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check` 通过。
Thread Section 分发继续使用短 v2 owner 引用，`lime-rs/crates/app-server/src/processor/dispatch.rs` 保持 `798` 行，
未新增 compat、mock 或旧 runtime 回流入口。

分类：section protocol/store/typed gateway/sidebar projection、custom CRUD 与 session move 为 `current`；旧
favorite/localStorage、`isPinned` 投影和侧栏时间二次排序为 `dead / deleted / forbidden-to-restore`；无
`compat`、`deprecated`。完成度 `100%`，无 blocker。架构影响：非重大；复用既有第 29 节 owner 和 Electron Desktop
主链，责任开发者确认：root，2026-08-06。下一刀回到 Codex planned method 或下一项主链能力，不恢复 TUI 或第二套
导航事实源。

### 2026-08-06 Desktop canonical reasoning 展开正文修复

目标与范围：修复桌面对话时间线中“已完成思考”展开后只剩图标、没有正文的问题。Lime 继续采用 Desktop
紧凑过程时间线，不复制 Codex TUI 布局；本刀只修 Thread/Turn/Item 的 Renderer 投影与既有 Electron fixture，
不触碰 `grok-build` 所属多模型、多模态控制面，也不修改 provider、协议或 App Server owner。

根因与完成结果：App Server canonical reader 已同时保留 Reasoning Item 的 `summary[]` 与 `content[]`，其中
`text` 仍是 summary 展示镜像；Renderer 的 `resolveReasoningDisplayText` 却只读取 `summary[] + text`，导致只有
canonical `content[]` 的完成态展开区没有可渲染正文。当前实现统一以 `summary[]` 生成默认摘要，以 `content[]`
生成展开正文；仅在 `content[]` 缺失时使用 Item `text`，相同摘要与正文继续去重，raw reasoning 不拼入最终回答。
忽略 canonical `content[]` 的旧投影行为为 `dead / deleted / forbidden-to-restore`；本刀没有新增 `compat`、
`deprecated`、开关或第二套时间线。

测试与 Gate B：projection/component/fixture guard 定向 Vitest `101/101`，其中 resolver `4/4`、Reasoning
timeline `15/15`、Electron fixture guard `82/82`；`npm run typecheck`、scoped ESLint、scoped Prettier 与
`git diff --check` 通过。`npm run smoke:agent-runtime-current-fixture` 完整通过，覆盖 current Runtime、历史恢复、
真实 Electron 主路径、approval、Plan、Skills、MCP 与 media，`liveProviderUsed=false`。强化后的
`reasoning-first-visible` 专门场景继续使用真实 Electron/preload/IPC、`app_server_handle_json_lines`、App Server
read model 与 external fixture backend：先点击“已处理”历史摘要，再点击“已完成思考”，断言 canonical Reasoning
`content[]` 在展开后可见；输入框恢复、真实 terminal、identity 一致，console/page/invoke error、
legacy command 与 production mock fallback 命中均为 `0`，`liveProviderNotUsed=true`。证据：
`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-summary.json` 与
`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-chat.png`。最终
`npm run verify:gui-smoke` 通过，证据为
`.lime/qc/project-gates/standalone-shell-01-20260806110718-43649/shell-01-electron-smoke/summary.json`。

分类与完成度：canonical Reasoning summary/content 投影、桌面两级过程展开与对应回归为 `current`；忽略
`content[]` 的旧 resolver 行为已删除；无新增兼容面、无 blocker，完成度 `100%`。验证强化过程中发现并修正的
double-terminal 仅属于 test-only external backend fixture，没有放宽生产状态机。架构影响：非重大，沿用既有
App Server read model -> Renderer timeline owner；责任开发者：root，2026-08-06。

### 2026-08-06 exact `plugin/search` current 切片

目标与窄写集：按 Codex HEAD `c4f42d161ae44a8d696ee9fb595709661979d187` 对齐 exact
`plugin/search` method 与 wire，写集限定为 v2 Plugin protocol/schema、App Server Plugin processor/runtime/local
catalog owner、typed package client、Renderer Plugin API、公开 JSON-RPC 测试与产品范围矩阵。Lime 保持
Desktop GUI，不复制 Codex TUI；本切片不触碰由 `grok-build` 对齐的 provider/model/media 控制面。

完成结果：新增 `PluginSearchParams` / `PluginSearchScope` / `PluginSearchResponse` 及 Codex Plugin summary
wire，注册 exact `plugin/search`，并经 `App Server JSON-RPC -> RuntimeCore -> PluginDataSource ->
LocalAppDataSource -> plugin_catalog` 唯一 owner 执行。当前支持 `searchTerm`、`global/workspace/personal`
scope、`cwds`、`cursor` 与 `limit`；package client 暴露 `searchPlugins`，Renderer gateway 暴露
`searchPluginCatalog`。未新增 Electron 业务后端、平行 Plugin catalog、production mock fallback 或兼容包装。

验证：Codex wire round-trip `1/1`、公开 `plugin_search_jsonrpc` `1/1`、Renderer Plugin API `2/2`、
product-scope boundary `4/4`、package App Server client `106/106` 通过；`npm run test:contracts` 全部通过，
包含 generated types 无漂移与 App Server client contract `292` checks。`npm run test:rust:related -- <plugin
slice paths>` 完整退出码为 `0`，其中 `agent-runtime 192/192`、`app-server 1690/1690`、
`app-server-client 34/34`、`app-server-daemon 38/38`、`app-server-protocol 100/100`。`npm run typecheck`、
`npm run governance:legacy-report`（分类漂移 `0`、边界违规 `0`）、workspace rustfmt 与 `git diff --check`
通过。本切片没有新增 GUI 消费路径或 Electron 边界，因此未冒充场景化 Gate B 证据。

分类与进度：本切片新增 protocol、handler、runtime owner、typed client 与 gateway 均为 `current`；无
`compat`、`deprecated` 或新的 `dead` surface。方法产品范围矩阵为 `76 implemented / 109 planned /
35 product-scope-excluded`，产品范围 method 完成度 `76 / 185 = 41.1%`。其余
`plugin/list/read/install/uninstall/share/skill/read` 等 `11` 个 Plugin method 仍是 `planned`，不由本方法
冒充 Plugins 整体对齐。架构影响：非重大，复用既有 App Server current 主链与 Plugin catalog owner；
责任开发者：root，2026-08-06。下一刀优先补 Skills/Plugins/Apps watcher/readiness 或 Hook lifecycle。

### 2026-08-06 Hook lifecycle completed slice

目标与阶段：继续按 Codex HEAD `c4f42d161ae44a8d696ee9fb595709661979d187` 对齐 exact
`hooks/list`、`hook/started` 与 `hook/completed`，并把现有 `tool-runtime` Hook owner 接入真实 provider sampling、
canonical Thread/Turn/Item 与 Desktop 时间线。Lime 继续使用紧凑 GUI，不复制 Codex TUI；本切片不触碰
`grok-build` 所属 provider/model/media 控制面。完成前方法矩阵保持 `76 implemented / 109 planned / 35
product-scope-excluded`，不得用 owner 单测或协议骨架提前移动完成度。

窄写集：`lime-rs/crates/tool-runtime/src/hook_*` 与 `turn_snapshot.rs`、Agent protocol/runtime 的 Hook event 接线、
App Server v2 Hook protocol/schema/dispatch/notification projection、Rust/TypeScript typed client、Renderer Hook
gateway/notification/Item timeline、对应公共 JSON-RPC/owner/projection/GUI fixture 测试，以及本计划、架构、Hook
事实源和产品范围矩阵。共享的 protocol registry、App Server dispatch、generated types 当前包含 Thread Section 与
Plugin Search 脏改动；只追加 Hook 注册点并保留现有改动。侧栏、Thread Section、Plugin Search、Reasoning 与五语言
navigation 写集只读避让。

唯一事实源与替换规则：Hook discovery 同时服务 `hooks/list` 与 sampling runtime；用户配置只读取
`CODEX_HOME/config.toml`、项目配置只读取 `<cwd>/.codex/config.toml`，Plugin Hook 只来自本轮已激活 package root。
旧 `{hooks:{pre_tool_use:[{command:...}]}}` 解析、默认信任、`FixedHookReporter::new(None)` 空执行和 Renderer
`known_unprojected` 均为 `dead / deleted / forbidden-to-restore`，不保留兼容包装。unmanaged Hook 默认
`untrusted`，仅 trusted hash 匹配或 managed 定义可执行；`modified`/`untrusted` 只能列出并 fail closed。

退出条件：public `hooks/list` 返回 exact metadata；真实 command Hook 在 tool handler 前后执行并发出同一
`hookRunId` 的 `hook/started`、`hook/completed`；App Server 将 notification 投影为 canonical Hook Item，历史/read
model 与 Renderer timeline 可恢复；production mock/legacy fallback 命中为 `0`。验证至少包含 Hook owner 与 Agent
runtime integration、public JSON-RPC、protocol round-trip、typed clients、notification -> Item -> timeline、
`npm run test:contracts`、Rust related、`npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke`、
专用真实 Electron Hook Gate B、`npm run governance:legacy-report`、typecheck、rustfmt、Prettier 与
`git diff --check`。架构影响：重大；完成时同步 `internal/aiprompts/architecture.md` 并由责任开发者确认。

完成证据（root，2026-08-07）：Rust/TS typed client 已接入 `hooks/list`，协议 schema registry 与生成物
已同步；v2 `hook/started` / `hook/completed` 已直投影为同一 `item_<hookRunId>` 的 canonical Item；Renderer
已接入 Hook timeline row 与五语言文案。`hooks_jsonrpc` 公共 JSON-RPC exact metadata 与
`thread/start -> turn/start -> event_appender -> thread/read + thread/items/list` 历史恢复测试为 `2/2`；Hook
owner/runtime、materializer、notification/canonical reader/drift/timeline 前端定向测试为 `117/117`。
`npm run test:contracts` 完整通过，其中 App Server client contract 为 `292` checks；
`npm run smoke:agent-runtime-current-fixture` 与 `npm run verify:gui-smoke` 完整通过；typecheck、scoped rustfmt、
Prettier、`git diff --check` 与 `npm run governance:legacy-report`（`0` 分类漂移、`0` 边界违规）通过。
`npm run governance:scripts` 也通过，脚本根目录与一级领域目录均无漂移。首页热路径继续保持
`InputbarModelExtra.backgroundPreload="disabled"`，没有为了 fixture 提前触发 model read/list。

专用 `npm run smoke:hook-lifecycle-gate-b` 以真实 Electron Desktop Host 启动，命中 preload/IPC、
`app_server_handle_json_lines`、`hooks/list`、`thread/start`、`turn/start`、`thread/read` 与 RuntimeCore provider/tool
sampling；受信 `PreToolUse(update_plan)` command 子进程只执行一次，同一 run id 的 started/completed notification
配对，public read model 恢复 `type=hook`、`id=item_<hookRunId>`、`run.status=completed`，Desktop 展开的紧凑过程
行可见“自动钩子已完成”和 `Gate B command hook completed`，console/page error 均为 `0`。证据：
`.lime/qc/gui-evidence/hook-lifecycle-gate-b/hook-lifecycle-gate-b-summary.json`（`ok=true`，八项 Hook assertion
全通过）与 `hook-lifecycle-gate-b-chat.png`。该场景使用 localhost OpenAI-compatible provider fixture，不调用
live Provider，不能冒充 live 模型能力证据。

分类与进度：Hook discovery/trust、sampling gate、lifecycle event、public method、canonical Item、read model 与
Desktop timeline 均为 `current`；无 `compat` 或 `deprecated`；旧 raw Hook shape、默认信任、空 reporter、Renderer
`known_unprojected` 和 production mock/fallback 为 `dead / deleted / forbidden-to-restore`。方法矩阵更新为
`77 implemented / 108 planned / 35 product-scope-excluded`，产品范围 method 完成度 `77 / 185 = 41.6%`。
架构影响：重大；`internal/aiprompts/architecture.md` 已同步 owner/data-flow 图，责任开发者 root 已确认目录归属、
数据流、依赖方向、协议边界和验证门禁。本切片完成；下一刀转向 Skills/Plugins/Apps watcher/readiness，或按风险
优先补仍缺真实 Gate B 的 Multi-Agent/host lifecycle，不再扩展 Hook 双轨。

### 2026-08-07 Skills list/watcher completed slice

目标与阶段：按 Codex HEAD `c4f42d161ae44a8d696ee9fb595709661979d187` 对齐 exact `skills/list` 的
`cwds + forceReload -> data[{cwd,skills,errors}]` contract，并校正已经存在的 exact `skills/changed` watcher
事实。Lime Desktop 只消费 typed catalog，不复制 Codex TUI Skills 弹层；本切片不触碰 grok-build 所属
provider/model/media 控制面。

窄写集：`skills` crate snapshot 复用点、App Server v2 Skill protocol/schema/dispatch/runtime、Rust/TypeScript
typed client、Renderer runtime Skill catalog、Electron 本地 Skill 目录壳、相关 public JSON-RPC/watcher/GUI fixture
tests、method 产品矩阵、本计划与必要架构说明。共享 protocol registry/dispatch/generated types 只追加或迁移
Skill 条目，保留现有 Hook、Plugin Search、Thread Section 和并行脏改动。

唯一事实源与替换规则：`lime_skills::AgentSkillSnapshot` 继续是 discovery owner；`skills/list` 按请求 cwd
构建 workspace roots，`forceReload=true` 只失效同一 snapshot cache，不引入第二 catalog。Renderer 与 Electron
迁到 plural method 后物理删除 singular `skill/list` protocol/handler/client/fixture 正向路径，不保留 alias 或
compat。`skill/read` 继续只承担稳定 Skill id 的正文/工作流详情读取，是独立产品能力，不冒充 Codex list。
既有 `skills/changed` typed watcher、mutation invalidation 与 GUI refresh 为 `current`；畸形 payload 继续 fail
closed。`skills/config/write` 与 `skills/extraRoots/set` 本切片保持 `planned`，不得用 management API 冒充。

退出条件：public `skills/list` 保持请求 cwd 顺序，返回 exact scope/path/interface/dependencies/enabled 与逐 cwd
errors；force reload 可观察；Renderer/Host 无 `skill/list` 正向引用；真实 Electron Skills fixture 命中
`skills/list + skills/changed` 且无 production mock/legacy fallback。验证至少包含 protocol round-trip、public
JSON-RPC、watcher、typed clients、Renderer projection、`npm run test:contracts`、Rust related、
`npm run smoke:agent-runtime-current-fixture`、Skills 专项 Electron smoke、`npm run verify:gui-smoke`、治理扫描、
typecheck、rustfmt、Prettier 与 `git diff --check`。完成前方法矩阵不移动 `skills/list`；`skills/changed` 仅在 exact
notification owner 回归通过后从 planned 改为 implemented。

实施结果（root，2026-08-07）：App Server v2 已接入 exact `skills/list`，按请求顺序返回
`data[{cwd,skills,errors}]`，保留 scope/path/interface/dependencies/enabled 并支持 `forceReload` 同时失效 snapshot
与 summary cache。Rust/TypeScript typed client、Renderer Composer catalog 与稳定 `skill/read` 详情投影已迁到
plural method；禁用项在 Renderer catalog 边界过滤。singular `skill/list`、v0 `SkillListResponse`、AppDataSource
旧 list 路径与零消费者 `get_local_skills_for_app` Electron facade 已物理删除，命令契约将其固定为 retired-only，
没有 alias、compat wrapper 或 production mock fallback。

验证：`app-server-protocol` 为 `102 + 1` tests，`app-server --test skills_jsonrpc` 为 `2/2`，`lime-skills`
错误路径回归为 `1/1`，Rust client 为 `34/34`；TypeScript client 为 `106/106`，Renderer catalog/read 定向测试为
`23/23`。`npm run test:contracts` 完整通过，其中生成协议为 `915` types、App Server client contract 为 `292`
checks；`npm run typecheck`、`npm run governance:scripts`、`npm run governance:legacy-report`、定向 rustfmt、Prettier
与 `git diff --check` 通过，legacy report 为 `0` 分类漂移、`0` 边界违规。`npm run
smoke:agent-runtime-current-fixture` 完整通过；`skills-runtime` 专项 Gate B 证据位于
`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-skills-runtime-regression-summary.json`，
证明真实 Electron renderer/preload/IPC、`app_server_handle_json_lines`、首次 `skills/list`、typed
`skills/changed {}` 与第二次自动 `skills/list` 全部命中，GUI 未点击手动刷新即显示新增 Skill，legacy/mock hit、
console error、page error 均为 `0`。该 Gate B 使用受控 fixture backend，不调用 live Provider，也不作为 live
模型能力证据。`npm run verify:gui-smoke` 以 `21/21` assertions 通过，证据为
`.lime/qc/project-gates/standalone-shell-01-20260807050709-91472/shell-01-electron-smoke/summary.json`。

分类与进度：plural list/watcher、`skill/read` detail projection、cache invalidation 与 Desktop GUI catalog refresh
均为 `current`；`compat` 无；`deprecated` 无；singular `skill/list`、`SkillListResponse`、旧 AppDataSource list 与
`get_local_skills_for_app` 为 `dead / deleted / forbidden-to-restore`。方法矩阵为
`79 implemented / 106 planned / 35 product-scope-excluded`，产品范围完成度 `79 / 185 = 42.7%`。架构影响：
重大；`internal/aiprompts/architecture.md` 第 28 节已同步，责任开发者 root 已确认目录归属、数据流、依赖方向、
协议边界和验证门禁。本切片完成；下一刀继续 `skills/config/write`、`skills/extraRoots/set`，或转向
Plugins/Apps watcher/readiness，不复制 Codex TUI。

### 2026-08-07 Skills config/extra roots completed slice

目标与阶段：继续按 Codex HEAD `c4f42d161ae44a8d696ee9fb595709661979d187` 对齐 exact
`skills/config/write` 与 `skills/extraRoots/set`。Desktop 不复制 Codex TUI：用户级启停配置进入 Lime 既有 YAML
事实源，extra roots 只在 App Server 进程生命周期内生效；本切片不触碰 grok-build 所属多模型/多模态控制面。

窄写集：App Server v2 Skill protocol/schema/dispatch/runtime、`lime-skills` snapshot roots/config projection、
RuntimeCore turn metadata、Rust/TypeScript typed client、Renderer App Server gateway、公共 JSON-RPC 测试、method
产品矩阵、本计划与架构/命令事实源。没有新增 Skills 管理 UI、Electron IPC、compat wrapper 或 production mock。

实施结果：`skills/config/write` 强制 exactly-one `path/name` selector，拒绝相对 path，写入
`Config.skills.config`，返回 `effectiveEnabled` 并清理 snapshot/summary cache；`skills/list` 与 Agent turn snapshot
应用同一启停规则。`skills/extraRoots/set` 原子替换进程级 root 列表，拒绝相对路径，去重绝对路径，允许缺失目录
静默为空，清理同一 cache 并发送 exact `skills/changed {}`。协议 schema、generated TypeScript、request client、
connection client 与 Renderer typed gateway 已同步。

验证：App Server 全量 lib `1695/1695`、`skills_jsonrpc 4/4`、`app-server-protocol 103/103` 与 `lime-skills
65/65` 通过；公共 JSON-RPC 覆盖 selector 校验、持久化后有效状态、root 替换/清空/缺失目录、相对路径
`INVALID_PARAMS` 和 notification。修正 serde 默认产生的 `path: null` fixture 后，协议 round-trip 组合测试全通过。
`npm run check:protocol-types` 报告生成 `919` types 且 `0` drift；`npm run test:contracts`、
`npm run governance:legacy-report`（`0` 分类漂移、`0` 边界违规）、`npm run governance:scripts`、
`npm run typecheck`、相关 Rust/TypeScript 格式检查与 `git diff --check` 均通过；真实 Electron
`npm run smoke:agent-runtime-current-fixture` 与 `npm run verify:gui-smoke` 也已在本切片完成并保持
`liveProviderUsed=false`。

分类与进度：两个 exact method、配置投影、进程 roots、typed clients 和 gateway 均为 `current`；`compat` 无；
`deprecated` 无；未新增 `dead` surface。方法矩阵更新为 `81 implemented / 104 planned /
35 product-scope-excluded`，产品范围完成度 `81 / 185 = 43.8%`。架构影响：重大；
`internal/aiprompts/architecture.md` 第 28 节已同步，责任开发者 root 已确认目录归属、数据流、依赖方向、协议边界
和验证门禁。下一刀回到 Plugins/Apps watcher/readiness 或 provider adapter/hosted tool 闭环。

### 2026-08-07 exact MCP resource/tool completed slice

目标与阶段：继续按 Codex exact MCP contract 收敛 `mcpServer/resource/read` 与
`mcpServer/tool/call`。Desktop 不复制 TUI：Settings 只浏览工具；Workspace/Agent 只从真实 Thread 发起工具调用。
本切片不修改 grok-build 所属多模型/多模态控制面。

窄写集：App Server v2 MCP protocol/schema/dispatch/runtime、Session-owned `McpThreadRuntime`、Rust/TypeScript
typed client、Renderer MCP gateway/Workspace harness、MCP smoke/contract guard、method 产品矩阵与本计划/架构事实源。
并行工作树中的 Hooks、Skills、Thread projection 和模型路由改动未夹写。

实施结果：`mcpServer/tool/call` 强制 canonical `threadId`，由 Thread 解析 Session 后经
`ExecutionBackend -> AgentRuntimeState -> McpThreadRuntime -> Session-owned McpClientManager` 执行；缺 Thread 或
未恢复 runtime 时 fail closed。`mcpServer/resource/read` 使用 exact `contents[]`，`threadId` 可选；携带时读取同一
Session-owned runtime，不携带时走 management manager，wire 不接受 `sessionId`。Renderer Settings 的无 Thread
工具执行入口已删除，Workspace call proof 在发送时注入真实 active Thread。旧 `mcpTool/call`、
`mcpTool/callWithCaller` 与 `mcpResource/read` 已从 v0 catalog/DTO/schema、App Server handler/AppDataSource、Rust/TS
typed client、Renderer API、timeout policy、smoke 和正向测试物理删除；contract guard 禁止三者回流 current 边界。

分类与进度：两个 exact method、typed clients、Renderer gateway、Thread runtime 与 smoke 为 `current`；
`compat` 无；`deprecated` 无；三个旧 method 为 `dead / deleted / forbidden-to-restore`。Settings 无 Thread 调用入口
同属 `dead / deleted`。方法矩阵为 `83 implemented / 102 planned / 35 product-scope-excluded`，产品范围完成度
`83 / 185 = 44.9%`。架构影响：重大；`internal/aiprompts/architecture.md` 已同步 owner/data flow，责任开发者
root 已确认目录归属、依赖方向、协议边界和验证门禁。

最终验证（root，2026-08-07）：`npm run smoke:mcp-current -- --allow-write-fixture
--allow-plugin-runtime-fixture` 通过，证据
`.lime/qc/gui-evidence/mcp-current/mcp-current-summary.json` 为 `ok=true`，实际命中 `thread/start`、
`mcpServer/tool/call`、`mcpServer/resource/read` 与 `agentSession/toolInventory/read`，fixture 与 plugin runtime 的
missing method 均为空，`legacyMcpCommandsSeen=[]`，失败 MCP server 也未污染健康 server 的 tool/resource read。
fixture 的 Thread 改用 `{ephemeral:true}` 让 current model catalog 自动选择可执行 route，不再硬编码
`openai/gpt-5.4`；真实 readiness/preflight 仍保持 fail closed，未引入 test-only production fallback。

真实 Electron `npm run smoke:mcp-workspace-plugin-runtime-electron-fixture` 通过，证据
`.lime/qc/gui-evidence/mcp-workspace-plugin-runtime-fixture/mcp-workspace-plugin-runtime-fixture-summary.json`
为 `ok=true`、`backendMode=runtime`，真实命中 preload/IPC、`app_server_handle_json_lines`、`thread/start`、
inventory、server start、caller-scoped list 与 exact tool call；缺失 required method、legacy command 与 console error
均为 `0`。`npm run smoke:agent-runtime-current-fixture` 完整通过，包含 MCP structuredContent 到 Agent Chat GUI
可见的真实 Electron 回归；`npm run verify:gui-smoke` 通过，证据为
`.lime/qc/project-gates/standalone-shell-01-20260807115648-65695/shell-01-electron-smoke/summary.json`。

组合门禁：`app-server` lib `1694/1694`、`app-server-protocol` lib `104/104` 通过；扩大测试发现的唯一
schema registry/type-name 顺序漂移已把五个 exact MCP v2 类型移回 MCP owner 段并补跑专项回归。既有
`npm run test:contracts`、App Server client contract `292` checks、`npm run typecheck` 与 generated protocol
`919` types / `0` drift 保持通过；本轮追加 `npm run governance:legacy-report`（扫描 `2383` 个文件，分类漂移
`0`、边界违规 `0`）、`npm run governance:scripts`、scoped rustfmt、Prettier 与 `git diff --check` 均通过。
current profiling 文档也已迁到 `mcpServer/tool/call`，旧 wire 只剩负向守卫或历史 evidence。下一刀回到
Plugins/Apps watcher/readiness 或 provider adapter/hosted tool 闭环，不恢复旧 MCP wire。

### 2026-08-07 Apps exact catalog/readiness boundary completed slice

目标与窄写集：继续按 Codex `app-server-protocol/src/protocol/v2/apps.rs` 与 App Server Apps processor 对齐
exact `app/list`、`app/read`、`app/installed` 和 `app/list/updated`。Lime 是 Desktop GUI，不复制 Codex TUI；
多模型、多模态控制面仍由 `/Users/coso/Documents/dev/rust/grok-build` 对齐，本切片不改 provider/model/media owner。

唯一 owner 与数据流：已安装 Plugin manifest 的 `apps` capability 由既有 `PluginDataSource -> local
plugin_catalog` 投影为 Apps catalog；没有新增第二 catalog。`app/list` 支持 cursor/limit，首页读取发布
`app/list/updated`；`app/read` 最多 100 个 id、去重保序并返回 partial `missingAppIds`；`app/installed` 报告
enabled/runtime 状态。可选 `threadId` 只接受已加载 canonical Thread，未知 Thread 返回 `SESSION_NOT_FOUND`。
本地 registry 不伪造 hosted refresh；local Plugin app 没有 model-visible hosted tool snapshot 时，`callable=false`
并使 Desktop readiness 保持 fail closed。

窄写集：`lime-rs/crates/app-server-protocol/src/protocol/v2/apps.rs`、Apps protocol catalogs/envelopes/schema
registry、App Server Apps processor/runtime/Plugin catalog、public `apps_jsonrpc` 与 protocol round-trip tests、
`packages/app-server-client` generated/request/notification client、`src/lib/api/appServer{Constants,Types,
ClientMethodSpecs,ClientMethods}.ts`、新增 `src/lib/api/apps.ts` gateway 与对应 tests，以及本计划、架构、命令和
product-scope fixture/matrix。共享 Skills/Hooks/MCP/Provider 脏区只读避让。

验证结果（root，2026-08-07）：`cargo check -p app-server-protocol` 与 `cargo check -p app-server` 通过；
`cargo test -p app-server-protocol --lib` `107/107`、`cargo test -p app-server --test apps_jsonrpc` `1/1`；
Apps Renderer gateway `4/4`；`@limecloud/app-server-client` package client `108/108`；协议 schema fixture 与
generated client 已由 `write_schema_fixtures`、`npm run generate:protocol-types` 同步；root `npm run typecheck`
通过。Apps parser 对畸形 notification/response fail closed，未使用 `window` 自定义事件。

收尾门禁（root，2026-08-07）：格式化后 Apps Renderer gateway `4/4` 与 root `npm run typecheck` 复跑通过；
`npm run governance:legacy-report` 扫描 `2384` 个文件，分类漂移与边界违规均为 `0`；
`npm run governance:scripts`、`git diff --check` 通过。`npm run smoke:agent-runtime-current-fixture` 完整退出 `0`，
覆盖 current RuntimeCore/read model/真实 Electron GUI 共享主链且 `liveProviderUsed=false`；`npm run verify:gui-smoke`
以 Gate B-F `21/21` 通过，`app_server_handle_json_lines` IPC 命中 `44` 次，console/invoke/legacy/mock fallback
均为 `0`。证据为 `.lime/qc/project-gates/standalone-shell-01-20260807145349-71233/shell-01-electron-smoke/summary.json`；
其 claim boundary 是 Desktop 壳、preload/IPC、App Server 与 Workbench/Settings readiness，不包含 Apps 专项页面闭环。

分类与进度：四个 Apps exact direction、protocol/schema、App Server owner、typed clients、Renderer gateway 和
public JSON-RPC evidence 为 `current`；无 `compat`、无 `deprecated`。旧 Apps planned 分类已删除并移入
`implemented`，产品矩阵更新为 `87 implemented / 98 planned / 35 product-scope-excluded`，完成度 `87 / 185 = 47.0%`。
真实 Electron Apps-specific Gate B、hosted connector model-visible tool snapshot 与真实 `callable=true` provider
路径仍是 open refs，不能由已通过的共享 GUI smoke 冒充完成。

### 2026-08-07 Apps Desktop GUI consumer completed slice

目标与窄写集：补齐上一 Apps exact contract 切片缺失的 Desktop 产品消费与专项 Gate B。现役唯一 App Center
owner 保持 `src/components/AppPageContent.tsx -> src/features/plugin/ui/PluginCatalogPage.tsx`；不恢复旧
`PluginsPage.tsx`，不新增 Apps 页面、第二 catalog、`window` 自定义事件或 production mock fallback。写集限定为
Plugin catalog 页面/详情/ViewModel、Apps readiness section、独立 `scripts/electron/apps-catalog-gate-b*`、事件投影表、
本计划与架构第 31 节，避让并行 Plugin/MCP Gate、五语言资源和 `package.json` 脏热区。

实施结果：Plugin 详情侧栏通过 `src/lib/api/apps.ts` 读取 `app/list + app/installed`，将 manifest Apps capability
投影为 `ready / disabled / pending`。本地 Plugin 没有 hosted connector tool snapshot 时保持
`enabled=true / callable=false / pending`，不把安装或启用冒充模型 readiness。组件订阅 typed
`app/list/updated`；notification 到达后重新读取同一 Apps snapshot。Apps 只作为 Plugin 详情内 capability/readiness
状态呈现，安装、启停和卸载仍由 Plugin catalog owner 控制。详情组件从 `PluginCatalogPage.tsx` 拆出后，主页面从
`916` 行降到 `771` 行，没有继续向接近千行的组件堆业务逻辑。

定向验证：`PluginCatalogPageViewModel.unit.test.ts + PluginCatalogPage.test.tsx` 为 `6/6`，覆盖 fail-closed
projection、初始 `callable=false` 和 typed notification 后的 disabled refresh；Gate 脚本结构守卫 `1/1`；root
`npm run typecheck`、scoped Prettier、Node syntax check 与 `git diff --check` 通过。五语言未新增 key，复用既有
`plugin.catalog.v2.status.disabled`、`plugin.apps.center.host.status.ready` 与
`plugin.apps.center.host.status.planned`，避免夹写并行 i18n 文件。

跨层收尾门禁：`npm run test:contracts` 完整通过，协议生成 `934` types 且无漂移，App Server client contract
为 `292` checks；`npm run governance:legacy-report` 扫描 `2386` 个文件，分类漂移和边界违规均为 `0`；
`npm run governance:scripts` 无 root/一级领域目录漂移。`npm run smoke:agent-runtime-current-fixture` 完整通过并
保持 `liveProviderUsed=false`。`npm run verify:gui-smoke` 以 Gate B-F `21/21` 通过，真实 Electron
`app_server_handle_json_lines` IPC 命中 `43` 次，console/page/invoke/trace/legacy/mock fallback 均为 `0`；证据为
`.lime/qc/project-gates/standalone-shell-01-20260807154019-68375/shell-01-electron-smoke/summary.json`。该共享 smoke
只证明 Desktop 壳与通用 bridge readiness，Apps 页面闭环仍由下述专项 Gate B 单独证明。全仓
`git diff --check` 通过。

专项 `node scripts/electron/apps-catalog-gate-b.mjs` 以真实 Electron Desktop Host 和隔离 app data 通过。证据
`.lime/qc/project-gates/standalone-apps-catalog-20260807T152703394Z-702520/apps-catalog-gate-b/apps-catalog-gate-b-summary.json`
为 `result=pass / proofLevel=Gate B`：真实 Electron renderer/preload/IPC 命中
`app_server_handle_json_lines`；`plugin/list -> plugin/install` 发布 typed `app/list/updated`；exact `app/list`、
`app/read`、`app/installed` 返回同一 App；App Center 可见行初始为
`enabled=true / callable=false / pending`。从 GUI 点击停用后命中 `plugin/enabled/set`，trace 中
`app/list + app/installed` 均由 `2` 次增长到 `7` 次，同一可见行切换为 `enabled=false / disabled`。console、
page、invoke、trace、legacy command 与 production mock fallback 均为 `0`；两张截图无文本溢出、遮挡或错误 ready
呈现。该场景使用 `APP_SERVER_BACKEND_MODE=unavailable`，不调用 live Provider，也不证明 hosted connector
`callable=true`。

分类与进度：App Center Apps readiness consumer、typed watcher、fresh read 和专项 Gate B 为 `current`；
`compat` 无；`deprecated` 无；旧平行 Apps 页面/状态源继续为 `dead / deleted / forbidden-to-restore`。方法矩阵仍为
`87 implemented / 98 planned / 35 product-scope-excluded`，产品范围完成度 `87 / 185 = 47.0%`，因为本切片为已计入
implemented 的 Apps directions 补齐产品消费与证据，不重复计数。架构影响：重大；第 31 节已同步，责任开发者
root 已确认唯一 Plugin catalog owner、Desktop GUI 投影、typed notification 数据流和 Gate B claim boundary。
下一刀应进入 hosted connector model-visible tool snapshot / 真实 `callable=true` provider readiness，或回到方法矩阵
中尚未实现的更高优先级 current owner；不得用本地 Plugin enabled 状态替代 hosted readiness。

### 2026-08-08 Hook/MCP lifecycle notification catalog convergence slice

盘点发现四个 exact Codex notification 已有完整 current 运行链，但产品矩阵和中央 catalog 漏记其中一条：
`hook/started`、`hook/completed` 由 `tool-runtime` hook run event 经 App Server v2 projector 投影；
`mcpServer/oauthLogin/completed` 与 `mcpServer/startupStatus/updated` 由 MCP manager/App Server processor 发布，
Renderer typed event bus 和 Settings MCP 页面消费。该切片没有新增 parallel owner、TUI UI、compat wrapper 或 mock fallback。

实施结果：将 `mcpServer/startupStatus/updated` 加入 `v2::NOTIFICATION_METHODS` 中央 catalog；新增 catalog 回流测试，
确保 OAuth 与 startup 两个 MCP lifecycle method 都以 `Notification` kind 出现；重新生成全部 protocol schema fixtures 与
TypeScript generated client。方法矩阵中 `hooks-notification-planned` 与 `mcp-notification-planned` 改为 current，
统一更新 evidence、gap 与计数。

验证：`cargo test -p app-server-protocol --lib` `109/109`、`cargo test -p app-server-protocol --test schema_fixtures`
`1/1`、`cargo test -p app-server processor::tests::mcp:: --lib` `6/6`、MCP/Renderer notification Vitest `47/47`；
`npm run test:contracts`（933 generated types 无漂移、292 client checks、command/harness/modality/docs/electron release
守卫全通过）、`npm run typecheck`、`npm run governance:legacy-report`（2386 文件、0 分类漂移、0 边界违规）、
`git diff --check` 均通过。

分类与进度：四个 lifecycle notification、producer、typed client、Renderer consumer 和 Electron evidence 为 `current`；
`compat`、`deprecated` 无新增；没有删除 surface。方法矩阵更新为 `92 implemented / 93 planned / 35 product-scope-excluded`，
产品范围完成度 `92 / 185 = 49.7%`。架构影响：重大；中央 protocol catalog、schema/generated client 和产品范围事实源均已同步。
下一刀继续从剩余 P1 current owner 选择，优先 `process/*` 或 `fs/*`，仍不得用旧 `executionProcess/*` / `fileSystem/*` 同义
契约冒充 Codex parity。

### 2026-08-08 exact memory/reset completed slice

目标与范围：继续按 Codex v2 的无参数全局动作收敛 memory reset。Lime 是 Desktop GUI，不复制 Codex TUI；
多模型、多模态控制面仍按 grok-build 的 model/provider/readiness 语义，本切片不改 `model-provider` owner。

唯一主链：Renderer Settings `memoryStore.ts` -> App Server JSON-RPC `memory/reset` -> `RuntimeCore::reset_memory`
-> `MemoryAppDataSource::reset_memory` -> `LocalMemoryBackend::reset`。请求只接受 omitted、`null` 或空对象
params，非空对象 fail closed，响应恒为 `{}`；reset 只清理全局 memory root 并重建受管目录，不删除 Thread/Turn/Item、
event log、projection store 或 soul 配置。

直接替换结果：旧 `memoryStore/reset`、`MemoryStoreResetParams/Response`、v0 catalog/typed client、Renderer 设置页
调用、旧计数文案和相关正向 fixture 已物理删除；没有新增 compat wrapper。`memory/reset` 协议、schema、Rust/TypeScript
generated client、Settings gateway、公共 JSON-RPC 和持久化隔离回归已接入唯一 current owner。

验证证据：`memory_store::tests::reset_clears_store_contents_preserves_layout_and_soul_boundary` 通过；
`session_archive_jsonrpc::memory_reset_does_not_delete_persisted_session_history` 通过；`cargo check -p app-server`
通过。此前已通过 App Server protocol `108/108`、Apps/Memory Settings/API 定向 Vitest `34/34`、generated protocol
`933` types、client contract `292` checks、`npm run typecheck`、`npm run test:contracts`、
`npm run governance:legacy-report`（扫描 `2386` 文件，分类漂移 `0`、边界违规 `0`）、scoped rustfmt、Prettier
和 `git diff --check`。

分类与进度：`memory/reset`、reset backend、Settings 消费和 durable isolation evidence 为 `current`；
`memoryStore/reset`、旧 DTO、旧 catalog/fixture、旧计数展示为 `dead / deleted / forbidden-to-restore`；
`compat` 与 `deprecated` 均无新增。方法矩阵保持 `88 implemented / 97 planned / 35 product-scope-excluded`，
产品范围完成度 `88 / 185 = 47.6%`。架构影响：重大；`internal/aiprompts/architecture.md` 第 32 节和
`internal/aiprompts/commands.md` 已同步唯一 owner、边界和验证门禁。下一刀从剩余 P1 current owner 中选择，优先
`permission profile` 或 process/fs，不恢复旧 memory wire，也不以 Apps 本地 enabled 冒充 hosted readiness。

### 2026-08-08 exact process lifecycle and retired executionProcess cleanup

目标与写集：继续按 Codex v2 收敛 `process/{spawn,writeStdin,resizePty,kill}` 与
`process/{outputDelta,exited}`，Desktop 不复制 TUI。公开 handle owner 固定为
`(ConnectionId, processHandle)`；Workspace command Item 仍属于 Thread/Turn/Item projection，只通过
`thread/backgroundTerminals/*` 暴露 Thread-owned 终止能力。本切片修改 process protocol/App Server/
`tool-runtime`、typed clients、Workspace gateway/view、schema、契约守卫和事实源，避让并行 Plugin/Workflow 删除热区。

完成结果：exact `process/*` 支持 connection isolation、response-before-notification、output-before-exited、raw bytes、
omitted/null/value 三态、PTY resize、stdin close、output cap、timeout，以及 disconnect/response/notification 失败终止。
Workspace 删除旧 status refresh、drain、signal-only interrupt 和 stdin 控件，改为
`thread/backgroundTerminals/list -> command itemId -> terminate`。公开 `executionProcess/*` method/catalog/dispatcher、
v0 DTO/schema、Rust/TypeScript typed helpers、Renderer gateway 与正向测试已物理删除；内部
`ExecutionProcessServer` 继续服务 unified exec、Thread shell、background terminal 和 live registry。

内部 owner 同轮收口：`tool-runtime::execution_process::live` 只保留领域级 `LiveExecutionRequest`、output query/batch
和 gateway，不再反向借用旧 v0 Params/Response；snapshot/status/output 直接复用 `tool-runtime` current 类型，删除
App Server 的重复 DTO 映射与空 response wrapper。契约 guard 禁止旧 method、helper、schema、protocol module 和 Renderer
gateway 回流，但不禁止内部 current `ExecutionProcessServer`。

已完成验证：`cargo check -p app-server-protocol -p tool-runtime -p app-server` 通过；`tool-runtime` unified exec
`10/10`、execution process `9/9`；package typecheck 与 tests `109/109`；protocol generated type `898` 个且 check
无漂移；Renderer background terminal/API/Workbench 定向回归 `50/50`；产品范围矩阵 `4/4`，声明与 method 实算均为
`98 implemented / 87 planned / 35 product-scope-excluded`。App Server `--lib` 定向测试当前被并行删除的
`runtime/tests/evidence_exports/plugin_task.rs` 和 `runtime_backend/tests/turn_flows.rs` 编译错误阻塞；完整 client contract
script 同样先被并行删除的 `src/lib/api/plugins.ts` `ENOENT` 阻塞。本切片未恢复这些 dead Plugin 文件，也未越界修复。
`governance:legacy-report` 完整通过（扫描 `2111` 文件，零引用候选 `0`、分类漂移 `0`、边界违规 `0`）。
`thread_background_terminals_jsonrpc` 真实进程集成 `1/1` 通过。`smoke:agent-runtime-current-fixture` 中前置
unit/script guard、Claw 多场景和 Coding Workbench 专项 Electron Gate B 均通过，最终只在 Content Factory 场景读取
并行删除的 `src/features/plugin/testing/fixtures/content-factory-app.json` 时阻塞。`verify:gui-smoke` 完整通过，真实
Electron renderer/preload/IPC、`app_server_handle_json_lines`、App Server sidecar 和 Workspace shell 均 ready；证据为
`.lime/qc/project-gates/standalone-shell-01-20260808052845-10838/shell-01-electron-smoke/summary.json`。

分类与进度：exact `process/*`、connection cleanup、local supervisor、typed clients、Thread-scoped Desktop terminal
projection 和内部 process supervisor 为 `current`；`compat` 与 `deprecated` 均为空；公开 `executionProcess/*`、旧 DTO/
schema/client/Renderer/UI surface 为 `dead / deleted / forbidden-to-restore`。产品范围完成度为
`98 / 185 = 53.0%`。架构影响：重大；`internal/aiprompts/architecture.md` 第 33 节、commands 与矩阵已同步，
责任开发者 root 已确认 connection/Thread owner 分界、目录归属、数据流、依赖方向、协议顺序和删除边界。
下一刀回到剩余 P1 current owner，优先 `fs/*`、command exec/review lifecycle 或 hosted connector readiness；不得恢复
旧 `executionProcess/*`，也不得把 connection handle 与 Thread item/process id 混用。

### 2026-08-08 exact filesystem protocol and retired fileSystem cleanup

目标与写集：继续按 Codex v2 收敛 `fs/readFile`、`fs/writeFile`、`fs/createDirectory`、`fs/getMetadata`、
`fs/readDirectory`、`fs/remove`、`fs/copy`、`fs/watch`、`fs/unwatch` 与 `fs/changed`。Desktop 只保留文件浏览、
预览、导入和工作台投影，不复制 TUI；多模型、多模态仍以 grok-build 的 catalog、capability、readiness 与
sampling 为事实源，本切片不改 provider owner。写集限定为 fs protocol/App Server/typed clients、Renderer
`fileBrowser` 与文档导入消费者、Electron 导入 fixture 的 method 追踪、schema、契约守卫、架构/命令事实源和
产品范围矩阵；并行 Plugin/Workflow/OEM 热区只读避让。

唯一主链：`src/lib/api/fileBrowser.ts -> typed App Server client -> fs/* -> App Server FsServer -> fs/changed`。
路径必须为绝对路径，raw bytes 统一 base64；`readFile` 当前上限为 512 MiB。watch owner 是
`(ConnectionId, watchId)`，notification 只回到 owner connection，断连只清理本连接 watcher。Desktop rename
不新增 Codex 不存在的 method，而是组合 `getMetadata -> copy -> remove`，明确为非原子操作。exact metadata
没有 size，Renderer 目录 DTO 当前投影 `size=0`。Office/PDF 文本抽取不属于 raw-byte fs；需要时必须在独立
current 文档能力 owner 重建，不能恢复旧 preview wire。

直接替换结果：旧 `fileSystem/*`、v0 DTO/schema、App Server `processor/file.rs`、RuntimeCore file projection、
services `file_browser_service`、旧 renderer aliases 和正向 fixture 已物理删除；没有 compat wrapper。Electron
系统壳能力 `get_file_manager_locations` 与 `get_file_icon_data_url` 继续由 Desktop Host 承接，不属于 fs fallback。
契约守卫同时要求 exact fs 正向面存在，并禁止旧 method、DTO/schema、processor/runtime/service owner 与六个
renderer alias 回流。

定向与协议验证：package App Server client `109/109`；fs gateway、文档导入与产品范围矩阵 Vitest `22/22`；
Rust FsServer `4/4`；public JSON-RPC fs integration `2/2`；`cargo fmt --all --check` 通过。完整
`npm run test:contracts` 通过，generated protocol 无漂移，App Server contract 为 `301 checks`，command、
modality、scripts governance、Electron release 和 docs boundary 均通过。扩大后的 `npm run test:rust:related`
执行到 App Server `1598 passed / 1 failed`，唯一失败是并行 read-model 测试
`read_session_current_projection_summary_preserves_process_items` 仍期望 `artifact.snapshot`、实际为 `None`，
不经过 fs owner，本切片未越界修改。

GUI 与治理证据：`npm run verify:gui-smoke` 通过，真实 Electron renderer/preload/IPC、
`app_server_handle_json_lines`、App Server sidecar 和 Workspace shell ready；证据为
`.lime/qc/project-gates/standalone-shell-01-20260808111024-394/shell-01-electron-smoke/summary.json`。
`npm run smoke:codex-import-click-through-electron-fixture` 通过，真实 Electron 从侧栏 scan/preview/commit，导入
`200` 个 Item，打开历史、附件与文件 artifact 后在同一 session 继续发送；证据为
`.lime/qc/gui-evidence/codex-import-click-through-fixture/codex-import-click-through-fixture-summary.json`，trace 命中
exact `fs/readDirectory`、`fs/getMetadata`，无 console error。导入后的 Markdown/HTML/Office/PDF 预览读取
canonical `artifact/read`，因此该场景不命中 `fs/readFile`；`fs/readFile` 的 claim 由 public JSON-RPC integration
和 Renderer gateway tests 承担，不把 artifact preview 冒充 fs read Gate B。`npm run governance:legacy-report`
扫描 `2111` 个 current 文件与 `1375` 个测试文件，零引用候选、分类漂移、边界违规均为 `0`。

聚合 `npm run smoke:agent-runtime-current-fixture` 的 fs 无关前置与多项 Electron 场景通过，最终被并行
Content Factory Article fixture 阻断：GUI 只有 `artifact-delivery-checklist`，fixture 目标
`artifact-article-1` 不存在。该失败不经过 fs protocol/Renderer consumer，保留为并行产品热区阻断，不恢复已删除
Plugin/Workflow 实现。分类与进度：exact `fs/*`、FsServer、connection-scoped watcher、typed clients、Renderer
文件浏览/导入消费和回流守卫为 `current`；`compat` 与 `deprecated` 均为空；旧 `fileSystem/*`、旧 DTO/schema 和
旧 owner 为 `dead / deleted / forbidden-to-restore`。产品范围矩阵为
`108 implemented / 77 planned / 35 product-scope-excluded`，产品范围完成度 `108 / 185 = 58.4%`。架构影响：
重大；`internal/aiprompts/architecture.md` 第 34 节与 `internal/aiprompts/commands.md` 已同步，责任开发者 root
确认唯一 owner、连接生命周期、Desktop 投影、非原子 rename 和文档抽取边界。下一刀回到剩余 P1 current owner，
优先 exact `command/*` 或 review lifecycle；不得恢复旧 file wire，也不得把 artifact/document owner 并入 raw fs。

## 8. 完成定义

本计划完成不等于“所有 Codex 产品面都复制”。完成定义是：

1. Codex v2 runtime 主链只有一套 Thread/Turn/Item、ThreadStore、tool lifecycle 和 recovery owner；
2. 多模型只有一个 `model-provider` network owner，grok/OpenCode 仅作为分层参考；
3. `agentSession`/`protocol/v0`/`lime-providers` 不再是 production current surface；
4. 所有产品范围内的 transport、method、Item、MCP、Multi-Agent、environment 和 evidence contract 有实现与验证，排除项有明确删除守卫；
5. Gate A/B、workspace compile、治理扫描和回流 guard 全部通过。

### 2026-08-08 Codex exact `command/exec*` current slice

目标：将 Codex standalone `command/exec`、`command/exec/{write,resize,terminate}` 与
`command/exec/outputDelta` 接入 Lime Desktop 的 App Server JSON-RPC current 主链。该能力不是
Thread/TUI 专属：Desktop coding terminal 需要独立、connection-scoped 的 PTY/stdio 控制，但不得
继续使用 Electron `project_shell_*`、旧 v0 DTO 或轮询 drain 协议。

窄写集：`app-server-protocol` v2 command schema/envelope、App Server command processor 与
connection notification hook、typed package/Renderer gateway、Electron command forwarder、旧
Project Shell UI/bridge 删除、产品范围矩阵、架构与治理回流守卫。避让发布、provider、Plugin、OEM
与其它并行未完成写集。

退出条件：

1. exact command request/response/notification shape 与 Codex 当前协议一致；process id 只在
   originating ConnectionId 内有效，断连终止进程，output notification 在最终 response 前发出。
2. Desktop terminal 只通过 App Server JSON-RPC current gateway 工作；生产代码零引用
   `project_shell_*`、`run_project_shell_command`、旧 project shell v0 method。
3. protocol schema/generated client、App Server lifecycle、Renderer/Electron contract 和
   negative governance guard 同步，矩阵四个 client method 与 notification 从 `planned` 移入
   `implemented`。
4. 定向 Rust/TypeScript/contract/governance/GUI smoke 验证通过；若某项受环境阻断，记录具体原因。

完成结果：Codex exact `command/exec`、`command/exec/write`、`command/exec/resize`、`command/exec/terminate` 与
`command/exec/outputDelta` 已进入 v2 protocol/schema、App Server `CommandExecServer`、连接生命周期清理、typed
package client、Renderer `commandExec` gateway 和 Desktop xterm terminal。一次性 stdout/stderr、流式 raw bytes、
stdin close、PTY resize、terminate、连接隔离、重复 id、output cap 与 timeout 均有 Rust owner 测试；timeout 退出码固定
为 `124`。Renderer 仅发送 base64 stdin，按 processId 过滤 outputDelta，卸载时终止 originating connection 的进程。

直接替换结果：旧 Project Shell Rust DTO/processor/schema、Electron 私有 IPC/host、Renderer gateway、session reconnect
语义、明文 stdin 和相关正向 fixture 已物理删除；未新增 compat/deprecated wrapper。旧命令名只保留在负向回流 guard、
历史 execution plan 和 immutable evidence。

验证证据：`cargo check -p app-server-protocol`、`cargo check -p app-server`、App Server command/exec Rust 定向测试
（含 output cap/timeout）通过；`npx vitest run "src/components/agent/chat/components/TaskCenterUtilityToolbar.integration.test.tsx" "src/lib/api/commandExec.test.ts"`
为 `36/36`；`npm run typecheck` 通过。相关 `npm run test:related` 的 smart runner 曾因把 `electron/` 目录误作输入触发
Vite `EISDIR`，已改用直接文件 runner；该 runner 错误不计为产品测试失败。contracts、治理、GUI smoke 和全量 fmt/diff
仍是本刀收尾门禁。

分类与进度：exact command/exec protocol、App Server owner、connection cleanup、typed clients、Renderer gateway、
Desktop terminal 和负向 guard 为 `current`；`compat` 与 `deprecated` 均为空；旧 Project Shell surface 为
`dead / deleted / forbidden-to-restore`。产品范围矩阵由 `108 implemented / 77 planned / 35 product-scope-excluded`
更新为 `113 implemented / 72 planned / 35 product-scope-excluded`，产品范围完成度 `113 / 185 = 61.1%`。架构影响：重大；
`internal/aiprompts/architecture.md` 第 6.1、33.1 节与 `internal/aiprompts/commands.md` 已同步，责任开发者 root
确认唯一 owner、跨层数据流、ConnectionId 边界、协议顺序、删除边界和验证门禁。

下一刀回到剩余 P1 current owner，优先 review lifecycle 或 hosted connector readiness；不得恢复 Project Shell、旧
`executionProcess/*`，也不得把 connection-scoped processId 与 Thread command item identity 混用。

### 2026-08-09 Codex exact `review/start` Desktop lifecycle slice

目标：继续按 Codex review lifecycle 对齐 Lime Desktop 的 `review/start`，保留 Desktop GUI 形态，不复制 Codex TUI
detached/background review；多模型与多模态控制面仍分别归 Grok-aligned `model-provider` 和 canonical content lowering。

窄写集：RuntimeCore review admission/context、App Server review handler、canonical/read-model/v2 ThreadItem projection、
Rust processor/runtime tests，以及 architecture/commands 事实源。Electron 只作为既有 JSONL Desktop Host 转发边界；不新增
第二 runtime、review transcript store、provider fallback 或 compat wrapper。

完成结果：`review/start` 先校验真实 thread/session 和 active turn，规范化 branch/sha/title/instructions 后异步提交
turn，立即返回 v2 `inProgress`。detached delivery 在 Desktop 明确 fail closed。review boundary 以
`enteredReviewMode` / `exitedReviewMode` Extension Item 写入 canonical event log，review output 优先收集 assistant
message/item 文本，没有输出时回退稳定 hint；v2 projection 分别返回 `EnteredReviewMode` 与 `ExitedReviewMode`。
未知 Extension 保持 fail closed，不被 review-specific 缺字段错误遮蔽。

根因与修复：GUI 监听建立晚于快速 Review 终态导致结果丢失；Workspace admission 后本地 `starting` 未复位；内部
Review prompt 被错误投影为普通用户消息。`src/lib/api/review.ts` 现在在 admission 前订阅同 thread 的
`agentSession/event/<threadId>`，捕获 `turn_completed`/`turn_failed`/`turn_canceled`、快速终态、请求失败和超时并解除
监听；Workspace 两个 Review 调用点在 admission 与 terminal 都刷新 canonical read model；CodeReviewSummaryPanel 与
Canvas Changes panel admission 后复位 `starting`；RuntimeCore 通过内部 `review.input`（`visibility=agent_only`、
`source=review`）把 prompt 留在 provider history，不生成普通 `user_visible message.created`。Electron fixture 增加
Review 专用 `message.delta -> message.completed -> turn.completed` 序列和 prompt 不可见、raw v2 boundary 与 backend
`turnId` 绑定断言。

定向验证：`cargo test -p app-server processor::thread::projection::tests` `20/20`、
`cargo test -p app-server processor::tests::review` `4/4`、Review gateway `9/9`、CodeReviewSummaryPanel `20/20`、
Canvas Workbench coding `10/10`、Electron fixture guard `8/8`、聚合 Vitest `47/47`、`npm run typecheck`、
`node --check`、`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check` 均通过。最终协议/治理门禁
`npm run check:protocol-types`、`npm run test:contracts`（301 checks）、`npm run governance:legacy-report`（2112 文件，
零引用候选/分类漂移/边界违规均为 0）、`npm run governance:scripts` 均通过。`npm run smoke:agent-runtime-current-fixture`
全量通过，`liveProviderUsed=false`；`npm run verify:gui-smoke` 通过，证据为
`.lime/qc/project-gates/standalone-shell-01-20260808231556-70202/shell-01-electron-smoke/summary.json`。
专项 Review Gate B 证据为
`.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/code-artifact-workbench-electron-fixture-summary.json`：
真实 Electron/preload/IPC 命中 `app_server_handle_json_lines` 与 `review/start`，canonical session/turn identity、
`enteredReviewMode`/`exitedReviewMode`、backend terminal、GUI 审查结果和内部 prompt 隔离断言全部通过，mock fallback 为零。
早期 `npm run test:related` smart runner 将 `electron/` 目录当文件导致 `EISDIR`，已改用精确 Vitest 文件 runner；不属于
Review 用例失败。

分类与进度：review/start v2 method、RuntimeCore admission、canonical boundary、read model/v2 projection、typed clients、
Desktop GUI gateway 与 Gate B evidence 为 `current`；`compat` 与 `deprecated` 均为空；旧 detached/background/raw
review side-channel 与未消费 facade 为 `dead / deleted / forbidden-to-restore`。command/exec 旧 Project Shell surface
同样保持 `dead / deleted / forbidden-to-restore`。产品范围矩阵已同步为 `114 implemented / 71 planned / 35
product-scope-excluded`，完成度 `114 / 185 = 61.6%`。架构影响：重大；architecture 第 35 节已记录真实 Gate B
证据、进程边界、identity 绑定和无 mock fallback，责任开发者 root 已确认。

下一刀：回到剩余 P1 current owner，优先 hosted connector model-visible tool snapshot / `callable=true` provider
readiness；不得恢复 TUI detached review、旧 raw review side-channel、Project Shell 或 compat wrapper。

### 2026-08-09 existing current method classification audit

目标与窄写集：继续清点 Codex exact method 矩阵，纠正已经存在 current owner、同名 generated manifest、typed
client/projection 与真实证据，却仍混在 planned 组中的分类漂移。本刀只修改产品矩阵、架构/命令事实源和本执行记录；
不改 Plugin/Provider/Workspace 业务热区，不新增 Electron backend、TUI surface 或 compat wrapper。

核验结果：

1. `plugin/list`、`plugin/read`、`plugin/install`、`plugin/uninstall`、`plugin/installed` 已接入 App Server Plugin
   processor、RuntimeCore PluginDataSource、local `plugin_catalog`、typed package/Renderer gateway，并由真实 Electron
   `mcp-elicitation-gate-b` 覆盖完整 list/install/read/installed/uninstall 生命周期。
2. `currentTime/read` 已接入 v2 server-request manifest、App Server exact-id waiter、Electron Host clock responder、
   timeout/range/invalid-response Rust 测试和 Host drain 隔离测试；它不暴露 Renderer clock API。
3. `item/tool/call` 已接入 v2 dynamic-tool contract、RuntimeCore waiter、冻结的 Desktop `desktop.appInfo` binding、
   canonical DynamicToolCall Item/read model 和 Electron Gate B；reverse request 不泄漏到 Renderer。
4. `turn/plan/updated` 已由 RuntimeCore `update_plan` producer 生成 durable fact，经 App Server v2 projector、typed
   package notification、Renderer projection 与 current fixture 投递。

5. `item/permissions/requestApproval` 已接入 tool-runtime permission parser、App Server exact-id waiter、RuntimeCore
   grant response 和统一 PendingInteractionController；profile、cwd、environment 与 response identity 均 fail closed。
6. `warning` / `error` 已接入 durable runtime producer、App Server v2 projector、canonical read model、typed client 和
   Renderer projection；typed error 的 retry success/failure 已有真实 Electron fixture guard。
7. `item/commandExecution/terminalInteraction` 已接入 command completion producer、脱敏摘要、v2 notification、cold
   read merge 与 Renderer bounded projection。

矩阵拆组后，基础 Plugin catalog 五个方法、三个 reverse request、typed `warning`/`error`、command terminal
interaction 和 `turn/plan/updated` 从 `planned` 移入 `implemented`；Plugin share/skill-read、deprecation/Guardian、
auto-approval review、`turn/diff/updated` 和 `turn/moderationMetadata` 继续保持 planned。计数从
`114 implemented / 71 planned / 35 product-scope-excluded` 更新为
`126 implemented / 59 planned / 35 product-scope-excluded`，产品范围完成度为 `126 / 185 = 68.1%`。

验证结果：产品范围矩阵、Plugin gateway、Electron current-time/dynamic-tool Host 与 Renderer plan projection 的精确
Vitest `57/57`；Apps/Plugin Electron fixture guard 与 current agent fixture guard `86/86`；permission/error/terminal/matrix
五组回归 `137/137`。Rust App Server `current_time` `5/5`、`turn_plan` `6/6`、`dynamic_tool_server_request` `1/1`、
permission request `1/1`，tool-runtime permission `2/2`，agent-runtime permission `3/3`，runtime warning `3/3`、error
`5/5`、terminal interaction `2/2`；`app-server-protocol` 全量 `110/110`。

`item/commandExecution/terminalInteraction` 原有 producer、projector、typed client 与 Renderer projection，但
`v2::NOTIFICATION_METHODS` 中央 catalog 漏列 exact method，导致 generated manifest 未覆盖该 notification；本刀已补入
catalog，并重新生成 schema bundle 与 TypeScript protocol types。严格串行执行 `npm run generate:protocol-types`、
`npm run check:protocol-types` 后确认 generated file 无 drift；最终 `npm run test:contracts` 全通过（App Server client
`301 checks`，command、modality、scripts、Electron release、cleanup 与 docs boundary 全绿）。
`npm run governance:legacy-report` 扫描 `2112` 个 current 文件与 `1376` 个测试文件，零引用候选、分类漂移和边界违规
均为 `0`；`git diff --check` 通过。本刀没有改变 GUI、Bridge 或 Runtime 行为，因此不重复运行 `verify:gui-smoke`；
真实 Electron coverage 复用并由 tracked guard 锁定的 Apps/Plugin 与 MCP elicitation Gate B。

早期使用 Node 原生 test runner 执行三个 Vitest 文件触发 runner state 错误，已用正确的 Vitest runner 重跑并全部通过；
另一次并行运行 protocol generator/check 触发 generated file 恢复竞态，改为严格串行生成与检查后消除。两者均属于
执行入口问题，不是产品失败。

分类：上述 12 个 exact method、current owners、typed clients/projections 与现有 Gate B/fixture 为 `current`；无新增
`compat` 或 `deprecated`；旧 Plugin 私有协议、Renderer 伪造 reverse request 与生产 mock fallback 继续为
`dead / deleted / forbidden-to-restore`。架构影响：无，本刀只让矩阵与既有架构一致；architecture 第 36 节已记录
核验边界，责任开发者 root 确认。下一刀回到真正未完成的 P1 surface，优先 deprecation/Guardian、auto-approval
review 或 remaining review notification；多模型和多模态继续由
Grok-aligned `model-provider` catalog/capability/readiness/sampling owner 承接。

### 2026-08-09 `turn/diff/updated` Desktop Changes slice

目标与窄写集：把 Codex Turn 级精确代码变更事件接入 Lime Desktop 唯一主链，完成
`apply_patch -> durable turn.diff.updated -> v2 turn/diff/updated -> typed client -> canonical Turn -> Desktop Changes`
闭环。本刀只修改 `tool-runtime` apply-patch metadata 与 App Server coding-event/projector、v2 protocol/schema、typed
package client、Renderer canonical conversation projection、Changes 工作台接线、产品范围矩阵和架构/命令事实源；不修改
Electron Host 业务边界、Provider、Command Exec、Review、Plugin/MCP 热区，不恢复 Codex TUI，也不新增 compat wrapper。

实现结果：

1. `apply_patch` 为 Add/Delete/Update/Move 记录内部 Turn-scoped mutation metadata；tracker 校验连续 old/new 内容并在
   Turn 内聚合精确 unified diff，支持纯 rename 与 net-zero。未知或不连续 mutation 会 invalidate 并发送空 diff 清理旧快照；
   原始 tool item 持久化前剥离内部 metadata。
2. Runtime durable `turn.diff.updated` 经 App Server v2 projector 发出严格 Codex shape：
   `turn/diff/updated { threadId, turnId, diff }`。typed client、schema、Renderer notification parser 与 sequence gate 已同步，
   额外字段 fail closed。
3. Renderer 将通知归并为 canonical conversation `Turn.unified_diff`；后续 `turn_started`/`turn_completed` 快照缺少 diff
   时保留已有值，空字符串保留为精确 net-zero。runtime handler 只回写 canonical Turn，不创建第二份 diff store。
4. Desktop Changes 从当前 canonical Turn 读取 `turnDiff`。previous-conversation 复制 `git apply` 时只要该字段已定义就
   优先使用精确 diff，即使为空也不回退到组件根据 item 拼装的 patch；Git branch/commit/unstaged 基准继续读取 Git backend。

事实源分类：`tool-runtime` coding tracker、App Server durable/projector、v2 protocol/schema、typed client、canonical Turn
projection 与 Desktop Changes 为 `current`；旧组件级 patch 拼装仅作为无 canonical diff 的历史 fixture fallback，不能覆盖已
定义的空 diff；不存在 compat/deprecated 新路径。Codex TUI review/diff UI、Renderer 第二 diff store 和生产 mock fallback
属于 `dead / deleted / forbidden-to-restore`。多模型、多模态控制面仍由 Grok-aligned `model-provider` owner 承接。

矩阵同步：将 `turn/diff/updated` 从 `review-notification-planned` 拆出为 `turn-diff-notification-current`，计数更新为
`127 implemented / 58 planned / 35 product-scope-excluded`，产品范围完成度 `127 / 185 = 68.6%`；`turn/moderationMetadata`
仍为 planned。

验证与退出条件：

- Rust 定向：`tool-runtime` apply-patch 回归、App Server coding tracker/projector/notification tests 通过。
- Protocol/schema：turn-diff round-trip、schema registry、`npm run generate:protocol-types`、`npm run check:protocol-types` 通过。
- Typed/Renderer：app-server-client 85/85；`npm run typecheck`；V2 notification/drift/conversation projection 与工作台 view-model
  回归通过；补 previous-conversation 精确 diff 优先级测试。
- 本轮收尾门禁：`npm run test:contracts`、`npm run test:rust:related -- <turn-diff paths>`、
  `npm run smoke:agent-runtime-current-fixture`、`npm run governance:legacy-report`、`npm run verify:gui-smoke`；若 GUI
  fixture 环境阻塞，记录具体原因，不以浏览器投影替代 Gate B。

下一刀：回到 remaining P1 review surface，优先 `turn/moderationMetadata` 或 deprecation/Guardian/auto-approval review
notifications；不得恢复 TUI detached review、旧 raw side-channel 或 provider 平行 owner。

### 2026-08-09 `turn/moderationMetadata` canonical Turn slice

主目标与窄写集：把 Codex trusted first-party Responses moderation metadata 接入 Lime Desktop 唯一 Agent 主链，完成
`Responses metadata -> model-provider canonical event -> agent-runtime -> durable event -> App Server exact notification ->
typed client -> canonical Turn`。本刀只修改 provider/runtime/agent 事件、App Server v2 protocol/projector/schema、typed
package client、Renderer canonical projection、产品矩阵和架构/命令事实源；不新增 Electron IPC、TUI UI、raw metadata
展示、provider 平行 owner、mock fallback 或 compat wrapper。

实现结果：

1. `model-provider` 仅在 trusted first-party Responses route 读取
   `response.metadata.openai_chatgpt_moderation_metadata`。SSE 与 WebSocket 复用同一 reducer；第三方兼容 route 不产生
   moderation event。metadata 保持任意 JSON，包括 object、array、scalar 和 `null`。
2. `CanonicalLlmEvent::TurnModerationMetadata` 经 `CurrentProviderTurnEvent`、`AgentEvent` 映射为 durable
   `turn.moderation_metadata`。该事件跨 sampling step 不去重，每次更新均保留；provider proxy 的 OpenAI/Anthropic 输出
   转换器明确忽略它，不泄漏 raw metadata。
3. App Server 投影 exact
   `turn/moderationMetadata { threadId, turnId, metadata }`；缺少 identity/metadata fail closed，wrapper 额外字段被拒绝，
   `null` 是有效值。typed client signal router、direct notification routing、sequence gate 和 drift catalog 已同步。
4. Renderer 将 opaque metadata 归并到 canonical `Turn.moderation_metadata`，按 last-write-wins 更新。后续未携带该字段的
   Turn snapshot 不覆盖既有值，cold/hydrate reader 使用同一字段；没有新增用户可见 raw JSON surface。

事实源分类：trusted Responses lowering、provider-neutral event、Agent durable event、App Server exact protocol/projector、
typed client 与 canonical Turn projection 为 `current`；`compat`、`deprecated` 均为空；第三方 metadata 冒充、TUI
surface、Electron 第二业务后端、raw side-channel 与生产 mock fallback 为 `dead / forbidden-to-restore`。Grok-aligned
`model-provider` 继续拥有多模型 catalog/default/model switch/capability/readiness/retry/circuit breaker 与多模态 sampling。

矩阵同步：`turn/moderationMetadata` 从 planned 移入 implemented，更新为
`128 implemented / 57 planned / 35 product-scope-excluded`，产品范围完成度 `128 / 185 = 69.2%`。整体 Codex 对齐仍未
完成，本刀不关闭总执行计划。

验证结果与退出条件：

- Rust：`model-provider` reducer `7/7`、`agent-runtime` metadata sampling `1/1`、`lime-agent` serde `1/1`、App Server
  moderation projector `2/2`、`app-server-protocol` `112/112` 通过；`cargo check -p lime-server` 通过。
- Protocol/schema：schema 先生成到临时目录，与仓库 schema 树逐文件一致；`npm run generate:protocol-types` 与
  `npm run typecheck` 通过。
- Typed/Renderer：app-server-client `113/113`，moderation notification/drift/timeline 三文件 `47/47` 通过。
- 执行入口记录：Node 原生 test runner 无法运行 Vitest 文件，改用 package 正式入口后通过；`test:related` smart runner
  因把 `electron/` 目录当文件读取而报 `EISDIR`，改用精确 Vitest 文件 runner 后通过，均非产品断言失败。
- 收尾门禁已完成：`npm run test:contracts` 全绿（App Server client `301 checks`）、相关 Rust related layer 全绿（含
  `lime-server` 的 provider metadata non-leak 分支）、`npm run smoke:agent-runtime-current-fixture` 通过且
  `liveProviderUsed=false`、`npm run governance:legacy-report` 扫描 `2112`/`1376` 文件并保持零引用候选/分类漂移/边界违规、
  `npm run verify:gui-smoke` 通过并生成真实 Electron/App Server evidence。矩阵守卫 `4/4` 与 `git diff --check` 也通过；GUI
  本刀无新增可见 surface，Gate B 复用现有 Electron/App Server current 主链。

架构影响：重大；`internal/aiprompts/architecture.md` 第 37 节已记录跨层数据流与边界。Responsible developer
confirmation: root, 2026-08-09. 已确认 first-party trust、SSE/WS、opaque JSON、无去重、last-write-wins、Desktop/TUI
分界、Electron 无新增 IPC，以及 Grok-aligned 多模型/多模态 owner 不变。

下一刀：继续 remaining planned method，优先 deprecation/Guardian/auto-approval review notification；不得恢复旧 raw
metadata 通道、TUI review UI、Provider 平行 owner 或兼容包装。

### 2026-08-09 `deprecationNotice` Desktop 产品范围清退

本轮先处理下一项候选中的分类漂移，而不是为没有 runtime producer 的诊断通知造协议壳。Codex 的
`deprecationNotice` 是开发/设置诊断；V2 投影事实源已经将它标为 `product-scope-excluded`，Lime Desktop
没有对话或全局通知消费者，也没有外部兼容负担。`guardianWarning` 仍因缺少真实 Guardian review producer
保持 `planned`，`item/autoApprovalReview/*` 也不借现有用户审批或 `strictAutoReview` 标记冒充。

改动：将 V1 fixture 的 diagnostics planned 组拆为 `guardian-notification-planned` 与
`deprecation-notification-excluded`，同步 `inventory.byStatus`、产品范围矩阵和架构事实源。当前分类为
`128 implemented / 56 planned / 36 product-scope-excluded`，产品范围完成度仍按
`128 / (128 + 56) = 69.6%` 计算；总上游 inventory 仍为 220 个方向化 identity。无新增 `current`、`compat`
或 `deprecated` 路径；deprecation surface 为 `product-scope-excluded`，现有旧实现若出现则按
`dead / deleted / forbidden-to-restore` 处理。

验证退出条件：V1 method scope boundary、`npm run test:contracts`、`npm run governance:legacy-report`、
`git diff --check` 通过；本轮未新增运行时/GUI 行为，不重复执行 Electron Gate B。下一刀回到有真实
producer 的 current owner，优先先完成 Guardian review producer/lifecycle，再接
`item/autoApprovalReview/{started,completed}` 的 exact Codex wire；不得恢复 TUI detached review 或兼容包装。

### 2026-08-09 V2 投影事实源漂移复核

本轮只收文档事实源，不新增协议或 compat surface。`internal/refactor/v2/EVENT-PROJECTIONS.md` 已同步
current owner：`hook/started` 与 `hook/completed` 是 Tool runtime 的 paired transient lifecycle，不创建
canonical ThreadItem；`turn/diff/updated` 是 Lime exact Turn diff，经 canonical Turn/Changes 共用快照；
`turn/moderationMetadata` 已由 trusted first-party Responses metadata 完成 `model-provider -> AgentEvent ->
durable event -> v2 notification -> typed client -> canonical Turn` 主链。`guardianWarning` 与
`item/autoApprovalReview/{started,completed}` 仍为 `planned`：当前没有 Guardian 第二模型 reviewer、风险决策、
取消/超时状态或真实 producer，现有用户审批和 `strictAutoReview` 不得冒充。

分类：文档修正为 `current` 的 Hook/Turn diff/moderation 继续由既有 owner 承接；无新增 `compat`、
`deprecated` 或 `dead` surface。验证：`git diff --check`；后续涉及实现时必须回到 tool-runtime +
model-provider + agent-runtime + App Server 这条唯一 Guardian 主链，并补 Rust owner 集成、typed protocol/client、
Renderer pending/timeline 与 Electron Gate B 证据后才能更新矩阵状态。

### 2026-08-09 Guardian auto-approval review current owner 收口

本轮完成上一条计划中的 Guardian auto-approval review 主链，产品目标仍是 Lime Desktop；Codex TUI 的 detached/background
review、raw side-channel 和第二套 Electron 业务后端不进入产品。唯一事实源为：

`strictAutoReview -> agent-runtime Guardian reviewer -> current session model-provider -> AgentEvent -> App Server v2 notification -> typed client -> Renderer ConversationProjection`

实现结果：

1. `agent-runtime` 新增真实 Guardian reviewer，复用当前 session 的 `model-provider` 做无工具结构化采样；provider 不可用、取消、30 秒超时、非法 JSON 和不确定结果均 fail closed。结果只允许 `approved`/`denied`，并带风险、授权、rationale 和 action 摘要。
2. durable `guardian_review_started/completed` AgentEvent 经 App Server v2 projector 投影为 exact
   `item/autoApprovalReview/started` 与 `item/autoApprovalReview/completed`；typed protocol/schema、manifest、generated
   TypeScript、strict decoder、lifecycle union、drift registry 和 sequence gate 同步更新。
3. Renderer `ConversationProjection` 将 started 投影为 `pending_interactions`，approved/denied/timedOut/aborted
   分别投影为 resolved/declined/cancelled；completed `inProgress` 被 App Server projector 拒绝。可选 Guardian 字段现在
   对“字段存在但类型非法”与“字段缺失”严格区分，前者 fail closed。
4. V1 方法矩阵将两个 `item/autoApprovalReview/*` 从 `planned` 移入 `implemented`，当前统计为
   `130 implemented / 54 planned / 36 product-scope-excluded`，产品范围完成度 `130 / 184 = 70.7%`。`guardianWarning`
   仍为 `planned`，因为没有独立的高优先级 warning producer，不能由 Guardian review lifecycle 冒充。

分类：Guardian reviewer、AgentEvent、App Server v2 projector、typed client、Renderer pending/timeline projection 为
`current`；无 `compat` 或新增 `deprecated`；TUI detached review、raw side-channel、生产 mock fallback 和旧审批冒充
Guardian 均为 `dead / deleted / forbidden-to-restore`。Grok-aligned `model-provider` 继续拥有多模型 catalog/default/model
switch/capability/readiness/retry/circuit breaker 与多模态 sampling，不复制 Codex TUI 控制面。

本轮退出条件已全部通过：`npm run check:protocol-types`、app-server-client build、`npm run typecheck`、Guardian/Rust/Renderer
定向回归、`npm run test:rust:related -- lime-rs/crates/app-server-protocol lime-rs/crates/app-server lime-rs/crates/agent lime-rs/crates/agent-runtime lime-rs/crates/model-provider lime-rs/crates/tool-runtime`、
`npm run test:contracts`（301 checks）、`npm run governance:legacy-report`（0 引用候选/分类漂移/边界违规）、
`git diff --check`、`npm run smoke:agent-runtime-current-fixture`（`liveProviderUsed=false`）与
`npm run verify:gui-smoke`（standalone shell evidence：`.lime/qc/project-gates/standalone-shell-01-20260809110637-94994/shell-01-electron-smoke/summary.json`）。
矩阵守卫最初拦截了 manifest 漏列 `item/autoApprovalReview/completed`；已补入 v2 `NOTIFICATION_METHODS` 中央 catalog，
重新生成 manifest/schema/generated TypeScript 后矩阵守卫 `4/4` 与协议 `112/112` 通过。

下一刀：回到 remaining planned producer/consumer，优先补 `guardianWarning` 的独立真实 producer 或其他 P1 current owner；
不得恢复 TUI detached review、旧 raw side-channel、生产 mock fallback 或 compat wrapper。

### 2026-08-09 `guardianWarning` denial circuit breaker current owner 收口

主目标与窄写集：在 Lime Desktop 唯一 Agent 主链中补齐 Codex `guardianWarning` 的真实 producer 和消费链；不复制 Codex
TUI detached review UI，不新增 Electron IPC、第二业务后端、provider 平行 owner 或 compat wrapper。唯一数据流为：

`strictAutoReview denial -> AgentRuntimeState circuit breaker -> AgentEvent guardian_warning -> durable guardian.warning -> App Server v2 guardianWarning -> typed client signal -> Renderer NoticeProjection`

实现结果：

1. `lime-agent` 为每个 session/turn 维护最近 5 次 denial 窗口和连续拒绝计数；同一 turn 连续 3 次拒绝只产生一次
   `AgentEvent::GuardianWarning`，取消当前 turn。provider unavailable denial 也计入；approved 和关闭 session 清理状态。
2. App Server durable mapper 将 `guardian_warning` lower 到 `guardian.warning`；v2 projector 严格要求非空 thread/message，
   输出独立 `guardianWarning { threadId, message }`，不降级为普通 `warning`。
3. protocol、Schema manifest/bundle、generated TypeScript、typed client strict decoder、signal union、Renderer direct
   route/sequence gate/drift registry 和 ConversationProjection notice 均已同步。Renderer 使用 `type: guardian_warning`、
   `code: guardian_warning` 的 warning notice，保留高优先级语义。
4. V1 产品矩阵将 `guardianWarning` 从 planned 移入 implemented，当前统计为 `131 implemented / 53 planned /
36 product-scope-excluded`，产品范围完成度 `131 / 184 = 71.2%`。

事实源分类：circuit breaker、AgentEvent、durable mapper、App Server v2 projector、typed client、Renderer notice 为
`current`；无 `compat` 或新增 `deprecated`；raw side-channel、普通 warning 冒充、TUI detached UI 和生产 mock fallback
为 `dead / deleted / forbidden-to-restore`。Grok-aligned `model-provider` 继续拥有多模型 catalog/default/model switch/
capability/readiness/retry/circuit breaker 与多模态 sampling/media lowering；Guardian 只复用当前 session provider。

验证结果：`cargo test -p lime-agent runtime_state`（18/18）、`cargo test -p app-server-protocol --lib`（113/113）、
`cargo test -p app-server runtime_backend::tool_events`（20/20）、`cargo test -p app-server processor::v2_notifications`
（49/49）、Schema writer、`npm run generate:protocol-types`、`npm run check:protocol-types`、app-server-client build，以及
Renderer V2/drift Vitest（49/49）均通过。收尾门禁也已完成：`npm run test:contracts`（301 checks）、`npm run typecheck`、
`npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke`、`npm run governance:legacy-report`（零分类漂移/边界违规）、
产品矩阵守卫（4/4）、`npm run verify:local` 与 `git diff --check` 均通过。下一步回到 remaining planned producer/consumer，
只选择具备唯一 owner、真实 Desktop consumer、恢复语义和 Gate B 证据的切片；不得为提升完成度制造协议 facade、compat 或生产 mock。

### 2026-08-09 `config/mcpServer/reload` Desktop 产品范围裁决

本轮继续 remaining planned producer/consumer 审计，选择 Codex `config/mcpServer/reload` 复核其是否应进入 Lime current
主链。Codex 语义是外部编辑 `config.toml` 后从磁盘重载 MCP registry，并在 loaded thread 下一次 active turn 刷新；Lime
Desktop 没有该配置入口。MCP 配置唯一事实源已经是
`Settings GUI -> App Server JSON-RPC mcpServer/list|create|update|delete -> RuntimeCore repository`，显式运行状态走
`mcpServer/start|stop` 与 typed startup notification。已有 Settings MCP lifecycle Gate B 证明 create/update/cold-read/delete，
且没有 legacy/mock fallback。

产品裁决：将 `config/mcpServer/reload` 从 `config-planned` 拆出并移入 `product-scope-excluded`。新增 exact method 会为同一
MCP 配置建立第二份 external-file owner，违反 Desktop 唯一事实源；不新增 protocol facade、Electron IPC、compat wrapper
或生产 mock。其余 `config/batchWrite`、`config/read`、`config/value/write` 与 `configRequirements/read` 仍保持 `planned`，
不能由 Lime MCP CRUD 冒充完成。

分类更新为 `131 implemented / 52 planned / 37 product-scope-excluded`，产品范围完成度
`131 / 183 = 71.6%`；总 inventory 仍为 220。`current` 是现有 App Server MCP CRUD/lifecycle，`compat` 与
`deprecated` 均无新增；Codex external-config reload 在 Lime 产品面为 `product-scope-excluded / forbidden-to-restore`。
本刀不改变架构 owner，不需要更新架构图；下一刀继续审计 remaining planned，优先选择具备真实 Desktop consumer 的
config 或 Plugin skill-read 切片。

### 2026-08-09 `plugin/skill/read` 远端预览产品范围裁决

本轮对照 Codex exact contract 与实现确认：`plugin/skill/read` 强制接收 `remoteMarketplaceName`、`remotePluginId`、
`skillName`，通过 remote plugin service 读取未安装插件的 Skill markdown；它不是本地 Skill body read。Lime Desktop 的
本地能力已经沿 `Skills catalog -> App Server skill/read -> local SKILL.md -> GUI/Agent runtime` 主链消费，且严格使用稳定
Skill identity 与 locator。把该路径包装成 `plugin/skill/read` 会把本地 catalog 冒充远端 marketplace backend。

产品裁决：从 `plugins-share-planned` 中拆出 `plugin/skill/read` 并移入 `product-scope-excluded`；不新增远端服务 facade、
alias、compat wrapper 或生产 mock。五个 `plugin/share/*` 方法仍保持 `planned`，本刀不顺带改变其未来产品裁决。

分类更新为 `131 implemented / 51 planned / 38 product-scope-excluded`，产品范围完成度
`131 / 182 = 72.0%`；总 inventory 仍为 220。Lime 本地 Skills catalog、`skill/read` 和按需 Skill body load 为
`current`；无新增 `compat`/`deprecated`；Codex remote uninstalled Plugin Skill preview 在 Lime 产品面为
`product-scope-excluded / forbidden-to-restore`。本刀不改变架构 owner；下一刀继续检查 remaining planned 中是否存在
可由真实 Desktop consumer 承接的 config capability。

### 2026-08-09 `collaborationMode/list` Desktop preset current owner 收口

主目标与窄写集：把已有 typed `CollaborationMode`、Plan tool gate 和 Composer Plan 开关收敛到 Codex exact
`collaborationMode/list`，但不复制 Codex TUI picker，不把 Grok-aligned model catalog 搬进 Agent owner。写集限定为 v2
collaboration mode protocol/schema/catalog、App Server handler/public JSON-RPC、package typed client、Renderer gateway/submit
builder、命令事实源、产品矩阵与定向测试。

唯一主链：`Desktop Plan intent -> src/lib/api/collaborationModes.ts -> app_server_handle_json_lines -> App Server
collaborationMode/list -> unique Plan mask -> typed turn/start collaborationMode -> RuntimeCore`。App Server 按 Codex 顺序返回
Plan/Default；Plan 的 `reasoning_effort=medium` 同时覆盖 collaboration settings 与 top-level Turn effort，`model=null` 继续使用
当前 Grok-aligned provider/model route。catalog 缺失、重复或非法时 Renderer fail closed，不回退本地 preset 或 production mock。

实现分类：protocol、handler、typed client、Renderer gateway 与 submit consumer 为 `current`；旧 Renderer
`buildCollaborationMode("plan", currentEffort)` 本地 preset 语义为 `dead / deleted / forbidden-to-restore`；无 compat/deprecated。
该切片不改变既有依赖方向和 model owner，因此不属于重大架构变更，不修改架构图。产品矩阵更新为
`132 implemented / 50 planned / 38 product-scope-excluded`，产品范围完成度 `132 / 182 = 72.5%`，总 inventory 仍为 220。

### 2026-08-09 `collaborationMode/list` Desktop Gate B 真实链路验收

本轮补齐上一节尚未完成的真实桌面证据。现有 `plan` Electron fixture 已直接复用 Composer Plan 开关和真实
`Electron preload -> app_server_handle_json_lines -> App Server JSON-RPC -> RuntimeCore -> GUI/read model` 主链；没有复制
Codex TUI picker，也没有引入第二套 Electron 业务后端或 Grok model catalog。Gate B evidence 现在额外固化：

`collaborationMode/list` 成功请求先于 `turn/start`，Desktop wire 使用 `mode=plan`、当前 `fixture-model`、
`settings.reasoning_effort=medium`、顶层 `effort=medium`，App Server lowering 后 runtime request 保留同一模型和
`reasoningEffort=medium`，wire/runtime mask 完全一致；Plan completed/read model/history hydrate 和 GUI decision drawer
仍然通过。

证据文件：

- 专项 Plan Gate B：`.lime/qc/gui-evidence/collaboration-mode-plan-gate-b/collaboration-mode-plan-current-summary.json`
- 专项 backend ledger：`.lime/qc/gui-evidence/collaboration-mode-plan-gate-b/collaboration-mode-plan-current-backend-ledger.json`
- 聚合 Agent fixture 的 Plan history hydrate：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-plan-history-hydrate-regression-summary.json`
- 独立 GUI shell Gate B：`.lime/qc/project-gates/standalone-shell-01-20260809154918-95017/shell-01-electron-smoke/summary.json`

实现补充：Gate B execution evidence 解析成功 Electron IPC trace 的 JSON-RPC 顺序，并对 wire/runtime collaboration mask
做严格一致性比较；backend ledger evidence 只保留脱敏后的 model、effort 和 collaboration mode 字段。Plan 场景断言
对 catalog-before-turn、medium effort、模型保持和 fail-closed runtime lowering 全部设为必需项。

验证结果与退出条件：

- `npm run smoke:agent-runtime-current-fixture` 通过，覆盖聚合中的 Plan revisioned thread item + history hydrate，且
  `liveProviderUsed=false`；专项 Plan Electron fixture 同样通过，`planPresetCatalogRequested`、
  `planPresetResolvedBeforeTurnStart`、`planPresetAppliedOnDesktopWire`、`planPresetReachedRuntime`、
  `planPresetWireRuntimeConsistent` 全为 `true`。
- `npm run verify:gui-smoke` 通过，真实 Electron/App Server shell evidence `result=pass`，无 renderer/page error、无
  legacy command、无 mock fallback。
- `npm run governance:legacy-report` 通过：扫描 `2113` 个源码文件、`1377` 个测试文件，零引用候选、零分类漂移、零
  边界违规。
- 定向 Gate B/fixture Vitest `88/88`、Prettier check、`git diff --check` 全通过；既有协议、client、typecheck、contracts
  与 Rust related 验证沿用上一节结果。

事实源分类：`collaborationMode/list` handler、protocol/schema、typed client、Renderer resolver、Gate B evidence 与
Plan fixture 为 `current`；旧本地 `buildCollaborationMode` preset 语义为 `dead / deleted / forbidden-to-restore`；无新增
`compat` 或 `deprecated`。Codex TUI picker、Electron 第二后端、provider/model 平行 catalog、生产 mock fallback 均不在
产品路径中。产品矩阵保持 `132 implemented / 50 planned / 38 product-scope-excluded`，完成度 `72.5%`。

本轮退出条件已满足，执行计划第 4 步可标记完成。下一刀继续检查剩余 planned 中具备真实 Desktop consumer 的唯一
owner；保持 Codex runtime 对齐、Grok model/multimodal owner、Desktop/TUI 分界和无兼容策略，不恢复旧入口。

### 2026-08-10 `experimentalFeature/*` Desktop config owner 收口

主目标与窄写集：将 Codex exact `experimentalFeature/list` 与
`experimentalFeature/enablement/set` 接入 Lime Desktop Settings 的唯一 App Server 主链；不复制 Codex TUI，不把
experimental catalog 扩成 Grok model catalog，不新增 Electron 业务后端或 compat wrapper。

唯一数据流：

`Settings Experimental -> src/lib/api/experimentalFeatures.ts -> AppServerClient -> app_server_handle_json_lines -> App Server experimentalFeature/* -> lime_core config.yaml`

实现结果：

1. v2 protocol、schema manifest/bundle、generated TypeScript、typed client 与 public JSON-RPC handler 已同步。catalog
   当前只包含真实 Desktop consumer `webmcp`，stage 为 `underDevelopment`，默认关闭；list 支持 cursor/limit。
2. enablement 只接受 `webmcp`，未知 key 忽略，空 map 为 no-op，并通过 `lime_core config.yaml` 持久化 Settings 选择。
   携带 `threadId` 时要求命中已加载 Thread；Lime 没有 Codex project-local feature config，不建立第二份 Thread store。
3. Settings Experimental 已迁移到 typed App Server gateway。旧 Electron `get_experimental_config` /
   `save_experimental_config`、IPC 白名单、正向 host contract、Renderer 直连和生产 mock handler 已删除；旧字符串只保留
   在负向 retired guard。
4. 产品矩阵将两个 method 从 planned 移入 implemented，统计更新为
   `134 implemented / 48 planned / 38 product-scope-excluded`，产品范围完成度 `134 / 182 = 73.6%`。

事实源分类：App Server protocol/handler、`lime_core config.yaml`、typed package client、Renderer Settings gateway 与
experimental catalog 为 `current`；旧 Electron/Tauri 配置业务入口、Renderer IPC 直连和生产 mock fallback 为
`dead / deleted / forbidden-to-restore`；无新增 `compat` 或 `deprecated`。多模型、多模态 sampling/media lowering 与
provider catalog/readiness/retry/circuit breaker 仍由 Grok-aligned `model-provider` 承接。

验证结果：App Server protocol `115/115`、public JSON-RPC experimental integration `1/1`、Settings/API/matrix
Vitest `15/15`、`npm run typecheck`、`npm run test:contracts`（App Server client `301` checks）、
`npm run governance:legacy-report`（扫描 `2113` 个源码文件、`1377` 个测试文件，零分类漂移/边界违规）、Prettier 与
`git diff --check` 均通过。首次 integration build 下载 `sherpa-onnx` 依赖耗时较长，但缓存完成后明确重跑通过，不再是
阻塞。`npm run verify:gui-smoke` 通过真实 Electron/preload/IPC/App Server shell，证据：
`.lime/qc/project-gates/standalone-shell-01-20260809165926-60736/shell-01-electron-smoke/summary.json`。本切片未新增
实验设置专项 Gate B 场景；Settings 页面行为由 component test 覆盖。

统一本地门禁的实际状态：`npm run verify:local` 在前端全量第 55/120 批由 Item inventory 漂移拦截，暴露
`item/autoApprovalReview/{started,completed}` 已实现但 fixture 仍标为 `gap`。修正为 `current` 并让守卫输出具体 method 后，
`npm run test:resume` 从第 55 批续跑至第 120 批；`.lime/test/vitest-smart-last-run.json` 最终记录 `status=passed`、
`120/120` 批通过。`local-ci.mjs` 没有跨阶段续跑能力，直接重启会无差别重跑全部前端批次，因此没有把多个独立成功
误报为“同一次 wrapper 通过”；原 wrapper 尚未触达的门禁已按其 task planner 等价补齐：`npm run test:contracts`、
`npm run test:rust:changed` 与 `npm run verify:gui-smoke` 均通过。changed-scope Rust 覆盖 19 个相关/反向依赖 crate，包含
`app-server` `1629/1629`、`app-server-protocol` `115/115`，命令退出码为 0。至此该切片达到本地可交付门槛，但总 Codex/Grok
对齐计划仍未完成；下一刀继续从 48 个 planned method 中选择具备真实 Desktop consumer 和唯一 current owner 的缺口。

### 2026-08-10 `permissionProfile/list` Desktop Turn permission owner 收口

主目标与窄写集：把 Codex exact `permissionProfile/list` 接入 Lime Desktop 已有只读、按需确认、完全访问选择器，并让
每次新 Turn 通过 App Server catalog 解析权限；不复制 Codex TUI picker，不读取 project-local 自定义 profile，不改变
Grok-aligned 多模型/多模态 owner。写集限定为 v2 permission profile protocol/schema、App Server handler/Turn lowering、
package typed client、Renderer gateway/submit builder、产品矩阵与命令/架构事实源。

唯一数据流：

`Desktop access mode -> permissionProfile/list -> unique allowed built-in profile -> turn/start { approvalPolicy, permissions } -> App Server resolver -> RuntimeRequest sandbox policy -> tool-runtime`

实现结果：

1. v2 protocol、schema manifest/bundle、generated TypeScript、typed client 与 public JSON-RPC handler 已同步。catalog 按
   Codex 内建顺序只公开 `:read-only`、`:workspace`、`:danger-full-access`，全部为 Desktop allowed profile。
2. Renderer 新回合提交前调用 typed gateway，要求目标 profile 唯一且 `allowed=true`；提交 wire 已从 legacy
   `sandboxPolicy` 直接替换为 `permissions`，catalog 缺失、重复、禁止或形状非法时不调用 `turn/start`。
3. App Server 将三个 profile 分别 lowering 为 `read-only`、`workspace-write`、`danger-full-access` runtime sandbox
   policy，并写入 `permissions` 与 `activePermissionProfile` provenance。未知 profile 和
   `permissions + sandboxPolicy` 双传均 fail closed。
4. `thread/settings/update.permissions` 仍在 RuntimeCore 边界明确拒绝，Codex project-local custom profile 不进入 Desktop；
   本切片不把 list/新 Turn lowering 冒充 settings mutation 完成。
5. 产品矩阵将 `permissionProfile/list` 从 planned 移入 implemented，统计更新为
   `135 implemented / 47 planned / 38 product-scope-excluded`，产品范围完成度 `135 / 182 = 74.2%`。

事实源分类：protocol/handler/catalog、typed package client、Renderer gateway、Turn resolver/lowering 与 runtime sandbox
policy 为 `current`；无新增 `compat` 或 `deprecated`；Renderer 新回合 legacy `sandboxPolicy` wire、Electron 权限业务命令、
本地重复 catalog、Codex TUI picker 与生产 mock fallback 为 `dead / deleted / forbidden-to-restore`。历史导入、read model 和
evidence 里的 canonical sandbox fact 继续属于 current projection，不是兼容入口。多模型、多模态 sampling/media lowering
与 provider catalog/readiness/retry/circuit breaker 仍由 Grok-aligned `model-provider` 承接。

最终验证：App Server protocol `116/116`、permission resolver/Turn lowering `3/3`、public JSON-RPC integration `1/1`、
app-server-client `116/116`、Renderer gateway/request builder/submit `35/35`、schema writer、generated type drift check 与
`npm run typecheck` 均通过。`npm run test:contracts` 通过 `301` 项 App Server client checks；
`npm run governance:legacy-report` 扫描结果为零分类漂移、零边界违规；
`npm run test:rust:related -- lime-rs/crates/app-server-protocol lime-rs/crates/app-server` 覆盖 `19` 个相关/反向依赖 crate
并全部通过。`npm run verify:gui-smoke` 已证明真实 Electron/preload/IPC/App Server shell；
`npm run smoke:agent-runtime-current-fixture` 已证明 current Agent fixture 聚合主链，且 `liveProviderUsed=false`。

Gate B evidence 额外断言 `permissionProfile/list` 先于 `turn/start`、Desktop wire 只含 `permissions`、runtime sandbox
policy 与 profile mapping 一致、`activePermissionProfile` provenance 到达匹配的 primary runtime Turn。合同回归为 `6/6`。
聚合 fixture 首次重跑暴露 image-command media workflow 没有匹配的 external text runtime Turn，而旧 evidence helper
错误地把全局最后一个 permission wire 与 primary GUI Turn 的空 runtime evidence 拼接；现已按 primary Turn identity
限定权限断言的适用范围。原始 `image-command` Electron 场景与完整 `smoke:agent-runtime-current-fixture` 均重新通过，
避免以不相关 runtime 证据制造假失败或假成功。相关 Prettier 检查、`npm run governance:scripts` 与最终
`git diff --check` 均通过；收尾重跑 `npm run test:contracts` 仍为 `301` 项 checks 全通过。

本切片达到当前 Lime Desktop 可交付门槛，但总 Codex/Grok 对齐计划仍未完成。下一刀继续从 `47` 个 planned method 中
选择具备真实 Desktop consumer 与唯一 current owner 的缺口；继续保持 Codex runtime、Grok model/multimodal 与
Desktop/TUI 产品边界，不恢复 legacy sandbox wire 或平行权限 catalog。

### 2026-08-10 配置控制面收口后的最终本地门禁与回归修复

本轮完成 Config Control Plane 切片的最终验证闭环。`verify:local` 的 smart Vitest 首次续跑在第 81/120 批暴露
`agentStreamUserInputSubmission.test.ts` 仍依赖旧 harness：测试未注入 current `permissionProfile/list` resolver，导致
无 Electron bridge 的单测环境提前失败，`submitOp` 未执行。修复为显式 mock current `resolveAllowedPermissionProfile`，并将
断言从 legacy `sandboxPolicy: "workspace-write"` 更新为 current `permissions: ":workspace"`；没有恢复旧命令或添加兼容层。

修复后使用 `npm test -- --resume` 从第 81 批继续，`.lime/test/vitest-smart-last-run.json` 最终为
`status=passed`、`120/120` 批通过。最终收尾门禁全部通过：`git diff --check`；`npm run governance:legacy-report`
（源码 2114、测试 1378、零引用候选、零分类漂移、零边界违规）；`npm run test:contracts`（301 checks，含生成类型无漂移、
命令/bridge、modality、脚本与文档边界）。`get_config/save_config` 扫描命中仅保留于负向 retired guard、历史诊断文本或
Rust 内部 YAML 持久化函数名，不存在 current Desktop/Renderer/fixture 调用。

事实源分类保持不变：Settings/Claw 配置读写走
`Desktop -> app_server_handle_json_lines -> config/read|config/batchWrite -> lime_core config.yaml`，为 `current`；
旧 Electron 配置命令、旧 `sandboxPolicy` wire 和未注入 current resolver 的测试假路径为 `dead / deleted / forbidden-to-restore`；
无新增 `compat` 或 `deprecated`。产品矩阵保持 `138 implemented / 43 planned / 39 product-scope-excluded`，产品范围完成度
`138 / 181 = 76.2%`。此前已通过的 `npm run smoke:agent-runtime-current-fixture`（`liveProviderUsed=false`）与
`npm run verify:gui-smoke`（真实 Electron/App Server shell、Settings/Claw/Memory/reload）继续作为 Gate 证据。

本轮把配置控制面主链和验证证据完整落库；总 Codex/Grok 对齐仍未完成，下一刀回到剩余 `43` 个 planned 中具备真实
Desktop consumer 的唯一 current owner，不恢复 TUI 专属 surface、第二套 Electron 后端或旧兼容入口。

### 2026-08-10 Windows Sandbox Readiness current slice

主目标与窄写集：只交付 Codex exact `windowsSandbox/readiness` 的 Desktop current control surface；不把尚未存在的
Windows restricted-token runner、setup flow 或通知伪装成完成，不复制 Codex TUI，并保持 Grok-aligned 多模型/多模态 owner
不变。写集限定为 v2 protocol/schema、App Server handler、typed package client、Renderer gateway、Execution Policy
Settings、五语言资源、产品矩阵与当前路线图事实源。

唯一数据流：

`Settings execution policy -> windowsSandbox/readiness -> App Server JSON-RPC -> tool-runtime sandbox plan -> Desktop status`

实现结果：

1. 新增 `WindowsSandboxReadiness` response/params/status enum，ingress 接受 omitted 或 `{}`，非空 params fail closed；
   handler 读取 `lime_core config`，复用 `plan_sandbox_backend`，只有 `Ready + enforced=true` 才映射为 `ready`。
2. typed package client 新增 `readWindowsSandboxReadiness`，schema/generated types 与 public JSON-RPC integration 已同步；
   Renderer gateway fail closed 校验三态，Settings execution-policy 只在 Windows 展示状态行和可重试入口。
3. 当前 Windows backend 仍由 `tool-runtime` 返回 `SandboxBackendStatus::Planned/enforced=false`，
   `prepare_sandbox_command(RestrictedToken)` 继续拒绝，因此 UI 显示 `需要更新`，没有 `setupStart` 假成功。
4. `windowsSandbox/setupStart`、`windows/worldWritableWarning` 与 `windowsSandbox/setupCompleted` 继续 planned；
   后续必须在 Windows 完成 runner enforcement、setup lifecycle 与 platform/Gate B evidence 后再提升状态。

事实源分类：protocol/handler、tool-runtime readiness mapping、typed client、Renderer gateway、Settings 状态投影和矩阵
为 `current`；未实现的 Windows runner/setup/notifications 为 `planned`；无新增 `compat` 或 `deprecated`。旧路线图中声称
Windows runner 已 enforce 的 current 文案已修正为 planned/fail-closed；历史执行记录保留为 evidence，不作为 owner。

验证：app-server-protocol readiness unit `1/1`，App Server readiness/ingress unit `4/4`，public JSON-RPC integration `1/1`，
`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server --test windows_sandbox_jsonrpc --no-fail-fast` `1/1`，
app-server-client `118/118`，Renderer API/Settings component `9/9`，`npx vitest run
"src/lib/governance/codexMethodProductScopeBoundary.test.ts"` `4/4`，`npm run test:contracts`、
`npm run governance:legacy-report`、`npm run test:rust:related -- lime-rs/crates/app-server-protocol lime-rs/crates/app-server`、
`npm run verify:gui-smoke` 均通过；`npm run check:protocol-types` 无漂移，Prettier、Rust fmt check 与 `git diff --check`
通过。Gate B 已由真实 macOS Electron Desktop Host、preload、App Server sidecar 和 GUI smoke 证明，但该证据不代表
Windows runner enforcement。尚未在 Windows 真机执行 runner/platform smoke，因此本 slice 只完成 readiness，不关闭
Windows runner blocker。产品矩阵更新为 `139 implemented / 42 planned / 39 product-scope-excluded`，产品范围完成度
`139 / 181 = 76.8%`。

统一本地门禁补充：`npm run verify:local` 首次在 lint 暴露
`WindowsSandboxReadinessStatus.tsx` 导出平台 helper 违反 `react-refresh/only-export-components`，已将 helper 移入独立
`windowsSandboxPlatform.ts`；第二次在前端第 `55/120` 批暴露旧治理断言仍要求过时的 ACL/token hardening 文案，已让
`legacySurfaceCatalog.test.ts` 对齐 current blocker“先在 `tool-runtime` 实现 runner”，定向回归 `224/224` 通过。
随后使用 `npm test -- --resume` 从第 55 批续跑至第 120 批，全部通过。`verify:local` wrapper 因第 55 批失败提前退出，
没有伪报为同一次 wrapper 全绿；其未触达阶段已按 task planner 分别完成：`npm run test:contracts`、
`npm run test:rust:related -- lime-rs/crates/app-server-protocol lime-rs/crates/app-server`、
`npm run verify:gui-smoke`、`npm run governance:legacy-report`、全量 Renderer/Node typecheck、src lint、i18n coverage 与
hardcoded copy scan 均通过。当前 readiness 切片达到本地可交付门槛，Windows 平台 runner 仍需独立实机证据。

下一刀回到 `windowsSandbox/setupStart` 前置的真实 restricted-token runner：在 `tool-runtime` current owner 建立
Windows 执行边界，先补 workspace write、外部路径拒绝、ACL 恢复、timeout 终止进程树和有界输出的平台证据；在此之前
不得新增 setup 成功通知或把 `ready` 作为默认状态。`setupStart`、`windows/worldWritableWarning`、
`windowsSandbox/setupCompleted` 继续 `planned`，无 `compat`/`deprecated` 新增；旧 runner 文案仅保留为 superseded
historical evidence，分类为 `dead / forbidden-to-restore` 的旧入口不得恢复。

### 2026-08-10 Windows restricted-token runner foundation

本轮沿 Codex `windows-sandbox-rs` 的 current owner 继续推进上一切片的 readiness blocker，写集限定为
`lime-rs/crates/tool-runtime/src/execution_process/{windows.rs,windows_acl.rs,windows_attr.rs}`、App Server
typed sandbox launch 接线、`tool-runtime` Windows 目标依赖、architecture 与本执行计划。Lime 仍是 Desktop：不复制
Codex TUI 的 setup/picker surface；多模型、多模态、provider catalog/readiness、retry/circuit breaker 和 media
lowering 继续归 Grok-aligned `model-provider`。

实现结果：

1. `LocalExecutionRequest` 增加 typed `LocalExecutionSandbox`；App Server 将 backend/policy/permission profile
   传给 `tool-runtime`，Seatbelt/bwrap 仍在 `tool-runtime` lowering，Windows `RestrictedToken` 进入专用 runner，
   不再由 App Server 拼接第二套 sandbox command。
2. Windows runner 使用 `OpenProcessToken`、`CreateRestrictedToken`、capability SID、默认 DACL 与
   `SeChangeNotifyPrivilege`；workspace/explicit write-root 通过 `icacls` 建立短生命周期 ACL lease，失败和 supervisor
   结束均 rollback，非法外部路径、glob、Unix special path 和非 existing path fail closed。
3. 进程使用 `CreateProcessAsUserW`、Job Object `KILL_ON_JOB_CLOSE`、`PROC_THREAD_ATTRIBUTE_JOB_LIST` 与显式
   `PROC_THREAD_ATTRIBUTE_HANDLE_LIST`；stdout/stderr 由 blocking pipe reader 接入既有 `ExecutionProcess`，复用
   128 KiB retained output、sequence、omitted bytes、timeout/interrupt/terminate 的 process-tree cleanup。
4. TTY/ConPTY 当前明确返回 `Unsupported`；网络策略只保留 Codex legacy 同层级的 offline env/proxy lowering，
   不宣称 WFP/firewall 强隔离。`windowsSandbox/setupStart`、`windows/worldWritableWarning`、
   `windowsSandbox/setupCompleted` 不在本轮伪造成功通知。

事实源分类：上述 target-gated runner、ACL plan/lease、typed launch request 与 App Server 接线为 `current` foundation；
`SandboxBackendStatus::Planned/enforced=false`、Windows readiness `updateRequired`、ConPTY/elevated setup/WFP 与
真实 Windows/Gate B evidence 仍为 `planned`/blocker；没有新增 `compat` 或 `deprecated`。旧 command wrapper、旧
runtime vendor、TUI setup surface 和 mock fallback 继续 `dead / deleted / forbidden-to-restore`，不得恢复。

验证：`rustfmt` 相关 Windows 文件、`git diff --check`、macOS `cargo check --manifest-path "lime-rs/Cargo.toml" -p
tool-runtime -p app-server` 通过；`cargo test --manifest-path "lime-rs/Cargo.toml" -p tool-runtime execution_process`
通过 `9/9`，此前 `app-server execution_process` 通过 `6/6`。Windows target 已安装但当前 macOS 宿主缺 MSVC/Windows
C toolchain，cross-check 在 `ring`/`zstd-sys` 原生依赖阶段阻塞，尚未取得 Windows API 类型检查或真机运行证据；不能把
runner 或 readiness 写成已交付 ready。

本轮推进了 readiness 前置的 `tool-runtime` current owner，但未关闭 Windows 平台 blocker。下一刀应在真实 Windows
toolchain/机器补 workspace write success、外部/metadata write denial、ACL rollback、timeout descendant kill、
bounded output 和 TTY/网络能力裁决；证据完成前保持 `ready` fail closed，继续不实现 setup 通知。

### 2026-08-10 Windows runner lifecycle and target API alignment follow-up

本轮继续沿同一窄写集复核 `tool-runtime` Windows runner，没有扩展到 Desktop/TUI setup surface 或 Grok model-provider。
实现修正：

1. Windows child environment 现在在 `env_clear=false` 时继承父环境，并按 Windows 大小写不敏感规则应用请求覆盖；
   `env_clear=true` 只保留显式覆盖。环境块补空环境双 NUL 终止和 embedded-NUL/非法变量名 fail-closed 校验，
   离线 lowering 补齐 `SBX_NONET_ACTIVE`、PIP/Git 约束标记。
2. Job Object 初始启用 `KILL_ON_JOB_CLOSE | BREAKAWAY_OK`。正常根进程退出后只移除 kill-on-close，reaper
   轮询 `ActiveProcesses` 并持有 ACL lease 到 Job 为空；interrupt、terminate、控制断开和 wait error 继续终止整棵树。
   这与 Codex `JobObject::preserve_descendants` 的根进程正常退出语义一致，同时不让短生命周期 ACL 在后代仍运行时提前 rollback。
3. 对照本地 `windows-sys 0.52` SDK 符号修正 `WAIT_*`/`SECURITY_ATTRIBUTES` 所属模块、`Win32_System_IO` feature 和
   `ReadFile`/`WriteFile` buffer 指针类型；ACL lease 在 icacls 部分失败时先登记路径，保证 rollback 覆盖已部分应用的 ACE。

验证：macOS `cargo test --manifest-path "lime-rs/Cargo.toml" -p tool-runtime execution_process` `13/13`、
`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server execution_process` `6/6`、
`cargo check --manifest-path "lime-rs/Cargo.toml" -p tool-runtime -p app-server` 通过，Rust fmt 与 `git diff --check`
通过；`npm run test:rust:related -- lime-rs/crates/tool-runtime lime-rs/crates/app-server` 的 7 个相关/反向依赖 crate
共 `2699` tests 全部通过。Windows target API 只完成本地 SDK 源码静态核对；完整 `x86_64-pc-windows-msvc` build 仍受 macOS 宿主缺少
MSVC/Windows C toolchain（`ring`/`zstd-sys` 缺少 Windows 头文件）阻塞，Windows 真机 runner/platform smoke 尚未执行。
因此 readiness 仍为 `SandboxBackendStatus::Planned/enforced=false`，`setupStart`、`windows/worldWritableWarning`、
`windowsSandbox/setupCompleted` 继续 `planned`；无新增 `compat`/`deprecated`，旧 runner/TUI setup surface 仍为
`dead / deleted / forbidden-to-restore`。本轮证明了 runner current owner 的生命周期和启动环境边界继续向 Codex
Windows sandbox 收敛，但没有关闭平台证据 blocker；下一刀仍是 Windows toolchain/真机 Gate B，之后才裁决 readiness 提升。

### 2026-08-10 Windows sandbox diagnostics fact-source correction

目标与范围：收敛 runner foundation 已接入后仍残留的“未实现”诊断，不改变 readiness、执行准入或 Desktop setup
surface。写集限定为 `tool-runtime` backend plan、对应 Agent 回归、coding active roadmap 与本执行计划。

完成结果：Windows plan 的 reason code 从 `sandbox_backend_windows_runner_not_implemented` 直接替换为
`sandbox_backend_windows_runner_platform_evidence_pending`，reason 明确 current execution process owner 已有
target-gated foundation，真正 blocker 是 Windows toolchain 与真机 enforcement evidence。状态继续为
`SandboxBackendStatus::Planned`、`enforced=false`，因此 App Server production decision 仍不会把未验证 runner
伪装为可交付 backend；Windows readiness 继续 `updateRequired`，没有新增 setup 成功通知。

分类：typed runner、ACL/Job/pipe lifecycle 与新诊断为 `current foundation`；Windows/MSVC build、真机 sandbox
smoke、ConPTY、elevated setup、WFP 与 Windows Electron Gate B 为 `planned`；旧“runner 完全未实现”诊断和指向已删除
owner 的 current 文案为 `dead / deleted / forbidden-to-restore`；无 `compat`、`deprecated`。下一刀仍是 Windows
toolchain/真机平台证据，不用 macOS 单测或源码存在推断 `Ready`。

### 2026-08-10 Desktop reasoning 完成态展开 identity 收口

目标与用户闭环：修复完成回合的思考摘要在点击展开后因 read-model 补全而重新折叠、点击目标从 DOM
消失的问题。用户从“已处理 / 已完成思考”摘要开始，一次点击后必须稳定看到 canonical reasoning 正文，且最终
回答继续位于思考过程之后。写集限定为 Turn timeline render projection、projection 回归、Claw 真实 Electron
fixture 展开断言和本执行计划。

根因与修复：`ConversationTurnTimeline` 以 process segment ID 保存历史详情展开状态并作为 React key；旧
`buildCanonicalTurnSegments` 用首尾 Item ID 生成该 identity。同一回合从 live reasoning Item 刷新为 App Server
read model 的两个 reasoning Item 后，segment ID 改变，展开集合失配并把真实 `AgentThreadTimeline` 换回
`HistoricalTimelinePreview`。process segment identity 已改为稳定的 `process:<turnId>:<ordinal>`；新增回归证明
reasoning Item 补全或追加不改变既有过程段 ID。fixture 只在过程块尚未打开时点击内部 summary，避免完成态默认
展开时先反向关闭再重开。

验证：projection Vitest `18/18`、Claw fixture guard `83/83`、Renderer/Node typecheck、scoped Prettier、
`node --check` 与 `git diff --check` 通过。`npm run test:related -- ...` 仍被仓库既有 Vitest 收集器
`EISDIR .../electron` 阻塞，已由上述精确测试入口替代。原始 Gate B 场景
`npm run smoke:claw-chat-current-fixture -- --scenario reasoning-first-visible --timeout-ms 240000` 通过：真实
Electron/preload/IPC、`app_server_handle_json_lines`、App Server/read model 与 GUI identity 一致；reasoning 在
final 前可见，完成态 canonical content 展开后稳定保留，`reasoningItemCount=2`；console/page error、legacy
command 与 production mock fallback 均为 `0`。证据：
`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-summary.json` 与同目录 chat 截图。

分类：稳定 Turn/process identity、canonical reasoning Item/read model、Desktop 展开交互与 Gate B fixture 为
`current`；首尾 Item 拼接 identity 和 fixture 对默认展开过程块的反向点击为 `dead / deleted /
forbidden-to-restore`；无 `compat`、`deprecated`。本切片不改变 Thread/Turn/Item 协议、provider lowering 或
架构 owner，因此无需更新架构图。

### 2026-08-10 Codex realtime / Grok multimodal 产品边界收口

目标与裁决：修正 method matrix、Renderer coverage 与既有产品架构之间的事实源冲突。Codex
`thread/realtime/*` 的 WebRTC/session request 与 notification 不进入 Lime Desktop；旧麦克风、录音和 realtime
voice GUI 已退役，恢复该 wire 会建立第二套 Thread lifecycle 和 Renderer 媒体队列。音频、语音和媒体能力继续按
grok-build 对齐 `model-provider` 的 catalog/capability/readiness 与 sampling/lowering，并由
`voice-core` / `media-runtime` 承接领域执行和 artifact 生命周期。

完成结果：14 个 realtime 方向化 method 从 `planned` 移入 `product-scope-excluded`，产品矩阵更新为
`139 implemented / 28 planned / 53 product-scope-excluded`，产品范围完成度为 `139 / 167 = 83.2%`。
upstream notification inventory 继续保留用于 drift 检测，但八个 realtime notification 统一只记录脱敏
method/field diagnostics，不进入 timeline、header、pending interaction 或音频播放；新增边界回归锁定它们不得进入
Lime generated manifest。无 `compat`、`deprecated` 新增，旧 realtime GUI/wire 继续为
`dead / deleted / forbidden-to-restore`。

架构确认：confirmed；责任开发者 root，2026-08-10。本切片没有新增 runtime 或协议，而是把既有 Desktop/Grok
边界写回架构、产品矩阵和 Renderer 守卫。下一刀从剩余 `28` 个 planned method 中选择具备真实 Desktop consumer
与唯一 current owner 的 surface，或回到 Windows 真机 sandbox evidence；不得用 Codex realtime/TUI surface 冒充
Grok 多模态对齐。

### 2026-08-10 Codex HEAD registry 与 current 字段增量审计

状态：completed。

目标与窄写集：将 method matrix 从 Codex `c4f42d161ae44a8d696ee9fb595709661979d187` 增量审计到
`c9c6c0daa994109cec50fddcb57d076fdf9e738c`，并只收敛已有 current boundary 的字段变化。写集限定为
`app-server-protocol` Model/Hook types、Grok-aligned model catalog projection、Renderer model registry、method 产品矩阵、
架构事实源和对应定向测试；不触碰 Windows runner 热区，不恢复 realtime/TUI，也不创建配置或诊断第二 owner。

增量裁决：Codex 新增的 `server/diagnostics` 只返回进程本地、无内容指标，没有 Lime Desktop 用户流程，归入
`product-scope-excluded / forbidden-to-restore`；矩阵变为 `139 implemented / 28 planned / 54 excluded`，总数
`221`，产品范围完成度仍为 `139 / 167 = 83.2%`。`model/list.multiAgentVersion` 从 model catalog 显式元数据透传，
缺失保持 `null`，禁止按 provider/model 名称或工具能力推断；Hook `executionMode` 直接来自冻结的
`HookSnapshot`。`configRequirements.autoReview` 与 MCP hook handler 仍没有本轮授权的 current 配置 consumer，
不以协议字段存在冒充实现。

验证计划：生成 Rust schema 与 typed client，运行 App Server protocol/Hook/model 定向 Rust 测试、Renderer
model registry 与 method boundary Vitest、`npm run test:contracts`、`npm run governance:legacy-report`、typecheck、
格式和 diff 检查。本切片不改变 GUI 交互、Electron/preload 或 Thread/Turn/Item producer，因此不重复运行 Gate B。

完成结果：Codex HEAD 已更新为 `c9c6c0daa994109cec50fddcb57d076fdf9e738c`。新增
`server/diagnostics` 已进入 method matrix 的 `product-scope-excluded / forbidden-to-restore`，没有进入 Lime manifest、
Electron 诊断后端或 provider readiness 事实源。`model/list.multiAgentVersion` 由 Grok-aligned model catalog 的显式
`multi_agent_version` 元数据透传，缺失保持 `null`；Hook `executionMode` 由冻结的 `HookSnapshot` 投影，缺省为 `sync`。
生成 schema 与 typed client 已同步。产品矩阵为 `139 implemented / 28 planned / 54 product-scope-excluded`，产品范围
完成度保持 `139 / 167 = 83.2%`。

验证结果：App Server protocol Model wire、App Server model propagation、公共 `hooks/list` JSON-RPC、Renderer model
registry `18/18` 与 method product-scope boundary `6/6` 定向测试通过；`npm run test:contracts` 通过（generated protocol
types `955`、App Server client contract `301` checks）；`npm run typecheck`、`npm run check:protocol-types`、定向
Prettier、`git diff --check` 与 `npm run governance:legacy-report` 通过，后者为零引用候选、零分类漂移、零边界违规。
Rust fmt 已完成。本切片没有改变 GUI、Electron/preload、Thread/Turn/Item producer 或 live provider sampling，故未重复
运行 Gate A/Gate B；已有 Electron 证据不被本字段增量冒充为新证据。

分类：新增 Model/Hook 字段、generated contract、Renderer registry projection 与 method matrix 为 `current`；
`server/diagnostics` 为 `product-scope-excluded / forbidden-to-restore`；没有新增 `compat` 或 `deprecated`。下一刀从剩余
`28` 个 planned method 中重新核对真实 Desktop consumer，优先选择已有唯一 owner、无需恢复 TUI 或第二套 provider
控制面的能力；Windows setup 仍受真机证据阻塞。

### 2026-08-10 Desktop fuzzy file mention current slice

状态：completed。

主目标：把 Composer 已有 `@` 输入面板缺失的项目文件检索接到 Codex exact 一发式
`fuzzyFileSearch { query, roots, cancellationToken }`，形成
`Desktop Composer -> Renderer typed gateway -> App Server JSON-RPC -> filesystem search owner` 唯一主链。选中结果只替换
当前 `@token` 为相对项目路径，沿用 Codex 的输入语义；不把文件路径伪装成 connector/plugin `Mention`，不复制 Codex TUI，
不改变 Grok-aligned 多模型/多模态 owner。

产品裁决：Codex `fuzzyFileSearch/sessionStart|sessionUpdate|sessionStop` 与
`fuzzyFileSearch/sessionUpdated|sessionCompleted` 在 upstream 文档中属于 experimental legacy fuzzy search flow。Lime
Desktop 使用请求级 cancellation token 丢弃陈旧结果，无需第二套长生命周期 session registry；因此本轮只把一发式
`fuzzyFileSearch` 提升为 `current`，其余五个 session surface 改为
`product-scope-excluded / forbidden-to-restore`。

窄写集：App Server v2 fuzzy file search protocol/schema/generated client、App Server filesystem search owner 与公共
JSON-RPC 测试、Renderer `src/lib/api` gateway、Composer file mention hook/pure insertion helper/panel wiring、五语言
`agentSkills` 文案、method/product/render matrix、`internal/aiprompts/{architecture,commands}.md` 与本执行计划。已有超过
`1000` 行的 `envelopes.rs`、`CharacterMention.tsx` 只允许协议注册或依赖注入接线；搜索、取消、状态与文本替换逻辑必须
落在新的短领域模块，不继续堆入超大文件。不触碰 Windows runner、Thread/Turn/Item producer、provider lowering 或并行
release 计划。

退出条件：空 query/空 root 返回空结果；root 必须是绝对目录；同一 cancellation token 的新请求取消旧扫描且不误删新
请求状态；结果限制、排序、相对路径、file/directory type 与 indices 有定向测试；Renderer 丢弃陈旧响应；只在 mention
模式且存在项目 root 时查询；选中带空格路径时正确加引号并只替换当前 `@token`。同步 generated contracts、五语言、产品
矩阵与回流守卫，并运行 Rust related/public JSON-RPC、Renderer unit/component、`npm run test:contracts`、
`npm run governance:legacy-report`、typecheck、GUI smoke 与风险匹配的 Gate A/Gate B。

完成结果：one-shot `fuzzyFileSearch` 已由 `CharacterMention` 接入 Renderer typed gateway、Electron
`app_server_handle_json_lines`、App Server JSON-RPC 与 filesystem search owner。Composer 已覆盖 loading/error/empty、
AbortSignal、稳定 cancellation token、request version 和陈旧响应丢弃；项目文件候选使用相对路径，选中后只替换当前
`@token`，空格路径加引号，不创建 connector/plugin `Mention`。协议 schema、generated client、Rust handler、五语言文案、
产品矩阵和负向回流守卫已同步。产品矩阵更新为 `140 implemented / 22 planned / 59 product-scope-excluded`，产品范围完成度
为 `140 / 162 = 86.4%`。

验证结果：Renderer gateway、插入 helper、hook 与组件定向测试 `29/29` 通过；产品矩阵、Renderer 投影覆盖和 notification
drift 定向测试 `20/20` 通过；Rust related 反向依赖扩圈退出码为 `0`；`npm run typecheck`、`npm run test:contracts`
（generated protocol types `959`、App Server client contract `301` checks）、`npm run governance:legacy-report`（零引用
候选、零分类漂移、零边界违规）和 `npm run verify:gui-smoke` 均通过。通用 GUI smoke 的真实 Electron/App Server shell
Gate B 证据位于 `.lime/qc/project-gates/standalone-shell-01-20260810115558-33013/shell-01-electron-smoke/summary.json`。

专用用户闭环 Gate B 已在真实 `http://127.0.0.1:1420/?nativeStartup=1` Electron 页签完成：
`window.__LIME_ELECTRON__ === true` 且 preload invoke 存在；当前项目为仓库根 `lime`，在 Composer 中输入
`保留前缀 @forge 保留后缀` 后看到 `forge.config.mjs` 等项目文件候选，点击首项后得到
`保留前缀 forge.config.mjs 保留后缀`，只替换当前 token 并关闭面板。脱敏 trace 证明
`transport=electron-ipc`、`command=app_server_handle_json_lines`、JSON-RPC method `fuzzyFileSearch`、单一 root 命中当前仓库且
cancellation token 存在；console error、page error、invoke error 与 production mock fallback 均为 `0`。截图证据：
`.lime/qc/gui-evidence/fuzzy-file-search-forge-candidates.png` 与
`.lime/qc/gui-evidence/fuzzy-file-search-forge-selected.png`。

分类：one-shot method、filesystem search owner、typed gateway、Composer 文件候选和本轮 Gate B 证据为 `current`；没有
`compat` 或 `deprecated`；三个 experimental session request 与两个 session notification 为
`product-scope-excluded / forbidden-to-restore`，旧 session registry、第二搜索入口和生产 mock fallback 为
`dead / deleted / forbidden-to-restore`。架构确认：confirmed；责任开发者 root，2026-08-10。多模型与多模态 control
plane 未改动，继续归 Grok-aligned `model-provider`。

### 2026-08-10 Codex remote environment and migration scope closure

状态：completed。

目标与窄写集：复核剩余 `planned` method 是否属于 Lime Desktop 产品范围。只审计 Codex exact
`environment/{add,info,status}`、`thread/environment/{connected,disconnected}`、
`externalAgentConfig/*`、`marketplace/*` 和 `plugin/share/*` 的协议语义、四层消费者与唯一 owner；不新增
远端 executor、配置迁移器、Plugin share service、marketplace backend、Electron IPC 或 renderer mock fallback。

事实裁决：Codex `environment/*` 通过 App Server `EnvironmentManager` 注册远端 exec-server WebSocket，读取 shell/cwd
并报告连接状态；Lime 当前只有本地执行环境、`TurnEnvironmentParams` identity、tool/MCP provenance 和 world-state cwd
投影，没有远端 registry、连接恢复或桌面环境选择 workflow。Codex external-agent config 是从其他 agent 产品检测和导入
配置/历史的迁移服务，Lime 没有 import wizard、source adapter 或 migration store。Codex `marketplace/*` 与
`plugin/share/*` 是远端 Plugin marketplace/share service 的管理和 principal mutation；Lime 的 marketplace 是独立
`skillMarketplace/install`，Plugin current owner 只承接本地 v3 catalog/install/activation。已有
`PluginShareContext` DTO 不是 handler、gateway 或 GUI consumer，不能冒充实现。

完成结果：19 个方向化 method 从 `planned` 移入 `product-scope-excluded / forbidden-to-restore`：
3 个 environment request、2 个 environment notification、4 个 external-agent request、2 个
external-agent notification、3 个 marketplace request、5 个 plugin/share request。矩阵更新为
`140 implemented / 3 planned / 78 product-scope-excluded`，产品范围完成度为 `140 / 143 = 97.9%`。
environment 与 external-agent notification 保留在 upstream drift inventory，但只进入 method/field-name 脱敏 diagnostics，
不得进入 Header、timeline、pending interaction 或 current projector；所有 excluded method 均由负向 manifest guard 锁定。

事实源分类：本地环境 identity、world-state、tool-runtime/MCP provenance、Plugin v3 catalog 和 Skills marketplace 为
`current`；本轮 excluded Codex remote/TUI/cloud surface 为 `product-scope-excluded / forbidden-to-restore`；无新增
`compat` 或 `deprecated`。未实现的 Windows setup/notifications 仍是 `planned`，必须等 Windows/MSVC toolchain、
restricted-token enforcement 和真实 Electron Gate B 证据，不得用 macOS 代码或 readiness `updateRequired` 伪造完成。

架构确认：confirmed；责任开发者 root，2026-08-10。本轮只收敛产品范围与 diagnostics 边界，未改变
`Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`，也未改变
Grok-aligned `model-provider` owner。下一刀回到 Windows 平台证据，或基于新发现的真实 Desktop consumer 重新做范围审计。

验证结果：JSON 解析通过；method matrix、environment/external-agent diagnostic-only 与 render projection 定向测试共
`22/22` 通过；`npm run test:contracts` 通过（generated protocol types `959`、App Server client contract `301` checks）；
`npm run governance:legacy-report`、`npm run typecheck`、受影响文件 Prettier 和 `git diff --check` 均通过。由于本轮无 GUI、
Electron、Rust runtime producer 改动，不重复 Gate B；已有 fuzzy-file Gate B 证据保持有效。

### 2026-08-11 CodeMode provider custom-tool canonical contract

状态：`completed / current foundation`（provider canonical contract 第六刀与 catalog/readiness 第七刀）。

主目标：沿 Codex CodeMode planning foundation 的下一刀，把 provider-neutral freeform/custom-tool contract
落到唯一的 Grok-aligned `model-provider` owner；保持 Lime 为 Desktop 产品，不恢复 Codex TUI 或第二套 runtime。

窄写集：`runtime-core` canonical tool/content/event、`model-provider` current request/lowering/Responses SSE reducer、
provider capability snapshot、`agent-runtime` provider-turn fail-closed、legacy provider-call handler negative boundary，
以及对应 Rust request-capture、lowering、stream 和架构/缺口文档。未引入 V8、JavaScript executor、`exec`/`wait` 广告或
compat wrapper。`current_client.rs`、`lowering.rs`、`stream.rs` 与 `provider_turn.rs` 已超过 1000 行，本刀只在既有
owner 点完成 contract 接线；下一次继续增加 provider/CodeMode 业务逻辑前，必须先把 current request DTO/conversion、
lowering tests 和 Responses custom reducer 分别拆到短领域模块，不能继续向这些大文件堆叠。

完成结果：新增 `ToolDefinition::Custom` + `FreeformToolFormat`、`CustomToolCall/CustomToolResult` canonical
variants；官方 OpenAI Responses route 在显式 `custom_tools` capability 下 lowering 为原生 `type: "custom"`、
`custom_tool_call`、`custom_tool_call_output`，custom input delta/done、output item 与 response.completed 均投影为
typed `CustomToolCall`；completed-only custom call 也补齐 exactly-once `ToolInputStart/Delta/End` lifecycle。grammar
fixture 与 Codex 保持 `syntax: "lark"`。Chat Completions、Anthropic、Gemini、第三方 Responses route 和 legacy
provider-call endpoint 对 custom 都 fail closed；provider turn 明确返回“需要可执行 CodeMode session”，不会把
不可执行 custom call 当普通工具。

验证：owner 单测 `model-provider 242/242`、`agent-runtime 196/196`、`lime-server 119/119`、`runtime-core 59/59`
通过；`cargo check -p runtime-core -p model-provider -p agent-runtime -p lime-server` 通过。`npm run
test:rust:related -- <本切片路径>` 对 14 个直接/反向依赖 crate 扩圈退出码 `0`，其中 `tool-runtime 323/323`；
`npm run governance:legacy-report` 为零引用候选、零分类漂移、零边界违规，Rust fmt 与 `git diff --check` 通过。
`npm run smoke:agent-runtime-current-fixture` 的真实 Electron/preload/IPC/App Server/read model 多场景 Gate B 通过，
`liveProviderUsed=false`；它证明 current runtime 无回归，不冒充 live provider 或可执行 CodeMode 证据。
新增 provider-turn 定向回归同时证明 custom call 在没有可执行 session 时为不可重试、不可 reroute 的
`InvalidRequest`，普通 tool executor 调用数与普通 lifecycle event 数均为零。

分类：canonical contract、official Responses lowering/stream、capability gate 和 negative boundaries 为 `current`；
未实现的 CodeMode session/V8、nested dispatch、yield/resume/terminate、approval/cancel 与 canonical terminal 为
`planned`；无 `compat`/`deprecated`。不得以当前 provider contract 宣称 CodeMode 已可执行。

第七刀完成 Grok-aligned catalog/readiness 接线。模型 taxonomy 新增标准 token `custom_tools`；基础
`coding/tools/streaming` requirement 只合入最终选中 profile slot 的 capability tags，fallback 后不继承原 slot，
未选中 review/fast/local 标签不污染当前 route。上游 Codex 八个 freeform 模型中，Lime canonical catalog 只有
`gpt-5.2 -> openai/gpt-5.2` 精确映射，因此只给该条目显式声明，不按 GPT/Codex 名称扩散。App Server effective
snapshot 取 authoritative model declaration 与 resolved provider protocol/host capability 的交集；仅官方 OpenAI
Responses 保留 `custom_tools`，第三方 Responses、Chat Completions、Azure 与 Ollama 均在 sampling 前返回不可重试的
`capability_gap / capability:custom_tools`。普通聊天未要求该 feature 时只裁剪 effective snapshot，不受阻塞。

第七刀写集为 `runtime-core` routing payload/model task、`core/services` runtime-feature taxonomy、
`model-provider` canonical catalog/provider capability、App Server resolved route contract 与本架构/计划。
`services/src/model_registry_service.rs` 已超过 1000 行，本刀只增加既有 taxonomy 映射；下一次继续增加 provider
catalog 解析或推断规则前，必须把 runtime-feature parsing/taxonomy 拆到独立短模块。

第七刀定向验证：selected-slot 与 fallback requirement `2/2`、runtime capability `1/1`、provider route capability
`1/1`、canonical catalog exact mapping `1/1`、App Server official/configured-provider/fail-closed/normal-chat route
`5/5` 通过。独立 `CARGO_TARGET_DIR=/tmp/lime-custom-tools-target` 下 App Server lib `1646/1646` 通过；
`npm run test:rust:related -- <本刀 Rust 路径>` 扩展到 20 个 current/反向依赖 crate 并退出 `0`，其中
`agent-runtime 196/196`、`model-provider 242/242`、`tool-runtime 323/323`。canonical JSON exact assertion、
`cargo fmt --all --check`、`git diff --check` 与 `npm run governance:legacy-report` 均通过，治理结果为零引用候选、
分类漂移和边界违规全部为 `0`。`npm run smoke:agent-runtime-current-fixture` 使用本轮最新 App Server sidecar
完成真实 Electron/preload/IPC/App Server/read model 聚合场景并通过，`liveProviderUsed=false`；它证明普通聊天、
审批、取消/继续、Plan、Skills、MCP、媒体与工作台 current 主链未被 readiness 交集回归，不冒充 live provider
或可执行 CodeMode 证据。第七刀完成度为 `100%`。

下一刀进入 thread-owned CodeMode session runtime，仍不得向生产模型广告尚不可执行的 `exec`/`wait`。

### 2026-08-11 CodeMode Agent loop executable boundary

状态：`completed / current foundation`（第八刀；P1-01 总项仍未关闭）。

主目标：把第六、七刀的 provider custom contract 接到可验证的 Agent loop/session 生命周期边界，同时保持
production fail closed；本刀不使用 Node、shell eval、Electron renderer 或测试 fake 冒充隔离 JavaScript host。

窄写集：`tool-runtime::code_mode` transport-neutral contract、`agent-runtime::provider_turn::code_mode` 调度模块、
`model-provider` custom failure lowering、对应 owner tests，以及本架构/执行计划。遵守上一刀大文件退出条件：
CodeMode 执行逻辑已从超过 1000 行的 `provider_turn.rs` 拆到短领域模块，新增 lowering 回归也放入独立
`current_client/code_mode_tests.rs`，没有继续向 `lowering.rs` 的 inline tests 堆业务场景。

完成结果：session contract 与 Codex 同构为 `StartedCell(cell_id + initial_response)`、`execute/wait/terminate/shutdown`、
保留 live/missing 的 wait outcome、session provider availability/create/limits 与 nested tool/notification/cell-close
delegate。response adapter 固定 `Script running/completed/failed/terminated`，yielded 输出稳定携带 `cell_id`，
terminated 是成功终态，失败同时保留 partial output 与 error，并复用统一 token truncation。

Agent loop 只在 frozen sampling-step snapshot 注入 executable session handle 时成组广告 `exec` custom tool 与
`wait` function tool；默认 production snapshot source 没有该 handle，因此不广告、不产生假 capability。
provider stream 先完整 materialize，再按同批并行策略执行 function/custom/wait，结果按原始 call 顺序回写；
`wait` 支持 `yield_time_ms/max_tokens/terminate`，turn cancel 对已启动 cell 调用 `terminate`。非 `exec` custom call、
无 session custom call 与 unsupported limits 均 fail closed。Responses lowering 对 failed custom result 优先发送
完整格式化 runtime output，不再只发送裸 error。

定向与 owner 验证：`agent-runtime 202/202`、`model-provider 243/243`、`tool-runtime 327/327` 全部通过；
`cargo check -p agent-runtime --lib` 通过。`npm run test:rust:related -- <本刀 13 个 current/反向依赖 crate 的路径>`
最终退出 `0`，其中 `app-server 1646/1646` 通过，仅有既有 test helper `dead_code` warning；拆分后
`cargo fmt --all -- --check` 与 `git diff --check` 均通过。新增回归覆盖 session handle 四操作、provider
default/non-default limits、yield/terminal/status、`exec`/`wait` 广告、wait resampling、mixed function/custom
结果顺序、session error recovery、无 session/非 exec fail closed、取消终止 active cell，以及 failed custom
output 的官方 Responses lowering。

production fail-closed 复核：current turn 在 `lime-agent::current_provider_turn` 中只从
`current_tool_step_snapshot_source` 捕获工具快照，该 source 只调用
`RuntimeToolStepSnapshot::with_tool_metadata(...)`；全仓非测试 Rust 源码没有
`with_code_mode_session(...)` 调用。因此 production sampling snapshot 不持有 executable handle，
`provider_turn` 的 `exec`/`wait` 成组广告分支保持关闭。

聚合门禁：`npm run governance:legacy-report` 通过，扫描 `2120` 个文件，零引用候选、分类漂移候选、边界违规
均为 `0`。`npm run smoke:agent-runtime-current-fixture` 通过；该门禁重建 App Server sidecar，并实际经过
Electron、preload/IPC、App Server、runtime/read model 与 GUI，覆盖历史恢复、流式终态、停止/继续、approval、
Plan、Skills、MCP、媒体引用、Coding Workbench 等 current fixture 场景；结果明确
`liveProviderUsed=false`，因此属于 Gate B external fixture 证据，不冒充 live provider 或 production CodeMode host。

分类：transport-neutral session/provider/delegate contract、Agent loop adapter、测试注入下的 `exec/wait` executable
boundary 与 custom failure lowering 为 `current foundation`；production thread-owned session registry、隔离 JS/V8
host、nested delegate dispatch、notification/cell-close implementation、thread interrupt/shutdown owner、canonical
CodeCell/Tool Item 与 GUI projection 为 `planned`；无 `compat`/`deprecated`，无新增 mock production fallback。

架构确认：confirmed；责任开发者 root，2026-08-11。确认范围包括
`tool-runtime -> agent-runtime -> model-provider` 依赖方向、Desktop/TUI 分界、StartedCell 生命周期、取消/终止、
mixed-call transcript 顺序和 production availability fail-closed。第八刀完成度为 `100%`，P1-01 不得据此标记完成。

下一刀建立按 canonical Thread identity 持有的 lazy CodeMode service：session actor interrupt 终止 active cells，
shutdown 关闭 session，provider availability 未就绪时不创建 owner。随后才能接真实隔离 host 和 nested dispatch；
在这两项完成前继续禁止 production 注入 handle 或对外宣称 CodeMode 可用。

### 2026-08-11 CodeMode thread-owned lazy service

状态：`completed / current foundation`（第九刀；P1-01 总项仍未关闭）。

主目标：把 CodeMode session 生命周期收进 canonical Thread 对应的 session actor，建立 lazy create、active cell
interrupt 与 thread shutdown 边界；本刀不接真实隔离 host、nested tool dispatch、canonical Item 或 production
sampling snapshot 注入。

窄写集：`agent-runtime::code_mode`、`agent-runtime::session_loop` actor/resource/context、App Server 中创建 session
actor 的 current 调用点、对应 owner tests，以及本架构/执行计划。退出条件：session actor 创建必须同时绑定
`session_id + thread_id` 且拒绝 identity 漂移；provider availability 失败不得创建 service/runtime session；首次
CodeMode operation 才能 lazy create；interrupt 必须 terminate active cells；shutdown 必须关闭已初始化 session，未初始化
service 的 shutdown 不得反向触发 create。

完成结果：新增 `agent-runtime::code_mode::RuntimeCodeModeServiceFactory` 与 per-thread
`RuntimeCodeModeService`，availability 和 delegate factory 任一失败都使 actor 不持有 CodeMode handle；runtime session
由 `OnceCell` lazy create，active cell 由 service 精确跟踪。`RuntimeSessionResources` 现在由 actor 唯一持有 canonical
`thread_id` 与可选 service，task context/input handle 只投影该资源。actor replace/interrupt 先终止 active cells，再
结束当前 task；显式 shutdown、command channel 关闭和 registry shutdown 都关闭已初始化 runtime session，未使用的
service 不会启动 provider session。

`RuntimeSessionRegistry::get_or_create` 已直接切换为 `(session_id, thread_id)`，空 identity 与同 session 的 thread
漂移均 fail closed；旧单参数 API 已删除。App Server compact、turn submit/steer/action、session operation 与 shell 等
current actor 创建点全部传入 stored canonical thread identity。production `RuntimeCore` 仍使用
`RuntimeSessionRegistry::default()`，没有注入 CodeMode factory；因此 production snapshot 不持有 handle，`exec/wait`
广告保持关闭。

验证结果：第九刀 lifecycle 定向覆盖 availability fail-closed、stable identity、unused shutdown、active cell
interrupt 与 session shutdown；完整 `agent-runtime 205/205`、`app-server 1646/1646` 已通过。`npm run
test:rust:related -- <第九刀 agent-runtime/App Server 路径>` 扩展到 `agent-runtime`、`app-server`、`lime-agent`、
`lime-scheduler`、`lime-server` 并退出 `0`，仅有既有 App Server test helper `dead_code` warning。`cargo fmt --all
-- --check` 与 `git diff --check` 通过。此前同一实现状态下的 `npm run smoke:agent-runtime-current-fixture` 已通过真实
Electron/preload/IPC/App Server/runtime/read model/GUI 聚合场景，`liveProviderUsed=false`；它只证明 Desktop current
主链无回归，不冒充 production CodeMode host 或 live provider 证据。

文件边界：`session_loop` 的 registry/resources 已从 actor 拆到短领域模块；`input_queue.rs` 当前 856 行，仅增加
资源引用和 context getter，未继续堆叠 lifecycle 逻辑。下一次修改 input queue 业务规则前，应把 task context/input
handle 与 pending queue 分离到独立模块，使非生成文件回落到 800 行以内。

分类：thread-owned service、canonical actor identity、lazy create、active-cell interrupt 与 shutdown 为
`current foundation`；production isolated host、nested delegate dispatch、notification/cell-close implementation、
canonical CodeCell/Tool Item 和 GUI projection 为 `planned`；无 `compat` 或 `deprecated`，旧单参数 actor API 为
`dead / deleted / forbidden-to-restore`。第九刀完成度为 `100%`，P1-01 不得据此标记完成。

下一刀进入真实隔离 JS/V8 host 与 nested delegate；在两者完成、canonical lifecycle 接线并通过 Desktop Gate B 前，
继续禁止 production 注入 CodeMode factory/handle 或对外宣称 CodeMode 可用。多模型、多模态 catalog/readiness 与
sampling/media lowering 继续归 Grok-aligned `model-provider`，本刀未改变这些 owner。

### 2026-08-12 CodeMode per-cell nested dispatch

状态：`completed / current foundation`（第十刀；P1-01 总项仍未关闭）。

主目标：把 CodeMode runtime 发起的 nested tool callback 精确路由到启动该 cell 的 frozen sampling-step executor，
保证它继续复用普通工具的权限、取消、lifecycle 与 output contract；本刀不注入 production factory，也不以 synthetic
session 冒充隔离 JavaScript host。

窄写集：`tool-runtime::code_mode` 的 transport-neutral per-cell delegate extension、`agent-runtime::code_mode` 的
route/gate 生命周期、`agent-runtime::provider_turn::code_mode` 的 frozen executor delegate、对应 owner tests，以及本
架构/执行计划。未修改超过 800 行的 `session_loop/input_queue.rs`，也未新增 crate、脚本或 parallel runtime owner。

完成结果：session contract 新增 `execute_with_delegate`，默认保持普通 `execute` 语义；thread-owned service 为每个
started cell 注册独立 delegate route，并用 watch gate 解决 runtime callback 先于 `StartedCell` 返回的竞态。terminal
initial response、wait terminal/missing、terminate、actor interrupt、cell close 与 shutdown 都清理 route/gate；fallback
只属于 session factory 创建时冻结的 thread delegate，不读取最近 active turn。

provider-turn nested delegate 按 `RuntimeCodeModeTool.global_name` 在当前 sampling step 的 frozen 工具集合中查找，拒绝
未启用工具与 `exec`/`wait` 递归调用；调用继续经过同一个 `RuntimeToolExecutorHandle.bind(...).execute_call(...)`，并
携带 canonical turn/session、working directory、turn context、cancellation token 与 lifecycle emitter。structured output
优先回传给 JS，失败保持 error；没有建立第二套权限、approval 或 handler registry。

验证结果：新增 service race 回归证明 cell route 在 nested callback dispatch 前完成绑定；provider-turn 回归证明 nested
`read` 只执行冻结 executor 一次，并发出同一普通 lifecycle 的 Started/Completed。完整 `agent-runtime 207/207`、
`app-server 1646/1646` 通过；`npm run test:rust:related -- <第十刀四个 owner 路径>` 扩展到
`agent-runtime`、`app-server`、`lime-agent`、`lime-mcp`、`lime-scheduler`、`lime-server`、`tool-runtime` 并退出
`0`，其中 `tool-runtime 327/327`，仅有既有 App Server test helper `dead_code` warning。`cargo fmt
--manifest-path lime-rs/Cargo.toml --all` 已应用，本轮继续以 fmt check、diff check 与治理扫描收尾。

分类：per-cell delegate contract、route/gate 生命周期与 frozen normal-tool dispatch 为 `current foundation`；production
isolated JS/V8 host、notify 注入、canonical CodeCell/Tool Item/GUI projection 与 App Server factory 注入为 `planned`；
无 `compat`/`deprecated`，未增加 mock production fallback。第十刀完成度为 `100%`，P1-01 不得据此标记完成。

下一刀先把 `exec`/`wait` 与 notify 接入 canonical Tool/Thread/Turn/Item 生命周期，继续保持 production unavailable；
随后再引入独立 V8 host。新增 `v8`/ICU 及 host binary 属于核心依赖与打包变更，必须单独确认并同步 Cargo/Forge、
macOS/Windows packaging 和 Gate B 证据。多模型、多模态 owner 继续是 Grok-aligned `model-provider`。

### 2026-08-12 CodeMode lifecycle 与 notify Desktop projection

状态：`completed / current foundation`（第十一刀；P1-01 总项仍未关闭）。

主目标：让 CodeMode control tools 不再绕过 canonical Tool lifecycle，并把 nested `notify` 接入 Lime Desktop 已有的
增量事件投影；本刀不注入 production factory，不把 GUI notification 冒充 provider transcript，也不建立平行
CodeCell 存储或第二套 lifecycle owner。

窄写集：`tool-runtime::tool_lifecycle` 的 transport-neutral output-delta extension、
`agent-runtime::provider_turn::code_mode` 的 outer control-tool lifecycle/notify delegate、Lime Agent current provider-turn
emitter、对应 owner tests，以及本架构/执行计划。没有修改 Electron/JSON-RPC schema、GUI 文案、provider wire、
workspace manifest 或核心依赖。

完成结果：`exec` 与 `wait` 现在都以原 provider call identity 发出 canonical Started/Completed lifecycle；完成输出保留
`code_mode_cell_id`、`code_mode_output_status`、`handler_executed=true` 与格式化 partial output，CodeMode failure 不再被
普通 failure normalization 吞掉正文。nested 普通工具仍使用自己的 lifecycle，outer `exec` 因此形成
`exec started -> nested started/completed -> exec completed` 的稳定顺序；`wait` 继续复用同一 thread-owned session。

`ToolLifecycleEmitter` 新增默认 no-op 的 `emit_output_delta` host capability，避免 transport-neutral owner 依赖 Agent
protocol。provider-turn delegate 把非空、未取消的 `notify` 投影为 `ToolOutputDeltaEvent`，绑定 outer `exec` call id、
canonical turn id、cell id、`tool_name=exec` 与 `notification_kind=code_mode_notify`；
`CurrentTurnToolLifecycleEmitter` 再把它接入既有 `AgentEvent::ToolOutputDelta` Desktop host event pipeline。该事件当前只供
App Server/GUI 消费；同一 notify 同时进入本 sampling step 的下一次 provider request，作为 outer call id 对应的
`custom_tool_call_output`，排在最终 exec output 前，补齐 Codex active-turn transcript 语义。该注入不经过
`RuntimeSessionInputHandle`，不创建 durable Item；独立 CodeCell Item/cell-close projection 仍未实现。

验证结果：notify correlation、outer `exec/wait` lifecycle、outer+nested lifecycle 顺序和 Agent output-delta projection
定向回归通过；完整 `agent-runtime 207/207` 与 `lime-agent 255/255` 通过。`npm run test:rust:related --
<第十一刀五个 owner 路径>` 扩展到 `agent-runtime 207/207`、`app-server 1646/1646`、`lime-agent 255/255`、
`lime-mcp 160/160`、`lime-scheduler 24/24`、`lime-server 119/119` 与 `tool-runtime 327/327`，完整退出 `0`。
`npm run governance:legacy-report` 扫描 `2120` 个 current 文件，零引用候选、分类漂移和边界违规均为 `0`；
`npm run smoke:agent-runtime-current-fixture` 在重建本轮 App Server sidecar 后通过真实 Electron/preload/IPC/
App Server/runtime/read model/GUI 聚合场景，`liveProviderUsed=false`；它证明 Desktop current 主链无回归，不冒充
production CodeMode host 或 live provider 证据。`cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check` 与
`git diff --check` 作为最终收尾门禁。

分类：outer control-tool lifecycle、普通 Tool Item projection 与 notify Desktop output-delta 为 `current foundation`；
本 sampling step 的 notify provider-transcript projection 也为 `current foundation`；production isolated JS/V8 host、
CodeCell trace/evidence owner、App Server factory 注入和 CodeMode 专项 Gate B 为 `planned`。公开 App Server ThreadItem
不新增 CodeCell variant；无 `compat`/`deprecated`，
未增加 mock production fallback。
第十一刀完成度为 `100%`，P1-01 不得据此标记完成，production 继续不广告 `exec/wait`。

下一刀是 production isolated session provider，但它需要新增 V8/ICU 核心依赖和 Desktop host 打包边界；按仓库高风险
包管理规则，必须取得明确确认后再修改 Cargo/lock/Forge。获批前最有价值的无依赖工作是核对 Codex rollout-trace 的
CodeCell 事实与 Lime current owner；不得把 CodeCell 伪造成 per-turn `Tool` Item 或新的公开 ThreadItem。late-notify
取消/终态守卫可独立落在 thread-owned runtime route，不改变 Thread/Turn/Item owner。

### 2026-08-12 CodeMode late-notify terminal guard

状态：`completed / current foundation`（第十二刀；P1-01 总项仍未关闭）。

主目标：对齐 Codex cell-close 后的终态路由语义，禁止迟到的 nested `invoke`/`notify` 重新创建空 dispatch gate、永久等待、
回落到 session fallback delegate，或继续写入 Desktop/provider transcript；本刀不新增 JS host、协议 schema、依赖或 durable
CodeCell Item。

窄写集：`agent-runtime::code_mode` 的 closed-cell route guard 与回归、`agent-runtime::provider_turn::code_mode` 的 nested
delegate 原子终态守卫、架构说明与本执行计划。terminal response、terminate、actor interrupt、host `cell_closed` 和
shutdown 都将 cell 标记为 closed；同 cell id 重新启动时先清除旧标记，再安装新的 route。late dispatch 在 route lookup
入口 fail closed，避免重新创建 watch gate；provider-turn delegate 在工具执行前后以及 notify 投影/ transcript 注入前检查
closed/cancelled，避免已关闭 cell 的输出越过终态边界。

验证：`agent-runtime` CodeMode 定向测试 `5/5`（含 late notify 不等待回归），完整受影响矩阵 `agent-runtime 208/208`、
`app-server 1646/1646`、`lime-agent 255/255`、`lime-scheduler 24/24`、`lime-server 119/119`，以及此前
`lime-mcp 160/160`、`tool-runtime 327/327` 与真实 Desktop fixture Gate B 均已通过；`cargo fmt --all -- --check`、
`npm run governance:legacy-report`（2120 文件，0 零引用候选/0 分类漂移/0 边界违规）与 `git diff --check` 均通过。独立 CodeCell
terminal owner、production isolated host/factory 和 CodeMode 专项 Gate B 仍为 `planned`，无新增 `compat`/`deprecated`。

### 2026-08-12 CodeMode CodeCell projection scope correction

状态：`completed / current governance`（撤回临时 per-turn CodeCell Item projection）。

对照 Codex 后确认，`app-server-protocol` 的公开 `ThreadItem` 没有 `CodeCell` variant；CodeCell 生命周期由
`rollout-trace::CodeCellTraceContext` 以 thread/turn/call/runtime-cell identity 写入内部 trace，并由 reducer 关联
model-visible `CustomToolCall`，不进入公开 GUI ThreadItem。Lime 当前没有 rollout-trace 等价 owner；App Server `RuntimeEvent`
是产品事件链，不能作为 runtime trace 的隐式存储或 GUI 旁路。因此已删除把 CodeCell started/closed 伪造成 per-turn
`Tool` Item 的临时实现与正向测试，保留 outer `exec`/`wait` canonical lifecycle、notify Desktop delta/provider transcript
projection，以及第十二刀 closed-cell late-notify guard。

分类：outer Tool lifecycle、notify projection、closed-cell route guard 为 `current foundation`；CodeCell trace owner、生产
isolated session provider/factory、CodeMode 专项 Gate B 为 `planned / alignment-open`；伪造的 per-turn CodeCell Item 为
`dead / deleted / forbidden-to-restore`；无 `compat`/`deprecated`。当前不新增 RuntimeEvent、ThreadItem union、GUI card 或
第二套 trace store。

验证：`agent-runtime 208/208`、`lime-agent 255/255`、`tool-runtime 327/327`、`cargo fmt --all -- --check`、
`npm run governance:legacy-report`（2120 文件，0/0/0）与 `git diff --check` 通过；`rg` 未发现
`CodeModeCellLifecycle*`/`emit_code_mode_cell` 回流。撤回后重新执行 `npm run smoke:agent-runtime-current-fixture`，
真实 Electron/preload/IPC/App Server/runtime/read model/GUI 聚合场景完整通过，`liveProviderUsed=false`；该证据证明
current Desktop 主链无回归，不冒充 production CodeMode host 或 live provider。

下一步：取得 V8/ICU 隔离 host、Cargo/lock 与 Electron Forge 打包边界的明确确认后，先接 production session provider/factory；
CodeCell trace 只有在出现真实 consumer 和 owner 后才实现，不得以 GUI Item 或 RuntimeEvent 临时承接。

### 2026-08-12 CodeMode production in-process V8 与 factory

状态：`completed / current foundation`（第十三刀；P1-01 总项仍未关闭）。

主目标：在 Lime Desktop current App Server sidecar 中接入真实 sandbox-enabled V8 session provider，把 production
Runtime backend factory、Grok-aligned model `tool_mode`、resolved provider `custom_tools` capability 与 thread-owned
executable session 三重门禁接成唯一采样链，并建立可重复、校验失败即停止的 V8 编译产物供应链。本刀不恢复公开
CodeCell Item，不把 Electron renderer/TUI/系统 Node 当 JS host，也不把 in-process isolate 宣称为 standalone host。

窄写集：`tool-runtime::code_mode::v8`、`agent-runtime::code_mode/session_config`、Lime Agent current sampling snapshot、
App Server runtime factory/model registry route、`core/services/model-provider` model metadata、Cargo manifest/lock、
`scripts/lib/rusty-v8-artifacts.mjs` 与现有 Rust/sidecar 构建入口、Windows workflow，以及架构/本执行计划。Forge
resources 仍只包含静态链接后的 `app-server` sidecar；V8 archive/binding 不作为运行时资源复制。

退出条件：每个 cell 使用 fresh sandbox V8 isolate，支持 nested tools、notify、timer、yield/wait/terminate、store/load
与 pragma limits；未知 pragma/tool mode、缺 provider capability、缺 session 与 `CodeModeOnly` 不可用均 fail closed；
只有 `code_mode|code_mode_only + custom_tools + executable session` 才附着 session 并广告 `exec/wait`。所有 current
Rust 与 Electron sidecar 构建入口从 Cargo.lock 解析精确 V8 version，只消费 Codex `ptrcomp_sandbox_release` 资产并
验证 archive/binding 两项 SHA-256。最低验证为 V8/agent/app-server 定向与相关测试、artifact/sidecar tests、真实
Electron CodeMode Gate B、scripts/legacy/version 治理、rustfmt 与 diff check。

当前完成：V8/ICU process 初始化、fresh isolate cell runtime、session store、nested callback、notify、timer、
yield/wait/terminate/cancel、sandbox verification 和 exec pragma 已落地；production factory 只注入 current runtime
backend，mock/external/unavailable 不注入。model registry/direct provider config 只接受 `direct / code_mode /
code_mode_only`，未知值回落 Direct；snapshot 三重门禁已补四场景回归。V8 内部无法到达 Lime nested-call contract 的
`ToolKind/CodeModeToolKind` 传播链已按 dead 删除。构建 helper 已在默认 cache 与显式 override 两种路径验证 Codex
release checksum，Rust layer、local CI、dev/rebuild sidecar、Electron assets build 和 Windows workflow 已接入。

当前验证：`cargo check -p app-server --lib` 无本刀 warning；model metadata、tool-mode normalization 与三重门禁新增
定向测试 `3/3` 通过；artifact/sidecar/Rust runner Vitest `36/36` 通过；真实无 override 下载与显式 override
checksum 校验均通过。遗漏的 provider model tool-mode ingestion `1/1`、agent-runtime custom exec `3/3` 与 mixed
function/custom result order `1/1` 已补跑。Responses fixture 与 Gate B 结构回归 `19/19` 通过。

专项 `npm run smoke:code-mode-electron-gate-b` 已通过，evidence 位于
`.lime/qc/gui-evidence/code-mode-electron-gate-b/code-mode-electron-gate-b-summary.json`，截图位于同目录 PNG。该 Gate B
证明真实 Electron/preload/IPC、`app_server_handle_json_lines`、runtime backend、official-host Responses route、production
V8 factory、custom `exec` 输出回采样、canonical `dynamicToolCall` completed 与 GUI 可见终态；Provider 请求 Host 仍为
`api.openai.com`，只由标准 `HTTP_PROXY` 路由到受控 fixture。公开 Item 类型为
`userMessage/dynamicToolCall/agentMessage`，没有 `CodeCell`，且 mock fallback、invoke/console/page/provider error 均为零。
它不证明 live OpenAI、standalone OS-process sandbox 或 macOS/Windows packaged parity。

收尾门禁：`npm run test:contracts` 通过 301 项 App Server client contract，并通过 protocol drift、command、harness、
modality、scripts、Electron release workflow 与 docs boundary；`npm run verify:gui-smoke` 通过真实 Electron Host、preload、
App Server 初始化、Claw shell 与设置页 smoke；`npm run smoke:agent-runtime-current-fixture` 完整通过当前聚合 Electron
场景，`liveProviderUsed=false`。`npm run governance:scripts`、`npm run governance:legacy-report`（2120 个 current 文件，
0 零引用候选 / 0 分类漂移 / 0 边界违规）、`npm run verify:app-version`、`cargo fmt --all -- --check` 与
`git diff --check` 均通过。

隔离边界：当前 provider 与 session actor 位于 App Server sidecar 进程内，V8 sandbox + fresh isolate 只提供 JS
内存边界，不提供 Codex 最新 standalone CodeMode host 的进程故障和 OS 资源隔离。该差距与 thread-owned CodeCell
trace/evidence owner 继续保持 `alignment-open`；在两者关闭前不得宣称 CodeMode 全面对齐完成。

模型目录审计：Codex 当前有 8 个 `apply_patch_tool_type=freeform` 模型，Lime 只对
`openai/gpt-5.2` 存在精确 canonical 映射；`gpt-5.6-sol/terra/luna`、`gpt-5.5`、`gpt-5.4/mini` 与
`codex-auto-review` 尚不在 Grok-aligned catalog。该差距不阻塞本刀 deterministic Gate B，但继续属于多模型控制面
`alignment-open`，只能在 `model-provider` catalog/capability/readiness owner 中补齐，不能按 provider 名称放宽。

### 2026-08-12 CodeMode standalone process host

状态：`completed / current`（第十四刀；P1-01 仅剩 CodeCell trace/evidence owner 未关闭）。

主目标：按 Codex `ProcessOwnedCodeModeSessionProvider -> code-mode-host` 的本地 stdio host 边界，把 production V8 执行
从 App Server sidecar 进程迁到独立 OS 进程。Lime 只实现 Desktop 当前需要的 process-owned stdio transport，不引入
Codex TUI、remote WebSocket/gRPC 控制面或第二套 product runtime；迁移完成后 production factory 直接替换为 process
provider，不保留 in-process fallback。

窄写集：`tool-runtime::code_mode` 的 length-prefixed protocol/process client/host、`code-mode-host` bin、
`agent-runtime::code_mode` production factory、现有 App Server/Electron sidecar build 与 resource packaging、CodeMode
专项 Gate B、对应 Rust/Node tests，以及架构/本计划。现有 V8 runtime 只作为 host 内部执行 owner，App Server 只持有
process client；不新增 crate、公开 ThreadItem、Renderer backend、compat wrapper 或 mock production fallback。

退出条件：protocol V1 handshake/session open/execute initial response/wait/terminate/shutdown、nested tool/notify/cancel/
cell-close correlation、frame size与 pending request 上限均 fail closed；host 缺失或崩溃不得回退 in-process，必须让当前
session 显式失败且不拖垮 App Server。dev、Rust layer、Electron asset、Forge 和 Windows CI 必须成组构建/校验两个
sidecar binary。专项 Electron Gate B 必须额外证明独立 `code-mode-host` PID、App Server PID 不同、完整 custom exec
回采样与 GUI terminal；随后再运行 contracts、GUI/current fixture、scripts/legacy/version、rustfmt 与 diff 门禁。

完成结果：production factory 已只使用 `ProcessCodeModeSessionProvider`；V8 provider 收敛为 host 内部 owner，没有
App Server in-process fallback。protocol V1 覆盖握手、session open/execute 两阶段 response、wait/terminate/shutdown、
nested tool/notify/cancel/cell-close、64 MiB frame、1024 in-flight/pending delegate 上限与连接失败传播。host 路径只从
App Server/测试二进制同目录或显式测试环境解析，缺失时 availability fail closed。

构建与资源结果：dev/Electron assets 在一次 Cargo invocation 中成组构建 `app-server` 与 `code-mode-host`；二者复制到
`dist-electron/app-server/<platform>/`，manifest 分别记录 `sha256`/`codeModeHostSha256`，packaged verifier 强制校验。
macOS arm64 实际资源为 `app-server 252775064 bytes` 与 `code-mode-host 66597536 bytes`，均为 `0755` 且双 SHA 复算一致。

验证结果：标准 sandbox V8 环境下 App Server/host 双 binary Cargo check 无 warning；process Rust tests `6/6`，
artifact/sidecar/assets/fixture/package/Gate B script tests `56/56`，Rust related 反向依赖矩阵、`npm run test:contracts`
（303 项）、rustfmt 与 diff check 通过。Windows quality 现在显式以同一次 Cargo invocation 检查 `app-server` 与
`code-mode-host`，client contract guard 固定该命令；Windows test package 继续由 `electron:build` 成组构建并由 packaged
resource verifier 校验双 binary。专项 `npm run smoke:code-mode-electron-gate-b` 重新通过，evidence thread
`019ff3ca-7f26-71d2-be81-6a16b7895515`；Electron/App Server/host PID 为 `44199/44203/44521`，host parent PID 为
`44203`。17 项 Gate B assertion 全通过，custom exec 两次 Responses 回采样、canonical `dynamicToolCall`、GUI final text、
IPC trace 均成立，production mock/invoke/console/page/provider error 为零。该证据不冒充 live OpenAI 或 Windows packaged
parity；后者由 release Windows runner 继续验证。

统一本地门禁收尾：`npm run verify:local` 首轮暴露三项既存事实源偏差。`useAgentChat` provider sync 回归在发送前
settings RPC 永久 pending，却等待 `sendMessage()` 完成后才释放 promise，形成测试自锁；测试现改为先观察 turn 未提交，
释放门禁后再等待发送完成。DevBridge 已退役 `pluginUiRuntime/start` 按 current 通用 App Server read policy 使用
`30000ms`，旧断言仍保留 `5000ms`，现只同步事实期望。App Server 同时直接拥有 `RuntimeProviderProtocol` 映射，违反
`model-provider` capability owner；canonical `ProtocolKind -> RuntimeProviderProtocol -> ProviderCapabilities` 转换现统一收回
`model-provider::ProviderCapabilities::from_route`，未放宽治理白名单，`runtime.rs` 通过复用 current imports 保持原行数预算。
修复后 `npm test -- --resume` 从第 50 批续跑至第 120 批全部通过；随后 fresh `npm run verify:local` 全绿，覆盖 120 个
Vitest smart batches、App Server client `303` checks、13 个 current/反向依赖 Rust crate、真实 Electron/App Server GUI
smoke、lint、typecheck、i18n、scripts/docs/version 门禁。Rust 仅保留既有 App Server test helper `dead_code` warning。

### 2026-08-18 CodeMode CodeCell trace/evidence owner 与 benchmark 收尾

状态：`completed / current`（P1-01 CodeCell trace/evidence 对齐刀；整体发布仍受 Windows、live/eval 与 DeepSWE verifier
平台证据约束）。

主目标：按 Codex rollout-trace 的生命周期语义，把 CodeCell evidence 接入 Lime 唯一 App Server trace owner，并证明
真实 Electron Gate B 能通过既有 diagnostics 读取该 trace；不新增公开 `ThreadItem`、`RuntimeEvent`、GUI card 或第二套
trace store。

完成结果：`tool-runtime` 发出 typed `CodeCellTraceEvent`，`agent-runtime` CodeMode lifecycle 和 Lime Agent emitter
转发到 App Server `RuntimeEventSink`；`TraceEventWriter` 以 JSONL 保存 summary-only record，内部 `code_cell` reducer/replay
关联 `source_item_observed -> started -> initial_response -> ended -> output_item_observed`，支持 source 晚到、yield 跨 Turn、
nested tool/wait、failed/canceled 自动闭合和 terminal late-event fail closed。源码只记录字符数与 SHA-256；source/output 使用
canonical `item_*` ID，model-visible call 保留 provider call ID；`diagnostics/trace/read` 真实消费 reducer。

真实 Gate B evidence：`.lime/qc/gui-evidence/code-mode-electron-gate-b/code-mode-electron-gate-b-summary.json`；fresh
thread/turn identity 与 Electron -> App Server -> `code-mode-host` process ownership 均由该机器可读文件记录，不在计划中
复制易变运行值。Electron/preload/IPC、App Server、
standalone `code-mode-host` parent、official-host Responses fixture、完整 CodeCell lifecycle、canonical item identity、
summary-only redaction、GUI terminal 与 zero mock/invoke/console/page/provider error 全部通过。

benchmark 适用项已完成：

- `SBX-01/02`：`npm run test:rust:unit -- -p tool-runtime`，356/356；覆盖 Seatbelt/Bubblewrap 权限 lowering、network
  denial、workspace path guard 与 sandbox retry。
- `DSW-05`：`npm run test:rust:unit -- -p agent-runtime provider_token_budget_stops_before_tool_execution_and_next_sampling`，1/1；
  token budget 终止在工具执行和下一次 sampling 之前。
- `DSW-06`：`npm run test:rust:unit -- -p tool-runtime` 已覆盖 `apply_patch` owner contract，`npx vitest run
  scripts/harness/deepswe-adapter.test.mjs` 为 21/21，`npm run harness:deepswe:preflight` 为 Release 20 20 题、61/61。

2026-08-18 收尾复跑：CodeMode Gate B 6/6 脚本合同与真实 Electron fixture 再次通过；DeepSWE adapter 21/21、
Release 20 preflight 61/61、App Server client contract 299 checks、`npm run test:contracts`、
`npm run smoke:agent-runtime-current-fixture`、legacy/scripts governance、`cargo fmt --all -- --check` 与
`git diff --check` 均通过。Rust fresh 复跑统一使用仓库 `npm run test:rust:unit` runner，由
`scripts/lib/rusty-v8-artifacts.mjs` 下载并校验 Codex release 的 V8 archive/binding；tool-runtime 356/356、
agent-runtime DSW-05 1/1 与 App Server 1686/1686 全部通过。直接运行 Cargo 曾遇到的 V8 archive 404 是绕过
仓库 runner 的无效验证路径，不再作为环境阻塞。

验证边界：本轮没有调用 live provider，不生成 DeepSWE score；Pier editable package 指向已删除 source，且本机无 Docker/
Podman/nerdctl/Colima，verifier 仍为 `blocked / environment`。Windows packaged/N-1 与 Developer ID/notarization 继续由
对应平台 runner 提供 evidence。分类：CodeCell trace/replay、diagnostics read 和 Gate B 为 `current`；公开 CodeCell
ThreadItem/GUI card/第二套 RuntimeEvent 为 `dead / forbidden-to-restore`；无 `compat`/`deprecated`。
