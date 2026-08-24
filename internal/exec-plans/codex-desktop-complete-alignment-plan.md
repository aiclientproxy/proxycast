# Codex Desktop 完整对齐执行计划

> status: `active`
> owner: root + 各领域 current owner
> upstream: `/Users/coso/Documents/dev/rust/codex`（当前观测 HEAD：`99660ab3c7b861c916e467581fa9b8723504d66b`）
> 参考原则：`/Users/coso/Documents/dev/rust/codex` 是协议、Thread/Turn/Item、runtime lifecycle、tool/approval 和 GUI 语义的直接实现参考源；Lime 只在产品范围、provider 和桌面宿主边界做显式适配，不以旧矩阵或同名 legacy API 代替对齐。
> 目标：把 Lime 的产品主链、协议字段、GUI 工作区、执行安全和真实 Desktop 证据推进到当前 Codex Desktop 语义；不恢复已退役 runtime、v0 或 `agentSession/*` 兼容主链。

## 1. 事实源与当前阶段

当前阶段：A-Queue、A-Project、`thread/revert`、Strict Review、D-1 provider capability、D-2 MCP event stream 与 D-3 Environment 的 current owner 和公共运行证据已收口，统一进入 `implementation-complete / evidence-pending`。Queue 公共 JSON-RPC 5/5、Project 4/4、Revert 5/5、MCP 3/3、Environment 3/3、Strict Review 1/1、provider capability 1/1 已使用本机缓存 V8 artifact 通过真实公共入口；MCP runtime generation 8/8、MCP owner 162/162、filesystem gateway 4/4、provider rollback prefix 1/1、remote process transport 1/1 与 remote reconnect 1/1 也已通过。`thread/revert` 的 append-only JSONL 序号、active interrupt、cold resume、provider prefix、workspace invariant 和 connection-scoped experimental gate 均有运行证据。当前剩余主缺口分成两类：Queue 与 provider capability 已有真实 Renderer consumer，只差独立 Electron Gate B；Project、Revert、Strict Review、MCP event stream 与 Environment lifecycle 仍缺稳定用户可见消费面，必须先补 GUI 产品闭环、五语言和组件回归，再补各自 Gate B。D-3 另缺远程 process 的公共 Thread/Turn JSON-RPC 与 cold resume 证据；C-3 Windows/MSVC、live provider/verifier 和平台真机证据仍未完成。计划继续保持 `active`，不以 owner fixture 冒充 GUI consumer、Desktop Gate B、public Thread/Turn transport 或 live evidence。

> 最新 D-2 状态（2026-08-24）：active ready barrier、terminated forwarding、Thread archive/delete/unload/connection cleanup、有限 runtime-generation reconnect 与旧 generation shutdown 已落地；公共 App Server JSON-RPC 3/3、MCP runtime generation 8/8、MCP owner 162/162 已在缓存 V8 环境实际运行。D-2 剩余为独立 Electron Gate B；远程 process/cold resume 属于 D-3 Environment 缺口。

唯一产品链：

```text
Renderer typed gateway
  -> Electron Desktop Host / preload（仅宿主能力与 JSONL 转发）
  -> App Server JSON-RPC
  -> RuntimeCore / model-provider / tool-runtime
  -> Thread / Turn / Item projection + durable store
  -> Codex App-style GUI
```

当前基线：

- Lime HEAD：`489499f124cd9154b5b99aa3a420592dc8b2f597`
- Codex HEAD：`daa48072f4f507221da313a748c3f7c551ae5500`
- 旧 parity matrix 锁定 `c9c6c0daa994109cec50fddcb57d076fdf9e738c`，不得继续作为当前完成度事实源。
- 当前工作树已有浏览器/Electron/benchmark 并行改动。本计划首轮只写本计划、协议事实源和协议 owner 文件，避让这些热区。

### 当前上游漂移（2026-08-24）

当前 Codex 工作树为 `99660ab3c7b861c916e467581fa9b8723504d66b`，相对历史矩阵锁定版本新增的能力必须先完成产品裁决和 owner 映射：

| 上游增量 | 当前结论 | 下一步 |
| --- | --- | --- |
| `thread/queue/{add,list,update,delete,reorder,start}`、`thread/queue/changed` | Lime current owner 已建立：RuntimeCore durable queue、App Server JSON-RPC、GUI 只读 projection；owner 与公共 JSON-RPC fixture 已运行通过 | 补独立 Electron Gate B；在 Gate B 前保持 `implementation-complete / evidence-pending` |
| `project/{list,read,create,import,update,move,delete}`、`project/changed`、`thread/project/updated` | Lime current owner 已建立：App Server Project SQLite repository、UUIDv7/opaque cursor、幂等键、排序、Thread assignment 与 connection-scoped experimental gate；公共 JSON-RPC fixture 已运行通过 | 补独立 Electron Gate B；在 Gate B 前保持 `implementation-complete / evidence-pending`，不得用 workspace API 冒充 |
| `thread/revert`、`thread/reverted` | Lime current owner 已建立：exact v2 contract/schema/generated types、Thread 独占串行 scope、RuntimeCore append-only `history.rollback` replacement、provider/read-model/cold hydration effective stream、metadata-only response 与 notification；旧 `thread/rollback` deprecated；真实 `run_json_lines` 5/5 已通过 | 补独立 Electron Gate B；证据完成前不新增 GUI 入口，状态保持 `implementation-complete / evidence-pending` |
| `autoApprovalReview/strictReviewRequired` | Lime current owner 已建立：Guardian strict-review signal、exact DTO/schema/client 与 connection-scoped experimental gate；真实双连接 `run_json_lines` 1/1 已通过 | 补独立 Electron Gate B；不能用普通 warning 代替 |
| `account/bedrock/{discover,setup}` | Lime 有 `amazon-bedrock` provider route，但没有 Codex account-auth setup owner | 与 model-provider credential boundary 做产品裁决；不能把 provider config 表单冒充 Codex account API |
| `mcpServerStatus/list` v2 分页、`runtimeStatus`，以及 `mcpServer/resource/read` origin/connector 字段 | Lime 仍由 v0 MCP Settings 快照承接，未达到 current v2 exact contract | 先迁移 MCP Settings 读链与 App Server handler，再同步 schema/TS/fixture；当前保持 evidence-pending |
| Browser/Computer Use requirements/config 与 realtime existing-call | Browser/embedded host 和模型能力已有 Lime 专属 owner；Codex realtime 仍 product-scope-excluded | 只补 requirements 的边界审计，不复制 Codex TUI/realtime 或建立第二套 computer-use config |

### GUI 产品面与 Gate B 分类（2026-08-24）

| 能力 | 当前 Renderer 事实 | 当前缺口与退出条件 |
| --- | --- | --- |
| Thread Queue | `ThreadQueueStatus` + `useAgentSessionThreadQueue` 已消费 `thread/queue/list` 与 `thread/queue/changed`，稳定 DOM 为 `thread-queue-status` | 只差独立 Electron Gate B：真实创建 Thread、添加 Queue submission、从 GUI 打开同一 Thread，并证明可见队列与 JSON-RPC identity 一致 |
| Provider capability | `ModelSelector` + `ModelProviderCapabilityBadges` 已消费 `modelProvider/capabilities/read`，稳定 DOM 为 `model-selector-provider-capability-panel` | 只差独立 Electron Gate B：真实打开 ModelSelector、显示三项 capability，并证明 `electron-ipc -> app_server_handle_json_lines -> modelProvider/capabilities/read` |
| Project | 只有 protocol/client/Thread assignment，没有项目目录或 Thread 归属的用户可见 owner | 先设计最小 Project consumer，再补五语言、组件回归和独立 Gate B；不得用 workspace 页面或 raw IPC 冒充 |
| Thread Revert | current transport/read model 已完成，但按现有产品裁决尚无 GUI action 或结果状态 | 先补符合“只替换历史、不回滚文件”的可见入口与确认/结果状态，再补五语言、组件回归和独立 Gate B |
| Strict Review | signal 已进入 typed event router，但没有稳定用户可见 strict-review required 状态 | 先在现有 approval/review surface 增加单一可见状态与下一步，再补五语言、组件回归和独立 Gate B |
| MCP event stream | Settings/status 与 typed event stream 已存在，但 stream active/terminated/reconnect 没有稳定用户可见消费面 | 先在现有 MCP 运行明细中投影 lifecycle，不新建第二套状态，再补五语言、组件回归和独立 Gate B |
| Environment lifecycle | typed lifecycle consumer 与 Environment selector 已存在，但远端 connected/disconnected/reconnect 没有稳定可见状态闭环 | 先由现有 Environment selector/status surface 消费同一 projection，再补五语言、组件回归和独立 Gate B |

当前执行顺序：先关闭 Provider capability Gate B，再关闭 Queue Gate B；随后按 Project -> Strict Review -> MCP event stream -> Environment lifecycle -> Thread Revert 的产品面依赖顺序逐项实现。每完成一项即更新本计划与运行证据，不批量提前勾选。

## 2. 交付分层

### P0：Thread 工作区产品面

- active Thread 成为唯一主画布对象，首页首屏优先新建任务。
- canonical timeline 只渲染一次 User/Agent/Reasoning/Tool/Approval/Plan/FileChange/SubAgent/Compaction。
- Composer 统一 send、steer、interrupt、queue、approval、request-user-input。
- 直接迁移 `agentSession/*` 主写链到 `thread/*`、`turn/*`，旧入口只允许 retired guard。
- Thread 生命周期形成 fork、resume、archive/unarchive、revert、steer、interrupt 的 GUI 闭环。

### P1：协议与数据合同

- 从当前 Codex HEAD 重新生成 method/notification/product-scope matrix。
- 对 queue、project、revert、MCP event stream、strict review、Windows setup/warning 做逐项产品裁决。
- 补 Thread/Item 字段 parity：`projectId`、typed source/status/path/cwd/threadSource，以及 delivery、plugin/script、readOnlyHint、typed reasoning effort。
- 验证 `experimental_api` 对 experimental methods/notifications 的 gating 行为；没有行为证据就不能宣称 parity。

### P1：Coding execution 与平台安全

- command/test 默认路径切到统一 sandbox-aware process owner，补实时输出、stdin、interrupt、terminate 和 Workbench 控制。
- 完成 Windows/MSVC、restricted token、workspace write、外部路径拒绝、ACL rollback、进程树终止和大输出证据。
- 补 session-scope approval key、sandbox denied 安全升级重试和规则草案。

### P1：模型、MCP、Environment、Multi-Agent

- 为当前 Codex freeform 模型补 canonical mapping、capability、readiness、model switch Electron Gate B。
- 补 MCP deferred tool expose/recover/resume、event stream 和独立 Gate B。
- 补 Environment instructions、world-state durable full/patch history、typed context-budget terminal。
- 补 Multi-Agent spawn/wait/followup/interrupt 的完整 Electron identity evidence。

### P2：证据与治理

- 受控 smoke 只能证明 product path；补 live provider、verifier artifact、patch match、Pier/DeepSWE evidence 后才能宣称 Desktop coding pass。
- 同步 current/compat/deprecated/dead 分类，修正旧 snapshot 与最新 Gate B 证据冲突。

## 3. 方法分类原则

| 分类 | 规则 |
| --- | --- |
| `current` | exact method/field、current owner、真实消费链和对应 evidence 都存在。 |
| `needs-re-audit` | 上游当前存在，但 Lime 产品范围、GUI 消费或生命周期尚未裁决。 |
| `product-scope-excluded` | Codex account/commerce、remote control、realtime、marketplace/share、external-agent migration 等不属于 Lime Desktop；Environment exec-server 已进入 D-3 current scope。 |
| `deprecated` | Codex 已明确 deprecated，例如 `thread/rollback`；不得恢复。 |
| `dead` | Lime 旧 `agentSession/*`、v0 wrapper、mock fallback 等已由 current owner 替代的路径。 |

## 4. 阶段、写集与退出条件

### A：当前协议矩阵与 Thread/Item contract

写集：`internal/refactor/v1/11-codex-method-product-scope-matrix.md`、协议 fixture、`lime-rs/crates/app-server-protocol/src/protocol/v2/**`、对应 generated contract tests。

退出条件：矩阵 upstream revision 更新；每个新增 method 有分类、owner、evidence；Thread/Item schema diff 有正向和负向测试；`npm run test:contracts` 通过。

当前子进度：

- [x] Item wire 字段对齐：`delivery`、`pluginId`、`scriptPath`、`readOnlyHint`、typed `reasoningEffort`，并从 canonical metadata 投影。
- [x] Thread wire 字段对齐：必需 `status`、必需可空 `projectId`、typed `SessionSource`/`ThreadSource`、`PathBuf` `path/cwd`、`gitInfo`/`historyMode`/`canAcceptDirectInput` 投影。
- [x] Rust schema fixture 与 `packages/app-server-client` generated protocol types 同步。
- [x] Thread/Item round-trip 与 projection 正向断言；`app-server-protocol` 123 tests、schema fixture test、`app-server-client` 34 tests、`npm run test:contracts` 通过。
- [ ] App Server projection crate 定向测试：被本机 `rusty_v8 v150.4.0` aarch64 预编译包 HTTP 404 阻塞，待环境依赖可用后重跑。

### A-Queue：Thread durable queue

写集：`lime-rs/crates/app-server/src/runtime/thread_queue.rs`、`runtime/tests/thread_queue.rs`、`processor/thread_queue.rs`、`tests/thread_queue_jsonrpc.rs`、Queue v2 protocol/schema/generated types，以及 `src/lib/api/agentRuntime/threadQueueClient.ts` 和 GUI Queue status consumer。

当前子进度：

- [x] Codex exact `thread/queue/{add,list,update,delete,reorder,start}` DTO、method catalog、schema、generated TypeScript 与 connection-scoped `experimentalApi` gate。
- [x] RuntimeCore durable owner：FIFO 顺序、重复 client ID、每 Thread 100 条容量、分页默认 25/最大 100、输入 sidecar、多模态 cold hydration、active/interrupted fail-closed 与 Completed/Failed 自动继续。
- [x] App Server processor 与 `thread/queue/changed` projection；GUI 通过 typed list gateway 和通知订阅消费 current Queue projection，不恢复旧 snapshot 写平面。
- [x] Rust RuntimeCore 单测覆盖 CRUD/reorder、重复 client ID、cold restart、多模态、active/interrupted 和 FIFO。
- [x] RuntimeCore/App Server 生命周期回归已写入：archive 后禁止 Queue 读写、unarchive 恢复 durable queue、delete 清理 session/sidecar/event state、cold persisted queue 在 `thread/resume` 后唤醒；Interrupted 语义保持暂停，显式 `thread/queue/start` 仍可启动。
- [x] 公共 JSON-RPC fixture 5/5 通过，覆盖 experimental gate、CRUD/分页/通知/FIFO、中断后显式恢复、多模态 cold restart 和 cold `thread/resume` 自动派发；cold resume 复用 admission 后后台执行/event hub 路径，真实连接可收到 `turn/completed`。
- [~] 独立 Electron Gate B 未完成；在真实 Desktop Host/preload/IPC/App Server/GUI evidence 前保持 `implementation-complete / evidence-pending`。

### A-Project：Project durable owner 与 Thread assignment

写集：`app-server-protocol` Project v2 contract、`app-server` Project processor / canonical SQLite owner / public JSON-RPC fixture，以及 Thread `projectId` start/list/metadata integration；不新增 workspace compat API。

当前子进度：

- [x] Codex exact `project/{list,read,create,import,update,move,delete}`、`project/changed`、`thread/project/updated` DTO、method/envelope、schema、generated TypeScript 与 connection-scoped `experimentalApi` gate。
- [x] Project SQLite current owner：UUIDv7 identity、opaque cursor、position、roots、metadata、时间戳、create/import 幂等、move/no-op/anchor 校验和 delete assignment cleanup。
- [x] Thread 集成：`thread/start.projectId`、`thread/list.projectId` omitted/null/value 三态、`thread/metadata/update.projectId`，assignment 持久化到 canonical `thread.metadata.projectId`；active/archived Thread 均由同一 owner 处理。
- [x] Owner 单测 3/3 通过，覆盖 CRUD/分页/幂等/排序、import 事务原子性、active/archived assignment、delete cleanup 和损坏 metadata fail-closed。
- [x] 公共 `run_json_lines` JSON-RPC fixture 4/4 通过，覆盖方法/字段门禁、双连接通知隔离、CRUD/分页/排序、Thread assignment、import 通知顺序与非法输入拒绝。
- [~] 独立 Electron Gate B 未完成；在真实 Desktop Host/preload/IPC/App Server/GUI evidence 前保持 `implementation-complete / evidence-pending`。

### B：Thread workspace GUI

写集：仅由 GUI owner 接管 `AgentChatWorkspace`、Thread header、timeline registry、Composer/action-required 相关文件；不得在本阶段新开协议 method。

退出条件：固定桌面/窄宽视口下 Thread、状态、下一步和变更位置可识别；同一 Item 不在多个 surface 重复成为主操作；`npm run verify:gui-smoke` 与真实 Electron Gate B 通过。

当前子进度：

- [x] Thread header 优先消费 canonical `thread/read.status`，Topic 状态只作缺失兜底。
- [x] Thread header 保留 `canAcceptDirectInput` 事实字段，并补 canonical status/direct-input 回归断言。
- [x] Thread header 接入现有 `thread/setName` current 链，提供本地化重命名菜单和回归测试。
- [x] Thread header action menu 通过 `AgentRuntimeAdapter -> useAgentSession -> workspace owner` 接入 `thread/archive`；归档成功后清理 active read model 并回到首页。
- [x] Thread header action menu 通过同一 current owner 接入 `thread/fork`；校验新 Thread identity，fork 成功后复用现有 `switchTopic` 打开新 Thread，保留原 Thread。
- [x] `thread/resume` 已由 `AgentRuntimeAdapter.resumeThread`、stream recovery 和 session hydration 消费；上游 `thread/rollback` 明确 deprecated，且不回滚本地文件，当前不新增 GUI 入口。
- [x] 真实 Electron Gate B 已取得：Shell-01 GUI smoke 21/21 assertions；session-history fixture 覆盖 archive/unarchive、restart readback、resume；Claw fixture 覆盖真实 renderer/preload/App Server bridge、输入发送、`turn.completed` 和 read model 完成态。证据见 `.lime/qc/project-gates/codex-alignment-gate-b-20260823/shell-01-electron-smoke/summary.json`、`.lime/qc/gui-evidence/agent-session-history-electron-fixture/agent-session-history-electron-fixture-summary.json`、`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-summary.json`。

### C：执行、审批和 Windows

写集：`lime-rs/crates/tool-runtime/src/execution_process/**`、agent execution owner、approval policy owner、对应 App Server contract 和测试；Windows 平台文件必须由同一 owner 交接。

退出条件：command/test 走统一 process owner；审批复用/拒绝重试语义可恢复；Windows/MSVC 和真机 evidence 覆盖 workspace、外部路径、ACL、终止、大输出；未验证时保持 fail-closed。

当前子进度：

- [x] C-1 已确认 `command/exec` 与 standalone `process/*` 归 connection-scoped App Server + `tool-runtime::execution_process` current owner；实时 raw-byte delta、stdin/PTY、interrupt、terminate、timeout、bounded output、跨连接隔离、响应前 notification barrier 和断连清理均已有 owner 与回归覆盖，不新增第二套 process supervisor。
- [x] C-2 已接入 session-scoped shell approval cache：key 绑定 command、cwd、requested sandbox policy 和 requested permissions；仅 `exec_command` 支持 `allow_for_session`，browser/guardian 不复用 shell cache；provider session close 清理内存 cache，且 cache 命中使用 `RuntimeToolApprovalSource::Reused` 跳过重复用户审批。
- [x] C-2 sandbox-denied retry 已由统一 orchestrator 承接：按 approval policy 判断是否允许 danger-full-access 重试；普通 user approval 复用于 sandbox retry，strict guardian 重新审核，拒绝和 cancel fail closed；新增 Cached 零审批、key 隔离、shell/browser contract 回归测试。
- [x] C-3 Windows 安全证据入口已落到 current `tool-runtime::execution_process` owner：新增 `windows_restricted_execution` integration test，并接入 `.github/workflows/quality.yml` 的 Windows runner；测试矩阵覆盖 workspace write、`.git/.codex/.agents` 与外部路径拒绝、bounded large output、Job Object descendant cleanup/terminate。
- [x] C-3 setup/warning Renderer consumer 已落到 current App Server gateway 与设置页：`readWindowsSandboxReadiness`、`startWindowsSandboxSetup`、`windowsSandbox/setupCompleted`、`windows/worldWritableWarning` 均有 typed API、schema 校验、事件订阅、Elevated/Unelevated 操作和五种 locale 文案；setup completion 不直接提升 readiness，仍以重新读取的真实状态为准。
- [ ] C-3 Windows/MSVC 与真机 evidence：Codex `windows-sandbox-rs` 对齐的 restricted-token、capability SID、TokenDefaultDacl、ACL lease/rollback、STARTUPINFOEX handle allowlist、Job Object descendant cleanup、bounded pipe reader 已存在于 current target-gated owner；仍缺 `x86_64-pc-windows-msvc` 编译、workspace write、外部/metadata path denial、ACL rollback、timeout/descendant kill、large output 和 Electron Gate B 证据，因此 production plan 继续 `SandboxBackendStatus::Planned/enforced=false`，不启用 setup 成功通知。

### D：模型/MCP/Environment/Multi-Agent evidence

写集：各自 current domain owner，不跨 owner 复制 catalog、snapshot 或 GUI 状态。

退出条件：每个能力有 provider request、cold read/recovery、Electron Gate B 或明确 product-scope exclusion；不存在 mock fallback。

当前子进度：

- [~] D-1 Codex `modelProvider/capabilities/read` 已在 Lime v2 protocol、App Server dispatch、schema、generated/typed client、`modelRegistry` facade 与 ModelSelector GUI consumer 建立 exact owner；App Server handler 复用 `model-provider::provider_capabilities`，并从当前 `RuntimeCore` 注入的 config route 读取 provider，未知/配置失败 fail closed；ModelSelector 以“当前运行 provider”语义展示三项能力，避免把全局快照误标成临时选择 provider。provider route 语义回归与公共 JSON-RPC fixture 已补齐；本轮修正 fixture 显式选择 `openai-response`，official Responses 与兼容网关路由 1/1 已实际通过。剩余为独立 Electron Gate B；未知响应继续在 GUI 侧 fail closed。
- [~] D-2 Codex MCP experimental event stream / deferred tool recovery 对齐中。Codex current v2 有 `mcpServer/event/stream/start`、`mcpServer/event/stream/stop`、`mcpServer/event/stream/notification`，并以 `experimental_api` gating；Lime 已补齐 exact DTO、method/envelope、schema fixture、generated TypeScript、protocol round-trip 和 connection-scoped capability gate。当前实现已将 `rmcp` custom `events/stream` 请求和 server notification 接入 session-owned `McpThreadRuntime`，App Server 按 `(connectionId, subscriptionId)` 持有真实转发 task，start 等待 `notifications/events/active` ready barrier，Thread/runtime/transport 结束转发 `notifications/events/terminated`，并在有限窗口内尝试 runtime-generation reconnect；archive/delete/unload/connection close 会统一清理 owner task，启动失败和未知 server、重复 subscription、非当前连接均 fail closed。真实 `run_json_lines` + LocalAppDataSource stdio fixture 已运行 3/3；`runtime_state::mcp_runtime_tests` 8/8、`lime-mcp --lib` 162/162 通过，覆盖旧 generation shutdown、失败保留前代、并发 ensure 和断线恢复。远端 process/cold resume 与独立 Electron Gate B 仍待补。远端已发送 terminated 时不再追加 synthetic terminated，避免重复通知。Lime v2 当前已有 MCP status/elicitation/tool-call 主体，`mcpTool/listForContext`、`mcpTool/search` 仍落在 v0/legacy surface，不能把它们当作 v2 parity；在剩余证据完成前不新增 GUI 入口或 mock fallback。
- [~] D-3 Codex current Environment `environment/add|info|status` 对齐中。Lime 已新增 exact DTO、v2 method/envelope/catalog、schema fixtures、generated TypeScript、protocol round-trip、connection-scoped `experimentalApi` gating、typed App Server client facade、公共 `run_json_lines` fixture，以及 `thread/environment/connected|disconnected` lifecycle notification contract 和 typed lifecycle consumer。真实 Environment public JSON-RPC 3/3 已通过，覆盖 local/unknown/gate 与远端 WebSocket registry；`tool-runtime --test filesystem_gateway` 4/4 已通过，覆盖远程 Read/Glob/Grep/apply_patch 的单一 gateway、断线透传和无 gateway fail-closed。已按 Codex current 将 `environment/info.cwd` 收敛为跨平台 canonical `file:` `PathUri`，native cwd 只在 App Server 边界转换为 URI。当前本地 `environment/status` 返回 local=ready；远端 registry 支持 Pending/Ready/Disconnected/Unknown，远端 URL 校验和连接失败均 fail closed，并具备有限后台健康探测/重连，不在连接异常时伪造 Ready。Thread/Turn 选择现在由 Environment registry 做 environment ID、绝对 cwd、workspace roots 去重和默认 root 规范化，并写入 Thread metadata；选择请求完成时按当前连接状态发射 connected/disconnected 通知，Pending 不伪造 connected。`thread/resume` 从持久化 Thread metadata 恢复 typed selection 并按当前 registry 状态重新发射生命周期通知，损坏 metadata fail closed。远端 registry 已支持 `environments.json` 冷启动重建，状态变化会主动广播到所有选中该 Environment 的 Thread；Environment selection 变更会追加 `world_state` full/patch 到 RuntimeCore canonical event log，重复快照不重复写。typed `RuntimeWorldState` 已新增多 Environment selection，以稳定 ID 顺序携带 id/cwd/workspace roots/primary/status/shell，并按 Codex 多环境 `<environment_context>` 进入 provider；执行链已把 Environment identity 从 ToolCall 贯通到 `LiveExecutionRequest`，本机 process owner 明确拒绝非 local identity。远端 process transport/lowering 与 fs transport/lowering 已接入 current owner；本轮 remote process WebSocket transport 1/1 与 socket-loss reconnect 1/1 实际通过。由于这些仍是 owner 单测，远端 process 的公共 Thread/Turn JSON-RPC、cold restart/resume 和独立 Electron Gate B 继续待补；不以 filesystem 或 owner fixture 冒充完整 Environment 交付，也不新增 mock fallback。

> D-2 子进度覆盖：上一条只读差距描述已由 2026-08-23 lifecycle/recovery implementation slice 更新；active/terminated、Thread cleanup 和有限 reconnect 已实现，剩余仅为可运行 integration、恢复证据和 Electron Gate B。

### E：Desktop coding evidence 与收尾治理

退出条件：controlled 与 live evidence 分开；live provider、verifier artifacts、patch SHA、recovery coverage 全通过；更新架构图、执行计划、事实源和治理回流守卫。

## 5. 必跑验证

- 协议/跨层：`npm run test:contracts`
- Rust：`npm run test:rust:related -- <changed paths>`
- Agent 主链：`npm run smoke:agent-runtime-current-fixture`
- GUI：`npm run verify:gui-smoke`
- 本地门禁：`npm run verify:local`
- 治理：`npm run governance:legacy-report`、`npm run governance:scripts`

高风险变更必须同时保留 Gate A/Gate B 证据，不能用 owner 单测替代真实 Electron 链。

## 6. 当前进度日志

- 2026-08-23：完成当前 Lime/Codex HEAD 只读差异审计，确认 GUI、协议新项、Thread/Item 字段、Windows、模型控制和 Desktop evidence 为主要缺口；确认 Hooks、compaction lineage、CodeMode、controlled smoke 等不应重复列为实现缺口。
- 2026-08-23：建立本计划，当前阶段为 A，下一刀是以 Codex current Rust DTO 为基线更新协议产品范围矩阵并补 Thread/Item contract。
- 2026-08-23：完成阶段 A 的 Thread/Item contract 子刀。直接参考 Codex `app-server-protocol/src/protocol/v2/thread_data.rs` 与 `item.rs`，收敛 Lime v2 DTO、projection、schema 和 TS 生成物；下一刀转入 B 阶段的 Thread workspace GUI 盘点。
- 2026-08-23：完成阶段 B 的第一刀。Thread workspace header 改为优先读取 canonical Thread status，并透传 `canAcceptDirectInput`；24 个受影响 Vitest 用例、TypeScript 检查和 Prettier 检查通过。`npm run test:related` 包装器因将 `electron` 目录误作为输入失败，已改用同配置 `vitest run` 完成验证。
- 2026-08-23：完成阶段 B 的第二刀。Thread header 通过现有 `renameTopic -> thread/setName` current owner 提供本地化重命名菜单；25 个受影响 Vitest 用例、TypeScript 检查和 Prettier 检查通过。`npm run verify:gui-smoke` 因并行 renderer build lock 等待，重试后仍未获得真实 smoke 证据；已终止本轮启动的两组 smoke 进程，待锁释放后重跑。
- 2026-08-23：完成阶段 B 的第三刀。补齐 `sessionClient.d.ts`、`AgentRuntimeAdapter` 可选 archive/fork 能力、`useAgentSession` 的 `archiveTopic`/`forkTopic`、workspace setup/scene/header 接线，并补齐五种 locale 的 fork/archive 文案；App Server session client 对 fork 返回 Thread identity 做 fail-closed 校验。定向 Vitest 74 tests、TypeScript、Prettier 和 `npm run test:contracts`（299 checks）通过。
- 2026-08-23：本轮 `npm run verify:gui-smoke` 已启动至 `build:renderer:electron:smoke`，约 90 秒持续停在 `[electron-renderer-build] waiting for renderer build lock`，以退出码 130 终止本轮等待；当前没有新增真实 Electron Gate B 证据，待并行 renderer build 释放后重跑。
- 2026-08-23：完成 B-2 生命周期语义复核。Codex current 的 `thread/resume` 是恢复/重建 Thread read model 的协议动作，Lime 已在 stream recovery/session hydration 通过同一 current adapter 消费；Codex `thread/rollback` 已 deprecated，且只改历史不回滚本地文件，因此不恢复为新的 GUI 主操作。B 阶段剩余真实 Electron Gate B 证据。
- 2026-08-23：`npm run i18n:check:json` 通过，五种 locale 的 8623 个源 key 全覆盖；`npm run smoke:agent-runtime-current-fixture` 的历史/流式/fixture guard 共 162 个已执行用例通过，随后在重建 renderer fixture 时再次受 renderer build lock 阻塞并以退出码 130 终止；未遗留本轮 smoke 进程。
- 2026-08-23：回收 renderer stale build lock 后，官方 `npm run verify:gui-smoke` 通过；Shell-01 Gate B 21/21 assertions、session-history Electron fixture、Claw current fixture 均通过，B 阶段真实桌面证据关闭。期间修复 Rust projection 中 typed `ThreadSource` 序列化与 `metadata` move 两处编译问题；未覆盖并行浏览器/Electron/benchmark 热区。
- 2026-08-23：C-1 执行链审计确认 `command/exec` 与 standalone `process/*` 已统一到 connection-scoped App Server + `tool-runtime::execution_process` owner，具备实时 raw-byte delta、stdin/PTY、terminate、timeout、output cap、跨连接隔离、响应前 notification barrier 和断连清理；后续不新开第二套 process supervisor，C 阶段下一刀转为 approval cache / sandbox-denied retry / Windows evidence。
- 2026-08-23：完成 C-2 implementation slice。以 Codex current approval cache / sandbox retry 语义为参考，在 `AgentRuntimeState` 增加 session-scoped、内存限定 256 条的 shell approval cache；`current_provider_turn/tool_executor/orchestration` 绑定 shell contract、approval key 和 `allow_for_session` 记录/命中；`tool-runtime::execution_orchestrator` 新增 `RuntimeToolInitialApproval::Cached` 并标记 `Reused`；App Server action response fallback 保留 `decision`/`decisionScope`，避免 session decision 在 live session route 丢失。新增回归覆盖 cache 零审批、command/cwd/sandbox/permissions key 隔离、browser once-only contract 与 session close 清理。`rustfmt`、`git diff --check` 通过；定向 Cargo 仍被 `rusty_v8 v150.4.0` aarch64 macOS 预编译包 HTTP 404 阻塞，未将该测试记为通过。
- 2026-08-23：C-3 对照 Codex `exec-server/src/process_sandbox.rs`、`exec-server/src/process.rs` 与 `windows-sandbox-rs/src/unified_exec/**` 完成 current owner 复核。Lime 已有 restricted-token/ACL/Job Object/pipe lifecycle foundation，但没有 Windows toolchain、Windows runtime 或 Desktop Gate B 证据；继续保持 `Planned/enforced=false`，下一刀只在 Windows runner 上执行平台证据矩阵，不恢复 retired runner/TUI setup 或伪造 `Ready`。
- 2026-08-23：C-3 平台证据矩阵实现入口完成。新增 `lime-rs/crates/tool-runtime/tests/windows_restricted_execution.rs`，直接启动 restricted-token current process owner，覆盖 workspace 可写、`.git/.codex/.agents` 和 workspace 外拒绝、400KB 输出 bounded/truncated、terminate 后 descendant marker 不落盘；`.github/workflows/quality.yml` 的 `windows_shell_runtime` job 新增该测试。macOS 静态 rustfmt、`git diff --check`、`npm run test:contracts` 与 `npm run governance:legacy-report` 通过；本机无 Windows target，尝试 `cargo test -p tool-runtime --test windows_restricted_execution --no-run` 仍在 `rusty_v8 v150.4.0` aarch64-apple-darwin 预编译包 HTTP 404 处阻塞，故不提升 Windows readiness。
- 2026-08-23：完成 C-3 协议 exact contract slice。按 Codex current `windows_sandbox.rs` 对齐 `WindowsSandboxSetupMode`、`windowsSandbox/setupStart`、`WindowsSandboxSetupStartParams/Response`、`windowsSandbox/setupCompleted`、`WindowsSandboxSetupCompletedNotification` 与 `windows/worldWritableWarning` DTO；同步 v2 method/envelope、schema registry、JSON fixtures、generated TypeScript 和 typed request/connection client facade。`cargo test -p app-server-protocol --lib` 124/124 通过；App Server handler、connection-scoped 异步 setup 和真实 Windows readiness 仍未完成，当前不改变 `SandboxBackendStatus::Planned/enforced=false`。
- 2026-08-23：完成 C-3 Renderer consumer slice。`src/lib/api/windowsSandbox.ts` 统一承接 readiness/setup typed request、绝对路径与 setup mode 校验、completion/warning notification projection；App Server frontend client method/types/spec 已注册，`WindowsSandboxReadinessStatus` 接入 Elevated/Unelevated setup、异步 completion、world-writable warning 和 fail-closed readiness refresh；五种 locale 文案同步。定向 Vitest 8/8、`npm run i18n:check:json`、`npm run check:protocol-types`、`npm run test:contracts`（299 checks）、`npm run governance:legacy-report` 和 `git diff --check` 通过。`npm run typecheck` 仍被并行改动的 `useAgentChatWorkspaceSceneRuntime.tsx` 三处既有 `string | null` 错误阻塞；`cargo fmt --check` 仍被并行 Rust 文件格式漂移阻塞，未修改其热区。
- 2026-08-23：D-1 只读盘点确认 Lime 已有 `model/list` 分页目录、typed capability snapshot、provider catalog 和 route capability current owner；与 Codex current 的独立 `modelProvider/capabilities/read` 仍存在 exact contract 差距。该方法下一步应直接复用 `model-provider::provider_capabilities` 计算，不复制第二套 model catalog，也不把 model/list 伪装成替代实现。
- 2026-08-23：D-1 protocol/App Server/typed client slice 完成。新增 `modelProvider/capabilities/read` exact 三布尔字段 contract，App Server 按当前 provider route 复用 `ProviderCapabilities`，未知 provider、配置加载失败和未知响应均 fail closed；protocol 125 tests、app-server-client 124 tests、Lime 相关 Vitest 55 tests、299 项 contracts 通过。App Server crate 公共编译仍被本机 `rusty_v8 v150.4.0` aarch64 archive HTTP 404 阻塞，尚未将 integration evidence 标记完成。
- 2026-08-23：D-1 GUI consumer slice 完成。ModelSelector 打开时通过 `modelRegistryApi.readModelProviderCapabilities()` 读取当前运行 provider 的 exact 三布尔快照，在同一 provider 区域展示工具命名空间、图片生成、网页搜索状态；请求失败和未知响应均 fail closed；新增 provider capability badge 组件、五语言文案和 46 个 ModelSelector/component tests，`i18n:check:json` 全量覆盖通过。补充 App Server route 语义回归（official OpenAI、openai-compatible、unsupported provider）；`model-provider` 4 tests 通过，`app-server` 公共 crate 集成证据仍受 `rusty_v8` archive HTTP 404 阻塞。
- 2026-08-23：D-2 MCP 只读审计确认 Codex current 的 event stream 三件套属于 experimental API，Lime v2 尚未注册对应 method/notification；Lime 当前 deferred tool/list-for-context/search 主要仍是 v0/legacy surface，虽有 inventory metadata 和 fail-closed API 测试，但不能宣称与 Codex 的 v2 event subscription、recovery/resume 语义一致。下一步先建立 method/product-scope matrix 和 gating 回归，再在 current MCP owner 内实现真实订阅链。
- 2026-08-23：完成 D-2 protocol/gating slice。直接对照 Codex `app-server-protocol/src/protocol/v2/mcp.rs` 和 `protocol/common.rs`，新增 `mcpServer/event/stream/start`、`stop`、`notification` 的 exact DTO、v2 method/envelope、schema registry/fixtures、generated TypeScript 与 125 个 protocol tests；将 Lime initialize capability 收敛为 wire `experimentalApi`，transport 按连接保存 capability，公共 JSON-RPC 入口在未声明时返回 `INVALID_REQUEST`，补 App Server public-entry regression。`npm run test:contracts`（299 checks）、`check:protocol-types`、rustfmt、diff check 通过；App Server crate 集成测试仍受本机 `rusty_v8 v150.4.0` aarch64 archive HTTP 404 阻塞。下一刀是 current MCP subscription owner 与 thread/cold recovery evidence，继续保持无 GUI 入口、无 mock fallback。
- 2026-08-23：完成 D-2 subscription owner implementation slice。参考 Codex `app-server/src/request_processors/mcp_event_stream.rs` 与 Lime `rmcp 0.12` current client service，将 server custom notification 原样投影到 `McpServerEventStreamNotification`；`McpThreadRuntime`/`McpClientManager` 以 session+thread generation 持有 broadcast owner，App Server 按连接和 subscription id 管理 start/stop task，start 强制当前连接已订阅 canonical Thread 且 server 已在该 Thread runtime 运行，stop/transport close/receiver close 清理任务。验证：`cargo test -p app-server-protocol --lib` 125/125、`npm run test:contracts` 299 checks、rustfmt、`git diff --check` 通过；`cargo check` 的 Lime MCP/App Server 编译仍被本机 `rusty_v8 v150.4.0` aarch64 archive HTTP 404 阻塞，故不提前宣称公共 integration 或 Electron Gate B 完成。下一刀是 active/terminated readiness、runtime 替换后的 cold recovery/resume/reconnect 和公共入口 integration test。
- 2026-08-23：完成 D-2 lifecycle/recovery implementation slice。对照 Codex `request_processors/mcp_event_stream.rs` 与 `core::CodexThread::start_mcp_event_stream`，Lime current MCP owner 新增真实 `events/stream` custom request、session-owned `McpEventStream` request/notification route、active ready barrier、terminated projection、runtime generation replacement 的有限 reconnect；App Server stream task 在 Thread archive/delete/unload、transport close 和 connection cleanup 时统一 abort，启动失败/超时 fail closed。验证：Rustfmt、`git diff --check`、`npm run test:contracts`（299 checks）通过；`cargo check -p lime-mcp -p lime-agent -p app-server --lib` 仍被本机 `rusty_v8 v150.4.0` aarch64 archive HTTP 404 阻塞，未将公共 integration 或 Electron Gate B 记为完成。下一刀是恢复环境后补 App Server public JSON-RPC integration、MCP current fixture 和独立 Electron Gate B。
- 2026-08-23：补入 D-2 public JSON-RPC integration/current fixture。新增 `mcp_exact_jsonrpc` 的真实 `run_json_lines` 场景：通过 LocalAppDataSource 安装 stdio Plugin MCP server、启动 Thread、声明 `experimentalApi`、等待 `notifications/events/active` 后转发 event，并通过 stop/transport close 清理；已有 transport-level 未声明 capability fail-closed 回归继续覆盖门禁。另将重复 subscription 检查前移，避免重复请求在正常路径触达远端。`rustfmt` 与 `git diff --check` 通过；定向 `cargo test -p app-server --test mcp_exact_jsonrpc --no-run` 仍因 rusty_v8 archive HTTP 404 未执行，故 integration/current fixture 只记为代码已写入、运行证据待环境恢复，不标记 D-2 完成。
- 2026-08-23：修复 D-2 终止通知生命周期边界。若远端已发送 `notifications/events/terminated`，App Server stream task 记录该事实并直接结束，不再追加 synthetic terminated；只有连接、Thread 或 runtime 在未收到远端终止事件时关闭，才发送一次合成终止通知。`rustfmt --check` 与 `git diff --check` 继续通过。
- 2026-08-23：完成 D-3 Environment public integration fixture slice。新增 `environment_jsonrpc`，通过真实 `run_json_lines` 覆盖 `initialize` 能力声明、`environment/info` local shell/cwd、`environment/status` ready/unknown、未知 `environment/info` 错误和远程 `environment/add` fail-closed；第二个场景验证未声明 `capabilities.experimentalApi` 时 Environment 方法统一返回 `INVALID_REQUEST`。新测试已通过 rustfmt/diff check，但本机 App Server crate 编译仍被 `rusty_v8 v150.4.0` aarch64 archive HTTP 404 阻塞，故仅记录 fixture 已写入，未宣称 App Server integration 或 Electron Gate B 完成。下一刀是恢复编译环境后执行该 fixture，并继续补 exec-server lifecycle/recovery 与 durable world-state history。
- 2026-08-23：按 Codex current `app-server-protocol/src/protocol/v2/environment.rs` 复核并修正 D-3 wire drift：`EnvironmentInfoResponse.cwd` 从普通字符串改为严格 canonical `file:` `PathUri`，拒绝非-file scheme、credentials、port、query、fragment 和 NUL；补 protocol 负向/正向 round-trip，重新生成 schema fixtures 与 TypeScript。`cargo test -p app-server-protocol --lib` 126/126、`npm run check:protocol-types` 和 `npm run test:contracts`（299 checks）通过；`cargo test -p app-server --test environment_jsonrpc --no-run` 重试仍在 `librusty_v8_ptrcomp_sandbox_release_aarch64-apple-darwin.a.gz` HTTP 404 处阻塞。下一刀保持为恢复 App Server integration 运行证据，再推进 exec-server connection lifecycle/recovery。
- 2026-08-23：按 Codex current app-server protocol 与 app-server suite 补 D-3 Environment connection lifecycle contract。新增 `thread/environment/connected`、`thread/environment/disconnected` exact notification DTO、v2 envelope/catalog、schema manifest/generated TypeScript、Rust round-trip（127/127）和 app-server-client lifecycle typed consumer；修复 Agent Runtime lifecycle union 的类型收窄并通过 `npm test --prefix packages/app-server-client`（125 tests）。当前仅完成协议/消费合同，尚未由 Lime runtime 发射真实 connected/disconnected；下一刀仍是 Environment registry + exec-server transport owner。
- 2026-08-23：完成 D-3 Environment registry/transport implementation slice。按 Codex `exec-server/src/environment.rs`、`client.rs` 和 `app-server/src/request_processors/environment_processor.rs` 重建 Lime current owner：`environment/add` 校验并替换远端 WebSocket registry entry，后台执行 `initialize`/`initialized`/`environment/info`/`environment/status` 握手；`environment/info` 等待初始连接并返回 canonical `PathUri`，`environment/status` 对 ready 连接使用现有 socket 做 fail-fast health probe，连接失败映射为 `disconnected`，不伪造 ready。新增公共 `environment_jsonrpc` WebSocket fixture，覆盖 remote add、Pending->Ready、info 和 server close 后 disconnected；fixture 已 rustfmt/diff check，但 App Server crate 仍因本机 `rusty_v8` v150.4.0 arm64 archive HTTP 404 无法编译运行。协议 127/127、app-server-client 125 tests、contracts 299 checks 通过。仍缺真实 Thread environment selection/lifecycle notification 发射、reconnect/resume、durable world-state history 和 Electron Gate B。
- 2026-08-23：完成 D-3 Thread/Turn environment selection implementation slice。按 Codex `resolve_turn_environment_selections` 与 `ThreadManager::validate_environment_selections` 语义，Environment registry 现在校验非空且唯一的 environment ID、绝对 POSIX/Windows cwd、workspace roots；roots 去重且缺省回填 cwd。`thread/start` 将规范化选择写入 canonical Thread metadata，`turn/start` 复用同一 lowering；local/ready 选择发射 `thread/environment/connected`，已知断线选择发射 `thread/environment/disconnected`，Pending 不伪造 connected。新增 registry 单测覆盖默认 root、重复 ID 和相对路径拒绝；rustfmt、git diff check 通过。App Server 定向编译仍被本机 `rusty_v8 v150.4.0` aarch64 archive HTTP 404 阻塞，因此未把 lifecycle 运行证据标记完成。下一刀是把 selection 绑定到 Thread resume/cold hydration，并实现远端连接主动断线、reconnect/resume 与 RuntimeCore 执行 lowering。
- 2026-08-23：完成 D-3 persisted selection resume slice。`thread/resume` 现在从 canonical Thread metadata 读取 typed environment selections，复用 registry 校验/规范化，并按当前 Ready/Disconnected/Pending 状态恢复 lifecycle 通知；损坏 metadata 或冷启动后缺失的远端 registry entry 均 fail closed。新增持久化 selection 解析回归。当前只关闭本地/已重建 registry 的 resume 绑定，远端 environment config 尚未进入 durable owner，因此 cold restart 自动重建、主动断线转发和 reconnect 仍未完成。
- 2026-08-23：完成 Codex product-scope/governance convergence slice。D-1 current `modelProvider/capabilities/read`、D-3 Environment methods/notifications 已从过时 excluded/dead guard 收回 current matrix，产品范围矩阵更新为 `146 implemented / 3 planned / 72 excluded`；新增 current manifest 同名契约回归。`npm run governance:legacy-report` 边界违规从 13 降为 0，范围守卫 9/9 通过；同步运行 `npm run generate:protocol-types` 后 `npm run test:contracts` 稳定通过。下一刀仍回到远端 registry cold restart、主动断线/reconnect 与 RuntimeCore lowering，不把 governance 收口当作 execution parity。
- 2026-08-23：完成 D-3 bounded reconnect implementation slice。Environment registry 为每个远端 entry 增加 retired generation、后台健康探测和 2 秒间隔的有限 reconnect；替换同 ID entry 会停止旧 watcher，连接失败保持 `Disconnected/Pending`，不伪造 `Ready`。这只覆盖 transport-level recovery，尚未把状态变化广播到每个已选 Thread，也未实现 cold-start registry hydration 或 RuntimeCore 远端执行 lowering。
- 2026-08-23：完成 D-3 registry durable/recovery slice。Environment registry 增加配置根 `environments.json` 的原子持久化与启动 hydration；只恢复通过 ID/URL 校验的远端条目，连接仍从 `Pending` 开始。registry status event bus 接入 App Server current notification hook，Ready/Disconnected 变化主动投影给所有选中该 Environment 的 Thread；archive/delete/unsubscribe 清理 Thread 选择索引。Environment selection 现在由 RuntimeCore canonical event log 追加 `world_state.v1` 的 environments full/patch，按 session 去重并保留完整 state，供冷启动重放。新增代码已 rustfmt、`git diff --check` 通过；`app-server-protocol` 127/127 与 `packages/app-server-client` 125 tests 通过。App Server crate 编译仍被本机 `rusty_v8 v150.4.0` aarch64 archive HTTP 404 阻塞，故不提前宣称 integration、完整 RuntimeWorldState provider lowering 或 Electron Gate B。下一刀是接入 RuntimeCore 真实远端 process/filesystem executor lowering，并为持久化 registry、广播和 world-state full/patch 补定向单测。
- 2026-08-23：完成 D-3 typed provider context 与 execution identity foundation slice。`agent-protocol::RuntimeWorldState` 新增 typed 多 Environment selection，App Server 在 `turn/start` 从 registry 捕获 id/cwd/workspace roots/primary/status/shell 请求级快照，provider renderer 按 Codex 单环境/多环境形态输出；primary cwd 不再被本机 workspace cwd 覆盖。`ToolCall -> RuntimeToolExecutionContext -> RuntimeUnifiedExecToolRequest -> LiveExecutionRequest` 贯通 Environment identity 和对应 cwd，`ExecutionProcessServer` 对非 `local` identity 明确返回 `UnsupportedEnvironment`，禁止静默本地执行。`agent-protocol` world-state 4/4、`git diff --check` 通过；Agent Runtime/App Server 测试仍在既有 `rusty_v8` HTTP 404 处阻塞。D-3 保持进行中，下一刀是让 current Environment registry 实现并注入 Codex exec-server `process/*` 与 `fs/*` transport。
- 2026-08-23：完成 D-3 远程 process transport implementation slice。新增 Environment 单一 WebSocket reader/writer actor 的 `process/start|read|write|signal|terminate` 并发请求；远程进程接入现有 `ExecutionProcessServer`，通过 `process/read` 轮询转换为统一 `ExecutionOutputDelta`，并写回统一 `ExecutionProcessSnapshot`、background terminal、terminate/signal/status 清理路径。remote sandbox lowering 只接受 Codex 已知 `read-only`、`workspace-write`、`danger-full-access`，未知策略返回 sandbox denial；新增 `RemoteProcessCommand: Debug` 与 wire lowering 回归。`cargo fmt --check`、`git diff --check` 通过；App Server crate 定向编译仍被 `rusty_v8 v150.4.0` aarch64 archive HTTP 404 阻塞。当前实现属于 `current` process owner，但尚未宣称远程 process 公共 JSON-RPC fixture、cold restart/reconnect/resume 或 Electron Gate B；下一刀是同一 Environment owner 的 Codex `fs/readFile|writeFile|open|readBlock|close|getMetadata|readDirectory|walk|canonicalize` transport 与 RuntimeCore/tool-runtime lowering。

- 2026-08-23：D-3 filesystem lowering slice 开始。新增写集限定为 `tool-runtime` filesystem gateway、Environment registry 的 `fs/*` lowering、RuntimeBackend/AgentRuntimeState 注入和远程 `Read/Glob/Grep/apply_patch` 执行分支；本地工具路径保持 current owner，不新增 renderer/Electron fallback。退出条件是远程工具调用按 Environment identity 使用 `fs/*`，未知或断线 Environment 继续 fail closed，补 gateway/remote fixture 回归；当前尚未完成，App Server crate 运行证据仍受本机 `rusty_v8` aarch64 archive HTTP 404 阻塞。
- 2026-08-23：完成 D-3 filesystem gateway/lowering implementation slice。`RuntimeFileSystemGateway` 作为 `tool-runtime` current filesystem owner 接入 Environment registry；远端复用单一 WebSocket actor 的 `fs/readFile|writeFile|open|readBlock|close|getMetadata|readDirectory|walk|canonicalize|createDirectory|remove`，`Read/Glob/Grep/apply_patch` 按 ToolCall Environment identity 走 gateway，本机 owner 保持不变。补充测试专用 `filesystem_gateway` fixture，覆盖远程 Read 不读本机、Glob/Grep 使用 walk/read、apply_patch 使用 gateway 与断线错误透传；远程 Environment 缺 gateway 时四条工具路径统一 fail closed。修正远程 filesystem sandbox：读取默认 `read-only`、写入/apply_patch 默认 `workspace-write`、`danger-full-access` 才允许 unrestricted，未知策略返回错误。rustfmt、`git diff --check` 通过；`cargo test -p tool-runtime --test filesystem_gateway` 已启动但在既有 `rusty_v8 v150.4.0` aarch64 archive HTTP 404 处阻塞，fixture 尚无运行证据。下一刀是恢复依赖后运行 gateway fixture，并扩展 App Server WebSocket fixture 覆盖真实 `fs/*` 请求和远程 apply_patch 的 create/remove。
- 2026-08-23：补入 D-3 EnvironmentRegistry 真实 WebSocket filesystem fixture。单测 server 覆盖 `initialize`/`environment/info`/`environment/status` 后，真实调用 registry gateway 的 `fs/readFile`、`fs/writeFile`、`fs/getMetadata`、`fs/canonicalize`、`fs/readDirectory`、`fs/walk`、apply_patch 所需 `fs/createDirectory` 与 `fs/remove`，并断言每个方法都经过同一连接。代码已 rustfmt、`git diff --check` 通过；App Server/tool-runtime 运行仍受 `rusty_v8` v150.4.0 aarch64 archive HTTP 404 阻塞，未宣称 fixture 运行证据。
- 2026-08-23：完成 D-3 filesystem gateway ownership slice。删除 `tool-runtime` 进程级 `OnceLock` gateway，改由 `RuntimeCore -> RuntimeBackend -> AgentRuntimeState -> RuntimeToolStepSnapshot -> RuntimeToolExecutionContext` 显式注入同一 Environment filesystem gateway；普通本机/单测 snapshot 默认无 gateway，CodeMode nested tool 复用当前 step snapshot，避免多个 App Server/test 实例互相污染。`RequestProcessor` 只在 RuntimeCore backend 接受注入时启用远程 Environment lowering；未知/不支持 backend 继续 fail closed。rustfmt、`git diff --check` 通过；`cargo check -p agent-runtime -p lime-agent --lib` 与 `cargo check -p tool-runtime --lib` 均在既有 `rusty_v8 v150.4.0` aarch64 archive HTTP 404 处阻塞，未将编译或 fixture 运行标记为完成。下一刀是依赖恢复后执行 tool-runtime filesystem fixture、App Server environment fixture，并补远程 process 公共 JSON-RPC 与 cold restart/reconnect/resume 证据。
- 2026-08-23：补入 D-3 Environment reconnect recovery fixture。真实 WebSocket server 接受两次连接，第一次完成 `initialize -> initialized -> environment/info -> environment/status` 后主动断开，第二次完成同样握手；registry 断线探测和 2 秒 bounded reconnect 断言 `Ready -> Disconnected -> Ready`，不伪造 Ready。远程 process 公共 fixture 已存在于 `execution_process/tests.rs`，本轮不重复实现。定向测试启动仍在 `rusty_v8 v150.4.0` aarch64 archive HTTP 404 处阻塞，保留为 evidence-pending。
- 2026-08-23：补入 D-3 cold-start Environment fixture。测试直接写入合法 `environments.json`，创建新 `EnvironmentRegistry` 后由 hydration 自动发起真实 WebSocket `initialize -> environment/info -> environment/status` 握手，并断言冷启动连接进入 `Ready` 且 cwd 保留 canonical `file:` URI。App Server 运行证据仍受 `rusty_v8` archive HTTP 404 阻塞。
- 2026-08-23：收紧 D-3 apply_patch 最小权限 lowering。远程 UpdateFile 先以 `read-only` sandbox 读取源文件，再以 `workspace-write` 写入目标、创建父目录和删除旧路径；避免把读取请求错误标成 `write`。rustfmt、`git diff --check` 继续通过。
- 2026-08-23：修复 D-2 runtime replacement cleanup 缺口。`AgentRuntimeState::ensure_mcp_runtime_generation` 现在只在新 generation 启动成功后发布替换，并在释放 runtime map 锁后显式 shutdown 旧 generation；线程不匹配失败路径也会关闭刚启动的临时 generation，避免旧/临时 stdio 连接和事件路由泄漏。新增 `replacing_generation_shuts_down_previous_servers` Node stdio fixture 回归；`rustfmt` 与 `git diff --check` 通过。定向 `cargo test -p lime-agent runtime_state::mcp_runtime_tests --lib` 仍因 `rusty_v8 v150.4.0` 在 `aarch64-apple-darwin` 缺少预编译 archive（GitHub HTTP 404）阻塞，未将该回归标记为运行通过。`npm run test:contracts`、`npm run governance:legacy-report` 和 `cargo test -p app-server-protocol --lib` 本轮已重新启动，仍需读取各命令最终退出结果；D-2/D-3 公共 App Server 与 Electron evidence 继续保持 `evidence-pending`。
- 2026-08-23：完成本轮验证收口。`cargo test -p app-server-protocol --lib` 127/127 通过；`npm run test:contracts` 299 checks、命令/脚本/发布流程/文档边界守卫全部通过；`npm run governance:legacy-report` 扫描 2048 文件、边界违规 0；`npm test --prefix packages/app-server-client` 完成 TypeScript build 与 app-server-client 测试。`cargo test -p lime-mcp --lib` 与 `cargo test -p lime-agent runtime_state::mcp_runtime_tests --lib` 均在同一 `rusty_v8 v150.4.0` `aarch64-apple-darwin` archive HTTP 404 处阻塞，未把 MCP replacement 回归标记为运行证据。计划仍为 `active`：D-2/D-3 的公共 App Server fixture、远程 process/filesystem 运行证据、cold restart/resume、独立 Electron Gate B，以及 C-3 Windows/MSVC/真机和 E live evidence 未完成。
- 2026-08-23：修复 D-2 replacement recovery 分支。App Server MCP event stream task 在订阅已进入 active 后，将旧 runtime shutdown 导致的 `McpEventStream::recv()` transport `Err` 与 clean EOF 一并纳入有限 reconnect；订阅尚未 active 时仍 fail closed，不会把启动错误误当恢复。`rustfmt --check` 与 `git diff --check` 通过；公共 App Server integration/Electron 运行证据仍受 V8 archive 阻塞，未标记为完成。
- 2026-08-23：补强 D-2 generation startup fail-closed。`ensure_mcp_runtime_generation` 在 `McpThreadRuntime::start` 任一失败（包括 bridge snapshot 构建失败）时显式 shutdown 临时 generation，确保部分启动的 MCP server 不会泄漏；随后才进入 replacement publish。`rustfmt --check` 与 `git diff --check` 通过。
- 2026-08-24：继续治理与验证收口。`runtime.rs` 的 `ExecutionBackend` trait 已迁入 `runtime/execution_backend.rs`，provider route/base-url helper 已归 `runtime/model_providers.rs`，runtime 文件降至 558 行；`processor/mod.rs` 的 MCP event-stream cleanup、Environment runtime/selection、通知 hook 与 connection helper 已迁入 `processor/mcp.rs`、`processor/environment.rs`、`processor/notifications.rs`，facade 降至 769 行，低于 800 行治理阈值。同步 item inventory 的 `commandExecution.pluginId/scriptPath` 与 `mcpToolCall.readOnlyHint` 字段事实；`npx vitest run src/lib/governance/appServerRuntimeBoundary.test.ts src/lib/governance/agentItemInventoryBoundary.test.ts` 29/29 通过，`cargo fmt --check`、`git diff --check` 通过。`cargo check -p app-server --lib` 仍在本机 `rusty_v8 v150.4.0` `aarch64-apple-darwin` 预编译 archive HTTP 404 处阻塞；未将 Rust 编译、D-2/D-3 公共 App Server integration 或 Electron Gate B 记为完成。计划保持 `active`，下一刀是依赖恢复后执行公共 fixture，并继续 C-3 Windows/MSVC/真机与 E live evidence。
- 2026-08-24：完成本轮全量前端回归收口。smart Vitest 在修复 Browser pending intent 可选字段形状后从第 110 批续跑至第 116 批，全部通过；无历史投影时 `historicalProjection` 不再显式写入 `null`，保留既有 exact-shape 合同，有历史数据时仍投影只读历史事实。随后 `npm run test:contracts`（299 checks）、`npm run governance:legacy-report`（2048 文件、边界违规 0）、`npm run governance:scripts`、`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check` 与 `git diff --check` 全部通过。Rust App Server 集成、D-2/D-3 公共 fixture 运行、Windows/MSVC/真机、独立 Electron Gate B 与 E live provider/verifier 证据仍未完成，计划继续保持 `active`。
- 2026-08-24：完成当前 Codex HEAD 漂移只读审计。上游从历史锁定 `c9c6c0d` 到当前 `99660ab3` 新增 24 个方向化 method：实验性 `thread/queue/*`（7）、`project/*`（7）、`thread/revert`/`thread/reverted`、`autoApprovalReview/strictReviewRequired`、`account/bedrock/{discover,setup}` 与 MCP event stream（3，Lime 已有实现）。同时发现 MCP status 已从 Lime 当前 v0 `servers` 产品快照漂移为 v2 `data/nextCursor`、`detail/threadId`、`runtimeStatus` exact contract；MCP resource read 新增 `originCallId/connectorId`，Thread/Item/permissions/config 还有字段漂移。Queue、Project、revert、strict review、Bedrock 尚无 Lime current owner；Browser/Computer Use requirements 属于配置策略面，需产品范围裁决；realtime existing-call 仍为 excluded。历史矩阵与 fixture 不在本条审计中擅自改 hash，以上全部登记为 open gap，计划保持 `active`，不得把旧 98.0% 产品范围数字当作当前 HEAD 完成度。
- 2026-08-24：完成 MCP status/resource current-contract implementation slice。按 Codex HEAD v2 DTO 新增 `McpServerStatus`、分页 `ListMcpServerStatusParams/Response`、`runtimeStatus`/`detail`/`threadId`、server/tool/resource/template typed inventory，以及 `mcpServer/resource/read` 的 `originCallId`/`connectorId`/`originCallId` response 字段；同步 v2 envelope/method ingress、schema registry/fixtures、generated TypeScript 和 App Server typed client。LocalAppDataSource 通过同一 MCP manager 生成状态页并执行 cursor/limit/detail lowering；RuntimeCore tool inventory 改读 v2 data，旧状态 dispatch 已从生产主链移除。Renderer Settings API 读取 v2 data 后只在配置 owner 做 identity lowering，旧配置 CRUD 仍走其 current owner。验证：`cargo test -p app-server-protocol --lib` 127/127、`npm run check:protocol-types`、`npm run test:contracts` 299 checks、MCP 前端定向 34/34、`cargo fmt --check`、`git diff --check` 通过。App Server crate 公共编译和 JSON-RPC fixture 仍受 `rusty_v8 v150.4.0` arm64 预编译 archive HTTP 404 阻塞，未将运行证据标记完成；旧 v0 status schema/手写客户端 DTO 仍列为后续治理清理项。
- 2026-08-24：进入 A-Queue protocol implementation。确认 Lime 没有 Queue durable current owner，`thread-store` 也没有 queue trait；本轮只落 Codex exact `thread/queue/{add,list,update,delete,reorder,start}` DTO、method catalog、v2 request/response/notification envelope、schema registry 与 generated TypeScript ingress，并复用 connection-scoped `experimentalApi` gate。Queue persistence、dispatch、turn execution、archive/delete/resume 生命周期和真实运行证据仍未完成，在 owner 建立前不得标记 `implemented`。
- 2026-08-24：完成 Strict Review 与 connection-scoped experimental API 收口。`autoApprovalReview/strictReviewRequired` 已有 exact DTO、method/envelope、schema registry/generated TypeScript、typed client validator/type guard；Guardian strict-auto-review 同时投影 `item/autoApprovalReview/started` 与 strict-review signal，并归入 Agent Runtime signal router。`AppServerEventBridge` 注入共享 `TransportExperimentalApi`，对 Queue、Strict Review、process、Environment、MCP stream、moderation 等 experimental notifications 在最终 connection send 边界 fail closed；双连接回归证明未声明 `experimentalApi` 的连接收不到 Queue/Strict Review 通知。协议 129/129、app-server-client 129/129、Queue 前端定向 7/7、contracts 299 checks 通过。
- 2026-08-24：完成 A-Queue RuntimeCore 生命周期实现与回归写入。新增 Queue archive/unarchive/delete/cold-resume integration cases；`thread/resume` 仅在最近非 Queue turn 不是 Canceled 时唤醒冷持久化队列，Interrupted 继续暂停，显式 `thread/queue/start` 可恢复；delete 清理 durable event/projection/sidecar 后同 ID Thread 重建为空。四个 Queue Rust 文件已 `rustfmt --check`、`git diff --check` 通过。`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server --test thread_queue_jsonrpc --offline` 仍在 `rusty_v8 v150.4.0` `aarch64-apple-darwin` archive HTTP 404 处阻塞，未把 fixture 运行标为完成。
- 2026-08-24：完成 A-Project current owner 与公共运行证据。Project SQLite repository、Thread assignment、七个 experimental method、两个 experimental notification 和三个 Thread Project 字段 gate 已落到唯一 App Server JSON-RPC 主链；新增 owner 回归与公共 `run_json_lines` fixture。使用本机缓存的 `rusty_v8 v150.4.0` archive/binding 显式注入后，Project owner 3/3、公共 JSON-RPC 4/4、`app-server-protocol` 130/130、`npm run check:protocol-types`、`npm run test:contracts` 299 checks、Rust fmt 与 `git diff --check` 全部通过。Project 状态提升为 `implementation-complete / evidence-pending`，剩余交付缺口为独立 Electron Gate B；整体计划仍保持 `active`，下一刀优先清理同一 V8 阻塞下的 Queue/MCP/Environment 公共 fixture，再进入 `thread/revert`。
- 2026-08-24：关闭 A-Queue 公共运行证据缺口。首次运行 `thread_queue_jsonrpc` 暴露 fixture 未注入 canonical store，以及 cold resume 同步模式把 runtime events 送入空 callback 的真实缺陷；补齐隔离 `ProjectionStore`，并让恢复队列复用 admission 后后台执行/event hub 主链。随后公共 JSON-RPC 5/5 与 cold-resume owner 单测 1/1 通过，覆盖 CRUD/分页/通知/FIFO、中断暂停、多模态重启与真实 `turn/completed` 投影。Queue 剩余缺口仅为独立 Electron Gate B，状态提升为 `implementation-complete / evidence-pending`。
- 2026-08-24：完成 `thread/revert` implementation slice。按 Codex experimental paginated history replacement 语义，补齐 v2 `ThreadRevertParams`/`ThreadRevertResponse`/`thread/reverted`、request/response/notification envelope、method catalog、schema registry/JSON fixture、Rust/TypeScript typed client 与 connection-scoped `experimentalApi` gate；RuntimeCore 以 append-only `history.rollback` marker 替换 effective event stream，保留原 Thread identity、返回 metadata-only response 与 turns/items backwards cursor，不截断旧 JSONL、不回滚本地 workspace 文件，并在 active turn 时复用 interrupt/cancel contract。公共 `thread_revert_jsonrpc` 已覆盖 replacement、notification、cursor、items hydration、repeated revert、active interrupt、cold resume/workspace invariant 与 missing-turn exact error（5 个场景已写入）；新增真实 `run_json_lines` 双连接 transport gate fixture 代码，当前因本机 `rusty_v8 v150.4.0` aarch64 archive HTTP 404、`V8_FROM_SOURCE=1` 又需要不可用的 GN/Ninja 网络下载而未运行。新增 provider-history 单测验证 rollback 后保留 prefix、移除 reverted turn。同步更新 `internal/aiprompts/commands.md`，将旧 `thread/rollback` 保持 `deprecated`，不新增 GUI 入口。`thread/revert` 状态为 `implementation-complete / evidence-pending`；下一刀是恢复 V8 依赖后运行公共 fixture，并补独立 Electron Gate B。
- 2026-08-24：完成本轮公共运行证据与 Revert 修复。通过 `target/debug/gn_out/obj/librusty_v8.a` 压缩归档和 registry `src_binding_ptrcomp_release_aarch64-apple-darwin.rs` 注入，`thread_queue_jsonrpc` 5/5、`mcp_exact_jsonrpc` 3/3、`environment_jsonrpc` 3/3、`thread_revert_jsonrpc` 5/5、`runtime::provider_history` prefix 单测 1/1、`tool-runtime filesystem_gateway` 4/4、`lime-agent runtime_state::mcp_runtime_tests` 8/8、`lime-mcp --lib` 162/162 全部通过。Revert 运行期间发现 append-only JSONL 的序号生成错误：rollback 后 effective `StoredSession.events.len()` 小于物理尾序号，导致重复 revert 写入序号回退；已改为从当前持久事件最后 sequence 单调递增，并对溢出 fail closed。active interrupt fixture 改为真实 `run_json_lines` transport，正确观察 `turn/completed(interrupted)` 与 `thread/reverted` 异步通知；cold resume 带 `excludeTurns=true`，符合 paginated exact contract。Provider history 运行证据确认 rollback 后第三轮保留 prefix、移除被撤销第二轮输入。D-2/D-3 公共 integration 与 owner evidence 已不再受 V8 HTTP 404 阻塞；剩余仍为远程 process/cold resume、Queue/Project/Revert/MCP/Environment 独立 Electron Gate B、Strict Review/D-1 provider Electron evidence、C-3 Windows/MSVC/真机和 E live provider/verifier。
- 2026-08-24：关闭 Strict Review 与 D-1 provider capability 公共运行证据缺口。新增 `strict_review_jsonrpc`，通过真实 `run_json_lines` 双连接验证已订阅同一 Thread 时 `experimentalApi` gate、`guardian.review.started -> item/autoApprovalReview/started + autoApprovalReview/strictReviewRequired`、canonical thread/turn identity，1/1 通过。`model_provider_capabilities_jsonrpc` 首次运行暴露 fixture 把默认 `openai` Chat Completions 路由误期望为 Responses hosted capability；fixture 改为显式 `openai-response` 后，official OpenAI Responses 与兼容网关 fail-closed 1/1 通过。D-3 remote process WebSocket transport 1/1、socket-loss reconnect 1/1 也已运行通过；由于仍是 owner 层测试，公共 Thread/Turn JSON-RPC 与 cold restart/resume 继续 evidence-pending。计划仍为 `active`，剩余为各能力独立 Electron Gate B、D-3 public/cold evidence、C-3 Windows/MSVC/真机和 E live provider/verifier。

## 7. 阻塞与避让

- 当前工作树的浏览器/Electron/benchmark 文件存在未知并行改动，本计划不覆盖、不回滚、不删除。
- Windows 真机、live provider、Pier/DeepSWE verifier 属于外部环境证据；在证据可用前保持 `Planned/enforced=false` 或 `evidence-pending`，不得用受控 fixture 冒充完成。
- 普通 Cargo 默认仍会因 `v8 v150.4.0` 的 `aarch64-apple-darwin` GitHub archive HTTP 404 失败；本机已有 `target/debug/gn_out/obj/librusty_v8.a` 和 registry binding，可通过显式 `RUSTY_V8_ARCHIVE` 与 `RUSTY_V8_SRC_BINDING_PATH` 运行定向测试。本轮已用该方式关闭 Queue/MCP/Environment/Revert/provider/filesystem 的公共/owner 运行证据缺口，但它不能冒充 Windows、Electron Gate B 或 live provider 证据。
