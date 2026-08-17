# Lime 全局架构图

状态：current

## 1. 目的与裁决顺序

本文件是 Lime 的唯一全局架构地图。它回答目录归属、跨层数据流、依赖方向、协议边界和验证证据应该落在哪里。新增能力、重大重构、目录调整和架构评审都先以本文件判断 owner；领域文档只能补充具体契约，不能建立第二套运行时或改写这里的边界。

裁决顺序：

1. 当前构建图、协议 schema、运行代码和稳定测试。
2. 本架构图与根 `AGENTS.md`。
3. `internal/aiprompts/` 的领域边界与质量规则。
4. `internal/exec-plans/` 的已确认执行计划。
5. 路线图、研究、历史 evidence 和 Git history。

Agent loop、Thread/Turn/Item、App Server、状态机、工具生命周期、MCP、Skills、Multi-Agent、历史恢复、projection 和交付门禁以 `/Users/coso/Documents/dev/rust/codex` 的 current 结构为准。多模型控制平面的 catalog、model switch、capability、retry/circuit breaker 以 `/Users/coso/Documents/dev/rust/grok-build` 为 primary reference；provider wire 的 endpoint union、canonical content、媒体 lowering 和多协议 stream 可选择性参考 `/Users/coso/Documents/dev/js/opencode`。两者不一致时，runtime ownership 服从 Codex，model control 服从 grok-build，provider wire 机制由 Lime `model-provider` 结合 OpenCode 的协议边界实现；两者都不替代 Lime 的 App Server、ThreadStore 或 GUI owner。

## 2. 仓库目录地图

| 目录                                  | 类型                      | Owner / 职责                                                             | 禁止放入                                                             |
| ------------------------------------- | ------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| `src/`                                | Renderer                  | React 产品 UI、view model、renderer gateway、i18n、局部显示状态          | Rust 业务实现、Electron main 逻辑、provider wire、生产 mock fallback |
| `electron/`                           | Desktop Host              | main、preload、IPC 白名单、窗口/托盘/系统能力、sidecar 生命周期、updater | Agent 状态机、Thread/Turn/Item、模型调用、业务 read model            |
| `lime-rs/crates/`                     | Rust workspace            | App Server、runtime、协议、provider、工具、持久化、领域服务              | 已删除的旧 Rust root 或 Tauri wrapper 的替代品                       |
| `packages/`                           | 可复用 TypeScript package | 跨 Renderer/Host 的 typed client、schema、projection、UI contract        | 仅单个页面使用的产品状态机、Electron main 实现                       |
| `scripts/`                            | 校验与自动化              | 可复用质量入口、fixture smoke、发布/治理脚本                             | 产品业务逻辑、未登记的根级临时脚本                                   |
| `resources/`、`lime-rs/resources/`    | 受版本控制资源            | 内置 skill、模板、静态运行时资源                                         | 运行时生成物、用户数据、机密                                         |
| `internal/aiprompts/`                 | current 工程事实源        | 架构、命令、质量、治理、GUI、目录准入                                    | 历史迁移日记和实现副本                                               |
| `internal/exec-plans/`                | 执行记录                  | 多轮计划、确认记录、进度、证据索引、blocker                              | 未经确认的长期架构规则                                               |
| `internal/roadmap/`                   | 未来规划                  | 阶段目标、优先级、产品路线                                               | current API 或 runtime owner 的唯一说明                              |
| `internal/research/`                  | 研究证据                  | 对照、审计、外部实现分析                                                 | 生产代码或 current 架构定义                                          |
| `internal/test/`、`internal/testing/` | 测试设计                  | 场景、质量矩阵、测试规范                                                 | 产品实现                                                             |
| `docs/`                               | 对外文档站                | 站点内容、配置、静态资源                                                 | 内部工程规则、执行计划、私有 evidence                                |
| `.codex/`                             | Codex 项目配置            | 可复用 skill、项目级 agent 配置                                          | 产品业务实现或第二套架构事实源                                       |
| `extensions/`                         | 独立扩展                  | Chrome extension 等独立宿主实现                                          | Renderer/Agent runtime 的共享业务 owner                              |

根目录的 `package.json`、`forge.config.mjs`、`vite.config.*`、`tsconfig*.json`、`eslint.config.*` 和 `lime-rs/Cargo.toml` 是构建与发布边界。变更它们必须同步锁文件、相关验证和架构图中受影响的边界。

## 3. 前端目录规范

### 3.1 Renderer 启动与页面

```text
src/main.tsx
  -> RootRouter.tsx
  -> App.tsx
  -> components / features / pages
```

| 路径                       | 准入规则                                                                                                                      |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `src/components/<domain>/` | 用户可见组件、组件专属 view model 和渲染/交互测试；不直接执行 App Server 业务逻辑。                                           |
| `src/features/<domain>/`   | 可独立演进的产品域；包含该域 UI、状态投影、domain test 和入口适配。                                                           |
| `src/pages/`               | 独立窗口或路由页面；只做页面组装和路由级边界，不堆领域状态机。                                                                |
| `src/hooks/`               | 跨产品域的 React composition；领域私有 hook 先留在 `components/<domain>/` 或 `features/<domain>/`。                           |
| `src/lib/api/`             | typed App Server / Desktop Host gateway、请求构造和响应 normalization；不得创建平行业务 runtime。                             |
| `src/lib/desktop-host/`    | renderer 对 Desktop Host 的能力探测与受控 test fixture；生产必须走真实 bridge。                                               |
| `src/lib/dev-bridge/`      | `safeInvoke`、HTTP transport、`app_server_handle_json_lines`、可用性和事件监听；旧命令 policy / mock 只能作为治理或测试辅助。 |
| `src/lib/<domain>/`        | 无 React 生命周期的领域 helper、纯 projection、formatter、schema adapter；不得变成无 owner 的杂物层。                         |
| `src/contexts/`            | 跨组件树的 UI context；不持有后端事实状态。                                                                                   |
| `src/stores/`              | renderer 局部交互状态；不能替代 App Server read model。                                                                       |
| `src/i18n/`                | resource key、locale 配置与语言边界；所有用户可见文案从这里消费。                                                             |
| `src/types/`               | renderer 专用稳定类型；跨边界类型优先放 protocol 或 `packages/`。                                                             |
| `src/test/`                | 测试共用 harness、fixture 和 matcher；不放产品实现。                                                                          |

组件测试只覆盖渲染、DOM 交互、hook 生命周期和关键接线。可纯化的筛选、分组、request builder、状态转移、formatter 和 projection 应拆为 `*.unit.test.ts` 覆盖。

### 3.2 Renderer 数据流

```text
UI event
  -> feature/component view model
  -> src/lib/api typed gateway
  -> Desktop Host bridge or app-server client
  -> App Server JSON-RPC
  -> v2 notification / thread/read
  -> pure projection
  -> UI state
```

Renderer 可以临时保存输入框、选择态、展开态和 optimistic UI；不能生成、修补或持久化 Thread/Turn/Item 真相。流完成、取消和失败必须消费 App Server terminal event/read model，禁止通过固定 timeout 或本地合成事件断言完成。

## 4. Electron Desktop Host 规范

| 路径/模块                             | 职责                                                                                           |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `electron/main.ts`                    | Electron 生命周期与 host 组装入口。                                                            |
| `electron/preload.ts`                 | `contextBridge`、最小暴露面和 IPC 调用入口。                                                   |
| `electron/ipcChannels.ts`             | IPC channel 常量与协议白名单。                                                                 |
| `electron/*Host.ts`                   | 一个桌面能力一个 owner，例如 App Server sidecar、文件/项目壳、窗口、通知、更新、浏览器、语音。 |
| `electron/appServerHost.ts`           | sidecar 生命周期与 `app_server_handle_json_lines` 的宿主边界。                                 |
| `electron/forge/`、`forge.config.mjs` | Forge 打包、maker、签名和 release 事实源。                                                     |

Electron 负责窗口、托盘、Dock、系统文件选择、权限、外部链接、自动更新和 sidecar 生命周期。它不得保存业务 session、解释 provider response、执行模型工具、拼 Thread/Turn/Item 或提供业务 mock fallback。

新增 Electron 命令前先判断是否只是已有 `app_server_handle_json_lines` 的转发。只有系统宿主能力才新增 IPC；业务能力一律优先新增 App Server JSON-RPC。

App Server 发起的 reverse JSON-RPC request 仍复用同一 JSONL/stdio、`app_server_drain_events` 与 `app_server_handle_json_lines` 通道。Electron 只负责从 typed connection drain notification/request，并把 Renderer 的 Response/Error 原样写回 sidecar；不得解释 server request method、生成业务 decision、持有 pending waiter 或把 request 降级成 Electron IPC 业务命令。

## 5. TypeScript Package 规范

| Package                             | 职责                                                             |
| ----------------------------------- | ---------------------------------------------------------------- |
| `packages/app-server-client`        | App Server JSON-RPC typed client 与生成协议工件。                |
| `packages/agent-runtime-client`     | 可复用的 Agent runtime client facade，不拥有 Renderer 状态。     |
| `packages/agent-runtime-projection` | 纯事件/read model projection、tool/display schema 等可测试逻辑。 |
| `packages/agent-runtime-ui`         | 可复用的 runtime UI primitives，不拥有 App Server transport。    |
| `packages/agent-ui-contracts`       | 跨 UI 的 schema、contract 和 generated/validated 类型。          |
| `packages/agent-workbench-adapter`  | 工作台与 runtime 的明确 adapter 边界。                           |
| `packages/agent-capability-catalog` | capability catalog 的稳定消费面。                                |
| `packages/lime-cli-npm`             | 发布的 Node CLI package。                                        |

package 只在至少两个独立 consumer 需要稳定边界时创建。单一 Renderer feature、单个 Electron host 或单个 Rust domain 不得先抽 package。`dist/`、`node_modules/` 不是事实源。

## 6. Rust Workspace 规范

### 6.1 App Server 与协议组

| Crate                    | 职责                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------- |
| `app-server-protocol`    | JSON-RPC method、params、result、notification、schema export。                                    |
| `app-server-transport`   | JSONL/transport framing、连接与传输错误。                                                         |
| `app-server-client`      | Rust client。                                                                                     |
| `app-server-test-client` | 测试专用 protocol client。                                                                        |
| `app-server-daemon`      | sidecar/daemon 生命周期。                                                                         |
| `app-server`             | request dispatch、RuntimeCore、host context、ProjectionStore、canonical read model、领域 data source。 |

App Server 是 Renderer、Electron、CLI、Plugin 与 runtime 的唯一跨应用业务协议入口。它可以做 transport、鉴权/初始化、请求编排、host context、projection 和 repository 接线；不能持有 provider-specific wire payload 或把工具实现复制进 handler。

App Server 的 request dispatcher 必须先把各 method handler future 装箱，再在单一 await 点执行；禁止让大型 async `match` 把所有分支 future 内联进同一个 poll 栈。stdio transport 在 `initialize` 完成前保持顺序执行；初始化后每个 client request 由独立 transport task 调度，notification/response 继续内联，并由 request id 关联响应、由 serialization scope 保证同一资源的共享/独占顺序。长 turn、MCP 或宿主 I/O 不得阻塞无冲突的 list/read request。

Desktop command/exec 与 Git 等宿主子进程必须有确定性测试注入和生产 deadline。测试不得读取真实用户 shell rc；
plain directory 的 Git status 先通过 `.git` ancestor preflight 返回，仓库内 Git 命令使用异步 process、`kill_on_drop`
和 5 秒上限，不得在 async handler 中调用无界 `std::process::Command::output`。交互终端必须复用 App Server
`CommandExecServer`，Electron 不得持有第二套 session 状态。

Server-originated request 使用与 client request 分离的 `serverRequest` catalog kind，并按以下方向流动：

```text
runtime/domain producer（私有 domain token）
  -> App Server server-request broker（独立 outer JSON-RPC id）
  -> Electron Desktop Host JSONL forward/drain
  -> Renderer typed server-request dispatcher
  -> JSON-RPC Response/Error（同一 outer id）
  -> App Server exact remove-once waiter
  -> runtime/domain exact resolver（私有 domain token）
```

outer JSON-RPC id 只负责 App Server 到客户端的响应关联，domain token 只负责领域内部 continuation；二者不得互相暴露或从 server/turn/tool/最近活动状态猜测。outer id 必须包含 App Server boot scope，不能只用进程内重置的 counter；pending registration 必须在 wait future 被 abort/drop 时 remove-on-drop。未知、重复、迟到或断连响应只能精确失败或取消原 waiter，不得扫描 pending 表命中其他请求。Renderer 对同一连接的 in-flight 与 settled outer id 都必须 at-most-once，只有 connection reset 才清 tombstone；未注册 method 必须返回 `METHOD_NOT_FOUND`，handler 失败返回 typed JSON-RPC error；生产路径禁止 mock fallback。

每个进入 Renderer 的 server request 还必须有显式 terminal 撤销信号。App Server 在 client Response/Error、domain cancellation 或连接级清理后，向创建 outer request 的同一 connection 有序发送 `serverRequest/resolved { requestId }`；不得广播给其他 client，也不得用新的 pending 表猜 owner。Renderer 必须先记录 resolved tombstone，再决定是否打开 UI，从而覆盖“resolved 早于 request”与同批 notification/request；每个 handler 使用独立 `AbortSignal`，远端 resolved 后静默关闭交互面并禁止迟到的 Response/Error。

### 6.2 Agent Runtime 组

| Crate            | 职责                                                                                                                              |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `agent-protocol` | Thread/Turn/Item、RuntimeEvent、稳定 runtime DTO。                                                                                |
| `agent-runtime`  | 回合生命周期、action-required、队列、取消、stream 和 runtime scope。                                                              |
| `agent`          | current provider turn、session/store adapter、runtime facade、业务级 agent 编排。                                                 |
| `runtime-core`   | 模型路由、上下文 fragment、message/media part、跨 provider 的运行时模型。                                                         |
| `thread-store`   | Codex raw canonical rollout item append、独立 metadata patch、Thread/Turn/Item 的存储、检索、分页、历史、graph/identity/mailbox。 |

会话提交的唯一执行 owner 是 `agent-runtime::session_loop`：每个 session 只有一个串行 actor，FIFO task、regular/review/compact kind、replace/interrupt/shutdown、迟到 completion 防护和 steer/mailbox 分流均在此收口。所有 actor 命令先进入统一 operation envelope；envelope 使用 UUIDv7 identity，并携带可选 client user message identity 与 W3C trace carrier，首次启动和 FIFO promote 必须把同一份不可变 metadata 交给 task context。普通用户输入只通过 typed `UserInput` operation 进入该队列：active regular task 存在时原子 steer，否则启动或排队 candidate task，同一份输入必须成为新 task 的 initial input；App Server 只能消费 actor 返回的 `Submitted/Steered` receipt，不能把 read model 的预判当成调度事实。Review、Compact、ThreadSettings、SetMemoryMode、RefreshMcp、ReloadConfig 与 RunShell 使用各自的 typed task operation，并校验 operation 与 task kind 一致；Approval、UserInput、Permission、DynamicTool 与 MCP elicitation response 使用五类独立 response operation，禁止恢复通用 response operation。durable inter-agent append 成功后，App Server 只向已经存在的 recipient actor 提交 typed `InterAgentCommunication` envelope，保留 message/root/sender/recipient/source-turn identity、kind、result status 与 delivery mode；actor 只发布 durable activity，不复制 store payload，也不得为了通知创建空 actor。child terminal result 必须先完成 canonical Turn 与 durable result mailbox append，再由正常 append、durable replay、restart recovery 三条路径统一发布同一 typed activity；通知失败不能回滚 durable record，也不能创建空 actor。provider boundary 只能由 durable loader 把同一记录转换为 typed runtime input，禁止重新解析自由 JSON。steer 与 mailbox activity 通过 typed watch 发布；mailbox pending 使用单调 generation，loader 成功只确认调用前捕获的 generation，失败或并发新 append 必须继续保持 pending。sampling step snapshot 单调增加 step index；pending context rollover 只由下一次 snapshot 原子消费并推进一次 context epoch，显式推进 epoch 也必须清除 pending rollover，禁止在后续 step 重复滚动。RuntimeCore 只负责把 provider/tool 执行、canonical event 持久化和 durable mailbox loader 接到该 actor；QueueOnly mailbox 只有在 provider step 明确打开 mailbox boundary 后才 append + ack，不能由 AgentControl 或 GUI 另起活动回合旁路。AgentControl wait 必须先向既有 actor 注册 activity receiver，再读取 pending snapshot；steer、mailbox 和 queued admission 由 actor 的 typed watch 唤醒，不能通过固定间隔扫描 RuntimeCore turns。

World state 的 typed DTO owner 固定为 `agent-protocol::world_state::RuntimeWorldState`。当前链路为：

```text
typed RuntimeRequest + session scope + resolved model selection
  -> App Server request_context::turn_context
  -> AgentTurnContext.metadata["world_state"]
  -> agent-runtime::provider_turn typed resolver + renderer
  -> provider-visible contextual user message
```

App Server 当前只投影有明确事实源的 environment（cwd/project root/workspace/thread/turn、
provider/model/reasoning）、permissions（approval/sandbox/web search）、collaboration mode 与 effective
multi-agent mode。`agent-protocol::MultiAgentMode` 复制 Codex exact typed union；effective mode 只从 resolved
reasoning effort 推导：`ultra -> proactive`，其余为 `explicitRequestOnly`。deprecated
`thread/start.multiAgentMode` / `turn/start.multiAgentMode` 只保留 typed wire 形状并被 runtime 忽略，不进入
durable metadata，也不能覆盖 effective mode。没有 typed 来源的 AGENTS/apps/plugins/environment
instructions 和 realtime 必须保持缺失，不得从 prompt 文案、进程环境或 provider 名称猜测。Agent Runtime
必须优先反序列化该 snapshot，并在当前 user input 前只渲染一次；metadata 存在但损坏时 fail closed，只有
非 App Server 调用者缺少 snapshot 时才通过同一 DTO 的 `from_cwd` 生成最小状态。旧的 runtime cwd XML
拼接与 arbitrary-JSON multi-agent 控制面已删除。该链路仍为 `partial`：provider-visible consumption 与
effective multi-agent producer 已落地，但其余 typed producer 与 Codex durable full/patch world-state history
尚未完成。后续只能在同一 DTO owner 补齐，禁止在 App Server、Agent prompt 与 provider lowering 分别拼接
第二套 environment context。

状态模型与 Codex 对齐：Thread 是会话上下文，Turn 是一次执行，Item 是可恢复的输入、输出、工具或审批活动。`ThreadStore` 的 raw canonical append 与独立 metadata patch 是持久化 contract；App Server `ProjectionStore` 只能是该 contract 的实现，queue payload、stream buffer 与 renderer cache 不得反向成为事实源。

#### 6.2.1 Storage alignment target（P1 `in_progress`）

2026-07-19 存储计划选定的目标合同是 `AgentRoot/sessions/YYYY/MM/DD/rollout-*.jsonl` 承载唯一 Thread/Turn/Item durable truth，`archived_sessions` 通过原子 move 表达 archive，SQLite 只保存定位、语义状态或经验证可重建的 read model。production 物理 owner 固定为：`sqlite/state.sqlite` 保存 `canonical_threads` 与 `canonical_thread_spawn_edges`；`sqlite/thread_history.sqlite` 保存 `canonical_turns`、`canonical_items` 与 `canonical_history_applies`；`runtime/projection_1.sqlite` 只保存 `projected_sessions/turns/items` 与 `projection_watermarks`，分类为 `deprecated / rebuildable`。组合事务始终以 state 为 main，按顺序 attach history 和 projection；单文件 constructor 仅供隔离测试，不是 production 路径。跨库不能建立 SQLite FK，因此 thread delete 显式按 item、turn、apply、metadata 顺序清理；history 内部 item -> turn 使用同库 FK。

production `ProjectionStore` 通过显式 `AgentRoot` 注入 `RolloutStore`：新 Thread 按创建时本地日期生成唯一相对 `rollout_path`，metadata 保留 UTC 创建时间，typed `ThreadHistoryChangeSet` 在 SQLite commit 前同步写入 JSONL；跨日续写与 restart 重试沿用原路径，同 sequence 同 fingerprint 幂等、不同 fingerprint fail closed。state/history 为空时，constructor 从 active/archived rollout 单事务重建 Thread/Turn/Item；canonical 已存在而 projection 四表全空时，也会独立从 canonical snapshot 重建 projection。已有非空 projected row 时 no-clobber，不清表、不隐式迁移；history `content_digest` 不匹配时启动 fail closed。用户消息全文只在 rollout，重建 SQLite 只保留现有 summary 上限内的单份摘要。`canonical_threads.last_sequence` 当前只作为跨库 commit watermark 暂留 state；迁入 history projection state 后才允许关闭这项 P1 债务。

flat EventLog 仍是 `deprecated / bounded diagnostics`，旧 `runtime/projection_1.sqlite` 是 `deprecated / frozen-for-removal` read model；二者不得新增 transcript payload、第二 canonical writer或长期兼容读取。启动期 queued recovery 禁止扫描 raw EventLog；projection 中的 queued row 只有在 `(session_id, thread_id)` 能 join current `canonical_threads`，且 production Thread 有非空 `rollout_path` 时才可恢复。启用 rollout 的生产实例不再允许 raw EventLog 重写 canonical ThreadStore；旧 `canonical_threads.rollout_path IS NULL` 行也拒绝继续 append，不导入旧 session/event/DB。Codex v2 `thread/archive` / `thread/unarchive`、macOS JSON-RPC/Gate B 和三库物理 inventory 已完成；P1 仍受 projection reader 退役、`last_sequence` owner 收口、旧 source exact-path cleanup guard 与 Windows Gate B 阻塞，不得标成完成。`sessionFile/*` 生成文件归 `AgentRoot/artifacts/sessions/<session-id>`；bounded text diagnostics 固定为 `AgentRoot/observability/log/lime.log`。其他旧数据只进入 exact-path cleanup manifest；允许迁移的只有模型控制面小型语义状态（provider、加密 key ciphertext、UI state、model preference、active tab）与已下载模型文件。模型控制面 source 只读并按 WAL 事务选取，不复制整库；`HostUserData / HostSessionData / AppDataRoot / AgentRoot / UserHome / Workspace` 的逐路径决策以 `internal/exec-plans/codex-lime-storage-alignment-plan.md` 与 `internal/refactor/data/03-one-to-one-storage-alignment-plan.md` 为准，责任开发者架构确认仍为 `pending`。

P0 的 current 组合根链固定为 `Electron appDataPaths -> APP_SERVER_APP_DATA_DIR + AgentRoot -> App Server StorageRoots/LocalAppDataSource -> domain root`。Windows `HostUserData` 保持 roaming，`HostSessionData` 固定到 `<AppDataRoot>/host-session`；portable/E2E override 必须同时显式注入 AppDataRoot 与 AgentRoot，禁止从 `AgentRoot.parent()` 反推。Product DB 只写 `AgentRoot/lime.db`；diagnostics、support bundle、Soul style-pack 与 MCP OAuth 不得自行解析平台默认根。旧 Product DB 整库复制、通用 migration manifest、启动 cleanup，以及 managed project/workspace/session path 启动迁移均为 `dead / deleted / forbidden-to-restore`；Windows 真机证据与责任开发者确认完成前，P0 继续保持 `in_progress`。

Message、Reasoning 与 Plan 必须遵循 Codex 的 canonical Item lifecycle。用户输入在 `message.created` 时直接形成带 `completed_at_ms` 的 completed UserMessage；provider text/reasoning 的 Start/Delta/End 必须以 canonical Turn + sampling attempt scoped Item identity 贯穿 `model-provider -> agent-runtime -> agent -> App Server`，同一 Turn 的后续 sampling 或同一 Thread 的后续 Turn 均不得复用前一 Item。assistant 只有出现真实正文时才启动 AgentMessage，并由同一 Item 在对应 End/`message.completed` 进入 completed，取消或中断映射为 `Interrupted`；terminal Item 拒绝 late delta。Plan 是独立的 `ThreadItemPayload::Plan`，`plan.delta` 与 `plan.final` 必须按 `(turn_id, revision_id)` 共享稳定 Item identity；delta 只表达流式过程，completed `plan.final` snapshot 是恢复和 GUI 决策绑定的权威内容。Plan parser 按 source Message Item 隔离 buffer，Plan 只记录 `sourceItemId` 而不复用 Message identity。Plan 前的纯空白按 Codex `leading_whitespace_by_item` 语义暂存：后续出现正文时随正文发出，Plan-only 输出则丢弃，因此不得创建空白 AgentMessage 或伪造 `message.completed`。历史恢复、live notification 与 read model 必须保留同一 revision identity、Plan steps/status 与 terminal timestamp，禁止用 `update_plan` Tool Item 或 Renderer 本地状态替代。

AgentMessage 的 canonical 正文只由 `ThreadItemPayload::AgentMessage { text, phase, content_parts }` 持有。`content_parts` 只允许 typed Text 与 Media reference；Media 必须引用 sidecar/artifact URI 并携带 MIME 等可验证元数据，禁止保存 provider raw payload、inline `data:` URI 或 presentation metadata escape hatch。`ThreadStore` 持久化、Codex v2 `thread/read` 与 live canonical event 必须保留相同 part 顺序和 reference；presentation 可以映射为 `contentParts`，但不得在读取或 Renderer 边界从 raw event、metadata、正文文案或第二 read model 补造。

用户输入中的 inline 图片必须在 App Server turn 输入边界先写入 `SidecarStore`，再以 `sidecar://` URI、MIME、bytes、sha256 和 typed sidecar reference 进入 EventLog、Thread/Turn/Item 与 read model；持久化事实源不得保留 base64。provider sampling 只克隆 canonical input/history，并在网络请求前通过 sha256 和大小上限读取 sidecar，瞬时 hydrate 为 provider 所需 data URL，不得把 hydrate 结果写回 session。当前回合的新图片遇到 text-only capability 必须在联网前拒绝；历史图片遇到 text-only capability 按 Codex 语义降为明确占位文本，使后续文本回合可继续。字段名统一复用 `output_refs` owner，禁止在 input writer/resolver 散落新的 sidecar truth key。

canonical 持久化的当前 P1 切片是 App Server `RolloutStore` 承接新 Thread 的 dated JSONL，`ProjectionStore` 暂时同时实现 `thread_store::ThreadStore` 的 state/history repository 与 deprecated read model。`state.sqlite` 只持有 Thread metadata/graph，`thread_history.sqlite` 只持有可由 rollout 重建的 Turn/Item/history apply snapshot；任何 history 表落回 state 或 projection 都是回流违规。`ThreadStore::append_items` 只追加已经 canonical 的 typed item；`update_thread_metadata` 是独立 patch API，append 不从 item 内容推断 metadata。typed `ThreadHistoryChangeSet`、sequence collision、rollback/remove、opaque cursor 和 metadata patch 在过渡期仍必须保持确定性；不得增加 `RuntimeStore` 适配层、第二个 transcript 数据库或 renderer 持久化副本。ThreadStore archive/unarchive 已在同一 AgentRoot 内原子移动 rollout 并更新 state `rollout_path`，重复 move 按 identity 幂等；旧 `agentSession/update` 与 `archive_many` 已删除，公共 owner 是 Codex v2 `thread/archive` / `thread/unarchive`。`thread/delete` 也只允许走 v2 App Server current 链：RuntimeCore 先冻结包含 persisted 与 pending-only descendants 的 deepest-first 子树快照，停止全部 session loop/backend owner，并幂等清理 rollout、event log、sidecar、trace 与 telemetry；随后 `ProjectionStore` 以 `BEGIN IMMEDIATE` 重读并校验快照，在一次 ATTACH transaction 中删除 goal/accounting/outbox、canonical history、projection、spawn graph、Agent identity/mailbox 与 canonical Thread。App Server 在 `{}` 响应后通过 per-thread listener 将 child-to-root `thread/deleted` 广播给发起连接和全部订阅连接，再取消 listener、resume barrier 与双向 connection index；同一子树的 pending server requests 先收到 `serverRequest/resolved` 并以 `REQUEST_CANCELLED` 终止。旧 `agentSession/delete` production method/DTO/helper 为 `dead / deleted / forbidden-to-restore`。每个 Item 的 canonical ordinal 只取该 Item 首次出现时的 Lime outer `AgentEvent.sequence`，后续 lifecycle merge 必须保留首次 ordinal；Tool、Message、Reasoning、Plan 和 import producer 自有 ordinal 均不得进入持久化 ordering，Codex `sourceEventSeq` 只能作为 provenance/metadata，ThreadStore 不得通过 `MAX+1` 或其他 store-side renumbering 生成 ordinal。旧 `thread-store::runtime_store`、`session_repository` 以及 production AgentSession read/list/history fallback 已是 `dead / deleted`；event/app-data fallback 与 Renderer detail synthesis 只允许历史测试 evidence，不得成为 production read path。production App Server 构造必须显式注入 projection/state/history/AgentRoot 四路径；`ProjectionStore::initialize*` 单文件构造只允许隔离测试，`AppServer::new()` 只存在于 unit-test build。

Codex v2 `thread/searchOccurrences` 的唯一读取链是：

```text
App Server thread/searchOccurrences
  -> RuntimeCore read boundary
  -> ThreadStore::search_thread_occurrences
  -> ProjectionStore
  -> history.canonical_turns + history.canonical_items
  -> typed Rust / TypeScript response client
```

搜索只读取已持久化的 canonical history，因此 cold、archived 与 fork 后已物化的 child Thread 使用同一 owner，active 但尚未持久化的内存 snapshot 不参与结果。正文必须从 typed `ThreadItemPayload` 提取，禁止对 `item_json` 做 `LIKE` 或回读 deprecated projection：UserMessage 按原 part 顺序拼接 Text，assistant 只搜索每个 Turn 最后一个 `final/final_answer` AgentMessage；Lime 在 Turn 尚无 `final_agent_item_id` 时以 canonical Item ordinal 选择该末项。匹配为大小写不敏感的 literal substring，assistant Markdown 先投影成纯文本；snippet range 使用 UTF-16 code unit。分页 cursor 必须同时绑定 Thread identity、原始 search term 与 occurrence 位置，`turnCursor` 是可直接交给 `thread/turns/list` 的 inclusive opaque cursor；非法 cursor、空 search term、未知 Thread 与 store unsupported 都按结构化错误 fail closed。Renderer 当前没有 Thread 内查找产品面，本切片不新增侧栏标题搜索复用或无交互闭环的 GUI 空壳。

Codex v2 `thread/search` 的唯一跨 Thread 内容搜索链是：

```text
App Server thread/search
  -> RequestProcessor v2 lowering
  -> RuntimeCore read boundary
  -> ThreadStore::search_threads
  -> ProjectionStore
  -> state.canonical_threads + history.canonical_items
  -> current Thread status projection + first content snippet
  -> typed Rust / TypeScript response client
```

该方法与 `thread/list.searchTerm`、Renderer 侧栏本地标题过滤不是同一能力：它只按已持久化的 UserMessage
与 AgentMessage conversation text 匹配，并返回每个 Thread 的首个正文 snippet，不搜索 name/preview。默认及空
`sourceKinds` 只包含 Codex interactive source `cli + vscode`；显式 source filter、active/archived 二选一、
`created_at/updated_at/recency_at` 排序和前后翻页全部由 store-owned opaque cursor 收口。cursor 绑定 trim 后的
search term、archive scope、sort key 与 source kinds；反向翻页只切换 `sortDirection`，不得由 handler 或 GUI
解析 cursor。默认 limit 为 `25`、范围为 `1..100`，空 term、损坏或跨查询 cursor 必须 fail closed。搜索结果
Thread 继续经过同一个 v2 projection，不建立第二套 session DTO；active 但尚未持久化的内存正文不参与结果。
当前 Renderer 尚未接入 snippet 搜索产品面，本切片不把现有标题 Dialog 冒充 `thread/search` consumer。

canonical 写入必须先在 SQLite 事务内完成 normalization 与约束校验，再同步 append rollout，最后提交 projection；这样无效 change set 不污染文件，而文件成功、DB commit 失败可由 `(thread_id, sequence, fingerprint)` 幂等重试。metadata patch 不得隐藏在 append。ThreadStore apply 失败必须显式 fail closed；P1 完成后未完成的 projection tail 只能由 rollout rebuild，不能再由 raw EventLog 重写 canonical history，也不能 warning-and-continue 造成 GUI 可见而 rollout 丢失。

`thread/settings/update` 与 `thread/memoryMode/set` 的生产写入沿同一个 session actor 串行执行，不创建 Turn/Item，也不修改 active Turn 已捕获的 runtime request；更新只进入后续 Turn 的 session defaults。持久化顺序固定为：

```text
App Server typed method
  -> RuntimeCore session operation
  -> session actor 读取并合并当前 metadata
  -> ProjectionStore 以 state.sqlite 为 main，attach history/projection
  -> 开启 IMMEDIATE transaction 并校验 canonical identity
  -> RolloutStore append/verify thread_metadata digest chain
  -> 同一 attached SQLite transaction 更新 state.canonical_threads + projection projected_sessions
  -> commit
  -> RuntimeCore 内存 session state
  -> typed response / settings notification
```

`thread_metadata` 可以与 history record 交错，但必须携带 `previous_content_digest`、`content_digest`、`updated_at_ms` 和完整 metadata；scan 按文件顺序验证 identity、摘要链与时间单调性，并把最后一条 metadata 应用到 rebuild initial Thread。rollout append 成功而 SQLite commit 失败时，同 metadata 重试复用 rollout 中已同步的时间戳并补齐 projection；不同 metadata 必须因 expected-old/next-new 冲突 fail closed，不得覆盖尚未投影的 durable record。启用 AgentRoot 的实例若 canonical row 缺少 `rollout_path`，session metadata mutation 必须拒绝继续写入。memory mode 为 `disabled` 时，后续 Turn 不注入 memory prompt；该配置必须能从纯 rollout 重建后的 read model 恢复。

`thread/shellCommand` 是显式用户 shell submission，只允许进入 `RuntimeCore -> RuntimeSessionOperation::RunShell -> ExecutionProcessServer`。active session 必须复用当前 Turn，并共享该 Turn 的 cancellation token；idle session 必须由 actor 启动并持有一次性 task，创建独立 Turn，期间保持 session busy。命令执行不进入 provider sampling、普通工具 approval 或第二套 shell runtime。durable 与通知顺序固定为：

```text
thread/shellCommand
  -> session actor RunShell
  -> active Turn auxiliary | idle standalone Turn
  -> local process start
  -> command.started durable Item + item/started
  -> process poll / active cancellation
  -> command.exited durable Item + item/completed
  -> idle Turn terminal
```

Command Item 必须持久化 `UserShell` source、process id、canonical cwd、聚合输出、exit code 与 duration；live notification、thread read、rollout/SQLite projection 和 restart read 使用同一 canonical Item。内部终态只使用 current `command.exited`，不得新增平行的 completed/failed/canceled command event family；失败与取消通过 exit code、Item status 和 Turn terminal 表达。空命令、未知或 archived Thread、缺少本地执行环境、session/thread identity 漂移均 fail closed。

Renderer 的显式用户 shell 入口固定为输入首字符 `!`。输入 owner 只负责识别和清空命令文本；提交必须依次经过 Agent Chat shell controller、runtime adapter、typed App Server client 和 `thread/shellCommand`，不得调用裸 bridge、ExecutionProcess API 或 Electron 业务命令。controller 在请求前确保 session 并解析 canonical thread identity，订阅同一 session event route；Item started/completed 只触发 current read model/detail 刷新，ack 不得在前端合成 terminal。空命令不发请求，请求错误必须进入本地化可见错误通道。

Codex v2 background terminal control 只允许走 thread-scoped current 链：

```text
thread/backgroundTerminals/{list,terminate,clean}
  -> RequestProcessor v2 thread serialization
  -> RuntimeCore background-terminal boundary
  -> ExecutionProcessServer authoritative thread index
  -> tool-runtime local process supervisor
```

Thread identity 必须由 `CurrentTurnToolExecutor` 或 `thread/shellCommand` 的 canonical context 显式下传，
禁止从 provider 可写 metadata、cwd、session id 或命令文本反解。对外 `processId` 是单调数字字符串，作为
稳定排序和 cursor；list 只返回当前 Thread 尚未隐藏的 running process，cursor anchor 已消失时继续返回更大
process id，默认返回全部且 `limit=0` 提升为 `1`。terminate 必须在同一 registry 临界区校验
`threadId + processId`，跨 Thread 返回 `terminated=false`；命中后立即从 list 隐藏、释放 registry 锁，再向
真实 supervisor 发终止信号，并清理 `unified_exec` session mapping。clean 对当前 Thread 的已登记后台进程
执行同一隐藏与终止语义，响应固定为空对象。`write_stdin` 也必须校验 canonical Thread ownership，不能借
内部 session id 跨 Thread 控制进程。旧全局 `executionProcess/*` wire 不参与 Codex parity，不得为这三个
method 新建 adapter、mock fallback 或第二套进程 owner。

Codex out-of-band elicitation accounting 使用独立的 Thread-local current 控制链：

```text
thread/{increment,decrement}_elicitation
  -> RequestProcessor v2 Thread serialization / exclusive access
  -> RuntimeCore loaded Thread registry
  -> thread-local volatile reference count
```

两个方法只接受 loaded canonical Thread，并使用 checked `i64` 计数。increment overflow、count 为零时
decrement、非法 Thread id、cold/unknown Thread 都 fail closed；`paused` 严格等于 `count > 0`，归零后删除
registry entry。archive、delete 与 idle unload 都清理该 entry，因此 resume 不会继承过期的 process-local
registration。该状态表达外部 helper 的 live registration，不写入 rollout 或 Thread/Turn/Item projection。

当前切片只完成 exact public control plane，尚未把 Thread-level state 接入 `agent-runtime` provider
active-time budget。现有 MCP `ElicitationPauseState` 是 connection-local owner，不能代替 Thread registry。
统一 active-time consumer 接入前，`paused: true` 只证明 registration state，不代表已经实现 Codex 的完整
timeout-pause parity。

Codex Guardian denial 的人工放行使用独立的 provider continuation current 链：

```text
thread/approveGuardianDeniedAction
  -> typed v2 RequestProcessor + Thread serialization
  -> RuntimeCore loaded canonical Thread lookup
  -> typed Guardian denial/action validation
  -> provider-only developer continuation
       -> active regular Turn: agent-runtime session pending input
       -> idle/racing Turn: durable canonical runtime event
  -> provider history Developer message
```

wire 上的 `event` 保持 Codex opaque JSON shape，RuntimeCore 必须在副作用前解析 `status` 和完整 action tag，
并拒绝缺字段、unknown variant 与非绝对本地路径。只有 `status=denied` 生成 continuation；其他合法终态按
Codex 语义无副作用返回空对象。developer text 使用 Codex 的 exact-action marker，只授权原 event 中的单个
action，不得转成 session-wide approval、通用 permission cache 或旧 `agentSession/action/respond` waiter。
活动回合交给既有 session actor 在下一 sampling boundary 消费；同时写入 canonical event，使 idle、finishing
race、restart 与后续 Turn 都从同一 provider-history owner 恢复。该 provider-only event 不投影为用户可见
Thread Item，也不引入 v0/compat/mock。Lime 当前尚未生产 Guardian assessment/review lifecycle；本 method
完成的是 Codex 手工放行 continuation boundary，不代表 Guardian reviewer 产品闭环已经完成。

Codex v2 raw response item injection 使用独立的 provider-history current 链：

```text
thread/inject_items
  -> typed v2 RequestProcessor + Thread serialization
  -> RuntimeCore canonical Thread lookup / cold resume
  -> durable response_item.injected provider-only event
       + active regular Turn: agent-runtime session pending raw input
  -> provider history RawResponseItem
  -> model-provider protocol lowering
       -> Responses: preserve exact item JSON
       -> non-Responses: fail closed
```

wire 上的 `items` 保持 Codex opaque JSON shape，但 RuntimeCore 必须在写入前按 current `ResponseItem` union
完整校验，并拒绝远程图片 URL。cold Thread 先从 canonical store hydrate/resume；archived、unknown、空数组或
非法 item 均 fail closed。durable event 是 restart、idle 和 active/finishing race 的事实源；active regular Turn
同时通过既有 session actor 在下一 sampling boundary 消费 raw item。该 event 只进入 provider history，不生成
用户可见 Thread Item，也不得经 `ThreadItemPayload::Extension` 建立第二套 rollout 表示。

`model-provider` 是唯一 lowering owner：Responses route 原样保留 item JSON，包括通过 validation 后的 provider
扩展字段；Chat Completions、Anthropic 与其他不支持 raw Responses item 的 route 在发网前拒绝。当前 method
boundary 不关闭 P0-03 的全局 canonical history/rollout、rollback/fork/replay 与未知记录一致性缺口。

session start 返回成功前必须先同步写入 rollout metadata 首行并提交 Thread metadata row，再把 session 暴露到 RuntimeCore 内存状态；文件或 SQLite 失败都不得留下 memory-only session，DB commit 失败后的 metadata 文件由同 identity 重试接管而不是覆盖。canonical `session_id`、`thread_id` 与非空 `rollout_path` 均为唯一 identity，跨 RuntimeCore 重启也不得一对多。显式 session delete 与 import replace 的文件/metadata 原子协议尚未完成，当前入口不得被描述为安全清理能力；GUI、AgentSession adapter 和首事件 lazy create 都不能充当 empty Thread fallback。首事件 ensure 只保留为防御式幂等边界，不再拥有 Thread 创建时点。

action-required/approval 与 ask-user 的 live continuation 归 `agent-runtime` session/turn scoped pending state；可持久化恢复事实归 RuntimeCore canonical events。Codex v2 typed server request 必须先从 canonical descriptor 恢复 scope，再校验 caller 参数；不得信任 caller type、从 presentation/read-model JSON 二次解析、或让 RuntimeBackend 回读 AppDataSource。重启后只能恢复 typed descriptor，不能伪造原 oneshot；无 continuation 时返回结构化 `action_not_resumable` 且不得写 resolved。ask-user 的否定响应必须消费 waiter 并投影 canceled。MCP server-originated elicitation 不属于 approval 链，不能写入这些 runtime event 或 canonical Item。

### 6.3 Provider 与工具组

| Crate                                            | 职责                                                                                                                                     |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `model-provider`                                 | provider route、canonical message/content、capability、protocol lowering、HTTP/Responses WebSocket stream 与 normalized provider event。 |
| `lime-providers`                                 | `dead / deleted / forbidden-to-restore`；只允许历史 evidence 与负向守卫，不得恢复 crate、依赖或兼容调用。                                |
| `tool-runtime`                                   | tool definition、参数解析、approval/sandbox、执行 dispatch、MCP connection、tool result normalization。                                  |
| `mcp`                                            | MCP server/client 的领域集成。                                                                                                           |
| `skills`                                         | skill discovery、读取与运行时集成。                                                                                                      |
| `patch-apply`                                    | 受控 patch 应用领域能力。                                                                                                                |
| `browser-runtime`、`media-runtime`、`voice-core` | 浏览器、媒体、语音等独立 runtime domain。                                                                                                |

Codex `thread/realtime/*` 的 WebRTC/session control 与通知不进入 Lime Desktop 产品协议，分类为
`product-scope-excluded / forbidden-to-restore`。旧录音、麦克风和 realtime voice GUI 已退役；不得为了
Codex method parity 恢复第二套 Thread lifecycle、SDP 通道或 Renderer 音频队列。音频、语音和媒体能力继续按
grok-build 对齐 `model-provider` 的 catalog/capability/readiness 与 sampling/lowering，再由
`voice-core` / `media-runtime` 承接领域执行和 artifact 生命周期。

当前媒体执行边界进一步收口：`app-server` 的 audio task worker 只接受 RuntimeCore 解析出的
`openai_audio_speech` `ResolvedModelRoute`，通过 durable `credentialRef` 取凭证后调用
`model-provider::audio`，将结果写入 workspace 内相对音频路径，并把 `audio_output`、`result`、
`llm_events` 与 `provider_diagnostics` 写回同一 task artifact；不得在 worker 或 `voice-core` 新增
第二套 OpenAI TTS HTTP。云端 embedding 与 OpenAI-compatible Whisper transcription 同样经过
`model-provider` wire；本地 ONNX、Whisper/SenseVoice、百度和讯飞保持各自领域 owner。转写 task
artifact worker 只接受 `openai_audio_transcription` `ResolvedModelRoute`，通过 durable `credentialRef`
读取 workspace 内音频或受控 HTTP(S) 源，写入 workspace 相对的 txt/json/srt/vtt transcript artifact，
并把 canonical `llm_events`、`provider_diagnostics`、失败重试、stale recovery 与取消竞态回写到同一媒体任务。
`speaker_labels` 目前只作为任务元数据保留，provider-specific diarization lowering 仍未开放；embedding
仍需接入同一 route consumer 后才能标记为完整 current caller。

Architecture impact: major；本段把 `TranscriptionGenerate` 的执行 owner 从 Skill/CLI 入口收敛到 App Server
transcription worker，同时保持 `transcription_generate` 作为入口与 task artifact 类型。Architecture diagram
updated: this paragraph and the media/provider chain above. Responsible developer confirmation: root, 2026-08-16.

模型目录控制面固定为：

```text
provider config + configured-provider readiness + ModelRegistry metadata
  -> AppDataSource provider-scoped ProviderModelCatalog
  -> RuntimeCore CandidateModelSet (task/capability/modality/status/cost/limit/continuity facts)
  -> RuntimeCore model/list Codex v2 picker fields + authoritative catalog snapshot
  -> Renderer typed gateway / Electron recovery projection
```

`ProviderModelCatalog` 是 App Server 内部 read projection，不是第二个 provider network owner；
`CandidateModelSet` 是当前 turn 的控制面候选事实，不是全量 catalog，也不承接 provider HTTP；它由
RuntimeCore 持有，App Server 只适配 configured Provider 声明模型和缓存目录。能力/任务族/模态过滤先于
OEM 偏好、连续性与成本排序，显式模型锁定保持首位但仍经过 capability gate。provider wire、capability
与执行 route 仍只归 `model-provider`，其控制面语义按 grok-build 对照，wire/lowering 只按 OpenCode 补充。
public `model/list` 必须保持 Codex
v2 的 request、分页和 picker 字段：`{ cursor?, limit?, includeHidden? } -> { data, nextCursor }`。Lime 的
`Model` 在保留 Codex picker/reasoning/input-modality 字段外，追加 `providerId`、`capabilitySnapshot`、
`contextWindow` 与 `maxOutputTokens`，使 GUI 与 RuntimeCore route 使用同一份 typed catalog fact。多 provider
identity 同时编入可逆 opaque `Model.id`：`route:<base64url(providerId)>.<base64url(stableModelId)>`；
`Model.model` 只表示 provider wire model id。Renderer 必须通过 typed decoder 恢复 route，并要求 decoded
provider 与 `providerId` 完全一致；不一致或 capability source 缺失时 fail closed，不能解析显示名或按
provider 名称猜路由。默认目录只投影 picker-visible executable model；管理面显式 `includeHidden=true`。Spawn
Agent 模型选项消费内部 provider-scoped catalog，不得反向依赖 public DTO 或借展示字段推断 capability。

跨 Turn 模型切换上下文继续由同一个 RuntimeCore provider-history owner 投影：

```text
latest completed Turn -> durable routing.decision.made -> previous provider/model
current Turn -> preflighted agentControlRoute schema v2 -> initial route
RuntimeBackend route attempt -> actual selected provider/model
  -> compare previous truth with each actual route selection
  -> append one Codex <model_switch> developer message at provider-history tail
  -> current user input -> model-provider
```

previous truth 只认存在 `turn.completed` 的 Turn；failed/canceled route、候选设置和未完成 provider 尝试
都不能提前消费切换提示。首 Turn、相同 provider/model、仅 reasoning effort 或 service tier 变化不注入。
marker 只为当前 sampling 临时生成，不写新的 event、pending flag、Thread Item 或协议字段；冷恢复仍从
durable routing event 推导，切换 Turn 完成后下一 Turn 自然不再生成。health-aware reroute 每次按实际
selection 重新投影 marker，不能把初始 preflight route 冒充最终执行 route。Lime 当前没有 Codex 的
model-specific base-instructions catalog，因此 marker 只复用 Codex preamble，携带 XML-escaped 的
previous/current route，并要求继续遵循当前 system/developer instructions，不伪造模型专属指令。

架构影响：非重大；只扩展既有 RuntimeCore provider-history 投影，不改变 owner、依赖方向或公开协议。
架构图确认：上图复用现有 Turn admission/provider-history 主链。责任开发者确认：root，2026-07-28。

canonical Tool display contract 是 `ThreadItemPayload::{Tool,McpToolCall,CollabAgentToolCall}` 加 `ToolOutput`；call identity、arguments、structured content、duration、truncation、output reference 与 error 必须是 typed 字段，不能藏在 metadata 或由 Renderer 解析文本。Approval Item 的 ordered `available_decisions` 与 resolution 使用 Codex 同义的 `Approved`、`ApprovedForSession`、`Denied`、`Abort`、`TimedOut`；pending 只由 Item status 表达，此时 `decision = null`。GUI 只允许显式 lower 为 `allow_once`、`allow_for_session`、`decline`、`cancel`，不得从 scope 反推丢失的 decision。`requestId` 是审批 identity，`actionId` 只能作为缺少 request identity 时的退场 fallback；ask-user 可以 terminal 且 `decision = null`，Turn 使用 `Resolved` 表达已回答。MCP server elicitation 始终是独立的瞬时 reverse request，不产生 Approval 或其他 Item。

Provider request 只能按以下方向流动：

```text
typed runtime request
  -> model-provider canonical content / capability / route
  -> provider-specific lowering
  -> normalized provider stream event
  -> RuntimeEvent
```

`ProtocolKind` 不能把未实现的 Bedrock/Fal 等协议伪装为
`Custom` 再走 Chat Completions。若 `model-provider` 没有完整 lowering/stream/retry
实现，route resolver 必须返回 typed `UnsupportedProtocol`；协议声明、能力快照和真实
网络实现必须一一对应。Vertex Gemini 必须使用 dedicated `VertexGemini` runtime identity，不能通过普通
Gemini API-key alias 或 `Custom("vertex_gemini")` 放行。

Responses WebSocket 的 capability 由 `ProviderRuntimeSpec.supports_websockets` 或 direct runtime request 的显式字段进入 current config；`model-provider` 不按 provider 名称猜测支持。`AgentRuntimeState` 按 session identity 持有 provider client，route/config 不变时跨 Turn 复用，session close 时清理；不同 session 不共享 transport fallback。支持 WebSocket 的 Responses client 在同一连接上串行发送 `response.create`，禁止 multiplex，并让 SSE 与 WebSocket frame 复用同一个 Responses event reducer。Upgrade 426 或连接重试耗尽后，当前 session sticky 切到 HTTP SSE；WebSocket 在任何用户可见 event 前失败时可用完整 request 安全重放 HTTP，已经发出 text/tool event 后禁止重放，避免重复正文或工具副作用。流被取消或未完整消费时连接必须淘汰，不得把未读 frame 带入下一 Turn。

`ResolvedModelRoute.auth.kind` 必须经 `ModelRouteProviderConfiguration` 显式下传为
`RuntimeProviderAuth`，不得再由 provider 名称或空 API key 推断。`NoAuth` 只接受已解析的 direct
route，并在投影时清除任何误带 key；HTTP 与 Responses WebSocket 均省略认证头。`ApiKey` route 缺
key 必须在发网前拒绝。`OemManaged` 目前没有 `model-provider` adapter，provider admission 必须 fail
closed；不得将其伪装为无认证或 API-key route。

provider readiness 表示“该 chat route 可被 current adapter 执行”，不只是 provider enabled 或存在
key。configured provider 必须以 effective provider type 查询 `model-provider` adapter availability；
provider capability upper bound 必须委托同一 adapter availability，禁止维护第二份 provider-type alias
白名单而产生 ready/capability 漂移。`namespace_tools`、hosted `image_generation` 与 hosted `web_search`
只有在 canonical schema、request lowering、stream reducer 和 route gate 全部可执行后才能置为 true；普通
function/client tool 支持不能代替这些 provider capability。
每个 current chat transport 都必须由真实 loopback request capture 同时证明 endpoint、认证头、canonical
system/user/media/tool call/tool result/tool definition/generation lowering 与协议原生 terminal stream，不能用纯
lowering 单测或 reducer fixture 冒充发网证据。OpenAI Chat Completions、OpenAI Responses HTTP、Anthropic
Messages、Gemini GenerateContent、Vertex Gemini、Azure OpenAI Responses 与 Ollama Responses 已具备这套证据；
Responses WebSocket 另有 handshake、`response.create` 与 HTTP replay capture。Anthropic 只保留服务端实际返回的
input/output usage，不合成未提供的 `total_tokens`。
Gemini GenerateContent、Vertex Gemini、Azure OpenAI Responses 与 Ollama Responses 已是 current adapter。Azure 使用 resource
root、`/openai`、`/openai/v1` 或完整 `/openai/v1/responses` endpoint，认证仅允许 `api-key`，typed
`api-version` 缺省为 `v1`；deployment URL、Bearer/NoAuth、Chat Completions、WebSocket 与 hosted tools 均 fail closed。
Vertex 从 provider store 的 typed `project/location` 生成
`/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent?alt=sse`，
认证只允许 `Authorization: Bearer <access-token>`；普通 Gemini 的 `x-goog-api-key`、缺失 project/location、
带路径的自定义 origin、NoAuth 和 WebSocket 均在发网前拒绝。Vertex 与 Gemini 只共享 canonical
lowering/SSE reducer，不共享 provider identity、endpoint 或 auth lowering。
Ollama provider type/name 只解析为
`OpenaiResponses`，使用 `NoAuth` 和 `/v1/responses`；`/api/tags` 只归模型发现，不参与 agent turn。独立
`OllamaChat` protocol、NDJSON turn adapter 与 Chat Completions fallback 均为
`dead / deleted / forbidden-to-restore`。Bedrock 和 Fal 在完整 lowering/auth/query/stream adapter 落地前
均返回 `unsupported_protocol`，不得进入 `model/list` selectable catalog、当前 provider capability read、profile
fallback 或 connection/chat probe。Store 未命中的 provider 名称不得生成 builtin ready；唯一无 Store 旁路
是带完整 route/config/capability 的 explicit direct request。direct request 与当前 selection 绑定，只允许
一次 admission，不复用同一 endpoint/credential 参与 profile fallback。

公开能力读取只有 `model/list.capabilitySnapshot`：它直接投影 ready provider-scoped executable catalog 的 typed
snapshot，并保留 `providerId`、provenance、task family、modality、runtime feature 与 limits。具体 Turn 继续使用
resolved route/model capability 决定实际工具暴露，不能把列表展示字段当成跨 provider 能力上界。Codex 的空参数
`modelProvider/capabilities/read` 读取单一全局 Provider，与 Lime 的 Thread-bound route 不同且没有产品消费者，已从
protocol、dispatcher、client、schema 和测试正向面物理删除；只允许出现在 excluded scope、历史 evidence 与 retired
guard 中，禁止恢复为静态全局 capability owner。

产品默认 provider 的配置事实源只有顶层 `Config.default_provider`；Electron AppConfig、App Server、
`lime-config` observer 与 `lime-server` 必须读取同一字段。`RoutingConfig` 只承载 model aliases，旧
`routing.default_provider` 已删除且 YAML 解析必须 fail closed，禁止恢复双读、优先级合并或迁移期 fallback。

provider health 的共享粒度是 `provider/model/base-url/api-version/protocol/credential-scope`，由
`model-provider::CurrentProviderHealthRegistry` 持有；同一 scope 的不同 session 复用同一 circuit
breaker，但 HTTP client、WebSocket 与 HTTP fallback 仍保持 session-local。持久化凭证 scope 使用
credential UUID；direct runtime key 只使用进程内 SHA-256 指纹，不保存原始 key；`NoAuth` 固定使用
`no-auth` scope，不能由 session 或误带 credential 分裂。不同 credential 的 429/5xx 不得打开彼此的
breaker。该 transport 诊断不产生 Codex `model/rerouted` 或
`model/verification`：两者只能由对应的 cyber safety runtime 事实 producer 投影。
breaker 的结构化 observer 必须在内部 mutex 释放后发出 `provider_health` tracing，只包含
provider/model/protocol、credential kind、不可逆 route hash、state/reason、probe admission 与
retry-after；禁止记录 base URL、credential UUID、API key 或请求内容。HalfOpen 已有 probe 时的拒绝只返回
`50ms` 上限的短退避，不能再次宣告完整 open cooldown。
运行期 control plane 只能通过 exact `RuntimeProviderConfig` 向该同一 registry 读取脱敏 snapshot；未知 route
返回 unknown，且读取不得创建虚假的 `closed` breaker。snapshot 只包含 Closed/Open/HalfOpen、closed-window 的
sample/failure count、probe-in-flight 与 retry-after；open/half-open 已丢弃 window 时计数必须明确为 unknown，
不能由静态 provider readiness 或其他 route 推断。`AgentRuntimeState` 只委托这一个 registry；在没有当前
Thread route 的 Renderer/Settings 消费者前，不新增孤立 App Server JSON-RPC method。
HTTP/WebSocket transport 的真实重试在 sleep 前由同一 route observer 发出 `provider_retry` tracing，记录
transport、reason、failed/next/max attempt、delay_ms、delay_source 与可选 HTTP status；仍复用相同的
credential-safe route hash，不记录 endpoint、错误正文或凭证。当前请求策略保持 Codex 语义：只对 5xx 做
request-layer retry，429/auth/content rejection 不被普通 transport retry 放大；没有实际重试就不发
`provider_retry`。该 evidence 只服务 provider retry/health 诊断，不进入模型切换通知或配置 readiness。
HTTP 与 Responses WebSocket upgrade 都必须消费服务端 `x-should-retry`；显式 `false` 覆盖状态码默认策略，立即
停止当前请求的后续 attempt，并把最终 provider error 标记为不可重试。该类 server-directed rejection 不计入
route circuit health failure，不能因连续 429/5xx 错误打开共享 breaker。

Provider transport 正常结束但没有产生可见 text 或 tool call 时，由 `agent-runtime::provider_turn` 按
`grok-build` 的 empty-response 语义执行有界重采样；这不是 HTTP retry，也不进入 `model-provider`。reasoning-only
与完全空的 `stop` 共用独立的两次重采样预算，不消耗 `max_turns` 工具回合额度，并复用同一 sampling step 的
tool/hook snapshot 与原始 Provider transcript；空尝试中的 reasoning/assistant 内容不得写回下一次请求。工具执行后的
空 final 同样重采样，并保留已提交的 tool result。`content_filter` 空终态合法完成，`length` 与 `error` 空终态直接
失败，Provider token budget 必须在开始下一次重采样前生效。reasoning-only 直接失败、工具后空终答成功以及让语义
重采样消耗工具回合额度均为 `dead / forbidden-to-restore`。

provider 运行期重路由仍沿唯一 current owner 链执行。`model-provider` 负责产生结构化失败分类与
`retryable` 事实；`agent-runtime` 只传播单一 route 的执行失败、是否已经产生输出以及用量，不选择备用
provider。尚未产生 text/reasoning/tool 输出、尚未消费 pending steer 且 provider 调用前未发出用户可见
structured-input warning 时，App Server 按 resolved route 是否绑定 repository credential 分两种 scope：

- 已绑定 credential 的 `authentication`、`permission`、`quota`、`rate_limit`、`provider_internal` 或
  `transport` 失败只生成 credential-specific exclusion；认证、权限与 quota 即使不可重试，也可以尝试同一
  provider/model key pool 的下一把 key，因为这不是重放同一 credential。RuntimeCore 不能因此排除整个
  provider/model candidate；App Server 必须保持 exact provider/model/endpoint，过滤失败 credential ref，并在
  durable ref 命中失败 key 时选择下一把。指定 provider 的 key 池不得按 provider type 跨到另一 provider。
- 未绑定 repository credential 的 keyless route 只有 `rate_limit`、`provider_internal` 或 `transport` 的
  retryable 失败可以生成整条 route exclusion，再由 RuntimeCore 选择下一条 ready profile route。

请求错误、context overflow、content policy、unknown、已经产生输出、已经消费 pending input 或已经发出上述
warning 的失败必须原样终止；explicit direct request 也必须原样终止，禁止借 profile fallback 偷换 endpoint 或
credential。credential 池耗尽时必须保留最近一次真实 provider 错误，不得转到 backup model、跨 provider 捞 key
或把需要凭证的 route 降成 `NoAuth`。RuntimeCore 负责 scope-aware exclusion 与 routing attempts，App Server 负责
有限重路由编排；route fallback 继续复用 `routing.fallback.applied`，不得新增平行 event、第二 resolver 或
renderer fallback。任何 evidence/debug 只允许 provider/model、classification、retryable、scope kind 与稳定
reason code，禁止写入 endpoint、credential ref、API key 或 provider 错误正文。

repository credential 的跨 Turn cooldown 只允许由真实 provider 恢复元数据驱动。`model-provider` 在非成功
HTTP/WebSocket 响应上解析 `Retry-After` 秒值或 HTTP-date，以及 exhausted request/token quota 对应的
`x-ratelimit-reset-*` / `anthropic-ratelimit-*-reset`；原始 header 不跨层，只把非零 `Duration` 随结构化
provider error 交给 `agent-runtime` 和 App Server。App Server 仅在该失败已经满足 credential-specific reroute
安全条件时登记 cooldown；`ApiKeyProviderService` 以内存 deadline 按内部 key ID 过滤后续 runtime selector，过期
即清理。durable preferred ref 命中 cooldown 时也改选同一 provider 的下一把 key，但 exact credential read 仍是
无副作用读取。explicit direct request、无 credential ref、零/过期/畸形 header、无恢复提示的普通
401/403/transport/5xx 均不得产生固定假 cooldown。request 与 token quota 同时耗尽时必须取较晚 reset，避免在
任一限制尚未恢复时过早复用 key。cooldown identity、deadline 和原始 header 不进入 route payload、Debug、
tracing、历史或外部 evidence；内部 request-layer sleep 可按 Codex 上限裁剪，但跨 Turn deadline 必须保留服务端
给出的完整恢复窗口。

Responses 服务端事实只从 wire evidence 进入 canonical 主链。HTTP/WS handshake 只读取
`openai-model` / `x-openai-model` header；stream event 只读取 `response.headers`（优先）或顶层
`headers`，不得把普通 `response.model` 当成服务端实际模型。该事实经
`CanonicalLlmEvent::ServerModel -> agent-runtime -> model.server_reported` 保存为诊断 evidence，按 Turn
去重，但不直接产生 Codex `model/rerouted`、warning 或 deprecated v2 side-channel。普通 provider fallback
继续只使用 `routing.fallback.applied`。未来只有受信任 OpenAI/Codex route 的 requested/server model 不一致
且具备 exact `highRiskCyberActivity` producer 时，才能新增 `model/rerouted`。

`model/verification` 的唯一 producer 是受信任 OpenAI/Codex Responses route 的
`response.metadata.metadata.openai_verification_recommendation`。只识别
`trusted_access_for_cyber`，未知值、非数组、错误 event type、HTTP recommendation header 和第三方
OpenAI-compatible route 一律忽略。信任必须由已解析 runtime provider、Responses protocol 与 exact
`api.openai.com` endpoint host 共同证明；provider selector、展示名和 compatible alias 都不是信任依据。
canonical verification 在同一 Turn 的 transport retry 与 tool-loop
sampling 间最多投影一次，经 `model.verification` runtime fact 进入 App Server exact v2
`model/verification` notification；缺 thread/turn/typed payload 必须 fail closed。该通知不加入历史 item
resume replay，也不得触发 reroute。schema、Rust envelope 与 generated TypeScript client 由
`app-server-protocol` 单一事实源生成。

业务层不得拼 OpenAI、Anthropic 或自定义 provider payload。工具执行只按以下方向流动：

provider-neutral request/event algebra 只能由 `runtime-core::llm_protocol::canonical` 的
`Request`、`Message`、`ContentPart` 与 `LlmEvent` 定义。chat/responses/anthropic wire lowering
由 `model-provider::current_client` 消费该 canonical request；图片与视频只复用
`model-provider::lowering` 的 canonical media body builder。旧 `LlmRequest`、
`ProviderWireRequest`、`LlmEvent -> LlmRuntimeEvent` mapper 与 generic
chat/gemini/ollama lowering 属于 `dead / deleted / forbidden-to-restore`，不得为测试、媒体或兼容
重新建立第二套 provider-neutral 类型；Responses image options 归 `model-provider` 自有边界。

首轮响应策略的唯一产品 owner 是 App Server turn policy。Renderer 只提交用户显式选择与结构化上下文，不解析自然语言、不按长度/关键词决定模型或工具面，也不得伪造 App Server policy metadata。App Server 可以基于首个 sampling turn、detached desktop session、workspace/project/附件/capability/search/scene 等结构化事实选择 `model_slot`、`tool_surface` 与 `auto_compact`；RuntimeCore 只消费通用 preferred model slot，provider sampling step 只消费结构化 tool surface。策略流固定为：

```text
Renderer user config / structured turn context
  -> App Server turn policy
  -> RuntimeCore preferred model slot
  -> provider sampling-step tool snapshot
  -> model-provider request
```

`fast_response_routing`、`fastResponseRouting`、renderer localStorage 快速响应开关、自然语言分类与字符阈值均为 `dead / deleted / forbidden-to-restore`。compact surface 必须继续保留 deferred `ToolSearch` 与必要 core tools，并使用 auto tool choice；required search、workspace、附件、capability、plugin/skill/expert/service scene 或后续回合不得被首轮轻量策略降级。Provider phase trace 由每次 sampling attempt 发出 `request.started -> first_event.received -> first_text_delta.received`，只记录耗时与关联 identity，不保存 prompt、provider payload 或完整错误。

Provider step/token 预算的执行 owner 是 `agent-runtime` reply loop，不是 benchmark adapter。App Server 只从受控 harness metadata 投影正整数 `max_provider_steps` 和 `token_budget`，其中 step cap 不得扩大 runtime 默认上限。每个 completed provider step 按 `max(0, input_tokens - cached_input_tokens) + output_tokens` 累计；如果带工具调用的 step 已耗尽 token budget，reply loop 必须在执行任何工具和发起下一次 sampling 前返回 canceled execution。`provider.step`、request trace、Turn terminal 和 DeepSWE evidence 必须保留同一 attempt/usage；adapter 的 evidence polling 只允许作为 timeout race fallback，不能成为第二个预算 owner。

```text
model-visible definition
  -> RuntimeTool（definition + exposure + executor）
  -> ToolCall（turn/call/environment identity）
  -> tool-runtime permission and dispatch
  -> ToolLifecycleEmitter（started/completed）
  -> NormalizedToolOutput
  -> model transcript + host event projection
  -> RuntimeEvent and Thread/Turn/Item projection
```

current provider 不得绕过 `RuntimeTool::execute_call` 直接调用 executor，也不得在 provider loop、lime-agent adapter 或 App Server 重复计时、归一化或合成 start/end。host emitter 负责把同一 lifecycle 直接投影为 canonical `item.started/item.completed`，并保证 `ItemStarted -> ActionRequired -> ItemCompleted` 的确定性顺序。执行上下文必须显式绑定 typed call/turn identity，current turn executor 必须持有已校验的 canonical thread identity；approval 和 request-user-input 不得从松散 metadata 反推 scope。`AgentEvent::ToolStart/ToolEnd`、App Server raw start/result mapper 与 backend event-name mapper、`core::agent::types::{StreamEvent,ToolExecutionResult,StreamResult}`、image-command raw lifecycle、live `tool.args` 与 imported raw Tool product wire 均为 `dead / deleted / forbidden-to-restore`。conversation import 只允许在 Codex source parser 输入边界读取 rollout，并在 source adapter 内维护 typed `CodexTimelineItem` / `CodexRolloutEvent` 解析态；`history_builder::build_canonical_history_events` 随即将其映射为 canonical Item。terminal-only、incomplete 和重复 lifecycle 必须在 source adapter 内确定性补齐或幂等忽略，随后在真实 session/thread/turn identity 边界写入 canonical Item。raw Tool intermediate 绝不得进入 normalizer 之后的链路、`StoredSession`、event log、ProjectionStore、read model、notification 或 GUI。

current provider 的工具面按 model sampling step 冻结，不按整个用户 Turn 永久冻结。每次发 provider request 前必须生成一个 `RuntimeToolStepSnapshot`，同一 snapshot 同时拥有 model-visible definitions 与 exact executor；本 step 返回的 tool call 只能调用该 definitions allowlist 中的名称，未广告名称仍产生 canonical failed lifecycle，但不得进入真实 native/gateway/MCP executor。MCP snapshot 的唯一 owner 是 `tool-runtime::mcp_connection`：它按 server 隔离 discovery error/timeout，并把 prefixed definition、per-tool caller policy、dispatch route 与 immutable connection handle 一起冻结；同一步不得回查 live registry。`tool_search` 只更新本 Turn 的 deferred selection，旧 snapshot 不变，下一 sampling step 才可重新 capture。MCP bridge 的已归一化 tool timeout 固化在 connection client 中，因此 registry replace 后旧 step 继续使用旧 handle/timeout，新 step 才看到新配置。

provider tool-call repair 也只能消费该 sampling step snapshot。`model-provider` 必须把 wire 中的
`raw_arguments` 与解析后的 input 一起投影到 canonical `LlmEvent::ToolCall`，malformed JSON、空名称和
scalar arguments 不能提前升级为整步 provider error。`agent-runtime` 随后调用 `tool-runtime::repair_tool_call`，
只允许按本 step definitions、ASCII case 和 current native alias 得到 canonical name，并记录参数 diff。
参数 normalization 后必须继续使用同一 `RuntimeToolDefinition.input_schema`：只允许把 schema 明确声明为
`integer`/`number` 的合法 JSON 数值字符串确定性转为 number，不能按字段名、描述或自然语言猜测；随后用完整
JSON Schema 校验 canonical arguments。schema 编译失败或参数仍不匹配都必须 fail closed，不能进入 handler。
repair success 在写入 assistant transcript 前替换为 canonical name/arguments；repair failure 统一变为
model-visible `invalid` call，由专用 `before_handler` executor 产生 exactly-one started/completed lifecycle 和
failed tool result。即使 snapshot 中存在同名动态工具，带 typed repair failure metadata 的 `invalid` call 也
不得进入真实 handler。原始 malformed payload 只保留在 typed repair metadata，不复制到 model-visible error
arguments；schema 错误必须 mask 实例值，不能把 provider 参数原文带回模型。Architecture impact: major；本段改变
canonical provider event 与 tool execution trust boundary，但不改变
`model-provider -> agent-runtime -> tool-runtime -> Thread/Turn/Item` owner 方向。架构确认：confirmed；责任开发者
root，2026-08-11。

repair 后的 cancellation/timeout 继续复用同一 canonical terminal，不建立 provider-loop 旁路终态。handler 已启动时
turn cancellation 必须先投影唯一 `aborted` completed lifecycle；即使 handler 自有后台工作随后返回 success，也不得
覆盖该终态、追加第二条 completed 或触发下一次 provider sampling。repair 工具成功完成后，后续 provider sampling
step 的 absolute timeout 也不得重放 handler 或 lifecycle；canonical call/result transcript 只供该次 request 使用。
这两条组合竞态由 `agent-runtime::provider_turn` 跨 owner 回归直接守住。

MCP resource、resource template、prompt 与 server status 属于 App Server 管理控制面，不是 model sampling-step inventory。GUI `mcpPrompt/*`、`mcpResource/*`、`mcpServerStatus/list` 每次通过 `LocalAppDataSource` 的全局 `lime-mcp::McpClientManager` 对当前 live connection 执行 typed read；它们不得进入 `McpStepSnapshot`、不得通过 caller-unaware registry dispatch 执行，也不得回写或替换 in-flight Tool snapshot。连接初始化返回的 server capabilities 只用于 manager status、tool filtering 与 bridge 装配事实；model bridge 只携带 tool discovery/call/notification 所需能力。MCP client initialize 只能广告已有 typed handler 的 client capability；Lime 没有 `sampling/createMessage` owner，必须与 Codex 一样保持 sampling absent，禁止先广告再由 rmcp 默认返回 method not found。Agent runtime 的唯一 owner 是 `AgentRuntimeState[sessionId] -> McpThreadRuntime`：创建时固定 canonical `threadId`，独立持有 runtime `McpClientManager`、真实 RMCP connection、bridge registry 与 immutable generation；runtime 只从管理面提供的 typed enabled server spec 创建连接，绝不复用管理面 `RunningService`。每个 enabled server 并发启动：`required=false` 的失败只使该 server 在候选 generation 中 absent，健康 server 的 bridge 仍可发布；任一 `required=true` 失败则关闭未发布候选的连接并拒绝替换，已发布 generation 与其 pending elicitation 不受影响。配置变化时只在候选 generation 完成启动策略和 snapshot 后原子发布，旧 sampling step 继续通过 `Arc` 持有原 connection handle；删除 session 才按精确 `(sessionId, threadId)` 关闭已发布 runtime，取消 turn 不关闭它。server-originated elicitation 独占 `mcpServer/elicitation/request` reverse JSON-RPC method，不得复用 `agentSession/action/respond`、Approval 或 `request_user_input`。它是 thread-scoped、turn-correlated 的瞬时 reverse request：App Server 只保留 exact in-memory waiter，`thread/read`、Thread/Turn/Item projection 和 durable store 不得写入 pending 或 terminal elicitation。公开 request contract 只有必填非空 `threadId`、可空 `turnId`、必填非空 `serverName` 与 typed `mode: "form"`；`sessionId`、`parentToolCallId`、raw MCP request id 和私有 token 均禁止进入 wire。per-call `McpCallScope` 只保留可空 `turnId` correlation；connection 已在 runtime 创建期绑定 session/thread owner，因此每次工具调用不得重传、推断或覆盖 owner。管理面 nested elicitation 因没有 runtime owner 必须在 MCP service 边界 fail closed；不得使用 singleton、最近 active turn、`sessionId` fallback、`parentToolCallId`、progress token 或 server metadata 猜测 owner。router 以 session owner 精确取消：未转发 waiter 直接 Cancel，已转发 waiter 只触发 closed，必须等待 App Server adapter 先发送 `serverRequest/resolved` 再释放 RMCP waiter；同一 server 的不同 session/thread 不串线。MCP 内部 opaque token 只捕获在 adapter task，App Server outer request id 只出现在 JSON-RPC，二者是双层精确 identity；`turnId` 只作 correlation，不参与路由，也不能伪造成 sampling-step capability。MCP operation timeout 由真实 connection handler 的 counted pause state 计算 active time；等待一个或多个用户 elicitation 不扣 tool timeout，turn cancellation 仍立即生效。elicitation capability 继续 absent：Lime 不广告没有独立协议 capability 的行为。

Codex exact App Server MCP 请求固定为以下两条链，Desktop 不复制 TUI 交互，也不建立第二个 manager：

```text
mcpServer/resource/read { threadId?, server, uri }
  -> RuntimeCore
  -> no threadId: LocalAppDataSource management McpClientManager
  -> threadId: canonical Thread -> sessionId -> ExecutionBackend -> McpThreadRuntime
  -> contents[]

mcpServer/tool/call { threadId, server, tool, arguments?, _meta? }
  -> canonical Thread -> sessionId
  -> ExecutionBackend -> AgentRuntimeState -> McpThreadRuntime
  -> Session-owned McpClientManager -> RMCP tools/call
  -> { content, structuredContent?, isError?, _meta? }
```

`mcpServer/tool/call` 不允许落到全局 management manager；request `_meta` 也不得冒充 provider result `_meta`。Renderer Settings 没有 canonical Thread，只展示 tools，不提供调用按钮。Workspace MCP App 只传真实 `threadId`，不把 `sessionId` 放进 exact wire；Thread 未恢复时必须 fail closed。旧 `mcpTool/call`、`mcpTool/callWithCaller` 与 `mcpResource/read` 已从协议、App Server、typed clients、Renderer 和 smoke 物理删除，不建 compat wrapper，分类为 `dead / deleted / forbidden-to-restore`。Architecture impact: major; owner/data flow and public method boundary changed. Architecture diagram updated: this paragraph. Responsible developer confirmation: root, 2026-08-07.

MCP form elicitation 的产品表现是主窗口全局 GUI 模态表单，不是 Codex TUI prompt 的移植。Renderer 只消费 typed `requestedSchema`：string、number/integer、boolean、enum 分别映射输入框、数字输入、复选/开关和选择器；无法渲染或校验的 schema 必须 fail closed 为 decline/cancel。主对象是发起请求的 MCP 连接，阶段是待确认，单一主操作是提交，拒绝和关闭分别表达 decline/cancel；远端 `serverRequest/resolved` 必须通过 handler `AbortSignal` 静默撤销弹窗。该 handler 在主窗口根部只注册一次，不依赖具体页面挂载，不读取 raw MCP id，不使用生产 mock fallback。

Multi-Agent parent/child topology、agent identity 与 inter-agent mailbox 是三个 owner。`thread-store::AgentGraphStore` 定义 storage-neutral Open/Closed directional edge，App Server `ProjectionStore` 在 canonical SQLite 中持久化 child-unique parent、状态与稳定 descendants traversal；生产 AgentControl 必须通过该 owner 写 spawn/status/recover，禁止继续扫描 `agent_sessions.extension_data_json` 重建树。`thread-store::{AgentIdentityStore,AgentMailboxStore}` 是同一 root-thread tree 的 durable identity/mailbox owner：identity 以 `thread_id` 与 `(root_thread_id, agent_path)` 双重唯一，`task_name` 只能由 canonical path 末段派生；mailbox 用稳定 `message_id` 幂等 append、冲突 fail-closed、`QueueOnly`/`TriggerTurn` 分流、按 `(created_at_ms,message_id)` FIFO、按 root/recipient 隔离，并只将状态更新为 delivered 保留 audit record。mailbox 不能复用 `RuntimeQueuedTurn` 用户输入队列。S4u 定义 durable storage；S4w 在 `RuntimeCore` 建立唯一内部 consumer：`message_id` 派生 canonical Item ID，`TriggerTurn` 使用确定性 turn ID，`QueueOnly` 仅在下一真实 turn 前注入。canonical Item 必须在 mailbox delivered ack 之前可读；canonical EventLog 仍是事件顺序事实源，因此 EventLog-first 后的 canonical projection 失败保留 mailbox pending，严格校验同一 session 的连续 durable tail 后才重放 canonical Item 并 ack，identity/sequence 不一致一律 fail-closed。不得以临时 map、legacy session metadata 或第二套队列绕过这些 owner。S4v 已在 `RuntimeCore` 建立第一段 current control boundary：仅已加载的 parent session 可创建 child session/thread，成功后才持久化 Open edge；edge 写入失败时必须删除刚创建的 child session/canonical Thread，补偿失败仍显式 fail closed。Closed edge 与 descendants traversal 继续由 `AgentGraphStore` contract 拥有；没有 current consumer 的 RuntimeCore close/read 包装已删除，禁止为测试或未来猜测恢复。S4x 以 `RuntimeCore(session,thread,turn) -> AgentControlGatewayHandle -> ExecutionRequest -> RuntimeBackend -> current provider` 接入六个 current 工具；handle 只在该 turn 有效，provider 仅在 handle 存在时广告并执行 `spawn_agent`、`send_message`、`followup_task`、`wait_agent`、`interrupt_agent`、`list_agents`。S4aa 将 canonical child terminal activity 补入同一 durable owner：completed/failed child Turn 先完成 canonical Turn/Item 持久化，再按 durable direct-parent edge 写一条稳定 ID 的 `Result + QueueOnly` mailbox；interrupted/canceled 不生成 FINAL_ANSWER。canonical apply 前失败与 canonical 成功/mailbox append 前失败均由 parent 的 wait/下一真实 turn 沿 direct-child EventLog 有效前缀恢复，只应用 canonical 缺失 tail，再幂等补 result；恢复不得把 child 插入 RuntimeCore、递归扫描 grandchild 或把 delivered record 降回 pending。`wait_agent` 对调用前已存在和等待中新增的 queued steer 都优先返回 `Wait interrupted by new input`，无 steer 时才消费 mailbox activity，active wait 以有界退避重查 durable terminal recovery；并发 wait 只能有一个消费同一 activity。S4z 已证明新 RuntimeCore hydrate root 时不递归加载 descendants，`send_message` QueueOnly 不加载 child，`followup_task`/`interrupt_agent` 只 hydrate exact target，Closed edge 不可寻址且不 reopen。`RuntimeBackend` 只能 opaque pass-through，不得持有或回调 `RuntimeCore`；全局 agent registry、legacy metadata、第二队列、JSON-RPC/GUI 扩张和 Team/旧 alias 均不得作为该链路 fallback。S4y 已物理删除 `tool-runtime::collab_agent`、旧 Team catalog/prompt/discovery/registry surface，并将工具执行 smoke 迁到六个 V2 名称；这些路径属于 `dead / deleted / forbidden-to-restore`。canonical `CollabAgentToolCall` / SubAgent 历史与展示 payload 仍是独立的 read/projection 边界，不等于可执行旧工具，也不得在本删除切片中混删。S4ae/S4ah 已完成 ThreadStore-backed Renderer 与真实 Electron canonical SubAgent 产品闭环；旧 synthetic Team fixture 不再计产品证据。

Codex Orchestrator Phase A/B 的 execution capacity、resident capacity 与 rollout budget 是三个独立的 RuntimeCore owner，不能合并为一个模糊的 Agent 数量限制。current 数据流固定为：

```text
AgentControl spawn_agent / idle followup_task
  -> durable root AgentIdentity
  -> AgentExecutionLimiter atomic reserve / turn-admission claim
  -> AgentResidency root-scoped LRU reserve / idle session-loop eviction
  -> RuntimeCore session loop / current provider
  -> provider.usage attempt snapshot
  -> root-scoped RolloutBudget
  -> canonical rollout_budget.reminder or turn cancellation
  -> EventLog / ProjectionStore / provider history
```

execution limiter 只统计同一 durable root 下正在执行的 child Turn；root Turn 占 Codex 总容量四中的一席，因此 Lime 默认只提供三个 child execution slots。gateway 必须在 TriggerTurn durable mutation 前 reserve，turn admission 以 session identity 原子 claim，terminal、rollback、cancellation 和失败路径由 guard 释放；不同 root 隔离，已 active target 的 steer 不重复占位，超限以 `agent_limit_reached` fail closed。resident capacity 只统计已加载的 child actor，默认同样为三个；仅可淘汰最老的 idle terminal child，检查或 shutdown 失败必须恢复 LRU candidate 并继续/显式失败。completed/failed child 保留 durable Thread，可按 exact target cold reload；已 canceled/interrupted actor 在成功 eviction 后写 root-scoped lost tombstone，禁止伪造可恢复 actor。清理其他 session 不得删除同 root 的 lost tombstone。

`agent.rollout_budget` 是可选 typed 配置；存在但不合法时 Runtime factory 启动期 fail closed。启用后预算按 durable root 跨 root/descendants 共享，Responses wire 的 `codex_rollout_budget_units` 是优先计费事实，否则按 non-cached prefill 与 sampling 权重计算；同一 `(root, thread, turn, routeAttempt, attempt)` 只累计 usage snapshot delta，不同 provider route attempt 独立。RuntimeCore 在 admission 前从 root tree 的 canonical `provider.usage` 与 `rollout_budget.reminder` hydrate，耗尽后拒绝新工作，执行中首次达到上限立即取消。reminder 以 compaction/rollback window 去重并先作为 canonical `rollout_budget.reminder` 写入，再通过 `PreappendedById` 将同一 durable event 注入当前 provider route；reroute 可以重新注入同一 fact，但不得重复发布或落盘。未来 Turn 从 canonical provider history 恢复为 developer message，产生 reminder 的当前 Turn 不重复注入。不得建立第二预算 transcript、按 session 分账或用 Renderer/Electron 估算 usage。Architecture impact: major; RuntimeCore admission/residency/budget owner and provider-history data flow changed. Architecture diagram updated: this block. Responsible developer confirmation: root, 2026-08-14.

Phase C 的工具执行也只有一个 current owner：`agent/current_provider_turn` 负责静态策略与审批投影，`tool-runtime::execution_orchestrator` 负责不可变 `RuntimeToolExecutionAttempt`、一次首轮执行、typed sandbox/managed-network denial、按 approval policy 的单次升级和 cancellation。shell、`apply_patch` 与 unified exec 都把 canonical `(turn_id, call_id)`、effective sandbox、filesystem/network grant 和 approval source 送入同一 attempt；App Server `ExecutionProcessServer` 只消费 attempt，不重复 policy decision 或审批。普通 handler failure、timeout 和 cancellation 不得被文本启发式转换为 retry，attempt telemetry 只写入现有 Tool result metadata，不创建第二条 lifecycle。

Codex Orchestrator Phase D 的 Skills/MCP 只允许沿以下 current 链路运行：

```text
App Server config.yaml
  -> RuntimeBackend config metadata / fail-closed loadError
  -> AgentRuntimeState[sessionId, threadId] -> McpThreadRuntime
  -> codex_apps MCP resources/list (bounded cursor discovery)
  -> lime-skills::AgentSkillSnapshot (source/authority=orchestrator)
  -> turn_context -> provider prompt + skill_search
  -> read_mcp_resource(server=codex_apps, uri=skill://.../SKILL.md)
```

`orchestrator.skills.enabled` 与 `orchestrator.mcp.enabled` 默认开启；配置读取失败时两者均 fail closed。Skills discovery 只消费 session-owned `codex_apps` connection，固定 10 秒 deadline、最多 10 页、100 个 model-visible skills、1000 个 hidden skills，并限制 name/qualified name/package URI/resource URI/正文大小与 cursor loop；package/resource ownership 和 `mcp/skill` MIME/matching text 不通过就丢弃。`orchestrator.mcp=false` 只隐藏 `codex_apps` catalog、definitions 和 dispatch route，普通 MCP 仍可用；当 Skills 保持开启时，当前 turn snapshot 中精确匹配的 remote `skill://` locator 仍可通过同一 connection read，禁止回退 `std::fs`。一次 turn discovery 结果冻结在 `AgentSkillSnapshot`，reroute 复用该结果，`skill_search` 只读冻结 snapshot 并返回 metadata/locator，不把远程正文注入 prompt；正文必须由模型显式调用 `read_mcp_resource` 读取。Renderer、DevBridge、Electron Desktop Host 不持有配置、MCP manager、Skill discovery 或第二业务后端。新增协议只同步 `McpResourceListResponse.nextCursor` 与 `SkillSource/SkillAuthority=orchestrator` 的 schema/generated client。Architecture impact: major; current owner/data flow changed. Architecture diagram updated: this block. Responsible developer confirmation: root, 2026-08-14.

S4ac/S4ad/S4ae 固定 AgentControl 的 canonical Item 边界：`wait_agent` 独占一个 `CollabAgentToolCall::Wait` lifecycle；`spawn_agent`、`send_message`、`followup_task`、`interrupt_agent` 继续产生普通 Tool lifecycle，并仅在 gateway 完整成功后紧随 Tool terminal 追加一个 distinct completed SubAgent Item。App Server gateway 只能用 durable identity owner 解析出的真实 `ThreadId` 产生 Started/Interacted/Interrupted fact，输入 target path 不得冒充 ThreadId。fact 只允许经 `RuntimeToolExecutionResult -> NormalizedToolOutput` 的 serde-skipped typed internal field 进入 host emitter，不得写入 model-visible output、structured content 或普通 Tool metadata；失败、started phase、空/多 fact、tool/activity mismatch、wait/list 一律不产生 SubAgent Item。Started/Interacted/Interrupted 是唯一 current activity wire；Spawned/MessageSent/Waiting/Resumed/Completed/Failed/Closed 没有外部数据兼容约束，属于 `dead / forbidden-to-restore`。GUI 只从 canonical ThreadStore cold read 与 live Thread/Turn/Item notification 消费相同 Item identity，activity Item 的 completed 只表示该活动事实已落盘，不得推断 child terminal；child completed/failed 只由 S4aa Result mailbox 与 child thread lifecycle 表达。Renderer 必须本地化三态并禁止 `real:subagent:*` synthetic sidecar、raw enum 文案与 activity worker-result notification。Renderer 也不得按文本长度、正则或 selected Team 在发送前构造本地 formation、虚拟成员、work-board event 或 assistant dispatch preview；开启 SubAgent 只控制 current AgentControl 工具可用性，成员与状态必须等真实 child Thread/Item 后再展示。`team-workspace-runtime` 这类重新订阅 raw subagent status/stream、维护本地 draft/tool/queue map 并再次写 projection store 的第二 runtime 属于 `dead / deleted / forbidden-to-restore`；Workspace 只可从真实 child session/parent context 派生入口可见性，停止操作直接委托 current turn owner。

S6k 固定 canonical child roster 的 GUI 读取链：App Server `thread/list` 将 durable AgentGraph 与 AgentIdentity join 为 typed child Thread identity，并通过 `agentState` 暴露 `pendingInit/running/interrupted/completed/errored/shutdown/notFound` 七态；Renderer selector 只能基于这些 typed 字段形成 roster 和计数，`agentState` 缺失时才以 Thread/Turn lifecycle 作 canonical fallback，不得读取 metadata 或 raw Team status event 推断成员状态。Workspace 用 parent SubAgent Item 中的 child ThreadId 补 `notFound`，用 child Thread 自带的 sessionId 导航；只有 roster 未知或 sessionId 缺失时才通过 current `thread/read` 解析。Harness、RuntimeStrip 和子线程导航必须消费同一 roster，不建立第二事件队列或本地成员表。

AgentControl child route 必须继承 parent Turn 已解析后的有效 runtime options，而不是只复制 renderer 显式请求。`RuntimeBackend` 复用既有 model selection、reasoning、App Server turn policy、workspace 与 search policy 解析，在执行前把 effective options 回写唯一 `StoredSession.turn_runtime_options[turn_id]`；随后 per-turn gateway 只能从该 map 复制到 child。child 必须清除 parent-only `event_name`、`queued_turn_id`、`expected_output`、`structured_output` 与 `output_schema`。禁止复制 `business_object_ref` / session metadata、增加第二 route map/resolver 或用 compat/fallback 猜测 route。

`spawn_agent.fork_turns` 的 current contract 归 `tool-runtime::agent_control`：缺省/空白/`all` 表示完整历史，`none` 表示空历史，正整数字符串表示最近 N 个拥有 canonical input 的非 queued Turn；`0`、非法字符串、`fork_context` 与未知字段 fail closed，Renderer validator 必须按 64 位发布目标的 Rust `usize` 语法和上限使用同一边界向量。App Server 只消费 typed `SpawnAgentForkMode`，在 child identity、初始 mailbox task 和执行调度之前，把选择后的父 Turn 以新的稳定 child Turn/Item identity 写入 child 自身的 EventLog、ProjectionStore 与 ThreadStore；非 `none` fork 同时持久化 `parent_thread_id` 与 `forked_from_id`，`none` 只保留 graph parent。source Thread/Turn/Item identity 通过 `forkedFromThreadId/forkedFromTurnId/forkedFromItemId` 写入 child EventLog 和 canonical Item metadata，禁止直接复用会与 parent projection 冲突的全局 legacy Turn identity。Turn 资格只认 completed canonical UserMessage，typed `AgentInput` 从同一 EventLog hydrate；assistant 只认已完成 Turn 中 `ItemStatus::Completed + phase=final_answer` 的 canonical AgentMessage，并重建完整 `message.delta + message.completed` lifecycle。commentary、reasoning、tool lifecycle、inter-agent communication、parent trace/request/run 字段与 raw Team 旁路一律不复制；provider history 必须从同一 child EventLog 派生。任一 history/lineage/graph/identity/mailbox 写入失败都 best-effort 清 ProjectionStore、EventLog/workflow audit、sidecar、approval cache 与内存 session 后显式报错，禁止半个 child、稳定 ID 重试污染、第二 history store、session metadata owner、`fork_context` compat 或恢复旧 subagent whitelist。child 的后续回合与 root 一样由 per-turn gateway 暴露六个 AgentControl 工具，递归树仍由 durable root-thread graph、权限与执行容量边界 fail closed。

`wait_agent` 的 canonical storage payload 继续是 `CollabAgentToolCall::Wait`，但 GUI presentation 必须是 `tool_call` + `tool_name=wait_agent`；它不是 SubAgent activity。只有 distinct `ThreadItemPayload::SubAgent` 才能投影 Started/Interacted/Interrupted 三值和 child Thread identity。AgentControl Gate B 必须同时看到六个 completed typed Tool row、三类 canonical SubAgent activity、v2 `thread/read` 的 `electron-ipc` trace、零 invoke/console error 与真实 Electron 页面；localhost provider fixture 不能冒充 live-provider proof。

Provider history、context compaction 与 canonical read model 也属于 canonical Tool 的生产 consumer：它们只允许从 nested `ThreadItemPayload::{Tool,McpToolCall,CollabAgentToolCall}` 读取 call identity、ItemStatus、arguments、metadata、structured output、output reference 与 MCP server identity。非 lifecycle 的领域 side-channel 可以按显式 allowlist 保留，但 raw `tool.started/result/failed/completed` 不得影响 transcript、摘要、统计、browser evidence 或 artifact 提取，只能存在于入口拒绝守卫、负向测试和历史 evidence。

Context compaction 只重写 model-visible provider history，不删除或改写 durable EventLog、Thread/Turn/Item read model。`context.compaction.completed` 的最新 `tailStartTurnId` 是唯一 provider transcript 边界：`session_context_compaction.v2` summary 只接续该 tail 之前被移除的 turn，tail 及后续 turn 继续从 canonical events 原样投影，本轮 user input 仍由 execution request 单独追加。找不到有效 tail event 时必须 fail open 为完整历史，禁止静默丢 turn。artifact policy 必须分别声明 `durableHistoryRewrite=false` 与 `providerHistoryRewrite=true`；不得恢复“只注入摘要但仍发送全部旧历史”的双份上下文。

大型工具输出的唯一正文来源同样是 nested `ToolOutput`。App Server 可以在 append 边界把过长 `text` 截为 preview，并把完整内容写入 `tool_output` sidecar；nested output 必须回写稳定 `outputRef + truncated`，outer event 只保留 `outputBytes/outputSnapshotFile/sidecarRef` 等持久投影，不能从 outer `output/result/runtimeEvent` 反向恢复正文。后续 provider history 只能消费 canonical preview，并继续对异常超长 inline output 执行有界截断；`outputRef` 的 full sidecar 只供显式 artifact/evidence/read owner 读取，禁止在多轮历史投影时自动回灌模型。Tool、MCP 与 Collab 使用同一 sidecar owner；raw `tool_end` 与 raw Tool lifecycle 一并在 EventStore normalization 前 fail-closed。

图片任务的 GUI media projection 只消费 `item.completed` 中 completed `ThreadItemPayload::Tool`：tool identity 来自 typed call/name，任务 owner facts 来自 `item.metadata`，结构化响应来自 `ToolOutput.structured_content`。只有 `normalized_status=succeeded` 且图片拥有可校验 sidecar reference 时才生成 final media content part；pending、非 terminal、失败、无 sidecar 或 raw Tool event 必须 fail closed。异步 worker 的最终结果继续由 media task store read owner enrich，不得把“任务创建完成”误报为“图片生成完成”。

### 6.3.1 Multi-Agent crash commit

S4am 的 crash contract 取代上段 S4v“child 创建后直接写 Open”的旧顺序。`AgentGraphStore` 的 current 状态为内部 `Pending`、产品态 `Open` 和审计态 `Closed`。spawn 的第一笔 mutation 必须原子 reserve Pending 并携带临时 child session identity；随后才能创建 child、写 `session.created` EventLog、fork history/lineage、identity 与初始 TriggerTurn mailbox，最后以 `(child_thread_id, child_session_id, Pending)` CAS 单次发布 Open。

Pending 必须在 canonical/projected/in-memory 的 Thread/session read/list、roster、terminal recovery 与 GUI/API 中全部隐藏。任一步返回错误都在 Pending 隐藏下清 ProjectionStore、EventLog/workflow audit、sidecar、approval cache、identity、mailbox 与内存 session，全部成功后才删 intent。硬崩溃由 App Server 在 EventLog/ProjectionStore/sidecar 装配完成且接收请求前全局回滚 Pending，并只继续 Open child 的 durable TriggerTurn；普通 descendants 继续按 Codex V2 lazy resume。禁止新增 metadata journal、第二 history store、Electron 后端或兼容入口。

### 6.4 领域与基础设施组

| Crate                                            | 准入                                                                                                           |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `config`、`infra`、`core`                        | 配置、平台无关基础设施和稳定公共模型；不接受默认塞入的新 runtime 逻辑。                                        |
| `services`、`processor`                          | 有明确领域 owner 的服务与处理器；中心 facade 只做 dispatch。                                                   |
| `knowledge`、`embedding`、`document-preview`     | 独立领域能力。                                                                                                 |
| `gateway`、`websocket`、`server`、`server-utils` | 网络/服务边界。                                                                                                |
| `providers` / `lime-providers`                   | `dead / deleted / forbidden-to-restore`；provider 网络、wire lowering、stream、catalog 只归 `model-provider`。 |
| `scheduler`、`automation_execution` 对应 owner   | 调度与自动化领域，不承接 turn loop。                                                                           |
| `lime-cli`                                       | Rust CLI 入口，不替代 App Server 产品协议。                                                                    |

新增 Rust crate 必须说明：现有 domain 为什么不适合、公开 contract 是什么、依赖方向是什么、如何避免落入 `core`/`services` 平铺层。

## 7. Agent 产品主链

```text
Renderer
  -> Electron preload / Desktop Host
  -> app_server_handle_json_lines
  -> App Server JSON-RPC v2 initialize + Thread/Turn commands
  -> RuntimeCore / agent-runtime
  -> model-provider and tool-runtime
  -> RuntimeEvent
  -> ProjectionStore / thread-store
  -> canonical Thread/Turn/Item read model + notifications + derived exports
  -> Renderer projection / GUI
```

Codex app-server 的核心约束在 Lime 中保持不变：先初始化连接；以 Thread 开始或恢复会话；以 Turn 驱动一次执行；以 Item 和 notification 报告过程；以明确 terminal turn 状态结束；从持久化 read model 恢复历史。任何 UI、Plugin 或桌面入口都必须进入这条主链。

Codex 对话导入同样服从这条主链。`conversationImport/*` 只负责只读发现来源、解析
Codex persisted rollout，并在 source adapter 内维护 `CodexTimelineItem` / `CodexRolloutEvent`
等解析态；`history_builder::build_canonical_history_events` 随即将其映射为 canonical
Thread / Turn / Item 历史。导入链不得创建 `ImportedRuntimeEvent`、imported-only tool
lifecycle、第二套完整历史 sidecar 或 Renderer 专用工具卡。历史 command、patch、MCP 和
tool call 只能作为已完成/失败的 canonical Item 写入，绝不重新执行。导入后的新 Turn
通过普通 `turn/start` 进入当前 provider loop 与 `tool-runtime`，使用当前模型、
审批和 sandbox；导入模块不得拥有 executor、pending approval 或 tool catalog。

超大 rollout 也不得把长解析和持久化占在单个 JSON-RPC 请求内。current 导入控制流固定为
`conversationImport/thread/commit -> RuntimeCore import job -> conversationImport/job/read`：
commit 只完成用户确认、同源 active job 去重和后台 worker 启动；worker 仍调用同一个
canonical commit owner，按 reading/building/persisting/finalizing 阶段报告 turn/item 进度；
只有 terminal result 才向 GUI 暴露可继续的 session。job registry 只拥有调度状态、进度和
terminal result，不保存第二份 Thread/Turn/Item，也不得绕过 EventLog、ThreadStore 或
ProjectionStore。Renderer 只能通过 typed gateway 轮询，Electron 不得承接 job runner。
批量确认时 Renderer 必须先为全部勾选会话启动或复用 job，再观察 terminal；关闭弹窗只
abort 当前 Renderer observer，不取消 App Server job。再次打开时 source scan 投影
`importing` 与 `importJobId`，Renderer 直接通过 `job/read` 继续观察；不得用第二次 commit
代替显式 job identity，以免 terminal 竞态创建重复会话。

Agent GUI 的聊天/工作台响应式布局同样不得按来源分叉。`LayoutTransition` 是唯一布局
owner：实际工作区容器宽度大于 `900px` 时可以并排展示聊天与工作台；不大于 `900px`
时必须切为聊天优先的单面板，并提供明确的“聊天 / 工作台”模式切换。断点切换只能改变
排列和显隐，不能重挂载消息树，否则会丢失 canonical timeline、滚动位置或输入草稿。
普通会话、历史恢复和导入续聊共享该 contract；真实 Electron 视觉门禁必须覆盖
desktop、compact、narrow，并拒绝模式控件与其他按钮发生矩形重叠。

历史 GUI 投影遵循 Codex App 的运行态边界：canonical Thread/Turn/Item 和 read model 永远
保留完整 command、reasoning、tool、approval、search、patch 等事实，但 Renderer 只有当前
active turn 才挂载 operational details。terminal turn（completed、failed、canceled 或
interrupted）只投影最终正文、附件、文件产物/变更与处理时长分隔；历史分隔是不可交互的
`div`，没有展开、预览恢复或按历史 item 重新挂载的入口。该规则不区分普通、恢复或 Codex
导入来源；read model 完整性必须由 App Server `thread/read` / `thread/items/list` 独立验证，
不能用 GUI 隐藏推断 canonical item 被裁剪。运行中审批卡、command/tool/search/reasoning
仍按当前 turn 的真实 lifecycle 展示，turn 终态后统一收口为 compact history。

canonical identity/control read edge 使用 Codex v2 `thread/read`、`thread/list`、`thread/turns/list` 与 `thread/items/list`，由 App Server handler 直接查询 `ThreadStore` 并返回 Thread/Turn/ThreadItem DTO 和 store-owned opaque cursor。`thread/list.includeArchived=false` 只返回 active thread，`true` 返回 active 与 archived thread；过滤和 cursor 顺序必须由 store 在同一查询边界完成。携带单一 `threadId` 的 read method 由 protocol catalog 声明 Thread scope + shared-read access；App Server request serialization 必须消费该 metadata，不能在 handler、client 或 GUI 另建并发策略。不存在第二个 session presentation endpoint；canonical detail 缺失或 store 失败必须显式失败，禁止 event/app-data fallback 或 Renderer 合成空 history。

loaded `thread/resume` 的运行态判断只允许读取 `agent-runtime` session actor 的有序
snapshot：App Server 先由 canonical ThreadStore 解析 `threadId -> sessionId`，cold hydrate
只恢复 RuntimeCore/read model，不得伪造 active actor；已存在 actor 才能报告唯一
`activeTurnId`。resume 投影据此保留 live turn 的 `inProgress`，把其它 stale
`inProgress` 映射为 `interrupted`，并将 Thread 归一为 `active`、`idle` 或保留
`systemError`。App Server 的 `ThreadStateManager` 是唯一 loaded-thread listener owner：
RuntimeEventHub 只按 canonical `AgentEvent.threadId` demux，缺失 threadId 直接 fail closed；
每个 listener generation 持有唯一 projector 和 command channel，在同一 actor 内先订阅精确
connection，再依次把 response、canonical token usage、canonical ThreadGoal snapshot、
thread-scoped pending server request replay 和后续 live event入队到该 connection 的既有 bounded
writer。`thread/start` 与 `thread/resume` 都走此订阅边界，
断连同步移除双向 subscription；external runtime append 不再保留全连接 raw publish 旁路。
token usage snapshot 由 `runtime/thread_usage.rs` 从 canonical event 严格读取；main runtime 的完整
flat terminal usage 在 EventLog 写入前 lower 为 cumulative `total/last/context-window`，其它 partial 或
untrusted usage 继续 fail closed。ThreadGoal 由 canonical ThreadStore 的 `thread_goals`、
`thread_goal_turn_accounting` 与 durable update outbox 持久化，禁止复用或映射旧
`ManagedObjective`。`turn.accepted` 绑定 exact goal id、Plan mode、累计 token 与 wall-clock baseline；
RuntimeCore 是 admission producer，synchronous collecting path 在 backend 首个 Turn lifecycle
进展前补齐缺失的 `turn.accepted`，但零事件 rejection 仍不持久化假 Turn。若 accepted 时尚无 Goal，
运行中的首次 active Goal mutation 在持有 RuntimeCore state 锁时，以已进入 canonical state 的累计 usage、
source watermark 和当前时间 late-bind 当前 Turn；后到或 replay 的 accepted 不得重置该 baseline。
已有 Goal 的 external set/clear 也保持 `RuntimeCore state mutex -> SQLite BEGIN IMMEDIATE` 单一锁序：
同一事务先按 exact goal id flush mutation 前 token/time，再写入 set/clear；Active set 即使 goal id 未变也会
把当前 cumulative usage/time 重置为新 baseline，pause/resume 排除 paused 区间，clear 删除旧 Turn binding，
因此同 Turn recreate 可以绑定新 goal id。mutation flush 不写 terminal outbox，set/clear response 后的
listener notification 是唯一最终 snapshot，避免旧 objective/status 延迟回放；同 source sequence 只允许
external mutation 推进 wall time，terminal/replay 仍要求严格递增 watermark，过期或 terminal rebind fail closed。
`turn.completed|failed|canceled`
在存在完整 cumulative usage snapshot 时于同一事务推进 source watermark、usage、
`budget_limited` 和 typed update outbox；`turn.failed.payload.reason` 只使用结构化
`turn_error|usage_limit_exceeded`，Active goal 分别转为 `blocked|usage_limited`，Plan turn 与
cancel 不改变 goal。per-thread listener 严格在对应 accounting event 后发送
`thread/goal/updated` 并确认 outbox。中间 provider usage 已由独立
`provider.usage` current event 承载：同 attempt 取最新 snapshot，不同 attempt 累加，
`provider.step` 仅保留生命周期和诊断，正常 `turn.completed` 不与中间 usage 双计。生产 usage
producer 的异常终止 flush、tool-finish/abort flush、
outbox crash-drain 与 MCP terminal notification 仍为 OPEN；禁止在 Renderer、Electron、RuntimeCore read
model 或 transport 层另建 accounting 或 resume 状态机。

ThreadGoal idle wall-clock accounting 的 current owner 是
`canonical_thread_store/goal_idle.rs`。它只保存进程内 `active_goal_id + last_accounted_at`
baseline；累计秒数仍原子写入 canonical `thread_goals`，不新增持久化状态机。单一 permit 覆盖
snapshot、SQLite usage write 与 baseline 推进，clone store 和并发 mutation 不会重复扣时。
active Goal 的 set/pause/clear 在 mutation 前 flush；非 Plan `turn.accepted` 在同一 SQLite
`IMMEDIATE` 事务内完成 idle wall seconds、typed outbox 与 exact Goal Turn bind，token delta 固定为零，
idle snapshot 以 accepted event timestamp 截断，投影延迟不得与 per-turn time 重叠；任一阶段失败
必须整体回滚。Plan admission 只清 idle baseline，不结算 prior idle。Turn terminal 只有在
persisted Goal 仍为 active 且不是 Plan mode 时才重新起 idle baseline；paused、blocked、
usage-limited、budget-limited、complete 与 Plan terminal 均清 baseline。同进程重复 `thread/resume`
对同一 active Goal 保留已有 baseline；cold `ProjectionStore` resume 才从当前时刻建立新 baseline，
进程离线时间不计入。accepted replay 先识别既有 Turn binding，不得清除当前 baseline、覆盖
late-bind baseline 或重复结算。

ThreadGoal 自动续跑的 current owner 是 `runtime/thread_goal_continuation.rs`。它把 Codex
`InternalModelContextFragment(source="goal")` 对应为独立 durable
`thread.goal.continuation` event：当前 Turn 通过 agent-only provider input 获取上下文，后续
Turn 从 provider history 恢复，但 canonical Thread/Turn/Item 和 v2 notification 不生成
`userMessage` 或 raw side-channel。Goal continuation 只 admission，不在同一 future 内同步递归；
driver 继续挂在现有 Tokio runtime，Completed 后重新进入 idle gate。per-session single-flight
保护 goal read + admission，queued 用户 Turn 和 TriggerTurn mailbox 由 pending-work gate 优先
清空，内部 pending-owned Turn 不重入 idle scheduler。idle `thread/goal/set(active)` 与无 live
Turn 的 `thread/resume` 可直接启动；set 路径必须先完成 response 与 `thread/goal/updated`
listener publish，resume 路径复用 barrier 保证 `response -> Goal snapshot -> continuation live events`。
Plan mode 跳过但不丢 Goal；普通 failed 先把 Goal 推进
`blocked|usage_limited`，显式 canceled 不立即续跑。

Renderer 的唯一 Goal owner 是 `useAgentSessionThreadGoal -> ThreadGoalPanel`，只接受 canonical
`thread_id + ThreadGoal`。Harness detail、Inputbar inline 和 TaskRail 共用同一 thread identity
规则；identity 不匹配时 fail closed，不从 session/workspace id 猜测 Goal，也不读取
`threadRead.managed_objective`。聊天旧 `ManagedObjectivePanel`、inline wrapper、criteria/audit/
continue UI、Automation Objective summary/details/audit、v0 `agentSession/objective/*`、
`managed_objectives` repository/table 和 `agentChat.managedObjective.*` 文案均为
`dead / deleted / forbidden-to-restore`。不得恢复 owner kind、criteria、evidence audit 或手动
Objective continue wrapper。Automation 只负责按 schedule 向 persisted Thread 提交 Turn；Goal
identity、状态、预算、用量和 idle continuation 仍由该 Thread 的 canonical `ThreadGoal` 唯一拥有。

public `thread/fork` 的 current owner 是
`protocol/v2::ThreadForkParams -> processor/thread_fork.rs -> RuntimeCore::thread_fork ->
canonical ThreadStore`，与 AgentControl child topology fork 完全独立。它从 canonical
Thread/Turn/Item 复制 full、`lastTurnId` 或 `beforeTurnId` 的 terminal prefix，并只改 target
thread/session identity；不得以 raw EventLog 或 AgentControl 的有损 history 充当第二个 fork owner。
`path` 与 ephemeral persistent store 尚未落地，必须 fail closed。普通 fork 不继承 Goal；只在
persistent `deferGoalContinuation=true` 时，`goal_fork.rs` 持 source Goal idle permit，在同一
SQLite `IMMEDIATE` 事务中 flush source usage、复制 exact Goal snapshot、写带 Goal 外键的 durable
deferral marker。fork/resume 都不自动启动 Goal；首个真实 `turn.accepted` 幂等消费 marker，随后仍由
既有 continuation owner 决定自动续跑。marker 随 target Goal 删除级联清理。已覆盖 full/last/before、
无 Goal、paused Goal、restart/resume、首个 explicit Turn、删除级联和参数 fail-closed。fork 的
provider history 只从 copied canonical Item 生成 typed in-memory seed；`forkSequence` 固化 copied
prefix 边界，baseline slot 保持 EventLog sequence 稠密，UserMessage、AgentMessage、Reasoning、Tool
与 McpToolCall 由 `provider_history` 唯一 lowering，MCP 名称必须经 `lime_mcp::naming` 规范化。
restart 时先 hydrate target EventLog tail，再按 sequence 合并 canonical seed；ProjectionRepair 也必须
用相同 seed 补齐 prefix 后再修复 canonical history，不能用只有 target tail 的 raw EventLog 删除 copied
Turn/Item。source raw EventLog 不复制到 target。当前 canonical payload 无法无损表达的 user/assistant
media、ContextCompaction replacement history、CollabAgentToolCall arguments 与未知 Extension 均在
`thread/fork` 创建边界 fail closed，不允许猜测、占位或 compat fallback。

Fork provider-history architecture confirmation: root, 2026-07-21. The current owner remains
`canonical ThreadStore -> RuntimeCore seed -> provider_history`; ProjectionRepair only preserves this
prefix while applying the target EventLog tail, and no second transcript or Electron owner was added.

工作区 artifact 持久化的 current 写入口是 App Server v2 `artifact/write`：Renderer 只提交
`threadId + optional turnId + typed ArtifactSnapshot`，App Server 解析 canonical thread identity，
由 RuntimeCore 追加 `artifact.snapshot` 并返回 event identity、sequence、persisted time 与
`relativePath/bytes/sha256/contentStatus` sidecar evidence。Electron 只转发 JSON-RPC，禁止直接写
artifact 正文、伪造 evidence 或恢复 `agentSession/runtimeEvents/append` 通用写入口。

代码/文档 `artifact.snapshot` 在进入 canonical Thread history 时按 completed `FileChange`
lower：路径来自 artifact sidecar 摘要，正文/preview 进入 typed diff；图片仍使用
`ImageView`。这让 Workbench cold read 继续消费 Codex v2 Item，而不是恢复
`agentSession/read` 的 artifacts/tool_calls 胖 detail。其它没有 v2 表示的 generic media 继续
fail closed，不能伪装成图片或塞进 metadata。

FileChange 的唯一事实源是 canonical `ThreadItemPayload::File { changes, status }`：一次
patch call 只生成一个 Item，完整保留 Add/Delete/Update/Move 的有序 `changes[]`，Move 的
`path` 始终是源路径、`move_path` 是目标路径。`patch.started/applied/declined/failed` 只能更新
同一 `patchId` Item，终态没有 snapshot 时保留 started 的 changes；patch 专属的逐文件
`file.changed` 旁路已删除。v2 `PatchChangeKind` 直接复制 Codex tagged wire：
`{ type: "add" | "delete" | "update", move_path? }`；拒绝审批投影为 terminal
`Declined`，取消后由 turn lifecycle 单独终止，禁止把用户拒绝伪装为执行失败或恢复旧单文件
Update 投影。

direct notification 主链已把 Codex v2 lifecycle 与流式 Item method 纳入统一 App Server method catalog、schema manifest 和 generated client：`thread/started`、`turn/started`、`turn/completed`、`item/started`、`item/completed`、`item/agentMessage/delta`、`item/commandExecution/outputDelta`、`item/fileChange/patchUpdated`、`item/plan/delta`、`item/mcpToolCall/progress`。App Server 只保留一个按 command/file-change/MCP/Plan owner 拆分的 `V2NotificationProjector`，对覆盖事件采用 direct / side-channel / reject 三态；畸形覆盖事件 fail closed，只有未覆盖的 provider/media 等旁路事件保留 `agentSession/event` raw envelope。Rust/TypeScript client、Electron 和 Renderer 已直接消费 v2 lifecycle/stream delta，delta 不伪造 durable sequence；`agentSession/event` 的 `typedEvent`、`canonicalEvent`、旧 lifecycle DTO、`agentSession/runtimeEvents/append` schema、fixture 和正向测试已物理删除。Renderer sequence gate 按 direct method/raw event 与 session 隔离，只允许 direct lifecycle/stream 与明确 raw side-channel，wrapper lifecycle/action 一律 fail closed；残留 `canonicalEvent` 字符串只允许作为待删除测试夹具或负向回流守卫，不是生产 contract。真实 Electron Gate B 必须证明动态 `turn/start` response、direct delta/Item/Turn terminal 与 `thread/read` Item 共享同一 thread/turn identity；Claw fixture 不得通过 wrapper event 或 deferred assertion 获得成功。`turn/start` response 只由 App Server/RuntimeCore admission owner 产生，Electron Host 必须按 generic JSON-RPC 转发；禁止从 notification、recent buffer、`thread/read` 或 `clientUserMessageId` 猜测并伪造 accepted response。

### 7.1 事件与完成态

- canonical live lifecycle 的 current contract 是 Codex v2 direct notification：`thread/started` 携带完整 `Thread`，`turn/started|completed` 携带 `threadId + Turn`，`item/started|completed` 携带 `threadId + turnId + ThreadItem` 和对应毫秒时间；AgentMessage/CommandExecution/Plan delta、FileChange patch update 与 MCP progress 都携带同一 canonical `threadId + turnId + itemId`。Rust/TypeScript client、Electron 与 GUI 只能向这些 typed notification + canonical read model 收敛。
- `typedEvent`、`canonicalEvent` 和 legacy lifecycle DTO 已是 `dead / deleted / forbidden-to-restore`。`agentSession/event` 当前只允许携带显式 allowlist 内的 raw 非 lifecycle side-channel；任何 Thread/Turn/Item 或 action wrapper 都必须拒绝，不能再通过 fixture、Renderer projector 或 client parser恢复。
- 单一 TypeScript codegen 继续对异构 nested `$defs` fail closed；禁止通过 allowlist、覆盖顺序、`Legacy*` 重命名、namespace compat 或第二套 flat codegen 选择其中一边。删除旧 lifecycle schema 后当前 codegen 为 731 个协议类型、0 生成失败、0 漂移；未来新增冲突必须修 owner，不得放宽生成器。
- 非 Thread 领域通知只能通过集中 allowlist 绕过 canonical sequence gate。当前允许 provider diagnostic、`runtime.status` 与 `image_task.presentation.generated/created/parameters.required`；它们只承载诊断或媒体任务展示，不得表达或修补 Thread/Turn/Item lifecycle。未知 raw event 与 raw Thread lifecycle 必须继续 fail-closed。
- 只有 production producer 全量发出 canonical entity、package/Renderer consumer 全量迁移、负向守卫覆盖旧 surface 后，S6 才能删除 raw lifecycle envelope；在此之前每个 slice 必须记录剩余 producer/consumer，不能把 optional canonical field 当作完成证据。
- `turn.completed`、失败、取消和中断是产品一等终态。
- 工具、审批、消息和产物是 Item/RuntimeEvent 的可投影活动，不是 renderer 私有日志。
- `request_user_input`/approval 由 session/turn scoped pending state 承接，不能建立进程全局单例。
- deprecated raw `action.required` 边界只能透传 runtime `data.availableDecisions`，不得在 App Server 或 Renderer 固定补一套按钮列表；退出条件是 GUI 分别消费 canonical Item 与其独立 typed server request，二者不得互相降级。
- canonical event log 的 gap、regression、equal-sequence divergence、malformed/unterminated tail 必须在 App Server repair 边界 fail-closed；只有可审计的尾部损坏允许按 `last_valid_offset` 截断并重建 ProjectionStore。
- canonical Item append 必须遵循 raw rollout append + metadata patch contract；ThreadStore apply 失败时 notification/mailbox ack 不得发生，restart/repair 从连续有效 durable tail 和 raw rollout 重建。
- Evidence refs、replay、analysis、review 与 GUI 从 App Server v2 read model 和 notification 消费同一 canonical 事实源；derived exports 只做下游投影，不提供 portable signed receipt。

#### 7.1.1 Renderer ConversationProjection 与 direct TurnTimeline

Renderer 的 current Item 状态 owner 已收敛到同一个 typed `ConversationProjection` reducer：

```text
direct App Server v2 notification
  -> direct notification router / sequence gate
  -> typed AgentEvent payload
  -> request-scoped ConversationProjection reducer
  -> canonical AgentThreadItem state
  -> pure canonical Turn render projection
  -> direct User / Agent / Media / Process segments
  -> MessageList

thread/read + thread/items/list + thread/turns/list
  -> bounded canonical history window
  -> canonical Item reader
  -> the same ConversationProjection reducer
  -> canonical AgentThreadItem state

thread/resume
  -> canonical Item reader
  -> install the same ConversationProjection reducer into active stream state
  -> subsequent live notifications reuse the same reducer

App Server reverse requests
  -> one PendingInteractionController
  -> semantic interaction identity only
  -> one actionable layer above Composer
```

direct notification 没有上游 event id 时，stream owner 只能在 router 去重之后分配 request 内单调到达序号；不能用内容 hash 去重，因为相同 chunk 可以合法重复，也不能把该序号当作跨重连 replay identity。Item identity 仍是 `thread_id + turn_id + item.id`，首次 canonical sequence 决定顺序，completed snapshot 权威覆盖 delta 草稿，terminal 后 late delta 只记录诊断。

Renderer session id 不得预绑定为 canonical thread id。live reducer 只能从 existing canonical Item 或首个带明确 direct-v2 protocol method 的事件建立 thread owner；无 protocol method 的 compat 事件继续委托 legacy lifecycle，不能抢占 current owner。session detail/read model 只接受 requested id 等于 canonical `sessionId` 或 `threadId` 的响应，其他 identity 必须在 App Server client 与 Renderer adapter 两层 fail closed，且不得写入 identity cache。

cold read 对 `historyMode=paginated` 的 `thread/read` embedded turns 一律视为 partial view，使用 store-owned `thread/items/list` cursor 选取 message window，再扫描 `thread/turns/list` 到 EOF，恢复选中 Item 所属 Turn 以及所有 active/failed/interrupted 等无 Item Turn。Renderer 分页以 opaque canonical Item ID `oldest_item_id` 为稳定锚点；旧数字 `oldest_message_id` 仅作迁出兼容并让位于 Item ID。`has_more` 是继续分页的事实，未到 Item EOF 时 `messages_count` 未知且不生成 `start_index`；只有 EOF 后才暴露精确总数和绝对索引，GUI 不得把观察下界显示为总数。

历史列表的首帧 DOM 必须继续受 Turn render window 约束，不能因为 canonical Item 已恢复就一次挂载全部历史。已完成的历史 assistant 纯文本复用唯一 `HistoricalAssistantMessagePreview`：正文超过 900 字先显示 compact preview，超过 24,000 字使用 2,000 字 long preview，用户显式展开后才挂载全文；streaming、A2UI 和含非文本 part 的 canonical Item 不得被错误折叠。受控 Electron Gate B 以 240 Turn / 720 Item SQLite fixture 证明首帧只挂载 10 个 canonical Turn、扫描 30 个 Item、residual Message 为 0、长正文尾部 marker 不进入 DOM；首次 MessageList paint 为 37ms、稳定 paint 为 200ms，long task、console error 与 page error 均为 0。该证据只证明受控历史恢复与 Renderer/Electron/App Server/read-model 链路，不代表 live Provider、真实用户历史或所有平台性能。

CommandExecution、Tool、WebSearch 与 Patch stdout/stderr 在 projection 边界限制为 256 KiB，超限保留尾部并带显式截断标记。unknown Item/notification 只向受控 diagnostics 记录 protocol revision、method、upstream type、Item identity 与脱敏字段名；raw 值、credential、metadata 和 provider payload 不进入 diagnostic。known-unprojected/unknown notification recorder 只提供 fail-visible 诊断，不能替代 V2-05 planned notification 的 producer、typed protocol、projection 或 Gate B。

canonical Item -> Message 的 tool/agent/reasoning `toolCalls/contentParts` 合成、`canonicalItemsToMessages` 与 MessageList canonical compatibility branch 已物理删除并有回流守卫，属于 `dead / deleted / forbidden-to-restore`。未被 canonical Item 覆盖的 optimistic/imported/local product surface 只作为 direct render projection 的 residual Message，不得成为第二个 Item store。HookPrompt、Sleep、Review boundary 使用 typed Item DTO；Hook 的 runtime identity 不进入可见 DOM。UserMessage audio/localAudio 不在 Lime 当前产品协议范围，reader fail closed；DynamicTool `inputAudio` 继续由真实 v2 schema/typed client/renderer 链承接。

command approval、file approval、requestUserInput 与 MCP elicitation 只由 `PendingInteractionController` 注册和终结。JSON-RPC transport id/action token 只能留在 dispatcher 请求闭包；React state 只持有 semantic interaction identity。旧 server-request controller、独立 MCP Dialog/controller 与第二 pending store 均为 `dead / deleted / forbidden-to-restore`。

Architecture impact: major; this changes the Renderer live/read/resume Item state owner, canonical Turn rendering and pending-interaction owner while preserving Electron as transport host, App Server/ThreadStore as canonical owner, and model-provider/tool-runtime boundaries. Architecture diagram updated: this section and the Renderer data-flow chain above. Responsible developer confirmation: root, 2026-07-29.

### 7.2 Plugin、Skills 与 MCP

Plugin v3 的 portable contract 固定为 Agent Plugins v1.0.0：包根 `plugin.json` 是唯一
manifest，包根 `skills/<skill>/SKILL.md` 与 `mcp.json` 是唯一组件位置。App Server plugin
domain 唯一拥有 discovery、install、installed、enabled、activation 与 package identity；Skills
与 MCP 分别由 `lime-skills`、`lime-mcp`、RuntimeCore 和 tool-runtime 注入当前 turn。Renderer
只消费 typed projection，Electron 只提供 Host 能力和转发 `app_server_handle_json_lines`。

```text
PluginCatalogPage
  -> Renderer pluginCatalog gateway
  -> typed AppServerClient
  -> Electron preload / app_server_handle_json_lines
  -> App Server Plugin processor
  -> RuntimeCore PluginDataSource
  -> plugin_catalog
       catalog discovery
       package validation + sha256 identity
       staging + atomic installed record
  -> <AgentRoot>/plugins/v3/{packages,installed,staging,marketplaces,data}
```

标准 manifest 必须声明
`https://agent-plugins.org/schemas/1.0.0/plugin.schema.json`；`name` 必填，`version` 可选且
不得被强制为 semver。未知顶层字段以及旧 `schemaVersion`、`contributions.runtime/workbench`
只报告并忽略，且永不参与发现、安装或激活语义；`app.runtime.yaml`、绝对或父级资源路径、
symlink 与超预算 package 必须 fail closed。安装 identity
至少包含 `pluginId + marketplaceId + version + contentDigest`；同 identity 同 digest 幂等，同
identity 不同 digest 拒绝。v3 不读取或转换旧 v2 installed/cache/data。

installed+enabled Plugin 由同一 store 生成 activation snapshot，冻结
`pluginId + version + contentDigest + marketplaceId + packageSourceUri`。Skills 只扫描 `skills/`
直接子目录并进入现有 skill snapshot；Claw 候选只从 typed `plugin/installed` 投影，Renderer
不得构造 activation、扫描包目录或维护第二 registry。未知、禁用或 identity 漂移必须 fail
closed，不能查询旧 renderer manifest/installed state 兜底。

```text
App Server Agent Plugins v3 installed store
  -> enabled activation snapshot
  -> RuntimeCore session config / Skill snapshot

Claw @ picker
  -> Renderer pluginCatalog gateway -> plugin/installed
  -> stable plugin://<pluginId> UserInput::Mention
  -> Electron IPC / app_server_handle_json_lines
  -> App Server mention selection
  -> selected activation -> Turn metadata / runtime context
```

Plugin MCP 只读取包根 `mcp.json`，必须声明 Agent Plugins v1 MCP schema。parser 先按官方
`stdio` / `streamable-http` contract 严格解析，再 lower 到内部 `McpServerConfig`；注入绝对
`PLUGIN_ROOT` 与持久化 `PLUGIN_DATA`，校验 placeholder、command/cwd containment、保留环境
变量、HTTP(S) URL 与 headers。文件级错误只禁用该 Plugin 的 MCP 组件，server 级错误只隔离
该 server；健康 sibling 继续进入 `McpThreadRuntime -> McpClientManager -> tool-runtime`。

Agent Plugins v1 没有 Lime 私有 worker、独立 UI runtime、renderer registry 或发布后台语义。
`pluginLocalPackage/*`、`pluginPackage/*`、`pluginInstalled/*`、`pluginHostLifecycle/*`、
`pluginShell/*`、`pluginUiRuntime/*`、Electron `plugin_runtime_*`、旧 `PluginManager` 与
`manifest.json` loader 全部属于 `dead / deleted / forbidden-to-restore`。Plugin MCP App 只从
canonical MCP Tool Item / resource 进入 Right Surface；非 MCP App/Hooks 能力只能按未来标准在
current owner 新建，禁止恢复旧 worker 或私有 manifest。

历史 v2 fixture 只能作为迁移 evidence，不是 v3 release evidence。v3 必须重新证明根标准包从
`plugin/install` 进入 enabled activation、Skills/MCP、canonical Item 与 Right Surface，并覆盖
Renderer reload/cold restore、卸载历史、macOS/Windows 和真实 Electron Gate B。

Architecture impact: major; portable package、runtime owner、Electron 边界与旧实现删除状态均已
改变。Architecture diagram updated: this section. Responsible developer confirmation: root,
2026-08-08.

MCP server 的执行环境身份只来自 `McpServerConfig.environment_id`，由 `lime-mcp::McpEnvironmentRegistry`
在 transport 启动前解析。当前 registry 只注册 `local`；未知显式身份必须 fail closed，禁止把
`remote` 或其它配置值降级成本机 stdio/HTTP 执行，也禁止从 `cwd` 猜测环境。Codex 的远端
`environment/{add,info,status}` 与 `thread/environment/*` 依赖 exec-server registry、WebSocket
连接恢复和远端 Thread 选择，Lime Desktop 没有对应产品 consumer，已归入
`product-scope-excluded / forbidden-to-restore`。远程 executor/backend 仍未接入 Lime current，禁止新增
compat fallback 或空壳 JSON-RPC。

本地 stdio 启动还必须复用 Codex 的平台核心环境变量 allowlist，并仅叠加配置显式 `env`；
不得把 Desktop Host 的完整环境（尤其凭证变量）隐式继承给 MCP 子进程。启动 deadline
严格使用配置的 `startup_timeout`，禁止按命令类型隐式放大。

### 7.3 Scheduled Tasks

已安排任务属于 App Server 调度领域，产品入口、协议、持久化和 Agent 执行沿单一主链：

```text
ScheduledTasksPage
  -> Renderer scheduledTasks typed gateway
  -> Electron Desktop Host / app_server_handle_json_lines
  -> App Server scheduledTask/* processor
  -> RuntimeCore
  -> LocalAppDataSource automation owner
       automation_jobs（当前唯一任务表）
       agent_runs（运行历史）
  -> RuntimeCore thread/session + turn submission
  -> canonical Thread / Turn / Item projection
  -> App Server typed scheduledTask/changed + scheduledTask/run/updated
  -> Renderer Scheduled Task notification policy / refresh bridge
  -> Electron Desktop Host system notification (壳能力)
  -> ScheduledTasksPage history / open conversation
```

公开 surface 为 `scheduledTask/list|read|create|update|delete`、`scheduledTask/enabled/set`、
`scheduledTask/run/start|list` 与 `scheduledTask/schedule/preview`。Schedule wire 使用
`hourly/daily/weekdays/weekly` 和 Codex weekday `MO..SU`；这是 Lime Desktop 的产品 CRUD 扩展，不能表述为 Codex
Desktop 已存在同构实现。任务写入继续复用 `automation_jobs`，不得新增第二张长期任务表或 Renderer store；运行历史从
`agent_runs.source_ref=taskId` 投影。`new_thread` 每次运行产生新 lineage，`continue_thread` 必须使用显式来源 lineage；
运行返回真实 `sessionId/threadId/turnId` 后 GUI 才能恢复 canonical 对话。

Electron 只负责 JSONL 转发和系统通知宿主能力，不承接 scheduler、任务 CRUD、运行状态或 timer。App Server 是任务变更与 canonical
terminal event 的 typed notification owner：任务 create/update/delete/enabled-set 发布 `scheduledTask/changed`，
`turn.completed/failed/canceled` 经过 Agent Run 幂等终态写回后发布 `scheduledTask/run/updated`。Renderer 只按任务的
`all_runs / failures / none` policy 决定是否调用 Desktop Host；`unsupported/failed` 必须进入可见错误通道，不能伪造发送成功。
	Renderer timer、生产 mock backend/fallback、browser session automation 与 SceneApp automation context 均禁止成为 current owner。旧
	`automationJob/*`、`automationSchedule/*`、`automationScheduler/*`、Renderer automation gateway 及旧 Settings 工作台已物理删除，
	分类为 `dead / deleted / forbidden-to-restore`；旧 method 字符串只允许作为 contract、fixture 或治理扫描的负向回流守卫。Rust
	`AutomationJob` DAO、`automation_jobs` 表与内部 execution helper 是 Scheduled Tasks 的 current 存储映射，不是公开双轨。scheduler 原子 claim、
missed/catch-up、DST、通知、软删除及并发运行合同已由 current owner 收口；任务删除写入 tombstone、禁用并清除未来调度，
但不取消已运行 Turn，canonical terminal write 仍保留 Agent Run 历史且不得复活 tombstone。真实 OS sleep/wake、Windows
Notification Center 与 Windows Gate B 仍是平台证据缺口，按 `internal/roadmap/task/scheduled-tasks/` 和对应执行计划收口。

Architecture impact: major; this adds the Scheduled Tasks public JSON-RPC/read-model boundary and top-level GUI workspace while
preserving Electron as transport/system-notification host and RuntimeCore/ThreadStore as the execution truth. Architecture diagram
updated: this section and `internal/aiprompts/commands.md#scheduled-tasks-主链`. Responsible developer confirmation: root,
2026-08-17. Confirmation content: 已核对 scheduledTask JSON-RPC、typed invalidation/terminal notifications、Agent Run
幂等终态、soft-delete tombstone、Renderer notification policy、Electron system notification boundary，以及未完成的
	Windows/sleep-resume 平台证据，以及旧 Automation 协议、GUI、client、smoke 与 Agent UI projection 的物理删除和回流守卫。

## 8. 命令、配置与数据边界

### 8.1 Turn 请求字段归属

`turn/start` 只有一组 Codex v2 请求结构。Renderer 必须由 typed gateway 构造；
`threadClient` 只负责转发、通知路由与 read-model 投影，不再接受或转换第二套
snake_case runtime request。

| 边界                            | 唯一 owner                              | 字段                                                                                                                                                                                                                                             |
| ------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TurnStartParams`               | Codex v2 Turn 协议                      | `threadId`、typed `input[]`、model/effort、cwd/workspace roots、approval/sandbox/permissions、output schema、service tier、environment、collaboration/personality 与 typed context metadata；deprecated typed `multiAgentMode` 被 runtime 忽略。 |
| `TurnSteerParams`               | Codex v2 Turn 协议                      | `threadId`、`expectedTurnId`、typed `input[]`、`clientUserMessageId`、additional context 与 Responses API client metadata。                                                                                                                      |
| `RuntimeRequest`                | App Server -> RuntimeCore 内部 lowering | provider/model/config、typed `collaborationMode`、reasoning、approval/sandbox、workspace 与执行 metadata；Renderer 不再直接构造或传输该内部 DTO。                                                                                                |
| `RuntimeOptions` / queue fields | dead for current turn protocol          | `runtimeOptions`、`queueIfBusy`、`skipPreSubmitResume`、caller-supplied `turnId` 不属于 v2 `turn/start`，不得通过 alias、metadata 或 compat wrapper 回流。                                                                                       |
| `hostOptions`                   | dead / deleted                          | 不在 current turn 协议中；不得作为宿主扩展、provider route、tool policy、workspace、session context 或任意 runtime JSON escape hatch 恢复。                                                                                                      |

`collaborationMode` 采用 Codex 的 `ModeKind + settings` 结构，由 `turn/start` typed wire 进入
`RuntimeRequest`，再进入 `TurnContextOverride` 和 Goal admission；Plan 判定只读取
`ModeKind::Plan`。GUI submit adapter 负责把当前输入框状态降为该结构并从
`additionalContext` 移除旧的 mode 字段，goal 等非模式 metadata 不得承接 mode 语义。

Renderer submit gate 只把 canonical `thread/read` 当作 start/steer 的输入事实：先解析
`threadId`，再读取唯一 `activeTurnId`；有 active turn 时提交 `turn/steer` 并携带精确
`expectedTurnId`，无 active turn 时才创建 `turn/start` 的 optimistic lifecycle。Renderer
本地 `ActiveStreamState.turnId`、busy/queued flag 和 draft status 都不是调度事实；ID 冲突
必须 fail closed，不能退回 start 或伪造 queue。steer 接受后不创建第二个 Turn、不绑定第二个
事件 listener，并刷新同一 canonical read model 以投影用户输入与原 Turn identity。Renderer
不暴露 queued user turn 的 promote/remove 写操作；durable pending-work 的排队、恢复与续跑只归
RuntimeCore/session loop，GUI 只消费 canonical status、count 与 evidence。

前端不会持有或修补 Thread/Turn/Item 真相。输入发送后，用户消息、流式增量、终态、
工具活动和历史恢复均从 App Server notification 与 canonical Thread read model 投影到 UI；
v2 `thread/read` 直接返回同一 ThreadStore-backed read model；固定 timeout、renderer cache、
host payload 或缺失 detail 的本地 fallback 不能合成 history 或完成态。

Renderer 不再接收或投影 user-turn 的 `queue_added`、`queue_removed`、`queue_started`、
`queue_cleared` raw side-channel，也不再将 `QueuedTurnSnapshot` 的 message/attachment/path
细节恢复为输入草稿。上述路径已删除，不得通过 protocol parser、stream lifecycle、projection
package 或 workspace prop 重新引入。Plugin/task 的通用 `task:queued -> queue.changed`、
Multi-Agent 的队列计数，以及 RuntimeCore/session loop 的 durable pending-work/recovery 仍由各自
current owner 负责；它们不是 user-turn queue GUI 的兼容替身。

一次跨层命令改动必须同时更新：

1. Renderer gateway 或 `packages/app-server-client`。
2. Electron preload/IPC 白名单（仅宿主转发需要时）。
3. `app-server-protocol` schema、App Server handler 与 Rust client。
4. catalog、受控 fixture、mock policy 和 `npm run test:contracts`。

生产路径不得回退 `defaultMocks`、`mockPriorityCommands`、`invokeMockOnly`、renderer mock 或 mock backend。受控 fixture 可以使用 external backend，但必须经过真实 App Server、read model 与产品 event 链。

配置变更必须成组更新 schema、validation、consumer、默认值、文档和 lockfile；用户数据、缓存、日志和凭证必须走统一路径/平台 API，不写入仓库或硬编码平台目录。

## 9. 脚本、文档与测试目录

### 9.1 Scripts

`scripts/` 是质量与自动化入口：

- `scripts/agent-runtime/`：Agent runtime fixture/smoke。
- `scripts/app-server/`：sidecar 与协议 smoke。
- `scripts/electron/`：真实 Electron fixture。
- `scripts/governance/`：结构与旧路回流检查。
- `scripts/harness/`：evidence/harness 验证。
- `scripts/i18n/`：locale 边界检查。
- `scripts/mcp/`：MCP smoke。
- `scripts/plugin/`：plugin fixture。
- `scripts/playwright/`：浏览器/Electron 交互辅助。
- `scripts/smoke/`：跨域最小 smoke。
- `scripts/lib/`：脚本共用实现。

根 `scripts/` 与一级目录是冻结边界。新脚本必须归入既有领域或 package；例外同时更新 `scripts/README.md`、基线和执行计划。

### 9.2 文档

| 文档位置               | 写入规则                                    |
| ---------------------- | ------------------------------------------- |
| `internal/aiprompts/`  | current owner、边界、验证、目录规范。       |
| `internal/exec-plans/` | 计划、责任人确认、进度、验证结果、blocker。 |
| `internal/roadmap/`    | 后续目标与优先级，不覆盖 current owner。    |
| `internal/research/`   | 外部对照与审计证据。                        |
| `docs/`                | 对外站点内容。                              |

历史路径只能留在历史 evidence；不得出现在 current 导航、架构规则、active checklist 或新代码说明中。

### 9.3 测试与交付证据

| 风险              | 最低验证                                                                  |
| ----------------- | ------------------------------------------------------------------------- |
| 纯逻辑/投影       | 定向 unit test。                                                          |
| Rust domain       | 受影响 crate 的 test/check，再按风险扩大。                                |
| JSON-RPC / bridge | `npm run test:contracts` + 定向 Rust/TS 测试。                            |
| Agent runtime     | `npm run smoke:agent-runtime-current-fixture` + 相关 current fixture。    |
| GUI 主路径        | `npm run verify:gui-smoke`。                                              |
| 真实桌面闭环      | Gate B：Electron、preload、IPC、App Server、runtime/read model、可见 UI。 |
| 发布/配置         | `npm run verify:app-version` 和对应 release/Forge 检查。                  |

Gate A 只证明 browser/renderer projection；Gate B 才证明真实 Electron 产品链。两者不得混用。

## 10. 依赖方向与禁止边界

```text
Renderer -> typed client / Desktop Host bridge -> App Server -> runtime owners
App Server -> agent-runtime, model-provider, tool-runtime, thread-store, domain services
agent-runtime -> agent-protocol, model-provider, tool-runtime
thread-store -> agent-protocol and storage primitives
model-provider / tool-runtime -> protocol and low-level utilities only
```

禁止：

- Renderer 导入 Rust implementation 语义或直接调用 provider/tool runtime。
- Electron 保存 RuntimeCore 状态、复制业务 API 或解释模型 stream。
- `model-provider`、`tool-runtime` 反向依赖 App Server、Electron 或 React。
- App Server handler 内拼 provider wire payload、实现工具权限或复制 tool dispatch。
- `core`、`services` 成为无明确边界的 runtime 垃圾桶。
- 已删除目录、旧 wrapper、临时 adapter 或 mock fallback 重新成为 current owner。

## 11. 重大架构变更与开发者确认

### 11.1 何时属于重大变更

满足任一项即为重大架构变更：

1. 新增、删除、移动或合并顶层目录、Rust crate、TypeScript package、运行时 host 或持久化 owner。
2. 改变 Renderer、Electron、App Server、RuntimeCore、provider、tool 或 Thread/Turn/Item 的职责/依赖方向。
3. 新增或替换 JSON-RPC transport、初始化握手、跨层 method、schema、event 或 read model。
4. 改变 provider protocol/lowering、工具权限/执行、MCP/Skill 注入或媒体 part 的唯一 owner。
5. 改变 session/thread/turn/item 持久化、ProjectionStore、canonical read model、replay 或历史恢复事实源。
6. 改变主窗口/独立窗口路由、Electron Host 模式、Forge 打包/更新链或跨宿主产品入口。
7. 改变 Gate A/Gate B 证据等级、GUI 交付门槛或生产 mock 边界。

纯局部 bug 修复、保持 owner 不变的内部重构、只改文案或只补测试不属于重大架构变更，但只要不确定，就按重大处理。

### 11.2 必须更新与确认

重大架构变更在实现同一变更集中必须：

1. 更新本文件中受影响的目录地图、owner、数据流、依赖方向或验证门禁。
2. 更新相关 `internal/aiprompts/` 领域文档和 `internal/exec-plans/` 执行计划。
3. 在执行计划和 PR 描述中填写以下确认，由负责开发者明确勾选并署名。每个 PR 都必须声明重大或非重大；触及架构敏感路径时，`npm run governance:architecture-confirmation` 只接受重大声明：

```text
架构影响：<重大变更项，或“无”>
架构图已更新：<章节/路径，或“不适用：原因”>
责任开发者确认：<姓名或账号>，<YYYY-MM-DD>
确认内容：已核对目录归属、数据流、依赖方向、协议边界和验证门禁。
```

未更新架构图、未写不适用原因或没有责任开发者确认的重大变更，不得标记为完成、不得进入 release evidence、不得合并为 current 架构结论。

### 11.3 评审问题

重大变更评审至少回答：

1. 新 owner 是否是唯一事实源，旧 owner 是否已删除或明确退场？
2. 跨层调用是否仍沿 Renderer -> Host -> App Server -> runtime 方向？
3. 状态是否仍通过 Thread/Turn/Item、RuntimeEvent、ProjectionStore/read model 收敛？
4. provider lowering 与 tool execution 是否仍在各自 owner？
5. 验证是否覆盖对应的 contract、fixture、Gate A 或 Gate B？

FileChange batch architecture confirmation: root, 2026-07-21. The canonical owner remains
`agent-protocol -> App Server materializer -> v2 projection`; no Electron, renderer, or raw
file-event owner was added.

## 12. 实施前检查

新增或重构前依次回答：

1. 它属于哪一个目录与 owner？为什么不是相邻现有 owner？
2. 是否需要新 JSON-RPC 契约，还是已有 method 足够？
3. 产生/消费哪些 Thread、Turn、Item、RuntimeEvent 和 read model？
4. provider、工具、持久化、Renderer 各自是否仍只做自己的职责？
5. 是否触发重大架构变更？若触发，架构图和开发者确认落在哪里？
6. 最小验证是 unit、contract、fixture、Gate A 还是 Gate B？

任何问题无法回答时，先补执行计划或架构图，不在临时 facade、无 owner helper、旧目录或 UI 层堆实现。

## 13. v2 Turn Admission and Background Completion

The current v2 `turn/start` path follows single-owner submission semantics:

```text
JSON-RPC turn/start
  -> RequestProcessor serialization scope
  -> RuntimeCore admission
  -> RuntimeSession actor submission
  -> immediate inProgress response
  -> owned completion driver
  -> RuntimeCore state/event-log/ProjectionStore append
  -> canonical ThreadGoal idle continuation (when active and idle)
  -> RuntimeEventHub
  -> one App Server v2 notification projector
  -> transport/GUI
```

`RuntimeCore::start_turn` remains the synchronous owner for internal workflows that still require
the completed result. v2 `processor/turn.rs` uses `start_turn_admitted`; it returns the canonical
turn identity after the session actor accepts the task and never captures the request's borrowed
event callback in the background task. Completion, failure and terminal events are appended by
the same RuntimeCore sink and forwarded through `RuntimeEventHub`. After a terminal completion,
the background driver re-enters the canonical ThreadGoal idle gate. The ThreadGoal owner applies
its status, budget, usage and pending-work guards before admitting a new Turn. The continuation
worker owns its callback and publishes through the same hub; it never captures the request callback
or creates a second turn queue.

Turn admission is actor-first and intentionally two-phase. RuntimeCore may submit steer only to an
already existing session actor. If that actor accepts the input, RuntimeCore resolves the canonical
active Turn and returns it; a missing canonical Turn fails closed. If no actor exists, or the actor
has no steerable active task, RuntimeCore performs the canonical new-turn/queue state check and then
submits the candidate task. RuntimeCore must not scan its read model first and bypass this decision
with a second steer path.

All non-user session operations use the same actor queue. Their handler context carries canonical
session identity, submission identity, active Turn identity, client message identity and trace
metadata. A variant is not production-complete until an App Server method and its existing domain
owner consume it. A same-named but semantically different memory, export, MCP-management or shell
surface must not be wired as a placeholder.

The App Server owns one event pump and one `V2NotificationProjector` instance per runtime. This is
required because turn lifecycle de-duplication is projector state, not transport state. Electron
is only a JSON-RPC transport/desktop host and must not infer turn identity by issuing a second
`thread/read` or by waiting for a streaming ACK.

Architecture impact: major; this changes the request completion boundary and the runtime-to-
transport event ownership while preserving the existing App Server JSON-RPC method and
Thread/Turn/Item projection owners.

Architecture diagram updated: this section and the main chain above.

Responsible developer confirmation: root, 2026-07-19.

Confirmation: the admission boundary, owned completion task, state/event-log append path,
RuntimeEventHub, single v2 projector and transport fan-out were checked against the Codex
`turn_processor`/session listener split. Admitted ThreadGoal continuation is connected to the same
RuntimeCore and event hub owners; its focused integration test must pass before this slice is used
as release evidence. The retired v0 ManagedObjective worker is not part of this architecture.

## 14. v2 Thread Resume Cold-Recovery Boundary

The current v2 `thread/resume` path has one App Server owner:

```text
JSON-RPC thread/resume
  -> v2 ingress and canonical threadId serialization scope
  -> RequestProcessor::handle_thread_resume_v2
  -> RuntimeCore::resume_thread
  -> canonical ThreadStore read + session hydration
  -> Thread/Turn/Item projection and optional turns page
  -> per-thread listener generation
  -> subscribe exact transport connection
  -> JSON-RPC response
  -> canonical token-usage snapshot
  -> canonical ThreadGoal updated/cleared snapshot
  -> pending typed server-request replay
  -> subsequent live AgentEvent projection
```

Resume never accepts or resolves the legacy `sessionId` shape, never pops a queued turn, and never
emits `thread/started`. `excludeTurns` controls the durable thread snapshot while
`initialTurnsPage` reuses the `thread/turns/list` owner. The listener owns response/replay/live event
ordering and thread-scoped connection isolation; history/path and runtime-config overrides still
fail closed rather than being acknowledged without changing durable or live state. The v2
serialization scope keys only on `threadId`; the v0 session resolver remains outside this method
boundary.

Architecture impact: major; this introduces the canonical public resume boundary and removes the
old queued-session resume route from the method dispatch path.

Architecture diagram updated: this section and the v2 App Server chain above.

Responsible developer confirmation: root, 2026-07-21.

Confirmation: the public JSON-RPC tests prove same thread/session identity, metadata-only resume,
initial turns page parity, paginated-history constraints, and fail-closed legacy/source/override
inputs. The listener owns exact connection subscription, detached pending-request claim,
thread-scoped event routing and ordered response/token-usage/ThreadGoal/pending/live enqueue. Raw
JSONL reconnect covers owner reclaim and the same replay barrier. Terminal ThreadGoal usage 已具备
durable baseline/watermark/outbox 与 listener FIFO；带完整 cumulative snapshot 的
completed/failed/canceled 已幂等入账，`provider.usage` 中间快照已覆盖异常前的已报告 usage；
普通 failure 与 provider usage-limit 已通过 structured reason 分别推进 `blocked` 与
`usage_limited`；Turn 中途首次创建 Active Goal 已从 mutation 时的 canonical cumulative usage late-bind，
不会计入创建前 token；已有 Goal external mutation 已原子 flush/rebind，pause 区间与 clear 前旧 binding
不会污染后续 Goal。生产 failed/canceled usage producer 的完整 flush、tool-finish/abort flush、
跨重启 provider history lowering、outbox crash-drain 和 Codex
thread-scoped first-terminal MCP 语义仍是明确 follow-up，不能把本切片报告为完整 Codex resume
parity。

## 15. Model Reroute Transient Notification Boundary

Codex `model/rerouted` 只表达 first-party Responses 服务端因高风险网络安全活动改变实际模型，不能复用
普通 provider retry/fallback。current 数据流为：

```text
trusted OpenAI Responses server-model metadata
  -> model-provider requested/server ASCII case-insensitive comparison
  -> RuntimeCore canonical ModelReroute
  -> Agent Runtime Turn-level de-duplication
  -> App Server cross-route Turn-level de-duplication
  -> transient RuntimeEvent sink (no state/EventLog append)
  -> exact v2 model/rerouted notification
  -> transport/GUI consumer
```

`model.server_reported` 继续走 durable diagnostic evidence，并包含 provider、requested model、selected model
与 route attempt；它不直接投影为公开通知。普通 runtime provider failure 仍只产生
`routing.fallback.applied`。selector、展示名、第三方兼容 endpoint 与 `response.model` 均不能建立 reroute
信任。App Server transient sink 只绕过持久化，不绕过 thread/turn identity 与 v2 typed projector；因此 cold
resume 不重放 reroute，畸形 payload 也不会回退 deprecated side channel。

Architecture impact: major; this adds an explicit non-durable branch to the existing runtime event pipeline while
preserving `model-provider` as the network trust owner, Agent Runtime/App Server as Turn orchestration owners, and the
v2 projector as the public protocol owner.

Architecture diagram updated: this section and the canonical chain above.

Responsible developer confirmation: root, 2026-07-27.

Confirmation: trusted mismatch, case-only equality, third-party endpoint rejection, sampling/route de-duplication,
ordinary provider fallback isolation, transient publication and exact v2 round-trip were checked against Codex current
semantics. No Renderer/Electron, compatibility wrapper or second provider backend was added.

## 16. Gemini GenerateContent Current Transport

Gemini API Key chat uses the same multi-model control plane and provider owner as OpenAI/Anthropic; OpenCode supplies
the wire/lowering reference, while RuntimeCore and Thread/Turn/Item ownership remain aligned with Codex:

```text
provider store/catalog/readiness
  -> RuntimeCore GeminiGenerateContent route admission
  -> model-provider canonical request lowering
  -> Google streamGenerateContent SSE
  -> canonical text/reasoning/tool/usage/finish events
  -> Agent Runtime tool loop
  -> ToolLifecycleEvent provider metadata
  -> ThreadItem.metadata persistence
  -> provider history lowering for later Turns
```

The current wire is `POST /v1beta/models/{model}:streamGenerateContent?alt=sse` with `x-goog-api-key`; Bearer auth is
forbidden. Canonical lowering covers system instruction, user/model roles, inline base64 images, function declarations,
function calls/results and generation controls. Remote HTTP media, unsupported content parts, blocked prompts, malformed
tool calls and truncated streams fail closed. Gemini `thoughtSignature` remains provider metadata: it is preserved on the
assistant function call, copied through generic tool lifecycle metadata, stored on the canonical Thread item and restored
when historical tool calls are lowered into a later provider request. It is not promoted to a product-level field.

`GeminiGenerateContent`, Vertex Gemini and Azure OpenAI Responses are `current`. Azure keeps a dedicated provider identity
through the shared Responses algebra so its `api-key` and typed `api-version` wire cannot degrade to OpenAI Bearer auth.
Vertex keeps a dedicated runtime identity through the shared Gemini canonical algebra so its project endpoint and Bearer
token cannot degrade to the Gemini API-key wire. Bedrock Converse and Fal chat remain unsupported and must not be admitted
through aliases or custom protocol strings.
Ollama Chat is deleted; Ollama now shares the current Responses algebra described below. No compatibility adapter or
parallel provider backend exists.

Architecture impact: major; this extends the current provider transport union and durable tool-history contract without
changing owner direction. Architecture diagram updated: this section and the provider/runtime chain above. Responsible
developer confirmation: root, 2026-07-27.

## 17. Ollama Responses Current Transport

Codex HEAD removed `wire_api = "chat"` and the `ollama-chat` provider. Lime follows the same single transport algebra:

```text
Ollama provider store + /api/tags discovery
  -> RuntimeCore OpenaiResponses route + NoAuth
  -> model-provider canonical Responses request
  -> POST /v1/responses without Authorization
  -> shared Responses SSE reducer
  -> Agent Runtime Thread/Turn/Item chain
```

Provider identity remains the resolved selection/provider id; the App Server does not relabel a keyless Ollama route as
`openai`. `ProtocolKind::OllamaChat`, `ollama_chat`, NDJSON agent turns and Ollama-specific lowering are
`dead / deleted / forbidden-to-restore`. `/api/tags` remains the independent model-discovery endpoint and must not become
a second execution transport. Hosted OpenAI verification/reroute evidence remains disabled because an Ollama route is not
a trusted first-party OpenAI endpoint.

Architecture impact: major; this removes a public protocol variant and admits Ollama through the existing Responses
owner without adding a provider-specific wire owner. Architecture diagram updated: this section and the provider/runtime
chain above. Responsible developer confirmation: root, 2026-07-27.

## 18. Official Responses Hosted Web Search

Hosted web search follows the Codex Responses item lifecycle without creating a second network or tool-execution owner:

```text
official OpenAI/Codex Responses route + exact api.openai.com host
  -> canonical WebSearch tool definition
  -> model-provider { type: "web_search", external_web_access: true } lowering
  -> Responses web_search_call item
  -> canonical provider-executed ToolCall / ToolResult
  -> Agent Runtime provider tool lifecycle
  -> Thread/Turn/Item projection + exact raw Responses item history
```

The hosted capability is true only when the resolved provider type selects the Responses protocol and the final endpoint
host is exactly `api.openai.com`. OpenAI-compatible gateways, Ollama, Chat Completions and unknown routes retain no hosted
capability and cannot promote a function tool to `web_search`. Only the current canonical `WebSearch` definition is
eligible; legacy aliases and MCP-shaped names remain ordinary function tools. A `provider_executed=true` search emits
started/completed Item lifecycle with environment `provider`, preserves the raw response item for later Responses
history, and never enters the local `WebSearch` executor or changes the response finish reason to a local tool call.

The official request lowering, Responses reducer, provider-executed lifecycle and capability projection are `current`.
Provider-name-only capability guesses, third-party hosted promotion, alias-based promotion and provider-executed search
falling through to local execution are `dead / forbidden-to-restore`; no `compat/deprecated` path exists.

Architecture impact: major; this adds a provider-executed tool branch to the existing provider/runtime chain while
preserving `model-provider` as network owner and `tool-runtime`/Agent Runtime as lifecycle owners. Architecture diagram
updated: this section and the provider/runtime chain above. Responsible developer confirmation: root, 2026-07-27.

## 19. Model Capability Provenance And Route Admission

Model catalog hints and executable route facts have separate trust levels:

```text
canonical registry / provider-explicit capability fields / typed direct config
  -> EnhancedModelMetadata capability_provenance
  -> App Server modelRegistry.modelCapabilities.provenance
  -> RuntimeCore authoritative snapshot admission
  -> capability gap check + resolved provider route

provider name / model name / models[] entry without capability
  -> inferred_hint catalog metadata
  -> picker/search diagnostics only
  -> capability_snapshot_missing
```

The only authoritative provenance values are `canonical` and `provider_explicit`. `inferred_hint` remains useful for
catalog grouping and display, but RuntimeCore must reject it before provider execution even when the inferred object is
non-empty. Direct runtime config is authoritative only when it carries an explicit capability snapshot. Renderer
`model/list` projection preserves Codex picker, reasoning-effort and input-modality fields while forwarding the selected
catalog record's typed `providerId` and `capabilitySnapshot` unchanged. It also forwards known context/output limits. The
optional `multiAgentVersion` is forwarded only when the catalog explicitly declares `disabled`, `v1`, or `v2`; absence
remains `null`, and provider name, model name, task family or tool capability must never infer a version. This model-declared
runtime support field does not replace the Grok-aligned provider route/readiness owner or Lime's Multi-Agent lifecycle
owner. The Renderer reads capabilities, task families, input/output modalities and runtime features from that snapshot and must not
synthesize tools, streaming, JSON mode, function calling, capability provenance or limits. `inferred_hint` is excluded
from executable `model/list` rather than being relabeled locally. Executable chat entries are checked by the same canonical
`ModelTaskRequest + route_capability_gap` contract used by Turn admission: they must explicitly advertise the `chat` task
family, accept `text`, produce `text`, and support streaming. Reasoning and vision may be additional capabilities, but cannot
replace the chat contract. Their current sampling input contract
is narrowed to ordered `text | image` parts, matching grok-build; image/audio/video/file-only models remain outside the chat
picker and are admitted only by their dedicated media owner. `CapabilitySnapshot.source` is the nested capability
`provenance`, while the registry lookup source remains `modelRegistry.source`; `reasonCode` records matching or lowering
reasons such as `chat_wire_text_image_only` and is never used as capability provenance.

Provider configuration has one shape: `Provider.models[]` entries contain `id`, optional `displayName` and optional
`capability`. Entries with capability become `provider_explicit`; entries without it remain `inferred_hint` and fail
closed. Protocol, endpoint and authentication can still be resolved for diagnostics, but an id-only entry cannot
authorize a Turn. The product-owned `lime-hub` gateway is the only additional hosting-provider identity allowed to map
an id-only entry through the bundled canonical registry: admission becomes `canonical` only when the model resolves to an
existing canonical record; unknown OEM model ids remain `inferred_hint` and fail closed. This restores persisted Windows
Lime Hub selections without mutating user data or treating the gateway name itself as capability evidence.
`modelProvider/fetchModels` carries backend capability provenance into the typed Renderer gateway; live API metadata
wins over same-id local hints, and model selectors exclude `inferred_hint` entries instead of offering a route that
RuntimeCore will reject. Bundled provider records may authorize documented upstream models such as Agnes 2.5, while
unknown chat, image and video ids remain catalog-only hints. In particular,
the Ollama Responses transport remains the single current wire owner while its
discovered id-only model entries are not execution-ready until provider configuration supplies a typed capability
snapshot. The deleted `custom_models/customModels` fields, name-based route authorization and Renderer false/default
capability fabrication are `dead / deleted / forbidden-to-restore`; no `compat/deprecated` path exists.

Architecture impact: major; this changes the trust boundary between model discovery, App Server route metadata and
RuntimeCore admission without adding a second catalog or execution owner. Architecture diagram updated: this section and
the model-provider/runtime chain above. The public catalog projection was extended on 2026-07-28 to preserve the same
authoritative snapshot through the Renderer boundary; it does not add a second owner. Responsible developer confirmation:
root, 2026-07-28.

## 20. Model Catalog Refresh And Turn Selection Reconciliation

Catalog refresh and durable Thread selection use one admission path:

```text
provider store mutation -> model_route_generation
  -> ready provider-scoped catalog
provider credential create/enable/delete
  -> foreground provider-scoped catalog refresh
       | success: credential-scoped cache + provider last-success snapshot
       | transient failure without last-success: one retry flight per provider
           -> immediate background retry, then bounded 5s exponential backoff
           -> at most 5 attempts, 60s delay cap
  -> successful cache transaction advances model_route_generation
  -> model/list/updated { generation, providerId }
  -> Renderer cache invalidation + forced model/list read
  -> App Server isDefault / ready catalog order is the only default-selection fact
  -> pending route recovery for the committed generation
  -> RuntimeCore::start_turn_inner reconciliation
  -> visible + authoritative + chat-capable candidates
  -> current route preflight
       | valid: keep current selection
       | rejected/missing: same-provider candidate, then catalog order
  -> session actor thread-settings preflight + durable metadata update
  -> thread/settings/updated
  -> Turn admission with the reconciled provider/model/effort/service tier
```

Every production Turn entry, including public `turn/start`, synchronous RuntimeCore callers, queued resume, ThreadGoal
continuation, workflow retry and mailbox TriggerTurn, reaches the same reconciliation boundary before provider execution.
Public `thread/start` resolves an omitted provider/model pair to the first ready, visible, authoritative chat model in the
same catalog order that produces `model/list.isDefault`, then performs the exact provider/model route preflight before
creating the session or thread. Partial or blank explicit routes fail closed. An unready provider, unknown model or
capability gap leaves no durable session/thread side effect. Switching provider/model without an explicit service tier
installs the target catalog model's validated default tier, or clears the previous tier when the target has no default;
model-only changes likewise clear an effort that the target model does not support. Renderer selection consumes only App
Server `model/list` and its `isDefault` bit. The Agent warmup and session gateway do not read `get_default_provider`, require
an explicit route, or arbitrate a local default; configured-provider reads remain diagnostics rather than a parallel
selection owner. Public v2 chat `InputModality` is exactly `text | image`; audio/video/file remain general provider catalog
taxonomy until a dedicated chat ingress, durable sidecar and exact wire lowering exist.
The boundary reads the route generation before and after catalog construction and retries at most three times; continuous
generation churn or the absence of an executable candidate fails closed. Candidate ordering preserves the current provider
first and then catalog order. Hidden models, `inferred_hint` capability records and explicit non-chat task families cannot
be selected. A catalog-present current selection is still checked by route preflight so stale credential, effort or service
tier state cannot bypass normal admission.

Credential mutation never creates a second catalog owner. The foreground refresh and every background retry pass through
the same provider-scoped coordinator and `ModelProviderAppDataSource`; overlapping retry loops for one provider are
suppressed. Background retry is scheduled only for `network`, `invalid_response`, or request-backed transient `other`
failures when no provider-level last-success snapshot exists. Authentication, permission, not-found and configuration
failures stop immediately. A successful retry commits the normal cache transaction, publishes typed `model/list/updated`
and schedules pending route recovery; Renderer does not poll or infer generation. Existing last-success data remains the
visible catalog during credential rotation failure and does not start a redundant retry loop.

Typed direct provider config is outside catalog replacement. An explicit request with a direct API key/base URL is retained,
and a durable AgentControl route records `routeSource=direct_provider_config`; neither is rewritten merely because the model
is absent from the provider catalog. A catalog route records `routeSource=catalog`. When reconciliation changes selection,
the previous route snapshot/provider config is removed, model/reasoning/service-tier defaults come from the chosen catalog
entry, and the existing session actor `thread/settings/update` preflight remains the only persistence gate.

Foreground `turn/start` returns exactly one `thread/settings/updated` notification in its response dispatch. Reconciliation
performed by background/internal Turn entry publishes the same exact notification through a transient RuntimeEvent and the
single v2 projector; it is not appended to EventLog and is not replayed by cold resume. The catalog, durable Thread settings,
RuntimeCore route preflight and v2 notification remain `current`. Silent fallback, inferred capability authorization,
catalog replacement of direct routes, stale route reuse, preflight-after-persist and a second Renderer model-selection
store are
`dead / forbidden-to-restore`; no `compat/deprecated` path exists.

Selection/default/reconciliation and public picker projection are owned by
`runtime/model_providers/selection.rs`; Provider CRUD, catalog refresh coordination and retry permits remain in
`runtime/model_providers.rs`. A dedicated media model can remain in the provider catalog for image/audio tasks, but it cannot
be retained as an Agent chat Thread selection or bypass `thread/start` chat preflight. Mixing those two admission paths is
`dead / forbidden-to-restore`.

Architecture impact: major; this adds a generation-aware admission stage and a transient settings-notification branch while
preserving the existing provider, session actor, Thread metadata and v2 projector owners. Architecture diagram updated: this
section and the RuntimeCore Turn admission chain above. Responsible developer confirmation: root, 2026-07-28.

## 21. Official Responses Hosted Image Generation

Hosted image generation reuses the same provider-executed lifecycle as hosted web search and projects the Codex exact Item:

```text
official OpenAI/Codex Responses route + exact api.openai.com host
  -> canonical ImageGeneration tool definition
  -> model-provider { type: "image_generation" } lowering
  -> Responses image_generation_call item
  -> canonical provider-executed ToolCall / ToolResult
  -> Agent Runtime terminal raw-item history upsert
  -> App Server ImageGenerationItem
  -> Renderer image_generation read model
```

Only the canonical `ImageGeneration` definition is promoted. OpenAI-compatible gateways, Ollama, Chat Completions,
aliases such as `ImageGenerationTool`, and the local `lime_create_image_generation_task` tool remain ordinary function
tools. Provider-executed image calls never enter the local executor and do not change the final provider finish reason.
The reducer de-duplicates added/done/completed events by response item identity and rejects a completed item without a
string `result`. Agent Runtime replaces an earlier `in_progress` raw response item with the terminal item before durable
provider history is reused.

App Server projects provider metadata with `type=image_generation_call` to the exact Codex item shape: required `id`,
`status` and string `result`, plus optional `revisedPrompt` and `savedPath`. Renderer consumes that dedicated item and
fails closed on malformed required fields; it does not downgrade hosted image state into a generic extension. The hosted
request/reducer/lifecycle/history/protocol/read-model chain is `current`. Third-party promotion, alias promotion, local
media-task promotion, local re-execution and the previous loose `result?: Value/status?: String` DTO are
`dead / deleted / forbidden-to-restore`; no `compat/deprecated` path exists.

Architecture impact: major; this adds one provider-executed item variant while preserving `model-provider` as network
owner, Agent Runtime as lifecycle/history owner, and App Server as Thread/Turn/Item projection owner. Architecture diagram
updated: this section and the provider/runtime chain above. Responsible developer confirmation: root, 2026-07-27.

## 22. Vertex Gemini Current Transport

Vertex Gemini uses the existing provider store and model catalog control plane while keeping a dedicated wire identity:

```text
Provider(project, location, declared models) + enabled access token
  -> RuntimeCore VertexGemini route admission
  -> RuntimeCredentialData::VertexKey
  -> regional/global Vertex project endpoint
  -> model-provider Gemini canonical lowering
  -> Authorization Bearer request
  -> Gemini SSE reducer
  -> Agent Runtime Thread/Turn/Item lifecycle
```

`model-provider` is the only endpoint owner. A regional location resolves to
`https://{location}-aiplatform.googleapis.com`; `global` resolves to `https://aiplatform.googleapis.com`. The path is
typed from project/location and always ends in `publishers/google/models/{model}:streamGenerateContent?alt=sse`.
Provider-supplied origins may replace the host for controlled gateways and fixtures, but must be HTTP(S), have no path,
query or fragment, and still receive the typed Vertex project path. The access-token string uses Bearer auth and never
uses `x-goog-api-key`.

Vertex request lowering, SSE reduction, tool history and usage semantics reuse the current Gemini canonical algebra.
Provider identity, auth, endpoint construction, readiness and health scope remain distinct. The exact route health key
includes protocol and credential scope, so a plain Gemini API-key route and a Vertex route never share circuit state even
when provider, model and gateway origin are otherwise equal. Configured readiness requires enabled provider, non-empty
project/location and an enabled credential; only declared authoritative models enter the selectable catalog. Automatic
Vertex model discovery remains unsupported and cannot create route authority.

Vertex Gemini is `current`. Plain Gemini aliasing, OpenAI-compatible lowering, API-key header auth, missing context,
unresolved direct endpoints, WebSocket and hosted OpenAI capability promotion are
`dead / forbidden-to-restore`; no `compat/deprecated` path exists. Bedrock and Fal chat remain fail closed; Fal video is
admitted only by the dedicated media task chain below.

Architecture impact: major; this extends the current provider transport union without changing owner direction.
Architecture diagram updated: this section and the provider/runtime chain above. Responsible developer confirmation:
root, 2026-07-28.

## 23. Dedicated Video Media Task Execution Skeleton

Video generation is a dedicated media task, not an Agent chat input or a chat picker model. Its model control and product
semantics follow `grok-build`; provider body lowering remains in Lime's `model-provider` owner:

```text
provider model capability { taskFamilies: [video_generation], outputModalities: [video] }
  -> RuntimeCore ModelTaskRequest + route_capability_gap
  -> App Server ResolvedModelRoute + exact credentialRef/auth header/prefix
  -> mediaTaskArtifact/video/create public JSON-RPC
  -> App Server media video worker
  -> model-provider protocol lowering + network execution
       -> Fal synchronous POST
       -> xAI POST /videos/generations -> durable request_id -> GET /videos/{id}
  -> media-runtime durable provider progress + terminal task artifact
  -> mediaTaskArtifact/get read model
```

`video_generation` must be provider-explicit or canonical and must produce `video`; model-name inference cannot authorize
execution. A dedicated video model remains outside `model/list` because that method is the executable text chat picker.
The worker accepts only `fal` or `xai_video`, resolves the exact durable credential reference, honors the route's header
name and prefix, and records failed state when route or credential resolution is incomplete. Generic OpenAI video routes,
hard-coded Bearer auth, an executable contract without binding, and creating an artifact without starting a worker are
`dead / replaced / forbidden-to-restore`; no `compat/deprecated` path was added.

This slice is intentionally a product skeleton, not full Grok video parity. Fal handles a synchronous provider result;
xAI handles asynchronous start/poll and the `done/failed/expired/timeout/cancelled` terminal set. `media-runtime` persists
`provider_task.protocol/request_id/status`, and the App Server scheduler resumes only stale running xAI tasks that already
have a request id, so recovery polls without issuing a duplicate POST. `model-provider` is the only video HTTP and lowering
owner; `media-runtime` owns task progress and artifacts only. Live provider validation, video GUI/history projection and
audio workers remain `alignment-open` and must fail closed until implemented.

Architecture impact: major; this adds the first executable video media task branch while preserving chat admission and
Thread/Turn/Item ownership. Architecture diagram updated: this section and the media/provider chain above. Responsible
developer confirmation: root, 2026-07-28.

## 24. Thread-Scoped Media Read And Fail-Visible Preview

Media references remain canonical Thread/Turn/Item data while the App Server owns bounded sidecar reads:

```text
canonical Media Item + host-safe sidecar URI
  -> Renderer typed media/read { threadId, uri, offset, length, maxBytes }
  -> Electron preload/contextBridge -> app_server_handle_json_lines
  -> App Server v2 dispatcher -> thread-scoped SidecarStore reader
  -> bounded bytes + range/size/digest metadata
       | available and browser-decodable: bounded object URL -> image/audio/video preview
       | unavailable/denied/invalid/too large: metadata fallback surface
       | browser decode failure: unsupported fallback surface
```

`media/read` is the single `current` read method for GUI media sidecars. It is a Lime-owned host-safe extension because
Codex has no equivalent public method; its owner is the App Server read model plus SidecarStore, not Electron or the
Renderer. Requests and responses use canonical `threadId`; the Renderer cannot send `sessionId`, absolute source paths,
credentials or inline unbounded payloads. The reader enforces Thread scope, range and maximum-size constraints, validates
the sidecar digest, and returns only bounded base64 content. The Renderer owns the resulting object URL lifecycle and
releases replaced, unmounted and over-budget URLs.

A read failure never restores filesystem access or a production mock fallback. Missing or unreadable sidecars preserve a
visible metadata artifact without exposing the underlying error or absolute path. Browser image, audio or video decode
failure switches the existing preview to the shared `unsupported` surface instead of leaving an empty media element.
Permission denial, unsupported format, oversized results, invalid ranges and digest mismatch are covered at the nearest
Rust or Renderer boundary. The controlled Electron Gate B additionally proves a 471-byte PNG success path and the
sidecar-unavailable metadata fallback through real preload/IPC/App Server JSON-RPC/read-model/GUI boundaries; it does not
claim live-provider execution or every audio/platform failure mode.

The former v0 `agentSession/media/read` method, `AgentSessionMediaRead*` types, `MediaReadParams.stream`,
`media.read.chunk`, `media.read.completed`, matching raw-event subscriptions and Renderer live-drain helpers are
`dead / deleted / forbidden-to-restore`. Media progress is derived only after each bounded range response passes offset,
length, digest and size validation. This closes only the media transient bypass; the broader V2-05 notification surface
remains incomplete. No production `compat` path was added.

Architecture impact: major; this replaces the GUI media read protocol and makes media failure rendering explicit while
preserving App Server, SidecarStore and Thread/Turn/Item ownership. Architecture diagram updated: this section and the
desktop/App Server projection chain above. Responsible developer confirmation: root, 2026-07-29. V2-05 media transient
retirement confirmation: root, 2026-07-31.

## 25. Host Capabilities And Product-Scope Reverse Requests

The remaining V2-04 reverse requests are host-owned capabilities on the same canonical product chain:

```text
Electron Desktop Host
  -> App Server JSON-RPC / server-request dispatcher
  -> RuntimeCore session loop and exact response waiter
  -> Thread/Turn/Item canonical lifecycle and DynamicToolCall projection
  -> GUI PendingInteraction / typed timeline
```

`currentTime/read` is a read-only Electron host capability. The host is the only system-clock reader; App Server validates
the canonical thread scope and deadline, and RuntimeCore resumes the exact waiter. The request does not create a Thread
Item or expose a renderer clock API. Non-integer, out-of-range, duplicate, late, or mismatched responses fail closed.

`item/permissions/requestApproval` is owned by `tool-runtime` (permission profile parsing), `agent-runtime` (typed action
and waiter), App Server (JSON-RPC request/response validation), and the existing PendingInteractionController (GUI
decision surface). Session, thread, turn, cwd, environment, profile subset and response identity are checked at every
boundary. A grant is applied only after the exact pending request resolves; Renderer cannot synthesize a grant or bypass
the runtime policy.

`item/tool/call` is an Electron Desktop Host binding, not a renderer command. `thread/start` and `thread/resume` receive
the host-owned `desktop.appInfo` namespace; after the canonical Thread identity is observed, the binding is frozen. The
host accepts only the exact namespace/tool/schema/arguments/call identity and returns app name, version, locale and
platform. It never accepts paths, shell commands, URLs, arbitrary IPC or handler parameters.

DynamicTool definitions are trusted only from session metadata. Runtime snapshots flatten namespace routes (for example
`desktop + appInfo -> desktop__appInfo`) and reject deferred loading, invalid schemas, reserved names and collisions with
native/MCP/gateway tools. The executor registers the exact `DynamicTool` waiter before emitting
`dynamic_tool.requested`. Canonical `ThreadItemPayload::DynamicToolCall` stores callId, namespace, tool, raw JSON
arguments, ordered text/image/audio content, success and duration as typed fields. App Server projection, provider history,
read model and GUI consume those fields directly; metadata is limited to non-contract enrichment.

The three reverse requests, the host binding, the typed dynamic tool payload, and the permission/current-time waiters are
`current`; no compatibility owner was added. V2-05 notification, transient bypass and broader recovery surfaces remain
`deprecated / migration-open`. Legacy MCP Desktop commands, renderer-forged bindings, metadata core-field inference and
production mock fallback are `dead / deleted / forbidden-to-restore`.

Architecture impact: major; this closes the V2-04 host-capability boundary while preserving Electron as a thin Desktop
Host, App Server as JSON-RPC/projection owner, RuntimeCore as session/turn owner, and Thread/Turn/Item as the durable
fact source. Architecture diagram updated: this section and the reverse-request path above. Responsible developer
confirmation: root, 2026-07-30.

## 26. Config Warning V2 Notification Owner

Local configuration warnings use the existing App Server product chain and a single typed v2 notification owner:

```text
initialize / turn/start
  -> App Server config parser and RequestProcessor
  -> v2 ConfigWarningNotification { summary, details, path?, range? }
  -> app_server_handle_json_lines response notification batch
  -> typed Renderer response projection
  -> deduplicated five-locale global warning toast
```

`configWarning` keeps its existing wire method and producer timing. `details` is nullable; `path` and `range` are optional,
and `TextPosition` uses the Codex 1-based line/column contract. The Renderer only projects a decoded notification and does
not parse configuration files, invent warnings, or fall back to a mock backend. The notification is global UI state and
does not create or mutate a Thread/Turn/Item.

The v2 `config` module, v2 `ServerNotification`, central method catalog and v2 schema registry are the sole `current`
typed owner. The former v0 constant, DTOs, notification variant, catalog entry, schema files and positive tests are
`dead / deleted / forbidden-to-restore`; a v0 decoder regression test keeps that boundary fail closed. No `compat` owner
was added.

Architecture impact: major because the public notification/schema owner moved from v0 to v2, although the wire and
runtime behavior did not change. The canonical product direction remains Electron Desktop Host -> App Server JSON-RPC ->
RuntimeCore -> Thread/Turn/Item projection -> GUI. Architecture diagram updated: this section and the App Server protocol
boundary above. Responsible developer confirmation: root, 2026-07-31.

## 27. Runtime Warning V2 Notification And Recovery Owner

Thread-scoped runtime warnings use one typed live path and one durable event fact source:

```text
AgentEvent::Warning { code?, message }
  -> persisted runtime.warning with canonical thread identity
  -> V2NotificationProjector
  -> v2 warning { threadId, message, code? }
  -> Renderer typed projection { type: "warning" }
  -> existing localized deduplicated toast

persisted runtime.warning
  -> full canonical read / history-limit projection summary
  -> derived historical warning item with message + code?
  -> the same GUI warning presentation contract
```

The Codex base contract remains `threadId?: string | null` plus required `message`. Lime adds only optional `code`, omitted
when absent, because the current structured-input producer already emits stable warning codes and the five-locale GUI must
not expose a backend-language message in place of localized Skill/Mention warnings. `code` changes presentation only; it
does not select providers, mutate runtime state, or create another warning owner. Malformed message/code and missing
thread identity fail closed in the current Agent Chat path. The generic protocol can still decode a global
`threadId: null` warning, but no global GUI delivery is claimed until a real global producer and GN owner exist.

`runtime.warning` remains the ordinary durable warning fact. It is not forced into `ThreadItemPayload` and does not create
an Item lifecycle. Full cold reads and limited projection-summary reads derive the ordinary warning beside canonical items
from the same event log, preserving message, sequence and localization code without a second store. Live raw
`agentSession/event` warning wrappers are `dead / forbidden-to-restore` and covered only by negative tests.
`guardian.warning` is a separate durable fact produced only by the Guardian denial circuit breaker; it projects to the
strict `guardianWarning` v2 notification and Desktop `NoticeProjection`, and must never be represented as ordinary `warning`.

Architecture impact: major because a live notification and recovery surface moved from the deprecated raw side channel
to the v2 protocol/projector/Renderer boundary. The product direction remains Electron Desktop Host -> App Server
JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI. Architecture diagram updated: this section and the App
Server notification boundary above. Responsible developer confirmation: root, 2026-07-31.

## 28. Skills Changed Catalog Invalidation Owner

Skill catalog discovery and invalidation use one current owner and one typed transient notification path:

```text
default Skill roots create / modify / remove
  or successful Lime Skill catalog mutation
  -> invalidate lime-skills snapshot + summary caches
  -> App Server v2 skills/changed {}
  -> app_server_handle_json_lines notification drain
  -> Renderer typed notification bus
  -> current skills/list { cwds, forceReload }
  -> RuntimeCore -> workspace-scoped AgentSkillSnapshot
  -> data[{ cwd, skills, errors }]
  -> Composer Skill catalog projection and GUI refresh

skills/extraRoots/set { extraRoots }
  -> replace process-scoped extra roots
  -> invalidate the same caches
  -> skills/changed {}

skills/config/write { exactly one of path/name, enabled }
  -> Lime YAML skills.config
  -> invalidate the same caches
  -> skills/list and Agent turn snapshots apply the same enablement policy
```

The App Server watches only default Skill roots. Existing roots are recursive watches; roots created after startup are
reconciled and watched. Filesystem create, modify and remove events invalidate both discovery caches and broadcast at
most once per ten-second throttle window. Successful Lime catalog mutations invalidate the same caches and attach the
same typed notification at the processor boundary; failed mutations and mutations for other apps do not notify.

`skills/list` is the sole executable catalog read for Composer. It matches the Codex v2 request and response shape:
empty `cwds` resolves to the App Server process cwd, explicit cwd order is preserved, `forceReload=true` clears both
snapshot and summary caches, and discovery errors remain attached to their cwd without discarding valid skills. Scope
lowering is `project -> repo`, `user -> user`, `app -> system`, `other -> admin`; `path` is the absolute `SKILL.md`
locator. The Renderer applies the Desktop-only projection, derives the stable detail-read id, filters `enabled=false`
entries and never adds a TUI surface. `skill/read` remains the independent body/workflow detail read.

`skillManagement/list` retains its management-center semantics and does not become a second Composer catalog owner.
Reconnect or remount performs a fresh list independently of notification delivery. `skills/changed` carries the strict
empty object only and is a process-level transient invalidation; it does not create a Thread/Turn/Item, durable event,
persistence record or replay requirement.

The v2 protocol/schema/generated client, App Server watcher and successful mutation producer, `lime-skills` cache
invalidation, Renderer typed event bus and current `skills/list` refresh are `current`. Singular `skill/list` and the
zero-consumer `get_local_skills_for_app` Desktop facade are `dead / deleted / forbidden-to-restore`; no `compat` path
exists. `skills/config/write` is the current user-level enablement writer: Lime Desktop persists it in the existing YAML
configuration owner rather than copying Codex's TUI-oriented TOML path. `skills/extraRoots/set` is current process state,
replaces the full root set, accepts absent directories as empty discovery inputs and never persists them. A second
catalog/read model, durable notification replay and production mock fallback are `dead / forbidden-to-restore` as
alternate implementations.

Architecture impact: major because the catalog read moved to Codex v2 while keeping the product direction unchanged:
Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore/domain owner -> GUI. Architecture diagram updated: this
section and the Skill catalog path above. Responsible developer confirmation: root, 2026-08-07. Directory ownership,
data flow, dependency direction, protocol boundary and required public JSON-RPC / Electron evidence gates are confirmed.

## 29. Durable Ordered Thread Section Owner

Thread organization uses the same durable App Server read model as Codex, with Desktop-specific grouping:

```text
threadSection/list
  -> ordered section catalog
  -> thread/list { sectionId, sortKey: "section_position" }
  -> App Server canonical Thread projection
  -> Renderer session gateway
  -> Pinned/custom section shelf, then unsectioned project/conversation groups

thread/section/move { threadId, sectionId: string | null }
  -> ThreadStore section membership and position
  -> canonical Thread section + sectionEnteredAt
  -> Desktop sidebar refresh
```

`threadSection/list/create/update/delete` and `thread/section/move` are the sole current protocol and storage
boundary for section catalog, membership and ordering. The built-in Pinned section has one stable UUID, cannot be
renamed or deleted, and uses `sectionId: null` to remove a thread from the section. Renderer session projection keeps
the returned section order and never re-sorts by `updatedAt`; sectioned sessions are excluded from project and
standalone groups so one thread has one visible navigation owner.

The v2 protocol/schema/generated client, App Server section store and JSON-RPC tests, Renderer typed gateway/session
projection, and Desktop Pinned/custom shelf are `current`. The former Thread `isPinned` metadata, localStorage
favorite-session list, favorite boolean projection and second timestamp sort are `dead / deleted / forbidden-to-restore`;
no compatibility owner was added. Custom section CRUD remains available through the current typed gateway; a separate
TUI-style section manager is not introduced into the Desktop shell.

Architecture impact: major because durable thread membership and navigation now share one ordered read model across the
App Server and Desktop GUI. The product direction remains Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore
-> Thread/Turn/Item projection -> GUI. Architecture diagram updated: this section and the Thread/Turn/Item projection
path above. Responsible developer confirmation: root, 2026-08-06.

## 30. Codex Hook Lifecycle Owner

Hook discovery and execution now use one current owner across the runtime and Desktop projection chain:

```text
CODEX_HOME/config.toml + <cwd>/.codex/config.toml + active Plugin catalog
  -> tool-runtime Hook discovery/trust snapshot
  -> Agent sampling step / command gate
  -> AgentEvent hook.started / hook.completed
  -> App Server v2 hook notifications
  -> Renderer transient Hook timeline projection

canonical Thread/Turn/Item materializer
  -> excludes hook.started / hook.completed
  -> thread/read + thread/items/list contain no Hook lifecycle Item
```

`hooks/list` is the sole public discovery contract. `tool-runtime` owns source loading, stable key/hash, trust and
execution; Agent runtime owns lifecycle events; App Server owns JSON-RPC dispatch, notification projection, durable
event append and public read-model exclusion; the Renderer validates the typed notification and may display a transient
timeline row. Started and completed notifications must carry the same run id, but they do not create or update a
canonical Item. This follows Codex's separate `hook/started` and `hook/completed` notifications and its ThreadItem union,
which contains `HookPrompt` but no Hook lifecycle Item.

The v2 protocol/schema/generated Rust and TypeScript clients, public JSON-RPC `hooks/list`, Hook event projector,
canonical history exclusion and five-locale transient timeline row are `current`. The former canonical
`ThreadItemPayload::Hook`, `item_<hookRunId>` identity, public history recovery path, raw hook config shape,
`known_unprojected` Hook drift path and production mock/fallback execution are `dead / deleted / forbidden-to-restore`;
no compatibility owner was added. Lime remains a compact Electron Desktop GUI and does not copy Codex TUI surfaces.
Provider/model/media behavior remains owned by the Grok-aligned `model-provider` control plane.

Architecture impact: major because the Hook lifecycle boundary is a typed live notification rather than a durable
Thread Item and recovery contract. Architecture diagram updated: this section and the App Server notification versus
Thread/Turn/Item projection split above. Responsible developer confirmation: root, 2026-08-09. Confirmation content:
已核对 Codex notification/ThreadItem 边界、Lime EventStore 分类、公共历史排除、Renderer transient 投影和验证门禁。

## 31. Apps Catalog And Readiness Owner

Apps/connectors 只有一个 catalog owner，并保持 Desktop 与 Codex TUI 的产品边界分离：

```text
standard Agent Plugin root manifest
  -> explicit Codex extension `apps: "./config.json"`
  -> independent `apps.{name}.{id,category?}` config
  -> App Server PluginDataSource / local plugin_catalog
  -> RuntimeCore app/list | app/read | app/installed
  -> App Server JSON-RPC
  -> Renderer typed Apps gateway
  -> Desktop Apps catalog/readiness projection

successful plugin/install | plugin/uninstall | plugin/enabled/set
  -> same Plugin catalog mutation owner
  -> app/list/updated { data: AppInfo[] }
  -> App Server notification event bus
  -> Renderer typed watcher -> fresh Apps read
```

`app/list` never reads a portable top-level `apps` field. It follows Codex's explicit client extension adapter:
`extensions.com.openai.apps`, with `.codex-plugin/plugin.json` only as the overlay fallback, must be a package-relative
path to a separate Apps JSON document. Connector `id` is the catalog identity; inline Apps objects fail closed and an
invalid Apps component is isolated from Skills/MCP and package installation. `app/read` deduplicates ids while preserving
first-request order and returns `missingAppIds`; the processor rejects more than 100 ids with `INVALID_PARAMS`. Optional
`threadId` on list/installed is validated against the loaded canonical Thread and fails closed with `SESSION_NOT_FOUND`.
The local registry is read fresh on every request, so `forceRefetch` and `forceRefresh` never fabricate a hosted cache
refresh.

`callable` is a readiness boundary, not an install flag. Until a local Plugin app has a committed hosted connector
model-visible tool snapshot, enabled local apps report `callable=false` and Desktop readiness remains false. No UI or
provider route may infer model-callability from `isEnabled`, manifest declaration or Plugin installation alone.

The v2 Apps DTOs, method/notification catalogs, generated schema/client, App Server Plugin catalog projection, public
JSON-RPC tests, Renderer typed gateway, typed notification parser and App Center readiness consumer are `current`.
`src/components/AppPageContent.tsx -> src/features/plugin/ui/PluginCatalogPage.tsx` remains the only App Center route
owner; Apps are projected inside the selected Plugin detail sidebar instead of creating a second Apps page or state
source. The consumer reads `app/list + app/installed`, renders `ready / disabled / pending`, and reruns the same fresh
read after typed `app/list/updated` arrives through the App Server event bus.

The Apps-specific Electron Gate B runner now uses isolated app data, a standard root Agent Plugins manifest and a Codex
extension path to an independent Apps JSON. Its prior 2026-08-07 artifact used the retired inline manifest shape and is
historical only. The migrated runner passed on 2026-08-09 and proved the transport/UI flow across real Electron
renderer/preload/IPC, `app_server_handle_json_lines`, `plugin/list -> plugin/install`, exact
`app/list` / `app/read` / `app/installed`, GUI `plugin/enabled/set`, the subsequent fresh Apps read and the same visible
row changing from `enabled=true / callable=false / pending` to `disabled`. Console, page, invoke, trace, legacy command
and production mock fallback counts were all zero. Evidence:
`.lime/qc/project-gates/standalone-apps-catalog-20260809T054741397Z-147740/apps-catalog-gate-b/apps-catalog-gate-b-summary.json`.
All seven required methods were observed; install notification and pending-to-disabled fresh read succeeded; console,
page, invoke, trace, mock fallback and legacy command counts were zero.

There is no second Apps catalog, `window` custom-event fact source, TUI-style Apps surface, compatibility wrapper or
production mock fallback. Hosted connector model-visible tool snapshot and a real `callable=true` provider path remain
open capability work; the local Gate B does not claim either one.

Architecture impact: existing cross-layer Apps owner corrected in place; no second runtime or storage owner added.
Architecture diagram updated: this section now distinguishes the portable Agent Plugins manifest from the explicit
Codex Apps extension path and independent config file.
Responsible developer confirmation: root, 2026-08-09. Confirmation content: 已核对 portable/extension
边界、Apps identity、失败隔离、JSON-RPC 数据流与 Gate B 重验要求。

Architecture impact: major because this adds a cross-layer catalog/readiness contract and live invalidation path while
reusing the existing Plugin catalog owner. The product direction remains Electron Desktop Host -> App Server JSON-RPC
-> RuntimeCore -> Thread/Turn/Item projection -> GUI; provider/model/media behavior remains owned by the Grok-aligned
`model-provider` control plane. Responsible developer confirmation: root, 2026-08-07. Confirmation content: 已核对唯一
Plugin catalog owner、`callable=false` fail-closed 边界、notification 数据流和 Desktop/TUI 分界。

## 32. Exact Memory Reset Owner

全局记忆重置复用现有 MemoryStore 领域 owner，不保留第二套 scoped reset wire：

```text
Desktop Settings
  -> Renderer resetMemory()
  -> App Server JSON-RPC memory/reset
  -> RuntimeCore::reset_memory
  -> MemoryAppDataSource::reset_memory
  -> LocalMemoryBackend::reset
  -> clear global memory root contents
  -> recreate managed memory layout
  -> {}
```

`memory/reset` 对齐 Codex 的无参数全局动作：omitted、`null` 和空对象 params 可接受，其他字段 fail closed；响应恒为
空对象。`LocalMemoryBackend` 仍是唯一文件删除 owner，只清理 global memory root 并立即重建摘要、memory、notes、
skills 与 index 等受管布局。ThreadStore、Thread/Turn/Item projection、event log、session history 和 memory root 外的
soul 配置不在该删除边界内。

v2 protocol/schema、App Server handler、RuntimeCore/AppDataSource 委托、Rust/TypeScript typed client、Renderer
Settings 消费与 public JSON-RPC durable isolation test 为 `current`。旧 `memoryStore/reset`、scoped params、富计数
response、typed clients 和设置页调用为 `dead / deleted / forbidden-to-restore`；没有 compat alias，也不保留未被产品
消费的 workspace reset。

Architecture impact: major because the public reset contract moved from a custom v0 method to the exact Codex v2
boundary while retaining the existing storage owner. The product direction remains Electron Desktop Host -> App Server
JSON-RPC -> RuntimeCore/domain owner -> GUI. Responsible developer confirmation: root, 2026-08-07. Confirmation content:
已核对目录归属、删除边界、依赖方向、协议形态和 durable history 隔离门禁。

## 33. Exact Process Control Owner

Codex exact process lifecycle 复用 `tool-runtime` 的 local process supervisor，但 public handle ownership 只属于
App Server transport connection：

```text
typed App Server client
  -> process/spawn
  -> App Server ProcessServer keyed by (ConnectionId, processHandle)
  -> tool-runtime LocalExecutionProcessHandle
  -> spawn response
  -> process/outputDelta (zero or more)
  -> process/exited (exactly one terminal notification)

process/{writeStdin,resizePty,kill}
  -> same (ConnectionId, processHandle)
  -> same supervisor control handle
```

同一 `processHandle` 可以由不同连接独立使用，同一连接内重复 active handle 必须失败。spawn 在 response 成功发送后才
activate notification pump，保证 response-before-notification；supervisor 通过单一 ordered event stream 保证全部 output
先于 exited。连接关闭、response 发送失败或 notification writer 失败都清除 owner 并终止进程。stdout/stderr raw bytes
以 base64 notification 投影；非流式终态保留 UTF-8 lossy 聚合文本，默认 output cap 为 1 MiB。`outputBytesCap` 与
`timeoutMs` 保持 omitted/null/value 三态；stdin close 后的非空写入、非 TTY resize、零值 terminal size、未知 handle
均 fail closed。

Desktop 不复制 Codex TUI process UI。Workspace command Item 属于 Thread/Turn/Item projection，其后台终端控制继续走
`thread/backgroundTerminals/list -> itemId 匹配 -> thread/backgroundTerminals/terminate`，不能把 command item 的
`processId` 当成任意 transport connection 的 `processHandle`。因此旧 GUI status refresh、drain output、signal-only
interrupt 和 stdin 控件均删除；Renderer 只保留 Codex 有明确 public Thread owner 的终止动作。

v2 protocol/schema、App Server dispatcher/connection cleanup、`ProcessServer`、local supervisor、Rust/TypeScript typed
clients 和 notification tests 为 `current`。内部 `tool-runtime::execution_process::live` 请求/查询类型和
`ExecutionProcessServer` 继续服务 unified exec、Thread shell 与 background terminal，不进入 public protocol。
旧 `executionProcess/*` JSON-RPC、v0 DTO/schema、typed helpers、Renderer gateway、status/drain/interrupt/stdin UI 为
`dead / deleted / forbidden-to-restore`；`compat` 与 `deprecated` 均为空。

Architecture impact: major because process ownership, notification ordering and Desktop projection boundaries now have one
explicit cross-layer contract. The product direction remains Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore/domain
owner -> Thread/Turn/Item projection -> GUI; process control does not create a second Electron backend. Responsible developer
confirmation: root, 2026-08-08. Confirmation content: 已核对 connection/Thread owner 分界、目录归属、数据流、依赖方向、
协议与 notification 顺序、删除边界和验证门禁。

### 33.1 Exact Command Exec Owner

Codex standalone `command/exec` 是 Desktop coding terminal 的唯一 current owner；它与 Thread-owned command Item、
Codex TUI 的 `process/*` 控制面分开，但共享 `tool-runtime` 本地进程 supervisor：

```text
Renderer commandExec gateway
  -> typed App Server client
  -> command/exec
  -> App Server CommandExecServer keyed by (ConnectionId, processId)
  -> tool-runtime LocalExecutionProcessHandle

command/exec/write|resize|terminate
  -> same (ConnectionId, processId)
command/exec/outputDelta
  -> same owner connection, raw bytes as deltaBase64
```

一次性命令 response 保留 UTF-8 聚合 stdout/stderr；开启流式输出时，所有 delta 在最终 response 前按顺序投影，
response 的 stdout/stderr 为空。`command/exec` 的 `outputBytesCap` 和 `timeoutMs` 使用 Codex exact 三态字段，默认
output cap 为 1 MiB，超时退出码为 `124`；PTY resize 只允许正数尺寸，stdin close 后非空写入失败。`processId` 可以由
客户端提供，也可由服务端生成，但只在 originating ConnectionId 内有效；断连和终止都会清理 session。

Desktop terminal 只消费 `src/lib/api/commandExec.ts`，通过 xterm 展示真实 outputDelta，不在 Renderer 伪造 prompt、
session reconnect、明文 stdin 或 fallback output。Electron 仅承担既有 JSONL sidecar 转发职责。

v2 protocol/schema、CommandExecServer、connection cleanup、typed client、Renderer gateway、GUI terminal 和负向回流
guard 为 `current`；旧 `project_shell_*`、`run_project_shell_command`、Project Shell v0 DTO/schema、旧 gateway 与
Electron host 为 `dead / deleted / forbidden-to-restore`；`compat` 与 `deprecated` 均为空。

Architecture impact: major because a public JSON-RPC command family replaced the private Project Shell IPC/session owner.
Architecture map updated: sections 6.1, 33.1 and command boundary document. Responsible developer confirmation: root,
2026-08-08. Confirmation content: 已核对 Desktop/App Server/runtime owner、ConnectionId 隔离、notification 顺序、
raw bytes lowering、删除边界和 GUI/contract 验证门禁。

## 34. Exact Filesystem Owner

Codex exact filesystem contract 由 App Server 独立 `FsServer` 承接，Desktop 只消费 typed wire 并投影富 GUI：

```text
Renderer fileBrowser/session-files gateway
  -> typed App Server client
  -> fs/readFile | fs/writeFile | fs/createDirectory
  -> fs/getMetadata | fs/readDirectory | fs/remove | fs/copy
  -> App Server FsServer
  -> tokio filesystem

fs/watch { watchId, absolute path }
  -> FsServer keyed by (ConnectionId, watchId)
  -> filesystem watcher
  -> fs/changed { watchId, changedPaths[] }
  -> owning connection only
```

协议路径必须为绝对路径，raw bytes 始终以 base64 传输。`readFile` 有 512 MiB 上限并在越界时 fail closed；
`writeFile` 不创建父目录；`createDirectory.recursive`、`remove.recursive/force` 与 `copy.recursive` 保持 Codex wire
语义。symlink metadata 显式投影，directory listing 只返回 exact entry shape；Desktop 再读取 metadata，生成文件类型、
隐藏状态、预览文本和其他 GUI 字段，不把这些投影写回协议。

watch owner 固定为 `(ConnectionId, watchId)`。同一连接重复 active watch id 失败，不同连接可独立复用同一 id；
notification 只投递给 owner，断连仅清理该连接 watcher。rename 不扩充 public protocol，由 Desktop 组合
`getMetadata -> copy -> remove`，明确为非原子操作。Electron 仍只拥有目录选择、系统快捷位置和图标读取等宿主能力，
不成为第二套文件业务后端。

v2 protocol/schema、App Server dispatcher/connection cleanup、`FsServer`、Rust/TypeScript typed clients、公共 JSON-RPC
和 Renderer file gateway 为 `current`。旧 `fileSystem/*`、v0 DTO/schema、App Server RuntimeCore file projection、
`processor/file.rs`、services `file_browser_service` 与 renderer aliases 为 `dead / deleted / forbidden-to-restore`；
`compat` 与 `deprecated` 均为空。旧 preview 曾承担的 Office/PDF 文本提取不属于 exact raw-byte fs contract；若产品继续
需要，必须在独立 current 文档能力 owner 中重建，不能恢复 `fileSystem/readFilePreview`。

Architecture impact: major because the public file contract, watch ownership and Desktop projection boundary moved to one
exact v2 owner. The product direction remains Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore/domain owner ->
Thread/Turn/Item projection -> GUI; filesystem IO does not create a second Electron backend or copy Codex TUI. Responsible
developer confirmation: root, 2026-08-08. Confirmation content: 已核对目录归属、绝对路径/base64 边界、connection watcher
owner、Desktop GUI 投影、旧 owner 删除与验证门禁。

## 35. Review Lifecycle Owner

Desktop review 复用 RuntimeCore 的异步 turn admission 和 Thread/Turn/Item canonical lifecycle；它不复制 Codex TUI
的 review UI 或后台入口：

```text
Desktop review gateway
  -> App Server JSON-RPC review/start
  -> RuntimeCore::start_review
  -> admitted Turn (inProgress)
  -> enteredReviewMode Extension Item
  -> provider/backend review turn
  -> exitedReviewMode Extension Item
  -> turn.completed
  -> v2 ThreadItem projection / GUI timeline
```

`review/start` 是 inline Desktop action。`threadId` 必须命中已加载 session，active turn、空 branch/sha/instructions
和 `delivery=detached` 均 fail closed；detached review 明确不属于 Lime Desktop。base branch、commit sha/title 和 custom
instructions 在 prompt 构造前统一 trim/校验，canonical boundary 与恢复数据只保存规范化后的 target，避免 prompt 与 read
model 分叉。

review admission 立即返回 v2 `turn.status=inProgress`，实际 backend 在 session loop 中异步执行。durable 事件至少保持
`item.started(enteredReviewMode) -> turn.accepted -> item.completed(exitedReviewMode) -> turn.completed` 的 review-specific
顺序；review 输出优先从 assistant message/item 事件聚合，没有输出时使用稳定的 user-facing hint。两个 Extension Item
分别投影为 v2 `ThreadItem::EnteredReviewMode` 与 `ThreadItem::ExitedReviewMode`，未知 Extension 继续 fail closed。

v2 protocol/schema、App Server review handler、RuntimeCore review context、canonical/read-model projection、Rust/TypeScript
clients 与 review 定向测试为 `current`。Desktop review gateway 可消费该 current method；Electron 只转发 JSONL，不承接
review runtime。Codex TUI 的 detached/background review、旧 raw side-channel 和未被 Desktop 消费的兼容入口为
`dead / deleted / forbidden-to-restore`；`compat` 与 `deprecated` 均为空。

Architecture impact: major because review now has one cross-layer asynchronous admission, durable boundary Item and v2
projection contract. The product direction remains Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore ->
Thread/Turn/Item projection -> GUI; model/provider behavior remains owned by the Grok-aligned `model-provider` control plane.
Responsible developer confirmation: root, 2026-08-09. Confirmation content: 已核对 review owner、target 规范化、事件顺序、
Desktop/TUI 边界、删除分类和 Rust/contract 验证门禁。真实 Electron Gate B review evidence 已建立：
`.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/code-artifact-workbench-electron-fixture-summary.json`。
证据确认真实 Electron preload/IPC 命中 `app_server_handle_json_lines` 与 typed `review/start`，backend `turnId` 与
raw v2 `enteredReviewMode`/`exitedReviewMode` 及 canonical `thread/read` 同一身份；GUI 显示“代码审查完成：未发现阻塞性问题”
和“审查已完成”，内部 Review prompt 不进入页面文本，生产 mock fallback 命中为零。共享壳证据为
`.lime/qc/project-gates/standalone-shell-01-20260808231556-70202/shell-01-electron-smoke/summary.json`。

## 36. Existing Current Method Classification Audit

本轮不新增平行实现，只修正产品范围矩阵中把 current owner 混入 planned 组的分类漂移。以下 13 个 exact method 已有
同方向 generated manifest、真实 owner、typed client/projection 和可追踪证据：

```text
PluginCatalogPage
  -> typed pluginCatalog gateway
  -> plugin/list | plugin/read | plugin/install | plugin/uninstall | plugin/installed
  -> App Server Plugin processor / RuntimeCore PluginDataSource
  -> local plugin_catalog

RuntimeCore waiter
  -> currentTime/read | item/permissions/requestApproval | item/tool/call
  -> Electron Desktop Host / unified PendingInteraction exact responder
  -> validated response identity
  -> canonical continuation / permission grant / DynamicToolCall Item

runtime.warning | runtime.error | command terminal interaction fact
  -> App Server warning | error | item/commandExecution/terminalInteraction
  -> typed client / canonical read model / Renderer projection

update_plan completion
  -> durable turn.plan.updated fact
  -> App Server turn/plan/updated v2 notification
  -> typed client / Renderer projection

apply_patch exact Turn delta
  -> durable turn.diff.updated fact
  -> App Server turn/diff/updated v2 notification
  -> typed client / canonical conversation Turn unified_diff
  -> Desktop Changes previous-conversation projection
```

基础 Plugin catalog 五个方法已经由 Plugin v3 current owner 承接，并在真实 Electron fixture 中经过
`app_server_handle_json_lines`；这不等于 Plugin share、`plugin/skill/read`、remote catalog watcher 或 hosted connector
readiness 已完成。`currentTime/read` 仍只读取 Host 时钟；`item/permissions/requestApproval` 仍经 tool-runtime
permission parser、App Server exact waiter 和统一 PendingInteractionController 返回 scope-bound grant；`item/tool/call`
仍只响应冻结的 Desktop dynamic-tool binding，不开放任意 Electron IPC。typed `warning` / `error` 继续由 durable
runtime fact 和 canonical read model 承接，terminal interaction 只保留 bounded redacted summary。
`turn/plan/updated` 只投影 RuntimeCore producer 的 typed plan fact，不把 Renderer 本地 checklist 或 Tool Item 变成第二事实源。
`turn/diff/updated` 只投影 `apply_patch` 在当前 Turn 内聚合出的精确 unified diff；连续 patch 由 RuntimeCore coding event tracker
校验并合并，未知或不连续 mutation 发送空 diff 清理旧快照。App Server projector 与 typed client 严格拒绝额外字段，Renderer
只把它归并到 canonical conversation Turn 的 `unified_diff`，Desktop Changes 在 previous-conversation 模式直接读取该字段。
空字符串是有效 net-zero 结果，不得回退到由 GUI items 拼装的第二份 patch。该链路不复制 Codex TUI，也不改变 provider owner；
多模型、多模态 sampling 和媒体 lowering 继续归 Grok-aligned `model-provider`。

矩阵中的 `plugin/share/*`、Codex `marketplace/*`、external-agent migration 与远端
`environment/*`/`thread/environment/*` 均按 Desktop 产品范围裁决为
`product-scope-excluded / forbidden-to-restore`；`plugin/skill/read` 的 Codex remote-plugin-only
wire 同样 excluded。Lime current 只保留本地 Plugin v3 catalog、Skills catalog/`skill/read` 与
独立 `skillMarketplace/install` owner；
`item/autoApprovalReview/*` 已由 Guardian current owner 承接；`turn/moderationMetadata` 由下一节 current
主链接管。没有新增 `compat` 或
`deprecated`；旧 Plugin 私有协议、Renderer 伪造 reverse request、raw diagnostic side-channel、未脱敏 terminal
interaction 和生产 mock fallback 继续为 `dead / deleted / forbidden-to-restore`。

`deprecationNotice` 已按 Desktop 产品范围裁决为 `product-scope-excluded`：它是 Codex 开发/设置诊断，不进入
对话通知链；旧实现无外部兼容负担时直接替换或删除，不恢复同名通知包装。

Codex `server/diagnostics` 同样保持 `product-scope-excluded / forbidden-to-restore`。它只返回进程本地、无内容的
上游诊断指标，Lime Desktop 没有对应用户流程；Electron 不新增诊断业务后端，也不把该接口的指标当作模型
capability、provider readiness 或 sandbox enforcement 证据。

Architecture impact: major; 本节新增了从 Turn-scoped 精确 delta producer、durable event、v2 notification、canonical Turn
到 Desktop Changes 的跨层数据流，并明确空 diff 清理与唯一事实源边界。Responsible developer confirmation: root,
2026-08-09. Confirmation content: 已核对 `apply_patch` 连续 mutation 校验、EventLog/projector 顺序、typed client 严格
解码、canonical Turn 恢复、Desktop/TUI 分界，以及多模型/多模态仍由 Grok-aligned `model-provider` 承接。

## 37. Turn Moderation Metadata Projection

Lime 只从 trusted first-party Responses transport 读取
`response.metadata.openai_chatgpt_moderation_metadata`，不接受第三方兼容端点伪造该字段。SSE 与 WebSocket 共用同一
Responses reducer，产生 provider-neutral `CanonicalLlmEvent::TurnModerationMetadata`；Agent runtime 不对该事件去重，
每次 sampling 更新都生成 durable `turn.moderation_metadata` fact。App Server 按事件顺序投影 exact
`turn/moderationMetadata { threadId, turnId, metadata }`，缺少 thread、turn 或 metadata 时 fail closed；显式 `null`
仍是有效 metadata。

```text
trusted first-party Responses response.metadata
  -> model-provider CanonicalLlmEvent::TurnModerationMetadata
  -> agent-runtime CurrentProviderTurnEvent / AgentEvent
  -> durable turn.moderation_metadata
  -> App Server turn/moderationMetadata
  -> typed client signal router
  -> Renderer canonical Turn.moderation_metadata
```

`metadata` 是 opaque JSON value，可以是 object、array、scalar 或 `null`。各层不得猜测供应商私有字段、生成第二份
typed schema 或直接展示 raw JSON；Renderer reducer 仅做 last-write-wins，后续不含该字段的 Turn snapshot 必须保留已有
值，cold/hydrate reader 也读取同一 canonical Turn 字段。Codex TUI 当前忽略该通知，Lime Desktop 不复制 TUI UI；
Electron 继续只转发 App Server JSONL，不新增 IPC 或第二业务后端。OpenAI moderation metadata 的可信 transport lowering
归 `model-provider`，但多模型 catalog、默认选择、model switch、provider capability/readiness、retry/circuit breaker 与
多模态 sampling/media lowering 仍由 Grok-aligned control plane 承接。

Architecture impact: major; 本节新增 first-party provider metadata 到 durable Turn projection 的跨层数据流，并固定
opaque JSON、无去重、last-write-wins 与 Desktop/TUI 分界。Responsible developer confirmation: root, 2026-08-09.
Confirmation content: 已核对 SSE/WS 共用 reducer、first-party trust gate、App Server exact wire、canonical Turn 恢复、
Electron 无新增业务边界，以及 Grok-aligned 多模型/多模态 owner 不变。

## 38. Guardian Auto Approval Review Projection

严格自动审查只在当前工具决策已经判定为 `strictAutoReview` 的 shell/`exec_command` 路径触发真实 Guardian
reviewer；它不是用户审批的重命名，也不复制 Codex TUI 的 detached review UI。reviewer 复用当前 session 的
`model-provider` 路由，以无工具结构化采样读取一次风险判断；provider 未就绪、取消、超时、非法 JSON 或不确定结果
均 fail closed 为拒绝。

```text
strictAutoReview tool decision
  -> agent-runtime Guardian reviewer (same session model-provider, no tools)
  -> AgentEvent guardian_review_started/completed
  -> App Server durable event projector
  -> item/autoApprovalReview/started|completed
  -> typed app-server client / Renderer sequence gate
  -> ConversationProjection pending_interactions
```

started 以 `reviewId` 建立 `kind: guardian_review` 的 pending interaction，并携带目标 Item、action 和风险审查快照；
completed 只接受 `agent` decision source 与 `approved|denied|timedOut|aborted` 终态，分别投影为
`resolved|declined|cancelled`。缺失 start、错 thread/turn、额外字段或非终态 completion 均 fail closed；Reducer 不创建
第二份审批 Item、Message synthesis 或独立 pending store。审查 rationale 是内部 bounded payload，GUI 不展示 provider
原始 JSON 或 prompt。

Electron 继续只转发 App Server JSONL，不新增 IPC 或第二业务后端。Codex TUI 的 detached/background review 不属于
Lime Desktop 产品面；`guardianWarning` 由独立 Guardian denial circuit breaker producer 接入 current 主链。Guardian 风险 lowering 归
`model-provider`，而 Grok-aligned 多模型 catalog/default/model switch/provider capability/readiness/retry/circuit
breaker 与多模态 sampling/media lowering owner 不变。

Architecture impact: major；本节新增 Guardian review 从工具决策、provider sampling、durable AgentEvent、v2 notification
到 GUI pending projection 的跨层数据流，并固定 fail-closed 与 Desktop/TUI 边界。Responsible developer confirmation:
root, 2026-08-09。Confirmation content: 已核对 strictAutoReview producer 范围、session provider 复用、超时/取消/非法响应
拒绝语义、App Server typed wire、Renderer pending 状态、Electron 无新增 IPC，以及 Grok-aligned 多模型/多模态 owner
不变。

## 39. Guardian Warning Circuit Breaker Projection

Guardian review denial 的高优先级告警现在由唯一的 `agent-runtime` current owner 产生。对同一 session/turn，状态机维护
连续拒绝计数和最近 5 次 review 窗口；连续 3 次拒绝时只发出一次 `AgentEvent::GuardianWarning`，并取消当前 turn。
provider 不可用导致的 Guardian denial 也进入同一计数器；approved 会清理该 turn 的计数，关闭 provider session 会清理
session 状态，不建立第二份持久化 store。

```text
strictAutoReview denial
  -> AgentRuntimeState guardian denial circuit breaker (3 consecutive / 5-review window)
  -> AgentEvent guardian_warning
  -> durable guardian.warning
  -> App Server v2 guardianWarning { threadId, message }
  -> typed client signal
  -> Renderer ConversationProjection NoticeProjection(code = guardian_warning)
```

`guardianWarning` 必须保持独立于普通 `warning`、Guardian review completed 和用户审批；非法 thread/message、额外协议字段
和未知 producer 均 fail closed。Lime Desktop 只呈现高优先级 notice，不复制 Codex TUI detached review UI；Electron 仍只
转发 App Server JSONL，不新增 IPC 或第二业务后端。无 `compat`/`deprecated`；旧 raw side-channel、普通 warning 冒充和
生产 mock fallback 为 `dead / deleted / forbidden-to-restore`。多模型、多模态控制面仍由 Grok-aligned `model-provider`
承接，Guardian lowering 只复用当前 session provider。

Architecture impact: major；本节新增 denial circuit breaker 到 Desktop notice 的跨层数据流，并固定一次性告警、turn
中断、fail-closed 和 Desktop/TUI 边界。Responsible developer confirmation: root, 2026-08-09。

## 40. Experimental Feature Configuration Owner

Lime Desktop 实验特性由 App Server catalog 与 `lime_core config.yaml` 共同构成唯一 current owner。Settings 不再通过
Electron 业务命令直接读写配置；Electron 只保留通用 App Server JSONL 转发职责。

```text
Settings Experimental
  -> Renderer typed experimentalFeatures gateway
  -> app_server_handle_json_lines
  -> experimentalFeature/list | experimentalFeature/enablement/set
  -> App Server feature catalog
  -> lime_core config.yaml
```

catalog 当前只包含真实 Desktop consumer `webmcp`。`threadId` 仅接受已加载 Thread identity；Lime 没有 Codex
project-local feature config，因此 Thread 不建立第二份 enablement store。Settings 写入是 Desktop 持久化配置语义，
未知 feature key 被忽略；多模型/多模态 catalog、provider capability/readiness、retry/circuit breaker 仍归 Grok-aligned
`model-provider`，不由 experimental feature catalog 承接。

旧 Electron `get_experimental_config` / `save_experimental_config`、Renderer IPC 直连、legacy Tauri facade 与生产 mock
fallback 为 `dead / deleted / forbidden-to-restore`；无 `compat` 或 `deprecated`。Architecture impact: major；本节将实验
配置业务 owner 从 Electron 收敛到 App Server current 主链。Responsible developer confirmation: root, 2026-08-10。

## 41. Desktop Permission Profile Owner

Lime Desktop 的新回合权限选择由 App Server permission profile catalog 与 Turn lowering 共同构成唯一 current owner。
Renderer 的 access mode 只是用户选择意图，不能直接把本地 sandbox 字符串当作 runtime 事实。

```text
Desktop access mode
  -> Renderer typed permissionProfiles gateway
  -> app_server_handle_json_lines
  -> permissionProfile/list
  -> unique allowed built-in profile
  -> turn/start { approvalPolicy, permissions }
  -> App Server profile resolver
  -> RuntimeRequest sandbox policy
  -> tool-runtime
```

catalog 只公开 `:read-only`、`:workspace`、`:danger-full-access` 三个 Desktop 内建 profile，并按 Codex 内建顺序返回。
Renderer 在每次新 Turn 提交前解析唯一且 `allowed=true` 的目标；App Server 分别 lowering 为 `read-only`、
`workspace-write`、`danger-full-access`，同时在 Turn metadata 保留 `permissions` 和 `activePermissionProfile` provenance。
未知/禁止/重复 profile、无效 catalog 以及 `permissions + sandboxPolicy` 组合均 fail closed。

Electron 仍只负责通用 App Server JSONL 转发，不新增权限 IPC 或第二份 catalog。Lime Desktop 不复制 Codex TUI picker，
也不读取 project-local `.codex/config.toml` 自定义 profile；`thread/settings/update.permissions` 尚未进入同一 resolver，
保持 planned/fail-closed。旧 Renderer 新回合 `sandboxPolicy` wire 为 `dead / deleted / forbidden-to-restore`；历史导入、
read model/evidence 中的 canonical sandbox fact 不属于兼容入口。多模型 catalog、默认选择、model switch、provider
capability/readiness、retry/circuit breaker 与多模态 sampling/media lowering 仍由 Grok-aligned `model-provider` 承接。

Architecture impact: major；本节新增 Desktop access mode 经 exact catalog 到 runtime sandbox owner 的跨层数据流，并固定
fail-closed、Desktop/TUI 分界和 settings mutation blocker。Responsible developer confirmation: root, 2026-08-10。

## Config Control Plane

Desktop 配置的唯一业务 owner 是 App Server config processor 与单一全局用户 `lime_core config.yaml`：

```text
Settings / fixture / internal provider selection
  -> AppServerClient config/read|config/value/write|config/batchWrite
  -> Electron app_server_handle_json_lines
  -> App Server config processor
  -> lime_core config.yaml
```

`config/read` 只暴露用户层并返回版本；写入必须经过当前版本、已知 key 和当前绝对 `filePath` 校验。project-local、
MDM、requirements 层与 Codex `configRequirements/read` 不属于 Lime Desktop 产品范围，必须 fail closed 或保持
product-scope-excluded。Electron 不承接配置业务，不恢复 `get_config` / `save_config`，Settings/Claw fixture 的证据只
记录 `app_server_handle_json_lines` 与 `config/*` method。MCP 外部 `config.toml` reload 同样 excluded，避免第二配置 owner。

Architecture impact: major；本节完成 Electron config facade 到 App Server config control plane 的 owner 迁移，并同步
Settings/Claw evidence。Responsible developer confirmation: root, 2026-08-10。

## 42. Windows Sandbox Readiness Owner

Windows sandbox readiness 是 Desktop 控制面能力；Windows restricted-token runner 的当前基础也已归入
`tool-runtime`，但 readiness 仍必须以真实平台 enforcement 证据为准。唯一数据流为：

```text
Settings execution policy
  -> typed windowsSandbox/readiness gateway
  -> app_server_handle_json_lines
  -> App Server windowsSandbox/readiness
  -> tool-runtime::plan_sandbox_backend
  -> WindowsSandboxReadiness { notConfigured | updateRequired | ready }
```

非 Windows 平台或未启用 workspace sandbox 返回 `notConfigured`；Windows 已启用但 backend 不是
`Ready + enforced=true` 返回 `updateRequired`；只有真实 enforcement 才能返回 `ready`。`tool-runtime`
当前拥有 target-gated restricted-token runner：受限 token、capability SID、workspace/explicit write-root
ACL lease、`.git/.codex/.agents` 写入拒绝、Job Object 进程树、显式继承句柄列表、stdout/stderr reader
和既有有界 retained output 均在同一 owner。Windows 环境块继承父环境并按大小写不敏感应用请求覆盖；正常根进程
退出时遵循 Codex 语义关闭 `KILL_ON_JOB_CLOSE`，由 reaper 持有 Job 与 ACL lease 到 Job 为空，取消、超时、控制断开或
等待失败则终止整棵进程树并立即 rollback；TTY/ConPTY、elevated setup、WFP/firewall 强网络隔离和
Windows 真机证据尚未完成。命令 lowering 仍对 `RestrictedToken` fail closed，但执行入口会在 Windows
直接进入 runner，不把 command wrapper 当作第二套 owner。因平台 evidence 尚未具备，当前
`SandboxBackendStatus::Planned`、`enforced=false` 保持不变，不能由 runner 源码、Settings 或 setup 文案
推断 ready。`windowsSandbox/setupStart`、`windows/worldWritableWarning` 与 `windowsSandbox/setupCompleted`
保持 planned，后续必须在 Windows 平台完成 runner/ConPTY/网络能力裁决与 Gate B evidence 后再推进。

Electron 只转发 App Server JSONL，不新增 Windows 业务 IPC 或第二套设置后端；Desktop Settings 只消费 readiness
状态，不复制 Codex TUI 的 setup UI。Windows runner 的 token、ACL、进程生命周期和网络/读限制仍归
`tool-runtime` sandbox owner；多模型、多模态 sampling/media lowering 与 provider readiness 仍归
Grok-aligned `model-provider`，与此控制面无关。

Architecture impact: major；本节固定 Windows readiness 与实际 enforcement 的 fail-closed 边界，并把
Desktop Settings、App Server 与 tool-runtime 的唯一数据流写入架构事实源。Responsible developer confirmation:
root, 2026-08-10。Confirmation content: 已核对 `SandboxBackendStatus::Planned`、`enforced=false`、
`prepare_sandbox_command(RestrictedToken)` 拒绝路径、非 Windows/未配置状态，以及 Desktop/TUI、Codex runtime 和
Grok model/multimodal owner 分界。

## 43. Desktop Composer Fuzzy File Search Owner

项目文件 `@` 补全属于对话工作台 Composer 的紧凑选择面板，不是新的文件浏览页，也不是 pending interaction。
唯一产品链为：

```text
CharacterMention current @token
  -> project root from App Server project read
  -> Renderer typed fuzzyFileSearch gateway
  -> Electron app_server_handle_json_lines
  -> App Server fuzzyFileSearch processor
  -> filesystem search owner
  -> relative project path candidates
  -> replace only the active @token
```

App Server 校验每个 root 是可读绝对目录，不跟随目录 symlink，并跳过 `.git/.hg/.svn/node_modules/target/dist/build/coverage`
目录；结果限制为 50 条并按 score/path 稳定排序。同 cancellation token 的新请求标记旧扫描取消，Renderer 再用
AbortSignal 与 request version 拒绝迟到响应。空 query/root 返回空结果；GUI 只显示项目相对路径，空格路径加引号，
不会把本地文件伪装为 `plugin://`、connector 或 canonical UserInput Mention。

Codex experimental session request/notification 会建立第二套长生命周期 search registry，Lime Desktop 没有对应产品
consumer，因此统一为 `product-scope-excluded / forbidden-to-restore`；notification inventory 只做脱敏 drift diagnostics，
不进入 current projector。Electron 不新增业务 IPC，Grok-aligned 多模型 catalog、route、capability/readiness 与多模态
sampling/media lowering owner 不变。

Architecture impact: major；本节新增 Composer 到 App Server filesystem search owner 的跨层数据流，并固定 one-shot
request、双层取消、Desktop/TUI 分界和 session surface 排除。Responsible developer confirmation: root,
2026-08-10。Confirmation content: 已核对 protocol/processor/client/gateway/Composer 依赖方向、绝对 root 与相对结果、
session 回流守卫、Electron 通用转发边界，以及 Grok 多模型/多模态 owner 不变。

## 44. Desktop CodeMode Runtime Boundary

CodeMode 是 Agent runtime 的工具编排模式，不是 Codex TUI 功能。Lime Desktop 只复用其 runtime contract；
模型 capability/readiness 与 provider wire 仍分别归 Grok-aligned `model-provider` 控制面和各协议 lowering。
目标唯一数据流为：

```text
Desktop turn request + selected model capability
  -> selected profile slot requiredCapabilities
  -> authoritative model runtime_features
  -> resolved provider protocol/host capability intersection
  -> App Server / agent-runtime sampling-step snapshot
  -> tool-runtime RuntimeToolMode + frozen CodeMode tool plan
  -> model-provider native freeform/custom-tool lowering
  -> thread-owned CodeMode session runtime
  -> exec/wait -> canonical nested RuntimeTool execution
  -> Tool lifecycle + Thread/Turn/Item projection -> GUI
```

当前已落地的 production foundation 包含 planning boundary、Agent loop executable boundary、thread-owned
session service、in-process V8 provider 与 Runtime backend factory 接线。
`RuntimeToolExposure` 与 Codex 一致为
`Direct / Deferred / DeferredModelOnly / DirectModelOnly / CodeModeOnly / Hidden`；规划器从同一个 frozen
tool snapshot 生成 direct model、searchable 和 nested surface，并固定 namespace 拼接、JavaScript identifier
normalization、`exec`/`wait` 保留名与 normalization collision 的确定性 first-winner。普通 `CodeMode` 在 runtime
不可用时仅可按显式策略回退 Direct；`CodeModeOnly` 或禁用 fallback 必须 fail closed。

`tool-runtime::code_mode` 是 transport-neutral session contract owner：`execute` 返回带稳定 `cell_id` 与独立
`initial_response` future 的 `RuntimeCodeModeStartedCell`，`wait/terminate` 返回保留 live/missing 语义的 outcome，
同一 handle 提供 `shutdown`；session provider 在启动 host 前报告 availability，并以 delegate 承接 nested tool、
notification 与 cell close，非默认 yield/heap limits 在 provider 未实现时 fail closed。模型可见结果固定包含
`Script running/completed/failed/terminated` 状态，yielded 输出必须带 `cell_id`，error 不能吞掉已产生的 output；
output token budget 复用 `tool-runtime` 的统一 token truncation。

`agent-runtime::provider_turn::code_mode` 只在 frozen sampling-step snapshot 持有 executable session handle 时，
成组广告 Codex `exec` custom tool 与 `wait` function tool；没有 handle 时继续拒绝 custom call，生产默认 snapshot
不会暴露这两个工具。provider response 必须先完整 materialize，再执行 function/custom/wait，混合结果按原始 call
顺序回写 transcript；同批并行策略仍保留。turn cancellation 在 cell 已启动后必须调用同一 session 的
`terminate`，不得只丢弃 future 留下后台 cell。`wait` 的 `yield_time_ms/max_tokens/terminate` 都由该窄模块解析，
普通 tool executor/lifecycle 不冒充 CodeMode runtime。

模型门禁继续复用 Grok-aligned catalog/readiness 唯一事实源。`tool_mode` 与 provider capability 是两个独立事实：
catalog/direct provider config 只接受 `direct / code_mode / code_mode_only` 三个精确 token，缺失或未知值统一为
`Direct`，禁止按 model/provider 名称推断。基础 coding requirement 固定为
`coding/tools/streaming`，只有最终选中的 profile slot 才能追加 `custom_tools`；review/fast/local 等未选中
slot 只保留诊断展示，fallback 后按最终 slot 重算。模型侧必须在 authoritative `runtime_features` 显式声明
`custom_tools`。当前只有 Codex 上游同样声明 freeform tool 的精确 canonical 映射 `openai/gpt-5.2` 带该 capability，
但该模型没有 `tool_mode` 声明，因此仍按 `Direct` 执行；capability 不能反向开启 CodeMode。resolved route 再与实际 provider protocol/host 求交集：仅官方 OpenAI
Responses route 保留该 feature；Chat Completions、Anthropic、Gemini、Azure、Ollama 和第三方 Responses
均从 effective snapshot 移除，若选中 slot 要求该能力则以不可重试的
`capability_gap / capability:custom_tools` 在 sampling 前失败。普通聊天没有该额外 requirement，不受影响。

`model-provider` 已建立 provider-neutral `ToolDefinition::Custom`、`FreeformToolFormat`、
`CustomToolCall/CustomToolResult` canonical contract，并只在官方 OpenAI Responses route 的显式
`custom_tools` capability 下 lowering 为原生 `type: "custom"`、`custom_tool_call` 和
`custom_tool_call_output`；Chat Completions、Anthropic、Gemini、第三方 Responses route 与无 capability
历史均在发网前 fail closed。失败 custom output lowering 必须优先保留格式化 runtime output，不得只发送裸 error。

`agent-runtime::session_loop` 现已成为 canonical thread-owned CodeMode lifecycle owner。actor 创建必须同时绑定
`session_id + thread_id`；同一 session 的 thread identity 漂移直接 fail closed，不保留旧单参数入口。provider
availability 通过后才建立 service，runtime session 在首次 CodeMode operation 时 lazy create；actor replace/interrupt
终止 active cells，shutdown 关闭已初始化 session，未初始化 service 的 shutdown 不得反向创建 runtime session。
task/input handle 只从 actor resources 读取同一 canonical `thread_id` 与 session handle，不另建全局 registry。

production Runtime backend factory 只给 current runtime backend 注入 `RuntimeCodeModeServiceFactory::production()`；
mock、external 与 unavailable backend 不注入。factory 只使用 `ProcessCodeModeSessionProvider`，通过同目录
`code-mode-host` 的 length-prefixed stdio protocol 创建 session；host 进程内部使用 `V8CodeModeSessionProvider`，每个
cell 在 fresh sandbox-enabled V8 isolate 中执行 async module。host 仅暴露 frozen nested tools、
text/image/audio/generatedImage、store/load、notify、timer、yield 与 exit，不提供 Node、filesystem、network 或
console。session store 只在同一 thread-owned host session 内共享，cell terminate 使用 V8 isolate handle；host
缺失、握手不兼容、崩溃或资源上限不支持均 fail closed，production 不回退 App Server 进程内 V8。
CodeMode model surface 必须同时满足模型 `tool_mode` 请求、resolved provider `custom_tools` capability 和当前
`RuntimeSessionInputHandle` 持有 executable session，随后才附着 session 并由 provider turn 成组广告 `exec/wait`。
普通 `CodeMode` 缺任一条件回落 Direct，`CodeModeOnly` 缺任一条件 fail closed，`Direct` 即使持有 session 也不广告。

每个 `exec` cell 现在通过 `RuntimeCodeModeService` 建立独立 dispatch route；runtime 在 `execute` 返回
`StartedCell` 前发起 nested callback 时，route gate 会等待 cell identity 绑定，而不是落到最近 turn 或 session-wide
可变 delegate。provider-turn delegate 只从本 sampling step 的 frozen `RuntimeCodeModeTool` 集合按 JavaScript global
name 查找定义，并复用同一个 `RuntimeToolExecutorHandle.bind(...).execute_call(...)`，因此 nested tool 继续经过普通
权限、取消、lifecycle 和 output projection；`exec`/`wait` control tool 禁止递归 nested 调用。cell terminal、terminate、
interrupt、cell-close 与 shutdown 都清理 route/gate，不能把下一 turn 的 executor 接到旧 cell。

`exec` 与 `wait` 现在都经过 canonical `ToolLifecycleEmitter` 发出 Started/Completed；完成输出保留 cell identity、
CodeMode terminal 状态、`handler_executed` 和格式化 partial output，并由既有 Tool lifecycle 投影形成普通 Tool Item。
nested `notify` 复用同一 turn/call/cell correlation 发出 `ToolOutputDelta`，经 Lime Agent 的既有 Desktop host event
pipeline 进入 App Server/GUI 可消费的事件链；同时以同一 outer `exec` call id 在本 sampling step 的下一次 provider
request 中追加 `custom_tool_call_output`，顺序位于最终 exec output 之前，等价于 Codex
`inject_if_running(CustomToolCallOutput)` 在当前 turn 内的模型可见结果。空通知静默，已取消通知 fail closed。该注入
只存在于当前 provider turn 的下一 sampling request，不经过 `RuntimeSessionInputHandle`，也不作为独立 durable Item
持久化；CodeCell 的 thread-owned trace/evidence owner 尚未建立，不能借用 current-turn `Tool` Item 或公开 ThreadItem
伪造跨 Turn 的 started/closed 阶段。cell route 还保留 closed-cell 终态集合：terminal response、terminate、
interrupt、shutdown 或 host `cell_closed` 后，迟到的 nested invoke/notify 直接返回 closed error，不会重新创建空 gate
并永久等待，也不会落到 fallback delegate；provider-turn delegate 同时以原子 closed 标志拒绝迟到的 Desktop delta 和
provider transcript output。

V8 build 输入固定从 `lime-rs/Cargo.lock` 读取精确 crate version，`scripts/lib/rusty-v8-artifacts.mjs` 只接受
Codex `ptrcomp_sandbox_release` 的受支持 platform target，下载 archive/binding/checksum manifest 后逐项校验 SHA-256，
再向 Rust test、local CI 和 App Server sidecar build 注入成对的 `RUSTY_V8_ARCHIVE` /
`RUSTY_V8_SRC_BINDING_PATH`。V8 archive 编译期只静态链接进 `code-mode-host`，不复制进 Electron resources。
dev、Electron asset 与 Windows build 使用同一次 Cargo invocation 成组产出 `app-server` 和 `code-mode-host`；Electron
release bundle 将二者放在同一平台目录，manifest 分别记录 `sha256` 与 `codeModeHostSha256`，packaged verifier 必须同时
校验。`default_code_mode_host_path` 只解析 App Server/测试二进制同目录，不搜索 PATH 上的替代 runtime。

CodeMode 专项 Electron Gate B 已通过
`.lime/qc/gui-evidence/code-mode-electron-gate-b/code-mode-electron-gate-b-summary.json` 建立：真实 Electron、
preload/contextBridge、IPC、`app_server_handle_json_lines`、App Server runtime backend、official-host Responses route、
production process factory、custom `exec` 回采样、canonical Thread/Turn/Tool Item 与 GUI 可见终态使用同一 identity；
Electron、App Server 与 `code-mode-host` PID 分别为 `44199 / 44203 / 44521`，host 的父 PID 精确为 App Server。公开 Item
类型只有 `userMessage/dynamicToolCall/agentMessage`，mock fallback、invoke/console/page/provider error 均为零。该场景通过
标准 `HTTP_PROXY` 把仍以 `api.openai.com` 为 Host 的请求路由到受控本地 fixture，只证明 official-host capability、
production lowering 与 macOS dev-process isolation，不冒充 live OpenAI、Windows/packaged parity 或公网稳定性证据。

当前 alignment blocker 只剩 thread-owned CodeCell trace/evidence owner；Codex 公开 App Server ThreadItem 本身没有
CodeCell variant，现阶段只消费既有 outer Tool Item/GUI lifecycle surface，不新增 CodeCell GUI card 或产品事件旁路。
不得借用系统 Node、shell eval、Electron renderer 或 Codex TUI 进程执行 JavaScript；CodeCell trace 只在建立真实
consumer 与唯一 trace/evidence owner 后实现。Electron 仍只做 Desktop Host、sidecar lifecycle 与标准 GUI 投影。

Architecture impact: major；本节固定 CodeMode 的唯一 owner、Desktop/TUI 分界、selected-slot requirement、
authoritative model declaration 与 resolved provider route 交集、provider custom contract、session lifecycle contract
和 Agent loop fail-closed 上线顺序。Responsible developer confirmation: root, 2026-08-12。Confirmation content:
已核对 Codex `ToolMode`/`ToolExposure`、`StartedCell`、session provider/delegate、sandbox V8、freeform `exec`、function
`wait`、yield/terminate/cancel 与 mixed-call transcript 顺序；确认 Lime 当前完成 transport-neutral contract、
canonical thread-owned lazy service、process-owned sandbox V8 host、production factory、三重门禁、per-cell nested
dispatch、outer Tool lifecycle、notify Desktop event/provider-transcript projection、双 sidecar 构建供应链与专项 Electron
Gate B。当前明确保留 thread-owned CodeCell trace/evidence owner blocker；未把受控 fixture 等同于 live provider，也未把
macOS dev Gate B 等同于 Windows/packaged parity。
