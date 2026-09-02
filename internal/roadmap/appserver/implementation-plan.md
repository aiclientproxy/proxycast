# App Server 分阶段实施计划

> 状态：current planning source
> 更新时间：2026-06-08
> 作用：把 App Server 从路线图推进到可执行的工程阶段，支持渐进式替换和多独立 App 复用。

## 1. 实施原则

1. 每一阶段都必须有可运行的最小闭环。
2. 先固定协议、RuntimeCore 和 ExecutionBackend，再迁复杂 runtime。
3. 不为单一 App 写专用 runtime 分支。
4. App Server 和 legacy desktop facade 必须共享 RuntimeCore。
5. 旧 command glue 只能迁出或下线；确需临时兼容时只允许在 current 边界投影，不继续长业务逻辑，也不在 `lime-rs/src/commands/**` 保留 facade。
6. 新 App 只能走 App Server Client，不直接链接 Lime 内部实现。
7. `lime-rs/src/commands/**` 已物理删除，不得恢复业务逻辑、API adapter、runtime 分支或领域服务实现。

## 2. 阶段总览

| 阶段 | 目标 | 关键产物 | 退出条件 |
| --- | --- | --- | --- |
| P0 | 文档和公共边界冻结 | 本目录 PRD / 架构 / 协议 / 图纸 / 实施计划 | 团队对 RuntimeCore / ExecutionBackend / HostBridge 无歧义。 |
| P1 | codex-rs app-server-style crate 家族骨架 | `app-server*` 六个 crate、protocol DTO、transport 空壳 | crate 命名和依赖方向与 codex-rs app-server 家族一致。 |
| P2 | RuntimeCore / ExecutionBackend | `RuntimeCore`、`ExecutionBackend`、`RuntimeEventSink`、`RuntimeHostContext`、`MockBackend` | MockBackend 可通过公共事件跑通最小 session/turn。 |
| P3 | RuntimeBackend adapter | Agent backend adapter、事件转换、cancel 桥 | 一个 Agent turn 可通过 RuntimeCore 跑通。 |
| P4 | App Server 接入 RuntimeCore | JSON-RPC router、stdio transport、server request processor | App Server 不直接拼 runtime，只调用 RuntimeCore。 |
| P5 | Lime Desktop Host bridge / compat projection | Electron Desktop Host bridge、legacy desktop projection、DesktopEventSink | Desktop 主路径不回退；旧 command name 如需临时存在，只投影到 current 边界并有退出条件。 |
| P6 | content-studio 试点 | Electron main client、sidecar 管理、业务对象绑定 | content-studio 可通过 App Server 发起 Agent session。 |
| P7 | Tool / Action / Artifact / Evidence | action/respond、tool events、artifact/evidence API | 审批、artifact、evidence 事件同源。 |
| P8 | 多 App 复用 | capability discovery、client isolation、本地 socket 评估 | 第二个独立 App 不新增 runtime 实现即可接入。 |
| P9 | 退场审计 | legacy desktop glue 分类、守卫、删除计划 | 旧 runtime glue 有退出条件和扫描守卫。 |

阶段推进时不得把 `lime-rs/src/commands/**` 当成新阶段产物目录。P3 / P4 / P7 的后端能力进入 App Server crates、RuntimeCore、ExecutionBackend 和 services；P5 的桌面壳能力进入 Electron Desktop Host bridge。legacy desktop facade 只允许在迁移期做参数投影和事件转发；如果该 facade 位于 `lime-rs/src/commands/**`，迁出后必须撤注册并删除，删不动只登记 blocker 和退出条件。

## 3. P0：文档和协议冻结

已完成目标：

1. 新增 `internal/roadmap/appserver/`。
2. 固定 App Server 是跨 App Agent runtime current 服务边界。
3. 固定 `RuntimeCore / ExecutionBackend / HostBridge / Protocol` 四层切分。
4. 固定 Agent 只是第一个 backend adapter。
5. 固定 stdio JSON-RPC 为第一 transport。

退出条件：

1. `README.md` 有事实源分类。
2. `prd.md` 有用户故事和验收。
3. `architecture.md` 有分层架构。
4. `protocol.md` 有方法和事件草案。
5. `sequences.md` / `flowcharts.md` 有时序和流程。

## 4. P1：codex-rs app-server-style crate 家族骨架

目标：

先参考 Codex CLI 仓库里 `codex-rs` 的 app-server crate 边界，避免公共代码继续散在壳层或单一后端里。该参考只覆盖 Rust App Server / protocol / client / daemon 分层，不覆盖 Codex App UI 或桌面壳实现。

建议产物：

1. `lime-rs/crates/app-server-protocol`
2. `lime-rs/crates/app-server-transport`
3. `lime-rs/crates/app-server`
4. `lime-rs/crates/app-server-client`
5. `lime-rs/crates/app-server-daemon`
6. `lime-rs/crates/app-server-test-client`

退出条件：

1. 六个 crate 可独立编译。
2. 依赖方向为 protocol <- transport/client/server，server 不反向污染 protocol。
3. 不依赖 retired desktop host crates。
4. 不出现 Agent 私有 DTO。

## 5. P2：RuntimeCore / ExecutionBackend

目标：

把公共 runtime 事实源和后端执行适配拆开。

建议产物：

1. `RuntimeCore`
2. `ExecutionBackend`
3. `RuntimeEventSink`
4. `RuntimeHostContext`
5. `MockBackend`
6. `SessionService`
7. `TurnExecutionService`

退出条件：

1. service crate 不依赖 retired desktop host crates。
2. `MockBackend` 能输出 `turn.started / message.delta / turn.completed`。
3. App Server 未来只调用 RuntimeCore。
4. `TestEventSink` 能收集 deterministic events。

## 6. P3：RuntimeBackend adapter

目标：

让现有 Agent runtime 作为 backend adapter 接入 RuntimeCore。

建议工作：

1. 把 `runtime_turn` 中 host-independent 的 orchestration 收进 `RuntimeBackend`。
2. 把 Agent 私有事件转换为公共 runtime events。
3. 把取消 token / runtime queue 接入 backend 合同。
4. 先接最小 submit / cancel / event stream。

退出条件：

1. 一个 Agent turn 可通过 RuntimeCore 执行。
2. Agent 私有 DTO 不进入 App Server 协议。
3. cancel 能终止 active turn 并发出 terminal event。
4. 定向 Rust 测试覆盖 start / cancel / event stream。

## 6.1 P4：App Server 接入 RuntimeCore

目标：

让 App Server 只做协议和进程边界，不拥有 runtime 业务逻辑。

建议工作：

1. `initialize / initialized`
2. `agentSession/start`
3. `agentSession/read`
4. `agentSession/turn/start`
5. `agentSession/turn/cancel`
6. `AppServerRuntimeFactory`
7. standalone backend mode guard
8. outbound notification channel
9. in-process host listener bridge

退出条件：

1. `initialize` 前业务方法返回 `Not initialized`。
2. `turn/start` 返回 accepted。
3. 同步 backend events 随 request response 后追加为 `agentSession/event` notification。
4. 外部 runtime events 可通过 outbound channel 写出 stdio JSONL。
5. legacy desktop Agent host 可通过轻量 `AppServerEventBridge` 追加外部事件，不持有完整 App Server。
6. fixture 覆盖 request / response / notification。
7. standalone `app-server` 只能启动 host-independent backend。
8. legacy desktop Agent host 只能通过 host bridge 注入，不进入 protocol / router / standalone CLI。
9. 不新增或恢复 `lime-rs/src/commands/**` 中的 App Server JSON-RPC command wrapper。

## 7. P6：content-studio 试点

目标：

让 content-studio 成为第一批独立 App client，验证 App Server 的 App 复用价值。

建议工作：

1. Electron main 增加 `AppServerClient`。
2. main 负责 sidecar 生命周期。
3. preload IPC 保持 renderer 隔离。
4. Agent session 绑定 content draft / scene / material 等业务对象 ref。
5. renderer 只消费 projection event。

退出条件：

1. content-studio 能启动 / 连接 App Server。
2. 能创建 session 并发起 turn。
3. 事件能进入现有 Agent UI projection。
4. 业务对象 ref 能回写到运行记录。

## 8. P5：Tool / Action / Artifact / Evidence

目标：

把完整 Agent 工作台需要的非文本执行面接入协议。

建议工作：

1. `agentSession/action/respond`
2. `tool.started / tool.result / tool.failed`
3. `artifact/read`
4. `artifact.changed`
5. `evidence/export`
6. `evidence.changed`

退出条件：

1. action required 可被独立 App 响应。
2. tool 事件不由 UI 推断。
3. artifact refs 可被 App 读取和展示。
4. evidence export 复用 runtime facts。

## 9. P6：多 App 复用

目标：

从单 App sidecar 进化到可被多个本地 App 复用。

建议工作：

1. `capability/list`
2. client subscription isolation
3. app scoped permission profile
4. local socket transport feasibility
5. server lifecycle manager

退出条件：

1. 两个 App 可连接同一 server。
2. session / event 不串线。
3. capability 根据 appId / policy 过滤。
4. 第二个 App 接入无需新增 runtime service。

## 10. P7：退场审计

目标：

封住旧路，避免 runtime 逻辑继续回流到壳层 glue。

建议工作：

1. 盘点 legacy desktop command 中剩余 runtime 业务逻辑。
2. 给 compat / deprecated 路径写退出条件。
3. 补治理扫描，禁止新 runtime 逻辑落回 command glue。
4. 删除无入口 adapter。
5. 按命令族删除或迁出 `lime-rs/src/commands/**` 剩余 wrapper；不得新增 stub 或平行 compat 实现。

退出条件：

1. command glue 已迁出或下线；仍暂留的兼容投影有明确退出条件，且不在 `lime-rs/src/commands/**` 保留 wrapper / stub / facade。
2. 新 App 只通过 App Server Client。
3. App Server 协议成为新增 Agent 能力的默认入口。
4. 治理报告能发现回流。
5. `lime-rs/src/commands/**` 中的旧 wrapper 已按领域迁出核心逻辑、撤注册并删除；暂时删不动的对象必须登记 blocker、阻塞文件和退出条件，不能保留薄委托、stub 或退场 tombstone 当完成态。

## 11. 验证计划

### 11.1 Rust

建议优先：

```bash
cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol
cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server
cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent
```

实际 crate 名称以后续实现为准。

### 11.2 Contract

涉及 legacy desktop facade 或前端 bridge 时：

```bash
npm run test:contracts
```

### 11.3 GUI

影响 Lime Desktop 主路径时：

```bash
npm run verify:gui-smoke
```

### 11.4 独立 App

content-studio 试点应在其仓库内跑：

```bash
npm run typecheck
npm run test:functional
npm run smoke:electron
```

## 12. 风险和缓解

| 风险 | 表现 | 缓解 |
| --- | --- | --- |
| service 抽象过大 | 一开始就想迁所有 command | P1-P3 只做 session / turn / cancel。 |
| App 专用分支 | 为 content-studio 写专用 runtime | 只允许业务对象 ref，不允许专用 execution loop。 |
| 壳层依赖泄漏 | service 依赖 retired desktop host crates | service crate 加结构测试。 |
| 事件不一致 | legacy desktop event 和 App Server 事件两套 | 统一 `RuntimeEventSink`。 |
| 协议漂移 | TS client 和 Rust DTO 不一致 | schema / fixture / contract test。 |
| 多 App 串线 | 事件广播给错误 client | session subscription isolation。 |

## 13. 完成判定

整体目标完成时应满足：

1. App Server 是新增跨 App Agent 能力的默认入口。
2. Lime Desktop Agent 主路径通过 service 或 App Server 复用同一 runtime。
3. content-studio 至少一个真实业务 Agent flow 通过 App Server 完成。
4. Tool / action / artifact / evidence 事件跨 App 语义一致。
5. legacy desktop command glue 的迁出 / 下线计划可验证，且不会把 `lime-rs/src/commands/**` 当成完成态。
