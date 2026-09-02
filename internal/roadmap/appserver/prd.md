# App Server PRD

> 状态：current planning source
> 更新时间：2026-06-08
> Owner：Lime Runtime / 独立 App 集成

## 1. 背景

Lime 当前已经具备较完整的本地 AI Agent runtime：会话、线程、回合、工具、技能、MCP、workspace、memory、artifact、evidence、review、policy、task 和 subagent 都已在 Rust runtime 主链中形成基础能力。

问题在于，这些能力仍主要服务于 Lime Desktop 的当前壳层。当 content-studio 或未来更多独立 App 需要复用同一套 Agent 能力时，如果每个 App 都各自接入、复制或裁剪 runtime，就会出现：

1. runtime 事实源分裂。
2. tool / skill / evidence / permission 行为不一致。
3. 事件、审批、取消、恢复和 artifact 写回协议重复发明。
4. Lime Desktop 和独立 App 的能力演进无法共享。
5. 后续替换壳层、替换前端框架或扩展 App 形态时，会被 runtime glue 绑定。

因此需要把 Lime Agent runtime 抽象为 App Server：壳层和独立 App 不直接拥有 runtime，而是通过稳定协议调用同一个本地服务。

## 2. 一句话目标

把 Lime AI Agent Runtime 服务化为本地 App Server，让 Lime Desktop、content-studio 和未来更多独立 App 通过统一 JSON-RPC 协议复用同一套 Agent 执行能力。

## 3. 业务目标

1. 让 content-studio 能复用 Lime Agent runtime，而不是自建完整执行层。
2. 让 Lime Desktop 继续可用，同时逐步把 legacy desktop command glue 迁出或下线；确需临时兼容时只能在 App Server / RuntimeCore / Electron Desktop Host current 边界内投影，不能继续落在 `lime-rs/src/commands/**`。
3. 让未来独立 App 拥有一致的 Agent 执行、工具审批、事件流、artifact 和 evidence 能力。
4. 让同一套 runtime 能服务多个产品形态：桌面工作台、内容工厂、垂直 Plugin、自动化任务入口。
5. 让 App 只关心业务对象和 UI 投影，不关心模型调用、工具调度、权限和证据链路的内部实现。

迁移边界：`lime-rs/src/commands/**` 已物理删除，只允许在负向回流守卫或不可变历史 evidence 中引用。PRD 中提到的 legacy desktop command glue 只表示历史迁移对象；新增 Rust 后端能力必须进入 App Server crates / RuntimeCore / services，桌面壳能力进入 Electron Desktop Host。

## 4. 工程目标

1. 固定 App Server 的进程边界和 JSON-RPC 协议。
2. 抽象 `RuntimeCore`，供 App Server 和 legacy desktop facade 同时委托。
3. 抽象 `ExecutionBackend`，让 Agent 和未来更多执行后端都通过同一合同接入。
4. 把 session / thread / turn / task / run / tool / action / artifact / evidence facts 统一到 RuntimeCore。
5. 支持 stdio JSONL 作为第一阶段 transport，后续再扩展本地 socket 或 websocket。
6. 为 TypeScript client 生成或维护稳定协议类型。
7. 让 content-studio 的 Electron main 成为 app-server client，renderer 继续只通过 preload IPC 消费状态。
8. 建立协议 fixture、Rust 定向测试和 client contract 测试。

## 5. 用户故事

### 5.1 独立 App 用户：我想在业务 App 内完成 Agent 协作

作为 content-studio 用户，
当我在内容工作台里发起内容生成、素材分析或多轮修改时，
我希望 Agent 在当前业务对象内运行并返回事件、草稿、artifact 和需要我确认的动作，
这样我不需要跳回 Lime 通用 Chat 才能完成工作。

验收：

1. content-studio Electron main 能启动或连接 App Server。
2. renderer 只消费业务投影和 Agent event，不直接管理 runtime。
3. 会话事件能映射到 content-studio 的当前业务对象。

### 5.2 Lime Desktop 用户：我不应该感知 runtime 被服务化

作为 Lime Desktop 用户，
当现有 Chat、Workspace、Plugin 或 automation 入口迁到 App Server 背后时，
我希望功能、事件、取消、审批和 evidence 导出保持一致，
这样服务化不会造成产品体验倒退。

验收：

1. Electron Desktop 主路径接入 App Server 后仍保持原有 GUI 行为。
2. legacy desktop 命令不再作为完成态；迁移期若确需兼容命令名，只能投影到 App Server / RuntimeCore / Electron Desktop Host current 边界，并写清退出条件。
3. `ThreadReadModel`、artifact、evidence 行为不分裂。

### 5.3 App 开发者：我只想接入协议

作为独立 App 开发者，
当我要接入 Lime Agent 能力时，
我希望只需要使用 App Server Client 和稳定协议类型，
这样不需要理解 Lime Desktop 内部目录、legacy desktop command 和 runtime_turn 细节。

验收：

1. 有 TypeScript client 或协议 schema。
2. 初始化、创建会话、发起 turn、取消、响应 action、读取状态都有稳定 API。
3. 错误码、事件、状态机和 capability discovery 可测试。

### 5.4 Runtime 维护者：我想只维护一条执行主链

作为 Lime runtime 维护者，
当多个 App 使用 Agent 能力时，
我希望所有执行仍进入同一个 RuntimeCore，并按 ExecutionBackend 分发，
这样 tool runtime、workspace、memory、artifact、evidence 和 policy 不会在多个 App 中复制。

验收：

1. legacy desktop facade 与 App Server 都委托 `RuntimeCore`。
2. Agent 只是 `ExecutionBackend`，不再污染公共协议。
3. 服务层不依赖具体壳层。
4. 新 App 不允许绕过服务层新增完整 runtime。

## 6. 核心需求

### 6.1 App Server 生命周期

首期必须支持：

1. `initialize`
2. `initialized`
3. server metadata 返回：版本、协议版本、平台、可用能力。
4. 客户端 metadata：app id、app name、版本、能力声明。
5. 单连接初始化门禁：初始化前拒绝业务方法。

### 6.2 Agent session

首期必须支持：

1. `agentSession/start`
2. `agentSession/read`
3. `agentSession/turn/start`
4. `agentSession/turn/cancel`
5. `agentSession/action/respond`
6. `agentSession/event` notification

### 6.3 事件流

事件至少覆盖：

1. session started / resumed
2. turn accepted / started / completed / failed / canceled
3. assistant message delta / completed
4. tool started / result / failed
5. action required / resolved
6. artifact changed
7. evidence changed
8. runtime status changed

### 6.4 Capability discovery

首期必须支持：

1. `capability/list`
2. `skill/list`
3. `tool/list`
4. `workspace/read`

这些接口只暴露 App 可消费的 capability，不暴露 Lime 内部模块路径。

### 6.5 独立 App 绑定

App 发起 session 时必须能声明：

1. `appId`
2. `workspaceId`
3. `businessObjectRef`
4. `artifactPolicyRef`
5. `evidencePolicyRef`
6. `locale`
7. `runtimeWorkspaceRoots`

runtime facts 使用稳定协议值；用户可见文案由 App 自己的 UI 层本地化。

## 7. 非功能需求

| 类别 | 要求 |
| --- | --- |
| 稳定性 | server 崩溃时 client 能感知并展示可恢复状态。 |
| 可测试性 | 协议 fixtures 和 Rust service tests 必须能覆盖最小闭环。 |
| 跨平台 | macOS / Windows 默认支持 stdio sidecar。 |
| 可观测 | server stderr 输出结构化日志；业务事件走 JSON-RPC notification。 |
| 安全 | App 只能请求声明过的 capability；工具和 workspace 权限由 runtime policy 决定。 |
| 性能 | 事件流必须支持 backpressure；长输出使用 delta。 |
| 版本 | 协议带 `protocolVersion` 和 capability flags，避免靠字符串猜行为。 |

## 8. 非目标

本阶段不做：

1. 不做云端 app-server。
2. 不支持多个远程用户共享同一 server。
3. 不把所有 legacy desktop commands 一次性迁走。
4. 不把 App Server 做成 UI host。
5. 不让业务 App 直接操作 runtime 数据库。
6. 不新增独立 evidence exporter 或 artifact store。
7. 不在 `lime-rs/src/commands/**` 新增或恢复 Tauri command wrapper、stub、compat facade 或业务实现；旧 wrapper 迁完后必须撤注册并删除，删不动只登记 blocker 和退出条件。

## 9. 验收标准

P1 最小闭环完成时必须满足：

1. App Server 可通过 stdio 启动。
2. client 完成 `initialize -> initialized`。
3. client 发起 `agentSession/start`。
4. client 发起 `agentSession/turn/start`。
5. server 返回 accepted，并发出 started / completed 或 failed notification。
6. client 可取消 active turn。
7. 协议 fixture 覆盖 request / response / notification。
8. legacy desktop command 迁移计划明确哪些保留为 compat facade。
9. `lime-rs/src/commands/**` 的旧 wrapper 已删除；治理守卫禁止路径、注册和命令面回流。

P4 content-studio 试点完成时必须满足：

1. content-studio Electron main 只通过 App Server Client 调用 Agent 能力。
2. renderer 不直接管理 runtime。
3. 业务对象和 Agent session 可互相追踪。
4. 事件、artifact、action、error 均能映射到 content-studio UI。

## 10. 成功指标

1. 新独立 App 接入 Agent 能力时，不需要新增 runtime 实现。
2. legacy desktop command 层 Agent 业务逻辑 LoC 逐步下降。
3. App Server 协议 fixture 成为跨 App 合同测试输入。
4. Tool / action / artifact / evidence 事件在 Lime Desktop 和 content-studio 中语义一致。
5. 后续替换壳层或新增 App 时，只新增 client adapter 和 UI projection。
