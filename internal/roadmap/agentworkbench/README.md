# Lime Agent Workbench 路线图

> 状态：active
> 更新时间：2026-06-12
> 范围：`/Users/coso/Documents/dev/ai/limecloud/lime-agent-workbench` 文档站、Lime AgentUI SDK 标准、AgentRuntime 配合边界、Subagents 标准化。

## 主目标

把 `lime-agent-workbench` 从“参考 AG-UI 的中文文档站”推进成 Lime 内部 AgentUI / AgentRuntime 标准指引，并让后续实现只能向 Lime current 主链收敛：

```text
App Server / RuntimeCore / ExecutionBackend
  -> RuntimeEvent / ThreadReadModel / TaskSnapshot / EvidencePack
  -> @limecloud/agent-ui-contracts
  -> @limecloud/agent-runtime-projection
  -> @limecloud/agent-runtime-ui
  -> Product Apps / Content Studio / Plugins
```

Workbench 不是外部 SDK 套壳，也不是第二套 runtime。外部 AG-UI、AI SDK、OpenAI Agents JS、LangGraph 等只能作为参考；Lime 的事实源仍是 App Server、RuntimeCore、ExecutionBackend、AgentUI Projection、Evidence / Replay / Review 和治理分类。

## 固定结论

1. AgentRuntime 拥有执行事实；AgentUI 只消费 facts 并投影 UI。
2. UI 标准使用 `UIMessageParts`、`ProcessTimeline`、`ExecutionGraph`、ToolGroup、ActionRequired、ArtifactRef、EvidenceRef；不再使用非标准组件树术语。
3. TypeScript 包线拆为四个 current 包，`@limecloud/agent-ui` 只允许作为 future facade。
4. 产品应用不能各自复制过程组件、runtime client、投影 reducer 或工具状态机。
5. `subagents` 必须进入 RuntimeEvent / TaskSnapshot / ExecutionGraph / `AgentUiProjectionState.subagents` 主链，不能靠 assistant 正文或本地 React state 表达。
6. 文档站必须和 Lime 现有代码、路线图、App Server / AgentRuntime / AgentUI 事实源对齐；不能写成外部标准翻译稿。

## 当前阶段

当前处于 `v0.2.x -> v0.4.x` 标准化阶段：

| 阶段 | 状态 | 目标 |
| --- | --- | --- |
| v0.1.0 | done | 建站、发布 GitHub Pages、形成中文文档骨架。 |
| v0.2.0 | done | 去掉非标准树形过程方向，初步对齐 AG-UI / AI SDK / Lime runtime 现实。 |
| v0.3.0 | active | 补齐 Subagents、SDK 包边界、runtime client、projection conformance、Lime 现有实现迁移路线。 |
| v0.4.0 | done | 从文档标准推进到 Lime 主仓包实现、guard、Artifact/Evidence refs surface、Plugin 真实 smoke 和 Content Studio 标准 UI surface adoption。 |
| v2.0 | done | 从“标准文档 + fixture replay”推进到可执行协议内核：sequence verifier、conformance fail closed。 |
| v2.1 | done | Runtime client fail-closed pipeline 和 projector 增量 apply。 |
| v2.2 | done | `state.delta` / JSON Schema seed，支撑跨语言机械校验。 |
| v2.3 | done | Workbench 文档站活体闭环和 `/subagents` 标准页收口。 |
| v2.4 | done | Runtime client middleware / adapter 协议演进层，Lime 本体 current event gateway 接入同一 pipeline。 |
| v2.5 | done | `state.delta` 接入 projection / read model apply，Lime Plugin current projection 同步消费。 |
| v2.6 | done | Rust/App Server RuntimeCore event 入库前接入 AgentUI runtime event / `state.delta` JSON Schema gate。 |
| v2.7 | done | Workbench 文档站活体 Demo 矩阵，真实 replay 多个 conformance fixtures。 |
| v2.8 | done | Lime 本体 App Server RuntimeCore event 入库前接入 AgentUI sequence gate，坏流不污染 session state。 |
| v2.9 | done | Runtime client `0..N` fan-out / flush substrate 与 Lime 本体 App Server、本地 publish、bridge listener、Plugin current 路径消费接入。 |
| v2.10 | done | Runtime / Provider capability manifest 与 resume contract：contracts seed、App Server protocol/RuntimeCore 承接、Lime 前端 current gateway 消费。 |
| v2.11 | planned | Projection reconciliation、tool args buffer、reasoning continuity 与外部 transport compatibility 边界。 |

## 阅读顺序

| 文档 | 作用 |
| --- | --- |
| [v2.md](v2.md) | v2 可执行协议内核：重新评估 AG-UI 深层机制，固定 sequence verifier / pipeline / middleware / demo 的分阶段实施。 |
| [iteration-plan.md](iteration-plan.md) | 本次迭代目标、缺口、写集和完成判定。 |
| [task-board.md](task-board.md) | 当前迭代可并行认领的任务板、写集、验收和推荐下一刀。 |
| [parallel-workstreams.md](parallel-workstreams.md) | 多进程并行认领方式、每条 workstream 的写集和验收。 |
| [acceptance.md](acceptance.md) | Workbench 标准、SDK、runtime、subagents 的统一验收矩阵。 |
| [standard-package-release-runbook.md](standard-package-release-runbook.md) | 标准 npm 包发布、Content Studio 安装接入和 GUI smoke 的阻塞解除步骤。 |

相关既有路线图：

- `internal/roadmap/agentui/README.md`
- `internal/roadmap/agentruntime/README.md`
- `internal/roadmap/agentruntime/agentui-adoption-gap.md`
- `internal/roadmap/plugin/README.md`

## current / compat / deprecated / dead

### current

- `lime-agent-workbench` GitHub Pages 中文标准站。
- Lime App Server JSON-RPC + RuntimeCore + ExecutionBackend runtime facts。
- `RuntimeEvent / ThreadReadModel / TaskSnapshot / EvidencePack`。
- `@limecloud/agent-ui-contracts`、`@limecloud/agent-runtime-projection`、`@limecloud/agent-runtime-ui`、`@limecloud/agent-runtime-client`。
- Product Apps 通过 runtime client + projection + React surfaces 复用 AgentUI。
- Plugin 运行页通过标准 `AgentRuntimeClient`、`projectAgentUiState` 和 `AgentUiProjectionView` 接入共享 SDK seed。
- Plugin Host action 决策已走 current `AgentRuntimeClient.respondAction -> agentSession/action/respond`；Host drawer 主体只保留抽屉、指标和标准 projection 接线，本地 process fallback / projection 输入增强已拆成独立模块。
- Plugin 真实 Electron 主路径已通过 smoke：`Plugins` 聚合入口 -> runtime iframe SDK -> App Server current `agentSession/*` -> Host Agent Run 标准 projection DOM。
- Claw / Plugin 的 Lime 前端 current event gateway 已接入 v2.9 runtime event pipeline：App Server `agentSession/event` notification、本地 publish、bridge listener 与 Plugin current runtime client options 都消费 `@limecloud/agent-runtime-client/sessionGateway` 的 browser-safe adapter / middleware / verifier 输出；pipeline 支持 `0..N` fan-out / flush，未配对 `tool.result` 等坏流 fail-closed，历史非 App Server payload 不误伤。
- Rust/App Server RuntimeCore 在 event 入库前已接入 AgentUI JSON Schema gate：App Server `AgentEvent` 会先规范化为 Workbench `AgentRuntimeExecutionEvent` 形状并通过 runtime event schema；`state.delta` payload 额外通过 state delta schema，非法 patch fail closed，不写入 session state。
- Rust/App Server RuntimeCore 在同一入库边界已接入 AgentUI sequence gate：`tool.result/failed` 必须有同 turn `tool.started`，`action.required` 必须由 `action.resolved / action.cancelled / action.canceled / action.expired` 收口，current turn terminal 只认 `turn.completed / turn.failed / turn.canceled`，终态前必须收口 active tool/action；`done / final_done / turn.done / turn.final_done / turn.cancelled / cancelled` 属于 legacy terminal dead surface，只允许负向 guard / test-only 证明不能关闭 current stream。violation 直接 fail closed，不写入 `StoredSession.events`，也不会提前改变 turn/session 状态。
- `@limecloud/agent-ui-contracts` 当前只承载 Runtime / Provider capability manifest 与 action lifecycle；旧 resume contract 类型、JSON Schema、校验器和 `agentSession/thread/resume` 参数已删除。Lime 本体 App Server `capability/list` 继续返回 `runtimeCapabilityManifest`，自动 capability negotiation 仍不是已完成能力。
- `@limecloud/agent-runtime-projection` 已消费 `state.delta`：`projectAgentUiState` 与 `createAgentUiProjector.apply()` 都会把 RFC 6902 patch 归并进 `AgentUiProjectionState` / `readModel`，batch 与 incremental 输出等价；patch 失败时进入 `hydration.status = "stale"` 并写入 diagnostics，不污染目标 state。read model 与 runtime status 使用同一 action terminal 语义，避免后端已收口但 UI 仍残留 pending action。
- Content Studio `AI agents` 工作台和通用 `AgentSessionPanel` 已统一通过 `AgentUiProjectionSurface` 消费共享 projection read model，并输出 `.agent-ui-projection` / `.agent-ui-main` / `.agent-ui-sidecar` 标准 DOM surface；其 standalone/dev App Server turn 主链已通过 `@limecloud/agent-runtime-client/sessionGateway` 包装为标准 `AgentRuntimeClient`。标准 npm 链路已发布并接入 `@limecloud/app-server-client@1.66.0` 与 `@limecloud/agent-runtime-client@0.1.1`，Content Studio 不能安装误发的无 scope 链路；真实 Electron E2E 已证明 conversation/runtime 标准 surface 在产品页面可运行。
- `@limecloud/agent-runtime-ui` 已有 `ArtifactRefList` / `EvidenceRefList`，引用内容由宿主 artifact workspace / evidence pack owner 打开。
- `@limecloud/agent-runtime-ui` 已有 `SubagentsView`，只消费 `state.subagents` / 已投影的 `AgentUiSubagentsModel`；`subagent-handoff` fixture 已覆盖 channel、artifact、handoff、review、task completed、state.delta 和 snapshot.updated 的完整闭环。

### compat

- Lime 现有 `packages/app-server-client` 作为 `@limecloud/app-server-client` 的本地开发事实源。
- 未来 `@limecloud/agent-ui` facade 作为非 current 可选聚合入口。
- 产品应用本地 `messages / executionEvents / process panel / host bridge service` 作为迁移缓存；已接入共享 SDK 的页面不得再新增平级本地过程事实源。
- Plugin task event 作为 projection 输入，但必须桥接到标准 event class。

退出条件：facade 明确不导出 runtime transport，产品应用只显式依赖 current 四包后，不再用规划名描述 current owner。

### deprecated

- 每个产品应用自建 Agent 过程组件。
- 从 assistant prose 解析工具成功、artifact 类型、evidence verdict。
- 把 UI component tree 当成 runtime protocol。
- AgentUI React 组件直接订阅 Provider 或读取 App Server DB。

### dead

- 非标准树形过程术语作为 Lime 标准术语或协议。
- 已清退的旧协作 UI surface 作为 current 类型、组件、state 字段或 SDK API。
- 第二套 Plugin runtime facts。
- 业务 App 直连 Provider API 作为生产 Agent 能力。
- 把外部 SDK 作为 Lime runtime owner 直接套用。

## 本轮下一刀

v2.0-v2.10 已完成，`state.delta` projection apply、Rust/App Server schema + sequence enforcement、Workbench 活体矩阵、runtime-client fan-out / flush substrate、Lime 本体消费路径和 capability / resume contract seed 已补齐。后续优先补 AG-UI 仍领先的协议执行深水区：

1. RuntimeCore tool orchestrator：参考成熟执行内核的 `ToolRouter + ToolOrchestrator` 边界，把 MCP / ACP / skills / shell / project tools 收敛到单一 tool lifecycle owner。
2. Projection reconciliation：message snapshot、tool result adjacency、partial tool args、reasoning continuity。
3. 外部 transport compatibility：SSE / protobuf / Accept negotiation 仅作为 gateway 候选，不覆盖 current JSON-RPC owner。

## 并行协作规则

多进程并行时，先只读 `git status --short`，再认领窄写集。默认不要多个进程同时编辑同一个文件；如果必须改同一导航文件，先完成内容页，再由一个进程统一接导航。

本路线图目录是并行协作入口。新进程先读 `task-board.md` 认领窄写集，完成后把进度写回 `iteration-plan.md` 的进度日志，避免聊天记录成为唯一状态源。

发布与 Content Studio 接入阶段先读 `standard-package-release-runbook.md`。当前 scoped registry 发布、Content Studio 依赖接入和 GUI smoke / 产品回归证据已完成；后续进入 v2.11 协议深水区或单独授权的发布 / 删除任务。
