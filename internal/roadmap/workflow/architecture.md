# Workflow 标准化架构

> 状态：current planning source
> 更新时间：2026-07-04
> 作用：定义 Workflow 的统一合同、owner 边界、状态机和 Lime 模块映射。

## 1. 架构目标

Workflow 架构只追求六个目标：

1. 让 workflow run 成为 App Server RuntimeCore 的 durable execution facts。
2. 让 Plugin、Skill、图片命令、内容工厂和 GUI 使用同一套 definition/run/step/event/read model 合同。
3. 让 `workflowRunId` 可以 join session、thread、turn、task、tool、artifact 和 evidence。
4. 让 UI 只做投影，不反向定义 workflow truth。
5. 让 cancel / retry / respond / failure 可以审计和回放；`resume` 需要重新定义 current API 后再进入控制面。
6. 让旧前端 Hook、前端 DSL 和旧 content workflow API 有明确退场路径。

## 2. 总体分层

| 层 | 角色 | 输入 | 输出 | Owner |
| --- | --- | --- | --- | --- |
| Definition Source | 声明 workflow 结构 | Plugin manifest、Skill metadata、内置业务 workflow | `WorkflowDefinition` | Plugin / Skill / App package |
| Surface Adapter | 绑定业务入口与运行上下文 | entryKey、taskKind、user input、object refs | `workflow.start` request | frontend gateway / App Server facade |
| Runtime Control Plane | 创建、取消、重试、恢复 run | definition、session/thread/turn/task | `WorkflowRun`、control actions | App Server RuntimeCore |
| Execution Orchestration | 调度 step、tool、subagent、hook | `WorkflowRun`、capability policy | `workflow.*` runtime events | RuntimeCore / ExecutionBackend |
| Durable Facts | 持久化事件与快照 | RuntimeEvent stream | event log、read model | App Server store |
| Evidence / Replay | 导出可信证据 | workflow facts、tool facts、artifacts | evidence pack、replay case | Harness / Evidence |
| UI Projection | 展示和用户操作 | Workflow Read Model | run detail、progress、retry actions | Agent UI / Workspace / Plugin iframe |

## 3. 统一对象模型

### 3.1 `WorkflowDefinition`

```text
definitionId
schemaVersion
sourceKind          plugin_manifest | skill | builtin | image_command | test_fixture
sourceRef
title
taskKind
inputSchema?
outputArtifactKind?
steps[]
policies
```

Definition 是静态或半静态声明，不携带运行状态。

### 3.2 `WorkflowStepDefinition`

```text
id
title
kind                agent_task | skill | subagent | tool | connector | hook | artifact | evidence | storage | manual_gate
dependsOn[]
skillRefs[]
subagentRef?
toolRefs[]
expectedOutput?
inputMapping?
outputMapping?
retryPolicy?
humanReview?
```

`kind` 是 current 合同。旧 runtime DSL 的 `storage.set / knowledge.search / agent.startTask / artifacts.create / evidence.record` 需要映射到上述 step kind，不再作为平级标准。

### 3.3 `WorkflowRun`

```text
workflowRunId
definitionId
workflowKey
status
sessionId
threadId
turnId
taskId
appId?
workspaceId?
sourceObjectRef?
startedAt
updatedAt
finishedAt?
steps[]
artifacts[]
evidenceRefs[]
failure?
```

Run 是运行事实。任意用户可见进度、恢复和审计都从 run 派生。

### 3.4 `WorkflowStepRun`

```text
workflowRunId
stepId
status
attempt
startedAt?
updatedAt?
finishedAt?
toolCallIds[]
artifactIds[]
evidenceRefs[]
progressMessage?
failure?
```

### 3.5 `WorkflowEvent`

事件命名继续沿用 App Server 当前方向：

```text
workflow.run.started
workflow.run.retrying
workflow.run.completed
workflow.run.failed
workflow.run.canceled
workflow.step.started
workflow.step.progress
workflow.step.retrying
workflow.step.completed
workflow.step.failed
workflow.step.canceled
workflow.tool.started
workflow.tool.completed
workflow.connector.requested
workflow.connector.completed
workflow.hook.started
workflow.hook.completed
workflow.artifact.delta
```

事件 payload 必须至少携带：

```text
workflowRunId
workflowKey
status
sessionId
threadId?
turnId
taskId?
stepId?       step 事件必需
updatedAt
metadata.pluginWorkflow?
```

## 4. 状态机

current StepStatus 固定为：

| 状态 | 含义 | terminal |
| --- | --- | --- |
| `queued` | 已创建，等待前置条件或调度 | 否 |
| `running` | 正在执行 | 否 |
| `waiting` | 等待用户审批、外部输入或资源 | 否 |
| `retrying` | 正在准备重试 | 否 |
| `completed` | 成功完成 | 是 |
| `failed` | 失败且当前 attempt 结束 | 是 |
| `canceled` | 被用户或系统取消 | 是 |
| `skipped` | 被策略或用户跳过 | 是 |

旧状态映射：

| 旧状态 | 映射 |
| --- | --- |
| `active` | `running` |
| `pending` | `queued` |
| `succeeded` | `completed` |
| `success` | `completed` |
| `error` | `failed` |
| `cancelled` | `canceled` |
| `timeout` | `failed` + `reasonCode=timeout` |

## 5. Owner 边界

### 5.1 Definition owner

Plugin、Skill、内置业务模块可以声明 definition，但必须输出统一 schema。

允许：

1. 声明 steps、capability refs、expected output、hook policy。
2. 声明用户输入 schema 和产物类型。
3. 声明人审、重试和权限策略。

禁止：

1. 自己执行 AI turn。
2. 自己写 terminal status。
3. 自己伪造 evidence success。
4. 把 UI local state 当 run status。

### 5.2 Runtime owner

App Server RuntimeCore 是 run、step、event、cancel、retry、respond 的 owner。

### 5.3 Projection owner

Agent UI / Workspace / Plugin iframe 只读取 read model：

```text
WorkflowReadModel
  -> run list
  -> active run
  -> step timeline
  -> actions
  -> artifact / evidence refs
```

UI 读取只能调用 current read API，写操作只能调用 current control API，例如：

```text
workflow/read
workflow/cancel
workflow/retry
workflow/respond
```

## 6. Lime 模块映射

| 能力 | current 或目标落点 |
| --- | --- |
| Workflow domain owner | `lime-rs/crates/app-server/src/runtime/workflow/**` |
| Workflow definition / status / read model | `runtime/workflow/definition.rs`、`runtime/workflow/status.rs`、`runtime/workflow/read_model.rs` |
| Workflow read/control | `runtime/workflow/read.rs` 承接 `workflow/read`；`runtime/workflow/control.rs` 已承接 `workflow/cancel`、`workflow/retry -> agentSession/turn/start` source turn 级重提交，以及 `workflow/respond -> agentSession/action/respond` waiting action bridge |
| Workflow event helpers | `runtime/workflow/events.rs`，统一事件构造和 payload 校验 |
| Source mapping | `runtime/workflow/source_map.rs`，当前统一 Plugin manifest、Skill `workflow_steps`、image command builtin workflow、content factory orchestration 与 test-only DSL -> `WorkflowDefinition`；source 规则继续膨胀后再拆 `manifest_map.rs / skill_map.rs` |
| workflow event producer | `lime-rs/crates/app-server/src/runtime/plugin_worker_workflow.rs` |
| turn cancel 补偿事件 | `lime-rs/crates/app-server/src/runtime/plugin_worker_workflow_cancel.rs`，只服务 `agentSession/turn/cancel` adapter，不是 `workflow/cancel` current owner |
| worker retry 补偿事件 | `lime-rs/crates/app-server/src/runtime/plugin_worker_workflow_retry.rs`，只服务 Plugin worker retry adapter，不是 `workflow/retry` current owner |
| 图片 workflow audit | `lime-rs/crates/app-server/src/runtime_backend/image_command/mod.rs` |
| evidence workflow ids | `lime-rs/crates/app-server/src/runtime/workflow/read_model.rs` |
| App Server event stream mapping | `src/lib/api/agentRuntime/appServerEventStream.ts`，workflow event 当前投影为 diagnostics-only `runtime_status` refresh signal，用于触发 current read model 刷新 |
| renderer workflow control client | `src/lib/api/appServerClientMethods.ts`，暴露 `readWorkflow / cancelWorkflow / retryWorkflow / respondWorkflow`，只走 App Server current method |
| legacy thread read model 兼容类型 | `src/lib/api/agentRuntime/types.ts`，仅保留 `workflow / workflow_runs / workflow_steps` 旧形状兼容字段；current UI 不从 `agentSession/read` 读取 workflow facts |
| workflow facts helper | `src/components/agent/chat/workspace/workspaceArticleWorkspaceWorkflowFacts.ts`，Article Workspace current read model adapter；已兼容 `workflow/read` response / `WorkflowReadModel`，后续只补 presentation |
| Plugin iframe workflow projection | `src/features/plugin/runtime/workflowReadProjection.ts`、`capabilityDispatcher.ts` 与 `hostBridge.ts`，通过 `lime.agent.readWorkflow` / `capability:subscribe topic=workflow` 只读投影 App Server `workflow/read` |
| Plugin manifest definition source | `src/features/plugin/manifest/types.ts` |
| 前端 DSL adapter | `src/features/plugin/runtime/workflowRuntimeHost.ts` 已删除；只保留 catalog-only dead guard、负向断言和 source mapping test-only fixture |
| 旧 workspace Hook | `src/components/workspace/hooks/useWorkflow.ts` 已删除；只保留 catalog-only dead guard |

## 7. 代码目录分层

后端 current 目录：

```text
lime-rs/crates/app-server/src/runtime/workflow/
  mod.rs
  definition.rs
  status.rs
  events.rs
  read_model.rs
  read.rs
  source_map.rs
  control.rs
  tests.rs
```

模块职责：

| 模块 | 职责 | 禁止事项 |
| --- | --- | --- |
| `definition.rs` | 定义 `WorkflowDefinition`、`WorkflowStepDefinition`、source refs 和 schema version。 | 不读取 Plugin / Skill 具体结构。 |
| `status.rs` | 定义 current status 与旧状态映射。 | 不在 runtime 主链写旧状态。 |
| `events.rs` | 构造和校验 `workflow.*` RuntimeEvent payload。 | 不做 read model 聚合。 |
| `read_model.rs` | 从 event log 投影 `WorkflowReadModel`。 | 不从 UI message 或 tool 文本反推状态。 |
| `read.rs` | 实现 `workflow/read`，复用 session runtime events 与 workflow audit log 的同源 projector。 | 不复制第二套读取规则，不承担 cancel / retry / respond。 |
| `source_map.rs` | 将 Plugin manifest、Skill `workflow_steps`、image command builtin workflow、content factory orchestration 和 test-only DSL 映射到统一 definition。 | 不执行 workflow。 |
| `control.rs` | 校验并执行 cancel / retry / respond。 | 不绕过 policy / permission scope。 |

`manifest_map.rs / skill_map.rs` 只是 `source_map.rs` 后续膨胀时的拆分候选；当前仓库尚未创建这两个文件，不应在实现或测试中引用它们。

协议和前端投影落点：

```text
lime-rs/crates/app-server-protocol/src/protocol/v0/workflow.rs
packages/app-server-client/src/generated/protocol-types.ts
src/lib/api/agentRuntime/types.ts
src/lib/api/agentRuntime/appServerReadModelProjection.ts
src/components/agent/chat/workspace/workspaceArticleWorkspaceWorkflowFacts.ts
```

旧文件处理：

1. `plugin_worker_workflow*.rs` 先作为 compat source adapter，逐步委托 `runtime/workflow/**`。
2. `workspaceArticleWorkspaceWorkflowFacts.ts` 只保留 Workflow Read Model 到 Article Workspace 的业务投影，不再拥有独立 workflow facts 规则。
3. `WorkflowRuntimeHost` 不得恢复为 test-only / compat adapter 文件；需要旧 DSL 语义时只读历史 fixture 或 source mapping test-only case。
4. `useWorkflow` 不再恢复旧主题步骤；旧 Hook 已删除，只保留治理 dead path。

## 8. Read Model 要求

新增或收敛后的 `WorkflowReadModel` 至少包含：

```text
threadId
activeWorkflowRunId?
workflowRuns[]
workflowSteps[]
actions[]
updatedAt
```

每个 run：

```text
workflowRunId
workflowKey
title
status
taskId
turnId
retry?
appId?
sourceKind
startedAt
updatedAt
finishedAt?
stepCounts
artifactRefs
evidenceRefs
failure?
```

每个 step：

```text
workflowRunId
stepId
title
kind
status
attempt
index
stepCount
retry?
progressMessage?
toolCallIds
artifactRefs
evidenceRefs
failure?
```

## 9. Contract Guard

后续实现必须补的守卫：

1. 禁止恢复 `WorkflowRuntimeHost` 前端 DSL runtime，生产代码也不得直接运行 AI workflow。
2. 禁止恢复或重新引用 `useWorkflow` 旧 Hook。
3. 检查 `workflow.run.* / workflow.step.*` terminal pairing。
4. 检查 App Server read model 不再删除 workflow facts。
5. 检查 manifest workflow steps 到 `WorkflowDefinition` 的映射 fixture。
6. 检查 status mapping 只出现在 adapter 层。
7. 检查重构完成后旧实现已删除或降为 `dead / test-only`，不能长期保留 `deprecated` 壳。

## 10. 与相邻路线图关系

1. AgentRuntime 负责 session/thread/turn/task/tool/action/evidence 主链。
2. App Server 负责 JSON-RPC、RuntimeCore 和 durable store。
3. Plugin 负责 package、manifest、UI runtime 和 Host capability。
4. Workflow 是 AgentRuntime surface 内的一类 orchestration facts，不是 Plugin 或 GUI 的第二套 runtime。
