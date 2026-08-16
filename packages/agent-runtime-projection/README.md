# @limecloud/agent-runtime-projection

`@limecloud/agent-runtime-projection` 是 Lime Cloud 系列应用共享的 Agent Runtime 事件投影层。它把 App Server / JSON-RPC / 本地运行时产生的 execution events 转成前端稳定可渲染的 read model，并复用 `@limecloud/agent-ui-contracts` 中的标准类型契约。

这个包只包含纯 TypeScript 类型和函数，不依赖 React，不调用 JSON-RPC，不访问 Electron IPC，也不绑定任何具体业务模块。

当前物理包名是 `@limecloud/agent-runtime-projection`。历史规划中出现过的 `@limecloud/agent-ui-projection` 只能作为未来 alias 或文档历史引用，不能写成 current owner。

## Standard Package Layering

Agent UI / Runtime 的标准链路由四个包共同组成：

```text
@limecloud/agent-runtime-client
  -> executionEvents
  -> @limecloud/agent-ui-contracts
  -> @limecloud/agent-runtime-projection
  -> @limecloud/agent-runtime-ui 或宿主 presentation adapter
```

各层边界：

- `@limecloud/agent-runtime-client` 只封装 App Server current runtime facade，不新增协议。
- `@limecloud/agent-ui-contracts` 只定义跨宿主类型契约，不承接投影逻辑。
- `@limecloud/agent-runtime-projection` 只做 headless projection 和 selectors，不接触宿主、网络、React 或业务对象。
- `@limecloud/agent-runtime-ui` 只消费 projection state 渲染标准 React primitives，不订阅运行时。

宿主负责把 App Server 事件适配为标准 execution events，再按 `AgentRuntimeClient -> projectAgentUiState -> AgentUiProjectionView` 或自己的 presentation adapter 渲染；如果需要展开细节，中间事件链路是 `AgentRuntimeClient -> executionEvents -> projectAgentUiState -> AgentUiProjectionView`。宿主可以保留本地化、点击路由、业务卡片和页面布局；不得重新实现 scope 匹配、latest index、summary selector 或运行时事实解释。

## Source Layout

实现必须按职责拆分，`src/index.ts`、`src/index.js` 和 `src/index.d.ts` 只能做 barrel exports only：

```text
src/actions.ts     -> HITL action.required / action.resolved projection helpers
src/appServerFacts.ts -> App Server thread/read 与 lifecycle facts -> execution events replay
src/artifactEvents.ts -> artifact_snapshot projection helpers
src/collaborationFacts.ts -> collaboration payload facts / Soul style metadata normalizer
src/contextEvents.ts -> context_trace / turn_context projection helpers
src/conversationEvents.ts -> message / text delta / reasoning delta projection helpers
src/contracts.ts   -> contracts 类型转导
src/diagnosticEvents.ts -> warning / cost metric projection helpers
src/envelope.ts    -> Agent UI event base fields / runtime entity envelope / sequence helper
src/eventStore.ts  -> event store / scope / latest selectors
src/hydrationEvents.ts -> historical hydration projection helpers
src/lifecycle.ts   -> session.opened / run lifecycle / runtime status / model routing / task profile projection helpers
src/normalization.ts -> 字段读取、文本预览、ID 列表与 payload compact
src/planApproval.ts -> plan approval metadata parser / action projection builders
src/permissionEvents.ts -> runtime_status permission.changed projection helpers
src/refs.ts        -> artifact refs 等跨宿主引用提取
src/routing.ts     -> routing run.status / decision payload projection helpers
src/runtimeFacts.ts  -> runtime entity/status/phase/topology 标准事实解释
src/summary.ts     -> summary / surface / lane selectors
src/subagents.ts  -> Subagents 标准模型 selector
src/threadItems.ts -> thread item -> AgentUiProjectionEvent builders / action item builders / subagent activity builders / task owner metadata parser
src/toolEvents.ts  -> tool_start / tool_end / tool_progress / tool delta -> AgentUiProjectionEvent builders
src/readModel.ts   -> execution events -> read model
src/uiState.ts     -> execution events -> AgentUiProjectionState / projector
src/index.ts       -> barrel exports only
```

## Installation

```bash
npm install @limecloud/agent-ui-contracts @limecloud/agent-runtime-projection
```

## Boundary

这个包负责：

- 识别事件属于 action、evidence、artifact、tool、permission、context、runtime status 还是 message。
- 构建历史恢复 / 快照 hydration 的标准 session、message snapshot 与 stale diagnostic 事件。
- 把 `action.required` 投影成可点击的人类待办。
- 用 `action.resolved` 标记已处理 action。
- 聚合 artifact refs、evidence refs、task refs、source count。
- 给 UI 提供 `UIMessageParts`、`ProcessTimeline`、`ExecutionGraph`、`actions`、`tools` 和 `readModel` 等标准渲染对象。
- 给 UI 提供 `subagents` 标准模型，统一表达子代理线程、委派调用、活动摘要、隔离摘要和协作 facts，避免 React 组件重新解释子代理事实。
- 为 `AgentUiProjectionEvent` 提供 host-neutral 的索引、scope selector 和 latest selector，宿主 store 只负责持久化和订阅。
- 为 `AgentUiProjectionEvent` 提供 host-neutral 的 summary selector，包括 action / task / artifact / evidence / diagnostics 计数、notable latest events、Subagents surface 聚合和 artifact latest lookup。
- 为 App Server `thread/read` 与 lifecycle facts 提供标准 execution events replay adapter。
- 为宿主事件 adapter 提供 host-neutral 的 action event builder、artifact snapshot builder、context trace / turn context builder、conversation event builder、diagnostic / cost metric builder、historical hydration builder、plan approval metadata parser / builder、runtime permission builder、queue event builder、routing status builder、runtime lifecycle / model routing / task profile builder、thread item builder、thread item action builder、subagent activity / worker notification builder、tool lifecycle builder、TaskUpdate owner metadata parser、event base 字段、sequence 编排、字段规整、artifact refs 提取、routing decision payload、runtime entity/status/phase/topology 和 worker usage 解释函数。

这个包不负责：

- 发起或订阅 JSON-RPC。
- 调用模型、工具或 Electron bridge。
- 渲染 React UI。
- 持久化 session 或执行状态。
- 定义某个业务 App 的 Prompt、知识库、素材、审核流程。

## Usage

优先按输入来源选择入口：

| 输入来源 | 推荐入口 | 说明 |
| --- | --- | --- |
| 已有 normalized `AgentRuntimeExecutionEvent[]` | `projectAgentUiState` | 新 UI 默认入口，输出 message / timeline / graph / action / read model。 |
| 只需要旧事实栏 | `projectAgentRuntimeReadModel` | 兼容入口，不适合作为新 surface 的唯一状态。 |
| 既有产品已有 transcript messages + read model | `projectAgentUiStateFromSessionSnapshot` | 迁移期入口，把旧消息转成 `UIMessageParts`，同时复用 read model source events 生成 timeline / graph / refs / actions。 |
| App Server `thread/read` + lifecycle facts | `replayAppServerFacts` | 不创建 client，只把已取得 facts replay 成标准 projection。 |
| contracts fixture | `replayAgentUiFixture` | 下游包和产品接入测试使用。 |
| 宿主私有事件 shape | `buildAgentUi*Event` helpers | 在 adapter 边界转成标准 projection event。 |

```ts
import {
  projectAgentUiState,
  projectAgentRuntimeReadModel,
  type AgentRuntimeExecutionEvent,
} from "@limecloud/agent-runtime-projection";

const events: AgentRuntimeExecutionEvent[] = [
  {
    id: "evt-action",
    kind: "action",
    status: "pending",
    eventClass: "action.required",
    title: "需要补充输入源",
    actionId: "action-1",
    payload: {
      actionKind: "add-input-source",
      targetModule: "knowledge-inputs",
    },
    createdAt: new Date().toISOString(),
  },
];

const readModel = projectAgentRuntimeReadModel({
  executionEvents: events,
  sourceCount: 0,
});

console.log(readModel.pendingActions[0]?.action?.decision);

const uiState = projectAgentUiState({
  executionEvents: events,
  sourceCount: 0,
});

console.log(uiState.messages, uiState.timeline, uiState.graph);
```

最小产品接入通常长这样：

```ts
import {
  projectAgentUiState,
  replayAppServerFacts,
} from "@limecloud/agent-runtime-projection";

const replay = replayAppServerFacts({
  readModel: await runtimeClient.readThread({ sessionId }),
  events: drainedLifecycleEvents,
});

const state = projectAgentUiState({
  executionEvents: replay.events,
  sourceCount: replay.state.readModel.sourceCount,
});
```

已有产品如果已经把消息 transcript 和 runtime read model 分开维护，不要在页面里继续拼 `AgentTimeline + RuntimeFactsPanel` 两套 surface。先在 adapter 边界合成标准 state，再交给 `AgentUiProjectionView`：

```ts
import {
  projectAgentUiStateFromSessionSnapshot,
} from "@limecloud/agent-runtime-projection";

const state = projectAgentUiStateFromSessionSnapshot({
  messages: session.messages.map((message) => ({
    id: message.id,
    role: message.role,
    content: message.content,
    createdAt: message.createdAt,
  })),
  readModel: projectAgentRuntimeReadModel({
    executionEvents: session.executionEvents,
    sourceCount: session.sourceCount,
  }),
});
```

如果宿主已经维护自己的 event store，应把 store 中的可见事件交给 selector，而不是在 UI 组件里重新筛选：

```ts
const visibleEvents = selectAgentUiProjectionEventsForScopeFromStore(store, {
  sessionId,
  turnId,
});
const summary = summarizeAgentUiProjectionEvents(visibleEvents);
```

## Event Shape

`AgentRuntimeExecutionEvent` 是跨应用的最小公共事件形状。业务应用可以使用自己的 session 类型，只要 execution event 字段兼容即可：

```ts
interface AppSession {
  executionEvents?: AgentRuntimeExecutionEvent[];
  sourceCount?: number;
}
```

常用事件类型：

- `action.required`
- `action.resolved`
- `evidence.changed`
- `artifact.changed`
- `tool.started`
- `tool.result`
- `tool.failed`
- `model.completed`
- `model.failed`
- `snapshot.updated`

## Standard UI Projection

`projectAgentUiState` 是前端标准入口。它把同一组 execution events 投影成：

- `messages`: 面向消息流的 `UIMessageParts`。
- `timeline`: 面向过程组件的 `ProcessTimeline`。
- `graph`: 面向任务、run、step、tool 和 action 的 `ExecutionGraph`。
- `tools`: 工具事件组。
- `actions`: 待用户处理的 action。
- `subagents`: 面向 subagent / handoff / review 的标准模型，包括 `threads`、`delegationCalls`、`activities`、`activeThreadIds`、`completedThreadIds`、`failedThreadIds` 和轻量 `isolation` 摘要。
- `readModel`: 兼容既有事实栏的 read model。
- `runtime`: 当前运行状态。
- `hydration`: 宿主应用判断快照 / 实时流状态的水位信息。

`projectAgentRuntimeReadModel` 仍然保留，用于只需要事实栏或旧组件的宿主；新 UI 默认使用 `projectAgentUiState`。如果宿主正在从旧 transcript UI 迁移，使用 `projectAgentUiStateFromSessionSnapshot` 作为过渡入口；它只接受消息快照和 read model，不读取 Provider、不生成 runtime truth，也不替代 App Server facts。

`AgentUiProjectionState` 输出字段的使用边界：

- `messages`: 交给 `UIMessagePartsView` 或宿主消息流。
- `timeline`: 交给 `ProcessTimelineView`，表达工具、action、artifact、diagnostic 的线性过程。
- `graph`: 交给 `ExecutionGraphView`，表达 run / turn / task / tool / subagent 的结构关系。
- `actions`: 交给 `ActionRequiredList`，只表达可响应的人类操作。
- `subagents`: 交给 `SubagentsView` 或宿主自己的子代理 presentation adapter；业务组件不能从 assistant 正文或局部 UI state 推断 subagent / handoff / review 状态，也不能从协作标题或中文状态反推 `collaborationFacts` / Soul style metadata。
- `runtime`: 顶部或侧栏运行状态，不作为 Provider 状态事实源。
- `hydration`: 宿主判断 snapshot / realtime stream 是否 stale，不用于业务结果判断。

## Projection Lifecycle

Projection 的生命周期固定为 headless state machine，不绑定 React：

```text
hydrate(read model / fixture / App Server facts)
  -> apply runtime events by id + sequence
  -> derive messages / timeline / graph / actions / tools
  -> derive subagents from runtime events
  -> expose hydration status
```

标准行为：

| Step | Rule |
| --- | --- |
| Hydrate | `hydrate(input)` 用新的 event 集合重建 state，不复用 React local state。 |
| Apply | `apply(event)` 按 event id 幂等追加，重复事件不改变 state。 |
| Sequence | projection 按输入顺序保留 timeline，fixture validation 负责发现非法 sequence gap。 |
| Stream merge | `model.delta` / `reasoning.*` 合并为稳定 `UIMessagePart`，final text 不重复追加。 |
| Snapshot repair | snapshot / read model 只能修复可见状态，不能让 UI 猜测 tool/action/artifact 终态。 |
| State delta | `state.delta` 按 RFC 6902 patch 修复 projection / readModel 子树；batch 与 incremental apply 等价，失败进入 stale diagnostics，后续同子树 runtime facts 优先。 |
| Subagents model | `subagents` 由 projection 构建；React 和产品应用不能再私有过滤 graph/read model；thread / delegation / activity 的协作上下文通过 `collaboration` view 携带。 |
| Reset | `reset()` 清空 events、read model 和 ephemeral UI。 |

当前 `hydration.status` 最小支持 `idle` 和 `live`；`stale`、`repairing`、`degraded` 是 contracts 预留的标准状态，必须由 runtime/read model repair 事实驱动，不能由 UI timeout 猜测。

## Selectors And State API

推荐按用途选择 API：

| API | 用途 |
| --- | --- |
| `projectAgentUiState` | 一次性把 events 投影成完整 UI state。 |
| `createAgentUiProjector` | 持有 headless projector，适合 event stream apply。 |
| `projectAgentRuntimeReadModel` | 兼容事实栏或旧 surface。 |
| `buildAgentUiSubagentsModel` | 从 execution events 构建标准 Subagents 模型，并把 `collaborationFacts`、`collaborationSurface`、`collaborationPhase`、`styleLevel`、`riskLevel`、`profileId`、`packId`、`toneVariant` 归一到 thread / delegation / activity 的 `collaboration` view。 |
| `indexAgentUiProjectionEvents` | 为宿主外部 event store 建 latest / scope index。 |
| `selectAgentUiProjectionEventsForScopeFromStore` | 按 session/thread/run/turn/task scope 取可见事件。 |
| `summarizeAgentUiProjectionEvents` | 生成 host-neutral summary，不进 React。 |

这些 API 都是纯函数或纯 TypeScript 对象，不会发起 JSON-RPC，不会访问 Provider，也不会读写业务 store。

## Event Store Selectors

宿主可以保留自己的 external store，但事件筛选、索引和 summary 聚合应复用本包：

```ts
import {
  createEmptyAgentUiProjectionEventStoreState,
  indexAgentUiProjectionEvents,
  selectAgentUiProjectionEventsForScopeFromStore,
  summarizeAgentUiProjectionEvents,
  type AgentUiProjectionEvent,
} from "@limecloud/agent-runtime-projection";

let state = createEmptyAgentUiProjectionEventStoreState();
const events: AgentUiProjectionEvent[] = [];

state = {
  events,
  ...indexAgentUiProjectionEvents(events),
};

const visible = selectAgentUiProjectionEventsForScopeFromStore(state, {
  sessionId: "session-1",
});
const summary = summarizeAgentUiProjectionEvents(visible);
```

不要在业务宿主里重新实现 `session / thread / run / turn / task` scope 匹配、latest event 索引、artifact / evidence / action lookup、事件类型分组、Subagents 聚合或 notable event summary。否则不同 App 会逐步长出不同的 Agent UI 解释口径。

也不要在业务宿主里重新实现 `buildAgentUiActionRequiredEvent`、`buildAgentUiActionResolvedEvent`、`buildAgentUiArtifactSnapshotEvent`、`buildAgentUiContextTraceEvent`、`buildAgentUiTurnContextEvents`、`buildAgentUiMessageSnapshotEvent`、`buildAgentUiTextDeltaEvent`、`buildAgentUiReasoningDeltaEvent`、`buildAgentUiHistoricalHydrationEvents`、`buildAgentUiPlanApprovalRequiredEvent`、`buildAgentUiPlanApprovalResolvedEvent`、`extractAgentUiPlanApprovalProjection`、`extractAgentUiPlanApprovalResponseProjection`、`buildAgentUiRuntimePermissionChangedEvent`、`buildAgentUiRoutingStatusEvent`、`buildAgentUiThreadStartedEvent`、`buildAgentUiRunStartedEvent`、`buildAgentUiRunFinishedEvent`、`buildAgentUiRunFailedEvent`、`buildAgentUiRuntimeStatusEvent`、`buildAgentUiModelChangeEvent`、`buildAgentUiTaskProfileResolvedEvent`、`buildAgentUiCollaborationPayloadMetadata`、`buildAgentUiThreadItemEvent`、`buildAgentUiThreadItemActionEvent`、`buildAgentUiThreadItemSubagentActivityEvent`、`buildAgentUiThreadItemBase`、`buildAgentUiToolStartEvents`、`buildAgentUiToolEndEvent`、`buildAgentUiToolEndEvents`、`buildAgentUiToolProgressEvent`、`buildAgentUiToolOutputDeltaEvent`、`buildAgentUiToolInputDeltaEvent`、`extractAgentUiTaskOwnerChangeProjection`、`buildAgentUiProjectionBase`、`sequenceAgentUiProjectionEvents`、`definedString`、`truncateText`、`metadataKeys`、`extractArtifactRefs`、`buildRoutingDecisionPayload`、runtime entity/status/phase/topology 这类通用事实解释。宿主 adapter 只把自己的事件 shape 映射到标准 projection event，通用规整与事实解释必须回到本包。

主聊天、Plugin、content-studio 等宿主可以继续保留本地化 label、文案格式化、点击行为和业务卡片，但这些属于 presentation adapter，不是 runtime projection 标准事实源。

## Backend Integration

后端交互应留在宿主应用内：

```text
App Server JSON-RPC
  -> app service/store
  -> executionEvents
  -> @limecloud/agent-runtime-projection
  -> @limecloud/agent-runtime-ui 或宿主 UI
```

不要让 UI 组件直接调用 JSON-RPC。这样主聊天、工作台和后续独立 App 可以共享同一套投影逻辑，同时保留自己的后端连接方式。

如果宿主已经拿到 App Server current facts，可以直接用纯 adapter 进入标准 projection：

```ts
import { replayAppServerFacts } from "@limecloud/agent-runtime-projection";

const result = replayAppServerFacts({
  readModel: sessionReadResult,
  events: drainedLifecycleEvents,
});

console.log(result.state.messages, result.state.timeline, result.state.graph);
```

`replayAppServerFacts` 只消费已取得的结构化 facts，不创建 `AppServerClient`，不订阅 JSON-RPC，也不直读数据库。`waitingAction` 会被表达为 `action.required` + waiting runtime state；artifact 和 evidence 只通过 refs 暴露大输出，避免 UI 从 assistant 正文或 provider 原始响应猜测事实状态。

`replayAppServerFacts` 的输入建议：

- `readModel`: `AgentRuntimeClient.readThread(...)` 或 App Server `thread/read` 的 result。
- `events`: App Server lifecycle notification 中的标准 execution facts 列表。

不要把 JSON-RPC envelope、provider stream chunk 或 assistant Markdown 原文直接传给 projection；这些应该先在 runtime / host adapter 边界规整成 facts。

宿主应用如果把 App Server 的 `artifact.snapshot` 物化成本地业务产物，例如 Prompt 草稿、文章版本或素材交付物，应只把最终业务产物对应的 `artifact.changed` 传给投影层。上游快照可以保留在 provider 原始事件或诊断 payload 中，不要再作为第二个可见 artifact 事件传入，否则 UI 会把“中间快照”和“最终业务产物”渲染成两个并列产物。

共享投影层不会按 `artifactId` 或 `turnId` 自动合并多条 artifact 事件，因为多产物会话是合法场景。是否折叠、隐藏中间快照或提升最终产物，由宿主的业务适配层决定。

## Fixture Replay

下游包和产品接入测试应复用 contracts fixtures：

```ts
import { getAgentUiFixture } from "@limecloud/agent-ui-contracts";
import { replayAgentUiFixture } from "@limecloud/agent-runtime-projection";

const fixture = getAgentUiFixture("subagent-handoff");
const replay = replayAgentUiFixture(fixture);

expect(replay.state.graph.length).toBeGreaterThan(0);
expect(replay.state.subagents.threads.length).toBeGreaterThan(0);
expect(replay.state.subagents.delegationCalls.length).toBeGreaterThan(0);
```

fixture replay 的目的不是模拟后端，而是证明同一份标准 facts 在 projection 和 UI 层得到一致解释。

## Host Adapter Pattern

宿主私有事件应只在 adapter 边界被翻译一次：

```ts
import { buildAgentUiToolStartEvents } from "@limecloud/agent-runtime-projection";

export function fromHostToolStarted(event: HostToolStarted) {
  return buildAgentUiToolStartEvents({
    id: event.id,
    sequence: event.sequence,
    sessionId: event.sessionId,
    threadId: event.threadId,
    runId: event.turnId,
    toolCallId: event.callId,
    toolName: event.name,
    createdAt: event.timestamp,
  });
}
```

adapter 可以读取宿主字段，但不能在组件里继续解释 tool lifecycle、subagent lineage、handoff status 或 action decision。

## Development

```bash
npm --prefix packages/agent-runtime-projection run build
npm --prefix packages/agent-runtime-projection run test
npm --prefix packages/agent-runtime-projection pack --dry-run
```

## Do Not

- 不要在本包里创建 `AppServerClient`、调用 renderer bridge helper、订阅 Electron IPC 或发 HTTP。
- 不要把 Provider stream chunk、API key、HTTP response、DB row 直接暴露给 UI。
- 不要在产品应用里复制 `projectAgentUiState`、scope selector、summary selector 或 `runtimeFacts` 解释逻辑。
- 不要恢复旧树形过程术语作为 projection 标准；标准输出是 `UIMessageParts`、`ProcessTimeline`、`ExecutionGraph`。
- 不要把业务 artifact 折叠策略写进共享 projection；业务物化和展示优先级由宿主 adapter 决定。

## Publish

首次发布到 npmjs：

```bash
npm --prefix packages/agent-runtime-projection publish --access public
```

发布前必须先确认版本号、npm 账号和 `npm pack --dry-run` 输出。
