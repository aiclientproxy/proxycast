# @limecloud/agent-runtime-ui

`@limecloud/agent-runtime-ui` 是 Lime Cloud 系列应用共享的 Agent Runtime React UI primitives。它面向标准 `AgentUiProjectionState`，提供消息部件、过程时间线、执行图、运行事实栏、action / evidence / artifact 卡片等组件。

这个包是 UI 渲染层，不直接调用 JSON-RPC，不依赖 Electron，不绑定任何宿主应用的业务 store。宿主应用负责后端交互、session 持久化、业务对象和页面布局。

默认 label 只作为英文 fallback。宿主应用必须通过 `labels` / formatter props 注入自己的 aria、标题、按钮和状态文案，并负责 Lime current 五语言资源覆盖。

## Installation

```bash
npm install @limecloud/agent-ui-contracts @limecloud/agent-runtime-ui @limecloud/agent-runtime-projection
```

React 是 peer dependency：

```json
{
  "react": ">=18 <20",
  "react-dom": ">=18 <20"
}
```

## Boundary

这个包负责：

- 渲染 `AgentUiProjectionState` 标准 UI surface。
- 渲染 `UIMessageParts` 消息部件。
- 渲染 `ProcessTimeline` 过程时间线。
- 渲染 `ExecutionGraph` 执行图。
- 渲染 `AgentWorkbenchSurface` / `AgentWorkbenchTaskCard` 标准工作台 shell。
- 渲染 `state.subagents` 标准子代理模型。
- 渲染 Agent 消息时间线。
- 渲染运行事实摘要。
- 渲染 action、evidence、artifact、tool 等 runtime facts。
- 渲染 `ArtifactRef` / `EvidenceRef` 轻量引用列表，并把打开意图交还宿主应用。
- 渲染标准多 action controls，并通过 `data-action-decision` 暴露稳定 DOM contract。
- 通过 callback 把用户处理 action 的意图交还给宿主应用。

这个包不负责：

- 订阅 JSON-RPC 或调用 App Server。
- 管理 session store。
- 打开业务页面或路由。
- 提供全局 CSS reset、主题系统或业务文案体系。
- 渲染 Prompt 草稿、知识库、素材、审核任务等宿主业务壳。

## Usage

最推荐的入口是 `AgentUiProjectionView`。宿主先用 projection 包得到 `AgentUiProjectionState`，再把 state 传给 UI 包：

```tsx
import { AgentUiProjectionView } from "@limecloud/agent-runtime-ui";
import { projectAgentUiState } from "@limecloud/agent-runtime-projection";

const state = projectAgentUiState({
  executionEvents: session.executionEvents,
  sourceCount: session.sourceSnapshots.length,
});

export function AgentPanel() {
  return (
    <AgentUiProjectionView
      state={state}
      onResolveAction={(event, action) => {
        // 宿主应用决定打开哪个业务模块或发送哪个 action response。
        console.log(event.id, action.decision);
      }}
      labels={{
        messagePartsAriaLabel: t("agent.messages"),
        processTimelineAriaLabel: t("agent.timeline"),
        actionRequiredAriaLabel: t("agent.actionRequired"),
        actionButtonLabel: (action) => t(`agent.action.${action.decision}`),
      }}
    />
  );
}
```

如果宿主已经通过 App Server facts replay 得到 state，也可以直接渲染：

```tsx
import { AgentUiProjectionView } from "@limecloud/agent-runtime-ui";
import { replayAppServerFacts } from "@limecloud/agent-runtime-projection";

const replay = replayAppServerFacts({
  readModel,
  events,
});

<AgentUiProjectionView
  state={replay.state}
  onResolveAction={(event, action) => {
    actionResponder.respond(event, action);
  }}
/>;
```

`onResolveAction` 只表达用户意图。真正的后端调用应回到宿主的 runtime client，例如 `AgentRuntimeClient.respondAction(...)`。

## React Surface Model

这个包采用 controlled component 模式：宿主传入 `AgentUiProjectionState` 和 command callbacks，组件只负责渲染、稳定 DOM contract 和用户意图回调。

| Surface | Standard input | Owner of business behavior |
| --- | --- | --- |
| Message parts | `state.messages` | 宿主提供消息列表区域布局和本地化 label。 |
| Process timeline | `state.timeline` | projection 负责排序和 entry kind；宿主负责样式。 |
| Execution graph | `state.graph` | projection 负责 parent/child edge；宿主负责布局密度。 |
| Action required | `state.actions` | 宿主通过 runtime client 提交 action response。 |
| Tool group | `state.tools` | projection 负责 tool lifecycle 解释。 |
| Tool calls | `state.toolCalls` | projection 负责工具调用、事件分组与 MCP 归并。 |
| MCP surface | `state.mcp` | projection 负责 MCP server / tool 归类与状态汇总。 |
| Subagents | `state.subagents` | projection 负责子代理线程、委派调用、活动摘要、隔离摘要和协作 facts。 |
| Artifact refs | `state.artifacts` | 宿主负责打开 artifact workspace 或详情页。 |
| Evidence refs | `state.evidence` | 宿主负责打开 evidence pack、review 或 replay。 |
| Runtime summary | `state.readModel` | projection/read model 负责计数；宿主负责显示文案。 |

当前包不提供 hooks。`useAgentRuntimeClient`、`useAgentRuntimeFacts`、`useAgentUiProjector` 等 hook 应留在产品应用或后续单独 adapter 包里；共享 React primitives 不订阅 runtime、不持有 session store、不读取 App Server。

## Components

| 组件 | 输入 | 用途 |
| --- | --- | --- |
| `AgentUiProjectionView` | `AgentUiProjectionState` | 标准组合入口，渲染消息、过程、事实栏、action 和 graph。 |
| `AgentWorkbenchSurface` | workbench view/state | 标准工作台 shell，渲染 task card、消息、运行事实、artifact 和 composer slot。 |
| `AgentWorkbenchTaskCard` | workbench task view | 标准任务胶囊，渲染当前任务、状态、检查点与事实计数。 |
| `UIMessagePartsView` | `UIMessageParts` | 渲染标准消息部件。 |
| `ProcessTimelineView` | `ProcessTimeline` | 渲染线性执行过程。 |
| `ExecutionGraphView` | `ExecutionGraph` | 渲染 run / task / tool / subagent 结构。 |
| `SubagentsView` | `AgentUiProjectionState` | 从 `state.subagents` 渲染子代理标准面。 |
| `SubagentThreadList` | `AgentUiSubagentThreadView[]` | 渲染子代理线程、身份、状态、refs 摘要。 |
| `SubagentDelegationList` | `AgentUiSubagentDelegationView[]` | 渲染 spawn / handoff / wait / interrupt 等委派调用。 |
| `SubagentActivityList` | `AgentUiSubagentActivityView[]` | 渲染 started / handoff / review / completed / failed 等活动摘要。 |
| `ArtifactRefList` | `AgentUiArtifactRefView[]` | 渲染 artifact refs，不读取大 payload。 |
| `EvidenceRefList` | `AgentUiEvidenceRefView[]` | 渲染 evidence refs，不读取证据内容。 |
| `AgentUiRefList` | `AgentUiRefView[]` | refs 通用 primitive。 |
| `RuntimeFactsPanel` | `AgentRuntimeReadModel` | 兼容事实栏入口。 |
| `RuntimeFactsSummary` | `AgentRuntimeReadModel` | 渲染 source / action / artifact / evidence 计数。 |
| `RuntimeEventList` | `AgentRuntimeEventProjection[]` | 渲染一般 runtime event 列表。 |
| `ToolGroup` | tool events | 渲染工具调用分组。 |
| `ToolCallSurface` | `state.toolCalls` | 渲染标准工具调用 surface。 |
| `ToolCallCard` | `AgentUiToolCallView` | 渲染单个工具调用卡片。 |
| `McpSurface` | `state.mcp` | 渲染 MCP server 与 MCP tools surface。 |
| `McpServerList` | `AgentUiMcpServerView[]` | 渲染 MCP server 汇总。 |
| `McpToolList` | `AgentUiMcpToolCallView[]` | 渲染 MCP tool 调用卡片。 |
| `ActionRequiredList` | action events | 渲染待处理 action。 |
| `RuntimeFactCard` | runtime event | 渲染单个 runtime fact。 |
| `ActionCard` / `EvidenceCard` / `ArtifactCard` | runtime event | 语义化 card alias，方便宿主样式分层。 |
| `AgentTimeline` | host message list | 兼容传统消息列表，不是新 projection 默认入口。 |

## Labels And I18n

默认 label 只是 fallback。Lime 产品应用必须传入自己的 i18n label 和 formatter：

```tsx
<AgentUiProjectionView
  state={state}
  labels={{
    messagePartsAriaLabel: t("agent.messages.aria"),
    processTimelineAriaLabel: t("agent.timeline.aria"),
    executionGraphAriaLabel: t("agent.graph.aria"),
    runtimeSummaryAriaLabel: t("agent.summary.aria"),
    actionRequiredAriaLabel: t("agent.actions.aria"),
    actionButtonLabel: (action) => t(`agent.action.${action.decision}`),
    eventStatusLabel: (event) => t(`agent.eventStatus.${event.status}`),
  }}
/>;
```

宿主需要覆盖 `zh-CN / zh-TW / en-US / ja-JP / ko-KR`，不要把本包 fallback 当成产品文案。

## Source Layout

实现必须按职责拆分，`src/index.ts` 只能做 barrel exports：

```text
src/types.ts           -> 公共 props / callback / message 类型
src/labels.ts          -> 默认 label、status 和 meta formatter
src/messages.tsx       -> AgentTimeline / UIMessagePartsView
src/processTimeline.tsx -> ProcessTimelineView
src/executionGraph.tsx -> ExecutionGraphView
src/refs.tsx           -> ArtifactRefList / EvidenceRefList / AgentUiRefList
src/runtimeFacts.tsx   -> RuntimeFactsPanel / RuntimeFactCard / action/tool lists
src/subagents.tsx      -> SubagentsView / SubagentThreadList / SubagentDelegationList / SubagentActivityList
src/projectionView.tsx -> AgentUiProjectionView 标准组合入口
src/index.ts           -> barrel exports only
```

新增 primitive 必须落在对应职责文件，或先新增职责明确的小模块；不得把实现重新合并回 `src/index.ts`。

## CSS Contract

组件只输出稳定 class names，不注入全局样式。宿主应用应按自己的设计语言提供样式。

核心 class names：

- `agent-turn`
- `agent-turn user`
- `agent-turn assistant`
- `agent-turn-avatar`
- `agent-turn-body`
- `agent-turn-head`
- `agent-turn-model`
- `agent-turn-details`
- `agent-message-parts`
- `agent-message-part`
- `agent-empty-session`
- `agent-runtime-event`
- `agent-runtime-summary`
- `agent-process-timeline`
- `agent-process-entry`
- `agent-execution-events`
- `agent-tool-group`
- `agent-tool-calls`
- `agent-tool-call`
- `agent-mcp-surface`
- `agent-mcp-servers`
- `agent-mcp-server`
- `agent-mcp-tools`
- `agent-mcp-tool`
- `agent-action-required-list`
- `agent-subagents`
- `agent-subagent-threads`
- `agent-subagent-thread`
- `agent-subagent-delegations`
- `agent-subagent-delegation`
- `agent-subagent-activities`
- `agent-subagent-activity`
- `agent-ref-list`
- `agent-ref-card`
- `agent-ref-action`
- `agent-artifact-refs`
- `agent-evidence-refs`
- `agent-execution-graph`
- `agent-execution-node`
- `agent-event-surface`
- `agent-event-action`
- `agent-event-actions`
- `agent-session-artifact`
- `agent-ui-projection` class name
- `agent-ui-main`
- `agent-ui-sidecar`

这些 class names 是包级 CSS contract。宿主应用可以用自己的设计系统实现样式，不应把样式文件或页面壳反向变成包依赖。

稳定 data attributes：

| Attribute | Surface | 用途 |
| --- | --- | --- |
| `data-runtime-status` | `AgentUiProjectionView` | 当前 runtime status。 |
| `data-hydration-status` | `AgentUiProjectionView` | hydration / stale / repair 状态。 |
| `data-action-decision` | action buttons | approve / reject / answer / retry 等 action intent。 |
| `data-node-id` / `data-node-type` / `data-parent-id` | graph | graph node 和 lineage。 |
| `data-subagent-count` / `data-delegation-count` / `data-activity-count` | Subagents | 子代理 smoke / snapshot 断言。 |
| `data-thread-id` / `data-subagent-id` / `data-parent-thread-id` | Subagents | 子代理线程身份和父子关系。 |
| `data-delegation-action` / `data-target-thread-ids` | Subagents | 委派调用类型和目标线程。 |
| `data-activity-kind` / `data-source-event-id` | Subagents | 活动分类和来源事件。 |
| `data-collaboration-facts` / `data-collaboration-surface` / `data-collaboration-phase` / `data-collaboration-kind` / `data-collaboration-source` | Subagents | 协作 facts、surface、phase、kind 和来源事件，供 smoke / snapshot / 样式消费。 |
| `data-soul-style-level` / `data-soul-risk-level` / `data-soul-tone-variant` / `data-soul-profile-id` / `data-soul-pack-id` | Subagents | 协作事件的 Soul style metadata；只作为 descriptor contract，不是 profile-specific 文案。 |
| `data-event-class` | runtime fact cards | `tool.*`、`handoff.*`、`review.*` 等事件族。 |
| `data-ref-kind` / `data-ref-id` / `data-source-event-id` | artifact / evidence refs | 引用类型、引用 id 和来源事件。 |

## Accessibility

默认 aria label 只是 fallback。产品应用必须通过 `labels` 注入本地化 aria label，并确保可点击 action 使用真实 `<button>`。组件不会把图标、颜色或状态文本作为唯一语义；状态同时通过 class / data attributes 暴露给宿主样式和测试。

## Backend Integration

推荐数据流：

```text
App Server JSON-RPC
  -> host app service/store
  -> executionEvents
  -> @limecloud/agent-ui-contracts
  -> @limecloud/agent-runtime-projection
  -> @limecloud/agent-runtime-ui
  -> host app action resolver
```

新页面默认使用 `projectAgentUiState(...) -> AgentUiProjectionView`。如果宿主只需要旧事实栏，可以继续使用 `projectAgentRuntimeReadModel(...) -> RuntimeFactsPanel`；这属于兼容入口，不是新增 surface 的默认路径。

`onResolveAction` 只返回用户意图，不直接处理后端：

```tsx
<RuntimeFactsPanel
  readModel={readModel}
  onResolveAction={(event, action) => {
    if (action.decision === "open-model-settings") {
      openSettings();
    }
  }}
/>
```

`RuntimeFactsPanel` 只渲染宿主传入的 read model，不会订阅 JSON-RPC，也不会判断某个 artifact 是否已经被业务层物化。宿主如果已经把 App Server `artifact.snapshot` 写成业务产物，应在进入 `@limecloud/agent-runtime-projection` 前去掉上游快照事件，只保留本地业务产物的 `artifact.changed`。这样右侧事实栏展示的是可继续编辑、可应用的交付物，而不是后端中间事件。

## Product App Pattern

产品应用通常这样接入：

```tsx
function ProductAgentPanel({ sessionId }: { sessionId: string }) {
  const runtime = useAgentRuntimeClient();
  const facts = useAgentRuntimeFacts(sessionId);
  const state = projectAgentUiState({
    executionEvents: facts.executionEvents,
    sourceCount: facts.sourceCount,
  });

  return (
    <AgentUiProjectionView
      state={state}
      artifact={<ProductArtifactWorkspace sessionId={sessionId} />}
      onResolveAction={(event, action) => {
        runtime.respondAction({
          sessionId,
          requestId: event.actionId ?? event.id,
          actionType: "ask_user",
          confirmed: action.decision !== "reject",
          response: action.decision,
        });
      }}
      labels={productAgentLabels}
    />
  );
}
```

这段模式里，UI 包只负责渲染和回调。`useAgentRuntimeClient`、`useAgentRuntimeFacts`、业务 artifact workspace、权限判断和路由都属于宿主。

Subagents 的接入也必须走标准 projection state：

```tsx
import { SubagentsView } from "@limecloud/agent-runtime-ui";

<SubagentsView
  state={state}
  labels={{
    subagentsAriaLabel: t("agent.subagents"),
    subagentThreadsAriaLabel: t("agent.subagents.threads"),
    subagentDelegationsAriaLabel: t("agent.subagents.delegations"),
    subagentActivitiesAriaLabel: t("agent.subagents.activities"),
  }}
/>;
```

`SubagentsView` 只读取 `state.subagents`。如果业务组件需要自定义子代理布局，也应消费同一个模型，而不是重新过滤 `state.graph` 或 `state.readModel.visibleEvents`。协作执行的 Soul metadata 必须来自 `state.subagents.*.collaboration`，不得从标题、中文状态或局部 UI state 反推。

## Testing

React surface 测试应使用 contracts fixture 和 projection replay，不要自己编私有状态：

```tsx
import { renderToStaticMarkup } from "react-dom/server";
import { getAgentUiFixture } from "@limecloud/agent-ui-contracts";
import { replayAgentUiFixture } from "@limecloud/agent-runtime-projection";
import { AgentUiProjectionView } from "@limecloud/agent-runtime-ui";

const replay = replayAgentUiFixture(getAgentUiFixture("hitl-action"));
const html = renderToStaticMarkup(
  <AgentUiProjectionView state={replay.state} />,
);

expect(html).toContain("data-action-decision");
```

组件测试只覆盖渲染、class / data attribute contract 和 callback 接线。复杂状态机、事件归并和 summary 归类应在 projection 包测试。

标准 fixture smoke 至少覆盖：

- `hitl-action` 渲染 `data-action-decision`。
- `tool-success` / `tool-failure` 渲染 ToolGroup。
- `artifact-evidence` 渲染 artifact / evidence refs。
- `subagent-handoff` 渲染 `SubagentsView`、threads、delegation calls、activities。

## Development

```bash
npm --prefix packages/agent-runtime-projection run build
npm --prefix packages/agent-runtime-ui run build
npm --prefix packages/agent-runtime-ui run test
npm --prefix packages/agent-runtime-ui pack --dry-run
```

## Package Metadata

| Item | Value |
| --- | --- |
| Runtime | Node `>=20`，ESM。 |
| Peer deps | `react >=18 <20`、`react-dom >=18 <20`。 |
| Dependencies | `@limecloud/agent-ui-contracts`。 |
| Dev dependency | `@limecloud/agent-runtime-projection` 仅用于 fixture replay 测试。 |
| Side effects | `false`，不注入全局 CSS。 |
| Public files | `dist`、`README.md`。 |
| License | `MIT`。 |

## Do Not

- 不要在本包里创建 runtime client、订阅 JSON-RPC、调用 Electron bridge 或访问 App Server。
- 不要在本包里拼 Provider 参数、读取 API key、解释 tool lifecycle 或合成 evidence。
- 不要把业务页面、workspace、Prompt 草稿、素材预览或审核流放进共享 UI 包。
- 不要在 React 组件里重写 projection reducer、scope selector、summary selector 或 runtime facts 解释。
- 不要让默认英文 fallback 进入 Lime 产品文案；产品应用必须注入本地化 label。

## Publish

首次发布到 npmjs：

```bash
npm --prefix packages/agent-runtime-ui publish --access public
```

发布前必须先发布同版本的 `@limecloud/agent-runtime-projection`，并确认 `npm pack --dry-run` 输出只包含 `dist` 和 `README.md`。
