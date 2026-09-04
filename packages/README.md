# Lime Packages

本目录是 Lime 对外复用的 TypeScript / React / CLI 包集合。其他 App 接入 Agent、App Server 或相关工作台能力时，先从本文件判断“该装哪些包、放在哪一层、哪些事情不能放进共享包”，再进入单包 `README.md` 查 API 细节。

这些包不是一个必须全量安装的 SDK。它们按职责拆成五层：

```text
App Server transport
  -> Agent Runtime client facade
  -> Agent UI contracts
  -> Agent Runtime projection
  -> Agent Runtime React UI

Product workbench control
  -> Agent workbench adapter
  -> Agent capability catalog

Task automation
  -> Lime CLI
```

共同原则：

- 运行时事实来自 App Server / Agent Runtime，不由 React UI、业务页面或 Product App 伪造。
- Product App 只负责 workspace、业务对象、页面路由、session 归属、权限入口和本地化文案。
- 共享包只提供协议、client、projection、UI primitives、capability catalog 或 headless workbench adapter。
- 生产构建应消费 npm registry 包；本地跨仓开发可以临时 alias 到 `packages/*/dist`。

## 先按 App 类型选

| 你的 App 类型                                              | 最小推荐组合                                                                                                                    | 为什么                                                                                             |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 独立 Electron / Node 宿主，需要自己启动 App Server sidecar | `@limecloud/app-server-client` + `@limecloud/agent-runtime-client`                                                              | 宿主拥有 sidecar 生命周期、JSON-RPC transport、`agentSession/*` 调用和事件分发。                   |
| 已有平台宿主 / bridge，只在 renderer 做业务 Agent 工作台   | `@limecloud/agent-runtime-client/sessionGateway` + `@limecloud/agent-workbench-adapter` + `@limecloud/agent-capability-catalog` | renderer 不应引入 sidecar / stdio；业务页只拼 intent、capability policy 和 turn payload。          |
| 只想把 runtime events 渲染成标准 Agent UI                  | `@limecloud/agent-runtime-projection` + `@limecloud/agent-runtime-ui`                                                           | projection 把 facts 变成 `AgentUiProjectionState`，UI 包只渲染 controlled state。                  |
| 只需要统一 capability / tool policy 名称                   | `@limecloud/agent-capability-catalog`                                                                                           | 把 `research`、`web_search`、`image_generation`、`pdf_extract` 等 alias 归一为稳定 capability id。 |
| 需要终端 Agent                                             | `@limecloud/lime`                                                                                                               | 使用交互式 TUI 或 `lime exec`，并与 Desktop 共享 App Server runtime。                              |

## 决策树

```text
1. 你的进程是否负责启动或管理 app-server sidecar？
   yes -> 用 @limecloud/app-server-client。
   no  -> 继续看 2。

2. 你是否已有 host bridge / App Server gateway，只想在 renderer 发起 turn？
   yes -> 用 @limecloud/agent-runtime-client/sessionGateway。
   no  -> 继续看 3。

3. 你是否已经拿到了 thread/read、direct v2 lifecycle notifications 或 executionEvents？
   yes -> 用 @limecloud/agent-runtime-projection。
   还要 React 标准组件 -> 再加 @limecloud/agent-runtime-ui。

4. 你的页面是否有 composer、quick intent、任务类型或 capability allowlist？
   yes -> 用 @limecloud/agent-workbench-adapter + @limecloud/agent-capability-catalog。

5. 你需要交互式或脚本化终端 Agent？
   yes -> 用 @limecloud/lime。
```

如果一个 App 同时满足多项，按层组合，不要把 transport、projection、React UI 和业务 intent 塞进同一个包。

## 包速查表

| 包                                    | 安装                                              | 主要入口                              | 典型 owner                                | 用在什么场景                                                                                                     | 不要用来做什么                                                                              | 关键导出                                                                                                                                                              |
| ------------------------------------- | ------------------------------------------------- | ------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `@limecloud/app-server-client`        | `npm install @limecloud/app-server-client`        | `@limecloud/app-server-client`        | Electron main / Node host / platform host | App Server JSON-RPC、stdio sidecar、manifest / sha256 / resources、`agentSession/*` transport、event router      | React UI、renderer bundle、业务 session store、provider key 策略                            | `AppServerClient`、`AppServerConnection`、`AppServerAgentEventRouter`、`startPackagedAppServerSidecar`、`createAgentRuntimeClient`                                    |
| `@limecloud/agent-runtime-client`     | `npm install @limecloud/agent-runtime-client`     | 根入口、`./sessionGateway`            | runtime gateway owner / renderer adapter  | 标准 Agent Runtime facade：`startTurn`、`readThread`、`respondAction`、`cancelTurn`、event subscription          | 新 JSON-RPC 协议、Electron IPC、projection、React、mock fallback、Lime-only evidence export | `createAgentRuntimeClient`、`createAgentRuntimeClientFromSessionGateway`、event pipeline / verifier                                                                   |
| `@limecloud/agent-ui-contracts`       | `npm install @limecloud/agent-ui-contracts`       | `@limecloud/agent-ui-contracts`       | contracts / adapter / tests               | 共享 Agent UI event、runtime read model、message、timeline、graph、Subagents、fixtures、validation 类型          | 投影逻辑、React 组件、App Server client                                                     | `AgentRuntimeExecutionEvent`、`AgentUiProjectionState`、`agentUiConformanceFixtures`、`validateRuntimeEvent`                                                          |
| `@limecloud/agent-runtime-projection` | `npm install @limecloud/agent-runtime-projection` | `@limecloud/agent-runtime-projection` | frontend adapter / store selector         | `executionEvents` -> messages、timeline、graph、actions、tools、artifacts、evidence、summary、Subagents          | transport、React 渲染、业务文案、session 持久化                                             | `projectAgentUiState`、`projectAgentRuntimeReadModel`、`replayAppServerFacts`、`projectAgentUiStateFromSessionSnapshot`                                               |
| `@limecloud/agent-runtime-ui`         | `npm install @limecloud/agent-runtime-ui`         | `@limecloud/agent-runtime-ui`         | React App presentation layer              | 渲染 `AgentUiProjectionState`、消息部件、过程时间线、执行图、action / artifact / evidence / subagents primitives | 调用 App Server、管理 store、打开业务页面、全局主题和产品文案                               | `AgentUiProjectionView`、`UIMessagePartsView`、`ProcessTimelineView`、`ExecutionGraphView`、`RuntimeFactsPanel`                                                       |
| `@limecloud/agent-capability-catalog` | `npm install @limecloud/agent-capability-catalog` | `@limecloud/agent-capability-catalog` | Product App / workbench adapter           | 稳定 capability id、alias 归一、metadata contract、tool policy、allowlist 校验                                   | 启动 turn、订阅 events、渲染工作台、定义业务 Prompt                                         | `resolveAgentCapabilityIds`、`buildAgentCapabilityPolicy`、`validateAgentCapabilities`、`AGENT_CAPABILITY_DEFINITIONS`                                                |
| `@limecloud/agent-workbench-adapter`  | `npm install @limecloud/agent-workbench-adapter`  | `@limecloud/agent-workbench-adapter`  | Product App workbench controller          | quick intent -> capability policy、composer submit mode、runtime facts summary、`lime.agent` turn payload 拼装   | React state、CSS、IPC、session store、provider key、业务文案                                | `DEFAULT_AGENT_WORKBENCH_INTENTS`、`resolveWorkbenchIntentCapabilityPolicy`、`resolveWorkbenchSubmitMode`、`summarizeAgentRuntimeFacts`、`buildAgentTurnStartPayload` |
| `@limecloud/lime`                     | `npm install -g @limecloud/lime`                  | `lime` binary                         | CLI / TUI host                            | `lime`、`lime exec`、`lime resume`、`lime thread`、`lime mcp`、`lime skills`                                     | App runtime、React UI、独立任务状态机                                                       | `lime` 命令                                                                                                                                                           |

## 标准 Runtime 边界

Agent runtime 的 current 主链只有一条：

```text
Product App business context
  -> AgentRuntimeClient / host bridge
  -> App Server agentSession/*
  -> RuntimeCore / provider store / tools
  -> direct v2 Thread/Turn/Item + thread/read
  -> projection
  -> UI
```

不要在 Product App、React UI 或 workbench adapter 中重建这些事实：

- provider key / token 保存；
- turn / run / task / tool / action 成功态；
- artifact / evidence 事实；
- runtime fallback；
- mock backend production fallback；
- 第二套 session / task protocol。

runtime 不可用时，调用方应 fail closed 或投影 blocked / unavailable facts。不要在 UI 中补造成功消息、artifact、evidence 或 action resolved。

## 常见接入配方

### 1. Electron main 拥有 App Server sidecar

适合独立桌面 App 或平台宿主。sidecar 生命周期留在 main / Node 层，renderer 只通过你自己的 bridge 消费事件和调用能力。

```ts
import {
  startPackagedAppServerSidecar,
  createAgentRuntimeClient,
} from "@limecloud/app-server-client";

const { connected, lifecycle } = await startPackagedAppServerSidecar(
  { clientInfo: { name: "product-app", version: app.getVersion() } },
  { resourcesPath: process.resourcesPath },
);

app.on("before-quit", () => void lifecycle.stop());

const runtime = createAgentRuntimeClient(connected.connection, {
  request: { timeoutMs: 120_000 },
});

runtime.subscribeLifecycleEvents((event) => {
  mainWindow.webContents.send("agent:lifecycle", event);
});

void (async () => {
  while (!mainWindow.isDestroyed()) {
    await runtime.nextEvent();
  }
})();
```

这个配方只说明包边界。App 自己仍要负责 IPC 白名单、窗口生命周期、workspace id、provider store 准备和错误提示。

### 2. Renderer 已有 host gateway，只需要标准 runtime facade

适合 Content Studio 这类 Product App：平台宿主已经提供 `lime.agent` 或 App Server gateway，renderer 不应把 `@limecloud/app-server-client` 的 stdio / sidecar 代码打进前端包。

```ts
import { createAgentRuntimeClientFromSessionGateway } from "@limecloud/agent-runtime-client/sessionGateway";

const runtime = createAgentRuntimeClientFromSessionGateway({
  startTurn: (params, options) => appServerGateway.startTurn(params, options),
  readSession: (params, options) =>
    appServerGateway.readSession(params, options),
  cancelTurn: (params, options) => appServerGateway.cancelTurn(params, options),
  respondAction: (params, options) =>
    appServerGateway.respondAction(params, options),
  nextEvent: (timeoutMs) => appServerGateway.nextEvent(timeoutMs),
});
```

如果 gateway 是 class instance，方法内部依赖 `this`，必须在宿主里包成闭包，不要裸传类方法。

### 3. Product workbench 从 quick intent 拼 turn payload

适合从 Plugin / Content Studio 迁移 composer、quick intent、capability policy 的页面。UI 只保存当前输入和业务对象；intent 到 capability policy 的规则走共享包。

```ts
import {
  buildAgentTurnStartPayload,
  resolveWorkbenchIntentCapabilityPolicy,
  resolveWorkbenchSubmitMode,
} from "@limecloud/agent-workbench-adapter";

const submitMode = resolveWorkbenchSubmitMode({
  view: currentView,
  hasActiveSession: Boolean(sessionId),
  busy: isRunning,
  workspaceReady: Boolean(workspacePath),
  prompt,
});

if (submitMode === "disabled") return;

const policy = resolveWorkbenchIntentCapabilityPolicy({
  intentId: selectedIntentId,
  selectedSkillSlugs,
  permissionMode: "ask",
});

const payload = buildAgentTurnStartPayload({
  pluginId: "content-studio",
  workspacePath,
  prompt,
  capabilityId: "content.draft.generate",
  requiredCapabilities: policy.requiredCapabilities,
  capabilityHints: policy.capabilityHints,
  selectedSkillSlugs: policy.selectedSkillSlugs,
  businessObjectRef: {
    kind: "article-draft",
    id: draftId,
    title: draftTitle,
  },
  metadata: {
    intentId: policy.intentId,
    taskKind: policy.taskKind,
  },
});

await hostBridge.invoke("lime.agent", payload);
```

这里的 `hostBridge.invoke("lime.agent", payload)` 是宿主示例。具体 App 可以映射到 `AgentRuntimeClient.startTurn(...)` 或自己的平台 bridge，但 provider key 不应进入 payload。

### 4. 把 App Server facts 投影成标准 React UI

适合已有 `thread/read`、direct v2 lifecycle 的 App；如果还存在 provider/media 或 Lime-only evidence export 等显式 raw side-channel，再把它作为补充 facts 输入，不要把它混入标准 runtime facade。先 replay / projection，再渲染 UI primitives。

```tsx
import {
  projectAgentUiState,
  replayAppServerFacts,
} from "@limecloud/agent-runtime-projection";
import { AgentUiProjectionView } from "@limecloud/agent-runtime-ui";

const replay = replayAppServerFacts({
  readModel,
  events: drainedAgentSessionEvents,
});

const state = projectAgentUiState({
  executionEvents: replay.events,
  sourceCount: replay.state.readModel.sourceCount,
});

<AgentUiProjectionView
  state={state}
  onResolveAction={(event, action) => {
    actionResponder.respond(event, action);
  }}
  labels={agentRuntimeLabels}
/>;
```

`AgentUiProjectionView` 是 controlled component。它只表达用户意图，真正的 action response、页面跳转、artifact 打开和本地化都由宿主负责。

### 5. 只统一 capability / allowlist

适合多 App 共享能力命名，但暂时不复用 UI 或 runtime client。

```ts
import {
  buildAgentCapabilityPolicy,
  resolveAgentCapabilityIds,
  validateAgentCapabilities,
} from "@limecloud/agent-capability-catalog";

const requiredCapabilities = resolveAgentCapabilityIds([
  "research",
  "pdf_extract",
]);

const issues = validateAgentCapabilities({
  capabilities: requiredCapabilities,
  allowlist: ["research", "pdf"],
});

if (issues.length) {
  throw new Error(issues.map((issue) => issue.message).join("\n"));
}

const toolPolicy = buildAgentCapabilityPolicy({
  permissionMode: "ask",
  selectedSkillSlugs: ["copywriting-master"],
  requiredCapabilities,
});
```

## Claw 迁移到 Agents 时怎么拆

迁移 Claw / Plugin 前端实现时，不要把整页 React、store、CSS 或命令面板复制到每个 Product App。按下面的边界拆：

| Claw 中的能力                                                    | 应落到哪里                                                                   | Product App 还要自己保留什么                             |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------- |
| `@` 命令、quick intent、任务类型到 capability 的映射             | `@limecloud/agent-workbench-adapter` + `@limecloud/agent-capability-catalog` | 页面入口、文案、业务对象选择和默认 intent。              |
| 历史 alias，例如 `web_search`、`image_generation`、`pdf_extract` | `@limecloud/agent-capability-catalog`                                        | 自己页面允许哪些能力的 allowlist。                       |
| runtime events 到消息 / timeline / action / artifact 的解释      | `@limecloud/agent-runtime-projection`                                        | 本地 session store、可见范围 selector、业务路由。        |
| 标准 Agent runtime 展示组件                                      | `@limecloud/agent-runtime-ui`                                                | 产品设计语言、i18n labels、页面布局、业务卡片。          |
| sidecar、App Server JSON-RPC、`agentSession/*`                   | `@limecloud/app-server-client` 或平台宿主                                    | 资源准备、provider store、IPC / bridge、安全边界。       |
| Product App 专属工作流，例如内容工厂草稿、素材库、审核交付       | Product App 自己                                                             | 不下沉到共享包，除非多个 App 已经证明需要同一 contract。 |

判断标准：如果代码依赖某个产品页面、路由、文案、素材结构或 CSS，它通常不该进入 `packages/`。如果代码只处理 runtime facts、capability id、intent policy 或纯 projection，它才适合进入共享包。

## Content Studio 当前用法

Content Studio 的 Agents 主链应保持：

```text
Content Studio agents
  -> LIME_RUNTIME_BRIDGE
  -> lime.agent
  -> app-server --backend runtime --data-dir
  -> provider store
  -> LLM / tools
  -> direct v2 Thread/Turn/Item + artifact/read
  -> projection / UI
```

当前可复用包的职责：

- `@limecloud/agent-capability-catalog`：统一 quick intent 使用的 capability id、alias、metadata contract、tool policy 和 allowlist 校验。
- `@limecloud/agent-workbench-adapter`：把首页 / 对话页 composer intent 映射为 capability policy，判断 `start` / `send` / `queue` / `disabled`，汇总 runtime facts，拼 `lime.agent` turn payload。
- `@limecloud/agent-runtime-client/sessionGateway`：当 renderer 通过平台 bridge 调 runtime 时，使用 browser-safe gateway 适配标准 runtime client。
- `@limecloud/agent-runtime-projection`：把 `AgentPromptSession.messages`、`executionEvents`、App Server facts 和 evidence 转成标准 UI state。
- `@limecloud/agent-runtime-ui`：可在需要标准 timeline / action / artifact / evidence / Subagents primitives 的位置复用，但业务对象编辑、交付按钮和 Content Studio 五语言文案仍留在本仓库。

Content Studio 不应：

- 在 React UI 中模拟 runtime 已成功；
- 把 provider key 写入 Product App env、payload 或 workspace；
- 恢复第二套 runtime adapter；
- 把工具结果、审批状态、artifact 或 evidence 塞进普通 assistant 正文代替事实投影。

## 不要这样用

- 不要把 `@limecloud/agent-runtime-ui` 当 runtime client；它没有 transport，也不订阅 App Server。
- 不要把 `@limecloud/agent-runtime-projection` 当 store；它是纯投影函数，session 持久化由宿主负责。
- 不要在 `@limecloud/agent-workbench-adapter` 放业务文案、React state、CSS、IPC 或页面路由。
- 不要在 Product App 保存 provider key，也不要通过 `lime.agent` payload 传 key / token / secret。
- 不要在 UI 中乐观标记 tool / action / artifact / evidence 成功；等待 runtime facts。
- 不要把 `backendMode: "mock"`、fake bridge 或 fixture 当 production fallback。
- 不要把历史规划名 `agent-ui-projection` / `agent-ui-react` 写成 current package owner；当前物理包名是 `agent-runtime-projection` / `agent-runtime-ui`。
- 不要为了一个 App 的特殊流程把共享包做成“万能工作台”；先留在产品仓库，等第二个 App 复用时再抽 contract。

## 本地开发

本地跨仓开发时，先构建被引用包：

```bash
npm --prefix packages/agent-ui-contracts run build
npm --prefix packages/app-server-client run build
npm --prefix packages/agent-runtime-client run build
npm --prefix packages/agent-capability-catalog run build
npm --prefix packages/agent-workbench-adapter run build
npm --prefix packages/agent-runtime-projection run build
npm --prefix packages/agent-runtime-ui run build
```

常见组合的最小验证：

```bash
npm --prefix packages/agent-capability-catalog run test
npm --prefix packages/agent-workbench-adapter run test
npm --prefix packages/agent-runtime-client run test
npm --prefix packages/agent-runtime-projection run test
npm --prefix packages/agent-runtime-ui run test
```

下游 App 可以在本地构建配置中 alias 到 `packages/*/dist`。发布或生产构建前，应切回 npm registry 包，并同步下游 `package.json` / lockfile。`@limecloud/agent-runtime-ui` 的 `react` / `react-dom` 是 peer dependency，下游 App 必须自己提供。

## 发布和维护规则

- 根 README 维护“如何选包、如何组合、边界是什么”；单包 README 维护 API 细节和更完整示例。
- 新包加入 `packages/` 时，必须更新本文件的 App 类型选择、决策树、包速查表和接入配方。
- 新共享能力先判断属于 transport、runtime client、contracts、projection、React UI、workbench adapter、capability catalog 还是 CLI，不要新增平行万能包。
- `src/index.ts` 优先保持 barrel export 或薄 facade；复杂逻辑进入按职责拆分的模块。
- 包内测试覆盖稳定 contract，不依赖某个 Product App 的 DOM 或私有 store。
- 涉及 App Server protocol、Agent runtime facts、projection state 或 capability id 的 breaking change，必须同步下游 App 的接入文档和 conformance 测试。
