# Clawstream 全链路护栏与旧实现清理路线图

> 状态：active roadmap
> 更新时间：2026-07-07
> 关联主线：`internal/aiprompts/architecture.md`、`internal/aiprompts/query-loop.md`
> 推进计划：`internal/exec-plans/clawstream-codex-derived-guardrail-plan.md`
> 当前 S1 计划：`internal/exec-plans/clawstream-s1-p0-implementation-plan.md`
> 场景骨架：`internal/roadmap/test/clawstream/scenario-registry.json`
> 目标：把 Claw 从输入到输出的流式链路固定成可复用测试账本，并清理多轮迭代留下的旧 fallback、重复 projection、mock / compat 旁路。

## 1. 结论

Claw 是当前 Agent GUI 的重灾区：消息、工具、reasoning、artifact、历史恢复和输入框状态已经被多轮迭代叠了多套实现。后续不能只靠“再补一个组件测试”维持正确性，必须先建立标准化全链路护栏，再按护栏删除旧实现。

本路线图继承 current 架构与 Query Loop 的 Thread / Turn / Item invariant：

```text
Thread 管历史，Turn 管执行，Item 管投影。
```

Clawstream 的唯一 current 主链是：

```text
Runtime Event
  -> Thread / Turn / Item materialization
  -> Projection / ReadModel / ContentPart
  -> UI snapshot / Electron E2E fixture evidence
```

任何 Claw streaming、MessageList、StreamingRenderer、Tool timeline、artifact、history hydrate、inputbar 状态或 performance 修复，都必须说明：

```text
它属于哪个 Thread？
它发生在哪个 Turn？
它落成什么 Item？
它如何进入 Projection / ReadModel / ContentPart？
GUI 和 Electron fixture 如何证明它没有回归？
```

答不清这几项，就不能继续在 Claw 主链上长逻辑。

## 2. 事实源绑定

本路线图不是独立测试体系，必须绑定这些现有事实源：

| 层级                | current owner                                                 | 说明                                                                                                    |
| ------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 架构 invariant      | `internal/aiprompts/architecture.md`、`internal/aiprompts/query-loop.md` | 所有场景按 Thread / Turn / Item 填表                                                                    |
| 验证工作流          | `internal/aiprompts/quality-workflow.md`                      | 按协议、runtime、projection、GUI、governance 的实际改动选择最小证据集                                  |
| streaming 正确性    | `internal/aiprompts/claw-streaming-rendering-correctness.md`  | `ContentPart` 保留 `sequence / turnId / itemId / phase / source`                                        |
| 流畅性目标          | `internal/roadmap/thread/streaming-fluidity-architecture.md`  | overlay live、process boundary commit、terminal commit 分层                                             |
| contract            | `packages/agent-ui-contracts`                                 | runtime event、terminal、sequence verifier 和 fixture schema                                            |
| projection replay   | `packages/agent-runtime-projection`                           | Event -> Projection / ReadModel 的 oracle                                                               |
| frontend projection | `src/components/agent/chat/components/*ContentParts*.test.ts` | `ContentPart[]` 排序、分段、owner 互斥                                                                  |
| hook state machine  | `src/components/agent/chat/hooks/agentStream*.test.ts`        | active stream、terminal、stale event、provider/model pair                                               |
| UI DOM              | `MessageList.*.test.tsx`、`StreamingRenderer.*.test.tsx`      | 首字前占位、reasoning、tool、artifact、完成态                                                           |
| Electron fixture    | `scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs`   | 真实 Electron Desktop Host + App Server sidecar + read model                                            |
| Codex 派生索引      | `internal/roadmap/test/clawstream/codex-derived-index.md`     | 从 Codex protocol / app-server / core / TUI tests 索引 Lime 还没细化的场景族                            |
| Codex 场景账本      | `internal/roadmap/test/clawstream/scenario-ledger.md`         | 把 Codex 具体测试函数索引成 Lime scenarioId、event item、projection oracle、Electron evidence、清理目标 |
| 场景骨架 registry   | `internal/roadmap/test/clawstream/scenario-registry.json`     | 固定全量 scenarioId、执行批次、evidence gate、细节顺序、优先级、状态、目标证据层和验证入口              |

Codex 不只是泛泛参考，而是 Clawstream 场景族的索引来源和默认架构准则：TUI snapshot、Thread / Turn / Item、app-server JSON-RPC integration、core session/tool runtime、tool / reasoning / approval lifecycle、MCP、Skills Runtime、Multi-Agent、Plan hydrate、compaction、resize-reflow 都要被转成 Lime 自己的 fixture 账本。opencode 只用于多模型 / 多模态 provider capability、media part、模型能力矩阵和 provider lowering，不引入其 Session/UI/Tool 架构；OpenAI / openai-agents-js 资料只作为 API event 命名证据，不替代 Codex 的工程分层。

## 3. Scenario Catalog

每个 Clawstream 场景必须进入场景账本，不能只散落在单个测试文件里。

推进顺序采用“先骨架、后细节”：`scenario-registry.json` 先固定所有 P0/P1/P2 场景的最小可执行骨架，包括 scenarioId、execution batch、`evidenceGate`、`detailOrder`、目标证据层、当前状态、下一步细节入口、`verificationCommands`、骨架字段定义和 `current / compat / deprecated / dead` 分类；`scenario-ledger.md` 再记录 Codex 来源、oracle 细节和 evidence；具体测试随后按 batch 的 `detailOrder` 逐场景把状态从 `missing` 推进到 `partial` / `partial+guard` / `covered-electron`。`scenario-registry.test.mjs` 会阻止 registry 与 ledger 脱节，也会阻止场景游离在批次之外、细节顺序缺失、验证入口为空，或 ledger 缺少 Codex 来源、标准事件项、Projection / GUI oracle 和清理目标。

| 场景                                            | 覆盖风险                                                   | v1 主线                                           |
| ----------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------- |
| `reasoning-before-text`                         | reasoning 应在首字前先出现，不被“正在生成回复”遮住         | P1-3 Event materialization / P1-5 UI projection   |
| `text-before-reasoning`                         | 早期 text 不能被后续 reasoning 或工具错误合并成最终答复    | P1-5 UI projection                                |
| `tool-before-text`                              | 工具边界前 legacy text 必须 flush 成普通 text part         | P1-3 Turn lifecycle                               |
| `tool-reasoning-text-interleaving`              | tool / reasoning / final text 按 `sequence` 混排           | P1-5 UI projection                                |
| `artifact-during-stream`                        | artifact snapshot 不绕过 read model / workbench projection | P1-6 Evidence / Replay                            |
| `artifact-after-final-text`                     | 完成后 artifact hydrate 与 live contentParts 同构          | P1-6 Evidence / Replay                            |
| `approval-request-and-resume`                   | `action.required` / `action.resolved` 不伪造 tool terminal | P2 Tool / Approval / Sandbox                      |
| `error-with-partial-content`                    | partial text + error 可见，但 Turn 终态来自结构化事件      | P1-3 Turn lifecycle                               |
| `cancel-then-continue`                          | stop 后输入框恢复，同一 session 第二轮能完成               | P1-3 Turn lifecycle                               |
| `history-hydrate-isomorphic`                    | live stream 与 read model hydrate 生成同构 `ContentPart[]` | P1-6 Replay                                       |
| `provider-model-pair-complete`                  | 不再发送 provider 有值、model 为空的半截配置               | P1-7 Provider / Model                             |
| `first-token-placeholder-without-startup-noise` | 保留“正在生成回复 / 正在输出”，删除启动说明闪现            | GUI product regression                            |
| `stale-terminal-does-not-stop-new-turn`         | 旧 terminal event 不能误停新 active stream                 | P1-3 Turn lifecycle                               |
| `no-natural-language-lifecycle-regex`           | 禁止用“已完成思考”、搜索文案、新闻正文等识别 lifecycle     | Projection / governance guard + Electron evidence |

### 3.1 Codex-derived 场景族

基础 streaming 场景不够。Codex 派生索引把 Clawstream 扩展为下列场景族，总览见 `./codex-derived-index.md`，逐项落地账本见 `./scenario-ledger.md`。

| 场景族                     | 必须覆盖                                                                                                        | Lime 当前状态                                                                                                                                             |
| -------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Thread lifecycle / hydrate | start、read、list、resume、fork、rollback、archive、settings/name update、pagination/items view                 | 部分覆盖，缺 fork/rollback/archive/page oracle                                                                                                            |
| Turn lifecycle / queue     | start、steer、interrupt、cancel/continue、pending approval、output-free restore、queued steer                   | 已覆盖 `cancel-then-continue`、`inputbar-restore-matrix`、`pending-steer-queue`；仍缺部分 P1/P2 lifecycle oracle                                          |
| Streaming parser boundary  | `output_item_added` / delta / completed 任意拆分、Plan block、reasoning/text phase                              | 缺 parser-boundary fixture                                                                                                                                |
| Plan hydrate               | proposed_plan -> Plan item、plan delta item id、decision drawer、revisioned history hydrate                     | 已覆盖：projection oracle、Electron owner/hydrate、无 proposed_plan 不生成 Plan UI 的 Electron 负向 owner oracle、local decision legacy update_plan guard |
| Tool / approval / patch    | command、file change、apply patch、request permissions、guardian/network approval                               | 基本缺口                                                                                                                                                  |
| MCP                        | tool structuredContent、resource read、elicitation form/OpenAI form、server status/name collision、thread scope | 已有 structuredContent，缺 resource/elicitation/status                                                                                                    |
| Skills Runtime             | natural、explicit `$skill`、manual enable、expert skillRefs、plaza/panel，全部 search -> read -> gate -> invoke | 已有 Electron fixture，缺 item-level oracle                                                                                                               |
| Multi-Agent Team           | parent Thread、child session、handoff、worker notification、review lane、resume lineage                         | 已有 Electron fixture，缺 taxonomy/snapshot                                                                                                               |
| Image / media              | local/remote image input、placeholder restore、ImageGeneration/media task artifact、audit log                   | 部分覆盖 image task，缺 input restore                                                                                                                     |
| Artifact / Content Factory | artifact document、workflow read model、inline image slot、worker turn、contract fail-closed                    | 已有 Electron fixture，缺 snapshot/oracle                                                                                                                 |
| Context / compaction       | compaction item、replacement history、token usage、resume/fork/rollback replay                                  | 基本缺口                                                                                                                                                  |
| UI visual snapshot         | markdown、diff、resume picker/sidebar、pager/live tail、resize/reflow、status layout                            | 基本缺口                                                                                                                                                  |

## 4. Fixture DSL

新增或迁移测试时，优先按以下结构建 fixture。字段名可按实际包内类型调整，但语义必须完整。

```ts
interface ClawstreamScenario {
  id: string;
  title: string;
  thread: {
    threadId: string;
    sessionId: string;
    affectsHistory: boolean;
    affectsExport: boolean;
  };
  turn: {
    turnId: string;
    status: "running" | "completed" | "failed" | "canceled";
    terminalEvent?: "turn.completed" | "turn.failed" | "turn.canceled";
  };
  runtimeEvents: Array<{
    id: string;
    sequence: number;
    threadId: string;
    turnId: string;
    itemId?: string;
    eventClass: string;
    status: string;
    phase?: string;
    sourceType: string;
  }>;
  expectedItems: Array<{
    itemId: string;
    type:
      | "agent_message"
      | "reasoning"
      | "tool"
      | "artifact"
      | "action_required"
      | "error";
    sequence: number;
    status: string;
  }>;
  expectedProjection: {
    contentPartTypes: string[];
    readModelStatus: string;
    timelineSparsePatchOnly?: boolean;
    inlineOwner?: "contentParts" | "timeline";
  };
  expectedDomMarkers: string[];
  forbiddenText: string[];
  performanceMarkers?: {
    hasProviderWaitMs: boolean;
    hasClientLocalOutputMs: boolean;
    maxFirstVisibleOutputMs?: number;
  };
  verificationCommands: string[];
}
```

规则：

1. `runtimeEvents[]` 必须带 `sequence / threadId / turnId`，process item 必须带 `itemId`。
2. `expectedItems[]` 校验 materialized Item，而不是 DOM 文案。
3. `expectedProjection` 校验 `ContentPart[]`、timeline sparse patch、read model status。
4. `expectedDomMarkers` 只能校验结构化 `data-testid` 和必要用户可见状态。
5. `forbiddenText` 固定收录“启动处理流程”“已接收请求”等不应闪现文案。
6. `performanceMarkers` 用于首字慢 / 输出慢回归，不把性能问题混成渲染正确性问题。
7. fixture 不保存 API key、完整 prompt、用户隐私、真实 provider response preview。

## 5. 旧实现分类与清理目标

Claw 旧实现按以下分类处理。没有用户兼容包袱时，优先删除，不新增过渡壳。

| 分类         | 对象                                                                                                                          | 处理                                    |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `current`    | typed runtime event、Thread / Turn / Item、read model、ProjectionStore、`ContentPart.metadata`、Electron + App Server fixture | 继续演进                                |
| `compat`     | 旧历史 `Message.content`、无 metadata text part、整轮无 process boundary 的 unphased legacy delta                             | 只允许历史兜底，必须有 fail-closed 分支 |
| `deprecated` | 同类型直接合并、completion suffix 盲追加、timeline 重建完整过程流、overlay 充当全局助手文本缓冲                               | 迁出后删除                              |
| `dead`       | 生产 mock fallback、`agent_runtime_*` 生产 surface、旧 Tauri wrapper、自然语言正文/展示文案 lifecycle helper                  | 删除或 retired guard-only               |

必须清理的旧路：

1. 用正文、展示文案、搜索导语、新闻内容、`Finding` 等判断 reasoning / search / final answer。
2. process boundary 后把无 `phase / itemId / sequence` 的 legacy text 当最终答复。
3. 完成态 hydrate 时从 timeline 重建第二套工具过程流，覆盖 live `contentParts`。
4. overlay 在工具 / reasoning / artifact 边界后继续充当 final answer 缓冲。
5. 旧 `agent_runtime_*` 或 renderer mock fallback 作为生产成功证据。
6. `MessageList`、`StreamingRenderer`、hook、workspace 之间重复维护同一 projection 判断。
7. 向超过 `1000` 行的 Claw 巨型文件继续追加业务逻辑。

## 6. 实施顺序

### P0：落账本与 guard

- 建立 Clawstream scenario catalog。
- 将现有 `MessageList.reasoningFlow`、`MessageList.streamingTurns`、`MessageList.runtimeStatus`、`MessageList.artifactsTimeline` 对齐到账本。
- 扩展 `streamingProjectionGuard.unit.test.ts`：禁止自然语言 lifecycle regex、旧 duplicate helper、legacy terminal 回流。
- 扩展 contract guard：terminal 只认 `turn.completed / turn.failed / turn.canceled`。

退出条件：

- 场景 catalog 至少覆盖 `reasoning-before-text`、`tool-reasoning-text-interleaving`、`cancel-then-continue`、`history-hydrate-isomorphic`。
- 旧文案判断 helper 被 guard 明确禁止。

### P1：统一 Event -> Item -> Projection oracle

- 在 `packages/agent-ui-contracts` 补 Clawstream fixture shape。
- 在 `packages/agent-runtime-projection` 用同一 fixture replay 出 projection oracle。
- 坏流必须 fail closed，不投影成“看起来正常”的 UI state。
- `ContentPart[]` owner 与 timeline sparse patch owner 互斥。

退出条件：

- 每个核心场景至少有 projection replay 断言。
- live stream 与 history hydrate 同构由 fixture 证明。

### P2：收敛前端 owner

- `agentStreamRuntimeHandler` 只做事件 reducer 边界，不做 DOM / 文案语义判断。
- `MessageList` 只消费 projection，不重建 lifecycle。
- `StreamingRenderer` 只渲染 `ContentPart[]`，不排序、不去重、不判断 terminal。
- `AgentStreamTextOverlayStore` 只承载 final answer live tail 或 process boundary 前 legacy tail。

退出条件：

- 工具 / reasoning / text / artifact 混排全部由 `sequence / itemId / phase` 驱动。
- 无变化的 state setter 返回原引用，避免每个 delta 重建 `messages / threadItems`。

### P3：删除旧 fallback 与拆大文件

- 删除已被 guard 覆盖的旧 helper、重复 projection、mock fallback 和 dead branch。
- 按 domain 拆分超大文件：状态机、projection、workspace action、render primitive 分开。
- 不为旧路径新增 compat 包装；清理失败的 residual 记录到执行计划或技术债。

退出条件：

- 旧路只剩 `test-only / retired guard`。
- 触碰文件不再继续向 `1000` 行以上巨型文件追加业务逻辑。

### P4：Electron fixture 组合证据

- `smoke:agent-runtime-current-fixture` 覆盖 current runtime fixture。
- `smoke:claw-chat-current-fixture` 覆盖 GUI 输入、sidecar、read model、backend ledger。
- `web-tools-rendering`、`cancel-then-continue`、artifact workbench 场景进入固定验证组合。

退出条件：

- GUI 证据能证明首字占位、reasoning 先出、工具 / artifact 混排、停止继续、read model completed。
- 不调用 live provider，不走 App Server mock backend，不走 renderer mock fallback。

## 7. 验证入口

### Unit / projection

```bash
npm test -- streamingProjectionGuard.unit.test.ts streamingContentPartOrder.unit.test.ts streamingContentPartSegments.unit.test.ts
npm test -- messageListTimelineContentParts.reasoning.unit.test.ts messageListItemProjection.webRetrieval.unit.test.ts
npm test -- agentStreamRuntimeHandler.test.ts agentStreamCompletionController.test.ts agentStreamUserInputSendPreparation.test.ts
```

### Component DOM

```bash
npm test -- MessageList.reasoningFlow.test.tsx MessageList.streamingTurns.test.tsx MessageList.runtimeStatus.test.tsx MessageList.artifactsTimeline.test.tsx
npm test -- StreamingRenderer.webSearch.sequence.test.tsx StreamingRenderer.processGroups.test.tsx
```

### Contract / runtime

```bash
npm run test:contracts
npm run smoke:agent-runtime-current-fixture
npm run smoke:claw-chat-current-fixture
```

### 高风险 GUI / artifact

```bash
node scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs --scenario web-tools-rendering
node scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs --scenario cancel-then-continue
npm run smoke:code-artifact-workbench-electron-fixture
npm run verify:gui-smoke
```

## 8. 清理准入规则

删除旧实现前必须满足：

1. current owner 已明确。
2. 场景账本已有对应 fixture。
3. 定向 unit / projection 测试覆盖旧实现曾经承担的用户风险。
4. mock / compat / deprecated 回流有 guard。
5. Electron fixture 或 GUI smoke 能证明主路径没有被删坏。

以下情况可直接判 `dead` 并删除：

- 已不在构建 / manifest / route / command catalog 中。
- 已有 current owner 承接同一能力。
- 只剩测试夹具、历史文档或 git history 引用。
- 用户已明确无需兼容，且当前无真实外部数据迁移约束。

以下情况只允许临时 `deprecated`：

- 仍有持久化历史数据需要一次性读取或迁移。
- 直接删除会破坏当前 read model hydrate。
- 已写清退出条件、最后删除入口和验证命令。

## 9. 完成判定

本路线图不是以“文档写完”为完成，而是以工程闭环完成为准。

| 阶段     | 完成标准                                                                                                                                     |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 文档阶段 | 本文件落地，场景 catalog / 清理分类 / 验证入口明确                                                                                           |
| 护栏阶段 | 核心场景进入 fixture replay、projection unit、component DOM                                                                                  |
| 清理阶段 | `dead / deprecated` 旧路被删除或只剩 retired guard                                                                                           |
| E2E 阶段 | Electron fixture 覆盖首字、reasoning、tool、artifact、cancel/resume、history hydrate                                                         |
| 收口阶段 | `npm run test:contracts`、`npm run smoke:agent-runtime-current-fixture`、`npm run smoke:claw-chat-current-fixture` 通过，必要 GUI smoke 通过 |

当前 P0 已清零，下一刀优先级是：

1. 把已有 Electron scenarios 补 item/projection oracle：`mcp-structured-content`、`skills-runtime`、`multi-agent-team`、`image-command`、`web-tools-rendering`；`thread-read-page-isomorphic` 已推进到 App Server + Electron `covered-electron`，后续只回头补 archived read / forked_from / failed idle flag 细节。
2. 用 guard 禁止无 turnId terminal fallback、legacy `update_plan` UI owner、自然语言 lifecycle owner 回流。
3. 删除已被 guard 覆盖且没有 current 调用的旧 helper / fallback。
