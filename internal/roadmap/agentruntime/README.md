# Lime AgentRuntime Profile 路线图

> 状态：implementation-audited
> 更新时间：2026-05-17
> 目标：把 Lime 已有的运行时、任务、模型路由、权限、证据、GUI read model 与 AgentRuntime 标准收敛成一条可执行、可测试、可回放的 current 主链。
> 上游标准草案：`/Users/coso/Documents/dev/ai/limecloud/agentruntime`，当前对齐 `Agent Runtime v0.4.0` 与 `Lime AgentRuntime Profile`。

依赖文档：

- `internal/aiprompts/query-loop.md`
- `internal/aiprompts/task-agent-taxonomy.md`
- `internal/aiprompts/state-history-telemetry.md`
- `internal/aiprompts/harness-engine-governance.md`
- `internal/roadmap/task/README.md`
- `internal/roadmap/agentui/README.md`
- `internal/roadmap/reliability/README.md`

配套文档：

- [./prd.md](./prd.md)：背景、目标、收益、用户故事、范围、验收口径
- [./architecture.md](./architecture.md)：分层、事实源、接口与 profile 映射
- [./adjacent-protocols.md](./adjacent-protocols.md)：`agentcontext`、`agentevidence`、`agentpolicy`、`agentui` 的 owner 边界与连接合同
- [../plugin/v3/README.md](../plugin/v3/README.md)：Agent Plugins v1 标准包、Plugin v3 current owner 与旧私有 runtime 删除边界
- [./agent-unfinished-session-recovery-plan.md](./agent-unfinished-session-recovery-plan.md)：未完成 Agent 会话自动检测、首页恢复与侧栏 / 输入框异步状态计划
- [./diagrams.md](./diagrams.md)：架构图、流程图、时序图
- [./implementation-plan.md](./implementation-plan.md)：分阶段落地计划、风险和测试策略
- [./test-cases.md](./test-cases.md)：结构测试、契约测试、回放测试、证据一致性和 GUI smoke 用例
- [./completion-audit.md](./completion-audit.md)：实现完成度、证据路径、弱项和后续收口口径

## 1. 这套路线图回答什么

Lime 当前最大问题不是功能不足，而是功能已经很多，却缺少统一运行事实源，导致：

1. 任务、会话、子代理、自动化 job、模型路由、证据导出各自有局部真相。
2. GUI、review、analysis、dashboard 会二次拼状态。
3. 测试只能验证局部函数，难以证明一次 agent turn 从提交到证据导出的完整闭环。
4. 标准协议很多，但没有一条强制绑定 Lime current 主链的 profile。

本路线图回答：

**Lime 如何用 AgentRuntime Profile 把 Objective / Session / Thread / Turn / Task / Tool / Action / Evidence 收敛成一条主链，并让 GUI、review、replay、analysis 只消费这条主链。**

## 2. 固定结论

1. **AgentRuntime 是运行事实主链，不是第十个平行协议。**
   `agentcontext`、`agentpolicy`、`agentevidence`、`agentui`、`agentartifact`、`agenttool` 都是相邻 owner；AgentRuntime 负责把它们串成可恢复、可审计的执行事实，边界见 [./adjacent-protocols.md](./adjacent-protocols.md)。

2. **Lime 先实现 Lime Profile，不直接追完整公开标准。**
   公开 AgentRuntime 保持跨产品通用；Lime Profile 收紧 required ids、event families、snapshot、fixtures 和验收场景。

3. **GUI 只能投影 runtime facts。**
   Workspace、Harness、review、analysis、dashboard 不再重新判断 task status、routing state、permission state、known gaps 或 evidence summary。

4. **Evidence pack 是运行事实导出，不是后处理报表。**
   replay、review、analysis handoff 与 UI diagnostics 必须和 evidence pack 消费同一组 runtime facts。

5. **落地顺序先主链，再扩展面。**
   先保证 submit turn -> runtime events -> thread read -> evidence pack -> UI projection 的闭环，再扩展子代理、job、remote channel、历史修复等高级能力。

6. **全球本地化不进入运行事实源。**
   `type/status/taskKind/source/failureCategory/reasonCode` 等 profile 字段必须是稳定协议值，不能按语言环境变化；用户可见标题、说明、错误提示、按钮和空态只能在 AgentUI / GUI projection 层通过 key-based i18n 渲染。Runtime 可以携带 `message` 作为诊断事实，但不能把中文或英文展示文案当成状态机、测试断言或跨模块 join 条件。

7. **Chat、Claw、Plugin、Automation 都只是 runtime surface。**
   完整 AI 能力只能向 RuntimeCore、Skills、MCP、Tools 与 canonical Item 主链收敛。Plugin 只通过 Agent Plugins v1 标准包和 Plugin v3 activation 进入这些 current owner；旧 `lime.agent` / `lime.workflow` 私有 facade 与 `plugin_runtime_*` 不得恢复，详见 [../plugin/v3/README.md](../plugin/v3/README.md)。

8. **能力复用必须回到 current catalog。**
   Chat、Plugin 与 Automation 只能消费 Skills、MCP、Tool Runtime 和 App Server 投影，不为单一 surface 复制工具实现、权限系统或私有 capability catalog。

## 3. 固定主链

后续所有实现必须收敛到下面这条链：

```text
Objective / User Input
  -> Session
  -> Thread
  -> Turn
  -> Step / Item
  -> ToolCall / Action / Process / Subagent
  -> RuntimeEvent
  -> Snapshot / ThreadReadModel / TaskSnapshot
  -> EvidencePack / Replay / Review / UI projection
```

这条链意味着：

1. `Session / Thread / Turn` 负责执行上下文。
2. `Task / Run / Attempt` 负责目标工作与重试历史。
3. `Step / Item` 负责有序运行项。
4. `ToolCall / Action / Process / Subagent` 负责可审计副作用与等待点。
5. `RuntimeEvent` 是 canonical event stream。
6. `Snapshot / ThreadReadModel` 是 GUI 和恢复入口。
7. `EvidencePack` 是 replay / review / analysis 的共同事实源。

## 4. current / compat / deprecated / dead 分类

### current

后续继续强化的主路径：

1. `agent_runtime_submit_turn -> runtime_turn -> runtime_queue -> stream_reply_once`。
2. `AgentRuntimeThreadReadModel` 与 current runtime event 链。
3. `agent_runtime_export_evidence_pack` 及其 replay / analysis / review 派生物。
4. `TaskProfile / RoutingDecision / LimitState` 已接入的模型路由事实。
5. Workspace / Harness GUI 对 runtime read model 的只读投影。
6. Plugin Runtime Surface 作为新调用面，委托 AgentRuntime 主链执行 App-scoped task；`PluginRuntimePage` 的 Host Bridge `lime.agent` 已接入 `plugin_runtime_*` facade，并支持 App 内响应 ask / elicitation / tool confirmation。Host Bridge 已支持 task subscription first-cut，通过 `capability:subscribe / capability:event` 把 Host 侧 `getTask` 轮询结果推回 App iframe；订阅时也会监听 `plugin_runtime:{appId}:{taskId}` Tauri / DevBridge runtime event，把后端 runtime event 直接转发给 App。AgentRuntime profile event 生成处现在会在同名 event bus 主动追加 `plugin_runtime:profileProjection` payload，把 `turn.* / tool.* / action.* / routing.* / model.*` 转成 App canonical `taskEvents`；高价值 `RuntimeAgentEvent` 也会追加 `plugin_runtime:runtimeEventProjection`，其中 artifact 事件可直接携带 `workspacePatch / contentFactoryWorkspacePatch`，runtime event / timeline metadata 中显式存在的 `evidenceRefs / verificationOutcomes` 也会被主动投影为 `evidence:recorded / evidence:verified`；Evidence Pack / analysis / review / save review 导出成功后也会按 Plugin scope 主动 emit `plugin_runtime:harnessExportProjection`，把导出 root、制品和 completion audit completed 事实投影为 App task events；成功终态如果暂未带 workspace patch，则继续短轮询 replay 最终 artifact。`AgentRuntimeCapabilityHost` 也会从 `threadRead.artifacts` 补投 `artifact:created` payload。Host dispatcher 已补 high-level manifest capability gate，未声明 `lime.agent` 等 Host capability 会被拒绝；Claw capability hint 还会校验 manifest `toolRefs[].capabilities` allowlist，避免 App 只声明一个 catalog key 后任意启动 Claw 能力。内容工厂实际 App 已把主生产结果和确认链结果写回 `lime.storage / lime.artifacts / lime.evidence`；AgentRuntime 原生 `FileArtifact` 已投影为 `artifact:created`，runtime task state 也已通过 Plugin storage 做跨刷新恢复第一刀；内容工厂 task 已在 runtime message 与 `harness.plugin_runtime_output_contract` 写明 `contentFactoryWorkspacePatch / workspacePatch` producer contract，真实宿主 iframe 已验证最终 patch 可物化到 App 页面；多个 capability hint 会被投影到 `harness.plugin_runtime_capability_workflow`，复合 output contract 任务保持 `metadata_only`，不强行启动单一 Claw Skill。
7. Claw Capability Catalog 作为已实现 `@` 能力的复用索引，后续供 Chat、Plugin、Automation 共用。

### compat

允许短期存在但只能委托的路径：

1. 旧字段映射为 AgentRuntime profile ids。
2. 旧 GUI 状态卡读取新 read model 后继续展示。
3. 旧 evidence/replay 命令通过 `agent_runtime_export_*` 导出。
4. `plugin_cmd.rs` 继续负责 package / installed state / UI runtime / scoped model env 注入，但不能扩展为完整 AgentRuntime owner。
5. 前端 `CapabilityHost` / `WorkflowRuntimeHost` 可暂时作为 mock 或 adapter，但生产 AI 任务必须迁向后端 Plugin Runtime Surface。

退出条件：一旦调用方能直接读取 profile read model 或 evidence pack，就删除 compat 映射。

### deprecated

禁止继续扩展的方向：

1. GUI 自己拼 task status / route status / known gaps。
2. analysis/review 各自重建 observability summary。
3. 子代理、automation job、task center 各自维护完成真相。
4. 只靠文本消息推断 tool success、permission denial 或 completion。
5. Plugin 直接使用模型 token / OpenAI-compatible API 完成主流程。
6. Plugin 通过嵌入通用 Chat 让用户手动复制结果。
7. 为内容工厂等垂直 App 增加专用 Agent Tauri command。

### dead

可以直接否定的方向：

1. `agentruntime_ui_state` 作为 UI 专用事实源。
2. `objective_runtime` 作为第四类 runtime taxonomy。
3. `evidence_summary_builder_v2` 作为 evidence pack 平行导出链。
4. 没有 `session/thread/turn` 关联键的 request telemetry 被当作会话级证据。
5. 新建第二套 `plugin_agent_runtime` 执行事实源。
6. 为 Plugin 复制 Claw `*_skill_launch.rs` 或单独工具权限系统。

## 5. 先读顺序

1. [./prd.md](./prd.md)
2. [./architecture.md](./architecture.md)
3. [./diagrams.md](./diagrams.md)
4. [./adjacent-protocols.md](./adjacent-protocols.md)
5. [../plugin/v3/README.md](../plugin/v3/README.md)
6. [./implementation-plan.md](./implementation-plan.md)
7. [./test-cases.md](./test-cases.md)
8. `internal/aiprompts/harness-engine-governance.md`
9. `internal/roadmap/task/README.md`

## 6. 完成判定

本路线图完成时，Lime 至少应该能做到：

1. 任意 agent turn 都能导出符合 Lime AgentRuntime Profile 的核心事件和 thread read snapshot。
2. GUI、evidence、review、replay、analysis 读取同一组 runtime facts。
3. tool approval、model routing、task retry、evidence export 都有 fixture 或结构测试覆盖。
4. 旧旁路状态源被分类为 `compat / deprecated / dead`，并有退出条件。
5. 主路径 GUI 冒烟能证明用户真实看到的是 profile read model 投影。

当前完成审计见 [./completion-audit.md](./completion-audit.md)。审计口径是 current MVP 工程闭环完成；Evidence service 已完成 request telemetry、profile projection、completion audit、modality contract、auxiliary runtime、verification / artifact validator、observability / signal coverage、known gaps 与 pack output renderer 拆分。Evidence `summary.md` 已支持 zh-CN / zh-TW / en-US / ja-JP / ko-KR，Replay / Analysis / Review Markdown 已接入 locale-aware presentation copy。完整产品化仍需继续补细粒度正文 copy 与更强 GUI E2E。
