# Project / Thread-first 对标 Codex 执行计划

> 状态：active
> 更新时间：2026-07-15
> 路线图：`internal/roadmap/projectthread/README.md`
> PRD：`internal/roadmap/projectthread/prd.md`

## 主目标

把 Lime 的专家、Skills、插件、子代理、浏览器和自动化能力全部收敛到 Codex 式 Project / Thread-first 主链：

```text
Project / Workspace
  -> Thread / Session
    -> Turn / Item
      -> Expert / Agent / Skill / Tool / Plugin / Browser / Workflow
```

用户先围绕项目和当前 Thread 连续工作；专家、Skill、插件等只作为当前 Thread 内的能力、profile、tool 或执行环境，不得成为 Thread 之上的分类、历史或记忆事实源。

## 本计划边界

### current

- App Server JSON-RPC Thread/Turn/Item current 主链；仍在 catalog 中的 `agentSession/start`、`agentSession/turn/start` 仅作为受控 task bridge，`agentSession/event` 只允许明确 allowlist 的 raw side-channel。
- `Project / Workspace -> Thread / Session -> Turn / Item` 作为唯一运行事实源。
- workspace / global memory store。
- 专家、Skill、插件来源只进入 session / turn / item / evidence metadata。
- 专家实例只允许保存 project scoped skill override 等 profile 配置，不保存最近会话。
- Agent Workspace 右侧专家面板内的 profile selector；切换只覆盖当前 Thread 的下一轮 `requestMetadata`，并写入 `harness.expert_role_switch` metadata fact。

### compat

- 专家广场：保留为能力发现和模板入口。
- Skills / 插件 / Browser / Automation 的独立管理页：保留管理和发现价值，运行时必须回到 current Thread。

### deprecated

- 专家启动硬编码默认项目。
- 入口文案暗示专家拥有独立工作空间。

### dead

- 每 Agent / Expert 独立长期记忆。
- 先选 Agent / Expert 再拥有上下文的默认产品流程。
- session / memory schema 以 `agent_id` / `expert_id` 作为一等主索引。
- `expertAgentInstances.latestSessionId`、`resume_or_create`、专家“继续对话 / 新对话”分叉 UI。
- 专家 profile switch 后继续把 skill override 写回原启动专家实例。

### 兼容前提

当前没有真实用户和存量兼容约束；如果后续盘点发现旧 Agent-first 入口、旧专家稳定会话、旧私有记忆或旧私有历史正在阻碍 Project / Thread-first 主线，默认直接判为 `deprecated` / `dead` 并删除或下线，不再新增长期 compat 包装层。删除前仍需确认 current owner、调用清零或同轮迁完，并补回流守卫。

## 写集约束

本计划推进期间优先触碰：

- `internal/exec-plans/projectthread-first-plan.md`
- `internal/roadmap/projectthread/**`
- `src/features/experts/**`
- `src/components/experts/**`
- `src/components/agent/chat/workspace/*Expert*`
- `src/types/page.ts`
- 必要的 governance / contract 测试文件

当前工作树中 `src/components/agent/chat/workspace/*Article*`、图片工作台、workflow draft 相关文件已有其他改动；除非它们直接阻塞 Project / Thread-first 主线，否则本计划不触碰。

## 阶段计划

### P0：封 Agent-first 回流

目标：先让后续实现无法继续把专家、Skill、插件变成独立会话 / 独立记忆体系。

任务：

1. 盘点专家入口是否仍硬编码默认项目、跨项目复用旧 `latestSessionId` 或把专家当 session 主分类。
2. 为专家实例缓存补 project scoped 约束，只允许保存 profile 配置，不允许保存最近 session。
3. 为专家 runtime metadata / workspace metadata 补测试，证明专家只写 metadata，不创建专家记忆 root。
4. 补治理守卫，禁止 session / memory schema 新增 Agent-first 主索引。

退出条件：

- 定向测试覆盖专家入口 current project / thread scoped 行为。
- `npm run test:contracts` 不暴露新的 Agent-first 命令或 mock fallback。
- 文档和计划记录仍保留的 compat 对象、原因和退出条件。

### P1：专家入口 Thread 化

目标：专家从“稳定 Agent 会话”改为“当前项目 / 当前 Thread 的 profile 模板”。

任务：

1. 专家广场启动优先继承当前 project / workspace。
2. 无 current project 时 fail closed 或显式 detached，不静默落默认项目。
3. 同一 Thread 内支持专家 profile 切换并产生 role switch / metadata fact。

退出条件：

- 从专家广场进入不会丢项目上下文。
- 同一 Thread 内切换专家不新建 session、不丢上下文。

### P2：Skills / 插件 / Browser / Automation 回流

目标：所有能力入口运行事实都回到 current Thread。

任务：

1. Skills 运行只注入当前 Thread 的 tool/context/workflow。
2. 插件 Agent task 复用 current session target。
3. Browser profile 只作为 execution environment。
4. Automation job 输出能回写 Project / Thread / Evidence。

退出条件：

- 关键能力入口都能从 Evidence Pack 追溯到 session / thread / turn。

### P3：多 Agent 团队执行层收口

目标：多 Agent 成为执行层，不再抢占产品第一分类。

任务：

1. subagent / team roster 统一挂 parent thread。
2. handoff / worker notification / review lane 投影为 thread items。
3. GUI 中多 Agent 状态进入运行控制区和 evidence 层。

退出条件：

- team facts 可见、可恢复、可导出。
- 没有独立子 Agent 会话历史列表绕开 parent thread。

## 验证策略

本计划按改动贴边界验证，不用全量测试替代主线证据。

P0 最小验证：

```bash
npm run test:related -- src/features/experts/expertAgentInstances.ts src/features/experts/expertRuntimeBinding.ts src/components/agent/chat/workspace/useWorkspaceExpertAgentLaunchSyncRuntime.ts
npm run test:contracts
```

若触碰 GUI 主路径：

```bash
npm run verify:gui-smoke
```

若触碰 App Server / Rust session / memory schema：

```bash
npm run test:rust:related -- lime-rs/crates/app-server lime-rs/crates/core
```

## 进度日志

### 2026-07-04

- 已新增 `internal/roadmap/projectthread/README.md` 和 `internal/roadmap/projectthread/prd.md`，固定 Project / Thread-first 产品判断、PRD、架构图、时序图、流程图和验收指标。
- 新增本执行计划，用于跟踪整体目标，不把任务停留在路线图文档层。
- P0-A 已处理专家入口最直接的 Agent-first 回流：
  - `ExpertPlazaPage` 不再硬编码 `projectId: "default"`。
  - 专家启动优先使用传入 `currentProjectId / projectId`，其次使用 `agent_last_project_id` remembered project；没有项目时保持 detached，不伪造默认项目。
  - `ExpertAgentInstance` 的 `latestSessionId` 当时先改为 project scoped；该字段已在 P1-B 中进一步删除。
  - `updateExpertAgentInstanceSession` 当时缺少 project scope 时 fail closed；该写回函数已在 P1-B 中删除。
  - `ExpertAgentLaunchParams` 和 workspace 同步 hook 透传 `projectId`。
- `.gitignore` 已为 `internal/exec-plans/projectthread-first-plan.md` 与 `internal/roadmap/projectthread/**` 增加白名单，保证本计划和路线图/PRD 能作为 repo 内 versioned artifact 被跟踪。
- 验证记录：
  - 通过：`npx vitest run "src/features/experts/expertAgentInstances.test.ts" "src/components/experts/ExpertPlazaPage.test.tsx" "src/components/agent/chat/workspace/useWorkspaceExpertAgentLaunchSyncRuntime.unit.test.tsx" "src/components/AppPageContent.test.tsx"`，4 个文件 / 46 个测试通过。
  - 通过：`npm run typecheck`。
  - 未通过但非本轮写集：`npm run test:related -- src/features/experts/expertAgentInstances.ts src/components/experts/ExpertPlazaPage.tsx src/components/agent/chat/workspace/useWorkspaceExpertAgentLaunchSyncRuntime.ts src/types/page.ts src/components/AppPageContent.tsx` 扩散到 `src/components/agent/chat/index.workbench01.test.tsx`，失败用例为通用工作台轻量预览；该文件不在本计划写集，且 `src/components/agent/chat/workspace/**` 当前已有多处无关脏改动。
  - 未通过但非本轮写集：`npm run test:contracts` 失败在 App Server / Agent / MCP contract 缺口，缺失文件和字符串集中于 `lime-rs/crates/app-server/**`、`lime-rs/crates/agent/**`、`lime-rs/crates/app-server-protocol/**`、`packages/app-server-client/**`，不涉及本轮专家入口写集。
  - 未通过但非本轮写集：`npm run verify:gui-smoke` 的 renderer / Electron host build 阶段通过，App Server warmup 失败于本地 app data YAML：`unknown variant sassy_cute_executor`，当前 repo 搜索未发现该旧值，说明阻塞来自本机数据残留或外部配置，而不是专家入口改动。
- 当前阶段：P0，下一刀是补 session / memory schema 的 Agent-first 主索引守卫，并把专家广场真实 current project 从导航层显式传入，而不是只依赖 remembered project。

### 2026-07-05

- P0-B 已补 Project / Thread-first 治理守卫：
  - 新增 `src/lib/governance/projectThreadFirstBoundary.test.ts`。
  - 扫描 App Server protocol `AgentSession* / MemoryStore* / Workspace* / Thread*` JSON schema，以及 current session / memory / workspace DB、repository、App Server processor、App Server client 和前端 API 身份边界。
  - 禁止 `agent_id / expert_id / agentId / expertId` 作为 session / memory schema 的一等字段回流；专家、Agent、Skill 来源只能进入 `businessObjectRef.metadata`、runtime metadata、thread item metadata 或 evidence metadata。
  - 断言 `AgentSessionStartParams` 继续只通过 `businessObjectRef.metadata` 承载能力来源，`MemoryStoreScope` 继续只允许 `global / workspace`。
- 验证记录：
  - 通过：`npx vitest run "src/lib/governance/projectThreadFirstBoundary.test.ts"`，1 个文件 / 3 个测试通过。
  - 通过：`npm run typecheck`。
  - 通过：`git diff --check -- "src/lib/governance/projectThreadFirstBoundary.test.ts" "internal/exec-plans/projectthread-first-plan.md"`。
- 当前阶段：P0 的专家入口 project scope 与 session / memory Agent-first 主索引守卫已完成。下一刀回到 P1-A：从导航 / 页面参数层显式把 current project 传给专家广场，减少 `agent_last_project_id` remembered fallback。
- P1-A 已把侧边栏专家入口接回当前 project scope：
  - `useAppSidebarConversationActions` 新增 `navigateToExperts()`，和 Skills / Workbench 一样由侧边栏会话动作层集中解析 project scoped navigation target。
  - `AppSidebar` 点击“专家”时不再走无参数通用导航，而是显式传入 `{ currentProjectId, projectId }`。
  - `AppPageContent` 既有专家页参数透传链路已补测试，证明 `ExpertsPageParams.currentProjectId` 能进入 `ExpertPlazaPage`。
  - 仍保留 `agent_last_project_id` 作为无活跃 Agent 项目时的最近项目兜底；它属于 compat 快捷入口，不是专家页自己的会话或记忆事实源。
- 验证记录：
  - 通过：`npx vitest run "src/components/AppSidebar.test.tsx" "src/components/AppPageContent.test.tsx" "src/components/experts/ExpertPlazaPage.test.tsx"`，3 个文件 / 47 个测试通过。
  - 通过：`npm run typecheck`。
- 当前阶段：P1-A 完成。下一刀进入 P1-B：删除专家稳定会话恢复链，再进入同一 Thread 内专家 profile 切换 / role switch fact。
- P1-B 已删除专家稳定会话恢复链：
  - `ExpertAgentLaunchParams` 只保留 `launchMode: "new_thread"`，不再携带 `latestSessionId` 或 `resume_or_create`。
  - `ExpertPlazaPage` 主按钮始终在当前项目下创建新 Thread，并通过 `initialRequestMetadata / initialAutoSendRequestMetadata` 注入专家 profile；即使本地存在旧专家实例缓存，也不再传 `initialSessionId`、不显示“继续对话”。
  - 详情页删除“新对话”二级按钮；专家广场作为发现入口只保留一个回到 current Project / Thread 主链的启动动作。
  - `useWorkspaceExpertAgentLaunchSyncRuntime` 删除 “sessionId 可用即写回专家 latestSessionId” 的副作用；专家实例只继续同步 project scoped skill override 配置。
  - `expertAgentInstances` 读取旧缓存 / 云端响应时丢弃 `latestSessionId`，同步到云端时也不再上传该字段。
  - 删除五语言 `agentExperts.actions.continue / newThread` 资源 key。
  - `projectThreadFirstBoundary.test.ts` 增加专家稳定会话回流守卫，禁止 `resume_or_create`、专家 `latestSessionId`、`expert-new-thread` 和继续/新对话 key 回到实现。
- 验证记录：
  - 通过：`npx vitest run "src/components/experts/ExpertPlazaPage.test.tsx" "src/features/experts/expertAgentInstances.test.ts" "src/components/agent/chat/workspace/useWorkspaceExpertAgentLaunchSyncRuntime.unit.test.tsx" "src/components/AppPageContent.test.tsx" "src/lib/governance/projectThreadFirstBoundary.test.ts"`，5 个文件 / 49 个测试通过。
  - 通过：`npm run typecheck`。
- 当前阶段：P1-B 完成。下一刀进入 P1-C：在已有 Agent Workspace Thread 内提供专家 profile switch / role switch fact，切换只影响下一 turn metadata，不新建 session。
- P1-C 已完成前端 current Thread profile switch：
  - `workspaceExpertMetadata` 新增 `buildThreadExpertProfileSwitchRequestMetadata`，用现有专家目录构造新的 `expert / harness.expert` metadata，并写入 `harness.expert_role_switch`，明确 `scope: "thread"`。
  - `ExpertInfoPanel` 新增当前专家 selector；切换专家时不导航、不创建 session，只把新的 profile metadata 回调给 Agent Workspace。
  - `AgentChatWorkspace` 新增 `threadExpertRequestMetadataOverride`，优先作为右侧专家面板、插件 runtime context 和 `workspaceRequestMetadataBase` 的来源；下一条消息继承新专家 profile 与同一 Thread history。
  - 手动切换专家后，`useWorkspaceExpertAgentLaunchSyncRuntime` 不再把后续 skill refs 编辑同步回原 `expertAgentLaunch` 实例，避免把 A 专家的配置错写到 B 专家。
  - `projectThreadFirstBoundary.test.ts` 扩展扫描 `AgentChatWorkspace`、`ExpertInfoPanel` 与 `workspaceExpertMetadata`，防止稳定专家会话字段回流。
  - 新增五语言 `agentExperts.info.profileSwitch.*` 文案。
- P1-C App Server / read model 投影已完成：
  - 新增 `runtime/expert_role_switch.rs`，从 `runtime_options.metadata.harness.expert_role_switch` 构造 `expert.profile_switch.completed` runtime event。
  - `agentSession/turn/start` 在 accepted turn 执行时把专家切换 metadata 写入 current session runtime events；排队 turn 仍只保存 runtime options，等实际执行时再投影。
  - `thread_item_projection` 将该 event 投影为 `expert_profile_switch` item，保留 `previous_expert_id / next_expert_id / release_id / switched_at` 和 `metadata.harness.expert_role_switch`。
  - 前端 `AgentThreadItem` 类型与 Agent Thread Timeline 已识别 `expert_profile_switch`，不再落到 unsupported runtime record 兜底。
  - evidence/export 会随 current runtime events 导出同一个专家切换事实。
- 当前阶段：P1 完成；下一刀进入 P2 Skills / 插件 / Browser / Automation 回流盘点。
- 体量风险记录：`AgentChatWorkspace.tsx` 已是超大文件，本轮只做专家 metadata 接线。后续若继续改 Workspace 专家运行时或进入 P2 能力入口接线，应优先抽出 `useThreadExpertProfileRuntime` 或等价 workspace 子模块，避免继续把业务状态追加到中心文件。
- 验证记录：
  - 通过：`npx vitest run "src/components/agent/chat/workspace/workspaceExpertMetadata.unit.test.ts" "src/components/agent/chat/experts/ExpertInfoPanel.test.tsx" "src/components/agent/chat/workspace/useWorkspaceExpertAgentLaunchSyncRuntime.unit.test.tsx" "src/lib/governance/projectThreadFirstBoundary.test.ts"`，4 个文件 / 31 个测试通过。
  - 通过：`npm run typecheck`。
  - 通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server read_session_projects_thread_expert_role_switch_metadata_into_items_and_evidence`，1 个定向测试通过。
  - 通过：`npx vitest run "src/components/agent/chat/components/AgentThreadTimeline.test.tsx" "src/i18n/__tests__/loadNamespace.test.ts" "src/i18n/__tests__/types.test.ts"`，3 个文件 / 25 个测试通过。
  - 通过（前端类型补齐后）：`npm run typecheck`。
  - 未完成但非代码失败：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::read_model::` 等待 `lime-rs/target` artifact 锁超过数分钟；进程盘点显示已有其它 Cargo 测试 / 构建占用同一 target，本轮中断该额外模块级验证。
- P2-A 已完成 Skills 工作台 runtime 入口收口：
  - 盘点结论：Skills 管理 / 发现页属于 `compat`；实际运行事实源是 `buildHomeAgentParams(...)` -> `AgentChatPage` -> `agentSession/turn/start`，通过 `harness.workspace_skill_runtime_enable` 注入 current turn。
  - 删除 Skill-first 默认项目行为：`useSkillsWorkspaceProject` 不再调用 `getOrCreateDefaultProject()`，改为只读取导航层传入的 `creationProjectId` 对应 project；无 current project 时不读取 workspace registered skills / bindings，不发起 workspace skill runtime enable。
  - `SkillsWorkspacePage` 试运行已保存 workspace skill 时，`projectId` 固定使用 current `creationProjectId`，不再用默认项目兜底。
  - `skillExecutionApi` 继续只保留 `skill/list` / `skill/read` 详情读取，不暴露独立 `executeSkill`。
  - `projectThreadFirstBoundary.test.ts` 新增 Skills 回流守卫，禁止 `getOrCreateDefaultProject`、`executeSkill`、`execute_skill`、`skillSessionId`、`skill_session_id` 回到 Skills runtime surface。
- P2-A 体量风险记录：`SkillsWorkspacePage.tsx` 约 `962` 行，已接近 `1000` 行硬边界。本轮只改 ProjectThread 接线；下一次继续改 Skills 工作台业务逻辑时，应优先把 project scope/runtime enable 接线抽到单独子模块或 hook，避免中心文件继续膨胀。
- P2-A 验证记录：
  - 通过：`npx vitest run "src/components/skills/SkillsWorkspacePage.test.tsx" "src/components/skills/workspaceSkillRuntimeLaunch.test.ts" "src/lib/governance/projectThreadFirstBoundary.test.ts"`，3 个文件 / 21 个测试通过。
- P2-B 插件 Agent task current Thread 化第一刀：
  - 盘点结论：插件 / App Center 管理和 UI runtime page 仍属于 `compat` surface；`lime.agent.startTask` 的 current 事实源是 `AgentRuntimeCapabilityHost` -> App Server `agentSession/start` / `agentSession/turn/start`。
  - 删除 Plugin-first 默认项目行为：`AgentRuntimeCapabilityHost` 不再导入或调用 `getOrCreateDefaultProject()`；缺显式 `workspaceId/projectId/sessionId` 时 fail closed，不自动创建默认项目。
  - `PluginRuntimePage` 只把页面参数中的显式 `projectId` 作为 workspace 注入 Host；`PluginsPage` 打开 runtime page 时透传已有 `pageParams.projectId`，没有项目态则不制造 project。
  - `agent-runtime/tasks/` 本地 storage 只判为 `compat` task projection cache，用于刷新后恢复任务面板；长期事实源仍必须回到 App Server session/thread/turn/evidence，后续若它继续承担列表 truth，应收口到 App Server read model。
  - `projectThreadFirstBoundary.test.ts` 新增插件回流守卫，禁止 `getOrCreateDefaultProject` / `defaultProject` 回到插件 Agent task runtime surface。
- P2-C Browser / Automation 盘点与 Automation 后端收口：
  - Browser 初步分类：`BrowserRuntimePageParams`、`BrowserRuntimeWorkspace`、Browser Assist / Site Skill 导航已能携带 `projectId / contentId / initialSessionId`，当前风险低于 Automation；仍需后续用 evidence/export 证明 Browser run 产物进入 current Thread。
  - Automation 分类：Automation job 管理页和 Scheduler 设置暂判为 `compat` surface；App Server `automationJob/runNow` -> `agentSession/start` / `agentSession/turn/start` 是 current 执行事实源。
  - 删除 Automation job-id 私有 Thread fallback：`build_automation_run_start` 不再用 `automation-session-${job.id}` / `automation-thread-${job.id}` 拼接隐式 session / thread，`agent_turn` payload 必须显式提供 `session_id` 和 `thread_id`。
  - 创建 / 更新校验同步 fail closed：`validate_automation_payload` 对 `agent_turn` 要求 `session_id / thread_id` 非空，避免写入之后必定不能运行的 job。
  - TS API 类型补齐 `AgentTurnAutomationPayload.session_id / thread_id`，为后续 Thread 内创建 workflow job 留出显式 lineage 字段，不新增 Automation-first 分类。
  - `projectThreadFirstBoundary.test.ts` 新增 Automation 回流守卫，禁止 `automation-session-` / `automation-thread-` fallback 字符串回到 current surface。
  - 体量风险记录：`automation_execution.rs` 约 `923` 行，已超过 `800` 行预警线；本轮只删 fallback 和补 guard。下一次继续改 Automation execution，应优先抽出 lineage / runtime-options 子模块，避免中心文件继续膨胀。
- P2-C 验证记录：
  - 通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server automation_execution`，4 个自动化执行测试通过。
  - 通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server validate_agent_turn_payload`，2 个自动化写入边界测试通过。
  - 通过：`npx vitest run "src/lib/governance/projectThreadFirstBoundary.test.ts"`，1 个文件 / 7 个测试通过。
  - 通过：`npm run typecheck`。
  - 通过：`rustfmt --edition 2021 --check "lime-rs/crates/app-server/src/automation_execution.rs" "lime-rs/crates/app-server/src/local_data_source/automation.rs"`。
- P2-C 前端创建入口收口：
  - `AgentTurnAutomationPayload.session_id / thread_id` 改为必填；同步 `src/lib/api/automation.d.ts`，防止 TS API 类型继续表达可选 lineage。
  - 新增 `automationThreadLineage.ts`，把 lineage 归一化和 `agent_turn` payload 构造从 `AutomationJobDialog.tsx` 拆出，避免继续向 1500 行巨型组件堆业务逻辑。
  - `AutomationJobDialog` 创建/编辑 payload 时必须从 `threadLineage` 或现有 job payload 取得显式 session/thread；缺失时使用五语言本地化错误 fail closed。
  - `AutomationSettings` 的创建入口在没有 `threadLineage` 时直接 toast 并拒绝打开弹窗；`AutomationPageParams` 预留 `sessionId / threadId / projectId` 显式传递通道。
  - `projectThreadFirstBoundary.test.ts` 扩展 Automation 类型守卫，禁止 `AgentTurnAutomationPayload` 的 session/thread 字段退回 optional。
- P2-C 前端验证记录：
  - 通过：`npx vitest run "src/components/settings-v2/system/automation/automationThreadLineage.unit.test.ts" "src/components/settings-v2/system/automation/AutomationJobDialog.test.tsx" "src/components/settings-v2/system/automation/index.test.tsx"`，3 个文件 / 29 个测试通过。
  - 通过：`npm run typecheck`。
- P2-C Thread 内 workflow / automation 创建链收口：
  - 盘点结论：聊天工作区内的 service skill automation draft 是 `current` 创建入口；顶层 Automation 管理页和 Scheduler 仍是 `compat` 管理面，不承担无 Thread lineage 的创建。
  - `AgentChatWorkspace` 将当前 `sessionId` 和 `threadRead.thread_id` 作为 `threadLineage` 传入 service skill automation actions 和 `AutomationJobDialog`；当前实现中旧 session 命名仍兼容承载 thread id，后续命名拆分另行收口。
  - `useWorkspaceServiceSkillEntryActions` 在创建本地自动化任务前调用 `ensureSessionForThreadLineage` 物化当前 Thread；无法得到 session/thread 时 fail closed，不再依赖后端 job-id fallback。
  - `buildServiceSkillAutomationSetupState` 将 lineage 挂到 pending automation，`buildServiceSkillAutomationSubmitRequest` 在提交前把 `session_id / thread_id` 写入 `agent_turn` payload，防止测试或内部调用绕过 Dialog lineage 构造。
  - `projectThreadFirstBoundary.test.ts` 增加 Thread 内 service skill automation 创建链守卫，要求前端 current 路径必须显式传入、归一化并写入 session/thread lineage。
- P2-C Thread 内创建链验证记录：
  - 通过：`npx vitest run "src/components/agent/chat/workspace/workspaceServiceSkillEntryActionsViewModel.unit.test.ts" "src/components/agent/chat/workspace/useWorkspaceServiceSkillEntryActions.test.tsx" "src/components/settings-v2/system/automation/automationThreadLineage.unit.test.ts" "src/components/settings-v2/system/automation/AutomationJobDialog.test.tsx" "src/lib/governance/projectThreadFirstBoundary.test.ts"`，5 个文件 / 41 个测试通过。
  - 通过：`npm run typecheck`。
- P2-C Automation 执行 lineage 证据补强：
  - `run_now_executes_agent_turn_and_persists_run_state` 增加断言：`ExecutionRequest.session.session_id / thread_id`、`workspace_id`、`businessObjectRef(kind=automation_job,id=job-1)`、runtime metadata、JSON-RPC response 和 `AgentRun.metadata.threadId` 均来自 job payload 的显式 lineage。
  - 同一测试继续调用 `export_evidence(session_id, turn_id, include_evidence_pack=true)`，断言导出的 session/thread、message.delta 事件和 Evidence Pack turn count 均来自 automation job 执行链。
  - 该测试证明 Automation job 执行不再停留在 job 私有历史，至少会进入 App Server current session / thread / turn / evidence export 链；后续剩余是从真实 GUI/fixture 入口触发同一证据形状。
- P2-C Automation 执行验证记录：
  - 通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server automation_execution`，4 个自动化执行测试通过。
- P2-D Skills evidence/export 证据复核：
  - 盘点结论：Skills runtime enable 的 current 链路由 canonical Thread/Turn/Item 与受控 fixture evidence 证明。前端 `workspace_skill_runtime_enable` harness metadata 进入 App Server `skill_runtime_enable` gate，SkillTool 结果 metadata 进入 current read-model/evidence projection；不得恢复已删除的 evidence provider facade。
  - 复用既有 Rust 测试 `export_evidence_records_skill_invocation_from_tool_metadata`，该测试通过 `export_evidence(session_id, turn_id, include_evidence_pack=true)` 证明 workspace skill invocation 能进入 Evidence Pack，而不是停留在 Skills 工作台或私有 skill history。
- P2-D Skills evidence/export 验证记录：
  - 通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server export_evidence_records_skill_invocation_from_tool_metadata`，1 个 evidence export 定向测试通过。
- P2-D Skills GUI / fixture 真实触发验证：
  - 通过：`npm run smoke:claw-chat-current-fixture -- --scenario skills-runtime`，真实 Electron fixture 通过。
  - 该场景覆盖普通 Skills prompt、显式 Skills prompt，以及用户从 Skills 工作台已保存技能面板点击“试用一次”的手动启用路径。
  - 断言通过：`manualEnableSkillsRuntimeMetadataReachedBackend`、`manualEnableSkillsRuntimeLaunchedFromSkillsWorkspace`、`manualEnableSkillsRuntimeUsedAgentSession`、`evidenceManualEnableWorkspaceRuntimeEnableObserved`、`evidencePackManualEnableSkillSearchObserved`、`evidencePackManualEnableSkillInvocationObserved`。
  - Evidence Pack 显示手动启用路径使用 `skillGateMode: "workspace_runtime_enable"`，`skillGateWorkspaceRuntimeEnable: true`，allowlist 为 `project:capability-report`；read model 中同一 session 的 turn 以 `completed` 终态收口。
- P2-E 插件 Agent task evidence/export 证据补强：
  - 前端 current host 入口已由 `AgentRuntimeCapabilityHost` 覆盖：`lime.agent.startTask/getTask/cancelTask/submitHostResponse` 均携带 `sessionId / turnId`，task snapshot 会保留 `evidence:recorded` 和 `artifact:created` refs。
  - 新增 App Server evidence export 测试 `export_evidence_pack_includes_plugin_agent_task_events_and_refs`：构造 current session/turn，追加 `plugin_task_worker` runtime event 和插件 artifact，导出 Evidence Pack 后断言 event、artifact、session/thread/turn lineage 均可追溯。
  - 该测试证明插件 Agent task 输出不会只停留在 `agent-runtime/tasks/` compat cache；后端 Evidence Pack 至少能通过 current session/thread/turn 导出插件 task 事件和 artifact refs。
- P2-E 插件验证记录：
  - 通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server export_evidence_pack_includes_plugin_agent_task_events_and_refs`，1 个 evidence export 定向测试通过。
  - 通过：`npx vitest run "src/features/plugin/runtime/agentRuntimeCapabilityHost.test.ts"`，1 个文件 / 10 个测试通过。
- P2-E 插件真实 Electron fixture 验证：
  - 通过：`npm run smoke:claw-chat-current-fixture -- --scenario content-factory-article-workspace --timeout-ms 240000`。
  - 该场景复用已安装 `content-factory-app` 插件 fixture，覆盖插件 worker 发起 `agentSession/turn/start`、runtime events append、Right Surface 请求、Article Workspace read model、artifact read、workflow respond/cancel/retry 和插件 worker 失败证据。
  - 断言通过：`contentFactoryArticleWorkspaceWorkerTurnExecuted`、`contentFactoryArticleWorkspaceRuntimeEventsAppended`、`contentFactoryArticleWorkspaceReadModelProjected`、`contentFactoryArticleWorkspaceRightSurfaceVisible`、`contentFactoryArticleWorkspaceArtifactsProjected`、`contentFactoryArticleWorkspaceWorkerFailureEvidence`、`contentFactoryArticleWorkspaceDoesNotUseModelTurn` 等。
  - 证据显示插件 worker 任务以 `content.article.generate` 完成并产出 `content_factory.workspace_patch`，read model 的 `workerDogfoodEvidence`、article object、artifact refs 均绑定同一 Article Workspace session；该场景不走普通模型 turn，避免把插件 worker 执行误报成 Agent-first 对话。
  - 仍保留后续项：`agent-runtime/tasks/` 本地 storage 目前仍是 compat task projection cache，是否下沉到 App Server read model 需要单独设计，不作为本轮 P2-E GUI 触发门槛。
- P2-F Browser evidence/export 证据复核：
  - 盘点结论：Browser Runtime 后端 Evidence Pack 已有专门导出测试，覆盖 browser session、snapshot artifact、action index、threadId、turnId、evidence refs 与 browser file artifacts。
  - 该测试证明 Browser 操作轨迹可以按 current session/thread/turn 导出；后续真实 right surface 入口已由 `right-surface-visual-matrix` Gate B fixture 补齐。
- P2-F Browser evidence/export 验证记录：
  - 通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server export_evidence_pack_includes_browser_session_and_snapshot_artifacts`，1 个 Browser evidence export 定向测试通过。
- P2-F Browser right surface 真实入口 Gate B 闭环：
  - 通过：`npm run smoke:claw-chat-current-fixture -- --scenario right-surface-visual-matrix --timeout-ms 180000`。
  - 证据：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-summary.json`，`ok: true`，`scenario: right-surface-visual-matrix`。
  - 该场景证明 App Server pending requests 可以驱动真实 Electron toolbar 打开 `files / objectCanvas / expertInfo / browser / appSurface` 五类右侧面板；Browser surface 可见并显示 `fixture-browser-session`，`pendingAfterClicks.count = 0`。
  - 断言通过：`rightSurfaceVisualMatrixRequestedThroughAppServer`、`rightSurfaceVisualMatrixBrowserSurfaceVisible`、`rightSurfaceVisualMatrixAppSurfaceVisible`、`rightSurfaceVisualMatrixAppSurfaceMultiInstanceTabs`、`rightSurfaceVisualMatrixSurfacesMutuallyExclusive`、`rightSurfaceVisualMatrixPendingConsumeKeepsSurfaceOpen`、`rightSurfaceVisualMatrixDoesNotUseModelTurn` 等。
- P3-A historical lineage implementation correction：
  - 早期 `SubagentParentContext`、`subagentTimeline`、raw `subagent_status_changed` 和本地 Team runtime 曾作为过渡实现；这些 session-first projection 已由 S6j-S6x 物理删除，不再是 `compat` 或 current owner。
  - current parent/child lineage 只来自 AgentGraph/AgentIdentity、canonical parent/child Thread、SubAgent Item 与 `thread/list` typed roster；GUI 导航使用 child ThreadId -> sessionId，不从旧 session metadata 或 raw status event恢复。
  - 原 P3-A 34 个 Renderer 测试和 14 个 Rust 测试只保留为当时历史验证记录，不能作为当前产品证明；current 回归见 S4ad-S4ah 与 S6k-S6x evidence。
- P3-B Team facts evidence/export 第一刀：
  - 盘点结论：App Server `evidence/export` 的 Basic Evidence Pack 是 P3 后端证据 current owner；前端 Agent UI projection 和 child session 兼容上下文不能替代后端导出事实。
  - 历史 `runtime/evidence_provider.rs` / `runtime/tests/evidence_exports/team_facts.rs` 方案已删除，不再作为 current owner 或正向测试入口。current 多 Agent 团队事实由 `runtime/tests/agent_control.rs`、`runtime/tests/agent_control/*`、`runtime/canonical_thread_store/agent_graph_tests.rs`、`runtime/tests/agent_mailbox_delivery.rs` 与 `thread-store/src/agent_graph.rs` 的 canonical Thread/Turn/Item、AgentGraph、AgentIdentity 和 mailbox 测试承接。
  - `projectThreadFirstBoundary.test.ts` 的 P3 守卫已迁到 `runtime/agent_control_gateway.rs`、`runtime/read_model/canonical_items.rs` 与 `runtime/exports/metrics.rs`，防止 team facts 回到 evidence provider 或前端 projection-only 路径。
- P3-B Team facts GUI recovery correction：
  - `restoredTeamFactsProjection`、`useTeamWorkspaceRuntime` 与 `useWorkspaceTeamSessionRuntime` 已随第二套 Team runtime 删除；禁止从 child session sidecar 重新灌 Agent UI projection store。
  - cold/live GUI 现在只消费 ThreadStore-backed `agentSession/read`、canonical SubAgent Item 与 `thread/list` child roster；重启恢复不得再合成 `subagent_status_changed`。
- P3-B 后端 evidence 验证记录：
  - current 验证：AgentControl、canonical agent graph、mailbox delivery 与 thread-store graph 定向测试通过；旧 `export_evidence_pack_includes_multi_agent_team_facts` 仅保留为历史 evidence，不再作为 current 门禁。
- P3-B GUI 恢复验证记录：原 3 文件 / 13 测试对应已删除实现，不再作为 current 门禁；当前使用 canonical SubAgent focused tests、AgentControl visible DOM Gate B 与 Team runtime boundary negative guard。
- P3-B historical correction / canonical Multi-Agent evidence：
  - 旧 `multi-agent-team` scenario 只是在 external fixture backend 伪造 `subagent_status_changed`、`team.changed`、`task.changed`、`agent.handoff`、`agent.completed` 与 `worker.notification`，不能证明 RuntimeCore AgentControl、durable graph/mailbox 或 canonical SubAgent Item，因此已判 `dead / deleted / forbidden-to-restore`。
  - `scripts/agent-runtime/multi-agent-team-fixture-scenario.mjs`、scenario CLI/flow/constants/assertions、read-model/evidence summary 与 regression consumer 已物理删除；聚合 current fixture 不再运行该场景。
  - current 证据改为 `.lime/qc/s4ah-agent-control-visible-dom-gate-b.json`：六个 AgentControl 工具 runtime 与 visible DOM 均 completed，Started/Interacted/Interrupted SubAgent activity 绑定真实 child Thread，`agentSession/read` 走 `electron-ipc`，console/invoke error 为 0。
  - remote task 等真实 structured source 的 `worker.notification` 继续保留；本次只删除 synthetic Team producer，不把合法领域事件一并清空。
- P2-G Automation 真实 fixture Gate B + completion audit 闭环：
  - `smoke:managed-objective-automation` 现在先通过 App Server `agentSession/start` 物化 Automation owner session/thread，再把显式 `session_id / thread_id` 写入 `automationJob/create` 的 `agent_turn` payload；`buildAutomationJobRequest` 缺 lineage 时 fail closed。
  - Automation smoke evidence 继续分成两层：`projectThreadStatus` 证明 Project / Thread lineage、run history、runtime completion 和 `evidence/export`；`status / completionAuditStatus` 代表 Managed Objective completion audit。本轮已用真实 workspace SkillTool invocation 和 artifact 证据把两层同时打到 pass。
  - 通过：`npm run smoke:managed-objective-automation -- --timeout-ms 180000`。
  - 证据：`.lime/qc/managed-objective-automation-smoke.json`，`status: "pass"`，`projectThreadStatus: "pass"`，`completionAuditStatus: "pass"`。
  - 本次 run：job `5191581f-9ad1-4660-9839-d282982146f1`，session `sess_88310f2436aa4211b8a2d2b9833b3c0b`，thread `thread_49be303af03c42568830569c120ac0f3`；latest run `status: "success"`，Evidence Pack `latestTurnStatus: "completed"`、`turnCount: 1`、`pendingRequestCount: 0`、`knownGaps: []`。
  - ProjectThread 断言通过：`jobPayloadHasExplicitLineage`、`runSessionMatchesJobPayload`、`ownerRunMatchesJob`、`managedObjectiveProjected`、`ownerRunHasAuditInputs`、`evidencePackExported`、`evidencePackSessionScopeMatchesRun`、`evidencePackThreadScopeMatchesJobPayload`、`runtimeTurnCompleted`、`evidencePackTurnCompleted`、`fixtureReceivedChatCompletion`。
  - Completion audit 断言通过：`ownerAuditInputReady`、`workspaceSkillToolCallRecorded`、`artifactRecorded`、`completionAuditCompleted` 均为 `true`；`decision: "completed"`，`workspaceSkillToolCallCount: 1`，`artifactCount: 1`，`requiredEvidence.workspaceSkillToolCall: true`，`requiredEvidence.artifactOrTimeline: true`，`ownerAuditStatuses: ["audit_input_ready"]`。
  - 主链证明：Automation owner job 经 App Server `agentSession/turn/start` 进入 current Runtime；`workspace_skill_runtime_enable` 在当前 session scope 注册并裁剪 `project:managed-objective-automation-smoke-report`；SkillTool 结果由 `payload.result.structuredContent` / `payload.message.content[].structuredContent` 进入 `evidence/export` 的 `skill_invocations` 与 completion audit，不依赖 mock 或旧 `agent_runtime_*`。
- P2-G Automation smoke 分层验证记录：
  - 通过：`node --check "scripts/managed-objective-automation-smoke.mjs"`。
  - 通过：`node --check "scripts/lib/managed-objective-automation-smoke-support.mjs"`。
  - 通过：`node --check "scripts/lib/openai-compatible-fixture-server.mjs"`。
  - 通过：`npx vitest run "scripts/lib/managed-objective-automation-smoke-support.test.mjs" "src/features/capability-drafts/workspaceSkillAgentAutomationDraft.test.ts" --silent=passed-only --disableConsoleIntercept`，2 个文件 / 15 个测试通过。
  - 通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent register_lime_project_skill_from_directory_registers_project_namespace`。
  - 通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server apply_runtime_enable_registers_workspace_skill_source`。
  - 通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server export_evidence_marks_completed_with_workspace_skill_and_artifact`。
  - 通过：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server export_evidence_records_skill_invocation_from_tool_metadata`。

## 未完成缺口

1. session / memory schema 的基础 Agent-first 主索引守卫已补；后续若新增 protocol schema 目录或 session identity owner，必须同步纳入该守卫。
2. `agent-runtime/tasks/` 本地 storage 目前仍是插件 task projection 的 compat cache；长期是否下沉到 App Server read model 需要单独设计。
