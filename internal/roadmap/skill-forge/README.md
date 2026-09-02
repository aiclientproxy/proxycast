# Lime Skill Forge 对照开发路线图

> 状态：P0-P10 第六刀完成；Read-Only HTTP API policy 已有 runtime evidence pack 端到端审计回归，并通过 GUI smoke
> 更新时间：2026-09-02
> 目标：把 Skill Forge 访谈里的 “Coding Agent 编码 CLI / API / tools、成功任务固化为 Agent、组织 AI Native 反馈闭环” 收敛成 Lime 可执行路线图，补强 skills pipeline 的生成、验证、注册、rerun 和 evidence 闭环。

当前依据：

- [../../aiprompts/skill-standard.md](../../aiprompts/skill-standard.md)
- [../../aiprompts/query-loop.md](../../aiprompts/query-loop.md)
- [../../research/pi-mono-coding-agent/README.md](../../research/pi-mono-coding-agent/README.md)
- [../../research/codex-goal/README.md](../../research/codex-goal/README.md)
- [../../exec-plans/skill-forge-completion-audit.md](../../exec-plans/skill-forge-completion-audit.md)
- [../../exec-plans/skill-forge-prompt-to-artifact-p5-plan.md](../../exec-plans/skill-forge-prompt-to-artifact-p5-plan.md)
- [../../exec-plans/skill-forge-readonly-http-api-p6-audit.md](../../exec-plans/skill-forge-readonly-http-api-p6-audit.md)
- [../../exec-plans/skill-forge-readonly-http-api-p7-plan.md](../../exec-plans/skill-forge-readonly-http-api-p7-plan.md)
- [../../exec-plans/skill-forge-readonly-http-api-p8-plan.md](../../exec-plans/skill-forge-readonly-http-api-p8-plan.md)
- [../../exec-plans/skill-forge-readonly-http-api-p9-plan.md](../../exec-plans/skill-forge-readonly-http-api-p9-plan.md)
- [../../exec-plans/skill-forge-readonly-http-api-p10-plan.md](../../exec-plans/skill-forge-readonly-http-api-p10-plan.md)

相关路线图：

- [../../aiprompts/architecture.md](../../aiprompts/architecture.md)：current ThreadGoal 与跨 turn idle continuation owner。
- [../ai-layered-design/README.md](../ai-layered-design/README.md)：AI 图层化设计路线图；它是 generated adapter 的潜在消费方，但 `LayeredDesignDocument`、Canvas Editor 和设计工程协议不归 Skill Forge 定义。

## 0. 当前落地状态

> 2026-07-22 更正：以下清单中的 `ManagedObjective`、`managed_objective` metadata 和
> completion-audit Goal 状态仅为历史实施证据，已由 canonical ThreadGoal 取代并物理删除；不得按
> 这些旧条目恢复协议、存储、Automation Goal 或 GUI。

截至 2026-05-06，Skill Forge 路线已经完成 **P0-P4 最小闭环**，并通过 [P0-P4 completion audit](../../exec-plans/skill-forge-completion-audit.md) 收口：

1. `Capability Draft` 已支持 create / list / get / verify / register 命令链。
2. verification gate 通过后，draft 才能进入 `verified_pending_registration`。
3. `capability_draft_register` 只复制标准合规草案到当前 workspace 的 `.agents/skills/<skill_directory>/`，并记录来源、verification report 与权限摘要。
4. `capability_draft_list_registered_skills` 已支持显式 `workspaceRoot` 扫描 `.agents/skills`，只投影带 `.lime/registration.json` 的 P3A 注册能力。
5. Skills 工作台已展示“Workspace 已注册能力”面板，包含来源、权限、标准检查、runtime gate 与 P3E “本回合启用”入口。
6. 注册与发现仍没有默认“自动化 / 继续这套方法”入口；“本回合启用”只写入当前 session 的显式 enable metadata，并由 ToolResult metadata 记录调用来源。
7. `agent_runtime_list_workspace_skill_bindings` 已在 `agent_runtime_* / inventory` 主链下返回 workspace skill binding readiness projection，用于说明哪些 registered skill 已经具备后续 Query Loop / `tool_runtime` 接入候选资格，以及当前仍卡在哪个 gate。
8. P3D 第一刀已支持 `request_metadata.harness.workspace_skill_bindings` / `workspaceSkillBindings`：当回合显式携带 P3C readiness 时，full runtime system prompt 会把最多 5 个 binding 投影为只读规划上下文。
9. P3D 第一刀不会打开 `allow_model_skills`、不会注入 `SkillTool` registry、不会改变默认 tool surface；`queryLoopVisible=false`、`toolRuntimeVisible=false`、`launchEnabled=false` 仍表示不可调用、不可自动化。
10. P3E 第一刀新增 `workspace_skill_runtime_enable` metadata：当前 session 可显式启用 P3C ready binding，并把 `SkillTool` 裁剪到 workspace-local allowlist；前端入口通过 `initialAutoSendRequestMetadata.harness` 传递，不写 `allow_model_skills`。
11. P3E 调用来源 metadata 已进入 `SkillTool` ToolResult：`workspace_skill_source` / `workspace_skill_runtime_enable` 会携带 source draft、verification report、registered directory 与 session 授权范围，timeline / evidence pack 可继续消费。
12. P3E 定向验证已覆盖前端 enable metadata、命令契约、Rust runtime turn gate 与 Rust SkillTool allowlist/source metadata；P4 不需要再补平行 runtime，只需要消费这些事实生成 Agent envelope。
13. P4 第一刀已新增 Agent envelope 草案 presentation：Workspace 已注册能力面板可展示 runbook、permission、manual rerun schedule 与 evidence 状态，但“转成 Agent 草案”仍是 disabled / explanation，不创建长期任务。
14. P4 evidence 第一刀已补 `timeline.json` source metadata 透传：ToolCall item 在存在 P3E metadata 时会保留 `workspaceSkillSource` / `workspaceSkillRuntimeEnable`，供后续 Agent envelope 和 evidence pack 展示消费。
15. P4 第二刀已新增 Managed Job 草案入口：ready binding 可在 Workspace 已注册能力面板打开现有持续流程弹窗，生成 `automation_job` 草案；草案默认暂停，提交后仍走既有 `createAutomationJob`。
16. Managed Job payload 仍是 `agent_turn`，`request_metadata.harness` 只写入 `agent_envelope` 与 `workspace_skill_runtime_enable`；scheduled run 不拥有第二套 Goal 状态机。
17. P4 evidence 第二刀已补 automation owner 导出：`agent_runtime_export_evidence_pack` 会把当前 session 关联的 `agent_runs` 写入 `runtime.json` / `artifacts.json` 的 `automationOwners`，保留 automation job、Agent envelope、Managed Objective 与 P3E runtime enable 的关系。
18. Workspace 已注册能力面板已能读取既有 automation jobs，并按 `agent_envelope.directory` / `skill` 反投影 Managed Job 状态、调度摘要与最近运行，避免只停留在“创建草案”入口。
19. Workspace 已注册能力面板已补暂停 / 恢复最小闭环：对匹配到的 Managed Job 复用 `updateAutomationJob(job.id, { enabled })` 切换状态，不新增平行 pause state。
20. Workspace Managed Objective 状态投影已补最小 audit 边界：`success` run 只显示为 `verifying`，等待 artifact / timeline / evidence 审计，不直接判为 `completed`。
21. Evidence pack 的 `automationOwners.runs[]` 已补 `completionAudit` 结构化输入：即使 automation run `success`，也只输出 `audit_input_ready` + `completionDecision=not_completed`，后续必须由 artifact / timeline / evidence audit 才能 completed。
22. Evidence pack 已补 `completionAuditSummary`：结合 automation owner run、workspace skill ToolCall source metadata 与 artifact / timeline 证据输出 `completed / blocked / needs_input / verifying` 判定；只有证据齐全时才允许出现 `completed`，缺 owner、run 失败、缺 audit input、缺 ToolCall evidence 均有定向回归覆盖。
23. Evidence pack 的 `summary.md` 已补 Completion Audit 人类可读入口：导出 decision、automation owner 成功计数、workspace skill ToolCall evidence、artifact evidence 与 blocking reasons，避免 completed 判定只藏在 JSON 中。
24. `agent_runtime_export_evidence_pack` 返回值已透出 `completionAuditSummary`，前端 normalizer 和 Harness 面板会展示 evidence-based decision、owner / ToolCall / artifact 计数与 blocking reasons，让 completion audit 不再只停留在落盘文件。
25. Agent envelope presentation contract 已能消费 `completionAuditSummary`：只有 `completed` 且 automation owner / Workspace Skill ToolCall / artifact-or-timeline evidence 三项齐全时才进入 `evidence_ready`，`verifying` 或缺证据不会误报为可固化。
26. Workspace 已注册能力面板已预留 `completionAuditSummariesByDirectory` 注入边界：当某个 skill 的 audit summary 为 evidence-based `completed` 时，“转成 Agent 草案”入口会复用现有 Managed Job 草案创建链；未 completed 或缺证据时仍禁用。
27. Workspace 已注册能力面板已补“审计最近运行”入口：对匹配 Managed Job 复用 `getAutomationRunHistory()` 当前网关，经 App Server `automationJob/runHistory` 找到最近 automation run 的 session，再调用 evidence export 当前网关获取 `completionAuditSummary` 并回填对应 skill，不新增查询命令或平行 evidence。
28. Agent envelope 草案摘要已补齐 `Memory / Widget / Permission / Schedule / Evidence / Runbook` 六块组成：Memory 引用 verification report 与运行修正，Widget 展示 Managed Job 状态、产物、审计结论和下一步动作。
29. Agent card / sharing 已采用派生形态：`workspace-local/<skill-directory>` 由已注册 Skill、Managed Job 和 completion audit 派生；共享范围先限制在当前 workspace / team，不进入 public Marketplace，也不新增 Agent card 存储表。
30. Workspace/team sharing discovery 边界已显式展示：同 workspace 成员通过 registered skill discovery 发现 `.agents/skills/<skill-directory>`，复用同一 Managed Job / evidence 事实源，不新增分享命令或 Marketplace。
31. Completion audit 已完成 P0-P4 要求映射；Agent envelope gate 已收紧为只有 `completionAuditSummary.decision=completed` 且 automation owner / Workspace Skill ToolCall / artifact-or-timeline 三项 evidence 齐全时才进入 `evidence_ready`，单独 `evidencePackId` 不再绕过 audit。
32. 主线下一刀已切到 [P5 prompt-to-artifact E2E 样例](../../exec-plans/skill-forge-prompt-to-artifact-p5-plan.md)：用“只读 CLI / 公开 API 每日报告”验证从 prompt、draft、verification、registration、first artifact、completion audit 到 Agent envelope 建议的完整产品证据链。
33. P5 已完成后，主线切到 [P6 只读 HTTP API adapter](../../exec-plans/skill-forge-readonly-http-api-p6-plan.md)，并已通过 [P6 completion audit](../../exec-plans/skill-forge-readonly-http-api-p6-audit.md)：verification gate 已要求网络只读权限声明、fixture input、`tests/` fixture、expected output、fixture dry-run 入口、dry-run expected-output binding、dry-run execute、structured evidence、dry-run offline 与 no credentials，并通过 Rust 定向测试、capability_draft 全套和 DevBridge smoke 证明正向 / 负向 draft 行为；`readonly_http_fixture_dry_run_execute` 会输出 script / expected output / duration / hash / stdout preview evidence，能力草案面板会消费本次 verification report 展示这些审计证据；只读 HTTP/API authoring 模板已抽到 `scripts/lib/readonly-http-api-draft-template.mjs`，默认生成 P6 所需 fixture / expected output / dry-run 文件；浏览器 mock verification 已对齐同一组 gate 与 evidence；仍不发真实 HTTP 请求，不新增 runtime。
34. P7 [只读 HTTP API 执行授权](../../exec-plans/skill-forge-readonly-http-api-p7-plan.md) 第一刀到第四刀已落地：在不打开真实联网 runtime 的前提下，verification gate 额外要求 `policy/readonly-http-session.json` 或等价 policy 声明 session-required、read-only GET、evidence-audited、受控 `credential_reference` 与 `execution_preflight` 边界；缺少该 policy、凭证引用策略或执行前检查计划的 API 草案不能进入 pending registration；正向 `readonly_http_execution_preflight` 会输出 approval request evidence，包含 preflight mode、endpoint source、GET、credential reference 与 evidence schema，并可被 API normalizer 和能力草案面板展示；authoring 模板、DevBridge smoke、浏览器 mock、API / UI 回归已对齐该 gate。
35. P8 [注册 provenance 延续](../../exec-plans/skill-forge-readonly-http-api-p8-plan.md) 第三刀已落地：`capability_draft_register` 会校验最新 passed verification report 与 manifest provenance 一致，并把 `readonly_http_execution_preflight` evidence 写入 registration summary；registered discovery、浏览器 mock、API normalizer 与 Workspace registered skills 面板会投影 / 展示该 preflight provenance，并持久化 `approvalRequests[]` pending artifact，包含 `approvalId`、`sourceCheckId`、`Endpoint=runtime_input`、`method=GET`、`credentialReferenceId=readonly_api_session`、`evidenceSchema`、`policyPath` 与 `createdAt`；本阶段仍不发真实 HTTP 请求、不保存 token、不新增 connector store、runtime 或 scheduler。
36. P9 [授权 artifact 消费门禁](../../exec-plans/skill-forge-readonly-http-api-p9-plan.md) 第十一刀已落地：每条 pending `approvalRequests[]` 已携带 `consumptionGate`、`credentialResolver`、`consumptionInputSchema`、`sessionInputIntake` 与 `sessionInputSubmissionContract`，并通过 `capability_draft_submit_approval_session_inputs` 校验当前 session 的授权确认、runtime endpoint、凭证引用确认和 evidence 捕获确认；校验通过会生成 `controlledGetPreflight` 与 `dryPreflightPlan`，只保留 `requestUrlHash=sha256(endpoint)` 与 planned evidence keys；`capability_draft_execute_controlled_get` 只在同一份 session 输入合同通过后执行一次性 GET，并返回 `responseStatus`、`responseSha256`、`executedAt`、截断 response preview 与 evidence；正向和 `request_failed` 结果会额外落盘 `.lime/capability-drafts/controlled-get-evidence/<artifact>.json`，只保存 hash、status、response metadata 与 evidence metadata，不保存 endpoint 明文、token 明文或 response preview 正文；`agent_runtime_export_evidence_pack` 现在会按当前 session 读取这些非敏感 artifact，并在 `runtime.json` / `artifacts.json` / `summary.md` 中输出 `capabilityDraftControlledGetEvidence` 可消费摘要；该摘要不复制 endpoint、token 或 response preview，也不注入 runtime、不进入 scheduler/default tool surface；校验失败返回 `blocked` 且不落 artifact；前端 API 网关、Rust command registration、DevBridge dispatcher、治理目录册、mock priority 与浏览器 mock 已同步，UI 仍不显示输入框、token 输入、调度或默认运行入口。
37. P10 [completion audit 消费](../../exec-plans/skill-forge-readonly-http-api-p10-plan.md) 第六刀已落地：`completionAuditSummary` 现在会消费当前 session 的 `capabilityDraftControlledGetEvidence` 摘要，输出受控 GET artifact / executed / scanned / skipped unsafe 计数、status 分布、`controlledGetEvidenceRequired` 与 `requiredEvidence.controlledGetEvidence`；前端 normalizer 与 Agent envelope presentation 已同步这些字段，Workspace 固化入口会在 evidence label 中展示 `受控 GET <executed>/<artifact> executed`，且未 completed 的 audit 会保持“等待 Completion Audit”而不是退回“还没有成功运行证据”；当 Workspace registered skill 带 `readonly_http_execution_preflight` verification gate / approval request 时，Managed Job 草案会在 `managed_objective.completion_evidence_policy` 声明必须具备 executed 受控 GET evidence，completion audit 因此会在缺该证据时输出 `missing_controlled_get_evidence` 并保持 `verifying`；前端 Agent envelope `evidence_ready` gate 也会防御性检查 `controlledGetEvidenceRequired`，即使收到异常 `decision=completed` 但缺 `requiredEvidence.controlledGetEvidence`，也不会启用固化入口；runtime evidence pack 已补端到端回归，证明同一个 owner metadata 在缺受控 GET evidence 时 `verifying`，补齐当前 session executed artifact 后才 `completed`，且不会复制 endpoint、token 或 response preview；GUI smoke 已通过，覆盖 workspace ready、browser runtime、site adapters、agent service skill entry、runtime tool surface、knowledge GUI 与 design canvas 主路径。普通非 HTTP 目标不受影响。该信号只作为 completion audit / Agent envelope 可见输入，不能单独把审计或固化入口推到 `completed / enabled`。
38. Codex 边界回收：Workspace 已保存技能面板移除“审计最近运行”按钮及其 `evidence/export` 调用；Codex `thread/read` 没有等价的 `completionAuditSummary`，因此 Agent envelope 在没有显式外部摘要注入时继续 fail closed。上面的 completion audit 结构保留为历史/后端 Lime 扩展事实，不得恢复为 Codex current 能力。

## 1. 先给结论

Lime 不应该另做一个 Skill Forge 式平行工具生成系统。

Lime 应该做的是：

**让 Coding Agent 把 API、CLI、网页流程生成并编译成 Lime 标准 Skill / Adapter，再由现有 Query Loop、tool_runtime、Workspace 和 evidence pack 受控执行。**

一句话北极星：

**Lime 的 skills pipeline 从“安装和调用技能”升级为“生成、编译、验证、注册、rerun，并把成功任务固化为可审计 Agent”。**

## 2. 权限宗旨

Skill Forge 路线的核心不是“无限放权”，而是：

**权限永远显式受控，能力逐级开放；限制的是未经验证、未经授权、不可审计的执行，不是限制 agent 的理解、设计和编码能力。**

固定原则：

1. Coding Agent 可以大胆理解需求、读文档、设计 adapter、写 draft、修 self-check。
2. 系统必须管住它真实执行什么、写到哪里、能不能注册、能不能长期跑。
3. P1A 默认限制 `bash / install / external write`，是为了控制第一阶段 blast radius，不代表长期永远低权限。
4. 后续只能通过 sandbox、verification gate、permission policy、用户确认和 evidence audit 逐级放开。
5. 任何外部写操作、花钱、发布、删除、改价、下单，都不能只靠模型自述安全，必须有结构化授权和可回放证据。

推荐长期分级：

```text
Level 0: read-only discovery
Level 1: draft-scoped write
Level 2: fixture dry-run
Level 3: sandbox shell
Level 4: workspace-local verified execution
Level 5: human-confirmed external write
Level 6: policy-approved scheduled external write
```

一句话：

**不是永远限制能力；是永远限制未经验证、未经授权、不可审计的执行。**

## 3. 固定主链

后续所有实现必须收敛到下面这条主链：

```text
用户目标
  -> Coding Agent / Skill Forge 识别能力缺口
  -> Coding Agent 探索 API / CLI / docs / website
  -> 生成 capability draft：Skill / Adapter / Script / Contract / Test
  -> verification gate 校验 contract / permission / dry-run / tests
  -> 注册到 workspace-local skill catalog / ServiceSkill 投影
  -> agent_runtime_submit_turn / tool_runtime 统一执行
  -> Managed Objective 判断是否继续、阻塞或完成
  -> automation job / subagent 可持久、可调度、可 rerun
  -> 成功任务主动建议固化为 Agent envelope
  -> artifact / evidence pack / telemetry / Workspace UI 统一展示
```

这条主链意味着：

1. `Coding Agent` 是能力作者，负责探索外部能力并写 adapter / contract / test。
2. `Skill Forge` 是生成、draft、验证、注册的产品和工程边界，不是执行系统。
3. `Generated Capability` 只是 draft 态，不是长期 runtime 主类型。
4. 注册后必须回到现有 Skill / ServiceSkill / Adapter / tool runtime 标准。
5. `Managed Objective` 只做目标推进控制，不是第四类执行实体。
6. 可调度任务必须复用 runtime queue、automation、subagent、evidence，不新增旁路。
7. `Agent envelope` 只是 Workspace 产品组合面：Skill / Memory / Widget / Schedule / Permission / Evidence，不是新 runtime。
8. outcome telemetry 与 evidence 相互引用但不混成同一个事实源。

## 4. 非目标

本路线图明确不做：

1. 不复制电商运营垂类产品。
2. 不新增平行 generated tools runtime。
3. 不绕过 Agent Skills 包标准。
4. 不让 agent 生成代码后直接长期执行。
5. 不新增独立 scheduler、queue、artifact、evidence 系统。
6. 不把外部 API / CLI 原始协议直接升格为 Lime 运行时协议。
7. 不在首期承诺高风险外部写操作全自动执行。
8. 不把 Codex `/goal` 照搬成 Lime 的平行 goal runtime。
9. 不复制传统 Vibe Coding / app builder，让 AI 给人生成传统 SaaS UI 作为主线。
10. 不把 public Marketplace 放在 workspace/team-scoped sharing 之前。
11. 不为组织 harness 新增平行 AI PM、AB testing、telemetry 或 evidence 系统。

## 5. 产品对象分层

### 5.1 Coding Agent / Agent Builder

`Coding Agent` 是 Skill Forge 启发里最核心的一层，负责把用户讲清楚的业务目标变成可验证的能力草案。运行边界由 [../../aiprompts/skill-standard.md](../../aiprompts/skill-standard.md) 与 [../../aiprompts/query-loop.md](../../aiprompts/query-loop.md) 共同约束。

本层可以参考 [../../research/pi-mono-coding-agent/README.md](../../research/pi-mono-coding-agent/README.md) 中对 `pi-mono` 的调研，但只参考 coding harness 的工程切面：会话分层、工具 allowlist、可插拔工具后端、事件生命周期和 deterministic test harness。Lime 不引入 pi-style 终端产品、JSONL session 事实源或全仓库 shell/write 权限。

它必须完成：

1. 理解用户目标、成功标准和风险边界。
2. 探索 API、CLI、docs、website、MCP 或本地代码入口。
3. 生成 adapter、wrapper、script、contract、permission summary 和 fixture test。
4. 根据 verification gate 的失败项修复 draft。
5. 通过验证后提交注册，而不是直接长期执行。

固定边界：

**Coding Agent 是 build-time capability author，不是新的 runtime，也不是 Managed Objective。**

### 5.2 Skill Forge

`Skill Forge` 是上游生成阶段，负责：

1. 从用户目标中识别能力缺口。
2. 探索 API / CLI / docs / website。
3. 生成 Skill Bundle、Adapter Spec、script、contract、test 草案。
4. 触发 verification gate。
5. 通过后提交注册。

固定边界：

**Skill Forge 不执行长期任务，不定义新的 runtime。**

### 5.3 Generated Capability Draft

`Generated Capability Draft` 是生成中间态，至少包含：

1. 用户目标摘要。
2. 来源能力说明。
3. 生成文件清单。
4. 输入输出 contract。
5. 权限声明。
6. 验证状态。
7. 注册目标。

固定边界：

**Draft 不能被当作 current tool 使用；验证和注册通过后，才投影为 Lime 标准对象。**

### 5.4 Workspace-local Skill

通过验证后的能力应落成 workspace-local skill：

1. 遵守 Agent Skills 包结构。
2. 可被 Skill Catalog / ServiceSkillCatalog 投影。
3. 可先被 Query Loop 读取为候选上下文；只有完成 session 显式 enable 与 `tool_runtime` 授权后才可调用。
4. 可被 workspace UI 展示来源、权限、最近运行和证据。

### 5.5 Agent Envelope

访谈中 Skill 更像 Agent 的 runbook；Lime 后续需要在 verified workspace-local skill 之上形成 `Agent envelope`，但它只属于 Workspace 产品组合面，不是新执行实体。

首期 Agent envelope 至少包含：

1. `Skill / Runbook`：已验证的 Agent Skill Bundle / Adapter。
2. `Memory`：用户偏好、历史修正、方法论和运行反馈的引用。
3. `Widget`：状态、输入、产物、阻塞点、证据入口。
4. `Schedule`：手动运行、定时运行、rerun 条件。
5. `Permission`：tool_runtime 授权、外部写确认、预算限制。
6. `Evidence`：生成、验证、注册、调用和 completion audit。

固定边界：

**Agent envelope 不执行任务；执行仍由 Query Loop、tool_runtime、automation job 和 Managed Objective 承载。**

### 5.6 Runtime Binding

执行绑定继续使用现有语义：

1. `agent_turn`
2. `browser_assist`
3. `automation_job`
4. `native_skill`

后续如果需要站点采集能力，先编译为 `SiteAdapterSpec`，再通过现有浏览器 runtime 执行。

### 5.7 Managed Objective

`Managed Objective` 是目标推进控制层，参考 [Codex `/goal` 研究](../../research/codex-goal/README.md)，负责：

1. 保存当前 managed skill / automation job 的目标和成功标准。
2. 判断是否需要继续下一轮 agent turn。
3. 在缺输入、阻塞、预算耗尽、完成或失败时停止自动续跑。
4. 要求 completion audit 消费 artifact / evidence，而不是只靠模型自报。

固定边界：

**目标推进只由 thread-owned ThreadGoal 控制，不允许恢复 ManagedObjective 或新的 runtime taxonomy。**

详细架构见 [../../aiprompts/architecture.md](../../aiprompts/architecture.md)。本路线图只描述 Skill Forge / generated skill 与 Automation 的衔接关系。

## 6. 分阶段路线

### P0：文档与边界收口

目标：固定研究、路线图、术语和禁止项。

交付：

1. 本路线图记录能力生成、验证、注册与运行边界。
2. `internal/roadmap/skill-forge/` 开发计划。
3. 明确 `Skill Forge` 不新增 runtime。
4. 明确 generated capability 必须进入 Skill / Adapter 标准。

验收：

1. 文档能解释三层架构。
2. 文档能解释和现有 skills pipeline 不冲突。
3. 文档明确 current / deprecated / dead 边界。

### P1：workspace-local skill scaffold

目标：让 agent 可以为一个明确目标生成 workspace-local skill 草案。

范围：

1. 生成 `SKILL.md`。
2. 生成 `metadata` 或等价 manifest 草案。
3. 生成 `scripts/`、`examples/`、`tests/` 的最小结构。
4. 在 Workspace 中展示 draft 状态。

验收：

1. 用户能从对话请求创建本地 skill draft。
2. draft 清楚标注来源、目标、权限、验证状态。
3. 未验证 draft 不会进入默认 tool surface。

### P2：verification gate

目标：注册前必须通过结构化校验。

最小 gate：

1. 包结构校验。
2. 输入输出 contract 校验。
3. 权限声明校验。
4. dry-run 或 fixture test。
5. 高风险权限人工确认。

验收：

1. 缺少 contract 的 draft 不能注册。
2. 未声明联网、写文件、外部写操作的 draft 不能注册。
3. 测试失败的 draft 只能保留为 draft。
4. verification 结果能进入 evidence 或等价运行记录。

### P3：registration / runtime binding

目标：通过验证的 workspace-local skill 先完成可审计注册与可审计发现，再进入现有 catalog 与 tool runtime。

范围：

1. P3A：复制为 `<workspaceRoot>/.agents/skills/<skill_directory>/`，并记录来源、verification report 与权限摘要。
2. P3B：显式按 `workspaceRoot` 发现带 `.lime/registration.json` 的 registered skill，只做 provenance projection。
3. P3C：返回 workspace skill binding readiness projection，说明 runtime binding 候选资格与下一道 gate。
4. P3D：由 `agent_runtime_submit_turn` 读取显式 `workspace_skill_bindings` metadata，并注入 Query Loop 只读规划上下文。
5. P3E：由 `request_metadata.harness.workspace_skill_runtime_enable` 显式启用，并由 `tool_runtime` / `SkillTool` session allowlist 统一裁剪和授权，只有通过 P3C ready gate 的 binding 才进入当前 session 可调用 surface。
6. P3E：调用记录写入 ToolResult metadata；P4 继续把 timeline、artifact 与 evidence 消费进 Agent envelope。

验收：

1. P3A 注册后的 skill 包只在当前 workspace 本地落盘，不修改全局 seeded skill。
2. P3A 不触发运行、自动化或外部写操作。
3. P3B 已注册 skill 可在当前 workspace 的只读 registered discovery 中看到，且包含 provenance、权限和标准检查。
4. P3B 不触发运行、自动化或外部写操作。
5. P3C readiness 只能说明 registered skill 是否具备后续接入候选资格，不能等同于可调用。
6. P3D metadata 只能让 Query Loop 读到候选上下文和 next gate，不能声称已运行或自动调用。
7. tool surface 仍由现有 runtime 控制。
8. evidence pack 能追踪 skill 来源、版本、调用结果。
9. P3E 后，`workspace_skill_source` / `workspace_skill_runtime_enable` 能把 source draft、verification report、registered directory 和 session 授权范围带到 ToolResult metadata。

### P4：managed execution / Agent envelope

目标：让验证后的 generated skill 可进入 scheduled / managed 任务，并在成功任务后形成可 rerun、可展示、可审计的 Agent envelope。

范围：

1. 绑定 `automation_job` 或 subagent team。
2. 支持暂停、恢复、阻塞、人工输入。
3. 任务产物进入 workspace artifact。
4. 可调度执行事实进入 evidence pack。
5. 成功任务后展示“继续这套方法 / 转成 Agent”的固化入口。
6. Agent card 展示 memory、widget、schedule、permission、evidence 摘要。

验收：

1. 用户关掉窗口后，任务仍能通过 runtime 状态恢复或明确阻塞。
2. 任务失败时能看到失败步骤、原因和下一步。
3. 高风险外部写操作默认要求确认。
4. Workspace 能展示最近运行、下次运行、证据入口。
5. Workspace 能把成功运行转成 Agent envelope，但不新增 runtime。

## 7. 最小可交付场景

首个场景不选电商全链路，避免范围失控。

推荐首个场景：

**给一个只读 CLI 或公开 API 生成 workspace-local skill，并定时产出 Markdown 报告。**

示例任务：

```text
每天上午 9 点读取某个公开数据源或本地 CLI 输出，生成一份趋势摘要，保存到 workspace，并在失败时提示我补配置。
```

选择理由：

1. 只读，风险低。
2. 能覆盖 CLI / API adapter 生成。
3. 能覆盖 contract、dry-run、注册、artifact、evidence。
4. 后续可自然扩展到网页、登录态和外部写操作。

## 8. 与 AI 图层化设计的关系

AI 图层化设计不是 Skill Forge 的子阶段。

固定边界：

1. `LayeredDesignDocument`、Canvas Editor、Layer Planner、设计项目导出，归 [../ai-layered-design/README.md](../ai-layered-design/README.md)。
2. Skill Forge 只负责生成和验证可复用能力，例如 provider adapter、PSD exporter wrapper、OCR / matting tool wrapper。
3. 通过验证后的 adapter 必须进入 workspace-local skill / ServiceSkill / tool_runtime 主链。
4. AI 图层化设计可以消费这些 verified adapter，但不能让它们反向定义设计文档协议。
5. 不为 AI 图层化设计新增平行 generated tools runtime。

## 9. 组织 Harness 与反馈闭环

Skill Forge 访谈中的组织 harness 不直接进入 P3E 主线，但会影响后续 roadmap 和 Workspace 设计。

固定收敛方式：

1. AI 发现候选需求只能进入 `internal/roadmap/`、`internal/exec-plans/` 或 Workspace task intake，不新增 AI PM 事实源。
2. 人类 planning 判断必须留下 repo artifact 或 evidence 引用，不只存在聊天上下文。
3. AB / telemetry 只证明 outcome，不能替代 evidence pack 的执行事实。
4. 用户成功任务、rerun 频率、阻塞原因、修复次数可以作为后续 proactive agentization 的触发信号。
5. 连续产品改造必须回到 current 主链：Skill / Query Loop / tool_runtime / Workspace / evidence。

## 10. 这一步与现有主线的关系

本路线图服务以下现有主线：

1. `skill-standard.md`：补上自动生成和编译阶段。
2. `query-loop.md`：所有执行继续走统一 submit turn 和 tool runtime。
3. `harness-engine-governance.md`：自动执行必须导出证据。
4. `remote-runtime.md`：未来远程触发只接入 current ingress，不自建 remote runtime。
5. `task/README.md`：generated skill 的任务画像、模型路由和成本限额仍归 runtime 底层。

一句话：

**这不是新产品旁路，而是把 Lime 现有 agent runtime 从“能用工具”推进到“能生产并治理工具”。**
