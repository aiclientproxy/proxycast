# Skill Forge 启发下的 Lime 实施计划

> 状态：P0-P10 第六刀完成；Read-Only HTTP API policy 已有 runtime evidence pack 端到端审计回归，并通过 GUI smoke
> 更新时间：2026-09-02
> 目标：把 Skill Forge / workspace-local generated skill / Agent envelope 的落地拆成可执行阶段，确保实现不偏离 Lime current 主链。

依赖文档：

- [./README.md](./README.md)
- [../../aiprompts/architecture.md](../../aiprompts/architecture.md)
- [../../aiprompts/skill-standard.md](../../aiprompts/skill-standard.md)
- [../../aiprompts/query-loop.md](../../aiprompts/query-loop.md)
- [../../exec-plans/skill-forge-completion-audit.md](../../exec-plans/skill-forge-completion-audit.md)
- [../../exec-plans/skill-forge-prompt-to-artifact-p5-plan.md](../../exec-plans/skill-forge-prompt-to-artifact-p5-plan.md)
- [../../exec-plans/skill-forge-readonly-http-api-p6-audit.md](../../exec-plans/skill-forge-readonly-http-api-p6-audit.md)
- [../../exec-plans/skill-forge-readonly-http-api-p7-plan.md](../../exec-plans/skill-forge-readonly-http-api-p7-plan.md)
- [../../exec-plans/skill-forge-readonly-http-api-p8-plan.md](../../exec-plans/skill-forge-readonly-http-api-p8-plan.md)
- [../../exec-plans/skill-forge-readonly-http-api-p9-plan.md](../../exec-plans/skill-forge-readonly-http-api-p9-plan.md)
- [../../exec-plans/skill-forge-readonly-http-api-p10-plan.md](../../exec-plans/skill-forge-readonly-http-api-p10-plan.md)

## 0. 当前实现进度

截至 2026-05-06，本计划已经完成 **P0-P4 最小闭环**，并通过 [P0-P4 completion audit](../../exec-plans/skill-forge-completion-audit.md) 收口：

1. P1A / P2 的最小文件事实源、静态 verification gate 和状态机已经落地。
2. P3A 已新增 `capability_draft_register`：只允许 `verified_pending_registration`，注册前复核 manifest 文件完整性与 Agent Skills 标准。
3. 注册结果只落到当前 workspace 的 `.agents/skills/<skill_directory>/`，并写入 draft 侧 `registration/latest.json` 与 registered skill 侧 `.lime/registration.json`。
4. P3B 已新增 `capability_draft_list_registered_skills`：显式传入 `workspaceRoot`，扫描当前项目 `.agents/skills`，只返回带 `.lime/registration.json` 的 P3A 注册能力。
5. 前端 Skills 工作台已经展示草案、验证、注册摘要和 Workspace 已注册能力面板；只对 P3C ready binding 展示“本回合启用”，不展示默认运行、自动化或外部写入口。
6. P3C 第一刀已新增 `agent_runtime_list_workspace_skill_bindings`：只说明当前 registered skill 是否可进入后续 runtime binding，不触发 reload、不进入默认 tool surface。
7. P3C 第一刀显式保持 `queryLoopVisible=false`、`toolRuntimeVisible=false`、`launchEnabled=false`。
8. P3D 第一刀已把显式携带的 `workspace_skill_bindings` / `workspaceSkillBindings` metadata 投影进 full runtime system prompt，支持最多 5 个 binding、snake_case / camelCase、长文本裁剪和禁止执行语义。
9. P3D 第一刀不打开 `allow_model_skills`，不注入 `SkillTool` registry，不改变默认 tool surface。
10. P3E 第一刀新增 `workspace_skill_runtime_enable` metadata，后端校验当前 workspace、P3C ready binding 与 registration provenance 后，才在当前 session scope 内打开 `SkillTool` allowlist；前端只通过 `initialAutoSendRequestMetadata.harness` 注入，不写 `allow_model_skills`。
11. P3E 调用证据最小来源链路已补：`SkillTool` ToolResult metadata 会写回 workspace skill source / runtime enable 信息；下一阶段仍要把这些 evidence 消费进长期 Managed Objective / Agent envelope 展示。
12. P3E 定向验证已覆盖前端 enable metadata、命令契约、Rust runtime turn gate 与 Rust SkillTool allowlist/source metadata；P4 应继续复用现有 runtime / automation / evidence 主链。
13. P4 第一刀已新增 Agent envelope 草案 presentation：Workspace 已注册能力面板可展示 runbook、permission、manual rerun schedule 与 evidence 状态，但不新增 runtime、scheduler 或长期授权。
14. P4 evidence 第一刀已补 `timeline.json` source metadata 透传：ToolCall item 在存在 P3E metadata 时会保留 `workspaceSkillSource` / `workspaceSkillRuntimeEnable`，供后续 Agent envelope 和 evidence pack 展示消费。
15. P4 第二刀已新增 Managed Job 草案入口：ready binding 可从 Workspace 已注册能力面板打开现有持续流程弹窗，生成默认暂停的 automation job 草案。
16. Managed Job 草案的 payload 仍是 `agent_turn`，并通过 `request_metadata.harness.agent_envelope` 与 `workspace_skill_runtime_enable` 绑定来源和 P3E session-scoped runtime enable；目标状态只归 ThreadGoal。
17. P4 evidence 第二刀已补 automation owner 导出：evidence pack 的 `runtime.json` / `artifacts.json` 会写入 `automationOwners`，用于审计 automation job、Agent envelope、Managed Objective 与 workspace skill runtime enable 的关系。
18. Workspace 已注册能力面板已补 managed job 状态投影：从既有 automation jobs 读取 `agent_envelope` metadata，显示草案/启用状态、调度摘要、最近运行与错误摘要。
19. Workspace 已注册能力面板已补暂停 / 恢复最小闭环：对匹配到的 Managed Job 直接复用 `updateAutomationJob` 修改 `enabled`，并用返回记录刷新状态投影。
20. Managed Objective 最小状态 / audit 投影已补：状态区显示 `planned` / `paused` / `running` / `blocked` / `verifying`；`success` run 只进入 `verifying`，不直接 completed。
21. Evidence pack 已补 completion audit input：`automationOwners.runs[].completionAudit` 检查 run status、Agent envelope、Managed Objective、workspace skill runtime enable 和 `completion_audit` 要求，并保持 `completionDecision=not_completed`。
22. Evidence pack 已补 completion audit summary：`runtime.json` / `artifacts.json` 会基于 automation owner、workspace skill ToolCall source metadata 和 artifact / timeline 证据给出 `completed / blocked / needs_input / verifying`，避免把 automation success 或模型自报误判为完成；负向回归已覆盖缺 owner、run 失败、缺 audit input 与缺 ToolCall evidence。
23. Evidence pack 已补 `summary.md` Completion Audit 摘要：人类先读入口可直接看到 decision、owner success count、ToolCall evidence count、artifact evidence count 与 blocking reasons。
24. Evidence export 返回值、前端 API normalizer 与 Harness 面板已接入 `completionAuditSummary`：导出问题证据包后可直接在 UI 看到 evidence-based decision 与阻塞原因。
25. Agent envelope presentation contract 已接入 completion audit gate：`completed` 且必要 evidence 齐全才进入 `evidence_ready`；非 completed audit 仍保持缺证据态，避免把 verifying 误报为可固化。
26. Workspace 已注册能力面板已补 evidence-gated Agent envelope 入口边界：按 skill directory 注入 completion audit summary 后，只有 completed + 必要 evidence 齐全才启用“转成 Agent 草案”，并复用现有 Managed Job 草案创建链。
27. Workspace 已注册能力面板已补最近运行审计入口：从匹配 Managed Job 调用 `getAutomationRunHistory()` 当前网关与 evidence export 当前网关，底层经 App Server `automationJob/runHistory` 和 `evidence/export` 把最近 automation run 的 evidence summary 回填到对应 skill 的 Agent envelope gate。
28. Agent envelope 草案摘要已补完整组成：Runbook、Memory、Widget、Permission、Schedule、Evidence 均有 presentation 字段与 Workspace 展示，仍只作为产品组合面，不新增执行实体。
29. Agent card 与 sharing 已收敛为派生展示：Agent card id 使用 `workspace-local/<skill-directory>`，事实源来自 registered skill、Managed Job 和 completion audit；sharing 先限定 workspace / team 范围，不做 public Marketplace。
30. Workspace/team sharing discovery 已明确：团队成员通过同一 workspace root 的 registered skill discovery 发现 `.agents/skills/<skill-directory>`，并复用同一 Managed Job / evidence 事实源。
31. Completion audit 已逐项映射 P0-P4 要求到代码、测试、命令验证与文档证据；Agent envelope gate 已收紧，单独 `evidencePackId` 不再进入 `evidence_ready`，必须由 completed completion audit 和三项 evidence 共同打开固化入口。
32. P5 已开主线计划：用真实 prompt-to-artifact 样例验证“只读 CLI 每日报告”从 draft 到 Agent envelope 建议的产品闭环；P5 只做 demo / smoke / evidence，不新增 runtime、scheduler 或 Marketplace。
33. P6 第一刀已落地并通过 [P6 completion audit](../../exec-plans/skill-forge-readonly-http-api-p6-audit.md)：只读 HTTP API 草案必须显式声明网络只读权限，并携带 fixture input、`tests/` fixture、expected output 与 fixture dry-run 入口，且 dry-run 入口必须绑定 expected output、保持离线，并在 verification 中执行出与 expected output 一致的 JSON；`readonly_http_fixture_dry_run_execute` 正向 check 会携带 script / expected output / duration / hash / stdout preview evidence，能力草案面板会消费本次 verification report 展示这些审计证据；只读 HTTP/API authoring 模板已抽到 `scripts/lib/readonly-http-api-draft-template.mjs`，默认生成 P6 所需 fixture / expected output / dry-run 文件并保留负向 gate 样例；浏览器 mock verification 已对齐同一组 gate 与 evidence，避免无后端 GUI 预览假通过；生成文件不得包含凭证字段；GET/read-only API 可通过 verification，未声明权限、缺 fixture 输入、缺 fixture、缺 expected output、缺 dry-run 入口、dry-run 未绑定 expected output、dry-run actual/expected 不一致、dry-run 入口真实联网、含凭证字段或外部写操作仍失败；Rust capability_draft 全套与 DevBridge smoke 已覆盖正向 / 负向 draft，不发真实 HTTP 请求、不注册、不进入 runtime。
34. P7 第一刀到第四刀已落地：只读 HTTP API 草案在 P6 gate 之外，还必须提供 session authorization policy、受控 `credential_reference` 与 `execution_preflight`，声明未来真实只读 API 执行需要用户 session 授权、只允许 GET、凭证只能从用户 session 配置解析且不进入 generated files，并能在真实请求前生成 approval request 与 evidence schema；正向 preflight check 会把 `preflightMode`、`endpointSource`、`method=GET`、`credentialReferenceId`、`evidenceSchema` 和 `policyPath` 写入 verification evidence，API normalizer 与能力草案面板可直接消费展示；本阶段仍不发真实 HTTP 请求、不保存 token、不新增 connector store 或 runtime。
35. P8 第三刀已落地：注册阶段会读取最新 passed verification report，校验其 report id 与 manifest `lastVerification` 一致，并把 `readonly_http_execution_preflight` evidence 写入 registration summary；registered discovery、浏览器 mock、API normalizer 与 Workspace registered skills 面板会投影 / 展示该 provenance，并持久化 pending `Session approval request artifact`，只记录未来真实 API 执行前需要请求授权的 endpoint 来源、GET 方法、凭证引用、证据 schema、policy 路径和 artifact id；本阶段仍只证明“注册包可追溯授权门禁并可审计授权请求”，不代表真实 HTTP 已可运行。
36. P9 第十一刀已落地：pending `Session approval request artifact` 现在不仅携带 `consumptionGate`、`credentialResolver`、`consumptionInputSchema`、`sessionInputIntake` 与 `sessionInputSubmissionContract`，还通过 `capability_draft_submit_approval_session_inputs` 校验当前 session 的授权确认、runtime endpoint、凭证引用确认和 evidence 捕获确认；校验通过返回 `validated_pending_runtime_gate`，并附带 `controlledGetPreflight.status=ready_for_controlled_get_preflight` 与 `dryPreflightPlan.status=planned_without_execution`；dry plan 只保留 `requestUrlHash=sha256(endpoint)`、planned evidence keys、policy path 和 credential reference；`capability_draft_execute_controlled_get` 只在同一份 session 输入合同通过后执行一次性 GET，并返回 `responseStatus`、`responseSha256`、`executedAt`、截断 response preview 与 evidence；正向和 `request_failed` 结果会额外写入 `.lime/capability-drafts/controlled-get-evidence/<artifact>.json`，只保存 hash、status、response metadata 与 evidence metadata，不保存 endpoint 明文、token 明文或 response preview 正文；`agent_runtime_export_evidence_pack` 现在会按当前 session 读取这些非敏感 artifact，并在 `runtime.json` / `artifacts.json` / `summary.md` 中输出 `capabilityDraftControlledGetEvidence` 可消费摘要；该摘要不复制 endpoint、token 或 response preview，也不注入 runtime、不进入 scheduler/default tool surface；校验失败返回 `blocked` 且不落 artifact。前端 API 网关、Rust command registration、DevBridge dispatcher、治理目录册、mock priority 与浏览器 mock 已同步，UI 仍不显示输入框、token 输入、调度或默认运行入口。
37. P10 第六刀已落地：`completionAuditSummary` 现在消费当前 session 的受控 GET evidence 摘要，输出 artifact / executed / scanned / skipped unsafe 计数、status 分布、`controlledGetEvidenceRequired` 与 `requiredEvidence.controlledGetEvidence`；前端 normalizer 与 Agent envelope presentation 已同步这些字段，Workspace 固化入口会展示受控 GET external evidence 计数，并在 audit 未 completed 时继续显示“等待 Completion Audit”；当 registered skill 带 `readonly_http_execution_preflight` verification gate / approval request 时，Managed Job 草案会声明 `completion_evidence_policy.controlled_get_evidence_required=true`，completion audit 会在缺 executed 受控 GET evidence 时保持 `verifying` 并输出 `missing_controlled_get_evidence`；前端 Agent envelope `evidence_ready` gate 也会防御性要求 `controlledGetEvidenceRequired -> requiredEvidence.controlledGetEvidence`，避免异常 summary 直接打开固化入口；runtime evidence pack 已补端到端回归，证明同一 Read-Only HTTP API policy 在缺受控 GET evidence 时 `verifying`，补齐当前 session executed artifact 后才 `completed`，并继续验证 endpoint / token / response preview 不进入 evidence pack；GUI smoke 已通过，覆盖 workspace ready、browser runtime、site adapters、agent service skill entry、runtime tool surface、knowledge GUI 与 design canvas 主路径；普通非 HTTP 目标仍按原三项 evidence 判定。该信号只让 completion audit / Agent envelope 可见外部只读 evidence，不能单独 completed 或启用固化入口，仍不注入 Query Loop、runtime、scheduler 或 default tool surface。

## 1. 实施总原则

1. **标准优先**
   - 生成能力必须编译为 Agent Skill Bundle / Adapter Spec / ServiceSkill 投影。

2. **验证先于注册**
   - 未通过 verification gate 的能力只能是 draft，不能进入默认 tool surface。

3. **执行回到主链**
   - 所有运行必须走 Query Loop、tool_runtime、runtime queue、artifact、evidence pack。

4. **权限显式**
   - 联网、写文件、外部写操作、花钱、发布、删除必须有结构化权限声明。

5. **能力逐级开放**
   - 首期限制的是未验证、未授权、不可审计的执行；后续可以通过 sandbox、verification gate、permission policy、用户确认和 evidence audit 逐级放开。

6. **先低风险闭环**
   - 首期只做只读 CLI / API / 文件输出，不做外部发布、下单、改价。

7. **Skill 不等于完整 Agent**
   - verified skill 只是 runbook / adapter；P4 才把 memory、widget、schedule、permission、evidence 组合成 Workspace 产品面的 Agent envelope。

8. **Outcome 不等于 Evidence**
   - evidence 证明生成、验证、注册、调用事实；telemetry / experiment 证明功能有没有改善用户结果，二者不能混成一个事实源。

权限分级口径固定为：

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

## 2. P0：文档和术语落盘

目标：让后续实现有稳定边界。

任务：

1. 把 Skill Forge 研究结论收敛到本路线图与 current 工程规则。
2. 建立 `internal/roadmap/skill-forge/` 路线图与实施计划。
3. 在文档中固定：`Skill Forge` 是生成阶段，不是 runtime。
4. 在文档中固定：`Generated Capability Draft` 不是长期主类型。
5. 在文档中固定：`Agent envelope` 是 Workspace 产品组合面，不是执行实体。
6. 在文档中固定：组织 harness 只回到 roadmap / exec-plan / telemetry / evidence 主链。

完成标准：

1. 文档能明确回答“是否和 skills pipeline 冲突”。
2. 文档能明确禁止 generated tools 平行 runtime。
3. 文档能给出 P1-P4 的实现顺序。
4. 文档能解释 Skill / Agent / Managed Objective 的边界。

## 2.5 P0.5：实现前架构补强

目标：进入代码前先固定 draft、verification、registration 与 runtime binding 硬边界，避免 P1A 变成不可治理的“生成脚本”。

任务：

1. 固定 draft store 文件结构。
2. 固定 draft manifest schema。
3. 固定 verification gate 最小检查矩阵。
4. 固定 Workspace draft 状态机。
5. 固定 generation / verification / registration 的 evidence 事件命名。
6. 固定 Capability Authoring Agent 的工具 profile：`author_readonly / author_draft_write / author_dryrun`。
7. 参考 [../../research/pi-mono-coding-agent/README.md](../../research/pi-mono-coding-agent/README.md)，明确 `author_full_shell / author_external_write` 在 P1A 默认不开放，后续只能通过升级授权放开。

完成标准：

1. 能回答 draft 放在哪里。
2. 能回答 unverified draft 如何隔离。
3. 能回答 permission summary 如何和实际行为校验。
4. 能回答 draft 如何进入后续 verification gate。
5. 能回答 Coding Agent 的每类工具权限来自哪个 backend、如何被测试。

## 3. P1：Coding Agent 生成 Skill draft scaffold

目标：支持 Coding Agent 从对话生成 workspace-local skill 草案。首期先证明“AI 能安全地产生能力”，不直接进入长期自动执行。

### 3.0 为什么 P1 必须先做 Coding Agent

Skill Forge 启发的核心不是已有工具多跑几轮，而是 Coding Agent 能把 CLI / API / docs / website 编译为可复用能力。

因此 P1 的最小实现对象应是：

```text
用户目标
  -> Coding Agent capability_generation turn
  -> Generated Capability Draft
  -> Workspace draft review
```

不是：

```text
用户目标
  -> Managed Objective
  -> 自动续跑
```

固定边界：

**Managed Objective 可以在 P3.5 / P4 接入，但不能替代 P1 的 Coding Agent 生成层。**

### 3.1 用户流

```text
用户：帮我把这个 CLI 包装成每天生成报告的技能
  -> agent 询问缺失输入
  -> agent 生成 skill draft
  -> workspace 展示草案、文件、权限、验证状态
```

### 3.2 Draft 最小内容

Draft 至少包含：

1. `SKILL.md`
   - 触发条件。
   - 任务说明。
   - 依赖和 setup。
   - 使用示例。
   - gotchas。

2. `metadata` 或等价 manifest
   - `name`
   - `description`
   - `source_kind`
   - `permission_summary`
   - `runtime_binding_target`
   - `verification_status`

3. `scripts/`
   - 最小 adapter 或 wrapper。
   - 禁止把业务状态机写死在单个脚本里。

4. `examples/`
   - 最小输入样例。
   - 期望输出样例。

5. `tests/`
   - fixture 或 dry-run 测试。

### 3.3 产品要求

1. Draft 必须清楚标注“未验证”。
2. Draft 默认不进入全局 skill catalog。
3. Draft 默认不可被自动任务调用。
4. 用户可以查看生成文件和权限声明。

完成标准：

1. 一个只读 CLI 能被生成成 skill draft。
2. draft 能在 workspace 中被发现。
3. 未验证 draft 不会出现在默认可调用工具面。

### 3.4 Capability Authoring Agent 工具面

P1 不做完整独立 Coding Agent。首期只做受控的 `Capability Authoring Agent`，工具面参考 [../../research/pi-mono-coding-agent/README.md](../../research/pi-mono-coding-agent/README.md) 的 read-only / coding tools 分级，但默认更保守：

| 工具档位                | 首期用途                                            | 状态                               |
| ----------------------- | --------------------------------------------------- | ---------------------------------- |
| `author_readonly`       | 读取 source refs、CLI help、OpenAPI、workspace docs | 必须支持                           |
| `author_draft_write`    | 写 draft root 内文件与 manifest                     | 必须支持                           |
| `author_dryrun`         | 执行 fixture / dry-run self-check                   | 可以最小支持                       |
| `author_full_shell`     | 任意 bash / install / 访问本机项目                  | P1 禁止，后续需 sandbox + 升级授权 |
| `author_external_write` | 发布、下单、改价、发消息                            | P1 禁止，后续需人工确认或策略批准  |

最小验收：

1. draft 文件路径逃逸会失败。
2. 未声明网络或写操作的 self-check 会失败。
3. CLI 探索只允许 allowlist 命令或 dry-run。
4. 生成失败和 self-check 失败都要进入 draft 状态，而不是静默重试。

## 4. P2：Verification gate

目标：把 generated skill 从“文件草案”变成“可注册能力”。

### 4.1 Gate 输入

1. Skill bundle 路径。
2. 目标 runtime binding。
3. 权限声明。
4. 输入输出 contract。
5. 测试或 dry-run 配置。

### 4.2 Gate 检查

最小检查：

1. 包结构存在且可解析。
2. `name / description / setup / examples` 足够完整。
3. 输入 contract 存在。
4. 输出 contract 存在。
5. 权限声明覆盖脚本行为。
6. dry-run 或 fixture test 通过。
7. 高风险权限需要人工确认。

### 4.3 Gate 输出

输出状态：

1. `draft`
2. `verification_failed`
3. `verified_pending_registration`
4. `registered`

失败结果必须包含：

1. 失败检查项。
2. 修复建议。
3. 是否可以让 agent 尝试修复。

完成标准：

1. 缺 contract 的 skill 无法注册。
2. 测试失败的 skill 无法注册。
3. 权限声明和实际行为不一致时无法注册。
4. 通过验证的结果能被 evidence / runtime 事实链消费。

## 5. P3：Catalog registration

目标：让通过验证的 skill 进入现有发现与调用主链。

实现拆分：

1. **P3A：workspace-local file registration**
   - 只把 `verified_pending_registration` 草案复制为 `<workspaceRoot>/.agents/skills/<skill_directory>/`。
   - 记录来源、verification report、权限摘要和目标目录。
   - 不触发 Skill reload，不接运行，不接 automation。

2. **P3B：workspace registered discovery**
   - 显式按 `workspaceRoot` 读取当前 workspace 的 `.agents/skills`。
   - 只投影带 `.lime/registration.json` 的 P3A 注册能力。
   - 返回 provenance、权限摘要、Agent Skills 标准检查与 `launchEnabled=false`。
   - 不触发 reload，不合并默认已安装方法列表，不接运行和自动化。

3. **P3C：workspace catalog binding / runtime binding**
   - 解决 workspace 选择、进程 cwd、SkillService root 与 runtime session 的一致性。
   - 第一刀已落 `agent_runtime_list_workspace_skill_bindings` 只读 readiness projection，返回 binding status / next gate / runtime visibility。
   - 暂不将 workspace-local skill 注入默认 Skill Catalog / ServiceSkillCatalog 或 SkillTool registry。

4. **P3D：Query Loop metadata projection**
   - 当前回合显式携带 `request_metadata.harness.workspace_skill_bindings` 时，`agent_runtime_submit_turn` 的 full runtime prompt 会注入候选能力上下文。
   - 该上下文只用于规划、解释 next gate 和提醒用户补授权；不能被模型当作可调用工具。
   - 前端提供 `workspaceSkillBindingsMetadata` builder，把 P3C binding 安全裁剪为 snake_case metadata fragment，且不写入 `allow_model_skills`。

5. **P3E：tool_runtime authorization**
   - 第一刀新增 `request_metadata.harness.workspace_skill_runtime_enable`，继续由 `agent_runtime_submit_turn` 承接，不新增平行命令。
   - Rust gate 校验当前 workspace、P3C ready binding、registration provenance 与 `.agents/skills` 目录边界。
   - Runtime 只在当前 session scope 内启用 `SkillTool`，并裁剪到 `project:<directory>` / `<directory>` allowlist。
   - P3E metadata 不写 `allow_model_skills`；`workspace_skill_bindings` 仍保持只读候选语义。
   - ToolResult metadata 写回 `workspace_skill_source` / `workspace_skill_runtime_enable`，让 P4 timeline、evidence pack 和 Agent envelope 能追踪 source draft、verification report、registered directory 与 session 授权范围。

### 5.1 注册位置

注册后的能力应投影到：

1. workspace-local skill catalog。
2. Skill Catalog / ServiceSkillCatalog 可发现对象。
3. Query Loop 的只读 skill binding metadata。
4. `workspace_skill_runtime_enable` 的 session-scoped tool_runtime allowlist。
5. tool_runtime 可裁剪的 tool surface 与后续 evidence。

### 5.2 注册规则

1. 只注册 `verified_pending_registration` 状态的 draft。
2. 注册必须记录来源、版本、校验摘要和权限摘要。
3. 注册不应修改全局 seeded skill。
4. workspace-local skill 只在当前 workspace 默认可见。

### 5.3 执行规则

1. 执行必须走现有 `agent_runtime_submit_turn`。
2. 工具可见性必须由 `tool_runtime` 决定。
3. 运行产物必须进入 artifact / timeline。
4. evidence pack 必须能追踪 skill 来源与调用结果。

完成标准：

1. P3B discovery 能在当前 workspace 只读发现已注册 skill。
2. P3B discovery 结果包含注册来源、verification report、权限摘要和标准检查。
3. P3B discovery 显式不可运行：`launchEnabled=false`，UI 不提供运行或自动化入口。
4. P3C 后，注册后的 skill 能在 runtime binding readiness projection 中被当前 workspace 发现。
5. P3D 后，显式 metadata 能被 Query Loop 读到，但仍不可直接调用。
6. P3E 后，注册后的 skill 只有经过 session 显式 enable 和 `tool_runtime` 授权裁剪，才能被当前 workspace 调用。
7. 其他 workspace 不会默认获得该 skill。
8. evidence pack 能看到注册来源和运行事实。
9. P3E ToolResult metadata 能看到 source draft、verification report、registered directory 与 session 授权范围。

## 6. P3.5：ThreadGoal 边界

目标：在进入长期任务前，先固定“目标推进控制层”不是新的 runtime。

参考研究：

- [../../research/codex-goal/README.md](../../research/codex-goal/README.md)
- [../../aiprompts/architecture.md](../../aiprompts/architecture.md)
- [../../aiprompts/query-loop.md](../../aiprompts/query-loop.md)

### 6.1 固定定义

`Managed Objective` 只回答：

1. 这个 managed skill / automation job 要完成什么目标。
2. 当前是否还需要继续下一轮 agent turn。
3. 当前是完成、暂停、缺输入、阻塞、预算耗尽还是失败。
4. 完成审计应消费哪些 artifact / evidence。

它不回答：

1. skill 如何生成。
2. adapter 如何编译。
3. 工具如何注册。
4. 后台任务如何调度。
5. evidence 如何导出。

固定边界：

**ThreadGoal 是挂在 canonical Thread 上的控制层，不是第四类执行实体；Automation 不拥有 Goal。**

详细状态机、预算、用量和自动续跑策略统一以 [../../aiprompts/architecture.md](../../aiprompts/architecture.md) 为准。旧 audit contract 与 automation owner binding 已删除并禁止恢复。

### 6.2 与现有主链关系

```text
workspace-local skill
  -> automation job / agent session
  -> managed objective state
  -> Query Loop agent turn
  -> artifact / evidence pack
  -> completion audit
  -> continue / needs_input / blocked / complete
```

实现时必须遵守：

1. durable 后台承载继续走 `automation job`。
2. 每轮模型执行继续走 `agent_runtime_submit_turn`。
3. 工具可见性继续由 `tool_runtime` 决定。
4. 完成审计必须引用 evidence pack 或等价 runtime facts。
5. 用户输入、暂停、预算限制优先于自动续跑。

### 6.3 P4 前置完成标准

进入 P4 前，文档和设计必须能回答：

1. Managed Objective 绑定到哪个 `automation_job` 或 `agent session`。
2. 下一轮 continuation turn 由谁触发。
3. 哪些状态会阻止自动续跑。
4. 完成审计读取哪些 evidence / artifact。
5. 哪些场景必须进入 `needs_input / blocked` 而不是继续自动跑。

## 7. P4：Managed execution / Agent envelope

目标：把 verified skill 绑定到可调度任务，并在成功运行后形成可 rerun、可展示、可审计的 Agent envelope。

### 7.1 任务形态

首期支持：

1. 手动运行。
2. 定时运行。
3. 失败后等待用户输入。
4. 用户暂停和恢复。

暂不支持：

1. 自动发布到外部平台。
2. 自动付款、下单、改价。
3. 跨 workspace 共享 generated skill。
4. 未确认的外部写操作。
5. 公开 Marketplace / Skill Store。
6. 无限自主长跑任务。

### 7.2 状态要求

可调度任务至少使用以下状态：

```text
planned
running
needs_input
blocked
verifying
completed
failed
paused
```

其中：

1. `planned / running / paused / failed` 属于任务执行生命周期。
2. `needs_input / blocked / verifying / completed` 必须能回到 Managed Objective 的完成审计语义。
3. `completed` 不能只由模型自报，必须有 artifact / evidence / verification 支撑。

### 7.3 Workspace 展示

Workspace 应展示：

1. 任务名称和绑定 skill。
2. 最近运行状态。
3. 下次运行时间。
4. 当前阻塞原因。
5. 最近产物。
6. evidence 入口。
7. 暂停、恢复、重新验证操作。
8. “继续这套方法 / 转成 Agent”的固化入口。
9. Agent card 的 memory、widget、schedule、permission、evidence 摘要。

完成标准：

1. 定时任务能运行一个 verified read-only skill。
2. app 重启后任务状态可恢复或明确标记阻塞。
3. 失败时用户能看到失败步骤和下一步。
4. evidence pack 能导出可调度运行事实。
5. 成功运行后能生成 Agent envelope 草案，但不新增 runtime。

## 8. 最小验收场景

### 场景：只读 CLI 每日报告

用户输入：

```text
把这个只读 CLI 包装成一个技能：每天 9 点运行，生成 Markdown 趋势摘要，保存到当前 workspace。失败时不要重试超过 2 次，提示我检查配置。
```

系统应完成：

1. 生成 skill draft。
2. 生成 wrapper script、示例和 fixture test。
3. 通过 verification gate。
4. 注册为 workspace-local skill。
5. 手动运行一次。
6. 创建 scheduled managed job。
7. 为该 job 绑定 Managed Objective，记录目标、成功标准和预算。
8. 产出 Markdown artifact。
9. evidence pack 可看到调用、产物、验证事实、completion audit 输入和 evidence-based completion audit summary。
10. 成功运行后，Workspace 可建议“继续这套方法 / 转成 Agent”，并生成 Agent envelope 草案。

不要求：

1. 发布到外部平台。
2. 操作浏览器登录态。
3. 连接付费 API。
4. 跨 workspace 共享。

## 9. 验证策略

### 9.1 P1 文档与 scaffold

最小验证：

1. skill draft 文件结构快照测试。
2. draft 状态不会进入默认 catalog 的单测。
3. workspace 展示 draft 状态的组件测试。

### 9.2 P2 gate

最小验证：

1. contract 缺失失败。
2. 权限缺失失败。
3. dry-run 失败阻断注册。
4. dry-run 通过允许进入 pending registration。

### 9.3 P3 registration / discovery / binding

最小验证：

1. P3A 注册只写当前 workspace 的 `.agents/skills/<skill_directory>`。
2. P3B discovery 只返回带 `.lime/registration.json` 的当前 workspace 注册项。
3. P3B UI 不展示运行、自动化或继续执行入口。
4. P3C runtime binding readiness 能发现注册 skill，并保留 `launchEnabled=false`。
5. P3D Query Loop 只读 metadata 能说明候选 skill、状态和 next gate，但不会打开 `allow_model_skills`。
6. P3E tool_runtime 仍能裁剪工具面。
7. P3E evidence pack 包含 skill source metadata。

### 9.4 P4 managed execution / Agent envelope

最小验证：

1. 定时任务状态机单测。
2. 失败后 `needs_input / blocked` 行为测试。
3. artifact 写入测试。
4. evidence pack 导出测试。
5. Agent envelope 组件测试：成功运行后显示固化入口，且不打开新的 runtime。
6. completion audit summary 测试：只有 automation owner success、workspace skill ToolCall source metadata 与 artifact / timeline 证据齐全时才输出 `completed`。
7. GUI 最小 smoke：创建、运行、查看证据。

## 10. 实现守卫

实现时必须守住以下约束：

1. 不新增 `GeneratedTool` 作为长期主类型。
2. 不新增独立 generated tool registry。
3. 不新增独立 queue / scheduler / evidence。
4. 不允许未验证 draft 进入默认 tool surface。
5. 不允许外部写操作在无人工确认时自动执行。
6. 不允许 adapter 直接成为前台产品入口。
7. 不允许来源 API / CLI schema 反向定义 Lime runtime 协议。
8. 不允许 Managed Objective 成为 `agent turn / subagent turn / automation job` 之外的第四类执行实体。
9. 不允许 generated capability 反向定义领域文档协议，例如 `LayeredDesignDocument`。
10. 不允许把 AI 图层化设计的 Canvas / document / export 主链搬进 Skill Forge runtime。
11. 不允许把 Agent envelope 实现成新 runtime、scheduler 或 evidence。
12. 不允许把 public Marketplace 放在 workspace/team-scoped sharing 之前。
13. 不允许为组织 harness 新增平行 AI PM、AB testing、telemetry 或 evidence 事实源。

## 11. 后续扩展顺序

完成只读 CLI 每日报告后，再按以下顺序扩展：

1. 只读 HTTP API adapter。
2. 只读网页采集 SiteAdapterSpec。
3. 需要登录态但只读的浏览器流程。
4. 外部写操作 draft，但默认只 dry-run。
5. 人工确认后的外部写操作。
6. 多 skill managed workflow。
7. 领域型 adapter 生成，例如图片 provider adapter、PSD exporter、OCR / matting wrapper；这些只能作为 AI 图层化设计的辅助能力，不接管 `LayeredDesignDocument` 或 Canvas Editor。
8. Agent envelope：把成功的 read-only skill run 固化为可 rerun Agent card。
9. Team-scoped sharing：只在同一 workspace / team 权限边界内共享 agent、skill、context。

一句话：

**先证明“生成能力可以被治理”，再证明“成功任务可以被固化为 Agent”，最后再扩大“能力可以做什么”。**
