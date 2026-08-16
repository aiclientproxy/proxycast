# Warp 对照多模态管理实施计划

> 状态：current planning source
> 更新时间：2026-05-05
> 目标：把 [README.md](./README.md) 的路线图拆成自下而上的执行阶段，先建设底层多模态运行合同，再把 `@` 命令、按钮和 Scene 这类上层入口绑定上来。

## 0. 排序修正

上一版计划把“盘点 `@` 命令入口”放在第一阶段，容易造成误解：好像多模态治理要从上层命令开始。

正确顺序应该是：

```text
底层运行事实源
  -> runtime contract schema
  -> model capability matrix
  -> execution profile
  -> artifact graph
  -> executor adapter / Browser typed action
  -> LimeCore catalog / policy
  -> @ / button / scene launch binding
  -> task index / replay / audit
```

固定原则：

1. `@` 命令是上层触发器，不是底层事实源。
2. Phase 0-6 不以 `@` 命令为主对象，只建设可被所有入口复用的底座。
3. Phase 7 才做 `@` / button / scene 的入口绑定。
4. `@配图`、`@浏览器`、`@读PDF` 等只作为验收样本，不作为架构起点。

## Phase 0：底层运行事实源盘点

当前落点：见 [runtime-fact-map.md](./runtime-fact-map.md)。

### 目标

先找清 Lime 已有的底层事实源和主链，不先盘点 `@` 命令，不改业务代码。

### 范围

盘点这些底层对象：

1. `session_id / thread_id / turn_id`
2. `task_id / run_id / content_id`
3. Agent turn context
4. Skill / Tool / ServiceSkill binding
5. model routing / `TaskProfile` / `CandidateModelSet` / `RoutingDecision`
6. permission / policy / OEM / tenant constraints
7. local artifact document / task file / media output
8. right viewer / workspace / image workbench / document viewer
9. evidence pack / replay / review / analysis export
10. Browser Assist action / observation / screenshot / DOM / network state
11. LimeCore bootstrap / `client/skills` / `client/scenes` / model catalog / Gateway policy

### 输出

新增 `runtime-fact-map.md` 或等价 JSON，字段至少包括：

1. `fact_source_key`
2. `owner_module`
3. `current_path`
4. `identity_keys`
5. `readers`
6. `writers`
7. `persistence`
8. `evidence_export`
9. `limecore_dependency`
10. `current / compat / deprecated / dead` 分类

### 验收

1. 能从底层事实源解释一次多模态运行，而不依赖 `@` 命令名称。
2. 每个 fact source 都有唯一 owner，不让 viewer、parser、runtime 同时写事实。
3. 找不到 current owner 的对象只能标 `compat` / `deprecated` / `dead`，不能继续承载新逻辑。

## Phase 1：ModalityRuntimeContract schema

当前落点：见 [contract-schema.md](./contract-schema.md) 与 `src/lib/governance/modalityRuntimeContracts.json`。

### 目标

把底层事实源收敛成可检查的 `ModalityRuntimeContract`，它服务所有上层入口。

### 改动面

优先新增：

1. `internal/roadmap/warp/contract-schema.md`
2. `internal/roadmap/warp/runtime-fact-map.md`
3. `src/lib/governance/` 下的轻量 schema 或检测脚本
4. `scripts/` 下的文档/JSON 一致性检查

暂不改 `@` 命令、按钮、Scene 的业务逻辑。

### Contract 字段

首版字段：

1. `contract_key`
2. `modality`
3. `runtime_identity`
4. `input_context_kinds`
5. `required_capabilities`
6. `permission_profile_keys`
7. `routing_slot`
8. `executor_binding`
9. `truth_source`
10. `artifact_kinds`
11. `viewer_surface`
12. `evidence_events`
13. `limecore_policy_refs`
14. `fallback_policy`
15. `detour_policy`
16. `owner_surface`
17. `bound_entries`

### 验收

1. Contract 不以 `@` 命令为主键，而以底层 modality runtime 能力为主键。
2. Contract 不允许引用不存在的 viewer、truth source、artifact kind 或 evidence event。
3. 新增多模态能力时必须先新增或复用 contract，再绑定上层入口。
4. `verify:local` 或 smart verify 能命中文档/contract 检查。

## Phase 2：模型能力矩阵接入

当前落点：见 [capability-matrix.md](./capability-matrix.md) 与 `src/lib/governance/modalityCapabilityMatrix.json`。

### 目标

让模型路由先理解底层模态能力需求，再服务具体入口。

### 改动面

优先接到现有 `internal/roadmap/task/` 主链：

1. `TaskProfile`
2. `CandidateModelSet`
3. `RoutingDecision`
4. `limit_state`
5. `thread_read`
6. evidence export

### 新增能力标签

首批：

1. `text_generation`
2. `vision_input`
3. `image_generation`
4. `image_edit`
5. `audio_transcription`
6. `voice_generation`
7. `browser_reasoning`
8. `browser_control_planning`
9. `web_search`
10. `local_file_read`
11. `structured_document_generation`
12. `long_context`
13. `cheap_summary`

### 验收

1. 任何要求 `image_generation` 的 contract 都不会路由到无图像生成能力的模型。
2. 任何要求本地文件读取的 contract 都会同时要求 `local_file_read` 能力和权限。
3. 任何要求浏览器操作的 contract 都会要求 `browser_reasoning` 与 `browser_control` 权限。
4. 只有一个候选模型时，仍输出 `single_candidate_only` 与能力说明。

## Phase 3：ModalityExecutionProfile

当前落点：见 [execution-profile.md](./execution-profile.md)、`src/lib/governance/modalityExecutionProfiles.json` 与 `src/lib/governance/modalityExecutionProfiles.ts`；最小 profile / executor adapter registry、前端 launch metadata、Rust runtime contract snapshot、Evidence / Replay、统一媒体任务索引快照、图片 / 视频 / 音频 / 转写 worker route execution、Browser Assist 真实动作前 preflight、`LimeSkillTool` current Skill 合同 metadata seed / 显式冲突合同阻断，以及 LimeCore policy refs/snapshot 种子已落地。`voice_generation` 与 `audio_transcription` 当前均是 `route_execution_status=executable` 的 current 合同，分别消费 `mediaTaskArtifact/audio/create` 与 `mediaTaskArtifact/transcription/create`，写入 `resolved_route` / `model_route_execution`；`pending_hit_refs` / `policy_value_hits` / `policy_value_hit_count` 已为真实 policy 命中值预留稳定接线，传入 `status=resolved` hit 时会把对应 ref 计入 `evaluated_refs` 并收缩 `missing_inputs / pending_hit_refs`；图片任务执行前已能从本地 model registry assessment 生成 `model_catalog` hit，并在进入真实执行器前从已解析的 runner config/API key 与 payload provider/model 生成 `provider_offer` hit；Browser Assist 与 Web Research 类 launch 已能从请求侧 `harness.oem_routing` 生成最小 `gateway_policy` hit，Workspace send metadata 也能从 OEM Cloud bootstrap `features` 生成最小 `tenant_feature_flags` hit；最小 `policy_input_evaluator` 已能在所有 refs resolved 时输出 `allow / ask / deny`。真实 `gateway:*` adapter preflight、同 turn 自动恢复/完整权限 GUI、用户锁定 gap 完整 GUI 自动恢复、云端 policy evaluator 与更完整可视化仍待继续。

### 目标

建立 Warp `AIExecutionProfile` 式的 Lime profile，把模型角色、权限策略、执行器策略合并到一个解释层。

### Profile 模块

首版包含：

1. model roles
2. permission roles
3. executor policy
4. artifact policy
5. LimeCore tenant override
6. user explicit lock
7. fallback behavior
8. audit / evidence requirement

### 模型角色

首批：

1. `base_model`
2. `vision_input_model`
3. `image_generation_model`
4. `image_edit_model`
5. `audio_transcription_model`
6. `voice_generation_model`
7. `browser_reasoning_model`
8. `report_generation_model`
9. `cheap_summary_model`

### 权限面

首批：

1. `read_files`
2. `write_artifacts`
3. `execute_commands`
4. `call_mcp`
5. `web_search`
6. `browser_control`
7. `media_upload`
8. `service_api_call`
9. `local_cli`
10. `ask_user_question`

### 验收

1. 用户禁用 `browser_control` 时，底层 browser contract 被阻断或询问，不会转成 WebSearch 假执行。
2. 租户禁止媒体上传时，图片/音频 contract 给出可解释降级。
3. 本地 explicit model lock 优先于自动优化，但能力不匹配时必须给出阻断原因。
4. Profile 决策写入 thread read 与 evidence，不只停留在设置页。
5. 每个 current contract 都必须被 `modalityExecutionProfiles.json` 覆盖，且 profile 的模型角色、权限、LimeCore policy、artifact policy 与 contract 对齐。

## Phase 4：领域化 Artifact Graph

当前落点：见 [artifact-graph.md](./artifact-graph.md) 与 `src/lib/governance/modalityArtifactGraph.json`。

### 目标

让多模态结果拥有领域 kind、恢复路径和 viewer 映射；上层入口只消费 artifact graph。

### 首批 artifact kind

1. `image_task`
2. `image_output`
3. `audio_task`
4. `audio_output`
5. `transcript`
6. `browser_session`
7. `browser_snapshot`
8. `pdf_extract`
9. `report_document`
10. `presentation_document`
11. `webpage_artifact`
12. `generic_file`

### 改动原则

1. `generic_file` 只做兜底。
2. 二进制图片输出不自动镜像成普通文件卡。
3. viewer 只消费 artifact graph，不自己猜类型。
4. evidence pack 导出 artifact kind、关联键与可查询索引；`browser_control` 当前索引事实源是 `snapshotIndex.browserActionIndex`。

### 验收

1. 图片结果不会出现“重复文件卡 + 点不开”。
2. 浏览器截图不会被当成普通图片生成结果。
3. PDF 抽取能按页码/引用恢复。
4. PPT / 网页 / 报告有各自 viewer 映射。
5. artifact 能回到原 `session/thread/turn/task/model routing/evidence`。

## Phase 5：Executor Adapter 与 Browser typed action

当前落点：见 [execution-profile.md](./execution-profile.md) 与 `src/lib/governance/modalityExecutionProfiles.json` 的 `executor_adapters`；最小 adapter registry 已覆盖 `image_generation`、`browser_control`、`pdf_extract`、`voice_generation`、`audio_transcription`、`web_research`、`text_transform`，前端 runtime contract snapshot、Rust runtime contract snapshot、Evidence / Replay 与统一媒体任务索引已携带 `executor_adapter` 与最小 LimeCore policy snapshot，其中默认 `policy_value_hits=[]` / `policy_value_hit_count=0` 明确不伪造真实 policy 命中。图片 / 视频 / 音频 / 转写 worker 已消费 `ResolvedModelRoute` 与对应的 App Server execution binding，并写回最小 `provider_offer` / provider diagnostics；Browser Assist 工具层也已在真实浏览器动作前消费同一 `browser_control` runtime contract 校验 profile / adapter / binding，失败时返回 `runtime_preflight` 工具错误并保留合同 metadata；通用 Skill 主链已由 `LimeSkillTool` 注入底层合同 metadata 并阻断显式冲突合同。Browser Assist 与 Web Research 类 launch 也会从 `harness.oem_routing` 写回最小 `gateway_policy` hit，Workspace send metadata 会从 OEM Cloud bootstrap `features` 写回最小 `tenant_feature_flags` hit，所有 refs resolved 时会由最小 `policy_input_evaluator` 折叠为 `allow / ask / deny`，并通过 thread read 与统一媒体任务索引暴露 evaluator explanation；真实 `gateway:*` adapter preflight 仍待继续，不能把通道 ingress 当成 executor adapter。

### 目标

先定义执行器边界，再允许上层入口调用它们；借鉴 Warp harness 和 computer use，但不把外部 CLI 作为 current 捷径。

### Executor contract

每个 executor 必须声明：

1. `executor_kind`
2. `binding_key`
3. `supported_contracts`
4. `supports_progress`
5. `supports_cancel`
6. `supports_resume`
7. `artifact_output_kinds`
8. `permission_requirements`
9. `credential_requirements`
10. `failure_mapping`

### Browser typed action

Browser Assist 必须收成：

1. action descriptor
2. permission profile check
3. screenshot / DOM / network observation
4. tool timeline 展示
5. replay / review / evidence 导出

### 验收

1. 浏览器操作不会退回普通 WebSearch。
2. 每次浏览器动作都有 action summary 与 observation。
3. 高风险动作按 profile 决定自动、询问或阻断。
4. evidence pack 能导出 browser trace，并在 `browserActionIndex` 汇总 action/session/URL/observation/screenshot；Harness evidence panel 能展示同一索引摘要，并能打开最小 `browser_replay_viewer`。
5. `local_cli` adapter 不支持 progress/resume/artifact 时必须显式标注，不可伪造 current 能力。
6. 每个可执行 current contract 的 `executor_binding` 必须能解析到 `executor_kind:binding_key` adapter，且 adapter 的 progress/cancel/resume/artifact 支持位、产物、权限与失败映射必须覆盖 contract；`route_execution_status=metadata_only` 的 compat contract 必须保留退出条件，不得补假 executor binding。

## Phase 6：LimeCore 目录与策略接线

当前落点：central runtime contract helper、前端 runtime contract resolver、Evidence Pack 与 `list_media_task_artifacts` 已携带 `limecore_policy_refs` 和最小本地默认 `limecore_policy_snapshot(status=local_defaults_evaluated, decision=allow, decision_source=local_default_policy, decision_scope=local_defaults_only, policy_inputs, missing_inputs, pending_hit_refs, policy_value_hits, policy_value_hit_count)`；这个 `allow` 只表示本地默认策略没有阻断 current 路由。默认 `policy_inputs` 是 `declared_only / limecore_pending` 输入清单，`pending_hit_refs` 指向等待真实值的 refs，`policy_value_hits=[]` / `policy_value_hit_count=0` 明确不伪造真实 tenant / provider / gateway 放行；如果已有 `status=resolved` 命中值，同一 seam 会把对应 ref 写入 `evaluated_refs`，将 input 标为 `resolved` 并使用 hit 的 `value_source`。当前已先把图片任务的本地 model registry assessment 接成 `model_catalog` hit producer，把已解析 runner config/API key 与 payload provider/model 接成最小 `provider_offer` hit producer，把请求侧 `harness.oem_routing` 接成 Browser Assist / Web Research 的最小 `gateway_policy` hit producer，并把 OEM Cloud bootstrap snapshot `features` 接成请求侧 `tenant_feature_flags` hit producer；`policy_evaluation` 会在所有 refs resolved 时用最小本地 `policy_input_evaluator` 折叠 `allow / ask / deny`，`thread_read.runtime_summary.limecorePolicy` 会把最近一次 runtime contract 的 policy decision explanation 投影给上层读取，`thread_read.runtime_summary.modalityRuntime` 会把同一合同的 profile / adapter / executor binding 摘要投影给上层读取，`lime_runtime.task_profile` 已开始携带同一合同的权限 profile、用户锁定策略摘要与 `routingSlot`，provider/model resolution 已用这些字段做最小模型能力 enforcement，`lime_runtime.permission_state` 已把权限 profile 声明推进成 runtime 可读摘要，并由 `AgentRuntimeThreadReadModel.permission_state`、Evidence Pack 与 Replay runtime facts 结构化暴露；当声明态需要确认时，runtime 事件流也会暴露 `permission_review` 状态，`confirmationStatus=not_requested` 明确说明尚未生成真实审批请求，且未 resolved 的 `requires_confirmation` 会在 prelude 后、模型执行前阻断 turn；Evidence / Replay 会把 `not_requested / requested` 未解决确认作为交付阻断事实；当同一 thread read 中已经存在真实 tool `ApprovalRequest` 时，读取模型会把 confirmation 派生为 `requested / resolved / denied` 并记录真实 request id；`list_media_task_artifacts.modality_runtime_contracts` 也会汇总 evaluation status / decision / decision source 与 blocking / ask / pending refs；配音/转写任务卡恢复层已把 input gap / deny / ask 投影为轻卡 meta，图片 viewer 与图片消息轻卡也会从 task artifact runtime contract 展示同一 policy evaluation meta。这些 hit、evaluator、thread read 摘要、TaskProfile 摘要、权限摘要、模型能力 gap、任务索引 explanation 与任务卡/viewer meta 都只解释已命中的控制面输入，不代表 LimeCore 云 run/poll 或云默认执行。

### 目标

把 LimeCore 作为云事实源接入 contract，而不是让它抢本地执行。

### LimeCore 负责

1. `client/skills`
2. `client/scenes`
3. `model-catalog`
4. Provider offer
5. Gateway runtime policy
6. Scene run policy
7. tenant feature flags
8. audit config

### Lime 负责

1. seeded/fallback catalog
2. 本地 ServiceSkill 执行
3. Browser Assist
4. artifact graph
5. evidence pack
6. viewer
7. 上层入口绑定

### 验收

1. 服务端目录缺失时，客户端 fallback 仍可用。
2. 服务端目录命中时，客户端不再维护第二份业务定义。
3. LimeCore policy 只影响可见性、默认值、权限、路由约束，不直接替本地执行。
4. Gateway call 和 Scene cloud run 作为显式执行面进入 audit。
5. Lime evidence 与 LimeCore audit 能通过关联键互相解释。

## Phase 7：上层入口绑定

当前落点：见 [entry-binding-inventory.md](./entry-binding-inventory.md) 与 `src/lib/governance/modalityRuntimeContracts.json` 的 `bound_entries[]`；机器守卫已要求 current contract 至少有一个 entry binding、entry key 全局唯一、`entry_source` 只引用同一 contract 下的入口、`launch_metadata_path` 停留在 `harness.*`，且 entry binding 不得携带 truth source / artifact / viewer / evidence / executor / routing 底层字段。`/scene-key` 当前明确保持 planned，等待 LimeCore `client_scenes / scene_policy`、Scene catalog fallback 与 Scene run audit 进入同一 contract 后再登记。

### 目标

在底层 contract、profile、artifact、executor、LimeCore policy 都存在后，再把 `@` 命令、按钮和 Scene 绑定到 contract。

### 绑定对象

首批只覆盖代表入口，不一次性扩张全量：

1. `@配图`
2. `@浏览器`
3. `@读PDF`
4. `@配音`
5. `@搜索`
6. `/scene-key`

### 绑定字段

每个入口只声明：

1. `entry_key`
2. `entry_kind`
3. `display_name`
4. `launch_metadata_path`
5. `contract_key`
6. `default_input_mapping`
7. `entry_visibility_policy`
8. `fallback_copy`

### 验收

1. 上层入口不得直接创建任务、写 artifact 或决定 viewer。
2. 上层入口不得绕过 model routing、permission profile 或 LimeCore policy。
3. `@` 命令只补 launch metadata，然后交给底层 contract 执行。
4. 找不到 contract 的入口只能标 `deprecated` 或 `dead`，不能继续扩展。

## Phase 8：任务索引与复盘

当前落点：见 [task-index-inventory.md](./task-index-inventory.md) 与 `src/lib/governance/modalityArtifactGraph.json` 的 `task_index_fields`；机器守卫已要求所有 current / partial artifact kind 至少携带 `task_id / contract_key / artifact_kind / status / created_at / updated_at`，且索引字段不得重复。媒体任务类 artifact 进一步要求 `entry_key / thread_id / turn_id / content_id / modality / skill_id / model_id / cost_state / limit_state / estimated_cost_class / limit_event_kind / quota_low / executor_kind / executor_binding_key / limecore_policy_snapshot_status`，`list_media_task_artifacts.modality_runtime_contracts` 已把 payload 中的 `entry_key / entry_source` 投影为 snapshot `entry_key` 与聚合 `entry_keys`，把 `thread_id / turn_id / content_id` 投影为 snapshot 身份锚点与聚合 `thread_ids / turn_ids / content_ids`，把 `modality / skill_id / model_id` 投影为 snapshot 查询字段与聚合 `modalities / skill_ids / model_ids`，把 `cost_state / limit_state`、`runtime_summary` 与 `task_profile` 中已有摘要投影为 `cost_states / limit_states / estimated_cost_classes / limit_event_kinds / quota_low_count`，并把 runtime contract executor binding 投影为聚合 `executor_kinds / executor_binding_keys`，用于按入口、运行身份、模态、技能、模型、成本/限额和执行器查询图片、音频与转写任务；Evidence Pack 也已新增 `modalityRuntimeContracts.snapshotIndex.taskIndex`，把 Browser / PDF / Web Research / Text Transform / Voice Service 等非媒体 runtime contract snapshot 的同组身份锚点、executor、成本/限额摘要归一到同一索引对象；Replay / grader 已消费该索引，`runtimeFacts.modalityTaskIndex`、suite tags、success criteria、blocking checks 与 `grader.md` 都会要求保留同一身份、executor 与 cost/limit 摘要；前端 `src/lib/agentRuntime/modalityTaskIndexPresentation.ts` 已把该索引转换为任务中心可复用的 facets / rows / filters，`HarnessTaskIndexSection` 消费同一查询模型展示“多模态任务索引”和内嵌“任务中心过滤列表”，按 entry / content / executor / cost / limit 过滤同一 rows；`HarnessStatusPanel` 只保留挂载面，`check-modality-runtime-contracts.mjs` 已用 `task index presentation guard` 固定 helper / section / panel 的 current 边界。policy snapshot 只稳定 status/decision 摘要，不复制完整云策略对象。后续如新增独立主任务中心入口，只能复用这套 rows，不能再建平行索引。

### 目标

建立可查询的多模态任务索引，服务任务中心、复盘、审计和客服诊断。

### 索引字段

1. `task_id`
2. `thread_id`
3. `turn_id`
4. `content_id`
5. `contract_key`
6. `entry_key`
7. `modality`
8. `skill_id`
9. `model_id`
10. `executor_kind`
11. `artifact_kind`
12. `status`
13. `cost_state`
14. `limit_state`
15. `estimated_cost_class`
16. `limit_event_kind`
17. `quota_low`
18. `limecore_policy_snapshot`
19. `created_at / updated_at`

### 验收

1. 能按 modality、contract、entry、executor、cost/limit、artifact kind 过滤任务。
2. 能从 artifact 回到原 turn。
3. 能从失败任务看到模型路由、权限、LimeCore policy 和 evidence。
4. 复盘能消费同一索引，不另建事实源。

## 最小推荐首刀

第一轮不要直接改 `@` 命令。

推荐首刀：

1. 新增 `runtime-fact-map.md`，先盘清底层事实源。
2. 新增 `contract-schema.md`，定义 `ModalityRuntimeContract`，但暂不绑定入口。
3. 补文档/JSON 校验脚本，先守住“contract 必须声明 truth source / artifact / evidence / viewer”。
4. 选择一个底层能力做贯通样本，例如 `image_generation` contract 或 `browser_control` contract。
5. 底层链路成立后，再把 `@配图` 或 `@浏览器` 作为上层入口绑定上来。

这样顺序更稳：先打地基，再接门面。
