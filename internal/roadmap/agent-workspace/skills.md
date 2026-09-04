# Agent Workspace Skills 评分卡

> 当前静态分：`3.6 / 5`  
> 更新时间：2026-06-15  
> 目标：把 Skill 作为 Agent Workspace 的独立核心能力评测，而不是把它折叠成 `Skill` 工具卡的一行 UI。这里评估的是“标准技能包如何进入产品、被发现、被选择、被授权、被执行、被回放、被持续改进”。

## 1. 为什么 Skill 必须独立评分

`Skill` 和 `Tool` 不是同一层对象。

| 对象 | 回答的问题 | 评测重点 |
| --- | --- | --- |
| Tool | 本轮 agent 可以调用什么函数，以及调用过程如何展示 | tool args、progress、result、error、permission、output offload |
| Skill Bundle | 一个可复用能力包里有哪些说明、资源、脚本、模板、依赖、触发语义 | 包结构、渐进披露、资源索引、触发准确率、依赖声明 |
| Skill Catalog / ServiceSkill / SceneSkill | Skill 进入产品后怎么展示、分组、补参、绑定运行时 | catalog 投影、slotSchema、readiness、home / mention / workspace surface |
| Runtime Binding | Skill 最终交给哪个执行面运行 | `agent_turn`、`browser_assist`、`automation_job`、`native_skill`、权限与 evidence |

因此 `tools-inventory.md` 里的 `Skill` 只评估 **Skill tool call rendering**：本次调用读取的 `SKILL.md` snapshot 是否可见、metadata 是否隐藏、tool process 是否可展开。完整 Skill 系统必须单独看 discovery、catalog、distribution、composer、runtime gate、artifact、security 和 evaluation。

## 2. 外部标准与调研约束

| 来源 | 本轮采用的标准 | 对 Lime 的约束 |
| --- | --- | --- |
| Codex 官方 Skills 文档 / Codex manual | Skill 是 reusable workflow authoring format，包含 instructions、resources、scripts；Codex 用 progressive disclosure；支持显式 `$skill` / 隐式匹配；插件是分发单位；`agents/openai.yaml` 可声明 UI metadata、implicit policy、tool dependencies | Lime 不能只把 Skill 当 prompt 字符串；必须展示来源、资源、依赖、触发策略、禁用/启用、插件或 package 来源 |
| Agent Skills standard | Skill 是包含 `SKILL.md` 的目录；至少有 `name` / `description`；metadata 先加载，full instructions 激活时加载，resources 按需加载；description 决定触发准确率 | 评分必须包含触发准确率、渐进加载、资源目录、description 过宽/过窄回归 |
| Claude Code Skills | Skills 用于扩展 Claude Code，可创建、管理、分享；subagent 可预加载 skills 作为领域知识 | Lime 需要把 Skill 和 subagent/team/profile 关联评测，不能只测主 agent |
| AG-UI | 前端应消费 run lifecycle、text、thinking、tool call、state snapshot/delta、messages snapshot、error | Skill 执行 UI 必须能把步骤、思考、工具、状态、结果按事件顺序回放 |
| Vercel AI SDK UIMessage | tool part 区分 `input-streaming`、`input-available`、`output-available`、`output-error`，并有 approval states | Skill 内部调用工具时，UI 不能只显示“技能运行中”；要有输入、审批、输出、失败状态 |
| MCP specification | MCP 把 tools、resources、prompts 分层；tools 有 schema，resources 有 URI，prompts 是 workflow 模板 | Skill 依赖 MCP 时要展示依赖健康、resource / prompt / tool 边界，而不是把 MCP 全部混成普通工具 |
| Google ADK eval | evalset 适合复杂多轮 agent integration tests；criteria 覆盖 tool trajectory、response quality、safety | Skill 评分必须有多轮场景、工具轨迹、最终质量和安全 verdict |
| Codex 本地源码 | `core-skills` 有 metadata / policy / dependencies / injection；`available_skills_instructions` 把可用 skill 注入上下文；app-server 有 `skills/list` 与 `skills/changed` 测试；TUI 有 skill popup / mention | Lime 需要同等层级的可见 skill catalog、变更通知、选择入口、注入证据和 telemetry |

参考链接：

- https://developers.openai.com/codex/skills
- https://agentskills.io/specification
- https://agentskills.io/client-implementation/adding-skills-support
- https://agentskills.io/skill-creation/optimizing-descriptions
- https://agentskills.io/skill-creation/best-practices
- https://docs.anthropic.com/en/docs/claude-code/skills
- https://docs.anthropic.com/en/docs/claude-code/sub-agents
- https://modelcontextprotocol.io/specification/2025-11-25
- https://adk.dev/evaluate/
- https://adk.dev/evaluate/criteria/

## 3. Lime Skill 事实源地图

| 层级 | Lime 事实源 | 当前结论 |
| --- | --- | --- |
| 标准 | `internal/aiprompts/skill-standard.md` | 已明确 Agent Skills 是唯一对齐的技能包格式标准；Skill 是 bundle；Lime 把 bundle 编译成 catalog / runtime binding |
| 本地默认包 | `lime-rs/resources/default-skills/**/SKILL.md` | 当前发现 `33` 个默认 `SKILL.md`，覆盖分析、调研、写作、媒体生成、知识构建、站点检索等 |
| App Server API | `packages/app-server-client/src/protocol.ts`、`src/lib/api/skills.ts` | current 方法覆盖 `skill/list`、`skill/read`、management、repository、cache、local inspect/scaffold/import/rename、remote inspect、package install/export、marketplace/download、workspace bindings |
| 管理 UI | `src/components/skills/**` | 有 Built-in / Local / Remote 分组、标准状态、资源摘要、metadata category、远程预检、本地详情、脚手架、仓库管理 |
| 产品投影 | `src/lib/api/skillCatalog.ts`、`src/lib/api/serviceSkills.ts` | 有 `SkillCatalogSkillEntry`、`CommandEntry`、`SceneEntry`、`ServiceSkillItem`、`slotSchema`、`readinessRequirements`、`skillBundle`、executor binding |
| Composer | `src/components/agent/chat/skill-selection/**`、`skillCommand.ts`、`runtimeInputCapabilityCatalog.ts` | 有 SkillSelector、mention / slash、scene command、skill id map；普通 slash skill 回到 Agent Runtime turn 主链 |
| Workspace binding | `workspaceSkillBindings/list`、`workspaceSkillBindingsMetadata.ts` | 有 workspace skill binding metadata 和 `ready_for_manual_enable`，但仍偏手动启用证据 |
| Runtime gate | `lime-rs/crates/agent/src/tools/skill_tool_gate.rs` | 有 session access gate、allowed skills、allowed skill sources、source metadata、allow / deny 测试 |
| Runtime 执行 | `lime-rs/crates/agent/src/skill_execution.rs` | 有 workflow / prompt execution、`skill-exec-*` runtime turn、allowed_tools 转发、TextDelta / runtime.status；回合终态由 App Server `turn.completed / turn.failed / turn.canceled` 收口 |
| Service / Site Skill | `ServiceSkillExecutionCard.tsx`、`useWorkspaceServiceSkillEntryActions.test.tsx`、`serviceSkillSceneLaunch.test.ts` | 有 site skill blocked/success/error、saved content、result file、browser prep、recent usage |
| Evidence | `scripts/agent-runtime/service-skill-entry-smoke.mjs`、`HarnessEvidencePackCard.tsx` | 有 SkillTool gate transcript proof 和 evidence count 字段，但 Skill 专项 evidence pack 仍未完整产品化 |
| Current boundary | `scripts/skills-current-smoke.mjs`、`src/lib/api/skills.current-boundary.test.ts` | 有 current App Server JSON-RPC smoke 和 legacy command 防回流守卫 |

## 4. 端到端能力链评分

| 能力链路 | 当前分 | 已有证据 | 主要缺口 | 5 分标准 |
| --- | ---: | --- | --- | --- |
| Skill 包标准与校验 | 4.0 | `skill-standard.md`、`ServiceSkillBundleSummary`、standard compliance、resource summary、33 个默认包 | 缺统一 CLI/GUI 的标准校验报告入口；重复/兼容包治理不足 | 每个包有 schema 校验、资源索引、依赖检查、标准/兼容/错误原因、自动修复建议 |
| Discovery / Catalog | 3.8 | `skill/list`、`skill/read`、`skillManagement/list`、Built-in / Local / Remote 分组 | catalog 与 runtime 可调用状态仍需同屏对齐；变更通知和 GUI restore 缺实测 | 用户看到的 skill、模型可调用的 skill、runtime allowlist 三者完全一致 |
| 安装 / 仓库 / 分发 | 3.6 | repository save/delete、cache refresh、local inspect/import/rename、package install/replace/export、marketplace/download install | 缺 package provenance、签名/校验、冲突版本 UI、失败恢复 GUI evidence | 支持安装前预检、版本冲突、来源可信度、回滚、离线包、审计日志 |
| 管理 UI | 3.7 | SkillsPage、SkillCard、SkillContentDialog、SkillScaffoldDialog、RepoManagerPanel、WorkflowProgress 测试 | 缺桌面真实 E2E；缺 package 依赖健康和运行态可用状态 | 一个页面能回答“哪里来的、标准吗、能运行吗、依赖齐吗、谁能用、上次何时成功” |
| Composer 选择与触发 | 3.3 | SkillSelector、mention/slash、runtime input capability catalog、`skillCommand` current 注释 | 隐式触发准确率无 eval；description 过宽/过窄无回归；required vs invoked 不够清晰 | 支持显式、隐式、推荐、预加载；能解释为什么触发/没触发；有误触发回归集 |
| Slot / Readiness / A2UI | 3.4 | `slotSchema`、`readinessRequirements`、site skill blocked/success/error 测试 | 缺统一 gate request -> A2UI -> resume 的产品证据 | 缺参、账号、浏览器、项目、权限都以结构化 gate 呈现并可恢复 |
| Runtime binding | 3.7 | current binding：`agent_turn`、`browser_assist`、`automation_job`、`native_skill`；`cloud_scene` compat 正规化 | binding 与 UI 卡、Evidence、automation 设置未完全一体化 | 每次执行都能看到 binding、执行位置、权限、owner、resume/retry/cancel |
| SkillTool gate / 权限 | 3.9 | session enable、allowed skill、allowed source、allow/deny Rust 测试、source metadata smoke | 供应链、依赖工具、secret、网络权限、脚本风险还未形成统一 UI | 默认 fail closed；每个 skill 的工具依赖、文件/网络/脚本权限和来源都可审计 |
| Runtime 执行过程 UI | 3.2 | `skill-exec-*` inline retention、WorkflowProgress、ToolCallDisplay Skill snapshot | Skill 内部步骤、子工具、资源读取、artifact 生成缺稳定 timeline | 像 AG-UI / AI SDK typed parts 一样展示 step、input、approval、tool、output、error |
| Artifact / Evidence / Replay | 3.1 | service skill result file、saved content、Harness evidence fields、SkillTool transcript proof | 缺 Skill 专项 evidence pack manifest 和 replay UI | 每次 skill 执行都有 prompt、skill version、resources hash、tools、artifacts、verdict |
| Subagent / Team 结合 | 3.0 | subagent profile 有 skills 字段；Claude Code 也支持 subagent skills 预加载 | Lime UI 未清晰展示哪个 agent 预加载了哪个 skill、是否调用 | Team roster 可见 skill scopes、preload、handoff、worker skill usage |
| Evaluation harness | 3.0 | `skills-current-smoke.mjs`、`service-skill-entry-smoke.mjs`、大量 unit/component tests | 缺 P0 evalset、触发准确率、真实 GUI、质量 grader | 每个核心 skill 有 evalset、trajectory、quality、safety、GUI evidence 和趋势分 |

## 5. 默认 Skill Inventory 静态盘点

| 分组 | Skill | 当前静态分 | 当前判断 | 必须补证 |
| --- | --- | ---: | --- | --- |
| 分析 / 调研 | `analysis` | 3.8 | Runtime contract 在 `SkillTool` gate 中有覆盖 | 真实分析任务 outcome、artifact、evidence pack |
| 分析 / 调研 | `research` | 3.8 | Runtime contract 有覆盖，适合 P0 eval | 来源引用、搜索轨迹、最终答案质量 grader |
| 分析 / 调研 | `report_generate` | 3.7 | Runtime contract 有覆盖，产物语义明确 | 报告 artifact、结构化评分、导出回放 |
| 分析 / 调研 | `summary` | 3.7 | Runtime contract 有覆盖 | 长文输入、大输出、引用保真 |
| 分析 / 调研 | `translation` | 3.7 | Runtime contract 有覆盖 | 双语质量、术语表、格式保真 |
| 文档 / 媒体输入 | `pdf_read` | 3.7 | Runtime contract 有覆盖 | PDF 页码引用、表格/图片失败分类 |
| 文档 / 媒体输入 | `transcription_generate` | 3.6 | Runtime contract 有覆盖 | 音频时间轴、说话人、导出证据 |
| 搜索 / 站点 | `site_search` | 3.6 | Runtime contract 有覆盖 | attached browser、来源截图、站点权限 |
| 搜索 / 站点 | `url_parse` | 3.2 | default skill + workbench task 名称存在 | URL 解析结果和最终引用一致 |
| 搜索 / 站点 | `modal_resource_search` | 3.2 | CLI/catalog 与 workbench task 存在 | 素材列表、来源、选择保存 |
| 内容生产 | `article-writer` | 3.2 | 默认包存在，业务价值高 | 写作质量 evalset、模板资源使用证据 |
| 内容生产 | `content-reviewer` | 3.1 | 默认包存在，适合 Reviewer 模式 | checklist、严重性、修复建议一致性 |
| 内容生产 | `content_post_with_cover` | 3.1 | 默认包存在，涉及多步产物 | 文案 + 封面 + 导出 pipeline evidence |
| 内容生产 | `broadcast_generate` | 3.2 | CLI/catalog 与 workbench task 存在 | 口播脚本、音频/字幕产物 |
| 内容生产 | `cover_generate` | 3.2 | CLI/catalog 与 cover task 存在 | 图片 artifact、版权/提示词保护 |
| 内容生产 | `image_generate` | 3.4 | CLI/catalog 与 image task 存在，UI 有图片 preview 基础 | prompt、任务状态、结果图、失败重试 |
| 内容生产 | `video_generate` | 2.4 | default skill 只允许 current typed `video_generate`，经 App Server 视频任务 owner 执行 | 保持 typed tool / worker / artifact 闭环 |
| 内容生产 | `presentation_generate` | 3.1 | CLI/catalog 存在 | PPT artifact、可编辑性、渲染校验 |
| 内容生产 | `form_generate` | 3.0 | CLI/catalog 存在 | 表单 schema、预览、导出 |
| 内容生产 | `webpage_generate` | 3.0 | CLI/catalog 存在 | 页面 artifact、浏览器预览、响应式检查 |
| 内容生产 | `typesetting` | 3.2 | CLI/catalog 与 workbench task 存在 | 排版前后 diff、导出、失败恢复 |
| 知识构建 | `knowledge-builder` | 2.8 | 默认包存在 | 与 `knowledge_builder` 的边界、输出位置、检索来源 |
| 知识构建 | `knowledge_builder` | 2.5 | 默认包存在，命名重复风险 | 合并/别名/兼容策略 |
| 知识构建 | `brand-persona-knowledge-builder` | 2.8 | 默认包存在 | 输入资料、知识库写回、版本证据 |
| 知识构建 | `brand-product-knowledge-builder` | 2.8 | 默认包存在 | 产品知识 schema、冲突处理 |
| 知识构建 | `campaign-operations-knowledge-builder` | 2.8 | 默认包存在 | 运营知识分类、复用效果 |
| 知识构建 | `content-operations-knowledge-builder` | 2.8 | 默认包存在 | 内容运营知识输出质量 |
| 知识构建 | `growth-strategy-knowledge-builder` | 2.8 | 默认包存在 | 策略模板、案例引用 |
| 知识构建 | `live-commerce-operations-knowledge-builder` | 2.8 | 默认包存在 | 直播电商场景 eval |
| 知识构建 | `organization-knowhow-knowledge-builder` | 2.8 | 默认包存在 | 组织知识权限与来源 |
| 知识构建 | `personal-ip-knowledge-builder` | 2.8 | 默认包存在 | 人设知识一致性 |
| 知识构建 | `private-domain-operations-knowledge-builder` | 2.8 | 默认包存在 | 私域运营知识回归 |
| 通用资源 | `library` | 2.6 | 默认包存在，边界较宽 | 是否应拆为资源索引 skill 或合并到 catalog |

静态结论：默认 Skill 的 **覆盖面已经很广**，但目前评分高低不是按“目录里有没有 `SKILL.md`”决定，而是按是否有 current runtime contract、产品投影、GUI evidence、artifact/evidence 和质量 eval。知识构建类数量多但缺少统一输出位置、质量 grader 和持久化证据，是后续最需要治理的区域。

## 6. P0 评测矩阵

| 场景 | 覆盖链路 | 必须证明 |
| --- | --- | --- |
| `skill-catalog-current-boundary` | App Server API / legacy guard | `skill/list`、management、repository、package、marketplace 走 current JSON-RPC；legacy command 不回流 |
| `skill-management-lifecycle` | 管理 UI / package | 安装前 inspection、scaffold、import、rename、export、replace、cache refresh、错误恢复 |
| `skill-explicit-invocation` | Composer / Runtime | `$skill` 或 slash/mention 选择后，进入 Agent Runtime turn，而不是 legacy execute path |
| `skill-implicit-trigger-eval` | Description / trigger | 正例触发、反例不触发、相似 skill 不误触发；记录 description 修改前后得分 |
| `skill-required-vs-invoked` | Runtime / UI | required skill、实际 invoked skill、读取的 `SKILL.md` snapshot、resources、tool dependencies 可查 |
| `skill-tool-gate-allow-deny` | Security / policy | 未启用默认拒绝；session allowlist 允许；source metadata、approval scope、permission summary 入 transcript |
| `skill-workflow-execution` | Runtime steps / UI | `skill-exec-*` inline process、step progress、TextDelta、runtime.status、current turn terminal、失败、取消、重试 |
| `skill-service-site-browser` | ServiceSkill / SiteSkill | 缺参 gate、attached browser、site run、saved content、result file、导出 Markdown |
| `skill-artifact-evidence-replay` | Artifact / Evidence | prompt、skill version、resources hash、allowed tools、tool calls、artifact refs、grader verdict 可回放 |
| `skill-subagent-preload` | Team / Subagent | subagent 预加载 skill、worker 调用 skill、handoff 结果和权限边界可见 |
| `skill-supply-chain-safety` | Distribution / Security | package 来源、版本、签名/校验、脚本/网络/文件权限、secret 泄漏扫描 |
| `skill-quality-evalset` | Evaluation | 每个 P0 skill 至少有 3 条多轮 evalset：成功、缺参、失败恢复 |

## 7. 失败模式分类

| 失败模式 | 现象 | 判定方式 | 应归属 |
| --- | --- | --- | --- |
| 触发失败 | 用户明确需要某 skill，但没有触发 | explicit / implicit trigger eval 失败 | Skill description / selector |
| 误触发 | 泛化描述导致无关 skill 被加载 | negative prompt 触发 | Skill metadata |
| 包不可读 | `SKILL.md` 缺字段、资源缺失、兼容字段错误 | inspection / standard compliance | Package standard |
| Catalog 漂移 | 管理页显示可用，但 runtime 不可调用 | `skill/list`、workspace binding、tool gate 不一致 | Catalog / binding |
| 权限绕过 | 未 enable session 仍能调用 SkillTool | gate deny 测试失败 | Runtime security |
| 资源黑盒 | Skill 执行引用了资源，但 UI / evidence 查不到 | transcript 缺 resource refs/hash | Evidence |
| 输出不可验证 | 最终说完成，但没有 artifact / file / browser state | grader 无客观证据 | Runtime / artifact |
| 兼容混淆 | `cloud_scene`、`lime_run_service_skill` 被当 current 能力 | current boundary guard 失败 | Governance |
| 供应链风险 | 远程包脚本、依赖、权限不透明 | package inspection 缺 provenance | Distribution / security |
| 恢复失败 | 中断 / 切换 session 后 skill 状态丢失 | session restore 场景失败 | Session / replay |

## 8. 下一刀

| 优先级 | 工作项 | 主线收益 |
| --- | --- | --- |
| P0 | 建立 `skill-catalog-current-boundary` 与 `skill-tool-gate-allow-deny` 固定 evidence pack | 先证明 Skill current 主链和权限边界不是假入口 |
| P0 | 给 `analysis`、`research`、`report_generate`、`pdf_read` 做 4 个 Skill P0 evalset | 覆盖最高价值且已有 runtime contract 的核心 skill |
| P0 | 在 Agent Workspace UI 中显式区分 required skill、invoked skill、loaded resources、allowed tools | 解决“前端表现影响后台输出”的关键可观察性 |
| P1 | 把 SkillsPage 增加运行态列：可调用、需手动 enable、依赖缺失、最近成功/失败 | 让管理页不只是包管理，而是 runtime readiness 面板 |
| P1 | 治理 `knowledge-builder` / `knowledge_builder` 与知识构建类输出目的地 | 降低 Skill 列表重复和产物不可追踪 |
| P1 | 增加 Skill package provenance / script permission / dependency health UI | 补齐安装分发与供应链安全 |
| P2 | 把 subagent skills preload、team usage 和 handoff 做成可观察 timeline | 对齐 Claude Code subagent skills 与团队协作能力 |

## 9. 结论

Lime 的 Skill 底座不是空白：标准文档、默认包、App Server current API、管理 UI、SkillCatalog / ServiceSkill 投影、composer 入口、runtime gate、workflow execution、smoke 和组件测试都已经存在。当前 `3.6 / 5` 的关键原因是：能力链已经成形，但还缺 **Skill 专项 GUI evidence、触发准确率 eval、artifact/evidence replay、供应链安全 UI、subagent skill 可观察性**。

要升到 `4.0+`，优先做 P0 evalset 和 evidence pack，而不是继续增加默认 Skill 数量。
