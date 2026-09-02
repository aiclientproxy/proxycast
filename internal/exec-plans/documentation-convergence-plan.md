# 全仓文档事实源收敛计划

状态：第七批完成 / current 文档、旧宿主残留与退役守卫已收口

## 1. 目标

把仓库文档收敛为可导航、可验证的 current 事实源。历史实现、已完成发布过程、被替代方案和失效索引不继续占用 active tree；需要追溯时使用 Git history、Release Notes 和不可变 evidence。

本计划不以“修改日期早”直接判死。只有同时具备替代 owner、无 current 执行责任、存在失效引用或明确历史属性的文档，才进入删除集。

## 2. 审计基线

审计日期：2026-09-02。

- Git 跟踪 Markdown：959 份。
- `internal/` 跟踪 Markdown：791 份，其中 `research` 293、`roadmap` 267、`exec-plans` 101。
- 全量 Markdown 本地链接初扫：53 个失效候选；第一批删除与导航收口后降至 28 个，明确断链与 current 引用修复后降至 20 个，第二批删除与引用迁移后降至 0 个。
- `harness:doc-freshness` 当前只监控 5 份文档，结果全绿不能代表全仓文档健康。
- 38 份文档仍直接引用 legacy catalog 中已物理删除的代码路径；历史 evidence 与负向守卫引用允许保留，current 导航引用必须移除。
- `internal/exec-plans/production-command-current-migration-plan.md` 与 `codex-alignment-v1-coordination-plan.md` 分别超过 5000 行，后续需要独立完成 active facts 提取后再裁剪，不能本批直接删除。

## 3. 分类

### current

- 根 `FEATURE-MAP.md`：仓库级能力导航，不承接第二套架构。
- `internal/aiprompts/architecture.md`、`commands.md`、`governance.md`、`quality-workflow.md`：当前架构、命令、治理和质量 owner。
- 当前版本 `internal/exec-plans/release-v1.139.0-plan.md`：记录本次发布候选、验证与远端复核。
- `internal/refactor/v2/IMPLEMENTATION-PLAN.md`：当前 Renderer / runtime 对齐执行入口。

### compat / deprecated

- 仍被 current 文档引用的历史术语只允许保留迁移说明，不继续作为导航目标。
- 超大执行计划在 active facts 提取前暂时保留；只能追加收口信息，不继续扩展第二套 owner。

### dead / deleted

1. `internal/exec-plans/release-v1.102.0-plan.md` 至 `release-v1.137.0-plan.md`，共 37 份。对应版本已过去，发布结果由 Git tag、GitHub Release、Release Notes 和 Git history 承接。
2. `internal/iteration-notes/` 全部 5 份。内容是旧凭证池、旧 session context 和阶段备忘，已由 current Provider、Memory/Compaction、执行计划与路线图取代。
3. `internal/develop/execution-tracker-*.md` 共 4 份，以及 `scheduler-task-governance-p1.md`。内容仍以 `heartbeat_executions`、`cron.*`、Tauri command 和旧 Settings 工作台为主，已被 Scheduled Tasks current owner 和 `task-agent-taxonomy.md` 取代。
4. `internal/research/refactor/v1/` 全部 15 份。该研究基线已被 `internal/research/refactor/v2/`、根架构图与 current 执行计划取代；目录内索引还引用 11 份已不存在的文档。

第一批已物理删除 62 份 Markdown；current 导航与直接引用已同步收口。

## 4. 第一批收口结果

已同步更新：

- `internal/README.md`：移除 `iteration-notes` 导航。
- `internal/exec-plans/README.md`：移除旧版本发布计划清单，明确 active tree 只保留当前发布计划。
- `internal/aiprompts/task-agent-taxonomy.md`：把 Execution Tracker 文档引用改为已删除裁决。
- `internal/develop/lime-borrow-codex-engineering-practices.md`：移除旧 Execution Tracker 文档导航。
- `internal/exec-plans/upstream-runtime-alignment-plan.md`：改指向 current taxonomy。
- `internal/roadmap/test/clawstream/README.md`、`internal/roadmap/test/README.md`：将 research/refactor/v1 invariant 改指向 current architecture/query-loop。
- `internal/roadmap/soul/README.md` 与 `internal/exec-plans/soul-style-output-surface-convergence-plan.md`：移除 research/refactor/v1 的 current owner 叙事。
- 与发布计划删除相关的索引只保留当前版本；历史 evidence 中的旧路径文字不因删除而改写。

本批不触碰现有 Rust、Renderer、Settings、协议、发布实现和并行执行计划业务内容。

## 5. 后续批次

第一批完成后继续审计，不直接批量删除：

1. `internal/roadmap/skill-forge/`：保留仍承接 current 进度的 `README.md` 与 `implementation-plan.md`；把稳定边界迁回这两个 owner 后，删除已被实现和 completion audit 取代的 `architecture-review.md`、`coding-agent-layer.md`、`diagrams.md`、`prototype.md`。
2. `internal/research/refactor/v2/13-evidence/`：先确认 current architecture 与 Gate B 索引不再依赖单篇 evidence 路径，再决定保留 manifest 还是从 active tree 移出。
3. 超过 1000 行的执行计划与路线图：先提取未完成 blocker 和 current decision，再删除完成日志。
4. `internal/aiprompts/converter.md`、`server.md`、`src/components/general-chat/README.md`、`src/i18n/README.md` 等断链模块文档：有 current module 的重写为短导航，无 current owner 的直接删除。
5. 把全仓相对链接和 deleted-surface current 引用纳入文档门禁，避免继续只监控 5 份文档。
6. 已删除 Plugin v3 明确裁决为 `dead / deleted / forbidden-to-restore` 的四份 Plugin Runtime v2 文档：`internal/roadmap/agentruntime/app-surface-runtime.md`、`backend-surface-facade-plan.md`、`claw-capability-sharing.md`、`plugin-runtime-completion-audit.md`；AgentRuntime 导航已改到 `internal/roadmap/plugin/v3/` current owner。

第二批已完成：删除上述 4 份 Plugin Runtime v2 文档，以及 `internal/roadmap/skill-forge/architecture-review.md`、`coding-agent-layer.md`、`diagrams.md`、`prototype.md` 4 份已被实现与 completion audit 取代的 proposal 文档。Skill Forge 稳定边界已回到保留的 `README.md`、`implementation-plan.md`、`internal/aiprompts/skill-standard.md` 与 `query-loop.md`。

第三批已完成：保留 `internal/roadmap/plugin/README.md` 与 `internal/roadmap/plugin/v3/**`，删除其余 23 份 Plugin v1/v2 历史材料：

- 根部旧文档与证据：`architecture.md`、`prd.md`、`implementation-plan.md`、`interface-contracts.md`、`technical-baseline.md`、`prototype.md`、`prototype.html`、`history-product-workspace.md`、`user-operations-guide.md`、`e2e-evidence.md`、`evidence/plugin-productization-e2e-summary.json`。
- 旧发布规划：`deverlop/plugin-publish-center-prd.md`、`deverlop/plugin-publish-limecore-server-plan.md`。
- Plugin v2 全部 10 份：`v2/README.md`、`00-research-findings.md`、`01-product-contract.md`、`02-package-marketplace-installation.md`、`03-architecture-and-command-contracts.md`、`04-app-center-and-claw-surfaces.md`、`05-migration-and-cleanup.md`、`06-implementation-plan.md`、`07-verification-contract.md`、`08-legacy-synthesis.md`。
- 同批删除 `internal/roadmap/agentruntime/agentruntime-standard-adoption-gap.md`；该文档仍以 `plugin_runtime_*`、`lime.agent / lime.workflow` 与 Plugin v2 worker 叙述旧主链，current 边界已由 Plugin v3、RuntimeCore、Skills、MCP 与 Tool Runtime 承接。

第三批共 24 份文件、8,517 行。已同步收短 Plugin 根导航，把 Agent Workbench 的 Adoption Gap 引用迁到 AgentRuntime README 与 Plugin v3，并将旧版本有效决策映射写入 Plugin v3 README、目标合同、基线与清理账本；文档边界守卫已切换到 v3 现役文件。

## 6. 验证与退出条件

第一批退出条件：

- 删除集全部物理移除，导航不再引用。
- `git diff --check` 通过。
- 全仓 Markdown 相对链接扫描无新增失效项，已知失效项数量下降。
- `npm run docs:boundary` 通过。
- `npm run harness:doc-freshness` 通过。
- `npm run governance:legacy-report` 通过，或明确记录仅由并行工作树造成的外部失败。

第一批验证结果：`git diff --check`、`npm run docs:boundary`、`npm run harness:doc-freshness`、`npm run governance:legacy-report` 全部通过；治理报告为 0 个边界违规。`harness:doc-freshness` 仍只监控 5 份文档，不能替代全仓链接扫描。

第二批验证结果：全仓本地链接扫描为 0 个失效项；`git diff --check`、`npm run docs:boundary`、`npm run harness:doc-freshness`、`npm run governance:legacy-report` 再次全部通过，治理报告为 0 个边界违规。

完成度：审计、前两批分类、删除与导航收口 100%；累计删除 70 份 Markdown，现存 889 份，全仓本地链接扫描为 0 个失效项；第三批内容时效性审计进行中。

第三批验证结果：`git diff --check`、全仓 Markdown 相对链接扫描、`npm run docs:boundary`、`npm run harness:doc-freshness`、`npm run governance:legacy-report` 均通过；Plugin v3 现役文档已补版本替换映射，根 Feature Map 已链接到 v3 current owner。

完成度：三批删除与版本替换收口 100%；累计删除 92 份 Markdown 与 2 份 HTML/JSON 材料，现存跟踪 Markdown 867 份，全仓本地链接扫描为 0 个失效项。后续仅处理有明确 current owner、具体替换目标和独立验证证据的文档，不按修改日期批量删除。

## 7. 第四批：现役版本替换导航

本批不做物理删除，先把“旧版本仍可打开”与“旧版本仍是 current owner”区分开，避免用户按旧入口继续扩展：

- 图片能力：`internal/roadmap/images/README.md` 降为 `legacy current reference`；current 背景、合同、架构和流程统一指向 `internal/roadmap/images/v2/**`。
- Writing：`internal/roadmap/Writing/**` v1 文档降为 `legacy current reference`；有效决策统一指向 `internal/roadmap/Writing/v2/**` 与 `internal/exec-plans/writing-v2-workflow-completion-plan.md`。
- Feature Map 新增版本替换索引，明确 Knowledge、LimeNext、Plugin 和 Refactor 的替换关系；其中 `internal/refactor/v1` 与 v2 明确标为不同 owner，不按版本号误删。
- 图片执行计划与进度记录已改为引用 v2 路线图；Writing v1 文档的 current 进度和验收入口均改为 v2 入口。
- LimeNext v1/v2 文档已统一降为 `legacy historical reference`；当前能力导航改由根 `FEATURE-MAP.md`、`internal/aiprompts/*` 与领域 v2/v3 路线图承接。
- 修复 `limenext/` 旧 README、PRD 与迁移图指向已删除 `internal/roadmap/ribbi/*` 的断链，并将 `limenext-plan.md` 明确为历史执行记录。
- Writing v1 的重复 `In Progress` 状态已改为历史参考，`article-frame-fix-plan.md` 与 `release-v1.100.0.md` 已补后续替代入口。

本批已完成现役版本替换导航；无 current 责任的旧版本正文、LimeNext 临时镜像与旧发布计划在后续批次按明确替代 owner 继续物理删除。

## 8. 第五批：LimeNext 历史化与旧入口替换

本批已完成文档替换但不做物理删除：

1. `internal/roadmap/limenext/` 与 `internal/roadmap/limenextv2/` 顶部均明确为 `legacy historical reference`，不再把 SceneApp 页面、命令或 `src-tauri` 路径写成 current。
2. 根 `FEATURE-MAP.md` 新增 LimeNext v1/v2 到现行架构与领域路线图的替代关系；后续能力查询不再进入 LimeNext 旧目录。
3. `internal/roadmap/Writing/` v1 文档和旧发布执行记录的重复 active 状态已清除，保留历史决策并指向 current v2 与根 Feature Map。

本批确认并删除：

- `internal/exec-plans/limenext-progress-from-finder.md`
- `internal/exec-plans/limenext-progress-from-finder-edit.md`
- `internal/exec-plans/release-v1.100.0.md`

上述文件均已有 current 替代入口或明确属于历史镜像，现已物理删除。`limenext-v2-fs-blocker-2026-04-22.md` 仍记录独立文件系统阻塞事实，不在本批删除范围。

## 9. 第六批：SceneApp 与 src-tauri 全面退役

用户已于 2026-09-02 明确确认删除 `src-tauri` 与 `SceneApp` 相关文档、代码和脚本。本批事实源声明：

`SceneApp` 与 `src-tauri` 均为 `dead / deleted / forbidden-to-restore`；应用目录归 Plugin current owner，任务执行归 App Server JSON-RPC + RuntimeCore + Agent Runtime，结果参考归通用 Curated Task / Memory reference owner。旧名称只允许出现在负向回流守卫及其测试中。

本批写集：

1. 删除 `src/lib/agent/legacySceneAppExecutionSummary*`、Agent Chat 中 `sceneApp*` / `SceneApp*` 执行摘要、复核、脚手架、内容产物与相关测试。
2. 从页面参数、Workspace bootstrap、Artifact surface 和 conversation landing surface 移除 `initialSceneAppExecutionSummary`、SceneApp summary card 与 review dialog 接线。
3. 将仍有 current 价值的“参考条目结果基线解析 / Curated Task continuation”收回 `curatedTaskReferenceSelection` 与通用 follow-up owner，并删除 `sceneapp_execution_summary` source kind。
4. 删除 SceneApp 专属五语言文案与正向测试夹具；更新受影响的组件和测试命名，不保留 SceneApp 兼容字段。
5. 删除 SceneApp 专题路线图与旧 Tauri wrapper 删除日志；混合 current 文档改写为 Plugin、Agent Runtime、Artifact、Memory 与 Electron/App Server 当前事实。
6. 将负向回流防护集中到 `legacySurfaceCatalog` 与结构测试，删除 ESLint 中仍把 SceneApp 表述为 compat/current 的规则、例外和目录 override。

退出条件：

- `src-tauri` 与 SceneApp 生产实现、正向测试、脚本和 active 文档为零；旧词仅存在于 `dead / forbidden-to-restore` 守卫或不可变历史 evidence。
- TypeScript 类型检查、受影响定向测试、i18n 检查、`npm run test:contracts`、`npm run governance:legacy-report`、`npm run docs:boundary` 与 `git diff --check` 通过。
- GUI 主路径至少完成 `npm run verify:gui-smoke`；若被工作区无关改动阻断，记录准确 blocker，不把 Gate A/B 误报为通过。

第六批结果：

1. `src-tauri` 与 SceneApp 命名的现存文件和目录均为零；15 份 Agent Chat / Agent lib 生产实现和正向测试已物理删除，通用 Artifact、Evidence、Curated Task 与 Memory owner 已接回有效职责。
2. 30 份复制旧后端结构的功能级 `code-structure.md` 已删除，命令运行时代码地图改为共享 current owner；混合文档已改写为 Plugin、Electron Desktop Host、App Server JSON-RPC、RuntimeCore / Agent Runtime 和 Memory reference。
3. `governance/surfaces.yml` 不再把不存在的 Tauri 入口标为 current，文件大小基线不再登记已删除组件；本地 release 指令已替换为 Lime 版本事实源与 Electron Forge 流程。
4. App Server contract 不再强制保留 P3.135/P3.137 历史流水账，改为验证 current 执行计划和测试口径中的唯一主链、生产 mock 零回退与 Gate B 边界。
5. 排除负向治理守卫、退役裁决、执行记录和不可变 evidence 后，全仓 `SceneApp|src-tauri` 内容扫描为零；历史 evidence 不作为 current owner 或导航入口。

验证结果：

- 定向 Vitest：6 files / 240 tests 通过。
- `npm run typecheck`、`npm run i18n:check` 通过；五语言覆盖 100%，missing / extra 均为 0。
- `npm run test:contracts` 通过；App Server client 299 checks、command catalog、modality、scripts、release workflow、cleanup 与 docs boundary 均通过。
- `npm run governance:legacy-report` 通过；零引用候选、分类漂移、边界违规均为 0。
- `npm run verify:gui-smoke` 通过；真实 Electron main/preload、App Server sidecar、Claw shell reload、3 个响应式 viewport 与 Memory settings ready。

第六批完成度：100%。

## 10. 第七批：旧宿主构建残留与发布收口

1. 删除 `lime-rs/capabilities/**` 的 Tauri v2 权限文件及本地生成 schema；Electron/Forge 是唯一 Desktop Host 与打包事实源。
2. 删除资源管理器生产路径对测试专用 `webviewWindow` 夹具的动态依赖，窗口关闭直接使用平台窗口语义。
3. 删除服务端 Tauri origin CORS 白名单，Rust current crate 注释统一改为 App Server / Desktop Host 或宿主无关表述。
4. i18n app metadata 审计移除已退役 capability 文件，scope、manifest、evidence 与测试改为 9 个 current 元数据字段。
5. 删除根目录 4 个未跟踪构建碎片；发布候选只包含源码、文档、测试、schema 与明确生成证据。

第七批退出条件：`npm run docs:boundary`、`npm run i18n:check`、`npm run test:contracts`、`npm run governance:legacy-report`、`npm run typecheck`、Rust 相关测试、`npm run verify:gui-smoke` 与 `git diff --check` 全部通过；随后发布 `v1.139.0`。

第七批结果：

1. `src-tauri`、SceneApp 命名的实体路径以及 `lime-rs/capabilities`、`lime-rs/gen/schemas` 空目录均已清零；旧名称只保留在负向回流守卫、退役裁决和不可变历史 evidence。
2. 删除无导航、无消费者且仍以 `agent_messages` / Tauri Commands 为事实源的 `internal/design/a2ui-persistence.md`；Chrome 扩展的多 Profile 说明改为 current `profileKey` 配置，不再引用已删除命令。
3. current 工程导航、App Server 架构/PRD/路线图和 AgentUI 代码地图统一声明 `lime-rs/src/commands/**` 已物理删除，不再把它描述为待清理的 compat 区域。
4. 本次累计删除 98 份 Markdown、1 份 HTML、4 份 JSON 和 16 份旧前端源码/测试；文档本地链接扫描、文档边界、时效性和 legacy 治理守卫均保持通过。

第七批完成度：100%；文档与旧宿主清理已完成，进入 `v1.139.0` 发布阶段。
