# `internal/exec-plans`

本目录存放会影响开发执行的 versioned artifact：执行计划、进度日志、阻塞记录、迁移清单、技术债追踪。

## 放什么

- 多轮实现、迁移、治理任务的执行计划
- 与计划绑定的阶段进度、阻塞项、决策记录
- 需要持续小额偿还的技术债与退出条件

## 命名约定

- 专项计划：`<topic>-plan.md`
- 进度日志：`<topic>-progress.md`
- 常驻追踪：使用固定文件名，例如 `tech-debt-tracker.md`

## 使用规则

1. 计划不是一次性文档，推进状态变化时要同步更新
2. 会改变实现顺序、范围或回滚策略的决策，必须记录在这里或链接到这里
3. 清理类工作如果不能直接回挂路线图，应登记到 `tech-debt-tracker.md`
4. 被替代的计划不要悬空，保留跳转说明或归档指针
5. Agent / Runtime / Agent App / Skill / Managed Objective / Harness / GUI 主链改动应先填写 [Agent Verification Contract 模板](./templates/agent-verification-contract.md)，明确预算标签、current 主链、Happy Path、Evidence Layers、必跑命令和 Agent QC 场景映射；普通开发默认低 token，不默认跑 full qcloop / live Provider。

## 关联入口

- 路线图主线：`internal/roadmap/`
- 参考运行时主链总计划：`internal/exec-plans/upstream-runtime-alignment-plan.md`
- 参考运行时主链进度日志：`internal/exec-plans/upstream-runtime-alignment-progress.md`
- Refactor v2 全项目 Gate A/B 验收计划：`internal/exec-plans/project-gate-a-b-acceptance-plan.md`
- Lime v1.108.0 发布执行计划：`internal/exec-plans/release-v1.108.0-plan.md`
- Lime v1.109.0 发布执行计划：`internal/exec-plans/release-v1.109.0-plan.md`
- Lime v1.110.0 发布执行计划：`internal/exec-plans/release-v1.110.0-plan.md`
- Lime v1.111.0 发布执行计划：`internal/exec-plans/release-v1.111.0-plan.md`
- Lime v1.111.1 发布执行计划：`internal/exec-plans/release-v1.111.1-plan.md`
- Lime v1.114.0 发布执行计划：`internal/exec-plans/release-v1.114.0-plan.md`
- Lime v1.115.0 发布执行计划：`internal/exec-plans/release-v1.115.0-plan.md`
- Lime v1.116.0 发布执行计划：`internal/exec-plans/release-v1.116.0-plan.md`
- Lime v1.121.0 发布执行计划：`internal/exec-plans/release-v1.121.0-plan.md`
- Lime v1.128.0 发布执行计划：`internal/exec-plans/release-v1.128.0-plan.md`
- Lime v1.129.0 发布执行计划：`internal/exec-plans/release-v1.129.0-plan.md`
- Lime v1.130.0 发布执行计划：`internal/exec-plans/release-v1.130.0-plan.md`
- Lime v1.131.0 发布执行计划：`internal/exec-plans/release-v1.131.0-plan.md`
- Lime v1.132.0 发布执行计划：`internal/exec-plans/release-v1.132.0-plan.md`
- Lime v1.136.0 发布执行计划：`internal/exec-plans/release-v1.136.0-plan.md`
- Codex App GUI 对齐执行计划：`internal/exec-plans/codex-app-gui-alignment-plan.md`
- Codex reasoning 投影与通知漂移修复：`internal/exec-plans/codex-reasoning-projection-and-notification-drift-fix.md`
- Codex 对话兼容重构：`internal/exec-plans/codex-conversation-compat-refactor-plan.md`
- Codex 对齐 v1 并行协调：`internal/exec-plans/codex-alignment-v1-coordination-plan.md`
- Codex Orchestrator 完整对齐：`internal/exec-plans/orchestrator-complete-alignment-plan.md`
- Codex / Lime 存储对齐执行计划：`internal/exec-plans/codex-lime-storage-alignment-plan.md`
- Codex / Lime 存储一一对照账本：`internal/refactor/data/03-one-to-one-storage-alignment-plan.md`
- Refactor v2 测试体系第二期计划：`internal/exec-plans/refactor-v2-test-phase-2-plan.md`
- Agent Runtime 单一投影入口收口计划：`internal/exec-plans/agent-runtime-single-projection-entry-plan.md`
- Claw Streaming Rendering Codex 对齐重构计划：`internal/exec-plans/claw-streaming-rendering-codex-refactor-plan.md`
- Claw Trace 系统实施全过程计划：`internal/exec-plans/claw-trace-system-implementation-plan.md`
- 生产命令 current 迁移计划：`internal/exec-plans/production-command-current-migration-plan.md`
- Agent App uninstall current UI 进度：`internal/exec-plans/agent-app-uninstall-current-ui-progress.md`
- P16 Diagnostics current fail-closed 进度：`internal/exec-plans/p16-diagnostics-current-fail-closed-progress.md`
- 旧 Tauri wrapper 清理入口已合并到生产命令 current 迁移计划；历史 quick cleanup 队列和机械 inventory 文件已删除，不再作为 current 执行计划入口。
- Provider 模型能力 taxonomy 进度日志：`internal/exec-plans/provider-model-taxonomy-progress.md`
- Windows Lime Hub 模型路由修复：`internal/exec-plans/windows-runtime-model-route-repair-plan.md`
- Windows 模型目录与更新可靠性闭环：`internal/exec-plans/windows-model-catalog-and-updater-reliability.md`
- Lime 核心用户主流程 E2E 审计：`internal/exec-plans/core-user-flow-e2e-audit-2026-07-31.md`
- MCP 现代化进度：`internal/exec-plans/mcp-modernization-progress.md`
- Right Surface 统一承载实施进度：`internal/exec-plans/right-surface-implementation-progress.md`
- Soul Style 输出面收敛计划：`internal/exec-plans/soul-style-output-surface-convergence-plan.md`
- Browser Runtime / Right Surface 骨架实施计划：`internal/exec-plans/browser-runtime-right-surface-plan.md`
- Lime 多模态运行合同实施计划：`internal/exec-plans/multimodal-runtime-contract-plan.md`
- Approval HITL Decision Model 执行计划：`internal/exec-plans/approval-hitl-decision-model-plan.md`
- Agent Phase 6 Provider Reply Backend 执行计划：`internal/exec-plans/agent-phase6-provider-reply-backend-plan.md`
- 云端套餐与支付边界收口计划：`internal/exec-plans/cloud-commerce-user-center-boundary.md`
- `@` 命令本地执行纠偏计划：`internal/exec-plans/at-command-local-execution-alignment-plan.md`
- Agent QC 运营级测试体系执行计划：`internal/exec-plans/agent-qc-ops-testing-plan.md`
- Agent App v2 独立安装与 Runtime 底座拆分执行计划：`internal/exec-plans/agentapp-v2-standalone-runtime.md`
- Plugin v3 标准化与旧实现清退计划：`internal/exec-plans/plugin-v3-standardization-and-retirement-plan.md`
- AI 图层化设计实现计划：`internal/exec-plans/ai-layered-design-implementation-plan.md`
- 图片能力 feature-flag / extension-tool 执行计划：`internal/exec-plans/image-capability-feature-flag-extension-tool-plan.md`
- 图片能力 feature-flag / extension-tool 进度：`internal/exec-plans/image-capability-feature-flag-extension-tool-progress.md`
- 图片能力历史草案已清理，不再保留独立 refactor plan。
- Soul Style 输出面收敛执行计划：`internal/exec-plans/soul-style-output-surface-convergence-plan.md`
- Skill Forge P1A Capability Authoring 执行计划：`internal/exec-plans/skill-forge-capability-authoring-p1a-plan.md`
- Skill Forge P1B Capability Verification 执行计划：`internal/exec-plans/skill-forge-capability-verification-p1b-plan.md`
- Skill Forge P3 Capability Registration 执行计划：`internal/exec-plans/skill-forge-capability-registration-p3-plan.md`
- Skill Forge P3B Capability Discovery 执行计划：`internal/exec-plans/skill-forge-capability-discovery-p3b-plan.md`
- Skill Forge P3C Runtime Binding 执行计划：`internal/exec-plans/skill-forge-runtime-binding-p3c-plan.md`
- Skill Forge P3D Query Loop Metadata 执行计划：`internal/exec-plans/skill-forge-query-loop-metadata-p3d-plan.md`
- Skill Forge P3E Tool Runtime Authorization 执行计划：`internal/exec-plans/skill-forge-tool-runtime-authorization-p3e-plan.md`
- Skill Forge P4 Managed Execution / Agent Envelope 执行计划：`internal/exec-plans/skill-forge-managed-agent-envelope-p4-plan.md`
- Skill Forge P0-P4 完成审计：`internal/exec-plans/skill-forge-completion-audit.md`
- Skill Forge Prompt-to-Artifact P5 执行计划：`internal/exec-plans/skill-forge-prompt-to-artifact-p5-plan.md`
- Skill Forge Prompt-to-Artifact P5 样例审计：`internal/exec-plans/skill-forge-prompt-to-artifact-p5-audit.md`
- Skill Forge P6 Read-Only HTTP API 执行计划：`internal/exec-plans/skill-forge-readonly-http-api-p6-plan.md`
- Skill Forge P6 Read-Only HTTP API 完成审计：`internal/exec-plans/skill-forge-readonly-http-api-p6-audit.md`
- Skill Forge P7 Read-Only HTTP API 执行授权计划：`internal/exec-plans/skill-forge-readonly-http-api-p7-plan.md`
- Skill Forge P8 Read-Only HTTP API 注册 provenance 计划：`internal/exec-plans/skill-forge-readonly-http-api-p8-plan.md`
- Skill Forge P9 Read-Only HTTP API 授权 artifact 消费门禁计划：`internal/exec-plans/skill-forge-readonly-http-api-p9-plan.md`
- Skill Forge P10 Read-Only HTTP API completion audit 消费计划：`internal/exec-plans/skill-forge-readonly-http-api-p10-plan.md`
- Skill Prompt 执行与历史保留计划：`internal/exec-plans/skill-prompt-execution-retention-plan.md`
- LimeNext 总实施计划（`legacy current reference`，当前主规划已切到 `internal/roadmap/limenextv2/README.md`）：`internal/exec-plans/limenext-plan.md`
- LimeNext 推进日志：`internal/exec-plans/limenext-progress.md`
- 技术债追踪：`internal/exec-plans/tech-debt-tracker.md`
- 模块级实施细节：`internal/aiprompts/README.md`

## 当前命令迁移边界

- 生产命令 current 主链固定为 `Frontend -> Electron Desktop Host IPC -> App Server JSON-RPC -> RuntimeCore / services`。
- `lime-rs/src/commands/**` 是旧 Tauri command wrapper 删除清理区，只允许迁出核心逻辑、撤 runner / DevBridge dispatcher / catalog / mock 注册、机械编译修复和删除旧 wrapper。
- 新增 Rust 后端能力进入 App Server crates / RuntimeCore / services；桌面壳能力进入 Electron Desktop Host。
- 任何执行计划如果仍需要在 `lime-rs/src/commands/**` 新增业务逻辑、API adapter、runtime 分支、compat wrapper、fail-closed stub、tombstone 或 thin facade，必须先改计划并登记 blocker，不能把该目录当完成态。
- 前端 `src/lib/dev-bridge/**` 按职责治理：`safeInvoke`、HTTP client、`app_server_handle_json_lines`、bridge availability / event listener capability 是 current renderer bridge；旧命令 policy / no-mock fallback 是迁移期 `compat / deprecated`；已迁旧命令名只能作为 `dead` / `test-only` guard。后续计划清命令时必须同步检查 policy、mock、fallback、旧 smoke 和 contract guard，不得把整目录删除当作默认治理动作；删不动且跨命令组长期存在的 residual 必须回挂 `tech-debt-tracker.md` 的 `CCD-012`。
