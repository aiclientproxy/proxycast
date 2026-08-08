# 测试分层治理执行进度

> 补充执行记录：`internal/roadmap/test/README.md` 已回填最新汇总；本文件继续保留更细的验证证据和临时执行上下文。

## 目标

- 将 Lime 前后端测试拆成纯单元、集成和 E2E / live-gated 层。
- 本地和 AI TDD 默认只跑纯单元测试。
- 前端复杂 UI / hook 逻辑优先抽到 View Model / projection / selector / helper，并用 `*.unit.test.ts` 覆盖；组件测试只保留渲染、事件接线和关键 UI 回归。
- 后端 Rust 单元测试默认不触发真实 Provider / ASR / live 网络路径。

## 已完成事实

- 前端分层入口已可用：`npm run test:unit`、`test:component`、`test:contract`、`test:integration`、`test:e2e`、`test:layers:stats`、`test:frontend:all`。
- Rust 分层入口已可用：`npm run test:rust:unit`、`test:rust:integration`、`test:rust:e2e`、`test:rust:layers:stats`。
- 测试分层规则已写入 `AGENTS.md` 和 `internal/aiprompts/quality-workflow.md`。
- CI quick gate 已接入分层入口：
  - PR 前端改动跑 `npm run test:unit` + `npm run test:contract`。
  - PR / main 前端验证在单测前运行 `node scripts/check-vitest-layer-budget.mjs --max-component-candidates 37`，防止 component VM 迁移候选数回升。
  - PR Rust 改动跑 `npm run test:rust:unit`。
  - main / 手动触发仍保留全量前端与 Rust 验证。
- 本地 Makefile 已接入分层入口：
  - `make tdd` 委托 `npm run test:unit`，作为本地 / AI TDD 的前端第一信号。
  - `make tdd-file FILE=...` 委托 `npm run test:unit -- <file>`，作为本地 / AI TDD 的单文件快速回路；传错层文件会被 Vitest 分层 runner 拦截。
  - `make tdd-rust` 委托 `npm run test:rust:unit`，作为后端第一信号。
  - `make test-layer-stats` 同时输出前端和 Rust 分层统计。
- 新增 component VM 迁移候选预算检查：
  - `scripts/check-vitest-layer-budget.mjs` 默认以 37 个候选为当前发布基线预算。
  - `make test-layer-budget` 会在候选数超过预算时失败，用于防止新 component 大测试回流。
- 新增 Rust E2E 默认运行预算检查：
  - `scripts/check-rust-layer-budget.mjs` 默认要求 Rust E2E 层文件不存在非 `ignore` 测试。
  - `make test-rust-layer-budget` 和 CI `Rust Quick` 在 `npm run test:rust:unit` 前执行该检查，防止 E2E / live-gated Rust 测试回流到 TDD 默认路径。
  - 该检查不把 unit 文件内“普通单测 + ignored live 测试”的混合结构当成失败；此类文件仍可运行普通单测，但 live 测试必须继续由 `#[ignore]` / `LIME_REAL_API_TEST=1` 门禁控制。
  - `scripts/rust-test-layer-classifier.test.mjs` 已补 mixed unit/live fixture，锁住该分类边界，避免后续把含 ignored live 的 unit 文件误迁到 E2E 或默认失败预算。
- 测试文件收集边界已补强：
  - `scripts/lib/vitest-test-file-filter.mjs` 排除 `*.testFixtures.*` / `*.test-fixture.*` 这类测试夹具支持文件。
  - `scripts/run-vitest-layer.mjs` 和 `scripts/lib/vitest-layer-report.mjs` 都复用该过滤边界，避免 fixture 被当成真实 component 测试运行或统计。
- Vitest 分层 runner 已补错层过滤器保护：
  - 显式传入测试文件时，`scripts/run-vitest-layer.mjs` 会校验每个过滤器至少命中当前层。
  - 如果文件只属于其他层或完全没有命中可运行测试，runner 直接失败并打印实际命中层，避免 AI / 本地 TDD 把“错层文件被跳过”误判为已验证。
- `HarnessStatusPanel` 已完成第一轮拆分：
  - `HarnessStatusPanel.tsx` 约 476 行。
  - `harnessStatusPanelViewModel.ts` 约 378 行。
  - 新增 `harnessEvidenceViewModel.ts`，承载 evidence / replay / review / LimeCore policy 纯逻辑。
  - 新增 `harnessEvidenceViewModel.unit.test.ts`，把 evidence 相关断言从总 VM 单测拆出。
  - `HarnessStatusPanel.test.tsx` 从巨型单文件拆成 focused component 测试集和共享 fixture，主文件约 300 行。

## 当前验证证据

2026-06-03 当前已验证：

- `npm run test:layers:stats`
  - Total files: 1012
  - Runnable by default: 1011
  - Live-gated: 1
  - unit: 485
  - component: 397
  - contract: 78
  - integration: 51
  - e2e: 1
  - Component unit-migration candidates: 8
  - 2026-06-24 发布门禁基线：当时候选数为 22，CI / Makefile / 脚本默认预算临时统一为 22；后续继续按 P2 分层治理把预算降回 12 / 8 或更低。
  - 2026-06-30 发版门禁复核：`v1.82.0` 与 `v1.83.0` 干净 worktree 均为 30 个候选，CI / Makefile / 脚本默认预算临时统一为 30，先恢复 main 发布门禁一致性，后续继续把预算降回 22 / 12 / 8。
  - 2026-08-07 `v1.123.0` Quality 复核：当前候选数为 37，CI / Makefile / 脚本默认预算同步为 37；后续继续按 P2 分层治理降低预算。
  - 注：该统计包含同轮并行新增的 `src/components/agent/chat/utils/toolNameFamily.unit.test.ts`，本轮未接管该写集。
- `npm run test:unit -- src/components/agent/chat/components/harnessStatusPanelViewModel.unit.test.ts src/components/agent/chat/components/harnessEvidenceViewModel.unit.test.ts`
  - 2 files / 24 tests passed
- `npm run test:unit`
  - 482 files / 2841 tests passed。
  - `real 138.63`，Vitest summary `Duration 136.09s`。
  - 前端 unit 默认入口已恢复绿色，unit runner 已取消默认 `singleFork` 且默认 `--environment node`；相比调整前 `real 213.05` 快约 `74s`。
  - Vitest summary 中 `environment 241ms`，全层 jsdom 环境成本已从 unit 移除。
- `npm run test:unit`（unit 默认 threads pool 后）
  - 483 files / 2851 tests passed。
  - `real 161.62`，Vitest summary `Duration 158.34s`。
  - 该轮说明默认入口继续绿色，但 wall time 受连续全量运行 / 并行写集影响明显，暂不把 `161.62s` 视为稳定退步或最终基准。
- `npm run test:unit`（unit 默认 threads pool 后再次复测）
  - 484 files / 2858 tests passed。
  - `real 293.89`，Vitest summary `Duration 272.93s`。
  - 当时机器负载后续确认为 `load averages: 95.67 82.56 71.54`，且存在多个高 CPU 进程；该轮不作为默认 threads 的稳定退步证据。
- `npm run test:unit`（临时取消显式 pool 后同一高负载复测）
  - 484 files / 2859 tests passed。
  - `real 366.78`，Vitest summary `Duration 359.21s`。
  - 该轮比高负载下的默认 threads 更慢，说明当前慢因主要是机器负载 / 全层 transform / collect 放大，而不是 threads 默认本身；runner 已恢复默认 Node + threads。
- `/usr/bin/time -p npm run test:unit -- --pool=threads`
  - 483 files / 2848 tests passed。
  - `real 115.59`，Vitest summary `Duration 113.07s`。
  - 该轮是 threads 作为速度优化的最佳证据；比 `138.63s` 快约 `23s`，但需在工作树安静后复跑确认稳定性。
- `/usr/bin/time -p npm run test:unit -- --pool=forks`
  - 484 files / 2858 tests passed。
  - `real 223.28`，Vitest summary `Duration 219.63s`。
  - 该轮包含并行新增 `toolNameFamily.unit.test.ts`；只用于证明同一阶段显式 forks 明显慢于 threads，不作为历史性能基准。
- 直接调用同一批 unit 文件、去掉 `scripts/run-vitest-layer.mjs` 当前强制的 `--poolOptions.forks.singleFork`
  - 486 files / 2849 tests passed。
  - `elapsed=175.46s`，Vitest summary `Duration 174.09s`。
  - 汇总块里 `environment 525.18s`、`collect 249.42s`，说明当前耗时主要不是断言执行，而是全层 jsdom 环境和模块收集 / 导入成本。
  - 结论：`test:unit` 默认单 fork 是高收益速度问题；本轮已改为 unit 默认并行，真实入口复测为 `real 183.71`。
- `npm run test:unit -- scripts/run-vitest-layer.unit.test.mjs`
  - 1 file / 15 tests passed。
  - 覆盖 `--single-fork` 参数解析、unit 默认不强制 single fork、unit 默认 Node 环境、unit 默认 threads pool、`LIME_VITEST_UNIT_POOL` 覆盖、显式 `--pool` / `--environment` 覆盖、非 unit 默认保留 single fork、以及 `LIME_VITEST_SINGLE_FORK=1` 回退。
- `npx eslint scripts/run-vitest-layer.mjs scripts/run-vitest-layer.unit.test.mjs --max-warnings 0`
  - 通过。
- `npm run test:unit -- src/components/agent/chat/workspace/workspaceModelSkillLaunchRequestContext.unit.test.ts`
  - 1 file / 7 tests passed。
  - 新增 current helper 覆盖 model skill launch request metadata 合并、session binding、播报、素材、转写、文本转换、URL、排版、网页、PPT 和表单 request context。
- `npm run test:component -- src/components/agent/chat/workspace/useWorkspaceSendActions.test.tsx`
  - 1 file / 135 tests passed。
  - 抽取 helper 后 hook 接线保持绿色；该 component 文件仍是 migration candidate。
- `npx eslint scripts/run-vitest-layer.mjs scripts/run-vitest-layer.unit.test.mjs src/components/agent/chat/workspace/useWorkspaceSendActions.ts src/components/agent/chat/workspace/workspaceModelSkillLaunchRequestContext.ts src/components/agent/chat/workspace/workspaceModelSkillLaunchRequestContext.unit.test.ts --max-warnings 0`
  - 通过。
- `git diff --check -- scripts/run-vitest-layer.mjs scripts/run-vitest-layer.unit.test.mjs src/components/agent/chat/workspace/useWorkspaceSendActions.ts src/components/agent/chat/workspace/workspaceModelSkillLaunchRequestContext.ts src/components/agent/chat/workspace/workspaceModelSkillLaunchRequestContext.unit.test.ts`
  - 通过。
- 直接调用同一批 unit 文件、去掉 single fork 并强制 `--environment node`
  - 482 files passed / 486 files total，2843 passed / 2849 tests total。
  - `elapsed=137.44s`，Vitest summary `Duration 136.34s`，`environment 335ms`。
  - 失败集中在 `4` 个仍依赖浏览器 / 调度环境的 unit 文件：`src/lib/crashReporting.test.ts`、`src/lib/workspaceHealthTelemetry.test.ts`、`src/lib/utils/scheduleMinimumDelayIdleTask.test.ts`、`src/components/agent/chat/hooks/agentStreamSubmitDraft.test.ts`。
  - 结论：纯 unit 切到 Node 环境是最大速度收益；本轮已将这 4 个 browser-dependent 测试移入 component，并让 unit 默认 Node。
- `npm run test:component -- src/lib/crashReporting.component.test.ts src/lib/workspaceHealthTelemetry.component.test.ts src/lib/utils/scheduleMinimumDelayIdleTask.component.test.ts src/components/agent/chat/hooks/agentStreamSubmitDraft.component.test.ts`
  - 4 files / 14 tests passed。
  - 这 4 个文件显式归入 component 层：`src/lib/crashReporting.component.test.ts`、`src/lib/workspaceHealthTelemetry.component.test.ts`、`src/lib/utils/scheduleMinimumDelayIdleTask.component.test.ts`、`src/components/agent/chat/hooks/agentStreamSubmitDraft.component.test.ts`。
- `npm run test:unit -- src/components/agent/chat/hooks/agentStreamRuntimeHandler.unit.test.ts src/components/agent/chat/hooks/agentStreamTurnEventBinding.test.ts src/components/agent/chat/utils/harnessState.test.ts src/components/agent/chat/workspace/workspaceServiceSkillEntryActionsViewModel.unit.test.ts scripts/lib/vitest-layer-classifier.unit.test.mjs scripts/run-rust-layer.unit.test.mjs`
  - 6 files / 64 tests passed。
  - chat 断言已对齐当前失败展示设计：无正文失败不再重复写入 assistant content，而是由 `runtimeStatus.detail` 承载错误原因。
- `npm run test:integration -- src/lib/layered-design/export.integration.test.ts`
  - 1 file / 16 tests passed，`Duration 29.83s`。
  - `src/lib/layered-design/export.test.ts` 已改名为 `export.integration.test.ts`，ZIP / PSD-like 二进制打包验证不再进入 unit 层。
- `npm run test:unit -- src/test/fastCheckRuns.unit.test.ts src/lib/artifact/streaming.test.ts src/lib/artifact/store.test.ts src/lib/artifact/registry.test.ts src/lib/artifact/parser.test.ts src/lib/api/importExport.test.ts src/lib/utils/apiKeyMask.test.ts`
  - 7 files / 63 tests passed。
  - Vitest duration `3.95s`。
  - 新增 `src/test/fastCheckRuns.ts`；本地 / AI TDD 默认 `25` runs，CI 保持原始 `50/100` runs，`LIME_FAST_CHECK_RUNS=100` 可本地满量复现。
  - 第一批迁移 `streaming/store/registry/parser/importExport/apiKeyMask`，覆盖此前 full unit 输出中靠前的 property 热点。
- `npm run test:unit -- src/components/api-key-provider/ApiKeyProviderSection.test.ts src/components/api-key-provider/ProviderSetting.test.ts src/components/api-key-provider/providerConfigUtils.test.ts src/components/artifact/ArtifactRenderer.test.ts src/components/artifact/ArtifactToolbar.test.ts src/icons/providers/providers-icons.test.ts src/lib/config/providers.test.ts`
  - 7 files / 105 tests passed。
  - Vitest duration `11.66s`。
  - 第二批迁移 `ApiKeyProviderSection/ProviderSetting/providerConfigUtils/ArtifactRenderer/ArtifactToolbar/providers-icons/providers`。
- unit fast-check 交叉检查
  - 当前 unit 层 fast-check 文件共 13 个，均已包含 `fastCheckRuns`；`missingHelper: []`。
- `CI=1 npm run test:unit -- src/lib/utils/apiKeyMask.test.ts`
  - 1 file / 7 tests passed。
  - 与 `fastCheckRuns.unit.test.ts` 一起证明 CI 路径仍可按原始 runs 执行。
- `npm run test:component -- src/components/agent/chat/components/HarnessStatusPanel*.test.tsx`
  - 7 files / 49 tests passed
- `npm run test:rust:layers:stats`
  - Total files: 779
  - Total test attributes: 8553
  - Runnable by default: 414
  - Live-gated: 10
  - unit: 771 files / 8536 tests / 410 runnable
  - integration: 4 files / 6 tests / 4 runnable
  - e2e: 4 files / 11 tests / 0 runnable
- `npm run test:rust:e2e`
  - 默认 gate 生效；未设置 `LIME_REAL_API_TEST=1` 时不运行 ignored/live Rust E2E。
- `/usr/bin/time -p npm run test:rust:unit`
  - 1551 passed / 1 ignored / 0 failed。
  - `real 61.76`，未出现 Cargo lock 等待提示；该数据可作为当前后端 TDD 默认入口的干净耗时基准。
- `.github/workflows/quality.yml`
  - 新增 `Rust Quick` job，PR Rust 变更默认执行 `npm run test:rust:unit`。
  - `Quality results` 汇总新增 `rust_quick` gate 与失败提示。
- `npm run test:unit -- scripts/quality-task-selector.test.ts`
  - 1 file / 1 test passed。
- `make -n help tdd tdd-rust test-layer-stats test-all-layers`
  - Makefile 目标解析通过。
- `make tdd-file FILE=scripts/run-vitest-layer.unit.test.mjs`
  - 1 file / 5 tests passed，耗时约 3.78s；可作为 AI 修改单个纯单元文件后的快速 TDD 回路。
- `make tdd-file FILE=scripts/rust-test-layer-classifier.test.mjs`
  - 预期失败通过；runner 输出 `wrong-layer matched layers=integration`，证明单文件 TDD 入口不会把 integration 文件伪装成 unit 绿色。
- `make tdd-file`
  - 预期失败通过；输出 `FILE is required`。
- `make test-layer-stats`
  - 前端和 Rust 分层统计均执行通过。
- `npm run test:unit -- scripts/check-vitest-layer-budget.test.mjs`
  - 1 file / 3 tests passed。
- `node scripts/check-vitest-layer-budget.mjs --max-component-candidates 37`
  - 2026-08-07 当前发布基线通过；候选数为 37，Status: ok。
- `make test-layer-budget`
  - 预算检查通过。
- `npm run test:unit -- scripts/check-rust-layer-budget.test.mjs scripts/check-vitest-layer-budget.test.mjs`
  - Rust / frontend 预算脚本单测通过。
- `npm run test:unit -- scripts/run-vitest-layer.unit.test.mjs scripts/check-rust-layer-budget.test.mjs`
  - 2 files / 9 tests passed；覆盖 Vitest runner 参数解析和错层过滤器保护。
- `npm run test:unit -- --list --explain | rg "run-vitest-layer|check-rust-layer-budget"`
  - 确认 `scripts/run-vitest-layer.unit.test.mjs` 和 `scripts/check-rust-layer-budget.test.mjs` 都属于 unit 层。
- `node -e "... spawnSync(... ['scripts/run-vitest-layer.mjs', 'unit', 'scripts/rust-test-layer-classifier.test.mjs'] ...)"`
  - 预期失败通过；runner 输出 `wrong-layer matched layers=integration`，证明错层文件不会再被静默跳过。
- `npm run test:integration -- scripts/rust-test-layer-classifier.test.mjs`
  - 1 file / 5 tests passed；覆盖 Rust unit / integration / e2e / excluded subcrate，以及 mixed unit + ignored live 分类边界。
- `npm run test:integration -- --list | rg "scripts/rust-test-layer-classifier.test.mjs"`
  - 确认 Rust classifier fixture 测试因使用临时文件系统而属于 integration 层，不进入 TDD 默认 `test:unit`。
- `node scripts/check-rust-layer-budget.mjs --max-e2e-runnable 0`
  - E2E files with non-ignored tests: 0，Status: ok。
- `make test-rust-layer-budget`
  - Rust E2E 默认运行预算检查通过。
- `make tdd-rust-filter FILTER=prop_valid_client_type_parsing`
  - 初始验证暴露 Cargo 默认行为：过滤器不存在时仍以 `0 tests` 成功退出。
  - `scripts/run-rust-layer.mjs` 已补过滤器空跑保护；当前预期失败通过，输出 `cargo test filter matched no executed tests`。
  - 该保护防止 AI / 本地 TDD 把拼错 Rust 测试名误判为绿色。
- `make tdd-rust-filter FILTER=workspace_support::tests::sanitize_project_dir_name_should_replace_invalid_chars`
  - 先执行 Rust E2E 默认运行预算检查，随后只运行 root `lime` lib 中 1 个 Rust unit 测试。
  - 1 test passed，1551 filtered out。
  - 调整为先校验 `FILTER`，再执行预算检查，避免缺参数时跑无意义校验。
  - `real 11.45`，可作为后端单测定向 TDD 回路证据；全量后端 unit 基准仍是约 `61.76s`。
  - 后续并发复测因两个 Cargo 进程争用 package cache / artifact directory lock 耗时 `real 173.24`，只证明行为仍正确，不作为性能基准。
- `make tdd-rust-filter`
  - 预期失败通过；缺 `FILTER` 时立即输出参数提示，不再先跑 Rust layer budget。
- `npm run test:unit -- scripts/run-rust-layer.unit.test.mjs`
  - 1 file / 5 tests passed；覆盖 Rust runner 的测试过滤器识别、Cargo 输出执行数统计、以及 `--list` 不启用空跑失败保护。
- `npx eslint scripts/run-rust-layer.mjs scripts/run-rust-layer.unit.test.mjs --max-warnings 0`
  - 通过。
- `npm run test:unit -- scripts/lib/vitest-test-file-filter.test.mjs scripts/lib/vitest-layer-report.unit.test.mjs scripts/run-vitest-layer.unit.test.mjs scripts/check-vitest-layer-budget.test.mjs`
  - 4 files / 11 tests passed。
- `.github/workflows/quality.yml`
  - `Frontend Quick` 和 `Frontend Full` 都已接入 `Test layer budget` step。
  - frontend quick / full 的失败提示包含预算检查复现命令。
  - `Rust Quick` 已接入 `Rust layer budget` step，并在失败提示中给出 Rust unit-layer 本地复现命令。
- `npm run test:component -- --list --explain`
  - `HarnessStatusPanel.testFixtures.tsx` 未出现在 component 层列表中。

## 当前阻塞与不可夹写范围

- 前端 `npm run test:unit` 已绿色，不再阻塞 TDD 默认入口。
- 当前工作区仍有非本轮写集：`internal/roadmap/soul/rollout-plan.md`、`src/components/agent/chat/utils/toolNameFamily.ts`、`src/components/agent/chat/utils/toolNameFamily.unit.test.ts` 等，本轮不接管。
- `verify:local` / `verify:gui-smoke` 未在本轮执行；本轮改动是测试分层、测试期望和测试文件换层，不改 GUI 壳、DevBridge、Workspace、Tauri 命令或 mock。

## 剩余主线

1. 前端 unit 速度主线已完成当前阶段：
   - 已完成第一刀：`scripts/run-vitest-layer.mjs` 的 `test:unit` 默认不再全层 `singleFork`；`--single-fork` 和 `LIME_VITEST_SINGLE_FORK=1` 保留为排查回退。
   - 已完成第二刀：4 个 browser-dependent unit 已移到 component，`test:unit` 默认 `--environment node`。
   - 已完成第三刀：`test:unit` 默认 `--pool threads`，并保留 `--pool=forks`、`LIME_VITEST_UNIT_POOL=...`、`--single-fork`、`LIME_VITEST_SINGLE_FORK=1` 回退 / 覆盖；高负载复测 `293.89s` 和临时取消 pool 后 `366.78s` 不作为稳定基准，下一轮先等机器安静或做慢文件画像。
   - 已完成第四刀：unit 层 fast-check property 热点已全部改用 `fastCheckRuns()`，本地 TDD 降采样、CI 保持满量；下一轮等机器安静后复测全量 unit，或做 transform / collect 慢文件画像。
2. 再治理 30 个 component migration candidates；当前发布基线暂为 30，下一轮优先把候选数降回 22 / 12 / 8：
   - `src/components/agent/chat/components/EmptyState.test.tsx`
   - `src/components/agent/chat/components/Inputbar/components/InputbarCore.test.tsx`
   - `src/components/agent/chat/components/MarkdownRenderer.test.tsx`
   - `src/components/agent/chat/components/MessageList.test.tsx`
   - `src/components/agent/chat/components/StreamingRenderer.test.tsx`
   - `src/components/agent/chat/hooks/agentSessionScopedStorage.test.ts`
   - `src/components/agent/chat/workspace/useWorkspaceConversationSceneRuntime.test.ts`
   - `src/components/agent/chat/workspace/useWorkspaceSendActions.test.tsx`
   - `src/components/agent/chat/workspace/useWorkspaceServiceSkillEntryActions.test.tsx`
   - `src/components/settings-v2/agent/providers/index.test.tsx`
   - `src/features/agent-app/ui/AgentAppsPage.test.tsx`
   - `src/features/knowledge/KnowledgePage.test.tsx`
3. `useWorkspaceSendActions` 下一刀：继续把已由 `workspaceModelSkillLaunchRequestContext.unit.test.ts` 覆盖的重复 metadata 挂载断言删除，或把该巨型 component suite 拆成 focused component suites；目标是降低 `large-component-suite` / `large-component-file` 风险。
4. 分析 Rust unit 默认入口约 `61.76s` 的主要耗时来源，决定是否继续把后端 TDD 第一信号拆到更窄 crate / package 默认集。
5. 待路线图主记录稳定后，决定是否把本补充记录归档到 README 日志尾部或保留为阶段执行证据。
