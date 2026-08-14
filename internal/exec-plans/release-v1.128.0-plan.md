# Lime v1.128.0 发布执行计划

状态：release-candidate-gates-partial
日期：2026-08-14
目标版本：`1.128.0`
目标 tag：`v1.128.0`

## 主目标

发布当前工作树中的 Codex Orchestrator 完整对齐候选，覆盖 AgentControl 容量/驻留/预算、工具执行编排、
Orchestrator Skills/MCP、协议 schema、GUI 投影、脚本与治理文档，并完成版本事实源、双语单页 release notes、
质量门禁、release commit、tag、main/tag 推送与远端复核。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、
  `lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划及执行计划导航。
- `candidate changes`：当前工作树全部已跟踪和未跟踪改动，包括 Agent runtime、App Server/protocol、MCP、Skills、
  tool-runtime、GUI、脚本、测试、架构、治理和 Gate B 证据相关文件。
- `excluded changes`：无。当前工作树改动属于本次 Orchestrator 候选；不拆分或覆盖并发写集。

## 退出条件

- 四个版本事实源与双语单页 release notes 统一为 `1.128.0`，本地/远端目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、受影响 Rust/前端定向测试、
  `npm run smoke:agent-runtime-current-fixture` 与 `npm run verify:gui-smoke` 通过；无法执行的门禁必须记录原因。
- staged 内容与候选范围一致，完成危险操作确认后创建 `Release v1.128.0` commit、`v1.128.0` tag，推送 `main` 和 tag，
  并复核本地/远端状态。
- 计划中记录当前/兼容/废弃/已删除分类、完成度和剩余证据缺口；不将 terminal-slot reuse/residency 或 rollout-budget
  exhaustion/cancellation/restart 的 GUI 证据误报为已完成。

## 当前验证记录

- 版本更新前：根应用、CLI npm 包与 Rust workspace 为 `1.127.0`；目标 tag 不存在；工作树候选为 141 条状态记录、
  127 个已跟踪文件差异及 14 个未跟踪文件。版本更新与 smoke 生成/并发写集收敛后，当前候选为 148 条状态记录、
  132 个已跟踪文件差异及 16 个未跟踪文件。
- `npm run verify:app-version`：通过，四个版本事实源统一为 `1.128.0`。
- `npm run typecheck`：通过。
- `npm run test:contracts`：通过（App Server client `312` checks；protocol/command/harness/modality/scripts/release
  workflow/docs boundary 全部通过）。
- `npm run governance:legacy-report`：通过，扫描 `2129` 个源码文件，零分类漂移、零边界违规。
- `npm run governance:scripts`：通过；`git diff --check`：通过。
- `npm run test:rust:related -- <Orchestrator owner paths>`：编译成功，`1691 passed; 8 failed`。失败集中在 external
  backend `turn.accepted` 顺序、canonical child terminal recovery 错误断言、coding payload rollback、queued restore、
  mock/unavailable backend 生命周期；新增 Orchestrator owner focused tests 在该批次内通过（AgentControl `52/52`、
  rollout budget `8/8`、tool orchestrator `8/8`、provider retry `1/1`）。
- `npm run smoke:agent-runtime-current-fixture`：失败于 Expert Panel Skills Runtime override；`skill_search` 已完成但
  read model turn 长时间保持 `inProgress`，证据见 `.lime/qc/gui-evidence/claw-chat-current-fixture/` 对应 summary；其余前置
  历史/流式/审批/取消/Skills/MCP/Workbench 场景通过。
- `npm run verify:gui-smoke`：通过；真实 Electron/App Server `1.128.0`，evidence
  `standalone-shell-01-20260814143128-49941`，主 smoke 全部通过。
- `npm run smoke:orchestrator-skills-gate-b`：通过；真实 Electron/runtime 远程 Skill 读取、配置控制面、禁用
  `codex_apps` 后普通 MCP 保持可用，summary 写入 `.lime/qc/gui-evidence/orchestrator-skills-gate-b/`。
- `npm run smoke:agent-runtime-tool-execution` 与 `npm run smoke:agent-runtime-tool-execution:managed`：均失败于
  `allTargetToolsCompleted`；`Read` 保持 `in_progress`、`apply_patch` 出现重复生命周期。专用 Vitest 脚本合同仍通过
  `24/24`，但不能替代真实 runtime 证据。
- 已完成 `git add -A`，暂存内容覆盖上述候选范围；尚未执行 `git commit`、`git tag` 或 `git push`。目标 tag 本地/远端在写操作前均不存在。

## 风险与限制

- 本机无法证明 Windows packaged parity 或三平台 Forge 发布，不能以 macOS Gate B 替代 Windows 证据。
- Orchestrator 对齐计划当前总体完成度为 `95%`；实现和聚合门禁已通过，但两类产品证据仍待后续补齐。

## 架构确认

架构影响：重大。`internal/aiprompts/architecture.md` 已在本候选中记录 RuntimeCore execution/residency/budget、
tool-runtime attempt 与 Skills/MCP owner/data flow。责任开发者：root，确认日期：2026-08-14。
