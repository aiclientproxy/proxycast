# Lime v1.113.0 发布执行计划

状态：release-commit-ready
日期：2026-07-27
目标版本：`1.113.0`
目标 tag：`v1.113.0`

## 主目标

将 `v1.112.0` 之后的 Runtime World State、Multi-Agent、Provider 健康与重路由、Tool Hook、存储根、GUI、协议、测试和文档改动整理为单一 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：完整 release candidate 已通过风险匹配的发版门禁与 staged 审计，Git 写操作已获明确授权。
- 下一刀：创建 release commit 与 tag，推送 `main`/tag 并复核本地、远端和发布工作树状态。

## Release Candidate

- 基线：`v1.112.0`。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes 与本计划。
- `candidate changes`：发版开始时工作树中全部 tracked 与 untracked 产品、协议、schema、generated client、测试、脚本和文档改动，包括既有 staged 删除。
- 最终盘点：`166` 个 tracked change、`11` 个 untracked 文件，共 `177` 个候选文件；包含发版期间完成的 `model/verification` schema/codegen 改动，以及门禁中补齐的 Server metadata 兼容和 Agent Runtime trace 断言。
- `excluded changes`：无。提交前若发现并发新增或临时文件，停止并重新判定范围。

## 写集与协作边界

- 本轮原写集为版本事实源、双语 release notes 与本计划；门禁暴露真实阻断后，窄幅扩展到 `lime-rs/crates/server/src/handlers/provider_calls.rs`、`lime-rs/crates/server/src/handlers/provider_calls/streaming.rs` 与 `lime-rs/crates/agent-runtime/src/reply_backend/tests.rs`。
- 现有产品、协议、测试、脚本和文档改动只读避让；验证结果代表当前完整工作树，不声明这些改动的作者归属。
- 未覆盖或回滚既有 staged 状态；已获 Git 写操作授权，并以 `git add -A` 收敛完整 release candidate。staged 审计为 `177` 个文件、`9259` 行新增、`6956` 行删除，无 unstaged 或 untracked 遗留。

## 退出条件

- 所有版本事实源同步为 `1.113.0`，双语 release notes 只保留当前版本单页。
- `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、风险匹配的 Rust related tests、`npm run governance:scripts`、`npm run governance:legacy-report` 与 `git diff --check` 通过。
- 正式 GUI 发布证据执行 `npm run verify:gui-smoke`；无法执行时记录具体环境限制。
- 获得 Git 写操作确认后，连续执行暂存、`Release v1.113.0` commit、`v1.113.0` tag、`main`/tag 推送，并复核本地与远端状态。

## 验证记录

- `npm run verify:app-version`：通过；根应用、CLI npm package、Rust workspace 与 30 个 workspace lockfile package 版本均为 `1.113.0`。
- `npm run typecheck`：通过；发版收尾复跑退出码为 0。
- `npm run test:contracts`：通过；805 个协议类型零漂移，App Server client 284 项 contract 通过，command、Harness、modality、scripts、release workflow 与 docs boundary 均通过。
- `npm run test:rust:changed`：通过；完整 workspace library tests 通过，包括 Agent Runtime `180/180`、App Server `1555/1555`、App Server protocol `85/85`、Tool Runtime `306/306`。
- App Server 公共 JSON-RPC integration：通过；6 个 targets、27 个 tests 全部通过。
- Server provider metadata 定向测试：通过；`9/9`，确认内部 `ServerModel` / `ModelVerification` metadata 不泄漏到 OpenAI/Anthropic compat 输出。
- Agent Runtime reply backend 定向测试：通过；`1/1`，Responses Lite 禁止并行时 trace 明确 fail closed 为 `Some(false)`。
- 前端/Electron 定向 Vitest：通过；6 个 files、80 个 tests 全部通过。
- `npm run test:related`：runner 在目录目标 `electron` 上报已知 `EISDIR`；已用直接 Vitest 覆盖实际受影响测试，未发现产品失败。
- `npm run smoke:agent-runtime-current-fixture`：通过；覆盖真实 Electron、历史恢复、流式终态、审批、Skills、MCP、媒体引用、专家入口与内容工厂主链。
- `npm run verify:gui-smoke`：通过；Renderer、Electron host、preload/current bridge、App Server sidecar `1.113.0`、reload、Workbench shell 与设置页均通过。
- `npm run governance:legacy-report`：通过；边界违规 0。
- `npm run governance:scripts`：通过；冻结脚本基线无新增违规。
- `npm run governance:electron-release-workflow`：通过。
- `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check`、`git diff --check`：通过；发版收尾均已复跑。

## 架构确认与治理分类

- 架构影响：重大。候选涉及 Runtime World State、effective Multi-Agent mode、Provider auth/readiness/health/reroute、Tool Hook owner 与平台存储根边界。
- 架构事实源：`internal/aiprompts/architecture.md` 已包含候选对应的 current owner、数据流、fail-closed 与 forbidden-to-restore 约束。
- 责任人：root（release owner，v1.113.0）；日期：2026-07-27；确认状态：validated-for-release。
- `current`：Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI，以及 typed world state、Provider current client、RuntimeCore route resolver、tool-runtime hooks 与组合存储根。
- `compat`：不新增 compat owner；历史边界只能委托 current 实现。
- `deprecated`：旧配置字段只保留必要 wire 形状且 runtime 忽略，不得重新进入 durable metadata 或控制面。
- `dead / deleted / forbidden-to-restore`：旧 HookManager、Product DB 整库迁移、通用 migration manifest、启动 cleanup、managed project path 迁移与 provider migration fixture 不得恢复。

当前发布完成度：`95%`；剩余 release commit、tag、推送与远端复核正在连续执行。
