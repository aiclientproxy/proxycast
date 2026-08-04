# Lime v1.120.0 发布执行计划

状态：completed
日期：2026-08-03
目标版本：`1.120.0`
目标 tag：`v1.120.0`

## 主目标

将 `v1.119.0` 之后当前工作树中的 Codex 历史导入 canonical metadata、用户消息去重、对话时间线与文件产物投影、Task Center 环境菜单及其协议、GUI、测试和文档作为同一个 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：`v1.120.0` 已完成 commit、tag、`main`/tag 推送和远端发布资产聚合。
- 下一刀：无；发布后质量流水线发现的 Rust metadata 边界问题转入 `release-v1.120.1` 补丁计划。

## Release Candidate

- 基线：`v1.119.0`。
- 初始盘点：108 个 tracked diff 文件、5 个未跟踪文件，tracked diff 为 19,631 行新增、1,571 行删除；加入 release metadata 与本计划后，当前候选待 staged 边界复核。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes 与本计划。
- `candidate changes`：当前工作树全部产品、协议、schema、生成客户端、GUI、文档、测试与脚本改动；重点包括 Codex 导入 provenance/去重、canonical timeline、文件产物、Task Center 环境菜单与 Codex-style 阅读列。
- `excluded changes`：无。未发现缓存、构建产物、凭证或个人临时文件；按“发版”默认规则全部纳入候选。
- 并行避让：除 release metadata 与本计划外，其余候选文件只读验证；若门禁暴露 blocker，只做最小可验证修复并回写本计划。

## 写集与退出条件

- 版本事实源与双语 release notes 统一为 `1.120.0`，release notes 只保留当前版本单页。
- 必跑 `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、受影响 Rust/前端定向测试、`npm run verify:gui-smoke` 与 `git diff --check`。
- 本候选不改变 public owner、唯一产品链或依赖方向，无需修改 `internal/aiprompts/architecture.md`；发布收口时确认。
- Git 写操作（commit、tag、push）须在验证完成后按危险操作格式取得一次明确确认。

## 验证记录

- `npm run verify:app-version`：通过，版本事实源统一为 `1.120.0`。
- `npm run typecheck`：通过。
- `npm run test:contracts`：通过；协议生成 846 类型、0 漂移，App Server client 292 checks，命令/脚本/文档边界通过。
- `npm run test:rust:changed`：通过；因 workspace manifest 触达边界扩大为 `cargo test --manifest-path lime-rs/Cargo.toml --lib --workspace`，全 workspace 单测通过，tool-runtime 313/313。
- 受影响前端定向 Vitest：通过，18 个文件、237 个用例。
- `npm run i18n:check:json`：通过；5 个 locale、13 个 namespace、10,003 个源键，缺失/多余键均为 0。
- `npm run smoke:codex-import-click-through-electron-fixture`：通过；真实 Electron 导入 200 items / 4 messages，覆盖预览、附件、文件产物、环境菜单和续聊；summary=`.lime/qc/gui-evidence/codex-import-click-through-fixture/codex-import-click-through-fixture-summary.json`。
- `npm run smoke:codex-import-continuation-electron-fixture`：通过；导入零回放、统一 exec、reload 恢复，provider requests=6；summary=`.lime/qc/gui-evidence/codex-import-continuation-fixture/codex-import-continuation-fixture-summary.json`。
- `npm run smoke:local-history-import-visual-audit`：通过；summary=`.lime/qc/gui-evidence/local-history-import-visual-audit/local-history-import-visual-audit-summary.json`。
- `npm run verify:gui-smoke`：通过；真实 Electron/App Server 版本 `1.120.0`，renderer/host/reload/workbench/memory settings smoke 通过；summary=`.lime/qc/project-gates/standalone-shell-01-20260803152335-39638/shell-01-electron-smoke/summary.json`。
- `git diff --check`：通过。
- 已完成：staged 候选摘要、release commit `944b7f40b`、`v1.120.0` tag、远端 `main` 与 tag 引用复核。
- 发布后质量流水线：GUI Smoke、Windows Shell Runtime、Integrity 通过；Frontend Full 因 layer budget `35 > 30` 失败，Rust Full 因 `contextCompaction` 泄漏私有 `sourceEventType` metadata 失败。

## 发布后质量收尾

- `v1.120.0` tag 保持不变，不删除、不重建、不 force-push。
- `projection.rs` 已将 typed item metadata 限制为 `imported=true` 的历史导入项；projection library 20/20 与 fork-compaction JSON-RPC 集成回归均通过。
- 补丁版本执行记录：`internal/exec-plans/release-v1.120.1-plan.md`。

## 分类

- `current`：Codex import canonical provenance、typed ThreadItem metadata、历史去重、timeline/file artifact projection、Task Center environment/location UI。
- `compat`：无新增。
- `deprecated`：既有 Refactor v2 未完成项，继续按原计划迁出。
- `dead / deleted / forbidden-to-restore`：无本轮新增删除；生产 mock 与旧 owner 不得恢复。

## 完成度

- 当前完成度：100%。`v1.120.0` 已完成候选收口、版本事实源、notes、门禁、commit、tag、push 和远端复核；后续质量修复由补丁计划承接。
