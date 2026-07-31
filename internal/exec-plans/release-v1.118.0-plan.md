# Lime v1.118.0 发布执行计划

状态：completed
日期：2026-08-01
目标版本：`1.118.0`
目标 tag：`v1.118.0`

## 主目标

将 `v1.117.0` 后当前工作树中的 Skills catalog 自动失效、typed error、`turn/plan/updated`、MCP OAuth 完成通知、Windows Provider 模型目录与 updater 可靠性，以及核心用户流程修复作为同一个 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：`v1.118.0` release candidate、版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端引用复核均已完成。
- 下一刀：无；后续增量进入下一版本，不改写已发布的 `v1.118.0` tag。

## Release Candidate

- 基线：`v1.117.0`。
- 最终候选盘点：167 个 tracked diff 文件、30 个未跟踪文件，tracked diff 为 12,614 行新增、1,563 行删除。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes 与本计划。
- `candidate changes`：当前工作树全部产品、协议、文档、测试、schema、生成物和脚本改动；重点包括 Skills catalog invalidation、typed error retry/terminal、canonical plan checklist、MCP OAuth typed notification、Provider model catalog 和 updater 状态修复。
- `excluded changes`：无。初始盘点未发现缓存、构建产物、凭证、个人临时文件或与候选无关的大文件。
- 并行避让：本轮除 release metadata、本计划和 Rust 门禁暴露的一处过时 fixture 外，其余候选文件只读验证，不覆盖已有业务改动。门禁期间新增的 MCP 架构/进度文档属于同一 current notification 候选，已纳入最终盘点与后续验证。

## 写集与退出条件

- 所有版本事实源同步为 `1.118.0`，release notes 只保留当前版本单页。
- 必跑 `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、`npm run test:rust:changed`、`npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke` 与 `git diff --check`。
- 协议、App Server、Electron、GUI、脚本与重大架构均有变化；门禁失败必须定位修复或记录明确阻塞，不能静默降级。
- 重大架构变更已同步 `internal/aiprompts/architecture.md`，并包含责任开发者确认。
- Git 写操作已获得明确确认；完整 release candidate 由同一 release commit 承载，`v1.118.0` tag、远端 `main` 与远端 tag 指向同一发布对象。

## 验证记录

- `npm run verify:app-version`：版本更新后与最终复跑均通过，根应用、CLI npm 包、Rust workspace 与锁文件一致为 `1.118.0`。
- `npm run typecheck`：通过，覆盖 renderer 与 Node TypeScript 配置。
- `npm run test:contracts`：通过；839 个 generated protocol types、0 generation failure、292 项 App Server client contract，以及 command、harness、modality、scripts、Electron release workflow 和 docs boundary guard 全部通过。
- `npm run test:rust:changed`：因 workspace 版本和 lockfile 变化扩大为 `cargo test --lib --workspace`。首次运行发现 `maps_item_and_terminal_lifecycle_to_direct_v2` 仍使用旧 `turn.failed` fixture；更新为当前 `error -> turn/completed` contract 后，定向测试 1/1 与最终 workspace 全部 suite 通过，关键结果包括 App Server 1650/1650、protocol 98/98、MCP 151/151，0 failed；既有联网 embedding 测试按规则 ignored。
- `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all --check` 与 `git diff --check`：通过。
- `npm run smoke:agent-runtime-current-fixture`：通过；覆盖 history/cache hydration、停止继续、approval、Skills catalog 自动刷新、typed error retry success/failure、Plan 历史恢复、MCP structuredContent、media、Coding Workbench 与 Article Editor，`liveProviderUsed=false`。
- `npm run verify:gui-smoke`：通过；App Server 报告 `version=1.118.0`，真实 Electron Gate B 摘要位于 `.lime/qc/project-gates/standalone-shell-01-20260731155713-55975/shell-01-electron-smoke/summary.json`。
- `npm run smoke:mcp-oauth-notification-electron-fixture`：通过；真实 Electron、外部浏览器回调与 App Server v2 notification 闭环命中 `mcpServer/oauthLogin/completed`，证据位于 `.lime/qc/mcp-oauth-notification/mcp-oauth-notification-fixture-summary.json`。
- 工作树稳定性：2026-08-01 00:01:15 与 00:01:34 两次采样的 status、tracked diff、untracked content 三组 SHA-256 完全一致；最终候选保持 167 个 tracked diff 文件与 30 个未跟踪文件。本地 `HEAD`、`origin/main` 与远端 `main` 均为 `66752a277`，目标 tag 仍不存在。
- Staged 边界：197 个候选文件全部暂存，摘要为 16,356 行新增、1,563 行删除；未暂存与未跟踪文件均为 0，`git diff --cached --check` 通过。
- 发布结果：release commit、`v1.118.0` tag、`main`/tag 推送与远端引用复核完成；发布后工作树 clean。
- 平台限制：Windows Squirrel packaged Gate B 尚未执行；macOS Electron fixture 不能替代 Windows 实机证据。

## 分类

- `current`：Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item -> GUI 产品链；typed v2 notifications、Skill catalog invalidation、canonical checklist/error recovery、Provider 模型目录和 Electron updater。
- `compat`：无新增。
- `deprecated`：Refactor v2 尚未迁完的 V2-05 notification、host capability 与 recovery surface，只允许继续迁出。
- `dead / deleted / forbidden-to-restore`：raw notification wrapper、第二 catalog/checklist/error owner、Provider inferred-only 可执行假设、updater 伪版本判断与生产 mock fallback。

## 完成度

- 当前完成度：100%。候选范围、发布元数据、发布门禁、staged 复核、commit、tag、push 与远端复核均已完成。
