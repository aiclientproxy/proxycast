# Lime v1.119.0 发布执行计划

状态：completed
日期：2026-08-01
目标版本：`1.119.0`
目标 tag：`v1.119.0`

## 主目标

将 `v1.118.0` 后当前工作树中的 MCP startup status typed notification、unknown Item 安全投影与恢复、unified exec terminal interaction 脱敏持久化和对应协议、GUI、测试、文档作为同一个 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：`v1.119.0` release candidate、版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送、远端引用、GitHub Release、Electron/CLI 资产与 updater 发布复核均已完成。
- 下一刀：后续增量进入下一版本，不改写已发布的 `v1.119.0` tag；Vitest component migration candidates 与 R2 旧资产清理工具能力作为独立治理任务处理。

## Release Candidate

- 基线：`v1.118.0`。
- 初始盘点：123 个 tracked diff 文件、8 个未跟踪文件，tracked diff 为 6,883 行新增、1,347 行删除。最终暂存前盘点为 128 个 tracked diff 文件、9 个未跟踪文件；完整 staged 候选共 137 个文件、7,964 行新增、1,427 行删除。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes 与本计划。
- `candidate changes`：初始工作树全部产品、协议、schema、生成客户端、GUI、文档、测试与脚本改动；重点包括 MCP startup status、unknown Item live/cold recovery 与 unified exec terminal interaction。
- `excluded changes`：无。初始盘点未发现缓存、构建产物、凭证、个人临时文件或与候选主题无关的文件。
- 并行避让：除 release metadata 与本计划外，其余候选文件只读验证，不覆盖已有业务改动；若门禁暴露 release blocker，只做可定位、可验证的最小修复并在本计划记录。

## 写集与退出条件

- 版本事实源同步为 `1.119.0`，release notes 只保留当前版本单页。
- 必跑 `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、`npm run test:rust:changed`、`npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke` 与 `git diff --check`。
- MCP startup、unknown Item 与 terminal interaction 已有专项 Gate B 证据；发布门禁若发现证据过期或相关回归，再定向复跑对应 fixture。
- 本候选未改变 public owner、唯一产品链或依赖方向，无需修改 `internal/aiprompts/architecture.md`；责任开发者将在发布收口时确认。
- Git 写操作已获得明确确认；完整 release candidate 由 release commit `f3d062b8c47f7b7682f3ce889337175bcf1cf8e6` 承载，`v1.119.0` tag、远端 `main` 与远端 tag 指向同一发布对象。

## 验证记录

- `npm run verify:app-version`：通过，所有版本事实源为 `1.119.0`。
- `npm run typecheck`：通过。
- `npm run test:contracts`：通过；844 generated protocol types、0 generation failures、292 app-server-client checks。
- `npm run test:rust:changed`：首轮暴露协议稳定名称断言遗漏 `item/commandExecution/terminalInteraction`；已在 `lime-rs/crates/app-server-protocol/src/protocol/v2/tests.rs` 做最小修复并重跑通过。关键结果：App Server 1662/1662、protocol 99/99、MCP 151/151、tool-runtime 313/313。
- `npm run smoke:agent-runtime-current-fixture`：通过；`liveProviderUsed=false`，包含 unknown Item 真实 Electron Gate B 场景。
- `npm run verify:gui-smoke`：通过；App Server version `1.119.0`。证据：`.lime/qc/project-gates/standalone-shell-01-20260801181323-22109/shell-01-electron-smoke/summary.json`。
- `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all --check`：通过。
- `git diff --check`：通过。
- Staged 边界：137 个候选文件全部暂存，摘要为 7,964 行新增、1,427 行删除；未暂存与未跟踪文件均为 0，`git diff --cached --check` 通过。commit hook 的 Level 0 AI 验证为 137/137 通过。
- 发布结果：release commit `f3d062b8c`、轻量 tag `v1.119.0`、`main`/tag 推送与远端引用复核完成；发布后工作树 clean。
- GitHub Release workflow `30717094281`：通过。Electron macOS x64、macOS arm64、Windows x64 构建、签名、资源验证与资产暂存全部成功；Windows 从 `v1.118.0` 安装包升级到 `v1.119.0` 的 Squirrel packaged smoke 及证据上传成功；Electron 资产聚合和 GitHub Release 发布成功。
- updater 与 CLI：Cloudflare R2 stable updater 资产上传成功；macOS x64、macOS arm64、Windows x64、Linux x64 CLI 构建与 GitHub Release 上传全部成功。R2 旧版本清理因当前 Wrangler 不支持 `r2 object list` 而按工作流警告跳过，不影响本版本资产上传。
- GitHub Release：`v1.119.0` 已发布并标记为 latest，非 draft / prerelease，共 13 个 uploaded 资产；页面为 `https://github.com/limecloud/lime/releases/tag/v1.119.0`。
- GitHub Quality workflow `30717086347`：Integrity、Lint、Typecheck、GUI Smoke、Rust Full 与 Windows Shell Runtime 均通过；Frontend Full 仅因仓库沿用自 `v1.118.0` 的 Vitest layer budget 基线 `35 > 30` 失败，未进入全量 Vitest。本地以同一命令复现为 35 个 candidates，未临时放宽阈值或改写已发布 tag。
- 验证等级：协议/跨层门禁、Rust owner 与 CI workspace 全量测试、Current fixture、真实 Electron Gate B、Windows packaged Gate B 和实际 macOS/Windows 发布产物均已覆盖；未获得本轮全量前端 Vitest 证据，原因是既有 layer budget 在收集前阻断。

## 分类

- `current`：App Server v2 typed MCP startup status、canonical unknown Item、command terminal interaction，以及对应 read model、客户端和 GUI 投影。
- `compat`：无新增。
- `deprecated`：Refactor v2 尚未完成且缺少完整 producer、consumer、持久化语义或 Gate B 的 planned notification，仅允许继续迁出。
- `dead / deleted / forbidden-to-restore`：旧 MCP lifecycle Desktop events、unknown Item null drop/raw payload/Extension fallback、独立 `write_stdin` Tool Item、raw stdin 持久化与生产 mock fallback。

## 完成度

- 当前完成度：100%。候选范围、发布元数据、发布门禁、staged 复核、commit、tag、push、远端引用、GitHub Release、Electron/CLI 资产与 updater 发布均已完成；既有 Vitest layer budget 与 R2 旧资产清理能力已记录为独立治理项，不改写本版本 tag。
