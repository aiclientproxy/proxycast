# Lime v1.119.0 发布执行计划

状态：in-progress
日期：2026-08-01
目标版本：`1.119.0`
目标 tag：`v1.119.0`

## 主目标

将 `v1.118.0` 后当前工作树中的 MCP startup status typed notification、unknown Item 安全投影与恢复、unified exec terminal interaction 脱敏持久化和对应协议、GUI、测试、文档作为同一个 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：发布门禁已完成，候选范围和 staged 边界待最终复核；等待 Git 写操作确认。
- 下一刀：在不修改真实 index 的前提下计算完整候选 staged 摘要，请求一次 commit/tag/push 明确确认；确认后连续完成 Git 写入、远端推送和 tag 复核。

## Release Candidate

- 基线：`v1.118.0`。
- 初始盘点：123 个 tracked diff 文件、8 个未跟踪文件，tracked diff 为 6,883 行新增、1,347 行删除。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes 与本计划。
- `candidate changes`：初始工作树全部产品、协议、schema、生成客户端、GUI、文档、测试与脚本改动；重点包括 MCP startup status、unknown Item live/cold recovery 与 unified exec terminal interaction。
- `excluded changes`：无。初始盘点未发现缓存、构建产物、凭证、个人临时文件或与候选主题无关的文件。
- 并行避让：除 release metadata 与本计划外，其余候选文件只读验证，不覆盖已有业务改动；若门禁暴露 release blocker，只做可定位、可验证的最小修复并在本计划记录。

## 写集与退出条件

- 版本事实源同步为 `1.119.0`，release notes 只保留当前版本单页。
- 必跑 `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、`npm run test:rust:changed`、`npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke` 与 `git diff --check`。
- MCP startup、unknown Item 与 terminal interaction 已有专项 Gate B 证据；发布门禁若发现证据过期或相关回归，再定向复跑对应 fixture。
- 本候选未改变 public owner、唯一产品链或依赖方向，无需修改 `internal/aiprompts/architecture.md`；责任开发者将在发布收口时确认。
- Git 写操作必须获得明确确认；确认前不执行 `git add`、`git commit`、`git tag` 或 `git push`。

## 验证记录

- `npm run verify:app-version`：通过，所有版本事实源为 `1.119.0`。
- `npm run typecheck`：通过。
- `npm run test:contracts`：通过；844 generated protocol types、0 generation failures、292 app-server-client checks。
- `npm run test:rust:changed`：首轮暴露协议稳定名称断言遗漏 `item/commandExecution/terminalInteraction`；已在 `lime-rs/crates/app-server-protocol/src/protocol/v2/tests.rs` 做最小修复并重跑通过。关键结果：App Server 1662/1662、protocol 99/99、MCP 151/151、tool-runtime 313/313。
- `npm run smoke:agent-runtime-current-fixture`：通过；`liveProviderUsed=false`，包含 unknown Item 真实 Electron Gate B 场景。
- `npm run verify:gui-smoke`：通过；App Server version `1.119.0`。证据：`.lime/qc/project-gates/standalone-shell-01-20260801181323-22109/shell-01-electron-smoke/summary.json`。
- `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all --check`：通过。
- `git diff --check`：通过。
- 验证等级：协议/跨层门禁、Rust owner 测试、Current fixture 与真实 Electron Gate B 均已覆盖；未执行全量 `npm test`、裸 `npm run lint` 和全量 Cargo 矩阵，按当前发版规范不作为默认阻断项。

## 分类

- `current`：App Server v2 typed MCP startup status、canonical unknown Item、command terminal interaction，以及对应 read model、客户端和 GUI 投影。
- `compat`：无新增。
- `deprecated`：Refactor v2 尚未完成且缺少完整 producer、consumer、持久化语义或 Gate B 的 planned notification，仅允许继续迁出。
- `dead / deleted / forbidden-to-restore`：旧 MCP lifecycle Desktop events、unknown Item null drop/raw payload/Extension fallback、独立 `write_stdin` Tool Item、raw stdin 持久化与生产 mock fallback。

## 完成度

- 当前完成度：85%。候选范围、版本事实源、双语 release notes 和全部默认发布门禁已收口；剩余 Git 写操作、远端推送与 tag/workflow 复核待明确确认后执行。
