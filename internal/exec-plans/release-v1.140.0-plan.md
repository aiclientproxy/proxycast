# Lime v1.140.0 发布执行计划

状态：`ready_for_release`
日期：2026-09-04
目标版本：`1.140.0`
目标 tag：`v1.140.0`

## 主目标

发布 `v1.139.0` 之后当前工作树中的 CLI/TUI 多 Surface、视频任务工具、Desktop Host 诊断、App Server client 会话传输、Electron 跨平台证据与治理收敛改动，完成版本事实源、双语单页 release notes、质量门禁、release commit、tag、main 推送和远端复核。

## Release Candidate

- `release metadata`：`package.json`、`packages/cli/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划与 `internal/exec-plans/README.md`。
- `candidate changes`：当前工作树中全部已跟踪产品、文档、测试、schema、workflow、脚本改动，以及新增 CLI/TUI、视频任务、Gate B/治理脚本和执行计划文件。
- `excluded changes`：`undefined/data/*` 下 7 个本地 SQLite/WAL 运行产物；它们是本机运行缓存，不属于产品或发布事实源。

## 架构确认

本轮新增 CLI/TUI Product Surface，但业务主链仍是 `Product Surface -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection`；Electron Desktop Host 与 CLI/TUI Host 共享协议、runtime、持久化和工具 owner，不建立第二套业务后端。该边界已同步到 `AGENTS.md`、`internal/aiprompts/architecture.md`、命令与质量文档。责任开发者：root / Codex，2026-09-04。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.140.0`；双语 Release Notes 只保留 v1.140.0；本地和远端目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 通过；按风险补充 contracts、Rust related、CLI/TUI fixture、GUI smoke 与 `git diff --check`。
- staged 内容覆盖全部 candidate changes 与 release metadata，明确排除 `undefined/data/*`；完成 `Release v1.140.0` commit、`v1.140.0` tag，并将发布提交推送到 `origin/main`、标签推送到远端后复核。

## 验证记录

- `npm run verify:app-version`：通过；根应用、CLI npm 包、Rust workspace 与 Cargo.lock 均为 `1.140.0`。
- `npm run typecheck`：通过；renderer 与 node TypeScript 均无错误。
- `npm run test:contracts`：通过；协议生成无漂移、App Server client 299 checks、command/harness/modality/scripts/release/CLI/TUI/docs boundary 全绿。
- `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check`：通过。
- `npm run test:rust:related -- ...`：核心相关 crate 全部通过；随后 `cargo test --manifest-path "lime-rs/Cargo.toml" -p tui --lib` 通过 55/55。
- CLI/TUI 收口复验：`cargo test --manifest-path "lime-rs/Cargo.toml" -p tui` 通过 57/57，`cargo test --manifest-path "lime-rs/Cargo.toml" -p cli` 通过 15/15，`cargo clippy --manifest-path "lime-rs/Cargo.toml" -p tui -p cli --no-deps -- -D warnings` 通过；覆盖 JSONL envelope、shell completion、turn 终态摘要与 Windows external-editor shim。
- `npm run smoke:cli-gate-b`：通过；真实 CLI/App Server stdio 链路输出 `turn.completed`，状态 `ready`。
- `npm run smoke:tui-gate-b`：通过；真实 PTY/alternate screen 链路完成，终端状态 `restored`。
- `npm run verify:gui-smoke`：通过；真实 Electron/App Server `version=1.140.0`，GUI smoke run `standalone-shell-01-20260903234112-2478`。
- 受影响前端显式 Vitest：11 个文件、159 项断言通过。`npm run test:related` 的 smart runner 因将 `electron` 目录误作文件触发 `EISDIR`，未产生测试断言失败，已改用显式文件验证。
- `npm run i18n:check`：通过；5 locales、34,716/34,716 keys，missing/extra 均为 0。
- `npm run governance:legacy-report`：通过；零引用候选、分类漂移、边界违规均为 0。
- `git diff --check`：通过；目标 tag 在本地和远端均不存在。
- 待执行：重新暂存并复核并发候选差异、release commit/tag、推送 `origin/main` 和 tag，以及远端状态复核。

## 收尾分类

- `current`：Electron Desktop Host、CLI/TUI Host、App Server JSON-RPC、RuntimeCore、Thread/Turn/Item projection、视频任务工具与 Desktop Gate B 证据。
- `compat`：无新增。
- `deprecated`：无新增。
- `dead / deleted`：旧 `lime-cli-npm` 包、旧 CLI skill/工具文档与其专用入口。

当前完成度：`80%`；版本 metadata、双语 release notes、质量门禁和候选范围已完成，待执行 git 写操作与远端发布复核。
