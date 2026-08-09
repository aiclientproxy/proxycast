# Lime v1.125.0 发布执行计划

状态：release-candidate
日期：2026-08-09
目标版本：`1.125.0`
目标 tag：`v1.125.0`

## 主目标

在不覆盖已发布的 `v1.124.0` tag 的前提下，发布当前工作树中的 Agent runtime、App Server/protocol、Plugin v3、GUI、文档与质量治理改动。

## Release Candidate

- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划。
- `candidate changes`：当前工作树全部已跟踪和未跟踪改动，包括 Agent runtime、App Server/protocol、Plugin v3、Electron/GUI、测试、治理和文档。
- `excluded changes`：无；用户已明确确认当前工作树整体纳入。

## 退出条件

- 根 `README.md` 为英文 canonical 入口，英文页面无二维码，中文页面保留二维码。
- 版本事实源与双语 release notes 统一到 `1.125.0`，不覆盖 `v1.124.0`。
- 通过版本一致性、typecheck、docs boundary、contracts、GUI smoke 或明确记录环境限制。
- 完成 release commit、`v1.125.0` tag、`main`/tag 推送和远端复核。
- 针对发布 commit SHA 触发 `.github/workflows/build-windows-test.yml`，轮询 Windows runner 直至完成并保存 artifact 结果。

## 验证记录

- `npm run verify:app-version`：通过，所有版本事实源为 `1.125.0`。
- `npm run typecheck`：通过。
- Guardian projection Vitest：`54/54` 通过。
- `npm run test:contracts`：通过。
- `npm run docs:boundary`：通过。
- `npm run governance:legacy-report`：通过，零引用候选、分类漂移、边界违规均为 `0`。
- `npm run governance:scripts`：通过。
- `npm run governance:electron-release-workflow`：通过。
- `npm run verify:gui-smoke`：通过；Electron evidence `standalone-shell-01-20260809113946-74365`，App Server `1.125.0`。
- `npm run test:rust:related -- lime-rs/crates/app-server lime-rs/crates/mcp lime-rs/crates/skills lime-rs/crates/runtime-core lime-rs/crates/tool-runtime`：通过，`311 passed; 0 failed`。
- `npm run smoke:agent-runtime-current-fixture`：通过，覆盖当前 Agent runtime 全部 fixture；报告 `liveProviderUsed=false`。
- `git diff --check`：通过。
- Release commit/tag/push：待执行。
- Windows runner：待触发并轮询。
