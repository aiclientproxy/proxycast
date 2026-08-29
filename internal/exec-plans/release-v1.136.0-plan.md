# Lime v1.136.0 发布执行计划

状态：`validation-complete / awaiting-release-confirmation`
日期：2026-08-29
目标版本：`1.136.0`
目标 tag：`v1.136.0`

## 主目标

发布 `v1.135.0` 之后当前分支的 Windows restricted execution、Scheduled Tasks route/provider、Agent Runtime projection 与 Provider transport 改动，完成版本事实源、双语单页 release notes、质量门禁及 release commit/tag/push。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划及执行计划索引。
- `candidate changes`：`main..HEAD` 的 40 个非 merge 提交，以及当前暂存的 87 个文件改动，覆盖 Windows sandbox/tool-runtime、Rust App Server/model-provider、Agent Runtime/GUI、Scheduled Tasks、Electron resource verifier、DeepSWE harness、测试和架构文档。
- `excluded changes`：无。用户请求为完整发布，当前工作树改动均纳入本轮候选。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.136.0`；双语 release notes 只保留 v1.136.0；目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 必须通过；按风险执行 contracts、current fixture、GUI smoke 和受影响 Rust/前端测试，未执行或失败项原样记录。
- staged 内容与候选范围一致；完成 `Release v1.136.0` commit、`v1.136.0` tag，推送 `main` 和 tag，并复核本地/远端状态。

## 架构确认

本轮不新增产品链 owner；Windows 执行能力继续归 `tool-runtime`，模型与 Provider 网络归 `model-provider`，会话/投影继续沿用 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`。架构事实源已在 `internal/aiprompts/architecture.md` 更新并由当前候选纳入。

## 验证记录

已完成：

```text
npm run verify:app-version
npm run typecheck
npm run test:contracts
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
npm run test:rust:related -- <受影响 Rust 路径>
```

验证结果：

- `npm run verify:app-version`：通过，根应用、CLI npm 包、Rust workspace 与 Cargo.lock 均为 `1.136.0`。
- `npm run typecheck`：通过。
- `npm run test:contracts`：通过，协议生成无漂移，App Server client 299 项及 command/harness/governance/docs 边界通过。
- `npm run smoke:agent-runtime-current-fixture`：通过，覆盖 Agent Runtime current Electron fixture 全集，`liveProviderUsed=false`。
- `npm run test:rust:related -- lime-rs/crates/tool-runtime lime-rs/crates/agent lime-rs/crates/app-server lime-rs/crates/model-provider lime-rs/crates/services`：通过，14 个 owner/反向依赖 crate 的 unit tests 全部通过。
- `npm run verify:gui-smoke`：通过，真实 Electron Shell-01 `21/21` assertions；App Server `appserver.v0` 版本 `1.136.0`，console/page/invoke/preload/IPC/legacy/mock fallback 错误均为 0。证据：`.lime/qc/project-gates/standalone-shell-01-20260829101933-66941/shell-01-electron-smoke/summary.json`。
- `git diff --check`：通过。

## 收尾记录

- `current`：Windows restricted execution、Agent Runtime projection、Scheduled Tasks route、Provider transport、真实 Electron/App Server bridge。
- `compat`：无新增 compat wrapper。
- `deprecated`：无新增 deprecated owner。
- `dead / deleted`：不恢复旧 runtime、catalog fallback、mock 生产路径或旧 Windows process supervisor。
- 当前完成度：`98%`；版本、notes、候选范围和门禁已完成，仅待用户确认后执行 commit/tag/push 及远端复核。
