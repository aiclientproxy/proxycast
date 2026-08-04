# Lime v1.120.1 发布执行计划

状态：completed
日期：2026-08-04
目标版本：`1.120.1`
目标 tag：`v1.120.1`

## 主目标

以 `v1.120.0` 为基线，发布 `projection.rs` 的 metadata 边界修复，阻止普通运行时 Thread Item 的私有 provenance 穿过 v2 projection；保留历史导入项的受控 typed metadata。

## Release Candidate

- 基线：`v1.120.0`。
- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`。
- `candidate changes`：`lime-rs/crates/app-server/src/processor/thread/projection.rs`、本执行计划及 `release-v1.120.0-plan.md` 的收尾记录。
- `excluded changes`：`.gitignore` 的本地 `.gstack/` 追加，不属于发布候选。

## 写集与退出条件

- 版本事实源与双语 release notes 统一为 `1.120.1`，notes 只保留补丁版单页。
- 必跑 `npm run verify:app-version`、`npm run typecheck`、受影响 Rust 定向测试、`git diff --check`。
- 不改写或删除已推送的 `v1.120.0` tag；创建并推送 `v1.120.1` 后复核远端 `main` 与 tag。

## 验证记录

- `cargo test --manifest-path lime-rs/Cargo.toml -p app-server --lib processor::thread::projection`：20/20 通过。
- `cargo test --manifest-path lime-rs/Cargo.toml -p app-server --test thread_fork_compaction_jsonrpc compacted_thread_fork_replays_replacement_and_surviving_tail_after_restart -- --exact`：通过。
- `npm run verify:app-version`：通过，版本事实源统一为 `1.120.1`。
- `npm run typecheck`：通过。
- `npm run test:contracts`：通过；协议生成 846 类型、0 漂移，App Server client 292 checks，命令/脚本/文档边界通过。
- `npm run verify:gui-smoke`：通过；真实 Electron/App Server 版本 `1.120.1`，renderer/host/reload/workbench/memory settings smoke 通过；summary=`.lime/qc/project-gates/standalone-shell-01-20260804001437-91924/shell-01-electron-smoke/summary.json`。
- `git diff --check`：通过。
- Git 收口：已按确认执行 staged 边界复核、release commit、`v1.120.1` tag、`main`/tag 推送与远端引用复核。

## 分类

- `current`：App Server v2 thread projection 的 typed metadata 边界。
- `compat`：无新增。
- `deprecated`：无新增。
- `dead / deleted / forbidden-to-restore`：不恢复任何旧 runtime 或 raw metadata fallback。

## 完成度

- 当前完成度：100%。修复、版本事实源、标准门禁与 Git 发布收口均纳入本次发版执行。
