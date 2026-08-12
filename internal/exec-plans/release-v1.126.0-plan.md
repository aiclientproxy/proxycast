# Lime v1.126.0 发布执行计划

状态：release-candidate-ready
日期：2026-08-12
目标版本：`1.126.0`
目标 tag：`v1.126.0`

## 主目标

发布 Code Mode process-owned runtime、Agent runtime/session loop、provider canonical transport、App Server read model、
Electron 双 sidecar 资源链、协议/GUI 投影以及对应治理文档；完成 release commit、tag、main/tag 推送与远端复核。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、
  `lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划。
- `candidate changes`：用户确认当前工作树全部纳入，包括已暂存、未暂存和未跟踪的 Rust、TypeScript、脚本、CI、
  测试、架构、治理与执行计划文件。
- `excluded changes`：无。

## 退出条件

- 版本事实源与双语单页 release notes 统一到 `1.126.0`，目标 tag 本地/远端不存在。
- Code Mode production 只走 `ProcessCodeModeSessionProvider -> code-mode-host -> sandbox V8`，无 in-process fallback。
- dev、Electron assets、Forge/Windows 构建成组携带并校验 `app-server` 与 `code-mode-host`。
- `npm run verify:app-version`、`npm run typecheck`、Rust related、contracts 与 GUI smoke 通过。
- Code Mode Electron Gate B 证明真实 Electron/App Server/standalone host PID、custom exec 回采样与 GUI terminal。
- 完成 release commit、`v1.126.0` tag、main/tag 推送和远端复核。

## 验证记录

- `npm run verify:app-version`：通过，版本事实源一致为 `1.126.0`。
- `npm run typecheck`：通过（发布 metadata 更新后复跑）。
- `npm run test:contracts`：通过；App Server client `301` 项检查及 command、harness、modality、scripts、Electron release workflow、docs boundary 子门禁全部通过。
- `npm run test:rust:related -- lime-rs/crates/agent-runtime lime-rs/crates/agent lime-rs/crates/app-server lime-rs/crates/model-provider lime-rs/crates/runtime-core lime-rs/crates/tool-runtime lime-rs/crates/services`：通过；相关 owner 与反向依赖 crate 单测无失败，存在一个既有测试辅助函数 dead-code warning。
- `npm run smoke:agent-runtime-current-fixture`：通过；history/cache、turn terminal、approval、steer、Plan、Skills、MCP、媒体、Workbench 等 current Electron fixture 闭环通过，`liveProviderUsed=false`。
- `npm run verify:gui-smoke`：通过；真实 Electron/App Server `1.126.0` 初始化、工作台 reload 与 memory settings 可见，evidence result 为 `pass`。
- `npm run governance:legacy-report`：通过；扫描 `2120` 个源码文件，分类漂移 `0`、边界违规 `0`。
- standalone Code Mode Cargo check：通过，无 warning。
- Code Mode process Rust tests：`4/4` 通过。
- sidecar/assets/fixture/Gate B script tests：`50/50` 通过。
- `npm run electron:build:app-server-assets`：通过；macOS arm64 双 sidecar 为 `0755`，manifest 双 SHA 复算一致。
- `npm run smoke:code-mode-electron-gate-b`：通过；thread `019ff39f-2ff2-7bb3-b579-cf1dc86ac042`，Electron/App Server/
  host PID 为 `10485/10494/10926`，host parent PID 为 `10494`，17 项 assertion 全通过。
- `cargo fmt --all -- --check`、`git diff --check`：通过。

## 待执行门禁

- 全候选 staged 复核、危险操作确认、commit/tag/push 与远端状态复核。
- Windows 双 sidecar 由 CI 的 Windows runner 执行；本地 macOS Gate B 不冒充 Windows packaged parity。

## 架构确认

架构影响：重大。已更新 `internal/aiprompts/architecture.md` 第 44 节，确认 production process owner、双 sidecar
构建/资源完整性、fail-closed 边界与 Gate B 证据。责任开发者：root，确认日期：2026-08-12。
