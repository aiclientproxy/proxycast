# Lime v1.131.0 发布执行计划

状态：`ready-for-confirmation / release-candidate`
日期：2026-08-17
目标版本：`1.131.0`
目标 tag：`v1.131.0`

## 主目标

发布当前 `main` 工作树中的 Scheduled Tasks current 迁移切片：协议与 schema、App Server/Runtime 调度闭环、typed terminal notification、Renderer/Electron bridge、软删除、五语种工作台、真实 Electron Gate B fixture、治理与路线图收敛；完成版本事实源、双语单页 release notes、质量门禁、release commit、tag、`main`/tag 推送和远端复核。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划及执行计划导航。
- `candidate changes`：当前工作树中的全部 224 个受跟踪改动和 12 个未跟踪文件（含本计划），覆盖 Scheduled Tasks 协议、Rust/App Server/Runtime、Renderer/Electron、脚本、测试、架构、路线图、治理和五语言资源；不拆分同一 current 切片。
- `excluded changes`：无。用户请求为完整发布，当前工作树改动均归属于本轮 Scheduled Tasks/Automation 收敛候选。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.131.0`，双语 release notes 只保留 v1.131.0，目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 必须执行；按风险执行 contracts、受影响 Rust/前端定向测试、Scheduled Tasks Electron fixture、Agent current fixture、GUI smoke、治理与本地门禁，未执行或失败项原样记录。
- staged 内容与上述候选范围一致；获得危险操作确认后创建 `Release v1.131.0` commit、`v1.131.0` tag，推送 `main` 和 tag，并复核本地/远端状态。
- 收尾记录 current/compat/deprecated/dead 分类、完成度、GUI/Gate B 状态和未覆盖的平台/打包证据，不把环境限制误报为通过。

## 当前验证记录

- 发布前基线为 v1.130.0（commit `b3e77b109`），发布前 `v1.131.0` 本地和远端 tag 均不存在。
- 当前候选在版本元数据与 release notes 写入后为 224 个受跟踪文件改动和 12 个未跟踪文件；暂存前仍需再次核对完整清单，确保没有超出本轮 current 切片的文件。
- Scheduled Tasks 实现计划已记录 protocol/client contracts、定向 Vitest、Rust related/changed、真实 Electron Gate B、Agent current fixture、GUI smoke、治理报告、`verify:local`、fmt 与 diff check 的通过证据；本次发版前门禁已全部重跑并通过。

## 门禁结果

- `npm run verify:app-version`：通过，版本一致为 `1.131.0`。
- `npm run typecheck`：通过。
- `npm run test:contracts`：通过。
- Scheduled Tasks 定向 Vitest：4 个文件、33 个测试通过。
- `npm run smoke:scheduled-tasks-electron-fixture -- --timeout-ms 180000`：通过，真实 Electron/preload/IPC/App Server/RuntimeCore/provider Gate B。
- `npm run smoke:agent-runtime-current-fixture`：通过，`liveProviderUsed=false`。
- `npm run verify:gui-smoke`：通过，App Server version `1.131.0`，evidence `standalone-shell-01-20260817163728-99465`。
- `npm run governance:legacy-report`：通过，扫描 2100 个源码文件，zero-reference candidates、classification drift、boundary violations 均为 0。
- `npm run verify:local`：通过，Vitest smart 119/119、contracts、Rust workspace、Electron shell smoke 均通过。
- `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check`：通过。
- `git diff --check`：通过。

本地不执行 Windows Notification Center、真实 macOS/Windows sleep-resume、签名、公证、正式 release asset 和 CI 门禁；这些证据必须由对应平台/CI runner 提供。

## 架构确认

架构影响：重大。候选更新了 Scheduled Tasks public JSON-RPC/read-model、typed notification、Renderer bridge 与 Electron Desktop Host 主链，相关确认已记录于 `internal/exec-plans/scheduled-tasks-implementation.md` 与 `internal/aiprompts/architecture.md#73-scheduled-tasks`。

责任开发者：root，确认日期：2026-08-17。

## 收尾记录

- `current`：Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI 主链，以及 Scheduled Tasks scheduler、notification bridge 和 model route selection。
- `compat`：旧 `automationJob/*` 管理接口及 Settings surface 仍按迁移计划保留为受控 compat，不承担新的 current 创建路径。
- `deprecated`：旧 Automation 双轨及其剩余 consumer/fixture 等待物理迁移清理；本版本不新增兼容包装。
- `dead / deleted`：本候选删除已脱离构建图的旧 Automation 页面、API、fixture、协议文件和相关回流入口。
- 当前完成度：95%；待 commit/tag/push 与远端复核完成后更新为 100%。Windows Notification Center、真实 macOS/Windows sleep-resume、签名、公证、正式 release asset 和 CI 门禁证据保持未覆盖状态。
