# Lime v1.141.0 发布执行计划

状态：进行中  
日期：2026-09-06  
目标版本：`1.141.0`  
目标 tag：`v1.141.0`

## 主目标

发布 v1.140.0 之后工作树中的 Code Mode 四层 crate、CLI/TUI 多 Surface、远程
transport、Codex TUI 对齐、App Server 边界修复与配套治理/测试改动；完成版本事实源、
双语单页 release notes、质量门禁、release commit、tag、main 推送和远端复核。

## Release Candidate

- `release metadata`：`package.json`、`packages/cli/package.json`、`lime-rs/Cargo.toml`、
  `lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划及
  `internal/exec-plans/README.md`。
- `candidate changes`：当前工作树中全部已跟踪产品、文档、测试、schema、workflow、脚本、
  Code Mode/CLI/TUI crate 与执行计划改动，以及全部未跟踪产品/测试/文档文件。
- `excluded changes`：`undefined/` 下本机 SQLite/WAL、runtime 数据库和 `.DS_Store` 运行产物；
  它们是本地缓存，不属于产品或发布事实源。

## 架构确认

本轮 Code Mode 与 CLI/TUI 扩展仍遵循唯一业务主链：

```text
Electron Desktop Host / CLI-TUI Host
  -> App Server JSON-RPC
  -> RuntimeCore
  -> Thread / Turn / Item projection
```

Code Mode 仅将协议、运行时、host 与 session facade 收敛为 current crate；远程连接通过
认证 transport 接入同一 App Server，不创建第二套 runtime、会话状态或持久化。相关边界已
同步到 `internal/aiprompts/architecture.md`、命令、治理与质量事实源。责任开发者：root / Codex，
2026-09-05。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.141.0`；双语 release notes
  只保留 v1.141.0；本地和远端目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 通过；按风险补充 contracts、Rust
  related、CLI/TUI/Code Mode fixture、GUI smoke、治理检查与 `git diff --check`。
- staged 内容覆盖全部 candidate changes 与 release metadata，明确排除 `undefined/data/*`；
  完成 `Release v1.141.0` commit、`v1.141.0` tag，并推送 `origin/main` 与远端 tag 后复核。

## 验证记录

验证记录如下；默认门禁为：

```bash
npm run verify:app-version
npm run typecheck
npm run test:contracts
npm run verify:gui-smoke
npm run test:rust:related -- <changed rust paths>
git diff --check
```

V8 预构建 archive 在本机 Darwin/aarch64 若仍返回 404，改用仓库已校验的本地 archive/binding
注入，并在收尾报告明确说明；不得将该环境限制误报为产品测试失败。

已完成验证：

- `npm run verify:app-version`：通过，版本为 `1.141.0`。
- `npm run typecheck`：通过。
- `npm run test:contracts`：通过（协议、命令、治理、文档边界检查通过）。
- Rust 定向测试：agent-runtime、app-server、app-server-client、app-server-test-client、CLI、
  Code Mode 四层 crate、lime-agent、lime-scheduler、lime-server、lime-mcp、tool-runtime 与 TUI
  均通过；TUI 当前全量为 267 项，CLI 为 50 项，Code Mode 各 package 测试通过。
- `npx vitest run scripts/lib/electron-dev-sidecar.test.mjs`：17/17 通过；sidecar 构建参数已
  同步独立 `code-mode-host` crate。
- `npm run verify:gui-smoke`：通过，真实 Electron、preload/IPC、App Server JSON-RPC、Code Mode
  sidecar、Claw workbench、响应式布局和 memory settings 均通过；证据位于
  `.lime/qc/project-gates/standalone-shell-01-20260905171031-93079/`。
- `npm run smoke:cli-gate-b`：通过，真实 `lime exec`、stdio transport、JSON-RPC、事件投影、
  JSONL/stdin/error-exit 检查通过。
- `npm run smoke:tui-gate-b`：通过，真实 CLI/TUI、stdio App Server、canonical projection、
  queue edit 与 terminal restore 检查通过。
- `git diff --check`：通过。

环境限制：本机 Darwin/aarch64 的 `rusty_v8 v150.4.0` 上游 archive 地址返回 404；验证使用本机
已校验的本地 archive/binding 注入，未将其纳入 release candidate。

## 收尾分类

- `current`：Code Mode protocol/runtime/host/facade、App Server、RuntimeCore、CLI/TUI、
  canonical Thread/Turn/Item projection、remote authenticated transport 与真实 Gate B fixture。
- `compat`：仅保留 `tool-runtime::code_mode` 的显式委托导出。
- `deprecated`：无新增。
- `dead / deleted`：旧 Code Mode process/V8 物理实现及已退役 CLI 入口。

当前完成度：验证与候选整理 95%；待用户确认后执行 release commit、tag、推送及远端复核。

## 2026-09-06 发布后 CI 修复

v1.141.0 首次质量与 Electron 发布 workflow 暴露了三类问题：CLI/TUI inventory JSON
被根 `.gitignore` 忽略而未进入 CI checkout；`code-mode-protocol` 的 `tonic-build`
依赖系统 `protoc`，导致 Linux/macOS/Windows Rust、GUI 和 Electron 构建均在代码生成阶段
失败；Windows 构建失败后仍执行 Squirrel cleanup，因 summary 尚未生成而产生二次 `ENOENT`
失败。修复如下：

- 将三份生成账本加入 `.gitignore` 例外，保留为静态治理事实源。
- 在 workspace 中加入 `protoc-bin-vendored`，由 `code-mode-protocol/build.rs` 设置
  `PROTOC`，消除平台工具链前置条件并同步 `Cargo.lock`。
- 导出并保护 `cleanupFromSummary`：summary 不存在时记录明确 no-op，不覆盖原始构建错误；
  增加缺失 summary 回归测试。

本地修复验证：`cargo check -p code-mode-protocol`、inventory/Windows Squirrel/release
workflow 共 75 项 Vitest、`npm run test:contracts` 均通过。待提交推送后复跑 Quality 与
Release workflows，确认跨平台构建及 Windows restricted execution setup 通过。

### 第二轮 Quality 修复

Quality run `33999878194` 在首轮修复后继续暴露三个独立问题：Windows restricted
execution 已执行 8 个测试，但 evidence collector 的 required matrix 仍只有 7 项；
Codex method product-scope fixture 引用了只在 Windows job 运行时生成、不会进入前端独立
checkout 的 `.lime/qc/windows-restricted-execution/summary.json`；App Server 在注册共享
process owner 后才检查同连接重复 `processId`，因此先返回 `RUNTIME_ERROR`，而不是协议要求的
`INVALID_REQUEST`。

当前修复口径：

- 将 `unelevated_mode_rejects_managed_network_before_setup` 纳入八项 required matrix，并由脚本
  单测自动比对 Rust integration target 的全部 `#[test]` / `#[tokio::test]`，防止测试清单再次
  漂移。
- 静态 product-scope fixture 只引用仓库内可追踪的 collector 与 Rust integration test；真实
  `.lime/qc/**` 继续只作为平台运行 artifact，不伪造、不提交。
- App Server 以 connection-scoped owner id 注册共享进程，在 spawn 前拒绝同连接 active 重复，
  并允许不同连接或同连接终态后复用公开 `processId`。

本地定向验证已通过：Windows evidence 与 Codex product-scope Vitest `19/19`；App Server
`command_exec` 相关 Rust 单测 `12/12`；真实工作树 `npm run test:contracts` 通过；只叠加本轮
候选文件的隔离 `HEAD` 快照中，`npm run typecheck` 与同一组 App Server 单测通过。当前工作树
直接执行 typecheck 只被未跟踪的并行 remote WebSocket 实现缺少 `ws` 类型声明阻断，该文件不
进入本轮修复提交。退出条件仍为 Quality 跨平台全绿；Windows 平台必须取得 current 八项矩阵
`8/8`，不能用历史 `7/7` 或本机非 Windows 测试替代。

### 第三轮 Quality 修复

Quality run `34009678395` 已确认 Windows restricted execution `8/8`、GUI Smoke、Integrity、
Frontend lint/typecheck/test-layer budget 通过；Frontend Full 在批次 57 命中治理测试的过期
路线图文案，Rust Full 在 `config_jsonrpc` 命中无外部写入的 `configVersionConflict`。修复批次
57 后继续续跑，又在批次 118 暴露 File Manager watcher 的默认 mock 返回同步 `void`，违反
`FileSystemWatchStop = () => Promise<void>` 契约并在卸载时触发 `.catch` 读取错误。

本轮修复口径：

- 治理测试改断言当前 README 中仍受保护的 P8 收口与 Windows unelevated runner 事实，不恢复
  已删除的旧路线图文案。
- File Manager 测试 fixture 返回真实异步 stop 函数，生产 watcher 契约和清理实现不降级。
- 配置版本摘要先递归规范化 JSON 对象键顺序再做 SHA-256；数组顺序、配置响应、持久化格式和
  乐观锁冲突语义保持不变。该修复消除 workspace 中 TUI 启用 `serde_json/preserve_order` 后，
  `Config` 内 `HashMap` 迭代顺序导致等价配置产生不同摘要的问题。

验证证据：Frontend 批次 57-117 在隔离候选中通过，修复 watcher fixture 后批次 118-119
`145/145` 通过；两个改动测试文件合并定向验证 `241/241` 通过，隔离候选的
`npm run typecheck` 通过，相关 ESLint、rustfmt 与 `git diff --check` 通过。Rust 已用同一
`cargo test --workspace` 场景在未修复隔离候选中精确复现 CI 冲突；修复后新增配置版本单测
`1/1` 通过，同一 workspace 特性统一场景中的
`config_control_plane_uses_the_single_desktop_yaml_layer` 通过且命令退出码为 `0`。共享工作树的
workspace 复验曾被并行 TUI 未完成模块写入阻断，因此不把该结果归因到本轮修复。Quality run
`34009678395` 的 Windows Shell Runtime 已最终通过，包括 restricted execution `8/8`、command
timeout、目录创建与 Agent Plugin MCP parity；该 run 仍因修复前的 Frontend Full 与 Rust Full
失败而保持失败结论。

现有 `v1.141.0` 本地与远端 tag 均继续指向 `5a7fe5af9`；本轮只追加 main 修复，不删除、覆盖
或重建既有 tag。退出条件仍为新修复提交推送后 Quality 全绿，并完成远端 main 状态复核。

### 第四轮 Quality 门禁超时修复

修复提交 `9714bb18d` 推送后，延迟触发的 push run `34014522753` 与手动 dispatch run
`34014175236` 均确认 Frontend Full、GUI Smoke 和 Integrity 通过，手动 run 还确认 Bridge &
Contracts 通过。两个独立 runner 的 Rust Full 都没有测试失败，而是在 job 启动满 `30m` 时由
GitHub Actions 取消 `Test Rust workspace`，annotation 明确为
`The job has exceeded the maximum execution time of 30m0s`。这说明 workspace 全量规模已超过旧
超时预算，继续重跑同一配置无法得到真实测试终态。

本轮将 Rust Full job 上限从 `30m` 调整为 `60m`，与 Windows Shell Runtime 的完整矩阵预算
一致；不改变测试命令、测试选择或失败语义。现有 workflow contract 已保护 Rust Full 的 V8
artifact 准备顺序，本次无需扩展协议或 bridge surface。退出条件仍是新提交触发的 Rust Full
取得真实成功终态，并与 Frontend Full、GUI Smoke、Integrity、Windows Shell Runtime 汇总为
Quality 全绿。
