# Lime v1.135.0 发布执行计划

状态：`release-ready / validation-complete`
日期：2026-08-25
目标版本：`1.135.0`
目标 tag：`v1.135.0`

## 主目标

发布当前 `main` 工作树中的 Agent Runtime MCP 生命周期与事件流、环境生命周期、Thread 控制、Provider capability、Strict Review、GUI projection 和多语言测试改动，完成版本事实源、双语单页 release notes、质量门禁及 release commit/tag/push。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划。
- `candidate changes`：当前工作树全部 162 个已跟踪 diff 文件与 46 个未跟踪文件，覆盖 Rust agent/app-server/mcp/tool-runtime、App Server protocol/schema、前端 Agent/MCP/Workspace、Electron Gate B、脚本治理、测试、文档与发版 metadata。
- `excluded changes`：无。用户请求为完整发布，当前工作树改动均纳入本轮 release candidate。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.135.0`；双语 release notes 只保留 v1.135.0；目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 必须通过；按风险执行 contracts、受影响 Rust/前端测试、current fixture、GUI smoke、Electron Gate B 与治理扫描，未执行或失败项原样记录。
- staged 内容与候选范围一致；完成 `Release v1.135.0` commit、`v1.135.0` tag，推送 `main` 和 tag，并复核本地/远端状态。

## 架构确认

本轮不新增产品链 owner 或 public boundary；候选继续沿用 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`。现有架构事实源为 `internal/aiprompts/architecture.md` 与 Codex 对齐执行计划。

## 验证记录

| 命令 | 结果 | 证据/说明 |
| --- | --- | --- |
| `npm run verify:app-version` | 通过 | 根应用、CLI npm 包、Rust workspace、Cargo.lock 与 sidecar 统一为 `1.135.0` |
| `npm run typecheck` | 通过 | renderer 与 node TypeScript 发布硬门禁通过 |
| `npm run check:protocol-types` | 通过 | 1037 个 definitions、1029 个 generated types，无漂移 |
| `npm run test:contracts` | 通过 | 299 项 App Server client 检查及 command/harness/modality/scripts/release/docs 边界通过 |
| 前端受影响定向 Vitest | 通过 | 28 个文件、264 项测试通过；覆盖 Agent/MCP/环境/Thread/Strict Review/事件投影与治理边界 |
| Electron Gate B 脚本守卫 | 通过 | 8 个文件、27 项测试通过 |
| `npm run test:rust:changed` | 通过 | 版本触达 workspace 边界后执行 `cargo test --lib --workspace`，全部 crate unit 通过 |
| `npm run verify:gui-smoke` | 通过 | 真实 Electron/preload/IPC/App Server `1.135.0`、Claw workbench reload 与 memory settings 通过；证据 `.lime/qc/project-gates/standalone-shell-01-20260825150650-63360/` |
| `npm run smoke:agent-runtime-current-fixture` | 通过 | 全部 current fixture 场景通过；`liveProviderUsed=false` |
| `npm run governance:legacy-report` | 通过 | 零引用候选、分类漂移与边界违规均为 0 |
| `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check` | 通过 | Rust 格式一致 |
| `git diff --check` | 通过 | patch hygiene |

`npm run test:changed` 因 workspace 依赖图扩展为 118 个长批次，在 6/118 前主动停止，改用上述 28 文件定向测试并全部通过。Rust workspace、current fixture 与 GUI smoke 初次并发构建曾触发 `install_name_tool cannot rename ... app-server.XXXXXX` artifact 竞态；Rust 测试完成后构建自动串行恢复，最终 current fixture 与 GUI smoke 均通过。

## 收尾记录

- `current`：Agent Runtime、App Server JSON-RPC、MCP、环境生命周期、Thread 控制、Provider capability、Strict Review、GUI projection。
- `compat`：不新增 compat wrapper；仅保留仓库现有迁移边界。
- `deprecated`：不新增 deprecated owner。
- `dead / deleted`：不恢复已删除 runtime、fallback、旧 catalog 或 mock 生产路径。
- 当前完成度：`98% / release-ready`。候选范围、版本、release notes、TypeScript/contracts/Rust/前端/GUI/current fixture/治理门禁已完成；仅待 release commit、tag、push 与远端复核。
