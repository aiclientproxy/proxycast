# Lime v1.137.0 发布执行计划

状态：`release-authorized / commit-tag-push-pending`
日期：2026-08-31
目标版本：`1.137.0`
目标 tag：`v1.137.0`

## 主目标

发布 `v1.136.0` 之后当前工作树中的 Agent Runtime/App Server、MCP 通知、权限 profile、环境选择、线程 fork、Workspace/GUI 与 Electron Gate B 改动，完成版本事实源、双语单页 release notes、质量门禁及 release commit/tag/push。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划及执行计划索引。
- `candidate changes`：当前工作树中 134 个已跟踪修改和 7 个产品/测试/脚本未跟踪文件，覆盖 App Server/Rust protocol、MCP、Agent Runtime、Workspace/Canvas、Electron bridge、脚本和五语言回归。
- `excluded changes`：`local-conversation-page-B5LUHmAw.js`。该文件在 Codex App GUI 对齐计划中标记为只读构建参考，不属于源代码或发布产物。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.137.0`；双语 release notes 只保留 v1.137.0；目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 必须通过；按风险执行 `npm run test:contracts`、`npm run verify:gui-smoke`、current fixture 及受影响 Rust/前端定向测试，未执行或失败项原样记录。
- staged 内容只包含 release metadata 与 candidate changes，不包含排除的构建参考文件；完成 `Release v1.137.0` commit、`v1.137.0` tag，推送 `main` 和 tag，并复核本地/远端状态。

## 架构确认

本轮不新增产品链 owner；权限 profile、MCP 工具生命周期与环境状态继续归 App Server/RuntimeCore 及各 Rust domain owner，Electron 只承接 Desktop Host/IPC，GUI 通过 App Server JSON-RPC 投影读取 Thread/Turn/Item。当前工作树的对齐计划和命令边界更新纳入候选。

## 验证记录

已完成：

```text
npm run verify:app-version
npm run typecheck
npm run test:contracts
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
npm run test:rust:related -- <受影响 Rust 路径>
npm run smoke:thread-fork-electron-gate-b
npm run smoke:mcp-list-changed-electron-gate-b
node --test packages/agent-runtime-projection/tests/threadSettingsLiveUpdate.test.mjs
npx vitest run <受影响前端测试文件>
git diff --check
```

验证结果：

- `npm run verify:app-version`：通过，根应用、CLI npm 包、Rust workspace 与 Cargo.lock 均为 `1.137.0`。
- `npm run typecheck`：通过，renderer 与 node TypeScript 均无错误。
- `npm run test:contracts`：通过，协议生成无漂移，App Server client 299 项及 command/harness/governance/docs 边界通过。
- `npm run test:rust:related -- lime-rs/crates/agent lime-rs/crates/app-server-protocol lime-rs/crates/app-server lime-rs/crates/mcp`：通过，相关 crate 与反向依赖 unit tests 全部通过。
- `npx vitest run <受影响前端测试文件>`：23 个 suite、398 个断言通过；Node 原生 `threadSettingsLiveUpdate` 另以 `node --test` 运行 8/8 通过。
- `npm run verify:gui-smoke`：通过，真实 Electron Shell-01；App Server `appserver.v0` 版本 `1.137.0`，preload/IPC/页面/console 错误均为 0。证据：`.lime/qc/project-gates/standalone-shell-01-20260831011954-42452/shell-01-electron-smoke/summary.json`。
- `npm run smoke:agent-runtime-current-fixture`：通过，覆盖 current Agent Runtime Electron fixture 全集，`liveProviderUsed=false`。
- `npm run smoke:thread-fork-electron-gate-b`：通过，证明 GUI fork、Thread lineage、read/resume 和 current JSON-RPC bridge。
- `npm run smoke:mcp-list-changed-electron-gate-b`：首次因脚本未统计 direct preload 调用而误报 `缺少 app_server_handle_json_lines`；修复观测逻辑后复跑通过，三种 MCP list-changed 通知和动态目录均可见，`mockFallbackHitCount=0`，证据：`.lime/qc/gui-evidence/mcp-list-changed-electron-gate-b/mcp-list-changed-electron-gate-b-summary.json`。
- `git diff --check`：通过。

## 收尾记录

- `current`：Agent Runtime canonical projection、MCP list-changed notification、permission profile policy、thread fork/environment state、真实 Electron/App Server bridge、Workspace activity/Right Surface。
- `compat`：无新增 compat wrapper。
- `deprecated`：无新增 deprecated owner。
- `dead / deleted`：不纳入构建参考文件，不恢复旧 runtime、catalog fallback 或生产 mock 路径。
- 当前完成度：`99%`；候选范围、版本元数据、release notes 和全部默认门禁已完成，已获得用户确认，正在执行 commit/tag/push 及远端复核。
