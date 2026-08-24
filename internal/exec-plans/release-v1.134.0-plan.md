# Lime v1.134.0 发布执行计划

状态：`release-ready / validation-complete`
日期：2026-08-24
目标版本：`1.134.0`
目标 tag：`v1.134.0`

## 主目标

发布当前 `main` 工作树中的 Browser historical surface、App Server v2 环境与 MCP 事件流、Provider capability、Windows sandbox readiness、Agent 历史投影和多语言验证切片，完成版本事实源、双语单页 release notes、质量门禁及 release commit/tag/push。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划。
- `candidate changes`：当前工作树中的全部产品、协议、schema、生成类型、文档、测试、脚本与治理改动（299 个已跟踪 diff 文件、108 个未跟踪候选文件），覆盖 Electron Browser/App Server Host、Browser historical Workspace、环境/MCP/Provider capability/Windows sandbox 协议、Rust agent/app-server/mcp/tool-runtime、Agent session projection、GUI、多语言、脚本、测试、架构、路线图和治理文档。
- `excluded changes`：无。用户请求为完整发布，当前工作树改动均纳入本轮 release candidate。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.134.0`；双语 release notes 只保留 v1.134.0；目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 必须通过；按风险执行 contracts、受影响 Rust/前端测试、Browser Electron Gate B、Agent current fixture、GUI smoke、治理与本地门禁，未执行或失败项原样记录。
- staged 内容与上述候选范围一致；获得危险操作确认后创建 `Release v1.134.0` commit、`v1.134.0` tag，推送 `main` 和 tag，并复核本地/远端状态。

## 当前验证记录

- 发布前基线为 `v1.133.0`（commit `489499f12`），`v1.134.0` 本地和远端 tag 均不存在。
- 版本事实源与双语单页 release notes 已更新；`npm run verify:app-version` 已确认根应用、CLI npm 包、Rust workspace、Cargo.lock 与 sidecar 版本统一为 `1.134.0`。
- 当前候选触及 Electron/preload/App Server JSON-RPC、Rust workspace、GUI 主路径和脚本治理；typecheck、contracts、Rust related、Browser Gate B、current fixture、GUI smoke、治理扫描和本地门禁已按下表完成或记录边界。

## 架构确认

架构影响：重大。候选扩展 App Server v2 的环境/MCP/Provider capability 边界、Browser historical surface、Agent Thread/Turn/Item projection 和 Windows sandbox readiness；确认以 `internal/aiprompts/architecture.md`、对应 Browser/Codex 执行计划和本计划为事实源。

责任开发者：root，确认日期：2026-08-23。

## 验证记录

| 命令 | 结果 | 证据/说明 |
| --- | --- | --- |
| `npm run verify:app-version` | 通过 | 版本一致性为 `1.134.0` |
| `npm run check:protocol-types` | 通过 | 1037 个 definitions、1029 个 generated types，无漂移 |
| `npm run typecheck` | 通过 | renderer 与 node TypeScript 发布硬门禁通过 |
| `npm run test:contracts` | 通过 | protocol schema/generated client 无漂移，299 项 app-server client 检查及 command/harness/governance/docs 边界通过 |
| 前端 Vitest 全量续跑 | 通过 | 116/116 批全部通过；MCP 定向测试 16 项通过 |
| `npm run test:rust:related -- ...` | 通过 | 20 个受影响及反向依赖 crate；App Server unit 1709/1709、binary unit 26/26 通过 |
| `thread/revert` public JSON-RPC integration | 通过 | 5/5；覆盖 revert 响应、通知与 read model |
| MCP event stream public JSON-RPC integration | 通过 | 连续 3 次通过；覆盖 event 早于 active 的启动竞态与激活前有界缓冲 |
| Agent Runtime Client package | 通过 | 24/24；覆盖 `thread/reverted` 非实体生命周期通知旁路 |
| `npm run smoke:browser-runtime-electron-gate-b` | 通过 | 真实 Electron Browser Gate B；证据 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-summary.json` |
| MCP structured content Electron Gate B | 通过 | 真实 Electron、preload/IPC、App Server JSON-RPC、runtime/read model 与可见状态通过 |
| `npm run smoke:agent-runtime-current-fixture` | 通过 | 全部 current fixtures 通过；`liveProviderUsed=false`，不作为 live provider 证据 |
| `npm run verify:gui-smoke` | 通过 | 真实 Electron renderer、preload/IPC、App Server `1.134.0`、Claw workbench、reload 与 memory settings 通过；证据 `.lime/qc/project-gates/standalone-shell-01-20260824082239-54560/` |
| `npm run governance:legacy-report` | 通过 | 零引用、分类漂移、边界违规均为 0 |
| `npm run verify:local` | 部分通过 | 前端与大多数门禁通过；旧运行在 Rust workspace 编译阶段暴露问题，已由最终 Rust related 与公共 JSON-RPC 定向测试独立收口 |
| App Server production assets build | 通过 | 正式 App Server sidecar assets 构建通过；不等同于完整 Forge release assets |
| `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check` | 通过 | Rust 格式一致 |
| `git diff --check` | 通过 | patch hygiene |

`npm run test:related -- ...` 在 Vitest 收集阶段因目录参数命中 `EISDIR electron`，未进入测试正文；已改用 Agent Runtime Client package 官方入口并通过 24/24。裸 Cargo 曾因 upstream `rusty_v8` archive HTTP 404 失败；仓库 wrapper 使用本地 artifact，相关 Rust 测试已通过。

本地不宣称 Windows Notification Center、真实 macOS/Windows sleep-resume、Windows installer、签名、公证、完整 Forge release assets 或 live provider 证据；这些由对应平台/CI runner 提供。

## 收尾记录

- `current`：Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI，以及 Browser historical、environment、MCP event stream、Provider capability、Windows sandbox readiness。
- `compat`：仅保留仓库中明确标记且不承接新 current 创建路径的迁移边界；本候选不新增 compat wrapper。
- `deprecated`：旧 Browser surface、旧 Playwright browser tool 和旧自动化边界按现有治理分类，仅由 current 路径迁出，不作为新发布 owner。
- `dead / deleted`：候选中已经删除的脱离构建图脚本和资源继续保持删除状态，不恢复旧 runtime、fallback 或 compat 包装。
- 当前完成度：`98% / release-ready`。候选范围、版本、release notes、前端/contracts/Rust/GUI/Gate B/current fixture/治理门禁已完成；仅待已获确认的 release commit、tag、push 与远端复核。
