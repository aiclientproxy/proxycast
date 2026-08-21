# Lime v1.133.0 发布执行计划

状态：`release-candidate / validation-passed / git-confirmation-pending`
日期：2026-08-22
目标版本：`1.133.0`
目标 tag：`v1.133.0`

## 主目标

发布当前 `main` 工作树中的 Browser Workspace / Right Surface、Browser 动态工具审批与用户接管、Agent 会话恢复和流控收敛切片，完成版本事实源、双语单页 release notes、质量门禁及 release commit/tag/push。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划。
- `candidate changes`：当前工作树中全部 104 个已跟踪 diff 和 14 个未跟踪文件（共 118 个 status 条目），覆盖 Electron BrowserTabHost/App Server Host、Browser Workspace、动态工具和审批协议、Rust agent/app-server/model-provider/tool-runtime、Agent 会话恢复与流控、Memory 设置提示生命周期、脚本、测试、架构、路线图和治理文档。
- `excluded changes`：无。用户请求为完整发布，当前工作树改动均纳入本轮 release candidate。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.133.0`；双语 release notes 只保留 v1.133.0；目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 必须通过；按风险执行 contracts、受影响 Rust/前端测试、Browser Electron Gate A/Gate B、Agent current fixture、GUI smoke、治理与本地门禁，未执行或失败项原样记录。
- staged 内容与上述候选范围一致；获得危险操作确认后创建 `Release v1.133.0` commit、`v1.133.0` tag，推送 `main` 和 tag，并复核本地/远端状态。

## 当前验证记录

- 发布前基线为 `v1.132.0`（commit `18fe8e5cd`），发布前 `v1.133.0` 本地和远端 tag 均不存在。
- 版本事实源与双语单页 release notes 已更新并通过一致性校验。
- 首轮完整 Vitest 在第 54 批暴露 Codex upstream native-tool policy 测试漂移；测试已改为兼容当前 `UnifiedExec/Disabled`、serde alias 与 `UnifiedExecShellMode`，该批及后续批次通过。第 83 批首跑出现 worker timeout，精确重跑和后续续跑通过。
- 首轮 `verify:local` 在原第 70 批暴露 Memory 设置页卸载后仍触发提示清理 timer；提示生命周期已收敛到父组件唯一 owner，重复提示替换 timer、卸载时清理，子面板不再创建独立 timer。Memory 页面、状态面板及相邻测试共 16 项定向复跑通过。
- 修复后 `npm test -- --resume` 从失败断点至第 116 批全部通过；最终 `npm run verify:local` 从头执行并以退出码 0 完成，116 批 Vitest、contracts、Rust workspace 单元测试和真实 Electron GUI smoke 均通过。
- `git diff --check` 通过；最终 candidate 无排除项，`v1.133.0` 本地与远端 tag 仍不存在。

## 架构确认

架构影响：重大。候选触及 BrowserTabHost 的 WebContentsView/BrowserRoute 唯一 owner、Browser action 两阶段审批、用户接管时的 live execution 撤销、App Server reverse request、Agent session-loop owner 投影与跨平台 tool-runtime 环境构造。相关确认已记录于 `internal/aiprompts/architecture.md`、`internal/roadmap/browser/README.md` 和对应执行计划。

责任开发者：root，确认日期：2026-08-21。

## 验证记录

| 命令 | 结果 | 证据/说明 |
| --- | --- | --- |
| `npm run verify:app-version` | 通过 | 根应用、CLI npm 包、Rust workspace、Cargo.lock 与 App Server sidecar 均为 `1.133.0` |
| `npm run typecheck` | 通过 | 发布硬门禁最终复跑通过；`verify:local` 内再次通过 |
| `npm run test:contracts` | 通过 | protocol types 无漂移，App Server client 299 checks，command/harness/modality/scripts/release/docs guards 通过 |
| `npm run test:related -- ...` | 通过 | 当前候选前端/Electron 定向测试通过；Memory timer 修复额外 16 项定向测试通过 |
| `npm run test:rust:related -- ...` | 通过 | agent/app-server/model-provider/tool-runtime 相关测试通过 |
| `npm run smoke:browser-runtime-gate-a` | 通过 | Browser projection Gate A 通过 |
| `npm run smoke:browser-runtime-electron-gate-b` | 通过 | lifecycle、approval、cancel、window-close、user-control、disconnect、permission、download 场景通过 |
| `npm run smoke:agent-runtime-current-fixture` | 通过 | Agent current fixture 通过 |
| `npm run verify:gui-smoke` | 通过 | 真实 Electron renderer、preload/IPC、App Server sidecar、Claw shell reload 与 Memory settings 通过 |
| `npm run governance:legacy-report` | 通过 | 零引用候选 0、分类漂移候选 0、边界违规 0 |
| `npm test -- --resume` / `npm test` | 通过 | 断点续跑至 116 批通过；最终 `verify:local` 内 116 批从头一次通过 |
| `npm run verify:local` | 通过 | 最终复跑退出码 0，覆盖 app-version、i18n、lint、typecheck、Vitest、contracts、Rust workspace 与 GUI smoke |
| `git diff --check` | 通过 | patch hygiene 无错误 |

本地不宣称 Windows Notification Center、真实 macOS/Windows sleep-resume、签名、公证、正式 release assets 或 live provider 证据；这些由对应平台/CI runner 提供。

## 收尾记录

- `current`：Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI，以及 BrowserTabHost、Browser Workspace、Browser dynamic capability/approval、Agent session recovery/flow control。
- `compat`：仅保留仓库中明确标记且不承接新 current 创建路径的迁移边界；本候选不新增 compat wrapper。
- `deprecated`：旧 Browser surface、旧自动化/外部 CDP 边界按现有治理分类，仅由 current 路径迁出，不作为新发布 owner。
- `dead / deleted`：候选中已经删除的脱离构建图文档/旧入口继续保持删除状态，不恢复旧 runtime 或 fallback。
- 当前完成度：`95% / validation-ready`。版本、release notes、候选范围和门禁已完成；仅待用户明确危险操作确认后执行 release commit、tag、push 与远端 tag 复核。
