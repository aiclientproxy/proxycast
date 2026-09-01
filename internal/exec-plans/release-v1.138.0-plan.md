# Lime v1.138.0 发布执行计划

状态：`release-ready / git-confirmation-pending`
日期：2026-09-01
目标版本：`1.138.0`
目标 tag：`v1.138.0`

## 主目标

发布 `v1.137.0` 之后当前工作树中的 Composer、prompt history、模型路由、Desktop capability/readiness、macOS native host、跨平台资源 manifest、Windows packaged Gate B、Agent GUI 与治理改动，完成版本事实源、双语单页 release notes、质量门禁及 release commit/tag/push。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md` 与本计划。
- `candidate changes`：当前工作树中的全部已跟踪产品/文档/测试/schema/生成物改动，以及新增的 Desktop Host/native helper、platform/resource contract、prompt history、Composer Controller、thread queue typed action和对应计划/测试文件。
- `excluded changes`：根目录 `history-DndxIXaj-D1E13r2C.js`、`history-Y-6ViEXW.js`、`local-conversation-page-B5LUHmAw.js`、`use-chatgpt-composer-controller-CrYCVUro.js`。这些文件是 Codex Desktop 压缩 bundle 只读参考，不是 Lime 源码或发布产物。
- `package.json` 边界：工作树曾被外部 Codex Desktop `openai-codex-electron@26.825.41651` manifest 替换；发布准备恢复 Lime `HEAD` manifest 的脚本、依赖和仓库身份，仅把版本提升到 `1.138.0`，不把外部工程 metadata 纳入候选。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.138.0`；双语 release notes 只保留 v1.138.0；本地和远端目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 必须通过；执行 `npm run test:contracts`、`npm run verify:gui-smoke`、Agent current fixture、受影响 Rust/前端定向测试和 `git diff --check`。
- staged 内容包含全部 candidate changes 和 release metadata，明确排除 4 个 Codex bundle 参考文件；完成 `Release v1.138.0` commit、`v1.138.0` tag，推送 `main` 和 tag，并复核远端 Release/Windows workflow。

## 架构确认

本轮重大架构变更已在 `internal/aiprompts/architecture.md` 记录并确认：Composer Controller、App Server PromptHistoryStore、Desktop capability/readiness、macOS native host 与 desktop resource manifest 均有唯一 current owner；Electron 继续只承接 Desktop Host/IPC/sidecar 生命周期，Agent 产品链保持 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`。责任开发者确认：root，2026-09-01。

## 验证记录

已通过：

```text
npm run verify:app-version
npm run typecheck
npm run test:contracts
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
npm test -- --resume                           # 119/119 批次
npx vitest run <受影响前端/Electron/脚本精确入口>  # 36/36
npx eslint <受影响前端路径> --max-warnings 0
git diff --check
```

- `test:contracts`：App Server client 299 checks，modality、命令、脚本、release workflow 与文档边界均通过。
- 完整 Vitest：首次运行前 56 批通过；修复 `electron/systemUtilityHost.ts` 已明确使用但未登记的 macOS security-scoped bookmark 存储根基线后，从第 57 批续跑完成全部 119/119 批次；定向治理测试 15/15 通过，未放宽扫描规则。
- Agent current fixture：完整覆盖首页长短输入、停止后继续、审批、Plan、Skills/MCP、媒体、代码工作台和文章工作台；`liveProviderUsed=false`。
- 首页 Gate B：`claw-chat-current-fixture-home-hotpath-prewarm-fix-summary.json` 与聚合回归均为 `ok=true`；预热 session 被正式发送复用，输入后、提交切换和完成后三个稳定窗口均通过。
- 最终 GUI smoke：`.lime/qc/project-gates/standalone-shell-01-20260901112400-63590/shell-01-electron-smoke/summary.json` 为 `result=pass`，App Server `protocol=appserver.v0 version=1.138.0`；此前完整 GUI shell Gate B-F `.lime/qc/project-gates/standalone-shell-01-20260901104344-8413/shell-01-electron-smoke/summary.json` 为 21/21，真实 Electron/preload/IPC/App Server JSON-RPC 命中，错误、legacy command 与 mock fallback 均为 0。
- `npm run test:related -- ...` 的收集器曾把 `electron/` 目录作为文件读取并报 `EISDIR`；直接 Vitest 精确入口覆盖相同受影响测试并全部通过，不属于测试断言失败。
- 额外执行的完整 Rust workspace 测试未完成：裸 `npm run test:rust` 下载 denoland `rusty_v8 v150.4.0` Apple Silicon 预编译包时遇到上游 URL 404；改用仓库 `resolveRustyV8CargoEnv` 后完成核心 crate 编译，但在链接 `app-server` 测试产物时因本机磁盘仅余约 1.4 GiB 报 `No space left on device`。该项不是默认发布硬门禁，未将环境失败表述为测试通过。
- Windows packaged Squirrel、安装后 Agent/CodeMode Gate B 和发布资产尚未执行；它们必须在 commit/tag 推送后由 `windows-2022` release workflow 生成平台证据。

## 收尾记录

- `current`：Composer Controller、prompt history、App Server typed client、Desktop capability/readiness、macOS native host、desktop resource manifest、Windows packaged Gate B、canonical model/reasoning/queue projection。
- `compat`：无新增 compat wrapper。
- `deprecated`：无新增 deprecated owner。
- `dead / deleted`：外部 Codex manifest 和 4 个压缩 bundle 不进入 release；不恢复旧 runtime、生产 mock fallback 或第二持久化 owner。
- 当前完成度：`85%`；release candidate、版本事实源、双语 release notes 与本地必要门禁已收口，等待危险操作确认、commit/tag/push、GitHub Release 和 Windows packaged Gate B。
