# Lime v1.111.0 发布执行计划

状态：post-release-fix-complete
日期：2026-07-24
目标版本：`1.111.0`
目标 tag：`v1.111.0`

## 主目标

将基于 `v1.110.0` 的当前工作树作为单一 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、main/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：release candidate 已完成版本事实源、双语 release notes、current contract 迁移、i18n dead key 清理与全部发布前只读门禁。
- 下一刀：补丁版本 `v1.111.1` 的独立发布计划见 `release-v1.111.1-plan.md`。

## Release Candidate

- 基线：`v1.110.0`；任务开始时本地 `main` 与 `origin/main` 同指 `15543076c`，目标 tag 在本地和远端均不存在。
- 初始候选：211 个修改、10 个删除、20 个新增，共 241 个代码、协议/schema、生成类型、GUI、脚本、测试与文档文件；tracked diff 为 9,946 insertions / 5,583 deletions。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes、本计划与执行计划索引。
- `candidate changes`：任务开始时工作树中的全部 current Rust/TypeScript、App Server v2 protocol、Agent runtime、GUI、脚本、治理、文档与测试改动，以及门禁需要的最小修复。
- `excluded changes`：无；只读扫描未发现凭证、日志、构建产物、缓存或临时文件。GUI 构建产物位于既有忽略目录，不进入候选。

## 窄写集与避让

发布准备只写版本事实源、双语 release notes、`internal/aiprompts/architecture.md`、本计划与执行计划索引；其余候选改动仅审阅、验证并纳入发布。门禁暴露缺陷时，只修改对应 current owner 的最小写集。

## 退出条件

1. 所有版本事实源同步为 `1.111.0`，双语 release notes 仅保留当前版本单页。
2. `npm run verify:app-version` 与 `npm run typecheck` 通过。
3. 协议、schema、generated client 与 renderer gateway 变更通过 `npm run test:contracts`。
4. Rust/runtime 改动通过 related/integration 验证；Agent 主链通过 `npm run smoke:agent-runtime-current-fixture`。
5. `npm run verify:gui-smoke` 取得真实 Electron Gate B，或记录可复现环境限制。
6. `npm run governance:legacy-report`、`npm run governance:scripts` 与 `git diff --check` 通过。
7. staged 集覆盖全部 candidate changes 与 release metadata，且不存在未纳入的本地改动。
8. 获得危险操作确认后，连续完成 `git add`、`git commit -m "Release v1.111.0"`、`git tag v1.111.0`、`git push origin main`、`git push origin v1.111.0` 与远端复核。

## 验证记录

- `npm run verify:app-version`：通过，全部版本事实源为 `1.111.0`。
- `npm run typecheck`：通过；renderer、node 与 Electron host typecheck 均通过。
- `npm run test:contracts`：通过；761 个生成协议类型无漂移，App Server client 286 项检查通过，命令、Harness、modality、scripts、release workflow、docs boundary 均通过。
- Rust related/integration：通过；changed scope 因 workspace manifest 触达扩大为 `cargo test --lib --workspace`，全 workspace lib 通过；App Server integration 通过，包含 1497 个 App Server lib tests 与目标 JSON-RPC tests。
- `npm run smoke:agent-runtime-current-fixture`：通过；覆盖 history/cache、stream completion、Claw/Electron、Workbench、media、Plan、Skills、MCP、approval allow-for-session/decline/cancel、session cache 与内容工厂 current fixtures。
- `npm run smoke:claw-chat-current-fixture -- --scenario approval-request-resume --timeout-ms 180000`：通过；typed approval pending 从 `thread/read` + renderer lifecycle 读取，显式 `acceptForSession` 与第二回合 session cache 均通过。
- `npm run verify:gui-smoke`：通过（最终候选复跑）；真实 Electron renderer/preload/IPC、App Server sidecar `appserver.v0` `1.111.0`、Workbench shell 与 memory settings evidence 均通过。
- `npm run governance:legacy-report`：通过，零分类漂移与边界违规。
- `npm run governance:scripts`：通过，root/一级 scripts 冻结基线无违规。
- `npm run i18n:check`：通过，5 locale、13 namespace、覆盖率 100%。
- `npm run i18n:unused -- --check`：通过，`unused=0`；删除 38 个生产零引用 dead key 并同步 5 locale。
- 定向候选 eslint：通过；相关 Agent stream、Plugin fixture 与 i18n tests 均无 lint error，相关回归测试通过。
- `git diff --check`：通过。
- `npm run verify:local`：已执行但被 13 个未被本候选修改的基线文件 lint error 阻断；候选触及的 lint error 已修复并定向验证。该限制不归因于 v1.111.0 候选，不能写成全量 local gate 通过。
- `cargo fmt --all --check`：被候选集中其他既有未格式化文件阻断（`agent-protocol/src/lib.rs`、`app-server/src/runtime/agent_mailbox_delivery.rs`、`app-server/tests/session_archive_jsonrpc.rs`）；本次修改的 `media_task_jsonrpc.rs` 单独格式检查通过，未擅自格式化脏热区。

## 架构确认

- 影响：重大。候选涉及 App Server v2 `artifact/write`、direct notification、RuntimeCore、Thread/Turn/Item projection、MCP/tool lifecycle、history/read model 与 Renderer gateway。
- 架构事实源：`internal/aiprompts/architecture.md` 同步记录 artifact write owner、扩展 direct v2 notification 与 legacy runtime-event append 的 dead 边界。
- 责任人：root（release owner，v1.111.0）。
- 日期：2026-07-24。
- 确认状态：confirmed；contracts、Rust、Agent current fixture 与真实 Electron Gate B 均证明产品链保持 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`，未引入第二套业务后端或生产 mock fallback。

## 分类与剩余限制

- `current`：App Server v2 `artifact/write`、direct Thread/Turn/Item notifications、RuntimeCore、canonical ThreadStore/history、typed client 与 Renderer projection。
- `compat`：仅保留未覆盖 provider/media diagnostic 的显式 raw side-channel，不承接 canonical lifecycle。
- `deprecated`：仍存在的旧入口只允许迁出，不扩展新能力。
- `dead / deleted / forbidden-to-restore`：`agentSession/runtimeEvents/append`、legacy plugin runtime gateway、旧 prompt builder、旧 tool permission/shell security 实现与正向测试夹具，以及本次清理的 38 个未引用 i18n key。
- 平台限制：macOS 真实 Electron Gate B 已通过；未执行 Windows 真机与正式 packaged artifact 验证，不将其写成通过。

当前 release candidate 完成度：`100%`；提交 `94b784718`、`main` 与 `v1.111.0` 已推送并完成远端一致性复核。

## 发布后故障修复（2026-07-25）

- GitHub Actions 失败证据：Release attempt 1 的 macOS x64 sherpa 下载、Windows sccache 下载与 macOS arm64 Electron 下载均返回 GitHub `504`；Docs 因工作流与根 `packageManager` 重复声明 pnpm 版本失败；Quality 因 13 个 lint error 与 2 个 warning 失败。
- Release 处理：已重跑失败作业，attempt 2 越过原始 sherpa 与 sccache 失败点；同时把 Electron Forge `504 Gateway Time-out` 纳入现有 macOS 瞬时错误重试守卫。
- 首轮修复写集：`.github/workflows/{release,deploy-docs}.yml`、release workflow guard、CI 报出的 lint 文件，以及共享 action 类型 owner `src/lib/api/agentActionTypes.ts`。
- 本轮新增修复写集：`src/components/agent/chat/workspace/workspaceSendHelpers.test.ts` 将 plan mode 的旧 Renderer goal metadata 正向断言改为负向回流守卫；`src/components/agent/chat/utils/executionStrategyCurrentBoundary.test.ts` 删除已删除 queue controller 和已不消费 execution runtime 类型的旧清单项。
- 版本与远端事实：`v1.111.0` 已发布，现有 release commit `94b784718`、`main` 与 tag 不重写；本轮只创建 post-release fix commit 推送 `main`。
- 验证结果：定向 workspace/helper 与 boundary 测试通过；`npm test -- --resume` 完成 `112/112`；`npm run lint`、`npm run typecheck`、`npm run verify:app-version`、`npm run test:contracts`、`npm run governance:legacy-report`、`npm run governance:scripts`、`npm run governance:electron-release-workflow`、`git diff --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server`、`npm run smoke:agent-runtime-current-fixture` 与 `npm run verify:gui-smoke` 均通过。
- `npm run verify:local`：未通过；smart 全量从第 1 批执行至第 4 批时，既有 `src/components/agent/chat/index.projectRestore.test.tsx` 出现漂移的异步超时 / workspace 断言失败，单文件复现同样会在不同测试间漂移。本轮未修改该无关业务/测试基础设施，也未将其写成通过。
- 退出条件：上述定向与跨层门禁通过，`verify:local` 的基线异步失败已记录；补充修复提交 `114455c39` 已推送到 `main`，`v1.111.0` tag 未改写。
