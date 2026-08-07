# Lime v1.123.0 发布执行计划

状态：git-release-authorized
日期：2026-08-07
目标版本：`1.123.0`
目标 tag：`v1.123.0`

## 主目标

以 `v1.122.0` 为基线，将当前工作树中的 Hook lifecycle、Skills exact control plane、Plugin search、Apps
catalog/readiness、Thread-scoped MCP resource/tool、模型目录恢复、App Server v2 canonical projection、GUI、测试、schema、
生成客户端和架构文档整理为单一 release candidate，完成版本事实源、双语 release notes、发布门禁、
release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：release candidate、版本事实源、双语 release notes 与发布门禁均已收口，用户已明确确认 Git 高风险操作。
- 下一刀：连续完成 stage、release commit、annotated tag、`main`/tag 推送、远端引用与 GitHub Release 工作流复核。

## Release Candidate

- 基线：`v1.122.0`；本地与远端 `v1.123.0` tag 均不存在；`main` 与 `origin/main` 起点一致。
- 最终盘点：225 个进入候选的 unstaged tracked diff 文件、1 个既有 staged 文件、68 个未跟踪文件，
  加上唯一排除的 `.gitignore` 共 295 个状态路径；tracked working-tree diff 为 226 个文件、21,752 行新增、
  5,613 行删除。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、
  `lime-rs/Cargo.lock`、双语 release notes、本计划，以及上一版已完成远端发布事实的
  `internal/exec-plans/release-v1.122.0-plan.md`。
- `candidate changes`：除 `.gitignore` 外的当前工作树全部产品、协议、schema、生成客户端、Rust runtime、
  Electron、GUI、五语资源、测试、脚本和文档改动。
- `excluded changes`：`.gitignore` 新增的本地 `.gstack/` 忽略项；该个人环境改动沿用 `v1.122.0` 的排除策略，
  不进入 release commit。

## 写集与退出条件

- Release owner 只修改版本事实源、双语 release notes 与本计划；其余 candidate 文件只读审计、验证和暂存。
- 所有版本事实源统一为 `1.123.0`，release notes 只保留当前版本单页。
- 必跑 `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、受影响 Rust related、
  `npm run smoke:agent-runtime-current-fixture`、`npm run smoke:hook-lifecycle-gate-b`、
  Provider generation PendingRoute Gate B、MCP current/Workspace Electron fixture、`npm run verify:gui-smoke`、
  `npm run governance:legacy-report`、`npm run governance:scripts` 与 `git diff --check`。
- 候选修改 `internal/aiprompts/architecture.md` 且新增 Hook/Skills/MCP current protocol owner；release owner
  必须在 Git 高风险操作确认中确认架构图、owner 与 current 产品链一致。
- Git 写操作（stage、commit、tag、push）须在门禁完成后按危险操作格式取得一次明确确认。

## 已知风险

- Provider generation + PendingRoute cold-restart 曾暴露持久化 AgentControl catalog route 在凭证暂不可用时进入
  reconciliation 并回退模型，同时 child route 仅写入内存/EventLog。当前候选已改为 schema-valid 持久化 route
  禁止自动 fallback，并在 child 可见前通过 `ProjectionStore::persist_session_metadata` 写入 canonical/projected
  metadata；专项 Gate B 与 owner 回归均已通过。
- Windows packaged live Provider 与 Explorer Skill 安装目录仍需平台实机证据；当前 macOS Gate B 不能替代该平台
  结论。若其余 current 主链门禁通过，该项作为已知平台剩余风险进入发布说明，不冒充已验证。

## 架构与治理确认

- `current`：Hook discovery/trust/lifecycle、plural Skills catalog/config/extra roots、Plugin search、Thread-scoped
  MCP exact methods、App Server v2 canonical projection、模型目录刷新与持久化 AgentControl exact route 均走唯一产品链。
- `compat`：无新增。
- `deprecated`：无新增。
- `dead / deleted / forbidden-to-restore`：singular `skill/list`、旧 MCP tool/resource wire、无 Thread 的 Settings
  tool call、旧 Electron Skills facade、production mock fallback、AgentControl catalog route 自动 fallback 与 Renderer
  `known_unprojected` Hook 路径。

## 验证记录

- `npm run verify:app-version`：通过，根应用、CLI npm 包、Rust workspace 与 Cargo.lock 均为 `1.123.0`。
- `npm run typecheck`：通过，Renderer 与 Node TypeScript 均无错误。
- `npm run test:contracts`：通过；934 个生成协议类型无漂移，App Server client contract 292 项通过，
  Electron host command 91 项通过，mock priority command 为 0。
- `npm run test:rust:related -- <6 paths>`：通过，App Server 相关测试 `1695/1695`；其中 session operations
  `15/15`、AgentControl restart `12/12`，覆盖 exact catalog route 禁止 fallback 与 canonical child route 持久化。
- `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check` 与 `git diff --check`：通过。
- Provider generation PendingRoute cold-restart Gate B：通过；证据
  `.lime/qc/provider-generation-pending-route-gate-b.json`，child Provider 请求恰好 1 次，mailbox Turn/Item
  各 1 个且终态，GUI 可见 child 结果，mock/invoke/console/page error 均为 0。
- Hook lifecycle Gate B、MCP current fixture 与 Workspace MCP Electron fixture：通过，均经过真实 Electron、preload/IPC、
  App Server JSON-RPC、runtime/read model 与 GUI current bridge。
- `npm run smoke:agent-runtime-current-fixture`：修复后重跑通过；覆盖 history/cache、turn/approval、MCP、Skills、
  Workbench、媒体与文章工作区等真实 Electron fixture 场景，`liveProviderUsed=false`。
- `npm run verify:gui-smoke`：修复后通过；证据
  `.lime/qc/project-gates/standalone-shell-01-20260807150939-91976/shell-01-electron-smoke/summary.json`。
- 前端定向测试 `168/168`、扩展 Rust related 25 crates、`npm run governance:legacy-report` 与
  `npm run governance:scripts`：通过；legacy 边界违规 0，scripts 冻结基线无漂移。
- 未执行：Windows packaged live Provider / Explorer Skill 实机 Gate B；原因是当前环境为 macOS，保留为平台剩余风险。

## Git 发布结果

- 2026-08-07 已获得用户明确确认；授权范围为除 `.gitignore` 本地 `.gstack/` 规则外的完整 release candidate，
  release commit `Release v1.123.0`、annotated tag `v1.123.0`、`origin/main` 与 tag 推送。
- 暂存复核：全部候选已进入 index，Git rename 检测后为 285 个 staged 文件、29,518 行新增、5,313 行删除；
  working tree 仅剩明确排除的 `.gitignore`，`git diff --cached --check` 通过。
- commit、tag 与远端引用结果以本计划所在 release commit、Git tag 和远端 refs 为最终事实源。

## 完成度

- 当前发版完成度：`95%`；候选、全部本地发布门禁与高风险确认已完成，正在执行 Git 发布与远端工作流复核。
