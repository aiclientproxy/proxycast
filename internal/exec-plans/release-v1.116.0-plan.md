# Lime v1.116.0 发布执行计划

状态：quality-fix-ready-for-git-confirmation
日期：2026-07-30
目标版本：`1.116.0`
目标 tag：`v1.116.0`

## 主目标

将 `v1.115.0` 后当前工作树中的 Renderer ConversationProjection、direct TurnTimeline、Pending Interaction、v2 media read、长历史性能、模型 taxonomy、Agent control state，以及对应协议、文档、测试、schema 和生成物作为同一个 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：release commit `5d1feb7e2`、tag `v1.116.0`、`main`/tag 推送和 GitHub Release 已完成；Release workflow 全绿。首轮 Quality run 暴露干净 CI 缺口，本地修复及 CI 同款全量前端、Rust、contracts、typecheck 已通过，准备 follow-up quality fix。
- 下一刀：获得第二次 Git 高风险写操作明确确认后，只暂存本计划声明的 8 个 quality-fix 文件，复核 staged 摘要，创建并推送 follow-up commit；保持 `v1.116.0` tag 固定指向 `5d1feb7e2`，监控新 Quality run 到成功。

## Release Candidate

- 基线：`v1.115.0`。
- 初始工作树：305 个已跟踪改动、37 个未跟踪文件，其中 14 个文件为删除；tracked diff 为 13,324 行新增、9,986 行删除，暂存区为空。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes、本计划和执行计划索引。
- `candidate changes`：当前工作树全部已跟踪和未跟踪产品、协议、文档、测试、schema 与生成物改动；核心包括 canonical ConversationProjection、direct TurnTimeline、Pending Interaction、MCP elicitation、thread-scoped `media/read`、bounded history/preview、protocol drift diagnostics、Agnes canonical capability、Agent state projection，以及云端多模型平台目标架构。
- `excluded changes`：无。未跟踪文件均与上述 current runtime、协议、fixture、回归或版本化架构文档直接关联；未发现本地缓存、构建产物、凭证或个人临时文件。
- 最终工作树：316 个已跟踪改动、41 个未跟踪文件，其中 14 个文件为删除；tracked diff 为 13,587 行新增、10,140 行删除。GUI 与 Electron 门禁生成的 `.lime/qc/**`、`dist/**` 和 `dist-electron/**` 证据/构建产物均未进入候选。
- release commit：`5d1feb7e21aa4f74863233e053b43fde1293264d`，message 为 `Release v1.116.0`；本地/远端 `v1.116.0` 与首轮 `origin/main` 均指向该 commit。
- follow-up quality fix 写集：`tsconfig.json`、`scripts/check-app-server-client-contract.mjs`、`lime-rs/crates/app-server/src/processor/thread.rs`、`lime-rs/crates/app-server/tests/media_task_jsonrpc.rs`、`lime-rs/crates/app-server/tests/model_selection_refresh_jsonrpc.rs`、`src/lib/governance/projectThreadFirstBoundary.test.ts`、`src/components/agent/chat/AgentChatWorkspace.teamRuntimeBoundaryGuard.test.ts` 和本计划。
- follow-up 排除项：并发工作树中的导航、ModelSelector、canonical item reader、turn direct-input、Gate B/smoke 调试等其它文件不属于本轮 quality fix，不暂存、不提交、不覆盖。

## 写集与退出条件

- 发布准备阶段只写版本事实源、双语 release notes、本计划和执行计划索引；其余候选产品文件只读避让。门禁若暴露候选缺陷，必须先声明最小补丁点再修改对应 owner。
- 所有版本事实源同步为 `1.116.0`，release notes 只保留当前版本单页。
- 必跑 `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、`npm run test:rust:changed`、`npm run test:changed -- "v1.115.0"`、`npm run smoke:agent-runtime-current-fixture`、`npm run smoke:agent-session-history-electron-fixture`、`npm run verify:gui-smoke` 与 `git diff --check`；失败必须定位、修复或记录明确阻塞，不能静默降级。
- 重大架构变更已在 `internal/aiprompts/architecture.md` 更新；Renderer single projection 和 thread-scoped media read 的责任开发者确认记录为 `root, 2026-07-29`。
- 获得 Git 写操作确认后，暂存全部 release candidate，复核 staged 摘要，连续执行 `git commit -m "Release v1.116.0"`、`git tag v1.116.0`、`git push origin main`、`git push origin v1.116.0`，并复核本地/远端 commit 与 tag。
- follow-up 不移动、不删除、不重打 `v1.116.0`；只允许普通 commit 推进 `main` 并触发新的 Quality run。

## 验证记录

- `npm run verify:app-version`：通过，根应用、CLI npm 包、Rust workspace、Cargo lock 与 Electron/App Server 均为 `1.116.0`；所有门禁结束后已再次复跑通过。
- `npm run typecheck`：通过，Renderer 与 Node TypeScript 工程均无类型错误；所有门禁结束后已再次复跑通过。
- `npm run test:contracts`：通过；App Server client 286 checks、command contract、Harness、modality、scripts、Electron release workflow 与 docs boundary 全部通过。
- `npm run test:rust:changed`：最终独占完整复跑通过；App Server `1623/1623`，其余受影响 workspace library tests 全部通过。
- `npm run test:changed -- "v1.115.0"`：最终独占完整复跑 `112/112` 批全部通过。
- `npm run smoke:agent-runtime-current-fixture`：通过；覆盖 history/cache hydration、turn/tool 收尾、首页热路径、Coding Workbench、图片与媒体、停止后续聊、approval、Inputbar restore、steer、Plan、Skills、MCP、Expert 与 Article Editor 等真实 Electron fixture，`liveProviderUsed=false`。
- `npm run smoke:agent-session-history-electron-fixture`：通过；覆盖 archive/restart/unarchive、分页同构读取、长列表读取与 resume，证据写入 `.lime/qc/gui-evidence/agent-session-history-electron-fixture/`。
- `npm run verify:gui-smoke`：通过；production renderer、Electron host/preload、App Server sidecar、Claw workspace reload 与设置页就绪，App Server 报告 `protocol=appserver.v0 version=1.116.0`，Gate B 证据写入 `.lime/qc/project-gates/standalone-shell-01-20260730020346-38855/`。
- `npm run governance:legacy-report`：通过；扫描 2370 个文件，零引用候选 0、分类漂移候选 0、边界违规 0。
- `git diff --check`：所有门禁结束后复跑通过。
- 计时抖动说明：首次并行执行时，App Server 1200-command 导入用例为 `30.90s > 30s`，独占精确复跑为 `4.19s`，随后完整 Rust changed-scope 独占复跑通过；`AppSidebar.search` 首用例首次超过 5 秒，独占精确复跑 `8/8` 通过且首用例为 326ms，随后 `112/112` changed-test 正式批次通过。两项均判定为并发资源竞争，没有修改产品代码或放宽断言。
- Tag 与基线复核：release commit `5d1feb7e21aa4f74863233e053b43fde1293264d` 已推送；本地与远端 `v1.116.0` 均固定指向该 commit，follow-up quality fix 不移动或重打 tag。

## 远端发布与 Quality 收口

- GitHub Release workflow `30509459720`：成功；Electron macOS arm64/x64、Windows x64、Windows N-1 Squirrel installed smoke、GitHub assets、Cloudflare R2 updater 与 macOS arm64/x64、Linux x64、Windows x64 CLI 发布全部通过。
- GitHub Release：`https://github.com/limecloud/lime/releases/tag/v1.116.0`，非 draft、非 prerelease，target commit 为 `5d1feb7e21aa4f74863233e053b43fde1293264d`，13 个资产均为 uploaded。
- 首轮 Quality run `30509386095`：失败。Frontend Full 在干净环境无法解析 `@limecloud/app-server-client/browser`；根因是根 `tsconfig.json` 缺少 browser 子路径映射，本地残留 `dist/browser.d.ts` 掩盖问题。已新增源码映射与 contract guard，并在移开 `packages/app-server-client/dist` 后验证 Renderer typecheck 仍通过。
- 首轮 Quality 的 Rust Full：`media_task_jsonrpc` 两个图片命令 fixture 在 `thread/start` 被 `capability_gap` 拒绝；根因是 fixture 把纯 ImageGeneration 模型作为 Agent chat Thread 模型。已拆分独立 Chat 与 Image provider，Thread/Turn 绑定 Chat，图片 application metadata 仍精确选择 Image provider，不放宽生产预检。
- 前端全量续跑：初次在 43/112 批发现两处治理守卫仍锁定旧的一参数 `switchTopic(sessionId)` 字符串；生产行为与单测已要求 detached session 选项。两处守卫改为验证调用与 `allowDetachedSession: true` 事实，精确测试 `15/15`、完整前端 `112/112` 批通过。
- Rust workspace 全量：首次在 `model_selection_refresh_jsonrpc` 发现 generation 基线硬编码为 0，当前 App Server 初始化已将基线推进为 1；fixture 改为读取实际基线并严格断言只递增 1。随后发现 `thread/start` 的 model/modelProvider 参数错误被错误映射为 JSON-RPC `-32600`；processor 边界现将该模型选择解析的 `InvalidRequest` 精确映射为 `INVALID_PARAMS (-32602)`。最终 `thread_v2_jsonrpc` `15/15` 与完整 `npm run test:rust` 通过，App Server 主库 `1623/1623`。
- follow-up 最终复跑：`npm run test:contracts` 通过（App Server client 286 checks）、`npm run typecheck` 通过、`npm run verify:app-version` 通过、`cargo fmt --all --check` 通过、`git diff --check` 通过。完整前端/Rust 结果验证当时工作树；follow-up staged 清单会排除并发写集，新 Quality run 负责验证精确提交。

## 分类

- `current`：ConversationProjection live/read/resume、direct TurnTimeline、PendingInteractionController、typed MCP/DynamicTool、v2 `media/read`、bounded history/preview、Agent control state facts 和 canonical model taxonomy。
- `compat`：无新增 compat owner。
- `deprecated`：`media.read.chunk` / `media.read.completed` transient notification、fileChange outputDelta、thread/compacted 与尚未迁完的 V2-05 notification 保持 bounded 迁出状态，本次不扩展。
- `dead / deleted / forbidden-to-restore`：v0 `agentSession/media/read` schema、canonical Item -> Message 合成、三个流式内容同步 hook、重复 approval/user-input API、旧 MCP elicitation dialog owner、通用 extension fallback 与生产 mock fallback。

## 完成度

- 当前完成度：98%。`v1.116.0` release commit、tag、推送、GitHub Release 和资产发布均完成；首轮 Quality 缺口已修复且本地完整矩阵通过。仅剩 follow-up quality-fix commit/push 与新 Quality run 成功复核。
