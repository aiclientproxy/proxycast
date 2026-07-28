# Lime v1.115.0 发布执行计划

状态：ready-for-git-confirmation
日期：2026-07-29
目标版本：`1.115.0`
目标 tag：`v1.115.0`

## 主目标

将 `v1.114.0` 后当前工作树中的 Provider transport、模型目录与路由、视频媒体任务、Agent 空响应与重试、图片投影、Codex 渲染 v2 规划及其协议、文档、测试和生成物作为同一个 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：版本事实源、双语单页 release notes、发布门禁、Gate B 与候选集终审均已完成；本地 `main`、`origin/main` 与 `v1.114.0` 基线均为 `c7e6dc841`，目标 tag 在本地和远端均不存在。
- 下一刀：请求 Git 写操作确认；确认后暂存完整候选集，连续完成 release commit、tag、`main`/tag 推送和远端复核。

## Release Candidate

- 基线：`v1.114.0`。
- 初始工作树：187 个已跟踪改动、17 个未跟踪文件，其中 5 个文件为删除；暂存区为空。发布准备及并行候选收口后终审为 194 个已跟踪改动、23 个未跟踪文件，其中 5 个文件为删除，共 217 个候选文件；tracked diff 为 10,795 行新增、2,955 行删除，暂存区仍为空。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes 和本计划。
- `candidate changes`：当前工作树全部已跟踪和未跟踪产品、协议、文档、测试、schema 与生成物改动；核心包括 Vertex Gemini、Azure OpenAI Responses、Fal/xAI 视频任务、`model/list/updated`、Provider health/retry、Agent empty-response resampling、Thread model admission/reconciliation、Hosted Image/sidecar 预览，以及 Codex 渲染 v2 覆盖蓝图。
- `excluded changes`：无。未跟踪文件均与上述协议、worker、transport、回归或版本化规划直接关联；未发现本地缓存、构建产物、凭证或个人临时文件。

## 写集与退出条件

- 发布准备阶段写版本事实源、双语 release notes、本计划，以及发布门禁直接暴露的 6 个协议/投影/fixture 文件；其余候选产品文件只读避让。
- 所有版本事实源同步为 `1.115.0`，release notes 只保留当前版本单页。
- 必跑 `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、`npm run test:rust:changed`、`npm run test:changed -- "v1.114.0"`、`npm run verify:gui-smoke` 与 `git diff --check`；失败必须定位、修复或记录明确阻塞，不能静默降级。
- 重大架构变更已在 `internal/aiprompts/architecture.md` 更新，模型目录 admission、Hosted Image、Vertex Gemini 与视频媒体任务的责任开发者确认均记录为 `root, 2026-07-27/28`。
- 获得 Git 写操作确认后，暂存全部 release candidate，复核 staged 摘要，连续执行 `git commit -m "Release v1.115.0"`、`git tag v1.115.0`、`git push origin main`、`git push origin v1.115.0`，并复核本地/远端 commit 与 tag。

## 验证记录

- `npm run verify:app-version`：通过；最终版本一致性为 `1.115.0`。
- `npm run typecheck`：通过；发布硬门禁在投影类型修复后完整复跑通过。
- `npm run test:contracts`：通过；App Server client contract 为 284 checks，command、harness、modality、scripts、Electron release workflow 与 docs boundary 均通过。
- `npm run test:rust:changed`：通过；workspace 版本变化触发 `cargo test --lib --workspace`，除测试自身声明的网络/TCP ignored cases 外全部通过。
- `npm run test:changed -- "v1.114.0"` + `npm run test:resume`：通过；修复失败批次后续跑完成 `112/112`。
- `npx vitest run "src/lib/api/agentRuntime/conversationProjection/reducer.test.ts" "src/lib/api/agentRuntime/appServerCanonicalThreadProjection.test.ts"`：通过，2 files / 10 tests；`test:related` 入口曾因 Vite 将 `electron` 目录当文件读取而报 `EISDIR`，已用明确测试文件覆盖本次最终投影补丁。
- `npm run verify:gui-smoke`：通过；Gate B-F `run_id=standalone-shell-01-20260728170431-14818`，21/21 assertions、33 次 `app_server_handle_json_lines` IPC 命中，无 console/page/invoke/preload 错误，无 legacy command 与 mock fallback 命中。
- `git diff --check`：通过。
- 远端状态：本地 `main` 与 `origin/main` 同步于 `c7e6dc841`；目标 tag 本地和远端均不存在。

## 门禁修复

- `lime-rs/crates/app-server-protocol/src/protocol/v2/tests.rs`：稳定通知清单补齐 `model/list/updated`。
- `internal/refactor/v1/fixtures/item-inventory.v0.1.json`：同步 `imageGeneration` 的 `revisedPrompt` / `savedPath` 投影字段并标记 aligned。
- `src/lib/api/agentRuntime/appServerCanonicalThreadProjection.test.ts`：未知 Item 改为验证 fail-visible `unknown_item` 安全诊断投影。
- `src/lib/api/agentRuntime/appServerCanonicalThreadProjection.ts` 与 `conversationProjection/{adapters,reducer}.ts`：修复 reducer 接入后的序号 fallback、协议对象收窄、event-id 字面量类型和线程 identity 穷尽分支。
- `src/lib/api/agentRuntime/conversationProjection/reducer.test.ts`：补齐 direct v2 Item fixture 的 canonical sequence 与生命周期时间，守住 live/read/replay 等价投影。

## 分类

- `current`：Vertex Gemini、Azure OpenAI Responses、Fal/xAI 视频任务、typed `model/list/updated`、exact-route Provider health/retry、Agent empty-response resampling、Thread model admission/reconciliation、Hosted Image 与 sidecar 预览。
- `compat`：无新增 compat owner。
- `deprecated`：flat EventLog 与旧 projection reader 保持既有 bounded/deprecation 状态，本次不扩展。
- `dead / deleted / forbidden-to-restore`：全局 `modelProvider/capabilities/read`、未绑定 exact credential 的视频执行、普通 OpenAI 视频伪执行、Provider 名称能力猜测、旧测试脚本、未知 Item 静默丢弃与第二套 Renderer timeline store。

## 完成度

- 当前完成度：90%。候选范围、架构确认、版本、release notes、门禁、Gate B 与远端/tag 预检均已完成；仅待 Git 写操作确认后执行 commit、tag、推送与远端复核。
