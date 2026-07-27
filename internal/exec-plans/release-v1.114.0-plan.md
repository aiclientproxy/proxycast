# Lime v1.114.0 发布执行计划

状态：release-approved
日期：2026-07-27
目标版本：`1.114.0`
目标 tag：`v1.114.0`

## 主目标

将 `v1.113.0` 后当前工作树中的 Provider transport、模型能力与路由控制、App Server 协议投影、前端配置和回归证据作为同一个 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：`1.114.0` 版本事实源、双语单页 release notes、候选修复与发布门禁均已收敛；本地 `main`、`origin/main` 与 `v1.113.0` 的准备基线为 `31595fc8c`，用户已于 2026-07-28 确认 Git 危险操作。
- 下一刀：连续执行 release commit、`v1.114.0` tag、`main`/tag 推送与远端复核。

## Release Candidate

- 基线：`v1.113.0`。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes、本计划和执行计划索引。
- `candidate changes`：当前工作树全部已跟踪和未跟踪产品、协议、文档、测试、schema 与生成物改动；核心包括 Gemini GenerateContent、Ollama Responses、官方 Responses Hosted Web Search、`model/rerouted` transient 通知、模型能力 provenance、Provider typed `models[]`、catalog refresh 与 Turn selection reconciliation，以及对应 GUI/fixture/JSON-RPC/Rust 回归。
- `excluded changes`：无。若门禁产生新的临时文件或发现并发进程改动，必须在提交前重新分类并记录，不能默认纳入或删除。

## 写集与退出条件

- 发布准备阶段只写版本事实源、双语 release notes、本计划和执行计划索引；现有候选产品文件只读避让。
- 所有版本事实源同步为 `1.114.0`，release notes 只保留当前版本单页。
- `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、`npm run test:rust:changed`、`npm run test:changed -- "v1.113.0"`、`npm run verify:gui-smoke` 和 `git diff --check` 通过；`npm run verify:local` 只允许保留已查明的基线 i18n 例外，不得隐去或放宽规则。
- 重大架构变更以 `internal/aiprompts/architecture.md` 第 15-20 节为事实源；责任开发者确认已记录为 `root, 2026-07-27`。
- 获得 Git 写操作确认后，暂存全部 release candidate，复核 staged 摘要，执行 `git commit -m "Release v1.114.0"`、`git tag v1.114.0`、`git push origin main`、`git push origin v1.114.0`，并复核本地/远端 commit 与 tag。

## 验证记录

- `npm run verify:app-version`：通过，根应用、CLI npm package、Rust workspace、Cargo lock 与 Electron/App Server 显示版本一致为 `1.114.0`。
- `npm run typecheck`：准备阶段和最终复验均通过，覆盖 renderer 与 node TypeScript project。
- `npm run test:contracts`：通过，覆盖 Desktop Host/preload、App Server protocol/client、gateway、catalog、schema 与 fixture 合同。
- `npm run test:rust:changed`：通过，受影响 workspace lib tests 全部通过。额外执行 `cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server --tests` 与 App Server 治理定向回归，均通过。
- `npm run test:changed -- "v1.113.0"`：111/111 批全部通过，最终进程退出码为 0。
- `npm run verify:gui-smoke`：准备阶段和最终复验均通过。Gate B 证据覆盖真实 Electron renderer、preload/IPC、`app_server_handle_json_lines`、App Server `1.114.0`、Claw workbench shell 首次加载/重载和 memory settings；最终 evidence `run_id=standalone-shell-01-20260727160827-63289`，结果 `pass`。
- `git diff --check`：准备阶段和最终复验均通过。
- `npm run verify:local`：未全部通过，唯一阻断为 changed-file i18n 扫描在 `src/components/channels/ImConfigPage.tsx` 报告 148 处中文硬编码。该文件相对 `v1.113.0` 只将 `provider.customModels` 迁移为 `provider.models`；基线与当前中文命中数均为 269，148 处报告均已存在于基线。本次不扩大为整页 i18n 重写，也不放宽扫描规则；其余 17 个变更前端文件扫描为 0 findings。
- 门禁修复：将 App Server route support helper 拆入 `runtime_backend/route_support.rs`，使 `runtime_backend.rs` 从 882 行降至 712 行；补齐 `Interrupted` image generation 失败投影及回归；对齐官方 OpenAI Responses web/image capabilities 与 custom host fail-closed 断言；修正前端 typed `models[]` test fixtures，未增加旧字符串格式兼容。
- 远端状态：待 Git 危险操作确认后补充 release commit、tag 和 origin 复核。

## 分类

- `current`：Gemini GenerateContent、Ollama Responses、官方 Responses Hosted Web Search、模型 reroute transient 通知、能力 provenance、typed Provider 配置、catalog refresh、Turn selection reconciliation 及其单一 App Server/RuntimeCore/GUI 投影。
- `compat`：无新增 compat owner。
- `deprecated`：无新增 deprecated 路径。
- `dead / deleted / forbidden-to-restore`：Ollama Chat/NDJSON 执行链、基于 Provider 或模型名称猜测执行能力、第三方 Hosted Web Search 提权、reroute 持久化重放、silent catalog fallback、`custom_models/customModels` 和第二套 selection store。
