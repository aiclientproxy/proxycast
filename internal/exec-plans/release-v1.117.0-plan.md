# Lime v1.117.0 发布执行计划

状态：ready-for-git-confirmation
日期：2026-07-31
目标版本：`1.117.0`
目标 tag：`v1.117.0`

## 主目标

将 `v1.116.0` 后当前工作树中的 Host capability、DynamicTool、权限请求、Multi-Agent、配置告警 v2、Lime Hub / Agnes 2.5 模型路由、媒体 transient retirement，以及 v1.116.0 发布后的质量修复作为同一个 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：release candidate、版本事实源、双语 release notes 与发布门禁均已收敛；基线 `main`、`origin/main` 与 `v1.116.0` 均为 `5d1feb7e2`，目标 tag `v1.117.0` 在本地和远端均不存在。
- 下一刀：完成工作树稳定性双采样与 staged candidate 复核，随后请求 Git 高风险写操作确认；确认后连续完成 release commit、tag、`main`/tag 推送和远端复核。

## Release Candidate

- 基线：`v1.116.0`。
- 暂存前最终候选盘点：179 个已跟踪改动、45 个未跟踪文件，其中 5 个文件为删除；tracked diff 为 9,114 行新增、2,622 行删除。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes 与本计划。
- `candidate changes`：当前工作树全部已跟踪和未跟踪产品、协议、文档、测试、schema 与生成物改动；包括 v1.116.0 follow-up quality fixes、Host capability、DynamicTool、request permissions、AgentControl、配置告警 v2、模型 capability provenance、媒体 transient retirement 和 Skill 安装目录打开语义。
- `excluded changes`：无。未跟踪文件均已映射到 current runtime 源码、typed schema、测试或版本化执行/evidence 文档；未发现本地缓存、构建产物、凭证、个人临时文件或异常大文件。
- 工作树准备阶段曾存在并发写入迹象；发布准备只写本计划、版本事实源与双语 release notes，其余候选文件只读避让。Git 确认前以两次间隔指纹采样证明候选集稳定且没有漏发。

## 写集与退出条件

- 所有版本事实源同步为 `1.117.0`，release notes 只保留当前版本单页。
- 必跑 `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、`npm run test:rust:changed`、`npm run verify:gui-smoke` 与 `git diff --check`。
- 协议、Bridge、runtime、GUI 与重大架构均有变化；门禁失败必须定位并修复，或记录明确阻塞，不能静默降级。
- 重大架构变更已同步 `internal/aiprompts/architecture.md`；Host capability、配置告警 v2、媒体 transient retirement 与模型 capability provenance 均已有责任开发者确认记录。
- 获得 Git 写操作明确确认后，暂存完整 release candidate，复核 staged 清单与摘要，连续执行 release commit、创建 `v1.117.0` tag、推送 `main` 和 tag，并复核本地与远端引用。

## 验证记录

- `npm run verify:app-version`：版本更新后与最终复跑均通过，根应用、CLI npm 包、Rust workspace 与锁文件一致为 `1.117.0`。
- `npm run typecheck`：初次执行与最终复跑均通过，覆盖 renderer 与 Node TypeScript 配置。
- `npm run test:contracts`：通过；generated protocol 无漂移，App Server client 共 289 项检查通过。
- `npm run test:rust:changed`：通过；workspace 版本变更使范围扩大为 `cargo test --lib --workspace`，所有已执行 Rust 测试零失败。
- `npm run test:changed -- "v1.116.0"`：通过；112/112 个 changed 批次全部完成，退出码 0。
- `npm run verify:gui-smoke`：通过；App Server 报告 `version=1.117.0`，真实 Electron Gate B 摘要位于 `.lime/qc/project-gates/standalone-shell-01-20260730191439-55464/shell-01-electron-smoke/summary.json`。
- `npm run smoke:agent-runtime-current-fixture`：通过；覆盖 history/cache hydration、turn/tool terminal、approval、steer、Plan、Skills、MCP、media、Coding Workbench 与 Article Editor 等真实 Electron current fixture，`liveProviderUsed=false`。
- `git diff --check`：版本更新前、版本更新后与最终复跑均通过。
- 工作树稳定性：2026-07-31 03:38:29 与 03:38:52 两次采样的 status、tracked diff、untracked content 三组 SHA-256 指纹完全一致，候选数量均为 224/179/45。
- staged candidate：完整候选已暂存；Git 将 3 组 schema 文件识别为 rename，摘要为 221 个文件、14,617 行新增、2,493 行删除，`git diff --cached --check` 通过，无排除项。
- 平台限制：Windows packaged Gate B 尚未执行；本轮 macOS Electron fixture 不能替代 Windows packaged 证据，该项保留为发布后平台残余风险。

## 分类

- `current`：Electron Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item -> GUI 产品链，Host capability、DynamicTool、AgentControl、typed v2 config warning、validated media read、canonical model capability provenance。
- `compat`：无新增。
- `deprecated`：Refactor v2 尚未迁完的 V2-05 notification、broader host capability 与 recovery surface，只允许继续迁出。
- `dead / deleted / forbidden-to-restore`：v0 配置告警与 schema、媒体 transient notification 消费链、Renderer live-drain helper、旧 protocol facade、Renderer 伪造 capability/binding、生产 mock fallback。

## 完成度

- 当前完成度：92%。候选范围、发布元数据、发布门禁、工作树稳定性和 staged 复核已完成；仅剩 Git 明确确认、commit、tag、push 和远端复核。
