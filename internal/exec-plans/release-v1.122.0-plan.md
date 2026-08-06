# Lime v1.122.0 发布执行计划

状态：publishing
日期：2026-08-06
目标版本：`1.122.0`
目标 tag：`v1.122.0`

## 主目标

以 `v1.121.0` 为基线，将当前工作树中的 Thread sections、App Server v2 协议与 canonical store、Agent chat/sidebar 投影、测试、schema、生成客户端和架构文档整理为单一 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：候选盘点、发版 metadata、发布门禁与危险操作确认已完成，正在执行 Git 发布收口。
- 下一刀：复核 staged candidate 后连续完成 release commit、`v1.122.0` tag、`main`/tag 推送和远端复核。

## Release Candidate

- 基线：`v1.121.0`；目标 tag 当前不存在。
- 初始盘点：138 个 tracked diff 文件、19 个未跟踪文件；tracked diff 为 4,966 行新增、1,077 行删除。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes、本计划。
- `candidate changes`：除 `.gitignore` 外的当前工作树全部产品、协议、schema、生成客户端、Rust runtime、GUI、五语资源、测试和文档改动。
- `excluded changes`：`.gitignore` 新增的本地 `.gstack/` 忽略项；该本地环境改动已在上一版发布计划中明确排除，继续保留在未提交工作树。候选冻结并取得确认后出现的 Model Selector 并发改动也不进入本版：`InputbarModelExtra.tsx`、`src/components/input-kit/ModelSelector*.tsx` 与五语 `common.json`，共 8 个路径，原样保留在工作树。
- 最终盘点：152 个 tracked diff 文件、38 个未跟踪文件；tracked diff 为 8,298 行新增、1,324 行删除。新增 schema 由 `write_schema_fixtures` 生成，包含 Plugin search 与 Thread sections current schema。

## 写集与退出条件

- 所有版本事实源统一为 `1.122.0`，release notes 只保留当前版本单页。
- 必跑 `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、受影响 Rust 验证、`npm run verify:gui-smoke` 与 `git diff --check`。
- 候选已修改 `internal/aiprompts/architecture.md`；release owner 需在 Git 高风险操作确认中确认架构图与 current 产品链一致。
- Git 写操作（stage、commit、tag、push）须在门禁完成后按危险操作格式取得一次明确确认。
- 完成 release commit、`v1.122.0` tag、`main` 与 tag 推送后，复核本地与远端引用，并保留 `.gitignore` 作为唯一排除项。

## 验证记录

- `npm run verify:app-version`：通过，根应用、CLI npm 包、Rust workspace 与 Cargo.lock 均为 `1.122.0`。
- `npm run typecheck`：通过；Renderer 与 Node TypeScript 检查通过。
- `npm --prefix "packages/app-server-client" run build`：通过；修复 schema fixture 漂移后 package build 通过。
- `npm run test:contracts`：通过；protocol types 889 个无漂移，App Server client 292 checks，命令/脚本/Forge release workflow、harness、modality、docs boundary 全绿。
- `npm run test:rust:related -- "lime-rs/crates/app-server" "lime-rs/crates/app-server-protocol" "lime-rs/crates/thread-store" "lime-rs/crates/app-server-client"`：通过；相关 workspace 单元测试全部通过，其中 `agent-runtime` 192、`app-server` 1690、`app-server-client` 34、`app-server-protocol` 99，未发现失败。
- `npm run verify:gui-smoke`：通过；真实 Electron/preload/IPC/App Server/GUI Gate B 通过，App Server protocol `appserver.v0` version `1.122.0`，summary=`.lime/qc/project-gates/standalone-shell-01-20260806121506-67837/shell-01-electron-smoke/summary.json`。
- GUI 构建记录已知非阻断告警：`oem-runtime-config.js` module 属性、Browserslist 数据过期、Electron `console-message` API 弃用，以及本机 `install_name_tool` 重复 rpath；sidecar 已准备且 smoke result 为 pass。
- `git diff --check`：通过；metadata、schema generated output 和当前候选无空白错误。

## 架构与治理确认

- `current`：Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI，以及 Thread sections 的 App Server/thread-store owner。
- `compat`：无新增；旧入口只能委托 current owner。
- `deprecated`：未新增。
- `dead / deleted / forbidden-to-restore`：旧 pinned 双轨入口不回流；`.gitignore` 中 `.gstack/` 为本地环境排除项，不进入发布提交。

## Git 发布确认

- 待确认操作：将除 `.gitignore` 外的全部 189 个 candidate 路径暂存，创建 `Release v1.122.0` commit，创建本地 `v1.122.0` tag，推送 `main` 与 tag 到 `origin`，并复核远端 tag。
- tag 当前不存在，不覆盖既有 tag。
- 架构确认：本候选继续使用唯一产品链 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`；Thread sections 归 App Server/thread-store current owner，无新增 compat/deprecated owner。release owner 已在危险操作确认中确认。
- 暂存复核：189 个 candidate 路径已暂存；`.gitignore` 与上述 8 个候选冻结后并发改动保持未暂存。

## 完成度

- 当前发版完成度：`90%`；候选范围、metadata、验证和 GUI Gate B 已完成，剩余 Git stage、commit、tag、push 与远端复核。
