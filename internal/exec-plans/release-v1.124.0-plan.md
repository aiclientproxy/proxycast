# Lime v1.124.0 发布执行计划

状态：ready_for_release_confirmation
日期：2026-08-08
目标版本：`1.124.0`
目标 tag：`v1.124.0`

## 主目标

将 Lime 的公开定位更新为具备代码、文件、终端、工具、Skills、MCP、多模态和多 Agent 协作能力的全栈 AI Agent，类似 Claude Code、WorkBuddy、Codex 的 agent 工作方式，同时保留 README 现有图片；完成版本事实源、双语发布说明、发布门禁与 v1.124.0 发版收口。

## 写集与候选范围

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本执行计划。
- `candidate changes`：根 `README.md`、`README.en.md`，以及当前工作树已有的产品、协议、运行时、GUI、测试、schema、脚本和文档改动。
- `excluded changes`：在 Git 写操作前若发现无法归类的本地实验或环境文件，必须单独列出并取得确认；不得覆盖或回滚其他开发者改动。

## 退出条件

- 中文和英文 README 均完成全栈 AI Agent 定位迁移，现有图片引用全部保留。
- 版本事实源统一为 `1.124.0`，双语 release notes 只保留当前版本单页。
- 通过 `npm run verify:app-version`、`npm run typecheck`、`git diff --check`；按当前候选风险执行协议、GUI 和相关 smoke 门禁。
- Git 写操作前完成 staged 范围复核，并按危险操作格式取得明确确认后再 commit、tag、push。

## 当前阶段与下一刀

- 当前阶段：README、版本事实源、release notes 和默认发版门禁已完成。
- 下一刀：汇总 staged 候选范围并请求 Git 写操作确认；确认后连续完成 commit、tag、main/tag 推送和远端复核。

## 验证记录

- `npm run verify:app-version`：通过，根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.124.0`。
- `npm run typecheck`：通过。
- `git diff --check`：通过，README、release notes 与版本事实源无空白错误。
- `npm run test:contracts`：通过，协议生成无漂移，App Server 301 项、命令/脚本/多模态/电子发布/文档边界通过。
- `npm run verify:gui-smoke`：通过，真实 Electron/preload/IPC/App Server 链路与版本 `1.124.0` 通过；summary=`.lime/qc/project-gates/standalone-shell-01-20260808112235-17473/shell-01-electron-smoke/summary.json`。
- 已知非阻断告警：Vite 的 `oem-runtime-config.js` module 提示、Browserslist 数据过期、Electron `console-message` API 弃用提示，以及 Rust 未使用代码 warning。
