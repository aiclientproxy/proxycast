# Lime v1.139.0 发布执行计划

状态：`completed`
日期：2026-09-03
目标版本：`1.139.0`
目标 tag：`v1.139.0`

## 主目标

发布 `v1.138.0` 之后当前工作树中的 Feature Map、文档事实源收敛、SceneApp / `src-tauri` 全面退役、Plugin/App Server/RuntimeCore owner 替换、跨平台 Desktop Host 证据、model-provider 路由与 Curated Task / Memory 改动，完成质量门禁、release commit、tag、main 推送和远端复核。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 `RELEASE_NOTES` 与本计划。
- `candidate changes`：当前工作树中的全部已跟踪产品、文档、测试、schema、workflow 与脚本改动，以及新增 Feature Map、model-provider reasoning effort、Desktop Host Gate B 脚本和 Curated Task 结果基线源码。
- `excluded changes`：`.lime/**`、`.gstack/**`、`dist/**`、`dist-electron/**` 与本地生成缓存；根目录 4 个压缩 bundle 构建碎片已物理删除，不进入候选。

## 架构确认

本轮重大架构变更已同步到 `internal/aiprompts/architecture.md`：应用目录归 Plugin，业务主链保持 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`；SceneApp、`src-tauri` 与 Tauri capability 不保留第二套 owner。责任开发者确认：root / Codex，2026-09-03。

## 退出条件

- 四个版本事实源统一为 `1.139.0`，双语 Release Notes 只保留本版本，本地和远端 tag 在创建前不存在。
- 通过版本、TypeScript、i18n、contracts、文档、治理、Rust related、GUI smoke、前端续跑与 `git diff --check` 门禁。
- staged 内容覆盖全部 release candidate；完成 `Release v1.139.0` commit、`v1.139.0` tag，非强制推送到 `origin/main` 与 tag，并复核 GitHub workflow / Release 状态。

## 验证记录

- `npm run verify:app-version`：通过；根应用、CLI npm 包、Rust workspace 与 Cargo lock 均为 `1.139.0`。
- `npm run typecheck`、`npm run i18n:check`：通过；五语言资源 missing / extra 均为 0。
- `npm run docs:boundary`、`npm run harness:doc-freshness`、全仓 Markdown 本地链接扫描：通过；SceneApp / `src-tauri` 实体路径为 0。
- `npm run governance:scripts`、`npm run governance:legacy-report`：通过；零引用候选、分类漂移、边界违规均为 0。
- `npm run test:contracts`：通过；App Server client 共 299 checks，command、modality、release workflow、cleanup 与文档边界合同均通过。
- `cargo fmt --all -- --check` 与 `npm run test:rust:related -- <changed Rust paths>`：通过；19 个相关 crate 全绿。
- Resource Manager / i18n / window 定向测试：9/9 通过。
- `npm run test:resume`：通过；`.lime/test/vitest-smart-last-run.json` 记录 Frontend Full `118/118` 批完成，最终状态 `passed`。
- `npm run verify:gui-smoke`：通过；run id `standalone-shell-01-20260902171042-60925`，真实 Electron/App Server `protocol=appserver.v0 version=1.139.0`，结果 `pass`。
- `git diff --check`：通过。
- 发布前远端复核：本地 `HEAD`、`origin/main` 与 merge base 均为 `8413925fe`，双向差异 `0/0`；本地和远端均无 `v1.139.0` tag。

## 发布结果

- release commit `a00f31fccee6f5e0136673ffac7911c80d2bc3fe` 已推送，`v1.139.0` tag 固定指向该提交；发布后的修复提交为 `81f89d567`（Docs workflow / lint guard）、`b87d27438`（Rust commands 清理守卫）与 `277fa91ec`（低高度窗口首页输入框布局）。
- 首次 Release run `33663513570` 在 Windows Squirrel 安装态 GUI 的 `layoutGeometryStable` 失败；macOS 两架构与 Windows 安装、N-1 升级和 App Server `1.139.0` 初始化均已通过。根因修复后，本地 `npm run typecheck`、定向 ESLint、4 个测试文件共 84 tests 及 `npm run verify:gui-smoke` 全绿；GUI smoke run id 为 `standalone-shell-01-20260902183119-37065`。
- Quality run `33667770446` 在 `main@277fa91ec` 全绿：Frontend Full、Rust Full、Integrity、GUI Smoke 与 Quality results 均成功。
- 修复后 Release run `33667794033` 在 `main@277fa91ec` 全绿：Windows x64、macOS x64/arm64 打包与资源校验、macOS packaged native-host Gate B、Windows Squirrel 安装/N-1 升级、Code Mode Gate B、native-host Gate B、四个平台 CLI、GitHub Release 上传及 Cloudflare R2 updater 发布全部成功。
- GitHub Release `Lime v1.139.0` 已正式发布，共 13 个 uploaded assets：Windows Setup/full nupkg/RELEASES，macOS x64/arm64 DMG、zip 与 updater JSON，以及 Windows、macOS x64/arm64、Linux x64 CLI 压缩包。
- Pages run `33666030217` 在 `81f89d567` 全绿，build 与 deploy 均成功。
- 最终远端身份：`origin/main=277fa91ec8c2b3fbcd80ff5c7ca7c5f64605587a`，`refs/tags/v1.139.0=a00f31fccee6f5e0136673ffac7911c80d2bc3fe`。标签未移动或重打；修复后的正式发布资产由 `main@277fa91ec` 构建。

## 收尾分类

- `current`：Feature Map、Plugin、Electron Desktop Host、App Server JSON-RPC、RuntimeCore / Agent Runtime、model-provider、Curated Task / Memory。
- `compat`：无新增。
- `deprecated`：无新增。
- `dead / deleted`：SceneApp、`src-tauri`、Tauri capability/schema、测试专用 WebviewWindow 生产入口及被替代历史文档。

当前完成度：`100%`；清理、替换、版本元数据、本地与远端质量门禁、release commit、tag、main/tag 推送、GitHub Release、跨平台资产及 Pages 部署均已完成。
