# Lime v1.132.0 发布执行计划

状态：`validated / awaiting-git-confirmation`
日期：2026-08-19
目标版本：`1.132.0`
目标 tag：`v1.132.0`

## 主目标

发布当前 `main` 工作树中的 Browser Workspace / Right Surface current 收敛切片：统一 Electron `WebContentsView` 与 Agent Browser 动态工具的同 Tab owner，完成 CodeCell trace/evidence App Server owner、旧 Browser surface 物理清退、协议与 GUI 重构、版本事实源、双语单页 release notes、质量门禁及 release commit/tag/push。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划及执行计划导航。
- `candidate changes`：当前工作树中的全部 500 个受跟踪改动和 24 个未跟踪文件（共 524 个 status 条目；`git diff --stat` 为 500 files changed, 5,756 insertions, 55,680 deletions），覆盖 BrowserTabHost、App Server/Runtime、Browser Workspace、协议/schema、Rust、Electron、脚本、测试、五语言资源、架构、路线图与治理。
- `excluded changes`：无。用户请求为完整发布，当前工作树改动均归属于本轮 Browser/Right Surface current 收敛候选。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.132.0`，双语 release notes 只保留 v1.132.0，目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 必须执行；按风险执行 contracts、受影响 Rust/前端定向测试、Browser Electron Gate B、Agent current fixture、GUI smoke、治理与本地门禁，未执行或失败项原样记录。
- staged 内容与上述候选范围一致；获得危险操作确认后创建 `Release v1.132.0` commit、`v1.132.0` tag，推送 `main` 和 tag，并复核本地/远端状态。

## 当前验证记录

- 发布前基线为 `v1.131.0`（commit `340c5c1ba`），发布前 `v1.132.0` 本地和远端 tag 均不存在。
- 当前候选已完成本轮完整门禁；本计划只记录实际执行结果，不以历史 v1.131.0 证据替代本轮证据。

## 门禁结果

- `npm run verify:app-version`：通过，根应用、CLI npm 包、Rust workspace 与 Cargo.lock 均为 `1.132.0`。
- `npm run typecheck`：通过；`npm run verify:local` 内再次通过 renderer/node typecheck。
- `npm run test:contracts`：通过（protocol types 无漂移、App Server client 299 checks、命令/Harness/modality contracts、docs boundary 均通过）。
- Browser Workspace / BrowserTabHost / CodeCell trace / 受影响 Rust 与前端定向测试：通过；完整 smart Vitest `116/116` 批次通过，changed-scope Rust workspace 单测全部通过。
- `npm run smoke:browser-runtime-electron-gate-b`、`npm run smoke:agent-runtime-current-fixture`：通过（完整 smart Vitest 含对应 Gate B/current fixture）；`npm run verify:gui-smoke`：通过，真实 Electron smoke evidence summary `result=pass`，App Server protocol/version 初始化为 `appserver.v0/1.132.0`。
- `npm run governance:legacy-report`：通过（扫描 2,046 个实现文件，零引用候选 0、分类漂移 0、边界违规 0）；`npm run governance:scripts`：通过；`npm run verify:local`：通过；`git diff --check`：通过。
- GUI smoke 曾输出一次 renderer/workbench 等待超时诊断，但 runner 最终生成 `result=pass` summary 并正常退出；该诊断保留为 residual evidence，不阻断本地门禁。

本地不执行 Windows Notification Center、真实 macOS/Windows sleep-resume、签名、公证、正式 release asset 和 CI 门禁；这些证据必须由对应平台/CI runner 提供。

## 架构确认

架构影响：重大。候选更新了 Browser Workspace 的 WebContentsView 唯一执行体、connection-owned reverse request、BrowserRoute identity、turn cleanup、CodeCell trace/evidence owner 与旧 Browser 删除边界；确认已记录于 `internal/aiprompts/architecture.md`、`internal/roadmap/browser/README.md` 和 `internal/exec-plans/browser-runtime-right-surface-plan.md`。

责任开发者：root，确认日期：2026-08-19。

## 收尾记录

- `current`：Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI，以及 BrowserTabHost、Browser Workspace、Browser dynamic capability 和 CodeCell trace reducer。
- `compat`：仅保留仓库中明确标记且不承接新 current 创建路径的迁移边界；本候选不新增 compat wrapper。
- `deprecated`：旧 Canvas Browser、外部 Chrome/CDP、BrowserSessionRef、site adapter 和 connector 路径已从主链迁出，等待验证后确认删除分类。
- `dead / deleted`：旧 Browser Runtime crate、旧 v0 Browser Session、旧 Browser 页面/API/fixture 与生产 fallback 随本候选删除。
- 当前完成度：版本元数据、release notes 与全部本地门禁已完成；仅剩危险操作确认后的 commit/tag/push 与远端复核，完成度 95%。
