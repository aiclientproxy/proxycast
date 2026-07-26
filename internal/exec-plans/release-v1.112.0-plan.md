# Lime v1.112.0 发布执行计划

状态：ready-for-recovery-git-confirmation
日期：2026-07-26
目标版本：`1.112.0`
目标 tag：`v1.112.0`

## 主目标

将 `v1.111.0` 之后的 current App Server、Agent Runtime、Provider、GUI、协议、测试和文档改动整理为单一 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：release commit、`v1.112.0` tag、`main`/tag 推送与 GitHub Release 已完成；远端 macOS x64 packaging 因 Apple timestamp 响应缺失失败，正在修复可重试分类。
- 下一刀：验证 workflow recovery guard 后，将修复提交推送至 `main`，以 `workflow_dispatch` 从 `v1.112.0` source ref 重跑发布 workflow，不重写已发布 tag。

## Release Candidate

- 基线：`v1.111.0`；未完成且未打 tag 的 `v1.111.1` metadata 直接由本版本替换，不形成兼容发布轨道。
- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、双语 release notes 与本计划。
- `candidate changes`：当前工作树中 `HEAD` 之后的全部 tracked 与 untracked 产品、协议、schema、generated client、测试、脚本和文档改动。
- 最终盘点：共 `507` 个候选文件，其中 tracked change `489` 个、untracked `18` 个；按顶层 owner 分布为 `lime-rs/ 280`、`src/ 135`、`scripts/ 49`、`internal/ 24`、`packages/ 13`、根目录 `4`、`electron/ 2`。现有 staged `427` 个、unstaged `130` 个，二者有重叠；确认后统一 `git add -A` 收敛为单一 staged candidate。
- `excluded changes`：无。提交前必须再次核对工作树，若发现临时文件或并发进程新增改动则停止并重新判定范围。

## 写集与协作边界

- 本轮写入版本事实源、双语 release notes 与本计划；门禁暴露的历史 Turn identity 缺陷只修改 Agent stream lifecycle current owner 及其定向测试，Right Surface 源码守卫只改为等价的跨行匹配。
- 现有 400 余个产品和文档改动只读避让；验证结果代表当前完整工作树，不声明这些改动的作者归属。
- 不覆盖、不回滚现有 staged 状态；Git 写操作确认后统一以 `git add -A` 收敛完整 release candidate，并在 commit 前复核 staged 文件与统计。

## 退出条件

- 所有版本事实源同步为 `1.112.0`，release notes 只保留当前版本单页。
- `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、`npm run governance:electron-release-workflow`、`npm run governance:scripts` 与 `git diff --check` 通过。
- 正式 GUI 发布证据执行 `npm run verify:gui-smoke`；若受环境限制无法执行，必须记录具体原因。
- 协议、App Server、Agent Runtime 与 Rust 大写集按风险追加 related/integration 或 current fixture 验证，避免以裸全量命令制造不可续跑噪音。
- 获得 Git 写操作确认后，连续执行暂存、`Release v1.112.0` commit、`v1.112.0` tag、`main`/tag 推送，并复核本地与远端状态。

## 验证记录

- `npm run verify:app-version`：通过；根应用、CLI npm package、Rust workspace 与 lockfile 版本事实均为 `1.112.0`。
- `npm run typecheck`：通过；Renderer 与 Node TypeScript 检查通过。`npm run electron:build` 与最终 GUI smoke 内 Electron host typecheck 同样通过。
- `npm run test:contracts`：通过；App Server client contract 284 项、command/Harness/modality/scripts/release workflow/docs boundary 均通过；App Server client 单测 `93/93`、build 与 `npm run check:protocol-types` 通过。
- 前端候选回归：`npm run test:changed` 通过，smart runner 完成 `112/112` 批；历史 Turn identity 定向 Vitest `7/7`，Claw fixture 源码守卫 `78/78`。
- Rust 候选回归：`npm run test:rust:changed` 通过；workspace manifest 变更使该入口扩大执行 `cargo test --manifest-path lime-rs/Cargo.toml --lib --workspace`，全部 workspace library tests 通过。`cargo fmt --check` 同样通过。
- `npm run smoke:agent-runtime-current-fixture`：最终候选复跑通过；覆盖 history/cache、stream completion、Electron/App Server guards `90/90`，以及首页、Coding Workbench、图片、approval、active steer、Plan、Skills、MCP、media 与 Article Editor 等真实 Electron Gate B 场景。
- Coding Workbench 历史折叠定向 Gate B：通过；`historicalTimelinePreviewCount=1`、`toolCallRowCount=0`、`operationalTimelineDetailsCount=0`、无 console/page errors，证据见 `.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/code-artifact-workbench-gui-coding-input-debug-summary.json`。
- `npm run verify:gui-smoke`：最终候选复跑通过；Renderer production build、Electron host/preload、App Server sidecar `appserver.v0` `1.112.0`、Workbench shell、reload 与 memory settings 均取得真实 Electron evidence。
- `npm run governance:scripts`、`npm run governance:electron-release-workflow` 与 `git diff --check`：通过。
- Git 写操作确认前最终复核：`npm run typecheck`、`npm run verify:app-version`、`git diff --check` 与 `git diff --cached --check` 再次通过；根应用版本仍为 `1.112.0`。
- `npm run test:related`：未通过；smart runner 将目录路径 `electron` 当文件读取，报 `EISDIR`。定向 Vitest、changed tests、contracts、current fixture 与 GUI smoke 已独立通过；该 runner 基础设施问题不写成产品门禁通过。

## 发布后恢复

- `Release v1.112.0` 已提交为 `00de4eb436573707220d99b5f3bcfe890b85562c`；本地 `main`、`origin/main`、本地 tag 与远端 `v1.112.0` 指向同一提交，GitHub Release 已创建。
- 首次 Release workflow `30192808153` 仅 `Build Electron macOS-x64` 失败。失败命令为对 Electron Framework `locale.pak` 的 `codesign --timestamp`，错误为 `A timestamp was expected but was not found`；macOS arm64、Windows x64 和 Windows Squirrel installed-candidate smoke 均成功。
- 根因分类：Apple timestamp 服务瞬态失败，而非应用代码、签名身份或 notarization 配置失效。修复将该稳定错误文本纳入既有 macOS package 三次重试规则，并由 release workflow guard 回归守护。
- 恢复写集：`.github/workflows/release.yml`、`scripts/electron/release-workflow-guard.mjs`、其测试与本计划。当前并发的 Rust/文档改动明确排除，不进入 workflow recovery commit。
- 恢复验证：定向 release workflow guard Vitest `27/27`、`npm run test:contracts`、`npm run governance:electron-release-workflow`、`npm run typecheck`、`npm run verify:app-version` 与 `git diff --check` 通过。

## 架构确认与治理分类

- 架构影响：重大。候选涉及 App Server v2 Thread search/metadata/control、background terminal、elicitation/Guardian continuation、Model catalog、RuntimeCore、Provider 与 canonical read model 边界。
- 架构事实源：`internal/aiprompts/architecture.md` 已同步记录上述唯一读取/控制链、typed lowering、fail-closed 规则和禁止恢复的 parallel owner。
- 责任人：root（release owner，v1.112.0）；日期：2026-07-26；确认状态：confirmed。
- 确认证据：contracts、Rust changed tests、Agent current fixture 与真实 Electron Gate B 证明产品链保持 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`，未新增第二套业务后端或生产 mock fallback。
- `current`：Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI，以及 v2 Thread/Model contracts、Provider capability owner 和 canonical read model。
- `compat`：不新增 compat owner；历史边界只能委托 current 实现。
- `deprecated`：已被 v2 contract 替代的旧会话/模型入口只允许迁出。
- `dead / deleted / forbidden-to-restore`：已删除 v0 AgentSession compact/update、旧 ModelList 与重复 Renderer workspace writeback 不得恢复。

平台限制：macOS arm64 真实 Electron Gate B 已通过；未执行 Windows 真机与正式 packaged artifact 安装验证，不将其写成通过。

当前发布恢复完成度：`95%`；原 release commit、tag、推送与 GitHub Release 已完成，剩余为危险操作确认后的 workflow recovery commit/push、从 `v1.112.0` source ref 重新 dispatch，并监控远端资产发布结果。
