# Lime v1.121.0 发布执行计划

状态：completed
日期：2026-08-05
目标版本：`1.121.0`
目标 tag：`v1.121.0`

## 主目标

以 `v1.120.1` 为基线，将当前工作树中的 Plugin v2 catalog/安装/启停、Agent runtime activation、Claw mention、MCP App Right Surface、受控 Browser HTML 宿主、Skills watcher 修复及其协议、GUI、测试和文档作为同一 release candidate，完成版本事实源、双语 release notes、发布门禁、release commit、tag、`main`/tag 推送与远端复核。

## 当前阶段与下一刀

- 当前阶段：版本事实源、双语 release notes、发布门禁、架构图确认及 Git 发布收口均已完成。
- 下一刀：无；后续由 `v1.121.0` 发布后的线上观察与下一版本执行计划承接。

## Release Candidate

- 基线：`v1.120.1`，本地 `main` 与 `origin/main` 均为 `bba332c65`。
- 初始盘点：145 个 tracked diff 文件、48 个未跟踪文件；tracked diff 为 7,838 行新增、537 行删除。
- 门禁期间并行候选补入 canonical store 快照持久化、SQLite busy timeout 与回归测试；发版文件及闭环修复纳入后为 154 个 tracked diff 文件、49 个未跟踪文件，其中 `.gitignore` 仍排除。
- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划及执行计划索引。
- `candidate changes`：除 `.gitignore` 外的当前工作树产品、协议、schema、生成客户端、Rust runtime、Electron host/fixture、GUI、五语资源、测试、bundled Plugin 资产与文档改动。
- `excluded changes`：`.gitignore` 中本地 `.gstack/` 忽略项；该改动已在 `v1.120.1` 发布计划中明确排除，本次继续保留在未提交工作树。
- 并行避让：除 release metadata 与本计划外，其余候选文件只读验证；门禁暴露 blocker 时只做最小可验证修复并回写本计划。

## 写集与退出条件

- 版本事实源与双语 release notes 统一为 `1.121.0`，notes 只保留当前版本单页。
- 必跑 `npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、受影响 Rust 验证、`npm run smoke:plugin-v2-current-electron-fixture`、`npm run verify:gui-smoke` 与 `git diff --check`。
- 候选已修改 `internal/aiprompts/architecture.md` 并补充 Plugin v2 current 图。直接发布 `main` 无 PR 描述，责任开发者必须在 Git 高风险操作确认中同时确认架构图，未确认不得进入 release evidence。
- Git 写操作（stage、commit、tag、push）须在门禁完成后按危险操作格式取得一次明确确认。
- 完成 release commit、`v1.121.0` tag、`main` 与 tag 推送后，复核本地与远端引用；`.gitignore` 应作为唯一未提交排除项保留。

## 已知非本版阻断项

- Windows 安装到卸载证据、360px 级五语最长文案截图矩阵、repo/personal marketplace 的完整用户路径、更新/授权/管理员策略与安装事务崩溃恢复未验收。
- 通用 Plugin Apps/Hooks、Browser intent、文件/结构化结果与完整 Plugin identity 追踪仍由 `plugin-v2-current-plan.md` 后续阶段承接。
- 本版只完成上述已实现的 macOS current 候选发布，不将 Plugin v2 整体路线图标记为完成。

## 验证记录

- `npm run verify:app-version`：通过，根应用、CLI npm 包、Rust workspace/lock 与 Electron/App Server 版本统一为 `1.121.0`。
- `npm run typecheck`：通过，Renderer 与 Node TypeScript 发布硬门禁无错误。
- `npm run test:contracts`：通过；862 个协议类型生成失败 0/漂移 0，App Server client 292 checks，命令、Harness、多模态、脚本、Forge 发布流程与文档边界通过。
- `npm run test:rust:changed -- --changed=origin/main`：版本/lock 触发后自动扩大为 `cargo test --lib --workspace`；首轮暴露 reasoning 指数重复和两个 Codex 导入性能失败，修复持久化二次 merge 及重复 delta 语义后重跑全 workspace 通过。
- Rust 定向回归：`thread-store` 重复 reasoning delta/final 快照与空 completion 保留快照 1/1；App Server canonical reasoning 线性持久化、空 completion 保留物化内容及持久化恢复 3/3；Codex 1,200 命令同步导入与 40 turn 后台导入 2/2，分别约 4.0s 与 9.6s。
- `npm run smoke:plugin-v2-current-electron-fixture`：通过，Gate B/runtime 覆盖 App Center 安装/启停、Claw mention、MCP tool/resource、Right Surface、reload、cold restart、卸载及历史恢复；summary=`.lime/qc/gui-evidence/plugin-v2-current-electron-fixture/plugin-v2-current-electron-fixture-summary.json`。
- `npm run verify:gui-smoke`：通过，真实 Electron/preload/IPC/`app_server_handle_json_lines`/App Server current 链路、版本 `1.121.0`、reload、Workbench 与 Settings 正常；summary=`.lime/qc/project-gates/standalone-shell-01-20260805091101-56033/shell-01-electron-smoke/summary.json`。
- `npm run i18n:check:json`：通过，5 个 locale、13 个 namespace、10,063 个源键，缺失/多余均为 0。
- `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check`：通过。
- `git diff --check`：通过。
- GUI 构建只输出已知非阻断告警：`oem-runtime-config.js` 非 module script、Browserslist 数据过期与 Electron `console-message` API 废弃提示；本次未更新核心依赖。
- 发布后 CI `30997182812` 首轮失败于 macOS `codesign --timestamp` 处理 `locale.pak`，错误为 `The timestamp service is not available.`；已修复非代码资源 timestamp 策略，并将该错误纳入 Forge package 重试分类。重跑 `31000622796` 暴露此前对全部嵌套文件关闭 timestamp 会导致 notarization 拒绝嵌套 Mach-O；现已仅对 `.pak`、`.bin`、`.dat`、`.asar` 资源关闭 timestamp，嵌套代码恢复安全 timestamp。
- 发布后补充 Agent chat terminal session-detail refresh 迁移及 contract guard；Forge 配置、发布工作流、相关守卫与 Agent chat 定向测试均通过。

## 架构确认

- `internal/aiprompts/architecture.md` 已记录 Plugin v2 catalog、activation snapshot、MCP runtime/Right Surface 和历史恢复主链图。
- 责任开发者确认：confirmed；Plugin v2 current 图与本次发布候选主链一致。

## 分类

- `current`：Plugin v2 App Server catalog/install/enabled owner、typed JSON-RPC gateway、activation snapshot、MCP runtime/resource 与 Right Surface current 投影。
- `compat`：无新增；保留的旧 surface 只能委托 current owner。
- `deprecated`：Plugin v1 manifest/parser/registry/worker/UI runtime 继续按 `plugin-v2-current-plan.md` 迁出，本次不恢复调用。
- `dead / deleted / forbidden-to-restore`：无本轮新增删除；生产 mock fallback、Plugin worker 第二 runtime 与 renderer 旧 registry 不得回流 current 主链。

## 完成度

- 当前发版完成度：100%。候选范围、发版文件、标准门禁、架构图确认、release commit、tag、`main`/tag 推送与远端复核均已完成。
