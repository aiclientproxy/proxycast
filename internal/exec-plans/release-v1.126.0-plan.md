# Lime v1.126.0 发布执行计划

状态：release-candidate-ready
日期：2026-08-12
目标版本：`1.126.0`
目标 tag：`v1.126.0`

## 主目标

发布 Code Mode process-owned runtime、Agent runtime/session loop、provider canonical transport、App Server read model、
Electron 双 sidecar 资源链、协议/GUI 投影以及对应治理文档；完成 release commit、tag、main/tag 推送与远端复核。

## Release Candidate

- `release metadata`：`package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、
  `lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划。
- `candidate changes`：用户确认当前工作树全部纳入，包括已暂存、未暂存和未跟踪的 Rust、TypeScript、脚本、CI、
  测试、架构、治理与执行计划文件。
- `excluded changes`：无。

## 退出条件

- 版本事实源与双语单页 release notes 统一到 `1.126.0`，目标 tag 本地/远端不存在。
- Code Mode production 只走 `ProcessCodeModeSessionProvider -> code-mode-host -> sandbox V8`，无 in-process fallback。
- dev、Electron assets、Forge/Windows 构建成组携带并校验 `app-server` 与 `code-mode-host`。
- `npm run verify:app-version`、`npm run typecheck`、Rust related、contracts 与 GUI smoke 通过。
- Code Mode Electron Gate B 证明真实 Electron/App Server/standalone host PID、custom exec 回采样与 GUI terminal。
- 完成 release commit、`v1.126.0` tag、main/tag 推送和远端复核。

## 验证记录

- `npm run verify:app-version`：通过，版本事实源一致为 `1.126.0`。
- `npm run typecheck`：通过（发布 metadata 更新后复跑）。
- `npm run test:contracts`：通过；App Server client `309` 项检查及 command、harness、modality、scripts、Electron release workflow、docs boundary 子门禁全部通过。
- `npm run test:rust:related -- lime-rs/crates/agent-runtime lime-rs/crates/agent lime-rs/crates/app-server lime-rs/crates/model-provider lime-rs/crates/runtime-core lime-rs/crates/tool-runtime lime-rs/crates/services`：通过；相关 owner 与反向依赖 crate 单测无失败，存在一个既有测试辅助函数 dead-code warning。
- `npm run smoke:agent-runtime-current-fixture`：通过；history/cache、turn terminal、approval、steer、Plan、Skills、MCP、媒体、Workbench 等 current Electron fixture 闭环通过，`liveProviderUsed=false`。
- `npm run verify:gui-smoke`：通过；真实 Electron/App Server `1.126.0` 初始化、工作台 reload 与 memory settings 可见，evidence result 为 `pass`。
- `npm run governance:legacy-report`：通过；扫描 `2120` 个源码文件，分类漂移 `0`、边界违规 `0`。
- standalone Code Mode Cargo check：通过，无 warning。
- Code Mode process Rust tests：`6/6` 通过。
- sidecar/assets/fixture/package/Gate B script tests：`56/56` 通过。
- `npm run electron:build:app-server-assets`：通过；macOS arm64 双 sidecar 为 `0755`，manifest 双 SHA 复算一致。
- `npm run smoke:code-mode-electron-gate-b`：通过；最新 rerun thread `019ff3ca-7f26-71d2-be81-6a16b7895515`，Electron/App Server/
  host PID 为 `44199/44203/44521`，host parent PID 为 `44203`，17 项 assertion 全通过。
- `cargo fmt --all -- --check`、`git diff --check`：通过。
- `npm run verify:local`：通过；fresh 120 个 Vitest smart batches、contracts、13-crate changed-scope Rust、真实 Electron/App Server GUI smoke、lint、typecheck、i18n、scripts/docs/version 门禁均通过；Rust 仅保留既有 App Server test helper `dead_code` warning。

## 已公开 Release 修复轮

- 首次公开提交为 `b48569b83b457eda1a9dac043d5ad4d470fc9e86`；Quality run `31563417376` 与 Release run `31563421104` 失败，GitHub Release 已创建但无 assets。用户明确要求保留 `v1.126.0`，不改发补丁版本。
- Frontend Full：修复 provider settings Promise 未释放前等待 `sendMessage()` 导致的测试超时；`useAgentChat` 主套件 `186/186` 通过。
- Rust Full：直接 workspace 测试前显式配置已校验 sandbox `rusty_v8` artifact，避免回退到不存在的上游 archive。
- Windows：根工具链与目标 workflow 固定 Rust `1.95.0`，复用 `scripts/lib/windows-msvc-linker.ps1` 导出完整 MSVC/UCRT 环境并选择 `rust-lld.exe`；Quality 将双 sidecar 从 `cargo check` 提升为真实 `cargo build` 链接，不使用 `/FORCE:MULTIPLE`。
- macOS package：`app-server` 与 `code-mode-host` 统一执行“manifest SHA 一致，或签名后通过 `codesign --verify --strict`”校验；非 macOS 和无效签名仍 fail closed，资源回归 `17/17` 通过。
- 修复后本地门禁：`npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、Rust related、`cargo fmt --all -- --check`、workflow YAML parse、Prettier、scripts governance 与 `git diff --check` 全通过。
- 修复后 Gate B：`npm run verify:gui-smoke` 通过；run id `standalone-shell-01-20260812112209-14292`，真实 Electron/preload/IPC/App Server `1.126.0`、Workbench reload、Memory settings、21 项 assertion 全通过，mock/legacy hit 均为 `0`。
- 本地无法证明 Windows MSVC 真链接与三平台 Forge 发布；重建 `v1.126.0` 后必须监控新的 Quality/Release run 到终态，并复核 Release assets。
- 已有 GitHub Release 的 workflow 分支会同步刷新 `target_commitish`、标题与 release notes，避免同名 tag 重建后页面元数据仍指向旧提交。
- 第二次公开提交 `979950a989c510734094e84bb63e80d7526ad27e` 触发 Quality run `31601406178` 与 Release run `31601426048`；
  GUI Smoke、Bridge & Contracts、Integrity、lint、typecheck 与 layer budget 已通过，但 Frontend Full 在 Linux runner 的
  `electron/updateHost.test.ts` 被 unsupported-platform 保护提前拦截，Windows Quality/Release 则在真实链接时证明
  `rust-lld` 与 `rusty_v8` 的 `allocator_shim_win_static.obj`/UCRT 符号不兼容；Rust Full 的 workspace test 还缺少
  standalone `code-mode-host` 构建前置条件，导致 `337` 项通过后 3 项 process test 失败。
- Frontend 测试现在显式固定受测 updater 平台为 macOS，并在用例后恢复 runner 平台；定向套件 `13/13` 通过。
- Windows 继续复用 `VsDevCmd.bat` 导出的完整环境，但改用同一 `VCToolsInstallDir` 的 x64 原生 `link.exe`；不使用
  `/FORCE:MULTIPLE`、`/NODEFAULTLIB` 或其它掩盖符号冲突的参数。下一轮 Windows runner 必须重新证明双 sidecar 与 Electron package 真链接。
- Rust Full 在 workspace test 前显式构建 `code-mode-host`，满足 process test 的真实 standalone binary 前置条件。
- 第二轮 CI 终态：Quality `31601406178` 为 failure（Frontend Full、Rust Full、Windows Shell Runtime 三项失败，
  GUI Smoke、Bridge & Contracts、Integrity 成功）；Release `31601426048` 为 failure（macOS arm64/x64 成功，Windows x64 失败）。
- 第二轮修复本地复验：CI 对应 Vitest 批次 `25/120` 为 `16 files / 128 tests` 全通过；在与 CI 相同的已校验
  sandbox `rusty_v8` artifact 环境下，`code-mode-host` 构建成功，3 个 standalone process tests 全通过；
  `npm run typecheck`、`npm run test:contracts`（`309 checks`）、`npm run verify:app-version`、scripts governance、
  release workflow guard、YAML parse、Prettier 与 `git diff --check` 均通过。
- 第三次公开提交 `84dc6723822e5468c51c796393fa9aaba14db0a9` 触发 Quality `31605835668` 与 Release
  `31605860011`。Frontend Full 在批次 `32/120` 发现 harness 直接导入的 `smol-toml` 仅由 `knip` 传递安装；
  GUI Smoke 和 macOS arm64/x64 Release 恢复了缺少 `gn_out/obj/rusty_v8` 的 Cargo 缓存指纹，导致 V8 archive
  未重新物化；Windows 原生 `link.exe` 已正确启用，但 Codex sandbox V8 静态 CRT 与 Lime 默认动态 UCRT 混用，
  触发 `allocator_shim_win_static.obj` 的 `LNK2005`/`LNK1169`。
- 第三轮修复将 `smol-toml` 声明为根开发依赖；所有 sidecar/Forge V8 构建在 artifact 校验后定向执行
  `cargo clean -p v8`，只刷新 V8 build-script 指纹；Windows MSVC 通过 artifact owner 导出的目标级
  `target-feature=+crt-static` 对齐 Codex 已在 Windows Rust 1.95 Cargo smoke 验证的 artifact 合同，不使用
  `/FORCE:MULTIPLE`、`/NODEFAULTLIB` 或其它链接器掩盖参数。
- 第三轮本地复验：pnpm `9.15.9` frozen lockfile 安装通过；Frontend 失败批次 `32/120` 为 `16 files / 113 tests`
  全通过；`rusty_v8`、Electron sidecar 与 DeepSWE adapter 定向回归为 `3 files / 44 tests` 全通过；
  `npm run typecheck`、`npm run verify:app-version`、`npm run test:contracts`（`311 checks`）、scripts governance、
  release workflow guard、三份 workflow YAML parse、Prettier 与 `git diff --check` 全通过。

## 待执行门禁

- 全候选 staged 复核、危险操作确认、commit/tag/push 与远端状态复核。
- Windows 双 sidecar 由 CI 的 Windows runner 执行；本地 macOS Gate B 不冒充 Windows packaged parity。

## 架构确认

架构影响：重大。已更新 `internal/aiprompts/architecture.md` 第 44 节，确认 production process owner、双 sidecar
构建/资源完整性、fail-closed 边界与 Gate B 证据。责任开发者：root，确认日期：2026-08-12。
