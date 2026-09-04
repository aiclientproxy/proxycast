# Lime v1.140.0 发布执行计划

状态：`release_recovery_in_progress`
日期：2026-09-04
目标版本：`1.140.0`
目标 tag：`v1.140.0`

## 主目标

发布 `v1.139.0` 之后当前工作树中的 CLI/TUI 多 Surface、视频任务工具、Desktop Host 诊断、App Server client 会话传输、Electron 跨平台证据与治理收敛改动，完成版本事实源、双语单页 release notes、质量门禁、release commit、tag、main 推送和远端复核。

## Release Candidate

- `release metadata`：`package.json`、`packages/cli/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划与 `internal/exec-plans/README.md`。
- `candidate changes`：当前工作树中全部已跟踪产品、文档、测试、schema、workflow、脚本改动，以及新增 CLI/TUI、视频任务、Gate B/治理脚本和执行计划文件。
- `excluded changes`：`undefined/data/*` 下 7 个本地 SQLite/WAL 运行产物；它们是本机运行缓存，不属于产品或发布事实源。

## 架构确认

本轮新增 CLI/TUI Product Surface，但业务主链仍是 `Product Surface -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection`；Electron Desktop Host 与 CLI/TUI Host 共享协议、runtime、持久化和工具 owner，不建立第二套业务后端。该边界已同步到 `AGENTS.md`、`internal/aiprompts/architecture.md`、命令与质量文档。责任开发者：root / Codex，2026-09-04。

## 退出条件

- 根应用、CLI npm 包、Rust workspace 与 Cargo.lock 统一为 `1.140.0`；双语 Release Notes 只保留 v1.140.0；本地和远端目标 tag 在写操作前不存在。
- `npm run verify:app-version`、`npm run typecheck` 通过；按风险补充 contracts、Rust related、CLI/TUI fixture、GUI smoke 与 `git diff --check`。
- staged 内容覆盖全部 candidate changes 与 release metadata，明确排除 `undefined/data/*`；完成 `Release v1.140.0` commit、`v1.140.0` tag，并将发布提交推送到 `origin/main`、标签推送到远端后复核。

## 验证记录

- `npm run verify:app-version`：通过；根应用、CLI npm 包、Rust workspace 与 Cargo.lock 均为 `1.140.0`。
- `npm run typecheck`：通过；renderer 与 node TypeScript 均无错误。
- `npm run test:contracts`：通过；协议生成无漂移、App Server client 299 checks、command/harness/modality/scripts/release/CLI/TUI/docs boundary 全绿。
- `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check`：通过。
- `npm run test:rust:related -- ...`：核心相关 crate 全部通过；随后 `cargo test --manifest-path "lime-rs/Cargo.toml" -p tui --lib` 通过 55/55。
- CLI/TUI 收口复验：`cargo test --manifest-path "lime-rs/Cargo.toml" -p tui` 通过 57/57，`cargo test --manifest-path "lime-rs/Cargo.toml" -p cli` 通过 15/15，`cargo clippy --manifest-path "lime-rs/Cargo.toml" -p tui -p cli --no-deps -- -D warnings` 通过；覆盖 JSONL envelope、shell completion、turn 终态摘要与 Windows external-editor shim。
- `npm run smoke:cli-gate-b`：通过；真实 CLI/App Server stdio 链路输出 `turn.completed`，状态 `ready`。
- `npm run smoke:tui-gate-b`：通过；真实 PTY/alternate screen 链路完成，终端状态 `restored`。
- `npm run verify:gui-smoke`：通过；真实 Electron/App Server `version=1.140.0`，GUI smoke run `standalone-shell-01-20260903234112-2478`。
- 受影响前端显式 Vitest：11 个文件、159 项断言通过。`npm run test:related` 的 smart runner 因将 `electron` 目录误作文件触发 `EISDIR`，未产生测试断言失败，已改用显式文件验证。
- `npm run i18n:check`：通过；5 locales、34,716/34,716 keys，missing/extra 均为 0。
- `npm run governance:legacy-report`：通过；零引用候选、分类漂移、边界违规均为 0。
- `git diff --check`：通过；目标 tag 在本地和远端均不存在。
- Release commit `1fe2ad260bb080a94a8dbda9e0d954f6747091bf`、本地/远端 `main` 与 `v1.140.0` tag 已完成并复核一致。
- Quality run `33820999125`：Bridge/Contracts、Frontend Full、GUI Smoke、Rust Full 与 Integrity 全绿；`Windows Shell Runtime` 的 timeout contract 使用 `Start-Sleep -Seconds 10`，在进入 20ms 进程超时路径前被生产长休眠策略正确拒绝，因此测试失败。远端 run 随后被移除，无法继续下载日志。
- 首次 Release run `33821027658`：macOS x64/arm64 打包、签名、公证、资源校验与 packaged native-host Gate B 通过；Windows 安装、N-1 更新、候选 `1.140.0` App Server 初始化和 SHELL-01 GUI smoke 通过，随后在 `Update.exe --uninstall` 清理已不存在的注册表子键时退出 1，导致 Windows 后续 Gate B 与资产发布跳过。GitHub Release 已创建为 draft，尚无资产。
- 失败复盘同时发现生命周期契约矛盾：Squirrel smoke 在 workflow 后续 CodeMode/native-host Gate B 前卸载候选，而 packaged evidence validator 又同时要求卸载成功和已安装 exe 存在。恢复补丁将卸载移动到全部 installed packaged Gate B 与身份校验之后；卸载仅容忍同时包含缺失子键文本、`RegistryKey.DeleteSubKeyTree` 与 `Squirrel.Update.Program.<Uninstall>` 的明确幂等错误，仍要求 Update.exe、候选目录、主程序与快捷方式全部消失，其他错误继续 fail closed。
- 恢复补丁本地验证：Windows Squirrel/packaged evidence/release workflow guard 定向 Vitest 3 文件 `66/66`；`npm run typecheck`、`npm run test:contracts`、`npm run verify:app-version`、`npm run governance:scripts`、release workflow guard、Prettier check 与 `git diff --check` 全部通过。
- Windows timeout 恢复补丁将测试负载改为不触发 shell 策略但仍可由 timeout 终止的 `.NET Thread.Sleep`，并给 Unix-only imports/helper 补齐 `cfg(unix)`，消除日志中的 4 个 `app-server` Windows 编译警告。本机 `npm run test:rust:related -- lime-rs/crates/app-server/src/command_exec/tests.rs lime-rs/crates/app-server/src/process/tests.rs` 通过 `1759/1759`，`cargo fmt --check` 通过；Windows 专属分支待恢复后的 Quality run 验证。
- 恢复提交 `9341eb297e6baecdf453eccf2aed2465bad99e4f6747091bf` 已推送到 `origin/main`；`v1.140.0` tag 仍固定在原始 release commit `1fe2ad260bb080a94a8dbda9e0d954f6747091bf`，未改写。
- 第二次 Release run `33826403443`：macOS x64/arm64 构建通过；Windows Electron 构建、Squirrel 安装、N-1 更新、SHELL-01、CodeMode Gate B、native-host Gate B 与 packaged evidence identity 全部通过。最后的卸载步骤失败，artifact `windows-squirrel-rc-evidence-x86_64-pc-windows-msvc` 显示主程序和快捷方式已移除，但 `C:\Users\runneradmin\AppData\Local\lime\Update.exe` 与 `app-1.140.0` 在 60 秒后仍存在，因此 Electron/CLI 资产和 R2 发布被跳过。
- 第二次失败根因是门禁把 Squirrel.Windows 的有意保留行为误判成卸载失败：其 `FullUninstall` 删除产品文件后会重建带 `.dead` 的安装根，并保留正在参与卸载钩子的 `Update.exe`。本轮修复先 fail-closed 验证候选 `Lime.exe` 与快捷方式已由 Squirrel 移除，再等待同路径 updater 进程退出，最后仅清理经过安装布局校验的候选 app 目录与 `Update.exe`；证据分别记录产品卸载和 runner 残留清理结果，未知卸载错误仍不容忍。路径/顺序/fail-closed 定向 Vitest 与 packaged/release guard 共 `68/68` 通过。
- 第二次 Windows 日志同时暴露 `tool-runtime` 的 4 条告警：删除从未进入生产调用图的 `should_preserve_windows_job` 及其自证测试；将 Windows SDK 的 `NERR_Success` / `NERR_GroupExists` 仅在模式匹配处导入为 Rust 风格大写别名。使用仓库校验后的 sandboxed rusty_v8 资产后，`npm run test:rust:unit -- -p tool-runtime` 通过 `382/382`，`cargo fmt -p tool-runtime --check` 通过；`cargo clippy -D warnings` 完成依赖构建后被仓库内 23 条既有 Rust 1.95 lint 阻断，不属于本轮 Windows rustc 告警。Windows 无告警编译待下一次 CI 给出平台证据。
- 第二轮恢复补丁发布门禁：`npm run typecheck`、`npm run test:contracts`、`npm run verify:app-version`、`npm run governance:scripts`、Prettier 与 `git diff --check` 均通过。
- 待执行：复核发布门禁后单独确认、提交并推送第二轮修复；通过 `workflow_dispatch version=v1.140.0` 从后续 `main` 修复提交重建资产，不移动已推送 tag；监控 Windows/macOS 构建、GitHub Release、CLI 资产与 R2 updater 完成。

## 收尾分类

- `current`：Electron Desktop Host、CLI/TUI Host、App Server JSON-RPC、RuntimeCore、Thread/Turn/Item projection、视频任务工具与 Desktop Gate B 证据。
- `compat`：无新增。
- `deprecated`：无新增。
- `dead / deleted`：旧 `lime-cli-npm` 包、旧 CLI skill/工具文档与其专用入口。

当前完成度：`88%`；版本 metadata、双语 release notes、原始 release commit/tag/push 与第二轮 Windows 根因修复已完成，等待提交推送后重新验证并发布远端资产。
