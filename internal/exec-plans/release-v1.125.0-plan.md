# Lime v1.125.0 发布执行计划

状态：released-with-windows-blocker
日期：2026-08-09
目标版本：`1.125.0`
目标 tag：`v1.125.0`

## 主目标

在不覆盖已发布的 `v1.124.0` tag 的前提下，发布当前工作树中的 Agent runtime、App Server/protocol、Plugin v3、GUI、文档与质量治理改动。

## Release Candidate

- `release metadata`：根 `package.json`、`packages/lime-cli-npm/package.json`、`lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`、`RELEASE_NOTES.md`、`RELEASE_NOTES.en.md`、本计划。
- `candidate changes`：当前工作树全部已跟踪和未跟踪改动，包括 Agent runtime、App Server/protocol、Plugin v3、Electron/GUI、测试、治理和文档。
- `excluded changes`：无；用户已明确确认当前工作树整体纳入。

## 退出条件

- 根 `README.md` 为英文 canonical 入口，英文页面无二维码，中文页面保留二维码。
- 版本事实源与双语 release notes 统一到 `1.125.0`，不覆盖 `v1.124.0`。
- 通过版本一致性、typecheck、docs boundary、contracts、GUI smoke 或明确记录环境限制。
- 完成 release commit、`v1.125.0` tag、`main`/tag 推送和远端复核。
- 针对发布 commit SHA 触发 `.github/workflows/build-windows-test.yml`，轮询 Windows runner 直至完成并保存 artifact 结果。

## 验证记录

- `npm run verify:app-version`：通过，所有版本事实源为 `1.125.0`。
- `npm run typecheck`：通过。
- Guardian projection Vitest：`54/54` 通过。
- `npm run test:contracts`：通过。
- `npm run docs:boundary`：通过。
- `npm run governance:legacy-report`：通过，零引用候选、分类漂移、边界违规均为 `0`。
- `npm run governance:scripts`：通过。
- `npm run governance:electron-release-workflow`：通过。
- `npm run verify:gui-smoke`：通过；Electron evidence `standalone-shell-01-20260809113946-74365`，App Server `1.125.0`。
- `npm run test:rust:related -- lime-rs/crates/app-server lime-rs/crates/mcp lime-rs/crates/skills lime-rs/crates/runtime-core lime-rs/crates/tool-runtime`：通过，`311 passed; 0 failed`。
- `npm run smoke:agent-runtime-current-fixture`：通过，覆盖当前 Agent runtime 全部 fixture；报告 `liveProviderUsed=false`。
- `git diff --check`：通过。
- Release commit：`8647d18fa`（`Release v1.125.0`）；evidence commit：`f55a35b49`（`Record Windows release runner result`）。`main` 与 `v1.125.0` 均已推送并完成远端复核。
- Windows runner 首次 run `31311967541`：失败于 `actions/checkout`，原因是 `source_ref` 使用短 SHA `8647d18fa`，workflow checkout 无法解析该引用；未进入构建。
- Windows runner 重试 run `31312029636`：Checkout、pnpm/Node/Rust/sccache、依赖安装均通过；`lime-mcp` Agent Plugin path contract 为 `10 passed / 3 failed`，失败集中在 Windows 扩展路径前缀与 8.3 短路径的字面断言（`agent_plugin_config_tests.rs`），未进入 sherpa、Electron、Squirrel smoke 或 Plugin Gate B，因此无 Windows 包 artifact。
- Windows runner 结论：发布提交已被真实 Windows runner 检出，但 Windows path contract 门禁阻断；不改写已推送的 `v1.125.0` tag。

## Windows 门禁修复

状态：fix-validated-locally

- 根因一：`build-windows-test` 直接把短 commit SHA 交给 `actions/checkout`，与输入声明的 commit SHA 能力不一致。
- 根因二：Windows MCP path contract 将同一文件系统路径的普通路径、extended-length 路径和 8.3 alias 做字面比较。
- 根因三：`guardianWarning` 已进入 generated notification union，但漏出 `AgentRuntimeSignalNotification`，导致 release/Quality 的 typecheck、GUI smoke 和三平台 Electron build 同源失败；该一行修复已由当前并发写集提供，本轮不覆盖。
- 窄写集：`.github/workflows/build-windows-test.yml`、`lime-rs/crates/mcp/src/agent_plugin_config{,_tests}.rs`、`scripts/electron/windows-squirrel-rc-smoke.test.mjs`、本计划。
- 退出条件：短 SHA 先解析为完整 commit SHA；Windows 断言按文件身份/Path component 比较；MCP 定向测试、workflow contract、typecheck 通过，并在真实 Windows runner 复核。
- 本地验证：MCP `8/8`、Windows workflow contract `16/16`、`npm run typecheck`、`npm run test:contracts`、`npm run governance:electron-release-workflow`、app-server-client 定向 `100/100`、`npm run verify:gui-smoke` 均通过；GUI evidence `standalone-shell-01-20260809135003-43144`。
- `npm run test:related -- packages/app-server-client/src/agent-runtime.ts`：入口在 Electron 目录解析时触发既有 `EISDIR`，未作为通过证据；改用直接 app-server-client 测试文件运行并通过。
- 真实 Windows runner：待提交并推送修复后复核。

## Windows runner 复核

- 修复提交 `b696976f3` 已推送到 `main`，但 runner `31317378340` 在编译 `lime-mcp` 测试时失败。
- 失败原因：Windows `std::os::windows::fs::MetadataExt::volume_serial_number` 与 `file_index` 仍依赖未稳定的 `windows_by_handle`，不能用于当前稳定 Rust toolchain。
- 修复方式：测试 helper 改用稳定标准库 `std::fs::canonicalize`，去除 extended-length 前缀后按 Windows 大小写不敏感比较；不引入生产依赖。
- runner `31317378340` 已确认 checkout、依赖安装成功，但未进入 sherpa、Squirrel、Gate B；需追加提交并重新触发 Windows runner。

## Windows packaged smoke 修复

- runner `31317797587`：path contract、sherpa runtime、Electron Squirrel package build 与 N-1 installer download 均通过；installed Squirrel smoke 在 app-server sidecar 首次启动时报 `thread 'main' has overflowed its stack`（退出码 `3221225725`），未进入 Plugin Gate B。
- 修复方式：Windows `app-server` 入口在 8 MiB 显式栈线程中启动 Tokio main；macOS/Linux 保持原入口。
- 本地验证：默认 target `cargo check -p app-server --bin app-server` 通过；Windows target 交叉 check 受本机缺少 MSVC C 头文件阻塞（`ring` 构建找不到 `assert.h`），需由 Windows runner 复核。

- runner `31318811255`：Windows path contract、sherpa runtime、Electron Windows package、N-1 installer download 与 Squirrel smoke 均通过；Plugin Gate B 在等待 MCP elicitation 表单 90 秒后超时。MCP server 已 initialize，但 ledger 没有 `tool_call`，启用插件后的 provider 请求数为 `0`，因此未产生用户可见的 `mcp_elicitation` pending interaction。
- 根因定位：入口线程栈修复已生效，但 Tokio worker 仍使用默认栈；App Server 的深层 agent/plugin 调用在 worker 上仍可能溢出或提前终止，导致 Gate B 没有进入 elicitation。
- 下一项修复：显式构造 Tokio multi-thread runtime，并在 Windows 将 worker stack 同样设置为 8 MiB；非 Windows 保持默认 worker stack 行为。

## Windows packaged Gate B 复核

- runner `31343997706` 使用 `dd97f79e6eb0a9fa2428416658fccefb13d03906`；Plugin 安装、启停边界、标准 manifest/mcp.json/Skill、MCP initialize capability 与 packaged Electron 均通过，但 `turn/start` 后 provider request 为 `0`，仅留下 MCP initialize，Gate B 超时。
- macOS packaged follow-up 使用同一当前工作树资源完成完整 Plugin Gate B，证明 packaged sidecar 资源解析和通用 Plugin/MCP runtime 主链可用；剩余问题限定为 Windows packaged turn execution 诊断。
- Gate B failure path 已补只读诊断，失败时保存 `thread/read`、`log/list`、`log/persistedTail`、`diagnostics/server/read`、renderer invoke trace、provider request 与 MCP ledger，供下一次 Windows runner 定位 turn 终态/错误；不改变通过条件。
- runner `31345998150` 使用完整 SHA `c228ffde93f14000d7ee6daa99113c96164a0a7a`；Checkout、插件路径契约、sherpa runtime、Electron Windows 包、N-1 Squirrel 下载与安装 smoke 均通过，Plugin Gate B 仍在 `submit-renderer-form` 等待 90 秒后失败。失败诊断确认 enabled turn 已 `completed/idle`、provider request 为 `0`、MCP ledger 仅有 runtime `initialize`，且失败截图显示 disabled boundary 文本；尚未取得结构化 turn/item 错误字段。
- 为下一轮 Windows runner 增加安全的 `thread/read` turn/item 状态摘要与 localhost provider connection diagnostics；本地 Plugin Gate B 已通过。下一步用该证据确认是 runtime turn 终态错误、Plugin snapshot/Skill 解析，还是 renderer session 投影漂移，再实施窄产品修复。
