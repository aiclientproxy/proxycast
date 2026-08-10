# Lime v1.125.0 发布执行计划

状态：release-git-confirmation-pending / windows-runner-pending
日期：2026-08-09
目标版本：`1.125.0`
目标 tag：`v1.125.0`

## 主目标

在不移动或覆盖已存在的 `v1.125.0` tag 的前提下，发布当前工作树中的 Agent runtime、App Server/protocol、Plugin v3、GUI、文档与质量治理改动。

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

## Windows target session 投影修复

状态：`fix-validated-locally / windows-runner-pending`

- runner `31347609969` 使用完整 SHA `22a96e5e1df9b771120ba6ea26ab7f562d5eafcd`；Windows path contract、sherpa runtime、Electron Windows package、N-1 Squirrel 安装 smoke 均通过，Plugin Gate B 仍在等待 `mcp_elicitation` 90 秒后失败。
- 失败证据确认 enabled canonical turn 已 `completed`、thread 已 `idle`，items 只有 user message 与空 agent message；enabled provider 只有 `/v1/models`，没有 `/v1/chat/completions`，而 disabled boundary provider 收到后续请求。失败截图标题已进入 enabled 会话，但正文仍显示 disabled boundary 结果，证明 renderer 的目标 session 与提交 thread 发生漂移。
- 根因：显式 `targetSessionId` 发送链完成 `ensureSession(target)` 后，canonical thread lookup 仍读取全局 `threadReadRef`。Windows 慢时序下该 ref 可能仍绑定旧会话，导致 enabled 请求提交到 disabled boundary thread/provider。
- 修复：`useAgentSession` 按 session 记录 canonical `thread_id`；显式目标提交只允许按目标 session 查询，映射缺失时刷新该目标的 read model，禁止回退旧会话 thread；同步收紧 prepared send / stream 类型契约并补时序回归测试。
- 窄写集：`src/components/agent/chat/hooks/useAgentSession.ts`、`agentStreamSubmitExecution.ts`、`agentStreamPreparedSendEnv.ts`、`useAgentStream.ts`、`agentStreamSubmitExecution.test.ts` 与本计划。上述部分文件同时含并发工作树改动，提交前必须重新确认 release candidate 范围。
- 本地验证：相关 Vitest `10/10` 通过；`npm run typecheck` 通过；`npm run smoke:plugin-package-electron-gate-b -- --timeout-ms 180000 --keep-temp` 通过，enabled provider request `2`、MCP elicitation accepted、provider final text observed、production mock fallback `0`。
- 下一步：提交并推送完整依赖闭包后，以新完整 SHA 重触发 `build-windows-test.yml`，持续跟踪到 Plugin Gate B 与 artifact 结论；不移动已发布的 `v1.125.0` tag。

## Windows provider selection sync 修复

状态：`fix-validated-locally / windows-runner-pending`

- runner `31352212702` 使用 `a5aae4e45bfd8b4e7379b0ba19c6a2af67080d01`；Windows path contract、sherpa runtime、Electron Windows 包、N-1 Squirrel 安装 smoke 均通过，Gate B 在 `mcp_elicitation` 等待 90 秒后失败。
- 失败证据显示 enabled thread 的公开 `modelProvider` 已是 enabled provider，但 durable `extra.providerSelector/providerName` 仍为 disabled provider；enabled 只收到 `/v1/models`，后续 `/v1/chat/completions` 仍进入 disabled provider。
- 根因：模型选择通过 microtask 异步执行 `thread/settings/update`，发送链只等待 UI 选择器状态，未等待目标 session 的 durable provider/model selection 写入完成；Windows 慢时序下 `turn/start` 先读取旧 route。
- 修复：`useAgentContext` 为每个 session 建立 selection-sync Promise 并暴露等待函数；`agentStreamSend` 在 prepare 前等待当前/目标 session，`useAgentStream` 将该能力注入真实 prepared send env；补 hook/send 时序回归。
- 本地验证：相关 Vitest `41/41`、改动文件 ESLint 通过；本地 macOS packaged Electron Gate B 通过，`ok=true`、`providerRequestCount=2`、MCP elicitation 已提交、provider final text 已观察、`productionMockFallbackHitCount=0`；Gate B evidence `plugin-package-electron-gate-b-20260810120829`（实际目录以本机生成结果为准）。
- 全量 `npx tsc --noEmit` 仍被工作树既有测试/协议类型错误阻断；过滤本轮文件未发现等待修复引入的新错误。Windows packaged Gate B 待新完整 commit 推送后复核。

## 当前发布候选复核

- `npm run typecheck`：通过（renderer 与 node 两个 tsconfig 均通过）。
- `npm run verify:app-version`：通过，版本事实源仍统一为 `1.125.0`。
- provider selection sync 窄写集 Vitest：7 个文件、`29 passed; 0 failed`。
- provider selection sync 窄写集 ESLint：通过，`--max-warnings 0`。
- `npm run test:contracts`：通过；protocol 生成无漂移、App Server client `301 checks`、命令/harness/modality/scripts/electron release/docs 边界均通过。
- 当前状态：`fix-validated-locally / release-git-confirmation-pending / windows-runner-pending`。
- 下一刀：获得危险操作确认后，将当前工作树全部纳入一个完整 release commit，创建并推送新的提交；由于 `v1.125.0` 已存在且不能覆盖，Windows workflow 使用该新完整 SHA 触发，继续跟踪 Plugin Gate B 与 artifact。

## Windows runner `31357775139`

- 使用完整 SHA `bc2d2aacc897deee60fd50c6479ff2c2aa483a15`，checkout、环境初始化和依赖安装通过。
- 在 Windows Agent Plugin path contract 编译阶段失败，错误为 `windows.rs` 的 `mod windows_acl` / `mod windows_attr` 找不到 sibling 文件；未进入 sherpa、Electron、Squirrel 或 Plugin Gate B。
- 根因：Rust 在内联模块文件 `execution_process/windows.rs` 中按默认规则寻找 `execution_process/windows/<module>.rs`，而两个模块文件位于 `execution_process/` 同级；macOS/Linux 的 cfg 未编译该路径，故本地未暴露。
- 修复：为两个声明增加显式 `#[path = "windows_acl.rs"]` 与 `#[path = "windows_attr.rs"]`；host 侧 tool-runtime related 测试通过，Windows 源文件 rustfmt 通过。本机 Windows target 交叉 check 仍受缺少 MSVC C 头文件阻断，待下一次 Windows runner 验证。
- 下一步：推送修复提交后重新触发 workflow，继续追踪到 Gate B 和 artifact。

## Windows runner `31358312342`

- 使用完整 SHA `05cd3e2b3c4d9a1a099732308bb0403423c90dd3`；checkout、Windows Agent Plugin path contract、sherpa runtime、Electron Windows x64 Squirrel package、N-1 installer download 与 Squirrel smoke 前置均通过。
- Plugin Gate B 在等待 `[data-testid="pending-interaction-layer"][data-interaction-kind="mcp_elicitation"]` 90 秒后失败；失败 artifact 已保存到本机临时目录，包含 summary/raw JSON、failure/install-review/plugin-mention 截图。
- 失败证据：enabled session `019fea2c-3d0a-7ad1-816f-f3ad959465b5` 的顶层 `thread.modelProvider` 已是新 provider，但 `thread.extra.providerName/providerSelector` 仍是旧 disabled provider；enabled provider 只收到 `GET /v1/models`，后续 `/v1/chat/completions` 进入旧 provider；MCP ledger 只有 `initialize`，enabled turn 最终为 `completed/idle`，没有 elicitation pending interaction。
- 根因：hydration 时 `useAgentSession.finalizeResolvedTopicDetail` 根据旧 session storage preference 延迟排队 metadata fallback；随后 Renderer 的 provider/model selection 虽通过 `useAgentContext` 写入新 route，但没有取消该 fallback，延迟 `thread/settings/update` 覆盖了 durable metadata。
- 修复：`useAgentChat` 共享 pending metadata cancel ref；`useAgentContext` 在用户 provider/model/reasoning selection 前取消旧 hydration fallback；`sessionMetadataSyncScheduler` 对包装任务增加取消标记，确保已进入 scheduler 队列的取消任务不再发起 RPC；补 context 与 scheduler 回归测试。
- 本地验证：`npm run typecheck` 通过；metadata controller/scheduler/context Vitest `30 passed; 0 failed`；本轮 ESLint `--max-warnings 0` 通过。
- 本机 packaged Plugin Gate B 复跑通过：evidence `plugin-package-electron-gate-b-summary.json`，`ok=true`、`providerRequestCount=2`、MCP elicitation accepted、provider final text observed、`productionMockFallbackHitCount=0`。
- 下一步：确认本轮 release candidate 后提交并推送完整 SHA，重新触发 `.github/workflows/build-windows-test.yml`，继续跟踪 Plugin Gate B 与 artifact；不移动已发布的 `v1.125.0` tag。

## Windows runner `31372663997`

- 使用完整 SHA `fb465072fbe14551aeadee530bf36c9c165a06c4`；Windows path contract、sherpa runtime、Electron Windows x64 Squirrel package、N-1 installer download 与 installed Squirrel smoke 全部通过，仍仅失败于 Plugin Gate B 等待 `mcp_elicitation` 90 秒超时。
- enabled provider 仍只有 `GET /v1/models`，disabled provider 收到后续 `/v1/chat/completions`；MCP ledger 只有 `initialize`。失败时 `thread.modelProvider` 保持 enabled provider，但 `thread.extra.providerSelector/providerName` 已回到 disabled provider。
- 新证据修正竞态顺序：Renderer provider selection 先于 hydration fallback 排队，因此 selection 时还没有可取消句柄；`finalizeResolvedTopicDetail` 随后因为 canonical `AgentSessionDetail` 缺少 `execution_runtime`，把 current thread 错判为“缺少 runtime route”，再次从旧 session storage 生成 provider fallback。
- 根因 owner：`readCanonicalThreadDetail` 已读取 current `Thread.modelProvider` 与 `Thread.extra.modelName`，但未投影 `AgentSessionDetail.execution_runtime`，使 hook 层失去 current route 事实源。
- 修复：canonical Thread projection 从顶层 `modelProvider`（优先于可能 stale 的 extra provider）和 metadata `modelName` 构造 session execution runtime；保留 imported source markers；补顶层 current provider 覆盖 stale extra provider 的回归测试。
- 本地验证：canonical projection/App Server session client/metadata controller Vitest `48 passed; 0 failed`；`npm run typecheck` 与 projection ESLint `--max-warnings 0` 通过；本机 packaged Plugin Gate B 复跑 `ok=true`、enabled provider request `2`、MCP elicitation/final text 完成、production mock fallback `0`。Windows packaged Gate B 待新完整 SHA 复核。

## Windows runner `31378335257`

状态：`navigation-fix-validated-locally / windows-runner-pending`

- 使用完整 SHA `88b5bf6a74dd8d911d71e46b1418c0f1dea1210c`；Windows path contract、sherpa runtime、Electron Windows x64 Squirrel package、N-1 installer download 与 installed Squirrel smoke 全部通过。
- Plugin Gate B 尚未进入安装、provider route 或 turn，在 `open-plugin-catalog-app-center` 等待 bundled Browser Plugin 卡片 600 秒超时。失败截图显示最终页面回到新建任务首页；原脚本此前已观察到插件页安装入口，证明一次侧边栏导航在 Windows packaged 启动尾部时序中被覆盖，而不是 `plugin/list` 或 canonical execution runtime 失败。
- `fb465072f..88b5bf6a7` 没有侧边栏、App Navigation、App Center 或 Gate B 脚本改动；本轮不把该平台时序误归因到 canonical route 修复。
- 修复：Plugin Gate B 继续只走真实侧边栏点击，但要求插件页安装入口、loading terminal 与 bundled Browser Plugin 卡片在同一次短窗口内共同收敛；Windows 启动尾部导航覆盖时最多重试 3 次，每次上限 30 秒，不再用全局 600 秒盲等。最终失败会附带 URL、active sidebar、可见 testid 与主内容摘要。
- 本地验证：脚本 `node --check`、定向 ESLint、Plugin Gate B guard 通过；本机 packaged Plugin Gate B 完整通过，`ok=true`、App Center 安装/启停与 disabled boundary 通过、provider request `2`、MCP elicitation accepted、provider final text observed、cold restore/卸载后历史通过、production mock fallback `0`。
- 本轮补充验证：`npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、`npm run test:rust:related -- lime-rs/crates/app-server-protocol lime-rs/crates/app-server`、`cargo test --manifest-path lime-rs/Cargo.toml -p app-server --test fuzzy_file_search_jsonrpc`、`cargo fmt --all -- --check`、`npm run verify:gui-smoke` 均通过；GUI evidence `standalone-shell-01-20260810111610-3054`。
- 当前 release candidate 继续纳入完整工作树，包括同期 fuzzy file search App Server v2 协议、schema、processor、公共 JSON-RPC 集成测试与协调计划；`v1.125.0` tag 保持指向既有 release commit，不移动或覆盖。
- 下一步：完成当前整批 Rust/protocol/contracts/typecheck/版本门禁后提交并推送新完整 SHA，重新触发 `build-windows-test.yml` 并跟踪 installed Plugin Gate B 与 artifact 最终结论。

## Release candidate 当前收口

- 全部 tracked/untracked 改动均纳入，无排除项；新增 fuzzy file search 使用 Codex current 一发式 `fuzzyFileSearch` 协议和 snake_case 结果字段，未恢复已排除的 sessionStart/sessionUpdate/sessionStop 双轨。
- `npm run typecheck`：通过。
- `npm run test:contracts`：通过，协议生成无漂移、App Server client `301 checks`、脚本治理、Electron release workflow 与 docs boundary 均通过。
- `npm run verify:app-version`：通过，版本事实源统一为 `1.125.0`。
- Composer/fuzzy file mention Vitest：`32 passed; 0 failed`；App Server client fuzzy Vitest：`2 passed; 0 failed`；Gate B 脚本契约 Vitest：`5 passed; 0 failed`。
- `cargo test --manifest-path lime-rs/Cargo.toml -p app-server --test fuzzy_file_search_jsonrpc`：`1 passed; 0 failed`。
- `cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check`、定向 ESLint、`git diff --check`：通过。
- `npm run verify:gui-smoke`：通过；最新 Electron evidence `standalone-shell-01-20260810112853-16763`，App Server `1.125.0`。
- 本机 packaged Plugin Gate B 既有证据保持通过：`ok=true`、provider request `2`、MCP elicitation accepted、provider final text、cold restore/卸载历史和 `productionMockFallbackHitCount=0`。
- 当前阻塞只剩危险 Git 写操作确认，以及新完整 SHA 推送后的 Windows runner Plugin Gate B/artifact 复核；既有 `v1.125.0` tag 固定在 `8647d18fa358e3a9c86e520348d39e4b3eba6041`。

## Windows runner `31384560366`

状态：`fix-validated-locally / windows-runner-pending`

- 使用完整 SHA `668699a4d29a4302e7db46aaab58644f2e46b61a`；Windows Plugin path contract、sherpa runtime、Electron Windows x64 Squirrel package、N-1 installer 下载与 installed Squirrel smoke 全部通过。
- Plugin Gate B 已通过 App Center 安装/启停、禁用边界、provider 两次请求、MCP elicitation、最终文本与 canonical MCP App item 投影；唯一失败是首次显式恢复后等待固定累计 `2` 次 resource read/HTML load。
- 失败 evidence 同时证明 WebContents marker 已加载且 MCP ledger 已有 `1` 次 `resource_read`。脚本在 reload 前没有等待隐式首次挂载，Windows 上 reload 抢先后只有一次有效加载；随后 60 秒等待使 240 条 invoke trace ring buffer 被 drain 记录覆盖，最终诊断误显示计数为 `0`。
- 修复：首次显式恢复只要求至少一组 resource read/HTML load 且一一对应；下一次 Renderer reload 按已观察基线要求各精确增加一次；cold restore 继续要求独立进程增加一次。最终有效最小计数从依赖隐式竞态的 `4` 收敛为确定性的 `3`，不放宽任何真实产品边界。
- 本地验证：Gate B 脚本 Vitest `8 passed; 0 failed`、定向 ESLint、Node 语法检查、`npm run typecheck`、`npm run test:contracts`、`npm run verify:app-version` 全部通过。本机真实 Electron Plugin Gate B `ok=true`，resource read/HTML load `4/4`、launch `2`、provider request `2`、reload/cold restore/卸载历史全部通过、`productionMockFallbackHitCount=0`、`missingRequiredMethods=[]`。
- 下一步：通过 Gate B 脚本单测、定向 ESLint、contracts 与本机 packaged Gate B 后，提交/推送窄修复并以新完整 SHA 重跑 Windows runner；`v1.125.0` tag 不移动。
