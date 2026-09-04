# Desktop + CLI/TUI 多 Surface 执行计划

状态：进行中
日期：2026-09-03
主目标：同一 App Server / RuntimeCore 同时承载 Desktop 与 CLI/TUI；未来 Cloud 只预留可验证的 transport 边界，不提前复制 runtime。

## 写集

- `lime-rs/crates/app-server-client/**`
- `lime-rs/crates/tui/**`
- `lime-rs/crates/cli/**`
- `lime-rs/Cargo.toml`、`lime-rs/Cargo.lock`
- `packages/cli/**`
- `pnpm-lock.yaml`
- `scripts/app-server/{stdio-smoke,cli-gate-b,tui-gate-b}.mjs`
- `scripts/governance/cli-boundary*.mjs`
- `src/components/agent/chat/utils/{toolDisplayConfig/content,limeTaskProtocolNoise,taskPreviewVideo,toolProcessSummaryBuilders}.ts`
- `src/components/agent/chat/workspace/generalWorkbenchHelpers{,.test}.ts`
- `src/components/agent/chat/{hooks,utils,components}/**/*video*.test.*` 中的 current 视频工具夹具
- `internal/prd/tools/{README,inventory,task-file-protocol-prd}.md`
- `internal/prd/skills/async-image-skill-task-prd.md`
- `internal/tech/lime-agent-harness-architecture.{mmd,svg}`
- `.gitignore`、`internal/exec-plans/README.md`
- `AGENTS.md`
- `internal/aiprompts/{architecture,commands,governance,quality-workflow}.md`
- 本计划

避让：`electron/**`、除上述精确 Renderer 文件外的 `src/**`、已有 Desktop parity/Goose 计划及其当前脏改动；本轮不改变 Desktop Host 命令或业务协议。

## 架构裁决

```text
Desktop Renderer -> Electron Desktop Host --\
CLI/TUI -> app-server-client ----------------> App Server JSON-RPC -> RuntimeCore
Future Cloud -> authenticated transport -----/ -> canonical Thread/Turn/Item
```

- `current`：App Server v2、RuntimeCore、ThreadStore、`app-server-client` session、`cli`、`tui`。
- `compat`：无。
- `deprecated`：无。
- `dead`：旧 `lime-cli` crate、`task/media/skill/doctor` 命令、`tools/lime-cli` task catalog、CLI 直连媒体 runtime/数据库，以及任何复制 Codex core/runtime、TUI 私有数据库、production mock backend 或 Cloud 假 endpoint 的方案。

## Codex 迁移策略

参考：`/Users/coso/Documents/dev/rust/codex/codex-rs/{cli,tui,app-server-client}`。

直接迁移并改接 Lime owner：

- app-server client actor：并发 pending request、ordered notification、reverse request、bounded shutdown。
- terminal lifecycle：raw mode、alternate screen、panic/drop 恢复。
- composer：Unicode 安全光标、粘贴、多行输入、历史。
- TUI state/event/render 分离，TestBackend/VT100 snapshot 方法。
- display width、grapheme truncation、窄终端 guard。
- thread start/resume、turn start/steer/interrupt、Item/Reasoning/Tool/Plan projection。

不直接复制：

- `codex-core`、Codex auth/config/state DB、provider manager、sandbox runtime。
- ChatGPT/Codex Cloud 专属 account、rate limit、marketplace 和 remote-control 产品逻辑。
- Codex 私有品牌文案、telemetry、更新器、Desktop launcher。

这些能力若属于 Lime 产品需求，只能通过既有 App Server/current domain owner 重建。

## 阶段

### P0 架构与连接底座（本轮）

- [x] 更新多 surface 架构、治理、命令和质量事实源。
- [x] 在 `app-server-client` 建立 transport-neutral async session 与本地 stdio transport。
- [x] 覆盖并发请求、notification-before-response、reverse request、server error、握手失败清理和 shutdown 测试。
- [x] 建立 `tui` crate、Codex width/truncation 移植与稳定 TestBackend 渲染测试。
- [x] `lime` 增加默认 interactive、显式 `tui` 与 `exec` 入口。
- [x] 把既有 App Server stdio smoke 从旧 `sessionId` DTO 迁到 current v2 `thread/start` fail-closed 合同。

退出条件：Rust related tests、`cargo test -p app-server-client -p tui -p cli`、`npm run test:contracts` 通过；架构确认完成。

### P1 可用 Agent TUI

- [x] 新建 Thread、发送 turn、interrupt 基础接线。
- [x] 恢复 Thread、active turn steer 与队列输入。
- [x] Message、Reasoning、Command、Patch、MCP、Plan、Multi-Agent Item 基础投影。
- [x] Command、Patch、MCP、Plan、Multi-Agent Item 基础专用布局；布局只消费 canonical `TranscriptEntry`，不复制第二套会话模型。
- [x] 补全上述 Item 的结果摘要、细粒度进度和终态投影。
- [x] command/file/permission approval 与 request_user_input。
- [x] session picker（`lime resume` 可省略 Thread id，通过 `thread/list` 选择后复用 `thread/resume`）。
- [x] effort/permission controls（Codex 风格 Alt-`,` / Alt-`.` 与 F7/F8，统一 lowering 到 `thread/settings/update`）；`/model` 通过 `model/list` 打开可见 catalog picker，选择后统一 lowering 到 `thread/settings/update`。
- [x] prompt history 通过 App Server `promptHistory/read|append` 持久化，TUI composer 只保留最近 200 条。
- [x] CLI `tui`/`exec`/`resume` 已支持 `--model`、`--provider`、`--effort`、`--permissions`，并统一 lowering 到 `thread/settings/update`；TUI 内部 model picker 与快捷控件同样走 typed App Server methods。
- [x] Unicode composer、paste、多行输入、历史、窄终端和 drop 恢复保护。
- [x] bounded reconnect、旧 connection pending interaction 清理和 panic hook 终端恢复。
- [x] 扩展 TUI Gate B：真实 PTY 覆盖 Thread 恢复、工具审批、request_user_input、运行中 interrupt、失败恢复和 alternate-screen 恢复。
- [x] 精确 resize/scrollback：按 Ratatui 实际换行高度计算可滚动范围，PageUp/PageDown 使用当前 transcript viewport 翻页并在 resize 后重新取尺寸。
- [ ] external editor 的完整 job control。
- [ ] 终端用户可见文案覆盖 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR` 并补稳定回归。

已完成扩展 Gate B：真实 `lime tui` + portable PTY + stdio App Server 覆盖新对话、键盘输入、Thread 恢复、工具审批、request_user_input、运行中 Ctrl-C interrupt、失败终态和 alternate-screen 恢复。所有场景均使用显式 external fixture backend，仅验证 App Server/RuntimeCore/TUI 生命周期，不调用正式 Provider。

退出条件：真实 `lime` + stdio App Server TUI Gate B 覆盖新对话、恢复、工具审批、中断和失败恢复。

### P2 CLI 完整面

- [x] 默认无子命令进入 TUI，支持规范命令 `tui` 与 `exec`。
- [x] 支持 `resume`（交互式恢复）以及 `thread list/show/archive/unarchive/delete/fork` typed 管理命令。
- [x] `mcp list` 与 `skills list` 通过 App Server v2 typed methods 查询状态/catalog；`mcp list` 自动遍历分页并防止 cursor 环。
- [x] 删除旧 `doctor`，诊断能力留在 App Server owner，不在 CLI 建第二套产品 surface。
- [x] 删除 `task/media` 和单数 `skill` 命令；媒体与内容任务由 App Server typed tools/workflows 承接。
- [x] 删除旧同步 task/media/doctor 实现、旧 task Skill catalog 及 `media-runtime`/本地数据库直接依赖。
- [x] 将 Rust owner 收敛为 `crates/cli`，外发包装收敛为 `packages/cli` / `@limecloud/lime`。
- [x] 增加 `governance:cli-boundary`，禁止旧路径、依赖和命令回流。
- [ ] JSON/JSONL、退出码、stdin、非 TTY 和 shell completion 稳定合同。

退出条件：CLI e2e fixture 与帮助快照覆盖 macOS/Windows；所有业务命令只走 App Server。

### P3 Codex TUI 完整对齐

- [ ] 按 Codex snapshot inventory 逐场景迁移 composer、history、reconnect、multi-agent、model switch、tool lifecycle。
- [ ] 每个 snapshot 标记 direct/merge/contract/defer/dead，不保留无 owner 测试。
- [ ] 终端 Markdown、diff、image/clipboard、narrow viewport 与 accessibility 行为对齐。

退出条件：迁移账本无未分类场景，关键 snapshot 和 TUI Gate B 全绿。

### P4 Cloud transport（未来，未启动）

- [ ] 先确认 identity、tenant、auth、TLS、protocol version、resume、rate limit、audit。
- [ ] 实现 `app-server-client` authenticated remote transport，不改变 TUI/CLI domain state。
- [ ] 建立跨网络断线/重连、租户隔离和凭证泄露负向测试。

退出条件：安全评审和真实远端 evidence；此前不得创建假 Cloud production path。

## 验证合同

- 风险：重大架构 + Rust workspace/依赖 + 新 TUI product surface。
- Unit：transport actor、composer、projection、width/truncation、TestBackend render。
- Integration：真实 app-server stdio initialize/thread/turn fixture。
- CLI Gate B：真实 `lime exec` 二进制、canonical identity 与完成输出。
- TUI Gate B：真实 `portable-pty`、alternate screen、键盘输入、Thread 恢复、工具审批、request_user_input、运行中 interrupt、失败终态和终端恢复；fixture backend 只作为显式测试边界。
- Desktop 回归：本轮不改 Desktop；协议边界变化仍运行 `npm run test:contracts`。
- 禁止：live Provider 默认运行、production mock fallback、固定 sleep 合成 terminal completion。

## 架构图确认

- [x] `internal/aiprompts/architecture.md` 已包含 Desktop/CLI/TUI/Cloud transport 图。
- [x] 责任开发者确认该图准确表达目标。
- [ ] PR 描述同步架构图确认与 CLI/TUI Gate B evidence。

## 本轮验证证据

- `cargo test -p app-server-client`：41 项通过。
- `cargo test -p tui`：57 项通过；除既有底盘外，新增 command 退出码/耗时、patch 文件统计、MCP/dynamic tool 结果与错误、multi-agent 状态分布、image generation 结果摘要、turn 终态收敛及 TestBackend 摘要渲染断言；projection 回归已拆到独立测试模块。真实 stdio/PTY 测试通过环境变量显式启用，不调用正式 provider。
- `cargo test -p cli`：15 项通过；覆盖 Thread 管理命令、`mcp list`/`skills list` 解析、model/effort/permission 参数、规范 `tui` 入口、`exec --jsonl` 单行 envelope、与 `--json` 互斥、stdin 非 TTY 输入、既有退出码语义，以及由统一 `Cli` 命令树生成的 `lime completion <shell>`；zsh 脚本已实际 smoke 输出，且不再包含旧 task/media/skill/doctor 正向测试。
- `npm run test:rust:related -- ...`：通过，并覆盖 `app-server-test-client` 反向依赖。
- CLI/TUI 增量后的 `npm run test:rust:related -- "lime-rs/crates/app-server-client/src" "lime-rs/crates/tui/src" "lime-rs/crates/cli/src"`：通过；覆盖 app-server-client、app-server-test-client、tui 与 cli。
- `npm run test:contracts`：通过。
- `DYLD_LIBRARY_PATH="lime-rs/target/debug:lime-rs/target/sherpa-onnx-prebuilt/sherpa-onnx-v1.13.0-osx-arm64-shared-lib/lib" npm run smoke:app-server-stdio`：通过，真实 App Server 子进程 + current v2 + non-mock fail-closed；未设置该路径时本地产物因缺少 `LC_RPATH` 无法启动。
- 真实二进制 `lime --help`、`lime exec --help`、`lime --version` 与 `lime exec --json` 错误 envelope：通过。
- 真实二进制 `lime thread --help`、`lime resume --help` 与 unavailable App Server 下的 `lime thread list`：通过；响应按 v2 typed response 解码。
- `npm run smoke:cli-gate-b`：通过；真实 `lime exec -> stdio -> App Server -> RuntimeCore -> canonical Item -> CLI JSON`，Thread/Turn identity 与 backend ledger 一致。
- `npm run smoke:tui-gate-b`：通过扩展矩阵 `complete,approval,user-input,interrupt,failure`；approval/user-input 的 `actionRespond`、interrupt 的 `turnCancel`、failure 的 `runtime.error,turn.failed` 均由 backend ledger 证明，并验证终端在每个场景退出后恢复。
- `npx vitest run scripts/app-server/{cli,tui}-gate-b.test.mjs`：4 项通过。
- `cargo clippy --no-deps -D warnings`：`app-server-client`、`tui`、`cli` 均通过；包含依赖的扩大检查仍被 `agent-protocol` 既有 `large_enum_variant` / `derivable_impls` 告警阻塞。
- `cargo build -p app-server`：未完成；`v8 150.4.0` 的 `aarch64-apple-darwin` 预编译 archive 上游返回 404。进程 smoke 复用了本地已有 App Server 开发产物，默认环境还受 sherpa 动态库 `LC_RPATH` 缺失影响，未宣称验证本轮 App Server 重建。

## 当前进度与下一刀

完成度：86%。P0、CLI Gate B、TUI 扩展 Gate B、Thread resume、session picker、bounded reconnect、active turn steer、queue input、Thread typed 管理命令、MCP/Skills typed 查询、prompt history、approval/request_user_input、panic-safe terminal、effort/permission shortcuts、model catalog picker、精确 resize/scrollback、专用 Item 布局、Item 结果摘要/细粒度终态，以及真实 PTY Gate B 测试拆分已完成。下一刀是多语言文案和 external editor job control；随后建立 Codex snapshot inventory。

代码体量退出条件：真实 PTY Gate B 已迁到 `tui/src/runtime_pty_tests.rs`，设置命令与 reconnect 分别迁到 `settings.rs` 和 `reconnect.rs`，`runtime.rs` 已降至 792 行；本轮将 projection 回归移到 `projection_tests.rs` 后，`projection.rs` 降至 707 行。后续 Item 布局和本地化不得回填到 runtime，且必须保持本计划中的相关 Gate B 证据不变。

## 2026-09-03 本地回归复验

- `cargo test -p tui`：45/45 通过；`cargo test -p app-server-client -p cli`：通过。
- 修复并验证 TUI model picker 的模块导出、Ratatui owned title、model catalog 过滤测试，以及 picker 弹层渲染；`/model` 选择仍通过 current App Server `model/list` 和 `thread/settings/update`，不形成第二 catalog 或 Session/Message owner。
- `npm run verify:local`：119/119 Vitest、Rust workspace lib/tests/doctest、contracts、治理和真实 Electron GUI smoke 全部通过。TUI Gate B 仍是现有 fixture 的 `lime tui -> portable PTY -> App Server JSON-RPC`，不调用正式 provider。
- `npm run smoke:tui-gate-b`：扩展五场景全部通过；`interrupt` 使用 Ctrl-C 取消活动回合后再退出，其他场景使用 Ctrl-D，避免终态后重复取消造成竞态。
- 新增 `external_backend_cancellation_kills_hanging_process` 单测，验证 CancellationToken 会终止悬挂 external backend 子进程并清理 stderr reader。
- 2026-09-03 拆分后复验：`runtime_pty_tests.rs` 独立承载真实 PTY Gate B；修复 TUI 不应以可选 prompt-history 告警覆盖 `active_turn_id` 中断资格的问题。使用重新构建的 `lime` 与 `app-server` 二进制重跑 `complete,approval,user-input,interrupt,failure`，五场景全部通过；`interrupt` 由 ledger 记录真实 `turnCancel`，并确认 UI 先显示 `interrupting` 后退出。
- `lime doctor`、`lime task`、`lime media`、单数 `lime skill` 及其旧同步实现已删除；诊断与媒体能力只允许由 App Server/current typed owner 承接。
- 视频生成的唯一 Agent tool 为 `video_generate`，经 `tool-runtime -> mediaTaskArtifact/video/create -> worker` 执行；旧视频工具名只允许存在于负向守卫或历史 evidence。

## 2026-09-04 TUI 职责与 scrollback 复验

- 设置命令与 bounded reconnect 已拆到职责单一模块；不改变 App Server method、Thread/Turn/Item projection 或重试语义。
- scrollback 改为使用 Ratatui 实际 rendered line count，修复窄终端长行换行后无法滚到真实顶部的问题；PageUp/PageDown 改为按当前 transcript viewport 翻页，并在 resize 后重新读取终端尺寸。
- `cargo test -p tui`：50/50 通过；`cargo test -p app-server-client -p cli`：53/53 通过；`cargo fmt --all -- --check` 与 `git diff --check` 通过。
- 使用本地 V8 archive 重新构建 `cli` 与 `app-server` 后运行 `npm run smoke:tui-gate-b`：`complete,approval,user-input,interrupt,failure` 五场景通过，ledger 保持真实 `actionRespond` / `turnCancel` 证据，终端均恢复。
- `npm run test:contracts` 与 `npx vitest run scripts/app-server/tui-gate-b.test.mjs`（2/2）通过；external backend 仍只作为显式测试 fixture，不进入 production fallback。
- Command、Patch、MCP、Plan、Multi-Agent 已按 canonical `ThreadItem` 分类到独立 terminal layout；Plan 使用稳定 checkbox 状态，Patch 保留 diff 增删着色，Command 输出与命令头分层。`cargo test -p tui` 增至 53/53，并由 TestBackend 证明布局进入真实终端投影；`cargo clippy -p tui -p cli --no-deps -- -D warnings` 通过，重新构建后五场景 TUI Gate B 保持通过。

## 2026-09-04 旧 CLI 与视频工具命名收口

- `pnpm-lock.yaml` workspace importer 已从已删除的 `packages/lime-cli-npm` 收敛到 `packages/cli`；Rust workspace 继续只包含 `cli`、`tui` 与 `app-server-client` current owners。
- Renderer 的视频工具显示、失败摘要、历史恢复和任务预览正向夹具全部切到 `video_generate`；旧视频工具精确名只保留在 retired contract guard、catalog `is_none()` 负向断言、协议泄漏清理负向测试和历史 evidence。
- active 工具文档、图片任务文档与已渲染架构图不再把 `lime-cli`、`lime task/media/doctor` 或旧视频工具当作 current；Task File PRD 明确归档为历史方案。
- 定向前端回归：9 个核心测试文件、154/154 通过；补充工作台 helper 与摘要回归 31/31 通过；受影响 TS/TSX ESLint 与 Prettier 通过。
- `npm run test:contracts`、`npm run governance:scripts`、`npm run governance:legacy-report`、`npm run governance:cli-boundary`、`npm run docs:boundary`、`npm run verify:app-version` 与 `git diff --check` 全部通过；`xmllint --noout internal/tech/lime-agent-harness-architecture.svg` 通过。
- 收口后的 `npm run verify:local` 完整通过：119/119 Vitest 批次、App Server client 299 项 contract 检查、Rust workspace lib/integration/doctest，以及真实 Electron GUI smoke 均为绿色；GUI smoke 经过真实 Desktop Host、preload/IPC、App Server 初始化、工作台 reload、3 个响应式视口与设置页，并生成 `standalone-shell-01-20260903181151-6148` Gate 证据。
- `pnpm list --depth -1 --recursive` 未执行成功：本机 Corepack 校验 pnpm `9.15.9` 时缺少匹配签名 key。未安装或升级包；锁文件改用本地 YAML 解析和 importer 精确断言验证。

## 2026-09-04 Item 摘要与终态复验

- `projection.rs` 将 Command、Patch、MCP、Dynamic Tool、Multi-Agent、Image Generation 的 canonical 结果字段投影为稳定摘要，并将 `TurnCompleted` 后仍为 `running` 的工具项收敛到 completed/failed/interrupted 终态。
- `entry.rs` 与真实 TestBackend transcript 渲染摘要；终端回归覆盖退出码、耗时、补丁文件统计和完成态文案。
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p tui`：56/56 通过；`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-client -p cli`：53/53 通过。
- `cargo clippy --manifest-path "lime-rs/Cargo.toml" -p tui -p cli --no-deps -- -D warnings`、`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check`、`npm run test:contracts`：通过。
- 使用本地 V8 资产重建 `cli` 与 `app-server` 后，`npm run smoke:tui-gate-b` 通过；真实 `lime + PTY + App Server` 场景终端恢复正常，external backend 仍仅作为显式测试 fixture。

## 2026-09-04 External Editor 跨平台收口

- `external_editor.rs` 通过 `TempPath` 在启动编辑器前关闭临时文件句柄，并在 Windows 目标平台解析 PATH 中的 `.cmd/.bat` shim；编辑器命令解析和临时文件编辑脚本回归通过。
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p tui`：54/54 通过；`cargo clippy --manifest-path "lime-rs/Cargo.toml" -p tui --no-deps -- -D warnings` 与 workspace fmt 检查通过。
- 完整 external editor job-control（独立进程组/信号恢复和 macOS/Windows 真实终端证据）仍未标记完成，待后续按 Codex TUI snapshot 与平台门禁补齐。
