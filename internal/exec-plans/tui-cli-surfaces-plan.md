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
- `scripts/app-server/{stdio-smoke,cli-gate-b,cli-surface-gate-b,cli-npm-gate-b,cli-structure-inventory,tui-gate-b,tui-snapshot-inventory}.mjs` 及相应测试
- `.github/workflows/build-windows-test.yml`
- `scripts/governance/cli-boundary*.mjs`
- `src/components/agent/chat/utils/{toolDisplayConfig/content,limeTaskProtocolNoise,taskPreviewVideo,toolProcessSummaryBuilders}.ts`
- `src/components/agent/chat/workspace/generalWorkbenchHelpers{,.test}.ts`
- `src/components/agent/chat/{hooks,utils,components}/**/*video*.test.*` 中的 current 视频工具夹具
- `internal/prd/tools/{README,inventory,task-file-protocol-prd}.md`
- `internal/prd/skills/async-image-skill-task-prd.md`
- `internal/tech/lime-agent-harness-architecture.{mmd,svg}`
- `.gitignore`、`internal/exec-plans/README.md`、`internal/exec-plans/{tui-codex-snapshot-inventory,cli-codex-test-inventory,cli-structure-inventory}.{md,json}`
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

参考：`/Users/coso/Documents/dev/rust/codex/codex-rs/{cli,tui,app-server-client}` 与 `/Users/coso/Documents/dev/rust/codex/codex-cli`。

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
- [x] 扩展 TUI Gate B：真实 PTY 覆盖 Thread 恢复、工具审批、request_user_input、运行中 interrupt、失败恢复、queued follow-up 编辑和 alternate-screen 恢复。
- [x] 精确 resize/scrollback：按 Ratatui 实际换行高度计算可滚动范围，PageUp/PageDown 使用当前 transcript viewport 翻页并在 resize 后重新取尺寸。
- [x] external editor 按 Codex 终端交接模型完成：暂停 TUI、恢复终端、继承前台 PTY stdio、异步等待并恢复 alternate screen；不创建会导致后台 TTY 读取风险的独立进程组。
- [x] 终端用户可见文案覆盖 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR` 并补稳定回归。

已完成扩展 Gate B：真实 `lime tui` + portable PTY + stdio App Server 覆盖新对话、键盘输入、Thread 恢复、工具审批、request_user_input、运行中 `Esc` interrupt、失败终态、`Tab -> Alt+Up` queued follow-up 编辑和 alternate-screen 恢复。所有场景均使用显式 external fixture backend，仅验证 App Server/RuntimeCore/TUI 生命周期，不调用正式 Provider。

退出条件：真实 `lime` + stdio App Server TUI Gate B 覆盖新对话、恢复、工具审批、中断和失败恢复。

### P2 CLI 完整面

- [x] 默认无子命令进入 TUI，支持规范命令 `tui` 与 `exec`。
- [x] 支持 `resume`（交互式恢复）以及 `thread list/show/archive/unarchive/delete/fork` typed 管理命令。
- [x] `mcp list/get/add/remove/start/stop/login/logout` 与 `skills list` 通过 App Server current typed methods 查询或变更 catalog；MCP 配置创建/删除只调用 `mcpServer/create/delete`，OAuth logout 在缺少协议方法时拒绝本地删凭据。
- [x] 删除旧 `doctor`，诊断能力留在 App Server owner，不在 CLI 建第二套产品 surface。
- [x] 删除 `task/media` 和单数 `skill` 命令；媒体与内容任务由 App Server typed tools/workflows 承接。
- [x] 删除旧同步 task/media/doctor 实现、旧 task Skill catalog 及 `media-runtime`/本地数据库直接依赖。
- [x] 将 Rust owner 收敛为 `crates/cli`，外发包装收敛为 `packages/cli` / `@limecloud/lime`。
- [x] 增加 `governance:cli-boundary`，禁止旧路径、依赖和命令回流。
- [x] JSON/JSONL、退出码、stdin、非 TTY 和 shell completion 稳定合同。
- [x] 按 Codex `codex-rs/cli` 对齐 Rust 目录、模块、类型、函数和测试命名；根命令使用 `MultitoolCli/Subcommand`，MCP/Plugin/Features/Queue/Debug 使用独立 current owner 模块。
- [x] 按 Codex `codex-cli` 对齐 npm launcher、platform alias、native payload staging、signal forwarding 与 npm package tests；目录和符号差异写入 `cli-structure-inventory.json`。
- [x] 真实 CLI surface Gate B 覆盖 `mcpServer/list/create/delete`、`features/list`、`plugin/list`、`memory/reset`，OAuth logout 在协议未提供凭据删除方法前保持 fail-closed。
- [x] 对齐 `/Users/coso/Documents/dev/rust/codex/codex-cli` 的 npm 根包/平台包架构：ESM launcher、optional dependency alias、异步 spawn、signal forwarding、退出原因镜像和 package manager 归属。
- [x] 平台包以 `vendor/<target-triple>/bin` 原子携带 `lime`、`app-server`、`code-mode-host`、Windows sandbox helpers 与 App Server 动态运行库；发布顺序固定为平台包先于根包。
- [x] 当前 catalog 只声明已有 release runner 的 macOS arm64/x64、Windows x64、Linux x64 GNU；Windows arm64/Linux arm64 留作真实构建证据驱动的扩展，不发布空 optional package。
- [x] 物理删除旧 `install.js`、`run.js`、`release-meta.js`、`build-release.js`，并以负向治理守卫禁止恢复 postinstall 下载、同步 wrapper、cargo fallback 与旧单二进制 archive。

退出条件：CLI e2e fixture 与帮助快照覆盖 macOS/Windows；所有业务命令只走 App Server；实际 npm staging/launcher 测试覆盖根包、平台包、完整 sidecar 载荷、退出码、signal 和缺失 optional dependency 的 fail-closed。

### P3 Codex TUI 完整对齐

- [x] 建立 [Codex snapshot inventory](./tui-codex-snapshot-inventory.md)，并记录上游 commit、内容 hash、Lime owner 与迁移分类。
- [ ] 按 Codex snapshot inventory 逐场景迁移 composer、history、reconnect、multi-agent、model switch、tool lifecycle。
- [x] 每个 snapshot 标记 direct/merge/contract/defer/dead，不保留无 owner 测试。
- [x] 终端 Markdown 第一刀：Assistant/Reasoning 在 TUI render 边界解析标题、强调、删除线、行内/围栏代码、列表、引用、任务标记、链接和基础表格；不改变 canonical Item 文本或协议状态。
- [x] unified diff 使用独立 terminal renderer，投影文件/hunk、增删行、行号 gutter 与 metadata 样式。
- [x] Clipboard 与 image paste 对齐 Codex：`/copy`/`Ctrl+O` 复制最后一条 canonical Assistant Markdown，本地剪贴板失败时使用 tmux/OSC 52/WSL fallback；`Ctrl/Alt+V` 读取图片或剪贴板文件、规范化为 PNG，并通过既有 `UserInput::LocalImage` 进入 start/steer/queue。
- [x] terminal OSC 8 hyperlink、宽度感知表格、narrow viewport 与 accessibility fallback 按 inventory 对齐：链接元数据与可见文本分离，wrap/scroll 后重映射到 Ratatui Buffer cell；仅安全 `http/https` 进入 OSC 8，非支持终端仍显示完整目标；表格在受限宽度下按内容收缩、换行或转为线性 key/value records。
- [x] Codex syntax highlighting 纯终端子集迁入 `tui::highlight`：Markdown fenced code 与 Diff 共用 `syntect + two-face` grammar、ANSI terminal palette、语言别名和资源上限；Diff 按 old/new 文件流保持独立语法状态。
- [x] Codex 风格 slash command popup：只枚举 Lime 已真实实现的 TUI 命令，复用唯一命令目录，覆盖前缀筛选、键盘选择、窄终端渲染与五语言描述。
- [x] Codex 风格 static pager 与 `/status`：只读取 current TUI/App Server session facts，支持滚动、翻页、跳转、关闭和窄终端换行，不向 canonical transcript 注入本地伪 Item。
- [x] Codex 风格 `Ctrl+T` transcript overlay：从 current `ConversationProjection` 实时派生完整历史视图，复用 Markdown/Diff/OSC 8 render owner 与 pager 滚动状态，不复制 `TranscriptEntry` 或 canonical session state。
- [x] Codex 风格 queued follow-up preview/edit：直接消费 App Server `thread/queue/list` 与 `thread/queue/changed` 的 canonical `QueuedSubmission`，展示文本/图片排队内容、恢复后的持久队列与窄终端截断；`Alt+Up` 通过 `thread/queue/delete` 取回最后一条可无损恢复的输入，不向 transcript 注入本地伪 Item。
- [x] 纯终端算法 owner 按 Codex 原名补齐：`wrapping.rs`（`RtOptions`、标准/URL-aware wrapping、UTF-8 ranges）、`terminal_palette.rs`（颜色等级与 ANSI256 量化）和 `table_detect.rs`（GFM 表格/围栏跟踪）；不复制 Codex 私有 terminal probe、runtime 或持久化状态。

退出条件：迁移账本无未分类场景，关键 snapshot 和 TUI Gate B 全绿。

### P4 Cloud transport（foundation，生产 Cloud 未启动）

服务端对应计划：`/Users/coso/Documents/dev/ai/limecloud/limecore/docs/exec-plans/lime-cloud-app-server-plan.md`。
Lime 与 LimeCore 只共享 App Server protocol/session 合同；租户隔离、凭证和 Cloud 运维能力由服务端计划单独验收。

- [x] Remote foundation 已确认服务端 identity、auth、TLS 与 protocol version：Bearer token 仅允许
  `wss://` 或 loopback `ws://`，握手在 `initialized` 前固定 `app-server` 身份并拒绝非
  `appserver.v0` 版本。
- [ ] Cloud tenant isolation、凭证存储、跨网络 resume、rate limit 与 audit 仍需真实服务端合同和 evidence。
- [x] 实现 `app-server-client` authenticated remote transport，不改变 TUI/CLI domain state；CLI/TUI 通过 `--remote` 与 `--remote-auth-token-env` 选择同一 session facade。
- [ ] 建立跨网络断线/重连、租户隔离和凭证泄露负向测试。

退出条件：安全评审和真实远端 evidence；此前不得创建假 Cloud production path。

## 验证合同

- 风险：重大架构 + Rust workspace/依赖 + 新 TUI product surface。
- Unit：transport actor、composer、projection、width/truncation、TestBackend render。
- Integration：真实 app-server stdio initialize/thread/turn fixture。
- CLI Gate B：真实 `lime exec` 二进制、canonical identity 与完成输出。
- TUI Gate B：真实 `portable-pty`、alternate screen、键盘输入、Thread 恢复、工具审批、request_user_input、运行中 interrupt、失败终态、queued follow-up 编辑和终端恢复；fixture backend 只作为显式测试边界。
- Desktop 回归：本轮不改 Desktop；协议边界变化仍运行 `npm run test:contracts`。
- 禁止：live Provider 默认运行、production mock fallback、固定 sleep 合成 terminal completion。

## 架构图确认

- [x] `internal/aiprompts/architecture.md` 已包含 Desktop/CLI/TUI/Cloud transport 图。
- [x] 责任开发者确认该图准确表达目标。
- [ ] PR 描述同步架构图确认与 CLI/TUI Gate B evidence。

## 本轮验证证据

- `cargo test -p app-server-client`：41 项通过。
- `cargo test -p tui`：168 项通过；除既有底盘外，新增 command 退出码/耗时、patch 文件统计、MCP/dynamic tool 结果与错误、multi-agent 状态分布、image generation 结果摘要、turn 终态收敛、locale 文案、Markdown/table/local-link、宽度感知 table、OSC 8 hyperlink、numbered unified diff、syntax highlighting、clipboard copy/image paste、图片-only/图片+文本 lowering、失败恢复、active-turn status、queued follow-up 及 TestBackend 布局断言；projection 回归已拆到独立测试模块。真实 stdio/PTY 测试通过环境变量显式启用，不调用正式 provider。
- `cargo test -p cli`：本轮最新为 library 8 项、binary 41 项，共 49 项通过；覆盖 Thread 管理命令、`mcp list`/`skills list` 解析、model/effort/permission 参数、规范 `tui` 入口、`exec --jsonl` 单行 envelope、与 `--json` 互斥、stdin 非 TTY 输入、既有退出码语义、Codex 原名的 sandbox parser 测试、`sandbox_setup` 测试、`execpolicy check` prefix/justification/alternative matching，以及由统一 `MultitoolCli` 命令树生成的 `lime completion <shell>`。
- `npm run test:rust:related -- ...`：通过，并覆盖 `app-server-test-client` 反向依赖。
- CLI/TUI 增量后的 `npm run test:rust:related -- "lime-rs/crates/tui/src" "lime-rs/crates/tui/Cargo.toml" "lime-rs/Cargo.lock"`：通过；因锁文件触达 workspace 边界自动扩大到 `cargo test --lib --workspace`，所有批次全绿。
- `npm run test:contracts`：通过。
- `DYLD_LIBRARY_PATH="lime-rs/target/debug:lime-rs/target/sherpa-onnx-prebuilt/sherpa-onnx-v1.13.0-osx-arm64-shared-lib/lib" npm run smoke:app-server-stdio`：通过，真实 App Server 子进程 + current v2 + non-mock fail-closed；未设置该路径时本地产物因缺少 `LC_RPATH` 无法启动。
- 真实二进制 `lime --help`、`lime exec --help`、`lime --version` 与 `lime exec --json` 错误 envelope：通过。
- 真实二进制 `lime thread --help`、`lime resume --help` 与 unavailable App Server 下的 `lime thread list`：通过；响应按 v2 typed response 解码。
- `npm run smoke:cli-gate-b`：通过；真实 `lime exec -> stdio -> App Server -> RuntimeCore -> canonical Item -> CLI JSON`，Thread/Turn identity 与 backend ledger 一致。
- `npm run smoke:tui-gate-b`：通过扩展矩阵 `complete,approval,user-input,interrupt,failure,queue-edit`；approval/user-input 的 `actionRespond`、interrupt 的 `turnCancel`、failure 的 `runtime.error,turn.failed` 由 backend ledger 证明；queue-edit 由 canonical event log 证明同一 Thread 上 `thread/queue/add -> thread/queue/delete` 顺序，并验证预览、composer 恢复和各场景终端恢复。

## 2026-09-04 Remote transport foundation

- `app-server-client/src/remote.rs` 新增 WebSocket `SessionTransport`，支持 `ws://`/`wss://`、Bearer token、ping/pong、文本 JSON-RPC、关闭和 128 MiB message/frame 上限；认证 token 仅允许 `wss://` 或 loopback `ws://`，不安全远端 fail closed。
- `ClientSession::start_remote`、TUI `AppServerSession::connect_remote` 与 CLI `start_session` 复用同一 initialize/request/event/shutdown actor；TUI reconnect 和 session picker 也按同一 remote 配置工作，未创建第二套 runtime、Thread 状态或持久化。
- CLI 参数与 Codex 形状对齐：`--remote URL`、`--remote-auth-token-env ENV_VAR`；缺失 endpoint、缺失/空 token 环境变量均在连接前失败。
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-client -p cli -p tui`：44 + 34 + 177 全部通过；新增 remote auth policy、WebSocket roundtrip、CLI parser 和 fail-closed 回归。
- P4 剩余未完成：identity、tenant isolation、凭证存储、协议版本协商、跨网络 resume/rate limit/audit 与真实 Cloud endpoint；在这些条件完成前，Cloud 继续保持架构扩展点，不进入生产发布路径。
- `npx vitest run scripts/app-server/{cli,tui}-gate-b.test.mjs scripts/app-server/tui-snapshot-inventory.test.mjs`：7 项通过；包含 Windows CLI/TUI workflow 结构守卫。
- `cargo clippy -p tui -p cli --no-deps -- -D warnings`：通过。
- `cargo build -p cli -p app-server`：直接运行仍因 `v8 150.4.0` 的 `aarch64-apple-darwin` 上游 archive 返回 404；使用仓库 `scripts/lib/rusty-v8-artifacts.mjs` 返回的已校验 archive/binding 显式注入后重建成功。随后 Gate B 使用本轮新二进制，不复用旧产物。

## 当前进度与下一刀

2026-09-06 TUI resume picker transcript expansion：继续按 Codex `resume_picker.rs`
对齐会话展开路径。`PickerState` 增加 `expanded_thread_id`、`SessionTranscriptState` 与
canonical `TranscriptEntry` 缓存；Ctrl+E 与 raw Ctrl-E 事件后台调用
`thread_transcript::load_session_transcript_with_handle`，并在 TestBackend 中渲染会话
metadata、加载/失败/空 transcript 和按入口类型区分的 transcript 内容。分页结果按
canonical thread id 去重，归档成功清理展开缓存；修复异步摘要的 transcript 顺序，并补充
状态、渲染、顺序和宽度回归，TUI lib tests 达到 282 项，Clippy、结构 inventory Vitest
与 `git diff --check` 通过。

本刀仍只消费 App Server `Thread/Turn/Item -> ConversationProjection`，没有复制 Codex
私有 rollout、history DB 或 `TranscriptCells`；Codex 完整 overlay/pager 仍属于后续
`resume_picker` 扩展，不得在当前 picker 外建立第二 transcript owner。

2026-09-05 TUI 命名继续对齐 Codex：`bottom_pane/approval.rs` 已迁移为
`bottom_pane/approval_overlay.rs`，`Approval`、`ApprovalRequest::Command` 与
`ApprovalRequest::FileChange` 分别收敛为 `ApprovalOverlay`、`Exec` 与 `ApplyPatch`；
`ChatComposer` 的结果变体统一为 `InputResult::Submitted`/`Queued`，并提供
Codex 形状的 `handle_key_event(KeyEvent)`。request-user-input 的 `Event` 入口只保留
粘贴和按键适配，实际按键处理统一走 `RequestUserInputOverlay::handle_key_event`。
渲染按可用宽度使用 Codex 的 grapheme-safe word-boundary ellipsis，并复制
`halfwidth_sound_marks_are_truncated_at_a_grapheme_boundary` 与
`halfwidth_sound_marks_are_truncated_and_rendered_at_a_grapheme_boundary` 测试。

当前不能声明 99% 完成。Codex Rust CLI 的 433 项测试账本现为 `covered=48`、`partial=90`、`pending=0`、`deferred=64`、`excluded=231`；结构账本同时记录 Codex 缺失面与 Lime-only 文件/符号，防止只做单向对照。P0、CLI/TUI 扩展 Gate B、Thread resume、session picker、bounded reconnect、active turn steer、queue input、Thread typed 管理命令、MCP/Skills typed 查询、prompt history、approval/request_user_input、panic-safe terminal、effort/permission shortcuts、model catalog picker、精确 resize/scrollback、专用 Item 布局、Item 结果摘要/细粒度终态、五语言 TUI 文案、Codex 同构 external editor 终端交接、CLI JSON/JSONL/stdin/退出码/completion 合同、`execpolicy check` 只读规则报告、802 项 Codex snapshot 分类账本、代表性终端 Markdown/diff、clipboard copy/image paste、OSC 8 hyperlink、宽度感知表格、ANSI syntax highlighting、slash command popup、`/status` static pager、`Ctrl+T` transcript overlay、active-turn status、reverse history search 与 canonical queued follow-up preview/edit 已完成。Windows CI 仍需补齐 external editor、clipboard/image paste、syntax highlighting native dependency 与六场景 CLI/TUI 平台运行证据。

当前刀已完成：`debug_sandbox.rs` 保留 Codex 的 `SandboxStateArgs`、平台命令类型、`SandboxType`、`DebugSandboxConfigOptions`、`ManagedRequirementsMode` 与 `run_command_under_*` 名称，并通过 `app-server-client -> command/exec` 进入唯一 sandbox owner；`sandbox_setup.rs` 复制 Codex 的目录、类型、函数和五个 parser 测试名，执行通过现有 `windowsSandbox/setupStart`，不直连 Windows helper；`ExecpolicyCommand`/`ExecpolicySubcommand`/`run_execpolicycheck` 归 `cli/src/main.rs`，`ExecPolicyCheckCommand`、`PolicyParser`、`Policy` 与 `RuleMatch` 归独立 `execpolicy` crate，重复 CLI 实现已移除；`exit_status.rs` 与 `handle_exit_status` 已补齐。CLI inventory 的 `missing` 已收敛为 0，剩余 Codex 专属路径均有明确 `excluded/deferred` 归类。

当前下一刀：本地权限 profile、inherited stdio、`execpolicy check`、remote identity/protocol/credential foundation 均已完成。后续只处理具备 Lime App Server current owner 和真实平台/Cloud evidence 的 `sandbox-state`、managed network、跨网络 resume、tenant isolation、rate limit、audit、macOS denial logger/PID tracker、Windows managed-user setup 与 code-mode host URL；在 owner 或 evidence 缺失时继续 fail closed，不在 CLI/TUI 伪造第二套 runtime。CLI Rust 结构已按 Codex 收敛：根命令、连接参数、Debug、Features、App Server、Thread/Exec/TUI/Execpolicy 类型与测试统一归 `src/main.rs`；仅保留 Codex 同名独立模块 `mcp_cmd.rs`、`plugin_cmd.rs`、`queue_cmd.rs`、`sandbox_setup.rs`、`debug_sandbox.rs`、`exit_status.rs`。

## 2026-09-04 CLI Rust 结构收敛

- 获得明确确认后，删除 Lime-only 的 `src/app_server_cmd.rs`、`src/commands.rs`、`src/commands_tests.rs`、`src/debug_cmd.rs`、`src/features_cmd.rs`。
- 按 Codex `codex-rs/cli/src/main.rs` 的 owner 形状，把 `ConnectionArgs`、`TuiCli`、`ExecCli`、`ResumeCommand`、Thread/Skills 命令、App Server 转发、Debug、Features、JSON/JSONL、remote transport、prompt 读取与对应测试统一收回 `src/main.rs`。
- `mcp_cmd.rs`、`plugin_cmd.rs`、`queue_cmd.rs`、`sandbox_setup.rs` 只通过 crate 根导出的 `ConnectionArgs`、`request_value`、`start_session` 依赖统一 owner；没有新增兼容包装或第二套 runtime。
- 结构账本重新生成：`rustFilesOnlyInLime=[]`；Codex product-specific/deferred 文件仍按双向差异记录，npm 平台专属文件继续单独保留 owner。
- 验证：`cargo check --manifest-path "lime-rs/Cargo.toml" -p cli`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p cli`（library 4 + binary 37）、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-client -p tui -p cli`（45 + 180 + 4 + 37）、CLI 结构/测试/surface Vitest 7/7、`node scripts/governance/cli-boundary.mjs`、CLI 写集 `git diff --check` 通过。
- 全 workspace `cargo fmt --check` 未纳入通过结论：并行热区 `code-mode-host/src/grpc/conversions_tests.rs` 已存在格式差异，本刀未修改该文件。

## 2026-09-05 npm launcher runtime handoff

- 根因：macOS 在 Node launcher 启动 native child 的边界会清掉 `DYLD_LIBRARY_PATH`；打包后的 App Server 使用 `@rpath` 加载 Sherpa/ONNX 动态库，导致 npm launcher 场景在 initialize 前收到 `app-server transport closed`。
- 修复：`packages/cli/bin/lime.js` 按 Codex launcher 的异步 spawn/信号转发结构保留原有行为，并将当前平台包 `vendor/<target-triple>/bin` 前置到 macOS `DYLD_LIBRARY_PATH`、Linux `LD_LIBRARY_PATH`；不改变 Windows 环境或协议。
- 证据：`npm run smoke:cli-npm-gate-b` 通过，真实 npm 根包 launcher -> sibling App Server -> RuntimeCore -> CLI/TUI；`node --test packages/cli/tests/npm-package.test.mjs` 8 项通过；`npm run smoke:tui-gate-b` 六场景通过；`cargo test --manifest-path "lime-rs/Cargo.toml" -p tui -p app-server-client -p cli` 全部通过。

代码体量退出条件：真实 PTY Gate B 已迁到 `tui/src/runtime_pty_tests.rs`，设置命令与 reconnect 分别迁到 `settings.rs` 和 `reconnect.rs`，`runtime.rs` 当前生产逻辑止于第 762 行；本轮将 projection 回归移到 `projection_tests.rs` 后，`projection.rs` 降至 707 行。后续 Item 布局和本地化不得回填到 runtime，且必须保持本计划中的相关 Gate B 证据不变。

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
- 对照 Codex `tui/src/external_editor.rs` 与 `Tui::with_restored` 后修正规划：Codex 同样继承当前前台 PTY，不创建独立进程组；Lime 保持该语义，避免未配套 `tcsetpgrp` 时触发后台读取 `SIGTTIN`。
- 编辑器执行改为 Tokio async child，显式继承 stdin/stdout/stderr；TUI 在调用前离开 alternate screen 并关闭 raw mode，退出后恢复 terminal guard。
- 真实 PTY fixture 在 Unix 上断言 fd 0/1/2 均为 TTY，编辑后的 prompt 继续沿 canonical `turn/start` 进入 App Server；Windows 使用 `.cmd` shim 和 ConPTY 继承路径，当前轮未在 Windows 主机实跑，留给 Windows CI evidence。

## 2026-09-04 TUI 多语言文案收口

- 新增 TUI locale owner，支持 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR` 的环境变量解析和显式渲染选择。
- Header、状态、条目终态、结果摘要、审批、request_user_input、model picker 与 session picker 均通过同一 locale owner 渲染；用户输入、模型原文、协议 enum/schema/evidence facts 和未知服务端错误保持不变。
- `view` TestBackend 回归覆盖五种 locale 的用户可见 header/status 文案；`locale` 单测覆盖 locale family、标签和未知文本保留。
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p tui`：62/62 通过；`cargo clippy --manifest-path "lime-rs/Cargo.toml" -p tui --no-deps -- -D warnings`、workspace fmt check 与 `git diff --check` 通过。

## 2026-09-04 CLI P2 合同收口

- `scripts/app-server/cli-gate-b.mjs` 在同一真实 `lime -> stdio App Server -> RuntimeCore -> canonical Item` fixture 链验证 `exec --json`、单行 `exec --jsonl`、pipe stdin 非 TTY、空 prompt 的退出码 1 + error envelope，以及 canonical `Cli` 命令树生成的 zsh completion。
- locale 只暴露给默认 `lime`、`lime tui` 和 `lime resume`；`exec`、Thread/MCP/Skills 管理命令的 JSON 合同继续保持语言无关。
- `npm run smoke:cli-gate-b` 通过，输出证据包含 `jsonl=ok stdin=ok error-exit=1 completion=zsh`；`node --check scripts/app-server/cli-gate-b.mjs` 与对应 Vitest source guard 2/2 通过。

## 2026-09-04 Codex TUI snapshot inventory

- 参考 `codex-rs/tui` commit `cac96cd7b1756ab42e8925d938817a2ac10ebb6e`，生成 `tui-codex-snapshot-inventory.json`：802/802 snapshot 均有相对路径、内容 SHA-256、分类和规则 owner。
- 排序后相对路径 SHA-256 为 `da5d7b14f30cccefa3132e2c33b4aaf1e0460ce9a342fc571f857ce15a21ef03`；分类计数：`direct=48`、`merge=579`、`contract=80`、`defer=25`、`dead=70`。
- Codex 产品专属 onboarding/account/update/migration 不进入 Lime current TUI；纯终端算法才允许 `direct`，跨 runtime/state owner 的行为必须按 canonical App Server contract 重建。
- `npm run inventory:tui-codex` 可在显式 `CODEX_TUI_REFERENCE` checkout 上刷新账本；静态 Vitest guard 精确锁定 commit、路径摘要、802 项分类计数与 Hook feedback 语义例外，CI 不依赖仓库外 Codex checkout。

## 2026-09-04 终端 Markdown 第一刀

- 新增 `tui/src/markdown.rs`，在 Ratatui render 边界消费 Assistant/Reasoning 的 Markdown 文本；支持标题、粗体/斜体/删除线、行内代码、围栏代码、无序/有序列表、任务标记、引用、水平线、链接目标与基础表格。
- 新增 `tui/src/diff_render.rs`，解析 unified diff hunk 范围并渲染稳定行号、gutter、file/hunk metadata 与增删颜色。
- `entry.rs` 在既有 `TranscriptEntry -> view` 边界渲染 Assistant/Reasoning Markdown 与 Patch diff；User、Command、Tool、MCP、Plan、协议字段和摘要继续走原有 projection/layout，不新增会话或 runtime owner。
- `cargo test -p tui`：当前并行工作树 77/77 通过；Markdown/diff 单测覆盖空输入、多行、代表性块级/行内元素、可见链接目标、基础表格、嵌套列表、Reasoning 样式、unified diff 行号与样式，并由 TestBackend 证明接入真实 transcript projection。
- `npm run test:rust:related -- "lime-rs/crates/tui/src" "lime-rs/crates/tui/Cargo.toml"` 通过并识别 `cli` 反向依赖；`cargo clippy -p tui --no-deps -- -D warnings`、TUI fmt check、`npm run governance:scripts` 与 `git diff --check` 通过。
- `npm run test:contracts`、CLI/TUI/inventory source guard 6/6、`npm run smoke:cli-gate-b` 与默认五场景 `npm run smoke:tui-gate-b` 通过；真实 App Server fixture 保持 canonical Thread/Turn/Item 与 terminal restore 证据。

## 2026-09-04 Clipboard 与 Image Paste 收口

- `clipboard_copy.rs` 对齐 Codex 的本地剪贴板优先策略，并补 tmux capability、OSC 52、SSH 与 WSL PowerShell fallback；`/copy` 和 `Ctrl+O` 只复制 canonical projection 最后一条 Assistant 原始 Markdown，不把命令写入 Turn。
- `clipboard_paste.rs` 对齐 Codex 的 `Ctrl/Alt+V` 图片入口：优先读取剪贴板文件列表，其次读取 RGBA image，支持 PNG/JPEG/GIF/WebP 解码并规范化为持久临时 PNG；Linux/WSL 在 native clipboard 不可用时使用 Windows PowerShell fallback。未创建独立媒体协议或 runtime owner。
- TUI 的 App Server facade 将 start/steer/queue 收敛为 `Vec<UserInput>` typed input；图片统一 lowering 为既有 `UserInput::LocalImage`，按 Codex 顺序置于 Text 之前。图片-only 提交有效，空文本不写 prompt history；发送失败恢复待提交附件。
- composer 在文本上方稳定显示 `[Image #N]`，附件行进入输入高度、光标偏移和 scrollback page-size 计算；空 composer 下 Backspace 移除最后一张附件。成功/失败状态覆盖 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`。
- `cargo test -p tui -p cli`：TUI 86/86、CLI 15/15 通过；`cargo clippy -p tui -p cli --no-deps -- -D warnings` 与 workspace fmt 通过。测试覆盖快捷键、图片-only、图片+文本顺序、失败恢复、PNG bytes/suffix、五语言状态和 TestBackend 附件布局。
- `npm run test:rust:related -- "lime-rs/crates/tui/src" "lime-rs/crates/tui/Cargo.toml" "lime-rs/Cargo.lock"` 因 lockfile 风险自动扩大到 workspace lib tests 并全绿；`npm run test:contracts` 通过 299 项 client checks、命令/模态/脚本/CLI 边界与文档守卫。
- 裸 `cargo build -p cli -p app-server` 首次命中上游 `rusty_v8 v150.4.0` Apple ARM archive 404；通过仓库 `scripts/lib/rusty-v8-artifacts.mjs` 的已校验本地 archive/binding 重建当前二进制成功。使用该二进制重跑 `npm run smoke:cli-gate-b` 与默认五场景 `npm run smoke:tui-gate-b`，两条真实 stdio/PTY App Server 主链和 terminal restore 均通过。

## 2026-09-04 OSC 8、表格、Diff 与 Windows current-path 门禁

- `terminal_hyperlinks` 在 Ratatui Buffer 边界注入 OSC 8，链接 metadata 与可见文本分离，并在 wrap/scroll 后重映射范围；仅带 host 的 `http/https` 进入控制序列，控制字符、危险 scheme 与超长 destination 均 fail closed。
- Markdown 表格按显式 viewport width 做内容感知列宽、cell wrap、系统性碎裂时的 key/value records 与窄屏 stacked fallback；本地链接支持 Unix、Windows drive、UNC、`file://`、percent decode 与行列锚点，但不生成 OSC 8。
- unified diff 展开 tab、动态计算行号 gutter，并按 viewport 折行与对齐续行；一列窄屏对宽字符显示省略占位，避免静默丢失内容。
- Windows test package workflow 已接入 `cargo test -p cli -p tui`、`cargo build -p cli -p app-server`、CLI Gate B 与当前默认六场景 TUI Gate B，并始终上传统一日志；结构测试锁定步骤、命令、场景来源和 artifact 合同。
- `cargo test -p tui -p cli`：TUI 109/109、CLI 15/15 通过；`cargo clippy -p tui -p cli --no-deps -- -D warnings`、workspace fmt、相关文件 Prettier 与 `git diff --check` 通过。
- `npm run test:rust:related -- "lime-rs/crates/tui/src" "lime-rs/crates/tui/Cargo.toml" "lime-rs/Cargo.lock"`：因锁文件触达 workspace 边界扩大到 `cargo test --lib --workspace`，全部通过；`npm run test:contracts` 与 `npm run governance:legacy-report` 同样通过，`cli-boundary` 明确报告 `current=cli+tui retired=lime-cli`。
- 裸 `cargo build -p cli -p app-server` 再次复现上游 `rusty_v8 v150.4.0` Apple ARM archive 404；通过仓库 `buildLocalAppServer -> resolveRustyV8CargoEnv` 的 SHA-256 校验资产重建成功。随后 CLI/TUI Gate B 均使用本轮产物并通过。
- 当前仅完成 Windows runner 接线与静态 workflow 回归，尚未取得本次改动在真实 Windows runner 上的 platform evidence，不能据此宣称 Windows 已通过。

## 2026-09-04 Diff metadata 与 syntax highlighting

- Diff 文件头新增 cwd-relative path、rename `source → destination`、`(+N -N)` 统计；无 hunk 的 add/delete 继续给稳定行号，多 hunk 用 `⋮` 分隔并隐藏原始 `@@` header。
- `tui::highlight` 复制 Codex 可独立复用的 `syntect + two-face` 核心，固定使用终端原生 ANSI palette；覆盖约 250 种 grammar、Codex 语言别名、CRLF、未知语言 plain fallback，以及 512 KiB/10000 行/4 KiB 单行 fail-closed 上限。
- Markdown fenced code 消费 CommonMark info string 的首个语言 token；unknown language 保留原文。Diff 从 rename 目标或 `---`/`+++` path 识别语言，old/new 文件流分别维持 parser state，删除到 `/dev/null` 时保留源扩展名，跨行字符串与 styled wrap 不丢文本或样式。
- 新依赖仅落在 `tui` render owner：`syntect = 5`、`two-face = 0.5`；不新增 App Server method、runtime 状态、主题配置事实源或 Cloud 分支。
- `cargo test -p tui`：130/130 通过；本轮新增 18 项 highlighter、Markdown/Diff 接线、rename/delete、双流 parser state、资源上限与 ANSI color mapping 回归。`cargo clippy -p tui -p cli --no-deps -- -D warnings` 通过。
- `npm run test:rust:related -- "lime-rs/crates/tui/src" "lime-rs/crates/tui/Cargo.toml" "lime-rs/Cargo.lock"` 因锁文件触达 workspace 边界扩大到 `cargo test --lib --workspace`，全部通过；最终 `cargo test -p tui -p cli` 为 TUI 130/130、CLI 15/15。
- `npm run test:contracts`、`npm run governance:legacy-report`、CLI/TUI/inventory Vitest 7/7、workspace fmt 与 `git diff --check` 通过；CLI Gate B 和真实 PTY TUI Gate B 使用本轮重建的 `lime` 二进制通过，terminal restore 正常。
- Windows 仍只有 workflow 接线和 source guard；本轮新增 `onig` native dependency 尚未取得真实 Windows runner platform evidence，readiness 保持 fail closed。

## 2026-09-04 Slash popup 与 `/status` static pager

- slash command popup 只枚举 `/model`、`/effort`、`/permissions`、`/status`、`/copy` 五个已实现命令；支持前缀筛选、上下选择、Tab 补全、Enter 执行、Esc 关闭，并以精确负向测试禁止 `task/media/skill/doctor` 回流。
- `/status` 使用借用型 `StatusFacts` 从 current App Server/TUI session facts 构造临时快照；pager 独占 frame，支持逐行滚动、翻页、Home/End、Esc/Q、resize 后范围裁剪与窄终端换行，不写 canonical transcript 或第二套 session model。
- popup 描述、status labels 与 pager footer 覆盖 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`；审批/request_user_input 激活时不展示 popup，服务端交互请求或断线会关闭 pager。
- `cargo test -p tui -p cli`：TUI 145/145、CLI 15/15 通过；`cargo clippy -p tui -p cli --no-deps -- -D warnings`、workspace fmt、Rust related tests 与 `git diff --check` 通过。
- `npm run smoke:tui-gate-b` 使用本轮二进制覆盖 `/status -> pager visible -> q -> external editor -> turn/start -> turn.completed -> terminal restore`；TUI Gate B/inventory Vitest 5/5 通过。`npm run smoke:cli-gate-b`、`npm run test:contracts` 与 `npm run governance:legacy-report` 保持通过，`cli-boundary` 报告 `current=cli+tui retired=lime-cli`。

## 2026-09-04 `Ctrl+T` transcript overlay

- `PagerOverlay` 按 Codex 区分 static 与 transcript 内容：`Ctrl+T` 打开时默认跟随底部，再次 `Ctrl+T`、Esc 或 Q 关闭；手动上滚后 canonical projection 增长不会强制跳回底部，End 可恢复跟随。
- transcript overlay 不持有 `TranscriptEntry` 副本；每次 draw 都从 current `ConversationProjection` 派生完整历史，并复用 `entry::hyperlink_lines_with_locale`、Markdown、Diff、syntax highlight 与 `HyperlinkParagraph`，保留窄终端换行和 OSC 8。
- 标题与 footer 覆盖 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`；overlay 独占 frame，不显示或提交 composer draft，不向 Thread/Turn/Item 写本地伪 Item。审批/request_user_input 与断线沿用既有优先级，继续关闭 pager。
- `cargo test -p tui -p cli`：TUI 150/150、CLI 15/15 通过；Rust related tests、`cargo clippy -p tui -p cli --no-deps -- -D warnings`、workspace fmt、Prettier 与 `git diff --check` 通过。
- TUI Gate B/inventory Vitest 5/5 通过；`npm run smoke:tui-gate-b` 使用本轮重建的 `lime`，真实 PTY 在 canonical message/tool 更新期间打开 transcript，验证内容可见、`Ctrl+T` 关闭、五场景事件矩阵与 terminal restore。
- `npm run test:contracts`、`npm run governance:legacy-report` 与 `npm run governance:cli-boundary` 通过；legacy 报告为零引用候选 0、分类漂移候选 0、边界违规 0，CLI 边界保持 `current=cli+tui retired=lime-cli`。
- 当前并行工作树复核将本地 `/status`、`/copy` 提前到 Composer submit/history 之前消费；新增回归证明 `/status` 不进入 prompt history 或 canonical projection，并覆盖 Down/PageDown/Up、Home/End、resize 后 scroll bound 重算、零/极窄区域、空状态到 ready 的五语言归一化与繁中 `狀態` 文案。最新 `cargo test --manifest-path "lime-rs/Cargo.toml" -p tui` 为 151/151，`cargo clippy --manifest-path "lime-rs/Cargo.toml" -p tui --no-deps -- -D warnings`、workspace fmt、Rust related workspace lib tests、`npm run test:contracts` 与 `git diff --check` 通过。
- 使用校验后的本地 rusty-v8 资产重建当前 `lime`/`app-server`；在并行 transcript fixture 改用稳定可见 marker 后，默认 `npm run smoke:tui-gate-b` 的 `complete,approval,user-input,interrupt,failure` 五场景全绿。真实 PTY 覆盖 `/status -> visible -> q -> composer -> Ctrl+T transcript -> turn.completed`、App Server `actionRespond` / `turnCancel` ledger 与 terminal restore，external backend 仍只作为显式 fixture。

## 2026-09-04 Queued follow-up preview

- 参考 Codex `bottom_pane/pending_input_preview.rs` 与 `status_indicator_widget` snapshot，但只迁入 Lime 已有 queue 产品语义：TUI 直接读取 App Server `thread/queue/list`，监听 scoped `thread/queue/changed`，并以返回的 canonical `QueuedSubmission` 作为唯一显示事实源。
- 新增独立 `pending_input_preview` render owner；最多展开前两条排队输入、每条最多两行并显示溢出提示，文本、图片、Skill 与 Mention 均从 typed `UserInput` 派生。预览位于 transcript 与 composer 之间，最多占 8 行，窄终端和小窗口不会挤掉最小 transcript/composer 区域。
- 本地 `Tab` queue 成功后先用 response 即时 upsert；add/update/delete/reorder/自动出队及 resume/reconnect 统一由 queue-changed 后重新分页读取校正。读取失败清空可能误导用户的旧预览并显示五语言错误，不创建 TUI 私有队列、数据库或第二套 notification。
- `Alt+Up` 只在 composer/附件区为空，且最后一条 queued submission 能无损还原为至多一个纯文本段和若干 `detail=None` 本地图片时出现；先等待 App Server `thread/queue/delete` 返回 `deleted=true`，再恢复草稿与附件。远程图片、带 `textElements` 文本、多文本段、Skill/Mention 不显示编辑入口，避免以字符串重建破坏 typed input 语义；失败时本地队列和草稿保持不变。
- 未迁入 Codex 的 pending steer/rejected steer、background terminal、hook status、account/rate-limit 等语义：前两者在 Lime 尚无对应已实现交互合同，后者属于 Codex 产品专属或本轮非优先 surface。
- 验证：`cargo test --manifest-path "lime-rs/Cargo.toml" -p tui` 168/168；`cargo clippy --manifest-path "lime-rs/Cargo.toml" -p tui --no-deps -- -D warnings`；`npm run test:rust:related -- <本轮 TUI 路径>`（`tui + cli`）；workspace fmt check；`npm run test:contracts`；`npm run governance:legacy-report`；TUI Gate B/snapshot inventory Vitest 5/5，均通过。
- 使用本轮重建的 `lime` 运行 `npm run smoke:tui-gate-b`，`complete,approval,user-input,interrupt,failure,queue-edit` 六场景与 terminal restore 通过。queue-edit 专用场景证明运行中 Turn 的 `Tab` 排队内容和编辑提示可见，Alt+Up 后 composer 恢复；canonical runtime event log 严格断言同一 Thread、同一 queued submission id 上 `thread/queue/add` 先于 `thread/queue/delete`。尚未取得真实 Windows runner platform evidence。

## 2026-09-04 Active-turn status indicator

- 参考 Codex `status_indicator_widget`，只迁入 Lime current owner 已具备的 active-turn 子集：状态行由 canonical `active_turn_id` 控制可见性，`App` 仅持有不持久化的单调起始时刻；每秒 redraw 不创建第二套 Turn 状态。
- active turn 时在 composer 上方显示本地化 `Working (elapsed • esc to interrupt)`，elapsed 使用 Codex 的秒/分/小时紧凑格式。状态行与 canonical queued preview 使用独立布局槽位；极窄与低高度终端均进行 display-width 截断，不挤掉最小 transcript/composer。
- `Esc` 只在无 pager、审批/request_user_input、model picker 或 slash popup 抢占时映射到既有 `AppAction::Interrupt -> turn/interrupt`；无 active turn 时不退出、不改草稿，`Ctrl+C` 仍保留既有 interrupt/退出行为。未迁入 background terminal、Hook status 或自定义 keymap，它们没有 Lime current 产品合同。
- 固定时间纯 render、TestBackend 宽/窄布局、active/inactive/terminal reducer、popup 优先级、status + queue + composer 排序回归通过。`cargo test -p tui -p cli` 为 TUI 168/168、CLI 15/15，Rust related、clippy、fmt/Prettier/diff check、TUI Gate B/inventory Vitest、contracts 与 governance 均通过。
- 真实 `npm run smoke:tui-gate-b` 六场景通过；interrupt 场景先验证 `esc to interrupt` 可见，再发送 Escape，并由 external backend ledger 证明 `turnCancel` 命中和 terminal restore；新增 queue-edit 场景保持同一 active-turn 状态提示和终端恢复合同。

## 2026-09-04 Composer reverse history search

- 对照 Codex `chat_composer/history_search`，在 Lime 现有 composer 的最近 200 条 App Server prompt history 上增加纯本地 `Ctrl+R`/`Ctrl+S` 反向搜索；不新增 `promptHistory` 方法、数据库或第二套历史事实源。
- 搜索态只持有当前查询、原始草稿和选中索引；进入搜索不替换空查询草稿，输入查询后才显示最近命中，`Ctrl+R`/Up 向旧匹配移动，`Ctrl+S`/Down 向新匹配移动，Esc/Ctrl+C 恢复原草稿，Enter 只接受真实命中。
- App 在搜索态优先把键盘事件交给 composer，避免 active turn 的 Esc 被误判为 `turn/interrupt`；footer 复用 TUI locale 显示 `reverse-i-search` 查询，不向 canonical transcript、prompt history 或 queue 写入本地伪 Item。
- 回归覆盖空查询不替换草稿、大小写不敏感匹配、旧/新匹配、接受/取消、active-turn Esc 优先级、中文 footer；`cargo test -p tui` 当前工作树 172/172 通过。

## 2026-09-04 Codex CLI npm 分发对齐

- 对齐参考从 Rust `codex-rs/cli` 扩展到 `/Users/coso/Documents/dev/rust/codex/codex-cli`。`packages/cli/bin/lime.js` 采用同构 ESM launcher 和平台 alias 解析；生产只执行平台 package 中的原生 `lime`，根包 `vendor` 仅作为 staging/development 的相同布局 fallback，不搜索源码 target、不调用 cargo、不在 postinstall 下载。
- 复制 Codex 可复用的包级骨架：`packages/cli/.gitignore` 隔离本地 `vendor/`，`packages/cli/scripts/README.md` 固化 staging/debug/release 用法，`packageManager` 元数据同时进入根包和平台包。Codex 的 `init_firewall.sh` 与 `run_in_container.sh` 绑定 OpenAI allowlist、Codex Docker image 和 `/etc/codex`，不属于 Lime npm 分发 owner，分类为 `product-specific / not-copied`。
- launcher 继承 stdio，转发 `SIGINT`/`SIGTERM`/`SIGHUP`，镜像子进程 exit code 或 signal，并设置 `LIME_MANAGED_PACKAGE_ROOT` 与 npm/pnpm/bun/Vite+ ownership 标记。对照 Codex 实现额外修复了 signal 重发前未注销 handler 会吞掉父进程 signal的问题，真实 shell fixture 证明退出原因保持为 `SIGTERM`。
- 新 `build_npm_package.py` 生成轻量根 tarball 和同 npm name、带平台版本后缀的平台 tarball；根包 optional dependency 通过 npm alias 指向四个平台版本。平台 staging fail closed 校验 `lime + app-server + code-mode-host`，Windows 额外校验两个 sandbox helper，所有平台校验 sherpa/onnx runtime marker。
- Release workflow 成组构建 `cli`、`app-server`、`code-mode-host` 和 Windows helpers，复制动态库，上传平台 tarball；独立 OIDC job 下载四个平台 tarball、生成根 tarball 并串行执行 platform-first publish。旧四个下载/同步/单二进制 archive 脚本已从 active package.json 与发布链断开并物理删除。

## 2026-09-04 Turn completion canonical repair

- 对齐 Codex `live_app_server_turn_completion_repairs_dropped_message_deltas`：`TurnCompleted` 现在先消费通知内 canonical `turn.items`，按 item id 替换或补回丢失的流式/`item/completed` 投影，再收敛剩余 running entries；不新增协议、数据库、TUI 私有状态或 provider fallback。
- 回归覆盖无任何 delta 时从 canonical turn 恢复最终回答、部分流式文本被 canonical Assistant item 替换为完整终态，以及缺失 UserMessage 插入到已知 Assistant 前保持 canonical 顺序；`cargo test --manifest-path "lime-rs/Cargo.toml" -p tui --lib` 为 177/177，通过 `npm run test:rust:related -- "lime-rs/crates/tui/src/projection.rs" "lime-rs/crates/tui/src/projection_tests.rs"` 的 `cli + tui` 相关测试。
- `cargo clippy --manifest-path "lime-rs/Cargo.toml" -p tui --no-deps -- -D warnings`、`npm run smoke:tui-gate-b`（六场景、queue-edit、terminal restore）和目标文件 `git diff --check` 通过。
- `npm run test:contracts` 未通过：并行工作树现有 `.github/workflows/quality.yml` 缺少 Windows App Server sidecar pinned build 命令；该失败与本切片无关，本轮未修改 workflow，也不将 Windows runner evidence 标记为完成。
- 同一 current MultiAgent projection 继续对齐 Codex `app_server_collab_spawn_completed_renders_requested_model_and_effort`：保留 typed `prompt`、`model`、`reasoning_effort` 为本地只读摘要，并补齐五语言 `model/effort/prompt` detail labels；不引入子线程控制、私有 agent graph 或第二套状态源。
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p tui --lib` 更新为 177/177，覆盖 SpawnAgent 请求上下文与 canonical TurnCompleted 缺失项有序插入；`cargo clippy --manifest-path "lime-rs/Cargo.toml" -p tui --no-deps -- -D warnings` 与 targeted rustfmt/diff check 通过。
- 架构分类：`current = cli/tui/app-server-client + authenticated remote transport foundation + npm launcher/platform staging`；`compat/deprecated = none`；`dead / deleted / forbidden-to-restore = install.js/run.js/release-meta.js/build-release.js`。Cloud production endpoint 仍未进入 npm launcher 或平台载荷。
- 定向证据：`npm --prefix packages/cli test` 8/8；CLI/npm/TUI inventory 与 governance source tests 10/10；`npm run smoke:cli-npm-gate-b` 从真实 Node launcher 和 optional platform package 进入 sibling App Server，canonical Thread/Turn identity、JSON/JSONL、stdin、错误退出码与 completion 全绿；裸 Rust CLI Gate B 与六场景真实 PTY TUI Gate B 同样通过。`npm run test:contracts`、`verify:app-version`、`governance:cli-boundary`、`governance:scripts`、legacy `0/0/0`、Electron release workflow guard、Node/Python syntax、Prettier 与 `git diff --check` 均通过；物理删除后的守卫复验保持全绿。

## 2026-09-04 CLI surface Gate B 与证据账本复验

- `lime-rs/crates/cli/src/queue_cmd.rs` 增加 Codex 形状的 `queue list --thread <THREAD> [--json]`，原有 `queue --thread ... --message ...` 保持 current 入口；队列读写均通过 App Server `thread/queue/list|add`，缺失 Thread 返回非零并 fail closed。
- `plugin search` 的 marketplace 多 cwd 参数改为 `--plugin-cwd`，避免与公共连接参数 `--cwd` 发生 Clap 重复 option；新增解析回归锁定该命名边界。
- `scripts/app-server/cli-surface-gate-b.mjs` 现在以隔离数据目录验证 MCP `list/add/get/remove/start/stop`、features `list/enable/disable`、plugin `add/list/read/search/enable/disable/remove`、debug `models --bundled`/`clear-memories`、queue `add/list` 与不可用 Thread fail-closed、OAuth logout fail-closed；queue fixture 仅使用显式 external backend，不调用正式 Provider或 mock backend。macOS 自动扫描 sherpa 预构建动态库目录并设置 `DYLD_LIBRARY_PATH`。
- `cli-test-inventory.mjs` 与 `cli-test-inventory.test.mjs` 将有真实 CLI/TUI/Gate B 证据的 current exec/MCP 条目标为 `covered`，当前账本统计 `covered=16 / partial=33 / pending=249 / deferred=60 / excluded=75`；MCP rationale 与脚本文档统一使用 current `mcpServer/list`，并将 `mcpServerStatus/list` 明确为独立 GUI/App Server 运行时状态控制面。
- 本轮验证：扩展 `smoke:cli-surface-gate-b` 通过；CLI surface/structure/test inventory Vitest `7/7`；`cargo test -p cli -p app-server-client`（29 + 41）；`npm --prefix packages/cli test` `8/8`；`cargo clippy -p cli --no-deps -- -D warnings`；`npm run test:contracts`（299 client checks + command/modality/script/governance/docs）；`npm run governance:legacy-report`（零引用候选、分类漂移、边界违规均为 0）；Prettier、Node syntax 与 `git diff --check` 全部通过。
- 分类保持：`current = lime-rs/crates/cli + lime-rs/crates/tui + app-server-client + authenticated remote transport foundation + packages/cli + App Server JSON-RPC`；`compat/deprecated = none`；`dead/deleted/forbidden-to-restore = lime-cli、旧 npm 下载/同步脚本、旧 task/media/skill/doctor 命令`。Cloud production endpoint 仍未启动；真实 Windows runner、Windows native `syntect/onig`、external editor/clipboard 原生证据仍未取得，继续 fail closed。

## 2026-09-04 Command live output projection

- 对齐 Codex `live_app_server_command_output_delta_{active,transcript,interrupted}` snapshot：running command 在 terminal projection 中稳定保留命令行与输出行边界，后续 delta chunk 原样拼接，不会因 chunk 在单词中间拆分而插入额外换行。
- `item/completed` 与 `turn/completed` 继续以 canonical `CommandExecution.aggregated_output` 替换实时文本并收敛终态；没有新增协议、TUI 私有 command 状态、第二套工具生命周期或 production mock fallback。超大实时输出 head/tail 截断与 background terminal interaction 未在本刀引入。
- Unit 回归覆盖 `item/started -> 两段 command/outputDelta` 及 `item/completed` canonical 覆盖；TestBackend 证明 `$ command [running]`、stdout、stderr 分行进入实际 Ratatui 投影。`cargo test -p tui` 与 related `cli + tui` 均为 TUI 180/180、CLI 4/4，通过 targeted rustfmt、`cargo clippy -p tui --no-deps -- -D warnings` 和 `git diff --check`。
- `npm run test:contracts` 完整通过；使用本轮重建的 `lime` 运行默认六场景 `npm run smoke:tui-gate-b`，真实 PTY、App Server、canonical item、queue-edit 与 terminal restore 通过。当前 Gate B fixture 由并行工作流维护且只发 command started/completed，没有发 `tool.output.delta`，因此实时 delta 的直接证据等级仍是 Unit + TestBackend，Gate B 只作为 current 主链无回归证据。
- 分类保持：`current = tui::projection -> entry/view + App Server typed command notifications`；`compat/deprecated = none`；`dead = none`。下一刀应先审计 Codex 大输出 head/tail 截断的资源上限与 Lime transcript/scrollback 边界，再决定是否迁入；不要借此引入 Codex background terminal 产品逻辑。

## 2026-09-05 Command output display bounds

- 对照 Codex `LiveCommandOutput` 的可复用资源护栏，在 Lime `tui::entry` display owner 增加命令输出的 head/tail 窗口：保留前后各 50 行，超出部分显示本地化省略标记；单行超过 16 KiB 时保留 UTF-8 安全的头尾并显示省略字节数。
- 截断只发生在 terminal render boundary。`ConversationProjection` 仍保留实时投影，`ItemCompleted`/`TurnCompleted` 仍以 canonical `aggregated_output` 覆盖；`final_answer`、`/copy`、`Ctrl+T` 的数据语义不变，不新增协议字段、TUI 私有 command 生命周期、第二套 output store 或 background terminal。
- `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR` 均补齐行/字节省略文案；entry unit、UTF-8 长行和 TestBackend 可见性回归覆盖窗口边界与 marker。`cargo test -p tui`：185/185；`cargo fmt --check`、`cargo clippy -p tui --no-deps -- -D warnings`、`git diff --check` 通过。
- 分类：`current = tui::entry -> view -> ConversationProjection`；`compat/deprecated = none`；`dead = none`。Codex 独有的独立 preview/transcript 双 owner、background terminal、ANSI 状态恢复仍为 `deferred / product-specific`，待协议提供明确 metadata 和 Lime 产品合同后再评估。

## 2026-09-05 Dynamic permission profile catalog

- TUI 启动、恢复和 bounded reconnect 统一通过既有 `permissionProfile/list` 查询当前 cwd 的 profile catalog；只把 `allowed=true` 的条目交给快捷切换，避免在 TUI 复制 App Server 的 sandbox/backend 判断。
- `AppServerSession` 保存 `thread/start` / `thread/resume` 返回的 `activePermissionProfile.id`，恢复线程且未显式传入 `--permissions` 时沿用服务端 active profile；malformed 或空 id 不进入本地状态。
- F7/F8 与对应快捷键现在消费服务端 catalog，支持 configured named profile；空/非法目录保持内建 profile fallback，设置更新仍统一走 `thread/settings/update` 并由 App Server 校验。
- 未迁入 Codex 的 project-local config merge、managed requirements、sandbox-state replay 和 network proxy：这些能力尚无 Lime current owner 或属于 Cloud/平台专属，继续 `deferred / fail-closed`，不在 CLI/TUI 伪造第二套策略。`execpolicy check` 的只读规则报告已归 `cli::execpolicy` current owner，但它不参与实际执行 lowering。
- 新增 dynamic catalog 去重/空值过滤、cycle 和 active profile id 回归；`cargo test -p tui`：187/187，`cargo clippy -p tui --no-deps -- -D warnings`、workspace rustfmt 和 `git diff --check` 通过。
- 重连后仅在未显式指定 `--permissions` 时回填新会话的 `activePermissionProfile.id`；空 catalog 会清空旧缓存并回退到 `:read-only` / `:workspace` / `:danger-full-access` 内建目录，避免跨 workspace 保留失效 named profile。
- TUI Gate B `complete` 场景现在注入临时 YAML `default_permissions: named-fixture`，由真实 App Server `permissionProfile/list` / `thread/start` 解析，PTY `/status` 断言 named profile 可见；configured named profile fixture evidence 已补齐。
- 2026-09-05 复验时先发现 Gate B 复用了旧 `app-server` 产物；使用本机缓存的 `rusty_v8` archive/binding 重建 `cli` 与 `app-server` 后，默认六场景 `npm run smoke:tui-gate-b` 全部通过（含 `complete` 的 named profile `/status` 可见性和 `queue-edit` canonical add/delete 证据）。
- 分类：`current = app-server permissionProfile/list -> tui::AppServerSession -> App::permission_profiles -> shortcut lowering`；`compat/deprecated = none`；`dead = none`。

## 2026-09-05 Inherited stdio sandbox execution

- `lime sandbox` 在 stdin/stdout 都是终端时复用 App Server `command/exec` 的既有 streaming 合同：生成客户端 `processId`，开启 `tty`、`streamStdin`、`streamStdoutStderr`，将 `command/exec/outputDelta` 按 stdout/stderr 原样转发，并通过 `command/exec/write` 传递 stdin 与 EOF。
- 非 TTY（管道、脚本、重定向）继续使用原有一次性 `CommandExecResponse` 捕获路径；没有新进程 owner、后台 terminal、CLI 私有 PTY 或第二套 sandbox backend。
- inherited stdio 收到未知 server request 时统一返回 `METHOD_NOT_FOUND`；App Server 断线、非法 base64、stdin/write 失败和 output stream 异常均 fail closed，并在结束时走既有 session shutdown。
- 新增 CLI 单元回归锁定 streaming 参数 lowering；`cargo test -p cli`：5 个 library + 37 个 binary 全部通过，`cargo clippy -p cli --no-deps -- -D warnings`、CLI rustfmt 和 `git diff --check` 通过。
- 输出 delta 解码抽为 binary-safe helper，并补非法 base64 fail-closed 回归；重连权限状态回填与本轮 CLI/TUI 相关门禁一起验证。
- `sandbox-state` replay、managed network、macOS denial logger/PID tracker、Windows managed-user setup 仍为 `deferred / fail-closed`：缺少 Lime App Server current owner 或属于 Cloud/平台专属，不从 Codex 直接复制。`execpolicy check` 只读命令规则并报告结果，已在 2026-09-05 独立切片接入 CLI。
- 分类：`current = cli::debug_sandbox -> app-server-client -> App Server command/exec -> tool-runtime execution_process`；`compat/deprecated = none`；`dead = none`。真实跨平台 inherited-stdio Gate B 尚未取得，继续记录为待 Windows/macOS 平台证据。

## 2026-09-05 Command Exec Permission Profile Lowering

- `app-server-protocol::CommandExecParams` 与 Codex exact wire 对齐：只暴露 `permissionProfile`，不暴露
  `grantedPermissions` 客户端字段；已有生成 schema/fixture 保持无该字段。
- `processor::command_exec` 在 JSON-RPC ingress 统一解析 YAML `default_permissions`、显式 builtin/named profile、
  inheritance 和 cwd/platform readiness；显式 `sandboxPolicy` 时不套用 default profile。客户端直接传入
  `grantedPermissions` 被 `INVALID_PARAMS` 拒绝，防止绕过 App Server 权限 owner。
- `CommandExecServer` 接收服务端派生的 `GrantedPermissionProfile`，只在 lowering 到 `tool-runtime::LocalExecutionSandbox`
  时传递；TUI、CLI、Desktop 不创建第二套 grants 或 sandbox 状态。
- `cli::debug_sandbox` 未显式指定 profile 时不再硬编码 `:workspace`，让 App Server 配置 default profile 生效；新增
  Codex 同名 active/named profile parser 回归并将 CLI inventory `missing` 从 15 降到 13。
- 定向验证：`cargo test -p app-server --lib permission_profile`（14/14）、`cargo test -p app-server --lib command_exec`
  （10/10）、`cargo test -p app-server-protocol`（133 + schema fixture 1）、`cargo test -p cli`（8 library + 37 binary）、
  app-server `cargo check`（使用已校验本地 rusty-v8 archive/binding）、`git diff --check` 全部通过。
- 架构文档同步：`internal/aiprompts/architecture.md` 与 `internal/aiprompts/commands.md` 明确权限 profile 的唯一
  App Server resolver、服务端 grants lowering、CLI/TUI/命令执行共用 owner，以及 Cloud profile 仅可未来经
  authenticated remote transport 提供。
- 分类：`current = app-server permission profile resolver + CommandExecServer lowering + cli/tui/app-server-client`；
  `compat/deprecated = none`；`dead/deleted/forbidden-to-restore = 客户端 grantedPermissions 注入路径`。Cloud
  managed requirements、sandbox-state replay、managed network 和跨平台 native evidence 继续 deferred/fail-closed；本地 `execpolicy check` 仅承担规则报告，不冒充 runtime execpolicy enforcement。

## 2026-09-05 Remote protocol version gate

- `app-server-client::ClientSession` 在 `initialize` 响应和 `initialized` 通知之间严格校验
  `serverInfo.protocolVersion == app_server_protocol::PROTOCOL_VERSION`；不支持的版本返回结构化
  `UnsupportedProtocolVersion`，并关闭 transport，不进入半初始化会话。
- 该校验同时覆盖 stdio 与 WebSocket remote，保持同一 session facade，不新增 Cloud 专用握手字段或第二套
  runtime。Remote transport 现有 Bearer/TLS/loopback 规则继续有效。
- 新增回归证明不支持版本不会发送 `initialized` 且会关闭 transport；远程 fixture 改用协议事实源常量，避免
  测试伪造版本。
- 验证：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-client -p cli -p tui`
  （46 + 8 + 37 + 187）；`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check` 通过。
- 分类：协议版本拒绝为 `current`；identity pinning、tenant isolation、凭证持久化、跨网络 resume、rate
  limit、audit 和真实 Cloud endpoint 仍为 `deferred / fail-closed`，待服务端合同与真实远端 evidence。

## 2026-09-05 Execpolicy check surface

- `lime execpolicy check` 采用 Codex 同名 `ExecpolicyCommand`、`ExecpolicySubcommand`、
  `ExecPolicyCheckCommand`、`PolicyParser`、`Policy`、`RuleMatch` 与 `run_execpolicycheck` 结构；支持重复
  `--rules`、prefix rule、`allow`/`prompt`/`forbidden` 严格度、可选 `justification`、basename resolution 和
  compact/pretty JSON 输出。该命令只读取规则并报告匹配，不执行命令、不持有 sandbox 或 approval owner。
- 新增 Codex 同名 unit 回归 `execpolicy_check_matches_expected_json` 与
  `execpolicy_check_includes_justification_when_present`，并补充 alternatives/host executable matching
  回归；真实 `execpolicy check` 已加入 `smoke:cli-surface-gate-b`；CLI binary tests `41/41` 通过，inventory 收敛为 `missing=0`。Codex Cloud
  managed profile、macOS PID tracker、hosted code-mode-host URL 和 network-proxy 平台集成已明确分类为
  deferred/excluded，继续 fail closed。
- 分类：`current = cli::execpolicy`；`compat/deprecated = none`；`dead = none`。执行权限与 runtime
  lowering 仍归 App Server/tool-runtime，Cloud managed execpolicy 不进入本地 CLI。

### 本刀验证记录

- `cargo test --manifest-path "lime-rs/Cargo.toml" -p cli`：library 8/8、binary 41/41、doc-tests 通过；
  `cargo clippy -p cli --tests --no-deps -- -D warnings` 与 CLI rustfmt 通过。
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-client -p tui -p cli`：47 + 187 +
  8 + 41 全部通过；`npm run test:contracts`、`npm run governance:legacy-report`、
  `npm run governance:cli-boundary`、`npm run smoke:cli-surface-gate-b` 与 CLI inventory/structure
  Vitest 7/7 通过。
- App Server `command_exec` 定向测试已写入 `processor/tests/command_exec.rs`，但重新编译时被并行工作树
  未完成的 `code-mode-runtime` 热区阻塞（`session_runtime::ToolKind` 导出和 `NestedToolCall.tool_kind`
  缺失）；该热区不属于本刀写集，不能将该测试标为已验证。

## 2026-09-05 Remote credential boundary

- `RemoteTransportConfig` 改用自定义 `Debug`，Bearer token 只显示 `<redacted>`；新增回归防止 token
  原文进入日志/诊断。
- remote 连接在 transport 建立前拒绝 URL userinfo、fragment、敏感 credential query 和空/全空白 auth
  token；token 继续只从 CLI 指定的环境变量进入 `Authorization` 请求头，不写入 URL、协议参数或持久化
  状态。配置 Debug 会对 URL 中潜在的敏感值做脱敏。
- 这些是 remote foundation 的凭证卫生和端点边界，不代表已完成 Cloud credential store、tenant
  isolation、token rotation、跨网络 resume 或 audit；后续仍需真实服务端合同与 evidence。
- 验证：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-client`（49/49）、
  `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check`、`cargo clippy --manifest-path
  "lime-rs/Cargo.toml" -p app-server-client --no-deps -- -D warnings`、`npm run test:contracts`、
  `npm run docs:boundary` 和 `git diff --check` 通过。联合 CLI/TUI Clippy 被并行 `execpolicy`
  热区的既有告警阻断，未归因于本刀。

## 2026-09-05 Codex owner 目录继续收敛

- 新增独立 `lime-rs/crates/execpolicy` workspace crate，对齐 Codex `codex-rs/execpolicy` 的目录骨架：
  `amend.rs`、`decision.rs`、`error.rs`、`execpolicycheck.rs`、`executable_name.rs`、`parser.rs`、
  `policy.rs`、`rule.rs`、`sandbox_migration.rs`、`tests/basic.rs`。当前 crate 提供 prefix/network 规则
  解析、strictest decision、host executable gating、规则写入和只读 migration 接口；实际执行权限仍由
  App Server/tool-runtime lowering 负责。
- `scripts/app-server/cli-structure-inventory.mjs` 已纳入 `codex-rs/execpolicy` 与
  `lime-rs/crates/execpolicy` 双向对照；该 crate 的文件差异为 0，Codex 专属 Starlark 完整实现继续由
  `execpolicySymbolNamesMissingInLime` 账本显式记录，不能宣称已完整复制。
- TUI 已按 Codex 真实模块树纠正根文件形状：`markdown_render.rs`、`status_indicator_widget.rs`、
  `resume_picker.rs`、`pager_overlay.rs` 均保留为根模块文件；`bottom_pane/chat_composer.rs` 与
  `bottom_pane/request_user_input/mod.rs` 对齐 Codex 的 bottom-pane owner。`Composer` 类型已改名为
  `ChatComposer`，通用 bottom-pane 渲染只保留审批职责，request-user-input 渲染下沉到其子模块；旧
  `composer.rs`、`bottom_pane/request_user_input.rs` 路径已删除，不保留兼容包装。
  新增 `tui-structure-inventory.mjs` 与测试，锁定 TUI 目录/符号双向差异和产品专属排除。
- 验证：`cargo test -p execpolicy`（34 个 crate 测试通过）、`cargo test -p tui -p cli`（188 + 49 通过）、
  `cargo clippy -p execpolicy --tests --no-deps -- -D warnings`、CLI/TUI 结构与测试 inventory Vitest 通过。
- npm launcher 内部符号继续采用 Codex 名称：`codexPackageRoot`、`findCodexExecutable`、
  `isPnpmOwnedCodexInstall`、`isVitePlusOwnedCodexInstall`；Lime 包名、平台 alias、环境变量和原生
  payload 名称保持 Lime current 合同。`run_command` 与 `stage_codex_sdk_sources` 属于 Codex SDK 专属
  staging，结构账本继续显式保留为 excluded，不引入 SDK 或第二分发 owner。`node --test
  packages/cli/tests/npm-package.test.mjs` 8/8 通过。

## 2026-09-05 CLI execpolicy owner 收敛

- CLI 子命令按 Codex 真实 owner 形状收回 `cli/src/main.rs`：`ExecpolicyCommand`、
  `ExecpolicySubcommand` 和 `run_execpolicycheck`；规则模型、解析器和 JSON 命令参数只由
  `lime-rs/crates/execpolicy` 承接。重复的 `lime-rs/crates/cli/src/execpolicy.rs` 已移除，
  不保留 compat 包装或第二套 parser。
- 新增同名黑盒回归 `lime-rs/crates/cli/tests/execpolicy.rs`，通过 `CARGO_BIN_EXE_lime`
  验证 `execpolicy check` 的 forbidden decision、matched prefix 和 justification JSON；
  测试使用隔离临时 `CODEX_HOME`，不依赖用户目录或 provider。
- 验证：`cargo test --manifest-path "lime-rs/Cargo.toml" -p cli --test execpolicy`（2/2）、
  `cargo test --manifest-path "lime-rs/Cargo.toml" -p cli -p execpolicy`（38 + 4 + 17）、
  `npm run smoke:cli-gate-b`、`npm run smoke:tui-gate-b`、`npm run test:contracts`、
  `npm run governance:legacy-report`、`node --test "packages/cli/tests/npm-package.test.mjs"`
  均通过。
- 分类：`current = cli/main.rs + execpolicy crate + cli/tests/execpolicy.rs`；
  `compat/deprecated = none`；`dead = cli/src/execpolicy.rs`（已删除）。Codex execpolicy 的
  Starlark parser、prefix/alternative/match/not_match、strictest decision、host executable、network
  rule、error location、overlay、migration、amend 与完整 `tests/basic.rs` 已迁入；路径值继续使用
  Lime 现有 `PathBuf` 类型 owner，不引入 Codex 外部 workspace crate。`cli-structure-inventory`
  当前确认 `execpolicyFilesMissingInLime=[]`、`execpolicySymbolNamesMissingInLime=[]`。

## 2026-09-05 TUI owner path correction

- 对照 Codex 当前 checkout 后纠正了四个误判的模块路径：Codex 使用根文件
  `markdown_render.rs`、`resume_picker.rs`、`pager_overlay.rs`、`status_indicator_widget.rs`，
  Lime 已恢复为相同形状；结构守卫不再要求错误的 `*/mod.rs` 路径。
- `bottom_pane/request_user_input` 迁移为 Codex 同名目录模块，并新增 `render.rs`，将问题渲染与
  `bottom_pane/render.rs` 的 approval 渲染职责分离。`bottom_pane/chat_composer.rs` 现为 composer
  唯一 owner，类型名为 `ChatComposer`；终端生命周期 owner 已从 Lime-only `terminal.rs` 迁到
  Codex 同名 `tui.rs`，`TerminalGuard` 仍保留其真实生命周期职责。
- 验证：`cargo test --manifest-path "lime-rs/Cargo.toml" -p tui`（188/188）、
  `cargo fmt --manifest-path "lime-rs/Cargo.toml" --all`、
  `npx vitest run "scripts/app-server/tui-structure-inventory.test.mjs" "scripts/app-server/tui-snapshot-inventory.test.mjs"`
  （5/5）通过；结构账本已重生成，所选 Codex 根模块和 bottom-pane 路径均不在
  `filesMissingInLime`。
- 分类：上述模块和渲染边界为 `current`；错误的 `*/mod.rs` 迁移路径与旧 `composer.rs`、
  `bottom_pane/request_user_input.rs`、`terminal.rs` 为 `dead / deleted / forbidden-to-restore`；未承接的
  Codex `app/`、`tui/`、account/onboarding/update 等产品或大规模 runtime 子模块仍按账本
  `merge/contract/defer/dead` 分类，不因文件名差异直接复制。

## 2026-09-05 TUI event stream and frame scheduling alignment

- 按 Codex `tui` 目录复制并接入 `tui/event_stream.rs`、`tui/frame_rate_limiter.rs` 与
  `tui/frame_requester.rs`。`EventBroker` 是唯一 crossterm 输入 owner，支持暂停/恢复并在
  `TuiEventStream` 中统一 key、paste、resize、focus 与 draw；`FrameRequester` 合并异步重绘
  并限制最高 120 FPS。
- `TerminalGuard` 现在持有 broker、draw channel 和 frame requester；外部编辑器 suspend/resume
  会释放并恢复 stdin。runtime 与 session picker 均消费 `TuiEventStream`，不再各自直接创建
  `EventStream`，业务事件仍由 App Server session 与 `App` 处理。
- 新增 Codex 同名事件流/帧调度回归 21 项，`cargo test -p tui` 当前为 214/214；结构账本刷新为
  624 文件，TUI/CLI structure、snapshot、test inventory 11/11，`cargo clippy -p tui -p cli
  --no-deps -- -D warnings` 与 workspace fmt 通过。
- 分类：`current = tui::event_stream + tui::frame_requester + TerminalGuard`；
  `compat/deprecated = none`；Codex job-control、inline scrollback、keyboard enhancement 与
  platform probe 仍按账本 `defer/product-specific`，本轮不伪造第二套 runtime 或平台 owner。

## 2026-09-05 PTY editor handoff recovery

- 外部编辑器返回后的 `TerminalGuard::resume` 只重新启用 raw/alternate/bracketed-paste 模式并恢复
  `EventBroker`，不再调用会同步发送 `CSI 6n` 的 Ratatui 清屏路径；下一轮正常 draw 负责重绘恢复后的
  composer。这样不会让 TUI 的事件 reader 与编辑器/PTY 的光标响应竞争 stdin。
- 删除 Gate B 中针对旧同步光标查询的 PTY 回写步骤，改为直接断言编辑器交接标记和恢复后的 prompt；
  保留真实 alternate-screen、stdio 继承和最终终端恢复断言。
- 验证：`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check`、
  `cargo test --manifest-path "lime-rs/Cargo.toml" -p tui`（214/214）、
  `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-client -p cli`（49 + 49）、
  `cargo clippy --manifest-path "lime-rs/Cargo.toml" -p tui -p cli --no-deps -- -D warnings`、
  `npm run smoke:tui-gate-b`（complete、approval、user-input、interrupt、failure、queue-edit 六场景）
  全部通过。
- 分类：`current = tui::tui::TerminalGuard + tui::runtime_pty_tests`；
  `compat/deprecated = none`；旧同步 CPR 查询与调试输出为 `dead / deleted / forbidden-to-restore`。

## 2026-09-05 Codex text formatting owner

- 按 Codex `tui/src/text_formatting.rs` 迁入终端文本格式化 owner：`capitalize_first`、
  `format_and_truncate_tool_result`、`format_json_compact`、`truncate_text`、
  `center_truncate_path` 与 `proper_join`，保留 Codex 同名函数和 Unicode grapheme 安全语义。
- 会话列表使用 `center_truncate_path` 先压缩 cwd，再按当前 viewport 合成行，窄终端优先保留路径
  前后段；不会改变 Thread/Turn/Item 或 App Server 协议内容。
- `serde_json` 在 TUI owner 启用 `preserve_order`，确保 JSON 摘要的字段顺序与 canonical 输入一致；
  不改变 workspace 其他 crate 的默认解析行为。
- 新增 21 个同名格式化回归，覆盖 JSON、路径、emoji、组合字符、截断边界和自然语言连接；
  `cargo test -p tui` 当前 235/235、`cargo clippy -p tui --no-deps -- -D warnings`、workspace
  fmt、TUI structure/snapshot inventory 5/5 通过。
- 分类：上述模块与会话 cwd 投影为 `current`；`wrapping`、`terminal_palette`、`table_detect` 在
  下一刀按纯终端算法 owner 迁入，不创建无 Lime owner 的兼容路径。

## 2026-09-05 Codex pure terminal algorithm owners

- 按 Codex 原名新增 `tui/src/wrapping.rs`、`tui/src/terminal_palette.rs`、`tui/src/table_detect.rs`。
  `wrapping` 复用 `textwrap 0.16` 的 `RtOptions`、标准/URL-aware wrapping、styled span 切片、UTF-8
  range 和多行 owned 输出；URL 识别只在 terminal render boundary 生效。
- `terminal_palette` 提供 `StdoutColorLevel`、`best_color`、ANSI256 nearest-color 量化和 Codex
  同名 default-color/probe hook；Lime 尚无 terminal probe owner 时 default colors fail closed，不伪造
  终端响应。`table_detect` 提供 escaped-pipe、GFM delimiter、blockquote/fenced-code tracker。
- `terminal_hyperlinks::wrap_hyperlink_line` 使用新 wrapping owner：显式 hyperlink 采用可分割的标准
  wrapping 以保持每个 fragment 的 OSC 8 destination，普通 URL 行使用 URL-aware wrapping。
- 新增纯算法单测与结构守卫，锁定 `RtOptions`、`adaptive_wrap_line(s)`、`word_wrap_line(s)`、
  `wrap_ranges(_trim)`、`url_preserving_wrap_options`、`FenceTracker`、`StdoutColorLevel` 等 Codex
  同名符号；不引入 Codex runtime、state DB、terminal probe 或 Cloud endpoint。
- 验证：`cargo test --manifest-path "lime-rs/Cargo.toml" -p tui`（249/249）、
  `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-client -p cli`（49 + 8 + 40 + 2
  black-box）、`cargo clippy -p tui -p app-server-client -p cli --no-deps -- -D warnings`、workspace
  fmt、TUI structure/snapshot Vitest（8/8）、`npm run smoke:tui-gate-b`（六场景）和
  `npm run test:contracts` 均通过。

## 2026-09-05 Codex resume picker owner continuation

- `session_picker.rs` 已迁入 Codex 同名根 owner `resume_picker.rs`，`PickerState`、
  `run_session_picker_with_app_server` 与 `run_resume_picker_with_app_server` 统一位于该模块；
  `lib.rs` 不再注册 Lime-only `session_picker`，旧文件已删除，不保留兼容转发。
- 新增 `resume_picker/page_loading.rs`，按 Codex `PaginationState`、`PageLoadMode`、
  `PendingLoad` 和 `PageCursor` 组织 cursor/request-token 状态；picker 通过真实 App Server
  `thread/list` 分页接口加载，拒绝重复 cursor 与超限请求。
- 新增根级 `resume_picker_transcript_preview.rs`，通过 `thread/read(includeTurns=true)` 读取
  canonical Thread/Turn/Item，仅保留最近六行用户与助手文本；不读取 rollout 文件或第二本地历史库。
- 在 `resume_picker.rs` 保留 Codex 同名 `SessionTarget`、`SessionSelection`、
  `SessionPickerAction` 与 `SessionPickerLaunchContext` 类型，作为 Desktop/TUI/未来
  authenticated Cloud transport 共用的选择结果骨架；当前 fork/agents/cloud runtime 仍不在生产路径。
- 新增 `resume_picker/archive.rs` 的 `ArchiveState` 与 `Ctrl+A` 归档动作，复用
  `thread/archive` JSON-RPC；重复请求去重，失败保留选中会话，成功从列表移除并修正选中索引。
- 继续按 Codex 同名 `archive_tests.rs` 与 `resume_picker_transcript_preview_tests.rs` 拆分测试；
  picker 现在通过 `thread/unarchive` 支持归档会话恢复，并通过 `thread/fork` 暴露真实分叉入口，
  同时接入当前目录/全部目录筛选、Active/Archived 状态、Created/Updated 排序、搜索和紧凑/舒适密度。
  所有动作均经 App Server JSON-RPC，未创建本地历史或 mock owner。
- 验证：`cargo test --manifest-path "lime-rs/Cargo.toml" -p tui --lib`（263/263）、
  `cargo clippy --manifest-path "lime-rs/Cargo.toml" -p tui --no-deps -- -D warnings`、
  workspace fmt、TUI structure/snapshot Vitest（5/5）和 `git diff --check` 通过。
- 分类：`current = resume_picker.rs + resume_picker/{archive,page_loading}.rs +
  resume_picker_transcript_preview.rs + resume_picker/archive_tests.rs +
  resume_picker_transcript_preview_tests.rs`；`dead/deleted/forbidden-to-restore = session_picker.rs`；
  Codex 的本地 state DB、归档恢复后台 loader、远端跨网络 resume 和完整密集表格 UI 仍为
  `deferred`，不能以本地伪实现替代 App Server owner。
