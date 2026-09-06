# Scripts 目录治理

`scripts/` 根目录当前是历史入口区，不再作为新增脚本的默认落点。npm scripts、GitHub Actions、文档和测试已经大量直接引用根目录脚本，物理迁移必须分批做；在迁移完成前，根目录和一级领域目录都用冻结基线守住，不允许继续无序变大。

## 当前分类

- `current`：`scripts/lib/` 中的共享实现、被 `package.json` / CI 明确引用的根入口脚本、与守卫绑定的测试脚本
- `compat`：仍在根目录但需要长期按领域迁移的历史入口脚本
- `deprecated`：只服务旧迁移、旧发布或旧宿主证据的脚本，后续只能下线或并入 current 入口
- `dead`：已删除或只允许作为 fail-fast fixture 出现的旧脚本 / 旧产物路径

## 新增规则

1. 新增可执行脚本默认不得放在 `scripts/` 根目录。
2. 领域脚本放到已有 `scripts/<domain>/`；共享库放到 `scripts/lib/`；属于某个 package 的脚本优先放回对应 package。
3. 根目录只允许保留历史入口、`README.md`、`script-root-governance-baseline.json` 和 `check-scripts-governance.mjs` 这类目录治理文件。
4. 新增一级领域目录必须先说明 owner / 使用入口 / 退出条件，并同步本 README、基线和执行计划；不能为了一个临时脚本新增目录。
5. 每新增脚本都要有稳定调用入口：优先通过 `package.json`、测试、CI workflow 或对应文档引用，不保留孤立手动脚本。
6. 跨平台脚本优先使用 Node / TypeScript；Shell、PowerShell、Python 只在目标平台或现有工具链明确需要时使用，并在入口文档说明平台边界。
7. 新脚本命名使用领域名，不使用 `Lime` / `lime_` / `lime-` 品牌前缀，除非对外资产名或第三方生态已经固定。

## 根目录冻结守卫

根目录和一级领域目录允许列表在：

```text
scripts/script-root-governance-baseline.json
```

检查入口：

```bash
npm run governance:scripts
```

该检查会：

- 拒绝新增的已纳入 git 跟踪的 `scripts/*` 根文件
- 拒绝新增的已纳入 git 跟踪的 `scripts/<new-domain>/**` 一级目录
- 对未跟踪的 `scripts/*` 根文件输出本地警告，避免并行工作区误挡；这些文件不得直接写入基线
- 对未跟踪的一级目录输出本地警告；`scripts/__pycache__/` 这类已忽略本地缓存只提示，不得提交
- 对任意 `scripts/**/__pycache__` 或 `*.pyc` Python 缓存文件输出本地提示；如果这类文件被 git 跟踪则直接失败
- 输出当前根目录脚本数量、领域桶统计、一级目录文件数和扩展名分布
- 提示已经迁走但仍留在基线里的文件或目录，便于后续缩小基线

如果确实需要新增根入口或一级领域目录，必须满足三个条件：

- 它是公开稳定入口，而不是一次性工具
- 不能放入已有 `scripts/<domain>/`、`scripts/lib/` 或 package 内；新增领域目录必须代表可长期维护的边界
- 同步更新本 README、执行计划和基线，并说明退出条件

## 迁移顺序

后续迁移按低风险分批：

1. 先迁零引用或仅测试引用脚本
2. 再迁单一领域且只由 `package.json` 引用的脚本
3. 最后迁 release、Electron、harness 这类 CI / 文档 / 测试多侧引用脚本；i18n 主批已迁入 `scripts/i18n/`

每迁一批都要同步：

- `package.json`
- `.github/workflows/*`
- 相关测试 / 文档 / 守卫
- `scripts/script-root-governance-baseline.json`
- 至少运行 `npm run governance:scripts` 和受影响定向测试

## 现有专题说明

### 根目录当前例外

以下根入口已被 `package.json` 明确引用，当前按 `current` 例外纳入冻结基线；后续迁移时优先进入对应领域目录，并同步缩小 `scripts/script-root-governance-baseline.json`：

- `scripts/check-file-size-governance.mjs`：文件体量治理入口，后续可迁到 `scripts/governance/`
- `scripts/check-import-boundaries.mjs`：导入边界治理入口，后续可迁到 `scripts/governance/`
- `scripts/generate-protocol-types.mjs`：App Server 协议类型生成入口，后续可迁到 `scripts/app-server/`

### Governance 脚本

文件体量棘轮的检查入口仍是历史根脚本 `scripts/check-file-size-governance.mjs`，对外使用：

```bash
npm run governance:file-size
```

基线刷新入口位于 `scripts/governance/update-file-size-baseline.mjs`，只在 R-60 维护或拆分收口后手动执行：

```bash
npm run governance:file-size:update
```

Codex Desktop 外部桌面参考边界由 `scripts/governance/desktop-reference-boundary.mjs` 守住，对外使用：

```bash
npm run governance:desktop-reference-boundary
```

该入口扫描生产源码、JavaScript/Cargo manifest 与版本化 evidence index，拒绝 Goose/ACP 平行 transport、Session/Message owner、Recipe runtime/storage、Autonomous default、第二 runtime/catalog 命名和依赖回流。通用 `session` / `recipe` 业务字段不在粗粒度禁用范围内；守卫只拒绝能明确证明平行 owner 的结构信号。索引同时固定 Codex 目标依据、Goose Apache-2.0 参考版本、`codeCopied=false` 与 `dependencyAdded=false`，并已接入 `npm run test:contracts`。

该脚本会重扫非测试、非生成的前端 / Rust 源文件，更新 `governance/file-size-baseline.json`，不进入 CI 自动链路。

### i18n 脚本

i18n workflow、report、benchmark、检测脚本和测试已整体迁到 `scripts/i18n/`。对外仍优先使用 `package.json` 里的 `detect-translations` 与 `i18n:*` npm scripts，不直接依赖根目录脚本路径。

历史 Python 翻译辅助脚本也位于 `scripts/i18n/`：

- `scripts/i18n/extract_remaining_todos.py`
- `scripts/i18n/import_translations.py`
- `scripts/i18n/translate_all.py`

新增 i18n 脚本继续进入 `scripts/i18n/` 或复用现有 `i18n:*` npm scripts。

### Knowledge 脚本

Knowledge release scope 审计入口已迁到 `scripts/knowledge/`。对外继续使用 `package.json` 里的 `knowledge:*` npm scripts，不直接依赖根目录脚本路径。

新增 Knowledge 脚本继续进入 `scripts/knowledge/` 或复用现有 `knowledge:*` npm scripts；共享实现仍放在 `scripts/lib/`。

### App Server 脚本

App Server release manifest 与 sidecar smoke 脚本已迁到 `scripts/app-server/`。对外继续使用 `package.json` 里的 `app-server:*` 与 `smoke:app-server-*` npm scripts，不直接依赖根目录脚本路径。所有会创建 session/turn 的 stdio、external 与 packaged smoke 必须把 `dataDir` 指向本轮临时目录；固定 fixture identity 不得写入真实用户 App Server data root。

`npm run smoke:cli-gate-b` 使用真实 `lime exec`、真实 App Server stdio 进程和测试专用 external backend，核对同一 canonical Thread/Turn identity、Item 事件序列、JSON/JSONL、pipe stdin、失败退出码与 shell completion；它不调用正式 Provider，也不允许 mock backend 或固定 timer 合成完成态。交互式 alternate-screen/PTY 证据归独立的 TUI Gate B，不用该 CLI smoke 冒充。

`npm run smoke:tui-gate-b` 使用 `portable-pty` 启动真实 `lime tui` 与真实 App Server stdio 进程，在可见 ready 状态后输入 prompt，等待 canonical `turn.completed` 投影出的完成文本，再通过 Ctrl-C 退出；`complete` 场景还通过 Ctrl-G 启动继承前台 PTY 的 external editor，验证草稿回写、标准 DSR 恢复与 alternate screen 重新进入。该测试不调用正式 Provider，也不使用 mock backend。

`npm run inventory:tui-codex` 从 `CODEX_TUI_REFERENCE`（默认本机 `/Users/coso/Documents/dev/rust/codex/codex-rs/tui`）读取上游 snapshot，刷新 `internal/exec-plans/tui-codex-snapshot-inventory.json`。账本为每个 snapshot 保存相对路径、SHA-256 和 `direct/merge/contract/defer/dead` 分类；CI 只校验已提交账本，不要求存在外部 Codex checkout。

`npm run inventory:cli-codex` 从 `CODEX_CLI_REFERENCE`（默认本机 `/Users/coso/Documents/dev/rust/codex/codex-rs/cli`）读取全部 Rust unit/integration test，刷新 `internal/exec-plans/cli-codex-test-inventory.json`。账本逐测试保存相对路径、名称、行号、源文件 SHA-256、迁移分类、状态与 Lime owner；CI 只校验已提交账本，不要求存在外部 Codex checkout。

`npm run inventory:cli-structure` 同时读取 `/Users/coso/Documents/dev/rust/codex/codex-rs/cli` 与 `/Users/coso/Documents/dev/rust/codex/codex-cli`，并与 `lime-rs/crates/cli`、`packages/cli` 做目录、模块、类型、函数和脚本符号对照，刷新 `internal/exec-plans/cli-structure-inventory.json`。账本明确记录缺失结构、产品专属排除、Cloud 延后和 Lime current owner；不得只复制行为而忽略 Codex 文件、函数和类型命名。

`npm run inventory:tui-structure` 读取 `/Users/coso/Documents/dev/rust/codex/codex-rs/tui/src` 与 `lime-rs/crates/tui/src`，做 TUI 目录、模块、类型和函数符号的双向对照，刷新 `internal/exec-plans/tui-structure-inventory.json`。Codex 产品专属、Cloud 或运行时 owner 缺口必须记录为排除/延后，不得在 TUI 创建第二套状态机。

`npm run smoke:cli-surface-gate-b` 使用真实 `lime`、真实 sibling App Server stdio 和隔离数据目录，验证 Codex 形状的 MCP `list/add/get/remove/start/stop`、`features list/enable/disable`、标准 plugin `add/list/read/search/enable/disable/remove`、`debug models --bundled`、`debug clear-memories`、queue `add/list` 与缺失 Thread fail-closed，以及 OAuth `logout` 的协议缺口 fail-closed。fixture backend 只用于 queue 的显式测试链，不调用正式 Provider；CLI 不直写 MCP 配置、不删除 OAuth 凭据文件。

`lime resume [thread-id]` 复用同一 TUI/session 主链；提供 id 时直接调用 current `thread/resume`，省略 id 时先调用 `thread/list` 打开 session picker，再用所选 id hydrate canonical Thread/Turn/Item 后进入交互界面。连接中断由同一 session owner 做 bounded reconnect，保留 draft 并重新 resume 原 Thread；它不创建第二套历史数据库或 runtime。

`lime thread list|show|archive|unarchive|delete|fork` 是非交互 Thread 管理命令，统一通过 `app-server-client` 调用 v2 `thread/*`，返回 typed JSON；CLI 不直接访问 ThreadStore、数据库或 runtime。

`lime mcp list` 和 `lime skills list` 是只读控制面命令，分别调用 v2 `mcpServer/list` 与 `skills/list`；MCP server 配置列表通过 current App Server owner 返回，Skill 查询支持多个 `--skill-cwd` 与 `--force-reload`。两者都通过 `app-server-client` 连接 App Server，不读取本地 MCP/Skill registry。`mcpServerStatus/list` 仅属于独立的 App Server/GUI 运行时状态控制面，不是 CLI `mcp list` 的事实源。

`lime tui`、`lime exec` 和 `lime resume` 共享 `--model`、`--provider`、`--effort`、`--permissions` 连接参数；模型路由与会话设置统一由 App Server `thread/start` / `thread/settings/update` 处理。

新增 App Server 脚本继续进入 `scripts/app-server/` 或复用现有 App Server npm scripts；涉及 Electron packaged sidecar / release asset 的脚本仍按 Electron / release 批次单独迁移。

### Rust 测试脚本

Rust 测试分层入口仍复用已登记的根脚本 `scripts/run-rust-layer.mjs` 与 `scripts/rust-test-layer-classifier.mjs`，变更范围推导共享实现位于 `scripts/lib/rust-test-scope-core.mjs`，不新增根脚本。对外优先使用 `package.json` 中的稳定入口：

```bash
npm run test:rust:changed
npm run test:rust:related -- <paths...>
npm run test:rust:unit
npm run test:rust:integration
npm run test:rust:layers:stats
```

`test:rust:changed` 默认比较 `HEAD`，也可通过 `npm run test:rust:unit -- --changed=<ref>` 指定 ref；`test:rust:related -- <paths...>` 按显式路径推导受影响 crate。二者都会把 `lime-rs/crates/**` 路径映射到 workspace package，再通过 `cargo metadata` 扩展反向依赖；触碰根 `Cargo.toml`、`Cargo.lock` 或 workspace 配置时自动扩大到 `--workspace`；命中 Rust 路径但无法映射 current workspace crate 时失败，避免静默通过 0 个测试。

macOS 上统一 Rust runner 会在调用者未设置时为 Cargo test worker 提供 `RUST_MIN_STACK=8388608`，与 App Server 大型 async fixture 的仓库验证口径一致；调用者显式设置的非空值始终优先。

### Codex sandboxed Rusty V8 资产

启用 `v8_enable_sandbox` 的 Rust 构建必须使用 Codex 发布的
`rusty-v8-v<version>` 资产。`scripts/lib/rusty-v8-artifacts.mjs` 从
`lime-rs/Cargo.lock` 读取精确版本，下载匹配 target 的 archive、binding 和 checksum，
校验 SHA-256 后注入成对的 `RUSTY_V8_ARCHIVE` / `RUSTY_V8_SRC_BINDING_PATH`。
默认缓存使用 macOS `~/Library/Caches/Lime/rusty-v8`、Windows
`%LOCALAPPDATA%\Lime\Cache\rusty-v8` 和 Linux `XDG_CACHE_HOME/lime/rusty-v8`，
也可通过 `LIME_RUSTY_V8_CACHE_DIR` 覆盖。

不要把 `RUSTY_V8_MIRROR` 指向 Codex release 根路径：`v8` crate 会拼接 Deno
风格的 `/v<version>`，而 Codex 资产位于 `/rusty-v8-v<version>`。Rust 验证应使用
`npm run test:rust:*` 或 `npm run verify:local`，只有显式使用已校验成对路径时才直接运行 Cargo。

新增 Rust 测试治理脚本优先进入 `scripts/lib/` 或未来已登记的 `scripts/governance/` / Rust 领域目录；不要继续向 `scripts/` 根目录添加平级 runner。

### MCP 脚本

MCP current 使用链路 smoke 位于 `scripts/mcp/`。对外继续使用 `package.json` 里的稳定入口：

```bash
npm run smoke:mcp-current
npm run smoke:mcp-current -- --allow-write-fixture
npm run smoke:mcp-current -- --allow-oauth-fixture
npm run smoke:mcp-config-electron-fixture
npm run smoke:model-provider-capabilities-electron-gate-b
npm run smoke:thread-queue-electron-gate-b
npm run smoke:thread-revert-electron-gate-b
npm run smoke:thread-fork-electron-gate-b
npm run smoke:project-directory-electron-gate-b
npm run smoke:mcp-oauth-notification-electron-fixture
npm run smoke:mcp-startup-notification-electron-fixture
npm run smoke:mcp-event-stream-electron-gate-b
npm run smoke:mcp-resource-origin-electron-gate-b
npm run smoke:settings-about-electron-fixture -- --run-id <run-id>
npm run smoke:settings-stats-electron-fixture -- --run-id <run-id>
npm run smoke:settings-environment-electron-fixture -- --run-id <run-id>
npm run smoke:settings-media-services-electron-fixture -- --run-id <run-id>
npm run smoke:settings-web-search-electron-fixture -- --run-id <run-id>
npm run smoke:settings-profile-electron-fixture -- --run-id <run-id>
npm run smoke:settings-appearance-electron-fixture -- --run-id <run-id>
npm run smoke:settings-developer-electron-fixture -- --run-id <run-id>
npm run smoke:scheduled-tasks-electron-fixture -- --timeout-ms 180000
npm run smoke:settings-execution-policy-electron-fixture -- --run-id <run-id>
npm run smoke:settings-mcp-lifecycle-electron-fixture -- --run-id <run-id>
npm run smoke:settings-provider-crud-electron-fixture -- --run-id <run-id>
npm run smoke:settings-memory-soul-electron-fixture -- --run-id <run-id>
npm run smoke:settings-archived-lifecycle-electron-fixture -- --run-id <run-id>
npm run smoke:mcp-context7-live-electron-fixture
npm run smoke:mcp-elicitation-gate-b
npm run smoke:orchestrator-skills-gate-b
```

默认入口只通过 `app_server_handle_json_lines -> App Server JSON-RPC` 验证 `mcpServer/list`、`mcpServerStatus/list`、`mcpTool/list|listForContext|search`、`mcpPrompt/list`、`mcpResource/list` 读链，并禁止旧 `mcp_*` / `get_mcp_servers` Tauri facade 作为成功证据。`--allow-write-fixture` 会创建临时 stdio MCP server 与测试 Thread，覆盖 `mcpServer/create|start|stop|delete`、`mcpServer/tool/call` 与 `mcpServer/resource/read`，并断言工具 `outputSchema` 暴露 `structuredContent`、调用结果保留 `structuredContent`。同一轮还会启动一个必然失败的 server，要求其 `mcpServer/start` 错误以 JSON-RPC error 穿过 Desktop Host 返回，同时健康 server 继续保持 running，且 tool list/call 与 resource read 均可用。
`--allow-oauth-fixture` 会创建本地 OAuth provider，覆盖 `mcpServer/oauth/login`、Electron `open_external_url` 系统浏览器网关、callback token exchange 与 `runtime_status.auth_status` 授权回流，用于复验动态 OAuth current 链路；该模式不依赖真实外部账号或 live Provider。

`npm run smoke:mcp-oauth-notification-electron-fixture` 验证 OAuth callback completion 的 App Server typed notification 与 GUI 自动刷新；`npm run smoke:mcp-startup-notification-electron-fixture` 验证 MCP startup 的 `starting -> ready`、`starting -> failed` typed notification、Settings 连接态与终态 status/tool 刷新。两者都启动隔离的真实 Electron Desktop Host、preload/IPC 与 App Server runtime，不调用正式模型或 live Provider，并要求旧 MCP lifecycle Desktop event、renderer mock fallback 与 App Server mock backend 命中为零。
`npm run smoke:mcp-event-stream-electron-gate-b` 验证 Codex current MCP event stream 的 `active -> event -> reconnect -> terminated` lifecycle：临时 stdio MCP server 通过真实 Thread 和 `mcpServer/event/stream/start` 发出 typed stream notification，Settings 运行状态页通过 `app_server_drain_events` 展示订阅 ID、最近事件、重连次数和终态。证据要求命中 Electron preload/IPC、`app_server_handle_json_lines`、`app_server_drain_events` 与 exact current methods，且 mock fallback、旧 MCP facade、invoke/console/page error 全部为零；该 Gate B 不调用模型或 live Provider。

`npm run smoke:mcp-resource-origin-electron-gate-b` 使用 localhost provider 与临时 `codex_apps` stdio fixture，从真实模型回合生成带 canonical `appContext` 的 completed MCP Item；Workspace 通过该 Item 的 `originCallId` 打开用户可见 HTML resource。Gate B 还会用伪造 Renderer `connectorId` 证明服务端只信任 canonical connector/link authority，并重启 Electron 与 App Server，验证同一 Thread/Turn/Item 冷恢复后再次打开 resource，且不重跑 provider turn 或 MCP tool。证据要求真实 preload/IPC、`app_server_handle_json_lines`、`mcpServer/resource/read`、WebContentsView HTML load、零 legacy/mock/error；不调用正式模型或远端 connector。

`npm run smoke:mcp-config-electron-fixture -- --run-id <run-id>` 是真实 Electron 设置页配置闭环 fixture：从桌面壳侧栏进入设置页，切到 MCP 配置管理，选择 Context7 preset，编辑 streamable HTTP URL 与 `env_http_headers` 环境变量名并保存，再通过 preload `app_server_handle_json_lines -> mcpServer/create|list` 验证 App Server current read model。显式 run-id 时证据写入 `.lime/qc/project-gates/<run-id>/settings-mcp-create-list/`，只有 Electron、preload、`electron-ipc`、current methods、GUI readback、零 legacy/mock/error 与截图全部成立才输出 `settingsScenarioProof={scenarioId:mcp-create-list,complete:true}`。该入口不启动 Context7、不调用真实 provider、不读取或写入真实 key，不走 App Server mock backend、renderer mock fallback 或旧 `mcp_*` Desktop facade。

`npm run smoke:model-provider-capabilities-electron-gate-b` 在隔离配置中固定 official OpenAI Responses route，从真实 Electron 首页打开 ModelSelector，断言 provider capability panel 显示 `namespaceTools=false`、`imageGeneration=true`、`webSearch=true`，并要求 trace 命中 `electron-ipc -> app_server_handle_json_lines -> modelProvider/capabilities/read`。证据默认写入 `.lime/qc/gui-evidence/model-provider-capabilities-electron-gate-b/`，只记录 capability 布尔、method、transport 和错误计数；不调用模型、不访问网络、不保存 Provider 配置、路径或凭证。

`npm run smoke:thread-queue-electron-gate-b` 在隔离 userData 和 unavailable backend 中经真实 preload 创建 canonical Thread、添加一条 durable Queue submission，再从真实 Electron 侧栏打开同一 Thread，断言 marker 在 canonical 时间线中唯一可见、`thread-queue-status` 只显示标题/数量且不重复正文、旧 `thread-queue-items` 节点不存在。setup 阶段由 preload 调用结果证明 `thread/start|thread/queue/add`，GUI 阶段由 safeInvoke trace 证明 `thread/read|thread/queue/list`；四段均绑定同一 Thread，mock/error 为 0。证据默认写入 `.lime/qc/gui-evidence/thread-queue-electron-gate-b/`，不启动 Turn、不调用模型，也不保存 Thread/Queue identity 或本机路径。

`npm run smoke:thread-revert-electron-gate-b` 在隔离 userData 和 workspace 中创建 paginated canonical Thread，并由本地 external backend 完成两个 Turn；随后从第二轮用户消息的真实 GUI 入口确认恢复历史。Gate B 要求同一 Thread 上的第一轮保留、第二轮移除、Thread header 不变、工作区文件内容不变，并证明 GUI action 经 `electron-ipc -> app_server_handle_json_lines -> thread/revert -> thread/read` 刷新 canonical read model，同时观察到 `app_server_drain_events`，mock/invoke/console/page error 为 0。证据默认写入 `.lime/qc/gui-evidence/thread-revert-electron-gate-b/`，包含确认态与成功态截图；不调用正式模型，也不保存 Thread/Turn identity 或本机路径。

`npm run smoke:thread-fork-electron-gate-b` 在隔离 userData 和 unavailable backend 中创建 paginated canonical source Thread，再从真实 Thread header 菜单执行 Fork。Gate B 要求新侧栏项成为 active Thread、成功 toast 可见、`thread/fork` 精确命中 source、`thread/read|resume` 精确命中 forked Thread、read model 与 `thread/started` notification 都以 Codex exact `forkedFromId` 保留同一来源关系，并要求 source/forked Thread 同时留在 `thread/list`；mock/invoke/console/page error 为 0。失败时证据只记录脱敏 method/error/toast/DOM 摘要，避免等待超时后丢失根因。证据默认写入 `.lime/qc/gui-evidence/thread-fork-electron-gate-b/`，不启动 Turn、不调用模型，也不保存 Thread identity 或本机路径；`parentThreadId` 只用于 SubAgent lineage，不作为普通 Fork 关系。

`npm run smoke:project-directory-electron-gate-b` 在隔离 userData 和 unavailable backend 中经真实 preload 创建两个 Project 与一个 canonical Thread，从真实 Electron 侧栏打开该 Thread，在顶栏 Project 目录切换归属，并用 `thread/read` 冷读回。证据要求命中 `electron-ipc -> app_server_handle_json_lines -> project/create|list + thread/start|read|metadata/update`、GUI 目录和选择状态、同一 Thread/Project identity，以及零 mock/console/page/invoke error。默认写入 `.lime/qc/gui-evidence/project-directory-electron-gate-b/`，不启动 Turn、不调用模型，也不保存 Thread/Project identity 或本机路径；该 Gate B 只证明 Project 目录与 Thread 归属产品链，不证明 live provider 或模型回合。

`npm run smoke:settings-mcp-lifecycle-electron-fixture -- --run-id <run-id>` 在隔离 userData 中从真实 MCP Settings GUI 创建配置、修改描述与 Lime 启用状态、冷重启读回、从 GUI 删除并再次冷重启确认不存在；要求命中 `mcpServer/list|create|update|delete`，且 Electron、preload、`electron-ipc`、`app_server_handle_json_lines`、零 legacy/mock/error 与三张终态截图同时成立。该 Gate B-F claim 不启动 live MCP server、不调用工具、不访问配置 URL，也不在 JSON evidence 中保存 server 配置、名称、描述、ID、路径、凭证、prompt、resource 或 tool output。

`npm run smoke:settings-provider-crud-electron-fixture -- --run-id <run-id>` 使用隔离 userData 与 localhost `/v1/models` fixture，从真实 Provider Settings GUI 创建自定义 Provider，先观察错误密钥导致的 401 可见状态，再从 GUI 修正密钥、获取并选择模型、完成连接测试、冷重启读回、从 GUI 删除并再次冷重启确认不存在。要求命中 `modelProvider/list|catalog/list|create|update|fetchModels|testConnection|delete` 与 `modelProviderKey/create`，且 Electron、preload、`electron-ipc`、`app_server_handle_json_lines`、零 legacy/mock/unexpected error 同时成立。该 Gate B-R claim 不访问 live Provider、不证明生产凭证或真实模型 Turn；JSON 与截图不保存 Provider 值、模型 ID、API Host、端口、密钥、路径、请求头或响应正文。

`npm run smoke:settings-memory-soul-electron-fixture -- --run-id <run-id>` 在隔离 userData 中从真实 Memory/Soul Settings GUI 选择 canonical style profile、应用模板、保存并冷重启读回；同时以同一 profile 调用现有 isolated `soul-style` Electron runtime fixture，只提取 `hasInteractionSoul`、`hasMemorySoulSchema`、`hasSavedConfigSource` 等 marker booleans。GUI 侧要求 App Server `config/read|config/batchWrite`、`soulStylePack/list`、Electron/preload/IPC、零 legacy/mock/error；runtime 侧必须证明 prompt marker 完整。该 claim 不保存 Soul 文本、完整 prompt、用户内容、Provider request/response、路径或凭证，也明确不声称 GUI 与 runtime 两次启动共享同一进程或 app-data 目录。

`npm run smoke:settings-archived-lifecycle-electron-fixture -- --run-id <run-id>` 复用 current session-history owner fixture，在隔离 userData 与 unavailable model backend 中从真实侧栏归档持久化对话、冷启动读回归档态、从 Settings 已归档对话页恢复，并再次冷启动读回恢复态。Settings adapter 只保留 `agentSession/list|read|update`、Electron/preload/IPC、生命周期布尔、独立 console/page/invoke/legacy/mock error 计数和归档/恢复截图，不保存对话正文、session/thread identity、数据库行、路径或 import payload。

`npm run smoke:settings-about-electron-fixture -- --run-id <run-id>` 在同一真实 Electron 窗口验证 About 与返回首页两个独立 Settings 场景：About 要求构建时 `VITE_APP_VERSION`、Desktop Host `app.getVersion()` 和用户可见版本一致，同时观察 `check_for_updates`、`get_update_install_session` 与 current App Server IPC；返回首页要求点击 `settings-home-button` 后 `home-start-surface` 可见、Settings header 消失。证据分别写入 `settings-about-fixture-summary.json` 和 `settings-about-fixture-home-summary.json`，两者各自带独立 `settingsScenarioProof`，不得互相代替。

`npm run smoke:settings-stats-electron-fixture -- --run-id <run-id>` 验证 Stats 页只走 `usageStats/read`、`usageStats/modelRanking/list`、`usageStats/dailyTrends/list` 三个 App Server current method；隔离数据为空时允许合法零值/空排行/空趋势，但不允许 loading、读取错误、旧 `get_usage_stats*` facade 或 mock fallback。`npm run smoke:settings-environment-electron-fixture -- --run-id <run-id>` 验证 Settings GUI 经真实 App Server `config/read` 与 Host `get_environment_preview` 读取 current config 和最终合并预览；其 JSON evidence 只记录 method/command 与断言，不记录环境变量名或值。

`npm run smoke:settings-media-services-electron-fixture -- --run-id <run-id>` 验证服务模型页经真实 `config/read`、`model/list`、`modelPreferences/list`、`modelSyncState/read` 与 `voice_models_list_catalog` 进入终态，并同时展示服务模型、图片、视频和语音四个 current 配置区域。隔离环境允许 Provider 清单为空，但配置控件必须可用，且证据不得记录 Provider 配置、模型 ID、凭证或本机路径；该场景不声称真实 Provider 生成请求成功。

`npm run smoke:settings-web-search-electron-fixture -- --run-id <run-id>` 在隔离用户目录中经 Web Search GUI 切换搜索引擎路由并命中 App Server `config/read|config/batchWrite`，冷重启读回后恢复原值，再次冷重启确认恢复完成。结构化 JSON 只记录生命周期布尔与 current method/command，不记录搜索路由值、Provider 配置或任何搜索 Key；截图只包含页面正常可见的路由状态。

`npm run smoke:settings-profile-electron-fixture -- --run-id <run-id>` 使用正式 window OEM runtime override 的 `enabled=false` 在隔离用户目录中进入 non-OEM local Profile，经 GUI 修改昵称并命中 App Server `config/read|config/batchWrite`，冷重启读回后恢复原值，再次冷重启确认恢复完成。该场景不声称默认 managed-account 登录面已完成资料持久化；结构化 evidence 不记录资料原值、fixture 标记值或完整 config。

`npm run smoke:settings-appearance-electron-fixture -- --run-id <run-id>` 同时验证 Appearance 的两个 current owner：主题模式写入 renderer localStorage，推荐行为写入 App Server `config/read|config/batchWrite`。fixture 在隔离用户目录中修改两者、冷重启读回、恢复两者并再次冷重启确认；结构化 evidence 不记录外观原值或完整 config。

`npm run smoke:settings-developer-electron-fixture -- --run-id <run-id>` 从 Developer GUI 点击“复制纯 JSON”，真实采集 `config/read`、`log/list`、`log/persistedTail`、`diagnostics/server/read`、`diagnostics/logStorage/read`、`diagnostics/windowsStartup/read`、`modelProvider/list` 与 `mcpServerStatus/list`。fixture 只用显式 test-only renderer sink 替代最终系统剪贴板写入，不替换任何诊断采集或 bridge；JSON evidence 仅记录写入次数、文本长度、payload shape 布尔与 current method/command，不记录剪贴板正文、日志、配置、路径、Provider/MCP 数据或凭证，也不声称系统剪贴板交付已验证。

`npm run smoke:scheduled-tasks-electron-fixture -- --timeout-ms 180000` 从真实 Electron 一级导航创建已安排任务并立即运行，要求命中 `scheduledTask/list|create|read|run/list|run/start`、RuntimeCore provider、canonical Thread/Turn/read model 与运行历史。fixture 使用隔离 userData 和 localhost OpenAI-compatible provider，不调用正式模型；旧 `automationJob/*`、`automationSchedule/*`、`automationScheduler/*`、legacy Desktop 命令与生产 mock fallback 命中必须为零。

`npm run smoke:settings-execution-policy-electron-fixture -- --run-id <run-id>` 验证 Execution Policy Settings 的持久化策略输入和 App Server 错误恢复：从 GUI 保存严格工作区限制与 Bash warning-bypass 输入，冷重启读回；再在隔离 userData 中把临时 `config.yaml` 短暂替换成同名目录，要求真实 `config/batchWrite` 返回且页面展示 `EISDIR`，随即恢复文件、重新加载、恢复原策略并再次冷重启确认。expected save failure 单列计数，`errors.*` 只统计 unexpected errors；evidence 不保存配置值、规则、prompt、路径或错误正文。该 B-F claim 不证明 RuntimeCore 实际执行某条允许/拒绝工具，后者必须单列 B-R。

`npm run smoke:mcp-context7-live-electron-fixture` 是真实 Electron + 远程 Context7 live fixture：复用设置页 GUI 创建 Context7 配置，经 `app_server_handle_json_lines` 启动 server、通过 `mcpTool/search` 找到 `resolve-library-id` / `query-docs`，再创建真实 Thread 并调用 `mcpServer/tool/call` 查询 “AI Agent 是什么”。该入口会访问远程 Context7；summary 只记录 host、工具名、header 名、env var 名、content 类型 / 数量和 `isError`，不记录 key、header value 或工具正文。

`npm run smoke:mcp-elicitation-gate-b` 是 server-originated elicitation 的真实 Electron Gate B：临时 localhost OpenAI-compatible provider 先请求 `mcp__<server>__release_check`，临时 stdio MCP server 在 scoped tools/call 内发出 `elicitation/create`，App Server 将其转为 typed reverse JSON-RPC，Renderer 在当前 Thread 的 Composer 上方唯一表单提交 `{ confirmed: true }`，随后断言 MCP ledger 收到 accept、provider 第二次请求获得最终文本、表单在 `serverRequest/resolved` 后关闭且没有根部 Dialog。Gate B 还要求实际接受 elicitation 的 runtime stdio 连接以 MCP `2025-06-18` 广告精确 `{"elicitation": {}}`，management 连接保持 capability absent；该入口禁止以 `mcpTool/callWithCaller`、`agentSession/action/respond`、mock backend 或 renderer mock 作为成功路径。

`npm run smoke:orchestrator-skills-gate-b` 是 Orchestrator-owned Skills/MCP 的真实 Electron Gate B：隔离环境创建固定名称 `codex_apps` 和一个普通 stdio MCP server，localhost provider 严格执行 `skill_search -> read_mcp_resource -> final text`，并断言 `skill://delivery/release-notes` 只在 Turn snapshot 发现一次、`SKILL.md` 通过 session-owned MCP 精确读取一次、Thread/Turn/Item 与 GUI 最终文本一致。随后脚本经 `config/read|config/batchWrite` 写入 `orchestrator.mcp.enabled=false`，新 Thread 的 provider catalog 必须隐藏 `mcp__codex_apps__apps_ping`，同时保留并成功执行 `mcp__ordinary_fixture__ordinary_ping`。该入口使用 `APP_SERVER_BACKEND_MODE=runtime`、真实 Electron/preload/IPC/App Server 与本地确定性 fixtures，不访问正式模型，不允许 App Server mock backend、renderer mock fallback 或 legacy MCP facade 充当成功证据。

新增 MCP control-plane 脚本继续进入 `scripts/mcp/` 或复用现有 `smoke:mcp-current` npm script；涉及真实 Electron Desktop Host GUI 的 MCP fixture 进入 `scripts/electron/`。共享实现仍放在领域子目录或 `scripts/lib/`。

### Scheduled Tasks 脚本

已安排任务的真实桌面闭环统一通过稳定入口执行：

```bash
npm run smoke:scheduled-tasks-electron-fixture -- --timeout-ms 180000
```

该入口验证 `ScheduledTasksPage -> app_server_handle_json_lines -> scheduledTask/* -> RuntimeCore -> Thread/Turn/Item + Agent Run -> GUI` 的 Gate B 主链。它不能替代真实 Windows Notification Center、macOS/Windows sleep-resume 或 packaged 平台证据。

### Electron 脚本

Electron release / updater 领域新增脚本进入 `scripts/electron/`。当前 `scripts/electron/update-feed-r2-upload-plan.mjs` 负责 R2 updater 上传计划，`scripts/electron/make-zip-local-feed.mjs` 负责用本地临时 feed 验证 Forge macOS ZIP / `RELEASES.json` 生成链路，`scripts/electron/windows-squirrel-rc-smoke.mjs` 负责 Windows N-1 Setup -> current updater -> candidate packaged `SHELL-01` 的 L8 证据，`scripts/electron/windows-native-host-gate-b.mjs` 负责从已安装 `Lime.exe` 启动 Windows native host，校验资源 digest，并验证 UI Automation、窗口/显示枚举、display watcher 和 Raw Input 启停，`scripts/electron/macos-native-host-gate-b.mjs` 负责从已安装 `Lime.app` 校验 helper bundle、协议握手、签名/digest、窗口/显示、真实 security-scoped bookmark create/resolve/start/stop、在三次隔离 Electron 进程中验证 stable bookmark 持久化、冷启动恢复、active lease revoke、撤销后拒绝和 regrant、在临时 Cocoa fixture 上验证窗口 anchor/stack/hide-for-task lease、权限查询和 Launch Services（默认 observe；未授权时窗口编排为 skipped，`--strict-permissions` 才要求 Accessibility、Input Monitoring、Screen Recording 和选定 Apple Events target ready），并通过 preload/IPC 调用 `macos_native_host_invoke`、`app_server_handle_json_lines`，验证 App Server workspace identity、GUI shell 和截图；`scripts/electron/release-workflow-guard.mjs` 负责结构化校验 GitHub Actions release workflow 的 Forge maker、签名、公证、Windows Squirrel 与旧链路拒绝规则。N-1 的 CDP 与隔离 feed 驱动只属于 `scripts/electron/lib/windows-squirrel-n-minus-one.mjs` 测试 helper，不是 production updater API。

Packaged 平台证据必须显式提供完整 Git commit SHA 和有界 `candidateRunId`（CLI 参数 `--candidate-sha` / `--run-id`，或环境变量 `LIME_CANDIDATE_SHA` / `LIME_GATE_RUN_ID`）。Windows 聚合器要求 Squirrel、Code Mode、native host、安装路径、版本与资源 manifest 属于同一 identity。macOS 的 `--release-trust` 只用于 Developer ID release 候选，并额外要求顶层 app、嵌套 helper、Gatekeeper 和 stapling 校验通过；本地 ad-hoc package 不得设置该标记或冒充 release 证据。

`scripts/lib/windows-restricted-execution-evidence.mjs` 是 Windows restricted-token 安全矩阵的唯一证据采集入口。真实 clean Windows runner 必须显式传 `--provision`，由同一入口先在隔离 `LIME_AGENT_RUNTIME_ROOT` 执行 `windows-sandbox-setup`，再执行 `tool-runtime` 的 `windows_restricted_execution` integration test；未显式 provision、setup 失败或矩阵不完整都 fail-closed。schema `windows-restricted-execution-evidence-v3` 分别记录 setup/test 结果与 stdout/stderr artifact；八项矩阵覆盖 unelevated managed-network preflight 拒绝、workspace/metadata denial、online/offline account 选择、offline Firewall loopback enforcement、bounded output、allowlisted stdin、ConPTY stdin/resize/combined-output、Everyone-write ACL audit 与 Job Object cleanup。setup 与 Cargo 各有固定超时，Windows ACL audit 对目录/总量/时限上限或 reparse 元数据错误会发出 `failedScan` warning。非 Windows 主机只输出 `evidence-pending` 并以非零退出，不能被当作平台通过。

对外优先使用 `package.json` 里的 `electron:*` npm scripts。`npm run electron:make:zip-local-feed -- --arch arm64` 只写 `.tmp/electron-forge-local-feed`，不能替代 `electron:dist`、release workflow、DMG、签名、公证或 Windows Squirrel 实机证据。

Packaged Gate B fixtures 共享 `dist/`。`build:renderer:electron:smoke` 必须设置 `LIME_VITE_EMPTY_OUT_DIR=0`，避免并行 fixture 运行期间删除 `dist/index.html`；`.lime/electron-fixture-build.lock` 只负责串行构建动作，不能替代对消费期共享产物的保护。

会话文件和 Deep Link 的真实 Electron fixture 使用稳定入口：

```bash
npm run smoke:session-files-electron-fixture
npm run smoke:connect-deep-link-current
npm run smoke:connect-open-deep-link-current
npm run smoke:connect-deep-link-save-current
```

这些入口都启动隔离的真实 Electron Desktop Host，验证 preload、
`app_server_handle_json_lines` 和对应 App Server current method；临时 userData/appData
不得替代真实用户目录。Connect save fixture 使用脚本内测试 key 和 fixture relay，证据不得保存
完整 key。

### Harness 脚本

Harness eval、history、trend、analysis brief 与 replay promote 入口已迁到 `scripts/harness/`。对外继续使用 `package.json` 里的 `harness:*` npm scripts，不直接依赖根目录脚本路径。

DeepSWE Coding 使用以下 current 入口：

```bash
npm run harness:deepswe:preflight
npm run harness:deepswe:run -- --task happy-dom-abort-pending-body-reads --allow-live-provider
npm run harness:deepswe:batch:plan
npm run harness:deepswe:batch:aggregate
```

`harness:deepswe:run` 在隔离 git workspace 中通过 `workspace/ensure -> agentSession/start -> agentSession/turn/start -> agentSession/read -> agentSession/turn/cancel` 执行 Lime current Agent。adapter v6 从 canonical read model 持续记录 `provider.step`、逐步 usage 和每次真实 sampling 的 tool catalog；provider step/token budget 通过 `runtimeRequest.metadata.harness.provider_budget` 下沉到 current reply loop，在工具执行和下一次 sampling 前终止。wall time 只作总兜底，触发后先请求 current turn cancel 并等待真实 terminal。terminal 或预算取消后固化 partial facts 与 `patch.diff`。只有存在 candidate patch 时才进入 Pier separate verifier preflight；缺少容器运行时会保留运行事实并记录独立 verifier blocker，不得生成伪造的 `reward.json`。真实执行默认 fail closed，必须显式允许 live Provider；已有 patch 可用 `--verifier-only --run-dir <path>` 续跑判分。

`harness:deepswe:batch:plan` 生成 Smoke 10 或 Release 20 的固定 identity trial plan；`harness:deepswe:batch:aggregate` 只聚合同一 source/schema/adapter identity 的 verifier-complete trial，并输出标准 `pass@1`、`pass@3`、`pass^3`、wall time、budget token 和 infra diagnostics。旧 identity 只计诊断，不占当前 trial 槽位；缺 live provider、candidate、Pier artifacts 或容器 runtime 时保持 `blocked`。

`harness:deepswe:desktop:controlled` 运行 Desktop Smoke 5 的真实 Electron 产品路径：GUI 提交原始 DeepSWE instruction，经过 preload/IPC/App Server/RuntimeCore，执行 `Read/Glob/Grep/apply_patch/exec_command`，运行五种语言的 native fixture tests，记录 Thread/Turn/Item、GUI terminal、diff、进程树 `cold_restart`、patch SHA 和零 mock/error。它使用 localhost controlled provider 与合成 task workspace，只能证明桌面产品链；不会产生 DeepSWE score 或 `DesktopCodingPass`。`harness:deepswe:desktop:preflight` 校验五题 source/task/separate verifier 合同；`harness:deepswe:desktop:aggregate -- <evidence-dir>` 只接受 `deepswe-desktop-trial-v1` evidence，并要求 live trial、同一 patch SHA、Pier artifacts 与 Gate B 同时通过。

新增 Harness 脚本继续进入 `scripts/harness/` 或复用现有 Harness npm scripts；共享实现仍放在 `scripts/lib/`。

### Agent QC 脚本

Agent QC report、GUI flow、qcloop、evidence、release summary 与 owner/checklist 入口已迁到 `scripts/agent-qc/`。对外继续使用 `package.json` 里的 `agent-qc:*` npm scripts，不直接依赖根目录脚本路径。

`npm run agent-qc:project-gate-candidate -- --codex-reference-repo <path>` 连续两次计算 tracked、untracked 与删除项的完整产品快照，默认间隔 5 秒；只有 product digest、tracked diff digest、Git HEAD、changed paths、exclusion、干净 Codex reference HEAD、owner tracker 状态和 34-surface contract 全部一致，才会在 `.lime/qc/project-gates/<run-id>/candidate.json` 生成候选摘要。Codex import tracker 只接受 `ready`、`ready-for-gate`、`completed` 或 `closed`；`active`、缺失或未知状态全部 fail closed。`internal/test/project-gate-surfaces.manifest.json` 固定 `17` 个 P0、`17` 个 P1 及每个 surface 所需 proof level，并进入 product snapshot；摘要另外保存其 digest。摘要只保存 Codex commit hash、tracker 相对路径/状态和仓库内 contract 路径，不保存外部仓库路径。Gate 日志、可变执行计划和 `internal/research/refactor/v2/13-evidence/project-gates/` 下的 Gate 汇总被显式排除，其他未跟踪产品文件必须进入 digest；`--snapshot-only` 只用于冻结前诊断，不生成候选。也可用 `LIME_CODEX_REFERENCE_REPO` 提供 reference 路径。

`npm run agent-qc:project-gate-settings-a -- --run-id <run-id>` 是 `SETTINGS-01` 的专用 Gate A browser-mirror runner：通过稳定 `data-testid` 覆盖全部 current primary tabs、desktop/compact/narrow 三视口、五个 locale 的关键 tab、导航恢复、raw key、loading/error buffer、console/page error 和页面级横向溢出；同时用显式 test-only `agentSession/list` 网络夹具验证已归档对话的 loading、empty、error 与 retry 状态。证据写入 `.lime/qc/project-gates/<run-id>/settings-01-gate-a/`，三种组件态、截图和常规矩阵全部完成时才写 `surfaceProof.complete=true`；它仍明确不声明 Electron main/preload/IPC 或 Gate B。

`npm run agent-qc:project-gate-settings-b -- --run-id <run-id> --source <kind>=<summary.json>...` 聚合同一 run-id 下的 SETTINGS-01 Gate B-F owner evidence。当前合同覆盖 15 个 primary tab 对应的 17 个稳定场景；source 必须位于 `.lime/qc/project-gates/<run-id>/`，且 candidateRunId、真实 Electron/preload/IPC、`app_server_handle_json_lines`、current method、零 legacy/mock 与 owner assertions 全部匹配。现有 `shell-memory` 只完成 `memory-ready`，不得冒充 Soul 持久化；Provider 只接受 current `provider-crud-model-auth` 场景，不再接受整库迁移证据。MCP、About/Home、Stats、Environment、Media Services、Web Search、Profile、Appearance、Developer、Automation 与 Execution Policy fixture 分别完成 `mcp-create-list`、`about-version-truth`、`home-navigation`、`stats-current-read`、`environment-current-read`、`media-services-readiness`、`web-search-route`、`profile-persistence`、`appearance-persistence`、`developer-current-diagnostics`、`automation-lifecycle` 与 `execution-policy-allow-deny-error`。其余场景继续使用 `settings-scenario` 结构化证据逐项补齐。聚合结果写入同一 run 的 `settings-01-gate-b-f/summary.json`，17 项未全部完成前 `surfaceProof.complete=false`。

每个 Wave 结束后使用 `npm run agent-qc:project-gate-candidate -- --verify-candidate .lime/qc/project-gates/<run-id>/candidate.json` 重算当前 snapshot；Git HEAD、product digest、tracked diff digest、changed paths 或 exclusion 任一漂移都会非零退出，并只输出 digest、计数和最多 50 个路径差异，不输出文件正文。

候选冻结后使用 `npm run agent-qc:project-gate-coverage -- --candidate <candidate.json>` 聚合 34 个
surface 的证据。聚合器只读取 evidence JSON 中显式声明的 `surfaceProof.surfaceId`、
`surfaceProof.proof` 和 `surfaceProof.complete=true`，同时要求 candidate run-id 一致、场景 `result=pass`
且 assertions 全部通过；不从文件名、场景名或 `proofLevel` 文案猜测覆盖关系。默认未达到 `34/34`
就非零退出；Wave 过程报表显式使用 `--progress-only`。失败或 blocked 证据必须包含
`failureClass` 与 `nextAction`，coverage summary 只保存 proof、计数和相对 evidence 路径，不复制请求、
对话正文或凭证。

新增 Agent QC 脚本继续进入 `scripts/agent-qc/` 或复用现有 Agent QC npm scripts；共享实现仍放在 `scripts/lib/`。

### Agent Runtime 脚本

Agent Runtime smoke 与 Service Skill 入口 smoke 已迁到 `scripts/agent-runtime/`。对外继续使用 `package.json` 里的 `smoke:agent-runtime-*` 与 `smoke:agent-service-skill-entry` npm scripts，不直接依赖根目录脚本路径。

`npm run smoke:agent-runtime-current-fixture` 是 Claw / Agent Runtime current 主路径的离线 fixture 回归聚合入口，覆盖历史 / 缓存恢复、流式终态收尾、Claw 终态 UI、Electron session history / 代码产物工作台 fixture guard、真实 GUI coding 输入到 Coding Workbench Electron fixture、Claw GUI current fixture guard，以及真实 Electron `cancel-then-continue` 场景。它默认禁止 live Provider 和 mock backend，只能作为进入 Electron / Playwright 真实闭环前的快速回归门槛，不能替代完整 GUI E2E。

`npm run smoke:browser-runtime-gate-a` 是 Browser Workspace 的独立 Gate A 投影回归：在真实 Electron fixture 中建立 canonical Right Surface，但不启动 Agent turn；通过固定 chrome test id 验证首屏 tab、同 tab 导航、查找、缩放、新建/选择/关闭 tab、收起/恢复、桌面视口 resize、canonical session/thread 投影和无横向溢出。证据写入 `.lime/qc/gui-evidence/browser-runtime-gate-a/`，明确标记 `proofLevel=Gate A`，不能证明 Agent 与用户操作同一 WebContents；同一 WebContents 和 runtime/tool/approval 仍由 `smoke:browser-runtime-electron-gate-b` 取证。

`npm run smoke:browser-runtime-locale-matrix` 顺序启动五个隔离的真实 Electron lifecycle Gate B（`zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`），通过真实侧边栏语言菜单切换并断言 `document.documentElement.lang`，为 renderer loading/ready、Browser loaded、Agent control、user takeover、released、destroyed 采集 35 张截图，同时聚合 approval/artifact/cancel/disconnect/download/permission/user-control/window-close 行为矩阵。`--aggregate-only` 只重建汇总，不重新启动 Electron。证据写入 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-locale-matrix-summary.json`；它不把普通 Chrome、mock 或单测当作 Gate B，也不扩大 open/reveal/clipboard/upload 的 Host 单测边界。

`npm run smoke:hook-lifecycle-gate-b` 复用 Claw 的真实 Electron `turn-plan-update` provider/tool-call 场景，并在隔离 `CODEX_HOME` 中注册受信 `PreToolUse(update_plan)` command Hook。门禁要求 `hooks/list` 返回 trusted exact metadata、command 进程实际记录工具上下文、同一 run id 的 `hook/started -> hook/completed` 被 Electron recent-notification bridge 观察、public `thread/read(includeTurns:true)` 恢复 completed canonical Hook Item，且 GUI 出现 `timeline-hook`。该入口使用 localhost OpenAI-compatible fixture，不调用 live Provider、不使用 App Server mock backend，也不把通用 GUI smoke 冒充 Hook 闭环。

`npm run smoke:agent-runtime-soak-current-fixture` 在同一真实 Electron/App Server 生命周期连续运行 10 轮 AgentControl current fixture，并执行两次 cold restart。它逐轮读取 public JSON-RPC 的 Thread/Turn/Item、PID/RSS 和 terminal 状态，重启后比较所有 canonical identity，最后验证 6 条工具 row、4 条 SubAgent activity、零 invoke/console error 和全部进程退出。该入口使用 localhost controlled provider，只关闭 fixture 的 idle connection；active provider request 必须由 Lime 正常释放，因此不会用强制断连掩盖 SSE 生命周期缺陷。它证明本地 SOAK 实现合同，不替代 live Provider、冻结 RC 或真实平台长稳。

`npm run smoke:agent-session-recovery-cdp-gate` 是未完成 Agent 会话恢复的真实 Electron CDP Gate B 骨架入口：启动 Electron Desktop Host，通过 `chromium.connectOverCDP` attach 到真实 renderer，验证 `window.__LIME_ELECTRON__`、preload invoke、`app_server_handle_json_lines`、`agentSession/start/read/list` 与侧栏打开同一 session。它使用 `APP_SERVER_BACKEND_MODE=unavailable`，不触发 `agentSession/turn/start`，不调用正式模型后端，也不证明 live Provider 或运行中 turn 输出。

`npm run smoke:expert-skills-live-gate` 是专家 Skills Runtime 的证据门禁：默认只读取 `.lime/qc` 中的确定性 Electron fixture summary，确认专家 declared / selected / invoked、`skill_search -> SKILL.md body read -> Skill gate -> Skill invocation`、Harness GUI Evidence Pack 导出与专家面板复盘证据完整；缺少显式 live Provider summary 时返回 `pending_live_provider`，不调用真实模型，也不把 deterministic fixture 误当完整 live 验收。

`npm run smoke:expert-skills-live-runner` 是专家 Skills Runtime 的 live Provider 验收入口骨架：默认 fail-fast，必须显式传 `--allow-live-provider` 或设置 live Provider smoke 环境变量。它可用 `--live-summary <path>` 归一化已有 live evidence，也可在额外传 `--execute-live-runtime` 时通过 App Server current JSON-RPC 提交真实 Provider turn，并输出 `.lime/qc/expert-skills-live-runner-summary.json` 供 `smoke:expert-skills-live-gate -- --live-summary <path>` 审计。

`npm run smoke:agent-session-history-electron-fixture` 是真实 Electron 历史恢复 fixture：通过 preload `app_server_handle_json_lines` 验证 App Server current `agentSession/start/read/update/list` 形状、最近对话可见和 hydrate detail 数组；它使用 `APP_SERVER_BACKEND_MODE=unavailable`，不触发 turn，也不调用模型后端。

`npm run smoke:codex-import-continuation-electron-fixture` 是真实 Electron 本地历史导入续聊 fixture：通过 preload `app_server_handle_json_lines` 导入一条本地 rollout fixture，验证 `agentSession/read.detail.items` 能恢复 reasoning、command、patch、web search、approval，再在同一个导入 session 上调用 `agentSession/turn/start` 继续对话。它使用本地 external backend fixture，不调用正式模型，不走 App Server mock backend、renderer mock fallback 或 legacy runtime command。

`npm run smoke:codex-import-click-through-electron-fixture` 是真实 Electron 本地历史导入点击闭环 fixture：使用临时 `CODEX_HOME` 写入 `session_index.jsonl` 与 rollout JSONL，从侧边栏点击“本地历史导入”，在确认弹窗预览“导入细节还原”后点击确认，稳定进入导入会话页，验证导入消息、reasoning、友好命令记录、patch、web search、approval 默认可见，再通过真实输入框发送 follow-up。该入口同时覆盖 commit 后导航不被 task-center 旧 tab fallback 抢回、imported timeline 工具细节默认展开、预览不暴露 raw source event / payload 字段、续聊不暴露 fixture 哨兵、消息列表主线不展示 `imported-source-banner` 或“本地历史导入 / 已还原”独立状态条；环境信息弹层不重复展示导入主线卡，也不暴露 `Approve Codex command` / `npm test` / 原始 thread id 等内部细节，以及同一 session 的 `agentSession/turn/start` backend ledger。脚本还会在 `visual-audit/` 下输出 `desktop / compact / narrow` 三种视口截图，并把输入框可见性、消息列表可见性、导入细节可见性和无导入主线卡写入 summary。它使用本地 external backend fixture，不读取真实 `~/.codex`，不调用正式模型，不走 App Server mock backend、renderer mock fallback 或 legacy runtime command。

`npm run smoke:local-history-import-visual-audit` 是本地历史导入的产品视觉边界审计：复用真实 Electron 点击闭环 fixture，再检查 `desktop / compact / narrow` 三视口的消息列表、输入框、导入命令 / 补丁 / 搜索 / 审批记录、无导入主线 banner / run control 卡，并扫描 GUI 可见文本，确保除导入来源 / provenance / fixture / 协议枚举外不泄漏来源品牌字眼。该入口仍使用临时本地历史 fixture 和 external backend，不读取真实用户历史目录，不调用正式模型。

`npm run smoke:local-history-import-real-sample-visual-audit` 是真实 Codex 样本 GUI 审计入口：启动真实 Electron Desktop Host 与隔离 App Server data 目录，从当前工作区匹配的 Codex 本地历史源只读 scan/preview，在可审计预算内选择复杂度最高线程导入后从侧边栏打开会话，采集 `desktop / compact / narrow` 三视口与 `top / middle / bottom` 滚动截图，并验证输入框、消息列表、导入命令 / 补丁 / 搜索 / 审批记录可见，普通 GUI 不暴露 source path、source thread id、raw event 字段或导入运行控制卡。默认预算为 5,000 rollout 行、200 消息、1,200 timeline item，可通过 `--max-source-lines`、`--max-source-messages`、`--max-source-items` 调整。该入口使用 `APP_SERVER_BACKEND_MODE=unavailable`，不调用正式模型，不走 App Server mock backend、renderer mock fallback，也不把真实对话正文写入证据 JSON。

`npm run smoke:code-artifact-workbench-electron-fixture` 是真实 Electron 代码产物工作台 fixture：使用本地 external backend fixture 生成 `artifact.snapshot`、标准 coding facts 与 current `turn.completed`，再从 GUI 历史会话打开工作台，验证代码产物入口、变更 / 输出 / 日志面板和工作台可见性；传入 `--scenario gui-coding-input` 时会先通过真实 GUI 输入框发送 coding 请求，再验证同一套 Workbench 证据。它不调用正式模型，不走 App Server mock backend。

`npm run smoke:claw-chat-current-fixture` 是更重的真实 Electron GUI fixture：通过真实输入框发送“整理今天的国际新闻”，验证用户输入可见、assistant 完成态输出可见、输入框不消失、App Server `agentSession/turn/start` 走 current JSON-RPC、WebSearch 不按关键词强制 required，并使用本地 external backend fixture 代替正式模型后端。修 Agent Runtime / Claw 输入、流式卡住、历史 hydrate 或新闻请求链路时，先跑聚合 guard，再按需要显式跑该入口；修无法停止或停止后无法继续输出时，还必须跑 `--scenario cancel-then-continue`，证明同一 current session 停止后能再次从 GUI 输入“继续输出”并完成第二轮。

`npm run smoke:unknown-item-recovery-electron-fixture` 复用同一真实 Electron fixture 注入一个受控 future Item type，验证 `item.started -> item.completed -> turn.completed` 经 preload/IPC、App Server runtime/read model 与 direct TurnTimeline fail-visible；GUI 和 cold read 只保留 upstream type 与脱敏字段名，不允许原始字段值、secret、raw payload 或生产 mock fallback。

`npm run smoke:claw-image-live` 是 `@配图` 真实 Provider live 验收入口：默认 fail-closed，必须显式传 `--allow-live-provider` 或设置 `LIME_ALLOW_LIVE_PROVIDER_SMOKE=1 / LIME_REAL_API_TEST=1`。它启动真实 Electron Desktop Host 与 `APP_SERVER_BACKEND_MODE=runtime`，通过 GUI 输入框发送 `@配图`，验证 Agent 普通对话流里的思考 / 引导文字、`Image Generation` 图片任务卡、真实图片预览、Token 显示、右侧 viewer 不自动展开，以及普通 UI 不暴露 task path、workflow 字段、provider 内部字段或模板 task id。传 `--setup-agnes-from-env` 时只从 `AGNES_API_KEY`（或 `--api-key-env` 指定变量）读取 key，summary 仅记录 `apiKeyConfigured: true`，不记录 key 值。该入口还会经 `mediaTaskArtifact/list|get`、`workflow/read` 与 task audit JSONL 验证后端事实源，确保图片任务只落可审计 JSONL / workflow summary，不靠 mock worker、renderer mock fallback 或右侧 viewer 展示内部 JSON。

新增 Agent Runtime 脚本继续进入 `scripts/agent-runtime/` 或复用现有 Agent Runtime npm scripts；共享实现仍放在 `scripts/lib/`。

### Plugin 脚本

Plugin consumer runtime、独立 UI runtime、standalone shell、content-factory 发布链和 package handoff 已归类为 `dead / deleted / forbidden-to-restore`。`scripts/plugin/` 当前只保留 connector production delivery 的 current 检查：

```bash
npm run plugin:connector-production-preflight
npm run plugin:connector-production-delivery-gate
npm run plugin:connector-production-webhook-delivery
```

Plugin v3 标准包真实 Electron 验证使用：

```bash
npm run smoke:plugin-package-electron-gate-b
```

该 Gate B 从 App Center 安装只含根 `plugin.json`、根 `mcp.json` 和合法
`skills/<name>/SKILL.md` 的本地标准包，经 Electron preload/IPC、
`app_server_handle_json_lines`、App Server、RuntimeCore 和 GUI 完成启停、新 Thread
隔离、canonical `plugin://<name>@<marketplace>` mention、Skill 正文注入、MCP tool、
elicitation、Right Surface、reload/cold restore 与卸载后历史读取；生产 mock fallback、
旧 worker 和旧 Plugin 命令命中必须为零。

Plugin v3 标准包验证只允许围绕根 `plugin.json`、`skills/`、`mcp.json` 和 App Server current catalog 建立；不得恢复独立 worker、iframe host、standalone shell 或旧 package 发布 helper。新增 Plugin 脚本继续进入 `scripts/plugin/`，真实 Electron fixture 进入 `scripts/electron/`，共享实现进入 `scripts/lib/`。

### 项目热力图

静态项目观察报告继续使用：

```bash
npm run heatmap:project
```

完整流程见：

- `internal/aiprompts/project-heatmap.md`
