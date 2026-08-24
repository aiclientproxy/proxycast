# Desktop Host 与 App Server 命令边界

状态：current

目录、依赖方向和完整协议层级以 [architecture.md](architecture.md) 为准。本页只定义命令应落在哪一层，以及跨层变更的同步要求。

## 唯一业务通道

```text
Renderer typed gateway
  -> preload / Desktop Host（仅宿主能力或 JSONL 转发）
  -> app_server_handle_json_lines
  -> App Server JSON-RPC method
  -> runtime/domain owner
```

业务能力只通过 App Server JSON-RPC 进入 Rust runtime。Electron IPC 只用于窗口、文件/目录选择、系统权限、外链、托盘、自动更新、sidecar 生命周期等宿主能力，或转发 `app_server_handle_json_lines`。Renderer 不直接调用 provider、tool runtime、数据库或 Electron main 私有实现。

## Owner 判定

| 需求                                                            | Owner                                               |
| --------------------------------------------------------------- | --------------------------------------------------- |
| Thread / Turn / Item、read model、evidence、业务查询与写入      | App Server protocol + handler + current Rust domain |
| 模型路由、canonical content、capability、provider wire lowering | `runtime-core` / `model-provider`                   |
| 工具定义、审批、sandbox、dispatch、MCP                          | `tool-runtime`                                      |
| 窗口、系统文件选择、通知、Dock、tray、updater、sidecar          | Electron Desktop Host                               |
| UI request builder、response normalization、projection          | Renderer `src/lib/api/` 或 typed package            |

禁止为业务调用新增第二个 Electron 后端、renderer mock fallback、临时 DevBridge 命令或 legacy wrapper。生产失败必须显式失败；mock 仅在测试夹具中显式注入。

## Codex 能力边界

Codex 的 `requestAttestation` 只用于客户端声明接收 `attestation/generate`，由 Desktop Host 生成不透明 token，再转成上游 `x-oai-attestation`。Lime 当前没有真实 token producer，因此 initialize 收到 `capabilities.requestAttestation=true` 时必须 fail closed；不得静默忽略、生成假 token 或新增 `attestation/generate` 兼容入口。

Codex 不定义 portable signed receipt，也不为 task、tool、approval、artifact 或 transcript 提供逐项签名。Lime 的 handoff、replay、analysis 和 review 导出只消费 canonical Thread/Turn/Item read model；其 digest 只能表示完整性，不能宣称来源真实性。本轮不新增签名字段、密钥管理或 BoundaryAttest 依赖。

## 协议变更清单

新增或修改跨层业务 method 时，同一变更集必须同步：

1. `app-server-protocol` method、params、result、notification 与 schema。
2. App Server handler、current domain owner 与 Rust client（如适用）。
3. `packages/app-server-client` 或 Renderer typed gateway。
4. Electron preload / IPC 白名单，仅当请求需要宿主转发或系统能力时。
5. catalog、受控 fixture、mock policy 和负向回流 guard。
6. `npm run test:contracts` 与受影响的 Rust / TypeScript 定向测试。

改变 method、schema、read model、notification、preload 边界或 sidecar 行为属于重大架构变更时，按 [architecture.md](architecture.md#11-重大架构变更与开发者确认) 更新架构图并由责任开发者确认。

## 验证入口

| 风险                    | 最低验证                                              |
| ----------------------- | ----------------------------------------------------- |
| Typed client / protocol | `npm run test:contracts`                              |
| Rust domain             | `npm run test:rust:related -- <paths...>`             |
| Desktop bridge          | `npm run test:contracts` + `npm run verify:gui-smoke` |
| Agent 主链              | `npm run smoke:agent-runtime-current-fixture`         |
| 真实桌面闭环            | Gate B Electron fixture / GUI smoke                   |

具体质量选择见 [quality-workflow.md](quality-workflow.md)。

## MCP 控制面主链

MCP 管理、发现和调用只允许走：

`src/lib/api/mcp.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server JSON-RPC -> lime-rs/crates/mcp`

current method 为 `mcpServer/list`、`mcpServerStatus/list`、`mcpServer/create`、`mcpServer/update`、`mcpServer/delete`、`mcpServer/enabled/set`、`mcpServer/importFromApp`、`mcpServer/syncAllToLive`、`mcpServer/oauth/login`、`mcpServer/oauthLogin/completed`、`mcpServer/startupStatus/updated`、`mcpServer/start`、`mcpServer/stop`、`mcpServer/resource/read`、`mcpServer/tool/call`、`mcpTool/list`、`mcpTool/listForContext`、`mcpTool/search`、`mcpPrompt/list`、`mcpPrompt/get`、`mcpResource/list`、`mcpResource/subscribe` 与 `mcpResource/unsubscribe`。`mcpServer/tool/call` 强制真实 `threadId` 并经 Session-owned `McpThreadRuntime` 执行；Settings 没有 Thread owner，只允许浏览工具。`mcpServer/resource/read` 可做 management read，也可携带真实 `threadId` 读取同一 runtime；`sessionId` 不进入 exact wire。旧 `mcpTool/call`、`mcpTool/callWithCaller` 与 `mcpResource/read` 已从 protocol catalog、schema、App Server、typed clients、Renderer、smoke 和正向测试物理删除，分类为 `dead / deleted / forbidden-to-restore`，只能出现在负向回流守卫或历史 evidence。OAuth 完成态只允许由 App Server v2 typed notification `mcpServer/oauthLogin/completed` 投影给 Renderer；MCP startup lifecycle 只允许由 `mcpServer/startupStatus/updated` 投影连接态并触发终态刷新。旧 `mcp:oauth_completed`、`mcp:server_started`、`mcp:server_stopped`、`mcp:server_error` 均为 `dead / deleted / forbidden-to-restore`。事件 `mcp:resources_updated` 和 `mcp:resource_updated` 必须经真实 MCP manager / Desktop Host event bridge 投影；浏览器模式不得静默回退 mock event fallback。

live evidence 仅通过 `smoke:mcp-current -- --allow-live-provider` 显式开启，且需要 `LIME_MCP_LIVE_SERVER_URL`。该 URL 不得包含 username、password、query 或 hash；认证只能引用环境变量名，不允许 inline secret。`network-invoke.json` 仅可记录脱敏的 host、环境变量名、header 名、范围和工具/资源摘要。

MCP server-originated elicitation 使用独立 reverse JSON-RPC method `mcpServer/elicitation/request`。该 method 在 protocol catalog 中属于 `serverRequest`，不属于 Renderer 发起的 `AppServerRequestMethod`。App Server 生成 outer JSON-RPC id 并按 id 精确等待 Response/Error；Electron `app_server_drain_events` 只上行 notification/request，`app_server_handle_json_lines` 只把 Renderer 回包原样写回 sidecar。Renderer 必须通过 typed server-request dispatcher 注册 method handler；未知 method 返回 `METHOD_NOT_FOUND`。禁止暴露 MCP raw request id、按 server/turn/tool 扫描 waiter，或复用 `agentSession/action/respond`、Approval、`request_user_input` 与生产 mock fallback。

MCP model Tool surface 与 GUI 管理读必须分层：`tool-runtime::McpStepSnapshot` 只冻结同一次 provider sampling 的 tool definitions、caller policy、exact route 和 connection handle；`mcpPrompt/*`、`mcpResource/*`、`mcpServerStatus/list` 继续由 App Server 直接向 `lime-mcp::McpClientManager` 做 live read。禁止让管理面经过 model bridge、让 GUI inventory 替换 in-flight snapshot，或用 caller-unaware live registry dispatch 绕过当前 step allowlist。

旧 MCP Desktop facade 已统一归类为 `dead / retired guard-only`：`get_mcp_servers`、`mcp_list_servers_with_status`、`mcp_list_tools`、`mcp_list_prompts`、`mcp_list_resources`、`mcp_call_tool`、`mcp_start_server`、`sync_all_mcp_to_live` 只能出现在负向 guard 或历史 evidence，禁止回到前端网关、Desktop Host、mock 或 App Server current 主链。

## Skills Catalog 主链

Composer 可执行 Skill catalog 只允许走：

`src/lib/api/skill-execution.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server skills/list -> RuntimeCore -> lime-skills AgentSkillSnapshot`

current list method 为 `skills/list`，contract 是 `cwds + forceReload -> data[{cwd,skills,errors}]`；catalog 变更只通过 typed `skills/changed {}` 触发 Renderer 刷新。`skill/read` 独立承担稳定 id 的正文/工作流详情读取，`skillManagement/*` 只属于管理中心，不得冒充 Composer catalog。singular `skill/list` 与 `get_local_skills_for_app` Desktop facade 为 `dead / deleted / forbidden-to-restore`，只能出现在负向守卫或历史 evidence。

Skill 运行时配置只允许走同一 App Server 主链：`skills/config/write` 使用 exactly-one `path/name` selector 写入 Lime 用户级 YAML `skills.config` 并返回 `effectiveEnabled`；`skills/extraRoots/set` 原子替换进程级 roots，不持久化，缺失目录按空目录处理，成功后发送 `skills/changed {}`。Renderer typed gateway 可以调用这些 current method，但不得用 `skillManagement/*`、Electron IPC 或 Codex TUI 配置路径建立第二套状态。

Orchestrator-owned remote Skills 不新增 Renderer/Electron 命令，且不经过全局管理面 MCP inventory。一次真实 turn 初始化后，App Server 通过 `AgentRuntimeState[sessionId, threadId] -> McpThreadRuntime` 连接固定 `codex_apps`，以 10 秒、10 页和资源/正文边界发现 `mcp/skill`，把结果合并进同一 `AgentSkillSnapshot`；reroute 复用该快照。模型可见的 `skill_search` 只返回 metadata 与 `skill://.../SKILL.md` locator，正文必须显式调用 `read_mcp_resource(server="codex_apps", uri=...)`，不能由 Skills 或 Tool runtime 直接读取本地文件。`orchestrator.mcp=false` 只过滤 `codex_apps` catalog、tool definitions 和 dispatch route；普通 MCP 与已冻结且精确匹配的 Skills resource read 不受影响。`mcpResource/list` 的 cursor 只能与指定 `server` 一起使用，并返回 `nextCursor`；GUI 管理面仍可从全局 `McpClientManager` live read，但不能替换 in-flight snapshot。Electron 只转发 `app_server_handle_json_lines`，不得持有这套配置、discovery 或读取逻辑。

## Apps Catalog 主链

Apps/connectors 只允许走同一个 Plugin catalog owner：

`src/lib/api/apps.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server app/* -> RuntimeCore -> PluginDataSource -> local plugin_catalog`

current method 为 `app/list`、`app/read`、`app/installed` 与 `app/list/updated`。Portable
Agent Plugins manifest 不允许顶层 `apps`；`app/list` 只从显式 Codex
`extensions.com.openai.apps`（或 overlay fallback）指向的独立 Apps JSON 构建分页 catalog，
配置项的 connector `id` 是 catalog identity。旧内联 Apps object fail closed；非法 Apps
配置只禁用该组件。`app/read` 最多接收 100 个 id，去重并保持首次请求顺序，未知 id 放入
`missingAppIds`；携带 `threadId` 时必须命中已加载 canonical Thread。`app/installed` 只报告有效 enabled/runtime
state；本地 Plugin 没有 hosted connector model-visible tool snapshot 时，`callable` 强制为 `false`，Desktop 不得
把安装或启用状态冒充模型 readiness。`forceRefetch` / `forceRefresh` 在本地 registry 上只是 fresh read，不伪造
hosted refresh。

成功的 `plugin/install`、`plugin/uninstall`、`plugin/enabled/set` 与首页 `app/list` 读取经过现有 server
notification hook 发布 typed `app/list/updated { data: AppInfo[] }`，Renderer 通过 App Server typed event bus 消费并
重新读取 Apps。禁止新增第二 Apps catalog、`window` 自定义事件事实源、TUI Apps UI、compat wrapper 或生产 mock
fallback。Apps 专用真实 Electron Gate B 已完成：App Center 经 typed `app/list/updated` 触发 fresh
`app/list + app/installed` 读取，并从本地 Plugin 的 `pending` 投影切换到停用后的 `disabled`。仍未完成的证据只剩
hosted connector model-visible tool snapshot 与真实 `callable=true` provider readiness；不得用本地 Plugin enabled
状态替代。

## Collaboration Mode Catalog 主链

Desktop Composer 的协作模式发现只允许走：

`src/lib/api/collaborationModes.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server collaborationMode/list -> collaboration mode catalog`

current method 为 exact `collaborationMode/list`，返回 Codex mask shape：`data[{name,mode,model,reasoning_effort}]`。
App Server 是 Default/Plan preset 的唯一事实源；Plan 固定以 `reasoning_effort=medium` 覆盖当前 Turn effort，`model=null`
表示继续使用 Grok-aligned `model-provider` 当前模型选择。Renderer 的 `task_mode` 只表达用户选择意图，真正提交前必须解析
唯一 Plan preset；catalog 缺失、重复或形状非法时 fail closed。不得在 Renderer 重建本地 preset、复制 Codex TUI picker、
新增 Electron 业务命令，或让 collaboration catalog 承接模型 catalog/capability/readiness。

## Experimental Feature 主链

Desktop 实验特性设置只允许走：

`Settings Experimental -> src/lib/api/experimentalFeatures.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server experimentalFeature/* -> lime_core config.yaml`

current method 为 exact `experimentalFeature/list` 与 `experimentalFeature/enablement/set`。catalog 当前只公开真实
Settings consumer 使用的 `webmcp`，默认关闭，stage 为 `underDevelopment`；enablement 只更新已知 key，未知 key
忽略，空 map 为 no-op。`list` 支持 cursor/limit；携带 `threadId` 时必须命中已加载 Thread，但 Lime Desktop 没有
Codex project-local feature config，enablement 仍由单一 `config.yaml` owner 计算并持久化。

Electron 只转发 App Server JSONL，不再读写实验配置。旧 `get_experimental_config`、`save_experimental_config`、
Renderer 直连 IPC、默认 mock handler 和 legacy Tauri facade 为 `dead / deleted / forbidden-to-restore`；不得新增 compat
wrapper 或让 Initialize capability/model catalog 冒充实验特性目录。多模型、多模态 catalog/capability/readiness 继续归
Grok-aligned `model-provider`，不进入 experimental feature owner。

## Memory Reset 主链

全局记忆重置只允许走：

`src/lib/api/memoryStore.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server memory/reset -> RuntimeCore -> MemoryAppDataSource -> LocalMemoryBackend`

current method 为 exact `memory/reset`，只接受 omitted、`null` 或空对象 params，返回空对象。它只清理全局 memory
root 并重建受管目录，不删除 Thread/Turn/Item、event log、projection store 或 memory root 外的 soul 配置。旧 scoped
`memoryStore/reset`、`MemoryStoreResetParams/Response`、typed client 与设置页调用均为
`dead / deleted / forbidden-to-restore`；workspace memory reset 不再作为未被产品消费的平级公开能力保留。

## Command Exec 主链

Codex exact 独立命令执行与 Desktop 交互终端只允许走 connection-scoped App Server JSON-RPC：

`src/lib/api/commandExec.ts -> typed App Server client -> command/exec -> App Server CommandExecServer -> tool-runtime local process supervisor`

一次性命令通过 `command/exec` 返回 `exitCode/stdout/stderr`；流式命令通过
`command/exec/outputDelta` 投影 raw bytes 的 `deltaBase64`，并由同一连接内的 `processId` 过滤。交互终端的
输入、PTY 尺寸和终止分别使用 `command/exec/write`、`command/exec/resize`、`command/exec/terminate`。
`outputBytesCap`、`timeoutMs` 保持 omitted/null/value 语义；stdin close 后的非空写入、非 TTY resize、零值尺寸、
未知 process id 和同一连接重复 active id 均 fail closed。断连、response 发送失败或 notification writer 失败都清理该
连接拥有的进程。Electron 只转发 App Server JSONL，不持有第二套终端会话、轮询 drain 或 renderer mock fallback。

旧 `project_shell_*`、`run_project_shell_command`、Project Shell v0 DTO/schema、旧 API gateway 和 Electron 私有
session host 均为 `dead / deleted / forbidden-to-restore`；没有 compat/deprecated wrapper。

## Process Control 主链

Codex exact 子进程控制只允许走 connection-scoped App Server 主链：

`typed App Server client -> process/{spawn,writeStdin,resizePty,kill} -> ProcessServer -> tool-runtime local process supervisor -> process/{outputDelta,exited}`

`processHandle` 只在发起请求的 `ConnectionId` 内唯一；spawn response 必须先于该进程的 notification，output 必须先于
exited。断连、response 发送失败或 notification writer 失败都终止该连接拥有的进程。stdin close 后的非空写入、
非 TTY resize、零值 terminal size、重复 active handle 和未知 handle 均 fail closed；output cap 默认 1 MiB，
omitted、`null` 与 value 保持三态，notification 的 `deltaBase64` 保留 raw bytes。

Desktop Workspace 不直接消费 connection handle。代码工作台只允许通过
`src/lib/api/backgroundTerminals.ts -> thread/backgroundTerminals/list -> command itemId 匹配 -> thread/backgroundTerminals/terminate`
终止 Thread-owned 后台终端；不提供旧 status refresh、drain、signal-only interrupt 或任意 stdin 写入控件。旧公开
`executionProcess/*`、v0 DTO/schema、typed helpers 和 Renderer gateway 均为
`dead / deleted / forbidden-to-restore`；内部 `ExecutionProcessServer` 继续作为 Thread shell、unified exec 和后台终端
的 current supervisor owner，不是公开兼容层。

## Filesystem 主链

Codex exact 文件 IO 与 watcher 只允许走 connection-aware App Server 主链：

`src/lib/api/fileBrowser.ts -> typed App Server client -> fs/{readFile,writeFile,createDirectory,getMetadata,readDirectory,remove,copy,watch,unwatch} -> App Server FsServer -> fs/changed`

所有路径必须为绝对路径；文件 bytes 在协议边界统一使用 base64。`watchId` 只在发起请求的 `ConnectionId` 内唯一，
`fs/changed` 只发送给该 owner；断连只清理本连接的 watcher。Desktop 文件浏览器继续保留目录、预览、文件类型和
symlink 等 GUI 投影，但不得定义第二套 file wire。rename 由 Renderer 组合 `getMetadata -> copy -> remove`，当前为非原子
Desktop 操作，不新增 Codex 不存在的 rename method。

旧 `fileSystem/*`、v0 DTO/schema、App Server `processor/file.rs`、RuntimeCore file projection、services
`file_browser_service` 与旧 renderer aliases 均为 `dead / deleted / forbidden-to-restore`。`get_file_manager_locations`
和 `get_file_icon_data_url` 仍是 Electron Desktop Host 的系统壳能力，不属于业务文件 IO，也不能承接 fs fallback。
Office/PDF 文本提取不属于 `fs/readFile`；若产品继续需要，应在独立 current 文档能力 owner 中重建，禁止恢复旧
`fileSystem/readFilePreview`。

## Fuzzy File Search 主链

Desktop Composer 项目文件补全只允许走一发式 current 链：

`CharacterMention -> src/lib/api/fuzzyFileSearch.ts -> typed App Server client -> app_server_handle_json_lines -> App Server fuzzyFileSearch -> filesystem search owner`

请求使用 `{ query, roots: [absoluteProjectRoot], cancellationToken }`，返回最多 50 条按 score/path 排序的相对路径、
file/directory、file name 与 match indices。Renderer 使用稳定 cancellation token、AbortSignal 和 request version
丢弃旧响应；选中结果只替换当前 `@token`，空格路径加引号，不创建 connector/plugin `Mention`。Electron 只转发
现有 JSONL，不新增文件搜索 IPC、业务后端或生产 mock fallback。

Codex experimental `fuzzyFileSearch/sessionStart|sessionUpdate|sessionStop` 与
`fuzzyFileSearch/sessionUpdated|sessionCompleted` 不属于 Lime Desktop 产品面，均为
`product-scope-excluded / forbidden-to-restore`。两个 notification 只允许 method/field-name 级 drift diagnostics，
不得进入 current protocol manifest、Composer state、pending interaction 或兼容 wrapper。

## Browser Workspace 主链

Browser 只有一个网页执行体：Right Surface 中用户可见的 Electron `WebContentsView`。用户操作与 Agent 动作分别从以下两条控制路径进入同一个 `BrowserTabHost` route：

```text
Renderer Browser Workspace -> src/lib/api/browserTab.ts -> browser_tab_* Desktop Host command -> BrowserTabHost
RuntimeCore / tool-runtime browser__* -> App Server item/tool/call -> AppServerDynamicToolHost -> BrowserTabHost
```

App Server 持有 Thread/Turn、动态工具生命周期、Browser read model 与 canonical identity；Electron 只持有 native view、window、`webContents.debugger`、下载和权限等宿主状态；Renderer 只消费 projection 并发送用户意图。Agent action 必须按 connection/thread/turn/session/tab/view/WebContents owner 精确路由，用户操作或 turn terminal 必须使旧控制权失效。

外部 Chrome/CDP `browserSession/*`、`BrowserRuntimeManager`、Settings Chrome relay、`BrowserSessionRef` adapter、Canvas Browser 和 `mcp__lime-browser__*` 正向工具均为 `dead / deleted / forbidden-to-restore`；只允许出现在负向回流守卫或不可变历史 evidence。Browser current command 变更必须同步 Electron Host/preload、Renderer typed gateway、App Server reverse-request/catalog、fixture、read model 与 `npm run test:contracts`。

## Review 主链

Desktop code review 只允许走 current `review/start`：

`src/lib/api/review.ts -> typed App Server client -> app_server_handle_json_lines -> App Server review/start -> RuntimeCore::start_review -> Thread/Turn/Item projection`

请求必须携带真实 `threadId` 和 typed `target`（`uncommittedChanges`、`baseBranch`、`commit` 或 `custom`）。App Server
拒绝 detached delivery；RuntimeCore 先检查 session/active turn，再规范化 target 字段并提交异步 turn。响应立即返回
`reviewThreadId` 与 `turn.status=inProgress`，review 结果通过同一 thread 的 canonical events/read model 回流 GUI。

review boundary 使用 `enteredReviewMode` / `exitedReviewMode` Extension Item，分别投影为 v2 `EnteredReviewMode` /
`ExitedReviewMode`，并在 turn terminal 前完成退出 item。Renderer 不扫描工作区猜测 review 状态，不创建第二套
review transcript，也不把 Codex TUI detached/background review 伪装成 Desktop 能力。

旧 review facade、raw `agentSession/event` review side-channel、detached/background 入口和生产 mock fallback 均为
`dead / deleted / forbidden-to-restore`；没有 compat/deprecated wrapper。该边界的最低验证是
`cargo test -p app-server processor::thread::projection::tests`、`cargo test -p app-server processor::tests::review`
和 `npm run test:contracts`。真实 Electron Gate B evidence 已建立于
`.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/code-artifact-workbench-electron-fixture-summary.json`，
证明 preload/IPC 命中 `app_server_handle_json_lines`、`review/start` 与 backend turn identity 绑定，GUI 可见终态与
内部 prompt 隔离，且无生产 mock fallback；不得用 TUI 或浏览器投影冒充该证据。

## Scheduled Tasks 主链

Desktop 已安排任务只允许走：

`ScheduledTasksPage -> src/lib/api/scheduledTasks.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server scheduledTask/* -> RuntimeCore -> LocalAppDataSource automation owner -> Thread/Turn/Item + Agent Run`

current method 为 `scheduledTask/list`、`scheduledTask/read`、`scheduledTask/create`、
`scheduledTask/update`、`scheduledTask/delete`、`scheduledTask/enabled/set`、
`scheduledTask/run/start`、`scheduledTask/run/list` 与 `scheduledTask/schedule/preview`。
Renderer 只持有筛选、选中项和编辑表单状态；任务、next run、revision 与运行历史由 App Server read model 提供。
`run/start` 必须经同一个 RuntimeCore execution service 创建或继续 canonical Thread，并提交真实 Turn；GUI 只在运行
返回真实 `sessionId` 时开放恢复对话。App Server 同时发布 typed `scheduledTask/changed`（create/update/delete/enabled-set）
和 `scheduledTask/run/updated`（canonical `turn.completed/failed/canceled` 终态、missed/catch-up/recovery 终态）；
Agent Run 使用 `finished_at IS NULL` 幂等门，事件重放不得重复终态通知。Renderer 通过全局 notification bridge 按
`all_runs / failures / none` 决定是否请求 Electron Desktop Host `show_desktop_notification`，Host 的 unsupported/failed
结果必须显示可见错误。任务删除写入 tombstone、禁用并清除未来调度，但不取消正在运行的 Turn；canonical terminal 写回
仍保留运行历史且不得复活 tombstone。Electron 继续只转发 JSONL 和系统通知壳能力，不新增任务 CRUD IPC、renderer timer 或第二调度器。

旧 `automationJob/*`、`automationSchedule/*`、`automationScheduler/*`、`src/lib/api/automation.ts` 与旧设置工作台已物理删除，
分类为 `dead / deleted / forbidden-to-restore`；旧 method 字符串只允许存在于 contract、Electron fixture 和治理扫描的负向
回流守卫中。`automation_jobs` 表、Rust `AutomationJob` DAO 与内部 execution helper 仍是 Scheduled Tasks 的 current 存储映射，
不构成第二公开协议或产品对象。scheduler 的原子 claim、24 小时休眠补跑、超窗 missed、DST、通知和删除并发合同已由 current
owner 收口。真实 OS sleep/wake、Windows Notification Center 和 Windows Gate B 仍是平台证据缺口。生产路径禁止 mock
fallback，测试专用 Rust backend 只能用于 public JSON-RPC fixture。

## Host Reverse Requests, Plan And Diff Notifications

`currentTime/read`、`item/permissions/requestApproval`、`item/tool/call` 使用同一 App Server server-request dispatcher：

```text
RuntimeCore waiter
  -> App Server JSON-RPC server-request
  -> Electron Desktop Host / PendingInteraction responder
  -> exact response id
  -> RuntimeCore continuation
```

`currentTime/read` 只能由 Electron Host 读取系统时钟，App Server 负责 thread scope、超时和响应校验；它不创建
Thread Item，也不提供 Renderer 时钟 API。`item/permissions/requestApproval` 只接受 tool-runtime 规范化后的
permission profile 和 canonical session/thread/turn/item/environment identity；统一 `PendingInteractionController`
只能返回 turn/session-scoped grant 或空 grant，不能扩大请求权限。`item/tool/call` 只能命中
`thread/start`/`thread/resume` 后冻结的 Desktop dynamic-tool binding，调用 identity、namespace、tool 和参数必须逐项
匹配；结果由 canonical DynamicToolCall Item 投影，Renderer 不得伪造 server request 或直接执行宿主能力。

`turn/plan/updated` 是 RuntimeCore `update_plan` producer 生成的 server notification，经 App Server v2 projector、
typed client 和 Renderer projection 进入同一 Thread/Turn/Item read model。计划 snapshot 的权威 owner 是 canonical
Plan Item；Renderer 本地 checklist 只做投影，不得替代 durable plan fact。

`turn.diff.updated` 是 RuntimeCore `apply_patch` coding event producer 在 Turn 范围内聚合精确 mutation 后生成的 durable
fact，经 App Server JSON-RPC projector 投影为 exact `turn/diff/updated { threadId, turnId, diff }`，再由 typed client
和 Renderer conversation reducer 写入 canonical Turn 的 `unified_diff`。Desktop Changes 的 previous-conversation 模式只
消费该字段；空字符串是有效 net-zero 清除信号，不得回退到本地 patch 拼装或第二份 diff store。Renderer 不承接 Codex
TUI 的 review surface，Electron 只做既有 Desktop Host JSONL 转发，不新增业务后端。

这条通知的 owner 是 App Server JSON-RPC + RuntimeCore durable event 链，不是 Electron IPC、旧 facade 或 provider。
多模型 catalog、model switch、provider capability/readiness、retry/circuit breaker 和多模态 sampling/media lowering
继续归 Grok-aligned `model-provider`；Codex 对齐只覆盖 Agent loop、Thread/Turn/Item、工具生命周期和 GUI 投影边界。

`turn.moderation_metadata` 是 trusted first-party Responses metadata producer 生成的 durable fact，经 App Server
JSON-RPC projector 投影为 exact
`turn/moderationMetadata { threadId, turnId, metadata }`。`metadata` 必须原样保持 JSON value；object、array、scalar 与
`null` 都有效，缺失字段或 wrapper 额外字段 fail closed。该事件不去重，每次更新都经 typed client signal router 写入
canonical Turn 的 `moderation_metadata`，Renderer 只做 last-write-wins 且不展示 raw JSON。Electron 不新增 IPC，Codex
TUI 忽略该通知的行为不复制为 Desktop UI；多模型与多模态控制面仍由 Grok-aligned `model-provider` 承接。

runtime diagnostics 与 command terminal interaction 只允许走 typed server notification：`runtime.warning` / `runtime.error`
由 App Server 分别投影为 `warning` / `error`，live 与 cold read 共用 durable event owner；`error.willRetry` 不直接生成
Turn terminal。`item/commandExecution/terminalInteraction` 只发送脱敏、bounded summary，并与 canonical
CommandExecution read model 合并。raw diagnostic side-channel、未脱敏 stdin/stdout 和 Renderer 自建 terminal history
均为 `dead / forbidden-to-restore`。

## Thread Revert 主链

Codex `thread/revert` 是 experimental 的 paginated history replacement，Lime current owner 只允许以下主链：

`AppServerClient.revertThread -> app_server_handle_json_lines -> App Server thread/revert -> RuntimeCore history replacement -> canonical Thread/Turn/Item projection`

Rust `app-server-client`、`packages/app-server-client`、v2 method/envelope、schema registry 和 generated TypeScript
必须同步 `ThreadRevertParams`、`ThreadRevertResponse`、`thread/reverted`。入口在 connection transport 边界检查
`initialize.capabilities.experimentalApi`；未声明时返回 `INVALID_REQUEST`，不得由 renderer 或 mock 绕过。每个 Thread
拥有独占串行 scope，活动 Turn 先复用现有 interrupt/cancel 流程，失败时保持原历史不变。

RuntimeCore 使用 append-only `history.rollback` replacement marker 重算 effective event stream，不截断旧 JSONL，
不回滚本地 workspace 文件；provider history、cold hydration、read model 和 turns/items cursor 都从同一 effective
stream 读取。响应保留原 Thread identity，`thread.turns` 返回空数组并携带分页回溯 cursor，成功后发送一次
`thread/reverted`。`thread/rollback` 属于 Codex deprecated surface，不恢复为新 GUI 主操作；在本地文件不回滚和真实
transport/Electron evidence 完成前不新增 GUI 入口。

该能力的定向证据必须覆盖 paginated replacement、metadata-only response、cursor、notification、missing-turn exact
error、transport experimental gate、active-turn interrupt、cold resume、重复 revert、provider prefix 保留与 workspace
文件不变。生产链禁止第二套历史存储、截断日志或 mock fallback。

## Config Control Plane 主链

Desktop 配置只允许走单一全局用户层：

`Settings/fixture -> AppServerClient.request(config/read|config/value/write|config/batchWrite) -> app_server_handle_json_lines -> App Server config processor -> lime_core config.yaml`

`config/read` 只返回 Lime 用户配置层；`cwd`、project-local layer、MDM/requirements layer 和非当前绝对
`filePath` 均 fail closed。写入必须携带当前版本，未知 key、版本冲突和无效 schema 均拒绝。Electron 只转发
`app_server_handle_json_lines`，不得恢复 `get_config` / `save_config` 或新增第二套配置业务后端。

Codex `configRequirements/read` 暂不属于 Lime Desktop 产品范围：仓库没有 MDM 或 `requirements/config.toml`
owner，不建立未消费的策略层。`config/mcpServer/reload` 同样 excluded，MCP 配置继续由
`mcpServer/list|create|update|delete` 与 `mcpServer/start|stop` 负责。

## Permission Profile 主链

Desktop 新回合的权限选择只允许走：

`Desktop access mode -> src/lib/api/permissionProfiles.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server permissionProfile/list -> turn/start { approvalPolicy, permissions } -> RuntimeRequest sandbox policy -> tool-runtime`

catalog 只公开 `:read-only`、`:workspace`、`:danger-full-access` 三个 Lime Desktop 内建 profile。Renderer 提交前必须
解析唯一且 `allowed=true` 的目标 profile；catalog 缺失、重复、禁止或形状非法时 fail closed。App Server 将 profile id
映射为 `read-only`、`workspace-write`、`danger-full-access` runtime sandbox policy，并在 Turn metadata 写入
`permissions` 与 `activePermissionProfile` provenance。`permissions + sandboxPolicy` 双传和未知 profile 必须拒绝。

Electron 不新增权限业务命令或本地 catalog；Renderer 不再从新回合 wire 发送 legacy `sandboxPolicy`。Lime Desktop
不复制 Codex TUI picker，也不读取 project-local `.codex/config.toml` 自定义 profile。`thread/settings/update.permissions`
尚未接入同一 resolver，保持 planned/fail-closed；不得用现有 list/Turn lowering 冒充 settings mutation 已完成。多模型
catalog、model switch、provider capability/readiness、retry/circuit breaker 与多模态 sampling/media lowering 继续归
Grok-aligned `model-provider`。
