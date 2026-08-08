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

## Apps Catalog 主链

Apps/connectors 只允许走同一个 Plugin catalog owner：

`src/lib/api/apps.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server app/* -> RuntimeCore -> PluginDataSource -> local plugin_catalog`

current method 为 `app/list`、`app/read`、`app/installed` 与 `app/list/updated`。`app/list` 从已安装 Plugin manifest 的
`apps` capability 构建分页 catalog；`app/read` 最多接收 100 个 id，去重并保持首次请求顺序，未知 id 放入
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

## Memory Reset 主链

全局记忆重置只允许走：

`src/lib/api/memoryStore.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server memory/reset -> RuntimeCore -> MemoryAppDataSource -> LocalMemoryBackend`

current method 为 exact `memory/reset`，只接受 omitted、`null` 或空对象 params，返回空对象。它只清理全局 memory
root 并重建受管目录，不删除 Thread/Turn/Item、event log、projection store 或 memory root 外的 soul 配置。旧 scoped
`memoryStore/reset`、`MemoryStoreResetParams/Response`、typed client 与设置页调用均为
`dead / deleted / forbidden-to-restore`；workspace memory reset 不再作为未被产品消费的平级公开能力保留。

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

## Browser Session 主链

浏览器会话检测、连接、读回、动作与关闭只允许走：

`src/lib/api/browserRuntime.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server browserSession/* -> BrowserRuntimeManager`

Settings 的浏览器页只消费 `browserSession/target/list`、`browserSession/open`、`browserSession/read` 与 `browserSession/close`；Renderer 只展示带 debugger endpoint 的 `page` target。旧 connector install、Chrome relay endpoint、backend priority 与静态 Electron diagnostic facade 不得回到 Settings 产品面。Browser Workspace 尚未迁完的旧 facade 属于 PAGE-08 blocker，不能作为 Settings 或 Browser Runtime current evidence。
