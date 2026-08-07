# MCP 服务器与工具管理

## 事实源

MCP 管理、发现、工具调用、提示词、资源和订阅控制面只允许继续收敛到：

`src/lib/api/mcp.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server JSON-RPC -> lime-rs/crates/mcp`

Electron Desktop Host 只负责 renderer bridge、事件转发、系统浏览器打开和 sidecar 生命周期，不承接 MCP 业务事实；desktop-host 默认 mock 不再提供 MCP fallback。

MCP 有两个不能混用的 current owner：GUI management read（server/status/resource/prompt）继续使用 `LocalAppDataSource` 的 management-only `McpClientManager`；Agent runtime MCP connection 必须按 Session/Thread 持有 immutable `threadId`。management manager 不拥有 Thread，也不允许 server-originated elicitation。`mcpServer/tool/call` 必须从 canonical Thread 解析 Session，并经 `ExecutionBackend -> AgentRuntimeState -> McpThreadRuntime` 执行；缺少可信 Thread owner 时直接拒绝，禁止从最近活动 Turn、singleton、`sessionId`、`parentToolCallId`、progress token 或 server metadata 猜测归属。

## 当前结构

```text
src/lib/api/
├── mcp.ts                 # 前端 MCP API 网关，唯一 renderer 调用入口
├── mcpTypes.ts            # 公开类型与 mcp__<server>__<tool> 命名 helper
└── mcpResponseGuards.ts   # App Server response fail-closed guard

src/hooks/
├── useMcp.ts              # MCP runtime 状态与事件刷新
└── useMcpEvents.ts        # MCP resource Desktop events 与 App Server typed notification bridge

src/components/mcp/
├── McpPage.tsx            # 设置页入口
├── McpPanel.tsx           # runtime 面板 facade
├── McpServerList.tsx      # server runtime 列表 facade
├── McpToolsBrowser.tsx    # tools browser facade
├── McpPromptsBrowser.tsx  # prompts browser facade
└── McpResourcesBrowser.tsx # resources browser facade

lime-rs/crates/
├── mcp/                   # MCP manager、stdio / streamable HTTP、OAuth、tools/prompts/resources
├── app-server-protocol/   # mcpServer* / mcpTool* / mcpPrompt* / mcpResource* schema
└── app-server/            # JSON-RPC processor、runtime projection、evidence/read model 接线

scripts/mcp/
├── current-smoke.mjs      # MCP current smoke 入口
├── live-provider-smoke.mjs
└── lib/                   # smoke transport、fixture、contract guard helper
```

## App Server methods

Current MCP method 固定为：

- `mcpServer/list`
- `mcpServerStatus/list`
- `mcpServer/create`
- `mcpServer/update`
- `mcpServer/delete`
- `mcpServer/enabled/set`
- `mcpServer/importFromApp`
- `mcpServer/syncAllToLive`
- `mcpServer/oauth/login`
- `mcpServer/oauthLogin/completed`
- `mcpServer/startupStatus/updated`
- `mcpServer/start`
- `mcpServer/stop`
- `mcpServer/resource/read`
- `mcpServer/tool/call`
- `mcpTool/list`
- `mcpTool/listForContext`
- `mcpTool/search`
- `mcpPrompt/list`
- `mcpPrompt/get`
- `mcpResource/list`
- `mcpResource/subscribe`
- `mcpResource/unsubscribe`

`mcpServer/resource/read` 的 `threadId` 可选；携带时只读取对应 Session-owned runtime，未携带时才走 management manager。`mcpServer/tool/call` 的 `threadId` 必填。Settings 管理面不提供工具执行按钮；Workspace/Agent 消费者只能传真实 Thread identity。旧 `mcpTool/call`、`mcpTool/callWithCaller` 与 `mcpResource/read` 为 `dead / deleted / forbidden-to-restore`；protocol catalog、schema、App Server、typed clients、Renderer 和 smoke 均不得再提供正向入口。

旧 Desktop facade 已归类为 `dead / retired guard-only`，包括 `get_mcp_servers`、`mcp_list_*`、`mcp_call_tool*`、`mcp_get_prompt`、`mcp_read_resource`、`mcp_start_server`、`mcp_stop_server`、`add_mcp_server`、`update_mcp_server`、`delete_mcp_server`、`toggle_mcp_server`、`import_mcp_from_app`、`sync_all_mcp_to_live`。这些名字只能出现在负向测试、contract forbidden snippet、smoke legacy 黑名单或历史 evidence 中。

## Lifecycle notification

- `mcpServer/start` 在调用 RuntimeCore 前发布 `mcpServer/startupStatus/updated { threadId: null, name, status: "starting", error: null, failureReason: null }`。
- 启动成功发布 `ready`；启动失败先发布带 typed error 的 `failed`，再保留原 JSON-RPC error。`cancelled` 与 `reauthenticationRequired` 保留为 Codex 对齐的严格协议值，当前 app-scoped start producer 不伪造它们。
- Renderer 必须同步订阅 App Server typed event bus；`starting` 投影 server connection phase，`ready` / `cancelled` / `failed` 刷新 `mcpServerStatus/list` 与 `mcpTool/list`，`failed` 显示连接错误。
- `mcp:server_started`、`mcp:server_stopped`、`mcp:server_error` 与 `mcp:oauth_completed` 均为 `dead / deleted / forbidden-to-restore`。`mcp:tools_updated`、`mcp:resources_updated`、`mcp:resource_updated` 仍由各自 current owner 承接，不得与 startup notification 混成第二生命周期 owner。

## Server-Originated Elicitation

`mcpServer/elicitation/request` 是 App Server `serverRequest`，不是 Renderer `AppServerRequestMethod`。公开 request 只包含必填非空 `threadId`、可空 `turnId`、必填非空 `serverName` 和 typed `mode: "form"`；`sessionId`、`parentToolCallId`、私有 scope token 和 raw MCP request id 都禁止跨 JSON-RPC 边界。

它是瞬时 reverse request：App Server 以 outer JSON-RPC id 精确等待 Response/Error，`serverRequest/resolved` 只回到创建 request 的同一 connection，且先于 exact domain waiter continuation。它不得生成 Thread/Turn/Item、read model 或 durable store 写入。Renderer 在主窗口根部用全局 GUI Modal 消费 typed form；不得复用 TUI、Approval、通用 answer control 或生产 mock。

## Transport 与 auth

- `stdio` 与 `streamable_http` 都由 `lime-rs/crates/mcp` 管理。
- HTTP header 只允许来自配置中的安全字段或环境变量引用；inline secret、非法 header、缺失 env var 必须 fail closed。
- OAuth 登录走 `mcpServer/oauth/login`，使用系统浏览器 current 网关打开授权 URL；callback 成功或失败后由 App Server 发布 `mcpServer/oauthLogin/completed` typed notification，Renderer 经 typed event bus 刷新 server/tool 状态。旧 `mcp:oauth_completed` Desktop event 为 `dead / deleted / forbidden-to-restore`。
- OAuth token store 使用 app data runtime 下的 versioned credential envelope，按 server name + URL 隔离。
- 显式 `oauth.client_id` / `oauth_resource` 目前仍按 unsupported fail-closed，不能伪装成可用。

## Runtime 工具命名

MCP runtime 工具命名唯一事实源：

- 工具全名：`mcp__<server>__<tool>`
- extension surface key：`mcp__<server>`
- UI 展示名：优先显示 server 原名
- deferred 工具通过 `ToolSearch` 时优先使用 `select:mcp__<server>__<tool>`

不要恢复裸 `server__tool`、临时 `server_tool` 或 inventory / mock / GUI 各自一套前缀的旧心智。

## Resources / prompts / evidence

- `mcpResource/list` 同时返回 `resources` 与 `resourceTemplates`；读取使用 exact `mcpServer/resource/read -> contents[]`。
- `mcpResource/subscribe` / `mcpResource/unsubscribe` 使用 MCP 标准 resource subscription。
- 标准通知 `notifications/resources/list_changed` 与 `notifications/resources/updated` 通过 `mcp:resources_updated` / `mcp:resource_updated` 事件刷新 GUI。
- GUI resource preview 必须截断大文本、只展示 image/blob 摘要，不能把完整 base64 或大正文压进 DOM。
- `ReadMcpResourceTool` 的 grounding 进入 evidence/export 的 `observability_summary.mcp_resource_reads`，只记录 server、URI 摘要、mime、content refs 等可回放元数据，不复制 resource 正文或 blob。

## 校验入口

最低校验按改动范围选择：

```bash
npx vitest run "src/lib/api/mcp.test.ts" "src/lib/api/mcp.failClosed.test.ts" "src/hooks/useMcp.test.tsx"
npm run test:contracts
npm run verify:gui-smoke
```

MCP smoke：

```bash
npm run smoke:mcp-current
npm run smoke:mcp-current -- --allow-write-fixture
npm run smoke:mcp-current -- --allow-oauth-fixture
npm run smoke:mcp-oauth-notification-electron-fixture
npm run smoke:mcp-startup-notification-electron-fixture
npm run smoke:mcp-current -- --allow-live-provider
```

`--allow-live-provider` 只能在提供 `LIME_MCP_LIVE_SERVER_URL` 和凭证环境后运行；缺 env 必须在 DevBridge / App Server 调用前 fail closed，不能把无凭证环境伪造成 live provider 成功。

## 相关文档

- [commands.md](commands.md) - MCP 控制面主链、旧命令 dead 分类、工具命名主链
- [quality-workflow.md](quality-workflow.md) - GUI / command / bridge 校验门槛
- [governance.md](governance.md) - current / compat / deprecated / dead 分类语言
- [../exec-plans/mcp-modernization-progress.md](../exec-plans/mcp-modernization-progress.md) - 当前 MCP 现代化进度与缺口
