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

current method 为 `mcpServer/list`、`mcpServerStatus/list`、`mcpServer/create`、`mcpServer/update`、`mcpServer/delete`、`mcpServer/enabled/set`、`mcpServer/importFromApp`、`mcpServer/syncAllToLive`、`mcpServer/oauth/login`、`mcpServer/oauthLogin/completed`、`mcpServer/startupStatus/updated`、`mcpServer/start`、`mcpServer/stop`、`mcpTool/list`、`mcpTool/listForContext`、`mcpTool/search`、`mcpTool/call`、`mcpTool/callWithCaller`、`mcpPrompt/list`、`mcpPrompt/get`、`mcpResource/list`、`mcpResource/read`、`mcpResource/subscribe` 与 `mcpResource/unsubscribe`。OAuth 完成态只允许由 App Server v2 typed notification `mcpServer/oauthLogin/completed` 投影给 Renderer；MCP startup lifecycle 只允许由 `mcpServer/startupStatus/updated` 投影连接态并触发终态刷新。旧 `mcp:oauth_completed`、`mcp:server_started`、`mcp:server_stopped`、`mcp:server_error` 均为 `dead / deleted / forbidden-to-restore`。事件 `mcp:resources_updated` 和 `mcp:resource_updated` 必须经真实 MCP manager / Desktop Host event bridge 投影；浏览器模式不得静默退回 mock event fallback。

live evidence 仅通过 `smoke:mcp-current -- --allow-live-provider` 显式开启，且需要 `LIME_MCP_LIVE_SERVER_URL`。该 URL 不得包含 username、password、query 或 hash；认证只能引用环境变量名，不允许 inline secret。`network-invoke.json` 仅可记录脱敏的 host、环境变量名、header 名、范围和工具/资源摘要。

MCP server-originated elicitation 使用独立 reverse JSON-RPC method `mcpServer/elicitation/request`。该 method 在 protocol catalog 中属于 `serverRequest`，不属于 Renderer 发起的 `AppServerRequestMethod`。App Server 生成 outer JSON-RPC id 并按 id 精确等待 Response/Error；Electron `app_server_drain_events` 只上行 notification/request，`app_server_handle_json_lines` 只把 Renderer 回包原样写回 sidecar。Renderer 必须通过 typed server-request dispatcher 注册 method handler；未知 method 返回 `METHOD_NOT_FOUND`。禁止暴露 MCP raw request id、按 server/turn/tool 扫描 waiter，或复用 `agentSession/action/respond`、Approval、`request_user_input` 与生产 mock fallback。

MCP model Tool surface 与 GUI 管理读必须分层：`tool-runtime::McpStepSnapshot` 只冻结同一次 provider sampling 的 tool definitions、caller policy、exact route 和 connection handle；`mcpPrompt/*`、`mcpResource/*`、`mcpServerStatus/list` 继续由 App Server 直接向 `lime-mcp::McpClientManager` 做 live read。禁止让管理面经过 model bridge、让 GUI inventory 替换 in-flight snapshot，或用 caller-unaware live registry dispatch 绕过当前 step allowlist。

旧 MCP Desktop facade 已统一归类为 `dead / retired guard-only`：`get_mcp_servers`、`mcp_list_servers_with_status`、`mcp_list_tools`、`mcp_list_prompts`、`mcp_list_resources`、`mcp_call_tool`、`mcp_start_server`、`sync_all_mcp_to_live` 只能出现在负向 guard 或历史 evidence，禁止回到前端网关、Desktop Host、mock 或 App Server current 主链。

## Browser Session 主链

浏览器会话检测、连接、读回、动作与关闭只允许走：

`src/lib/api/browserRuntime.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server browserSession/* -> BrowserRuntimeManager`

Settings 的浏览器页只消费 `browserSession/target/list`、`browserSession/open`、`browserSession/read` 与 `browserSession/close`；Renderer 只展示带 debugger endpoint 的 `page` target。旧 connector install、Chrome relay endpoint、backend priority 与静态 Electron diagnostic facade 不得回到 Settings 产品面。Browser Workspace 尚未迁完的旧 facade 属于 PAGE-08 blocker，不能作为 Settings 或 Browser Runtime current evidence。
