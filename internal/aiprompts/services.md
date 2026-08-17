# 服务层

## 概述

Lime 的后端业务事实源已经从旧 `lime-rs/src/services/**` 收敛到 App Server workspace crates、RuntimeCore、domain crates 与 Electron Desktop Host 壳能力。

新增业务逻辑不要恢复 `lime-rs/src/**`，也不要把 Electron main / preload 当第二套后端。默认路径是：

`前端 API 网关 -> App Server JSON-RPC -> lime-rs/crates/** domain owner`

Electron Desktop Host 只承接桌面壳能力，例如窗口、IPC、系统浏览器、文件选择、托盘、updater 和 sidecar 生命周期。

## Current owner 地图

| 能力                                             | current owner                                                 | 入口                                                            |
| ------------------------------------------------ | ------------------------------------------------------------- | --------------------------------------------------------------- |
| MCP server / tools / prompts / resources / OAuth | `lime-rs/crates/mcp` + App Server `mcp*` processor / protocol | `src/lib/api/mcp.ts`                                            |
| Agent session / turn / read model / evidence     | `lime-rs/crates/app-server` runtime 与 `lime-rs/crates/agent` | `src/lib/api/agentRuntime/**`                                   |
| Provider / model registry / API key              | App Server `modelProvider*` / `modelProviderKey*` methods     | `src/lib/api/modelRegistry.ts`、`src/lib/api/apiKeyProvider.ts` |
| Scheduled Tasks / Agent Run history              | App Server `scheduledTask/*` + RuntimeCore + scheduler        | `src/lib/api/scheduledTasks.ts`                                 |
| Gallery materials                                | current gallery material command family                       | `src/lib/api/galleryMaterials.ts`                               |
| Document export                                  | Electron Desktop Host 本地文件壳能力                          | `src/lib/api/document-export.ts`                                |
| Browser connector / remote runtime               | Desktop Host bridge + App Server current methods              | `src/lib/webview-api.ts`、`remote-runtime.md`                   |

如果一个能力没有出现在表里，先查 `commands.md`、`governance.md` 和对应领域文档，确认唯一事实源后再改代码。

## MCP 服务边界

MCP 不再是旧单体 `McpService` 或 desktop-host mock surface。当前 MCP 主链固定为：

`src/lib/api/mcp.ts -> AppServerClient.request(...) -> app_server_handle_json_lines -> App Server JSON-RPC -> lime-rs/crates/mcp`

要点：

- server 配置、stdio / streamable HTTP transport、OAuth、tools/prompts/resources 都由 `lime-rs/crates/mcp` 管理。
- App Server protocol 暴露 `mcpServer/*`、`mcpTool/*`、`mcpPrompt/*`、`mcpResource/*` methods。
- runtime 工具命名固定为 `mcp__<server>__<tool>`。
- 旧 `get_mcp_servers`、`mcp_list_*`、`mcp_call_tool*`、`add_mcp_server` 等 Desktop facade 是 `dead / retired guard-only`。
- desktop-host 默认 mock 不得为 MCP 返回空成功；缺真实通道必须 fail closed。

详细规则见 `mcp.md` 与 `commands.md` 的 “MCP 控制面主链”。

## 新增服务规则

1. 先确认是否已有 App Server method / domain crate 可承接。
2. 前端只从 `src/lib/api/*` 网关进入，不在组件或普通 Hook 里散落裸 `invoke`。
3. App Server protocol、processor、client、治理目录册和 mock/test fixture 必须成组更新。
4. 生产路径不得依赖 `defaultMocks`、`mockPriorityCommands`、`invokeMockOnly` 或 renderer mock fallback。
5. 旧 `compat` 层只允许委托、适配、告警；不能新增业务逻辑。

## 校验入口

服务层或命令边界改动至少运行：

```bash
npm run test:contracts
```

影响 GUI 壳、Workspace、DevBridge 或主页面路径时补：

```bash
npm run verify:gui-smoke
```

Rust 变更先跑受影响 crate / 模块定向测试；从仓库根运行时必须带 workspace manifest：

```bash
cargo test --manifest-path "lime-rs/Cargo.toml" -p <crate> <filter>
```

## 相关文档

- [commands.md](commands.md) - Desktop Host / App Server 命令边界
- [mcp.md](mcp.md) - MCP 服务器与工具管理
- [governance.md](governance.md) - current / compat / deprecated / dead 分类
- [quality-workflow.md](quality-workflow.md) - 校验门槛
