# Plugin v2 架构与命令合同

状态：`implementation in progress`

## 唯一产品链

```text
Electron Desktop Host
  -> app_server_handle_json_lines
  -> App Server plugin JSON-RPC
  -> Plugin catalog/install owner
  -> Skills / MCP / Hooks activation
  -> RuntimeCore Thread/Turn/Item
  -> Renderer projection
  -> App Center / Claw / Right Surface
```

Electron 只负责 desktop host、文件选择、窗口/webContents 和受控原生能力；它不解析 marketplace、不决定 plugin readiness，也不运行第二套 plugin task runtime。

## Owner 分工

| Owner                    | 责任                                                    | 禁止承担                      |
| ------------------------ | ------------------------------------------------------- | ----------------------------- |
| `app-server-protocol`    | plugin/marketplace JSON-RPC schema、notification        | 文件扫描、安装实现            |
| App Server plugin domain | discovery、read、install、update、uninstall、projection | provider request、UI 状态拼装 |
| app paths/repository     | 跨平台目录、installed store、原子写入                   | 运行时工具执行                |
| Skills owner             | Skill discovery、namespace、selection、prompt fragment  | marketplace 安装              |
| MCP owner                | server lifecycle、OAuth、tools/resources/apps           | plugin catalog UI             |
| Hooks owner              | hook config、trust、lifecycle enforcement               | 任意插件私有 worker           |
| RuntimeCore              | thread/turn snapshot、工具与审批、item projection       | 安装包文件操作                |
| Renderer gateway         | typed JSON-RPC client、notification subscription        | manifest 解析、registry 合并  |
| App Center               | 目录、详情和生命周期操作                                | catalog 事实源                |
| Claw Right Surface       | UI/result projection                                    | 独立业务后端                  |

## 协议方法

Plugin v2 采用下列 current 方法：

```text
marketplace/add
marketplace/remove
marketplace/upgrade
plugin/list
plugin/installed
plugin/read
plugin/install
plugin/uninstall
plugin/enabled/set
```

`plugin/enabled/set` 是 Lime 需要的明确扩展，用于表达本地启停；其语义不得混入 install/uninstall。

后续按真实需求再加入：

```text
plugin/update
plugin/share/save
plugin/skill/read
```

不要把旧 `pluginInstalledSave`、`pluginLocalPackageInspect`、`pluginUiRuntimeStart` 等方法机械改名后继续保留同一实现。

## 核心 DTO

### PluginSummary

```text
id
remotePluginId?
name
version?
localVersion?
source
installed
installedAt?
enabled
installPolicy
installPolicySource?
authPolicy
availability
disabledReason?
interface?
keywords[]
```

### PluginDetail

```text
marketplaceName
marketplacePath?
summary
description?
skills[]
hooks[]
apps[]
mcpServers[]
screenshots[]
sourceDiagnostics[]
```

### Source authority

资源 locator 必须携带 owner：

```text
Bundled { releaseId, relativePath }
Workspace { workspaceId, relativePath }
Local { marketplaceId, relativePath }
Installed { pluginId, digest, relativePath }
Remote { remotePluginId, releaseId, resourceId }
```

跨 App Server 边界不暴露可被 Renderer 任意读取的绝对路径。需要展示来源时返回 sanitized label 和可审计 source descriptor。

## Discovery 与缓存

App Server 每次 list 以 workspace roots、configured marketplaces、release bundle 和 remote policy 为输入，生成 source-aware projection。

缓存必须以以下维度键控：

```text
workspace roots
marketplace snapshot versions
installed store revision
workspace/admin policy revision
remote catalog revision
```

不得只按 plugin name 缓存，也不得由 Renderer localStorage 充当 installed fact。

## 安装时序

```text
Renderer plugin/install
  -> App Server resolve marketplace entry
  -> installer staging + validation
  -> optional confirmation/authorization request
  -> atomic store update
  -> activate component descriptors
  -> pluginsChanged notification
  -> Renderer refresh list/detail
```

安装请求必须包含 marketplace identity 与 plugin name，不能只传 URL 或任意文件路径。导入本地目录时先建立显式 local marketplace/source authority，再走同一安装流程。

## Runtime 装配

### Thread snapshot

启动或恢复 thread 时，RuntimeCore 获取当前 enabled plugins snapshot：

- plugin identity/version/digest
- enabled Skills roots
- MCP server/app declarations
- trusted Hooks
- policy and auth readiness

snapshot 写入 thread/runtime metadata，只保存 marker 和 identity，不保存完整 Skill body、secret 或 MCP token。

### Turn

- 显式 `@plugin` mention 进入 turn input 的结构化 selection。
- Skills selector 可从 enabled plugin skill roots 发现候选。
- MCP tool registry 只暴露已启动、已授权且 policy 允许的 server tools。
- Hooks 在 tool/runtime owner 的统一 lifecycle 上执行。
- 安装/启停变化不修改正在运行的 turn tool schema。

### MCP current 装配

Plugin v2 的 MCP 声明已进入统一 runtime owner：

```text
LocalAppDataSource
  -> list_mcp_runtime_server_specs
  -> RuntimeBackend::ensure_thread_mcp_runtime_if_available
  -> AgentRuntimeState::ensure_mcp_runtime
  -> McpThreadRuntime
  -> McpClientManager
```

当前合同如下：

- manifest `mcpServers` 可以指向包内 JSON 文件，也可以直接声明 inline object；manifest 未声明时发现 package root 的 `.mcp.json`。
- `.mcp.json` 同时接受 `{ "mcpServers": {...} }` 和直接 server map，并复用 `lime_mcp::McpServerConfig` 解析。
- stdio server 默认 `cwd` 是 installed package root；显式相对或绝对 `cwd` canonicalize 后必须仍位于该 root 内。
- disabled Plugin 不生成 runtime spec；单个无效 server fail closed，但不丢弃有效 sibling。
- runtime server name 固定为 `plugin__<plugin-id>__<server-id>`，tool name 继续由统一 MCP owner 投影为 `mcp__<runtime-server-name>__<tool-name>`。
- 用户 MCP 与 Plugin MCP 重名时保留用户配置并跳过冲突的 Plugin server。
- activation descriptor 记录可审计的 `runtimeCapabilities.mcpServers`，但 `mcpBindings` 不伪造 wildcard tool binding；具体工具只能在 server initialize/list tools 后进入 snapshot。

该实现不新增 Plugin worker、第二套 MCP manager 或私有 IPC。Apps 与 Hooks 的 current lifecycle 装配仍待完成。

### Item projection

tool item 必须保留：

```text
pluginId?
skillName?
mcpServerName?
toolName
callId
approvalId?
surfaceDescriptor?
status
```

这样 Claw timeline、Right Surface 和历史恢复使用同一 identity。

## Right Surface 合同

插件可以通过标准 tool result 声明 surface：

```text
McpAppResource { server, resourceUri, csp, initialState }
BrowserTarget { url, browserFamily?, ownership }
StructuredResult { rendererKind, dataRef }
FilePreview { pathRef, mimeType }
```

Renderer 只根据受支持 descriptor 打开已有 Right Surface owner。插件不能注册任意 React component import、任意 iframe src 或 Electron IPC channel。

UI 内的动作回到 MCP tool call/elicitation/approval；不建立 `pluginUiRuntimeStart -> plugin worker -> custom event` 私有闭环。

## Auth 与权限

```text
Plugin installed
  != Connector authorized
  != MCP ready
  != Tool allowed
  != External write approved
```

App Server projection 分别返回这些状态。授权由 MCP/OAuth owner 执行，secret 由统一凭证 owner 保存；approval 由 tool-runtime/agent-runtime 执行。

## Notifications

至少需要：

```text
plugin/changed
marketplace/changed
mcpServer/statusUpdated
mcpServer/oauthLoginCompleted
skills/changed
```

通知只表示 revision 改变，Renderer 收到后重新读取 projection；不要把整份 catalog 通过事件重复广播。

## 错误模型

错误按 owner 分类：

- `marketplace_*`：来源不可达、格式错误、冲突
- `package_*`：manifest、路径、digest、archive、版本
- `policy_*`：管理员禁用、计划不满足、产品不支持
- `auth_*`：未授权、scope 不足、登录失败
- `runtime_*`：MCP/Hook/Skill activation 失败
- `surface_*`：UI resource/CSP/renderer 不支持

错误必须包含稳定 code、sanitized message、retryability 与 recovery action；不向 Renderer 泄露 secret 或未清理的本地绝对路径。

## 架构删除线

以下方向禁止进入 Plugin v2：

- Renderer -> filesystem 直接扫描 plugin package
- Renderer -> Electron plugin worker 作为业务后端
- App Center registry 与 RuntimeCore 各维护一份 enabled plugins
- 插件包注册任意 IPC/capability 名称
- 生产路径回退 mock capability host
- 为旧 `lime.plugin.package.v1` 新增 compat wrapper
- 用 `pluginId@tenant` 字符串拼装代替结构化 source identity

## 架构确认项

实现阶段的责任开发者需要在执行计划与 PR 描述确认：

- [ ] App Server 是 catalog/installed 唯一事实源
- [ ] RuntimeCore 使用同一 plugin identity snapshot
- [ ] Right Surface 没有第二套业务后端
- [ ] 旧 manifest 和 worker 入口已迁出或删除
- [ ] macOS/Windows 路径走统一平台 owner
- [ ] Gate B trace 能证明 current JSON-RPC 与用户可见状态
