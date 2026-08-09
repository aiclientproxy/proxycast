# Plugin v3 目标合同

状态：`normative target / implementation-active`

## Manifest

标准 loader 只接受根目录 `plugin.json`，并按官方 schema 处理：

- 必填 `$schema`：`https://agent-plugins.org/schemas/1.0.0/plugin.schema.json`。
- 必填 `name`，遵守官方长度、字符集、重复符号约束。
- `version` 可选；不得强制 semver。
- 顶层字段闭合；未知字段只能报告并忽略。
- `extensions` 是唯一 client-specific namespace；未知 namespace 不得解释。
- manifest 解析失败时拒绝整个插件，不能继续发现组件。

## 固定组件发现

```text
skills/  -> 直接子目录/<SKILL.md>
mcp.json -> 根 JSON 文档
```

manifest 不得声明 `skills`、`mcpServers`、`apps` 或任意路径来覆盖固定位置。缺失组件是
合法的“没有该能力”；组件错误只禁用该组件类型，不影响其它组件。

## Codex Apps extension

Portable manifest 仍不得出现顶层 `apps`。Codex client extension 可通过
`extensions.com.openai.apps`，或在 inline extension 缺失时通过
`.codex-plugin/plugin.json` 的 `apps`，声明一个包内相对 JSON 路径。该文件按 Codex
`apps.{name}.{id,category?}` 形状解析：connector `id` 是 Apps catalog identity，name 是
展示名。旧内联 Apps object 必须拒绝，非法/缺失 Apps 配置只禁用 Apps 组件；不得覆盖
portable `name`、`version`、Skills 或 MCP 固定位置。

## MCP

标准 parser 与内部 `McpServerConfig` 分离：先严格解析官方 `mcp.json`，再 lowering 到
`lime-mcp` 内部类型。

- `stdio`：裸 executable 或 `./` 包内 command；`args`、`env` value、`cwd` 才允许
  `${PLUGIN_ROOT}` / `${PLUGIN_DATA}` 单次展开。
- `cwd`：只能是 `./`、`${PLUGIN_ROOT}` 或 `${PLUGIN_DATA}` 下的 containment 路径。
- 启动时注入绝对、filesystem-resolved `PLUGIN_ROOT` 和持久化 writable `PLUGIN_DATA`。
- 禁止插件覆盖两个保留环境变量。
- `streamable-http`：绝对 HTTP/HTTPS URL；非 loopback 必须 HTTPS；拒绝 userinfo 和
  fragment；`headers` 只能是 package 中的字面量，不能携带 secret。
- legacy `sse` transport 必须 fail closed，不得进入 runtime。
- 单个 server 失败只禁用该 server；顶层 mcp 文件错误只禁用 MCP 组件，不影响 Skills。

## Codex parity

Codex parity 的范围是行为和测试：manifest format selection、direct-child Skills、MCP
normalization、Apps extension path/config、path containment、reserved env、failure isolation、
installed/activated state 和 reload/cold restore。Codex 私有内部类型、存储和 UI 不复制到
Lime。

## Lime integration

- App Server 负责 discovery、install、installed、enabled、activation snapshot。
- RuntimeCore/agent-runtime 负责 turn context；tool-runtime 负责 MCP/tool lifecycle。
- Renderer 只消费 typed JSON-RPC projection。
- Electron 只做 Host 能力和 JSONL 转发。
- Right Surface 只渲染 canonical resource/result，不执行 plugin worker。

## 当前实现锚点

- Package loader/store：`lime-rs/crates/app-server/src/local_data_source/plugin_catalog.rs`。
- 标准 MCP parser/lowering：`lime-rs/crates/mcp/src/agent_plugin_config.rs`。
- JSON-RPC：`plugin/list`、`plugin/search`、`plugin/read`、`plugin/install`、
  `plugin/uninstall`、`plugin/installed`、`plugin/enabled/set`。
- Renderer gateway：`src/lib/api/pluginCatalog.ts`；不包含包解析、worker 或本地 installed state。

标准包真实 Agent turn/Right Surface Gate B 已在 macOS arm64 通过。实现尚未满足的合同是
完整 Codex parity matrix、Windows 路径行为证据与最终全量门禁；这些缺口不得通过 compat
或 mock 填补。

## 发布边界

Plugin consumer runtime 不继续承载旧 `pluginLocalPackage/*`、`pluginPackage/*` 发布后台。
这些方法要么迁到独立发布工具，要么从 Lime 删除；不得作为 v3 loader 的隐式输入。
