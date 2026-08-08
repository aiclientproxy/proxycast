# Plugin v3 当前基线

状态：`implemented-baseline / 2026-08-08`

## 结论

Lime 当前已经建立 Agent Plugins v1.0.0 package current owner。标准 loader、installed
store、activation snapshot、Skills/MCP lowering 和 `plugin/*` App Server contract 已落地；
macOS 标准包真实 Agent turn/Right Surface Gate B 已通过。剩余工作是 Codex parity 独立
证据、Windows 路径矩阵与最终全量门禁。

## 当前 owner 分类

| 分类 | 路径/能力 | 处置 |
|---|---|---|
| `current` | `lime-rs/crates/app-server/src/local_data_source/plugin_catalog.rs` | 标准 package discovery/install/installed/enabled/activation 唯一 owner |
| `current` | `lime-rs/crates/app-server/src/local_data_source/impls/plugins.rs`、App Server v2 `plugin/*` | JSON-RPC 与 typed projection 唯一产品边界 |
| `current` | `lime-rs/crates/mcp/src/agent_plugin_config.rs`、`lime-rs/crates/skills`、`agent-runtime`、`tool-runtime` | 标准 MCP/Skills lowering 与 turn/tool 生命周期 owner |
| `current` | Renderer `pluginCatalog`、Claw mention、Right Surface | 只消费 App Server projection，不解析包 |
| `compat` | Codex `.codex-plugin/plugin.json` extension adapter | 只适配 `com.openai` UI metadata；不成为 portable owner，不解释 Lime 私有字段 |
| `dead` | 旧 package API/发布链、renderer SDK/runtime、Electron worker/UI runtime | 已删除并由 contract/governance guard 阻止回流 |
| `dead` | `lime-core::plugin`、processor Plugin hook、`installed_plugins` DAO/schema、孤立 Plugin errors | 已删除，不保留构建或存储双轨 |
| `dead` | MCP smoke 私有 runtime inventory fixture 与 `agentChat.harness.pluginMcpTargets.*` 文案 | 已删除；current inventory 明确忽略私有 metadata 且不投影私有 targets |
| `historical reference` | `internal/roadmap/plugin` 根部旧 PRD、v1/v2 历史证据 | 只读，不再作为实现依据；完成 v3 后可归档或删除 |

## 关键证据

- `plugin_catalog.rs` 只以根 `plugin.json` 为 portable manifest，并要求 v1.0.0 schema。
- `plugin_catalog/tests.rs` 覆盖标准安装、digest 幂等/冲突、optional version、direct-child
  Skills、Codex extension precedence、symlink、根 `mcp.json` 和 enabled snapshot。
- `agent_plugin_config.rs` 覆盖 `stdio` / `streamable-http`、placeholder、保留环境变量、
  cwd containment、URL 安全与 sibling failure isolation。
- `agentCommandCatalog.json` 只登记 `plugin/list|search|read|install|uninstall|installed|enabled/set`。
- `runtime_backend_tool_inventory_does_not_project_plugin_private_targets` 固化私有 metadata
  不得进入 current inventory；MCP smoke 只验证标准 MCP JSON-RPC surface。
- `npm run governance:legacy-report` 当前结果为边界违规 `0`、分类漂移候选 `0`。
- `.lime/qc/gui-evidence/plugin-package-electron-gate-b/plugin-package-electron-gate-b-summary.json`
  证明 macOS arm64 标准包 install/enable/turn/MCP/Right Surface/cold restore/uninstall
  历史闭环，canonical mention 为 `plugin://mcp-elicitation-plugin@local`，生产 mock 命中为零。

## 不允许的判断

- 不能把 Codex extension adapter 称为 portable manifest owner。
- 不能把本地 Plugin installed/enabled 状态冒充 MCP tool/provider readiness。
- 不能恢复旧 loader、worker、SDK、UI runtime、发布后台或 `installed_plugins` 数据双轨。
- 不能用 browser mirror、非标准包 fixture 或 mock 代替已定义的标准包 Electron Gate B。
