# Plugin v3 标准化与清理路线图

状态：`in-progress / current-owner-established`

更新时间：2026-08-09

## 主目标

让 Lime 成为一个行为上与 Codex 同步、符合 Agent Plugins v1.0.0 portable package
规范、架构上仍由 Lime current owner 承载的客户端。v3 不保留 Lime 私有包标准的长期
兼容，不为旧实现增加包装层，不以历史数据或旧文档阻止标准化。

唯一产品链：

```text
Electron Desktop Host
  -> App Server JSON-RPC
  -> RuntimeCore / Skills / MCP / Hooks
  -> Thread / Turn / Item projection
  -> App Center / Claw / Right Surface
```

## 外部基准优先级

1. Agent Plugins v1.0.0：portable package、manifest、固定组件位置、MCP schema、路径和
   placeholder 语义。
2. `/Users/coso/Documents/dev/rust/codex`：客户端加载、Codex extension、失败隔离、安全
   校验和回归测试语义。
3. Lime 架构与 current owner：App Server、RuntimeCore、tool-runtime、Skills、MCP、Thread/
   Turn/Item、Right Surface 和 Desktop Host 的边界。

规范优先于 Codex 私有字段；Codex 私有扩展只能放在明确 namespace 或显式 adapter 中，
不得污染 portable manifest。

## 当前进度

| 阶段 | 状态 | 当前事实 |
|---|---|---|
| V3-0 | `complete` | 标准合同、唯一 owner、删除口径和架构确认已落盘 |
| V3-1 | `complete` | 根 `plugin.json`、直接子目录 Skills、根 `mcp.json` loader 已进入 App Server current owner |
| V3-2 | `complete on macOS/Unix / Windows gap` | MCP lowering、安全校验、失败隔离和 activation snapshot 已实现；Unix parity 已有独立测试，Windows 文件系统语义仍待真实 runner |
| V3-3 | `complete` | `plugin/*` protocol/client/Renderer catalog 与标准包真实 Agent turn + Right Surface Gate B 已完成 |
| V3-4 | `complete` | 旧 package API、发布链、SDK、renderer runtime 和 worker 已物理删除 |
| V3-5 | `complete` | 旧 `PluginManager`、processor hook、孤立 DAO/schema/error surface 已物理删除 |
| V3-6 | `in-progress` | macOS 标准包 Gate B 已通过；等待 parity、Windows 与最终全量门禁 |

当前下一刀是完成 Windows 路径矩阵与 V3-6 最终分层门禁，
不再继续扩展任何旧 Plugin 命名或兼容入口。

## v3 包合同

```text
plugin-root/
├── plugin.json
├── skills/
│   └── <skill>/SKILL.md
└── mcp.json
```

- 根 `plugin.json` 必须包含官方 `$schema` 和 `name`。
- Skills 只扫描 `skills/` 直接子目录。
- MCP 只读取根 `mcp.json`，manifest 不得内联或重定向 MCP/Skills 位置。
- `mcp.json` 必须使用官方 `$schema`、`mcpServers`、`stdio` 或 `streamable-http` 语义；
  legacy `sse` 输入必须 fail closed，不属于 v3 支持传输。
- stdio 必须提供持久化 `PLUGIN_ROOT` 和 `PLUGIN_DATA`，实现官方 placeholder 与路径
  containment。
- `.codex-plugin/plugin.json` 只作为显式 Codex 私有扩展 adapter，不是 portable owner。
- 未知顶层字段只报告并忽略；旧 Lime 私有字段不参与任何发现、安装或激活语义。

## 禁止回流

以下内容不进入 v3 current：

- `lime.plugin.package.v1`、`schemaVersion`、`contributions.runtime/workbench`。
- 以 `manifest.json` 为入口的旧 PluginManager/PluginLoader。
- `app.runtime.yaml`、独立 Plugin worker、独立 UI runtime 和 renderer registry。
- `pluginLocalPackage/*`、`pluginPackage/*` 作为消费端运行时入口。
- renderer 扫描插件目录、合并 marketplace 或维护第二份 installed/readiness 状态。

上述生产实现均已删除；这些名称只允许出现在 retired guard、负向测试或历史 evidence。

## 完成定义

- 标准根包可以被发现、读取、安装、启用并进入真实 Agent turn。
- Codex parity fixture 覆盖 manifest、Skills、MCP、路径、安全和失败隔离。
- App Server 是唯一 catalog、installed、activation 和 runtime capability 事实源。
- 旧 Lime manifest、旧 package projection、旧 worker、旧协议、旧前端事实源和孤立
  `installed_plugins` DAO/schema 的生产引用全部清零并删除。
- 旧 `lime-server -> lime-processor -> lime-core::plugin` 钩子链不在构建图；不得恢复。
- 通过 macOS/Windows、Rust/contract、真实 Electron Gate B 和 legacy 回流守卫。

## 文档导航

- [当前基线](./00-current-baseline.md)
- [目标合同](./01-target-contract.md)
- [清理账本](./02-cleanup-ledger.md)
- [实施计划](./03-execution-plan.md)
- [验证合同](./04-verification.md)
- [Codex parity 与平台矩阵](./05-codex-parity-matrix.md)
