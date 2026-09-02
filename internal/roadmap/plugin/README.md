# Lime Plugin 路线图

更新时间：2026-08-08

状态：`Plugin v3 in progress / current owner established`

## 当前实现与目标路线

当前实现已迁到 [v3/README.md](./v3/README.md) 定义的 Agent Plugins v1.0.0 portable
package owner。v3 以标准规范和 Codex 的行为/测试语义为外部基准；旧 Lime 私有标准、
worker、package API、renderer runtime 与旧 manager 已删除，不做长期双读、双写或自动转换。

```text
Electron Desktop Host
  -> App Server JSON-RPC
  -> RuntimeCore / MCP / Skills / Hooks
  -> Thread / Turn / Item projection
  -> App Center / Claw / Right Surface
```

App Center 参考 Codex 的信息架构和状态语义，但继续使用 Lime 当前主题。Claw 通过 `@plugin` 调用能力，并在唯一 Right Surface 中承载 MCP/App UI、Browser、结构化结果和文件预览。

v3 的目标包结构固定为：

```text
plugin-root/
├── plugin.json
├── skills/<skill>/SKILL.md
└── mcp.json
```

`.codex-plugin/plugin.json` 只允许作为显式 Codex 私有扩展适配，不得成为 v3 portable
manifest owner；`lime.plugin.package.v1`、`manifest.json` 和 `contributions.*` 均不得进入
v3 current 主链。

## 阅读顺序

1. [v3 总览](./v3/README.md)
2. [当前基线](./v3/00-current-baseline.md)
3. [目标合同](./v3/01-target-contract.md)
4. [清理账本](./v3/02-cleanup-ledger.md)
5. [实施计划](./v3/03-execution-plan.md)
6. [验证合同](./v3/04-verification.md)

## 历史边界

旧 Lime 私有标准、worker、package API、renderer runtime、旧 manager 以及 v1/v2 文档已退出 active tree。
需要追溯时使用 Git history；新的决策和实现只允许写入 v3 合同并通过 current owner 审核。
