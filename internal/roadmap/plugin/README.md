# Lime Plugin 路线图

更新时间：2026-08-08

状态：`Plugin v3 in progress / v2 frozen as historical reference`

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
7. [v2 历史路线图](./v2/README.md)

## Legacy 声明

本目录根部旧 PRD、架构、接口、原型、证据和实施跟踪文件保留为 `historical reference`，用于回看产品洞察、交互方案、历史决策和既有证据。它们不再是 current 设计、实现或验收事实源，也不得独立继续演进。

旧 `lime.plugin.package.v1`、旧根 `plugin.json` projection、`contributions.runtime/workbench`、
插件专用 worker、旧 renderer registry 和旧 package API 均已按 v3 清理账本删除。
历史文档可以保留为 evidence，但不再是实现依据；实现只允许引用 v3 合同。

## 历史参考导航

### 产品与设计

- [旧版 PRD](./prd.md)
- [旧版架构](./architecture.md)
- [旧版接口合同](./interface-contracts.md)
- [旧版技术基线](./technical-baseline.md)
- [旧版交互原型说明](./prototype.md)与[静态原型](./prototype.html)
- [历史工作区与恢复设计](./history-product-workspace.md)
- [旧版用户操作指南](./user-operations-guide.md)

### 历史跟踪与证据

- [旧版实施计划](./implementation-plan.md)
- [旧版 E2E 证据](./e2e-evidence.md)
- [历史 E2E 摘要](./evidence/plugin-productization-e2e-summary.json)
- [旧发布中心 PRD](./deverlop/plugin-publish-center-prd.md)
- [旧发布服务端计划](./deverlop/plugin-publish-limecore-server-plan.md)

从历史文档复用的有效决策必须先写入 v3 对应合同并通过 current owner 审核，不能让实现直接同时依赖旧路线图与 v3。
