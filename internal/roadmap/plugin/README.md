# Lime Plugin 路线图

更新时间：2026-08-04

状态：`Plugin v2 proposed / implementation-ready`

## 当前事实源

Plugin 当前路线图统一进入 [v2/README.md](./v2/README.md)。v2 直接采用 Codex-compatible `.codex-plugin/plugin.json`、marketplace、Skills、MCP servers/apps 与 Hooks 模型，并固定以下生产主链：

```text
Electron Desktop Host
  -> App Server JSON-RPC
  -> RuntimeCore / MCP / Skills / Hooks
  -> Thread / Turn / Item projection
  -> App Center / Claw / Right Surface
```

App Center 参考 Codex 的信息架构和状态语义，但继续使用 Lime 当前主题。Claw 通过 `@plugin` 调用能力，并在唯一 Right Surface 中承载 MCP/App UI、Browser、结构化结果和文件预览。

## 阅读顺序

1. [研究结论](./v2/00-research-findings.md)
2. [产品合同](./v2/01-product-contract.md)
3. [包、市场与安装](./v2/02-package-marketplace-installation.md)
4. [架构与命令合同](./v2/03-architecture-and-command-contracts.md)
5. [App Center 与 Claw Surface](./v2/04-app-center-and-claw-surfaces.md)
6. [迁移与清理账本](./v2/05-migration-and-cleanup.md)
7. [实施计划](./v2/06-implementation-plan.md)
8. [Gate B 验收合同](./v2/07-verification-contract.md)

## Legacy 声明

本目录根部旧 PRD、架构、接口、原型、证据和实施跟踪文件保留为 `historical reference`，用于回看产品洞察、交互方案、历史决策和既有证据。它们不再是 current 设计、实现或验收事实源，也不得独立继续演进。

旧 `lime.plugin.package.v1`、根 `plugin.json`、`contributions.runtime/workbench`、插件专用 worker 和 renderer registry 仍处于 `deprecated`，不得新增调用或能力；只有 current 主链迁移和 Gate B 验收完成后才能删除。文档保留与旧实现退役是两个独立治理维度，详见 [迁移与清理账本](./v2/05-migration-and-cleanup.md)。

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

从历史文档复用的有效决策必须先写入 v2 对应合同并通过 current owner 审核，不能让实现直接同时依赖 v1 与 v2。
