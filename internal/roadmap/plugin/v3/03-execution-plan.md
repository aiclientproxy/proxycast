# Plugin v3 实施计划

状态：`in-progress`

## 写集纪律

每阶段开始前在 `internal/exec-plans/` 更新执行计划，声明精确写集、脏热区避让、
current/deprecated/dead 变化、架构影响、验证命令和阻塞。协议、App Server plugin
domain、MCP/Skills、Renderer gateway、清理守卫不能在同一阶段由不同变更重写同一事实源。

## 阶段

| 阶段 | 目标 | 主要写集 | 退出条件 |
|---|---|---|---|
| V3-0 | `complete` | v3 文档、governance scanner、执行计划 | 标准合同、删除分类和架构确认已落盘 |
| V3-1 | `complete` | `plugin_catalog.rs`、MCP adapter、fixtures | 根 `plugin.json` + direct-child `skills/` + 根 `mcp.json` 已由 current loader 读取 |
| V3-2 | `in-progress` | MCP lowering、PLUGIN_ROOT/DATA、path/security tests、activation snapshot | 实现已落地；仍需独立 Codex parity matrix 与 Windows evidence |
| V3-3 | `complete` | protocol、typed client、plugin gateway、Claw、Right Surface | typed `plugin/*` 与标准包真实 Electron install/turn/tool/restore Gate B 已完成 |
| V3-4 | `complete` | package API、发布脚本、SDK/renderer/worker | 旧 protocol/client/bridge/scripts/fixtures 正向引用已清零并删除 |
| V3-5 | `complete` | processor/core plugin manager 与孤立 storage/error | 旧 manager/build hook/DAO/schema/error 已物理删除 |
| V3-6 | `in-progress` | parity evidence、文档、guards、全量验证 | parity、macOS/Windows、`verify:local` 全部通过 |

## 不可跳过的删除顺序

```text
标准 parser + negative guard
  -> 迁移 App Server/GUI/fixture
  -> 清零旧 protocol 和 scripts
  -> 清零发布与 server 构建入口
  -> 物理删除旧 package/worker/manager
  -> 更新架构和路线图事实源
  -> final legacy report + Gate B
```

任何阶段失败都只能修复 current owner；不得恢复旧实现作为 fallback。

## 架构确认点

`internal/aiprompts/architecture.md` 已更新 Plugin owner、标准 package boundary、Codex
extension boundary、MCP 安全边界和旧实现删除状态；责任开发者确认已于 2026-08-08 记录。

## 下一执行刀

1. 固化并逐项执行 [Codex parity matrix](./05-codex-parity-matrix.md)：manifest、Skills、MCP、
   installed/enabled、reload/cold restore；实现存在但未达 Codex 测试粒度的行不得标绿。
2. 补 Windows junction/reparse point、环境变量大小写与 root/data 行为证据。
3. 执行 contracts、治理、Rust related、Agent fixture、GUI smoke 与 `verify:local` 最终门禁。

macOS 标准包 Gate B 已于 2026-08-08 通过，证据位于
`.lime/qc/gui-evidence/plugin-package-electron-gate-b/plugin-package-electron-gate-b-summary.json`。
