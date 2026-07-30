# Lime 第二轮重构 v2

> status: historical research baseline
> owner: runtime-architecture
> last_verified: 2026-07-18
> codex_commit: `2e4f55608b4ad26d9c48ea45a6fcd20bfd5e9fe8`
> opencode_commit: `08fb47373509ba64b13441061314eeacf4264f51`
> scope: research, architecture, cleanup plan

## 目标

v2 是研发期的清场方案，不是 v1 文档的续写。仓库没有外部用户和发布兼容负担，因此默认直接迁移、直接删除、直接替换。只有当前构建图仍然需要的能力才保留；历史名称、旧命令、旧状态机、旧 projection、旧 fixture 和旧 adapter 不因为“以后可能有用”而继续存在。

Lime 的产品形态是 Electron + React GUI。Codex 只提供 Agent runtime、协议、状态和测试体系的可复制原点；Codex TUI 的布局、输入法、终端渲染和 CLI 产品入口不属于 Lime 目标。OpenCode 只补两类信息：

1. 多模型、多模态的 provider/model/capability/content/lowering 代数。
2. monorepo 中 schema、domain、client、server、UI package 的 owner 和依赖方向。

固定产品链：

```text
React GUI
  -> typed gateway
  -> Electron Desktop Host / preload
  -> App Server JSON-RPC
  -> agent-runtime / RuntimeCore
  -> model-provider + tool-runtime
  -> Thread / Turn / Item materialization
  -> thread-store / ProjectionStore / read model
  -> GUI projection / evidence
```

## 动作分类

| 动作     | 含义                                                                    | 默认处理                                                    |
| -------- | ----------------------------------------------------------------------- | ----------------------------------------------------------- |
| `copy`   | Codex 代码、测试或协议可原样/近原样迁入 Lime current owner              | 先做许可证和依赖审计，再保留上游 provenance；不另写同义实现 |
| `adapt`  | 语义可复制，但必须适配 Rust crate、GUI、Electron、provider 或多模态边界 | 只允许一个适配 owner，适配后删除临时双轨                    |
| `delete` | 旧路径没有用户/持久化/发布约束，或已脱离构建图                          | 删除入口、实现、catalog、正向 fixture，并加负向回流守卫     |
| `watch`  | 上游变化暂不影响 Lime current 主链                                      | 只记版本、路径和触发条件，不进入实现 backlog                |

`compat` 不是 v2 的默认动作。若发现兼容层仍被 current 消费，先把调用迁到唯一 owner，再删除兼容层；不能用新的 wrapper 延长寿命。

## 文档索引

| 文档                                                                    | 作用                                                                |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [00-decisions/record.md](./00-decisions/record.md)                      | v2 裁决、非目标和清理原则                                           |
| [01-current-facts/snapshot.md](./01-current-facts/snapshot.md)          | 代码和上游仓库的可复核快照                                          |
| [02-codex/current-map.md](./02-codex/current-map.md)                    | Codex current 模块、可复制实现和拒绝项                              |
| [03-opencode/module-map.md](./03-opencode/module-map.md)                | OpenCode package owner、依赖方向和允许参照边界                      |
| [04-target/architecture.md](./04-target/architecture.md)                | Lime GUI 目标架构和数据流                                           |
| [05-boundaries/ownership.md](./05-boundaries/ownership.md)              | crate/package/host 的唯一 owner 与禁依赖                            |
| [06-protocol/contracts.md](./06-protocol/contracts.md)                  | method、scope、schema、notification 和 client 契约                  |
| [07-runtime/lifecycle.md](./07-runtime/lifecycle.md)                    | Thread/Turn/Item、队列、工具、取消、恢复状态机                      |
| [08-projection/read-model.md](./08-projection/read-model.md)            | materialization、read model、GUI projection 和分页                  |
| [09-provider/capability.md](./09-provider/capability.md)                | provider/model/capability/content/lowering 设计                     |
| [10-tool-context/policy.md](./10-tool-context/policy.md)                | 工具、审批、sandbox、context、skills、multi-agent                   |
| [11-gui/electron-gates.md](./11-gui/electron-gates.md)                  | GUI 分层、桌面适配和 Gate A/B                                       |
| [12-plan/slices.md](./12-plan/slices.md)                                | 可执行重构切片、删除清单和退出条件                                  |
| [历史多进程实施计划](../../../exec-plans/refactor-v2-implementation.md) | 已完成旧广义切片的协调与 evidence，不再定义当前渲染 owner           |
| [当前渲染对齐计划](../../../refactor/v2/IMPLEMENTATION-PLAN.md)         | ConversationProjection、ItemRenderer 与 V2-00 至 V2-05 当前执行状态 |
| [13-evidence/verification.md](./13-evidence/verification.md)            | 验证矩阵、证据格式和完成定义                                        |
| [14-upstream/ledger.md](./14-upstream/ledger.md)                        | Codex/OpenCode 版本账本和 allowlist 规则                            |
| [15-cleanup/inventory.md](./15-cleanup/inventory.md)                    | v1 残留和 Lime 旧路径的清理盘点                                     |
| [16-v1-crosswalk.md](./16-v1-crosswalk.md)                              | v1 丰富文档到 v2 current owner 的逐文件迁移对照                     |
| [99-archive/policy.md](./99-archive/policy.md)                          | 历史文档归档和断链防护                                              |

## v2 完成定义

一轮切片只有同时满足以下条件才算完成：

1. 唯一 current owner、输入输出契约和依赖方向已写入边界表。
2. 可复制的 Codex 实现已经复制或明确记录了不能复制的具体原因。
3. 旧入口、旧实现、旧正向 fixture 已删除；保留的唯一内容是负向回流守卫或不可变 evidence。
4. 有 protocol/schema、行为单测、runtime fixture 和 GUI Gate A/B 中与风险匹配的证据。
5. 正常、失败、中断、排队、过期事件、恢复和分页路径均有可验证退出条件。

## 历史收尾位置

S5ag 的 Workspace owner 拆分已完成：`Entry -> Setup -> Command -> Scene` 均已落到独立 current owner，主 runtime 仅保留无条件组合。该研究树和旧多进程计划只保留为 dated evidence；当前下一刀由 `internal/refactor/v2/IMPLEMENTATION-PLAN.md` 定义，不得用旧 `agentSession/*` owner、compat wrapper 或生产 mock fallback 覆盖 current 计划。
