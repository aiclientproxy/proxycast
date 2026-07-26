# Codex 对齐重构 v1

状态：执行中（2026-07-19）

本目录是 Lime 对齐本地 Codex 的执行方案。目标不是复制 Codex 的 CLI、TUI 或 ChatGPT 专属产品，而是把 Codex 已验证的 runtime 语义、可恢复状态、App Server 协议、工具生命周期和多 Agent 控制面收敛到 Lime current owner。

多模型/provider 采用分层参考：以 `/Users/coso/Documents/dev/rust/grok-build` 作为模型控制平面的 primary reference（目录、选择、切换、能力、重试/熔断），以 `/Users/coso/Documents/dev/js/opencode` 作为 provider wire 平面的 secondary reference（endpoint、canonical content、lowering、媒体和多协议 stream）。两者都不负责 Lime 的 Thread/Turn/Item、App Server、Agent loop 或 GUI owner。

## 目标边界

```text
Electron Desktop Host
  -> App Server JSON-RPC
  -> RuntimeCore / agent-runtime
  -> Thread/Turn/Item + EventLog + ThreadStore
  -> model-provider / tool-runtime
  -> typed projection / GUI / evidence
```

Codex 对齐的是上图从协议到恢复的语义；grok-build 对齐的是 `model-provider` 内部的 model control（route、capability matrix、catalog、model switch），OpenCode 补充 provider-neutral content、endpoint/lowering 和多协议 stream。任何新能力必须落入已有 owner，不得建立第二套 runtime、history、模型路由或 GUI 状态机。

## 文件索引

| 文件                                                                                                                 | 用途                                                       |
| -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [01-comparison-matrix.md](01-comparison-matrix.md)                                                                   | Codex 领域逐项对照、Lime owner、状态和缺口                 |
| [02-multi-model-grok-build.md](02-multi-model-grok-build.md)                                                         | grok-build 多模型/provider 设计拆解与 Lime 裁决            |
| [06-grok-vs-opencode.md](06-grok-vs-opencode.md)                                                                     | grok-build 与 OpenCode 的逐维度比较和最终取舍              |
| [03-target-architecture.md](03-target-architecture.md)                                                               | 终态分层、数据流、依赖方向和禁止路径                       |
| [04-execution-plan.md](04-execution-plan.md)                                                                         | P0-P4 分阶段执行计划、写集和退出条件                       |
| [05-verification-and-guardrails.md](05-verification-and-guardrails.md)                                               | 测试、Gate B 证据、治理扫描和回流守卫                      |
| [07-second-audit-gap-register.md](07-second-audit-gap-register.md)                                                   | 第二轮查缺、P0 阻塞项、删除顺序和回流守卫                  |
| [08-third-audit-gap-register.md](08-third-audit-gap-register.md)                                                     | 第三轮协议、恢复、provider protocol 与产品范围补充审计     |
| [11-codex-method-product-scope-matrix.md](11-codex-method-product-scope-matrix.md)                                   | Codex App Server method 三态产品范围矩阵与守卫             |
| [../../exec-plans/codex-alignment-v1-coordination-plan.md](../../exec-plans/codex-alignment-v1-coordination-plan.md) | 多进程窄写集、交接顺序、删除闸门和统一验证                 |
| [../data/01-storage-alignment-plan.md](../data/01-storage-alignment-plan.md)                                         | 只对照实际 `~/.codex` 的存储职责、平台根和分阶段方案       |
| [../data/03-one-to-one-storage-alignment-plan.md](../data/03-one-to-one-storage-alignment-plan.md)                   | Codex 56 项、Lime AppData 63 项、`~/.lime` 12 项的一一账本 |

## 参考快照

| 参考仓库                                    | commit                                     | 允许借鉴                                                                                                    |
| ------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `/Users/coso/Documents/dev/rust/codex`      | `2e4f55608b4ad26d9c48ea45a6fcd20bfd5e9fe8` | runtime、App Server、Thread/Turn/Item、工具、MCP、Skills、Plugins、Multi-Agent、恢复和测试语义              |
| `/Users/coso/Documents/dev/rust/grok-build` | `98c3b2438aa922fbbe6178a5c0a4c48f85edc8ce` | model control plane：catalog、model selection/switch、capability、tool subset、retry/circuit breaker        |
| `/Users/coso/Documents/dev/js/opencode`     | `fab213312927ea64cf968832c527206e8c944f9e` | provider wire plane：endpoint union、canonical content/lowering、媒体、协议 stream reducer、provider policy |

## 分类口径

- `current`：Lime 当前唯一 owner，允许继续演进。
- `partial`：已有 owner，但与参考语义或验证证据不完整；按 current owner 补齐，不新建平级实现。
- `missing`：尚无可用 current owner，必须先定义 owner 再实现。
- `wrong owner`：能力存在，但落在错误边界；迁移后删除旧入口。
- `compat`：只允许外部协议或一次性数据迁移适配，不得承接新业务逻辑。
- `deprecated`：只允许迁出和删除，并写退出条件。
- `dead`：无入口或被 current 替代，删除并加回流守卫。

## 总裁决

1. P0 先完成 Codex 的 canonical state、App Server、持久化和恢复闭环；没有这个闭环，多模型切换只能产生不可恢复的旁路状态。
2. P1 完成工具、sandbox、approval、MCP、Skills、Plugins 和 Apps 的工具生命周期闭环。
3. P2 把 grok-build 的模型控制设计和 OpenCode 的 provider wire 机制接入同一个 `model-provider` owner，模型选择结果必须在每个 Turn 固化并进入 read model/evidence。
4. P3 完成 Multi-Agent graph、identity、mailbox、fork、wait 和真实 Electron 证据对齐。
5. P4 再补 CLI/SDK/TUI 等消费面；这些是 App Server 的客户端，不得反向改变 runtime owner。

## 当前执行顺序

本轮先完成三项能解除后续迁移风险的 Codex 主链工作：

1. `P0-01`：建立 `app-server-protocol/src/protocol/v2/**` current owner，直接复制 Codex v2 的 Thread/Turn/Item wire contract、method registry、server request/notification 和 round-trip tests；不再向 `protocol/v0` 增加新字段。
2. `P0-05`：恢复缺少 provider/model route 时返回可重试的 typed pending 状态，保留 AgentControl graph/mailbox，不让 App Server warmup 退出。
3. `P0-06`：未实现 provider protocol 在联网前 fail closed，所有已支持协议通过唯一 `model-provider` lowering。

每个切片必须在协调计划中登记实际写集、Codex 参考路径、定向验证和 OPEN_REF；只有 Gate A/B 通过后才执行旧 `v0`、`agentSession` 和重复 provider owner 的物理删除。

## 当前阻塞

- **P0 协议阻塞**：Thread/Turn/Item lifecycle 已切 direct v2，旧 lifecycle DTO/schema 已删除；剩余阻塞是 approval/runtime-events 等 `agentSession/*` side-channel、完整 v2 server request/item inventory 和旧 test-only canonical wrapper 清理。
- **P0 history 阻塞**：ThreadStore raw canonical append、独立 metadata patch、ThreadHistoryBuilder coalesce/rollback 和 Codex compaction replacement lineage 尚未证明。
- **P0 provider 阻塞**：`lime-providers` 已物理删除且禁止恢复；剩余阻塞是 provider capability/credential/route preflight、durable default，以及把 provider/runtime 私有字段迁出 Codex `additionalContext`。
- **P1 transport 阻塞**：stdio/ws/unix、逐连接 notification filtering 与 slow-client 已有实现和定向测试；剩余阻塞是 Windows transport 语义、真实 reconnect/overload Gate B 和产品范围确认。
- **P1 lifecycle 阻塞**：Item 字段级 inventory、hook/deferred tool、MCP immutable snapshot、Skills/Plugins/Apps watcher 和 environment/config-lock 仍缺 contract。

本轮已获授权删除旧路径和直接重构，不保留长期兼容层。迁移期间只允许短期编译适配；完成后必须物理删除 `protocol/v0`、`agentSession/*` production surface、`lime-providers` 和未实现的 transport 声明，并由 [07-second-audit-gap-register.md](07-second-audit-gap-register.md) 的扫描守卫阻止回流。
