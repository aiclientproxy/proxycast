# Codex 对齐重构 v1

状态：执行中（2026-07-26）。产品范围完成度 `69 / 180 = 38.3%`（见
[11-codex-method-product-scope-matrix.md](11-codex-method-product-scope-matrix.md)）；
[04-execution-plan.md](04-execution-plan.md) 的六条完成定义尚无一条全满足。

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
| [09-tool-turn-snapshot-progress.md](09-tool-turn-snapshot-progress.md)                                               | V1-05 Tool/Hook turn snapshot 骨架、写集避让与接线退出条件 |
| [10-item-inventory-skeleton.md](10-item-inventory-skeleton.md)                                                       | V1-04 Codex v2 ThreadItem 变体/字段/增量通知清单           |
| [../../test/snapshots/README.md](../../test/snapshots/README.md)                                                     | Codex 663 个 `.snap` 的场景映射与 Gate A/B 实现状态        |
| [../../exec-plans/codex-alignment-v1-coordination-plan.md](../../exec-plans/codex-alignment-v1-coordination-plan.md) | 多进程窄写集、交接顺序、删除闸门和统一验证                 |
| [../data/01-storage-alignment-plan.md](../data/01-storage-alignment-plan.md)                                         | 只对照实际 `~/.codex` 的存储职责、平台根和分阶段方案       |
| [../data/03-one-to-one-storage-alignment-plan.md](../data/03-one-to-one-storage-alignment-plan.md)                   | Codex 56 项、Lime AppData 63 项、`~/.lime` 12 项的一一账本 |

## 参考快照

| 参考仓库                                    | commit                                     | 允许借鉴                                                                                                    |
| ------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `/Users/coso/Documents/dev/rust/codex`      | `c4f42d161ae44a8d696ee9fb595709661979d187` | runtime、App Server、Thread/Turn/Item、工具、MCP、Skills、Plugins、Multi-Agent、恢复和测试语义              |
| `/Users/coso/Documents/dev/rust/grok-build` | `6e386420825bd44ae648c63e7c8cba12fcec9401` | model control plane：catalog、model selection/switch、capability、tool subset、retry/circuit breaker        |
| `/Users/coso/Documents/dev/js/opencode`     | `fab213312927ea64cf968832c527206e8c944f9e` | provider wire plane：endpoint union、canonical content/lowering、媒体、协议 stream reducer、provider policy |

三个 commit 均为对应参考仓库当前 HEAD，是本主线唯一的 reference lock。

### Revision lock 口径

同一主线只能有一个 reference lock；下游 fixture 和切片记录允许暂时落后，但必须显式登记：

| 位置                                                                                | 当前 revision | 口径                                                                 |
| ----------------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------- |
| 本表、`internal/test/snapshots/**`、`fixtures/codex-method-product-scope.v0.1.json` | `4c43465…`    | 已对齐 lock                                                          |
| `fixtures/item-inventory.v0.1.json`、[09](09-tool-turn-snapshot-progress.md)        | `9fc715c0…`   | 落后 lock；升级必须重新执行 Item/tool 审计，禁止只改 hash 让守卫通过 |
| [04](04-execution-plan.md) 各切片记录、`internal/exec-plans/**` 历史证据            | 按当轮记录    | 历史 evidence，保留写入时的 revision，不回改                         |

`internal/research/refactor/v2/**` 与 `internal/exec-plans/project-gate-a-b-acceptance-plan.md`
是各自独立的基线，不受本 lock 约束。

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

切片编号以 [04-execution-plan.md](04-execution-plan.md) 的 `V1-xx` 为准（旧 `P0-0x` 编号已废弃，
不再使用）。开工前先建立的三项主链前提均已落地：v2 protocol current owner 已建立并删除 v0
lifecycle DTO；缺 provider/model route 已 fail closed 且不再让 App Server warmup 退出；
`lime-providers` 已物理删除，所有协议统一经 `model-provider` lowering。

已收口（closed）：V1-16 rollout 追加热路径、V1-17 canonical Turn hydration、V1-18 侧边栏 Thread
查询与首页 Gate B、V1-19 EventLog/projection schema、V1-20 Reasoning 可见性与 Skills 初始化、
V1-21 Reasoning lifecycle 重复分段、V1-22 Plan typed delta、V1-23 CommandExecution outputDelta、
V1-24 FileChange patchUpdated、typed `artifact/write` 与旧 runtime append 删除。

进行中：

1. `V1-25` MCP Tool Call progress 生命周期。request-token correlation 与 canonical `itemId` 已完成，
   剩余退出条件是 `smoke:agent-runtime-current-fixture` 与 `verify:gui-smoke` 复跑，以及
   `tests/legacy_permission_surfaces.rs` 这一 `dead-candidate` 的删除确认。
2. `V1-04` Item inventory：下一刀是 MCP result/error typed shape 或 Command terminal interaction 的
   真实 Runtime source，不在 Renderer 伪造字段。
3. `V1-05` Tool/Hook turn snapshot：`RuntimeTurnSnapshot` 仍是注册快照骨架，sampling step 接线未做。
   本轮新增 `tool-runtime/src/hook_lifecycle.rs` 作为 Hook 生命周期**裁决** owner：block/abort/rewrite/
   inject 四类终态、按 display order 聚合、稳定 `run_id`，并对 `MissingDecision`、未实现 handler、
   不可信/被改写来源、非法 matcher、非 Sync 执行模式全部 fail closed（与 `try_new` 同口径）。
   定向验证 10/10。仍缺 discovery producer、真实 handler 执行、canonical Item 投影与 App Server
   notification，因此 `V1-05` 保持 `alignment-open`。
   同轮新增 `tool-runtime/src/hook_runtime.rs` 承接 discovery 与 `Command` 执行（单一 Codex 事件
   分组配置格式、跨平台 shell、stdin 上下文、超时、blocking 语义、stdout 结构化结果）。两模块定向
   25/25，crate 全量 286/286。旧 `agent/src/hooks.rs` 与 `pub mod hooks` 已物理删除，回流守卫为
   `rust-retired-agent-hook-manager`；旧扁平 `{hooks:[...]}` 格式、已退役事件名与 `async_exec` 归类为
   `dead / deleted / forbidden-to-restore`，不保留双轨。
4. `V1-08`–`V1-10` 多模型控制平面：优先补 `model/rerouted`、`model/verification` 与 provider
   readiness/retry 的结构化 evidence。
5. `V1-07` Skills/Plugins/Apps watcher/readiness 与 Hook lifecycle。

每个切片必须在协调计划中登记实际写集、Codex 参考路径、定向验证和 OPEN_REF；只有 Gate A/B 通过后才执行旧 `v0`、`agentSession` 和重复 provider owner 的物理删除。

## 当前阻塞

- **P0 协议阻塞**：Thread/Turn/Item lifecycle 已切 direct v2，旧 lifecycle DTO/schema 已删除；剩余阻塞是 approval/runtime-events 等 `agentSession/*` side-channel、完整 v2 server request/item inventory 和旧 test-only canonical wrapper 清理。
- **P0 history 阻塞**：ThreadStore raw canonical append、独立 metadata patch、ThreadHistoryBuilder coalesce/rollback 尚未证明。Codex compaction replacement lineage **已不在此列**：`runtime/context_compaction.rs` 已实现并校验 `replacementHistory` 与 `windowNumber/firstWindowId/previousWindowId/windowId` 链，非法 lineage fail closed，且有 9 个 owner 测试与冷重启 public JSON-RPC replay（`thread_fork_compaction_jsonrpc.rs::compacted_thread_fork_replays_replacement_and_surviving_tail_after_restart`）；剩余只是 `history-compaction-replacement` 的三条 Gate B fixture。
- **P0 provider 阻塞**：`lime-providers` 已物理删除且禁止恢复；剩余阻塞是 provider capability/credential/route preflight、durable default，以及把 provider/runtime 私有字段迁出 Codex `additionalContext`。
- **P1 transport 阻塞**：stdio/ws/unix、逐连接 notification filtering 与 slow-client 已有实现和定向测试；剩余阻塞是 Windows transport 语义、真实 reconnect/overload Gate B 和产品范围确认。
- **P1 lifecycle 阻塞**：Item 字段级 inventory、hook/deferred tool、MCP immutable snapshot、Skills/Plugins/Apps watcher 和 environment/config-lock 仍缺 contract。

本轮已获授权删除旧路径和直接重构，不保留长期兼容层。迁移期间只允许短期编译适配；完成后必须物理删除 `protocol/v0`、`agentSession/*` production surface、`lime-providers` 和未实现的 transport 声明，并由 [07-second-audit-gap-register.md](07-second-audit-gap-register.md) 的扫描守卫阻止回流。
