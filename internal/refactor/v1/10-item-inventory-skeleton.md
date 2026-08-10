# V1-04 Item Inventory 骨架

> status: `mcp-progress-current / inventory-open`
> owner: `app-server-protocol` + `thread-store` + `agent-runtime-projection`
> upstream: `/Users/coso/Documents/dev/rust/codex@9fc715c0861c956c894a91890b78dc05b304ba29`
> fixture: `internal/refactor/v1/fixtures/item-inventory.v0.1.json`

本清单与 fixture 落后 `README.md` 的 reference lock（`4c43465...`）一个 revision。升级前必须按
lock revision 重新对照 `codex-rs/app-server-protocol/src/protocol/v2/item.rs` 审计 variant、字段与
增量通知，再同步 fixture 的 `upstream.revision`；只改 hash 让守卫通过属于伪对齐。

## 目标

持续维护 Codex v2 `ThreadItem` 变体、字段和增量通知的可审计清单，并随 current owner
逐刀补齐 protocol、App Server、typed client、Renderer 与验证证据。

## 当前结论

- Codex 与 Lime 共享 18 个顶层 variant；`MemoryCitation` 仅是 `AgentMessage` 的嵌套字段，
  不能成为第二套 item 生命周期。
- Lime 当前已接入 `item/started`、`item/completed`，以及 AgentMessage、Plan、Reasoning、
  CommandExecution output、FileChange patch 与 McpToolCall progress 的 typed 增量通知。
- Plan 已完成 `item/started -> item/plan/delta* -> item/completed` 的单一 typed item 生命周期，
  每个 delta 复用 canonical `itemId`，并在 terminal 后 fail closed。内部 `plan.delta` 只保留为
  App Server event-log/projection 的 current 内部表达，不作为第二套 public wire。
- 已知字段级缺口集中在 Codex 强类型被 Lime `string`/opaque `Value` 替代的边界：
  `AgentMessage.phase`、`CommandExecution.cwd`、`CollabAgentToolCall.reasoningEffort`、
  `ImageView.path`、MCP result/error，以及媒体/搜索扩展字段。
- `UserInput` 还缺 Audio/LocalAudio；DynamicToolCall output 还缺 InputAudio，且 Lime 当前
  content item wire 不是 Codex 的 `type` tagged union。两项均为 `gap`，不能标字段对齐。
- item 邻接方法已按 notification/server request 分类。auto-approval review、Command terminal
  interaction、permissions approval 和 client dynamic tool call 均已进入 current owner；
  `item/fileChange/outputDelta` 在 Codex 已 deprecated 且不再发送，分类为 excluded，不实现兼容。
- fixture 中 `shape: gap` 只表示字段或 lifecycle 尚未收敛，不代表可以在 GUI 侧补造 synthetic
  item；生产链仍必须是 `App Server JSON-RPC -> ThreadStore -> projection`。

## 下一刀

`McpToolCall` 已完成 `item/started -> item/mcpToolCall/progress* -> item/completed` 的单一
生命周期：progress 只来自真实 MCP `ProgressNotification`，复用 canonical `itemId`，并在 started 前、
terminal 后、item 类型不匹配、空 message 或 provenance 不合法时 fail closed。下一刀回到 MCP
result/error typed shape、UserInput Audio/LocalAudio 与 DynamicToolCall InputAudio，不在 Renderer 伪造字段，
也不恢复 Codex deprecated 的 `item/fileChange/outputDelta`。

完成 V1-04 前，fixture 中每个条目都必须有 cold/live/replay/GUI evidence；当前只完成 inventory
骨架，不能标记为 `completed`。
