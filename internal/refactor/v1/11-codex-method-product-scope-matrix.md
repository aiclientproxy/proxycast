# V1-01 Codex Method 产品范围矩阵

> status: `inventory-current / implementation-open`
> owner: `app-server-protocol` + 对应 current domain owner
> fixture: `internal/refactor/v1/fixtures/codex-method-product-scope.v0.1.json`
> upstream: `/Users/coso/Documents/dev/rust/codex@c4f42d161ae44a8d696ee9fb595709661979d187`

## 目标

把 Codex App Server 注册表的全部 method 变成可审计产品裁决，禁止用“已有同类模块”或
legacy 同义命令冒充协议对齐。矩阵覆盖 `clientRequest`、`serverRequest`、
`serverNotification` 与 `clientNotification` 四个方向；每个方向化 method 只能属于：

- `implemented`：Lime generated manifest 存在同方向、同名 current contract，并有 owner/evidence。
- `planned`：能力属于 Lime 产品范围，但 exact method、shape、lifecycle 或证据尚未完成。
- `product-scope-excluded`：Codex/ChatGPT 专属、test-only、internal 或 deprecated surface，禁止复制或恢复兼容层。

## 当前盘点

| 状态                     | 数量 | 裁决                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------ | ---: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `implemented`            |  130 | 连接握手、核心 Thread/Turn/Item、durable ordered Thread Section、thread subscription/lifecycle/content search/raw item injection/background terminals/elicitation/Guardian continuation、基础 Plugin catalog（list/read/install/uninstall/installed）与 Plugin Search、Hook discovery/lifecycle notifications、Skills list/config/extra roots/watcher、Apps exact catalog/read/installed 与 typed `app/list/updated` watcher、exact memory reset、connection-scoped process lifecycle、exact fs IO/watch、exact command/exec lifecycle、inline review/start 与 entered/exitedReviewMode boundary、`currentTime/read`、`item/permissions/requestApproval`、`item/tool/call`、typed `error`/`warning`、`item/commandExecution/terminalInteraction`、`item/autoApprovalReview/{started,completed}`、`turn/plan/updated`、精确 `turn/diff/updated`、`turn/moderationMetadata`、exact MCP resource/tool request/lifecycle notifications、typed approval/MCP server request 与 model control plane |
| `planned`                |   54 | Plugin share/skill-read、`guardianWarning` 与其余 review notifications、config、Realtime 与 Windows sandbox                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `product-scope-excluded` |   36 | Codex account/commerce、attestation、remote control、test-only、internal raw response、deprecated surface、Desktop 不消费的开发/设置 deprecation notice，以及只表达单一全局 Provider 的 capability read                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 合计                     |  220 | `136` client request、`11` server request、`72` server notification、`1` client notification                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

`implemented` 只说明 method boundary 已存在并接入 current owner，不代表字段、恢复、GUI 或 Gate B 已全面 parity。
字段和 lifecycle 缺口继续由 Item inventory、gap register 与对应执行切片管理。

## 产品裁决

1. Codex/ChatGPT account、billing、attestation 和 remote-control client administration 不进入 Lime；不建 compat。
2. `applyPatchApproval`、`execCommandApproval`、`thread/rollback`、`item/fileChange/outputDelta`、`thread/compacted` 为 deprecated，禁止恢复。
3. `rawResponse/*` 绕过 canonical Thread/Turn/Item，保持 internal/excluded。
4. `executionProcess/*`、`fileSystem/*`、旧 skill/plugin method 即使功能相似，也不能算 Codex method parity；只有 generated manifest 中同名、同方向且接入 current owner 的 Plugin catalog 方法才计入。
5. Realtime 与 Windows sandbox 属于产品范围，当前标 `planned`，后续只能在既有 owner 补齐；Desktop review 已由 inline `review/start` current owner 承接，Codex TUI detached/background review 不进入 Lime。
6. `modelProvider/capabilities/read` 读取 Codex 单一全局 Provider；Lime 的 provider/model 选择绑定 Thread route。该空参数方法没有产品消费者，不能把静态全局值冒充当前 route truth，归入 excluded；能力继续由 executable model catalog、resolved route 与 provider lowering 承接。
7. Codex HEAD 的 durable ordered Thread Section 已进入 Lime current 主链：section/store/order、五个 exact method、typed client、冷启动恢复与 Desktop 分组/置顶均消费同一事实源。旧 `isPinned` metadata、Renderer 时间重排与 localStorage 收藏为 `dead / deleted / forbidden-to-restore`。
8. `plugin/list`、`plugin/read`、`plugin/install`、`plugin/uninstall`、`plugin/installed` 和 `plugin/search` 已使用 exact Codex method、params/result wire、typed client 与公共 JSON-RPC/Electron 证据接入本地 Plugin catalog owner；Codex remote catalog、share、skill-read、watcher 和 readiness 仍由 planned method 承接，不由这些基础方法冒充完成。
9. `hooks/list` 已接入 `tool-runtime` 唯一 discovery/trust owner、exact v2 contract、公共 JSON-RPC、command Hook sampling lifecycle、canonical Hook Item 恢复与真实 Electron Gate B；旧 raw config、默认信任、空 reporter 和 Renderer drift fallback 为 `dead / deleted / forbidden-to-restore`。
10. `skills/list` 已接入 `lime-skills::AgentSkillSnapshot` 唯一 discovery owner、exact `cwds + forceReload -> data[{cwd,skills,errors}]` contract、公共 JSON-RPC、typed clients 与 Renderer catalog projection；`skills/changed {}` 通过真实 Electron Gate B 驱动 GUI 自动刷新。singular `skill/list`、`SkillListResponse` 与 `get_local_skills_for_app` 为 `dead / deleted / forbidden-to-restore`。
11. `skills/config/write` 与 `skills/extraRoots/set` 已接入 exact v2 contract、公共 JSON-RPC、typed clients 与同一 `lime-skills::AgentSkillSnapshot` owner。Desktop 将用户级启停配置持久化到 Lime YAML `skills.config`，extra roots 只做进程级原子替换；成功设置 roots 发送 `skills/changed {}`。两者不引入 Codex TUI 或第二套管理 catalog。
12. `app/list`、`app/read`、`app/installed` 与 `app/list/updated` 已接入 exact v2 contract、公共 JSON-RPC、typed package client 和 Desktop Apps gateway。Apps catalog 唯一复用 Plugin catalog；本地 Plugin 没有 hosted connector 的 model-visible tool snapshot 时，`callable` 必须为 `false`，不能冒充 readiness 或模型可调用能力。
13. `process/spawn`、`process/writeStdin`、`process/resizePty`、`process/kill`、`process/outputDelta` 与 `process/exited` 已接入 exact connection-scoped contract、真实 local supervisor 和 typed package client。Desktop Workspace 继续通过 `thread/backgroundTerminals/*` 投影 Thread-owned 终止能力，不复制 Codex TUI，也不把 Thread item id 冒充 connection process handle；旧 `executionProcess/*` 为 `dead / deleted / forbidden-to-restore`。
14. `fs/readFile`、`fs/writeFile`、`fs/createDirectory`、`fs/getMetadata`、`fs/readDirectory`、`fs/remove`、`fs/copy`、`fs/watch`、`fs/unwatch` 与 `fs/changed` 已接入 exact v2 contract、App Server `FsServer`、typed clients 和公共 JSON-RPC 证据。raw bytes 始终使用 base64，路径必须为绝对路径，watch id 按 connection 隔离；Desktop 文件浏览/预览只做 GUI 投影。旧 `fileSystem/*`、v0 DTO/schema 和 file browser Rust owner 为 `dead / deleted / forbidden-to-restore`。
15. `currentTime/read` 由 Electron Desktop Host 唯一读取系统时钟，App Server waiter 负责 canonical thread scope、响应校验和超时；它不创建 Thread Item，也不暴露 Renderer 时钟 API。`item/tool/call` 由冻结的 Desktop dynamic-tool binding 响应，RuntimeCore waiter 与 canonical DynamicToolCall Item 负责生命周期和恢复，Renderer 不能伪造调用。两者均有同名 generated manifest、typed contract、边界单测和 Electron Gate B 证据。
16. `turn/plan/updated` 由 RuntimeCore `update_plan` producer 生成 durable `turn.plan.updated` fact，经 App Server v2 projector、typed package notification 和 Renderer projection 投递；Plan snapshot 仍由 canonical Plan Item/read model 承担，不能用本地 GUI 状态或 Tool Item 冒充。
17. `item/permissions/requestApproval` 由 `tool-runtime` 解析并收紧 permission profile，App Server exact-id waiter 传递 canonical session/thread/turn/item/environment identity，Renderer 统一 `PendingInteractionController` 返回 turn/session grant 或 fail-closed 空 grant；相对路径、越界 profile、重复和迟到响应均被拒绝。
18. `warning` 与 `error` 由 runtime durable event producer、App Server v2 projector、canonical read model、typed client 和 Renderer 共同承接；`error.willRetry` 不伪造 Turn terminal，`warning.code` 只用于五语言展示。`item/commandExecution/terminalInteraction` 由 command completion producer 生成脱敏摘要，live notification、cold read 和 GUI 都消费同一 bounded typed projection。
19. `turn/diff/updated` 由 `apply_patch` 的 Turn-scoped 精确 delta producer 生成 durable `turn.diff.updated` fact，经 App Server v2 projector、typed client 和 canonical conversation Turn 投影；Desktop Changes previous-conversation 只读取该 Turn 的 `unified_diff`，空字符串表示 net-zero 清除，不回退到第二套本地 patch 拼装。
20. `turn/moderationMetadata` 只接受 trusted first-party Responses `response.metadata.openai_chatgpt_moderation_metadata`，经 `model-provider` canonical event、Agent runtime durable `turn.moderation_metadata`、App Server exact v2 notification 和 Renderer canonical Turn 投影。metadata 保持任意 JSON，不猜字段、不展示 raw JSON；每次更新均投递并按 last-write-wins 合并，`null` 是有效覆盖值。Codex TUI 当前忽略该通知，Lime Desktop 不复制 TUI surface，也不新增 Electron IPC 或改变 Grok-aligned 多模型/多模态 owner。

21. `deprecationNotice` 属于 Codex 开发/设置诊断，不进入 Lime Desktop 对话或通知事实源。Lime 对无外部兼容负担的退役实现直接替换/删除，不新增同名兼容通知；`guardianWarning` 只有在独立的高优先级 warning producer 落地后才重新评估。

## 守卫

`src/lib/governance/codexMethodProductScopeBoundary.test.ts` 固定以下事实：

- 220 个方向化 identity 无遗漏、无重复，状态和方向计数稳定。
- planned 必须写 gap，excluded 必须写 rationale，所有组必须写 owner/evidence/priority。
- `implemented` 必须能在 Lime generated manifest 找到同方向、同名 contract。

上游 Codex revision 变化时，先重跑注册表审计并更新矩阵；不得只改 hash 或计数让守卫通过。

## 下一刀

`thread/inject_items` 已对齐 exact method/shape、Codex current `ResponseItem` validation union、active Turn
session actor delivery、durable provider-only history、Responses exact lowering 与非 Responses fail-closed。
Guardian reviewer producer/lifecycle 已进入 current 主链；elicitation provider active-time pause consumer 仍是 runtime lifecycle
blocker，但不影响已实现的 Guardian method boundary。多模型控制平面的
`model/verification` 与 `model/rerouted` 已接入可信 Responses metadata producer、Turn 级去重、exact v2
notification、schema 与 generated client。reroute 只接受 first-party requested/server mismatch，使用
`highRiskCyberActivity`，并通过 transient sink 实时投影而不进入 EventLog/resume item replay；普通 provider
fallback 继续只产生 `routing.fallback.applied`。下一刀处理 provider adapter/hosted tool 闭环、
Plugin share/skill-read、`guardianWarning` 与 remaining review notifications 和 Apps watcher/readiness。`deprecationNotice`
已按 Desktop 产品范围裁决为 excluded。每完成一个
method，必须同步 exact protocol、handler、typed client、fixture/evidence，再将其移入 `implemented`。
Codex 已明确将 `thread/rollback` 标记为即将删除，Lime 不新增该公开方法。当前产品范围完成度为
`130 / 184 = 70.7%`。本切片将 `item/autoApprovalReview/{started,completed}` 接入真实 Guardian reviewer 到 Desktop pending interaction 的完整 current 主链；此前本切片将 `turn/moderationMetadata` 接入 trusted first-party Responses metadata 到 Desktop canonical Turn 的完整 current 主链，并将 `deprecationNotice` 按 Desktop 产品范围移入 excluded；更早已将基础 Plugin catalog 五个方法、`currentTime/read`、`item/permissions/requestApproval`、`item/tool/call`、`warning`、`error`、`item/commandExecution/terminalInteraction`、`turn/plan/updated` 与 `turn/diff/updated` 从混合 planned 组拆出；`review/start` client request、`enteredReviewMode`/`exitedReviewMode` canonical boundary 与 Desktop Gate B evidence，以及 command/exec、fs、process 与 memory slices 也已同步进入 current owner，并物理删除对应旧 public surface；再前一切片收口已有 Hook lifecycle 与 MCP lifecycle notification 的中央 catalog/产品裁决；相对更早切片新增 Apps 的三个 client request 与一个 server notification，以及
`mcpServer/resource/read` 与 `mcpServer/tool/call` 两个 client request，
均有 exact contract、公共 JSON-RPC 和 typed client evidence。resource read 的 `threadId` 可选；存在时只读取
对应 Session-owned MCP runtime。tool call 强制真实 `threadId`，经 `ExecutionBackend -> AgentRuntimeState ->
McpThreadRuntime` 执行，不经过全局 management manager。Settings 只浏览工具，不伪造 Thread owner。
此前相对注册表基线新增的 `28` 个方向已全部进入 `implemented`：五个 durable Thread Section
管理/移动 method、基础 Plugin catalog 五个方法、exact `plugin/search`、`hooks/list`、三个 Apps client request、Apps notification、三个 Host/product reverse requests、typed `warning`/`error`、command terminal interaction、`turn/plan/updated`、`turn/diff/updated`、两个 MCP method、`turn/moderationMetadata` 与两个 Guardian auto-approval review notifications。Plugin share/skill-read、`guardianWarning` 与其余 review notifications 仍保持 planned，`deprecationNotice` 已按 Desktop 产品范围移入 excluded。
Gemini GenerateContent transport 虽已完成 request/stream/tool/history
闭环，但没有新增 exact Codex App Server method，因此不改变本矩阵计数；不得把 `70.7%` 解释成多模型或
整个 Codex 对齐工程的完成度。
