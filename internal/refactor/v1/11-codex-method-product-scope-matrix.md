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

| 状态                     | 数量 | 裁决                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------ | ---: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `implemented`            |   87 | 连接握手、核心 Thread/Turn/Item、durable ordered Thread Section、thread subscription/lifecycle/content search/raw item injection/background terminals/elicitation/Guardian continuation、Plugin Search、Hook discovery、Skills list/config/extra roots/watcher、Apps exact catalog/read/installed 与 typed `app/list/updated` watcher、exact MCP resource/tool request、typed approval/MCP server request 与 model control plane |
| `planned`                |   98 | 其余 Plugins、process/fs、config、review 与 Windows sandbox                                                                                                                                                                                                                                                                                                                                                                      |
| `product-scope-excluded` |   35 | Codex account/commerce、attestation、remote control、test-only、internal raw response、deprecated surface，以及只表达单一全局 Provider 的 capability read                                                                                                                                                                                                                                                                        |
| 合计                     |  220 | `136` client request、`11` server request、`72` server notification、`1` client notification                                                                                                                                                                                                                                                                                                                                     |

`implemented` 只说明 method boundary 已存在并接入 current owner，不代表字段、恢复、GUI 或 Gate B 已全面 parity。
字段和 lifecycle 缺口继续由 Item inventory、gap register 与对应执行切片管理。

## 产品裁决

1. Codex/ChatGPT account、billing、attestation 和 remote-control client administration 不进入 Lime；不建 compat。
2. `applyPatchApproval`、`execCommandApproval`、`thread/rollback`、`item/fileChange/outputDelta`、`thread/compacted` 为 deprecated，禁止恢复。
3. `rawResponse/*` 绕过 canonical Thread/Turn/Item，保持 internal/excluded。
4. `executionProcess/*`、`fileSystem/*`、旧 skill/plugin method 即使功能相似，也不能算 Codex method parity。
5. Realtime、review、process、Windows sandbox 属于产品范围，当前标 `planned`，后续只能在既有 owner 补齐。
6. `modelProvider/capabilities/read` 读取 Codex 单一全局 Provider；Lime 的 provider/model 选择绑定 Thread route。该空参数方法没有产品消费者，不能把静态全局值冒充当前 route truth，归入 excluded；能力继续由 executable model catalog、resolved route 与 provider lowering 承接。
7. Codex HEAD 的 durable ordered Thread Section 已进入 Lime current 主链：section/store/order、五个 exact method、typed client、冷启动恢复与 Desktop 分组/置顶均消费同一事实源。旧 `isPinned` metadata、Renderer 时间重排与 localStorage 收藏为 `dead / deleted / forbidden-to-restore`。
8. `plugin/search` 已使用 exact Codex method、params/result wire、typed client 与公共 JSON-RPC 证据接入本地 Plugin catalog owner；Codex remote catalog、share、watcher 和 readiness 仍由其余 planned method 承接，不由该 method 冒充完成。
9. `hooks/list` 已接入 `tool-runtime` 唯一 discovery/trust owner、exact v2 contract、公共 JSON-RPC、command Hook sampling lifecycle、canonical Hook Item 恢复与真实 Electron Gate B；旧 raw config、默认信任、空 reporter 和 Renderer drift fallback 为 `dead / deleted / forbidden-to-restore`。
10. `skills/list` 已接入 `lime-skills::AgentSkillSnapshot` 唯一 discovery owner、exact `cwds + forceReload -> data[{cwd,skills,errors}]` contract、公共 JSON-RPC、typed clients 与 Renderer catalog projection；`skills/changed {}` 通过真实 Electron Gate B 驱动 GUI 自动刷新。singular `skill/list`、`SkillListResponse` 与 `get_local_skills_for_app` 为 `dead / deleted / forbidden-to-restore`。
11. `skills/config/write` 与 `skills/extraRoots/set` 已接入 exact v2 contract、公共 JSON-RPC、typed clients 与同一 `lime-skills::AgentSkillSnapshot` owner。Desktop 将用户级启停配置持久化到 Lime YAML `skills.config`，extra roots 只做进程级原子替换；成功设置 roots 发送 `skills/changed {}`。两者不引入 Codex TUI 或第二套管理 catalog。
12. `app/list`、`app/read`、`app/installed` 与 `app/list/updated` 已接入 exact v2 contract、公共 JSON-RPC、typed package client 和 Desktop Apps gateway。Apps catalog 唯一复用 Plugin catalog；本地 Plugin 没有 hosted connector 的 model-visible tool snapshot 时，`callable` 必须为 `false`，不能冒充 readiness 或模型可调用能力。

## 守卫

`src/lib/governance/codexMethodProductScopeBoundary.test.ts` 固定以下事实：

- 220 个方向化 identity 无遗漏、无重复，状态和方向计数稳定。
- planned 必须写 gap，excluded 必须写 rationale，所有组必须写 owner/evidence/priority。
- `implemented` 必须能在 Lime generated manifest 找到同方向、同名 contract。

上游 Codex revision 变化时，先重跑注册表审计并更新矩阵；不得只改 hash 或计数让守卫通过。

## 下一刀

`thread/inject_items` 已对齐 exact method/shape、Codex current `ResponseItem` validation union、active Turn
session actor delivery、durable provider-only history、Responses exact lowering 与非 Responses fail-closed。
Guardian reviewer producer/lifecycle 和 elicitation provider active-time pause consumer 仍是 runtime lifecycle
blocker，但不影响这两个 method boundary 的 implemented 分类。多模型控制平面的
`model/verification` 与 `model/rerouted` 已接入可信 Responses metadata producer、Turn 级去重、exact v2
notification、schema 与 generated client。reroute 只接受 first-party requested/server mismatch，使用
`highRiskCyberActivity`，并通过 transient sink 实时投影而不进入 EventLog/resume item replay；普通 provider
fallback 继续只产生 `routing.fallback.applied`。下一刀处理 provider adapter/hosted tool 闭环、
Plugins/Apps watcher/readiness。每完成一个
method，必须同步 exact protocol、handler、typed client、fixture/evidence，再将其移入 `implemented`。
Codex 已明确将 `thread/rollback` 标记为即将删除，Lime 不新增该公开方法。当前产品范围完成度为
`87 / 185 = 47.0%`。相对上一切片新增 Apps 的三个 client request 与一个 server notification，以及
`mcpServer/resource/read` 与 `mcpServer/tool/call` 两个 client request，
均有 exact contract、公共 JSON-RPC 和 typed client evidence。resource read 的 `threadId` 可选；存在时只读取
对应 Session-owned MCP runtime。tool call 强制真实 `threadId`，经 `ExecutionBackend -> AgentRuntimeState ->
McpThreadRuntime` 执行，不经过全局 management manager。Settings 只浏览工具，不伪造 Thread owner。
此前相对注册表基线新增的 `13` 个方向已全部进入 `implemented`：五个 durable Thread Section
管理/移动 method、exact `plugin/search`、`hooks/list`、三个 Apps client request、Apps notification 与这两个 MCP method。现有自定义 Plugin
list/read/install 仍不能冒充其余 Codex Plugin method parity。
Gemini GenerateContent transport 虽已完成 request/stream/tool/history
闭环，但没有新增 exact Codex App Server method，因此不改变本矩阵计数；不得把 `37.8%` 解释成多模型或
整个 Codex 对齐工程的完成度。
