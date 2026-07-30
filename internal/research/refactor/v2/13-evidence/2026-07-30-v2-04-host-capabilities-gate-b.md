# V2-04 Host Capabilities 与 Product-Scope Reverse Request evidence

日期：2026-07-30

## 结论

V2-04 的 `currentTime/read`、`item/permissions/requestApproval`、`item/tool/call` 已完成
current producer、typed protocol、runtime waiter、Electron Host binding、canonical projection 与定向回归。
DynamicTool 的真实 Electron Gate B 通过；V2-04 标记 `completed`。这不表示 Refactor v2 整体完成或
release-ready，V2-05 notification、host capability 与 recovery 仍开放。

## 产品链

```text
Electron Desktop Host
-> preload/contextBridge + electron IPC
-> app_server_handle_json_lines
-> App Server JSON-RPC
-> RuntimeCore/session loop exact waiter
-> canonical Thread/Turn/Item
-> GUI PendingInteraction / typed timeline
```

- `currentTime/read`：Electron Host 是唯一系统时钟读取者；App Server 校验 thread scope/deadline，
  RuntimeCore 只恢复对应 waiter；不创建 Item，不暴露 Renderer 时钟 API。
- `item/permissions/requestApproval`：typed profile、cwd、reason、environment 和 grant 经过
  tool-runtime、agent-runtime、App Server server-request 与既有 PendingInteractionController；权限提升、
  相对路径、身份不匹配、重复和迟到响应均 fail closed。
- `item/tool/call`：Electron Host 冻结 `desktop.appInfo` binding；Renderer 无法注入 dynamicTools、
  伪造 namespace/tool/schema/arguments 或观察执行请求；返回仅含 app name/version/locale/platform。
- DynamicTool canonical payload 显式保存 callId、namespace、tool、原始 JSON arguments、有序
  text/image/audio content、success、duration；provider history、read model 和 App Server projection 不从
  metadata 猜测核心字段。

## Gate B

命令：

```bash
npm run smoke:mcp-elicitation-gate-b -- --prefix v2-04-host-capabilities-final
```

证据：

- summary：`.lime/qc/gui-evidence/mcp-elicitation-gate-b/v2-04-host-capabilities-final-summary.json`
- raw：`.lime/qc/gui-evidence/mcp-elicitation-gate-b/v2-04-host-capabilities-final-raw.json`
- screenshot：`.lime/qc/gui-evidence/mcp-elicitation-gate-b/v2-04-host-capabilities-final.png`

关键断言：`ok=true`、`proofLevel=Gate B`、真实 Electron/preload/IPC/
`app_server_handle_json_lines`/App Server/runtime/read model/GUI 可见；`dynamicToolProviderResultObserved=true`、
`dynamicToolCanonicalCompleted=true`、`dynamicToolStartedObserved=true`、
`dynamicToolRequestHiddenFromRenderer=true`；provider request count=3；console errors、missing required
methods、legacy MCP commands 均为 0。fixture 使用受控 provider，不宣称 live Provider。

该 Gate B 直接覆盖 DynamicTool host binding 和 canonical projection。`currentTime/read` 与
`item/permissions/requestApproval` 的 Electron Host/bridge 行为由本轮 Electron 39 tests 和 Rust typed
waiter/profile tests 覆盖；没有把 DynamicTool Gate B 误写成这两项的独立真实交互证据。

## 定向验证

- `cargo check --manifest-path lime-rs/Cargo.toml -p agent-protocol -p lime-agent -p app-server`：通过。
- typed DynamicTool：agent-protocol 1/1、lime-agent 3/3、app-server 5/5。
- permission：agent-runtime 3/3、lime-agent 1/1、tool-runtime 2/2、app-server 1/1。
- current time：app-server 5/5、tool-runtime 2/2。
- `npm run typecheck:electron`：通过；Electron Host Vitest 3 files / 39 tests 通过。
- `npm run test:contracts`：通过，protocol types 无漂移、app-server-client contract 286 checks，command/
  harness/modality/scripts/docs/release guards 均通过。
- `npm run verify:gui-smoke`：通过；`.lime/qc/project-gates/standalone-shell-01-20260730133414-77042/
  shell-01-electron-smoke/summary.json`。

## 治理分类

- `current`：三项 reverse request、Electron `desktop.appInfo` binding、typed DynamicToolCall、permission
  waiter、current-time host。
- `compat`：无。
- `deprecated`：V2-05 尚未实现的 notification/transient bypass、broader host capability 与 recovery，
  仅允许迁出。
- `dead / deleted / forbidden-to-restore`：旧 MCP Desktop command、Renderer 伪造 binding、metadata 核心
  字段推断、生产 mock fallback。

## 未验证与边界

没有对 `currentTime/read` 和 `item/permissions/requestApproval` 单独再建一套真实 Electron provider
Gate B；它们已由 Electron Host 39 项回归与 Rust typed/profile/waiter 回归覆盖，DynamicTool 则有独立 Gate B。
该边界不影响实现已通过的 contract 和生产链约束，但后续 V2-05 可补 host-capability 专用组合 fixture。
