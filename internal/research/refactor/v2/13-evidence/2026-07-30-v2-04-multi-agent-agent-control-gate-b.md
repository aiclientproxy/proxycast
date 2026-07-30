# V2-04 Multi-Agent AgentControl Gate B evidence

日期：2026-07-30

## 结论

`.lime/qc/agent-runtime-tool-execution-smoke.json` 为真实 Electron Gate B，状态 `pass`，
39/39 assertions 全真且 `failedAssertions=[]`。本证据关闭 V2-04 的 Multi-Agent 子切片，不表示
V2-04 或 Refactor v2 整体完成。

场景使用 localhost OpenAI-compatible provider fixture，证明真实 Electron Host、preload/IPC、
`app_server_handle_json_lines`、App Server、RuntimeCore、canonical read model 与 GUI；不证明 live
Provider。

## 产品链证据

- `spawn_agent`、`list_agents`、`send_message`、`followup_task`、`interrupt_agent`、
  `wait_agent` 在 provider request、runtime inventory、canonical read model 与可见 DOM 中一致。
- 六个 AgentControl Tool row 各出现一次，全部 `completed` 且 visible；旧 Team 工具未出现。
- Started、两条 Interacted、Interrupted 共四条 SubAgent activity 使用 canonical Item 与 child
  Thread identity；`wait_agent` 保持独立 typed state，不再投影为 `subagent_activity(kind=wait)`。
- `thread/read`、`thread/list` 均经 `electron-ipc` 成功命中 current App Server method。
- final assistant text 可见，invoke error `0`，console error `0`。

## Parent-owned child

canonical child Thread 的 `parentThreadId` 指向 parent，`canAcceptDirectInput=false`。真实 GUI 打开
child route 后：

- textarea、发送、access mode、model selector 与 task mode 均禁用；ModelSelector 有稳定
  `data-testid`。
- placeholder 使用 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR` 专用文案集合。
- Enter 与 disabled send 的 UI 尝试前后 `turn/start` 数量均为 `0`。
- App Server public JSON-RPC 返回 `-32600`：
  `direct app-server input is not allowed for parent-owned threads`。

截图：`.lime/qc/agent-runtime-tool-execution-smoke-parent-owned-child.png`。

## 冷重启

- Electron PID：`40827 -> 42257`，不是 `page.reload`。
- 旧进程树已退出，`remainingPids=[]`。
- 六个 Tool 的 `(itemId, name, status)`、四个 SubAgent activity 的
  `(itemId, kind, childThreadId)`、child Thread 与 `wait_agent` state 跨重启一致。
- 重启后恢复 6 个 Tool row、4 个 SubAgent activity row 与 final assistant text。

截图：

- `.lime/qc/agent-runtime-tool-execution-smoke-pre-restart-visible-dom.png`
- `.lime/qc/agent-runtime-tool-execution-smoke-cold-restart-visible-dom.png`

## 验证

- focused Vitest：3 files / 88 tests passed。
- AgentControl Gate assertion 回归：13/13 passed。
- Rust App Server：1623 个单测及相关 integration targets passed。
- `npm run build`：Renderer、Electron Host 与 App Server sidecar build passed。
- `npm run test:contracts`：passed。
- `npm run governance:legacy-report`：零引用候选 0、分类漂移 0、边界违规 0。
- `npm run smoke:agent-control-cold-restart-gate-b`：passed。
- `npm run smoke:agent-runtime-current-fixture`：passed，`liveProviderUsed=false`。
- `npm run verify:gui-smoke`：passed；证据为
  `.lime/qc/project-gates/standalone-shell-01-20260730095250-16113/shell-01-electron-smoke/summary.json`，
  21/21 assertions，legacy/mock/console/page/invoke errors 均为 0。

`npm run test:related` 因 runner 把工作树中的 `electron/` 目录误当文件而报 `EISDIR`；随后使用
精确 Vitest 文件模式完成 3 files / 88 tests。该 runner 缺陷不记录为通过。

## 分类

- `current`：六工具 AgentControl、typed wait states、canonical SubAgent activity、parent-owned
  direct-input policy、cold-restart identity、真实 Electron GUI。
- `compat`：无。
- `deprecated`：`currentTime/read`、`item/permissions/requestApproval`、`item/tool/call`；仍缺
  product-scope producer 与 Gate B。
- `dead / deleted / forbidden-to-restore`：裸旧 Team 工具、raw output 状态推断、生产 mock
  fallback。

## 下一刀

V2-04 继续关闭三项 product-scope reverse request 的真实 producer、统一 PendingInteraction 与 Gate B；
Multi-Agent owner 不再重复改写。完成后进入 V2-05 notification、host capability 与 recovery。
