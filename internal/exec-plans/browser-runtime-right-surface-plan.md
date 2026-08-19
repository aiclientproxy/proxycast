# Browser Workspace 同 Tab 重构执行计划

更新时间：2026-08-19

状态：In Progress / P1-P3

## 目标

把 Browser 收敛到唯一 current 主链：用户在 Right Surface 看到的 Electron `WebContentsView`，与 Agent 通过 App Server connection-owned reverse request 操作的页面必须是同一个 native view。外部 Chrome/CDP、Canvas Browser owner、`BrowserSessionRef` adapter 和 `mcp__lime-browser__*` 不再作为内嵌 Browser 的兼容路径。

## 当前事实

```text
Renderer Browser Workspace
  -> Electron Desktop Host host capability（mount / bounds / user chrome）
  -> BrowserTabHost
  -> WebContentsView + webContents.debugger

Agent Browser capability
  -> RuntimeCore / tool-runtime dynamic tool binding
  -> App Server item/tool/call server request
  -> Electron AppServerDynamicToolHost
  -> 同一个 BrowserTabHost route
```

App Server 仍是业务与 turn owner；Electron 只执行原生 WebContents、debugger、权限、下载和窗口宿主能力。Dynamic tool binding 的 connection/thread/turn owner 与 BrowserTabHost 的 session/tab/view identity 必须在每次请求中精确校验。

## 写集与避让

### 本轮认领写集

- `internal/exec-plans/browser-runtime-right-surface-plan.md`
- `internal/aiprompts/architecture.md` 末尾独立 Browser owner 小节
- `electron/embeddedBrowserHost.ts` 及后续拆出的 BrowserTabHost/测试
- `electron/appServerHost.ts`、`electron/appServerDynamicToolHost.ts` 及对应测试
- `src/lib/api/embeddedBrowser.ts` 迁移后的 Browser typed gateway 与测试
- `src/components/agent/chat/workspace/right-surface/browser/**`
- Browser 专项 protocol/catalog/fixture/negative guard 文件
- `lime-rs/crates/agent/src/agent_tools/catalog.rs`、`inventory.rs` 中仅限
  `BrowserAssist`、`mcp__lime-browser__*`、`lime_site_*` 静态工具面；原因是这些旧 catalog
  仍与 `browser__*` dynamic capability 竞争唯一 Browser owner。验证使用 agent/app-server
  Rust related tests、`npm run test:contracts` 与 legacy guard。
- App Server tool inventory / workspace skill binding 的 `browser_assist` protocol、runtime、typed
  client 与直接测试 consumer；只删除该字段和旧正向断言，不改相邻 AgentControl/CodeCell 行为。

### 只读避让写集

- `lime-rs/crates/agent-runtime/**`
- `lime-rs/crates/agent/**` 现有并行 CodeCell/AgentControl 改动
- `lime-rs/crates/app-server/src/runtime/event_sink.rs`
- `lime-rs/crates/app-server/src/runtime/trace_store*`
- `lime-rs/crates/tool-runtime/src/tool_lifecycle.rs`
- `src/components/input-kit/**`
- 其它未与 Browser 纵切直接相关的用户改动

如必须触碰避让写集，先登记具体文件、函数、原因和验证命令，不覆盖既有改动。

## 删除集

完成迁移后直接删除，不新增 compat wrapper：

- `lime-rs/crates/app-server-protocol/src/protocol/v0/browser_session.rs` 及其 v0 schema/catalog 正向引用。
- `lime-rs/crates/app-server/src/runtime/browser_session.rs` 对外部 CDP 的生产调用。
- `lime-rs/crates/browser-runtime/**` 的内嵌 Browser 主链调用；若无其它 current consumer，物理删除 crate。
- `src/features/browser-runtime/**` 的外部 Chrome/connector Settings 工作台与正向入口。
- `BrowserSessionRef`、`browserSession/*` 的 Renderer current gateway 和 `mcp__lime-browser__*` catalog 正向依赖。
- `CanvasWorkbenchBrowserPanel` 及 Browser 专属 Canvas 子组件；共享的 Host policy 逻辑迁入 Browser Workspace 后删除旧 owner。
- `embedded_browser_view_*` 作为 Browser 产品命令；Plugin 若仍需要 WebContents，迁到独立宿主 owner，不回流 Browser 双轨。

允许出现在负向回流 guard、不可变历史 evidence 和执行计划历史记录中；不得出现在 production method、mock priority、catalog 正向断言或 GUI 主路径。

## 阶段与退出条件

### P0：事实源与 route identity

状态：骨架完成，negative guard 待补。

- [x] 重写本计划和 Browser 路线图。
- [x] 追加架构 owner 图与责任确认。
- [x] 定义 Electron Host `BrowserRoute`、`BrowserTab`、`BrowserTabMark` 边界。
- [x] 明确 `threadId + turnId + windowId + ownerWebContentsId + sessionId + tabId + viewId + webContentsId` 的校验与 stale 行为。
- [x] 将 identity 收敛到 canonical App Server read model，删除旧 `BrowserSessionRef` 输入。
- [ ] 为旧 BrowserSession/Canvas/MCP 前缀补负向回流守卫。

退出条件：协议、dynamic tool binding、Host route、Renderer projection 使用同一 identity；任何只凭 `BrowserSessionRef` 或外部 CDP target 的完成项均失效。

### P1：同一 WebContents 纵切

状态：基础纵切、active-turn identity Gate B 与 renderer termination cleanup Gate B 已通过，disconnect/window-close 等扩展生命周期仍待补。

- [x] BrowserTabHost 按稳定 `sessionId/tabId/viewId` 创建或复用 `WebContentsView`。
- [x] App Server `item/tool/call` reverse request 能由 Electron Host 精确路由到该 tab。
- [x] `navigate`、`observe`、`screenshot`、用户后退/刷新与 Agent action 返回同一 identity。
- [x] Host attach debugger 前后校验 owner；destroyed、window mismatch、stale turn 一律 fail closed。
- [x] `turnEnded` 清理 debugger，并按 origin/mark close 或 release。
- [x] 真实 Electron fixture 证明 Renderer mount 与 Agent `openTabs -> claimTab -> observe` 使用相同 `browserSessionId/tabId/threadId/webContentsId`，并绑定同一 active turn。
- [x] Electron Host 断连会 detach debugger，并对 user tab release、Agent tab close；原生 `WebContents` destroyed 会回收 Browser route。
- [ ] 补齐 clipboard、download/permission grant、窗口关闭和上述断连路径的真实 Electron turn-scoped evidence。
- [x] 真实 Electron fixture 证明用户人工操作后的 Agent observe 和同一 `webContentsId`。

退出条件：真实 Electron 中 Agent navigate/observe 改变用户可见页面，且 `browserSessionId/tabId/webContentsId` 全部相等；不是普通 Chrome、browser mirror 或 mock 证据。

### P2：Right Surface Browser Workspace

状态：组件骨架完成，Gate A 待复核。

- [x] 用稳定 tab identity mount Right Surface，不再使用 React `useId()` 生成产品 view identity。
- [x] 独立 Browser Workspace 承担 tab strip、chrome、viewport、权限/下载/接管状态。
- [x] 用户切换、收起、恢复、resize 只改变同一 view 的 bounds/visibility，不重建页面。
- [x] 五语言文案已补齐，Browser component test 已覆盖基础 DOM identity。
- [ ] `verify:gui-smoke` 与 Gate A evidence 复核无第二右栏、无系统浏览器、无重复 WebContents。

退出条件：Gate A 打开、切换、收起、恢复和 resize 通过，无系统浏览器弹出、第二右栏、Canvas 镜像或重复 WebContents。

### P3：旧链清退、Agent Browser API 与安全

状态：进行中；Browser identity、stale-control、active-turn 与 native renderer termination cleanup 已完成，旧链清退和剩余扩展生命周期证据仍阻塞。

- [x] Dynamic Browser capability 以稳定对象语义暴露 tabs/open/claim/release/navigate/observe/screenshot/mark。
- [x] 空白 Claw 任务打开 Browser 前先建立 canonical runtime session/thread；owner 申请失败显示明确错误，不再永久显示 loading。
- [x] App Server 校验 `runtimeSessionId -> canonical threadId`，拒绝跨 session 或伪造 Browser identity。
- [x] Browser action 校验 `pageRevision + snapshotId`，页面导航或动作后使旧 snapshot 失效。
- [ ] 重建 protocol schema/generated client，删除 `browserSession/*`、外部 CDP Settings 和旧 fixture。
- [ ] 删除 `BrowserSessionRef`、Canvas Browser owner 与 `mcp__lime-browser__*` 正向依赖。
- [ ] DOM snapshot 使用稳定 node identity；click/fill/press/select 只对 actionability 通过的目标执行。
- [ ] 普通读取/导航自动执行；提交、发布、删除、支付、身份信息、上传下载和脚本等高风险动作进入 human takeover 或审批闭环。
- [x] 用户人工操作后 Agent observe 能继续读取同一 tab，不创建新 session。

退出条件：至少一个真实网页工作流完成 observe、导航、用户操作、Agent observe 和 evidence join；旧 MCP Browser 工具不再是 current owner。

### P4：Evidence、恢复与清退

状态：待实施。

- [ ] action/evidence 关联 thread/turn/session/tab/view/webContents/action identity。
- [ ] 历史只恢复 snapshot/replay，不自动恢复 debugger 或 pending mutation。
- [ ] 删除外部 CDP、Canvas owner、BrowserSessionRef、旧 v0 schema、旧 Settings connector 和 Browser mock 正向路径；Rust crate/source 已开始物理删除，schema/client/Renderer 仍未收口。
- [ ] 负向 guard 阻止旧路径回流。

退出条件：治理扫描无 Browser 分类漂移；断连、窗口关闭、turn cancel 和 stale route 均有可观察终态。

## Gate A / Gate B

### Gate A：投影与布局

使用显式 test-only fixture 或 renderer projection 验证 Right Surface Browser Workspace 的 DOM、chrome、bounds、状态和五语言文案。Gate A 不能证明 Agent 使用同一 WebContents。

### Gate B：真实 Electron 同 tab

必须使用真实 Electron Desktop Host、preload/contextBridge、Electron IPC、`app_server_handle_json_lines`、App Server current turn/tool chain 和用户可见页面，证明：

```text
Agent action.browserSessionId == visible sessionId
Agent action.tabId            == visible tabId
Agent action.webContentsId    == Host visible webContentsId
Agent action.threadId         == canonical threadId
Agent action.turnId           == active turnId
```

证据至少包含 Electron Host route、reverse request id、App Server method、debugger attach/detach、用户人工操作后的 observe，以及 `turnEnded` close/release。普通 Chrome CDP、外部 target、Renderer `data-*`、静态截图、mock bridge 或 external backend 单独通过均不足以标记 Gate B。

2026-08-19 核心 identity Gate B 已通过：`npm run smoke:browser-runtime-electron-gate-b -- --timeout-ms 60000` 启动真实 Electron 与 runtime provider fixture，经 `app_server_handle_json_lines` 建立 current Thread/Turn，完成 `browser__openTabs -> browser__claimTab -> browser__observe -> final assistant` provider round trip。Agent observation 与 GUI 的 `browserSessionId/tabId/threadId/webContentsId` 全等，`viewId/windowId/ownerWebContentsId` 为有效 native identity，turn 以 `completed` 收口，Renderer 地址栏随后通过真实 `browser_tab_navigate` 操作相同 tab；console/page error 为零，生产 mock fallback 命中为零。脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-summary.json`。

同一 fixture 已扩展为 active-turn 生命周期场景：用户导航后旧 snapshot mutation fail-closed，Agent 重新 claim/observe 使用同一 tab/WebContents，terminal release detach debugger；随后通过真实 Electron renderer-process-gone 事件回收 Browser route。该证据标记 identity、stale-control、turn cleanup 与 renderer termination cleanup 为 pass；直接 `destroyed` 事件仍由 Electron Host 单测覆盖，不把两者混为同一触发器。尚未覆盖 approval、cancel、sidecar disconnect、window close、权限/下载和历史恢复的真实 Electron 终态。

## 验证矩阵

| 改动                             | 最低验证                                                       |
| -------------------------------- | -------------------------------------------------------------- |
| Browser protocol/catalog/client  | `npm run test:contracts` + protocol/Rust related tests         |
| Electron BrowserTabHost          | Host unit/integration + `npm run typecheck:electron`           |
| Right Surface Workspace          | Browser component tests + `npm run verify:gui-smoke`（Gate A） |
| Agent dynamic Browser capability | tool/runtime related tests + current Agent fixture             |
| 真实同 tab 闭环                  | Electron Gate B fixture / Playwright identity evidence         |
| legacy 删除                      | `npm run governance:legacy-report` + negative guards           |

每次验证记录实际命令、scenario、identity、terminal/pending 状态和未验证原因；生产路径禁止 mock fallback。

## 架构确认

本计划对应 `internal/roadmap/browser/README.md` 的 P0-P4。责任开发者：root，2026-08-18。确认内容：已复核 Codex Desktop Browser plugin 的 tab cleanup/claiming/safety 合同、Lime App Server connection-owned reverse request、Electron `WebContentsView` Host 与现有 dynamic tool server request；确认内嵌 Browser 唯一执行体为 Electron BrowserTabHost + 同一 `webContents.debugger`，外部 Chrome/CDP、Canvas Browser owner、BrowserSessionRef adapter 与 MCP 前缀不再作为 compat 主链。Gate A 与真实 Electron Gate B 分开取证，未把普通 Chrome、fixture 或 Renderer projection 冒充 Gate B。

## 进度记录

- 2026-08-18：路线图重构完成，确认 Browser 当前综合完成度约 20%；文档和治理校验通过。
- 2026-08-18：完成本执行计划重写和架构确认，下一刀为 BrowserTabHost route identity 与同 WebContents debugger 纵切。
- 2026-08-18：完成 Electron `BrowserTabHost`、`browser__*` dynamic tool reverse request、Right Surface `BrowserWorkspace` 和五语言基础投影；定向 Electron/Renderer 测试与 Electron typecheck 已通过。当前综合完成度约 45%，下一刀为 schema/client 重建与旧 Browser 链物理清退，Gate A/Gate B 尚未标记完成。
- 2026-08-19：定位并修复用户截图中的“正在打开网页”根因：空白 Claw 任务先显示 Right Surface，但 canonical Thread 尚未创建，Browser identity 申请被拒绝，native `WebContentsView` 没有合法 owner。Renderer 现在在打开 Browser 前调用 `ensureSession -> getThreadIdForSubmit -> refreshSessionReadModel`，再申请 App Server Browser identity。
- 2026-08-19：Browser Host 增加 `pageRevision + snapshotId` stale-control；Electron dynamic tool 增加 connection/thread/turn/native owner 校验。`npx vitest run electron/browserTabHost.test.ts electron/appServerDynamicToolHost.test.ts` 通过（10 tests）；`npm run smoke:agent-runtime-current-fixture` 通过；`npm run verify:gui-smoke` 通过。前置 `npm run typecheck`、`npm run test:contracts` 和 Browser/Renderer 定向组件测试已通过。
- 2026-08-19：Rust related 入口执行 1660 个 app-server lib tests，1659 通过、1 个 approval projection 断言失败；失败文件属于并行 approval 改动，不在本轮 Browser 写集。额外按测试名重跑时触发 `v8 150.4.0` macOS arm64 预编译包 404，未形成新的 Browser 结果。
- 2026-08-19：现有 `right-surface-visual-matrix` 在通用 files 面板 geometry 断言处提前失败，且仍携带旧 Canvas/CDP fixture，未进入 Browser；因此不能作为 Gate A/B 结论。当前运行中的旧 Lime 进程未加载本轮改动，9223 CDP 端口也未启用，未中断现有进程。
- 2026-08-19：继续定位 Agent Browser 工具永久 `inProgress`：`dynamic_tool.requested` 未被 App Server `V2NotificationProjector` 接受，事件泵在创建 `item/tool/call` reverse request 前直接跳过。当前将该 typed server-request 生命周期事件投影为空通知，Electron App Server client 增加持续 server-request handler，Host 在初始连接和 sidecar restart 后安装 handler；Host 已消费请求的 `serverRequest/resolved` 不再泄漏为 GUI 协议诊断。Rust projector 精确回归 `1 passed`，App Server client `122 tests` 通过，Electron Host/Gate 定向回归 `29 tests` 通过。
- 2026-08-19：Browser 核心 Gate B 通过。真实 Electron evidence 证明同一 session/tab/thread/turn/WebContents、四次 dynamic tool/provider round trip、用户地址栏导航同 tab、final assistant 可见、turn completed、console/page error 为零；证据已去除整页正文和本机路径。stale snapshot 在用户导航后的重新 claim/observe 与完整 terminal cleanup 仍是下一轮 blocker。
- 2026-08-19：重新执行 `npm run test:contracts`，协议生成无漂移，App Server client、command、harness、modality、scripts、Electron release、cleanup 与 docs boundary 全部通过，命令退出码为 `0`。
- 2026-08-19：重新执行 `npm run verify:gui-smoke`。SHELL-01 真实 Electron summary 的 `21/21` 产品断言全部通过，覆盖启动、reload、preload/IPC、App Server current method、Workbench/Settings、console/page/invoke error 为零、legacy/mock 命中为零；但 evidence 写入后 Electron 主进程未自行退出，launcher 在 120 秒触发 watchdog，命令最终退出码为 `1`。该结果分类为 smoke 生命周期清理缺口，不标记最新 GUI smoke 门禁通过，也不推翻 Browser 专项 Gate B 的独立通过证据。
- 2026-08-19：`npm run smoke:agent-runtime-current-fixture` 的最新运行在既有 `unknown-item` Electron 场景超过声明的 240 秒仍未收口；只清理本轮创建的进程树，未终止其它并行任务。该项记录为 current fixture/harness 阻塞，不能作为 Browser dynamic-tool 回归失败，也不能标记通过。
- 2026-08-19：扩展 `smoke:browser-runtime-electron-gate-b` 为 active-turn 生命周期场景并通过，退出码为 `0`。同一真实 Electron `WebContentsView` 完成初次 Agent observe（pageRevision=3）、用户地址栏导航（同 tab/webContents，controlOwner=user，pageRevision=8）、旧 snapshot mutation fail-closed、重新 claim/observe（新 snapshot、同 tab/webContents），turn `completed` 后用户 tab `released`，debugger 从 attached 变为 detached；24 项 Gate B 断言全部通过，console/page error 为 0。脱敏 evidence 仍位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-summary.json`。
- 2026-08-19：补齐 Electron sidecar disconnect 与 native `WebContents` destroyed 的统一清理：`AppServerHost` 在 stop、sidecar exit、restart failure 和 stale connection 触发 `BrowserTabHost.connectionLost`；user-origin tab release 并 detach debugger，Agent-origin 无 mark tab close；原生 `destroyed` 事件发出 `browser-tab-closed`。`embeddedBrowserHost`、`browserTabHost`、`appServerDynamicToolHost` 定向回归共 32 tests 通过，`npm run typecheck:electron` 与 `git diff --check` 通过。尚未把该单测结果冒充真实 Electron disconnect/window-close Gate B evidence。
- 2026-08-19：真实 Electron Gate B 以 `npm run smoke:browser-runtime-electron-gate-b -- --timeout-ms 60000 --interval-ms 500` 通过（退出码 `0`）。25 项断言全部通过：同一 session/tab/thread/WebContents、用户接管后的 stale snapshot 拒绝与重新 observe、turn release/debugger detach，以及 renderer-process-gone 后 Browser route 关闭；console/page error 为零。证据文件为 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-summary.json`。该运行验证的是 renderer termination 触发器；`destroyed` 事件路径仍以 Host 单测为准。
- 2026-08-19：补齐取消/失败终态的 fail-closed 边界：`AppServerDynamicToolHost` 仅在 `turn/completed` 携带 `completed/failed/interrupted/canceled/cancelled` 状态时调用 `BrowserTabHost.turnEnded`，缺少状态或仍为 `inProgress` 时不释放 Browser 控制；Electron DynamicTool/Browser/AppServerHost 定向回归 62 tests、`npm run typecheck:electron` 通过。真实 Electron cancel、sidecar disconnect 和 window-close evidence 仍待补。

## 当前结论

- 当前综合完成度约 74%；owner/identity/Host 基础纵切、reverse-request 回环、同 tab Gate B、用户接管后的 stale-control/reclaim、turn terminal cleanup、sidecar disconnect/native renderer termination 清理已落地，旧链清退、Browser 专项 Gate A、权限/下载/窗口关闭的真实 evidence、完整安全审批与历史恢复仍未完成。
- 用户闭环：空白 Claw 任务会先补齐 canonical owner，再进入 Browser mount；Agent Browser 工具不再永久停在 `inProgress`，真实 Electron 已完成 `openTabs -> claimTab -> observe -> final assistant`。
- Gate A：Browser 组件与普通 GUI shell 已验证，但专项 visual matrix 被旧 files geometry 断言阻塞，尚未形成 Browser 专项证据。
- Gate B：核心 identity/dynamic-tool round trip、同一 `WebContentsView`、用户地址栏导航同 tab、用户导航后的 stale snapshot 拒绝、重新 claim/observe、debugger detach/close-release 和 renderer-process-gone route cleanup 均为 pass；直接 `destroyed` 事件已有 Host 单测，approval、cancel、disconnect、window-close、权限/下载和历史恢复真实证据仍未完成。
- 通用门禁：contracts 已通过；最新 GUI smoke 的产品断言通过但进程生命周期收尾失败；Agent current fixture 在既有 `unknown-item` 场景挂起。二者均需按 harness 缺口单独修复和重跑，当前不能宣称完整本地门禁全绿。
- 下一刀：补齐 cancel、窗口关闭、权限/下载的真实 Electron turn-scoped 终态证据；随后清理旧 visual matrix 的 Canvas/CDP 正向 fixture并补专项 Gate A。
