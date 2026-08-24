# Browser Workspace 同 Tab 重构执行计划

更新时间：2026-08-23

状态：Completed / P1-P4（旧链正向路径已物理清理）

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
- `lime-rs/crates/app-server-protocol/src/protocol/v2/item.rs`、生成 schema/client 与协议测试；原因是
  `item/tool/call` 需要表达 `preflight / approvedExecute` 和 typed approval descriptor。
- `lime-rs/crates/agent/src/current_provider_turn/dynamic_tool_bridge.rs`、
  `tool_executor/orchestration.rs`；原因是 Browser dynamic tool 必须复用 canonical `action_required`
  waiter，批准后恢复同一 dynamic call，不新增 runtime 状态机。
- `lime-rs/crates/app-server/src/dynamic_tool_server_request.rs` 与精确测试；原因是 App Server 负责把
  dynamic-tool phase/token 从 Runtime 事件投影到 connection-owned reverse request。
- `lime-rs/crates/agent/src/agent_tools/catalog.rs`、`inventory.rs` 中仅限
  `BrowserAssist`、`mcp__lime-browser__*`、`lime_site_*` 静态工具面；原因是这些旧 catalog
  仍与 `browser__*` dynamic capability 竞争唯一 Browser owner。验证使用 agent/app-server
  Rust related tests、`npm run test:contracts` 与 legacy guard。
- App Server tool inventory / workspace skill binding 的 `browser_assist` protocol、runtime、typed
  client 与直接测试 consumer；只删除该字段和旧正向断言，不改相邻 AgentControl/CodeCell 行为。

### 只读避让写集

- `lime-rs/crates/agent-runtime/**`
- `lime-rs/crates/agent/**` 除上述两个 Browser dynamic-tool 文件外的现有并行 CodeCell/AgentControl 改动
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
- [x] 为旧 BrowserSession/Canvas/MCP 前缀补 current-owner 负向回流守卫，覆盖 Browser Host、typed gateway、Right Surface 与通用 Agent prompt。

退出条件：协议、dynamic tool binding、Host route、Renderer projection 使用同一 identity；任何只凭 `BrowserSessionRef` 或外部 CDP target 的完成项均失效。

### P1：同一 WebContents 纵切

状态：基础纵切、active-turn identity、native 用户接管、cancel、disconnect、window-close、permission、download 与 renderer termination cleanup Gate B 已通过。

- [x] BrowserTabHost 按稳定 `sessionId/tabId/viewId` 创建或复用 `WebContentsView`。
- [x] App Server `item/tool/call` reverse request 能由 Electron Host 精确路由到该 tab。
- [x] `navigate`、`observe`、`screenshot`、用户后退/刷新与 Agent action 返回同一 identity。
- [x] Host attach debugger 前后校验 owner；destroyed、window mismatch、stale turn 一律 fail closed。
- [x] `turnEnded` 清理 debugger，并按 origin/mark close 或 release。
- [x] 真实 Electron fixture 证明 Renderer mount 与 Agent `openTabs -> claimTab -> observe` 使用相同 `browserSessionId/tabId/threadId/webContentsId`，并绑定同一 active turn。
- [x] Electron Host 断连会 detach debugger，并对 user tab release、Agent tab close；原生 `WebContents` destroyed 会回收 Browser route。
- [x] 真实终止 App Server sidecar 后，pending provider request 取消、user tab release、debugger detach，且 Electron main 不再进入 read-pump 自旋。
- [x] 真实页面调用 geolocation 后，Electron permission handler fail closed，并把 canonical tab identity 与 blocked 状态投影到 viewport 外的 GUI 状态带。
- [x] 真实页面触发下载后，同一 Electron session 产生 `started -> cancelled`，并把 canonical tab identity 与取消终态投影到 viewport 外的 GUI 状态带；fixture 不写入用户下载目录。
- [x] Host current owner 已实现受控 `artifactRef`、artifact/write sidecar 记录、approval-scoped open/reveal/copy、clipboard read/write、permission pending/grant 和 upload preflight；Electron completed-download artifact Gate B 已通过，证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-artifact-summary.json`。Gate B fixture 覆盖 download -> artifact/write -> copy ref -> turn grant/evidence；open/reveal/read/write clipboard/upload 仍以 Host 定向单测为边界，未声称 live provider 或跨平台 packaged coverage。
- [x] 真实 Electron fixture 证明用户人工操作后的 Agent observe 和同一 `webContentsId`。
- [x] 用户直接点击/输入 native `WebContentsView` 会立即 detach debugger、撤销 Agent control/snapshot/pending approval token，并使旧 `allow_once` fail closed；Agent 自身 CDP input 不触发人工接管。
- [x] 真实 Electron GUI 停止操作经 `turn/interrupt` 将 read model 投影为 `interrupted`，release 用户 tab 并 detach debugger，同时取消 pending provider response。
- [x] 真实 Electron owner `BrowserWindow` 关闭后，Renderer page、native Browser WebContents 与 Browser route 同步销毁，close event 保留 canonical thread/tab/view identity 和 `window-closed` 原因。

退出条件：真实 Electron 中 Agent navigate/observe 改变用户可见页面，且 `browserSessionId/tabId/webContentsId` 全部相等；不是普通 Chrome、browser mirror 或 mock 证据。

### P2：Right Surface Browser Workspace

状态：组件骨架、原生状态带、五语言真实 Electron 生命周期截图与状态矩阵已通过；Gate A/Gate B 证据边界已记录。

- [x] 用稳定 tab identity mount Right Surface，不再使用 React `useId()` 生成产品 view identity。
- [x] 独立 Browser Workspace 承担 tab strip、chrome、viewport、权限/下载/接管状态。
- [x] 用户切换、收起、恢复、resize 只改变同一 view 的 bounds/visibility，不重建页面。
- [x] 五语言文案已补齐，Browser component test 已覆盖基础 DOM identity。
- [x] permission/download/load-failed 只接受当前选中 native tab 的完整 identity；状态带位于 native viewport 外，`ResizeObserver` 同步下移 WebContents bounds。
- [x] 独立 Browser Gate A projection 场景已覆盖首屏 mount、同 tab 导航、查找、缩放、新建/选择/关闭 tab、收起/恢复、桌面视口 resize、canonical session/thread 投影和横向溢出；证据标记为 Gate A，不替代 Gate B。
- [x] `verify:gui-smoke` 与五语言 Gate B 截图矩阵复核无第二右栏、无系统浏览器或布局重叠；同一 WebContents identity 继续只由 Gate B 证明。

退出条件：Gate A 打开、切换、收起、恢复和 resize 通过，无系统浏览器弹出、第二右栏、Canvas 镜像或重复 WebContents。

### P3：旧链清退、Agent Browser API 与安全

状态：已完成；Browser identity、stale-control、active-turn、native 用户接管、permission/download Host 投影、native lifecycle cleanup、Browser action approval、artifact/grant 与旧链物理清退均已收口。

- [x] Dynamic Browser capability 以稳定对象语义暴露 tabs/open/claim/release/navigate/observe/screenshot/mark。
- [x] 空白 Claw 任务打开 Browser 前先建立 canonical runtime session/thread；owner 申请失败显示明确错误，不再永久显示 loading。
- [x] App Server 校验 `runtimeSessionId -> canonical threadId`，拒绝跨 session 或伪造 Browser identity。
- [x] Browser action 校验 `pageRevision + snapshotId`，页面导航或动作后使旧 snapshot 失效。
- [x] 重建 protocol schema/generated client，删除 `browserSession/*`、外部 CDP Settings 和旧 fixture；当前协议只保留 canonical BrowserRoute/BrowserTab。
- [x] 删除 `BrowserSessionRef`、Canvas Browser owner 与 `mcp__lime-browser__*` 正向依赖；保留负向回流守卫和历史 evidence。
- [ ] DOM snapshot 使用稳定 node identity；click/fill/press/select 只对 actionability 通过的目标执行。
- [x] 普通读取/导航自动执行；危险 click、敏感 fill 与 Enter 提交进入 canonical action approval，批准后恢复原 action，拒绝不执行 mutation。
- [x] Browser dynamic tool 已进入 `agent/current_provider_turn` + `tool-runtime` 的通用 execution approval lifecycle；Electron 关键词/敏感字段 heuristic 只生成 fail-closed preflight 风险事实，不保存用户决策或建立第二套审批状态机。
- [x] action-time 风险覆盖已扩展到 upload、download completed、permission grant、artifact open/reveal/copy 与 clipboard；一次性 approval、turn-scoped grant/evidence 和 route identity 已进入 Host。任意脚本执行不在 current Browser dynamic tool catalog，继续 fail closed。上传/permission/open/reveal/clipboard 的行为证据由 Host 单测覆盖，artifact completed-download/copy 的真实 Electron Gate B 证据已落盘。
- [x] 用户人工操作后 Agent observe 能继续读取同一 tab，不创建新 session。
- [x] native 用户输入优先于 pending approval：旧 snapshot/token 同时失效，Host 不允许延迟到达的 `approvedExecute` 重放页面 mutation。

#### P3 action approval 两阶段合同

```text
dynamic tool call
  -> item/tool/call(phase=preflight)
  -> completed | approval descriptor
  -> canonical action_required(toolFamily=browser_action, allow_once/decline/cancel)
  -> item/tool/call(phase=approvedExecute, one-time approvalToken)
  -> completed | failed closed
```

- `DynamicToolCallParams.phase` 只允许 `preflight / approvedExecute`；第二阶段必须携带 Host 在第一阶段生成的
  一次性 opaque token。
- approval descriptor 只携带 action kind、reason、risk class、tab/snapshot/node lineage 和 token；普通 tool
  content 不承担控制协议解析。
- Rust `agent/current_provider_turn` 是审批等待与 resume owner；`tool-runtime::execution_approval` 生成
  `browser_action` projection，固定 `session_cache_supported=false`，因此只允许
  `allow_once / decline / cancel`。
- Electron 只做风险 preflight 和 token/identity/snapshot 校验。批准执行必须匹配原
  `thread/turn/call/tool/tab/view/WebContents/snapshot/backendNode`，token 消费一次；stale、重放、跨 tab 或跨
  snapshot 一律 fail closed。
- decline 不发第二阶段 mutation；cancel 继续使用 canonical turn/action cancellation。Browser Host 不保存用户
  决策，不把风险动作改成 `human_takeover` 后再假装完成。

退出条件：至少一个真实网页工作流完成 observe、导航、用户操作、Agent observe 和 evidence join；旧 MCP Browser 工具不再是 current owner。

### P4：Evidence、恢复与清退

状态：已完成；历史只读投影、五语言 Gate B 证据与旧链物理清退均已接入并验证。

- [x] action/evidence 关联 thread/turn/session/tab/view/webContents/action identity：Host grant/result 已写入该 identity；真实 Electron artifact Gate B 已证明 artifact/write sidecar + copy ref 的 route/turn evidence。
- [x] 历史恢复不自动恢复 debugger、claim 或 pending mutation：Browser Host 不持久化 route/debugger/approval；新 Host 实例只接受 Renderer current tab mount，`turnEnded`/disconnect 先清理 pending permission 与 approval。snapshot/replay 的 App Server historical read model 保持唯一恢复入口。
- [x] 删除外部 CDP、Canvas owner、BrowserSessionRef、旧 v0 schema、旧 Settings connector 和 Browser mock 正向路径；物理清理后仅保留负向 guard、历史文档/evidence 与 current identity 字段。
- [x] 负向 guard 阻止旧路径回流；`browserCurrentBoundary.test.ts` 与治理扫描继续把旧 BrowserSession/Canvas/MCP 前缀限制在历史 evidence、compat/deprecated 或负向测试范围。

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

同一 fixture 已扩展为 active-turn lifecycle、approval、user-control、cancel、disconnect、window-close、permission、download 与 completed artifact 场景。lifecycle 场景覆盖用户导航后旧 snapshot mutation fail-closed、Agent 重新 claim/observe、terminal release/debugger detach 和真实 renderer-process-gone route cleanup；approval 场景覆盖危险 click 的 `preflight -> action_required -> allow_once -> approvedExecute`，同一 tab/snapshot/action 只执行一次，再以 fresh snapshot 触发第二次审批并拒绝，页面 mutation count 保持为 1；user-control 场景在审批等待中直接点击 native 页面，证明同一 `webContentsId` 上 `controlOwner=user`、`activeTurnId=null`、debugger detached，旧 token 被拒绝；cancel、disconnect、window-close、permission、download 的既有 Gate B 证据继续通过。artifact 场景新增 completed download -> `artifact/write` -> 受控 ref copy、turn-scoped grant/evidence 与 terminal release；open/reveal/clipboard/upload 的 Host 定向单测边界已在 claimBoundary 中明确。

五语言真实 Electron 截图和状态矩阵证据：`.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-locale-matrix-summary.json`。矩阵覆盖 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`，每种语言 7 个生命周期截图（renderer-loading/ready、browser-loaded、agent-controlled、user-takeover、released、destroyed），目标 locale 与 `document.documentElement.lang` 逐项相等，行为矩阵引用 approval/artifact/cancel/disconnect/download/permission/user-control/window-close Gate B summary，console/page error 均为空。

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
- 2026-08-19：`smoke:browser-runtime-electron-gate-b -- --scenario cancel` 以真实 Electron 退出码 `0` 通过。GUI 停止按钮经 `safeInvoke -> app_server_handle_json_lines -> turn/interrupt` 取消 active Browser turn；Electron IPC trace 的 thread/turn identity 与 Agent 控制态一致，read model 为 `interrupted`，pending provider response 在完成前关闭且没有续发请求，同一 user tab 保留原 `webContentsId` 并进入 `released`，debugger `attached -> detached`。15 项断言全部通过，console/page/invoke error 与生产 mock fallback 均为零；脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-cancel-summary.json`。Browser/Electron 定向回归 68 tests、`npm run test:contracts` 均通过；默认 `--scenario lifecycle` 随后重跑，25 项断言和真实 Electron 退出码仍为 pass。
- 2026-08-19：`smoke:browser-runtime-electron-gate-b -- --scenario window-close` 真实 Electron 场景通过。Agent 已 claim/observe 同一用户 tab 且 debugger attached 后，fixture 在主进程关闭对应 owner `BrowserWindow`；Renderer page 随之关闭，原生窗口与 Browser WebContents 均不可解析，`BrowserTabHost` 广播 `browser-tab-closed(reason=window-closed)`，其中 thread/tab/view identity 与关闭前 route 一致，provider 无续发请求。12 项断言全部通过，console/page error 为零；脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-window-close-summary.json`。Browser/Electron 定向回归 69 tests 与 `npm run typecheck:electron` 通过。
- 2026-08-19：`smoke:browser-runtime-electron-gate-b -- --scenario disconnect` 真实 Electron 场景通过。根因是 `AppServerConnection` 安装持续 server-request handler 后，sidecar 退出令 transport 立即 reject，而 read pump 在 `finally` 中因 handler 仍存在而无间隔自启，微任务自旋使 Electron main 占满 CPU，并饿死 sidecar restart timer 与 Renderer IPC。connection 现把首个非 timeout transport error 固化为 terminal read 状态，失败现有与后续读取且不再重启 pump；回归测试先在旧实现稳定观察到第二次 read，再于修复后通过。真实 `SIGTERM` 场景 12 项断言全部通过：pending provider response `responseFinished=false`、同一 user tab `released`、debugger `attached -> detached`、无 provider 续发、console/page error 与生产 mock fallback 为零，命令退出码为 `0`。脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-disconnect-summary.json`。App Server client 123 tests、Browser/Electron fixture 36 tests、`npm run typecheck:electron`、`npm run test:contracts`、`npm run governance:scripts` 与 `npm run docs:boundary` 通过；随后 lifecycle 25 项和 window-close 12 项真实 Gate B 回归均以退出码 `0` 通过。`npm run verify:gui-smoke` 的 SHELL-01 真实 Electron summary 与命令退出码均为 pass；`npm run smoke:agent-runtime-current-fixture` 也完整通过，此前挂起的 unknown Item 场景已收口，聚合 fixture 同时覆盖 history/cache、terminal、cancel-then-continue、approval、Skills/MCP/media 与 Coding Workbench，且 `liveProviderUsed=false`。
- 2026-08-19：修复 Browser Host 事件投影的 tab 漂移和 native view 遮挡。Renderer 现在按 session/thread/tab/view/WebContents/owner/window 完整 identity 接受 permission/download/load-failed 事件，切 tab 清除上一 tab 状态；error/permission/download 状态带移到 native viewport 外，bounds observer 让真实 `WebContentsView` 从状态带下方开始。Browser Host 与 Workspace 回归覆盖 canonical identity、selected-tab 过滤、状态带顺序和非 absolute 布局。
- 2026-08-19：`smoke:browser-runtime-electron-gate-b -- --scenario permission` 真实 Electron 场景通过。native page 的 geolocation 调用经 Electron permission handler 返回 `User denied Geolocation`，Host 决策为 `blocked`；事件与 GUI 当前 tab 的 session/thread/tab/view/WebContents/owner/window identity 全等，permission band 底边与 native viewport 顶边对齐，turn 以 `interrupted` 收口。11 项断言全部通过，console/page error 与生产 mock fallback 为零；脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-permission-summary.json`。`navigator.permissions.query` 在 Electron 中仍可能报告 `granted`，不能代替真实 API 调用结果。
- 2026-08-19：`smoke:browser-runtime-electron-gate-b -- --scenario download` 真实 Electron 场景通过。native page 触发 data URL 下载，Electron `will-download` 产生同一 `downloadId` 的 `started -> cancelled`，GUI 当前 tab 显示取消状态且不遮挡 native viewport，evidence 不含 `savePath` 或用户目录，turn 以 `interrupted` 收口。11 项断言全部通过，console/page error 与生产 mock fallback 为零；脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-download-summary.json`。fixture 主动取消下载，故不证明文件落盘、artifact ref、完成后的 open/reveal 或 packaged 跨平台行为。
- 2026-08-19：收尾回归使用最新 Electron 构建全部通过：Browser Host/Workspace/Gate assertion 21 tests，根与 Electron typecheck，`npm run test:contracts`，`npm run smoke:agent-runtime-current-fixture`，`npm run verify:gui-smoke`，lifecycle Gate B 25/25、permission Gate B 11/11、download Gate B 11/11，`npm run governance:scripts`、`npm run docs:boundary` 与全工作树 `git diff --check`。三份 Gate B evidence 的 `proofLevel=Gate B`、`status=pass`、`failedAssertions=[]`；本轮未运行 live provider、Windows/macOS packaged 下载完成、Browser 专项 Gate A 或历史恢复。
- 2026-08-20：确认 dynamic Browser route 在通用 orchestration 前直接返回是审批旁路根因；确认
  `RuntimeSessionInputHandle` 可在前一 waiter resolve 后以同一 call id 顺序注册第二阶段。冻结上述 typed
  `preflight -> action_required -> approvedExecute` 合同，开始实现协议、canonical approval 与 Host 一次性恢复。
- 2026-08-20：Browser action approval 主链与真实 Electron Gate B 安全行为收口。协议/schema/generated client 已表达
  `preflight / approvedExecute / approvalToken / typed approval descriptor`；Rust current provider turn 复用 canonical
  `action_required`，Browser 固定 `allow_once / decline / cancel` 且关闭 session cache；Electron Host 对危险 click、敏感
  fill 与 Enter 做 preflight，并以一次性 token 校验原 thread/turn/call/tool/tab/view/WebContents/snapshot/node 后执行。
  最新 `--scenario approval` 中首次 `allow_once` 后 mutation count 为 1，fresh snapshot 的第二次危险 click 经 GUI
  `decline` 后仍为 1，user tab `released`、debugger detached，console/page/invoke error 与 production mock fallback 均为
  零；此前 canonical turn 曾因并行 read-model 终态投影漂移为 `interrupted`，形成 14/15、退出码 `1` 的中间结果；
  该结果已由后续 15/15 重跑证据取代，未放宽断言，也未覆盖并行 `runtime/read_model.rs` / `runtime/load_context.rs`。
- 2026-08-20：新增 native 用户接管 Gate B。`--scenario user-control` 在 canonical approval 等待期间直接点击同一
  `WebContentsView`，Host 立即 detach debugger、将 `pageRevision 4 -> 5`、`activeTurnId=null`、`controlOwner=user`，
  并清除 snapshot 与 pending token；随后 GUI 的旧 `allow_once` 被拒绝，mutation count 在用户点击后和 stale approval 后
  均为 1。canonical turn 以 `interrupted` 收口，12/12 断言、console/page/invoke error 与 production mock fallback 门禁
  全部通过，退出码 `0`。脱敏 evidence 位于
  `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-user-control-summary.json`。
- 2026-08-20：本轮 Host + Gate assertion 22/22、`npm run typecheck:electron`、`npm run test:contracts`、
  `npm run governance:scripts`、`npm run verify:gui-smoke` 和 lifecycle Gate B 均通过。最新
  `npm run smoke:agent-runtime-current-fixture` 已通过 history/cache、streaming、Electron guard、unknown Item 与 Claw
  首页热路径，但最终在 Coding Workbench recovery 退出 `1`：App Server recovery turn 已为 `completed` 且成功预览可见，
  GUI 未出现 recovery assistant 文本。该并行投影阻塞不在 Browser 写集，不能把聚合门禁标记为全绿。
- 2026-08-20：收口 Browser P0 current-owner negative guard。生产 prompt 不再写入旧
  `mcp__lime-browser__*` 正向工具名，测试 fixture 改用中性的 `mcp__site__*` 命名；新增
  `src/lib/governance/browserCurrentBoundary.test.ts` 扫描 Browser Host、App Server dynamic-tool host、typed gateway、
  Right Surface 与通用 prompt，阻止旧 MCP/Canvas/BrowserSessionRef owner 回流。Browser 相关治理与 prompt 定向测试
  41 项通过，`git diff --check` 通过；protocol/schema/generated client 与历史 fixture 清退仍未完成。
- 2026-08-20：重跑 Browser action approval Gate B。最新并行 read-model 终态投影已稳定为 `completed`，真实 Electron
  `--scenario approval` 15/15 断言、退出码 `0`；首个危险 click 经 `allow_once` 仅执行一次，第二次 fresh snapshot
  经 `decline` 不产生 mutation，tab release、debugger detach、console/page/invoke error 与 production mock fallback
  均通过。证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-approval-summary.json`。
- 2026-08-20：修复 Browser Right Surface 收起/恢复时已有 route 被重新导航的问题：`BrowserTabHost` 对已有 tab remount
  只恢复 native view bounds/visibility，不再次传入初始 URL；补充 Electron Host 回归，确认同一 WebContents 与当前 URL
  保留。Gate A fixture 同时等待导航终态 `isLoading=false`，避免加载完成覆盖 zoom。重跑
  `npm run smoke:browser-runtime-gate-a -- --timeout-ms 120000 --interval-ms 500`，真实 Electron fixture 在不启动 Agent
  turn 的前提下完成首屏 canonical tab、同 tab 导航、查找、缩放恢复、新建/选择/关闭 tab、收起/恢复、session/thread
  identity 稳定、桌面视口 resize、可用 viewport 和横向溢出检查，16 项断言全部通过；console/page error 为零。证据位于
  `.lime/qc/gui-evidence/browser-runtime-gate-a/browser-runtime-gate-a-summary.json`，proofLevel 明确为 `Gate A`，不宣称
  Agent 与用户共享 WebContents。
- 2026-08-21：Browser 状态带补齐稳定 `data-browser-workspace-status` 语义与 live-region/alert 角色，覆盖
  `loading`、`host-unavailable`、`load-error`、`host-error`、`permission-blocked` 与下载终态；新增
  `BrowserWorkspaceStatus.test.tsx`，逐项检查 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR` 资源、插值文案、状态
  identity 和非 viewport 布局护栏。该结果属于 Gate A 组件级证据，尚未替代真实 Electron 的多语言/状态截图矩阵。
- 2026-08-21：补充 `BrowserWorkspace.test.tsx` 的真实 `load-failed` 事件回归，确认当前 native tab 的 DNS 失败会进入
  `load-error` 状态带并保留错误详情，其他 tab 的同类事件被忽略；Browser Workspace/Status 定向回归 7 tests、根
  typecheck、Prettier、`git diff --check` 与真实 Electron Gate A projection 16/16 通过。
- 2026-08-21：按 source 拆分 `load-error` 与 `host-error`，避免 Host/IPC 操作失败被误记为页面加载失败；重跑
  `npm run verify:gui-smoke`，SHELL-01 真实 Electron summary 为 pass、退出码 `0`。最终 Browser 定向回归 7/7、Gate A
  16/16、根 typecheck、docs boundary、Prettier 与 scoped `git diff --check` 全部通过；仍未生成五语言逐项截图或状态
  matrix，不扩大本轮 proof claim。
- 2026-08-22：Browser Host 补齐受控下载 artifact 注册表：completed download 仅向 Renderer 投影 `artifactRef` 与脱敏 metadata，绝对落盘路径保留在 Electron；Host 异步调用现有 App Server `artifact/write` 写入 sidecar。`openArtifact`、`revealArtifact`、`copyArtifactRef`、`uploadArtifact` 均只接受该 ref，不能由 Renderer 或 Agent 提供任意本机路径。clipboard read/write、permission grant、artifact open/reveal/copy、上传均使用既有 dynamic-tool `preflight -> approvedExecute` 一次性 token，结果带 thread/turn/session/tab/view/WebContents evidence；turn completion 与连接断开撤销所有 pending permission callback 与 approval。新增五语言 permission-pending 文案。Electron Host/Dynamic Tool/Status 定向回归 46 tests、根 typecheck 与 diff check 通过。
- 2026-08-23：完成真实 Electron artifact Gate B 与五语言生命周期截图矩阵。`browser-runtime-electron-gate-b-artifact-summary.json` 证明 completed download、`artifact/write` sidecar、artifact ref copy、action approval、turn-scoped grant/evidence 和 terminal release；open/reveal/clipboard/upload 仍明确为 Host 定向单测边界。`browser-runtime-locale-matrix-summary.json` 的五语言 rendererLocale、35 张截图、七个生命周期状态与既有行为证据全部通过，console/page error 为零。
- 2026-08-23：按确认完成旧 Browser 正向路径物理清理：删除 `scripts/playwright/` 旧交互脚本与 package、Browser Assist 旧入口资源，并从 scripts 冻结基线与架构导航移除目录；负向回流守卫、历史 evidence、current `browserSessionId` 身份字段保留。Gate B fixture 增加一次 stale claim 后的 `openTabs -> claimTab` 重试，宿主继续 fail-closed；重跑 lifecycle/artifact Gate B、五语言矩阵、`verify:gui-smoke`、`smoke:agent-runtime-current-fixture`、`test:contracts`、legacy/scripts governance、bridge health 均通过。

## 当前结论

- 当前 Browser 纵切完成度 100%；owner/identity/Host 基础纵切、reverse-request 回环、同 tab Gate B、native 用户接管与 stale-control/reclaim、turn complete/cancel cleanup、真实 sidecar disconnect、window-close、permission block、download cancel/completed artifact、Browser action approval、turn-scoped grant/evidence、历史只读恢复、五语言截图矩阵、current-owner negative guard 与旧正向路径物理清理均已落地。
- 用户闭环：空白 Claw 任务会先补齐 canonical owner，再进入 Browser mount；Agent Browser 工具不再永久停在 `inProgress`，真实 Electron 已完成 `openTabs -> claimTab -> observe -> final assistant`。
- Gate A：独立 Browser projection 场景已形成 `Gate A` 证据并通过 16 项 chrome/identity/layout 断言，包含收起/恢复后同一 session/tab/thread/address；五语言真实 Electron 生命周期截图矩阵已完成，error/permission/download/approval/takeover/released 行为引用 Gate B summary，旧 files geometry visual matrix 继续保持 dead fixture，不作为 Browser Gate A/B。
- Gate B：核心 identity/dynamic-tool round trip、同一 `WebContentsView`、用户地址栏导航同 tab、用户导航和 native 点击后的 stale snapshot/token 拒绝、重新 claim/observe、Browser action `allow_once/decline` 的 mutation 安全性、cancel/disconnect debugger detach/release、window-close、permission block、download cancel 和 renderer-process-gone route cleanup 已有真实证据；`user-control` 为 12/12 pass，最新 approval 为 15/15 pass。
- 通用门禁：Host/Gate 定向回归、Electron typecheck、contracts、scripts governance、legacy report、GUI smoke、Browser lifecycle/artifact Gate B、五语言矩阵、Agent current fixture 与 Bridge health 均取得退出码 `0`；跨平台打包与 live provider 不在本轮 claim boundary。
- 后续只保留能力扩展：open/reveal/clipboard/upload 继续保持 Host 定向单测 claim boundary，不能扩大为未执行的 live provider 或 packaged 跨平台证据；旧 owner 与旧正向 mock 不得恢复。
