# Browser Workspace 同页控制路线图

更新时间：2026-08-21

状态：Active。本文只记录稳定的产品目标、架构边界、阶段和验收合同；逐刀写集、当前进度、验证日志与阻塞统一记录在 `internal/exec-plans/browser-runtime-right-surface-plan.md`。

## 1. 主目标

Browser 只有一个产品现场：

```text
用户可见页面 == Agent 操作页面 == 同一个 Electron WebContentsView
```

Renderer 与 Agent 是两条控制路径，不是两套 Browser。两条路径必须汇入同一个 `BrowserTabHost` route：

```text
User
  -> Renderer Browser Workspace
  -> browser_tab_* host command -----------------------+
                                                       |
Agent turn                                             v
  -> RuntimeCore / tool-runtime                  BrowserTabHost
  -> App Server item/tool/call reverse request          |
  -> Electron AppServerDynamicToolHost -----------------+
                                                       |
                                                       v
                                same WebContentsView + webContents.debugger
```

App Server 持有 thread、turn、tool lifecycle、Browser 产品身份和 read model；Electron 持有原生 view、window、debugger、权限、上传和下载等宿主能力；Renderer 只投影状态并发送用户操作。任何层都不得再维护第二个 Browser 执行体或状态机。

### 1.1 用户结果

1. Browser 始终在 Agent Workspace 的 Right Surface 内打开，不跳到系统浏览器。
2. 用户与 Agent 看到和操作同一个网页现场，URL、标题、像素和交互结果一致。
3. 用户可以随时观察、接管、恢复或终止 Agent 控制，不产生幽灵操作。
4. 网页动作可审批、可审计、可恢复，敏感信息不会进入普通 evidence。
5. Browser 在 macOS 与 Windows 上都走 Electron current bridge，不依赖外部浏览器调试环境。

### 1.2 非目标

- 外部 Chrome / Edge extension、remote debugging port 或任意外部 CDP target。
- Canvas Browser、DOM 镜像、截图假页面、`<webview>` 或 iframe Browser Runtime。
- 复制 Codex Desktop 的专有 plugin runtime、Node REPL、opaque browser id 或扩展协议。
- 恢复 `BrowserSessionRef`、旧 `browserSession/*`、`mcp__lime-browser__*` 或 Browser Runtime Settings。
- 生产 mock、Renderer fallback、第二个 Browser daemon 或兼容包装层。

### 1.3 2026-08-20 当前 checkpoint

用户反馈的“Browser 一直正在打开网页”包含两个连续阻塞。第一处是空白 Claw 任务先显示 Right Surface、canonical Thread 尚未落成，导致 Browser identity 无合法 owner；当前 Renderer 会先确保 `runtimeSessionId -> canonical threadId`。第二处是 runtime 已生成 `dynamic_tool.requested`，但 App Server v2 projector 拒绝该内部事件，事件泵因此未创建 `item/tool/call` reverse request；UI 已收到 `item.started`，所以工具永久显示 `inProgress`。当前 projector 接受该内部生命周期但不向 GUI 发送冗余通知，App Server client 持续读取 server request，Electron Host 在连接与 sidecar restart 后安装 dynamic-tool handler，并隐藏 Host 已消费请求的 `serverRequest/resolved`。

Browser 核心 Gate B 已通过：`npm run smoke:browser-runtime-electron-gate-b -- --timeout-ms 60000` 在真实 Electron、preload/IPC、`app_server_handle_json_lines`、App Server RuntimeCore/provider/read model 上完成 `browser__openTabs -> browser__claimTab -> browser__observe -> final assistant`。Agent observation 与用户可见 Workspace 的 `browserSessionId/tabId/threadId/webContentsId` 全等，active turn 对齐，native `viewId/windowId/ownerWebContentsId` 有效；Renderer 地址栏随后导航同一 tab，turn 为 `completed`，console/page error 和生产 mock fallback 均为零。脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-summary.json`。

该结果只完成 P1 核心 identity/dynamic-tool round trip，不等于完整 Gate B。旧 `right-surface-visual-matrix` 仍是 Canvas/CDP fixture，不能作为 Gate A/B；current-owner negative guard 已覆盖 Browser Host、typed gateway、Right Surface 与 prompt，但 protocol/schema/client 的旧链清退、artifact/grant、直接 destroyed WebContents 和历史恢复仍需扩展真实 Electron evidence。

同日 active-turn 生命周期 Gate B 已通过：在同一真实 Electron turn 内完成初次 observe、用户地址栏导航、旧 snapshot mutation fail-closed、重新 claim/observe 和 terminal cleanup，并通过 renderer-process-gone 触发 Browser route 关闭。用户导航后的 `controlOwner=user`、page revision 递增、重新 observe 使用新 snapshot；session/tab/thread/webContents identity 全程不变，turn `completed` 后 user tab `released`，debugger `attached -> detached`，25 项断言全部通过，console/page error 为零。脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-summary.json`。直接 `destroyed` 事件仍由 Electron Host 单测覆盖。

Electron Host 现已把 sidecar stop、退出、重启失败和 stale connection 统一映射为 Browser disconnect cleanup：user-origin tab release 并 detach debugger，Agent-origin 无 mark tab close；native `destroyed` 与 `render-process-gone` 均回收 route 并发出 close event。真实 disconnect Gate B 已向当前 sidecar 发送 `SIGTERM`，证明 pending provider response 未完成即关闭、同一 user tab release、debugger detach、无 provider 续发且生产 mock 命中为零。此前 Electron main 高 CPU 的根因是 `AppServerConnection` 在 terminal transport error 后因持续 server-request handler 立即重启 read pump；connection 现将非 timeout transport error 固化为 closed 终态，不再自旋或饿死 restart timer/Renderer IPC。12 项断言与命令退出码均通过，脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-disconnect-summary.json`。该场景不证明 restart 后原 turn 自动恢复。

同日验证收口：`npm run test:contracts` 已取得明确退出码 `0`。Browser lifecycle Gate B 25 项、disconnect Gate B 12 项、window-close Gate B 12 项均以退出码 `0` 通过；最新 `npm run verify:gui-smoke` 的 SHELL-01 真实 Electron summary 与命令退出码均为 pass。最新 `npm run smoke:agent-runtime-current-fixture` 已通过 history/cache、streaming、Electron guard、unknown Item 与 Claw 首页热路径，但最终在 Coding Workbench recovery 退出 `1`：App Server recovery turn 已完成、成功预览可见，GUI 未投影 recovery assistant 文本。该并行投影问题不在 Browser 写集，因此不把聚合门禁标记为全绿；跨平台打包与 live provider 不在本轮 claim boundary。

Browser P0 current-owner negative guard 已补齐：生产 prompt 不再写入旧 `mcp__lime-browser__*` 正向工具名，测试 fixture 使用中性的 `mcp__site__*` 命名；`browserCurrentBoundary.test.ts` 扫描 Browser Host、App Server dynamic-tool host、typed gateway、Right Surface 与通用 prompt，阻止旧 MCP/Canvas/BrowserSessionRef owner 回流。该守卫不等于 protocol/schema/generated client 清退，后者仍在 P3 待办。

Browser terminal cleanup 已增加 fail-closed 状态门禁：Electron Host 只接受带有终态 status 的 `turn/completed` 通知来 release/close Browser route，`inProgress` 或缺少 status 的异常通知不会提前释放控制；该边界已有 Electron 定向回归覆盖。真实 Electron cancel Gate B 也已通过：`--scenario cancel` 由 GUI 停止按钮经 current Electron IPC 发出 `turn/interrupt`，read model 投影为 `interrupted`，pending provider response 被取消，同一 user tab 保留原 `webContentsId` 后进入 `released`，debugger `attached -> detached`。15 项断言、console/page/invoke error 和生产 mock fallback 门禁全部通过，命令退出码为 `0`；脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-cancel-summary.json`。

真实 Electron window-close Gate B 同样已通过：`--scenario window-close` 在 Agent claim/observe 且 debugger attached 后关闭 Browser 所属 owner `BrowserWindow`，Renderer page、owner window 和内嵌 Browser WebContents 均关闭；测试观察到 `browser-tab-closed(reason=window-closed)`，thread/tab/view identity 与关闭前 route 一致，provider 无续发请求。12 项断言、console/page error 门禁通过；脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-window-close-summary.json`。

真实 Electron permission Gate B 已通过：`--scenario permission` 由同一 native page 调用 geolocation，Electron permission handler 返回 `User denied Geolocation` 并发出 `decision=blocked`；Host 补齐的 session/thread/tab/view/WebContents/owner/window identity 与 GUI 当前 tab 全等，permission 状态带位于 native viewport 外，Host bounds 与 Renderer viewport 对齐，turn 以 `interrupted` 收口。11 项断言、console/page error 与生产 mock fallback 门禁全部通过；脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-permission-summary.json`。Electron 的 `navigator.permissions.query` 可能仍报告 `granted`，授权结果必须以真实 API callback 为准。

真实 Electron download Gate B 已通过：`--scenario download` 由同一 native page 触发 data URL 下载，Electron `will-download` 发出同一 `downloadId` 的 `started -> cancelled`，GUI 当前 tab 显示取消状态且不遮挡 native viewport；evidence 不含 `savePath`、macOS 或 Windows 用户目录，turn 以 `interrupted` 收口。11 项断言、console/page error 与生产 mock fallback 门禁全部通过；脱敏证据位于 `.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-download-summary.json`。fixture 主动取消下载，因此该结果不证明文件落盘、artifact ref、完成后的 open/reveal 或 packaged 跨平台行为。

真实 Electron action approval Gate B 已通过：`--scenario approval` 在同一用户可见 `WebContentsView` 中观察危险
`Delete account` 目标，经 `preflight -> canonical action_required -> GUI allow_once -> approvedExecute` 恢复同一
tab/snapshot/action，页面 mutation count 精确变为 1；fresh observe 后第二次危险 click 产生新审批 identity，GUI
`decline` 后 mutation count 仍为 1。Browser approval 只提供 `allow_once/decline/cancel`，不提供
`allow_for_session`；canonical turn 以 `completed` 收口，tab release、debugger detach、console/page/invoke error
与 production mock fallback 门禁全部通过，最新场景为 15/15、退出码 `0`。脱敏 evidence 位于
`.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-approval-summary.json`。该场景不证明
上传、下载完成、任意脚本、permission grant、artifact open/reveal 或跨平台 packaged 行为。

真实 Electron native 用户接管 Gate B 已通过：`--scenario user-control` 在上述审批等待期间直接点击同一
`WebContentsView`，Host 将 `pageRevision 4 -> 5`、`activeTurnId=null`、`controlOwner=user`，并立即 detach debugger、
清除 snapshot 与 pending approval token。随后点击旧 `allow_once` 得到 stale token 失败，页面 mutation count 在用户点击后
和 stale approval 后均为 1；canonical turn 以 `interrupted` 收口。12/12 断言、console/page/invoke error 与 production
mock fallback 门禁全部通过，退出码 `0`；脱敏 evidence 位于
`.lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-user-control-summary.json`。该场景证明
mouse/keyboard native input 的 Host 接管合同，不代表触控输入或 packaged 跨平台验证。

独立 Browser Gate A projection 已通过：`npm run smoke:browser-runtime-gate-a -- --timeout-ms 120000 --interval-ms 500`
在真实 Electron fixture 中只建立 canonical Right Surface、不启动 Agent turn，完成首屏 tab、同 tab 导航、查找、缩放、
新建/选择/关闭 tab、收起/恢复、session/thread identity 稳定、桌面视口 resize、可用 viewport 和横向溢出检查，16 项断言、
console/page error 均通过。
证据位于 `.lime/qc/gui-evidence/browser-runtime-gate-a/browser-runtime-gate-a-summary.json`，明确为 `proofLevel=Gate A`，
不能替代同 WebContents 的 Gate B；五语言逐项截图和 loading/load-failed/permission/download/approval/
takeover/Host unavailable 状态矩阵仍待补齐。

2026-08-21 已先补齐状态投影的组件级合同：loading、host unavailable、load error、host error、permission blocked 与 download
状态带现在带有稳定 `data-browser-workspace-status`、`role`/`aria-live` 语义，并由
`BrowserWorkspaceStatus.test.tsx` 逐项校验五个支持语言的资源和插值文案。该结果只推进 Gate A 的可观测性与无障碍
护栏，尚未替代真实 Electron 的逐语言截图和状态矩阵。

同日补充 Workspace 的 `load-failed` 事件回归：只有当前选中 native tab 的完整 identity 能进入 `load-error` 状态带，
DNS 错误详情会保留，其他 tab 的失败事件会被丢弃；Browser 定向回归 7 tests 与真实 Gate A projection 16/16 均通过。
Host/IPC 操作异常则单独投影为 `host-error`，并已通过 SHELL-01 真实 Electron smoke（退出码 `0`）；本轮仍不把组件
级多语言断言或单次 smoke 扩大为五语言状态截图矩阵。

## 2. 事实源与参考边界

实现决策按以下优先级取事实：

1. Lime current 代码，以及 `internal/aiprompts/architecture.md`、`commands.md`、`governance.md`。
2. 本机 Codex Desktop Browser plugin 的可观察行为合同。
3. `/Users/coso/Documents/dev/rust/codex` 的 Thread/Turn/Item、工具、审批、恢复和 App Server 语义。
4. Electron `WebContentsView` 与 `webContents.debugger` 官方合同。

本机分析时的 Codex Desktop 参考版本：

```text
Application: /Applications/ChatGPT.app
Bundle ID: com.openai.codex
Version: 26.810.52044
Chromium: 151.0.7922.137
```

主要行为证据位于：

```text
/Applications/ChatGPT.app/Contents/Resources/plugins/openai-bundled/plugins/browser/
  skills/control-in-app-browser/SKILL.md
  scripts/browser-client.mjs
  docs/api.json
  docs/tab-cleanup-iab.md
  docs/tab-claiming-iab.md
  docs/api-use-behavior.md
  docs/browser-control-interruption.md
  docs/browser-safety.md
  docs/confirmations.md
  docs/visibility.md
  docs/screenshots.md
  docs/file-uploads.md
  docs/webmcp.md
```

安装包和专有 plugin 只提供行为证据，不成为 Lime 的依赖、vendor、协议或运行时事实源。Codex Desktop 同时支持 in-app Browser 和外部 Chrome/Edge；Lime 只对齐 in-app Browser 合同。

### 2.1 必须对齐的行为合同

| Codex Desktop 行为                                          | Lime 合同                                                                                  |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 明确选择 in-app Browser 后保持同一 browser family           | Right Surface Browser 是显式产品 surface，不能静默切到系统浏览器、外部 Chrome 或 connector |
| Browser binding 可复用，tab handle 可能 stale               | session 与 tab 生命周期分离；stale tab 必须 fail closed，不能偷偷创建替代页面              |
| claim user tab 时核对 browser、tab、title 和 URL snapshot   | claim 必须携带不可歧义的 snapshot/version，页面变化后拒绝旧 claim                          |
| Agent tab 在 turn 结束时关闭，claimed user tab 默认 release | `turnEnded` 必须 detach、close/release，并清理 turn-scoped mark 与资源                     |
| 后台执行是默认，显式观察时才显示                            | visibility 是产品状态，不能通过销毁或重建页面实现                                          |
| 每次交互后读取最便宜的新状态                                | 默认循环为 `observe -> act -> observe`，locator 必须来自最新 snapshot                      |
| 页面内容和 WebMCP 描述均是不可信输入                        | 高风险动作在 action-time 确认，页面文字不能替代用户授权                                    |
| 用户接管会中断 Agent 控制                                   | 用户操作必须使旧 turn 控制权失效；Agent 重新 claim/observe 后才能继续                      |
| screenshot、file chooser、download 都绑定当前 tab           | 截图、上传和下载必须返回受控 artifact ref，不暴露本机绝对路径                              |
| history 高隐私，WebMCP handle 可 stale                      | 两者均是 current tab 的受控能力，不能成为第二个 Browser owner；P6 前冻结                   |

## 3. 唯一架构与 Owner

### 3.1 分层职责

| 层                                  | 唯一职责                                                                        | 禁止事项                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Renderer Browser Workspace          | 显示 Browser chrome、状态和 App Server projection；发送用户命令                 | 生成 canonical 产品身份、直接调用 CDP、自建 tool lifecycle、生产 fallback |
| Electron `BrowserTabHost`           | 维护 tab/view/WebContents route、bounds、visibility、debugger 和幂等终态        | 持有第二套 thread/turn 业务状态、绕过 App Server 执行业务动作             |
| Electron `AppServerDynamicToolHost` | 将 connection-owned `item/tool/call` 绑定到 frozen host capability 和同一 route | 按 URL 猜 tab、跨 window 操作、复用已消费 call identity                   |
| App Server / RuntimeCore            | 创建 Browser 产品身份，编排 turn/tool/approval，写 read model 和 evidence       | 把 Electron 变成第二后端、恢复旧 `browserSession/*` 协议                  |
| `tool-runtime`                      | 定义工具、权限、执行策略与 lifecycle                                            | 直接拥有 WebContents 或 Renderer UI 状态                                  |

### 3.2 Current owner 路径

| 能力                                   | Owner                                                                            |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| 用户可见 Browser Workspace             | `src/components/agent/chat/workspace/right-surface/browser/BrowserWorkspace.tsx` |
| Right Surface 集成                     | `RightSurfaceBrowserPanel` + `WorkspaceRightSurfaceHostRuntime`                  |
| Renderer typed gateway                 | `src/lib/api/browserTab.ts`                                                      |
| 原生 `WebContentsView` 能力            | `electron/embeddedBrowserHost.ts`                                                |
| Browser 产品 route 与 Agent 控制       | `electron/browserTabHost.ts`                                                     |
| App Server reverse request host        | `electron/appServerDynamicToolHost.ts`                                           |
| Thread/Turn/Item 与 Browser read model | `lime-rs/crates/app-server/**`、`agent-runtime`、`thread-store` 对应 owner       |

`embeddedBrowserHost` 只提供原生 view 原语，`BrowserTabHost` 才是 Browser 产品 route owner。前者不得演化为另一套 session/runtime，后者也必须在扩展 observe/action 前按 route/lifecycle、observation、action、policy 拆分单一职责。

### 3.3 不变量

1. 一个 `tabId` 在活动生命周期内只映射一个 `viewId/webContentsId`。
2. 用户看见的 URL、title 和像素必须来自 Agent 正在操作的同一个 WebContents。
3. Renderer 收起、恢复、切 tab 和 resize 只改变 bounds/visibility，不改变产品身份。
4. Agent mutation 必须绑定 active turn、route 和最新 snapshot；身份漂移或 snapshot 过期时 fail closed。
5. 用户操作优先于 Agent 控制，并使旧 control lease 失效。
6. `turnEnded`、cancel、disconnect、window close 和 dispose 收敛到同一个幂等终态。
7. 历史恢复只恢复可见 tab 与 snapshot/replay，不恢复 debugger、claim 或 pending mutation。

## 4. 身份与生命周期合同

Browser canonical identity 必须贯穿 App Server read model、Electron Host route、动态工具结果和 Renderer projection：

```text
threadId             App Server canonical Thread owner
turnId               App Server 当前执行 Turn；Renderer 不生成、不猜测
browserSessionId     App Server Browser workspace identity；不是旧 CDP 连接，也不以 threadId 兜底
tabId                App Server 产品 tab identity
viewId               Electron Host native view identity
webContentsId        Electron 原生 WebContents identity
windowId             Desktop Host window identity
ownerWebContentsId   Desktop route owner，防止跨窗口操作
snapshotId           最近一次 observe 的页面版本；mutation 必须引用它
actionId             单次动作 identity，用于审批、恢复和 evidence join
```

身份创建规则：App Server 创建 `browserSessionId/tabId`；Electron 创建并回写 `viewId/webContentsId/windowId/ownerWebContentsId`；Renderer 只消费 projection，不得使用 `crypto.randomUUID()`、React identity、scene id 或 thread id fallback 创建产品身份。

执行前必须重新校验 `threadId + turnId + browserSessionId + tabId + viewId + webContentsId + windowId + ownerWebContentsId`。claim user tab 还必须校验用户所见的 `title + url + page revision`；click/fill/press 等 mutation 必须校验 `snapshotId + backendNodeId`。route 不存在、turn 过期、窗口不匹配、WebContents destroyed、snapshot 过期或身份漂移时必须失败，不能自动重建一个“看起来能用”的 session，也不能在旧坐标上重放动作。

### 4.1 Tab 来源与 turn 终态

| 来源             | 创建方式                   | `turnEnded` 默认行为           |
| ---------------- | -------------------------- | ------------------------------ |
| Agent tab        | `browser__newTab`          | 关闭                           |
| User tab         | Workspace 新建或恢复       | 释放 Agent 控制，页面保留      |
| Claimed user tab | `browser__claimTab`        | 释放 Agent 控制，页面保留      |
| Handoff tab      | `browser__markHandoff`     | 保留页面，后续 turn 重新 claim |
| Deliverable tab  | `browser__markDeliverable` | 保留页面并释放 Agent 控制      |

幂等终态顺序：detach debugger，终止 pending debugger/action waiter，撤销 turn-scoped permission/download/upload/clipboard grant，清理 active turn 和 mark，再按 tab origin/mark 执行 close 或 release。重复调用不得改变已完成的终态。

## 5. 命令边界

### 5.1 Renderer -> Desktop Host

Renderer 只通过 `src/lib/api/browserTab.ts` 调用 `browser_tab_*`：

```text
mount / set_bounds / navigate / reload / stop
find_in_page / stop_find_in_page / set_zoom
go_back / go_forward / select / close
```

这些命令只负责原生 view mount、bounds、visibility、用户导航和 UI 状态。Renderer 页面不得直接调用 CDP、外部 Chrome 或旧 `browserSession/*` gateway。

### 5.2 Agent -> App Server -> Desktop Host

```text
Agent dynamic tool
  -> RuntimeCore / tool-runtime
  -> App Server item/tool/call
  -> Electron AppServerDynamicToolHost
  -> Electron BrowserTabHost
  -> same WebContentsView.debugger
```

动态工具命名空间为 `browser`，runtime name 为 `browser__<tool>`。P1 最小工具集：

```text
openTabs / newTab / claimTab / releaseTab
goto / observe / screenshot
click / fill / press
markHandoff / markDeliverable
```

旧 `mcp__lime-browser__*`、旧 `browser_*` MCP alias 和外部 Chrome extension relay 不得接入 current catalog。

### 5.3 跨层同步

任何跨层 Browser method 变更必须在同一变更集中更新 App Server protocol/schema、Rust handler/runtime、generated TS types、typed client、Electron Host/preload、Renderer gateway、catalog、fixture、negative guard 和 contract tests。

## 6. Browser Workspace 产品合同

```text
Right Surface / Browser
  tab strip       favicon / title / control state / close / new tab
  toolbar         back / forward / reload-stop / address / find / zoom
  status band     load error / permission / download / approval / takeover
  native viewport exactly one mounted WebContentsView
```

Browser 是 Right Surface 的一级工作区，不嵌套大型侧栏、解释卡片或第二套导航。网页像素只来自 native viewport；Renderer toolbar/status 使用稳定尺寸。`WebContentsView` 属于原生层，不能依赖普通 DOM z-index 在其上方叠加关键 UI；需要用户操作的状态必须放在 viewport 外的固定 band，或由 Host 调整 bounds 后展示。

| 区域              | 产品合同                                                                               |
| ----------------- | -------------------------------------------------------------------------------------- |
| Tab strip         | favicon/title/selected/loading/control owner 可扫描；关闭与新建不改变其他 tab identity |
| Navigation        | 使用图标按钮与 tooltip；reload/loading 稳定切换为 stop，不改变 toolbar 尺寸            |
| Address           | 展示 Host 返回的 URL/origin 与安全状态，不由输入框猜测导航成功                         |
| Find/Zoom         | per-tab 状态；切 tab 后恢复；五语言最长文案不挤压 viewport                             |
| Status            | error、permission、download、approval、takeover 均有明确来源、动作和终态               |
| Viewport          | 一个 tab 对应一个 `WebContentsView`；切换/收起/resize 不使用截图占位冒充 live page     |
| User control      | 用户操作中断旧 Agent control；UI 可显示 owner 状态，但不暴露 turn id、CDP 等内部术语   |
| Empty/Unavailable | 首屏提供可操作地址栏；Host 不可用时明确失败，不显示静态网页 mock                       |

## 7. 操作、安全与人工接管

Browser 动作按目标站点、具体动作和数据敏感性决定审批，不把所有 click/type 固化为同一种策略：

| 动作                                             | 默认策略                                                                        |
| ------------------------------------------------ | ------------------------------------------------------------------------------- |
| URL/title/DOM/截图/滚动                          | 已授权 tab 内自动执行                                                           |
| 普通导航                                         | 遵循站点访问和跨 origin policy                                                  |
| 普通 click、非敏感草稿输入                       | 无不可逆副作用时自动执行                                                        |
| 密码、OTP、token、支付、身份信息                 | 输入即传输；动作时说明数据和目标站点并确认，必要时人工接管；不进入普通 evidence |
| 提交、发布、删除、发送、购买、授权               | 动作时确认；批准后恢复原 action；页面文字不能替代用户授权                       |
| 上传                                             | 明确 artifact ref、目标站点和控件；使用短时、turn-scoped grant                  |
| 下载                                             | 返回受控 artifact ref；运行新下载的软件必须另行确认                             |
| camera/microphone/geolocation/account permission | 窄范围预授权或动作时确认                                                        |
| CAPTCHA、修改密码最终提交、安全拦截              | 按策略确认或交给用户；禁止绕过 paywall 和安全 interstitial                      |
| 任意 JavaScript/full CDP                         | 高风险 capability，默认关闭                                                     |

审批必须形成 `required -> approved/denied -> resume/terminal` 的真实 action lifecycle，并在同一 tab、同一 snapshot lineage 上恢复原 action。关键词正则、“Enter 一律转人工”或 Renderer 伪造 tool result 都不能作为完成实现。

Browser dynamic tool 已把 action approval 收回 `agent/current_provider_turn` 与 `tool-runtime` 的既有 execution approval owner。由于 click 风险需要 Host 读取 AX node/页面动作语义，当前采用两阶段 `preflight -> required/approved/denied -> execute/resume`，而不是把所有 `browser__click` 固定为 requires approval；Host 对危险 click 文案、敏感 fill 和 Enter 只生成 fail-closed 风险事实，不保存用户决策。`item/permissions/requestApproval` 只负责 filesystem/network grant，不得冒充 Browser action approval。

两阶段协议固定为 `item/tool/call(phase=preflight)` 返回 completed 或 typed approval descriptor；descriptor 进入
canonical `action_required`，其 `toolFamily/contractKey` 均为 `browser_action`，且只允许
`allow_once/decline/cancel`。批准后 Runtime 以同一 call identity 发起
`item/tool/call(phase=approvedExecute, approvalToken)`；Electron 校验一次性 token 与原 tab/view/WebContents/
snapshot/node lineage 后执行一次。decline 不产生第二阶段请求，cancel 沿用 turn cancellation；任何 stale 或 token
重放都 fail closed。

用户在页面中导航、点击、输入或接管时，Host 必须撤销当前 Agent control lease、detach 或暂停对应 debugger 控制，并发出可投影的 interruption。Agent 只能重新 claim 并 observe 新 snapshot 后继续。

## 8. Evidence 与恢复合同

每次 Browser action/evidence 至少关联：

```text
threadId / turnId / browserSessionId / tabId
viewId / webContentsId / windowId / ownerWebContentsId
snapshotId / actionId / origin
action / status / reasonCode
urlBefore / urlAfter / evidenceRefs
startedAt / finishedAt
```

cookie、token、password、authorization header、本机绝对路径和页面敏感正文不得进入模型正文或普通 evidence。DOM、screenshot、network、console 必须有 payload cap、redaction 和 retention policy；模型摘要、结构化结果和 UI 私有 metadata 必须分离。

历史恢复只恢复 Browser tab projection、已持久化 snapshot 和 replay evidence。不得自动恢复 debugger attachment、control lease、approval grant、pending action 或下载/上传权限。

## 9. Current / Compat / Deprecated / Dead

| 分类       | Surface                                                                            | 处理                                                                                                       |
| ---------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| current    | `BrowserWorkspace` + Right Surface `browser`                                       | 唯一用户可见 Browser 页面                                                                                  |
| current    | Electron `BrowserTabHost` + `WebContentsView` + debugger                           | 唯一网页执行体和控制 route                                                                                 |
| current    | App Server JSON-RPC + `item/tool/call`                                             | 唯一 Agent 业务与宿主路由                                                                                  |
| current    | `browser__*` dynamic capability                                                    | 唯一 Agent Browser 正向工具 owner                                                                          |
| current    | canonical Browser read model/projection                                            | 唯一产品身份与历史恢复来源                                                                                 |
| compat     | 无                                                                                 | 仓库没有外部兼容负担，不新增包装层                                                                         |
| deprecated | 无                                                                                 | 旧链直接物理删除，不维持迁移期                                                                             |
| dead       | `browserSession/*`、`lime-rs/crates/browser-runtime`                               | 删除协议、schema、handler、runtime、client、fixture 和文档入口                                             |
| dead       | Canvas Workbench Browser、Object Canvas `browserSession` snapshot/replay owner     | 删除 model/view/replay/persistence、数据库、入口、i18n 和正向测试                                          |
| dead       | `BrowserSessionRef`、Browser Assist session/runtime/artifact/service-skill binding | 删除 adapter、preheat、artifact、executor binding、prompt、fixture 和 consumer                             |
| dead       | 外部 Chrome/CDP Settings、connector/profile facade                                 | 删除页面、命令、脚本和 `webview-api` 中 Browser 控制语义；通用 site-adapter API 只能保留独立 current owner |
| dead       | `mcp__lime-browser__*` 和旧 `browser_*` MCP alias                                  | 从 production/mock catalog、prompt 和正向 fixture 删除                                                     |
| dead       | production mock/fallback、第二 Browser daemon、`<webview>`/iframe Runtime          | 禁止恢复                                                                                                   |

Dead 路径不得改名为 compat。最终只允许在 owner negative guard 或不可变历史 evidence 中出现旧名称。

## 10. 分阶段实施

阶段状态和逐项进度只在执行计划维护；路线图只定义依赖与退出条件。

### P0：事实源与 canonical identity

- App Server read model 创建并投影 `browserSessionId/tabId`。
- Electron 创建并回写 native identity；Renderer 删除随机 id 与 scene/thread fallback。
- claim 校验 title/URL/page revision；mutation 校验 `snapshotId + backendNodeId`。
- owner negative guard 阻止旧 session、Canvas、Browser Assist 和 MCP 前缀回流。

退出条件：冷启动和历史恢复都从 canonical Thread/Turn/BrowserTab read model 重建同一产品身份；Renderer 不创建产品 id，Host 不持有第二份业务 session。

### P1：同一 WebContents 纵切

- Agent dynamic tool 通过 App Server `item/tool/call` 路由到 `BrowserTabHost`。
- `goto/observe/screenshot/click/fill/press` 使用用户可见 tab 的同一个 debugger。
- user tab claim/release、Agent tab close、handoff/deliverable 生命周期完整。
- `turnEnded/cancel/disconnect/window close/dispose` 收敛到幂等终态并覆盖 pending waiter。
- 用户操作中断旧控制，重新 claim 后 observe 仍读取同一个 WebContents。
- `browserTabHost.ts` 在扩展前按职责拆分，保持单一外部 owner。

退出条件：真实 Electron 中 Agent action 与用户可见页面报告相同 session/tab/view/WebContents/thread/turn identity，且用户操作、cancel 与断连都不会留下控制权。

### P2：Right Surface 与 Gate A

- Browser chrome、状态带和 native viewport 使用稳定布局。
- 新建、选择、关闭、导航、查找、缩放、收起、恢复和 resize 不改变错误的身份。
- Renderer 消费 App Server BrowserTab projection，不自建 tab/view id。
- 五语言文案、Host unavailable、loading、error、permission、download、approval 和 takeover 状态完整。

退出条件：Gate A 全部通过，并明确记录 Renderer 证据不能替代真实 Electron identity。

### P3：旧链物理清退

- 删除 Rust `browserSession/*`、`browser-runtime` crate、generated client 和专用 fixture。
- 删除 Canvas Workbench Browser 与 Object Canvas `browserSession` owner。
- 删除 Browser Assist session/runtime/preheat/artifact/service-skill binding 和正向测试。
- 删除外部 Chrome/CDP Settings、脚本、connector/profile facade 和 consumer。
- 删除 `mcp__lime-browser__*` production/mock catalog、prompt 和正向 fixture。
- 保留通用 site-adapter 能力时，必须拆成独立 current owner，不得继续借 Browser facade 存活。

退出条件：build graph、production catalog、mock catalog、fixture 和 active docs 中只剩 current owner；contracts、Browser TypeScript/Rust related tests 和治理扫描通过。

### P4：稳定观察、动作与审批

- `observe` 输出版本化 DOM/AX snapshot、stable node、frame/shadow DOM、visible/actionable 属性和 payload cap。
- click/fill/press/select/scroll/wait 使用 semantic locator 或最新 `snapshotId + backendNodeId`。
- 每次 action 后按需要重新 observe；失败先读状态，不盲目重试或退化坐标点击。
- approval/takeover/deny/resume/interruption 形成真实闭环；危险 click 的 allow-once/decline 已由真实 Electron Gate B 证明，其余敏感动作继续按风险矩阵扩展。
- permission block 与 download `started -> cancelled` 已完成 Host/Renderer/Gate B 纵切；上传、下载完成、clipboard、permission grant 和敏感数据仍需使用 turn-scoped grant 与 artifact ref 收口。

退出条件：一个真实网页流程完成 `observe -> action -> observe`；一个高风险流程完成 `required -> approved/denied -> resume/terminal`；两者均保持同一 tab 和 snapshot lineage。

### P5：Evidence、恢复与 Gate B

- action/evidence 补齐 join key、redaction、payload cap、retention 和 terminal status。
- 历史只恢复 snapshot/replay，不自动续跑 debugger、claim 或 pending mutation。
- 真实 Electron 覆盖 attach/detach、用户操作后 observe、turn 终态、stale snapshot、cancel、断连和窗口关闭。
- 生产路径 mock 命中为零，用户可见 URL/title/pixel 与 Agent observation 属于同一 WebContents。

退出条件：Gate B evidence 可追溯到 Electron Host、preload/IPC、App Server method、RuntimeCore/read model、action lifecycle 和 GUI 可见终态。

### P6：后台 Browser、WebMCP 与高级工具

P0-P5 全部完成后才能解冻 hidden background host、temporary content tabs、WebMCP 页面工具发现、device toolbar、history/settings、screenshot 区域评论和元素批注。所有高级能力仍绑定 current tab/evidence，不得引入第二 Browser owner。

建议执行依赖顺序：`P3 -> P0 -> P1 -> P4 -> P2 Gate A -> P5 Gate B -> P6`。物理清退先消除双轨，identity 与 Host 稳定后再扩展动作能力，Gate A 与 Gate B 分别证明投影和真实主链。

## 11. Gate A / Gate B 证据合同

### 11.1 Gate A：Renderer/Workspace 投影

使用显式 test-only fixture 或 Renderer projection 验证 DOM、chrome、bounds、状态和五语言文案。至少覆盖：

1. 打开 Browser 后只有一个 Right Surface、一个选中 tab 和一个 viewport mount point。
2. 新建、选择、关闭 tab；后退、前进、刷新/停止；地址栏、查找和缩放均可操作。
3. 收起、恢复、resize 后 session/tab/view identity 不变，不出现重叠、空白、第二右栏或系统浏览器。
4. loading、load failed、permission、download、approval、takeover、Host unavailable 和 pending 状态不改变稳定布局。
5. `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR` 的最长文案不溢出控件。

Gate A 只能证明 Renderer/Workspace 投影，不证明 Agent 使用同一个 WebContents，也不能用 `data-*` identity 冒充 Host identity。

### 11.2 Gate B：真实 Electron 同 tab

Gate B 必须使用真实 Electron Desktop Host、preload/contextBridge、Electron IPC、`app_server_handle_json_lines`、App Server current turn/tool chain、RuntimeCore/read model 和用户可见页面。最低身份断言：

```text
Agent action.browserSessionId   == visible sessionId
Agent action.tabId              == visible tabId
Agent action.viewId             == Host visible viewId
Agent action.webContentsId      == Host visible webContentsId
Agent action.windowId           == Host visible windowId
Agent action.ownerWebContentsId == Renderer owner webContentsId
Agent action.threadId           == canonical threadId
Agent action.turnId             == active turnId
```

至少覆盖：

1. Renderer 通过真实 preload/IPC mount 页面，Host 返回 native identity。
2. App Server current turn 发出 `item/tool/call`，Electron 按 connection/thread/turn owner 路由到同一 tab。
3. Agent navigate/click/fill 后用户在同一 WebContents 看到变化，后续 observe 引用新 `snapshotId`。
4. 用户人工操作使旧 Agent 控制失效；重新 claim 后 observe 读取同一 tab 的新状态，不创建新 session/view。
5. title/URL/revision 已变化的 claim 和旧 snapshot mutation 均 fail closed。
6. 高风险 action 进入 required，approve/deny 后产生正确终态；敏感输入和本机绝对路径不进入普通 evidence。
7. turn 结束后 debugger detach；Agent tab close、User tab release、handoff/deliverable retain 符合合同。
8. cancel、disconnect、window close、destroyed WebContents 与 pending request 都有唯一终态。
9. evidence 关联 thread/turn/session/tab/view/WebContents/action/snapshot，生产 mock 命中为零。

普通 Chrome、外部 CDP target、静态截图、Renderer `data-*`、mock bridge、external backend 或 Host 单测单独通过都不能标记 Gate B。

## 12. 验证矩阵

| 风险                     | 最低验证                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| 文档改动                 | `git diff --check -- "internal/roadmap/browser/README.md"`                                              |
| protocol/catalog/client  | `npm run test:contracts`、`npm --prefix "packages/app-server-client" run typecheck`、Rust related tests |
| Renderer consumer 清退   | Browser 定向 TypeScript/Vitest、根 typecheck                                                            |
| Electron BrowserTabHost  | Browser Host 定向测试、`npm run typecheck:electron`                                                     |
| Right Surface GUI        | Browser component tests、`npm run verify:gui-smoke`，形成 Gate A evidence                               |
| Agent dynamic capability | dynamic tool/Runtime related tests、current Agent fixture                                               |
| 真实同 tab 闭环          | Browser 专项 Electron Gate B fixture；Playwright/CDP 只用于取证，不替代产品主链                         |
| legacy/script 清退       | `npm run governance:legacy-report`、`npm run governance:scripts`、owner negative guards                 |

每次验证都在执行计划记录实际命令、scenario、identity、terminal/pending 状态和未验证原因。生产路径不得使用 mock fallback。

## 13. 完成判定

Browser 只有在以下条件全部满足后才能标记完成：

1. P0-P5 的退出条件全部满足，P6 保持冻结或作为独立后续范围。
2. Current owner 之外的 Browser 生产、mock、fixture 和 active docs 正向路径全部删除。
3. Gate A 与 Gate B 都有可复现证据，且 Gate B 明确证明用户与 Agent 操作同一个 WebContents。
4. stale identity、用户接管、审批、cancel、disconnect、window close 和历史恢复均有唯一可观察终态。
5. contracts、相关 Rust/TypeScript/Electron 测试、GUI smoke 和治理门禁通过。
6. `internal/aiprompts/architecture.md`、执行计划和 PR evidence 中的架构图确认与实际实现一致。

局部单测、Renderer 截图、普通 Chrome CDP、mock bridge 或完成度百分比都不能替代上述完成判定。
