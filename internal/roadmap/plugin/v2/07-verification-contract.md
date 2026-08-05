# Plugin v2 验收合同

状态：`active / macOS P0 core Gate B passed / release pending`

更新时间：2026-08-05

## 验收目标

证明 Plugin v2 不是静态目录页或 renderer fixture，而是从真实安装来源经过 Electron/App Server/RuntimeCore 到 Claw 用户可见状态的生产主链。

Gate A 用于快速验证浏览器投影；Gate B 才是功能验收。任何 P0 场景只有 Gate A、组件测试或 mock 证据时，一律记为未通过。

## 质量层级

| 层级        | 证明内容                                                                       | 能否替代 Gate B |
| ----------- | ------------------------------------------------------------------------------ | --------------- |
| Unit        | parser、state reducer、transaction、projection 的局部行为                      | 否              |
| Contract    | protocol、Electron IPC、preload、gateway、catalog 一致                         | 否              |
| Integration | App Server 与 filesystem/MCP/RuntimeCore 协作                                  | 否              |
| Gate A      | 浏览器或静态 fixture 的页面信息架构与布局                                      | 否              |
| Gate B      | 真实 Electron、真实 bridge、真实 App Server、真实 runtime/read model 和可见 UI | 是              |
| Release     | Gate B 加跨平台、升级、恶意输入和残留检查                                      | 是，最终门禁    |

## Gate B 必须穿过的链路

```text
Electron Desktop Host
  -> preload / IPC
  -> app_server_handle_json_lines
  -> App Server JSON-RPC
  -> plugin domain / installed store
  -> RuntimeCore / MCP / Skills / Hooks
  -> Thread / Turn / Item read model
  -> renderer gateway
  -> App Center / Claw / Right Surface
```

证据必须能关联：

```text
pluginId + marketplaceId + version/digest
threadId + turnId + itemId/toolCallId
surfaceId/resourceUri
```

若任一层使用 renderer mock、测试专用 bridge、旧 plugin worker 或手工注入 read model，则该次运行不能标记 Gate B。

## 当前 Gate B 证据

2026-08-05 已 fresh 通过单一 Plugin v2 Electron fixture：

```bash
npm run smoke:plugin-v2-current-electron-fixture
```

该 fixture 强制从源码重建 App Server sidecar，并从真实 GUI 实际经过：

```text
App Center sidebar
  -> plugin/list / bundled Browser identity
  -> window.electronAPI.dialog.open
  -> app:dialog:open / Electron dialog.showOpenDialog
  -> install review / confirm
  -> plugin/install
  -> plugin/read
  -> plugin/installed
  -> App Center menu disable / enable
  -> plugin/enabled/set(false / true)
  -> Claw model selector
  -> Plus / Plugins / installed Plugin option
  -> plugin://mcp-elicitation-plugin structured mention
  -> thread/start
  -> GUI send / turn/start
  -> McpThreadRuntime / McpClientManager
  -> mcp__plugin__mcp-elicitation-plugin__demo__release_check
  -> mcpServer/elicitation/request
  -> Renderer 表单提交
  -> provider final
  -> thread/read
  -> mcpResource/read
  -> MCP App Right Surface / WebContentsView
  -> Renderer reload restore
  -> App Center uninstall confirm
  -> plugin/uninstall / plugin/installed cleared
  -> reopen original Thread
  -> historical mention / tool item / Right Surface unavailable state
```

自动断言结果：

```text
proofLevel=Gate B
backendMode=runtime
electronPreloadBridge=true
appServerHandleJsonLinesSeen=true
appCenterBundledListVisible=true
appCenterBundledIdentityStable=true
nativeDirectoryDialogObserved=true
appCenterInstallReviewVisible=true
appCenterInstallCompleted=true
appCenterEnableToggleCompleted=true
clawFixtureModelSelected=true
clawPluginPickerVisible=true
clawPluginBadgeVisible=true
clawStructuredMentionObserved=true
clawThreadTurnItemIdentityStable=true
rendererFormVisible=true
rendererConfirmedSubmitted=true
dynamicToolCanonicalCompleted=true
runtimeInitializeProtocolVersion=2025-06-18
missingRequiredMethods=[]
legacyMcpCommandsSeen=[]
pluginWorkerHitCount=0
productionMockFallbackHitCount=0
providerFinalTextObserved=true
mcpLedgerAccepted=true
mcpAppRightSurfaceVisible=true
mcpAppResourceReadCount=4
mcpAppHtmlLoadCount=4
mcpAppToolCallCount=1
mcpAppRestoredAfterReload=true
mcpAppHistoryUnavailableAfterUninstall=true
mcpAppCanonicalIdentityStable=true
uninstallViaAppCenterCompleted=true
installedProjectionClearedAfterUninstall=true
historyReadableAfterUninstall=true
historyMcpRuntimeNotRestartedAfterUninstall=true
historyProviderNotReexecutedAfterUninstall=true
historyToolNotReexecutedAfterUninstall=true
consoleErrors=[]
```

证据位于 `.lime/qc/gui-evidence/plugin-v2-current-electron-fixture/`，包含 summary、脱敏 raw evidence 与 7 张当前 Electron 截图。人工复核确认 App Center、安装 review、Claw picker/mention、Right Surface、cold restore、卸载确认和卸载后历史均非空，无重叠、裁切、变形或主题偏离。

summary 记录应用版本、平台/架构、Electron 版本、repository commit、起止时间，以及 `plugin/source/version/digest + session/thread/turn/user item/tool item/tool call/surface/resource` 身份。MCP App resource/HTML 累计计数按“首次打开与显式恢复 2 次、Renderer reload 3 次、跨进程 cold restore 4 次、卸载后保持 4 次”精确断言；MCP tool call 始终仅 1 次。卸载后只恢复 canonical 历史 item 与 surface identity，GUI 明确显示“仅保留历史”，不挂载 WebContentsView，不读取 MCP resource，也不重启 MCP runtime、provider 或 tool。

当前 claim boundary 覆盖 bundled 列表、真实 Electron 目录选择与 App Center 安装/启停/卸载、Claw picker 与结构化 mention、manifest 选择的默认 `.mcp.json` 自动启动、真实 MCP tool、elicitation、provider final、canonical tool terminal、`thread/read`、MCP App Right Surface、Renderer reload、跨进程 cold restore 和卸载后历史不可重跑状态。`.mcp.json` 是 MCP 配置文件，唯一 Plugin manifest 仍是 `.codex-plugin/plugin.json`。

该证据不证明 repo/personal marketplace、授权、更新、管理员策略、Windows release gate 或五语言窄窗矩阵已通过。

## P0 场景矩阵

| ID    | 场景                      | 当前 Gate B | 操作                                                         | 必须观察到的结果                                                          |
| ----- | ------------------------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------------------- |
| GB-01 | bundled 列表              | PASS        | 启动 Desktop，打开 App Center                                | `plugin/list` 返回 bundled source；卡片 identity/version 与 manifest 一致 |
| GB-02 | repo/personal marketplace | PENDING     | 打开含 repo marketplace 的 workspace，再添加 personal source | 来源分区正确；同 ID 冲突按 authority/policy 给出确定结果                  |
| GB-03 | 本地安装                  | PASS        | 从详情安装未安装插件                                         | staging 校验、原子提交、installed projection 和按钮状态一致               |
| GB-04 | 安装时授权                | PENDING     | 安装 `ON_INSTALL` connector 插件                             | 安装与授权状态分开；取消授权不伪装成未安装                                |
| GB-05 | 使用时授权                | PENDING     | 调用 `ON_USE` 插件                                           | 首次真实调用时出现授权；授权后原 turn 按合同继续或明确重试                |
| GB-06 | 启用/禁用                 | PASS        | 禁用已安装插件，创建新 thread，再启用                        | 新 thread 不发现禁用能力；启用后的新边界恢复发现                          |
| GB-07 | `@plugin` mention         | PASS        | 在 Claw picker 选择已安装插件并发送                          | composer、turn、trace、tool item 使用同一 plugin ID                       |
| GB-08 | MCP tool call             | PASS        | 调用插件 MCP 工具                                            | 真实 MCP lifecycle、权限、流式 tool item 和结果完成                       |
| GB-09 | MCP/App UI                | PASS        | 工具返回 UI resource                                         | Right Surface 打开对应 App UI，surface identity 与 tool item 一致         |
| GB-10 | Browser surface           | PENDING     | 插件发出 browser intent                                      | 复用 Right Surface browser tab，不弹出第二右栏或系统浏览器                |
| GB-11 | 历史恢复                  | PASS        | 关闭并恢复 thread                                            | plugin/tool/surface 投影可恢复；不偷偷重跑旧动作                          |
| GB-12 | 更新                      | PENDING     | 安装有新版本的插件                                           | 新版本校验后原子替换；失败保留旧版本；新 thread 使用新 digest             |
| GB-13 | 卸载                      | PASS        | 卸载已使用插件                                               | catalog/installed/activation 清理；历史仍可读；共享 auth 不误删           |
| GB-14 | 管理员禁用                | PENDING     | 加载 disabled-by-admin 插件                                  | 可解释但不能启用/调用；UI 不展示无效动作                                  |

`PARTIAL` 只表示底层 current contract 或受控 fixture 已覆盖部分步骤，不能替代该行要求的真实用户动作。当前 release gate 仍为未通过。

GB-01、03、06、07、08、09、11、13 已通过 macOS 真实 Electron current 链路。其余 P0 场景、Windows 和五语言窄窗矩阵仍阻塞 release。

## 安全与鲁棒性场景

| ID     | 输入/故障                              | 预期                                                  |
| ------ | -------------------------------------- | ----------------------------------------------------- |
| SEC-01 | 缺失或非法 `.codex-plugin/plugin.json` | 拒绝解析，错误定位到字段，不写 installed store        |
| SEC-02 | `../`、绝对路径、symlink 逃逸          | 拒绝越过 package/source authority                     |
| SEC-03 | archive bomb、超大文件、文件数爆炸     | 在解包前/中按预算停止并清理 staging                   |
| SEC-04 | digest/signature 不一致                | 拒绝安装或更新，保留上一可用版本                      |
| SEC-05 | 重复 ID、source spoofing               | 按 marketplace authority 处理，不静默覆盖             |
| SEC-06 | 安装中断或进程崩溃                     | 重启后无半安装态，可重试                              |
| SEC-07 | 卸载时文件占用                         | 明确残留和重试动作，不谎报成功                        |
| SEC-08 | MCP server 启动失败                    | installed 保持，readiness 失败可诊断，不回落旧 worker |
| SEC-09 | connector token 失效                   | 进入重新授权，不重装插件，不泄漏 token                |
| SEC-10 | UI resource 非受信 origin              | Host policy 拒绝或隔离，不允许任意本地 bridge         |
| SEC-11 | Hook 请求高风险动作                    | 进入统一权限/确认，不因插件来源绕过                   |
| SEC-12 | 管理员策略运行中变化                   | 新请求停止，用户看到策略原因，历史状态保持可读        |

## UI 验收

### App Center

- All、Installed 和 marketplace 分区可切换，状态不互相污染。
- 搜索覆盖名称、描述、开发者和能力关键词。
- 卡片只保留主动作；启停、更新、卸载、授权和来源等低频动作使用详情或菜单。
- 详情展示 Skills、Hooks、Apps、MCP Servers、source、auth、privacy 和 terms。
- 使用 Lime 当前主题，不复制 Codex 配色；按钮、图标、间距和状态反馈遵循 Lime design language。
- 360px 级窄窗口、常规桌面和宽屏下无文字溢出、卡片变形、重叠或横向滚动。
- 加载、空、错误、离线、安装中、待授权、管理员禁用和更新失败状态均可恢复。

### Claw 与 Right Surface

- `@` picker 中“已安装可调用”和“可安装建议”视觉与行为不同。
- mention 显示名变化不影响稳定 plugin identity。
- 后台插件结果只增加 pending badge，不抢当前 tab。
- MCP/App UI、Browser、文件和结构化结果复用现有 Right Surface registry。
- 关闭右侧不丢 thread/tool item；恢复时不会自动重复外部动作。
- surface action 回流 current action/turn contract，网络面板中不存在私有 plugin worker 请求。

### 多语言

以下 locale 必须同时验收：

```text
zh-CN
zh-TW
en-US
ja-JP
ko-KR
```

至少对最长文案 locale 做窄宽度截图，不能只验证 key 存在。

## 协议与身份断言

Gate B fixture 应自动断言：

1. `plugin/list`、`plugin/install`、`plugin/read`、`plugin/installed`、`plugin/uninstall` 请求经过 App Server JSON-RPC。
2. Desktop 调用经过 preload/IPC 和 `app_server_handle_json_lines`。
3. Renderer 未直接读取 filesystem、marketplace JSON 或 plugin manifest。
4. runtime item 包含 plugin identity、capability identity 和 source authority。
5. UI surface 的 `resourceUri/surfaceId` 可以反查同一 tool item。
6. thread restore 后 identity 不按 display name 重建。
7. legacy command hit 为 `0`。
8. production mock fallback hit 为 `0`。
9. plugin worker process/IPC hit 为 `0`。

## 建议证据包

```text
.lime/qc/gui-evidence/plugin-v2-gate-b/<run-id>/
  summary.json
  bridge-events.jsonl
  app-server-requests.jsonl
  runtime-items.jsonl
  plugin-state-before.json
  plugin-state-after.json
  screenshots/
    app-center-list.png
    plugin-detail.png
    claw-mention.png
    tool-running.png
    right-surface.png
    restored-thread.png
```

`summary.json` 至少记录：应用版本、平台、commit、scenario、plugin/source/version/digest、thread/turn/item/surface IDs、legacy/mock hit count、开始结束时间和最终 verdict。

证据中不得包含 token、cookie、用户输入的敏感字段、绝对凭证路径或完整环境变量。

## 最低验证命令

按改动范围执行，最终 release gate 至少包括：

```bash
npm run test:contracts
npm run test:rust:related -- <affected-paths...>
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
npm run governance:legacy-report
npm run verify:local
npm run verify:app-version
```

Plugin v2 的单一 current Gate B 命令为：

```bash
npm run smoke:plugin-v2-current-electron-fixture
```

该命令稳定前不得删除旧 fixture 的最后覆盖；稳定后应删除旧 plugin runtime/UI worker smoke，而不是永久并存。

## 失败分类

| 分类                        | 判定                                  | 处理                                       |
| --------------------------- | ------------------------------------- | ------------------------------------------ |
| Product failure             | 用户流程、状态、文案或恢复不符合合同  | 修复后重跑对应场景和相邻场景               |
| Bridge failure              | preload/IPC/App Server 不通或走旁路   | Gate B 失败，禁止用 Gate A 替代            |
| Runtime failure             | MCP/Skill/Hook 未进入 current item 链 | 修复 owner，不回落 plugin worker           |
| Projection failure          | runtime 成功但 GUI 状态错误           | 修复 read model/gateway/projection         |
| Test infrastructure failure | fixture、端口、构建或环境故障         | 单独标记，不能计为产品通过                 |
| External auth failure       | 第三方服务不可用或账号缺失            | 保留本地链路证据并标记阻塞，不伪造授权成功 |

## 验收报告模板

```text
主目标：
当前阶段：
平台与应用版本：
场景：GB-xx
结果：PASS / FAIL / BLOCKED
真实链路：Electron -> preload/IPC -> App Server -> RuntimeCore -> read model -> GUI
身份：plugin / thread / turn / item / surface
legacy command hits：0
mock fallback hits：0
证据路径：
失败原因与修复：
未验证项：
current / deprecated / dead 变化：
完成度：xx%
```

## Release 阻塞条件

出现任一项即阻塞发布：

- P0 Gate B 场景失败或仅有 mock/Gate A 证据。
- Plugin identity 在 composer、runtime、read model 或 surface 之间断链。
- 任意 production path 命中旧 plugin worker、旧 manifest 或 renderer registry fallback。
- 安装/update 可能留下半安装态或覆盖上一可用版本。
- 卸载会误删共享凭证、用户文件或不可解释残留。
- macOS/Windows 任一支持平台没有安装到卸载的真实证据。
- 五语言缺失或窄窗口存在遮挡、变形和不可操作控件。
- 架构图确认、执行计划或 evidence summary 未落在仓库内。
