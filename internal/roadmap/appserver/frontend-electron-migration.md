# Electron Desktop Host Current 方案

> 状态：current implementation source
> 更新时间：2026-06-08
> 作用：固定 Lime Desktop 前端宿主由 Electron 全面接管后的 renderer、Desktop Host bridge、App Server sidecar、桌面壳能力和验收口径。
> 关联：[architecture.md](./architecture.md)、[frontend-integration-matrix.md](./frontend-integration-matrix.md)、[release-updater.md](./release-updater.md)、[consumer-integration.md](./consumer-integration.md)。

## 1. 结论

本版本的 Lime Desktop GUI 宿主已经由 Electron 全面接管。这里不是“后续切换”计划，而是 current 契约：

1. `npm run dev / build / preview` 默认进入 Electron。
2. `npm run verify:gui-smoke` 默认验证 Electron GUI。
3. 前端生产入口只允许经 Electron Desktop Host IPC 与 App Server JSON-RPC 进入后端事实源。
4. `lime-rs/` 是 Rust Runtime / App Server workspace，不是前端宿主事实源。
5. Codex CLI / `codex-rs` 只提供 App Server protocol、client、transport、daemon 和 runtime 分层参考；Lime 不参考 Codex App UI 或桌面壳实现。

当前主链固定为：

```text
Frontend
  -> safeInvoke / desktop-host API
  -> Electron Desktop Host bridge
  -> app_server_handle_json_lines
  -> App Server JSON-RPC
  -> RuntimeCore / ExecutionBackend
```

Electron 只负责 Desktop Host bridge、preload / IPC 白名单、窗口、托盘、Dock、菜单、updater、签名发布和 sidecar 生命周期；它不是第二套后端，也不是 Agent runtime adapter。

## 2. 事实源分类

| 分类             | 对象                                                                                             | 说明                                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `current`        | `electron/main.ts`、`electron/preload.ts`、`electron/hostCommands.ts`、`electron/ipcChannels.ts` | Desktop Host bridge、白名单 IPC、窗口与 sidecar 生命周期                                                               |
| `current`        | `src/lib/dev-bridge/safeInvoke.ts`、`http-client.ts`、`index.ts`、`app_server_handle_json_lines` | renderer 命令与 App Server JSON-RPC 传输收口，覆盖 Electron IPC / 本地调试 HTTP bridge、可用性探测、事件监听和错误追踪 |
| `current`        | `src/lib/desktop-host/*`                                                                         | renderer-safe 桌面能力 API，生产走 Electron bridge，测试夹具显式注入                                                   |
| `current`        | `src/lib/api/appServer.ts`、`packages/app-server-client/src/protocol.ts`                         | renderer 侧 App Server JSON-RPC gateway 与协议事实源                                                                   |
| `current`        | `electron/updateHost.ts`、`forge.config.mjs`、`.github/workflows/release.yml`                    | updater、签名、公证和发布包事实源                                                                                      |
| `dead / deleted` | `lime-rs/src/commands/**`                                                                        | 已物理删除；只允许负向回流守卫或不可变历史 evidence 引用，不得恢复 stub / compat wrapper                             |
| `compat`         | legacy desktop facade                                                                            | 只允许迁移期委托和投影，不继续承接新业务逻辑                                                                           |
| `deprecated`     | `lime-rs/` 下仍被 git 跟踪的旧宿主配置文件                                                       | 仅作为物理清理候选保留；不得作为 package scripts、Electron Forge、updater、CI 或版本同步输入                           |
| `dead`           | 旧 builder 配置 / CLI、自定义 Windows installer maker、旧 YAML / blockmap updater metadata       | 不得作为 release、updater、签名、公证、CI、i18n app metadata 或版本同步输入                                            |
| `dead`           | 生产 mock fallback                                                                               | 生产路径不能靠 mock 成功；mock 只允许测试夹具和契约守卫                                                                |

旧宿主配置的 current 封禁规则：

1. `package.json` scripts 不得重新引用旧宿主 CLI、旧宿主配置或旧 workspace 名。
2. `forge.config.mjs` 只允许消费 Electron Forge 配置、Electron desktop assets 与 App Server packaged resources。
3. `scripts/check-app-version-consistency.mjs` 只以 `lime-rs/Cargo.toml`、根 `package.json` 和 CLI npm package 版本为事实源。
4. 当前仍被 git 跟踪的旧宿主配置文件后续应单独删除或迁移；在删除前只能标记为 `deprecated cleanup candidate`，不能作为 current release、updater、签名、公证或开发入口证据。
5. 旧 builder 配置 / CLI、自定义 Windows installer maker 与旧 updater metadata 已进入 `dead`，不能作为 Electron current 打包、发布或 metadata 事实源。

`lime-rs/src/commands/**` 不再是 Electron current 后端实现目录。若 renderer 需要新增 Agent runtime、workspace、artifact、evidence、Knowledge、MCP 等后端能力，应进入 App Server JSON-RPC / RuntimeCore / services；若需要新增窗口、shell、托盘、Dock、updater、deep link 等桌面壳能力，应进入 Electron Desktop Host bridge。旧 wrapper 缺能力时必须 fail closed 并登记 current 缺口，不能在 `commands/` 里补 stub、compat wrapper 或平行实现，不得新增业务逻辑。

`src/lib/dev-bridge/**` 不等同于旧 Rust DevBridge，也不是整体删除对象。`safeInvoke`、HTTP client、`app_server_handle_json_lines`、事件监听和可用性探测继续属于 current renderer bridge；后续清理只收缩旧命令在 `commandPolicy.ts`、`mockPriorityCommands.ts`、desktop-host mock、旧 smoke 和 retired guard 里的 production truth / mock fallback。目录级退场必须先有新的 renderer bridge 事实源覆盖这些 current 能力，并在 `CCD-012` 和执行计划里写清退出条件。

## 3. Renderer 契约

业务组件、页面、hook 和 View Model 不直接碰桌面 IPC。推荐路径是：

```text
组件 / Hook
  -> src/lib/api/* 或 src/lib/desktop-host/*
  -> safeInvoke
  -> Electron Desktop Host bridge
  -> App Server JSON-RPC 或桌面壳能力
```

约束：

1. 前端业务代码不直接 import host 私有对象，不直接 spawn sidecar，不读写 sidecar stdout。
2. Agent runtime / session / turn / event / artifact / evidence 默认走 `src/lib/api/appServer.ts`。
3. 窗口、对话框、shell、快捷键、deep link、tray、Dock、updater 等桌面壳能力走 `src/lib/desktop-host/*` 或对应 API 网关。
4. 生产无 Electron bridge 时 fail-closed；开发 HTTP bridge 只服务本地调试和 smoke 证据，不作为发布包降级。
5. `invokeMockOnly`、explicit mock fallback、内存事件 / 窗口 / 对话框夹具只允许测试文件或显式测试夹具使用。

## 4. Electron Host 契约

Electron main / preload 的职责：

1. 暴露最小 `window.electronAPI`，不暴露完整 `ipcRenderer`。
2. 通过白名单 IPC 接收 renderer 请求。
3. 启动、握手、重启和关闭 App Server sidecar。
4. 把 App Server JSON-RPC lines 透传为 `app_server_handle_json_lines` / `app_server_drain_events`。
5. 将 sidecar notification fanout 到 renderer event router。
6. 管理窗口、菜单、托盘、Dock、deep link、global shortcut、dialog、shell 和 updater。
7. 打包时携带 `app-server.release.json` 与 `app-server/` packaged resource。

Electron main 禁止：

1. 复制 RuntimeCore 业务逻辑。
2. 自建第二套 session / thread / turn read model。
3. 在桌面壳层判断 Agent 执行完成。
4. 用 mock backend 证明生产 GUI 可交付。

## 5. App Server 契约

App Server 是后端事实源：

```text
Electron Desktop Host bridge
  -> App Server JSON-RPC
  -> RuntimeCore
  -> ExecutionBackend
  -> RuntimeBackend / future backend
```

固定约束：

1. in-process App Server 或 typed channel 只允许去掉进程边界，不允许引入第二响应合同。
2. JSON-RPC result envelope 与 `agentSession/event` notification 是 renderer / Electron / App Server 的共同合同。
3. `agentSession/turn/start`、`agentSession/turn/cancel`、`agentSession/read`、`agentSession/event` 是 Agent 主链 current 证据。
4. `agent_runtime_*` 只允许作为迁移期 compat facade，不得重新成为前端事实源。
5. `mock backend` 只能作为协议、client、packaging 或测试夹具，不是生产降级。

## 6. 桌面壳能力

| 能力               | current owner                     | 前端入口                                             |
| ------------------ | --------------------------------- | ---------------------------------------------------- |
| 对话框             | Electron `dialog`                 | `src/lib/desktop-host/plugin-dialog.ts`              |
| 外部打开 / 路径    | Electron `shell`                  | `src/lib/desktop-host/plugin-shell.ts`               |
| 全局快捷键         | Electron `globalShortcut`         | `src/lib/desktop-host/plugin-global-shortcut.ts`     |
| 深链               | Electron `app` protocol handlers  | `src/lib/desktop-host/plugin-deep-link.ts`           |
| 窗口               | Electron `BrowserWindow`          | `src/lib/desktop-host/window.ts`、`webviewWindow.ts` |
| 更新               | Electron 内置 `autoUpdater`       | `electron/updateHost.ts` 与更新 API 网关             |
| 托盘 / Dock / 菜单 | Electron main                     | Electron host commands / lifecycle                   |
| App Server sidecar | Electron main + app-server-client | `app_server_handle_json_lines`                       |

这些能力属于 Desktop Host bridge。除非是测试夹具，renderer 不能用浏览器 API 或 mock 状态伪造桌面壳结果。

## 7. 当前完成态

已完成：

1. 默认开发、预览、构建和 GUI smoke 入口进入 Electron。
2. Electron smoke 已证明 renderer 加载、Electron Desktop Host bridge 和 App Server `initialize` 可用。
3. Agent session create / list / read、thread read model、turn start / cancel、action respond 已经迁到 App Server JSON-RPC 前端 gateway。
4. `src/lib/desktop-host/*` 生产路径已按 Electron bridge 优先、无 bridge fail-closed、测试显式 mock 的规则收口。
5. release / updater / signing / notarization 已由 Electron Forge 和 Electron updater 文档锁定。
6. `lime-rs/src/commands/**` 已物理删除；新实现不得恢复该目录。

仍需继续推进：

1. 真实 GUI 发送后的 `agentSession/event -> read model refresh -> timeline` 业务 E2E。
2. artifact / evidence UI gateway 继续迁到 App Server。
3. 更宽的 legacy command facade 退场审计。
4. 低并发环境下持续验证 `electron:package:dir && electron:verify:package`。

## 8. 验收口径

最小 current 验收：

```bash
npm run test:contracts
npm run verify:gui-smoke
```

涉及 Electron host、updater、发布包、sidecar 或桌面壳能力时，追加：

```bash
npm run typecheck:electron
npm test -- "scripts/electron/current-docs-guard.test.mjs"
npm run electron:verify:package
```

涉及真实用户路径时，还必须跑业务 GUI E2E 或 Playwright 续测。只通过 lint、typecheck、Rust 单测或 mock 单测，不等于 Lime Desktop GUI 可交付。
