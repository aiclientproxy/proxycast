# Remote runtime 主链

## 这份文档回答什么

本文件定义 Lime 当前 `Remote / SDK / Server Mode` 的唯一 remote runtime 事实源，主要回答：

- 哪些远程入口才算当前产品主链
- `消息渠道`、`浏览器连接器 / ChromeBridge`、`DevBridge`、`telegram_remote` 分别属于哪一层
- 哪些路径负责真实 ingress / control plane，哪些只是安装壳、调试桥或兼容入口
- 后续新增 remote 能力应该往哪里收敛，而不是继续长平级旁路

它是 **远程入口与控制面的 current 文档**，不是单条渠道命令的局部注释。

## 什么时候先读

遇到以下任一情况时，先读本文件：

- 调整 `gateway_channel_*`、`gateway_tunnel_*`、渠道 probe 或多账号渠道运行时
- 调整浏览器连接器、ChromeBridge、远程调试入口或 browser backend policy
- 调整 `DevBridge` HTTP 桥、浏览器 dev shell 的后端接通逻辑
- 调整单通道 Telegram 远程触发入口，或评估是否应该继续保留它

如果一个需求同时碰到“远程触发 + 浏览器接入”“渠道入口 + 本地 Gateway”“调试桥 + 产品运行时”中的两项以上，默认属于本主链。

## 固定 remote 主链

后续 Lime 的 remote runtime 只允许向下面这条主链收敛：

`外部入口（消息渠道 / 浏览器连接器） -> 当前本地 control plane -> agent/browser runtime -> 现有 session/task/evidence 事实源`

这条主链的固定判断是：

1. `消息渠道 runtime` 是当前 IM 远程入口主链
2. `浏览器连接器 / ChromeBridge` 是当前浏览器侧远程 transport 主链
3. `DevBridge` 只作为 debug-only 开发桥，不再冒充产品 remote runtime
4. `telegram_remote_cmd` 是旧单通道入口，不再继续扩成长期主链

固定规则只有一句：

**后续新增 remote 能力时，只允许接到 `消息渠道 runtime` 或 `浏览器连接器 / ChromeBridge` 这两条 current ingress；不允许再造第三条并列 remote runtime。**

补充边界：`lime-rs/src/commands/**` 已物理删除，只允许作为负向回流守卫或不可变历史 evidence 出现。新增 remote control plane、agent/browser runtime 或跨 App remote 能力应进入 App Server / RuntimeCore / services；桌面壳、浏览器窗口、CDP 和系统打开能力进入 Electron Desktop Host。

## 代码入口地图

### 1. `消息渠道 runtime`

- `src/lib/api/channelsRuntime.ts`
- `gateway_channel_*` / `gateway_tunnel_*` 命令面（旧 `gateway_channel_cmd.rs` 只作为 cleanup reference）
- `lime_gateway::{telegram, feishu, discord, wechat}`

当前这里负责：

1. 多渠道 start / stop / status
2. 渠道账号 probe、登录、运行时模型绑定
3. tunnel / webhook 暴露与同步
4. 远程入站请求到本地 agent runtime 的 current 渠道入口

固定规则：

- 渠道远程入口统一走 `gateway_channel_*` 与 `gateway_tunnel_*`
- 前端当前主入口是 `channelsRuntime.ts`
- 不再把单独某个平台 bot runtime 重新拉回产品级总入口

### 2. `浏览器连接器 / ChromeBridge`

- `src/lib/webview-api.ts`
- 浏览器连接器命令面（旧 `browser_connector_cmd.rs` 只作为 cleanup reference）
- `lime-rs/src/services/browser_connector_service.rs`
- Webview / browser runtime 命令面（旧 `webview_cmd.rs`、`browser_runtime_cmd.rs` 只作为 cleanup reference）

当前这里负责：

1. 浏览器连接器安装、启停与权限配置
2. ChromeBridge 端点、连接状态与 connector session 断开
3. 外部 Chrome profile / CDP / managed browser backend 状态
4. Browser Assist 与远程浏览器接入的 current transport 事实

固定规则：

- 浏览器侧 remote transport 统一收口到 `webview-api.ts`
- 浏览器连接器设置与安装入口继续通过现有命令 surface 暴露；旧 `browser_connector_cmd.rs` 只作为 cleanup reference
- session / backend / remote debugging 状态继续由 Webview / browser runtime surface 暴露；旧 `webview_cmd.rs`、`browser_runtime_cmd.rs` 只作为 cleanup reference

### 3. 已删除的本地 Gateway 兼容壳

OpenClaw 安装、Gateway、Dashboard 与运行态管理模块已经判定为 `dead` 并从前端入口、legacy adapter 命令、DevBridge dispatcher、mock 与 core helper 中移除。

固定规则：

- 不再恢复 `src/components/openclaw/*`、`src/lib/api/openclaw.ts`、`lime-rs/src/commands/openclaw_cmd.rs` 或 `lime-rs/src/services/openclaw_service/*`
- 不再把本地 Gateway / Dashboard 壳当成 remote runtime 或一级系统入口
- 如需新增远程入口，只能回到 `消息渠道 runtime` 或 `浏览器连接器 / ChromeBridge` current 主链

### 4. `DevBridge`

- `lime-rs/src/dev_bridge.rs`
- `lime-rs/src/dev_bridge/*`
- `src/lib/dev-bridge/safeInvoke.ts`
- `src/lib/dev-bridge/http-client.ts`
- `src/lib/dev-bridge/commandPolicy.ts`

当前这里负责：

1. Rust `lime-rs/src/dev_bridge.rs` 与 `lime-rs/src/dev_bridge/dispatcher/**` 已物理删除，不得恢复旧 `3030` HTTP 桥或 Tauri command facade
2. 已删除路径只允许出现在负向回流守卫或不可变历史 evidence 中
3. 前端 `src/lib/dev-bridge/safeInvoke.ts` / `http-client.ts` 仍是 renderer 进入 Electron Desktop Host / App Server JSON-RPC 的 current 传输、可用性探测、事件监听和错误追踪边界
4. 前端 `src/lib/dev-bridge/commandPolicy.ts` 只承接旧命令 truth / no-mock fallback 的迁移期分类，不是 remote runtime 事实源

固定规则：

- `DevBridge` 是 debug-only 开发桥，不是产品 remote runtime
- Rust DevBridge 可以作为旧命令清理参考，但不能反向定义 current remote taxonomy
- 前端 `src/lib/dev-bridge/**` 不能按整目录退场；清理时只按命令组收缩 `commandPolicy`、mock fallback、旧 smoke 和 retired guard，保留 current renderer bridge
- 后续只允许继续做开发态适配、调试支撑和 current bridge 传输治理，不允许在 legacy dispatcher 上新增业务逻辑

### 5. `telegram_remote_cmd`

- `lime-rs/src/commands/telegram_remote_cmd.rs`

当前这里负责：

1. Telegram 单通道轮询
2. 将命令映射到 `agent.run / agent.wait / agent.stop / cron.* / sessions.*`

固定规则：

- 它当前没有前端主入口
- 多渠道 current 主链已经迁到 `gateway_channel_*` / `gateway_tunnel_*` 命令面；旧 `gateway_channel_cmd.rs` 只作为 cleanup reference
- 后续只允许迁移、收口或下线，不再继续扩功能

## current / compat / deprecated / dead

### `current`

- `src/lib/api/channelsRuntime.ts`
- `gateway_tunnel_*`
- `lime_gateway::{telegram, feishu, discord, wechat}`
- `src/lib/webview-api.ts`
- `lime-rs/src/services/browser_connector_service.rs`
- `src/lib/dev-bridge/safeInvoke.ts`
- `src/lib/dev-bridge/http-client.ts`
- `app_server_handle_json_lines`
- `internal/aiprompts/remote-runtime.md`

这些路径共同构成当前 remote 主链：

- IM 远程入口看 `gateway_channel_*`
- 远程浏览器 transport 看 `browser connector / ChromeBridge`
- 真实会话、任务与执行结果继续回到既有 agent/browser runtime 真相

### `compat`

- `lime-rs/src/commands/gateway_channel_cmd.rs`
- `lime-rs/src/commands/browser_connector_cmd.rs`
- `lime-rs/src/commands/webview_cmd.rs`
- `lime-rs/src/commands/browser_runtime_cmd.rs`
- `lime-rs/src/dev_bridge.rs`
- `lime-rs/src/dev_bridge/*`
- `src/lib/dev-bridge/commandPolicy.ts` 中的旧命令 truth / no-mock fallback

保留原因：

- 这些旧 command wrapper 仍可能承接迁移期命令面，但不能再拥有 remote runtime 业务事实。
- Rust DevBridge 仍是浏览器开发模式历史桥接层，后续只能按命令组退场。
- 前端 `commandPolicy.ts` 只作为迁移期 fail-closed 分类，不能继续承接 production truth 或 mock fallback。

退出条件：

- remote 能力迁到 App Server / services / Electron Desktop Host 后，撤旧 runner / dispatcher / catalog / mock 注册并删除对应 wrapper；删不动时登记 blocker。
- Rust DevBridge 继续只做开发态桥接参考，不再承担产品 remote 叙事。
- 已迁旧命令必须从前端 `commandPolicy.ts`、`mockPriorityCommands.ts`、desktop-host mock、旧 smoke 和 catalog 撤出；删不动且跨组长期存在的 residual 回挂 `CCD-012`。

### `deprecated`

- `lime-rs/src/commands/telegram_remote_cmd.rs`
- 任何继续把单渠道 bot runtime 直接定义为 remote 总入口的新实现
- 任何继续把 `DevBridge` HTTP 桥当成产品 remote runtime 真相的新实现

### `dead`

- `src/components/openclaw/*`
- `src/lib/api/openclaw.ts`
- `src/lib/api/openclawDashboardWindow.ts`
- `lime-rs/src/commands/openclaw_cmd.rs`
- `lime-rs/src/services/openclaw_service/*`
- `lime-rs/src/dev_bridge/dispatcher/openclaw.rs`
- `lime-rs/crates/core/src/openclaw_install.rs`

## 最低验证要求

如果本轮改动涉及本主链，至少按边界选择最贴近的验证：

- 纯文档 / 分类回写：`npm run harness:doc-freshness`
- 改渠道命令 / tunnel / webhook：`npm run test:contracts` 与相关渠道定向测试
- 改浏览器连接器 / ChromeBridge：浏览器相关定向测试或 `verify:gui-smoke`
- 改 `DevBridge`：至少补桥接定向测试
- 触碰已删除 OpenClaw surface：至少运行 `npm run governance:legacy-report` 与 `npm run test:contracts`，确认没有命令或入口回流

## 这一步如何服务主线

`M3` 的目标不是一次性重写所有 remote 功能，而是先把 remote 真相收成一条 current 主链。

从现在开始：

- 解释 IM 远程入口时，回到 `gateway_channel_*`
- 解释远程浏览器 transport 时，回到 `browser connector / ChromeBridge`
- 解释开发态浏览器接通时，回到 `DevBridge` compat
- 解释已删除本地 Gateway / Dashboard 壳时，回到 OpenClaw `dead` 分类与治理目录册
- 解释旧 Telegram 单通道触发时，视为 `deprecated`

这样后续 `M4 Memory / Compaction` 和 `M5 State / History / Telemetry` 就不必继续被 remote 入口语言打断。
