# Codex Desktop 跨平台底层对比与对齐计划

> status: `active / P1-windows-packaged-and-macos-gate-b`
> owner: Desktop Host + App Server + tool-runtime 各 current owner
> started: 2026-09-01
> upstream source: `/Users/coso/Documents/dev/rust/codex`
> installed reference: `/Applications/ChatGPT.app` (`com.openai.codex`, `26.825.41651`)
> Lime product chain: `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI`

## 1. 目的与范围

本计划用于重新、完整地对比 Codex Desktop 与 Lime 在 macOS、Windows 两端的底层实现。重点是平台能力和真实交付证据，不是 Composer 外观，也不把同名 API 数量当成完成度。

必须先区分三类事实源：

1. `/Users/coso/Documents/dev/rust/codex`：公开的 Rust runtime、App Server、sandbox、PTY、Code Mode、网络和持久化实现。
2. `/Applications/ChatGPT.app`：Codex Desktop 私有安装包，包含 Electron/Chromium、预编译 Rust 二进制以及未公开的 Swift/Objective-C 原生 framework、helper、`.node` 模块和插件。
3. Lime 仓库：Electron Desktop Host、App Server JSON-RPC、Rust crates、GUI、Forge/CI 和当前证据。

macOS 的 Swift 能力只能依据已安装的 Desktop 二进制和其系统链接关系记录，不能声称来自开源 Codex 仓库。Windows 的 restricted token、ACL、WFP、ConPTY、Job Object 等能力则同时有 Codex Rust 源码和 Lime Rust owner，可以做实现级对照。

本计划不恢复已退役 runtime、旧 v0/`agentSession/*` 生产主链、旧 Tauri command、生产 mock fallback，也不新建 Electron 第二套业务后端。

## 2. 证据分层

| 等级                | 当前可证明的内容                                                                                | 不能证明的内容                                   |
| ------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `static-bundle`     | ChatGPT.app 的 Mach-O、framework、Info.plist、entitlements、资源和 asar 字符串                  | 运行时权限是否已授予、真实窗口交互、真实用户流程 |
| `codex-rust`        | 开源 Codex crate、平台分支、测试和编译配置                                                      | 私有 Desktop Swift 行为、打包后资源是否正确      |
| `lime-local`        | Lime 当前源码、manifest、配置、单元/集成测试                                                    | 目标平台打包和系统权限                           |
| `gate-a`            | 浏览器/Renderer projection、DOM、GUI 交互                                                       | Electron main、preload、IPC、系统 API            |
| `gate-b`            | 真实 Electron、preload/IPC、`app_server_handle_json_lines`、App Server、runtime/read model、GUI | live provider、未显式启动的系统权限和另一个平台  |
| `platform-packaged` | 实际 macOS/Windows 产物、签名、安装、sidecar、系统 API                                          | 未运行的平台、未覆盖的权限路径                   |
| `live/eval`         | 明确 provider、账号、模型和配置下的行为                                                         | 其他 provider、地区、平台的普遍正确性            |

静态 `app.asar`、fixture、V8 cache 和源码编译通过都不能升级为 `platform-packaged` 或 `gate-b` 证据。每个能力都必须记录证据等级和未验证原因。

## 3. Codex Desktop macOS 原生层清单

### 3.1 私有 framework、helper 与插件

当前安装包已发现：

| 组件                                  | 证据位置                                                        | 已观察到的职责                                                                                                                                                    | Lime 当前分类                                                                 |
| ------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `Codex Framework.framework`           | `Contents/Frameworks/Codex Framework.framework`                 | Electron 宿主之外的 macOS 原生窗口、系统 API、Computer Use、PIP、显示和浏览器承载                                                                                 | 缺口；新增能力应归 Desktop Host current                                       |
| `Sparkle.framework`                   | `Contents/Frameworks/Sparkle.framework`                         | macOS 更新检查、下载、签名 feed 和安装                                                                                                                            | Lime `electron/updateHost.ts` + Forge updater 为 current；Sparkle bridge 缺失 |
| `sky.node`                            | `Contents/Resources/native/sky.node`                            | `NSWindow`/`CGWindow`、原生窗口句柄、窗口锚定/堆叠、控制 overlay、隐藏任务窗口、RemoteHostedPIPContent、浏览器 presentation、Computer Use 光标/位置、display link | 缺口；不能用 Renderer CSS 或普通 BrowserWindow 冒充                           |
| `input-monitoring-permission.node`    | `Contents/Resources/native/input-monitoring-permission.node`    | 查询 macOS Input Monitoring 权限                                                                                                                                  | 缺口；需要 Desktop Host 原生 helper/bridge                                    |
| `hid-topology-watcher.node`           | `Contents/Resources/native/hid-topology-watcher.node`           | `IOHIDDevice` 拓扑、Bluetooth/HID 设备变化、Codex Micro 接口发现                                                                                                  | 缺口；不得在 Renderer 伪造设备状态                                            |
| `bare-modifier-monitor`               | `Contents/Resources/native/bare-modifier-monitor`               | Swift 原生监听 modifier 单键按下、双击、释放；`--request-permission`                                                                                              | 缺口；需要 macOS helper 和权限生命周期                                        |
| `remote-control-device-key.node`      | `Contents/Resources/native/remote-control-device-key.node`      | Secure Enclave/硬件保护设备密钥的创建、删除、公钥和签名                                                                                                           | 缺口；密钥不可由 JS 或普通文件存储替代                                        |
| `browser-use-peer-authorization.node` | `Contents/Resources/native/browser-use-peer-authorization.node` | Browser-use native pipe 对端授权                                                                                                                                  | 缺口；必须与 Browser host 会话身份绑定                                        |
| `launch-services-helper`              | `Contents/Resources/native/launch-services-helper`              | 按路径启动应用、查询 bundle id/URL handler、下载完成通知、Dock 图标偏好/缓存和 Dock Tile 修复                                                                     | 缺口；应归 Desktop Host Launch Services owner                                 |
| `devicecheck.node`                    | `Contents/Resources/native/devicecheck.node`                    | macOS DeviceCheck/设备完整性相关桥                                                                                                                                | 未形成 Lime 产品需求；先记录为 excluded，禁止猜测实现                         |
| `sparkle.node`                        | `Contents/Resources/native/sparkle.node`                        | Node 到 Sparkle 的更新桥                                                                                                                                          | 缺口；是否采用由 updater owner 和签名策略裁决                                 |
| `CodexDockTilePlugin.docktileplugin`  | `Contents/PlugIns/CodexDockTilePlugin.docktileplugin`           | Dock Tile 插件                                                                                                                                                    | 缺口；需评估是否为 Lime 产品范围                                              |
| `codex_chronicle`                     | `Contents/Resources/codex_chronicle`                            | 独立屏幕/媒体/视觉管线；链接 ScreenCaptureKit、Metal、Vision、AVFoundation、CoreML、CoreAudio                                                                     | 缺口；不得把屏幕捕获塞进 App Server                                           |
| `codex-code-mode-host`                | `Contents/Resources/codex-code-mode-host`                       | 独立 Code Mode host                                                                                                                                               | Lime 已有 `tool-runtime` binary；需完成 packaged host 证据                    |
| `codex`                               | `Contents/Resources/codex`                                      | Rust runtime/App Server/执行链                                                                                                                                    | Lime Rust current owner 对齐目标                                              |

### 3.2 系统 Framework 与权限面

实现更新：Lime 不复制 Codex 私有 `.node` 模块，而是在自有 Swift helper 中提供等价的受控
`hidTopology.watch.*` 和 `bareModifierMonitor.*` JSONL 接口；两者仅代表能力调用面，不能替代
Codex Desktop 的私有 HID/Swift 实现，也不会把权限或硬件可用性伪造为 `ready`。

`Codex Framework.framework` 已链接 AppKit、Accessibility、ApplicationServices、Security、LocalAuthentication、ScreenCaptureKit、CoreGraphics、CoreText、Foundation、AVFoundation、AVFAudio、AudioUnit、CoreAudio、CoreLocation、CoreBluetooth、IOBluetooth、Vision、CoreML、DiskArbitration、ServiceManagement、UserNotifications、AuthenticationServices、Metal、MetalKit、Network、CFNetwork、`libsandbox` 等系统库。

主包 `Info.plist` 已确认的声明包括：

- Apple Events automation、AppleScript enabled；
- 摄像头、麦克风、音频采集；
- Desktop 文件夹访问；
- 日历、提醒事项、定位；
- `codex://`、`http`、`https` URL scheme；
- Folder、CSV、DOCX、PPTX、TSV、XLS、XLSM、XLSX、`.skill`、`public.data` 文件关联；
- `LSMinimumSystemVersion=13.0`。

主包 entitlements 已确认：

- `App Sandbox=false`；
- Apple Events automation；
- application groups `2DC432GLL2.com.openai.codex.notifications`、`2DC432GLL2.com.openai.sky.CUAService`；
- JIT、unsigned executable memory、camera/audio input、user-selected file read-write、network client；
- calendar、Keychain access groups、production push entitlement。

这说明“在 Lime 的 `Info.plist` 增加一个 usage description”不足以实现同等能力。权限声明、原生查询、授权跳转、恢复状态、签名 entitlement、helper 的 bundle identity 和 Gate B 证据必须成组交付。

## 4. Codex 开源 Rust 平台 owner

| 能力域               | Codex current owner                                                | macOS 重点                                                                | Windows 重点                                                                                              |
| -------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| App Server/transport | `codex-rs/app-server`, `app-server-daemon`, `app-server-transport` | stdio/sidecar、连接清理、托管安装/更新                                    | executable 命名、sidecar lifecycle、managed install/update                                                |
| 命令与进程           | `exec-server`, `app-server` process/command processors             | 进程组、退出/输出 drain、PTY                                              | ConPTY、Job Object、descendant cleanup、管道与终止                                                        |
| 沙箱                 | `sandboxing`                                                       | `/usr/bin/sandbox-exec`、seatbelt policy、网络/文件 profile、symlink 保护 | `windows-sandbox-rs` restricted token、ACL/DACL、deny-read、reparse point、workspace SID、private desktop |
| Windows setup/runner | `windows-sandbox-rs` 与相关 setup/runner binary                    | 不适用                                                                    | sandbox users、DPAPI、Firewall/WFP、elevated helper、framed IPC、runner materialization                   |
| PTY                  | `utils/pty`                                                        | Unix PTY、process group、resize、signal                                   | ConPTY、Job Object、终止和 terminal lifecycle                                                             |
| 进程加固             | `process-hardening`                                                | `ptrace(PT_DENY_ATTACH)`、`RLIMIT_CORE=0`、清除 `DYLD_*`                  | 平台等价的环境/子进程约束由 Windows execution owner 承接                                                  |
| 网络                 | `network-proxy`                                                    | Security framework native cert trust、MITM/proxy policy                   | Schannel native cert trust、Windows network isolation、WFP ingress                                        |
| 凭证/密钥            | `keyring-store`                                                    | keyring crate -> macOS Keychain                                           | keyring crate -> Windows Credential Store；sandbox identity 另由 Windows owner 管理                       |
| Code Mode            | `code-mode`, `code-mode-host`, `code-mode-runtime`                 | V8 sandbox-enabled Darwin asset、gRPC/stdio/session                       | Windows GNU sandbox-enabled；MSVC 使用不同预构建资产，需单独验证                                          |
| 屏幕捕获             | `app-server/src/screen/macos.rs` 等                                | ScreenCaptureKit 相关 Rust 能力                                           | 不把 macOS API 迁移到 Windows                                                                             |
| 状态/恢复            | `state`, `thread-store`, App Server                                | Thread/Turn/Item、cold resume、日志/投影一致性                            | 同一协议与持久化语义，外加 sidecar/runner 状态恢复                                                        |

Codex 开源仓库没有 Swift 源文件。Swift/Objective-C 原生层属于安装包私有实现，必须以 Desktop Host capability contract 形式接入 Lime，而不是把二进制逆向结果写成 Rust API。

## 5. Lime 当前 owner 盘点

### 5.1 已有 `current`

- App Server JSON-RPC、RuntimeCore、Thread/Turn/Item projection 和持久化主链；
- `lime-rs/crates/tool-runtime/src/execution_process/` 的 Windows ConPTY、Job Object、ACL、private desktop、runner supervisor、协议和审计；
- `lime-rs/crates/tool-runtime/src/windows_setup/` 的 sandbox account/read access/setup；
- `app-server-daemon` 与 `app-server-client` 的 sidecar manifest、生命周期和进程管理；
- `forge.config.mjs` 的 macOS DMG/ZIP、Windows Squirrel、extra resources、签名/公证配置；
- `scripts/lib/electron-desktop-resources.mjs` 的跨平台资源清单、哈希和架构校验；
- `electron/native/macos/macos-native-host.swift`、`electron/macosNativeHost.ts` 的签名资源与 JSONL 系统宿主；
- `electron/updateHost.ts`、Windows Squirrel startup、资源 digest verifier；
- `tool-runtime` 的 `code-mode-host`、`windows-sandbox-setup`、`windows-sandbox-runner` binary；
- Electron embedded browser、权限/媒体处理、托盘/通知/更新窗口；
- Rust sandbox backend 规划、macOS seatbelt 检测和统一 approval/sandbox orchestrator。

### 5.2 明显缺口或证据未闭环

- Swift 原生 helper 现已支持 IOHID topology 一次性读取与 `hidTopology.changed` 变化事件，以及 `bareModifierMonitor.start/stop`；权限、硬件和签名 Gate B 仍未闭环。
- Swift 原生 helper 已支持 Accessibility/Input Monitoring、Apple Events 授权查询/consent request、Launch Services、security-scoped bookmark、CGWindow/NSScreen、IOHID、LocalAuthentication、Secure Enclave device-key 和 Application Group 查询；窗口编排已用 AX 应用级隐藏实现可恢复 lease 并有 packaged fixture 证据。Dock Tile、PIP、ScreenCaptureKit 等更高阶闭环以及权限/硬件/签名 Gate B 仍未完成；
- 无 Dock Tile plugin、Sky 风格原生 PIP/overlay 和 display link host；窗口锚定、堆叠与 hide-for-task lease 已有原生实现和本地 packaged fixture 证据；
- 无 Chronicle/ScreenCaptureKit 媒体管线的原生 helper；当前 helper 已支持受 Screen Recording 权限保护的全屏/显示器/窗口 PNG 快照，但媒体保留策略、签名包和 Gate B 仍缺；
- `lime-rs/Info.plist` 现有 `http`/`https`/`lime`、Apple Events、摄像头和麦克风描述；
- `lime-rs/entitlements.plist` 现有 JIT、unsigned memory、library validation、camera/audio-input、Apple Events automation、selected file、network client/server；
- macOS security-scoped bookmark 已由 Desktop Host current owner 管理 helper 生命周期、稳定 ID 持久化、冷启动 resolve/start、活动 token stop 和 revoke；packaged Gate B 已覆盖临时目录 create/resolve/start/stop，真实用户授权授予/撤销恢复仍待补证。
- 原生窗口/显示/HID/Screen Recording snapshot/LocalAuthentication/设备密钥已进入同一 Swift helper 的结构化调用面；窗口控制覆盖 AX frame、raise 和所属应用 hide/unhide，状态合同保持 `unverified`，不把 API 可调用误报为授权或 Secure Enclave readiness。
- LocalAuthentication/Keychain 的 Electron Host 专用桥未形成 owner；
- Windows 源码安全矩阵已有真实 CI 证据，UI Automation/Raw Input helper 已进入资源组，但 packaged sidecar、Squirrel 安装后 runner/runtime 与 Electron Gate B 仍需独立证据；
- Code Mode/V8 的 Darwin、Windows GNU、Windows MSVC 资产配对和安装包资源校验尚未形成跨平台矩阵；
- browser-use native pipe peer authorization、remote-control device key、Computer Use 与模型能力/readiness 的端到端 contract 未闭环。

## 6. 跨平台能力对比矩阵

状态含义：`已对齐` 表示 current owner 和最低证据存在；`部分` 表示源码有实现但平台/打包/消费者证据未闭环；`缺口` 表示没有可用 current owner；`excluded` 表示产品裁决后不复制 Codex 专属能力。

矩阵更新说明：Lime macOS 的 HID 行现已覆盖 `hidTopology.read` 与 `hidTopology.watch.start/stop` 的
`hidTopology.changed` unsolicited event；bare modifier 行现已覆盖
`bareModifierMonitor.start/stop`。文件访问/bookmark 行现已覆盖 Desktop Host 的稳定 ID 持久化、冷启动
resolve/start、活动 token stop 和 revoke，以及 packaged Gate B 的真实
`bookmark.create/resolve/start/stop`；屏幕捕获行现已覆盖 Screen Recording 权限 query/request 与
`screenCapture.snapshot` PNG 快照，但未实现 Chronicle 媒体管线。窗口编排和 bookmark 生命周期已有本地 packaged helper 证据；上述能力仍保持 `部分`，因为真实签名包、
权限授予/撤销、媒体保留策略和硬件条件的 Gate B 尚未完成。

| 能力域                     | Codex Desktop macOS                                                       | Codex Desktop Windows                                                         | Lime macOS                                                                                                                               | Lime Windows                                                        | 分类/下一步                                                |
| -------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------- |
| Electron/Chromium 宿主     | Electron + Codex Framework 原生层                                         | Electron + Windows native resources                                           | Electron current + Swift native helper；高阶原生层仍缺                                                                                   | Electron current；Squirrel/sidecar 部分已有                         | `current`；建立统一 capability/readiness contract          |
| 原生窗口/句柄/overlay      | `sky.node`、NSWindow/CGWindow、anchor、stack、hide-for-task               | 私有实现未从安装包完全解出                                                    | CGWindow 查询、应用激活、Accessibility AX frame/raise、窗口所属应用 hide/unhide、anchor/stack/hide-for-task lease 已实现；overlay 未闭环 | `windows.window.read` 枚举 HWND、进程、标题、类名和边界；不控制窗口 | `部分`；P2 native window owner，先不在 Renderer 模拟       |
| Remote Hosted PIP          | `RemoteHostedPIPContent`、presentation lifecycle                          | 需从 Windows 包和运行时补证据                                                 | 缺                                                                                                                                       | 缺                                                                  | `缺口`；P2，必须有系统级 Gate B                            |
| Input Monitoring           | 原生 `.node` 查询和 System Settings 跳转                                  | Windows Raw Input 注册和会话权限需另证                                        | Swift helper query/request/settings 已实现                                                                                               | `windows-native-host` 只读 Raw Input modifier watcher               | `部分`；P1 permission contract                             |
| Accessibility/Apple Events | Accessibility、ApplicationServices、AppleScript、Apple Events entitlement | Windows UI Automation/相关能力需另证                                          | Swift helper Accessibility + `appleEvents.targets/read/request`，按目标 bundle 查询授权并跳转 Automation 设置                            | `windows-native-host` 有界 UI Automation read-only tree             | `部分`；P1 fail-closed readiness                           |
| HID/Bluetooth              | `hid-topology-watcher.node`、IOHIDDevice                                  | Windows HID/Raw Input 需另证                                                  | IOHID topology 查询、watcher 启停和 `hidTopology.changed` 已实现；硬件 Gate B 未闭环                                                     | Raw Input 只覆盖修饰键，不提供 HID topology                         | `部分`；P2，产品需求确认后实现                             |
| Bare modifier monitor      | Swift helper，支持 request permission                                     | Windows hotkey/Raw Input 需另证                                               | Swift helper `bareModifierMonitor.start/stop` 和 down/up 事件已实现；权限/硬件 Gate B 未闭环                                             | `windows.bareModifierMonitor.start/stop` 只读 down/up 事件          | `部分`；P1，Windows Gate B 待补                            |
| Secure Enclave/device key  | `remote-control-device-key.node`                                          | 硬件密钥/DPAPI 需另证                                                         | Swift helper Secure Enclave create/read/sign/delete 已实现；硬件/签名 Gate B 未闭环                                                      | DPAPI 用于 sandbox setup，非同一 device-key owner                   | `部分`；P2 remote-control owner                            |
| Browser pipe authorization | `browser-use-peer-authorization.node`                                     | native pipe/ACL 需另证                                                        | 缺                                                                                                                                       | 普通 browser host，未有同等级 peer auth                             | `缺口`；P2，绑定 connection/thread identity                |
| 文件关联/Launch Services   | helper 查询 bundle/URL handler、按路径启动、下载完成通知                  | Shell association/URL handler 需另证                                          | URL scheme + Swift helper path/bundle/URL handler 已实现                                                                                 | Squirrel startup 有；无统一 association helper                      | `部分`；P1 Desktop Host owner                              |
| 文件访问/bookmark          | user-selected read-write、Desktop 权限、security-scoped 生命周期          | ACL/path normalization/reparse point                                          | 普通选择文件 + Desktop Host security-scoped bookmark 稳定 ID、冷启动恢复、revoke；真实授权 Gate B 未闭环                                 | ACL/path normalization current                                      | `部分`；P1 macOS bookmark，Windows 保持 tool-runtime owner |
| 进程/沙箱                  | seatbelt、sandbox-exec、process hardening                                 | restricted token、ACL、deny-read、private desktop、Firewall/WFP、setup/runner | Rust seatbelt 检测/规划，需真实 macOS 执行证据                                                                                           | Rust backend + Windows CI 7/7，packaged 证据待补                    | `current/部分`；P0/P1 platform evidence                    |
| PTY/终端                   | Unix PTY/process group                                                    | ConPTY + Job Object                                                           | Rust PTY owner，需 macOS Gate B                                                                                                          | Rust ConPTY/Job Object 有 current                                   | `current/部分`；补跨平台 packaged matrix                   |
| 网络隔离/证书              | Security framework trust、proxy policy                                    | Schannel、WFP/network isolation                                               | network client entitlement；native cert/isolated execution 未闭环                                                                        | WFP/Firewall/readiness 已有 current                                 | `部分/current`；P1 packaged/readiness                      |
| 摄像头/麦克风/音频         | AVFoundation/AVFAudio/AudioUnit/CoreAudio、权限声明                       | Windows media API 需另证                                                      | Swift helper `mediaPermissions.read/request` 查询/请求摄像头和麦克风；主包与 helper 均有 usage description                               | Electron media handling                                             | `部分`；P2 native media/permission contract                |
| 屏幕捕获/视觉              | ScreenCaptureKit、`codex_chronicle`、Vision/CoreML/Metal                  | Windows capture/vision 需另证                                                 | Swift helper 提供 Screen Recording 权限和 CGWindow/CGDisplay PNG 快照；Chronicle 管线未实现                                              | 无等价 Chronicle                                                    | `部分`；P2，补媒体管线和系统证据                           |
| Computer Use 光标/窗口     | `sky.node` + Accessibility/CGWindow                                       | Windows desktop/UI automation 需另证                                          | `accessibilityTree.read` 提供有界只读控件树；无 cursor/window 控制注入                                                                   | UI Automation、HWND 和显示器只读观察；无 cursor/window 控制注入     | `部分`；P2，控制注入仍需安全边界                           |
| Code Mode/V8               | 独立 host；Darwin sandbox-enabled V8                                      | Windows GNU sandbox-enabled，MSVC 资产不同                                    | `code-mode-host` Rust binary，Darwin 资产配对待证                                                                                        | host/setup/runner binary，有 MSVC 资源风险                          | `部分`；P0 resource manifest + P1 matrix                   |
| MCP/Skills/Apps            | Desktop UI + App Server/Rust current chain                                | 同一协议，系统权限由 host 承接                                                | App Server/GUI current                                                                                                                   | App Server/GUI current                                              | `current`；不为平台复制第二套 catalog                      |
| Thread/Turn/Item 恢复      | App Server/Rust current                                                   | App Server/Rust current + sidecar 状态                                        | current                                                                                                                                  | current；需 packaged resume evidence                                | `current/部分`；Gate B/packaged                            |
| Sidecar/managed install    | `codex`、Code Mode host、Sparkle/托管生命周期                             | managed install、runner/setup、Windows executable naming                      | `app-server-daemon` current                                                                                                              | current；安装后资源组待证                                           | `部分`；P0 资源和生命周期闭环                              |
| 更新/签名/安装             | Sparkle、Developer ID、notarization、stapling、Sparkle feed               | Squirrel/MSIX/signing 分支                                                    | Forge DMG/ZIP、签名/公证配置                                                                                                             | Forge Squirrel current；MSIX 非主路径                               | `部分/current`；P1 真实产物证据                            |
| Dock/通知/登录项           | Dock Tile、UserNotifications、ServiceManagement                           | Toast/startup/taskbar 需另证                                                  | tray/notification 有；Dock Tile 缺                                                                                                       | startup/tray 有                                                     | `部分`；按产品范围实现                                     |
| 诊断/遥测/设备完整性       | DeviceCheck、Chronicle 等私有能力                                         | Windows 诊断能力需另证                                                        | 未形成专属 owner                                                                                                                         | 未形成专属 owner                                                    | `excluded/缺口`；没有明确需求不得猜测                      |

## 7. 当前/兼容/废弃/已删除裁决

### `current`

- Electron Desktop Host 只承接窗口、文件选择、系统权限、外链、托盘、自动更新、sidecar 生命周期和 `app_server_handle_json_lines` 转发。
- App Server JSON-RPC + RuntimeCore + `model-provider` + `tool-runtime` 是业务唯一主链。
- Windows restricted execution 继续由 `tool-runtime` 的 `execution_process`、`windows_setup`、runner/setup binary 承接。
- macOS seatbelt、Unix PTY、process hardening、native cert/keyring 继续由对应 Rust crate 承接；新增 macOS 原生能力必须通过 Desktop Host capability contract 暴露。
- Forge `forge.config.mjs`、`electron/updateHost.ts`、Squirrel 和现有 release/resource verifier 是打包/更新事实源。

### `compat`

- 仅允许对外部平台协议或历史文件位置做边界委托；不得在 compat 层加入新的权限、沙箱、窗口或 runtime 业务。
- `.lime/AGENTS.md` 读取可继续作为受控旧路径委托，不能扩散为新的配置事实源。

### `deprecated`

- 旧 `thread/rollback` 仅保留迁出/负向守卫，current 为 `thread/revert`。
- 旧 `agentSession/event` 仅可作为诊断旁路，不得承接 GUI 生命周期事实。
- 任何旧 updater metadata、旧 installer builder、自定义 Windows installer maker 若仍被发现，只能迁出并删除。

### `dead / deleted / forbidden-to-restore`

- 旧 Tauri command、`protocol/v0`、生产 `agentSession/*` method、`project_shell_*`、旧 `fileSystem/*`、旧 `executionProcess/*`、旧 provider crate 和已删除 runtime 均不得恢复。
- 不得把 `sky`、Chronicle、Input Monitoring 或 remote-control 原生能力做成 Renderer mock、App Server mock 或新的 legacy facade。
- 没有产品需求和唯一 owner 的 DeviceCheck、realtime existing-call、Codex 专属 MDM/Computer Use config 先保持 `product-scope-excluded`，不建立空壳 API。

## 8. 分阶段执行顺序

### P0：能力合同、资源组和证据基线

owner：Desktop Host + App Server + Forge/CI。

- 建立跨平台 `desktop capability/readiness` current contract，至少覆盖：平台、架构、原生模块、权限状态、沙箱 backend、Code Mode host、sidecar、签名/资源完整性、更新能力。
- 为每个平台建立 resource manifest：`app-server`、`code-mode-host`、Windows setup/runner、macOS helper/framework、digest、架构和最小系统版本；缺失或 digest 漂移必须 fail closed。
- 将 macOS 原生层标记为 private Desktop resource，不把 Swift 二进制伪装成开源 Rust source evidence。
- 完成 Windows packaged sidecar/Squirrel 安装后启动、runner/setup/resource verifier 和 Electron Gate B；保留现有 Windows 7/7 restricted matrix 作为源码安全证据，不与 packaged 证据混淆。
- 为 Darwin、Windows GNU、Windows MSVC 分别验证 V8 archive/binding 配对；MSVC 不能沿用 GNU sandbox 资产的假设。

退出条件：

- contract/schema、App Server consumer、Electron/preload、manifest、CI fixture 和文档一致；
- macOS/Windows 至少各有一条真实 packaged Gate B；
- 资源缺失、digest 漂移、平台/架构不匹配均有负向测试；
- `npm run test:contracts`、相关 Rust 测试、`npm run governance:legacy-report`、`npm run governance:scripts` 通过。

### P1：macOS 权限、文件访问和 Launch Services

owner：新增 Desktop Host macOS native owner；Rust runtime 只消费 capability/readiness。

- 先实现最小原生 helper：Input Monitoring/Screen Recording query/open settings、Accessibility/Apple Events readiness、security-scoped bookmark start/stop/revoke、Launch Services path/bundle/URL handler、CGWindow/NSScreen、IOHID 查询与拓扑 watcher、bare modifier、LocalAuthentication 与 Secure Enclave device-key 查询/操作。
- Windows helper 以 UI Automation COM 只读树观察和 Raw Input modifier watcher 作为当前 owner；同样要求资源 digest、签名、生命周期和权限拒绝语义，不实现控制注入。
- 明确 helper 的 bundle identity、签名、entitlement、应用组、IPC 协议、超时和权限拒绝语义；权限不可用时返回结构化 `NotGranted/Unavailable/UpdateRequired`，不得默认放行。
- 将文件选择结果和 bookmark 生命周期落到受管用户数据目录/平台 API；不把绝对路径或授权状态写入 Renderer 私有缓存事实源。
- `SystemUtilityHost` 以稳定 ID 将 bookmark 编码数据写入 `appDataRoot/macos/security-scoped-bookmarks`，冷启动按 ID 恢复，`bookmark.revoke` 删除受管记录；ID 仅允许安全文件名字符并拒绝路径穿越。
- 完成真实 macOS Electron Gate B：权限已授予、未授予、撤销后恢复、外部应用启动、URL/file association、冷启动恢复。

退出条件：真实签名包在 macOS 13+ 完成上述场景；Gate B trace 同时包含 Electron IPC、native helper、App Server/runtime identity；无 mock/fallback。

### P1：Windows packaged 与安装更新闭环

owner：`tool-runtime` Windows owner + `app-server-daemon` + Forge/CI。

- 将 `app-server.exe`、`code-mode-host.exe`、`windows-sandbox-setup.exe`、`windows-sandbox-runner.exe` 成组打包。
- 验证 Squirrel 安装、升级、卸载、快捷方式、安装目录 ACL、启动参数、sidecar PID/IPC 和崩溃后清理。
- 验证真实 packaged app 运行 `app_server_handle_json_lines`、restricted runner、ConPTY、Job Object、WFP/readiness；源码 7/7 不能替代这一门禁。
- release workflow 在 Squirrel 安装 smoke 后继续运行已安装候选的 Code Mode Gate B，并上传独立证据；资源 manifest verifier 作为同一候选产物门禁，不允许仅依赖 Forge hook。
- Windows MSIX 只在产品决定采用时接入；不得让 Squirrel 和 MSIX 同时成为两个 current updater owner。

退出条件：Windows packaged Gate B、安装升级 smoke、资源 digest verifier、restricted execution evidence 同一候选产物通过。

### P2：原生窗口、PIP、Computer Use、输入设备和屏幕媒体

owner：Desktop Host macOS native owner，Windows 等价能力另行裁决；App Server 只承接权限/能力结果和业务意图。

- 按产品需求逐项实现 `NSWindow/CGWindow` handle、window anchor/stack、hide-for-task、PIP presentation、control overlay、display link。
- 设计 Computer Use 安全边界：cursor/window 操作必须绑定当前 Thread/Turn、用户授权、平台 readiness 和可审计 identity。
- 将 HID topology、bare modifier、Bluetooth/Raw Input 监听纳入生产前，仍需确认硬件/快捷键产品需求和隐私提示；当前 helper 接口只提供受权限保护的实验性事件流，未获产品裁决时保持不可用。
- 评估 Chronicle/ScreenCaptureKit/AVFoundation/Vision/Metal 管线是否纳入 Lime；若纳入，独立 helper、权限、生命周期、媒体数据保留和打包证据必须单独建 owner。
- remote-control device key 使用 Secure Enclave/Windows 硬件或受管密钥存储；禁止 JS 文件密钥。

退出条件：每个能力有平台 API、权限、失败语义、签名资源、Gate B 和隐私/数据保留说明；未满足时保持 fail closed 或 excluded。

## 9. P0 首个纵向切片（2026-09-01）

已实现：

- `electron/platformCapabilities.ts`：统一生成跨平台 `desktopCapabilities`，包含平台、架构、包状态、Accessibility、Input Monitoring、Apple Events、Application Group、原生模块、窗口编排、Accessibility tree、display watcher、媒体权限和 sandbox readiness。
- macOS Accessibility 使用 `systemPreferences.isTrustedAccessibilityClient(false)` 真实查询；查询异常返回 `unavailable`。
- `get_environment_preview` 通过 `SystemUtilityHost` 返回该合同，现有环境设置 current 链可读取，不新增第二个业务 IPC 或 App Server 方法。
- Application Group 当前明确为 `not_configured` 且 identifiers 为空；测试守卫确认结果不会出现 `com.openai.*` 或 OpenAI Team ID。
- Windows sandbox 状态明确为 `unverified`，由 `tool-runtime`/packaged verifier 继续负责，不从 Electron 平台字段推断 ready。

`scripts/lib/electron-desktop-resources.mjs` 为 Darwin、Windows 生成 `desktop-resources.manifest.json`，记录 platform/arch/version、sidecar/helper 路径、sha256、最小 macOS 版本和 entitlement/application-group 元数据；资源缺失、哈希漂移、路径越界和架构不匹配均 fail closed。

Forge `afterCopyExtraResources` 在签名之前编译并写入 macOS `macos-native-host`；Windows 将 app-server、Code Mode、sandbox setup/runner 作为同一资源组登记。`electron/macosNativeHost.ts` 通过受管 JSONL 子进程协议接入 `macos_native_host_invoke`，只接受资源清单中哈希匹配，或在 macOS Forge 重签后通过严格 `codesign` 校验且身份/架构一致的 helper；`before-quit` 会终止 helper 并清理未完成请求。`SystemUtilityHost` 负责 bookmark 稳定 ID 持久化、冷启动恢复、活动 token stop 与 revoke。

Swift helper 当前不申请或声明 OpenAI Application Group；`applicationGroup.read` 仅查询 Lime 自有签名容器，默认返回 `not_configured`。当前 helper 还提供 `window.read/focus/raise/setFrame/setOwnerVisibility`、`window.anchor`、`window.stack`、`window.hideForTask.start/stop/read`、`display.read`、`hidTopology.read`、`hidTopology.watch.*`、`bareModifierMonitor.*`、`screenCapture.read/request/snapshot`、`appleEvents.targets/read/request/openSettings`、`localAuthentication.*` 和 `deviceKey.*`，但这些能力必须通过签名 entitlement、目标应用授权、硬件和平台 Gate B 才能升级状态。Apple Events 接口只查询目标 bundle 的自动化授权或触发系统 consent，不发送实际控制事件。hide-for-task 使用 helper 进程内 task lease 保存每个 owner 的原始隐藏状态，退出时自动恢复；anchor/stack 仅操作真实 CGWindow/Accessibility，不宣称私有 overlay 或 PIP。HID/bare modifier 事件经 `SystemUtilityHost` 转发到 Electron `evt:*` 广播，未连接 renderer 时不会创建第二事实源。本切片的 `lime-local` 证据已覆盖资源清单、Swift 编译、宿主分发、bookmark 持久化、事件通道和负向校验；真实签名包、权限授予/撤销、屏幕捕获媒体保留策略、Windows packaged Gate B 仍未完成，不能升级为 `platform-packaged`。

本轮验证（2026-09-01）：macOS helper 多文件 Swift 编译与 JSONL 能力/事件启停通过，`screenCapture.snapshot` 已在当前桌面生成全屏和显示器 PNG 快照，`window.raise` 与幂等 `window.setOwnerVisibility` smoke 通过；最终 Darwin arm64 Forge 产物已完成，主 app、app-server、code-mode-host 和 native helper 的 `codesign --verify` 通过，资源清单 verifier 与 packaged helper JSONL smoke 通过。macOS/Electron、bookmark、capability、资源 verifier、entrypoint、Windows Squirrel/Code Mode/release guard 共 12 个定向测试文件、`174/174` 通过；Electron `tsconfig`、命令契约、脚本治理、文档边界和 `git diff --check` 通过。仓库根 TypeScript 全量检查仍受并行分支既有 fixture/type 错误阻塞，未将其归因于本切片。

本轮增量（2026-09-01）：新增 `macos-window-orchestration.swift` 并纳入 Forge 编译与入口守卫；`window.anchor` 按真实 CGWindow bounds 计算 top/bottom/left/right + start/center/end 位置，`window.stack` 以明确的 front-to-back 顺序执行 Accessibility raise，`window.hideForTask.*` 以 task ID lease 保存并恢复 owner 原始可见性，参数、数量和路径均 fail closed。`desktopCapabilities.capabilities.windowOrchestration` 与 `appConfigTypes` 已同步，状态只报告 `unverified/not_configured/unsupported`。本机 `swiftc` 对 `arm64-apple-macos13.0` 与 `x86_64-apple-macos13.0` 均编译通过；Electron/资源/Windows Squirrel 定向 Vitest `62/62`、Electron TypeScript、`test:contracts`、`verify:gui-smoke` 和 `git diff --check` 通过。GUI smoke 证明真实 Electron/preload/App Server 构建与启动链未回归；待下一次 packaged smoke 在真实授权窗口上验证 anchor/stack/hide-for-task，未将本地编译或普通 GUI smoke 升级为窗口能力 Gate B。

本轮增量（2026-09-01）：新增 `macos-accessibility-tree.swift` 并纳入 Forge 编译与入口守卫；`accessibilityTree.read` 绑定真实窗口 owner 和 Accessibility element，限制最大深度 32、节点 10000、文本 512 字符，返回稳定 path、role/title/value/identifier、enabled/focused、frame 和 children；未授权、窗口消失、超限均 fail closed。`desktopCapabilities.capabilities.accessibilityTree` 与 `appConfigTypes` 已同步。该接口只覆盖 Computer Use observation，不实现鼠标/键盘注入，不提升为完整 Computer Use；真实应用 tree 的 packaged Gate B 仍待验证。

本轮增量（2026-09-01）：新增 `macos-display-watcher.swift` 并纳入 Forge 编译与入口守卫；`display.watch.start/stop` 使用 CoreGraphics display reconfiguration callback，事件 payload 同时包含 display ID、原始 flags 和最新 `display.read` 列表，重复 start/退出注销保持幂等。`desktopCapabilities.capabilities.displayWatcher` 与 `appConfigTypes` 已同步；显示变化事件仍属于只读 observation，未宣称 Codex display link 或媒体管线已完成。

本轮增量（2026-09-01）：新增 `macos-media-permissions.swift` 并纳入 Forge 编译与入口守卫；`mediaPermissions.read/request` 分别查询或请求摄像头、麦克风 TCC 状态，helper 与主包补齐 usage description 和 camera/audio-input entitlement，Swift 编译参数显式链接 `AVFoundation`；主窗口媒体权限 handler 同时支持受控音频、视频和音视频组合请求。`desktopCapabilities.capabilities.mediaPermissions.microphone/camera` 与 `appConfigTypes` 已同步；未完成真实签名包和权限授予/撤销 Gate B 时继续报告 `unverified`。

本轮验证（2026-09-01）：Darwin `arm64`/`x86_64` helper 编译通过；本机真实 helper `mediaPermissions.read` 返回麦克风 `ready`、摄像头 `not_granted`，未调用 request、不触发系统授权弹窗；重新生成的 `dist-electron` 开发资源中 helper 具备 `AVFoundation` 链接、Lime bundle ID 和两项 usage description。媒体/能力/资源/主窗口权限 Vitest `10/10`、Electron 与 renderer/node TypeScript、`test:contracts`、`verify:gui-smoke` 和 `git diff --check` 通过。开发资源 manifest 的 `signedByForge=false`，因此仍属于 `lime-local/live`，不能升级为签名发布 Gate B。

本轮增量（2026-09-01）：新增 `macos-apple-events.swift` 并纳入 Forge 编译与入口守卫；`appleEvents.targets` 列出当前运行且有 bundle identifier 的应用，`appleEvents.read/request` 要求目标 bundle identifier，先确认目标应用正在运行，再通过 `AEDeterminePermissionToAutomateTarget` 查询或显式触发系统 consent，`appleEvents.openSettings` 负责跳转 Automation 隐私设置，返回 `ready/not_granted/unavailable`、系统状态码、`requiresUserConsent` 和设置 URL。该接口不发送实际 Apple Event，不作为 Computer Use 控制注入通道；`desktopCapabilities.appleEvents` 与 `appConfigTypes` 已同步，Windows/非 Desktop 平台保持 `unsupported`。

本轮验证（2026-09-01）：Darwin arm64/x86_64 helper（含 Apple Events 源文件）编译通过；本机对 Finder 执行 `appleEvents.read` 返回 `ready`，对未运行 bundle 返回 `unavailable`/`statusCode=-600`，未调用 `appleEvents.request`，因此未触发授权弹窗。资源编译参数、helper `Info.plist` usage description、automation entitlement manifest/verifier、entrypoint guard、macOS native host JSONL forwarding、capability contract 定向测试覆盖新增方法；相关 Vitest `47/47`、Electron/renderer/node TypeScript、`test:contracts`、`verify:gui-smoke`、`docs:boundary` 和 `git diff --check` 通过。签名 helper、目标应用授权撤销恢复和 packaged Gate B 仍未验证。

本轮增量（2026-09-01）：新增 `electron/native/windows/windows-native-host.cpp`，通过 Windows UI Automation COM API 提供有界只读控件树观察（`windows.uiAutomation.read`），并通过 `RegisterRawInputDevices` 提供仅修饰键按下/释放的 `windows.bareModifierMonitor.start/stop` 事件；明确禁止 ValuePattern 写入和鼠标/键盘注入。新增 `WindowsNativeHostClient`、`windows_native_host_invoke` Host/IPC 命令、`desktopCapabilities.uiAutomation/rawInput`，并将 helper 路径、SHA-256、API 和 `readOnly` 元数据登记到 Windows desktop resource manifest。Windows helper 已列为必需资源，verifier/readiness 对缺失、路径越界、digest 漂移和元数据异常 fail closed；未将非 Windows 主机上的源码或本地 manifest 测试升级为 Windows 签名/安装后 Gate B。受影响 Vitest `137/137`、Electron TypeScript 通过；Windows MSVC/GNU 编译、Raw Input/UIA 真机授权、Squirrel 安装后 sidecar 和 Gate B 待在 Windows runner 完成。

本轮增量（2026-09-01，Windows 窗口/显示观察）：同一 native host 增加 `windows.window.read`（可见顶层 HWND、进程 ID、标题、类名、边界和最小化状态）与 `windows.display.read`（显示器边界、工作区、设备名和主显示标记），继续保持只读，不实现窗口移动/激活/隐藏或 overlay。`desktopCapabilities.windowHandles/displays` 在 helper 资源存在时报告 `unverified`；资源 manifest API 元数据同步加入 `WindowEnumeration` 与 `DisplayEnumeration`。本机仅能验证资源生成、命令契约和负向校验，Windows SDK 编译与真实桌面枚举仍待 Windows Gate B。

本轮增量（2026-09-01，Windows display watcher）：同一 native host 增加 `windows.displayWatcher.start/stop`，通过独立隐藏顶层窗口接收 `WM_DISPLAYCHANGE`，以 `display.changed` 事件返回位深、分辨率和最新 `windows.display.read` 快照；启停、线程退出和窗口注销保持幂等，继续只读，不实现 display link、屏幕捕获或窗口控制。资源 manifest API 元数据加入 `DisplayWatcher`，`desktopCapabilities.displayWatcher` 在 helper 存在时报告 `unverified`。本机仅完成源码/合同/资源测试，Windows SDK 编译、显示器热插拔事件和 Squirrel 安装后 Gate B 仍待 Windows runner。

本轮增量（2026-09-01，Windows native host Gate B 入口收口）：新增 `npm run smoke:windows-native-host-gate-b` 作为安装后 helper 验证入口，并在 Windows test/release workflow 中固定执行与上传证据；`windows-native-host-gate-b.mjs` 从已安装 `Lime.exe` 定位 resources，复核 manifest 身份、helper 路径和 SHA-256，再真实调用 UI Automation、窗口/显示枚举、display watcher 和 Raw Input 启停。脚本在中途失败时仍写出包含已完成检查、失败原因和 digest 的 `summary.json`，随后以非零状态让 CI fail closed。当前 macOS 主机仅验证入口解析、脚本/工作流合同和负向测试，未将其升级为 Windows packaged/platform Gate B；真实 MSVC 编译、UIA 树、显示器事件、Raw Input 和签名/Squirrel 行为仍需 Windows runner。

本轮增量（2026-09-01，Windows JSONL 协议回归）：`electron/windowsNativeHost.test.ts` 补齐并发乱序响应、`display.changed` 事件转发、helper 退出时批量拒绝 pending 请求和请求超时清理；这些测试通过可执行 Node fixture 验证 Electron Host 生命周期，不把 macOS 主机上的进程 fixture 当作 Windows API 或 packaged 证据。

本轮真实 observation smoke（2026-09-01）：使用重新编译的 arm64 helper 对当前 Lime 窗口执行 `window.read` + `accessibilityTree.read(maxDepth=3,maxNodes=200)`，真实 Accessibility 授权返回 `AXWindow` 根节点、标题 `Lime`、`nodeCount=8`、`truncated=false`；`display.watch.start -> start -> stop -> stop` 返回 `alreadyRunning/stopped` 幂等结果并包含当前两块显示器。该证据属于本机 native `lime-local/live`，未使用用户输入注入，也未升级为 packaged Gate B；重建 dist packaged helper 期间因并行 Cargo 构建占满磁盘（`No space left on device`）失败，未篡改资源清单或绕过构建链。

## 10. 计划内证据和禁止混淆项

以下证据必须分开记录：

| 证据                                                | 可用于                                   | 不可用于                              |
| --------------------------------------------------- | ---------------------------------------- | ------------------------------------- |
| ChatGPT.app `Info.plist`/entitlements/`file`/链接库 | 证明安装包声明和原生依赖存在             | 证明 Lime 已实现或权限已授予          |
| asar JS 字符串/模块名                               | 证明 Desktop 有调用面和资源              | 证明 API 语义、系统行为或跨平台等价   |
| Codex Rust 源码和测试                               | 证明开源 current owner 与算法/协议       | 证明私有 Swift helper、打包和系统 API |
| Lime Rust related/integration                       | 证明 current owner 的源码行为            | 证明 Electron/preload、真实 OS        |
| Browser Gate A                                      | 证明 Renderer projection                 | 证明 Desktop Host 和系统权限          |
| Electron Gate B                                     | 证明 Lime current desktop chain          | 证明 live provider 或另一个 OS        |
| macOS signed/notarized candidate                    | 证明签名、entitlement、资源和系统启动    | 证明 Windows 或未运行的能力           |
| Windows packaged candidate                          | 证明安装、sidecar、runner 和 Windows API | 证明 macOS Swift 层                   |

禁止以下表述：

- “有 `Info.plist` 权限描述，所以已经支持 Input Monitoring/Accessibility”；
- “有 Windows Rust 源码，所以 Windows packaged Desktop 已完成”；
- “有 `sky.node`/asar 字符串，所以 Lime 已支持 Computer Use/PIP”；
- “V8 cache 能编译，所以 Darwin/Windows MSVC Code Mode 资源正确”；
- “浏览器 fixture 通过，所以真实 Electron 或系统权限通过”。

## 11. 最低验证清单

纯文档阶段：

```bash
git diff --check
npm run governance:legacy-report
```

新增或修改 capability/protocol/resource contract 后：

```bash
npm run test:contracts
npm run test:rust:related -- <changed-rust-paths...>
npm run governance:scripts
```

GUI/Desktop Host 改动后：

```bash
npm run verify:gui-smoke
npm run smoke:agent-runtime-current-fixture
```

平台交付必须额外保存：

- macOS：签名 identity、entitlements、notarization/stapling、helper/framework 嵌套签名、权限授予/撤销 Gate B、DMG/ZIP digest；
- Windows：Squirrel 安装/升级/卸载、sidecar/runner 资源 digest、MSVC restricted matrix、ConPTY/Job Object/WFP/readiness、安装后 Electron Gate B；
- 两端：平台、架构、版本、候选 SHA、证据等级、未执行原因、失败时的 fail-closed 状态。

## 12. 下一刀与完成定义

当前最直接推进主线的一刀是 **P1 Windows packaged/Squirrel 与 macOS 权限/bookmark 撤销恢复 Gate B**：在同一候选产物中证明 sidecar/runner 安装生命周期，并在真实 macOS 用户授权撤销后验证 bookmark/readiness 恢复语义；Windows watcher 需要在同一候选上补显示器热插拔事件证据。Chronicle/PIP/Computer Use 仍需独立产品裁决和 owner，不得用本切片的窗口接口冒充完成。

本计划只有在以下条件全部满足后才能标记完成：

1. macOS 私有原生层清单、边界和产品范围已由仓库事实源维护；
2. Windows Rust、安全矩阵、packaged sidecar、安装更新和 Electron Gate B 已形成同一候选产物证据；
3. macOS 至少完成 P1 权限/Launch Services/bookmark current owner 和真实 Gate B；
4. 需要的 P2 原生窗口、PIP、Computer Use、屏幕/输入/密钥能力逐项有 owner、协议和平台证据；不需要的能力明确 `product-scope-excluded`；
5. 没有新的第二业务后端、生产 mock fallback、旧 runtime 恢复或无退出条件的 compat 层。

路线图关系：本计划是对现有 Codex runtime/GUI 对齐计划的平台底座补充；完成 P0 后回到 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore/tool-runtime -> Thread/Turn/Item -> GUI` 主链推进，而不是继续扩展静态 Desktop 逆向清单。

## 13. 2026-09-01 Codex 业务层 `update_plan` 对齐

- 对比依据：Codex HEAD `d58d0e5841` 的 `tools.update_plan.enabled` 默认值为 `false`，只有显式配置开启时才把 `update_plan` 暴露给模型；Plan Mode 是独立的编排模式，不能用来推断工具是否注册。
- Lime current 实现：`lime-core::ToolExecutionPolicyConfig.update_plan_enabled` 作为唯一持久化字段，支持 snake/camel 配置别名，默认关闭且默认值序列化为空；App Server `current_agent_runtime_config_metadata` 只投影非默认配置，`tool-runtime::update_plan_enabled_from_metadata` 只接受 trusted runtime/config metadata，Agent inventory 和 current provider tool surface 共用该 gate；设置页提供五语言开关并写回 `agent.tool_execution.update_plan_enabled`。
- 回归证据：配置序列化/round-trip `3` 个 Rust 测试通过，设置页 `5` 个 Vitest 通过；Agent inventory `15/15`、current provider tool snapshot `6/6`、App Server provider metadata `3/3`、App Server inventory `3/3` 均通过，`test:rust:related` 全部受影响 Rust 包通过，另有 `lime-core` cargo check、Prettier、`git diff --check`。默认未注入缓存时 `rusty_v8 v150.4.0` Darwin arm64 下载地址返回 404；本轮使用仓库已有临时预编译缓存完成验证，未声称全新环境的 V8 下载可用。
- 分类：`current` 为上述配置、trusted gate、Agent tool inventory/current provider 投影和设置页；`compat` 仅保留既有 `UpdatePlan`/`UpdatePlanTool`/`update_plan_tool` 历史别名；`deprecated` 不新增入口；`dead/forbidden-to-restore` 为 Renderer mock、第二套计划状态机及 OpenAI 私有 Application Group/`sky` 平级实现；Codex TUI/account/rate-limit、未确认的 model-owned token budget defaults、完整 Computer Use/Chronicle/PIP 标记为 `product-scope-excluded` 或 `gap / pending owner`。

## 14. 2026-09-02 跨平台共用业务语义继续对齐

- 本轮不新增 macOS Swift、Windows native host 或 Electron 业务后端；Desktop 两端共用同一条 `model catalog -> App Server route metadata -> session/turn context -> model-provider wire` current 链。平台 host 只继续承接系统能力。
- 已实现 Codex 模型目录拥有的 instructions/personality、context/auto-compact、Multi-Agent v2 mode copy 和 Ultra reasoning lowering。`ultra` 留在 Thread/Turn World State 表达主动协作意图，真实 provider 请求使用 catalog 声明或合法 fallback；`persistent` wire 使用 `disabled`。
- `model/list.supportsPersonality` 已改为模型目录事实，不按模型名或操作系统推断；空变量值仍表示显式支持。该行为在 macOS/Windows Composer 和 Thread 设置上共享，不产生平台分叉。
- 当前分类：上述路径全部为 `current`；未新增 `compat`/`deprecated`；OpenAI 私有 Application Groups、`sky`/CUAService 和签名 entitlement 继续为 `dead/forbidden-to-copy`；model-owned approvals/permissions/auto-review、Guardian v2 与完整 token-budget 仍为 `gap / pending owner`。
- 验证状态：`cargo check -p app-server`、目录/provider/session 定向回归和 `test:rust:related` 推导的 20 个反向依赖包全部通过；`test:contracts` 通过 protocol types、App Server client 299 checks 及 command/harness/docs 治理边界；`smoke:agent-runtime-current-fixture` 在 `liveProviderUsed=false` 下通过；`verify:gui-smoke` 通过真实 Electron Desktop Host、preload/IPC、App Server `appserver.v0`、Claw shell reload 和 memory settings Gate B；Rust fmt 与 `git diff --check` 通过。Codex release 的 Darwin arm64 archive/binding 均可下载并校验；历史 404 仅对应 Deno 默认 sandbox URL 或错误的 mirror 路径，本轮已将缓存固定到稳定 OS 用户缓存目录并增加下载重试/超时，不以 mock 或第二 backend 绕过。

## 15. 2026-09-02 Codex Rusty V8 资产下载收口

- 继续采用 Codex 的 `rusty-v8-v<version>` release 资产，不引入 Lime 自建镜像或关闭 `v8_enable_sandbox`。`RUSTY_V8_MIRROR` 不作为入口，因为 v8 crate 的 `/v<version>` 拼接规则与 Codex release tag 不兼容。
- `scripts/lib/rusty-v8-artifacts.mjs` 现在使用 macOS/Windows/Linux 稳定用户缓存目录，并为 curl 下载增加有限重试、连接超时和总超时；显式 `LIME_RUSTY_V8_CACHE_DIR` 仍可用于隔离 CI 或维护者验证。
- 验证：helper 真实解析 `v8=150.4.0` 并成功取得 Codex Darwin arm64 archive/binding；资产 supply-chain Vitest `6/6` 通过，输出路径位于 `~/Library/Caches/Lime/rusty-v8`。该切片不改变 Cargo.lock、V8 feature 或运行时资源打包边界。

## 16. 2026-09-02 Windows packaged Gate B identity contract

- Windows packaged 证据现在由 `scripts/electron/windows-packaged-evidence.mjs` 统一收口。它要求 Squirrel summary 的候选版本、`candidateRunId`、安装路径和 `Lime.exe` 版本目录一致，并从安装后的 `resources/desktop-resources.manifest.json` 校验 app-server、code-mode-host、Windows sandbox helpers 和 read-only native host 的路径与 SHA-256。
- CodeMode 与 Windows native host Gate B 都接收同一个 `LIME_GATE_RUN_ID`，并记录 packaged executable、resources root 与 helper path；validator 要求三份证据都来自同一个 Squirrel-installed `Lime.exe`，拒绝旧版本路径、`target/debug`/开发 sidecar、缺失 summary 或未完成 Gate B。
- `build-windows-test.yml` 与 `release.yml` 在安装后 Gate B 后执行 validator，并始终上传结构化 `windows-packaged-evidence` summary。工作流 guard 和 Vitest 覆盖参数、哈希、路径、候选 identity、缺失证据与执行顺序。
- 证据边界：这只提升 Windows packaged 安装、sidecar、native host 和业务 Gate B 的候选一致性；当前工作树没有真实 `windows-2022` 运行结果，Windows 安装/升级/卸载、进程树和 GUI 视觉对照仍保持 `OPEN_REF`，不以 macOS 本地 smoke 替代。

## 17. 2026-09-02 跨平台 GUI responsive layout contract

- Lime 的 macOS/Windows Desktop Host 共用同一套 GUI 几何回归入口：真实 Electron `BrowserWindow.setSize/getSize` 在 `1536x960`、`1280x800`、`980x680` 三档采集 workspace shell、Composer、输入框和可选 Thread/timeline 节点；布局证据与 `SHELL-01` summary 绑定，三张截图和几何断言缺失时 fail closed。
- 短窗口的 EmptyState Composer 已按最小 `980x680` 窗口收敛到视口内，未改变平台 host、App Server、RuntimeCore 或 provider 业务 owner。该 contract 只证明 Lime responsive layout 和真实 Electron 主链，不证明 Codex Desktop 像素级视觉一致性、Windows packaged 安装或 macOS 系统权限。
- 当前验证：真实 macOS Electron smoke 24/24 assertions 通过；Windows 必须在 `windows-2022` 安装后使用同一候选 `Lime.exe` 重跑 GUI/CodeMode/native host Gate B，不能用本机结果替代。Codex Desktop 实时窗口因 AppleEvent 阻塞仍为 `OPEN_REF`。

## 18. 2026-09-02 GUI responsive layout Gate B-F 复核

- 真实 Electron smoke 重新构建并运行后，summary `.lime/qc/project-gates/standalone-shell-01-20260902105148-51228/shell-01-electron-smoke/summary.json` 通过 `24/24` assertions；三档 `1536x960`、`1280x800`、`980x680` 的窗口尺寸、viewport、必要 workspace/Composer/input 节点和无横向溢出均通过，三张布局截图存在。
- `npm run test:contracts`、`npm run governance:scripts`、`npm run typecheck:electron` 与定向 26/26 回归通过；证据仍只覆盖 macOS 本地真实 Electron/preload/IPC/App Server shell 和 Lime responsive layout contract。
- 统一 `npm test -- --resume` 的 119/119 批全部通过；i18n 设置文案保持来源中性，治理边界测试读取 current 架构事实源。
- Windows packaged/Squirrel、sidecar/native host/CodeMode 以及 Codex Desktop 实时 accessibility/screenshot 没有新增平台运行证据，继续保持 `OPEN_REF`，不得用本机 smoke 替代。

## 19. 2026-09-02 macOS native helper readiness handshake

- macOS 资源 manifest 的 native helper metadata 现在固定声明 `protocolVersion=1`；Swift `capabilities.read`
  同时返回 `protocolVersion`、`helperId`、`platform` 和 helper bundle identity。Electron
  `MacOSNativeHostClient` 对声明协议版本的 packaged helper 在首次业务调用前执行一次握手；协议、平台或
  `com.limecloud.lime.native-host` identity 不匹配时终止子进程并返回 `protocol_mismatch`，不会把可执行文件存在
  当成 runtime ready。未声明版本的隔离测试 fixture 不进入生产资源清单。
- 新增 `scripts/electron/macos-native-host-gate-b.mjs` 与 `npm run smoke:macos-native-host-gate-b`。该入口从
  `Lime.app/Contents/MacOS/Lime` 解析 Resources，校验 manifest、helper bundle、SHA-256/严格 codesign 和握手，
  再真实观察 `window.read`、`display.read`、display watcher、权限查询、Apple Events target 列表和
  `launchServices.bundleIdentifier`。默认权限模式为 `observe` 并记录 TCC 状态；`--strict-permissions` 才要求
  Accessibility、Input Monitoring、Screen Recording 全部为 `ready`。
- release workflow 的 macOS arm64/x64 job 在资源 verifier 后执行该 Gate B 并上传 evidence；当前工作树只完成入口、
  manifest/握手负向测试和本地 helper 运行，尚无 CI 签名包权限授予/撤销结果，不能将 observe 证据升级为完整 macOS
  permission Gate B，也不能替代 Windows packaged 或 Codex Desktop 实时视觉证据。
- 相关验证：待本轮执行 `macosNativeHost`、资源清单/verifier、Gate B 入口、current entrypoints、contracts、
  Electron typecheck、GUI smoke 与 `git diff --check`；Windows packaged、真实签名权限和 Codex Desktop
  accessibility/screenshot 继续为 `OPEN_REF`。

## 20. 2026-09-02 本地 packaged macOS helper Gate B 证据

- 重新执行 `npm run electron:build` 和 `electron-forge package --platform darwin --arch arm64`，使用生成的
  `release-electron/Lime-darwin-arm64/Lime.app` 运行 `macos-native-host-gate-b.mjs`。Gate B 校验了顶层 Lime
  app bundle identity/codesign、嵌套 helper bundle、manifest protocol/digest、`capabilities.read` 握手、窗口
  枚举（46）、显示器枚举（2）、display watcher、security-scoped bookmark 的真实
  `create/resolve/start/stop`、权限查询、Apple Events targets（89）和 Launch Services bundle ID。
- observe 与 `--strict-permissions` 两种模式均通过；证据分别写入
  `.lime/qc/gui-evidence/macos-native-host-gate-b-local/summary.json` 和
  `.lime/qc/gui-evidence/macos-native-host-gate-b-local-strict/summary.json`。helper 在 ad-hoc/本地签名后
  SHA-256 与 manifest 不同，但严格 `codesign` 通过，Gate 正确记录 `digestMatches=false`、`signed=true`，没有
  绕过完整性校验。
- 该结果属于本机 `lime-local/packaged`，不等同于 Developer ID/notarization、权限撤销后恢复或 Codex Desktop
  实时 accessibility/screenshot 对照；macOS release runner、Windows `windows-2022` packaged 和上述撤销场景仍为
  `OPEN_REF`。
- 本轮最终验证：受影响 Gate B/资源/host 相关测试 `47/47`，完整资源 verifier 35 tests，Electron typecheck、
  `npm run test:contracts`、`npm run governance:electron-release-workflow`、`npm run verify:gui-smoke` 和
  `git diff --check` 通过。

## 21. 2026-09-02 macOS window lease 与 bookmark lifecycle Gate B

- `macos-window-fixture.swift` 提供仅用于 Gate B 的临时标准 Cocoa `.app`，运行时创建两个带稳定标题的窗口；Gate 脚本只匹配该 fixture 的 PID，不选择 Finder、Terminal 或其它用户窗口。
- `macos-window-orchestration.swift` 的 hide-for-task lease 改为在 Accessibility 已授权时设置目标应用的 AX `kAXHiddenAttribute`，隐藏前激活目标并保留每个 owner 的原始状态；stop 和 helper 退出路径恢复原始状态。`window.anchor`、`window.stack` 与 lease 的 start/read/stop 在同一 fixture 上完成。
- `SystemUtilityHost` 将按稳定 ID 启动的 bookmark token 保存在 current Host 内存中；`bookmark.stop` 支持稳定 ID，`bookmark.revoke` 会先调用 helper `bookmark.stop` 再删除受管记录，避免撤销后仍持有 active security-scoped resource。
- 本机新 packaged arm64 证据：`.lime/qc/gui-evidence/macos-native-host-gate-b-final-observe/summary.json`（observe）和 `.lime/qc/gui-evidence/macos-native-host-gate-b-final-strict/summary.json`（strict）均通过；窗口编排与 bookmark lifecycle 均为 `passed`，TCC 三项为 `ready`。该证据仍是本机 `lime-local/packaged`，不升级为 Developer ID/notarization 或权限撤销后恢复证据。
- 本轮定向验证：bookmark/system utility 与 Gate B/current entrypoint 测试 `24/24`，`npm run electron:build:host`、Swift arm64 编译、Forge packaged Gate B observe/strict 和 `git diff --check` 通过；Windows packaged、macOS release runner 和 Codex Desktop 实时 accessibility/screenshot 仍为 `OPEN_REF`。

## 22. 2026-09-02 macOS native host 真实 Electron/preload/IPC 闭环

- `scripts/electron/macos-native-host-gate-b.mjs` 在同一候选 `Lime.app` 上先完成 packaged helper 资源、协议、签名、窗口/显示、bookmark、权限和 Launch Services 检查，再启动真实 Electron fixture；不会切换到第二个 Electron 后端或 renderer mock。
- Electron 阶段通过 preload `window.electronAPI.invoke` 调用 `macos_native_host_invoke`，校验 helper `capabilities.read` identity、`window.read`、`display.read`、三项 TCC 查询和 bookmark `create/resolve/start/stop`；同时从同一 preload 进入 `app_server_handle_json_lines`，执行 `initialize`、`workspace/default/ensure`，要求返回稳定 workspace identity。
- 终态证据同时记录真实 Electron/preload 标记、App Server JSON-RPC 方法、native host IPC 方法、GUI 当前 shell 可见状态、invoke/console/page 错误和截图。`app_server_handle_json_lines` 必须在 renderer trace 中命中 `electron-ipc`，否则 fail closed。
- 该改动只扩展 Gate B 证据与测试 helper，未新增业务 owner、Application Group、`sky`/`CUAService` 或 Swift 私有业务后端；native 系统能力仍归 Desktop Host，Thread/Turn/Item 业务仍归 App Server/runtime。
- 本轮验证已完成：Gate B/current entrypoint 合同 `19/19`、`npm run typecheck:electron` 通过；使用同一 arm64 packaged `Lime.app` 的 observe 证据 `.lime/qc/gui-evidence/macos-native-host-gate-b-electron-observe/summary.json` 与 strict 证据 `.lime/qc/gui-evidence/macos-native-host-gate-b-electron-strict/summary.json` 均为 `result=passed`。两档均证明真实 Electron/preload、`app_server_handle_json_lines`（observe trace 13 次）、`macos_native_host_invoke` 10 个方法、workspace identity、GUI 设置页截图和零 console/page/invoke error；strict 还证明 Electron IPC 路径上的 Accessibility、Input Monitoring、Screen Recording 均为 `ready`。
- 仍未覆盖：Windows `windows-2022` packaged/Squirrel、macOS release runner 权限撤销恢复及 Codex Desktop 实时 Accessibility/screenshot；这些继续保持 `OPEN_REF`，本机 packaged 证据不能替代跨平台或签名发布证据。

## 23. 2026-09-02 macOS Electron Gate B owner 收口

- Electron/preload/App Server/native host 闭环已从 `macos-native-host-gate-b.mjs` 抽取到
  `scripts/electron/lib/macos-native-host-electron-gate-b.mjs`，主入口只保留已安装 helper 的资源、协议、权限、窗口和
  Launch Services 校验，并委托唯一 helper 执行真实 Electron 阶段；主入口由 1059 行降至 779 行，避免继续堆叠跨职责业务逻辑。
- current entrypoint 守卫已登记 helper，Gate B 合同测试直接读取 helper，保留 `launchElectronFixture`、
  `window.electronAPI.invoke`、`app_server_handle_json_lines`、`macos_native_host_invoke`、trace fail-closed 和 GUI 终态断言。
- 本轮抽取后验证：Gate B/current entrypoint `20/20`、Electron TypeScript、`test:contracts`、脚本治理、`verify:gui-smoke` 和
  `git diff --check` 通过；重新使用同一 arm64 packaged `Lime.app` 的 observe/strict 两档 Gate B 均为 `passed`，
  两档都命中 `app_server_handle_json_lines`、`macos_native_host_invoke`、workspace identity、设置页终态和零
  console/page/invoke error。Electron helper 的业务 owner 未产生第二套后端或 mock fallback。
