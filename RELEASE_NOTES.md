## Lime v1.118.0

### 新功能

- Skill 目录现在会在默认目录新增、修改或删除 Skill，以及 Lime 本地 Skill 管理成功后自动刷新；Composer 始终重新读取 App Server 的 current `skill/list`，无需手动刷新或重启应用。
- Agent 执行计划接入 typed `turn/plan/updated` 通知，实时对话与冷恢复会展示同一份 checklist；失败或畸形的计划快照不会覆盖最近一次有效状态。
- Agent 错误升级为 typed retry/terminal 链路：重试中、最终失败与权威 Turn terminal 各自有明确状态，并可从 canonical read model 恢复。
- MCP OAuth 登录完成态改由 App Server v2 typed notification 直接投影到 Renderer，统一登录窗口、MCP 管理状态与事件 identity。

### 修复

- 修复 Windows 用户已显式配置的 Provider 模型因缺少权威 capability 来源而不出现在聊天模型选择器中的问题；仅靠启发式推断且不可执行的模型继续 fail closed。
- 修复 Electron updater 将同版本、旧版本、无效版本或缺失版本误报为可安装更新的问题；只有严格更高的语义版本进入下载和安装流程。
- 修复更新失败、已是最新和发现新版本共用提示的问题，并统一侧边栏、关于页与更新通知页的下载、重试和重启安装状态。
- 修复 Provider 路由不可执行错误在发送入口和 runtime 事件链中出现不一致提示，以及部分独立页面缺少全局 toast 宿主的问题。
- 修复 Provider 添加流程在尚未保存配置时显示“完成添加”，并在已保存后正确返回可用模型上下文。

### 优化与重构

- 将 `skills/changed`、`error`、`turn/plan/updated` 与 `mcpServer/oauthLogin/completed` 收敛到 App Server v2 protocol、schema、generated client 和 Renderer typed event bus，不新增生产 mock 或第二事件 owner。
- Skill catalog cache 统一由 `lime-skills` 失效；默认 Skill roots 使用递归 watcher 与节流通知，重连和 remount 仍主动读取 current catalog。
- canonical `update_plan` 工具项继续作为恢复事实源，但不再生成重复 Plan Item、工具卡或决策面板。
- updater 的版本比较与会话状态拆到独立纯 owner，收紧 `available -> downloaded -> installing/restarting` 状态转换。

### 测试与质量

- 新增 Skills 自动刷新、typed error、执行计划、MCP OAuth、Provider 模型目录与 updater 语义的 Rust、Electron、协议、Renderer 和脚本回归。
- 扩展 current Agent fixture，覆盖 typed error 成功/失败、计划实时更新与冷恢复、Skills runtime 自动刷新，以及真实 Electron/preload/IPC/App Server/read model identity。
- 核心用户流程审计覆盖启动、Provider/模型选择、Agent 对话、停止后继续、历史恢复、关于页与更新状态；生产 mock fallback 命中为零。

### 文档

- 更新 App Server v2 notification、Skills catalog invalidation、MCP OAuth、typed error/plan recovery 与 updater/模型目录的架构和执行事实源。
- 记录 Windows 模型目录与更新可靠性、核心用户主流程 E2E 审计，以及 Refactor v2 V2-05 当前完成情况。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与锁文件版本统一提升到 `1.118.0`。
- Windows Squirrel 从 N-1 版本发现、下载、重启安装的 packaged Gate B 仍需 Windows 实机验收；macOS Electron 证据不替代该平台验证。

**完整变更**: `v1.117.0` -> `v1.118.0`
