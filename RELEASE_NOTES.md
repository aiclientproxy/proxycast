## Lime v1.137.0

### 新功能

- 新增线程活动面板，统一展示子任务、MCP 工具和技能执行状态，并支持从工作区导航打开。
- 新增线程 fork 的 Electron Gate B 场景与环境选择投影，保留真实 Thread/Turn/Item 身份和环境世界状态。
- 新增 MCP 工具、提示词和资源列表变更通知，GUI 收到通知后自动刷新对应数据。

### 修复

- 修复 Electron App Server 连接层提前消费通知后，近期通知无法回放的问题，并为 MCP 进度通知补齐 typed projection。
- 修复权限配置文件目录与运行时策略脱节的问题，线程启动、恢复和 fork 现在使用同一套允许性校验。
- 修复 Agent Runtime canonical thread projection 的环境、Provider 路由和会话元数据归一化，避免 GUI 读取原始业务元数据。
- 修复 Workspace、Canvas Workbench 变更摘要、线程导航和会话生命周期在多状态切换下的显示与交互边界。

### 优化与重构

- 将权限 profile、环境选择和线程 fork 状态收敛到 App Server/RuntimeCore 的 current owner，保持 Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> GUI 单一路径。
- 重构 MCP 事件桥接、App Server session client、线程活动与 Right Surface 投影，减少重复状态读取并统一五语言 UI 文案。
- 扩展 Agent Runtime、MCP、Session History 和 Workspace 的可观察性与 Gate B smoke 脚本，支持真实通知回放和终态断言。

### 测试与质量

- 新增线程活动面板、线程 fork、MCP list-changed、权限 profile、环境生命周期和 Canvas Workbench 回归测试。
- 扩展 App Server protocol/client contracts、MCP notification projection、Agent current fixture、Electron Gate B 和五语言资源校验。
- 发布前执行版本一致性、TypeScript、协议 contracts、受影响 Rust/前端定向测试及真实 Electron GUI smoke；结果记录在发布执行计划。

### 文档

- 更新命令边界、Codex App GUI 对齐计划和脚本目录说明，记录 MCP 通知、权限 profile、环境投影与线程活动的 current owner。
- 新增 v1.137.0 发布执行计划，明确候选范围、排除的构建参考文件、验证证据和发布收口步骤。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.137.0`。
- Windows 正式签名、公证、跨平台 installer/release assets 和 live provider 证据仍需对应平台或 CI runner，本地不作虚假声明。

**完整变更**: `v1.136.0` -> `v1.137.0`
