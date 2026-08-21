## Lime v1.133.0

### 新功能

- 完成 Right Surface Browser Workspace 的同 Tab current 闭环，统一 Electron `WebContentsView`、BrowserRoute、动态工具和用户可见状态投影。
- 新增 Browser 高风险 action 的 preflight/approvedExecute 两阶段审批合同，支持 allow-once、decline、cancel、用户接管和 fail-closed 身份校验。
- 新增 Browser Gate A/Gate B 场景及 approval、cancel、disconnect、download、permission、user-control、window-close 证据入口。
- 增强 Agent 会话恢复、流式续接、历史 Thread/Turn/Item 投影和未完成回合的 cold-read 归一化。

### 修复

- 修复 Browser 动态工具、Electron reverse request、App Server JSON-RPC 与 RuntimeCore 之间的 session/thread/turn/item/call identity 边界。
- 修复用户 native 输入与 Agent CDP 输入的区分，用户接管时撤销 debugger、snapshot、approval token 和 active turn，阻止旧 mutation 重放。
- 修复动态工具生命周期、审批终态、断连/取消/窗口关闭以及下载和权限状态的投影与清理。
- 修复 Agent 流控、resume binding、历史合并、终态消息和中断内容在重复通知、晚到事件及 reload 场景下的不一致。
- 修复跨平台执行进程环境构造，收敛 Windows、macOS/Linux 的环境变量与 shell 继承行为。

### 优化与重构

- 将 Browser、审批和会话状态收敛到 Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI 唯一产品链。
- 重构 BrowserTabHost、动态工具 Host、会话状态 hook、Browser Workspace 状态面板及相关协议/schema/generated client，移除重复状态机和旧残留入口。
- 补充架构、浏览器路线图、Agent runtime 恢复计划、基准与测试治理记录，明确 current/compat/deprecated/dead 边界。

### 测试与质量

- 新增和扩展 BrowserTabHost、动态工具、App Server、Rust tool-runtime、Agent 会话状态、流控、Browser Workspace 与协议合同测试。
- 增加真实 Electron Browser Gate A/Gate B 及多场景证据脚本，覆盖审批、取消、断连、下载、权限、用户接管和窗口关闭。
- 发布门禁将执行版本一致性、TypeScript 类型检查、协议合同、受影响 Rust/前端测试、GUI smoke、Agent current fixture、治理与本地验证；实际结果记录在本计划中。

### 文档

- 更新全局架构、Browser current owner、审批合同、用户接管、会话恢复和 Gate A/Gate B 证据说明。
- 新增 v1.133.0 发布执行计划，记录候选范围、验证结果和平台/打包证据边界。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.133.0`。
- Windows 安装器、签名、公证、正式 release assets 和 live provider 证据仍需对应平台或 CI runner，不在本地候选中冒充完成。

**完整变更**: `v1.132.0` -> `v1.133.0`
