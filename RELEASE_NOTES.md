## Lime v1.135.0

### 新功能

- 完成 Agent Runtime 的 MCP 生命周期、事件流和结构化资源投影，统一 Thread/Turn/Item 与 GUI 状态。
- 新增环境生命周期、项目目录、Thread queue、Thread revert、Strict Review 和模型 Provider capability 的 current Electron Gate B 场景。
- 扩展 MCP 资源来源、OAuth/启动通知、事件流状态和多语言 Agent 工作区入口。

### 修复

- 修复 MCP manager/client、工具执行、资源读取和认证通知在启动竞态、生命周期终态与错误边界上的一致性。
- 修复 Agent Thread/Turn/Item 事件排序、回合恢复、消息操作、环境状态和工作区投影在 current bridge 上的漂移。
- 修复 App Server JSON-RPC、Rust runtime、前端 gateway 与 Electron preload 之间的通知和协议接线问题。

### 优化与重构

- 将环境、MCP、Provider capability、Thread 控制和严格审查收敛到 current App Server JSON-RPC 与 RuntimeCore 产品链。
- 重构 Rust App Server/runtime、前端 projection/reducer、Agent session gateway 与 MCP 状态组件，保持真实 Electron bridge 单一路径。
- 补充 Electron Gate B、公共 JSON-RPC、current fixture、资源来源和事件序列的稳定测试与证据路径。

### 测试与质量

- 扩展 App Server protocol contract、MCP exact JSON-RPC、Rust agent/app-server/tool-runtime、Agent session projection 和五语言 GUI 测试。
- 增加环境、Thread queue/revert、项目目录、Strict Review、MCP event stream/resource origin、Provider capability 的真实 Electron Gate B 验证入口。

### 文档

- 更新 Codex 对齐计划、App Server command/protocol、事件 projection、脚本治理和协议 schema 事实源，记录 current/compat/deprecated/dead 边界。
- 新增 v1.135.0 发布执行计划，记录候选范围、验证结果和平台/打包证据边界。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.135.0`。
- Windows 安装器、签名、公证、正式 release assets 和 live provider 证据仍需对应平台或 CI runner，不在本地候选中冒充完成。

**完整变更**: `v1.134.0` -> `v1.135.0`
