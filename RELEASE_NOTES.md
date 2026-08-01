## Lime v1.119.0

### 新功能

- MCP Server 启动过程接入 App Server typed 状态通知，设置页可实时展示 `starting`、`ready`、`failed` 与 `cancelled`，并在终态自动刷新服务器与工具目录。
- Agent 对话现在会安全展示未来协议产生的 unknown Item；实时流、完成态、历史恢复与重新加载共享同一 canonical Item，不再静默丢失未知能力记录。
- unified exec 的后续终端输入会回挂原始 Command Item，并以脱敏摘要呈现在实时和历史对话中；原始 stdin 不进入通知、持久化或 GUI。

### 修复

- 修复 Agent 回合完成刷新后 unknown Item 从 direct timeline 消失的问题，并保留同一 thread、turn、item identity 进行冷恢复。
- 修复 `write_stdin` 被投影为独立 Tool Item、会话结束后绑定未及时释放，以及跨 Thread 写入未明确失败的问题。
- 修复 MCP 启动状态依赖旧 Desktop events 导致连接态与 App Server 权威状态可能不同步的问题。

### 优化与重构

- 收敛 MCP lifecycle 到 `mcpServer/startupStatus/updated` 单一 owner，删除旧 `mcp:server_started`、`mcp:server_stopped` 与 `mcp:server_error` 生产路径并补回流守卫。
- 统一 command terminal interaction、unknown Item 在协议、schema、App Server projection、read model、客户端与 Renderer 中的 typed 表达。
- unknown Item 仅暴露上游类型和脱敏字段名，禁止 raw value、event metadata 与通用 extension fallback 进入产品链。

### 测试与质量

- 新增 MCP startup notification 真实 Electron Gate B，覆盖成功与失败状态、自动刷新、IPC/App Server 命中及零生产 mock fallback。
- 新增 unknown Item live/cold recovery Gate B，覆盖安全字段展示、终态恢复、身份一致性与敏感值不泄漏。
- 扩展 Codex import continuation、Agent runtime fixture、协议合同、Rust owner 与 Renderer 回归，覆盖终端输入摘要的实时、历史和 reload 路径。

### 文档

- 更新 MCP current lifecycle、命令边界、Refactor v2 event projection 与执行进度，记录 current、排除范围和 dead surface 裁决。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与锁文件版本统一提升到 `1.119.0`。

**完整变更**: `v1.118.0` -> `v1.119.0`
