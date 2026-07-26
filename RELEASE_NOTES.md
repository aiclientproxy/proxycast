## Lime v1.113.0

### 新功能

- 新增 typed Runtime World State，将工作目录、项目、模型、权限、协作模式与有效 Multi-Agent 模式统一投影到 Provider 上下文。
- 为 Provider 运行链补充按路由与凭证隔离的健康状态、结构化重试遥测和无输出前的有限备用路由重选。
- 将工具 Hook 的发现、生命周期裁决与执行收敛到 `tool-runtime`，支持统一的调用前后门禁。
- 补齐 App Server v2 `model/verification` typed 通知与 schema，支持可信访问验证状态的标准投影。

### 修复

- 修复项目菜单移动到“新增项目”子菜单时命中区域断开、子菜单意外关闭的问题。
- 修复 Provider 认证类型、adapter readiness、能力上界与实际路由不一致导致的错误可选、错误回退或认证头投影问题。
- 修复 Windows Chromium session 数据落入 roaming 目录，以及 portable/E2E 场景从 AgentRoot 反推应用数据根的问题。
- 修复可重试 Provider 故障在已产生输出、已消费 steer 或 direct route 场景中被不安全重放的问题。

### 优化与重构

- 统一 AppDataRoot、AgentRoot、HostSessionData、Product DB、diagnostics、Soul 与 MCP OAuth 的存储根组合链。
- 删除 Product DB 整库复制、通用 migration manifest、启动清理和 managed project path 迁移等已退役路径，不保留兼容回退。
- 将默认 Provider、模型路由、认证配置与 capability admission 收敛到单一 current owner，未实现 adapter 默认 fail closed。
- 移除 `lime-agent` 中旧 HookManager 与旧 Provider migration fixture，补充防止 dead surface 回流的治理规则。

### 测试与质量

- 扩充 Runtime World State、Multi-Agent mode、Hook lifecycle、Provider health/retry/reroute、workspace scope 与 App Server JSON-RPC 回归测试。
- 补充 Electron 存储根、Windows sessionData、项目子菜单连续指针命中和旧迁移路径负向守卫。
- 同步 App Server v2 schema 与 TypeScript generated protocol types，更新项目 Gate B 场景合同。

### 文档

- 更新 Agent Runtime、Provider、存储对齐、架构事实源、实现快照与 Codex 对齐执行记录。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与锁文件版本统一提升到 `1.113.0`。

**完整变更**: `v1.112.0` -> `v1.113.0`
