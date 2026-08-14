## Lime v1.128.0

### 新功能

- 完成 Codex Orchestrator 对齐的主要运行时能力：AgentControl 具备 root-scoped 子 Agent 执行容量、驻留容量与冷恢复路径。
- 新增可选的共享 `agent.rollout_budget`，按 root tree 汇总 provider usage，支持提醒、耗尽拒绝、执行中取消与重启恢复。
- 新增统一工具执行编排器，统一 shell、`apply_patch` 和 unified exec 的审批、沙箱/网络拒绝分类、一次升级重试、取消和 attempt telemetry。
- 新增 Orchestrator-owned Skills/MCP：通过 session-owned `codex_apps` 进行有界资源发现与精确正文读取，支持配置开关和 fail-closed 行为。

### 修复

- 修复 AgentControl 并发超限、root 隔离、终态释放、驻留淘汰和失败恢复的边界行为，超限统一返回 canonical `agent_limit_reached`。
- 修复 managed-network 拒绝后的 host/protocol 审批上下文传递，确保升级重试复用同一 Tool identity 并保留原沙箱权限。
- 修复文件变更生命周期投影的重复行，GUI 对相同路径聚合为单一可见记录。
- 修复 Plugins 工作区 Skills/Experts 页签的项目作用域传递，聚合 Agent fixture 不再依赖已退役的独立 Skills 入口。

### 优化与重构

- 将工具 attempt、AgentControl capacity/residency、rollout budget 和 Orchestrator Skills/MCP 收敛到各自唯一 Rust owner，Electron/Renderer 继续只承载 JSON-RPC 转发与 canonical projection。
- 扩展 MCP 资源分页 cursor、网络审批上下文及 Orchestrator Skill source/authority 的协议 schema 与生成客户端。
- 让 provider usage、rollout reminder、reroute 和 history 恢复沿同一 canonical EventLog/Thread/Turn/Item 主链传递，避免第二套 transcript 或预算状态。

### 测试与质量

- AgentControl、execution capacity/residency/restart、rollout budget、tool execution orchestrator 与 provider retry 定向 Rust 测试通过（`52/52`、`8/8`、`8/8`、`1/1`）。
- 受影响前端与 fixture 测试 `126/126`，fixture source-contract `83/83`；`npm run typecheck`、`npm run test:contracts`（312 项）、`npm run governance:legacy-report` 与 `npm run governance:scripts` 通过。
- 已保留 Orchestrator Skills/MCP 开关与资源读取、Agent capacity 拒绝、sandbox/managed-network 升级重试的 current bridge/read-model Gate B 证据；基础 GUI smoke `21/21` 通过且 mock/legacy/console/invoke 错误为零。工具执行聚合 smoke 的最新重跑问题已记录在执行计划，不在本版本宣称为全绿。

### 文档

- 更新全局架构、命令边界与 Orchestrator 对齐执行计划，记录唯一 owner、配置/协议边界、Gate B 证据与未完成证据项。
- 新增 Orchestrator Skills、Agent capacity、工具重试和 Electron Gate B 的稳定 smoke/fixture 脚本与回归测试。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.128.0`。
- terminal-slot reuse/residency 的完整 GUI 证据与 rollout-budget exhaustion/cancellation/restart 的完整 GUI 证据仍是后续执行计划项；本版本不宣称这些证据缺口已闭合。

**完整变更**: `v1.127.0` -> `v1.128.0`
