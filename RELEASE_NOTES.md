## Lime v1.123.0

### 新功能

- 新增 Codex 对齐的 Hook lifecycle：支持 `hooks/list`、受信 command Hook、`hook/started` / `hook/completed` 通知、canonical Hook Item 与桌面时间线展示。
- 将可执行 Skill catalog 收口到 exact `skills/list`，补齐 `skills/changed` 自动刷新、`skills/config/write` 用户级启停和 `skills/extraRoots/set` 进程级目录控制。
- 新增 exact `plugin/search`，支持按关键词、scope、workspace、cursor 与 limit 搜索 current Plugin catalog。
- 新增 Apps catalog/readiness 的 `app/list`、`app/read`、`app/installed` 与 `app/list/updated`，复用 Plugin App capability 与真实运行时状态。
- 将 MCP 资源读取与工具调用迁到 `mcpServer/resource/read` 和 Thread-scoped `mcpServer/tool/call`，保留结构化工具结果并统一 Session-owned runtime。

### 修复

- 修复已有会话在模型能力目录缺失或失效时直接拒绝回合的问题：`turn/start` 会先刷新当前 Provider 的权威目录，再重新执行 fail-closed route admission。
- 修复持久化 AgentControl 精确路由在凭证暂不可用时被 catalog reconciliation 自动回退的问题；通过 schema 校验的 route 现在保持原模型与 Provider，并在 child 会话可见前写入 canonical metadata。
- 修复 App Server v2 notification、canonical Thread/Turn/Item、历史恢复与 Renderer timeline 之间的 Hook、Skill、MCP 项投影漂移。
- 修复 Model Selector 展示不可执行推断模型的问题，继续只允许具有 canonical 或 provider-explicit capability 的模型进入 Agent 路由。

### 优化与重构

- 物理删除 singular `skill/list`、旧 MCP tool/resource wire、无 Thread 的 Settings 工具执行入口及对应 Electron facade，不保留 compat 包装或生产 mock fallback。
- 收敛 App Server v2 protocol、schema registry、Rust/TypeScript typed client、runtime owner 与 GUI gateway，保持 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI` 单一产品链。
- 将 Hook discovery、信任校验、sampling gate、生命周期事件和恢复投影统一到 `tool-runtime`、Agent runtime 与 App Server current owner。

### 测试与质量

- 新增 Hook lifecycle、Skills list/config/extra roots、exact MCP resource/tool、Plugin search、canonical notification 与历史恢复的 Rust、TypeScript 和 public JSON-RPC 回归。
- 扩展 Agent current fixture、MCP current fixture、Workspace MCP fixture、Provider generation PendingRoute 与真实 Electron Hook Gate B 证据。
- 同步 generated protocol schema/types、命令契约、legacy 回流守卫、脚本治理与五语言 GUI 回归。

### 文档

- 更新 App Server 命令边界、MCP/Skills/Hook current owner、重大架构图、Codex 对齐计划与 Windows 模型路由修复证据。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.123.0`。

**完整变更**: `v1.122.0` -> `v1.123.0`
