## Lime v1.124.0

### 新功能

- 将 Lime 的产品定位升级为全栈 AI Agent：统一覆盖代码、文件、终端、工具、MCP、Skills、多模态、Provider 和多 Agent 长任务。
- 新增 current App Server v2 文件系统与进程能力，支持目录/文件读写、复制、删除、监听、终端进程启动、输出流、stdin、终止与状态恢复。
- 将 Agent 工作区、Thread/Turn/Item 投影、artifact 和桌面 GUI 连接到同一条可追踪执行链，面向类似 Claude Code、WorkBuddy、Codex 的任务型 Agent 使用方式。
- 补齐 current Plugin package、Browser 能力资产和 Gate B 打包/运行验证入口。

### 修复

- 修复文件、进程、后台终端和 Agent 任务状态在 App Server、Electron host、gateway 与 GUI 之间的投影不一致。
- 修复 current runtime 在长任务、取消、输出流和历史恢复场景中的生命周期边界，保持失败关闭和权限审阅语义。
- 修复协议 schema、生成客户端和模型/Provider 能力目录在 v2 迁移后的漂移，避免不可执行模型进入 Agent 路由。

### 优化与重构

- 物理删除已退役的 Plugin runtime、旧 worker、旧 v0 文件/进程/插件 wire 与脱离构建图的 Electron/Renderer facade，不保留生产 mock fallback 或兼容双轨。
- 收敛 App Server v2 protocol、schema registry、Rust/TypeScript typed client、runtime owner 与 GUI gateway，保持 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI` 单一产品链。
- 将文件、进程、工具、Skills、MCP、Plugin 和多 Agent 能力归回各自 current owner，减少重复入口和跨层状态复制。

### 测试与质量

- 新增文件系统、进程、后台终端、Agent runtime、Plugin package 和协议 v2 的 Rust、TypeScript、JSON-RPC 与真实 Electron 回归。
- 扩展 current Agent fixture、工具/Skills/MCP 场景、Gate B 打包验证、历史恢复和取消/重试路径证据。
- 同步 generated protocol schema/types、命令契约、legacy 回流守卫、脚本治理、五语言 GUI 回归与 README 定位文档。

### 文档

- 重写中英文根 README，明确 Lime 是类似 Claude Code、WorkBuddy、Codex 的全栈桌面 AI Agent，并保留原有产品图片。
- 更新 App Server 文件/进程边界、Agent runtime 主链、Plugin current owner、治理路线图和 v2 协议说明。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.124.0`。

**完整变更**: `v1.123.0` -> `v1.124.0`
