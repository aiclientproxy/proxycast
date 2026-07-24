## Lime v1.111.0

### 新功能

- 新增 App Server v2 `artifact/write`，将工作区文档保存收敛到 typed JSON-RPC、canonical artifact snapshot 与可校验 sidecar evidence。
- 新增命令输出、文件补丁、Plan delta 与 MCP progress 的 v2 直接通知，并打通 Rust/TypeScript client、Renderer 投影和 GUI 更新。
- 扩展 token usage/read model，记录 cache write input tokens，并保持 Goal 用量、历史恢复与前端统计一致。

### 修复

- 修复 thread resume/replay 后 Turn 内 Item 不同步、历史本地合并误配、topic 切换与侧边栏导航状态漂移。
- 修复直接通知与 raw side-channel 共存时的 sequence gate 互相污染，避免流式事件误丢、重复或错误终止。
- 修复工具审批、MCP snapshot、provider turn 与终态投影在恢复和多回合执行中的生命周期不一致。
- 对齐 Codex 的 `AGENTS.override.md` 覆盖语义并保留符号链接工作目录身份，避免加载过时或错误作用域的指令。

### 优化与重构

- 删除旧 `agentSession/runtimeEvents/append` 协议与 schema、legacy plugin runtime gateway、旧 prompt builder 和旧权限实现，封住 parallel owner 回流。
- 将 App Server v2 notification projector 拆分到 command、file change、MCP 与 Plan 领域 owner，并同步 method catalog、schema 与 generated client。
- 统一 MCP tool result 的 content、structured content、错误和 sidecar metadata 投影，减少 provider history 与 GUI 的兼容解释。

### 测试与质量

- 新增 artifact write/replay JSON-RPC、v2 直接通知、thread history、approval、MCP 与 cache usage 的 Rust/TypeScript 回归。
- 扩展 App Server client、Agent current fixture、plugin/workspace GUI smoke 与 forbidden-to-restore 治理守卫。

### 文档

- 更新 runtime instructions、tool owner、存储对齐、协议收敛与 Refactor v1 执行记录。

### 其他

- 版本事实源更新到 `1.111.0`：根应用、CLI npm package、Rust workspace、`lime-rs/Cargo.lock` 和 release notes。

**完整变更**: `v1.110.0` -> `v1.111.0`
