## Lime v1.141.0

### 新功能

- 新增独立 Code Mode protocol、runtime、host 与 session facade crate，统一承载 stdio/gRPC、V8 执行、typed content、取消、重连和 session lease 生命周期。
- CLI/TUI 继续扩展为统一 App Server 产品面：支持 Thread 恢复与管理、MCP/Skills/Plugin 查询、模型与权限控制、JSON/JSONL、prompt history、审批、request_user_input 和队列输入编辑。
- 新增远程 WebSocket session transport，支持 `ws/wss`、Bearer token、协议身份校验、ping/pong 与 fail-closed 认证策略；CLI/TUI 复用同一 session facade，不复制 runtime 或持久化。
- TUI 新增 Codex 对齐的 Markdown、Diff、语法高亮、OSC 8 链接、宽度感知表格、剪贴板/图片粘贴、slash command popup、static pager、transcript overlay 和 queued follow-up 预览。

### 修复

- 收紧 App Server `command/exec` 权限边界，拒绝客户端注入的授权字段并保持 typed lowering fail-closed。
- 修复 Code Mode host/client 的断线、重复 execution、stale cell、pending callback 和 session close 清理，避免旧代际状态泄漏到重连会话。
- 修复 CLI/TUI 终端生命周期、external editor、Unicode 光标、窄终端截断、队列恢复与失败终态投影边界。
- 修复文件系统 watch、Agent Runtime typed content、MCP 通知投影和工具生命周期摘要的回归断言。

### 优化与重构

- 将 Code Mode 旧 `tool-runtime` 内嵌 process/V8 实现物理迁出并删除，生产调用统一切换到四层 current crate；旧路径仅保留显式兼容导出。
- CLI Rust 目录与命令 owner 按 Codex 形状收敛，npm 根包/平台包 launcher、native payload、signal forwarding 与发布顺序统一到 `packages/cli`。
- TUI 交互、渲染和终端算法按 snapshot inventory 分层迁移，保持 canonical Thread/Turn/Item 为唯一会话事实源。
- 补充 CLI/TUI/Code Mode、远程 transport、Windows restricted execution、Electron release workflow 和治理边界守卫；Cloud 仅保留经认证 transport 的扩展点，不进入生产路径。

### 测试与质量

- 新增 Code Mode protocol/runtime/host/facade 生命周期、typed content、gRPC、重连与取消测试，并补齐 App Server 权限边界回归。
- 扩展 CLI Gate B、TUI Gate B、真实 stdio/PTY、npm packaged launcher、远程认证、snapshot inventory、终端渲染和跨平台发布守卫。
- 保持五语言文案、协议生成、脚本/CLI 边界、Electron Forge、版本一致性和 legacy 治理检查为发布门禁。

### 文档

- 更新架构、命令、治理、质量工作流与 CLI/TUI、Code Mode 执行计划，记录 Product Surface -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item 的唯一业务主链及 Cloud transport 边界。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.141.0`。
- 本次发布候选排除 `undefined/` 下本地 SQLite/WAL、runtime 数据库和 `.DS_Store` 运行产物；Code Mode 旧物理实现归 `dead/deleted`，不新增平行 runtime、compat 或 deprecated owner。

**完整变更**: `v1.140.0` -> `v1.141.0`
