## Lime v1.140.0

### 新功能

- 新增 CLI/TUI 产品面：CLI 通过 stdio App Server 进入统一 RuntimeCore，TUI 提供真实 PTY、alternate screen、键盘输入和终端恢复能力。
- CLI `exec` 新增单行 JSONL envelope 与 shell completion，便于脚本集成并保持与统一命令树一致。
- 新增视频任务工具与媒体产物投影，统一任务生命周期、参数校验、执行结果和聊天预览。
- 新增 Desktop Host 诊断与外部 backend 能力，补齐 App Server client 会话、传输和跨进程状态观测。

### 修复

- 修复 Electron Desktop Host、App Server 和 Runtime 状态诊断中的边界与清理问题，避免会话或工具状态泄漏。
- 修复 Agent 聊天中的任务协议噪声、媒体预览、工具摘要、历史压缩和运行时路由断言漂移。
- 修复崩溃恢复面板、短窗口首页输入区及设置页运行时入口的用户可见行为。

### 优化与重构

- 将 CLI 从旧 `lime-cli-npm` 入口迁移到 `packages/cli`，Rust CLI/TUI crate 统一复用 App Server protocol、client 和 canonical Thread/Turn/Item 投影。
- external editor 在 Windows 上支持 PATH 中的 `.cmd/.bat` shim，并在启动编辑器前关闭临时文件句柄。
- 清理旧 CLI skill、工具文档和已退役入口，补充 CLI/TUI、Desktop Host、release candidate 与治理边界守卫。
- 强化 macOS native host、Windows Squirrel、打包资源、发布身份和 Gate B 证据脚本，发布流程继续以 Electron Forge 为唯一打包事实源。

### 测试与质量

- 新增 CLI/TUI 真实 stdio/PTY Gate B fixture、视频工具测试、Desktop Host 诊断测试和边界治理回归。
- 扩展 App Server client、命令合同、Electron release workflow、Windows packaged evidence 及 GUI 主路径回归覆盖。
- 统一五语言文案边界、脚本治理、文档边界和版本一致性检查。

### 文档

- 更新架构、命令、治理、质量工作流、Feature Map、工具目录和 CLI/TUI 执行计划，明确 Product Surface 到 App Server JSON-RPC 的唯一业务主链。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.140.0`。
- 本次发布候选排除 `undefined/data/*` 本地 SQLite/WAL 运行产物；不新增 compat 或 deprecated owner。

**完整变更**: `v1.139.0` -> `v1.140.0`
