## Lime v1.125.0

### 新功能

- 对齐 Agent Plugins v1.0.0 标准包：根 `plugin.json`、直接子目录 Skills 和根 `mcp.json`。
- 新增 Codex Apps extension adapter、标准 Apps JSON catalog 与 `app/list`、`app/read`、`app/installed` 主链。
- 补齐 App Server `command/exec` 与 `review/start` 的 typed JSON-RPC、事件投影和 GUI 接入。

### 修复

- 修复 Thread/Turn/Item、文件、进程、后台终端、审核和 Agent 状态在 App Server、Electron 与 GUI 之间的投影漂移。
- 修复 MCP Plugin placeholder、路径 containment、HTTP header 过滤、失败隔离和 `PLUGIN_DATA` 持久化行为。
- 修复协议 schema、生成客户端、模型能力与 provider lowering 的一致性问题。

### 优化与重构

- 物理删除旧 Plugin package、worker、manager、renderer runtime、旧 v0 filesystem/process/plugin wire 和 detached facade。
- 将业务能力收敛到 `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI` 单一产品链。
- 清理旧 Plugin Lab/sidebar 文案、旧技术标准文档和无引用治理 surface，禁止旧路径回流。

### 测试与质量

- 通过协议 contracts、Rust related tests、治理扫描、文档边界、Agent fixture、GUI smoke 和 macOS Electron Gate B。
- 增加 Windows runner 的环境变量、UNC/extended path、junction/reparse、数据持久化与 Squirrel Gate B 验证入口。
- 发布门禁持续跟踪 Windows runner artifact，未用 macOS 或旧 Windows 证据替代。

### 文档

- 更新架构、命令边界、Plugin v3 合同、Codex parity matrix、清理账本和发布流程文档。
- 根 README 采用英文 canonical 入口，并保留中文独立页面。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.125.0`。

**完整变更**: `v1.124.0` -> `v1.125.0`
