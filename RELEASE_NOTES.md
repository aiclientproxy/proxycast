## Lime v1.134.0

### 新功能

- 扩展 Browser Workspace 的历史页面与同 Tab 状态投影，支持恢复、导航和可见状态的统一读取。
- 新增 App Server 环境管理、MCP 事件流和模型 Provider capability 查询的 v2 JSON-RPC 合同。
- 增加 Windows sandbox setup/status、Provider capability badges 和多语言 Browser runtime 矩阵入口。

### 修复

- 修复 Browser、Electron reverse request、App Server 与 RuntimeCore 在历史页面、下载、权限和窗口生命周期中的 identity 与终态清理。
- 修复 Agent Thread/Turn/Item 历史合并、恢复连接、流式事件和未完成回合的 GUI 投影一致性。
- 修复 MCP client/manager 生命周期、工具执行编排、执行环境和 Windows sandbox readiness 的边界行为。

### 优化与重构

- 将环境、MCP、Provider capability、sandbox 和 Browser historical surface 收敛到 current App Server JSON-RPC 与 RuntimeCore 产品链。
- 重构 App Server v2 schema、generated client、Agent session gateway、模型选择器和设置页状态投影，删除脱离构建图的旧 Playwright browser tool 入口与资源。
- 补充 Electron Gate B artifact、冷启动、locale matrix、环境和 Provider capability 的测试与证据路径。

### 测试与质量

- 扩展 App Server protocol contract、MCP exact JSON-RPC、Rust tool-runtime、Browser Workspace historical、模型 capability、Windows sandbox 和 Agent session 测试。
- 增加真实 Electron Browser Gate B、冷重启和多语言场景的验证入口；发布门禁覆盖版本一致性、类型检查、协议合同、相关 Rust/前端测试、GUI smoke、current fixture 和治理扫描。

### 文档

- 更新架构、Browser 路线图、Codex 对齐计划、桌面 smoke、脚本治理和协议 schema 事实源，记录 current/compat/deprecated/dead 边界。
- 新增 v1.134.0 发布执行计划，记录候选范围、验证结果和平台/打包证据边界。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.134.0`。
- Windows 安装器、签名、公证、正式 release assets 和 live provider 证据仍需对应平台或 CI runner，不在本地候选中冒充完成。

**完整变更**: `v1.133.0` -> `v1.134.0`
