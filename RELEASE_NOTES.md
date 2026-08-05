## Lime v1.121.0

### 新功能

- 推出 Plugin v2 current 主链：App Center 可从 bundled、仓库、个人与本地目录 catalog 发现 Codex-compatible Plugin，完成详情查看、安装确认、启停与卸载。
- 已安装 Plugin 通过不可变 activation snapshot 进入当前 Agent runtime，支持 Claw 结构化 `plugin://` mention、Skill/MCP 注入、MCP App Right Surface 与历史恢复。
- 新增受控的 Electron WebContentsView HTML 宿主与 bundled Browser Plugin 资产，并将 Plugin 资产纳入 Forge 打包链路。

### 修复

- 修复 Plugin MCP provider namespace 被误当作 canonical server identity 的问题，使资源读取、tool trace、Right Surface 与 reload 恢复使用同一稳定 identity。
- 修复 Skills watcher 在 throttle 窗口内丢失变更，改为首次立即通知并合并尾沿更新。
- 修复会话切换、初始导航与 Plugin Right Surface 的过期状态，并确保 Plugin 卸载后只保留历史证据，不重启 runtime 或重放 tool。
- 修复 canonical history 持久化对已物化快照二次 merge 导致的 reasoning delta 重复、重复片段丢失与 Codex 大历史导入退化，并为 SQLite 短暂锁竞争增加有界等待。
- 修复 Agent chat 在 App Server 终态事件后的同步边界，改为通过带 terminal reconcile 的会话详情刷新保持当前状态一致。

### 优化与重构

- 将 Plugin catalog、installed/enabled state、包校验与安装事务收口到 App Server current owner，Renderer 统一通过 typed JSON-RPC gateway 访问。
- 完善 MCP resource provenance、canonical Thread/Turn/Item 投影与 Agent Workspace 右侧承载，移除对 renderer 旧 registry 和 Plugin worker 的生产依赖。

### 测试与质量

- 新增 Plugin v2 App Center、typed gateway、mention/activation、MCP resource、Right Surface、历史恢复与卸载语义的 Rust、Vitest、contract 和真实 Electron Gate B 覆盖。
- 补充 Electron 打包资产、内嵌 Browser HTML、GUI smoke、Skills watcher 和 tool lifecycle 的回归守卫。
- 新增 canonical reasoning 线性持久化与 1,200 条命令历史导入性能回归，覆盖重复 delta、final 快照替换和后台进度完成。
- 增加 macOS 正式签名嵌套资源的 timestamp 关闭策略与时间戳服务不可用重试守卫。

### 文档

- 新增 Plugin v2 产品合同、安装模型、命令边界、App Center/Claw Surface、迁移清理与验收计划，并同步 current 架构图。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与锁文件版本统一提升到 `1.121.0`。

**完整变更**: `v1.120.1` -> `v1.121.0`
