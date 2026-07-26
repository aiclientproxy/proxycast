## Lime v1.112.0

### 新功能

- 扩展 App Server v2 Thread 管理能力，新增压缩、命名、已加载列表、搜索、metadata 更新、取消订阅以及状态/关闭通知。
- 新增 Thread 后台终端列表、清理与终止协议，并投影模型 safety buffering 状态更新。
- 完善 Thread fork 与恢复流程，支持压缩历史、中途回合 fork、直接输入策略和 canonical Thread/Turn/Item 投影。
- 建立 v2 模型与 Provider 能力协议，覆盖输入模态、推理档位、服务层级、可用性提示、升级信息和动态 capability 读取。

### 修复

- 修复 Thread resume/replay/fork 场景中的 identity、历史回放、状态通知和 read model 同步偏差。
- 修复模型路由、Provider capability lowering、健康检查与有效配置投影不一致的问题。
- 修复缺少展示字段的 service tier 在 App Server 模型投影中被静默丢弃的问题。
- 修复历史代码任务恢复后残留 optimistic Turn、重复展示完整工具过程的问题，默认只保留紧凑过程摘要与最新终态。
- 修复对话滚动、布局过渡、侧边栏会话操作和 Right Surface 状态同步中的可见回归。

### 优化与重构

- 收敛 App Server protocol、Rust client、TypeScript client、schema 与 generated types，移除已被 v2 owner 替代的旧会话和模型入口。
- 统一 Agent Runtime 的 tool options、runtime options、queued intent、会话配置和 unified exec 生命周期。
- 收敛文章工作区 canonical projection 与写回边界，删除重复的 Renderer selection writeback 路径。

### 测试与质量

- 新增 Thread compact、fork、search、metadata、unsubscribe、closed notification、model list 与 Provider capability 的公共 JSON-RPC 回归测试。
- 扩充 App Server client、Agent Runtime current fixture、Electron Gate B、模型选择器、消息滚动和 Workspace 布局测试。
- 建立前端与 Runtime contract 测试快照、source-to-scenario 映射及 CLI/TUI 测试资产清单。

### 文档

- 更新 Codex 方法到 Lime 产品面的对齐矩阵、Agent Runtime 协调计划、memory/query loop 规则和测试体系说明。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与锁文件版本统一提升到 `1.112.0`。

**完整变更**: `v1.111.0` -> `v1.112.0`
