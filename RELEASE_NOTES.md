## Lime v1.120.0

### 新功能

- Codex 历史导入接入 canonical provenance 与安全 metadata，历史消息、图片、命令执行、文件变更和动态工具记录可以在实时、完成态与冷恢复之间保持统一身份。
- Task Center 新增紧凑的位置与环境菜单，展示本地项目、Git 变更摘要和当前分支，并支持搜索、切换或创建分支、创建工作树与打开 Codex web。
- 对话时间线新增回合级过程摘要与文件产物分类，已结束回合合并为单条可读摘要，文件读取类历史产物可按只读语义展示。

### 修复

- 修复 Codex 导入同时产生 `response_item` 与 `event_msg` 时重复显示同一条用户消息的问题，并保留重复记录中更完整的文本和图片数据。
- 修复历史窗口、分页、重新加载和回合完成刷新时 canonical Thread/Turn/Item 投影不一致、工具明细重复或历史预览缺失的问题。
- 修复图片附件、文件变更摘要和导入历史中的来源标记在不同渲染路径下丢失的问题。

### 优化与重构

- 将导入来源标记以受控 typed metadata 贯通 App Server、协议 schema、生成客户端、read model 与 Renderer，禁止原始事件载荷进入产品链。
- 统一消息正文、助手气泡、回合摘要和输入栏的 Codex-style 阅读列宽度，拆分 Task Center 位置/环境面板并复用既有 Git gateway。
- 补齐历史导入、canonical projection、时间线聚合、图片附件、文件产物和环境菜单的单元、组件与集成回归。

### 测试与质量

- 扩展 Codex import click-through、current runtime fixture、App Server projection、协议 schema、Rust owner 与 Renderer 测试，覆盖实时流、历史恢复、分页和 reload。
- 五语言资源同步更新，新增稳定 DOM 与环境菜单交互断言。

### 文档

- 更新 Codex GUI 对齐、会话兼容重构、Trace 布局与 Refactor v1/v2 进度记录，明确 current owner、验证证据与未完成边界。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与锁文件版本统一提升到 `1.120.0`。

**完整变更**: `v1.119.0` -> `v1.120.0`
