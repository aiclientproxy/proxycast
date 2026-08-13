## Lime v1.127.0

### 新功能

- 新增已安排任务一级工作台，支持列表与详情、创建与编辑、启停、立即运行、日程预览和运行历史。
- 新增 9 个 `scheduledTask/*` App Server JSON-RPC method，并为 Renderer 提供生成的类型协议与 typed gateway。
- 调度运行复用 RuntimeCore 与 canonical Thread/Turn/Item 主链，支持新对话和继续来源对话两种模式。

### 修复

- 补齐原子 claim、24 小时窗口补跑、超窗 missed 记录、overlap 跳过和 one-shot 终态，避免重复领取或丢失运行历史。
- 修复纽约 DST 缺失/重复小时、手动运行日程锚点、时钟回拨与暂停任务立即运行的边界行为。
- 启动恢复会幂等终止遗留的 queued/running Agent Run，并在 canonical Turn 已有终态时复用真实结果收口。

### 优化与重构

- 已安排任务继续复用 `automation_jobs` 作为唯一任务表，运行历史统一投影自 `agent_runs`，不引入第二套持久化 owner。
- 收敛任务中心与侧边栏导航，Renderer 只保留筛选、选中项和编辑表单状态，任务事实由 App Server read model 提供。

### 测试与质量

- 增加 Scheduled Tasks 公共 JSON-RPC、scheduler claim/recovery、RuntimeCore lineage、read model 和失败关闭回归。
- 增加已安排任务工作台、typed gateway、导航与五语种资源的稳定测试。
- 发布候选已通过 TypeScript typecheck、协议合同、Rust related、70 项前端定向测试与真实 Electron GUI smoke。

### 文档

- 新增已安排任务的产品需求、交互、协议、运行时架构、迁移账本、验证合同和 Codex 对齐矩阵。
- 更新全局架构与命令边界，确认 Electron 只承接 JSONL 转发，调度、CRUD 和运行状态归 App Server owner。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.127.0`。
- 旧 `automationJob/*`、旧 Settings consumer、terminal notification、软删除与专项跨平台证据仍按执行计划继续收口，本版本不宣称路线图全部完成。

**完整变更**: `v1.126.0` -> `v1.127.0`
