## Lime v1.131.0

### 新功能

- 上线 Scheduled Tasks current 工作台，覆盖创建、编辑、暂停、立即运行、运行历史与对话回链。
- 新增 `scheduledTask/changed` 和 `scheduledTask/run/updated` typed notification，经 App Server、Renderer bridge 与 Electron Desktop Host 投影任务和运行终态。
- 支持运行中删除的软删除合同：保留 Agent Run 与 canonical Thread/Turn 历史，完成写回不会复活已删除任务。

### 修复

- 修复 Scheduled Task `new_thread` 在 `modelId=null` 时绕过 canonical 模型选择，导致真实 Runtime backend 缺少 provider/model selection 的问题。
- 修复立即运行失败后页面不刷新运行历史的问题；失败 Run、终态通知和错误提示现在会及时回显。
- 修复通知 Host 返回 `unsupported/failed` 时错误被伪装为成功的问题，并补齐任务状态刷新与删除确认语义。

### 优化与重构

- 将旧 Automation 管理面迁移到 Scheduled Tasks current 协议、唯一存储映射和 typed Renderer gateway，清理已脱离构建图的旧页面、fixture 与 API 双轨。
- 收敛调度 worker 的 manual/due/catch-up/missed/recovery 终态、重叠策略、one-shot CAS、DST 与启动恢复行为，保持 RuntimeCore/Thread/Turn/Agent Run 单一主链。
- 更新五语种 Scheduled Tasks 文案、协议 schema、生成客户端、治理目录和执行计划，补齐真实 Electron current fixture。

### 测试与质量

- 通过 App Server protocol/client contracts、Scheduled Tasks 定向 Vitest 与 Rust related/changed 验证。
- 通过 Scheduled Tasks Electron Gate B、Agent runtime current fixture、GUI smoke、治理报告、`verify:local`、格式检查与 diff 检查。

### 文档

- 更新 Scheduled Tasks 架构、命令边界、迁移账本、路线图与实现计划，记录通知、软删除和 provider route 的 current owner。
- 新增 v1.131.0 发布执行计划，明确平台证据与旧 Automation 物理清理的剩余边界。

### 其他

- 将根应用、CLI npm 包、Rust workspace 与 Cargo.lock 版本统一提升到 `1.131.0`。
- Windows Notification Center、真实 macOS/Windows sleep-resume、签名、公证和正式 release asset 证据仍需对应平台或 CI runner，不在本地候选中冒充完成。

**完整变更**: `v1.130.0` -> `v1.131.0`
