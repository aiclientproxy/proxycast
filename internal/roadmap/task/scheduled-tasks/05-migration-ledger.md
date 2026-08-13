# 已安排任务迁移与清理账本

状态：`target ledger / no deletion executed`

## 1. 原则

仓库无外部用户与历史兼容负担。目标合同落地后直接迁移调用并删除旧入口，不保留“已安排任务”和“持续流程”两套产品，也不建立 compat owner。若存在本地持久化记录，只允许启动期一次性数据迁移；迁移完成后旧 schema、协议、读写和 UI 同轮删除。

## 2. 账本

| 当前项                          | 当前分类                    | 目标分类                            | 动作                                                 | 退出条件                                            |
| ------------------------------- | --------------------------- | ----------------------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| `automationJob/*` JSON-RPC      | `current`                   | `dead/deleted/forbidden-to-restore` | 同一变更集迁到 `scheduledTask/*`，不保留委托 handler | 所有生产 consumer 和 fixture 迁完，负向守卫禁止回流 |
| `automationSchedule/*`          | `current`                   | `migrated/deleted`                  | 迁为 `scheduledTask/schedule/preview`                | schema/client/GUI 同步完成                          |
| `automationScheduler/*`         | `current settings`          | `internal/diagnostic`               | 不进入任务主页面；保留必要系统诊断                   | 任务运行不依赖设置页打开                            |
| `AutomationJob` store rows      | `current`                   | `migrated`                          | 一次性 schema/data migration 到目标 Task             | migration idempotent，冷启动读写仅一份表/owner      |
| `TaskSchedule::Every/Cron/At`   | `current`                   | `dead/deleted/forbidden-to-restore` | 启动期一次迁移；不可映射记录默认暂停，不继续执行     | 迁移后旧类型读写与 schema 为 0，guard 覆盖          |
| `agent_turn` payload            | `current`                   | `current normalized`                | 收敛为 execution snapshot + Thread policy            | task/run/thread/turn identity 贯穿                  |
| `browser_session` payload       | `deprecated`                | `dead/deleted`                      | 禁止创建和执行，删除 GUI/protocol/runtime 分支       | 生产引用为 0，旧行有明确删除/失效策略               |
| SceneApp legacy context         | `deprecated`                | `dead/deleted`                      | 删除 projection、文案和测试正向路径                  | 只允许历史 evidence/negative guard                  |
| 设置页完整 Automation workspace | `current duplicate surface` | `deprecated/deleted`                | 业务功能迁到一级页面                                 | 设置页只剩必要系统级项，不重复 CRUD                 |
| `AutomationPage` 复用设置组件   | `current`                   | `rewritten current`                 | 改为独立主从工作台 owner                             | 新页面不再依赖 Settings 业务大组件                  |
| automation draft/projection     | `current partial`           | `current normalized`                | 对齐 `scheduled_task_draft` 和确认创建               | Agent 不可未经确认直接创建                          |
| 旧 i18n `settings.automation.*` | `current`                   | `migrated/deleted`                  | 新增 `scheduledTasks.*`，迁完删除旧正向 key          | 五语种完整且无 orphan key                           |
| 旧 Electron automation fixture  | `current`                   | `replaced`                          | 更新为真实 scheduled task Gate B                     | 证明 app-server/turn/read model 而非旧 UI           |

## 3. 数据迁移规则

- `cron` 可解析为 daily/weekdays/weekly 时自动转换，并保留原 timezone。
- `every` 只有整小时且 1-24 小时范围时转换为 hourly；秒/分钟间隔标记 `needs_attention`，默认暂停。
- `at` 一次性任务不进入 P0 目标合同；未来时间记录标记需处理，过去记录归档。
- `browser_session` 记录一律暂停并标记 unsupported；若仓库确认无用户数据，可直接删除，但实施前需数据审计证据。
- `delivery` 非 `none` 的任务迁移时保留只读摘要并暂停，直到独立投递需求完成；不能静默丢投递语义继续运行。
- migration 必须幂等、可在事务失败后重试，并记录 schema version；迁完切换新表/新字段并删除旧读写，不双读双写。

## 4. 回流守卫

最终 guard 至少禁止：

- Renderer/业务文档重新出现 `browser_session` 自动化创建路径。
- `settings.automation` 继续作为任务产品主入口。
- 生产代码写入 `TaskSchedule::Every/Cron/At`。
- 页面/Hook 直接调用裸 invoke 或 renderer timer 触发任务。
- Electron main 注册任务 CRUD 或 scheduler 业务 handler。
- 生产 mock 返回 scheduled task 成功。

## 5. 删除顺序

```text
冻结 ScheduledTask contract
  -> 建立同一 domain owner 与 migration
  -> 迁移 typed client / GUI / Agent draft
  -> 迁移 fixture 与 Gate B
  -> 清零旧 protocol/Settings/browser/SceneApp 正向引用
  -> 物理删除旧分支和孤立文案
  -> contracts + governance + cold-start evidence
```

任何失败只能修 current owner，不能恢复旧 UI 或旧 handler 作 fallback。
