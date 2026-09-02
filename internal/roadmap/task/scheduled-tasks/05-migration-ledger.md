# 已安排任务迁移与清理账本

状态：`executed / public-dual-track-deleted`

## 1. 原则

仓库无外部用户与历史兼容负担。目标合同落地后直接迁移调用并删除旧入口，不保留“已安排任务”和“持续流程”两套产品，也不建立 compat owner。若存在本地持久化记录，只允许启动期一次性数据迁移；迁移完成后旧 schema、协议、读写和 UI 同轮删除。

## 2. 账本

| 项目 | 最终分类 | 已执行动作 | 当前退出条件 |
| --- | --- | --- | --- |
| `automationJob/*` JSON-RPC | `dead / deleted / forbidden-to-restore` | protocol/schema/client/handler/consumer/fixture 已删除 | 负向 contract guard 禁止回流 |
| `automationSchedule/*` | `dead / deleted / forbidden-to-restore` | 公开预览迁到 `scheduledTask/schedule/preview` | 负向 contract guard 禁止回流 |
| `automationScheduler/*` | `dead / deleted / forbidden-to-restore` | Settings 诊断与公开 method 已删除 | scheduler 只由 current worker 内部驱动 |
| `AutomationJob` / `automation_jobs` | `current internal storage mapping` | 继续承载 Scheduled Task 唯一持久化，不新增第二表 | 不得重新暴露旧公开协议或产品命名 |
| `TaskSchedule::Every/Cron/At` | `current internal lowering` | `ScheduledTaskSchedule` 在 App Server 边界统一 lower/raise | 不得出现在 Renderer 或公开 Scheduled Task wire |
| `agent_turn` payload | `current normalized` | 收敛为 execution snapshot + Thread policy | task/run/thread/turn identity 贯穿 |
| `browser_session` Scheduled Task payload | `dead / rejected` | 创建边界不再生成，执行边界 fail closed | 与 current `browserSession/*` 独立能力不得混淆 |
| 旧应用编排 context | `dead / deleted` | projection、文案和测试正向路径已删除 | 只允许 negative guard |
| 设置页完整 Automation workspace | `dead / deleted` | 业务功能迁到一级 Scheduled Tasks 页面 | 不得恢复重复 CRUD |
| 旧 `AutomationPage` | `dead / deleted` | 替换为 `src/components/scheduled-tasks/**` | 一级导航只进入 current 工作台 |
| automation draft / Agent UI projection | `split` | Service Skill 创建迁到 typed Scheduled Task；`automation_job_projection` 删除 | 不得重建 `background_teammate` 平行 read model |
| 旧 i18n `settings.automation.*` | `dead / deleted` | 五语种 2015 个 key 删除，current 文案进入 `scheduledTasks.json` | i18n 负向守卫与 100% coverage |
| 旧 Electron automation fixture | `dead / deleted` | 替换为 Scheduled Tasks 真实 Electron Gate B | old method/mock hit 必须为 0 |

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
- Renderer 或公开协议写入 `TaskSchedule::Every/Cron/At`；这些类型只允许在内部 storage lowering 使用。
- 页面/Hook 直接调用裸 invoke 或 renderer timer 触发任务。
- Electron main 注册任务 CRUD 或 scheduler 业务 handler。
- 生产 mock 返回 scheduled task 成功。

## 5. 删除顺序

```text
冻结 ScheduledTask contract
  -> 建立同一 domain owner 与 migration
  -> 迁移 typed client / GUI / Agent draft
  -> 迁移 fixture 与 Gate B
  -> 清零旧 protocol、Settings、browser 与应用编排正向引用
  -> 物理删除旧分支和孤立文案
  -> contracts + governance + cold-start evidence
```

任何失败只能修 current owner，不能恢复旧 UI 或旧 handler 作 fallback。
