# 已安排任务当前基线

状态：`current / implementation-complete`

更新时间：2026-08-17

## 结论

Lime 已完成已安排任务的产品化重构与协议收敛。产品只保留 `scheduledTask/*` 公开协议、一级 Scheduled Tasks 工作台、
RuntimeCore 执行、canonical Thread/Turn/Item 与 Agent Run 历史。旧 Automation 页面、Settings 工作台、公开协议、typed client、
smoke 和 Agent UI projection 已物理删除并由负向守卫封口。

当前未闭环的只是真实 Windows Notification Center、Windows Gate B 与 macOS/Windows sleep-resume 平台证据，不是第二套实现缺口。

## Current owner 地图

| 责任 | current owner | 当前事实 |
| --- | --- | --- |
| Renderer 页面 | `src/components/scheduled-tasks/**` | 一级主从工作台，承接列表、筛选、创建/编辑、详情、启停、删除和运行历史 |
| Typed gateway | `src/lib/api/scheduledTasks.ts` | 所有业务调用经 `AppServerClient.request`，无页面裸 `invoke` 或 Automation fallback |
| JSON-RPC protocol | `lime-rs/crates/app-server-protocol/src/protocol/v0/scheduled_task.rs`、`protocol/v2/envelopes.rs` | 9 个 `scheduledTask/*` method 与 2 个 typed notification |
| App Server handler | `lime-rs/crates/app-server/src/processor/automation.rs` | Scheduled Task params、result、通知和 public JSON-RPC 入口 |
| Domain/read model | `lime-rs/crates/app-server/src/local_data_source/automation/**` | CRUD、next run、软删除、Agent Run history；内部 Automation 命名只表示存储 owner |
| 执行编排 | `lime-rs/crates/app-server/src/automation_execution.rs` | canonical model route、Thread/session 与 Turn submission |
| Scheduler | `lime-rs/crates/scheduler/**` | 原子 claim、catch-up/missed、DST、overlap、恢复和时钟回拨合同 |
| 持久化 | `automation_jobs`、`agent_runs` | current 唯一任务表映射与运行历史；不新增第二表或第二 read model |
| 对话内创建 | `useWorkspaceServiceSkillEntryActions.ts`、`ScheduledTaskDialog.tsx` | Service Skill 可创建 typed Scheduled Task，缺失 lineage 时使用 Scheduled Tasks 文案 |
| Gate B | `scripts/electron/scheduled-tasks-fixture-smoke.mjs` | 真实 Electron/preload/IPC/App Server/RuntimeCore/provider/canonical read model |

## 当前能力盘点

### current 能力

- 创建、读取、更新、删除任务。
- 启用/停用、立即运行、软删除、next run 预览与五语种 GUI。
- `hourly/daily/weekdays/weekly` 日程表达与 Codex weekday `MO..SU`。
- `new_thread` 的 canonical model route/preflight，以及 `continue_thread` 的 durable lineage 恢复。
- 原子 claim、24 小时 catch-up、超窗 missed、overlap、one-shot、DST 和启动恢复。
- typed terminal notification、`all_runs/failures/none` policy 与 Desktop Host 系统通知。
- Agent Run 历史、canonical Thread/Turn/Item read model 和打开同一对话。

### 已删除或禁止恢复

- `automationJob/*`、`automationSchedule/*`、`automationScheduler/*` 公开协议和 typed clients。
- `src/lib/api/automation.ts`、旧 Automation 一级页面、Settings 工作台和 2015 个五语种旧设置文案。
- browser session automation、SceneApp automation context、renderer timer 和生产 mock fallback。
- `automation_job_projection`、Agent UI `runtimeEntity=automation_job` 与 `background_teammate` surface。
- 旧 Automation 专项 smoke、fixture 和正向协议测试。

旧 method 字符串只允许存在于 contract、Scheduled Tasks Gate B 或治理扫描的负向回流守卫。

## 剩余证据缺口

1. 真实 Windows Notification Center 与 Windows packaged/Gate B 尚需 Windows runner。
2. macOS 与 Windows 的真实 sleep-resume/catch-up 仍需平台级证据；受控时钟和 Rust 回归不能替代 OS 事件。
3. Base Setup 与 Service Skill catalog 中的 `automation_job` binding family 是 `compat taxonomy`；当前仍有真实消费者，退出前需另立迁移计划，不能和已删除公开协议混为一谈。

## 外部参考边界

`/Users/coso/Documents/dev/rust/codex` 当前可直接验证：

- `ScheduledTaskSummary { key, name, prompt, schedule }`。
- `ScheduledTaskSchedule::{Hourly, Daily, Weekdays, Weekly}`。
- `Hourly` 支持 `intervalHours` 和可选 weekdays。
- `Weekly` 支持 weekdays；weekday 为 `MO..SU`。
- 这些字段出现在 Plugin detail 的 `scheduledTasks` 摘要中。

当前未在该仓库验证到独立桌面任务 CRUD、持久化、调度执行器或截图 UI 源码。因此 Lime 只对齐可证实的协议语义与截图行为，不复制不存在的内部实现假设。

## 当前边界

- Rust `AutomationJob` DAO、`automation_jobs` 表和 App Server 内部 automation module 是 Scheduled Tasks 的 current 存储 owner，不代表公开双轨。
- Scheduled Task 每次执行仍进入 RuntimeCore 并生成 canonical Agent Turn；Agent UI 不再维护独立 background teammate projection。
- 生产失败显式失败，不回退 mock；Electron 只负责 JSONL 转发和系统通知宿主能力。
