# 已安排任务当前基线

状态：`audited / implementation-baseline`

更新时间：2026-08-13

## 结论

Lime 已具备任务 CRUD、SQLite 持久化、日程计算、轮询调度、真实 Agent Turn 执行、失败治理、Agent Run 历史与 typed renderer gateway。缺口主要在产品合同和 owner 收敛：当前 UI 仍以“持续流程”为中心，暴露了过多运行/投递/兼容字段，Codex 风格日程与主从工作台尚未建立，对话内创建虽有 draft/projection 基础但未形成截图所示的完整闭环。

因此本任务是产品化重构和协议收敛，不是从零实现。

## Current owner 地图

| 责任               | 当前路径                                                                                                            | 当前事实                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Renderer 页面      | `src/components/automation/AutomationPage.tsx`                                                                      | 一级 `automation` 页面已存在，内部复用设置页工作台                          |
| Renderer 复杂 UI   | `src/components/settings-v2/system/automation/**`                                                                   | 已有列表、模板、创建/编辑、详情、健康面板、运行历史                         |
| Typed gateway      | `src/lib/api/automation.ts`                                                                                         | 所有业务调用经 `AppServerClient.request`，无页面裸 `invoke`                 |
| JSON-RPC protocol  | `lime-rs/crates/app-server-protocol/src/protocol/v0/automation.rs`                                                  | 已有 config/status/job CRUD/runNow/health/history/schedule preview/validate |
| App Server handler | `lime-rs/crates/app-server/src/processor/automation.rs`                                                             | 当前 method handler 与参数解析入口                                          |
| Domain/read model  | `lime-rs/crates/app-server/src/local_data_source/automation.rs`                                                     | CRUD、next run、运行开始/结束、Agent Run history                            |
| 执行编排           | `lime-rs/crates/app-server/src/automation_execution.rs`                                                             | 将任务投影到真实 Agent 执行链                                               |
| Scheduler          | `lime-rs/crates/scheduler/**`                                                                                       | SQLite DAO、到期查询、任务状态、失败/冷却治理                               |
| 对话内 draft       | `src/components/agent/chat/service-skills/automationDraft.ts`                                                       | 已有从 Agent UI payload 生成任务草稿的基础                                  |
| UI 回归            | `src/components/settings-v2/system/automation/*.test.tsx`、`scripts/electron/settings-automation-fixture-smoke.mjs` | 已有组件与 Electron fixture 证据入口                                        |

## 当前能力盘点

### 已有且应复用

- 创建、读取、更新、删除任务。
- 启用/停用、立即运行、next run 预览与校验。
- `every/cron/at` 日程表达。
- `agent_turn` payload，包含 prompt、thread lineage、web search、approval、sandbox 与 request metadata。
- 运行超时、重试、连续失败、自动冷却。
- 运行历史基于 `AgentRunDao::list_runs_by_source_ref("automation", id, limit)`。
- 多语言资源与现有 UI 测试基础。

### 已有但不应原样进入目标首屏

- 调度器轮询秒数、全局运行状态和健康统计。
- `execution_mode = intelligent/skill/log_only`。
- webhook/Telegram/本地文件/Google Sheets 投递细节。
- 原始 Cron、秒级 fixed interval、一次性 `at`。
- timeout/max retries 等高级治理项。

这些能力应进入高级设置、运行诊断或后续阶段，不能压过创建任务的主流程。

### 需要迁出或删除

- `browser_session` 旧 payload 的创建和执行入口。
- SceneApp retired context 与旧兼容说明。
- 设置页中重复的完整业务工作台。
- 任何 renderer 定时器直接触发任务的路径。

## 现状问题

1. 产品对象命名分裂：代码叫 automation/job，用户界面叫“持续流程”，目标叫“已安排任务”。
2. 任务目录与详情未形成稳定主从关系，用户进入后需要理解模板、健康、设置等多个平级区域。
3. 创建表单过度暴露底层字段，无法达到截图中“说明要做什么 + 选择在哪里/何时运行”的低门槛体验。
4. 日程模型与 Codex 可证实的 `hourly/daily/weekdays/weekly` 不一致。
5. 运行上下文当前强依赖已有 session/thread；“新聊天”语义需要产品合同化。
6. 任务摘要、执行实例和 Thread/Turn 的标识关系没有形成面向 GUI 的统一 read model。

## 外部参考边界

`/Users/coso/Documents/dev/rust/codex` 当前可直接验证：

- `ScheduledTaskSummary { key, name, prompt, schedule }`。
- `ScheduledTaskSchedule::{Hourly, Daily, Weekdays, Weekly}`。
- `Hourly` 支持 `intervalHours` 和可选 weekdays。
- `Weekly` 支持 weekdays；weekday 为 `MO..SU`。
- 这些字段出现在 Plugin detail 的 `scheduledTasks` 摘要中。

当前未在该仓库验证到独立桌面任务 CRUD、持久化、调度执行器或截图 UI 源码。因此 Lime 只对齐可证实的协议语义与截图行为，不复制不存在的内部实现假设。

## 基线风险

- 协议仍在 `v0/automation.rs`；实施时若升级为 v2，必须一次迁移所有消费者，不能长留双协议。
- `scheduler` 的类型命名和表结构可能携带旧 payload；协议冻结前需完成字段级数据审计。
- 设置页与一级页面共用同一超大组件，重构容易形成双 UI owner。
- 任务运行需要长期后台生命周期；Electron 退出/休眠/升级后的补跑政策尚未产品化。
