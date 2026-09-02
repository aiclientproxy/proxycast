# Task / Agent taxonomy 主链

## 这份文档回答什么

本文件定义 Lime 当前 `Task / Agent / Coordinator` 的唯一 taxonomy，主要回答：

- 哪些对象才算当前一等执行实体
- `agent turn`、`subagent turn`、`scheduled task`、`scheduler tick`、`agent run` 分别是什么关系
- Agent Run / scheduler / subagent / Scheduled Tasks 各自属于哪一层，而不是继续互相抢“主入口”
- 哪些旧文档、旧术语、旧路径只能当专项说明或兼容壳，不能再反向定义当前主链

它是 **长时执行与协作编排的 current 事实源**，不是执行追踪专项计划，也不是单个服务的实现说明。

## 什么时候先读

遇到以下任一情况时，先读本文件：

- 调整 App Server / RuntimeCore / `lime-rs/crates/agent` 子代理 spawn、send input、wait、resume、close 能力
- 调整已安排任务的创建、调度、执行、通知或运行历史
- 调整 `agent_runs`、执行状态聚合或 run 级读模型
- 调整 `scheduled_task_worker`、`lime-rs/crates/scheduler` 或任何后台到期触发逻辑
- 讨论“这是回合、子代理、已安排任务还是调度器”的边界归属

如果一个需求同时碰到“子代理 + 自动化”“调度 + 执行追踪”“会话回合 + 长时后台任务”里的两项以上，默认属于本主链。

如果这个需求还需要继续细分“内部服务任务、主对话任务、`service_models` 对应任务画像”，继续补读：

- `internal/roadmap/task/task-taxonomy.md`

## 固定 taxonomy

当前 Lime Agent runtime 只承认下面两类一等执行实体：

1. `agent turn`
   前台会话回合。统一走 `agentSession/turn/start -> RuntimeCore -> Query Loop` 主链。

2. `subagent turn`
   父会话派生出的 child session / teammate 回合。它是 `agent turn` 的协作变体，不是另一套执行引擎。

下面三类不是 Agent runtime 的一等执行实体：

- `scheduled task`
  是可持久化、可延时、可周期触发的产品 coordinator；每次执行必须进入 RuntimeCore 并生成 canonical Agent Turn。

- `scheduler tick`
  只是“发现到期任务并触发执行”的领域触发器，不单独代表一个任务分类。

- `agent run`
  是跨入口的执行摘要与生命周期记录，不是 coordinator。本层由 `agent_runs` 与 App Server read model 承载。

固定规则只有一句：

**新的模型执行只能落成 `agent turn` 或 `subagent turn`；需要延时或周期触发时由 Scheduled Task 协调，但运行仍回到 Agent Turn，不得新增平行 runtime taxonomy。**

补充迁移边界：旧 `agent_runtime_*`、`automation_*`、`execution_run_*` 这类命令名只允许作为 retired guard、历史 evidence、测试 fixture 或受控迁移面；`lime-rs/src/commands/**` 已删除，不是新的 taxonomy、coordinator 或 runtime 实现目录。新增长时执行、子代理、自动化或执行摘要能力应进入 RuntimeCore / services / App Server protocol，不得恢复旧 wrapper。

补充边界：

[Codex `/goal`](../research/codex-goal/README.md) 对应 Lime current `ThreadGoal`，它只能作为“目标推进控制层”理解，不能成为第四类执行实体：

1. 前台即时推进仍归 `agent turn`。
2. 协作拆分仍归 `subagent turn`。
3. durable 后台推进由 Scheduled Task 协调，实际模型执行仍归 `agent turn`。
4. ThreadGoal status、budget / usage / pause / resume 只控制 thread 是否继续，不单独定义新的 run source、queue、scheduler 或 evidence。

旧 `ManagedObjective` owner、criteria/audit/evidence 和 Automation Goal 已删除，禁止以第四类执行实体或 ThreadGoal compat 恢复。

## 固定心智模型

当前主链统一按下面这张图理解：

`Scheduled Task / foreground request -> agent turn / subagent turn -> agent_runs + Thread/Turn/Item read model`

这条主链意味着：

1. `agent turn` 是前台交互入口，主事实源仍然是 `query-loop.md`
2. `subagent turn` 是 child session 的协作入口，复用当前 agent runtime 与会话事实，不单独发明另一套 task 状态机
3. Scheduled Task 是唯一 durable 后台任务入口，可以触发 agent turn，但不定义 Agent UI `runtimeEntity` 或第二套 run 摘要
4. `agent_runs` 只记录“这次执行怎么开始、怎么结束、归因到哪里”，不负责调度、分工或 parent/child 编排
5. `scheduler tick` 只负责原子 claim 和触发 due task，不负责定义产品层 task taxonomy

## 代码入口地图

### 1. `agent turn`

- `internal/aiprompts/query-loop.md`
- App Server `agentSession/turn/start`
- App Server RuntimeCore / RuntimeBackend
- `lime-rs/crates/agent`

固定规则：

- 前台回合只有一条 Query Loop 主链
- 子代理回合如果进入模型执行，仍然复用这条主链
- 不允许为长时任务再造第二套“聊天执行入口”

### 2. `subagent turn`

- App Server / RuntimeCore 子代理 session projection
- `lime-rs/crates/agent` Team / subagent runtime support

当前这里负责：

1. spawn child session / teammate
2. 写入父子会话与 team membership
3. 将 child turn 放入后台执行
4. 处理 `send_input / wait / resume / close`
5. 发出 runtime stream/status 事件，维持父子状态投影

固定规则：

- `subagent turn` 当前不是新的 `RunSource`
- 在执行摘要层，它继续复用 `chat` 会话型 run 与 `session_id / parent-child context / evidence` 关联
- 需要新增子代理能力时，优先扩展这里，而不是绕去 scheduler 或自动化任务

### 3. `scheduled task`

- `src/lib/api/scheduledTasks.ts`
- App Server `scheduledTask/*`
- `lime-rs/crates/app-server/src/automation_execution.rs`
- `lime-rs/crates/app-server/src/scheduled_task_worker.rs`
- `lime-rs/crates/scheduler/*`

当前这里负责：

1. `automation_jobs` 的创建、更新、删除与启停
2. 原子 claim、到期执行、catch-up/missed、DST 和启动恢复
3. `scheduledTask/run/start` 手动触发
4. typed terminal notification 与 Agent Run 历史
5. RuntimeCore canonical Thread/session 与 Turn submission

固定规则：

- durable 后台任务统一走 Scheduled Tasks current owner
- 如果一个需求需要“稍后执行 / 周期执行 / 无前台会话也能继续跑”，默认先判断能否落到 `scheduledTask/*`
- Scheduled Task 可以触发 agent turn，但不允许维护第二份 Thread/Turn/Item 或 Agent UI runtime projection

### 4. `agent run`

- `lime-rs/crates/core/src/database/dao/agent_run.rs`
- App Server Scheduled Tasks read model

当前这里负责：

1. 为 `chat / skill / scheduled task` 记录统一生命周期摘要
2. 暴露 `agent_runs` 只读查询
3. 统一终态与错误归一化

固定规则：

- Agent Run 是观测层，不是 coordinator
- `source=chat` 覆盖前台与子代理会话型回合
- `source=skill` 代表独立 skill 执行摘要，不是新的 task taxonomy
- `source=automation` 是 Scheduled Tasks 的内部持久化 discriminator，不是公开 Automation 协议

### 5. `scheduler tick`

- `lime-rs/crates/scheduler/*`
- `lime-rs/crates/app-server/src/scheduled_task_worker.rs`

当前这里负责：

1. 查询并 claim `automation_jobs` 中的 due Scheduled Task
2. 发现 due task
3. 执行并标记完成 / 失败

固定规则：

- 它是 Scheduled Tasks current domain trigger，但不是 Agent runtime taxonomy
- claim、catch-up/missed、DST、overlap 与恢复逻辑归 scheduler owner
- 不允许在这里新增公开协议、GUI read model 或第二套编排模型

## current / compat / deprecated / dead

### `current`

- `internal/aiprompts/task-agent-taxonomy.md`
- `internal/aiprompts/query-loop.md`
- `lime-rs/crates/agent/src` 与 `lime-rs/crates/app-server/src` 承接协作 runtime；新逻辑按职责进入 RuntimeCore / services
- `src/lib/api/scheduledTasks.ts`
- `lime-rs/crates/app-server/src/automation_execution.rs`
- `lime-rs/crates/app-server/src/scheduled_task_worker.rs`
- `lime-rs/crates/scheduler/*`
- `agent_runs`
- `automation_jobs`

这些路径共同构成当前唯一 taxonomy：

- 前台执行看 `agent turn`
- 协作执行看 `subagent turn`
- 后台 durable 协调看 `scheduled task`
- 执行摘要看 Agent Run

### `compat`

- Base Setup 与 Service Skill catalog 的 `automation_job` binding family

该名称仍有真实 catalog consumer，只表达“可绑定到 durable schedule”的旧分类。它不得映射回旧公开 Automation 协议、
Agent UI `runtimeEntity=automation_job` 或 `background_teammate`；退出条件是 binding schema 与全部消费者一次迁到 Scheduled Task 领域名。

### `deprecated`

- Execution Tracker / heartbeat 旧 taxonomy（专项文档已删除，历史仅从 Git history 查阅）
- 任何新增的 `heartbeat_executions` 写路径或读取依赖
- 任何把 `heartbeat` 当成与 `chat / skill / automation` 并列 run source 的新设计
- 任何把 `scheduler tick / cron / 心跳任务` 当成独立 task taxonomy 的新设计

这些概念不再承担 current taxonomy 定义权，也不得恢复独立执行实体或公开协议。

### `dead`

- `automation_jobs.payload.browser_session`
- `automationJob/*`、`automationSchedule/*`、`automationScheduler/*`
- `src/lib/api/automation.ts` 与旧 Settings Automation 工作台
- Agent UI `automation_job_projection`、`runtimeEntity=automation_job` 与 `background_teammate`

这些旧 surface 只能迁移、删除或作为负向守卫引用，不能继续创建、更新或恢复为 current 能力。

## 最低验证要求

如果本轮改动涉及本主链，至少按边界选择最贴近的验证：

- 纯文档 / 分类回写：`npm run harness:doc-freshness`
- 改 `agent_runs`：相关定向 Rust 测试
- 改子代理 runtime：`subagent_runtime.rs` 或 `runtime_turn` 的定向测试
- 改 `scheduledTask/*` / 执行服务：App Server public JSON-RPC、Rust related 与 `test:contracts`
- 改 scheduler：scheduler related、Scheduled Tasks Gate B 与必要的 sleep/catch-up 证据

## 这一步如何服务主线

`M2` 的目标不是把所有长时执行代码一次性重写，而是先把 taxonomy 收成唯一事实源。

从现在开始：

- 解释前台协作执行时，回到 `agent turn / subagent turn`
- 解释后台 durable 协调时，回到 `scheduled task`
- 解释执行摘要时，回到 Agent Run
- 解释 scheduler 时，把它视为 Scheduled Tasks current trigger，而不是 Agent runtime taxonomy

这样后续的 `M3 Remote runtime`、`M4 Memory / Compaction`、`M5 State / History / Telemetry` 才不会继续被长时任务边界反复打断。
