# 已安排任务领域与协议合同

状态：`target contract / field review required`

## 1. 领域对象

协议层使用短领域名 `ScheduledTask`；UI 使用“已安排任务”；存储迁移期可继续读取 `AutomationJob`，但不得长期双写。

```text
ScheduledTask 1 --- N ScheduledTaskRun
ScheduledTask --- 1 Schedule
ScheduledTask --- 1 ExecutionContextSnapshot
ScheduledTaskRun --- 1 Thread
ScheduledTaskRun --- 1 Turn
```

## 2. Task 合同

建议 v2 目标形状：

```json
{
  "id": "task_...",
  "title": "每日简报",
  "prompt": "...",
  "enabled": true,
  "schedule": {
    "type": "weekdays",
    "time": "08:00",
    "timezone": "Asia/Shanghai"
  },
  "execution": {
    "threadMode": "new_thread",
    "sourceThreadId": "thread_...",
    "projectId": "project_...",
    "cwd": "/absolute/current/project/path",
    "modelId": null,
    "reasoningEffort": "high",
    "approvalPolicy": "on_request",
    "sandboxPolicy": {}
  },
  "notificationPolicy": "all_runs",
  "overlapPolicy": "skip_if_running",
  "nextRunAt": "2026-08-13T00:00:00Z",
  "lastRunSummary": null,
  "createdAt": "...",
  "updatedAt": "..."
}
```

约束：

- `id` 由服务端生成；外部 Plugin 的 `key` 不能直接作为本地 identity。
- `title/prompt/time/timezone` 必须服务端校验。
- `cwd` 必须来自 project/platform path owner，不接受 renderer 任意字符串写入。
- `modelId` 使用 canonical stable id；`null` 表示运行时继承当前默认选择政策。
- snapshot 中不得写入 API key、token 或 inline secret。

## 3. Schedule 合同

```rust
enum ScheduledTaskSchedule {
    Hourly {
        interval_hours: u32,
        days: Option<Vec<Weekday>>,
        minute: u8,
        timezone: String,
    },
    Daily { time: String, timezone: String },
    Weekdays { time: String, timezone: String },
    Weekly { days: Vec<Weekday>, time: String, timezone: String },
}
```

Lime 在 Codex 可验证语义上增加 `timezone` 和 hourly `minute`，因为调度器必须确定绝对执行时刻。这是 Lime runtime 必需字段，不冒充 Codex 原字段。

## 4. Run 合同

```json
{
  "id": "run_...",
  "taskId": "task_...",
  "scheduledFor": "...",
  "startedAt": "...",
  "finishedAt": null,
  "status": "running",
  "sessionId": "session_...",
  "threadId": "thread_...",
  "turnId": "turn_...",
  "trigger": "schedule",
  "attempt": 1,
  "summary": null,
  "error": null
}
```

`trigger` 为 `schedule/manual/catch_up`。运行结果只投影恢复入口所需的 Session identity、canonical Thread/Turn identity 和摘要，不复制完整消息或工具事件。

## 5. JSON-RPC method 集

目标 method 使用领域名，不新增品牌前缀：

| Method                           | 作用                                          |
| -------------------------------- | --------------------------------------------- |
| `scheduledTask/list`             | 分页、搜索、按 enabled 状态筛选               |
| `scheduledTask/read`             | 读取详情与派生状态                            |
| `scheduledTask/create`           | 创建并返回 canonical task                     |
| `scheduledTask/update`           | 部分更新，带 revision/updatedAt 乐观并发      |
| `scheduledTask/delete`           | 软删除定义，保留运行历史                      |
| `scheduledTask/enabled/set`      | 暂停/恢复的窄命令                             |
| `scheduledTask/run/start`        | 立即运行；返回 Run + Thread/Turn identity     |
| `scheduledTask/run/list`         | 分页历史                                      |
| `scheduledTask/schedule/preview` | 返回后续至少 5 个时刻和 DST warning           |
| `scheduledTask/changed`          | task create/update/delete/status notification |
| `scheduledTask/run/updated`      | run terminal/attention notification           |

迁移原则：

- `automationJob/*` 与目标 method 不得形成双 owner。
- 在同一协议迁移变更集中替换 typed consumers，并删除旧 handler、`protocol/v0/automation.rs`、schema 和正向 fixture；不新增 compat handler。
- 页面、组件和 hook 只经 `src/lib/api/scheduledTasks.ts` typed gateway。
- Electron 只转发 App Server JSONL；系统通知单独走 Desktop Host capability。

## 6. 列表/详情 read model

列表响应不返回完整 prompt、sandbox 结构或大量历史：

```json
{
  "items": [
    {
      "id": "task_...",
      "title": "每日简报",
      "enabled": true,
      "attention": false,
      "scheduleSummary": {
        "type": "weekdays",
        "time": "08:00",
        "timezone": "Asia/Shanghai"
      },
      "nextRunAt": "...",
      "lastRun": {
        "status": "completed",
        "startedAt": "...",
        "threadId": "..."
      }
    }
  ],
  "nextCursor": null
}
```

详情响应才返回 prompt、execution snapshot、通知规则和最近运行。

## 7. 幂等与并发

- Scheduler claim 使用 task id + scheduled window 生成幂等键。
- 同一 scheduled window 最多产生一个 Run。
- `run/start` 每次手动运行产生新幂等键，但重复请求需支持 client request id 去重。
- 更新需检查 `revision` 或 `updatedAt`，避免两个窗口互相覆盖。
- 删除/暂停与 claim 竞争时：claim 事务后再读 enabled/revision；已暂停不得启动新 Turn。

## 8. 通知合同

- `scheduledTask/changed` 只触发列表/详情失效刷新，不携带完整 Task。
- `scheduledTask/run/updated` 携带 taskId、runId、status、threadId、turnId 和 attention flag。
- Thread/Turn/Item 的 streaming 仍使用现有事件，不复制到 scheduledTask namespace。
- OS 系统通知由 Renderer/Host 消费 terminal run projection 后请求 Desktop Host；App Server 不直接调用 Electron API。

## 9. Plugin scheduled task 边界

Codex Plugin detail 的 `scheduledTasks` 是模板/摘要来源，不自动成为本地任务，也不绕过用户确认。后续若接入：

```text
plugin/read scheduledTasks
  -> prefilled ScheduledTaskDraft
  -> user review/confirm
  -> scheduledTask/create
```

Plugin 更新不得静默修改已创建的本地任务。
