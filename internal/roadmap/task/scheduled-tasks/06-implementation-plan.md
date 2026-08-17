# 已安排任务实施计划

状态：`implemented / platform-evidence-pending`

执行结果：ST-0、ST-1/2、ST-4、ST-5 已完成；ST-3 的 domain/runtime 合同已完成，真实 OS sleep-resume 证据待补；
ST-6 的旧公开 Automation 双轨已物理删除并补负向守卫，Windows Notification Center 与 Windows Gate B 待平台 runner。
实际进度与验证证据以 `internal/exec-plans/scheduled-tasks-implementation.md` 为准。以下分阶段内容保留为实施输入，
不再作为 current owner 或未完成清单。

## 1. 写集纪律

每个阶段开始前在 `internal/exec-plans/` 建立长任务计划，声明精确写集、脏热区、架构影响、迁移范围、验证和退出条件。协议、scheduler、Agent runtime 和主页面不能由不同并行修改同时重写同一事实源。

## 2. 分阶段计划

| 阶段 | 目标                   | 主要写集                                                      | 退出条件                                          |
| ---- | ---------------------- | ------------------------------------------------------------- | ------------------------------------------------- |
| ST-0 | 合同冻结与数据审计     | 路线图、exec plan、DB/协议 inventory                          | 字段、状态、迁移和删除口径获确认                  |
| ST-1 | Domain/schema 收敛     | scheduler、App Server domain、migration                       | 单一 ScheduledTask owner，旧行迁移可重复执行      |
| ST-2 | JSON-RPC/typed gateway | protocol/schema/client/catalog/tests                          | `scheduledTask/*` 成组通过 contracts，无裸 invoke |
| ST-3 | 真实运行闭环           | RuntimeCore、Thread/Turn metadata、Agent Run                  | due/manual/catch-up 都产出 canonical Thread/Turn  |
| ST-4 | 主从工作台             | sidebar/router/page/VM/i18n/tests                             | 截图核心列表/创建/详情/历史交互完成               |
| ST-5 | 对话内创建与通知       | Agent UI projection、draft confirm、Desktop Host notification | 自然语言 draft -> 用户确认 -> create；通知可用    |
| ST-6 | 迁移删除与 Gate B      | 旧 Settings/browser/SceneApp、guards、fixtures                | 双轨清零，macOS/Windows 与真实 Electron 门禁完成  |

## 3. ST-0：合同冻结

- 审计 task 表、Agent Run source_ref、现有旧 payload 数量和形状。
- 决定目标存储表是原表原位迁移还是新表单次迁移；禁止长期双表。
- 评审 schedule DST、catch-up、overlap、soft delete、thread policy。
- 更新 `internal/aiprompts/architecture.md` 并完成责任开发者架构图确认。

## 4. ST-1：Domain/schema

- 在既有 `scheduler`/App Server owner 中定义 ScheduledTask、Schedule、Run summary。
- 增加原子 claim/幂等窗口、重启 reconcile 和配置失效投影。
- 落地数据 migration 与负向测试。
- 单元测试覆盖 DST、休眠、时钟跳变、重复 claim、暂停竞争和迁移矩阵。

## 5. ST-2：协议

- 同步 app-server-protocol method/params/result/notification/schema。
- 同步 App Server dispatch/handler、Rust client、TS client/gateway、catalog 与测试 fixture。
- 页面不得直接依赖 protocol raw JSON。
- 在同一变更集中迁移所有 `automationJob/*` consumer，并删除旧 handler、schema 和正向 fixture；不建立短期委托或 compat method。

## 6. ST-3：运行闭环

- 手动与到期执行共用一个 start service。
- `new_thread/continue_thread` 都走标准 Thread/Turn 方法和 runtime policy。
- terminal Turn 驱动 Run terminal；审批、取消、超时、崩溃恢复有稳定状态。
- Agent Run 与 Task Run 通过稳定 id 关联，历史只读 canonical projection。

## 7. ST-4：GUI

- 抽纯 View Model：filter/group/schedule formatter/form request builder/status projection。
- 重写 `AutomationPage` 为独立工作台，拆出列表、详情、编辑器和运行历史组件。
- 侧边栏、路由、页面参数、最近任务恢复同步。
- 五语种 i18n 与稳定组件测试同步。
- 不继续向接近/超过 800 行的组件追加业务逻辑。

## 8. ST-5：Agent 创建/通知

- 定义 `scheduled_task_draft` Agent UI schema 与校验。
- 当前对话/新对话只创建 draft；明确确认后才调用 create。
- Draft 展示 title/prompt/schedule/timezone/context/permission/notification。
- 终态 run 按 notification policy 请求 Desktop Host 通知；无宿主能力时 fail visibly，不伪造成功。

## 9. ST-6：清理与交付

- 完成 [迁移账本](./05-migration-ledger.md)。
- 删除旧设置业务工作台、browser payload、SceneApp context、orphan i18n/tests。
- 加 retired guard，更新 commands/architecture/quality 事实源。
- 通过本地门禁和真实 Electron Gate B，补 Windows 证据。

## 10. 每阶段退出纪律

- 每阶段先跑受影响定向测试，再扩大。
- 任一阶段未满足退出条件不得标记 complete。
- mock-only、浏览器镜像或静态截图不能替代真实 App Server/RuntimeCore 证据。
- 任何兼容分支必须有 owner、删除日期、负向 guard 和可观测命中数。
