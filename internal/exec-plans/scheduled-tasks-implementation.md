# 已安排任务实现计划

状态：`in-progress / protocol-runtime-gui-slice`

更新时间：2026-08-13

## 主目标

完成 `internal/roadmap/task/scheduled-tasks/` 定义的已安排任务产品，收敛现有 Automation UI、协议、调度、运行和历史到唯一 current 主链。

## 当前阶段与下一刀

- 当前阶段：ST-1/2/3/4 的首个垂直切片已落盘。9 个 `scheduledTask/*` method、唯一 `automation_jobs` 存储映射、手动运行到 RuntimeCore、typed Renderer gateway、主从工作台、一级导航和五语种资源已建立。protocol enum、schema fixtures 和 generated TypeScript client 已重建并通过 drift 检查。current read model 只接受显式 Scheduled Task marker，列表/详情已投影最近 Agent Run；`continue_thread` 在运行时通过 canonical Thread 恢复真实 session identity，不再持久化或拼接伪 session。P0 catch-up 已完成：最近 24 小时窗口最多折叠补跑一次并保留 skipped window metadata，超窗窗口写入 `missed` Agent Run 并推进 `next_run_at`，不创建 Thread/Turn。ST-3 本轮补齐了 overlap `skip_if_running` missed 历史、one-shot `NULL next_run_at` CAS、纽约 DST 缺失/重复小时、手动运行日程锚点保留和暂停任务立即运行；并新增启动前遗留 ownership 恢复：active queued/running Run 幂等标记 `scheduled_run_interrupted`，canonical Turn 终态可收口 Task，保留 claim 推进的 `next_run_at`，时钟回拨不重复 claim。
- 下一刀：接 terminal notification、软删除/运行中删除合同，再物理删除 `automationJob/*` 双轨和旧 Settings consumer；随后补 Agent current fixture、GUI smoke、Electron Gate B 与 Windows 平台证据。

## 窄写集

- `internal/roadmap/task/scheduled-tasks/**`
- `internal/exec-plans/scheduled-tasks-implementation.md`
- `.gitignore`
- `lime-rs/crates/app-server-protocol/**`
- `lime-rs/crates/app-server/**` 中 automation/scheduled-task owner
- `lime-rs/crates/scheduler/**`
- `packages/app-server-client/**`
- `src/lib/api/automation.ts`、目标 `src/lib/api/scheduledTasks.ts`
- `src/components/scheduled-tasks/**`
- `src/components/automation/AutomationPage.tsx`
- `src/lib/navigation/sidebarNav.ts`、侧边栏相关测试
- 五语种 navigation / scheduled tasks 文案
- 对话内 automation draft/projection 的目标迁移文件

## 热区与避让

- 开始时工作树只有本任务上一阶段的 roadmap 与 `.gitignore` 改动。
- 不覆盖无关模块，不提交、不推送、不创建分支。
- 任何工作中出现的外部改动按 `parallel-agent-collaboration.md` 重新审计。

## 分类

- `current`：`scheduledTask/* -> App Server -> scheduler -> RuntimeCore -> Thread/Turn/Item -> Agent Run -> GUI`。
- `deprecated`：旧 Automation 页面、`automationJob/*`、`TaskSchedule::{Every,Cron,At}`、旧设置业务工作台。
- `dead target`：`browser_session` 自动任务、SceneApp automation context、生产 mock fallback、renderer timer。
- 不建立长期 `compat` owner；存量只允许一次性迁移。

## 阶段

| 阶段 | 状态 | 退出条件 |
|---|---|---|
| ST-0 合同与审计 | `complete` | 需求、owner、迁移和验证合同已落盘，现有主链已盘点 |
| ST-1/2 协议与领域 | `in-progress` | 9 个 method、唯一表映射、协议 enum/schema/generated client 已落盘；旧 method 删除和旧 consumer 清理尚未完成 |
| ST-3 运行闭环 | `in-progress` | manual run 与在线 Runtime backend 的 due run 均复用 RuntimeCore/Thread/Turn/Agent Run；原子 claim、同一 run id、启动前复核、启动失败终态、canonical lineage、24 小时 catch-up、超窗 missed、overlap missed、one-shot CAS、DST、手动/暂停运行、启动恢复、canonical terminal 收口与时钟回拨合同已通过定向回归；真实 OS sleep/wake 事件和跨平台证据尚未完成 |
| ST-4 GUI | `in-progress` | 一级入口、主从工作台、创建/编辑/暂停/运行历史和五语种已落盘；GUI smoke/Gate B 尚未完成 |
| ST-5 对话与通知 | `pending` | draft 确认创建和 Desktop Host 通知完成 |
| ST-6 清理与验证 | `pending` | 旧路清零，Rust/contracts/fixture/GUI/Gate B 通过 |

## 已执行验证

```text
cargo test -p app-server-protocol scheduled_task -- --nocapture
  2 passed

npx vitest run src/components/scheduled-tasks/ScheduledTasksPage.test.tsx \
  src/components/scheduled-tasks/scheduledTaskViewModel.unit.test.ts \
  src/lib/api/scheduledTasks.test.ts
  11 passed

npx vitest run src/lib/navigation/sidebarNav.test.ts \
  src/components/AppSidebar.preferences.test.tsx \
  src/components/scheduled-tasks/scheduledTaskViewModel.unit.test.ts \
  src/lib/api/scheduledTasks.test.ts
  23 passed

npx eslint <scheduled tasks / navigation 定向文件>
  passed

npm run detect-translations -- --format json
  5 locales / scheduledTasks 155 keys / no issues
```

新增 `lime-rs/crates/app-server/tests/scheduled_tasks_jsonrpc.rs`，覆盖 public JSON-RPC preview、CRUD、启停、手动运行、真实 lineage identity、运行历史和最近运行 read model。

2026-08-13 边界收口验证：

```text
cargo test -p app-server scheduled_task --lib -- --nocapture
  8 passed

cargo test -p app-server --test scheduled_tasks_jsonrpc -- --nocapture
  2 passed

cargo check -p app-server --lib
  passed
```

2026-08-13 scheduler worker 验证：

```text
cargo test -p lime-scheduler claim -- --nocapture
  5 passed

cargo test -p app-server --lib automation_execution -- --nocapture
  6 passed

cargo test -p app-server --lib scheduled_task_worker -- --nocapture
  3 passed

cargo test -p app-server --test scheduled_tasks_jsonrpc -- --nocapture
  2 passed

cargo check -p app-server --lib --bin app-server
  passed

cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check
git diff --check
  passed
```

2026-08-13 catch-up/missed 验证：

```text
cargo test -p lime-core database::dao::agent_run -- --nocapture
  6 passed

cargo test -p app-server --lib scheduled_task_read_models -- --nocapture
  2 passed（ptrcomp archive）

cargo test -p lime-scheduler claim -- --nocapture
  5 passed（ptrcomp archive）

cargo test -p app-server --lib automation_execution -- --nocapture
  6 passed（ptrcomp archive）

cargo test -p app-server --lib scheduled_task_worker -- --nocapture
  6 passed（ptrcomp archive）

npx vitest run src/components/scheduled-tasks/ScheduledTasksPage.test.tsx \
  src/components/scheduled-tasks/scheduledTaskViewModel.unit.test.ts \
  src/lib/api/scheduledTasks.test.ts
  11 passed

npm run detect-translations -- --format json
  5 locales / scheduledTasks 156 keys / no issues

cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check
git diff --check
  passed
```

worker 只在 `AppServerBackendMode::Runtime` 启动；每轮扫描 `automation_jobs`，按 Scheduled Task marker 过滤后通过 scheduler 原子 claim，复用 claim 创建的 `run_id`，在启动前复核 task revision/enabled/window/ownership，并用真实 RuntimeCore Turn 驱动终态。旧 `scheduled_tasks` 表未被读取或双写。

2026-08-13 ST-3 overlap/DST/manual 验证（使用已发布 `ptrcomp` Apple ARM archive 与对应 binding）：

```text
cargo test -p lime-scheduler claim -- --nocapture
  7 passed

cargo test -p app-server --lib automation_execution -- --nocapture
  8 passed

cargo test -p app-server --lib scheduled_task_worker -- --nocapture
  9 passed

cargo test -p app-server --test scheduled_tasks_jsonrpc -- --nocapture
  3 passed

cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check
git diff --check
  passed
```

覆盖证据：overlap 扫描写入 `scheduled_run_overlap` missed 历史且不释放当前 running ownership；terminal 写回不覆盖已推进的下一窗口；one-shot 的 `NULL next_run_at` 仍可原子 claim/start/finish；America/New_York 春季缺失小时顺延到第一合法时刻、秋季重复小时只执行一次；暂停任务允许立即运行，启用任务手动运行不漂移 next-run 锚点。`trigger` 按合同区分 `schedule`、`catch_up`、`manual`。

2026-08-13 启动恢复/时钟跳变验证（使用同一 `ptrcomp` archive）：

```text
cargo test -p lime-scheduler claim -- --nocapture
  11 passed

cargo test -p app-server --lib scheduled_task_worker -- --nocapture
  13 passed

cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check
git diff --check
  passed
```

覆盖证据：worker 启动首轮只处理带 Scheduled Task marker 的遗留 ownership，旧 Automation 的 running 状态不被改写；queued/running Run 原子终止为 `error / scheduled_run_interrupted` 并清除 Task ownership，普通 claim 已推进的 `next_run_at` 保留，运行中编辑则按最新启用状态和 schedule 重算；canonical Thread/Turn 已有终态时复用真实终态收口，不伪造 interrupted；重复启动恢复幂等；时钟回拨后同一 deterministic run id 仅保留一次。

上述 3 条 App Server 命令在本机使用上游 v150.4.0 已发布的 `ptrcomp` Apple ARM archive 与对应 binding 显式覆盖，仅用于源码编译/定向测试；仓库默认启用的 `ptrcomp_sandbox` 归档未发布，默认命令仍按下述阻塞记录。测试证明：旧/非法 Automation 行不进入 Scheduled Task read model 且 read/update/delete/run fail closed；列表/详情投影最近运行与失败 attention；`continue_thread` 使用 canonical session/thread identity。

## 后续验证

```bash
npm run test:contracts
npm run test:rust:related -- <changed-rust-paths>
npm run test:related -- <changed-frontend-files>
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
npm run governance:legacy-report
npm run verify:local
```

2026-08-13 v1.127.0 发布候选验证：

```text
npm run verify:app-version
  passed（1.127.0）

npm run typecheck
  passed

npm run test:contracts
  passed（protocol types 无漂移；app-server client 311 checks；command contracts passed）

npm run test:rust:related -- lime-rs/crates/app-server-protocol lime-rs/crates/app-server \
  lime-rs/crates/core/src/database/dao/agent_run.rs lime-rs/crates/scheduler
  passed（包含 app-server 1675、agent-runtime 208、app-server-protocol 122、tool-runtime 340 项）

npx vitest run <scheduled-tasks / navigation / app-shell 定向测试>
  8 files / 70 tests passed

npm run verify:gui-smoke
  passed（真实 Electron + App Server 初始化，result=pass）

git diff --check
  passed
```

`npm run test:related -- <changed frontend paths>` 因测试编排器将仓库 `electron/` 目录作为文件读取而报
`EISDIR`；相同功能面已改用显式 Vitest 文件清单执行并全部通过。该结果记为编排器路径解析缺陷，
不是产品断言失败。

## 阻塞与风险

- 早期 `ptrcomp_sandbox` archive 404 阻塞已不再阻断本地发布候选：本轮默认 `npm run test:rust:related -- ...` 已完整编译并通过受影响 owner 与反向依赖。该结论只覆盖当前 macOS arm64 工具链，不替代 Windows 平台证据。
- 旧 `automationJob/*`、`automationSchedule/*`、`automationScheduler/*` 和旧 Settings consumer 仍存在，分类为 `deprecated / migration-pending`，当前不是唯一协议面。
- 启动恢复、interval sleep/wake reconcile 与时钟跳变源码合同已完成定向验证；真实 macOS/Windows OS sleep/resume 进程证据、terminal notification、软删除和运行中删除合同尚未完成。
- 仓库级真实 Electron GUI smoke 已通过；Scheduled Tasks 专项 Agent current fixture、用户路径 Gate B 和 Windows 平台证据尚未完成。

## 架构确认

```text
架构影响：重大；新增 Scheduled Tasks public JSON-RPC/read-model 边界与一级工作台，运行复用 RuntimeCore/Thread/Turn/Item。
架构图已更新：internal/aiprompts/architecture.md#73-scheduled-tasks；internal/aiprompts/commands.md#scheduled-tasks-主链。
责任开发者确认：root（2026-08-13，v1.127.0 发布）
确认内容：已核对目录归属、数据流、依赖方向、协议边界和验证门禁。
```

架构确认已满足 release evidence 入口；Windows 平台证据仍必须使用真实 Windows runner，macOS 不能替代该平台结论。
