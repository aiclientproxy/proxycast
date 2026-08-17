# Lime 已安排任务路线图

状态：`implemented / platform-evidence-pending`

更新时间：2026-08-17

## 主目标

在 Lime Desktop 中提供一套可直接工作的“已安排任务”能力：用户可以从独立入口或当前对话创建任务，配置运行上下文与日程，查看启用/暂停状态、下一次运行和历史结果，并能从任一运行回到真实 Agent 对话。

本路线不是新建一套任务后端。产品层统一命名为“已安排任务”，实现必须演进现有 current owner：

```text
Electron Desktop Host
  -> App Server JSON-RPC scheduledTask/*
  -> scheduler + App Server automation execution
  -> RuntimeCore -> Thread / Turn / Item projection
  -> AgentRun read model
  -> Scheduled Tasks GUI
```

## 需求来源与事实等级

| 来源                                                   | 用途                                                                   | 事实等级              |
| ------------------------------------------------------ | ---------------------------------------------------------------------- | --------------------- |
| 用户提供的 5 张 ChatGPT/Codex Desktop 截图             | 页面信息架构、列表/详情/创建/对话内创建体验                            | 产品目标              |
| `/Users/coso/Documents/dev/rust/codex`                 | `ScheduledTaskSummary`、`hourly/daily/weekdays/weekly` 与 weekday 枚举 | 可验证 parity 基准    |
| Lime current `scheduledTask/*`、scheduler、Agent Run | CRUD、持久化、调度、真实执行和历史能力                                  | 已实现基线            |
| Lime 当前“持续流程”页面                                | 可迁移功能与治理输入                                                   | 迁移来源，不是目标 UX |

不得把截图中不可见的行为写成“Codex 已实现”；不得把 Lime 现有字段未经产品审查全部暴露到新主界面。

## 固定产品结论

1. 侧边栏提供一级入口“已安排任务”，打开后先看到任务目录，不进入设置页。
2. 页面采用桌面主从工作台：窄列表 + 宽详情；无选中项时使用居中目录态。
3. 创建有两条等价入口：“使用 Lime 创建”和“手动设置”。前者进入真实 Agent 对话，通过结构化 draft 确认后创建；后者进入表单。
4. 任务运行必须创建或继续 canonical Thread，并通过标准 `turn/start` 进入 RuntimeCore；运行结果不是独立日志文本。
5. 运行历史每一项必须可打开对应 Thread/Turn，失败项必须展示可行动原因。
6. 首期产品日程只暴露 `hourly/daily/weekdays/weekly`，语义对齐 Codex；现有 Cron/一次性/秒级间隔作为迁移输入，不在首期主表单直接暴露。
7. 任务状态使用“已启用/已暂停/运行中/需处理/失败”产品语义；底层执行状态继续归 Agent Run/Turn owner。
8. 通知、权限、模型与推理是任务运行快照的一部分，不能在调度触发时隐式读取漂移后的 UI 临时状态。
9. 生产路径不得回退 mock、renderer timer 或 Electron 第二后端。

## Current / Deprecated / Dead

| 分类                | 能力/路径                                                                            | 处置                                                                           |
| ------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `current` | App Server `scheduledTask/*`、`src/lib/api/scheduledTasks.ts` | 唯一公开 CRUD、运行、预览和通知合同 |
| `current` | `lime-rs/crates/scheduler`、`automation_jobs` | 唯一持久化调度 owner；旧 Automation 命名只存在于内部存储映射 |
| `current` | RuntimeCore Thread/Turn/Item、Agent Run | 真实执行、投影与历史 |
| `current` | `src/components/scheduled-tasks/**` | 一级主从工作台与 Service Skill 创建对话框 |
| `compat` | Base Setup / Service Skill `automation_job` binding family | 只做 catalog 分类；不得恢复旧协议或 Agent UI projection，待独立 schema 迁移 |
| `dead / deleted` | `automationJob/*`、`automationSchedule/*`、`automationScheduler/*`、旧页面/Settings/文案/smoke/Agent UI projection | 只允许负向回流守卫和历史 evidence |

本路线不设置长期 `compat` owner。若审计发现本地存量，只允许启动期一次性数据迁移；迁移完成后旧协议、旧日程写入和旧 UI 入口立即删除。

## 文档导航

1. [当前基线](./00-current-baseline.md)
2. [产品需求合同](./01-product-requirements.md)
3. [交互与页面状态](./02-interaction-and-ui.md)
4. [领域与协议合同](./03-domain-and-protocol-contract.md)
5. [运行架构](./04-runtime-architecture.md)
6. [迁移与清理账本](./05-migration-ledger.md)
7. [实施计划](./06-implementation-plan.md)
8. [验收与验证合同](./07-verification-contract.md)
9. [Codex parity 矩阵](./08-codex-parity-matrix.md)

## 完成定义

- 用户可从一级入口与当前对话创建任务。
- 手动表单覆盖标题、指令、运行上下文、项目、模型、推理、日程、通知和启用状态。
- 到期任务与“立即运行”都走同一真实 Thread/Turn/Item 链，且每次运行可追溯。
- 暂停、恢复、编辑、删除、搜索、筛选、运行历史和失败恢复均可用。
- 冷启动后任务、next-run 和历史一致；macOS 与 Windows 行为一致。
- 旧“持续流程”入口、旧浏览器任务与旧 SceneApp 上下文完成迁出/删除，不形成双轨。
- 通过 Rust related、contracts、current fixture、GUI smoke 与真实 Electron Gate B。

## 当前下一刀

补真实 Windows Notification Center、Windows Gate B 与 macOS/Windows sleep-resume 平台证据。不得用受控时钟回归或 macOS Electron Gate B 代替跨平台证据。
