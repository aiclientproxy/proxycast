# Lime 已安排任务路线图

状态：`requirements-ready / implementation-not-started`

更新时间：2026-08-13

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
| Lime current `automationJob/*`、`scheduler`、Agent Run | CRUD、持久化、调度、真实执行和历史能力                                 | 实现基线              |
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
| `current`           | App Server `automationJob/*`、`automationSchedule/*`                                 | 实施前唯一 CRUD owner；同一迁移变更集替换为已安排任务 typed contract，之后删除 |
| `current`           | `lime-rs/crates/scheduler`                                                           | 继续承担持久化调度语义；不得平级新建 scheduler crate                           |
| `current`           | RuntimeCore Thread/Turn/Item、Agent Run                                              | 继续承担真实执行、投影与历史                                                   |
| `current target`    | `src/components/automation/AutomationPage.tsx`                                       | 重构为已安排任务主从工作台                                                     |
| `deprecated target` | `TaskSchedule::{Every,Cron,At}` 与设置页完整“持续流程”业务工作台                     | 只允许同一变更集迁出；迁移完成即物理删除，不建立 compat owner                  |
| `dead target`       | `protocol/v0/automation.rs`、`browser_session`、SceneApp 旧上下文、renderer 定时触发 | 替换后删除并补回流守卫                                                         |

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

先冻结 `ScheduledTask` v2 产品合同与旧 `AutomationJob` 迁移规则，再做页面。未完成 schema、owner、迁移和失败语义评审前，不应先用现有复杂表单换皮。
