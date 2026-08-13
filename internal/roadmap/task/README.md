# Lime 任务层 / 模型层 / 经济调度路线图

> 状态：方案已落盘，current runtime 基础链路已接入
> 更新时间：2026-04-23
> 作用：把 Lime 的“任务层与模型层分离、成本/限额沉底、OEM 与本地多模型协同、自动与设置平衡”收敛成一套可直接指导实现的专题路线图，而不是继续散落在聊天选择器、`service_models` 设置、OEM offer 与 runtime 局部逻辑里。
> 依赖文档：
>
> - `internal/aiprompts/query-loop.md`
> - `internal/aiprompts/task-agent-taxonomy.md`
> - `internal/aiprompts/state-history-telemetry.md`
> - `internal/aiprompts/providers.md`
> - `internal/exec-plans/upstream-runtime-alignment-plan.md`

## 1. 这套文档回答什么

本专题统一回答下面几类问题：

1. Lime 为什么要把“任务层”和“模型层”拆开，即使当前只配了一个模型也要拆。
2. Lime 当前已有的设置面，例如聊天模型选择、`workspace_preferences.service_models`、媒体默认模型、OEM `providerPreference/defaultModel`，将来在调度里各算什么角色。
3. 自动调度如何和设置平衡，而不是二选一。
4. 当当前只有一个 key、一个 provider、一个模型、一个 OEM session 时，系统应该如何退化，而不是伪装成“自动多模型”。
5. 成本、限额、配额、并发预算事件为什么必须属于 runtime 底层，而不是 UI 提示层。

## 2. 固定结论

这套路线图固定以下结论：

1. **任务层永远存在**
   不管当前可用模型有 1 个还是 10 个，系统都必须先知道“这次要做什么任务”，再决定“当前能不能做、用谁做、如何降级”。

2. **模型层不等于多模型层**
   模型层首先负责解析真实候选集 `CandidateModelSet`，只有候选大于 1 时才进入真正优选；只有 1 个候选时也仍然需要模型层负责能力校验、成本标注、限额判断和降级解释。

3. **设置不是障碍物，也不是唯一事实源**
   Lime 现有设置面必须进入调度链，但不能继续各自偷偷决定运行时真实模型。

4. **自动必须与设置平衡**
   - 显式锁定优先
   - 任务级服务模型设置优先于泛化默认值
   - 自动策略只能在允许范围内优化
   - 当设置指向失效或能力不匹配模型时，系统必须可解释回退，而不是硬撞失败

5. **成本/限额是 runtime 事实，不是页面装饰**
   `cost_estimated`、`rate_limit_hit`、`quota_low`、`single_candidate_only` 这类信号要进入 runtime、telemetry、thread read 和 evidence 导出。

## 3. 当前实现快照

截至 `2026-04-23`，current 主链已经接入下面这组基础事实：

1. `runtime_turn.rs` 已把 `task_profile / routing_decision / limit_state` 写入 `turn_context.metadata.lime_runtime`
2. `request_model_resolution.rs` 已返回完整 resolution envelope，不再只返回 `provider_config`
3. `service_models.translation` 已进入 current runtime 路由链
4. `task_profile_resolved / routing_decision_made / limit_state_updated` 已进入 runtime event 与前端协议
5. `thread_read` 已能稳定读出 `task_kind / service_model_slot / routing_mode / decision_source / candidate_count / capability_gap / limit_state`
6. `npm run verify:local` 已完整通过，包含 `verify:gui-smoke`

## 4. 先读顺序

推荐阅读顺序：

1. [overview.md](./overview.md)
2. [architecture.md](./architecture.md)
3. [task-taxonomy.md](./task-taxonomy.md)
4. [model-routing.md](./model-routing.md)
5. [oem-and-local-policy.md](./oem-and-local-policy.md)
6. [cost-limit-events.md](./cost-limit-events.md)
7. [event-chain.md](./event-chain.md)
8. [runtime-integration.md](./runtime-integration.md)
9. [diagrams.md](./diagrams.md)
10. [rollout-plan.md](./rollout-plan.md)
11. [acceptance.md](./acceptance.md)

## 4.1 子路线

- [已安排任务产品化路线](./scheduled-tasks/README.md)：复刻桌面端“已安排任务”的创建、管理、运行与历史体验，并把现有 `automationJob/*`、`scheduler` 和 Agent Thread/Turn/Item 主链收敛为唯一实现 owner。

## 5. 文档分工

- [overview.md](./overview.md)：先把问题空间和固定原则讲清楚
- [architecture.md](./architecture.md)：定义分层、边界、固定职责
- [diagrams.md](./diagrams.md)：统一架构图、流程图、时序图和降级图
- [task-taxonomy.md](./task-taxonomy.md)：定义 `TaskProfile` 与任务画像维度
- [model-routing.md](./model-routing.md)：定义 `CandidateModelSet`、`RoutingDecision` 与优先级
- [oem-and-local-policy.md](./oem-and-local-policy.md)：说明 OEM 控制面与本地运行时如何协同
- [cost-limit-events.md](./cost-limit-events.md)：定义成本、限额、配额、并发预算事件底座
- [event-chain.md](./event-chain.md)：对照 Claude Code 事件分层，固定 Lime 任务 / 路由 / 经济事件如何接到 current 主链
- [runtime-integration.md](./runtime-integration.md)：说明如何挂回 Lime current 主链
- [rollout-plan.md](./rollout-plan.md)：给出分阶段落地路径
- [acceptance.md](./acceptance.md)：给出验收标准和关键场景

## 6. 当前事实源与本专题的关系

本专题不会替代现有 current 主链，而是对它们做专题级补全：

- `query-loop.md` 负责当前 turn 提交、组包、执行、证据导出主链
- `task-agent-taxonomy.md` 负责当前 `agent turn / subagent turn / automation job` taxonomy
- `providers.md` 负责 provider 接入与凭证边界
- 本专题负责“任务画像 -> 候选集解析 -> 路由决策 -> 成本/限额事件 -> 设置与自动平衡”这条缺口

## 7. 当前必须避免的误区

1. 把“任务层分离”误解成“必须默认多模型”。
2. 把“自动调度”误解成“随意覆盖用户设置”。
3. 把 `service_models`、聊天模型选择器、OEM 默认模型继续当作彼此独立的事实源。
4. 把成本和限额只留在 UI 提示，而不进入 runtime 记录。
5. 在只有一个模型时继续讲“智能路由”，却不说明真实限制。

## 8. 这一步如何服务主线

这套文档的直接主线收益只有一句话：

**它把 Lime 当前分散在会话模型选择、服务模型设置、OEM offer、configured providers / API Key Provider / 模型注册表和 runtime 恢复链里的模型决策，收敛成一条可解释、可落盘、可验收的 current 设计主线。**
