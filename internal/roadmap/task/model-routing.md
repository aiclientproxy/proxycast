# Lime 模型路由与候选解析

> 状态：提案
> 更新时间：2026-04-23
> 作用：定义 `CandidateModelSet`、`RoutingDecision`、优先级和“自动与设置平衡”的统一规则。
> 依赖文档：
> - `./task-taxonomy.md`
> - `./oem-and-local-policy.md`
> - `./cost-limit-events.md`
> - `./runtime-integration.md`

当前实现注记：RuntimeCore 已持有 typed `CandidateModelSet`；App Server 只负责把 configured Provider 的声明模型和缓存目录适配为候选来源。能力/任务族/模态过滤先于 OEM 偏好、连续性和成本排序，显式选择不被自动策略覆盖；候选集合会进入 routing decision evidence。凭证 readiness、真实 provider protocol 和未声明能力仍在后续 route gate fail closed。

## 1. 固定目标

模型层后续统一拆成两步：

1. `Candidate Resolution`
2. `Routing Decision`

也就是：

- 先回答“当前真实可选空间是什么”
- 再回答“在这个空间里怎么选，或者为什么根本选不了”

## 2. `CandidateModelSet`

### 2.1 含义

`CandidateModelSet` 不是模型注册表全量目录，而是当前这个 turn 在真实 runtime 约束下可用的候选集合。

### 2.2 主要来源

候选来源统一包括：

1. 显式 per-turn override
2. 当前会话锁定 provider/model
3. 当前任务命中的 `service_models`
4. `service_scene_launch` 指定首选
5. OEM `providerPreference/defaultModel` 与 offer
6. configured providers / API Key Provider
7. 模型注册表能力与定价信息

### 2.3 候选集必须考虑的过滤条件

1. 凭证是否存在
2. provider 是否启用
3. OEM 是否允许
4. 当前任务要求的基本能力是否满足
5. 当前模型是否已废弃或不可用
6. 当前并发预算与限额是否允许继续投递

## 3. `RoutingDecision`

固定字段建议如下：

| 字段 | 含义 |
| --- | --- |
| `routing_mode` | `no_candidate / single_candidate / multi_candidate` |
| `candidate_count` | 当前候选数量 |
| `selected_provider` | 最终 provider |
| `selected_model` | 最终 model |
| `decision_source` | 决策来源 |
| `decision_reason` | 决策解释 |
| `degradation_reason` | 能力缺口或降级原因 |
| `capability_gap` | 缺失的关键能力 |
| `fallback_chain` | 已尝试或可尝试的回退链 |
| `requires_user_override` | 是否需要用户手动切换 |
| `estimated_cost_class` | 成本档 |
| `limit_state_snapshot` | 当前限额快照 |

## 4. 决策来源枚举

建议固定为：

- `explicit_turn_override`
- `session_lock`
- `service_model_preference`
- `service_scene_preference`
- `oem_hard_constraint`
- `oem_soft_preference`
- `policy_auto`
- `runtime_fallback`
- `no_candidate`

## 5. 路由优先级

后续统一按这个优先级执行：

1. 显式 per-turn override
2. 会话硬锁定
3. OEM 硬约束
4. 当前任务命中的 `service_models`
5. `service_scene_launch` 首选
6. OEM 软推荐
7. 会话默认偏好
8. 自动策略
9. 运行时 fallback

固定规则：

- 只有前 3 项属于真正硬约束。
- `service_models` 是强提示，但不是绝对不可动。
- 自动策略只能在上面这些约束允许的范围内工作。

## 6. 自动与设置平衡规则

### 6.1 显式锁定永远优先

如果用户本轮明确指定模型，路由层不得偷偷覆盖。

### 6.2 `service_models` 优先于普通默认模型

如果任务命中了 `service_models` 对应槽位，就优先把它作为首选候选。

### 6.3 但 `service_models` 不等于绝对硬锁

如果出现以下情况，允许回退：

1. 对应 provider/model 已失效
2. 当前任务能力不满足
3. 当前限额已阻断
4. OEM 明确禁止

回退时必须：

- 记录 `decision_reason`
- 记录 `degradation_reason`
- 对 UI 可解释

### 6.4 自动策略只在“有空间”时生效

自动优选的前提是：

1. 无显式锁定
2. 非 OEM managed 硬锁
3. 当前首选设置允许回退
4. 候选集大于 1

## 7. 三种路由模式

### 7.1 `no_candidate`

含义：

- 当前找不到任何可执行候选

系统动作：

- 阻断执行
- 记录 `routing_not_possible`
- 提示缺少凭证、配额、能力或候选

### 7.2 `single_candidate`

含义：

- 当前只有 1 个真实可用候选

系统动作：

- 透传执行
- 或标记为 `single_candidate_capability_gap`
- 或阻断

### 7.3 `multi_candidate`

含义：

- 当前存在多个可用候选

系统动作：

- 按能力、成本、限额、连续性做优选

## 8. 能力优先于价格

固定原则：

1. 先满足硬能力需求
2. 再看 OEM 约束
3. 再看会话连续性
4. 再看成本与限额

不能为了便宜而选一个根本不支持任务能力的模型。

## 9. 连续性规则

以下场景优先保持 provider/model 连续性：

1. 当前会话已有 provider continuation state
2. 当前正在同一任务链中连续追问
3. 当前 artifact / service scene 明确要求同链继续

但连续性不是无条件优先：

- 若当前模型能力不足、不可用、超限或 OEM 禁止，则必须中断连续性并解释

## 10. 建议的成本档

`estimated_cost_class` 建议统一为：

- `unknown`
- `low`
- `medium`
- `high`
- `blocked_by_budget`

说明：

- registry 没有价格数据时，必须允许 `unknown`
- 不允许假装一定能算出真实价格

## 11. `service_models` 在模型路由里的固定角色

`service_models` 后续应进入 `settings_source` 和 `decision_source`：

- 若命中：`decision_source = service_model_preference`
- 若回退：`decision_source = runtime_fallback`，并保留原始设置来源

它的职责是：

- 给当前服务任务提供首选
- 不越过 runtime 做最终决定

## 12. 当前必须避免的误区

1. 把多模型自动优选当成默认现实。
2. 把 `service_models` 当成不可碰的绝对硬锁。
3. 让自动策略偷偷覆盖显式锁定。
4. 单模型时仍然输出“已自动优选最佳模型”这种假解释。

## 13. 这一步如何服务主线

本文件的主线收益是：

**让 Lime 的模型决策从“很多设置各自生效”变成“同一条候选解析与路由决策链统一生效”。**
