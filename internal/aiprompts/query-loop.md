# Query Loop 主链

## 这份文档回答什么

本文件定义 Lime 当前运行时 `Query Loop` 的唯一主链，主要回答：

- 哪个入口才是当前回合提交的唯一开始点
- 请求在进入模型前，system prompt、工具策略、记忆、项目上下文、场景 metadata 是怎样被组装的
- runtime queue、tool runtime、流式执行、artifact 落盘、记忆沉淀、evidence 导出分别挂在哪一段
- 哪些旧路线图或专题文档只能作为下游实现说明，不能再反向充当 Query Loop 事实源

它是 **当前运行时主循环的工程入口文档**，不是通用架构总览，也不是单次专题路线图。

## 什么时候先读

遇到以下任一情况时，先读本文件：

- 调整 App Server `agentSession/turn/start`
- 调整 App Server `agentSession/action/respond` 或 elicitation / ask 恢复路径
- 调整 turn 级 system prompt / metadata / provider routing / continuation
- 调整 runtime queue、流式执行、自动压缩或记忆预取
- 调整 `@` 命令、`/scene`、`service_scene_launch`、`*_skill_launch` 这类“提交前补 metadata”能力
- 调整 evidence / replay / review 与主回合执行的衔接

如果一个需求同时碰到“提交入口 + prompt/metadata 组装 + 工具执行 + 证据导出”里的两步以上，默认属于 Query Loop 改动。

如果这个需求还同时涉及“任务画像、候选模型解析、自动与设置平衡、成本/限额事件底座”，继续补读：

- `internal/roadmap/task/runtime-integration.md`

## 固定主链

当前 Lime 的 Query Loop 统一按下面这条链理解：

`agentSession/turn/start -> RuntimeCore turn_execution -> RuntimeBackend -> TurnInputEnvelope -> runtime_queue -> stream_reply_once -> timeline / artifact / memory -> agentSession/read thread_read -> evidence/export / agentSession/*/export`

这条主链意味着：

1. **统一入口是 App Server `agentSession/turn/start`**
   其他 `@` 命令、`/scene`、图片/素材/站点搜索等场景，只允许在提交前补 `runtimeOptions.runtimeRequest.metadata`，不能绕开 App Server turn start 另建第二条执行链。旧 `agent_runtime_submit_turn` 只能作为前端 compat request 形状或 retired guard，不再是 current 主入口。

2. **RuntimeCore `turn_execution` 与 RuntimeBackend 是当前组包主边界**
   App Server 负责 session/turn lifecycle、runtime options、事件与读模型；RuntimeBackend / `lime-rs/crates/agent` 继续承接 Claw 原链里的 provider 解析、workspace 解析、execution profile、request tool policy、prompt augmentation、sandbox、preload、turn state。旧 `runtime_turn.rs` 只允许作为 git history / 执行计划参考。

3. **`TurnInputEnvelope` 是当前 turn 输入快照**
   它记录最终 system prompt、history source、provider routing、tool policy、continuation、turn context metadata，不允许下游再各自重组另一份“真实输入”。

4. **`runtime_queue.rs` 是当前排队执行边界**
   queue/resume/promote/remove 统一走 runtime queue，不允许每个调用方自己维护另一套忙碌态与排队状态。

5. **工具面仍统一属于 `tool_runtime.rs`**
   浏览器、workspace、search、service skill、subagent、social/image 等工具都从这里注册和裁剪，不允许在 Query Loop 外再长出第二套 registry 语义。

6. **证据链是主链的下游消费，不是旁路真相**
   `thread_read / evidence / replay / review` 只能复用主回合产生的 runtime facts，不能反向定义 Query Loop 真相。

### Rust 命令目录迁移边界

`agent_runtime_*` 等命令名只能作为 retired guard、历史 evidence、测试 fixture 或短期前端 compat request 形状存在；`lime-rs/src/commands/**` 已删除，不是新的 Query Loop 实现目录。本文如提到旧 `runtime_turn.rs`、`tool_runtime.rs`、`command_api/runtime_api.rs`，只表示旧实现来源，不代表 current owner。新增运行时能力必须进入 App Server JSON-RPC、RuntimeCore、ExecutionBackend、services 或 `lime-rs/crates/agent`，不得恢复 compat wrapper、退场 stub 或平行 runtime 分支。

### ThreadGoal / `/goal` 类能力边界

[Codex `/goal`](../research/codex-goal/README.md) 证明了 “persistent objective -> idle continuation turn” 可以由 runtime 管理；Lime current owner 是 thread-owned `ThreadGoal`，不能因此新增第二条 Query Loop。

1. continuation turn 仍必须通过 App Server v2 Turn admission / RuntimeCore queue 进入本主链。
2. Automation 只负责按 schedule 向 persisted Thread 提交 Turn，不拥有第二套 Goal 状态机。
3. Goal 完成只通过 canonical `update_goal` 语义和 persisted ThreadGoal status 表达，不新增 audit RPC。
4. `auto_continue` 仍只表示当前已有文稿续写的 prompt augmentation，不等同于 persistent goal。
5. 如果用户有 queued input、pending elicitation、pause、usage/budget limit 或 blocked 状态，不能自动续跑下一轮。

旧 `ManagedObjective`、`agentSession/objective/*` 和手动 continue 已删除并禁止恢复。任何 ThreadGoal continuation turn 都必须回到 Query Loop 主链。

## 代码入口地图

### 1. 提交入口

- `src/lib/api/agentRuntime/threadClient.ts`
  - 将旧 UI request 形状投影到 App Server `agentSession/turn/start`、`agentSession/turn/cancel`、`agentSession/action/respond`
- `lime-rs/crates/app-server/src/runtime/turn_execution.rs`
  - RuntimeCore turn lifecycle、queue、action respond 与 backend dispatch
- `lime-rs/crates/app-server/src/runtime_backend.rs`
  - App Server 到 Agent / Claw backend host 的当前适配层
- App Server `agentSession/read`

这里的固定规则：

- App Server `agentSession/turn/start` 是唯一 current 提交入口
- App Server `agentSession/action/respond` 这类恢复路径只能复用当前 turn context snapshot 组装，不能再旁路拼第二份 turn context 真相
- App Server `agentSession/read` 负责消费稳定读模型，不负责重新解释提交逻辑
- App Server `thread/compact/start` / RuntimeCore compaction 是主链内的上下文治理动作，不是独立聊天系统

### 2. turn 归一化与输入组装

- App Server `RuntimeOptions.runtimeRequest`
- `lime-rs/crates/app-server/src/runtime/turn_execution.rs`
- `lime-rs/crates/app-server/src/runtime_backend.rs`
- `lime-rs/crates/agent/src/turn_input_envelope.rs`

当前组包边界负责的关键步骤：

1. 确保 agent、session store、runtime support tools 已准备好
2. 解析 provider config、workspace、session recent harness/runtime context
3. 归一化 `request.metadata`
4. 解析 `request_tool_policy` 与 `execution_profile`
5. 在 FullRuntime 下补 `*_skill_launch` / `service_scene_launch` metadata
6. 组装 prompt augmentation：
   - runtime agents
   - memory / turn prefetch
   - web search
   - request tool policy
   - artifact
   - 各类 skill launch
   - team preference / auto continue / service skill preload
7. 生成 `TurnState`
8. 生成 `TurnInputEnvelope` 与 diagnostics snapshot
9. 初始化 runtime turn、发出 runtime status
10. 进入 tracker + stream 执行

`TurnInputEnvelope` 当前固定记录：

- session / workspace / project / thread / turn
- system prompt 来源与最终值
- prompt augmentation stages
- request tool policy snapshot
- provider routing snapshot
- provider continuation state
- effective user message
- turn output schema snapshot
- approval / sandbox policy
- turn context metadata

后续如果一个改动说不清应该落在 App Server turn execution / RuntimeBackend 还是 `TurnInputEnvelope`，先判断它属于：

- **协议与 session/turn lifecycle**：落 App Server protocol / RuntimeCore turn execution
- **Claw 原链组包逻辑**：落 RuntimeBackend / `lime-rs/crates/agent`
- **turn 输入快照字段**：落 `turn_input_envelope.rs`

### 3. queue 与恢复

- `lime-rs/crates/agent/src/runtime_queue.rs`

当前 queue 主链固定为：

- `submit_runtime_turn`
- `resume_runtime_queue_if_needed`
- `clear_runtime_queue`
- `list_runtime_queue_snapshots`
- `remove_runtime_queued_turn`
- `promote_runtime_queued_turn`
- `resume_persisted_runtime_queues_on_startup`

这里的固定规则：

- queue busy / enqueue / start-now 统一由 runtime queue service 决定
- 启动恢复、回合完成后的下一条接力，都复用同一套 queue service
- 不允许前端、命令层或业务专题自己重建另一套排队真相

### 4. 工具面与沙箱

- `lime-rs/crates/agent/src/agent_tools/**`
- `lime-rs/crates/agent/src/request_tool_policy.rs`

当前这里负责：

- runtime support tools 注册
- fast chat 与 full runtime 的工具面裁剪
- workspace sandbox 权限注入
- browser / site / service skill / subagent / workspace / search 等工具面注册或下线

固定规则：

- Query Loop 不直接 new tool registry
- 所有工具面增删都先回到 `lime-rs/crates/agent` 的工具 catalog / policy owner
- `request_tool_policy` 只能在这里影响实际工具可见面，不能在别处再复制一套“隐藏工具”逻辑

### 5. 流式执行与主回合副作用

- `lime-rs/crates/app-server/src/runtime/turn_execution.rs`
- `lime-rs/crates/app-server/src/runtime_backend.rs`

主回合执行当前固定通过：

- `ExecutionTracker::with_run_custom(...)`
- `stream_reply_once(...)`

执行中的 current 副作用包括：

- runtime event -> timeline recorder
- artifact document persist / fallback
- runtime memory capture
- provider continuation state 更新
- runtime status / warning / preload 事件投影

如果一个功能想在“模型回复完成后”做收尾动作，应优先挂到这里，而不是在 UI 或单独 service 旁路触发。

### 6. 压缩、读模型与证据导出

当前 Query Loop 的下游消费统一挂在下面几处：

- App Server exact `thread/compact/start`：params 为 `{ threadId }`，立即返回 `{}`；压缩进度通过标准 `turn/*`、`item/*` lifecycle 投影
- App Server `agentSession/read`
- App Server `evidence/export`
- App Server `agentSession/handoffBundle/export`
- App Server `agentSession/replayCase/export`
- App Server `agentSession/analysisHandoff/export`
- App Server `agentSession/reviewDecisionTemplate/export`
- App Server `agentSession/reviewDecision/save`

固定规则：

- 自动压缩与手动压缩都属于同一 runtime 主链
- 手动压缩不保留旧 session-scoped RPC alias、双发或 fallback
- `thread_read` 是稳定读模型，不重新定义提交逻辑
- `evidence / replay / review` 必须消费 runtime facts，不允许自己拼第二份 Query Loop 真相

## 专项场景如何挂回主链

所有下列能力都只能视为 Query Loop 的“提交前 metadata / prompt 归一化扩展”，不能单独定义新的执行主链：

- `@配图` / `@封面` / `@视频` / `@播报` / `@素材`
- `@搜索` / `@深搜` / `@研报` / `@竞品`
- `@站点搜索` / `@读PDF` / `@总结` / `@翻译` / `@分析`
- `@转写` / `@链接解析` / `@抓取` / `@网页读取`
- `@排版` / `@网页` / `@PPT` / `@表单`
- `service_scene_launch`

它们当前的正确角色是：

1. 前端发送边界写入结构化 `request_metadata.harness.*`
2. RuntimeCore / RuntimeBackend 在提交前归一化 metadata 与 prompt
3. Agent 首刀按系统提示走 skill / tool / service skill 主链

不允许回流成：

- 前端直建任务
- 前端直调云端 run
- viewer 自己推断执行状态
- 单个场景自己维护一套独立 queue / runtime / evidence

## current / compat / deprecated

### `current`

- App Server `agentSession/turn/start`
- App Server `agentSession/turn/cancel`
- App Server `agentSession/action/respond`
- RuntimeCore `turn_execution`
- RuntimeBackend / `lime-rs/crates/agent` Claw 原链适配
- `TurnInputEnvelope`
- `runtime_queue.rs`
- `lime-rs/crates/agent` 工具 catalog / request tool policy
- App Server `agentSession/read`
- App Server `thread/compact/start`
- App Server `evidence/export` 与 `agentSession/*/export`

### `compat`

- `internal/roadmap/lime-agent-codex-alignment-roadmap.md`

这份历史档案与专用命令仍可保留各自职责，但不再承担 Query Loop 唯一事实源职责。
`agent_generate_title` 属于专用一次性会话能力：允许显式拼自己的临时 `SessionConfig`，但不能参与 submit turn、runtime queue 或 evidence 真相定义。
它允许为本地 auxiliary session 附带最小 `lime_runtime` metadata，用于记录 `task_profile / routing_decision / cost_state` 一类辅助任务分类事实；必要时也可以把该 auxiliary session 的 `execution_runtime` 诊断快照回传到命令结果，但这份快照只服务该一次性会话自己的诊断与可观测性，不进入 current Query Loop 的 thread / turn 真相。
当前命令层允许保留的原始执行面只剩 `action_runtime` current 恢复链与 `agent_generate_title` 受控 compat 一次性命令。旧 `persona_cmd` wrapper 已删除；persona 生成语义若继续保留，应通过 runtime auxiliary service / App Server current 边界承接。旧 `agent_theme_context_search` 已下线，主题上下文搜索 current 事实源是 `src/lib/api/themeContextSearch.ts -> App Server agentSession/start + agentSession/turn/start + agentSession/read`。

### `dead`

旧主题上下文搜索 Tauri 命令不得重新注册到 legacy desktop facade；如果后续继续增强主题上下文搜索，只能扩展 App Server current session turn 网关与 RuntimeCore/backend。

### `deprecated`

- 让任何 `@` 场景、slash scene 或 viewer 自己维护执行状态
- 在 UI、专题 service 或证据导出层重新拼装“真实模型输入”
- 绕开 App Server `agentSession/turn/start` 直接定义第二条主回合执行链
- 在 Desktop Host / legacy adapter 命令层继续新增未分类的原始 `agent.reply(...)` / `stream_reply_with_policy(...)` 调用

## 最低验证要求

如果本轮改动涉及 Query Loop，至少按边界选择最贴近的验证：

- Rust 定向测试：
  - `runtime_turn.rs`
  - `turn_input_envelope.rs`
  - `runtime_queue.rs`
  - 相关 evidence / replay / review service
- `npm run governance:legacy-report`
- 文档改动额外跑 `npm run harness:doc-freshness`

如果还改了命令边界、MCP 注入、GUI 时间线或工具展示，再回看：

- `internal/aiprompts/commands.md`
- `internal/aiprompts/quality-workflow.md`
- `internal/aiprompts/harness-engine-governance.md`

## 一句话

> Lime 当前 Query Loop 的唯一主链，是 App Server `agentSession/turn/start` 驱动的 turn 组包、runtime queue、tool runtime、流式执行与 App Server evidence / export 下游消费链；任何新能力都只能往这条链收敛，不能平行再长一套执行真相。
