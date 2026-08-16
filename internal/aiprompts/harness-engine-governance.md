# Lime Harness Engine 治理规范

## 这份文档回答什么

本文件定义 Lime 的 harness Engine 应该如何长期演进，主要回答：

- 运行时诊断、runtime facts、replay case、analysis handoff、review template 分别谁负责什么
- 什么才算 harness 的唯一事实源，什么只是消费层或展示层
- 哪些证据可以导出，哪些只能在“适用时”导出，不能把不存在的信号硬写成全局缺口
- 当 Lime 对齐 Codex 的运行时边界时，哪些原则应该真正落到代码里

它是 **Harness Engine 的长期治理文档**，不是某次分析导出功能的临时设计说明。

## 对齐目标

这份规范中的 Claude Code 内容只保留为历史 evidence，不是 Lime 的能力事实源。Lime 的 Agent runtime、Thread/Turn/Item、审批和导出边界只按 Codex 收敛；Harness evidence/replay/analysis/review 仍是 Lime 产品扩展，不得冒充 Codex portable signed receipt。当前保留的治理原则是：

1. **单一事实源**
2. **先采集，再解释**
3. **只有可关联的遥测，才算会话级证据**
4. **只有真正发生过的验证，才进入 evidence**
5. **消费层不重新拼装另一套运行时真相**

一句话说，就是：

**Harness Engine 不是“再做一层报表”，而是把 canonical runtime facts 稳定投影，供 replay / analysis / review / UI 复用。**

## 事实源声明

从现在开始，Harness Engine 相关能力默认收敛到以下主链。Lime 的 derived export 只消费 canonical Thread/Turn/Item facts，不得把导出结果描述为签名回执：

- 运行时事实：App Server `agentSession/read` / `thread/read`
- 回放样本导出：App Server `agentSession/replayCase/export`
- 外部诊断交接：App Server `agentSession/analysisHandoff/export`
- 人工审核模板：App Server `agentSession/reviewDecisionTemplate/export`、`agentSession/reviewDecision/save`

其中：

- canonical read model 是 **运行时事实源**
- `replay / analysis / review / history-record summary` 是 **派生物**
- GUI 面板例如 `HarnessStatusPanel` 与 nightly dashboard 是 **展示层**

Codex 对齐补充：`HarnessStatusPanel` 的运行时事实区只消费 canonical `thread/read`
投影；Workspace 已保存技能面板也不维护独立 transcript。只有仍有明确产品消费者的
Lime 派生链才可暂时保留，不能把导出结果当作 Codex current 或来源真实性证明。

允许的数据方向只有：

`runtime thread/session` -> `canonical read model` -> `replay / analysis / review / summary` -> `trend / cleanup / dashboard` -> `UI`

禁止反向或旁路：

- `analysis` 自己再拼一套 observability summary
- `replay` 不复用 evidence pack，改为本地猜测缺口
- 旧 Goal audit 重新读取 evidence pack 并建立第二套完成状态
- `UI` 的显示状态反过来成为事实源

ThreadGoal 不属于 Harness evidence 派生物；它只消费 canonical Thread 状态并由 RuntimeCore 管理。旧 ManagedObjective audit 已删除，禁止作为新的证据导出链恢复。

## 分类语言

Harness Engine 相关 surface 继续沿用仓库统一分类：

- `current`：当前唯一主路径，后续功能只允许继续向这里收敛
- `compat`：兼容层，只允许委托和适配
- `deprecated`：已知要退出的旧路径，只允许迁移和下线
- `dead`：已无入口或确认停用，优先删除

在 harness 语境里，常见判断如下：

- `current`
  - App Server `agentSession/read` / `thread/read`
  - canonical Thread/Turn/Item facts 与 artifact/read
- `compat`
  - 临时存在但只做委托的旧导出入口
  - 仍保留的旧 GUI 打开目录/复制命令适配层
- `deprecated`
  - 各消费方各自维护的 observability builder
  - 无条件导出固定 known gaps 的旧逻辑
- `dead`
  - 已没有入口引用的旧 handoff / review / diagnostics 拼装路径

## 核心治理原则

### 1. 先导出事实，再导出判断

evidence pack 先负责导出：

- 关联键
- timeline
- warnings / failed tool / failed command
- 最近产物
- 已真实发生的 verification 记录

其中 `observabilitySummary` 应优先承载紧凑的 verification outcome 摘要；更完整的明细仍可放在 `artifacts.json`，但下游不应为了得到“是否通过 / 是否失败 / 是否缺失”再单独重建第二套判断。

analysis brief、replay input、review template 只能在此基础上做解释，不能自己再回头拼装另一套事实。

### 2. 信号必须按适用性导出

可验证运行时的关键不是“信号越多越好”，而是“只导出当前回合真正成立的信号”。

因此 Lime 的 harness 必须遵守：

- `requestTelemetry`
  - RequestLog 必须携带 `session_id / thread_id / turn_id / pending_request_id / queued_turn_id / subagent_session_id`
  - evidence pack 必须按这些关联键导出会话级 request telemetry 摘要
  - 如果当前线程没有匹配请求，导出空摘要即可，不能再伪造 `unlinked`
- `artifactValidator`
  - 只有最近产物里存在 ArtifactDocument 或等价 artifact schema 时，才允许出现
- `browserVerification`
  - 只有线程里真实出现浏览器工具或浏览器命令时，才允许出现
- `guiSmoke`
  - 只有线程里真实执行了 `verify:gui-smoke` 时，才允许出现

不适用的信号不应进入 `known_gaps`，更不应被硬编码成所有线程的默认缺口。

### 3. gap 代表“已证明缺失”，不是“历史上想接”

`known_gaps` 只能来自：

- 已定义的信号覆盖表
- 且状态不是 `exported`
- 且该信号对当前线程确实适用

禁止继续保留以下旧做法：

- 所有线程都写 `artifactValidator/browserVerification/guiSmoke`
- 即使没有相关产物或相关命令，也输出空 verification 壳
- 用“未来计划支持”冒充“当前已识别缺口”

补充约束：

- trend / cleanup report 必须区分 `current` 样本里的 gap 和 `degraded` 样本故意保留的 gap
- 只有 `current` 样本出现的 observability gap 才应被当成主线回归风险
- `degraded` 样本的作用是保留诊断语义与 known gap 事实，不能反过来把 cleanup 优先级抬高

### 4. 关联键先于遥测摘要

Codex 的 rollout / Thread / Turn / Item 之所以可用，前提是事件先有稳定的 session 级关联。

因此 Lime 必须把下面这些键视为 harness 级基础设施，而不是锦上添花：

- `session_id`
- `thread_id`
- `turn_id`
- `pending_request_id`
- `queued_turn_id`
- `subagent_session_id`

如果 RequestLog 还没稳定携带这些键，就不能把 request telemetry 当成会话级证据；但一旦键已补齐，后续 analysis / replay / review / UI 必须统一消费同一份 request telemetry 摘要，不能继续并存旧的 `unlinked` 语义。

### 5. 展示层不能成为新事实源

以下内容只能消费 evidence pack，不能反向定义事实：

- `HarnessStatusPanel`
- 外部 AI copy prompt
- review-summary / analysis-brief 里的文案摘要
- GUI 上看到的 warning / gap / status 卡片

展示层至少要如实透出：

- `current / degraded` observability gap 角色
- `observabilityVerificationOutcomes` 里的紧凑 outcome 焦点
- cleanup 主事实源里已经存在的 `blocking_failure / advisory_failure / recovered` 摘要

对 cleanup / dashboard / nightly report 这条推荐动作链，字段语义也必须固定：

- failure 焦点只允许走 `focusVerificationFailureOutcomes`
- recovered 焦点只允许走 `focusVerificationRecoveredOutcomes`
- 不允许继续把两类 outcome 混在一个 recommendation 字段里，再由展示层二次猜测

一旦 UI 文案和后端 evidence 冲突，以 evidence pack 和运行时导出为准。

### 6. 诊断链路必须复用下游制品

标准导出链路应保持为：

1. `evidence pack` 先落盘
2. `replay case` 复用 evidence pack
3. `analysis handoff` 复用 handoff bundle + evidence pack + replay case
4. `history-record` 复用 summary，并继续派生 `trend / cleanup / dashboard`
5. `review decision template` 再复用 analysis handoff

任何一层如果又回去重新读取 thread 并拼第二套摘要，都视为治理倒退。

### 7. 旁路系统也必须服从同一事实源

Harness Engine 不只是导出文件目录。以下旁路也必须收敛：

- diagnostics 摘要
- export copy prompt
- review template
- GUI 状态卡
- 后续 eval / trend / promote 使用的样本事实

如果主导出已经修了，旁路仍沿旧字段判断，就说明治理还没完成。

## 典型反模式

出现以下行为，视为 Harness Engine 治理倒退：

- 在 `analysis / replay / UI` 各自维护一份 observability 拼装逻辑
- 给所有线程硬塞 `artifactValidator`、`browserVerification`、`guiSmoke`
- verification 还没发生，就先导出空数组或空对象占位
- RequestLog 还无法 join，会话摘要却写成“已有 request telemetry”
- 后端证据已经修正，前端或 handoff 文案仍沿旧 gap 模板输出
- 为保留旧行为再新增一层 compat 包装，而不是删掉错误事实

## 变更顺序

涉及 Harness Engine 改动时，默认按这个顺序做：

1. 先确认唯一事实源是不是 App Server `agentSession/read` / `thread/read`
2. 再确认改动属于 `current / compat / deprecated / dead` 哪一类
3. 先修 canonical facts，再修 replay / analysis / review / UI
4. 先删错误默认值，再考虑是否需要新增字段
5. 最后补定向测试与最小 GUI / 契约验证

如果说不清这次改动要收敛到哪个导出入口，就不要继续扩。

## 最低验证要求

Harness Engine 改动至少应覆盖：

- 受影响 Rust 服务的定向测试
- evidence pack 导出结果检查
- replay / analysis 是否复用 evidence pack 的回归检查
- 如果改动触及 cleanup report / dashboard 推荐动作契约，默认把 cleanup contract 校验纳入最小门槛：

```bash
npm run harness:cleanup-report:check
```

如果要校验某个指定产物，再显式传入：

```bash
node scripts/check-generated-slop-report.mjs --input "<cleanup-json>"
```

如同时触及命令边界，再补：

```bash
npm run test:contracts
```

如同时触及 GUI 主路径或 `HarnessStatusPanel`，再补：

```bash
npm run verify:gui-smoke
```

人工 spot check 至少看一次：

- `.lime/harness/sessions/<session_id>/evidence/runtime.json`
- `.lime/harness/sessions/<session_id>/evidence/artifacts.json`
- `.lime/harness/sessions/<session_id>/replay/input.json`
- `.lime/harness/sessions/<session_id>/analysis/analysis-context.json`

## 当前最值得继续优化的一刀

如果继续沿这条主线推进，优先级最高的是：

**把已导出的 request telemetry 继续沉淀到 trend / replay / review 守卫里，确保后续任何新入口都只能复用同一份会话级摘要，而不是再次旁路读取或重建第二套真相。**

这样才能保持“先记录 canonical runtime facts，再做下游解释”的治理边界，而不会把 Lime evidence 扩展误写成 Codex receipt。

## 相关文档

- `internal/aiprompts/governance.md`
- `internal/aiprompts/commands.md`
- `internal/aiprompts/quality-workflow.md`
- `internal/aiprompts/playwright-e2e.md`

## 一句话总结

**Harness Engine 的治理目标不是导出更多文件，而是让所有诊断、回放、审核和 GUI 都只复用同一份运行时事实。**
