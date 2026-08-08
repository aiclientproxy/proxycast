# 基础 Prompt 主链

## 这份文档回答什么

本文件只定义 Lime 当前真正入模的基础 Prompt 主链，回答四个问题：

1. 当前 provider 最终看到的 system prompt 是怎么拼出来的
2. `project / session / frontend / runtime AGENTS / prompt_context / augmentation` 分别处在什么位置
3. 哪些 prompt 文件和 builder 属于 `current`，哪些只是 `compat` / `deprecated`
4. `query-loop.md`、功能样板文档、历史工作台说明与基础 Prompt 主链的关系是什么

一句话事实源声明：

> 后续所有基础 Prompt、system prompt、subagent prompt、plan prompt、augmentation 顺序与 diagnostics 判断，统一向 `App Server runtime_backend/request_context/session_config.rs -> lime-agent current_provider_turn -> agent-runtime provider_turn -> model-provider lowering` 这一条 current 主链收敛。

路径边界：`lime-rs/src/**` 已删除；旧 `runtime_turn.rs`、`prompt_context.rs` 只允许从 git history / 执行计划只读参考，不是 current owner。新增 Prompt 能力、augmentation stage、diagnostics 或 provider prompt 组装逻辑应进入 App Server / RuntimeCore / prompt services / `lime-rs/crates/agent`，不能恢复 `lime-rs/src/commands/**` 业务逻辑、compat wrapper 或退场 stub。

## Current 主链总览

```text
lime-rs/crates/app-server/src/runtime_backend/request_context.rs
  -> 选择 base session prompt 来源（project > session > frontend > none）
  -> merge_system_prompt_with_runtime_agents(...)
  -> merge_system_prompt_with_request_tool_policy(...)
  -> 可选 merge_system_prompt_with_web_search_preflight_context(...)
  -> TurnInputEnvelope 记录 base/final prompt 与 augmentation stages
  -> SessionConfig.system_prompt

lime-agent::current_provider_turn
  -> merge_system_prompt_with_request_tool_policy(...)
  -> agent-runtime::provider_turn
  -> CurrentProviderRequest.system_prompt
  -> model-provider lowering
  -> provider 实际收到的最终 system prompt / instructions / system
```

关键事实：

- `TurnInputEnvelope` 里记录的 `base_system_prompt_len / final_system_prompt_len` 只覆盖 Lime 侧的 `session prompt` 片段，不等于 provider 侧最终收到的完整 system prompt 长度。
- `current_provider_turn` 在开始 provider turn 前合并 request tool policy；随后 `agent-runtime::provider_turn` 只将同一份 `system_prompt` 放入 `CurrentProviderRequest`。
- 因此，排查“Prompt 为什么变长”“Prompt cache 为什么失效”时，必须先看 App Server session config、runtime agents / skills / plugin / memory / soul context、`TurnInputEnvelope` 和 `current_provider_turn`，而不是已删除 runtime 的 prompt manager 或模板链。

## 基础 Prompt 的 current 事实源

### 1. Base Session Prompt 入口

- `lime-rs/crates/app-server/src/runtime_backend/request_context.rs`
- `lime-rs/crates/agent/src/agent_state_support.rs`
- 优先级固定为：
  1. `project prompt`
  2. `session prompt`
  3. `request.system_prompt`（frontend）
  4. `None`

对应实现：

- `build_project_system_prompt(...)`
- `session_state_snapshot.system_prompt()`
- `request.system_prompt`
- `TurnInputEnvelopeBuilder::set_base_system_prompt(...)`

### 2. Lime 侧 augmentation 主链

- `lime-rs/crates/app-server/src/runtime_backend/request_context.rs`
- `lime-rs/crates/agent/src/request_tool_policy.rs`
- `lime-rs/crates/agent/src/prompt/runtime_agents.rs`
- `lime-rs/crates/agent/src/turn_input_envelope.rs`

FullRuntime 固定顺序：

1. `RuntimeAgents`
2. `ExplicitLocalPathFocus`
3. `Memory`
4. `WebSearch`
5. `RequestToolPolicy`
6. `Artifact`
7. `ImageSkillLaunch`
8. `CoverSkillLaunch`
9. `VideoSkillLaunch`
10. `BroadcastSkillLaunch`
11. `ResourceSearchSkillLaunch`
12. `ResearchSkillLaunch`
13. `ReportSkillLaunch`
14. `DeepSearchSkillLaunch`
15. `SiteSearchSkillLaunch`
16. `PdfReadSkillLaunch`
17. `PresentationSkillLaunch`
18. `FormSkillLaunch`
19. `SummarySkillLaunch`
20. `TranslationSkillLaunch`
21. `AnalysisSkillLaunch`
22. `TranscriptionSkillLaunch`
23. `UrlParseSkillLaunch`
24. `TypesettingSkillLaunch`
25. `WebpageSkillLaunch`
26. `ServiceSkillLaunch`
27. `Elicitation`
28. `TeamPreference`
29. `AutoContinue`
30. `ServiceSkillLaunchPreload`（在主组装后追加，只在 `FullRuntime` 生效）

FastChat 固定顺序：

1. `RuntimeAgents`
2. `ExplicitLocalPathFocus`
3. `RequestToolPolicy`

这里的“固定顺序”以 App Server request context 和 `TurnPromptAugmentationStageKind` 为准；文档、前端假设或样板说明不得自行重排。

### 3. Current provider turn

最终 provider request 由以下 current owner 串联：

- `lime-rs/crates/app-server/src/runtime_backend/request_context/session_config.rs`
- `lime-rs/crates/agent/src/prompt/runtime_agents.rs`
- `lime-rs/crates/agent/src/request_tool_policy/policy_config.rs`
- `lime-rs/crates/agent/src/current_provider_turn.rs`
- `lime-rs/crates/agent-runtime/src/provider_turn.rs`
- `lime-rs/crates/model-provider/src/current_client/{lowering,stream,transport}.rs`

`model-provider` 会按 provider protocol 将同一份 `CurrentProviderRequest.system_prompt` lower 为 Chat Completions `system` message、Responses `instructions` 或 Anthropic `system` 字段。没有第二套 prompt manager、模板目录或 retired source adapter 可以改写该事实。

## Current / Compat / Deprecated 边界

### Current

以下路径是当前唯一允许继续演进的基础 Prompt 事实源：

- `lime-rs/crates/app-server/src/runtime_backend/request_context.rs`
- `lime-rs/crates/agent/src/request_tool_policy.rs`
- `lime-rs/crates/agent/src/prompt/runtime_agents.rs`
- `lime-rs/crates/agent/src/turn_input_envelope.rs`
- `lime-rs/crates/app-server/src/runtime_backend/request_context/session_config.rs`
- `lime-rs/crates/app-server/src/runtime_backend/agent_skills_context.rs`
- `lime-rs/crates/agent/src/current_provider_turn.rs`
- `lime-rs/crates/agent-runtime/src/{session_config,provider_turn}.rs`
- `lime-rs/crates/model-provider/src/current_client/{lowering,stream,transport}.rs`

### Compat

以下路径是局部 helper 或历史 evidence，不属于基础 Prompt 主链，后续不要继续把新能力长进去：

- `lime-rs/crates/agent/src/prompt/builder.rs`
- `lime-rs/crates/agent/src/prompt/templates.rs`
- `internal/aiprompts/query-loop.md`
  这是 Query Loop current 文档，但不是基础 Prompt 逐层拼装的唯一事实源
- `internal/aiprompts/content-creator.md`
  这是归档工作台说明，不是基础 Prompt 主入口

这些 compat 路径可以继续被读取、测试或保留导出，但不能再被当成“当前 prompt 主链定义处”。

### 特殊说明

- 已删除 runtime 的 prompt manager、embedded prompt 和 vendor 文件只能作为 git history / 执行计划 evidence，不属于基础 Prompt current 主链。
- `internal/prd/gongneng/**`、`internal/roadmap/**`、`x-article-export/**` 等功能样板或产品文档，只能消费主链事实，不能反向定义基础 Prompt 顺序。

## 谁可以定义基础 Prompt，谁只能消费

可以定义基础 Prompt 的边界：

- `lime-rs/crates/app-server/src/runtime_backend/request_context.rs`
- `lime-rs/crates/agent/src/request_tool_policy.rs`
- `lime-rs/crates/agent/src/prompt/runtime_agents.rs`
- `lime-rs/crates/agent/src/turn_input_envelope.rs`
- `lime-rs/crates/agent/src/agent_state_support.rs`
- `agent-runtime::provider_turn` 与 `model-provider::current_client` 的 request / lowering 边界

只能消费、解释或验证基础 Prompt 的边界：

- 前端 Workspace / Chat UI
- `internal/aiprompts/query-loop.md`
- `internal/prd/**` / `internal/roadmap/**`
- 功能样板文档，例如 `x-article-export`
- 历史工作台归档文档，例如 `content-creator.md`

如果消费层文档与上述 current 事实源冲突，以 current 代码边界为准，并同步回写文档。

## 与 Query Loop 的关系

- `query-loop.md` 负责解释 submit turn、queue、tool runtime、context compaction、evidence 等主循环。
- 本文只负责解释“系统提示词是如何形成并进入 provider”的主链。
- 两者关系是并列 current 文档，但基础 Prompt 的更细粒度事实源以本文为准。

遇到以下改动时，先读本文，再回到 `query-loop.md` 看提交链：

- 改 `system prompt`
- 改 `subagent prompt`
- 改 `plan prompt`
- 改 App Server request context 或历史 `prompt_context.rs` 迁移锚点
- 改 augmentation 顺序或 marker
- 改 `TurnInputEnvelope` diagnostics
- 排查 token、Prompt Cache、prompt 变长、无声注入等问题

## 对齐结论

本轮治理后的统一口径是：

- Lime 的基础 Prompt 主链不是某一份前端 `systemPrompt`、某一份 PRD，或某个样板包
- Lime 的基础 Prompt 主链也不是单独某个 builder 文件
- 真正的 current 主链是 “App Server request context 先组 session prompt，`lime-agent` 追加 turn policy，`agent-runtime` 生成 provider request，`model-provider` lower 成上游请求”
- 后续所有 Prompt 相关治理，必须围绕这条主链做减法和收口，而不是再新增平级 builder、平级模板或平级文档解释
