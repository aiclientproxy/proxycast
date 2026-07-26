# grok-build 与 OpenCode 多模型对比

参考快照：

- grok-build：`98c3b2438aa922fbbe6178a5c0a4c48f85edc8ce`
- OpenCode：`fab213312927ea64cf968832c527206e8c944f9e`

grok-build 快照落后 `README.md` 的 reference lock（`6e386420...`）。本文只裁决 grok-build 与
OpenCode 的分层取舍，该结论不随 revision 变化；具体控制平面字段以
[02-multi-model-grok-build.md](02-multi-model-grok-build.md) 和 lock revision 为准。

## 结论先行

对 Lime 最合适的不是完整二选一，而是分层采用：

```text
grok-build = 模型控制平面主参考
  catalog / default / selection / switch / capability / child subset
  auth-aware refresh / retry / circuit breaker

OpenCode = provider wire 平面辅参考
  endpoint union / protocol lowering / content parts / media
  tool-call-result / reasoning / stream reducer / provider plugin shape

Codex = runtime 与持久化事实源
  Thread / Turn / Item / EventLog / ThreadStore / App Server
```

如果只能选一个作为“多模型总参考”，选 **grok-build**；如果只选一个作为“跨 provider 协议适配参考”，选 **OpenCode**。这两个结论不冲突。

## 逐项比较

| 维度 | grok-build | OpenCode | 对 Lime 的判断 |
| --- | --- | --- | --- |
| 运行时语言/边界 | Rust workspace，`ModelsManager`、session handler、tool runtime 与 route 状态都在同一类型系统内 | TypeScript + Effect，Catalog/LLM/Session 分层清楚，但需要跨 Rust/Electron 边界接入 | **grok 更适合做 Lime current model control owner** |
| Catalog 数据模型 | bundled JSON + remote fetch/cache/ETag + auth visibility + allow/disabled/hidden glob；`ModelEntry/ModelInfo` 可承载 xAI 专属字段 | `Provider.Info` + 嵌套 `Model.Info`，models.dev 刷新、TTL、flock、provider/model enabled、status、cost、limits、variants | OpenCode 的 provider/model schema 更通用；grok 的刷新与选择状态更适合 runtime |
| 默认模型解析 | CLI/env/config/remote/bundled 分层，首次真实 catalog 与后续 refresh 有不同重选语义 | `Catalog.model.default()` 先用显式 default，否则按 release date 取 available；简单可预测 | **grok 用于 precedence/refresh；OpenCode 的 explicit no-silent-fallback 规则保留** |
| Provider readiness/认证 | `ModelFetchAuth` 区分 custom endpoint/session/deployment/API key；catalog 与 auth visibility 绑定 | `Integration` 判断 provider available；credential 由 route auth 应用，catalog policy 与 credentials 分离 | grok 的 auth-aware catalog 值得采用；OpenCode 的 policy/config 分离值得采用 |
| Endpoint/协议覆盖 | 以 xAI/Anthropic sampling 为主，wire 类型清晰但 provider 生态较窄 | `Endpoint` 明确覆盖 OpenAI Responses/Completions、Anthropic Messages、AISDK、unknown；协议模块覆盖 OpenAI/Anthropic/Gemini/Bedrock 等 | **OpenCode 胜出，作为 Lime provider lowering 参考** |
| Canonical message/content | `MessageContent`/`ContentBlock` 主要服务 wire API，强项是 Anthropic block 与 stream | `LLM.ContentPart` 统一 Text/Media/ToolCall/ToolResult/Reasoning，`LLMRequest` 是 provider-neutral 输入 | **OpenCode 胜出；Lime canonical 继续由 Rust `model-provider` 持有，借鉴其类型边界** |
| 多模态 | 有图片 block、工具结果 content、thinking；需要按 provider 类型处理 | MediaPart 明确 mediaType/data/filename，协议 lowering 对 text/media/tool/reasoning 做支持检查并 fail closed | OpenCode 的 media/lowering 更完整；Lime sidecar 约束继续服从 Codex |
| Capability matrix | `CanonicalModel` 有 task family、modality、runtime feature、context/pricing；`CapabilityMode` 对 ToolKind 穷举并防 child widening | Model capabilities 有 tools/input/output、reasoning options、modalities、limits；provider shared `supportsContent` 做协议能力校验 | **组合使用**：grok 负责 runtime capability/child subset，OpenCode 负责 wire content support |
| Session model switch | 检查 active agent/harness compatibility；已有 Turn 不兼容则拒绝；zero-turn 可 rebuild；reasoning effort 有 gate；watch generation 防竞态 | `session.switchModel` 追加 durable `ModelSwitched` event，projector 更新 session；没有同等 harness compatibility/rebuild 规则 | **grok 明显更强，直接作为 Lime switch 语义参考** |
| Variant/模型级覆盖 | per-model config override、reasoning/compaction/backend search 等配置丰富 | `Model.Info.variants` + provider/model request body/header overlay，适合 profile/variant | OpenCode 的 variant overlay 值得引入；必须写入 effective Turn options，不放 Renderer |
| Stream reducer | `xai-grok-sampler` 对 message/content block/usage/stop/error 有稳定 reducer，未知 stop reason 保留 | 多协议 parser + `LLMEvent`，native/AI SDK 两种 runtime 都归一化为同一 event stream | **OpenCode 胜出广度；grok 的 unknown/partial stream 保护保留** |
| Retry | retry 分类、退避和 compaction retry 有清晰边界 | `SessionRetry` 支持 retry-after、指数退避、5xx/429/rate-limit 识别 | 两者都可用；OpenCode 的 HTTP header 处理可借鉴 |
| Circuit breaker | 独立 `xai-circuit-breaker` crate，有 open/half-open/close、并发 probe、observer | 当前主要是 per-request retry，未见同等独立 breaker 控制面 | **grok 胜出，作为 provider transport 健康状态参考** |
| Tool capability filtering | explicit `CapabilityMode`、穷举 ToolKind、parent-child subset | LLM tool definitions、tool call/result、repairToolCall；工具能力更多在 session/tool registry | grok 负责“谁能看到/调用什么”；OpenCode 负责“如何编码 tool call/result” |
| Policy | 主要聚焦 model allow/disabled/hidden 与 session capability | `provider.use` policy 独立于 provider config，支持 wildcard、文档层级和最后匹配 | **OpenCode 胜出，作为 provider policy 边界参考** |
| 测试形态 | Rust 单测对 refresh、model switch、capability exhaustive、retry/breaker 覆盖好 | Effect/TypeScript 测试对 catalog、route、variant、protocol lowering、session projector 覆盖好 | 两者都保留测试思想；Lime 需在 Rust current owner 重建 |
| 对 Codex 主链侵入 | 更容易只取 model control，不带独立 UI/transport | 容易把 AI SDK、Effect runtime、Session message store 一并带入，造成第二 runtime | **grok 更适合主参考；OpenCode 只能抽取 provider wire 机制** |

## 为什么不是“OpenCode 全面替代 grok-build”

OpenCode 的 Catalog 和 LLM 边界非常适合多 provider，但它的 session model switch 主要是：

```text
session.switchModel
  -> publish ModelSwitched event
  -> projector 更新 session.model
  -> 下一次 runner resolve model
```

它没有 grok-build 那样的 active agent/harness compatibility、zero-turn rebuild、model switch generation 和 child capability subset。直接把 OpenCode switch 语义搬入 Lime，会重新制造本轮已经暴露的 provider/model 恢复缺口。

OpenCode 的 Catalog 还包含 Location plugin、Integration、Effect service 和 AI SDK runtime。Lime 若整体迁入，会把 provider catalog、App Server、Agent loop 和 Electron bridge 连接成第二个服务图，违背 Codex-first owner 规则。

## 为什么也不是“grok-build 全面替代 OpenCode”

grok-build 的模型控制非常适合 Rust runtime，但其采样类型和协议实现明显带有 xAI/Anthropic 业务背景。Lime 当前需要 OpenAI Responses、OpenAI Chat Completions、Anthropic Messages、Gemini、图片和 sidecar 多模态，直接复制 grok wire 会产生：

- provider-specific content 类型渗入 `agent-runtime`；
- 多协议 lowering 重复实现；
- 媒体/工具结果 capability 判断分散在 provider adapter；
- stream event 到 canonical Item 的 reducer 重新分叉。

OpenCode 的 provider-neutral `LLM.ContentPart`、endpoint union、protocol-specific lowering 和 shared content support 检查正好覆盖这些风险，但 Lime 必须将其翻译为 Rust `model-provider` 类型，不能引入 TypeScript Effect runtime。

## Lime 最终采用矩阵

| Lime 模块 | 主参考 | 辅助参考 | 不复制 |
| --- | --- | --- | --- |
| `model-provider::canonical` | grok `CanonicalModel` | OpenCode `Model.Info` / `LLM.ContentPart` | OpenCode SDK/Effect 类型本身 |
| `model-provider::runtime_provider` | grok `ModelsManager`、`ModelFetchAuth` | OpenCode Catalog available/default | grok xAI auth/session 产品 |
| `app-server/runtime_backend/model_*` | grok selection/readiness/switch | OpenCode policy/provider availability | OpenCode HTTP session handler |
| `model-provider::lowering` | Lime existing Rust lowering | OpenCode endpoint/protocol modules | provider raw payload 进入 runtime |
| `model-provider::provider_stream` | grok reducer 的 unknown/partial/error 语义 | OpenCode 多协议 `LLMEvent` 生命周期 | 第二套 item projection |
| `tool-runtime` | grok `CapabilityMode`/typed context | OpenCode ToolDefinition/ToolCall/ToolResult | AI SDK tool executor |
| retry/health | grok circuit breaker | OpenCode Retry-After/backoff | breaker 状态进入 Thread/Turn/Item |
| provider policy | Lime current policy owner | OpenCode wildcard/last-match 规则 | 让 provider config 自己决定 policy |

## 第二轮差异：不能只比较 catalog 和 wire

第二轮审计发现，grok-build 与 OpenCode 的真正差异还包括“有效请求选项”和身份边界：

| 维度 | grok-build | OpenCode | Lime 裁决 |
| --- | --- | --- | --- |
| 有效请求配置 | `SamplerConfig`/`ModelEntryConfig` 固化 auth、headers、context、timeout、retry、stream tool、compaction 与 auxiliary model | `projectModel` 合并 provider/model API、headers/body、variant/options | 在 `model-provider` 生成不可变 `EffectiveModelOptions`；Renderer/App Server 不重复 merge |
| Provider availability | catalog 与 auth method/identity 绑定，失败保留旧 catalog | `Integration`/credential policy 与 catalog 分离，显式 default 后再选 available model | 采用 grok auth-aware refresh + OpenCode policy 分离；enabled、credential、endpoint、model match 全部通过才 ready |
| Variant/reasoning | per-model override、reasoning effort、backend search、compaction/session summary/image description route | variants、reasoning effort/toggle/budget、interleaved reasoning、body/header overlay | 统一 `ModelVariant`/`ReasoningOptions`/auxiliary route；未实现字段明确列为产品范围排除 |
| Cost/quota/identity | model/account/deployment 可参与 route 与 catalog cache | cost tiers、limits、status/release 与 provider/model info | `ResolvedModelRoute`、usage、breaker、cache key 同时绑定 credential/tenant/account identity；不能只存 usage_count/error_count |
| 多凭证健康 | retry/breaker 可按 provider/endpoint/credential 细分 | credential 由 Integration 应用，request retry 较轻 | API Key Provider 的 round-robin 必须接 provider health/breaker、429/401 cooldown、quota/reset、credential generation |
| 未知能力 | 显式 capability/ToolKind subset | `supportsContent` 做 wire 能力校验 | 不采用 Lime 现有 heuristic/default-true；未知 capability 直接 typed RouteFailure |

## 对 v1 方案的修正

原先“多模型完全参考 grok-build、OpenCode 彻底退出”的表述不够精确，应改成：

> grok-build 是 Lime 多模型控制平面的 primary reference；OpenCode 是 provider wire、canonical content/lowering 和多协议 stream 的 secondary reference。Codex 仍是 runtime、Thread/Turn/Item、App Server、持久化、工具生命周期和恢复的唯一主参考。

这不是三套 runtime 并存，而是一个 `model-provider` owner 内的两类输入：grok 约束控制面，OpenCode 约束 wire 面。

第二轮的负向结论同样明确：不能把 OpenCode 的 Effect session/catalog 或 grok 的 xAI
auth/session/tool server 直接带进 Lime；它们只能被翻译成 `model-provider` 的 typed
catalog、effective options、lowering 和 normalized stream。Codex 的 Thread/Turn/Item、
ThreadStore、App Server 和恢复语义仍拥有最终裁决权。
