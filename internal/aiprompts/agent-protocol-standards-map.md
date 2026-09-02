# Agent 协议与标准地图

本文是 Lime 中所有 Agent 相关开发协议、外部标准仓库与未来可拆标准的索引。它回答三个问题：

1. 哪些语义已经从 Lime 实现中提炼成公共标准？
2. Lime 内部继续保留哪些开发协议和工程边界？
3. 未来哪些方向值得继续拆成独立标准仓库？

本文只做事实源导航，不替代各标准本体。新增标准或调整归属时，先更新本文，再更新对应 PRD / roadmap / aiprompts 文档。

Codex 是 Lime Agent runtime 的唯一实现参考。本文列出的 Agent Evidence、in-toto、SLSA、PROV 等内容只保留为外部或历史参考，不得成为 current runtime owner，也不得驱动 portable signed receipt、签名 artifact 或新的 trust protocol。

## 分层原则

| 层级 | 目标 | 放在哪里 | 判断标准 |
| --- | --- | --- | --- |
| 公共标准协议 | 跨产品、跨实现复用的 Agent 语义 | 独立标准仓库与 GitHub Pages | 不依赖 Lime 私有实现，能被其他 Agent 产品采用 |
| Lime 开发协议 | Lime 当前实现的命令、运行时、目录、UI、治理边界 | `internal/aiprompts/`、`internal/roadmap/`、代码契约测试 | 直接约束 Lime 源码、GUI、Electron/App Server/Rust/React 主链 |
| 路线图候选 | 已在 Lime 中出现稳定模式，但还没足够独立 | `internal/roadmap/` | 有多处实现压力，但标准边界仍需验证 |
| 外部参考标准 | 已存在的行业协议或生态规范 | 本文引用与各标准 `research-sources` | 只作为对齐对象，不复制其产品边界 |

一句话：**公共标准定义可移植语义，Lime 开发协议定义本仓库如何实现这些语义。**

## 已拆出的公共标准

| 标准 | 当前定位 | Lime 中的来源压力 | 公开链接 | Lime 对应文档 |
| --- | --- | --- | --- | --- |
| Agent Knowledge | Agent-readable knowledge packs；源材料、知识包、编译视图、引用与维护状态 | Knowledge v2、Builder Skill、document-first KnowledgePack、Resolver 主链 | [site](https://limecloud.github.io/agentknowledge/) / [llms-full](https://limecloud.github.io/agentknowledge/llms-full.txt) / [repo](https://github.com/limecloud/agentknowledge) | `../roadmap/knowledge/prd-v2.md`、`../roadmap/knowledge/prd-v2-diagrams.md` |
| Agent UI | Agent interaction surfaces；composer、message parts、status、tool UI、task capsule、artifact workspace、timeline/evidence | Workspace projection、Agent chat、artifact workspace、human-in-the-loop、timeline evidence | [site](https://limecloud.github.io/agentui/) / [llms-full](https://limecloud.github.io/agentui/llms-full.txt) / [repo](https://github.com/limecloud/agentui) | `../roadmap/agentui/README.md`、`workspace.md`、`design-language.md` |
| Agent Runtime | Agent execution runtime；events、control plane、tasks、tools、permissions、sandbox、remote channels、replay refs | Query loop、task lifecycle、tool execution、subagents、remote runtime、model routing、snapshots | [site](https://limecloud.github.io/agentruntime/) / [llms-full](https://limecloud.github.io/agentruntime/llms-full.txt) / [repo](https://github.com/limecloud/agentruntime) | `query-loop.md`、`task-agent-taxonomy.md`、`remote-runtime.md`、`state-history-telemetry.md` |
| Agent Evidence | Evidence, provenance, verification, review, replay, redaction, telemetry correlation, export manifests | Harness evidence pack、requestTelemetry、review/replay/export、source grounding、artifact review | [site](https://limecloud.github.io/agentevidence/) / [llms-full](https://limecloud.github.io/agentevidence/llms-full.txt) / [repo](https://github.com/limecloud/agentevidence) | `harness-engine-governance.md`、`state-history-telemetry.md`、`persistence-map.md` |
| Agent Policy | Policy decisions, approvals, permission grants, risk scopes, redaction, retention, waivers, policy traces | human approval、sandbox、retention、risk、waiver、permission bridge 横跨 runtime/evidence/UI | [site](https://limecloud.github.io/agentpolicy/) / [llms-full](https://limecloud.github.io/agentpolicy/llms-full.txt) / [repo](https://github.com/limecloud/agentpolicy) | `quality-workflow.md`、`remote-runtime.md`、`harness-engine-governance.md` |
| Agent Artifact | Durable deliverables；artifact identity、parts、versions、diffs、render/export manifests、source links、handoff packages | FileArtifact、sidecar、version、checkpoint、snapshot refs、artifact workspace、evidence refs | [site](https://limecloud.github.io/agentartifact/) / [llms-full](https://limecloud.github.io/agentartifact/llms-full.txt) / [repo](https://github.com/limecloud/agentartifact) | `persistence-map.md`、`../roadmap/artifacts/roadmap.md`、`../roadmap/agentui/README.md` |
| Agent Tool | Tool declarations；tool surfaces、input/output contracts、execution/permission profiles、invocation lifecycle、progress、result refs、errors | tools、MCP、site adapter、native skill、browser assist、capability matrix、permission profile、tool timeline | [site](https://limecloud.github.io/agenttool/) / [llms-full](https://limecloud.github.io/agenttool/llms-full.txt) / [repo](https://github.com/limecloud/agenttool) | `skill-standard.md`、`site-adapter-standard.md`、`mcp.md`、`command-runtime.md`、`../roadmap/warp/capability-matrix.md` |
| Agent Context | Context surfaces；context items、source refs、selection/ranking、budget/window、assembly/injection、compaction/summary、missing context | Memory / Compaction、Prompt Foundation、Query Loop 输入快照、Knowledge selection、tool-visible context | [site](https://limecloud.github.io/agentcontext/) / [llms-full](https://limecloud.github.io/agentcontext/llms-full.txt) / [repo](https://github.com/limecloud/agentcontext) | `memory-compaction.md`、`prompt-foundation.md`、`query-loop.md`、`state-history-telemetry.md` |

## 友链与外部对齐

| 外部标准 / 项目 | Lime 采用方式 | 注意边界 |
| --- | --- | --- |
| [Agent Skills](https://agentskills.io/) | 作为技能包格式、语法风格与 AI-friendly 文档的关键参考 | Agent Skills 是包格式参考，不等于 Lime runtime、目录、UI 或分发协议 |
| [Model Context Protocol](https://modelcontextprotocol.io/specification) | 作为 tools / resources / prompts / roots 的标准能力接入参考 | MCP tool schema 不替代 Lime 的 skill/catalog/runtime binding |
| [Agent2Agent Protocol](https://github.com/a2aproject/A2A) | 作为 peer agent task、message、artifact、native id 的远程协作参考 | A2A peer ids 应保留为 refs，不应吞并本地 Agent Runtime identity |
| [OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) | 作为 traces、spans、GenAI operation names 与 telemetry correlation 参考 | Telemetry 解释运行行为，Evidence 解释信任关系，不能合并 |
| [CloudEvents](https://github.com/cloudevents/spec/blob/main/cloudevents/spec.md) | 作为 portable event envelope 的参考 | 只借鉴 envelope，不强制所有 Lime 事件都用 CloudEvents |
| [W3C PROV](https://www.w3.org/TR/prov-dm/Overview.html) | 作为 provenance 的 entity / activity / agent 模型参考 | 不直接把 Agent runtime 压成通用 provenance 图而丢失执行语义 |
| [W3C Web Annotation](https://www.w3.org/TR/annotation-model/) | 作为 source selector、text position、target anchoring 参考 | UI citation 不是 evidence 本体，selector 要归入 Source Map |
| [in-toto Attestation](https://github.com/in-toto/attestation) / [SLSA Provenance](https://slsa.dev/spec/v1.1/provenance) | 作为 signed metadata、provenance predicate、export trust 的参考 | 适合 export/signature，不替代 Agent Evidence 的 claim map |
| [OpenLineage](https://openlineage.io/docs/spec/facets/) | 作为 run/job/dataset/facet lineage 的参考 | 数据 lineage 可参考，不替代 Agent task/runtime lifecycle |
| [CycloneDX](https://cyclonedx.org/specification/overview/) | 作为 claims、evidence、counter-evidence、attestation、confidence 的 audit packaging 参考 | 适合 audit vocabulary，不替代 Agent Evidence pack |

## Lime 内部开发协议与事实源

| 内部协议 / 标准 | 事实源 | 负责内容 | 不应负责 |
| --- | --- | --- | --- |
| Query Loop | `query-loop.md` | turn 提交、prompt 组包、runtime 主链、执行入口 | 公共 runtime 标准全文 |
| Agent UI / Runtime 标准边界 | `agent-ui-runtime-standard.md` | 四包职责、宿主接入、主聊天 projection 迁移分类、标准包守卫 | 公共 Agent UI / Agent Runtime 标准全文 |
| Command Runtime | `command-runtime.md` | `@` / `/` / 轻卡 / viewer / 功能方案包执行边界 | 新造独立命令协议 |
| Desktop Host / App Server Command Boundary | `commands.md` | `safeInvoke`、Electron host / preload、App Server protocol / client、legacy adapter、command catalog、mock 多侧同步 | UI 或业务语义定义 |
| Skill Standard | `skill-standard.md` | Skill 包解析、标准摘要层、runtime binding、目录投影 | 把 `SKILL.md` 原文当 runtime 协议 |
| Site Adapter Standard | `site-adapter-standard.md` | 站点适配器字段、来源导入、执行收敛 | 第二套浏览器 runtime |
| State / History / Telemetry | `state-history-telemetry.md` | session/thread/request/evidence/history 事实链 | Evidence 标准的全部 review 语义 |
| Harness Engine Governance | `harness-engine-governance.md` | evidence pack、replay、analysis、review、GUI 导出事实源 | 通用 Agent Evidence 标准仓库 |
| Task / Agent Taxonomy | `task-agent-taxonomy.md` | task、agent、coordinator、subagent、scheduler taxonomy | 将 Agent task 再拆成独立 runtime 外协议 |
| Remote Runtime | `remote-runtime.md` | remote peer、resume cursor、permission bridge、channel lifecycle | 直接实现 A2A 全协议 |
| Memory / Compaction | `memory-compaction.md` | working memory、durable memory、context compaction、missing context | Knowledge pack 标准 |
| Persistence Map | `persistence-map.md` | FileArtifact、sidecar、version、checkpoint、snapshot refs | Artifact 标准全文 |
| Quality Workflow | `quality-workflow.md` | 本地校验、GUI smoke、contracts、风险分类 | 产品标准定义 |

当前 Agent UI / Runtime 的宿主复用验收链路固定为 `AgentRuntimeClient -> projectAgentUiState -> AgentUiProjectionView`；公共标准只定义可移植语义，Lime 实现必须在 `agent-ui-runtime-standard.md` 中继续收敛到这条链路。

## 未来可拆标准候选

这些方向已经在 Lime 中出现稳定压力，但是否独立成公共标准，取决于是否能脱离 Lime 私有实现并服务其他 Agent 产品。

| 候选标准 | 建议名称 | 为什么值得拆 | 当前 Lime 事实源 | 拆分状态 |
| --- | --- | --- | --- | --- |
| Evaluation / Benchmark | Agent Evaluation | acceptance scenarios、provider E2E、quality review、harness evals、rubric 与 evidence 可复用 | `../test/harness-evals.md`、`../roadmap/knowledge/completion-audit-20260508.md` | 候选，适合从 Evidence 扩展 |
| Workflow / Scene | Agent Workflow | ServiceSkill scene、功能方案包、multi-step content workflow、browser-grounded scene 需要 portable workflow facts | `command-runtime.md`、`task-agent-taxonomy.md` | 中期候选 |
| Model Routing / Economy | Agent Model Routing | provider registry、model profile、cost、quota、fallback、task profile 有通用标准价值 | `providers.md`、`credential-pool.md`、`../roadmap/task/model-routing.md` | 中期候选 |
| Connector / Channel | Agent Channel | remote runtime、A2A peer、webhook、deep link、browser session、desktop bridge 都是跨系统通道 | `remote-runtime.md`、`../content/08.open-platform/4.connect.md` | 中期候选 |
| UI Theme / Design Tokens | Agent Design Surface | Agent UI 定义语义，不定义视觉；但跨 Agent 产品可能需要 design token / accessibility / density 标准 | `design-language.md` | 低优先级，除非多端复用稳定 |

## 拆分判断标准

一个 Lime 内部协议要升级为公共标准，应同时满足：

1. **跨实现**：至少能被两个不同 runtime、UI、产品或服务采用。
2. **可移植**：不依赖 Lime 路径、宿主命令、内部组件名或私有服务。
3. **边界清晰**：能一句话说清“它拥有何种事实，不拥有何种事实”。
4. **有 schema / events / acceptance**：不只是散文，需要最小机器可验证结构。
5. **能保留原生 id**：对外部协议只做 refs 与映射，不吞并其身份模型。
6. **不会制造第二套 current 主链**：Lime 内部仍只有一个 current 实现入口。

不满足以上条件时，先留在 `internal/aiprompts/` 或 `internal/roadmap/`，不要急着开新仓库。

## 新增或引用标准的流程

1. 先在 Lime 中定位事实源：代码、roadmap、aiprompts、验收脚本。
2. 写清楚 owner：标准拥有的对象、事件、schema、边界和不负责事项。
3. 对照外部标准：只引用，不照搬产品边界。
4. 建立 `llms.txt` / `llms-full.txt`、public schemas、版本快照与 release notes。
5. 回到本文登记：公开链接、Lime 事实源、当前状态、未来关系。
6. 如果影响 Lime 实现，再同步对应开发协议与契约测试。

## 一句话地图

```text
Agent Knowledge -> 给 Agent 什么可信资料
Agent Runtime   -> Agent 工作如何执行和恢复
Agent UI        -> Agent 工作如何被看见和控制
Agent Evidence  -> Agent 结果为什么可信、如何审计和回放
Agent Policy    -> Agent 权限、风险、保留、豁免如何统一表达
Agent Artifact  -> Agent 生成物如何版本化、编辑、导出和交接
Agent Tool      -> Agent 能力如何被声明、授权、调用和审计
Agent Context   -> Agent 当前可用、选择、组装、压缩、缺失并注入了什么上下文
```
