# Lime 记忆与上下文治理路线图

> 状态：current planning source
> 更新时间：2026-06-19
> 主目标：把 Lime 记忆主线收敛到文件化 memory store，并把上下文防腐、上下文压缩、多模型预算、稳定摘要注入、专用记忆工具、backend 抽象和可重建检索层纳入同一条可验证主链；Soul 继续作为用户可编辑的交互身份与沟通节奏配置保留。

## 1. 本路线图回答什么

本目录只回答 Lime 记忆系统后续如何收敛：

1. 记忆事实源如何从数据库式长期对象收敛到可读、可审计、可版本化的 memory folder。
2. 运行时如何通过结构化 context packet 注入受控内容，而不是把历史、召回结果或诊断全量塞进 prompt。
3. 上下文防腐层如何标记来源、信任级别、预算、失效策略和敏感风险，阻止未审计文本直接进入模型可见上下文。
4. 上下文压缩如何作为 session context 治理动作保留，不替代长期记忆，也不重写历史事实源。
5. 多模型如何通过统一 provider context profile 分配上下文窗口、输出保留、工具 schema 预算、reasoning / summary 能力和压缩阈值。
6. Agent 如何通过 `memory_list` / `memory_read` / `memory_search` / `memory_add_note` 访问记忆。
7. 记忆读、写、整理、清空如何拆成独立边界。
8. Soul 如何作为 `memory.soul` 配置保留，并通过 `SOUL.md` 导入 / 复制和 generation brief 参与交互风格。
9. SQLite 如何只保留线程 / turn 状态、压缩 checkpoint、清理标记和非语义派生索引元数据；当前 reset 不再处理旧 memory stage data。
10. 如果未来需要向量或全文索引，如何作为可删除、可重建的派生索引接入，而不是成为第二套记忆真相。

## 2. 当前裁决

### 2.1 `current`

Lime 后续记忆主线只允许向这组边界收敛：

1. `memory store`：平台无关 App data / workspace scoped 记忆目录，由统一路径入口解析。
2. `memory_summary.md`：每轮默认可注入的稳定摘要，必须有 token 上限。
3. `MEMORY.md`：人工或显式 consolidation 后的记忆登记册，作为搜索入口。
4. `rollout_summaries/`：会话 / 运行证据摘要，只在被 `MEMORY.md` 指向时读取。
5. `extensions/ad_hoc/notes/`：用户或 Agent 显式添加的待整理备注。
6. `MemoryBackend` trait：统一承接 list/read/search/add note/consolidate，当前实现是本地文件系统。
7. `memory tools`：Agent 通过工具读取和搜索记忆，结果必须带 path/line/citation。
8. `ContextPacket`：所有模型可见外部片段的结构化 envelope，必须带 source、scope、role、tokenBudget、trustLevel、cacheKey 和 invalidation policy。
9. `ContextContributor`：memory summary、Soul、项目规则、压缩摘要、工具 schema、运行时状态等只能通过 contributor 进入上下文。
10. `ContextAssembler`：按 provider profile 和 turn budget 汇总 packet，执行硬上限、排序、截断、去重和防腐检查。
11. `ContextCompaction`：App Server exact `thread/compact/start` 负责手动触发 thread 上下文续接；params 为 `{ threadId }`，立即返回 `{}`，状态通过标准 `turn/*`、`item/*` lifecycle 投影；压缩摘要只作为 session artifact / context packet，不是 memory store truth。
12. `ProviderContextProfile`：从模型目录和 provider 能力解析 context window、output reserve、tool schema 限制、reasoning / summary 支持和 cache 策略。
13. `prompt contributor`：只把截断后的 summary / Soul / 压缩摘要等受控 packet 注入。
14. `memory.soul` / `MemorySoulConfig`：用户可编辑的交互身份、沟通节奏和 artifact voice 配置。
15. `SOUL.md`：`memory.soul` 的导入和复制快照；运行时使用保存后的配置，不依赖该文件路径。

### 2.2 `dead`

这些内容本轮直接退出：

1. `internal/roadmap/memory/make-next-generation-more-like-me.md`
2. 新增 `inspiration_*` 平行长期事实源。
3. 旧 `unified_memory_*` 长期记忆主线。
4. 旧 `memory_runtime_*` 默认召回主线。
5. 旧 MemoryPage 灵感库 / 高级诊断混合视图。
6. SQLite embedding BLOB 与全表余弦扫描。
7. active memory recall preview。
8. auto organization / dreaming 实验。
9. external memory provider。
10. 默认引入任何具体外部索引库或向量数据库。
11. 旧 `companion_*` 桌宠命令链路。
12. 把旧 `lime-rs/src/**` 或旧 Tauri command wrapper 写回记忆路线图。
13. 把外部 provider 或向量索引当成 memory current truth。
14. 任意生产代码把原始字符串直接 append 到 model-visible context。
15. provider-specific prompt 拼装绕过 `ContextAssembler`。
16. 用压缩摘要替代 memory store，或把压缩摘要自动回写长期记忆。
17. 自动改写历史消息来“节省上下文”。

## 3. Soul 边界

Soul 继续支持，但它不是长期记忆本体：

1. Soul 是 `memory.soul` 配置，负责交互身份、语气、解释深度、挑战方式、避免项和 artifact voice。
2. `SOUL.md` 是导入 / 复制快照，导入时保留 `project_rules`、`local_path`、`secret_like`、`too_long` 等风险提示。
3. artifact voice 只生成 `generation_brief_only`，不得写入 `MEMORY.md`、`memory_summary.md` 或长期记忆文件。
4. 专家 persona 可以继承全局 Soul 的 `communication_rhythm`，但不得回写全局 Soul。
5. Soul 与旧 `companion_*` 桌宠命令链路无关；后者按 `dead` 处理。

## 4. 核心工程形态

关键不是“搜索更高级”，而是边界清楚：

1. 记忆先是文件，不是黑盒数据库。
2. summary 小而稳定，默认注入；原文按需工具读取。
3. 搜索先做轻量文本匹配，命中后再读取精确行范围。
4. backend 是 trait，当前本地文件系统，后续可替换远端或索引实现。
5. 写入必须显式、可追踪、可审核；整理必须走显式 consolidation，不能直接污染当前 turn。
6. SQLite 只保存线程 / turn 状态、配置、清理标记、reset / rebuild 记录和非语义派生索引元数据，不做记忆语义事实源，也不作为当前 reset 的记忆内容对象。
7. 所有模型可见上下文都先进入 typed packet，再由 `ContextAssembler` 统一预算和防腐；禁止 feature 自己拼 prompt。
8. 上下文压缩是 session lifecycle 能力，保留原始历史和尾部窗口，不把压缩结果当长期记忆或项目知识。
9. 多模型差异只进入 provider profile 和 adapter，不允许散落到 memory tools、Soul、GUI 或单个功能 prompt。

## 5. 上下文治理边界

### 5.1 防腐层

防腐层的职责是隔离“外部文本”和“模型可见上下文”：

1. memory、project rules、workspace 文件、rollout candidate、tool output、user attachment、Soul 导入内容都先标记 source 和 trustLevel。
2. 未经过 contributor 的原始文本不得进入 prompt。
3. 每个 packet 都必须有 token 硬上限；单个 packet 超过 1k token 需要专项回归，超过 10k token 直接拒绝或转文件引用。
4. 敏感、prompt injection、路径泄露、secret-like 内容只能进入 review / diagnostic，不进入默认 context。
5. packet 必须能在日志或 GUI 证据里说明来源与截断状态，但不能把敏感原文摊开给普通用户。

### 5.2 压缩层

压缩层只处理 session context：

1. 手动压缩只走 App Server exact `thread/compact/start`，wire params 只能是 `{ threadId }`；旧 session-scoped method 不保留 alias、双发或 fallback。
2. 请求被接受后立即返回 `{}`；压缩进度只通过标准 `turn/started`、`item/started` / `item/completed` 的 `contextCompaction` item 和 `turn/completed` 投影。
3. 自动溢出压缩由 runtime threshold 触发并复用同一 compaction lifecycle，不由 GUI 或 provider adapter 另开入口。
4. 压缩不得重写历史消息，只能生成新的 compaction artifact，并记录 `contextEpoch` / `tailStart` / `summaryTokens`。
5. 压缩结果进入后续 turn 的 context packet，保留最近尾部消息作为原文窗口。
6. 压缩结果如需长期沉淀，只能先写 `rollout_summaries/*.md` 候选，再由显式 consolidation 审核进入 memory store。
7. 压缩失败不能污染 memory store，不能让 reset 误删 thread / turn 状态。

### 5.3 多模型预算层

多模型支持只认统一 profile：

1. provider / model catalog 提供 context window、max output、tool schema 支持、reasoning summary、cache 和 streaming 能力。
2. `ContextBudgetPlanner` 为每轮预留 output、tool schema、developer policy、memory summary、Soul、project context、compaction summary 和 tail history 预算。
3. 切换模型只重新规划 packet admission，不改变 memory store、Soul 或 session 历史事实源。
4. provider adapter 只负责 schema / message shape 转换，不拥有业务 context 选择权。
5. GUI 和 E2E 必须能证明小窗口模型会触发压缩或降级，而大窗口模型不会无界注入更多内容。

## 6. 文件分工

1. [prd.md](./prd.md)：产品与工程需求，定义 P0/P1/P2 和非目标。
2. [architecture.md](./architecture.md)：目标架构、事实源分类、backend 与数据生命周期。
3. [diagrams.md](./diagrams.md)：读路径、工具路径、写路径、Soul 路径和清理路径图。
4. [rollout-plan.md](./rollout-plan.md)：分阶段实施计划、验证和退出条件。
5. [acceptance.md](./acceptance.md)：验收标准和不通过判定。

## 7. 必须避免

1. 为了补性能，直接把向量库接成记忆事实源。
2. 在旧记忆或旧灵感库上继续追加新一代记忆语义。
3. 让前端页面自己扫描磁盘或拼装 prompt。
4. 把 raw recall、外部 provider、active recall 当默认能力。
5. 让自动候选或定时整理绕过显式 consolidation 并影响当前生成。
6. 把 Soul 错判为旧 companion 桌宠链路。
7. 在文档或实现里恢复 `lime-rs/src/**` 旧路径。
8. 为每个 provider 单独拼 prompt 或单独实现压缩逻辑。
9. 让上下文预算只存在配置里，没有运行时硬上限和 E2E 证据。

## 8. 这一步如何服务主线

本轮收口把 memory 路线图从“旧灵感库 + 旧记忆实现延续”改成单一文件化记忆主线，并把上下文防腐、上下文压缩和多模型预算纳入同一条 current 规划，避免后续在每个 provider、每个页面或每个工具里再次拼出平行 prompt 链路。`MemoryBackend` / 本地 memory store / `memoryStore/*` 前端网关已接入 App Server current 边界。`memory_summary.md` 已通过 RuntimeCore turn start 读取并以受控 prompt contributor 注入；Soul 已作为保存后的 `memory.soul` current 配置注入受控交互片段；`memory_list` / `memory_read` / `memory_search` / `memory_add_note` 已注册为 App Server runtime native tools，并真实调用 current `MemoryAppDataSource`。`memoryStore/consolidate` 已提供显式整理入口：只处理 current 文件化 ad-hoc notes 与 rollout summary candidates，接受项归档到 processed 并追加 `MEMORY.md` / `memory_summary.md`，敏感或冲突项归档到 review，不迁移旧数据。`memoryStore/review/list` / `memoryStore/review/resolve` 已补审阅闭环：设置页可接受或拒绝 review note，接受项复用 consolidation 写入语义，拒绝项只归档到 rejected。handoff bundle、replay case、analysis handoff 与保存后的 review decision 已自动写入 workspace scoped `rollout_summaries/*.md` 候选，记录 source、exportedAt、export root 与 referenced artifacts；候选不会直接改长期记忆，只有显式整理后才进入 `MEMORY.md` / `memory_summary.md`。设置页已通过默认工作区 root 读取 workspace scoped `rollout_summaries/`，展示候选来源、导出位置和相关交付物，并提供显式整理候选按钮；前端仍只走 `memoryStore/list` / `memoryStore/read` / `memoryStore/consolidate` current 网关，不直接扫描磁盘。旧 MemoryPage 灵感库、旧 `src/lib/api/memory.ts` 聚合网关、旧 `project_memory_get` Electron Host facade、旧 `unifiedMemory` 网关、旧 `memoryRuntime` 召回网关和旧 memory crate 已进入 `dead / deleted / forbidden-to-restore`。

真实 Electron GUI smoke 已覆盖设置页记忆主路径：日常记忆面板、运行摘要候选刷新、显式整理、审阅刷新、重建索引、Soul 面板和高级导入 / 复制入口均走 current 桌面链路。下一刀只在补充更多 export 类型候选策略时继续走文件化 memory store 与显式 consolidation，并补对应写入测试和 GUI 可见性回归，不恢复旧记忆或旧灵感库入口。
