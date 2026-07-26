# Lime 记忆与上下文治理目标架构

> 状态：current architecture plan
> 更新时间：2026-06-19
> 目标：以文件化 memory store 为唯一记忆事实源，App Server 提供受控工具、context contributor、上下文防腐和压缩主链，Soul 作为独立交互配置保留，SQLite 只保留状态、压缩 checkpoint 与清理边界。

## 1. 架构原则

### 1.1 单事实源

记忆内容只认：

```text
memory store folder
```

默认上下文只认：

```text
memory_summary.md
```

按需读取只认：

```text
memory tools -> MemoryBackend
```

交互身份只认：

```text
memory.soul -> MemorySoulConfig
```

派生索引只认：

```text
index/ 可删除、可重建、可降级
```

模型可见上下文只认：

```text
ContextPacket -> ContextAssembler -> provider adapter
```

会话压缩只认：

```text
session history -> ContextCompaction -> compaction artifact -> next-turn context packet
```

### 1.2 读写分离

```text
read path:
  memory_summary.md -> context contributor
  MEMORY.md / rollout_summaries -> memory tools
  memory.soul -> interaction contributor / generation brief
  compaction artifact -> session context contributor

write path:
  explicit user request -> ad-hoc note
  handoff / replay / analysis / review decision save -> rollout_summaries candidate
  explicit memoryStore/consolidate -> MEMORY.md / memory_summary.md
  Soul editor / SOUL.md import -> memory.soul config
  manual / overflow compact -> compaction artifact
```

写入不得在同一 turn 里立即改变当前 prompt。Soul 的 artifact voice 只影响明确的 generation brief，不回写长期记忆。
压缩不得改写历史消息，也不得自动回写 memory store。

### 1.3 数据库边界

SQLite 允许保存：

1. thread / turn 状态。
2. 非语义 memory mode / feature flag。
3. 旧实现清理标记。
4. derived index metadata。
5. reset / rebuild 记录。
6. context epoch、compaction checkpoint、tail start id 和 summary token usage。

SQLite 不允许保存：

1. 下一代记忆 canonical content。
2. embedding BLOB 作为默认检索事实源。
3. 多套 provider 输出的 raw recall truth。
4. Soul artifact voice 的长期记忆副本。
5. provider-specific prompt 拼装结果。
6. 被压缩摘要替代的历史正文副本。

## 2. 目标目录结构

```text
memories/
  memory_summary.md
  MEMORY.md
  rollout_summaries/
    <timestamp>-<session-export-kind>.md
  skills/
    <skill-name>/
      SKILL.md
      references/
  extensions/
    ad_hoc/
      notes/
        <timestamp>-<slug>.md
  index/
    manifest.json
    local-search/
  context/
    packets/
    compactions/
    epochs/
```

规则：

1. 目录根通过统一 path resolver 获取。
2. workspace 级记忆与全局记忆必须可区分。
3. 工具层只暴露相对路径。
4. hidden path、symlink、path traversal 默认拒绝。
5. `index/` 可被删除并重建，不影响记忆本体。
6. `SOUL.md` 是导入 / 复制快照，不是 memory store 的 canonical 文件。
7. `context/` 只保存会话上下文治理 artifact，不参与 memory search canonical source。
8. compaction artifact 如需长期沉淀，必须复制为 `rollout_summaries/` 候选后再显式整理。

## 3. 核心接口

```rust
trait MemoryBackend: Clone + Send + Sync + 'static {
    fn list(&self, request: ListMemoriesRequest) -> Future<Result<ListMemoriesResponse>>;
    fn read(&self, request: ReadMemoryRequest) -> Future<Result<ReadMemoryResponse>>;
    fn search(&self, request: SearchMemoriesRequest) -> Future<Result<SearchMemoriesResponse>>;
    fn add_ad_hoc_note(&self, request: AddAdHocMemoryNoteRequest) -> Future<Result<AddAdHocMemoryNoteResponse>>;
    fn write_rollout_summary(&self, request: RolloutSummaryWriteRequest) -> Future<Result<AddAdHocMemoryNoteResponse>>;
    fn consolidate(&self, request: ConsolidateMemoryRequest) -> Future<Result<ConsolidateMemoryResponse>>;
    fn list_review_notes(&self, request: ReviewListRequest) -> Future<Result<ReviewListResponse>>;
    fn resolve_review_note(&self, request: ReviewResolveRequest) -> Future<Result<ReviewResolveResponse>>;
    fn health(&self, request: MemoryRootRequest) -> Future<Result<MemoryHealthResponse>>;
    fn reset(&self, request: MemoryResetRequest) -> Future<Result<MemoryResetResponse>>;
    fn rebuild_index(&self, request: MemoryRootRequest) -> Future<Result<MemoryIndexRebuildResponse>>;
}
```

上下文治理合同：

```rust
struct ContextPacket {
    id: ContextPacketId,
    source: ContextSource,
    scope: ContextScope,
    role: ContextRole,
    trust_level: TrustLevel,
    cache_key: Option<String>,
    invalidation: InvalidationPolicy,
    token_budget: TokenBudget,
    body: ContextBody,
    citations: Vec<ContextCitation>,
    sensitivity: SensitivityFlags,
}

trait ContextContributor: Clone + Send + Sync + 'static {
    fn collect(&self, request: ContextCollectRequest) -> Future<Result<Vec<ContextPacket>>>;
}

trait ContextAssembler: Clone + Send + Sync + 'static {
    fn assemble(&self, request: ContextAssembleRequest) -> Future<Result<AssembledContext>>;
}

trait ProviderContextProfileResolver: Clone + Send + Sync + 'static {
    fn resolve(&self, request: ProviderContextProfileRequest) -> Future<Result<ProviderContextProfile>>;
}
```

固定规则：

1. contributor 只产出 packet，不直接拼 provider message。
2. assembler 负责 admission、排序、截断、去重、防腐扫描和 telemetry。
3. provider adapter 只能把 `AssembledContext` 转成目标模型消息格式，不能重新选择业务上下文。
4. 新增上下文来源必须新增 typed packet、预算测试和 E2E 证据。

P0 backend：

```text
LocalMemoryBackend
```

P2 可选 backend / index：

```text
IndexedMemoryBackend(LocalMemoryBackend + DerivedIndex)
RemoteMemoryBackend
```

任何 backend 必须保持同一输出合同，不能把 provider-specific 字段泄露给普通调用方。

## 4. 层级划分

### 4.1 Store Layer

职责：

1. 创建和校验 memory folder。
2. 读取 / 写入 memory 文件。
3. 路径 canonicalize 与越界防护。
4. reset 与 rebuild 的文件级操作。

### 4.2 Context Contributor

职责：

1. 读取 `memory_summary.md`。
2. 按 token 预算截断。
3. 生成 memory summary context packet。
4. 当 summary 缺失或为空时返回空。

不允许：

1. 在 prompt contributor 里搜索全量 memory。
2. 在 prompt contributor 里调用 embedding provider。
3. 注入 raw rollout summary。
4. 把 Soul artifact voice 写入 summary。

### 4.3 Context Anticorruption Layer

职责：

1. 统一接收 memory、Soul、project rules、workspace files、tool output、compaction artifact、attachments 和 runtime diagnostics 的 packet。
2. 为每个 packet 标记 source、scope、trust level、sensitivity、citation、cache key 和 invalidation policy。
3. 执行单 packet token 上限、总预算上限、secret-like 扫描、prompt injection 扫描、路径泄露降级和重复片段合并。
4. 把不合格 packet 送 review / diagnostic，而不是静默注入。

不允许：

1. 在 feature 代码里直接 append 原始 prompt 文本。
2. 用 provider adapter 绕过防腐层。
3. 把 diagnostic / raw tool output 默认注入。
4. 用自动截断掩盖 secret-like 内容。

### 4.4 Tool Contributor

职责：

1. 注册 `memory_list`、`memory_read`、`memory_search`、`memory_add_note`。
2. 将工具参数转换为 backend request。
3. 记录工具调用 telemetry。
4. 输出 JSON 与 citation 字段。

### 4.5 Soul Contributor

职责：

1. 读取保存后的 `memory.soul` 配置。
2. 为交互身份、沟通节奏和解释深度提供 prompt 片段。
3. 为 artifact voice 生成 `generation_brief_only` 元数据。
4. 支持 `SOUL.md` 导入预览、warning 和复制输出。

不允许：

1. 把 `SOUL.md` 文件路径当运行时事实源。
2. 把 artifact voice brief 写入长期记忆。
3. 让专家 persona 回写全局 Soul。
4. 重新接回旧 `companion_*` 桌宠命令链路。

### 4.6 Context Compaction Pipeline

入口：

1. 用户以 `{ threadId }` 显式调用 App Server exact `thread/compact/start`；请求被接受后立即返回 `{}`，后续状态只通过标准 Thread / Turn / Item lifecycle 投影。
2. `ContextBudgetPlanner` 检测到下一轮会超过模型 profile 的自动压缩阈值。

职责：

1. 基于 session history 生成 compaction artifact。
2. 保留 `contextEpoch`、`tailStartId`、summary token 数、使用模型和触发原因。
3. 下轮通过 session context contributor 以 packet 形式注入压缩摘要。
4. 保留最近尾部消息作为原文窗口，避免压缩摘要覆盖最新事实。
5. 写入 telemetry 和 GUI 证据，便于判断是否因小上下文模型触发。

不允许：

1. 重写历史消息。
2. 自动写 `MEMORY.md` 或 `memory_summary.md`。
3. 在 provider adapter 内私自做压缩。
4. 压缩失败时删除原始 thread / turn 状态。

### 4.7 Provider Context Profile

职责：

1. 从模型目录、用户配置和 provider 能力生成 `ProviderContextProfile`。
2. 统一提供 context window、max output、tool schema 限制、reasoning / summary 支持、cache 支持、streaming 终态和自动压缩阈值。
3. 为 `ContextBudgetPlanner` 输出本轮预算：output reserve、system / developer、tools、memory、Soul、project context、compaction summary、tail history。
4. 支持多模型切换时重新规划 packet admission。

不允许：

1. 在 memory store、Soul 或 GUI 里写 provider 专属 prompt 分支。
2. 切模型时改变长期记忆事实源。
3. 把 provider-specific 字段泄露给 memory tool 响应。

### 4.8 Consolidation Pipeline

入口：

1. 设置页或受控维护动作显式调用 `memoryStore/consolidate`。
2. 后续如增加调度器，也必须调用同一 current 边界，记录审计事件，且不得绕过 review 队列直接改写 summary。

职责：

1. 把 ad-hoc note 和 rollout summary candidate 整理进 `MEMORY.md`。
2. 生成或刷新 `memory_summary.md`。
3. 执行 secret / injection scan。
4. 把敏感、冲突或空正文送入 review 队列。
5. 把接受、拒绝和整理动作追加到文件化审计日志，便于追踪。

不允许：

1. 无用户确认时保存凭证或敏感原文。
2. 直接写旧 `unified_memory_*` 作为下一代事实源。
3. 把 consolidation 失败变成 turn 失败。
4. 从旧记忆表批量生成 current 文件。

### 4.9 Derived Index Layer

职责：

1. 对 memory folder 建派生索引。
2. 支持健康检查、删除、重建。
3. 索引缺失或损坏时降级到文本扫描。

候选实现类型：

1. 内嵌全文索引。
2. 嵌入式搜索索引。
3. 可选向量索引。
4. 远端索引适配器。

固定规则：这些都不是 P0 current truth，只是 P2 之后的派生检索实现。

## 5. 目标分类

### 5.1 `current`

1. `memory store folder`
2. `MemoryBackend`
3. `LocalMemoryBackend`
4. `memory_summary.md` prompt contributor
5. `memory tools`
6. `ad-hoc note` 写入队列
7. `memoryStore/consolidate` 显式整理入口
8. `extensions/ad_hoc/processed` / `extensions/ad_hoc/review` 归档边界
9. `rollout_summaries` 证据摘要与 runtime export 自动生成候选，且不自动入库
10. `memoryStore/review/list` / `memoryStore/review/resolve` 审阅边界
11. `audit/memory_events.jsonl` 文件化审计日志
12. `ContextPacket`
13. `ContextContributor`
14. `ContextAssembler`
15. `ContextBudgetPlanner`
16. `ContextAnticorruption`
17. `ContextCompaction`
18. `ProviderContextProfile`
19. `contextEpoch` / compaction checkpoint
20. `memory.soul` / `MemorySoulConfig`
21. `SOUL.md` 导入 / 复制快照
22. `memory.soul.artifact_voice` generation brief

### 5.2 `dead`

1. `inspiration_*` 新长期事实源。
2. `make-next-generation-more-like-me.md` 旧扩展路线图。
3. `unified_memory_*` 旧长期记忆主线。
4. `memory_runtime_*` 旧默认召回主线。
5. 旧 MemoryPage 灵感库 / 高级诊断混合视图。
6. 旧 `companion_*` 桌宠命令链路。
7. 旧 `lime-rs/src/**` 记忆路径引用。
8. 生产路径依赖 mock memory backend。
9. 生产路径直接拼 model-visible context 原始字符串。
10. provider-specific prompt builder 承接业务上下文选择。
11. 压缩摘要自动写长期记忆。
12. 历史消息重写式压缩。

## 6. 生命周期

### 6.1 Thread Start

```text
thread/start
  -> load memory config
  -> resolve memory store root
  -> ContextContributor collects memory_summary.md packet
  -> ContextContributor collects compaction packet
  -> load memory.soul config
  -> ProviderContextProfile resolves model budget
  -> ContextAssembler admits / truncates / rejects packets
  -> provider adapter renders messages
```

### 6.2 Tool Read

```text
model decides memory is relevant
  -> memory_search
  -> MEMORY.md hit
  -> memory_read exact path/line
  -> answer with memory citation metadata
```

### 6.3 Soul Use

```text
settings.memory.soul
  -> MemorySoulConfig
  -> interaction contributor
  -> expert persona inherits communication_rhythm only
  -> artifact voice produces generation_brief_only
```

### 6.4 Context Assemble

```text
turn/start
  -> resolve provider context profile
  -> collect packets from memory / Soul / project / compaction / tools
  -> run anticorruption checks
  -> reserve output and tool schema budget
  -> admit stable packets first
  -> trim large packets by source-specific policy
  -> emit AssembledContext + telemetry
```

### 6.5 Explicit Write

```text
user says remember / add note
  -> memory_add_note
  -> extensions/ad_hoc/notes/<timestamp>-slug.md
  -> consolidation pending
  -> explicit memoryStore/consolidate
  -> accepted note moves to extensions/ad_hoc/processed
  -> risky note moves to extensions/ad_hoc/review
  -> review accept moves to processed and updates summary
  -> review reject moves to rejected without updating summary
  -> audit/memory_events.jsonl records operation
  -> later MEMORY.md / memory_summary.md refresh
```

### 6.6 Compaction

```text
manual thread/compact/start { threadId } or overflow threshold
  -> manual request returns {}
  -> turn/started
  -> item/started { type: contextCompaction }
  -> read durable session history
  -> keep recent tail window
  -> summarize older window
  -> write compaction artifact + contextEpoch
  -> item/completed { type: contextCompaction }
  -> turn/completed
  -> next turn injects compaction packet
  -> optional rollout candidate only after explicit export
```

手动入口不保留旧 session-scoped RPC alias、双发或 fallback；自动阈值只复用同一 compaction lifecycle，不新增第二个 GUI 或 provider 入口。

### 6.7 Reset

```text
memory/reset
  -> clear memory folder
  -> clear derived index
  -> preserve thread / turn state
  -> preserve memory.soul by default
  -> optional future scope can reset memory.soul explicitly
```

### 6.8 Old Path Cleanup

```text
old unified/runtime memory entry
  -> reject new write
  -> fail-fast or retired guard
  -> no data import into memory store
  -> guard prevents reactivation
```

## 7. 第一刀边界

第一刀只需要完成：

1. 文档和事实源收口。
2. `MemoryBackend` 合同设计。
3. memory folder layout 设计。
4. `ContextPacket` / `ContextContributor` / `ContextAssembler` 合同设计。
5. prompt contributor 预算规则。
6. memory tools 参数和输出合同。
7. compaction 与 memory store 的分离边界。
8. 多模型 provider profile 预算规则。
9. Soul current 边界说明。
10. 旧路线图引用清理。

第一刀不需要实现：

1. 向量索引。
2. 外部 provider。
3. 恢复旧灵感库产品页。
4. 旧数据导入。
