# Codex 渲染对齐重构 v2

状态：proposed / implementation-ready

日期：2026-07-28

## 1. 目标

v2 把 Lime 的对话界面重构为 Codex App Server v2 的完整语义投影。目标不是把协议对象压缩为旧 Message、ToolCall 或活动文本，而是让每一个当前产品范围内的上游事实都有唯一、可恢复、可测试的显示出口：

1. Turn 内保持原序的 Item 时间线；
2. Turn 级计划、文件变更、用量和终态面板；
3. Composer 上方唯一的阻塞交互层；
4. Header、状态区和应用通知；
5. 受控的开发诊断视图。

任何支持的未知 Item 或 notification 都必须 fail visible：显示类型、关联 identity 和脱敏后的字段名摘要，并记录协议漂移。不得静默丢弃，也不得使用 raw JSON 或文本正则伪造业务状态。

本轮只重构渲染投影、其必要的协议/bridge/read-model 接线和已经重复的旧前端降级路径；不新增第二套 Agent runtime，不把 Renderer 变成 provider 或工具执行 owner。

## 2. 参考与裁决

| 领域                                                      | 主参考                                         | Lime current owner                                           | v2 裁决                                                                       |
| --------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| Thread、Turn、Item、通知、reverse request、恢复           | Codex 4c43465133428898aa84f0bfc02c306ed65fb66a | app-server-protocol、app-server、thread-store、agent-runtime | 按 Codex 重构并删除重复投影                                                   |
| Item 历史形态、流式渲染和活动单元                         | Codex tui/history_cell 与 tui/streaming        | React 对话时间线                                             | 复制语义，不复制 TUI 外观                                                     |
| 多模型 catalog、route、capability、switch、retry/breaker  | grok-build                                     | model-provider                                               | 保持 v1 的 Grok 控制面裁决，不迁移为 Codex 单模型假设                         |
| provider wire、canonical content、media 和多协议 lowering | OpenCode                                       | model-provider                                               | 保持 v1 的 OpenCode wire 裁决，不把 provider raw 类型带入 runtime 或 Renderer |
| 页面信息架构与全量覆盖清单                                | LimeShot internal/roadmap/xuanlan              | 本目录                                                       | 已迁入并按 Lime current owner 改写                                            |

多模型和多模态边界不可被本次 Codex 对齐改变：

- ResolvedModelRoute、EffectiveModelOptions、capability snapshot、provider readiness、retry/circuit breaker 仍唯一归 model-provider。
- 文本、图片、音频、工具结果和 reasoning 的 provider lowering 仍由 model-provider 的 canonical content 到 provider wire 处理；OpenCode 只提供这一层的参考。
- Codex Item 只携带已归一化、可持久化的用户可见事实。Renderer 不检查 provider 名称猜测能力，不解码 provider raw media，不自行选择模型或 fallback。
- 删除范围只限旧的渲染合成、重复 projection 和已经脱离 current build graph 的前端路径；不得删除多模型、多模态 current owner、schema、lowering 或其回归。

本目录从下列材料迁入并替换 LimeShot 产品名与旧实现假设：

- /Users/coso/Documents/dev/ai/limecloud/limeshot/internal/roadmap/xuanlan/README.md
- /Users/coso/Documents/dev/ai/limecloud/limeshot/internal/roadmap/xuanlan/ITEM-PROJECTIONS.md
- /Users/coso/Documents/dev/ai/limecloud/limeshot/internal/roadmap/xuanlan/EVENT-PROJECTIONS.md
- internal/refactor/v1/01-comparison-matrix.md
- internal/refactor/v1/02-multi-model-grok-build.md
- internal/refactor/v1/03-target-architecture.md
- internal/refactor/v1/04-execution-plan.md
- internal/refactor/v1/05-verification-and-guardrails.md
- internal/refactor/v1/06-grok-vs-opencode.md
- internal/refactor/v1/10-item-inventory-skeleton.md
- internal/refactor/v1/11-codex-method-product-scope-matrix.md

## 3. 当前事实与差距

2026-07-28 的当前代码已经有 18 类 Rust ThreadItem 骨架，但 Renderer 还不是完整 Item Scene：

| 层              | 当前事实                                                                                                 | v2 缺口                                                                                                    |
| --------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Rust Item 协议  | lime-rs/crates/app-server-protocol/src/protocol/v2/item.rs 有 18 个顶层变体和 8 类流式增量               | 部分字段仍是 String 或 opaque Value；DynamicToolCall 缺音频；没有 renderer 专用安全 display shape          |
| v2 notification | ServerNotification 当前有 28 个 typed method                                                             | Codex 72 项覆盖表尚未逐项作产品裁决、typed contract 和 UI 出口                                             |
| reverse request | 当前只有 MCP elicitation、命令审批、文件审批、用户输入 4 类 typed request                                | currentTime、permission、dynamic tool dispatch 和 legacy 去重路径尚未完成产品裁决/接线                     |
| 前端直连路由    | appServerV2Notification.ts 只直接接 14 个 notification                                                   | 非直连 method 缺统一 coverage map，无法保证未知协议 fail visible                                           |
| canonical lower | appServerCanonicalItemReader.ts 可 lower 18 类上游 type；合法未知 camelCase type 已投影为 `unknown_item` | 多个类型被压成 extension 或 tool_call；declined 仍被伪装为 failed；ConversationProjection/reducer 尚未统一 |
| 时间线          | 现有 history hydration 会把 item 合并为 Message 与 tool content part                                     | Message -> Tool -> Message 的原始交错顺序不能稳定保留                                                      |
| 可见性          | ContextCompaction 已保留并复用现有低干扰信息行 renderer                                                  | live/cold/replay 共享 reducer 与专门 Gate B 证据尚未完成                                                   |
| 交互            | 既有 DecisionPanel 和 MCP form 可处理部分 pending action                                                 | pending interaction 没有统一的 item anchor、terminal/断线对账与跨 Thread 队列合同                          |

因此，v2 的第一原则是把当前 AgentThreadItem 的二次 lower 从主渲染通路移出。它可在迁移期间作为只读兼容输入，但不能继续增长为第二套 Item 模型。

## 4. 目标架构

    Electron Desktop Host
      -> App Server JSON-RPC v2
      -> typed Renderer gateway
      -> ConversationProjection reducer
      -> TurnTimeline / PendingInteractionStore / NoticeStore
      -> ItemRenderer(type) + Turn panels + Composer interaction layer

读路径与直播路径必须共享同一 reducer：

    thread/read or thread/items/list
      -> typed Thread / Turn / Item snapshot
      -> ConversationProjection reducer

    server notification
      -> typed notification reducer
      -> ConversationProjection reducer

Renderer 只能消费以下 UI 事实：

    ConversationProjection
      = thread header/status
      + ordered turns
      + discriminated item projections
      + pending interactions
      + notices
      + diagnostics

Item identity 是 threadId + turnId + item.id。首次 item/started 的 sequence 决定顺序；delta、progress 和 completed 只能更新同一 identity，不能插入第二个 Message 或重排。completed Item 和 completed Turn 分别是 Item、Turn 的权威终态。

## 5. 页面信息架构

对话页属于项目贯穿式复杂工作台，不是消息卡片墙。页面主对象是当前 Thread；当前阶段是本 Turn 的运行、等待输入或已完成；主动作始终是继续输入、完成待处理交互或中断 Turn。

    ConversationHeader
      - Thread 名称、连接/运行状态、模型与能力的用户级摘要
      - 用量、目标、告警入口
    ConversationViewport
      - TurnTimeline[]
        - UserMessageItem
        - ActivityCluster
          - ItemRenderer by type
        - AssistantMessageItem
        - TurnPlanPanel
        - TurnDiffPanel
        - TurnTerminalNotice
    PendingInteractionLayer
      - Command approval / File approval / Permission approval
      - Request user input / MCP elicitation
    Composer
      - 文本、图片、音频、Skill、Mention 输入
      - 队列、steer、send、interrupt
    ConversationStatusRegion
      - 连接、Hook、MCP、重试与安全告警

连续低信息活动可以被视觉聚合，但聚合不得合并 identity、生命周期或输出。Assistant Markdown 维持正文形态，不嵌套进装饰卡片；工具、搜索、Shell 和 Diff 使用紧凑活动行；长输出和长 Diff 在同一实体的受控抽屉中展开。

设计约束来自 internal/aiprompts/design-language.md：

- 主画布保留 Thread 上下文与唯一下一步，阻塞交互只存在一个可操作表面。
- 状态色固定为 emerald、amber、sky/slate、rose/red；状态不只依赖颜色。
- 使用实体表面、清晰边框和有限层级，不增加半透明主面板或卡片套卡。
- 普通用户区使用业务动作词；Agent、Runtime、Token、Gateway 等实现词仅进入运行明细或诊断。
- 所有用户文案同时覆盖 zh-CN、zh-TW、en-US、ja-JP、ko-KR。

## 6. 生命周期、恢复与安全

生命周期归并规则：

1. item/started 创建完整类型的活动实体，不能创建只有空 text 的占位。
2. delta 只更新其目标字段；reasoning 按 summaryIndex/contentIndex 分段，file patch 使用 snapshot replace。
3. completed Item 整体覆盖流式草稿；plan 的 completed text 不得假设等于 delta 拼接。
4. completed Turn 覆盖 Turn status、timing、usage、error 和权威 items，但保留本地折叠状态等纯 UI 状态。
5. 重复通知幂等；started 前到达的 delta 放进有界 orphan buffer；terminal 后拒绝 late delta 并留下诊断。
6. turn completed、thread closed、serverRequest/resolved、transport EOF 都必须终结 spinner、提交中表单和无效 action。

恢复规则：

- 历史只从 App Server Thread read/list 获取；不从 Renderer delta 或 Message cache 重建。
- live 与 replay 使用同一 ItemRenderer，仅由 renderSource: live 或 replay 控制能否执行交互。
- replay 不会重放审批、工具、外链或媒体生成；断线后由 Electron resume/read 对账当前 pending state。
- itemsView 不是 full 时必须显示历史未完整加载，不能将缺页视为空 Turn。

安全规则：

- Electron main 持有 raw reverse request id 与一次性 response authority；Renderer 只得到 semantic interactionId 和 action token。
- 参数、stdout/stderr、diff、MCP content、Hook output 都有大小限制、控制字符清理和敏感字段遮蔽。
- 绝对路径、媒体读取、外链、打开文件或 Thread 都经 host semantic action；Renderer 不直接访问系统或 raw protocol。
- structuredContent、MCP \_meta、raw response、provider metadata 默认不直接渲染，只有 allowlist 后的 display projection 可以显示。

## 7. 文件地图

| 文档                   | 目的                                                                     |
| ---------------------- | ------------------------------------------------------------------------ |
| ITEM-PROJECTIONS.md    | 18 类 ThreadItem 的渲染合同、字段边界和 current/gap 裁决                 |
| EVENT-PROJECTIONS.md   | 72 项上游 notification、11 类 reverse request 的产品范围、出口与终结合同 |
| IMPLEMENTATION-PLAN.md | 分阶段写集、删除清单、测试和 Gate A/B 退出条件                           |
| 本文                   | 目标架构、参考边界、迁移原则和全局验收                                   |

## 8. 完成定义

v2 只有同时满足以下条件才可标记完成：

1. 18/18 ThreadItem 都有独立 renderer 或明确的不可见控制语义，且没有合法类型被 null、filter 或 Message 合成静默丢弃。
2. Codex 72 个 notification、11 个 reverse request 的 coverage map 中每个 method 都有 current、planned、product-scope-excluded 或 deprecated 裁决；Lime-owned 多模型/多模态扩展有独立 inventory，未知输入分别 fail visible 或 fail closed。
3. 直播、冷读和恢复后的同一 Turn 产生语义等价的时间线顺序和终态。
4. 多模型与多模态仍由 model-provider current owner 控制，v2 不引入单模型 fallback、provider name heuristic 或第二 media lowering。
5. 旧 renderer projection、重复 timeline synthesis 和脱离构建图的兼容路径被物理删除，并有回流守卫。
6. Gate A、Recovery 和真实 Electron Gate B 均有针对 Markdown、Reasoning、Shell、Diff、MCP、dynamic tool、审批、用户输入、图片/音频、interrupt、恢复和 unknown drift 的证据。
