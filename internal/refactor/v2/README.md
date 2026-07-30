# Codex 渲染对齐重构 v2

状态：implementation in progress（V2-02 / V2-04 收尾，V2-05 planned surface 待实现）

日期：2026-07-29

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

2026-07-29 的 current 代码已经让 18 类 Rust ThreadItem 通过 typed canonical reader 进入 direct TurnTimeline，但专项 Gate B 与 V2-05 高级通知仍未完整：

| 层              | 当前事实                                                                                                        | v2 缺口                                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Rust Item 协议  | app-server-protocol v2 有 18 个顶层变体；MCP result/error 与 DynamicTool `inputAudio` 为 typed current contract | UserMessage audio/localAudio 不在 Lime 产品协议范围；部分 host/media 异常态 Gate B 待补 |
| v2 notification | 72 项 coverage、pinned schema/hash/method drift gate 与 unknown notification recorder 已落地                    | 大量 planned notification 尚无完整 producer、GUI outlet 与 Gate B                       |
| reverse request | MCP elicitation、命令审批、文件审批、用户输入 4 类 typed request 共用唯一 pending owner                         | currentTime、permission、dynamic tool dispatch 仍须按产品范围补真实 owner               |
| canonical lower | 18 类上游 type 均投影为 typed Item；unknown camelCase type 进入 `unknown_item`，畸形/旧 shape fail closed       | 不再允许 extension 或 raw JSON fallback；专项 malformed/Gate B 仍需扩圈                 |
| 时间线          | MessageList 直接消费 canonical Turn render projection；User/Agent/Media/Process 保持 sequence                   | 长列表性能和完整 Electron 场景矩阵仍待验证                                              |
| 恢复            | live、cold read 与 production `thread/resume` 使用同一 reducer；resume 后 live 继续复用该实例                   | restart/disconnect 的完整 V2-05 产品矩阵仍待补                                          |
| 交互            | 四类 pending request 使用单一 PendingInteractionController 和 Composer 上方唯一交互层                           | planned permission/tool-call request 尚未进入 current union                             |

因此，v2 的第一原则是把当前 AgentThreadItem 的二次 lower 从主渲染通路移出。迁移期间只有旧 Message/read 渲染链可把它作为只读兼容输入；canonical ThreadItem state 仍是 current live/read 状态，不得把二次 lower 继续增长为第二套 Item 模型。

### 3.1 当前 Renderer 主链

```text
direct v2 notification
  -> current notification router
  -> request-scoped ConversationProjection reducer
  -> canonical AgentThreadItem state
  -> existing timeline renderer

thread/read + thread/items/list + thread/turns/list
  -> canonical item reader
  -> the same ConversationProjection reducer
  -> canonical AgentThreadItem state

thread/resume
  -> canonical item reader
  -> install the same ConversationProjection reducer into active stream state
  -> subsequent live notifications reuse the same reducer

canonical Thread/Turn/Item state
  -> pure turnTimelineRenderProjection
  -> direct User / Agent / Media / Process segments
  -> MessageList
```

direct notification 没有上游 event id 时，live owner 只在 sequence gate/router 之后分配 request 内到达序号；不得用内容 hash 合并合法重复 chunk，也不得把该序号宣称为跨重连 replay identity。CommandExecution、Tool、WebSearch 和 Patch 的可见输出在 projection 边界限制为 256 KiB 并保留尾部。unknown Item diagnostic 只记录 revision、method、type、identity 与脱敏字段名。

canonical Item -> Message 的 tool/agent/reasoning 合成入口、`canonicalItemsToMessages` 与旧 MessageList canonical compatibility branch 已物理删除，属于 `dead / deleted / forbidden-to-restore`。未被 canonical Item 覆盖的 optimistic/imported/local product surface 仍以 residual Message 显示，但它只是 direct projection 的纯派生补充，不是第二个 Item store。

## 4. 目标架构

    Electron Desktop Host
      -> App Server JSON-RPC v2
      -> typed Renderer gateway
      -> ConversationProjection reducer
      -> TurnTimeline / PendingInteractionController / NoticeStore
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
