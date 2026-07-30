# Codex ThreadItem 全量渲染合同

状态：implementation in progress / exhaustive baseline

上游集合：Codex 4c43465133428898aa84f0bfc02c306ed65fb66a，18 类顶层 ThreadItem

Lime 协议基线：lime-rs/crates/app-server-protocol/src/protocol/v2/item.rs

实施快照：18 类 coverage baseline 与 pinned upstream drift gate 已固化；live、thread/read 和 production `thread/resume` replay 复用同一 ConversationProjection reducer。direct TurnTimeline 已接管 MessageList，typed Item renderer 覆盖 18 类 Item；CommandExecution 输出有 256 KiB 上限，unknown Item/notification 有脱敏 drift diagnostic。下表的 `current / gate-pending` 表示 typed producer/reader/renderer 已存在，但不代表该类型已经取得专项 Electron Gate B。

## 1. 通用合同

每个 Item 以 threadId + turnId + item.id 定位，并保留上游 type 判别。通用外壳只负责 icon、动词标题、状态、耗时、折叠、详情抽屉、live/replay 来源、pending interaction anchor 与诊断；它不得吸收任何类型的结构字段。

通用状态至少覆盖：

| 上游/运行状态  | Renderer 语义                                                    |
| -------------- | ---------------------------------------------------------------- |
| inProgress     | 正在进行，保留最新进度和稳定高度                                 |
| completed      | 已完成，显示耗时与摘要                                           |
| failed         | 失败，显示脱敏错误摘要与可展开记录                               |
| declined       | 用户拒绝，不伪装成系统失败                                       |
| interrupted    | Turn 中断，保留部分输出并停止 spinner                            |
| awaiting input | 有待办交互，在 Item 附近锚定且在 Composer 上方提供唯一可操作表面 |

当前投影保留 interrupted Turn 终态；CommandExecution/FileChange 的 declined 使用 completed lifecycle 加 typed 业务状态表达，显示层不得把拒绝伪装成系统失败或成功。

连续活动可进入 ActivityCluster，但这只影响视觉。每个 item.id 仍有独立更新键、状态、输出和可访问名称；展开后必须恢复原 sequence。

## 2. 全量矩阵

|   # | 上游 type           | 目标主投影              | 当前裁决               | v2 renderer             |
| --: | ------------------- | ----------------------- | ---------------------- | ----------------------- |
|   1 | userMessage         | 用户消息与附件          | current / gate-pending | UserMessageItem         |
|   2 | hookPrompt          | Hook 注入上下文         | current / gate-pending | HookPromptItem          |
|   3 | agentMessage        | Assistant Markdown 正文 | current / gate-pending | AgentMessageItem        |
|   4 | plan                | 建议方案                | current / gate-pending | ProposedPlanItem        |
|   5 | reasoning           | 推理摘要与受控原始内容  | current / gate-pending | ReasoningItem           |
|   6 | commandExecution    | Shell / 文件探索活动    | current / gate-pending | CommandExecutionItem    |
|   7 | fileChange          | 文件变更与 Diff         | current / gate-pending | FileChangeItem          |
|   8 | mcpToolCall         | MCP 调用                | current / gate-pending | McpToolCallItem         |
|   9 | dynamicToolCall     | 业务工具调用            | current / gate-pending | DynamicToolCallItem     |
|  10 | collabAgentToolCall | 多 Agent 调度           | current / gate-pending | CollabAgentToolCallItem |
|  11 | subAgentActivity    | 子 Agent 生命周期       | current / gate-pending | SubAgentActivityItem    |
|  12 | webSearch           | 搜索活动与结果          | current / gate-pending | WebSearchItem           |
|  13 | imageView           | 图片查看                | current / gate-pending | ImageViewItem           |
|  14 | sleep               | 等待                    | current / gate-pending | SleepItem               |
|  15 | imageGeneration     | 图片生成                | current / gate-pending | ImageGenerationItem     |
|  16 | enteredReviewMode   | Review 进入边界         | current / gate-pending | ReviewBoundaryItem      |
|  17 | exitedReviewMode    | Review 退出边界         | current / gate-pending | ReviewBoundaryItem      |
|  18 | contextCompaction   | 上下文压缩信息行        | current / gate-pending | ContextCompactionItem   |

所有类型都通过 typed canonical reader 进入 direct timeline；专项 Gate B 未覆盖的类型仍不得扩大为产品完整性结论。未知或畸形 shape 禁止降级到 extension/raw JSON，必须 fail visible 或 fail closed。

## 3. UserMessage

字段是 id、clientId、content[]。content 必须保持原序，不得只提取 text。

| UserInput type | 投影                 | 当前差距                                                                        |
| -------------- | -------------------- | ------------------------------------------------------------------------------- |
| text           | 文本与 text elements | 当前 parser 对非法 UTF-8 byte range fail closed；v2 改为跳过该 range 并保留文本 |
| image          | 远端图片缩略图       | 需要受控 URL、lazy load、失败态和 lightbox                                      |
| localImage     | host 介导本地图片    | 当前会携带 source path；v2 改为受控 media handle                                |
| audio          | 不在当前产品协议范围 | Lime Rust UserInput/AgentInput 无此 variant；reader fail closed                 |
| localAudio     | 不在当前产品协议范围 | Lime Rust UserInput/AgentInput 无此 variant；reader fail closed                 |
| skill          | 只读 Skill token     | 当前有基础字段，需 semantic open action                                         |
| mention        | 文件/资源 mention    | 当前有基础字段，需 host allowlist action                                        |

音频和图片的 canonical content、provider support 和 wire lowering 继续归 model-provider。v2 只消费已经安全解析的媒体 reference，不能把 OpenCode 或任何 provider 的 raw content union 复制到 React。

## 4. AgentMessage、Plan 与 Reasoning

### AgentMessage

text 使用安全 GFM Markdown。item/agentMessage/delta 形成流式草稿，completed Item 的 text 是权威终态。phase 只能是低干扰阶段标记，不能改变 Item 顺序。memoryCitation 的 path、行范围和历史 Thread 通过 host action 打开，不能暴露 raw protocol method。

当前历史 hydration 会筛选阶段并把多个 Item 拼进 assistant draft。v2 改为直接渲染每个 AgentMessageItem；只有明确的视觉聚合才能相邻显示，不能丢失中间工具或 reasoning。

### Plan

Plan 是模型建议的 Markdown 计划，不等同 Rust ProductionPlan 或 Turn plan checklist。item/plan/delta 只生成临时草稿；completed plan.text 必须整体替换。默认展开，长内容显示摘要后可折叠，不显示业务审批按钮。

### Reasoning

summary[] 和 content[] 必须按 index 分段维护。summaryPartAdded 创建段，summaryTextDelta 和 textDelta 只追加对应段。默认只显示 summary；原始 content 仅在产品策略允许的运行明细中显示。completed Item 全量覆盖流式数组；Turn interrupted 后保留已到达的 summary 并结束 spinner。

## 5. CommandExecution 与 FileChange

### CommandExecution

优先按 commandActions 显示读取、列出文件、搜索或运行命令；一条 command 的多个 action 保持上游顺序，原 command 始终可展开。不得在 Renderer 重新实现 Shell parser。

- output delta 进入有界 buffer，ANSI/control character 安全化；预览保留头尾，完整记录进入同一 Item 抽屉。
- status 是视觉终态权威，exitCode 只是诊断。declined 必须独立显示。
- cwd 和路径下发前转为安全相对展示；process handle、stdin channel 和绝对路径不进入 Renderer。
- terminalInteraction 只显示已发送输入的脱敏摘要；单进程终止只在 host 有明确 capability 时暴露。

### FileChange

changes[] 以完整快照显示新增、修改、删除和重命名摘要，文件和 hunk 保持上游顺序。Diff 使用 parser，不用字符串替换模拟。item/fileChange/patchUpdated 替换当前 changes，不 append；deprecated outputDelta 只进兼容诊断，不能覆盖 patch。失败和拒绝仍保留 Diff 供检查。

## 6. MCP 与 Dynamic Tool

### McpToolCall

标题是 server / tool；可读 appContext 名称优先，技术 identity 为次级详情。arguments 使用 JSON tree 并遮蔽 token、password、authorization 等敏感键。progress 是有界列表，主行只显示最新一条。

结果已使用 typed shape 区分 content、structuredContent、error 和 \_meta：

- content 按 text、image、audio、embedded resource、resource link、unknown JSON 顺序渲染；
- structuredContent 进入 schema-aware viewer，不与正文重复；
- \_meta 默认不显示，只读取 allowlisted UI metadata；
- result/error 只按 typed、脱敏 display shape 进入 renderer；畸形 required field fail closed；
- 媒体和资源只能经 host capability 读取。

MCP elicitation 不是 MCP result。它是独立 pending interaction；若有 thread/turn/item 关联，只在 Item 旁提供 anchor，表单仍在 Composer 上方完成。

### DynamicToolCall

标题显示业务可读名称和 namespace/tool，详情可查看 stable technical identity。arguments 按工具 schema 显示，禁止任意 HTML。Item 本身仅观察状态；实际 item/tool/call 由 Electron main 校验 binding 后路由 current ToolHost。

contentItems 必须按原序渲染 text、image、audio。DynamicTool 的真实协议、schema、generated client 与 reader 已保留 `inputAudio`；这与不支持 UserMessage audio/localAudio 是两个不同边界。completed 加 success=false 表示业务失败，不能因为 Item lifecycle completed 而显示成功。

## 7. Multi-Agent、搜索与媒体

### CollabAgentToolCall

覆盖 spawnAgent、sendInput、resumeAgent、wait、closeAgent。显示 receiver 的友好名称、model、reasoning effort、prompt 摘要和 agent states；raw Thread id 仅在诊断详情。点击只能触发 semantic open-thread action。

### SubAgentActivity

显示 started、interacted、interrupted。它更新 Agent switcher 的活动提示，但不从 activity 文本推导子 Turn 终态。

### WebSearch

action 可能为 search、openPage、findInPage 或 other。显示 query、URL host/path 与 allowlisted 的 title/url/snippet/source 结果；未知 opaque result 字段不可整块 stringify。外链经 Electron URL allowlist 打开。

### ImageView 与 ImageGeneration

图片查看使用安全 media handle、缩略图、lightbox 与定位 action；绝对 path 不给 Renderer。图片生成使用稳定尺寸占位、revised prompt、可信结果 media reference 和 saved path 的 host action。completed Item 是权威结果，失败不显示破图。

## 8. Hook、Sleep、Review 与 Compaction

HookPrompt 是持久化上下文事实，不是用户消息。它以 typed fragments 显示为低干扰信息行；`hookRunId` 不进入可见 DOM，replay 只显示历史事实而不触发 Hook。

Sleep 显示可访问的等待状态和 reduced-motion 静态进度，但只能由 Item/Turn lifecycle 结束；用户只能 interrupt Turn。replay 只显示历史记录。

enteredReviewMode 和 exitedReviewMode 是顺序边界，不是 Assistant final answer。它们更新 Composer mode UI；缺少对应进入/退出事件时仍 fail visible。

ContextCompaction 显示低干扰信息行。`threadTimelineView.ts` 的静默过滤分支已删除，并复用现有 `ContextCompactionCard`。历史所有权仍归 Codex/App Server，Renderer 不保存或重建压缩前完整 history；live、cold read 与 production resume replay 已复用同一 reducer。

## 9. Unknown Item

升级后遇到未知 camelCase type 时，current Renderer 会收到 `unknown_item`：

1. 时间线显示暂不支持的活动和 upstream type；
2. 保留 identity、时间、状态和脱敏字段名列表，不展示未经审核的原值；
3. 记录 protocol revision、method、type 和 schema drift；
4. 不阻断同 Turn 后续 Item；
5. 若有关联 pending interaction，仍按通用交互合同处理，绝不自动批准或自动失败。

## 10. Item 验收

- 18 类 Item 各有 started、completed、replay fixture；有 delta/progress 的类型另有流式 fixture。
- 有状态的 Item 覆盖 completed、failed、declined、interrupted。
- UserInput 覆盖 7 类；CommandAction 覆盖 4 类；Collab tool 覆盖 5 类；agent states 覆盖 7 类。
- 验证乱序、重复、late delta、completed overwrite、断线、resume/read 和 itemsView 非 full。
- 同一 Turn 的 Message -> Tool -> Message -> Tool 顺序在 live、cold read、replay 一致。
- 未知 Item 可见、无副作用，且对后续 Item 不造成渲染阻断。
