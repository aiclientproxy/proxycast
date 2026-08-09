# Codex Event 与 Reverse Request 全量投影表

状态：implementation in progress / coverage baseline

上游事实源是 Codex c4f42d161ae44a8d696ee9fb595709661979d187。表中的 72 个 notification 和 11 个 reverse request 是审计全集，不代表 Lime 会无条件复制所有 Codex 产品功能。xuanlan 原稿记录的 10 类 request 未包含当前 revision 的 currentTime/read，本文件以 Codex/v1 机器清单为准。

每项必须有且只有一种裁决：

- current：Lime current typed protocol/catalog 已存在；这只说明协议边界当前有效，不代表 GUI、恢复和 Gate B 已完整对齐；
- planned：属于 Lime 产品范围，必须在 current owner 补全；
- product-scope-excluded：不属于 Lime 产品，禁止做 compat 模拟；仅保留版本审计和 fail-closed 诊断；
- deprecated：上游已退役，仅防重复或安全诊断，不新增用户功能。

出口代码：TL 时间线，TP Turn 面板，PI pending interaction，HS Header/状态区，GN 应用通知，DX 仅开发诊断。

实施快照：direct `item/started -> item/commandExecution/outputDelta* -> item/completed` 通过 typed adapter 和共享 reducer；production `thread/resume` 安装同一 replay reducer，后续 live notification 继续复用。completed snapshot 权威覆盖 delta 草稿，输出限制为 256 KiB。`write_stdin` 复用原始 `exec_command` Item identity，typed terminal interaction 与 canonical cold read 只保留 `sent N chars` 脱敏摘要。`turn.plan.updated` 由 canonical `update_plan` 的 `ToolOutput.structured_content` 派生，实时与 canonical cold read 共用 checklist 投影；`update_plan` 工具项保留在 read model，但不生成 `ThreadItem.plan` 或 Plan UI。Hook lifecycle 由 current Hook runtime 产生 paired `hook.started`/`hook.completed`，只做 transient timeline 投影，不创建 canonical ThreadItem。trusted first-party Responses 的 moderation metadata 已走 `model-provider -> AgentEvent -> durable event -> v2 turn/moderationMetadata -> typed client -> canonical Turn`，保持 opaque JSON 与 last-write-wins。strictAutoReview 的 shell/`exec_command` 触发真实 Guardian reviewer，经同 session `model-provider` 无工具结构化采样生成 durable `guardian.review.started/completed`，再投影为 typed `item/autoApprovalReview/*` 与 Renderer `pending_interactions`；provider 不可用、取消、超时和非法响应全部拒绝。unknown Item 已沿 canonical typed payload、v2 `thread/read`、Renderer 终态合并与 direct TurnTimeline fail-visible，只保留 upstream type 和脱敏字段名，并有专项 Electron Gate B；unknown/known-unprojected notification drift recorder 仍只提供诊断，不能把 72 notification 中的 planned surface 标记完成。

## 1. Thread、Turn 与 Hook

|   # | Method                          | 目标出口 | 当前裁决               | v2 投影                                                           |
| --: | ------------------------------- | -------- | ---------------------- | ----------------------------------------------------------------- |
|   1 | error                           | TP/HS    | current                | typed live/durable；true 重试，false 等权威 Turn                  |
|   2 | thread/started                  | HS       | current                | 建立 Thread metadata，不把空 turns 当完整 history                 |
|   3 | thread/status/changed           | HS/PI    | current                | notLoaded、idle、systemError、active 与 waiting flags             |
|   4 | thread/archived                 | GN/HS    | current                | 侧栏归档，当前页只读                                              |
|   5 | thread/deleted                  | GN       | current                | 当前页已删除态，不删除 Project                                    |
|   6 | thread/unarchived               | GN/HS    | current                | 恢复可见/可操作状态                                               |
|   7 | thread/closed                   | HS       | current                | 清除 live spinner 和 pending interaction                          |
|   8 | skills/changed                  | GN       | current                | 瞬时失效并重新读取 Composer current Skill catalog                 |
|   9 | thread/name/updated             | HS       | current                | Header 与侧栏名称                                                 |
|  10 | thread/goal/updated             | HS       | current                | goal/阶段入口，不映射 Rust ProductionPlan                         |
|  11 | thread/goal/cleared             | HS       | current                | 清除当前 goal indicator                                           |
|  12 | thread/environment/connected    | HS       | planned                | 受控环境状态，无本地 fallback                                     |
|  13 | thread/environment/disconnected | HS       | planned                | 断开与受影响能力                                                  |
|  14 | thread/settings/updated         | HS       | current                | 下一 Turn model、reasoning、permission 摘要                       |
|  15 | thread/tokenUsage/updated       | TP/HS    | current                | 本 Turn/总用量，节流更新                                          |
|  16 | turn/started                    | TL/TP    | current                | 建立 Turn 与原始 Item 顺序                                        |
|  17 | hook/started                    | TL/HS    | current                | current Hook producer 的 transient activity，run id 保留          |
|  18 | turn/completed                  | TL/TP    | current                | 权威 Turn 终态并清理 pending                                      |
|  19 | hook/completed                  | TL       | current                | paired Hook producer 的 transient status，不写入 canonical Item   |
|  20 | turn/diff/updated               | DX       | current                | Lime exact Turn diff，canonical Turn/Changes 使用同一快照       |
|  21 | turn/plan/updated               | TP       | current                | canonical update_plan checklist，实时/冷恢复一致                  |

## 2. Item 生命周期、流与进程

|   # | Method                                    | 目标出口 | 当前裁决               | v2 投影                                                     |
| --: | ----------------------------------------- | -------- | ---------------------- | ----------------------------------------------------------- |
|  22 | item/started                              | TL       | current                | 按 typed 联合建立 Item；未知安全 fail visible               |
|  23 | item/autoApprovalReview/started           | PI/TL    | current                | 目标 Item 的 Guardian review 进行中                         |
|  24 | item/autoApprovalReview/completed         | PI/TL    | current                | approved/denied/timedOut/aborted 与风险摘要                 |
|  25 | item/completed                            | TL       | current                | Item 权威终态覆盖流式草稿                                   |
|  26 | rawResponseItem/completed                 | DX       | product-scope-excluded | 不参与正式 Item 或终态合成                                  |
|  27 | rawResponse/completed                     | DX       | product-scope-excluded | 不进入普通时间线                                            |
|  28 | item/agentMessage/delta                   | TL       | current                | 合批 Markdown stream                                        |
|  29 | item/plan/delta                           | TL       | current                | 临时计划草稿，completed 整体替换                            |
|  30 | command/exec/outputDelta                  | DX/GN    | product-scope-excluded | 独立 command API，不能混入 Agent command                    |
|  31 | process/outputDelta                       | DX       | product-scope-excluded | standalone unsandboxed process/spawn 流；不进入 Lime 产品链 |
|  32 | process/exited                            | DX       | product-scope-excluded | standalone process 终态；不进入 Lime 产品链                 |
|  33 | item/commandExecution/outputDelta         | TL       | current                | 有界 stdout/stderr buffer                                   |
|  34 | item/commandExecution/terminalInteraction | TL/DX    | current                | 复用原 Command Item；typed/cold read 仅保留脱敏摘要         |
|  35 | item/fileChange/outputDelta               | DX       | deprecated             | 只记录兼容诊断，不覆盖 patch                                |
|  36 | item/fileChange/patchUpdated              | TL/TP    | current                | replace changes snapshot                                    |
|  37 | serverRequest/resolved                    | PI       | current                | 按 interaction identity 终结表单                            |
|  38 | item/mcpToolCall/progress                 | TL       | current                | 有界 MCP 进度列表                                           |

## 3. MCP、账号、应用与系统资源

|   # | Method                               | 目标出口 | 当前裁决               | v2 投影                                   |
| --: | ------------------------------------ | -------- | ---------------------- | ----------------------------------------- |
|  39 | mcpServer/oauthLogin/completed       | GN/PI    | current                | 结束 OAuth 等待并刷新 MCP status          |
|  40 | mcpServer/startupStatus/updated      | HS/GN    | current                | typed starting/ready/failed，终态刷新 GUI |
|  41 | account/updated                      | GN       | product-scope-excluded | 不写入对话 history                        |
|  42 | account/rateLimits/updated           | HS/GN    | product-scope-excluded | 若未来纳入，独立产品范围变更              |
|  43 | app/list/updated                     | GN       | current                | App Center Apps readiness fresh read      |
|  44 | remoteControl/status/changed         | GN       | product-scope-excluded | 不进入 Thread                             |
|  45 | externalAgentConfig/import/progress  | GN       | product-scope-excluded | 仅设置页导入任务                          |
|  46 | externalAgentConfig/import/completed | GN       | product-scope-excluded | 同上                                      |
|  47 | fs/changed                           | GN/DX    | planned                | 只路由给对应 watch consumer               |
|  48 | item/reasoning/summaryTextDelta      | TL       | current                | 按 summaryIndex 更新分段                  |
|  49 | item/reasoning/summaryPartAdded      | TL       | current                | 创建指定 summary 分段                     |
|  50 | item/reasoning/textDelta             | TL       | current                | 按 contentIndex 更新受控原始推理          |
|  51 | thread/compacted                     | TL/DX    | deprecated             | 无 ContextCompaction Item 时仅显示一次    |

## 4. Model、安全、告警与搜索

|   # | Method                           | 目标出口 | 当前裁决               | v2 投影                                                      |
| --: | -------------------------------- | -------- | ---------------------- | ------------------------------------------------------------ |
|  52 | model/rerouted                   | TL/HS    | current                | from/to 与 allowlisted reason；不改变 route owner            |
|  53 | model/verification               | HS/DX    | current                | 脱敏验证结论                                                 |
|  54 | turn/moderationMetadata          | DX       | current                | trusted first-party metadata，opaque Turn state，last-write-wins    |
|  55 | model/safetyBuffering/updated    | HS       | current                | 安全缓冲提示，不伪造模型选择                                 |
|  56 | warning                          | HS/GN    | current                | typed threadId/message/code?；实时去重 toast 与冷读恢复      |
|  57 | guardianWarning                  | HS/TL    | planned                | 高优先级安全 warning，不被普通 warning 吞掉                  |
|  58 | deprecationNotice                | GN       | product-scope-excluded | 开发/设置诊断，不污染对话流                                  |
|  59 | configWarning                    | GN       | current                | initialize/turn producer；typed path/range 经去重 toast 展示 |
|  60 | fuzzyFileSearch/sessionUpdated   | PI       | planned                | Composer mention 搜索，丢弃陈旧 session                      |
|  61 | fuzzyFileSearch/sessionCompleted | PI       | planned                | 终结 loading、显示空/失败态                                  |

## 5. Realtime、Windows 与登录

|   # | Method                            | 目标出口 | 当前裁决               | v2 投影                                      |
| --: | --------------------------------- | -------- | ---------------------- | -------------------------------------------- |
|  62 | thread/realtime/started           | HS/PI    | planned                | 建立语音会话状态                             |
|  63 | thread/realtime/itemAdded         | TL       | planned                | 以结构化 Item 投影，不 stringify raw payload |
|  64 | thread/realtime/transcript/delta  | TL/PI    | planned                | provisional 语音转写                         |
|  65 | thread/realtime/transcript/done   | TL       | planned                | final transcript 替换 provisional            |
|  66 | thread/realtime/outputAudio/delta | PI       | planned                | 有界音频播放队列                             |
|  67 | thread/realtime/sdp               | DX       | product-scope-excluded | Electron/WebRTC owner 消费                   |
|  68 | thread/realtime/error             | HS/PI    | planned                | 可恢复错误，不推断普通 Turn failed           |
|  69 | thread/realtime/closed            | HS/PI    | planned                | 释放媒体状态，保留最终 transcript            |
|  70 | windows/worldWritableWarning      | GN       | planned                | Windows 安全摘要和设置入口                   |
|  71 | windowsSandbox/setupCompleted     | GN/PI    | planned                | setup success/error 与下一步                 |
|  72 | account/login/completed           | GN/PI    | product-scope-excluded | credential 流程不进入对话                    |

v2 的实现门槛不是把所有 planned method 同时实现，而是首先将这张表固化为类型检查的 coverage map。新增 Codex method 时，CI 必须要求它先获得裁决，不能落入 default silent return。standalone `process/outputDelta` 与 `process/exited` 虽保留在 upstream method inventory 和 drift recorder 中，但明确为 `product-scope-excluded`：不得进入 Lime current protocol、Renderer projector、时间线或用户级通知；对应 standalone `process/spawn` 控制面不能借 planned 名义回流。Lime exact `turn/diff/updated` 已由 `apply_patch -> durable fact -> v2 projector -> canonical Turn/Changes` current owner 承接，不等同于 Codex TUI 的 raw diff surface。

## 6. Lime-owned 扩展事件

Codex 72 项是 runtime/rendering 上游基线，不是删除 Lime 产品扩展的 allowlist。下列 current surface 来自多模型或多模态 owner，必须由其自身 inventory 管理：

| Method             | Owner                       | 出口  | 裁决                                                         |
| ------------------ | --------------------------- | ----- | ------------------------------------------------------------ |
| model/list/updated | model-provider + App Server | HS/GN | current；保留 Grok-style catalog refresh，不计入 Codex 72 项 |

后续发现的 Lime-owned method 也必须在独立 fixture 中登记 owner、schema、consumer 和 evidence。只要它属于 model-provider 的 catalog/route/capability 或 canonical media/lowering，就不能因为 Codex 没有同名 method 而删除；但也不得借扩展名创建第二套 Thread/Turn/Item 生命周期。

## 7. Reverse Request

raw JSON-RPC transport id/action token 只由 server-request dispatcher 的请求闭包持有。React projection 只接收 semantic interaction identity，不能持久化或显示 transport identity。

|   # | Method                                | 当前裁决               | GUI/host 处理                                                                                 |
| --: | ------------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------- |
|   1 | item/commandExecution/requestApproval | current                | 命令审批，显示 command/actions/risk；映射 accept、session、policy、network、decline、cancel   |
|   2 | item/fileChange/requestApproval       | current                | Patch 审批，显示 Diff/reason/grantRoot                                                        |
|   3 | item/tool/requestUserInput            | current                | 1-3 个问题、option、Other、secret、auto-resolution                                            |
|   4 | mcpServer/elicitation/request         | current                | form、openai/form、url；结构化校验                                                            |
|   5 | currentTime/read                      | current                | Electron Host 独占系统时钟；thread-scoped reverse request 不进入时间线，Renderer 不直接读时钟 |
|   6 | item/permissions/requestApproval      | current                | typed cwd/reason/environment/profile 经统一 PendingInteraction；exact waiter fail closed      |
|   7 | item/tool/call                        | current                | Electron 校验 frozen binding 后执行 ToolHost；Renderer 只观察 typed DynamicToolCall Item      |
|   8 | account/chatgptAuthTokens/refresh     | product-scope-excluded | credential broker host-only，绝不显示 token                                                   |
|   9 | attestation/generate                  | product-scope-excluded | 平台能力 host-only，无伪造 fallback                                                           |
|  10 | applyPatchApproval                    | deprecated             | legacy 与 v2 file approval 去重，同一动作只有一个 prompt                                      |
|  11 | execCommandApproval                   | deprecated             | legacy 与 v2 command approval 去重，同一动作只有一个 prompt                                   |

command/file approval、requestUserInput、MCP elicitation 与 permission approval 已统一注册到一个 `PendingInteractionController`，共享提交幂等、abort、Turn/thread 终结和 Composer 上方唯一交互表面。`currentTime/read` 与 `item/tool/call` 由 Electron Host 在 Renderer 之前处理；旧 server-request controller、独立 MCP Dialog/controller 和第二 pending store 已删除。product-scope-excluded request 不得复用这些 handler 伪造支持。

## 8. Pending Interaction 合同

    PendingInteractionProjection
      = CommandApproval
      | FileApproval
      | UserInputRequest
      | McpElicitation
      | PermissionApproval

`PermissionApproval` 使用 typed cwd、reason、environment 与 permission profile diff；grant 只响应 App Server 当前 exact waiter，未知 decision、abort、detach 和越权 profile 均 fail closed。

每个 projection 都包含 semantic interactionId、thread/turn/item anchor、createdAt、可选 expiresAt、pending/submitting/resolved/expired/disconnected 状态、本地化文案、结构化选项与 secret/network/filesystem/session 风险标签。

规则：

1. 同一 Thread 多个请求按 started time 排队；跨 Thread 待办可切回目标 Thread。
2. submit 后立刻 disabled，等待 server response 或 serverRequest/resolved；resolved 后禁止重投。
3. Turn completed、thread closed、断线或 server resolved 终结交互；Renderer timer 不宣告权威完成。
4. secret 不进入日志、遥测、React error boundary 或历史 Item。
5. 待办交互只有一个可操作表面；时间线 anchor 只用于定位和状态，不复制第二张表单。

## 9. 非 Item 投影与状态终结

Turn plan 是 canonical `update_plan` 的执行 checklist，不是 ThreadItem.plan 或 Rust ProductionPlan；实时 `turn.plan.updated` 与 cold read 从同一结构化快照恢复，非法快照不覆盖上一份有效计划。Turn diff 是本 Turn 聚合 Diff，不能取代 fileChange Item。Header 只显示当前 model、reasoning、permission/sandbox、environment 与 active flags 的用户级摘要；底层 route、provider、credential 和 capability 明细进入受控运行详情。

| 触发                        | 必须终结的 UI                                         |
| --------------------------- | ----------------------------------------------------- |
| item/completed              | Item spinner、progress、该 Item 草稿                  |
| turn/completed: completed   | Turn spinner、可结束 Hook、无效 pending               |
| turn/completed: interrupted | 所有运行 Item 转中断态，保留部分输出                  |
| turn/completed: failed      | Turn error 与运行 Item 失败/中断态                    |
| serverRequest/resolved      | 对应表单和按钮                                        |
| thread/closed               | Thread live status、全部 pending                      |
| transport EOF/crash         | submitting 交互转 disconnected，等待 resume/read 对账 |

## 10. 覆盖测试

- 18 类 Item 的 started、delta/progress、completed、replay fixture；
- 72 个 notification 的唯一出口与分类守卫；
- 11 类 reverse request 的 owner、semantic projection/host-only 路由、response mapping；
- 参数化覆盖 UserInput、CommandAction、Collab tool、Agent status、WebSearch action；
- MCP content 的 text/image/audio/resource/resourceLink/unknown JSON、structuredContent 和 error；
- approval 的 accept/session/policy/network/decline/cancel 与 Guardian 五终态；
- 乱序、重复、late delta、completed overwrite、disconnect、resume/read、itemsView 非 full；
- unknown Item live/terminal/cold read 均 fail visible 且不泄漏 raw values；unknown notification fail visible，unknown reverse request fail closed；
- Gate B 真实覆盖 Markdown、Search、Shell output、Diff、MCP、dynamic tool、审批、用户输入、interrupt、媒体、历史恢复与 unknown Item 专项恢复。
