# Agent Workspace Tools UI 评分卡

> 当前静态分：`3.3 / 5`  
> 更新时间：2026-06-15  
> 目标：评估 Agent Workspace 是否能把几十个工具的执行过程做成可观察、可控、可恢复、可审计的桌面端 UI，而不是只把工具输出塞进聊天正文。逐项 inventory 见 `tools-inventory.md`。

说明：本文件只评估工具 UI。固定 catalog 里的 `Skill` 在这里仅代表 **Skill tool call rendering**；动态 `mcp__*`、`ListMcpResourcesTool`、`ReadMcpResourceTool` 在这里仅代表 **MCP tool/resource card rendering**。完整 Skill 包标准见 `skills.md`，完整 MCP server / tools / resources / prompts / auth / elicitation / evidence 评测见 `mcp.md`。

## 1. 评分口径

本评分卡不再把“工具注册成功”当成“工具 UI 支持”。每个工具按 5 层证据评分：

| 层级 | 证据 | 说明 |
| --- | --- | --- |
| Catalog | `lime-rs/crates/agent/src/agent_tools/catalog.rs` | 工具是否在 current catalog 中，有 surface / capability / lifecycle / permission / default allow |
| Runtime inventory | `agentSession/toolInventory/read` | 工具是否能在当前 session surface 中被读取、过滤、显示可见性、展示 policy |
| Frontend family | `src/components/agent/chat/utils/toolDisplayInfo.ts` | 工具是否有可读 label、family、动作文案、图标和分组 |
| Renderer / test | `ToolCallDisplay*`、`toolProcessSummary*`、`toolDisplayInfo*` | 是否有输入摘要、输出预览、错误隐藏、分组、artifact / image / diff 等组件级证据 |
| GUI / evidence | Playwright / GUI smoke / Evidence Pack | 是否经过真实桌面端流程验证，包含 runtime transcript、截图或 snapshot、最终 outcome |

静态分含义：

| 分数 | 含义 |
| --- | --- |
| 0 | 未发现工具或 UI 证据 |
| 1 | 仅历史、deprecated 或不可作为 current 能力 |
| 2 | Catalog 存在，但前端只落 generic 或缺关键测试 |
| 3 | Catalog + 前端 family 可读，缺逐工具深水位 GUI 证据 |
| 4 | 有专用渲染或组件测试，主要缺真实 GUI / Evidence Pack |
| 5 | 有 catalog、runtime inventory、专用 UI、失败恢复、真实 GUI evidence 和回归门禁 |

## 2. 调研依据

| 来源 | 本轮使用方式 | 对评分的约束 |
| --- | --- | --- |
| Context7 AG-UI `/ag-ui-protocol/ag-ui` | 查询 run、text、thinking、tool call、state、error 事件 | 工具 UI 至少要覆盖 start、args、end、result、error，并能和 thinking / text 顺序共存 |
| Context7 Vercel AI SDK `/vercel/ai` | 查询 `UIMessage.parts` 与 tool states | 工具状态至少区分 `input-streaming`、`input-available`、`output-available`、`output-error` |
| WebSearch: OpenAI Evals | 评估 LLM 系统要有 task、dataset、grader、trace | Agent Workspace 工具评测必须记录 transcript、tool calls、UI evidence 和 verdict |
| WebSearch: Anthropic agent evals / effective agents | 先做可观察 eval，按失败模式优化 | 不按“功能多”打分，按真实任务结果、可调试性和可恢复性打分 |
| WebSearch: SWE-bench / Terminal-Bench | coding 与 terminal 任务要用真实环境和测试验证 | Bash、PowerShell、apply_patch、Read/Write/Edit 必须能关联 diff、exit code、测试结果 |
| WebSearch: WebArena / OSWorld / GAIA / AgentBench | browser、桌面、通用工具任务需要独立 benchmark | browser / site / web search 工具不能只看调用成功，要有操作轨迹、来源、页面状态和 outcome |
| Codex app-server | thread / turn / item、stream notifications、approval、command、fs、skills | Lime UI 要消费 runtime facts，不解析 assistant 文本充当事实源 |
| CodexMonitor | workspace/thread、queue vs steer、reasoning/tool/diff、approval、terminal dock | 桌面端 Plugin 的工具 UI 应和工作区、文件、终端、审批、diff 一体化 |
| Claude Code `MCPTool/UI.tsx` | input 截断、progress、大输出 token warning、image placeholder、small JSON flatten | Lime 对 MCP / workbench / browser 工具需要输入摘要、progress、长输出保护和结构化结果 |

## 3. 当前工具事实源

| 事实源 | 结论 |
| --- | --- |
| `catalog.rs` | 固定 catalog 共 `60` 个条目：`58 current`、`1 compat`、`1 deprecated` |
| Surface | `Core 42`、`Workbench 12`、`BrowserAssist 6` |
| Permission | `ParameterRestricted 12`、`SessionAllowlist 47`、`CallerFiltered 1` |
| 默认允许 | `workspace_default_allow=true 45`、`false 15` |
| 动态工具 | `mcp__lime-browser__*`、其他 MCP server tools 不能按固定清单评估，必须用 runtime inventory snapshot 评估 |
| 前端 family | 当前有 `subagent / task / plan / skill / write / read / edit / command / search / list / browser / fetch / vision / generic` |

逐项表见 `tools-inventory.md`。该表把 60 个 catalog 条目逐项映射到 surface、lifecycle、permission、UI family、现有证据、静态分和必须补证。

## 4. 分组评分

| 领域 | 固定工具 | 当前分 | 已有能力 | 主要缺口 |
| --- | --- | ---: | --- | --- |
| Workspace IO / Coding | `Read`、`Write`、`Edit`、`apply_patch`、`Glob`、`Grep`、`LSP`、`NotebookEdit`、`view_image`、`EnterWorktree`、`ExitWorktree` | 3.3 | 文件读写编辑、diff review、图片预览、路径摘要、历史别名归一化 | `Read/Write/Edit/NotebookEdit/LSP` 缺真实 GUI coding loop 证据，缺权限审批与 diff/test outcome 贯通 |
| Command / Execution | `Bash`、`PowerShell`、`Workflow`、`Sleep`、`RemoteTrigger`、`TaskOutput`、`TaskStop` | 3.3 | command summary、stdout/stderr 分流、exit code、sandbox、offload 提示、命令分组 | 缺 PTY / resize / interrupt / retry 的桌面实测，`Workflow/Sleep` 仍偏 generic |
| Plan / Session / User Interaction | `TaskCreate`、`TaskList`、`TaskGet`、`TaskUpdate`、`EnterPlanMode`、`ExitPlanMode`、`AskUserQuestion`、`SendUserMessage`、`StructuredOutput`、`Config`、`Cron*` | 3.1 | 计划类 family、交互类动作句、cron / config 历史展示、用户消息展示 | 缺 plan-ready / needs-input 的完整桌面状态机，`StructuredOutput` 还没有和 final answer evidence 强绑定 |
| Search / MCP tool card rendering | `WebSearch`、`WebFetch`、`ToolSearch`、`ListMcpResourcesTool`、`ReadMcpResourceTool`、动态 MCP tools | 3.5 | WebSearch 列表与悬浮预览、ToolSearch 结构化摘要、MCP search/list/read/mutation 分类 | `WebFetch`、MCP resource helpers 缺真实页面 / resource 预览证据；完整 MCP 系统见 `mcp.md` |
| Skill tool call rendering | `Skill` | 3.6 | Skill 调用能展示本次读取的 `SKILL.md` snapshot，隐藏内部 metadata，有组件测试 | 只作为工具卡渲染评分；required vs invoked、权限、资源、触发与包系统见 `skills.md` |
| Workbench Content | `social_generate_cover_image`、`lime_create_*`、`lime_search_web_images`、`lime_run_service_skill` | 3.2 | 内容任务 label / failure 文案、协议噪声隐藏、图片结果预览、任务结果 preview | 多数 `lime_create_*` 缺逐工具真实任务 outcome，`lime_run_service_skill` 是 compat，`video` 是 deprecated 不应进入 current P0 |
| Browser / Site | `lime_site_*`、`mcp__lime-browser__*` | 3.5 | site run saved content、Markdown 导出、未保存原因、browser action 文案、动态 browser family | 站点目录/推荐/搜索/详情缺 GUI 结果面板证据，browser 需要 WebArena / OSWorld 风格轨迹与截图 |
| Delegation / Team | `Agent`、`SendMessage`、`TeamCreate`、`TeamDelete`、`ListPeers` | 3.4 | subagent/team/list peers 动作句、分组展示、历史兼容别名 | 缺多 agent lineage、handoff、worker output、team roster、失败恢复的桌面 evidence |

## 5. 5 分工具 UI 标准

| 能力 | 5 分要求 | Lime 当前判断 |
| --- | --- | --- |
| 生命周期 | started、args streaming、input available、running/progress、result、failed、cancelled 全可见 | running/completed/failed 基础可见，args streaming/progress/cancelled 仍不足 |
| 输入摘要 | 默认显示安全关键字段，长输入截断，可展开，secret key 不 inline | 有 primary subject 与部分参数摘要，secret / 大 payload 规则还需专项验证 |
| 输出预览 | 小输出 inline，大输出 offload 到 file / artifact / evidence，支持 copy / open | command/offload/site/image/diff 有基础，通用大输出与 MCP 大结果仍需加强 |
| Progress | 支持百分比、阶段文案、长任务状态，后台任务只在需要用户注意时抢焦点 | 目前多为 running label，缺通用 progress bar 与 long-running state |
| 错误恢复 | failed reason、retry、copy diagnostics、open detail、fallback 可见 | 已隐藏协议噪声，但 retry / diagnostics / fallback 还不足 |
| Skill tool call 卡 | 本次 `Skill` 工具调用读取的 `SKILL.md` 内容可查，内部 metadata 不直接泄露 | invoked `SKILL.md` snapshot 有基础；Skill 系统级可见性见 `skills.md` |
| Dynamic tool card | MCP / browser / extension 工具能按 runtime inventory 展示来源、可见性、允许 caller、调用方式 | inventory API 有类型和 UI section，缺与 ToolCallDisplay 的端到端联动评分；MCP server 系统评分见 `mcp.md` |

## 6. P0 实测矩阵

| 场景 | 覆盖工具 | 必须通过 |
| --- | --- | --- |
| `tool-workspace-read-edit-patch` | `Read`、`Edit`、`apply_patch`、`Grep`、`Glob` | 工具 args、diff、文件路径、最终 answer、文件变更一致；权限受控 |
| `tool-command-failure-recovery` | `Bash`、`PowerShell`、`TaskOutput`、`TaskStop` | stdout/stderr、exit code、sandbox、失败原因、重跑或终止动作可见 |
| `tool-large-output-offload` | `Bash`、`WebFetch`、MCP mutation/read | 大输出不刷屏，有 summary、warning、result file / artifact ref；MCP 专项 evidence 见 `mcp.md` |
| `tool-websearch-grounded-answer` | `WebSearch`、`ToolSearch`、动态 MCP web search | 搜索 query、结果列表、来源 URL、raw detail、最终引用一致 |
| `tool-skill-call-rendering` | `Skill` | 本次工具调用、读取的 `SKILL.md` snapshot、隐藏 metadata、失败摘要可查；系统级 required/invoked 见 `skills.md` |
| `tool-content-workbench-task` | `lime_create_image_generation_task`、`lime_create_audio_generation_task`、`social_generate_cover_image` | 任务发起、任务 id / saved artifact、失败不泄露协议噪声、图片/音频结果可打开 |
| `tool-site-run-saved-content` | `lime_site_list`、`lime_site_search`、`lime_site_info`、`lime_site_run` | 目录/搜索/详情/执行链路可见，保存内容和导出 Markdown 可打开 |
| `tool-browser-dynamic-actions` | `mcp__lime-browser__*` | navigate/click/type/screenshot/logs 有逐步轨迹、截图或 snapshot、失败原因和 cleanup |
| `tool-team-delegation` | `Agent`、`SendMessage`、`TeamCreate`、`ListPeers`、`TaskOutput` | 子任务 lineage、worker 状态、handoff message、最终结果引用可见 |
| `tool-runtime-inventory` | 全部 catalog + MCP / extension | inventory 展示 catalog/runtime/extension/MCP counts、visible、caller_allowed、policy，和实际工具卡一致；MCP server 级能力见 `mcp.md` |

## 7. 阻断缺口

| 缺口 | 影响 | 下一刀 |
| --- | --- | --- |
| 缺逐工具 GUI evidence | 当前只能说“静态支持”，不能说“产品级支持” | 把 `tools-inventory.md` 中 `<=3.0` 的 current 工具转成 P0/P1 GUI flows |
| 动态 MCP / browser 工具卡没有 snapshot 回归 | MCP 工具会随环境变化，固定表无法证明真实可用 | 每次评测保存 `agentSession/toolInventory/read` snapshot；server 级 P0 见 `mcp.md` |
| Progress 与 cancel 事件弱 | 长任务、browser、content generation 体验会像“卡住” | 对齐 AG-UI / AI SDK tool states 增加 progress/cancel rendering |
| 大输出策略不统一 | MCP / WebFetch / command 可能刷屏或吞结果 | 统一 offload/ref/artifact/detail UI 与 token warning |
| compat 混在 current 表现里 | 容易误把历史能力当主线能力 | `lime_run_service_skill` 只按兼容评测；视频只评测 current `video_generate` typed tool |

## 8. 结论

Lime 对工具 UI 已经有比普通聊天更强的基础：catalog、runtime inventory、family 映射、ToolCallDisplay、diff review、WebSearch 预览、Skill tool call snapshot、site saved content、command summary 都存在。问题是这些能力目前分布不均：核心命令、WebSearch、ToolSearch、Skill tool call、site run 较强，Notebook/LSP/Workflow/Sleep/WebFetch/MCP resource/content long-running/browser dynamic tools 缺逐工具 GUI 证据。

因此当前静态分是 `3.3 / 5`。要升到 `4.0+`，优先把 `tools-inventory.md` 里的低分 current 工具补成可重复 GUI evidence，而不是继续增加新工具名。
