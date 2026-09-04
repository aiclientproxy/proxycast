# Agent Workspace 工具逐项评测表

> 状态：static-inventory-audit  
> 更新时间：2026-06-15  
> 范围：固定 catalog 来自 `lime-rs/crates/agent/src/agent_tools/catalog.rs`，前端展示来自 `src/components/agent/chat/utils/toolDisplayInfo.ts` 与 `ToolCallDisplay*` 测试。动态 MCP / browser 工具在本文只按工具卡渲染和 inventory snapshot 评测；完整 MCP 系统见 `mcp.md`。

## 1. 总览

| 项 | 数量 |
| --- | ---: |
| 固定 catalog 条目 | 60 |
| Current | 58 |
| Compat | 1 |
| Deprecated | 1 |
| Core surface | 42 |
| Workbench surface | 12 |
| BrowserAssist surface | 6 |
| ParameterRestricted | 12 |
| SessionAllowlist | 47 |
| CallerFiltered | 1 |
| workspace 默认允许 | 45 |
| workspace 默认不允许 | 15 |

## 2. 评分规则

| 分数 | 解释 |
| --- | --- |
| `4.0+` | 有专用 UI 或强组件测试，缺少真实 GUI evidence 才不能满分 |
| `3.0-3.9` | catalog + 前端 family 可读，有部分测试或通用渲染，但缺逐工具实测 |
| `2.0-2.9` | catalog 存在，UI 偏 generic 或缺关键结果面板 |
| `<2.0` | compat / deprecated / 不应作为 current P0 能力 |

## 3. 逐项表

| # | 工具 | surface | capability | lifecycle | permission | 默认 | UI family | 当前证据 | 静态分 | 必须补证 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| 1 | `Read` | Core | WorkspaceIo | Current | ParameterRestricted | 否 | read | `toolDisplayInfo` 文件读取；别名测试覆盖 `FileReadTool/read_file` | 3.4 | 真实 GUI 中读取文件、路径、内容预览、final answer 引用一致 |
| 2 | `Write` | Core | WorkspaceIo | Current | ParameterRestricted | 否 | write | 写文件 label；artifact path 提取测试覆盖嵌套路径 | 3.4 | 写入前后文件证据、权限提示、artifact / diff 关联 |
| 3 | `Edit` | Core | WorkspaceIo | Current | ParameterRestricted | 否 | edit | 文件编辑 family 与别名归一化 | 3.1 | 单文件编辑 GUI evidence、失败恢复、diff 关联 |
| 4 | `apply_patch` | Core | WorkspaceIo | Current | ParameterRestricted | 否 | edit | 文件级 diff review、多文件 scope、长 patch 展开、canvas 打开测试 | 4.2 | 真实 coding loop 中 patch、测试、最终说明闭环 |
| 5 | `Glob` | Core | WorkspaceIo | Current | ParameterRestricted | 否 | list | 文件匹配 label；动作句测试覆盖 pattern | 3.4 | 大目录结果截断、空结果、路径点击 |
| 6 | `Grep` | Core | WorkspaceIo | Current | ParameterRestricted | 否 | search | 内容检索 family | 3.1 | 匹配行预览、上下文行、无匹配语义成功 |
| 7 | `Bash` | Core | Execution | Current | ParameterRestricted | 否 | command | command summary、stdout/stderr、exit code、sandbox、offload、diff stdout 测试 | 4.1 | PTY、interrupt、retry、长运行 progress 的 GUI smoke |
| 8 | `LSP` | Core | WorkspaceIo | Current | ParameterRestricted | 否 | read | 代码分析 label；历史 gated 工具可读测试 | 3.0 | definition/diagnostics/references 结果结构化面板 |
| 9 | `Skill` | Core | SkillExecution | Current | SessionAllowlist | 是 | skill | `SKILL.md` snapshot 面板、metadata 隐藏、InlineToolProcessStep 测试 | 3.6 | 仅评分 tool call rendering；完整 Skill 系统见 `skills.md` |
| 10 | `Workflow` | Core | Execution | Current | SessionAllowlist | 是 | generic | 工作流执行 label | 2.8 | workflow step/progress/result 面板，避免只显示 generic |
| 11 | `TaskCreate` | Core | Planning | Current | SessionAllowlist | 是 | plan | 计划类 label 与动作文案 | 3.0 | 创建任务后的计划状态、task id、后续状态联动 |
| 12 | `TaskList` | Core | Planning | Current | SessionAllowlist | 是 | plan | 任务列表 label | 3.0 | 列表结果结构化，而非 raw JSON |
| 13 | `TaskGet` | Core | Planning | Current | SessionAllowlist | 是 | plan | 任务详情 label | 3.0 | 任务详情字段、状态、owner、时间展示 |
| 14 | `TaskUpdate` | Core | Planning | Current | SessionAllowlist | 是 | plan | 任务更新 label | 3.0 | update diff、状态变更、失败恢复 |
| 15 | `TaskOutput` | Core | Execution | Current | SessionAllowlist | 是 | task | 动作句测试覆盖 task id；可与命令分组共存 | 3.5 | 子任务 / background task 输出与原任务 lineage 绑定 |
| 16 | `TaskStop` | Core | Execution | Current | SessionAllowlist | 是 | task | 终止任务 label；`KillShell` 归一化 | 3.1 | stop 后 runtime 状态、残留输出、取消原因 |
| 17 | `NotebookEdit` | Core | WorkspaceIo | Current | ParameterRestricted | 否 | edit | notebook edit label | 2.6 | notebook cell diff、执行结果、失败定位 |
| 18 | `view_image` | Core | WorkspaceIo | Current | ParameterRestricted | 否 | vision | 图片查看 label；图片结果 overlay / source 归一化测试 | 3.6 | 本地图片路径、缩放、失败、model-visible image 证据 |
| 19 | `EnterPlanMode` | Core | Planning | Current | SessionAllowlist | 是 | plan | 进入计划模式 label | 3.0 | plan mode banner、approve/exit、turn 状态贯通 |
| 20 | `ExitPlanMode` | Core | Planning | Current | SessionAllowlist | 是 | plan | 退出计划模式 label | 3.0 | plan-ready / approval CTA 与 runtime resolved |
| 21 | `EnterWorktree` | Core | WorkspaceIo | Current | SessionAllowlist | 是 | generic | 工作树 label；历史 gated 展示测试 | 3.2 | worktree 路径、branch、隔离状态、退出清理 |
| 22 | `ExitWorktree` | Core | WorkspaceIo | Current | SessionAllowlist | 是 | generic | 工作树 label；历史 gated 展示测试 | 3.2 | 退出后的 workspace 恢复与变更归档 |
| 23 | `WebFetch` | Core | WebSearch | Current | ParameterRestricted | 否 | fetch | 网页抓取 label | 3.0 | URL、状态码、正文摘要、来源、失败原因 |
| 24 | `WebSearch` | Core | WebSearch | Current | SessionAllowlist | 是 | search | 搜索列表、悬浮预览、raw detail、连续搜索分组测试 | 4.1 | 真实联网 query、引用最终答案、来源质量评分 |
| 25 | `AskUserQuestion` | Core | Planning | Current | SessionAllowlist | 是 | generic | 等待输入动作句；正式工具卡隐藏原始名测试 | 3.6 | 多选/自由输入/超时/取消与 runtime action_required |
| 26 | `SendUserMessage` | Core | SessionControl | Current | SessionAllowlist | 是 | generic | 用户消息动作句测试 | 3.5 | 消息发送与 transcript / visible bubble 一致 |
| 27 | `StructuredOutput` | Core | SessionControl | Current | SessionAllowlist | 否 | generic | 最终答复 label | 3.0 | 结构化结果和 final answer / artifact schema 对齐 |
| 28 | `Config` | Core | SessionControl | Current | SessionAllowlist | 是 | generic | 配置 label；历史 gated 可读测试 | 3.2 | 配置变更前后、来源、回滚 |
| 29 | `Sleep` | Core | Execution | Current | SessionAllowlist | 是 | generic | 等待 label | 2.6 | 倒计时、取消、长等待不误判卡死 |
| 30 | `PowerShell` | Core | Execution | Current | ParameterRestricted | 否 | command | command summary 单测覆盖 PowerShell、encoding、sandbox | 3.6 | Windows GUI smoke、stderr encoding、权限提示 |
| 31 | `RemoteTrigger` | Core | Execution | Current | SessionAllowlist | 是 | command | 远程触发动作句；历史失败隐藏协议噪声 | 3.3 | 远程状态、重试、触发来源、失败诊断 |
| 32 | `CronCreate` | Core | Planning | Current | SessionAllowlist | 是 | task | cron create label；历史动作句测试 | 3.2 | schedule 参数、下一次触发、启停状态 |
| 33 | `CronList` | Core | Planning | Current | SessionAllowlist | 是 | list | cron list label | 3.1 | cron 列表结构化、空态、筛选 |
| 34 | `CronDelete` | Core | Planning | Current | SessionAllowlist | 是 | task | cron delete 动作句测试 | 3.2 | 删除确认、删除后列表刷新 |
| 35 | `ToolSearch` | Core | WebSearch | Current | SessionAllowlist | 是 | search | 结构化工具摘要、streaming 不自动展开、分组子行测试 | 4.0 | 与 runtime inventory 的实际可调用状态一致 |
| 36 | `ListMcpResourcesTool` | Core | WebSearch | Current | SessionAllowlist | 是 | list | MCP resource list label | 3.2 | 仅评分 resource helper 工具卡；完整 resources / templates / subscriptions 见 `mcp.md` |
| 37 | `ReadMcpResourceTool` | Core | WebSearch | Current | SessionAllowlist | 是 | read | MCP resource read label | 3.2 | 仅评分 resource helper 工具卡；完整 resource grounding 见 `mcp.md` |
| 38 | `Agent` | Core | Delegation+SessionControl | Current | SessionAllowlist | 是 | subagent | 子任务动作句测试 | 3.6 | spawn lineage、子线程链接、状态、失败恢复 |
| 39 | `SendMessage` | Core | Delegation+SessionControl | Current | SessionAllowlist | 是 | subagent | 补充说明 label | 3.4 | 发送到哪个 agent / team、回执、失败 |
| 40 | `TeamCreate` | Core | Delegation+SessionControl | Current | SessionAllowlist | 是 | subagent | team create 动作句测试 | 3.5 | team roster、成员状态、权限 |
| 41 | `TeamDelete` | Core | Delegation+SessionControl | Current | SessionAllowlist | 是 | subagent | team delete 动作句测试 | 3.5 | 删除确认、残留任务处理 |
| 42 | `ListPeers` | Core | Delegation+SessionControl | Current | SessionAllowlist | 是 | list | list peers 动作句测试 | 3.5 | peers 列表结构化、在线/忙碌/失败状态 |
| 43 | `social_generate_cover_image` | Workbench | ContentCreation | Current | SessionAllowlist | 是 | task | 内容任务 copy、封面生成过程文案、图片结果基础 | 3.5 | 真实生成任务、图片 artifact、重试与失败诊断 |
| 44 | `video_generate` | Workbench | ContentCreation | Current | SessionAllowlist | 是 | task | typed gateway 委托 App Server `mediaTaskArtifact/video/create` | 1.5 | current 视频任务创建工具；旧 branded facade 禁止回流 |
| 45 | `lime_create_audio_generation_task` | Workbench | ContentCreation | Current | SessionAllowlist | 是 | task | 配音生成 label 与过程文案测试 | 3.3 | 音频任务 id、进度、结果播放/下载、失败 |
| 46 | `lime_create_transcription_task` | Workbench | ContentCreation | Current | SessionAllowlist | 是 | task | 转写 label、sourcePath 过程文案测试 | 3.3 | 转写结果预览、时间轴、导出 |
| 47 | `lime_create_broadcast_generation_task` | Workbench | ContentCreation | Current | SessionAllowlist | 是 | task | 口播生成 label | 3.0 | 口播任务进度、结果预览、失败 |
| 48 | `lime_create_cover_generation_task` | Workbench | ContentCreation | Current | SessionAllowlist | 是 | task | 封面生成 label | 3.0 | 与封面图 artifact / 图片预览打通 |
| 49 | `lime_create_modal_resource_search_task` | Workbench | ContentCreation | Current | SessionAllowlist | 是 | task | 素材检索 label 与过程文案测试 | 3.3 | 素材列表、来源、选择、保存 |
| 50 | `lime_search_web_images` | Workbench | WebSearch | Current | SessionAllowlist | 是 | search | 联网搜图 label；图片 preview 基础能力 | 3.2 | 图片来源、授权、批量预览、引用 |
| 51 | `lime_create_image_generation_task` | Workbench | ContentCreation | Current | SessionAllowlist | 是 | task | 图片生成 failure 不泄露协议噪声；图片结果 preview | 3.6 | 任务状态、生成结果 artifact、重试、prompt 保护 |
| 52 | `lime_create_url_parse_task` | Workbench | ContentCreation | Current | SessionAllowlist | 是 | task | 链接解析 label 与过程文案测试 | 3.3 | URL 解析结果、引用、失败原因 |
| 53 | `lime_create_typesetting_task` | Workbench | ContentCreation | Current | SessionAllowlist | 是 | task | 排版 label 与过程文案测试 | 3.3 | 排版 artifact、预览、导出、失败 |
| 54 | `lime_run_service_skill` | Workbench | Execution | Compat | SessionAllowlist | 是 | generic | 兼容服务技能 label 与过程文案测试 | 2.4 | 不作为 current truth；迁到 site / skill current 路径或退场 |
| 55 | `lime_site_list` | BrowserAssist | BrowserRuntime+WebSearch | Current | SessionAllowlist | 是 | list | 站点目录过程文案测试 | 3.3 | 目录结果卡、登录要求、参数摘要 |
| 56 | `lime_site_recommend` | BrowserAssist | BrowserRuntime+WebSearch | Current | SessionAllowlist | 是 | search | 站点推荐过程文案测试 | 3.4 | 推荐理由、可执行状态、下一步动作 |
| 57 | `lime_site_search` | BrowserAssist | BrowserRuntime+WebSearch | Current | SessionAllowlist | 是 | search | 站点搜索过程文案测试 | 3.4 | 搜索结果列表、过滤、空态 |
| 58 | `lime_site_info` | BrowserAssist | BrowserRuntime+WebSearch | Current | SessionAllowlist | 是 | read | 站点详情过程文案测试 | 3.4 | 参数 schema、登录要求、示例输入 |
| 59 | `lime_site_run` | BrowserAssist | BrowserRuntime+WebSearch | Current | SessionAllowlist | 是 | generic | saved content、Markdown 导出、打开已保存内容、未保存原因测试 | 4.0 | 真实 attached session、browser trace、站点结果 evidence |
| 60 | `mcp__lime-browser__*` | BrowserAssist | BrowserRuntime | Current | CallerFiltered | 否 | browser | 动态 browser matcher、navigate/click/type/screenshot/logs 动作句测试 | 3.6 | 本表仅评估 browser MCP 工具卡；server / auth / evidence 见 `mcp.md` |

## 4. 动态工具评测规则

固定表无法枚举所有 MCP / extension 工具。本节只评估动态工具在 Agent Workspace 工具卡中的可见性；MCP server、tools/resources/prompts、auth、elicitation 和 evidence 的完整评分见 `mcp.md`。

| 动态来源 | 事实源 | 必须记录 | 评分规则 |
| --- | --- | --- | --- |
| MCP server tools | `agentSession/toolInventory/read.mcp_tools` | `server_name`、`name`、`caller_allowed`、`visible_in_context`、`tags`、`input_examples_count` | search/list/read/browser/mutation 分类正确得基础分；完整 server 级能力见 `mcp.md` |
| Runtime extension tools | `extension_tools`、`runtime_tools` | extension 名称、source kind、deferred loading、visible | 证明 discover、load、call、result 四段，不只显示 catalog |
| Browser compatibility | `mcp__lime-browser__*` prefix | 实际 inner tool、页面 URL、操作、截图/snapshot、失败 | 按 WebArena / OSWorld 口径评估 task success、trajectory、state verification |
| Resource helper tool cards | `ListMcpResourcesTool`、`ReadMcpResourceTool` | server、resource URI、mime / content type、preview/offload | 工具卡必须可读；resource grounding / templates / subscriptions 见 `mcp.md` |

## 5. 优先级清单

| 优先级 | 工具 | 原因 |
| --- | --- | --- |
| P0 | `Read`、`Write`、`Edit`、`apply_patch`、`Bash`、`Grep`、`Glob`、`WebSearch`、`ToolSearch` | Codex App / Claude Code 类桌面 agent 的基本工具闭环；`Skill` 系统 P0 见 `skills.md` |
| P0 | `AskUserQuestion`、`TaskOutput`、`TaskStop`、`lime_site_run`、`mcp__lime-browser__*` | HITL、长任务、站点执行、browser 自动化会直接影响后台输出 |
| P1 | `PowerShell`、`LSP`、`view_image`、`ListMcpResourcesTool`、`ReadMcpResourceTool`、`Agent`、`TeamCreate`、`ListPeers` | 跨平台、代码理解、MCP helper tool cards、团队协作关键能力；MCP 系统 P0 见 `mcp.md` |
| P1 | `lime_create_image_generation_task`、`lime_create_audio_generation_task`、`lime_create_transcription_task`、`lime_create_modal_resource_search_task` | Workbench 内容生产的高频 current 工具 |
| P2 | `Workflow`、`Sleep`、`Config`、`Cron*`、`StructuredOutput`、`NotebookEdit` | 需要补 UI 语义，但不应抢 P0 coding/search/browser 闭环 |
| 退场 | `lime_run_service_skill` | compat，不作为 current 标准能力宣传；旧 `lime_create_video_generation_task` 已从 catalog 删除并由 guard 禁止回流 |

## 6. 结论

Lime 当前不是“没有工具 UI”，而是“工具 UI 证据分布不均”。强项集中在 `Bash`、`apply_patch`、`WebSearch`、`ToolSearch`、`Skill` tool call rendering、`lime_site_run`；中间层工具大多有 family 和过程文案；低分项集中在缺少专用结果面板或真实 GUI evidence 的工具，以及 compat / deprecated 工具。

下一步应从本表中选择 P0 工具补 GUI evidence，而不是继续泛泛描述“支持工具使用”。
