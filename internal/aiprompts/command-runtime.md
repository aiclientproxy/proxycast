# Lime 命令运行时实施手册

## 这份文档回答什么

本文件定义 Lime 中 `@` 原子命令、产品型 `/` 场景命令及其结果卡 / viewer 的实施规则，主要回答：

- 什么时候一个需求已经属于“命令运行时改动”，而不是普通 UI 或普通 skill 改动
- `@`、`/`、`skill`、`ServiceSkill`、`task`、`viewer` 之间的固定关系是什么
- 服务端统一目录与客户端 seeded / fallback 应该如何配合
- 为什么命令能力不能“先写代码再补 PRD”
- 新增一个命令功能时，最少要先补哪些设计文档
- 公共设计包和单功能方案包分别放在哪里

它是 **命令运行时的长期指导文档**，不是某个单一功能的 PRD。

## 什么时候先读

遇到以下任一情况时，先读本文件，再决定是否开始改代码：

- 新增或调整 `@` 命令
- 新增或调整产品型 `/` 场景命令
- 把某个能力接到“聊天轻卡 + 右侧查看区”主链
- 调整 `ServiceSkill` 与 slash 场景的关系
- 调整配图、视频、转写、修图这类异步 task 型能力
- 调整命令恢复、作用域、重试、取消、viewer 描述
- 想为某个能力补“完整方案包”

如果当前需求已经涉及“命令触发 -> Agent 分析 -> binding -> 轻卡 -> viewer”其中两步以上，就默认属于命令运行时改动。
如果命令还涉及网页登录态访问、网页导出、Markdown 保存、图片下载，再补读 [web-browser-scene-skill.md](web-browser-scene-skill.md)。

## 固定产品判断

Lime 的命令体系固定按以下关系理解：

1. `@` 是能力原子入口  
   它表达“系统具备什么能力”，不是最终执行器类型。

2. `/` 是场景组合入口  
   它表达“为了某个用户目标，系统如何把多个能力编排起来”。

3. `ServiceSkill` 是成熟场景真相  
   首页场景卡、产品型 slash、slot filling、delivery 语义优先对齐 `ServiceSkill`。

4. `skill` 是能力绑定抽象  
   它背后可能是：
   - 本地 CLI
   - 本地 runtime
   - 浏览器 / workspace / automation 执行器
   - 目录控制面下发的配置绑定

5. `task file` 只是异步媒体/资源能力的真相之一  
   它不是所有命令的统一产品真相。

6. UI 的正式消费对象是统一 `CommandRunSnapshot`  
   聊天区轻卡和右侧 viewer 不应直接绑定底层 task、run 或原始响应结构。

补充边界：

- [Codex `/goal`](../research/codex-goal/README.md) 是 persistent objective / continuation loop 参考，不是 Lime 产品型 `/` 场景命令模板。
- 如果后续 Lime 出现目标推进入口，它也必须触发现有 `ServiceSkill / Scheduled Task / agent turn` 主链，而不是在 slash 层新增一套 goal 执行壳。

## 创作主线护栏

当前 Lime 的命令运行时默认服务“创作生产与交付”主线。

这意味着：

1. 一级优先命令应优先覆盖创作生成、素材获取、研究拆解、发布交付。
2. 搜索、浏览器、网页读取、代码等能力只有在能明确支撑创作主链时，才应进入当前命令建设优先级。
3. `@发布合规` 的定位是创作交付前的风险检查，只回答“这份内容能不能发、风险在哪里、怎么改”，不是泛法务协议。
4. 如果一个新命令主要服务泛办公、泛法务或泛开发场景，而不能回挂到创作主线，应先暂停并重新论证优先级。
5. `scene` 的命名、推荐文案和补参文案也应优先使用创作语义，例如选题、脚本、配图、转写、发布预览、发布合规；不要默认长出“建立”“法务”这类脱离创作目标或过泛的场景表达。

## 固定主链

所有命令能力统一按这条主链设计：

`命令触发 -> Agent 分析 -> skills / tools / workflow / task / ServiceSkill binding -> 聊天区轻量结果卡 -> 右侧查看区`

这条主链意味着：

- Agent 是编排层
- binding 是执行分发层
- truth source 是状态事实层
- 轻卡是用户第一反馈层
- viewer 是详情查看层

### Agent 执行环境与路径规则

Agent turn 必须把当前 OS、工作目录、shell 运行时和本机路径格式注入系统提示，作为执行环境事实源。

- Windows 原生运行时默认使用 PowerShell 语义和 Windows 路径，不能把系统盘 / C 盘猜成 `/mnt/c`
- `/mnt/<drive>` 只适用于明确处于 WSL / Linux shell 的会话；没有证据时，先查询 `$env:SystemDrive`、`$env:SystemRoot` 或 `Get-PSDrive -PSProvider FileSystem`
- macOS / 非 Windows 运行时不能臆造 `C:\`、`D:\` 等盘符；用户要求 Windows 专属路径时，应说明当前环境不可直接访问或请求确认映射
- 本规则是执行主链的一部分，不应下沉为某个 skill 的私有提示；skill 可提供平台路由示例，但不能覆盖运行时事实源

对图片任务再补一条固定约束：

`@配图/@修图/@重绘`、文稿 inline 配图、封面位、图片工作台编辑/变体和统一目录显式声明的图片模型标签，current 主链统一是 `Agent turn -> App Server ImageCommandWorkflow -> 标准 image task artifact + worker`。前端只负责把 `entry_source / provider_id / model / executor_mode / modality_contract_key / runtime_contract / routing_slot` 等上下文并入 `harness.image_command_intent.image_task`，发送前的 `session_id` 绑定仍必须走统一发送边界。App Server 在普通聊天模型路由前识别 `ImageCommandIntent`：参数完整时创建 `.lime/tasks/image_generate/*.json`，参数不足时返回补参状态，provider/model 不可执行时 fail closed。`Skill(image_generate)`、旧 `harness.image_skill_launch`、`Bash -> lime media image generate --json`、`lime task create image --json` 和前端直接 `create_image_generation_task_artifact` 都不能再作为聊天 @图片命令首发事实源；旧 `harness.image_skill_launch` 已判 `dead / forbidden-to-restore`，只能在剥离逻辑、retired guard 和负向测试中出现。未在 catalog 中声明的任意 `@模型名` 不得自动变成图片 API 入口。图片 command 还必须显式压制 `ToolSearch / WebSearch / Read / Glob / Grep` 这类通用偏航工具，避免模型在“搜技能目录”里空转或把权限错误暴露给用户。

对所有 skills 再补一条全局约束：

- 是否走 `Bash CLI`，由 binding / executor / runtime 或显式操作者决定，不由模型在首刀自由写 shell。
- 全局优先级固定为原生结构化 binding；旧 `local_cli` 与自由形态 `Bash CLI` 任务入口均为 `dead / forbidden-to-restore`。
- 第一判断维度不是 `key` 在本地还是 OEM 云端，而是“谁真正执行、正式真相源写到哪里、viewer 读谁”。
- 如果只是凭证托管在 OEM 云端，但执行仍在客户端，本质上不等于“应该走本地 CLI”；它可能仍是 `server_api` 或 `hybrid`。
- 如果执行本身就在 OEM / 服务端 runtime，当前应优先走 `server_api` 或 `hybrid`，而不是为了统一表面形式强行绕回 `Bash CLI`。
- 只有当参数已经定稿、CLI 是唯一稳定执行 facade、且输出能稳定映射回同一真相源时，才允许把 CLI 作为 direct entry。
- 只要当前还需要模型做补参、作用域绑定、viewer 回填或结构化 metadata 投影，就不要把 CLI 当 current 首发路径。
- 共享决策锚点见 `internal/roadmap/gongneng/command-runtime/architecture.md`；单功能如 `@配图` 还要继续服从各自 `internal/prd/gongneng/<feature>/architecture.md`。

- 显式图片动作允许先在前端补 `image_command_intent` metadata；如果仍读到旧 `image_skill_launch`，发送边界必须剥离或 fail closed，不能在 App Server 入口再标准化成 current intent。发送前的 `session_id` 绑定仍必须走统一发送边界；如果 metadata 里暂时还是本地 draft key，必须在真正发起 send 时替换成真实会话 ID，而不是在图片动作入口提前额外建一个图片专用会话。
- `.lime/tasks/**/*.json` 继续作为图片主链的唯一恢复事实源，但它们属于内部任务快照，默认不应直接渲染成用户可见 artifact 卡片或时间线文件卡；用户面看到的应该是轻结果卡、工具过程和右侧查看。

图片结果进入 UI 时还必须遵守以下 viewer 收口规则：

- 图片任务的主结果事实源是 `image task preview + 图片工作台 outputs`，不是通用文本 artifact
- 空内容的二进制图片文件（如 `output_image.jpg`）不能再镜像成通用 artifact 卡片，否则会出现“重复文件卡 + 点不开”的假结果
- `tool_result` 产物在 general workspace 中默认后台入库，不自动选中、不自动展开右侧工作台；抢焦点只允许发生在用户显式点击或仍在流式写入的文档类产物上
- 同一产物路径的 `basename / 相对路径 / 绝对路径` 必须在前端视为同一文件，避免一张图被重复挂成多份结果

不要再把命令能力直接叙述成：

- “前端某个按钮直接调接口”
- “某个工作台自己维护一套状态”
- “viewer 自己推断任务状态”

## 统一目录与兜底规则

命令运行时的可发现性必须统一收敛到同一份目录协议，而不是前端各处各写一份静态数组。

当前固定规则如下：

1. `SkillCatalog.entries` 是当前统一目录投影。
2. `entries.kind=command` 驱动 `@` 原子命令。
3. `entries.kind=scene` 驱动产品型 `/` 场景命令。
4. `entries.kind=skill` 驱动首页技能卡、技能中心、启动推荐和补参入口。
5. 在线主路径优先消费：
   - `bootstrap.skillCatalog`
   - `GET /v1/public/tenants/{tenantId}/client/skills`
6. 客户端必须保留本地 seeded catalog 作为韧性兜底：
   - 未登录
   - 服务端未升级
   - 远端拉取失败
   - 返回 legacy `items` 但未返回 `entries`
7. 如果服务端暂时只返回 legacy `items`，客户端允许在网关层兼容构造 `entries`，但这只是 compat 过渡，不是新的长期事实源。
8. 输入区、提及面板、slash 场景面板、首页技能入口都应消费同一份 catalog selector；不要继续在组件内维护第二套硬编码命令列表。
9. 如果服务端下发了 Lime 尚未支持的展示类型，优先由服务端回退到已有 `renderContract`；客户端也必须退化到通用 `tool_timeline` 或 `artifact` 展示，而不是直接失能。

当前 `scene` slash 的第一刀执行也固定如下：

- `useWorkspaceSendActions` 先识别 `/scene-key ...`
- 从统一 `SkillCatalog.entries` 里解析 `scene -> linkedSkillId -> ServiceSkillHomeItem`
- 前端只负责把结构化 `service_scene_launch` 写进当前 turn metadata，不负责任何服务端 run 创建
- Rust 侧会把该 turn 收口到 `workbench`，并通过系统提示强约束 Agent 直接按当前本地 `service_scene_launch` 上下文执行；如果实现里仍沿用 `lime_run_service_skill` 一类历史命名，也只允许视为 compat 护栏
- `service_scene_launch` 只表达目录命中和本地运行时路由提示，保证 slash scene 继续走 `Agent -> tool -> timeline` 主链
- 未命中统一目录的 slash 文本必须继续回到普通 slash 流程，不能被错误吞成“未找到本地 Skill”

当前 `scene` slash 还必须遵守下面三条长期规则：

- `Scene Skill` 是产品场景真相；slash 只是触发入口，不能在前端把流程写死成某个站点分支
- 推荐用 `Pipeline` 作为主模式，再按需要叠加 `Inversion`、`Generator`、`Tool Wrapper`
- 聊天区“saved content / viewer 预览 / 运行摘要”都只是消费层投影，不能反过来定义 `ServiceSkill` 或 slash 场景的执行真相

如果 `scene` 绑定的是 `site_adapter / browser_assist` 型技能，还要额外遵守以下边界：

- 用户可见入口继续以 `entries.kind=scene` 为准，不要求把底层 site skill 强行暴露成首页技能卡；但运行时解析 `scene -> linkedSkillId` 时，不能只依赖首页可见 skill 列表，必须能回退完整 `ServiceSkill` 目录做绑定解析，否则会出现“slash 菜单里能选、发送时却找不到 skill”的假入口
- 参数补齐协议继续只落在 `slotSchema`；如果 slash scene 或技能入口需要补参，运行时应先产出结构化 `scene gate request`，再由渲染层把它映射成 `a2ui`，但不要把 `a2ui` 结构写进 `SkillCatalog`、`request_metadata` 或 runtime 协议
- 如果 skill 声明了 `readinessRequirements.requiresProject=true`，或 `saveMode=project_resource` 需要真实项目目录落盘，则输入框里的 slash scene 必须复用当前选中的项目；当前没有项目时，前端要显式打开 `scene gate` 收集项目，而不是 toast 一下后结束，更不能静默创建或回退到 default 项目，以免结果写进错误目录
- 系统侧如果为了稳定性对 `site_adapter / browser_assist` 做了 preload，这一步仍必须回放成当前 assistant 消息里的真实过程步骤；不要把 preload 只塞进系统提示，也不要把它额外渲染成脱离对话的工具卡
- preload 成功或失败后，本回合都不应再回退到 `webReader / WebFetch / WebSearch / research` 这类通用网页阅读或检索工具；要么直接消费 preload 结果继续答复，要么直接把失败原因告诉用户
- 如果 preload 成功返回的是 `markdown_bundle`，且请求参数里带了 `target_language`，则后续步骤必须被视为通用的“已保存 Markdown 后处理”协议：Agent 只允许使用 `Read / Write / Edit` 读取并覆写项目里的真实 Markdown 文件，翻译时保留代码块、链接目标、相对图片路径与 Markdown 结构，不要再重新抓站点，也不要生成第二份摘要 artifact

一句话：

> 目录发现要服务端优先，但体验稳定性必须由客户端 seeded/fallback 托底。

## 四种产品分型

先固定一个边界：

下面这四种分型回答的是：

**这个能力在产品面属于什么类型。**

它们不回答：

**这个 skill / scene 内部是按 `Pipeline`、`Inversion`、`Generator`、`Tool Wrapper` 还是 `Reviewer` 组织的。**

也就是说，后续新增命令或 Plugin workflow 时，至少要做三次分类：

1. **产品分型**
   - `Agent + Task`
   - `Agent + ServiceSkill`
   - `Agent + Workflow`
   - `Agent + Prompt`
2. **Workflow 运行形态与基础设施画像**
   - `execution_profile`
   - `infra_profile`
3. **编排模式分型**
   - `Pipeline`
   - `Inversion`
   - `Generator`
   - `Tool Wrapper`
   - `Reviewer`

同一个产品分型可以搭配不同的 workflow 运行形态、基础设施画像与模式组合。

新增命令前，必须先判断它属于哪一种产品分型：

### 1. `Agent + Task`

适合：

- `@配图`
- `@修图`
- `@重绘`
- `@视频`
- `@播报`
- `@素材`
- `@转写`
- `@排版`

特点：

- 异步
- 耗时
- 可恢复
- 有结构化结果

常见模式组合：

- `Pipeline + Generator`
- `Pipeline + Tool Wrapper`

其中图片类能力当前已经有额外运行时纪律：

- `@配图` / `@修图` / `@重绘` 的 current 主链必须保留原始用户消息进入 Agent turn，并在普通聊天模型路由前进入 App Server `ImageCommandWorkflow`
- 文稿 inline 配图、封面位、图片工作台编辑/变体等显式动作也必须补成同构的 `harness.image_command_intent`，而不是绕过 Agent turn 直建任务
- 前端只负责补 `ImageCommandIntent` 结构化上下文，不负责预翻成 slash skill、偷偷发起 task，也不通过 prompt 强迫模型调用工具
- App Server workflow 以 task artifact 创建、补参状态或结构化失败作为 turn 完成依据，不能把 `Skill(image_generate)` 成功或 assistant 文本成功误判成任务已提交
- 不要为了“找技能”再先走 `ToolSearch`；如果运行时发现 `@配图` 在 `ToolSearch / WebSearch / Read / Glob / Grep` 上空转，应视为图片主链断裂
- 聊天区轻卡与 viewer 只消费后端真实运行态，不伪造“已完成”

`@素材` 在这个分型里是一个混合分流特例：

- 命令仍必须先进入 `Agent -> Skill(modal_resource_search)` 主链
- 当 `resource_type=image` 且关键词明确时，skill 应优先调用 `lime_search_web_images`，直接复用现有 `Pexels API Key` 设置返回候选
- `lime_search_web_images` 命中后，聊天区应直接展示真实 tool result 生成的素材轻卡与缩略图，点击后在右侧打开同回合 artifact document，而不是只留一段文本总结
- 当资源类型是 `bgm / sfx / video`，或图片直搜失败时，再通过 `lime_create_modal_resource_search_task` 进入 `modal_resource_search` 的 typed task binding；不得回退已删除的 CLI 任务入口。
- 无论走直搜还是 task，都必须保留真实 `tool_timeline`，不能回到前端直连图库或隐藏底层 tools

### 2. `Agent + ServiceSkill`

适合：

- `/复刻短视频`
- `/每日获取趋势赛题`
- `/账号自动增长`

特点：

- 场景化
- 有 slot schema
- 有 run / delivery / managed 语义

常见模式组合：

- 主模式优先用 `Pipeline`
- 缺参或权限门禁叠加 `Inversion`
- 结果要稳定落文稿时叠加 `Generator`
- 依赖站点、浏览器、服务端能力时叠加 `Tool Wrapper`
- 发布前复核或质量守门再按需叠加 `Reviewer`

当前客户端第一刀收口规则：

- `/scene-key` 不再直接落回本地 slash skill 预处理
- 先按统一目录找到 `scene` 与其 `linkedSkillId`
- 把 `service_scene_launch` 作为当前 turn 的 binding 上下文，而不是任何云端 run 协议
- 由 Agent 直接依据当前 `service_scene_launch` 上下文完成本地执行；若实现里仍保留 `lime_run_service_skill` 这类历史名字，也只应理解为 compat 护栏，而不是 current 运行时桥接
- 服务端目录失联或 scene 未命中时，客户端 seeded/fallback 仍要保证 slash 输入能回到普通工作区主链
- 如果 `ServiceSkill` 底层绑定的是 `site_adapter / browser_assist`，允许 Rust runtime 先做一次预执行收口浏览器上下文与保存逻辑；但这次预执行必须继续走标准 `tool_start / tool_end` 事件，并以内联过程步骤显示在当前对话中

### 3. `Agent + Workflow`

适合：

- `@技能中心`
- `@工作流`
- `@浏览器`

特点：

- 更像打开一个工作区或会话型工作流
- 不一定需要独立异步任务协议

常见模式组合：

- `Pipeline`
- `Pipeline + Inversion`

### 4. `Agent + Prompt`

适合：

- `@搜索`
- `@深搜`
- `@研报`
- `@站点搜索`
- `@读PDF`
- `@总结`
- `@翻译`
- `@分析`

特点：

- 首期轻量
- 保留真实 skills / tools timeline
- 可以先不独立恢复
- 后续可升级为更重的形态

常见模式组合：

- `Tool Wrapper`
- `Reviewer`
- `Inversion + Generator`

当前 `@搜索` 已按这条主链收口：

- 前端只补 `harness.research_skill_launch`
- Agent 首刀优先调用 `Skill(research)`
- `research` skill 再驱动 `search_query`
- 不走 task file，也不允许前端伪造“已搜索完成”

当前 `@深搜` 也已按这条主链收口：

- 前端只补 `harness.deep_search_skill_launch`
- Agent 首刀优先调用 `Skill(research)`
- `research` skill 继续驱动 `search_query`，但系统提示强约束至少多轮扩搜
- 不走 task file，也不允许前端把深搜伪装成“普通搜索加强版”

当前 `@研报` 也已按这条主链收口：

- 前端只补 `harness.report_skill_launch`
- Agent 首刀优先调用 `Skill(report_generate)`
- `report_generate` skill 再驱动 `search_query`，并把结果写成结构化研究报告
- 不走 task file，也不允许前端本地先拼报告再伪装成 skill 结果

当前 `@站点搜索` 也已按这条主链收口：

- 前端只补 `harness.site_search_skill_launch`
- Agent 首刀优先调用 `Skill(site_search)`
- `site_search` skill 再驱动 `lime_site_info / lime_site_run / lime_site_search`
- 不走 task file，也不允许前端先退回 `research / WebSearch`

当前 `@读PDF` 也应按这条主链收口：

- 前端只补 `harness.pdf_read_skill_launch`
- Agent 首刀优先调用 `Skill(pdf_read)`
- `pdf_read` skill 再最小化驱动 `list_directory / read_file`
- 不走 task file，也不允许前端本地直接解析 PDF 或伪造“已读结果”

当前 `@总结` 也已按这条主链收口：

- 前端只补 `harness.summary_skill_launch`
- Agent 首刀优先调用 `Skill(summary)`
- `summary` skill 默认直接总结 `summary_request.content` 或当前对话上下文；当用户显式给出本地路径时，才最小化使用 `list_directory / read_file`
- 不走 task file，也不允许前端本地直接总结后再伪装成 skill 结果

当前 `@翻译` 也已按这条主链收口：

- 前端只补 `harness.translation_skill_launch`
- Agent 首刀优先调用 `Skill(translation)`
- `translation` skill 默认直接翻译 `translation_request.content` 或当前对话上下文；当用户显式给出本地路径时，才最小化使用 `list_directory / read_file`
- 不走 task file，也不允许前端本地直接翻译后再伪装成 skill 结果

当前 `@分析` 也已按这条主链收口：

- 前端只补 `harness.analysis_skill_launch`
- Agent 首刀优先调用 `Skill(analysis)`
- `analysis` skill 默认直接分析 `analysis_request.content` 或当前对话上下文；当用户显式给出本地路径时，才最小化使用 `list_directory / read_file`
- 不走 task file，也不允许前端本地直接分析后再伪装成 skill 结果

## 公共设计包在哪里

命令运行时的公共实施设计包统一在：

- `internal/roadmap/gongneng/command-runtime/roadmap.md`
- `internal/roadmap/gongneng/command-runtime/architecture.md`
- `internal/roadmap/gongneng/command-runtime/flowcharts.md`
- `internal/roadmap/gongneng/command-runtime/sequences.md`
- `internal/roadmap/gongneng/command-runtime/code-structure.md`
- `internal/roadmap/gongneng/command-runtime/feature-document-standard.md`

它们负责定义：

- 总体架构
- 主流程图
- 关键时序图
- 前后端代码分层
- 单功能方案包标准

如果是“所有命令共享的规则”，应更新这里，而不是写回单个功能 PRD。

## 单功能方案包规则

从现在开始，命令运行时相关功能默认必须先有完整方案包，再进入正式实现。

单功能目录统一放在：

`internal/prd/gongneng/<feature>/`

最少包含：

- `prd.md`
- `architecture.md`
- `flowcharts.md`
- `sequences.md`
- `tasks.md`

共享代码 owner 统一由 `internal/roadmap/gongneng/command-runtime/code-structure.md`
记录；单功能只在 `architecture.md` / `tasks.md` 记录特有落点，不复制整套路径快照。

当前已落地或已定型的完整功能包包括：

- `internal/prd/gongneng/peitu/`
- `internal/prd/gongneng/xiutu/`
- `internal/prd/gongneng/sousuo/`
- `internal/prd/gongneng/shensou/`
- `internal/prd/gongneng/zhandiansousuo/`
- `internal/prd/gongneng/zongjie/`
- `internal/prd/gongneng/fanyi/`
- `internal/prd/gongneng/fenxi/`

旧平铺文档如果仍保留，只能作为 compat 索引，不再作为 current 主文档。

## 新增命令功能的标准步骤

### 1. 先判产品分型

先明确它是：

- `Agent + Task`
- `Agent + ServiceSkill`
- `Agent + Workflow`
- `Agent + Prompt`

如果这一步说不清，禁止直接开始实现。

### 2. 再补执行设计卡

至少要明确：

- `execution_profile`
- `pattern_primary`
- `pattern_stack`
- `infra_profile`
- `execution_entity`
- `delivery_contract`

固定要求：

- `execution_profile` 回答它主要是本地即时、本地 durable、浏览器依赖还是混合型
- `pattern_primary / pattern_stack` 回答它内部怎么组织逻辑
- `infra_profile` 回答它到底用了浏览器、server skill、CLI、schedule、db、markdown 还是 json
- 如果这一卡片写不出来，说明这个命令还没被真正设计清楚

### 3. 再判 binding family 和 executor kind

至少要明确：

- 主 binding family 是什么
- 是否涉及 `skill`
- 如果涉及 `skill`，背后是 CLI、API 还是 hybrid
- 底层 truth source 是什么

### 4. 再判目录来源与兜底策略

至少要明确：

- 这项能力是否需要出现在统一 `SkillCatalog.entries`
- 它是 `command`、`scene` 还是 `skill`
- 对应目录项由 `limecore client/skills` 下发，还是暂时由客户端 seeded
- 服务端未返回该目录项时，客户端如何回退
- 如果这项能力依赖新 render type，Lime 当前是否已经支持

### 5. 再补方案包

方案包至少要回答：

- 执行设计卡怎么填
- Agent 如何判断
- 如何补参
- 目录项由谁下发，客户端如何兜底
- 轻卡长什么样
- viewer 看什么
- scope / 恢复 / 重试 / 取消怎么做
- 前端改哪里
- Electron / App Server / legacy adapter 改哪里
- 哪些路径是 current，哪些只是 compat

### 6. 再进入实现

实现时优先遵守：

- 不继续扩 compat / deprecated 路径
- 新入口优先落在 current 主路径
- viewer 只吃统一 snapshot
- 结果卡与 viewer 语义保持一致

### 7. 实现后回挂

实现完成后，要回写：

- 当前事实源
- compat / current / deprecated / dead 分类
- 测试与 GUI 冒烟结果

## 与 `commands.md` 的分工

两份文档不要混用：

- `internal/aiprompts/commands.md`
  - 关注 Electron IPC、App Server JSON-RPC、legacy adapter、`safeInvoke`、mock 与契约同步

- `internal/aiprompts/command-runtime.md`
  - 关注命令运行时产品模型、公共设计包、单功能方案包、轻卡 / viewer / truth source 关系

如果本轮改动既改命令边界，又改命令运行时主链，两份都要看。

## 与质量门禁的关系

命令运行时改动的最低质量要求，除了常规校验外，还要回答：

1. 是否已有完整方案包
2. 是否明确产品分型、binding family、executor kind、truth source
3. 是否覆盖轻卡、viewer、恢复、重试、取消
4. 是否需要 `npm run test:contracts`
5. 是否需要 `npm run verify:gui-smoke`

详细质量门禁继续以 `internal/aiprompts/quality-workflow.md` 为准。

## 当前实施纪律

从 2026-04-05 起，命令运行时相关工作默认遵守以下纪律：

1. 先设计包，后实现
2. 公共规则放公共设计包
3. 单功能能力放单功能方案包
4. compat 文档不再继续长 current 细节
5. 每一刀都要能回挂路线图主线

## 一句话判断标准

如果一个改动会让某个能力“从输入被触发，到在聊天区出现轻卡，再到右侧查看区打开详情”，那它就不是普通小改动，而是命令运行时改动，必须先按本文件与公共设计包收口。
