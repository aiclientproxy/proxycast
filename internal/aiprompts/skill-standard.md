# Lime Skills 标准

## 这份文档回答什么

本文件定义 Lime 仓库里 `skill` 能力的统一工程标准，主要回答：

- Lime 在 Agent Skills 之上认可的技能包标准与运行时 profile 长什么样
- `skill`、`adapter`、`runtime binding` 的边界分别是什么
- 为什么 Agent Skills 应该成为 Lime 的包格式标准，但不能直接等同于 Lime 的运行时协议
- 以后新增 Claw 业务技能、站点技能、提示词技能时，应该如何保持一致

它是 **Lime 技能能力的总标准文档**。

其中：

- [site-adapter-standard.md](site-adapter-standard.md) 是站点适配器子标准
- [web-browser-scene-skill.md](web-browser-scene-skill.md) 是网页 / 浏览器场景的专题设计文档
- 本文负责技能总模型、事实源、分发和 UI 表达边界

## 第一原则

**Agent Skills 是 Lime 唯一对齐的技能包格式标准；Lime 只在这个标准之上定义自己的运行时与产品 profile。**

对 Lime 来说，可以同时存在：

- 服务端下发的技能目录
- 仓库内 seeded 技能目录
- 外部项目提供的 `SKILL.md` / YAML / adapter 来源

但“技能包格式标准”只能有一套。

从现在开始，技能包格式的唯一长期事实源应收敛到：

> `Agent Skills Specification`

外部仓库只能提供：

- 标准包结构
- 说明层模板
- 来源层原料
- 触发语义参考

不能直接提供：

- Lime 的运行时协议
- Lime 的分发协议
- Lime 的 UI 表达标准
- Lime 的自动化与浏览器行为边界

换句话说：

- `Agent Skills` 负责回答“技能包长什么样”
- `Lime` 负责回答“技能包进入产品后怎么分发、怎么补参、怎么执行、怎么交付”

## 什么时候先读

出现以下任一情况时，先读本文件，再决定是否写代码：

- 想新增一个 Claw 业务技能
- 想把站点 adapter 封装成业务 skill
- 想新增 prompt-only 技能或说明型技能
- 想扩展服务端 `serviceSkillCatalog` / 未来 `skillCatalog`
- 想修改 `ServiceSkillItem`、`ClientServiceSkillCatalog` 或对应 UI 入口
- 想讨论 skill 与 adapter、Tool Hub、Scene、Claw 的边界
- 发现仓库里开始出现多套 skill 定义、多套入口术语或多套运行语义

如果问题已经缩小到站点适配器字段、脚本、导入和执行，先回到 [site-adapter-standard.md](site-adapter-standard.md)。
如果问题已经是“网页登录态访问、导出 Markdown、下载图片、保存网页内容”这一类浏览器场景，再看 [web-browser-scene-skill.md](web-browser-scene-skill.md)。

## 非目标

本标准明确不负责以下目标：

- 定义另一套浏览器 runtime
- 让外部 `SKILL.md` 直接成为 Lime 运行时协议
- 把 adapter 当成 skill 本体
- 为了兼容来源而长期维护第二套技能包协议
- 在第一阶段把所有既有实现一次性重命名重构完

尤其不要把“支持更多技能”误解成：

- 再造一个平级的 `service skill` 协议
- 再造一个平级的 `site skill` 协议
- 再造一个平级的 `prompt package` 协议

## 第二原则

**Skill 是 bundle，不是单一 Markdown。**

Lime 在工程上必须明确接受这一点：

- Skill 可以有主说明文件
- 但 Skill 的价值不应只存在于主说明文件
- references、examples、assets、templates、scripts、data、config、memory 都可以是 Skill 的组成部分

也就是说，Skill 不只是“告诉模型做什么”，还应当有能力承载：

- 领域知识
- 模板与示例
- 可执行脚本
- 验证步骤
- gotchas
- setup 信息
- 长期记忆

如果一个 Skill 只有一段说明文字，没有任何额外知识、资产、脚本、验证或渐进披露结构，那么它更接近提示词说明，而不是强 Skill。

## 第三原则

**Lime 不直接执行 `SKILL.md`，而是把 Agent Skills 包编译成自己的目录投影与运行时绑定。**

这也是为什么 Lime 可以兼容 Agent Skills，但不需要把产品层降级成“SkillToolset 产品”：

`Agent Skill Bundle -> 标准解析 / 校验 -> SkillBundle 摘要层 -> ServiceSkillCatalog / SkillCatalog / SceneCatalog 投影 -> Runtime Binding 执行`

固定结论：

1. `SKILL.md` 不是 Lime 的最终产品对象。
2. `ServiceSkill` / `Scene` 也不是新的包格式标准。
3. `ServiceSkill` / `Scene` 是 Lime 在 Agent Skills 之上的产品投影层。
4. Skill Forge / Capability Draft 注册只负责把已验证草案复制成 workspace-local Agent Skill 包；它不等同于运行时绑定，也不能绕过 Query Loop 与 `tool_runtime` 直接执行。

因此，generated skill 的最小安全顺序固定为：

```text
Capability Draft
  -> verification gate
  -> workspace-local Agent Skill package
  -> registered discovery / catalog projection
  -> runtime binding
  -> artifact / evidence
```

其中前三步只建立包和来源事实；`registered discovery` 只证明当前 workspace 里有带 provenance 的标准 Skill 包，仍不等于可运行。只有进入 runtime binding 和 `tool_runtime` 授权后，才回答“是否可被本轮 Agent 看到和调用”。

## 设计原则补充

除了分层边界，Lime Skill 在设计上还应遵守下面几条补充原则。

### 1. 不要重复模型默认知道的常识

Skill 最有价值的内容，是把模型从默认思路里“推出来”。

应该优先写：

- 组织独有规则
- 项目独有约束
- 常见踩坑点
- 质量判断标准

### 2. 高信号内容优先写成 gotchas

如果某个 Skill 经常失败，先补 gotchas，而不是先补更长的背景介绍。

### 3. 优先使用渐进披露

主文件负责：

- 触发
- 路由
- 总规则

详细内容优先拆到：

- `references/`
- `examples/`
- `templates/`
- `scripts/`

不要把所有内容都塞进单一说明文件。

### 4. 能用脚本和模板解决的，不要全靠自然语言重复描述

给 Skill 提供脚本、模板、示例，通常比在说明中反复描述更稳。

### 5. 需要长期使用的 Skill，应考虑 setup、memory 与 hooks

高价值 Skill 往往不是一次性调用。

设计时要考虑：

- 是否需要用户配置
- 是否需要项目级配置
- 是否要记录历史执行结果
- 是否存在只在调用期间启用的 hooks

## 标准分层

Lime 的技能标准必须分成五层：

### 1. 包标准层

作用：

- 定义一个 Skill 包在磁盘或远程仓库里长什么样
- 明确前置字段、资源目录、兼容校验与渐进加载边界

当前固定对齐：

- `Agent Skills Specification`

标准字段优先包括：

- `name`
- `description`
- `license`
- `compatibility`
- `metadata`
- `allowed-tools`

固定规则：

1. 不再新造第二种 Skill 包格式。
2. Lime 私有字段统一进入 `metadata.Lime_*`，不新增顶层私有 frontmatter。
3. 包标准层只回答“技能包长什么样”，不回答“在 Lime 里怎么执行”。

### 2. 说明层

作用：

- 回答“这是什么技能、何时使用、依赖什么、怎么触发”
- 给用户、模型、运营和后台治理看得懂

来源可以参考外部 `SKILL.md` 的优点，例如：

- `name`
- `description`
- `when to use`
- `setup`
- `examples`

但说明层不是 Lime 的运行时事实源。

### 3. 输入与产品投影层

作用：

- 把标准 Skill 包投影成 Lime 可消费的输入、展示和产品对象

当前主承载结构是：

- `src/lib/api/serviceSkills.ts` 里的 `ServiceSkillItem`
- `slotSchema`
- `readinessRequirements`
- `sceneBinding`
- `skillBundle`

新增技能时，优先补结构化输入字段，不要继续把参数要求散落在 prompt 和按钮文案里。

固定边界：

- `slotSchema` / `readinessRequirements` 是技能补参真相
- `ServiceSkillItem` 是 Lime 客户端产品投影，不是 Agent Skills 原始包
- `a2ui` 只允许作为 GUI 渲染层，把缺失信息映射成表单
- 不要把 `a2ui` 结构直接写进 skill catalog、runtime metadata 或协议字段

### 4. 运行时层

作用：

- 定义技能最终走哪种执行器

当前允许的主执行绑定为：

- `agent_turn`
- `browser_assist`
- `native_skill`

兼容说明：

- `automation_job` 只允许作为 Base Setup / Service Skill catalog 的 compat binding family；客户端必须把它正规化为 Scheduled Task 创建，不得恢复旧 Automation protocol 或 Agent UI runtime entity
- `cloud_scene` 只允许作为历史目录输入或 compat binding family 理解
- current 客户端主链必须先把它正规化为 `agent_turn / service_scene` 语义，再继续进入工作区执行

运行时层回答的是“怎么执行”，不是“对用户如何命名”。

补充边界：

- [Codex `/goal`](../research/codex-goal/README.md) 这类 persistent objective 只回答“目标是否继续推进”，不回答“skill 绑定到哪个执行器”。
- ThreadGoal 只能控制 canonical Thread，不得引用 skill executor binding 或新增 `goal_runtime`。
- `Pipeline` 是 skill 内部组织模式；跨 turn 目标控制只归 ThreadGoal，不能写成同一个字段或 runtime。

### 5. 分发层

作用：

- 定义技能如何被服务端发布、客户端缓存、bootstrap 注入和独立刷新

当前已存在的事实源包括：

- Lime 本地 seeded catalog
- `client/service-skills`
- `bootstrap.serviceSkillCatalog`

长期目标应收敛到统一的 `client/skills` 与 `bootstrap.skillCatalog`，但兼容期内允许保留现有 `serviceSkillCatalog` 投影。

## 统一对象关系

Lime 技能能力必须明确区分四个对象：

### 1. Skill Bundle

作用：

- 作为标准能力包与内容载体
- 解决“这个能力包里有哪些说明、参考资料、模板、脚本和元数据”

不是谁：

- 不是最终产品入口
- 不直接等于 `ServiceSkill / Scene`

### 2. Adapter / Tool

作用：

- 提供底层站点、工具或外部能力的执行工件
- 解决“怎么访问、参数是什么、脚本怎么跑”

### 3. Runtime Binding

作用：

- 把 skill 绑定到具体执行面
- 解决“最终交给谁执行”

### 4. Scene Skill

作用：

- 把产品型 slash scene 组织成可复用的技能流程
- 解决“为了达成一个目标，需要按什么步骤驱动 skill / adapter / runtime”

不是谁：

- 不是原始 `SKILL.md`
- 不等于上层 Plugin 或 workflow
- 更接近 `Scene / ServiceSkill` 背后的流程层与产品投影层

固定规则：

- `Skill Bundle` 是包标准层，`ServiceSkill / Scene` 才是产品对象层
- `/scene` 的长期真相是 `Scene Skill`，不是前端 if/else，不是单站点特判
- `site-adapter` 只能作为 `Scene Skill` 某一步的执行提供者，不能反客为主变成 scene runtime 本体
- 缺失信息时，优先由 `Scene Skill` 产出结构化 gate request，再由 GUI 层映射成 `a2ui`
- gate request 负责“缺什么、补什么、补完后怎么恢复”；`a2ui` 只负责“怎么收集”
- `Scene Skill` 产生的过程默认应回到当前 assistant 对话流里；如果 runtime 为了稳定性做了 preload、预检查或首刀绑定，这些步骤也要回放成对话内联过程，而不是只写隐藏 prompt 或额外工具卡

推荐模式组合：

- 主模式优先用 `Pipeline`
- 缺参或门禁用 `Inversion`
- 产物结构化输出用 `Generator`
- 封装站点 / CDP / 浏览器能力用 `Tool Wrapper`
- 只有在确实需要产物复核时再叠加 `Reviewer`

这些模式回答的是：

**skill / scene skill 内部怎么组织逻辑。**

它们不回答：

- 这个场景是本地还是云端
- 这个场景是不是 `ServiceSkill`
- 这个场景最终走哪条 runtime binding

### Skill / Scene Skill 内容设计模式

| 模式 | 回答什么 | 什么时候优先用 | 常用目录 | 常见运行时搭配 |
|------|------|------|------|------|
| `Tool Wrapper` | 如何把某个库、站点、协议、规范封装成专家上下文 | 封装 framework、site adapter、browser protocol、内部规范 | `references/`、`scripts/` | `browser_assist`、`native_skill`、`agent_turn` |
| `Generator` | 如何稳定地产出结构化结果 | 输出模板固定、格式不能漂移 | `assets/`、`references/` | `agent_turn`、Artifact |
| `Reviewer` | 如何按 checklist 打分、归类严重性、提出修复建议 | QA、合规、发布前复核、代码审查 | `references/checklist*.md` | `agent_turn`、review/evidence |
| `Inversion` | 如何先提问、补参、过 gate，再继续执行 | 需求不完整、项目/权限/账号/审批门禁 | 问题清单、gate template | `slotSchema`、`scene gate`、`a2ui` |
| `Pipeline` | 如何强制按顺序执行多步流程，并在 checkpoint 处停住 | 多步任务、外部依赖多、不能跳步 | `references/`、`assets/`、`scripts/` 全部都可能 | `agent_turn`、`browser_assist`、`automation_job`、`native_skill` |

固定规则：

1. 一个 skill 应只有一个 `主模式`。
2. 一个 skill 可以有多个 `辅助模式`。
3. `Pipeline` 常常是复杂 `Scene Skill` 的主模式。
4. `Tool Wrapper` 更适合做某一步的能力封装，不适合直接冒充产品对象。
5. `Reviewer` 只有在“复核本身是产品价值”时才应显式叠加，不要默认所有场景都加。

推荐组合：

| 组合 | 适合什么 |
|------|------|
| `Pipeline + Inversion` | 先补参，再执行严格多步流程 |
| `Pipeline + Generator` | 多步流程后输出固定结构 Artifact |
| `Pipeline + Tool Wrapper` | 流程中某一步依赖站点、浏览器、协议专家上下文 |
| `Pipeline + Reviewer` | 产物生成后还要显式质量检查 |
| `Inversion + Generator` | 先采访用户，再生成结构化结果 |
| `Tool Wrapper + Reviewer` | 以内部规范或站点规则为准做审查 |

如果后续需要把这些模式暴露给 Lime 的产品投影层，优先把原始包信息放在：

- `metadata.Lime_pattern_primary`
- `metadata.Lime_pattern_stack`
- `metadata.Lime_interaction_mode`
- `metadata.Lime_checkpoint_policy`

不要新增新的顶层 frontmatter 字段。

补充边界：

- `patternPrimary / patternStack` 可以来自 Skill Bundle 编译结果
- `infra_profile` 不属于原始 Skill Bundle 的强制字段，它属于上层 Plugin workflow 装配声明
- 不要把“这个 skill 是 `Pipeline` 型”误写成“这个场景就是云端 / 浏览器 / 本地 durable”

必须遵守：

- adapter 不是 skill
- skill 可以引用 adapter，但 adapter 不能冒充 skill
- 一个 skill 只能有一个主执行绑定
- 多 adapter 编排不属于普通 site skill，属于后续 scene / orchestration 范畴

## Lime Runtime Profile v1

这一节定义的不是新的包格式，而是：

**一个 Agent Skills 兼容包进入 Lime 之后，最少需要被编译成哪些产品投影与运行时字段。**

### 1. 技能分类

第一阶段只允许三类技能：

- `service`
- `site`
- `prompt`

含义如下：

- `service`
  - 业务交付型技能，通常产出主稿、方案、报告、草案
- `site`
  - 业务语义入口，但底层依赖 adapter / 站点工件执行
- `prompt`
  - 说明型或提示词型技能，强调触发语义和使用约束，不强制要求结构化 runtime

### 2. 统一信息清单

无论哪一类技能，新增时都必须回答以下信息。

#### 身份字段

- `id`
- `skillKey`
- `version`
- `source`

说明：

- 这些字段可以来自远端目录或本地编译结果
- 不要求直接写在 `SKILL.md` 顶层 frontmatter

#### 展示字段

- `title`
- `summary`
- `entryHint`
- `aliases`
- `category`
- `outputHint`

#### 触发字段

- `surfaceScopes`
- `triggerHints`

说明：

- 当前结构化模型尚未正式包含 `triggerHints`
- 在结构化字段补齐前，新增技能也必须在服务端模板或伴随文档中写清楚，不允许缺失

#### 输入字段

- `slotSchema`
- `default values`
- `validation`

#### 编排字段

- `patternPrimary`
- `patternStack`
- `interactionMode`
- `checkpointPolicy`

说明：

- 这组字段回答的是“skill 内部怎么组织逻辑”，不是“最终走哪种 runtime binding”。
- 如果原始包要携带这些信息，优先通过 `metadata.Lime_*` 命名空间进入编译层。
- `ServiceSkill / Scene` 的产品投影如果需要展示“这是一个 `Pipeline` 型场景，还是 `Reviewer` 型场景”，应消费这里，而不是重新猜测 prompt 内容。

#### 执行字段

- `defaultExecutorBinding`
- `executionLocation`
- `readinessRequirements`

#### 运行时引用字段

- `siteCapabilityBinding`
- `promptTemplateKey`
- 未来可扩展的 `toolHubBinding`
- 未来可扩展的 `additionalTools`

#### 产物字段

- `defaultArtifactKind`
- `output destination`

说明：

- 产品投影层可以直接提供 `outputDestination`
- 标准摘要层统一收敛到 `skillBundle.metadata.Lime_output_destination`
- 新增技能时必须明确写清结果会回到：当前主稿、资源文档、工作区消息、自动化结果还是云端运行结果

#### 说明字段

- `usageGuidelines`
- `setupRequirements`
- `examples`

说明：

- 这些字段可以先由服务端模板或说明文档承接
- 长期目标是结构化，而不是永久只写在 README / prompt 里

## Google ADK 与 ClaudeCode 借鉴边界

Google ADK 对 Lime 最值得借鉴的，不是“再做一个 ADK SkillToolset”，而是它的分层方式。

可直接借鉴的点：

1. **轻发现、重加载**
   - 列表阶段只读 frontmatter
   - 命中后再加载正文和 `references/assets/scripts`
2. **严格 validator**
   - 目录名与 `name` 一致
   - 未知顶层字段报错
   - `allowed-tools` / `allowed_tools` alias 正规化
3. **显式 skill 工具面**
   - `list_skills`
   - `load_skill`
   - `load_skill_resource`
   - `run_skill_script`
4. **激活后再开放额外工具**
   - ADK 用 `metadata.adk_additional_tools`
   - Lime 可借鉴为 `metadata.Lime_additional_tools`
5. **脚本前资源物化**
   - 在临时工作目录里重建 skill bundle，再执行脚本

ClaudeCode 对 Lime 最值得借鉴的，不是“再做一个 ClaudeCode skills 系统”，而是它把 skills 当成宿主治理对象来处理。

可直接借鉴的点：

1. **多 root 扫描与优先级**
   - managed / user / project / additional dirs 并行加载
   - 不同 root 的覆盖顺序明确且可解释
2. **严格目录约定与 legacy compat 分离**
   - `/skills/` 只接受 `skill-name/SKILL.md`
   - `/commands/` 单独作为兼容层，而不是继续污染主标准
3. **按真实文件身份去重**
   - 用 `realpath` 规避软链接和重复父目录带来的重复加载
4. **条件技能激活**
   - `paths` 命中的 skill 先进入待激活池
   - 只有用户实际触碰相关文件时才进入动态技能集
5. **动态发现嵌套技能目录**
   - 随文件操作向上发现子目录下的 `.claude/skills`
   - 深层目录优先于浅层目录
6. **内置技能、磁盘技能、MCP 技能分层**
   - bundled skills 是宿主内建能力
   - file-based skills 是外部技能包
   - MCP skills 是远端技能来源

不应直接照搬的点：

1. 不把 `SkillToolset` 当成 Lime 最终产品形态。
2. 不把已激活 skill 直接等同于 `ServiceSkill` / `Scene`。
3. 不把 Skill 包原文直接当成 `service_scene_launch`、`browser_assist`、`automation_job` 等运行时协议。
4. 不把 ClaudeCode 的 `.claude/skills`、`/commands/`、plugin-only policy、bare mode 直接当成 Lime 的产品事实源。
5. 不把 ClaudeCode 的 bundled skill 注册机制，误当成开放技能包标准。

## 执行绑定标准

### 1. `agent_turn`

适用于：

- Claw 业务技能
- 结构化 prompt + 当前工作区继续执行

要求：

- 输出是业务结果，不是“进入某个工作台”
- 不能把底层技术入口当成用户动作文案

### 2. `browser_assist`

适用于：

- 必须依赖真实浏览器登录态或页面上下文的技能

要求：

- 业务 skill 可以引用 adapter
- 不能把“浏览器工作台 / 调试面板”作为主产品语义
- 不允许隐式后台自动化

### 3. compat：`automation_job`

适用于：

- 定时或持续跟踪技能

要求：

- 必须说明首轮结果、后续调度、失败处理和结果回流方式
- current 执行入口必须是 `scheduledTask/* -> RuntimeCore -> Agent Turn`，不能调用旧 Automation gateway
- 该 binding family 完成 schema 与 consumer 迁移后应改为 Scheduled Task 领域命名

### 4. compat：`cloud_scene`

适用于：

- 只用于读取历史目录、旧 planner 或旧 skill 包里的 compat binding

要求：

- current 文档、当前前台对象和当前运行时都不应再把它描述成“云端托管执行”
- 客户端命中后必须先正规化成 `agent_turn / service_scene` 语义，再继续本地执行
- 不把普通本地即时技能错误迁回旧 `cloud_scene` 叙事

## UI 表达标准

技能 UI 必须表达业务动作，而不是暴露底层实现。

### 1. 卡片与入口

每个 skill 卡片至少要能回答：

- 这是什么
- 何时用
- 怎么执行
- 需要什么依赖
- 结果去哪

### 2. 补参与启动承载

Agent / Claw 主路径里的 skill 补参与启动，统一承载在当前对话输入区上方的 A2UI 卡片里，不再允许主产品流程回退到独立启动弹窗。

当前 `Claw` 首页、空态推荐、Skills 工作台、`@` / slash 场景入口，都应走“选技能 -> 在当前对话补参 -> 继续当前对话执行”。

只有开发调试或尚未迁完的历史兼容壳，才允许短期保留弹窗；这类弹窗不能作为新增入口，也不能继续代表主产品交互。

对话内 A2UI 卡统一应包含：

- 技能摘要
- 补参表单
- 执行方式说明
- 依赖条件说明
- 结果写入位置说明

### 3. 文案禁止项

禁止把以下内容直接当成主产品术语：

- 浏览器工作台
- 调试面板
- 脚本目录
- runtime debug
- adapter 执行器

这些只能作为实现说明，不能作为用户主动作文案。

## 分发与事实源标准

### 标准层与产品层的边界

当前必须明确区分两件事：

- `skillBundle`
  - 对外对齐 Agent Skills 思路的**标准摘要层**
  - 负责表达：`name`、`description`、`license`、`compatibility`、`metadata`、`allowedTools`
  - 以及 Lime 运行时真正需要的标准状态：`resourceSummary`、`standardCompliance`
- `ServiceSkillCatalog` / `ClientServiceSkillCatalog`
  - Lime 面向 Claw / 工作区 / 对话内 A2UI 卡的**产品投影层**
  - 负责表达：卡片文案、补参表单、执行绑定、结果去向、主题目标、自动化入口等业务语义

强约束：

- 不要把 `ServiceSkillItem` 上的产品展示字段误认为标准本体
- 也不要把外部 `SKILL.md` 原文直接当成 Lime 客户端协议
- 标准层与产品层可以共存，但标准层必须有唯一投影：`skillBundle`

### 当前事实源

客户端现状：

- 本地 seeded `SkillCatalog`
- `bootstrap.skillCatalog`
- `client/skills`
- `client/service-skills` compat 投影
- `siteAdapterCatalog`

服务端现状：

- `control-plane-svc` 负责客户端统一技能目录聚合
- Tool Hub 方向负责 tool / adapter 工件真相源

### 长期收敛方向

长期收敛规则固定如下：

1. 统一技能目录继续收敛到 `client/skills`
2. `bootstrap.skillCatalog` 与独立刷新必须消费同一份目录协议
3. `SkillCatalog.entries` 必须承载三类统一目录项：
   - `skill`
   - `command`
   - `scene`
4. `client/service-skills` 只允许作为 compat 投影继续保留，不再承接新的目录标准定义
5. adapter / tool 工件目录继续独立，不与 skill 目录混用

### 统一目录补充分层

`SkillCatalog.entries` 是分发层的 current 投影，但它不改变 skill 的产品与执行边界：

- `skill` 目录项回答“这是一个什么业务能力”
- `command` 目录项回答“用户可以通过哪个 `@` 原子入口触发它”
- `scene` 目录项回答“用户可以通过哪个 `/` 场景把多个能力编排起来”

固定约束：

- `/` 场景不是新的客户端硬编码系统，而是统一目录中的 `scene`
- `@` 命令不是独立于 skill 的第二套协议，而是统一目录中的 `command`
- `command` / `scene` 可以绑定到 skill、CLI、服务端 API 或 hybrid executor，但绑定规则仍属于运行时层
- Lime 客户端必须保留 seeded catalog 作为 offline / degrade 兜底，不能只依赖服务端在线目录

## 外部 `SKILL.md` 参考边界

外部 `SKILL.md` 仓库对 Lime 的帮助不只在“说明书结构”，还包括：

1. 标准包结构
2. 触发语义与说明组织
3. `references/assets/scripts` 的渐进加载组织方式
4. 最小 validator 与标准状态表达

但它仍不能直接成为：

- Lime 的目录协议
- Lime 的执行绑定协议
- Lime 的客户端 UI 标准
- Lime 的租户分发标准

一句话：

> 外部 `SKILL.md` 可以是 Lime 的技能包标准来源，但不能直接拿走 Lime 的运行时与产品定义权。

## 新增技能的最低检查单

新增一个技能时，至少要回答以下问题：

1. 它属于 `service / site / prompt` 哪一类
2. 它的主执行绑定是什么
3. 它是否依赖 adapter、浏览器、模型、项目或云端运行
4. 它的结果会写回哪里
5. 它的用户主动作文案是否仍然是业务语义，而不是底层实现
6. 它是否继续沿用当前主目录协议，而不是再造平级协议
7. 如果它引用 adapter，是否仍然遵守 [site-adapter-standard.md](site-adapter-standard.md)

## 当前主链

当前新增能力的主链固定如下：

- 在线目录：继续收敛到 `client/skills` 与 `bootstrap.skillCatalog`
- 本地兜底：继续收敛到 seeded `SkillCatalog`
- compat 投影：仅在迁移期保留 `client/service-skills`
- 标准摘要层：继续收敛到 `skillBundle`
- 站点工件目录：继续收敛到 `siteAdapterCatalog`
- 业务 skill 引用站点能力：通过 `siteCapabilityBinding.adapterName`

对输入面板和启动入口再补一条固定约束：

- `entries.kind=command` 驱动 `@`
- `entries.kind=scene` 驱动产品型 `/`
- `entries.kind=skill` 驱动首页技能入口与技能中心
- 服务端未返回 `entries` 时，客户端允许由 compat `items` 投影构造，但不得继续在组件层手写平行常量

不要在这个阶段再引入：

- 平级 `skill.json` 目录协议
- 平级 Markdown-only 技能协议
- 平级浏览器技能协议

## 相关文档

- [overview.md](overview.md)
- [site-adapter-standard.md](site-adapter-standard.md)
- [commands.md](commands.md)
- [quality-workflow.md](quality-workflow.md)
- [limecore-collaboration-entry.md](limecore-collaboration-entry.md)
