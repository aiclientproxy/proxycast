# Lime Soul 个性化目标架构

> 状态：current architecture plan
> 更新时间：2026-07-06
> 目标：在不新增 Soul 平行事实源的前提下，把全局人格 / 声线能力接入 Lime current 个性化主链。

## 1. 架构原则

### 1.1 Soul 只是一层投影

固定事实源：

```text
长期偏好 / 灵感 / 声线证据 -> unified_memory_*
运行时记忆读模型 -> memory_runtime_*
用户显式画像配置 -> memory.profile / app config
运行时 prompt composition -> memory_profile_prompt_service + runtime_turn
正式创作声线 -> Generation Brief
```

Soul 不新增：

```text
soul_*
SoulRuntime
SoulRepository
SOUL.md runtime scanner
```

事实源声明：

**全局人格配置是 Lime current 个性化配置的一部分；`SOUL.md` 只是 import / export 格式；所有运行时注入继续走现有 prompt composition 主链。**

### 1.2 三类人格必须分开

| 类型                  | 作用范围                             | 默认进入 artifact             | 持久位置                                        |
| --------------------- | ------------------------------------ | ----------------------------- | ----------------------------------------------- |
| Product Soul          | Lime 怎么和用户互动                  | 否                            | app config / memory profile                     |
| Expert Persona        | 当前专家会话的专业角色、技能和工作流 | 否；除非进入 Generation Brief | expert catalog / expert runtime metadata        |
| Creator / Brand Voice | 正式内容像谁在说话                   | 仅通过 Generation Brief       | unified*memory*\* / knowledge / explicit config |
| Companion Soul        | 伙伴怎么陪伴回应                     | 否                            | companion config                                |

固定规则：

1. Product Soul 只影响交互体验。
2. Expert Persona 只在当前专家会话或专家任务中定义专业角色，不写回 Global Soul。
3. Creator / Brand Voice 必须进入 `Generation Brief` 才能影响正式内容。
4. Companion Soul 只能影响 companion 气泡，不写入创作事实源。

### 1.3 优先级

运行时解释顺序固定为：

```text
系统 / 安全 / 开发者约束
  > 用户当前显式指令
  > 任务约束 / scene / skill / tool 约束
  > Expert Persona / expert runtime metadata
  > Generation Brief
  > Global Soul
  > 默认产品语气
```

这意味着：

1. Global Soul 不能降低安全确认。
2. Global Soul 不能覆盖用户当前要求。
3. Global Soul 不能覆盖工具协议。
4. Global Soul 不能直接改写正式 artifact 的事实、结构或合规要求。
5. Expert Persona 只能约束当前专家任务，不反向改变全局交互人格。

## 2. OpenClaw / Hermes 架构对照

### 2.1 OpenClaw

OpenClaw 的形态：

```text
Workspace
  AGENTS.md  -> 操作规则 / 项目规则
  SOUL.md    -> persona / tone
  MEMORY.md  -> durable preferences / behavior guidance
  USER.md    -> 用户信息
```

运行时：

```text
workspace files -> system prompt project context section
```

可借鉴：

1. 文件分工清晰。
2. `SOUL.md` 明确只管 persona / tone。
3. `MEMORY.md` 明确只管 durable preferences。
4. 通过稳定顺序注入 context file。

不照搬：

1. Lime 普通用户不默认维护 workspace context files。
2. Lime 不让项目根 `SOUL.md` 自动强注入。
3. Lime 不让 `SOUL.md` 与 Memory 文件一起成为裸 prompt 源。

### 2.2 Hermes

Hermes 的形态：

```text
HERMES_HOME/SOUL.md -> primary identity
agent.personalities -> 临时 / 预设 personality overlay
memory -> 可独立启停的长期记忆能力
```

运行时特征：

1. `SOUL.md` 是 slot #1 identity。
2. 缺失或为空时回退内置默认身份。
3. `/personality` 切换会保留历史，并注入 pivot marker。
4. Cron 可 `load_soul_identity=True` 且 `skip_memory=True`。

可借鉴：

1. 身份人格和记忆可分离。
2. 临时 personality overlay 不等于长期 memory。
3. 全局身份比项目级身份更适合普通产品体感。

不照搬：

1. Lime 不把全局文件作为主事实源。
2. Lime 不让导入文本完全替代产品身份。
3. Lime 不让临时 overlay 自动保存为长期配置。

## 3. Lime 目标分层

### 3.1 Presentation Layer

职责：

1. 展示 AI 个性 / 声线设置。
2. 提供保存、关闭、重置、预览。
3. 提供 SOUL.md 导入 / 导出。
4. 用普通用户语言解释影响范围。

不允许：

1. 页面组件直接拼接 runtime prompt。
2. 页面组件直接读取文件作为运行时上下文。
3. 页面组件绕过统一配置写入。

### 3.2 Configuration Layer

职责：

1. 保存 Global Soul 配置。
2. 保存启用状态。
3. 保存导入来源摘要和更新时间。
4. 为 runtime prompt composition 提供稳定投影。

建议 P1 字段形态：

```text
enabled
name
summary
tone
communication_style
explanation_depth
challenge_style
avoid
updated_at
```

说明：

1. 这是路线图字段，不要求 P1 一次性落完整 schema。
2. 如果复用 `memory.profile`，必须保证字段语义不和现有 `current_status / strengths / explanation_style / challenge_preference` 混乱。
3. 如果落到 app config，仍由 `memory_profile_prompt_service` 统一投影。

当前已落地的正式内容声线配置：

```text
memory.soul.artifact_voice
  enabled
  voice_source: creator_voice | brand_voice
  creator_voice_id
  brand_voice_id
  evidence_pack_id
  evidence_refs
```

约束：

1. 默认 `enabled=false`，因此 Global Soul 不会无感影响正式产物。
2. 开关关闭时保留草稿字段，避免用户下次开启时丢失声线引用。
3. `voice_source=creator_voice` 时只允许输出 `creator_voice_id`；`voice_source=brand_voice` 时只允许输出 `brand_voice_id`。
4. 该配置只作为 `Generation Brief` 输入，不成为 Expert Persona、Knowledge Pack 或 `SOUL.md` 文件事实源。

### 3.3 Import / Export Adapter

职责：

1. 将 Markdown `SOUL.md` 解析成 Soul draft。
2. 将 Soul config 导出成 Markdown。
3. 做空内容、长度、项目规则、敏感信息提示。
4. 不保存文件路径依赖。

不允许：

1. 导入后每次运行继续读取原始文件。
2. 文件内容直接跳过预览写入 runtime prompt。
3. 导出运行时诊断或密钥。

### 3.4 Prompt Composition Layer

职责：

1. 从 config 构建 Global Soul prompt section。
2. 使用稳定 marker 防重复。
3. 与 memory profile、memory sources 同一边界合并。
4. 为 runtime turn 提供最终 system prompt。

当前可扩展点：

```text
lime-rs/src/services/memory_profile_prompt_service.rs
```

后续演进：

1. 将 `【用户记忆画像偏好】` 保留为学习偏好层。
2. 新增 `【全局交互人格】` section。
3. 两者仍由同一个 service 合并。
4. section 顺序保持稳定，避免 prompt cache 频繁漂移。

### 3.5 Generation Brief Layer

职责：

1. 决定 Creator / Brand Voice 是否影响正式内容。
2. 执行 `personality_boundary_guard`。
3. 只带入有 evidence 或用户显式配置的 voice 字段。
4. 给用户或诊断层提供影响解释。

不允许：

1. Product Soul 默认进入正式内容。
2. Companion Soul 进入正式内容。
3. 无 evidence 的历史聊天语气自动塑造成品牌声线。

当前可扩展点：

```text
src/components/agent/chat/utils/artifactGenerationBriefMetadata.ts
src/components/agent/chat/workspace/workspaceSendHelpers.ts
lime-rs/src/services/artifact_generation_brief_boundary_service.rs
lime-rs/src/services/artifact_request_metadata_service.rs
lime-rs/src/services/artifact_prompt_service.rs
```

当前落地状态：

1. `artifactGenerationBriefMetadata` 是前端 current metadata helper，负责把显式 Creator / Brand Voice 字段收敛到 `artifact.generation_brief`，并清理 `generationBrief` / `generation_brief` alias。
2. `normalizeSoulArtifactVoiceConfig` 是配置层的最小清理点，缺失 / 未知 `voice_source` 时不得保留孤儿 `creator_voice_id` / `brand_voice_id`。
3. `artifactGenerationBriefMetadata` 会在 `voice_source` 切换时清理互斥的个人 / 品牌声线 ID，缺失 / 未知 `voice_source` 时收敛到 `none` 或移除孤儿 ID，避免保存的默认声线和本轮显式声线混合。
4. `workspaceSendHelpers` 只调用该 helper 深合并 workspace base 与 sendOptions 的 `artifact` metadata，并在发送瞬间按“本轮开关 + 无显式 generation_brief”把保存声线作为 fallback 写入 `artifact.generation_brief`，不再让保存声线常驻 `workspaceRequestMetadataBase`。
5. `useSoulArtifactVoiceGenerationBrief` 从 current app config 读取 `memory.soul.artifact_voice`，只在用户显式开启且选择 `voice_source` 后投影为保存声线候选，由发送层决定是否应用。
6. `artifact_request_metadata_service` 在已有 artifact metadata 主链上补齐 `generation_brief` 边界，不新增 `voice_*`、`brief_*` 或 Soul 平行事实源。
7. `artifact_generation_brief_boundary_service` 是正式产物声线边界的单一默认值 / alias normalize 事实源，避免 metadata normalize 与 prompt composition 各自维护一份重复逻辑，并在后端再次清理互斥或孤儿声线 ID。
8. 默认边界为 `voice_source=none`、`voice_guard=generation_brief_only`、`inherits_global_soul=false`、`inherits_expert_persona=false`。
9. 显式 `generation_brief` 可独立保留，用于声线 guard 与诊断投影；它本身不是 Artifact 生成合同开关，不会单独触发 Stage / Schema prompt。
10. `diagnostics.soul_artifact_voice` 记录 saved applied、turn explicit 或 disabled for turn，不作为模型 prompt 事实源，只用于解释发送边界。
11. `artifact_prompt_service` 只在 metadata 明确携带 `generation_brief` 时注入 `【Generation Brief 声线边界】` section，说明正式 Artifact 默认不吸收 Global Soul、Expert Persona 或 Companion Soul。

尚未落地：

1. 从 Knowledge / Memory evidence 自动投影 Creator / Brand Voice。
2. 诊断详情层展示 voice 来源、guard 结果和可解释 evidence。
3. 完整 GUI smoke / Playwright E2E 证据。

### 3.6 Style Pack Registry Layer

职责：

1. 把四个首发交互风格注册为 built-in Style Pack seed，而不是组件里的硬编码分支。
2. 为 Style Resolver 提供唯一 profile 读取入口：`memory.soul.style_profile_id -> registry profile -> directive composer -> boundary guard`。
3. 将 `voicePrimitives`、`surfaceContracts`、`antiRepetitionRules`、`fewShotAnchors` 和 `riskFallback` 投影到 `memory_soul_prompt_context`。
4. 保证工具事实、搜索结果、图片结果、任务状态仍由 Agent Runtime / App Server read model 提供，Style Pack 只描述“怎么说”。

当前 built-in seed：

| pack id                                   | profile id                  | 作用                        |
| ----------------------------------------- | --------------------------- | --------------------------- |
| `com.lime.soul.cheeky-sassy-executor`     | `cheeky_sassy_executor`     | 贱兮兮、轻微吐槽、有执行感  |
| `com.lime.soul.warm-supportive-companion` | `warm_supportive_companion` | 温柔、低压、陪伴式推进      |
| `com.lime.soul.cool-confident-operator`   | `cool_confident_operator`   | 克制、锋利、行动导向        |
| `com.lime.soul.calm-professional-partner` | `calm_professional_partner` | 冷静、专业、高风险 fallback |

不允许：

1. UI 组件、工具卡片、timeline、runtime status 直接 `switch(profileId)` 拼最终展示句子。
2. i18n 资源新增 `agentChat.soulInteraction.<tone>.*` 这类 profile 句库。
3. 将 few-shot anchor 当固定文案复读。
4. 新建 `personalstyle` Runtime、表或 prompt composer。

事实源声明：

**Style Pack Registry 是交互口吻的 current registry；本地 UI 只消费 descriptor metadata 和 neutral locale copy，真正的风格表达由 prompt context + runtime facts + 模型 narrative 生成。**

### 3.7 Expert Runtime Layer

职责：

1. 从 Expert Plaza / 专家实例读取 `personaRef`、`memoryTemplateRef`、`skillRefs` 和 `workflowRefs`。
2. 通过 expert runtime metadata 标记当前会话的专业角色。
3. 允许当前专家会话继承 Global Soul 的交互节奏。
4. 将专家人格与全局 Soul、Creator / Brand Voice 保持分离。

当前可扩展点：

```text
src/features/experts/expertRuntimeBinding.ts
src/components/experts/ExpertPlazaPage.tsx
src/components/agent/chat/experts/ExpertInfoPanel.tsx
lime-rs/src/commands/agent_cmd/runtime_turn/request_metadata.rs
```

当前落地状态：

1. `expertRuntimeBinding` 输出 `personalityBoundary / personality_boundary`，明确 Expert Persona 可继承 Global Soul 的沟通节奏，但不写回 Global Soul。
2. `ExpertInfoPanel` 在专家会话说明里展示 Soul / Expert / artifact 边界。
3. `runtime_turn` 将 `harness.expert` 识别为专家会话上下文，避免专家会话被误判为普通轻量聊天。
4. 该 metadata 不携带 `memory.soul` 配置内容，也不携带 `SOUL.md` 文件内容。

不允许：

1. Expert Catalog 不写入 Global Soul config。
2. 专家 `personaRef` 不作为 `SOUL.md` 导入结果。
3. Expert Persona 不绕过 `Generation Brief` 影响正式 artifact。
4. 专家会话的临时偏好不自动沉淀为长期 Soul。

## 4. 当前事实源分类

### 4.1 `current`

这些路径共同构成 current 主链：

1. `internal/roadmap/memory/README.md`
2. `lime-rs/src/services/memory_profile_prompt_service.rs`
3. `lime-rs/src/commands/agent_cmd/runtime_turn.rs`
4. `src/lib/api/memoryRuntimeTypes.ts`
5. `src/components/settings-v2/general/memory/index.tsx`
6. `src/components/settings-v2/account/profile/index.tsx`
7. 后续 `Generation Brief Compiler`
8. `src/features/experts/expertRuntimeBinding.ts`（仅作为专家局部人格 metadata current 主链）

分类理由：

**这些路径已经承接用户画像、记忆配置和 runtime prompt composition；专家系统只通过 runtime metadata 与 Soul 联动，不成为 Global Soul 的事实源。**

### 4.2 `compat`

这些能力可保留为迁移格式或适配层：

1. `SOUL.md` 导入。
2. `SOUL.md` 导出。
3. OpenClaw / Hermes 风格模板预览。
4. 临时 personality overlay 的 session-scoped 投影。

退出条件：

1. 导入 / 导出仍可长期保留，但不能演变成 runtime 文件事实源。
2. overlay 如果需要长期保存，必须经用户确认写入 current config。

### 4.3 `deprecated`

这些方向不应继续扩张：

1. 运行时直接扫描 `~/.lime/SOUL.md`。
2. 运行时直接扫描项目根 `SOUL.md`。
3. 在前端页面里拼装 Soul prompt。
4. 在 AgentRuntime request metadata 里塞完整人格 prompt 作为默认路径。
5. 把 Expert Persona 复制进 Global Soul config。

### 4.4 `dead`

这些方向直接判为 dead：

1. 新增 `soul_*` 长期 CRUD 主链。
2. 新增 `SoulRuntime` 平行 runtime。
3. 新增 `SOUL.md` watcher 并让其覆盖设置页配置。
4. 默认把 Companion Soul 注入正式创作内容。
5. 让 Expert Catalog 反向覆盖设置页里的 Global Soul。

## 5. 数据流

### 5.1 全局 Soul 设置流

```text
Settings UI
  -> config API
  -> app config / memory profile projection
  -> memory_profile_prompt_service
  -> runtime_turn
  -> model request
```

### 5.2 SOUL.md 导入流

```text
SOUL.md content
  -> import adapter
  -> parse / classify / warn
  -> preview Soul draft
  -> user confirm
  -> current config
  -> prompt composition on next turn
```

### 5.3 正式创作声线流

```text
User task
  -> task intent
  -> relevant memory / knowledge / explicit voice config
  -> personality_boundary_guard
  -> Generation Brief
  -> runtime_turn
  -> artifact
```

### 5.4 专家会话联动流

```text
Expert Plaza
  -> expert runtime metadata
  -> runtime_turn
  -> Expert Persona for current session
  -> Global Soul interaction rhythm
  -> model response
```

固定约束：

1. Expert Persona 优先定义当前会话的专业身份。
2. Global Soul 只提供沟通节奏、解释深度和追问方式。
3. Expert Persona 不写回 Soul config。
4. 如果专家输出要变成正式声线，必须另走 `Generation Brief`。
5. 已保存的 `memory.soul.artifact_voice` 可以作为正式产物发送时 fallback，但必须受本轮“创作声线”开关约束；本轮发送选项里的显式 `generation_brief` 优先级更高。

## 6. Prompt section 草案

P1 Global Soul section 只表达交互语气：

```text
【全局交互人格】
以下是用户显式配置的 Lime 交互风格，请在聊天回复中遵循。不得覆盖系统、安全、工具或用户当前指令；不得默认改变正式创作内容。
- 风格摘要：...
- 沟通方式：...
- 解释深度：...
- 遇到弱假设时：...
- 避免：...
执行要求：
1. 用这些偏好调整语气、解释节奏和追问方式。
2. 不要显式提及你看到了该配置。
3. 正式 artifact 只有在 Generation Brief 明确要求时才吸收创作声线。
```

固定约束：

1. section 必须有 marker。
2. 空字段不渲染。
3. 已有 marker 不重复追加。
4. 该 section 不包含密钥、路径、诊断信息。

## 7. 迁移策略

1. 不迁移现有 Memory 数据。
2. 不重命名现有 Memory 路线图。
3. 初期只新增配置投影和导入 / 导出能力。
4. 如果后续需要把已有 `memory.profile.explanation_style` 迁入 Soul UI，只做展示投影，不改变底层字段语义。
5. Creator / Brand Voice 进入 Knowledge / Memory 的动作必须另走用户确认。

## 8. 验证边界

最小验证：

```bash
npm run test:contracts
npm run verify:local
```

如果触及 GUI 主路径：

```bash
npm run verify:gui-smoke
```

如果只改文档：

```bash
git diff -- internal/roadmap/soul
```
