# Lime Soul 个性化路线图

> 状态：current planning source
> 更新时间：2026-07-08
> 目标：把 OpenClaw / Hermes 的 `SOUL.md` 个性化经验转译成 Lime 的全局人格 / 声线能力，同时继续收敛到 Memory 个性化主链，不新增平行事实源。

## 1. 本路线图回答什么

本目录统一回答下面几类问题：

1. Lime 是否应该引入类似 `SOUL.md` 的个性化能力。
2. Soul 与 `memory`、`profile`、`Generation Brief`、Companion 的边界如何划分。
3. 为什么 Lime 采用“全局 AI 个性 / 声线设置”作为主入口，而不是默认读取裸 `SOUL.md` 文件。
4. OpenClaw 的 workspace `SOUL.md` 与 Hermes 的全局 `SOUL.md` 哪些可借鉴，哪些不能照搬。
5. 后续实现如何继续走 `memory_profile_prompt_service`、`runtime_turn` 和 `Generation Brief`，不新增第二套 prompt composer。
6. `SOUL.md` 如何作为高级导入 / 导出和可移植编辑格式服务用户，而不是成为运行时事实源。
7. Expert Plaza / 专家系统应如何与 Soul 联动，而不是把专家人格合并为全局人格。
8. 如何把 Ribbi 式固定口吻收敛为 Soul 下的可切换 Style Profile，而不是新增 PersonalStyle 平行系统。
9. 如何让 Style Profile 覆盖整轮对话细节，而不是只改变欢迎语或少量状态文案。

## 2. 参考事实源

Lime current 主链：

1. [../memory/README.md](../memory/README.md)
2. [../memory/make-next-generation-more-like-me.md](../memory/make-next-generation-more-like-me.md)
3. [../memory/architecture.md](../memory/architecture.md)
4. [../../aiprompts/governance.md](../../aiprompts/governance.md)
5. [../../aiprompts/commands.md](../../aiprompts/commands.md)

OpenClaw 本地调研：

1. `/Users/coso/Documents/dev/js/openclaw/docs/concepts/soul.md`
2. `/Users/coso/Documents/dev/js/openclaw/docs/reference/templates/SOUL.md`
3. `/Users/coso/Documents/dev/js/openclaw/src/agents/workspace.ts`
4. `/Users/coso/Documents/dev/js/openclaw/src/agents/system-prompt.ts`

Hermes 本地调研：

1. `/Users/coso/Documents/dev/python/hermes-agent/website/docs/guides/use-soul-with-hermes.md`
2. `/Users/coso/Documents/dev/python/hermes-agent/hermes_cli/default_soul.py`
3. `/Users/coso/Documents/dev/python/hermes-agent/cli.py`
4. `/Users/coso/Documents/dev/python/hermes-agent/tui_gateway/server.py`
5. `/Users/coso/Documents/dev/python/hermes-agent/cron/scheduler.py`

## 3. 固定结论

### 3.1 Soul 是 Memory 个性化路线的子路线

固定判断：

**Soul 不是 Memory 的平行系统，而是 `Personality Layer` 的产品化子路线。**

这意味着：

1. 不新增 `soul_*` 数据库主链。
2. 不新增独立 `Soul Runtime`。
3. 不新增第二套 prompt composer。
4. 不让裸 `SOUL.md` 文件成为运行时事实源。
5. Soul 的长期证据继续来自用户显式配置、`memory.profile` 和 `unified_memory_*`。
6. Soul 的运行时注入继续收敛到 `memory_profile_prompt_service`、`runtime_turn` 和后续 `Generation Brief Compiler`。

### 3.2 职责边界

| 层                    | 回答的问题                                       | 是否属于 Memory 事实源                             |
| --------------------- | ------------------------------------------------ | -------------------------------------------------- |
| Memory                | 用户长期偏好、灵感、历史证据、禁忌是什么         | 是                                                 |
| Soul                  | Lime 默认怎么和用户说话、以什么人格互动          | 使用 Memory 主链，不另建事实源                     |
| Expert Persona        | 某个专家在当前会话里以什么角色、技能、工作流执行 | 不属于 Global Soul；通过专家 runtime metadata 生效 |
| Creator / Brand Voice | 正式内容应该像谁在说话                           | 通过 `Generation Brief` 生效                       |
| Product Personality   | Lime 产品默认交互语气                            | 只影响交互，不默认进入 artifact                    |
| Companion Soul        | 伙伴怎么陪伴、吐槽、回应点名                     | 只影响 companion 气泡                              |

一句话：

**Memory 是证据和偏好底座；Soul 是全局交互人格投影；Expert 是局部任务角色 / 能力包；正式创作声线必须经过 `Generation Brief`。**

### 3.3 默认全局，但不是全局文件

固定判断：

**Lime 应该提供全局 AI 个性 / 声线设置，但不应把 `~/.lime/SOUL.md` 作为运行时事实源。**

原因：

1. Lime 是 GUI 桌面产品，普通用户不应被要求理解文件路径心智。
2. Lime 已有 `user_profile`、`memory.profile`、`unified_memory_*` 和 runtime prompt composition 主链。
3. 裸文件事实源会与设置页和记忆系统分叉。
4. 全局人格确实符合“Lime 越来越像我”的产品目标，但应落在 Lime current 配置与记忆事实源里。

### 3.4 `SOUL.md` 只做高级导入 / 导出

固定判断：

**`SOUL.md` 在 Lime 中是高级可移植编辑格式，不是默认运行时协议。**

允许：

1. 从 OpenClaw / Hermes 风格的 `SOUL.md` 导入到 Lime 全局人格设置。
2. 从 Lime 全局人格设置导出 `SOUL.md`，方便用户编辑、迁移或备份。
3. 在开发者 / 高级入口展示 Markdown 预览。

不允许：

1. 每轮运行时直接扫描项目根 `SOUL.md`。
2. 默认读取用户数据目录里的裸 `SOUL.md` 并覆盖设置页。
3. 让 `SOUL.md` 绕过注入顺序、用户开关、Generation Brief 或审计边界。

### 3.5 正式创作默认不受 Product Soul 污染

固定判断：

**全局 Soul 默认影响聊天交互；正式创作 artifact 只有通过 `Generation Brief` 才能吸收 Creator / Brand Voice。**

这意味着：

1. 普通聊天、欢迎语、解释风格、追问方式可以默认受 Global Soul 影响。
2. 文章、脚本、海报文案、PPT、发布内容等正式产物默认不直接注入 Product Personality。
3. 当用户明确启用创作声线时，Creator / Brand Voice 进入 `Generation Brief` 并可解释、可关闭。
4. Companion Soul 不得偷渡进正式内容。

### 3.6 Expert Plaza 要联动，但不能合并

固定判断：

**Expert Plaza / 专家系统应与 Soul 联动，但不能合并成同一份人格数据。**

原因：

1. Expert 已经拥有 `personaRef`、`memoryTemplateRef`、`skillRefs`、`workflowRefs`，本质是局部任务角色 / 能力包。
2. Global Soul 回答“Lime 默认怎么和我说话”，Expert 回答“本轮由哪个专业角色处理任务”。
3. 专家人格不应反向改写全局 Soul，否则一次专家会话会污染所有后续普通聊天。
4. 正式创作仍要通过 `Generation Brief` 决定是否吸收 Creator / Brand Voice，不能让 Expert 或 Soul 默认进入 artifact。

允许的联动：

1. 专家会话默认可继承 Global Soul 的交互节奏，但专家 `personaRef` 优先定义专业角色。
2. Expert runtime metadata 可声明 `memoryTemplateRef / skillRefs / workflowRefs`，由 runtime turn 消费。
3. 专家结果如果要沉淀为个人 / 品牌声线，必须经过用户确认并进入 Memory / Knowledge evidence。

不允许的联动：

1. 专家目录不作为 Global Soul 的事实源。
2. 专家启动不把完整 persona prompt 写进 Soul config。
3. Expert Persona 不绕过 `personality_boundary_guard` 进入正式创作。

### 3.7 Style Profile 是 Soul 的子能力

固定判断：

**交互口吻预设属于 Soul，不新增独立 PersonalStyle 系统。**

这意味着：

1. `Style Profile` 回答“同一份可靠事实应该用什么语气说出来”。
2. 首版规划四个 built-in Style Pack seed：贱兮兮执行官、温柔陪伴型助理、拽酷行动派、冷静专业型搭档。四个 seed 以独立 manifest 进入统一 `Style Pack Registry`，不是组件里的四个文案分支。
3. 每个 Style Pack 必须承载 `voicePrimitives`、`surfaceContracts`、`antiRepetitionRules`、`fewShotAnchors` 和 `riskFallback`；few-shot 是风格锚点，不是 UI 可直接复读的固定句子。
4. Style Profile 只影响聊天交互、工具叙事、缺参数追问和失败解释，不作为事实源。
5. 工具结果、搜索来源、图片结果、媒体引用和任务状态继续以 Agent Runtime / App Server read model 为事实源。
6. 正式 artifact 默认不吸收 Product Soul；需要正式内容声线时仍走 `Generation Brief`。
7. 高风险、权限、删除、生产 API、医疗、法律、财务等场景必须降级到冷静专业口吻。
8. LoRA / QLoRA 小模型不是首版依赖；只有 Prompt / 模板 / guard 方案评测不稳定时再进入对照评估。
9. 风格包是 Style Profile 的分发形态，不与 Agent Skills 合成同一种 runtime package；未来可以共享 Cloud catalog / 下载 / 签名基础设施，但必须按 package `kind` 分流到不同目录、validator 和 runtime owner。
10. i18n 只负责 L0 和极少量 L1 的产品 / 状态框架文案；不得新增 `agentChat.soulInteraction.<tone>.*` 这类按风格展开的本地句库。
11. 用户可感知的 Soul 必须覆盖工具调用前、中、后、正文段落、转折、失败恢复和结尾建议；只改变欢迎语不算完成。

事实源声明：

**后续交互口吻能力只允许向 `memory.soul` + Soul `Style Pack Registry` / `Style Resolver` + Memory/Soul prompt composition + Agent Runtime facts 收敛；不得新增 PersonalStyle 平行系统，也不得让 UI / i18n 组件按 profile id 写固定句库。**

### 3.8 当前 P6 输出面骨架事实

截至 `2026-07-08`，Style Profile 已不只停留在欢迎语：

1. 工具 lifecycle、协作执行、artifact/export、media/contentParts、图片生成块和 plugin host-managed generation 都已有 current metadata 骨架。
2. 图片生成块由 App Server `runtime_backend/image_command/presentation_soul.rs` 写入 L0/L1/L2/L3 `styleLevels`、`generationBriefBoundary`、`soul_lifecycle` 和 Style Pack metadata；前端 `ImageGenerationSoulMetadata` 读取并透传到 chat preview / workbench / DOM `data-soul-*` evidence。
3. Plugin host-managed generation 的过程说明由 `plugin_worker_generation.rs` 写入 L1/L2 presentation metadata，Host Drawer projection 保留 `data-soul-*`；插件正文仍是 L3 Generation Brief only。
4. Media/contentParts 的大型媒体 preview policy、page-window 读取、Workbench 分页、server-side cancellation、`refId/sourceUri/relativePath-only` owner normalization、artifact/cache media owner 和 direct live shell cancel/terminate 骨架已进入 current owner。`media/read` 只保留 canonical `threadId` + bounded range response；前端在每段 response 通过范围、MIME、totalBytes 与 digest 校验后累计 progress，并继续支持 pagination/object URL/metadata fallback。V2-05 已物理删除 MediaRead `stream`、chunk/completed transient notification、raw subscription、response envelope、streaming parser 与 live drain，contract guard 禁止恢复第二数据通道。
5. P2 Plugin / Skills / MCP 已有 current smoke、Electron 点击骨架与生产 Harness target 可见性证据：`smoke:mcp-current -- --allow-plugin-runtime-fixture` 通过真实 Electron Host DevBridge -> App Server JSON-RPC 验证插件 runtime metadata 能进入 `agentSession/toolInventory/read`、caller-scoped `mcpTool/listForContext` 和显式 `mcpTool/callWithCaller`；`smoke:mcp-workspace-plugin-runtime-electron-fixture` 进一步证明真实 Electron/preload/App Server/MCP current JSON-RPC 下，page-level Workspace Harness 点击面板可触发 candidate `mcpServer/start`、caller-scoped `mcpTool/listForContext` 和显式 `mcpTool/callWithCaller`；生产 Harness 工具库存区已展示 `plugin_mcp_targets` 的 plugin/caller/server/tool、runtime/prepare 状态、prepare/call proof 与 default list proof，五语言文案已补齐；默认 list proof 不自动调用工具，旧 `mcp_*` facade、`defaultMocks` 和 `mockPriorityCommands` 不在成功路径。未完成项是 Playwright 自然度验收、插件安装 / 选择全链路、带副作用 MCP import/start/call 自动触发策略、更多业务协作入口、业务域 risk taxonomy、Cloud 风格包安装验收，以及其它网络长 IO / live Provider owner 证据。

## 4. OpenClaw / Hermes 借鉴结论

### 4.1 OpenClaw 值得借鉴的点

1. `SOUL.md` 与 `AGENTS.md` 分工明确：前者管人格语气，后者管项目规则。
2. `SOUL.md` 与 `MEMORY.md` 在 prompt 中有不同说明：前者 persona / tone，后者 durable preferences。
3. workspace 模板让工程用户可以版本化自己的 Agent 风格。

Lime 不照搬的点：

1. 不默认使用 workspace 文件作为普通用户主入口。
2. 不把多个 Markdown context file 都直接暴露给普通用户。
3. 不让项目根文件自动改变正式创作产物。

### 4.2 Hermes 值得借鉴的点

1. `SOUL.md` 是 primary identity，回答“谁在说话”。
2. `/personality` 是临时 overlay，不等同长期记忆。
3. Cron 中可以加载 Soul 但跳过 Memory，说明身份人格和记忆可分离。

Lime 不照搬的点：

1. 不把 `~/.hermes/SOUL.md` 式文件路径作为默认用户心智。
2. 不让 Soul 完全替代内置身份与产品边界。
3. 不让临时 personality overlay 写入长期偏好，除非用户确认沉淀。

## 5. 目录文档分工

1. [prd.md](./prd.md)
   - 产品背景、用户、目标、范围、需求、非目标和阶段能力。
2. [architecture.md](./architecture.md)
   - Soul 与 Memory 的事实源、prompt 注入、current / compat / deprecated / dead 分类。
3. [diagrams.md](./diagrams.md)
   - 架构图、流程图、时序图和注入边界图。
4. [rollout-plan.md](./rollout-plan.md)
   - 分阶段实施计划、风险、回滚和验证入口。
5. [acceptance.md](./acceptance.md)
   - 普通用户、进阶用户、创作链路和工程边界验收。
6. [personal-style-profiles.md](./personal-style-profiles.md)
   - Soul 下的四种可切换交互口吻、Style Profile 抽象、架构图、时序图、流程图和评测标准。
7. [personal-style-output-surfaces.md](./personal-style-output-surfaces.md)
   - 用户可见输出面、工具生命周期、i18n owner、图片生成块、输入框、Claw 验收矩阵。
8. [personal-style-pack-installation.md](./personal-style-pack-installation.md)
   - 风格包安装、目录、manifest、安全校验、状态机、Cloud 下载和 Agent Skills 分流边界。

## 6. 分阶段总览

| 阶段      | 目标                          | 主产物                                                                   |
| --------- | ----------------------------- | ------------------------------------------------------------------------ |
| Phase 0   | 固定口径与文档                | 本目录路线图                                                             |
| Phase 1   | 全局人格设置                  | 设置页配置 + current config                                              |
| Phase 2   | 聊天交互注入                  | prompt composition section                                               |
| Phase 2.5 | 可切换 Style Profiles         | 四个 built-in Style Pack seed + Style Resolver + prompt surface contract |
| Phase 2.6 | Style Pack 安装规范           | Style Pack Registry + install status guard + App Server local store core + JSON-RPC API + 设置页 GUI 管理骨架 + deferred Cloud 分发边界 |
| Phase 3   | SOUL.md 导入 / 导出           | 高级可移植编辑入口                                                       |
| Phase 4   | 创作声线进入 Generation Brief | Creator / Brand Voice guard                                              |
| Phase 5   | Expert Persona 联动收口       | expert runtime metadata boundary                                         |
| Phase 6   | 品牌 / 项目声线包             | Knowledge / Memory evidence projection                                   |
| Phase 7   | 临时 personality overlay      | session-scoped style pivot                                               |

## 7. 当前落地状态

截至 2026-07-08，本路线图已进入分阶段实现：

1. Phase 0 已落地：本目录固定了 Soul、Memory、Expert Persona、Generation Brief 和 `SOUL.md` 的边界。
2. Phase 1 已落地：设置页提供 `AI 个性 / 声线` 配置，配置保存到 current app config 的 `memory.soul`，不新增数据库表或 Tauri 命令。
3. Phase 2 已落地：`memory_profile_prompt_service` 统一注入 `【全局交互人格】` section，并保持 marker 去重和关闭状态不注入。
4. Phase 3 已落地：`SOUL.md` 作为高级可移植编辑格式进入设置页，导入前预览，导入后不依赖原始文件。
5. Phase 4 已接入 current 主链：设置页提供显式 `正式内容声线` 开关，配置保存到 `memory.soul.artifact_voice`；`useSoulArtifactVoiceGenerationBrief` 将已保存声线投影为发送候选，Workspace 输入区提供“创作声线”本轮开关；前端发送层只在本轮开关开启、且没有显式 `generation_brief` 时把保存声线 fallback 归一化到 root `artifact.generation_brief`，并通过 `diagnostics.soul_artifact_voice` 解释来源、开关和 guard 结果；`voice_source` 切换时会清理互斥的 `creator_voice_id` / `brand_voice_id`；后端 Artifact request metadata 归一化会为正式产物补 `generation_brief` 声线边界，默认 `voice_source=none`，并通过 Artifact prompt section 明确正式产物不默认吸收 Global Soul / Expert Persona；显式 `generation_brief` 可独立保留并覆盖保存声线，但不会单独触发 Artifact 交付合同。
6. Phase 5 已部分落地：`expertRuntimeBinding` 标记 Expert Persona 与 Global Soul 的作用域边界，`runtime_turn` 将 `harness.expert` 识别为专家会话上下文；专家人格不写回 Global Soul。
7. Phase 2.5 当前规划已更新为四个内置 Style Pack seed + registry：四个 seed 已迁到 `src/lib/soul/style-profiles/packs/*.json` manifest，前端 `manifest.ts` / `registry.ts` 已可合并 built-in 与 installed `local_import/cloud_download` manifests，resolver / directive composer / 设置页 selector 都消费同一 registry；公共 `style-profiles` barrel 不再导出 built-in profile list / pack id map / 旧 built-in helper，测试和 i18n 覆盖也通过 registry 遍历 pack/profile；App Server `runtime/soul/style_profile.rs` 读取同一 built-in manifest，`style_pack_registry.rs` 已只读加载 `<app-data>/soul/style-packs/registry.json` 中 `status: "enabled"`、`integrity.digest` 和五语言 locale key 齐全的 installed manifest，缺 `status`、旧 `enabled: true`、顶层 `digest` 旧 schema 均 fail closed；`style_pack_install.rs` 已固定 install status 状态机，只有 `Enabled` 可进入 prompt read model；`style_pack_paths.rs` / `style_pack_store.rs` 已落 App Server 本地 store core，覆盖安全 id、staging 写入、atomic replace、registry 备份、rollback、disable 和 disabled uninstall；`soulStylePack/list|install|status/set|uninstall` 已接入 App Server JSON-RPC、generated TS protocol 和前端 API 网关；设置页已具备本地包 list / manifest + 五语言 locale 导入 / 启用 / 禁用 / disabled uninstall 管理骨架，并在当前 profile 被禁用 / 卸载 / 未加载时回退默认 built-in；配置层 `memory.soul.style_profile_id` 已从四枚举演进为 registry profile id 字符串，旧 `sassy_cute_executor` alias 不再隐式映射；bundled locale 守卫已禁止 `agentChat.soulInteraction.<tone>.*` 句库回流；后续仍需把 Cloud catalog / download、签名 / digest 实测校验、安装审计、工具 lifecycle facts / UI read model / GUI evidence 补齐。
8. Phase 2.6 已新增规划文档：`personal-style-pack-installation.md` 固定风格包安装目录、manifest、状态机、Cloud 下载 deferred 边界，以及与 Agent Skills 包共享分发但 runtime 分离的规则；当前只读 read model、状态机 guard、App Server 本地 store core、JSON-RPC / 前端 API 和设置页 GUI 管理骨架已落地，Cloud 下载、签名校验和安装审计仍 deferred。
9. `/internal/research/refactor/v1` 对齐项已推进 media / contentParts current 骨架：`agent_message.contentParts.media` 从 App Server read model 进入 GUI media reference card，`sourcePath` 已通过 Electron `asset://` current fixture 显示真实图片，inline `data:` owner fail closed。唯一 sidecar reader 是 v2 `media/read`：按 canonical `threadId` 执行 bounded range、max-bytes、full-file sha256 与 request-id cancellation；GUI 只有在每段 response 通过 offset/length/MIME/totalBytes/digest 校验后才累计 progress，并负责 object URL budget/lifecycle、分页或 metadata fallback。artifact/cache media sidecar alias、media task data/file/workspace-local/受控远程输出写入 SidecarStore、tool/task owner facts 到 synthetic `contentParts.media` 投影均沿用 current owner。V2-05 已删除 MediaRead `stream`、chunk/completed transient notification、raw subscription、response notification envelope、streaming parser 和 live-drain helper，禁止恢复第二事件通道。该项仍不代表其它网络长 IO、live Provider 或跨重启恢复证据完成。
10. P6 artifact / export 骨架已补到 App Server L3：Harness `evidence/export` UI 只保留 L0 操作层五语言 i18n；`agentSession/handoffBundle|replayCase|analysisHandoff|reviewDecisionTemplate|reviewDecision` current export 已消费 `runtime/soul/locale_copy.rs`，覆盖正式 artifact title、导出 Markdown 正文、`copy_prompt` 与 review checklist 的五语言 copy，并在正文写入 `Generation Brief` 边界，默认不吸收 Product Soul；前端 `agentRuntime/exportClient.ts` 只透传 normalized locale，不按四种 Style Pack 写固定句库。
11. 2026-07-08 补充：Media/contentParts read owner normalization、artifact/cache owner read skeleton 与 direct live shell process cancel/terminate skeleton 已完成第一刀，`MessageMediaReference.refId` 从 App Server contentParts `ref_id/refId` hydrate 到 GUI reference，`mediaReferencePreviewArtifacts.ts` 已支持非 sidecar 展示 URI + `refId`、sidecar `sourceUri`、relativePath-only `sidecarRef` 三类 owner 进入同一 `media/read`；App Server `session_media_refs.rs` 已支持 MIME / kind 明确为 image/audio/video 的 artifact/cache sidecar alias 进入同一读取链；`lime-agent` tool orchestrator 已在 turn cancel 后 terminate direct live shell process；direct URI / `previewUrl` / `sourcePath` owner 不重复读取，inline `data:` 和普通 text/markdown artifact 继续 fail closed。剩余项转为其它网络长 IO / live Provider owner 证据、GUI / Playwright 自然度验收和业务协作入口盘点。
12. 2026-07-08 补充：MCP plugin runtime current smoke、Workspace Electron 点击骨架和生产 Harness target 可见性已完成第一刀，`npm run smoke:mcp-current -- --allow-plugin-runtime-fixture` 证明 plugin runtime fixture 经 current DevBridge / App Server JSON-RPC 完成 tool inventory、caller-scoped list 和显式 call proof，且 default proof 不触发工具调用；`npm run smoke:mcp-workspace-plugin-runtime-electron-fixture` 证明 page-level Workspace Harness 点击骨架能经真实 preload `app_server_handle_json_lines` 执行 `agentSession/toolInventory/read`、candidate `mcpServer/start`、caller-scoped `mcpTool/listForContext` 和显式 `mcpTool/callWithCaller`；生产 Harness 已展示 `plugin_mcp_targets` 的 runtime/prepare 状态和 proof 边界。旧 MCP facade / mock fallback 判为 `dead` 成功路径。剩余项转为 Playwright 自然度验收、插件安装 / 选择全链路、带副作用自动触发策略和 archive/cloud package skill materialization。

2026-07-31 cutover：媒体 sidecar 读取继续以 Lime-owned v2 `media/read` 为唯一生产入口；公共请求、响应、Renderer gateway 与 package client 只使用 canonical `threadId` 和 bounded range 参数。`MediaReadParams.stream`、`media.read.chunk` / `media.read.completed`、App Server streaming producer、客户端 raw subscription、response notification envelope 与 Renderer live-drain 已物理删除，progress 只由已校验 response 累计。`media-reference` 真实 Electron fixture 与独立 GUI smoke 均通过；这一刀只关闭 V2-05 media transient bypass，不代表其它 planned notification、host capability 或 recovery 已完成。

尚未完成：

1. Phase 4：自动 voice evidence 投影、诊断详情 UI 和完整 GUI / E2E 证据。
2. Phase 2.5：真实 Claw 风格自然度仍需持续评测，重点是工具调用前后、正文段落转折、失败恢复和结尾建议都受 profile 影响，同时防固定口头禅、工具进度自然化、失败恢复不卖萌，以及拽酷风格不过度装腔。
3. Refactor/v1 media / MCP 骨架后续项：direct live shell process cancel/terminate、artifact/cache media owner 读取骨架、MCP plugin runtime current smoke、Workspace Electron 点击骨架与生产 Harness target 可见性已进入 current；后续补其它网络长 IO / live Provider owner 证据、Playwright 自然度验收、带副作用自动触发策略和业务协作入口盘点。
4. Phase 6：品牌 / 项目声线包的 Knowledge / Memory evidence projection；export L3 copy service 仍需接入更完整的 creator / brand voice evidence、诊断详情 UI 与真实 GUI/E2E。
5. Phase 7：session-scoped personality overlay。

## 8. 当前必须避免的误区

1. 把 Soul 做成 Memory 旁边的第二套长期系统。
2. 把 `SOUL.md` 文件当作 Lime 运行时事实源。
3. 把用户个人资料、长期偏好、品牌声线、产品人格和 companion 人格混成一个大 prompt。
4. 默认让全局产品人格影响正式创作内容。
5. 为了照搬 OpenClaw / Hermes，牺牲 Lime 的 GUI 产品心智。
6. 在设置页、记忆页、runtime turn 各自拼装人格 prompt。
7. 无 evidence 地把历史聊天语气自动沉淀成长期声线。
8. 把专家系统的 `personaRef` 当成全局 Soul，导致一次专家会话污染所有普通聊天。
9. 把可切换口吻或风格包做成 `personalstyle` 平行路线图、独立 Runtime 或 UI 硬编码模板。
10. 把风格稳定误解成固定口头禅复读；人格应来自行为规则、节奏和具体表达，不是每句前缀。
11. 把四种风格拆成四套 `agentChat.soulInteraction.<tone>.*` i18n 句库；这会让风格千篇一律，属于 forbidden-to-restore。
12. 只用欢迎语、首句或 profile id 进入 prompt 来证明 Soul 完成；验收必须覆盖工具前、中、后和正文细节。

## 9. 这一步如何服务主线

这套路线图的主线收益是：

**把 `memory` 路线图中已经定义的 `Personality Layer` 产品化为可配置、可解释、可导入导出的全局 Soul 能力，同时继续维持 `unified_memory_*` / `memory_runtime_*` / `Generation Brief` 的单事实源边界。**
