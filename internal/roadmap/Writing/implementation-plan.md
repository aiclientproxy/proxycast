# Writing 实施计划

> 状态：`legacy current reference`。本文的 v1 实施切片已由 `../../exec-plans/writing-v2-workflow-completion-plan.md` 替换；current 退出条件、验证和 blocker 以 v2 执行计划为准。

更新时间：2026-07-05
状态：`legacy current reference`；current 退出条件、验证和 blocker 以 `../../exec-plans/writing-v2-workflow-completion-plan.md` 为准

## 1. 当前主目标

把写文章从宿主硬编码入口收敛为 Agent Plugins v1.0.0 portable 包下的内容工厂插件闭环，并完成最小可用闭环：

```text
已安装内容工厂插件包 -> @写文章 -> 任务卡 / 过程态留在对话流 -> content_article_workflow -> ArtifactFrame(articleArtifacts renderer) -> 右侧 Article Editor（dock / tab 标准见 ../rightsurface/README.md）
```

## 2. 当前状态

- 已有内容工厂标准包、Skills、MCP 和基础 workflow，是当前实现输入；workflow/evidence 不属于 portable manifest。
- 已有宿主 typed plugin contract、输入栏候选、activation metadata、ArtifactDocument / Article Workspace 基础路径，是当前实现输入。
- 内容工厂样板包已收敛为 `plugin.json`、`mcp.json` 与 `skills/<skill>/SKILL.md`；App Server/runtime 负责生成 workflow、subagent、CLI、connector、hook 和 Article Workspace projection。
- 插件包标准事实源已落到 `internal/roadmap/plugin/v3/README.md` 与 `01-target-contract.md`。
- 宿主本地安装链已收敛为只读取标准 `plugin.json`，旧入口归类为 dead。
- LimeCore 服务端控制面本轮暂不修改；本轮只把宿主本地安装、插件投影和内容工厂写作主链收敛到标准 `plugin.json` / `mcp.json` / Skills。等 v3 package identity、manifest hash 和安装投影稳定后，再统一迁移云端 release / upload 校验。

## 3. 实施切片

### P0：Lime 插件包标准事实源

状态：完成骨架

- [x] 在 `internal/roadmap/plugin/v3/` 建立 Agent Plugins v1.0.0 标准文档。
- [x] 内容工厂外部包以 `plugin.json`、`mcp.json` 和 `skills/<skill>/SKILL.md` 作为唯一 portable 入口。
- [x] workflow、subagents、CLI、connectors、hooks 和 resources 从 package manifest 移出，统一由 App Server/runtime projection 承接。
- [x] 旧说明文档退出机器事实源。
- [x] validator 按 v3 标准校验 `plugin.json`、`mcp.json`、direct-child Skills、路径安全和组件隔离。

### P1：内容工厂样板包

状态：骨架完成，runtime readiness 投影和 prompt/task hook 生命周期执行已接入，connector 授权仍待深化

- [x] App Server/runtime projection 声明 `content_article_workflow`，并物化 `articleDraft` / `articleArtifacts` contract。
- [x] workflow 明确绑定 research、strategy、draft、review、image-plan 五个步骤。
- [x] 每个 subagent 有 `prompt.md` 和输出格式。
- [x] skills 使用稳定 id，被 workflow 引用。
- [x] CLI inspect / run / validate 可证明插件包自洽。
- [ ] subagent references / scripts / templates 继续深化。
- [x] hooks 从声明型骨架升级为受控 runtime 生命周期执行和历史 evidence 投影。
- [ ] connectors 从声明型骨架升级为宿主授权和可用性状态。
- [x] 宿主 activation metadata 和历史 runtime context 已投影 workflow 级 CLI / connector / hook readiness；registry-only connector 标记为 `declared`，不误报已授权完成。

### P2：宿主 contract 投影

状态：安装链、activation 投影、详情能力展示、`clis` / `hooks` 一等 contract、runtime readiness 投影和 hook lifecycle evidence 完成；旧 `contentFactoryWorkspacePatch` 临时字段与内容工厂 raw artifact path / kind 兼容读取均已收口到统一 helper / 插件模块白名单

- [x] plugin contract 类型支持 Agent Plugins v1.0.0 根 `plugin.json`、根 `mcp.json` 与 direct-child Skills。
- [x] workflow / Article Workspace / renderer projection 由 App Server/read model 派生，不从 manifest 私有路径读取。
- [x] 本地安装从 `plugin.json` 读取插件包入口。
- [x] 本地安装不再支持旧入口；只有负向测试可写旧文件名证明拒绝。
- [x] 本地 / 云包 inspect 从标准包读取 identity、Skills、MCP，并由 App Server 投影 activation entries、workflow、Article Workspace 和 articleDraft 恢复 contract。
- [x] App Server 包解析 / 投影从 `local_data_source/plugins` 抽到 `plugin_packages`，`local_data_source` 只保留 installed state / uninstall / 本地持久化委托。
- [x] activation entry contract 保留 `taskKind`、`workflowKey`、`outputArtifactKind`、`rightSurface` 和 `expectedObjects`，不再在前端 contract 归一化时丢失 App Server projection。
- [x] 插件中心详情页展示 typed projection 的 subagents、workflows、skills、CLI、connectors、hooks、授权和可用性。
- [x] marketplace summary 合并 installed plugin typed projection 的 workflows / connectors / hooks / clis，不再把这些运行时能力误写成 manifest 字段。
- [x] `pluginContract.ts` 从 `1000` 行以上拆回阈值内；plugin package component normalizer、Plugin 投影和通用工具已拆分，并补齐 `clis` / `hooks` 的一等 contract 类型。
- [x] workflow contract 保留 `cliRefs`、`connectorRefs` 和 `hookPolicy`，发送 metadata 同步写入 `runtime_readiness` / `plugin_runtime_readiness`，App Server `<plugin_activation_context>` 渲染 runtime readiness，历史恢复可反投影同一状态。
- [x] 右侧 Article Workspace 预览 artifact 不再生产 `contentFactoryWorkspacePatch` 旧临时字段，只保留通用 `workspacePatch` / `articleWorkspace` metadata；读取侧暂保留旧历史兼容。
- [x] 旧 `contentFactoryWorkspacePatch` / `content_factory_workspace_patch` 读取 fallback 集中到 `workspaceArticleWorkspaceMetadata`，生产投影 / 打开入口 / pending / message / history / 隐藏过滤不再各自散落旧字段读取，并补回流守卫。
- [x] 内容工厂 raw artifact path / kind 的最后 compat 命名依赖已收口：通用聊天 / 历史 / 可见性生产代码只调用 `workspaceArticleWorkspaceMetadata` 的 workspace patch helper；旧内容工厂 path / kind 字符串只允许留在 helper 的历史只读白名单和 `plugin-content-factory` 插件专属模块。

### P3：输入栏与激活

状态：基础链路完成，编排 metadata 继续补强

- [x] 输入建议读取 installed plugin contract。
- [x] 未安装时不硬编码内容工厂候选。
- [x] `@写文章` / `@写作` 从安装态插件包 activation entries 映射到 `content_article_generate`。
- [x] 发送时写入 plugin activation metadata，且不回流旧 `writing_runtime`。
- [x] metadata 包含 workflow、subagents、skill refs 和 default prompts。
- [x] metadata 包含 CLI refs、connector refs、hook policy 和 runtime registry 路径。
- [x] 显式 `@写文章` / `@内容工厂` 优先使用插件 activation entry 声明，发送 metadata 同时写入 entry 侧和 intent 侧 workflow key，避免靠自然语言二次匹配覆盖精确入口。

### P4：通用 ArtifactFrame 与右侧 Article Editor

状态：基础可编辑主链已完成；右侧 Article Editor 已按原型调整为文章画布优先的可编辑工作区，dock / tab 规则统一见 `../rightsurface/README.md`，编辑稿持久化、刷新和历史恢复已通过 Electron fixture 验证

- [x] 右侧 Artifact Workbench 和 plugin workspace 基础路径已有。
- [x] 内容工厂 articleDraft object / workspace patch 基础路径已有。
- [x] 补充 `ArtifactFrame(articleArtifacts renderer)` -> Article Editor 静态 HTML 原型，先确认产品方向。
- [x] 清理前一版把完整文章塞进非文章编辑器面板的偏航实现。
- [x] 新增通用 `ArtifactFrame` shell：从 artifact contract 选择 renderer，支持文章、图片集、表格、演示稿、网页、报告等后续产物框。
- [x] 新增通用 `artifact frame registry`：文章先作为首个 renderer，后续 artifact 只需注册自己的 renderer 与 matcher，不必改主列表分发。
- [x] 新增 `articleArtifacts` renderer：从 articleDraft / runtime artifact 在框内流式输出完整文章，不落到普通 assistant message。
- [x] 新增 Article Editor renderer：右侧显示可编辑正文、工具条、结构、引用、配图规划和动作。
- [x] Article Editor 正文画布独立为 Tiptap 组件，避免继续膨胀右侧工作台主文件。
- [x] Article Editor 后续 action 携带当前本地编辑 Markdown，避免改写 / 导出丢失画布内编辑上下文。
- [x] `ArtifactFrame` 点击后打开 Article Editor，不再以 `right_surface_article_workspace` 作为用户可见入口。
- [x] 文章 renderer 改为专用文章产物框：只承载完整文章产物本身（标题、流式/完成状态、完整正文容器、右侧编辑器入口），不再把写作过程汇总卡 / 统计 chips 塞进产物框，也不再显示成通用 Document 文件卡；右侧布局规则统一归 `../rightsurface/README.md`。
- [x] 多个 articleDraft 并存时，聊天产物框和右侧 Article Editor 默认选择多轮检索后的最终稿，不再被初始短草稿覆盖。
- [x] Article Editor 会话内编辑正文覆盖当前 selected articleDraft，后续 action / 预览默认携带当前画布 Markdown。
- [x] Article Editor 编辑内容通过 `agentSession/update.articleWorkspaceEditedDraft` 持久写回 App Server read model，后续读取优先使用最新编辑正文。
- [x] Article Editor 布局改为“主文章画布 + 辅助资料栏”：窄右栏默认单栏防变形，宽面板自动展开为画布 / 资料双栏；大纲、检索、引用、配图、标题候选、写作计划进入辅助栏。
- [x] 补刷新 / 历史恢复 E2E，证明 persisted `articleWorkspaceEditedDraft` 可跨 Electron 重启恢复到 Article Editor。
- [x] Playwright / Electron fixture 真实点击验证：聊天先出现任务卡 / 对话流过程态，再出现独立产物框，框内输出最终文章，点击展开右侧 articleDraft Article Editor。

### P5：真实 workflow 执行质量

状态：基础 runtime dogfood 已跑通；runtime 已输出 host 可执行检索请求、pending 检索证据、审稿清单和配图规划，宿主已回填真实检索 evidence，文章失败态已 fail closed

- [x] 内容工厂 runtime 拆出 `article-planning.mjs`，入口模块降到 800 行以下。
- [x] runtime fixture 输出 research -> strategy -> draft -> review -> image plan 的结构化写作对象。
- [x] 多轮检索 evidence 已进入 articleDraft metadata，并在聊天产物框与右侧 Article Editor 可见。
- [x] `articleDraft` / App Server evidence projection 输出 `searchRequests`、`searchEvidence`、`reviewChecklist` 和 `imagePlan`。
- [x] 宿主 connector / tool timeline 执行 `searchRequests` 并把真实 evidence 回填到 articleDraft metadata。
- [x] 写作失败时不产出假 articleDraft。
- [x] 审稿和配图规划进入 articleDraft metadata。
- [x] 图片任务失败 / 取消后，聊天轻卡和右侧 ImageTaskViewer 都提供可见重试入口，并复用通用 `image_workbench_retry` 任务恢复链路。

### P6：历史恢复

状态：完成；articleDraft / 编辑稿恢复、对象优先级和边缘 fallback 已收口

- [x] 历史打开时优先恢复 selected articleDraft。
- [x] 无 selected 时恢复 primary articleDraft。
- [x] 只在没有 plugin workspace / artifact 时回退聊天。
- [x] 补内容工厂专属历史恢复 E2E，覆盖 `articleWorkspaceEditedDraft` 写回、刷新和历史恢复。

### P7：云端控制面同步

状态：稍后处理

当前判断：不在本轮同步修改 `/Users/coso/Documents/dev/ai/limecloud/limecore`。

原因：

- 宿主标准包安装链、activation projection、内容工厂 `@写文章` E2E 还在收敛中；服务端此时跟进会提前固化半成品合同。
- 服务端当前边界是 catalog、release metadata、tenant enablement、license / registration、package URL / hash 下发，不执行插件 runtime，不渲染 Article Editor，不托管 UI runtime。
- `manifestHash` 已规划为标准 manifest hash，`packageHash` 已规划为包内容 hash；这两个口径必须先在 `plugin/list` / `plugin/read` / `plugin/install` 链路稳定。
- 内容工厂当前主风险在客户端主链：已安装插件可见、`@写文章` 激活、独立 ArtifactFrame、右侧 Article Editor、多轮搜索后写作。先改服务端不能直接证明这些主风险收口。

本阶段服务端分类：

- `current`：继续保持云端控制面，只负责可见性、授权、release metadata 和包引用下发。
- `deferred`：标准 package upload 校验、release summary、OpenAPI / SDK / 类型同步。
- `dead`：在服务端新增插件 runtime 执行、UI runtime 托管、Article Editor 渲染或 `/plugins/*/run` 路由。

后续进入条件：

- 宿主 `plugin/list`、`plugin/read`、`plugin/install` 和前端 install review 均稳定只认标准包入口。
- Agent Plugins v1.0.0 的 `manifestHash` 口径固定为标准 manifest hash，`packageHash` 口径固定为包内容 hash。
- 内容工厂外部包完成 v3 validator、sample runtime、local install 和 `@写文章` E2E。

服务端待办：

- [ ] 平台 package upload 校验发布包中必须存在唯一根 `plugin.json`、根 `mcp.json`（如声明 MCP）和 `skills/`。
- [ ] release metadata / manifestSummary 从旧摘要迁到 Agent Plugins v1.0.0 摘要。
- [ ] `content-factory-app` seeded catalog 指向 v1 package 版本和新 hash。
- [ ] 未激活注册码时继续不下发 package URL / hash，避免本地绕过企业定制授权。
- [ ] API client、OpenAPI、docs 和 contract tests 同步 v3 字段。
- [ ] 保持 LimeCore 不执行插件 runtime、不托管 UI runtime、不新增 `/plugins/*/run`。

## 4. 验证入口

### 内容工厂外部包

```bash
npm test
npm run runtime:sample
npm run validate:app
npm run cli:inspect
```

### Lime 定向测试

```bash
npx vitest run \
  src/features/plugin/manifest/pluginContract.unit.test.ts \
  src/components/agent/chat/workspace/workspacePluginInputSuggestions.unit.test.ts \
  src/components/agent/chat/workspace/workspacePluginActivation.unit.test.ts \
  src/features/plugin/marketplace/pluginMarketplaceViewModel.unit.test.ts
```

### Lime GUI / Playwright

```bash
npm run verify:gui-smoke
```

Playwright 真实交互用例：

1. 打开插件中心。
2. 确认未登录云端时本地内容工厂仍可见。
3. 查看内容工厂详情页，确认 subagents / tools / skills 可见。
4. 回到 Claw 输入框。
5. 输入 `@写文章 写一篇关于 AI Agent 工作流的文章`。
6. 确认输入建议来自内容工厂。
7. 发送后确认不是普通聊天正文长文，而是独立 `ArtifactFrame`。
8. 确认文章正文在框内流式输出，点击产物框后右侧 Article Editor 展开。

最新证据：

- `2026-06-29`：`npm run smoke:claw-chat-current-fixture -- --scenario content-factory-article-workspace --timeout-ms 180000` 通过。
- 证据文件：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-summary.json`。
- 关键 GUI 证据：聊天区 `article-artifact-frame` 显示“文章产物 / 完整正文 / 正文草稿 / 展开右侧编辑器”，且 `contentFactoryArticleWorkspaceArtifactFrame.hasWorkerResearchText = true`、`hasWorkerDraftText = true`；点击后 `contentFactoryArticleWorkspaceRightSurface.activeSurface = articleWorkspace`、`rootVisible = true`、`canvasVisible = true`、`hasLoadedDraftStatus = true`、`hasWorkerResearchText = true`、`hasWorkerDraftText = true`。
- 关键 read model 证据：`contentFactoryArticleWorkspaceReadModel.workerArticleObject.researchRoundCount >= 3`、`markdownIncludesResearch = true`、`markdownIncludesDraft = true`、`imageSlotCount >= 3`。
- `2026-06-29`：Article Editor 本地编辑覆盖层已补单测，后续 action 会携带当前画布 Markdown；`npm run smoke:claw-chat-current-fixture -- --scenario content-factory-article-workspace --timeout-ms 180000` 继续通过。
- `2026-06-29`：Article Editor 编辑正文写回 App Server current `agentSession/update.articleWorkspaceEditedDraft`，read model 重建 `article_workspace` 时覆盖匹配 `articleDraft.source.markdown`，保留 research / outline / citations / image slots 等 worker metadata。
- `2026-06-29`：Article Editor UI 从纵向信息卡堆叠改成原型化编辑工作区：主列显示 Tiptap 正文画布，辅助栏承载动作、结构、检索、引用、配图和关联产物；CSS 使用容器查询，窄右栏单栏防止变形，宽面板自动切换双栏。
- `2026-06-29`：内容工厂 worker 拆出 `src/runtime/article-planning.mjs`，`articleDraft.source`、`workerEvidence` 和 `imageGenerationSet.source` 同步输出 `searchRequests`、`searchEvidence`、`reviewChecklist`、`imagePlan`；外部插件包 `npm test`、`npm run runtime:sample`、`npm run validate:app`、`npm run cli:inspect` 通过，宿主 fixture 与 contract/unit 回归通过。
- `2026-06-29`：Electron fixture 验证 `E2E_EDITED_ARTICLE_DRAFT_RESTORED` 出现在聊天产物框、右侧 Tiptap canvas 和 read model，证明编辑稿可刷新 / 历史恢复。
- `2026-06-29 16:43`：重跑 `npm run smoke:claw-chat-current-fixture -- --scenario content-factory-article-workspace --timeout-ms 180000` 通过；summary session=`claw-chat-current-1782722594659-2154`，`contentFactoryArticleWorkspaceEditedDraftRestored = true`，`noConsoleErrors = true`，`appServerJsonRpcUsed = true`，`liveProviderNotUsed = true`。
- `2026-06-29 16:45`：重跑外部插件包 `npm test` / `npm run runtime:sample` / `npm run validate:app` / `npm run cli:inspect` 通过；宿主 `contentFactoryWorkerContract`、`contentFactoryWorkspacePatch`、`workspaceArticleWorkspaceModel`、`workspaceArticleWorkspaceMessageArtifacts`、`MessageArtifactCards`、`artifactFrameRegistry` 定向回归通过。
- `2026-06-29`：历史恢复 fallback 已收口：`artifactRefs` 统一从顶层 snapshot、selected / primary object ref 和 plugin workspace object 聚合；只有没有 plugin workspace / object / artifact 时才进入纯聊天。定向回归 `pluginHistoryRestore`、`workspacePluginHistoryRestoreRuntime`、`workspacePluginHistoryRestoreLanding`、`workspacePluginHistoryRestoreArtifacts` 通过。
- `2026-06-30 10:41 CST`：插件 activation contract 收口：前端 `PluginActivationEntryDeclaration` 保留 workflow / task / right surface / expected objects；`workspacePluginIntentRouting` 同时读取顶层 `activationEntries` 与 runtime intents，显式 `@` 前缀优先命中插件声明；`plugin_activation` 和 `plugin_activation_intent` 均写入 `content_article_workflow`。定向回归 `pluginContract`、`pluginActivation`、`workspacePluginIntentRouting`、`workspacePluginActivation`、`useWorkspaceSendActions` 通过，插件中心 / 应用中心相邻回归通过。
- `2026-06-30 11:00 CST`：插件中心详情页能力编排补齐：`PluginMarketplaceCapabilityProfile` 从 manifest summary / agentRuntime / toolRefs 投影 workflows、CLI / worker、connectors、lifecycle hooks、subagents 和 skills；详情页新增工作流、连接器、生命周期钩子分组和五语言文案；marketplace loader 合并 installed summary 的 workflows / connectors / hooks / clis。定向回归 `pluginMarketplaceViewModel`、`pluginMarketplace`、`marketplaceRegistryLoader`、`PluginMarketplacePage`、`pluginMarketplaceActions`、`pluginContract` 通过；Prettier 通过。
- `2026-06-30 12:15 CST`：右侧 Article Editor 固定为 compact 事实源：`WorkspaceArticleEditorRightSurface` 显式向 `WorkspaceArticleEditorSurface` 传 `compact`，窄栏下隐藏辅助面板、压缩工具栏和正文字号，避免把完整桌面编辑器强塞进右侧栏；定向回归 `WorkspaceArticleEditorRightSurface` 通过。
- `2026-06-30 12:45 CST`：右侧 Article Editor rail 从固定 `392px` 改为文档编辑自适应宽度 `clamp(440px, 30vw, 640px)`，compact 模式移除窄栏统计块和 raw ISO 更新时间，保留文章标题、打开预览、正文画布和语言化更新时间；定向回归 `WorkspaceArticleEditorRightSurface`、`WorkspaceShellScene` 通过。Playwright 复核停留在首页，当前本地 App Server 返回 `Server overloaded; retry later`，未能复现已打开 Article Editor 的真实会话态。
- `2026-06-30 14:42 CST`：App Server `content.article.generate` 在 `turn.accepted` 后立即发出 `content_factory.workspace_patch` streaming snapshot，使用与 worker 最终稿一致的 articleDraft object ref，并按 `agent_response_language / locale` 选择五语言初始产物文案；签名失败 / 禁用 / 未授权输出不产生假 streaming 产物，非文章动作如配图重生成不产生文章初始框。定向回归 `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server plugin_worker_turn -- --nocapture` 通过，前端 artifact 过滤 / 文章小框回归 34 tests passed，`npm run verify:gui-smoke` 通过。
- `2026-07-01 22:11 CST`：重跑 `npm run smoke:claw-chat-current-fixture -- --scenario content-factory-article-workspace --timeout-ms 180000 --evidence-dir ".lime/qc/gui-evidence/claw-chat-current-fixture-writing" --prefix "writing-content-factory"` 通过；summary session=`claw-chat-current-1782915088989-20694`，证据文件 `.lime/qc/gui-evidence/claw-chat-current-fixture-writing/writing-content-factory-summary.json`。本轮同步修正 fixture 判定口径：内容工厂场景以 `ArticleArtifactFrame` 完整文章产物、右侧 Article Editor 主画布、`articleWorkspaceEditedDraft` 恢复和 App Server read model 中的配图 / 分镜 / 检查清单对象为 P0 证据；compact 右栏下隐藏的辅助面板不再被误判为写文章主链失败。
- `2026-07-01 22:44 CST`：修正右侧 Article Editor compact 模式隐藏全部写作技能的问题：主画布上方新增可见 writing plan / skills 条带，直接展示 `writingPlan.owner` 与 `writingPlan.skillRef`，不恢复会挤压画布的完整辅助栏；fixture 断言从“隐藏 DOM 存在”升级为 `compactWritingPlanVisible` 与 `hasVisibleSkillRef`。使用 dev renderer 重跑 `npm run smoke:claw-chat-current-fixture -- --app-url http://127.0.0.1:1421/ --scenario content-factory-article-workspace --timeout-ms 180000 --evidence-dir ".lime/qc/gui-evidence/claw-chat-current-fixture-writing" --prefix "writing-content-factory-skills-dev"` 通过；summary session=`claw-chat-current-1782917034701-58214`，证据文件 `.lime/qc/gui-evidence/claw-chat-current-fixture-writing/writing-content-factory-skills-dev-summary.json`，可见 `visibleSkillRefs=["article-research","article-strategy","article-writing","article-editing","article-image-plan"]`。
- `2026-07-01 23:41 CST`：右侧 compact 主画布上方的写作技能条带升级为通用 plugin orchestration rail：UI 组件不绑定文章场景，Article Editor 只把 `articleWorkspace.workerEvidence` 和 `writingPlan` fallback 映射成通用 workflow / step / subagent / skill / connector / hook model；App Server read model 同步保留 workspace patch 内完整 `workerEvidence`，保证历史会话恢复后仍可还原内容工厂 workflow 编排证据。使用 dev renderer 重跑 `npm run smoke:claw-chat-current-fixture -- --app-url http://127.0.0.1:1421/ --scenario content-factory-article-workspace --timeout-ms 180000 --evidence-dir ".lime/qc/gui-evidence/claw-chat-current-fixture-writing" --prefix "writing-content-factory-orchestration-dev"` 通过；summary session=`claw-chat-current-1782920501161-76441`，证据文件 `.lime/qc/gui-evidence/claw-chat-current-fixture-writing/writing-content-factory-orchestration-dev-summary.json`，可见 `visibleSubagents=["content-researcher","content-strategist","article-writer","copy-editor","image-planner"]`、`visibleSkillRefs=["article-research","article-strategy","article-writing","article-editing","article-image-plan",...]`、`visibleConnectors=["lime-knowledge","web-research","media-generation"]`、`hasVisibleHooks=true`，read model `workerDogfoodEvidence.workflowKey="content_article_workflow"` 且 `orchestrationStepCount=5`；同次 evidence 仍通过 `contentFactoryArticleWorkspaceEditedDraftRestored=true`，证明历史 / 刷新恢复后 Article Editor 画布可还原。
- `2026-07-02 00:16 CST`：精修通用 plugin orchestration rail 边界：样式从 `WorkspaceArticleEditorSurface.css` 抽到 `WorkspacePluginOrchestrationRail.css`，Article Editor 不再拥有通用编排组件的视觉实现；新增 `workspaceArticleEditorOrchestrationModel.unit.test.ts` 覆盖完整 `workerEvidence` 优先、`writingPlan` fallback、connectors / hooks / subagents / skills 历史恢复。使用 dev renderer 重跑 `npm run smoke:claw-chat-current-fixture -- --app-url http://127.0.0.1:1421/ --scenario content-factory-article-workspace --timeout-ms 180000 --evidence-dir ".lime/qc/gui-evidence/claw-chat-current-fixture-writing" --prefix "writing-content-factory-orchestration-refine"` 通过；summary session=`claw-chat-current-1782922534697-2560`，证据文件 `.lime/qc/gui-evidence/claw-chat-current-fixture-writing/writing-content-factory-orchestration-refine-summary.json`，可见 `contentFactoryOrchestrationVisible=true`、`contentFactoryOrchestrationStepCount=5`、`visibleSubagents=["content-researcher","content-strategist","article-writer","copy-editor","image-planner"]`、`visibleSkillRefs=["article-research","article-strategy","article-writing","article-editing","article-image-plan",...]`、`visibleConnectors=["lime-knowledge","web-research","media-generation"]`、`visibleHooks=["prompt:prompt-submit","task:task-[redacted]"]`；read model `workerDogfoodEvidence.workflowKey="content_article_workflow"`、`orchestrationStepCount=5`，同次证据 `contentFactoryArticleWorkspaceEditedDraftRestored.markerVisibleInCanvas=true`。
- `2026-07-02 01:10 CST`：Article Editor 配图位动作接入现有图片主链，不新增后端协议：点击配图位会构造 `@配图 生成 ...` synthetic user message，并通过 `handleImageWorkbenchCommand -> harness.image_skill_launch -> image_generate Skill -> image task/timeline` 发送；`applyTarget.kind="canvas-insert"` 保留 `slotId`、`anchor_section_title`、`anchor_text`、`projectId`、`contentId`，避免重新随机生成文稿配图位。新增 `workspaceArticleEditorImageSlotDispatch.unit.test.ts`、`WorkspaceArticleEditorRightSurface.test.tsx` 和 `imageSkillLaunch.test.ts` 覆盖按钮意图、slot / anchor metadata 与 `usage="document-inline"`；`useWorkspaceImageWorkbenchActionRuntime.test.tsx`、`npm run test:contracts`、Rust current image skill launch guards 和 `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server image_tools` 通过。使用 dev renderer 重跑 `npm run smoke:claw-chat-current-fixture -- --app-url http://127.0.0.1:1421/ --scenario content-factory-article-workspace --timeout-ms 180000 --evidence-dir ".lime/qc/gui-evidence/claw-chat-current-fixture-writing" --prefix "writing-content-factory-image-slot"` 通过；summary session=`claw-chat-current-1782925602316-1976`，证据文件 `.lime/qc/gui-evidence/claw-chat-current-fixture-writing/writing-content-factory-image-slot-summary.json`，同次证据继续满足 `contentFactoryArticleWorkspaceEditedDraftRestored=true`、`contentFactoryArticleWorkspaceArticleWritingStructureVisible=true`、`contentFactoryArticleWorkspaceWorkerTurnExecuted=true`、`noConsoleErrors=true`、`appServerJsonRpcUsed=true`、`liveProviderNotUsed=true`。
- `2026-07-02 01:26 CST`：补齐图片任务完成后的文稿 slot 原位替换与手动应用透传：`CanvasImageInsertRequest` 增加 `taskId / slotId / sectionTitle / anchorText`，`DocumentCanvas` 对带 slot / anchor 的请求绕过编辑器光标插入，改走通用 `documentImageInsertRequest` Markdown 定位，优先替换 `lime:image-task-slot:*` 占位，缺 marker 时按 `anchor_text` / `anchor_section_title` 插入；无定位元数据的普通图库 / 手动插图仍保留编辑态光标插入语义。`useWorkspaceImageWorkbenchActionRuntime` 手动应用结果时透传 task / slot / section / anchor，重试任务保留原 `slotId`；新增 `documentImageInsertRequest.unit.test.ts` 覆盖 slot 原位替换、anchor 插入、普通文末插入、失败 / 取消状态不误写成功图片 URL。定向回归 `documentImageInsertRequest.unit.test.ts`、`useWorkspaceImageWorkbenchActionRuntime.documentApply.test.tsx`、`useWorkspaceImageTaskPreviewRuntime.test.tsx`、`useWorkspaceImageWorkbenchActionRuntime.test.tsx`、`useWorkspaceImageWorkbenchActionRuntime.taskActions.test.tsx`、`imageSkillLaunch.test.ts`、`workspaceArticleEditorImageSlotDispatch.unit.test.ts`、`autoImageInsert.test.ts` 通过；`npm run test:contracts` 与 `npm run verify:gui-smoke` 通过。使用临时 dev renderer 重跑 `npm run smoke:claw-chat-current-fixture -- --app-url http://127.0.0.1:1421/ --scenario content-factory-article-workspace --timeout-ms 180000 --evidence-dir ".lime/qc/gui-evidence/claw-chat-current-fixture-writing" --prefix "writing-content-factory-image-slot-replace"` 通过；summary session=`claw-chat-current-1782926628185-73236`，证据文件 `.lime/qc/gui-evidence/claw-chat-current-fixture-writing/writing-content-factory-image-slot-replace-summary.json`。实现备注：`useWorkspaceImageWorkbenchActionRuntime.ts` 已超过 `1000` 行，本轮只做 metadata 接线；后续若继续扩图片工作台业务逻辑，应先按 provider readiness、task actions、apply actions 拆分。
- `2026-07-02 01:40 CST`：补齐多图 slot 分配，并把 document-inline 回填从超大 runtime 中拆出：新增 `workspaceDocumentInlineImageTaskSync.ts`，按每个 `ImageWorkbenchOutput.slotId` 逐个替换文稿中的多个 `lime:image-task-slot:*` 占位；没有 per-output slot 时仍保留首图使用任务级 slot 的兼容行为，失败 / 取消状态继续只更新占位状态，不写入成功图片 URL。`useWorkspaceImageTaskPreviewRuntime.ts` 删除内联 document sync 实现，只保留调用点，文件从 `4477` 行降到 `4368` 行；新 helper 控制在 `213` 行，后续图片任务逻辑应继续按 domain helper 拆分。新增 `workspaceDocumentInlineImageTaskSync.unit.test.ts` 覆盖多 slot 替换、失败不误替换、hook 接线；定向回归 `workspaceDocumentInlineImageTaskSync.unit.test.ts`、`useWorkspaceImageTaskPreviewRuntime.test.tsx`、`documentImageInsertRequest.unit.test.ts`、`useWorkspaceImageWorkbenchActionRuntime.documentApply.test.tsx`、`imageSkillLaunch.test.ts` 通过；定向 ESLint、Prettier、`git diff --check`、`npm run test:contracts`、`npm run verify:gui-smoke` 通过。
- `2026-07-02 01:58 CST`：补齐图片任务失败 / 取消后的可见恢复路径：聊天区 `ImageWorkbenchMessagePreview` 对 `failed` / `cancelled` 且未显式 `retryable=false` 的任务显示重试按钮，继续发出通用 `lime:image-workbench-task-action`；右侧 `ImageTaskViewer` 对 `error` / `cancelled` 选中任务显示重试按钮，并透传到 `handleRetryImageWorkbenchTask`，复用既有 `image_workbench_retry` 创建任务逻辑，保留历史任务 file lookup、document-inline slot / anchor 和当前 provider/model 选择。新增 / 更新 `ImageWorkbenchMessagePreview.test.tsx`、`ImageTaskViewer.test.tsx`、`ImageTaskViewerViewModel.unit.test.ts` 覆盖取消态重试、不可重试隐藏和右侧 viewer 重试；定向回归、ESLint、`MessageList.imageTasks.test.tsx` 和 `npm run verify:gui-smoke` 通过。`npm run typecheck` 在并行脏工作树的其他文件失败，失败点未落到本轮改动文件。
- `2026-07-02 02:02 CST`：拆分图片工作台任务动作纯逻辑：新增 `imageWorkbenchTaskActions.ts`，迁出 task payload 读取、重试 mode / target / usage 推导、pending task 选择和 task action context 匹配；`useWorkspaceImageWorkbenchActionRuntime.ts` 从 `1101` 行降到 `869` 行，只保留 React 状态接线、toast 副作用和命令提交主链。新增 `imageWorkbenchTaskActions.unit.test.ts` 覆盖 slot / anchor / title generation 结果恢复、reference images 去重、重试 usage / target 推导、pending task 选择和跨项目事件过滤；定向回归 `imageWorkbenchTaskActions.unit.test.ts`、`useWorkspaceImageWorkbenchActionRuntime.taskActions.test.tsx`、`useWorkspaceImageWorkbenchActionRuntime.test.tsx`、`useWorkspaceImageWorkbenchActionRuntime.documentApply.test.tsx` 与定向 ESLint 通过。
- `2026-07-02 02:18 CST`：继续拆分图片任务预览 runtime：新增 `imageTaskPreviewRuntimeGuards.ts`，迁出 deferred auxiliary loads 启用判定、历史 seed 任务收集、pending 图片命令恢复签名、document-inline 占位恢复信号、过程态 content part 识别和 thinking content 合并；`useWorkspaceImageTaskPreviewRuntime.ts` 从 `4368` 行降到 `4107` 行，避免继续把恢复 guard 堆在巨型 hook 顶部。新增 `imageTaskPreviewRuntimeGuards.unit.test.ts` 覆盖 seed 去重、失败 / 已有预览不触发 pending 恢复、document-inline / cached task / pending 命令 catalog probe、deferred 启用判定和 thinking 合并；定向回归 `imageTaskPreviewRuntimeGuards.unit.test.ts`、`useWorkspaceImageTaskPreviewRuntime.test.tsx`、定向 ESLint、`git diff --check` 和 `npm run verify:gui-smoke` 通过。
- `2026-07-02 02:30 CST`：继续拆分图片任务预览 runtime 的 payload / presentation / storyboard 解析层：新增 `imageTaskPreviewRuntimePayload.ts`，迁出宽松 task record 读取、presentation intro / caption 读取、多模态 runtime contract snapshot 投影、storyboard slot 规范化与合并；`useWorkspaceImageTaskPreviewRuntime.ts` 从 `4107` 行降到 `3682` 行，主 hook 进一步收敛为轮询、事件监听和消息状态接线。新增 `imageTaskPreviewRuntimePayload.unit.test.ts` 覆盖基础读取、presentation 状态文案、路由阻止合约投影和 storyboard 合并；定向回归 `imageTaskPreviewRuntimePayload.unit.test.ts`、`imageTaskPreviewRuntimeGuards.unit.test.ts`、`useWorkspaceImageTaskPreviewRuntime.test.tsx`、定向 ESLint、`git diff --check` 和 `npm run verify:gui-smoke` 通过。
- `2026-07-02 02:44 CST`：继续拆分图片任务预览 runtime 的 snapshot / 状态投影层：新增 `imageTaskPreviewRuntimeSnapshot.ts`，迁出 task status 归一、pending snapshot、completed task record snapshot、artifact output fallback、输出 seed / previewImages 构建、runtime contract 接线后的 message / workbench task / outputs 投影；`useWorkspaceImageTaskPreviewRuntime.ts` 从 `3682` 行降到 `2778` 行，主 hook 只保留加载、轮询、事件监听、缓存合并和 React state 接线。新增 `imageTaskPreviewRuntimeSnapshot.unit.test.ts` 覆盖状态归一、预览图去重限量、pending provider / model / storyboard 投影、completed record 多图输出、runtime contract 和 artifact record / pending fallback；定向回归 `imageTaskPreviewRuntimeSnapshot.unit.test.ts`、`imageTaskPreviewRuntimePayload.unit.test.ts`、`imageTaskPreviewRuntimeGuards.unit.test.ts`、`useWorkspaceImageTaskPreviewRuntime.test.tsx`、定向 ESLint、scoped `git diff --check`、`npm run test:contracts` 和 `npm run verify:gui-smoke` 通过。
- `2026-07-02 03:03 CST`：继续拆分图片任务预览 runtime 的消息合并层：新增 `imageTaskPreviewRuntimeMessages.ts`，迁出 preview identity、同 turn / 同 task 合并、相邻 preview 去重、草稿 preview 清理、skill failure 归一和 `upsertPreviewMessage`；`useWorkspaceImageTaskPreviewRuntime.ts` 从 `2778` 行降到 `2145` 行，主 hook 进一步只保留 workbench state 同步、任务恢复、轮询和事件监听。新增 `imageTaskPreviewRuntimeMessages.unit.test.ts` 覆盖 task/path/running fallback 判同源、runtime turn 占位替换、重复 preview 去重、过程 content parts 保留、thinking / tool calls 合并、草稿清理和失败态可重试归一；定向回归 `imageTaskPreviewRuntimeMessages.unit.test.ts`、`imageTaskPreviewRuntimeSnapshot.unit.test.ts`、`imageTaskPreviewRuntimePayload.unit.test.ts`、`imageTaskPreviewRuntimeGuards.unit.test.ts`、`useWorkspaceImageTaskPreviewRuntime.test.tsx`、定向 ESLint、scoped `git diff --check`、`npm run test:contracts` 和 `npm run verify:gui-smoke` 通过。
- `2026-07-02 03:16 CST`：继续拆分图片任务预览 runtime 的 workbench state 同步层：新增 `imageTaskPreviewRuntimeState.ts`，迁出 `mergeImageTaskSnapshot`、workbench state -> 消息 patch、cached preview 历史恢复、用户消息补回、匹配用户轮次挂载 preview 和 selected output 延续逻辑；`useWorkspaceImageTaskPreviewRuntime.ts` 从 `2145` 行降到 `1432` 行，主 hook 进一步收敛为任务加载、轮询、事件监听和 React effect 接线。新增 `imageTaskPreviewRuntimeState.unit.test.ts` 覆盖更旧进度不覆盖已完成任务、selected output 通过 URL 延续、已有 preview 从 workbench outputs 补全、空历史恢复 user + assistant，以及 cached preview 合入匹配用户消息后的 assistant 而不是乱追加；定向回归 `imageTaskPreviewRuntimeState.unit.test.ts`、`imageTaskPreviewRuntimeMessages.unit.test.ts`、`imageTaskPreviewRuntimeSnapshot.unit.test.ts`、`imageTaskPreviewRuntimePayload.unit.test.ts`、`imageTaskPreviewRuntimeGuards.unit.test.ts`、`useWorkspaceImageTaskPreviewRuntime.test.tsx`、定向 ESLint、scoped `git diff --check`、`npm run test:contracts` 和 `npm run verify:gui-smoke` 通过。
- `2026-07-02 03:29 CST`：继续拆分图片任务预览 runtime 的恢复 / 事件映射层：新增 `imageTaskPreviewRuntimeRecovery.ts`，迁出任务恢复筛选、workspace candidate path 扫描、runtime event scope 匹配、cache 是否已满足、loaded snapshot 优先级和 task family 归一；新增 `imageTaskPreviewRuntimeEvents.ts`，迁出 creation task event -> pending preview snapshot / document-inline task record 映射。`useWorkspaceImageTaskPreviewRuntime.ts` 从 `1432` 行降到 `983` 行，回到仓库 `1000` 行阈值内，主 hook 只保留 React effect、事件监听、读取/轮询副作用和状态更新接线。新增 `imageTaskPreviewRuntimeRecovery.unit.test.ts` 与 `imageTaskPreviewRuntimeEvents.unit.test.ts` 覆盖恢复窗口、跨会话过滤、cache 命中、候选路径扫描、非 image 事件过滤和 slot/document-inline metadata 透传；定向回归 `imageTaskPreviewRuntimeEvents.unit.test.ts`、`imageTaskPreviewRuntimeRecovery.unit.test.ts`、`imageTaskPreviewRuntimeState.unit.test.ts`、`imageTaskPreviewRuntimeMessages.unit.test.ts`、`imageTaskPreviewRuntimeSnapshot.unit.test.ts`、`imageTaskPreviewRuntimePayload.unit.test.ts`、`imageTaskPreviewRuntimeGuards.unit.test.ts`、`useWorkspaceImageTaskPreviewRuntime.test.tsx` 共 `79` tests 通过，定向 ESLint、scoped `git diff --check`、`npm run test:contracts` 和 `npm run verify:gui-smoke` 通过。
- `2026-07-02 03:52 CST`：收口 plugin contract 体量与编排能力投影：新增 `pluginContractPlugin.ts`、`pluginContractComponents.ts`、`pluginContractUtils.ts`、`pluginContractErrors.ts`，把 Plugin -> Plugin manifest 投影、component/contribution/interface/activation/CLI/hook normalizer 和通用读取工具从 `pluginContract.ts` 拆出；`pluginContract.ts` 从 `1000+` 行降到 `501` 行。`PluginManifest` / `PluginContract` 补齐 `cli` / `clis` / `hooks` 与 `PluginCliDeclaration`、`PluginHookDeclaration`，内容工厂 fixture 的 `skills`、`subagents`、`workflows`、`agentRuntime.cli`、`agentRuntime.hooks.handlers` 均进入 contract；marketplace item 和 installed summary 也保留 `clis` / `hooks`，避免详情页能展示但历史恢复 contract 丢编排能力。定向回归 `pluginContract.unit.test.ts`、`pluginMarketplace.unit.test.ts`、`pluginMarketplaceViewModel.unit.test.ts`、`marketplaceRegistryLoader.unit.test.ts` 共 `35` tests 通过；Workspace plugin activation / input suggestions / runtime context / history restore / renderer projection 共 `28` tests 通过；定向 ESLint、scoped `git diff --check`、`npm run test:contracts` 和 `npm run verify:gui-smoke` 通过。
- `2026-07-02 04:06 CST`：开始收口内容工厂旧临时字段生产面：`workspaceArticleWorkspacePreviewArtifact` 移除对 `CONTENT_FACTORY_PLUGIN_ID` 的生产判断，Article Workspace 预览 artifact metadata 不再写 `contentFactoryWorkspacePatch`，统一写通用 `workspacePatch` / `articleWorkspace`；`workspaceArticleWorkspacePreviewArtifact.unit.test.ts` 补负向断言防止旧字段重新写出。定向回归 `workspaceArticleWorkspacePreviewArtifact.unit.test.ts`、`articleArtifactProjection.unit.test.ts`、`MessageArtifactCards.test.tsx` 共 `13` tests 通过，定向 ESLint、scoped `git diff --check`、`npm run test:contracts` 和 `npm run verify:gui-smoke` 通过。读取旧历史 artifact / SDK fixture 中的 `contentFactoryWorkspacePatch` 暂保留为 compat，后续继续按通用 `workspacePatch` 收口。
- `2026-07-02 04:14 CST`：继续收口旧临时字段读取优先级：`articleArtifactProjection` 与 `AgentChatWorkspace` 打开右侧 Article Editor 的入口改为优先读取 current `workspacePatch` / `workspace_patch`，`contentFactoryWorkspacePatch` / `content_factory_workspace_patch` 只作为旧历史 compat fallback；`articleArtifactProjection.unit.test.ts` 补 current-only、legacy fallback、新旧字段冲突时 current 胜出的断言。定向回归 `articleArtifactProjection.unit.test.ts` 与 `MessageArtifactCards.test.tsx` 共 `10` tests 通过；`MessageArtifactCards` 仍有既有 React act warning。剩余旧字段集中在 `workspaceArticleWorkspaceModel`、`workspaceArticleWorkspaceMessageArtifacts` 与 `agentChatHistoryArtifacts` 的历史兼容读边界，下一刀继续按 current `workspacePatch` 优先收口。
- `2026-07-02 04:25 CST`：继续收口 workspace model / message / history 读取边界：`buildWorkspaceArticleWorkspaceFromPendingRequests`、`buildWorkspaceArticleWorkspaceFromMessageArtifacts` 与 `agentChatHistoryArtifacts` 均按 current `articleWorkspace`、`workspacePatch` / `workspace_patch` 优先，`contentFactoryWorkspacePatch` / `content_factory_workspace_patch` 只保留为 compat fallback；历史 artifact summary 过滤新增 current `workspace_patch` / `*.workspace_patch` kind，避免历史恢复把 Article Workspace 中间补丁重新渲染成普通文件卡。补充新旧字段冲突时 current 胜出的 pending/message 回归、旧 pending metadata fallback 回归，以及 current `workspacePatch` history summary 不恢复成中间产物消息的回归。定向回归 `workspaceArticleWorkspaceModel.unit.test.ts`、`workspaceArticleWorkspaceMessageArtifacts.unit.test.ts`、`agentChatHistory.timeline.test.ts` 共 `38` tests 通过，定向 ESLint 和 scoped `git diff --check` 通过。
- `2026-07-02 05:10 CST`：封口旧临时字段兼容面：新增 `workspaceArticleWorkspaceMetadata.ts` 作为唯一 `contentFactoryWorkspacePatch` / `content_factory_workspace_patch` compat 读取 helper，`articleArtifactProjection`、`AgentChatWorkspace`、`workspaceArticleWorkspaceModel`、`workspaceArticleWorkspaceMessageArtifacts`、`agentChatHistoryArtifacts`、`internalArtifactVisibility` 和内容工厂 workspace patch utility 均改为调用该 helper；新增 `workspaceArticleWorkspaceMetadata.unit.test.ts` 覆盖 current `workspacePatch` 优先、旧字段 fallback、artifact-like 候选收集、workspace patch kind 识别，并扫描生产代码保证旧字段读写只允许出现在 helper。定向回归 `workspaceArticleWorkspaceMetadata.unit.test.ts`、`articleArtifactProjection.unit.test.ts`、`workspaceArticleWorkspaceModel.unit.test.ts`、`workspaceArticleWorkspaceMessageArtifacts.unit.test.ts`、`agentChatHistory.timeline.test.ts`、`internalArtifactVisibility.test.ts`、`contentFactoryWorkspacePatch.unit.test.ts` 共 `60` tests 通过；定向 ESLint、scoped `git diff --check`、`npm run test:contracts` 和 `npm run verify:gui-smoke` 通过。
- `2026-07-02 05:28 CST`：继续封口 raw workspace patch artifact 兼容命名：`workspaceArticleWorkspaceMetadata` 新增 `isWorkspaceArticlePatchArtifactPath`，把 `.lime/artifacts/*/workspace-patch.json` current path 与旧 `.lime/artifacts/content-factory-workspace-patch.json` 历史 path 统一成 helper 白名单；`internalArtifactVisibility` 不再有 `isContentFactoryWorkspacePatchPath`，`agentChatHistoryArtifacts` 也不再直接特判 `content_factory.workspace_patch`，统一走 `isWorkspaceArticlePatchArtifactKind`。`workspaceArticleWorkspaceMetadata.unit.test.ts` 新增生产扫描守卫，保证旧 raw artifact kind/path 只允许出现在 helper 或 `plugin-content-factory` 插件专属模块。定向回归 `workspaceArticleWorkspaceMetadata.unit.test.ts`、`articleArtifactProjection.unit.test.ts`、`workspaceArticleWorkspaceModel.unit.test.ts`、`workspaceArticleWorkspaceMessageArtifacts.unit.test.ts`、`agentChatHistory.timeline.test.ts`、`internalArtifactVisibility.test.ts`、`contentFactoryWorkspacePatch.unit.test.ts` 共 `62` tests 通过；定向 ESLint、scoped `git diff --check`、`npm run test:contracts` 和 `npm run verify:gui-smoke` 通过。`npm run verify:local` 仍失败在并行 / 既有 i18n unused key：`plugin.marketplace.*`、`agentChat.messageList.articleArtifact.*`、`workspace.articleWorkspace.*`，本轮未新增用户可见文案。
- `2026-07-02 05:55 CST`：补齐插件 runtime readiness 投影：`PluginWorkflowDeclaration`、本地 Plugin 投影和 marketplace summary 均保留 `cliRefs`、`connectorRefs`、`hookPolicy`；新增 `workspacePluginRuntimeReadiness.ts`，从 installed plugin contract + `InstalledPluginState.readiness` 推导 CLI / connector / hook item 状态，并区分 `ready`、`declared`、`needs_setup`、`degraded`、`blocked`。`mergePluginActivationSendOptions` 发送 `plugin_activation.runtime_readiness` 和 harness 级 `plugin_runtime_readiness`；`workspacePluginRuntimeContext` 会在 active / history metadata 中重建同一 readiness；App Server `<plugin_activation_context>` 渲染 runtime readiness，让 Query Loop 看到 connectors / hooks / CLI 的宿主状态。注意：内容工厂当前 connector registry 只声明了 refs，未展开逐项授权，因此 `web-research` 等 connector 标记为 `declared`，不误报真实授权完成。定向回归 `workspacePluginRuntimeReadiness.unit.test.ts`、`workspacePluginActivation.unit.test.ts`、`workspacePluginRuntimeContext.unit.test.ts`、`workspacePluginInputSuggestions.unit.test.ts`、`pluginContract.unit.test.ts`、`pluginMarketplace.unit.test.ts`、`CharacterMention.catalog.test.tsx` 共 `63` tests 通过；Rust 定向 `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server plugin_activation_context -- --nocapture` 通过；定向 ESLint、scoped `git diff --check`、`npm run test:contracts`、`npm run smoke:agent-runtime-current-fixture` 和 `npm run verify:gui-smoke` 通过。`npm run verify:local` 仍失败在并行 / 既有 i18n unused key：`plugin.marketplace.*`、`agentChat.messageList.articleArtifact.*`、`workspace.articleWorkspace.*`，本轮未新增用户可见文案。
- `2026-07-02 06:25 CST`：补齐内容工厂 hook lifecycle 真实执行第一刀：App Server worker runtime 新增受控 `PluginHookRunRequest`，复用 package-relative entrypoint、stdin JSON、Node 子进程、超时和敏感环境剥离；`@写文章` worker turn 会在 `turn.accepted` 后执行 `prompt.submit` hook，在 worker artifact 输出后、`turn.completed` 前执行 `task.complete` hook，并写入 `plugin_worker.hook` runtime event。内容工厂 fixture 补齐 `hooks/prompt-submit.mjs` 与 `hooks/task-complete.mjs`，历史 read model 的 `article_workspace.workerEvidence` 可还原 `hookKey / hookEvent / hookScope / hookEntrypoint / resultSummary`。缺 handler 声明或文件时投影为 `skipped / reasonCode`，不再误报 hook 已完成。定向回归 `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server plugin_worker_turn -- --nocapture` 通过 `19` tests；前端 `workspaceArticleWorkspaceWorkerEvidence`、`workspaceArticleEditorOrchestrationModel`、`workspacePluginRuntimeReadiness`、`workspacePluginActivation` 共 `20` tests 通过；定向 ESLint 通过。
- `2026-07-05 02:03 CST`：真实 Electron CDP Gate B 复测 `@写文章` 通过。启动命令使用 `LIME_ELECTRON_APP_SERVER_WATCH=0 LIME_ELECTRON_REMOTE_DEBUGGING_PORT=9223 npm run electron:dev`，保持 App Server `--backend runtime` 和真实 Electron IPC，只关闭源码 watcher 以避免验证中途重启；`npm run bridge:health -- --timeout-ms 120000` 通过。用 Playwright `chromium.connectOverCDP("http://127.0.0.1:9223")` 连接真实 Lime/Electron 页签，断言 `window.__LIME_ELECTRON__ === true`、`window.electronAPI.invoke` 存在。真实发送 `@写文章 ... WRITING_CDP_1783188149738 ...` 后，独立保存 `agentSession/turn/start` 即时 trace evidence：`.lime/qc/gui-evidence/writing/writing-cdp-WRITING_CDP_1783188149738-turn-start-trace.json`，其中 `command=app_server_handle_json_lines`、`transport=electron-ipc`、`status=success`、`sessionId=sess_986a1c6327914bd58233cebfa50a530e`、`turnId=a337d0fa-f791-43eb-bb34-0380b005b8b7`，metadata 含 `plugin_activation.workflow_key=content_article_workflow` 与 `plugin_activation_intent.workflow_key=content_article_workflow`。同轮 GUI 证据 `.lime/qc/gui-evidence/writing/writing-cdp-WRITING_CDP_1783188149738-summary.json` 全部断言为 true：真实 Electron CDP、Electron IPC turn/start、写作 metadata、用户 prompt 可见、写作思路 / 过程文案可见、`article-artifact-frame` 可见且有正文、点击后 Article Editor 打开、`lime_invoke_error_buffer_v1` 无错误。截图：`writing-cdp-WRITING_CDP_1783188149738-after-send.png`、`writing-cdp-WRITING_CDP_1783188149738-final.png`、`writing-cdp-WRITING_CDP_1783188149738-editor.png`。备注：`lime_invoke_trace_buffer_v1` 是 240 条 ring buffer，后续 debug/list 调用会挤掉早期 turn/start；本次验证已在观察到匹配 turn/start 的瞬间单独落盘，不再把最终 ring buffer 是否仍保留该条作为失败条件。

## 5. 剩余缺口优先级

| 优先级 | 缺口                             | 原因                                                                                                                                                            |
| ------ | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1     | LimeCore v1 package release 同步 | 宿主闭环稳定后再改云端控制面，避免服务端提前固化半成品合同。                                                                                                    |
| P2     | connectors grant resolver runtime 化 | 宿主 contract、输入 metadata、workspace patch 兼容面、runtime readiness 投影和内容工厂 prompt/task hook lifecycle 已收口；剩余 P2/P1 交界缺口是 connectors 仍未接入真实宿主授权 / grant resolver，registry-only refs 只能标记为 `declared`。 |
| P3     | 图片 workflow 深化               | 配图位入口、任务完成 slot 原位替换、多图 slot 分配、失败 / 取消不误替换、失败 / 取消可见重试已补；`useWorkspaceImageWorkbenchActionRuntime.ts` 已降到 `1000` 行以下，`useWorkspaceImageTaskPreviewRuntime.ts` 已降到 `983` 行，图片任务 preview runtime 体量风险已回到仓库阈值内；`pluginContract.ts` 已降到 `501` 行，`clis` / `hooks` 一等 contract 已补齐。 |

## 6. 完成判定

MVP 只有在以下条件同时满足时才算完成：

- `@写文章` 来自内容工厂已安装插件。
- request metadata 带完整 workflow orchestration。
- runtime 产生 articleDraft artifact / workspace patch。
- 聊天先出现任务卡 / 对话流过程态，再出现独立 `ArtifactFrame`，最终文章在框内流式输出。
- 点击产物框展开右侧 Article Editor。
- Playwright 通过真实点击验证上述路径。

## 7. 禁止事项

- 不新增宿主内置 `writing_runtime` 作为 `@写文章` 主入口。
- 不在 `browserTaskRequirement.ts` 或类似工具里继续 hard code 内容工厂。
- 不为未安装内容工厂提供假入口。
- 不把完整文章塞进 assistant message。
- 不把非文章编辑器面板当文章编辑器。
- 不绕过内容工厂插件 workflow 直接调 provider。
