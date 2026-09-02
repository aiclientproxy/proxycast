# Writing 路线图

更新时间：2026-06-30
状态：`legacy current reference`（已由 Writing v2 路线图替换）
主线：Agent Plugins v1.0.0 标准包下的内容工厂写文章闭环

当前路线图：`internal/roadmap/Writing/v2/README.md`

> 版本替换说明：本文和同目录 v1 文档保留基础 UI 闭环、迁移证据与历史设计记录，不再定义 Writing 的 current 执行合同。普通 Agent turn 编排、workflow audit、段落级 artifact、JSONL 审计、真实 Electron/CDP 验收和生产 readiness 统一以 [Writing v2](./v2/README.md)、[v2 产品需求](./v2/product-requirements.md)、[内容工厂插件重新梳理](./v2/content-factory-plugin-reframe.md) 与 [v2 执行计划](../../exec-plans/writing-v2-workflow-completion-plan.md) 为准。

## 1. 目标

Writing 的目标很简单：用户在 Claw 里发起“写一篇文章”，Lime 应该启动已安装的内容工厂插件，通过标准包提供的 Skills 与 MCP 能力完成资料搜索、结构策划、正文写作、审稿和配图规划。workflow、子智能体、CLI、连接器和 hooks 属于 App Server/runtime 的产品投影，不再扩展 portable manifest。对话区任务卡和对话流过程态留在对话流里，最终文章成熟后再进入独立 `ArtifactFrame`。文章类 `ArtifactFrame` 内部使用 `articleArtifacts` renderer，可流式输出最终文章；点击框头或打开按钮后，按 `../rightsurface/README.md` 的 dock / tab 标准展开右侧 Article Editor 可编辑画布。

```text
@写文章 / @写作
  -> 内容工厂插件激活
  -> 任务卡 / 对话流过程态
  -> content_article_workflow
  -> 声明 searchRequests，由宿主 connector / tool timeline 执行并回填 evidence
  -> 策划 / 写作 / 校对 / 配图规划
  -> 聊天 ArtifactFrame(articleArtifacts renderer) 最终产物
  -> 右侧 Article Editor 可编辑画布
```

## 2. 设计结论

| 结论                       | 口径                                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 采用 Agent Plugins portable 标准 | 包根只保留 `plugin.json`、`mcp.json` 和 `skills/<skill>/SKILL.md`；Lime 私有运行时字段不进入包合同。 |
| 写作不是宿主内置能力       | `@写文章`、workflow、subagents、Skills、CLI 和 hooks 都来自已安装包的 typed activation projection。                  |
| 产物有独立框架             | 普通 assistant message 不承载整篇文章；过程态先在对话区回显，最终文章在独立 `ArtifactFrame` 内流式输出，右侧 Article Editor 负责编辑和深加工。 |
| 旧 Profile 路径已删除      | Article Workspace 是右侧产物事实源，Article Editor 是唯一文章编辑界面；旧 Profile 路径不再保留。                         |
| 右侧布局归宿主             | 右侧 dock / tab / pane 规则统一遵循 `../rightsurface/README.md`；内容工厂只贡献结果与 renderer 投影。 |
| 不登录也能用本地已安装插件 | 云端 marketplace 登录失败不能阻断本地 installed catalog 和本地插件激活。                                                 |
| 运行时投影不伪装成包合同   | workflow、evidence、Article Workspace 和 renderer 由 App Server/read model 承接，不从旧 YAML manifest 读取。 |

## 3. 插件标准

内容工厂是 [Plugin v3 标准](../plugin/v3/README.md) 的第一个样板闭环。包结构、字段、目录契约和 validator 规则以 [目标合同](../plugin/v3/01-target-contract.md) 为事实源。

Writing 路线图只描述写文章闭环；运行时 workflow、evidence、ArtifactFrame 和 Article Workspace 由 App Server 与 Thread/Turn/Item projection 承接。
右侧 dock / tab / pane 的统一标准见 `../rightsurface/README.md`；Writing 只保留 articleDraft / Article Editor 相关子面说明。

## 4. 文档索引

| 文档                                                                                     | 用途                                                                                                          |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| [`v2/product-requirements.md`](./v2/product-requirements.md)                             | current 背景、产品合同、流程、数据模型和验收标准。                                                           |
| [`v2/content-factory-plugin-reframe.md`](./v2/content-factory-plugin-reframe.md)         | current 执行卡片、文章产物、右侧编辑器和后台审计边界。                                                       |
| [`../../exec-plans/writing-v2-workflow-completion-plan.md`](../../exec-plans/writing-v2-workflow-completion-plan.md) | current 实施切片、验证入口和剩余缺口。                                      |
| [`product-requirements.md`](./product-requirements.md)                                   | v1 历史产品需求，effective decisions 已迁入 v2。                                                            |
| [`architecture.md`](./architecture.md)                                                   | v1 历史架构记录，effective decisions 已迁入 v2。                                                            |
| [`workflow-design.md`](./workflow-design.md)                                             | v1 历史 workflow 记录，effective decisions 已迁入 v2。                                                      |
| [`sequence-diagrams.md`](./sequence-diagrams.md)                                         | v1 历史时序记录，effective decisions 已迁入 v2。                                                            |
| [`implementation-plan.md`](./implementation-plan.md)                                     | v1 历史实施记录，current 进度以 v2 执行计划为准。                                                           |
| [`prototypes/article-artifacts-editor.html`](./prototypes/article-artifacts-editor.html) | 通用 `ArtifactFrame`、文章 renderer 和右侧 Article Editor 的静态交互原型。                                    |

## 5. 与 Plugin v3 的关系

Writing 是 Plugin v3 内容工厂主线的第一个可用闭环。v3 的总边界如下：

- 插件是分发和授权根对象。
- Plugin 是插件内 UI 能力，不是宿主内置页面。
- Claw 中间区域保留对话、运行过程和审批。
- 右侧 Article Editor 是文章产物的唯一编辑承载区。
- Article Workspace 承接插件工作区事实，旧 Profile 路径不再作为内部或外部事实源。
- 历史恢复恢复插件上下文和业务对象，不只恢复聊天。

## 6. MVP 完成判定

- [x] 内容工厂使用 Agent Plugins v1.0.0 portable 包，`plugin.json`、`mcp.json` 和 `skills/` 是唯一包入口。
- [x] workflow、evidence、Article Workspace 与 renderer 投影退出包 manifest，统一由 App Server/read model 承接。
- [x] App Server/runtime projection 暴露 `@写文章` / `@写作` / `@内容工厂` 激活入口。
- [x] App Server/runtime 为内容工厂生成 `content_article_workflow`、subagents、skillRefs、CLI、connectors 和 hook policy 的 activation metadata。
- [x] 宿主只读取 typed projection，并投影 activation entries、Skills、MCP、`ArtifactFrame` 和 articleArtifacts renderer contract。
- [x] `@写作` 激活时向 request metadata 写入 workflow、subagents、skill refs、CLI refs 和 hook policy。
- [x] Electron fixture 真实点击验证：插件中心可见内容工厂，输入框可 `@写文章`，发送后先出现任务卡 / 对话流过程态，再出现独立 `ArtifactFrame`，框内流式输出最终文章，点击展开右侧 Article Editor。
- [x] App Server 在 `content.article.generate` 接受后立即发出 `content_factory.workspace_patch` streaming snapshot；最终 runtime patch 覆盖同一 articleDraft，不被初始草稿污染历史恢复。
- [x] 宿主 connector / tool timeline 执行 `searchRequests` 并把真实 evidence 回填到 articleDraft metadata，不退化成普通聊天长文。
- [x] 历史会话恢复后默认看到 articleDraft Article Editor，并恢复已编辑正文。

## 7. 非目标

- 不把写文章做成 `src/components` 里的硬编码内置入口。
- 不恢复旧内容工厂独立 App shell。
- 不把整篇文章直接散落在普通 assistant message；完整文章必须在独立 `ArtifactFrame` 中输出。
- 不为未安装插件伪造 `@写文章` 候选。
- 不让内容工厂 runtime 直接拥有右侧栏布局。
- 不恢复旧 Profile 调试面板或相关兼容入口。
- 不把其他产品的插件格式作为 Lime 发布标准。
