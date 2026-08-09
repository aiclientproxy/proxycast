# Writing Workflow 设计

更新时间：2026-06-30
状态：In Progress

## 1. Workflow 定义

`content_article_workflow` 是 App Server/runtime 从标准插件激活态生成的文章生产 workflow projection。它不是 portable manifest 字段，也不是普通聊天 prompt。

右侧 dock / tab / pane 的统一规则见 `../rightsurface/README.md`；本文件只描述 articleDraft 产物、写作 workflow 和 Article Editor 子面内容。

```text
content_article_generate
  -> content_article_workflow
  -> task card / process state
  -> content-researcher
  -> content-strategist
  -> article-writer
  -> copy-editor
  -> image-planner
  -> ArtifactFrame(articleArtifacts)
  -> articleArtifacts.articleDraft
```

## 2. 子 Agent

| 子 Agent             | 责任                                     | 输出                           |
| -------------------- | ---------------------------------------- | ------------------------------ |
| `content-researcher` | 搜索主题、事实、案例、竞品和上下文材料。 | research notes、引用、风险点。 |
| `content-strategist` | 判断角度、受众、平台和结构。             | brief、标题候选、文章大纲。    |
| `article-writer`     | 根据 brief 和 research 写正文。          | Markdown draft、摘要、标题。   |
| `copy-editor`        | 校对、压低 AI 味、检查事实和表达。       | revised draft、问题清单。      |
| `image-planner`      | 规划封面、段落配图和图片提示词。         | image plan、slot refs。        |

## 3. Skills

| skill                | 责任                         | 备注                               |
| -------------------- | ---------------------------- | ---------------------------------- |
| `article-research`   | 资料检索、事实整理和引用归档。 | 标准包 Skill，由插件提供，不由宿主硬编码。 |
| `article-strategy`   | 选题、角度、受众和结构策划。   | 输出标题候选、大纲和写作计划。     |
| `article-writing`    | 中文文章写作和正文成稿。       | 负责 `articleDraft.source.markdown`。 |
| `article-editing`    | 审稿、校对、事实检查和表达调整。 | 输出审稿清单和修改建议。           |
| `article-image-plan` | 封面、段落配图和图片提示词规划。 | 输出 image slots 和 image plan。    |

## 4. CLI / Connectors / Hooks / 工具

| 能力                          | 责任                                                                       |
| ----------------------------- | -------------------------------------------------------------------------- |
| `search_query` / WebSearch    | 由 App Server/runtime projection 生成 `searchRequests`，由宿主 connector / tool timeline 执行真实检索、事实补充和引用确认，并把 evidence 回填到 articleDraft metadata。 |
| runtime projection            | 执行内容工厂 workflow，产出 workspace patch 和结构化 evidence。              |
| typed plugin projection       | 投影 Skills、MCP、工具、授权和可用性；不把这些运行时能力写回 manifest。       |
| connectors                    | 由 App Server/runtime 记录搜索、知识库、云端账号、媒体生成等外部依赖和授权状态。 |
| tool refs                     | 记录当前 turn 的工具依赖与能力标签；它不是 connector 列表。                  |
| hooks                         | 在 prompt / tool / task 生命周期中注入运行约束、路由策略和 evidence 归档。 |
| `artifact writer`             | 保存 Markdown / workspace patch / evidence。                               |
| `right surface action router` | 处理继续改写、生成配图、导出等受控动作。                                   |

## 5. 编排流程

```mermaid
flowchart TD
  Start[触发 @写文章] --> Parse[解析主题 / 平台 / 受众 / 约束]
  Parse --> TaskCard[先在对话流回显任务卡 / 过程态]
  TaskCard --> Research[content-researcher 声明多轮检索请求]
  Research --> HostSearch[宿主 connector 执行 searchRequests 并回填 evidence]
  HostSearch --> Strategy[content-strategist 选题和结构]
  Strategy --> Draft[article-writer 正文写作]
  Draft --> Edit[copy-editor 审稿校对]
  Edit --> ImagePlan[image-planner 配图规划]
  ImagePlan --> Materialize[物化 articleArtifacts.articleDraft]
  Materialize --> Frame[聊天 ArtifactFrame 最终产物框]
  Frame --> Editor[右侧 Article Editor]
```

## 6. ArtifactFrame 规则

`ArtifactFrame` 是聊天区里的通用独立产物框，承担“承载产物、展示状态、流式更新、点击进入右侧画布”的入口。文章只是其中一种 renderer，后续还应支持图片集、表格、演示稿、网页、报告、代码和媒体产物。注册链以 `ArtifactFrame` 为事实源，不再使用 message 专用命名。写作流程里，任务卡和过程态先回显在对话区，最终文章成熟后才进入 `ArtifactFrame(articleArtifacts)`，右侧展开行为遵循 `../rightsurface/README.md`。

当前 runtime projection 的安全契约是 `directProviderAccess=false`、`directFilesystemAccess=false`：runtime 可以输出 `searchRequests`、pending `searchEvidence`、`reviewChecklist` 和 `imagePlan`，但不能直接联网或读写宿主文件。真实检索必须由宿主 connector / tool timeline 执行并回填；在这一步完成前，不能把“多轮检索结构已生成”写成“真实搜索已完成”。

通用框架必须包含：

- 标题。
- 状态：`running`、`ready`、`needs_review`、`failed`。
- renderer 类型：如 `articleArtifacts`、`imageSetArtifacts`、`tableArtifacts`。
- 内部内容区：文章类允许完整正文流式输出；图片类显示网格；表格类显示表格预览。
- 框头动作：展开 / 收起、打开右侧、继续处理。
- object ref / artifact ref，用于右侧展开、历史恢复和后续动作。

产物框不应包含：

- 大段 prompt。
- provider 或 stack trace。
- 旧 Profile 调试字段。

文章 `ArtifactFrame` 规则：

- 可以完整显示正文，不只显示摘要。
- 正文更新必须进入框内流式内容区，不进入普通 assistant message。
- 内容区应有最大高度或内部滚动策略，避免长文把输入区挤出视口。
- 点击框头或打开按钮进入右侧 Article Editor。

## 7. Article Editor 规则

右侧 `articleDraft` Article Editor 至少包含：

> 右侧布局、dock 和 tab 规则统一见 `../rightsurface/README.md`，这里仅列写作编辑器的内容构成。

| 区域       | 内容                                                 |
| ---------- | ---------------------------------------------------- |
| 标题栏     | 文章标题、状态、版本、来源 workflow。                |
| 左侧工具条 | 大纲、编辑、引用、配图、版本、导出等文章级工具入口。 |
| 正文画布   | Markdown / ArtifactDocument 草稿，支持段落级编辑。   |
| 结构面板   | 大纲、摘要、目标平台、受众。                         |
| 引用面板   | 搜索来源和事实依据。                                 |
| 配图面板   | 配图 slot、提示词、生成状态。                        |
| 动作       | 继续改写、补充搜索、生成配图、导出。                 |

旧 Profile 路径不再作为内部事实源、文章主界面或右侧用户可见标题；Article Workspace 是唯一工作区事实源。

## 8. 失败处理

| 阶段       | 失败表现                                  | 用户可做                   |
| ---------- | ----------------------------------------- | -------------------------- |
| 未安装插件 | 不显示 `@写文章` 候选，插件中心提示安装。 | 安装内容工厂。             |
| 插件不可用 | 候选置灰并显示 blocker code。             | 查看插件详情或授权。       |
| 搜索失败   | 卡片停在 research failed。                | 重试、跳过搜索或补充素材。 |
| 写作失败   | 卡片停在 failed，不生成假正文。           | 重试或查看 evidence。      |
| 物化失败   | 不打开右侧空白 Article Editor。           | 显示 artifact error card。 |

## 9. 验证点

- `@写文章` 输入建议来自 installed plugin contract。
- activation metadata 包含 workflow、subagents、skills、CLI refs、connector refs 和 hook policy。
- runtime evidence 包含 `workflowKey` 和 `orchestration`。
- runtime evidence 包含 `searchRequests`、pending `searchEvidence`、`reviewChecklist` 和 `imagePlan`。
- 宿主 connector 执行 `searchRequests` 后，真实检索结果回填到 articleDraft metadata 和工具时间线。
- 聊天出现独立 `ArtifactFrame`，文章正文在框内流式输出。
- 产物框点击能打开右侧 `articleDraft` Article Editor。
- 历史恢复能恢复 selected object。
