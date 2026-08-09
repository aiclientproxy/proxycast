# Writing 产品需求

更新时间：2026-06-30
状态：In Progress

## 1. 背景

当前用户目标不是“打开一个内容工厂页面”，而是在 Lime 中完成一件具体任务：写一篇文章。01Agent 的交互可以参考，但 Lime 的布局边界不同：Lime 的中间区域是 Claw 对话和运行过程，画布 / 侧边栏在右侧，不应复制 01Agent 的左侧或全屏画布。

内容工厂已经作为插件存在，因此写作入口不应再通过宿主硬编码。正确路径是让内容工厂按 Agent Plugins v1.0.0 portable 包提供 `plugin.json`、`mcp.json` 和 Skills；workflow、子智能体、CLI、connectors、hooks、运行 metadata 和 Article Workspace 属于 App Server/runtime projection，不再写入私有 manifest。宿主只负责安装态发现、显式激活、独立 `ArtifactFrame`、article renderer 和右侧 Article Editor，右侧布局标准统一见 `../rightsurface/README.md`。

前一版把 Profile 当成右侧文章主界面是错误方向。当前不再保留这条兼容路径；Article Workspace 承接插件工作区事实，用户看到的产物界面必须是 Article Editor 画布，并遵循 `../rightsurface/README.md` 的 dock / tab 规则。

## 2. 目的

1. 用户输入 `@写文章`、`@写作` 或选择内容工厂写作入口后，明确进入内容工厂插件 workflow。
2. 写作过程不是一次普通聊天回答，而是先展示搜索、策划、写作、校对和配图规划的多步编排，再进入最终产物。
3. 文章正文最终作为结构化 `articleArtifacts` 出现，并在独立 `ArtifactFrame` 内完整、流式输出。
4. 用户点击 `ArtifactFrame` 的打开入口后，右侧 Article Editor 展开同一篇文章草稿、结构、引用、配图规划和后续动作。
5. 未登录云端账号时，本地已安装内容工厂仍可见、可 `@`、可启动。

## 3. 收益

| 角色        | 收益                                                                                                 |
| ----------- | ---------------------------------------------------------------------------------------------------- |
| 普通用户    | 用一句“写一篇文章”进入完整写作流程，不需要理解插件包或 workflow 概念。                               |
| 内容创作者  | 能在右侧 Article Editor 中看到文章结构、引用、配图建议和审稿状态，后续可继续改写。                   |
| 插件开发者  | 能通过标准插件包提供 Skills/MCP，并由 typed activation projection 暴露写作能力，不需要宿主写死能力。 |
| Lime 宿主   | 保持 Claw / Right Surface / 历史恢复的一致架构，不为单个插件新增分叉页面。                           |
| 运营 / 分发 | 插件中心能解释“内容工厂用了哪些 Agent、工具、授权和 skills”，降低黑盒感。                            |

## 4. 用户故事

| 编号  | 用户故事                                                                 | 验收                                                                                                            |
| ----- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| WS-01 | 作为用户，我想在输入框 `@写文章`，让 Lime 帮我写一篇文章。               | 输入建议来自已安装内容工厂插件；未安装时不出现伪候选。                                                          |
| WS-02 | 作为用户，我想看到系统先搜索和整理，再开始写文章。                       | timeline / 小卡状态先体现 research、strategy、writing、editing 等步骤，最终再出现文章产物框。                 |
| WS-03 | 作为用户，我希望文章产物在一个独立框里完整输出，而不是混进普通聊天正文。 | 过程说明留在对话区，`ArtifactFrame` 内只展示最终文章并可流式输出。                                               |
| WS-04 | 作为用户，我想点击文章产物框后在右侧继续编辑或生成配图。                 | `ArtifactFrame` 打开入口点击后打开右侧 `articleDraft` Article Editor。                                          |
| WS-05 | 作为用户，我即使没有登录云端账号，也能使用本地已安装内容工厂。           | cloud marketplace 401/403 只影响云端列表，不阻断 installed registry。                                           |
| WS-06 | 作为插件开发者，我想让标准包参与写作 workflow。                            | App Server 从安装态与 Skills/MCP 生成 `activationEntries`、`workflows`、`subagents`、`skillRefs`、CLI、connectors 和 hooks，不靠 hard code。 |

## 5. 用户用例

### UC-01：从 `@写文章` 发起

1. 用户在 Claw 输入框键入 `@写文章 写一篇关于 AI Agent 工作流的公众号文章`。
2. 输入栏命中内容工厂插件的 activation entry。
3. 发送时 request metadata 写入 `plugin_activation`、`workflow_key=content_article_workflow`、subagents、skill refs、CLI refs、connector refs 和 hook policy。
4. Runtime 按 workflow 执行写作。
5. 聊天中先出现任务卡和过程态，再出现独立 `ArtifactFrame`，框内 `articleArtifacts` renderer 流式输出最终文章。
6. 用户点击产物框打开入口，右侧 Article Editor 打开同一篇文章草稿。

### UC-02：从插件中心安装后使用

1. 用户打开插件中心。
2. 内容工厂卡片展示已安装或可安装状态。
3. 详情页展示 Subagents、Skills、CLI tools、Connectors、Hooks、authorization。
4. 用户安装或确认已安装后，输入框可 `@写文章`。

### UC-03：未登录云端时使用本地插件

1. 云端 marketplace 返回认证失败。
2. 插件中心仍展示本地 installed 内容工厂。
3. 输入栏 `@写文章` 仍可用。
4. 发送后启动本地内容工厂 workflow。

### UC-04：历史恢复

1. 用户从历史打开之前的写作会话。
2. Lime 读取 session plugin workspace。
3. 如果有 selected `articleDraft`，右侧 Article Editor 直接恢复该文章。
4. 如果没有 selected object，则恢复 primary article object。
5. 只有无 workspace / artifact 时才回退聊天历史。

## 6. 体验约束

1. 输入建议、插件详情和 workflow 激活必须来自插件事实源。
2. 普通聊天正文不承载完整文章；普通聊天正文只承载任务卡和过程态，完整文章只允许进入独立 `ArtifactFrame`。
3. `ArtifactFrame` 需要稳定尺寸、标题栏和内部滚动策略，避免流式内容把布局撑乱。
4. 右侧 Article Editor 可展开 / 收起，并保留宿主 tab 行为。
5. 子 Agent、skills、CLI、connectors 和 hooks 在插件中心可解释，在 runtime metadata 可追踪。
6. 写作 workflow 失败时，卡片显示失败状态和可重试动作，不把异常堆栈显示给普通用户。
7. 旧 Profile 字段和兼容入口不能再进入 current 读写链；诊断信息必须挂在 Article Workspace 或运行明细下。

## 7. 验收标准

- `@写文章` 和 `@写作` 只在内容工厂已安装并可激活时出现。
- 激活 metadata 包含 `workflow_key`、`workflow.steps`、`subagents`、`skill_refs`、`cli_refs`、`connector_refs` 和 `hook_policy`。
- Runtime evidence 能看到 workflow orchestration。
- 写作产物物化为 `articleDraft` / `articleArtifacts` / `content_factory.workspace_patch`。
- UI 先在对话流显示任务卡和过程态，再显示独立 `ArtifactFrame`，框内完整文章可流式输出，点击后右侧 Article Editor 打开。
- Playwright 覆盖插件中心可见、输入栏 `@`、发送、`ArtifactFrame` 流式内容、右栏展开。
