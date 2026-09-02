# i18n Glossary

> 目标：统一 P3 自动翻译、review、PR 标注和后续术语审阅口径。

## 产品名

- Lime：桌面主 App 名称，所有 locale 保持同一专名。
- Lime Cloud：云端服务名，避免写成一般意义的 cloud。

## 功能名

- Agent Chat：主对话工作区。
- Workspace：承载任务、文稿、运行态和结果的主工作台。
- Browser Runtime：浏览器协助运行时。
- Plugin：可安装、启停和发现的扩展包及其产品入口。
- Skills Workspace：技能生成、浏览、安装与运行工作台。
- Project Knowledge：项目资料中枢。
- Memory：记忆与上下文沉淀页。

## Agent 术语

- task：面向用户可理解的任务，不翻译成过于抽象的 work item。
- thread：对话线程，保留线程语义，不混成 session。
- handoff：交接，不翻译成 hand-over 之类生硬表达。
- evidence：证据 / 运行证据，优先保持审计语义。
- projection：投影，表示从 runtime 事实映射到 UI 状态。
- curated task：策划型任务模板，保留 curated 语义。

## Browser Runtime 术语

- browser runtime：浏览器运行时，不缩写为 browser。
- site adapter：站点适配器。
- site profile：站点配置档。
- assistant / assist：浏览器协助，避免和 agent 混淆。
- preview：预览，保留为运行前或结果前的可视草稿。

## Workflow 术语

- workflow：由 Plugin、Skill 或 Agent Turn 提供的可执行流程。
- run projection：运行投影。
- follow-up：后续动作 / 续接，不翻译成一般意义上的 follow up 如果会失去流程语义。
- result destination：结果去向。
- artifact：产物，保留文件 / 页面 / 结果语义。

## 翻译规则

- 产品名、专名、路径名、协议名保持稳定，不随 locale 自由意译。
- 作为 key 的术语优先一致翻译，不要在同一 namespace 里前后不一致。
- 自动翻译只生成 PR 草案，不直接覆盖 source locale。
- 新增或调整翻译前，先查本 glossary，再补充 namespace 级别约定。
