# 内容工厂 App 开发说明

本仓库只开发内容工厂 Lime Plugin Package v1，不迁移旧 `content-studio` 或旧内容工厂程序。

## 宿主边界

- Claw 中间区负责对话、输入和文章产物流式反馈；workflow run / step / tool / connector / hook 明细进入宿主 JSONL 审计。
- 右侧 `articleWorkspace` 展示产物对象和操作入口。
- App 输出 `content_factory.workspace_patch`，由 Lime 宿主物化成 Article Workspace。
- 产物操作回流为 Claw turn，不新增 `content_factory_*` 垂直命令。
- App 不能直接访问 provider key、Electron IPC、文件系统、secret 或 App Server transport。

## 本地开发顺序

1. 修改 `plugin.json`、`app.workbench.yaml` 或 `app.runtime.yaml`。
2. 更新 `artifacts/content-factory-workspace-patch.schema.json`。
3. 更新 `examples/workspace-patch.sample.json`，保持能被 Lime 右侧 Article Workspace 识别。
4. 更新 `src/runtime/content-factory-worker.mjs` 和 `examples/runtime-request.sample.json`。
5. 运行 `npm test`、`npm run runtime:sample` 和 `npm run validate:app`。
6. 在 Lime 应用中心安装本地目录，打开 Claw 历史会话检查右侧 Article Workspace。

## Agent Skills

内容工厂的 `skills/` 层采用 Agent Skills 目录和 frontmatter 规则：

- 每个技能使用 `skills/<skill-name>/SKILL.md`。
- `<skill-name>` 只使用小写字母、数字和连字符。
- `SKILL.md` frontmatter 的 `name` 必须与父目录一致。
- `description` 必须说明技能做什么以及何时使用。
- 禁止使用 `skills/article_writing` 这类下划线 legacy 目录。

当前技能目录是 `article-research`、`article-strategy`、`article-writing`、`article-editing` 和 `article-image-plan`。`app.runtime.yaml` 的 `skillRefs` 必须引用这些稳定 skill name，不引用文件路径。

## Worker Runtime

当前 worker 是可运行的 producer-side artifact streaming 实现：

```bash
npm run runtime:sample
```

输入是 `examples/runtime-request.sample.json`。宿主 worker request 会输出 NDJSON：

- 前若干行包含 `kind=runtime.event / eventType=workflow.connector.requested` 的 audit-only connector 请求，必须带 `stepId=research`、`connectorRef=web-research` 和 `toolName=WebSearch`。Lime 宿主会用插件 workflow manifest 绑定 `workflowRunId + stepId` 并只写入 `workflow-events.jsonl`。
- 随后是 `kind=runtime.event / eventType=artifact.snapshot` 的段落级 partial，`metadata.streamSource=worker_delta`，`metadata.streamSequence` 递增，并进入右侧 Article Workspace 产物投影。
- 最后一行是 `schemaVersion=content-factory.worker-response.v1` 的最终 response，包含 complete workspace patch。

如果 Lime RuntimeBackend 执行了 workspace patch 里的 `searchRequests`，真实 `WebSearch` 结果会先回填到 `objects[].source.hostSearchEvidence`，再由宿主追加 audit-only `workflow.connector.completed` 到 `workflow-events.jsonl`。该 completed 事件不由 worker stdout 直接输出，也不进入右侧 UI。

插件 hook lifecycle 由 Lime 宿主执行后追加 audit-only `workflow.hook.completed` 到 `workflow-events.jsonl`。hook 事件不进入普通 runtime event stream、workerEvidence 或右侧 Article Workspace。

最终 patch 至少包含：

- `artifactKind=content_factory.workspace_patch`
- `patch.schemaVersion=article-workspace.v1`
- `patch.appId=content-factory-app`
- `patch.objects[]`，按 `taskKind` 包含内容简报、文章草稿、图片生成组、视频脚本、视频分镜或交付检查清单

worker 不直接访问 provider key、Electron IPC、文件系统或 secret。真实模型正文生成由 Lime 宿主在启动 worker 前执行，并把结果注入 `hostManagedGeneration.outputs[]` / `runtime.hostManagedGenerationResult`；worker 只消费受控结果，不持有 provider key。

当前优先级：

- 宿主声明 `app.runtime.yaml#agentRuntime.worker.hostManagedGeneration`。
- 宿主按当前模型路由生成正文后，把 `status/provider/model/outputs[]` 写回 worker request。
- worker 优先使用 `articleDraft.documentText` 对应 output。
- 如果宿主无可用 provider、生成失败，或当前 backend 没返回 `hostManagedGenerationResult`，`content.article.generate` / `content.factory.generate` 必须 fail closed，不得使用 deterministic fallback 生成合同样例或伪造文章正文。

Lime 宿主会从 `plugin.json#contributions.runtime` 与 `app.runtime.yaml#agentRuntime.worker/tasks` 读取 task runtime 合同，并在插件安装态和 runtime 状态中暴露 worker readiness。worker 输出的 partial snapshot 必须由宿主通用透传和持久化，宿主不得用最终 `documentText` 二次切片来伪造流式。

workflow step / connector / hook / tool / evidence 的审计事实由 Lime 宿主写入 `workflow-events.jsonl`；内容工厂 App 不在右侧 Article Workspace 里自建流程轨或步骤面板。

## Article Workspace Patch

`examples/workspace-patch.sample.json` 是当前最小输出合同。worker / task runtime 后续实现时，必须至少输出：

- `schemaVersion`
- `appId`
- `sessionId`
- `objects`
- `primaryObjectRef`
- `selectedObjectRef`
- `layoutState.openTabKinds`

如果新增业务对象，先扩 `app.workbench.yaml` 的 `productionObjects / objectSurfaces`，再扩 schema 和样例。

`selectedObjectRef` 表示 App 输出 workspace patch 时建议宿主默认打开的对象。用户在 Lime 右侧 Article Workspace 内点击切换对象时，瞬时选择由 Renderer 本地维护；需要持久化时，宿主通过 `artifact/write` 写入包含完整 `workspacePatch` 和最新 `selectedObjectRef` 的 artifact snapshot。选择状态不写入 session metadata，内容工厂 App 也不需要为 UI selection 新增独立命令。

## 右侧预览字段

Lime 宿主会从 `objects[].source` 读取以下可选字段来渲染右侧 Article Workspace：

- `markdown`：文章草稿或视频脚本正文预览。
- `images[]`：图片候选，字段包含 `id / title / url / localPath / filePath / cachedPath / alt / prompt`。`url` 可用于远程预览；`localPath / filePath / cachedPath` 用于宿主可访问的本地缓存文件，建议提供绝对路径，宿主会转换为 Electron 可渲染地址，非绝对路径只作为来源证据保留。
- `shots[]`：视频分镜，字段包含 `id / title / description / visualPrompt / duration`。
- `items[]`：交付检查清单，字段包含 `id / title / status / notes`。
- `fields[]`：内容简报字段，字段包含 `key / label / value`。

这些字段只是展示合同，后续修改仍必须通过 Claw turn 或受控 action intent 回到宿主主链。

`images[].cache` 和 `shots[].cache` 是媒体缓存 executor 合同，当前 worker 只声明：

- `executor=content-factory.media-cache.v1`
- `status=pending_executor`
- `kind=image | video`
- `relativePath / manifestPath / mimeType`

这些字段表示后续宿主或受控 executor 应把真实媒体写入哪个相对路径；内容工厂 App 本身仍不能直接读写文件、访问模型 key 或绕过 Claw / App Server 主链。

## 正式预览打开链路

右侧 Article Workspace 的“打开预览”由 Lime 宿主完成：

- 文档 / 简报 / 分镜 / 清单会被宿主投影为 source-backed Markdown preview artifact。
- 图片组优先使用 `images[0].url`；没有远程 URL 时会尝试使用 `images[0].localPath / filePath / cachedPath` 投影为 media preview artifact。
- 如果结构化预览为空，宿主会退回 `previewArtifactId` 或 `artifactIds[]` 生成可打开的占位预览。

内容工厂 App 只负责输出 Article Workspace Patch，不要在 App 内自建 viewer、iframe、Electron IPC 或文件读取旁路。
