# 内容工厂插件重新梳理

更新时间：2026-07-05
状态：Draft + ordinary Agent turn orchestration current-turn verified + Electron/CDP real desktop baseline verified + Electron/CDP Gate B product acceptance verified + App Server current-turn host tool evidence verified + live Provider current-turn verified + production preflight/readiness pipeline fail-closed verified + production signature cryptographic preflight verified + inline host command shortcode contract added

> Plugin v3 对齐：portable 包只提供根 `plugin.json`、`mcp.json` 和 `skills/<skill>/SKILL.md`。本文早期记录中的独立 package execution、旧运行时 YAML 与旧本地包检查方法仅是历史验证材料，不属于 current Plugin owner；当前激活合同由 App Server typed activation snapshot 派生，普通 Agent turn 负责执行。

## 1. 结论

当前 Image #1 的问题不是文章卡本身错误，也不只是右侧面板错误，而是内容工厂插件的优势没有进入用户可感知的普通 Agent 主链：用户只看到占位框一闪而过、假完成或 JSON / 文件卡，感受不到 Agent 正在寒暄、思考、检索、写作、审稿和规划配图。

参考 Image #2，Writing v2 的目标应调整为：

- 聊天中间区展示可展开的执行卡片，例如“资料检索”“网络搜索”“文章生成”，用户可查看调用参数、执行结果和来源摘要。
- 文章正文仍按段落进入同一个文章产物框，完成后可打开右侧 Article Workspace。
- 右侧 Article Workspace 只承载文章、图片组、视频分镜和交付清单，不展示 workflow 流程轨。
- 完整 workflow run / step / tool / connector / hook 明细继续进入后台 `workflow-events.jsonl`，用于审计、排障和质量复盘。

也就是说，过程不能被简化到完全不可见；但可见过程必须是用户可理解的工具 / 资料卡，不是内部 workflow 面板。

2026-07-05 主线校正：

- `@写文章` 必须走普通 Agent 对话流程，不走旧右侧执行器 fast path，也不能用 mock/fixture 抢跑生成。
- 内容工厂的作用是提供 `workflow_contract`、host tool / artifact / shortcode 合同和包内执行能力；首发回合由普通 Agent turn 编排并对用户自然说明。
- 聊天区应包含自然引导文字、可展开执行卡片、同一文章产物框和完成后的自然总结；文案不能写死成模板，也不能出现特定品牌称呼。
- 右侧 Article Workspace 不自动打开；只有点击文章产物或显式动作才打开。
- raw `workspace-patch.json`、内部 JSON、workflow step 列表只进入审计或 read model，不作为普通聊天消息展示。
- CDP 验收必须区分 baseline 和 product acceptance：baseline 证明真实 Electron turn/start 与文章产物可见；product acceptance 才证明右侧不自动打开、历史恢复、执行卡片顺序和 raw JSON 隐藏。

## 2. 产品结构

页面类型：复杂内容生产工作台，主对象是 `articleDraft`，当前阶段是“生成首版文章并等待审核”。

内容工厂插件应暴露五类能力：

| 能力         | 用户可见形态                                            | 后台事实源                                                                                      |
| ------------ | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 激活入口     | `@写文章` / `@写作`                                     | App Server typed activation snapshot（由 `plugin.json`、`mcp.json` 与 `skills/` 派生）          |
| 资料检索     | 聊天主链可展开执行卡片                                  | `articleDraft.source.hostToolRequests` + App Server tool events                                 |
| 正文生成     | 同一文章产物框段落级增长                                | `artifact.snapshot` streaming partial                                                           |
| 正文内联命令 | 文章正文中的 `[@配图 ...]` 占位，宿主转为图片任务并回填 | `hostCommandRequests` / `image_command_intent.image_task` / `.lime/tasks/image_generate/*.json` |
| 后续产物     | 右侧 Article Workspace tabs / objects                   | `content_factory.workspace_patch`                                                               |
| 审计复盘     | 普通 UI 默认不展示                                      | `workflow-events.jsonl` metadata-only audit                                                     |

## 3. 主流程

1. 用户输入 `@写文章 ...`。
2. Lime 命中内容工厂插件 activation entry，把 `workflow_contract` 写入本轮 metadata。
3. 本轮进入普通 `agentSession/turn/start`，Agent 先输出自然引导、说明将按内容工厂流程编排，不直接宣布完成。
4. App Server 为本轮创建后台 workflow audit context；workflow run / step / connector 只写 `workflow-events.jsonl`。
5. 内容工厂合同在 `articleDraft.source.hostToolRequests[]` 暴露资料检索请求，宿主把它转成聊天主链里的工具卡。
6. 宿主执行 `WebSearch`，工具卡展示参数、结果摘要和来源；完整结果回填 `hostSearchEvidence`，并追加 `workflow.connector.completed` 到 JSONL。
7. Agent / 宿主按内容工厂合同生成正文，产物以同一 `content_factory.workspace_patch` 的段落级 `artifact.snapshot` 增长。
8. 正文中的 `[@配图 ...]` shortcode 由宿主解析为 `hostCommandRequests[]`，首批只自动执行 `@配图`，并复用 `image_command_intent` 生成 document-inline 图片任务。
9. final snapshot 在 `turn.completed` 前封口；聊天区显示文章完成状态和自然总结，右侧 Article Workspace 等待用户点击打开。

## 4. 插件包合同

外部 `/Users/coso/Documents/dev/ai/limecloud/content-factory-app` 必须守住以下合同：

- `plugin.json` 说明中明确“中间执行卡 + 右侧编辑器 + 后台审计”的产品结构。
- App Server activation snapshot 声明执行卡片来源是 `hostToolRequests`；该字段不进入 portable manifest。
- `articleDraft.source.hostToolRequests[]` 是当前事实源，旧 `searchRequests[]` 只作为兼容输入保留。
- `articleDraft.source.hostCommandRequests[]` 是更通用的宿主命令合同；当前 `hostToolRequests[]` 是其中 WebSearch 子集，首批正文 shortcode 只把 `[@配图 ...]` 自动执行为 `image_generate` document-inline 请求。
- streaming partial 不能为了精简正文而剥掉 `hostToolRequests`，否则宿主无法把插件能力投影为中间执行卡。
- shortcode 只能作为结构化命令占位输出，不能要求宿主用裸正则替换 Markdown；宿主会跳过代码块、inline code、链接和图片 alt，并为每篇文章最多自动处理 `3` 个 `@配图`。
- `workflow.connector.requested` / `workflow.connector.completed` 仍是 audit-only，不能直接作为普通 UI 的流程轨。
- `content.article.generate` / `content.factory.generate` 缺少宿主托管正文时必须 fail closed，不用模板正文伪造成功。
- `plugin_activation` 不能直接触发独立插件 runtime；右侧 pane/action 只响应用户显式动作，并继续委托 App Server current command。
- `workflow_contract` 由 manifest / workflow 派生，不能把参考图文案或固定文章模板硬编码到宿主。

## 5. 禁止项

- 不把 workflow run / step 列表塞回右侧 Article Workspace。
- 不在普通用户主界面显示 `Agent`、`Runtime`、`Artifact`、`manifest` 等工程词。
- 不让插件包直接输出 `tool.started` / `tool.result` 绕过宿主工具生命周期。
- 不让插件包直接创建图片任务、写 `.lime/tasks/image_generate/*.json` 或绕过 `@配图` current 主链。
- 不把最终正文拆成多张文章卡；必须更新同一个 article artifact。

## 6. 验收标准

- 外部插件包测试能证明 `articleDraft.source.hostToolRequests[]` 存在，且带 `presentation.surface=conversation_timeline`。
- 流式 partial 中仍保留 `hostToolRequests[]`。
- Lime 宿主 read model 能把 workspace patch host tool events 投影为 `web_search` / `tool_use` timeline item，并显示在文章产物卡之前。
- 正文中的 `[@配图 ...]` 会物化为稳定 slot marker，通过 `image_command_intent.image_task` 生成 document-inline 图片任务，并在 running / completed 状态回填 pending 占位或真实图片。
- Article Workspace 仍只恢复文章和后续产物，不恢复 workflow 流程轨。
- `workflow-events.jsonl` 继续输出 metadata-only 审计摘要，不暴露 raw prompt / provider config / 正文结果。
- App Server current-turn smoke 必须证明普通 Agent turn 里有 `message.delta`、host tool timeline、段落级 `artifact.snapshot` 和最终 read model；不能只证明 package fixture 本地运行。

## 7. 已验证证据

2026-07-03 已补强 `npm run smoke:content-factory-current-turn:host-generation`：同一 current turn smoke 现在同时断言 streaming / final `artifact.snapshot` 都保留 `3` 条 `hostToolRequests`，普通事件流和 session JSONL 都出现 `3` 组 `tool.started -> tool.args -> tool.result`，事件 `source=workspace_patch_host_tool_requests`，`agentSession/read.detail.items` 与 `thread_read.tool_calls` 都投影出 `3` 个 completed `web_search` 工具项，`evidence/export` 也能从 artifact snapshot 证明 `hostToolEvidence` 已回填。

本次证据文件：

- `.lime/qc/gui-evidence/plugins/content-factory-current-turn-smoke-host-generation-2026-07-03T12-31-25-916Z.json`
- `.lime/qc/gui-evidence/plugins/content-factory-current-turn-smoke-host-generation-2026-07-03T12-31-25-916Z.workflow-events.jsonl`
- `.lime/qc/gui-evidence/plugins/content-factory-current-turn-smoke-cloud-release-host-generation-2026-07-03T12-32-45-939Z.json`
- `.lime/qc/gui-evidence/plugins/content-factory-current-turn-smoke-cloud-release-host-generation-2026-07-03T12-32-45-939Z.workflow-events.jsonl`

## 8. 最新验证与下一刀

App Server current 普通 turn 已修到：`plugin_activation -> 普通 Agent turn -> 内容工厂 artifact/materialization -> JSONL audit -> turn.completed`。真实 Electron/CDP Gate B product acceptance 已通过，证明自然引导、执行卡片、文章产物、右侧不自动打开和历史恢复都在真实桌面主链闭环。下一步不再继续证明旧 package fixture，也不再把 baseline 当验收缺口；剩余主线转向 live Provider product flow 与远程 / GUI production 安装签名闭环。

2026-07-05 更新：当前实现方向已经落到 RuntimeCore terminal 顺序修复，而不是新增一个内容工厂专用 UI 分支。普通 Agent backend 发出的 `turn.completed` 会被暂存，内容工厂 activation producer 在 terminal 前补齐 article artifact、host tool timeline 和 JSONL audit；完成后再封口。验收仍以真实 current-turn smoke / Electron CDP 为准，不能把测试 fixture execution 或静态 mock 当成用户链路完成证据。

2026-07-05 后端验证已通过：`content-factory-current-turn-debug-host-generation-2026-07-05T04-19-07-937Z.json` 显示普通 current turn 中有 `message.delta`、`7` 个 `artifact.snapshot`、`3` 组 `tool.started -> tool.args -> tool.result`，`workflow-events.jsonl` 有 `16` 条 metadata-only audit 事件，最终 `turn.completed` 位于 artifact / tool 事件之后。后续 Electron/CDP acceptance 已验证真实桌面 GUI 的聊天排版、右侧不自动打开、历史恢复和 raw JSON / 文件卡隐藏。

2026-07-05 Electron/CDP baseline 已有真实桌面证据：`.lime/qc/gui-evidence/writing/writing-cdp-WRITING_CDP_1783188149738-summary.json` 显示 `usedElectronCdp=true`、`usedRealElectron=true`、`turnStartViaElectronIpc=true`、`writingActivationMetadataPresent=true`、`articleArtifactFrameVisible=true`、`articleArtifactHasBody=true`、`processOrGuidanceCaptured=true`、`noInvokeErrors=true`；同名 `turn-start-trace.json` 记录 `app_server_handle_json_lines`、`electron-ipc`、`content_article_workflow` 和真实 session / turn id。该证据不能替代最终验收，因为脚本主动打开了 Article Editor，且没有覆盖历史恢复和 raw JSON / 文件卡负向断言。

2026-07-05 read model / 历史恢复补丁已落地：普通 `agentSession/read` 的 artifacts 只保留用户可见文章 artifact document，过滤 raw workspace patch；前端历史恢复不再误杀 `artifact_document + articleWorkspace`，但仍隐藏 `content_factory.workspace_patch` / `workspace_patch` / `workspace-patch.json` 文件卡。插件侧仍可以继续输出 `content_factory.workspace_patch` 作为 Article Workspace 和审计事实源；它不会直接显示成聊天里的第二个文件框。App Server 同步修正 `item.updated` 累计文本合并，避免普通 Agent 引导文字在历史 read model 中重复拼接。

2026-07-05 Electron/CDP product acceptance 已通过：`/tmp/lime-writing-evidence/writing-final-WRITING_LIVE_1783229659461-2026-07-05T06-02-47-474Z-summary.json` 显示真实 Electron 页面 `http://127.0.0.1:1420/?nativeStartup=1` 中目标 session `sess_f781bf079f074b7aa2ec0941bade095d` 满足 `gateB / historyRestored / naturalLeadVisible / toolProcessVisible / articleFrameVisible / rawPatchHidden / workflowStepsHiddenInChat / rightSurfaceNotAutoOpened / rightSurfaceOpensOnClick / traceHasElectronRead` 全部为 `true`。这条证据确认用户面主链不再暴露 raw JSON / workflow step，不自动打开右侧，历史恢复后点击文章产物可打开 Article Editor。

2026-07-19 Codex 对齐复核：`thread/resume` 只做 Thread rejoin/history hydrate，不恢复 queued turn，也不写 workflow audit。只有 `workflow/respond` / `agentSession/action/respond` 的 typed response 显式携带 `metadata.workflowResume` 时，才向后台 `workflow-events.jsonl` 追加 `workflow.step.resuming` / `workflow.run.resuming`；response 原文不写入审计。

2026-07-19 signed release gate 已同步该边界：production GUI evidence 必须证明真实 `electron-ipc -> app_server_handle_json_lines -> agentSession/turn/start`、typed action response 或 `workflow/respond` 的 `metadata.workflowResume` 与匹配的 `workflow.step.resuming` / `workflow.run.resuming` audit 事件同时存在；只有 signed release、live Provider、Article Workspace 和 `workflow-events.jsonl` 路径不足以 ready。真实远程包 workflow response 上报该 metadata 的生产证据仍未完成。

2026-07-05 live Provider current-turn 已通过：App Server `local_folder` current-turn 使用 Agnes OpenAI-compatible `agnes-2.0-flash` 完成 host-managed generation，证据 `.lime/qc/gui-evidence/agent-apps/content-factory-current-turn-live-provider-2026-07-05T07-53-24-361Z.json` 显示 `liveProviderUsed=true`、`hostManagedGenerationStatus=completed`、段落级 partial `68` 个、workflow audit `16` 条。该证据只关闭“真实 Provider 能否驱动 host-managed generation”的缺口；远程 signed release、真实 GUI `cloud_release` 安装、signature verification 和 resume lifecycle 仍是下一刀。

2026-07-05 production GUI evidence collector 已补：`plugin:content-factory-production-gui-evidence` 通过真实 Electron CDP 读取 `pluginInstalled/list`、`agentSession/read`、`evidence/export` 和 trace，不安装插件、不跑 Provider、不写密钥。当前 local_folder 桌面 session 复跑按预期失败，缺 `sourceKind=cloud_release`、signature verified、目标 turn/start Electron IPC trace 和 workflowResume lifecycle；这条失败证据是 production 门槛的 guard，不是产品链路完成证据。

2026-07-05 21:13 追加真实 Electron/CDP live 写作复测：真实输入框发送 `@写文章` 后，`agentSession/turn/start` 经 `electron-ipc -> app_server_handle_json_lines` 进入 App Server，session `sess_c791014cba9e42caabe337db7b81467c`；用户面有自然引导和文章正文，未显示 raw JSON / `workspace-patch.json`。后端普通 session JSONL 写入 `3061` 行，workflow audit JSONL 写入 `16` 行。证据为 `.lime/qc/gui-evidence/agent-apps/content-factory-writing-cdp-WRITING_CDP_1783257215276.json`；production collector 复跑仍 `status=failed`，因为当前不是 signed `cloud_release` 且缺 resume lifecycle。该复测证明本地真实桌面用户链路已走普通 Agent turn 和 JSONL 审计，不改变 production gate 的 blocked 结论。

2026-07-05 production preflight 已补：`plugin:content-factory-production-preflight` 读取真实 `.lapp` 并通过 App Server current `pluginLocalPackage/inspect` 取得 release manifestHash，当前包事实为 `packageHash=sha256:89aec20e637713c668f8bc34c303256ac83806c5d2e75486e6453bd638ac3f8c`、`manifestHash=sha256:c1d3aa37d4b2f6c3c4a006525a1bba4b4ee407f61fe9cff8192704b48a209248`。该 preflight 明确 blocked 于签名文件、可信根、production catalog、bootstrap 和 fetchCloud evidence，作为发布准备清单，不替代 production GUI evidence。

2026-07-05 21:42 继续复核：Studio 发布入口已从旧 inspect 方法切到 current `pluginLocalPackage/inspect`，并用 dry-run `releaseReadiness` 将缺 packageUrl、app.signature、tenantId 和 developer token 前置暴露；证据为 `.lime/qc/gui-evidence/agent-apps/content-factory-studio-publish-dry-run-live-continue-2026-07-05.json`。最新 production preflight、bundle 和 readiness report 分别写入 `.lime/qc/gui-evidence/agent-apps/content-factory-production-preflight-studio-dry-run-continue-2026-07-05T13-42-15-968Z.json`、`.lime/qc/gui-evidence/agent-apps/content-factory-production-evidence-bundle-studio-dry-run-continue-2026-07-05/`、`.lime/qc/gui-evidence/agent-apps/content-factory-production-readiness-report-studio-dry-run-continue-2026-07-05.json`，均保持 blocked。该结果把剩余问题限定为 signed remote release 输入和真实 GUI `cloud_release` 安装运行：插件重构层不再增加新的右侧展示、旧执行器 fast path 或 hard-coded 写作模板来绕过 production gate。

2026-07-05 21:46 readiness report 已链接 Studio dry-run：`.lime/qc/gui-evidence/agent-apps/content-factory-production-readiness-report-studio-dry-run-linked-2026-07-05.json` 证明发布侧 dry-run 与 Lime preflight 读取同一 packageHash / manifestHash，且 drift 为空；blocked 原因只剩真实 production 输入与 GUI `cloud_release` lifecycle。插件重构口径因此保持收窄：不要在内容工厂插件或宿主 UI 中新增假详情页、弹窗、右侧自动展示或写死模板来补 production evidence。

2026-07-05 22:23 继续收口 production 输入事实：`.lime/qc/gui-evidence/agent-apps/content-factory-production-readiness-report-env-missing-continue-2026-07-05.json` 证明在签名私钥、packageUrl、tenantId、token 和 production API base 均未配置时，当前工具链只能产出 blocked readiness。插件层下一步不应新增 mock execution、硬编码模板或本地 fixture 来“补齐”这个缺口；真正的下一刀是接入真实 signed remote release 输入并跑 GUI `cloud_release` 证据。

2026-07-05 22:37 env 别名收口：Studio CLI 与 Lime preflight 已统一 tenant/API base/packageUrl env 别名，新 report `.lime/qc/gui-evidence/agent-apps/content-factory-production-readiness-report-env-alias-continue-2026-07-05.json` 仍 blocked。插件重构层继续不新增旁路；后续要么提供真实 production 输入跑通，要么保持 fail-closed。

2026-07-05 22:53 production readiness pipeline 已补：`.lime/qc/gui-evidence/agent-apps/content-factory-production-readiness-pipeline-env-alias-continue-2026-07-05/content-factory-production-readiness-pipeline.json` 统一记录 preflight、Studio `publish --dry-run`、evidence bundle 和 readiness report；pipeline 不传 `--publish`，不签名、不上传、不安装、不调用 Provider 或 production publish API。随后补 `--fetch-cloud-output` 自动落盘，真实 production catalog 到位后 pipeline 可把 App Server `pluginPackage/fetchCloud` 结果直接送入 bundle，减少手工 JSON 搬运。最新无 catalog 复跑 `.lime/qc/gui-evidence/agent-apps/content-factory-production-readiness-pipeline-fetchcloud-output-continue-2026-07-05/content-factory-production-readiness-pipeline.json` 仍 `status=blocked`，且 packageHash / manifestHash 在 preflight 与 Studio dry-run 间对齐。插件重构结论不变：内容工厂插件只承接 workflow contract 和 article artifact，不新增 mock execution、hard-coded 写作模板、弹窗详情或右侧自动展示来绕过 production gate。

2026-07-05 23:16 production readiness phase plan 已补：readiness report / pipeline 现在输出 `blockerPlan.nextPhase`，把缺口分成签名 proof / trust、Studio 发布输入、catalog/bootstrap、fetchCloud 和真实 desktop `cloud_release` E2E；无 production catalog 时 pipeline 只记录 `fetchCloudFromCatalog.skippedReason=catalog_missing`，不再让 preflight 命令失败。最新复跑 `.lime/qc/gui-evidence/agent-apps/content-factory-production-readiness-phase-plan-rerun-2026-07-05-2026-07-05T15-16-14-303Z/content-factory-production-readiness-pipeline.json` 仍 blocked，`nextPhase=release_signing_and_trust`。插件层继续不新增旁路：下一刀必须是真实签名和可信根，再进入 remote release / GUI 复测。

2026-07-05 23:36 production 签名验真已补：preflight 不再停留在字段完整性检查，而是按外部发布工具 canonical payload v2 重建 payload 并用 trust root `publicKey` 验证 detached signature；signed gate 同步要求 preflight 已验证、payloadHash 匹配、bootstrap 匹配 trust root 带 `publicKey`。最新只读复跑 `.lime/qc/gui-evidence/agent-apps/content-factory-production-readiness-signature-verify-final2-2026-07-05-2026-07-05T15-48-09-615Z/content-factory-production-readiness-pipeline.json` 仍 blocked，`nextPhase=release_signing_and_trust`。插件层仍不新增弹窗详情、右侧自动展示、mock execution 或 hard-coded 写作模板来绕过 production gate。

2026-07-05 23:56 production catalog sourceKind 已收紧并复跑：signed gate / readiness pipeline 只接受 `cloud_release`，`remote` 不再是 production signed catalog ready。pipeline 也会默认消费内容工厂目录里的 `app.signature.yaml` 和 `plugin-signature-trust-root.json`，避免 preflight 与 Studio dry-run 使用两套签名输入。最新只读证据 `.lime/qc/gui-evidence/agent-apps/content-factory-production-readiness-signature-defaults-2026-07-05-2026-07-05T16-02-26-791Z/content-factory-production-readiness-pipeline.json` 仍 blocked，`fetchCloudFromCatalog.skippedReason=catalog_missing`，`nextPhase=release_signing_and_trust`。插件层继续只承接 workflow contract / article artifact / JSONL audit，不通过假详情页、弹窗、右侧自动展示、mock execution 或写死模板补 production evidence。

2026-07-06 pipeline ready path 回归已补：完整 production evidence 组合在 test-only 层可以让 pipeline / readiness report / signed gate 变为 `ready`，但该绿色路径要求 GUI `cloud_release` signature verified、Electron IPC trace 和 workflow resume lifecycle 全部存在。插件重构口径不变：这不是 mock execution 或本地 fixture 的通行证，真实产品仍必须从 production catalog 安装 signed `cloud_release` 后跑 CDP 证据。

2026-07-06 packageUrl operator 入口已补：readiness pipeline 可以显式接收 `--package-url <https-url>`，并以脱敏环境变量传给 preflight / Studio dry-run。最新只读复跑 `.lime/qc/gui-evidence/agent-apps/content-factory-production-readiness-operator-inputs-2026-07-06-2026-07-05T16-15-09-136Z/content-factory-production-readiness-pipeline.json` 仍 blocked。插件层仍不保存或展示原始 package URL，也不把 packageUrl 单独视为 ready；它只是 signed `cloud_release` 发布输入之一。

2026-07-06 auth operator 入口已补：readiness pipeline 支持 `--tenant-id / --api-base / --studio-token-env`，token 只从环境变量读取并传给子进程。插件层不存 token、不展示 token、不把 auth 输入当业务结果；它只服务 signed `cloud_release` 发布审计。

2026-07-06 operatorReadiness 已补：readiness pipeline 会把 operator 输入状态写成非敏感 `operatorReadiness`，只记录 configured 布尔值、env 名和 evidence 文件存在状态，并打印脱敏 `operatorCommand`。插件层仍不能通过这些 marker 绕过 signed release gate；真实 ready 仍必须来自 production catalog/bootstrap、fetchCloud verified 和真实 GUI `cloud_release` 证据。

2026-07-06 operator missingKeys 已补：`operatorReadiness` 增加 `ready / missingKeys`，当前真实复跑显示缺签名、releaseId/publicKeyId、trust root、packageUrl、租户/API/token、catalog/bootstrap、fetchCloud 和 GUI evidence。插件层不生成假签名、不上传包、不把 local_folder 伪装为 cloud_release；这些 missingKeys 只是下一步 operator 输入清单。

2026-07-06 operator missingActions 已补：readiness evidence 会把 missingKeys 映射到安全 actions，例如显式运行 pipeline `--generate-signature-proof` 让同轮 Studio dry-run hash 驱动外部真实签名工具、提供 production HTTPS packageUrl、走 current bulk publish 后读取 catalog/bootstrap、再用 App Server fetchCloud 和真实 Electron CDP 采集证据。插件层仍不新增旧执行器 fast path、mock release 或手写 ready JSON。

2026-07-06 signing command hint 已补：readiness evidence 会用当前 preflight 的真实 packageHash / manifestHash 生成签名命令提示，但 packageUrl、releaseId 和公钥 ID 都保持占位符，私钥只通过 `--private-key-env PLUGIN_SIGNING_PRIVATE_KEY_PEM` 引用本地环境变量。插件层仍不执行签名、不写 trust root、不上传包；真实签名与 catalog/bootstrap 仍由 production operator 链路完成。

2026-07-06 signing command CLI 输出已补：readiness pipeline 在终端直接打印同一条签名命令提示，避免 operator 手动打开 JSON 复制 hash。插件层仍只提供审计辅助，不承接签名、发布或安装。

2026-07-06 operator command 口径收口：`operatorCommand` 与 `blockerPlan.nextPhase.commandHint` 现在都推荐 `--generate-signature-proof --signing-private-key-env PLUGIN_SIGNING_PRIVATE_KEY_PEM`，不再把 `--app-signature / --trust-root` 作为 operator 第一入口，也不再输出 `<private-key>` / `<token>` 这种容易被误粘贴进 shell history 的 secret 占位。默认 pipeline 保持只读；只有显式签名 proof 生成且具备真实 HTTPS packageUrl、releaseId、publicKeyId 和本地 signing private key env/file 时，才会生成签名文件。

2026-07-06 releaseId / publicKeyId operator 输入已补：readiness pipeline 接受 `--release-id / --public-key-id`，并把这两个非敏感签名上下文写入 `operatorReadiness.inputs`；缺真实签名文件时，未配置也会进入 `missingKeys / missingActions`。插件层仍不硬编码 release id、不生成 mock trust root、不把这些字段当 production ready。

2026-07-06 releaseId 绑定门禁已补：外部 `content-factory-app` 签名工具、Lime preflight 和 signed release gate 现在共同拒绝未绑定具体 releaseId 的 `signatureRef`。插件层不再允许 `sigstore:content-factory-app@2.2.2` 这种只绑定版本的签名示例进入 production readiness；真实 catalog 必须带 `releaseId`，且 `signatureRef` 以 `:<releaseId>` 结尾。

2026-07-06 Studio-first pipeline 已补：readiness pipeline 先跑 Studio dry-run 刷新 `.lapp`，再跑 preflight，避免同轮 preflight 读取旧 dist package。最新只读证据 `.lime/qc/gui-evidence/agent-apps/content-factory-production-readiness-studio-first-2026-07-06-2026-07-05T17-07-47-825Z/content-factory-production-readiness-pipeline.json` 仍 blocked，但 Studio/preflight packageHash drift 已消失。插件层仍只提供真实 signed release 的审计链，不新增 mock release 或手写 ready JSON。

2026-07-06 optional signing proof generation 已补：readiness pipeline 新增显式 `--generate-signature-proof`，用于 operator 在本地具备真实 signing key、HTTPS packageUrl、releaseId 和 publicKeyId 时，复用外部 `content-factory-app/scripts/sign-release.mjs` 生成 `app.signature.yaml` / trust root。默认不签名；缺任一输入时 `production_signature_generation_inputs_missing` fail-closed；签名阶段 evidence 只写入脱敏状态、missingKeys、路径存在性和字节数，不写入 URL、私钥、公钥或签名值。该能力只减少手工签名串联错误，不新增旧执行器、mock release、硬编码模板、弹窗详情或右侧自动展示。

2026-07-06 preflight signingCommand hygiene 已补：preflight 输出层也不再保留 `PLUGIN_SIGNING_PRIVATE_KEY_PEM=$PRIVATE_KEY_PEM` 示例，统一指向 readiness pipeline 的 `--generate-signature-proof --signing-private-key-env PLUGIN_SIGNING_PRIVATE_KEY_PEM`；CLI help 也有回归守卫，防止重新提示粘贴 key/token 值，并明确 preflight 只通过 current `pluginLocalPackage/inspect` 做本地包事实和缺口检查，不签名、不上传、不安装、不调用 Provider、不写 passing `cloud_release` evidence。这只是 operator 审计提示收口，不改变插件运行形态；真实 ready 仍必须由 signed catalog/bootstrap、App Server fetchCloud verified 和真实 Electron/CDP `cloud_release` GUI evidence 同时关闭。

2026-07-06 signing proof args 审计性已补：readiness pipeline evidence 会保留 `--private-key-env <ENV_NAME>` / `--studio-token-env <ENV_NAME>` 这类环境变量名，继续隐藏真实 URL、tenant、token、key file path 和私钥值。插件层不读取这些值，也不因此获得发布能力；它只让 production operator 之后能审计“使用了哪个本地 env 名”。

2026-07-06 release evidence sourceKind 推断已收紧：`content-factory-production-release-evidence` 从 LimeCore marketplace 抓取 catalog 时不再因为存在 HTTPS `packageUrl` 就生成 `identity.sourceKind=cloud_release`；只有 package/source 明确声明 `cloud_release` 才能关闭 catalog sourceKind 要求。缺显式 sourceKind 时 summary 保持 `status=blocked` 并输出 `catalogSourceKindCloudRelease`，避免远程包 URL、fixture 或旧 marketplace 字段被误当 production signed release。最新只读 pipeline `.lime/qc/gui-evidence/agent-apps/content-factory-production-release-sourcekind-audit-2026-07-06-2026-07-06T01-13-12-511Z/content-factory-production-readiness-pipeline.json` 仍 `status=blocked`、missing codes `17` 个。

2026-07-06 fetchCloud evidence 字段门禁已收紧：signed release gate 现在拒绝只写 `matched=true / verified / ready` 的 fetchCloud JSON，必须同时包含本次拉取的 `packageHash`、`manifestHash`、非 localhost HTTPS `packageUrl/sourceUri`、`signatureRef` 和 `signatureProof`，并继续与 catalog / preflight 比对。`content-factory-production-fetch-cloud-evidence` 会把 catalog proof 的非密钥审计字段写入本地 evidence；gate/report summary 不复制原始 package URL 或 detached signature 原文。该改动只提高 production 证据可信度，不新增 mock release 或旧执行器 fast path。

2026-07-06 GUI evidence provenance 门禁已收紧：signed release gate 不再接受只有 `status=passed`、`liveProviderUsed=true` 和 workflow JSONL 路径的手写 GUI JSON；必须带 production GUI collector schema、真实 Electron CDP attached/usedRealElectron、matched turn-start trace、current App Server `turn/start + read + evidence/export` method trace、workflow JSONL event count 和 `generatedArticleMarkerClean=true`。collector 同步写入 workflow JSONL event count/type 摘要。该改动不改变用户 UI，只防止 production GUI evidence 被 fixture 或手写 ready JSON 伪造。

2026-07-06 workflow audit export 门禁已收紧：production GUI evidence 不能只证明 workflow JSONL 文件存在，还必须证明同一 session 经 current App Server `evidence/export` 产出 metadata-only `workflow_audit` 摘要。signed gate 要求 `status=exported`、`source=workflow-events.jsonl`、`eventCount>0`、`metadataOnly=true`、`rawContentIncluded=false`、`redactionPolicy=workflow_audit_metadata_only` 和 redaction 覆盖事件数。该改动继续保持 workflow facts 不进右侧 UI，只让审计链具备可导出的非原文摘要。

2026-07-06 GUI installed release identity 门禁已收紧：production GUI evidence 的 `installedState` 不能只写 `signature verified`、`packageHashMatched=true` 或 `manifestHashMatched=true`；必须同时携带 `appVersion / packageHash / manifestHash / releaseId / signatureRef`，其中 hash 必须是 `sha256:<64 hex>`，并逐项与 production catalog、preflight 和 fetchCloud evidence 一致。缺字段、hash 非法、releaseId / signatureRef 漂移或 fixture / localhost marker 都会 blocked。该改动只提高 GUI evidence 与同一轮 release 的绑定强度，不新增 UI、旧执行器 fast path、mock release 或手写 ready JSON。
