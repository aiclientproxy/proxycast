# Lime 测试分层路线图

## 专题路线图

- [Clawstream 全链路护栏与旧实现清理](./clawstream/README.md)：绑定 current 架构与 Query Loop 的 Thread / Turn / Item 主线，覆盖 Claw 输入到输出的 streaming fixture、projection、GUI evidence 与旧 fallback 清理。

## 背景

当前 `npm test` 同时承载纯单元、React/jsdom 组件、DevBridge/Tauri 契约、脚本集成和少量 live-gated 测试。按当前工作区实测，前端完整 Vitest 分批跑完约 `306.95s`，Rust 后端 `cargo test --manifest-path "lime-rs/Cargo.toml"` 约 `11.00s`。这说明 Lime 的主要反馈瓶颈在前端测试分层，而不是 Rust 后端。

本路线图目标是把本地 TDD 反馈环从“全量前端 Vitest”收敛到“快速纯单元 + 定向相关测试”，同时保留 `verify:local`、`test:contracts`、`verify:gui-smoke` 对交付风险的覆盖。

## 原则

1. **单元测试优先测业务逻辑边界** - 前端复杂逻辑应优先沉淀到 View Model / projection / presentation / selector / command planner 等纯函数边界，再用纯单元测试覆盖。
2. **组件测试只测必要契约** - React Testing Library / jsdom 测试只保留关键渲染、可访问性、事件接线和回归点，不承担大段业务状态机验证。
3. **核心用户流交给 GUI smoke / E2E** - 用户必须真实完成的主路径，用 `verify:gui-smoke`、Playwright 或已有产品 E2E 验证，不把完整流程塞进单个组件测试。
4. **不引入重 MVVM 框架** - Lime 继续使用现有 React + projection/helper 模式；View Model 是测试边界，不是新的运行时框架。
5. **先分层入口，再逐步迁移** - 第一阶段不批量重命名 800+ 测试文件，先用 runner 分类和显式脚本形成反馈环；后续再按热点迁移。

## 分层定义

| 层级            | 命令                        | 覆盖范围                                                                                    | 本地 TDD 默认         |
| --------------- | --------------------------- | ------------------------------------------------------------------------------------------- | --------------------- |
| Unit            | `npm run test:unit`         | 纯函数、parser、formatter、projection、presentation、selector、View Model 状态转换          | 是                    |
| Component       | `npm run test:component`    | React/jsdom 组件与 hook 渲染、事件接线、关键 UI 断言                                        | 按 UI 改动定向跑      |
| Contract        | `npm run test:contract`     | `safeInvoke`、DevBridge、Tauri mock、command catalog、schema 契约                           | 按命令/桥接改动跑     |
| Integration     | `npm run test:integration`  | 文件系统、子进程、本地 fixture server、多模块脚本流程                                       | CI 或本地按需         |
| E2E             | `npm run test:e2e`          | Vitest 内显式 E2E / smoke / live-gated 测试；真实产品主路径仍以 GUI smoke / Playwright 为准 | 默认不进 TDD          |
| Frontend All    | `npm run test:frontend:all` | 现有前端 Vitest 全量兼容入口                                                                | 交付前/CI             |
| Layer Stats     | `npm run test:layers:stats` | 按同一分类事实源输出分层统计、默认可运行数和 live-gated 数                                  | 统计 / 治理           |
| Rust            | `npm run test:rust`         | Rust workspace 测试                                                                         | Rust 改动定向后再全量 |
| GUI Smoke / E2E | `npm run verify:gui-smoke`  | Tauri 壳、DevBridge、Workspace、主产品路径                                                  | GUI 主路径改动/交付前 |

## 治理后统计

最后统计时间：2026-06-03。

| 范围                  | 命令                                                                            | 文件 / 用例                                                | 实测耗时          | 备注                                                                                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 前端分层统计          | `npm run test:layers:stats`                                                     | Vitest 总 `1012` 个文件；默认可运行 `1011`；live-gated `1` | 统计脚本级        | 同一事实源来自 `scripts/lib/vitest-layer-classifier.mjs`；unit `485`、component `397`，component VM 迁移候选当前发布基线为 `22` 个；后续继续降回 `12` / `8` 或更低                              |
| 前端 Unit             | `npm run test:unit`                                                             | 最近有效默认复测 `483` 个文件，`2851` 个 case              | `161.62s`（通过） | 本地 / AI TDD 默认第一轮信号保持绿色；unit runner 默认 Node + threads pool，`--single-fork` / `--pool=forks` 可回退；高负载下的 `293.89s` / `366.78s` 只作为环境噪声记录，不作为稳定基准 |
| 前端全量 Vitest       | `npm run test:frontend:all`                                                     | 默认可运行 `891` 个文件；最近完整跑完 `63/63` 批次         | 约 `306.95s`      | 交付前 / CI 全量入口；runner 不输出聚合 case 总数                                                                                                                                        |
| 前端 Integration 定向 | `npm run test:integration -- src/lib/layered-design/export.integration.test.ts` | `1` 个文件，`16` 个 case                                   | `29.83s`（通过）  | `layered-design` ZIP / PSD 导出属于重二进制打包验证，已移出 unit                                                                                                                         |
| Rust Unit             | `npm run test:rust:unit`                                                        | `1551` passed，`1` ignored                                 | `61.76s` wall     | 后端 TDD 默认第一轮信号；另有 `make tdd-rust-filter FILTER=...` 做单测试名定向回路                                                                                                       |

### 2026-06-03 速度优先恢复点

如果后续 Agent 从本文件恢复本任务，先处理速度，再处理候选数量治理。当前有效证据如下：

- 现有 `npm run test:unit` 绿色；调整前由于 `scripts/run-vitest-layer.mjs` 强制 `--poolOptions.forks.singleFork`，486 个 unit 文件全层单 fork 跑完约 `213.05s`。
- 用同一批 unit 文件直接调用 Vitest、去掉 `singleFork` 后，486 files / 2849 tests 全部通过，耗时 `175.46s`；这说明优先去掉 unit 默认单 fork 预计可直接节省约 `37s`。
- 已完成第一刀：`test:unit` 默认不再强制 `singleFork`，`--single-fork` 与 `LIME_VITEST_SINGLE_FORK=1` 可作为排查回退；真实入口复测 486 files / 2853 tests 全部通过，耗时 `183.71s`，相比 `213.05s` 快约 `29s`。
- 进一步加 `--environment node` 后，耗时降到 `137.44s`，且只有 `4` 个文件、`6` 条断言失败；这说明大多数 unit 已经能在 Node 环境运行，当前最大的速度收益来自把少数浏览器依赖 unit 隔离出去。
- 已完成第二刀：unit 默认追加 `--environment node`；以下 4 个 browser-dependent 测试已显式移到 component 层：
  - `src/lib/crashReporting.component.test.ts`
  - `src/lib/workspaceHealthTelemetry.component.test.ts`
  - `src/lib/utils/scheduleMinimumDelayIdleTask.component.test.ts`
  - `src/components/agent/chat/hooks/agentStreamSubmitDraft.component.test.ts`
- 真实入口复测 `npm run test:unit` 482 files / 2841 tests 全部通过，耗时 `138.63s`，已达到当前 `140s` 级目标；Vitest summary 中 `environment 241ms`，说明全层 jsdom 成本已从 unit 移除。
- 已完成第三刀：unit 默认追加 `--pool threads`，`--pool=forks`、`--single-fork`、`LIME_VITEST_SINGLE_FORK=1` 可回退。实测证据：
  - `/usr/bin/time -p npm run test:unit -- --pool=threads`：483 files / 2848 tests 通过，`real 115.59`。
  - `/usr/bin/time -p npm run test:unit`：483 files / 2851 tests 通过，`real 161.62`。
  - `/usr/bin/time -p npm run test:unit -- --pool=forks`：484 files / 2858 tests 通过，`real 223.28`；该轮已包含并行新增的 `toolNameFamily.unit.test.ts`。
  - `/usr/bin/time -p npm run test:unit`（默认 threads 后高负载复测）：484 files / 2858 tests 通过，`real 293.89`。
  - `/usr/bin/time -p npm run test:unit`（临时取消显式 pool 后同一高负载复测）：484 files / 2859 tests 通过，`real 366.78`。
  - 当时机器 `load averages: 95.67 82.56 71.54`，且存在两个 `tsc --noEmit` 和多个高 CPU GUI / browser 进程；因此这两轮只证明当前机器负载会严重放大 wall time，不应作为回退 threads 默认的稳定证据。
  - 结论：threads pool 已验证绿色，且同阶段快于显式 forks；当前保留 unit 默认 threads。后续速度基准必须在工作树和机器负载安静后复测。
- 已完成第四刀：新增 `src/test/fastCheckRuns.ts`，把本地 / AI TDD 的 fast-check 属性测试 runs 默认降到 `25`，CI 保持原始 `50/100` runs，`LIME_FAST_CHECK_RUNS=100` 可在本地强制满量；当前 unit 层 fast-check 文件已全部接入 helper，覆盖 `src/lib/artifact/streaming.test.ts`、`store.test.ts`、`registry.test.ts`、`parser.test.ts`、`src/lib/api/importExport.test.ts`、`src/lib/config/providers.test.ts`、`src/lib/utils/apiKeyMask.test.ts`、`src/components/artifact/ArtifactRenderer.test.ts`、`ArtifactToolbar.test.ts`、`src/components/api-key-provider/ApiKeyProviderSection.test.ts`、`ProviderSetting.test.ts`、`providerConfigUtils.test.ts`、`src/icons/providers/providers-icons.test.ts`。
  - `npm run test:unit -- src/test/fastCheckRuns.unit.test.ts src/lib/artifact/streaming.test.ts src/lib/artifact/store.test.ts src/lib/artifact/registry.test.ts src/lib/artifact/parser.test.ts src/lib/api/importExport.test.ts src/lib/utils/apiKeyMask.test.ts`：7 files / 63 tests 通过，Vitest duration `3.95s`。
  - `npm run test:unit -- src/components/api-key-provider/ApiKeyProviderSection.test.ts src/components/api-key-provider/ProviderSetting.test.ts src/components/api-key-provider/providerConfigUtils.test.ts src/components/artifact/ArtifactRenderer.test.ts src/components/artifact/ArtifactToolbar.test.ts src/icons/providers/providers-icons.test.ts src/lib/config/providers.test.ts`：7 files / 105 tests 通过，Vitest duration `11.66s`。
  - `CI=1 npm run test:unit -- src/lib/utils/apiKeyMask.test.ts`：1 file / 7 tests 通过，证明 CI 环境路径可执行；`fastCheckRuns.unit.test.ts` 机械锁住 CI 仍返回原始 runs。
- 已开始治理 `useWorkspaceSendActions.test.tsx` 候选：`src/components/agent/chat/workspace/workspaceModelSkillLaunchRequestContext.ts` 承载 model skill launch request context、session binding 和 metadata wrapper 的纯 current helper；`workspaceModelSkillLaunchRequestContext.unit.test.ts` 进入 unit 层，覆盖播报、素材、转写、文本转换、URL、排版、网页、PPT、表单和 session 绑定。`useWorkspaceSendActions.ts` 从约 5475 行降到约 5039 行；component 候选数暂未下降。

下一次重启优先级：

1. **先稳基准**：工作树和机器负载安静后复跑一次 `/usr/bin/time -p npm run test:unit`，确认默认 Node + threads + 本地 fast-check 降采样后的稳定耗时；如果仍明显慢于 `138.63s`，优先做 transform / collect 慢文件画像，而不是继续试错 pool 默认。
2. **再治理候选数**：当前发布基线剩余 22 个 component migration candidates 仍是下一优先级；优先继续 `useWorkspaceSendActions.test.tsx`，把已由 `workspaceModelSkillLaunchRequestContext.unit.test.ts` 覆盖的重复 metadata 挂载断言删除或拆成 focused component suites，先降回 `12` / `8`。
3. **最后看 Rust**：`npm run test:rust:unit` 当前约 `61.76s`，后端后续优化重点应是定向 crate / filter 回路，而不是先全量重构 Rust 测试。

前端 Vitest 当前分层：

| 层级        | 文件数 | 默认可运行 | Live-gated |
| ----------- | -----: | ---------: | ---------: |
| Unit        |  `485` |      `485` |        `0` |
| Component   |  `397` |      `397` |        `0` |
| Contract    |   `78` |       `78` |        `0` |
| Integration |   `51` |       `51` |        `0` |
| E2E         |    `1` |        `0` |        `1` |

## 前端 View Model 策略

### 适合抽到 View Model 的逻辑

- 从 runtime/session/thread/state 计算 UI 展示模型
- 从用户输入、选中项、Provider 能力推导按钮状态和提交参数
- 消息、工具调用、artifact、task preview 的分组、排序、去重、状态折叠
- 表单草稿、筛选、分页、空态、错误态、loading 态的状态转换
- GUI 事件映射为 command/action 的规划逻辑

### 不应放进 View Model 的内容

- DOM 测量、滚动、焦点、快捷键监听等真实浏览器行为
- Tauri / DevBridge / 文件系统 / 网络调用本身
- 纯视觉布局、CSS、动画细节
- 需要真实壳或真实用户流才能证明的行为

### 测试分配

- VM / projection / selector：大量纯单元测试，作为 TDD 默认反馈环。
- 组件：少量接线测试，证明 VM 输出被正确渲染、关键按钮事件会触发对应 action。
- E2E / smoke：覆盖“用户能完成任务”的主路径，不重复测试 VM 的所有分支。

## 阶段计划

### P0：建立测试分层入口

- 新增 Vitest 分层 runner，支持 `unit/component/contract/integration/e2e`。
- 保留 `npm test` 原语义，新增分层命令，不破坏现有 CI。
- 分类规则先基于文件名和静态特征，避免第一刀批量迁移测试文件。
- 完成标准：`npm run test:unit -- --list` 能快速列出纯单元候选，`npm run test:unit -- <file>` 能运行指定纯单元测试，`npm run test:layers:stats` 能输出治理后统计。

### P1：把 TDD 默认入口切到 Unit

状态：第一刀已完成。

- 在工程文档中明确：AI / 本地 TDD 默认先跑 `npm run test:unit` 和相关文件。
- `verify:local` 继续作为交付入口，不被 `test:unit` 替代。
- 完成标准：普通纯逻辑改动无需跑完整 `npm test` 就能得到第一轮信号。

### P2：热点组件 VM 化

状态：已开始，先从 `AgentChatPage` shell 路由抽取小型 View Model 作为模板。

优先迁移当前耗时和复杂度最高的前端测试：

1. `src/components/agent/chat/index.test.tsx`
2. `src/components/agent/chat/hooks/useAgentChat.test.tsx`
3. `src/components/agent/chat/workspace/*PreviewRuntime.test.tsx`
4. `src/components/agent/chat/components/HarnessStatusPanel.test.tsx`

迁移方式：

- 从大组件/大 hook 中抽出纯 projection、selector、command planner。
- 新增 `*.unit.test.ts` 覆盖状态转换和边界分支。
- 原 `*.test.tsx` 降为少量组件接线测试，或改名为 `*.component.test.tsx`。
- 核心用户路径交给 `verify:gui-smoke` 或 Playwright 续测。

完成标准：`src/components/agent/chat/index.test.tsx` 单文件耗时从约 `55s` 降到可接受范围，且对应 VM 单测进入 `test:unit`。

当前模板：

- `src/components/agent/chat/agentChatPageShellViewModel.ts`：承载 `new-task` 直达工作区意图判断、`claw` 强制切换和聊天面板展示策略。
- `src/components/agent/chat/agentChatPageShellViewModel.unit.test.ts`：覆盖文本、图片、站点技能、服务技能、输入能力、资料包、项目文件和浏览器协助等纯路由分支。
- `src/components/agent/chat/agentChatWorkspaceShellViewModel.ts`：承载工作区 shell 的展示消息、聊天面板、侧栏切换和工作区图片任务恢复策略。
- `src/components/agent/chat/agentChatWorkspaceShellViewModel.unit.test.ts`：覆盖空白首页、执行态、任务中心草稿压制、画布/主题工作台、紧凑工作台和非 `new-task` 入口等纯布局分支。
- `src/components/agent/chat/index.shell-routing.test.tsx`：保留 React 接线层测试，证明 VM 输出被传给 `AgentChatWorkspace`。
- `src/components/agent/chat/components/harnessStatusPanelViewModel.ts`：承载 Harness 状态面板中的状态标签、Badge 变体、工具友好标签、子任务摘要、文件审阅、diff summary、输出信号展示、URL / path 文本识别、runtime 任务展示、工具库存 label / filter / stats / sort、handoff / evidence / replay / analysis 制品展示、browser replay artifact 构造、LimeCore policy 展示和人工确认风险 / 描述等纯 presentation / selector 逻辑。
- `src/components/agent/chat/components/harnessStatusPanelViewModel.unit.test.ts`：覆盖子任务 runtime/session type、工具标签、子任务汇总、文件审阅、active write 描述、diff 状态 / summary、输出信号状态 / 路径 / 卡片展示、URL 归一化 / 文本切分 / 文件路径识别 / 搜索输出识别、runtime phase / status / progress / task presentation、工具库存执行来源 / 策略标签 / 筛选统计 / runtime 工具排序、handoff / evidence / replay / analysis 制品 label / size / time、browser replay artifact metadata、LimeCore policy 引用 / 缺失输入 / 决策汇总、审核状态/风险/权限/限制和 replay 推广命令构造等纯展示分支。
- `src/components/agent/chat/components/HarnessStatusPanelPrimitives.tsx`：承载 Harness 面板通用展示 primitive，包括 diff mini panel、可交互文本、路径链接、可点击 badge、summary card 和库存统计卡片。
- `src/components/agent/chat/components/HarnessSearchOutputCards.tsx`：承载搜索输出单卡与批次卡，主面板只保留输出信号分组和打开详情的事件接线。
- `src/components/agent/chat/components/HarnessEvidenceSummarySections.tsx`：承载 Browser Assist replay 索引摘要和 LimeCore policy 索引摘要，主面板只保留证据包导出 / 预览状态。
- `src/components/agent/chat/components/harnessPanelText.ts`：承载 Harness 面板 generated key 的 `agent` namespace 文案 fallback helper，避免拆分组件复制 i18n fallback 逻辑。
- `src/components/agent/chat/hooks/agentChatAutoTitleViewModel.ts`：承载 `useAgentChat` 自动标题中的占位标题判断、assistant 预览标题判断、用户文本存在性判断、标题生成上下文构造，以及生成标题回填 topic 的纯 reducer。
- `src/components/agent/chat/hooks/agentChatAutoTitleViewModel.unit.test.ts`：覆盖自动标题占位、预览派生标题、是否触发标题生成、用户文本过滤、生成上下文 `1000` 字符裁剪，以及生成标题回填命中 / 未命中 / 无变化复用。
- `src/components/agent/chat/utils/submitOpRuntimeCompaction.ts`：承载 turn submit 前的 provider/model、execution strategy、thinking、web_search、access mode 与 Harness metadata 去重事实源，避免通过重 hook 测试间接验证纯配置裁剪。
- `src/components/agent/chat/utils/submitOpRuntimeCompaction.test.ts`：覆盖 runtime / synced preference 已承接时裁掉重复字段、未同步时保留显式变更、fast response / image routing 特例、team selection 与 access mode metadata 裁剪。
- `src/components/agent/chat/hooks/agentSessionTopicViewModel.ts`：承载 `useAgentSession` 中 topic/session detail 映射、runtime thread 状态判定、topic upsert 排序、新建 session 草稿插入、远端校验 session 补入、execution strategy 写回、live snapshot 写回 reducer、transient messages/turns/items tail selector，以及 restore candidate 工作区隔离清洗计划。
- `src/components/agent/chat/hooks/agentSessionTopicViewModel.unit.test.ts`：覆盖自动恢复运行态、queued/waiting/failed 状态映射、排队预览、topic 本地 pin/unread/tag 保留、新建 session 草稿插入 / 去重、远端校验 session 补入 / 已存在复用、execution strategy 写回、live snapshot 写回 / 无变化复用、transient 历史窗口裁剪，以及 restore candidate 空值/辅助会话/旧默认工作区/跨工作区/合法映射清洗。
- `src/components/agent/chat/hooks/agentSessionRestoreViewModel.ts`：承载 `useAgentSession` 中工作区切换 / 恢复时从 transient storage 与 cached snapshot 推导首屏 session snapshot、timeline、历史窗口和缓存快照打点上下文的纯状态逻辑。
- `src/components/agent/chat/hooks/agentSessionRestoreViewModel.unit.test.ts`：覆盖直接使用 cached snapshot、合并 cached/transient messages、timeline 回退、current turn 优先级、scoped snapshot 规范化，以及 cached topic snapshot 的历史窗口和 metric context 派生。
- `src/components/agent/chat/hooks/agentSessionState.ts`：承载 session snapshot / hydration 状态转换、detail hydration 策略、restorable topic 选择，以及当前会话缺失于 topics 时清空 / 跳过 / 远程校验的纯决策。
- `src/components/agent/chat/hooks/agentSessionState.test.ts`：覆盖空会话快照、恢复目标选择、detail hydration 延后策略、同会话 hydration 合并，以及 missing session from topics 的 inactive / detached / auxiliary / remote verify 分支。
- `src/components/agent/chat/hooks/sessionSwitchSnapshotController.ts`：承载 topic switch 中 cached snapshot 加载 / 应用 / 刷新策略计划、defer hydration 状态计划、pending shell 状态计划、切换开始状态重置计划、切换指标上下文，以及 in-flight switch 复用的纯决策。
- `src/components/agent/chat/hooks/sessionSwitchSnapshotController.test.ts`：覆盖 cached snapshot 命中、pending shell、指标上下文、缓存读取 / 应用 / 立即刷新策略、defer hydration 直接/延迟加载模式、pending shell 空会话壳策略、切换开始状态计划，以及普通同 topic 切换可复用 / 强刷 / 自动恢复 / detached / session start hooks 禁止复用分支。
- `src/components/agent/chat/hooks/sessionFinalizeController.ts`：承载会话 detail finalize 阶段的 workspace restore 拒绝、执行策略 fallback / override，以及成功 finalize 后 restore/hydration 状态收尾计划。
- `src/components/agent/chat/hooks/sessionFinalizeController.test.ts`：覆盖 known workspace 解析、跨 workspace 拒绝上下文、shadow execution strategy fallback、最终 execution strategy override，以及 finalize 成功后的状态收尾计划。
- `src/components/agent/chat/hooks/sessionMetadataSyncController.ts`：承载 finalize 成功路径中的 metadata sync 输入选择、access mode / provider / execution strategy patch 规划、本地状态应用计划、metadata sync 成功应用计划、topic execution strategy 回填 reducer、switch success 指标上下文和 metadata sync 执行 fallback。
- `src/components/agent/chat/hooks/sessionMetadataSyncController.test.ts`：覆盖 runtime provider preference 优先级、session storage preference fallback、runtime metadata 无回填、session storage / workspace default patch、finalize 本地状态应用计划、metadata sync 成功应用计划、topic execution strategy 回填 reducer、switch success metric 和 metadata sync runtime fallback。
- `src/components/agent/chat/hooks/sessionPostFinalizePersistenceController.ts`：承载 finalize 成功后的 workspace 持久化、topic workspace 回填、provider preference 应用、副作用应用入口计划和 topic workspace 回填 reducer。
- `src/components/agent/chat/hooks/sessionPostFinalizePersistenceController.test.ts`：覆盖 topic workspace 解析顺序、持久化 workspace 解析、post finalize persistence plan、副作用 apply plan 和 runtime workspace 回填 reducer。
- `src/components/agent/chat/hooks/agentChatActionState.ts`：承载 action_required 写入 assistant 消息、确认提交后在 pending action / submitted in-flight / message actionRequests / contentParts / runtime status 之间同步状态、fallback ask 回答排队和真实 request 匹配，以及 replay request 结果到 ActionRequired 的纯 reducer / planner / mapper。
- `src/components/agent/chat/hooks/agentChatActionState.unit.test.ts`：覆盖 ask/elicitation 提交后保留 submitted 面板、tool confirmation 确认后移除请求、未命中 requestId 不污染消息、submitted in-flight upsert / 清理、fallback ask 排队、同 assistant 消息真实请求匹配、queued 状态同步，以及 replay request questions/options/scope 映射。
- `src/components/agent/chat/workspace/generalWorkbenchHelpers.ts`：承载通用工作台运行中 workflow steps 的工具调用到业务步骤标题 / 状态投影。
- `src/components/agent/chat/workspace/generalWorkbenchWorkflowSteps.unit.test.ts`：覆盖写文件、封面图、搜索、浏览器导航、点击、页面分析和媒体命令等工具调用到 workflow steps 的纯投影分支，组件层只保留 `workflowSteps` 传给 `Inputbar` 的接线断言。
- `src/components/agent/chat/utils/taskCenterTabs.ts`：承载任务中心标签排序 / 可见性 / fallback restore planner / detached 会话判断、route sync / fallback reconcile、topic 切换参数、失败回滚和关闭标签 fallback plan。
- `src/components/agent/chat/utils/taskCenterTabs.test.ts`：覆盖默认标签、workspace tab map、fallback topic / fallback restore planner、detached 隐藏、预览焦点、切换中状态、初始会话 detached / waiting / workspace_error 恢复参数、active topic 重开跳过、topic 切换选项、失败回滚和关闭标签 plan 等纯 selector 分支。
- `src/components/agent/chat/workspace/taskCenterDraftTabs.ts`：承载任务中心草稿标签构造、upsert/remove、running/failed 标记、active draft 解析、warmup 条件和关闭草稿 fallback plan。
- `src/components/agent/chat/workspace/taskCenterDraftTabs.unit.test.ts`：覆盖草稿标签去重插入、运行态/失败态标记、active draft 解析、预热条件和关闭草稿后的 fallback 选择。
- `src/components/agent/chat/utils/workflowInputState.ts`：承载 Inputbar 工作流快捷动作、队列、进度、摘要和生成面板显示状态的纯 builder，`useWorkflowInputState` 只保留 React 薄包装。
- `src/components/agent/chat/utils/workflowInputState.test.ts`：去掉 `createRoot` / `act` 挂载，直接覆盖非工作区场景、等待 gate、开放步骤队列 / 进度和发送态生成面板等纯状态分支。

后续迁移要求：

- 每次从重组件测试中抽出一个 VM / projection / selector，都要补一个 `*.unit.test.ts`。
- 原 `*.test.tsx` 只保留关键渲染和事件接线，不重复覆盖 VM 的所有分支。
- 若 unit 层测试依赖真实 `setTimeout`、idle callback、DOM 或浏览器全局，应优先改成可控调度器或移入 component / integration 层。

### P3：命名和 CI 收敛

状态：CI 分层口径已落地，PR 保持快速反馈，`main` push / 手动触发保留全量质量面。

- 新增或迁移测试时使用后缀：
  - `*.unit.test.ts`
  - `*.component.test.tsx`
  - `*.contract.test.ts`
  - `*.integration.test.ts`
  - `*.e2e.test.ts`
  - `*.live.test.ts`
- `*.live.test.ts` 归入 E2E 层，但默认受 live Provider gate 排除；必须显式设置 `LIME_ALLOW_LIVE_PROVIDER_SMOKE=1` / `LIME_REAL_API_TEST=1` 才会进入默认可运行集合。
- 显式后缀不能降低风险层级；`*.unit.test.*` 如果触碰 React/jsdom、DevBridge/Tauri、文件系统、网络或 Playwright，分类器会自动提升到 component / contract / integration / e2e，避免非纯单元进入本地 TDD 默认入口。
- PR 前端快速门禁跑 `lint`、`typecheck`、`test:unit`、`test:contract`，不再把全量 Vitest 作为默认 PR 前端反馈环。
- `main` push / 手动触发继续跑 `lint`、`typecheck`、`test:frontend:all`、`test:rust`、`verify:gui-smoke`。
- `Bridge & Contracts` 仍按 changed-path 单独跑 `test:bridge` 与 `test:contracts`；这里的 `test:contract` 是 Vitest contract layer，`test:contracts` 是命令 / 治理契约聚合入口。

## 风险与边界

- `test:unit` 不是交付证明，只是 TDD 快速反馈。
- GUI 产品改动仍必须按 `internal/aiprompts/quality-workflow.md` 补 GUI smoke / E2E。
- 迁移测试时不能为了追求纯单元而删除真实回归覆盖；应先把覆盖迁到 VM 或 E2E 后再删重组件断言。
- live Provider 测试继续保持显式 opt-in，不进入默认单元测试。

## 当前状态

- 测试分层入口已落地；分类事实源为 `scripts/lib/vitest-layer-classifier.mjs`。
- 当前开发遵循 related-first：先跑受影响 unit/component/contract，再按风险扩大。
- Agent、App Server、Electron 与 GUI 的证据等级以 `internal/aiprompts/quality-workflow.md` 为准。
- 历史逐刀日志不再保留在 active tree；需要追溯时使用 Git history。
- 下一步只维护未完成 blocker、分层规则和最新可复现验证，不追加按文件拆分的流水账。
