# Codex App GUI 对齐执行计划

> status: active / G1-in-progress
> owner: agent-ui + app-server
> started: 2026-07-18
> target: Codex App GUI interaction model
> backend reference: `/Users/coso/Documents/dev/rust/codex/codex-rs`
> architecture baseline: `internal/research/refactor/v2/04-target/architecture.md`
> verification baseline: `internal/research/refactor/v2/11-gui/electron-gates.md`

## 1. 主目标

把 Lime 的 Agent 主界面从“营销首页 + 聊天消息 + 多套诊断/工作台入口”收敛为以当前 Thread 为中心的桌面工作区：用户持续知道正在处理哪个任务、使用哪个本地环境、当前 Turn 到了哪一步、是否需要响应，以及结果和变更在哪里。

固定产品链继续为：

```text
Electron Desktop Host
  -> App Server JSON-RPC
  -> RuntimeCore / agent-runtime
  -> Thread / Turn / Item projection
  -> Codex App-style React GUI
```

Codex App 只作为 GUI 信息架构和交互层级参考；后端 Thread/Turn/Item、审批、队列、恢复、环境和工具生命周期直接对齐 `codex-rs`。不复制 TUI cell、ANSI、终端快捷键或 CLI onboarding。

## 2. 当前阶段与下一刀

- 当前阶段：v2 runtime/projection 骨架完成后的 GUI 差距审计。
- 已完成：源码结构、三张参考截图、浏览器镜像桌面视口和 Codex App Server/TUI 能力对照。
- 下一刀：`G1 Thread workspace shell`，先收敛 active Thread 的桌面壳，不新增协议，不重写 runtime。
- 当前写集：`src/hooks/useAppShellLayout.ts`、`src/components/app-sidebar/useAppSidebarConversationActions.ts`、`src/components/agent/chat/AgentChatWorkspace.tsx`、`src/components/agent/chat/workspace/ThreadWorkspaceHeader.tsx`、`src/components/agent/chat/workspace/threadWorkspaceHeaderViewModel.ts`、`src/components/agent/chat/workspace/WorkspaceConversationScene.tsx`、`src/components/agent/chat/workspace/WorkspaceMainArea.tsx`、Task Center tab projection/接线、五语言 `agent.json` 及相邻测试。
- 避让：`lime-rs/crates/app-server/src/runtime/read_model/tests.rs` 当前存在未知改动，本计划不触碰。

## 3. 基线证据

### 3.1 已有但未形成单一产品面

- `TaskCenterUtilityToolbar.tsx` 已有打开位置、环境信息、Git 分支和 panel 入口。
- `InputbarProjectContextBar.tsx` 已有项目、分支和 worktree 控制。
- `CanvasWorkbenchLayout` 已有文件、结果、变更和 review surface。
- canonical `thread/read`、Item materialization、审批、计划、SubAgent 和恢复投影已有 v2 evidence。

### 3.2 当前结构性差距

- `AgentChatWorkspace.tsx` 仍有约 2422 行，超过 v2 目标 `< 800` 行；scene composition 仍承接大量编排。
- `TaskCenterUtilityToolbar.tsx` 约 972 行，同时拥有环境读取、任务轨道、App 打开和多个 panel 控制。
- 全局 App Sidebar 是 Codex App 风格的稳定一级导航，应继续保留；当前 Thread 缺少稳定、紧凑的页头主对象，工作区内部也需避免再造第二套左侧会话导航。
- 首页首屏仍是皮肤 Hero、营销文案和技能入口，和 Codex App 的任务工作区心智不一致。
- 环境、计划、审批、运行状态同时出现在 Toolbar popover、Inputbar、Timeline、Harness、Canvas session view 等多套 surface。
- 主写链仍有大量 `agentSession/*`；Codex-rs current contract 是 `thread/*`、`turn/*`，并包含 `turn/steer`、`thread/fork`、`thread/rollback`、`environment/info`、`environment/status` 等能力。

### 3.3 本轮 GUI 证据边界

- `http://127.0.0.1:1420/` 浏览器镜像在 1536x960 下完成首页、空任务和历史会话外壳检查；控制台 error 为 0。
- 证据等级为 Gate A，仅证明 Renderer 当前布局；Electron 已运行但未开放 `9223` CDP，未取得 Gate B。

## 4. 差距优先级

| 优先级 | 目标 | 当前差距 | 依赖 |
| --- | --- | --- | --- |
| P0 | 当前 Thread 成为主画布对象 | 全局 Sidebar 需要保留，但项目 tabs、任务 tabs 和消息区仍竞争上下文；活跃任务页头不稳定 | GUI-only，可先做 |
| P0 | 单一 canonical 时间线 | commentary、reasoning、tool、状态摘要存在重复渲染与过量纵向展开 | `thread/read` Item projection |
| P0 | 单一 action-required 入口 | Approval/Plan/request-user-input 可同时出现在输入区、时间线和环境任务轨道 | canonical Approval/Plan Item |
| P0 | 写链对齐 Codex-rs | `agentSession/turn/start`、cancel、resume 等仍是主写 contract | App Server protocol/runtime |
| P1 | Thread 绑定的环境面 | 现有环境浮层只读 project Git，且把任务轨道混入环境面；缺 environment identity/status | `environment/*` + project Git |
| P1 | 变更审查闭环 | 已有 diff/workbench，但环境面没有增删行、比较基线和进入 review 的明确动作 | project Git + workbench command |
| P1 | Thread 生命周期完整 | list/read 基本存在，fork/rollback/archive/resume/steer 未形成一致 GUI | `thread/*`、`turn/*` |
| P1 | Composer 稳定 | 首页与会话输入区形态、装饰和高度变化明显；运行态/排队态入口分散 | command model |
| P2 | 产品/诊断分层 | Harness、Trace、Shell、Browser、Workbench 在主 Toolbar 近似等权 | 右侧 surface catalog |
| P2 | 首页对齐任务产品 | Hero 和技能陈列压过新任务输入，不像工作型桌面应用 | shell/navigation |

## 5. 实施切片

### G1：Thread workspace shell

目标：active Thread 首屏只保留紧凑任务页头、主时间线、稳定 composer 和一个右侧上下文入口。

动作：

- 从 canonical Topic/Thread metadata 派生标题、状态和工作目录。
- 普通历史会话继续保留全局 App Sidebar；它只负责一级导航和会话入口，工作区内部不再新增平行左侧会话导航。
- “打开位置”使用带文字的明确命令；环境入口和右侧 surface 保持单一。
- 不在本切片新增 Git、Thread 或 Environment method。

退出条件：用户 5 秒内能识别当前任务、当前状态和下一步；桌面宽屏保留全局 Sidebar，同时 Thread 页头与工作区内容不再重复表达同一层导航。

### G2：Canonical timeline

目标：一个 Thread Item 只由一个 renderer 呈现。

动作：

- User/Agent/Reasoning/Tool/Approval/FileChange/Artifact/SubAgent/Compaction 使用稳定 Item renderer registry。
- completed tool 默认压成紧凑语义行；running/failed/pending 才展开必要详情。
- commentary 与 final answer 保持连续阅读；删除重复的“先发起这一步/已找到/已处理”二次摘要。
- 全局只保留一个 active Turn 状态行。

退出条件：同一 tool/approval/plan 不会在 Timeline、Harness 和 environment popover 同时成为主操作。

### G3：Composer 与 action-required

目标：composer 是 send/steer/interrupt/approval/request-user-input 的单一命令面。

动作：

- 空闲态发送，运行态 steer/queue，执行态 interrupt，等待态响应表单。
- Approval、Plan decision、request-user-input 互斥占用 action slot；历史结果只在时间线只读回显。
- 权限、模型和工作目录显示有效值，不从 Renderer 猜 runtime truth。

退出条件：任何时刻只有一个主按钮和一个需要用户处理的面。

### G4：Environment 与 changes

目标：环境面只回答“任务在哪里运行、代码处于什么状态、下一步如何检查/交付”。

动作：

- 后端复制并适配 Codex-rs `environment/info`、`environment/status`；Thread 持有 environment selection。
- Git 状态提供 branch、upstream/base、文件数、added/deleted lines 和 refresh 状态。
- “比较分支”进入现有 changes workbench；commit/push 未实现时不显示假按钮。
- 删除 Toolbar 与 Inputbar 内重复的 Git fetch owner，收敛为 typed projection。

退出条件：环境状态刷新不依赖组件各自发请求；失败可见且不回退 mock。

### G5：Thread lifecycle

目标：GUI 对齐 Codex-rs 的 `thread/start|list|loaded/list|read|resume|fork|archive|unarchive|rollback` 与 `turn/start|steer|interrupt`。

动作：直接迁移调用并删除对应 `agentSession/*` 主入口，不新增 rename wrapper。

退出条件：新建、恢复、分叉、回滚、归档、排队、打断都由 canonical notification/read model 回填 GUI。

### G6：Surface cleanup

目标：删除对 Codex App 主工作流无贡献的平行产品面。

候选删除/降级：

- active Thread 中的营销 Hero 和重复技能陈列。
- 主 Toolbar 里的 Harness/Trace 工程词；迁入开发者诊断面。
- environment popover 内嵌的完整 Task Rail。
- 重复 session overview、runtime strip 和历史状态摘要。
- `agentSession/*` 正向 GUI fixture、旧 slash command 占位和无消费 compat surface。

退出条件：生产 GUI 只消费 current Thread/Turn/Item projection；旧名只留负向 guard 或历史 evidence。

## 6. 验证门禁

### 6.1 Agent Verification Contract

```text
改动名称：Codex App GUI 对齐
执行计划文件：internal/exec-plans/codex-app-gui-alignment-plan.md
负责人：agent-ui + app-server
预算标签：budget:normal
风险等级：P0
影响模块：Agent GUI shell、Thread/Turn/Item projection、App Server typed gateway
不做范围：TUI 移植、ANSI/CLI 快捷键、live Provider 质量评估、发布产物
```

Current 主链：

```text
前端入口：AgentChatPage -> AgentChatWorkspace scene composition
前端网关：src/lib/api/agentRuntime + packages/app-server-client
Electron Desktop Host bridge：app_server_handle_json_lines
App Server method：当前 read 为 thread/read；目标写链为 thread/* + turn/*
RuntimeCore / service owner：RuntimeCore + agent-runtime + tool-runtime
read model：Thread / Turn / Item canonical read model
runtime event：accepted/started/queued/running/completed/failed/interrupted + Item lifecycle
Evidence Pack 字段：threadId、turnId、itemId、method、transport、status；不记录 secret/完整 prompt
GUI surface：Thread workspace header、timeline、composer、environment/changes right surface
```

Happy Path：

```text
用户输入 / Agent 输入：在 active Thread composer 提交任务或运行中 steer
预期 runtime events：同一 thread/turn identity 的 accepted -> started -> terminal
预期 tool calls：以 canonical Tool Item 在 timeline 单次投影
预期 approval / sandbox：action slot 单点响应，历史只读回显
预期 artifact：进入 workbench/right surface，不在消息正文伪造状态
预期 evidence：Gate B trace + read model + 可见 DOM identity 一致
预期 GUI 状态：主对象、当前阶段、阻塞和下一步同时可见
失败时应停在哪一层：typed gateway/App Server/runtime 显式失败；禁止生产 mock fallback
```

Evidence Layers：

| Layer | 本次是否需要 | 证据路径 / 计划路径 | 不需要的原因 |
| --- | --- | --- | --- |
| deterministic-smoke | 是 | related tests、contracts、current fixture | - |
| gui-trace | 是 | 每切片 Gate A + Electron Gate B evidence | - |
| runtime-transcript | 是 | current fixture 的 Thread/Turn/Item 摘要 | - |
| release-artifact | 否 | - | 本计划不是发版计划 |

Agent QC 场景映射：

```text
P0：Claw 新建/恢复 Thread、发送、stream、interrupt、approval、历史 hydration
P1：environment/changes、fork/rollback/archive、SubAgent activity、窄窗口布局
P2：首页和诊断 surface 收口
为什么需要：直接改变 Agent GUI 主路径和 action owner
为什么不需要其它 P0：媒体/浏览器/live provider 只在对应切片触及时加入
是否允许单场景 sidecar：允许，budget:normal 下按受影响场景选择
是否允许进入 official evidence：只有 Gate B 与必跑命令全部通过后允许
```

Supervisor：本计划不使用 LLM judge；确定性 contract、DOM、trace 和截图足以判断。失败必须回写到最接近 owner 的 unit/component/contract/Gate B fixture。

每个用户可见切片至少执行：

```bash
npm run test:related -- <changed-paths...>
npm run i18n:check:json
npm run verify:gui-smoke
```

协议/后端切片追加：

```bash
npm run test:contracts
npm run test:rust:related -- <changed-rust-paths...>
npm run smoke:agent-runtime-current-fixture
```

每个 P0/P1 产品切片必须记录：

- 1536x960、1280x800 和窄窗口 Gate A 截图/DOM 证据。
- 真实 Electron Gate B：preload/IPC、`app_server_handle_json_lines`、current method、同一 thread/turn/item identity、mock fallback 为 0。
- `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR` 稳定文案回归。

## 7. 治理分类

- `current`：Electron -> App Server -> RuntimeCore -> Thread/Turn/Item -> GUI；Codex App-style Thread workspace。
- `compat`：本计划不新增。
- `deprecated`：迁移期间尚未删除的 `agentSession/*` GUI 写入口，必须逐切片写明退出条件。
- `dead / forbidden-to-restore`：重复 Renderer 状态机、生产 mock fallback、与 canonical Item 重复的主操作 surface、仅复刻 TUI 的 UI。

## 8. 完成度

- 差距审计：100%。
- GUI 对齐实现：0%。
- 当前总完成度：10%（已有 v2 runtime/projection 与若干 Codex-like surface，不代表 GUI 对齐完成）。

## 9. 架构图确认

```text
架构影响：G1-G3 预期为现有 GUI owner 内收口；G4-G5 会修改 protocol、read/write path 和跨层 owner，属于重大架构变更。
架构图已更新：当前不适用，尚未实施 G4-G5；实现时必须同步 internal/aiprompts/architecture.md。
责任开发者确认：待 G4/G5 实现责任开发者填写，未确认不得标记完成或进入 release evidence。
确认内容：待核对目录归属、数据流、依赖方向、协议边界和验证门禁。
```

完成标准：G1-G6 退出条件全部满足，必跑命令通过，五语言回归完成，Gate B 证明同一 Thread/Turn/Item identity 且生产 mock fallback 为 0。当前尚不可进入 release evidence；下一刀仍为 G1。

## 10. 2026-08-03 环境菜单对齐增量

- 范围：继续收口 G1 顶部工具栏与 G4 环境面，只调整既有 project Git current gateway 的 GUI 投影，不新增协议或第二套 Git owner。

- 写集：`TaskCenterUtilityToolbar.tsx`、`TaskCenterLocationPanel.tsx`、`TaskCenterEnvironmentPanel.tsx`、工具栏集成测试和五语言 `agent.json`。
- 结果：打开位置收为单一紧凑菜单；环境面展示真实增删行、运行位置和当前分支；分支菜单复用既有 checkout/create API，提供搜索、当前分支未提交摘要、分支切换和创建入口。分支操作后只刷新局部 Git projection，不重载窗口或中断对话。
- 稳定回归：`TaskCenterUtilityToolbar.integration.test.tsx` 30 项通过，覆盖菜单结构、分支列表和真实 checkout gateway 调用；TypeScript、Prettier、五语言完整性检查通过。
- GUI 证据：`npm run verify:gui-smoke` 通过，Electron Host、preload/App Server 初始化、Renderer 首次加载和重载均成功；最新 evidence summary 位于 `.lime/qc/project-gates/standalone-shell-01-20260803115109-24101/shell-01-electron-smoke/summary.json`。
- 视觉续测：系统 Chrome + Playwright 在 `1536x960` 的 browser mirror 中实测“打开位置”为 `216x159.5px`、“环境信息”为 `300x200px`，“比较分支”同时包含比较与外链图标，console error 为 0；该证据等级为 Gate A，不替代真实 Electron CDP 的 Gate B 交互证据。
- 未完成：本增量不实现 commit/push、比较基线选择、environment protocol 或 Thread-bound environment identity，G4 仍为进行中，不能进入 release evidence。

## 11. 2026-08-03 会话阅读列与回合摘要收口

- 范围：G1/G2 的视觉对齐，只调整现有消息列、输入栏共用的阅读宽度 token 和历史回合摘要布局，不触碰 Thread/Turn/Item 协议与 runtime 投影。
- 结果：正文、文件变更卡片、回合状态线、inline/floating composer 统一使用 `clamp(640px, 68%, 720px)`；历史回合摘要固定按“已处理与耗时 / 步骤和工具数 / 展开”三段横向排布，并使用整列分隔线保持与正文、卡片对齐。
- 稳定回归：导入 Codex 历史 fixture 断言回合摘要始终输出步骤数和工具步骤数，避免元信息再次被挤压为只剩耗时。

## 12. 2026-08-03 Canonical timeline 合并增量

- 范围：收口 G2 canonical Thread/Turn/Item 时间线的回合级显示，不新增协议或第二套消息 owner。
- 实现：运行中的 turn 保留 process segment 时序；已结束 turn 将多个 process segment 合并为一个 `process:<turn>:merged` 摘要 owner；canonical turn 的普通 assistant 操作栏仅保留最后一个有正文的 assistant segment。
- 退出条件：同一已结束 turn 只渲染一条历史过程摘要；同一 turn 的普通正文不再逐段重复圆形操作栏；产物、审批和文件卡片仍由既有 timeline owner 渲染。
- 验证：`MessageList.directTimeline.test.tsx`、`MessageList.messageActions.test.tsx` 及历史/产物/失败工具/reasoning 相关定向测试通过；`i18n:check:json`、`git diff --check`、`verify:gui-smoke` 通过。
