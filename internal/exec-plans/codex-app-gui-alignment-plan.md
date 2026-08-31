# Codex App GUI 对齐执行计划

> status: active / D0-desktop-baseline-blocked -> D2-D3-complete -> D4-multi-agent-durable-complete -> D4-windows-enforcement-phase
> owner: agent-ui + app-server
> started: 2026-07-18
> last reviewed: 2026-08-31
> target: Codex App GUI interaction model
> backend reference: `/Users/coso/Documents/dev/rust/codex/codex-rs`
> architecture baseline: `internal/aiprompts/architecture.md`
> verification baseline: `internal/aiprompts/playwright-e2e.md`

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

- 当前阶段：`D2 Environment and lifecycle`、`D3 Runtime surfaces`、D4 ThreadStore、Permission Policy 和 Multi-Agent durable semantics 均已完成。Environment/status、Changes、Thread lifecycle、Thread Activity、Skills/MCP change、240 Turn / 720 Item 长历史和进程级冷启动恢复都有真实 Electron Gate B；canonical rollout 已补 crash-tail byte-offset repair、sequence gap continuation 和同步 writer teardown/discard 证据；Thread 创建、Turn 启动、settings mutation、GUI 提交前 catalog 查询以及 AgentControl child resume 共用 current App Server/runtime/read-model 链；Codex Desktop 实时窗口读取仍因 macOS AppleEvent / Computer Use 超时停留在 D0 blocked。
- 已确认 current：Fork 从 canonical `thread/fork` response 继承标题、cwd 与 `forkedFromId`；Renderer 只把该摘要用于即时侧栏投影，随后仍由 `thread/read|resume|list` 收敛。paginated Fork、archive/unarchive、revert、turn start/steer/interrupt 和冷启动恢复均沿同一 Thread/Turn/Item identity，未新增第二套 metadata fallback、history store、legacy wrapper 或生产 mock。
- 下一刀：进入 `D4 Runtime parity` 的真实 Windows packaged enforcement 与 ACL/ConPTY/网络隔离证据；custom permission profile、deny-read 与 MDM/project-local requirements 保持 product-scope excluded，除非先建立真实 consumer 和唯一 owner。
- 本轮收口写集：`lime-rs/crates/app-server/src/permission_profile.rs`、`processor/{permission_profile,thread,turn}.rs`、`runtime/session_operations.rs`、`tests/permission_profile_jsonrpc.rs`、`src/lib/api/permissionProfiles*`、`src/components/agent/chat/hooks/{agentStreamPreparedSendEnv,agentStreamUserInputSubmission,agentStreamSubmitExecution,useAgentStream,useAgentChat}*`、`internal/aiprompts/commands.md` 与本执行计划；D4 Multi-Agent 追加 `lime-rs/crates/app-server/src/runtime/agent_control.rs`、`runtime/tests/agent_control/restart.rs`、`tests/thread_direct_input_policy_jsonrpc.rs` 和 `scripts/agent-runtime/tool-execution-managed-smoke.mjs`；未新增协议 method、Electron 业务命令、GUI 状态或权限 compat API。
- 避让：其余 MCP、Right Surface、Claw fixture 热区和未跟踪参考文件 `local-conversation-page-B5LUHmAw.js` 均只读。

## 3. 基线证据

### 3.1 已有但未形成单一产品面

- `TaskCenterUtilityToolbar.tsx` 已有打开位置、环境信息、Git 分支和 panel 入口。
- `InputbarProjectContextBar.tsx` 已有项目、分支和 worktree 控制。
- `CanvasWorkbenchLayout` 已有文件、结果、变更和 review surface。
- canonical `thread/read`、Item materialization、审批、计划、SubAgent 和恢复投影已有 v2 evidence。

### 3.2 当前结构性差距

- `AgentChatWorkspace.tsx` 已拆分为 13 行入口；主要编排已下沉到 `WorkspaceConversationScene.tsx`（692 行）和相关 hooks，后续重点是减少 scene 层状态竞争，而不是继续堆叠入口文件。
- `TaskCenterUtilityToolbar.tsx` 当前约 853 行，仍同时拥有环境读取、任务轨道、App 打开和多个 panel 控制，接近非生成文件 800 行拆分阈值。
- 全局 App Sidebar 是 Codex App 风格的稳定一级导航，应继续保留；当前 Thread 缺少稳定、紧凑的页头主对象，工作区内部也需避免再造第二套左侧会话导航。
- 首页首屏仍是皮肤 Hero、营销文案和技能入口，和 Codex App 的任务工作区心智不一致。
- 环境、计划、审批、运行状态同时出现在 Toolbar popover、Inputbar、Timeline、Harness、Canvas session view 等多套 surface。
- GUI 仍有大量 `agentSession/event/*` 绑定和历史 fixture；Codex-rs current 写链应以 `thread/*`、`turn/*` 为准，`agentSession/event` 只能作为受控旁路诊断，不能成为 GUI 生命周期事实源。Codex 的 `thread/rollback` 已 deprecated，Lime GUI current 以 `thread/revert` 为准。

### 3.3 本轮 GUI 证据边界

- `http://127.0.0.1:1420/` 浏览器镜像在 1536x960 下完成首页、空任务和历史会话外壳检查；控制台 error 为 0。
- 证据等级为 Gate A，仅证明 Renderer 当前布局；Electron 已运行但未开放 `9223` CDP，未取得 Gate B。

### 3.4 已安装 Codex Desktop 参考基线（2026-08-29）

- Bundle：`/Applications/ChatGPT.app`，`CFBundleIdentifier=com.openai.codex`，`CFBundleShortVersionString=26.818.41509`，资源包为 `/Applications/ChatGPT.app/Contents/Resources/app.asar`（约 271 MB）。
- 运行链：`ChatGPT.app -> Codex Framework -> Resources/codex ... app-server`；本机进程实测为 `/Applications/ChatGPT.app/Contents/Resources/codex -c features.code_mode_host=true app-server --analytics-default-enabled`。
- 静态资源中存在以下 GUI surface：`local-conversation-page`、`local-conversation-thread-turn-entries`、`thread-virtualizer`、`composer-action-bar-run-location-dropdown`、`composer-utility-bar`、`worktree-environment-dropdown`、`local-environment-workflow-messages`、`local-conversation-thread`、`thread-side-panel-tabs`、`toggle-thread-summary-panel`。
- 静态代码/文案显示的交互语义：Thread header、usage/summary 面板、最新回合预览、完整/折叠 transcript、虚拟化回合内容、运行位置（本机/云端）、worktree/environment 状态、环境错误重试、Git/PR checks 与 “Fix all” 入口、归档 Thread 恢复。
- 静态打包代码还出现 `thread/start|list|read|resume|fork|archive|unarchive|revert`、`turn/start|steer|interrupt`、`thread/summary|environment|queue|status|goal|backgroundTerminals|section`、`environment/info|status`、`project/*`、`item/*`、`mcpServer/*`、`skills/list|changed` 等 method family；这些只用于确认参考面，不等于 Lime 已完成对应 wire 或 GUI 闭环。
- 证据限制：本轮 Computer Use 读取桌面窗口收到 macOS AppleEvent 超时（`-1743`），因此没有实时 accessibility tree、截图或点击路径；`app.asar` 只能作为设计参考，不能升级为 Gate A/B 产品证据。

### 3.5 Codex Desktop / Lime 对比矩阵

| Codex Desktop surface（静态基线）                               | Lime current                                                                                         | 未对齐点                                                                                                 | owner / 优先级                   |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 一个 Thread 页面：header、summary、timeline、composer           | Workspace shell、多个 timeline、Toolbar/Inputbar/Canvas 并存                                         | 主对象和 action owner 仍分散，存在重复状态摘要                                                           | agent-ui / P0                    |
| `local-conversation-thread-turn-entries` + `thread-virtualizer` | canonical Thread/Turn/Item 使用 opaque cursor 和 10 Turn / 30 Item 有界首帧窗口                      | renderer registry 仍需继续去重；历史分页、首帧窗口和进程级恢复已有 Gate B                                | agent-ui + app-server / P0       |
| 运行位置下拉：本机 / 云端及文件访问说明                         | 项目上下文、打开位置和环境 panel 分散                                                                | 缺 Thread-bound environment selection/status 的单一投影；云端仅作为明确 scope，不强行复制                | app-server + agent-ui / P1       |
| worktree/environment dropdown、环境错误重试                     | `TaskCenterEnvironmentPanel`、Git branch 菜单已有                                                    | 缺 environment identity、失败恢复闭环和 worktree/branch 变更摘要                                         | app-server + tool-runtime / P1   |
| Thread lifecycle、archived restore、PR/changes side panel       | lifecycle、Changes、archived restore、crash-tail repair 与 writer teardown/discard 已有 current 证据 | D2 与 D4 ThreadStore 第一阶段已完成；`thread/rollback` 不恢复为 current，剩余平台/权限项不回挂 lifecycle | thread-store + app-server / done |
| background subagent activity、MCP/Skills side panels            | Activity、Skills change、MCP 三类 list-change 均有 current projection 和 Gate B                      | durable Multi-Agent ack/retry/residency 归 D4；不再重复建设 runtime surface                              | agent-runtime + app-server / P2  |
| 稳定窄窗口/长历史体验                                           | 240 Turn / 720 Item 长历史与进程级冷启动恢复已有 Electron Gate B                                     | 尚未取得 1536x960、1280x800、窄窗三档 Codex Desktop 实时对照截图                                         | agent-ui / P0                    |

### 3.6 跨层语义差距（Codex runtime 参考）

| 领域                  | Codex-rs 参考能力                                                                                                                 | Lime current                                                                                                                                    | 尚未对齐                                                                                                                                       | owner / 优先级                    |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| Windows sandbox       | restricted token、ACL/DACL read-back、ConPTY、Job Object、网络隔离和 packaged resource 守卫                                       | `windowsSandbox/setupStart` 等协议与 `tool-runtime::windows_setup` 已有                                                                         | 真实 Windows/MSVC enforcement、审计和 Gate B evidence 未完成                                                                                   | tool-runtime + app-server / P0    |
| Permission profile    | config layer/managed requirements、动态 allowed、custom profile、deny-read 和 cwd 解析                                            | 三个内置 profile、动态 `allowed`、cwd 物化、全局 sandbox config/platform/setup readiness、Thread/Turn/settings lowering 与 GUI 提交前查询已实现 | custom profile、deny-read、MDM/project-local requirements 与真实 Windows packaged enforcement 未实现；cwd 没有 project-local policy layer      | app-server + tool-runtime / P1    |
| ThreadStore lifecycle | live writer、flush/shutdown/discard、history materialization、latest model context、paginated fork lineage、ordinal/offset repair | canonical SQLite projection、sequence/cursor、revert、paginated Fork、干净重启和 crash-tail repair 已有                                         | 已完成：合法 unterminated record 保留、非法尾部按 byte offset 截断、sequence gap 续写、中段损坏 fail closed；同步 writer 无额外 pending buffer | thread-store + app-server / done  |
| Multi-Agent           | parent-child registry、resident unload/reload、execution limiter、control tools、durable graph/mailbox                            | graph、mailbox、control gateway、activity projection、parent-owned child canonical resume、capacity/residency cold restart 均有受控 fixture 与 Gate B | 真实 Windows packaged enforcement 与跨平台资源限制仍待平台证据；Multi-Agent current 语义本身已收口                                      | agent-runtime + thread-store / done |

## 4. 差距优先级

| 优先级 | 目标                       | 当前差距                                                                                                                        | 依赖                            |
| ------ | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| P0     | 当前 Thread 成为主画布对象 | 全局 Sidebar 需要保留，但项目 tabs、任务 tabs 和消息区仍竞争上下文；活跃任务页头不稳定                                          | GUI-only，可先做                |
| P0     | 单一 canonical 时间线      | commentary、reasoning、tool、状态摘要存在重复渲染与过量纵向展开                                                                 | `thread/read` Item projection   |
| P0     | 单一 action-required 入口  | Approval/Plan/request-user-input 可同时出现在输入区、时间线和环境任务轨道                                                       | canonical Approval/Plan Item    |
| P0     | 写链对齐 Codex-rs          | GUI 仍由 session-oriented hooks 与 `agentSession/event/*` 历史绑定承接部分生命周期；current 主写应明确落到 `thread/*`、`turn/*` | App Server protocol/runtime     |
| P1     | Thread 绑定的环境面        | 现有环境浮层只读 project Git，且把任务轨道混入环境面；缺 environment identity/status                                            | `environment/*` + project Git   |
| P1     | 变更审查闭环               | 已有 diff/workbench，但环境面没有增删行、比较基线和进入 review 的明确动作                                                       | project Git + workbench command |
| done   | ThreadStore 崩溃恢复       | partial JSONL、byte offset/sequence gap repair、同步 writer close 与 delete/discard 已由 D4 第一阶段收口                        | thread-store + App Server       |
| P1     | Composer 稳定              | 首页与会话输入区形态、装饰和高度变化明显；运行态/排队态入口分散                                                                 | command model                   |
| P2     | 产品/诊断分层              | Harness、Trace、Shell、Browser、Workbench 在主 Toolbar 近似等权                                                                 | 右侧 surface catalog            |
| P2     | 首页对齐任务产品           | Hero 和技能陈列压过新任务输入，不像工作型桌面应用                                                                               | shell/navigation                |

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

目标：GUI 对齐 Codex-rs 的 `thread/start|list|loaded/list|read|resume|fork|archive|unarchive|revert` 与 `turn/start|steer|interrupt`；`thread/rollback` 仅保留 deprecated/底层兼容语义，不作为 GUI current 入口。

动作：直接迁移 GUI 生命周期调用并删除对应 session-oriented `agentSession/*` 主入口，不新增 rename wrapper；保留必要的 action response 作为明确边界；`thread/revert` 成功后按协议要求回到 `thread/read`/`thread/resume` 的 canonical projection。

退出条件：新建、恢复、分叉、revert、归档、排队、打断都由 canonical notification/read model 回填 GUI。

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

| Layer               | 本次是否需要 | 证据路径 / 计划路径                       | 不需要的原因       |
| ------------------- | ------------ | ----------------------------------------- | ------------------ |
| deterministic-smoke | 是           | related tests、contracts、current fixture | -                  |
| gui-trace           | 是           | 每切片 Gate A + Electron Gate B evidence  | -                  |
| runtime-transcript  | 是           | current fixture 的 Thread/Turn/Item 摘要  | -                  |
| release-artifact    | 否           | -                                         | 本计划不是发版计划 |

Agent QC 场景映射：

```text
P0：Claw 新建/恢复 Thread、发送、stream、interrupt、approval、历史 hydration
P1：environment/changes、fork/revert/archive、SubAgent activity、窄窗口布局
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

- 源码与静态打包资源差距审计：100%；真实 Codex Desktop 视觉基线：0%（D0 受 accessibility blocker 阻塞）。
- GUI 对齐实现：D1、D2、D3 已完成；D4 ThreadStore、Permission provenance/settings mutation、动态 policy producer 与 Multi-Agent durable semantics 均已完成，D0 真实视觉基线和 Windows 平台 enforcement 未完成。
- 当前总完成度：86%。D2/D3 已证明 lifecycle、Activity、MCP/Skills change、长历史分页/有界首帧和进程级冷启动恢复；D4 已补 crash-tail repair、同步 writer lifecycle、动态 permission policy，以及 AgentControl parent-child 的 canonical resume、ack/retry、capacity/residency 和 cold-restart 证据；D0 实时视觉基线与 Windows packaged enforcement 仍未完成。

## 9. 架构图确认

```text
架构影响：本轮 D4 permission settings mutation 复用既有 `thread/settings/update`、App Server RuntimeCore metadata owner 和 Electron JSONL 转发，不新增 public method、schema owner、进程边界或依赖方向，不构成重大架构变更。
架构图已更新：不需要；`internal/aiprompts/architecture.md` 的 current 产品链和 owner 边界保持不变。
责任开发者确认：本轮无需新增架构图确认；后续若引入 managed requirements/policy owner 或改变 public boundary，必须在同一变更集重新确认。
确认内容：权限 catalog、Thread/Turn/settings lowering 均由 App Server current owner 收口；Electron 与 Renderer 未建立第二套 policy owner。
```

完成标准：G1-G6 退出条件全部满足，必跑命令通过，五语言回归完成，Gate B 证明同一 Thread/Turn/Item identity 且生产 mock fallback 为 0。当前尚不可进入 release evidence；D1-D3、D4 ThreadStore、Permission Policy 与 Multi-Agent durable semantics 已完成，下一刀为真实 Windows packaged enforcement，D0 仍受 AppleEvent blocker 影响。

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

## 13. 2026-08-29 新增：Codex Desktop 对齐计划

### 13.1 目标与边界

目标是把 Codex Desktop `26.818.41509` 已确认的 Thread 工作区信息架构转成 Lime 的可验证 current 实现：一个 active Thread 页面、一个 canonical timeline、一个 action-required/composer 槽位，以及一个 Thread-bound environment/changes 入口。

本计划只对齐桌面工作流，不把 Codex Desktop 的云端执行、PR 自动化、Slack/连接器或远程产品面默认为 Lime 需求；这些 surface 只有在 Lime scope matrix 明确纳入后才单独立项。Codex Desktop 的 `app.asar` 静态资源是参考输入，不能替代 Lime 的 Gate A/Gate B。

### 13.2 实施切片

| 切片                         | 交付                                                                                                                                                  | 主要写集                                                                                                                                                                   | owner                                    | 退出条件                                                                                                                                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- | ------ | ---- | ------- | --------- | -------------------- | ----- | ----------------------------------------------------------------------------------- |
| D0 Desktop baseline          | 取得 Codex Desktop 1536x960、1280x800、窄窗截图/DOM 观察，记录 header、summary、timeline、composer、run-location、environment/worktree 和错误恢复状态 | `.lime/qc/gui-evidence/` 证据、计划记录；不改生产代码                                                                                                                      | agent-ui                                 | 真实窗口观察成功后才可进入视觉验收；若仍受 macOS accessibility blocker 影响，D0 必须标记 blocked，D1 不得用静态资源替代视觉验收；每张截图标注版本和 evidence level                         |
| D1 Thread page convergence   | 收敛 Thread header、canonical timeline、单一 action slot；移除重复 summary/主操作 surface                                                             | `WorkspaceConversationScene.tsx`、`WorkspaceMainArea.tsx`、`ThreadWorkspaceHeader.tsx`、timeline/composer hooks 与五语言资源                                               | agent-ui                                 | active Thread 的 title/status/next action 5 秒内可识别；同一 Item 只有一个主 renderer；send/steer/interrupt/approval/request-user-input 只有一个 action owner                              |
| D2 Environment and lifecycle | 补 Thread-bound environment/status、worktree/branch/changes 摘要，并完成 lifecycle 回填                                                               | App Server environment/thread processor、typed gateway、Task Center environment/changes、lifecycle tests                                                                   | app-server + agent-ui                    | `thread/start                                                                                                                                                                              | read | resume | fork | archive | unarchive | revert`与`turn/start | steer | interrupt` 的 response/notification/read model identity 一致；失败可见且不回退 mock |
| D3 Runtime surfaces          | 将 SubAgent activity、MCP/Skills notification、history paging/virtualization 收敛到 Thread side panel 或 diagnostics surface                          | `agent-runtime`/`thread-store` projection、side-panel tabs、history window、相关 fixture                                                                                   | agent-runtime + app-server               | parent-child、list_changed、skill change、冷启动恢复和长历史均有 deterministic fixture；诊断面不与主 Thread action 竞争                                                                    |
| D4 Runtime parity            | 补齐 GUI 依赖的权限、安全和存储语义证据                                                                                                               | `lime-rs/crates/tool-runtime/src/windows_setup.rs`、`app-server/src/processor/permission_profile.rs`、`thread-store` lifecycle/repair、`agent-runtime` agent graph/mailbox | tool-runtime + app-server + thread-store | Windows enforcement/ACL/ConPTY/网络隔离、permission provenance、ThreadStore crash/fork repair、Multi-Agent ack/retry/residency 均有平台或受控 fixture 证据；不能用状态文案代替 enforcement |

### 13.3 每片验证与证据

- D0/D1：`npm run verify:gui-smoke`、`npm run i18n:check:json`、受影响组件定向测试；同时保留 1536x960、1280x800、窄窗 Gate A 截图/DOM。
- D2-D4：追加 `npm run test:contracts`、`npm run smoke:agent-runtime-current-fixture`、`npm run test:rust:related -- <changed-rust-paths...>`；必要时运行对应 crate 测试；Windows 证据必须标注真实 MSVC/packaged 与 macOS 受控 fixture 的平台边界。
- 每个 P0/P1 切片必须有真实 Electron Gate B：Electron/preload/IPC、`app_server_handle_json_lines`、current App Server method、RuntimeCore/read model 和用户可见 DOM 共享同一 `threadId`/`turnId`/`itemId`；生产 mock fallback 为 0。
- 所有用户可见文案回归 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`；完成前运行 `npm run governance:legacy-report`，确认没有新增 `agentSession/*` 正向入口或重复 owner。

### 13.4 当前阻塞与决策记录

- Codex Desktop 实时窗口读取当前因 macOS AppleEvent 超时（`-1743`）未完成；在 D0 解除前，静态 `app.asar` 证据只可用于信息架构假设，不能作为视觉验收或 Gate B。
- 已安装客户端实际版本和资源路径已记录，但不把其远程/cloud/PR 能力自动移入 Lime scope。
- Environment/Changes/lifecycle 已由 D2 收口；MCP subscriptions/list_changed、Skills change、SubAgent Activity 和长历史干净进程重启已由 D3 收口；ThreadStore crash recovery/writer lifecycle 与动态 permission policy 已由 D4 前三阶段收口。Windows sandbox 与 Multi-Agent durable semantics 仍是 D4 跨层差距，不能通过 GUI 文案宣称完成。

### 13.5 计划完成标准

D0-D4 的退出条件全部满足，Gate A 与真实 Electron Gate B 证据可追溯，contracts、GUI smoke、current fixture、i18n、相关 Rust 测试和治理报告通过；架构变更同步 `internal/aiprompts/architecture.md` 并完成责任开发者确认后，才允许将本计划标记为完成或进入 release evidence。

## 14. 2026-08-30 D1 第一刀：Thread 页头与 Composer action slot

- 范围：只在既有 GUI owner 内收敛 Codex Desktop 风格的 active Thread shell；不新增 App Server 方法、不改变 Thread/Turn/Item 协议、不引入生产 mock。
- 实现：`WorkspaceConversationScene` 不再把 Thread 页头限制在 `task-center`，所有有 active Thread 的聊天态稳定展示标题、状态和工作目录；返回新建任务、资源、项目管理、画布和设置动作压缩到同一页头，原顶部导航不再与页头并列竞争。
- 实现：任务中心工具栏不再依赖会话标签条存在，深链接/恢复会话仍能获得统一环境、工作台和诊断入口；Composer 与浮层输入区统一标记 `data-action-owner="composer"`。
- 实现：`PendingInteractionLayer` 和 approval/plan replacement 统一标记为 Composer owner；存在 typed reverse request 时由其独占 action slot，Plan 决策不会在其下方重复挂载；历史 Timeline 仍只读呈现结果。
- 稳定回归：Thread header、Workspace Main Area、Workspace Conversation Scene、Inputbar runtime 定向测试共 58 项通过；新增默认 workspace active Thread 页头导航和 reverse request/Plan 互斥断言。
- 验证：`npm run typecheck`、`npm run lint -- --no-warn-ignored`、`npm run i18n:check:json` 通过；`npm run smoke:agent-runtime-current-fixture` 全部 current fixture 场景通过（liveProviderUsed=false）；`npm run verify:gui-smoke` 通过，Electron Gate B summary：`.lime/qc/project-gates/standalone-shell-01-20260830015800-56717/shell-01-electron-smoke/summary.json`。
- 未完成：D0 真实 Codex Desktop 窗口仍受 macOS AppleEvent `-1743` 阻塞；D2 changes/lifecycle 全场景证据、D3 runtime side panels、D4 Windows sandbox/permission/ThreadStore/Multi-Agent parity 尚未实现，不能标记计划完成或进入 release evidence。

## 15. 2026-08-30 D2 第一阶段：Canonical Environment 与 lifecycle 回填

- 范围：沿用既有 `environment/info`、`environment/status`、`thread/environment/connected|disconnected` 和 `thread/*` current 方法，不新增协议或生产 mock。
- 实现：`thread/start` 将规范化环境选择和当前 world-state snapshot 持久化到 Thread metadata；canonical Thread read projection 统一输出 `environment_selections`（identity、cwd、workspace roots、primary、status、shell、error），session 兼容 projection 仅在 canonical 字段缺失时回填一次。
- 实现：GUI 环境生命周期 hook 只从 canonical Thread selection 建立订阅和首帧状态；按语义 key 去重 Thread read 刷新，接收通知时严格校验 `threadId`，status 查询保留服务端 error；环境 popover 显示断线错误，不回退 mock。
- 实现：`thread/unarchive` 和 `thread/fork` 从持久化 Thread metadata 恢复 Environment registry 绑定，并向目标 Thread 发出 connected/disconnected 通知；`thread/archive` 的解除绑定行为保持不变。
- 稳定回归：canonical Thread projection、session read projection、environment lifecycle hook/status reader、Environment popover 定向测试通过；新增 archive/unarchive 与 remote turn 的 environment world-state 断言。
- 验证：前端 `typecheck`、i18n、受影响文件 lint、Prettier 和 `git diff --check` 通过；`npm run test:contracts` 通过（app-server-client 299 checks）；`npm run smoke:agent-runtime-current-fixture` 与最终 `npm run verify:gui-smoke` 通过，最新 Gate B summary 为 `.lime/qc/project-gates/standalone-shell-01-20260830024741-61858/shell-01-electron-smoke/summary.json`（21/21 assertions，Electron IPC、App Server JSON-RPC、无 legacy/mock fallback）。Electron smoke 同时成功准备本地 app-server sidecar。workspace `npm run lint -- --no-warn-ignored` 仍受未改动的 `useAgentRuntimeSyncEffects.ts:634` 缺失依赖 warning 阻塞。Rust `cargo fmt --all -- --check` 通过；独立 `cargo check -p app-server` 因本机 V8 `v150.4.0` prebuilt archive URL 返回 404 未完成编译，但 Electron sidecar 构建路径已成功复用并产出 app-server。
- 未完成：D0 视觉基线仍受 AppleEvent `-1743` 阻塞；D2 尚需 changes workbench 和 `thread/start|read|resume|fork|archive|unarchive|revert`、`turn/start|steer|interrupt` 的真实 Electron Gate B 全路径证据；D3 runtime side panels 与 D4 平台/存储语义仍未开始，计划不可标记完成。

## 16. 2026-08-30 D2 第二阶段：Changes Workbench Git 摘要投影

- 范围：在既有 `CanvasWorkbenchChangesPanel` / `projectGit/diff` current owner 内补齐 Codex Desktop 风格的变更上下文摘要；不新增协议、Git 写操作或生产 mock。
- 实现：`projectGit/diff` 返回的 `currentRef`、`comparisonBaseRef`、`repositoryRoot`、`uncommittedFileCount` 与解析后的文件变更统一进入 changes toolbar；筛选后的文件树同时显示文件数和增删行统计。上一轮对话基线不伪造 Git 元数据，Git 读取失败时清空投影并保留显式错误。
- 视觉收口：摘要与审查基线、文件树、diff 统计保持同一 toolbar 层级；使用实体白底、浅边框、紧凑等宽 metadata，避免重复的环境卡片或隐藏主操作。
- 稳定回归：新增 `summarizeCanvasWorkbenchChanges` 纯投影测试；Canvas Workbench 组件补充 current ref、comparison base、文件数和未提交数 DOM 断言；五语言新增 `summary.base/files/uncommitted` 文案并通过完整性检查。
- 验证：`npm run typecheck`、`npm run i18n:check:json`、目标文件 ESLint、`git diff --check`、changes ViewModel 与 Canvas Workbench 组件共 18 项测试通过。`npm run test:contracts` 通过（app-server-client 299 checks，mock priority commands 0）；`npm run smoke:agent-runtime-current-fixture` 通过（liveProviderUsed=false，包含 coding workbench、active steer、approval、Plan history、MCP/Skills 等 Electron 场景）；`npm run verify:gui-smoke` 通过，最新 Gate B summary 为 `.lime/qc/project-gates/standalone-shell-01-20260830031004-96406/shell-01-electron-smoke/summary.json`。`npm run test:related` 当前被 related 扫描器将 `electron/` 目录当作文件读取（`EISDIR`）阻塞，已改用直接 Vitest 路径完成定向回归。
- lifecycle 现状：current client 已有 `thread/start|list|read|resume|fork|archive|unarchive`、`thread/revert` typed gateway 与 `turn/start|steer|interrupt` 接线；本阶段未新增兼容 wrapper。仍缺一个专用真实 Electron Gate B 场景，证明这些动作的 response/notification/read model 与 GUI 共享同一 Thread/Turn/Item identity。
- 当前完成度：D2 从第一阶段推进到 changes projection 第二阶段；整体计划保持 active，不进入 release evidence。D0 AppleEvent `-1743`、D3、D4 和 lifecycle 全路径 Gate B 仍为 OPEN_REF。

## 17. 2026-08-30 D3 第一阶段：Thread Activity 右侧面板

- 范围：沿用现有 Right Surface registry、canonical `threadItems`、子 Thread roster 与 tool inventory；补齐 Codex Desktop 风格的 Thread activity side panel，不新增协议、第二套 runtime owner 或生产 mock。
- 实现：新增 `activity` Right Surface。面板统一展示 canonical 子任务状态、subagent activity item、MCP server/tool 摘要和 `skill_execution` 工具；活动为空时给出稳定空态。面板从 Thread header 工具栏进入，与 Harness 保持互斥，详情只读，不承接 send/approval 等主操作。
- 接线：`WorkspaceRightSurfaceHostRuntime`、Right Surface 状态/切换动作、Task Center toolbar 和五语言资源同步更新；activity 入口只在存在 active Thread 时启用，数据全部来自当前 Thread/read model。
- 稳定回归：`ThreadActivityPanel.test.tsx`、Right Surface registry/projection/chrome 与 Workspace Scene 定向测试共 38 项通过；`npm run typecheck`、`npm run i18n:check:json`、受影响文件 ESLint 通过。
- 未完成：D3 仍需 MCP/Skills `list_changed` 真实 Gate B、长历史窗口/虚拟化和冷启动恢复证据；D2 lifecycle 全路径 Gate B、D0 AppleEvent `-1743`、D4 平台/存储语义仍为 OPEN_REF。
- 当前完成度：D3 进入 in-progress（activity side panel 已落地）；整体计划约 36%，继续保持 active，不进入 release evidence。

## 18. 2026-08-30 D3 Activity 专用 Electron Gate B 回归

- 范围：为已落地的 Thread Activity side panel 增加独立的 `thread-activity-panel` current fixture 场景，避免依赖仍引用历史 Object Canvas 入口的旧全矩阵；不恢复旧 toolbar 入口、不新增协议或生产 mock。
- 实现：场景创建并打开专家 Thread 后，通过真实 `task-center-activity-toggle` 打开 Activity；Gate B identity 绑定场景创建的 `sessionId/threadId`。Activity root 允许填充 right surface 的 active pane（tabs header 与 canvas panel 高度差异不再误报），并断言面板可见、root 填充 active pane、没有模型 Turn。
- 实现：`parseArgs` 白名单、scenario flow、assertion context、common/not-applicable/scenario assertions 和 smoke guard 同步接线；Activity 场景的 App Server 证据以创建后真实 `thread/read` 为边界，不伪造 `workspaceRightSurface/request`。
- 专用 Gate B 证据：`.lime/qc/gui-evidence/claw-chat-current-fixture/codex-app-gui-thread-activity-panel-regression-summary.json`；`ok=true`，`window.__LIME_ELECTRON__=true`，preload `electronAPI.invoke` 可用，App Server IPC 命中 85 次，current `thread/read` 投影与场景 Thread identity 一致，legacy command/mock fallback 均为 0；Activity `thread-activity-panel` 可见且 `rootFillsActivePane=true`，backend ledger 无 `turnStart`。
- 定向回归：Activity、Right Surface registry/projection/chrome 与 current fixture smoke guard 共 94 项测试通过；`npm run i18n:check:json`（5 locale、14 namespace、100%）、`npm run typecheck`、`git diff --check` 通过。两个既有长断言脚本保留原排版以避免无关 diff；完整 Prettier 仍报告其预存排版 warning，未扩大范围。
- 全量回归：`npm run verify:gui-smoke` 通过，最新 summary：`.lime/qc/project-gates/standalone-shell-01-20260830041534-78027/shell-01-electron-smoke/summary.json`；`npm run smoke:agent-runtime-current-fixture` 全部通过，`liveProviderUsed=false`。
- 旧验证缺口：`right-surface-visual-matrix` 仍包含历史 `task-center-object-canvas-toggle` 断言，而 current toolbar 已不提供该入口；该缺口与 Activity 接线无关，不能通过恢复退役入口来“修复”，应在后续 D3 surface cleanup 中重写或删除旧矩阵。
- 当前完成度：D3 Activity 第一阶段具备真实 Electron Gate B 证据；整体计划约 40%，继续保持 active，不进入 release evidence。下一刀为 MCP/Skills `list_changed`、长历史 paging/virtualization 与 cold restart recovery。

## 19. 2026-08-30 D3 第二阶段：MCP `list_changed` catalog 刷新

- 范围：沿用 current MCP tool-progress 通知和 `useMcp` catalog owner，对齐 Codex Desktop 在 MCP 能力变化后的实时列表刷新；不新增第二套 MCP 事件通道、不恢复 legacy mock fallback。
- 实现：`McpToolCallProgressNotification` 增加可选 `notificationKind`；Rust v2 MCP 投影接受 `mcp_resources_changed`、`mcp_tools_changed`、`mcp_prompts_changed` 三类 list-change kind，并保持普通 `mcp_progress` 的兼容序列化形状。
- 实现：App Server renderer 投影保留并校验 `notificationKind`；`useMcpEvents` 通过真实 App Server notification bus 分别触发 `refreshResources`、`refreshTools`、`refreshPrompts`，未知 kind fail closed，不刷新任何 catalog。
- 稳定回归：新增 Rust v2 projection、public JSON-RPC、协议 round-trip 和 schema fixture 覆盖；`useMcp` 验证三类 catalog 各刷新一次且未知 kind 不触发；App Server v2 renderer projection 验证 metadata 保留。前端定向测试 59 项通过，协议 round-trip 2 项通过。
- 验证：`npm run typecheck`、`npm run check:protocol-types`、`npm run i18n:check:json`、`git diff --check` 通过；`cargo fmt --all -- --check` 与 `cargo test --test schema_fixtures -p app-server-protocol` 通过。`npm run test:related` 仍被既有 related 扫描器把 `electron/` 目录当文件读取（`EISDIR`）阻塞，已用直接 Vitest 路径完成受影响 Hook/投影测试。App Server crate 测试无法启动，受本机 `rusty_v8 v150.4.0` aarch64 macOS 预编译 archive URL 404 阻塞。
- Gate B 边界：本轮协议与 Hook 已具备 current App Server notification 接线和 unit/domain evidence，但尚未新增真实 Electron fixture 证明 MCP server 实际发出三类 list-change 后 GUI Activity/catalog 同屏更新；因此不能把本轮标为完整 Gate B。
- 未完成：D3 仍需 MCP/Skills list-change 真实 Electron Gate B、长历史 paging/virtualization、冷启动恢复；D2 lifecycle 全路径 Gate B、D0 AppleEvent `-1743`、D4 平台/存储语义仍为 OPEN_REF。
- 当前完成度：D3 从 Activity 第一阶段推进到 MCP list-change current contract 第二阶段，整体计划约 44%，继续保持 active，不进入 release evidence。下一刀优先是把 Skills change 刷新提升为 Activity/catalog 可见的真实 Electron evidence，随后处理长历史和冷启动恢复。

## 20. 2026-08-30 D3 Skills `changed` Electron Gate B 证据

- 复用既有 `skills-runtime` current fixture，不新增平行场景或测试专用生产入口；该场景通过真实 Electron Desktop Host、preload/contextBridge、`app_server_handle_json_lines` 和 App Server `skills/list` 完成 Skill catalog 回填。
- 证据：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-skills-runtime-list-change-regression-summary.json`，`ok=true`，`proofLevel=Gate B controlled fixture`，`skills/changed` marker 计数 1，自动 `skills/list` 通过 `electron-ipc` 增量 1，新增 `notification-refresh` 在 selector 可见，手动刷新点击数 0，console errors 0。
- 该证据证明 Skills change 的 current bridge 与 GUI catalog 自动刷新闭环；不把它扩大解释为 live provider 或所有 MCP server list-change 的普遍证明。
- 验证：`npm run smoke:claw-chat-current-fixture -- --scenario skills-runtime --prefix claw-chat-current-fixture-skills-runtime-list-change-regression --timeout-ms 240000` 通过；此前 `npm run test:contracts`（299 checks、mock priority commands 0）和 `npm run verify:gui-smoke`（standalone Electron Gate B）均通过。
- 未完成：MCP 三类 list-change 仍缺真实 Electron fixture 发射与 GUI catalog 可见证据；长历史 paging/virtualization、冷启动恢复、D2 lifecycle 全路径 Gate B、D0 AppleEvent `-1743`、D4 平台/存储语义仍为 OPEN_REF。
- 当前完成度：D3 Skills change 已从 contract 级推进到真实 Electron Gate B，整体计划约 47%，继续保持 active，不进入 release evidence。下一刀优先补 MCP list-change Gate B，再处理长历史和冷启动恢复。

## 21. 2026-08-30 D3 Thread resume 订阅恢复接线

- 根因：页面 reload 或从侧栏重新打开已有 Thread 时，GUI 只通过 `thread/read` hydrate canonical read model；当前连接没有重新执行 `thread/resume`，因此 runtime 已持久化的 MCP `list_changed` 通知不会进入 Electron notification observer。
- 实现：`useAgentSession` 的唯一 Thread 打开/切换详情入口在读取 canonical `threadId` 后调用现有 `AgentRuntimeAdapter.resumeThread`，重新建立 App Server 订阅；只读刷新与历史分页仍保持 `thread/read`，不引入 legacy wrapper 或 mock fallback。Adapter 在没有 replay consumer 时允许 metadata-only `thread/resume` 成功，供 GUI 订阅恢复使用。
- 回归：更新首页恢复、排队会话和 running Thread 的 hook 断言，要求通过真实 App Server client 形状 `{ threadId }` 调用 `thread/resume`，并保留“不自动启动 Turn”的断言。
- 验证：`useAgentChat.test.tsx` 186 项、`agentRuntimeAdapter.test.ts` 151 项通过；下一步重跑 `test:contracts` 与 MCP list-change Electron Gate B，确认 typed `mcpToolCall/progress` 进入 renderer recent buffer。
- 当前完成度：实现已落地，D3 MCP list-change Gate B 仍待真实证据；整体计划保持 active，约 49%。

## 22. 2026-08-30 D3 MCP `list_changed` Electron Gate B 完成

- 根因收口：真实 Electron Host 已收到 App Server 的三条 `item/mcpToolCall/progress`，但 Gate B fixture 直接调用 preload 时遗漏了主进程约定的 `{ request: ... }` 包络，导致 `app_server_drain_events` 使用默认参数读取空队列。修正 fixture 包络后，renderer 可从 Host recent buffer 读取同一轮通知。
- 实现：`scripts/electron/mcp-list-changed-gate-b.mjs` 通过真实 Electron `electronAPI.invoke("app_server_drain_events", { request })` 取证，并仅保存返回结构、method 与通知 kind；`electron/appServerHost.ts` 保留受 `LIME_DEBUG_MCP_NOTIFICATIONS` 控制的收包/排空诊断，不改变生产路径。
- 正式证据：`.lime/qc/gui-evidence/mcp-list-changed-electron-gate-b/mcp-list-changed-electron-gate-b-final-scrub-summary.json`，`ok=true`、`proofLevel=Gate B controlled fixture`；同一 `threadId`/`turnId`/`itemId` 下，`notifications/tools|prompts|resources/list_changed` 映射为 `mcp_tools_changed`、`mcp_prompts_changed`、`mcp_resources_changed`，动态 tool/prompt/resource 均在 MCP 设置页可见。
- Bridge 证据：`window.__LIME_ELECTRON__=true`、preload `electronAPI.invoke` 可用、`app_server_handle_json_lines` 命中、`app_server_drain_events` recent 返回 28 条消息，`mockFallbackHitCount=0`、`failedInvokeCount=0`；console/page error 均为 0。
- 验证：`npm run electron:build:host:dev`、`node --check scripts/electron/mcp-list-changed-gate-b.mjs`、官方 `npm run smoke:mcp-list-changed-electron-gate-b`、`npm run test:contracts`（299 checks）、相关前端 Vitest（99 tests）、`npm run i18n:check:json`、`cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check`、`npm run test:rust:related`、`npm run smoke:agent-runtime-current-fixture` 和 `npm run verify:gui-smoke` 均通过；standalone Electron smoke 最新 summary 为 `.lime/qc/project-gates/standalone-shell-01-20260830085839-11471/shell-01-electron-smoke/summary.json`。
- 本地总门禁：`npm run verify:local` 在全仓 `npm run lint -- --max-warnings 0` 阶段被未改动文件 `src/components/agent/chat/hooks/useAgentRuntimeSyncEffects.ts:634` 的既有 `react-hooks/exhaustive-deps` warning 阻断；本轮受影响文件定向 lint/typecheck 与其余等价门禁均通过。
- 边界：该证据证明受控 MCP server 的三类 list-change 从 RuntimeCore 到 Electron/GUI catalog 的闭环，不宣称 live provider、打包发布产物或所有远程 MCP 实现已完成。
- 当前完成度：D3 MCP list-change 从“待 Gate B”推进为已完成；整体计划约 52%，继续保持 active，不进入 release evidence。剩余下一刀为长历史 paging/virtualization 与冷启动恢复，另有 D2 lifecycle 全路径 Gate B、D0 AppleEvent `-1743`、D4 平台/存储语义 OPEN_REF。

## 23. 2026-08-30 D3 长历史与进程级冷启动恢复完成

- 范围：复用 canonical `thread/items/list`、`thread/turns/list` opaque cursor 和既有 session-history Electron fixture；不新增第二套 virtualizer、history store、runtime owner、私有 handler 入口或生产 mock fallback。
- 实现：打开持久化 Thread 后，`useAgentSession` 先通过 `thread/read` hydrate canonical detail，再调用现有 `thread/resume` 恢复当前 Electron/App Server connection 的 Thread 订阅。fixture 首次从真实侧栏打开 240 Turn / 720 Item Thread，清空 trace 后关闭 Electron/App Server，再用同一隔离 user-data 启动新进程并从侧栏重新打开同一 Thread。
- 正式证据：`.lime/qc/gui-evidence/agent-session-history-electron-fixture/codex-app-gui-long-history-cold-restart-final-summary.json`，`ok=true`、`proofLevel=Gate B controlled fixture`。首次和冷启动均使用同一 `sessionId/threadId=01a05240-c622-7832-8381-7b8d8f028974`，首帧均为 10 个 canonical Turn、30 个 Item、0 residual Message；冷启动 `navigation=sidebar`、`processRestart=true`，并重新命中 `thread/read`、`thread/items/list`、`thread/turns/list`、`thread/resume`。
- 性能与错误：冷启动 `clickToFirstMessageListPaintMs=65`、`messageListComputeMaxMs=0`、`longTaskCount=0`，console/page error 均为 0。截图：`.lime/qc/gui-evidence/agent-session-history-electron-fixture/codex-app-gui-long-history-cold-restart-final-long-list-cold-restart.png`。
- 稳定回归：session-history smoke guard 6 项、pagination controller 7 项、`useAgentChat` / adapter / MessageList 相关测试 224 项通过；`npm run typecheck`、`npm run test:contracts`（app-server-client 299 checks、mock priority commands 0）、Prettier 与 `git diff --check` 通过。测试仍输出既有 React `act(...)` warning，无失败。
- 全量主路径：`npm run smoke:agent-runtime-current-fixture` 全部通过，`liveProviderUsed=false`；`npm run verify:gui-smoke` 21/21 assertions 通过，最新 summary 为 `.lime/qc/project-gates/standalone-shell-01-20260830104901-60115/shell-01-electron-smoke/summary.json`，覆盖 Electron/preload、`electron-ipc`、App Server current method、reload、无 legacy/mock fallback 和 console/page/invoke error 为 0。
- 边界：该 Gate B 证明受控、完整数据下的进程级冷启动和有界首帧，不证明 crash 中断、partial JSONL 修复、Windows packaged 行为或 live provider；这些分别留在 D4/平台证据。Codex Desktop 实时视觉基线仍受 AppleEvent `-1743` 阻塞，静态资源不冒充实时视觉证据。
- 当前完成度：D3 Runtime surfaces 的 Activity、Skills/MCP change、长历史和冷启动退出条件已满足，D3 标记完成；整体计划约 58%，继续保持 active，不进入 release evidence。下一刀为 D2 lifecycle 全路径 Gate B，之后进入 D4 ThreadStore crash recovery / permission provenance / Multi-Agent durable semantics。

## 24. 2026-08-30 D2 Thread Fork 与 lifecycle Gate B 矩阵完成

- 范围：补齐 D2 最后一个 Fork 产品闭环，并复核 `thread/start|read|resume|fork|archive|unarchive|revert` 与 `turn/start|steer|interrupt` 的既有 Gate B；不新增协议、第二套标题事实源、legacy `agentSession/*` wrapper 或生产 mock fallback。
- Rust current owner：移除 paginated source 的 Fork 拒绝分支，继续复用 canonical history/materialization owner。公共 JSON-RPC 回归证明 Fork 截止到指定 Turn、source/target Thread 与 Session identity 独立、标题和 `forkedFromId` 保留，并在 App Server 冷启动后通过 `thread/resume|read|list|turns/list` 保持一致。
- Renderer 根因与修复：`thread/fork` response 原本已有 canonical 标题，但 session gateway 只向侧栏广播新 Session ID，首轮 `thread/list` 并发窗口会先插入“未命名对话”。内部 gateway 现返回 Fork canonical 摘要，公共 `forkAgentRuntimeSession(): Promise<string>` 合同不变；`sessions-changed` 的 created detail 携带 `name/threadId/workingDir`，侧栏直接建立正确 optimistic entry，后续仍由 canonical list/read 收敛。
- 专用 Gate B：`.lime/qc/gui-evidence/thread-fork-electron-gate-b/codex-app-gui-thread-fork-final-summary.json`，`ok=true`、`proofLevel=Gate B controlled fixture`。真实 Electron header 菜单命中 `electron-ipc -> app_server_handle_json_lines -> thread/fork -> thread/read|resume`；source/target 同名但 identity 不同，child 成为 sidebar/header/composer active Thread，`thread/started` 与 read model 保留同一 `forkedFromId`，mock/invoke/console/page error 均为 0。
- lifecycle 矩阵：archive/unarchive 由 `.lime/qc/gui-evidence/agent-session-history-electron-fixture/agent-session-history-electron-fixture-summary.json` 证明 notification、rollout 移动和进程重启 readback；revert 由 `.lime/qc/gui-evidence/thread-revert-electron-gate-b/thread-revert-electron-gate-b-summary.json` 证明；turn start/steer/interrupt 分别由 current fixture 的普通回合、`claw-chat-current-fixture-active-steer-regression-summary.json` 和 `claw-chat-current-fixture-cancel-then-continue-regression-summary.json` 证明。所有场景均经过真实 Electron/preload/App Server current bridge，mock fallback 为 0。
- 定向验证：前端 `appServerSessionClient`、`sessionClient`、`AppSidebar` 共 88 项与 Fork Gate B oracle 6 项通过；`thread_fork_jsonrpc` 4/4 通过；`npm run typecheck` 和受影响文件 ESLint 通过。`npm run test:related` 仍被 related 扫描器将 `electron/` 目录当文件读取的 `EISDIR` 阻塞，已用直接 Vitest 精确入口替代，不是测试断言失败。
- 扩大门禁：`npm run test:contracts`、`npm run governance:scripts`、`npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke` 均通过。最新 standalone Electron smoke：`.lime/qc/project-gates/standalone-shell-01-20260830132244-22700/shell-01-electron-smoke/summary.json`。
- 边界与完成度：该组证据不启动 live provider，不证明 packaged/Windows 或 ThreadStore crash repair。D2 退出条件已满足并标记完成；D0 仍受 AppleEvent `-1743` 阻塞，整体计划约 68%，下一刀进入 D4 ThreadStore crash recovery、partial JSONL/ordinal repair 与 writer flush/shutdown/discard。

## 25. 2026-08-30 D4 ThreadStore crash-tail repair 与 writer lifecycle 完成

- 范围与参考：只修改 App Server 的 canonical rollout current owner，对照 Codex `rollout/src/recorder.rs` 的 `ensure_rollout_is_newline_terminated`、`open_rollout_for_append`，以及 `thread-store/src/local/{live_writer,thread_history_materialization}.rs`；不恢复 legacy history owner、不新增 renderer fallback、后台 writer 或无消费者 lifecycle API。
- crash repair：所有 current rollout 读写入口先经过单一 `tail_repair`。文件以换行结束时保持原快速路径；完整且语义有效的 unterminated record 在全链 identity/digest/sequence 校验后补换行并 `flush + sync_data`；无法解析的末尾残片先校验完整前缀，再按最后一个换行的 byte offset 精确截断并 `sync_data`。首条记录残缺、完整但身份/digest 无效、中段 malformed、sequence 回退或 divergence 均不修改源文件并 fail closed。
- ordinal/offset 语义：Lime canonical ordinal 继续取外层 `AgentEvent.sequence`，不在 store 内重新编号。受控 crash fixture 证明丢失的 partial sequence 2 可截断，后续 sequence 3 按原 ordinal 继续，冷启动 snapshot 仍能 materialize `[1, 3]`；合法 sequence 2 仅缺 delimiter 时则保留并形成 `[1, 2, 3]`。
- writer lifecycle 裁决：Lime current 没有 Codex 的 deferred in-memory recorder。每次 `ensure_thread`、history append 和 metadata patch 在返回前已执行 `write_all -> flush -> sync_data`，`File` 随调用结束关闭；session teardown 由既有 `session_loop.shutdown` 承担，discard 由 `thread/delete -> RolloutStore::delete` 承担。因此不新增语义为空的 `persist/flush/shutdown/discard` ThreadStore API；原子 delete retry 回归继续证明外部文件清理与数据库删除不会留下半提交状态。
- owner 回归：`cargo test -p app-server canonical_rollout` 12/12，通过新增的合法尾部、非法尾部、sequence gap、冷 snapshot、中段损坏与完整语义损坏 5 类场景；`canonical_thread_store_tests` 38/38，`runtime::tests::thread_delete` 1/1。标准 `npm run test:rust:related -- <canonical-rollout-paths>` 运行 App Server 全量 unit，1729/1729 通过。
- 跨层验证：`npm run test:contracts` 通过（typed client 299 checks、mock priority 0、scripts governance/docs boundary 全绿）；`npm run smoke:agent-runtime-current-fixture` 全部通过，`liveProviderUsed=false`，覆盖 Electron 中的多轮 append、stop/continue、active steer、Plan/history hydrate、Skills/MCP 与 read-model reopen；`npm run verify:gui-smoke` 通过，最新 Gate B summary 为 `.lime/qc/project-gates/standalone-shell-01-20260830140019-5589/shell-01-electron-smoke/summary.json`。
- 治理分类：canonical rollout、SQLite rebuild projection、session-loop shutdown 与 `thread/delete` 均为 `current`；本轮 `compat/deprecated/dead` 新增为 0。未触碰并行 MCP/Right Surface/Claw 源码热区和未跟踪 `local-conversation-page-B5LUHmAw.js`。
- 边界与完成度：该证据是 macOS 受控 partial-file/cold-store fixture 与真实 Electron current 主链，不宣称真实断电、Windows packaged filesystem 或 live provider。D4 ThreadStore 第一阶段标记完成，整体约 73%；下一刀为 permission profile policy/runtime/GUI provenance，之后是 Multi-Agent durable residency 与真实 Windows enforcement。

## 26. 2026-08-30 D4 Permission Profile provenance 第一阶段完成

- 范围：复用唯一 `permission_profile` built-in resolver，不引入 Codex MDM、requirements/config.toml、project-local custom profile 或第二套 policy owner；动态 `allowed`、cwd policy 计算和 `thread/settings/update.permissions` 继续列为后续差距。
- App Server：`thread/start.permissions` 先经 resolver 校验，成功后把 `permissions` 与 `{ "id": ... }` 的 `activePermissionProfile` 写入同一 canonical Thread metadata，并从相同 metadata 返回 `thread/start.activePermissionProfile`；unknown profile 在生成 Thread 前以 `INVALID_PARAMS` fail closed。`turn/start.permissions` 继续由同一 resolver lowering 为 `read-only`、`workspace-write` 或 `danger-full-access` sandbox policy，并写入 Turn metadata。
- Renderer：新会话创建从当前 access mode 生成内建 profile id，经 `appServerSessionClient -> thread/start` 透传；不在 Electron 或 Renderer 建立本地 catalog。已有 `permissionProfile/list` 仍通过 App Server JSON-RPC，catalog 缺失、重复、禁止或形状非法时继续 fail closed。
- owner 回归：公共 `permission_profile_jsonrpc` 新增 Thread start provenance 与 unknown profile 负向场景；前端 `appServerSessionClient` 与 `permissionProfiles` 定向测试共 29 项通过；Rust 文件级 rustfmt 与 `git diff --check` 通过。Rust crate 编译被当前 macOS `rusty_v8 v150.4.0` 预编译包缺失（GitHub release 返回 404）阻断，未能重新执行 crate integration。
- 质量边界：全量 `tsc --noEmit` 仍被工作树既有 fixture/protocol 类型错误阻断；本轮新增前端测试无失败。`npm run test:contracts`、`npm run smoke:agent-runtime-current-fixture` 与 `npm run verify:gui-smoke` 需在下一轮合并窗口重新执行，不能用本轮定向测试替代跨层证据。
- 治理分类：`permission_profile`、Thread/Turn lowering、session gateway 为 `current`；未新增 `compat/deprecated/dead`。动态 policy producer、cwd/managed requirements 和 Windows enforcement 仍为 D4 P1 OPEN_REF。
- 边界与完成度：本阶段证明 profile identity 在新 Thread 创建、Turn sandbox lowering、canonical metadata 和 GUI access-mode 之间使用同一 resolver，不证明 Codex 的动态 allowed/custom profile/MDM 语义。D4 Permission Provenance 第一阶段标记完成，整体约 76%；下一刀是动态 permission policy 与 Multi-Agent durable residency，随后补真实 Windows enforcement。

## 27. 2026-08-30 D4 Permission Settings Mutation 第二阶段完成

- 范围：把既有 `thread/settings/update.permissions` 接入第一阶段的唯一 built-in resolver；不新增 App Server method、Electron 权限命令、Renderer catalog、production mock fallback、custom profile 或 managed requirements owner。
- App Server：`thread/start`、`turn/start` 与 settings mutation 共用 `permission_profile` owner。settings 更新在同一次 metadata patch 中持久化 `permissions`、`activePermissionProfile` 和 lowering 后的 `sandboxPolicy`；未知 profile 在持久化前 fail closed。legacy `sandboxPolicy` 更新会清除旧 profile provenance，避免 canonical Thread 同时保留矛盾事实。
- Renderer/projection：GUI access mode 保存从 legacy `approvalPolicy + sandboxPolicy` 收敛为 `approvalPolicy + permissions`；`thread/settings/updated.activePermissionProfile.id` 可回填 live settings snapshot，不在组件或 Electron 重建映射。
- owner 回归：前端 adapter/useAgentChat 210/210、agent-runtime-projection 293/293、App Server unit 1731/1731、permission profile 相关 unit 5/5 与 public JSON-RPC integration 4/4 通过；受影响文件 ESLint、Prettier、文件级 rustfmt、全量 TypeScript typecheck 与 `git diff --check` 通过。全 workspace `cargo fmt --all -- --check` 仅被避让热区 `tests/thread_fork_jsonrpc.rs` 的既有格式漂移阻断，本轮 Rust 写集无格式差异。
- 跨层门禁：`npm run test:contracts` 通过（generated protocol 无漂移、app-server-client 299 checks、mock priority 0）；`npm run smoke:agent-runtime-current-fixture` 全矩阵通过，`liveProviderUsed=false`；`npm run verify:gui-smoke` 21/21 通过，最新 summary 为 `.lime/qc/project-gates/standalone-shell-01-20260830145449-27148/shell-01-electron-smoke/summary.json`。
- 权限 Gate B：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-approval-request-full-access-regression-summary.json` 为 `Gate B controlled fixture`，真实 Electron GUI 命中 `thread/settings/update` 与 `permissionProfile/list`；wire profile 为 `:danger-full-access`，legacy `sandboxPolicy` 未上行，RuntimeCore 投影 `sandboxPolicy=danger-full-access`、`activePermissionProfileId=:danger-full-access`，两侧一致，legacy/mock/console/page error 均为 0。
- 治理分类：permission resolver、Thread/Turn/settings metadata patch 和 Renderer live projection 均为 `current`；本轮没有新增 `compat/deprecated/dead`。旧 `sandboxPolicy` settings 输入只保留协议兼容语义，并在使用时主动清理 profile provenance，不是 GUI current 写入口。
- 边界与完成度：该阶段证明三个内建 profile 的 GUI settings mutation、public JSON-RPC、canonical readback、Turn runtime lowering 与真实 Electron 可见行为闭环；不证明动态 `allowed`、custom profile、deny-read、cwd/managed requirements、Windows packaged enforcement 或 live provider。D4 Permission Settings Mutation 第二阶段标记完成，整体约 79%；下一刀为真实 permission policy producer，之后处理 Multi-Agent durable residency 与 Windows enforcement。

## 28. 2026-08-30 D4 Dynamic Permission Policy 第三阶段完成

- 范围与裁决：对照 Codex 的 permission profile `allowed` producer，但只接入 Lime 已有真实 owner：全局 `agent.workspace_sandbox`、请求 cwd 物化、当前平台 sandbox backend 与 Windows setup artifact readiness。不复制 Lime 没有 consumer 的 MDM、`requirements.toml`、project-local `.codex/config.toml`、custom profile 或 deny-read 产品面。
- App Server：`permissionProfile/list`、`thread/start`、`turn/start` 与 `thread/settings/update` 共用 `RuntimeCore::current_permission_profile_policy`。strict 模式要求受限 backend 且 backend 不可用时，`:read-only`、`:workspace` 返回 `allowed=false` 并在写入口 fail closed；`:danger-full-access` 保持可选。`thread/start` 同时持久化 lowering 后的 `sandboxPolicy`，避免 profile provenance 与 runtime policy 分离。
- Renderer：Turn 提交前通过 `src/lib/api/permissionProfiles.ts` 查询唯一且允许的 profile，并把当前 Thread `workingDir` 规范化为 cwd 传给 `permissionProfile/list`；不在组件、Electron 或本地状态中复制策略计算，也不恢复 legacy `sandboxPolicy` 上行。
- owner 回归：Permission 前端定向测试 12/12、Rust policy unit 9/9、public JSON-RPC integration 4/4、全量 TypeScript typecheck、受影响文件 ESLint/Prettier、Rust 文件级 rustfmt 与 `git diff --check` 通过。公共 integration 使用隔离 `RuntimeCore::with_app_config_path(...)`，未读取真实用户配置。
- 跨层门禁：`npm run test:contracts` 通过（generated protocol 无漂移、app-server-client 299 checks、mock priority commands 0）；`npm run smoke:agent-runtime-current-fixture` 全矩阵通过且 `liveProviderUsed=false`；`npm run verify:gui-smoke` 通过，最新 Gate B summary 为 `.lime/qc/project-gates/standalone-shell-01-20260830155248-31273/shell-01-electron-smoke/summary.json`。
- 治理与证据边界：permission policy、Thread/Turn/settings lowering、Renderer cwd request builder 均为 `current`；本轮没有新增 `compat/deprecated/dead` 或生产 mock fallback。macOS Gate B 证明真实 Electron/preload/IPC、`app_server_handle_json_lines`、App Server/runtime/read model 与用户可见 GUI 可启动，不证明 live provider、真实 Windows/MSVC restricted execution 或 packaged setup artifact。
- 完成度：D4 Dynamic Permission Policy 第三阶段完成，整体约 82%。下一刀进入 Multi-Agent durable ack/retry/residency；真实 Windows packaged enforcement 继续保留为 D4 P0 平台证据缺口。

## 29. 2026-08-31 D4 Multi-Agent durable semantics 第四阶段完成

- 根因与裁决：parent-owned child 原先只在 session metadata 中保存 provider/model，canonical Thread 的 `model_provider` 为空；GUI 按 current 语义调用 `thread/resume` 时收到 `modelProvider` 非空校验错误并回退父 Thread。修复不绕过 `thread/resume`，而是在 AgentControl child 首次 `start_session` 时写入统一的 provider/model/route/service-tier metadata，使 canonical Thread 从创建开始具备可恢复身份。
- App Server：新增 `agent_control_session_default_metadata()` 作为唯一默认 metadata producer；child `BusinessObjectRef`、首次 session 建立和已有 metadata 持久化均复用它。parent-owned child 仍只允许 `thread/read|resume`，`turn/start`、steer、settings、compact、shell 等直接输入继续由 App Server fail closed。
- 回归：restart unit 新增 canonical child `model_provider` 断言；`thread_direct_input_policy_jsonrpc` 新增 parent-owned child `thread/resume` 成功与 `canAcceptDirectInput=false` 断言。`npm run test:rust:integration -- -p app-server --test thread_direct_input_policy_jsonrpc` 通过，App Server profile 1735 个 unit、26 个 main、全部 integration suites 均为 green；`npx vitest run scripts/agent-runtime/tool-execution-smoke.test.mjs` 21/21 通过。
- 真实 Electron Gate B：`codex-app-gui-d4-agent-control-cold-restart-final.json` 证明 child 在 Electron 进程替换后通过 `thread/resume` 保持 identity、Activity 和只读 composer；`codex-app-gui-d4-agent-control-capacity-final.json` 证明并发 child 数量限制和拒绝路径；`codex-app-gui-d4-agent-control-residency-final.json` 证明 terminal slot reuse、LRU cold reload、follow-up 复用既有 child identity。三组均为 `status=pass`、真实 `electron-ipc`、console/invoke error 为 0、生产 mock fallback 为 0。
- 诊断与治理：managed smoke 仅增加 invoke error 与 parent-owned child raw `thread/resume` probe 的失败诊断，不改变生产协议或 GUI 行为；`agent-control`、Thread/Turn/Item projection、activity 和 durable graph/mailbox 均保持 `current`，未新增 `compat/deprecated/dead`。首次 residency 运行有一次 child wait 时序失败，重跑后完整 Gate B 通过，证据以独立输出路径保存。
- 边界与完成度：Multi-Agent durable ack/retry/residency 阶段标记完成，整体约 86%。本阶段为 macOS 受控 fixture，不宣称真实 Windows/MSVC packaged enforcement、live provider 或 Codex Desktop 实时视觉；下一刀为 Windows sandbox/ACL/ConPTY/网络隔离平台证据，D0 AppleEvent `-1743` 仍保持 blocked。
