# Codex App GUI 对齐执行计划

> status: active / D0-desktop-baseline-blocked -> D2-D3-complete -> D4-multi-agent-durable-complete -> D4-windows-enforcement-phase
> owner: agent-ui + app-server
> started: 2026-07-18
> last reviewed: 2026-09-02
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
- 下一刀：完成 `D4 Runtime parity` 的 Windows packaged sidecar/runtime 与 Gate B 证据。真实 `windows-2022` CI 已证明 ACL/ConPTY/网络隔离安全矩阵；custom permission profile、deny-read 与 MDM/project-local requirements 保持 product-scope excluded，除非先建立真实 consumer 和唯一 owner。
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
- 首页首屏的皮肤 Hero、Banner、营销文案和技能入口按用户裁决保留，不纳入 Codex App 对齐范围；Codex 对齐只作用于已进入 Thread 的工作区。
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

| 领域                  | Codex-rs 参考能力                                                                                                                 | Lime current                                                                                                                                          | 尚未对齐                                                                                                                                       | owner / 优先级                      |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| Windows sandbox       | restricted token、ACL/DACL read-back、ConPTY、Job Object、网络隔离和 packaged resource 守卫                                       | `windowsSandbox/setupStart` 等协议与 `tool-runtime::windows_setup` 已有；真实 `windows-2022` CI security matrix 与 provision 7/7 通过                 | packaged sidecar/runtime 全路径与 Windows Electron Gate B 尚未完成                                                                             | tool-runtime + app-server / P0      |
| Permission profile    | config layer/managed requirements、动态 allowed、custom profile、deny-read 和 cwd 解析                                            | 三个内置 profile、动态 `allowed`、cwd 物化、全局 sandbox config/platform/setup readiness、Thread/Turn/settings lowering 与 GUI 提交前查询已实现       | custom profile、deny-read、MDM/project-local requirements 与真实 Windows packaged enforcement 未实现；cwd 没有 project-local policy layer      | app-server + tool-runtime / P1      |
| ThreadStore lifecycle | live writer、flush/shutdown/discard、history materialization、latest model context、paginated fork lineage、ordinal/offset repair | canonical SQLite projection、sequence/cursor、revert、paginated Fork、干净重启和 crash-tail repair 已有                                               | 已完成：合法 unterminated record 保留、非法尾部按 byte offset 截断、sequence gap 续写、中段损坏 fail closed；同步 writer 无额外 pending buffer | thread-store + app-server / done    |
| Multi-Agent           | parent-child registry、resident unload/reload、execution limiter、control tools、durable graph/mailbox                            | graph、mailbox、control gateway、activity projection、parent-owned child canonical resume、capacity/residency cold restart 均有受控 fixture 与 Gate B | 真实 Windows packaged enforcement 与跨平台资源限制仍待平台证据；Multi-Agent current 语义本身已收口                                             | agent-runtime + thread-store / done |

## 4. 差距优先级

| 优先级   | 目标                       | 当前差距                                                                                                                        | 依赖                            |
| -------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| P0       | 当前 Thread 成为主画布对象 | 全局 Sidebar 需要保留，但项目 tabs、任务 tabs 和消息区仍竞争上下文；活跃任务页头不稳定                                          | GUI-only，可先做                |
| P0       | 单一 canonical 时间线      | commentary、reasoning、tool、状态摘要存在重复渲染与过量纵向展开                                                                 | `thread/read` Item projection   |
| P0       | 单一 action-required 入口  | Approval/Plan/request-user-input 可同时出现在输入区、时间线和环境任务轨道                                                       | canonical Approval/Plan Item    |
| P0       | 写链对齐 Codex-rs          | GUI 仍由 session-oriented hooks 与 `agentSession/event/*` 历史绑定承接部分生命周期；current 主写应明确落到 `thread/*`、`turn/*` | App Server protocol/runtime     |
| P1       | Thread 绑定的环境面        | 现有环境浮层只读 project Git，且把任务轨道混入环境面；缺 environment identity/status                                            | `environment/*` + project Git   |
| P1       | 变更审查闭环               | 已有 diff/workbench，但环境面没有增删行、比较基线和进入 review 的明确动作                                                       | project Git + workbench command |
| done     | ThreadStore 崩溃恢复       | partial JSONL、byte offset/sequence gap repair、同步 writer close 与 delete/discard 已由 D4 第一阶段收口                        | thread-store + App Server       |
| P1       | Composer 稳定              | 首页与会话输入区形态、装饰和高度变化明显；运行态/排队态入口分散                                                                 | command model                   |
| P2       | 产品/诊断分层              | Harness、Trace、Shell、Browser、Workbench 在主 Toolbar 近似等权                                                                 | 右侧 surface catalog            |
| excluded | 首页 Banner / 皮肤 Hero    | 按用户裁决保留现有首页视觉，不纳入 Codex App 对齐范围                                                                           | product scope                   |

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
- GUI 对齐实现：D1、D2、D3 已完成；D4 ThreadStore、Permission provenance/settings mutation、动态 policy producer 与 Multi-Agent durable semantics 均已完成；Windows restricted execution security matrix 已取得真实 CI evidence，但 packaged sidecar/runtime 与 D0 真实视觉基线仍未完成。
- 当前总完成度：86%。D2/D3 已证明 lifecycle、Activity、MCP/Skills change、长历史分页/有界首帧和进程级冷启动恢复；D4 已补 crash-tail repair、同步 writer lifecycle、动态 permission policy，以及 AgentControl parent-child 的 canonical resume、ack/retry、capacity/residency 和 cold-restart 证据；Windows 安全矩阵已在真实 Windows/MSVC runner 通过，packaged enforcement 与 D0 实时视觉基线仍未完成。

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
- Environment/Changes/lifecycle 已由 D2 收口；MCP subscriptions/list_changed、Skills change、SubAgent Activity 和长历史干净进程重启已由 D3 收口；ThreadStore crash recovery/writer lifecycle、动态 permission policy 与 Multi-Agent durable semantics 已由 D4 收口。Windows security matrix 已有真实 CI evidence，但 packaged sidecar/runtime 与 Electron Gate B 仍是 D4 跨层差距，不能通过 GUI 文案宣称完成。

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

## 30. 2026-08-31 D4 Windows enforcement 阶段进展与 packaged 阻塞

- 现状审阅：`tool-runtime::windows_setup` 已包含账户/DPAPI artifact、ACL/DACL、重解析点拒绝、默认读取权限、Firewall/WFP、ConPTY、Job Object 与 packaged sidecar 路径；`windows_restricted_execution` 集成矩阵覆盖 workspace 写入边界、offline 网络隔离、输出上限、stdin、ConPTY、Everyone 可写 ACL 审计和进程树终止。
- 真实 Windows CI evidence：workflow run `33348108423` 的 `Windows Shell Runtime`（`windows-2022`、MSVC）已完成 provision 与安全矩阵，`summary.json` 为 `platform=win32`、`result=pass`、`setup.result=pass`、`tests.passed=7`、`matrix.complete=true`、`blockers=[]`。证据已归档到 `.lime/qc/windows-restricted-execution/github-33348108423/`，覆盖账户/ACL/网络规则、7 项 restricted execution 测试及 stdout/stderr；该证据证明安全 enforcement，不等价于 Electron packaged Gate B。
- 本机与前置回归：`npm run test:rust:unit -- -p tool-runtime windows_setup::tests` 通过 7/7，`npm run test:contracts`、`npm run typecheck`、`npm run i18n:check:json`、全量 ESLint、`npm run governance:legacy-report` 与 `npm run verify:gui-smoke` 均通过；本次 Electron smoke evidence 为 `.lime/qc/project-gates/standalone-shell-01-20260831023247-8207/shell-01-electron-smoke/summary.json`。hook 定向测试 37/37 通过；macOS evidence runner 的 `.lime/qc/windows-restricted-execution/codex-app-gui-d4-windows-final.json` 仍正确保持 `evidence-pending`，没有伪造平台结果。
- 当前边界：run `33348108423` 已最终为 `failure`：`Windows Shell Runtime` 在 `Prepare sherpa-onnx runtime` 阶段被取消，Windows sidecar build、shell fallback、timeout、file browser 与 Agent Plugin MCP parity 均未执行；该旧 run 的 Frontend Full 也曾因 `useAgentRuntimeSyncEffects.ts:634` React Hook dependency 缺失而失败，当前工作树已补齐依赖且本地全量 ESLint 通过。packaged enforcement 继续保持 OPEN_REF，不进入 release evidence；D0 Codex Desktop 实时视觉仍因 macOS AppleEvent `-1712/-1743` blocked。完成度保持 86%，下一步是修复 Windows runner 的 sherpa-onnx 准备阶段并取得 packaged sidecar/runtime 的真实证据。

## 31. 2026-08-31 D1 Thread 阅读列宽度与密度收口

- 用户反馈与根因：对照 Codex Desktop Thread 页面时，assistant 正文和过程时间线没有落在同一内容轴；原 `CONVERSATION_CONTENT_TARGET_WIDTH` 使用 `68%`，嵌套在外层 `720px` 上限后会让正文实际收缩到约 `640px`，而过程摘要仍按外层列宽展开，形成右侧错位。历史过程块的纵向间距也明显松于 Codex Desktop。
- 实现：在既有 `conversationLayoutTokens` owner 内将目标宽度收敛为 `clamp(640px, 100%, 720px)`，让外层 workspace 负责上限、正文与时间线共同填充内容列；`MessageList`、inline/floating composer 继续复用同一 token。消息列和 turn 间距收紧为 `gap-3/py-3`、`py-1`，过程摘要、思考详情、工具行、计划行统一为紧凑纵向节奏；新增 `data-content-axis="thread"` 与 `data-layout-density="compact"` 作为稳定 DOM 回归标记。
- 写集：`conversationLayoutTokens.ts`、`styles/index.ts`、`MessageList.tsx`、`AgentThreadTimeline.tsx`、`AgentThreadTimelineItemRenderers.tsx`、`StreamingProcessGroup.tsx` 及对应布局/思考测试；未新增协议、命令、runtime owner 或生产 mock。
- 稳定回归：`MessageList.test.tsx` 20/20、`AgentThreadTimeline.reasoning.test.tsx` 15/15，共 35/35；断言单一内容轴、紧凑密度、assistant 宽度 token、过程详情收纳和历史首帧布局。
- 跨层验证：`npm run lint`、`npm run typecheck`、`npm run test:contracts`、`npm run i18n:check:json`、`npm run verify:gui-smoke` 与 `git diff --check` 均通过。最新真实 Electron shell evidence 为 `.lime/qc/project-gates/standalone-shell-01-20260831033444-66411/shell-01-electron-smoke/summary.json`，`result=pass`、`proofLevel=Gate B-F`、console/page/preload/invoke/trace/legacy/mock 错误均为 0；Thread fixture 的 DOM geometry 已确认 assistant 与过程时间线同为约 `680px`。
- 视觉与证据边界：本轮遵循 Thread 工作区“单一主阅读列、信息优先、紧凑时间线、composer 与正文复用内容轴”的 Lime 规则；当前主对象为 active Thread，阶段为阅读/执行中，下一步由既有 composer/action slot 承接。Codex Desktop 实时窗口仍受 macOS AppleEvent `-1712/-1743` 阻塞，因此不宣称 D0 实时视觉验收；整体计划完成度保持 86%，D1 该增量已完成，Windows packaged enforcement 仍为 OPEN_REF。

## 32. 2026-08-31 D4 Windows Sherpa 准备阶段收口

- 根因与范围：Windows `windows_shell_runtime` 在 Sherpa 预置运行时阶段曾被取消，后续 sidecar 构建和 packaged Gate B 未执行。本轮只修复准备脚本的跨平台下载/解压边界，不改变 Windows sandbox owner、协议或 GUI；首页 Banner / 皮肤 Hero 仍保留且不纳入对齐。
- 实现：`prepare-sherpa-onnx-runtime.mjs` 在 Windows 使用明确的 `curl.exe`，下载写入 `.part` 后原子重命名，并设置 5 分钟进程级下载超时；Windows 解压按 `7z`、系统 7-Zip 路径、`tar` 顺序尝试，每个 extractor 设置 120 秒超时，失败时清理目标目录并 fail closed。既有 macOS `tar` 与 Mach-O rpath 路径保持不变。
- 稳定回归：`scripts/prepare-sherpa-onnx-runtime.test.mjs` 15/15，覆盖 Windows `curl.exe` 下载命令、`.part` 临时文件、7-Zip/tar 回退及驱动器路径；`scripts/electron/verify-package-resources.test.mjs` 与 `scripts/electron/release-workflow-guard.test.mjs` 共 48/48；Prettier、Node syntax check 与 `git diff --check` 通过。
- 跨层回归：`npm run governance:scripts`、`npm run test:contracts`（299 checks）、`npm run typecheck`、`npm run lint` 与 `npm run verify:gui-smoke` 均通过；最新 Electron shell evidence 为 `.lime/qc/project-gates/standalone-shell-01-20260831053656-76569/shell-01-electron-smoke/summary.json`，`result=pass`、21/21 assertions，console/page/preload/invoke、renderer crash/unresponsive、legacy/mock fallback 均为 0，Renderer、preload、IPC、App Server 初始化和重载均正常。
- 当前 macOS packaged 复核：隔离目录 `.tmp/codex-alignment-packaged-current-20260831` 通过 `verify-package-resources.mjs --platform darwin --arch arm64`；`codesign --verify --deep --strict`、主包/Helper identity、`app-server` 与 `code-mode-host` 资源完整。`npm run smoke:app-server-packaged-external-backend-failure` 通过，证明 packaged external backend 在部分输出后以结构化失败收口。
- 证据边界：当前只能证明本地脚本契约和既有 macOS packaged/resource 检查；真实 Windows runner 需要重新执行 `windows-2022` 的 Sherpa、sidecar build、Windows Squirrel 安装与 Electron Gate B，不能用本机结果替代。D0 Codex Desktop 实时窗口仍因 AppleEvent `-1712/-1743` blocked；计划整体仍保持约 86%，Windows packaged enforcement 为 OPEN_REF。

## 33. 2026-08-31 本地门禁复核与外部基线阻塞

- 本轮复核：`npm run verify:local` 已通过版本一致性、i18n 检查、全量 ESLint 和类型检查，随后在 Vitest 第 26/118 批次停止；失败测试为 `scripts/harness/deepswe-benchmark.test.mjs`，夹具仍写死 `deepswe-current-chain-adapter-v6`，而 current owner `deepswe-adapter-core.mjs` 已要求 `v7`，导致 `infraValid=false`。该漂移不在本轮写集，未修改 DeepSWE owner 或测试。
- 受影响范围的独立证据：Thread 布局/密度测试 35/35、Sherpa 准备脚本测试 15/15、`npm run test:contracts` 299 checks 通过；最新 `npm run verify:gui-smoke` 为 Gate B-F、21/21 assertions，通过证据为 `.lime/qc/project-gates/standalone-shell-01-20260831064342-75335/shell-01-electron-smoke/summary.json`；`git diff --check` 无错误。失败仅阻断完整 `verify:local`，不改变本轮 GUI/Sherpa 定向结论。
- CI 状态：`gh run list` 未发现新的 Windows runner；`33348108423` 仍是旧 run，Windows `Prepare sherpa-onnx runtime` 被取消，未产生 packaged sidecar/runtime 或 Electron Gate B 证据。未未经确认触发 workflow。
- 下一步与边界：由责任开发者在 `windows-2022` 重新执行 Sherpa、sidecar、Squirrel 安装和 Electron Gate B；Codex Desktop 实时窗口继续受 AppleEvent `-1712/-1743` 阻塞。整体完成度保持约 86%，Windows packaged enforcement 与 D0 视觉基线仍为 `OPEN_REF`。

## 34. 2026-08-31 DeepSWE 契约收口与治理基线阻塞

- 修复：current adapter 已升级为 `deepswe-current-chain-adapter-v7`，同步更新 `internal/test/deepswe-coding-slice-v2.json`、`deepswe-coding-slice.test.mjs` 和 `deepswe-benchmark.test.mjs`；benchmark 测试直接从 `deepswe-adapter-core.mjs` 导入 owner 常量，避免再次写死或依赖未导出的符号。该变更只修复测试/清单契约，不改变产品 runtime 或 GUI。
- 定向证据：DeepSWE benchmark 4/4、coding-slice 3/3、adapter 25/25 均通过；Thread 布局 35/35、Sherpa 15/15、`npm run test:contracts` 299 checks 和 Electron Gate B 21/21 仍保持通过。
- 完整门禁：`npm run verify:local` 曾在第 56/118 个 Vitest 批次被 6 个源级治理断言阻断；本轮已将断言同步到 current Rust owner 的 `ModelInfo::usable_context_window`、`ResolvedStepSettings`、`CoreToolPlanContext` 和迁移后的 reasoning switch 语义，并补齐 `persistent` reasoning effort，相关治理/模型定向测试 20/20 通过。
- 下一步与边界：Windows packaged Gate B 和 Codex Desktop 实时视觉基线仍为 `OPEN_REF`，整体完成度保持约 86%。

## 35. 2026-08-31 current Codex reasoning policy 对齐

- 范围：对照已安装 Codex current source 收口模型 reasoning policy 的源级事实，不扩展到新的 provider 或 runtime owner；首页 Banner / 皮肤 Hero 继续保留，不纳入对齐。
- 实现：`MODEL_REASONING_EFFORTS` 增加 Codex 正式 `persistent` 值；输入栏 `Record<ModelReasoningEffortLevel, string>`、Model Selector 和 `common` / `agentInputbar` 的 `zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR` 文案同步，避免将内部 wire token 裸露给用户并补齐 max/ultra 的选择器翻译。治理测试改为验证 current owner 的委托关系与公式：context 通过 `ModelInfo::usable_context_window`，reasoning summary 通过 `ResolvedStepSettings`，apply-patch 通过 `CoreToolPlanContext.model_info`，reasoning effort 通过 step settings/default 与 current switch clamp。
- 稳定回归：Codex policy origin 四组测试、`modelReasoningPolicy` 共 20/20；`ModelSelector` 与 Inputbar plan status 共 47/47；`npm run i18n:check:json` 通过（5 locale、14 namespace、8709 keys、无缺失/多余 key）；`npx tsc --noEmit` 已启动复核。
- 证据边界：本轮验证证明 Lime 的 policy projection 和 GUI 选择器可承接 Codex current reasoning values；不宣称 Windows packaged enforcement 或 Codex Desktop 实时视觉，二者继续保持 `OPEN_REF`。

## 36. 2026-08-31 Codex 通知 identity 与 Lime 模型扩展边界收口

- 根因：`model/list/updated` 是 Lime `model-provider + App Server` 自有扩展，覆盖 fixture 明确将其放在 `limeExtensions`，但 Renderer drift catalog 同时把它登记进 Codex notification identity，导致 current method inventory（78 项）与 `listCodexV2NotificationMethods()`（79 项）漂移。
- 修复：从 `CODEX_V2_NOTIFICATION_METHODS` 移除 `model/list/updated`，保留 `DIAGNOSTIC_ONLY_NOTIFICATION_METHODS` 登记，使该通知继续分类为 `known_diagnostic_only` 且不产生 conversation warning；未改变 App Server producer、typed protocol、model registry consumer 或 GUI 行为。
- 回归：`appServerNotificationDrift.test.ts` 与 `renderProjectionCoverageBoundary.test.ts` 共 15/15 通过，`git diff --check` 通过。
- 边界：Codex 72 项通知与 Lime model catalog refresh 继续保持不同 owner；首页 Banner / 皮肤 Hero 不纳入对齐。Windows packaged Gate B 与 Codex Desktop 实时视觉基线仍为 `OPEN_REF`。

## 37. 2026-08-31 本地门禁与 Thread 模型恢复夹具收口

- 夹具同步：`ChatModelSelector.integration.test.tsx` 的 session detail 补齐 canonical `thread_id`，runtime adapter 的 `resumeThread` 返回成功，恢复 current `thread/resume` 合同；生产代码仍保持缺失 identity 或恢复失败时 fail closed。
- 完整门禁：`npm run verify:local` 通过版本一致性、i18n 结构/未引用 key、ESLint、硬编码扫描、TypeScript、Vitest 118/118 批次和 GUI smoke。`npm test` 全量批次无失败，模型切换恢复测试 4/4、通知 drift/coverage 15/15 均通过。
- Gate B 证据：`.lime/qc/project-gates/standalone-shell-01-20260831090637-48611/shell-01-electron-smoke/summary.json` 为 `Gate B-F`、21/21 assertions，通过真实 Electron/preload/IPC、`app_server_handle_json_lines` 与 App Server 初始化；console/page/preload/invoke/renderer crash/unresponsive、legacy command 和 mock fallback 均为 0。
- 边界：该证据覆盖 macOS Electron 壳与 current bridge，不等价于 Windows packaged enforcement，也不解除 Codex Desktop 实时视觉基线的 AppleEvent `-1712/-1743` 阻塞；首页 Banner / 皮肤 Hero 继续保留。整体完成度仍约 86%，上述两项保持 `OPEN_REF`。

## 38. 2026-08-31 收尾门禁复核

- `git diff --check` 通过；`npm run test:contracts` 通过（协议生成无漂移、App Server client 299 checks、命令/脚本/文档边界均通过）。
- 最新 `npm run verify:gui-smoke` 通过，Gate B-F 证据为 `.lime/qc/project-gates/standalone-shell-01-20260831091007-54581/shell-01-electron-smoke/summary.json`：21/21 assertions，真实 Electron/preload/IPC、`app_server_handle_json_lines`、current App Server method、runtime/read model 与 GUI 链路命中；console/page/preload/invoke、renderer crash/unresponsive、legacy command 和 mock fallback 均为 0。
- 本轮未新增产品实现；首页 Banner / 皮肤 Hero 仍按用户裁决保留。Windows packaged enforcement 与 Codex Desktop 实时视觉基线继续为 `OPEN_REF`，整体完成度保持约 86%，不进入 release evidence。

## 39. 2026-08-31 Agnes 官方模型路由修复

- 用户反馈：首页当前模型为 `agnes-2.5-pro` 时显示“当前模型通道暂时不可用”，但该模型是 Agnes 官方文档公布的 OpenAI 兼容聊天模型；问题不是首页 Banner / 皮肤 Hero，不纳入本刀对齐。
- 根因：自定义 Provider 只保存模型 ID 时，`resolve_provider_model_metadata` 使用 Provider ID 构造声明模型，官方 Agnes endpoint 没有参与 canonical provider 识别，导致 `agnes-2.5-pro` 变成 `inferred_hint`，RuntimeCore 在 Turn admission 前拒绝执行；选择器随后只能呈现通用 Provider 不可用提示。
- 实现：canonical registry 新增 `agnes/agnes-2.5-pro`（文本/图片输入、文本输出、流式、工具、推理）；模型声明元数据新增受控 endpoint-aware 构造器，仅 `https://apihub.agnes-ai.com:443` 识别为 Agnes，未知 custom host 继续保持 `inferred_hint`。App Server 官方 Agnes route 回归改为验证 `agnes-2.5-pro` 可执行，非官方 host 负向回归保留。
- 写集：`lime-rs/crates/model-provider/src/canonical/data/canonical_models.json`、`canonical/name_builder.rs`、`services/model_registry_service.rs`、`services/model_registry_service/runtime_metadata.rs`、`app-server/src/runtime_backend/model_route_resolver.rs`；未修改数据库、API key、Electron bridge 或首页 Banner。
- 定向验证：`lime-services` 全量 216 passed / 4 ignored，本轮再次复核 `model_registry_service` 76/76；`model-provider` canonical tests 11/11；前端 `ModelSelector` + provider compatibility 52/52；Rustfmt 与 `git diff --check` 通过。App Server 测试 profile 的 Agnes 路由测试仍被本机 `rusty_v8 v150.4.0` macOS arm64 预编译包 404 阻断，未伪造通过证据；dev profile 已成功构建包含本轮修复的新 App Server sidecar。
- 跨层验证：`npm run test:contracts` 通过（App Server client 299 checks、modality contracts 7/7、命令/脚本/文档边界无漂移）；`npm run verify:gui-smoke` 通过，Gate B-F 证据为 `.lime/qc/project-gates/standalone-shell-01-20260831153535-57886/shell-01-electron-smoke/summary.json`，21/21 assertions，真实 Electron/preload/IPC、`app_server_handle_json_lines`、App Server 初始化和 GUI shell 命中，console/page/preload/invoke、legacy command 与 mock fallback 均为 0。
- 治理分类：官方 Agnes endpoint canonical 映射与模型注册为 `current`；未知 endpoint、无 capability 的 custom 模型仍是 `inferred_hint` / fail closed；无新增 `compat/deprecated/dead`，不恢复旧 runtime 或 fallback。
- 边界与完成度：代码路径和新 sidecar 已修复，实际 Agnes 网络成功仍取决于用户 API key 的授权/额度；当前既有 Electron 窗口需重启后加载新 sidecar。真实窗口只读复核被 macOS AppleEvent `-1712` 阻断，本轮未发送 live Provider 请求，因此不宣称 Agnes 网络 Gate B。Windows packaged enforcement 与 Codex Desktop 实时视觉基线仍为 `OPEN_REF`，整体计划约 88%。

## 40. 2026-09-01 Composer 受控输入与生产 bundle 稳定性复核

- 根因与修复：Electron smoke 使用旧 `dist` 产物时暴露 `EmptyState` 的 `pendingImages` TDZ（压缩 bundle 将同步 effect 提前到后置局部变量之前），同时 `BaseComposer` 历史快捷键依赖数组漏列。当前 `EmptyState` 以 `ComposerController` 快照作为唯一输入事实，外部输入同步改为幂等 `setText`，提交直接读取 controller 文档；`BaseComposer` 补齐 `controller`、`setText`、`textareaRef` hook 依赖。Controller 自身做文档相等判断，不产生额外更新。
- 写集：`src/components/agent/chat/components/EmptyState.tsx`、`EmptyStateComposerPanel.tsx`、`src/components/input-kit/BaseComposer.tsx`、`ComposerController.ts`、`useComposerController.ts` 及对应测试；不改变 Thread/Turn/Item 协议、Electron bridge、Provider 配置或首页 Banner / 皮肤 Hero。
- 稳定回归：ComposerController 6/6、BaseComposer 16/16、EmptyState 62/62，共 84/84；目标 ESLint 无 warning；`npm run typecheck` 通过；`git diff --check` 通过。
- 跨层验证：`npm run test:contracts` 通过（App Server client 299 checks、modality contracts 7/7、脚本/命令/文档边界通过）；重新构建压缩 renderer、Electron main/preload 与本地 app-server sidecar 后，`npm run verify:gui-smoke` 通过，最新 Gate B-F summary 为 `.lime/qc/project-gates/standalone-shell-01-20260831234458-29753/shell-01-electron-smoke/summary.json`：21/21 assertions，renderer crash/unresponsive、console/page/preload/invoke error、legacy command 和 mock fallback 均为 0。
- 视觉与证据边界：本轮保持首页 Hero/Banner，修复的是 Codex Desktop 风格工作区 Composer 的稳定性；当前主对象为输入草稿，阶段为可提交/可中断，下一步由 `start|steer|interrupt` typed action 承接。证据为真实 Electron Gate B-F shell，不包含 Agnes live provider 请求、Windows packaged enforcement 或 Codex Desktop 实时视觉验收；后两项继续 `OPEN_REF`，整体计划约 89%。

## 41. 2026-09-01 Windows packaged 资源链本地守卫补强

- 范围：继续 D4 Windows packaged sidecar/runtime 对齐；不修改 Windows sandbox owner、协议或 Electron 业务命令，也不把首页 Banner / 皮肤 Hero 纳入对齐。
- 实现：确认 `verify-package-resources` 对 Windows `windows-sandbox-setup.exe` 与 `windows-sandbox-runner.exe` 均执行存在性和 manifest SHA-256 校验；补充 setup helper 缺失时的 fail-closed 回归。Windows Squirrel workflow 回归现在锁定 `Prepare sherpa-onnx runtime -> electron:build/Forge make -> verify-package-resources -> stage-release-assets -> installed Squirrel SHELL-01 -> Agent Plugin Gate B` 的存在与顺序，避免工作流退化成只打包或只上传。
- 验证：`scripts/electron/verify-package-resources.test.mjs` 21/21、`scripts/electron/windows-squirrel-rc-smoke.test.mjs` 18/18；此前已通过的 `npm run test:contracts`、`npm run typecheck`、`npm run verify:gui-smoke` 证据继续有效；本轮 `git diff --check` 通过，`npm run governance:scripts` 通过。
- 证据边界：本轮只增加 macOS 可执行的资源/工作流契约守卫，不证明 Windows 安装、Squirrel N-1 更新、packaged sidecar 实际启动或 Windows Electron Gate B。必须由 `windows-2022` 重新执行 workflow 并上传 `.lime/qc/windows-squirrel-rc/**` 与 packaged Gate B evidence；Codex Desktop 实时视觉基线仍受 AppleEvent `-1712/-1743` 阻塞。计划继续保持 `active`，Windows packaged enforcement 与 D0 视觉基线为 `OPEN_REF`，整体约 89%。

## 42. 2026-09-01 Windows packaged CodeMode Gate B 接入

- 范围：继续 D4 Windows packaged sidecar/runtime 对齐，补齐安装后 CodeMode 的真实 Electron Gate B 入口；不修改 CodeMode 协议、App Server 业务命令、Windows sandbox owner，也不把首页 Banner / 皮肤 Hero 纳入对齐。
- 实现：`code-mode-electron-gate-b.mjs` 新增 `--electron-executable`，packaged 模式直启安装后的 Lime executable，并强制 `APP_SERVER_BIN=""`，避免源码 sidecar override；Windows 进程树采集通过 PowerShell `Win32_Process` 识别安装包内 `app-server.exe`，继续要求 `code-mode-host.exe` 为 App Server 子进程。`build-windows-test.yml` 在安装后 Agent Plugin Gate B 之后追加 CodeMode Gate B，并上传结构化 evidence；工作流守卫锁定 Sherpa 准备、Windows sidecar 打包、安装后 Gate B 的存在与顺序。
- 稳定回归：CodeMode Gate B、Windows Squirrel RC 与 package resource 定向测试共 46/46 通过；`npm run test:contracts`（App Server client 299 checks、modality/命令/脚本/文档边界）、`npm run typecheck`、`npm run governance:scripts` 与 `git diff --check` 均通过。
- 证据边界：本轮只有 macOS 本地静态/单元/契约证据，尚未证明 Windows Squirrel 实际安装、`Lime.exe -> app-server.exe -> code-mode-host.exe` 真实进程链、restricted runner 或 packaged Electron Gate B；不得将本地测试标记为 platform pass。必须由责任开发者确认后重新运行 `windows-2022` workflow 并上传 `.lime/qc/windows-squirrel-rc/**` 与 `.lime/qc/gui-evidence/code-mode-electron-gate-b-windows/**`。Codex Desktop 实时视觉基线继续受 AppleEvent `-1712/-1743` 阻塞，计划保持 `active`，Windows packaged enforcement 与 D0 视觉基线为 `OPEN_REF`，整体完成度约 90%。

## 43. 2026-09-01 首页预热 session 会话接管时机修复

- 最终根因：`input_warmup` 的职责仅是提前准备底层 session；预热成功后错误调用 `markTaskCenterLocalSessionOverride`，会让路由同步 owner 把尚未 commit 的 session 当作正式本地会话接管，继而清空 active draft。首页因此从保留 Banner / Hero 的 EmptyState 切成空 MessageList，正式发送也无法继续复用预热 session。此前“预热后立即设置 local override”的诊断和方案已被真实 Electron 热路径证据推翻，本节以最终结论为准。
- 实现：预热阶段只创建并缓存 session，不建立正式本地会话接管；用户发送并执行 commit 时才沿既有 owner 建立 override、切换路由和提交 turn。首页 Banner / 皮肤 Hero 继续作为明确产品设计保留，不纳入 Codex Desktop 对齐删除范围。
- 稳定回归：`useTaskCenterDraftMaterializationRuntime.test.tsx`、`useWorkspaceInputbarSceneRuntime.test.tsx` 等定向 Vitest 共 36/36 通过；新增断言锁定预热不得调用正式会话接管，且草稿标签与 active draft 必须保持。
- 真实验证：`.lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-home-hotpath-prewarm-fix-summary.json` 与聚合 `claw-chat-current-fixture-home-hotpath-regression-summary.json` 均为 `ok=true`。输入后、提交切换和完成后三个稳定窗口均无闪烁，正式 turn 复用预热 session `01a05c87-572a-7bc1-ba94-9143a538a6fe`，用户消息、助手完成态和 Thread/Turn/Item identity 一致；真实 Electron/preload/IPC/App Server JSON-RPC 命中，legacy command 与 mock fallback 为 0，`liveProviderUsed=false`。
- 证据边界：完整 `npm run smoke:agent-runtime-current-fixture` 已通过受控 provider fixture；最新 GUI shell Gate B-F 为 `.lime/qc/project-gates/standalone-shell-01-20260901104344-8413/shell-01-electron-smoke/summary.json`，21/21 assertions。上述证据不等价于 Windows packaged enforcement 或 live provider，二者按各自门禁继续验证。

## 44. 2026-09-01 Codex Composer 持久化历史与异步合并收口

- 对比结论：Codex TUI 的 `~/.codex/history.jsonl` 是 append-only 纯文本历史，支持 newest-first cursor 分页、Unix inode / Windows creation-time log identity 和坏行容错；Lime 将同一语义收敛到 App Server current 主链，不复制 TUI 的 ANSI/终端状态。
- 实现：新增稳定 v2 `promptHistory/read|append` 与 typed App Server client；Rust `PromptHistoryStore` 在受控 `data_root/prompt_history.jsonl` 下以文件锁、Unix `0600`、4 MiB 上限、malformed-row skip 和跨平台 `logId` identity 提供分页读写。Composer Controller 保持本地 session history 与受控 draft，成功提交后追加纯文本；服务端历史加载按 oldest-first 合并，读取期间发生的本地提交通过边界重叠去重，不再被异步快照覆盖。
- 治理分类：`current` 为 `ComposerController`、`promptHistory/*`、`PromptHistoryStore` 和两层 typed client；`compat/deprecated` 无新增；`dead/forbidden-to-restore` 为 renderer `localStorage`/旧 Desktop history facade/第二持久化 owner。附件、路径引用、mention 与大粘贴原文只保留会话内，不能进入 JSONL。
- 验证：`cargo check -p app-server-protocol`、`cargo test -p app-server-protocol`（133/133，含新增契约测试）、`npm run generate:protocol-types`、`npm run check:protocol-types`、`npm run typecheck`、Composer + `promptHistory` Vitest 12/12、`npm run test:contracts` 均通过；`cargo check -p app-server --lib` 仍受本机 `rusty_v8 v150.4.0` Apple Silicon 预编译包 404 阻塞，不能伪造 App Server profile 通过。Electron sidecar dev build 已通过且无新增 Rust warning。
- 证据边界：真实 Electron Gate B 只证明 current bridge/runtime/read-model 链，不等价于 Windows packaged enforcement；Codex Desktop 实时 accessibility/screenshot 继续受 AppleEvent `-1712/-1743` 阻塞。下一刀回到 `windows-2022` Sherpa、Squirrel 安装、双 sidecar 进程链和 packaged CodeMode Gate B，整体计划保持 `active`，Windows packaged 与 D0 视觉基线为 `OPEN_REF`。

## 45. 2026-09-01 Windows packaged Gate B 远程证据前置核对

- 用户已确认可以触发真实 `windows-2022` GitHub Actions；本次先核对远程分支 `codex/windows-restricted-evidence-20260826`，其远端仍指向 `fd1059e94feeddf614648b8679808844380fbb8e`，与当前 `HEAD` 相同。
- 远程 workflow 仍只有安装后的 Agent Plugin Gate B，没有本轮新增的 `Run installed Windows CodeMode Gate B`、`--electron-executable` 入口或对应 evidence 上传；远程近期没有可复用的 packaged CodeMode run。因此不能触发并把旧 workflow 结果宣称为本轮实现证据。
- 当前 CodeMode workflow、Gate B 脚本和首页预热 session 修复均仍是未提交工作树改动；根 `package.json` 同时存在外部 Codex Desktop 工程替换，导致本地 Lime Electron 入口和 `verify:app-version` 脚本不可用。上述外部改动未覆盖、未回滚。
- 可交付本地证据保持有效：使用已安装 Vitest runner 重跑 CodeMode/Windows Squirrel/package resource 定向回归 46/46、相关 task-center owner 测试 54/54、目标 TypeScript/ESLint、`npm run test:contracts`、`npm run verify:gui-smoke` 21/21、开发态 CodeMode Electron Gate B 均通过。Windows 安装、双 sidecar 进程链和 packaged CodeMode Gate B 继续为 `OPEN_REF`，需要责任开发者将本轮改动发布到远程分支后再触发 workflow。
- 计划状态保持 `active`，完成度不因静态或旧版本 CI 结果上调；首页 Banner/皮肤 Hero 继续按用户裁决保留，不纳入对齐。

## 46. 2026-09-01 v1.138.0 发布前最终复核

- Lime 根 manifest、CLI npm 包、Rust workspace 与 Cargo.lock 已统一为 `1.138.0`；外部 Codex Desktop manifest 漂移已从 release candidate 排除，4 个根目录压缩 bundle 继续仅作只读参考，不进入版本提交。
- 本地发布门禁已通过：`npm run verify:app-version`、`npm run typecheck`、`npm run test:contracts`、`npm run smoke:agent-runtime-current-fixture`、定向 Vitest 36/36、目标 ESLint、`git diff --check`。`npm run verify:gui-smoke` 最新 Gate B-F 为 21/21，真实 Electron/preload/IPC、`app_server_handle_json_lines` 和 current App Server method 命中，全部错误、legacy command 与 mock fallback 计数为 0。
- `npm run test:related -- ...` 曾在测试收集阶段把 `electron/` 目录当成文件读取并报 `EISDIR`；直接 Vitest 精确入口已覆盖同一受影响集合并全部通过，该问题记录为 runner 收集缺陷，不是产品断言失败。
- `v1.138.0` 本地与远端 tag 均不存在。下一步是获得 release Git 写操作确认，提交完整 candidate、创建并推送 tag；随后由 release workflow 在 `windows-2022` 上验证 Squirrel 安装、packaged CodeMode Gate B 与发布资产。首页 Banner / 皮肤 Hero 保持明确产品例外。

## 47. 2026-09-01 业务层工具暴露语义对齐

- Codex HEAD `d58d0e5841` 的 `tools.update_plan.enabled` 默认关闭；显式开关才注册 checklist tool，且与 Plan Mode 解耦。Lime 已将这一语义放入 current `ToolExecutionPolicyConfig`、App Server config metadata、Agent inventory/current provider tool filtering 和设置页，不复制 Codex TUI 的终端交互。
- GUI 设置页新增“计划工具”开关，默认关闭，开启后通过现有 `config` 写链保存；`zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR` 文案和 summary 状态同步，未新增第二业务后端或 Renderer mock。
- 验证：配置相关 Rust 单测 `3/3`、Agent inventory `15/15`、current provider tool snapshot `6/6`、App Server provider/inventory 各 `3/3`、`test:rust:related` 全部受影响包通过；设置页 Vitest `5/5`、`lime-core` cargo check、`verify:gui-smoke`（真实 Electron/preload/IPC/App Server 链，`21/21`）及 `git diff --check` 通过。未注入 V8 缓存时 Darwin arm64 预编译资产下载返回 404，本轮使用仓库已有临时缓存完成 Rust 验证，fresh-environment 下载仍未验证。
- 治理分类：`current` 为计划工具配置/投影/GUI；`compat` 为历史工具名别名；`deprecated` 无新增；`dead/forbidden-to-restore` 为旧 UI 二次 owner、Renderer mock 和第二套状态机；Codex account/rate-limit/TUI 专属业务、未确认 token budget/proactive catalog contract 归 `product-scope-excluded` 或 `gap / pending owner`。

## 48. 2026-09-02 模型目录驱动的回合语义

- 对齐 Codex `ModelInfo/ModelMessages` 与 `core::session::multi_agents`：可信模型目录现在保留 instructions template/variables、context/max-context/auto-compact/effective-percent、Multi-Agent v2 文案与 Ultra provider override；App Server 每回合只从选中 route 的 catalog metadata 合并到 current session/turn context。
- Multi-Agent v2 精确按字段存在性解析：`hint_text`（含空字符串）优先；Ultra 使用 `proactive`，缺失时使用内置 proactive；非 Ultra 使用 `explicit`，缺失时保持 explicit-request-only。World State 保留用户选择的 `ultra`，provider wire 则按目录 override、`max`、最高非 Ultra、`medium` 的顺序 lowering；本地 `persistent` 在 provider wire 上 lowering 为 `disabled`。
- `model/list.supportsPersonality` 不再硬编码：只有 instructions template 含 `{{ personality }}`，且 default/friendly/pragmatic 三个变量都是字符串时才为 true，空字符串仍算目录显式声明。人格模板渲染、model limits 和 API 解析均补定向回归。
- 验证状态：`cargo check -p app-server` 通过；模型目录 `2/2`、reasoning effort `4/4`、provider configuration `7/7`、Multi-Agent turn context `13/13`、session/config `19/19`、personality model list `1/1` 和 direct provider metadata `1/1` 定向回归通过；`npm run test:rust:related -- <本轮 Rust 写集>` 推导的 20 个反向依赖包全部通过。`npm run test:contracts` 通过 protocol types、App Server client 299 checks、command/harness/contracts、scripts governance 和 docs boundary；`npm run smoke:agent-runtime-current-fixture` 通过且 `liveProviderUsed=false`；`npm run verify:gui-smoke` 通过真实 Electron Desktop Host、preload/IPC、App Server `appserver.v0`、Claw shell reload 和 memory settings Gate B。`cargo fmt --all -- --check` 与 `git diff --check` 通过。Codex release 的 Darwin arm64 archive/binding 均可下载并校验；历史 404 仅对应 Deno 默认 sandbox URL 或错误的 mirror 路径，本轮已将缓存固定到稳定 OS 用户缓存目录并增加下载重试/超时。
- 后续 blocker：model-owned `approvals`、`permissions`、`auto_review`、Guardian v2 和完整 token-budget 语义仍未进入可信 catalog -> App Server -> provider/tool runtime 消费链；`model_messages.token_budget` 不得直接映射现有 provider hard token budget。

## 49. 2026-09-02 Codex Rusty V8 资产下载收口

- 继续采用 Codex 的 `rusty-v8-v<version>` release 资产，不引入 Lime 自建镜像或关闭 `v8_enable_sandbox`。`RUSTY_V8_MIRROR` 不作为入口，因为 v8 crate 的 `/v<version>` 拼接规则与 Codex release tag 不兼容。
- `scripts/lib/rusty-v8-artifacts.mjs` 现在使用 macOS/Windows/Linux 稳定用户缓存目录，并为 curl 下载增加有限重试、连接超时和总超时；显式 `LIME_RUSTY_V8_CACHE_DIR` 仍可用于隔离 CI 或维护者验证。
- 验证：helper 真实解析 `v8=150.4.0` 并成功取得 Codex Darwin arm64 archive/binding；资产 supply-chain Vitest `6/6` 通过，输出路径位于 `~/Library/Caches/Lime/rusty-v8`。该切片不改变 Cargo.lock、V8 feature 或运行时资源打包边界。

## 50. 2026-09-02 Windows packaged GUI evidence identity

- Windows workflow 的 CodeMode/native host Gate B 现在显式继承 Squirrel 安装候选的 `candidateRunId`，并由 `windows-packaged-evidence.mjs` 绑定安装后的 `Lime.exe`、资源清单、sidecar 进程命令和 native helper 路径。GUI Gate B 因此只能声明同一 packaged candidate 的 Electron/preload/IPC/App Server/runtime/read-model/DOM 证据。
- 这不等于 Codex Desktop 实时视觉一致：本轮没有获得 ChatGPT.app accessibility tree 或 1536x960、1280x800、窄窗口的真实对照截图，也没有 Windows packaged GUI 运行结果；视觉基线和 Windows GUI Gate B 继续保持 `OPEN_REF`。
- 当前 GUI current owner 仍是 Thread/Turn/Item projection、ThreadWorkspaceHeader、canonical timeline、Composer 和右侧 context surface；平台 host 只提供系统能力，不新增 Windows 业务后端或 macOS Swift 平行业务链。

## 51. 2026-09-02 GUI 三档窗口几何回归

- 范围：将现有真实 Electron `SHELL-01` smoke 扩展为 `1536x960`、`1280x800`、`980x680` 三档 `BrowserWindow.setSize/getSize` 几何契约；不把它表述为 Codex Desktop 像素级视觉一致性，也不替代 Windows packaged Gate B。
- 实现：`electron/smokeChecks.ts` 在 renderer reload 后采集窗口尺寸、viewport、文档溢出、workspace shell/main area、Composer、输入框、可选 message list/Thread header 的稳定矩形，断言必要节点可见、在视口内、横向无意外溢出、标题与 header actions 不遮挡，并保存三张 `layout-<width>x<height>.png` 截图。`electron/smokeEvidence.ts` 将 `layout.proofLevel`、三档 viewport、断言和 screenshot 列表纳入同一 `SHELL-01` summary；summary validator 对布局 proof、几何稳定性和三张截图 fail closed。
- 产品修复：首页 EmptyState 的 Composer frame 在短窗口使用基于 `100vh` 的有限负上边距，使 `980x680` 最小受支持窗口中 205px 高 Composer 完整进入视口；高窗口和业务链路保持不变。
- 验证：`electron/smokeEvidence`、入口守卫、summary validator、EmptyStateLayout 定向测试通过；`npm run typecheck:electron`、Prettier 通过；真实 `npm run verify:gui-smoke` 通过，最新 summary 为 `.lime/qc/project-gates/standalone-shell-01-20260902095726-86015/shell-01-electron-smoke/summary.json`，24/24 assertions，三档 `layoutGeometryStable=true`，`980x680` Composer bottom=`678.67px`，console/page/preload/invoke/legacy/mock 均为 0。
- 证据边界与下一步：本轮证明 Lime 真实 Electron responsive layout contract 和现有 preload/IPC/App Server shell 主链未回归；没有取得 Codex Desktop 实时 accessibility/screenshot，也没有 Windows packaged GUI 结果。后续仍由 `windows-2022` 真实候选完成 Squirrel 安装、sidecar/native host/CodeMode 与 packaged GUI Gate B，D0 实时视觉继续 `OPEN_REF`。

## 52. 2026-09-02 GUI 几何证据复核

- 复核范围：重新构建 renderer、Electron main/preload 与本地 App Server sidecar，重新执行三档 `BrowserWindow.setSize/getSize` 几何采集和 summary fail-closed 校验；不改变布局 owner 或平台业务边界。
- 最新证据：`.lime/qc/project-gates/standalone-shell-01-20260902105148-51228/shell-01-electron-smoke/summary.json` 为 `Gate B-F`、24/24 assertions；`1536x960`、`1280x800`、`980x680` 均 `windowSizeMatchesRequest`、必要节点可见且在视口内、无横向溢出、header actions 不遮挡，`composerHeightStable=true`，三张布局截图均存在。真实 Electron/preload/IPC、`app_server_handle_json_lines`、App Server 初始化命中，console/page/preload/invoke、renderer crash/unresponsive、legacy command 和 mock fallback 均为 0。
- 交叉门禁：`npm run test:contracts` 与 `npm run governance:scripts` 通过；本轮定向 Vitest 4 个文件 26/26、`npm run typecheck:electron`、Prettier 和 `git diff --check` 通过。
- 稳定性补充：统一 `npm test -- --resume` 续跑 119/119 批全部通过；修复五语言执行策略描述中的来源品牌词，并将治理测试从已删除历史文档迁移到 current `internal/aiprompts/architecture.md` 事实源，相关 i18n/治理测试通过。
- 证据边界：该复核证明 Lime macOS Electron responsive layout contract 和 current shell 主链未回归，不证明 Codex Desktop 像素级视觉一致、Windows packaged 安装/升级或系统权限。Windows `windows-2022` packaged GUI/CodeMode/native host Gate B 与 Codex Desktop 实时 accessibility/screenshot 继续为 `OPEN_REF`。
