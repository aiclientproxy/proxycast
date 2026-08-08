# Refactor v2 全项目 Gate A/B 验收计划

> status: active / candidate-freeze-blocked-by-import-owner
> owner: quality-workflow + product-domain owners
> created: 2026-07-15
> last_updated: 2026-07-17
> target_candidate: pending；上一候选 `20260715T134155Z-751fa87b9458` 已失效
> source: `internal/research/refactor/v2/**`
> architecture_impact: none; 本计划只定义验证；`fork_turns` 产品变更另按重大架构变更登记

## 1. 主目标

对 Refactor v2 之后的整个 Lime 产品做一次可审计的项目级验收，证明以下两件事：

1. Gate A：所有 current Renderer 页面、projection、状态、交互和失败文案在可控数据下正确。
2. Gate B：关键产品路径真实经过 Electron Desktop Host、preload/IPC、
   `app_server_handle_json_lines`、App Server JSON-RPC、RuntimeCore/backend、
   Thread/Turn/Item read model，并最终形成用户可见状态。

固定产品链不变：

```text
React Renderer
  -> Electron Desktop Host / preload
  -> app_server_handle_json_lines
  -> App Server JSON-RPC
  -> RuntimeCore / agent-runtime
  -> model-provider + tool-runtime
  -> Thread / Turn / Item + ProjectionStore
  -> Renderer projection / visible GUI
```

本计划不是“把现有 smoke 全跑一遍”。目标是建立候选快照、覆盖目录、证据等级、
故障注入、跨平台复验和最终签字条件，避免局部脚本通过却无法证明整个产品可交付。

### 1.1 当前执行状态

| Wave     | 状态    | 当前证据/下一步                                                       |
| -------- | ------- | --------------------------------------------------------------------- |
| Wave 0   | pending | SHELL-01/02 standalone B-F 已就绪；等待 import owner 退出后冻结新候选 |
| Wave 1   | pending | 旧 run 的 1-55、修复后的 56-110 只作诊断；新候选必须从第 1 批完整重跑 |
| Wave 2   | pending | 新候选 Wave 1 全绿后才进入 Gate A Page union / SettingsTabs           |
| Wave 3-5 | pending | Gate A 签字后执行真实 Electron Gate B 与故障注入                      |
| Wave 6-8 | pending | live provider 授权、packaged/platform、最终回放                       |

上一候选已在 Wave 1 第 56 批暴露陈旧 source boundary guard 后失效；随后发生的测试清理和
`spawn_agent.fork_turns` 产品变更也不属于该快照。当前处于冻结前收口阶段，只能在产品代码、
架构事实源与 focused validation 完整闭环后生成新 `run-id`；禁止复用旧 run 的后半程结果签字。

## 2. 当前事实与结论

### 2.1 可复用证据

`2026-07-15-s7ag-final-local-gates-and-project-shell-env.md` 已证明当时的候选通过：

- frontend smart suite `110/110`；
- changed Rust scope 中 App Server `1119/1119`；
- legacy report `0/0/0`；
- `verify:gui-smoke`；
- scoped rustfmt 和 `git diff --check`。

S7ah/S7ai 又完成了 S1/S2/S4/S5 状态对账。v2 当前只剩真实 PR
event/body/base 上的 architecture confirmation，不能在本地伪造通过。

### 2.2 不能直接继承的部分

当前工作树仍包含 Agent GUI、Team Memory shadow/runtime 删除和 Rust projection 调整。
这些改动发生在上述最终证据之后，因此旧证据只能作为历史基线，不能证明当前候选。

已有 Gate B 在 AgentControl、MCP、历史恢复、Coding Workbench、Content Factory
方向很深，但项目覆盖不对称：Knowledge、Automation、Channels、Resources、
Browser Runtime GUI、设置全量、桌面宿主能力和 packaged app 尚未形成同一证据目录。

因此当前结论是：

```text
Refactor v2 历史候选：本地实现闭环已证明。
当前工作树候选：Gate A/B 未开始，禁止沿用旧摘要宣称全项目通过。
```

### 2.3 当前候选与阻塞记录

| 候选                            | 结果          | 原因                                                                     |
| ------------------------------- | ------------- | ------------------------------------------------------------------------ |
| `20260715T124619Z-a752c9e89468` | `invalidated` | Wave 1 ESLint 暴露 10 个已退役 Team binding                              |
| `20260715T130224Z-1dd4cb78fd2a` | `invalidated` | S6v Workspace `agentTeam` 删除在冻结后继续写入                           |
| `20260715T132305Z-168506e7be24` | `invalidated` | Agent UI 标准与路线图在冻结后继续写入，changed path 数超过记录的 129     |
| `20260715T134155Z-751fa87b9458` | `invalidated` | Wave 1 第 56 批陈旧 boundary guard；后续测试清理与 `fork_turns` 产品变更 |

当前阻塞不是某条 Gate 已失败，而是产品候选尚未重新冻结：Wave 1 陈旧测试已清理，Codex
`fork_turns` parity 已进入 current owner，但扩大验证、架构事实源收口和新快照 digest 尚未完成。
在这些步骤完成前，旧 run 的分批通过只能作为定位证据，不能拼接成可交付结论。

### 2.4 Codex 对齐基线

本轮以只读 Codex 仓库本地 HEAD `2e4f55608b4ad26d9c48ea45a6fcd20bfd5e9fe8` 为实现基线。
该基线于 `2026-07-16` 审计；当时本地 `origin/main` 仅领先
`38b064c31b1f7464b281006316ec878ed23fea77`，只修改 TUI prompt 提交时序，不改变本计划的
Runtime/App Server/Multi-Agent 语义。Candidate 生成时必须再次从干净的只读 Codex 仓库读取 HEAD，
将 commit hash 写入 candidate JSON；不保存外部仓库路径。

MultiAgentV2 的模型工具面是：

```text
spawn_agent / send_message / followup_task / wait_agent / interrupt_agent / list_agents
```

Codex app-server 的 durable facts 是 `ThreadItem::CollabAgentToolCall` 与
`ThreadItem::SubAgentActivity(Started|Interacted|Interrupted)`；父子关系由 child Thread、
`parentThreadId`/agent path 和 Thread 状态承接，不存在 synthetic Team snapshot 或 raw status
旁路。Lime 的六工具命名、queue-only `send_message`、trigger-turn `followup_task`、wait/list/
interrupt 基本方向一致。

原 P0 parity gap 已在 current owner 关闭：`tool-runtime` 以 typed `SpawnAgentForkMode` 唯一解析
`fork_turns`，App Server 在 child 初始 mailbox task 前把 full/last-N history 写入 child 自身
EventLog、ProjectionStore 与 ThreadStore；`none` 保持空历史，reasoning/tool/inter-agent 旁路被
过滤，失败补偿删除不可用 child，provider transcript 从同一 child EventLog 派生。未建立 v1 alias、
`fork_context` compat、第二 child store 或 session metadata owner。focused 证据为 tool-runtime
`6/6`、App Server AgentControl `17/17`、projection package `294/294`；`agent_type/model/
reasoning_effort/service_tier` 仍因缺少完整 current owner 而 fail closed。该变更使旧候选永久失效，
只有 contracts、current fixture 和新候选 Wave 1 全绿后才允许进入 Gate A。

从旧基线 `5c19155` 到当前基线的相关增量还要求 Gate 覆盖：paginated thread history 的
read/resume/materialization、paginated source fork 显式拒绝、`fork_turns=last-N` 对 trigger-turn
inter-agent message/rollback/startup prefix 的截断语义、`list_agents` 不返回 task/message 正文，以及
queue-only agent mail 在 final answer 后延迟到下一 Turn、显式 follow-up 才重新采样。这些是 current
Codex 事实，不允许用旧 history 全量装载或 Renderer synthetic mailbox 状态替代。

## 3. 写集与避让集

### 3.1 本计划阶段写集

- `internal/exec-plans/project-gate-a-b-acceptance-plan.md`
- `internal/exec-plans/README.md`
- `.gitignore` 中本计划的精确跟踪例外
- `internal/test/project-gate-surfaces.manifest.json`
- `scripts/agent-qc/project-gate-candidate*`
- `scripts/agent-qc/project-gate-coverage*`
- `scripts/lib/project-gate-candidate-core.mjs`
- `scripts/lib/project-gate-coverage-core.mjs`
- `scripts/README.md` 与 `package.json` 中对应稳定入口
- `electron/main.ts`、`electron/smokeChecks.ts`、`electron/smokeEvidence*` 与
  `electron/smokeMemorySettings.ts` 中的 SHELL-01 smoke owner
- `scripts/electron/smoke.mjs` 与 `scripts/electron/current-entrypoints.test.mjs`
- `verify:gui-smoke` 与 current Settings fixtures 中的 SHELL-02 AgentRoot/config isolation evidence owner；旧 Product DB migration fixture 已删除

### 3.2 执行阶段写集

执行时只有以下内容可在未发现产品缺陷时写入：

- 原始本地证据：`.lime/qc/project-gates/<run-id>/**`
- 不可变摘要：`internal/research/refactor/v2/13-evidence/project-gates/<run-id>.md`
- 本计划的状态、阻塞和完成度字段

发现缺陷后必须先停止对应 lane，另开窄写集修复。修复完成会产生新的候选摘要和
`run-id`，不得覆盖失败证据，也不得在 gate-runner 过程中顺手修改产品代码。

## 4. 证据等级

| 等级      | 必经链路                                                                                          | 可以证明                                       | 不能证明                                                  |
| --------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------- |
| Preflight | lint/typecheck/unit/contract/governance                                                           | 构建图、静态边界和局部行为                     | GUI 或跨进程产品链                                        |
| Gate A    | Chrome/browser fixture -> Renderer -> 可控 bridge/read model                                      | 页面、projection、DOM、交互和文案              | Electron main、preload、IPC                               |
| Gate B-F  | 真实 Electron -> preload/IPC -> App Server -> external/unavailable fixture -> read model -> GUI   | 桌面主链、持久化、恢复和可见状态               | live provider 质量；external 模式下的最终 provider prompt |
| Gate B-R  | 真实 Electron -> App Server -> RuntimeCore -> 本地 provider/MCP/tool fixture -> read model -> GUI | RuntimeCore、provider lowering、工具和事件闭环 | 商业 provider 的网络与账号质量                            |
| Gate B-L  | Gate B-R 链路改用显式授权的 live provider                                                         | 生产 provider、鉴权、流式和媒体交付            | 其他 provider/地区的普遍可用性                            |
| Gate B-P  | 打包后的 app -> bundled sidecar/resources -> 同一产品链                                           | 包内容、启动、路径、升级/失败行为              | 未实际运行的平台                                          |

Gate B-F/B-R/B-L/B-P 都属于 Gate B 的不同运行环境，不建立第二套协议或业务后端。
最终报告必须写明 proof level，不能用 Gate A 或 external fixture 冒充 live provider 证据。

## 5. 候选冻结合同

### 5.1 冻结时机

先补齐本计划标记为“缺稳定入口”的最小 harness，再冻结产品候选。冻结后不再修改源码；
任何修复都会使候选失效并生成新 `run-id`。

### 5.2 候选摘要

每次运行至少记录：

```yaml
run_id: 20260715Txxxxxx-<short-digest>
git_head: <commit>
codex_reference_commit: <audited Codex commit>
blocking_trackers:
  - path: internal/roadmap/codeximport/implementation-tracker.md
    status: <ready | ready-for-gate | completed | closed>
surface_contract:
  path: internal/test/project-gate-surfaces.manifest.json
  surface_count: 34
  priority_counts: { P0: 17, P1: 17 }
  digest: <sha256>
product_snapshot_digest: <status + path + current file bytes digest>
git_diff_digest: <tracked git diff binary digest>
changed_paths: <path list>
digest_excludes: <Gate run logs and this mutable execution plan only>
pnpm_lock_digest: <sha256>
cargo_lock_digest: <sha256>
os_arch: <platform / version / arch>
node_npm: <versions>
rust_toolchain: <rustc / cargo>
electron_version: <version>
backend_mode: <unavailable | external | runtime | live>
provider_protocol: <none | responses | chat | anthropic | media>
secrets_present: <boolean markers only>
started_at: <timestamp>
```

不得在摘要、trace、截图旁文件或日志中保存 API key、Authorization、完整 system prompt、
provider request/response、用户真实对话正文或真实历史路径。

`product_snapshot_digest` 必须覆盖 tracked 与 untracked 产品文件，删除项以 status/path 进入
摘要。只排除 `.lime/qc/project-gates/**`、本计划和
`internal/research/refactor/v2/13-evidence/project-gates/**`；这些路径在候选
冻结后仍需追加执行记录，但不得改变产品代码。`git_diff_digest` 作为补充审计值，不能单独
代表包含 untracked 文件的当前候选。

### 5.3 冻结前只读检查

权威 candidate 入口：

```bash
npm run agent-qc:project-gate-candidate -- \
  --codex-reference-repo "/Users/coso/Documents/dev/rust/codex"
```

该命令先要求 Codex import tracker 明确进入 `ready`、`ready-for-gate`、`completed` 或 `closed`；
`active`、缺失和未知状态全部 fail closed。随后连续计算两次完整产品快照，间隔至少 5 秒；只有
product digest、tracked diff digest、Git HEAD、changed paths、exclusion、tracker 状态、34-surface
contract digest 和干净 Codex reference HEAD 全部一致才生成 candidate JSON。冻结前临时检查可用
`npm run agent-qc:project-gate-candidate -- --snapshot-only`，但单次结果不能签字。

每个 Wave 结束和最终签字前必须运行：

```bash
npm run agent-qc:project-gate-candidate -- --verify-candidate .lime/qc/project-gates/<run-id>/candidate.json
```

任一 digest、Git HEAD、changed paths 或 exclusion 漂移都使 candidate 立即失效。

```bash
git status --short
git diff --name-status
git diff --check
node --version
npm --version
rustc -Vv
cargo -V
```

候选摘要生成后再次运行 `git diff --check`。最终签字前重算 digest，必须与开始时一致。

## 6. Gate B 硬断言

每个 Gate B 场景必须同时满足：

1. `window.__LIME_ELECTRON__ === true`。
2. `typeof window.electronAPI?.invoke === "function"`。
3. 页面不是普通 Chrome 打开的 `127.0.0.1:1420` mirror。
4. invoke trace 至少命中 `transport: "electron-ipc"`。
5. invoke trace 命中 `command: "app_server_handle_json_lines"`。
6. trace 中存在本场景声明的 current JSON-RPC method。
7. GUI identity 与 read model identity 一致，例如 sessionId/threadId/turnId/itemId。
8. `console error = 0`、`page error = 0`、`invoke error = 0`，除非有带 owner 和退出日期的窄白名单。
9. legacy command 命中数为 `0`。
10. `defaultMocks`、`mockPriorityCommands`、`invokeMockOnly`、renderer mock fallback、
    App Server mock backend 命中数为 `0`。
11. 场景结束时有明确 terminal 或明确 pending 状态，不允许靠固定 sleep 猜完成。
12. 截图、结构化断言、filtered trace 和 read model 摘要属于同一 `run-id`。

Gate B-R 额外要求 RuntimeCore/backend/provider fixture 的 request marker 和 terminal event
能与同一个 Turn 对上。Soul 最终 prompt 只能保存 marker booleans，不能从 Renderer trace
里的旧 host options 猜测注入成功。

## 7. 优先级与停止规则

| 优先级 | 定义                                                           | 处理                                   |
| ------ | -------------------------------------------------------------- | -------------------------------------- |
| P0     | 启动、配置、Agent 主链、数据损坏、权限越界、生产 mock fallback | 首次失败立即停止后续依赖场景           |
| P1     | current 可见页面核心任务、恢复、插件/知识/自动化/资源主流程    | 可继续无依赖 lane，但不能签字          |
| P2     | 视觉、次级操作、非关键诊断和低频边界                           | 允许形成带 owner/期限的 release waiver |

以下情况无条件 stop-ship：

- 生产路径使用 mock fallback；
- Electron trace 没有 `electron-ipc` 或绕过 App Server；
- Thread/Turn/Item identity 串线、重复、丢失或恢复后变化；
- 用户数据被写入非平台路径或测试触碰真实用户目录；
- console/page/invoke 出现未分类错误；
- 五语言任一关键状态退回 raw key 或硬编码；
- packaged app 使用开发 sidecar、开发资源路径或 HTTP DevBridge 作为产品依赖。

## 8. 全产品覆盖矩阵

“现有入口”只表示可以复用，不表示当前候选已经通过。

| ID          | Surface                                     | Gate A                             | Gate B                                             | 现有入口                                        | 本轮动作                                     |
| ----------- | ------------------------------------------- | ---------------------------------- | -------------------------------------------------- | ----------------------------------------------- | -------------------------------------------- |
| SHELL-01    | 启动页、侧栏、页面导航、崩溃边界            | 全 Page route 渲染和切换           | Electron 冷启动、ready、重载                       | `verify:gui-smoke`                              | 补结构化 startup/route summary               |
| SHELL-02    | 配置、current AgentRoot、隔离 userData      | schema/consumer 单测               | 冷启动、重启、current root 注入、无权限失败        | `verify:gui-smoke` + current Settings fixtures  | 补 current AgentRoot 隔离与权限失败场景      |
| AGENT-01    | 新会话、发送、stream、terminal              | projection/state tests             | complete、failed、canceled、stale                  | `claw-chat-current-fixture`                     | 复用并统一证据字段                           |
| AGENT-02    | queue、steer、cancel、继续输出              | reducer/command model              | multi-queue、pop-front、cancel-then-continue       | 同上 scenarios                                  | P0 全跑                                      |
| AGENT-03    | 历史、分页、重启、导入、归档                | history/read/pagination projection | Electron history、Codex import、cold restart       | 多个 history/import fixture                     | 补 pagination、archive/unarchive 后恢复      |
| AGENT-04    | Approval、sandbox、执行策略                 | decision/form tests                | resume/decline/cancel/full-access                  | claw approval scenarios、approval sandbox       | 复用，补 app restart pending approval        |
| AGENT-05    | Tool/MCP output、sidecar、artifact          | item renderer/projection           | typed Tool row、structured output、large output    | managed tool smoke                              | 补大输出/磁盘失败                            |
| AGENT-06    | Multi-Agent graph/mailbox/GUI               | canonical roster tests             | 六工具、fork/mail boundary、cold restart           | `smoke:agent-control-cold-restart-gate-b`       | 补 Codex last-N/final-answer/privacy 边界    |
| AGENT-07    | MCP config/tool/resource/prompt/elicitation | schema/manager/GUI tests           | config、workspace runtime、form、fault isolation   | MCP Electron fixtures                           | 复用；补 OAuth/reconnect GUI B               |
| AGENT-08    | Skills、专家、skill gate                    | catalog/selection/invocation tests | skills-runtime、expert plaza/panel                 | claw skills scenarios、expert live gate         | fixture 必跑，live 单列 B-L                  |
| PROVIDER-01 | Provider CRUD、model/capability、迁移       | schema/lowering/capability         | 设置保存 -> 新 Turn -> effective route             | provider migration fixture                      | 补多协议 B-R 与 auth recovery                |
| HOME-01     | 新任务首页、附件、入口路由                  | desktop/compact/narrow             | 真实输入、图片、route/session 创建                 | home-hotpath scenarios                          | 复用并补键盘/拖放                            |
| WORK-01     | 通用 Workspace、timeline、Right Surface     | projection/visual matrix           | read model -> timeline/workbench                   | right-surface scenario                          | 复用；三视口截图                             |
| WORK-02     | Coding Workbench、文件、patch、terminal     | VM/component tests                 | GUI coding input、recovery、file tabs              | code artifact + session files Electron fixture  | 复用已注册入口；补磁盘失败                   |
| WORK-03     | Content Factory、Article Editor             | article/action/history tests       | host generation、editor、recovery                  | content factory fixture                         | 复用；cloud/live 单列                        |
| WORK-04     | 图片、音频、视频、media reference           | media projection tests             | image task/read model/viewer                       | image-command、media-reference、claw-image-live | 图片已有；音频/视频需新增 B-R                |
| WORK-05     | Browser Assist / Browser Runtime            | intent/view model tests            | attach、read、action、detach、恢复                 | `smoke:browser-runtime`                         | 现入口偏 bridge；新增 Electron visible-DOM B |
| PAGE-01     | Experts                                     | 列表、筛选、启动参数               | 启动专家 -> 新 Thread -> skill evidence            | expert plaza scenario                           | 补空态/失败/重启                             |
| PAGE-02     | Skills                                      | store/builtin/installed/manage     | 安装、启用、执行、更新、卸载                       | skills-runtime scenario                         | 补真实 Skills 页面 Electron B                |
| PAGE-03     | Plugins / Plugin Runtime                    | catalog/install state/UI contract  | 安装、侧栏入口、iframe/task、卸载                  | `smoke:plugin-*-electron-fixture`               | 稳定 npm 入口已注册；纳入 P1                 |
| PAGE-04     | Knowledge                                   | overview/import/detail/save/states | 选择资料 -> Agent -> 保存回资料                    | `knowledge:product-e2e`                         | 当前是 Chrome+DevBridge；新增 Electron B     |
| PAGE-05     | Automation                                  | job list/editor/history projection | create/run/cancel/restart/history                  | `smoke:automation-current`                      | 当前是 Gate A；新增 Electron B-R             |
| PAGE-06     | Channels                                    | config/status/log projection       | 保存、连接失败、重启恢复                           | 无统一 Gate B                                   | 新增隔离 fixture，禁止真实发送               |
| PAGE-07     | Resources                                   | 文档/图/音/视频列表与筛选          | 导入、预览、打开到 Agent、删除                     | 零散组件测试                                    | 新增 Electron B-F                            |
| PAGE-08     | Browser Runtime 独立页                      | 页面/调试面板                      | attached Chrome current session 操作               | browser runtime smoke                           | 补页面点击和断线恢复                         |
| SETTINGS-01 | 现有 SettingsTabs 全量                      | 每 tab 渲染、保存、错误文案        | provider/MCP/memory/policy/env/chrome relay 持久化 | provider/MCP fixture                            | 建立 settings tab 参数化 Gate B              |
| HOST-01     | 文件/目录、外链、通知、窗口、托盘           | capability/disabled state          | preload IPC 与 OS 行为                             | GUI smoke 部分覆盖                              | macOS/Windows 分平台新增                     |
| HOST-02     | Deep Link、单实例、恢复导航                 | parser/navigation tests            | cold/warm open、重复事件                           | `smoke:connect-*-deep-link-current`             | npm 入口已注册；补真实 protocol handler      |
| RELEASE-01  | packaged app/sidecar/assets                 | release guards                     | package dir 冷启动、失败提示                       | `electron:verify:package`                       | macOS 和 Windows 都要实跑                    |
| RELEASE-02  | updater                                     | feed/schema/guard                  | upgrade、拒绝旧版本、失败恢复                      | local feed + workflow guards                    | macOS ZIP 与 Windows Squirrel 实证           |
| CROSS-01    | 五语言                                      | key/unused/hardcode                | 关键状态逐语言 visible DOM                         | i18n scripts                                    | P0 场景五语言，P1 页面抽样                   |
| CROSS-02    | 主题、响应式、可访问性                      | DOM/a11y assertions                | desktop/compact/narrow、键盘路径                   | 多个 visual scenario                            | 建统一视口与 axe 报告                        |
| CROSS-03    | 安全、隐私、凭证                            | contract/secret scan               | 权限拒绝、日志脱敏、隔离数据目录                   | 零散 guard                                      | 建立项目级 negative evidence                 |
| CROSS-04    | 性能、并发、资源                            | benchmark/unit                     | 冷启动、首 token、双 session、长流、重启           | startup/Agent QC 零散入口                       | 形成基线和阈值，不以单次数字签字             |

### 8.1 优先级与固定分母

结构化事实源是 `internal/test/project-gate-surfaces.manifest.json`。它进入 product snapshot，candidate
还会单独记录其 SHA-256；本节是人类可读投影。覆盖矩阵共 `34` 个 surface，candidate 冻结后不得
通过删行、合并 ID 或降级优先级改变分母。

- P0（`17`）：`SHELL-01`、`SHELL-02`、`AGENT-01` 至 `AGENT-07`、`PROVIDER-01`、
  `HOME-01`、`WORK-01`、`SETTINGS-01`、`HOST-01`、`RELEASE-01`、`CROSS-01`、`CROSS-03`。
- P1（`17`）：`AGENT-08`、`WORK-02` 至 `WORK-05`、`PAGE-01` 至 `PAGE-08`、`HOST-02`、
  `RELEASE-02`、`CROSS-02`、`CROSS-04`。
- 当前没有 P2 surface。P2 只允许描述 surface 内不影响主任务的次级视觉/诊断缺陷，不能把整个
  current 页面、宿主能力或产品工作流降为 P2。

每个 surface 只能处于以下一种状态：

| 状态          | 计入完成度 | 含义                                                     |
| ------------- | ---------- | -------------------------------------------------------- |
| `unstarted`   | 否         | 当前 candidate 尚无 Gate A/B 证据                        |
| `gate-a-only` | 否         | Renderer 投影已通过，但没有匹配风险的真实产品链证据      |
| `gate-b-only` | 否         | 有真实链证据，但 Page/状态/语言/视口的 Gate A 覆盖不完整 |
| `complete`    | 是         | 同一 candidate 下 Gate A 与所需 Gate B 均通过            |
| `blocked`     | 否         | 有 owner、失败分类、阻塞证据和退出条件，但仍禁止签字     |

P0 不允许 release waiver；任一 P0 非 `complete` 都是 stop-ship。P1 可记录 blocker，但最终项目
Gate 签字前仍必须全部为 `complete`。

### 8.2 稳定入口责任账

`T0` 是 schema v3 candidate 成功冻结的时间。以下 12 个互斥责任组覆盖全部 34 个 surface；
“现状/阻塞”不等于 waiver，target 到期仍未补齐时必须把对应 surface 标为 `blocked` 并停止签字。

| Surface ID                   | Owner                                                      | 现状/阻塞                                                      | Target           |
| ---------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- | ---------------- |
| `SHELL-01/02`、`HOST-01/02`  | Desktop Host + `app_paths`                                 | GUI smoke/Deep Link 部分可用；缺通用迁移、宿主负向和实机行为   | Wave 3/5，T0+6d  |
| `AGENT-01` 至 `AGENT-07`     | App Server + agent/thread/tool/MCP owners                  | current fixtures 深；缺 pagination、mail/fork、磁盘/OAuth 边界 | Wave 3/5，T0+6d  |
| `AGENT-08`、`PAGE-01/02`     | Skills + Agent GUI                                         | skills/expert fixture 可用；缺页面安装/失败/重启全链           | Wave 4/6，T0+8d  |
| `PROVIDER-01`、`SETTINGS-01` | model-provider + Settings                                  | provider/MCP migration 部分可用；缺参数化 tabs 与多协议恢复    | Wave 4/6，T0+8d  |
| `HOME-01`、`WORK-01/02`      | Agent GUI + Workspace                                      | hotpath/right-surface/code fixture 可用；缺键盘、磁盘失败      | Wave 3/5，T0+7d  |
| `WORK-03`、`PAGE-04/05`      | Content Factory + Knowledge + Automation                   | Gate A/专用 fixture 部分可用；缺统一 Electron B                | Wave 4/5，T0+9d  |
| `WORK-04`                    | media runtime + model-provider                             | 图片已有；音频/视频 B-R 尚缺                                   | Wave 4/6，T0+10d |
| `WORK-05`、`PAGE-08`         | Browser Runtime                                            | bridge smoke 可用；缺 Electron visible-DOM 与 detach 恢复      | Wave 4/5，T0+9d  |
| `PAGE-03`                    | Plugin Runtime                                             | 已注册 Electron fixture；需纳入同 run-id 证据                  | Wave 4/5，T0+8d  |
| `PAGE-06/07`                 | Channels + Resources                                       | 无统一 Gate B；需隔离 fixture，禁止真实发送                    | Wave 4/5，T0+10d |
| `RELEASE-01/02`              | release-workflow                                           | package/local feed guard 可用；缺 macOS/Windows 实际闭环       | Wave 7，T0+13d   |
| `CROSS-01` 至 `CROSS-04`     | quality-workflow + locale/a11y/security/performance owners | 零散守卫可用；缺统一五语言、axe、negative、基线证据            | Wave 2-8，T0+15d |

## 9. 环境矩阵

### 9.1 必跑平台

| 环境                   | Gate A             | Gate B-F/R        | Gate B-P                          | 目的                |
| ---------------------- | ------------------ | ----------------- | --------------------------------- | ------------------- |
| macOS arm64 当前开发机 | 全量               | 全量              | package dir + ZIP/updater         | 主验收环境          |
| Windows x64 CI/实体 VM | Gate A 核心        | P0/P1 宿主与主链  | Squirrel install/update/uninstall | 验证平台 API 和路径 |
| macOS x64 CI/实体机    | 核心抽样           | P0                | package/签名抽样                  | 验证架构差异        |
| Linux CI               | 静态/Rust/contract | 不作为桌面 Gate B | 不适用                            | 编译和跨平台纯逻辑  |

Windows Gate B 不能由 macOS 编译成功替代。窗口、托盘、路径、Squirrel 和 updater 必须在
Windows 运行。签名、公证、发布 feed 属于 release candidate gate，不在普通本地 fixture 中伪造。

### 9.2 数据与端口隔离

每个 Electron lane 必须使用独立：

- `ELECTRON_E2E_USER_DATA_DIR`
- `LIME_AGENT_RUNTIME_ROOT`
- `HOME`、`APPDATA`、`LOCALAPPDATA`、`XDG_DATA_HOME`
- Electron CDP、Vite、DevBridge、provider fixture、MCP fixture 端口
- project/workspace 临时目录

测试不得读取真实 `~/.codex`、真实 MCP 配置、真实 provider keychain 或用户工作区，除非场景
明确属于只读 real-sample audit，且证据不保存正文和真实路径。

## 10. 执行波次

### Wave 0：补 harness 与冻结候选

目标：先修测试系统缺口，再冻结产品代码。

1. 为已有但未注册的 plugin UI/runtime、session files、automation 和 deep-link 脚本补稳定
   `package.json#scripts` 入口，不复制脚本实现。
2. 新增最小参数化 Electron harness，优先复用现有 `_electron.launch`、临时 runtime root、
   trace 过滤和 summary writer。
3. 只补矩阵中明确缺失的 P0/P1 场景，不建“万能 E2E 框架”。
4. harness 自测通过后生成候选摘要，之后禁止继续改源码。

退出条件：每个 P0/P1 surface 要么有稳定入口，要么已进入 8.2 的责任账；candidate 冻结后开始计算
`T0` 期限，任何到期未补齐项转为 `blocked`，不得静默延后。

#### 2026-07-15 Wave 0 执行记录

- 已为 Automation、Deep Link、session files、Plugin UI/runtime/task 的 8 个既有脚本补稳定
  npm 入口；只注册已有实现，没有新增平行 harness。8 个 `--help` 入口全部通过，相关 fixture
  单测 `13/13` 通过。
- 首次 `npm run test:contracts` 发现 Renderer 已删除 `recentTeamSelection`，而 App Server
  protocol、Rust read model、generated client 和正向测试仍保留。已按无兼容原则删除整条
  `recent_team_selection` surface，并把合同改为负向回流守卫。
- 提交边界现在无条件清理 retired Team selection/shadow metadata；路线图、Playwright 指南
  和 S6u evidence 已同步到 canonical SubAgent Thread/Turn/Item owner。
- 已删除 `runtime_backend/tests.rs` 中零引用的 `TestMcpAutostartDataSource` 死夹具。权威
  `npm run test:rust:related -- lime-rs/crates/app-server/src/runtime_backend/tests.rs` 通过
  `1124/1124`，无该夹具的 unused warning。
- raw `cargo test -p app-server --lib` 在 macOS libtest 默认小栈下触发 MCP async fixture
  stack overflow；这是非权威命令环境。仓库 runner 按既有合同设置 `RUST_MIN_STACK=8MiB`，
  同一精确测试与 app-server 全量 `1124/1124` 均通过，不作为产品缺陷或 waiver。
- 已通过：`npm run test:contracts`、`npm run governance:scripts`、
  `npm run governance:legacy-report`（零引用/分类漂移/边界违规均为 0）、
  `npm run verify:app-version`、`npm run typecheck`、App Server protocol `50/50`、
  受影响前端 `309/309`，以及相关 Rust 19 crate 全量。
- 全 workspace `cargo fmt --check` 仍会报告本轮窄写集之外的 Agent/MCP 格式差异；本轮
  Rust 文件 `rustfmt --check` 通过。该事实保留到 Wave 1，由 `verify:local:full` 判定候选。
- 后续窄验证又通过：`npm run lint`、`npm run typecheck`、`npm run test:contracts`
  （App Server client contract `289 checks`）、五语言结构与 unused 检查、受影响
  Harness/Workspace `252/252`、runtime lifecycle `7/7`、projection package `293/293`、
  App Server `1124/1124`、Agent Runtime `116/116`、相关 Rust runner 18/19 crate 扩展及
  `git diff --check`。这些验证只证明当时工作树，不可替代新候选的 Wave 1 aggregate。
- 三次候选均已失效。重新冻结前必须先消除 active fact-source 矛盾，连续两次计算完整产品
  快照 digest（间隔至少 5 秒）并确认 changed path 集完全一致；随后生成全新 `run-id`，不得
  复用或覆盖上述失败候选。
- active roadmap 的 raw status/synthetic Team 矛盾已收口；`governance:legacy-report` 为
  `0/0/0`，contracts `289 checks`、projection package 定向测试 `54/54`、相关 Vitest 通过。
  产品快照连续两次得到 `751fa87b94582c1bc4a02a6830658490a347dc61def547dad55c5618c78ee3e5`
  和 140 个 changed paths，新候选 `20260715T134155Z-751fa87b9458` 冻结完成。
- 该候选的 Wave 1 前 55 批通过；第 56 批暴露 6 个 Workspace command-wiring source guard、
  synthetic `team.changed` projection 断言和孤儿 Team memory helper 正向测试。旧断言已改为 current
  domain owner 或直接删除，修复后的第 56-110 批通过，但不能与旧快照的前 55 批拼接签字。
- 随后的 `spawn_agent.fork_turns` 垂直切片进一步改变产品源码和 canonical history 语义，旧候选
  已标记 `invalidated`。重新冻结前必须完成 contracts、Rust related、current Agent runtime fixture、
  scoped format/diff 与连续 digest；新候选从 Wave 1 第 1 批完整执行。
- 2026-07-16 并行写集复核：相邻进程独占
  `lime-rs/crates/app-server/src/runtime/conversation_import/**` 及其直接 read-model 修复，本 Gate
  协调 lane 不夹写、不替该 owner 修改断言或生产 lowering。该 lane 完成前不生成候选 digest；
  完成后只消费其退出信号，按本计划重跑 current fixture、contracts、Rust related 和完整快照。
- 同日对本计划中的稳定入口做结构化审计；新增 candidate 入口后共引用 `46` 个唯一
  `npm run` 命令，`package.json#scripts` 缺失数为 `0`。Automation 的 raw `node` 示例已统一到
  已注册入口，避免计划绕过命令事实源。
- Claw Chat 既有 Electron fixture 的第一批 Gate B 通用硬合同已从场景断言中拆出专用 owner：
  summary 现在显式记录真实 Electron renderer、preload invoke、`electron-ipc`、
  `app_server_handle_json_lines` current method、retired command/mock fallback 命中数、page error 和
  renderer crash；证据只保留 method、计数和脱敏 marker，不保存请求正文。对应 contract/source
  guard `56/56`、`npm run test:contracts`（App Server client `290 checks`）、
  定向 ESLint、`npm run governance:scripts`、harness/modality/release/cleanup/docs boundary 均通过。
- 同一合同继续补齐 project `run-id` 和 evidence artifact 硬断言：Gate runner 可通过
  `--run-id` 或 `LIME_GATE_RUN_ID` 注入安全 ID，standalone 运行生成显式 ID；summary、脱敏 backend
  ledger 和 screenshot 必须同目录，且 screenshot 必须在合同断言前真实落盘。缺失 run-id、跨目录
  或截图失败都会令场景失败。
- 同一 owner 已补齐 `identity` 与 `terminal-or-pending`：只收集 Renderer DOM、IPC turn params、
  App Server response、external runtime ledger 和 read model 中的 session/thread/turn/item ID；正文不进入
  evidence。terminal/pending 必须由 GUI stop 状态与 read model terminal/pending 状态一致证明，runtime
  terminal event 只作附加 marker，不能替代 read model。identity 错配、GUI/read model 状态错配均有负向
  回归，terminal 与 pending 均有正向回归。本轮 contract/source guard `58/58`、定向 ESLint、
  `git diff --check`、`npm run governance:scripts`、harness contracts、CLI `--help` 和非法 run-id
  fail-closed 检查均通过。上述结果关闭 Wave 0 的通用观察合同缺口，但不替代候选冻结、Wave 1 或
  任何实际 Gate B 场景运行；首次真实 Electron 场景仍可能暴露 harness/product 接线缺陷。
- 已新增 `agent-qc:project-gate-candidate` 作为唯一 candidate digest 入口：覆盖 tracked、untracked、
  deleted path 和当前文件 bytes，显式排除 Gate 日志与本可变计划，并在单次读取内检查 inventory/stat
  漂移；只有两次至少间隔 5 秒的 product digest、tracked diff digest、Git HEAD、changed paths 与
  exclusion 全部一致才生成 candidate JSON。临时 Git 仓库与 CLI fail-closed 测试 `5/5`、CLI help、
  定向 ESLint、Prettier 与 `governance:scripts` 通过。真实工作树 `--snapshot-only` 在
  `2026-07-16 01:33:06 +08:00`
  只读覆盖 `8050` 个 path、`75` 个删除项，得到诊断 digest
  `31ef09e095ff4206de65a773643237db27db30b5ef08584c366944dde5a2fa85`；相邻 import owner 在捕获前
  仍新增/修改 `conversation_import/codex/history_builder*` 等 current 文件，tracker 仍为 `S2 in_progress`，
  所以该结果明确不生成 candidate，`target_candidate` 继续保持 pending。
- `2026-07-16 01:45 +08:00` 再次复核 import tracker，状态仍为 `active`、`CCI-002 in_progress`，
  import owner 最后写入时间已推进到 `01:43:04`。Gate 协调 lane 继续不运行 candidate freeze、
  current Agent fixture、contracts 或 Rust related，避免对正在漂移的源码签字。独立 Gate 基础设施
  已用正确 Vitest 入口复验 `8/8`，定向 ESLint、`governance:scripts` 与窄写集
  `git diff --check` 通过；这些结果只证明 harness，不计入 Wave 1 或 Gate A/B 完成度。
- 随后补跑 Claw fixture 完整 source/contract guard `55/55`、harness contracts 与 candidate CLI
  `--help`，全部通过；加上前述新增合同测试，本轮 Wave 0 harness 定向回归共 `66/66`。因此当前
  candidate freeze 的已知阻塞仅剩 import owner 尚未退出，而不是 Gate runner 或观察合同失败。
- 同日发现计划仍引用旧 Codex commit `5c19155`，已只读审计本地 Codex HEAD `2e4f556` 及
  `origin/main` 的单个 TUI-only 增量，并补入 paginated history、last-N/rollback、final-answer mail
  boundary 和 `list_agents` privacy 验收项。Candidate runner 升级为 schema v3：生成时要求显式干净的
  Codex reference repo，两次 HEAD 一致，并且只保存 commit hash、不保存外部路径。Candidate runner
  正/负向回归 `8/8`、ESLint、Prettier、`governance:scripts` 和 CLI help 全部通过。
- `2026-07-16 01:56 +08:00` 用 candidate runner 对真实工作树执行 `--snapshot-only` 成功，覆盖
  `8050` 个路径和 `75` 个删除项，诊断 digest 为
  `5967d885b169731b9bbe055acd748216bff0734256c2fa9f3b5e8123a1237199`。由于 import tracker 仍为
  `active / CCI-002 in_progress`，该 digest 不生成 candidate、不得用于 Wave 1 或 Gate A/B 签字。
- 同一 schema v3 runner 已增加 import owner fail-closed guard：tracker 只接受 `ready`、
  `ready-for-gate`、`completed` 或 `closed`，并把相对路径/状态写入 candidate；`active`、缺失、未知状态
  均禁止冻结。真实工作树完整 candidate 命令已验证因 `status=active` 在 snapshot 前非零退出，未生成
  candidate；`--snapshot-only` 仍只提供诊断能力。
- 已把可变计划中的完成度分母下沉为 `internal/test/project-gate-surfaces.manifest.json`：固定 34 个
  surface、`P0=17/P1=17`、owners 与每个 surface 所需 Gate A/B proof level。Manifest 进入产品 snapshot，
  candidate 另存其 SHA-256；ID、顺序、分母、owner 或 proof level 漂移全部 fail closed。真实
  `--snapshot-only` 覆盖 `8051` 个路径、`75` 个删除项，surface contract digest 为
  `8d6841b1438ed8509d3380435a626dbf38cfe1012b73f4bfd62f6307319a9667`；candidate 回归 `8/8`。
- `2026-07-16 02:45 +08:00` 完成 SHELL-01 既有 `verify:gui-smoke` owner 的结构化证据收口，未新增
  平行 Electron harness。`electron/main.ts` 只保留生命周期接线，Workbench 检查与 evidence writer
  收敛到 `electron/smokeChecks.ts`，主文件由 `1719` 行降至 `1545` 行；Memory Settings smoke 现在返回
  最终 route snapshot。runner 通过 `LIME_GATE_RUN_ID` 绑定同一项目 run，在
  `.lime/qc/project-gates/<run-id>/shell-01-electron-smoke/` 同目录写入 `summary.json`、脱敏
  `trace-summary.json` 和截图，只保留 route、method、transport、计数与布尔 marker。
- SHELL-01 summary 对 startup -> Workbench -> Memory Settings、真实 Electron renderer、preload invoke、
  `electron-ipc`、`app_server_handle_json_lines` current method、App Server host initialize、console/page/
  invoke/load/preload/crash/unresponsive、retired command、mock fallback、trace 与 screenshot 做 fail-closed
  断言；launcher 复核 run-id、零失败断言和 artifact 实体，非法 run-id 在创建临时 userData 前失败，
  超时有 SIGTERM/SIGKILL 收口。随后补齐同一 Electron 进程内 renderer reload：reload 前累计 page
  error，reload 后重新安装页面错误捕获并再次等待 Workbench ready，之后才进入 Memory Settings。
  claim boundary 明确不证明 provider Turn、Thread/Turn/Item identity、live provider 或 packaged app。
- 本轮通过：evidence 与 entrypoint 定向测试 `19/19`、全仓 `npm run typecheck`、
  `npm run typecheck:electron`、定向 ESLint、`node --check scripts/electron/smoke.mjs`、非法 run-id
  fail-closed、`npm run governance:scripts`、Electron host/preload bundle（`263 + 2` modules）和窄写集
  `git diff --check`。因 import tracker 仍为 `active`，按并行协议未运行真实 `verify:gui-smoke`、current
  Agent fixture、contracts、Rust related 或 candidate freeze；完成度保持 `0/34 = 0%`。
- 已新增 `agent-qc:project-gate-coverage` 作为 34-surface 完成度的唯一机器聚合入口。它读取 schema v3
  candidate、candidate 固定的 manifest 和同 run-id evidence，只认显式
  `surfaceProof.surfaceId/proof/complete=true`、`result=pass` 与全绿 assertions；不根据文件名、scenario
  或自由文本 `proofLevel` 推断覆盖。部分观察使用 `complete=false`，失败/blocked evidence 必须声明
  `failureClass` 与 `nextAction`。默认不足 `34/34` 非零退出，`--progress-only` 仅用于 Wave 过程报表。
  聚合器会输出每个 surface 的 `unstarted/gate-a-only/gate-b-only/blocked/complete`、缺失 proof、P0/P1
  分母与相对 evidence 路径，不复制请求、对话正文或凭证。
- Coverage core/CLI 回归 `7/7`、candidate 原有 schema v3 回归 `8/8`、CLI help、定向 ESLint、
  `governance:scripts`、package JSON 与 `git diff --check` 通过。真实临时 candidate 证明
  `--progress-only` 在 `0/34` 返回过程报表，严格模式同样写摘要但非零退出。SHELL-01 runner 现在只在
  startup、首次 Workbench、renderer reload 后 Workbench、Settings route 和全部 Gate B-F 硬断言同时
  通过时写 `surfaceProof={surfaceId:SHELL-01,proof:gate-b-f,complete:true}`；失败 launcher/runner 一律
  写 `complete=false`、`failureClass` 与 `nextAction`。该结构尚未在新 candidate 上真实运行，且
  SHELL-01 的 Gate A 仍未执行，因此当前聚合分子仍为 `0`。
- `2026-07-16 03:01 +08:00` 再次复核相邻 import owner：tracker 仍为 `active`，`CCI-002/CCI-005`
  仍为 `in_progress`，且 import 写集在 `02:58:45` 与 `03:00:40` 分别继续写入 `commit.rs` 和
  `tests/idempotency.rs`。因此本 Gate lane 保持 candidate-freeze blocked，不运行 current fixture、
  contracts、Rust related、真实 GUI smoke 或 snapshot；Codex reference 仓库保持干净，HEAD 仍为
  `2e4f55608b4ad26d9c48ea45a6fcd20bfd5e9fe8`。
- `2026-07-16 03:05 +08:00` 完成 Wave 0 冻结前基础设施复核：本计划引用的 `48` 个唯一
  `npm run` 入口缺失数为 `0`；candidate/coverage CLI help 通过；SHELL-01 evidence、Electron
  entrypoint 与 coverage 定向回归 `26/26`。真实工作树 candidate 命令再次在读取 snapshot 前因
  tracker `status=active` fail closed，未生成 candidate。相邻 owner 最新写入为 `03:02:13`
  `conversation_import/tests/idempotency.rs`，所以当前不是陈旧 tracker，而是产品源码仍在变化。
  Wave 0 当前可独立收口的 candidate、coverage 和 SHELL-01 harness 已就绪；下一步只能等待 owner
  进入 `ready/ready-for-gate/completed/closed`，随后按顺序运行 current fixture、contracts、Rust
  related、治理门槛与双 snapshot，不能提前进入 Wave 1 或 Gate A/B。
- `2026-07-16 10:13 +08:00` 本轮继续复核确认 tracker 仍为 `status=active`，`CCI-002/CCI-005`
  仍为 `in_progress`；import 写集最新文件时间仍为 `03:09:39`，未收到 owner 的退出状态。唯一
  candidate 入口以 Codex reference HEAD `2e4f55608b4ad26d9c48ea45a6fcd20bfd5e9fe8` 复跑，仍在
  snapshot 前 fail closed，退出码 `1`，未生成新 candidate 或 Gate artifact。Gate 完成度继续保持
  `0/34`；本轮只读核验及本计划记录未触碰 conversation import 写集。
- `2026-07-16 10:16 +08:00` 独立治理复核通过：`governance:legacy-report` 摘要为零引用候选、零分类
  漂移、零边界违规，`governance:scripts` 通过。`agent-qc:project-gate-candidate --snapshot-only`
  诊断覆盖 `8066` 个路径、`77` 个删除项，product snapshot digest 为
  `2b6d9fbed96b1e80a5176b3a9032d01dc8a6a456ad8aedb35546a7a91159d22c`，但因 tracker 仍 active 不得
  生成 candidate 或进入 Wave 1。
- `2026-07-16 14:05 +08:00` import owner 已把 `CCI-002` 至 `CCI-006/CCI-008` 标为 completed，
  `CCI-007` 为 `completed-with-scope`，整体进入 `S4 验证与架构确认 / 95%`；剩余超大 rollout
  性能、Windows/压缩历史证据和责任开发者架构确认，tracker 因而仍为 active。相邻进程继续新增
  `conversation_import/tests/performance.rs` 并运行 Rust 回归，本 Gate lane 未进入其写集。当前 HEAD
  `a44d88585b7d3f216f5a860789506c9a8225d134` 的 snapshot-only 覆盖 `7993` 个文件、`2` 个删除项，
  product snapshot digest 为 `dda80e0d732c67ce7be91295dbf25ade061f661f8865c57cae771e5e1aa9b60f`；
  34-surface contract digest 未漂移，但该诊断仍不能生成 candidate。
- `2026-07-16 15:42 +08:00` 在继续避让 import 热区的同时，SHELL-02 既有 Provider migration
  Electron fixture 已补 project run-id 与结构化 partial evidence。原 `1212` 行 runner 把 seed、临时
  runtime、脱敏和 stdio JSON-RPC 逻辑抽到 `scripts/electron/lib/` 后降为 `944` 行；summary 现在明确
  断言真实 Electron renderer/preload、`electron-ipc`、`app_server_handle_json_lines`、current methods、
  migration marker、旧库 schema 清理、目标 DB、GUI Provider、console/page/invoke/crash 和 screenshot。
  该 lane 只声明 `claimScope=provider-migration-only`、`complete=false`，并列出尚缺的 restart 与
  permission-failure，coverage 可识别但不会提前增加 SHELL-02 分子；失败 evidence 会带
  `failureClass/nextAction` 并标为 blocked。定向 Vitest 与 coverage/entrypoint contract `27/27`、
  ESLint、Node syntax、Prettier、`governance:scripts`、非法 run-id fail-closed 和窄写集
  `git diff --check` 通过。因 tracker 仍为 `active / 95%`，未运行真实 Electron fixture。
- `2026-07-16 15:55 +08:00` 随后以四个独立 standalone run-id 实跑 SHELL-02 migration fixture，
  保留前三次失败证据并逐层收口 harness 漂移：v1 暴露旧 `{provider}` create DTO，v2 暴露旧
  snake_case ProviderInfo 读取，v3 证明 direct preload invoke 在 GUI gateway 调用前不会产生
  `safeInvoke` trace。最终 v4 `standalone-shell-02-20260716T1553-v4` 通过，summary `15/15`，真实
  Electron renderer/preload、`electron-ipc`、`app_server_handle_json_lines`、current provider methods、
  migration marker、旧 DB schema 清零、目标 DB、GUI 可见 Provider、零 console/page/invoke/crash 和
  screenshot 均有同目录证据；截图人工复核非空、无重叠。summary 只保存同目录相对 artifact 名，
  API key 已脱敏。该证据仍明确 `complete=false`，因为 restart 与 permission-failure 尚未补齐，故
  SHELL-02 和项目总完成度都不增加。
- 同一轮定向 Vitest/entrypoint/coverage contract `28/28`、ESLint、Prettier、Node syntax、
  `governance:scripts` 与窄写集 `git diff --check` 通过。`npm run test:contracts` 在相邻 AgentUI
  event-store/tool-lifecycle 重构的三条 source guard 上失败；只读 diff 证明 sequence/tool lifecycle
  校验已收敛到 `EventValidationContext.validate_and_observe`，不是产品校验被删除，而是 contract 仍匹配
  旧逐事件调用。对应 Rust owner 正在其他写集修改，本 Gate lane 不夹写，保留为 candidate 前 preflight
  blocker。import tracker 同时仍为 `active / 95%`。
- `2026-07-16 23:47 +08:00` 已补齐 SHELL-02 restart 与 permission-failure，并把迁移失败从兼容回退
  改为 current owner fail closed。`core::app_paths` 在目标 Product DB 不可替换时返回
  `数据库迁移失败，拒绝回退旧路径`，不写 marker、不把旧 DB 当 current owner、源 DB 保持不变；
  `npm run test:rust:related -- lime-rs/crates/core/src/app_paths.rs` 通过。设置页在 Provider 读取失败时显示
  五语言通用权限错误和重试入口，不再把失败误投影为空列表，也不展示原始本地路径；组件 `18/18`、i18n
  `10/10`、定向 ESLint 与 JSON parse 通过。
- 最终真实 Electron evidence 为
  `standalone-shell-02-final-v3-20260716T154657Z/shell-02-provider-migration/`：summary
  `42/42`，`surfaceProof={surfaceId:SHELL-02,proof:gate-b-f,complete:true}`，
  `claimScope=shell-02-config-path-migration-isolation`，`missingScenarios=[]`。首次启动与同一 `userData`
  重启均证明 renderer、preload、`electron-ipc`、`app_server_handle_json_lines`、`modelProvider/list`、
  `modelProviderUiState/read`、Provider/API key/custom model/UI state 和 GUI 可见状态一致；legacy Provider
  command、console/page/invoke error、renderer crash 均为零。
- permission-failure 使用第二套隔离 Product DB 与只读 App Server data-dir，真实 Electron GUI 显示
  `provider-load-error` 和可用重试按钮；IPC 上 `modelProvider/list` 明确失败，失败原因命中 fail-closed
  migration error。源 DB SHA-256 与 `86` 个 schema object 保持，marker 与目标 DB 均不存在。
  三张 `1440x1000` 截图已人工复核非空、无重叠；evidence secret scan 只命中 `sk-[redacted]`，未命中
  真实 API key、用户目录或私钥。失败 run 保留了 Web renderer 绝对资源污染、Electron 关闭后第二次 seed
  `SIGABRT` 和跨重启 localStorage 关机 buffer 三类 harness 诊断；最终 runner 已改为 Electron 启动前完成
  两套 seed、按 launch timestamp 隔离 trace/error 并去重，定向 entrypoint/coverage/fixture 回归 `31/31`。
- 该 Gate B-F 证据完整关闭 SHELL-02 的迁移、重启、隔离与权限失败场景，但不是冻结候选 evidence。
  Manifest 对 SHELL-02 的 required proofs 仍是 `gate-a + gate-b-f`；Gate A 和新 candidate 尚未形成，故
  SHELL-02 不增加项目分子，完成度继续为 `0/34`。Windows chmod 注入未实现，不能把 macOS 证据冒充
  Windows platform proof。
- `2026-07-16 23:54 +08:00` 扩大当前工作树门禁通过：`npm run test:contracts` 完整退出 `0`，
  protocol generated types 无漂移、App Server client `291 checks`、command/harness/modality/release/docs
  boundary 均通过；此前 event validation stale source guard blocker 已关闭。`governance:legacy-report`
  摘要继续为零引用候选、零分类漂移、零边界违规，`governance:scripts` 通过。Provider 设置页组件
  `18/18`；五语言 missing/hardcoded 检查为零，`settings` unused key 为零。
- 同次 `npm run verify:gui-smoke` 真实通过并生成
  `standalone-shell-01-20260716155351-75214/shell-01-electron-smoke/`：SHELL-01 Gate B-F
  `21/21`、`surfaceProof.complete=true`。证据覆盖 startup、Workbench、renderer reload 后 Workbench、
  Memory Settings、真实 Electron/preload、34 次 `app_server_handle_json_lines` IPC 和 App Server
  `appserver.v0` initialize；console/page/invoke/load/preload/crash/unresponsive、legacy command 与 mock
  fallback 均为零。截图人工复核非空、无重叠，secret scan 未命中真实路径、密钥或私钥。
- Codex reference 仓库保持干净，HEAD 仍为 `2e4f55608b4ad26d9c48ea45a6fcd20bfd5e9fe8`。唯一 candidate
  命令随后按预期在 snapshot 前 fail closed：`internal/roadmap/codeximport/implementation-tracker.md`
  仍为 `status=active / S4 / 95%`，只接受 `ready/ready-for-gate/completed/closed`。退出码 `1`，未生成
  candidate 或项目 Gate artifact；SHELL-01/02 standalone B-F 仍只作为 harness/product 诊断证据，
  项目完成度保持 `0/34`。
- `2026-07-17 01:21 +08:00` 的 `npm run smoke:agent-runtime-current-fixture` 在此前 history/cache
  `31/31`、streaming terminal `32 passed`、Electron fixture guards `75/75` 及普通 chat、Workbench、
  图片、cancel/continue、approval、rich draft、pending steer 场景通过后，稳定暴露 Plan history hydrate
  缺陷：reload 前 decision 已绑定 `proposed_plan:fixture-1`，reload 后 canonical plan thread item 仍保留
  revision/turn identity，但 decision 降级为 message candidate，revision identity 变为 `null`。失败 evidence
  为 `claw-chat-current-fixture-plan-history-hydrate-regression-summary.json`。
- 根因位于 current GUI selector
  `src/components/agent/chat/workspace/planImplementationDecision.ts`：message、thread item 与 plan state
  仅按完成时间排序，hydrate 后同一计划的较新 message 副本覆盖 canonical structured candidate。修复只在
  `planText` 完全相同时优先复用 `thread_item`、其次 `plan_state` identity；不同文本仍保留较新 message，
  没有新增 compat、重复 read model 或协议分支。回归覆盖“同文回绑 canonical revision”和“异文不被旧
  revision 覆盖”；selector/component `24/24`、定向 ESLint、`npm run typecheck` 均通过。
- `2026-07-17 02:01-02:14 +08:00` 三次重新构建后的真实 Electron 专项 run
  `claw-chat-current-fixture-plan-history-canonical-identity`、`...-v2` 与 `...-v3` 均证明 live GUI decision
  已正确绑定 `proposed_plan:fixture-1`，且 backend ledger 完整到 `turn.completed`。三次均在进入
  reload/hydrate 前的
  `agentSession/read` RPC 等待处以 `timed out waiting for app-server message after 30000ms` 终止；因此它们
  是有效的修复前半程诊断证据，不是完整 Gate B pass，也不得替代原 hydrate 场景复验。
- 新阻塞落在相邻 Codex import owner 正在修改的 `electron/appServerHost.ts` /
  `packages/app-server-client/src/connection.ts` 并发 transport read-lock 热区；Gate lane 按避让合同不修改、
  不回滚该写集。第三次已在大样本 import v26 退出、renderer 与 App Server sidecar 均按最新源码重建后
  复现同一超时，故不能再归为单纯资源竞争。退出条件：import owner 结束并冻结上述边界后，先由该 owner
  关闭确定性的 `agentSession/read` transport timeout，再以已构建资产重跑 plan history 专项，
  必须穿过 read-model、reload 和 hydrate，且 hydrate 后 `planDecisionRevisionBound=true`；随后从头重跑
  `npm run smoke:agent-runtime-current-fixture`。在此之前 candidate 与完成度仍保持 `0/34`。
- 热区 owner 级定向测试当前均为绿色：`packages/app-server-client` `66/66`（包括 pending request 下让出
  transport read 的并发用例），`electron/appServerHost.test.ts` + `electron/hostCommands.test.ts` `82/82`。
  因此退出条件不能只写“owner 单测通过”；必须补或跑到真实 Electron streaming turn 与随后
  `agentSession/read` 共用 connection 的集成证据，并以 plan history v3 的确定性超时作为回归输入。
- `2026-07-17 02:14-02:22 +08:00` 对上述 Plan v1-v3 超时重新做进程级归因后，确认不是 App Server
  transport/read-lock 产品缺陷。本机 `/usr/local/bin/git -> /usr/local/git/bin/git` 是 2021 年
  `x86_64/i386` 二进制，在 arm64 macOS 上连 `git --version` 都进入不可中断 I/O；App Server stdio 主循环
  同步等待 `projectGit/status` 时会连带阻塞后续 `agentSession/read`。`/usr/bin/git` 为 Apple Git 2.50.1
  `arm64e`，以系统目录优先的 PATH 重跑
  `claw-chat-current-fixture-plan-history-canonical-identity-v4-native-git-summary.json` 后完整穿过 read model、
  sidebar reload 与 hydrate，`planDecisionRevisionBound=true`。因此前述三次证据应重分类为
  `environment / incompatible git executable`，不再把 `electron/appServerHost.ts` 或 client transport
  当成未关闭的确定性缺陷。
- `2026-07-17 02:30-03:09 +08:00` aggregate 随后首次暴露 Content Factory 多文档 identity 缺口：会话
  同时展示交付清单与正文，旧 fixture 固定点击首个 `ArticleArtifactFrame`，并且 Renderer 合成的
  `preview-artifact-*` 未携带 worker canonical `artifactRef`。current 修复让 Article Workspace preview
  仅在真实 `artifactIds[0]` 存在时传播 `artifactRef/appServerArtifactRef`，Article frame 暴露只读
  `data-artifact-ref`；fixture 先从 public `agentSession/read` 读取本轮 worker article ref，再按同一 ref
  完成首次点击、编辑、reload/reopen、再次点击、artifact/read 与后置 read-model identity 对账，不再按
  DOM 顺序或中文标题猜测。纯 projection/component/fixture guards 共 `28/28`，TypeScript typecheck、ESLint、
  Prettier 与窄写集 diff check 通过。
- Content Factory 真实 Electron Gate B-R 通过证据为
  `claw-chat-current-fixture-content-factory-article-workspace-dynamic-ref-v5-summary.json`；默认调用方 PATH 下、
  不再手工前置 `/usr/bin` 的复验证据为
  `claw-chat-current-fixture-content-factory-article-workspace-native-path-v6-summary.json`。两次均证明真实
  Electron/preload、`electron-ipc`、`app_server_handle_json_lines`、worker/read model/GUI identity、双文档
  frame、编辑后恢复稿和 explicit terminal；legacy/mock/console/page/invoke error 均为零。截图人工复核中间
  两张文档卡与右侧 Article Editor 同时可见、无重叠。该专项不是冻结 candidate evidence，不增加分子。
- macOS arm64 Electron fixture PATH 已收敛到 `scripts/lib/electron-fixture-runtime-env.mjs` 单一 helper：
  Claw aggregate、session history 与 code artifact workbench 三条 current Gate B owner 统一前置
  `/usr/bin:/bin:/usr/sbin:/sbin` 并去重，其他平台保持调用方 PATH；不修改系统配置。fixture/runtime guards
  `97/97`、`governance:scripts` 和完整 `npm run test:contracts` 通过；contracts 包含 generated protocol
  `697` types 无漂移、App Server client `291 checks`、command/harness/modality/release/docs boundary 全绿。
  legacy report 继续为零引用候选、零分类漂移、零边界违规。
- `2026-07-17 03:13 +08:00` 从头恢复 aggregate 时，history/cache `31/31`、streaming terminal
  `32 passed`、Electron guards `75/75` 通过，随后在重建最新 sidecar 时被相邻 import owner 的
  `conversation_import/codex.rs` Rust move error 阻塞：`timestamp` 先传值移动，后又在 rollout event closure
  中 clone。Gate lane 按避让合同不修改该文件、不复用旧 sidecar 绕过编译；tracker 仍为
  `active / S4 / 95%`。退出条件是 import owner 恢复 App Server clean build 并退出 active，随后从头重跑
  aggregate。当前 candidate 仍未冻结，项目完成度保持 `0/34`。
- `2026-07-17 03:25-03:26 +08:00` 相邻 import owner 已修复上述 Rust move error；Gate lane 未修改
  conversation import 写集。随后从头运行的 `npm run smoke:agent-runtime-current-fixture` 完整退出 `0`：
  history/cache `31/31`、streaming terminal `32 passed`、Electron fixture guards `75/75`，以及首页热路径、
  Coding Workbench、图片意图、cancel/continue、approval、Inputbar queue/restore、Plan revision history、
  Skills、MCP structured content、media reference、Expert Skills 与 Content Factory Article Editor 全部通过。
  Plan aggregate evidence 为 `claw-chat-current-fixture-plan-history-hydrate-regression-summary.json`；Content
  Factory aggregate evidence 为
  `claw-chat-current-fixture-content-factory-article-workspace-regression-summary.json`，其中 canonical worker
  article ref、编辑后 reload/reopen 恢复、双文档 frame、explicit terminal、零 legacy/mock/console/page/invoke
  error 均成立。aggregate 明确 `liveProviderUsed=false`，只声明可重复 current fixture / Gate B，不冒充 live
  provider。
- 同轮 `npm run verify:gui-smoke` 完整退出 `0`，生成
  `standalone-shell-01-20260716192605-22462/shell-01-electron-smoke/summary.json`：SHELL-01 Gate B-F
  `21/21`，真实 Electron renderer/preload、`electron-ipc`、`app_server_handle_json_lines`、App Server
  `appserver.v0` initialize、Workbench reload 与 Memory Settings 均通过；33 次 current App Server IPC，
  console/page/invoke/load/preload/crash/unresponsive、legacy command 与 mock fallback 均为零。该 run 使用最新
  Renderer、Electron host 与 App Server sidecar 构建，但仍是 standalone 诊断证据。import tracker 仍为
  `active / S4 / 95%`，因此不生成 candidate、不进入 Wave 1，项目完成度继续保持 `0/34`。
- 同一工作树复跑 `npm run test:contracts` 时，首次在 `governance:scripts` 的裸 `git ls-files` 命中本机
  `/usr/local/bin/git` 后进入不可中断 I/O；Gate lane 中止该父会话并按既有环境归因重跑，不记录为产品失败。
  以 `/usr/bin:/bin:/usr/sbin:/sbin` 优先的 PATH 从头重跑后完整退出 `0`：generated protocol `697` types
  无漂移、App Server client `291 checks`，command/harness/modality/scripts/release/cleanup/docs boundary 全绿，
  mock priority commands 为 `0`。`npm run governance:legacy-report` 同样在系统 PATH 下通过，扫描源码 `2394`、
  Rust `1155` 个文件，零引用候选、分类漂移和边界违规均为 `0`；`npm run typecheck` 通过。
- `2026-07-17 05:10 +08:00` 将上述环境修复从 Electron fixture 专项提升为仓库质量 runner 的单一
  `scripts/lib/native-executable-env.mjs` owner：macOS arm64 子进程统一把
  `/usr/bin:/bin:/usr/sbin:/sbin` 前置并去重，其他平台保留调用方 PATH，不修改系统配置。Electron fixture
  helper 只保留领域包装；scripts governance、project candidate、Rust changed-scope、architecture
  confirmation、docs boundary、quality planner、local-ci 与 Agent QC verify wrapper 复用同一 owner。
  `scripts-governance-core` 的 Git 失败同时改为 fail closed，不再把失败静默投影为零 tracked 文件。
- helper、scripts governance、quality planner、verify wrapper 与 candidate 回归 `52/52`，定向 ESLint、
  Prettier、Node syntax 与窄写集 diff check 通过。未手工前置 PATH 的默认调用环境下，
  `npm run governance:scripts` 通过；`npm run agent-qc:project-gate-candidate -- --snapshot-only` 正常覆盖
  `8012` 个文件、`3` 个删除项，诊断 digest 为
  `1ee03b40dbd0520de6791694c7b0f52f8a7f67a2896132260187d36728f4cca5`；Rust `--changed --list`
  正确识别 `Cargo.lock` 并扩大到 workspace；完整 `npm run test:contracts` 在默认 PATH 下退出 `0`，包括
  docs boundary。tracker 仍为 `active / S4 / 95%`，所以该 snapshot 不生成 candidate、不进入 Wave 1，
  项目完成度继续保持 `0/34`。
- `2026-07-17 05:13 +08:00` 再次以当前工作树复核：tracker 仍为 `active / S4 / 95%`，相邻 import owner
  正在独立 `.lime/cargo-target/codex-import-v25` 中编译/运行 conversation import 专项 Rust 测试，因此不是
  陈旧状态。本 Gate lane 未修改该写集。默认 PATH 下的 snapshot-only 覆盖 `8012` 个文件、`3` 个删除项、
  `234` 个 changed paths，诊断 digest 为
  `caad6d0fc43af9d93e05cd265aa92c863adc2a58cedb0437c361f8b503e049b4`；34-surface contract digest 仍为
  `8d6841b1438ed8509d3380435a626dbf38cfe1012b73f4bfd62f6307319a9667`。quality planner 同时正确识别
  `232` 个当前改动并选择 integrity/i18n/frontend/Rust/bridge/GUI/docs 风险集合。
- 带只读 Codex reference 的正式 `npm run agent-qc:project-gate-candidate` 在读取 snapshot 前按 tracker
  guard 退出 `1`，明确报告只接受 `ready/ready-for-gate/completed/closed`；未生成 candidate 或 Gate artifact。
  所以当前仍停留在 Wave 0，不能运行 Wave 1，也不能增加 `0/34` 分子。
- `2026-07-17 09:11 +08:00` 完成 SETTINGS-01 standalone Gate A 的第一轮终态收口。v1
  `standalone-settings-a-v1/settings-01-gate-a/summary.json` 的 `9/17` 来自 runner 误连没有 Desktop Host bridge
  的普通 browser 环境，`ERR_CONNECTION_REFUSED`、更新/配置读取和 App Server invoke 失败统一重分类为
  `environment / wrong fixture`，不是 Settings 产品失败。切到明确的 browser projection fixture 后，v2 完成
  `72` 个观察点、`3` 个视口、`5` 种语言和 `16` 个 primary tab，结果 `15/17`；剩余失败精确落在 Renderer
  直接请求 OEM `/voice-model-catalog` 触发的 CORS console error，以及 `media-services` 截图仍带 `3` 个
  `.animate-pulse`，分别属于重复网络事实源和终态等待合同缺口。
- 语音目录现已收敛到
  `VoiceSettings -> safeInvoke -> Electron VoiceModelHost` 单一 `current` owner，Renderer 的 OEM `fetch`、解析和
  下载双轨已删除，未增加 compat。voice 定向回归 `73/73`；v3 结构化断言首次达到 `17/17`，但人工截图复核
  发现 About 把 `get_skill_package_file_association_status`、`Desktop Host current` 和
  `electron-host-diagnostic` 内部诊断当成产品能力展示。继续下钻确认文件关联 Host 命令只返回 degraded
  diagnostic，设置动作并不真实可用，因此 About 卡片、typed API/type/validator、Electron allowlist/dispatcher、
  `SystemUtilityHost` diagnostic、五语言死文案和正向测试已直接删除；旧命令只允许留在负向测试与 retired
  guard，分类为 `dead / deleted / forbidden-to-restore`。Electron Host 命令数由 `95` 收敛到 `93`，mock
  priority 仍为 `0`。
- v4 结构化结果仍为 `17/17`，但人工复核首次 About 截图仍停在 lazy fallback 的“正在加载关于页面...”，证明
  仅检查文本和 pulse 不能保证截图已到终态。Settings 统一 lazy fallback 已补 `role=status`、`aria-busy=true`
  与稳定 `data-testid=settings-page-loading`；Gate runner 现在等待可见 `aria-busy`、loading test id 和 pulse
  清零，并拒绝上述内部诊断、已删除文件关联命令及“尚未接入真实”文案。layout/core 回归 `19/19`。
- 最终 v5 evidence 为
  `.lime/qc/project-gates/standalone-settings-a-v5/settings-01-gate-a/summary.json`：`17/17`，覆盖同一组 `72`
  个观察点、`3` 个视口、`5` 种语言、`16` 个 primary tab；console/page/invoke error、可见 loading、problem
  text 和 viewport overflow 均为零。`7` 张截图已人工复核为终态，无混语、内部诊断、重叠或溢出。直接受影响
  测试 `92/92`、Gate core `5/5`、ESLint、Prettier、`npm run typecheck`、`npm run test:contracts`、
  `npm run i18n:check`、Settings `npm run i18n:unused`、`npm run governance:legacy-report` 与
  `npm run verify:gui-smoke` 均通过；contracts 记录 App Server client `291 checks`、Electron Host `93`
  commands 和 mock priority `0`，五语言 missing/extra 为 `0`，治理分类漂移与边界违规为 `0`。
- SETTINGS-01 v5 仍严格写入
  `surfaceProof={surfaceId:SETTINGS-01,proof:gate-a,complete:false}`；唯一缺口是
  `empty/loading/error component-state evidence`，补齐并接入 summary 前不得改为 complete，也不得增加项目分子。
  同轮最新真实 Electron SHELL-01 evidence 为
  `.lime/qc/project-gates/standalone-shell-01-20260717010740-64015/shell-01-electron-smoke/summary.json`，Gate B-F
  `21/21`，证明真实 Electron renderer/preload、`electron-ipc`、`app_server_handle_json_lines`、current App
  Server method 和用户可见状态，legacy/mock/console/page/invoke error 均为零；它仍是 standalone 证据，不是
  冻结 candidate。import tracker 复核仍为 `active / S4 / 95%`，所以不生成 candidate、不进入 Wave 1，项目
  完成度继续保持 `0/34`。
- `2026-07-17 09:28 +08:00` 已关闭 SETTINGS-01 的唯一 Gate A 缺口。`已归档对话` 作为现有 current
  component-state owner，为 loading 增加 `role=status / aria-busy=true`，为空态增加稳定 status marker，为
  error 增加 `role=alert`、稳定 error/retry marker；未新增测试专用产品 UI、compat 或 production mock。
  Gate A 使用 Playwright 网络层显式 test-only fixture，只拦截 `archivedOnly=true` 的 current
  `agentSession/list`：pending request 证明 loading，空 `sessions` 证明 empty，JSON-RPC error 证明本地化 error
  和可用 retry；其他 App Server request 全部 pass through。fixture marker 未进入用户可见文案，预期错误日志
  单独记录，不能掩盖其它 console error。
- evidence core 不再接受调用方直接注入 `stateCoverage.complete=true`，而是 fail closed 校验三条结构化观察的
  tab/viewport/locale、test id、role、busy/retry、fixture method/outcome、raw key、overflow 与截图归属。状态编排
  从 `801` 行 runner 拆到 `scripts/lib/project-gate-settings-a-states.mjs`，主 runner 回落到 `542` 行；纯
  JSON-RPC 拦截合同、evidence core 与组件回归合计 `14/14`，定向 ESLint、Prettier、`npm run typecheck`、
  `npm run governance:scripts` 和窄写集 `git diff --check` 通过。
- 最终 v6 evidence 为
  `.lime/qc/project-gates/standalone-settings-a-v6/settings-01-gate-a/summary.json`：`18/18`，
  `surfaceProof={surfaceId:SETTINGS-01,proof:gate-a,complete:true}`、`missingScenarios=[]`。常规矩阵仍为 `72`
  个观察点、`3` 个视口、`5` 种语言和 `16` 个 primary tab；state checks 为
  `loading=true / empty=true / error=true`，非预期 console/page error 为零。总计 `10` 张截图，其中三张状态
  截图已人工复核为真实 Settings 布局、无混语/泄漏/重叠/溢出，error 只显示本地化失败文案和重试按钮。
- 同轮 `npm run verify:gui-smoke` 完整退出 `0`，生成
  `.lime/qc/project-gates/standalone-shell-01-20260717012710-87315/shell-01-electron-smoke/summary.json`：
  SHELL-01 Gate B-F `21/21`，真实 Electron renderer/preload、`electron-ipc`、
  `app_server_handle_json_lines`、current App Server method、Workbench reload 与 Memory Settings 均通过，
  legacy/mock/console/page/invoke/load/preload/crash/unresponsive 全零。该 evidence 与 SETTINGS v6 都是
  standalone 诊断证据，不是冻结 candidate evidence。并发 import owner 已把 tracker 推进到
  `active / S4 / 97%`，但仍未退出 active；因此不生成 candidate、不进入 Wave 1，项目完成度继续保持
  `0/34`。
- SETTINGS-01 Gate B-F 复用盘点确认四条 current owner 已存在：SHELL-02 provider migration fixture 覆盖
  Provider 迁移/重启/权限失败，MCP config Electron fixture 覆盖 GUI create 与 `mcpServer/create/list`，
  Electron GUI smoke 覆盖 Memory Settings ready，session history Electron fixture 覆盖侧栏归档、Settings
  恢复和 archive/unarchive restart readback。它们分别服务 SHELL-02、MCP、SHELL-01 与 history owner，现有
  summary 的 run-id、claim 和 schema 不同；禁止直接改 `surfaceProof` 或拼接旧 summary 冒充 SETTINGS-01。
  下一刀应建立 SETTINGS-01 专用 Gate B-F 聚合合同，让同一 candidate/run-id 下的 owner evidence 以稳定
  scenario ID 登记并由聚合器 fail closed。仍缺的真实场景是 Provider CRUD/model selection/鉴权恢复、MCP
  update/delete/start-failure/restart、Memory AI 个性保存与恢复、Media Services readiness、Web Search route、
  Environment/Execution Policy allow-deny-error、Chrome Relay install/connect/disconnect、Appearance 主题持久化
  和 About 版本事实源；这些缺口关闭前 SETTINGS-01 只能是 `gate-a-only`。
- `2026-07-17 09:54 +08:00` 建立 SETTINGS-01 专用 Gate B-F 聚合合同：
  `scripts/agent-qc/project-gate-settings-b.mjs` 只接受同一
  `.lime/qc/project-gates/<run-id>/` 下的 owner summary，显式校验 candidate run-id、真实 Electron/preload、
  `electron-ipc`、`app_server_handle_json_lines`、current method、零 legacy/mock/error 和 owner assertions；19 个
  scenario 未全部出现前始终写 `surfaceProof.complete=false`。core/CLI 回归 `9/9`，首份同 run evidence
  `standalone-settings-b-v1-20260717T0954` 复用 Memory 与 Provider migration 两条真实 owner claim，聚合结果
  `2/19`，没有把 Soul、Provider GUI CRUD 或其它 Settings tab 猜成完成。
- 首次同 run Electron 构建被相邻 Codex import owner 的
  `conversation_import/commit.rs` 中整数类型推导阻塞；本 Gate lane 未修改该文件。相邻 owner 自行收敛为
  `usize` 并完成重建后，SHELL Memory `21/21`、Provider migration `42/42`、contracts、typecheck、scripts
  governance 与 legacy report 均通过。tracker 复核仍为 `active / S4 / 97%`，因此不生成 candidate、不进入
  Wave 1，项目完成度继续保持 `0/34`。
- `2026-07-17 10:16 +08:00` 将现有 MCP config fixture 升级为同 run 的结构化
  `settings-scenario` owner。run-id、路径和证据签字抽到
  `scripts/electron/lib/mcp-config-fixture-evidence.mjs`，原 784 行 runner 回落到约 635 行；只有真实 GUI 创建
  Context7、`mcpServer/create|list` 的 `electron-ipc` trace、App Server readback、零 legacy/mock/console/page/
  invoke error 和截图全部成立才完成 `mcp-create-list`。同 run evidence 为
  `settings-mcp-create-list/mcp-config-fixture-summary.json`，`14/14`；MCP 相关 current consumer/guard 回归
  `18/18`，人工截图确认配置页终态无 loading、遮挡、混语或内部诊断。
- `2026-07-17 10:21 +08:00` 新跑 session history owner 到同一 run 的
  `settings-archived-lifecycle-owner/`：侧栏 GUI 归档、Settings 恢复、archive/unarchive 与两次重启读回均通过，
  所有可见 trace 都是 `electron-ipc` current `agentSession/list|read|update`。但 owner 当前只记录 console error，
  没有独立 `pageerror` observer；且该 2400 行脚本属于相邻 import/history 脏写集，因此本 lane 不夹写、不把
  `pageErrorCount=0` 伪造进 SETTINGS source，`archived-lifecycle-recovery` 暂不计入。
- `2026-07-17 11:23 +08:00` 新增只读 About/Home Electron fixture，同 run evidence 位于
  `settings-about-version/`。首次结构化 About `16/16` 已证明 workspace、Renderer、Desktop Host
  `app.getVersion()` 都是 `1.106.0`，但人工截图发现 zh-CN 版本行误用英文 `Version`；修正资源为
  `版本 {{version}}（{{build}}）`，并把按 locale 校验版本标签加入 Gate 断言。复跑 About `17/17`、Home
  navigation `13/13`，截图分别确认 About 无混语/内部诊断、返回首页后 `home-start-surface` 可见且 Settings
  header 消失。About/Home core 与布局回归 `20/20`，About 组件与 evidence 回归 `13/13`，五语言 key 覆盖
  `100%`。
- 同轮 SETTINGS-01 Gate B-F 聚合提升到 `5/19`：已完成 `home-navigation`、`memory-ready`、
  `provider-migration-recovery`、`mcp-create-list`、`about-version-truth`；仍缺 14 项，summary 为
  `.lime/qc/project-gates/standalone-settings-b-v1-20260717T0954/settings-01-gate-b-f/summary.json`。
  `mcp-context7-live-fixture-smoke.mjs` 另补标准 executable entry guard，避免 import helper 时误启动 live 网络；
  guard 回归 `4/4` 且动态 import 只加载模块。该 live 产物不计入 SETTINGS evidence。
- `2026-07-17 11:35 +08:00` 收尾验证基于当前工作树通过：受影响定向回归 `56/56`、`npm run typecheck`、
  `npm run governance:scripts`、`npm run verify:app-version`、定向 ESLint/Prettier/diff check 全绿；完整
  `npm run test:contracts` 记录 protocol types `704` 无漂移、App Server client `291 checks`、Electron Host
  `93` commands、mock priority `0`，harness/modality/release/docs boundary 全绿；legacy report 扫描源码
  `2395`、Rust `1156`，零引用候选、分类漂移和边界违规均为 `0`。`npm run verify:gui-smoke` 完整退出
  `0`，最新 SHELL-01 evidence 为
  `.lime/qc/project-gates/standalone-shell-01-20260717033517-70603/shell-01-electron-smoke/summary.json`，
  Gate B-F `21/21`。
- 并行进程在本轮期间创建 `62d5c997d8ac19707d744094e08ec25c6c5e9d5e`（Release v1.106.0），把本轮
  早段 Gate/MCP/About/i18n 工作纳入 HEAD；本 Gate lane 未执行 add/commit/push/reset。最新 Codex import
  tracker 为 `active / S4 / 99%`，只剩责任开发者架构确认，仍不满足 candidate guard 的
  `ready/ready-for-gate/completed/closed`，因此不生成 candidate、不进入 Wave 1，项目完成度保持 `0/34`。
- `2026-07-17 11:55 +08:00` 继续补只读 Settings 场景。Stats fixture 经真实 Electron 命中
  `usageStats/read`、`usageStats/modelRanking/list`、`usageStats/dailyTrends/list`，隔离空数据以合法零值、空排行和
  空趋势展示，evidence `13/13`；core、组件和 current-boundary 回归 `11/11`。Environment fixture 初次因
  Renderer config cache 让 `get_config` 发生在 trace 清理前而 fail closed，失败截图证明页面本身终态；修正为挂载
  页面前更新既有 `lime.app-config.changed-at` 缓存戳，让正式 `getConfig()` gateway 重新走 Host，未绕过
  `safeInvoke`。最终真实 `get_config`、`get_environment_preview`、同窗口 App Server IPC 与 GUI 终态
  `16/16`，API/组件/core 回归 `25/25`，JSON evidence 不记录环境变量名或值。
- SETTINGS-01 Gate B-F 同 run 聚合现为 `7/19`：新增 `stats-current-read` 与
  `environment-current-read`，已完成场景为 `home-navigation`、`stats-current-read`、`memory-ready`、
  `provider-migration-recovery`、`mcp-create-list`、`environment-current-read`、`about-version-truth`；仍缺 12 项，
  `surfaceProof.complete=false`。candidate 状态与项目分子不变。
- `2026-07-17 12:22 +08:00` 完成 `media-services-readiness`。专用真实 Electron fixture 经
  `get_config`、`voice_models_list_catalog` 与 App Server `model/list`、`modelPreferences/list`、
  `modelSyncState/read` 证明服务模型、图片、视频和语音四个 current 配置区域进入终态；配置控件可用，
  loading/error、console/page/invoke、legacy/mock 均为零，结构化 evidence `23/23`。首版把
  `modelProvider/list` 设为页面后置 required method，暴露其可能在同一 renderer 冷启动预热并被模块缓存复用的
  取证竞态；最终合同改为该页面每次稳定消费的三条模型选择器 current read，没有放宽 Electron/preload/IPC
  或错误门槛。raw evidence 只保留协议、current method/command 与 readiness 布尔，不记录 file URL、本机路径、
  Provider 配置、模型 ID 或凭证；四张截图覆盖页面顶部及图片/视频/语音滚动位置。summary 为
  `.lime/qc/project-gates/standalone-settings-b-v1-20260717T0954/settings-media-services-readiness/settings-media-services-fixture-summary.json`。
- SETTINGS-01 Gate B-F 同 run 聚合提升到 `8/19`，新增 `media-services-readiness`；仍缺 11 项，
  `surfaceProof.complete=false`。下一刀进入 `web-search-route`，必须证明真实 current 配置读取与路由保存/恢复，
  不能只证明页签可见。candidate 状态与项目分子不变。
- `2026-07-17 12:31 +08:00` 完成 `web-search-route`。专用 fixture 在隔离用户目录内执行三次真实
  Electron 冷启动：首次从 GUI 切换搜索引擎并经 `get_config/save_config` 保存；第二次冷启动读回变更后从 GUI
  恢复原值；第三次冷启动确认恢复完成。三次均证明 Electron/preload、`electron-ipc`、
  `app_server_handle_json_lines` 与 current App Server method，同轮 console/page/invoke、legacy/mock 全零，结构化
  evidence `24/24`。JSON 只记录 lifecycle 布尔和 method/command，不记录路由值、搜索 Key、Provider 配置或完整
  config；三张截图只展示页面正常可见的保存、重启读回和恢复态。summary 为
  `.lime/qc/project-gates/standalone-settings-b-v1-20260717T0954/settings-web-search-route/settings-web-search-fixture-summary.json`。
- SETTINGS-01 Gate B-F 同 run 聚合提升到 `9/19`，新增 `web-search-route`；仍缺 10 项，
  `surfaceProof.complete=false`。下一刀优先合并执行 `profile-persistence` 与 `appearance-persistence` 的隔离持久化
  场景，但必须输出两个独立 owner summary，不能互相冒充。candidate 状态与项目分子不变。
- `2026-07-17 13:08 +08:00` 完成 `profile-persistence` 与 `appearance-persistence`。Profile 先暴露两类真实
  缺口：当前 branded runtime 默认展示 managed-account 登录页，local Profile 只属于 non-OEM 现役部署模式；
  fixture 因此使用正式 window OEM runtime override `enabled=false`，没有把本地资料冒充默认登录面。首次恢复空昵称
  失败又定位到 `ProfileSettings` 用 truthy 回退导致用户无法清空昵称/简介，已改为 nullish 归一并补组件回归；重建
  renderer 后，三次 Electron 冷启动的修改、读回、恢复、最终读回 `25/25`。Profile 组件、evidence core 与共享
  fixture 回归合计 `16/16`，summary 为
  `.lime/qc/project-gates/standalone-settings-b-v1-20260717T0954/settings-profile-persistence/settings-profile-fixture-summary.json`。
- 同轮修复共享 `launchElectronFixture` 的启动失败资源泄漏：`firstWindow/renderer ready` 抛错时现在先关闭已启动
  Electron 再原样抛错，并补静态守卫。该修复只影响测试 fixture 生命周期，不改变生产边界。
- Appearance 以同一场景成组验证两个 current owner：主题模式 localStorage 与推荐行为 `get_config/save_config`；
  三次真实 Electron 冷启动完成修改、双读回、双恢复和最终双读回，evidence `28/28`，structured JSON 不记录
  外观具体值。summary 为
  `.lime/qc/project-gates/standalone-settings-b-v1-20260717T0954/settings-appearance-persistence/settings-appearance-fixture-summary.json`。
- SETTINGS-01 Gate B-F 同 run 聚合提升到 `11/19`；剩余 `memory-soul-persistence`、
  `archived-lifecycle-recovery`、`provider-crud-model-auth`、`mcp-lifecycle-recovery`、
  `execution-policy-allow-deny-error`、`chrome-relay-lifecycle`、`automation-lifecycle`、
  `developer-current-diagnostics` 共 8 项，`surfaceProof.complete=false`。下一刀先做只读 Developer diagnostics，
  history/import 热区继续避让。candidate 状态与项目分子不变。
- `2026-07-17 13:35 +08:00` 完成 `developer-current-diagnostics`。首轮真实 Electron 运行虽然 GUI 已复制成功、
  payload shape 完整，但 Gate 因缺 `modelProvider/list` fail closed；根因是应用冷启动已预热 Provider 模块缓存，
  `collectRuntimeSnapshotForDiagnostic()` 又使用默认缓存读取，导致用户点击时的“当前诊断”可能复用旧快照。current
  owner 已改为 `apiKeyProviderApi.getProviders({ forceRefresh: true })`，没有新增命令、compat 或第二套诊断路径，并补
  单测锁住 fresh-read 语义。失败 summary/截图保留在默认 prefix；修复后用 `settings-developer-fixture-v2` 新 prefix
  复跑，没有覆盖失败证据。
- 最终 Developer fixture 从 GUI 点击“复制纯 JSON”，真实命中 `get_config` 与 App Server `log/list`、
  `log/persistedTail`、`diagnostics/server/read`、`diagnostics/logStorage/read`、
  `diagnostics/windowsStartup/read`、`modelProvider/list`、`mcpServerStatus/list`，Electron/preload/IPC 与
  `app_server_handle_json_lines` 全部成立，evidence `24/24`。显式 test-only renderer clipboard sink 只替代最终系统
  剪贴板写入并仅保留 write count、text length、JSON/payload shape 布尔；诊断采集仍全走真实 current owner，
  summary/raw 不保存剪贴板正文、日志、配置、路径、Provider/MCP 数据或凭证，也不声称系统剪贴板交付已验证。
  定向回归 `26/26`，ESLint、typecheck、完整 `npm run build` 均通过。summary 为
  `.lime/qc/project-gates/standalone-settings-b-v1-20260717T0954/settings-developer-current-diagnostics/settings-developer-fixture-v2-summary.json`。
- SETTINGS-01 Gate B-F 同 run 聚合提升到 `12/19`；剩余 `memory-soul-persistence`、
  `archived-lifecycle-recovery`、`provider-crud-model-auth`、`mcp-lifecycle-recovery`、
  `execution-policy-allow-deny-error`、`chrome-relay-lifecycle`、`automation-lifecycle` 共 7 项，
  `surfaceProof.complete=false`。history/import 热区继续避让；下一刀优先从非热区的
  `execution-policy-allow-deny-error` 或 `automation-lifecycle` 选择已有 current owner 做真实 GUI 生命周期闭环。
  candidate 状态与项目分子不变。
- `2026-07-17 13:48 +08:00` 完成 `automation-lifecycle`。职责盘点确认 Settings 中的 Automation current
  产品面只管理 scheduler config；job create/update/delete/runNow 已属于 Automation Workspace，禁止为了凑 Settings
  Gate 把两者重新耦合。专用 fixture 因此验证真实设置 owner：读取 `automationScheduler/config/read`、
  `automationScheduler/status`、`automationJob/list`、`automationJob/health`，从 GUI 同时改变启用、历史保留和
  轮询间隔并通过 `automationScheduler/config/update` 保存，第二次冷启动读回后从 GUI 恢复，第三次冷启动确认恢复。
- 首轮运行的四条 current read、Electron/preload/IPC、零 error/legacy/mock 均成立，但就绪 oracle 只匹配“持续流程”与
  “风险项”，没有接受真实页面“持续流程数 / 风险提醒”，因此在任何写入前 fail closed；默认 prefix 的失败 summary/
  截图保留。v2 只扩展本地化摘要正则，不改产品和 Gate method，最终三次冷启动 evidence `23/23`。summary/raw 只记录
  lifecycle 布尔与 method，不记录 scheduler 原值/变更值、job 数据、prompt、路径或运行输出；summary 为
  `.lime/qc/project-gates/standalone-settings-b-v1-20260717T0954/settings-automation-lifecycle/settings-automation-fixture-v2-summary.json`。
- SETTINGS-01 Gate B-F 同 run 聚合提升到 `13/19`；剩余 `memory-soul-persistence`、
  `archived-lifecycle-recovery`、`provider-crud-model-auth`、`mcp-lifecycle-recovery`、
  `execution-policy-allow-deny-error`、`chrome-relay-lifecycle` 共 6 项，`surfaceProof.complete=false`。
  history/import 热区继续避让，下一刀进入 `execution-policy-allow-deny-error`。candidate 状态与项目分子不变。
- `2026-07-17 15:02 +08:00` 完成 `execution-policy-allow-deny-error` 的 Settings Gate B-F。专用 fixture
  在隔离 userData 中从 GUI 保存 strict workspace restriction 与 Bash warning-bypass 两类策略输入，经
  `get_config/save_config` 真实 Electron Host current path 持久化并冷启动读回；随后把 fixture 自己的临时
  `config.yaml` 短暂替换成同名目录，要求真实 `save_config` 产生一条 `EISDIR` trace/error、页面展示错误，立即
  恢复文件并从 GUI 重新加载，最后恢复原策略并第三次冷启动确认。finally 对临时碰撞有独立恢复兜底，不触碰真实
  用户配置。
- expected save failure 与 unexpected error 分开：summary 明确记录 expected command/transport/cause 与 `1/1`
  trace/invoke error，聚合要求的 `errors.console/page/invoke/legacy/mock` 仍全零，不能通过吞掉未知错误达标。真实
  Electron/preload/IPC、同窗 App Server current method、两次成功保存和全部恢复 evidence `31/31`；summary/raw
  不保存 config 值、规则、prompt、路径或错误正文。该 proof 只证明 Settings policy input，不声称 RuntimeCore 已
  实际执行允许或拒绝工具；真实决策执行必须列入 Gate B-R。summary 为
  `.lime/qc/project-gates/standalone-settings-b-v1-20260717T0954/settings-execution-policy-allow-deny-error/settings-execution-policy-fixture-summary.json`。
- SETTINGS-01 Gate B-F 同 run 聚合提升到 `14/19`；剩余 `memory-soul-persistence`、
  `archived-lifecycle-recovery`、`provider-crud-model-auth`、`mcp-lifecycle-recovery`、
  `chrome-relay-lifecycle` 共 5 项，`surfaceProof.complete=false`。history/import 热区继续避让，下一刀进入
  `chrome-relay-lifecycle`。candidate 状态与项目分子不变。
- `2026-07-17 15:49 +08:00` 完成 `chrome-relay-lifecycle`，并把该 Settings 产品面从旧 Chrome relay
  facade 直接替换为 current Browser Session。四层盘点确认 `lime-rs/crates/browser-runtime` 与 App Server
  `browserSession/target/list|open|read|close` 已是唯一 runtime owner；原 3000 行设置页展示的扩展安装、系统连接器、
  backend priority、独立引导窗及其写操作没有 current owner，真实 Electron 只会命中静态 diagnostic facade 或
  fail-closed，因此按 `dead / deleted` 删除独立引导路由、4 个 guide/test helper 文件、旧正向 mock 测试和
  `settings.chromeRelay.*` 资源族。新页面只保留调试端口、page target、连接读回和断开，五语言资源改为
  `settings.browserConnection.*`；非 `page` CDP target 与缺少 debugger endpoint 的内部 target 会在 Renderer
  projection 过滤。
- 首轮 fixture 因 Playwright Chromium 缓存声明路径不存在而在 spawn 前失败，未形成合格 product evidence；runner
  随后增加 Playwright/system Chrome/Chromium 的现有可执行文件解析与 spawn error 捕获。`v2` 真实 Electron 失败
  evidence 证明陈旧 `dist` 仍渲染旧页面并命中 7 个 diagnostic facade；保留失败 summary/截图后执行完整
  `npm run build`。`v3` 已通过，但截图发现 extension background page 与 service worker 混入 target 列表；修复
  projection、补回归并重建 renderer 后使用 `v4` 形成最终证据，不覆盖前述 evidence。
- 最终 Gate B-R 使用隔离 userData 与本地 Chromium CDP target，经真实 Electron/preload/IPC、
  `app_server_handle_json_lines`、App Server、RuntimeCore 完成 target detect、select、open、readback、visible connected、
  close 与 visible closed，`21/21`，console/page/invoke/legacy/mock 全零。summary/raw 不保存页面正文、URL、session
  identity、端口、路径或浏览器日志；summary 为
  `.lime/qc/project-gates/standalone-settings-b-v1-20260717T0954/settings-browser-session-lifecycle/settings-browser-session-fixture-v4-summary.json`。
  Settings Gate A 在重构后以独立 run `standalone-settings-a-browser-v2-20260717T1548` 全量复跑，三视口、五语言关键
  tab、导航恢复及 archived component states 共 `72` observations，`surfaceProof.complete=true`。
- SETTINGS-01 同 run 聚合使用新目录 `settings-01-gate-b-f-v2/summary.json` 保留原 `14/19` summary，并提升到
  `15/19`；剩余 `memory-soul-persistence`、`archived-lifecycle-recovery`、`provider-crud-model-auth`、
  `mcp-lifecycle-recovery` 共 4 项。history/import 热区继续避让；下一刀进入 `mcp-lifecycle-recovery`。旧
  `webview-api` Browser Workspace facade、Electron diagnostic reads 和 extension bridge 不属于本 Settings claim，
  继续作为 PAGE-08 / Browser Runtime 的 current blocker，不能用本证据冒充完成。
- `2026-07-17 18:00 +08:00` 完成 `mcp-lifecycle-recovery`。新增专用真实 Electron fixture，在隔离 userData
  中从 MCP Settings GUI 创建 Context7 配置、修改描述与 Lime 启用状态、冷重启读回、从 GUI 删除并再次冷重启
  确认不存在；三次启动均证明 Electron/preload、`electron-ipc`、`app_server_handle_json_lines`，并命中
  `mcpServer/list|create|update|delete`。结构化 evidence `18/18`，console/page/invoke、legacy/mock 全零，raw 只
  保存生命周期布尔与四个 current method，不保存 server 配置、名称、描述、ID、路径、凭证、prompt、resource 或
  tool output。三张截图人工确认修改态、重启恢复态和最终空态无 loading、遮挡、内部诊断或混语；该 claim 不启动
  live MCP、不调用工具，也不访问配置 URL。summary 为
  `.lime/qc/project-gates/standalone-settings-b-v1-20260717T0954/settings-mcp-lifecycle-recovery/settings-mcp-lifecycle-fixture-summary.json`。
- SETTINGS-01 同 run 聚合使用新目录 `settings-01-gate-b-f-v3/summary.json` 保留旧 summary，并提升到 `16/19`；
  剩余 `memory-soul-persistence`、`archived-lifecycle-recovery`、`provider-crud-model-auth` 共 3 项，
  `surfaceProof.complete=false`。history/import 热区继续避让；下一刀优先盘点并补 `provider-crud-model-auth`，不得
  用既有 Provider migration 或只读模型目录 evidence 冒充 GUI CRUD、模型刷新与鉴权错误恢复。
- `2026-07-17 19:20 +08:00` 完成 `provider-crud-model-auth`。四层盘点确认 Provider Settings 已经由
  `ApiKeyProviderSection -> apiKeyProvider/modelRegistry typed gateway -> app_server_handle_json_lines -> App Server ->
services/model-provider` 形成唯一 current owner，不新增命令或 compat。专用 fixture 使用隔离 userData 与 localhost
  OpenAI-compatible `/v1/models|chat/completions`：从 GUI 创建自定义 Provider，先以错误密钥观察 401 可见状态，再从
  GUI 修正密钥、获取并选择模型、完成连接/聊天 readiness、冷重启读回、从 GUI 删除并再次冷重启确认不存在。
- 首轮失败 evidence 证明模型获取与密钥恢复已经成立，但 localhost fixture 缺少 `POST /v1/chat/completions`，真实
  `modelProvider/testChat` 返回 404；没有通过拉长 timeout 掩盖。补齐 fixture route 后 `v2` 通过；`v3`/`v4` 继续
  暴露截图隐私与 React 投影等待竞态，最终改为 evidence-only 动态值隐藏和目标模型 web-first 可见等待，使用 `v5`
  形成终证据。`v5` Gate B-R `24/24`，命中 `modelProvider/list|catalog/list|create|update|fetchModels|testConnection|
testChat|delete` 与 `modelProviderKey/create`，localhost 观察 1 次 unauthorized、3 次 authorized，console/page/invoke、
  legacy/mock 全零。JSON 不保存 Provider 值、模型 ID、Host、端口、Key、路径、header 或 body；四张截图人工确认只
  保留错误/成功 tone、表单结构、重启恢复结构和最终空态。summary 为
  `.lime/qc/project-gates/standalone-settings-b-v1-20260717T0954/settings-provider-crud-model-auth/settings-provider-crud-fixture-v5-summary.json`。
- SETTINGS-01 同 run 聚合使用新目录 `settings-01-gate-b-f-v4/summary.json`，提升到 `17/19`；剩余
  `memory-soul-persistence` 与 `archived-lifecycle-recovery`，`surfaceProof.complete=false`。history/import 热区继续
  避让；下一刀进入 `memory-soul-persistence`，必须证明 GUI 保存/冷重启恢复与 Runtime prompt marker，不能把现有
  Memory ready、renderer trace 或 external backend 输出误报为 Soul 最终 prompt 注入。

### Wave 1：Preflight 与全量本地门禁

权威入口：

```bash
npm run verify:local:full
```

若 frontend smart suite 中断，使用：

```bash
npm run test:resume
```

全项目重构验收还必须从 aggregate summary 确认以下任务未被 selector 跳过；跳过时才单独补跑：

```bash
npm run test:contracts
npm run test:rust
npm run governance:legacy-report
npm run governance:scripts
npm run governance:file-size
npm run governance:import-boundaries
npm run i18n:check:json
npm run i18n:scan:json
npm run i18n:unused:json
npm run verify:app-version
npm run agent-qc:verify-local-gate
```

退出条件：所有任务 pass；不存在“因为无关所以跳过”的 v2 核心 crate/package；失败历史保留。

### Wave 2：Gate A 全页面投影

先启动开发宿主并确认 bridge：

```bash
LIME_ELECTRON_REMOTE_DEBUGGING_PORT=9223 npm run electron:dev
npm run bridge:health -- --timeout-ms 120000
```

Gate A 按 Page union 覆盖 `agent`、`experts`、`skills`、`plugin`、`plugins`、`plugin-lab`、
`knowledge`、`automation`、`channels`、`resources`、`browser-runtime`、`settings`。

现有可复用入口包括：

```bash
npm run knowledge:product-e2e
npm run smoke:knowledge-gui
npm run smoke:design-canvas
npm run smoke:browser-runtime -- --remote-debugging-port 9222
npm run smoke:at-command-registry
npm run smoke:automation-current
```

每个页面至少断言：加载完成、主操作可见、空/错/加载/成功状态、导航可恢复、无 raw key、
无 console error。Gate A 统一跑 desktop/compact/narrow 三视口，关键入口跑五语言。

退出条件：Page union 和 SettingsTabs 均有记录；Chrome/DevBridge 证据明确标为 Gate A。

### Wave 3：Gate B 基础设施与 Agent P0

先跑宿主和 current fixture 聚合门槛：

```bash
npm run verify:gui-smoke
npm run smoke:app-server-stdio
npm run smoke:app-server-sidecar-lifecycle
npm run smoke:agent-runtime-current-fixture
```

然后逐场景跑真实 Electron：

```bash
npm run smoke:claw-chat-current-fixture -- --scenario complete
npm run smoke:claw-chat-current-fixture -- --scenario cancel-then-continue
npm run smoke:claw-chat-current-fixture -- --scenario inputbar-pending-steer-multi-queue
npm run smoke:claw-chat-current-fixture -- --scenario inputbar-pending-steer-pop-front-resume
npm run smoke:claw-chat-current-fixture -- --scenario live-tail-commit
npm run smoke:claw-chat-current-fixture -- --scenario terminal-failed-after-answer
npm run smoke:claw-chat-current-fixture -- --scenario terminal-canceled-after-answer
npm run smoke:claw-chat-current-fixture -- --scenario terminal-stale-guard
npm run smoke:claw-chat-current-fixture -- --scenario approval-request-resume
npm run smoke:claw-chat-current-fixture -- --scenario approval-request-decline
npm run smoke:claw-chat-current-fixture -- --scenario approval-request-cancel
npm run smoke:claw-chat-current-fixture -- --scenario approval-request-full-access
```

历史与恢复：

```bash
npm run smoke:agent-session-history-electron-fixture
npm run smoke:codex-import-continuation-electron-fixture
npm run smoke:codex-import-click-through-electron-fixture
npm run smoke:code-artifact-workbench-electron-fixture -- --scenario gui-coding-input
```

History/Thread 额外断言对齐当前 Codex：分页读取与 resume 不重复、不丢 Item，冷重启后 cursor/checkpoint
稳定；paginated source fork 必须显式 fail closed，不能静默退回全量 materialization。Import 场景继续由
相邻 owner 的 current fixture 承接，本 Gate lane 不另建 imported history 旁路。

Multi-Agent、MCP 与 Skills：

```bash
npm run smoke:agent-control-cold-restart-gate-b
npm run smoke:agent-runtime-tool-execution:managed -- --batch mcp-deferred-tool-search-gate-b
npm run smoke:mcp-config-electron-fixture
npm run smoke:mcp-elicitation-gate-b
npm run smoke:claw-chat-current-fixture -- --scenario skills-runtime
npm run smoke:claw-chat-current-fixture -- --scenario expert-plaza-skills-runtime
```

Multi-Agent 额外断言对齐当前 Codex：`fork_turns=none/all/last-N` 覆盖 trigger-turn inter-agent
message、rollback marker 和 startup prefix；queue-only `send_message` 到达 final answer 后留给下一 Turn，
`followup_task` 才触发 idle agent 新 Turn；`list_agents` 只返回 identity/status，不泄漏 task/message 正文。

退出条件：Agent P0 正常、失败、取消、排队、审批、恢复、Multi-Agent、MCP、Skills 全绿，
且全部满足第 6 节硬断言。

### Wave 4：Gate B 工作台与产品页面

工作台场景：

```bash
npm run smoke:claw-chat-current-fixture -- --scenario reasoning-first-visible
npm run smoke:claw-chat-current-fixture -- --scenario right-surface-visual-matrix
npm run smoke:claw-chat-current-fixture -- --scenario media-reference
npm run smoke:claw-chat-current-fixture -- --scenario image-command
npm run smoke:claw-chat-current-fixture -- --scenario electron-resize-reflow
npm run smoke:claw-chat-current-fixture -- --scenario content-factory-article-workspace
npm run smoke:claw-chat-current-fixture -- --scenario content-factory-inline-image-article-workspace
npm run smoke:content-factory-current-turn:host-generation
```

产品页面按矩阵 PAGE-01 至 PAGE-08 和 SETTINGS-01 执行。没有真实 Electron 入口的页面先补
最小专用场景，要求从侧栏或正式导航进入，禁止用 `page.evaluate` 直接注入业务完成态。

Settings 参数化场景至少覆盖：

- Providers：增改删、model selection、错误鉴权恢复；
- MCP Server：增改删、启动失败、重启恢复；
- Memory：AI 个性、历史恢复和清理；
- Media Services：图片/音频/视频 provider readiness；
- Web Search：启停和 route 生效；
- Environment / Execution Policy：允许、拒绝和错误定位；
- Chrome Relay：未安装、连接、断开；
- Archived Conversations：归档、反归档、重启后可见；
- Appearance / About：主题持久化、版本事实源。

退出条件：所有 current 可见 Page 的主任务至少有一个 Gate B；P1 缺口为零。

### Wave 5：故障注入与恢复

所有故障都在隔离目录和 fixture 服务中执行：

| 故障                                       | 必须观察的结果                                            |
| ------------------------------------------ | --------------------------------------------------------- |
| App Server sidecar 缺失/退出/启动超时      | 启动或页面显式失败；无 mock fallback；可重试恢复          |
| JSONL malformed/unknown method             | 当前 request fail closed；其他 session 不串线             |
| provider 401/429/5xx/timeout               | Turn 明确 failed 或可重试；输入框恢复；不伪造 completed   |
| provider 流中断/重复 terminal/late delta   | 单 terminal；late event 丢弃；read model 可恢复           |
| MCP optional/required server failure       | optional 隔离，required fail closed，健康 server 不受影响 |
| MCP elicitation cancel/disconnect          | GUI 关闭、waiter 清理、resolved 顺序正确                  |
| tool permission denied/sandbox reject      | Approval/Tool terminal 一致；无实际副作用                 |
| disk/path permission/sidecar write failure | 用户可见失败；无半写 Item 或错误引用                      |
| Electron 冷重启/renderer reload            | durable identity 不变；pending 状态按 owner 恢复          |
| 双 session 并发/同 session queue           | event、tool、mailbox、artifact 不串线                     |
| paginated history cursor/checkpoint/fork   | read/resume 无重复丢失；unsupported fork 显式 fail closed |
| queue-only mail 到达 final answer 边界     | 不重启已完成 Turn；消息在下一 Turn 可见                   |
| Chrome target detach                       | Browser Runtime 显式 disconnected，可重新 attach          |
| plugin signature/package invalid           | 拒绝安装或激活；旧已安装状态不被破坏                      |
| updater feed 不可用/版本回退               | 保持当前版本；显示可恢复错误；不进入半更新状态            |

退出条件：所有 P0 故障有 Gate B 负向证据；P1 故障有自动场景或责任人签字的实机记录。

### Wave 6：Gate B-L live provider

live 场景必须显式授权并使用测试账号/预算，不纳入默认本地命令：

- 文本：至少覆盖 OpenAI Responses、OpenAI-compatible Chat、Anthropic 三类 current protocol；
- 多模态：图片输入和 reference-only lowering；
- 媒体：`@配图` live，音频/视频仅在对应 provider 已进入 current 支持时执行；
- Skills：专家 skill search/read/invoke 的真实 provider Turn；
- MCP：真实 provider 发起工具选择，但 MCP server 仍可使用隔离 fixture；
- 错误：失效 key、限流和超时使用专用测试配置，不污染真实账号。

已有入口：

```bash
npm run smoke:expert-skills-live-runner -- --allow-live-provider --execute-live-runtime
npm run smoke:claw-image-live -- --allow-live-provider
```

退出条件：每个 current provider protocol 至少一条成功和一条可控失败；证据只保存 marker、usage
摘要和公开错误分类，不保存 prompt、响应正文或凭证。

### Wave 7：Gate B-P 打包、安装和更新

macOS 本地/CI：

```bash
npm run electron:package:dir
npm run electron:verify:package
npm run electron:make:zip-local-feed -- --arch arm64
```

release workflow 必须继续验证 Forge maker、签名、公证、Windows Squirrel 和旧 builder/updater
回流守卫。Windows x64 需要实机/VM 执行 install -> first launch -> main P0 -> update -> relaunch ->
uninstall，检查用户数据保留/清理策略。

退出条件：开发目录、package dir 和安装后 app 三者的 App Server/asset 路径均来自 current
packaging 事实源；没有开发 HTTP bridge 或本地源码路径依赖。

### Wave 8：最终回放与架构确认

1. 对所有修复过的 surface 跑定向 Gate A/B。
2. 再跑全部 P0 Gate B 和 `npm run verify:local:full`。
3. 重算候选 digest，确认执行期间源码未漂移。
4. 在真实 PR event/body/base 上运行：

```bash
npm run governance:architecture-confirmation
```

5. 写入不可变项目 gate evidence，中央计划只引用最新摘要，不覆盖历史失败。

## 11. 五语言、视口和可访问性

### 11.1 五语言

P0 用户可见状态必须逐语言验证：`zh-CN`、`zh-TW`、`en-US`、`ja-JP`、`ko-KR`。
至少覆盖启动、发送、running、queued、approval pending、completed、failed、interrupted、
recovery、provider error、MCP error 和 update error。

P1 页面可以采用“每页默认语言 + 每个 locale 一次全导航 sweep”，但任何新增 key 必须五语言
齐全。协议 enum、schema、trace 和 evidence facts 不本地化。

### 11.2 视口

统一三档：

- desktop：`1440x1000`
- compact：`1024x768`
- narrow：`760x900`，用于桌面窄窗，不冒充移动端产品

验证侧栏、输入框、modal、timeline、right surface、workbench、设置导航不重叠、不被裁切，
动态内容不引发工具栏或固定控件跳动。

### 11.3 可访问性

- 主流程全键盘可完成；
- focus 在 modal 打开/关闭后回到正确触发点；
- icon button 有 aria-label/tooltip；
- form error 与控件关联；
- axe 严重级别 serious/critical 为零；
- reduced motion、高对比度和 200% zoom 做 P1 抽样。

## 12. 性能与稳定性阈值

首轮先记录基线，不拍脑袋设绝对值。用同一机器连续运行 5 次，去掉一次 warm-up，记录
median/p95：

- Electron process start -> startup overlay 消失；
- startup overlay 消失 -> sidebar/workspace ready；
- send click -> turn.accepted；
- turn.accepted -> first visible assistant/reasoning item；
- terminal event -> input composer ready；
- history click -> canonical items visible；
- 1000 Item paginated Thread 的滚动、分页读取、切换和恢复；
- 两个并行 session 的 CPU、RSS、event lag；
- Electron/App Server 冷重启后的恢复时间。

签字阈值：相对 S7ag 或本计划首次稳定基线回退超过 20% 时阻塞；绝对 20% 规则只作预警，
必须同时结合 p95、错误率和用户可见卡顿判断。长流、取消和页面切换后不得残留持续增长的
timer/listener/session runtime。

## 13. 证据结构

每个场景写入：

```yaml
scenario_id: AGENT-01-complete
priority: P0
proof_level: Gate B-R
candidate_run_id: <run-id>
platform: <os/arch>
entry_command: <command>
backend_mode: runtime
required_methods:
  - agentSession/start
  - agentSession/turn/start
  - agentSession/read
bridge:
  electron: true
  preload_invoke: true
  transport: electron-ipc
  command: app_server_handle_json_lines
assertions:
  total: <n>
  passed: <n>
console_errors: 0
page_errors: 0
invoke_errors: 0
legacy_command_hits: 0
mock_fallback_hits: 0
screenshots: <relative paths>
trace_summary: <relative path>
read_model_summary: <relative path>
result: pass | fail | blocked
failure_class: null | product | harness | environment | flaky | live-provider
owner: <domain owner>
next_action: <exact action>
```

上面的 YAML 是人类可读摘要。任何要计入固定 `34` 分母的机器 evidence JSON 还必须包含：

```json
{
  "schemaVersion": 1,
  "candidateRunId": "<candidate run-id>",
  "surfaceProof": {
    "surfaceId": "AGENT-01",
    "proof": "gate-b-r",
    "complete": true
  },
  "result": "pass",
  "assertions": { "total": 12, "passed": 12, "failed": [] }
}
```

`complete=true` 是该 evidence 已覆盖对应 surface/proof 全部 claim 的显式签字，不是“脚本退出 0”。
部分 route、单场景或 claim boundary 尚未覆盖完整 surface 时必须写 `complete=false`，且不计入分子。
失败/blocked evidence 还必须写非空 `failureClass` 和 `nextAction`。每个 Wave 用：

```bash
npm run agent-qc:project-gate-coverage -- \
  --candidate .lime/qc/project-gates/<run-id>/candidate.json \
  --progress-only
```

最终签字去掉 `--progress-only`；不足 `34/34` 必须非零退出。

截图和 JSON 只放 `.lime/qc/project-gates/<run-id>`。仓库中的 Markdown evidence 记录摘要、
命令、assertion count、脱敏 marker 和相对证据路径，不提交大二进制、token 或用户正文。

## 14. 失败归因与重跑规则

1. 首次失败先原样保留 evidence，不立即覆盖。
2. 同候选、同环境、无源码变化重跑一次，用于区分确定性失败和疑似 flaky。
3. flaky 不能直接忽略，必须有 owner、复现概率和退出条件。
4. product 修复后生成新 `run-id`，跑定向 Gate、所有依赖 P0 Gate 和最终 aggregate。
5. harness 修复只允许改变观察/驱动方式，不得降低产品断言；仍生成新 harness digest。
6. live-provider 网络故障可以分类为环境阻塞，但 B-F/B-R 必须继续可复验。
7. 任何 mock fallback、identity 串线或用户数据污染不允许按 flaky 处理。

## 15. 并行策略

候选冻结后可以并行，但只并行无共享状态的 lane：

| Lane | 内容                                 | 并行约束                                                   |
| ---- | ------------------------------------ | ---------------------------------------------------------- |
| L0   | Preflight/Rust/contract              | 与 GUI 分开 target/cache；避免抢占导致 1 秒 I/O 测试假失败 |
| L1   | Gate A Chrome                        | 独立 browser profile、Vite/DevBridge 端口                  |
| L2   | Agent Electron Gate B                | 同一时刻默认单实例，场景串行复用构建产物                   |
| L3   | Plugin/Knowledge/Automation Electron | 每 lane 独立 userData/appData/端口                         |
| L4   | package/platform                     | 独立机器或 VM，不与开发 Electron 共用目录                  |

初始轮不建议同时跑多个重 Rust test 和多个 Electron fixture。S7ag 已出现并发资源竞争导致的
1 秒 I/O 假失败，计划默认“编译可并行，产品 Gate B 串行或最多两个完全隔离实例”。

## 16. 预计工期

在没有 P0 产品缺陷的前提下：

| 阶段                                             | 预计                          |
| ------------------------------------------------ | ----------------------------- |
| harness 缺口和候选冻结                           | 1.5 至 2.5 人日               |
| Preflight + Gate A                               | 1 至 2 人日                   |
| Agent/MCP/Skills 核心 Gate B                     | 2 至 3 人日                   |
| 产品页面与 Settings Gate B                       | 3 至 5 人日                   |
| 故障注入、live provider、packaged/Windows        | 3 至 5 人日                   |
| 修复回放、最终摘要、PR architecture confirmation | 1 至 2 人日，不含产品缺陷修复 |

纯验收约 `8.5 至 14.5` 人日。若发现 identity、持久化、platform 或 provider lowering 缺陷，
修复工期独立计算；不通过压缩场景或降低 proof level 追回时间。

## 17. 最终退出条件

只有同时满足以下条件，才能把本计划标记为 completed：

1. 候选开始/结束 digest 一致。
2. `verify:local:full`、contract、Rust、governance、i18n、version 全绿。
3. 所有 P0/P1 current surface 均有 Gate A 和匹配风险的 Gate B。
4. macOS 主环境 P0/P1 全绿；Windows P0/P1 宿主和 package 场景全绿。
5. P0 failure injection 全绿，P1 无未归属 blocker。
6. console/page/invoke error 为零；legacy/mock fallback 命中为零。
7. Thread/Turn/Item、Tool、SubAgent、MCP、artifact identity 在 live/cold/restart 中稳定；分页历史和
   Multi-Agent mail/fork 边界与 candidate 记录的 Codex commit 一致。
8. 五语言关键状态、三视口和 serious/critical a11y 问题为零。
9. 每个 current provider protocol 有 B-R；需要发布的 provider 有 B-L。
10. package dir、macOS updater 和 Windows Squirrel 有真实运行证据。
11. 真实 PR 上 architecture confirmation 通过。
12. evidence 中没有密钥、完整 prompt、用户正文或真实隐私路径。

完成度按 surface 计算，不按命令数计算：

```text
completion = 已同时满足 Gate A + 所需 Gate B 的 P0/P1 surface / P0/P1 surface 总数
```

固定分母为 `34`。当前完成度：`0/34 = 0%`。Wave 0 的 harness 和 Preflight 基线已形成，但历史 v2 evidence 只能减少
重建成本；新候选未冻结前，任何 surface 都不计入当前 Gate A/B 完成度。

## 18. 治理分类与下一刀

- `current`：本计划、现有 App Server JSON-RPC/Electron Gate B fixture、v2 不可变 evidence。
- `compat`：无新增；Gate 不能为了旧测试引入 compat wrapper。
- `deprecated`：Chrome + DevBridge current smoke 若只证明 Renderer/HTTP bridge，继续作为 Gate A，
  不冒充 Gate B。
- `dead`：旧 Team synthetic fixture、legacy command、production mock fallback 和旧 runtime evidence
  不进入当前验收目录。

下一刀：由相邻 Codex import owner 完成其 current lowering/read-model 收口；本 Gate 协调 lane
不进入该写集。收到退出信号后执行 `fork_turns` 的 contracts、Rust related 与 current Agent
runtime fixture；若全绿，连续两次确认完整产品 digest 稳定并生成全新候选，从 Wave 1 第 1 批
完整重跑。Wave 1 未全绿前不进入 Gate A，更不启动 live-provider 或 packaged lane，避免把基础
回归、宿主问题和外部环境问题混在同一失败里。

## 19. 2026-07-17 SETTINGS-01 Memory/Soul 收口

- 本轮窄写集只覆盖 Memory Settings UI 稳定锚点、Memory/Soul 真实 Electron fixture 与对应 evidence
  contract；继续避让 Codex import/history 热区，不修改相邻进程的 Agent message、conversation import 或 history
  owner 文件。
- `settings-memory-soul-fixture-v4-summary.json` 为本轮唯一聚合源。真实 Electron 两次冷启动完成 Memory 全局
  开启、`cheeky_sassy_executor` profile、`direct` template 保存与重启恢复；断言 `20/20`，包括
  `get_config/save_config`、`soulStylePack/list`、`memoryEnabled`、`soulEnabled`、隔离 userData、runtime
  marker 完整、console/page/invoke/legacy/mock 全零。独立 runtime fixture 只保存 marker booleans，不保存 Soul
  文本、完整 prompt、provider payload、用户正文、路径或凭证；GUI 与 runtime 的进程/app-data 边界已写入 claim。
- 新增 `settings-memory-toggle` 与 Soul 状态 `data-state` 稳定锚点；fixture 对旧 dist bundle 保留同页唯一
  `role=switch` 和可见状态文本兜底，但 current 源码路径优先使用新锚点。保存/恢复截图已人工审计，恢复态明确显示
  `已启用`、`已形成` 和 `直接审阅者`。
- SETTINGS-01 Gate B-F 使用新目录 `settings-01-gate-b-f-v5/summary.json`，从 `17/19` 提升为 `18/19`；当前
  唯一缺口为 `archived-lifecycle-recovery`。该场景仍由相邻 Codex import/history owner 处理，本 lane 不伪造
  `pageErrorCount=0`，也不把已有 history evidence 冒充完成。

## 20. 2026-07-17 SETTINGS-01 archived lifecycle 收口

- 保持 Codex import、local-history import、MessageList、Sidebar history、conversation_import Rust owner 只读；只在
  clean `session-history-fixture-smoke.mjs` 增加独立 `pageerror` observer 和结构化 request 的 `command/transport`
  marker，并新增 Settings 专用脱敏 adapter。
- `settings-archived-lifecycle-fixture-v3-summary.json` 经过真实 Electron owner fixture，归档前/恢复后截图人工审计：
  归档列表为 `1` 条并显示恢复按钮，恢复后为空态为 `0` 条。adapter 断言 `18/18`，包括 Electron/preload、
  `electron-ipc -> app_server_handle_json_lines`、`agentSession/list|read|update`、侧栏归档、Settings 恢复、归档
  冷读回、恢复冷读回、console/page/invoke/legacy/mock 全零。
- `v1` 失败为 adapter 误读被截断原始 trace，`v2` 失败为侧栏点击在并行 Electron 负载下发生交互漂移；两份失败
  evidence 保留，`v3` 才是唯一聚合源。没有通过放宽等待、伪造 page error 或复用旧 summary 绕过失败。
- SETTINGS-01 Gate B-F 已使用新目录 `settings-01-gate-b-f-v6/summary.json` 聚合 `19/19`，
  `surfaceProof.complete=true`。该结果证明 Settings current owner 的 Gate B-F 已闭环，不等同于整个 34-surface
  项目门禁、Windows/package、live provider 或 failure-injection 全部完成。

## 21. 2026-07-18 候选冻结前置收尾

- 前端 smart suite 在同一当前工作树上先跑第 `1-68` 批，第 68 批暴露 `EmptyState` en-US 回归仍断言旧通用
  hero 文案；根因是首页 hero 文案事实源已迁到当前 skin 的五语言 `skinContent`。测试改为从
  `DEFAULT_LIME_COLOR_SCHEME_ID + en-US` 读取 current skin copy，原失败用例 `1/1`、`EmptyState` 全文件
  `62/62`、ESLint、Prettier 与 diff check 通过；随后从第 68 批续跑至 `111/111`，状态在
  `2026-07-18 01:25 +08:00` 完整通过。
- 候选前置合同与治理通过：`test:contracts` 的 App Server client `291 checks`、command/harness/modality、
  Electron release workflow、cleanup contract 与 docs boundary 全绿；`governance:legacy-report` 为零引用候选
  `0`、分类漂移 `0`、边界违规 `0`；`governance:scripts` 无 root/一级目录漂移。
- App Server read-model 定向测试 `51/51` 通过。package 级 `test:rust:related` 实际执行 App Server lib
  `1183` 个测试，结果为 `1177 passed / 6 failed`：3 个 conversation import/性能场景、2 个 plugin worker
  1 秒超时、1 个 provider WebSocket fixture 超时。conversation import 正由相邻 owner 修改，本 Gate lane 未夹写；
  这 6 个失败必须在新 candidate 前由对应 owner 复验，不得用 `51/51` 掩盖。
- `smoke:agent-runtime-current-fixture` 完整通过，覆盖 history/cache、Claw 首页、Coding Workbench、图片任务、
  stop/continue、approval、Inputbar queue/restore、Plan history、Skills、MCP/media、Expert Skills 与 Article Editor；
  `liveProviderUsed=false`。首次冷构建共转换 `9278` 个模块，renderer build 为 `17m52s`，作为 CROSS-04
  后续基线风险保留，不把 fixture 冒充 live provider 或全项目 Gate B。
- 为继续聚合门禁而第二次启动的 `verify:local` 会再次从第 1 批无差别执行 111 批；按本轮收尾决定在第 3 批
  主动中止，退出码 `130`。因此本轮只声明其版本、i18n、lint、typecheck 与前端阶段分别已有通过证据，
  不声明聚合 `verify:local` 完整退出；`verify:gui-smoke` 也未单独执行。
- import tracker 复核仍为 `active / S4 / 99%`，只剩责任开发者架构确认及后续 Windows runner 证据；不满足
  `ready/ready-for-gate/completed/closed`，所以不生成 candidate、不进入 Wave 1。项目签字完成度保持
  `0/34 = 0%`；下一次只需从 owner 退出、6 个 Rust 失败复验、聚合 local/GUI smoke 和 candidate digest
  稳定检查继续，不再重复本轮已经完成的局部迁移。
