# 已安排任务实现计划

状态：`implementation-complete / route-selection-repair-complete / local-validation-complete / platform-evidence-pending`

更新时间：2026-08-28

## 主目标

完成 `internal/roadmap/task/scheduled-tasks/` 定义的已安排任务产品，收敛现有 Automation UI、协议、调度、运行和历史到唯一 current 主链。

## 当前阶段与下一刀

- 2026-08-17 实现收口：9 个 `scheduledTask/*` method、唯一 `automation_jobs` 内部存储映射、RuntimeCore/Thread/Turn/Item/Agent Run、typed notification、软删除、一级 Scheduled Tasks 工作台、五语言资源和真实 Electron Gate B 已建立。`continue_thread` 复用 canonical Thread identity；`new_thread` 复用 canonical model catalog、route metadata 与 preflight。
- 公开 Automation 双轨已物理删除：旧 App Server method/schema/client、旧页面和 Settings 工作台、旧 renderer projection/smoke、旧 i18n namespace consumer 均归类为 `dead / deleted / forbidden-to-restore`。最后残留的 `Page = "automation"`、`AutomationPageParams`、侧栏 `id = "automation"` 与 `SettingsTabs.Automation` 已迁为唯一 `scheduled-tasks` 一级路由，Settings 不再重复挂载 Scheduled Tasks 页面。
- 2026-08-18 本地收尾已完成：Rust related、Scheduled Tasks Gate B、Agent current fixture、GUI smoke、contracts、governance、`verify:local`、fmt 与 diff check 全部通过；磁盘空间不再构成阻塞。
- 2026-08-28 输入框 Provider/模型路由修复已完成：Scheduled Tasks 创建、服务技能自动化、历史任务显式迁移和 RuntimeCore 执行统一使用 Composer 当前选择的 opaque route；专项 Gate B 已证明实际 Provider 请求、Authorization、canonical Thread/Turn/Run 与 GUI 闭环一致。
- 当前下一刀只剩平台证据：真实 Windows Notification Center、Windows Gate B，以及 macOS/Windows 真实 sleep-resume。不得以当前 macOS arm64 fixture 替代这些结论。

完成度：实现 `100%`；本地门禁 `100%`；平台证据 `pending`。

## 2026-08-28 输入框模型选择与 Provider transport 修复

- 主目标：已安排任务必须使用创建入口当前选中的完整 `provider + model` 路由；不得在 `modelId` 缺失时因 Provider 排序变化静默切到另一模型。
- 根因：侧栏进入 Scheduled Tasks 时没有透传当前项目；手动创建与服务技能创建都把 `execution.modelId` 留空。后台 `new_thread` 只能从模型目录选择第一项，实际任务因此从用户选中的 `agnes-2.5-flash` 漂移到 `lime-hub / gpt-5.2-pro`。
- 路由修复：复用现有 `route:<base64-provider>.<base64-model>` opaque selector，在创建任务时冻结当前项目输入框选中的 provider/model；编辑器只读展示同一 `provider / model`，不再提供第二个手工模型入口。历史 `modelId=null` 任务在用户点击“立即运行”时先用该任务项目的当前选择重存为显式 route，再启动；进入编辑器也预填同一路由。后台执行仍未迁移的历史任务时 fail closed，不再按 catalog 第一项猜测。
- Provider 基建：Provider base URL 统一按结构化 URL 处理；endpoint 只修改 path，保留非租户 query、移除 fragment，并把合法 `lime_tenant_id` 投影为 `X-Lime-Tenant-ID`。HTTP 与 Responses WebSocket 复用同一规则。
- 诊断修复：参考 Codex `http-client` 的 `reqwest::Error::without_url()` 边界，持久化错误保留 transport source chain 与最终尝试次数，同时不暴露 URL path/query/credential。
- 窄写集：`packages/app-server-client/src/model-route.ts`、`src/components/{AppSidebar.tsx,scheduled-tasks/**}`、`src/components/agent/chat/workspace/{useAgentChatWorkspaceSetupRuntime.ts,useWorkspaceServiceSkillEntryActions*,workspaceServiceSkillEntryActionsViewModel*}`、五语言 `scheduledTasks.json`、`lime-rs/crates/app-server/{src/runtime/model_providers/selection.rs,src/scheduled_task_worker/tests.rs,tests/scheduled_tasks_jsonrpc.rs}`、`lime-rs/crates/model-provider/src/{lib.rs,provider_url.rs,current_client.rs,current_client/transport.rs}`、复用 tenant 解析的 `agent` / `services` current consumers、Scheduled Tasks Electron fixture 与本计划。
- 避让：未修改当前脏热区 `tool-runtime/**`、`agent/current_provider_turn/tool_executor.rs`、`app-server/execution_process/tests.rs`、`agentRuntime` i18n、无关 Electron/harness/release 脚本；不修改用户数据库，不提交、不推送。
- 预算标签：`budget:normal`；风险等级：`P1`。
- Current 主链：`Composer model selection -> workspace preference -> ScheduledTaskCreateRequest.execution.modelId -> scheduledTask/create -> automation_jobs -> scheduledTask run -> RuntimeCore route selection -> model-provider HTTP/WebSocket -> canonical Thread/Turn/Run -> GUI`。
- Happy Path：用户在当前项目选中 `agnes-2.5-flash`，从侧栏或服务技能创建任务；持久化 payload 含唯一 route selector；手动/后台运行均请求同一 provider/model；Lime Hub fragment tenant host 能正确请求 endpoint；失败历史展示脱敏根因而不是裸 `connect_error`。
- Evidence Layers：deterministic-smoke=`passed`，runtime/provider capture=`passed`，真实 Electron Gate B=`passed`，release-artifact=`not required`。
- 已跑：前端定向 Vitest、fixture 守卫、`npm run typecheck`、`npm run test:rust:related -- <paths>`、`npm run test:contracts`、`npm run governance:legacy-report`、Scheduled Tasks Electron Gate B、`npm run verify:gui-smoke`、`cargo fmt --check` 与 `git diff --check`。
- 架构影响：非重大架构变更；未新增协议 method、crate、跨层 owner 或第二套状态源，只补全现有 opaque route 与 Provider 网络 owner。

### 完成证据

- `current`：Composer 项目偏好是 Provider/模型选择的唯一 UI 事实源；任务只持久化完整 opaque route，编辑器只读展示该选择。`continue_thread` 继续使用 canonical Thread route。
- `dead / forbidden-to-restore`：`modelId=null -> catalog 第一项`、裸 model 反推 Provider、Scheduled Tasks 自建模型选择器，以及 fixture 依赖 `sortOrder/isDefault`。
- 历史 `null`、裸 model 或损坏 route 不在页面加载时静默写库；用户点击“立即运行”时按任务项目当前选择显式迁移，后台未迁移任务保持 fail closed。
- Provider URL 的唯一 owner 为 `lime-rs/crates/model-provider/src/provider_url.rs`；HTTP 与 Responses WebSocket 统一处理 path/query/fragment tenant，并注入 `X-Lime-Tenant-ID`。
- 参考 Codex `reqwest::Error::without_url()` 后，transport 错误保留脱敏 source chain 与尝试次数，不暴露 URL path/query 或凭证。

```text
npx vitest run <6 个相关前端与 app-server-client 测试文件>
  6 files / 143 tests passed

npx vitest run scripts/electron/scheduled-tasks-fixture-smoke.test.mjs
  4 tests passed

npm run typecheck
  passed

npm run test:rust:related -- <10 个相关 Rust 路径>
  passed（14 个 owner/反向依赖，退出码 0）

npm run test:contracts
  passed（protocol 1029 types no drift；app-server-client 299 checks）

npm run governance:legacy-report
  passed（zero-reference candidates=0 / classification drift=0 / boundary violations=0）

npm run smoke:scheduled-tasks-electron-fixture -- --timeout-ms 180000 --keep-temp
  Gate B passed（provider request=1；Authorization matched；Composer selection 与 persisted route matched；Electron IPC=56；legacy/mock/invoke/console/page errors=0）
  evidence: .lime/qc/gui-evidence/scheduled-tasks-electron-fixture/scheduled-tasks-electron-fixture-summary.json

ELECTRON_E2E_USER_DATA_DIR=/tmp/lime-gui-smoke-userdata.IMJV87 npm run verify:gui-smoke
  passed（真实 Electron startup/reload、preload/IPC、App Server sidecar、Claw workbench、memory settings；21/21 assertions）
  evidence: .lime/qc/project-gates/standalone-shell-01-20260828045400-27737/shell-01-electron-smoke/summary.json

cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check
git diff --check
  passed
```

## 2026-08-17 Terminal Notification 与软删除完成

- 主链：`scheduledTask/* -> App Server typed notification -> Renderer global bridge -> Electron Desktop Host notification`；App Server 不调用 Electron API，Renderer 不维护业务 timer。
- 窄写集：`lime-rs/crates/app-server-protocol/**`、`lime-rs/crates/app-server/src/{processor/automation.rs,scheduled_task_worker.rs,main.rs,lib.rs,local_data_source/automation/**}`、`lime-rs/crates/core/src/database/{schema.rs,dao/automation_job.rs}`、`lime-rs/crates/scheduler/src/claim{,/recovery}.rs`、`packages/app-server-client/**`、`src/lib/api/scheduledTasks.ts`、`src/components/ScheduledTaskNotificationBridge*`、`src/components/scheduled-tasks/**`、`src/App.tsx`、五语种 `scheduledTasks` 文案、相关 contract/fixture 测试和本计划文件。
- 避让：不修改 `runtime.rs`、`runtime/skills.rs`、`skills_jsonrpc.rs`、`tool-runtime/execution_process.rs` 与其他未知脏热区；前序 Scheduled Tasks 热区只做增量修改，不覆盖已完成的 route/Gate B 修复。
- notification 退出条件：新增 typed `scheduledTask/changed` 与 `scheduledTask/run/updated`；create/update/delete/enabled-set 发布失效通知，manual/due/catch-up/recovery/missed terminal Run 发布终态投影；`all_runs / failures / none` 在 Renderer 统一决策；Desktop Host 返回 `unsupported/failed` 时显示可见错误，不伪造成功。
- 删除退出条件：`scheduledTask/delete` 写入 tombstone、禁用并清除未来调度，但保留 Agent Run 与 canonical Thread/Turn；运行中的 Run 不自动取消，完成写回不得复活 tombstone；GUI 在运行中删除前明确说明影响，当前 runtime 未提供 Scheduled Task cancel，因此不展示“同时取消运行”。
- 验证：protocol/schema/generated client、DAO migration/软删除、public JSON-RPC notification、worker terminal notification、Renderer policy/Host failure、组件确认文案；再运行 related、`test:contracts`、Scheduled Tasks Gate B、Agent current fixture、GUI smoke、`governance:legacy-report`、`verify:local`、fmt 与 diff check。
- 明确剩余证据：Windows Notification Center、Windows Gate B 与 macOS/Windows 真实 sleep-resume 仍需对应平台 runner；macOS 证据不能替代 Windows 结论。

### 完成证据

- typed validator 拒绝额外字段、空 identity、非法 `status/policy`，并支持按 task/run 去重订阅；协议 schema、Rust 类型和 TypeScript client 已同步生成。
- App Server 在 create/update/delete/enabled-set、manual/due/catch-up/recovery/missed terminal Run 时发布 `scheduledTask/changed` 或 `scheduledTask/run/updated`；canonical terminal event 是终态事实源，重复终态写回保持幂等。
- Renderer 全局 `ScheduledTaskNotificationBridge` 统一执行 `all_runs / failures / none` 策略；Host `unsupported/failed` 显示可见 toast，不伪造成功。Scheduled Tasks 页面收到通知后合并刷新列表、详情和历史，并在运行中删除前明确“不会取消当前运行、完成后仍保留历史”。
- 软删除写入 tombstone、禁用并清除未来调度；active Run 不自动取消，完成写回不会复活 tombstone；active run 查询使用专用 DAO，不再依赖 `usize::MAX` 分页。

本轮定向验证：

```text
npm run detect-translations -- --format json
  5 locales / 14 namespaces / 100% coverage / no issues

npm run governance:legacy-report
  zero-reference candidates: 0 / classification drift: 0 / boundary violations: 0

npm run test:contracts
  passed（protocol schema/validator、generated client、app-server-client 299 checks）

npx vitest run "src/components/ScheduledTaskNotificationBridge.test.tsx" \
  "src/components/scheduled-tasks/ScheduledTasksPage.test.tsx" \
  "src/lib/api/scheduledTasks.test.ts" \
  "packages/app-server-client/tests/direct-notifications.test.mjs"
  4 files / 31 tests passed（validator、通知策略、Host failure、实时刷新、软删除与运行中删除确认）

npm run smoke:scheduled-tasks-electron-fixture -- --timeout-ms 180000
  Gate B passed；真实 Electron/preload/IPC/App Server/RuntimeCore/provider 链路，legacy/mock/invoke error 均为 0

npm run smoke:agent-runtime-current-fixture
  passed

npm run verify:gui-smoke
  passed（真实 Electron + App Server）

npm run test:rust:changed -- --changed=origin/main
  passed（changed-scope 受影响 owner 与反向依赖全部通过）

npm run test:resume
  no pending batches

npm run verify:local
  passed（Vitest smart 121/121 batches、changed-scope Rust owner 及反向依赖、真实 Electron GUI smoke）

cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check
git diff --check
  passed
```

## 2026-08-17 测试迁移与最终本地门禁

本轮先将残留的旧 Automation 测试 fixture 收敛到 Scheduled Task current 合约；随后在公开 GUI 路由收口中删除了剩余的 Automation 页面/Settings 入口：

- `WorkspaceRegisteredSkillsPanel.runtime.test.tsx` 改用 `scheduledTasksApi.listDetailed/setEnabled` 和 typed `ScheduledTask`，移除列表加载制造 `automation_job_projection/background_teammate` 的旧断言。
- `automationLinkStorage.test.ts` 改用 typed `ScheduledTask`、`execution.requestMetadata`、`nextRunAt/lastRunSummary` 和 current 时间字段。
- `useWorkspaceServiceSkillEntryActions.test.tsx` 改用 `scheduledTasksApi.create`，断言 daily schedule、`execution.threadMode/sourceThreadId/projectId` 和 `requestMetadata`，移除旧 `agent.changed` projection 断言；覆盖已有 contentId 复用和缺失 session 先物化 Thread 两条路径。

本轮验证证据：

```text
npx vitest run src/components/agent/chat/workspace/useWorkspaceServiceSkillEntryActions.test.tsx
  14 passed

npm test -- --resume
  从 113/121 续跑至 121/121，全部通过

npm run verify:local
  退出码 0；Vitest smart 121/121、i18n/lint/typecheck、test:contracts、changed-scope Rust、真实 Electron GUI smoke 全部通过

npm run test:contracts
  protocol 无漂移；app-server-client 299 checks；command/harness/modality/scripts/docs 门禁通过

npm run governance:legacy-report
  zero-reference candidates=0 / classification drift=0 / boundary violations=0

npm run verify:gui-smoke
  真实 Electron/App Server sidecar 初始化、reload 后 Claw workbench、memory settings 和 smoke evidence 均通过

cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check
git diff --check
  通过
```

`WorkspaceRegisteredSkillsPanel.tsx` 已拆为 Panel、Card 与 RegistrationDetails 三个 owner，单文件均低于 1000 行。当前仍需显式保留的退出条件只有 Windows Notification Center、Windows Gate B 与真实 macOS/Windows sleep-resume，且必须由对应平台 runner 补证。

## 2026-08-17 Scheduled Tasks Gate B 与失败历史修复

- Gate B 首次失败的实际根因不是 provider endpoint：保留的真实 Electron rollout 显示 `modelId=null` 继承了隔离环境中排序更靠前的 `lime-hub / gpt-5.2-pro`，请求未到 localhost fixture，最终返回 404。专项 fixture 原先只确认 provider 出现在 `model/list`，没有确认它是默认模型。
- 当时的 fixture 修复：创建 provider 后使用 `sortOrder: -1`，并断言匹配项 `model/list.isDefault === true`。该 `modelId=null` 继承行为已在 2026-08-28 归类为 `dead`；current fixture 必须显式冻结 opaque route，不能再依赖 catalog 排序。
- GUI 修复：`ScheduledTasksPage.runNow` 在 `startRun` 成功或失败后统一重新加载任务详情、任务列表和运行历史。Runtime 已落盘失败 Run 时，启动请求即使返回错误，页面也会立即显示失败状态、错误文案和运行记录，不再停留在“0 次运行 / 还没有运行记录”。
- 回归写集：`scripts/electron/scheduled-tasks-fixture-smoke.mjs`、对应 fixture 单测、`src/components/scheduled-tasks/ScheduledTasksPage.tsx`、对应组件单测、`package.json` smoke 入口。
- 真实证据：`.lime/qc/gui-evidence/scheduled-tasks-electron-fixture/scheduled-tasks-electron-fixture-summary.json` 与同目录 raw/screenshot。

```text
npx vitest run scripts/electron/scheduled-tasks-fixture-smoke.test.mjs \
  src/components/scheduled-tasks/ScheduledTasksPage.test.tsx
  8 passed

npm run smoke:scheduled-tasks-electron-fixture -- --timeout-ms 180000 --keep-temp
  Gate B passed
  real Electron/preload/IPC/App Server RuntimeCore/provider: passed
  provider request count: 1; selectedAsDefault: true; authorization: matched
  canonical Thread/Turn/provider route/final text: matched
  bridge IPC hits: 34; missing current methods: 0
  legacy methods/commands: 0; mock fallback: 0; invoke errors: 0
  console/page errors: 0

npm run test:contracts
  passed

npm run smoke:agent-runtime-current-fixture
  passed（真实 Electron/App Server current fixture，liveProviderUsed=false）

npm run verify:gui-smoke
  passed（真实 Electron + App Server，result=pass）

npm run verify:local
  passed（Vitest smart 120/120 batches、changed-scope Rust、GUI smoke）

git diff --check
  passed
```

本轮 Gate B 证明的主线是：一级导航创建使用输入框已选 Provider/模型的任务 -> `scheduledTask/run/start` -> Runtime provider 实际请求 -> canonical Thread/Turn/read model -> 运行历史打开同一对话；没有依赖 App Server mock 或 renderer fallback。`--keep-temp` 仅用于保留隔离诊断目录，未删除用户数据。

## 2026-08-17 故障修复写集与退出条件

- 窄写集：`lime-rs/crates/app-server/src/automation_execution.rs`、`lime-rs/crates/app-server/src/runtime/model_providers/selection.rs`、`lime-rs/crates/app-server/src/scheduled_task_worker/tests.rs`、`lime-rs/crates/app-server/tests/scheduled_tasks_jsonrpc.rs`、`scripts/electron/scheduled-tasks-fixture-smoke.mjs`、`scripts/electron/scheduled-tasks-fixture-smoke.test.mjs`、`src/components/scheduled-tasks/ScheduledTasksPage.tsx`、`src/components/scheduled-tasks/ScheduledTasksPage.test.tsx`、`package.json`、本计划文件。
- 历史行为（现 `dead`）：`new_thread + modelId=null` 曾继承 `model/list` 第一项；该路径会因 catalog 排序漂移，已禁止恢复。
- `new_thread + explicit modelId`：只接受 opaque route selector 并解析为唯一 provider/model；缺失、裸 model、无匹配或非法 selector 均 fail closed。
- `continue_thread`：继续复用 durable canonical session route，不创建替代 session，不用 Scheduled Task 的空 route 覆盖 source thread。
- 回归：public `scheduledTask/run/start` 使用要求 provider selection 的 route-aware backend 与真实 model catalog fixture，断言 explicit selection、preflight、canonical Thread/Turn/Run lineage；即使 catalog 存在，缺失 route 也必须 fail closed。
- 验证：Rust 定向/related、contracts、Agent current fixture、GUI smoke、`verify:local`、fmt 和 diff check 全部通过；真实 Electron 复走“已安排任务 -> 每日项目简报 -> 更多 -> 立即运行”，确认错误消失、canonical Thread/Turn 已创建且运行历史可见。

### 2026-08-17 故障修复完成

- 该阶段曾允许 `modelId=null` 继承默认模型和裸 stable id/alias 反推 provider；这两条兼容路径已在 2026-08-28 删除。current `new_thread` 只接受 `route:<base64-provider>.<base64-model>` 并要求唯一 provider/model。
- session metadata 写入 `providerSelector/providerName/modelName/serviceTier`，并在真实 Runtime backend 下执行 `preflight_thread_start`；turn runtime options 使用同一路由。`continue_thread` 保留 canonical session identity 和原有 route。
- route-aware public JSON-RPC 回归覆盖显式 opaque selector、preflight 次数、Thread/Turn route lineage，以及缺失 route 在 catalog 存在时仍 fail closed；worker fixture 已补齐真实 chat provider/model/key，避免测试绕过 route 合同。
- 真实链路与质量门禁结果：

```text
cargo test -p app-server --test scheduled_tasks_jsonrpc -- --nocapture
  5 passed

cargo test -p app-server --lib automation_execution -- --nocapture
  8 passed

cargo test -p app-server --lib scheduled_task_worker -- --nocapture
  13 passed

cargo test -p app-server --lib -- --nocapture
  1678 passed

npm run test:rust:related -- lime-rs/crates/app-server/src/automation_execution.rs \
  lime-rs/crates/app-server/src/runtime/model_providers/selection.rs \
  lime-rs/crates/app-server/src/scheduled_task_worker/tests.rs \
  lime-rs/crates/app-server/tests/scheduled_tasks_jsonrpc.rs
  passed

npm run test:contracts
  passed

npm run smoke:agent-runtime-current-fixture
  passed（真实 Electron/Preload/App Server，liveProviderUsed=false）

npm run verify:gui-smoke
  passed（result=pass）

npm run verify:local
  passed

cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check
git diff --check
  passed
```

原始用户路径“已安排任务 -> 每日项目简报 -> 更多 -> 立即运行”已复走，`App Server runtime backend requires provider/model selection` 不再出现；canonical Thread/Turn 创建成功，运行历史可见。默认上游 Deno `ptrcomp_sandbox` 地址仍会 404，但仓库 wrapper 使用本地 v150.4.0 Apple ARM artifact 完成上述 Rust 验证；该证据不替代 Windows 平台验证。

## 2026-08-17 公开 Automation GUI 路由清理

- 一级入口统一为 `Page = "scheduled-tasks"`、侧栏 `id = "scheduled-tasks"`、`ScheduledTasksPageParams` 和 `selectedTaskId`；旧 `AutomationPageParams`、`AutomationWorkspaceTab` 与 `Page = "automation"` 已删除。
- Settings 不再暴露 `SettingsTabs.Automation` 或重复挂载 Scheduled Tasks；Settings 首页快捷入口直接导航到一级 `scheduled-tasks`，五语言 `settings`/`navigation` key 同步改名。
- `legacySurfaceCatalog` 新增 `frontend-retired-automation-page-route` 负向守卫，覆盖 route、sidebar、AppPageContent、Settings layout、i18n key，防止公开 Automation surface 回流。

本轮公开 GUI 收口验证：

```text
npx vitest run <AppPageContent/AppSidebar/sidebarNav/useAppNavigation/Settings layout/Settings home/legacySurfaceCatalog/i18n>
  9 files / 308 tests passed

npm run typecheck
  passed

npm run detect-translations -- --format json
  5 locales / 14 namespaces / 100% coverage / no issues

npm run governance:legacy-report
  zero-reference candidates=0 / classification drift=0 / boundary violations=0

npm run test:contracts
  protocol 967 types no drift / app-server-client 299 checks / command, scripts and docs gates passed

git diff --check
  passed
```

Rust related 首次修复后曾在链接 `app-server` 测试二进制时因本机仅余约 1.7 GB 被阻断；随后 Data volume 恢复到约 68 GB，未删除源码、用户数据、测试证据或缓存，原 related 范围已完整复跑通过。

## 2026-08-18 最终本地收尾

- 26 个 Scheduled Task Rust owner/consumer 路径的 related 回归完整结束，所选 crate 与反向依赖均为 `0 failed`；最后删除了仅供测试调用、生产零引用的 `build_automation_run_start` 旧公共 helper，测试统一进入 `build_scheduled_task_manual_run_start` 或 claimed run current 入口。
- Scheduled Tasks Gate B 首次复跑在创建任务前等待旧 `app-sidebar-nav-automation` selector 超时；根因是 fixture 未随公开一级路由迁移。fixture 已改用 `app-sidebar-nav-scheduled-tasks`，并补 current/retired selector 源码守卫，随后完整 Gate B 通过。
- Gate B 最终证据为真实 Electron/preload/IPC、`app_server_handle_json_lines`、App Server RuntimeCore 与本地 provider fixture；创建、立即运行、运行历史和 canonical 对话均可见，provider request 为 1，missing method、legacy、mock fallback、invoke/console/page error 均为 0。
- 当前 Vitest smart ledger 为 `119/119 passed`，无 pending batch；本轮 selector 与 dead helper 增量修改均已通过最贴 owner 的回归。

最终本地门禁：

```text
CARGO_INCREMENTAL=0 CARGO_PROFILE_TEST_DEBUG=0 npm run test:rust:related -- <26 个 Scheduled Task Rust owner/consumer paths>
  passed（所选 owner 与反向依赖 0 failed）

CARGO_INCREMENTAL=0 CARGO_PROFILE_TEST_DEBUG=0 npm run test:rust:related -- \
  lime-rs/crates/app-server/src/automation_execution.rs
  app-server 1683 passed / 0 failed

npx vitest run scripts/electron/scheduled-tasks-fixture-smoke.test.mjs
  4 passed

npm run smoke:scheduled-tasks-electron-fixture -- --timeout-ms 180000
  Gate B passed（backendMode=runtime；真实 Electron；provider request=1；legacy/mock/error=0）

npm run smoke:agent-runtime-current-fixture
  passed（完整 current fixture 回归；liveProviderUsed=false）

npm run verify:gui-smoke
  passed（真实 Electron + App Server，result=pass）

npm run test:contracts
  passed（967 protocol types；app-server-client 299 checks；scripts/docs gates passed）

npm run governance:legacy-report
  zero-reference candidates=0 / classification drift=0 / boundary violations=0

npm run verify:local
  passed（smart ledger 119/119；无 pending batch）

cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check
git diff --check
  passed
```

## 窄写集

- `internal/roadmap/task/scheduled-tasks/**`
- `internal/exec-plans/scheduled-tasks-implementation.md`
- `.gitignore`
- `lime-rs/crates/app-server-protocol/**`
- `lime-rs/crates/app-server/**` 中 automation/scheduled-task owner
- `lime-rs/crates/scheduler/**`
- `packages/app-server-client/**`
- `src/lib/api/automation.ts`、目标 `src/lib/api/scheduledTasks.ts`
- `src/components/scheduled-tasks/**`
- `src/components/automation/AutomationPage.tsx`
- `src/lib/navigation/sidebarNav.ts`、侧边栏相关测试
- 五语种 navigation / scheduled tasks 文案
- 对话内 automation draft/projection 的目标迁移文件

## 热区与避让

- 开始时工作树只有本任务上一阶段的 roadmap 与 `.gitignore` 改动。
- 不覆盖无关模块，不提交、不推送、不创建分支。
- 任何工作中出现的外部改动按 `parallel-agent-collaboration.md` 重新审计。

## 分类

- `current`：`scheduledTask/* -> App Server -> scheduler -> RuntimeCore -> Thread/Turn/Item -> Agent Run -> GUI`；`automation_jobs` 与 `AutomationJob` DAO 仅是 current 内部存储映射，`TaskSchedule::{Every,Cron,At}` 仅是 scheduler lowering。
- `compat`：Base Setup / Service Skill 的 `automation_job` binding family 仍有真实消费者，只允许边界委托和一次性迁移，不得扩展第二套任务协议。
- `dead / deleted / forbidden-to-restore`：旧 Automation public method/schema/client、页面、Settings tab、renderer projection、smoke 与旧 i18n consumer；`browser_session` 自动任务、已退役应用编排 context、生产 mock fallback、renderer timer。
- 不建立新的长期 compat owner；现有 compat taxonomy 必须继续向 Scheduled Task current API 收敛。

## 阶段

| 阶段 | 状态 | 退出条件 |
|---|---|---|
| ST-0 合同与审计 | `complete` | 需求、owner、迁移和验证合同已落盘，现有主链已盘点 |
| ST-1/2 协议与领域 | `complete` | 9 个 method、唯一表映射、协议 enum/schema/generated client 已落盘；旧 public method 删除、旧 consumer 清理和 GUI route 收口完成 |
| ST-3 运行闭环 | `complete / platform-evidence-pending` | manual run 与在线 Runtime backend 的 due run 均复用 RuntimeCore/Thread/Turn/Agent Run；原子 claim、同一 run id、启动前复核、启动失败终态、canonical lineage、24 小时 catch-up、超窗 missed、overlap missed、one-shot CAS、DST、手动/暂停运行、启动恢复、canonical terminal 收口与时钟回拨合同已通过定向回归；真实 OS sleep/wake 事件和跨平台证据尚未完成 |
| ST-4 GUI | `complete` | 一级入口、主从工作台、创建/编辑/暂停/运行历史和五语种已落盘；仓库 GUI smoke 与 Scheduled Tasks 专项真实 Electron Gate B 已通过 |
| ST-5 对话与通知 | `complete` | typed terminal notification、Renderer 全局 bridge、Desktop Host failure toast、实时刷新和五语种文案已完成并通过回归 |
| ST-6 清理与验证 | `complete / platform-evidence-pending` | 旧 Automation 双轨、validator、软删除和运行中删除合同已完成；Rust related、contracts、frontend typecheck、GUI/navigation/i18n/governance、Scheduled Tasks Gate B、Agent fixture、GUI smoke、`verify:local`、fmt 与 diff check 全部通过；只剩 Windows Notification Center、Windows Gate B 和真实 OS sleep/resume 平台证据 |

## 已执行验证

```text
cargo test -p app-server-protocol scheduled_task -- --nocapture
  2 passed

npx vitest run src/components/scheduled-tasks/ScheduledTasksPage.test.tsx \
  src/components/scheduled-tasks/scheduledTaskViewModel.unit.test.ts \
  src/lib/api/scheduledTasks.test.ts
  11 passed

npx vitest run src/lib/navigation/sidebarNav.test.ts \
  src/components/AppSidebar.preferences.test.tsx \
  src/components/scheduled-tasks/scheduledTaskViewModel.unit.test.ts \
  src/lib/api/scheduledTasks.test.ts
  23 passed

npx eslint <scheduled tasks / navigation 定向文件>
  passed

npm run detect-translations -- --format json
  5 locales / scheduledTasks 155 keys / no issues
```

新增 `lime-rs/crates/app-server/tests/scheduled_tasks_jsonrpc.rs`，覆盖 public JSON-RPC preview、CRUD、启停、手动运行、真实 lineage identity、运行历史和最近运行 read model。

2026-08-13 边界收口验证：

```text
cargo test -p app-server scheduled_task --lib -- --nocapture
  8 passed

cargo test -p app-server --test scheduled_tasks_jsonrpc -- --nocapture
  2 passed

cargo check -p app-server --lib
  passed
```

2026-08-13 scheduler worker 验证：

```text
cargo test -p lime-scheduler claim -- --nocapture
  5 passed

cargo test -p app-server --lib automation_execution -- --nocapture
  6 passed

cargo test -p app-server --lib scheduled_task_worker -- --nocapture
  3 passed

cargo test -p app-server --test scheduled_tasks_jsonrpc -- --nocapture
  2 passed

cargo check -p app-server --lib --bin app-server
  passed

cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check
git diff --check
  passed
```

2026-08-13 catch-up/missed 验证：

```text
cargo test -p lime-core database::dao::agent_run -- --nocapture
  6 passed

cargo test -p app-server --lib scheduled_task_read_models -- --nocapture
  2 passed（ptrcomp archive）

cargo test -p lime-scheduler claim -- --nocapture
  5 passed（ptrcomp archive）

cargo test -p app-server --lib automation_execution -- --nocapture
  6 passed（ptrcomp archive）

cargo test -p app-server --lib scheduled_task_worker -- --nocapture
  6 passed（ptrcomp archive）

npx vitest run src/components/scheduled-tasks/ScheduledTasksPage.test.tsx \
  src/components/scheduled-tasks/scheduledTaskViewModel.unit.test.ts \
  src/lib/api/scheduledTasks.test.ts
  11 passed

npm run detect-translations -- --format json
  5 locales / scheduledTasks 156 keys / no issues

cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check
git diff --check
  passed
```

worker 只在 `AppServerBackendMode::Runtime` 启动；每轮扫描 `automation_jobs`，按 Scheduled Task marker 过滤后通过 scheduler 原子 claim，复用 claim 创建的 `run_id`，在启动前复核 task revision/enabled/window/ownership，并用真实 RuntimeCore Turn 驱动终态。旧 `scheduled_tasks` 表未被读取或双写。

2026-08-13 ST-3 overlap/DST/manual 验证（使用已发布 `ptrcomp` Apple ARM archive 与对应 binding）：

```text
cargo test -p lime-scheduler claim -- --nocapture
  7 passed

cargo test -p app-server --lib automation_execution -- --nocapture
  8 passed

cargo test -p app-server --lib scheduled_task_worker -- --nocapture
  9 passed

cargo test -p app-server --test scheduled_tasks_jsonrpc -- --nocapture
  3 passed

cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check
git diff --check
  passed
```

覆盖证据：overlap 扫描写入 `scheduled_run_overlap` missed 历史且不释放当前 running ownership；terminal 写回不覆盖已推进的下一窗口；one-shot 的 `NULL next_run_at` 仍可原子 claim/start/finish；America/New_York 春季缺失小时顺延到第一合法时刻、秋季重复小时只执行一次；暂停任务允许立即运行，启用任务手动运行不漂移 next-run 锚点。`trigger` 按合同区分 `schedule`、`catch_up`、`manual`。

2026-08-13 启动恢复/时钟跳变验证（使用同一 `ptrcomp` archive）：

```text
cargo test -p lime-scheduler claim -- --nocapture
  11 passed

cargo test -p app-server --lib scheduled_task_worker -- --nocapture
  13 passed

cargo fmt --manifest-path lime-rs/Cargo.toml --all -- --check
git diff --check
  passed
```

覆盖证据：worker 启动首轮只处理带 Scheduled Task marker 的遗留 ownership，旧 Automation 的 running 状态不被改写；queued/running Run 原子终止为 `error / scheduled_run_interrupted` 并清除 Task ownership，普通 claim 已推进的 `next_run_at` 保留，运行中编辑则按最新启用状态和 schedule 重算；canonical Thread/Turn 已有终态时复用真实终态收口，不伪造 interrupted；重复启动恢复幂等；时钟回拨后同一 deterministic run id 仅保留一次。

上述 3 条 App Server 命令在本机使用上游 v150.4.0 已发布的 `ptrcomp` Apple ARM archive 与对应 binding 显式覆盖，仅用于源码编译/定向测试；裸 `cargo test` 未经仓库 wrapper 时仍会请求未发布的 `ptrcomp_sandbox` archive，质量门禁统一使用 `npm run test:rust:*` wrapper。测试证明：旧/非法 Automation 行不进入 Scheduled Task read model 且 read/update/delete/run fail closed；列表/详情投影最近运行与失败 attention；`continue_thread` 使用 canonical session/thread identity。

## 平台侧后续验证

- Windows runner：真实 Notification Center 展示与 Host failure 行为。
- Windows runner：Scheduled Tasks Gate B，证明 Electron/preload/IPC/App Server/RuntimeCore/GUI 同一 identity 闭环。
- macOS 与 Windows：真实 OS sleep-resume，证明 catch-up、missed、overlap 与 one-shot 恢复行为。
- 当前仓库没有可复用的 sleep/wake 或 Windows Notification Center runner；本机未执行系统休眠/唤醒，也未把受控时钟或 macOS Electron fixture 记作平台证据。

2026-08-13 v1.127.0 发布候选验证：

```text
npm run verify:app-version
  passed（1.127.0）

npm run typecheck
  passed

npm run test:contracts
  passed（protocol types 无漂移；app-server client 311 checks；command contracts passed）

npm run test:rust:related -- lime-rs/crates/app-server-protocol lime-rs/crates/app-server \
  lime-rs/crates/core/src/database/dao/agent_run.rs lime-rs/crates/scheduler
  passed（包含 app-server 1675、agent-runtime 208、app-server-protocol 122、tool-runtime 340 项）

npx vitest run <scheduled-tasks / navigation / app-shell 定向测试>
  8 files / 70 tests passed

npm run verify:gui-smoke
  passed（真实 Electron + App Server 初始化，result=pass）

git diff --check
  passed
```

`npm run test:related -- <changed frontend paths>` 因测试编排器将仓库 `electron/` 目录作为文件读取而报
`EISDIR`；相同功能面已改用显式 Vitest 文件清单执行并全部通过。该结果记为编排器路径解析缺陷，
不是产品断言失败。

## 阻塞与风险

- 本地实现与门禁无阻塞。裸 `cargo test` 仍可能因未发布的 `ptrcomp_sandbox` archive 返回 404；仓库 `npm run test:rust:*` wrapper 已固定本地 v150.4.0 Apple ARM artifact，且相关 related 回归已通过。该结论只覆盖当前 macOS arm64 工具链，不替代 Windows 平台证据。
- 公开 Automation 双轨、旧 Settings consumer、旧 GUI route 已完成 `dead / deleted / forbidden-to-restore` 清理；Base Setup / Service Skill 的 `automation_job` binding family 仍属于 `compat` taxonomy，不得机械删除。
- 启动恢复、interval sleep/wake reconcile 与时钟跳变源码合同已完成定向验证；真实 macOS/Windows OS sleep/resume 进程证据仍需补齐，且 Windows Notification Center 必须由真实 Windows runner 证明。

## 架构确认

```text
架构影响：重大；新增 Scheduled Tasks public JSON-RPC/read-model 边界与一级工作台，运行复用 RuntimeCore/Thread/Turn/Item。
架构图已更新：internal/aiprompts/architecture.md#73-scheduled-tasks；internal/aiprompts/commands.md#scheduled-tasks-主链。
责任开发者确认：root（2026-08-17）
确认内容：已核对目录归属、数据流、依赖方向、协议边界、通知/软删除行为和验证门禁。
```

架构确认已满足 release evidence 入口；Windows 平台证据仍必须使用真实 Windows runner，macOS 不能替代该平台结论。
