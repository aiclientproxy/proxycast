# Coding Profile 实施计划

> 状态：active
> 更新时间：2026-06-15

## 完成口径

Coding profile 完成不是单次模型能改文件，而是以下链路成立：

```text
turn start
  -> coding profile resolved
  -> model/tool/policy resolved
  -> file/patch/command/test/action facts emitted
  -> read model hydratable
  -> AgentUI projection stable
  -> Coding Workbench usable
  -> evidence export joinable
```

本计划按全量实现制定，不按 MVP 口径收口。每个阶段必须同时回答三件事：

1. `current owner` 是否明确，不能把 compat adapter 或旧 UI 当主链。
2. 运行事实是否可 replay / hydrate，不能只靠当前页面内存可见。
3. GUI 是否能展示下一步动作，不能只证明事件或单测存在。

## 阶段依赖

| 阶段                | 依赖                                                | 交付物                                                | 不能替代它的事项                        |
| ------------------- | --------------------------------------------------- | ----------------------------------------------------- | --------------------------------------- |
| P0 标准与盘点       | Agent Workbench 标准、现有 Workspace coding surface | current/compat/deprecated/dead 分类和退出条件         | 继续修旧 `code_orchestrated` 局部行为。 |
| P1 RuntimeEvent     | P0                                                  | schema、sequence、fixture、Rust emission contract     | 只在前端 adapter 合成事件。             |
| P2 ExecutionBackend | P1                                                  | file/patch/command/test/search/browser/MCP tool owner | 模型直接输出 shell 或 patch 文本。      |
| P3 Projection       | P1                                                  | `CodingWorkbenchView` 与 replay/hydration guard       | React 组件自建状态机。                  |
| P4 UI               | P3，部分依赖 P2 fixture                             | 完整 Coding Workbench surfaces 和 GUI smoke           | 只迁 change view 或只显示正文。         |
| P5 多模型           | P1                                                  | profile slot、routing diagnostics、provider readiness | 产品页直连 key 或固定供应商。           |
| P6 外部 harness     | P1/P3                                               | compat event adapter 和 degraded UI                   | 让外部 CLI 成为生产必需主链。           |
| P7 Conformance      | P1-P6 持续输入                                      | fixture、contract、projection、GUI、evidence 验收矩阵 | 单一 `verify:local` 结果。              |
| P8 清理守卫         | P1-P7 current 闭环成立后                            | 删除/限制旧入口和回流守卫                             | 口头约定旧入口不再用。                  |

## 实现 owner 矩阵

| 能力                   | Rust owner                                               | TS / UI owner                       | 必需测试                                           |
| ---------------------- | -------------------------------------------------------- | ----------------------------------- | -------------------------------------------------- |
| coding profile resolve | `lime-rs/crates/app-server` + `lime-rs/crates/agent`     | runtime client request builder      | Rust profile unit + client contract。              |
| model slot routing     | Provider Store / Model Registry crates                   | settings / diagnostics projection   | provider readiness unit + UI diagnostics test。    |
| file read/write        | ExecutionBackend file module + artifact/checkpoint owner | `CodingWorkbenchView.files/changes` | Rust file tool + projection fixture。              |
| patch apply            | `patch-apply` crate + ExecutionBackend patch module      | `PatchView` / change tab            | patch parser/apply unit + failure fixture。        |
| command execution      | ExecutionBackend command module + Project Shell bridge   | output/log tab                      | command lifecycle Rust test + GUI fixture。        |
| test execution         | ExecutionBackend test module                             | test result / continue fix action   | test lifecycle unit + projection fixture。         |
| approval / policy      | Policy service + RuntimeCore action owner                | action card callbacks               | action required/resolved contract + hydrate test。 |
| sandbox blocked        | sandbox manager + policy service                         | blocked state / diagnostics         | platform policy unit + UI blocked fixture。        |
| evidence export        | evidence owner + read model                              | evidence lane / diagnostics drawer  | evidence join test + replay fixture。              |

## P0：标准与现状盘点

目标：把现有编程能力分类到 `current / compat / deprecated / dead`，避免后续实现继续长平行链路。

动作：

- 盘点 `code_orchestrated`、Project Shell、file checkpoint、AgentUI projection、sequence gate、Workspace Harness 编程面板。
- 把每个能力映射到 `RuntimeEvent / ThreadReadModel / TaskSnapshot / EvidencePack`。
- 确认当前哪些 facts 已可从 App Server current API 读取。
- 建立 coding conformance fixture 名单。
- 在 `internal/roadmap/coding/` 内维护 current/compat/deprecated/dead 表，所有后续代码实现必须回挂到对应阶段。
- 对旧 coding 实现执行“无兼容需求优先清退”判断：如果旧实现只是在 UI 内解释 thread item，不再承接新业务逻辑。

验收：

- `runtime-capability-map.md` 的 `copy/rewrite/reference/forbidden` 分类可执行。
- 旧 UI / hook / local state 都有退出条件。
- 路线图中不再把外部品牌、外部 CLI 或参考仓库名写成 Lime current owner。
- 文档能直接指导下一刀进入 `lime-rs/crates/**` current owner。

当前状态：`in_progress`。Workspace coding UI 已向 `RuntimeEvent / thread_read / CodingWorkbenchView` current 输入收敛；file checkpoint 仍作为 current read model 的辅助 summary，旧 thread item 只允许历史测试 / fixture 残留，不再作为 production coding 输入。

## P1：RuntimeEvent 完整事件族

目标：让 coding turn 的关键事实进入标准事件流。

必须新增或对齐事件语义：

- `file.read`
- `file.changed`
- `patch.started`
- `patch.applied`
- `patch.failed`
- `command.started`
- `command.output`
- `command.exited`
- `test.started`
- `test.completed`
- `sandbox.blocked`
- `permission.requested`
- `permission.resolved`
- `action.required`
- `action.resolved`
- `artifact.changed`
- `evidence.changed`
- `snapshot.updated`

动作：

- 事件入库前执行 schema gate。
- 扩展 sequence gate，检查 command/patch/test/action 配对。
- 大输出写入 output ref，不重复塞事件。
- 失败事件包含 failure category、exit code、recovery hint refs。
- Rust App Server 分发前复用同等 schema/sequence 语义，不只在 TS conformance 包里校验。
- `ThreadReadModel` 同步记录 active / terminal command、patch、test、action，支持断流后 hydrate。
- `state.delta` 或 read model repair 不能绕过同一事件字段要求。

验收：

- 纯文本 turn、文件写入、补丁失败、命令审批、沙箱阻断、测试失败修复、hydration repair fixture 可 replay。
- Projection 不从 assistant prose 推断状态。
- sequence gate 能拒绝孤立 `patch.applied`、`command.exited`、`test.completed`。
- `turn.completed / failed / canceled` 后同 turn 不能再追加 file/patch/command/test/action 执行流事件。
- App Server fixture backend 产出的 events 能通过 `@limecloud/agent-ui-contracts` 校验。

当前状态：`completed for P1 spine / backend read-model enrichment completed / P3 read-model adapter connected`。已完成 App Server Rust 侧 schema gate、patch/command/test sequence gate、backend emission fail-closed 测试；RuntimeCore 会拒绝孤立 `command.exited`、缺 artifact 的 `file.changed` 和 turn 终态后的 execution stream。active command/test/action 后端 read model 与 current timeline hydrate 已完成；`@limecloud/agent-runtime-projection` 已能从 `thread_read.commands/tests/pending_requests` 合并出 `CodingWorkbenchView.commands/tests/actions` 和 active ids，Workspace coding adapter 已把 `threadRead` 传入 selector，输出 tab 已直接渲染 `CodingWorkbenchView.commands/tests/actions/diagnostics`，日志 tab 与 action-submit view 也已改为从同一 `CodingWorkbenchView` 派生。剩余是补 GUI smoke evidence，并继续扩诊断抽屉里的 provider / policy / evidence 聚合细节。

## P2：ExecutionBackend Coding Tools

目标：补齐文件、补丁、命令、测试、搜索、MCP、browser 执行面，并接 policy。

### 2026-06-13 Runtime 模块化进度

本轮完成 App Server runtime 巨型文件拆分收口：

- `lime-rs/crates/app-server/src/runtime.rs` 已从约 7500 行级中心文件收缩到约 527 行，保留 facade、核心类型、构造和共享 helper。
- 领域能力下沉到 `runtime/**`，当前约 66 个 runtime 子文件；新增 coding/runtime 后端逻辑不得再回填中心文件。
- `runtime/app_data.rs` 拆成 facade + `runtime/app_data/*.rs` 分域 trait，`AppDataSource` 只作为组合 trait 保留。
- `local_data_source.rs` 收缩到约 420 行，分域 impl 下沉到 `local_data_source/impls/**`，避免数据源 impl 继续堆回根文件。
- `RuntimeGatewayAgentRunner` 接入 App Server current `RuntimeCore::start_turn`，Gateway 入站消息不再把 `DbConnection` 误传给 Telegram / Feishu / Discord runner 参数。

验证证据：

- `cargo fmt --manifest-path "lime-rs/Cargo.toml" -p app-server -p lime-gateway -p lime-websocket --check`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::evidence_exports --no-run`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::gateway_runner::tests -- --nocapture`

剩余风险：

- gateway crate 仍有历史未用函数 warning，属于后续 gateway 运行时清理，不阻塞本轮 runtime 拆分。
- AppServer 仍有未用 `runtime_arc` helper warning，属于 JSON-RPC server 侧后续清理；本轮不扩大为无关重构。
- 真实 output snapshot、file checkpoint snapshot、历史 current timeline hydrate、active command/test/action read model、Basic Evidence Pack snapshot artifact 与基础 policy/sandbox diagnostics facts 已接入 RuntimeCore / RuntimeBackend mirror；sandbox 基础 owner 已进入 `agent_tools::execution::sandbox`，完整 Policy service、分平台 sandbox backend 与 P3/P4 UI evidence 仍属于后续主线。

动作：

- 文件读取/写入统一产生 artifact/checkpoint refs。
- 补丁应用返回结构化 diff 和失败原因。
- 命令执行接入 approval policy、sandbox、stdout/stderr spill refs。
- 测试执行标记 `passed/failed/canceled/timed_out`，不靠输出文本判断。
- 搜索/上下文工具输出 source refs，供 evidence join。
- MCP / browser 工具进入同一 tool inventory，不另建编程专用 registry。
- 所有 tool outcome 都写入 RuntimeEvent，再由 ReadModel 派生 UI。
- 将可复制的 patch parser、policy rule、file search、output truncation 等实现迁入 Lime current owner；迁入时重命名、补许可证记录和定向测试。
- 任何 shell/network/filesystem 副作用先过 policy decision；需要用户介入时必须先产生 `action.required`。
- 浏览器预览和测试服务器属于 tool surface，不允许由 React 组件直接启动。

验收：

- 拒绝审批后工具不继续执行。
- 沙箱阻断后 UI 显示 blocked。
- 测试失败可生成继续修复 turn metadata。
- 大输出不进入 event payload，只进入 output ref / evidence ref。
- 命令、测试和 patch 的失败分类可被 projection 直接消费。
- Windows / macOS / Linux 平台差异进入 policy/sandbox diagnostics，不隐藏为普通失败。

当前状态：`in_progress / current coding skeleton usable / P2-A live-process retained output bounded / P2-B no-sandbox live process landed`。P2 主干已接入 runtime tool event mirror、真实 `patch-apply` apply service、per-file checkpoint/content/diff refs、policy/sandbox/action facts、output snapshot、file checkpoint snapshot、history hydrate、active command/test/action read model 与 Basic Evidence Pack snapshot artifact。`tool-runtime::sandbox` 已承接 sandbox label、命令文本抽取、只读 shell 分类和 workspace sandbox backend plan；Agent 前台 `Bash/PowerShell` 已能消费 workspace sandbox context，macOS seatbelt 与 Linux bubblewrap 在可用时进入 `ready/enforced=true` current runner。Windows 只完成 `restricted_token` backend 类型、tool orchestrator 映射和 App Server/Settings readiness 控制面；`tool-runtime` plan 仍为 `Planned/enforced=false`，command lowering fail closed，当前没有 Windows runner 或 Windows-only runner smoke 可作为 enforce evidence。P2-A 当前只证明 `tool-runtime::execution_process` 的 live process retained output 有界；通用 sandbox output buffer 仍需在各 backend 执行边界补齐，RuntimeCore 继续作为 output snapshot owner。P2-B 已新增 `tool-runtime::execution_process` owner：process snapshot、stdout/stderr delta、有界 retained output、stdin write、interrupt、terminate、status 与本地 process runner；App Server current 已新增 `executionProcess/start|writeStdin|interrupt|terminate|status|drainOutput` JSON-RPC 控制面，并同步 App Server protocol schema、`packages/app-server-client` 方法、processor 分发与 JSON-RPC 定向测试；no-sandbox shell path 已在 `ToolExecutionDecision` 允许、无需 workspace sandbox backend、且 Agent registry permission/safety preflight 通过后走 live process，并输出 `tool.output.delta` 与 process terminal metadata；`executionProcess/start` 已收紧为受控 current 入口，使用 `workingDirectory` 作为唯一实际 cwd，忽略兼容字段 `cwd`，并在 workspace sandbox backend required / enforced 时 fail-closed。再次对照上游后，P2 仍有两条高价值骨干缺口：P2-B 下一刀需要把 command/test 默认执行从 batch outcome bridge 切到 sandbox-aware process runner/control owner，并复用 `executionProcess/*` current API；P2-C 必须先实现 Windows restricted-token runner，再补持久 capability SID、TokenDefaultDacl、STARTUPINFOEXW handle allowlist / desktop、read deny 和 network deny。

2026-06-14 增量：

- `agent_tools::execution` 已拆为 facade + `execution/policy.rs` + `execution/decision.rs` + `execution/tests.rs`，中心入口只 re-export，避免继续把 policy、decision、测试堆回单文件。
- 执行前置决策 owner 已进入 `agent_tools::tool_orchestrator`：真实 `ToolRegistry::execute` 前先计算 `ToolExecutionDecision`，`on_request / unless_trusted / granular` shell 风险会先产出 `ActionRequired(tool_confirmation)` 与 `action.required` metadata，`never` 不做前置人工确认，仍交给既有权限 / sandbox 执行路径判定。
- `ToolExecutionDecision` 已补 `SandboxBlocked` 分支：`read-only` sandbox 下的非只读 shell 命令会在真实执行前产出 `sandbox.blocked` metadata，保守放行 `pwd/ls/find/rg/grep/cat/git status/git diff` 等只读探查命令；主 Agent 工具链通过 `WorkspaceToolPolicyInspector` 复用同一 decision，避免主链绕过 preflight。
- `agent_tools::execution::sandbox` 已成为基础 sandbox owner：集中解析 `read-only/workspace-write/danger-full-access`，抽取 shell command，并用定向测试覆盖 read-only sandbox 的允许/阻断判定；`ToolExecutionDecision` 只消费 sandbox evaluation。
- `agent_tools::execution::rules` 已成为默认 policy rule catalog owner：`policy.rs` 只保留 facade / permission construction，默认工具规则集中声明并通过 catalog guard 测试确保只引用 current 已注册 canonical 工具名，避免未注册工具被伪装成现役 policy；shell command rule classifier 已输出 `commandRuleId/commandRuleSource/commandRiskLevel/commandRiskReasonCode/commandRiskReason` metadata，覆盖 git state mutation、递归/强制删除、提权、网络下载、权限变更和包管理器变更等基础风险，且 `rm` 风险判定已收窄到真实 `-r/-R/-f/--recursive/--force` 选项，避免路径名误触发危险删除规则。
- `agent_tools::execution::service` 已成为 policy resolver owner：集中合并默认规则、持久化 `agent.tool_execution`、请求级 `harness.executionPolicy`，并统一输出 metadata facts；`policy.rs` 继续作为公开 facade 与 permission construction owner，不再承接 override 解析细节。
- `agent.tool_execution.shellCommandRules` 与请求级 `harness.executionPolicy.shellCommandRules` 已接入同一 classifier，支持 `ruleId/pattern/riskLevel/reasonCode/reason` camelCase / snake_case 输入；命中优先级按风险级别和来源共同排序，同风险下 `runtime > persisted > default`，请求级规则可覆盖默认命令风险 metadata。
- `execution/service.rs` 已支持多来源 runtime policy layer：`organizationExecutionPolicy`、`userExecutionPolicy`、`executionPolicy`、`requestExecutionPolicy` 按 `organization -> user -> runtime -> request` 合并，`ToolExecutionPolicySource` 与 `commandRuleSource` 会分别输出 `organization/user/runtime/request`，用于后续 UI / evidence 解释策略来源。
- `ToolExecutionCommandRuleConfig.matchType` 已支持 `regex / prefix / exact`，默认仍为 `regex` 保持兼容；组织 / 用户 / 请求级策略可用 prefix/exact rule 表达稳定命令前缀或精确命令，不必把所有规则写成正则。
- `ToolExecutionPolicyConfig` 已拆出 `config/tool_execution.rs`，避免继续膨胀 `config/types.rs`；`networkRules` 已接入 persisted / organization / user / runtime / request 同一 policy layer，支持 `target=url|host` 与 `matchType=regex|prefix|exact`，可为 `WebFetch` URL 和 `curl/wget` shell 命令中的 URL 输出 `networkRuleId/networkRuleSource/networkRiskLevel/networkRiskReasonCode/networkRiskReason/networkRuleTarget/networkUrl/networkHost` metadata。
- `agent_tools::execution::sandbox` 已补分平台 sandbox backend plan owner：按 macOS / Linux / Windows / unsupported 映射 `seatbelt / linux_sandbox / restricted_token / none`，输出 `sandboxBackend/sandboxBackendStatus/sandboxBackendEnforced/sandboxBackendReasonCode/sandboxBackendPlatform/workspaceSandbox*` diagnostics；macOS `sandbox-exec` 与 Linux `bwrap` 可用时标记 `ready/enforced=true`，并通过 `ToolContext.workspace_sandbox` 接入 Agent 前台 `Bash/PowerShell` runner；Windows restricted token 只接到 backend 类型和 tool orchestrator 映射，plan 固定为 `Planned/enforced=false`，command lowering fail closed，不静默回退无沙箱。
- `WorkspaceSandboxConfig` 已进入 App Server runtime turn context metadata：非默认 `agent.workspaceSandbox` 会与 `agent.toolExecution` 一起注入 `config.agent`，execution decision 可从 persisted / organization / user / runtime / request 多路径解析 workspace sandbox 配置；显式 `enabled=true, strict=true` 且 backend 未 enforce 时会在执行前返回 `SandboxBlocked`，`danger-full-access` 不要求 workspace sandbox backend。
- `ToolExecutionBatchInput` 已显式携带 `auto_mode` 与 `bypass_restrictions`，让执行器使用同一份 workspace tool policy，而不是由模型或调用点隐式决定。
- `NativeAgentConfig.agent.tool_execution` 已进入 App Server runtime turn context：`RuntimeBackend` 在 turn start 边界读取当前配置，并把非默认 `agent.toolExecution` 注入 `TurnContextOverride.metadata.config`；`agent_tools::execution`、`tool_orchestrator`、`WorkspaceToolPolicyInspector` 与 App Server tool inventory 均可从同一 metadata / persisted policy 输入解析 effective policy。运行时 `harness.executionPolicy` 仍高于持久化配置。
- `agent_tools/tool_orchestrator.rs` 已按仓库体量规则拆出 `agent_tools/tool_orchestrator/tests.rs`，中心文件从 1000 行以上收回到约 500 行，只保留 production orchestration；后续 policy/sandbox 测试继续进子模块，避免中心 owner 再次膨胀。
- `workspaceConversationWorkbenchViewModel` 已收缩为 current projection facade，legacy thread item 直连适配已删除；current 主路径继续只消费 projection，不再把旧 item 适配逻辑散落在同一个 facade 文件中。
- 验证证据：`cargo fmt --manifest-path "lime-rs/Cargo.toml" -p lime-agent --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_orchestrator -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_policy_inspector -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::tool_events -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::coding_events -- --nocapture`。
- 追加验证证据：`cargo fmt --manifest-path "lime-rs/Cargo.toml" -p lime-agent -p app-server --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::tests -- --nocapture`。
- 本轮追加验证证据：`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::tool_inventory -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution -- --nocapture`。
- 本轮新增验证证据：`cargo test --manifest-path "lime-rs/crates/agent-rust/crates/agent/Cargo.toml" --target-dir "lime-rs/target" --lib handle_approval_tool_requests_should_resume_after_manual_confirmation -- --nocapture` 证明人工确认后 Agent 工具请求会继续入队执行；`cargo fmt --manifest-path "lime-rs/Cargo.toml" -p lime-agent --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution -- --nocapture` 证明默认 policy rule catalog、PolicyService resolver owner、shell command rule classifier 与基础 sandbox owner 稳定。
- 本轮追加验证证据：`cargo fmt --manifest-path "lime-rs/Cargo.toml" -p lime-core -p lime-agent --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-core tool_execution_policy_config -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-coding-policy-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution -- --nocapture`、`git diff --check` 窄范围检查，证明 `shellCommandRules` config、runtime override metadata 与 classifier 回归稳定。
- 本轮快速落地验证证据：`cargo fmt --manifest-path "lime-rs/Cargo.toml" -p lime-agent --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution::tests -- --nocapture`、`git diff --check` 窄范围检查，证明组织 / 用户 / runtime / 请求策略层级合并和请求级 shell rule 覆盖稳定。
- 本轮 prefix catalog 验证证据：`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-core tool_execution_policy_config -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution::tests -- --nocapture` 证明 `matchType=prefix/exact` 配置解析、roundtrip 与 shell classifier 命中稳定。
- 本轮 sandbox backend 验证证据：`CARGO_TARGET_DIR="/tmp/lime-coding-sandbox-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution::sandbox_backend_tests -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-coding-sandbox-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-coding-sandbox-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::tests::injected_workspace_sandbox_config_flows_to_turn_context_metadata -- --nocapture`，证明 backend plan、strict fallback、danger-full-access 例外、request 覆盖 persisted sandbox 配置与 App Server 配置注入稳定。
- 2026-06-15 命令事实增量：`runtime_backend/coding_events/command.rs` 已成为 App Server coding command facts owner，负责把 shell wrapper / argv / PowerShell command 规范化为 `canonicalCommand`、`commandSummary`、`commandArgv`、`commandArgvSource`；`command.started / command.exited / test.started / test.completed / permission.denied / sandbox.blocked` 均透传同一组命令事实，`thread_read.commands/tests` 和 `CodingWorkbenchView` 可 hydrate 后继续使用稳定命令身份，输出 / 日志 / session overview 优先显示规范化摘要，完整 raw command 只作为详情保留。
- 2026-06-15 命令事实验证证据：`cargo fmt --manifest-path "lime-rs/Cargo.toml" -p app-server --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::coding_events -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::coding_events::read_model_projects_active_coding_activity_and_pending_action -- --nocapture`、`npm --prefix "packages/agent-runtime-projection" run build`、`node --test --test-name-pattern "projectCodingWorkbenchViewFromEvents consumes current thread read model coding facts" "packages/agent-runtime-projection/tests/projection.test.mjs"`、`npx vitest run "src/components/agent/chat/workspace/CodingWorkbenchOutputPanel.test.tsx" --silent=passed-only --disableConsoleIntercept`。
- 2026-06-15 变更摘要增量：`runtime/coding_activity_projection.rs` 已从 `file.changed` 与 `patch.*` 派生 `thread_read.change_summary`，输出 changed files、patch 计数、running / failed / applied patch 计数、source event ids 与 latest sequence；`packages/agent-runtime-projection` 将其投影为 `CodingWorkbenchView.changeSummary`，Workspace session overview 直接消费该字段展示变更进度，不再从正文或旧 thread item 猜测“本轮改了什么”。
- 2026-06-15 变更摘要验证证据：`cargo fmt --manifest-path "lime-rs/Cargo.toml" -p app-server --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::coding_events -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::coding_events::read_model_projects_active_coding_activity_and_pending_action -- --nocapture`、`npm --prefix "packages/agent-runtime-projection" test`、`npx vitest run "src/components/agent/chat/workspace/codingSessionOverviewProjection.unit.test.ts" "src/components/agent/chat/workspace/CodingWorkbenchOutputPanel.test.tsx" --silent=passed-only --disableConsoleIntercept`。
- 2026-06-15 P2-B first-core-slice 增量：`agent_tools::execution::process` 已成为一等 process owner，提供 process snapshot、stdout/stderr output delta、有界 retained output、stdin write、interrupt、terminate、status、本地 process runner 与 manager；`agent_tools::tool_orchestrator` 的现有 shell batch bridge 会在 `tool_end` 前补 `tool_output_delta`，并把 `processId / executionProcessStatus / outputBytes / outputOmittedBytes / outputTruncated / exit_code` 合入 terminal metadata；App Server `runtime_backend::coding_events` 会把 output delta metadata 透传到 `command.output` payload。
- 2026-06-15 P2-B first-core-slice 验证证据：`rustfmt --edition 2021` 覆盖 `agent_tools/execution.rs`、`agent_tools/execution/process.rs`、`agent_tools/tool_orchestrator.rs`、`agent_tools/tool_orchestrator/tests.rs`、`app-server/src/runtime_backend/coding_events/tests.rs`；`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution::process -- --nocapture` 证明 process owner、本地 stdout/stderr runner 与 terminate 状态稳定；`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_orchestrator::tests::execute_planned_shell_tool_process -- --nocapture` 证明 shell batch bridge 输出 process delta、保留失败 exit code；`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::coding_events::tests::shell_tool_output_delta_preserves_process_lifecycle_metadata -- --nocapture` 证明 App Server mirror 透传 process lifecycle metadata。
- 2026-06-15 P2-B control API 增量：`app-server-protocol` 新增 `ExecutionProcess*` DTO 与 `executionProcess/start|writeStdin|interrupt|terminate|status|drainOutput` 方法；`app-server/src/execution_process.rs` 持有 live process control handle、保留终态 snapshot、缓存 bounded output deltas；`processor/execution_process.rs` 通过 current JSON-RPC 暴露控制面；`packages/app-server-client` 已同步常量、request builder、高层 client 方法和 generated protocol types。
- 2026-06-15 P2-B control API 验证证据：`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution::process -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server execution_process -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server processor::tests::execution_process_methods_start_drain_and_report_status -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol protocol::v0::tests::catalog -- --nocapture`、`node scripts/check-app-server-client-contract.mjs`、`npm run check:protocol-types`、`npm --prefix "packages/app-server-client" run build`。
- 2026-06-15 P2-B no-sandbox live process 增量：`ToolRegistry::check_tool_permissions` 已把 Agent tool-level safety / permission preflight 暴露给外部 process owner；`agent_tools::tool_orchestrator` 只在 shell tool、非 background、`sandboxBackendRequired=false`、`sandboxBackendEnforced=false`、无 `workspace_sandbox` context、registry permission preflight 通过时启动 live process。live process 固定使用 Runtime 注入的 `ToolContext.working_directory`，忽略工具参数中的 `cwd` 覆盖，避免 permission/sandbox 按 workspace 判断但实际进程跑到外部目录。需要 workspace sandbox backend 的命令继续走 Agent sandbox executor，不走裸进程。
- 2026-06-15 P2-B no-sandbox live process 验证证据：`rustfmt --edition 2021 --check "lime-rs/crates/agent/src/agent_tools/tool_orchestrator.rs" "lime-rs/crates/agent/src/agent_tools/tool_orchestrator/tests.rs" "lime-rs/crates/agent/src/agent_tools/execution/process.rs" "lime-rs/crates/agent-rust/crates/agent/src/tools/registry.rs"`、`cargo check --manifest-path "lime-rs/Cargo.toml" -p lime-agent`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_orchestrator -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution::process -- --nocapture`、`cargo test --manifest-path "lime-rs/crates/agent-rust/Cargo.toml" -p agent-core tools::registry -- --nocapture`、`git diff --check -- ...`。`cargo fmt --manifest-path "lime-rs/Cargo.toml" -p lime-agent --check` 仍会报告同 package 其他历史脏文件格式差异，本轮未格式化无关文件。
- 2026-06-15 P2-B App Server process policy 收紧增量：`executionProcess/start` 不再是裸进程入口；它只接受 shell canonical tool，必须携带 `workingDirectory`，启动前复用 Lime `ToolExecutionDecision` 和 Agent `ToolRegistry::check_tool_permissions`，并在 `sandboxBackendRequired=true` 或 `sandboxBackendEnforced=true` 时返回错误，要求继续走 Agent sandbox executor。兼容字段 `cwd` 保留为输入字段但不参与实际进程 cwd，避免按 workspace 做决策却从外部目录启动。
- 2026-06-15 P2-B App Server process policy 验证证据：`cargo check --manifest-path "lime-rs/Cargo.toml" -p app-server`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server execution_process -- --nocapture`、`cargo run --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol --bin write_schema_fixtures`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server-protocol schema -- --nocapture`、`npm run generate:protocol-types`、`npm run check:protocol-types`、`node scripts/check-app-server-client-contract.mjs`、`npm --prefix "packages/app-server-client" test`、`rustfmt --edition 2021 --check "lime-rs/crates/app-server/src/execution_process.rs" "lime-rs/crates/app-server/src/processor/execution_process.rs" "lime-rs/crates/app-server/src/processor/mod.rs" "lime-rs/crates/app-server-protocol/src/protocol/v0/execution_process.rs"`。
- 2026-06-15 P2-B policy/sandbox owner 收口增量：`ToolExecutionDecision` 新增 `requires_sandboxed_execution()` 与 `workspace_sandbox_backend_enforced()` typed helper，`agent_tools::tool_orchestrator` 和 App Server `executionProcess/start` 不再各自读取 `sandboxBackendRequired / sandboxBackendEnforced` metadata key 来决定裸进程或 sandbox context；后续 process / command / test 执行分支必须继续消费 decision helper，避免 policy/sandbox 语义在调用点漂移。
- 2026-06-15 P2-B policy/sandbox owner 验证证据：`cargo fmt --manifest-path "lime-rs/Cargo.toml" -p lime-agent -p app-server --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_orchestrator -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server execution_process -- --nocapture`。
- 2026-06-15 P2-B process facts projection 增量：Lime 已有 policy / sandbox owner，不再重造；`runtime_backend::coding_events` 已把 Agent Bash / PowerShell embedded、sandbox 与 live process metadata 统一投影为 `processId / executionProcessStatus / executionSurface / outputBytes / outputOmittedBytes / outputTruncated / stdoutBytes / stderrBytes`，`runtime/coding_activity_projection.rs` 将这些字段 hydrate 到 `thread_read.commands`，`packages/agent-runtime-projection` 将其暴露到 `CodingWorkbenchView.commands`。这解决主对话 command/test 终态只在 tool metadata 中可见、Workbench read model 不可稳定消费的问题；实时 stdout/stderr 深控制仍归 P2-B process control 下一刀。
- 2026-06-15 P2-B live process start hydrate 增量：`agent_tools::tool_orchestrator` 在 no-sandbox live process 成功启动后立即发出 metadata-only `tool.output.delta`，携带 `processId / executionProcessStatus=running / executionSurface=live_process`；`runtime_backend::coding_events` 允许空 delta 只作为 process lifecycle fact 投影为 `command.output`，不把它计入真实 stdout/stderr，也不阻止 terminal result 继续产出真实输出。这样长时间无 stdout 的命令也能在 `thread_read.commands` 与 Workbench 控制面中立即出现 `processId`。
- 2026-06-15 P2-B live process start hydrate 验证证据：`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_orchestrator -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::coding_events -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::coding_events::read_model_projects_active_coding_activity_and_pending_action -- --nocapture`、`cargo fmt --manifest-path "lime-rs/Cargo.toml" --all -- --check`、`git diff --check -- ...`。
- 2026-06-26 P2-B live process terminal hydrate 增量：`agent_tools::tool_orchestrator` 现在会在 no-sandbox live process `ToolEnd` 前再发出一条 metadata-only `tool.output.delta`，携带 `processId / executionProcessStatus=exited|interrupted|terminated|failed / executionSurface=live_process / exit_code / outputBytes` 等终态 fact；启动 delta、stdout/stderr delta、终态 delta、terminal `ToolEnd` 的顺序被测试锁住，让 Workbench/read model 不必只等最终 `ToolEnd.metadata` 才知道进程已结束。
- 2026-06-26 P2-B live process terminal hydrate 验证证据：`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_orchestrator::tests::execute_planned_shell_tool_emits_process_output_delta_before_terminal_event -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_orchestrator -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution::process -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::coding_events -- --nocapture`、`rustfmt --edition 2021 --check "lime-rs/crates/agent/src/agent_tools/tool_orchestrator.rs" "lime-rs/crates/agent/src/agent_tools/tool_orchestrator/tests.rs"`、`git diff --check -- ...`。
- 2026-06-26 P2-B live process registry seam 增量：`agent_tools::tool_orchestrator` 新增 `LiveExecutionProcessRegistry` 注入点，no-sandbox live process 启动时注册 `LocalExecutionProcessControlHandle`，输出时记录 `ExecutionOutputDelta`，结束时写入最终 `ExecutionProcessSnapshot`；terminal metadata 会标记 `executionProcessControlStatus=registered` 或记录注册错误，避免 UI 控制面只能看到 read model facts、却无法控制同一个 live process。
- 2026-06-26 P2-B App Server process registry owner 增量：`ExecutionProcessServer` 实现 `LiveExecutionProcessRegistry`，并新增 `register_process_handle / record_process_output / finish_process` 三个 owner 方法，把 agent live process control handle、bounded output replay 与 final snapshot 收敛到现有 `executionProcess/status|drainOutput|interrupt|terminate` current API；本轮不新增 JSON-RPC method、不改前端 protocol/client、不接回 legacy command 或 mock。
- 2026-06-26 P2-B live process registry 验证证据：`rustfmt --edition 2021 "lime-rs/crates/agent/src/agent_tools/tool_orchestrator.rs" "lime-rs/crates/agent/src/agent_tools/tool_orchestrator/tests.rs" "lime-rs/crates/agent/src/request_tool_policy/web_search_preflight.rs" "lime-rs/crates/app-server/src/execution_process.rs"`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_orchestrator -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server execution_process -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::coding_events -- --nocapture`。
- 2026-06-26 P2-B RuntimeBackend shared process owner 增量：Agent `Agent` 新增 `NativeToolExecutionHook` seam，并 re-export hook request / `ToolCallResult`；App Server 新增 `runtime_backend/live_execution_process.rs`，只接管 Bash / PowerShell foreground no-sandbox 且 `ToolExecutionDecision` 允许的命令，通过同一个 `ExecutionProcessServer` 启动本地 process、输出 MCP logging lifecycle notification、在 `CallToolResult.meta` 写入 `processId / executionProcessStatus / executionSurface / outputBytes / exit_code` 等 facts；`RuntimeFactory` 在 runtime backend mode 创建同一个 `ExecutionProcessServer` 注入 RuntimeBackend 与 RuntimeCore，`RequestProcessor` 自动复用 RuntimeCore 携带的 server，Workbench `executionProcess/status|drainOutput|interrupt|terminate` 与主对话 shell 执行共享 owner。RuntimeBackend 当前只是外部 host adapter，不直接执行 native tools；后续若要共享 process owner，必须扩展 `RuntimeBackendHost` 合同，不能在 adapter 内伪造本地 registry。
- 2026-06-26 P2-B RuntimeBackend shared process owner 验证证据：`CARGO_TARGET_DIR="/tmp/lime-p2b-live-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::live_execution_process -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-p2b-live-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent event_converter -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-p2b-live-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server execution_process -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-p2b-live-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::coding_events -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-p2b-live-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_factory -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-p2b-live-target" cargo test --manifest-path "lime-rs/crates/agent-rust/crates/agent/Cargo.toml" --lib test_agent_has_tool_registry -- --nocapture`、`rustfmt --edition 2021 --config skip_children=true --check ...` 和 `git diff --check -- ...` 窄范围检查通过。`npm run test:contracts` 已通过 `check:protocol-types`、`check-app-server-client-contract`、`check-command-contracts`、`check-harness-contracts`、`governance:modality-contracts`、`governance:scripts`、`governance:electron-release-workflow`、`harness:cleanup-report:check`，最终 `docs:boundary` 因无关 `internal/roadmap/plugin/*` 旧内部文档路径引用失败，本轮不改插件路线图。
- 2026-06-26 P2-B status hydrate 增量：RuntimeBackend live process lifecycle metadata 现在稳定输出 `executionProcessControlStatus=registered` 与 `stdinWritable/stdin_writable`，MCP logging notification -> `ToolOutputDelta.metadata` -> `command.output` -> `thread_read.commands` -> `CodingWorkbenchView.commands` 全链路透传；Workbench `status|drain|interrupt|terminate` 控制成功后会刷新当前 session read model，不在组件内自建临时状态，也不新增 JSON-RPC method。
- 2026-06-26 P2-B status hydrate 验证证据：`CARGO_TARGET_DIR="/tmp/lime-p2b-live-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::live_execution_process -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-p2b-live-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::coding_events -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-p2b-live-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server read_model_projects_active_coding_activity_and_pending_action -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-p2b-live-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent event_converter::tests::test_convert_mcp_notifications_to_tool_stream_events -- --nocapture`、`npm --prefix "packages/agent-runtime-projection" run build`、`node --test --test-name-pattern "projectCodingWorkbenchViewFromEvents consumes current thread read model coding facts" "packages/agent-runtime-projection/tests/projection.test.mjs"`、`npx vitest run "src/lib/api/executionProcess.test.ts" "src/components/agent/chat/workspace/CodingWorkbenchOutputPanel.test.tsx" --silent=passed-only --disableConsoleIntercept`、`rustfmt --edition 2021 --config skip_children=true --check ...`、`git diff --check -- ...`。`npm run test:contracts` 仍只在无关 `internal/roadmap/plugin/*` docs boundary 失败，前置契约全部通过。
- 2026-06-26 P2-B stdin UX 增量：`CodingWorkbenchOutputPanel` 现在只在 `CodingWorkbenchView.commands` 同时具备 `processId`、live `executionProcessStatus` 与 `stdinWritable=true` 时显示 stdin 输入；提交后通过 `workspaceConversationCodingViews.tsx` 注入的 handler 调用 `src/lib/api/executionProcess.ts -> executionProcess/writeStdin` current API，成功后刷新 current session read model。组件只消费 injected control，不直接散落 JSON-RPC；未新增 App Server method、legacy command、DevBridge mock 或 renderer fallback。
- 2026-06-26 P2-B stdin UX 验证证据：`npx prettier --check "src/components/agent/chat/workspace/CodingWorkbenchOutputPanel.tsx" "src/components/agent/chat/workspace/CodingWorkbenchOutputPanel.test.tsx" "src/components/agent/chat/workspace/workspaceConversationCodingViews.tsx" "src/components/agent/chat/workspace/workspaceConversationCodingViews.unit.test.tsx" "src/i18n/resources/zh-CN/agent.json" "src/i18n/resources/zh-TW/agent.json" "src/i18n/resources/en-US/agent.json" "src/i18n/resources/ja-JP/agent.json" "src/i18n/resources/ko-KR/agent.json"`、`npx vitest run "src/components/agent/chat/workspace/CodingWorkbenchOutputPanel.test.tsx" "src/components/agent/chat/workspace/workspaceConversationCodingViews.unit.test.tsx" --silent=passed-only --disableConsoleIntercept`、`git diff --check -- ...`。
- 2026-06-26 P2-B RuntimeBackend host contract 增量：`RuntimeBackendSubmitRequest` 新增 `process_control` 合同字段，默认 `RuntimeBackend::new` 使用 `RuntimeBackendProcessControlCapabilities::none()`，不会伪造共享 registry；显式 `RuntimeBackend::new_with_process_control` 与 `AppServerRuntimeFactory::agent_runtime_core_with_execution_process_server / agent_app_server_with_execution_process_server` 才声明 shared `ExecutionProcessServer` 的 `status / drain_output / interrupt / terminate / write_stdin` 能力，并让 `RuntimeCore` 携带同一个 process server 供 `RequestProcessor` 复用。本轮不新增 JSON-RPC method、不改前端 protocol/client、不把本地 process owner 塞进 RuntimeBackend adapter。代码搜索确认当前仓库没有生产 `RuntimeBackendHost` 实现，只有 feature-gated 测试 host；外部 host 消费合同必须等真实 host adapter 出现，不能为继续主线伪造入口。
- 2026-06-26 P2-B RuntimeBackend host contract 验证证据：`CARGO_TARGET_DIR="/tmp/lime-p2b-agent-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server --features agent-backend runtime_backend -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-p2b-agent-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server --features agent-backend agent_factory_can_share_execution_process_owner_with_runtime_core -- --nocapture`、`CARGO_TARGET_DIR="/tmp/lime-p2b-agent-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server --features agent-backend runtime_backend_json_rpc_agent_flow_smoke_covers_artifact_read_and_action_response -- --nocapture` 均通过。`rustfmt --edition 2021 --config skip_children=true --check "lime-rs/crates/app-server/src/runtime_backend.rs" "lime-rs/crates/app-server/src/runtime_factory.rs" "lime-rs/crates/app-server/src/lib.rs"`、`npx prettier --check "internal/roadmap/coding/README.md" "internal/roadmap/coding/implementation-plan.md"`、`git diff --check -- ...` 已通过；全量 `rustfmt --edition 2021 --check ...` 曾因无关脏文件 `lime-rs/crates/app-server/src/runtime/plugin_worker_turn.rs` 需要格式化而失败，本轮未改该文件。
- 2026-06-26 P2-B live process lifecycle metadata consistency 增量：`agent_tools::execution::process` 现在在 output delta、start snapshot 与 terminal snapshot metadata 中统一输出 `stdinWritable / stdin_writable`；`agent_tools::tool_orchestrator` 在 no-sandbox live process 注册到 `LiveExecutionProcessRegistry` 后，会把 `executionProcessControlStatus / execution_process_control_status` 同步写入 start lifecycle delta 与 terminal lifecycle delta。这样默认 batch/live shell 路径不只在最终 `ToolEnd.metadata` 才暴露 stdin/control capability，`command.output -> thread_read.commands -> CodingWorkbenchView.commands` 可以从启动阶段恢复同一 process owner 的控制状态。
- 2026-06-26 P2-B live process lifecycle metadata consistency 验证证据：`CARGO_TARGET_DIR="/tmp/lime-p2b-process-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution::process -- --nocapture` 通过 6 个 process owner 测试；`CARGO_TARGET_DIR="/tmp/lime-p2b-process-target" cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_orchestrator -- --nocapture` 通过 14 个 orchestrator 测试，覆盖启动 lifecycle delta 的 `stdinWritable=true`、终态 lifecycle delta 的 `stdinWritable=false`、注册后 start delta 带 `executionProcessControlStatus=registered`，以及 registry output / final snapshot 记录。
- 未完成：P2-B 下一刀回到默认 command/test sandbox-aware process runner/control owner；外部 Agent host 消费 `process_control` 合同只在真实生产 host adapter 出现时继续；Windows restricted-token runner 尚未实现，Windows smoke 和平台 evidence 必须等 runner 进入 `Ready/enforced=true` 后再补；network / policy 单条规则深链属于 P7/P8 增强。P3/P4 UI evidence、Provider slot diagnostics 与 GUI coding input baseline 已进入 current 骨架闭环，不再作为主线 blocker。

## P3：AgentUI Coding Projection

目标：把 RuntimeEvent / ReadModel 投影成 Coding Workbench view model。

动作：

- 在 projection 包内新增或扩展 coding selectors。
- 派生 `FileChangeView / PatchView / CommandOutputView / TestRunView / ApprovalView`。
- Hydration 支持 active command/test/action 恢复。
- `model.completed` 可修正 streaming delta，不重复追加文本。
- sequence gap 标记 `stale`，通过 read model repair。
- selector 输入统一为 RuntimeEvent / ReadModel / TaskSnapshot，不直接接受 Workspace thread item。
- 已删除的旧 thread item adapter 不得恢复；如需历史 hydrate，必须在 current read model / projection 边界完成，不得让 Workspace coding UI 直读旧 item。
- diagnostics 要明确 `missing_scope`、`missing_ref`、`sequence_gap`、`unsupported_fact`、`blocked`。

验收：

- fixture replay 输出稳定 projection snapshot。
- 重复 event 幂等。
- 乱序或缺 id 进入 degraded，不伪造完成态。
- `CodingWorkbenchView` 包含 files / changes / patches / commands / tests / actions / artifacts / evidence / diagnostics。
- file checkpoint 只能作为 current read model summary 输入进入 coding selector / workbench view；旧 thread item 不再作为 production coding selector 输入。
- `coding-command-approval` resolved 后不要求 pending actions 数量保留历史 required/resolved 两条；生命周期历史应由 timeline/diagnostics 或 raw events 表达。
- `expected.coding` fixture 能证明每类 coding fact 都被投影到对应 view。

当前状态：`in_progress / read-model coding facts connected / output-log-action-submit-diagnostics-session projection-driven`。`@limecloud/agent-runtime-projection` 已提供 `CodingWorkbenchView` selector，并可从 RuntimeEvent 与 App Server `thread_read` 双路径合并 `commands/tests/actions/changeSummary`；`active_command_id / active_test_run_id` 会进入 `mainObject`，`pending_requests` 会投影为标准 action projection；Workspace adapter 已将 `threadRead` 传入 selector，输出 tab 已直接渲染 `CodingWorkbenchView.commands/tests/actions/diagnostics`，日志 tab 已通过 `CodingWorkbenchLogPanel` 从同一 `CodingWorkbenchView` 渲染文件、命令、测试、确认和诊断日志，session overview 也已通过 `codingSessionOverviewProjection.ts` 从同一 `CodingWorkbenchView` 生成中性 activity items 与变更摘要，不再合成或直读旧 `AgentThreadItem`。runtime workbench 计数也从同一 projection 推导。`CodingWorkbenchActionPanel` 已从 `CodingWorkbenchView.actions` 生成待确认卡，可对可映射为 `tool_confirmation` 的 current action 调用既有 runtime callback 提交允许 / 拒绝；无法映射的 action 只显示恢复路径，不伪造提交。`CodingWorkbenchDiagnosticPanel` 已展示 fail-closed 策略和 source event evidence 基础事实。下一步补 GUI smoke evidence 与 provider / policy / evidence 聚合细节，不再让 React 组件维护 command/test/action 状态机。

## P4：前端 Coding Workbench

目标：把编程首屏改成中央主画布 + 右侧对话 + 诊断抽屉，并完全消费 projection。

动作：

- 中央固定 tab：预览 / 文件 / 变更 / 输出 / 日志。
- 右侧固定对话、任务进度、审批和输入框。
- 诊断抽屉承载 runtime capability、provider readiness、policy、evidence。
- 失败输出、测试失败、补丁失败共用继续修复入口。
- 用户可见文案覆盖 `zh-CN / zh-TW / en-US / ja-JP / ko-KR`。
- 复杂工作台采用对象带 + 阶段状态 + 主任务画布 + 收纳式诊断，不做营销式 hero 或多层套卡。
- `changeView / outputView / logView / actionView / diagnosticsView` 全部从 `CodingWorkbenchView` adapter 派生。
- 审批、继续修复、停止、刷新预览只调用 runtime client callbacks，不写 runtime truth。
- tab disabled/empty 状态稳定保留，运行中不能因数据为空造成布局跳动。
- 诊断抽屉可显示技术词；普通主流程文案使用业务动作词。

验收：

- 有预览 artifact 时默认展示预览。
- 无预览但有文件时展示主文件。
- 变更、输出、日志 tab 不改变 runtime facts。
- GUI smoke 覆盖提交编程需求、查看输出、继续修复。
- 审批 pending 刷新后可恢复。
- sandbox blocked 不显示成普通失败回答。
- 窄屏和宽屏下按钮文字不溢出，右侧对话不遮挡主画布。
- `*.test.tsx` / view model 单测覆盖 tab 默认选择、blocked、failed、waiting action、changes ready。

当前状态：`in_progress / change-output-log-action-recovery-session current projection connected / runtime hook split / reliability diagnostics policy-network evidence connected / execution policy recovery settings connected / recovery-gui-smoke-evidence connected / gui-coding-input-smoke-aggregated / evidence-export-coding-summary connected / network-rule-recovery-draft connected / provider-focus-recovery connected / policy-source-layer-visibility connected / policy-field-source-reliability connected / sandbox-backend-diagnostics connected`。现有 `CanvasWorkbenchLayout` coding mode 仍是承载壳，但 `WorkspaceConversationSceneRuntime` 已把 `sessionView`、`changeView`、`outputView`、`logView` 从 `CodingWorkbenchView` 派生：session overview 消费中性 activity projection，变更面板消费 projection change facts 与 file checkpoint summary，输出面板消费 commands/tests/actions/diagnostics/recovery，日志面板消费同一 projection 的文件、命令、测试、确认和诊断条目。coding workbench 视图组装已下沉到 `workspaceConversationCodingViews.tsx`，中心 hook 只保留投影延迟、场景参数和 runtime callback 接线；`CodingWorkbenchOutputPanel` 已拆出 `CodingWorkbenchActionPanel`、`CodingWorkbenchDiagnosticPanel`、`CodingWorkbenchRecoveryPanel` 与共享状态 badge，主面板只做组合；action submit 通过 `handlePermissionResponse` 走 existing runtime action response 主链，失败继续修复通过 `handleSendFromEmptyState({ textOverride, sendOptions.requestMetadata.harness.coding_workbench_recovery })` 走 existing send 主链，不直连 App Server、不新增命令、不依赖 mock。诊断面板已展示 fail-closed policy 与 source event evidence 基础信息，Reliability 诊断信息面板已把 policy / sandbox / network 冲突 facts 接入可见 UI 与复制诊断文本，并可打开系统设置里的执行策略页恢复 `agent.workspace_sandbox` 与 `agent.tool_execution`；network rule 现在会派生 `deny / ask / unknown` 中性判定、reason code、解释摘要和可恢复标记，参考本地 network policy decision 的“明确 deny 与可复核 ask 分流”思路，但继续消费 Lime current `thread_read` facts。执行策略设置页已消费 `ExecutionPolicyFocusContext`：命中已有网络规则时高亮定位，未命中时可基于诊断值一键生成 exact 网络规则草案并通过既有 `save_config` current 写链保存，并展示 default / current config / organization / user / runtime / request 的合并顺序与可编辑边界。App Server current `evidence/export` 已在 `observability_summary.coding` 聚合 file change、patch、command、test、action、recovery request 与 output/diff/checkpoint/artifact/evidence/source event refs，Workbench 可见事实可进入 Evidence Pack 复盘链路。provider/key 深链已接到 Settings Providers 与 Provider 设置页，诊断入口可直接聚焦目标服务商并在缺 key / 缺 model 时给出恢复动作；策略字段级来源事实已接入 Reliability 诊断卡与复制诊断文本，能展示 `warning / restriction / sandbox` 分别来自 user / runtime / request 等来源；sandbox backend facts 已接入 Reliability 诊断卡与复制诊断文本，能展示 `backend/status/enforced/required/platform/source/reason`，使非 Windows fallback、缺 backend 和历史 Windows `restricted_token unavailable/enforced=false` 这类状态不再只藏在 raw metadata。下一刀必须先在 `tool-runtime` 实现 Windows restricted-token runner，再在 Windows 真机补 enforce smoke；跨平台 GUI smoke 证据只能验证 current readiness 投影，不能替代 Windows 平台证据。

2026-06-14 P4 增量：

- `CodingWorkbenchActionPanel` 已从 `CodingWorkbenchView.actions` 渲染待确认动作，支持 command approval 的允许 / 拒绝按钮，并通过 `onRespondToAction` 复用现有 runtime callback；`submittedActionsInFlight` 会禁用按钮并显示提交中，避免重复提交。
- 无法安全映射为 current `tool_confirmation` 的 action 只展示提示与请求 id，不伪造提交按钮，符合 fail-closed 交互。
- `CodingWorkbenchDiagnosticPanel` 已从 `CodingWorkbenchView.diagnostics` 展示诊断标题、详情、状态、fail-closed 策略与 source event evidence id，作为后续诊断抽屉聚合的基础 surface。
- `AgentThreadPolicyEvidenceCard` 与 `runtimePolicyEvidence.ts` 已接入 `thread_read.diagnostics`、`thread_read.model_routing`、最近失败 command / tool metadata 中的 `policyName / policyProfile / sandboxPolicy / primary_blocking_* / networkRule* / networkRisk* / networkHost / networkUrl` current facts；Reliability 诊断信息面板会展示 policy / sandbox / network 冲突分组，且普通 routing decision / fallback 不会单独误触发策略卡。
- “复制诊断文本”和 raw JSON 已复用同一 `resolveRuntimePolicyEvidence` 事实源输出 `runtime_policy_evidence`，避免 UI 与导出文本各自解析 policy/network 字段。
- `SettingsTabs.ExecutionPolicy` 已接入设置系统的系统分组、懒加载、预加载与 active tab；`ExecutionPolicySettings` 直接复用 current `get_config/save_config` 写链，不新增 App Server method / Desktop Host command / legacy facade，保存到后端已支持的 `agent.workspace_sandbox` 与 `agent.tool_execution`。
- Reliability policy/network 诊断卡新增“打开执行策略设置”恢复入口；Workspace/Harness 传入回调时打开 `SettingsTabs.ExecutionPolicy`，无回调时不显示假入口。保存 tool override 时 canonical key 固定为 `bash`，并清理 legacy `Bash` override，避免双轨。
- `settings.executionPolicy.*` 与 `agentChat.threadReliability.routingEvidence.policyOpenSettings` 已覆盖 `zh-CN / zh-TW / en-US / ja-JP / ko-KR`。
- `codingWorkbenchRecovery.ts` 已从 `CodingWorkbenchView.commands/tests/patches/diagnostics` 聚合失败命令、失败测试、失败补丁和失败诊断，并带入相关文件与最近 file checkpoint 生成继续修复 prompt 和结构化 `coding_workbench_recovery` context；`CodingWorkbenchRecoveryPanel` 只负责展示失败摘要和主按钮。
- 继续修复按钮已通过 `workspaceConversationCodingViews.tsx -> useWorkspaceConversationSceneRuntime.tsx` 复用 `handleSendFromEmptyState({ textOverride, sendOptions.requestMetadata.harness.coding_workbench_recovery })`，不新增 App Server method、Desktop Host command 或 mock fallback。
- 五语言 `zh-CN / zh-TW / en-US / ja-JP / ko-KR` 已补齐 coding action / diagnostics / recovery presentation 文案。
- `workspaceConversationCodingViews.tsx` 已成为 coding workbench presentation helper，集中构造 `CodingWorkbenchView`、session counters、session/output/log/change view 与 `handlePermissionResponse` callback 传递；`useWorkspaceConversationSceneRuntime.tsx` 从 1000 行以上收回到 1000 行以下，后续不得再向该 hook 追加 coding UI 业务逻辑。
- `codingSessionOverviewProjection.ts` 已从 `CodingWorkbenchView.commands/tests/changes/actions/diagnostics` 生成中性 `CanvasSessionOverviewActivity`，`CanvasSessionOverviewPanel` 新增 `activityItems` current 输入；coding session overview 不再通过旧 `AgentThreadItem` / `threadItems` 合成 command/file/action 展示项。
- 本轮新增验证证据：`npx vitest run "src/components/agent/chat/utils/runtimeRoutingEvidence.test.ts" "src/components/agent/chat/utils/runtimePolicyEvidence.test.ts" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --silent=passed-only --disableConsoleIntercept`、`npx eslint "src/components/agent/chat/components/AgentThreadPolicyEvidenceCard.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" "src/components/agent/chat/utils/runtimePolicyEvidence.ts" "src/components/agent/chat/utils/runtimePolicyEvidence.test.ts" "src/components/agent/chat/utils/runtimeRoutingEvidence.ts" "src/components/agent/chat/utils/runtimeRoutingEvidence.test.ts" "src/components/agent/chat/utils/threadReliabilityDiagnosticText.ts" --max-warnings 0`、`npx prettier --check ...`、`git diff --check -- ...`、`npm run smoke:agent-runtime-current-fixture`。
- 2026-06-14 追加验证证据：`npm test -- "src/components/settings-v2/system/execution-policy/index.test.tsx"` 覆盖 current config 读取、workspace sandbox 修改、canonical `bash` tool override 保存、shell/network rules 持久化保留；`npm test -- "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx"` 覆盖 policy/network 诊断卡恢复入口点击与无回调不显示假入口；`npm test -- "src/components/settings-v2/_layout/index.test.tsx" "src/components/settings-v2/hooks/useSettingsCategory.test.tsx" "src/components/agent/chat/utils/runtimeRoutingEvidence.test.ts" "src/components/agent/chat/utils/runtimePolicyEvidence.test.ts"` 覆盖设置页直达、系统导航与 routing/policy evidence helpers。`npm run typecheck` 本地长时间无输出，未计为通过。
- 验证证据：`npx vitest run "src/components/agent/chat/workspace/CodingWorkbenchOutputPanel.test.tsx" "src/components/agent/chat/workspace/codingWorkbenchRecovery.unit.test.ts" "src/components/agent/chat/workspace/useWorkspaceConversationSceneRuntime.test.ts" --silent=passed-only --disableConsoleIntercept` 覆盖 projection-driven output/action/recovery、submitted 状态与 hook callback 接线；`npx eslint ... --max-warnings 0` 覆盖 touched coding workbench 文件；`npx prettier --check ...` 与 `git diff --check -- ...` 覆盖 touched frontend/i18n/roadmap 文件；`npm run verify:gui-smoke` 覆盖 renderer build、Electron host typecheck/build、App Server sidecar、Claw workbench shell ready；`npm run smoke:code-artifact-workbench-electron-fixture` 覆盖代码 artifact 会话创建、历史打开、hydrate 与 workbench 打开。全量 `tsc --noEmit` 本地仍长时间无输出，本轮不计为通过。
- 2026-06-15 追加验证证据：`npm run typecheck -- --pretty false` 通过；`npm run build:renderer:electron` 刷新 Electron renderer `dist`；`npm run smoke:code-artifact-workbench-electron-fixture` 已覆盖真实 Electron Desktop Host + App Server JSON-RPC current 链路下的代码 artifact 会话创建、历史打开、hydrate、changes / outputs / logs 面板查看，以及点击“继续修复”后 `agentSession/turn/start` 的 `requestMetadata.harness.coding_workbench_recovery` 到达 external fixture backend。summary 证据位于 `.lime/qc/gui-evidence/code-artifact-workbench-electron-fixture/code-artifact-workbench-electron-fixture-summary.json`，关键断言为 `codingRecoveryGuiSubmitted=true`、`codingRecoveryReachedBackend=true`。
- 2026-06-15 GUI coding 输入增量：`scripts/electron/code-artifact-workbench-fixture-smoke.mjs --scenario gui-coding-input` 会先经真实 GUI 输入框发送 `@代码 ...` 编程请求，再等待同一 App Server current 会话生成标准 `artifact.snapshot / file.changed / patch.* / command.* / test.*` coding facts，并验证 Workbench changes / outputs / logs / recovery metadata；`npm run smoke:agent-runtime-current-fixture` 已把该场景纳入 current regression。该证据证明“提交编程需求、查看输出、继续修复”已进入非 live Provider、非 mock backend、非 renderer mock fallback 的日常门槛。
- 2026-06-15 network policy decision 增量：`runtimePolicyEvidence.ts` 已从 `thread_read.diagnostics / model_routing / latest_failed_command` 中的 `networkRule* / networkRisk* / networkHost / networkUrl / primary_blocking_*` 派生 `networkDecision`，区分明确阻断 `deny` 与需要策略复核 `ask`，并把解释输出到 Reliability policy card 与复制诊断文本；本轮只做 UI/evidence projection，不新增 App Server method、Desktop Host command 或 mock fallback。
- 2026-06-15 network policy decision 验证证据：`npx vitest run "src/components/agent/chat/utils/runtimePolicyEvidence.test.ts" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --silent=passed-only --disableConsoleIntercept`、`npx eslint "src/components/agent/chat/utils/runtimePolicyEvidence.ts" "src/components/agent/chat/utils/runtimePolicyEvidence.test.ts" "src/components/agent/chat/components/AgentThreadPolicyEvidenceCard.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --max-warnings 0`。
- 2026-06-15 evidence export coding summary 增量：`lime-rs/crates/app-server/src/runtime/evidence_provider.rs` 已把 coding facts 写入 `EvidencePackSummary.observability_summary.coding`，schema 为 `coding-evidence-summary.v1`，包含 file change、patch、failed patch、command、failed command、test、failed test、action required/resolved、recovery request 计数，以及 `outputRefs / diffRefs / checkpointRefs / artifactRefs / evidenceRefs / sourceEventIds`；恢复请求中嵌套的 `harness.coding_workbench_recovery` 也会参与 refs 聚合。
- 2026-06-15 evidence export coding summary 验证证据：`cargo fmt --manifest-path "lime-rs/Cargo.toml" -p app-server --check`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::evidence_exports -- --nocapture` 覆盖 snapshot artifact、失败 patch、失败 command、失败 test、action lifecycle、recovery context 与 Evidence Pack coding summary。
- 2026-06-15 network rule recovery 增量：`ExecutionPolicyNetworkFocusPanel` 与 `executionPolicyFocus.ts` 已让 Reliability policy/network 诊断传入的 `ExecutionPolicyFocusContext` 真正进入执行策略设置页；已有规则会高亮定位，缺失规则会展示候选 exact 网络规则，并可一键加入 `agent.tool_execution.network_rules`，继续通过现有保存按钮落 `save_config`，不新增 App Server method、Desktop Host command 或 mock fallback。
- 2026-06-15 network rule recovery 验证证据：`npx vitest run "src/components/settings-v2/system/execution-policy/index.test.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --silent=passed-only --disableConsoleIntercept`、`npx eslint "src/components/settings-v2/system/execution-policy/index.tsx" "src/components/settings-v2/system/execution-policy/executionPolicyFocus.ts" "src/components/settings-v2/system/execution-policy/ExecutionPolicyNetworkFocusPanel.tsx" "src/components/settings-v2/system/execution-policy/index.test.tsx" "src/components/agent/chat/components/AgentThreadPolicyEvidenceCard.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --max-warnings 0`、`npx prettier --check ...`、`git diff --check -- ...`。
- 2026-06-15 provider focus recovery 增量：`SettingsPageParams.providerFocus` 已从 `AppPageContent` 透传到 `SettingsLayoutV2` 与 `CloudProviderSettings`，`ApiKeyProviderSection` 已按 `providerId / modelId` 深链选中目标服务商并避免误落默认 provider；`ProviderSetting` 已在缺少 API Key 或模型时展示恢复提示，并提供一键加入模型优先级的动作。
- 2026-06-15 provider focus recovery 验证证据：`npx vitest run "src/components/AppPageContent.test.tsx" "src/components/settings-v2/_layout/index.test.tsx" "src/components/api-key-provider/ApiKeyProviderSection.ui.test.tsx" "src/components/api-key-provider/ProviderSetting.ui.test.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --silent=passed-only --disableConsoleIntercept`、`npx eslint "src/components/AppPageContent.tsx" "src/components/AppPageContent.test.tsx" "src/components/settings-v2/_layout/index.tsx" "src/components/settings-v2/_layout/index.test.tsx" "src/components/settings-v2/agent/providers/index.tsx" "src/components/api-key-provider/ApiKeyProviderSection.tsx" "src/components/api-key-provider/ApiKeyProviderSection.ui.test.tsx" "src/components/api-key-provider/ProviderSetting.tsx" "src/components/api-key-provider/ProviderSetting.ui.test.tsx" "src/types/page.ts" --max-warnings 0`、`npx prettier --check ...`。
- 2026-06-15 policy source layer visibility 增量：`ExecutionPolicySettings` 已在右侧摘要中展示策略来源层级，明确 default / current config / organization / user / runtime / request 的合并顺序、优先级和可编辑边界；本页只保存 current config 层，organization / user / runtime / request 仍按 App Server request metadata current 链路进入运行诊断，不在设置页伪造本地写入。
- 2026-06-15 policy source layer visibility 验证证据：`npx vitest run "src/components/settings-v2/system/execution-policy/index.test.tsx" --silent=passed-only --disableConsoleIntercept`、`npx eslint "src/components/settings-v2/system/execution-policy/index.tsx" "src/components/settings-v2/system/execution-policy/index.test.tsx" --max-warnings 0`、`npx prettier --check ...`。
- 2026-06-15 policy field source reliability 增量：`runtimePolicyEvidence.ts` 已从 current `thread_read.diagnostics / latest_failed_command / latest_failed_tool / model_routing / runtime_summary` 解析 `warningPolicySource / restrictionProfileSource / sandboxPolicySource`，Reliability policy card 展示 `warning=user · restriction=runtime · sandbox=request` 这类字段级来源，复制诊断文本同步输出同一行；本轮只消费已有 App Server metadata，不新增 App Server method、Desktop Host command 或 mock fallback。
- 2026-06-15 policy field source reliability 验证证据：`npx vitest run "src/components/agent/chat/utils/runtimePolicyEvidence.test.ts" "src/components/agent/chat/utils/runtimeRoutingEvidence.test.ts" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --silent=passed-only --disableConsoleIntercept`、`npx eslint "src/components/agent/chat/utils/runtimePolicyEvidence.ts" "src/components/agent/chat/utils/runtimePolicyEvidence.test.ts" "src/components/agent/chat/utils/runtimeRoutingEvidence.ts" "src/components/agent/chat/utils/runtimeRoutingEvidence.test.ts" "src/components/agent/chat/components/AgentThreadPolicyEvidenceCard.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --max-warnings 0`、`npx prettier --check ...`、`git diff --check -- ...`。
- 2026-06-15 sandbox backend diagnostics 增量：`runtimePolicyEvidence.ts` 已从 current policy metadata 解析 `sandboxBackend / sandboxBackendStatus / sandboxBackendEnforced / sandboxBackendRequired / sandboxBackendReasonCode / sandboxBackendReason / sandboxBackendPlatform / workspaceSandboxConfigSource`，Reliability policy card 和复制诊断文本会展示 `backend=restricted_token · status=unavailable · enforced=false · required=true · platform=windows · source=request · reason=...` 这类后端事实；本轮只消费已有 App Server metadata，不把 Windows runner 伪装成已 enforce。
- 2026-06-15 sandbox backend diagnostics 验证证据：`npx vitest run "src/components/agent/chat/utils/runtimePolicyEvidence.test.ts" "src/components/agent/chat/utils/runtimeRoutingEvidence.test.ts" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --silent=passed-only --disableConsoleIntercept`、`npx eslint "src/components/agent/chat/utils/runtimePolicyEvidence.ts" "src/components/agent/chat/utils/runtimePolicyEvidence.test.ts" "src/components/agent/chat/utils/runtimeRoutingEvidence.ts" "src/components/agent/chat/utils/runtimeRoutingEvidence.test.ts" "src/components/agent/chat/components/AgentThreadPolicyEvidenceCard.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --max-warnings 0`、`npx prettier --check ...`、`git diff --check -- ...`。
- 2026-08-10 current 代码复核更正：以下 2026-06-15 Windows runner、hardening、pipe reader 与 smoke 记录已被 current 仓库事实推翻，只保留为 superseded historical evidence，不得作为 current owner 或交付证据。当前 `tool-runtime::plan_sandbox_backend` 返回 `Planned/enforced=false` 和 `sandbox_backend_windows_runner_not_implemented`，`prepare_sandbox_command` 对 `RestrictedToken` fail closed；仓库中不存在这些记录声称的 Windows runner 实现。
- 2026-06-15 Windows restricted token executor boundary 增量：Agent sandbox executor 已新增 `RestrictedToken` 一等 sandbox type，tool orchestrator 会把 `sandboxBackend=restricted_token` 映射到 `ToolContext.workspace_sandbox`，非 Windows 平台执行层 fail-closed，不再静默回退为无沙箱执行；Windows runner 已落地真实 enforce 边界：`CreateRestrictedToken` 创建 write-restricted token，per-run capability SID 通过 workspace writable ACL 授权，`denied_paths` 追加 write deny，最终经 `CreateProcessAsUserW` 启动子进程并捕获 stdout/stderr。Windows 平台 plan 进入 `ready/enforced=true`，后续仍需 Windows 实机 smoke。
- 2026-06-15 Windows restricted token executor boundary 验证证据：`cargo test --manifest-path "lime-rs/crates/agent-rust/Cargo.toml" -p agent-core sandbox::executor -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution::sandbox_backend_tests -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_orchestrator::tests::workspace_sandbox_config_maps_restricted_token_backend -- --nocapture`、`npx vitest run "src/components/agent/chat/utils/runtimePolicyEvidence.test.ts" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --silent=passed-only --disableConsoleIntercept`。
- 2026-06-15 Windows restricted token minimal enforce 验证证据：`cargo check --manifest-path "lime-rs/Cargo.toml" -p agent-core`、`cargo test --manifest-path "lime-rs/crates/agent-rust/crates/agent/Cargo.toml" --target-dir "lime-rs/target" --lib sandbox::executor -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::execution::sandbox_backend_tests -- --nocapture`、`cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-agent agent_tools::tool_orchestrator::tests::workspace_sandbox_config_maps_restricted_token_backend -- --nocapture`。
- 2026-06-15 Windows restricted token hardening 增量：restricted token runner 已补 Job Object `KILL_ON_JOB_CLOSE`、timeout 时 `TerminateJobObject` 杀进程树、ACL rollback RAII 恢复 workspace / denied path DACL，以及 stdout/stderr 并发 pipe reader，避免大输出填满 pipe 后父进程仍阻塞在 `WaitForSingleObject`。
- 2026-06-15 P2-A 输出流控增量：`sandbox/output_buffer.rs` 已成为 Agent 有界输出捕获 owner；通用 sandbox executor 从 `Command::output()` 改为 spawn + 并发读取 stdout/stderr 到 head/tail buffer，Windows restricted token pipe reader 改为同一 buffer，embedded Bash 前台执行也改为同一有界捕获路径；`BashTool` metadata 继续输出 stdout/stderr 原始字节数、省略字节数、截断标记，以及 `outputBytes / outputOmittedBytes / outputTruncated`，交给 App Server output refs / evidence refs 消费。
- 2026-06-15 P2-A 验证证据：`cargo test --manifest-path "lime-rs/crates/agent-rust/crates/agent/Cargo.toml" --target-dir "lime-rs/target" --lib sandbox::output_buffer -- --nocapture`、`cargo test --manifest-path "lime-rs/crates/agent-rust/crates/agent/Cargo.toml" --target-dir "lime-rs/target" --lib sandbox::executor::tests::unsandboxed -- --nocapture`、`cargo test --manifest-path "lime-rs/crates/agent-rust/crates/agent/Cargo.toml" --target-dir "lime-rs/target" --lib tools::bash::tests::test_execute_large_output_reports_bounded_capture_metadata -- --nocapture`、`cargo test --manifest-path "lime-rs/crates/agent-rust/crates/agent/Cargo.toml" --target-dir "lime-rs/target" --lib tools::bash::tests::test_executor_result_metadata_preserves_output_truncation_stats -- --nocapture`。上述测试通过时仅有既有 `worktree_tools.rs` unused `SessionManager` warning。
- 2026-06-15 P2-B Workbench process control 增量：`src/lib/api/appServer.ts` 已补齐 App Server current `executionProcess/start|writeStdin|interrupt|terminate|status|drainOutput` client 方法，`src/lib/api/executionProcess.ts` 作为唯一前端领域网关返回 `response.result`，不走 `safeInvoke` 旧命令或 mock fallback；`CodingWorkbenchOutputPanel` 只对 `processId + running/starting` 的 `CodingWorkbenchView.commands` 渲染 icon controls，并由 `workspaceConversationCodingViews.tsx` 注入 interrupt / terminate / status / drain handlers。该增量只完成 UI 控制面接入，不代表 command/test 默认执行已切到 sandbox-aware process control owner；下一刀仍需迁默认执行 lifecycle、stdin 与 status hydrate。
- 2026-06-15 P2-B Workbench process control 验证证据：`npx vitest run "src/lib/api/executionProcess.test.ts" "src/components/agent/chat/workspace/CodingWorkbenchOutputPanel.test.tsx" "src/lib/api/appServer.test.ts" --silent=passed-only --disableConsoleIntercept` 覆盖前端网关不调用 `safeInvoke`、App Server JSON-RPC method 名、live process 控件渲染与 processId 传递。
- 2026-06-15 Windows restricted token smoke gate 增量：`restricted_token.rs` 已新增 Windows-only smoke，直接走 `execute_in_sandbox_with_options` current owner，覆盖 workspace write 允许、外部路径写入拒绝、denied path write deny、ACL DACL restore、large output drain、timeout exit code `124` 和 Job Object 杀子进程树。Windows 实机验证入口固定为 `cargo test --manifest-path "lime-rs/crates/agent-rust/crates/agent/Cargo.toml" --target-dir "lime-rs/target" --lib restricted_token -- --nocapture`。

## P5：多模型 Routing 与 Profile Slot

目标：让 coding profile 显式使用 Lime 多模型能力。

动作：

- 定义 base/coding/review/fast/local 槽位。
- Provider Store 支持 custom endpoint model 的 stable id、alias 和 capability tags。
- Runtime 记录 routing decision、fallback reason、provider readiness diagnostics。
- 前端只展示选择和诊断，不持有 key。
- slot resolver 输出 `model_slot.resolved` 或等价 diagnostics fact，供 evidence/replay join。
- 未配置或能力不足时，Runtime 以 `needs_setup / blocked` 结束或等待 action，不降级 mock。
- review / fast / local 只作为 task/subtask routing，不改变主 coding turn owner。

验收：

- 未配置 Provider 时返回 needs-setup / blocked。
- fallback 有 runtime diagnostics。
- 不存在产品页 local key 生产 fallback。
- review / fast / local 槽位不抢占 coding 主 turn owner。
- 自定义兼容端点作为 Provider Store entry 参与 registry，不写入 Coding Workbench 本地设置。
- UI diagnostics 可解释当前使用哪个槽位、为什么 fallback、下一步如何配置。

当前状态：`in_progress / App Server routing facts, Provider metadata join, read-model projection and frontend diagnostics connected`。

2026-06-14 增量：

- `lime-rs/crates/app-server/src/runtime_backend/model_routing.rs` 已成为 coding profile model slot diagnostics owner：解析 `harness.coding_model_slots / codingModelSlots / modelSlots` 中的 `base/coding/review/fast/local` 槽位；`coding` 槽位优先作为主 turn selection，`base` 作为 fallback，`review/fast/local` 只进入 diagnostics slots，不抢占主 coding turn owner。
- `RuntimeBackend` 的 turn start 已输出增强 `routing.decision.made` payload：包含 `routingDecision / routing_decision`、`modelSlot / model_slot`、`providerReadiness / provider_readiness`、`serviceModelSlot`、requested/selected provider/model、fallback chain、required coding capabilities。未 ready provider 会先写 `routing.not_possible`，再 fail-closed 返回 backend error，不降级 mock、不从 Workbench 持有 key。
- Provider readiness 先接入 App Server current Provider Store 事实：直传 fixture provider config 标记 `direct_provider_config`；已配置 Provider Store entry 记录 enabled/key count/provider type；非聊天 Provider、disabled provider、missing enabled key、未知 custom provider 都输出 `needs_setup / blocked` reason code。内置 runtime provider 仍按现有运行时兼容标记为 `builtin_runtime_provider`，后续 Provider Store 完整接管时再收紧。
- `runtime/read_model.rs` 已把最新 `routing.decision.made / routing.fallback.applied / routing.not_possible` 聚合到 `thread_read.model_routing`，并提升 `thread_read.service_model_slot` 与 `runtime_summary.decisionSource/serviceModelSlot`，让现有 `runtimeRoutingEvidence.ts` 与 Workbench diagnostics 能消费同一事实源。
- `lime-rs/crates/services/src/model_registry_service/runtime_metadata.rs` 已补 Provider model metadata resolver：优先匹配 10 天 Provider `/models` 缓存里的 stable id / provider model id / canonical id，其次匹配 Provider Store `custom_models` 声明并复用 model registry taxonomy/capability 推断；Runtime `routing.decision.made` 现在输出 `modelRegistry / model_registry`，包含 `source/status/reasonCode`、`modelCapabilities`、`modelAlias`、`reasoning`、cache count 和 declared model 标识。直传 fixture provider config 继续标记为 `direct_provider_config_not_in_registry`，不伪装成 registry fact。
- `runtime/read_model.rs` 已把 `modelRegistry / model_registry` 一并投影到 `thread_read.model_routing`，后续 UI diagnostics 可直接读取模型能力、别名来源、默认 reasoning effort 与 registry missing reason，不需要从 Workbench 本地设置补假数据。
- `src/components/agent/chat/utils/runtimeRoutingEvidence.ts` 与 `AgentThreadRoutingEvidenceCard` 已消费 `thread_read.model_routing.providerReadiness / provider_readiness` 和 `modelRegistry / model_registry`：ReliabilityPanel 的当前路由事实区会展示 provider readiness source/status/reason、provider type、key count、恢复动作、registry source/reason、capability tags、alias 与 reasoning support；复制诊断文本也会输出同一组 facts。provider readiness 存在恢复动作且 Workspace/Harness 提供 `onManageProviders` 时，诊断卡会复用现有 AI 服务商设置入口打开 `SettingsTabs.Providers`，无回调时不显示假按钮。新增文案已覆盖 `zh-CN / zh-TW / en-US / ja-JP / ko-KR`。

验证证据：

- `cargo fmt --manifest-path "lime-rs/Cargo.toml" -p app-server --check`
- `rustfmt --edition 2021 "lime-rs/crates/app-server/src/runtime_backend/model_registry_metadata.rs" "lime-rs/crates/app-server/src/runtime_backend/model_routing.rs" "lime-rs/crates/app-server/src/runtime_backend.rs" "lime-rs/crates/app-server/src/runtime/read_model.rs" "lime-rs/crates/app-server/src/runtime/tests/read_model.rs" "lime-rs/crates/services/src/model_registry_service.rs" "lime-rs/crates/services/src/model_registry_service/runtime_metadata.rs"`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend:: -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime_backend::model_registry_metadata -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p app-server runtime::tests::read_model::read_session_projects_model_routing_into_thread_read -- --nocapture`
- `cargo test --manifest-path "lime-rs/Cargo.toml" -p lime-services model_registry_service::runtime_metadata -- --nocapture`
- `npx vitest run "src/components/agent/chat/utils/runtimeRoutingEvidence.test.ts" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" --silent=passed-only --disableConsoleIntercept`
- `npx eslint "src/components/agent/chat/utils/runtimeRoutingEvidence.ts" "src/components/agent/chat/utils/runtimeRoutingEvidence.test.ts" "src/components/agent/chat/components/AgentThreadReliabilityPanel.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" "src/components/agent/chat/components/AgentThreadRoutingEvidenceCard.tsx" --max-warnings 0`
- `npx prettier --check "src/components/agent/chat/utils/runtimeRoutingEvidence.ts" "src/components/agent/chat/utils/runtimeRoutingEvidence.test.ts" "src/components/agent/chat/components/AgentThreadReliabilityPanel.tsx" "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx" "src/components/agent/chat/components/AgentThreadRoutingEvidenceCard.tsx" "src/i18n/resources/zh-CN/agentRuntime.json" "src/i18n/resources/en-US/agentRuntime.json" "src/i18n/resources/zh-TW/agentRuntime.json" "src/i18n/resources/ja-JP/agentRuntime.json" "src/i18n/resources/ko-KR/agentRuntime.json" "internal/roadmap/coding/implementation-plan.md"`
- `npm run smoke:agent-runtime-current-fixture`
- 2026-06-14 追加：同一组 vitest / eslint / prettier / `git diff --check` 与 `npm run smoke:agent-runtime-current-fixture` 再次通过，覆盖 provider readiness `needs_setup / missing_enabled_api_key` 的解析、面板渲染和复制文本。
- 2026-06-14 追加：provider readiness 恢复入口已从诊断 code 推进到可点击设置路径，`GeneralWorkbenchDialogSection` 与 `GeneralWorkbenchHarnessDialogSection` 均透传现有 `handleManageProviders / SettingsTabs.Providers` 目标；组件回归覆盖有回调时点击打开设置、无回调时不显示假入口。

剩余风险：

- Provider Store 的 stable model id、provider model id、canonical id、custom model capability 推断和 reasoning metadata 已进入 routing diagnostics；ProviderAliasConfig 的独立 alias cache 仍缺真实写链，当前 `modelAlias` 先来自 EnhancedModelMetadata 的 canonical/provider/alias_source 字段。
- UI diagnostics 已为 provider readiness 与 model registry facts 做基础分组展示；provider readiness 恢复动作已联动现有 AI 服务商设置入口，但尚未做到按具体 provider/key 深链到编辑态；network / policy 冲突解释已进入 Reliability 诊断信息面板与复制诊断文本，仍缺按具体策略 / 网络规则深链到编辑态。
- `model_slot.resolved` 目前以 `routing.decision.made` 内的 `modelSlot/model_slot` 等价 diagnostics fact 表达，后续如果 evidence join 需要独立事件，再补专门事件并纳入 P7 conformance。

## P6：External Harness Compat Adapter

目标：允许外部 CLI agent 作为兼容执行器接入，但不让它成为主链。

动作：

- 定义 `external_harness` event adapter。
- 解析 session start、prompt submit、permission request、tool complete、stop 等事件。
- 输出标准 RuntimeEvent 和 diagnostics refs。
- 所有 artifact/evidence 写入仍委托 Lime owner。
- 外部 harness 的 Provider、模型、权限、MCP 配置只进入 diagnostics，不覆盖 Lime current facts。
- 外部 harness 产物必须先进入 artifact/evidence owner，不能直接喂 UI component。
- 外部 harness 缺 scope id 时，adapter 产生 degraded diagnostics，不补假 id 伪造完整链路。

验收：

- 外部 harness 缺事件时 UI degraded。
- 外部 harness 不能直接写 Provider key、artifact truth、evidence verdict。
- 生产默认不依赖外部 CLI。
- App Server current coding turn 不需要 external harness 也能完成 file/patch/command/test 主闭环。

当前状态：`pending`。仅作为兼容，不是全量完成的前置事实源。

## P7：Conformance 与证据闭环

目标：coding profile 可被机械验证。

Fixture 矩阵：

| Fixture                   | 覆盖                                    |
| ------------------------- | --------------------------------------- |
| `coding-text-basic`       | 文本 turn、delta/final reconciliation。 |
| `coding-file-change`      | 文件写入、checkpoint、diff。            |
| `coding-patch-failure`    | patch failed、recovery hint。           |
| `coding-command-approval` | action required/resolved。              |
| `coding-sandbox-blocked`  | sandbox blocked、UI blocked state。     |
| `coding-test-failure-fix` | 测试失败、继续修复 turn。               |
| `coding-hydration-repair` | sequence gap、read model repair。       |

验证矩阵：

| 验证层       | 必跑条件                                       | 命令 / 入口                                                                                       | 证明内容                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------ | ---------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Contract     | 改 RuntimeEvent、fixture、schema、sequence     | `npm --prefix packages/agent-ui-contracts run test` 或 `npm run test:contracts`                   | event shape 和配对规则。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Projection   | 改 selector、fixture replay、UI adapter        | `npm --prefix packages/agent-runtime-projection run test`                                         | `CodingWorkbenchView` 可 replay。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Workspace VM | 改 Workspace adapter / view model              | 定向 vitest 覆盖 workspace view model / runtime tests                                             | Coding Workbench 不再恢复旧 thread item adapter；历史 hydrate 必须落在 current read model / projection 边界。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Rust runtime | 改 App Server / RuntimeCore / ExecutionBackend | `cargo test --manifest-path "lime-rs/Cargo.toml" -p <crate> <filter>`                             | current crates 能产生同构 events。P2 tool mirror 入口至少覆盖 `runtime_backend::coding_events`、`runtime::tool_lifecycle`、`agent_ui_event_schema`、`coding_events`、`evidence_exports`、`read_model`、`runtime_agent_tool_events_are_mirrored_to_coding_facts`、`runtime_agent_read_tool_result_is_mirrored_to_file_read`、`runtime_agent_shell_apply_patch_is_mirrored_to_patch_lifecycle`、`runtime_agent_permission_denied_fact_precedes_tool_failed_terminal`、`append_external_runtime_events_infers_action_resolved_tool_from_pending_action`、`read_model_projects_active_coding_activity_and_pending_action`、`read_model_clears_resolved_coding_activity`、`start_turn_hydrates_persisted_coding_snapshot_refs_into_runtime_state`、`start_turn_hydrates_persisted_active_coding_read_model_facts`、`export_evidence_pack_includes_coding_snapshot_artifacts`。 |
| GUI smoke    | 改 Coding Workbench 可见主路径                 | `npm run smoke:agent-runtime-current-fixture`、`npm run verify:gui-smoke` 或 coding fixture smoke | GUI 能提交编程需求、查看变更、输出、审批、失败继续；current 聚合门槛必须实际运行 Coding Workbench Electron fixture 的 `gui-coding-input` 场景，不能只跑脚本文本守卫。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Governance   | 改旧入口分类 / mock / bridge                   | `npm run governance:legacy-report` + `npm run test:contracts`                                     | 旧路未回流，生产不 mock。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

建议验证：

```bash
npm run test:contracts
npm run governance:legacy-report
npm exec vitest run "packages/agent-runtime-projection/tests/*.test.mjs"
npm exec vitest run "packages/agent-runtime-ui/tests/*.test.mjs"
npm run verify:gui-smoke -- --reuse-running --timeout-ms 600000
cargo test --manifest-path "lime-rs/Cargo.toml"
```

按实际改动选择最小验证；触及 GUI 主路径时必须补 GUI smoke。

当前状态：`completed for P7 baseline / conformance fixture + projection replay + GUI smoke evidence 已成闭环 / gui coding input aggregation tightened / evidence export coding summary covered`。`coding-file-change`、`coding-command-approval`、`coding-sandbox-blocked`、`coding-patch-failure`、`coding-test-failure-fix`、`coding-hydration-repair` 已在 `agent-ui-contracts` 与 `agent-runtime-projection` 机械验收；`expected.coding` 已成为 fixture 约束的一部分，`verifyRuntimeEventSequence` 的 lifecycle 违规也已被 conformance 测试显式拦截。`smoke:code-artifact-workbench-electron-fixture -- --scenario gui-coding-input` 已通过真实 Electron Desktop Host + GUI 输入框 + App Server JSON-RPC + external fixture backend 提交编程请求，创建包含 `file.changed / patch.* / command.* / test.*` 标准 coding facts 的会话，并在 GUI 工作台验证 changes / outputs / logs 三个面板和继续修复 metadata 都能显示对应证据；会话打开通过 current Task Center open-task 事件，不依赖 renderer mock 或 legacy 命令。`smoke:agent-runtime-current-fixture` 已把该真实 Electron fixture 场景纳入聚合 regression，防止 current 主路径只停留在脚本守卫或绕过用户输入。`runtime::tests::evidence_exports` 已覆盖 App Server current Evidence Pack 的 coding 聚合，防止 Workbench 可见事实无法导出复盘。后续 P7 增强只补 provider / policy / network 单条规则定位，不再影响 baseline 闭环完成判定。

## P8：旧实现清理与守卫

目标：全量完成后不保留第二套 coding 事实源。

动作：

- `code_orchestrated` 只保留为 profile 选择或迁移入口，不再拥有独立状态机。
- Workspace / Harness thread item 只允许作为通用消息列表 / 历史 fixture 残留，不能直接驱动 Coding Workbench。
- 删除从正文解析文件名、测试结果、补丁结果的路径。
- 合约守卫禁止生产 mock fallback、legacy desktop facade 和旧 Tauri wrapper 回流。
- 文档和 GUI smoke 明确 current 路径。
- 旧 UI adapter 已删除；历史 thread item hydrate 不得恢复成 production adapter，必须通过 current read model / projection 边界承接。
- 清理完成后，`code_orchestrated` 相关测试只证明迁移入口或 legacy guard，不再作为 current 可交付证据。
- 对 `mockPriorityCommands`、`defaultMocks`、legacy command catalog 做负向守卫，禁止新增 coding 主链 mock fallback。

验收：

- `rg "code_orchestrated"` 只剩兼容、测试或迁移说明。
- Coding Workbench 可只用 RuntimeEvent / ReadModel hydrate。
- `npm run test:contracts` 和 GUI smoke 证明 current 主链可用。
- `rg "file_artifact|command_execution|approval_request"` 在 Workspace coding 主路径中只出现在 adapter、测试或历史 hydrate 说明。
- 新增 coding 能力没有进入 `lime-rs/src/**`、legacy desktop facade 或产品页本地 Provider key。

当前状态：`completed for skeleton / P7 baseline 后防回流守卫已封口 / session overview projection-only / production mock fallback fail-closed`。

2026-06-14 P8 增量：

- `legacySurfaceCatalog` 已新增 `agent-chat-coding-workbench-legacy-thread-item-facts` 守卫：Coding Workbench current presentation/runtime 文件不得直接消费 `file_artifact / command_execution / approval_request / code_orchestrated` legacy fact；这些旧形状只能留在历史测试/fixture 或已删除适配器证据中，不得恢复为生产适配器。
- `workspaceConversationWorkbenchViewModel` 进一步收缩为 current facade：旧 thread item 适配路径已删除，current 文件不再承接旧形状转换逻辑；对应守卫改为精确 regex，避免 current 函数名误伤。
- 同步修正 `agent-chat-code-workbench-workspace-send-hardcode` 描述：自然语言编程底座当前收敛到 `react` Agent runtime 主链，legacy `code_orchestrated` 只能在兼容边界归一到 `react`，不得被文档误写成 current 主链。
- 新增 `agent-chat-code-orchestrated-current-entry-ban` 守卫，并把 `agentProtocol` / `runtimeInputCapabilityCatalog` 的归一函数命名收敛为 legacy compat helper，防止 current 发送 / 协议 / 运行时入口继续把 `code_orchestrated` 当现役 execution strategy。
- 新增 `agent-chat-coding-workbench-legacy-adapter-reexport` 守卫：`workspaceConversationSceneViewModel` / `workspaceConversationWorkbenchViewModel` 不得 re-export 已删除的 legacy thread item adapter；任何恢复都视为旧路回流。
- Agent Runtime current lifecycle 不可用时的生产报错已移除 `legacy agent_runtime_* commands` 恢复暗示，新增 `agent-runtime-current-unavailable-legacy-command-recovery-text` 守卫；App Server lifecycle 缺失继续 fail closed，但不把旧命令写成 current 旁路。
- `code_orchestrated / auto -> react` 历史策略归一已收进 TS `src/lib/api/agentRuntime/executionStrategyCompat.ts` 与 Rust `lime-rs/crates/agent/src/execution_strategy_compat.rs` 单一 compat 边界；`agentProtocol`、`runtimeInputCapabilityCatalog` 与 `session_store` 只调用 helper，不再各自维护本地归一逻辑。新增 `agent-runtime-execution-strategy-compat-helper-single-boundary` 守卫，防止 `code_orchestrated` 或旧 `normalizeLegacy*ExecutionStrategy` / `normalize_execution_strategy` 函数名回流到 current 调用点。
- 验证证据：`npx vitest run "src/lib/governance/legacySurfaceCatalog.test.ts" --silent=passed-only --disableConsoleIntercept` 与 `npm run governance:legacy-report` 均通过；legacy report 对新增 coding workbench guard 无边界违规，整体边界违规为 `0`。

2026-06-15 P8 增量：

- `CanvasSessionOverviewPanel` 已移除 `threadItems` 输入和旧 thread item fallback，session overview 只接受中性的 `activityItems`；coding 调用仍由 `codingSessionOverviewProjection.ts` 从 `CodingWorkbenchView` 派生活动项，面板本体不再直读 `file_artifact / command_execution / approval_request / code_orchestrated`。
- `agent-chat-coding-workbench-legacy-thread-item-facts` 守卫已覆盖 `CanvasSessionOverviewPanel.tsx`，并新增 catalog 测试锁定该 current 面板不得恢复旧 coding fact 类型。
- P8 residual 盘点结论：生产 `src / packages / electron` 主路径未发现 `agent_runtime_*` 直接命令调用；剩余 `agent_runtime_*` 字符串集中在 contract guard、negative smoke、retired command 扫描、test-only fixture 或明确 compat 文档中。`scripts/check-app-server-client-contract.mjs` 已守住 active markdown 不得把 retired `agent_runtime_*` 描述成 current，也已禁止脚本通过 `safeInvoke/invoke/postInvoke` 等方式调用 retired `agent_runtime_*` 命令；`scripts/check-command-contracts.mjs` 继续把 retired 命令族当负向 catalog 守卫。后续只清真实 production truth / recovery text / mock fallback，不再把 guard/test-only 字符串当主线 blocker。
- README 口径已同步：P1/P2/P3/P4/P5/P7/P8 骨架闭环具备 current facts、projection、GUI smoke、evidence export 与生产 mock / legacy command 防回流守卫；后续最高价值下一刀先在 `tool-runtime` 建立 Windows restricted-token runner 基线，再补 Windows 平台 enforce evidence；或按 P6 接外部 harness 的 RuntimeEvent adapter，不再重复补已完成的 P5/P7/P8 baseline。
- `legacySurfaceCatalog` 中已删除 parser 的说明不再把 `code_orchestrated runtime` 写成 current 编程底座；统一改为 `react Agent runtime` current 主链，legacy `code_orchestrated` 只能在 compat 边界归一到 `react`。`lime-core` 旧会话类型注释也同步标明 `code_orchestrated / auto` 仅为历史输入，避免 Rust 类型注释被后续实现误读成可继续演进的 profile。
- 验证证据：`npx vitest run "src/components/agent/chat/components/CanvasSessionOverviewPanel.test.tsx" "src/lib/governance/legacySurfaceCatalog.test.ts" --silent=passed-only --disableConsoleIntercept` 与 `npx eslint "src/components/agent/chat/components/CanvasSessionOverviewPanel.tsx" "src/components/agent/chat/components/CanvasSessionOverviewPanel.test.tsx" "src/lib/governance/legacySurfaceCatalog.test.ts" --max-warnings 0` 通过；`rg` 确认 `CanvasSessionOverviewPanel.tsx` 与对应测试不再包含旧 coding fact 字符串；`npm run governance:legacy-report` 边界违规为 `0`。

2026-06-15 P8 收口验收：

- `test:contracts` 已通过，`check-app-server-client-contract` / `check-command-contracts` / harness / modality / scripts / Electron release / docs boundary 全部通过；输出显示 `mock priority commands: 0`，证明生产路径没有重新依赖 `mockPriorityCommands`。
- `legacySurfaceCatalog.test.ts` 184 项通过；`governance:legacy-report` 边界违规为 `0`。既有分类漂移候选仍限于历史治理项，不构成 coding P8 blocker。
- App Server evidence export 守卫已跟随 runtime 拆分后的 current owner：`runtime/exports.rs` 继续从 current session events 与 `artifact_projection::stored_artifact_summaries_for_turn(&stored, params.turn_id.as_deref())` 构建 Evidence Pack，不回退旧 runtime facade。
- `session-history-fixture-smoke` 已补真实 GUI 归档 / 恢复链路：侧栏会话菜单点击 `app-sidebar-conversation-menu-archive` 必须产生 `agentSession/update archived=true`；已归档对话设置页点击 `settings-archived-conversation-restore` 必须产生 `agentSession/update archived=false`；两者都经真实 Electron Desktop Host `app_server_handle_json_lines` / App Server JSON-RPC trace 验证，不使用 App Server mock backend、renderer mock fallback 或 legacy `agent_runtime_*` 命令。

P8 骨架阶段剩余口径：旧 thread item 在 Coding Workbench current UI 中已由守卫封口，只允许历史消息列表、通用会话历史展示或测试夹具继续保留。`code_orchestrated` 只保留为单一 compat helper 输入，`agent_runtime_*` 只保留为 retired guard / contract negative test / test-only fixture / 明确 compat 文档。后续不再把这些 guard/test-only 字符串当主线 blocker；新增 coding fact / command / GUI 主路径时再同步扩守卫。

## 风险

| 风险                                       | 缓解                                                                  |
| ------------------------------------------ | --------------------------------------------------------------------- |
| 复制外部 runtime 后形成第二套事实源        | 先落 RuntimeEvent / ReadModel adapter，禁止 UI 直连。                 |
| 多模型能力被单 Provider 假设污染           | Provider slot 和 routing decision 作为 profile facts。                |
| UI 继续从正文猜状态                        | Projection tests 和 conformance fixture fail closed。                 |
| 大输出拖慢流式 UI                          | output spill refs + timeline 摘要。                                   |
| 外部 CLI adapter 变成主链                  | 标记 compat，生产默认不依赖。                                         |
| 文档完成被误当实现完成                     | 每阶段必须写 owner、测试入口和退出条件；最终汇报区分本轮/整体完成度。 |
| 只迁 change view，输出/审批/诊断仍走旧状态 | P4 把五个 view 全部列为退出条件。                                     |
| UI 技术词外露                              | 普通主流程使用业务动作词，技术状态只进诊断抽屉。                      |

## 当前实施入口

P1/P2/P3/P4/P5/P7/P8 骨架已经进入 App Server / RuntimeCore current crates 与 Workbench current projection：真实 runtime backend tool stream 会派生 `file.read/file.changed/patch/command/test/permission/sandbox/action` 基础 coding events，并经 schema / sequence / tool lifecycle guard 入库；`patch-apply` 已具备真实 workdir apply service，并已接入 agent `apply_patch` current tool 入口；tool output、file checkpoint previous content、历史 current timeline hydrate、active command/test/action read model、Basic Evidence Pack snapshot artifacts、`observability_summary.coding`、policy/sandbox/network 基础 diagnostics、network rule recovery draft、provider/model routing diagnostics、sandbox 基础 owner、process lifecycle owner、App Server `executionProcess/*` 控制 API、no-sandbox shell live process、live process registry seam、RuntimeBackend 共享 `ExecutionProcessServer` hook、Workbench stdin 写入入口、RuntimeBackend host process control 合同、inventory 配置来源测试、GUI coding input smoke、session overview projection-only 守卫、production mock fallback fail-closed 与 legacy command 防回流契约都已具备 current facts。后续实施不要回到纯文档、旧 UI 修补或旧命令 inventory；下一刀回到默认 command/test sandbox-aware process runner/control owner，外部 Agent host 消费合同等真实生产 adapter 出现后再做。

下次恢复入口固定为 P2-B live process lifecycle fifth slice：先读 `internal/roadmap/coding/README.md` 的“下次从这里开始”，再从 `lime-rs/crates/agent/src/agent_tools/tool_orchestrator.rs`、`lime-rs/crates/agent/src/agent_tools/execution/process.rs`、`lime-rs/crates/app-server/src/runtime_backend/live_execution_process.rs`、`lime-rs/crates/app-server/src/runtime_factory.rs`、`lime-rs/crates/app-server/src/runtime.rs` 和 `lime-rs/crates/app-server/src/execution_process.rs` 开始。第一目标不是新增 UI 或文档，也不是伪造外部 Agent host，而是补默认 command/test sandbox-aware process runner/control owner；`RuntimeBackendSubmitRequest.process_control` 合同已就绪，只有真实生产 host adapter 出现时才继续消费。同时继续通过 `ToolExecutionDecision.requires_sandboxed_execution()` 防止 sandbox required 命令走裸进程。
