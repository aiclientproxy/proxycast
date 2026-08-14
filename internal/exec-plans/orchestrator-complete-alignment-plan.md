# Codex Orchestrator Complete Alignment Plan

> Status: active; major implementation aligned, strict budget prompt parity and dedicated product evidence remain
> Owner: root
> Started: 2026-08-13
> Upstream baseline: `/Users/coso/Documents/dev/rust/codex` at `95aada11c4150e4ba28d6279c50f0995c1d93e5a`
> Current phase: Phase E aggregate gates and pinned-baseline re-audit complete; sandbox and managed-network retry Gate B proofs are closed, with one prompt-semantic gap and one product proof gap remaining

## 1. Objective

Fully align Lime with the three distinct Codex surfaces that use orchestrator semantics:

1. Tool execution orchestration: approval, sandbox selection, managed-network approval, first
   attempt, classified denial, approval-aware escalation, retry, cancellation and telemetry.
2. Multi-Agent V2 control: root-thread-scoped identity/control, six collaboration tools, execution
   capacity, resident-agent capacity, history fork, mailbox delivery, restart recovery and shared
   rollout budget.
3. Orchestrator-owned Skills/MCP: configuration gates, remote resource discovery, bounded reads,
   source authority, model exposure and exact server access policy.

The target is behavioral alignment in Lime's product architecture, not copying Codex names or
creating a second backend. The only product chain remains:

```text
Electron Desktop Host
  -> App Server JSON-RPC
  -> RuntimeCore / agent-runtime / tool-runtime / skills / MCP owner
  -> canonical Thread / Turn / Item projection
  -> GUI
```

## 2. Scope And Write Set

Current Phase D/E write set:

- `internal/{aiprompts/architecture.md,exec-plans/orchestrator-complete-alignment-plan.md}`
- `lime-rs/crates/app-server/src/runtime/{agent_execution,agent_residency,rollout_budget}.rs`
- App Server AgentControl gateway, turn execution, event sink, provider history, runtime factory and
  focused tests required by those owners
- `lime-rs/crates/{agent-runtime,agent,model-provider,runtime-core}/**` only for the typed provider
  usage/reminder propagation contract
- Rust and TypeScript config types for `agent.rollout_budget`; no new settings UI

Later phases may claim these paths only after rechecking the worktree:

- `lime-rs/crates/tool-runtime/src/**`
- `lime-rs/crates/agent/src/current_provider_turn/**`
- `lime-rs/crates/skills/src/**`
- `lime-rs/crates/app-server/src/runtime/{skills,mcp}.rs`
- protocol/config/schema/client files only if the exact Codex contract requires a public setting
- `internal/aiprompts/architecture.md`
- `internal/aiprompts/commands.md`
- focused governance tests and GUI evidence assets

Current exclusions:

- Scheduled Tasks files and their active dirty worktree.
- Legacy Tauri command surfaces.
- Restoring `lime-rs/crates/agent/src/agent_tools/tool_orchestrator.rs`; that path stays
  `dead / deleted / forbidden-to-restore`.
- Production mock fallback.

## 3. Alignment Ledger

| Capability | Codex authority | Lime current owner | Current evidence | Exit condition |
| --- | --- | --- | --- | --- |
| Tool approval and sandbox orchestration | `core/src/tools/orchestrator.rs` | `tool-runtime` + current provider turn | Satisfied: `RuntimeToolExecutionAttempt` owns approval, sandbox, grants, cancellation and typed denial; `.lime/qc/gui-evidence/orchestrator-sandbox-retry-gate-b.json` proves a real Electron/runtime `workspace-write` denial -> typed file-change approval -> same provider identity retry -> completed visible file-change group | Keep the dedicated sandbox retry Gate B and owner tests green |
| Managed-network denial approval | Tool orchestrator + network approval | `tool-runtime` + App Server execution process | Satisfied: host-aware typed denial is independently approved and retried with the same identity; `.lime/qc/gui-evidence/orchestrator-managed-network-retry-gate-b.json` proves real Electron/runtime `managed_network_denied` -> Codex-exact `networkApprovalContext` approval -> same-item retry while preserving `workspace-write` -> one endpoint hit -> completed visible `exec_command` row | Keep the dedicated managed-network retry Gate B and owner/App Server tests green |
| Multi-Agent six-tool V2 surface | `tools/handlers/multi_agents_v2/**` | `tool-runtime::agent_control` + App Server gateway | Satisfied: `.lime/qc/agent-runtime-tool-execution-smoke.json` is a real Electron/current-chain cold-restart Gate B with all six tools and `39/39` assertions | Keep exact schemas/semantics and the six-tool Gate B green |
| Root-tree execution capacity | `agent/control/execution.rs` | App Server RuntimeCore | `current`: three child slots per root, atomic reserve/claim/release, root isolation and terminal reuse; fresh related owner/runtime tests pass inside the `52/52` AgentControl group; `.lime/qc/gui-evidence/orchestrator-agent-capacity-gate-b.json` is a real Electron/runtime Gate B with four parallel `spawn_agent` calls, three admitted children and one visible failed fourth row carrying canonical capacity denial | Keep the capacity Gate B green and add terminal-slot-reuse evidence |
| V2 resident-agent capacity | `agent/control/residency.rs` | App Server RuntimeCore + session loop | `current`: three resident children per root, deterministic idle LRU eviction, exact cold reload and interrupted/lost tombstones; fresh related owner/runtime tests pass inside the `52/52` AgentControl group | Add a real Electron/runtime idle-LRU eviction and exact-target cold-reload Gate B |
| Rollout budget | `rollout_budget.rs` + AgentControl | agent-runtime/App Server | Core accounting is `current`: typed config, root-shared attempt deltas, provider units, canonical reminder persistence, exhaustion/cancellation, restart and window semantics; fresh owner tests pass `8/8`. Strict Codex prompt parity still differs before the first threshold: Codex emits the current remainder on the initial request, while Lime intentionally suppresses that reminder until a threshold is crossed (`rollout_budget.rs` test). | Decide and test initial-remainder prompt parity, then add real Electron/runtime shared exhaustion, cancellation and restart-rejection evidence |
| Orchestrator Skills source | `ext/skills/src/provider/orchestrator.rs` | `skills` + App Server | `current`: bounded paginated discovery/read through session MCP resource owner; `ORCHESTRATOR-01` proves one frozen discovery, exact resource read, provider history and GUI final state | Satisfied; keep the dedicated Gate B and owner tests green |
| Orchestrator Skills config | `Config.orchestrator_skills_enabled` | App Server config owner | `current`: `orchestrator.skills.enabled`, default=true, load failure fail closed, typed Rust/TS config; current config control plane read/write proved by `ORCHESTRATOR-01` | Satisfied; keep config schema/control-plane tests and Gate B green |
| Orchestrator MCP config/access | `Config.orchestrator_mcp_enabled` | App Server config + MCP runtime | `current`: `orchestrator.mcp.enabled`; `ORCHESTRATOR-01` proves exact `codex_apps` definition/dispatch gate while ordinary MCP remains model-visible and executable | Satisfied; keep disabled-boundary Gate B and dispatch tests green |
| GUI/read-model proof | Codex public Thread items + client surfaces | ThreadStore + Renderer | Six-tool AgentControl cold restart, execution-capacity rejection, sandbox retry, managed-network retry and `ORCHESTRATOR-01` have direct current-chain Electron/read-model/GUI evidence; all use deterministic local providers and zero production mock/legacy fallback | Keep those proofs green and add terminal reuse/residency plus rollout-budget evidence |

No row can be marked complete from a code search alone. Each row needs focused tests and the
appropriate product evidence.

## 4. Implementation Order

### Phase A: Multi-Agent execution capacity

1. Add one root-thread-scoped limiter owned by RuntimeCore.
2. Resolve the durable root identity before a child TriggerTurn starts.
3. Count the root turn as one occupied slot and allow at most three concurrent child turns under
   the Codex default total capacity of four.
4. Reserve atomically before starting a previously idle child turn; already-active target messages
   do not consume a second slot.
5. Hold the reservation for the full admitted child task and release on completed/failed/canceled,
   rollback and cancellation paths.
6. Reject with stable `agent_limit_reached` evidence; do not silently queue beyond the concurrency
   contract.

### Phase B: Resident capacity and rollout budget

1. Separate durable thread existence from loaded session-loop residency.
2. Track V2 resident children per durable root; evict only the oldest idle candidate and protect
   the target being resumed.
3. Preserve interrupted/lost semantics exactly where Codex cannot resume an evicted live actor.
4. Add a root-tree rollout budget owner shared by root and descendants; account provider token
   usage, emit bounded reminders and reject new work after exhaustion.
5. Persist only the budget facts needed for restart correctness; do not create a second transcript.

### Phase C: Tool execution orchestrator

1. Introduce one transport-neutral execution-attempt contract in `tool-runtime`.
2. Move the current one-shot decision/approval/handler sequence behind that contract.
3. Carry approval cache, Guardian strict review, requested sandbox, granted filesystem/network
   permissions and cancellation in one immutable attempt context.
4. Classify sandbox denial, managed-network denial, timeout, cancellation and ordinary handler
   failure without parsing user-visible text.
5. Apply Codex approval-policy semantics before an unsandboxed or permission-expanded retry.
6. Route shell, apply_patch and unified exec through the same orchestrator and retain canonical
   Tool lifecycle identity across attempts.

### Phase D: Orchestrator-owned Skills/MCP

1. Add typed App Server config settings for `orchestrator.skills.enabled` and
   `orchestrator.mcp.enabled`, defaulting to enabled like Codex.
2. Add `Orchestrator` skill authority/source without changing local workspace/user/application
   authorities.
3. Discover `mcp/skill` resources through the session-owned Apps MCP connection, with Codex bounds:
   10-second deadline, 10 pages, 100 visible skills, URI/name/content bounds and cursor-loop guard.
4. Read resources through the same session connection and preserve provider/resource identity.
5. Gate only the orchestrator-owned Apps MCP server when disabled; ordinary MCP remains available.
6. Project the catalog through the existing Skills read/model context path and add a real consumer.

### Phase E: Product proof and governance

1. Update `internal/aiprompts/architecture.md` and `internal/aiprompts/commands.md` in the same
   change set after active parallel edits are reconciled.
2. Keep the existing negative guard against restoring the deleted Agent tool orchestrator and add
   the Orchestrator Host boundary guard against Electron/DevBridge business logic.
3. Run focused Rust tests after every phase, then `npm run test:contracts`,
   `npm run governance:legacy-report`, `npm run smoke:agent-runtime-current-fixture` and
   `npm run verify:gui-smoke` as risk expands.
4. Produce dedicated Electron Gate B evidence for Multi-Agent capacity, orchestrated retry and
   Orchestrator-owned Skill discovery.
5. Re-run the ledger audit against the pinned Codex baseline and record every divergence explicitly.

## 5. Agent Verification Contract

```text
改动名称：Codex Orchestrator Complete Alignment
执行计划文件：internal/exec-plans/orchestrator-complete-alignment-plan.md
负责人：root
预算标签：budget:normal
风险等级：P0
影响模块：App Server RuntimeCore、agent-runtime、tool-runtime、Skills、MCP、ThreadStore、GUI projection
不做范围：旧 Tauri command、生产 mock fallback、恢复已删除 Agent runtime/tool orchestrator
```

Current 主链：

```text
前端入口：Agent chat / SubAgent roster / Skills catalog
前端网关：existing typed App Server clients; no new gateway unless a public config surface is required
Electron Desktop Host bridge：app_server_handle_json_lines only
App Server method：existing thread/turn/skill/mcp/config current methods
RuntimeCore / service owner：RuntimeCore + agent-runtime + tool-runtime + skills/MCP session owner
read model：ThreadStore canonical Thread/Turn/Item and Skills catalog
runtime event：typed Tool lifecycle, SubAgent activity/state, approval/sandbox facts
Evidence Pack 字段：tool identity/attempt outcome, child thread identity/state, skill source/resource identity
GUI surface：Agent timeline/roster and Skills catalog
```

Happy Path：

```text
用户输入 / Agent 输入：root turn delegates concurrent tasks; tool needs sandbox escalation; model reads remote skill
预期 runtime events：one canonical lifecycle per tool/turn, stable child state and bounded remote discovery
预期 tool calls：six V2 collaboration tools plus existing shell/apply_patch/unified exec and skill read
预期 approval / sandbox：approval precedes side effect; classified denial can request and perform one valid retry
预期 artifact：durable graph/mailbox/budget facts and bounded skill resource content
预期 evidence：focused Rust, contract, fixture and Electron Gate B summaries
预期 GUI 状态：capacity failure/retry/remote skill state is visible from current read models
失败时应停在哪一层：RuntimeCore/tool-runtime/skills owner; never Electron or Renderer fallback
```

Evidence layers:

| Layer | Required | Planned evidence |
| --- | --- | --- |
| deterministic-smoke | yes | focused crate tests + contracts + current fixture |
| gui-trace | yes | dedicated Electron Gate B |
| runtime-transcript | yes | canonical Thread/Turn/Item and tool-attempt evidence |
| release-artifact | no | required only when included in a release candidate |

Required commands:

```bash
npm run test:rust:related -- <changed Rust paths>
npm run test:contracts
npm run governance:legacy-report
npm run smoke:agent-runtime-current-fixture
npm run verify:gui-smoke
```

Completion standard:

- Every ledger row has direct current-state evidence.
- No `compat` execution path or production mock fallback carries the new behavior.
- Architecture documentation and Gate B evidence cover the final owner/data flow.
- A fresh comparison against the pinned Codex baseline finds no required Orchestrator capability gap.

## 6. Progress Log

### 2026-08-13

- Completed the initial three-surface audit and pinned the upstream baseline.
- Confirmed AgentControl V2 six-tool/durable graph/mailbox is `current` but root-tree execution
  capacity, residency and rollout budget remain incomplete.
- Confirmed tool policy/approval/sandbox pieces are `current` but Codex denial escalation/retry is
  not owned by one execution orchestrator.
- Confirmed Orchestrator-owned Skills/MCP is currently absent rather than aligned.
- Started Phase A with a narrow App Server runtime write set; Scheduled Tasks dirty files are excluded.
- Completed Phase A with a root-thread-scoped child execution limiter. The Codex default total
  capacity of four is represented as three child slots; root turns, ordinary sessions and active
  child steering do not consume additional slots.
- `spawn_agent` and idle `followup_task` now reserve before durable mutation and wait only through
  the canonical turn-admission boundary. The fourth child fails closed with
  `agent_limit_reached`; terminal task, failure, cancellation and rollback release by RAII.
- Added root isolation, atomic claim/release and product-path concurrency coverage, including
  three admitted children, fourth-child rejection before durable child creation and terminal slot
  reuse.
- Validation: `npm run test:rust:related -- <Phase A paths>` passed all App Server unit tests:
  `1677 passed; 0 failed`.

### 2026-08-14

- Completed the Phase B implementation with separate RuntimeCore owners for child execution slots,
  loaded child residency and the shared root rollout budget. Resident eviction is root-scoped LRU,
  protects active/pending children, restores rejected or failed-shutdown candidates, reloads
  completed children from the durable Thread and preserves canceled/interrupted eviction as lost.
- Added optional typed `agent.rollout_budget` config across Rust and TypeScript. Invalid configured
  budgets fail closed at Runtime factory construction; absence keeps the feature disabled.
- Preserved Responses `codex_rollout_budget_units` from wire lowering through canonical usage and
  provider events. Without provider units, accounting uses configured non-cached prefill and sampling
  weights. Snapshot deltas are isolated by root/thread/turn/route-attempt/attempt.
- Added root/child sharing, root isolation, provider reroute, exhaustion/admission/cancellation,
  restart hydration, compaction/rollback reminder windows and residency/lost-tombstone coverage.
- Canonical `rollout_budget.reminder` is appended before provider injection. `PreappendedById`
  retrieves and forwards that exact durable event, so a reroute may re-inject it without duplicate
  publication or persistence. The synchronous no-callback path now also uses the appending sink to
  preserve `turn.accepted`/`turn.started` sequence order.
- Provider history omits the reminder from its originating Turn and restores it as a developer
  message for future Turns. Empty-response retry reuses the same attempt snapshot and does not
  re-read or re-publish the reminder source.
- Validation passed: `cargo fmt --all`, `git diff --check`, `npm run typecheck`, the focused
  `model-provider` Responses usage test (`1 passed`) and `cargo check -p agent-runtime -p app-server
  --tests` using the installed generated V8 binding. The check has one unrelated existing dead-code
  warning for `article_workspace_snapshot_event_without_search`.
- At this Phase B checkpoint, formal App Server test execution was blocked before test linking
  because upstream
  `rusty_v8 v150.4.0` returns HTTP 404 for
  `librusty_v8_ptrcomp_sandbox_release_aarch64-apple-darwin.a.gz`. Generated-binding type checking
  is evidence of syntax/type correctness, not a substitute for executing those tests.
- Repository gates passed: `npm run governance:legacy-report` reported zero boundary violations,
  `npm run governance:scripts` passed the frozen baseline, `npm run test:contracts` passed protocol,
  client, command, harness and docs boundaries, and `npm run verify:gui-smoke` produced a passing
  real Electron/App Server evidence summary.
- At this Phase B checkpoint, `npm run smoke:agent-runtime-current-fixture` passed its frontend
  groups, the 101-test Electron fixture guard and multiple real Electron scenarios, then stopped at
  the existing Skills workspace
  step because the script still clicks `app-sidebar-nav-skills`. Current `AppSidebar` tests explicitly
  require that retired standalone Skills entry to be absent. Do not restore the dead UI; the fixture
  must be migrated to the current Plugin/Skills product entry before this aggregate command can pass.

- Completed Phase C tool execution orchestration. `tool-runtime` now owns the immutable attempt
  contract and one retry boundary; current provider turn owns only decision/projection and dispatch
  adapters. `require_escalated` fails closed under `never/unknown`, ordinary shell retries follow
  Codex policy, `apply_patch` preserves denied filesystem entries and can execute an approved
  danger-full-access retry, managed-network denial carries the host into scoped approval metadata,
  and Strict Guardian reviews both initial and escalated attempts. App Server consumes attempt
  permissions without a second approval decision. Verification: generated-binding cargo check for
  `tool-runtime`, `lime-agent`, `app-server`; 8 `tool-runtime` orchestrator tests passed with the
  locally cached V8 archive; App Server no-double-approval and macOS managed-network retry tests
  passed directly from the linked test binary. The normal upstream V8 sandbox archive remains a
  404, so full workspace test execution still requires the local archive workaround.

- Completed Phase D Orchestrator-owned Skills/MCP implementation. Added typed
  `orchestrator.skills` and `orchestrator.mcp` config with enabled defaults and load-error fail-closed
  behavior; introduced Orchestrator Skill source/authority/scope; and routed bounded `mcp/skill`
  discovery/read through the session-owned Apps MCP connection. Discovery is frozen once per turn
  and reused on reroute; `skill_search` consumes the snapshot and remote reads use exact
  `codex_apps`/`skill://` identity.
- Added protocol `nextCursor` and Orchestrator enum schema/generated-client updates. Deterministic
  verification passed: MCP page cursor, Apps gate, server filtering, config defaults, App Server
  snapshot merge, remote `skill_search`, schema fixtures, generated protocol types, and TypeScript
  typecheck. The linked test path uses the locally cached `rusty_v8 v150.4.0 ptrcomp_sandbox` archive;
  the upstream archive URL remains a 404.
- At this Phase D checkpoint, Phase E documentation and governance guard were updated. Its remaining
  exit condition was product
  evidence: dedicated Electron Gate B for remote discovery/read and disabled Apps boundary, plus
  final aggregate gates. The retired standalone Skills sidebar entry remains `dead / deleted /
  forbidden-to-restore`; the existing aggregate Agent fixture must migrate its fixture step instead
  of restoring that UI.
- Added `orchestrator` to the App Server current config control-plane root-key allowlist and executed
  `orchestrator_config_is_writable_through_current_control_plane` (`1 passed; 0 failed`). This closes
  the gap where typed config existed but `config/batchWrite` rejected the new root.
- Added dedicated scenario `ORCHESTRATOR-01-remote-skill-and-mcp-boundary` and passed it on real
  Electron 42.3.3/macOS arm64 with `backendMode=runtime`. The enabled turn performed
  `skill_search -> read_mcp_resource -> final`; the first provider request contained only remote
  Skill metadata, while the final request history contained the exact `SKILL.md` body. The
  session-owned Apps MCP process performed one frozen `resources/list` and one exact
  `resources/read`.
- The same Gate B wrote `orchestrator.mcp=false` through `config/batchWrite`, proved `codex_apps`
  absent from the next provider tool catalog, and executed the ordinary MCP tool once. Both turns
  retained stable Thread/Turn/User/Tool/final-text read-model facts and refreshed to a GUI-visible
  final state. Electron IPC, `app_server_handle_json_lines` and all required current methods were
  observed; console, page, invoke, mock-fallback and legacy-command counts were all zero.
- Gate B evidence is stored at
  `.lime/qc/gui-evidence/orchestrator-skills-gate-b/orchestrator-skills-gate-b-summary.json`. Its
  claim remains deterministic local provider/MCP behavior, not live provider or remote-network
  reliability. At this checkpoint, the remaining Phase E work was the aggregate current fixture
  migration/gates and a fresh ledger audit against the pinned Codex baseline.

### 2026-08-14 final pinned-baseline audit

- Re-audited the exact upstream tree at
  `/Users/coso/Documents/dev/rust/codex@95aada11c4150e4ba28d6279c50f0995c1d93e5a` rather than the
  moving working tree. The authority paths remain `core/src/tools/orchestrator.rs`,
  `core/src/agent/control/{execution,residency}.rs`, `core/src/rollout_budget.rs`,
  `core/src/tools/handlers/multi_agents_v2/**` and
  `ext/skills/src/provider/orchestrator.rs`.
- The implementation contract is aligned at the owner boundary. Lime keeps one transport-neutral
  tool attempt owner; one root-tree AgentControl with six V2 tools, separate execution/residency
  capacity and shared rollout budget; and one session-owned Orchestrator Skills/Apps MCP route.
  Electron and Renderer remain transport/projection consumers and do not host a second backend.
- The retired standalone Skills and Experts sidebar entries were not restored. The aggregate
  fixture now enters the unique current Plugin workspace and selects its Skills/Experts tabs.
  `PluginsPageParams.currentProjectId`, `AppSidebar` and `AppPageContent` preserve project scope
  across those tabs, so project-registered Skills remain visible from the current entry. Negative
  source guards reject both retired sidebar selectors.
- `npm run smoke:agent-runtime-current-fixture` now passes end to end with
  `liveProviderUsed=false`. This replaces the earlier recorded fixture blocker; deterministic
  Skills Runtime, Expert Plaza and Expert Panel Electron scenarios all use the current Plugin
  workspace path.
- Direct product evidence is complete for the six collaboration tools/cold restart,
  Orchestrator Skills/MCP enabled/disabled behavior, execution-capacity rejection, terminal-slot
  reuse/resident LRU cold reload, shared rollout-budget exhaustion and restart admission rejection,
  plus both sandbox and managed-network approval-aware retries. A dedicated shared-budget
  cancellation Gate B remains open; the existing current-chain cancel-then-continue fixture and
  RuntimeCore cancellation owner tests cover the transport/runtime cancellation contract but do
  not claim budget-specific cancellation accounting.
- Strict pinned-Codex budget prompt parity is also not yet complete. Codex's
  `pending_reminder` returns the current remainder at reminder index zero for a new request,
  while Lime's `pending_reminder` returns `None` until the first configured threshold is crossed.
  This is a model-visible behavior difference, separate from the already passing exhaustion and
  restart-admission Gate B.

#### Product-scope differences

- Lime maps Codex behavior into `Electron -> App Server JSON-RPC -> RuntimeCore -> canonical
  Thread/Turn/Item -> GUI`; it does not copy Codex's terminal renderer, TUI cells, slash-command
  navigation, key bindings, terminal status rows or terminal-specific approval presentation.
- Codex's internal Rust type names and TOML layout are authority references, not public Lime API
  requirements. Lime intentionally keeps short domain names and its existing typed App Server
  config/control plane while preserving default values, fail-closed behavior and runtime semantics.
- Local deterministic provider/MCP fixtures prove the product chain and identity contracts. They do
  not claim live provider availability, remote Apps MCP reliability or production network quality.

#### Final validation checkpoint

- Fresh Rust owner/runtime tests: AgentControl, execution capacity, residency and restart group
  `52/52`; rollout budget `8/8`; tool execution orchestrator `8/8`; provider-turn retry policy
  `1/1`. The only warning is the pre-existing unused test helper
  `article_workspace_snapshot_event_without_search`.
- Frontend and aggregate verification: `npm run typecheck`; affected component/fixture tests
  `126/126`; fixture source-contract `83/83`; full current Agent Runtime fixture passed with
  `liveProviderUsed=false`, including a fresh rerun after the managed-network changes.
- Repository gates: `npm run test:contracts` passed with `312` App Server client checks;
  `npm run governance:legacy-report` reported zero boundary violations and zero classification
  drift; `npm run governance:scripts` passed.
- Orchestrator Gate B:
  `.lime/qc/gui-evidence/orchestrator-skills-gate-b/orchestrator-skills-gate-b-summary.json` has
  `result=pass`, `backendMode=runtime`, stable canonical identity/GUI final text and zero
  console/page/invoke/mock/legacy errors.
- Final real Electron GUI smoke passed at
  `.lime/qc/project-gates/standalone-shell-01-20260814114459-36331/shell-01-electron-smoke/summary.json`
  with `result=pass`, `proofLevel=Gate B-F`, `21/21` assertions and all recorded error counts zero.
- Added and passed `.lime/qc/gui-evidence/orchestrator-agent-capacity-gate-b.json` with
  `backendMode=runtime` through the current managed Electron harness. The root provider returned
  one parallel batch of four `spawn_agent` calls; the canonical projection contains three admitted
  child identities and a fourth `failed` Tool row, while the runtime evidence contains one
  `agent_limit_reached` denial. The GUI snapshot proves the same rows/identities through
  Electron IPC and `thread/read`, with zero invoke/console errors. The fixture deliberately waits
  for child provider work, so the root completion gate does not claim terminal-slot reuse.
- Added and passed `.lime/qc/gui-evidence/orchestrator-sandbox-retry-gate-b.json` through the real
  managed Electron/runtime chain. The turn submits `approvalPolicy=on-request` and
  `sandboxPolicy=workspace-write`; `apply_patch` targets the workspace parent, receives one typed
  `item/fileChange/requestApproval`, and completes only after the captured outer JSON-RPC request is
  accepted. Runtime evidence retains one provider Tool identity and verifies the outside-workspace
  proof file. The restored GUI contains one visible, completed file-change group with one aggregated
  file row and the final assistant text; Electron IPC `thread/read`, invoke and console checks pass
  with zero errors. This is deterministic localhost-provider proof, not live-provider evidence.
- The sandbox GUI proof also exposed duplicate file rows from repeated lifecycle projections. The
  current timeline file-change owner now reuses `aggregateFileChangeSummaries`, so identical paths
  converge to one visible row; affected timeline and harness tests pass `42/42`.
- Added and passed `.lime/qc/gui-evidence/orchestrator-managed-network-retry-gate-b.json` through
  the real managed Electron/runtime chain. One canonical `exec_command` Item records two attempts;
  the first outcome is `managed_network_denied`, the final attempt is escalated, and the effective
  sandbox remains `workspace-write`. The App Server emits one
  `item/commandExecution/requestApproval` carrying
  `networkApprovalContext={host:"127.0.0.1",protocol:"http"}`, receives one response, retries the
  same Item identity and reaches the deterministic localhost endpoint exactly once. The restored
  GUI shows the completed command, endpoint proof and final assistant text with zero invoke or
  console errors; the screenshot is
  `.lime/qc/gui-evidence/orchestrator-managed-network-retry-gate-b-managed-network-retry-visible-dom.png`.
- Rechecked the exact Codex public wire at the pinned commit. `NetworkApprovalContext` contains only
  `host` and `protocol`, and `CommandExecutionRequestApprovalParams` carries it as the optional
  `networkApprovalContext`. Lime therefore does not add public `approvalPhase` or `denialKind`
  fields: typed network approval is proved by `networkApprovalContext`, while escalation and denial
  classification remain runtime evidence (`toolEscalated=true` and
  `firstAttemptOutcome=managed_network_denied`).
- Added and passed `.lime/qc/gui-evidence/orchestrator-agent-residency-gate-b.json` through the real
  managed Electron/runtime chain. Four child identities are visible, a terminal slot is reused,
  the oldest resident child is cold-reloaded from durable history, and the follow-up keeps the
  original child identity after Electron restart; all residency/read-model/GUI assertions pass
  with zero invoke/console errors. The screenshot is
  `.lime/qc/gui-evidence/orchestrator-agent-residency-gate-b-residency-visible-dom.png`.
- Added and passed `.lime/qc/gui-evidence/orchestrator-rollout-budget-gate-b.json` through the
  real managed Electron/runtime chain. Electron `config/read` confirms the isolated user-data
  config enables `limit_tokens=1`; the fixture usage exhausts the root-shared budget, the canonical
  turn is failed with `rollout budget is exhausted`, and post-restart `turn/start` is rejected
  with `data.reason=rollout_budget_exhausted` and `retryable=false`. The screenshot is
  `.lime/qc/gui-evidence/orchestrator-rollout-budget-gate-b-rollout-budget-visible-dom.png`.

#### Current classification and remaining exit conditions

- `current`: tool execution orchestrator; six-tool AgentControl; root-scoped execution/residency;
  shared rollout budget; Orchestrator Skills/MCP; Plugin workspace Skills/Experts entry; canonical
  Thread/Turn/Item and GUI projections.
- `compat`: none for the new Orchestrator behavior.
- `deprecated`: none inside this plan's owned product path.
- `dead / deleted / forbidden-to-restore`: retired Agent tool orchestrator, standalone Skills/Experts
  sidebar entries, legacy Team tools, Renderer/Electron business backend and production mock
  fallback.
- Overall implementation/evidence completion remains `98%`: aggregate gates are complete, and dedicated
  Electron/runtime evidence covers execution-capacity rejection, terminal-slot reuse/resident LRU
  cold reload, rollout-budget exhaustion/restart rejection, sandbox retry and managed-network
  denial/approval/same-identity retry. The plan stays active only for a budget-specific
  cancellation/accounting Gate B and strict initial-remainder prompt parity. The cancellation item
  is an evidence gap, while initial-remainder behavior is a small implementation decision; neither
  is a permission blocker. The local Rust rebuild is currently limited by the unavailable
  `rusty_v8` v150.4.0 prebuilt archive.
