## Lime v1.128.0

Simplified Chinese release notes are the primary version.

### New Features

- Delivered the main Codex Orchestrator runtime alignment: AgentControl now has root-scoped child execution capacity, resident capacity, and cold-reload paths.
- Added optional shared `agent.rollout_budget` accounting across a root tree, with reminders, exhaustion rejection, in-flight cancellation, and restart hydration.
- Added one tool execution orchestrator for shell, `apply_patch`, and unified exec approval, typed sandbox/network denial, one approval-aware escalation retry, cancellation, and attempt telemetry.
- Added Orchestrator-owned Skills/MCP discovery and exact content reads through the session-owned `codex_apps` connection, with config gates and fail-closed behavior.

### Fixes

- Fixed AgentControl capacity overflow, root isolation, terminal release, resident eviction, and recovery boundaries; excess work now fails with canonical `agent_limit_reached`.
- Fixed managed-network denial approval context propagation so an escalated retry preserves the same Tool identity and original sandbox permission.
- Fixed duplicate file-change lifecycle rows by aggregating identical paths into one visible GUI record.
- Fixed project scoping across the current Plugins Skills/Experts tabs so the aggregate Agent fixture no longer depends on the retired standalone Skills entry.

### Improvements and Refactoring

- Converged tool attempts, AgentControl capacity/residency, rollout budget, and Orchestrator Skills/MCP into their dedicated Rust owners while Electron/Renderer remain JSON-RPC transport and canonical projection consumers.
- Extended protocol schemas and generated clients for paginated MCP resources, network approval context, and Orchestrator Skill source/authority.
- Kept provider usage, rollout reminders, reroutes, and history restoration on the canonical EventLog/Thread/Turn/Item chain without a second transcript or budget state.

### Testing and Quality

- Focused Rust tests passed for AgentControl execution capacity/residency/restart, rollout budget, tool execution orchestration, and provider retry (`52/52`, `8/8`, `8/8`, `1/1`).
- Affected frontend and fixture tests passed `126/126`, fixture source contracts `83/83`; `npm run typecheck`, `npm run test:contracts` (`312` checks), `npm run governance:legacy-report`, and `npm run governance:scripts` passed.
- Current-bridge/read-model Gate B evidence is retained for Orchestrator Skills/MCP boundaries, Agent capacity rejection, and sandbox/managed-network escalation retries; the baseline GUI smoke passed `21/21` with zero mock, legacy, console, or invoke errors. The latest tool-execution aggregate rerun issue is recorded in the execution plan and is not claimed as fully green here.

### Documentation

- Updated the global architecture, command boundary, and Orchestrator alignment plan with owner/data-flow contracts, configuration and protocol boundaries, Gate B evidence, and remaining evidence items.
- Added stable Orchestrator Skills, Agent capacity, tool retry, and Electron Gate B smoke scripts and regression tests.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.128.0`.
- Complete GUI evidence for terminal-slot reuse/residency and rollout-budget exhaustion/cancellation/restart remains tracked in the execution plan; this release does not claim those evidence gaps are closed.

**Full changes**: `v1.127.0` -> `v1.128.0`
