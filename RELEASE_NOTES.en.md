## Lime v1.113.0

<sub>The Simplified Chinese release notes are the primary version. This English page is a companion for international readers.</sub>

### New Features

- Added typed Runtime World State projection for working directories, projects, models, permissions, collaboration, and effective multi-agent mode in provider context.
- Added credential-scoped provider health state, structured retry telemetry, and bounded fallback route selection before any user-visible output.
- Converged tool hook discovery, lifecycle decisions, and execution in `tool-runtime` with consistent pre- and post-call gates.
- Added the typed App Server v2 `model/verification` notification and schemas for standard trusted-access verification projection.

### Fixes

- Fixed the pointer hit-area gap that could close the Add Project submenu while moving into it.
- Fixed provider auth kinds, adapter readiness, capability upper bounds, and route execution drifting into invalid selection, fallback, or authentication headers.
- Fixed Chromium session data using the roaming directory on Windows and portable/E2E storage deriving AppDataRoot from AgentRoot.
- Fixed retryable provider failures being replayed after output, consumed steer input, or explicit direct-route admission.

### Improvements and Refactors

- Unified the storage-root composition chain for AppDataRoot, AgentRoot, HostSessionData, the product database, diagnostics, Soul, and MCP OAuth.
- Removed retired whole-database copying, generic migration manifests, startup cleanup, and managed-project path migration without compatibility fallbacks.
- Converged default-provider configuration, model routing, authentication projection, and capability admission on current owners; unsupported adapters now fail closed.
- Removed the legacy `lime-agent` HookManager and provider-migration fixture, with governance guards against restoring dead surfaces.

### Tests and Quality

- Expanded regression coverage for Runtime World State, multi-agent mode, hook lifecycle, provider health/retry/reroute, workspace scope, and App Server JSON-RPC.
- Added Electron storage-root, Windows sessionData, project submenu pointer continuity, and retired-migration negative guards.
- Synchronized App Server v2 schemas and generated TypeScript protocol types, and updated the Project Gate B scenario contract.

### Documentation

- Updated Agent Runtime, provider, storage alignment, architecture sources, implementation snapshots, and Codex alignment execution records.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and lockfile versions to `1.113.0`.

**Full changes**: `v1.112.0` -> `v1.113.0`
