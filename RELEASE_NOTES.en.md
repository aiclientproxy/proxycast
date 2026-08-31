## Lime v1.137.0

Simplified Chinese release notes are the primary version.

### New Features

- Added a thread activity panel for subagents, MCP tools, and skills, with workspace navigation entry points.
- Added a real Electron Gate B thread-fork scenario and environment-selection projection that preserves Thread/Turn/Item identity and environment world state.
- Added MCP tools, prompts, and resources list-changed notifications with automatic GUI refresh for the affected data.

### Fixes

- Fixed recent notification replay after the Electron App Server connection consumed a notification early, and completed typed MCP progress projection.
- Fixed permission-profile catalog and runtime policy drift so thread start, resume, and fork share the same allow-list checks.
- Fixed canonical Agent Runtime thread projection for environments, Provider routes, and session metadata instead of exposing raw business metadata to the GUI.
- Fixed Workspace, Canvas Workbench change summaries, thread navigation, and session lifecycle behavior across state transitions.

### Improvements and Refactoring

- Converged permission profiles, environment selections, and thread-fork state on App Server/RuntimeCore current owners while preserving the real Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> GUI path.
- Refactored MCP event bridging, the App Server session client, thread activity, and Right Surface projection to reduce duplicate state reads and keep five-locale UI copy aligned.
- Expanded Agent Runtime, MCP, Session History, and Workspace observability plus Gate B smoke scripts for real notification replay and terminal assertions.

### Testing and Quality

- Added regressions for the thread activity panel, thread fork, MCP list-changed events, permission profiles, environment lifecycle, and Canvas Workbench.
- Expanded App Server protocol/client contracts, MCP notification projection, Agent current fixture, Electron Gate B, and five-locale resource checks.
- Release validation covers version consistency, TypeScript, protocol contracts, affected Rust/frontend tests, and real Electron GUI smoke; results are recorded in the release plan.

### Documentation

- Updated command boundaries, the Codex App GUI alignment plan, and script documentation with current owners for MCP notifications, permission profiles, environment projection, and thread activity.
- Added the v1.137.0 release execution plan with candidate scope, the excluded build-reference artifact, validation evidence, and release closeout steps.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.137.0`.
- Windows official signing/notarization, cross-platform installer/release assets, and live-provider evidence still require the corresponding platform or CI runner and are not claimed locally.

**Full changes**: `v1.136.0` -> `v1.137.0`
