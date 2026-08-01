## Lime v1.119.0

Simplified Chinese release notes are the primary version.

### New Features

- MCP server startup now uses typed App Server status notifications, allowing Settings to show `starting`, `ready`, `failed`, and `cancelled` states and refresh server and tool catalogs at terminal states.
- Agent conversations now surface future-protocol unknown Items safely. Live streaming, completion, history recovery, and reload share the same canonical Item instead of silently dropping unknown capabilities.
- Follow-up terminal input in unified exec is attached to the original Command Item and shown as a redacted summary in live and historical conversations; raw stdin never enters notifications, persistence, or the GUI.

### Fixes

- Fixed unknown Items disappearing from the direct timeline after the terminal turn refresh, while preserving the same thread, turn, and item identity during cold recovery.
- Fixed `write_stdin` being projected as a separate Tool Item, stale session bindings after completion, and cross-thread writes not failing explicitly.
- Fixed MCP connection state relying on legacy Desktop events that could drift from the authoritative App Server state.

### Improvements and Refactoring

- Consolidated MCP lifecycle ownership under `mcpServer/startupStatus/updated`, removed the production paths for `mcp:server_started`, `mcp:server_stopped`, and `mcp:server_error`, and added regression guards.
- Unified the typed representation of command terminal interactions and unknown Items across protocol schemas, App Server projections, read models, clients, and the Renderer.
- Restricted unknown Items to upstream type and redacted field names, preventing raw values, event metadata, and generic extension fallbacks from entering the product path.

### Testing and Quality

- Added a real Electron Gate B fixture for MCP startup notifications, covering successful and failed startup, automatic refresh, IPC/App Server hits, and zero production mock fallback.
- Added a Gate B fixture for live and cold unknown Item recovery, covering safe field display, terminal recovery, identity consistency, and sensitive-value redaction.
- Expanded Codex import continuation, Agent runtime fixture, protocol contract, Rust owner, and Renderer regressions for live, historical, and reload terminal-input summaries.

### Documentation

- Updated the MCP current lifecycle, command boundaries, Refactor v2 event projections, and execution progress with current, excluded, and dead-surface decisions.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and lockfile versions to `1.119.0`.

**Full changes**: `v1.118.0` -> `v1.119.0`
