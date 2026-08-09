## Lime v1.125.0

Simplified Chinese release notes are the primary version.

### New Features

- Aligned the plugin package path with Agent Plugins v1.0.0: root `plugin.json`, direct-child Skills, and root `mcp.json`.
- Added the Codex Apps extension adapter, standard Apps JSON catalog, and typed `app/list`, `app/read`, and `app/installed` flow.
- Added typed App Server JSON-RPC and GUI integration for `command/exec` and `review/start`.

### Fixes

- Fixed Thread/Turn/Item, filesystem, process, background terminal, review, and Agent state projection drift across App Server, Electron, and GUI.
- Fixed MCP Plugin placeholder lowering, path containment, HTTP header filtering, sibling isolation, and persistent `PLUGIN_DATA` behavior.
- Fixed protocol schema, generated client, model capability, and provider lowering drift.

### Improvements and Refactoring

- Physically removed the retired Plugin package, worker, manager, renderer runtime, v0 filesystem/process/plugin wires, and detached facades.
- Converged business capabilities on the single `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI` chain.
- Removed retired Plugin Lab/sidebar copy, old technical Plugin standards, and zero-reference governance surfaces; return paths are guarded.

### Testing and Quality

- Passed protocol contracts, Rust related tests, governance scans, docs boundary checks, Agent fixtures, GUI smoke, and macOS Electron Gate B.
- Added Windows runner coverage for environment variables, UNC/extended paths, junction/reparse containment, data persistence, and Squirrel Gate B.
- Windows runner artifacts remain a release gate; macOS and historical Windows evidence are not substituted.

### Documentation

- Updated architecture, command boundaries, Plugin v3 contract, Codex parity matrix, cleanup ledger, and release workflow documentation.
- Made the root README the English canonical entry while retaining a standalone Chinese page.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.125.0`.

**Full changes**: `v1.124.0` -> `v1.125.0`
