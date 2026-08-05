## Lime v1.121.0

Simplified Chinese release notes are the primary version.

### New Features

- Introduced the Plugin v2 current path. App Center can discover Codex-compatible plugins from bundled, repository, personal, and local-directory catalogs, then review, install, enable, disable, and uninstall them.
- Enabled installed plugins in the current Agent runtime through immutable activation snapshots, including structured `plugin://` mentions in Claw, Skill/MCP injection, MCP App Right Surface rendering, and history restoration.
- Added a controlled Electron WebContentsView HTML host and bundled Browser Plugin assets, and included plugin assets in the Forge packaging path.

### Fixes

- Fixed Plugin MCP provider namespaces being treated as canonical server identities, so resource reads, tool traces, Right Surface rendering, and reload recovery use one stable identity.
- Fixed Skills watcher changes being dropped inside the throttle window by notifying on the leading edge and coalescing a trailing update.
- Fixed stale session-switch, initial-navigation, and Plugin Right Surface state, and made uninstalled plugins retain history without restarting runtimes or replaying tools.
- Fixed canonical history persistence merging already-materialized snapshots a second time, which duplicated reasoning deltas, dropped repeated fragments, and regressed large Codex history imports; added bounded waits for transient SQLite lock contention.

### Improvements and Refactoring

- Consolidated Plugin catalog, installed/enabled state, package validation, and installation transactions in the App Server current owner; Renderer access now goes through the typed JSON-RPC gateway.
- Improved MCP resource provenance, canonical Thread/Turn/Item projection, and Agent Workspace Right Surface hosting while removing production dependence on the renderer legacy registry and Plugin worker.

### Testing and Quality

- Added Rust, Vitest, contract, and real Electron Gate B coverage for Plugin v2 App Center, typed gateway, mention/activation, MCP resources, Right Surface, history restoration, and uninstall semantics.
- Added regression guards for Electron packaged assets, embedded Browser HTML, GUI smoke, Skills watcher behavior, and the tool lifecycle.
- Added canonical reasoning linear-persistence and 1,200-command history-import performance regressions covering repeated deltas, final snapshot replacement, and background progress completion.

### Documentation

- Added Plugin v2 product contracts, installation model, command boundaries, App Center/Claw surfaces, migration cleanup, verification plans, and the corresponding current architecture diagrams.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and lockfile versions to `1.121.0`.

**Full changes**: `v1.120.1` -> `v1.121.0`
