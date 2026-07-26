## Lime v1.112.0

<sub>The Simplified Chinese release notes are the primary version. This English page is a companion for international readers.</sub>

### New Features

- Expanded App Server v2 thread management with compaction, naming, loaded-thread listing, search, metadata updates, unsubscribe, and status/closed notifications.
- Added thread background-terminal list, clean, and terminate contracts plus model safety-buffering status updates.
- Completed thread fork and recovery flows for compacted history, mid-turn forks, direct-input policy, and canonical Thread/Turn/Item projection.
- Added v2 model and provider capability contracts for input modalities, reasoning efforts, service tiers, availability notices, upgrades, and dynamic capability reads.

### Fixes

- Fixed identity, replay, status-notification, and read-model drift across thread resume, replay, and fork scenarios.
- Fixed inconsistencies across model routing, provider capability lowering, health checks, and effective configuration projection.
- Fixed service tiers without display fields being silently dropped from the App Server model projection.
- Fixed restored coding tasks retaining an optimistic Turn and exposing duplicate full tool timelines; historical runs now keep a compact process summary and the latest terminal state.
- Fixed visible regressions in conversation scrolling, layout transitions, sidebar conversation actions, and Right Surface synchronization.

### Improvements and Refactors

- Converged App Server protocols, Rust and TypeScript clients, schemas, and generated types while removing superseded legacy session and model entry points.
- Unified Agent Runtime tool options, runtime options, queued intents, session configuration, and unified-exec lifecycle handling.
- Converged canonical article-workspace projection and writeback boundaries, removing the duplicate Renderer selection-writeback path.

### Tests and Quality

- Added public JSON-RPC regression coverage for thread compact, fork, search, metadata, unsubscribe, closed notifications, model listing, and provider capabilities.
- Expanded App Server client, current Agent Runtime fixture, Electron Gate B, model selector, message scrolling, and workspace layout coverage.
- Added frontend and runtime contract snapshots, source-to-scenario mappings, and CLI/TUI test asset inventories.

### Documentation

- Updated the Codex-method product alignment matrix, Agent Runtime coordination plan, memory/query-loop rules, and test-system documentation.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and lockfile versions to `1.112.0`.

**Full changes**: `v1.111.0` -> `v1.112.0`
