## Lime v1.134.0

Simplified Chinese release notes are the primary version.

### New Features

- Extended the Browser Workspace with historical pages and same-tab state projection for unified restore, navigation, and visible state reads.
- Added v2 JSON-RPC contracts for App Server environment management, MCP event streams, and model-provider capability queries.
- Added Windows sandbox setup/status, provider capability badges, and a multilingual Browser runtime matrix entry point.

### Fixes

- Fixed identity and terminal cleanup across Browser, Electron reverse requests, App Server, and RuntimeCore for historical pages, downloads, permissions, and window lifecycles.
- Fixed GUI projection consistency for historical Thread/Turn/Item merges, session recovery, streamed events, and unfinished Agent turns.
- Fixed MCP client/manager lifecycle boundaries, tool-execution orchestration, execution environments, and Windows sandbox readiness behavior.

### Improvements and Refactoring

- Converged environment, MCP, provider capability, sandbox, and Browser historical surfaces on the current App Server JSON-RPC and RuntimeCore product chain.
- Refactored App Server v2 schemas, generated clients, Agent session gateways, model selection, and settings projections; removed the retired Playwright browser-tool entry and assets.
- Added Electron Gate B artifact, cold-start, locale-matrix, environment, and provider-capability test and evidence paths.

### Testing and Quality

- Expanded App Server protocol contracts, MCP exact JSON-RPC, Rust tool-runtime, historical Browser Workspace, model capability, Windows sandbox, and Agent session tests.
- Added real Electron Browser Gate B, cold-restart, and multilingual scenarios; release gates cover version consistency, typecheck, protocol contracts, related Rust/frontend tests, GUI smoke, the current fixture, and governance scans.

### Documentation

- Updated architecture, Browser roadmap, Codex alignment plans, desktop smoke, script governance, and protocol-schema sources with current/compat/deprecated/dead boundaries.
- Added the v1.134.0 release execution plan with candidate scope, validation results, and platform/packaging evidence boundaries.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.134.0`.
- Windows installers, signing, notarization, official release assets, and live-provider evidence still require the corresponding platform or CI runner and are not claimed locally.

**Full changes**: `v1.133.0` -> `v1.134.0`
