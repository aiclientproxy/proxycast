## Lime v1.135.0

Simplified Chinese release notes are the primary version.

### New Features

- Completed current Agent Runtime MCP lifecycle, event-stream, and structured-resource projection across Thread/Turn/Item and the GUI.
- Added current Electron Gate B scenarios for environment lifecycle, project directory, Thread queue/revert, Strict Review, and model-provider capabilities.
- Expanded MCP resource-origin, OAuth/startup notification, event-stream status, and multilingual Agent workspace surfaces.

### Fixes

- Fixed MCP manager/client, tool execution, resource reads, and auth notifications across startup races, terminal lifecycle states, and error boundaries.
- Fixed event ordering, turn recovery, message actions, environment state, and workspace projection drift across current Agent bridges.
- Fixed notification and protocol wiring across App Server JSON-RPC, Rust runtime, frontend gateways, and Electron preload.

### Improvements and Refactoring

- Converged environment, MCP, provider capability, Thread controls, and strict review on the current App Server JSON-RPC and RuntimeCore product chain.
- Refactored Rust App Server/runtime, frontend projections/reducers, Agent session gateways, and MCP state components while keeping the real Electron bridge as the only production path.
- Added stable Electron Gate B, public JSON-RPC, current-fixture, resource-origin, and event-sequence test and evidence paths.

### Testing and Quality

- Expanded App Server protocol contracts, MCP exact JSON-RPC, Rust agent/app-server/tool-runtime, Agent session projection, and five-locale GUI tests.
- Added real Electron Gate B scenarios for environment, Thread queue/revert, project directory, Strict Review, MCP event stream/resource origin, and provider capabilities.

### Documentation

- Updated Codex alignment plans, App Server commands/protocols, event projections, script governance, and protocol-schema sources with current/compat/deprecated/dead boundaries.
- Added the v1.135.0 release execution plan with candidate scope, validation results, and platform/packaging evidence boundaries.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.135.0`.
- Windows installers, signing, notarization, official release assets, and live-provider evidence still require the corresponding platform or CI runner and are not claimed locally.

**Full changes**: `v1.134.0` -> `v1.135.0`
