## Lime v1.139.0

Simplified Chinese release notes are the primary version.

### New Features

- Added a repository-level `FEATURE-MAP.md` organized by user capability, stable entry point, current owner, and protocol boundary across Agent, Workspace, Provider, MCP, Skills, Plugins, Memory, Artifact, Scheduled Tasks, and Desktop Host.
- Expanded the macOS native host and Windows packaged evidence for window orchestration, permissions, resource identity, Squirrel installed-state checks, and real Electron/App Server Gate B evidence.
- Completed model-provider reasoning-effort and provider-route contracts so capability, defaults, switching, readiness, and reasoning levels share one control plane.

### Fixes

- Fixed the Resource Manager production path dynamically loading a test-only window fixture; close now uses the current window semantics directly.
- Removed CORS access for `tauri://localhost` and `tauri.localhost`, retaining only current local development origins.
- Fixed source-of-truth drift across Curated Task references, Memory continuation, model configuration, and provider routing.

### Improvements and Refactoring

- Fully retired SceneApp and `src-tauri`: the app directory now belongs to Plugin, business calls use Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore / Agent Runtime, and result references belong to Curated Task and Memory.
- Removed Tauri capabilities/schemas plus SceneApp-specific components, hooks, gateways, copy, and positive tests; legacy names remain only in negative restoration guards and historical evidence.
- Updated current Rust crates to use host-neutral or App Server/Desktop Host terminology instead of in-process desktop commands, Tauri state wrappers, and the former root-crate narrative.

### Documentation

- Added a Lime-specific Feature Map based on this product and architecture rather than copying the reference project's capability structure.
- Removed superseded release plans, Plugin v1/v2 documents, old refactor material, iteration notes, and phase research while updating current navigation and architecture sources.
- Removed 98 Markdown files, 1 HTML file, 4 JSON files, and 16 legacy frontend source/test files. The unowned A2UI/Tauri design was retired, while still-current documentation now points to Electron Desktop Host and App Server boundaries. Git history, Release Notes, and immutable evidence retain historical traceability.

### Testing and Quality

- Expanded App Server contract, Desktop Host, macOS native-host, Windows packaged-evidence, model-provider, Curated Task, Memory, and governance regressions.
- Release gates cover version consistency, TypeScript, i18n, protocol contracts, related Rust tests, documentation boundaries, governance scans, real Electron GUI smoke, and resumable full frontend tests.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.139.0`.
- Added no compat or deprecated owner; SceneApp, `src-tauri`, Tauri capabilities, and the test-only window fixture are `dead / deleted / forbidden-to-restore`.

**Full changes**: `v1.138.0` -> `v1.139.0`
