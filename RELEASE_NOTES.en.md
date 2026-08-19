## Lime v1.132.0

Simplified Chinese release notes are the primary version.

### New Features

- Added the Right Surface Browser Workspace, unifying the user-visible page and Agent operations on one Electron `WebContentsView`, tab identity, and BrowserRoute.
- Added the Browser dynamic-tool and Electron Desktop Host reverse-request path for page observation, actions, downloads, permissions, and takeover state projection.
- Added an App Server owner for CodeCell trace/evidence with JSONL lifecycle writes, reducer/replay, and `diagnostics/trace/read` access.
- Added controlled desktop Deepswe smoke, contract, and benchmark harnesses with a real Electron Gate B evidence entry point.

### Fixes

- Fixed Browser opening remaining in a loading state when the canonical Thread had not been created before the Right Surface identity request.
- Fixed missing Browser route validation; stale, cross-window, duplicate, and unknown calls now fail closed across thread/turn/session/tab/view/webContents identities.
- Fixed CodeCell trace joining and closure for late source/output Items, yielded cells spanning Turns, nested invokes, and events arriving after terminal state.
- Fixed provider/model selection, tool inventory, App Server serialization, and workspace skill projection boundaries across the current product chain.

### Improvements and Refactoring

- Removed the retired Browser Runtime, Canvas Browser, external Chrome/CDP, BrowserSessionRef, site-adapter, and connector paths, converging on the Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> GUI product chain.
- Refactored Right Surface, Workbench, Artifact, Service Skill, and five-locale resources to remove duplicate state machines and retired compat entries.
- Updated App Server protocol/schema, generated clients, command catalogs, governance inventories, architecture documentation, and the Browser roadmap.

### Testing and Quality

- Added focused tests for BrowserTabHost, Browser Workspace, dynamic tools, CodeCell trace, protocol contracts, and desktop harnesses.
- Release validation covers version consistency, TypeScript typecheck, protocol contracts, affected Rust/frontend tests, GUI smoke, the Agent current fixture, governance, and local gates; exact results are recorded in the release execution plan.

### Documentation

- Documented the Browser single owner, shared-tab identity, turn cleanup, CodeCell trace owner, and Gate A/Gate B evidence contracts.
- Added the v1.132.0 release execution plan with candidate scope, validation results, and platform evidence boundaries.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.132.0`.
- Windows, packaged signing, notarization, release assets, and live-provider evidence still require the corresponding platform or CI runner and are not claimed locally.

**Full changes**: `v1.131.0` -> `v1.132.0`
