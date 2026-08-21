## Lime v1.133.0

Simplified Chinese release notes are the primary version.

### New Features

- Completed the Right Surface Browser Workspace same-tab current path, unifying Electron `WebContentsView`, BrowserRoute, dynamic tools, and visible state projection.
- Added a two-phase preflight/approvedExecute contract for high-risk Browser actions with allow-once, decline, cancel, user takeover, and fail-closed identity checks.
- Added Browser Gate A/Gate B scenarios and evidence entry points for approval, cancel, disconnect, download, permission, user control, and window close.
- Improved Agent session recovery, stream resumption, historical Thread/Turn/Item projection, and cold-read normalization for unfinished turns.

### Fixes

- Fixed session/thread/turn/item/call identity boundaries across Browser dynamic tools, Electron reverse requests, App Server JSON-RPC, and RuntimeCore.
- Fixed separation of native user input from Agent CDP input; user takeover now revokes the debugger, snapshot, approval token, and active turn so stale mutations cannot replay.
- Fixed dynamic-tool lifecycle, approval terminal states, disconnect/cancel/window-close cleanup, and download/permission state projection.
- Fixed Agent flow control, resume binding, history merging, terminal messages, and interrupted content across duplicate notifications, late events, and reloads.
- Fixed cross-platform execution-process environment construction and converged Windows and macOS/Linux environment and shell inheritance behavior.

### Improvements and Refactoring

- Converged Browser, approval, and session state on the single Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI product chain.
- Refactored BrowserTabHost, the dynamic-tool Host, session-state hooks, Browser Workspace status, and related protocol/schema/generated client surfaces to remove duplicate state machines and retired entries.
- Updated architecture, Browser roadmap, Agent runtime recovery, benchmark, and test-governance records with explicit current/compat/deprecated/dead boundaries.

### Testing and Quality

- Added and expanded BrowserTabHost, dynamic-tool, App Server, Rust tool-runtime, Agent session-state, flow-control, Browser Workspace, and protocol-contract tests.
- Added real Electron Browser Gate A/Gate B evidence scripts covering approval, cancellation, disconnect, download, permission, user takeover, and window close.
- Release validation covers version consistency, TypeScript typecheck, protocol contracts, affected Rust/frontend tests, GUI smoke, the Agent current fixture, governance, and local gates; exact results are recorded in the release plan.

### Documentation

- Updated the architecture, Browser current-owner, approval, user-takeover, session-recovery, and Gate A/Gate B evidence documentation.
- Added the v1.133.0 release execution plan with candidate scope, validation results, and platform/packaging evidence boundaries.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.133.0`.
- Windows installers, signing, notarization, official release assets, and live-provider evidence still require the corresponding platform or CI runner and are not claimed locally.

**Full changes**: `v1.132.0` -> `v1.133.0`
