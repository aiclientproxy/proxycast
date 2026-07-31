## Lime v1.118.0

<sub>The Simplified Chinese release notes are the primary version. This English page is a companion for international readers.</sub>

### New Features

- Skill catalogs now refresh automatically when Skills are created, changed, or removed under default roots, or after successful local Lime Skill mutations. Composer always reloads the current App Server `skill/list`, without requiring a manual refresh or app restart.
- Agent plans now use the typed `turn/plan/updated` notification, showing the same checklist during a live Turn and after cold restoration. Failed or malformed snapshots cannot replace the latest valid plan.
- Agent errors now follow a typed retry and terminal chain: retrying, final failure, and the authoritative Turn terminal have distinct states and can be restored from the canonical read model.
- MCP OAuth completion now reaches the Renderer through a direct App Server v2 typed notification, keeping the login window, MCP management state, and event identity aligned.

### Fixes

- Fixed explicitly configured Provider models on Windows being hidden from the chat model selector when authoritative capability metadata was unavailable. Inferred-only, non-executable models remain fail closed.
- Fixed Electron updater sessions treating equal, older, invalid, or missing versions as installable updates. Only a strictly newer semantic version can enter download and installation.
- Fixed update failures, up-to-date results, and available updates sharing the same message, and aligned download, retry, and restart states across the sidebar, About page, and update notification window.
- Fixed inconsistent presentation of non-executable Provider routes between submit failures and runtime events, and missing global toast hosts on standalone pages.
- Fixed the Provider setup flow showing “Finish adding” before a configuration had been persisted and returning to the wrong model context after a successful save.

### Improvements and Refactors

- Consolidated `skills/changed`, `error`, `turn/plan/updated`, and `mcpServer/oauthLogin/completed` into the App Server v2 protocol, schemas, generated client, and Renderer typed event bus without adding a production mock or second event owner.
- Unified Skill catalog cache invalidation in `lime-skills`; default roots use recursive watchers and throttled notifications, while reconnect and remount still perform an active current-catalog read.
- Kept the canonical `update_plan` tool item as the restoration source without generating a duplicate Plan Item, tool card, or decision surface.
- Moved updater version comparison and session transitions into a focused pure owner and tightened the `available -> downloaded -> installing/restarting` state flow.

### Tests and Quality

- Added Rust, Electron, protocol, Renderer, and script regressions for Skills refresh, typed errors, plan updates, MCP OAuth, Provider model catalogs, and updater semantics.
- Expanded the current Agent fixture with typed-error success/failure, live and restored plan updates, Skills runtime refresh, and real Electron/preload/IPC/App Server/read-model identity checks.
- Audited the core user flow across startup, Provider/model selection, Agent chat, stop-and-continue, history restoration, About, and updater states, with zero production mock fallbacks.

### Documentation

- Updated architecture and execution sources for App Server v2 notifications, Skill catalog invalidation, MCP OAuth, typed error/plan recovery, and updater/model-catalog behavior.
- Recorded the Windows model-catalog and updater reliability work, core user-flow E2E audit, and the current Refactor v2 V2-05 status.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and lockfile versions to `1.118.0`.
- The Windows Squirrel packaged Gate B for N-1 discovery, download, restart, and installation still requires a real Windows host; macOS Electron evidence does not replace that platform check.

**Full changes**: `v1.117.0` -> `v1.118.0`
