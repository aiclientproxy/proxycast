## Lime v1.127.0

Simplified Chinese release notes are the primary version.

### New Features

- Added a top-level Scheduled Tasks workspace with list and detail views, create and edit flows, enable controls, immediate runs, schedule previews, and run history.
- Added nine `scheduledTask/*` App Server JSON-RPC methods with generated protocol types and a typed Renderer gateway.
- Routed scheduled execution through RuntimeCore and the canonical Thread/Turn/Item path, supporting both new conversations and continuation from explicit source conversations.

### Fixes

- Added atomic claims, a bounded 24-hour catch-up window, missed-run records, overlap skipping, and one-shot terminal handling to prevent duplicate claims and missing run history.
- Fixed New York DST gaps and repeated hours, manual-run schedule anchors, clock rollback behavior, and immediate runs for paused tasks.
- Added startup recovery that terminalizes stale queued or running Agent Runs while reusing the real result when a canonical Turn is already terminal.

### Improvements and Refactoring

- Reused `automation_jobs` as the only task table and projected run history from `agent_runs`, avoiding a second persistence owner.
- Converged the task-center and sidebar navigation so the Renderer retains only filters, selection, and editor form state while App Server read models own task facts.

### Testing and Quality

- Added public JSON-RPC, scheduler claim and recovery, RuntimeCore lineage, read-model, and fail-closed regression coverage for Scheduled Tasks.
- Added stable tests for the Scheduled Tasks workspace, typed gateway, navigation, and all five bundled locales.
- The release candidate passed TypeScript type checking, protocol contracts, Rust related tests, 70 targeted frontend tests, and a real Electron GUI smoke run.

### Documentation

- Added Scheduled Tasks product requirements, interaction design, protocol and runtime architecture, migration ledger, verification contract, and Codex parity matrix.
- Updated the global architecture and command boundaries to keep Electron as the JSONL transport host while App Server owns scheduling, CRUD, and runtime state.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.127.0`.
- Legacy `automationJob/*` methods, the old Settings consumer, terminal notifications, soft deletion, and dedicated cross-platform evidence remain tracked follow-up work; this release does not claim the full roadmap is complete.

**Full changes**: `v1.126.0` -> `v1.127.0`
