## Lime v1.141.0

Simplified Chinese release notes are the primary version.

### New Features

- Added dedicated Code Mode protocol, runtime, host, and session-facade crates for stdio/gRPC, V8 execution, typed content, cancellation, reconnect, and session-lease lifecycles.
- Extended the shared App Server CLI/TUI surface with thread resume and management, MCP/Skills/Plugin queries, model and permission controls, JSON/JSONL, prompt history, approvals, request-user-input, and queued-input editing.
- Added authenticated remote WebSocket session transport with `ws/wss`, Bearer tokens, protocol identity checks, ping/pong, and fail-closed policy; CLI/TUI reuse the same session facade without duplicating runtime or persistence.
- Added Codex-aligned TUI Markdown, diff, syntax highlighting, OSC 8 links, width-aware tables, clipboard/image paste, slash-command popup, static pager, transcript overlay, and queued follow-up preview.

### Fixes

- Tightened the App Server `command/exec` permission boundary to reject client-supplied grants and preserve fail-closed typed lowering.
- Fixed Code Mode host/client disconnect, duplicate execution, stale-cell, pending-callback, and session-close cleanup so old-generation state cannot leak into reconnects.
- Fixed CLI/TUI terminal lifecycle, external-editor handoff, Unicode cursor, narrow-terminal truncation, queue recovery, and failed-terminal projection boundaries.
- Fixed regressions in file-system watching, Agent Runtime typed content, MCP notification projection, and tool-lifecycle summaries.

### Improvements and Refactoring

- Migrated Code Mode out of the old `tool-runtime` embedded process/V8 implementation; production calls now use the four current crates, with only explicit compatibility exports left at the old boundary.
- Converged CLI Rust command owners and Codex-shaped modules, and aligned npm root/platform launchers, native payload staging, signal forwarding, and release ordering under `packages/cli`.
- Migrated TUI interaction, rendering, and terminal algorithms by snapshot-inventory classification while keeping canonical Thread/Turn/Item as the only session source of truth.
- Added CLI/TUI/Code Mode, remote transport, Windows restricted-execution, Electron release-workflow, and governance guards; Cloud remains only an authenticated-transport extension point, not a production path.

### Testing and Quality

- Added Code Mode protocol/runtime/host/facade lifecycle, typed-content, gRPC, reconnect, and cancellation tests, plus App Server permission-boundary regressions.
- Expanded CLI Gate B, TUI Gate B, real stdio/PTY, npm packaged launcher, remote-auth, snapshot-inventory, terminal-rendering, and cross-platform release guards.
- Kept five-language copy, protocol generation, script/CLI boundaries, Electron Forge, version consistency, and legacy-governance checks in the release gate.

### Documentation

- Updated architecture, command, governance, quality-workflow, and CLI/TUI/Code Mode execution-plan documentation to record the single Product Surface -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item path and the Cloud transport boundary.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.141.0`.
- Excluded local SQLite/WAL, runtime database, and `.DS_Store` artifacts under `undefined/` from this release candidate; the old Code Mode implementation is `dead/deleted`, with no parallel runtime, compat, or deprecated owner added.

**Full changes**: `v1.140.0` -> `v1.141.0`
