## Lime v1.111.0

<sub>The Simplified Chinese release notes are the primary version. This English page is a companion for international readers.</sub>

### New Features

- Added App Server v2 `artifact/write`, moving workspace document saves onto typed JSON-RPC, canonical artifact snapshots, and verifiable sidecar evidence.
- Added direct v2 notifications for command output, file patches, Plan deltas, and MCP progress across Rust/TypeScript clients, Renderer projection, and GUI updates.
- Extended token usage and read models with cache-write input tokens while keeping Goal accounting, history recovery, and frontend statistics aligned.

### Fixes

- Fixed stale Turn item lists after thread resume/replay, incorrect local-history matches, and topic/sidebar navigation drift.
- Isolated sequence gates for direct notifications and raw side channels to prevent dropped, duplicated, or prematurely terminal streaming events.
- Fixed lifecycle inconsistencies across tool approvals, MCP snapshots, provider turns, recovery, and multi-turn terminal projection.
- Aligned `AGENTS.override.md` precedence with Codex while preserving symlinked working-directory identity and instruction scope.

### Improvements and Refactors

- Removed the legacy `agentSession/runtimeEvents/append` protocol and schemas, plugin runtime gateway, prompt builders, and parallel permission implementation, with restoration guards.
- Split App Server v2 notification projection into command, file-change, MCP, and Plan owners while synchronizing catalogs, schemas, and generated clients.
- Normalized MCP tool result content, structured content, errors, and sidecar metadata across provider history and GUI consumers.

### Tests and Quality

- Added Rust and TypeScript regression coverage for artifact write/replay JSON-RPC, direct v2 notifications, thread history, approvals, MCP, and cache usage.
- Expanded App Server client, Agent current fixture, plugin/workspace GUI smoke, and forbidden-to-restore governance coverage.

### Documentation

- Updated runtime instruction, tool owner, storage alignment, protocol convergence, and Refactor v1 execution documentation.

### Other

- Bumped version facts to `1.111.0` across the root app, CLI npm package, Rust workspace, `lime-rs/Cargo.lock`, and release notes.

**Full changes**: `v1.110.0` -> `v1.111.0`
