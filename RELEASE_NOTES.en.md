## Lime v1.140.0

Simplified Chinese release notes are the primary version.

### New Features

- Added CLI/TUI product surfaces: the CLI enters the shared RuntimeCore through the stdio App Server, while TUI covers real PTY, alternate-screen, keyboard-input, and terminal-restoration flows.
- Added a single-line JSONL envelope and shell completion to `lime exec`, keeping script integration aligned with the canonical command tree.
- Added video task tools and media artifact projection with one lifecycle, parameter-validation, execution-result, and chat-preview model.
- Added Desktop Host diagnostics and external-backend support, including App Server client sessions, transport, and cross-process state observation.

### Fixes

- Fixed boundary and cleanup issues across the Electron Desktop Host, App Server, and runtime diagnostics so session and tool state do not leak.
- Fixed Agent chat task-protocol noise, media previews, tool summaries, compaction, and runtime-routing assertions.
- Fixed crash-recovery panel behavior, short-window home composer visibility, and settings runtime entry points.

### Improvements and Refactoring

- Migrated the CLI from the retired `lime-cli-npm` entry point to `packages/cli`; Rust CLI/TUI crates now reuse the App Server protocol, client, and canonical Thread/Turn/Item projection.
- External editor handling now supports PATH `.cmd/.bat` shims on Windows and closes the temporary-file handle before launch.
- Removed retired CLI skills, tool documentation, and entry points, and added CLI/TUI, Desktop Host, release-candidate, and governance boundary guards.
- Strengthened macOS native-host, Windows Squirrel, packaged-resource, release-identity, and Gate B evidence scripts; Electron Forge remains the sole packaging source of truth.

### Testing and Quality

- Added real stdio/PTY CLI/TUI Gate B fixtures, video-tool tests, Desktop Host diagnostics tests, and boundary governance regressions.
- Expanded App Server client, command-contract, Electron release-workflow, Windows packaged-evidence, and GUI-main-path coverage.
- Unified five-language copy-boundary, script-governance, documentation-boundary, and version-consistency checks.

### Documentation

- Updated architecture, command, governance, quality-workflow, Feature Map, tool inventory, and CLI/TUI execution-plan documentation to define the single Product Surface -> App Server JSON-RPC business path.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.140.0`.
- Excluded local SQLite/WAL runtime artifacts under `undefined/data/*` from this release candidate; no compat or deprecated owner was added.

**Full changes**: `v1.139.0` -> `v1.140.0`
