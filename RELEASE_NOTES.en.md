## Lime v1.124.0

Simplified Chinese release notes are the primary version.

### New Features

- Upgraded Lime's public positioning to a full-stack AI agent covering code, files, terminals, tools, MCP, Skills, multimodal work, Providers, and long-running multi-agent tasks.
- Added current App Server v2 filesystem and process capabilities for directory/file operations, watching, terminal process startup, output streams, stdin, termination, and status restoration.
- Connected the Agent workspace, Thread/Turn/Item projection, artifacts, and desktop GUI through one traceable execution chain for a task-oriented workflow similar to Claude Code, WorkBuddy, and Codex.
- Added current Plugin package assets plus Gate B packaging and runtime verification entry points.

### Fixes

- Fixed projection drift for files, processes, background terminals, and Agent task state across App Server, Electron host, gateway, and GUI.
- Fixed lifecycle boundaries for long-running work, cancellation, output streams, and history restoration while preserving fail-closed permissions and review semantics.
- Fixed protocol schema, generated client, and model/Provider capability drift during the v2 migration so non-executable models cannot enter Agent routes.

### Improvements and Refactoring

- Physically removed retired Plugin runtimes, workers, legacy v0 filesystem/process/plugin wires, and detached Electron/Renderer facades without production mock fallbacks or compatibility dual paths.
- Converged the App Server v2 protocol, schema registry, Rust/TypeScript typed clients, runtime owners, and GUI gateways around the single `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI` product chain.
- Returned filesystem, process, tools, Skills, MCP, Plugin, and multi-agent capabilities to their current owners, reducing duplicate entry points and cross-layer state copies.

### Testing and Quality

- Added Rust, TypeScript, public JSON-RPC, and real Electron regression coverage for filesystem, process, background terminal, Agent runtime, Plugin package, and protocol v2 flows.
- Expanded current Agent fixtures, tool/Skills/MCP scenarios, Gate B packaging evidence, history restoration, and cancellation/retry paths.
- Synchronized generated protocol schemas/types, command contracts, legacy return guards, script governance, five-locale GUI regressions, and README positioning docs.

### Documentation

- Rewrote the Chinese and English root READMEs to present Lime as a full-stack desktop AI agent similar to Claude Code, WorkBuddy, and Codex while retaining the existing product images.
- Updated App Server filesystem/process boundaries, the Agent runtime chain, Plugin current ownership, governance roadmap, and v2 protocol documentation.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.124.0`.

**Full changes**: `v1.123.0` -> `v1.124.0`
