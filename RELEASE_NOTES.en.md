## Lime v1.126.0

Simplified Chinese release notes are the primary version.

### New Features

- Added the Desktop Code Mode flow, allowing models to orchestrate frozen tool snapshots through native custom `exec` and `wait` tools with asynchronous cells, notifications, shared storage, waiting, termination, and cancellation.
- Added a standalone `code-mode-host` process that executes JavaScript in sandbox-enabled V8 isolates; App Server only owns the process client and never falls back to in-process V8 in production.
- Added official OpenAI Responses custom-tool lowering and model `tool_mode` plus `custom_tools` capability/readiness gates.

### Fixes

- Fixed provider tool-call name repair, argument coercion, and JSON Schema validation so malformed calls fail closed before handlers while preserving exactly one lifecycle terminal.
- Fixed cross-process correlation for nested Code Mode tools, notifications, cell close, cancellation, and timeouts, preventing late callbacks from leaking into later sampling steps.
- Fixed canonical request, stream, and usage projection drift across OpenAI Responses, Chat Completions, Anthropic, Gemini, Vertex, Azure, and Ollama transports.

### Improvements and Refactoring

- Converged the Agent session loop on thread-owned resources and registries, unifying actor replacement, interruption, shutdown, and active-cell cleanup.
- Restricted the Code Mode V8 provider to the host-internal owner; dev, Electron assets, and Windows builds now produce `app-server` and `code-mode-host` together.
- Added separate SHA-256 values for both sidecars to the Electron release manifest and made packaged verification enforce binary presence and integrity.

### Testing and Quality

- Added loopback request capture across providers for endpoints, authentication, canonical content, tool definitions, generation lowering, and terminal streams.
- Added regression coverage for the Code Mode protocol/process, V8 runtime, provider lowering, session lifecycle, sidecar assets, and resource integrity.
- A real Electron Gate B now proves distinct Electron, App Server, and `code-mode-host` parent/child processes together with custom-exec resampling, canonical Tool Items, and a visible GUI terminal state.

### Documentation

- Updated the Code Mode process owner, provider/tool trust boundary, dual-sidecar build path, product-scope matrix, and execution plans.
- Clarified that remote environments, Codex TUI, and surfaces without a Desktop consumer are outside Lime's current product path.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.126.0`.

**Full changes**: `v1.125.0` -> `v1.126.0`
