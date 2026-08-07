## Lime v1.123.0

Simplified Chinese release notes are the primary version.

### New Features

- Added a Codex-aligned Hook lifecycle with `hooks/list`, trusted command Hooks, `hook/started` / `hook/completed` notifications, canonical Hook Items, and Desktop timeline rendering.
- Consolidated the executable Skill catalog on exact `skills/list`, with `skills/changed` refresh, user-level enablement through `skills/config/write`, and process-level roots through `skills/extraRoots/set`.
- Added exact `plugin/search` with keyword, scope, workspace, cursor, and limit filtering over the current Plugin catalog.
- Added Apps catalog/readiness methods `app/list`, `app/read`, `app/installed`, and `app/list/updated`, backed by Plugin App capabilities and real runtime state.
- Migrated MCP resource reads and tool calls to `mcpServer/resource/read` and Thread-scoped `mcpServer/tool/call`, preserving structured tool results through the Session-owned runtime.

### Fixes

- Fixed existing Threads rejecting turns when model capability metadata is missing or stale: `turn/start` now refreshes the authoritative Provider catalog before retrying fail-closed route admission.
- Fixed persisted exact AgentControl routes being automatically reconciled to a fallback model when credentials are temporarily unavailable; schema-valid routes now preserve their model and Provider and reach canonical metadata before child Threads become visible.
- Fixed Hook, Skill, and MCP projection drift across App Server v2 notifications, canonical Thread/Turn/Item state, history restoration, and the Renderer timeline.
- Fixed the Model Selector exposing non-executable inferred models; Agent routes remain limited to canonical or provider-explicit capabilities.

### Improvements and Refactoring

- Physically removed singular `skill/list`, legacy MCP tool/resource wires, Settings tool execution without a Thread owner, and their Electron facades, without compat wrappers or production mock fallbacks.
- Converged the App Server v2 protocol, schema registry, Rust/TypeScript typed clients, runtime owners, and GUI gateways around the single `Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item projection -> GUI` product chain.
- Unified Hook discovery, trust validation, sampling gates, lifecycle events, and restoration projections across `tool-runtime`, Agent runtime, and the current App Server owner.

### Testing and Quality

- Added Rust, TypeScript, and public JSON-RPC regression coverage for Hook lifecycle, Skills list/config/extra roots, exact MCP resource/tool methods, Plugin search, canonical notifications, and history restoration.
- Expanded the Agent current fixture, MCP current fixture, Workspace MCP fixture, Provider-generation PendingRoute gate, and real Electron Hook Gate B evidence.
- Synchronized generated protocol schemas/types, command contracts, legacy return guards, script governance, and five-locale GUI regressions.

### Documentation

- Updated App Server command boundaries, MCP/Skills/Hook current ownership, major architecture diagrams, the Codex alignment plan, and Windows model-route repair evidence.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.123.0`.

**Full changes**: `v1.122.0` -> `v1.123.0`
