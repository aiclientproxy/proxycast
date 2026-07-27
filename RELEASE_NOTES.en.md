## Lime v1.114.0

<sub>The Simplified Chinese release notes are the primary version. This English page is a companion for international readers.</sub>

### New Features

- Added the current Gemini GenerateContent transport with SSE, text and image inputs, tool calls, usage/finish events, and durable `thoughtSignature` tool history.
- Unified Ollama execution on the unauthenticated OpenAI Responses transport while retaining `/api/tags` as the independent model discovery endpoint.
- Added official OpenAI Responses Hosted Web Search and Image Generation with provider-executed tool lifecycle projection and history preservation across turns.
- Added the App Server v2 transient `model/rerouted` notification for trusted model changes reported by official Responses endpoints.

### Fixes

- Fixed credential-scoped model discovery caches not being read correctly by the catalog, aligning cache lookup with provider readiness.
- Fixed stale models, credentials, reasoning effort, or service tier surviving provider configuration changes; turn startup now reconciles against the latest catalog.
- Fixed provider/model names and capability-free `models[]` entries granting execution authority; `inferred_hint` now fails closed consistently.
- Fixed model-selection changes projecting inconsistent settings across foreground turns, background continuations, queued resume, workflow retry, and mailbox paths.

### Improvements and Refactors

- Converged typed provider `models[]`, capability provenance, catalog refresh, route preflight, and durable thread selection on one current control chain.
- Unified provider/model resolution across Agent, media generation, settings, and model pickers to reduce drift between frontend defaults and executable backend capabilities.
- Removed the `OllamaChat`/NDJSON execution protocol and `custom_models/customModels` dual paths without name-based or legacy selection-store fallbacks.
- Published reroute and settings changes through the existing transient RuntimeEvent/v2 projector without EventLog persistence or cold-resume replay.

### Tests and Quality

- Added Rust regressions for Gemini, Ollama Responses, Hosted Web Search, model reroute, capability admission, and catalog reconciliation.
- Added public JSON-RPC model-selection refresh coverage and expanded provider settings, model picker, image/video/voice entry-point, and GUI fixture tests.
- Synchronized App Server JSON schemas and generated TypeScript protocol types, strengthening current bridge, real Electron Gate B, and zero mock-fallback evidence.

### Documentation

- Updated model transports, capability trust boundaries, catalog/turn selection data flow, database fields, and Codex alignment execution records.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and lockfile versions to `1.114.0`.

**Full changes**: `v1.113.0` -> `v1.114.0`
