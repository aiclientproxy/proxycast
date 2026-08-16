## Lime v1.130.0

Simplified Chinese release notes are the primary version.

### New Features

- Added Turn-scoped Responses provider WebSocket sessions with strict prefix checks, `previous_response_id` deltas, prompt cache keys, and complete-request fallback.
- Routed asynchronous Hook results through the session-owned lifecycle: active turns use steer, idle turns use a mailbox, and lifecycle, warning, and cancellation facts are projected.
- Connected transcription to the App Server transcription worker through the `openai_audio_transcription` route, producing workspace transcript artifacts and provider diagnostics.

### Fixes

- Fixed global and workspace discovery for standard `AGENTS.md` / `AGENTS.override.md`, while retaining `.lime` rules as a controlled fallback.
- Fixed MCP concurrency gating so only tools with explicit `read_only_hint=true` and server opt-in run concurrently; missing or false hints remain serial.
- Fixed recovery and state projection around session mailboxes, turn lifecycle, external backend events, and the ProjectThread-first boundary.

### Improvements and Refactoring

- Converged request lowering, incremental response state, WebSocket reuse, and media input handling in the current model-provider client owner.
- Split Hook discovery, runtime, and lifecycle into observable tool-runtime owners and removed duplicate compatibility paths and the retired evidence-export test entry.
- Updated modality contracts, the media-task index, Harness architecture assets, governance catalogs, Claw fixtures, and five locales so transcription and runtime facts share one current chain.

### Testing and Quality

- Added Rust regression coverage for Responses deltas and fallback, MCP read-only concurrency, standard AGENTS discovery, async Hook mailboxes, and session-loop behavior.
- Updated media-task, runtime-facts, ProjectThread-first, modality, governance, and current-fixture contracts; final gate results are recorded in the release execution plan.

### Documentation

- Updated architecture, Playwright E2E, model capability, Skills, Workflow, Warp, and Agent Runtime roadmap documentation to identify the transcription-worker and provider-route owners.
- Added the `v1.130.0` release execution plan and execution-plan navigation entry, plus source and rendered assets for the current Lime Agent Harness architecture.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.130.0`.
- Windows/macOS packaged parity, signing, notarization, and formal release-asset evidence are not claimed locally and remain dependent on CI/platform workflows.

**Full changes**: `v1.129.0` -> `v1.130.0`
