## Lime v1.116.0

<sub>The Simplified Chinese release notes are the primary version. This English page is a companion for international readers.</sub>

### New Features

- Agent conversations now render canonical Turns and Items directly, with one ConversationProjection shared by live events, cold reads, and production resume; Hook, Sleep, Review, MCP, and dynamic-tool states appear in the unified timeline.
- Added one Pending Interaction layer for approvals, additional user input, MCP elicitation, and dynamic-tool interaction, including restoration of requests that are still pending after resume.
- Migrated media reads to the Lime-owned v2 `media/read` method with canonical `threadId`, bounded sidecar access, and stable available, unavailable, and abnormal preview states.
- Added canonical capability recognition for `agnes-2.0-flash` on the official Agnes API Hub, including vision, tools, streaming, and reasoning; non-official or non-HTTPS endpoints do not receive this authorization.

### Fixes

- Fixed duplicate synthesized messages, drift between live and historical content, repeated reasoning/tool text, and lost pending interactions during long-session restoration.
- Fixed slow conversation opening caused by mounting every historical Item and full assistant body; the GUI now renders a bounded Turn window and expands earlier entries or full text on demand.
- Fixed sidecar media reads depending on v0 session identity, silently swallowed read failures, and abnormal payloads entering previews; protocol drift and unprojected notifications now produce visible diagnostics.
- Fixed multi-agent tool results carrying activity events without typed state facts, allowing completed, failed, and waiting child-agent states to reach the canonical projection.

### Improvements and Refactors

- Removed the second canonical Item-to-Message synthesis path, three streaming content-sync hooks, duplicate approval/user-input APIs, and the old MCP elicitation dialog owner, leaving one current Renderer projection chain.
- Tightened assistant message phases, MCP tool results/errors, dynamic-tool multimodal output, and agent-control state into typed protocols synchronized across Rust, JSON Schema, and the TypeScript client.
- Made history windows, long-text previews, media abnormal states, and notification drift fail-visible boundaries instead of relying on generic extensions or production mock fallback.
- Model taxonomy can now inherit task families, modalities, and runtime features from the canonical catalog while retaining strict endpoint provenance checks.

### Tests and Quality

- Expanded frontend and protocol regressions for direct TurnTimeline rendering, history previews, resume replay, Pending Interaction, MCP elicitation, media abnormal states, and protocol drift.
- Added long-list Electron fixtures, canonical thread seeds/oracles, Agent runtime screenshots, and a current tool-execution contract covering real preload/IPC, App Server JSON-RPC, read models, and GUI state.
- Updated Refactor v2 Item/Event projection inventories, render-coverage fixtures, legacy-surface guards, and verification evidence to protect the single current owner and zero production mock fallback.

### Documentation

- Updated the architecture sources and Refactor v2 execution record for Renderer ConversationProjection, direct TurnTimeline, thread-scoped media reads, abnormal-state presentation, and long-list performance.
- Added the target cloud multi-model platform architecture for LimeCore, AsterRouter, and Codex App Server, defining the unique owners for the commercial control plane, model data plane, and desktop runtime.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and lockfile versions to `1.116.0`.

**Full changes**: `v1.115.0` -> `v1.116.0`
