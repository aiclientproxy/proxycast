## Lime v1.115.0

<sub>The Simplified Chinese release notes are the primary version. This English page is a companion for international readers.</sub>

### New Features

- Added the current Vertex Gemini transport with project/location endpoint construction, Bearer access-token authentication, and shared Gemini canonical lowering, SSE, and tool-history semantics.
- Added the current Azure OpenAI Responses route with resource-root URLs, typed `api-version`, and `api-key` authentication while unsupported Chat Completions, WebSocket, and hosted tools fail closed.
- Completed public Fal and xAI video-generation media tasks; xAI request IDs, polling state, and terminal state are durable so App Server restarts resume polling without creating a duplicate task.
- Added the typed `model/list/updated` notification so mounted model views invalidate and refresh after provider, credential, or catalog mutations.
- Added direct Agent timeline previews for Hosted Image Generation results and session sidecar images, with a stable unavailable-media state.

### Fixes

- Fixed stale reasoning effort, service tier, or collaboration model surviving model switches; Thread settings now preflight against the latest catalog before atomic persistence.
- Fixed provider turns ending too early after reasoning-only or empty successful responses by adding a separate bounded resampling budget that preserves tool snapshots and provider token-budget enforcement.
- Fixed HTTP and WebSocket transports ignoring server `x-should-retry`, `Retry-After`, and quota-reset guidance, preventing explicitly non-retryable requests from being amplified.
- Fixed circuit health being shared across different credentials, protocols, endpoints, or API versions, and added exact-route health snapshots that do not expose credentials or endpoints.
- Fixed dedicated media models surviving as Agent chat selections after catalog refresh and capability records without trusted provenance gaining execution authority.

### Improvements and Refactors

- Made `model/list.capabilitySnapshot` the only public capability source and removed the unconsumed global `modelProvider/capabilities/read` protocol, schema, client, and positive tests.
- Converged video lowering, network execution, and provider status in `model-provider`; media runtime now owns progress and durable artifacts while App Server owns route, credential, and worker orchestration.
- Split provider catalog selection, defaults, and refresh coordination into one owner and unified model-setting projection across foreground, background, queued, retry, and mailbox paths.
- Consolidated Agent image, streaming media-reference, and canonical Item content projections to reduce duplicate conversion between history recovery and live rendering.

### Tests and Quality

- Added Rust regressions for Azure Responses, Vertex Gemini, xAI/Fal video, provider health/retry, credential reroute, empty-response resampling, and model catalog refresh.
- Added public JSON-RPC coverage for video tasks and model catalog updates, and synchronized App Server JSON schemas, generated TypeScript protocol types, and Electron direct-notification tests.
- Expanded frontend coverage for Agent image attachments, Hosted Image, streaming media references, model-registry auto-refresh, and the unavailable state in all five locales.
- Added Codex Item/Event rendering coverage inventories and governance guards as a verifiable baseline for the future single ConversationProjection read model.

### Documentation

- Updated current architecture facts for model catalog/admission, provider health/retry, Vertex Gemini, video media tasks, and Agent empty-response handling.
- Added the Codex rendering alignment v2 Item/Event projection matrices, implementation plan, and completion definition while preserving current multi-model and multimodal owners.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and lockfile versions to `1.115.0`.

**Full changes**: `v1.114.0` -> `v1.115.0`
