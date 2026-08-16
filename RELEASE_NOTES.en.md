## Lime v1.129.0

Simplified Chinese release notes are the primary version.

### New Features

- Added candidate model sets and OEM routing policy, filtering candidates by task family, modality, runtime feature, and capability while exposing candidate count, routing mode, candidate set, and policy facts.
- Added unified model-provider and lowering support for speech synthesis, transcription, and text embeddings through OpenAI-compatible multimodal protocols.
- Added media-task worker routing, scheduling, and artifact recording for speech synthesis, including credential-usage tracking and task-state updates.

### Fixes

- Fixed routing projections for no-candidate, capability-gap, and provider-readiness failures so unavailable candidates and stale profile-slot modes are not reported as usable.
- Fixed camelCase/snake_case consistency across App Server, RuntimeCore, and the frontend for model routing, media tasks, thread facts, and pending requests.
- Removed Skills, Harness, and workspace dependencies on the retired evidence-export surface, reducing regressions from legacy APIs, fixtures, and duplicate state sources.

### Improvements and Refactoring

- Converged candidate models, OEM policy, provider capability/readiness, and routing decisions into the current RuntimeCore and model-provider owners; removed the retired evidence-provider/export projection and its dedicated UI/tests.
- Replaced the Harness “problem evidence pack” surface with thread-level runtime facts covering turns, items, pending requests, artifacts, evidence references, and routing decisions.
- Simplified App Server protocol schemas, client types, scripts, fixtures, and governance catalogs by removing dead evidence-export and duplicate compatibility paths.

### Testing and Quality

- Added focused Rust and frontend regression coverage for candidate model sets, OEM routing modes, multimodal provider lowering, media-task artifacts, and current runtime facts.
- Updated App Server client/protocol, modality, scripts, Harness, governance, and five-locale contract coverage; release gate results are recorded in the execution plan.

### Documentation

- Updated architecture, command boundaries, model-routing, App Server/frontend integration, Harness governance, and multimodal runtime-contract documentation.
- Added the `v1.129.0` release execution plan and execution-plan navigation entry.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.129.0`.
- Windows/macOS packaged parity, signing, notarization, and formal release-asset evidence are not claimed locally and remain dependent on CI/platform workflows.

**Full changes**: `v1.128.0` -> `v1.129.0`
