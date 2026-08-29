## Lime v1.136.0

Simplified Chinese release notes are the primary version.

### New Features

- Added the Windows restricted-execution chain: sandbox accounts, restricted tokens, ACL leases, Job Objects, ConPTY, bounded stdin/output, and Firewall/WFP network isolation.
- Added Windows sandbox setup/runner sidecars, resource digest verification, readiness read-back, and a seven-case real execution evidence matrix.
- Scheduled Tasks now freeze the Composer's selected Provider/model as an opaque route for creation, service skills, and migrated history.

### Fixes

- Fixed the Agent Runtime second-turn start/accept race where a historical completed read model could close the active stream before the real `turnId` arrived.
- Fixed Agent reasoning and Markdown block projection so summaries remain visible while headings, tables, lists, fenced code, and inline semantics render correctly.
- Fixed false protocol-drift warnings for `model/list/updated`, Provider URL tenant/path/query handling, and redacted transport error reporting.
- Fixed restricted Windows child startup, handle inheritance, account permissions, desktop/pipe lifecycle, output draining, and firewall read-back boundaries.

### Improvements and Refactoring

- Converged the Windows execution surface on the single `tool-runtime` current owner; Electron only packages and verifies sidecar resources.
- Refactored Scheduled Tasks, Provider selection, Agent session projection, Markdown normalization, and App Server stdio transport while keeping the real Electron/App Server JSON-RPC path.
- Expanded DeepSWE provider evidence, candidate continuation, failure artifacts, and resumable isolated harness flows.

### Testing and Quality

- Added restricted-execution coverage for workspace ACLs, network blocking, output limits, allowlisted stdin, ConPTY, world-writable audits, and process-tree termination.
- Expanded Agent Runtime, Scheduled Tasks, Provider transport, Markdown/streaming, notification catalog, Electron resource verifier, and five-locale GUI regressions.
- Passed TypeScript, protocol contracts, Agent current fixture, real Electron GUI smoke, related Rust tests, and formatting checks.

### Documentation

- Updated architecture and Codex-alignment plans with the Windows sandbox data flow, fail-closed readiness boundaries, Provider route rules, and current/compat/deprecated/dead classifications.
- Added the reasoning-projection and notification-drift execution record and this release candidate's scope and validation evidence.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.136.0`.
- Windows official signing/notarization, cross-platform installer/release assets, and live-provider/verifier evidence still require the corresponding platform or CI runner and are not claimed locally.

**Full changes**: `v1.135.0` -> `v1.136.0`
