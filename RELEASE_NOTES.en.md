## Lime v1.138.0

Simplified Chinese release notes are the primary version.

### New Features

- Added a unified Composer Controller so the home and conversation composers share structured drafts, submit intents, history recall, and start, queue, steer, and interrupt routing.
- Added persistent App Server `promptHistory/read|append` history with cross-process JSONL paging, file locking, malformed-row tolerance, and cross-platform log identity.
- Added a Desktop capability/readiness contract, a macOS native helper, and security-scoped bookmark lifecycle support for permissions, Launch Services, displays, HID, device keys, and packaged-resource status.
- Added a cross-platform desktop resource manifest covering App Server, Code Mode, Windows sandbox helpers, and the macOS native host with identity, architecture, and SHA-256 verification.

### Fixes

- Fixed home warm-up writing the production local override too early, which made the routing owner clear the active draft, hide the Banner/skin Hero, and prevent send from reusing the warmed session; warm-up now only prepares the session and routing changes at commit time.
- Fixed the official Agnes `agnes-2.5-pro` endpoint being treated as a non-executable custom model; unknown endpoints remain fail closed.
- Fixed Composer production-bundle initialization order, asynchronous history merging, failed-draft retention, and large-paste expansion boundaries.
- Fixed drift in queued-thread actions, model restoration, reasoning projection, and narrow-window Header and Timeline layouts.

### Improvements and Refactoring

- Converged Composer, prompt history, Desktop capabilities, and resource integrity on single current owners without adding an Electron business backend or Renderer persistence source.
- Aligned the current Codex reasoning policy with `persistent` effort, five-locale selector copy, and source-of-truth guards for model context and tool policy.
- Strengthened the Windows Squirrel release path with an installed Code Mode Gate B that verifies the `Lime.exe -> app-server.exe -> code-mode-host.exe` process chain.
- Updated DeepSWE adapter contracts and governance assertions to prevent fixture manifests from drifting from current owners.

### Testing and Quality

- Expanded App Server protocol/schema/generated-client, Composer, prompt-history, Desktop Host, macOS native-host, resource-manifest, Windows Squirrel, and Code Mode Gate B regressions.
- Release validation covers version consistency, TypeScript, protocol contracts, affected frontend/Rust tests, real Electron GUI smoke, and release-workflow guards.
- Windows packaged Gate B evidence is produced by the `windows-2022` release workflow; local static tests are not presented as a Windows platform pass.

### Documentation

- Updated the global architecture, Codex GUI alignment plan, and Desktop platform parity plan with current owners, capability matrices, evidence levels, and remaining platform gaps.
- Added the v1.138.0 release execution plan with candidate scope, excluded Codex build references, validation, and release closeout steps.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.138.0`.
- The live Codex Desktop visual baseline remains blocked by macOS AppleEvent permissions; the home Banner and skin Hero remain an explicit product exception.

**Full changes**: `v1.137.0` -> `v1.138.0`
