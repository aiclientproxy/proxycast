## Lime v1.117.0

<sub>The Simplified Chinese release notes are the primary version. This English page is a companion for international readers.</sub>

### New Features

- Completed the `currentTime/read`, `item/permissions/requestApproval`, and `item/tool/call` host capabilities, keeping clock reads, permission escalation, and desktop dynamic-tool calls on the same Electron Host, App Server JSON-RPC, and RuntimeCore product chain.
- Promoted dynamic-tool calls to canonical Thread Items with namespace, tool, raw arguments, ordered text/image/audio output, status, and duration, preserving the lifecycle across session restoration.
- Completed the six AgentControl tools, typed wait states, child-agent activity projection, and cold-restart restoration for Multi-Agent conversations; parent-owned child threads now explicitly reject direct input.
- Added canonical execution recovery for known Lime Hub models through the bundled registry and canonical Agnes 2.5 Flash/Pro catalog entries, while unknown models remain fail-closed.

### Fixes

- Fixed clean-environment resolution of `@limecloud/app-server-client/browser`, image-task fixtures using an invalid Chat model, model-list generation baseline drift, and incorrect JSON-RPC codes for invalid model parameters.
- Fixed model selectors exposing inferred-only models that RuntimeCore cannot execute, live provider metadata being overwritten by local hints, and persisted Windows Lime Hub models failing on the first Turn.
- Fixed media previews depending on transient chunk/completed notifications and Renderer live-drain helpers; every ranged response now validates offset, length, digest, and size before advancing progress.
- Fixed boundary cases in initial-session and child-agent navigation, canonical Item reads, event sequencing, and pending-interaction restoration.

### Improvements and Refactors

- Migrated `configWarning` from v0 DTOs and schemas to one typed v2 owner; the Renderer only projects decoded global warnings and shows deduplicated messages across all five locales.
- Removed the v0 config-warning and TextPosition/TextRange schemas, media transient-notification consumers, the old protocol facade, and duplicate preview synchronization modules without production compatibility wrappers.
- Tightened permission-profile, sandbox-command, dynamic-tool routing, provider-history, and read-model boundaries; reserved names, schema collisions, identity mismatches, duplicate responses, and late responses all fail closed.
- Switched Skill installation locations to the authoritative backend `localDirectoryPath`, using Electron `openPath` for directories and `showItemInFolder` for files.

### Tests and Quality

- Added Rust, Electron, protocol, and Renderer regressions for host capabilities, dynamic tools, permission profiles, current time, AgentControl, parent-owned threads, v2 config warnings, and model routing.
- Added real Electron Gate B evidence covering preload/IPC, `app_server_handle_json_lines`, App Server, RuntimeCore, canonical read models, visible GUI state, and cold-restart identity.
- Strengthened guards for the app-server-client browser subpath, generated protocols, five-locale resources, legacy surfaces, and the current bridge.

### Documentation

- Updated architecture sources for host capabilities, the canonical DynamicTool payload, v2 config warnings, media transient retirement, and Lime Hub capability provenance.
- Updated Refactor v2 V2-04/V2-05 progress, Gate B evidence, the Windows runtime model-routing plan, and the Soul output-surface roadmap.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and lockfile versions to `1.117.0`.

**Full changes**: `v1.116.0` -> `v1.117.0`
