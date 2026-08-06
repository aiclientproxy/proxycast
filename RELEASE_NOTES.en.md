## Lime v1.122.0

Simplified Chinese release notes are the primary version.

### New Features

- Added Thread sections with create, rename, delete, move, persistence, and sidebar organization for conversations.
- Wired Thread section listing, ordering, and persistence through the App Server v2 protocol, Rust thread store, typed client, and GUI.

### Fixes

- Fixed ordering, restoration, and historical preview consistency across conversation lists and Agent chat state updates.
- Fixed field synchronization boundaries between Thread metadata, notifications, projections, read models, and JSON-RPC schemas.

### Improvements and Refactoring

- Consolidated conversation grouping, section projection, and sidebar actions in the current App Server and typed gateway, removing the legacy pinned dual path.
- Converged Thread v2 envelopes, schemas, generated types, notifications, and canonical store implementation around the Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> GUI product chain.

### Testing and Quality

- Added Rust JSON-RPC, thread store, typed gateway, GUI, and restoration regression coverage for Thread sections.
- Expanded contract and component coverage for App Server v2 protocol, Agent chat, sidebar conversation projections, and the current runtime fixture.
- Added a macOS release-signing policy that disables timestamps only for non-code resources, retries when Apple's timestamp service is unavailable, and preserves secure timestamps for nested Mach-O binaries required by notarization.

### Documentation

- Synchronized v2 Thread projection, App Server protocol boundaries, refactoring execution plans, and current architecture confirmation records.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.122.0`.

**Full changes**: `v1.121.0` -> `v1.122.0`
