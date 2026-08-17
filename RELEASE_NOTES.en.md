## Lime v1.131.0

Simplified Chinese release notes are the primary version.

### New Features

- Shipped the Scheduled Tasks current workspace with create, edit, pause, run-now, run history, and conversation lineage flows.
- Added typed `scheduledTask/changed` and `scheduledTask/run/updated` notifications projected through the App Server, Renderer bridge, and Electron Desktop Host.
- Added a soft-delete contract for deleting tasks while a run is active: Agent Run and canonical Thread/Turn history remain, and terminal writes cannot resurrect a tombstone.

### Fixes

- Fixed `new_thread` Scheduled Task runs with `modelId=null` bypassing canonical model selection and failing the real Runtime backend provider/model-selection contract.
- Fixed the run-now page staying stale after a failed run; failed Runs, terminal notifications, and error messages now appear immediately.
- Fixed Host `unsupported/failed` notification results being presented as success, and aligned task refresh and delete-confirmation behavior.

### Improvements and Refactoring

- Migrated the old Automation management surface to the Scheduled Tasks current protocol, single storage mapping, and typed Renderer gateway; removed retired pages, fixtures, and API dual paths.
- Converged manual/due/catch-up/missed/recovery terminal states, overlap policy, one-shot CAS, DST, and startup recovery in the scheduler worker while keeping RuntimeCore/Thread/Turn/Agent Run as the single product chain.
- Updated five locales, protocol schemas, generated clients, governance catalogs, and execution plans, with a real Electron current fixture.

### Testing and Quality

- Passed App Server protocol/client contracts, Scheduled Tasks focused Vitest, and affected Rust related/changed validation.
- Passed the Scheduled Tasks Electron Gate B, Agent runtime current fixture, GUI smoke, governance report, `verify:local`, formatting, and diff checks.

### Documentation

- Updated Scheduled Tasks architecture, command boundaries, migration ledger, roadmap, and implementation plan to record notification, soft-delete, and provider-route owners.
- Added the v1.131.0 release execution plan, including platform evidence and remaining retired Automation cleanup boundaries.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and Cargo.lock versions to `1.131.0`.
- Windows Notification Center, real macOS/Windows sleep-resume, signing, notarization, and formal release-asset evidence still require the corresponding platform or CI runners and are not claimed locally.

**Full changes**: `v1.130.0` -> `v1.131.0`
