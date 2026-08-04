## Lime v1.120.1

Simplified Chinese release notes are the primary version.

### New Features

### Fixes

- Prevented internal provenance metadata from ordinary runtime Thread Items crossing the v2 projection boundary, so public items such as `contextCompaction` no longer expose private fields such as `sourceEventType`; imported history keeps its controlled typed metadata.

### Improvements and Refactoring

### Testing and Quality

- Added projection library coverage and a fork-compaction JSON-RPC integration regression for the metadata boundary and replacement/tail recovery after restart.

### Documentation

### Other

- Bumped the root app, CLI npm package, Rust workspace, and lockfile versions to `1.120.1`.

**Full changes**: `v1.120.0` -> `v1.120.1`
