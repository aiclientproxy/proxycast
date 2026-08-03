## Lime v1.120.0

Simplified Chinese release notes are the primary version.

### New Features

- Codex history import now carries canonical provenance and safe metadata, keeping imported messages, images, command executions, file changes, and dynamic tool records on one identity across live, completed, and cold-recovery states.
- Task Center adds compact location and environment menus for the local project, Git change summary, and current branch, with branch search, checkout/create actions, worktree creation, and a Codex web entry point.
- Conversation timelines add turn-level process summaries and file-artifact classification; completed turns collapse into one readable summary and read-only file history is presented with the correct semantics.

### Fixes

- Fixed duplicate imported user messages when Codex emits both `response_item` and `event_msg`, while retaining the most complete text and image data.
- Fixed inconsistent canonical Thread/Turn/Item projection, repeated tool details, and missing historical previews across history windows, pagination, reload, and turn completion refreshes.
- Fixed image attachments, file-change summaries, and imported-history source markers being lost across rendering paths.

### Improvements and Refactoring

- Propagated controlled typed import metadata through App Server, protocol schemas, generated clients, read models, and the Renderer without allowing raw source payloads into the product path.
- Unified the Codex-style reading column for message text, assistant bubbles, turn summaries, and composers; split Task Center location/environment panels while reusing the existing Git gateway.
- Added focused unit, component, and integration coverage for history import, canonical projection, timeline aggregation, image attachments, file artifacts, and environment menus.

### Testing and Quality

- Expanded Codex import click-through, current runtime fixture, App Server projection, protocol schema, Rust owner, and Renderer coverage for live streaming, history recovery, pagination, and reload.
- Synchronized five-language resources and added stable DOM and environment-menu interaction assertions.

### Documentation

- Updated Codex GUI alignment, conversation compatibility refactoring, Trace layout, and Refactor v1/v2 progress records with current owners, evidence, and remaining boundaries.

### Other

- Bumped the root app, CLI npm package, Rust workspace, and lockfile versions to `1.120.0`.

**Full changes**: `v1.119.0` -> `v1.120.0`
