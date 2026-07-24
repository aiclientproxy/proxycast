# @limecloud/app-server-client

`@limecloud/app-server-client` is the TypeScript client surface for independent apps
that talk to `app-server` over JSON-RPC.

Current scope:

- build initialize / initialized requests;
- build current `thread/*` and `turn/*` requests plus the remaining
  session-scoped action, media, and evidence requests;
- encode and decode newline-delimited JSON-RPC messages;
- resolve sidecar binary names, packaged paths, and stdio launch args;
- read release manifest metadata and select the current platform artifact;
- build sidecar launch config from manifest + resources path;
- verify the sidecar binary sha256 before launch;
- spawn / connect a stdio sidecar with the initialize handshake;
- supervise sidecar crash, startup failure, and restart with deterministic backoff;
- route direct v2 Thread / Turn / Item lifecycle notifications into app-owned
  state, including `item/agentMessage/delta`;
- keep `agentSession/event` only as the raw `media.read.*` side-channel;
- use `AppServerConnection` for typed App Server request / response flows;
- resolve or reject typed reverse `serverRequest` messages by their outer JSON-RPC
  request id; missing response wiring fails closed;
- use `AgentRuntimeClient` as the standard facade for runtime turn, action,
  thread read, evidence export, and event subscription flows.

Session archive semantics:

- list archived sessions with `agentSession/list` and `archivedOnly: true`;
- archive or unarchive sessions with `agentSession/update` and `archived: true`
  or `archived: false`;
- preserve App Server JSON-RPC errors as `AppServerRequestError`; callers must
  fail closed instead of falling back to legacy `agent_runtime_*` commands or
  mock responses.

Electron main integration shape:

```ts
import {
  AppServerAgentEventRouter,
  startPackagedAppServerSidecar,
} from "@limecloud/app-server-client";

const { connected, lifecycle } = await startPackagedAppServerSidecar(
  {
    clientInfo: { name: "content_studio", version: app.getVersion() },
  },
  {
    resourcesPath: process.resourcesPath,
    restartPolicy: { maxAttempts: 3, initialDelayMs: 500, maxDelayMs: 30_000 },
  },
);

const { connection } = connected;
app.on("before-quit", () => void lifecycle.stop());

const eventRouter = new AppServerAgentEventRouter();
eventRouter.subscribeLifecycle((notification) => {
  mainWindow.webContents.send("agent:lifecycle", notification);
});
eventRouter.subscribe((mediaEvent) => {
  mainWindow.webContents.send("agent:media", mediaEvent);
});

void (async () => {
  while (!mainWindow.isDestroyed()) {
    await eventRouter.dispatch(await connection.nextNotification());
  }
})();

const session = await connection.startSession({
  serviceName: "host-app",
  threadSource: "desktop",
  historyMode: "paginated",
});

const turn = await connection.startTurn({
  threadId: session.result.thread.id,
  input: [{ type: "text", text: "生成草稿" }],
  model: "gpt-5-codex",
});
```

Agent runtime SDK facade:

```ts
import { createAgentRuntimeClient } from "@limecloud/app-server-client";

const runtime = createAgentRuntimeClient(connection, {
  request: { timeoutMs: 120_000 },
});

runtime.subscribeLifecycleEvents((notification) => {
  mainWindow.webContents.send("agent:lifecycle", notification);
});
runtime.subscribeEvents((mediaEvent) => {
  mainWindow.webContents.send("agent:media", mediaEvent);
});

await runtime.startTurn({
  threadId: session.result.thread.id,
  input: [{ type: "text", text: "整理资料并生成草稿" }],
});

const thread = await runtime.readThread({
  threadId: session.result.thread.id,
  includeTurns: true,
});

await runtime.exportEvidence({
  sessionId: thread.result.thread.sessionId,
  includeEvents: true,
});
```

`AgentRuntimeClient` is a facade over the current App Server JSON-RPC methods.
`readThread` maps to `thread/read` and requires a hydrated `threadId`.
Lifecycle delivery uses the direct v2 server notifications
`thread/started`, `turn/started`, `turn/completed`, `item/started`,
`item/completed`, and `item/agentMessage/delta`. The raw event subscription is
reserved for `media.read.*`; it is not a second lifecycle protocol.

Typed reverse requests are answered through the same connection that received
them. Use `respondServerRequest(requestId, result)` or
`rejectServerRequest(requestId, error)`; do not reconstruct an action payload
from thread or turn metadata, and do not fall back to `agentSession/action/respond`.

This package does not import Lime Rust crates, Tauri commands, Agent DTOs, or
renderer UI code. Electron apps should use it from main / preload boundaries and
project events into their own renderer state.

Sidecar `backendMode: "mock"` is test-only. Production hosts must use `runtime`,
`external`, or fail closed; they must not treat the mock backend as a fallback
for Agent Runtime, evidence export, or renderer UI flows.
