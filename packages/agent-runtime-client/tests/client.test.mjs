import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import * as runtimeClientPackage from "../dist/index.js";

const {
  AgentRuntimeEventPipeline,
  AgentRuntimeSequenceViolationError,
  AppServerConnection,
  createAgentRuntimeClient,
  createAgentRuntimeClientFromSessionGateway,
  runtimeExecutionEventFromLifecycleNotification,
} = runtimeClientPackage;

test("createAgentRuntimeClient delegates current App Server requests", async () => {
  const sent = [];
  const responses = [
    {
      id: 1,
      result: {
        turn: {
          id: "turn-1",
          status: "inProgress",
        },
      },
    },
    {
      id: 2,
      result: {
        turnId: "turn-1",
      },
    },
    {
      id: 3,
      result: {
        tools: [],
        updatedAt: "2026-05-15T00:00:01.000Z",
      },
    },
  ];
  const runtime = createAgentRuntimeClient(
    new AppServerConnection({
      send(message) {
        sent.push(message);
      },
      async nextMessage() {
        const response = responses.shift();
        assert.ok(response, "unexpected App Server request");
        return response;
      },
    }),
  );

  const result = await runtime.startTurn({
    threadId: "thread-1",
    input: [{ type: "text", text: "生成草稿" }],
  });
  const steered = await runtime.steerTurn({
    threadId: "thread-1",
    expectedTurnId: "turn-1",
    input: [{ type: "text", text: "补充约束" }],
  });
  await runtime.readToolInventory({ sessionId: "session-1" });

  assert.equal(sent[0].method, "turn/start");
  assert.equal(sent[1].method, "turn/steer");
  assert.equal(sent[2].method, "agentSession/toolInventory/read");
  assert.equal(result.result.turn.id, "turn-1");
  assert.equal(steered.result.turnId, "turn-1");
});

test("session gateway delegates the standard runtime client surface", async () => {
  const calls = [];
  const gateway = {
    ...createMinimalSessionGateway(calls),
    async readToolInventory(params, options) {
      calls.push(["readToolInventory", params, options]);
      return requestResult(6, {
        tools: [],
        updatedAt: "2026-05-15T00:00:01.000Z",
      });
    },
  };
  const runtime = createAgentRuntimeClientFromSessionGateway(gateway);
  const options = { timeoutMs: 120_000 };

  await runtime.startTurn(
    {
      threadId: "thread-1",
      input: [{ type: "text", text: "生成草稿" }],
    },
    options,
  );
  await runtime.steerTurn(
    {
      threadId: "thread-1",
      expectedTurnId: "turn-1",
      input: [{ type: "text", text: "补充约束" }],
    },
    options,
  );
  await runtime.readThread(
    { threadId: "thread-1", includeTurns: true },
    options,
  );
  await runtime.readToolInventory({ sessionId: "session-1" }, options);
  await runtime.cancelTurn({ threadId: "thread-1", turnId: "turn-1" }, options);
  await runtime.respondAction(
    {
      sessionId: "session-1",
      requestId: "request-1",
      actionType: "ask_user",
      confirmed: true,
      response: "继续",
    },
    options,
  );
  assert.deepEqual(
    calls.map(([name]) => name),
    [
      "startTurn",
      "steerTurn",
      "readThread",
      "readToolInventory",
      "cancelTurn",
      "respondAction",
    ],
  );
  assert.equal(calls[2][2].timeoutMs, 120_000);
});

test("package root exports the direct lifecycle mapper without compatibility aliases", () => {
  assert.equal(
    typeof runtimeExecutionEventFromLifecycleNotification,
    "function",
  );
  assert.equal(
    "runtimeExecutionEventFromCanonicalEvent" in runtimeClientPackage,
    false,
  );
  assert.equal(
    "runtimeExecutionEventFromAgentEvent" in runtimeClientPackage,
    false,
  );
  assert.equal(
    "createSchemaVersionCompatibilityMiddleware" in runtimeClientPackage,
    false,
  );
});

test("package root type surface exports canonical thread read types only", async () => {
  const declarations = await readFile(
    new URL("../dist/index.d.ts", import.meta.url),
    "utf8",
  );

  assert.equal(declarations.includes("type ThreadReadParams"), true);
  assert.equal(declarations.includes("type ThreadReadResponse"), true);
  assert.equal(declarations.includes("type TurnStartParams"), true);
  assert.equal(declarations.includes("type TurnSteerParams"), true);
  assert.equal(declarations.includes("type TurnInterruptParams"), true);
  assert.equal(declarations.includes("AgentSessionReadParams"), false);
  assert.equal(declarations.includes("AgentSessionReadResponse"), false);
  assert.equal(declarations.includes("AgentSessionTurnStartParams"), false);
  assert.equal(declarations.includes("AgentSessionTurnCancelParams"), false);
});

test("direct lifecycle mapper reads typed Thread Turn and Item", () => {
  const thread = runtimeExecutionEventFromLifecycleNotification(
    threadStartedNotification(),
  );
  const turn = runtimeExecutionEventFromLifecycleNotification(
    turnNotification("interrupted", { completedAt: 220 }),
  );
  const item = runtimeExecutionEventFromLifecycleNotification(
    itemNotification({ method: "item/completed", status: "completed" }),
  );
  const command = runtimeExecutionEventFromLifecycleNotification(
    itemNotification({
      type: "commandExecution",
      method: "item/completed",
      status: "completed",
    }),
  );

  assert.equal(thread.status, "running");
  assert.equal(thread.title, "Research");
  assert.equal(turn.eventClass, "turn.canceled");
  assert.equal(turn.status, "canceled");
  assert.equal(item.eventClass, "tool.result");
  assert.equal(item.toolCallId, "tool_call_1");
  assert.equal(command.eventClass, "command.exited");
});

test("malformed direct lifecycle notifications fail closed with explicit mapper errors", async () => {
  for (const runtime of [
    createRuntimeClient(),
    createAgentRuntimeClientFromSessionGateway(createMinimalSessionGateway()),
  ]) {
    assert.equal(
      await runtime.dispatchEvent({
        method: "turn/started",
        params: { threadId: "thread-1", turn: { status: "inProgress" } },
      }),
      false,
    );
  }
  assert.throws(
    () =>
      runtimeExecutionEventFromLifecycleNotification({
        method: "thread/started",
        params: { thread: undefined },
      }),
    /Canonical Thread must be an object/,
  );
  assert.throws(
    () =>
      runtimeExecutionEventFromLifecycleNotification({
        method: "turn/started",
        params: { threadId: "thread-1", turn: undefined },
      }),
    /Canonical Turn must be an object/,
  );
  assert.throws(
    () =>
      runtimeExecutionEventFromLifecycleNotification({
        method: "item/started",
        params: {
          threadId: "thread-1",
          turnId: "turn-1",
          startedAtMs: 1,
          item: undefined,
        },
      }),
    /Canonical ThreadItem must be an object/,
  );
});

test("root client dispatches direct lifecycle and delta only to lifecycle listeners", async () => {
  const runtime = createRuntimeClient();
  const lifecycle = [];
  const signals = [];
  runtime.subscribeLifecycleEvents((event) => lifecycle.push(event));
  runtime.subscribeSignalEvents((event) => signals.push(event));

  for (const notification of directToolLifecycle()) {
    assert.equal(await runtime.dispatchEvent(notification), true);
  }
  const delta = agentMessageDeltaNotification();
  assert.equal(await runtime.dispatchEvent(delta), true);
  assert.equal(await runtime.dispatchEvent(errorNotification(true)), true);
  assert.equal(
    await runtime.dispatchEvent(turnPlanUpdatedNotification()),
    true,
  );
  assert.equal(
    await runtime.dispatchEvent(rawNotification("turn.completed")),
    false,
  );

  assert.deepEqual(
    lifecycle.map((event) => event.method),
    [
      "turn/started",
      "item/started",
      "item/completed",
      "turn/completed",
      "item/agentMessage/delta",
    ],
  );
  assert.deepEqual(signals, [
    errorNotification(true),
    turnPlanUpdatedNotification(),
  ]);
});

test("session gateway dispatches direct lifecycle and rejects wrapper lifecycle", async () => {
  const runtime = createAgentRuntimeClientFromSessionGateway(
    createMinimalSessionGateway(),
  );
  const lifecycle = [];
  const signals = [];
  runtime.subscribeLifecycleEvents((event) => lifecycle.push(event));
  runtime.subscribeSignalEvents((event) => signals.push(event));

  assert.equal(
    await runtime.dispatchEvent(turnNotification("inProgress")),
    true,
  );
  assert.equal(
    await runtime.dispatchEvent(agentMessageDeltaNotification()),
    true,
  );
  assert.equal(await runtime.dispatchEvent(planDeltaNotification()), true);
  assert.equal(
    await runtime.dispatchEvent(mcpToolCallProgressNotification()),
    true,
  );
  assert.equal(await runtime.dispatchEvent(errorNotification(false)), true);
  assert.equal(
    await runtime.dispatchEvent(turnPlanUpdatedNotification()),
    true,
  );
  assert.equal(
    await runtime.dispatchEvent(rawNotification("turn.started")),
    false,
  );

  assert.deepEqual(
    lifecycle.map((event) => event.method),
    [
      "turn/started",
      "item/agentMessage/delta",
      "item/plan/delta",
      "item/mcpToolCall/progress",
    ],
  );
  assert.deepEqual(signals, [
    errorNotification(false),
    turnPlanUpdatedNotification(),
  ]);
});

test("disabling sequence verification does not restore raw lifecycle", async () => {
  const runtime = createAgentRuntimeClientFromSessionGateway(
    createMinimalSessionGateway(),
    { sequenceVerifierMode: "off" },
  );

  assert.equal(
    await runtime.dispatchEvent(rawNotification("tool.result")),
    false,
  );
  assert.equal(
    await runtime.dispatchEvent(
      itemNotification({ method: "item/completed", status: "completed" }),
    ),
    true,
  );
});

test("nextEvent consumes direct notifications from gateway sources", async () => {
  const directNotification = itemNotification({ type: "agentMessage" });
  const directRuntime = createAgentRuntimeClientFromSessionGateway({
    ...createMinimalSessionGateway(),
    async nextEvent(timeoutMs) {
      assert.equal(timeoutMs, 500);
      return directNotification;
    },
  });
  const directReceived = [];
  directRuntime.subscribeLifecycleEvents((event) => directReceived.push(event));

  const directEvent = await directRuntime.nextEvent(500);

  assert.equal(directEvent.method, "item/started");
  assert.equal(directReceived[0].params.item.type, "agentMessage");

  const drainedNotification = turnNotification("inProgress");
  const drainedRuntime = createAgentRuntimeClientFromSessionGateway({
    ...createMinimalSessionGateway(),
    async drainEvents(limit) {
      assert.equal(limit, 1);
      return [{ method: "log/list", params: {} }, drainedNotification];
    },
  });

  const drainedEvent = await drainedRuntime.nextEvent();
  assert.equal(drainedEvent.method, "turn/started");

  const directSignal = errorNotification(true);
  const signalRuntime = createAgentRuntimeClientFromSessionGateway({
    ...createMinimalSessionGateway(),
    async nextEvent() {
      return directSignal;
    },
  });
  const signals = [];
  signalRuntime.subscribeSignalEvents((event) => signals.push(event));

  const signalEvent = await signalRuntime.nextEvent();
  assert.equal(signalEvent.method, "error");
  assert.deepEqual(signals, [directSignal]);

  const drainedSignalRuntime = createAgentRuntimeClientFromSessionGateway({
    ...createMinimalSessionGateway(),
    async drainEvents() {
      return [{ method: "log/list", params: {} }, errorNotification(false)];
    },
  });
  assert.equal((await drainedSignalRuntime.nextEvent()).method, "error");
});

test("sequence verifier fails closed for orphan direct tool completion", async () => {
  const notification = itemNotification({
    method: "item/completed",
    status: "completed",
  });
  const runtime = createRuntimeClient();

  assert.equal(await runtime.dispatchEvent(notification), false);

  const streamingRuntime = createAgentRuntimeClient(
    new AppServerConnection({
      send() {},
      async nextMessage() {
        return notification;
      },
    }),
  );
  await assert.rejects(
    streamingRuntime.nextEvent(),
    AgentRuntimeSequenceViolationError,
  );
});

test("collect-diagnostics keeps direct dispatch observable", async () => {
  const runtime = createAgentRuntimeClientFromSessionGateway(
    createMinimalSessionGateway(),
    { sequenceVerifierMode: "collect-diagnostics" },
  );
  const received = [];
  runtime.subscribeLifecycleEvents((event) => received.push(event));

  const handled = await runtime.dispatchEvent(
    itemNotification({ method: "item/completed", status: "completed" }),
  );

  assert.equal(handled, true);
  assert.equal(received[0].params.item.status, "completed");
});

test("direct lifecycle adapters run before verification and may fan out", async () => {
  const runtime = createRuntimeClient({
    adapters: [
      ({ notification, event }) => {
        if (event.method !== "item/started") return;
        return [
          notification,
          itemNotification({
            method: "item/completed",
            status: "completed",
          }),
        ];
      },
    ],
  });
  const received = [];
  runtime.subscribeLifecycleEvents((event) => {
    received.push(event.method);
  });

  const handled = await runtime.dispatchEvent(
    itemNotification({ method: "item/started", status: "inProgress" }),
  );

  assert.equal(handled, true);
  assert.deepEqual(received, ["item/started", "item/completed"]);
});

test("nextEvent returns direct adapter fanout before another transport read", async () => {
  let reads = 0;
  const source = itemNotification({
    method: "item/started",
    status: "inProgress",
  });
  const runtime = createAgentRuntimeClient(
    new AppServerConnection({
      send() {},
      async nextMessage() {
        reads += 1;
        return source;
      },
    }),
    {
      adapters: [
        () => [
          source,
          itemNotification({ method: "item/completed", status: "completed" }),
        ],
      ],
    },
  );

  const first = await runtime.nextEvent();
  const second = await runtime.nextEvent();

  assert.equal(first.method, "item/started");
  assert.equal(second.method, "item/completed");
  assert.equal(reads, 1);
});

test("direct lifecycle middleware can drop an item before listener dispatch", async () => {
  const runtime = createRuntimeClient({
    middlewares: [
      ({ event }) => (event.method === "item/started" ? false : undefined),
    ],
  });
  const received = [];
  runtime.subscribeLifecycleEvents((event) => received.push(event));

  const handled = await runtime.dispatchEvent(
    itemNotification({ type: "agentMessage" }),
  );

  assert.equal(handled, false);
  assert.deepEqual(received, []);
});

test("pipeline flush verifies buffered direct notifications", async () => {
  const completed = itemNotification({
    method: "item/completed",
    status: "completed",
  });
  const pipeline = new AgentRuntimeEventPipeline({
    middlewares: [
      {
        transform() {},
        flush() {
          return completed;
        },
      },
    ],
  });

  const processed = await pipeline.process(
    itemNotification({ method: "item/started", status: "inProgress" }),
  );
  const flushed = await pipeline.flush();

  assert.equal(processed.accepted, true);
  assert.equal(processed.notifications[0].method, "item/started");
  assert.equal(flushed.accepted, true);
  assert.equal(flushed.notifications[0].method, "item/completed");
});

test("item streaming notifications cross the pipeline without synthesizing verifier sequence", async () => {
  let pushes = 0;
  const pipeline = new AgentRuntimeEventPipeline({
    sequenceVerifier: {
      push() {
        pushes += 1;
        return [];
      },
      getViolations() {
        return [];
      },
    },
  });

  const notifications = [
    agentMessageDeltaNotification(),
    commandOutputDeltaNotification(),
    commandTerminalInteractionNotification(),
    fileChangePatchUpdatedNotification(),
    mcpToolCallProgressNotification(),
    planDeltaNotification(),
    reasoningNotification("item/reasoning/summaryTextDelta", {
      delta: "summary",
      summaryIndex: 0,
    }),
    reasoningNotification("item/reasoning/summaryPartAdded", {
      summaryIndex: 1,
    }),
    reasoningNotification("item/reasoning/textDelta", {
      contentIndex: 0,
      delta: "raw",
    }),
  ];
  const processed = [];
  for (const notification of notifications) {
    processed.push(await pipeline.process(notification));
  }

  assert.deepEqual(
    processed.map((result) => result.accepted),
    [true, true, true, true, true, true, true, true, true],
  );
  assert.deepEqual(
    processed.map((result) => result.notification.method),
    notifications.map((notification) => notification.method),
  );
  assert.equal(pushes, 0);
});

test("session gateway fails closed when optional runtime surfaces are absent", async () => {
  const runtime = createAgentRuntimeClientFromSessionGateway(
    createMinimalSessionGateway(),
  );

  await assert.rejects(
    runtime.nextEvent(),
    /does not expose direct lifecycle notifications/,
  );
});

test("session gateway propagates transport errors without mock fallback", async () => {
  const runtime = createAgentRuntimeClientFromSessionGateway({
    ...createMinimalSessionGateway(),
    async startTurn() {
      throw new Error("bridge offline");
    },
  });

  await assert.rejects(
    runtime.startTurn({
      threadId: "thread-1",
      input: [{ type: "text", text: "生成草稿" }],
    }),
    /bridge offline/,
  );
});

test("sessionGateway subpath is browser-safe and direct-lifecycle-only", async () => {
  const subpath = await import("../dist/sessionGateway.js");
  const source = await readFile(
    new URL("../dist/sessionGateway.js", import.meta.url),
    "utf8",
  );

  assert.equal(typeof subpath.AgentRuntimeEventSequenceGate, "function");
  assert.equal(typeof subpath.AgentRuntimeEventPipeline, "function");
  assert.equal(
    typeof subpath.runtimeExecutionEventFromLifecycleNotification,
    "function",
  );
  assert.equal("runtimeExecutionEventFromAgentEvent" in subpath, false);
  assert.equal("createSchemaVersionCompatibilityMiddleware" in subpath, false);
  assert.equal(source.includes("node:"), false);
  assert.equal(source.includes("sidecar"), false);
});

function createRuntimeClient(options = {}) {
  return createAgentRuntimeClient(
    new AppServerConnection({
      send() {},
      async nextMessage() {
        throw new Error("unused");
      },
    }),
    options,
  );
}

function createMinimalSessionGateway(calls = []) {
  return {
    async startTurn(params, options) {
      calls.push(["startTurn", params, options]);
      return requestResult(1, {
        turn: {
          id: "turn-1",
          status: "inProgress",
        },
      });
    },
    async steerTurn(params, options) {
      calls.push(["steerTurn", params, options]);
      return requestResult(2, { turnId: params.expectedTurnId });
    },
    async readThread(params, options) {
      calls.push(["readThread", params, options]);
      return requestResult(3, {
        thread: {
          cliVersion: "0.0.0-test",
          createdAt: 1778803200,
          cwd: "/tmp/project",
          ephemeral: false,
          id: params.threadId,
          modelProvider: "openai",
          preview: "",
          sessionId: "session-1",
          source: "appServer",
          status: { type: "active" },
          turns: [],
          updatedAt: 1778803201,
        },
      });
    },
    async cancelTurn(params, options) {
      calls.push(["cancelTurn", params, options]);
      return requestResult(4, {});
    },
    async respondAction(params, options) {
      calls.push(["respondAction", params, options]);
      return requestResult(5, {});
    },
  };
}

function requestResult(id, result) {
  return {
    id,
    result,
    response: { jsonrpc: "2.0", id, result },
    notifications: [],
    messages: [],
  };
}

function directToolLifecycle() {
  return [
    turnNotification("inProgress"),
    itemNotification({ method: "item/started", status: "inProgress" }),
    itemNotification({ method: "item/completed", status: "completed" }),
    turnNotification("completed", { completedAt: 220 }),
  ];
}

function threadStartedNotification(overrides = {}) {
  return {
    method: "thread/started",
    params: {
      thread: {
        cliVersion: "0.0.0-test",
        createdAt: 100,
        cwd: "/tmp/project",
        ephemeral: false,
        id: "thread-1",
        modelProvider: "openai",
        preview: "Research thread",
        sessionId: "session-1",
        source: "appServer",
        name: "Research",
        status: { type: "active", activeFlags: ["waitingOnApproval"] },
        updatedAt: 110,
        ...overrides,
      },
    },
  };
}

function turnNotification(status, overrides = {}) {
  return {
    method: status === "inProgress" ? "turn/started" : "turn/completed",
    params: {
      threadId: "thread-1",
      turn: {
        id: "turn-1",
        startedAt: 200,
        status,
        ...overrides,
      },
    },
  };
}

function itemNotification({
  method = "item/started",
  type = "mcpToolCall",
  status = "inProgress",
  id = type === "agentMessage"
    ? "msg_1"
    : type === "commandExecution"
      ? "command_1"
      : "tool_call_1",
  observedAtMs = method === "item/started" ? 210 : 215,
  ...overrides
} = {}) {
  let item;
  if (type === "agentMessage") {
    item = { id, type, text: "hello", ...overrides };
  } else if (type === "commandExecution") {
    item = {
      command: "npm test",
      cwd: "/tmp/project",
      id,
      status,
      type,
      ...overrides,
    };
  } else {
    item = {
      arguments: {},
      id,
      server: "test-server",
      status,
      tool: "shell",
      type,
      ...overrides,
    };
  }
  return {
    method,
    params: {
      threadId: "thread-1",
      turnId: "turn-1",
      item,
      ...(method === "item/started"
        ? { startedAtMs: observedAtMs }
        : { completedAtMs: observedAtMs }),
    },
  };
}

function agentMessageDeltaNotification(overrides = {}) {
  return {
    method: "item/agentMessage/delta",
    params: {
      delta: "hello",
      itemId: "msg_1",
      threadId: "thread-1",
      turnId: "turn-1",
      ...overrides,
    },
  };
}

function errorNotification(willRetry, overrides = {}) {
  return {
    method: "error",
    params: {
      error: { message: "provider stream reconnecting" },
      threadId: "thread-1",
      turnId: "turn-1",
      willRetry,
      ...overrides,
    },
  };
}

function turnPlanUpdatedNotification(overrides = {}) {
  return {
    method: "turn/plan/updated",
    params: {
      explanation: "继续执行",
      plan: [
        { step: "读现状", status: "completed" },
        { step: "补主链", status: "inProgress" },
      ],
      threadId: "thread-1",
      turnId: "turn-1",
      ...overrides,
    },
  };
}

function commandOutputDeltaNotification(overrides = {}) {
  return {
    method: "item/commandExecution/outputDelta",
    params: {
      delta: "stdout\n",
      itemId: "command-1",
      threadId: "thread-1",
      turnId: "turn-1",
      ...overrides,
    },
  };
}

function commandTerminalInteractionNotification(overrides = {}) {
  return {
    method: "item/commandExecution/terminalInteraction",
    params: {
      itemId: "command-1",
      processId: "unified-exec-1000",
      stdin: "sent 9 chars",
      threadId: "thread-1",
      turnId: "turn-1",
      ...overrides,
    },
  };
}

function fileChangePatchUpdatedNotification(overrides = {}) {
  return {
    method: "item/fileChange/patchUpdated",
    params: {
      changes: [
        {
          diff: "-old\n+new",
          kind: { type: "update" },
          path: "src/index.ts",
        },
      ],
      itemId: "item_patch-1",
      threadId: "thread-1",
      turnId: "turn-1",
      ...overrides,
    },
  };
}

function mcpToolCallProgressNotification(overrides = {}) {
  return {
    method: "item/mcpToolCall/progress",
    params: {
      itemId: "tool_call_1",
      message: "正在读取文档索引",
      threadId: "thread-1",
      turnId: "turn-1",
      ...overrides,
    },
  };
}

function planDeltaNotification(overrides = {}) {
  return {
    method: "item/plan/delta",
    params: {
      delta: "- [ ] verify",
      itemId: "plan-1",
      threadId: "thread-1",
      turnId: "turn-1",
      ...overrides,
    },
  };
}

function reasoningNotification(method, fields) {
  return {
    method,
    params: {
      itemId: "reasoning-1",
      threadId: "thread-1",
      turnId: "turn-1",
      ...fields,
    },
  };
}

function rawNotification(type, overrides = {}) {
  return {
    method: "agentSession/event",
    params: {
      event: {
        eventId: `event-${type}`,
        sequence: 1,
        sessionId: "session-1",
        threadId: "thread-1",
        turnId: "turn-1",
        type,
        timestamp: "2026-05-15T00:00:02.000Z",
        payload: {},
        ...overrides,
      },
    },
  };
}
