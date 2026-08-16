import assert from "node:assert/strict";
import test from "node:test";

import * as projection from "../dist/index.js";
import {
  projectAppServerEventsToExecutionEvents,
  projectAppServerSessionReadToExecutionEvents,
  replayAppServerFacts,
} from "../dist/index.js";

const timestamp = "2026-06-10T00:00:00.000Z";

test("App Server projection exposes only canonical read-model and lifecycle inputs", () => {
  assert.equal(
    "projectAppServerEvidenceExportToExecutionEvents" in projection,
    false,
  );
});

test("App Server events replay into standard projection surfaces", () => {
  const result = replayAppServerFacts({
    events: [
      appServerEvent("evt-message", 1, "message.delta", {
        text: "你好，Lime。",
        messageId: "msg-1",
      }),
      appServerEvent("evt-tool", 2, "tool.started", {
        toolCallId: "tool-1",
        toolName: "search",
      }),
      appServerEvent("evt-action", 3, "action.required", {
        request_id: "request-1",
        action_type: "tool_confirmation",
      }),
      appServerEvent("evt-artifact", 4, "artifact.snapshot", {
        artifactRef: "artifact-1",
        artifactId: "artifact-1",
        path: "draft.md",
      }),
      appServerEvent("evt-evidence", 5, "evidence.changed", {
        evidenceRef: "evidence-1",
      }),
      appServerEvent("evt-completed", 6, "turn.completed", {
        text: "完成",
      }),
    ],
  });

  assert.deepEqual(result.diagnostics, []);
  assert.equal(result.state.runtime.status, "completed");
  assert.equal(result.state.messages[0].text, "你好，Lime。");
  assert.equal(result.state.tools.length, 1);
  assert.equal(result.state.readModel.pendingActions.length, 1);
  assert.deepEqual(
    result.state.readModel.pendingActions[0].actions.map(
      (action) => action.decision,
    ),
    ["approve", "reject"],
  );
  assert.deepEqual(
    result.state.artifacts.map((artifact) => artifact.id),
    ["artifact-1"],
  );
  assert.deepEqual(
    result.state.evidence.map((evidence) => evidence.id),
    ["evidence-1"],
  );
});

test("App Server read model hydrates snapshot and turn lifecycle facts", () => {
  const events = projectAppServerSessionReadToExecutionEvents({
    session: {
      sessionId: "session-1",
      threadId: "thread-1",
      appId: "content-studio",
      status: "running",
      createdAt: timestamp,
      updatedAt: "2026-06-10T00:00:02.000Z",
    },
    turns: [
      {
        turnId: "turn-1",
        sessionId: "session-1",
        threadId: "thread-1",
        status: "completed",
        startedAt: timestamp,
        completedAt: "2026-06-10T00:00:01.000Z",
      },
      {
        turnId: "turn-2",
        sessionId: "session-1",
        threadId: "thread-1",
        status: "waitingAction",
        startedAt: "2026-06-10T00:00:02.000Z",
      },
    ],
  });

  assert.equal(events.length, 3);
  assert.equal(events[0].eventClass, "snapshot.updated");
  assert.equal(events[1].eventClass, "turn.completed");
  assert.equal(events[2].eventClass, "action.required");

  const result = replayAppServerFacts({
    readModel: {
      session: {
        sessionId: "session-1",
        threadId: "thread-1",
        status: "running",
        updatedAt: timestamp,
      },
      turns: [
        {
          turnId: "turn-2",
          status: "waitingAction",
        },
      ],
    },
  });

  assert.equal(result.state.runtime.status, "waiting");
  assert.equal(result.state.readModel.pendingActions.length, 1);
  assert.equal(
    result.state.graph.some(
      (node) => node.nodeId === "turn-2" && node.nodeType === "turn",
    ),
    true,
  );
});

test("App Server event adapter keeps projection package runtime-client free", () => {
  const events = projectAppServerEventsToExecutionEvents([
    appServerEvent("evt-1", 1, "turn.failed", {
      message: "boom",
    }),
  ]);

  assert.equal(events[0].eventClass, "turn.failed");
  assert.equal(events[0].status, "failed");
  assert.equal(events[0].detail, "boom");
});

test("App Server facts preserve canceled turn terminal instead of completing it", () => {
  const events = projectAppServerEventsToExecutionEvents([
    appServerEvent("evt-cancel", 1, "turn.canceled", {
      status: "canceled",
      message: "stopped by user",
    }),
  ]);

  assert.equal(events[0].eventClass, "turn.canceled");
  assert.equal(events[0].status, "canceled");
  assert.equal(events[0].phase, "canceled");
  assert.equal(events[0].completedAt, timestamp);
});

test("App Server facts do not promote legacy final_done to current terminal", () => {
  const events = projectAppServerEventsToExecutionEvents([
    appServerEvent("evt-legacy-final", 1, "turn.final_done"),
  ]);

  assert.equal(events[0].eventClass, "turn.final_done");
  assert.equal(events[0].status, "running");
  assert.equal(events[0].completedAt, undefined);
});

test("App Server facts ignore legacy terminal payload status", () => {
  const events = projectAppServerEventsToExecutionEvents([
    appServerEvent("evt-legacy-done", 1, "done", { status: "done" }),
    appServerEvent("evt-legacy-final", 2, "turn.final_done", {
      status: "completed",
    }),
    appServerEvent("evt-legacy-cancelled", 3, "turn.cancelled", {
      status: "cancelled",
    }),
  ]);

  assert.deepEqual(
    events.map((event) => event.eventClass),
    ["done", "turn.final_done", "turn.cancelled"],
  );
  assert.deepEqual(
    events.map((event) => event.status),
    ["running", "running", "running"],
  );
  assert.deepEqual(
    events.map((event) => event.completedAt),
    [undefined, undefined, undefined],
  );
});

test("App Server tool events normalize Soul lifecycle metadata for package projection", () => {
  const events = projectAppServerEventsToExecutionEvents([
    appServerEvent("evt-tool-progress", 1, "tool.progress", {
      toolCallId: "tool-1",
      metadata: {
        soul_lifecycle: {
          surface: "tool_lifecycle",
          phase: "tool_progress",
          styleLevel: "L1",
          riskLevel: "high",
          profileId: "calm_professional_partner",
          packId: "com.lime.soul.calm-professional-partner",
        },
        tool_process_facts: {
          source: "runtime_facts",
          toolCallId: "tool-1",
          status: "progress",
          riskLevel: "high",
        },
      },
    }),
  ]);

  assert.equal(events[0].eventClass, "tool.progress");
  assert.equal(events[0].toolCallId, "tool-1");
  assert.deepEqual(events[0].payload?.soulLifecycle, {
    surface: "tool_lifecycle",
    phase: "tool_progress",
    styleLevel: "L1",
    riskLevel: "high",
    profileId: "calm_professional_partner",
    packId: "com.lime.soul.calm-professional-partner",
  });
  assert.deepEqual(events[0].payload?.toolProcessFacts, {
    source: "runtime_facts",
    toolCallId: "tool-1",
    status: "progress",
    riskLevel: "high",
  });
  assert.equal(events[0].payload?.soulSurface, "tool_lifecycle");
  assert.equal(events[0].payload?.soulPhase, "tool_progress");
  assert.equal(events[0].payload?.styleLevel, "L1");
  assert.equal(events[0].payload?.riskLevel, "high");
  assert.equal(events[0].payload?.profileId, "calm_professional_partner");
  assert.equal(
    events[0].payload?.packId,
    "com.lime.soul.calm-professional-partner",
  );
});

test("App Server tool process facts drive shared tool surface classification", () => {
  const result = replayAppServerFacts({
    events: [
      appServerEvent("evt-tool-started", 1, "tool.started", {
        toolCallId: "tool-1",
        toolName: "RuntimeProvidedTool",
        metadata: {
          tool_process_facts: {
            source: "runtime_facts",
            toolCallId: "tool-1",
            toolName: "RuntimeProvidedTool",
            toolFamily: "search",
            operationKind: "web_search",
            subject: "Soul output surface",
          },
        },
      }),
    ],
  });

  assert.equal(result.state.toolCalls.calls.length, 1);
  assert.equal(result.state.toolCalls.calls[0].family, "webSearch");
  assert.equal(result.state.toolCalls.calls[0].operationKind, "web_search");
});

test("App Server replay rejects retired raw subagent status events", () => {
  const events = projectAppServerEventsToExecutionEvents([
    appServerEvent("evt-subagent", 1, "subagent.status_changed", {
      status: "running",
      subagentId: "child-1",
      taskId: "child-1",
      parentThreadId: "thread-1",
      transcriptRef: "child-1:turn-1",
      metadata: {
        soul_lifecycle: {
          profileId: "cheeky_sassy_executor",
          packId: "com.lime.soul.cheeky-sassy-executor",
          toneVariant: "cheeky_sassy",
        },
      },
    }),
  ]);

  assert.deepEqual(events, []);
});

function appServerEvent(eventId, sequence, type, payload = {}) {
  return {
    eventId,
    sequence,
    sessionId: "session-1",
    threadId: "thread-1",
    turnId: "turn-1",
    type,
    timestamp,
    payload,
  };
}
