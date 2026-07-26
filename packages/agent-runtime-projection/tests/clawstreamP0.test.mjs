import test from "node:test";
import assert from "node:assert/strict";

import {
  projectAgentUiState,
  replayAppServerFacts,
} from "../dist/index.js";

function event(overrides) {
  return {
    id: overrides.id,
    kind: overrides.kind ?? "state",
    status: overrides.status ?? "running",
    eventClass: overrides.eventClass,
    title: overrides.title ?? overrides.eventClass,
    turnId: overrides.turnId ?? "turn-clawstream-p0",
    sequence: overrides.sequence,
    createdAt: `2026-07-06T00:00:${String(overrides.sequence ?? 0).padStart(2, "0")}.000Z`,
    ...overrides,
  };
}

test("clawstream startup-prewarm-first-output keeps startup noise out of visible messages", () => {
  const state = projectAgentUiState({
    executionEvents: [
      event({
        id: "evt-turn-started",
        eventClass: "turn.started",
        title: "启动处理流程",
        sequence: 1,
      }),
      event({
        id: "evt-prewarm",
        eventClass: "runtime.prewarm.started",
        title: "已接收请求",
        sequence: 2,
      }),
      event({
        id: "evt-reasoning",
        kind: "note",
        eventClass: "reasoning.delta",
        title: "思考",
        detail: "先判断请求形状。",
        sequence: 3,
      }),
      event({
        id: "evt-text",
        kind: "model",
        eventClass: "model.delta",
        title: "模型输出",
        payload: { messageId: "msg-answer", text: "正文首字" },
        sequence: 4,
      }),
    ],
  });

  assert.deepEqual(
    state.messages.map((part) => [part.type, part.text]),
    [
      ["reasoning", "先判断请求形状。"],
      ["text", "正文首字"],
    ],
  );
  assert.equal(
    state.messages.some((part) =>
      /启动处理流程|已接收请求/.test(part.text ?? ""),
    ),
    false,
  );
  assert.equal(state.runtime.status, "running");
});

test("clawstream terminal-contract-after-answer clears running status without synthesizing text", () => {
  const state = projectAgentUiState({
    executionEvents: [
      event({
        id: "evt-turn-started",
        eventClass: "turn.started",
        title: "开始执行",
        sequence: 1,
      }),
      event({
        id: "evt-reasoning",
        kind: "note",
        eventClass: "reasoning.delta",
        title: "思考",
        detail: "先核对工具结果。",
        sequence: 2,
      }),
      event({
        id: "evt-answer",
        kind: "model",
        eventClass: "model.delta",
        title: "模型输出",
        payload: { messageId: "msg-answer", text: "最终正文。" },
        sequence: 3,
      }),
      event({
        id: "evt-turn-completed",
        eventClass: "turn.completed",
        status: "completed",
        title: "完成",
        sequence: 4,
      }),
    ],
  });

  assert.deepEqual(
    state.messages.map((part) => [part.type, part.text]),
    [
      ["reasoning", "先核对工具结果。"],
      ["text", "最终正文。"],
    ],
  );
  assert.equal(state.runtime.status, "completed");
  assert.equal(state.runtime.latestEventId, "evt-turn-completed");
});

test("clawstream terminal without assistant text fails closed instead of fabricating final answer", () => {
  const state = projectAgentUiState({
    executionEvents: [
      event({
        id: "evt-turn-started",
        eventClass: "turn.started",
        title: "开始执行",
        sequence: 1,
      }),
      event({
        id: "evt-tool-started",
        kind: "tool",
        eventClass: "tool.started",
        status: "running",
        title: "运行工具",
        toolCallId: "tool-1",
        sequence: 2,
      }),
      event({
        id: "evt-tool-result",
        kind: "tool",
        eventClass: "tool.result",
        status: "completed",
        title: "工具完成",
        detail: "工具输出",
        toolCallId: "tool-1",
        sequence: 3,
      }),
      event({
        id: "evt-turn-completed",
        eventClass: "turn.completed",
        status: "completed",
        title: "完成",
        sequence: 4,
      }),
    ],
  });

  assert.deepEqual(
    state.messages.map((part) => [part.type, part.text]),
    [["tool-preview", "工具输出"]],
  );
  assert.equal(state.runtime.status, "completed");
});

test("tool-unified-exec-wait: waiting observation and final message coexist in event order", () => {
  const result = replayAppServerFacts({
    events: [
      {
        eventId: "event-unified-exec-wait",
        sessionId: "session-unified-exec-wait",
        threadId: "thread-unified-exec-wait",
        turnId: "turn-unified-exec-wait",
        sequence: 1,
        timestamp: "2026-07-06T00:00:01.000Z",
        type: "tool.result",
        payload: {
          toolCallId: "call-unified-exec-wait",
          toolName: "write_stdin",
          detail: "Waiting for background terminal",
          status: "completed",
          metadata: {
            unified_exec_observation: "waiting",
            unified_exec_empty_poll_count: 2,
            session_id: 1000,
          },
        },
      },
      {
        eventId: "event-unified-exec-final-message",
        sessionId: "session-unified-exec-wait",
        threadId: "thread-unified-exec-wait",
        turnId: "turn-unified-exec-wait",
        sequence: 2,
        timestamp: "2026-07-06T00:00:02.000Z",
        type: "message.completed",
        payload: {
          messageId: "message-unified-exec-final",
          text: "Final response after the background terminal wait.",
          status: "completed",
        },
      },
      {
        eventId: "event-unified-exec-turn-completed",
        sessionId: "session-unified-exec-wait",
        threadId: "thread-unified-exec-wait",
        turnId: "turn-unified-exec-wait",
        sequence: 3,
        timestamp: "2026-07-06T00:00:03.000Z",
        type: "turn.completed",
        payload: { status: "completed" },
      },
    ],
  });

  const waitEvent = result.events.find(
    (event) => event.id === "appserver:event-unified-exec-wait",
  );
  assert.equal(
    waitEvent?.payload?.metadata?.unified_exec_observation,
    "waiting",
  );
  assert.equal(waitEvent?.payload?.metadata?.unified_exec_empty_poll_count, 2);
  assert.deepEqual(
    result.state.messages.map((part) => [part.type, part.text]),
    [
      ["tool-preview", "Waiting for background terminal"],
      ["text", "Final response after the background terminal wait."],
    ],
  );
  assert.equal(result.state.runtime.status, "completed");
});

test("approval-parallel-aggregate: one terminal review keeps the other request waiting", () => {
  const pendingEvents = [
    event({
      id: "evt-turn-started-parallel-approval",
      eventClass: "turn.started",
      title: "开始执行",
      sequence: 1,
    }),
    event({
      id: "evt-approval-required-a",
      kind: "action",
      eventClass: "action.required",
      status: "blocked",
      title: "Review first approval",
      actionId: "approval-a",
      payload: { actionKind: "tool_confirmation" },
      sequence: 2,
    }),
    event({
      id: "evt-approval-required-b",
      kind: "action",
      eventClass: "action.required",
      status: "blocked",
      title: "Review second approval",
      actionId: "approval-b",
      payload: { actionKind: "tool_confirmation" },
      sequence: 3,
    }),
  ];
  const approvalAResolved = event({
    id: "evt-approval-resolved-a",
    kind: "action",
    eventClass: "action.resolved",
    status: "completed",
    title: "First approval resolved",
    actionId: "approval-a",
    sequence: 4,
  });
  const partiallyResolved = projectAgentUiState({
    executionEvents: [...pendingEvents, approvalAResolved],
  });

  assert.equal(partiallyResolved.runtime.status, "waiting");
  assert.deepEqual(
    partiallyResolved.readModel.pendingActions.map((action) => action.actionId),
    ["approval-b"],
  );
  assert.equal(
    partiallyResolved.readModel.events.find(
      (action) => action.actionId === "approval-a",
    )?.resolved,
    true,
  );

  const completed = projectAgentUiState({
    executionEvents: [
      ...pendingEvents,
      approvalAResolved,
      event({
        id: "evt-approval-expired-b",
        kind: "action",
        eventClass: "action.expired",
        status: "failed",
        title: "Second approval timed out",
        detail: "Automatic approval review timed out.",
        actionId: "approval-b",
        sequence: 5,
      }),
      event({
        id: "evt-turn-completed-parallel-approval",
        eventClass: "turn.completed",
        status: "completed",
        title: "完成",
        sequence: 6,
      }),
    ],
  });

  assert.deepEqual(completed.readModel.pendingActions, []);
  assert.equal(completed.runtime.status, "completed");
  assert.equal(
    completed.diagnostics.some(
      (diagnostic) => diagnostic.id === "evt-approval-expired-b",
    ),
    true,
  );
});

test("clawstream stream-parser-boundary keeps completed full text out of a duplicate finish tail", () => {
  const result = replayAppServerFacts({
    events: [
      {
        eventId: "event-output-item-added-seed",
        sessionId: "session-clawstream-p0",
        threadId: "thread-clawstream-p0",
        turnId: "turn-clawstream-p0",
        sequence: 1,
        timestamp: "2026-07-06T00:00:01.000Z",
        type: "message.delta",
        payload: {
          messageId: "message-seeded-answer",
          text: "先给",
          status: "streaming",
        },
      },
      {
        eventId: "event-message-delta-rest",
        sessionId: "session-clawstream-p0",
        threadId: "thread-clawstream-p0",
        turnId: "turn-clawstream-p0",
        sequence: 2,
        timestamp: "2026-07-06T00:00:02.000Z",
        type: "message.delta",
        payload: {
          messageId: "message-seeded-answer",
          delta: "出结论。",
          status: "streaming",
        },
      },
      {
        eventId: "event-message-completed-full-text",
        sessionId: "session-clawstream-p0",
        threadId: "thread-clawstream-p0",
        turnId: "turn-clawstream-p0",
        sequence: 3,
        timestamp: "2026-07-06T00:00:03.000Z",
        type: "message.completed",
        payload: {
          messageId: "message-seeded-answer",
          text: "先给出结论。",
          status: "completed",
        },
      },
      {
        eventId: "event-turn-completed",
        sessionId: "session-clawstream-p0",
        threadId: "thread-clawstream-p0",
        turnId: "turn-clawstream-p0",
        sequence: 4,
        timestamp: "2026-07-06T00:00:04.000Z",
        type: "turn.completed",
        payload: {
          status: "completed",
        },
      },
    ],
  });

  assert.deepEqual(
    result.state.messages.map((part) => [
      part.type,
      part.text,
      part.state,
      part.sourceEventId,
    ]),
    [
      [
        "text",
        "先给出结论。",
        "final",
        "appserver:event-message-completed-full-text",
      ],
    ],
  );
  assert.equal(result.state.runtime.status, "completed");
  assert.equal(
    result.state.messages.some((part) =>
      /先给出结论。先给出结论。|先给出结论。先给出结论/.test(part.text ?? ""),
    ),
    false,
  );
});

test("turn-stream-repair: completed item repairs a dropped middle delta once", () => {
  const completedText =
    "The transport kept this.\nAnd dropped this.\n\n- Finding — /tmp/file.rs:1\n  Keep ::git-stage{cwd=/tmp} literal.";
  const result = replayAppServerFacts({
    events: [
      {
        eventId: "event-message-delta-prefix",
        sessionId: "session-turn-stream-repair",
        threadId: "thread-turn-stream-repair",
        turnId: "turn-turn-stream-repair",
        sequence: 1,
        timestamp: "2026-07-06T00:00:01.000Z",
        type: "message.delta",
        payload: {
          messageId: "message-turn-stream-repair",
          text: "The transport kept this.\n",
          status: "streaming",
        },
      },
      {
        eventId: "event-item-completed-repair",
        sessionId: "session-turn-stream-repair",
        threadId: "thread-turn-stream-repair",
        turnId: "turn-turn-stream-repair",
        sequence: 2,
        timestamp: "2026-07-06T00:00:03.000Z",
        type: "item.completed",
        payload: {
          messageId: "message-turn-stream-repair",
          text: completedText,
          status: "completed",
        },
      },
      {
        eventId: "event-turn-completed-repair",
        sessionId: "session-turn-stream-repair",
        threadId: "thread-turn-stream-repair",
        turnId: "turn-turn-stream-repair",
        sequence: 3,
        timestamp: "2026-07-06T00:00:04.000Z",
        type: "turn.completed",
        payload: { status: "completed" },
      },
    ],
  });

  assert.deepEqual(
    result.state.messages.map((part) => [
      part.type,
      part.messageId,
      part.text,
      part.state,
      part.sourceEventId,
    ]),
    [
      [
        "text",
        "message-turn-stream-repair",
        completedText,
        "final",
        "appserver:event-item-completed-repair",
      ],
    ],
  );
  assert.equal(result.state.runtime.status, "completed");
  assert.equal(
    result.state.messages.filter(
      (part) => part.messageId === "message-turn-stream-repair",
    ).length,
    1,
  );
  assert.equal(
    result.state.messages[0].text.includes(
      "The transport kept this.\nThe transport kept this.",
    ),
    false,
  );
});

test("clawstream plan-parser-boundary materializes proposed_plan across stream boundaries", () => {
  const result = replayAppServerFacts({
    events: [
      {
        eventId: "event-plan-added-prefix",
        sessionId: "session-clawstream-p0",
        threadId: "thread-clawstream-p0",
        turnId: "turn-clawstream-p0",
        sequence: 1,
        timestamp: "2026-07-06T00:00:01.000Z",
        type: "message.delta",
        payload: {
          messageId: "message-plan-answer",
          text: "前置说明：\n<proposed_",
          status: "streaming",
        },
      },
      {
        eventId: "event-plan-delta-open",
        sessionId: "session-clawstream-p0",
        threadId: "thread-clawstream-p0",
        turnId: "turn-clawstream-p0",
        sequence: 2,
        timestamp: "2026-07-06T00:00:02.000Z",
        type: "message.delta",
        payload: {
          messageId: "message-plan-answer",
          delta: "plan>\n- 读取资料",
          status: "streaming",
        },
      },
      {
        eventId: "event-plan-delta-close",
        sessionId: "session-clawstream-p0",
        threadId: "thread-clawstream-p0",
        turnId: "turn-clawstream-p0",
        sequence: 3,
        timestamp: "2026-07-06T00:00:03.000Z",
        type: "message.delta",
        payload: {
          messageId: "message-plan-answer",
          delta: "\n- 输出结论\n</proposed_plan>\n后续正文。",
          status: "streaming",
        },
      },
      {
        eventId: "event-plan-completed-full-text",
        sessionId: "session-clawstream-p0",
        threadId: "thread-clawstream-p0",
        turnId: "turn-clawstream-p0",
        sequence: 4,
        timestamp: "2026-07-06T00:00:04.000Z",
        type: "message.completed",
        payload: {
          messageId: "message-plan-answer",
          text: "前置说明：\n<proposed_plan>\n- 读取资料\n- 输出结论\n</proposed_plan>\n后续正文。",
          status: "completed",
        },
      },
    ],
  });

  assert.deepEqual(
    result.state.messages.map((part) => [
      part.type,
      part.text?.trim(),
      part.state,
    ]),
    [
      ["text", "前置说明：", "final"],
      ["plan", "- 读取资料\n- 输出结论", "final"],
      ["text", "后续正文。", "final"],
    ],
  );
  assert.equal(
    result.state.messages.some((part) =>
      /<proposed_plan>|<\/proposed_plan>/.test(part.text ?? ""),
    ),
    false,
  );
});

test("clawstream plan-parser-boundary projects structured plan events as plan items", () => {
  const result = replayAppServerFacts({
    events: [
      {
        eventId: "event-plan-final-structured",
        sessionId: "session-clawstream-p0",
        threadId: "thread-clawstream-p0",
        turnId: "turn-clawstream-p0",
        sequence: 1,
        timestamp: "2026-07-06T00:00:01.000Z",
        type: "plan.final",
        payload: {
          plan: [
            { step: "核对实现", status: "completed" },
            { step: "补齐回归", status: "in_progress" },
          ],
          status: "completed",
          revisionId: "proposed_plan:turn-clawstream-p0",
        },
      },
    ],
  });

  assert.deepEqual(
    result.state.messages.map((part) => [
      part.type,
      part.text,
      part.state,
      part.sourceEventId,
    ]),
    [
      [
        "plan",
        "- [completed] 核对实现\n- [in_progress] 补齐回归",
        "final",
        "appserver:event-plan-final-structured",
      ],
    ],
  );
});
