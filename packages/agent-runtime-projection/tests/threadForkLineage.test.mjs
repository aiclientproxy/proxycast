import test from "node:test";
import assert from "node:assert/strict";

import {
  buildCodexThreadForkLineageProjectionEvent,
  extractCodexThreadForkLineageSnapshot,
} from "../dist/index.js";

function turn(id, itemIds) {
  return {
    id,
    status: "completed",
    items: itemIds.map((itemId) => ({ id: itemId })),
  };
}

function forkLineageInput(overrides = {}) {
  return {
    sourceThreadId: "thread-parent",
    forkRequest: {
      threadId: "thread-parent",
    },
    forkResponse: {
      thread: {
        id: "thread-fork",
        sessionId: "thread-fork",
        forkedFromId: "thread-parent",
        parentThreadId: null,
        preview: "Saved user message",
        turns: [turn("turn-1", ["item-user-1"])],
      },
    },
    readResponse: {
      thread: {
        id: "thread-fork",
        forkedFromId: "thread-parent",
      },
    },
    sourceTurnOrder: ["turn-1"],
    sidebarEntries: [
      {
        threadId: "thread-fork",
        forkedFromId: "thread-parent",
        forkedAtTurnId: "turn-1",
      },
    ],
    historyEvents: [
      {
        threadId: "thread-fork",
        forkedFromId: "thread-parent",
        forkedAtTurnId: "turn-1",
        parentTitle: "Saved user message",
      },
    ],
    sourceRolloutUnchanged: true,
    ...overrides,
  };
}

test("thread fork lineage keeps parent binding across read, sidebar, history and evidence", () => {
  const event = buildCodexThreadForkLineageProjectionEvent(
    forkLineageInput(),
    {
      sequence: 141,
      timestamp: "2026-07-09T00:00:00.000Z",
    },
  );

  assert.deepEqual(
    {
      type: event.type,
      sourceType: event.sourceType,
      sequence: event.sequence,
      sessionId: event.sessionId,
      threadId: event.threadId,
      turnId: event.turnId,
      owner: event.owner,
      scope: event.scope,
      phase: event.phase,
      surface: event.surface,
      persistence: event.persistence,
      control: event.control,
      runtimeStatus: event.runtimeStatus,
    },
    {
      type: "context.changed",
      sourceType: "thread_fork_lineage_projection",
      sequence: 141,
      sessionId: "thread-fork",
      threadId: "thread-fork",
      turnId: "turn-1",
      owner: "context",
      scope: "thread",
      phase: "completed",
      surface: "timeline_evidence",
      persistence: "snapshot",
      control: "open_detail",
      runtimeStatus: "completed",
    },
  );
  assert.equal(event.payload.sourceThreadId, "thread-parent");
  assert.equal(event.payload.forkedFromId, "thread-parent");
  assert.equal(event.payload.lineageComplete, true);
  assert.deepEqual(event.payload.parentItemIds, ["item-user-1"]);
  assert.deepEqual(event.payload.validationIssues, []);
});

test("thread fork lineage at lastTurnId keeps only the terminal prefix", () => {
  const snapshot = extractCodexThreadForkLineageSnapshot(
    forkLineageInput({
      forkRequest: {
        threadId: "thread-parent",
        lastTurnId: "turn-2",
      },
      forkResponse: {
        thread: {
          id: "thread-fork",
          sessionId: "thread-fork",
          forkedFromId: "thread-parent",
          turns: [turn("turn-1", ["item-1"]), turn("turn-2", ["item-2"])],
        },
      },
      readResponse: {
        thread: {
          id: "thread-fork",
          forkedFromId: "thread-parent",
        },
      },
      sourceTurnOrder: ["turn-1", "turn-2", "turn-3"],
      sidebarEntries: [
        {
          threadId: "thread-fork",
          forkedFromId: "thread-parent",
          forkedAtTurnId: "turn-2",
        },
      ],
      historyEvents: [
        {
          threadId: "thread-fork",
          forkedFromId: "thread-parent",
          forkedAtTurnId: "turn-2",
        },
      ],
    }),
  );

  assert.deepEqual(snapshot.copiedTurnIds, ["turn-1", "turn-2"]);
  assert.equal(snapshot.forkedAtTurnId, "turn-2");
  assert.deepEqual(snapshot.validationIssues, []);
});

test("thread fork lineage fails when forkedFromId is missing or read model loses it", () => {
  const missingForkedFrom = extractCodexThreadForkLineageSnapshot(
    forkLineageInput({
      forkResponse: {
        thread: {
          id: "thread-fork",
          sessionId: "thread-fork",
          parentThreadId: "thread-parent",
          turns: [turn("turn-1", ["item-user-1"])],
        },
      },
      readResponse: {
        thread: {
          id: "thread-fork",
        },
      },
    }),
  );

  assert.deepEqual(
    missingForkedFrom.validationIssues.map((item) => item.code),
    ["missing_forked_from_id", "parent_thread_id_confused_with_forked_from_id"],
  );

  const readModelLost = extractCodexThreadForkLineageSnapshot(
    forkLineageInput({
      readResponse: {
        thread: {
          id: "thread-fork",
        },
      },
    }),
  );

  assert.deepEqual(
    readModelLost.validationIssues.map((item) => item.code),
    ["read_model_lost_forked_from_id"],
  );
});

test("thread fork lineage fails when lastTurnId suffix leaks into the fork", () => {
  const snapshot = extractCodexThreadForkLineageSnapshot(
    forkLineageInput({
      forkRequest: {
        threadId: "thread-parent",
        lastTurnId: "turn-2",
      },
      forkResponse: {
        thread: {
          id: "thread-fork",
          sessionId: "thread-fork",
          forkedFromId: "thread-parent",
          turns: [
            turn("turn-1", ["item-1"]),
            turn("turn-2", ["item-2"]),
            turn("turn-3", ["item-3"]),
          ],
        },
      },
      readResponse: {
        thread: {
          id: "thread-fork",
          forkedFromId: "thread-parent",
        },
      },
      sourceTurnOrder: ["turn-1", "turn-2", "turn-3"],
    }),
  );

  assert.deepEqual(
    snapshot.validationIssues.map((item) => item.code),
    ["copied_turn_suffix_leaked"],
  );
});

test("thread fork lineage requires sidebar and history surfaces", () => {
  const snapshot = extractCodexThreadForkLineageSnapshot(
    forkLineageInput({
      sidebarEntries: [],
      historyEvents: [],
      renderedPlainThread: true,
    }),
  );

  assert.equal(snapshot.lineageComplete, false);
  assert.deepEqual(
    snapshot.validationIssues.map((item) => item.code),
    [
      "sidebar_lineage_missing",
      "history_lineage_missing",
      "fork_rendered_as_plain_thread",
    ],
  );
});

test("thread fork lineage fails if source rollout mutates or parent item ids are lost", () => {
  const snapshot = extractCodexThreadForkLineageSnapshot(
    forkLineageInput({
      forkResponse: {
        thread: {
          id: "thread-fork",
          sessionId: "thread-fork",
          forkedFromId: "thread-parent",
          turns: [turn("turn-1", [])],
        },
      },
      parentItemIds: [],
      sourceRolloutUnchanged: false,
    }),
  );

  assert.deepEqual(
    snapshot.validationIssues.map((item) => item.code),
    ["parent_item_ids_missing", "source_rollout_mutated"],
  );
});
