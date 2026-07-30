import path from "node:path";

function canonicalTurn({ sessionId, threadId, turn, timestampMs }) {
  return {
    sessionId,
    threadId,
    turnId: turn.turnId,
    status: "completed",
    admission: "accepted",
    queue: { state: "running" },
    approval: "notRequired",
    items: [],
    itemsView: "notLoaded",
    createdAtMs: timestampMs,
    updatedAtMs: timestampMs + 3_000,
    startedAtMs: timestampMs,
    completedAtMs: timestampMs + 3_000,
    durationMs: 3_000,
  };
}

function canonicalItem({
  sessionId,
  threadId,
  turnId,
  itemId,
  sequence,
  timestampMs,
  kind,
  payload,
  metadataSource,
}) {
  return {
    sessionId,
    threadId,
    turnId,
    itemId,
    sequence,
    ordinal: sequence,
    createdAtMs: timestampMs,
    updatedAtMs: timestampMs,
    completedAtMs: timestampMs,
    kind,
    status: "completed",
    payload,
    metadata: { source: metadataSource },
  };
}

function sqlRow(values) {
  return `(${values.join(", ")})`;
}

export function seedCanonicalHistoryThread({
  runtimeEnv,
  runSqlite,
  sqlLiteral,
  thread,
  fixture,
  metadataSource,
}) {
  const sessionId = String(thread?.sessionId || "").trim();
  const threadId = String(thread?.id || "").trim();
  if (!sessionId || !threadId) {
    throw new Error("thread/start 未返回 canonical session/thread identity");
  }
  if (
    !fixture?.title ||
    !Array.isArray(fixture.turns) ||
    !fixture.turns.length
  ) {
    throw new Error("canonical history fixture 必须提供 title 与非空 turns");
  }
  fixture.sessionId = sessionId;
  fixture.threadId = threadId;

  const sqliteRoot = path.join(
    runtimeEnv.electronUserDataDir,
    "app-server",
    "sqlite",
  );
  const statePath = path.join(sqliteRoot, "state.sqlite");
  const threadHistoryPath = path.join(sqliteRoot, "thread_history.sqlite");
  const turnStepMs = 5_000;
  const baseTimestampMs = Date.now() - (fixture.turns.length + 1) * turnStepMs;
  let sequence = 0;
  const turns = [];
  const items = [];

  for (const [index, turn] of fixture.turns.entries()) {
    const turnTimestampMs = baseTimestampMs + index * turnStepMs;
    const canonical = canonicalTurn({
      sessionId,
      threadId,
      turn,
      timestampMs: turnTimestampMs,
    });
    turns.push({ ordinal: index + 1, lastSequence: sequence + 3, canonical });
    items.push(
      canonicalItem({
        sessionId,
        threadId,
        turnId: turn.turnId,
        itemId: `item_user-${turn.turnId}`,
        sequence: ++sequence,
        timestampMs: turnTimestampMs + 500,
        kind: "userMessage",
        payload: {
          type: "userMessage",
          content: turn.userInputs,
          client_id: `client-${turn.turnId}`,
        },
        metadataSource,
      }),
      canonicalItem({
        sessionId,
        threadId,
        turnId: turn.turnId,
        itemId: turn.reasoningItemId,
        sequence: ++sequence,
        timestampMs: turnTimestampMs + 1_500,
        kind: "reasoning",
        payload: {
          type: "reasoning",
          summary: [turn.reasoningText],
          content: [turn.reasoningText],
        },
        metadataSource,
      }),
      canonicalItem({
        sessionId,
        threadId,
        turnId: turn.turnId,
        itemId: turn.assistantItemId,
        sequence: ++sequence,
        timestampMs: turnTimestampMs + 2_500,
        kind: "agentMessage",
        payload: {
          type: "agentMessage",
          text: turn.assistantText,
          phase: "final_answer",
        },
        metadataSource,
      }),
    );
  }

  const turnRows = turns
    .map(({ ordinal, lastSequence, canonical }) =>
      sqlRow([
        sqlLiteral(threadId),
        sqlLiteral(canonical.turnId),
        ordinal,
        lastSequence,
        sqlLiteral(JSON.stringify(canonical)),
      ]),
    )
    .join(",\n  ");
  const itemRows = items
    .map((item) =>
      sqlRow([
        sqlLiteral(threadId),
        sqlLiteral(item.turnId),
        sqlLiteral(item.itemId),
        item.ordinal,
        item.sequence,
        sqlLiteral(JSON.stringify(item)),
      ]),
    )
    .join(",\n  ");
  const updatedAtMs =
    baseTimestampMs + (fixture.turns.length - 1) * turnStepMs + 3_000;
  const latestTurn = fixture.turns.at(-1);

  runSqlite(
    statePath,
    `
PRAGMA busy_timeout = 5000;
ATTACH DATABASE ${sqlLiteral(threadHistoryPath)} AS thread_history;
BEGIN IMMEDIATE;
DELETE FROM thread_history.canonical_items WHERE thread_id = ${sqlLiteral(threadId)};
DELETE FROM thread_history.canonical_turns WHERE thread_id = ${sqlLiteral(threadId)};
INSERT INTO thread_history.canonical_turns (
  thread_id, turn_id, ordinal, last_sequence, turn_json
) VALUES
  ${turnRows};
INSERT INTO thread_history.canonical_items (
  thread_id, turn_id, item_id, ordinal, sequence, item_json
) VALUES
  ${itemRows};
UPDATE canonical_threads
SET thread_json = json_set(
      thread_json,
      '$.status', json('{"type":"idle"}'),
      '$.updatedAtMs', ${updatedAtMs},
      '$.recencyAtMs', ${updatedAtMs},
      '$.preview', ${sqlLiteral(latestTurn.assistantText)},
      '$.name', ${sqlLiteral(fixture.title)}
    ),
    updated_at_ms = ${updatedAtMs},
    recency_at_ms = ${updatedAtMs},
    last_sequence = ${sequence}
WHERE thread_id = ${sqlLiteral(threadId)};
COMMIT;
DETACH DATABASE thread_history;
`,
  );

  return {
    statePath,
    threadHistoryPath,
    sessionId,
    threadId,
    rolloutPath: thread.path ?? null,
    turnCount: turns.length,
    itemCount: items.length,
    turnIds: fixture.turns.map((turn) => turn.turnId),
    itemIds: items.map((item) => item.itemId),
  };
}
