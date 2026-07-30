import type { ThreadResumeResponse } from "@limecloud/app-server-client";
import { readCanonicalThreadDetail } from "../appServerCanonicalThreadProjection";
import type {
  ConversationProjectionReducer,
  ConversationProjectionStatus,
} from "./contracts";
import { RENDER_PROJECTION_REFERENCE_REVISION } from "./protocolDrift";
import { createConversationProjectionReducer } from "./reducer";

const THREAD_RESUME_METHOD = "thread/resume";

export function createThreadResumeConversationProjection(
  response: ThreadResumeResponse,
): ConversationProjectionReducer | null {
  const threadId = response.thread.id.trim();
  if (!threadId) {
    return null;
  }

  const detail = readCanonicalThreadDetail({
    thread: {
      ...response.thread,
      turns: mergeResumeTurns(
        response.thread.turns ?? [],
        response.initialTurnsPage?.data ?? [],
      ),
    },
  });
  if (!detail || detail.thread_id !== threadId) {
    return null;
  }

  const reducer = createConversationProjectionReducer({ threadId });
  for (const turn of detail.turns ?? []) {
    reducer.dispatch({
      type: turn.status === "running" ? "turn_started" : "turn_completed",
      source: "replay",
      event_id: `thread-resume:turn:${threadId}:${turn.id}`,
      protocol_method: THREAD_RESUME_METHOD,
      protocol_revision: RENDER_PROJECTION_REFERENCE_REVISION,
      turn,
    });
  }
  for (const item of detail.items ?? []) {
    reducer.dispatch({
      type: item.status === "in_progress" ? "item_started" : "item_completed",
      source: "replay",
      event_id: `thread-resume:item:${threadId}:${item.turn_id}:${item.id}`,
      protocol_method: THREAD_RESUME_METHOD,
      protocol_revision: RENDER_PROJECTION_REFERENCE_REVISION,
      item,
    });
  }
  reducer.dispatch({
    type: "thread_started",
    source: "replay",
    event_id: `thread-resume:thread:${threadId}`,
    protocol_method: THREAD_RESUME_METHOD,
    protocol_revision: RENDER_PROJECTION_REFERENCE_REVISION,
    thread_id: threadId,
    status: resumeProjectionStatus(detail.thread_read?.status),
  });
  return reducer;
}

function mergeResumeTurns(
  threadTurns: ThreadResumeResponse["thread"]["turns"],
  pageTurns: NonNullable<ThreadResumeResponse["initialTurnsPage"]>["data"],
): NonNullable<ThreadResumeResponse["thread"]["turns"]> {
  const turnsById = new Map(
    pageTurns.map((turn, index) => [turn.id, { turn, index }]),
  );
  for (const [index, turn] of (threadTurns ?? []).entries()) {
    turnsById.set(turn.id, { turn, index: pageTurns.length + index });
  }
  return [...turnsById.values()]
    .sort((left, right) => {
      const leftStartedAt = left.turn.startedAt;
      const rightStartedAt = right.turn.startedAt;
      if (
        typeof leftStartedAt === "number" &&
        typeof rightStartedAt === "number" &&
        leftStartedAt !== rightStartedAt
      ) {
        return leftStartedAt - rightStartedAt;
      }
      return left.index - right.index;
    })
    .map(({ turn }) => turn);
}

function resumeProjectionStatus(
  status: string | null | undefined,
): ConversationProjectionStatus {
  switch (status) {
    case "running":
    case "waitingAction":
    case "blocked":
    case "queued":
      return "running";
    case "failed":
      return "failed";
    case "cancelled":
    case "interrupted":
      return "interrupted";
    case "completed":
      return "completed";
    case "idle":
    case "stale":
    case "unknown":
    default:
      return "idle";
  }
}
