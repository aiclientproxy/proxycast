import { AppServerClient } from "./appServer";

export interface ThreadRevertRequest {
  threadId: string;
  beforeTurnId: string;
}

export interface ThreadRevertResult {
  threadId: string;
  turnsBackwardsCursor: string | null;
  itemsBackwardsCursor: string | null;
}

export type ThreadRevertAppServerClient = Pick<AppServerClient, "revertThread">;

export async function revertThreadHistory(
  request: ThreadRevertRequest,
  appServerClient: ThreadRevertAppServerClient = new AppServerClient(),
): Promise<ThreadRevertResult> {
  const threadId = requireIdentifier(request.threadId, "threadId");
  const beforeTurnId = requireIdentifier(request.beforeTurnId, "beforeTurnId");
  const response = await appServerClient.revertThread({
    threadId,
    beforeTurnId,
  });
  const result = response.result;
  const thread = result?.thread;
  if (!thread || thread.id !== threadId || !Array.isArray(thread.turns)) {
    throw new Error("thread/revert returned an invalid thread");
  }
  if (thread.turns.length !== 0) {
    throw new Error("thread/revert must return metadata-only history");
  }
  return {
    threadId,
    turnsBackwardsCursor: result.turnsBackwardsCursor ?? null,
    itemsBackwardsCursor: result.itemsBackwardsCursor ?? null,
  };
}

function requireIdentifier(value: string, label: string): string {
  const normalized = value.trim();
  if (!normalized) {
    throw new Error(`${label} is required`);
  }
  return normalized;
}
