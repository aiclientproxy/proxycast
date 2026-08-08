import {
  AppServerClient,
  type AppServerThreadBackgroundTerminalsListResponse,
  type AppServerThreadBackgroundTerminalsTerminateResponse,
} from "@/lib/api/appServer";

const PAGE_LIMIT = 100;

export type BackgroundTerminalsAppServerClient = Pick<
  AppServerClient,
  | "listThreadBackgroundTerminals"
  | "terminateThreadBackgroundTerminal"
>;

export async function terminateBackgroundTerminalForItem(
  params: { threadId: string; itemId: string },
  client: BackgroundTerminalsAppServerClient = new AppServerClient(),
): Promise<AppServerThreadBackgroundTerminalsTerminateResponse> {
  const threadId = requiredId(params.threadId, "threadId");
  const itemId = requiredId(params.itemId, "itemId");
  const terminal = await findBackgroundTerminal(threadId, itemId, client);
  if (!terminal) {
    return { terminated: false };
  }
  return (
    await client.terminateThreadBackgroundTerminal({
      threadId,
      processId: terminal.processId,
    })
  ).result;
}

async function findBackgroundTerminal(
  threadId: string,
  itemId: string,
  client: BackgroundTerminalsAppServerClient,
) {
  let cursor: string | null = null;
  const visitedCursors = new Set<string>();
  while (true) {
    const page: AppServerThreadBackgroundTerminalsListResponse = (
      await client.listThreadBackgroundTerminals({
        threadId,
        cursor,
        limit: PAGE_LIMIT,
      })
    ).result;
    const terminal = page.data.find((candidate) => candidate.itemId === itemId);
    if (terminal) {
      return terminal;
    }
    const nextCursor = page.nextCursor ?? null;
    if (!nextCursor) {
      return null;
    }
    if (visitedCursors.has(nextCursor)) {
      throw new Error("background terminal pagination cursor repeated");
    }
    visitedCursors.add(nextCursor);
    cursor = nextCursor;
  }
}

function requiredId(value: string, name: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new Error(`background terminal ${name} is required`);
  }
  return trimmed;
}
