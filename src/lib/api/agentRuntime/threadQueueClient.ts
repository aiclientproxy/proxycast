import { AppServerClient } from "@/lib/api/appServer";
import type {
  QueuedSubmission,
  ThreadQueueListResponse,
  UserInput,
} from "@limecloud/app-server-client";

const THREAD_QUEUE_PAGE_LIMIT = 100;

export type ThreadQueueAppServerClient = Pick<
  AppServerClient,
  "listThreadQueue"
>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function isUserInput(value: unknown): value is UserInput {
  if (!isRecord(value) || typeof value.type !== "string") {
    return false;
  }
  switch (value.type) {
    case "text":
      return typeof value.text === "string";
    case "image":
      return typeof value.url === "string";
    case "localImage":
      return typeof value.path === "string";
    case "skill":
    case "mention":
      return typeof value.name === "string" && typeof value.path === "string";
    default:
      return false;
  }
}

export function parseQueuedSubmission(value: unknown): QueuedSubmission | null {
  if (
    !isRecord(value) ||
    typeof value.id !== "string" ||
    !value.id.trim() ||
    typeof value.clientUserMessageId !== "string" ||
    !value.clientUserMessageId.trim() ||
    !Array.isArray(value.input) ||
    value.input.length === 0 ||
    !value.input.every(isUserInput)
  ) {
    return null;
  }
  return value as unknown as QueuedSubmission;
}

function parseQueuePage(value: unknown): ThreadQueueListResponse {
  if (!isRecord(value) || !Array.isArray(value.data)) {
    throw new Error("thread/queue/list returned an invalid page");
  }
  const data = value.data.map(parseQueuedSubmission);
  if (data.some((submission) => submission === null)) {
    throw new Error("thread/queue/list returned an invalid queued submission");
  }
  const nextCursor = value.nextCursor;
  if (
    nextCursor !== undefined &&
    nextCursor !== null &&
    typeof nextCursor !== "string"
  ) {
    throw new Error("thread/queue/list returned an invalid nextCursor");
  }
  return {
    data: data as QueuedSubmission[],
    nextCursor: nextCursor ?? null,
  };
}

export function createThreadQueueClient({
  appServerClient = new AppServerClient(),
}: {
  appServerClient?: ThreadQueueAppServerClient;
} = {}) {
  async function listThreadQueue(
    threadId: string,
  ): Promise<QueuedSubmission[]> {
    const normalizedThreadId = threadId.trim();
    if (!normalizedThreadId) {
      throw new Error("threadId is required to read ThreadQueue");
    }

    const submissions: QueuedSubmission[] = [];
    const seenCursors = new Set<string>();
    let cursor: string | null = null;
    do {
      const response = await appServerClient.listThreadQueue({
        threadId: normalizedThreadId,
        limit: THREAD_QUEUE_PAGE_LIMIT,
        ...(cursor ? { cursor } : {}),
      });
      const page = parseQueuePage(response.result);
      submissions.push(...page.data);
      cursor = page.nextCursor ?? null;
      if (cursor) {
        if (seenCursors.has(cursor)) {
          throw new Error("thread/queue/list returned a repeated nextCursor");
        }
        seenCursors.add(cursor);
      }
    } while (cursor);

    return submissions;
  }

  return { listThreadQueue };
}

export const { listThreadQueue } = createThreadQueueClient();
