import { AppServerClient } from "@/lib/api/appServer";
import type {
  QueuedSubmission,
  ThreadQueueDeleteResponse,
  ThreadQueueReorderResponse,
  ThreadQueueStartResponse,
  ThreadQueueUpdateResponse,
  UserInput,
} from "@limecloud/app-server-client";

export type ThreadQueueActionAppServerClient = Pick<
  AppServerClient,
  | "addThreadQueue"
  | "updateThreadQueue"
  | "deleteThreadQueue"
  | "reorderThreadQueue"
  | "startThreadQueue"
>;

function requireId(value: string, field: string): string {
  const normalized = value.trim();
  if (!normalized) {
    throw new Error(`${field} is required for ThreadQueue action`);
  }
  return normalized;
}

function requireInput(input: readonly UserInput[]): UserInput[] {
  if (input.length === 0) {
    throw new Error("input is required for ThreadQueue action");
  }
  return [...input];
}

export function createThreadQueueActions({
  appServerClient = new AppServerClient(),
}: {
  appServerClient?: ThreadQueueActionAppServerClient;
} = {}) {
  async function addThreadQueue(params: {
    threadId: string;
    clientUserMessageId: string;
    input: readonly UserInput[];
  }): Promise<QueuedSubmission> {
    const response = await appServerClient.addThreadQueue({
      threadId: requireId(params.threadId, "threadId"),
      clientUserMessageId: requireId(
        params.clientUserMessageId,
        "clientUserMessageId",
      ),
      input: requireInput(params.input),
    });
    return response.result.queuedSubmission;
  }

  async function updateThreadQueue(params: {
    threadId: string;
    queuedSubmissionId: string;
    input: readonly UserInput[];
  }): Promise<ThreadQueueUpdateResponse["queuedSubmission"]> {
    const response = await appServerClient.updateThreadQueue({
      threadId: requireId(params.threadId, "threadId"),
      queuedSubmissionId: requireId(
        params.queuedSubmissionId,
        "queuedSubmissionId",
      ),
      input: requireInput(params.input),
    });
    return response.result.queuedSubmission;
  }

  async function deleteThreadQueue(params: {
    threadId: string;
    queuedSubmissionId: string;
  }): Promise<ThreadQueueDeleteResponse["deleted"]> {
    const response = await appServerClient.deleteThreadQueue({
      threadId: requireId(params.threadId, "threadId"),
      queuedSubmissionId: requireId(
        params.queuedSubmissionId,
        "queuedSubmissionId",
      ),
    });
    return response.result.deleted;
  }

  async function reorderThreadQueue(params: {
    threadId: string;
    queuedSubmissionIds: readonly string[];
  }): Promise<ThreadQueueReorderResponse> {
    const threadId = requireId(params.threadId, "threadId");
    const queuedSubmissionIds = params.queuedSubmissionIds.map((id) =>
      requireId(id, "queuedSubmissionId"),
    );
    const response = await appServerClient.reorderThreadQueue({
      threadId,
      queuedSubmissionIds,
    });
    return response.result;
  }

  async function startThreadQueue(params: {
    threadId: string;
    queuedSubmissionId?: string | null;
  }): Promise<ThreadQueueStartResponse["turn"]> {
    const response = await appServerClient.startThreadQueue({
      threadId: requireId(params.threadId, "threadId"),
      ...(params.queuedSubmissionId == null
        ? {}
        : {
            queuedSubmissionId: requireId(
              params.queuedSubmissionId,
              "queuedSubmissionId",
            ),
          }),
    });
    return response.result.turn;
  }

  return {
    addThreadQueue,
    updateThreadQueue,
    deleteThreadQueue,
    reorderThreadQueue,
    startThreadQueue,
  };
}

export const {
  addThreadQueue,
  updateThreadQueue,
  deleteThreadQueue,
  reorderThreadQueue,
  startThreadQueue,
} = createThreadQueueActions();
