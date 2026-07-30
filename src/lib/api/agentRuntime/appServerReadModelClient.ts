import { AppServerClient } from "@/lib/api/appServer";
import { readCanonicalThreadDetail } from "./appServerCanonicalThreadProjection";
import type { AgentRuntimeThreadReadModel } from "./sessionTypes";

export type AppServerSessionReadClient = Pick<AppServerClient, "readThread">;

export interface AppServerReadModelClientDeps {
  appServerClient?: AppServerSessionReadClient;
}

export function createAppServerReadModelClient({
  appServerClient = new AppServerClient(),
}: AppServerReadModelClientDeps = {}) {
  async function getAgentRuntimeThreadRead(
    threadId: string,
  ): Promise<AgentRuntimeThreadReadModel> {
    const normalizedThreadId = threadId.trim();
    if (!normalizedThreadId) {
      throw new Error(
        "threadId is required to read canonical App Server thread",
      );
    }

    const response = await appServerClient.readThread({
      threadId: normalizedThreadId,
      includeTurns: true,
    });
    return projectAppServerThreadReadResult(response.result);
  }

  return {
    getAgentRuntimeThreadRead,
  };
}

export function projectAppServerThreadReadResult(
  value: unknown,
): AgentRuntimeThreadReadModel {
  const detail = readCanonicalThreadDetail(value);
  const readModel = detail?.thread_read;
  if (!detail || !readModel) {
    throw new Error("thread/read did not return canonical thread read model");
  }
  return {
    ...readModel,
    thread_id: detail.thread_id ?? readModel.thread_id,
    thread_items: detail.items,
    updated_at: detail.updated_at ?? readModel.updated_at,
  };
}
