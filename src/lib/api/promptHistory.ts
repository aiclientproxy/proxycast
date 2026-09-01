import { AppServerClient } from "./appServerClient";
import type {
  AppServerPromptHistoryEntry,
  AppServerPromptHistoryReadParams,
} from "./appServerTypes";

export type PromptHistoryClient = Pick<
  AppServerClient,
  "readPromptHistory" | "appendPromptHistory"
>;

export async function readPromptHistory(
  client: PromptHistoryClient = new AppServerClient(),
  limit = 50,
): Promise<AppServerPromptHistoryEntry[]> {
  const entries: AppServerPromptHistoryEntry[] = [];
  let cursor: string | undefined;
  let logId: string | undefined;
  do {
    const params: AppServerPromptHistoryReadParams = {
      ...(cursor ? { cursor } : {}),
      ...(logId ? { logId } : {}),
      limit: Math.min(100, Math.max(1, limit)),
    };
    const response = await client.readPromptHistory(params);
    const page = response.result;
    logId = page.logId;
    entries.push(...page.data);
    cursor = page.nextCursor ?? undefined;
  } while (cursor && entries.length < limit);
  return entries.slice(0, limit);
}
