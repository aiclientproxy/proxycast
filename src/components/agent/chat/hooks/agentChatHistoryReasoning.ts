import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import type { HistoryToolCall } from "./agentChatHistoryTypes";
import { resolveThreadItemTimelinePosition } from "./agentChatHistoryPrimitives";

export function contentPartMetadataFromThreadToolItem(
  item: AgentThreadItem,
  toolCall: HistoryToolCall,
): Record<string, unknown> | undefined {
  const metadata =
    toolCall.metadata && typeof toolCall.metadata === "object"
      ? { ...(toolCall.metadata as Record<string, unknown>) }
      : {};
  metadata.source = "agent_thread_item";
  metadata.threadItemId = item.id;
  metadata.sequence = resolveThreadItemTimelinePosition(item);
  metadata.turnId = item.turn_id;
  return Object.keys(metadata).length > 0 ? metadata : undefined;
}
