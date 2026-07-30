import type { TypedPendingInteraction } from "@/lib/api/agentRuntime/pendingInteractionController";

export function selectActivePendingInteraction(
  interactions: readonly TypedPendingInteraction[],
  threadId?: string | null,
): TypedPendingInteraction | null {
  const normalizedThreadId = threadId?.trim();
  return (
    interactions.find(
      (interaction) =>
        interaction.status === "pending" &&
        (!normalizedThreadId || interaction.thread_id === normalizedThreadId),
    ) ?? null
  );
}
