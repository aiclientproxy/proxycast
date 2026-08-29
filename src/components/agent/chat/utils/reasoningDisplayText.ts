import type { AgentThreadItem } from "../types";

type ReasoningItem = Extract<AgentThreadItem, { type: "reasoning" }>;

export function resolveVisibleReasoningSourceText(item: ReasoningItem): string {
  const summaryText = (item.summary || [])
    .map((line) => line.trim())
    .filter(Boolean)
    .join("\n\n");
  if (summaryText) {
    return summaryText;
  }

  const hasCanonicalRawContent = (item.content || []).some((part) =>
    Boolean(part.trim()),
  );
  return hasCanonicalRawContent ? "" : item.text.trim();
}
