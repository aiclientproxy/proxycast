import type { AgentThreadItem } from "../../types";
import { normalizeProcessDisplayText } from "../../utils/processDisplayText";
import { resolveVisibleReasoningSourceText } from "../../utils/reasoningDisplayText";
import { normalizeTurnSummaryDisplayText } from "../../utils/turnSummaryPresentation";

export function resolveReasoningDisplayText(
  item: Extract<AgentThreadItem, { type: "reasoning" }>,
): {
  summaryText: string;
  bodyText: string;
  combinedText: string;
} {
  const summaryText = normalizeProcessDisplayText(
    (item.summary || [])
      .map((line) => line.trim())
      .filter(Boolean)
      .join("\n\n"),
  );
  return {
    summaryText,
    bodyText: "",
    combinedText: normalizeProcessDisplayText(
      resolveVisibleReasoningSourceText(item),
    ),
  };
}

export function resolveThinkingDisplayText(
  item: Extract<AgentThreadItem, { type: "reasoning" }>,
): string {
  return resolveReasoningDisplayText(item).combinedText;
}

export function resolveTurnSummaryDisplayText(
  item: Extract<AgentThreadItem, { type: "turn_summary" }>,
): string {
  return normalizeTurnSummaryDisplayText(item.text);
}
