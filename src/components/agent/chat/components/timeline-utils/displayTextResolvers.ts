import type { AgentThreadItem } from "../../types";
import { normalizeProcessDisplayText } from "../../utils/processDisplayText";
import { normalizeTurnSummaryDisplayText } from "../../utils/turnSummaryPresentation";
import { normalizeComparableThinkingText } from "./textFormatting";

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
  const canonicalContentText = normalizeProcessDisplayText(
    (item.content || [])
      .map((part) => part.trim())
      .filter(Boolean)
      .join("\n\n"),
  );
  const bodyText =
    canonicalContentText || normalizeProcessDisplayText(item.text.trim());

  if (!summaryText) {
    return {
      summaryText: "",
      bodyText,
      combinedText: bodyText,
    };
  }

  if (!bodyText) {
    return {
      summaryText,
      bodyText: "",
      combinedText: summaryText,
    };
  }

  if (
    normalizeComparableThinkingText(summaryText) ===
    normalizeComparableThinkingText(bodyText)
  ) {
    return {
      summaryText,
      bodyText: "",
      combinedText: summaryText,
    };
  }

  return {
    summaryText,
    bodyText,
    combinedText: normalizeProcessDisplayText(`${summaryText}\n\n${bodyText}`),
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
