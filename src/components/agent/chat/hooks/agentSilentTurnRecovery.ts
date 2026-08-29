import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import { normalizeLegacyThreadItems } from "@/lib/api/agentTextNormalization";

const SILENT_TURN_RECOVERY_GRACE_MS = 15_000;
const SILENT_TURN_RECOVERY_ACTIVITY_SKEW_MS = 5_000;

const TERMINAL_TURN_STATUSES = new Set([
  "aborted",
  "canceled",
  "cancelled",
  "completed",
  "done",
  "failed",
  "interrupted",
]);

function normalizeTimestampMs(
  value: string | number | null | undefined,
): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value >= 1_000_000_000_000 ? value : value * 1000;
  }
  if (typeof value !== "string") {
    return Number.NaN;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return Number.NaN;
  }

  const numericValue = Number(trimmed);
  if (Number.isFinite(numericValue)) {
    return numericValue >= 1_000_000_000_000
      ? numericValue
      : numericValue * 1000;
  }

  const parsed = Date.parse(trimmed);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function maxTimestampMs(
  ...values: Array<string | number | null | undefined>
): number {
  const normalizedValues = values
    .map((value) => normalizeTimestampMs(value))
    .filter((value) => Number.isFinite(value));
  return normalizedValues.length > 0
    ? Math.max(...normalizedValues)
    : Number.NaN;
}

function normalizeText(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function isTerminalTurnStatus(status: unknown): boolean {
  return TERMINAL_TURN_STATUSES.has(String(status || "").toLowerCase());
}

export function hasRecoverableTerminalTurnActivity(
  detail: Pick<AgentSessionDetail, "turns">,
  requestStartedAt: number,
  promptText: string,
  turnId?: string | null,
): boolean {
  const recoveryThresholdMs = requestStartedAt - SILENT_TURN_RECOVERY_GRACE_MS;
  const activityThresholdMs =
    requestStartedAt - SILENT_TURN_RECOVERY_ACTIVITY_SKEW_MS;
  const normalizedPrompt = promptText.trim();
  const normalizedTurnId = turnId?.trim() || null;

  return (detail.turns ?? []).some((turn) => {
    if (!isTerminalTurnStatus(turn.status)) {
      return false;
    }

    const latestTurnTimestampMs = maxTimestampMs(
      turn.started_at,
      turn.created_at,
      turn.updated_at,
      turn.completed_at,
    );
    if (latestTurnTimestampMs < activityThresholdMs) {
      return false;
    }

    if (normalizedTurnId && turn.id === normalizedTurnId) {
      return latestTurnTimestampMs >= recoveryThresholdMs;
    }

    return (
      normalizedPrompt.length > 0 &&
      normalizeText(turn.prompt_text) === normalizedPrompt &&
      latestTurnTimestampMs >= recoveryThresholdMs
    );
  });
}

export function hasRecoverableSilentTurnActivity(
  detail: Pick<AgentSessionDetail, "turns" | "items">,
  requestStartedAt: number,
  promptText: string,
): boolean {
  const recoveryThresholdMs = requestStartedAt - SILENT_TURN_RECOVERY_GRACE_MS;
  const activityThresholdMs =
    requestStartedAt - SILENT_TURN_RECOVERY_ACTIVITY_SKEW_MS;
  const normalizedPrompt = promptText.trim();
  const normalizedItems = normalizeLegacyThreadItems(detail.items ?? []);

  const hasRecentMatchingTurn = (detail.turns ?? []).some((turn) => {
    const promptMatches =
      normalizedPrompt.length > 0 &&
      normalizeText(turn.prompt_text) === normalizedPrompt;
    const latestTurnTimestampMs = maxTimestampMs(
      turn.started_at,
      turn.created_at,
      turn.updated_at,
      turn.completed_at,
    );
    return promptMatches && latestTurnTimestampMs >= recoveryThresholdMs;
  });
  if (hasRecentMatchingTurn) {
    return true;
  }

  const hasRecentMatchingUserMessage = normalizedItems.some((item) => {
    if (
      item.type !== "user_message" ||
      item.content.trim() !== normalizedPrompt
    ) {
      return false;
    }
    const latestItemTimestampMs = maxTimestampMs(
      item.started_at,
      item.updated_at,
      item.completed_at,
    );
    return latestItemTimestampMs >= recoveryThresholdMs;
  });
  if (hasRecentMatchingUserMessage) {
    return true;
  }

  const hasRecentTurnActivity = (detail.turns ?? []).some((turn) => {
    return (
      maxTimestampMs(
        turn.started_at,
        turn.created_at,
        turn.updated_at,
        turn.completed_at,
      ) >= activityThresholdMs
    );
  });
  if (hasRecentTurnActivity) {
    return true;
  }

  const hasRecentItemActivity = normalizedItems.some((item) => {
    return (
      maxTimestampMs(item.started_at, item.updated_at, item.completed_at) >=
      activityThresholdMs
    );
  });
  if (hasRecentItemActivity) {
    return true;
  }

  return false;
}
