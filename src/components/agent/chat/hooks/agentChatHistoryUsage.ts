import type { AgentTokenUsage } from "@/lib/api/agentProtocol";
import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import {
  asHistoryRecord,
  readHistoryString,
} from "./agentChatHistoryPrimitives";

function readNonNegativeNumber(
  record: Record<string, unknown>,
  keys: string[],
): number | undefined {
  for (const key of keys) {
    const value = record[key];
    if (
      typeof value === "number" &&
      Number.isFinite(value) &&
      value >= 0
    ) {
      return value;
    }
  }
  return undefined;
}

export function normalizeHistoryUsage(
  usage: unknown,
): AgentTokenUsage | undefined {
  if (!usage || typeof usage !== "object" || Array.isArray(usage)) {
    return undefined;
  }

  const record = usage as Record<string, unknown>;
  const inputTokens = readNonNegativeNumber(record, [
    "input_tokens",
    "inputTokens",
  ]);
  const outputTokens = readNonNegativeNumber(record, [
    "output_tokens",
    "outputTokens",
  ]);
  if (inputTokens === undefined || outputTokens === undefined) {
    return undefined;
  }

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    cached_input_tokens: readNonNegativeNumber(record, [
      "cached_input_tokens",
      "cachedInputTokens",
    ]),
    cache_creation_input_tokens: readNonNegativeNumber(record, [
      "cache_write_input_tokens",
      "cacheWriteInputTokens",
      "cache_creation_input_tokens",
      "cacheCreationInputTokens",
    ]),
  };
}

function historyTurnIdFromRecord(value: unknown): string {
  const record = asHistoryRecord(value);
  return (
    readHistoryString(record?.turn_id) ||
    readHistoryString(record?.turnId) ||
    readHistoryString(record?.id)
  );
}

function historyUsageFromTurnRecord(value: unknown): AgentTokenUsage | undefined {
  const record = asHistoryRecord(value);
  return normalizeHistoryUsage(
    record?.usage ?? record?.token_usage ?? record?.tokenUsage,
  );
}

function findTurnUsageInRecords(
  turns: readonly unknown[],
  runtimeTurnId?: string | null,
): AgentTokenUsage | undefined {
  const normalizedTurnId = runtimeTurnId?.trim() || "";
  return [...turns]
    .reverse()
    .filter((turn) => {
      if (!normalizedTurnId) {
        return true;
      }
      return historyTurnIdFromRecord(turn) === normalizedTurnId;
    })
    .map(historyUsageFromTurnRecord)
    .find((usage): usage is AgentTokenUsage => Boolean(usage));
}

export function resolveSessionDetailTurnUsage(
  detail: AgentSessionDetail,
  runtimeTurnId?: string | null,
): AgentTokenUsage | undefined {
  return (
    findTurnUsageInRecords(detail.thread_read?.turns || [], runtimeTurnId) ??
    findTurnUsageInRecords(detail.turns || [], runtimeTurnId) ??
    normalizeHistoryUsage(
      asHistoryRecord(detail.thread_read?.diagnostics)?.latest_turn_usage ??
        asHistoryRecord(detail.thread_read?.diagnostics)?.latestTurnUsage ??
        asHistoryRecord(detail.thread_read?.runtime_summary)?.latest_turn_usage ??
        asHistoryRecord(detail.thread_read?.runtime_summary)?.latestTurnUsage,
    )
  );
}
