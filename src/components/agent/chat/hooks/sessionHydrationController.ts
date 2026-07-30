import { AGENT_RUNTIME_DEFAULT_HISTORY_LIMIT } from "@/lib/api/agentRuntime/requestTypes";

export const SESSION_DETAIL_HISTORY_LIMIT = AGENT_RUNTIME_DEFAULT_HISTORY_LIMIT;

export interface SessionDetailHydrationOptions {
  historyLimit: number;
  resumeSessionStartHooks?: true;
  source?: string;
}

export function normalizeSessionDetailHistoryLimit(
  value: number | null | undefined,
): number {
  if (!Number.isFinite(value) || !value || value <= 0) {
    return SESSION_DETAIL_HISTORY_LIMIT;
  }
  return Math.trunc(value);
}

export function buildSessionDetailHydrationOptions(params?: {
  resumeSessionStartHooks?: boolean;
  historyLimit?: number | null;
  source?: string | null;
}): SessionDetailHydrationOptions {
  const options: SessionDetailHydrationOptions = {
    historyLimit: normalizeSessionDetailHistoryLimit(params?.historyLimit),
  };
  if (params?.resumeSessionStartHooks) {
    options.resumeSessionStartHooks = true;
  }
  const source = params?.source?.trim();
  if (source) {
    options.source = source;
  }
  return options;
}

export function isCurrentSessionHydrationRequest(params: {
  currentRequestVersion?: number | null;
  requestVersion?: number | null;
  currentSessionId?: string | null;
  targetSessionId?: string | null;
}): boolean {
  if (
    params.requestVersion !== undefined &&
    params.requestVersion !== null &&
    params.currentRequestVersion !== params.requestVersion
  ) {
    return false;
  }

  const targetSessionId = params.targetSessionId?.trim();
  if (!targetSessionId) {
    return true;
  }

  return params.currentSessionId?.trim() === targetSessionId;
}
