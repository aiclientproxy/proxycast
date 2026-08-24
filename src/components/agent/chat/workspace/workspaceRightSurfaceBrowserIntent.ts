import type { WorkspaceRightSurfacePendingRequest } from "@/lib/api/workspaceRightSurface";
import {
  readBrowserTabHistoricalProjection,
  type BrowserTabHistoricalProjection,
} from "@/lib/api/browserTab";

export interface WorkspaceRightSurfaceBrowserIntent {
  source: "rightSurfacePending";
  sourceRequestId: string;
  origin: string;
  reason?: string | null;
  priority: "foreground" | "background";
  launchUrl?: string | null;
  title?: string | null;
  historicalProjection?: BrowserTabHistoricalProjection | null;
}

export function buildWorkspaceRightSurfacePendingBrowserIntent(
  pendingRequests: readonly WorkspaceRightSurfacePendingRequest[],
): WorkspaceRightSurfaceBrowserIntent | null {
  for (const request of pendingRequests) {
    if (request.status !== "pending" || request.surfaceKind !== "browser") {
      continue;
    }

    const metadata = asRecord(request.metadata);
    const browser = asRecord(metadata?.browser);
    const launchUrl =
      firstString(browser?.launchUrl) ??
      navigableCandidate(request.candidateId);

    const historicalProjection = readBrowserTabHistoricalProjection(
      browser?.historicalProjection ?? browser?.snapshot,
    );
    const intent: WorkspaceRightSurfaceBrowserIntent = {
      source: "rightSurfacePending",
      sourceRequestId: request.requestId,
      origin: request.origin,
      reason: request.reason ?? null,
      priority: request.priority === "foreground" ? "foreground" : "background",
      launchUrl,
      title: firstString(browser?.title),
    };
    if (historicalProjection) {
      intent.historicalProjection = historicalProjection;
    }
    return intent;
  }

  return null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function firstString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function navigableCandidate(value?: string | null): string | null {
  const normalized = firstString(value);
  if (!normalized) {
    return null;
  }
  if (
    /^[a-z][a-z0-9+.-]*:/i.test(normalized) ||
    /^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?::\d{1,5})?(?:[/?#].*)?$/i.test(
      normalized,
    ) ||
    /^(?:localhost|127(?:\.\d{1,3}){3}|\[::1\])(?::\d{1,5})?(?:[/?#].*)?$/i.test(
      normalized,
    )
  ) {
    return normalized;
  }
  return null;
}
