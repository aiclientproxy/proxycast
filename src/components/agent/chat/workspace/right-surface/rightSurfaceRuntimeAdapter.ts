import {
  createWorkspaceRightSurfaceOpenIntent,
  type WorkspaceRightSurfaceIntent,
} from "./rightSurfaceIntentQueue";
import type { WorkspaceRightSurfacePendingRequest } from "@/lib/api/workspaceRightSurface";
import type { WorkspaceRightSurfaceCommandOrigin } from "./rightSurfaceCommand";
import type { WorkspaceRightSurfaceRequestPriority } from "./rightSurfaceScheduler";
import {
  normalizeWorkspaceRightSurfaceKind,
  type WorkspaceRightSurfaceKind,
} from "./rightSurfaceTypes";

export interface WorkspaceRightSurfaceRuntimeOpenSignal {
  id: string;
  kind: WorkspaceRightSurfaceKind;
  origin: Extract<
    WorkspaceRightSurfaceCommandOrigin,
    "runtime" | "skill" | "mcpTool"
  >;
  createdAt: number;
  priority?: WorkspaceRightSurfaceRequestPriority;
  ttlMs?: number;
  reason?: string;
}

export interface WorkspaceRightSurfaceHarnessPendingInput {
  enabled: boolean;
  pendingCount: number;
  createdAt: number;
  ttlMs?: number;
}

export interface WorkspaceRightSurfaceFilePreviewInput {
  enabled: boolean;
  relativePath?: string | null;
  createdAt: number;
  origin?: Extract<
    WorkspaceRightSurfaceCommandOrigin,
    "runtime" | "skill" | "mcpTool"
  >;
  ttlMs?: number;
}

export interface WorkspaceRightSurfaceMcpShellOutputInput {
  enabled: boolean;
  outputId?: string | null;
  createdAt: number;
  priority?: WorkspaceRightSurfaceRequestPriority;
  ttlMs?: number;
}

const DEFAULT_RUNTIME_PENDING_INTENT_TTL_MS = 60_000;
const KNOWN_RIGHT_SURFACE_KINDS = new Set<WorkspaceRightSurfaceKind>([
  "workbench",
  "appSurface",
  "articleWorkspace",
  "expertInfo",
  "browser",
  "files",
  "shell",
  "harness",
]);
const KNOWN_RIGHT_SURFACE_ORIGINS = new Set<
  Extract<WorkspaceRightSurfaceCommandOrigin, "runtime" | "skill" | "mcpTool">
>(["runtime", "skill", "mcpTool"]);

export function buildWorkspaceRightSurfaceRuntimeOpenIntents(
  signals: readonly WorkspaceRightSurfaceRuntimeOpenSignal[],
): WorkspaceRightSurfaceIntent[] {
  return signals.map((signal) =>
    createWorkspaceRightSurfaceOpenIntent({
      id: signal.id,
      kind: signal.kind,
      origin: signal.origin,
      createdAt: signal.createdAt,
      priority: signal.priority ?? "background",
      ttlMs: signal.ttlMs,
      reason: signal.reason,
    }),
  );
}

export function buildWorkspaceRightSurfaceAppServerPendingIntents(
  pending: readonly WorkspaceRightSurfacePendingRequest[],
  now: number,
): WorkspaceRightSurfaceIntent[] {
  return buildWorkspaceRightSurfaceRuntimeOpenIntents(
    pending.flatMap((request) => {
      if (request.status !== "pending") {
        return [];
      }

      const kind = normalizeSurfaceKind(request.surfaceKind);
      const origin = normalizeSurfaceOrigin(request.origin);
      if (!kind || !origin) {
        return [];
      }

      return [
        {
          id: `app-server:${request.requestId}`,
          kind,
          origin,
          createdAt: parseRequestedAt(request.requestedAt) ?? now,
          priority:
            request.priority === "foreground" ? "foreground" : "background",
          ttlMs: request.ttlMs ?? undefined,
          reason: request.reason ?? undefined,
        },
      ];
    }),
  );
}

export function buildWorkspaceRightSurfaceHarnessPendingIntents({
  enabled,
  pendingCount,
  createdAt,
  ttlMs = DEFAULT_RUNTIME_PENDING_INTENT_TTL_MS,
}: WorkspaceRightSurfaceHarnessPendingInput): WorkspaceRightSurfaceIntent[] {
  if (!enabled || pendingCount <= 0) {
    return [];
  }

  const normalizedPendingCount = Math.floor(pendingCount);
  return Array.from({ length: normalizedPendingCount }, (_, index) =>
    createWorkspaceRightSurfaceOpenIntent({
      id: `runtime:harness:pending:${index + 1}`,
      kind: "harness",
      origin: "runtime",
      createdAt,
      priority: "background",
      ttlMs,
      reason: "harness_pending_approval",
    }),
  );
}

export function buildWorkspaceRightSurfaceFilePreviewIntents({
  enabled,
  relativePath,
  createdAt,
  origin = "runtime",
  ttlMs = DEFAULT_RUNTIME_PENDING_INTENT_TTL_MS,
}: WorkspaceRightSurfaceFilePreviewInput): WorkspaceRightSurfaceIntent[] {
  const normalizedPath = normalizePreviewPath(relativePath);
  if (!enabled || !normalizedPath) {
    return [];
  }

  return buildWorkspaceRightSurfaceRuntimeOpenIntents([
    {
      id: `runtime:file-preview:${normalizedPath}`,
      kind: "files",
      origin,
      createdAt,
      priority: "background",
      ttlMs,
      reason: "file_preview_ready",
    },
  ]);
}

export function buildWorkspaceRightSurfaceMcpShellOutputIntents({
  enabled,
  outputId,
  createdAt,
  priority = "background",
  ttlMs = DEFAULT_RUNTIME_PENDING_INTENT_TTL_MS,
}: WorkspaceRightSurfaceMcpShellOutputInput): WorkspaceRightSurfaceIntent[] {
  const normalizedOutputId = normalizeIntentKey(outputId);
  if (!enabled || !normalizedOutputId) {
    return [];
  }

  return buildWorkspaceRightSurfaceRuntimeOpenIntents([
    {
      id: `mcp:shell-output:${normalizedOutputId}`,
      kind: "shell",
      origin: "mcpTool",
      createdAt,
      priority,
      ttlMs,
      reason: "mcp_shell_output_ready",
    },
  ]);
}

function normalizePreviewPath(value?: string | null): string {
  return (value || "").trim().replace(/\\/g, "/");
}

function normalizeIntentKey(value?: string | null): string {
  return (value || "").trim().replace(/\s+/g, "-");
}

function normalizeSurfaceKind(
  value?: string | null,
): WorkspaceRightSurfaceKind | null {
  const normalized = normalizeWorkspaceRightSurfaceKind(value);
  return normalized && KNOWN_RIGHT_SURFACE_KINDS.has(normalized)
    ? normalized
    : null;
}

function normalizeSurfaceOrigin(
  value?: string | null,
): Extract<
  WorkspaceRightSurfaceCommandOrigin,
  "runtime" | "skill" | "mcpTool"
> | null {
  const normalized = (value || "").trim() as Extract<
    WorkspaceRightSurfaceCommandOrigin,
    "runtime" | "skill" | "mcpTool"
  >;
  return KNOWN_RIGHT_SURFACE_ORIGINS.has(normalized) ? normalized : null;
}

function parseRequestedAt(value: string): number | null {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}
