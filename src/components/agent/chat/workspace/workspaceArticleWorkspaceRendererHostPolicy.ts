export type WorkspaceArticleWorkspaceRendererHostPolicyStatus =
  | "placeholder"
  | "blocked";

export type WorkspaceArticleWorkspaceRendererExecutionModel =
  "host_placeholder_only";

export type WorkspaceArticleWorkspaceRendererEntryLoadPolicy = "not_loaded";

export interface WorkspaceArticleWorkspaceRendererHostPolicy {
  status: WorkspaceArticleWorkspaceRendererHostPolicyStatus;
  executionMode: string;
  rendererExecutionModel: WorkspaceArticleWorkspaceRendererExecutionModel;
  entryLoadPolicy: WorkspaceArticleWorkspaceRendererEntryLoadPolicy;
  canLoadEntry: false;
  reasonCode: string;
  requestedOutputArtifactKind: string | null;
  allowedOutputArtifactKinds: string[];
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function readStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const seen = new Set<string>();
  const result: string[] = [];
  for (const item of value) {
    const normalized = readString(item);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    result.push(normalized);
  }
  return result;
}

export function buildWorkspaceArticleWorkspaceRendererHostPolicy(
  runtimeAuthorization: Record<string, unknown> | null,
): WorkspaceArticleWorkspaceRendererHostPolicy {
  const authorization = asRecord(runtimeAuthorization);
  const status = readString(authorization?.status);
  const executionMode =
    readString(
      authorization?.executionMode,
      authorization?.execution_mode,
    ) ?? "host_placeholder";
  const reasonCode =
    readString(
      authorization?.reasonCode,
      authorization?.reason_code,
    ) ?? "app_declared_renderer_placeholder_only";
  const requestedOutputArtifactKind = readString(
    authorization?.requestedOutputArtifactKind,
    authorization?.requested_output_artifact_kind,
  );
  const allowedOutputArtifactKinds = readStringArray(
    authorization?.allowedOutputArtifactKinds ??
      authorization?.allowed_output_artifact_kinds,
  );

  return {
    status: status === "denied" ? "blocked" : "placeholder",
    executionMode,
    rendererExecutionModel: "host_placeholder_only",
    entryLoadPolicy: "not_loaded",
    canLoadEntry: false,
    reasonCode,
    requestedOutputArtifactKind,
    allowedOutputArtifactKinds,
  };
}
