import { AppServerClient } from "@/lib/api/appServer";
import {
  METHOD_COLLABORATION_MODE_LIST,
  type CollaborationModeListResponse,
  type CollaborationModeMask,
  type ModeKind,
} from "@limecloud/app-server-client";

type CollaborationModeAppServerClient = Pick<AppServerClient, "request">;

export async function listCollaborationModes(
  appServerClient: CollaborationModeAppServerClient = new AppServerClient(),
): Promise<CollaborationModeMask[]> {
  const response = await appServerClient.request<CollaborationModeListResponse>(
    METHOD_COLLABORATION_MODE_LIST,
    {},
  );
  if (!Array.isArray(response.result.data)) {
    throw new Error("App Server collaborationMode/list did not return data");
  }
  return response.result.data.map(assertCollaborationModeMask);
}

export async function resolveCollaborationModeMask(
  mode: ModeKind,
  appServerClient?: CollaborationModeAppServerClient,
): Promise<CollaborationModeMask> {
  const matches = (await listCollaborationModes(appServerClient)).filter(
    (preset) => preset.mode === mode,
  );
  if (matches.length !== 1) {
    throw new Error(
      `App Server collaborationMode/list must return exactly one ${mode} preset`,
    );
  }
  return matches[0];
}

function assertCollaborationModeMask(
  value: CollaborationModeMask,
  index: number,
): CollaborationModeMask {
  if (!value || typeof value !== "object") {
    throw new Error(
      `App Server collaborationMode/list returned invalid preset at index ${index}`,
    );
  }
  const name = value.name?.trim();
  if (!name) {
    throw new Error(
      `App Server collaborationMode/list returned unnamed preset at index ${index}`,
    );
  }
  if (value.mode !== "default" && value.mode !== "plan") {
    throw new Error(
      `App Server collaborationMode/list returned invalid mode at index ${index}`,
    );
  }
  const model = normalizeOptionalOverride(value.model, "model", index);
  const reasoningEffort = normalizeOptionalOverride(
    value.reasoning_effort,
    "reasoning_effort",
    index,
  );
  return {
    name,
    mode: value.mode,
    model,
    reasoning_effort: reasoningEffort,
  };
}

function normalizeOptionalOverride(
  value: string | null | undefined,
  field: string,
  index: number,
): string | null {
  if (value == null) {
    return null;
  }
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(
      `App Server collaborationMode/list returned invalid ${field} at index ${index}`,
    );
  }
  return value.trim();
}
