import {
  buildExpertRuntimeMetadata,
  type ExpertCatalog,
  type ExpertProfile,
} from "@/features/experts";

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export interface ResolveExpertPanelRequestMetadataParams {
  initialAutoSendRequestMetadata?: Record<string, unknown>;
  initialRequestMetadata?: Record<string, unknown>;
  sessionRequestMetadata?: Record<string, unknown> | null;
}

export function resolveExpertPanelRequestMetadata({
  initialAutoSendRequestMetadata,
  initialRequestMetadata,
  sessionRequestMetadata,
}: ResolveExpertPanelRequestMetadataParams): Record<string, unknown> | null {
  return (
    initialAutoSendRequestMetadata ??
    initialRequestMetadata ??
    sessionRequestMetadata ??
    null
  );
}

export function mergeExpertSkillRefsIntoRequestMetadata(
  metadata: Record<string, unknown> | null | undefined,
  skillRefs: string[] | null,
): Record<string, unknown> | null {
  if (!metadata || !skillRefs) {
    return metadata ?? null;
  }

  const root: Record<string, unknown> = { ...metadata };
  const expert = asRecord(root.expert);
  const harness = asRecord(root.harness);
  const harnessExpert = asRecord(harness?.expert);

  if (expert) {
    root.expert = {
      ...expert,
      skillRefs: [...skillRefs],
    };
  }

  if (harness || harnessExpert) {
    root.harness = {
      ...(harness ?? {}),
      expert: {
        ...(harnessExpert ?? {}),
        skill_refs: [...skillRefs],
      },
    };
  }

  return root;
}

export function resolveWorkspaceRequestMetadataWithExpertSkills({
  activeRequestMetadata,
  expertSkillRefsOverride,
  initialAutoSendRequestMetadata,
  initialRequestMetadata,
  sessionRequestMetadata,
}: ResolveExpertPanelRequestMetadataParams & {
  activeRequestMetadata?: Record<string, unknown> | null;
  expertSkillRefsOverride: string[] | null;
}): Record<string, unknown> | null {
  const metadataWithExpertSkills = mergeExpertSkillRefsIntoRequestMetadata(
    activeRequestMetadata ??
      initialRequestMetadata ??
      initialAutoSendRequestMetadata ??
      sessionRequestMetadata ??
      null,
    expertSkillRefsOverride,
  );
  return metadataWithExpertSkills &&
    Object.keys(metadataWithExpertSkills).length > 0
    ? metadataWithExpertSkills
    : null;
}

export function resolveSessionExpertRequestMetadata(
  threadRead?: {
    session_business_object_ref_metadata?: Record<string, unknown> | null;
  } | null,
): Record<string, unknown> | null {
  const metadata = asRecord(threadRead?.session_business_object_ref_metadata);
  const harness = asRecord(metadata?.harness);
  if (!asRecord(metadata?.expert) && !asRecord(harness?.expert)) {
    return null;
  }
  return { ...metadata };
}

export function shouldAllowDetachedInitialAutoSend(
  initialAutoSendRequestMetadata?: Record<string, unknown>,
): boolean {
  const metadata = asRecord(initialAutoSendRequestMetadata);
  const harness = asRecord(metadata?.harness);
  return Boolean(
    asRecord(metadata?.expert) ||
    asRecord(harness?.expert) ||
    asRecord(harness?.plugin_activation_intent) ||
    asRecord(harness?.pluginActivationIntent),
  );
}

export interface BuildThreadExpertProfileSwitchRequestMetadataParams {
  currentMetadata?: Record<string, unknown> | null;
  expert: ExpertProfile;
  catalog: Pick<ExpertCatalog, "tenantId" | "version">;
  switchedAt?: string;
}

function readExpertMetadataString(
  metadata: Record<string, unknown> | null | undefined,
  expertKey: string,
  harnessKey: string,
): string | undefined {
  const root = asRecord(metadata);
  const expert = asRecord(root?.expert);
  const harnessExpert = asRecord(asRecord(root?.harness)?.expert);
  const expertValue = expert?.[expertKey];
  if (typeof expertValue === "string" && expertValue.trim()) {
    return expertValue.trim();
  }
  const harnessValue = harnessExpert?.[harnessKey];
  return typeof harnessValue === "string" && harnessValue.trim()
    ? harnessValue.trim()
    : undefined;
}

export function buildThreadExpertProfileSwitchRequestMetadata({
  currentMetadata,
  expert,
  catalog,
  switchedAt,
}: BuildThreadExpertProfileSwitchRequestMetadataParams): Record<
  string,
  unknown
> {
  const root = asRecord(currentMetadata);
  const currentHarness = asRecord(root?.harness);
  const nextMetadata = buildExpertRuntimeMetadata(expert, {
    catalogVersion: catalog.version,
    tenantId: catalog.tenantId,
  });
  const previousExpertId = readExpertMetadataString(
    currentMetadata,
    "expertId",
    "expert_id",
  );
  const previousReleaseId = readExpertMetadataString(
    currentMetadata,
    "releaseId",
    "release_id",
  );
  const roleSwitch: Record<string, unknown> = {
    kind: "expert_profile_switch",
    scope: "thread",
    source: "expert_info_panel",
    next_expert_id: nextMetadata.expert.expertId,
    next_release_id: nextMetadata.expert.releaseId,
  };
  if (previousExpertId) {
    roleSwitch.previous_expert_id = previousExpertId;
  }
  if (previousReleaseId) {
    roleSwitch.previous_release_id = previousReleaseId;
  }
  if (switchedAt) {
    roleSwitch.switched_at = switchedAt;
  }

  return {
    ...(root ?? {}),
    expert: nextMetadata.expert,
    harness: {
      ...(currentHarness ?? {}),
      ...nextMetadata.harness,
      expert_role_switch: roleSwitch,
    },
  };
}
