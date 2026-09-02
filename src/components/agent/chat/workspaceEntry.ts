import { buildClawAgentParams } from "@/lib/workspace/navigation";
import type {
  AgentInitialInputCapabilityParams,
  AgentPageParams,
  AgentPendingServiceSkillLaunchParams,
  AgentProjectFileOpenTarget,
  AgentSiteSkillLaunchParams,
} from "@/types/page";
import type { CreationMode } from "./components/types";
import type { MessageImage } from "./types";
import type { ChatToolPreferences } from "./utils/chatToolPreferences";

export interface AgentChatWorkspaceBootstrap {
  projectId?: string;
  contentId?: string;
  initialUserPrompt?: string;
  initialUserImages?: MessageImage[];
  initialRequestMetadata?: Record<string, unknown>;
  initialAutoSendRequestMetadata?: Record<string, unknown>;
  autoRunInitialPromptOnMount?: boolean;
  theme?: string;
  initialCreationMode?: CreationMode;
  lockTheme?: boolean;
  openBrowserAssistOnMount?: boolean;
  initialSiteSkillLaunch?: AgentSiteSkillLaunchParams;
  initialPendingServiceSkillLaunch?: AgentPendingServiceSkillLaunchParams;
  initialInputCapability?: AgentInitialInputCapabilityParams;
  initialProjectFileOpenTarget?: AgentProjectFileOpenTarget;
  newChatAt?: number;
}

export interface WorkspaceEntryPayload {
  prompt?: string;
  images?: MessageImage[];
  projectId?: string;
  contentId?: string;
  initialRequestMetadata?: Record<string, unknown>;
  initialAutoSendRequestMetadata?: Record<string, unknown>;
  autoRunInitialPromptOnMount?: boolean;
  openBrowserAssistOnMount?: boolean;
  initialSiteSkillLaunch?: AgentSiteSkillLaunchParams;
  initialPendingServiceSkillLaunch?: AgentPendingServiceSkillLaunchParams;
  initialInputCapability?: AgentInitialInputCapabilityParams;
  initialProjectFileOpenTarget?: AgentProjectFileOpenTarget;
  toolPreferences?: ChatToolPreferences;
  themeOverride?: string;
  lockTheme?: boolean;
}

export interface ResolveWorkspaceEntryInput {
  projectId?: string | null;
  activeTheme: string;
  creationMode: CreationMode;
  defaultToolPreferences: ChatToolPreferences;
  payload: WorkspaceEntryPayload;
  now?: () => number;
}

export type WorkspaceEntryRejectedReason = "missing_project" | "empty_payload";

export type ResolvedWorkspaceEntry =
  | {
      ok: false;
      reason: WorkspaceEntryRejectedReason;
    }
  | {
      ok: true;
      toolPreferences: ChatToolPreferences;
      targetTheme: string;
      nextNewChatAt: number;
      navigationParams: AgentPageParams;
      workspaceBootstrap: AgentChatWorkspaceBootstrap;
    };

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function hasServiceSkillLaunchMetadata(
  metadata?: Record<string, unknown>,
): boolean {
  const record = asRecord(metadata);
  if (!record) {
    return false;
  }

  const harness = asRecord(record.harness) ?? record;
  const launch =
    asRecord(harness.service_skill_launch) ??
    asRecord(harness.serviceSkillLaunch);
  if (!launch) {
    return false;
  }

  const adapterName = launch.adapter_name ?? launch.adapterName;
  return typeof adapterName === "string" && adapterName.trim().length > 0;
}

export function resolveWorkspaceEntry(
  input: ResolveWorkspaceEntryInput,
): ResolvedWorkspaceEntry {
  const {
    projectId,
    activeTheme,
    creationMode,
    defaultToolPreferences,
    payload,
    now = () => Date.now(),
  } = input;

  const hasPrompt = Boolean(payload.prompt?.trim());
  const hasImages = Boolean(payload.images?.length);
  const resolvedProjectId = payload.projectId?.trim() || projectId || undefined;
  const hasContentId = Boolean(payload.contentId?.trim());
  const hasSiteSkillLaunch = Boolean(
    payload.initialSiteSkillLaunch?.adapterName?.trim(),
  );
  const hasPendingServiceSkillLaunch = Boolean(
    payload.initialPendingServiceSkillLaunch?.skillId?.trim(),
  );
  const hasInitialInputCapability = Boolean(
    payload.initialInputCapability?.capabilityRoute,
  );
  const toolPreferences = payload.toolPreferences ?? defaultToolPreferences;
  const targetTheme = payload.themeOverride ?? activeTheme;
  const lockTheme =
    payload.lockTheme ??
    (targetTheme === "general" &&
      (hasSiteSkillLaunch ||
        hasServiceSkillLaunchMetadata(payload.initialAutoSendRequestMetadata) ||
        hasServiceSkillLaunchMetadata(payload.initialRequestMetadata)));
  const openBrowserAssistOnMount = payload.openBrowserAssistOnMount;
  const autoRunInitialPromptOnMount = payload.autoRunInitialPromptOnMount;

  if (
    !openBrowserAssistOnMount &&
    !hasSiteSkillLaunch &&
    !hasInitialInputCapability &&
    !resolvedProjectId
  ) {
    return {
      ok: false,
      reason: "missing_project",
    };
  }

  if (
    !openBrowserAssistOnMount &&
    !hasSiteSkillLaunch &&
    !hasPendingServiceSkillLaunch &&
    !hasInitialInputCapability &&
    !hasPrompt &&
    !hasImages &&
    !hasContentId
  ) {
    return {
      ok: false,
      reason: "empty_payload",
    };
  }

  const nextNewChatAt = now();
  const shared = {
    projectId: resolvedProjectId,
    contentId: payload.contentId,
    theme: targetTheme,
    lockTheme,
    initialCreationMode: creationMode,
    initialUserPrompt: payload.prompt,
    initialUserImages: payload.images,
    initialRequestMetadata: payload.initialRequestMetadata,
    ...(payload.initialAutoSendRequestMetadata
      ? {
          initialAutoSendRequestMetadata:
            payload.initialAutoSendRequestMetadata,
        }
      : {}),
    ...(autoRunInitialPromptOnMount !== undefined
      ? { autoRunInitialPromptOnMount }
      : {}),
    openBrowserAssistOnMount,
    ...(payload.initialSiteSkillLaunch
      ? { initialSiteSkillLaunch: payload.initialSiteSkillLaunch }
      : {}),
    ...(payload.initialPendingServiceSkillLaunch
      ? {
          initialPendingServiceSkillLaunch: {
            ...payload.initialPendingServiceSkillLaunch,
            requestKey:
              payload.initialPendingServiceSkillLaunch.requestKey ??
              nextNewChatAt,
          },
        }
      : {}),
    ...(payload.initialInputCapability
      ? {
          initialInputCapability: {
            ...payload.initialInputCapability,
            requestKey:
              payload.initialInputCapability.requestKey ?? nextNewChatAt,
          },
        }
      : {}),
    ...(payload.initialProjectFileOpenTarget
      ? {
          initialProjectFileOpenTarget: payload.initialProjectFileOpenTarget,
        }
      : {}),
    newChatAt: nextNewChatAt,
  } satisfies AgentPageParams;

  return {
    ok: true,
    toolPreferences,
    targetTheme,
    nextNewChatAt,
    navigationParams: buildClawAgentParams(shared),
    workspaceBootstrap: {
      projectId: resolvedProjectId,
      contentId: payload.contentId,
      initialUserPrompt: payload.prompt,
      initialUserImages: payload.images,
      initialRequestMetadata: payload.initialRequestMetadata,
      ...(payload.initialAutoSendRequestMetadata
        ? {
            initialAutoSendRequestMetadata:
              payload.initialAutoSendRequestMetadata,
          }
        : {}),
      autoRunInitialPromptOnMount,
      theme: targetTheme,
      lockTheme,
      initialCreationMode: creationMode,
      openBrowserAssistOnMount,
      ...(payload.initialSiteSkillLaunch
        ? { initialSiteSkillLaunch: payload.initialSiteSkillLaunch }
        : {}),
      ...(payload.initialPendingServiceSkillLaunch
        ? {
            initialPendingServiceSkillLaunch: {
              ...payload.initialPendingServiceSkillLaunch,
              requestKey:
                payload.initialPendingServiceSkillLaunch.requestKey ??
                nextNewChatAt,
            },
          }
        : {}),
      ...(payload.initialInputCapability
        ? {
            initialInputCapability: {
              ...payload.initialInputCapability,
              requestKey:
                payload.initialInputCapability.requestKey ?? nextNewChatAt,
            },
          }
        : {}),
      ...(payload.initialProjectFileOpenTarget
        ? {
            initialProjectFileOpenTarget: payload.initialProjectFileOpenTarget,
          }
        : {}),
      newChatAt: nextNewChatAt,
    },
  };
}
