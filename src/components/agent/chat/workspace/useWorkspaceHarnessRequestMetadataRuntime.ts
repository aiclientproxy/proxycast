import { useMemo } from "react";
import type { AgentRuntimeWorkspaceSkillBinding } from "@/lib/api/agentRuntime/toolInventoryTypes";
import { buildHarnessRequestMetadata } from "../utils/harnessRequestMetadata";
import type { ChatToolPreferences } from "../utils/chatToolPreferences";
import type { WorkspaceSkillRuntimeEnableInput } from "../utils/workspaceSkillBindingsMetadata";

interface WorkspaceHarnessRequestMetadataParams {
  enabled?: boolean;
  agentResponseLanguage?: string | null;
  contentId?: string | null;
  currentGateKey?: string | null;
  effectiveChatToolPreferences: Pick<ChatToolPreferences, "task" | "subagent">;
  isThemeWorkbench: boolean;
  mappedTheme: string;
  themeWorkbenchActiveQueueTitle?: string | null;
  workspaceSkillBindings?: AgentRuntimeWorkspaceSkillBinding[];
  workspaceSkillRuntimeEnable?: WorkspaceSkillRuntimeEnableInput | null;
}

const EMPTY_WORKSPACE_HARNESS_REQUEST_METADATA: Record<string, unknown> =
  Object.freeze({});

export function resolveWorkspaceHarnessRequestMetadata({
  enabled = true,
  agentResponseLanguage,
  contentId,
  currentGateKey,
  effectiveChatToolPreferences,
  isThemeWorkbench,
  mappedTheme,
  themeWorkbenchActiveQueueTitle,
  workspaceSkillBindings,
  workspaceSkillRuntimeEnable,
}: WorkspaceHarnessRequestMetadataParams): Record<string, unknown> {
  if (!enabled) {
    return EMPTY_WORKSPACE_HARNESS_REQUEST_METADATA;
  }

  return buildHarnessRequestMetadata({
    theme: mappedTheme,
    preferences: {
      task: effectiveChatToolPreferences.task,
      subagent: effectiveChatToolPreferences.subagent,
    },
    sessionMode: isThemeWorkbench ? "general_workbench" : "default",
    gateKey: isThemeWorkbench ? currentGateKey : undefined,
    runTitle: themeWorkbenchActiveQueueTitle?.trim() || undefined,
    contentId: contentId || undefined,
    workspaceSkillBindings:
      workspaceSkillBindings && workspaceSkillBindings.length > 0
        ? workspaceSkillBindings
        : undefined,
    workspaceSkillRuntimeEnable,
    agentResponseLanguage,
  });
}

export function useWorkspaceHarnessRequestMetadataRuntime(
  params: WorkspaceHarnessRequestMetadataParams,
): Record<string, unknown> {
  const {
    enabled = true,
    agentResponseLanguage,
    contentId,
    currentGateKey,
    effectiveChatToolPreferences,
    isThemeWorkbench,
    mappedTheme,
    themeWorkbenchActiveQueueTitle,
    workspaceSkillBindings,
    workspaceSkillRuntimeEnable,
  } = params;
  const taskPreferenceEnabled = effectiveChatToolPreferences.task;
  const subagentPreferenceEnabled = effectiveChatToolPreferences.subagent;

  return useMemo(
    () =>
      resolveWorkspaceHarnessRequestMetadata({
        enabled,
        agentResponseLanguage,
        contentId,
        currentGateKey,
        effectiveChatToolPreferences: {
          task: taskPreferenceEnabled,
          subagent: subagentPreferenceEnabled,
        },
        isThemeWorkbench,
        mappedTheme,
        themeWorkbenchActiveQueueTitle,
        workspaceSkillBindings,
        workspaceSkillRuntimeEnable,
      }),
    [
      agentResponseLanguage,
      contentId,
      currentGateKey,
      enabled,
      isThemeWorkbench,
      mappedTheme,
      subagentPreferenceEnabled,
      taskPreferenceEnabled,
      themeWorkbenchActiveQueueTitle,
      workspaceSkillBindings,
      workspaceSkillRuntimeEnable,
    ],
  );
}
