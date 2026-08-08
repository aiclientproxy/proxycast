import { useCallback, useEffect, useMemo, useState } from "react";
import type { ExpertAgentLaunchParams } from "@/types/page";
import { useExpertWorkspaceSkillRuntime } from "./useExpertWorkspaceSkillRuntime";
import { useWorkspaceExpertAgentLaunchSyncRuntime } from "./useWorkspaceExpertAgentLaunchSyncRuntime";
import { useWorkspacePluginCatalogSuggestions } from "./useWorkspacePluginCatalogSuggestions";
import {
  resolveExpertPanelRequestMetadata,
  resolveSessionExpertRequestMetadata,
  resolveWorkspaceRequestMetadataWithExpertSkills,
} from "./workspaceExpertMetadata";

interface UseWorkspaceExpertSkillPanelRuntimeParams {
  activeSessionKey: string | null;
  activeTheme: string;
  deferredDelayMs?: number;
  expertAgentLaunch?: ExpertAgentLaunchParams | null;
  initialAutoSendRequestMetadata?: Record<string, unknown>;
  initialRequestMetadata?: Record<string, unknown>;
  newChatAt?: number;
  onOpenSkillsManage?: () => void;
  serviceSkillsLoading: boolean;
  skillsLoading: boolean;
  threadRead?: Parameters<typeof resolveSessionExpertRequestMetadata>[0];
  workspaceRoot?: string | null;
}

export function useWorkspaceExpertSkillPanelRuntime({
  activeSessionKey,
  activeTheme,
  deferredDelayMs,
  expertAgentLaunch,
  initialAutoSendRequestMetadata,
  initialRequestMetadata,
  newChatAt,
  onOpenSkillsManage,
  serviceSkillsLoading,
  skillsLoading,
  threadRead,
  workspaceRoot,
}: UseWorkspaceExpertSkillPanelRuntimeParams) {
  const [
    threadExpertRequestMetadataOverride,
    setThreadExpertRequestMetadataOverride,
  ] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    setThreadExpertRequestMetadataOverride(null);
  }, [activeSessionKey, newChatAt]);

  const handleThreadExpertProfileSwitch = useCallback(
    (requestMetadata: Record<string, unknown>) => {
      setThreadExpertRequestMetadataOverride({ ...requestMetadata });
    },
    [],
  );

  const sessionExpertRequestMetadata = useMemo(
    () => resolveSessionExpertRequestMetadata(threadRead),
    [threadRead],
  );
  const baseExpertPanelRequestMetadata = useMemo(
    () =>
      resolveExpertPanelRequestMetadata({
        initialAutoSendRequestMetadata,
        initialRequestMetadata,
        sessionRequestMetadata: sessionExpertRequestMetadata,
      }),
    [
      initialAutoSendRequestMetadata,
      initialRequestMetadata,
      sessionExpertRequestMetadata,
    ],
  );
  const expertPanelRequestMetadata =
    threadExpertRequestMetadataOverride ?? baseExpertPanelRequestMetadata;

  const expertWorkspaceSkillRuntime = useExpertWorkspaceSkillRuntime({
    activeTheme,
    requestMetadata: expertPanelRequestMetadata,
    workspaceRoot,
    deferredDelayMs,
    onOpenSkillsManage,
  });
  const workspaceSkillBindings =
    expertWorkspaceSkillRuntime.bindingsRuntime.bindings;
  const combinedSkillsLoading =
    skillsLoading ||
    serviceSkillsLoading ||
    expertWorkspaceSkillRuntime.bindingsRuntime.loading;
  const { expertSkillRefsOverride, handleExpertSkillRefsChange } =
    useWorkspaceExpertAgentLaunchSyncRuntime({
      expertAgentLaunch: threadExpertRequestMetadataOverride
        ? null
        : expertAgentLaunch,
      expertPanelRequestMetadata,
      pruneWorkspaceSkillRuntimeEnableRefs:
        expertWorkspaceSkillRuntime.pruneEnabledRefsForSkillRefs,
    });

  const workspaceRequestMetadataWithExpertSkills = useMemo(
    () =>
      resolveWorkspaceRequestMetadataWithExpertSkills({
        activeRequestMetadata: threadExpertRequestMetadataOverride,
        expertSkillRefsOverride,
        initialAutoSendRequestMetadata,
        initialRequestMetadata,
        sessionRequestMetadata: sessionExpertRequestMetadata,
      }),
    [
      threadExpertRequestMetadataOverride,
      expertSkillRefsOverride,
      initialAutoSendRequestMetadata,
      initialRequestMetadata,
      sessionExpertRequestMetadata,
    ],
  );
  const [pluginSuggestionsEnabled, setPluginSuggestionsEnabled] =
    useState(false);
  const workspacePluginCatalogSuggestions =
    useWorkspacePluginCatalogSuggestions({ enabled: pluginSuggestionsEnabled });
  const refreshWorkspacePluginCatalogSuggestions =
    workspacePluginCatalogSuggestions.refresh;
  const handlePluginSuggestionsNeeded = useCallback(() => {
    setPluginSuggestionsEnabled(true);
    refreshWorkspacePluginCatalogSuggestions();
  }, [refreshWorkspacePluginCatalogSuggestions]);

  return {
    combinedSkillsLoading,
    expertPanelRequestMetadata,
    expertPanelRuntimeKey: expertWorkspaceSkillRuntime.runtimeKey,
    expertSkillRefsOverride,
    expertWorkspaceSkillRuntimeEnableBindings:
      expertWorkspaceSkillRuntime.enabledBindings,
    expertWorkspaceSkillRuntimeEnableInput:
      expertWorkspaceSkillRuntime.enableInput,
    expertWorkspaceSkillRuntimeEnableRefs:
      expertWorkspaceSkillRuntime.enabledRefs,
    handleEnableExpertWorkspaceSkillRuntime:
      expertWorkspaceSkillRuntime.handleEnableWorkspaceSkillRuntime,
    handleExpertSkillRefsChange,
    handlePluginSuggestionsNeeded,
    handleThreadExpertProfileSwitch,
    workspacePluginInputSuggestions:
      workspacePluginCatalogSuggestions.suggestions,
    workspacePluginSuggestionsError: workspacePluginCatalogSuggestions.error,
    workspacePluginSuggestionsLoading:
      workspacePluginCatalogSuggestions.loading,
    workspaceRequestMetadataWithExpertSkills,
    workspaceSkillBindings,
  };
}
