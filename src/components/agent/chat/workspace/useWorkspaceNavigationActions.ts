import { useCallback, type Dispatch, type SetStateAction } from "react";
import { toast } from "sonner";
import { requestChatHostOpenPath } from "../host/chatHostCapabilities";
import { updateProject as updateProjectById } from "@/lib/api/project";
import { notifyProjectRuntimeAgentsGuide } from "@/components/workspace/services/runtimeAgentsGuideService";
import type {
  ExecutionPolicyFocusContext,
  Page,
  PageParams,
  ProviderSettingsFocusContext,
} from "@/types/page";
import { SettingsTabs } from "@/types/settings";
import type { WorkspacePathMissingState } from "../hooks/agentChatShared";

interface UseWorkspaceNavigationActionsParams {
  applyProjectSelection: (projectId: string | null) => void;
  compactSession: () => Promise<void>;
  dismissWorkspacePathError: () => void;
  fixWorkspacePathAndRetry: (newPath: string) => Promise<void>;
  agentEntry?: "new-task" | "claw";
  externalProjectId?: string | null;
  onNavigate?: (page: Page, params?: PageParams) => void;
  projectId?: string;
  setEntryBannerVisible: Dispatch<SetStateAction<boolean>>;
  setWorkspaceHealthError: Dispatch<SetStateAction<boolean>>;
  workspacePathMissing: WorkspacePathMissingState | boolean | null;
}

interface OpenRuntimeMemoryWorkbenchParams {
  sessionId?: string | null;
  workingDir?: string | null;
  userMessage?: string | null;
}

export function useWorkspaceNavigationActions({
  applyProjectSelection,
  compactSession,
  dismissWorkspacePathError,
  fixWorkspacePathAndRetry,
  agentEntry = "claw",
  externalProjectId,
  onNavigate,
  projectId,
  setEntryBannerVisible,
  setWorkspaceHealthError,
  workspacePathMissing,
}: UseWorkspaceNavigationActionsParams) {
  const handleManageProviders = useCallback(
    (focus?: ProviderSettingsFocusContext) => {
      onNavigate?.("settings", {
        tab: SettingsTabs.Providers,
        providerView: "settings",
        ...(focus ? { providerFocus: focus } : {}),
      });
    },
    [onNavigate],
  );

  const handleOpenExecutionPolicySettings = useCallback(
    (focus?: ExecutionPolicyFocusContext) => {
      onNavigate?.("settings", {
        tab: SettingsTabs.ExecutionPolicy,
        ...(focus ? { executionPolicyFocus: focus } : {}),
      });
    },
    [onNavigate],
  );

  const handleBackToResources = useCallback(() => {
    onNavigate?.("resources");
  }, [onNavigate]);

  const handleProjectChange = useCallback(
    (newProjectId: string | null) => {
      const normalizedProjectId = newProjectId?.trim() || null;
      applyProjectSelection(normalizedProjectId);

      const normalizedExternalProjectId = externalProjectId?.trim() || null;
      if (
        !onNavigate ||
        !normalizedExternalProjectId ||
        normalizedExternalProjectId === normalizedProjectId
      ) {
        return;
      }

      onNavigate("agent", {
        agentEntry,
        ...(normalizedProjectId ? { projectId: normalizedProjectId } : {}),
      });
    },
    [agentEntry, applyProjectSelection, externalProjectId, onNavigate],
  );

  const handleSelectWorkspaceDirectory = useCallback(async () => {
    const newPath = await requestChatHostOpenPath({
      directory: true,
      multiple: false,
    });
    if (!newPath) {
      return;
    }

    if (workspacePathMissing) {
      await fixWorkspacePathAndRetry(newPath);
      return;
    }

    if (!projectId) {
      return;
    }

    try {
      await updateProjectById(projectId, { rootPath: newPath });
      setWorkspaceHealthError(false);
      notifyProjectRuntimeAgentsGuide(
        {
          id: projectId,
          rootPath: newPath,
        },
        {
          successMessage: "工作区目录已更新",
        },
      );
    } catch (error) {
      toast.error(
        `更新路径失败: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }, [
    fixWorkspacePathAndRetry,
    projectId,
    setWorkspaceHealthError,
    workspacePathMissing,
  ]);

  const handleOpenAppearanceSettings = useCallback(() => {
    onNavigate?.("settings", {
      tab: SettingsTabs.Appearance,
    });
  }, [onNavigate]);

  const handleOpenRuntimeMemoryWorkbench = useCallback(
    (_params?: OpenRuntimeMemoryWorkbenchParams) => {
      onNavigate?.("settings", {
        tab: SettingsTabs.Memory,
      });
    },
    [onNavigate],
  );

  const handleOpenKnowledgeManagement = useCallback(
    (workingDir?: string | null) => {
      onNavigate?.(
        "knowledge",
        workingDir?.trim()
          ? {
              workingDir: workingDir.trim(),
            }
          : undefined,
      );
    },
    [onNavigate],
  );

  const handleOpenChannels = useCallback(() => {
    onNavigate?.("channels");
  }, [onNavigate]);

  const handleCompactContext = useCallback(() => {
    void compactSession();
  }, [compactSession]);

  const handleDismissEntryBanner = useCallback(() => {
    setEntryBannerVisible(false);
  }, [setEntryBannerVisible]);

  const handleWorkspaceAlertSelectDirectory = useCallback(() => {
    void handleSelectWorkspaceDirectory();
  }, [handleSelectWorkspaceDirectory]);

  const handleDismissWorkspaceAlert = useCallback(() => {
    setWorkspaceHealthError(false);
    dismissWorkspacePathError();
  }, [dismissWorkspacePathError, setWorkspaceHealthError]);

  return {
    handleBackToResources,
    handleCompactContext,
    handleDismissEntryBanner,
    handleDismissWorkspaceAlert,
    handleManageProviders,
    handleOpenChannels,
    handleOpenExecutionPolicySettings,
    handleOpenKnowledgeManagement,
    handleOpenRuntimeMemoryWorkbench,
    handleOpenAppearanceSettings,
    handleProjectChange,
    handleWorkspaceAlertSelectDirectory,
  };
}
