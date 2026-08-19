import type { CanvasState as GeneralCanvasState } from "@/components/general-chat/bridge";
import type { Message } from "../types";
import { useWorkspaceArtifactSelectionRuntime } from "./useWorkspaceArtifactSelectionRuntime";
import { useWorkspaceArtifactStoreRuntime } from "./useWorkspaceArtifactStoreRuntime";
import { useWorkspaceArtifactViewModeControl } from "./useWorkspaceArtifactViewModeControl";
import { useWorkspaceBrowserAssistRequestRuntime } from "./useWorkspaceBrowserAssistRequestRuntime";

interface UseWorkspaceArtifactCanvasRuntimeParams {
  activeTheme: string;
  generalCanvasState: GeneralCanvasState;
  isSending: boolean;
  messages: Message[];
  projectId?: string | null;
  sessionId?: string | null;
}

export function useWorkspaceArtifactCanvasRuntime({
  activeTheme,
  generalCanvasState,
  isSending,
  messages,
  projectId,
  sessionId,
}: UseWorkspaceArtifactCanvasRuntimeParams) {
  const selection = useWorkspaceArtifactSelectionRuntime({
    activeTheme,
    generalCanvasState,
  });
  const { handleOpenBrowserRuntimeForBrowserAssist } =
    useWorkspaceBrowserAssistRequestRuntime({
      projectId,
      sessionId,
    });
  const store = useWorkspaceArtifactStoreRuntime({
    activeTheme,
    artifacts: selection.artifacts,
    defaultSelectedArtifactId: selection.defaultSelectedArtifactId,
    isSending,
    liveArtifact: selection.liveArtifact,
    messages,
    preferGeneralCanvasFilePreview: selection.preferGeneralCanvasFilePreview,
    selectedArtifact: selection.selectedArtifact,
    selectedArtifactId: selection.selectedArtifactId,
    setArtifacts: selection.setArtifacts,
    setSelectedArtifactId: selection.setSelectedArtifactId,
    upsertGeneralArtifact: selection.upsertGeneralArtifact,
  });
  const viewMode = useWorkspaceArtifactViewModeControl({
    activeTheme,
    displayedArtifact: store.displayedCanvasArtifact,
    activeArtifactId: store.activeArtifactViewTargetId,
  });

  return {
    ...selection,
    ...store,
    ...viewMode,
    handleOpenBrowserRuntimeForBrowserAssist,
  };
}
