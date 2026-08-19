import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import {
  createInitialCanvasState,
  type CanvasStateUnion,
} from "@/components/workspace/canvas/canvasUtils";
import { createInitialDocumentState } from "@/components/workspace/canvas/canvasUtils";
import {
  DEFAULT_CANVAS_STATE,
  type CanvasState as GeneralCanvasState,
} from "@/components/general-chat/bridge";
import type { LayoutMode, ThemeType } from "@/lib/workspace/workbenchContract";
import { isSpecializedWorkbenchTheme } from "@/lib/workspace/workbenchContract";
import type { CanvasWorkbenchLayoutMode } from "../components/CanvasWorkbenchLayout";

const FALLBACK_CANVAS_CONTENT = "";

interface UseWorkspaceCanvasLayoutRuntimeParams {
  activeTheme: string;
  isThemeWorkbench: boolean;
  hasPendingA2UIForm: boolean;
  layoutMode: LayoutMode;
  showChatPanel: boolean;
  showSidebar: boolean;
  defaultTopicSidebarVisible: boolean;
  hasMessages: boolean;
  canvasWorkbenchLayoutMode: CanvasWorkbenchLayoutMode;
  autoCollapsedTopicSidebarRef: MutableRefObject<boolean>;
  mappedTheme: ThemeType;
  normalizedEntryTheme: ThemeType;
  shouldPreserveBlankHomeSurface: boolean;
  shouldBootstrapCanvasOnEntry: boolean;
  canvasState: CanvasStateUnion | null;
  generalCanvasState: GeneralCanvasState;
  hasCurrentCanvasArtifact: boolean;
  currentImageWorkbenchActive: boolean;
  onHasMessagesChange?: (hasMessages: boolean) => void;
  setShowSidebar: Dispatch<SetStateAction<boolean>>;
  setLayoutMode: Dispatch<SetStateAction<LayoutMode>>;
  setGeneralCanvasState: Dispatch<SetStateAction<GeneralCanvasState>>;
  setCanvasState: Dispatch<SetStateAction<CanvasStateUnion | null>>;
  setCanvasWorkbenchLayoutMode: Dispatch<
    SetStateAction<CanvasWorkbenchLayoutMode>
  >;
}

export function useWorkspaceCanvasLayoutRuntime({
  activeTheme,
  isThemeWorkbench,
  hasPendingA2UIForm,
  layoutMode,
  showChatPanel,
  showSidebar,
  defaultTopicSidebarVisible,
  hasMessages,
  canvasWorkbenchLayoutMode,
  autoCollapsedTopicSidebarRef,
  mappedTheme,
  normalizedEntryTheme,
  shouldPreserveBlankHomeSurface,
  shouldBootstrapCanvasOnEntry,
  canvasState,
  generalCanvasState,
  hasCurrentCanvasArtifact,
  currentImageWorkbenchActive,
  onHasMessagesChange,
  setShowSidebar,
  setLayoutMode,
  setGeneralCanvasState,
  setCanvasState,
  setCanvasWorkbenchLayoutMode,
}: UseWorkspaceCanvasLayoutRuntimeParams) {
  const previousThemeWorkbenchStateRef = useRef(false);

  useEffect(() => {
    autoCollapsedTopicSidebarRef.current = false;
    setShowSidebar(defaultTopicSidebarVisible);
  }, [
    autoCollapsedTopicSidebarRef,
    defaultTopicSidebarVisible,
    setShowSidebar,
  ]);

  useEffect(() => {
    if (showChatPanel) {
      setLayoutMode((previous) =>
        previous === "canvas" ? "chat-canvas" : previous,
      );
      return;
    }

    setShowSidebar(false);

    if (shouldPreserveBlankHomeSurface) {
      if (layoutMode !== "chat") {
        setLayoutMode("chat");
      }

      if (activeTheme === "general") {
        const isDefaultGeneralCanvasState =
          generalCanvasState.isOpen === DEFAULT_CANVAS_STATE.isOpen &&
          generalCanvasState.contentType === DEFAULT_CANVAS_STATE.contentType &&
          generalCanvasState.content === DEFAULT_CANVAS_STATE.content &&
          generalCanvasState.isEditing === DEFAULT_CANVAS_STATE.isEditing;

        if (!isDefaultGeneralCanvasState) {
          setGeneralCanvasState(DEFAULT_CANVAS_STATE);
        }
      }

      if (canvasState) {
        setCanvasState(null);
      }

      return;
    }

    if (layoutMode === "canvas") {
      return;
    }

    if (layoutMode === "chat-canvas") {
      setLayoutMode("canvas");
      return;
    }

    if (activeTheme === "general") {
      if (generalCanvasState.content.trim().length === 0) {
        return;
      }

      setGeneralCanvasState((previous) => {
        const contentType =
          previous.contentType === "empty" ? "markdown" : previous.contentType;
        if (previous.isOpen && previous.contentType === contentType) {
          return previous;
        }
        return {
          ...previous,
          isOpen: true,
          contentType,
        };
      });
    } else if (!canvasState) {
      const initialState =
        createInitialCanvasState(mappedTheme, FALLBACK_CANVAS_CONTENT) ||
        createInitialDocumentState(FALLBACK_CANVAS_CONTENT);
      setCanvasState(initialState);
    }

    setLayoutMode("canvas");
  }, [
    activeTheme,
    canvasState,
    generalCanvasState.content,
    generalCanvasState.contentType,
    generalCanvasState.isEditing,
    generalCanvasState.isOpen,
    layoutMode,
    mappedTheme,
    setCanvasState,
    setGeneralCanvasState,
    setLayoutMode,
    setShowSidebar,
    showChatPanel,
    shouldPreserveBlankHomeSurface,
  ]);

  useEffect(() => {
    const wasThemeWorkbench = previousThemeWorkbenchStateRef.current;
    previousThemeWorkbenchStateRef.current = isThemeWorkbench;

    if (
      !isThemeWorkbench ||
      wasThemeWorkbench ||
      !showChatPanel ||
      hasPendingA2UIForm
    ) {
      return;
    }

    setShowSidebar(true);
  }, [hasPendingA2UIForm, isThemeWorkbench, setShowSidebar, showChatPanel]);

  useEffect(() => {
    if (!hasPendingA2UIForm) {
      return;
    }

    if (isThemeWorkbench && showSidebar) {
      setShowSidebar(false);
    }

    if (layoutMode !== "chat") {
      setLayoutMode("chat");
    }

    if (activeTheme === "general") {
      setGeneralCanvasState((previous) =>
        previous.isOpen ? { ...previous, isOpen: false } : previous,
      );
    }
  }, [
    activeTheme,
    hasPendingA2UIForm,
    isThemeWorkbench,
    layoutMode,
    setGeneralCanvasState,
    setLayoutMode,
    setShowSidebar,
    showSidebar,
  ]);

  useEffect(() => {
    if (
      isThemeWorkbench ||
      activeTheme !== "general" ||
      layoutMode !== "chat-canvas"
    ) {
      setCanvasWorkbenchLayoutMode("split");
    }
  }, [activeTheme, isThemeWorkbench, layoutMode, setCanvasWorkbenchLayoutMode]);

  useEffect(() => {
    if (activeTheme !== "general") {
      return;
    }

    if (!hasCurrentCanvasArtifact && !currentImageWorkbenchActive) {
      return;
    }

    setGeneralCanvasState((previous) =>
      previous.isOpen ? { ...previous, isOpen: false } : previous,
    );
  }, [
    activeTheme,
    currentImageWorkbenchActive,
    hasCurrentCanvasArtifact,
    setGeneralCanvasState,
  ]);

  useEffect(() => {
    if (
      isThemeWorkbench ||
      activeTheme !== "general" ||
      !showChatPanel ||
      layoutMode === "chat" ||
      hasCurrentCanvasArtifact ||
      currentImageWorkbenchActive
    ) {
      return;
    }

    if (generalCanvasState.isOpen) {
      return;
    }

    setLayoutMode("chat");
  }, [
    activeTheme,
    currentImageWorkbenchActive,
    generalCanvasState,
    hasCurrentCanvasArtifact,
    isThemeWorkbench,
    layoutMode,
    setLayoutMode,
    showChatPanel,
  ]);

  useEffect(() => {
    const shouldAutoHideTopicSidebar =
      showChatPanel &&
      !isThemeWorkbench &&
      activeTheme === "general" &&
      layoutMode === "chat-canvas" &&
      canvasWorkbenchLayoutMode === "stacked";

    if (shouldAutoHideTopicSidebar) {
      if (showSidebar) {
        autoCollapsedTopicSidebarRef.current = true;
        setShowSidebar(false);
      }
      return;
    }

    const shouldRestoreAutoCollapsedSidebar =
      autoCollapsedTopicSidebarRef.current &&
      !showSidebar &&
      (!showChatPanel ||
        isThemeWorkbench ||
        activeTheme !== "general" ||
        layoutMode !== "chat-canvas");

    if (shouldRestoreAutoCollapsedSidebar) {
      autoCollapsedTopicSidebarRef.current = false;
      setShowSidebar(true);
    }
  }, [
    activeTheme,
    autoCollapsedTopicSidebarRef,
    canvasWorkbenchLayoutMode,
    isThemeWorkbench,
    layoutMode,
    setShowSidebar,
    showChatPanel,
    showSidebar,
  ]);

  useEffect(() => {
    onHasMessagesChange?.(hasMessages);
  }, [hasMessages, onHasMessagesChange]);

  const handleToggleCanvas = useCallback(() => {
    if (activeTheme === "general" && !isThemeWorkbench) {
      const shouldManageStandaloneGeneralCanvas =
        !hasCurrentCanvasArtifact && !currentImageWorkbenchActive;

      if (layoutMode !== "chat") {
        if (shouldManageStandaloneGeneralCanvas) {
          setGeneralCanvasState((previous) => ({ ...previous, isOpen: false }));
        }
        setLayoutMode("chat");
        return;
      }

      if (shouldManageStandaloneGeneralCanvas) {
        setGeneralCanvasState((previous) => ({
          ...previous,
          isOpen: true,
          contentType:
            previous.contentType === "empty"
              ? "markdown"
              : previous.contentType,
          content: previous.content || FALLBACK_CANVAS_CONTENT,
        }));
      }
      setLayoutMode("chat-canvas");
      return;
    }

    setLayoutMode((previous) => {
      if (previous === "chat") {
        if (!canvasState) {
          const initialState =
            createInitialCanvasState(mappedTheme, FALLBACK_CANVAS_CONTENT) ||
            createInitialDocumentState(FALLBACK_CANVAS_CONTENT);
          setCanvasState(initialState);
        }
        return "chat-canvas";
      }

      return "chat";
    });
  }, [
    activeTheme,
    canvasState,
    currentImageWorkbenchActive,
    hasCurrentCanvasArtifact,
    isThemeWorkbench,
    layoutMode,
    mappedTheme,
    setCanvasState,
    setGeneralCanvasState,
    setLayoutMode,
  ]);

  const handleCloseCanvas = useCallback(() => {
    setLayoutMode("chat");
    if (activeTheme === "general") {
      setGeneralCanvasState((previous) => ({ ...previous, isOpen: false }));
    }
  }, [activeTheme, setGeneralCanvasState, setLayoutMode]);

  const resolvedCanvasState = useMemo<CanvasStateUnion | null>(() => {
    if (canvasState) {
      return canvasState;
    }

    if (shouldBootstrapCanvasOnEntry) {
      return (
        createInitialCanvasState(normalizedEntryTheme, "") ||
        createInitialDocumentState("")
      );
    }

    if (isThemeWorkbench && isSpecializedWorkbenchTheme(activeTheme)) {
      return (
        createInitialCanvasState(mappedTheme, "") ||
        createInitialDocumentState("")
      );
    }

    return null;
  }, [
    activeTheme,
    canvasState,
    isThemeWorkbench,
    mappedTheme,
    normalizedEntryTheme,
    shouldBootstrapCanvasOnEntry,
  ]);

  return {
    handleToggleCanvas,
    handleCloseCanvas,
    resolvedCanvasState,
  };
}
