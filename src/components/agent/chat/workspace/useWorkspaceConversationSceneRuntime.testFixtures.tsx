import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, vi } from "vitest";
import { useWorkspaceConversationSceneRuntime } from "./useWorkspaceConversationSceneRuntime";

vi.mock("react-i18next", async () => {
  const { agentZhCNResource } = await import("@/i18n/agentResources");
  const agentZhCN = agentZhCNResource as Record<string, string>;

  return {
    useTranslation: () => ({
      i18n: {
        language: "zh-CN",
      },
      t: (key: string, options?: Record<string, unknown>) => {
        const template = agentZhCN[key] ?? key;
        return template.replace(/{{\s*([^}]+?)\s*}}/g, (_, name: string) =>
          String(options?.[name.trim()] ?? ""),
        );
      },
    }),
  };
});

vi.mock("react-syntax-highlighter/dist/esm/prism", () => ({
  default: ({ children }: { children?: unknown }) =>
    React.createElement(
      "pre",
      { "data-testid": "syntax-highlighter-mock" },
      String(children ?? ""),
    ),
}));

vi.mock("react-syntax-highlighter/dist/esm/styles/prism", () => ({
  oneLight: {},
  oneDark: {},
}));

export type HookProps = Parameters<
  typeof useWorkspaceConversationSceneRuntime
>[0];

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

export function createBaseParams(overrides: Record<string, unknown> = {}) {
  const noop = vi.fn();
  const setCanvasWorkbenchLayoutMode = vi.fn();
  const messageListRuntimeOverrides =
    (overrides.messageListRuntime as Record<string, unknown> | undefined) ?? {};

  return {
    navigationActions: {
      handleDismissEntryBanner: noop,
      handleWorkspaceAlertSelectDirectory: noop,
      handleDismissWorkspaceAlert: noop,
      handleManageProviders: noop,
      handleOpenExecutionPolicySettings: noop,
      handleProjectChange: noop,
      handleOpenAppearanceSettings: noop,
      handleOpenChannels: noop,
      handleOpenChromeRelay: noop,
      handleBackToResources: noop,
      handleCompactContext: noop,
      handleOpenRuntimeMemoryWorkbench: noop,
    },
    inputbarScene: {
      inputbarNode: null,
      generalWorkbenchDialog: undefined,
      runtimeToolAvailability: null,
    },
    canvasScene: {
      hasLiveCanvasPreviewContent: false,
      liveCanvasPreview: null,
      shouldShowCanvasLoadingState: false,
      canvasWorkbenchDefaultPreview: null,
      handleOpenCanvasWorkbenchPath: noop,
      handleRevealCanvasWorkbenchPath: noop,
      handleCloseCanvasWorkbench: noop,
      renderCanvasWorkbenchPreview: noop,
    },
    handleSendFromEmptyState: noop,
    landingSurface: {
      entryBannerVisible: Boolean(overrides.entryBannerVisible),
      entryBannerMessage: overrides.entryBannerMessage as string | undefined,
      onDismissEntryBanner: noop,
      creationReplaySurface:
        overrides.creationReplaySurface === undefined
          ? null
          : (overrides.creationReplaySurface as never),
      sceneAppExecutionSummaryCard:
        overrides.sceneAppExecutionSummaryCard as React.ReactNode,
      serviceSkillExecutionCard:
        overrides.serviceSkillExecutionCard as React.ReactNode,
      emptyStateProps: {
        serviceSkills: overrides.serviceSkills ?? [],
        onSelectServiceSkill: overrides.onSelectServiceSkill ?? noop,
      } as never,
      ...((overrides.landingSurface as Record<string, unknown> | undefined) ??
        {}),
    },
    shellChromeRuntime: {
      showChatLayout: true,
      isWorkspaceCompactChrome: false,
      workflowLayoutBottomSpacing: {
        messageViewportBottomPadding: "0px",
        shellBottomInset: "0px",
      },
      shouldHideGeneralWorkbenchInputForTheme: false,
      shouldRenderTopBar: true,
      layoutTransitionChatPanelWidth: undefined,
      layoutTransitionChatPanelMinWidth: undefined,
      shouldShowGeneralWorkbenchFloatingInputOverlay: false,
      shouldRenderInlineA2UI: false,
    },
    entryBannerVisible: false,
    entryBannerMessage: undefined,
    sceneAppExecutionSummaryCard: undefined,
    serviceSkillExecutionCard: undefined,
    contextWorkspaceEnabled: false,
    canonicalChildren: overrides.canonicalChildren ?? [],
    messageListRuntime: {
      emptyStateVariant: overrides.messageListEmptyStateVariant,
      quoteInput: (overrides.input as string | undefined) ?? "",
      onQuoteInputChange:
        (overrides.setInput as ((value: string) => void) | undefined) ?? noop,
      providerType: overrides.providerType ?? "mock-provider",
      model: overrides.model ?? "mock-model",
      reasoningEffort: overrides.reasoningEffort,
      accessMode: overrides.accessMode ?? "default",
      messages: overrides.displayMessages ?? [],
      turns: overrides.turns ?? [],
      threadItems: overrides.effectiveThreadItems ?? [],
      todoItems: overrides.todoItems ?? [],
      currentTurnId: overrides.currentTurnId ?? null,
      threadRead: overrides.threadRead ?? false,
      executionRuntime: overrides.executionRuntime,
      pendingActions: overrides.pendingActions ?? [],
      submittedActionsInFlight: overrides.submittedActionsInFlight ?? [],
      sessionHistoryWindow: overrides.sessionHistoryWindow,
      onLoadFullHistory: overrides.loadFullSessionHistory,
      isSending: Boolean(overrides.isSending),
      onInterruptCurrentTurn: overrides.stopSending ?? noop,
      onReplayPendingRequest: overrides.replayPendingAction ?? noop,
      onDeleteMessage: overrides.deleteMessage ?? noop,
      onEditMessage: overrides.editMessage ?? noop,
      onA2UISubmit: overrides.handleA2UISubmit ?? noop,
      onWriteFile: overrides.handleWriteFile ?? noop,
      onFileClick: overrides.handleFileClick ?? noop,
      onOpenArtifactFromTimeline:
        overrides.handleOpenArtifactFromTimeline ?? noop,
      onOpenSavedSiteContent: overrides.handleOpenSavedSiteContent ?? noop,
      onArtifactClick: overrides.handleArtifactClick ?? noop,
      onOpenUrlPreview: overrides.handleOpenUrlPreview,
      onOpenMessagePreview: overrides.handleOpenMessagePreview,
      onSaveMessageAsSkill: overrides.handleSaveMessageAsSkill,
      onSaveMessageAsKnowledge: overrides.handleSaveMessageAsKnowledge,
      onOpenSubagentSession: overrides.handleOpenSubagentSession ?? noop,
      onPermissionResponse: overrides.handlePermissionResponse ?? noop,
      onRefreshSessionReadModel: overrides.onRefreshSessionReadModel,
      pendingPromotedA2UIActionRequest:
        overrides.pendingPromotedA2UIActionRequest ?? null,
      collapseCodeBlocks: Boolean(overrides.shouldCollapseCodeBlocks),
      shouldCollapseCodeBlock: overrides.shouldCollapseCodeBlockInChat ?? noop,
      onCodeBlockClick: overrides.handleCodeBlockClick ?? noop,
      focusedTimelineItemId: overrides.focusedTimelineItemId ?? null,
      timelineFocusRequestKey:
        (overrides.timelineFocusRequestKey as number | undefined) ?? 0,
      ...messageListRuntimeOverrides,
    },
    chatToolPreferences: {
      task: false,
      subagent: false,
    },
    setChatToolPreferences: noop,
    creationMode: "guided",
    setCreationMode: noop,
    activeTheme: "general",
    setActiveTheme: noop,
    lockTheme: false,
    artifacts: [],
    generalCanvasContent: "",
    resolvedCanvasState: null,
    contentId: null,
    selectedText: "",
    handleRecommendationClick: noop,
    projectCharacters: [],
    skills: [],
    serviceSkills: [],
    skillsLoading: false,
    onSelectServiceSkill: noop,
    handleNavigateToSkillSettings: noop,
    handleRefreshSkills: noop,
    handleOpenBrowserAssistInCanvas: noop,
    browserAssistLaunching: false,
    projectId: "project-1",
    showChatPanel: true,
    topBarChrome: "full",
    onBackToProjectManagement: undefined,
    fromResources: false,
    handleBackHome: noop,
    rightSurfaceChrome: {
      content: overrides.rightSurfaceContent as React.ReactNode,
      launchers: overrides.rightSurfaceLaunchers as React.ReactNode,
      objectCanvasOpen: Boolean(overrides.rightSurfaceObjectCanvasOpen),
      onToggleObjectCanvas:
        (overrides.onToggleRightSurfaceObjectCanvas as
          | (() => void)
          | undefined) ?? noop,
      browserOpen: Boolean(overrides.rightSurfaceBrowserOpen),
      onToggleBrowser:
        (overrides.onToggleRightSurfaceBrowser as (() => void) | undefined) ??
        noop,
      filesOpen: Boolean(overrides.rightSurfaceFilesOpen),
      onToggleFiles:
        (overrides.onToggleRightSurfaceFiles as (() => void) | undefined) ??
        noop,
      traceOpen: Boolean(overrides.rightSurfaceTraceOpen),
      onToggleTrace:
        (overrides.onToggleRightSurfaceTrace as (() => void) | undefined) ??
        noop,
      shellOpen: Boolean(overrides.rightSurfaceShellOpen),
      onToggleShell:
        (overrides.onToggleRightSurfaceShell as (() => void) | undefined) ??
        noop,
      showHarnessToggle: Boolean(overrides.showHarnessToggle),
      harnessPanelVisible: Boolean(overrides.navbarHarnessPanelVisible),
      onToggleHarnessPanel:
        (overrides.handleToggleHarnessPanel as (() => void) | undefined) ??
        noop,
      showExpertInfoToggle: Boolean(overrides.showExpertInfoToggle),
      expertInfoPanelVisible: Boolean(overrides.expertInfoPanelVisible),
      onToggleExpertInfoPanel:
        (overrides.handleToggleExpertInfoPanel as (() => void) | undefined) ??
        noop,
      harnessPendingCount: (overrides.harnessPendingCount as number) ?? 0,
      harnessAttentionLevel:
        (overrides.harnessAttentionLevel as "idle" | "active" | "warning") ??
        "idle",
      harnessToggleLabel: overrides.harnessToggleLabel as string | undefined,
      ...((overrides.rightSurfaceChrome as
        | Record<string, unknown>
        | undefined) ?? {}),
    },
    isRestoringSession: false,
    sessionId: null,
    syncStatus: "idle",
    pendingA2UIForm: undefined,
    pendingA2UISource: null,
    a2uiSubmissionNotice: undefined,
    handlePendingA2UISubmit: noop,
    handleToggleCanvas: noop,
    currentImageWorkbenchActive: false,
    hideInlineStepProgress: false,
    isSpecializedThemeMode: false,
    hasMessages: false,
    steps: [],
    currentStepIndex: 0,
    goToStep: noop,
    layoutMode: "chat-canvas",
    isThemeWorkbench: false,
    settledWorkbenchArtifacts: [],
    taskFiles: [],
    selectedFileId: undefined,
    projectRootPath: "/tmp/project-1",
    handleHarnessLoadFilePreview: noop,
    setCanvasWorkbenchLayoutMode,
    workspacePathMissing: false,
    workspaceHealthError: false,
    ...overrides,
  } as any;
}

export function renderHook(initialProps: HookProps) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latestValue: ReturnType<
    typeof useWorkspaceConversationSceneRuntime
  > | null = null;

  function Probe(currentProps: HookProps) {
    latestValue = useWorkspaceConversationSceneRuntime(currentProps);
    return null;
  }

  const render = (nextProps: HookProps) => {
    act(() => {
      root.render(React.createElement(Probe, nextProps));
    });
  };

  render(initialProps);
  mountedRoots.push({ root, container });

  return {
    getValue: () => {
      if (!latestValue) {
        throw new Error("hook 尚未初始化");
      }
      return latestValue;
    },
    render,
  };
}

export function getRenderedSceneProps(
  params: ReturnType<typeof createBaseParams>,
) {
  const { getValue } = renderHook(params);
  return (getValue().mainAreaNode as any).props;
}

export function buildHeavySessionRuntimeFixture(sessionId = "session") {
  const messages = Array.from({ length: 24 }, (_, index) => ({
    id: `${sessionId}-msg-${index}`,
    role: index % 2 === 0 ? "user" : "assistant",
    content: `${sessionId} 消息 ${index}`,
    timestamp: new Date(2026, 3, 30, 10, index),
  }));
  const turns = Array.from({ length: 6 }, (_, index) => ({
    id: `${sessionId}-turn-${index}`,
    thread_id: `${sessionId}-thread`,
    prompt_text: `${sessionId} 任务 ${index}`,
    status: "completed",
    started_at: `2026-04-30T10:0${index}:00.000Z`,
    created_at: `2026-04-30T10:0${index}:00.000Z`,
    updated_at: `2026-04-30T10:0${index}:01.000Z`,
  }));
  const threadItems = Array.from({ length: 28 }, (_, index) => ({
    id: `${sessionId}-item-${index}`,
    thread_id: `${sessionId}-thread`,
    turn_id: `${sessionId}-turn-${Math.min(5, Math.floor(index / 5))}`,
    sequence: index + 1,
    status: "completed",
    started_at: `2026-04-30T10:00:${String(index).padStart(2, "0")}.000Z`,
    updated_at: `2026-04-30T10:00:${String(index).padStart(2, "0")}.500Z`,
    type: "tool_call",
    tool_name: "Read",
    arguments: { index },
  }));

  return { messages, turns, threadItems };
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      break;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.restoreAllMocks();
});
