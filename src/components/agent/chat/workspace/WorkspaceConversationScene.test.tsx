import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WorkspaceConversationScene } from "./WorkspaceConversationScene";
import type { WorkspaceRightSurfaceLauncherProjection } from "./right-surface";

const {
  mockWorkspaceMainArea,
  mockWorkspacePendingA2UIPanel,
  mockMessageList,
  mockTaskCenterUtilityToolbar,
} = vi.hoisted(() => ({
  mockWorkspaceMainArea: vi.fn(),
  mockWorkspacePendingA2UIPanel: vi.fn(),
  mockMessageList: vi.fn(),
  mockTaskCenterUtilityToolbar: vi.fn(),
}));

vi.mock("../components/CanvasWorkbenchLayout", () => ({
  CanvasWorkbenchLayout: (props: { topRightTools?: React.ReactNode }) => (
    <div data-testid="canvas-layout-stub">
      {props.topRightTools ? (
        <div data-testid="canvas-layout-top-right-tools">
          {props.topRightTools}
        </div>
      ) : null}
    </div>
  ),
}));

vi.mock("../components/ChatNavbar", () => ({
  ChatNavbar: () => <div data-testid="chat-navbar-stub" />,
}));

vi.mock("../components/TaskCenterUtilityToolbar", () => ({
  TaskCenterUtilityToolbar: ({
    onToggleShellPanel,
    onToggleObjectCanvasPanel,
    onToggleBrowserPanel,
    onToggleFilesPanel,
    showHarnessToggle,
    harnessPanelVisible,
    onToggleHarnessPanel,
    showExpertInfoToggle,
    expertInfoPanelVisible,
    onToggleExpertInfoPanel,
    rightSurfaceLaunchers,
  }: {
    onToggleShellPanel?: () => void;
    onToggleObjectCanvasPanel?: () => void;
    onToggleBrowserPanel?: () => void;
    onToggleFilesPanel?: () => void;
    showHarnessToggle?: boolean;
    harnessPanelVisible?: boolean;
    onToggleHarnessPanel?: () => void;
    showExpertInfoToggle?: boolean;
    expertInfoPanelVisible?: boolean;
    onToggleExpertInfoPanel?: () => void;
    rightSurfaceLaunchers?: Array<{
      kind: string;
      active: boolean;
      disabled: boolean;
      pendingCount: number;
      collapseTarget: string;
    }>;
  }) => {
    const props = {
      onToggleShellPanel,
      onToggleObjectCanvasPanel,
      onToggleBrowserPanel,
      onToggleFilesPanel,
      showHarnessToggle,
      harnessPanelVisible,
      onToggleHarnessPanel,
      showExpertInfoToggle,
      expertInfoPanelVisible,
      onToggleExpertInfoPanel,
      rightSurfaceLaunchers,
    };
    mockTaskCenterUtilityToolbar(props);
    const harnessLauncher = rightSurfaceLaunchers?.find(
      (launcher) => launcher.kind === "harness",
    );
    const filesLauncher = rightSurfaceLaunchers?.find(
      (launcher) => launcher.kind === "files",
    );
    const objectCanvasLauncher = rightSurfaceLaunchers?.find(
      (launcher) => launcher.kind === "objectCanvas",
    );
    const browserLauncher = rightSurfaceLaunchers?.find(
      (launcher) => launcher.kind === "browser",
    );
    return (
      <>
        <button
          type="button"
          data-testid="task-center-utility-toolbar-stub"
          onClick={onToggleShellPanel}
        >
          toolbar
        </button>
        {onToggleObjectCanvasPanel ? (
          <button
            type="button"
            data-testid="task-center-object-canvas-toggle-stub"
            data-expanded={objectCanvasLauncher?.active ? "true" : "false"}
            data-disabled={objectCanvasLauncher?.disabled ? "true" : "false"}
            data-pending-count={String(objectCanvasLauncher?.pendingCount ?? 0)}
            onClick={onToggleObjectCanvasPanel}
          >
            object canvas
          </button>
        ) : null}
        {onToggleFilesPanel ? (
          <button
            type="button"
            data-testid="task-center-files-toggle-stub"
            data-expanded={filesLauncher?.active ? "true" : "false"}
            data-disabled={filesLauncher?.disabled ? "true" : "false"}
            data-pending-count={String(filesLauncher?.pendingCount ?? 0)}
            onClick={onToggleFilesPanel}
          >
            files
          </button>
        ) : null}
        {onToggleBrowserPanel ? (
          <button
            type="button"
            data-testid="task-center-browser-toggle-stub"
            data-expanded={browserLauncher?.active ? "true" : "false"}
            data-disabled={browserLauncher?.disabled ? "true" : "false"}
            data-pending-count={String(browserLauncher?.pendingCount ?? 0)}
            onClick={onToggleBrowserPanel}
          >
            browser
          </button>
        ) : null}
        {showHarnessToggle ? (
          <button
            type="button"
            data-testid="task-center-harness-toggle-stub"
            data-expanded={
              harnessLauncher?.active || harnessPanelVisible ? "true" : "false"
            }
            data-disabled={harnessLauncher?.disabled ? "true" : "false"}
            data-pending-count={String(harnessLauncher?.pendingCount ?? 0)}
            onClick={onToggleHarnessPanel}
          >
            harness
          </button>
        ) : null}
        {showExpertInfoToggle ? (
          <button
            type="button"
            data-testid="task-center-expert-info-toggle-stub"
            data-expanded={expertInfoPanelVisible ? "true" : "false"}
            onClick={onToggleExpertInfoPanel}
          >
            expert
          </button>
        ) : null}
      </>
    );
  },
}));

vi.mock("../components/TaskCenterShellPanel", () => ({
  TASK_CENTER_SHELL_PANEL_DEFAULT_HEIGHT_PX: 236,
  TASK_CENTER_SHELL_PANEL_MAX_HEIGHT_RATIO: 0.82,
  TaskCenterShellPanel: ({
    heightPx,
    maximized,
    projectRootPath,
    onHeightChange,
    onToggleMaximize,
  }: {
    heightPx: number;
    maximized: boolean;
    projectRootPath?: string | null;
    onHeightChange?: (heightPx: number) => void;
    onToggleMaximize?: () => void;
  }) => (
    <div
      data-testid="task-center-shell-panel-stub"
      data-height={heightPx}
      data-maximized={maximized ? "true" : "false"}
    >
      <span>{projectRootPath}</span>
      <button
        type="button"
        data-testid="task-center-shell-height-stub"
        onClick={() => onHeightChange?.(360)}
      >
        height
      </button>
      <button
        type="button"
        data-testid="task-center-shell-maximize-stub"
        onClick={onToggleMaximize}
      >
        maximize
      </button>
    </div>
  ),
}));

vi.mock("../components/EmptyState", () => ({
  EmptyState: () => <div data-testid="empty-state-stub" />,
}));

vi.mock("../components/MessageList", () => ({
  MessageList: (props: {
    leadingContent?: React.ReactNode;
    trailingContent?: React.ReactNode;
  }) => {
    mockMessageList(props);
    const { leadingContent, trailingContent } = props;
    return (
      <div data-testid="message-list-stub">
        {leadingContent}
        {trailingContent}
      </div>
    );
  },
}));

vi.mock("./WorkspacePendingA2UIPanel", () => ({
  WorkspacePendingA2UIPanel: (props: {
    pendingA2UIForm?: { id?: string } | null;
    placement?: string;
  }) => {
    mockWorkspacePendingA2UIPanel(props);
    return (
      <div
        data-testid="workspace-pending-a2ui-panel"
        data-placement={props.placement || "dock"}
      >
        {props.pendingA2UIForm?.id || ""}
      </div>
    );
  },
}));

vi.mock("./WorkspaceMainArea", () => ({
  WorkspaceMainArea: ({
    navbarNode,
    threadHeaderNode,
    taskCenterUtilityToolbarNode,
    taskCenterTabsNode,
    taskCenterShellPanelNode,
    chatContent,
    canvasContent,
    rightSurfaceContent,
    ...rest
  }: {
    navbarNode?: React.ReactNode;
    threadHeaderNode?: React.ReactNode;
    taskCenterUtilityToolbarNode?: React.ReactNode;
    taskCenterTabsNode?: React.ReactNode;
    taskCenterShellPanelNode?: React.ReactNode;
    chatContent?: React.ReactNode;
    canvasContent?: React.ReactNode;
    rightSurfaceContent?: React.ReactNode;
    [key: string]: unknown;
  }) => (
    <div
      data-testid="workspace-main-area-stub"
      ref={() => {
        mockWorkspaceMainArea({
          navbarNode,
          threadHeaderNode,
          taskCenterUtilityToolbarNode,
          taskCenterTabsNode,
          taskCenterShellPanelNode,
          chatContent,
          canvasContent,
          rightSurfaceContent,
          ...rest,
        });
      }}
    >
      {navbarNode}
      {threadHeaderNode}
      {taskCenterUtilityToolbarNode}
      {taskCenterTabsNode}
      {chatContent}
      {canvasContent}
      {rightSurfaceContent}
      {taskCenterShellPanelNode}
    </div>
  ),
}));

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

function importedThreadItems() {
  return [
    {
      id: "imported-command",
      type: "command_execution",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 1,
      status: "completed",
      command: "npm test",
      cwd: "/workspace/imported-local-history",
      metadata: {
        source_client: "external-history",
        source_provenance: {
          sourceClient: "external-history",
          sourceThreadId: "thread-imported-20260617abcdef",
        },
        importFidelity: {
          messages: 6,
          reasoning: 2,
          commands: 1,
          tools: 4,
          patches: 1,
          approvals: 1,
          webSearch: 1,
        },
      },
      started_at: "2026-06-17T10:00:00.000Z",
      completed_at: "2026-06-17T10:00:01.000Z",
      updated_at: "2026-06-17T10:00:01.000Z",
    },
  ];
}

type RenderSceneProps = Partial<
  React.ComponentProps<typeof WorkspaceConversationScene>
> &
  Record<string, unknown>;

function renderScene(props?: RenderSceneProps) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const propsRecord = (props || {}) as Record<string, unknown>;
  const defaultLandingSurface = {
    entryBannerVisible: Boolean(propsRecord.entryBannerVisible),
    entryBannerMessage: propsRecord.entryBannerMessage as string | undefined,
    onDismissEntryBanner:
      (propsRecord.onDismissEntryBanner as (() => void) | undefined) ?? vi.fn(),
    creationReplaySurface:
      propsRecord.creationReplaySurface === undefined
        ? null
        : (propsRecord.creationReplaySurface as never),
    sceneAppExecutionSummaryCard:
      propsRecord.sceneAppExecutionSummaryCard as React.ReactNode,
    serviceSkillExecutionCard:
      propsRecord.serviceSkillExecutionCard as React.ReactNode,
    emptyStateProps: {} as never,
  };

  const defaultProps: React.ComponentProps<typeof WorkspaceConversationScene> =
    {
      landingSurface: defaultLandingSurface,
      showChatLayout: true,
      compactChrome: false,
      contextWorkspaceEnabled: false,
      messageListProps: {
        messages: [],
        turns: [],
        threadItems: [],
        currentTurnId: null,
        threadRead: false,
        pendingActions: [],
        submittedActionsInFlight: [],
        isSending: false,
        onInterruptCurrentTurn: vi.fn(),
      } as any,
      workspaceAlertVisible: false,
      onSelectWorkspaceDirectory: vi.fn(),
      onDismissWorkspaceAlert: vi.fn(),
      shouldHideGeneralWorkbenchInputForTheme: false,
      inputbarNode: null,
      input: "",
      setInput: vi.fn(),
      onSendMessage: vi.fn(),
      emptyStateIsLoading: false,
      emptyStateDisabled: false,
      providerType: "openai",
      setProviderType: vi.fn(),
      model: "gpt-4.1",
      setModel: vi.fn(),
      executionStrategy: "react",
      setExecutionStrategy: vi.fn(),
      accessMode: "default" as any,
      setAccessMode: vi.fn(),
      onManageProviders: vi.fn(),
      toolPreferences: {
        task: false,
        subagent: false,
      },
      onToolPreferenceChange: vi.fn(),
      creationMode: "guided",
      onCreationModeChange: vi.fn(),
      activeTheme: "general",
      onThemeChange: vi.fn(),
      themeLocked: false,
      artifactsCount: 0,
      generalCanvasContent: "",
      resolvedCanvasState: null,
      selectedText: "",
      onRecommendationClick: vi.fn(),
      characters: [],
      skills: [],
      serviceSkills: [],
      serviceSkillGroups: [],
      isSkillsLoading: false,
      onSelectServiceSkill: vi.fn(),
      onNavigateToSettings: vi.fn(),
      onRefreshSkills: vi.fn(),
      onLaunchBrowserAssist: vi.fn(),
      browserAssistLoading: false,
      projectId: null,
      onProjectChange: vi.fn(),
      onOpenSettings: vi.fn(),
      runtimeToolAvailability: null,
      runtimeTaskCard: null,
      onOpenMemoryWorkbench: vi.fn(),
      onOpenChannels: vi.fn(),
      onOpenChromeRelay: vi.fn(),
      navbarVisible: false,
      isRunning: false,
      navbarChrome: "default" as any,
      onBackToProjectManagement: undefined,
      onBackToResources: undefined,
      layoutMode: "chat" as any,
      onToggleCanvas: vi.fn(),
      onBackHome: vi.fn(),
      showHarnessToggle: false,
      harnessPanelVisible: false,
      onToggleHarnessPanel: vi.fn(),
      harnessPendingCount: 0,
      harnessAttentionLevel: "idle" as any,
      harnessToggleLabel: undefined,
      showContextCompactionAction: false,
      contextCompactionRunning: false,
      onCompactContext: vi.fn(),
      isThemeWorkbench: false,
      contentId: undefined,
      syncStatus: "idle",
      hasLiveCanvasPreviewContent: false,
      liveCanvasPreview: null,
      currentImageWorkbenchActive: false,
      shouldShowCanvasLoadingState: false,
      canvasWorkbenchLayoutProps: {
        artifacts: [],
      } as any,
      shellBottomInset: "0px",
      chatPanelWidth: undefined,
      chatPanelMinWidth: undefined,
      generalWorkbenchDialog: null,
      showFloatingInputOverlay: false,
      hasPendingA2UIForm: false,
    } as any;

  const sceneProps = {
    ...defaultProps,
    ...props,
    landingSurface: {
      ...defaultLandingSurface,
      ...((props?.landingSurface as Record<string, unknown> | undefined) ?? {}),
    },
  };

  act(() => {
    root.render(<WorkspaceConversationScene {...sceneProps} />);
  });

  mountedRoots.push({ root, container });
  return container;
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
  vi.clearAllMocks();
});

describe("WorkspaceConversationScene", () => {
  it("入口提示条应把长文案和关闭按钮分开渲染", () => {
    const onDismissEntryBanner = vi.fn();
    const container = renderScene({
      entryBannerVisible: true,
      entryBannerMessage: "欢迎回来：继续处理上次的内容并保留当前上下文。",
      onDismissEntryBanner,
    });

    const banner = container.querySelector(
      '[data-testid="workspace-entry-banner"]',
    );
    const bannerText = container.querySelector(
      '[data-testid="workspace-entry-banner-text"]',
    );
    const closeButton = banner?.querySelector<HTMLButtonElement>("button");

    expect(banner).not.toBeNull();
    expect(bannerText?.textContent).toContain("欢迎回来");
    expect(closeButton).not.toBeNull();
    expect(
      [
        "关闭",
        "Close",
        "agentChat.workspaceConversation.entryBanner.close",
      ].includes(closeButton?.textContent || ""),
    ).toBe(true);

    act(() => {
      closeButton?.click();
    });

    expect(onDismissEntryBanner).toHaveBeenCalledTimes(1);
  });

  it("普通导入会话不应在主线消息列表前展示独立导入状态条", () => {
    const container = renderScene({
      messageListProps: {
        messages: [
          {
            id: "imported-user",
            role: "user",
            content: "请帮我修复运行时问题",
            timestamp: new Date("2026-06-17T10:00:00.000Z"),
          },
          {
            id: "imported-assistant",
            role: "assistant",
            content: "已完成修复并补充测试。",
            timestamp: new Date("2026-06-17T10:00:01.000Z"),
          },
        ],
        turns: [],
        threadItems: importedThreadItems(),
        currentTurnId: null,
        threadRead: false,
        pendingActions: [],
        submittedActionsInFlight: [],
        isSending: false,
        onInterruptCurrentTurn: vi.fn(),
      } as any,
    });

    const banner = container.querySelector(
      '[data-testid="imported-source-banner"]',
    );
    expect(
      mockMessageList.mock.calls.some((call) =>
        Boolean(call[0]?.leadingContent),
      ),
    ).toBe(false);
    expect(banner).toBeNull();
    expect(container.textContent).not.toContain("本地历史导入");
    expect(container.textContent).not.toContain("已还原");
    expect(container.textContent).not.toContain("imported_read_only");
    expect(container.textContent).not.toContain("thread-imported");
    expect(container.textContent).not.toContain("npm test");
  });

  it("生成应显示当前带入的灵感横条", () => {
    const container = renderScene({
      creationReplaySurface: {
        kind: "memory_entry",
        eyebrow: "当前带入灵感",
        badgeLabel: "参考",
        title: "品牌风格样本",
        summary: "保留轻盈但专业的表达。",
        hint: "后续结果模板会默认把它一起带入。",
        defaultReferenceMemoryIds: ["memory-1"],
        defaultReferenceEntries: [
          {
            id: "memory-1",
            title: "品牌风格样本",
            summary: "保留轻盈但专业的表达。",
            category: "context",
            categoryLabel: "参考",
            tags: ["品牌", "语气"],
          },
        ],
      },
    });

    expect(container.textContent).toContain("当前带入灵感");
    expect(container.textContent).toContain("品牌风格样本");
    expect(container.textContent).toContain("后续结果模板会默认把它一起带入。");
  });

  it("任务中心场景应固定展示顶部导航，不再传入自动隐藏开关", () => {
    renderScene({
      navbarVisible: true,
      navbarChrome: "workspace-compact",
      navbarContextVariant: "task-center",
      taskCenterTabsNode: <div data-testid="task-center-tabs-stub">tabs</div>,
    });

    expect(mockWorkspaceMainArea).toHaveBeenCalled();
    expect(
      mockWorkspaceMainArea.mock.calls.at(-1)?.[0]?.autoHideTaskCenterNavbar,
    ).toBeUndefined();
  });

  it("任务中心场景应使用统一工具栏承接 Shell 和工作台入口", () => {
    const container = renderScene({
      navbarVisible: true,
      navbarChrome: "workspace-compact",
      navbarContextVariant: "task-center",
      taskCenterTabsNode: <div data-testid="task-center-tabs-stub">tabs</div>,
      projectRootPath: "/tmp/project",
      showHarnessToggle: true,
    });

    const toolbar = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-utility-toolbar-stub"]',
    );
    expect(toolbar).not.toBeNull();
    expect(
      mockWorkspaceMainArea.mock.calls.at(-1)?.[0]
        ?.taskCenterUtilityToolbarNode,
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="canvas-layout-top-right-tools"]'),
    ).toBeNull();

    act(() => {
      toolbar?.click();
    });

    expect(
      container.querySelector('[data-testid="task-center-shell-panel-stub"]')
        ?.textContent,
    ).toContain("/tmp/project");
    expect(mockWorkspaceMainArea.mock.calls.at(-1)?.[0]?.shellBottomInset).toBe(
      "calc(0px + 236px)",
    );
  });

  it("active Thread 应使用紧凑页头承接标题、状态、目录和统一工具栏", () => {
    const container = renderScene({
      navbarVisible: true,
      navbarChrome: "workspace-compact",
      navbarContextVariant: "task-center",
      taskCenterTabsNode: <div data-testid="task-center-tabs-stub">tabs</div>,
      threadHeader: {
        sessionId: "thread-1",
        title: "对齐 Codex App GUI",
        status: "running",
        workingDirectory: "/workspace/lime",
      },
    });

    expect(
      container.querySelector('[data-testid="thread-workspace-header"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="thread-workspace-header-title"]')
        ?.textContent,
    ).toBe("对齐 Codex App GUI");
    expect(
      container.querySelector(
        '[data-testid="thread-workspace-header-actions"] [data-testid="task-center-utility-toolbar-stub"]',
      ),
    ).not.toBeNull();
    expect(
      mockWorkspaceMainArea.mock.calls.at(-1)?.[0]
        ?.taskCenterUtilityToolbarNode,
    ).toBeNull();
  });

  it("任务中心场景应把专家信息按钮状态透传给统一工具栏", () => {
    const onToggleExpertInfoPanel = vi.fn();
    const container = renderScene({
      navbarVisible: true,
      navbarChrome: "workspace-compact",
      navbarContextVariant: "task-center",
      taskCenterTabsNode: <div data-testid="task-center-tabs-stub">tabs</div>,
      showExpertInfoToggle: true,
      expertInfoPanelVisible: false,
      onToggleExpertInfoPanel,
    });

    expect(mockTaskCenterUtilityToolbar).toHaveBeenCalledWith(
      expect.objectContaining({
        showExpertInfoToggle: true,
        expertInfoPanelVisible: false,
        onToggleExpertInfoPanel,
      }),
    );

    const expertToggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-expert-info-toggle-stub"]',
    );
    expect(expertToggle?.getAttribute("data-expanded")).toBe("false");

    act(() => {
      expertToggle?.click();
    });

    expect(onToggleExpertInfoPanel).toHaveBeenCalledTimes(1);
  });

  it("任务中心场景应把 Right Surface launcher 投影透传给统一工具栏", () => {
    const onToggleHarnessPanel = vi.fn();
    const rightSurfaceLaunchers: WorkspaceRightSurfaceLauncherProjection[] = [
      {
        kind: "harness",
        active: true,
        disabled: false,
        pendingCount: 2,
        collapseTarget: "topToolbar",
      },
    ];
    const container = renderScene({
      navbarVisible: true,
      navbarChrome: "workspace-compact",
      navbarContextVariant: "task-center",
      taskCenterTabsNode: <div data-testid="task-center-tabs-stub">tabs</div>,
      showHarnessToggle: true,
      harnessPanelVisible: false,
      onToggleHarnessPanel,
      rightSurfaceLaunchers,
    });

    expect(mockTaskCenterUtilityToolbar).toHaveBeenCalledWith(
      expect.objectContaining({
        showHarnessToggle: true,
        harnessPanelVisible: false,
        onToggleHarnessPanel,
        rightSurfaceLaunchers,
      }),
    );

    const harnessToggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-harness-toggle-stub"]',
    );
    expect(harnessToggle?.getAttribute("data-expanded")).toBe("true");
    expect(harnessToggle?.getAttribute("data-disabled")).toBe("false");
    expect(harnessToggle?.getAttribute("data-pending-count")).toBe("2");

    act(() => {
      harnessToggle?.click();
    });

    expect(onToggleHarnessPanel).toHaveBeenCalledTimes(1);
  });

  it("任务中心场景不应再把 Trace 渲染成顶部独立按钮", () => {
    const rightSurfaceLaunchers: WorkspaceRightSurfaceLauncherProjection[] = [
      {
        kind: "trace",
        active: true,
        disabled: false,
        pendingCount: 1,
        collapseTarget: "topToolbar",
      },
    ];
    const container = renderScene({
      navbarVisible: true,
      navbarChrome: "workspace-compact",
      navbarContextVariant: "task-center",
      taskCenterTabsNode: <div data-testid="task-center-tabs-stub">tabs</div>,
      rightSurfaceLaunchers,
    });

    expect(mockTaskCenterUtilityToolbar).toHaveBeenCalledWith(
      expect.objectContaining({
        rightSurfaceLaunchers,
      }),
    );

    const traceToggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-trace-toggle-stub"]',
    );
    expect(traceToggle).toBeNull();
  });

  it("任务中心场景应把文件 surface 入口透传给统一工具栏", () => {
    const onToggleRightSurfaceFiles = vi.fn();
    const rightSurfaceLaunchers: WorkspaceRightSurfaceLauncherProjection[] = [
      {
        kind: "files",
        active: true,
        disabled: false,
        pendingCount: 1,
        collapseTarget: "topToolbar",
      },
    ];
    const container = renderScene({
      navbarVisible: true,
      navbarChrome: "workspace-compact",
      navbarContextVariant: "task-center",
      taskCenterTabsNode: <div data-testid="task-center-tabs-stub">tabs</div>,
      rightSurfaceFilesOpen: true,
      onToggleRightSurfaceFiles,
      rightSurfaceLaunchers,
    });

    expect(mockTaskCenterUtilityToolbar).toHaveBeenCalledWith(
      expect.objectContaining({
        onToggleFilesPanel: onToggleRightSurfaceFiles,
        rightSurfaceLaunchers,
      }),
    );

    const filesToggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-files-toggle-stub"]',
    );
    expect(filesToggle?.getAttribute("data-expanded")).toBe("true");
    expect(filesToggle?.getAttribute("data-disabled")).toBe("false");
    expect(filesToggle?.getAttribute("data-pending-count")).toBe("1");

    act(() => {
      filesToggle?.click();
    });

    expect(onToggleRightSurfaceFiles).toHaveBeenCalledTimes(1);
  });

  it("任务中心场景应把对象画布 surface 入口透传给统一工具栏", () => {
    const onToggleRightSurfaceObjectCanvas = vi.fn();
    const rightSurfaceLaunchers: WorkspaceRightSurfaceLauncherProjection[] = [
      {
        kind: "objectCanvas",
        active: true,
        disabled: false,
        pendingCount: 1,
        collapseTarget: "topToolbar",
      },
    ];
    const container = renderScene({
      navbarVisible: true,
      navbarChrome: "workspace-compact",
      navbarContextVariant: "task-center",
      taskCenterTabsNode: <div data-testid="task-center-tabs-stub">tabs</div>,
      rightSurfaceObjectCanvasOpen: true,
      onToggleRightSurfaceObjectCanvas,
      rightSurfaceLaunchers,
    });

    expect(mockTaskCenterUtilityToolbar).toHaveBeenCalledWith(
      expect.objectContaining({
        onToggleObjectCanvasPanel: onToggleRightSurfaceObjectCanvas,
        rightSurfaceLaunchers,
      }),
    );

    const objectCanvasToggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-object-canvas-toggle-stub"]',
    );
    expect(objectCanvasToggle?.getAttribute("data-expanded")).toBe("true");
    expect(objectCanvasToggle?.getAttribute("data-disabled")).toBe("false");
    expect(objectCanvasToggle?.getAttribute("data-pending-count")).toBe("1");

    act(() => {
      objectCanvasToggle?.click();
    });

    expect(onToggleRightSurfaceObjectCanvas).toHaveBeenCalledTimes(1);
  });

  it("任务中心场景应把浏览器 surface 入口透传给统一工具栏", () => {
    const onToggleRightSurfaceBrowser = vi.fn();
    const rightSurfaceLaunchers: WorkspaceRightSurfaceLauncherProjection[] = [
      {
        kind: "browser",
        active: true,
        disabled: false,
        pendingCount: 1,
        collapseTarget: "topToolbar",
      },
    ];
    const container = renderScene({
      navbarVisible: true,
      navbarChrome: "workspace-compact",
      navbarContextVariant: "task-center",
      taskCenterTabsNode: <div data-testid="task-center-tabs-stub">tabs</div>,
      rightSurfaceBrowserOpen: true,
      onToggleRightSurfaceBrowser,
      rightSurfaceLaunchers,
    });

    expect(mockTaskCenterUtilityToolbar).toHaveBeenCalledWith(
      expect.objectContaining({
        onToggleBrowserPanel: onToggleRightSurfaceBrowser,
        rightSurfaceLaunchers,
      }),
    );

    const browserToggle = container.querySelector<HTMLButtonElement>(
      '[data-testid="task-center-browser-toggle-stub"]',
    );
    expect(browserToggle?.getAttribute("data-expanded")).toBe("true");
    expect(browserToggle?.getAttribute("data-disabled")).toBe("false");
    expect(browserToggle?.getAttribute("data-pending-count")).toBe("1");

    act(() => {
      browserToggle?.click();
    });

    expect(onToggleRightSurfaceBrowser).toHaveBeenCalledTimes(1);
  });

  it("右侧 Surface 内容应透传给主区域并由内层承载区渲染", () => {
    const container = renderScene({
      rightSurfaceContent: (
        <div data-testid="workspace-right-surface-probe">expert</div>
      ),
    });

    expect(
      mockWorkspaceMainArea.mock.calls.at(-1)?.[0]?.rightSurfaceContent,
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="workspace-right-surface-probe"]'),
    ).not.toBeNull();
  });

  it("任务中心 Shell 调整高度后应同步避让工作台内容", () => {
    const container = renderScene({
      navbarVisible: true,
      navbarChrome: "workspace-compact",
      navbarContextVariant: "task-center",
      taskCenterTabsNode: <div data-testid="task-center-tabs-stub">tabs</div>,
      projectRootPath: "/tmp/project",
      shellBottomInset: "12px",
      showHarnessToggle: true,
    });

    act(() => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="task-center-utility-toolbar-stub"]',
        )
        ?.click();
    });

    expect(mockWorkspaceMainArea.mock.calls.at(-1)?.[0]?.shellBottomInset).toBe(
      "calc(12px + 236px)",
    );

    act(() => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="task-center-shell-height-stub"]',
        )
        ?.click();
    });

    const panel = container.querySelector(
      '[data-testid="task-center-shell-panel-stub"]',
    );
    expect(panel?.getAttribute("data-height")).toBe("360");
    expect(panel?.getAttribute("data-maximized")).toBe("false");
    expect(mockWorkspaceMainArea.mock.calls.at(-1)?.[0]?.shellBottomInset).toBe(
      "calc(12px + 360px)",
    );
  });

  it("有消息来源的 pending A2UI 不应再渲染底部补参面板", () => {
    const container = renderScene({
      pendingA2UIForm: {
        id: "assistant-inline-a2ui",
        root: "root",
        components: [],
      },
      onPendingA2UISubmit: vi.fn(),
      messageListProps: {
        messages: [],
        activePendingA2UISource: {
          kind: "assistant_message",
          messageId: "msg-assistant-a2ui",
        },
      } as any,
    });

    expect(
      container.querySelector('[data-testid="workspace-pending-a2ui-panel"]'),
    ).toBeNull();
    expect(mockWorkspacePendingA2UIPanel).not.toHaveBeenCalled();
  });

  it("无消息来源的 pending A2UI 应作为消息列表尾部卡片渲染", () => {
    const container = renderScene({
      pendingA2UIForm: {
        id: "service-skill-a2ui",
        root: "root",
        components: [],
      },
      onPendingA2UISubmit: vi.fn(),
      messageListProps: {
        messages: [],
        activePendingA2UISource: {
          kind: "service_skill",
          skillId: "daily-trend-briefing",
          requestKey: "req-1",
          messageId: undefined,
        },
      } as any,
    });

    const pendingPanel = container.querySelector(
      '[data-testid="workspace-pending-a2ui-panel"]',
    );
    expect(pendingPanel?.getAttribute("data-placement")).toBe("message");
    expect(pendingPanel?.textContent).toContain("service-skill-a2ui");
    expect(mockWorkspacePendingA2UIPanel).toHaveBeenCalledWith(
      expect.objectContaining({
        placement: "message",
        pendingA2UIForm: expect.objectContaining({
          id: "service-skill-a2ui",
        }),
      }),
    );
  });

  it("inline 输入区应留在聊天 flex 栈内，避免矮窗口被父级 overflow 裁切", () => {
    const container = renderScene({
      contextWorkspaceEnabled: false,
      shouldHideGeneralWorkbenchInputForTheme: false,
      showFloatingInputOverlay: false,
      inputbarNode: (
        <textarea
          data-testid="workspace-inline-inputbar"
          defaultValue="整理今天的国际新闻"
        />
      ),
    });

    const messageList = container.querySelector(
      '[data-testid="message-list-stub"]',
    );
    const inputSlot = container.querySelector(
      '[data-testid="workspace-inline-input-slot"]',
    );
    const inputbar = container.querySelector(
      '[data-testid="workspace-inline-inputbar"]',
    );

    expect(messageList).not.toBeNull();
    expect(inputSlot).not.toBeNull();
    expect(inputbar).not.toBeNull();
    expect(inputSlot?.parentElement?.contains(messageList)).toBe(true);
    expect(inputSlot?.parentElement?.lastElementChild).toBe(inputSlot);
    expect(
      mockWorkspaceMainArea.mock.calls.at(-1)?.[0]?.showFloatingInputOverlay,
    ).toBe(false);
  });

  it("浮动输入区应在消息滚动内容尾部预留响应式避让空间", () => {
    const container = renderScene({
      contextWorkspaceEnabled: true,
      showFloatingInputOverlay: true,
    });

    const spacer = container.querySelector(
      '[data-testid="message-list-floating-input-spacer"]',
    );

    expect(spacer).not.toBeNull();
    expect(spacer?.className).toContain("h-8");
    expect(spacer?.className).toContain("max-[960px]:h-16");
    expect(
      container
        .querySelector('[data-testid="message-list-stub"]')
        ?.contains(spacer),
    ).toBe(true);
  });
});
