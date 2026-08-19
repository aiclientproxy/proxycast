import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  clickButton,
  createMockAgentChatUnifiedState,
  createMockThemeContextWorkspaceState,
  createProject,
  flushEffects,
  GENERAL_CONTEXT_HINT,
  getIndexTestMocks,
  getSendMessageCall,
  installMockAgentChatUnifiedState,
  textContainsAny,
  type MockInputbarSendPayload,
  type MockInputbarSendProps,
  mountPage,
  renderPage,
  sharedSendMessageMock,
  waitForElement,
  WORKSPACE_HARNESS_DESCRIPTION_CANDIDATES,
  WORKSPACE_HARNESS_TITLE_CANDIDATES,
} from "./index.testFixtures";

const {
  mockCanvasWorkbenchLayout,
  mockCanvasWorkbenchLayoutState,
  mockEmptyState,
  mockGetAgentRuntimeToolInventory,
  mockGetProject,
  mockInputbar,
  mockIsSpecializedWorkbenchTheme,
  mockMessageList,
  mockRequestWorkspaceRightSurface,
  mockUseAgentChatUnified,
  mockUseDeveloperFeatureFlags,
  mockUseThemeContextWorkspace,
} = getIndexTestMocks();

describe("AgentChatPage 通用工作台", { timeout: 20_000 }, () => {
  it("空白新建任务首页应去掉项目栏和会话标签，只保留右上工具区", async () => {
    const container = renderPage({
      agentEntry: "new-task",
      showChatPanel: false,
      theme: "general",
      projectId: "project-home",
    });
    await flushEffects(10);

    expect(container.querySelector('[data-testid="chat-navbar"]')).toBeNull();
    expect(
      container.querySelector('[data-testid="task-center-chrome-shell"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="task-center-tab-strip"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="task-center-tab-new-task-home"]'),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="task-center-tab-close-new-task-home"]',
      ),
    ).toBeNull();
    expect(container.textContent).not.toContain("新对话");
    expect(
      container.querySelector(
        '[data-testid="task-center-home-top-toolbar-host"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="task-center-utility-toolbar"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="toggle-harness"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="toggle-settings"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="empty-state"]'),
    ).not.toBeNull();
  });

  it("空白新建任务首页不应再展示顶部加号和草稿标签", async () => {
    const mounted = mountPage({
      agentEntry: "new-task",
      showChatPanel: false,
      theme: "general",
      projectId: "project-home",
    });
    await flushEffects(10);

    expect(
      mounted.container.querySelector(
        '[data-testid="task-center-tab-new-task-home"]',
      ),
    ).toBeNull();
    expect(
      mounted.container.querySelector(
        '[data-testid="task-center-tab-create-button"]',
      ),
    ).toBeNull();
    const draftTabs = mounted.container.querySelectorAll(
      '[data-testid^="task-center-tab-task-draft-"]',
    );
    expect(draftTabs).toHaveLength(0);
    expect(
      mounted.container.querySelector(
        '[data-testid="task-center-home-top-toolbar-host"]',
      ),
    ).not.toBeNull();
  });

  it("空白新建任务首页刷新后不应自动恢复最近会话", async () => {
    renderPage({
      agentEntry: "new-task",
      newChatAt: 1234567890,
      projectId: "project-home",
      showChatPanel: false,
      theme: "general",
    });
    await flushEffects();

    const workspaceCall = mockUseAgentChatUnified.mock.calls
      .map(
        (call) =>
          call[0] as {
            disableSessionRestore?: boolean;
            workspaceId?: string;
          },
      )
      .find((options) => options.workspaceId === "project-home");

    expect(workspaceCall?.disableSessionRestore).toBe(true);
  });

  it("空白新建任务发送后应立即展示轻量对话预览", async () => {
    type MockEmptyStateProps = {
      input?: string;
      setInput?: (value: string) => void;
      onSend?: (payload?: MockInputbarSendPayload) => void;
    };
    mockEmptyState.mockImplementation((props?: MockEmptyStateProps) => (
      <div data-testid="empty-state" data-input={props?.input || ""}>
        <button
          type="button"
          data-testid="mock-empty-type"
          onClick={() => props?.setInput?.("你好")}
        >
          输入
        </button>
        <button
          type="button"
          data-testid="mock-empty-send"
          onClick={() =>
            props?.onSend?.({ textOverride: props.input || "你好" })
          }
        >
          发送
        </button>
      </div>
    ));
    sharedSendMessageMock.mockImplementationOnce(
      () => new Promise<void>(() => undefined),
    );
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        sessionId: null,
        topics: [],
      }),
    );

    const mounted = mountPage({
      agentEntry: "new-task",
      showChatPanel: false,
      theme: "general",
      projectId: "project-home",
    });
    await flushEffects();

    expect(
      await waitForElement(
        mounted.container,
        '[data-testid="mock-empty-type"]',
      ),
    ).not.toBeNull();
    clickButton(mounted.container, "mock-empty-type");
    await flushEffects(1);
    clickButton(mounted.container, "mock-empty-send");

    const latestMessageListProps = mockMessageList.mock.calls.at(-1)?.[0] as
      | { messages?: Array<{ content?: string; role?: string }> }
      | undefined;
    expect(
      mounted.container.querySelector('[data-testid="message-list"]'),
    ).not.toBeNull();
    expect(latestMessageListProps?.messages?.[0]).toMatchObject({
      content: "你好",
      role: "user",
    });
    expect(latestMessageListProps?.messages?.[1]).toMatchObject({
      role: "assistant",
    });
  });

  it("空白新建任务发送派发完成但真实消息未接管前应继续保留轻量预览", async () => {
    type MockEmptyStateProps = {
      input?: string;
      setInput?: (value: string) => void;
      onSend?: (payload?: MockInputbarSendPayload) => void;
    };
    mockEmptyState.mockImplementation((props?: MockEmptyStateProps) => (
      <div data-testid="empty-state" data-input={props?.input || ""}>
        <button
          type="button"
          data-testid="mock-empty-type"
          onClick={() => props?.setInput?.("你好")}
        >
          输入
        </button>
        <button
          type="button"
          data-testid="mock-empty-send"
          onClick={() =>
            props?.onSend?.({ textOverride: props.input || "你好" })
          }
        >
          发送
        </button>
      </div>
    ));
    sharedSendMessageMock.mockResolvedValueOnce(undefined);
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        sessionId: null,
        topics: [],
      }),
    );

    const mounted = mountPage({
      agentEntry: "new-task",
      showChatPanel: false,
      theme: "general",
      projectId: "project-home",
    });
    await flushEffects();

    expect(
      await waitForElement(
        mounted.container,
        '[data-testid="mock-empty-type"]',
      ),
    ).not.toBeNull();
    clickButton(mounted.container, "mock-empty-type");
    await flushEffects(1);
    clickButton(mounted.container, "mock-empty-send");
    await flushEffects(12);
    mounted.rerender();
    await flushEffects(4);

    expect(
      mounted.container.querySelector('[data-testid="empty-state"]'),
    ).toBeNull();
    expect(
      mounted.container.querySelector('[data-testid="message-list"]'),
    ).not.toBeNull();
    const latestMessageListProps = mockMessageList.mock.calls.at(-1)?.[0] as
      | { messages?: Array<{ content?: string; role?: string }> }
      | undefined;
    expect(latestMessageListProps?.messages?.[0]).toMatchObject({
      content: "你好",
      role: "user",
    });
  });

  it("新建任务首页即使记住了文档项目也应保留真实首页", async () => {
    mockIsSpecializedWorkbenchTheme.mockImplementation(
      (theme?: string) => theme !== "general",
    );
    mockUseThemeContextWorkspace.mockImplementation(
      ({ activeTheme }: { activeTheme?: string }) =>
        createMockThemeContextWorkspaceState({
          enabled: activeTheme !== "general",
        }),
    );
    mockGetProject.mockResolvedValue({
      ...createProject("project-document-home"),
      workspaceType: "general",
    });

    const container = renderPage({
      agentEntry: "new-task",
      projectId: "project-document-home",
      theme: "general",
    });
    await flushEffects(10);

    expect(
      container.querySelector('[data-testid="empty-state"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="canvas-workbench-layout-mock"]'),
    ).toBeNull();
  });

  it("聊天态应通过顶栏按钮展开画布，并支持在展开后再次折叠", async () => {
    const container = renderPage({
      theme: "general",
      lockTheme: true,
    });
    await flushEffects();

    const toolbar = container.querySelector(
      '[data-testid="task-center-utility-toolbar"]',
    ) as HTMLDivElement | null;
    const toggleCanvasButton = container.querySelector(
      '[data-testid="toggle-canvas"]',
    ) as HTMLButtonElement | null;

    expect(toolbar?.dataset.showCanvasToggle).toBe("true");
    expect(toolbar?.dataset.canvasOpen).toBe("false");
    expect(toggleCanvasButton?.textContent).toContain("展开画布");
    expect(
      container
        .querySelector('[data-testid="layout-transition"]')
        ?.getAttribute("data-mode"),
    ).toBe("chat");

    act(() => {
      toggleCanvasButton?.click();
    });
    await flushEffects();

    expect(
      (
        container.querySelector(
          '[data-testid="task-center-utility-toolbar"]',
        ) as HTMLDivElement | null
      )?.dataset.canvasOpen,
    ).toBe("true");
    expect(
      container
        .querySelector('[data-testid="layout-transition"]')
        ?.getAttribute("data-mode"),
    ).toBe("chat-canvas");

    act(() => {
      (
        container.querySelector(
          '[data-testid="toggle-canvas"]',
        ) as HTMLButtonElement | null
      )?.click();
    });
    await flushEffects();

    expect(
      (
        container.querySelector(
          '[data-testid="task-center-utility-toolbar"]',
        ) as HTMLDivElement | null
      )?.dataset.canvasOpen,
    ).toBe("false");
    expect(
      container
        .querySelector('[data-testid="layout-transition"]')
        ?.getAttribute("data-mode"),
    ).toBe("chat");
  });

  it("普通画布应将工作台触发按钮并入头部工具栏，避免覆盖关闭区", async () => {
    mockCanvasWorkbenchLayoutState.renderPreviewProbe = true;

    const container = renderPage({
      theme: "general",
      lockTheme: true,
    });
    await flushEffects();

    clickButton(container, "toggle-canvas");
    await flushEffects(4);

    const workbench = container.querySelector(
      '[data-testid="canvas-workbench-layout-mock"]',
    ) as HTMLDivElement | null;

    expect(workbench).not.toBeNull();
    expect(
      workbench?.querySelector(
        '[data-testid="canvas-workbench-default-preview-probe"]',
      ),
    ).not.toBeNull();
  });

  it("点击 WebSearch 来源应请求右侧 Browser，而不是打开旧 Canvas 预览", async () => {
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        messages: [
          {
            id: "msg-search-assistant",
            role: "assistant",
            content: "已整理搜索来源。",
            timestamp: new Date("2026-06-24T10:00:00.000Z"),
          },
        ],
      }),
    );

    const mounted = mountPage({
      theme: "general",
      lockTheme: true,
    });
    await flushEffects();

    const latestMessageListProps = mockMessageList.mock.calls.at(-1)?.[0] as
      | {
          onOpenUrlPreview?: (item: {
            id: string;
            title: string;
            url: string;
            hostname: string;
            snippet?: string;
            snapshotContent?: string;
          }) => void;
        }
      | undefined;
    expect(latestMessageListProps?.onOpenUrlPreview).toEqual(
      expect.any(Function),
    );

    act(() => {
      latestMessageListProps?.onOpenUrlPreview?.({
        id: "source-1",
        title: "Reuters World News",
        url: " https://www.reuters.com/world/ ",
        hostname: "www.reuters.com",
        snippet: "搜索结果摘要",
        snapshotContent: "# Reuters snapshot\n\n正文来自 WebFetch。",
      });
    });
    await flushEffects(4);

    expect(
      mounted.container
        .querySelector('[data-testid="layout-transition"]')
        ?.getAttribute("data-mode"),
    ).toBe("chat");

    expect(mockRequestWorkspaceRightSurface).toHaveBeenCalledWith({
      surfaceKind: "browser",
      origin: "user",
      priority: "foreground",
      reason: "open_url_preview",
      sessionId: null,
      workspaceId: null,
      candidateId: "source-1",
      metadata: {
        browser: {
          launchUrl: "https://www.reuters.com/world/",
          title: "Reuters World News",
        },
      },
    });
  });

  it("通用模式空闲时应保留顶部 Harness 入口", async () => {
    const container = renderPage({
      theme: "general",
      lockTheme: true,
    });
    await flushEffects();

    const toolbar = container.querySelector(
      '[data-testid="task-center-utility-toolbar"]',
    ) as HTMLDivElement | null;
    expect(toolbar?.dataset.showHarnessToggle).toBe("true");
    expect(toolbar?.dataset.harnessToggleLabel).toBe("Harness");
    expect(
      textContainsAny(
        document.body.textContent,
        WORKSPACE_HARNESS_TITLE_CANDIDATES,
      ),
    ).toBe(false);
    expect(document.body.textContent).not.toContain(GENERAL_CONTEXT_HINT);

    clickButton(container, "toggle-harness");
    await flushEffects();

    expect(
      textContainsAny(
        document.body.textContent,
        WORKSPACE_HARNESS_TITLE_CANDIDATES,
      ),
    ).toBe(true);
    expect(document.body.textContent).toContain(GENERAL_CONTEXT_HINT);
  });

  it("通用模式有处理活动时应通过顶部 Harness 按钮打开弹窗，而不是常驻右侧占位", async () => {
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        isSending: true,
      }),
    );

    const container = renderPage({
      theme: "general",
      lockTheme: true,
    });
    await flushEffects();

    const toolbar = container.querySelector(
      '[data-testid="task-center-utility-toolbar"]',
    ) as HTMLDivElement | null;
    expect(toolbar?.dataset.showHarnessToggle).toBe("true");
    expect(toolbar?.dataset.harnessToggleLabel).toBe("Harness");
    expect(
      textContainsAny(
        document.body.textContent,
        WORKSPACE_HARNESS_TITLE_CANDIDATES,
      ),
    ).toBe(false);
    expect(document.body.textContent).not.toContain(GENERAL_CONTEXT_HINT);

    clickButton(container, "toggle-harness");
    await flushEffects();

    expect(
      textContainsAny(
        document.body.textContent,
        WORKSPACE_HARNESS_TITLE_CANDIDATES,
      ),
    ).toBe(true);
    expect(document.body.textContent).toContain(GENERAL_CONTEXT_HINT);
    expect(
      textContainsAny(
        document.body.textContent,
        WORKSPACE_HARNESS_DESCRIPTION_CANDIDATES,
      ),
    ).toBe(true);
  });

  it("处理工作台调试信息开关开启时右侧 Harness surface 应触发工具库存读取", async () => {
    const container = renderPage({
      theme: "general",
      lockTheme: true,
    });
    await flushEffects();

    const toolbar = container.querySelector(
      '[data-testid="task-center-utility-toolbar"]',
    ) as HTMLDivElement | null;
    expect(toolbar?.dataset.showHarnessToggle).toBe("true");
    expect(toolbar?.dataset.harnessPanelVisible).toBe("false");
    expect(
      container.querySelector('[data-testid="task-center-harness-toggle"]'),
    ).not.toBeNull();
    expect(mockGetAgentRuntimeToolInventory).not.toHaveBeenCalled();

    clickButton(container, "task-center-harness-toggle");
    await flushEffects();

    expect(toolbar?.dataset.harnessPanelVisible).toBe("true");
    expect(mockGetAgentRuntimeToolInventory).toHaveBeenCalledWith(
      expect.objectContaining({
        caller: "assistant",
        workbench: false,
      }),
    );
  });

  it("处理工作台调试信息开关关闭时仍应保留入口，但不触发工具库存读取", async () => {
    mockUseDeveloperFeatureFlags.mockReturnValue({
      clawTraceEnabled: false,
      workspaceHarnessEnabled: false,
    });

    const container = renderPage({
      theme: "general",
      lockTheme: true,
    });
    await flushEffects();

    const toolbar = container.querySelector(
      '[data-testid="task-center-utility-toolbar"]',
    ) as HTMLDivElement | null;
    expect(toolbar?.dataset.showHarnessToggle).toBe("true");
    expect(toolbar?.dataset.harnessToggleLabel).toBe("Harness");
    expect(
      textContainsAny(
        document.body.textContent,
        WORKSPACE_HARNESS_TITLE_CANDIDATES,
      ),
    ).toBe(false);
    expect(mockGetAgentRuntimeToolInventory).not.toHaveBeenCalled();

    clickButton(container, "toggle-harness");
    await flushEffects();

    expect(
      textContainsAny(
        document.body.textContent,
        WORKSPACE_HARNESS_TITLE_CANDIDATES,
      ),
    ).toBe(true);
    expect(mockGetAgentRuntimeToolInventory).not.toHaveBeenCalled();
  });

  it("窄屏工作台切换 stacked/split 时不应恢复旧对话侧栏入口", async () => {
    mockCanvasWorkbenchLayoutState.renderPreviewProbe = true;

    const container = renderPage({
      theme: "general",
      lockTheme: true,
    });
    await flushEffects();

    expect(container.querySelector('[data-testid="chat-sidebar"]')).toBeNull();
    expect(
      container.querySelector('[data-testid="toggle-history"]'),
    ).toBeNull();

    clickButton(container, "toggle-canvas");
    await flushEffects(4);

    const getWorkbenchProps = () =>
      (mockCanvasWorkbenchLayout.mock.calls.at(-1)?.[0] || null) as {
        onLayoutModeChange?: (mode: "split" | "stacked") => void;
      } | null;

    expect(
      container.querySelector('[data-testid="canvas-workbench-layout-mock"]'),
    ).not.toBeNull();

    act(() => {
      getWorkbenchProps()?.onLayoutModeChange?.("stacked");
    });
    await flushEffects(4);

    expect(container.querySelector('[data-testid="chat-sidebar"]')).toBeNull();
    expect(
      container.querySelector('[data-testid="toggle-history"]'),
    ).toBeNull();

    act(() => {
      getWorkbenchProps()?.onLayoutModeChange?.("split");
    });
    await flushEffects(4);

    expect(container.querySelector('[data-testid="chat-sidebar"]')).toBeNull();
  });

  it("发送时不再先调用本地 Team 规划模型，而是直接发送并透传已选 Team 约束", async () => {
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        executionStrategy: "react",
      }),
    );
    localStorage.setItem(
      "lime.chat.team_selection.v1.general",
      JSON.stringify({
        id: "code-triage-team",
        source: "builtin",
      }),
    );

    const mounted = mountPage({
      projectId: "project-team-runtime",
      theme: "general",
      lockTheme: true,
    });
    await flushEffects(10);

    let latestInputbarProps = mockInputbar.mock.calls.at(-1)?.[0] as
      | MockInputbarSendProps
      | undefined;

    act(() => {
      latestInputbarProps?.onToolStatesChange?.({
        subagent: true,
      });
    });
    await flushEffects(8);

    latestInputbarProps = mockInputbar.mock.calls.at(-1)?.[0] as
      | MockInputbarSendProps
      | undefined;

    act(() => {
      void latestInputbarProps?.onSend?.({
        images: [],
        textOverride: "请帮我拆解并推进这个修复任务",
      });
    });
    await flushEffects(8);

    expect(sharedSendMessageMock).toHaveBeenCalledTimes(1);
    const sendCall = getSendMessageCall();
    expect(sendCall.content).toBe("请帮我拆解并推进这个修复任务");
    expect(sendCall.images).toEqual([]);
    expect(sendCall.webSearch).toBeUndefined();
    expect(sendCall.thinking).toBeUndefined();
    expect(sendCall.skipUserMessage).toBe(false);
    expect(sendCall.executionStrategy).toBe("react");
    expect(sendCall.modelOverride).toBeUndefined();
    expect(sendCall.autoContinue).toBeUndefined();
    const sendOptions = sendCall.options;
    expect(sendOptions?.requestMetadata?.harness).not.toHaveProperty(
      "selected_team_id",
    );
    expect(sendOptions?.requestMetadata?.harness).not.toHaveProperty(
      "selected_team_roles",
    );
    expect(
      sendOptions?.requestMetadata?.harness?.turn_team_decision,
    ).toBeUndefined();
    expect(
      sendOptions?.requestMetadata?.harness?.turn_team_blueprint,
    ).toBeUndefined();
    expect(sendOptions?.assistantDraft).toBeUndefined();

    expect(
      mounted.container
        .querySelector('[data-testid="layout-transition"]')
        ?.getAttribute("data-mode"),
    ).toBe("chat");
    const formedRuntimeDock = mounted.container.querySelector(
      '[data-testid="team-workspace-dock"]',
    );
    if (formedRuntimeDock) {
      expect(formedRuntimeDock.getAttribute("data-runtime-status")).not.toBe(
        "forming",
      );
    }
  });
});
