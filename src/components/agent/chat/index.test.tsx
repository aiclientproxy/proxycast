import { describe, expect, it, vi } from "vitest";
import * as harnessStateModule from "./utils/harnessState";
import * as crashDiagnosticModule from "@/lib/crashDiagnosticAgentUiPerformance";
import * as traceHistoryModule from "@/lib/agentUiPerformanceTraceHistory";
import * as agentTaskRuntimeModule from "./utils/agentTaskRuntime";
import {
  clickButton,
  createMockAgentChatUnifiedState,
  FIXED_TOPIC_UPDATED_AT,
  flushEffects,
  getIndexTestMocks,
  getSendMessageCall,
  installMockAgentChatUnifiedState,
  type MockEmptyStateProps,
  mountPage,
  renderPage,
  sharedSendMessageMock,
} from "./index.testFixtures";
import { buildHomeAgentParams } from "@/lib/workspace/navigation";
import { requestTaskCenterDraftTask } from "./taskCenterDraftTaskEvents";

const {
  mockEmptyState,
  mockExpertInfoPanel,
  mockExecutionRunListGeneralWorkbenchHistory,
  mockGetProjectMemory,
  mockMessageList,
  mockSkillExecutionGetDetail,
  mockSkillsGetLocal,
  mockUseAgentChatUnified,
  mockUseSessionFiles,
  mockUseTrayModelShortcuts,
} = getIndexTestMocks();

vi.mock("./utils/harnessState", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./utils/harnessState")>();
  return {
    ...actual,
    deriveHarnessSessionState: vi.fn(actual.deriveHarnessSessionState),
  };
});

vi.mock("./utils/agentTaskRuntime", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("./utils/agentTaskRuntime")>();
  return {
    ...actual,
    buildAgentTaskRuntimeCardModel: vi.fn(
      actual.buildAgentTaskRuntimeCardModel,
    ),
  };
});

describe("AgentChatPage 任务中心顶部工具区", () => {
  it("任务中心不应再渲染顶部项目栏和会话标签", async () => {
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        sessionId: "topic-current",
        topics: [
          {
            id: "topic-current",
            title: "当前会话",
            updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
            workspaceId: "workspace-test",
          },
        ],
      }),
    );

    const container = renderPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    expect(container.querySelector('[data-testid="chat-navbar"]')).toBeNull();
    expect(
      container.querySelector('[data-testid="task-center-chrome-shell"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="task-center-tab-strip"]'),
    ).toBeNull();
    expect(mockUseAgentChatUnified).toHaveBeenCalledWith(
      expect.objectContaining({ workspaceId: "workspace-test" }),
    );
  });

  it("默认 Claw 对话不应提前触发完整 Harness 派生", async () => {
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        sessionId: "topic-current",
        topics: [
          {
            id: "topic-current",
            title: "当前会话",
            updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
            workspaceId: "workspace-test",
          },
        ],
      }),
    );

    renderPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    expect(harnessStateModule.deriveHarnessSessionState).not.toHaveBeenCalled();
  });

  it("默认 Claw 对话不应提前读取 Trace 历史和诊断摘要", async () => {
    const historySpy = vi.spyOn(
      traceHistoryModule,
      "listAgentUiPerformanceTraceHistory",
    );
    const overviewSpy = vi.spyOn(
      traceHistoryModule,
      "getAgentUiPerformanceTraceHistoryOverview",
    );
    const summarySpy = vi.spyOn(
      crashDiagnosticModule,
      "buildAgentUiPerformanceDiagnosticSummary",
    );

    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        sessionId: "topic-current",
        topics: [
          {
            id: "topic-current",
            title: "当前会话",
            updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
            workspaceId: "workspace-test",
          },
        ],
      }),
    );

    renderPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    expect(historySpy).not.toHaveBeenCalled();
    expect(overviewSpy).not.toHaveBeenCalled();
    expect(summarySpy).not.toHaveBeenCalled();
  });

  it("默认 Claw 对话不应首帧读取已安装 Plugin 插件列表", async () => {
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        sessionId: "topic-current",
        topics: [
          {
            id: "topic-current",
            title: "当前会话",
            updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
            workspaceId: "workspace-test",
          },
        ],
      }),
    );

    renderPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

  });

  it("默认 Claw 对话不应首帧读取本地 Skills 候选", async () => {
    mockSkillsGetLocal.mockClear();
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        sessionId: "topic-current",
        topics: [
          {
            id: "topic-current",
            title: "当前会话",
            updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
            workspaceId: "workspace-test",
          },
        ],
      }),
    );

    renderPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    expect(mockSkillsGetLocal).not.toHaveBeenCalled();
  });

  it("默认 Claw 对话不应预取通用工作台历史和 Skill 详情", async () => {
    mockExecutionRunListGeneralWorkbenchHistory.mockClear();
    mockSkillExecutionGetDetail.mockClear();
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        sessionId: "topic-current",
        topics: [
          {
            id: "topic-current",
            title: "当前会话",
            updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
            workspaceId: "workspace-test",
          },
        ],
      }),
    );

    renderPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    expect(mockExecutionRunListGeneralWorkbenchHistory).not.toHaveBeenCalled();
    expect(mockSkillExecutionGetDetail).not.toHaveBeenCalled();
  });

  it("默认 Claw 对话不应首帧同步托盘模型候选", async () => {
    mockUseTrayModelShortcuts.mockClear();
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        sessionId: "topic-current",
        topics: [
          {
            id: "topic-current",
            title: "当前会话",
            updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
            workspaceId: "workspace-test",
          },
        ],
      }),
    );

    renderPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    expect(mockUseTrayModelShortcuts).toHaveBeenCalled();
    for (const [options] of mockUseTrayModelShortcuts.mock.calls) {
      expect(options).toEqual(
        expect.objectContaining({
          autoSyncEnabled: false,
          deferInitialSync: true,
        }),
      );
    }
  });

  it("默认 Claw 对话不应提前渲染专家信息面板", async () => {
    mockExpertInfoPanel.mockClear();
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        sessionId: "topic-current",
        topics: [
          {
            id: "topic-current",
            title: "当前会话",
            updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
            workspaceId: "workspace-test",
          },
        ],
      }),
    );

    renderPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    expect(mockExpertInfoPanel).not.toHaveBeenCalled();
  });

  it("默认 Claw 对话有消息时不应构建空态 runtime task card", async () => {
    vi.mocked(
      agentTaskRuntimeModule.buildAgentTaskRuntimeCardModel,
    ).mockClear();
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        isSending: false,
        sessionId: "topic-current",
        topics: [
          {
            id: "topic-current",
            title: "当前会话",
            updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
            workspaceId: "workspace-test",
          },
        ],
      }),
    );

    renderPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    expect(
      agentTaskRuntimeModule.buildAgentTaskRuntimeCardModel,
    ).not.toHaveBeenCalled();
  });

  it("侧边栏新建任务事件应在任务中心内嵌首页起手页，不恢复顶部标签", async () => {
    const onNavigate = vi.fn();
    vi.mocked(buildHomeAgentParams).mockClear();
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: "topic-current",
      topics: [
        {
          id: "topic-current",
          title: "当前会话",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          workspaceId: "workspace-test",
        },
      ],
    });
    const createFreshSession = vi.fn(async () => {
      return "new-topic";
    });
    const clearMessages = vi.fn();
    state.createFreshSession = createFreshSession;
    state.clearMessages = clearMessages;
    installMockAgentChatUnifiedState(state);

    const mounted = mountPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
      onNavigate,
    });
    const { container } = mounted;
    await flushEffects();

    expect(
      requestTaskCenterDraftTask({
        source: "sidebar",
        projectId: "workspace-test",
      }),
    ).toBe(true);
    await flushEffects();
    mounted.rerender();
    await flushEffects();

    expect(createFreshSession).not.toHaveBeenCalled();
    expect(clearMessages).toHaveBeenCalledWith({ showToast: false });
    expect(buildHomeAgentParams).not.toHaveBeenCalled();
    expect(onNavigate).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-testid="task-center-chrome-shell"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="task-center-tab-strip"]'),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid^="task-center-tab-task-draft-"][data-active="true"]',
      ),
    ).toBeNull();
    expect(container.querySelector('[data-testid="chat-navbar"]')).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="task-center-home-top-toolbar-host"]',
      ),
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
    expect(container.querySelector('[data-testid="message-list"]')).toBeNull();
  });

  it("侧边栏新建任务事件应保留在当前页面且不恢复顶部草稿标签", async () => {
    const onNavigate = vi.fn();
    vi.mocked(buildHomeAgentParams).mockClear();
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: "topic-current",
      topics: [
        {
          id: "topic-current",
          title: "当前会话",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          workspaceId: "workspace-test",
          messagesCount: 1,
        },
      ],
      messages: [
        {
          id: "msg-existing-user",
          role: "user",
          content: "上一轮历史消息",
          timestamp: new Date(FIXED_TOPIC_UPDATED_AT),
        },
      ],
    });
    const createFreshSession = vi.fn(async () => "new-topic");
    const clearMessages = vi.fn(() => {
      state.messages = [];
    });
    const switchTopic = vi.fn(async () => "success");
    state.createFreshSession = createFreshSession;
    state.clearMessages = clearMessages;
    state.switchTopic = switchTopic;
    installMockAgentChatUnifiedState(state);

    const mounted = mountPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
      onNavigate,
    });
    const { container } = mounted;
    await flushEffects();

    expect(switchTopic).not.toHaveBeenCalled();

    expect(
      requestTaskCenterDraftTask({
        source: "sidebar",
        projectId: "workspace-test",
      }),
    ).toBe(true);
    await flushEffects();
    mounted.rerender();
    await flushEffects();

    expect(createFreshSession).not.toHaveBeenCalled();
    expect(clearMessages).toHaveBeenCalledWith({ showToast: false });
    expect(switchTopic).not.toHaveBeenCalled();
    expect(buildHomeAgentParams).not.toHaveBeenCalled();
    expect(onNavigate).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-testid="task-center-tab-strip"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid^="task-center-tab-task-draft-"]'),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="task-center-home-top-toolbar-host"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="empty-state"]'),
    ).not.toBeNull();
    expect(container.querySelector('[data-testid="message-list"]')).toBeNull();
  });

  it("历史会话 route 收到无项目新建任务事件时应沿用当前项目打开任务首页", async () => {
    const onNavigate = vi.fn();
    vi.mocked(buildHomeAgentParams).mockClear();
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: "topic-project",
      topics: [
        {
          id: "topic-project",
          title: "项目会话",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          workspaceId: "workspace-test",
        },
      ],
    });
    state.clearMessages = vi.fn();
    installMockAgentChatUnifiedState(state);

    const mounted = mountPage({
      agentEntry: "claw",
      initialSessionId: "topic-project",
      projectId: "workspace-test",
      onNavigate,
    });
    const { container } = mounted;
    await flushEffects();

    expect(
      requestTaskCenterDraftTask({ source: "sidebar", projectId: null }),
    ).toBe(true);
    await flushEffects();
    mounted.rerender();
    await flushEffects();

    expect(buildHomeAgentParams).not.toHaveBeenCalled();
    expect(onNavigate).not.toHaveBeenCalled();
    expect(state.clearMessages).toHaveBeenCalledWith({ showToast: false });
    expect(
      container.querySelector(
        '[data-testid="task-center-home-top-toolbar-host"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="empty-state"]'),
    ).not.toBeNull();
    expect(container.querySelector('[data-testid="message-list"]')).toBeNull();
  });

  it("项目会话收到顶部新建任务事件时应保留当前项目上下文", async () => {
    const onNavigate = vi.fn();
    vi.mocked(buildHomeAgentParams).mockClear();
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: "topic-project",
      topics: [
        {
          id: "topic-project",
          title: "项目会话",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          workspaceId: "workspace-test",
        },
      ],
    });
    state.clearMessages = vi.fn();
    installMockAgentChatUnifiedState(state);

    mountPage({
      agentEntry: "claw",
      initialSessionId: "topic-project",
      projectId: "workspace-test",
      onNavigate,
    });
    await flushEffects();

    expect(
      requestTaskCenterDraftTask({
        source: "sidebar",
        projectId: "workspace-test",
      }),
    ).toBe(true);
    await flushEffects();

    expect(buildHomeAgentParams).not.toHaveBeenCalled();
    expect(onNavigate).not.toHaveBeenCalled();
    expect(state.clearMessages).toHaveBeenCalledWith({ showToast: false });
  });

  it("打开旧会话后草稿预热创建新对话时不应继承旧消息快照", async () => {
    mockEmptyState.mockImplementation((props?: MockEmptyStateProps) => (
      <div
        data-testid="empty-state"
        data-active-theme={props?.activeTheme || ""}
        data-input={props?.input || ""}
        data-session-id={props?.sessionId ?? ""}
      >
        {props?.activeTheme === "general" ? (
          <div data-testid="home-start-surface" />
        ) : null}
        <button
          type="button"
          data-testid="mock-empty-type"
          onClick={() => props?.setInput?.("新的独立对话")}
        >
          输入
        </button>
      </div>
    ));

    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: "topic-current",
      messages: [
        {
          id: "old-user",
          role: "user",
          content: "旧会话问题",
          timestamp: new Date(FIXED_TOPIC_UPDATED_AT),
        },
        {
          id: "old-assistant",
          role: "assistant",
          content: "旧会话回答",
          timestamp: new Date(FIXED_TOPIC_UPDATED_AT + 1_000),
        },
      ],
      topics: [
        {
          id: "topic-current",
          title: "旧会话",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          workspaceId: "workspace-test",
        },
      ],
    });
    const createFreshSession = vi.fn(async () => "new-topic");
    const clearMessages = vi.fn();
    state.createFreshSession = createFreshSession;
    state.clearMessages = clearMessages;
    installMockAgentChatUnifiedState(state);

    const mounted = mountPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    mockMessageList.mockClear();
    expect(requestTaskCenterDraftTask({ source: "sidebar" })).toBe(true);
    await flushEffects();
    mounted.rerender();
    await flushEffects();

    expect(mounted.container.textContent).not.toContain("旧会话问题");
    expect(mounted.container.textContent).not.toContain("旧会话回答");
    expect(clearMessages).toHaveBeenCalledWith({ showToast: false });
    expect(
      mounted.container.querySelector('[data-testid="empty-state"]'),
    ).not.toBeNull();
    expect(
      mounted.container.querySelector('[data-testid="home-start-surface"]'),
    ).not.toBeNull();
    const latestEmptyStateProps = mockEmptyState.mock.calls.at(-1)?.[0] as
      | MockEmptyStateProps
      | undefined;
    expect(latestEmptyStateProps?.activeTheme).toBe("general");
    expect(latestEmptyStateProps?.sessionId).toBeNull();

    clickButton(mounted.container, "mock-empty-type");
    await flushEffects(8);

    expect(createFreshSession).toHaveBeenCalledWith(
      "新对话",
      expect.objectContaining({
        preserveCurrentSnapshot: false,
      }),
    );
  });

  it("隐藏草稿首页输入后应预热创建会话，发送时复用同一次创建", async () => {
    const onNavigate = vi.fn();
    vi.mocked(buildHomeAgentParams).mockClear();
    mockEmptyState.mockImplementation((props?: MockEmptyStateProps) => (
      <div
        data-testid="empty-state"
        data-active-theme={props?.activeTheme || ""}
        data-input={props?.input || ""}
        data-session-id={props?.sessionId ?? ""}
      >
        {props?.activeTheme === "general" ? (
          <div data-testid="home-start-surface" />
        ) : null}
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
          onClick={() => props?.onSend?.({ textOverride: "你好" })}
        >
          发送
        </button>
      </div>
    ));

    const creationController: { resolve?: () => void } = {};
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: "topic-current",
      isAutoRestoringSession: true,
      topics: [
        {
          id: "topic-current",
          title: "当前会话",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          workspaceId: "workspace-test",
        },
      ],
    });
    const createFreshSession = vi.fn(
      () =>
        new Promise<string>((resolve) => {
          creationController.resolve = () => {
            state.sessionId = "new-topic";
            state.topics = [
              {
                id: "new-topic",
                title: "新对话",
                updatedAt: new Date(FIXED_TOPIC_UPDATED_AT + 1_000),
                workspaceId: "workspace-test",
              },
              ...((state.topics as Array<Record<string, unknown>>) || []),
            ];
            resolve("new-topic");
          };
        }),
    );
    state.createFreshSession = createFreshSession;
    sharedSendMessageMock.mockImplementation(async () => {
      state.currentTurnId = "pending-turn:test";
      state.isSending = true;
    });
    installMockAgentChatUnifiedState(state);

    const mounted = mountPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
      onNavigate,
    });
    await flushEffects();

    mockMessageList.mockClear();
    expect(requestTaskCenterDraftTask({ source: "sidebar" })).toBe(true);
    await flushEffects();
    mounted.rerender();
    await flushEffects();

    expect(
      mounted.container.querySelector('[data-testid="empty-state"]'),
    ).not.toBeNull();
    expect(
      mounted.container.querySelector('[data-testid="home-start-surface"]'),
    ).not.toBeNull();
    expect(
      mounted.container.querySelector('[data-testid="message-list"]'),
    ).toBeNull();
    const latestEmptyStateProps = mockEmptyState.mock.calls.at(-1)?.[0] as
      | MockEmptyStateProps
      | undefined;
    expect(latestEmptyStateProps?.activeTheme).toBe("general");
    expect(latestEmptyStateProps?.sessionId).toBeNull();
    const latestDraftMessageListProps = mockMessageList.mock.calls.at(
      -1,
    )?.[0] as
      | {
          emptyStateVariant?: string;
          messages?: unknown[];
          sessionId?: string | null;
        }
      | undefined;
    if (latestDraftMessageListProps) {
      expect(latestDraftMessageListProps.emptyStateVariant).toBe("none");
      expect(latestDraftMessageListProps.messages).toEqual([]);
      expect(latestDraftMessageListProps.sessionId).toBeNull();
    }

    clickButton(mounted.container, "mock-empty-type");
    await flushEffects(8);
    expect(createFreshSession).toHaveBeenCalledTimes(1);

    clickButton(mounted.container, "mock-empty-send");
    await flushEffects(2);
    expect(createFreshSession).toHaveBeenCalledTimes(1);
    expect(sharedSendMessageMock).not.toHaveBeenCalled();
    expect(
      mounted.container.querySelector('[data-testid="empty-state"]'),
    ).toBeNull();

    expect(creationController.resolve).toBeTruthy();
    creationController.resolve?.();
    await flushEffects(10);
    mounted.rerender();
    await flushEffects();

    expect(sharedSendMessageMock).toHaveBeenCalledTimes(1);
    expect(getSendMessageCall().content).toBe("你好");
    mounted.rerender();
    await flushEffects();
    expect(mockUseSessionFiles).toHaveBeenLastCalledWith(
      expect.objectContaining({
        sessionId: "new-topic",
        autoInit: false,
      }),
    );
    expect(
      (
        getSendMessageCall().options?.requestMetadata as
          | Record<string, Record<string, unknown>>
          | undefined
      )?.agentUiPerformanceTrace?.sessionId,
    ).toBe("new-topic");
    expect(createFreshSession).toHaveBeenCalledTimes(1);
    expect(mounted.container.textContent).not.toContain("正在恢复生成会话");
    expect(buildHomeAgentParams).not.toHaveBeenCalled();
    expect(onNavigate).not.toHaveBeenCalled();
  }, 60_000);

  it("从导航栏直达会话时应延后加载 topics，优先恢复目标会话详情", async () => {
    installMockAgentChatUnifiedState(
      createMockAgentChatUnifiedState({
        sessionId: "topic-selected",
        topics: [
          {
            id: "topic-selected",
            title: "目标对话",
            updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          },
        ],
      }),
    );

    renderPage({
      agentEntry: "claw",
      initialSessionId: "topic-selected",
      projectId: "workspace-test",
    });
    await flushEffects(1);

    const workspaceCall = mockUseAgentChatUnified.mock.calls
      .map(
        (call) =>
          call[0] as {
            workspaceId?: string;
            initialTopicsLoadMode?: string;
            initialTopicsDeferredDelayMs?: number;
            initialRuntimeWarmupLoadMode?: string;
            initialRuntimeWarmupDeferredDelayMs?: number;
          },
      )
      .find((options) => options.workspaceId === "workspace-test");
    expect(workspaceCall?.initialTopicsLoadMode).toBe("deferred");
    expect(workspaceCall?.initialTopicsDeferredDelayMs).toBe(0);
    expect(workspaceCall?.initialRuntimeWarmupLoadMode).toBe("deferred");
    expect(workspaceCall?.initialRuntimeWarmupDeferredDelayMs).toBe(45_000);
    expect(mockSkillsGetLocal).not.toHaveBeenCalled();
    expect(mockGetProjectMemory).not.toHaveBeenCalled();
  });
});
