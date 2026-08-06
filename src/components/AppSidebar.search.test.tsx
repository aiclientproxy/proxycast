import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  cleanupAppSidebarTest,
  flushEffects,
  mockGetProject,
  mockListAgentRuntimeSessions,
  mockRecordAgentUiPerformanceMetric,
  mountSidebarContainer,
  resetAppSidebarTest,
  setInputValue
} from "./AppSidebar.testFixtures";
import type { AgentPageParams } from "./AppSidebar.testFixtures";

describe("AppSidebar search", () => {
  beforeEach(async () => {
    await resetAppSidebarTest();
    mockGetProject.mockImplementation(async (projectId: string) => ({
      id: projectId,
      name: projectId,
      rootPath: `/repo/${projectId}`,
      isFavorite: false,
    }));
  });
  afterEach(cleanupAppSidebarTest);

  it("搜索按钮应打开标题搜索弹窗，并按会话标题过滤结果", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-agent",
        name: "写一篇AI Agent的公众号",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
        messages_count: 3,
      },
      {
        id: "session-daily",
        name: "啊啊啊啊啊",
        created_at: 1712900000,
        updated_at: 1712900600,
        archived_at: null,
        workspace_id: "project-1",
        messages_count: 1,
      },
    ]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "new-task",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    act(() => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-search-button"]',
        )
        ?.click();
    });
    await flushEffects(1);

    const dialog = document.body.querySelector<HTMLElement>(
      '[data-testid="app-sidebar-search-dialog"]',
    );
    expect(dialog).not.toBeNull();
    expect(dialog?.textContent).toContain("新建对话");
    expect(dialog?.textContent).toContain("写一篇AI Agent的公众号");
    expect(dialog?.textContent).toContain("啊啊啊啊啊");

    const input = document.body.querySelector<HTMLInputElement>(
      '[data-testid="app-sidebar-search-input"]',
    );
    expect(input).not.toBeNull();

    await act(async () => {
      setInputValue(input as HTMLInputElement, "Agent");
      await Promise.resolve();
    });

    expect(dialog?.textContent).toContain("匹配结果");
    expect(dialog?.textContent).toContain("写一篇AI Agent的公众号");
    expect(dialog?.textContent).not.toContain("啊啊啊啊啊");

    await act(async () => {
      setInputValue(input as HTMLInputElement, "不存在");
      await Promise.resolve();
    });

    expect(dialog?.textContent).toContain("没有匹配的对话标题");
  });

  it("搜索弹窗应支持查看更多对话并展示后续结果", async () => {
    const sessions = Array.from({ length: 12 }, (_, index) => {
      const number = String(index + 1).padStart(2, "0");
      return {
        id: `session-${number}`,
        name: `对话 ${number}`,
        created_at: 1713000000 + index,
        updated_at: 1713000600 + index,
        archived_at: null,
        workspace_id: "project-1",
        messages_count: 1,
      };
    }).reverse();
    mockListAgentRuntimeSessions.mockResolvedValue(sessions);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "new-task",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-search-button"]',
        )
        ?.click();
      await Promise.resolve();
    });

    let dialog = document.body.querySelector<HTMLElement>(
      '[data-testid="app-sidebar-search-dialog"]',
    );
    expect(dialog?.textContent).toContain("查看更多对话");
    expect(dialog?.textContent).toContain("对话 12");
    expect(dialog?.textContent).not.toContain("对话 02");

    await act(async () => {
      dialog
        ?.querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-search-more"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(2);

    dialog = document.body.querySelector<HTMLElement>(
      '[data-testid="app-sidebar-search-dialog"]',
    );
    expect(dialog?.textContent).toContain("对话 02");
    expect(dialog?.textContent).not.toContain("查看更多对话");
    expect(mockListAgentRuntimeSessions).toHaveBeenLastCalledWith({
      limit: 21,
      cwd: "/repo/project-1",
    });
  });

  it("搜索结果点击应复用会话导航并关闭弹窗", async () => {
    const onNavigate = vi.fn();
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-target",
        name: "目标历史会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
        messages_count: 3,
      },
    ]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        projectId: "project-1",
      } as AgentPageParams,
      onNavigate,
    });
    await flushEffects(2);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-search-button"]',
        )
        ?.click();
      await Promise.resolve();
    });

    const dialog = document.body.querySelector<HTMLElement>(
      '[data-testid="app-sidebar-search-dialog"]',
    );

    act(() => {
      dialog
        ?.querySelector<HTMLButtonElement>('button[title="目标历史会话"]')
        ?.click();
    });
    await flushEffects(1);

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-target",
        initialSessionName: "目标历史会话",
      }),
    );
    expect(mockRecordAgentUiPerformanceMetric).toHaveBeenCalledWith(
      "sidebar.conversation.click",
      expect.objectContaining({
        cwd: "/repo/project-1",
        projectId: "project-1",
        sessionId: "session-target",
        source: "sidebar_search",
      }),
    );
    expect(
      document.body.querySelector('[data-testid="app-sidebar-search-dialog"]'),
    ).toBeNull();
  });

  it("搜索结果从项目上下文打开无项目对话时不应继承项目 ID", async () => {
    const onNavigate = vi.fn();
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-standalone-search",
        name: "搜索无项目会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: null,
        messages_count: 2,
      },
    ]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-project-current",
      } as AgentPageParams,
      onNavigate,
    });
    await flushEffects(2);

    act(() => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-search-button"]',
        )
        ?.click();
    });
    await flushEffects(1);

    const dialog = document.body.querySelector<HTMLElement>(
      '[data-testid="app-sidebar-search-dialog"]',
    );

    act(() => {
      dialog
        ?.querySelector<HTMLButtonElement>('button[title="搜索无项目会话"]')
        ?.click();
    });
    await flushEffects(1);

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "claw",
        initialSessionId: "session-standalone-search",
      }),
    );
    expect((onNavigate.mock.calls[0]?.[1] as AgentPageParams).projectId).toBe(
      undefined,
    );
  });

  it("搜索结果悬停不应再触发旧会话预取，避免抢占点击切换", async () => {
    vi.useFakeTimers();
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-prefetch-search",
        name: "搜索预取历史会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: "project-1",
        messages_count: 3,
      },
    ]);

    try {
      const container = mountSidebarContainer({
        currentPage: "agent",
        currentPageParams: {
          agentEntry: "new-task",
          projectId: "project-1",
        } as AgentPageParams,
      });
      await flushEffects(2);

      await act(async () => {
        container
          .querySelector<HTMLButtonElement>(
            '[data-testid="app-sidebar-search-button"]',
          )
          ?.click();
        await Promise.resolve();
      });
      await flushEffects(5);

      const dialog = document.body.querySelector<HTMLElement>(
        '[data-testid="app-sidebar-search-dialog"]',
      );
      const resultButton = dialog?.querySelector<HTMLButtonElement>(
        'button[title="搜索预取历史会话"]',
      );
      expect(resultButton?.disabled).toBe(false);

      await act(async () => {
        resultButton?.dispatchEvent(
          new Event("pointerover", { bubbles: true }),
        );
        await Promise.resolve();
      });

      act(() => {
        vi.advanceTimersByTime(899);
      });

      act(() => {
        vi.advanceTimersByTime(1);
      });

      expect(mockRecordAgentUiPerformanceMetric).not.toHaveBeenCalledWith(
        "sidebar.conversation.prefetchFired",
        expect.anything(),
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it("搜索结果快速点击不应再触发预取计时器并应直接导航", async () => {
    vi.useFakeTimers();
    const onNavigate = vi.fn();
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-click-search",
        name: "搜索点击历史会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: "project-1",
        messages_count: 3,
      },
    ]);

    try {
      const container = mountSidebarContainer({
        currentPage: "agent",
        currentPageParams: {
          agentEntry: "new-task",
          projectId: "project-1",
        } as AgentPageParams,
        onNavigate,
      });
      await flushEffects(2);

      await act(async () => {
        container
          .querySelector<HTMLButtonElement>(
            '[data-testid="app-sidebar-search-button"]',
          )
          ?.click();
        await Promise.resolve();
      });

      const dialog = document.body.querySelector<HTMLElement>(
        '[data-testid="app-sidebar-search-dialog"]',
      );
      const resultButton = dialog?.querySelector<HTMLButtonElement>(
        'button[title="搜索点击历史会话"]',
      );

      await act(async () => {
        resultButton?.focus();
        resultButton?.click();
        await Promise.resolve();
      });

      act(() => {
        vi.advanceTimersByTime(900);
      });

      expect(mockRecordAgentUiPerformanceMetric).not.toHaveBeenCalledWith(
        "sidebar.conversation.prefetchFired",
        expect.anything(),
      );
      expect(onNavigate).toHaveBeenCalledWith(
        "agent",
        expect.objectContaining({
          agentEntry: "claw",
          projectId: "project-1",
          initialSessionId: "session-click-search",
        }),
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it("搜索弹窗的新建对话入口应复用现有新建导航", async () => {
    const onNavigate = vi.fn();
    mockListAgentRuntimeSessions.mockResolvedValue([]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        projectId: "project-1",
      } as AgentPageParams,
      onNavigate,
    });
    await flushEffects(2);

    act(() => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-search-button"]',
        )
        ?.click();
    });
    await flushEffects(1);

    const newConversationButton =
      document.body.querySelector<HTMLButtonElement>(
        '[data-testid="app-sidebar-search-new-conversation"]',
      );
    expect(newConversationButton).not.toBeNull();

    act(() => {
      newConversationButton?.click();
    });
    await flushEffects(1);

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "new-task",
      }),
    );
    expect((onNavigate.mock.calls[0]?.[1] as AgentPageParams).projectId).toBe(
      undefined,
    );
    expect(
      document.body.querySelector('[data-testid="app-sidebar-search-dialog"]'),
    ).toBeNull();
  });

  it("Meta/Ctrl + K 应打开搜索弹窗，Escape 应关闭弹窗", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([]);

    mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "new-task",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    expect(
      document.body.querySelector('[data-testid="app-sidebar-search-dialog"]'),
    ).toBeNull();

    await act(async () => {
      window.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "k",
          metaKey: true,
          bubbles: true,
          cancelable: true,
        }),
      );
      await Promise.resolve();
    });
    await flushEffects();

    expect(
      document.body.querySelector('[data-testid="app-sidebar-search-dialog"]'),
    ).not.toBeNull();

    await act(async () => {
      document.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Escape",
          bubbles: true,
          cancelable: true,
        }),
      );
      await Promise.resolve();
    });

    expect(
      document.body.querySelector('[data-testid="app-sidebar-search-dialog"]'),
    ).toBeNull();

    await act(async () => {
      window.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "k",
          ctrlKey: true,
          bubbles: true,
          cancelable: true,
        }),
      );
      await Promise.resolve();
    });

    expect(
      document.body.querySelector('[data-testid="app-sidebar-search-dialog"]'),
    ).not.toBeNull();
  });
});
