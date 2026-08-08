import React from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { SidebarOpenedProjectSummary } from "@/components/app-sidebar/sidebarConversationGroups";
import {
  AppSidebar,
  TASK_CENTER_CREATE_DRAFT_TASK_EVENT,
  act,
  buildMockConversationImportPreview,
  cleanupAppSidebarTest,
  clickConversationMenuItem,
  emitMockAgentUiPerformanceMetricRecorded,
  flushEffects,
  mockDeleteAgentRuntimeSession,
  mockDeleteProject,
  mockEnsureProjectWorkspace,
  mockCreateProjectGitWorktree,
  mockCommitConversationImportThread,
  mockGetProject,
  mockListAgentRuntimeSessions,
  mockPreviewConversationImportThread,
  mockRevealPathInFinder,
  mockRecordAgentUiPerformanceMetric,
  mockArchiveAgentRuntimeSession,
  mockScanConversationImportSource,
  mockScheduleMinimumDelayIdleTask,
  mockOpenDesktopDialog,
  mockToastSuccess,
  mockSetAgentRuntimeThreadName,
  mockListThreadSections,
  mockCreateThreadSection,
  mockUpdateThreadSection,
  mockDeleteThreadSection,
  mockMoveThreadToSection,
  mockUpdateProject,
  mockWaitForConversationImportJob,
  mountSidebar,
  mountSidebarContainer,
  openConversationMenu,
  openProjectMenu,
  resetAppSidebarTest,
} from "./AppSidebar.testFixtures";
import type { AgentPageParams } from "./AppSidebar.testFixtures";
import { useAppSidebarProjectActions } from "@/components/app-sidebar/useAppSidebarProjectActions";
import { AGENT_RUNTIME_SESSIONS_CHANGED_EVENT } from "@/lib/api/agentRuntime/sessionClient";

describe("AppSidebar conversations", () => {
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

  type ProjectActions = ReturnType<typeof useAppSidebarProjectActions>;

  async function mountProjectActionsHarness(options?: {
    currentProjectId?: string | null;
  }) {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root: Root = createRoot(container);
    const refreshSidebarSessions = vi.fn(async () => undefined);
    const onNavigate = vi.fn();
    const actionsRef: { current?: ProjectActions } = {};

    function Harness() {
      actionsRef.current = useAppSidebarProjectActions({
        currentProjectId: options?.currentProjectId ?? "project-1",
        onNavigate,
        refreshSidebarSessions,
      });
      return null;
    }

    await act(async () => {
      root.render(<Harness />);
      await Promise.resolve();
    });

    const mountedActions = actionsRef.current;
    if (!mountedActions) {
      throw new Error("Project actions harness did not mount");
    }

    return {
      actions: mountedActions,
      onNavigate,
      refreshSidebarSessions,
      cleanup: () => {
        act(() => {
          root.unmount();
        });
        container.remove();
      },
    };
  }

  async function mountProjectMenuScenario() {
    mockGetProject.mockResolvedValue({
      id: "project-1",
      name: "示例项目",
      rootPath: "/repo/example",
      isFavorite: false,
    });
    mockListAgentRuntimeSessions.mockImplementation(
      async (options?: { limit?: number; cwd?: string | string[] }) =>
        options?.cwd === "/repo/example"
          ? [
              {
                id: "session-project",
                name: "项目内会话",
                created_at: 1714000000,
                updated_at: 1714000600,
                archived_at: null,
                workspace_id: "project-1",
                working_dir: "/repo/example",
              },
            ]
          : [],
    );

    const onNavigate = vi.fn();
    mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
      onNavigate,
    });
    await flushEffects(3);

    return { onNavigate };
  }

  function captureTaskCenterDraftRequests() {
    const receivedDetails: unknown[] = [];
    const listener = (event: Event) => {
      receivedDetails.push(
        event instanceof CustomEvent ? event.detail : undefined,
      );
      event.preventDefault();
    };
    window.addEventListener(TASK_CENTER_CREATE_DRAFT_TASK_EVENT, listener);

    return {
      receivedDetails,
      dispose: () => {
        window.removeEventListener(
          TASK_CENTER_CREATE_DRAFT_TASK_EVENT,
          listener,
        );
      },
    };
  }

  it("任务中心内悬停已有会话不应再触发旧会话预取", async () => {
    vi.useFakeTimers();
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-prefetch",
        name: "可预取历史会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
        messages_count: 3,
      },
    ]);

    try {
      const container = mountSidebarContainer({
        currentPage: "agent",
        currentPageParams: {
          agentEntry: "claw",
          projectId: "project-1",
          initialSessionId: "session-current",
        } as AgentPageParams,
      });
      await flushEffects(2);

      await act(async () => {
        container
          .querySelector<HTMLButtonElement>('button[title="可预取历史会话"]')
          ?.focus();
        await Promise.resolve();
      });

      act(() => {
        vi.advanceTimersByTime(920);
      });

      expect(mockRecordAgentUiPerformanceMetric).not.toHaveBeenCalledWith(
        "sidebar.conversation.prefetchFired",
        expect.anything(),
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it("点击已有会话时不应再触发旧会话预取抢占切换链路", async () => {
    vi.useFakeTimers();
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-click",
        name: "立即打开历史会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
        messages_count: 3,
      },
    ]);

    try {
      const onNavigate = vi.fn();
      const container = mountSidebarContainer({
        currentPage: "agent",
        currentPageParams: {
          agentEntry: "claw",
          projectId: "project-1",
          initialSessionId: "session-current",
        } as AgentPageParams,
        onNavigate,
      });
      await flushEffects(2);

      const button = container.querySelector<HTMLButtonElement>(
        'button[title="立即打开历史会话"]',
      );

      await act(async () => {
        button?.focus();
        button?.click();
        await Promise.resolve();
      });

      act(() => {
        vi.advanceTimersByTime(200);
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
          initialSessionId: "session-click",
          initialSessionName: "立即打开历史会话",
        }),
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it("任务中心内点击项目会话应通过 current 页面导航打开历史会话", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-click",
        name: "立即打开历史会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
        messages_count: 3,
      },
    ]);

    const onNavigate = vi.fn();
    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-current",
      } as AgentPageParams,
      onNavigate,
    });
    await flushEffects(2);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>('button[title="立即打开历史会话"]')
        ?.click();
      await Promise.resolve();
    });

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-click",
      }),
    );
  });

  it("仅带 workspace_id 的项目会话也应显示在项目分组并通过 current 导航打开", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-workspace-only",
        name: "工作区会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: "project-1",
        messages_count: 3,
      },
    ]);

    const onNavigate = vi.fn();
    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-current",
      } as AgentPageParams,
      onNavigate,
    });
    await flushEffects(2);

    const projectSection = container.querySelector(
      '[data-testid="app-sidebar-project-conversations"]',
    );
    const standaloneSection = container.querySelector(
      '[data-testid="app-sidebar-recent-conversations"]',
    );
    expect(projectSection?.textContent).toContain("工作区会话");
    expect(standaloneSection?.textContent).not.toContain("工作区会话");
    const openButton = Array.from(
      container.querySelectorAll<HTMLButtonElement>(
        '[data-testid="app-sidebar-conversation-open"]',
      ),
    ).find((button) => button.getAttribute("title") === "工作区会话");
    expect(openButton).not.toBeNull();

    await act(async () => {
      openButton?.click();
      await Promise.resolve();
    });

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-workspace-only",
      }),
    );
  });

  it("当前会话切换到新 session 时应高亮真实 active 会话而不是入口会话", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-old",
        name: "旧会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
        messages_count: 3,
      },
      {
        id: "session-new",
        name: "新会话",
        created_at: 1713001000,
        updated_at: 1713001600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
        messages_count: 1,
      },
    ]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-old",
      } as AgentPageParams,
      activeAgentSessionId: "session-new",
    });
    await flushEffects(2);

    const activeButtons = Array.from(
      container.querySelectorAll<HTMLButtonElement>(
        'button[aria-current="page"]',
      ),
    ).map((button) => button.getAttribute("title"));

    expect(activeButtons).toContain("新会话");
    expect(activeButtons).not.toContain("旧会话");
  });

  it("导入本地历史对话应先预览，取消时不提交 commit", async () => {
    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(3);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-import-conversation-button"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(mockScanConversationImportSource).toHaveBeenCalledWith({
      sourceClient: "codex",
      sourceRoot: undefined,
      projectPath: undefined,
      includeArchived: true,
      limit: 40,
    });
    expect(mockPreviewConversationImportThread).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceClient: "codex",
        sourceThreadId: "codex-thread-1",
        limit: 12,
      }),
    );
    expect(document.body.textContent).toContain("导入对话");
    expect(document.body.textContent).toContain("请帮我修复运行时问题");
    expect(document.body.textContent).toContain("回合");
    expect(document.body.textContent).toContain("时间线");
    expect(document.body.textContent).toContain(
      "将保留工具、命令、补丁、确认与思考记录。",
    );
    expect(document.body.textContent).toContain("工具");
    expect(document.body.textContent).toContain("命令");
    expect(document.body.textContent).toContain("附件 1");
    expect(document.body.textContent).toContain("本地历史");
    expect(document.body.textContent).not.toMatch(/\bcodex\b/i);
    expect(document.body.textContent).not.toContain(".codex");
    expect(document.body.textContent).not.toContain("codex-thread-1");

    await act(async () => {
      Array.from(document.body.querySelectorAll<HTMLButtonElement>("button"))
        .find((button) => button.textContent?.trim() === "取消")
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(mockCommitConversationImportThread).not.toHaveBeenCalled();
    expect(
      document.body.querySelector(
        '[data-testid="app-sidebar-conversation-import-dialog"]',
      ),
    ).toBeNull();
  });

  it("项目范围导入本地历史对话确认后应带 confirmed=true 并打开导入会话", async () => {
    mockGetProject.mockResolvedValue({
      id: "project-1",
      name: "示例项目",
      rootPath: "/repo/project-1",
      isFavorite: false,
    });

    const onNavigate = vi.fn();
    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
      onNavigate,
    });
    await flushEffects(4);

    expect(
      container.querySelector(
        '[data-testid="app-sidebar-project-import-conversation"]',
      ),
    ).toBeNull();

    const projectMenu = await openProjectMenu("示例项目");
    expect(projectMenu?.textContent).toContain("导入对话");
    expect(
      projectMenu
        ?.querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-project-menu-import-conversation"]',
        )
        ?.getAttribute("aria-label"),
    ).toContain("示例项目");
    expect(
      projectMenu
        ?.querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-project-menu-import-conversation"]',
        )
        ?.getAttribute("title"),
    ).toContain("示例项目");

    await act(async () => {
      document.body
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-project-menu-import-conversation"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(4);

    await act(async () => {
      document.body
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-conversation-import-confirm"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(mockScanConversationImportSource).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceClient: "codex",
        projectPath: "/repo/project-1",
      }),
    );
    expect(mockCommitConversationImportThread).toHaveBeenCalledTimes(1);
    expect(mockCommitConversationImportThread).toHaveBeenCalledWith({
      sourceClient: "codex",
      sourceRoot: "/Users/example/.codex",
      sourceThreadId: "codex-thread-1",
      sourcePath: "/Users/example/.codex/sessions/codex-thread-1.jsonl",
      workspaceId: "project-1",
      confirmed: true,
    });
    expect(mockWaitForConversationImportJob).toHaveBeenCalledTimes(1);
    expect(mockWaitForConversationImportJob).toHaveBeenCalledWith(
      expect.objectContaining({
        status: "completed",
        result: expect.objectContaining({
          importedMessages: 2,
        }),
      }),
      expect.objectContaining({
        signal: expect.any(AbortSignal),
        onProgress: expect.any(Function),
      }),
    );
    expect(mockToastSuccess).toHaveBeenCalledWith("已导入 2 条历史消息");
    const projectSection = container.querySelector(
      '[data-testid="app-sidebar-project-conversations"]',
    );
    expect(projectSection?.textContent).toContain("本地历史修复记录");
    expect(projectSection?.textContent).not.toContain("本地历史第二条记录");
    expect(projectSection?.textContent).not.toMatch(/\bcodex\b/i);
    expect(projectSection?.textContent).not.toContain(".codex");
    expect(projectSection?.textContent).not.toContain("暂无聊天");
    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-imported",
      }),
    );
  });

  it("已导入本地历史确认后应清理并重新导入", async () => {
    mockScanConversationImportSource.mockResolvedValue({
      source: {
        sourceClient: "codex",
        status: "ready",
        sourceRoot: "/Users/example/.codex",
        readable: true,
        threadCount: 1,
        indexedAt: "2026-06-16T00:00:00.000Z",
        statePath: "/Users/example/.codex/state_5.sqlite",
      },
      threads: [
        {
          sourceClient: "codex",
          sourceThreadId: "codex-thread-1",
          title: "本地历史修复记录",
          createdAt: "2026-06-15T00:00:00.000Z",
          updatedAt: "2026-06-16T00:00:00.000Z",
          cwd: "/repo/project-1",
          source: "cli",
          modelProvider: "openai",
          archived: false,
          sourcePath: "/Users/example/.codex/sessions/codex-thread-1.jsonl",
          importStatus: "imported",
        },
      ],
    });
    const importedPreview = buildMockConversationImportPreview();
    mockPreviewConversationImportThread.mockResolvedValueOnce(
      buildMockConversationImportPreview({
        thread: {
          ...importedPreview.thread,
          importStatus: "imported",
        },
      }),
    );

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(3);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-import-conversation-button"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(document.body.textContent).toContain("已导入的会先清理后重新导入。");
    expect(document.body.textContent).toContain("重新导入");

    await act(async () => {
      document.body
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-conversation-import-confirm"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(mockCommitConversationImportThread).toHaveBeenCalledWith({
      sourceClient: "codex",
      sourceRoot: "/Users/example/.codex",
      sourceThreadId: "codex-thread-1",
      sourcePath: "/Users/example/.codex/sessions/codex-thread-1.jsonl",
      workspaceId: undefined,
      confirmed: true,
      replaceExisting: true,
    });
  });

  it("导入弹窗按组选择后应批量提交同组可导入对话", async () => {
    mockScanConversationImportSource.mockResolvedValue({
      source: {
        sourceClient: "codex",
        status: "ready",
        sourceRoot: "/Users/example/.codex",
        readable: true,
        threadCount: 3,
        indexedAt: "2026-06-16T00:00:00.000Z",
        statePath: "/Users/example/.codex/state_5.sqlite",
      },
      threads: [
        {
          sourceClient: "codex",
          sourceThreadId: "codex-thread-1",
          title: "本地历史修复记录 A",
          createdAt: "2026-06-15T00:00:00.000Z",
          updatedAt: "2026-06-16T08:00:00.000Z",
          cwd: "/repo/project-1",
          source: "cli",
          modelProvider: "openai",
          archived: false,
          sourcePath: "/Users/example/.codex/sessions/codex-thread-1.jsonl",
          importStatus: "not_imported",
        },
        {
          sourceClient: "codex",
          sourceThreadId: "codex-thread-2",
          title: "本地历史修复记录 B",
          createdAt: "2026-06-15T00:00:00.000Z",
          updatedAt: "2026-06-16T09:00:00.000Z",
          cwd: "/repo/project-1",
          source: "cli",
          modelProvider: "openai",
          archived: false,
          sourcePath: "/Users/example/.codex/sessions/codex-thread-2.jsonl",
          importStatus: "not_imported",
        },
        {
          sourceClient: "codex",
          sourceThreadId: "codex-thread-3",
          title: "本地历史修复记录 C",
          createdAt: "2026-07-01T00:00:00.000Z",
          updatedAt: "2026-07-01T08:00:00.000Z",
          cwd: "/repo/project-1",
          source: "cli",
          modelProvider: "openai",
          archived: false,
          sourcePath: "/Users/example/.codex/sessions/codex-thread-3.jsonl",
          importStatus: "not_imported",
        },
      ],
    });
    mockPreviewConversationImportThread.mockResolvedValue(
      buildMockConversationImportPreview(),
    );

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(4);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-import-conversation-button"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(4);

    await act(async () => {
      Array.from(document.body.querySelectorAll<HTMLButtonElement>("button"))
        .find((button) => button.textContent?.trim() === "选择本组")
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(2);

    await act(async () => {
      document.body
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-conversation-import-confirm"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(mockCommitConversationImportThread).toHaveBeenCalledTimes(2);
    expect(mockCommitConversationImportThread).toHaveBeenNthCalledWith(1, {
      sourceClient: "codex",
      sourceRoot: "/Users/example/.codex",
      sourceThreadId: "codex-thread-1",
      sourcePath: "/Users/example/.codex/sessions/codex-thread-1.jsonl",
      workspaceId: undefined,
      confirmed: true,
    });
    expect(mockCommitConversationImportThread).toHaveBeenNthCalledWith(2, {
      sourceClient: "codex",
      sourceRoot: "/Users/example/.codex",
      sourceThreadId: "codex-thread-2",
      sourcePath: "/Users/example/.codex/sessions/codex-thread-2.jsonl",
      workspaceId: undefined,
      confirmed: true,
    });
  });

  it("导入弹窗默认扫描归档会话，并支持只查看归档对话", async () => {
    mockScanConversationImportSource.mockResolvedValue({
      source: {
        sourceClient: "codex",
        status: "ready",
        sourceRoot: "/Users/example/.codex",
        readable: true,
        threadCount: 2,
        indexedAt: "2026-06-16T00:00:00.000Z",
        statePath: "/Users/example/.codex/state_5.sqlite",
      },
      threads: [
        {
          sourceClient: "codex",
          sourceThreadId: "active-thread",
          title: "未归档修复记录",
          createdAt: "2026-06-15T00:00:00.000Z",
          updatedAt: "2026-06-16T08:00:00.000Z",
          cwd: "/repo/project-1",
          source: "cli",
          modelProvider: "openai",
          archived: false,
          sourcePath: "/Users/example/.codex/sessions/active-thread.jsonl",
          importStatus: "not_imported",
        },
        {
          sourceClient: "codex",
          sourceThreadId: "archived-thread",
          title: "已归档修复记录",
          createdAt: "2026-06-15T00:00:00.000Z",
          updatedAt: "2026-06-16T09:00:00.000Z",
          cwd: "/repo/project-1",
          source: "cli",
          modelProvider: "openai",
          archived: true,
          sourcePath:
            "/Users/example/.codex/archived_sessions/archived-thread.jsonl",
          importStatus: "not_imported",
        },
      ],
    });
    mockPreviewConversationImportThread.mockResolvedValue(
      buildMockConversationImportPreview(),
    );

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(4);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-import-conversation-button"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(mockScanConversationImportSource).toHaveBeenCalledWith({
      sourceClient: "codex",
      sourceRoot: undefined,
      projectPath: undefined,
      includeArchived: true,
      limit: 40,
    });
    expect(document.body.textContent).toContain("未归档修复记录");
    expect(document.body.textContent).toContain("已归档修复记录");

    await act(async () => {
      Array.from(document.body.querySelectorAll<HTMLButtonElement>("button"))
        .find((button) => button.textContent?.trim() === "已归档")
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(document.body.textContent).not.toContain("未归档修复记录");
    expect(document.body.textContent).toContain("已归档修复记录");
    expect(document.body.textContent).toContain("可导入对话 1");
  });

  it("导入弹窗支持通过系统目录选择器切换本地历史数据目录", async () => {
    mockOpenDesktopDialog.mockResolvedValueOnce(
      "/Users/example/Library/Application Support/local-history",
    );

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(4);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-import-conversation-button"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(4);

    await act(async () => {
      document.body
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-conversation-import-choose-directory"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(mockOpenDesktopDialog).toHaveBeenCalledWith({
      title: "选择本地历史数据目录",
      directory: true,
      multiple: false,
    });
    expect(mockScanConversationImportSource).toHaveBeenLastCalledWith({
      sourceClient: "codex",
      sourceRoot: "/Users/example/Library/Application Support/local-history",
      projectPath: undefined,
      includeArchived: true,
      limit: 40,
    });
    expect(
      document.body.querySelector<HTMLInputElement>(
        'input[value="/Users/example/Library/Application Support/local-history"]',
      ),
    ).not.toBeNull();
  });

  it("项目没有可导入本地历史时项目菜单仍保留导入入口", async () => {
    mockGetProject.mockResolvedValue({
      id: "project-1",
      name: "示例项目",
      rootPath: "/repo/project-1",
      isFavorite: false,
    });
    mockScanConversationImportSource.mockResolvedValueOnce({
      source: {
        sourceClient: "codex",
        status: "ready",
        sourceRoot: "/Users/example/.codex",
        readable: true,
        threadCount: 0,
        message:
          "Found 2 matching Codex index records, but their rollout files are missing.",
        indexedAt: "2026-06-16T00:00:00.000Z",
        statePath: "/Users/example/.codex/state_5.sqlite",
      },
      threads: [],
    });

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(4);

    expect(
      container.querySelector(
        '[data-testid="app-sidebar-project-import-conversation"]',
      ),
    ).toBeNull();
    const projectMenu = await openProjectMenu("示例项目");

    expect(projectMenu?.textContent).toContain("导入对话");
    expect(
      document.body.querySelector(
        '[data-testid="app-sidebar-project-menu-import-conversation"]',
      ),
    ).not.toBeNull();

    await act(async () => {
      document.body
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-project-menu-import-conversation"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(4);

    expect(mockScanConversationImportSource).toHaveBeenCalledWith({
      sourceClient: "codex",
      projectPath: "/repo/project-1",
      includeArchived: true,
      limit: 40,
    });
    expect(document.body.textContent).toContain(
      "Found 2 matching Codex index records",
    );
  });

  it("从项目会话切到无项目对话时不应继承当前项目 ID", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-standalone",
        name: "无项目历史对话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: null,
        messages_count: 2,
      },
    ]);

    const onNavigate = vi.fn();
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

    const button = container.querySelector<HTMLButtonElement>(
      'button[title="无项目历史对话"]',
    );

    await act(async () => {
      button?.click();
      await Promise.resolve();
    });

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "claw",
        initialSessionId: "session-standalone",
      }),
    );
    expect((onNavigate.mock.calls[0]?.[1] as AgentPageParams).projectId).toBe(
      undefined,
    );
  });

  it("一级导航下方只展示已打开项目和非项目对话，归档不再作为左侧分区加载", async () => {
    localStorage.setItem(
      "agent_last_project_id",
      JSON.stringify("stale-project"),
    );
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-standalone",
        name: "非项目会话",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        workspace_id: null,
      },
      {
        id: "session-local-dir",
        name: "本地目录会话",
        created_at: 1713950000,
        updated_at: 1713950600,
        archived_at: null,
        workspace_id: null,
        working_dir: "/repo/local-history",
      },
      {
        id: "session-hidden-project",
        name: "未打开项目会话",
        created_at: 1713900000,
        updated_at: 1713900600,
        archived_at: null,
        workspace_id: "project-hidden",
      },
      {
        id: "session-archived",
        name: "归档会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: 1713003600,
        workspace_id: null,
      },
    ]);

    const container = mountSidebarContainer({
      currentPage: "settings",
    });
    await flushEffects(2);

    expect(container.textContent).toContain("项目");
    expect(container.textContent).toContain("对话");
    expect(container.textContent).toContain("非项目会话");
    expect(container.textContent).toContain("本地目录会话");
    expect(container.textContent).not.toContain("未打开项目会话");
    expect(container.textContent).not.toContain("归档会话");
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledTimes(1);
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
    });
    expect(mockListAgentRuntimeSessions).not.toHaveBeenCalledWith(
      expect.objectContaining({
        archivedOnly: true,
      }),
    );
    expect(mockRecordAgentUiPerformanceMetric).toHaveBeenCalledWith(
      "appSidebar.recentConversations.loadBreakdown",
      expect.objectContaining({
        limit: 11,
        projectCwds: [],
      }),
    );

    const mainNav = container.querySelector(
      '[data-testid="app-sidebar-main-nav"]',
    );
    const conversationShelf = container.querySelector(
      '[data-testid="app-sidebar-conversation-shelf"]',
    );
    const recentConversationList = container.querySelector(
      '[data-testid="app-sidebar-recent-conversations"]',
    );

    expect(mainNav).not.toBeNull();
    expect(conversationShelf).not.toBeNull();
    expect(recentConversationList).not.toBeNull();
    expect(getComputedStyle(recentConversationList as Element).overflowY).toBe(
      "auto",
    );
    expect(
      conversationShelf?.querySelector(
        '[data-testid="app-sidebar-archived-conversations"]',
      ),
    ).toBeNull();
    expect(
      conversationShelf?.querySelector('button[aria-expanded="false"]'),
    ).toBeNull();
    expect(
      Boolean(
        mainNav &&
        conversationShelf &&
        (mainNav.compareDocumentPosition(conversationShelf) &
          Node.DOCUMENT_POSITION_FOLLOWING) !==
          0,
      ),
    ).toBe(true);
  });

  it("点击项目标题行应折叠并重新展开项目下的对话列表", async () => {
    mockListAgentRuntimeSessions.mockImplementation(
      async (options?: { limit?: number; cwd?: string | string[] }) =>
        options?.cwd === "/repo/project-1"
          ? [
              {
                id: "session-project",
                name: "项目内会话",
                created_at: 1714000000,
                updated_at: 1714000600,
                archived_at: null,
                workspace_id: "project-1",
                working_dir: "/repo/project-1",
              },
            ]
          : [],
    );

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    const projectGroup = container.querySelector<HTMLElement>(
      '[data-testid="app-sidebar-project-conversation-group"]',
    );
    const projectButton = projectGroup?.querySelector<HTMLButtonElement>(
      'button[title="project-1"]',
    );

    expect(projectGroup).not.toBeNull();
    expect(projectButton).not.toBeNull();
    expect(projectButton?.getAttribute("aria-expanded")).toBe("true");
    expect(projectGroup?.textContent).toContain("项目内会话");

    await act(async () => {
      projectButton?.click();
      await Promise.resolve();
    });

    expect(projectButton?.getAttribute("aria-expanded")).toBe("false");
    expect(projectGroup?.textContent).toContain("project-1");
    expect(projectGroup?.textContent).not.toContain("项目内会话");

    await act(async () => {
      projectButton?.click();
      await Promise.resolve();
    });

    expect(projectButton?.getAttribute("aria-expanded")).toBe("true");
    expect(projectGroup?.textContent).toContain("项目内会话");
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-1",
    });
  });

  it("项目标题行应提供项目菜单", async () => {
    await mountProjectMenuScenario();

    const menu = await openProjectMenu("示例项目");
    expect(menu?.textContent).toContain("置顶项目");
    expect(menu?.textContent).toContain("显示位置");
    expect(menu?.textContent).toContain("创建永久工作树");
    expect(menu?.textContent).toContain("重命名项目");
    expect(menu?.textContent).not.toContain("归档对话");
    expect(menu?.textContent).toContain("移除");
  });

  it("项目菜单应支持置顶项目", async () => {
    await mountProjectMenuScenario();

    await openProjectMenu("示例项目");
    await clickConversationMenuItem("app-sidebar-project-menu-pin");
    await flushEffects(1);

    expect(mockUpdateProject).toHaveBeenCalledWith("project-1", {
      isFavorite: true,
    });
  });

  it("项目菜单应支持显示项目位置", async () => {
    await mountProjectMenuScenario();

    await openProjectMenu("示例项目");
    await clickConversationMenuItem("app-sidebar-project-menu-reveal");

    expect(mockRevealPathInFinder).toHaveBeenCalledWith("/repo/example");
  });

  it("项目动作应支持创建永久工作树", async () => {
    mockCreateProjectGitWorktree.mockResolvedValue({
      worktreePath: "/repo/example-worktree",
      branch: "main",
      status: {
        rootPath: "/repo/example",
        hasGitRepository: true,
        currentBranch: "main",
        branches: ["main"],
        uncommittedFileCount: 0,
      },
    });
    mockEnsureProjectWorkspace.mockResolvedValue({
      id: "project-worktree",
      name: "example-worktree",
      rootPath: "/repo/example-worktree",
      workspaceType: "general",
    });
    const harness = await mountProjectActionsHarness();
    const project: SidebarOpenedProjectSummary = {
      id: "project-1",
      name: "示例项目",
      rootPath: "/repo/example",
      isFavorite: false,
    };

    try {
      await act(async () => {
        await harness.actions.handleCreateProjectWorktree(project);
      });

      expect(mockCreateProjectGitWorktree).toHaveBeenCalledWith(
        "/repo/example",
      );
      expect(mockEnsureProjectWorkspace).toHaveBeenCalledWith({
        name: "example-worktree",
        rootPath: "/repo/example-worktree",
        workspaceType: "general",
      });
      expect(harness.onNavigate).toHaveBeenCalledWith("agent", {
        agentEntry: "claw",
        projectId: "project-worktree",
      });
    } finally {
      harness.cleanup();
    }
  });

  it("项目动作应支持重命名项目", async () => {
    vi.spyOn(window, "prompt").mockReturnValue("重命名项目");
    const harness = await mountProjectActionsHarness();
    const project: SidebarOpenedProjectSummary = {
      id: "project-1",
      name: "示例项目",
      rootPath: "/repo/example",
      isFavorite: false,
    };

    try {
      await act(async () => {
        await harness.actions.handleRenameProject(project);
      });

      expect(window.prompt).toHaveBeenCalledWith("重命名项目", "示例项目");
      expect(mockUpdateProject).toHaveBeenCalledWith("project-1", {
        name: "重命名项目",
      });
      expect(harness.refreshSidebarSessions).toHaveBeenCalledTimes(1);
    } finally {
      harness.cleanup();
    }
  });

  it("项目菜单应支持移除项目记录", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    await mountProjectMenuScenario();

    await openProjectMenu("示例项目");
    await clickConversationMenuItem("app-sidebar-project-menu-remove");
    await flushEffects(1);

    expect(window.confirm).toHaveBeenCalledWith(
      "确定要移除“示例项目”吗？本地目录不会被删除，后续仍可重新打开。",
    );
    expect(mockDeleteProject).toHaveBeenCalledWith("project-1", false);
  });

  it("最近对话不应把运行时错误包络展示成会话标题", async () => {
    localStorage.setItem("agent_last_project_id", JSON.stringify("project-1"));
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-provider-error",
        name: "Ran into this erro...",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        workspace_id: null,
      },
    ]);

    const container = mountSidebarContainer({
      currentPage: "settings",
    });
    await flushEffects(2);

    const recentConversationList = container.querySelector(
      '[data-testid="app-sidebar-recent-conversations"]',
    );

    expect(recentConversationList?.textContent).toContain("未命名对话");
    expect(recentConversationList?.textContent).not.toContain("Ran into");
  });

  it("对话加载失败时不应让记忆项目污染 App Server 查询参数", async () => {
    localStorage.setItem("agent_last_project_id", JSON.stringify("project-1"));
    const warnSpy = vi
      .spyOn(console, "warn")
      .mockImplementation(() => undefined);
    mockListAgentRuntimeSessions.mockImplementation(
      async (_options?: {
        includeArchived?: boolean;
        limit?: number;
        workspaceId?: string;
      }) => {
        throw new Error("recent failed");
      },
    );

    try {
      const container = mountSidebarContainer({
        currentPage: "settings",
      });
      await flushEffects(3);

      const recentConversationList = container.querySelector(
        '[data-testid="app-sidebar-recent-conversations"]',
      );
      expect(recentConversationList?.textContent).not.toContain("正在加载对话");
      expect(recentConversationList?.textContent).toContain("暂无聊天");
      expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
        limit: 11,
      });
      expect(mockListAgentRuntimeSessions).not.toHaveBeenCalledWith(
        expect.objectContaining({
          archivedOnly: true,
        }),
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it("没有当前工作区时应加载全局非项目对话", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-standalone",
        name: "全局对话",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        workspace_id: null,
      },
    ]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects(2);

    expect(container.textContent).toContain("对话");
    expect(container.textContent).toContain("全局对话");
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
    });
  });

  it("新任务首页应从全局分页结果归组记忆项目且不重复按 workspaceId 查询", async () => {
    localStorage.setItem("agent_last_project_id", JSON.stringify("project-1"));
    mockListAgentRuntimeSessions.mockImplementation(
      async (options?: {
        limit?: number;
        workspaceId?: string;
        cwd?: string | string[];
      }) =>
        options?.limit === 11 && !options.workspaceId && !options.cwd
          ? [
              {
                id: "session-workspace-memory",
                name: "记忆项目会话",
                created_at: 1714000000,
                updated_at: 1714000600,
                archived_at: null,
                workspace_id: "project-1",
              },
            ]
          : [],
    );

    const onNavigate = vi.fn();
    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
      onNavigate,
    });
    await flushEffects(4);

    const projectSection = container.querySelector(
      '[data-testid="app-sidebar-project-conversations"]',
    );
    expect(projectSection?.textContent).toContain("记忆项目会话");
    expect(mockListAgentRuntimeSessions).toHaveBeenCalled();
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({ limit: 11 });
    expect(
      mockListAgentRuntimeSessions.mock.calls.every(
        ([options]) => options?.limit === 11 && !options.workspaceId,
      ),
    ).toBe(true);

    const openButton = Array.from(
      container.querySelectorAll<HTMLButtonElement>(
        '[data-testid="app-sidebar-conversation-open"]',
      ),
    ).find((button) => button.getAttribute("title") === "记忆项目会话");
    expect(openButton).not.toBeNull();

    await act(async () => {
      openButton?.click();
      await Promise.resolve();
    });

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-workspace-memory",
      }),
    );
  });

  it("当前项目路径尚未解析时应先加载全局对话，避免首屏空列表", async () => {
    let resolveProject:
      | ((value: {
          id: string;
          name: string;
          rootPath: string;
          isFavorite: boolean;
        }) => void)
      | null = null;
    mockGetProject.mockReturnValue(
      new Promise((resolve) => {
        resolveProject = resolve;
      }),
    );
    mockListAgentRuntimeSessions.mockImplementation(
      async (options?: { limit?: number; cwd?: string | string[] }) =>
        options?.cwd === "/repo/project-1"
          ? [
              {
                id: "session-project",
                name: "项目路径会话",
                created_at: 1714000000,
                updated_at: 1714000600,
                archived_at: null,
                workspace_id: "project-1",
                working_dir: "/repo/project-1",
              },
            ]
          : [
              {
                id: "session-fixture",
                name: "fixture 会话",
                created_at: 1714000000,
                updated_at: 1714000600,
                archived_at: null,
                workspace_id: "project-1",
                working_dir: "/repo/project-1",
              },
            ],
    );

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    expect(container.textContent).toContain("fixture 会话");
    const projectSection = container.querySelector(
      '[data-testid="app-sidebar-project-conversations"]',
    );
    expect(projectSection?.textContent).toContain("fixture 会话");
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
    });
    expect(mockListAgentRuntimeSessions).not.toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-1",
    });

    await act(async () => {
      resolveProject?.({
        id: "project-1",
        name: "示例项目",
        rootPath: "/repo/project-1",
        isFavorite: false,
      });
      await Promise.resolve();
    });
    await flushEffects(3);

    expect(container.textContent).toContain("项目路径会话");
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-1",
    });
  });

  it("新任务首页空侧栏应短 idle 加载最近对话", async () => {
    const scheduledTasks: Array<{
      task: () => void;
      options?: { minimumDelayMs?: number; idleTimeoutMs?: number };
    }> = [];
    mockScheduleMinimumDelayIdleTask.mockImplementation(
      (
        task: () => void,
        options?: { minimumDelayMs?: number; idleTimeoutMs?: number },
      ) => {
        scheduledTasks.push({ task, options });
        return () => undefined;
      },
    );
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-new-task",
        name: "Claw fixture 会话",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        workspace_id: null,
      },
    ]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects(2);

    const deferredSessionLoad = scheduledTasks.find(
      (entry) =>
        entry.options?.minimumDelayMs === 0 &&
        entry.options?.idleTimeoutMs === 0,
    );
    expect(deferredSessionLoad).toBeDefined();

    await act(async () => {
      deferredSessionLoad?.task();
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
    });
    expect(container.textContent).toContain("Claw fixture 会话");
  });

  it("窗口重新聚焦时应低优先级刷新会话列表", async () => {
    localStorage.setItem("agent_last_project_id", JSON.stringify("project-1"));
    const cancelFocusRefresh = vi.fn();
    mockScheduleMinimumDelayIdleTask.mockImplementation((task: () => void) => {
      task();
      return cancelFocusRefresh;
    });

    mountSidebarContainer({
      currentPage: "settings",
    });
    await flushEffects(2);
    mockScheduleMinimumDelayIdleTask.mockClear();
    mockListAgentRuntimeSessions.mockClear();

    await act(async () => {
      window.dispatchEvent(new Event("focus"));
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(mockScheduleMinimumDelayIdleTask).toHaveBeenCalledTimes(1);
    expect(mockScheduleMinimumDelayIdleTask).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        minimumDelayMs: expect.any(Number),
        idleTimeoutMs: expect.any(Number),
      }),
    );
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
    });
  });

  it("会话列表变更事件应立即刷新最近对话", async () => {
    mockListAgentRuntimeSessions
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        {
          id: "session-external",
          name: "外部创建的会话",
          created_at: 1714000000,
          updated_at: 1714000600,
          archived_at: null,
          workspace_id: null,
        },
      ]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects(2);
    mockListAgentRuntimeSessions.mockClear();

    await act(async () => {
      window.dispatchEvent(
        new CustomEvent(AGENT_RUNTIME_SESSIONS_CHANGED_EVENT, {
          detail: {
            reason: "external",
            sessionId: "session-external",
          },
        }),
      );
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
    });
    expect(container.textContent).toContain("外部创建的会话");
  });

  it("创建会话事件应立即显示带稳定身份的占位入口", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "new-task",
      } as AgentPageParams,
    });
    await flushEffects(2);

    await act(async () => {
      window.dispatchEvent(
        new CustomEvent(AGENT_RUNTIME_SESSIONS_CHANGED_EVENT, {
          detail: {
            reason: "created",
            sessionId: "session-created-immediately",
            workspaceId: null,
          },
        }),
      );
      await Promise.resolve();
    });
    await flushEffects(1);

    const openButton = container.querySelector<HTMLButtonElement>(
      '[data-testid="app-sidebar-conversation-open"][data-session-id="session-created-immediately"]',
    );
    expect(openButton).not.toBeNull();
    expect(openButton?.getAttribute("title")).toBe("未命名对话");
  });

  it("首页发送热路径期间会话列表变更不应立即抢占 listSessions", async () => {
    const dateNowSpy = vi.spyOn(Date, "now").mockReturnValue(1_780_000_000_000);
    const scheduledTasks: Array<{
      task: () => void;
      options?: { minimumDelayMs?: number; idleTimeoutMs?: number };
    }> = [];
    mockScheduleMinimumDelayIdleTask.mockImplementation(
      (
        task: () => void,
        options?: { minimumDelayMs?: number; idleTimeoutMs?: number },
      ) => {
        scheduledTasks.push({ task, options });
        return () => undefined;
      },
    );
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-hot-path",
        name: "热路径前已有会话",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        workspace_id: null,
      },
    ]);

    try {
      mountSidebarContainer({
        currentPage: "agent",
        currentPageParams: {
          agentEntry: "new-task",
        } as AgentPageParams,
      });
      await flushEffects(2);

      await act(async () => {
        scheduledTasks.shift()?.task();
        await Promise.resolve();
      });
      await flushEffects(2);
      expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
        limit: 11,
      });

      mockListAgentRuntimeSessions.mockClear();
      scheduledTasks.splice(0, scheduledTasks.length);

      await act(async () => {
        emitMockAgentUiPerformanceMetricRecorded({
          id: 1,
          phase: "homeInput.submit",
          sessionId: "task-draft-hot",
          source: "task-center-empty-state",
        });
        window.dispatchEvent(
          new CustomEvent(AGENT_RUNTIME_SESSIONS_CHANGED_EVENT, {
            detail: {
              reason: "external",
              sessionId: "session-created-during-send",
            },
          }),
        );
        await Promise.resolve();
      });
      await flushEffects(1);

      expect(mockListAgentRuntimeSessions).not.toHaveBeenCalled();
      expect(scheduledTasks).toHaveLength(1);
      expect(scheduledTasks[0]?.options).toEqual(
        expect.objectContaining({
          minimumDelayMs: expect.any(Number),
          idleTimeoutMs: expect.any(Number),
        }),
      );
      expect(scheduledTasks[0]?.options?.minimumDelayMs ?? 0).toBeGreaterThan(
        0,
      );

      const deferredTask = scheduledTasks.shift()?.task;
      dateNowSpy.mockReturnValue(1_780_000_031_000);
      await act(async () => {
        deferredTask?.();
        await Promise.resolve();
      });
      await flushEffects(2);

      expect(mockListAgentRuntimeSessions).toHaveBeenCalledTimes(1);
      expect(scheduledTasks).toHaveLength(0);
    } finally {
      dateNowSpy.mockRestore();
    }
  });

  it("当前会话 metadata 更新应延迟合并刷新最近对话", async () => {
    const scheduledTasks: Array<() => void> = [];
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-current",
        name: "当前会话",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        workspace_id: null,
      },
    ]);

    mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        initialSessionId: "session-current",
      } as AgentPageParams,
    });
    await flushEffects(2);
    mockListAgentRuntimeSessions.mockClear();
    mockScheduleMinimumDelayIdleTask.mockImplementation((task: () => void) => {
      scheduledTasks.push(task);
      return () => undefined;
    });

    await act(async () => {
      window.dispatchEvent(
        new CustomEvent(AGENT_RUNTIME_SESSIONS_CHANGED_EVENT, {
          detail: {
            reason: "updated",
            sessionId: "session-current",
          },
        }),
      );
      await Promise.resolve();
    });
    await flushEffects(1);

    expect(mockListAgentRuntimeSessions).not.toHaveBeenCalled();
    expect(scheduledTasks).toHaveLength(1);

    await act(async () => {
      scheduledTasks[0]?.();
      await Promise.resolve();
    });

    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
    });
  });

  it("最近对话应限制初始渲染数量，并保留当前会话可见", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue(
      Array.from({ length: 25 }, (_, index) => {
        const order = index + 1;
        return {
          id: `session-${order}`,
          name: `会话 ${order}`,
          created_at: 1714000000 - order,
          updated_at: 1714000600 - order,
          archived_at: null,
          workspace_id: "project-1",
          working_dir: "/repo/project-1",
        };
      }),
    );

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-25",
      } as AgentPageParams,
    });
    await flushEffects(2);

    expect(container.querySelector('button[title="会话 1"]')).not.toBeNull();
    expect(container.querySelector('button[title="会话 25"]')).not.toBeNull();
    expect(container.querySelector('button[title="会话 24"]')).toBeNull();
    expect(container.textContent).toContain("查看更多对话");
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-1",
    });

    const targetButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("查看更多对话"),
    );

    expect(targetButton).not.toBeUndefined();

    await act(async () => {
      (targetButton as HTMLButtonElement | undefined)?.click();
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(container.querySelector('button[title="会话 19"]')).not.toBeNull();
  });

  it("切换人物或项目上下文时不应把已有最近对话重置成加载态", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-recent",
        name: "最近会话",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
      },
    ]);

    const mounted = mountSidebar({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    expect(mounted.container.textContent).toContain("最近会话");
    expect(mounted.container.textContent).not.toContain("正在加载对话");
    expect(mockListAgentRuntimeSessions).toHaveBeenNthCalledWith(1, {
      limit: 11,
    });
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-1",
    });

    await act(async () => {
      mounted.root.render(
        <AppSidebar
          currentPage="agent"
          currentPageParams={
            {
              agentEntry: "claw",
              projectId: "project-2",
            } as AgentPageParams
          }
          onNavigate={vi.fn()}
        />,
      );
      await Promise.resolve();
    });
    await flushEffects(2);

    const projectGroups = [
      ...mounted.container.querySelectorAll<HTMLElement>(
        '[data-testid="app-sidebar-project-conversation-group"]',
      ),
    ];
    expect(projectGroups.map((group) => group.textContent)).toEqual(
      expect.arrayContaining([
        expect.stringContaining("project-1"),
        expect.stringContaining("project-2"),
      ]),
    );
    expect(mounted.container.textContent).toContain("暂无聊天");
    expect(mounted.container.textContent).not.toContain("正在加载对话");
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-2",
    });
  });

  it("打开已有会话时若导航已有缓存任务，不应立即刷新最近对话列表", async () => {
    const scheduledTasks: Array<() => void> = [];
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-current",
        name: "最近会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
        messages_count: 3,
      },
    ]);

    const mounted = mountSidebar({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "new-task",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
    });
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-1",
    });
    const callsBeforeNavigation =
      mockListAgentRuntimeSessions.mock.calls.length;

    mockScheduleMinimumDelayIdleTask.mockImplementation((task: () => void) => {
      scheduledTasks.push(task);
      return () => undefined;
    });

    act(() => {
      mounted.root.render(
        <AppSidebar
          currentPage="agent"
          currentPageParams={
            {
              agentEntry: "claw",
              projectId: "project-1",
              initialSessionId: "session-current",
            } as AgentPageParams
          }
          onNavigate={vi.fn()}
        />,
      );
    });
    await flushEffects(2);

    expect(mockListAgentRuntimeSessions).toHaveBeenCalledTimes(
      callsBeforeNavigation,
    );
    expect(scheduledTasks).toHaveLength(1);

    await act(async () => {
      scheduledTasks[0]?.();
      await Promise.resolve();
    });

    expect(mockListAgentRuntimeSessions.mock.calls.length).toBeGreaterThan(
      callsBeforeNavigation,
    );
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-1",
    });
  });

  it("流式输出期间应延后最近对话列表刷新到终态后", async () => {
    const scheduledTasks: Array<() => void> = [];
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-current",
        name: "正在输出的会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
        messages_count: 3,
      },
    ]);

    const mounted = mountSidebar({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-current",
      } as AgentPageParams,
      activeAgentSessionId: "session-current",
    });
    await flushEffects(2);

    const callsBeforeStreaming = mockListAgentRuntimeSessions.mock.calls.length;
    expect(callsBeforeStreaming).toBeGreaterThan(0);

    mockScheduleMinimumDelayIdleTask.mockImplementation((task: () => void) => {
      scheduledTasks.push(task);
      return () => undefined;
    });

    await act(async () => {
      mounted.root.render(
        <AppSidebar
          currentPage="agent"
          currentPageParams={
            {
              agentEntry: "claw",
              projectId: "project-1",
              initialSessionId: "session-current",
            } as AgentPageParams
          }
          activeAgentSessionId="session-current"
          activeAgentStreaming={true}
          onNavigate={vi.fn()}
        />,
      );
      await Promise.resolve();
    });

    act(() => {
      window.dispatchEvent(
        new CustomEvent(AGENT_RUNTIME_SESSIONS_CHANGED_EVENT, {
          detail: {
            reason: "updated",
            sessionId: "session-current",
          },
        }),
      );
    });
    await flushEffects(2);

    const runtimeStatus = mounted.container.querySelector(
      '[data-testid="app-sidebar-conversation-runtime-status"]',
    );
    expect(runtimeStatus?.getAttribute("data-status")).toBe("running");
    expect(runtimeStatus?.getAttribute("aria-label")).toBe("正在输出");

    expect(mockListAgentRuntimeSessions).toHaveBeenCalledTimes(
      callsBeforeStreaming,
    );
    expect(scheduledTasks).toHaveLength(0);

    await act(async () => {
      mounted.root.render(
        <AppSidebar
          currentPage="agent"
          currentPageParams={
            {
              agentEntry: "claw",
              projectId: "project-1",
              initialSessionId: "session-current",
            } as AgentPageParams
          }
          activeAgentSessionId="session-current"
          activeAgentStreaming={false}
          onNavigate={vi.fn()}
        />,
      );
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(scheduledTasks).toHaveLength(1);

    await act(async () => {
      scheduledTasks[0]?.();
      await Promise.resolve();
    });

    expect(mockListAgentRuntimeSessions.mock.calls.length).toBeGreaterThan(
      callsBeforeStreaming,
    );
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-1",
    });
  });

  it("回到首页时应按后台未完成会话状态显示侧栏运行图标", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-current",
        name: "正在后台输出的会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        messages_count: 3,
      },
    ]);

    const mounted = mountSidebar({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
      } as AgentPageParams,
      activeAgentSessionId: null,
      activeAgentStreaming: false,
      backgroundAgentSessionRuntime: {
        sessionId: "session-current",
        status: "running",
      },
    });
    await flushEffects(2);

    const runtimeStatus = mounted.container.querySelector(
      '[data-testid="app-sidebar-conversation-runtime-status"]',
    );
    expect(runtimeStatus?.getAttribute("data-status")).toBe("running");
    expect(runtimeStatus?.getAttribute("aria-label")).toBe("正在输出");
    expect(
      runtimeStatus?.closest("[data-active]")?.getAttribute("data-active"),
    ).toBe("false");
  });

  it("回到首页时应按后台排队会话状态显示侧栏排队图标", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-current",
        name: "正在后台排队的会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        messages_count: 3,
      },
    ]);

    const mounted = mountSidebar({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
      } as AgentPageParams,
      activeAgentSessionId: null,
      activeAgentStreaming: false,
      backgroundAgentSessionRuntime: {
        sessionId: "session-current",
        status: "queued",
      },
    });
    await flushEffects(2);

    const runtimeStatus = mounted.container.querySelector(
      '[data-testid="app-sidebar-conversation-runtime-status"]',
    );
    expect(runtimeStatus?.getAttribute("data-status")).toBe("queued");
    expect(runtimeStatus?.getAttribute("aria-label")).toBe("排队中");
    expect(
      runtimeStatus?.closest("[data-active]")?.getAttribute("data-active"),
    ).toBe("false");
  });

  it("后台状态不会覆盖已完成的侧栏会话终态", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-current",
        name: "已完成的会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: null,
        messages_count: 3,
        latest_turn_status: "completed",
      },
    ]);

    const mounted = mountSidebar({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
      } as AgentPageParams,
      backgroundAgentSessionRuntime: {
        sessionId: "session-current",
        status: "running",
      },
    });
    await flushEffects(2);

    expect(
      mounted.container.querySelector(
        '[data-testid="app-sidebar-conversation-runtime-status"]',
      ),
    ).toBeNull();
  });

  it("点击会话菜单归档动作时应走 thread/archive", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-recent",
        name: "最近会话",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
      },
    ]);

    mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    const menu = await openConversationMenu("最近会话");
    expect(menu).not.toBeNull();
    expect(menu?.textContent).toContain("重命名");
    expect(menu?.textContent).toContain("置顶");
    expect(menu?.textContent).toContain("归档");
    expect(menu?.textContent).not.toContain("多选");
    expect(menu?.textContent).toContain("删除");
    const archiveMenuItem = document.body.querySelector<HTMLElement>(
      '[data-testid="app-sidebar-conversation-menu-archive"]',
    );
    expect(archiveMenuItem).not.toBeNull();
    expect(getComputedStyle(archiveMenuItem as Element).fontSize).toBe("13px");
    expect(getComputedStyle(archiveMenuItem as Element).minHeight).toBe("36px");
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-1",
    });

    await clickConversationMenuItem("app-sidebar-conversation-menu-archive");

    expect(mockArchiveAgentRuntimeSession).toHaveBeenCalledWith(
      "session-recent",
    );
  });

  it("项目上下文只展示项目会话，对话区保留新建入口", async () => {
    mockListAgentRuntimeSessions.mockImplementation(
      async (options?: { limit?: number; cwd?: string | string[] }) =>
        options?.cwd === "/repo/project-1"
          ? [
              {
                id: "session-project",
                name: "项目内会话",
                created_at: 1714000000,
                updated_at: 1714000600,
                archived_at: null,
                workspace_id: "project-1",
                working_dir: "/repo/project-1",
              },
            ]
          : [
              {
                id: "session-standalone",
                name: "独立会话",
                created_at: 1713900000,
                updated_at: 1713900600,
                archived_at: null,
                workspace_id: null,
              },
            ],
    );

    const onNavigate = vi.fn();
    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
      onNavigate,
    });
    await flushEffects(2);

    const shelf = container.querySelector(
      '[data-testid="app-sidebar-conversation-shelf"]',
    );
    const projectSection = container.querySelector(
      '[data-testid="app-sidebar-project-conversations"]',
    );
    const conversationSection = container.querySelector(
      '[data-testid="app-sidebar-recent-conversations"]',
    );
    const projectMenuButton = container.querySelector<HTMLButtonElement>(
      '[data-testid="app-sidebar-project-menu-button"]',
    );
    const newConversationButton = container.querySelector<HTMLButtonElement>(
      '[data-testid="app-sidebar-new-conversation-button"]',
    );

    expect(shelf?.textContent).toContain("项目");
    expect(shelf?.textContent).toContain("对话");
    expect(projectSection?.textContent).toContain("项目内会话");
    expect(projectSection?.textContent).not.toContain("独立会话");
    expect(conversationSection?.textContent).toContain("独立会话");
    expect(mockListAgentRuntimeSessions).toHaveBeenNthCalledWith(1, {
      limit: 11,
    });
    expect(mockListAgentRuntimeSessions).not.toHaveBeenCalledWith(
      expect.objectContaining({ workspaceId: "project-1" }),
    );
    expect(mockListAgentRuntimeSessions).toHaveBeenCalledWith({
      limit: 11,
      cwd: "/repo/project-1",
    });
    expect(projectMenuButton).not.toBeNull();
    expect(newConversationButton).not.toBeNull();
  });

  it("项目会话只有 workspaceId 时也应展示并可从侧栏打开", async () => {
    mockListAgentRuntimeSessions.mockImplementation(
      async (options?: {
        limit?: number;
        cwd?: string | string[];
        workspaceId?: string;
      }) =>
        options?.limit === 11 && !options.workspaceId && !options.cwd
          ? [
              {
                id: "session-workspace-only",
                name: "仅工作区会话",
                created_at: 1714000000,
                updated_at: 1714000600,
                archived_at: null,
                workspace_id: "project-1",
                working_dir: null,
              },
            ]
          : [],
    );

    const onNavigate = vi.fn();
    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
      onNavigate,
    });
    await flushEffects(2);

    const button = container.querySelector<HTMLButtonElement>(
      'button[title="仅工作区会话"]',
    );
    expect(button).not.toBeNull();
    expect(mockListAgentRuntimeSessions).not.toHaveBeenCalledWith(
      expect.objectContaining({ workspaceId: "project-1" }),
    );

    await act(async () => {
      button?.click();
      await Promise.resolve();
    });

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "claw",
        projectId: "project-1",
        initialSessionId: "session-workspace-only",
      }),
    );
  });

  it("项目上下文里的顶部新建任务应显式创建项目内对话", async () => {
    const taskCenterRequests = captureTaskCenterDraftRequests();

    try {
      const container = mountSidebarContainer({
        currentPage: "agent",
        currentPageParams: {
          agentEntry: "claw",
          projectId: "project-1",
        } as AgentPageParams,
      });
      await flushEffects(2);

      await act(async () => {
        container
          .querySelector<HTMLButtonElement>(
            '[data-testid="app-sidebar-nav-home-general"]',
          )
          ?.click();
        await Promise.resolve();
      });

      expect(taskCenterRequests.receivedDetails).toEqual([
        { source: "sidebar", projectId: "project-1" },
      ]);
    } finally {
      taskCenterRequests.dispose();
    }
  });

  it("项目上下文里的对话加号应显式创建无项目对话", async () => {
    const taskCenterRequests = captureTaskCenterDraftRequests();

    try {
      const container = mountSidebarContainer({
        currentPage: "agent",
        currentPageParams: {
          agentEntry: "claw",
          projectId: "project-1",
        } as AgentPageParams,
      });
      await flushEffects(2);

      await act(async () => {
        container
          .querySelector<HTMLButtonElement>(
            '[data-testid="app-sidebar-new-conversation-button"]',
          )
          ?.click();
        await Promise.resolve();
      });

      expect(taskCenterRequests.receivedDetails).toEqual([
        { source: "sidebar", projectId: null },
      ]);
    } finally {
      taskCenterRequests.dispose();
    }
  });

  it("任务中心不可拦截时，对话加号 fallback 也不应继承当前项目 ID", async () => {
    mockListAgentRuntimeSessions.mockImplementation(
      async (options?: { limit?: number; cwd?: string | string[] }) =>
        options?.cwd === "/repo/project-1"
          ? [
              {
                id: "session-project",
                name: "项目内会话",
                created_at: 1714000000,
                updated_at: 1714000600,
                archived_at: null,
                workspace_id: "project-1",
                working_dir: "/repo/project-1",
              },
            ]
          : [],
    );

    const onNavigate = vi.fn();
    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
      onNavigate,
    });
    await flushEffects(2);

    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-new-conversation-button"]',
        )
        ?.click();
      await Promise.resolve();
    });

    expect(onNavigate).toHaveBeenCalledWith(
      "agent",
      expect.objectContaining({
        agentEntry: "new-task",
        newChatAt: expect.any(Number),
      }),
    );
    const nextParams = onNavigate.mock.calls[0]?.[1] as AgentPageParams;
    expect(nextParams.projectId).toBe(undefined);
    expect(nextParams.initialSessionId).toBe(undefined);
  });

  it("项目分组的新建对话入口应显式创建项目内对话", async () => {
    const taskCenterRequests = captureTaskCenterDraftRequests();

    try {
      await mountProjectMenuScenario();
      const projectNewConversationButton =
        document.body.querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-project-new-conversation"]',
        );

      await act(async () => {
        projectNewConversationButton?.click();
        await Promise.resolve();
      });

      expect(taskCenterRequests.receivedDetails).toEqual([
        { source: "sidebar", projectId: "project-1" },
      ]);
    } finally {
      taskCenterRequests.dispose();
    }
  });

  it("侧边栏不再暴露旧区块菜单、多选和批量归档入口", async () => {
    mockListAgentRuntimeSessions.mockImplementation(
      async (options?: { limit?: number; cwd?: string | string[] }) =>
        options?.cwd === "/repo/project-1"
          ? [
              {
                id: "session-project",
                name: "项目内会话",
                created_at: 1714000000,
                updated_at: 1714000600,
                archived_at: null,
                workspace_id: "project-1",
                working_dir: "/repo/project-1",
              },
            ]
          : [
              {
                id: "session-standalone",
                name: "独立会话",
                created_at: 1713900000,
                updated_at: 1713900600,
                archived_at: null,
                workspace_id: null,
              },
            ],
    );

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    expect(
      container.querySelector(
        '[data-testid="app-sidebar-projects-shelf-menu-button"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector(
        '[data-testid="app-sidebar-conversations-shelf-menu-button"]',
      ),
    ).toBeNull();
    expect(
      document.body.querySelector(
        '[data-testid="app-sidebar-conversation-shelf-menu-archive-all"]',
      ),
    ).toBeNull();

    expect(container.textContent).toContain("独立会话");
    expect(await openConversationMenu("独立会话")).not.toBeNull();
    expect(
      document.body.querySelector(
        '[data-testid="app-sidebar-conversation-menu-multiselect"]',
      ),
    ).toBeNull();
  });

  it("归档会话不应出现在左侧导航恢复菜单中", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-archived",
        name: "归档会话",
        created_at: 1713000000,
        updated_at: 1713000600,
        archived_at: 1713003600,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
      },
    ]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    expect(container.textContent).not.toContain("归档会话");
    expect(
      container.querySelector(
        '[data-testid="app-sidebar-archived-conversations"]',
      ),
    ).toBeNull();
    expect(mockListAgentRuntimeSessions).not.toHaveBeenCalledWith(
      expect.objectContaining({
        archivedOnly: true,
      }),
    );
    expect(await openConversationMenu("归档会话")).toBeNull();
  });

  it("会话菜单应支持重命名并同步更新 session 名称", async () => {
    vi.spyOn(window, "prompt").mockReturnValue("重命名后的会话");
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-recent",
        thread_id: "thread-recent",
        name: "最近会话",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
      },
    ]);

    mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    await openConversationMenu("最近会话");
    await clickConversationMenuItem("app-sidebar-conversation-menu-rename");
    await flushEffects(2);

    expect(window.prompt).toHaveBeenCalledWith("重命名对话", "最近会话");
    expect(mockSetAgentRuntimeThreadName).toHaveBeenCalledWith({
      threadId: "thread-recent",
      name: "重命名后的会话",
    });
    expect(mockToastSuccess).toHaveBeenCalledWith("已重命名对话");
  });

  it("会话菜单应支持删除并在执行前要求确认", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-recent",
        name: "最近会话",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
      },
    ]);

    mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    await openConversationMenu("最近会话");
    await clickConversationMenuItem("app-sidebar-conversation-menu-delete");
    await flushEffects(2);

    expect(window.confirm).toHaveBeenCalledWith(
      "确定要删除“最近会话”吗？删除后无法恢复。",
    );
    expect(mockDeleteAgentRuntimeSession).toHaveBeenCalledWith(
      "session-recent",
    );
    expect(mockToastSuccess).toHaveBeenCalledWith("已删除对话");
  });

  it("会话菜单置顶应写入 canonical Pinned section 并进入置顶分组", async () => {
    let pinned = false;
    mockMoveThreadToSection.mockImplementation(async () => {
      pinned = true;
    });
    mockListAgentRuntimeSessions.mockImplementation(async () => [
      {
        id: "session-recent",
        thread_id: "thread-recent",
        name: "最近会话",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        workspace_id: "project-1",
        working_dir: "/repo/project-1",
        ...(pinned
          ? {
              section: {
                id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
                name: "Pinned",
              },
            }
          : {}),
      },
    ]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    await openConversationMenu("最近会话");
    await clickConversationMenuItem("app-sidebar-conversation-menu-pin");
    await flushEffects(2);

    expect(mockMoveThreadToSection).toHaveBeenCalledWith({
      threadId: "thread-recent",
      sectionId: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
    });
    expect(
      container.querySelector(
        '[data-testid="app-sidebar-thread-section"][data-section-id="01984de2-8f74-7c91-a3b2-5c5e937cf318"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="app-sidebar-thread-section"]')
        ?.textContent,
    ).toContain("置顶");

    const pinnedMenu = await openConversationMenu("最近会话");
    expect(pinnedMenu?.textContent).toContain("取消置顶");
    expect(
      container.querySelector(
        '[data-testid="app-sidebar-conversation-multiselect-toolbar"]',
      ),
    ).toBeNull();
  });

  it("侧栏应支持新建、重命名和删除空 custom section", async () => {
    mockListThreadSections.mockResolvedValue([
      {
        id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
        name: "Pinned",
      },
      { id: "section-active", name: "进行中" },
    ]);
    mockCreateThreadSection.mockResolvedValue({
      id: "section-created",
      name: "待处理",
    });
    const prompt = vi
      .spyOn(window, "prompt")
      .mockReturnValueOnce("待处理")
      .mockReturnValueOnce("已确认");
    vi.spyOn(window, "confirm").mockReturnValue(true);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    expect(
      container.querySelector(
        '[data-testid="app-sidebar-thread-section"][data-section-id="section-active"]',
      )?.textContent,
    ).toContain("进行中");
    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-new-thread-section-button"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(prompt).toHaveBeenNthCalledWith(1, "新建分组");
    expect(mockCreateThreadSection).toHaveBeenCalledWith({ name: "待处理" });
    const createdSection = container.querySelector<HTMLElement>(
      '[data-testid="app-sidebar-thread-section"][data-section-id="section-created"]',
    );
    expect(createdSection?.textContent).toContain("待处理");

    await act(async () => {
      createdSection
        ?.querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-thread-section-rename"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(prompt).toHaveBeenNthCalledWith(2, "重命名分组", "待处理");
    expect(mockUpdateThreadSection).toHaveBeenCalledWith({
      sectionId: "section-created",
      name: "已确认",
    });
    expect(
      container.querySelector(
        '[data-testid="app-sidebar-thread-section"][data-section-id="section-created"]',
      )?.textContent,
    ).toContain("已确认");

    await act(async () => {
      container
        .querySelector<HTMLElement>(
          '[data-testid="app-sidebar-thread-section"][data-section-id="section-created"]',
        )
        ?.querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-thread-section-delete"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(window.confirm).toHaveBeenCalledWith(
      "确定要删除分组“已确认”吗？其中的对话会回到未分组列表。",
    );
    expect(mockDeleteThreadSection).toHaveBeenCalledWith({
      sectionId: "section-created",
    });
    expect(
      container.querySelector(
        '[data-testid="app-sidebar-thread-section"][data-section-id="section-created"]',
      ),
    ).toBeNull();
  });

  it("会话菜单应在 custom section 与未分组之间移动", async () => {
    let sectionId: string | null = null;
    mockListThreadSections.mockResolvedValue([
      {
        id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
        name: "Pinned",
      },
      { id: "section-active", name: "进行中" },
    ]);
    mockMoveThreadToSection.mockImplementation(async (params) => {
      sectionId = params.sectionId;
    });
    mockListAgentRuntimeSessions.mockImplementation(async () => [
      {
        id: "session-recent",
        thread_id: "thread-recent",
        name: "最近会话",
        created_at: 1714000000,
        updated_at: 1714000600,
        archived_at: null,
        ...(sectionId ? { section: { id: sectionId, name: "进行中" } } : {}),
      },
    ]);

    const container = mountSidebarContainer({
      currentPage: "agent",
      currentPageParams: {
        agentEntry: "claw",
        projectId: "project-1",
      } as AgentPageParams,
    });
    await flushEffects(2);

    await openConversationMenu("最近会话");
    await clickConversationMenuItem(
      "app-sidebar-conversation-menu-move-section",
    );
    await act(async () => {
      document.body
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-conversation-menu-section-option"][data-section-id="section-active"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(mockMoveThreadToSection).toHaveBeenNthCalledWith(1, {
      threadId: "thread-recent",
      sectionId: "section-active",
    });
    expect(
      container.querySelector(
        '[data-testid="app-sidebar-thread-section"][data-section-id="section-active"]',
      )?.textContent,
    ).toContain("最近会话");

    await openConversationMenu("最近会话");
    await clickConversationMenuItem(
      "app-sidebar-conversation-menu-move-section",
    );
    await act(async () => {
      document.body
        .querySelector<HTMLButtonElement>(
          '[data-testid="app-sidebar-conversation-menu-section-option"][data-section-id="unsectioned"]',
        )
        ?.click();
      await Promise.resolve();
    });
    await flushEffects(2);

    expect(mockMoveThreadToSection).toHaveBeenNthCalledWith(2, {
      threadId: "thread-recent",
      sectionId: null,
    });
    expect(
      container.querySelector(
        '[data-testid="app-sidebar-recent-conversations"]',
      )?.textContent,
    ).toContain("最近会话");
  });
});
