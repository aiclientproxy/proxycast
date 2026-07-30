import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  mockStreamingRenderer,
  mockAgentThreadTimeline,
  render,
  createConversationMessages,
  getAgentUiPerformanceMetrics,
} from "./MessageList.testHarness";
import type {
  AgentThreadItem,
  AgentThreadTurn,
  Message,
} from "./MessageList.testHarness";

describe("MessageList history window", () => {
  it("旧会话首屏只加载尾部历史时应提供继续加载入口", () => {
    const onLoadFullHistory = vi.fn();
    const container = render(createConversationMessages(2), {
      sessionHistoryWindow: {
        loadedMessages: 2,
        totalMessages: 320,
        isLoadingFull: false,
        error: null,
      },
      onLoadFullHistory,
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-persisted-history-window"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain(
      "latest 1 conversation entries are shown first",
    );
    expect(container.textContent).not.toContain("/ 320");

    const button = container.querySelector(
      '[data-testid="message-list-load-full-history"]',
    ) as HTMLButtonElement | null;
    expect(button).not.toBeNull();
    expect(button?.textContent).toContain("Load more history");

    act(() => {
      button?.click();
    });

    expect(onLoadFullHistory).toHaveBeenCalledTimes(1);
  });

  it("历史总数未知时不应展示猜测总数", () => {
    const container = render(createConversationMessages(2), {
      sessionHistoryWindow: {
        loadedMessages: 2,
        totalMessages: null,
        hasMore: true,
        isLoadingFull: false,
        error: null,
      },
      onLoadFullHistory: vi.fn(),
    });

    expect(container.textContent).toContain(
      "latest 1 conversation entries are shown first; earlier history is still available",
    );
    expect(container.textContent).not.toContain("latest 1 /");
    expect(
      container.querySelector('[data-testid="message-list-load-full-history"]'),
    ).not.toBeNull();
  });

  it("canonical cursor 已到 EOF 时不应按旧 Message 总数显示加载入口", () => {
    const container = render(createConversationMessages(2), {
      sessionHistoryWindow: {
        loadedMessages: 2,
        totalMessages: 320,
        hasMore: false,
        isLoadingFull: false,
        error: null,
      },
      onLoadFullHistory: vi.fn(),
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-persisted-history-window"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="message-list-load-full-history"]'),
    ).toBeNull();
  });

  it("canonical Turn 与 residual Message 数量不同时应由 render entry 负责加载计数", () => {
    const timestamp = "2026-04-25T10:00:00.000Z";
    const turns: AgentThreadTurn[] = Array.from({ length: 3 }, (_, index) => ({
      id: `turn-entry-owner-${index + 1}`,
      thread_id: "thread-entry-owner",
      prompt_text: `canonical ${index + 1}`,
      status: "completed",
      started_at: timestamp,
      completed_at: timestamp,
      created_at: timestamp,
      updated_at: timestamp,
    }));
    const threadItems: AgentThreadItem[] = turns.map((turn, index) => ({
      id: `item-entry-owner-${index + 1}`,
      thread_id: turn.thread_id,
      turn_id: turn.id,
      sequence: 1,
      status: "completed",
      started_at: timestamp,
      completed_at: timestamp,
      updated_at: timestamp,
      type: "tool_call",
      tool_name: "Read",
      arguments: { file_path: `/repo/entry-${index + 1}.ts` },
    }));
    const container = render(
      [
        {
          id: "residual-local-message",
          role: "assistant",
          content: "本地残留消息",
          timestamp: new Date(timestamp),
        } as Message,
      ],
      {
        currentTurnId: null,
        turns,
        threadItems,
        sessionHistoryWindow: {
          loadedMessages: 1,
          totalMessages: null,
          hasMore: true,
          isLoadingFull: false,
          error: null,
        },
        onLoadFullHistory: vi.fn(),
      },
    );

    expect(container.textContent).toContain(
      "latest 3 conversation entries are shown first",
    );
    expect(
      container.querySelectorAll('[data-render-entry-kind="canonical_turn"]'),
    ).toHaveLength(2);
    expect(
      container.querySelectorAll('[data-render-entry-kind="message_group"]'),
    ).toHaveLength(1);
  });

  it("旧会话首帧应先渲染消息文本并延后历史 timeline", async () => {
    vi.useFakeTimers();
    const messages = createConversationMessages(60);
    const turns: AgentThreadTurn[] = Array.from({ length: 30 }, (_, index) => {
      const startMinute = String(index * 2).padStart(2, "0");
      const completedMinute = String(index * 2 + 1).padStart(2, "0");
      return {
        id: `turn-${index + 1}`,
        thread_id: "thread-history",
        prompt_text: `消息 ${index * 2 + 1}`,
        status: "completed",
        started_at: `2026-04-25T10:${startMinute}:00.000Z`,
        completed_at: `2026-04-25T10:${completedMinute}:00.000Z`,
        created_at: `2026-04-25T10:${startMinute}:00.000Z`,
        updated_at: `2026-04-25T10:${completedMinute}:00.000Z`,
      };
    });
    const threadItems: AgentThreadItem[] = turns.map((turn, index) => ({
      id: `reasoning-${index + 1}`,
      thread_id: turn.thread_id,
      turn_id: turn.id,
      sequence: 1,
      status: "completed",
      started_at: turn.started_at,
      completed_at: turn.completed_at,
      updated_at: turn.updated_at,
      type: "tool_call",
      tool_name: "Read",
      arguments: { file_path: `/repo/history-${index + 1}.ts` },
    }));

    const container = render(messages, {
      currentTurnId: "turn-30",
      turns,
      threadItems,
    });

    expect(container.textContent).toContain("消息 60");
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(1_000);
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector('[data-testid="agent-thread-timeline:leading"]'),
    ).toBeNull();
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
  });

  it("旧会话已分页窗口首帧应只挂载更小的尾部批次", () => {
    vi.useFakeTimers();
    const messages = createConversationMessages(40);
    const container = render(messages, {
      sessionHistoryWindow: {
        loadedMessages: 40,
        totalMessages: 188,
        isLoadingFull: false,
        error: null,
      },
    });

    expect(container.textContent).toContain(
      "latest 20 conversation entries are shown first",
    );
    expect(container.textContent).toContain("消息 40");
    expect(container.textContent).toContain("消息 21");
    expect(container.textContent).not.toContain("消息 20Saturday");
    expect(
      container.querySelectorAll('[data-testid="message-turn-group"]'),
    ).toHaveLength(10);
    expect(container.textContent).toContain(
      "10 earlier entries can be expanded",
    );

    act(() => {
      vi.advanceTimersByTime(2_000);
    });

    expect(container.textContent).not.toContain("消息 20Saturday");

    const expandButton = container.querySelector(
      '[data-testid="message-list-expand-history"]',
    ) as HTMLButtonElement | null;

    act(() => {
      expandButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(
      container.querySelectorAll('[data-testid="message-turn-group"]'),
    ).toHaveLength(20);
  });

  it("旧会话消息较少但执行过程很多时也应延后构建 timeline", async () => {
    vi.useFakeTimers();
    const turn: AgentThreadTurn = {
      id: "turn-history-many-items",
      thread_id: "thread-history-many-items",
      prompt_text: "检查慢历史",
      status: "completed",
      started_at: "2026-04-25T10:00:00.000Z",
      completed_at: "2026-04-25T10:01:00.000Z",
      created_at: "2026-04-25T10:00:00.000Z",
      updated_at: "2026-04-25T10:01:00.000Z",
    };
    const threadItems: AgentThreadItem[] = Array.from(
      { length: 30 },
      (_, index): AgentThreadItem => {
        const base = {
          id: `history-heavy-item-${index + 1}`,
          thread_id: turn.thread_id,
          turn_id: turn.id,
          sequence: index + 1,
          status: "completed" as const,
          started_at: "2026-04-25T10:00:00.000Z",
          completed_at: "2026-04-25T10:01:00.000Z",
          updated_at: "2026-04-25T10:01:00.000Z",
        };

        if (index % 2 === 0) {
          return {
            ...base,
            type: "tool_call",
            tool_name: "Bash",
            arguments: { command: `echo ${index}` },
            output: `输出 ${index}`,
          };
        }

        return {
          ...base,
          type: "reasoning",
          text: `思考 ${index}`,
        };
      },
    );
    const container = render(
      [
        {
          id: "msg-user-history-many-items",
          role: "user",
          content: "检查慢历史",
          timestamp: new Date("2026-04-25T10:00:00.000Z"),
        } as Message,
        {
          id: "msg-assistant-history-many-items",
          role: "assistant",
          content: "历史结果",
          timestamp: new Date("2026-04-25T10:01:00.000Z"),
        } as Message,
      ],
      {
        currentTurnId: turn.id,
        turns: [turn],
        threadItems,
        sessionHistoryWindow: {
          loadedMessages: 2,
          totalMessages: 170,
          isLoadingFull: false,
          error: null,
        },
      },
    );

    expect(container.textContent).toContain("历史结果");
    expect(
      container.querySelector(
        '[data-testid="message-list-historical-timeline-preview:leading"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-runtime-status-line"]'),
    ).toBeNull();
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(880);
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-historical-timeline-preview:leading"]',
      ),
    ).not.toBeNull();

    act(() => {
      vi.advanceTimersByTime(60);
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-historical-timeline-preview:leading"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="inputbar-runtime-status-line"]'),
    ).not.toBeNull();
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
  });

  it("已分页旧会话首帧应建立 canonical ownership 并延后历史细节", () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const turns: AgentThreadTurn[] = Array.from({ length: 8 }, (_, index) => {
      const minute = String(index + 1).padStart(2, "0");
      return {
        id: `turn-window-${index + 1}`,
        thread_id: "thread-windowed-history",
        prompt_text: `历史问题 ${index + 1}`,
        status: "completed",
        started_at: `2026-04-25T10:${minute}:00.000Z`,
        completed_at: `2026-04-25T10:${minute}:30.000Z`,
        created_at: `2026-04-25T10:${minute}:00.000Z`,
        updated_at: `2026-04-25T10:${minute}:30.000Z`,
      };
    });
    const threadItems: AgentThreadItem[] = turns.flatMap((turn, turnIndex) =>
      Array.from(
        { length: 5 },
        (_, itemIndex): AgentThreadItem => ({
          id: `turn-window-${turnIndex + 1}-item-${itemIndex + 1}`,
          thread_id: turn.thread_id,
          turn_id: turn.id,
          sequence: itemIndex + 1,
          status: "completed",
          started_at: turn.started_at,
          completed_at: turn.completed_at,
          updated_at: turn.updated_at,
          type: "tool_call",
          tool_name: "Read",
          arguments: { file_path: `/repo/file-${itemIndex + 1}.ts` },
        }),
      ),
    );

    const container = render(
      [
        {
          id: "msg-user-windowed-history",
          role: "user",
          content: "打开尾部旧会话",
          timestamp: new Date("2026-04-25T10:08:00.000Z"),
        } as Message,
        {
          id: "msg-assistant-windowed-history",
          role: "assistant",
          content: "这是尾部旧会话结果",
          timestamp: new Date("2026-04-25T10:08:30.000Z"),
        } as Message,
      ],
      {
        sessionId: "session-windowed-history",
        currentTurnId: "turn-window-8",
        turns,
        threadItems,
        sessionHistoryWindow: {
          loadedMessages: 2,
          totalMessages: 220,
          isLoadingFull: false,
          error: null,
        },
      },
    );

    const commit = getAgentUiPerformanceMetrics().find(
      (entry) => entry.phase === "messageList.commit",
    );

    expect(commit?.metrics).toEqual(
      expect.objectContaining({
        renderedTurnsCount: 8,
        shouldDeferHistoricalTimeline: true,
        threadItemsCount: 40,
        threadItemsScanDeferred: false,
        turnsCount: 8,
      }),
    );

    act(() => {
      vi.advanceTimersByTime(940);
    });

    const idleCommit = getAgentUiPerformanceMetrics()
      .filter((entry) => entry.phase === "messageList.commit")
      .find(
        (entry) =>
          entry.metrics.threadItemsScanDeferred === false &&
          entry.metrics.canBuildHistoricalTimeline === true,
      );
    expect(idleCommit?.metrics).toEqual(
      expect.objectContaining({
        renderedTurnsCount: 8,
        threadItemsCount: 40,
        turnsCount: 8,
      }),
    );
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
    expect(container.textContent).not.toContain("/repo/file-");
  });

  it("旧历史窗口发送中但 active turn 尚未出现时仍应保持 canonical ownership", () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const turns: AgentThreadTurn[] = Array.from({ length: 8 }, (_, index) => {
      const minute = String(index + 1).padStart(2, "0");
      return {
        id: `turn-sending-window-${index + 1}`,
        thread_id: "thread-sending-windowed-history",
        prompt_text: `历史提问 ${index + 1}`,
        status: "completed",
        started_at: `2026-04-25T11:${minute}:00.000Z`,
        completed_at: `2026-04-25T11:${minute}:30.000Z`,
        created_at: `2026-04-25T11:${minute}:00.000Z`,
        updated_at: `2026-04-25T11:${minute}:30.000Z`,
      };
    });
    const threadItems: AgentThreadItem[] = turns.flatMap((turn, turnIndex) =>
      Array.from(
        { length: 5 },
        (_, itemIndex): AgentThreadItem => ({
          id: `turn-sending-window-${turnIndex + 1}-item-${itemIndex + 1}`,
          thread_id: turn.thread_id,
          turn_id: turn.id,
          sequence: itemIndex + 1,
          status: "completed",
          started_at: turn.started_at,
          completed_at: turn.completed_at,
          updated_at: turn.updated_at,
          type: "tool_call",
          tool_name: "Read",
          arguments: { file_path: `/repo/sending-${itemIndex + 1}.ts` },
        }),
      ),
    );

    const container = render(
      [
        {
          id: "msg-user-sending-windowed-history",
          role: "user",
          content: "打开旧会话继续追问",
          timestamp: new Date("2026-04-25T11:08:00.000Z"),
        } as Message,
        {
          id: "msg-assistant-sending-windowed-history",
          role: "assistant",
          content: "这是旧会话尾部结果",
          timestamp: new Date("2026-04-25T11:08:30.000Z"),
        } as Message,
        {
          id: "msg-user-sending-follow-up",
          role: "user",
          content: "继续整理重点",
          timestamp: new Date("2026-04-25T11:09:00.000Z"),
        } as Message,
      ],
      {
        sessionId: "session-sending-windowed-history",
        currentTurnId: "turn-follow-up-not-yet-visible",
        isSending: true,
        sessionHistoryWindow: {
          loadedMessages: 3,
          totalMessages: 220,
          isLoadingFull: false,
          error: null,
        },
        threadItems,
        turns,
      },
    );

    expect(container.textContent).toContain("继续整理重点");
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
    const commit = getAgentUiPerformanceMetrics().find(
      (entry) => entry.phase === "messageList.commit",
    );

    expect(commit?.metrics).toEqual(
      expect.objectContaining({
        renderedTurnsCount: 8,
        shouldDeferHistoricalTimeline: true,
        threadItemsCount: 40,
        threadItemsScanDeferred: false,
        turnsCount: 8,
      }),
    );

    act(() => {
      vi.advanceTimersByTime(940);
    });

    const idleCommit = getAgentUiPerformanceMetrics()
      .filter((entry) => entry.phase === "messageList.commit")
      .find(
        (entry) =>
          entry.metrics.threadItemsScanDeferred === false &&
          entry.metrics.canBuildHistoricalTimeline === true,
      );

    expect(idleCommit?.metrics).toEqual(
      expect.objectContaining({
        renderedTurnsCount: 8,
        threadItemsCount: 40,
        turnsCount: 8,
      }),
    );
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
    expect(container.textContent).not.toContain("/repo/sending-");
  });

  it("旧历史窗口发送中仍应保持尾部消息窗口，避免一次挂载完整历史", async () => {
    vi.useFakeTimers();
    const messages = createConversationMessages(40);
    const container = render(messages, {
      isSending: true,
      sessionId: "session-sending-history-window",
      sessionHistoryWindow: {
        loadedMessages: 40,
        totalMessages: 188,
        isLoadingFull: false,
        error: null,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(container.textContent).toContain(
      "latest 20 conversation entries are shown first",
    );
    expect(container.textContent).toContain("消息 40");
    expect(container.textContent).toContain("消息 21");
    expect(container.textContent).not.toContain("消息 20Saturday");
    expect(container.textContent).toContain(
      "10 earlier entries can be expanded",
    );

    const commit = getAgentUiPerformanceMetrics().find(
      (entry) => entry.phase === "messageList.commit",
    );

    expect(commit?.metrics).toEqual(
      expect.objectContaining({
        hiddenHistoryCount: 10,
        renderedMessagesCount: 20,
        visibleMessagesCount: 40,
      }),
    );
  });

  it("旧会话首帧应延后历史助手 contentParts 与 Markdown 细节扫描", () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const turn: AgentThreadTurn = {
      id: "turn-history-content-parts",
      thread_id: "thread-history-content-parts",
      prompt_text: "检查 content parts",
      status: "completed",
      started_at: "2026-04-25T10:00:00.000Z",
      completed_at: "2026-04-25T10:01:00.000Z",
      created_at: "2026-04-25T10:00:00.000Z",
      updated_at: "2026-04-25T10:01:00.000Z",
    };
    const threadItems: AgentThreadItem[] = Array.from(
      { length: 30 },
      (_, index): AgentThreadItem => ({
        id: `history-content-parts-tool-${index + 1}`,
        thread_id: turn.thread_id,
        turn_id: turn.id,
        sequence: index + 1,
        status: "completed",
        started_at: turn.started_at,
        completed_at: turn.completed_at,
        updated_at: turn.updated_at,
        type: "tool_call",
        tool_name: "Read",
        arguments: { file_path: `/repo/history-${index + 1}.ts` },
      }),
    );
    const container = render(
      [
        {
          id: "msg-user-history-content-parts",
          role: "user",
          content: "检查 content parts",
          timestamp: new Date("2026-04-25T10:00:00.000Z"),
        } as Message,
        {
          id: "msg-assistant-history-content-parts",
          role: "assistant",
          content: "历史 content parts 正文",
          contentParts: [
            {
              type: "text",
              text: "历史 content parts 正文",
            },
            {
              type: "tool_use",
              toolCall: {
                id: "tool-history-content-parts",
                name: "Read",
                arguments: JSON.stringify({ file_path: "/repo/history.ts" }),
                status: "completed",
                result: { success: true, output: "ok" },
                startTime: new Date("2026-04-25T10:00:10.000Z"),
                endTime: new Date("2026-04-25T10:00:11.000Z"),
              },
            },
          ],
          timestamp: new Date("2026-04-25T10:01:00.000Z"),
        } as Message,
      ],
      {
        sessionId: "session-history-content-parts",
        currentTurnId: turn.id,
        turns: [turn],
        threadItems,
        sessionHistoryWindow: {
          loadedMessages: 2,
          totalMessages: 180,
          isLoadingFull: false,
          error: null,
        },
      },
    );

    expect(
      container.querySelector(
        '[data-testid="message-list-historical-markdown-preview"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="streaming-renderer"]'),
    ).toBeNull();
    expect(mockStreamingRenderer).not.toHaveBeenCalled();
    const markdownPreview = container.querySelector(
      '[data-testid="message-list-historical-markdown-preview"]',
    );
    const previewMarkdownRenderer = markdownPreview?.querySelector(
      '[data-testid="markdown-renderer"]',
    );
    expect(previewMarkdownRenderer?.getAttribute("data-render-mode")).toBe(
      "light",
    );
    expect(
      previewMarkdownRenderer?.getAttribute("data-render-a2ui-inline"),
    ).toBe("false");
    expect(markdownPreview?.textContent).toContain("历史 content parts 正文");
    const commit = getAgentUiPerformanceMetrics().find(
      (entry) => entry.phase === "messageList.commit",
    );
    expect(commit?.metrics).toEqual(
      expect.objectContaining({
        historicalContentPartsDeferredCount: 1,
        historicalMarkdownDeferredCount: 1,
        messageListComputeMs: expect.any(Number),
        messageListThreadItemsScanMs: expect.any(Number),
        messageListTimelineBuildMs: expect.any(Number),
        threadItemsScanDeferred: false,
      }),
    );

    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
  });

  it("旧会话助手 content 为空但 contentParts 有最终正文时首帧不应空白", () => {
    vi.useFakeTimers();
    const turn: AgentThreadTurn = {
      id: "turn-history-parts-only-body",
      thread_id: "thread-history-parts-only-body",
      prompt_text: "总结关键文件",
      status: "completed",
      started_at: "2026-04-25T10:00:00.000Z",
      completed_at: "2026-04-25T10:01:00.000Z",
      created_at: "2026-04-25T10:00:00.000Z",
      updated_at: "2026-04-25T10:01:00.000Z",
    };
    const threadItems: AgentThreadItem[] = Array.from(
      { length: 30 },
      (_, index): AgentThreadItem => ({
        id: `history-parts-only-tool-${index + 1}`,
        thread_id: turn.thread_id,
        turn_id: turn.id,
        sequence: index + 1,
        status: "completed",
        started_at: turn.started_at,
        completed_at: turn.completed_at,
        updated_at: turn.updated_at,
        type: "tool_call",
        tool_name: "Read",
        arguments: { file_path: `/repo/key-file-${index + 1}.md` },
      }),
    );
    const container = render(
      [
        {
          id: "msg-user-history-parts-only-body",
          role: "user",
          content: "总结关键文件",
          timestamp: new Date("2026-04-25T10:00:00.000Z"),
        } as Message,
        {
          id: "msg-assistant-history-parts-only-body",
          role: "assistant",
          content: "",
          contentParts: [
            {
              type: "text",
              text: "我先查看关键文件，再整理评分卡。",
            },
            {
              type: "tool_use",
              toolCall: {
                id: "tool-history-parts-only-read",
                name: "Read",
                arguments: JSON.stringify({
                  file_path: "/repo/scorecard.md",
                }),
                status: "completed",
                result: { success: true, output: "file contents" },
                startTime: new Date("2026-04-25T10:00:10.000Z"),
                endTime: new Date("2026-04-25T10:00:11.000Z"),
              },
            },
            {
              type: "text",
              text: "## 文件总结\n\n这是一份 Agent Workspace 工具 UI 评分卡。",
            },
          ],
          timestamp: new Date("2026-04-25T10:01:00.000Z"),
        } as Message,
      ],
      {
        sessionId: "session-history-parts-only-body",
        currentTurnId: turn.id,
        turns: [turn],
        threadItems,
        sessionHistoryWindow: {
          loadedMessages: 2,
          totalMessages: 180,
          isLoadingFull: false,
          error: null,
        },
      },
    );

    const markdownPreview = container.querySelector(
      '[data-testid="message-list-historical-markdown-preview"]',
    );

    expect(markdownPreview).not.toBeNull();
    expect(markdownPreview?.textContent).toContain("文件总结");
    expect(markdownPreview?.textContent).toContain(
      "这是一份 Agent Workspace 工具 UI 评分卡。",
    );
    expect(markdownPreview?.textContent).not.toContain("file contents");
    expect(
      container.querySelector('[data-testid="streaming-renderer"]'),
    ).toBeNull();
  });

  it("已分页旧会话的完成执行过程应先显示轻量摘要，点击后挂载真实 timeline", () => {
    const turn: AgentThreadTurn = {
      id: "turn-history-heavy",
      thread_id: "thread-history-heavy",
      prompt_text: "打开慢历史",
      status: "completed",
      started_at: "2026-04-25T10:00:00.000Z",
      completed_at: "2026-04-25T10:01:00.000Z",
      created_at: "2026-04-25T10:00:00.000Z",
      updated_at: "2026-04-25T10:01:00.000Z",
    };
    const threadItems: AgentThreadItem[] = Array.from(
      { length: 10 },
      (_, index) => ({
        id: `history-tool-${index + 1}`,
        thread_id: turn.thread_id,
        turn_id: turn.id,
        sequence: index + 1,
        status: "completed",
        started_at: "2026-04-25T10:00:00.000Z",
        completed_at: "2026-04-25T10:01:00.000Z",
        updated_at: "2026-04-25T10:01:00.000Z",
        type: "tool_call",
        tool_name: "Bash",
        arguments: { command: `echo ${index + 1}` },
        output: `输出 ${index + 1}`,
      }),
    );
    const commentary = "我先核对当前 tracker，再给出结论。";
    const finalAnswer = "这是旧会话的最终回复";
    const inlineToolParts = threadItems.map((item, index) => ({
      type: "tool_use" as const,
      metadata: { sequence: index + 2, turnId: turn.id },
      toolCall: {
        id: item.id,
        name: "Bash",
        arguments: JSON.stringify({ command: `echo ${index + 1}` }),
        status: "completed" as const,
        result: { success: true, output: `输出 ${index + 1}` },
        startTime: new Date("2026-04-25T10:00:00.000Z"),
        endTime: new Date("2026-04-25T10:01:00.000Z"),
      },
    }));
    const container = render(
      [
        {
          id: "msg-user-heavy-history",
          role: "user",
          content: "打开慢历史",
          timestamp: new Date("2026-04-25T10:00:00.000Z"),
        } as Message,
        {
          id: "msg-assistant-heavy-history",
          role: "assistant",
          content: finalAnswer,
          contentParts: [
            {
              type: "text",
              text: commentary,
              metadata: {
                phase: "commentary",
                sequence: 1,
                turnId: turn.id,
              },
            },
            ...inlineToolParts,
            {
              type: "text",
              text: finalAnswer,
              metadata: {
                phase: "final_answer",
                sequence: 12,
                turnId: turn.id,
              },
            },
          ],
          timestamp: new Date("2026-04-25T10:01:00.000Z"),
        } as Message,
      ],
      {
        turns: [turn],
        threadItems,
        currentTurnId: turn.id,
        sessionHistoryWindow: {
          loadedMessages: 2,
          totalMessages: 170,
          isLoadingFull: false,
          error: null,
        },
      },
    );

    expect(
      container.querySelector(
        '[data-testid="message-list-historical-timeline-preview:leading"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("Processed for 1m 0s");
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
    expect(mockStreamingRenderer).toHaveBeenLastCalledWith(
      expect.objectContaining({
        content: finalAnswer,
        contentParts: undefined,
        markdownRenderMode: "light",
      }),
    );

    const historicalSummary = container.querySelector(
      '[data-testid="message-list-historical-timeline-preview:leading"]',
    ) as HTMLElement | null;

    expect(historicalSummary?.tagName).toBe("DIV");
    expect(mockAgentThreadTimeline).not.toHaveBeenCalled();
    expect(mockStreamingRenderer).toHaveBeenLastCalledWith(
      expect.objectContaining({
        content: finalAnswer,
        markdownRenderMode: "light",
      }),
    );
  });
});
