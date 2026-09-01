import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  mockStreamingRenderer,
  render,
  renderZh,
  setScrollMetrics,
  createConversationMessages,
  upsertAgentStreamTextOverlay,
  createRoot,
  MessageList,
  mountedRoots,
} from "./MessageList.testHarness";
import type { Message } from "./MessageList.testHarness";

describe("MessageList layout and scrolling", () => {
  it("消息列表应暴露当前 canonical session identity", () => {
    const container = render([], { sessionId: "session-canonical-1" });

    expect(
      container
        .querySelector('[data-testid="message-list-frame"]')
        ?.getAttribute("data-session-id"),
    ).toBe("session-canonical-1");
  });

  it("Thread 消息列应使用单一内容轴和紧凑密度", () => {
    const container = render([
      {
        id: "msg-thread-axis",
        role: "assistant",
        content: "保持正文与过程时间线在同一列。",
      } as Message,
    ]);

    const messageColumn = container.querySelector<HTMLElement>(
      '[data-testid="message-list-column"]',
    );
    const assistantBubble = container.querySelector<HTMLElement>(
      '[data-message-role="assistant"]',
    );

    expect(messageColumn?.dataset.contentAxis).toBe("thread");
    expect(messageColumn?.dataset.layoutDensity).toBe("compact");
    expect(messageColumn?.className).toContain("gap-3");
    expect(messageColumn?.className).toContain("py-3");
    expect(assistantBubble).not.toBeNull();
    expect(getComputedStyle(assistantBubble!).maxWidth).toContain(
      "clamp(640px, 100%, 720px)",
    );
  });

  it("应在同一滚动区域顶部渲染 leadingContent", () => {
    const container = render(
      [
        {
          id: "assistant-1",
          role: "assistant",
          content: "第一条消息",
        } as Message,
      ],
      {
        leadingContent: (
          <div data-testid="leading-probe">scene summary heading</div>
        ),
      },
    );

    const leadingContent = container.querySelector(
      '[data-testid="message-list-leading-content"]',
    );
    const messageColumn = container.querySelector(
      '[data-testid="message-list-column"]',
    );

    expect(leadingContent?.textContent).toContain("scene summary heading");
    expect(messageColumn?.firstElementChild).toBe(leadingContent);
    expect(messageColumn?.textContent).toContain("第一条消息");
    expect(messageColumn?.className).toContain("justify-start");
  });

  it("应在同一滚动区域尾部渲染 trailingContent", () => {
    const container = render(
      [
        {
          id: "assistant-1",
          role: "assistant",
          content: "第一条消息",
        } as Message,
      ],
      {
        trailingContent: (
          <div data-testid="trailing-probe">inline a2ui card</div>
        ),
      },
    );

    const trailingContent = container.querySelector(
      '[data-testid="message-list-trailing-content"]',
    );
    const messageColumn = container.querySelector(
      '[data-testid="message-list-column"]',
    );

    expect(trailingContent?.textContent).toContain("inline a2ui card");
    expect(messageColumn?.lastElementChild?.previousElementSibling).toBe(
      trailingContent,
    );
  });

  it("短对话发送首帧也应吸顶展示，避免完成前后跳动", () => {
    const container = render(
      [
        {
          id: "msg-user-first-frame",
          role: "user",
          content: "你好",
          timestamp: new Date("2026-04-25T10:00:00.000Z"),
        } as Message,
      ],
      { isSending: true },
    );

    const messageColumn = container.querySelector(
      '[data-testid="message-list-column"]',
    );

    expect(messageColumn?.textContent).toContain("你好");
    expect(messageColumn?.className).toContain("min-h-full");
    expect(messageColumn?.className).toContain("justify-start");
    expect(messageColumn?.className).not.toContain("justify-end");
  });

  it("长消息流应让尾部 turn 保持即时渲染并隔离离屏历史工作", () => {
    const container = render(createConversationMessages(8));
    const groups = Array.from(
      container.querySelectorAll<HTMLElement>(
        '[data-testid="message-turn-group"]',
      ),
    );

    expect(groups).toHaveLength(4);
    expect(groups.map((group) => group.dataset.renderPriority)).toEqual([
      "offscreen-deferred",
      "offscreen-deferred",
      "tail",
      "tail",
    ]);
    expect(document.head.textContent).toContain("content-visibility:auto");
    expect(document.head.textContent).toContain(
      "contain-intrinsic-size:auto 320px",
    );
  });

  it("对话列表顶部应保留工作台呼吸区，避免首条消息贴近窗口顶栏", () => {
    const container = render([
      {
        id: "msg-top-spacing",
        role: "user",
        content: "整理一下今天的国际新闻",
        timestamp: new Date("2026-06-01T02:57:00.000Z"),
      } as Message,
    ]);

    const scrollContainer = container.querySelector<HTMLElement>(
      '[data-testid="message-list-scroll-container"]',
    );

    expect(scrollContainer).not.toBeNull();
    expect(scrollContainer?.className).toMatch(/sc-/);
    expect(document.head.textContent).toContain("padding:22px 0 16px");
  });

  it("用户消息应使用紧凑中性气泡，并把时间与操作区放在气泡外", async () => {
    const onEditMessage = vi.fn();
    const container = await renderZh(
      [
        {
          id: "msg-user-visual-footer",
          role: "user",
          content: "帮我整理一下今天的国际新闻",
          timestamp: new Date("2026-06-01T02:57:00.000Z"),
        } as Message,
      ],
      { onEditMessage },
    );

    const userBubble = container.querySelector<HTMLElement>(
      '[data-message-role="user"][data-visual-tone="neutral-user"]',
    );
    const footer = container.querySelector<HTMLElement>(
      '[data-testid="user-message-footer"]',
    );
    const timestamp = container.querySelector<HTMLElement>(
      '[data-testid="user-message-timestamp"]',
    );
    const editButton = container.querySelector<HTMLButtonElement>(
      'button[aria-label="编辑消息"]',
    );

    expect(userBubble).not.toBeNull();
    expect(userBubble?.textContent).toContain("帮我整理一下今天的国际新闻");
    expect(userBubble?.contains(footer)).toBe(false);
    expect(footer).not.toBeNull();
    expect(footer?.className).toContain("user-message-footer");
    expect(timestamp?.textContent?.trim()).toBeTruthy();
    expect(
      container.querySelector('button[aria-label="复制消息"]'),
    ).not.toBeNull();
    expect(editButton).not.toBeNull();

    act(() => {
      editButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(onEditMessage).toHaveBeenCalledWith(
      "msg-user-visual-footer",
      "帮我整理一下今天的国际新闻",
    );
  });

  it("消息气泡和 turn group 应暴露 runtimeTurnId，供 GUI fixture 按轮验收", () => {
    const container = render(
      [
        {
          id: "user-turn-scope",
          role: "user",
          content: "第一轮问题",
          timestamp: new Date("2026-06-01T02:57:00.000Z"),
          runtimeTurnId: "turn-scope-1",
        } as Message,
        {
          id: "assistant-turn-scope",
          role: "assistant",
          content: "第一轮回答",
          timestamp: new Date("2026-06-01T02:57:01.000Z"),
          runtimeTurnId: "turn-scope-1",
          contentParts: [
            {
              type: "text",
              text: "第一轮回答",
              metadata: {
                source: "agent_text_delta",
                itemId: "item-agent-message-turn-scope",
                threadItemId: "item-agent-message-turn-scope",
                phase: "final_answer",
              },
            },
          ],
        } as Message,
      ],
      {
        threadRead: {
          thread_id: "thread-scope",
          status: "completed",
          thread_items: [
            {
              id: "item-user-message-turn-scope",
              type: "user_message",
              status: "completed",
              turn_id: "turn-scope-1",
            },
            {
              id: "item-agent-message-turn-scope",
              type: "agent_message",
              status: "completed",
              turn_id: "turn-scope-1",
            },
          ],
        },
        turns: [
          {
            id: "turn-scope-1",
            thread_id: "thread-scope",
            prompt_text: "第一轮问题",
            status: "completed",
            started_at: "2026-06-01T02:57:00.000Z",
            completed_at: "2026-06-01T02:57:01.000Z",
            created_at: "2026-06-01T02:57:00.000Z",
            updated_at: "2026-06-01T02:57:01.000Z",
          },
        ],
      },
    );

    const group = container.querySelector<HTMLElement>(
      '[data-testid="message-turn-group"]',
    );
    const assistantBubble = container.querySelector<HTMLElement>(
      '[data-message-id="assistant-turn-scope"]',
    );
    const userBubble = container.querySelector<HTMLElement>(
      '[data-message-id="user-turn-scope"]',
    );

    expect(group?.getAttribute("data-runtime-turn-id")).toBe("turn-scope-1");
    expect(group?.getAttribute("data-runtime-turn-status")).toBe("completed");
    expect(group?.getAttribute("data-last-assistant-message-id")).toBe(
      "assistant-turn-scope",
    );
    expect(assistantBubble?.getAttribute("data-runtime-turn-id")).toBe(
      "turn-scope-1",
    );
    expect(assistantBubble?.getAttribute("data-thread-item-id")).toBe(
      "item-agent-message-turn-scope",
    );
    expect(userBubble?.getAttribute("data-thread-item-id")).toBe(
      "item-user-message-turn-scope",
    );
  });

  it("执行中新消息应贴底自动跟随且不显示查看最新入口", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    mountedRoots.push({ container: host, root });
    const originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
    const originalRequestAnimationFrame = window.requestAnimationFrame;
    const scrollIntoViewMock = vi.fn();
    const initialMessages = createConversationMessages(6);
    const nextMessages: Message[] = [
      ...initialMessages,
      {
        id: "assistant-live-tail",
        role: "assistant",
        content: "最新输出还在继续",
        timestamp: new Date("2026-04-25T10:06:00.000Z"),
      },
    ];

    HTMLElement.prototype.scrollIntoView = scrollIntoViewMock;
    window.requestAnimationFrame = ((callback: (timestamp: number) => void) => {
      callback(0);
      return 1;
    }) as typeof window.requestAnimationFrame;

    try {
      act(() => {
        root.render(<MessageList messages={initialMessages} isSending />);
      });
      scrollIntoViewMock.mockClear();

      const scrollContainer = host.querySelector<HTMLElement>(
        '[data-testid="message-list-scroll-container"]',
      );
      expect(scrollContainer).not.toBeNull();
      expect(
        host.querySelector('[data-testid="message-list-jump-to-latest"]'),
      ).toBeNull();
      setScrollMetrics(scrollContainer as HTMLElement, {
        scrollTop: 536,
        scrollHeight: 1000,
        clientHeight: 400,
      });

      act(() => {
        root.render(<MessageList messages={nextMessages} isSending />);
      });

      expect(scrollIntoViewMock).toHaveBeenCalledWith({
        behavior: "smooth",
        block: "end",
      });
      expect(
        host.querySelector('[data-testid="message-list-jump-to-latest"]'),
      ).toBeNull();
    } finally {
      HTMLElement.prototype.scrollIntoView = originalScrollIntoView;
      window.requestAnimationFrame = originalRequestAnimationFrame;
    }
  });

  it("执行中流式追加同一条消息时也应贴底自动跟随", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    mountedRoots.push({ container: host, root });
    const originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
    const originalRequestAnimationFrame = window.requestAnimationFrame;
    const scrollIntoViewMock = vi.fn();
    const messages: Message[] = [
      {
        id: "user-stream-follow",
        role: "user",
        content: "继续输出",
        timestamp: new Date("2026-04-25T10:00:00.000Z"),
      },
      {
        id: "assistant-stream-follow",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-25T10:00:01.000Z"),
      },
    ];

    HTMLElement.prototype.scrollIntoView = scrollIntoViewMock;
    window.requestAnimationFrame = ((callback: (timestamp: number) => void) => {
      callback(0);
      return 1;
    }) as typeof window.requestAnimationFrame;

    try {
      act(() => {
        root.render(<MessageList messages={messages} isSending />);
      });
      scrollIntoViewMock.mockClear();

      const scrollContainer = host.querySelector<HTMLElement>(
        '[data-testid="message-list-scroll-container"]',
      );
      expect(scrollContainer).not.toBeNull();
      setScrollMetrics(scrollContainer as HTMLElement, {
        scrollTop: 0,
        scrollHeight: 1200,
        clientHeight: 400,
      });

      act(() => {
        upsertAgentStreamTextOverlay({
          messageId: "assistant-stream-follow",
          eventName: "response.output_text.delta",
          content: "第一段流式内容",
          updatedAt: 1,
        });
      });

      expect(scrollIntoViewMock).toHaveBeenCalledWith({
        behavior: "auto",
        block: "end",
      });
      expect(
        host.querySelector('[data-testid="message-list-jump-to-latest"]'),
      ).toBeNull();
    } finally {
      HTMLElement.prototype.scrollIntoView = originalScrollIntoView;
      window.requestAnimationFrame = originalRequestAnimationFrame;
    }
  });

  it("窗口 resize/reflow 时若仍在底部跟随，应重新贴到底部锚点", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    mountedRoots.push({ container: host, root });
    const originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
    const originalScrollTo = HTMLElement.prototype.scrollTo;
    const originalRequestAnimationFrame = window.requestAnimationFrame;
    const originalCancelAnimationFrame = window.cancelAnimationFrame;
    const originalResizeObserver = globalThis.ResizeObserver;
    const scrollIntoViewMock = vi.fn();
    const scrollToMock = vi.fn();
    const observeMock = vi.fn();
    type TestResizeObserverCallback = (
      entries: unknown[],
      observer: MockResizeObserver,
    ) => void;
    let resizeCallback: TestResizeObserverCallback | null = null;

    class MockResizeObserver {
      constructor(callback: TestResizeObserverCallback) {
        resizeCallback = callback;
      }

      observe = observeMock;
      disconnect = vi.fn();
    }

    HTMLElement.prototype.scrollIntoView = scrollIntoViewMock;
    HTMLElement.prototype.scrollTo = scrollToMock;
    window.requestAnimationFrame = ((callback: (timestamp: number) => void) => {
      callback(0);
      return 1;
    }) as typeof window.requestAnimationFrame;
    window.cancelAnimationFrame = vi.fn() as typeof window.cancelAnimationFrame;
    globalThis.ResizeObserver =
      MockResizeObserver as unknown as typeof ResizeObserver;

    try {
      act(() => {
        root.render(
          <MessageList messages={createConversationMessages(8)} isSending />,
        );
      });
      scrollIntoViewMock.mockClear();

      const scrollContainer = host.querySelector<HTMLElement>(
        '[data-testid="message-list-scroll-container"]',
      );
      const messageColumn = host.querySelector<HTMLElement>(
        '[data-testid="message-list-column"]',
      );
      expect(scrollContainer).not.toBeNull();
      expect(messageColumn).not.toBeNull();
      expect(observeMock.mock.calls.map(([target]) => target)).toEqual([
        scrollContainer,
        messageColumn,
      ]);
      setScrollMetrics(scrollContainer as HTMLElement, {
        scrollTop: 600,
        scrollHeight: 1000,
        clientHeight: 400,
      });

      act(() => {
        resizeCallback?.([], {} as MockResizeObserver);
      });

      expect(scrollToMock).toHaveBeenCalledWith({
        behavior: "auto",
        top: 1000,
      });

      scrollToMock.mockClear();
      act(() => {
        window.dispatchEvent(new Event("resize"));
      });
      expect(scrollToMock).toHaveBeenCalledWith({
        behavior: "auto",
        top: 1000,
      });

      scrollIntoViewMock.mockClear();
      scrollToMock.mockClear();
      setScrollMetrics(scrollContainer as HTMLElement, {
        scrollTop: 400,
        scrollHeight: 1000,
        clientHeight: 400,
      });
      act(() => {
        scrollContainer?.dispatchEvent(
          new WheelEvent("wheel", { bubbles: true, deltaY: -120 }),
        );
        scrollContainer?.dispatchEvent(new Event("scroll", { bubbles: true }));
        resizeCallback?.([], {} as MockResizeObserver);
        window.dispatchEvent(new Event("resize"));
      });
      expect(scrollIntoViewMock).not.toHaveBeenCalled();
      expect(scrollToMock).not.toHaveBeenCalled();
    } finally {
      HTMLElement.prototype.scrollIntoView = originalScrollIntoView;
      HTMLElement.prototype.scrollTo = originalScrollTo;
      window.requestAnimationFrame = originalRequestAnimationFrame;
      window.cancelAnimationFrame = originalCancelAnimationFrame;
      globalThis.ResizeObserver = originalResizeObserver;
    }
  });

  it("用户上拉阅读时流式追加不应强制滚回底部", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    mountedRoots.push({ container: host, root });
    const originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
    const originalRequestAnimationFrame = window.requestAnimationFrame;
    const scrollIntoViewMock = vi.fn();
    const messages: Message[] = [
      {
        id: "user-stream-scroll-away",
        role: "user",
        content: "继续输出",
        timestamp: new Date("2026-04-25T10:00:00.000Z"),
      },
      {
        id: "assistant-stream-scroll-away",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-25T10:00:01.000Z"),
      },
    ];

    HTMLElement.prototype.scrollIntoView = scrollIntoViewMock;
    window.requestAnimationFrame = ((callback: (timestamp: number) => void) => {
      callback(0);
      return 1;
    }) as typeof window.requestAnimationFrame;

    try {
      act(() => {
        root.render(<MessageList messages={messages} isSending />);
      });

      const scrollContainer = host.querySelector<HTMLElement>(
        '[data-testid="message-list-scroll-container"]',
      );
      expect(scrollContainer).not.toBeNull();
      setScrollMetrics(scrollContainer as HTMLElement, {
        scrollTop: 780,
        scrollHeight: 1200,
        clientHeight: 400,
      });
      scrollIntoViewMock.mockClear();

      act(() => {
        scrollContainer?.dispatchEvent(
          new WheelEvent("wheel", { bubbles: true, deltaY: -120 }),
        );
        upsertAgentStreamTextOverlay({
          messageId: "assistant-stream-scroll-away",
          eventName: "response.output_text.delta",
          content: "新的流式内容到达，但用户正在上拉阅读",
          updatedAt: 1,
        });
      });

      expect(scrollIntoViewMock).not.toHaveBeenCalled();
    } finally {
      HTMLElement.prototype.scrollIntoView = originalScrollIntoView;
      window.requestAnimationFrame = originalRequestAnimationFrame;
    }
  });

  it("流式正文 overlay 应优先渲染并替代首 token 占位", () => {
    upsertAgentStreamTextOverlay({
      messageId: "assistant-overlay",
      eventName: "agent-runtime-overlay-test",
      content: "overlay 正文已经可见",
      phase: "final_answer",
    });

    const container = render(
      [
        {
          id: "assistant-overlay",
          role: "assistant",
          content: "",
          isThinking: true,
          timestamp: new Date("2026-05-09T10:00:00.000Z"),
          runtimeTurnId: "turn-overlay",
          contentParts: [
            {
              type: "tool_use",
              toolCall: {
                id: "tool-1",
                name: "read_file",
                status: "running",
                startTime: new Date("2026-05-09T10:00:00.000Z"),
              },
            },
          ],
        } as Message,
      ],
      {
        currentTurnId: "turn-overlay",
        isSending: true,
        turns: [
          {
            id: "turn-overlay",
            thread_id: "thread-overlay",
            prompt_text: "overlay",
            status: "running",
            started_at: "2026-05-09T10:00:00.000Z",
            created_at: "2026-05-09T10:00:00.000Z",
            updated_at: "2026-05-09T10:00:00.000Z",
          },
        ],
      },
    );

    expect(container.textContent).not.toContain("正在准备回复");
    expect(
      mockStreamingRenderer.mock.calls.find(
        ([props]) =>
          (props as { content?: string }).content === "overlay 正文已经可见",
      )?.[0],
    ).toEqual(
      expect.objectContaining({
        content: "overlay 正文已经可见",
        contentParts: expect.arrayContaining([
          expect.objectContaining({
            type: "text",
            text: "overlay 正文已经可见",
          }),
        ]),
        isStreaming: true,
      }),
    );
  });

  it("流式正文 overlay 不应刷掉当前 thinking 过程", () => {
    upsertAgentStreamTextOverlay({
      messageId: "assistant-overlay-thinking",
      eventName: "agent-runtime-overlay-thinking-test",
      content: "正文已经开始输出。",
    });

    render([
      {
        id: "assistant-overlay-thinking",
        role: "assistant",
        content: "",
        isThinking: true,
        thinkingContent: "先检查目录结构，再输出结论。",
        timestamp: new Date("2026-05-30T09:00:00.000Z"),
      } as Message,
    ]);

    expect(mockStreamingRenderer).toHaveBeenLastCalledWith(
      expect.objectContaining({
        content: "正文已经开始输出。",
        thinkingContent: "先检查目录结构，再输出结论。",
        contentParts: [
          { type: "thinking", text: "先检查目录结构，再输出结论。" },
          { type: "text", text: "正文已经开始输出。" },
        ],
        isStreaming: true,
      }),
    );
  });

  it("任务中心发送首帧也应吸顶展示", () => {
    const container = render(
      [
        {
          id: "msg-user-task-center-first-frame",
          role: "user",
          content: "从任务中心开始对话",
          timestamp: new Date("2026-04-25T10:00:00.000Z"),
        } as Message,
      ],
      {
        emptyStateVariant: "task-center",
        isSending: true,
      },
    );

    const messageColumn = container.querySelector(
      '[data-testid="message-list-column"]',
    );

    expect(messageColumn?.textContent).toContain("从任务中心开始对话");
    expect(messageColumn?.className).toContain("justify-start");
    expect(messageColumn?.className).not.toContain("justify-end");
  });

  it("已完成的旧会话短消息应吸顶展示，避免打开历史时贴近输入区", () => {
    const container = render(
      [
        {
          id: "msg-user-history-short",
          role: "user",
          content: "打开旧会话",
          timestamp: new Date("2026-04-25T10:00:00.000Z"),
        } as Message,
        {
          id: "msg-assistant-history-short",
          role: "assistant",
          content: "这是历史回复",
          timestamp: new Date("2026-04-25T10:00:01.000Z"),
        } as Message,
      ],
      {
        currentTurnId: "turn-history-completed",
        turns: [
          {
            id: "turn-history-completed",
            thread_id: "session-history-short",
            prompt_text: "打开旧会话",
            status: "completed",
            started_at: "2026-04-25T10:00:00.000Z",
            completed_at: "2026-04-25T10:00:01.000Z",
            created_at: "2026-04-25T10:00:00.000Z",
            updated_at: "2026-04-25T10:00:01.000Z",
          },
        ],
      },
    );

    const messageColumn = container.querySelector(
      '[data-testid="message-list-column"]',
    );

    expect(messageColumn?.textContent).toContain("打开旧会话");
    expect(messageColumn?.className).toContain("justify-start");
    expect(messageColumn?.className).not.toContain("justify-end");
  });

  it("空消息列表默认不再渲染旧新对话空态，由工作区首页接管", () => {
    const container = render([]);

    expect(
      container.querySelector('[data-testid="message-list-empty-default"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="message-list-empty-task-center"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="message-list-column"]'),
    ).not.toBeNull();
    expect(container.textContent).not.toContain("Start a new conversation");
    expect(container.textContent).not.toContain("The composer is ready");
  });

  it("任务中心空消息列表仍应显式展示最近对话空态", () => {
    const container = render([], { emptyStateVariant: "task-center" });

    expect(
      container.querySelector('[data-testid="message-list-empty-task-center"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("Recent chats");
    expect(container.textContent).toContain(
      "Recent chats, sessions to continue, and earlier archives",
    );
  });

  it("自动恢复生成会话时应展示恢复占位而不是空白引导", () => {
    const container = render([], { isRestoringSession: true });

    expect(
      container.querySelector('[data-testid="message-list-restoring-session"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("Restoring generation session...");
    expect(container.textContent).toContain(
      "Syncing the latest generation session. Please wait.",
    );
    expect(container.textContent).not.toContain("Start a new conversation");
  });
});
