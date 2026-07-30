import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  mockUseConfiguredProviders,
  mockFindConfiguredProviderBySelection,
  mockTokenUsageDisplay,
  mockStreamingRenderer,
  render,
  createConversationMessages,
  getAgentUiPerformanceMetrics,
} from "./MessageList.testHarness";
import type {
  Message,
  MockConfiguredProvider,
} from "./MessageList.testHarness";

describe("MessageList imported history", () => {
  it("旧会话里的超长历史助手消息应先渲染轻量预览，点击后再展开完整正文", () => {
    const longContent = `开头内容 ${"长历史 ".repeat(8000)} 末尾完整内容`;
    const container = render(
      [
        {
          id: "msg-user-long-history",
          role: "user",
          content: "打开超长历史",
          timestamp: new Date("2026-04-25T10:00:00.000Z"),
        } as Message,
        {
          id: "msg-assistant-long-history",
          role: "assistant",
          content: longContent,
          timestamp: new Date("2026-04-25T10:00:01.000Z"),
        } as Message,
      ],
      {
        sessionHistoryWindow: {
          loadedMessages: 2,
          totalMessages: 120,
          isLoadingFull: false,
          error: null,
        },
      },
    );

    const preview = container.querySelector(
      '[data-testid="message-list-long-history-preview"]',
    );

    expect(preview).not.toBeNull();
    expect(preview?.textContent).toContain("This history message is long");
    expect(preview?.textContent).toContain("plain-text preview");
    expect(preview?.textContent).not.toContain("末尾完整内容");
    expect(mockStreamingRenderer).not.toHaveBeenCalled();

    const expandButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("Show full content"),
    ) as HTMLButtonElement | undefined;

    act(() => {
      expandButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-long-history-preview"]',
      ),
    ).toBeNull();
    expect(mockStreamingRenderer).toHaveBeenLastCalledWith(
      expect.objectContaining({
        content: expect.stringContaining("末尾完整内容"),
        markdownRenderMode: "light",
      }),
    );
  });

  it("旧会话里的长助手回复应先用轻量 Markdown 预览，避免首帧挂载完整渲染器", () => {
    const oldAssistantContent = [
      "## BADOUCMS 架构分析",
      "",
      "| 发现 | 说明 |",
      "| --- | --- |",
      "| **底层框架** | `ThinkPHP` |",
      "",
      `旧回复开头 ${"历史分析 ".repeat(360)} 旧回复末尾完整内容`,
    ].join("\n");
    const latestAssistantContent = "最新回复保持完整";
    const container = render(
      [
        {
          id: "msg-user-old-compact",
          role: "user",
          content: "旧问题",
          timestamp: new Date("2026-04-25T10:00:00.000Z"),
        } as Message,
        {
          id: "msg-assistant-old-compact",
          role: "assistant",
          content: oldAssistantContent,
          timestamp: new Date("2026-04-25T10:00:01.000Z"),
        } as Message,
        {
          id: "msg-user-latest-compact",
          role: "user",
          content: "最新问题",
          timestamp: new Date("2026-04-25T10:01:00.000Z"),
        } as Message,
        {
          id: "msg-assistant-latest-compact",
          role: "assistant",
          content: latestAssistantContent,
          timestamp: new Date("2026-04-25T10:01:01.000Z"),
        } as Message,
      ],
      {
        sessionHistoryWindow: {
          loadedMessages: 4,
          totalMessages: 88,
          isLoadingFull: false,
          error: null,
        },
      },
    );

    const preview = container.querySelector(
      '[data-testid="message-list-historical-assistant-preview"]',
    );

    expect(preview).not.toBeNull();
    const previewMarkdown = preview?.querySelector(
      '[data-testid="markdown-renderer"]',
    );
    expect(previewMarkdown?.getAttribute("data-render-mode")).toBe("light");
    expect(previewMarkdown?.getAttribute("data-render-a2ui-inline")).toBe(
      "false",
    );
    expect(previewMarkdown?.getAttribute("data-read-only-a2ui")).toBe("yes");
    expect(previewMarkdown?.textContent).toContain("## BADOUCMS 架构分析");
    expect(preview?.textContent).toContain("This assistant reply is long");
    expect(preview?.textContent).not.toContain("旧回复末尾完整内容");
    expect(container.textContent).toContain(latestAssistantContent);
    expect(mockStreamingRenderer).not.toHaveBeenCalledWith(
      expect.objectContaining({
        content: expect.stringContaining("旧回复末尾完整内容"),
      }),
    );

    const expandButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("Show full content"),
    ) as HTMLButtonElement | undefined;

    act(() => {
      expandButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(
      container.querySelector(
        '[data-testid="message-list-historical-assistant-preview"]',
      ),
    ).toBeNull();
    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        content: expect.stringContaining("旧回复末尾完整内容"),
        markdownRenderMode: "light",
      }),
    );
  });

  it("非旧会话助手正文应保持标准 Markdown 渲染模式", () => {
    render([
      {
        id: "msg-user-live-standard-markdown",
        role: "user",
        content: "实时对话",
        timestamp: new Date("2026-04-25T10:00:00.000Z"),
      } as Message,
      {
        id: "msg-assistant-live-standard-markdown",
        role: "assistant",
        content: "```ts\nconsole.log('live')\n```",
        timestamp: new Date("2026-04-25T10:00:01.000Z"),
      } as Message,
    ]);

    expect(mockStreamingRenderer).toHaveBeenLastCalledWith(
      expect.objectContaining({
        markdownRenderMode: "standard",
      }),
    );
  });

  it("任务中心空列表时应展示最近对话空态而不是普通新对话文案", () => {
    const container = render([], {
      emptyStateVariant: "task-center",
    });

    expect(
      container.querySelector('[data-testid="message-list-empty-task-center"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="message-list-empty-task-center"]')
        ?.className,
    ).toContain("max-w-[560px]");
    expect(
      container.querySelector('[data-testid="message-list-empty-task-center"]')
        ?.className,
    ).not.toContain("rounded-[30px]");
    expect(container.textContent).toContain("Chat");
    expect(container.textContent).toContain("Recent chats");
    expect(container.textContent).toContain(
      "Recent chats, sessions to continue, and earlier archives are gathered here so you can return to the last working context.",
    );
    expect(container.textContent).toContain(
      "When there are no chats yet, start from “New chat”. Results, materials, and intermediate steps will stay here later.",
    );
    expect(container.textContent).toContain(
      "Chats to continue appear on the left first",
    );
    expect(container.textContent).toContain(
      "Recent chats and archives are organized by time",
    );
    expect(container.textContent).toContain(
      "Restoring sessions return here automatically",
    );
    expect(container.textContent).not.toContain("Start a new conversation");
  });

  it("应过滤空白 user 消息，避免渲染空白气泡", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-user-empty",
        role: "user",
        content: "",
        timestamp: now,
      },
      {
        id: "msg-user-text",
        role: "user",
        content: "请继续生成",
        timestamp: now,
      },
      {
        id: "msg-assistant",
        role: "assistant",
        content: "好的，我继续处理。",
        timestamp: now,
      },
    ];

    const container = render(messages);

    const markdownTexts = Array.from(
      container.querySelectorAll('[data-testid="markdown-renderer"]'),
    ).map((node) => node.textContent);
    expect(markdownTexts).toEqual(["请继续生成"]);

    const streamingTexts = Array.from(
      container.querySelectorAll('[data-testid="streaming-renderer"]'),
    ).map((node) => node.textContent);
    expect(streamingTexts).toEqual(["好的，我继续处理。"]);
  });

  it("大历史会话应先展示最近消息，并允许用户立即展开更早内容", () => {
    const messages = createConversationMessages(90);
    const container = render(messages);

    const historyWindow = container.querySelector(
      '[data-testid="message-list-history-window"]',
    );
    const expandButton = container.querySelector(
      '[data-testid="message-list-expand-history"]',
    ) as HTMLButtonElement | null;

    expect(historyWindow).not.toBeNull();
    expect(container.textContent).toContain("To open the chat faster");
    expect(container.textContent).toContain(
      "latest 36 conversation entries are shown first",
    );
    expect(container.textContent).toContain("9 earlier entries");
    expect(container.textContent).toContain("消息 90");
    expect(container.textContent).not.toContain("消息 1Saturday");
    expect(expandButton).not.toBeNull();

    act(() => {
      expandButton?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(
      container.querySelector('[data-testid="message-list-history-window"]'),
    ).toBeNull();
    expect(container.textContent).toContain("消息 1Saturday");
  });

  it("user peer 包络正文应直接渲染为专门协作卡片", () => {
    const container = render([
      {
        id: "msg-user-peer",
        role: "user",
        content: `<teammate-message teammate_id="researcher" summary="同步结果">
继续验证
</teammate-message>`,
      } as Message,
    ]);

    expect(
      container.querySelector('[data-testid="runtime-peer-message-cards"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("协作者消息");
    expect(container.textContent).toContain("来自 researcher");
    expect(container.textContent).toContain("同步结果");
    expect(container.textContent).toContain("继续验证");
    expect(container.textContent).not.toContain("teammate-message");
  });

  it("应向助手消息透传内联 A2UI 开关", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant",
        role: "assistant",
        content: "```a2ui\n{}\n```",
        timestamp: now,
      },
    ];

    render(messages);
    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({ renderA2UIInline: true }),
    );

    render(messages, { renderA2UIInline: false });
    expect(mockStreamingRenderer).toHaveBeenLastCalledWith(
      expect.objectContaining({ renderA2UIInline: false }),
    );
  });

  it("assistant 消息带 contextTrace 时不应在聊天主线渲染上下文轨迹块", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-context-trace",
        role: "assistant",
        content: "我已经处理完成。",
        timestamp: now,
        contextTrace: [
          {
            stage: "memory_injection",
            detail: "query_len=8,injected=2",
          },
        ],
      },
    ];

    const container = render(messages);

    expect(container.textContent).toContain("我已经处理完成。");
    expect(container.textContent).not.toContain("上下文轨迹");
    expect(container.textContent).not.toContain("memory_injection");
    expect(container.textContent).not.toContain("query_len=8,injected=2");
  });

  it("anthropic-compatible 自定义 Provider 无缓存命中时应透传自动缓存提示", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-usage",
        role: "assistant",
        content: "本轮已完成。",
        timestamp: now,
        usage: {
          input_tokens: 1_500,
          output_tokens: 500,
          cached_input_tokens: 0,
        },
      },
    ];

    mockUseConfiguredProviders.mockImplementation(() => ({
      providers: [
        {
          key: "custom-provider-id",
          label: "GLM Anthropic",
          registryId: "custom-provider-id",
          type: "anthropic-compatible",
          providerId: "custom-provider-id",
        },
      ],
      loading: false,
    }));
    mockFindConfiguredProviderBySelection.mockImplementation(
      (
        providers: MockConfiguredProvider[],
        selection?: string | null,
      ): MockConfiguredProvider | null =>
        Array.isArray(providers)
          ? (providers.find((provider) => provider.key === selection) ?? null)
          : null,
    );

    const container = render(messages, {
      providerType: "custom-provider-id",
    });

    expect(container.textContent).toContain("未声明自动缓存");
    expect(mockTokenUsageDisplay).toHaveBeenCalledWith(
      expect.objectContaining({
        promptCacheNotice: expect.objectContaining({
          label: "未声明自动缓存",
        }),
      }),
    );
  });

  it("anthropic-compatible 自定义 Provider 存在缓存写入时不应再透传自动缓存提示", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-cache-write",
        role: "assistant",
        content: "本轮已完成。",
        timestamp: now,
        usage: {
          input_tokens: 1_500,
          output_tokens: 500,
          cached_input_tokens: 0,
          cache_creation_input_tokens: 256,
        },
      },
    ];

    mockUseConfiguredProviders.mockImplementation(() => ({
      providers: [
        {
          key: "custom-provider-id",
          label: "Kimi Anthropic",
          registryId: "custom-provider-id",
          type: "anthropic-compatible",
          providerId: "custom-provider-id",
        },
      ],
      loading: false,
    }));
    mockFindConfiguredProviderBySelection.mockImplementation(
      (
        providers: MockConfiguredProvider[],
        selection?: string | null,
      ): MockConfiguredProvider | null =>
        Array.isArray(providers)
          ? (providers.find((provider) => provider.key === selection) ?? null)
          : null,
    );

    const container = render(messages, {
      providerType: "custom-provider-id",
    });

    expect(container.textContent).not.toContain("未声明自动缓存");
    expect(mockTokenUsageDisplay).toHaveBeenCalledWith(
      expect.objectContaining({
        promptCacheNotice: undefined,
      }),
    );
  });

  it("旧会话恢复首帧不应立即自动加载 Provider 缓存提示配置", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-restored-usage",
        role: "assistant",
        content: "旧会话结果。",
        timestamp: now,
        usage: {
          input_tokens: 1_200,
          output_tokens: 300,
          cached_input_tokens: 0,
        },
      },
    ];

    render(messages, {
      providerType: "custom-provider-id",
      sessionHistoryWindow: {
        loadedMessages: 40,
        totalMessages: 320,
        isLoadingFull: false,
        error: null,
      },
    });

    expect(mockUseConfiguredProviders).toHaveBeenCalledWith({
      autoLoad: false,
    });
  });

  it("旧会话首帧应记录可汇总的渲染采样数值", async () => {
    const messages = createConversationMessages(32);

    render(messages, {
      sessionId: "session-metrics",
      sessionHistoryWindow: {
        loadedMessages: 32,
        totalMessages: 160,
        isLoadingFull: false,
        error: null,
      },
    });

    await act(async () => {
      await Promise.resolve();
    });

    const commit = getAgentUiPerformanceMetrics().find(
      (entry) => entry.phase === "messageList.commit",
    );
    expect(commit).toEqual(
      expect.objectContaining({
        sessionId: "session-metrics",
        metrics: expect.objectContaining({
          hiddenHistoryCount: expect.any(Number),
          messagesCount: 32,
          persistedHiddenHistoryCount: 1,
          renderedMessagesCount: expect.any(Number),
        }),
      }),
    );
  });
});
