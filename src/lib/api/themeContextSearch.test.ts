import { describe, expect, it, vi } from "vitest";
import { APP_SERVER_METHOD_AGENT_SESSION_EVENT } from "@/lib/api/appServer";
import { searchThemeContextWithAppServer } from "./themeContextSearch";

function appServerClientMock() {
  const client = {
    startSession: vi.fn(),
    startTurn: vi.fn(),
    readThread: vi.fn(),
  };

  client.startSession.mockResolvedValue({
    id: 1,
    result: {
      approvalPolicy: null,
      approvalsReviewer: null,
      cwd: "/tmp/workspace-1",
      model: "gpt-4.1-mini",
      modelProvider: "openai-compatible",
      sandbox: null,
      thread: {
        cliVersion: "0.1.0",
        cwd: "/tmp/workspace-1",
        ephemeral: false,
        id: "thread-theme-context-search",
        modelProvider: "openai-compatible",
        preview: "上下文搜索",
        sessionId: "__lime_theme_context_search__-session",
        source: "appServer",
        status: { type: "idle" },
        createdAt: Date.parse("2026-06-08T00:00:00.000Z") / 1000,
        updatedAt: Date.parse("2026-06-08T00:00:00.000Z") / 1000,
      },
    },
    response: { id: 1, result: {} },
    notifications: [],
    messages: [],
  });

  client.startTurn.mockImplementation(async () => ({
    id: 2,
    result: {
      turn: {
        id: "turn-theme-context-search",
        status: "completed",
        startedAt: Date.parse("2026-06-08T00:00:01.000Z") / 1000,
        completedAt: Date.parse("2026-06-08T00:00:02.000Z") / 1000,
      },
    },
    response: { id: 2, result: {} },
    notifications: [
      {
        method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
        params: {
          event: {
            type: "turn.completed",
            turnId: "turn-theme-context-search",
            payload: {
              attempts: "web search completed",
            },
          },
        },
      },
    ],
    messages: [],
  }));

  client.readThread.mockImplementation(async () => {
    const turnId = "turn-theme-context-search";
    const rawResponse = JSON.stringify({
      title: "国际新闻速览",
      summary: "今天国际新闻聚焦地缘政治、市场波动与科技监管。",
      citations: [{ title: "Example", url: "https://example.com/news" }],
    });
    return {
      id: 3,
      result: {
        thread: {
          id: "thread-theme-context-search",
          sessionId: "__lime_theme_context_search__-session",
          status: { type: "idle" },
          createdAt: Date.parse("2026-06-08T00:00:00.000Z") / 1000,
          updatedAt: Date.parse("2026-06-08T00:00:02.000Z") / 1000,
          turns: [
            {
              id: turnId,
              status: "completed",
              items: [
                {
                  id: `${turnId}:assistant`,
                  type: "agentMessage",
                  text: rawResponse,
                },
              ],
            },
          ],
        },
      },
      response: { id: 3, result: {} },
      notifications: [],
      messages: [],
    };
  });

  return client;
}

describe("themeContextSearch API", () => {
  it("应通过 App Server current session turn 执行上下文搜索", async () => {
    const appServerClient = appServerClientMock();

    const result = await searchThemeContextWithAppServer(
      {
        workspaceId: " workspace-1 ",
        projectId: " project-1 ",
        providerType: " openai-compatible ",
        model: " gpt-4.1-mini ",
        query: " 今天的国际新闻 ",
        mode: "web",
      },
      appServerClient,
    );

    expect(result.rawResponse).toContain("国际新闻速览");
    expect(result.attemptsSummary).toBe("web search completed");
    expect(appServerClient.startSession).toHaveBeenCalledWith(
      expect.objectContaining({
        modelProvider: "openai-compatible",
        model: "gpt-4.1-mini",
        serviceName: "上下文搜索",
        threadSource: "appServer",
        historyMode: "paginated",
        baseInstructions: expect.stringContaining("资料检索助手"),
      }),
    );

    expect(appServerClient.startTurn).toHaveBeenCalledWith(
      expect.objectContaining({
        threadId: "thread-theme-context-search",
        input: [
          {
            type: "text",
            text: expect.stringContaining("检索主题：今天的国际新闻"),
          },
        ],
        additionalContext: expect.objectContaining({
          source: {
            kind: "application",
            value: "theme_context_search",
          },
        }),
      }),
    );

    const startTurnParams = appServerClient.startTurn.mock.calls[0]?.[0];
    const serialized = JSON.stringify(startTurnParams);
    expect(serialized).not.toContain("web_search");
    expect(serialized).not.toContain("webSearch");
    expect(serialized).not.toContain("search_mode");
    expect(serialized).not.toContain("searchMode");
  });

  it("read model 尚未水合时应使用同 turn 的 message.delta 输出", async () => {
    const appServerClient = appServerClientMock();

    appServerClient.readThread.mockResolvedValue({
      id: 3,
      result: {
        thread: {
          id: "thread-theme-context-search",
          sessionId: "__lime_theme_context_search__-session",
          status: { type: "idle" },
          createdAt: Date.parse("2026-06-08T00:00:00.000Z") / 1000,
          updatedAt: Date.parse("2026-06-08T00:00:02.000Z") / 1000,
          turns: [],
        },
      },
      response: { id: 3, result: {} },
      notifications: [],
      messages: [],
    });
    appServerClient.startTurn.mockImplementation(async () => ({
      id: 2,
      result: {
        turn: {
          id: "turn-theme-context-search",
          status: "completed",
          startedAt: Date.parse("2026-06-08T00:00:01.000Z") / 1000,
          completedAt: Date.parse("2026-06-08T00:00:02.000Z") / 1000,
        },
      },
      response: { id: 2, result: {} },
      notifications: [
        {
          method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
          params: {
            event: {
              type: "message.delta",
              turnId: "turn-theme-context-search",
              payload: {
                text: '{"title":"社媒趋势","summary":"社媒讨论正在升温",',
              },
            },
          },
        },
        {
          method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
          params: {
            event: {
              type: "message.delta",
              turnId: "turn-theme-context-search",
              payload: {
                text: '"citations":[{"title":"Example","url":"https://example.com"}]}',
              },
            },
          },
        },
        {
          method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
          params: {
            event: {
              type: "turn.completed",
              turnId: "turn-theme-context-search",
              payload: {
                attemptsSummary: "notifications completed",
              },
            },
          },
        },
      ],
      messages: [],
    }));

    const result = await searchThemeContextWithAppServer(
      {
        workspaceId: "workspace-1",
        providerType: "openai-compatible",
        model: "gpt-4.1-mini",
        query: "社媒趋势",
        mode: "social",
      },
      appServerClient,
    );

    expect(result.rawResponse).toContain("社媒趋势");
    expect(result.attemptsSummary).toBe("notifications completed");
  });

  it("App Server 未返回 assistant 输出时应 fail closed", async () => {
    const appServerClient = appServerClientMock();

    appServerClient.readThread.mockResolvedValue({
      id: 3,
      result: {
        thread: {
          id: "thread-theme-context-search",
          sessionId: "__lime_theme_context_search__-session",
          status: { type: "idle" },
          createdAt: Date.parse("2026-06-08T00:00:00.000Z") / 1000,
          updatedAt: Date.parse("2026-06-08T00:00:02.000Z") / 1000,
          turns: [],
        },
      },
      response: { id: 3, result: {} },
      notifications: [],
      messages: [],
    });

    await expect(
      searchThemeContextWithAppServer(
        {
          workspaceId: "workspace-1",
          providerType: "openai-compatible",
          model: "gpt-4.1-mini",
          query: "国际新闻",
          mode: "web",
        },
        appServerClient,
      ),
    ).rejects.toThrow("App Server 上下文搜索未返回 assistant 输出");
    expect(appServerClient.readThread).toHaveBeenCalled();
  });

  it("缺少模型配置时应 fail closed", async () => {
    const appServerClient = appServerClientMock();

    await expect(
      searchThemeContextWithAppServer(
        {
          workspaceId: "workspace-1",
          providerType: " ",
          model: "gpt-4.1-mini",
          query: "国际新闻",
          mode: "web",
        },
        appServerClient,
      ),
    ).rejects.toThrow("当前未选择可用模型，无法执行上下文搜索");

    expect(appServerClient.startSession).not.toHaveBeenCalled();
    expect(appServerClient.startTurn).not.toHaveBeenCalled();
  });
});
