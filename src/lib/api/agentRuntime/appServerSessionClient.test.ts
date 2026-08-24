import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  AppServerThread,
  AppServerThreadStartResponse,
} from "@/lib/api/appServer";
import {
  createAppServerSessionClient,
  type AppServerSessionRpcClient,
} from "./appServerSessionClient";

function rpcResult<T>(result: T) {
  return {
    id: 1,
    result,
    response: { id: 1, result },
    notifications: [],
    messages: [],
  };
}

function canonicalThread(
  overrides: Record<string, unknown> = {},
): AppServerThread {
  const thread: AppServerThread = {
    cliVersion: "0.1.0",
    cwd: "/tmp/workspace-1",
    modelProvider: "openai-compatible",
    source: "appServer",
    id: "thread-1",
    sessionId: "session-1",
    preview: "新对话",
    ephemeral: false,
    createdAt: 1780704000,
    updatedAt: 1780704000,
    status: { type: "idle" },
    turns: [],
  };
  return Object.assign(thread, overrides);
}

function canonicalThreadStartResponse(): AppServerThreadStartResponse {
  const thread = canonicalThread();
  return {
    approvalPolicy: null,
    approvalsReviewer: null,
    cwd: thread.cwd,
    model: "gpt-5.4",
    modelProvider: thread.modelProvider,
    sandbox: null,
    thread,
  };
}

function appServerClientMock(): AppServerSessionRpcClient {
  const client = {
    startSession: vi
      .fn()
      .mockResolvedValue(rpcResult(canonicalThreadStartResponse())),
    listThreadSections: vi
      .fn()
      .mockResolvedValue(rpcResult({ data: [], nextCursor: null })),
    request: vi.fn().mockResolvedValue(rpcResult({ data: [] })),
    readThread: vi
      .fn()
      .mockResolvedValue(rpcResult({ thread: canonicalThread() })),
    updateThreadSettings: vi.fn().mockResolvedValue(rpcResult({})),
    archiveThread: vi.fn().mockResolvedValue(rpcResult({})),
    forkThread: vi.fn().mockResolvedValue(
      rpcResult({
        thread: canonicalThread({
          id: "thread-forked",
          sessionId: "session-forked",
        }),
      }),
    ),
    unarchiveThread: vi
      .fn()
      .mockResolvedValue(rpcResult({ thread: canonicalThread() })),
    deleteThread: vi.fn().mockResolvedValue(rpcResult({})),
  };
  return client as unknown as AppServerSessionRpcClient;
}

describe("appServerSessionClient", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-06T00:00:00.000Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("create 应通过 thread/start 创建桌面会话", async () => {
    const appServerClient = appServerClientMock();
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.createAgentRuntimeSession(" workspace-1 ", "  新会话  ", "react", {
        runStartHooks: false,
        metadata: {
          providerSelector: "fixture-provider",
          modelName: "fixture-model",
        },
      }),
    ).resolves.toBe("session-1");

    expect(appServerClient.startSession).toHaveBeenCalledWith({
      cwd: undefined,
      model: "fixture-model",
      modelProvider: "fixture-provider",
      serviceName: "新会话",
      threadSource: "appServer",
      historyMode: "paginated",
    });
  });

  it("create 缺少显式 route 时应由 App Server 解析 ready default", async () => {
    const appServerClient = appServerClientMock();
    const client = createAppServerSessionClient({ appServerClient });

    await expect(client.createAgentRuntimeSession()).resolves.toBe("session-1");
    expect(appServerClient.startSession).toHaveBeenCalledWith({
      cwd: undefined,
      serviceName: "新对话",
      threadSource: "appServer",
      historyMode: "paginated",
    });
  });

  it("create 收到半截 session 时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.startSession).mockResolvedValueOnce(
      rpcResult({ thread: { id: "thread-1" } }) as never,
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.createAgentRuntimeSession(undefined, undefined, undefined, {
        metadata: {
          providerSelector: "fixture-provider",
          modelName: "fixture-model",
        },
      }),
    ).rejects.toThrow("thread/start did not return canonical Thread");
  });

  it("list 应通过 thread/list 读取并投影 canonical Thread", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.request).mockResolvedValueOnce(
      rpcResult({
        data: [
          canonicalThread({
            id: "thread-list",
            sessionId: "session-list",
            preview: "Runtime Session",
            modelProvider: "gpt-5.4",
            createdAt: 1780704000,
            updatedAt: 1780704002,
            cwd: "/tmp/workspace-1",
            metadata: {
              workspaceId: "workspace-1",
              workingDir: "/tmp/workspace-1",
              executionStrategy: "react",
            },
            status: { type: "active", activeFlags: [] },
            turns: [
              {
                id: "turn-running",
                status: "inProgress",
                queue: { state: "running" },
              },
              {
                id: "turn-queued",
                status: "inProgress",
                queue: { state: "queued", position: 0 },
              },
            ],
          }),
        ],
        nextCursor: null,
        backwardsCursor: null,
      }),
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.listAgentRuntimeSessions({
        includeArchived: true,
        workspaceId: " workspace-1 ",
        limit: 12.8,
      }),
    ).resolves.toEqual([
      expect.objectContaining({
        id: "session-list",
        thread_id: "thread-list",
        name: "Runtime Session",
        model: "gpt-5.4",
        created_at: 1780704000000,
        updated_at: 1780704002000,
        workspace_id: "workspace-1",
        working_dir: "/tmp/workspace-1",
        execution_strategy: "react",
        thread_status: "running",
        latest_turn_status: "running",
        active_turn_id: "turn-running",
        queued_turn_count: 1,
      }),
    ]);

    expect(appServerClient.listThreadSections).toHaveBeenCalledWith({
      limit: 100,
    });
    expect(appServerClient.request).toHaveBeenCalledWith("thread/list", {
      archived: false,
      limit: 12,
      sectionId: null,
    });
  });

  it("list 应保持 section catalog 与 section_position 的服务端顺序", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.listThreadSections).mockResolvedValueOnce(
      rpcResult({
        data: [
          {
            id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
            name: "Pinned",
          },
          { id: "section-active", name: "Active" },
        ],
        nextCursor: null,
      }),
    );
    vi.mocked(appServerClient.request)
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            canonicalThread({
              id: "thread-pinned-first",
              sessionId: "session-pinned-first",
              updatedAt: 10,
              section: {
                id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
                name: "Pinned",
              },
              sectionEnteredAt: 8,
            }),
            canonicalThread({
              id: "thread-pinned-second",
              sessionId: "session-pinned-second",
              updatedAt: 30,
              section: {
                id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
                name: "Pinned",
              },
              sectionEnteredAt: 9,
            }),
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            canonicalThread({
              id: "thread-active",
              sessionId: "session-active",
              updatedAt: 40,
              section: { id: "section-active", name: "Active" },
              sectionEnteredAt: 7,
            }),
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            canonicalThread({
              id: "thread-unsectioned",
              sessionId: "session-unsectioned",
              updatedAt: 50,
            }),
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      );
    const client = createAppServerSessionClient({ appServerClient });

    const sessions = await client.listAgentRuntimeSessions();

    expect(sessions.map((session) => session.id)).toEqual([
      "session-pinned-first",
      "session-pinned-second",
      "session-active",
      "session-unsectioned",
    ]);
    expect(sessions[0]).toMatchObject({
      section: {
        id: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
        name: "Pinned",
      },
      section_entered_at: 8_000,
    });
    expect(appServerClient.request).toHaveBeenNthCalledWith(1, "thread/list", {
      archived: false,
      limit: 100,
      sectionId: "01984de2-8f74-7c91-a3b2-5c5e937cf318",
      sortKey: "section_position",
    });
    expect(appServerClient.request).toHaveBeenNthCalledWith(2, "thread/list", {
      archived: false,
      limit: 100,
      sectionId: "section-active",
      sortKey: "section_position",
    });
    expect(appServerClient.request).toHaveBeenNthCalledWith(3, "thread/list", {
      archived: false,
      limit: 100,
      sectionId: null,
    });
  });

  it("list 后首次 get 仍应通过 thread/read 刷新 canonical 产品投影", async () => {
    const appServerClient = appServerClientMock();
    const articleWorkspace = {
      schemaVersion: "article-workspace.v1",
      objects: [
        {
          ref: {
            appId: "content-factory-app",
            kind: "articleDraft",
            id: "article-1",
            sessionId: "session-navigation",
          },
        },
      ],
    };
    vi.mocked(appServerClient.request)
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            canonicalThread({
              id: "thread-navigation",
              sessionId: "session-navigation",
              historyMode: "legacy",
              turns: [],
            }),
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [],
          nextCursor: null,
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [],
          nextCursor: null,
          backwardsCursor: null,
        }),
      );
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({
        thread: canonicalThread({
          id: "thread-navigation",
          sessionId: "session-navigation",
          historyMode: "legacy",
          metadata: { articleWorkspace },
          turns: [],
        }),
      }),
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(client.listAgentRuntimeSessions()).resolves.toHaveLength(1);
    await expect(
      client.getAgentRuntimeSession("session-navigation"),
    ).resolves.toMatchObject({
      id: "session-navigation",
      thread_id: "thread-navigation",
      messages: [],
      thread_read: {
        articleWorkspace,
      },
    });
    expect(appServerClient.readThread).toHaveBeenCalledTimes(1);
    expect(appServerClient.readThread).toHaveBeenCalledWith({
      threadId: "thread-navigation",
      includeTurns: false,
    });
    expect(appServerClient.request).toHaveBeenCalledTimes(3);
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      2,
      "thread/items/list",
      {
        threadId: "thread-navigation",
        limit: 40,
        sortDirection: "desc",
      },
    );
  });

  it("list 收到非 canonical envelope 时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.request).mockResolvedValueOnce(
      rpcResult({ success: true }) as never,
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(client.listAgentRuntimeSessions()).rejects.toThrow(
      "thread/list did not return session list",
    );
  });

  it("get 应从 canonical Thread items 恢复消息并分离排队回合", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({
        thread: canonicalThread({
          id: "thread-codex",
          sessionId: "session-codex",
          preview: "Codex canonical thread",
          modelProvider: "openai",
          cwd: "/tmp/codex",
          createdAt: 1780704000,
          updatedAt: 1780704002,
          status: { type: "active", activeFlags: [] },
          turns: [
            {
              id: "turn-completed",
              status: "completed",
              startedAt: 1780704000,
              completedAt: 1780704001,
              items: [
                {
                  id: "item-user",
                  type: "userMessage",
                  content: [{ type: "text", text: "继续整理" }],
                },
                {
                  id: "item-agent",
                  type: "agentMessage",
                  text: "已完成整理。",
                  phase: "final_answer",
                },
              ],
            },
            {
              id: "turn-queued",
              status: "inProgress",
              queue: { state: "queued", position: 0 },
              startedAt: 1780704002,
            },
          ],
        }),
      }),
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.getAgentRuntimeSession("session-codex"),
    ).resolves.toMatchObject({
      id: "session-codex",
      thread_id: "thread-codex",
      messages: [],
      items: [
        { type: "user_message", content: "继续整理" },
        { type: "agent_message", text: "已完成整理。" },
      ],
      turns: [{ id: "turn-completed", status: "completed" }],
    });

    expect(appServerClient.readThread).toHaveBeenCalledWith({
      threadId: "session-codex",
      includeTurns: false,
    });
  });

  it("get 对 paginated Thread 应通过 current 分页保持 thread/turn/item identity", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({
        thread: canonicalThread({
          id: "thread-paginated",
          sessionId: "session-paginated",
          historyMode: "paginated",
          status: { type: "idle" },
          turns: [],
        }),
      }),
    );
    vi.mocked(appServerClient.request)
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              turnId: "turn-page-2",
              item: {
                id: "item-page-2",
                type: "agentMessage",
                text: "第二页",
              },
            },
          ],
          nextCursor: "cursor-items-page-2",
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              id: "turn-page-2",
              status: "completed",
              startedAt: 1780704002,
              completedAt: 1780704003,
            },
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              id: "turn-page-2",
              status: "completed",
              startedAt: 1780704002,
              completedAt: 1780704003,
            },
          ],
          nextCursor: "cursor-page-2",
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              id: "turn-page-1",
              status: "completed",
              startedAt: 1780704000,
              completedAt: 1780704001,
            },
          ],
          nextCursor: null,
          backwardsCursor: "cursor-page-1",
        }),
      );
    const client = createAppServerSessionClient({ appServerClient });

    const detail = await client.getAgentRuntimeSession("session-paginated");

    expect(detail).toMatchObject({
      id: "session-paginated",
      thread_id: "thread-paginated",
      turns: [{ id: "turn-page-2", thread_id: "thread-paginated" }],
      items: [
        {
          id: "item-page-2",
          thread_id: "thread-paginated",
          turn_id: "turn-page-2",
        },
      ],
      messages: [],
      messages_count: 1,
      history_limit: 40,
      history_cursor: {
        item_cursor: "cursor-items-page-2",
        turn_cursor: null,
        loaded_entry_count: 2,
        loaded_turn_count: 1,
        loaded_item_count: 1,
        has_more: true,
      },
      history_truncated: true,
    });
    expect(appServerClient.readThread).toHaveBeenCalledTimes(1);
    expect(appServerClient.readThread).toHaveBeenCalledWith({
      threadId: "session-paginated",
      includeTurns: false,
    });
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      1,
      "thread/items/list",
      {
        threadId: "thread-paginated",
        limit: 40,
        sortDirection: "desc",
      },
    );
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      2,
      "thread/turns/list",
      {
        threadId: "thread-paginated",
        limit: 40,
        sortDirection: "desc",
        itemsView: "summary",
      },
    );
    expect(appServerClient.request).toHaveBeenCalledTimes(2);
  });

  it("get 应把 opaque Item cursor 原样传给 current owner", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({
        thread: canonicalThread({
          id: "thread-paginated-cursor",
          sessionId: "session-paginated-cursor",
          historyMode: "paginated",
          turns: [
            {
              id: "turn-partial",
              status: "completed",
              startedAt: 1780704000,
              items: [
                {
                  id: "999",
                  type: "agentMessage",
                  text: "partial embedded item must not become history truth",
                },
              ],
            },
          ],
        }),
      }),
    );
    vi.mocked(appServerClient.request)
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              turnId: "turn-cursor",
              item: {
                id: "message-latest-uuid",
                type: "agentMessage",
                text: "最新",
              },
            },
            {
              turnId: "turn-cursor",
              item: {
                id: "message-cursor-uuid",
                type: "userMessage",
                content: [{ type: "text", text: "游标消息" }],
              },
            },
            {
              turnId: "turn-cursor",
              item: {
                id: "message-older-uuid",
                type: "agentMessage",
                text: "更早一页",
              },
            },
            {
              turnId: "turn-cursor",
              item: {
                id: "message-earliest-uuid",
                type: "userMessage",
                content: [{ type: "text", text: "最早" }],
              },
            },
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              id: "turn-cursor",
              status: "completed",
              startedAt: 1780704000,
              completedAt: 1780704001,
            },
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.getAgentRuntimeSession("session-paginated-cursor", {
        historyItemCursor: "message-cursor-uuid",
        historyLimit: 1,
      }),
    ).resolves.toMatchObject({
      messages: [],
      items: [
        { id: "message-earliest-uuid", type: "user_message" },
        { id: "message-older-uuid", type: "agent_message" },
        { id: "message-cursor-uuid", type: "user_message" },
        { id: "message-latest-uuid", type: "agent_message" },
      ],
      messages_count: 4,
      history_limit: 1,
      history_cursor: {
        item_cursor: null,
        turn_cursor: null,
        loaded_entry_count: 5,
        loaded_turn_count: 1,
        loaded_item_count: 4,
        has_more: false,
      },
      history_truncated: false,
    });
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      1,
      "thread/items/list",
      {
        threadId: "thread-paginated-cursor",
        cursor: "message-cursor-uuid",
        limit: 1,
        sortDirection: "desc",
      },
    );
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      2,
      "thread/turns/list",
      {
        threadId: "thread-paginated-cursor",
        limit: 1,
        sortDirection: "desc",
        itemsView: "summary",
      },
    );
  });

  it("导入 canonical Thread 应自动读取 Item/Turn 到 EOF，不能从中间开始", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({
        thread: canonicalThread({
          id: "thread-imported-full-history",
          sessionId: "session-imported-full-history",
          historyMode: "paginated",
          metadata: {
            source_client: "codex",
            source_thread_id: "codex-thread-20260729",
          },
          turns: [],
        }),
      }),
    );
    vi.mocked(appServerClient.request)
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              turnId: "turn-imported",
              item: {
                id: "item-latest",
                type: "agentMessage",
                text: "最新回复",
              },
            },
          ],
          nextCursor: "item-page-2",
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              turnId: "turn-imported",
              item: {
                id: "item-earliest",
                type: "userMessage",
                content: [{ type: "text", text: "最早问题" }],
              },
            },
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              id: "turn-imported",
              status: "completed",
              startedAt: 1780704000,
              completedAt: 1780704001,
            },
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.getAgentRuntimeSession("session-imported-full-history", {
        historyLimit: 40,
      }),
    ).resolves.toMatchObject({
      history_limit: 0,
      history_cursor: {
        item_cursor: null,
        turn_cursor: null,
        loaded_item_count: 2,
        has_more: false,
      },
      items: [
        { id: "item-earliest", type: "user_message" },
        { id: "item-latest", type: "agent_message" },
      ],
      history_truncated: false,
    });

    expect(appServerClient.request).toHaveBeenNthCalledWith(
      1,
      "thread/items/list",
      {
        threadId: "thread-imported-full-history",
        limit: 100,
        sortDirection: "desc",
      },
    );
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      2,
      "thread/items/list",
      {
        threadId: "thread-imported-full-history",
        cursor: "item-page-2",
        limit: 100,
        sortDirection: "desc",
      },
    );
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      3,
      "thread/turns/list",
      {
        threadId: "thread-imported-full-history",
        limit: 100,
        sortDirection: "desc",
        itemsView: "summary",
      },
    );
    expect(appServerClient.request).toHaveBeenCalledTimes(3);
  });

  it("get 应允许一个 owner 到 EOF 后只继续另一个 opaque cursor", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({
        thread: canonicalThread({
          id: "thread-turn-cursor",
          sessionId: "session-turn-cursor",
          historyMode: "paginated",
          turns: [],
        }),
      }),
    );
    vi.mocked(appServerClient.request).mockResolvedValueOnce(
      rpcResult({
        data: [
          {
            id: "turn-failed-page-2",
            status: "failed",
            startedAt: 1780704000,
            completedAt: 1780704001,
            error: { message: "failed before creating an item" },
          },
        ],
        nextCursor: null,
        backwardsCursor: null,
      }),
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.getAgentRuntimeSession("session-turn-cursor", {
        historyItemCursor: null,
        historyLimit: 40,
        historyTurnCursor: " opaque-turn-page-2 ",
      }),
    ).resolves.toMatchObject({
      turns: [
        {
          id: "turn-failed-page-2",
          status: "failed",
          error_message: "failed before creating an item",
        },
      ],
      items: [],
      history_cursor: {
        item_cursor: null,
        turn_cursor: null,
        loaded_entry_count: 1,
        loaded_turn_count: 1,
        loaded_item_count: 0,
        has_more: false,
      },
    });
    expect(appServerClient.request).toHaveBeenCalledTimes(1);
    expect(appServerClient.request).toHaveBeenCalledWith("thread/turns/list", {
      threadId: "thread-turn-cursor",
      cursor: " opaque-turn-page-2 ",
      limit: 40,
      sortDirection: "desc",
      itemsView: "summary",
    });
  });

  it("get 应保留无 Item 的 failed Turn", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({
        thread: canonicalThread({
          id: "thread-failed",
          sessionId: "session-failed",
          turns: [
            {
              id: "turn-failed",
              status: "failed",
              startedAt: 1780704000,
              completedAt: 1780704001,
              error: { message: "provider failed" },
              items: [],
            },
          ],
        }),
      }),
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.getAgentRuntimeSession("session-failed"),
    ).resolves.toMatchObject({
      id: "session-failed",
      messages: [],
      turns: [
        {
          id: "turn-failed",
          status: "failed",
          error_message: "provider failed",
        },
      ],
      items: [],
    });
    expect(appServerClient.request).not.toHaveBeenCalled();
  });

  it("get 应返回 Turn owner cursor 而不是扫描后续页", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({
        thread: canonicalThread({
          id: "thread-failed-page-2",
          sessionId: "session-failed-page-2",
          historyMode: "paginated",
          turns: [],
        }),
      }),
    );
    vi.mocked(appServerClient.request)
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              turnId: "turn-selected-page-1",
              item: {
                id: "message-selected-page-1",
                type: "agentMessage",
                text: "最新回复",
              },
            },
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              id: "turn-selected-page-1",
              status: "completed",
              startedAt: 1780704002,
              completedAt: 1780704003,
            },
          ],
          nextCursor: "turn-page-2",
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              id: "turn-failed-page-2",
              status: "failed",
              startedAt: 1780704000,
              completedAt: 1780704001,
              error: { message: "failed before creating an item" },
            },
          ],
          nextCursor: null,
          backwardsCursor: "turn-page-1",
        }),
      );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.getAgentRuntimeSession("session-failed-page-2"),
    ).resolves.toMatchObject({
      turns: [{ id: "turn-selected-page-1", status: "completed" }],
      items: [{ id: "message-selected-page-1" }],
      history_cursor: {
        item_cursor: null,
        turn_cursor: "turn-page-2",
        has_more: true,
      },
    });
    expect(appServerClient.request).toHaveBeenCalledTimes(2);
  });

  it("get 对空 embedded Thread 应走 current Turn/Item owner page", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({
        thread: canonicalThread({
          id: "thread-legacy-history",
          sessionId: "session-legacy-history",
          historyMode: "legacy",
          turns: [],
        }),
      }),
    );
    vi.mocked(appServerClient.request)
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              turnId: "turn-legacy-history",
              item: {
                id: "item-agent-latest",
                type: "agentMessage",
                text: "最新回复",
              },
            },
            {
              turnId: "turn-legacy-history",
              item: {
                id: "item-user-older",
                type: "userMessage",
                content: [{ type: "text", text: "历史回读" }],
              },
            },
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              id: "turn-legacy-history",
              status: "completed",
              startedAt: 1780704000,
              completedAt: 1780704001,
            },
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.getAgentRuntimeSession("session-legacy-history", {
        historyLimit: 1,
      }),
    ).resolves.toMatchObject({
      thread_id: "thread-legacy-history",
      messages: [],
      items: [
        {
          id: "item-user-older",
          type: "user_message",
        },
        {
          id: "item-agent-latest",
          type: "agent_message",
        },
      ],
      messages_count: 2,
      history_limit: 1,
      history_cursor: {
        item_cursor: null,
        turn_cursor: null,
        loaded_entry_count: 3,
        loaded_turn_count: 1,
        loaded_item_count: 2,
        has_more: false,
      },
      history_truncated: false,
    });
    expect(appServerClient.readThread).toHaveBeenCalledTimes(1);
    expect(appServerClient.readThread).toHaveBeenCalledWith({
      threadId: "session-legacy-history",
      includeTurns: false,
    });
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      1,
      "thread/items/list",
      {
        threadId: "thread-legacy-history",
        limit: 1,
        sortDirection: "desc",
      },
    );
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      2,
      "thread/turns/list",
      {
        threadId: "thread-legacy-history",
        limit: 1,
        sortDirection: "desc",
        itemsView: "summary",
      },
    );
  });

  it("get 达到历史消息窗口后不应继续扫描剩余 Item 页面", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({
        thread: canonicalThread({
          id: "thread-bounded-history",
          sessionId: "session-bounded-history",
          historyMode: "legacy",
          turns: [],
        }),
      }),
    );
    vi.mocked(appServerClient.request)
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              turnId: "turn-bounded-history",
              item: {
                id: "item-agent-latest",
                type: "agentMessage",
                text: "最新回复",
              },
            },
          ],
          nextCursor: "cursor-with-more-items",
          backwardsCursor: null,
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              id: "turn-bounded-history",
              status: "completed",
              startedAt: 1780704000,
              completedAt: 1780704001,
            },
          ],
          nextCursor: null,
          backwardsCursor: null,
        }),
      );
    const client = createAppServerSessionClient({ appServerClient });

    const detail = await client.getAgentRuntimeSession(
      "session-bounded-history",
      { historyLimit: 1 },
    );

    expect(detail).toMatchObject({
      messages: [],
      items: [{ id: "item-agent-latest", type: "agent_message" }],
      history_limit: 1,
      history_cursor: {
        item_cursor: "cursor-with-more-items",
        turn_cursor: null,
        loaded_entry_count: 2,
        loaded_turn_count: 1,
        loaded_item_count: 1,
        has_more: true,
      },
      history_truncated: true,
    });
    expect(detail.messages_count).toBe(1);
    expect(appServerClient.request).toHaveBeenCalledTimes(2);
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      1,
      "thread/items/list",
      {
        threadId: "thread-bounded-history",
        limit: 1,
        sortDirection: "desc",
      },
    );
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      2,
      "thread/turns/list",
      {
        threadId: "thread-bounded-history",
        limit: 1,
        sortDirection: "desc",
        itemsView: "summary",
      },
    );
  });

  it("get 遇到旧 session envelope 时应显式拒绝，不恢复兼容解析", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({ session: {}, turns: [] }) as never,
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.getAgentRuntimeSession("session-legacy"),
    ).rejects.toThrow("thread/read did not return canonical session detail");
  });

  it("get 应拒绝不属于请求 session/thread 的 canonical Thread", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce(
      rpcResult({
        thread: canonicalThread({
          id: "thread-other",
          sessionId: "session-other",
          turns: [],
        }),
      }),
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.getAgentRuntimeSession("session-requested"),
    ).rejects.toThrow("thread/read canonical identity mismatch");
    expect(appServerClient.request).not.toHaveBeenCalled();
  });

  it("get 缺少 sessionId 时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    const client = createAppServerSessionClient({ appServerClient });

    await expect(client.getAgentRuntimeSession(" ")).rejects.toThrow(
      "sessionId is required to read App Server session",
    );
    expect(appServerClient.readThread).not.toHaveBeenCalled();
  });

  it("tool preferences 应通过 current thread/settings/update 写入", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.request).mockResolvedValueOnce(
      rpcResult({ data: [canonicalThread()] }),
    );
    const client = createAppServerSessionClient({ appServerClient });

    await client.updateAgentRuntimeThreadToolPreferences(" session-1 ", {
      task: true,
      subagent: false,
    });

    expect(appServerClient.updateThreadSettings).toHaveBeenCalledWith({
      threadId: "thread-1",
      toolPreferences: { task: true, subagent: false },
    });
  });

  it("archive 应解析 canonical threadId 后调用 thread/archive", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.request).mockResolvedValueOnce(
      rpcResult({ data: [canonicalThread()] }) as never,
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.archiveAgentRuntimeSession(" session-1 "),
    ).resolves.toBeUndefined();
    expect(appServerClient.archiveThread).toHaveBeenCalledWith({
      threadId: "thread-1",
    });
  });

  it("fork 应解析 canonical threadId 并返回新 session identity", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.request).mockResolvedValueOnce(
      rpcResult({ data: [canonicalThread()] }) as never,
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(client.forkAgentRuntimeSession(" session-1 ")).resolves.toBe(
      "session-forked",
    );
    expect(appServerClient.forkThread).toHaveBeenCalledWith({
      threadId: "thread-1",
    });
  });

  it("fork 返回不完整 Thread 时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.request).mockResolvedValueOnce(
      rpcResult({ data: [canonicalThread()] }) as never,
    );
    vi.mocked(appServerClient.forkThread).mockResolvedValueOnce(
      rpcResult({ thread: { id: "thread-forked" } }) as never,
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(client.forkAgentRuntimeSession("session-1")).rejects.toThrow(
      "thread/fork returned an incomplete canonical Thread",
    );
  });

  it("unarchive 应校验 App Server 返回的 restored thread", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.request).mockResolvedValueOnce(
      rpcResult({ data: [canonicalThread({ archived: true })] }) as never,
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.unarchiveAgentRuntimeSession("session-1"),
    ).resolves.toBeUndefined();
    expect(appServerClient.unarchiveThread).toHaveBeenCalledWith({
      threadId: "thread-1",
    });
  });

  it("unarchive 返回错误 thread 身份时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.request).mockResolvedValueOnce(
      rpcResult({ data: [canonicalThread({ archived: true })] }) as never,
    );
    vi.mocked(appServerClient.unarchiveThread).mockResolvedValueOnce(
      rpcResult({ thread: canonicalThread({ id: "thread-other" }) }) as never,
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.unarchiveAgentRuntimeSession("session-1"),
    ).rejects.toThrow("thread/unarchive did not return the restored thread");
  });

  it("delete 应物理清理 current session", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.request).mockResolvedValueOnce(
      rpcResult({ data: [canonicalThread()] }) as never,
    );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.deleteAgentRuntimeSession(" session-1 "),
    ).resolves.toBeUndefined();
    expect(appServerClient.deleteThread).toHaveBeenCalledWith({
      threadId: "thread-1",
    });
  });
});
