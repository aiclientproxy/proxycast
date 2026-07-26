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
    request: vi.fn().mockResolvedValue(rpcResult({ data: [] })),
    readThread: vi
      .fn()
      .mockResolvedValue(rpcResult({ thread: canonicalThread() })),
    updateThreadSettings: vi.fn().mockResolvedValue(rpcResult({})),
    archiveThread: vi.fn().mockResolvedValue(rpcResult({})),
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

  it("create 缺少 current provider/model route 时应在 gateway fail closed", async () => {
    const appServerClient = appServerClientMock();
    const client = createAppServerSessionClient({ appServerClient });

    await expect(client.createAgentRuntimeSession()).rejects.toThrow(
      "thread/start requires current providerSelector and modelName",
    );
    expect(appServerClient.startSession).not.toHaveBeenCalled();
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

    expect(appServerClient.request).toHaveBeenCalledWith("thread/list", {
      archived: false,
      limit: 12,
    });
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
      messages: [
        { role: "user", content: [{ type: "text", text: "继续整理" }] },
        {
          role: "assistant",
          content: [{ type: "text", text: "已完成整理。" }],
        },
      ],
      turns: [{ id: "turn-completed", status: "completed" }],
    });

    expect(appServerClient.readThread).toHaveBeenCalledWith({
      threadId: "session-codex",
      includeTurns: false,
    });
  });

  it("get 对 paginated Thread 应读取完整 turns 页面并保持 thread/turn/item identity", async () => {
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
              id: "turn-page-1",
              status: "completed",
              startedAt: 1780704000,
              completedAt: 1780704001,
              items: [
                { id: "item-summary-1", type: "userMessage", content: [] },
              ],
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
              id: "turn-page-2",
              status: "completed",
              startedAt: 1780704002,
              completedAt: 1780704003,
              items: [
                { id: "item-summary-2", type: "agentMessage", text: "第二页" },
              ],
            },
          ],
          nextCursor: null,
          backwardsCursor: "cursor-page-1",
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          data: [
            {
              turnId: "turn-page-1",
              item: {
                id: "item-page-1",
                type: "userMessage",
                content: [{ type: "text", text: "第一页" }],
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
              turnId: "turn-page-2",
              item: {
                id: "item-page-2",
                type: "agentMessage",
                text: "第二页",
              },
            },
          ],
          nextCursor: null,
          backwardsCursor: "cursor-items-page-1",
        }),
      );
    const client = createAppServerSessionClient({ appServerClient });

    const detail = await client.getAgentRuntimeSession("session-paginated");

    expect(detail).toMatchObject({
      id: "session-paginated",
      thread_id: "thread-paginated",
      turns: [
        { id: "turn-page-1", thread_id: "thread-paginated" },
        { id: "turn-page-2", thread_id: "thread-paginated" },
      ],
      items: [
        {
          id: "item-page-1",
          thread_id: "thread-paginated",
          turn_id: "turn-page-1",
        },
        {
          id: "item-page-2",
          thread_id: "thread-paginated",
          turn_id: "turn-page-2",
        },
      ],
      messages: [
        { id: "item-page-1", role: "user" },
        { id: "item-page-2", role: "assistant" },
      ],
    });
    expect(appServerClient.readThread).toHaveBeenCalledTimes(1);
    expect(appServerClient.readThread).toHaveBeenCalledWith({
      threadId: "session-paginated",
      includeTurns: false,
    });
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      1,
      "thread/turns/list",
      {
        threadId: "thread-paginated",
        limit: 100,
        sortDirection: "asc",
        itemsView: "summary",
      },
    );
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      2,
      "thread/turns/list",
      {
        threadId: "thread-paginated",
        cursor: "cursor-page-2",
        limit: 100,
        sortDirection: "asc",
        itemsView: "summary",
      },
    );
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      3,
      "thread/items/list",
      {
        threadId: "thread-paginated",
        limit: 100,
        sortDirection: "asc",
      },
    );
    expect(appServerClient.request).toHaveBeenNthCalledWith(
      4,
      "thread/items/list",
      {
        threadId: "thread-paginated",
        cursor: "cursor-items-page-2",
        limit: 100,
        sortDirection: "asc",
      },
    );
  });

  it("get 对 legacy Thread 空 turns 应回读 includeTurns=true", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread)
      .mockResolvedValueOnce(
        rpcResult({
          thread: canonicalThread({
            id: "thread-legacy-history",
            sessionId: "session-legacy-history",
            historyMode: "legacy",
            turns: [],
          }),
        }),
      )
      .mockResolvedValueOnce(
        rpcResult({
          thread: canonicalThread({
            id: "thread-legacy-history",
            sessionId: "session-legacy-history",
            historyMode: "legacy",
            turns: [
              {
                id: "turn-legacy-history",
                status: "completed",
                startedAt: 1780704000,
                items: [
                  {
                    id: "item-legacy-history",
                    type: "userMessage",
                    content: [{ type: "text", text: "历史回读" }],
                  },
                ],
              },
            ],
          }),
        }),
      );
    const client = createAppServerSessionClient({ appServerClient });

    await expect(
      client.getAgentRuntimeSession("session-legacy-history"),
    ).resolves.toMatchObject({
      thread_id: "thread-legacy-history",
      messages: [
        { role: "user", content: [{ type: "text", text: "历史回读" }] },
      ],
    });
    expect(appServerClient.readThread).toHaveBeenNthCalledWith(2, {
      threadId: "session-legacy-history",
      includeTurns: true,
    });
    expect(appServerClient.request).not.toHaveBeenCalled();
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
