import { beforeEach, describe, expect, it, vi } from "vitest";
import { listenAgentRuntimeEvent } from "@/lib/api/agentRuntimeEvents";
import {
  createAgentRuntimeAdapter,
  defaultAgentRuntimeAdapter,
} from "./agentRuntimeAdapter";

const { mockCreateAgentRuntimeClient, mockRuntimeClient } = vi.hoisted(() => {
  const mockRuntimeClient = {
    compactAgentRuntimeSession: vi.fn(),
    createAgentRuntimeSession: vi.fn(),
    deleteAgentRuntimeSession: vi.fn(),
    generateAgentRuntimeSessionTitle: vi.fn(),
    getAgentRuntimeSession: vi.fn(),
    getAgentRuntimeThreadRead: vi.fn(),
    readAgentRuntimeThread: vi.fn(),
    getRuntimeProviderSelection: vi.fn(),
    interruptAgentRuntimeTurn: vi.fn(),
    listAgentRuntimeSessions: vi.fn(),
    replayAgentRuntimeRequest: vi.fn(),
    resumeThread: vi.fn(),
    respondAgentRuntimeAction: vi.fn(),
    runUserShellCommand: vi.fn(),
    setAgentRuntimeThreadName: vi.fn(),
    steerAgentRuntimeTurn: vi.fn(),
    submitAgentRuntimeTurn: vi.fn(),
    updateAgentRuntimeThreadSettings: vi.fn(),
  };

  return {
    mockCreateAgentRuntimeClient: vi.fn(() => mockRuntimeClient),
    mockRuntimeClient,
  };
});

vi.mock("@/lib/api/agentRuntimeEvents", () => ({
  listenAgentRuntimeEvent: vi.fn(),
}));

vi.mock("@/lib/api/agentRuntime/clientFactory", () => ({
  createAgentRuntimeClient: mockCreateAgentRuntimeClient,
}));

describe("defaultAgentRuntimeAdapter", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("应通过 agentRuntimeEvents 代理 turn 事件监听", async () => {
    const unlisten = vi.fn();
    vi.mocked(listenAgentRuntimeEvent).mockResolvedValue(unlisten);

    const handler = vi.fn();

    await expect(
      defaultAgentRuntimeAdapter.listenToTurnEvents("turn-event", handler),
    ).resolves.toBe(unlisten);
    expect(listenAgentRuntimeEvent).toHaveBeenCalledWith("turn-event", handler);
  });

  it("应允许注入自定义 runtime 事件监听器", async () => {
    const injectedListen = vi.fn().mockResolvedValue(vi.fn());
    const adapter = createAgentRuntimeAdapter({
      listenRuntimeEvent: injectedListen,
    });
    const handler = vi.fn();

    await adapter.listenToTurnEvents("turn-event-2", handler);

    expect(injectedListen).toHaveBeenCalledWith("turn-event-2", handler);
  });

  it("应允许注入自定义 runtime client", async () => {
    const client = {
      ...mockRuntimeClient,
      createAgentRuntimeSession: vi.fn().mockResolvedValue("session-9"),
    };
    const adapter = createAgentRuntimeAdapter({
      client,
    });

    await expect(
      adapter.createSession("workspace-9", "新会话", "react", {
        runStartHooks: false,
      }),
    ).resolves.toBe("session-9");

    expect(client.createAgentRuntimeSession).toHaveBeenCalledWith(
      "workspace-9",
      "新会话",
      "react",
      { runStartHooks: false },
    );
  });

  it("runUserShellCommand 应透传 canonical thread identity 与 event route", async () => {
    const client = {
      ...mockRuntimeClient,
      runUserShellCommand: vi.fn().mockResolvedValue(undefined),
    };
    const adapter = createAgentRuntimeAdapter({ client });

    await adapter.runUserShellCommand(
      "thread-9",
      "printf ready",
      "agentSession/event/session-9",
    );

    expect(client.runUserShellCommand).toHaveBeenCalledWith(
      { threadId: "thread-9", command: "printf ready" },
      "agentSession/event/session-9",
    );
  });

  it("renameSession 应只调用 current thread/setName", async () => {
    const client = {
      ...mockRuntimeClient,
      getAgentRuntimeSession: vi.fn().mockResolvedValue({
        id: "session-9",
        thread_id: "thread-9",
        messages: [],
      }),
      setAgentRuntimeThreadName: vi.fn().mockResolvedValue(undefined),
    };
    const adapter = createAgentRuntimeAdapter({ client });

    await adapter.renameSession("session-9", "新标题");

    expect(client.setAgentRuntimeThreadName).toHaveBeenCalledWith({
      threadId: "thread-9",
      name: "新标题",
    });
  });

  it("listSessions 应透传筛选参数给 runtime client", async () => {
    const client = {
      ...mockRuntimeClient,
      listAgentRuntimeSessions: vi.fn().mockResolvedValue([]),
    };
    const adapter = createAgentRuntimeAdapter({
      client,
    });

    await expect(
      adapter.listSessions({
        workspaceId: "workspace-9",
      }),
    ).resolves.toEqual([]);

    expect(client.listAgentRuntimeSessions).toHaveBeenCalledWith({
      workspaceId: "workspace-9",
    });
  });

  it("getSessionReadModel 应先从 session detail 解析 canonical thread_id", async () => {
    const client = {
      ...mockRuntimeClient,
      getAgentRuntimeSession: vi.fn().mockResolvedValue({
        id: "session-9",
        thread_id: "thread-9",
        created_at: 1,
        updated_at: 2,
        messages: [],
        thread_read: null,
      }),
      getAgentRuntimeThreadRead: vi.fn().mockResolvedValue({
        thread_id: "thread-9",
      }),
    };
    const adapter = createAgentRuntimeAdapter({ client });

    await expect(adapter.getSessionReadModel("session-9")).resolves.toEqual({
      thread_id: "thread-9",
    });
    expect(client.getAgentRuntimeSession).toHaveBeenCalledWith(
      "session-9",
      undefined,
    );
    expect(client.getAgentRuntimeThreadRead).toHaveBeenCalledWith("thread-9");
  });

  it("getSessionReadModel 应拒绝错配 session/thread identity 且不得读取其他 Thread", async () => {
    const client = {
      ...mockRuntimeClient,
      getAgentRuntimeSession: vi.fn().mockResolvedValue({
        id: "session-other",
        thread_id: "thread-other",
        created_at: 1,
        updated_at: 2,
        messages: [],
        thread_read: null,
      }),
      getAgentRuntimeThreadRead: vi.fn(),
    };
    const adapter = createAgentRuntimeAdapter({ client });

    await expect(
      adapter.getSessionReadModel("session-requested"),
    ).rejects.toThrow("canonical session detail identity mismatch");
    expect(client.getAgentRuntimeThreadRead).not.toHaveBeenCalled();
  });

  it("getThreadTurnControl 应透传 canonical threadId 并返回窄投影", async () => {
    const client = {
      ...mockRuntimeClient,
      readAgentRuntimeThread: vi.fn().mockResolvedValue({
        thread: {
          archived: false,
          createdAt: 0.1,
          id: "thread-9",
          sessionId: "session-9",
          status: { type: "active" },
          turns: [
            {
              id: "turn-active",
              items: [],
              status: "inProgress",
            },
          ],
          updatedAt: 0.2,
        },
      }),
    };
    const adapter = createAgentRuntimeAdapter({ client });

    await expect(adapter.getThreadTurnControl("thread-9")).resolves.toEqual({
      threadId: "thread-9",
      updatedAtMs: 200,
      activeTurnId: "turn-active",
      queuedTurnIds: [],
    });
    expect(client.readAgentRuntimeThread).toHaveBeenCalledWith("thread-9");
  });

  it("getThreadTurnControl 应拒绝非 full canonical read", async () => {
    const client = {
      ...mockRuntimeClient,
      readAgentRuntimeThread: vi.fn().mockResolvedValue({
        thread: {
          archived: false,
          createdAt: 0.1,
          id: "thread-9",
          sessionId: "session-9",
          status: { type: "active" },
          updatedAt: 0.2,
        },
      }),
    };
    const adapter = createAgentRuntimeAdapter({ client });

    await expect(adapter.getThreadTurnControl("thread-9")).rejects.toThrow(
      "canonical turn-control projection rejected",
    );
  });

  it("getThreadTurnControl 应拒绝返回其他 thread identity", async () => {
    const client = {
      ...mockRuntimeClient,
      readAgentRuntimeThread: vi.fn().mockResolvedValue({
        thread: {
          archived: false,
          createdAt: 0.1,
          id: "thread-other",
          sessionId: "session-9",
          status: { type: "idle" },
          turns: [],
          updatedAt: 0.2,
        },
      }),
    };
    const adapter = createAgentRuntimeAdapter({ client });

    await expect(adapter.getThreadTurnControl("thread-9")).rejects.toThrow(
      "canonical turn-control thread identity mismatch",
    );
  });

  it("submitOp 应只委托 typed turn/start", async () => {
    const client = {
      ...mockRuntimeClient,
      submitAgentRuntimeTurn: vi.fn().mockResolvedValue(undefined),
    };
    const adapter = createAgentRuntimeAdapter({ client });

    await adapter.submitOp({
      type: "user_input",
      eventName: "event-start",
      turn: {
        threadId: "thread-9",
        clientUserMessageId: "user-9",
        input: [{ type: "text", text: "开始" }],
        approvalPolicy: "on-request",
        sandboxPolicy: "workspace-write",
      },
    });

    expect(client.submitAgentRuntimeTurn).toHaveBeenCalledWith({
      threadId: "thread-9",
      clientUserMessageId: "user-9",
      input: [{ type: "text", text: "开始" }],
      approvalPolicy: "on-request",
      sandboxPolicy: "workspace-write",
      additionalContext: {
        rendererEventName: {
          kind: "application",
          value: "event-start",
        },
      },
    });
    expect(client.steerAgentRuntimeTurn).not.toHaveBeenCalled();
  });

  it("steerTurn 应只委托 typed turn/steer 并返回原 turn identity", async () => {
    const client = {
      ...mockRuntimeClient,
      steerAgentRuntimeTurn: vi.fn().mockResolvedValue({
        result: { turnId: "turn-active" },
        notifications: [],
      }),
    };
    const adapter = createAgentRuntimeAdapter({ client });
    const request = {
      threadId: "thread-9",
      expectedTurnId: "turn-active",
      clientUserMessageId: "user-steer",
      input: [{ type: "text" as const, text: "补充约束" }],
    };

    await expect(adapter.steerTurn(request)).resolves.toEqual({
      turnId: "turn-active",
    });
    expect(client.steerAgentRuntimeTurn).toHaveBeenCalledWith(request);
    expect(client.submitAgentRuntimeTurn).not.toHaveBeenCalled();
  });

  it("resumeThread 应把真实 response snapshot 投影后交给 stream consumer", async () => {
    const client = {
      ...mockRuntimeClient,
      resumeThread: vi.fn().mockResolvedValue({
        result: {
          approvalPolicy: null,
          approvalsReviewer: null,
          cwd: "/workspace",
          model: "gpt-5.4",
          modelProvider: "openai",
          sandbox: null,
          thread: {
            cliVersion: "0.0.0",
            createdAt: 1_753_132_800,
            cwd: "/workspace",
            ephemeral: false,
            id: "thread-resume",
            modelProvider: "openai",
            preview: "Resume",
            sessionId: "session-resume",
            source: "appServer",
            status: { type: "idle" },
            turns: [
              {
                id: "turn-resume",
                status: "completed",
                startedAt: 1_753_132_800,
                completedAt: 1_753_132_801,
                items: [
                  {
                    id: "item-resume",
                    type: "agentMessage",
                    text: "已恢复",
                  },
                ],
              },
            ],
            updatedAt: 1_753_132_801,
          },
        },
        notifications: [],
      }),
    };
    const adapter = createAgentRuntimeAdapter({ client });
    const consumeReplay = vi.fn();

    await expect(
      adapter.resumeThread("thread-resume", consumeReplay),
    ).resolves.toBe(true);

    expect(client.resumeThread).toHaveBeenCalledWith({
      threadId: "thread-resume",
    });
    expect(consumeReplay).toHaveBeenCalledTimes(1);
    expect(consumeReplay.mock.calls[0]?.[0].getProjection()).toMatchObject({
      thread_id: "thread-resume",
      items: [expect.objectContaining({ id: "item-resume" })],
      turns: [
        expect.objectContaining({ id: "turn-resume", status: "completed" }),
      ],
    });
  });

  it("resumeThread identity 错配时应 fail closed 且不发布 replay", async () => {
    const client = {
      ...mockRuntimeClient,
      resumeThread: vi.fn().mockResolvedValue({
        result: { thread: { id: "thread-other" } },
        notifications: [],
      }),
    };
    const adapter = createAgentRuntimeAdapter({ client });
    const consumeReplay = vi.fn();

    await expect(
      adapter.resumeThread("thread-requested", consumeReplay),
    ).resolves.toBe(false);
    expect(consumeReplay).not.toHaveBeenCalled();
  });

  it("getSession 应合并同一会话同一请求形状的并发读取", async () => {
    let resolveSession!: (value: { id: string; messages: unknown[] }) => void;
    const sessionPromise = new Promise<{ id: string; messages: unknown[] }>(
      (resolve) => {
        resolveSession = resolve;
      },
    );
    const client = {
      ...mockRuntimeClient,
      getAgentRuntimeSession: vi.fn().mockReturnValue(sessionPromise),
    };
    const adapter = createAgentRuntimeAdapter({
      client,
    });

    const first = adapter.getSession("session-9", {
      historyLimit: 40,
      source: "runtimeSync.poll",
    });
    const second = adapter.getSession("session-9", {
      historyLimit: 40,
      source: "switchTopic.direct",
    });

    expect(client.getAgentRuntimeSession).toHaveBeenCalledTimes(1);

    resolveSession({ id: "session-9", messages: [] });

    await expect(first).resolves.toMatchObject({ id: "session-9" });
    await expect(second).resolves.toMatchObject({ id: "session-9" });
  });

  it("详情与 read model 应共享默认 40 条历史窗口读取", async () => {
    let resolveSession!: (value: {
      id: string;
      messages: unknown[];
      thread_id?: string;
    }) => void;
    const sessionPromise = new Promise<{
      id: string;
      messages: unknown[];
      thread_id?: string;
    }>((resolve) => {
      resolveSession = resolve;
    });
    const client = {
      ...mockRuntimeClient,
      getAgentRuntimeSession: vi.fn().mockReturnValue(sessionPromise),
    };
    const adapter = createAgentRuntimeAdapter({ client });

    const detail = adapter.getSession("session-9", { historyLimit: 40 });
    vi.mocked(client.getAgentRuntimeThreadRead).mockResolvedValue({
      thread_id: "thread-9",
    });
    const readModel = adapter.getSessionReadModel("session-9");

    expect(client.getAgentRuntimeSession).toHaveBeenCalledTimes(1);

    resolveSession({
      id: "session-9",
      thread_id: "thread-9",
      messages: [],
    });
    await expect(detail).resolves.toMatchObject({ id: "session-9" });
    await expect(readModel).resolves.toMatchObject({
      thread_id: "thread-9",
    });
    expect(client.getAgentRuntimeThreadRead).toHaveBeenCalledWith("thread-9");
  });

  it("getSession 不应合并不同历史窗口请求，完成后下一次应重新读取", async () => {
    const client = {
      ...mockRuntimeClient,
      getAgentRuntimeSession: vi
        .fn()
        .mockResolvedValueOnce({ id: "session-9", messages: ["latest"] })
        .mockResolvedValueOnce({ id: "session-9", messages: ["older"] })
        .mockResolvedValueOnce({ id: "session-9", messages: ["fresh"] }),
    };
    const adapter = createAgentRuntimeAdapter({
      client,
    });

    await Promise.all([
      adapter.getSession("session-9", { historyLimit: 40 }),
      adapter.getSession("session-9", {
        historyLimit: 50,
        historyItemCursor: "item-page-2",
        historyTurnCursor: "turn-page-2",
      }),
    ]);
    await adapter.getSession("session-9", { historyLimit: 40 });

    expect(client.getAgentRuntimeSession).toHaveBeenCalledTimes(3);
    expect(client.getAgentRuntimeSession).toHaveBeenNthCalledWith(
      1,
      "session-9",
      { historyLimit: 40 },
    );
    expect(client.getAgentRuntimeSession).toHaveBeenNthCalledWith(
      2,
      "session-9",
      {
        historyLimit: 50,
        historyItemCursor: "item-page-2",
        historyTurnCursor: "turn-page-2",
      },
    );
    expect(client.getAgentRuntimeSession).toHaveBeenNthCalledWith(
      3,
      "session-9",
      { historyLimit: 40 },
    );
  });

  it("getSession 不应合并不同 opaque Item/Turn cursor 请求", async () => {
    const client = {
      ...mockRuntimeClient,
      getAgentRuntimeSession: vi
        .fn()
        .mockResolvedValueOnce({ id: "session-9", messages: ["page-a"] })
        .mockResolvedValueOnce({ id: "session-9", messages: ["page-b"] }),
    };
    const adapter = createAgentRuntimeAdapter({ client });

    await adapter.getSession("session-9", {
      historyLimit: 50,
      historyItemCursor: "item-a",
      historyTurnCursor: "turn-a",
    });
    await adapter.getSession("session-9", {
      historyLimit: 50,
      historyItemCursor: "item-b",
      historyTurnCursor: "turn-b",
    });

    expect(client.getAgentRuntimeSession).toHaveBeenCalledTimes(2);
  });

  it("generateSessionTitle 应透传标题预览文本", async () => {
    const client = {
      ...mockRuntimeClient,
      generateAgentRuntimeSessionTitle: vi.fn().mockResolvedValue("新标题"),
    };
    const adapter = createAgentRuntimeAdapter({
      client,
    });

    await expect(
      adapter.generateSessionTitle?.("session-9", "user：请整理支付异常"),
    ).resolves.toBe("新标题");

    expect(client.generateAgentRuntimeSessionTitle).toHaveBeenCalledWith(
      "session-9",
      "user：请整理支付异常",
    );
  });

  it("updateSessionMetadata 应把多个会话元数据合并成一次更新", async () => {
    const client = {
      ...mockRuntimeClient,
      getAgentRuntimeSession: vi.fn().mockResolvedValue({
        id: "session-9",
        thread_id: "thread-9",
        messages: [],
      }),
      updateAgentRuntimeThreadSettings: vi.fn().mockResolvedValue({
        result: {},
        notifications: [],
      }),
    };
    const adapter = createAgentRuntimeAdapter({
      client,
    });

    await adapter.updateSessionMetadata?.("session-9", {
      accessMode: "full-access",
      providerType: "openai",
      model: "gpt-5.4-mini",
      executionStrategy: "react",
    });

    expect(client.updateAgentRuntimeThreadSettings).toHaveBeenCalledTimes(1);
    expect(client.updateAgentRuntimeThreadSettings).toHaveBeenCalledWith({
      threadId: "thread-9",
      approvalPolicy: "never",
      sandboxPolicy: "danger-full-access",
      modelProvider: "openai",
      model: "gpt-5.4-mini",
    });
  });

  it("setSessionProviderSelection 应原子提交 model/provider/effort 到 current thread", async () => {
    const client = {
      ...mockRuntimeClient,
      getAgentRuntimeSession: vi.fn().mockResolvedValue({
        id: "session-9",
        thread_id: "thread-9",
        messages: [],
      }),
      updateAgentRuntimeThreadSettings: vi.fn().mockResolvedValue({
        result: {},
        notifications: [],
      }),
    };
    const adapter = createAgentRuntimeAdapter({ client });

    await adapter.setSessionProviderSelection(
      "session-9",
      "xai",
      "grok-4.5",
      "xhigh",
    );

    expect(client.updateAgentRuntimeThreadSettings).toHaveBeenCalledWith({
      threadId: "thread-9",
      modelProvider: "xai",
      model: "grok-4.5",
      effort: "xhigh",
    });
  });
});
