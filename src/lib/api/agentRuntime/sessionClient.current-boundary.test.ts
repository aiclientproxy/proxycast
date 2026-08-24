import { describe, expect, it, vi } from "vitest";
import type { AppServerSessionRpcClient } from "./appServerSessionClient";
import {
  AGENT_RUNTIME_SESSIONS_CHANGED_EVENT,
  createSessionClient,
} from "./sessionClient";

function appServerClientMock(): AppServerSessionRpcClient {
  const canonicalThread = (sessionId: string, threadId: string) => ({
    cliVersion: "0.1.0",
    createdAt: 1_780_704_000,
    cwd: "/tmp/workspace-1",
    ephemeral: false,
    id: threadId,
    modelProvider: "openai-compatible",
    preview: sessionId,
    sessionId,
    source: "appServer",
    status: { type: "idle" },
    turns: [],
    updatedAt: 1_780_704_000,
  });
  return {
    startSession: vi.fn(),
    readThread: vi.fn(),
    updateThreadSettings: vi.fn().mockResolvedValue({
      id: 1,
      result: {},
      response: { id: 1, result: {} },
      notifications: [],
      messages: [],
    }),
    request: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        data: [
          canonicalThread("session-recent", "thread-recent"),
          canonicalThread("session-archived", "thread-archived"),
          canonicalThread("session-bulk", "thread-bulk"),
          canonicalThread("session-created", "thread-created"),
          canonicalThread("session-deleted", "thread-deleted"),
        ],
      },
      response: { id: 1, result: {} },
      notifications: [],
      messages: [],
    }),
    archiveThread: vi.fn().mockResolvedValue({
      id: 2,
      result: {},
      response: { id: 2, result: {} },
      notifications: [],
      messages: [],
    }),
    forkThread: vi.fn().mockResolvedValue({
      id: 2,
      result: {
        thread: canonicalThread("session-forked", "thread-forked"),
      },
      response: { id: 2, result: {} },
      notifications: [],
      messages: [],
    }),
    unarchiveThread: vi.fn().mockResolvedValue({
      id: 3,
      result: {
        thread: canonicalThread("session-created", "thread-created"),
      },
      response: { id: 3, result: {} },
      notifications: [],
      messages: [],
    }),
    deleteThread: vi.fn().mockResolvedValue({
      id: 3,
      result: {},
      response: { id: 3, result: {} },
      notifications: [],
      messages: [],
    }),
  };
}

describe("agentRuntime sessionClient current App Server boundary", () => {
  it("delete 使用 canonical thread/delete，不承载 metadata 更新", async () => {
    const appServerClient = appServerClientMock();
    const client = createSessionClient({
      appServerClient,
    });

    await expect(
      client.deleteAgentRuntimeSession(" session-deleted "),
    ).resolves.toBeUndefined();

    expect(appServerClient.deleteThread).toHaveBeenCalledWith({
      threadId: "thread-deleted",
    });
    expect(appServerClient.request).toHaveBeenCalledWith("thread/list", {
      archived: false,
      limit: 100,
    });
  });

  it("archive projection must use thread/archive instead of session update", async () => {
    const appServerClient = appServerClientMock();
    const client = createSessionClient({
      appServerClient,
    });

    await expect(
      client.archiveAgentRuntimeSession(" session-bulk "),
    ).resolves.toBeUndefined();

    expect(appServerClient.archiveThread).toHaveBeenCalledWith({
      threadId: "thread-bulk",
    });
  });

  it("unarchive projection must use thread/unarchive", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.unarchiveThread).mockResolvedValueOnce({
      id: 3,
      result: {
        thread: {
          cliVersion: "0.1.0",
          createdAt: 1_780_704_000,
          cwd: "/tmp/workspace-1",
          ephemeral: false,
          id: "thread-archived",
          modelProvider: "openai-compatible",
          preview: "session-archived",
          sessionId: "session-archived",
          source: "appServer",
          status: { type: "idle" },
          turns: [],
          updatedAt: 1_780_704_000,
        },
      },
      response: { id: 3, result: {} },
      notifications: [],
      messages: [],
    });
    const client = createSessionClient({ appServerClient });

    await expect(
      client.unarchiveAgentRuntimeSession("session-archived"),
    ).resolves.toBeUndefined();

    expect(appServerClient.unarchiveThread).toHaveBeenCalledWith({
      threadId: "thread-archived",
    });
  });

  it("delete projection must resolve canonical id and use thread/delete", async () => {
    const appServerClient = appServerClientMock();
    const client = createSessionClient({
      appServerClient,
    });

    await expect(
      client.deleteAgentRuntimeSession(" session-deleted "),
    ).resolves.toBeUndefined();

    expect(appServerClient.deleteThread).toHaveBeenCalledWith({
      threadId: "thread-deleted",
    });
    expect(appServerClient.request).toHaveBeenCalledWith("thread/list", {
      archived: false,
      limit: 100,
    });
  });

  it("session mutation should notify current GUI session-list subscribers", async () => {
    const appServerClient = appServerClientMock();
    const sessionStartResult = {
      thread: {
        id: "thread-created",
        sessionId: "session-created",
      },
    };
    vi.mocked(appServerClient.startSession).mockResolvedValueOnce({
      id: 1,
      result: sessionStartResult,
      response: { id: 1, result: sessionStartResult },
      notifications: [],
      messages: [],
    });
    const client = createSessionClient({
      appServerClient,
    });
    const listener = vi.fn();
    window.addEventListener(AGENT_RUNTIME_SESSIONS_CHANGED_EVENT, listener);

    try {
      await client.createAgentRuntimeSession(
        "workspace-1",
        "新会话",
        undefined,
        {
          metadata: {
            providerSelector: "fixture-provider",
            modelName: "fixture-model",
          },
        },
      );
      await client.archiveAgentRuntimeSession("session-created");
      await client.unarchiveAgentRuntimeSession("session-created");
      await client.deleteAgentRuntimeSession("session-created");
    } finally {
      window.removeEventListener(
        AGENT_RUNTIME_SESSIONS_CHANGED_EVENT,
        listener,
      );
    }

    expect(listener).toHaveBeenCalledTimes(4);
    expect(
      listener.mock.calls.map(([event]) =>
        event instanceof CustomEvent ? event.detail.reason : null,
      ),
    ).toEqual(["created", "archived", "unarchived", "deleted"]);
  });

  it("cwd-only session create event must not publish an empty legacy workspaceId", async () => {
    const appServerClient = appServerClientMock();
    const sessionStartResult = {
      thread: {
        id: "thread-cwd",
        sessionId: "session-cwd",
      },
    };
    vi.mocked(appServerClient.startSession).mockResolvedValueOnce({
      id: 1,
      result: sessionStartResult,
      response: { id: 1, result: sessionStartResult },
      notifications: [],
      messages: [],
    });
    const client = createSessionClient({
      appServerClient,
    });
    const listener = vi.fn();
    window.addEventListener(AGENT_RUNTIME_SESSIONS_CHANGED_EVENT, listener);

    try {
      await client.createAgentRuntimeSession(" ", "空项目对话", undefined, {
        workingDir: "/repo/skill-think",
        metadata: {
          providerSelector: "fixture-provider",
          modelName: "fixture-model",
        },
      });
    } finally {
      window.removeEventListener(
        AGENT_RUNTIME_SESSIONS_CHANGED_EVENT,
        listener,
      );
    }

    expect(listener).toHaveBeenCalledTimes(1);
    const event = listener.mock.calls[0]?.[0];
    expect(event).toBeInstanceOf(CustomEvent);
    expect((event as CustomEvent).detail).toEqual({
      reason: "created",
      sessionId: "session-cwd",
      workspaceId: undefined,
    });
  });
});
