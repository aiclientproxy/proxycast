import { describe, expect, it, vi } from "vitest";
import { createAppServerReadModelClient } from "./appServerReadModelClient";
import type { AppServerSessionReadClient } from "./appServerReadModelClient";

function appServerClientMock(): AppServerSessionReadClient {
  return {
    readThread: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        thread: {
          id: "thread-1",
          sessionId: "session-1",
          cwd: "/tmp/workspace",
          modelProvider: "openai",
          status: { type: "active", activeFlags: [] },
          createdAt: Date.parse("2026-06-06T00:00:00.000Z") / 1000,
          updatedAt: Date.parse("2026-06-06T00:00:02.000Z") / 1000,
          turns: [
            {
              id: "turn-1",
              status: "inProgress",
              startedAt: Date.parse("2026-06-06T00:00:01.000Z") / 1000,
              items: [
                {
                  id: "file-change-1",
                  type: "fileChange",
                  status: "completed",
                  changes: [
                    {
                      path: "src/App.tsx",
                      kind: { type: "update" },
                      diff: "+export const ready = true;",
                    },
                  ],
                },
              ],
            },
          ],
        },
      },
      response: {
        id: 1,
        result: {} as never,
      },
      notifications: [],
      messages: [],
    }),
  };
}

describe("appServerReadModelClient", () => {
  it("应通过 thread/read 读取并投影 thread read model", async () => {
    const appServerClient = appServerClientMock();
    const client = createAppServerReadModelClient({ appServerClient });

    const readModel = await client.getAgentRuntimeThreadRead(" session-1 ");
    expect(readModel).toMatchObject({
      thread_id: "thread-1",
      status: "running",
      profile_status: "running",
      turns: [
        {
          turn_id: "turn-1",
          status: "running",
          native_status: "running",
        },
      ],
    });
    expect(readModel.active_turn_id).toBe("turn-1");
    expect(readModel.thread_items).toEqual([
      expect.objectContaining({
        id: "file-change-1",
        type: "patch",
        paths: ["src/App.tsx"],
      }),
    ]);

    expect(appServerClient.readThread).toHaveBeenLastCalledWith({
      threadId: "session-1",
      includeTurns: true,
    });
  });

  it("缺少 threadId 时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    const client = createAppServerReadModelClient({ appServerClient });

    await expect(client.getAgentRuntimeThreadRead(" ")).rejects.toThrow(
      "threadId is required to read canonical App Server thread",
    );

    expect(appServerClient.readThread).not.toHaveBeenCalled();
  });

  it("thread/read 返回假成功 envelope 时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce({
      id: 1,
      result: { success: true } as never,
      response: {
        id: 1,
        result: {} as never,
      },
      notifications: [],
      messages: [],
    });
    const client = createAppServerReadModelClient({ appServerClient });

    await expect(client.getAgentRuntimeThreadRead("session-1")).rejects.toThrow(
      "thread/read did not return canonical thread read model",
    );
  });

  it("thread/read 返回错误 turn 状态时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.readThread).mockResolvedValueOnce({
      id: 1,
      result: {
        thread: {
          id: "thread-1",
          sessionId: "session-1",
          createdAt: Date.parse("2026-06-06T00:00:00.000Z") / 1000,
          updatedAt: Date.parse("2026-06-06T00:00:02.000Z") / 1000,
          status: { type: "active", activeFlags: [] },
          turns: [{ status: "almost_done" }],
        },
      } as never,
      response: {
        id: 1,
        result: {} as never,
      },
      notifications: [],
      messages: [],
    });
    const client = createAppServerReadModelClient({ appServerClient });

    await expect(client.getAgentRuntimeThreadRead("session-1")).rejects.toThrow(
      "thread/read did not return canonical thread read model",
    );
  });
});
