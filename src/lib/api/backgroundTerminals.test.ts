import { describe, expect, it, vi } from "vitest";
import {
  terminateBackgroundTerminalForItem,
  type BackgroundTerminalsAppServerClient,
} from "./backgroundTerminals";

function appServerResult<T>(result: T) {
  return {
    id: 1,
    result,
    response: { jsonrpc: "2.0" as const, id: 1, result },
    notifications: [],
    messages: [],
    configWarnings: [],
  };
}

function clientMock(): BackgroundTerminalsAppServerClient {
  return {
    listThreadBackgroundTerminals: vi.fn(),
    terminateThreadBackgroundTerminal: vi.fn(),
  };
}

describe("backgroundTerminals API", () => {
  it("按 Thread 和 item 解析 canonical processId 后终止后台终端", async () => {
    const client = clientMock();
    vi.mocked(client.listThreadBackgroundTerminals).mockResolvedValueOnce(
      appServerResult({
        data: [
          {
            itemId: "item-1",
            processId: "7",
            command: "npm test",
            cwd: "/workspace",
          },
        ],
        nextCursor: null,
      }),
    );
    vi.mocked(client.terminateThreadBackgroundTerminal).mockResolvedValueOnce(
      appServerResult({ terminated: true }),
    );

    await expect(
      terminateBackgroundTerminalForItem(
        { threadId: " thread-1 ", itemId: " item-1 " },
        client,
      ),
    ).resolves.toEqual({ terminated: true });
    expect(client.listThreadBackgroundTerminals).toHaveBeenCalledWith({
      threadId: "thread-1",
      cursor: null,
      limit: 100,
    });
    expect(client.terminateThreadBackgroundTerminal).toHaveBeenCalledWith({
      threadId: "thread-1",
      processId: "7",
    });
  });

  it("遍历分页后找不到 item 时不发终止请求", async () => {
    const client = clientMock();
    vi.mocked(client.listThreadBackgroundTerminals)
      .mockResolvedValueOnce(
        appServerResult({ data: [], nextCursor: "5" }),
      )
      .mockResolvedValueOnce(appServerResult({ data: [], nextCursor: null }));

    await expect(
      terminateBackgroundTerminalForItem(
        { threadId: "thread-1", itemId: "missing" },
        client,
      ),
    ).resolves.toEqual({ terminated: false });
    expect(client.listThreadBackgroundTerminals).toHaveBeenLastCalledWith({
      threadId: "thread-1",
      cursor: "5",
      limit: 100,
    });
    expect(client.terminateThreadBackgroundTerminal).not.toHaveBeenCalled();
  });

  it("空 identity fail closed", async () => {
    const client = clientMock();

    await expect(
      terminateBackgroundTerminalForItem(
        { threadId: " ", itemId: "item-1" },
        client,
      ),
    ).rejects.toThrow("background terminal threadId is required");
    expect(client.listThreadBackgroundTerminals).not.toHaveBeenCalled();
  });
});
