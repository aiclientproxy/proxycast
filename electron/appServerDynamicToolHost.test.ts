import { describe, expect, it, vi } from "vitest";
import type {
  JsonRpcMessage,
  JsonRpcRequest,
} from "@limecloud/app-server-client";

vi.mock("./electronRuntime", () => ({
  app: {
    getLocale: () => "en-US",
    getName: () => "Lime",
    getVersion: () => "1.116.0",
  },
}));

import { AppServerDynamicToolHost } from "./appServerDynamicToolHost";

function connection() {
  return {
    respondServerRequest: vi.fn(),
    rejectServerRequest: vi.fn(),
  };
}

function call(overrides: Record<string, unknown> = {}): JsonRpcMessage {
  return {
    id: "request-1",
    method: "item/tool/call",
    params: {
      threadId: "thread-1",
      turnId: "turn-1",
      callId: "call-1",
      namespace: "desktop",
      tool: "appInfo",
      arguments: {},
      ...overrides,
    },
  };
}

describe("AppServerDynamicToolHost", () => {
  it("overrides untrusted thread dynamicTools with the immutable desktop registry", () => {
    const host = new AppServerDynamicToolHost();
    const prepared = host.prepareClientRequest({
      id: "thread-start-1",
      method: "thread/start",
      params: {
        dynamicTools: [
          {
            type: "function",
            name: "forged",
            description: "forged",
            inputSchema: { type: "object" },
          },
        ],
      },
    } as JsonRpcRequest);

    const dynamicTools = (
      prepared.params as {
        dynamicTools: Array<{
          name: string;
          tools: Array<{ name: string; inputSchema: unknown }>;
        }>;
      }
    ).dynamicTools;
    expect(dynamicTools.map(({ name }) => name)).toEqual([
      "desktop",
      "browser",
    ]);
    expect(dynamicTools[0]?.tools[0]).toMatchObject({
      name: "appInfo",
      inputSchema: {
        type: "object",
        additionalProperties: false,
      },
    });
    expect(dynamicTools[1]?.tools.map(({ name }) => name)).toEqual(
      expect.arrayContaining(["openTabs", "claimTab", "observe"]),
    );
    expect(JSON.stringify(prepared.params)).not.toContain("forged");
  });

  it("executes one exact bound call and keeps the request inside Electron", async () => {
    const host = new AppServerDynamicToolHost(() => ({
      locale: "zh-CN",
      name: "Lime",
      platform: "darwin",
      version: "1.116.0",
    }));
    const transport = connection();
    host.observeClientResult("thread/start", { thread: { id: "thread-1" } });

    expect(await host.tryHandle(transport, call())).toBe(true);
    expect(transport.respondServerRequest).toHaveBeenCalledWith("request-1", {
      contentItems: [
        {
          type: "inputText",
          text: JSON.stringify({
            locale: "zh-CN",
            name: "Lime",
            platform: "darwin",
            version: "1.116.0",
          }),
        },
      ],
      success: true,
    });
    expect(transport.rejectServerRequest).not.toHaveBeenCalled();

    expect(await host.tryHandle(transport, call())).toBe(true);
    expect(transport.rejectServerRequest).toHaveBeenCalledWith("request-1", {
      code: -32602,
      message: "item/tool/call identity was already consumed",
    });
  });

  it("fails unknown binding and schema drift closed", async () => {
    const host = new AppServerDynamicToolHost();
    const transport = connection();
    host.observeClientResult("thread/resume", { thread: { id: "thread-1" } });

    expect(
      await host.tryHandle(transport, call({ threadId: "other-thread" })),
    ).toBe(true);
    expect(transport.rejectServerRequest).toHaveBeenLastCalledWith(
      "request-1",
      expect.objectContaining({
        code: -32602,
        message: expect.stringContaining("frozen host capability binding"),
      }),
    );

    expect(
      await host.tryHandle(
        transport,
        call({ callId: "call-2", arguments: { path: "/tmp" } }),
      ),
    ).toBe(true);
    expect(transport.rejectServerRequest).toHaveBeenLastCalledWith(
      "request-1",
      expect.objectContaining({
        code: -32602,
        message: expect.stringContaining("does not accept arguments"),
      }),
    );
  });

  it("routes Browser calls through the WebContents owner bound to the thread", async () => {
    const browserHost = {
      executeTool: vi.fn(async () => ({
        status: "completed" as const,
        data: [],
      })),
      turnEnded: vi.fn(),
    };
    const host = new AppServerDynamicToolHost(undefined, browserHost as never);
    const transport = connection();
    host.observeClientResult(
      "thread/start",
      { thread: { id: "thread-1" } },
      { ownerWebContentsId: 41 },
    );

    expect(
      await host.tryHandle(
        transport,
        call({ namespace: "browser", tool: "openTabs" }),
      ),
    ).toBe(true);
    expect(browserHost.executeTool).toHaveBeenCalledWith({
      arguments: {},
      callId: "call-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "openTabs",
      turnId: "turn-1",
    });
    expect(transport.respondServerRequest).toHaveBeenCalledWith(
      "request-1",
      expect.objectContaining({ success: true }),
    );

    host.observeServerMessage({
      method: "turn/completed",
      params: {
        threadId: "thread-1",
        turn: { id: "turn-1", status: "completed" },
      },
    });
    expect(browserHost.turnEnded).toHaveBeenCalledWith("thread-1", "turn-1");
  });

  it.each(["failed", "interrupted", "canceled"])(
    "cleans Browser control after a %s terminal turn",
    (status) => {
      const browserHost = {
        executeTool: vi.fn(async () => ({ status: "completed" as const })),
        turnEnded: vi.fn(),
      };
      const host = new AppServerDynamicToolHost(
        undefined,
        browserHost as never,
      );

      host.observeServerMessage({
        method: "turn/completed",
        params: {
          threadId: "thread-1",
          turn: { id: "turn-1", status },
        },
      });

      expect(browserHost.turnEnded).toHaveBeenCalledWith("thread-1", "turn-1");
    },
  );

  it("does not release Browser control for a non-terminal turn status", () => {
    const browserHost = {
      executeTool: vi.fn(async () => ({ status: "completed" as const })),
      turnEnded: vi.fn(),
    };
    const host = new AppServerDynamicToolHost(undefined, browserHost as never);

    host.observeServerMessage({
      method: "turn/completed",
      params: {
        threadId: "thread-1",
        turn: { id: "turn-1", status: "inProgress" },
      },
    });

    expect(browserHost.turnEnded).not.toHaveBeenCalled();
  });

  it("accepts revision zero for a complete Browser claim snapshot", async () => {
    const browserHost = {
      executeTool: vi.fn(async () => ({ status: "completed" as const })),
      turnEnded: vi.fn(),
    };
    const host = new AppServerDynamicToolHost(undefined, browserHost as never);
    const transport = connection();
    host.observeClientResult(
      "thread/start",
      { thread: { id: "thread-1" } },
      { ownerWebContentsId: 41 },
    );

    await host.tryHandle(
      transport,
      call({
        namespace: "browser",
        tool: "claimTab",
        arguments: {
          pageRevision: 0,
          tabId: "tab-1",
          title: "Example",
          url: "https://example.com/",
        },
      }),
    );

    expect(browserHost.executeTool).toHaveBeenCalledWith(
      expect.objectContaining({
        arguments: expect.objectContaining({ pageRevision: 0 }),
        tool: "claimTab",
      }),
    );
    expect(transport.rejectServerRequest).not.toHaveBeenCalled();
  });

  it("fails Browser calls closed when the thread has no desktop owner", async () => {
    const browserHost = {
      executeTool: vi.fn(),
      turnEnded: vi.fn(),
    };
    const host = new AppServerDynamicToolHost(undefined, browserHost as never);
    const transport = connection();
    host.observeClientResult("thread/resume", {
      thread: { id: "thread-1" },
    });

    await host.tryHandle(
      transport,
      call({ namespace: "browser", tool: "openTabs" }),
    );

    expect(browserHost.executeTool).not.toHaveBeenCalled();
    expect(transport.respondServerRequest).toHaveBeenCalledWith(
      "request-1",
      expect.objectContaining({
        contentItems: [
          expect.objectContaining({
            text: expect.stringContaining("not bound to this desktop thread"),
          }),
        ],
        success: false,
      }),
    );
  });
});
