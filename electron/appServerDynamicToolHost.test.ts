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

    expect(prepared.params).toMatchObject({
      dynamicTools: [
        {
          type: "namespace",
          name: "desktop",
          tools: [
            {
              type: "function",
              name: "appInfo",
              inputSchema: {
                type: "object",
                additionalProperties: false,
              },
            },
          ],
        },
      ],
    });
    expect(JSON.stringify(prepared.params)).not.toContain("forged");
  });

  it("executes one exact bound call and keeps the request inside Electron", () => {
    const host = new AppServerDynamicToolHost(() => ({
      locale: "zh-CN",
      name: "Lime",
      platform: "darwin",
      version: "1.116.0",
    }));
    const transport = connection();
    host.observeClientResult("thread/start", { thread: { id: "thread-1" } });

    expect(host.tryHandle(transport, call())).toBe(true);
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

    expect(host.tryHandle(transport, call())).toBe(true);
    expect(transport.rejectServerRequest).toHaveBeenCalledWith("request-1", {
      code: -32602,
      message: "item/tool/call identity was already consumed",
    });
  });

  it("fails unknown binding and schema drift closed", () => {
    const host = new AppServerDynamicToolHost();
    const transport = connection();
    host.observeClientResult("thread/resume", { thread: { id: "thread-1" } });

    expect(host.tryHandle(transport, call({ threadId: "other-thread" }))).toBe(
      true,
    );
    expect(transport.rejectServerRequest).toHaveBeenLastCalledWith(
      "request-1",
      expect.objectContaining({
        code: -32602,
        message: expect.stringContaining("frozen host capability binding"),
      }),
    );

    expect(
      host.tryHandle(transport, call({ arguments: { path: "/tmp" } })),
    ).toBe(true);
    expect(transport.rejectServerRequest).toHaveBeenLastCalledWith(
      "request-1",
      expect.objectContaining({
        code: -32602,
        message: expect.stringContaining("does not accept arguments"),
      }),
    );
  });
});
