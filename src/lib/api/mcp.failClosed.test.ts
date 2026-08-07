import { beforeEach, describe, expect, it, vi } from "vitest";
import { safeInvoke } from "@/lib/dev-bridge";

import { mcpApi, type McpServer } from "./mcp";

const appServerRequestMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api/appServer", () => ({
  AppServerClient: vi.fn().mockImplementation(() => ({
    request: appServerRequestMock,
  })),
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(),
}));

const server: McpServer = {
  id: "server-1",
  name: "docs",
  server_config: {
    command: "node",
    args: ["server.js"],
  },
  enabled_lime: true,
  enabled_claude: false,
  enabled_codex: true,
  enabled_gemini: false,
};

function mockAppServerResult(result: unknown): void {
  appServerRequestMock.mockResolvedValueOnce({ result });
}

describe("mcp App Server current API fail-closed", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    appServerRequestMock.mockReset();
  });

  it("MCP CRUD / import / sync 在 App Server 失败时应 fail closed 且不回退 legacy", async () => {
    const appServerError = new Error("App Server unavailable");
    const cases: Array<[string, unknown, () => Promise<unknown>]> = [
      ["mcpServer/create", { server }, () => mcpApi.addServer(server)],
      ["mcpServer/update", { server }, () => mcpApi.updateServer(server)],
      [
        "mcpServer/delete",
        { id: "server-1" },
        () => mcpApi.deleteServer("server-1"),
      ],
      [
        "mcpServer/enabled/set",
        { id: "server-1", appType: "codex", enabled: true },
        () => mcpApi.toggleServer("server-1", "codex", true),
      ],
      [
        "mcpServer/importFromApp",
        { appType: "codex" },
        () => mcpApi.importFromApp("codex"),
      ],
      ["mcpServer/syncAllToLive", {}, () => mcpApi.syncAllToLive()],
    ];

    for (const [method, params, action] of cases) {
      appServerRequestMock.mockRejectedValueOnce(appServerError);

      await expect(action()).rejects.toThrow("App Server unavailable");
      expect(appServerRequestMock).toHaveBeenLastCalledWith(method, params);
    }
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("MCP CRUD / import / sync 收到 malformed current 响应时应 fail closed", async () => {
    const cases: Array<[string, unknown, () => Promise<unknown>, string]> = [
      [
        "mcpServer/create",
        {},
        () => mcpApi.addServer(server),
        "mcpServer/create did not return servers",
      ],
      [
        "mcpServer/update",
        { success: true },
        () => mcpApi.updateServer(server),
        "mcpServer/update did not return servers",
      ],
      [
        "mcpServer/delete",
        { error: "failed" },
        () => mcpApi.deleteServer("server-1"),
        "mcpServer/delete did not return servers",
      ],
      [
        "mcpServer/enabled/set",
        { servers: null },
        () => mcpApi.toggleServer("server-1", "codex", true),
        "mcpServer/enabled/set did not return servers",
      ],
      [
        "mcpServer/importFromApp",
        { importedCount: 1 },
        () => mcpApi.importFromApp("codex"),
        "mcpServer/importFromApp did not return servers",
      ],
      [
        "mcpServer/syncAllToLive",
        {},
        () => mcpApi.syncAllToLive(),
        "mcpServer/syncAllToLive did not return servers",
      ],
    ];

    for (const [method, result, action, message] of cases) {
      mockAppServerResult(result);

      await expect(action()).rejects.toThrow(message);
      expect(appServerRequestMock).toHaveBeenLastCalledWith(
        method,
        expect.any(Object),
      );
    }
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("MCP lifecycle 与 runtime 在 App Server 失败时应 fail closed 且不回退 legacy", async () => {
    const appServerError = new Error("App Server unavailable");
    const cases: Array<[string, () => Promise<unknown>]> = [
      ["mcpServer/start", () => mcpApi.startServer("docs")],
      ["mcpServer/stop", () => mcpApi.stopServer("docs")],
      ["mcpServer/oauth/login", () => mcpApi.loginOAuthServer("docs")],
      [
        "mcpTool/listForContext",
        () => mcpApi.listToolsForContext("assistant", true),
      ],
      ["mcpTool/search", () => mcpApi.searchTools("docs", "tool_search", 5)],
      [
        "mcpServer/tool/call",
        () =>
          mcpApi.callServerTool({
            threadId: "thread-1",
            server: "docs",
            tool: "search",
          }),
      ],
      ["mcpPrompt/get", () => mcpApi.getPrompt("docs", "summarize", {})],
      [
        "mcpServer/resource/read",
        () => mcpApi.readResource("docs", "docs://readme"),
      ],
      [
        "mcpResource/subscribe",
        () => mcpApi.subscribeResource("docs", "docs://readme"),
      ],
      [
        "mcpResource/unsubscribe",
        () => mcpApi.unsubscribeResource("docs", "docs://readme"),
      ],
    ];

    for (const [method, action] of cases) {
      appServerRequestMock.mockRejectedValueOnce(appServerError);

      await expect(action()).rejects.toThrow("App Server unavailable");
      expect(appServerRequestMock).toHaveBeenLastCalledWith(
        method,
        expect.any(Object),
      );
    }
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("MCP lifecycle 收到 mock-like current 响应时应 fail closed", async () => {
    mockAppServerResult({ success: true });
    mockAppServerResult({ error: "failed" });

    await expect(mcpApi.startServer("docs")).rejects.toThrow(
      "mcpServer/start did not return empty lifecycle result",
    );
    await expect(mcpApi.stopServer("docs")).rejects.toThrow(
      "mcpServer/stop did not return empty lifecycle result",
    );
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("MCP OAuth login 收到 malformed current 响应时应 fail closed", async () => {
    mockAppServerResult({ authorization_url: "http://127.0.0.1/callback" });

    await expect(mcpApi.loginOAuthServer("docs")).rejects.toThrow(
      "mcpServer/oauth/login did not return OAuth login response",
    );
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServer/oauth/login",
      {
        name: "docs",
      },
    );
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("MCP runtime usage 收到 malformed current 响应时应 fail closed", async () => {
    const cases: Array<[unknown, () => Promise<unknown>, string]> = [
      [
        { content: [{ type: "image", data: "aW1hZ2U=" }], isError: false },
        () =>
          mcpApi.callServerTool({
            threadId: "thread-1",
            server: "docs",
            tool: "search",
          }),
        "mcpServer/tool/call did not return canonical MCP content",
      ],
      [
        { messages: [{ role: "user", content: { type: "text" } }] },
        () => mcpApi.getPrompt("docs", "summarize", {}),
        "mcpPrompt/get did not return prompt result",
      ],
      [
        { contents: [{ mimeType: "text/plain", text: "README" }] },
        () => mcpApi.readResource("docs", "docs://readme"),
        "mcpServer/resource/read did not return resource URI",
      ],
      [
        { ok: true },
        () => mcpApi.subscribeResource("docs", "docs://readme"),
        "mcpResource/subscribe did not return empty result",
      ],
      [
        { ok: true },
        () => mcpApi.unsubscribeResource("docs", "docs://readme"),
        "mcpResource/unsubscribe did not return empty result",
      ],
    ];

    for (const [result, action, message] of cases) {
      mockAppServerResult(result);

      await expect(action()).rejects.toThrow(message);
    }
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("MCP prepare requests 应拒绝非 candidate、未知方法和 malformed params", async () => {
    await expect(
      mcpApi.executePrepareRequests([
        {
          method: "mcpServer/start",
          params: { name: "docs" },
          status: "completed",
        },
      ]),
    ).rejects.toThrow("MCP prepare request must be candidate");

    await expect(
      mcpApi.executePrepareRequests([
        {
          method: "mcpServer/delete",
          params: { id: "docs" },
          status: "candidate",
        },
      ]),
    ).rejects.toThrow(
      "Unsupported MCP prepare request method: mcpServer/delete",
    );

    await expect(
      mcpApi.executePrepareRequests([
        {
          method: "mcpServer/start",
          params: {},
          status: "candidate",
        },
      ]),
    ).rejects.toThrow("mcpServer/start prepare params require name");

    expect(appServerRequestMock).not.toHaveBeenCalled();
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("MCP call proof requests 应拒绝非 candidate、未知方法、malformed params 和工具错误", async () => {
    await expect(
      mcpApi.executeCallProofRequests(
        [
          {
            method: "mcpServer/tool/call",
            params: {
              server: "docs",
              tool: "search",
              arguments: { q: "lime" },
            },
            status: "completed",
          },
        ],
        "thread-1",
      ),
    ).rejects.toThrow("MCP call proof request must be candidate");

    await expect(
      mcpApi.executeCallProofRequests(
        [
          {
            method: "mcpTool/call",
            params: {
              toolName: "mcp__docs__search",
              arguments: { q: "lime" },
            },
            status: "candidate",
          },
        ],
        "thread-1",
      ),
    ).rejects.toThrow(
      "Unsupported MCP call proof request method: mcpTool/call",
    );

    await expect(
      mcpApi.executeCallProofRequests(
        [
          {
            method: "mcpServer/tool/call",
            params: {
              server: "docs",
              tool: "search",
              arguments: "lime",
            },
            status: "candidate",
          },
        ],
        "thread-1",
      ),
    ).rejects.toThrow(
      "mcpServer/tool/call prepare params require arguments object",
    );

    mockAppServerResult({
      content: [{ type: "text", text: "failed" }],
      isError: true,
    });
    await expect(
      mcpApi.executeCallProofRequests(
        [
          {
            method: "mcpServer/tool/call",
            params: {
              server: "docs",
              tool: "search",
              arguments: { q: "lime" },
            },
            status: "candidate",
          },
        ],
        "thread-1",
      ),
    ).rejects.toThrow("MCP call proof returned tool error");

    expect(appServerRequestMock).toHaveBeenCalledTimes(1);
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServer/tool/call",
      {
        threadId: "thread-1",
        server: "docs",
        tool: "search",
        arguments: { q: "lime" },
      },
    );
    expect(safeInvoke).not.toHaveBeenCalled();
  });
});
