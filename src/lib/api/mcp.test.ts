import { beforeEach, describe, expect, it, vi } from "vitest";
import { safeInvoke } from "@/lib/dev-bridge";

import { getMcpInnerToolName, mcpApi } from "./mcp";

const appServerRequestMock = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api/appServer", () => ({
  AppServerClient: vi.fn().mockImplementation(() => ({
    request: appServerRequestMock,
  })),
}));

vi.mock("@/lib/dev-bridge", () => ({
  safeInvoke: vi.fn(),
}));

function mockAppServerResult<T>(result: T): void {
  appServerRequestMock.mockResolvedValueOnce({ result });
}

describe("mcp", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    appServerRequestMock.mockReset();
  });

  it("应在已知 server 名下提取 inner tool 名", () => {
    expect(getMcpInnerToolName("mcp__docs__search_docs", "docs")).toBe(
      "search_docs",
    );
  });

  it("应保留 inner tool 名中的双下划线片段", () => {
    expect(getMcpInnerToolName("mcp__docs__admin__search_docs", "docs")).toBe(
      "admin__search_docs",
    );
  });

  it("对非 MCP 工具名保持原样", () => {
    expect(getMcpInnerToolName("WebSearch", "docs")).toBe("WebSearch");
  });

  it("列表命令应通过 App Server current 空态返回空数组", async () => {
    const cases: Array<[string, string, () => Promise<unknown>]> = [
      ["mcpServer/list", "servers", () => mcpApi.getServers()],
      ["mcpTool/list", "tools", () => mcpApi.listTools()],
      ["mcpPrompt/list", "prompts", () => mcpApi.listPrompts()],
      ["mcpResource/list", "resources", () => mcpApi.listResources()],
    ];

    for (const [method, field, action] of cases) {
      mockAppServerResult({ [field]: [] });

      await expect(action()).resolves.toEqual([]);
      expect(appServerRequestMock).toHaveBeenLastCalledWith(method, {});
    }
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("状态列表空态应只接受 current v2 data 分页响应", async () => {
    mockAppServerResult({ data: [], nextCursor: null });
    mockAppServerResult({ servers: [] });

    await expect(mcpApi.listServersWithStatus()).resolves.toEqual([]);
    expect(appServerRequestMock).toHaveBeenNthCalledWith(
      1,
      "mcpServerStatus/list",
      {},
    );
    expect(appServerRequestMock).toHaveBeenNthCalledWith(
      2,
      "mcpServer/list",
      {},
    );
  });

  it("资源列表应保留 current resource templates 投影", async () => {
    const resource = {
      uri: "docs://readme",
      name: "README",
      description: "Readme resource",
      mime_type: "text/markdown",
      server_name: "docs",
    };
    const resourceTemplate = {
      uri_template: "docs://{path}",
      name: "docs-path",
      title: "Docs Path",
      description: "Read a docs path",
      mime_type: "text/markdown",
      server_name: "docs",
    };
    mockAppServerResult({
      resources: [resource],
      resourceTemplates: [resourceTemplate],
    });
    mockAppServerResult({
      resources: [resource],
      resourceTemplates: [resourceTemplate],
    });

    await expect(mcpApi.listResources()).resolves.toEqual([resource]);
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpResource/list",
      {},
    );

    await expect(mcpApi.listResourcesWithTemplates()).resolves.toEqual({
      resources: [resource],
      resourceTemplates: [resourceTemplate],
    });
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpResource/list",
      {},
    );
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("单资源操作应拒绝空白 target 且不发送请求", async () => {
    await expect(mcpApi.readResource(" ", "docs://readme")).rejects.toThrow(
      "server cannot be empty",
    );
    await expect(mcpApi.subscribeResource("docs", " ")).rejects.toThrow(
      "URI cannot be empty",
    );
    await expect(mcpApi.unsubscribeResource(" ", " ")).rejects.toThrow(
      "server cannot be empty",
    );

    expect(appServerRequestMock).not.toHaveBeenCalled();
  });

  it("runtime resource owner 应要求真实 thread identity", async () => {
    await expect(
      mcpApi.readResource("docs", "ui://demo/report.html", {
        threadId: " ",
      }),
    ).rejects.toThrow("threadId cannot be empty");
    await expect(
      mcpApi.readResource("docs", "ui://demo/report.html", {
        threadId: "thread-1",
        originCallId: " ",
      }),
    ).rejects.toThrow("originCallId cannot be empty");
    await expect(
      mcpApi.readResource("docs", "ui://demo/report.html", {
        threadId: "thread-1",
        connectorId: " ",
      }),
    ).rejects.toThrow("connectorId cannot be empty");

    expect(appServerRequestMock).not.toHaveBeenCalled();
  });

  it("提示词操作应拒绝空白 target 且不发送请求", async () => {
    await expect(mcpApi.getPrompt(" ", "summarize", {})).rejects.toThrow(
      "server cannot be empty",
    );
    await expect(mcpApi.getPrompt("docs", " ", {})).rejects.toThrow(
      "name cannot be empty",
    );

    expect(appServerRequestMock).not.toHaveBeenCalled();
  });

  it("旧 servers 状态响应应 fail closed", async () => {
    mockAppServerResult({ servers: [] });

    await expect(mcpApi.listServersWithStatus()).rejects.toThrow(
      "mcpServerStatus/list did not return data",
    );
    expect(appServerRequestMock).toHaveBeenCalledTimes(1);
  });

  it("current v2 状态分页应投影回配置页所需的 server identity", async () => {
    const config = {
      id: "server-v2",
      name: "remote-docs",
      server_config: {
        type: "streamable_http",
        url: "https://example.com/mcp",
        enabled: true,
      },
      enabled_lime: true,
      enabled_claude: false,
      enabled_codex: true,
      enabled_gemini: false,
    };
    mockAppServerResult({
      data: [
        {
          name: "remote-docs",
          runtimeStatus: "connected",
          pluginId: "connector-docs",
          serverInfo: {
            name: "remote-docs",
            title: null,
            version: "1.0.0",
            description: null,
            icons: null,
            websiteUrl: null,
          },
          tools: {
            search: {
              name: "search",
              description: "Search docs",
              inputSchema: {},
            },
          },
          resources: [],
          resourceTemplates: [],
          authStatus: "unsupported",
        },
      ],
      nextCursor: null,
    });
    mockAppServerResult({ servers: [config] });

    await expect(mcpApi.listServersWithStatus()).resolves.toMatchObject([
      {
        id: "server-v2",
        name: "remote-docs",
        plugin_id: "connector-docs",
        is_running: true,
        config: config.server_config,
        runtime_status: {
          is_running: true,
          auth_status: { mode: "none", available: true },
        },
      },
    ]);
    expect(appServerRequestMock).toHaveBeenNthCalledWith(
      1,
      "mcpServerStatus/list",
      {},
    );
    expect(appServerRequestMock).toHaveBeenNthCalledWith(
      2,
      "mcpServer/list",
      {},
    );
  });

  it("状态列表应遍历 opaque cursor 并投影未登录 OAuth", async () => {
    const config = {
      id: "server-v2",
      name: "remote-docs",
      server_config: {
        type: "streamable_http" as const,
        url: "https://example.com/mcp",
        scopes: ["search.read"],
      },
      enabled_lime: true,
      enabled_claude: false,
      enabled_codex: true,
      enabled_gemini: false,
    };
    mockAppServerResult({ data: [], nextCursor: "page-2" });
    mockAppServerResult({
      data: [
        {
          name: "remote-docs",
          runtimeStatus: "notStarted",
          pluginId: null,
          serverInfo: null,
          tools: {},
          resources: [],
          resourceTemplates: [],
          authStatus: "notLoggedIn",
        },
      ],
      nextCursor: null,
    });
    mockAppServerResult({ servers: [config] });

    await expect(mcpApi.listServersWithStatus()).resolves.toMatchObject([
      {
        name: "remote-docs",
        runtime_status: {
          auth_status: {
            mode: "oauth",
            available: false,
            reason_code: "oauth_login_required",
            action_plan: {
              kind: "oauth_login",
              state: "login_required",
              scopes: ["search.read"],
            },
          },
        },
      },
    ]);
    expect(appServerRequestMock).toHaveBeenNthCalledWith(
      2,
      "mcpServerStatus/list",
      { cursor: "page-2" },
    );
  });

  it("状态列表应拒绝重复 cursor", async () => {
    mockAppServerResult({ data: [], nextCursor: "same" });
    mockAppServerResult({ data: [], nextCursor: "same" });

    await expect(mcpApi.listServersWithStatus()).rejects.toThrow(
      "mcpServerStatus/list returned a repeated nextCursor",
    );
  });

  it("MCP runtime 使用命令应通过 App Server current method 传递参数", async () => {
    const tool = {
      name: "mcp__docs__search_docs",
      server_name: "docs",
      description: "search",
      input_schema: {},
      output_schema: {
        type: "object",
        properties: {
          content: {
            type: "array",
            items: { type: "object" },
          },
          structuredContent: {
            type: "object",
            properties: {
              results: { type: "array" },
            },
          },
          isError: { type: "boolean" },
          _meta: { type: "object" },
        },
        required: ["content"],
        additionalProperties: false,
      },
    };
    mockAppServerResult({ tools: [tool] });
    mockAppServerResult({ tools: [tool] });
    mockAppServerResult({
      content: [{ type: "image", data: "aW1hZ2U=", mimeType: "image/png" }],
      structuredContent: { source: "exact" },
      isError: false,
    });
    mockAppServerResult({
      description: "prompt",
      messages: [{ role: "user", content: { type: "text", text: "hello" } }],
    });
    mockAppServerResult({
      contents: [
        {
          uri: "docs://readme",
          mimeType: "text/plain",
          text: "README",
        },
      ],
    });

    await expect(
      mcpApi.listToolsForContext("assistant", true),
    ).resolves.toEqual([tool]);
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpTool/listForContext",
      {
        caller: "assistant",
        includeDeferred: true,
      },
    );

    await expect(
      mcpApi.searchTools("search", "tool_search", 5),
    ).resolves.toEqual([tool]);
    expect(appServerRequestMock).toHaveBeenLastCalledWith("mcpTool/search", {
      query: "search",
      caller: "tool_search",
      limit: 5,
    });

    await expect(
      mcpApi.callServerTool({
        threadId: "thread-1",
        server: "docs",
        tool: "search_docs",
        arguments: { q: "lime" },
        meta: { requestId: "desktop-1" },
      }),
    ).resolves.toEqual({
      content: [{ type: "image", data: "aW1hZ2U=", mime_type: "image/png" }],
      structuredContent: { source: "exact" },
      is_error: false,
    });
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServer/tool/call",
      {
        threadId: "thread-1",
        server: "docs",
        tool: "search_docs",
        arguments: { q: "lime" },
        _meta: { requestId: "desktop-1" },
      },
    );

    await expect(
      mcpApi.getPrompt("docs", "summarize", { topic: "lime" }),
    ).resolves.toEqual({
      description: "prompt",
      messages: [{ role: "user", content: { type: "text", text: "hello" } }],
    });
    expect(appServerRequestMock).toHaveBeenLastCalledWith("mcpPrompt/get", {
      server: "docs",
      name: "summarize",
      arguments: { topic: "lime" },
    });

    await expect(mcpApi.readResource("docs", "docs://readme")).resolves.toEqual(
      {
        uri: "docs://readme",
        mime_type: "text/plain",
        text: "README",
      },
    );
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServer/resource/read",
      {
        server: "docs",
        uri: "docs://readme",
      },
    );

    mockAppServerResult({
      contents: [
        {
          uri: "ui://demo/report.html",
          mimeType: "text/html;profile=mcp-app",
          text: "<main>Plugin report</main>",
          _meta: {
            ui: { csp: { connectDomains: ["https://api.example.com"] } },
          },
        },
      ],
    });
    await expect(
      mcpApi.readResource("plugin__demo__server", "ui://demo/report.html", {
        threadId: "thread-1",
        originCallId: "item-1",
        connectorId: "connector-demo",
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        uri: "ui://demo/report.html",
        meta: expect.objectContaining({ ui: expect.any(Object) }),
      }),
    );
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServer/resource/read",
      {
        server: "plugin__demo__server",
        uri: "ui://demo/report.html",
        threadId: "thread-1",
        originCallId: "item-1",
        connectorId: "connector-demo",
      },
    );

    mockAppServerResult({});
    await expect(
      mcpApi.subscribeResource("docs", "docs://readme"),
    ).resolves.toBeUndefined();
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpResource/subscribe",
      {
        server: "docs",
        uri: "docs://readme",
      },
    );

    mockAppServerResult({});
    await expect(
      mcpApi.unsubscribeResource("docs", "docs://readme"),
    ).resolves.toBeUndefined();
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpResource/unsubscribe",
      {
        server: "docs",
        uri: "docs://readme",
      },
    );
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("MCP lifecycle / OAuth login 应通过 App Server current method 传递参数", async () => {
    mockAppServerResult({});
    mockAppServerResult({});
    mockAppServerResult({
      authorizationUrl: "http://127.0.0.1:49152/oauth/start",
      state: "pending",
    });

    await expect(mcpApi.startServer("docs")).resolves.toBeUndefined();
    expect(appServerRequestMock).toHaveBeenLastCalledWith("mcpServer/start", {
      name: "docs",
    });

    await expect(mcpApi.stopServer("docs")).resolves.toBeUndefined();
    expect(appServerRequestMock).toHaveBeenLastCalledWith("mcpServer/stop", {
      name: "docs",
    });

    await expect(
      mcpApi.loginOAuthServer("remote-docs", {
        scopes: ["search.read"],
        timeoutSecs: 120,
      }),
    ).resolves.toEqual({
      authorizationUrl: "http://127.0.0.1:49152/oauth/start",
      state: "pending",
    });
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServer/oauth/login",
      {
        name: "remote-docs",
        scopes: ["search.read"],
        timeoutSecs: 120,
      },
    );
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("MCP CRUD / import / sync 应通过 App Server current method 传递参数", async () => {
    const server = {
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
    mockAppServerResult({ servers: [server] });
    mockAppServerResult({ servers: [server] });
    mockAppServerResult({ servers: [] });
    mockAppServerResult({ servers: [server] });
    mockAppServerResult({ importedCount: 2, servers: [server] });
    mockAppServerResult({ servers: [server] });

    await expect(mcpApi.addServer(server)).resolves.toBeUndefined();
    expect(appServerRequestMock).toHaveBeenLastCalledWith("mcpServer/create", {
      server,
    });

    await expect(mcpApi.updateServer(server)).resolves.toBeUndefined();
    expect(appServerRequestMock).toHaveBeenLastCalledWith("mcpServer/update", {
      server,
    });

    await expect(mcpApi.deleteServer("server-1")).resolves.toBeUndefined();
    expect(appServerRequestMock).toHaveBeenLastCalledWith("mcpServer/delete", {
      id: "server-1",
    });

    await expect(
      mcpApi.toggleServer("server-1", "codex", true),
    ).resolves.toBeUndefined();
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServer/enabled/set",
      {
        id: "server-1",
        appType: "codex",
        enabled: true,
      },
    );

    await expect(mcpApi.importFromApp("codex")).resolves.toBe(2);
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServer/importFromApp",
      {
        appType: "codex",
      },
    );

    await expect(mcpApi.syncAllToLive()).resolves.toBeUndefined();
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServer/syncAllToLive",
      {},
    );
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("MCP prepare requests 应顺序执行 App Server current 方法", async () => {
    const tool = {
      name: "mcp__context7__resolve-library-id",
      server_name: "context7",
      description: "resolve",
      input_schema: {},
    };
    mockAppServerResult({
      importedCount: 1,
      servers: [{ id: "codex-docs", name: "codex-docs" }],
    });
    mockAppServerResult({});
    mockAppServerResult({ tools: [tool] });

    await expect(
      mcpApi.executePrepareRequests([
        {
          method: "mcpServer/importFromApp",
          params: { appType: "codex" },
          reason: "server_missing",
          status: "candidate",
        },
        {
          method: "mcpServer/start",
          params: { name: "context7" },
          reason: "server_stopped",
          status: "candidate",
        },
        {
          method: "mcpTool/listForContext",
          params: { caller: "plugin:docs-plugin", includeDeferred: true },
          reason: "tool_listing",
          status: "candidate",
        },
      ]),
    ).resolves.toEqual([
      {
        method: "mcpServer/importFromApp",
        status: "completed",
        importedCount: 1,
      },
      {
        method: "mcpServer/start",
        status: "completed",
      },
      {
        method: "mcpTool/listForContext",
        status: "completed",
        toolCount: 1,
        tools: [tool],
      },
    ]);

    expect(appServerRequestMock.mock.calls).toEqual([
      ["mcpServer/importFromApp", { appType: "codex" }],
      ["mcpServer/start", { name: "context7" }],
      [
        "mcpTool/listForContext",
        { caller: "plugin:docs-plugin", includeDeferred: true },
      ],
    ]);
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("MCP call proof requests 应通过 Thread-scoped exact 方法执行", async () => {
    mockAppServerResult({
      content: [{ type: "text", text: "ok" }],
      structuredContent: { libraryId: "/facebook/react" },
      is_error: false,
    });

    await expect(
      mcpApi.executeCallProofRequests(
        [
          {
            method: "mcpServer/tool/call",
            params: {
              server: "context7",
              tool: "resolve-library-id",
              arguments: { libraryName: "react" },
            },
            reason: "tool_call_proof",
            status: "candidate",
          },
        ],
        "thread-1",
      ),
    ).resolves.toEqual([
      {
        method: "mcpServer/tool/call",
        status: "completed",
        result: {
          content: [{ type: "text", text: "ok" }],
          structuredContent: { libraryId: "/facebook/react" },
          is_error: false,
        },
      },
    ]);

    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServer/tool/call",
      {
        threadId: "thread-1",
        server: "context7",
        tool: "resolve-library-id",
        arguments: { libraryName: "react" },
      },
    );
    expect(safeInvoke).not.toHaveBeenCalled();
  });
});
