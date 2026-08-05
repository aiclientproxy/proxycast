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
      ["mcpServerStatus/list", "servers", () => mcpApi.listServersWithStatus()],
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

  it("runtime resource owner 应要求完整 canonical identity", async () => {
    await expect(
      mcpApi.readResource("docs", "ui://demo/report.html", {
        sessionId: " ",
        threadId: "thread-1",
      }),
    ).rejects.toThrow("sessionId cannot be empty");
    await expect(
      mcpApi.readResource("docs", "ui://demo/report.html", {
        sessionId: "session-1",
        threadId: " ",
      }),
    ).rejects.toThrow("threadId cannot be empty");

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

  it("状态列表应保留 streamable HTTP 配置与 runtime status", async () => {
    const server = {
      id: "server-http",
      name: "remote-docs",
      description: "remote docs",
      config: {
        type: "streamable_http",
        url: "https://example.com/mcp",
        bearer_token_env_var: "MCP_TOKEN",
        scopes: ["search.read"],
        oauth: { client_id: "lime-client" },
        oauth_resource: "https://example.com",
        startup_timeout: 30,
        tool_timeout: 15,
        enabled_tools: ["search"],
        disabled_tools: ["delete"],
      },
      is_running: true,
      server_info: {
        name: "remote-docs",
        version: "1.0.0",
        supports_tools: true,
        supports_prompts: false,
        supports_resources: true,
      },
      runtime_status: {
        name: "remote-docs",
        transport: "streamable_http",
        enabled: true,
        is_running: true,
        required: false,
        supports_parallel_tool_calls: true,
        startup_timeout: 30,
        tool_timeout: 15,
        enabled_tools: ["search"],
        disabled_tools: ["delete"],
        auth_status: {
          mode: "oauth",
          available: false,
          reason_code: "oauth_runtime_not_implemented",
          action_plan: {
            kind: "oauth_login",
            state: "runtime_not_connected",
            required_runtime: "mcp_server_oauth_login",
            scopes: ["search.read"],
            oauth_resource: "https://example.com",
            client_id: "lime-client",
          },
        },
        server_info: {
          name: "remote-docs",
          version: "1.0.0",
          supports_tools: true,
          supports_prompts: false,
          supports_resources: true,
        },
      },
      enabled_lime: true,
      enabled_claude: false,
      enabled_codex: true,
      enabled_gemini: false,
    };
    mockAppServerResult({ servers: [server] });

    await expect(mcpApi.listServersWithStatus()).resolves.toEqual([server]);
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServerStatus/list",
      {},
    );
    expect(safeInvoke).not.toHaveBeenCalled();
  });

  it("动态 OAuth 状态应指向 current 登录 runtime", async () => {
    const server = {
      id: "server-http-dynamic-oauth",
      name: "remote-docs",
      config: {
        type: "streamable_http",
        url: "https://example.com/mcp",
        scopes: ["search.read"],
      },
      is_running: false,
      runtime_status: {
        name: "remote-docs",
        transport: "streamable_http",
        enabled: true,
        is_running: false,
        required: false,
        supports_parallel_tool_calls: false,
        startup_timeout: 30,
        tool_timeout: 30,
        disabled_tools: [],
        auth_status: {
          mode: "oauth",
          available: true,
          reason_code: "oauth_login_required",
          action_plan: {
            kind: "oauth_login",
            state: "login_required",
            required_runtime: "mcp_server_oauth_login",
            scopes: ["search.read"],
          },
        },
      },
      enabled_lime: true,
      enabled_claude: false,
      enabled_codex: true,
      enabled_gemini: false,
    };
    mockAppServerResult({ servers: [server] });

    await expect(mcpApi.listServersWithStatus()).resolves.toEqual([server]);
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpServerStatus/list",
      {},
    );
    expect(safeInvoke).not.toHaveBeenCalled();
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
      content: [{ type: "text", text: "ok" }],
      structuredContent: {
        results: [{ title: "MCP current" }],
      },
      is_error: false,
    });
    mockAppServerResult({
      content: [{ type: "text", text: "caller ok" }],
      structuredContent: {
        results: [{ title: "MCP caller current" }],
      },
      is_error: false,
    });
    mockAppServerResult({
      description: "prompt",
      messages: [{ role: "user", content: { type: "text", text: "hello" } }],
    });
    mockAppServerResult({
      uri: "docs://readme",
      mime_type: "text/plain",
      text: "README",
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
      mcpApi.callTool("mcp__docs__search_docs", { q: "lime" }),
    ).resolves.toEqual({
      content: [{ type: "text", text: "ok" }],
      structuredContent: {
        results: [{ title: "MCP current" }],
      },
      is_error: false,
    });
    expect(appServerRequestMock).toHaveBeenLastCalledWith("mcpTool/call", {
      toolName: "mcp__docs__search_docs",
      arguments: { q: "lime" },
    });

    await expect(
      mcpApi.callToolWithCaller(
        "mcp__docs__search_docs",
        { q: "lime" },
        "assistant",
      ),
    ).resolves.toEqual({
      content: [{ type: "text", text: "caller ok" }],
      structuredContent: {
        results: [{ title: "MCP caller current" }],
      },
      is_error: false,
    });
    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpTool/callWithCaller",
      {
        toolName: "mcp__docs__search_docs",
        arguments: { q: "lime" },
        caller: "assistant",
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
    expect(appServerRequestMock).toHaveBeenLastCalledWith("mcpResource/read", {
      server: "docs",
      uri: "docs://readme",
    });

    mockAppServerResult({
      uri: "ui://demo/report.html",
      mime_type: "text/html;profile=mcp-app",
      text: "<main>Plugin report</main>",
      meta: { ui: { csp: { connectDomains: ["https://api.example.com"] } } },
    });
    await expect(
      mcpApi.readResource("plugin__demo__server", "ui://demo/report.html", {
        sessionId: "session-1",
        threadId: "thread-1",
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        uri: "ui://demo/report.html",
        meta: expect.objectContaining({ ui: expect.any(Object) }),
      }),
    );
    expect(appServerRequestMock).toHaveBeenLastCalledWith("mcpResource/read", {
      server: "plugin__demo__server",
      uri: "ui://demo/report.html",
      sessionId: "session-1",
      threadId: "thread-1",
    });

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

  it("MCP call proof requests 应通过 caller-scoped current 方法执行", async () => {
    mockAppServerResult({
      content: [{ type: "text", text: "ok" }],
      structuredContent: { libraryId: "/facebook/react" },
      is_error: false,
    });

    await expect(
      mcpApi.executeCallProofRequests([
        {
          method: "mcpTool/callWithCaller",
          params: {
            toolName: "mcp__context7__resolve-library-id",
            caller: "plugin:docs-plugin",
            arguments: { libraryName: "react" },
          },
          reason: "tool_call_proof",
          status: "candidate",
        },
      ]),
    ).resolves.toEqual([
      {
        method: "mcpTool/callWithCaller",
        status: "completed",
        result: {
          content: [{ type: "text", text: "ok" }],
          structuredContent: { libraryId: "/facebook/react" },
          is_error: false,
        },
      },
    ]);

    expect(appServerRequestMock).toHaveBeenLastCalledWith(
      "mcpTool/callWithCaller",
      {
        toolName: "mcp__context7__resolve-library-id",
        arguments: { libraryName: "react" },
        caller: "plugin:docs-plugin",
      },
    );
    expect(safeInvoke).not.toHaveBeenCalled();
  });
});
