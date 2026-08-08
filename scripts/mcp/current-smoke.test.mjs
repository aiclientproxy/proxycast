import fs from "node:fs";
import { describe, expect, it } from "vitest";

import { sanitizeJson } from "./lib/current-smoke-core.mjs";
import {
  readLiveProviderConfig,
  runMcpLiveProviderSmoke,
} from "./live-provider-smoke.mjs";

function readCurrentSmoke() {
  return fs.readFileSync("scripts/mcp/current-smoke.mjs", "utf8");
}

function readCurrentSmokeCore() {
  return fs.readFileSync("scripts/mcp/lib/current-smoke-core.mjs", "utf8");
}

function readCurrentSmokeFixture() {
  return fs.readFileSync("scripts/mcp/lib/current-smoke-fixture.mjs", "utf8");
}

function readCurrentSmokeTransport() {
  return fs.readFileSync("scripts/mcp/lib/current-smoke-transport.mjs", "utf8");
}

function readCurrentSmokeSurface() {
  return [
    readCurrentSmoke(),
    readCurrentSmokeCore(),
    readCurrentSmokeFixture(),
    readCurrentSmokeTransport(),
  ].join("\n");
}

function readOAuthFixtureSmoke() {
  return fs.readFileSync("scripts/mcp/oauth-fixture-smoke.mjs", "utf8");
}

function readLiveProviderSmoke() {
  return fs.readFileSync("scripts/mcp/live-provider-smoke.mjs", "utf8");
}

describe("mcp current smoke guard", () => {
  it("keeps MCP smoke on App Server current JSON-RPC and rejects legacy facade", () => {
    const content = readCurrentSmokeSurface();

    expect(content).toContain("APP_SERVER_HANDLE_JSON_LINES_COMMAND");
    expect(content).toContain('"app_server_handle_json_lines"');
    expect(content).toContain('"mcpServer/oauth/login"');
    expect(content).toContain('"mcpServer/tool/call"');
    expect(content).toContain('"mcpServer/resource/read"');
    expect(content).toContain('"resources/templates/list"');
    expect(content).toContain("resourceTemplatesSeen");
    expect(content).toContain("resourceTemplateUriTemplate");
    expect(content).toContain("outputSchemaStructuredContentSeen");
    expect(content).toContain("structuredContentEcho");
    expect(content).toContain("structuredContent: {");
    expect(content).toContain("outputSchema: {");
    expect(content).toContain("runFailureIsolationChecks");
    expect(content).toContain("missing-mcp-server.mjs");
    expect(content).toContain("healthyToolCallAfterFailure");
    expect(content).toContain("healthyResourceReadAfterFailure");
    expect(content).toContain("ephemeral: true");
    expect(content).not.toContain(
      '{ model: "gpt-5.4", modelProvider: "openai" }',
    );
    expect(content).toContain("LEGACY_MCP_COMMANDS");
    expect(content).toContain("summary.legacyMcpCommandsSeen.length === 0");
  });

  it("keeps optional OAuth fixture evidence behind the MCP current smoke entry", () => {
    const content = readCurrentSmokeSurface();
    const oauthFixture = readOAuthFixtureSmoke();

    expect(content).toContain("--allow-oauth-fixture");
    expect(content).toContain("runMcpOAuthFixtureSmoke");
    expect(content).toContain("summary.oauthFixture");
    expect(oauthFixture).toContain("startMcpOAuthFixtureProvider");
    expect(oauthFixture).toContain('"mcpServer/oauth/login"');
    expect(oauthFixture).toContain('"open_external_url"');
    expect(oauthFixture).toContain('"mcpServerStatus/list"');
    expect(oauthFixture).toContain('transport: "streamable_http"');
    expect(oauthFixture).toContain('scopes: ["fixture.read"]');
    expect(oauthFixture).toContain("tokenRequestCount");
    expect(oauthFixture).not.toContain("window.open");
  });

  it("rejects the retired plugin-private MCP inventory fixture", () => {
    const content = readCurrentSmokeSurface();

    expect(content).not.toContain("--allow-plugin-runtime-fixture");
    expect(content).not.toContain("runPluginRuntimeFixtureChecks");
    expect(content).not.toContain("pluginRuntimeFixture");
    expect(content).not.toContain("plugin_runtime_capabilities");
    expect(content).not.toContain("plugin_mcp_targets");
  });

  it("keeps real provider evidence live-gated behind explicit environment", () => {
    const content = readCurrentSmokeSurface();
    const liveProvider = readLiveProviderSmoke();

    expect(content).toContain("--allow-live-provider");
    expect(content).toContain("Live provider 安全约束");
    expect(content).toContain("runMcpLiveProviderSmoke");
    expect(content).toContain("summary.liveProvider");
    expect(liveProvider).toContain("LIME_MCP_LIVE_SERVER_URL");
    expect(content).toContain(
      "LIME_MCP_LIVE_BEARER_TOKEN_ENV_VAR 必须是环境变量名",
    );
    expect(content).toContain(
      "LIME_MCP_LIVE_ENV_HTTP_HEADERS_JSON 的 value 必须是环境变量名",
    );
    expect(liveProvider).toContain('"mcpServer/create"');
    expect(liveProvider).toContain('"mcpServer/start"');
    expect(liveProvider).toContain('"mcpTool/list"');
    expect(liveProvider).toContain('"mcpTool/listForContext"');
    expect(liveProvider).toContain('"mcpTool/search"');
    expect(liveProvider).toContain('"mcpResource/list"');
    expect(liveProvider).toContain('"mcpServer/tool/call"');
    expect(liveProvider).toContain('"mcpServer/resource/read"');
    expect(liveProvider).toContain("ephemeral: true");
    expect(liveProvider).not.toContain('modelProvider: "openai"');
    expect(liveProvider).not.toContain("defaultMocks");
    expect(liveProvider).not.toContain("mockPriorityCommands");
  });

  it("keeps live provider URL secrets out of evidence", () => {
    for (const url of [
      "ftp://example.com/mcp",
      "https://user@example.com/mcp",
      "https://example.com/mcp?token=secret",
      "https://example.com/mcp#token",
    ]) {
      expect(() =>
        readLiveProviderConfig({ LIME_MCP_LIVE_SERVER_URL: url }),
      ).toThrow(/LIME_MCP_LIVE_SERVER_URL/);
    }
    expect(() =>
      readLiveProviderConfig({
        LIME_MCP_LIVE_SERVER_URL: "https://example.com/mcp",
        LIME_MCP_LIVE_BEARER_TOKEN_ENV_VAR: "sk-inline-token",
      }),
    ).toThrow(/inline secret/);
    expect(() =>
      readLiveProviderConfig({
        LIME_MCP_LIVE_SERVER_URL: "https://example.com/mcp",
        LIME_MCP_LIVE_ENV_HTTP_HEADERS_JSON: '{"X-Api-Key":"sk-inline-token"}',
      }),
    ).toThrow(/inline secret/);

    const config = readLiveProviderConfig({
      LIME_MCP_LIVE_SERVER_URL: "https://example.com/mcp",
      LIME_MCP_LIVE_SERVER_NAME: "Example Provider",
      LIME_MCP_LIVE_BEARER_TOKEN_ENV_VAR: "MCP_PROVIDER_TOKEN",
      LIME_MCP_LIVE_ENV_HTTP_HEADERS_JSON:
        '{"X-Api-Key":"MCP_PROVIDER_API_KEY"}',
      LIME_MCP_LIVE_SCOPES: "read, write",
      LIME_MCP_LIVE_TOOL_NAME: "echo",
      LIME_MCP_LIVE_TOOL_ARGS_JSON: '{"message":"hello"}',
      LIME_MCP_LIVE_RESOURCE_URI: "mcp://example/resource",
    });

    expect(config.serverName).toBe("Example-Provider");
    expect(config.serverConfig).toMatchObject({
      transport: "streamable_http",
      url: "https://example.com/mcp",
      bearer_token_env_var: "MCP_PROVIDER_TOKEN",
      env_http_headers: { "X-Api-Key": "MCP_PROVIDER_API_KEY" },
      scopes: ["read", "write"],
    });
    expect(config.expected).toMatchObject({
      toolName: "echo",
      toolArguments: { message: "hello" },
      resourceUri: "mcp://example/resource",
    });
    expect(config.evidence).toMatchObject({
      urlHost: "example.com",
      bearerTokenEnvVar: "MCP_PROVIDER_TOKEN",
      envHttpHeaderNames: ["X-Api-Key"],
      scopes: ["read", "write"],
      toolName: "echo",
      resourceUriProvided: true,
      resourceUriSummary: {
        scheme: "mcp",
        host: "example",
        hasPath: true,
        pathDepth: 1,
        hasQuery: false,
        hasHash: false,
      },
    });
    expect(config.evidence).not.toHaveProperty("url");
    expect(JSON.stringify(config.evidence)).not.toContain(
      "mcp://example/resource",
    );
  });

  it("summarizes URL and URI fields in network evidence", () => {
    const sanitized = sanitizeJson({
      server_config: {
        url: "https://example.com/private/mcp?q=1#fragment",
      },
      resourceUri: "mcp://example/private/resource?q=1",
      authorization: "Bearer abc123",
    });

    expect(sanitized.server_config.url).toMatchObject({
      scheme: "https",
      host: "example.com",
      hasPath: true,
      pathDepth: 2,
      hasQuery: true,
      hasHash: true,
    });
    expect(sanitized.resourceUri).toMatchObject({
      scheme: "mcp",
      host: "example",
      hasPath: true,
      pathDepth: 2,
      hasQuery: true,
      hasHash: false,
    });
    const serialized = JSON.stringify(sanitized);
    expect(serialized).not.toContain("private");
    expect(serialized).not.toContain("q=1");
    expect(serialized).not.toContain("fragment");
    expect(serialized).not.toContain("abc123");
    expect(serialized).toContain("[redacted]");
  });

  it("fails live provider smoke when requested tool call returns MCP error", async () => {
    const previousEnv = {
      LIME_MCP_LIVE_SERVER_URL: process.env.LIME_MCP_LIVE_SERVER_URL,
      LIME_MCP_LIVE_SERVER_NAME: process.env.LIME_MCP_LIVE_SERVER_NAME,
      LIME_MCP_LIVE_TOOL_NAME: process.env.LIME_MCP_LIVE_TOOL_NAME,
      LIME_MCP_LIVE_TOOL_ARGS_JSON: process.env.LIME_MCP_LIVE_TOOL_ARGS_JSON,
    };
    process.env.LIME_MCP_LIVE_SERVER_URL = "https://example.com/mcp";
    process.env.LIME_MCP_LIVE_SERVER_NAME = "Example Provider";
    process.env.LIME_MCP_LIVE_TOOL_NAME = "echo";
    process.env.LIME_MCP_LIVE_TOOL_ARGS_JSON = '{"message":"hello"}';

    let createdServerName = "";
    const invokeAppServerMethod = async (_options, method, params) => {
      if (method === "mcpServer/create") {
        createdServerName = params.server.name;
        return { servers: [params.server] };
      }
      if (method === "mcpServer/start") {
        return {};
      }
      if (method === "mcpServerStatus/list") {
        return {
          servers: [
            {
              name: createdServerName,
              is_running: true,
              runtime_status: {
                transport: "streamable_http",
                server_info: {
                  supports_tools: true,
                  supports_resources: false,
                },
              },
            },
          ],
        };
      }
      if (
        method === "mcpTool/list" ||
        method === "mcpTool/listForContext" ||
        method === "mcpTool/search"
      ) {
        return {
          tools: [{ name: `mcp__${createdServerName}__echo` }],
        };
      }
      if (method === "thread/start") {
        return { thread: { id: "thread-live" } };
      }
      if (method === "mcpServer/tool/call") {
        return {
          content: [{ type: "text", text: "bad arguments" }],
          isError: true,
        };
      }
      if (method === "mcpServer/stop" || method === "mcpServer/delete") {
        return {};
      }
      if (method === "mcpResource/list") {
        return { resources: [], resourceTemplates: [] };
      }
      throw new Error(`unexpected method ${method}`);
    };

    try {
      await expect(
        runMcpLiveProviderSmoke({
          options: {},
          entries: [],
          invokeAppServerMethod,
        }),
      ).rejects.toThrow(/isError=true/);
    } finally {
      for (const [name, value] of Object.entries(previousEnv)) {
        if (value === undefined) {
          delete process.env[name];
        } else {
          process.env[name] = value;
        }
      }
    }
  });
});
