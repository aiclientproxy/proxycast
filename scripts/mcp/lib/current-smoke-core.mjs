import path from "node:path";

import {
  invokeAppServerMethod,
  sanitizeJson,
  summarizeInvokeEntries as summarizeTransportInvokeEntries,
} from "./current-smoke-transport.mjs";

export {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  invokeAppServerMethod,
  invokeBridgeCommand,
  sanitizeJson,
  waitForHealth,
  writeJsonFile,
} from "./current-smoke-transport.mjs";

export const REQUIRED_READ_METHODS = [
  "mcpServer/list",
  "mcpServerStatus/list",
  "mcpTool/list",
  "mcpTool/listForContext",
  "mcpTool/search",
  "mcpPrompt/list",
  "mcpResource/list",
];

export const FIXTURE_METHODS = [
  "thread/start",
  "mcpServer/create",
  "mcpServer/start",
  "mcpServerStatus/list",
  "mcpTool/list",
  "mcpServer/tool/call",
  "mcpResource/list",
  "mcpServer/resource/read",
  "mcpServer/stop",
  "mcpServer/delete",
];

export const OAUTH_FIXTURE_METHODS = [
  "mcpServer/create",
  "mcpServer/oauth/login",
  "mcpServerStatus/list",
  "mcpServer/delete",
];

export const PLUGIN_RUNTIME_FIXTURE_METHODS = [
  "thread/start",
  "mcpServer/create",
  "mcpServer/start",
  "agentSession/toolInventory/read",
  "mcpTool/listForContext",
  "mcpServer/tool/call",
  "mcpServer/stop",
  "mcpServer/delete",
];

export function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function assertArrayField(method, result, field) {
  assert(
    result && typeof result === "object" && Array.isArray(result[field]),
    `${method} did not return ${field}`,
  );
  return result[field];
}

function assertEmptyObject(method, result) {
  assert(
    result && typeof result === "object" && !Array.isArray(result),
    `${method} did not return object result`,
  );
  assert(
    Object.keys(result).length === 0,
    `${method} did not return empty lifecycle result`,
  );
}

function assertToolOutputSchema(method, tool, expectedToolName) {
  assert(
    tool && typeof tool === "object" && tool.name === expectedToolName,
    `${method} did not return expected fixture tool ${expectedToolName}`,
  );
  const outputSchema = tool.output_schema ?? tool.outputSchema;
  assert(
    outputSchema && typeof outputSchema === "object",
    `${method} did not return output_schema for ${expectedToolName}`,
  );
  const structuredContentSchema =
    outputSchema.properties?.structuredContent ??
    outputSchema.properties?.structured_content;
  assert(
    structuredContentSchema && typeof structuredContentSchema === "object",
    `${method} output_schema did not expose structuredContent`,
  );
  assert(
    structuredContentSchema.properties?.echoedMessage?.type === "string",
    `${method} structuredContent schema did not expose echoedMessage`,
  );
  return {
    outputSchemaStructuredContentSeen: true,
    structuredContentSchemaKeys: Object.keys(
      structuredContentSchema.properties ?? {},
    ).sort(),
  };
}

function assertToolResult(
  method,
  result,
  expectedText,
  expectedStructuredContent,
) {
  assert(
    result && typeof result === "object" && Array.isArray(result.content),
    `${method} did not return content`,
  );
  assert(
    (result.isError ?? result.is_error) === false,
    `${method} returned isError=true`,
  );
  assert(
    result.content.some(
      (item) => item?.type === "text" && item?.text === expectedText,
    ),
    `${method} did not return expected text ${expectedText}`,
  );
  const structuredContent =
    result.structuredContent ?? result.structured_content ?? null;
  if (expectedStructuredContent) {
    assert(
      structuredContent && typeof structuredContent === "object",
      `${method} did not return structuredContent`,
    );
    for (const [key, value] of Object.entries(expectedStructuredContent)) {
      assert(
        structuredContent[key] === value,
        `${method} structuredContent.${key} drifted`,
      );
    }
  }
  return structuredContent;
}

function assertResourceResult(method, result, expectedText) {
  const content = Array.isArray(result?.contents) ? result.contents[0] : null;
  assert(
    content && content.uri === "fixture://status",
    `${method} did not return fixture resource uri`,
  );
  assert(
    content.text === expectedText,
    `${method} did not return expected text`,
  );
}

async function startFixtureThread(options, entries) {
  const result = await invokeAppServerMethod(
    options,
    "thread/start",
    { ephemeral: true },
    entries,
  );
  const threadId = result?.thread?.id;
  assert(
    typeof threadId === "string" && threadId.length > 0,
    "thread/start did not return a thread id",
  );
  return threadId;
}

function assertResourceTemplate(method, templates, expectedUriTemplate) {
  const template = templates.find(
    (item) =>
      item?.uri_template === expectedUriTemplate ||
      item?.uriTemplate === expectedUriTemplate,
  );
  assert(template, `${method} did not return ${expectedUriTemplate}`);
  return template;
}

export function summarizeInvokeEntries(entries) {
  return summarizeTransportInvokeEntries(entries, {
    requiredReadMethods: REQUIRED_READ_METHODS,
    fixtureMethods: FIXTURE_METHODS,
    oauthFixtureMethods: OAUTH_FIXTURE_METHODS,
    pluginRuntimeFixtureMethods: PLUGIN_RUNTIME_FIXTURE_METHODS,
  });
}

export async function runReadChecks(options, entries) {
  assertArrayField(
    "mcpServer/list",
    await invokeAppServerMethod(options, "mcpServer/list", {}, entries),
    "servers",
  );
  assertArrayField(
    "mcpServerStatus/list",
    await invokeAppServerMethod(options, "mcpServerStatus/list", {}, entries),
    "servers",
  );
  assertArrayField(
    "mcpTool/list",
    await invokeAppServerMethod(options, "mcpTool/list", {}, entries),
    "tools",
  );
  assertArrayField(
    "mcpTool/listForContext",
    await invokeAppServerMethod(
      options,
      "mcpTool/listForContext",
      { caller: "assistant", includeDeferred: true },
      entries,
    ),
    "tools",
  );
  assertArrayField(
    "mcpTool/search",
    await invokeAppServerMethod(
      options,
      "mcpTool/search",
      { query: "fixture", caller: "tool_search", limit: 5 },
      entries,
    ),
    "tools",
  );
  assertArrayField(
    "mcpPrompt/list",
    await invokeAppServerMethod(options, "mcpPrompt/list", {}, entries),
    "prompts",
  );
  const resourceList = await invokeAppServerMethod(
    options,
    "mcpResource/list",
    {},
    entries,
  );
  assertArrayField("mcpResource/list", resourceList, "resources");
  assertArrayField("mcpResource/list", resourceList, "resourceTemplates");
}

async function runFailureIsolationChecks({
  options,
  entries,
  fixture,
  healthyServerName,
  healthyToolName,
  threadId,
}) {
  const failedServerId = `mcp-current-failed-${Date.now()}`;
  const failedServerName = failedServerId.replace(/[^a-zA-Z0-9_-]/g, "-");
  let startError = null;

  try {
    assertArrayField(
      "mcpServer/create",
      await invokeAppServerMethod(
        options,
        "mcpServer/create",
        {
          server: {
            id: failedServerId,
            name: failedServerName,
            description: "MCP failure isolation fixture",
            server_config: {
              command: process.execPath,
              args: [path.join(fixture.root, "missing-mcp-server.mjs")],
              cwd: fixture.root,
              timeout: 3,
            },
            enabled_lime: true,
            enabled_claude: false,
            enabled_codex: false,
            enabled_gemini: false,
            created_at: Date.now(),
          },
        },
        entries,
      ),
      "servers",
    );

    try {
      await invokeAppServerMethod(
        options,
        "mcpServer/start",
        { name: failedServerName },
        entries,
      );
    } catch (error) {
      startError = error;
    }
    assert(
      startError instanceof Error,
      "broken MCP server unexpectedly started",
    );
    assert(
      startError.message.startsWith("mcpServer/start error:"),
      "broken MCP server failure did not cross App Server JSON-RPC",
    );

    const statusServers = assertArrayField(
      "mcpServerStatus/list",
      await invokeAppServerMethod(options, "mcpServerStatus/list", {}, entries),
      "servers",
    );
    const healthyStatus = statusServers.find(
      (server) => server?.name === healthyServerName,
    );
    const failedStatus = statusServers.find(
      (server) => server?.name === failedServerName,
    );
    assert(
      healthyStatus?.is_running === true,
      "healthy MCP server stopped after another server failed",
    );
    assert(
      failedStatus?.is_running === false,
      "failed MCP server was reported as running",
    );

    const tools = assertArrayField(
      "mcpTool/list",
      await invokeAppServerMethod(options, "mcpTool/list", {}, entries),
      "tools",
    );
    assert(
      tools.some((tool) => tool?.name === healthyToolName),
      "healthy MCP tool disappeared after another server failed",
    );
    const structuredContent = assertToolResult(
      "mcpServer/tool/call",
      await invokeAppServerMethod(
        options,
        "mcpServer/tool/call",
        {
          threadId,
          server: healthyServerName,
          tool: healthyToolName.split("__").at(-1),
          arguments: { message: "after failed MCP server" },
        },
        entries,
      ),
      "echo: after failed MCP server",
      {
        echoedMessage: "after failed MCP server",
        messageLength: "after failed MCP server".length,
      },
    );
    assertResourceResult(
      "mcpServer/resource/read",
      await invokeAppServerMethod(
        options,
        "mcpServer/resource/read",
        { server: healthyServerName, uri: "fixture://status" },
        entries,
      ),
      "fixture resource ok",
    );

    return {
      failedServerId,
      failedServerName,
      failedStartObserved: true,
      failedServerReportedStopped: true,
      healthyServerStillRunning: true,
      healthyToolStillListed: true,
      healthyToolCallAfterFailure: sanitizeJson(structuredContent),
      healthyResourceReadAfterFailure: true,
    };
  } finally {
    await invokeAppServerMethod(
      options,
      "mcpServer/stop",
      { name: failedServerName },
      entries,
    ).catch(() => {});
    await invokeAppServerMethod(
      options,
      "mcpServer/delete",
      { id: failedServerId },
      entries,
    ).catch((error) => {
      console.warn(
        `[smoke:mcp-current] failed fixture delete failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    });
  }
}

export async function runFixtureChecks(options, entries, fixture) {
  const serverId = `mcp-current-${Date.now()}`;
  const serverName = serverId.replace(/[^a-zA-Z0-9_-]/g, "-");

  try {
    assertArrayField(
      "mcpServer/create",
      await invokeAppServerMethod(
        options,
        "mcpServer/create",
        {
          server: {
            id: serverId,
            name: serverName,
            description: "Current MCP JSON-RPC smoke fixture",
            server_config: {
              command: "node",
              args: [fixture.serverPath],
              cwd: fixture.root,
              timeout: 3,
            },
            enabled_lime: true,
            enabled_claude: false,
            enabled_codex: false,
            enabled_gemini: false,
            created_at: Date.now(),
          },
        },
        entries,
      ),
      "servers",
    );

    assertEmptyObject(
      "mcpServer/start",
      await invokeAppServerMethod(
        options,
        "mcpServer/start",
        { name: serverName },
        entries,
      ),
    );
    const threadId = await startFixtureThread(options, entries);

    const statusServers = assertArrayField(
      "mcpServerStatus/list",
      await invokeAppServerMethod(options, "mcpServerStatus/list", {}, entries),
      "servers",
    );
    assert(
      statusServers.some(
        (server) =>
          server?.name === serverName &&
          server?.is_running === true &&
          server?.server_info?.supports_tools === true &&
          server?.server_info?.supports_resources === true,
      ),
      "mcpServerStatus/list did not report running fixture capabilities",
    );

    const tools = assertArrayField(
      "mcpTool/list",
      await invokeAppServerMethod(options, "mcpTool/list", {}, entries),
      "tools",
    );
    const fixtureToolName = `mcp__${serverName}__echo`;
    const fixtureTool = tools.find((tool) => tool?.name === fixtureToolName);
    assert(fixtureTool, `mcpTool/list did not return ${fixtureToolName}`);
    const outputSchemaEvidence = assertToolOutputSchema(
      "mcpTool/list",
      fixtureTool,
      fixtureToolName,
    );
    const toolsForContext = assertArrayField(
      "mcpTool/listForContext",
      await invokeAppServerMethod(
        options,
        "mcpTool/listForContext",
        { caller: "assistant", includeDeferred: true },
        entries,
      ),
      "tools",
    );
    assertToolOutputSchema(
      "mcpTool/listForContext",
      toolsForContext.find((tool) => tool?.name === fixtureToolName),
      fixtureToolName,
    );
    const searchedTools = assertArrayField(
      "mcpTool/search",
      await invokeAppServerMethod(
        options,
        "mcpTool/search",
        { query: "echo", caller: "tool_search", limit: 5 },
        entries,
      ),
      "tools",
    );
    assertToolOutputSchema(
      "mcpTool/search",
      searchedTools.find((tool) => tool?.name === fixtureToolName),
      fixtureToolName,
    );

    const structuredContent = assertToolResult(
      "mcpServer/tool/call",
      await invokeAppServerMethod(
        options,
        "mcpServer/tool/call",
        {
          threadId,
          server: serverName,
          tool: "echo",
          arguments: { message: "hello current MCP" },
        },
        entries,
      ),
      "echo: hello current MCP",
      {
        echoedMessage: "hello current MCP",
        messageLength: "hello current MCP".length,
      },
    );

    const resourceList = await invokeAppServerMethod(
      options,
      "mcpResource/list",
      {},
      entries,
    );
    const resources = assertArrayField(
      "mcpResource/list",
      resourceList,
      "resources",
    );
    const resourceTemplates = assertArrayField(
      "mcpResource/list",
      resourceList,
      "resourceTemplates",
    );
    assert(
      resources.some(
        (resource) =>
          resource?.uri === "fixture://status" &&
          (resource?.server_name ?? resource?.serverName) === serverName,
      ),
      `mcpResource/list did not return ${serverName}/fixture://status`,
    );
    const fixtureResourceTemplate = assertResourceTemplate(
      "mcpResource/list",
      resourceTemplates,
      "fixture://item/{id}",
    );

    assertResourceResult(
      "mcpServer/resource/read",
      await invokeAppServerMethod(
        options,
        "mcpServer/resource/read",
        { server: serverName, uri: "fixture://status" },
        entries,
      ),
      "fixture resource ok",
    );
    const failureIsolation = await runFailureIsolationChecks({
      options,
      entries,
      fixture,
      healthyServerName: serverName,
      healthyToolName: fixtureToolName,
      threadId,
    });

    return {
      serverId,
      serverName,
      fixtureToolName,
      ...outputSchemaEvidence,
      structuredContentEcho: sanitizeJson(structuredContent),
      structuredContentKeys: Object.keys(structuredContent ?? {}).sort(),
      resourceTemplateUriTemplate:
        fixtureResourceTemplate.uri_template ??
        fixtureResourceTemplate.uriTemplate,
      resourceTemplatesSeen: true,
      failureIsolation,
    };
  } finally {
    await invokeAppServerMethod(
      options,
      "mcpServer/stop",
      { name: serverName },
      entries,
    ).catch((error) => {
      console.warn(
        `[smoke:mcp-current] fixture stop failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    });
    await invokeAppServerMethod(
      options,
      "mcpServer/delete",
      { id: serverId },
      entries,
    ).catch((error) => {
      console.warn(
        `[smoke:mcp-current] fixture delete failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    });
  }
}

function buildPluginRuntimeCapabilities({
  pluginId,
  serverName,
  includeCallProof,
}) {
  return {
    pluginId,
    skills: [],
    mcpBindings: [
      {
        serverId: serverName,
        toolKey: `${serverName}/echo`,
        provider: "mcp",
        required: true,
        ...(includeCallProof
          ? {
              callProof: {
                arguments: { message: "hello plugin MCP" },
              },
            }
          : {}),
      },
    ],
    workflowBindings: [],
  };
}

function pluginRuntimeInventoryParams({
  pluginId,
  serverName,
  includeCallProof,
}) {
  return {
    caller: "assistant",
    workbench: true,
    browserAssist: true,
    metadata: {
      harness: {
        plugin_runtime_capabilities: buildPluginRuntimeCapabilities({
          pluginId,
          serverName,
          includeCallProof,
        }),
      },
    },
  };
}

function assertPluginRuntimeTarget(inventory, expected) {
  assert(
    inventory && typeof inventory === "object",
    "agentSession/toolInventory/read did not return inventory object",
  );
  const targets = Array.isArray(inventory.plugin_mcp_targets)
    ? inventory.plugin_mcp_targets
    : [];
  assert(targets.length === 1, "tool inventory did not return one MCP target");
  const target = targets[0];
  assert(
    target.pluginId === expected.pluginId,
    "plugin_mcp_targets pluginId drifted",
  );
  assert(
    target.caller === `plugin:${expected.pluginId}`,
    "plugin_mcp_targets caller drifted",
  );
  assert(
    target.expectedToolName === expected.expectedToolName,
    "plugin_mcp_targets expectedToolName drifted",
  );
  assert(
    target.runtimeStatus === "available",
    `plugin_mcp_targets runtimeStatus drifted: ${target.runtimeStatus}`,
  );
  assert(
    target.prepareStatus === "ready",
    `plugin_mcp_targets prepareStatus drifted: ${target.prepareStatus}`,
  );
  assert(
    target.serverAvailable === true &&
      target.serverRunning === true &&
      target.toolAvailable === true,
    "plugin_mcp_targets did not report available running tool",
  );
  assert(
    target.toolListRequest?.caller === `plugin:${expected.pluginId}` &&
      target.toolListRequest?.includeDeferred === true,
    "plugin_mcp_targets toolListRequest drifted",
  );
  assert(
    Array.isArray(target.prepareRequests) &&
      target.prepareRequests.length === 0,
    "available plugin MCP target should not emit prepareRequests",
  );
  return target;
}

async function runPluginRuntimeListProof({
  options,
  entries,
  target,
  expectedToolName,
}) {
  const listProof = await invokeAppServerMethod(
    options,
    "mcpTool/listForContext",
    target.toolListRequest,
    entries,
  );
  const tools = assertArrayField("mcpTool/listForContext", listProof, "tools");
  assert(
    tools.some((tool) => tool?.name === expectedToolName),
    "MCP plugin list proof did not expose expected tool",
  );
  return tools.length;
}

function countExactToolCalls(entries) {
  return entries
    .flatMap((entry) => entry.appServerRequests ?? [])
    .filter((request) => request.method === "mcpServer/tool/call").length;
}

export async function runPluginRuntimeFixtureChecks(options, entries, fixture) {
  const serverId = `mcp-plugin-runtime-${Date.now()}`;
  const serverName = serverId.replace(/[^a-zA-Z0-9_-]/g, "-");
  const pluginId = "mcp-current-plugin";
  const expectedToolName = `mcp__${serverName}__echo`;

  try {
    assertArrayField(
      "mcpServer/create",
      await invokeAppServerMethod(
        options,
        "mcpServer/create",
        {
          server: {
            id: serverId,
            name: serverName,
            description: "Plugin runtime MCP smoke fixture",
            server_config: {
              command: "node",
              args: [fixture.serverPath],
              cwd: fixture.root,
              timeout: 3,
            },
            enabled_lime: true,
            enabled_claude: false,
            enabled_codex: false,
            enabled_gemini: false,
            created_at: Date.now(),
          },
        },
        entries,
      ),
      "servers",
    );

    assertEmptyObject(
      "mcpServer/start",
      await invokeAppServerMethod(
        options,
        "mcpServer/start",
        { name: serverName },
        entries,
      ),
    );
    const threadId = await startFixtureThread(options, entries);

    const explicitInventory = await invokeAppServerMethod(
      options,
      "agentSession/toolInventory/read",
      pluginRuntimeInventoryParams({
        pluginId,
        serverName,
        includeCallProof: true,
      }),
      entries,
    );
    const explicitTarget = assertPluginRuntimeTarget(
      explicitInventory.inventory,
      {
        pluginId,
        expectedToolName,
      },
    );
    assert(
      explicitTarget.callProofRequest?.method === "mcpServer/tool/call",
      "plugin_mcp_targets did not emit explicit callProofRequest",
    );

    const explicitListProofToolCount = await runPluginRuntimeListProof({
      options,
      entries,
      target: explicitTarget,
      expectedToolName,
    });
    const callProofResult = await invokeAppServerMethod(
      options,
      "mcpServer/tool/call",
      { threadId, ...explicitTarget.callProofRequest.params },
      entries,
    );
    const structuredContent = assertToolResult(
      "mcpServer/tool/call",
      callProofResult,
      "echo: hello plugin MCP",
      {
        echoedMessage: "hello plugin MCP",
        messageLength: "hello plugin MCP".length,
      },
    );

    const beforeDefaultProofCallCount = countExactToolCalls(entries);
    const defaultInventory = await invokeAppServerMethod(
      options,
      "agentSession/toolInventory/read",
      pluginRuntimeInventoryParams({
        pluginId,
        serverName,
        includeCallProof: false,
      }),
      entries,
    );
    const defaultTarget = assertPluginRuntimeTarget(
      defaultInventory.inventory,
      {
        pluginId,
        expectedToolName,
      },
    );
    assert(
      defaultTarget.callProofRequest === null ||
        defaultTarget.callProofRequest === undefined,
      "default list proof target should not emit callProofRequest",
    );
    const defaultListProofToolCount = await runPluginRuntimeListProof({
      options,
      entries,
      target: defaultTarget,
      expectedToolName,
    });
    const afterDefaultProofCallCount = countExactToolCalls(entries);
    assert(
      afterDefaultProofCallCount === beforeDefaultProofCallCount,
      "default MCP list proof unexpectedly called a tool",
    );

    return {
      serverId,
      serverName,
      pluginId,
      expectedToolName,
      runtimeStatus: explicitTarget.runtimeStatus,
      prepareStatus: explicitTarget.prepareStatus,
      explicitCallProofSeen: true,
      explicitListProofToolCount,
      explicitCallProofStructuredContent: sanitizeJson(structuredContent),
      defaultListProofToolCount,
      defaultProofDidNotCallTool: true,
      prepareRequestsWhenAvailable: explicitTarget.prepareRequests.length,
    };
  } finally {
    await invokeAppServerMethod(
      options,
      "mcpServer/stop",
      { name: serverName },
      entries,
    ).catch((error) => {
      console.warn(
        `[smoke:mcp-current] plugin runtime fixture stop failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    });
    await invokeAppServerMethod(
      options,
      "mcpServer/delete",
      { id: serverId },
      entries,
    ).catch((error) => {
      console.warn(
        `[smoke:mcp-current] plugin runtime fixture delete failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    });
  }
}
