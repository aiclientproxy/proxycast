import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";

export async function writeMcpFixture({ initializeDelayMs = 0 } = {}) {
  if (!Number.isFinite(initializeDelayMs) || initializeDelayMs < 0) {
    throw new Error("initializeDelayMs must be a non-negative number");
  }
  const root = await fsp.mkdtemp(path.join(os.tmpdir(), "lime-mcp-current-"));
  const serverPath = path.join(root, "mcp-current-fixture.mjs");
  await fsp.writeFile(
    serverPath,
    `import readline from "node:readline";

const INITIALIZE_DELAY_MS = ${Math.floor(initializeDelayMs)};

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

function send(message) {
  process.stdout.write(\`\${JSON.stringify(message)}\\n\`);
}

function result(id, value) {
  send({ jsonrpc: "2.0", id, result: value });
}

function error(id, code, message) {
  send({ jsonrpc: "2.0", id, error: { code, message } });
}

rl.on("line", (line) => {
  if (!line.trim()) return;
  const message = JSON.parse(line);
  const { id, method, params } = message;

  if (method === "initialize") {
    const respond = () =>
      result(id, {
        protocolVersion: "2025-03-26",
        capabilities: {
          tools: {},
          resources: {},
        },
        serverInfo: {
          name: "fixture-mcp",
          version: "1.0.0",
        },
      });
    if (INITIALIZE_DELAY_MS > 0) {
      setTimeout(respond, INITIALIZE_DELAY_MS);
    } else {
      respond();
    }
    return;
  }

  if (method === "notifications/initialized") return;

  if (method === "tools/list") {
    result(id, {
      tools: [
        {
          name: "echo",
          description: "Echo a message for current MCP tests",
          inputSchema: {
            type: "object",
            "x-lime": {
              deferred_loading: true,
              always_visible: true,
              allowed_callers: [
                "assistant",
                "tool_search",
                "plugin:mcp-current-plugin"
              ],
            },
            properties: {
              message: { type: "string" },
            },
          },
          outputSchema: {
            type: "object",
            properties: {
              echoedMessage: { type: "string" },
              messageLength: { type: "number" },
              fixture: {
                type: "object",
                properties: {
                  server: { type: "string" },
                  tool: { type: "string" },
                },
                required: ["server", "tool"],
                additionalProperties: false,
              },
            },
            required: ["echoedMessage", "messageLength", "fixture"],
            additionalProperties: false,
          },
        },
      ],
    });
    return;
  }

  if (method === "tools/call") {
    const message = params?.arguments?.message ?? "";
    result(id, {
      content: [
        {
          type: "text",
          text: \`echo: \${message}\`,
        },
      ],
      structuredContent: {
        echoedMessage: message,
        messageLength: String(message).length,
        fixture: {
          server: "mcp-current-fixture",
          tool: "echo",
        },
      },
      isError: false,
    });
    return;
  }

  if (method === "resources/list") {
    result(id, {
      resources: [
        {
          uri: "fixture://status",
          name: "status",
          description: "Current MCP fixture status",
          mimeType: "text/plain",
        },
      ],
    });
    return;
  }

  if (method === "resources/templates/list") {
    result(id, {
      resourceTemplates: [
        {
          uriTemplate: "fixture://item/{id}",
          name: "fixture-item",
          title: "Fixture Item",
          description: "Current MCP fixture resource template",
          mimeType: "text/plain",
        },
      ],
    });
    return;
  }

  if (method === "resources/read") {
    result(id, {
      contents: [
        {
          uri: params?.uri ?? "fixture://status",
          mimeType: "text/plain",
          text: "fixture resource ok",
        },
      ],
    });
    return;
  }

  error(id, -32601, \`unsupported fixture method: \${method}\`);
});
`,
    "utf8",
  );
  return { root, serverPath };
}
