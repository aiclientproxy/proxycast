import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { once } from "node:events";

import { afterEach, describe, expect, it } from "vitest";

import {
  DEFAULT_FIXTURE_API_KEY,
  DEFAULT_FIXTURE_MODEL,
  startOpenAiCompatibleFixtureServer,
} from "./openai-compatible-fixture-server.mjs";

const runningServers = [];

const STRUCTURED_OUTPUT_TOOL = {
  type: "function",
  function: {
    name: "StructuredOutput",
    parameters: {
      type: "object",
      properties: {},
    },
  },
};

async function startFixture(options) {
  const fixture = await startOpenAiCompatibleFixtureServer(options);
  runningServers.push(fixture);
  return fixture;
}

afterEach(async () => {
  while (runningServers.length > 0) {
    const fixture = runningServers.pop();
    await fixture.close();
  }
});

describe("openai-compatible-fixture-server", () => {
  it("不得通过强杀 active connection 掩盖 provider 生命周期缺陷", () => {
    const source = fs.readFileSync(
      path.resolve("scripts/lib/openai-compatible-fixture-server.mjs"),
      "utf8",
    );
    expect(source).toContain("closeIdleConnections");
    expect(source).not.toContain("closeAllConnections");
  });

  it("应返回 Direct providerConfig，默认只指向 localhost fixture", async () => {
    const fixture = await startFixture();

    expect(fixture.baseUrl).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
    expect(fixture.provider).toEqual({
      providerPreference: "fixture-openai",
      providerName: "openai",
      modelPreference: DEFAULT_FIXTURE_MODEL,
      source: "localhost-fixture",
      providerConfig: {
        providerId: "fixture-openai",
        providerName: "openai",
        modelName: DEFAULT_FIXTURE_MODEL,
        apiKey: DEFAULT_FIXTURE_API_KEY,
        baseUrl: fixture.baseUrl,
        toolCallStrategy: "native",
        modelCapabilities: {
          capabilities: {
            vision: true,
            tools: true,
            streaming: true,
            jsonMode: true,
            functionCalling: true,
            reasoning: false,
            reasoningEffort: null,
          },
          taskFamilies: ["chat"],
          inputModalities: ["text"],
          outputModalities: ["text"],
          runtimeFeatures: ["streaming", "tool_calling"],
        },
      },
    });
  });

  it("模型发现应返回可执行的显式 chat capability", async () => {
    const fixture = await startFixture();
    const response = await fetch(`${fixture.baseUrl}/v1/models`, {
      headers: {
        authorization: `Bearer ${DEFAULT_FIXTURE_API_KEY}`,
      },
    });

    expect(response.ok).toBe(true);
    await expect(response.json()).resolves.toMatchObject({
      data: [
        {
          id: DEFAULT_FIXTURE_MODEL,
          task_families: ["chat"],
          input_modalities: ["text"],
          output_modalities: ["text"],
          runtime_features: ["streaming", "tool_calling"],
          capabilities: {
            tools: true,
            streaming: true,
            function_calling: true,
          },
        },
      ],
    });
  });

  it("应支持 OpenAI-compatible streaming chat completions", async () => {
    const fixture = await startFixture({ content: "MO_OK_STREAM" });
    const response = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${DEFAULT_FIXTURE_API_KEY}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "user", content: "ping" }],
        stream: true,
      }),
    });

    expect(response.ok).toBe(true);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    const text = await response.text();
    expect(text).toContain("MO_OK_STREAM");
    expect(text).toContain("data: [DONE]");
    expect(fixture.requests).toHaveLength(1);
    expect(fixture.requests[0]).toMatchObject({
      method: "POST",
      path: "/v1/chat/completions",
      authorization: `Bearer ${DEFAULT_FIXTURE_API_KEY}`,
    });
    expect(fixture.requests[0].body.stream).toBe(true);
  });

  it("关闭 fixture 时不等待已完成请求的 keep-alive 超时", async () => {
    const fixture = await startOpenAiCompatibleFixtureServer({
      content: "MO_OK_CLOSE",
    });
    const response = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "user", content: "ping" }],
      }),
    });
    expect(response.ok).toBe(true);
    await response.json();

    const closeStartedAt = Date.now();
    await expect(fixture.close()).resolves.toBeUndefined();
    expect(Date.now() - closeStartedAt).toBeLessThan(2_000);
  });

  it("关闭 fixture 时只清理从未承载 HTTP request 的预连接", async () => {
    const fixture = await startOpenAiCompatibleFixtureServer();
    const { port } = new URL(fixture.baseUrl);
    const socket = net.createConnection(Number(port), "127.0.0.1");
    await once(socket, "connect");
    const socketClosed = new Promise((resolve) =>
      socket.once("close", resolve),
    );
    let socketError = null;
    socket.once("error", (error) => {
      socketError = error;
    });

    const closeStartedAt = Date.now();
    await expect(fixture.close()).resolves.toBeUndefined();
    await socketClosed;
    expect(Date.now() - closeStartedAt).toBeLessThan(2_000);
    expect(socket.destroyed).toBe(true);
    expect([null, "ECONNRESET"]).toContain(socketError?.code || null);
  });

  it("应支持非 streaming JSON chat completions，方便后续 smoke 复用", async () => {
    const fixture = await startFixture({ content: "MO_OK_JSON" });
    const response = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "user", content: "ping" }],
      }),
    });

    expect(response.ok).toBe(true);
    const payload = await response.json();
    expect(payload.choices[0].message.content).toBe("MO_OK_JSON");
    expect(payload.usage.total_tokens).toBe(2);
  });

  it("应在请求 StructuredOutput 工具时返回非 streaming tool call", async () => {
    const fixture = await startFixture();
    const response = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "user", content: "draft artifact" }],
        tools: [STRUCTURED_OUTPUT_TOOL],
      }),
    });

    expect(response.ok).toBe(true);
    const payload = await response.json();
    const choice = payload.choices[0];
    const toolCall = choice.message.tool_calls[0];
    const argumentsPayload = JSON.parse(toolCall.function.arguments);

    expect(choice.finish_reason).toBe("tool_calls");
    expect(choice.message.content).toBeNull();
    expect(toolCall.type).toBe("function");
    expect(toolCall.function.name).toBe("StructuredOutput");
    expect(argumentsPayload).toMatchObject({
      type: "artifact_document_draft",
      document: {
        schemaVersion: "artifact_document.v1",
        kind: "report",
        title: "Offline Fixture Artifact Report",
        status: "ready",
        language: "zh-CN",
      },
    });
    expect(argumentsPayload.document.blocks[0]).toMatchObject({
      id: "summary",
      type: "rich_text",
    });
    expect(argumentsPayload.document.sources[0]).toMatchObject({
      id: "localhost-fixture",
      type: "tool",
      label: "localhost OpenAI-compatible fixture",
      reliability: "primary",
    });
  });

  it("应在请求 StructuredOutput 工具时返回 streaming tool call SSE", async () => {
    const fixture = await startFixture();
    const response = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "user", content: "draft artifact" }],
        tools: [STRUCTURED_OUTPUT_TOOL],
        stream: true,
      }),
    });

    expect(response.ok).toBe(true);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    const text = await response.text();
    expect(text).toContain('"tool_calls"');
    expect(text).toContain('"StructuredOutput"');
    expect(text).toContain('"finish_reason":"tool_calls"');
    expect(text).toContain("data: [DONE]");
  });

  it("应支持 scripted streaming tool call，供真实 runtime 编程 smoke 复用", async () => {
    const fixture = await startFixture({
      scriptedResponses: [
        {
          type: "tool_call",
          id: "call-read-fixture",
          name: "Read",
          arguments: {
            path: ".lime/qc/code-runtime-fixture/src/greeting.ts",
          },
        },
        {
          type: "text",
          content: "CODE_RUNTIME_OK",
        },
      ],
    });

    const first = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "user", content: "fix code" }],
        tools: [
          {
            type: "function",
            function: {
              name: "Read",
              parameters: { type: "object", properties: {} },
            },
          },
        ],
        stream: true,
      }),
    });

    expect(first.ok).toBe(true);
    const firstText = await first.text();
    expect(firstText).toContain('"tool_calls"');
    expect(firstText).toContain('"Read"');
    expect(firstText).toContain(
      ".lime/qc/code-runtime-fixture/src/greeting.ts",
    );

    const second = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "tool", content: "read ok" }],
        stream: true,
      }),
    });

    expect(second.ok).toBe(true);
    const secondText = await second.text();
    expect(secondText).toContain("CODE_RUNTIME_OK");
    expect(fixture.requests).toHaveLength(2);
  });

  it("scripted tool call 默认必须确认目标工具已进入 provider request", async () => {
    const fixture = await startFixture({
      scriptedResponses: [
        {
          type: "tool_call",
          id: "call-read-fixture",
          name: "Read",
          arguments: {
            path: "README.md",
          },
        },
      ],
    });

    const response = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "user", content: "read a file" }],
        tools: [],
        stream: true,
      }),
    });

    expect(response.ok).toBe(false);
    expect(response.status).toBe(400);
    const payload = await response.json();
    expect(payload.error.message).toContain("provider request tools=<none>");
  });

  it("scripted tool call 可延迟到下一次有工具面的请求，避免中间模型请求消耗脚本", async () => {
    const fixture = await startFixture({
      content: "INTERMEDIATE_OK",
      deferScriptedToolCallsUntilAvailable: true,
      scriptedResponses: [
        {
          type: "tool_call",
          id: "call-read-fixture",
          name: "Read",
          arguments: { path: "README.md" },
        },
        {
          type: "tool_call",
          id: "call-write-fixture",
          name: "Write",
          arguments: { path: "out.txt", content: "ok" },
        },
      ],
    });

    const first = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "user", content: "read" }],
        tools: [
          {
            type: "function",
            function: {
              name: "Read",
              parameters: { type: "object", properties: {} },
            },
          },
        ],
        stream: true,
      }),
    });
    expect(first.ok).toBe(true);
    expect(await first.text()).toContain('"Read"');

    const intermediate = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "user", content: "classify" }],
        stream: false,
      }),
    });
    expect(intermediate.ok).toBe(true);
    expect((await intermediate.json()).choices[0].message.content).toBe(
      "INTERMEDIATE_OK",
    );

    const second = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "tool", content: "read ok" }],
        tools: [
          {
            type: "function",
            function: {
              name: "Write",
              parameters: { type: "object", properties: {} },
            },
          },
        ],
        stream: true,
      }),
    });
    expect(second.ok).toBe(true);
    expect(await second.text()).toContain('"Write"');
    expect(fixture.requests).toHaveLength(3);
  });

  it("scripted response 函数应能基于上一轮 tool result 动态生成下一次 tool call", async () => {
    const fixture = await startFixture({
      scriptedResponses: [
        {
          type: "tool_call",
          id: "call-background-bash",
          name: "Bash",
          arguments: {
            command: "node -e \"setInterval(() => console.log('tick'), 1000)\"",
            background: true,
          },
        },
        ({ body, requestIndex, requests }) => {
          expect(requestIndex).toBe(1);
          expect(requests).toHaveLength(2);
          const messagesText = (body.messages || [])
            .map((message) =>
              typeof message?.content === "string"
                ? message.content
                : JSON.stringify(message?.content || ""),
            )
            .join("\n");
          const taskId = messagesText.match(/"task_id"\s*:\s*"([^"]+)"/)?.[1];
          if (!taskId) {
            throw new Error("dynamic scripted response missing task_id");
          }
          return {
            type: "tool_call",
            id: "call-dynamic-task-output",
            name: "TaskOutput",
            arguments: {
              task_id: taskId,
              block: true,
              timeout: 2000,
            },
          };
        },
      ],
    });

    const first = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [{ role: "user", content: "start background task" }],
        tools: [
          {
            type: "function",
            function: {
              name: "Bash",
              parameters: { type: "object", properties: {} },
            },
          },
        ],
        stream: true,
      }),
    });

    expect(first.ok).toBe(true);
    expect(await first.text()).toContain('"Bash"');

    const second = await fetch(`${fixture.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_FIXTURE_MODEL,
        messages: [
          {
            role: "tool",
            content: JSON.stringify({
              task_id: "runtime-background-task-42",
              status: "running",
            }),
          },
        ],
        tools: [
          {
            type: "function",
            function: {
              name: "TaskOutput",
              parameters: { type: "object", properties: {} },
            },
          },
        ],
        stream: true,
      }),
    });

    expect(second.ok).toBe(true);
    const secondText = await second.text();
    expect(secondText).toContain('"TaskOutput"');
    expect(secondText).toContain("runtime-background-task-42");
    expect(fixture.requests).toHaveLength(2);
  });
});
