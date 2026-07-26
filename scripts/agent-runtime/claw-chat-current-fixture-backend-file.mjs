import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { withElectronFixtureSystemPath } from "../lib/electron-fixture-runtime-env.mjs";
import {
  ASSISTANT_DONE_TEXT,
  FIXTURE_MODEL,
  IMAGE_COMMAND_PRESENTATION_CAPTION,
  IMAGE_COMMAND_PRESENTATION_INTRO,
  IMAGE_FIXTURE_MODEL,
  NEWS_PROMPT,
  TEXT_PROVIDER_FIXTURE_API_KEY,
  TEXT_FIXTURE_PROVIDER_NAME,
} from "./claw-chat-current-fixture-constants.mjs";
import { writeFixtureBackend } from "./claw-chat-current-fixture-backend-script.mjs";
import {
  buildSoulStyleFixtureAssistantText,
  summarizeSoulPromptMarkers,
} from "./claw-chat-current-fixture-soul-style.mjs";
import {
  ACTIVE_STEER_DONE_TEXT,
  ACTIVE_STEER_FINAL_TEXT,
  ACTIVE_STEER_FIRST_TEXT,
  ACTIVE_STEER_INITIAL_PROMPT,
  ACTIVE_STEER_INPUT,
} from "./claw-chat-current-fixture-active-steer.mjs";

export const LOCAL_IMAGE_SERVER_API_KEY = "pc_claw_image_fixture_local_key";
export const IMAGE_PROVIDER_FIXTURE_API_KEY = "sk-claw-image-fixture";
export const IMAGE_PROVIDER_FIXTURE_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAbCAMAAAANt/xAAAAAsVBMVEW398yz9M6v8tCr79Kn7dSj6taf6Nif59eE0bhWqYEuh1Efej+b5dqH1MEuh1KX4tx3x7IxhU2Bt4zV6s33/+iT4N6B0MVsqXuP3eCP3d8thlKL2+J1x8CH2ORMo4aD1uYrhVN/0+d70OkqhFR3zutFnolzy+1iuslvye9vyO8phFRrxvFfutVnw/NUsMSCt4xjwfVYttgng1VfvvdSsNE7mI8nglVbvPlXuftTt/1PtP9WydZkAAAA4UlEQVQ4y5XSx27DQAwE0EmcuI2binvv3XFP+f8PSwIb1EqgDnxnDhZLDmD1YoRXI2SM8GaEdyNkY3L5QpHFQj6XTYOSq1zhQ6VcSoGqo0Z6fhAGvkfWqjrUIw2yGT40yUZdhZZod2T+P9FptzToih69MOKx19WgLwb0nYDPQV+DoRgxcAIBR0MNxoIMXeRYg4mYJl6YTjSYiXniD/OZBguxTGxpudBgJdab2B0265UG28gudundVoW94/Ds0sdflw57HY6u0/nZ1vPpmAKXuOvtzvvtekmFTyN8GeHbCD9Gv4DxhnNTaC+HAAAAAElFTkSuQmCC";

function writeFixtureConfig(configPath, overrides = {}) {
  const serverHost = overrides.serverHost ?? "127.0.0.1";
  const serverPort = overrides.serverPort ?? 8999;
  const serverApiKey = overrides.serverApiKey ?? LOCAL_IMAGE_SERVER_API_KEY;
  const imageProviderId = String(overrides.imageProviderId ?? "").trim();
  const imageModelId = String(overrides.imageModelId ?? "").trim();
  const soulStyleProfileId = String(overrides.soulStyleProfileId ?? "").trim();
  const imageDefaults = ["      allowFallback: false"];
  if (imageProviderId) {
    imageDefaults.push(`      preferredProviderId: ${imageProviderId}`);
  }
  if (imageModelId) {
    imageDefaults.push(`      preferredModelId: ${imageModelId}`);
  }
  fs.mkdirSync(path.dirname(configPath), { recursive: true });
  fs.writeFileSync(
    configPath,
    [
      "server:",
      `  host: ${serverHost}`,
      `  port: ${serverPort}`,
      `  api_key: ${serverApiKey}`,
      "workspace_preferences:",
      "  media_defaults:",
      "    image:",
      ...imageDefaults,
      ...(soulStyleProfileId
        ? [
            "memory:",
            "  enabled: true",
            "  soul:",
            "    enabled: true",
            `    style_profile_id: ${soulStyleProfileId}`,
            "    imported_from: manual",
          ]
        : []),
      "",
    ].join("\n"),
  );
}

export function createTempRuntimeEnv() {
  const tempRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "claw-chat-current-fixture-"),
  );
  const home = path.join(tempRoot, "home");
  const xdgConfigHome = path.join(tempRoot, "xdg-config");
  const xdgDataHome = path.join(tempRoot, "xdg-data");
  const localAppData = path.join(tempRoot, "local-app-data");
  const roamingAppData = path.join(tempRoot, "roaming-app-data");
  const electronUserDataDir = path.join(tempRoot, "electron-user-data");
  const agentRoot = path.join(electronUserDataDir, "app-server");
  const backendPath = path.join(tempRoot, "claw-chat-backend.mjs");
  const backendLedgerPath = path.join(tempRoot, "claw-chat-backend.jsonl");
  const cancelSignalPath = path.join(tempRoot, "claw-chat-cancel.signal");
  const mediaReferenceSourcePath = path.join(
    tempRoot,
    "fixture-media-reference.png",
  );

  for (const dir of [
    home,
    xdgConfigHome,
    xdgDataHome,
    localAppData,
    roamingAppData,
    electronUserDataDir,
    agentRoot,
  ]) {
    fs.mkdirSync(dir, { recursive: true });
  }
  const configPath = path.join(xdgConfigHome, "lime", "config.yaml");
  const homeConfigPath = path.join(home, ".config", "lime", "config.yaml");
  const macConfigPath = path.join(
    home,
    "Library",
    "Application Support",
    "lime",
    "config.yaml",
  );
  writeFixtureConfig(configPath);
  writeFixtureConfig(homeConfigPath);
  writeFixtureConfig(macConfigPath);
  fs.writeFileSync(backendLedgerPath, "");
  fs.writeFileSync(
    mediaReferenceSourcePath,
    Buffer.from(IMAGE_PROVIDER_FIXTURE_DATA_URL.split(",")[1] || "", "base64"),
  );
  writeFixtureBackend(backendPath, { mediaReferenceSourcePath });

  return {
    tempRoot,
    agentRoot,
    electronUserDataDir,
    backendPath,
    backendLedgerPath,
    cancelSignalPath,
    mediaReferenceSourcePath,
    configPath,
    macConfigPath,
    homeConfigPath,
    writeFixtureConfig: (overrides = {}) => {
      writeFixtureConfig(configPath, overrides);
      writeFixtureConfig(homeConfigPath, overrides);
      writeFixtureConfig(macConfigPath, overrides);
    },
    env: withElectronFixtureSystemPath({
      ...process.env,
      HOME: home,
      XDG_CONFIG_HOME: xdgConfigHome,
      XDG_DATA_HOME: xdgDataHome,
      APPDATA: roamingAppData,
      LOCALAPPDATA: localAppData,
      LIME_AGENT_RUNTIME_ROOT: agentRoot,
    }),
  };
}

export async function startImageProviderFixtureServer() {
  const requests = [];
  const server = http.createServer((request, response) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
    });
    request.on("end", () => {
      requests.push({
        method: request.method,
        url: request.url,
        headers: request.headers,
        authorization: request.headers.authorization ? "present" : "missing",
        body,
      });

      if (
        request.method !== "POST" ||
        request.url !== "/v1/images/generations"
      ) {
        response.writeHead(404, { "content-type": "application/json" });
        response.end(JSON.stringify({ error: { message: "not found" } }));
        return;
      }

      response.writeHead(200, { "content-type": "application/json" });
      response.end(
        JSON.stringify({
          created: Math.floor(Date.now() / 1000),
          data: [
            {
              url: IMAGE_PROVIDER_FIXTURE_DATA_URL,
              revised_prompt: "fixture revised prompt",
            },
          ],
        }),
      );
    });
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });

  const address = server.address();
  const port = address && typeof address === "object" ? address.port : null;
  if (!port) {
    await new Promise((resolve) => server.close(resolve));
    throw new Error("图片 Provider fixture server 未返回端口");
  }

  return {
    baseUrl: `http://127.0.0.1:${port}/v1`,
    host: "127.0.0.1",
    port,
    requestCount: () => requests.length,
    requests: () =>
      requests.map((entry) => ({
        method: entry.method,
        url: entry.url,
        authorization: entry.authorization,
        providerId: (() => {
          try {
            return JSON.parse(entry.body || "{}").provider_id ?? null;
          } catch {
            return null;
          }
        })(),
        headerProviderId: (() => {
          const value = entry.headers?.["x-provider-id"];
          return Array.isArray(value) ? value[0] : (value ?? null);
        })(),
        model: (() => {
          try {
            return JSON.parse(entry.body || "{}").model ?? null;
          } catch {
            return null;
          }
        })(),
        bodyIncludesModel: entry.body.includes(IMAGE_FIXTURE_MODEL),
      })),
    close: () =>
      new Promise((resolve) => {
        server.close(() => resolve());
      }),
  };
}

function fixtureTextForChatRequest(body) {
  const serialized = typeof body === "string" ? body : JSON.stringify(body);
  if (serialized.includes(ACTIVE_STEER_INPUT)) {
    return `${ACTIVE_STEER_FINAL_TEXT}\n${ACTIVE_STEER_DONE_TEXT}`;
  }
  if (serialized.includes(ACTIVE_STEER_INITIAL_PROMPT)) {
    return ACTIVE_STEER_FIRST_TEXT;
  }
  const presentationText = JSON.stringify({
    assistant_intro: IMAGE_COMMAND_PRESENTATION_INTRO,
    completion_caption: IMAGE_COMMAND_PRESENTATION_CAPTION,
  });
  if (
    serialized.includes("image_task_presentation.v1") ||
    serialized.includes(
      "Generate user-visible copy for one image generation turn.",
    ) ||
    serialized.includes("image_command_presentation")
  ) {
    return presentationText;
  }
  const soulStyleTranscriptText = buildSoulStyleFixtureAssistantText(
    serialized,
    ASSISTANT_DONE_TEXT,
  );
  if (soulStyleTranscriptText) {
    return soulStyleTranscriptText;
  }
  if (serialized.includes(NEWS_PROMPT)) {
    return [
      "今日国际新闻简要整理：",
      "全球市场继续关注能源、供应链和主要经济体政策变化。",
      "国际组织呼吁各方保持沟通，降低地区冲突外溢风险。",
      ASSISTANT_DONE_TEXT,
    ].join("\n");
  }
  return ASSISTANT_DONE_TEXT;
}

function previewText(value, maxLength = 260) {
  const text =
    typeof value === "string" ? value : JSON.stringify(value ?? null);
  return text.replace(/\s+/g, " ").slice(0, maxLength);
}

function readChatContentText(content) {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        if (part && typeof part === "object") {
          return (
            part.text ??
            part.content ??
            part.input_text ??
            part.output_text ??
            ""
          );
        }
        return "";
      })
      .filter(Boolean)
      .join(" ");
  }
  if (content && typeof content === "object") {
    return content.text ?? content.content ?? "";
  }
  return "";
}

function summarizeChatCompletionRequestBody(body, expectedSoulStyle) {
  let parsed = {};
  try {
    parsed = JSON.parse(body || "{}");
  } catch {
    parsed = {};
  }
  const messages = Array.isArray(parsed.messages) ? parsed.messages : [];
  const serialized = body || "";
  return {
    stream: parsed.stream ?? null,
    messageCount: messages.length,
    model: parsed.model ?? null,
    responseFormatType:
      parsed.response_format?.type ?? parsed.responseFormat?.type ?? null,
    toolChoice: parsed.tool_choice ?? parsed.toolChoice ?? null,
    soulMarkers: summarizeSoulPromptMarkers(serialized, expectedSoulStyle),
    bodyIncludesPresentationContract:
      serialized.includes("image_task_presentation.v1") ||
      serialized.includes(
        "Generate user-visible copy for one image generation turn.",
      ) ||
      serialized.includes("image_command_presentation"),
    bodyIncludesActiveSteerInitialPrompt: serialized.includes(
      ACTIVE_STEER_INITIAL_PROMPT,
    ),
    bodyIncludesActiveSteerInput: serialized.includes(ACTIVE_STEER_INPUT),
    messages: messages.slice(-4).map((message) => ({
      role: message?.role ?? null,
      contentLength: readChatContentText(message?.content).length,
    })),
  };
}

function openaiChatCompletionChunk({ content, finishReason = null }) {
  return {
    id: "chatcmpl-claw-text-fixture",
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: FIXTURE_MODEL,
    choices: [
      {
        index: 0,
        delta: finishReason
          ? {}
          : {
              role: "assistant",
              content,
            },
        finish_reason: finishReason,
      },
    ],
  };
}

function openaiChatCompletionUsageChunk() {
  return {
    id: "chatcmpl-claw-text-fixture",
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: FIXTURE_MODEL,
    choices: [],
    usage: {
      prompt_tokens: 31_000,
      completion_tokens: 0,
      total_tokens: 31_000,
    },
  };
}

function openaiChatCompletionBody(content) {
  return {
    id: "chatcmpl-claw-text-fixture",
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: FIXTURE_MODEL,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content,
        },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 31_000,
      completion_tokens: 0,
      total_tokens: 31_000,
    },
  };
}

function openaiChatCompletionSseBody(content) {
  const firstChunk = JSON.stringify(
    openaiChatCompletionChunk({ content, finishReason: null }),
  );
  const finalChunk = JSON.stringify(
    openaiChatCompletionChunk({ content: "", finishReason: "stop" }),
  );
  const usageChunk = JSON.stringify(openaiChatCompletionUsageChunk());
  return `data: ${firstChunk}\n\ndata: ${finalChunk}\n\ndata: ${usageChunk}\n\ndata: [DONE]\n\n`;
}

function openaiModelsBody() {
  return {
    object: "list",
    data: [
      {
        id: FIXTURE_MODEL,
        object: "model",
        owned_by: TEXT_FIXTURE_PROVIDER_NAME,
        display_name: FIXTURE_MODEL,
        input_modalities: ["text", "image"],
        output_modalities: ["text"],
        task_families: ["chat", "vision_understanding"],
        runtime_features: ["streaming"],
        capabilities: {
          vision: true,
          streaming: true,
        },
      },
    ],
  };
}

export async function startTextProviderFixtureServer({
  soulStyleExpectation = null,
  activeSteerDelayMs = 0,
} = {}) {
  const requests = [];
  const expectedAuthorization = `Bearer ${TEXT_PROVIDER_FIXTURE_API_KEY}`;
  const server = http.createServer((request, response) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
    });
    request.on("end", () => {
      requests.push({
        method: request.method,
        url: request.url,
        headers: request.headers,
        authorization: request.headers.authorization ? "present" : "missing",
        body,
      });

      if (request.headers.authorization !== expectedAuthorization) {
        response.writeHead(401, { "content-type": "application/json" });
        response.end(
          JSON.stringify({
            error: {
              message: "fixture text provider requires its scoped credential",
              type: "authentication_error",
            },
          }),
        );
        return;
      }

      const pathname = (() => {
        try {
          return new URL(request.url || "/", "http://127.0.0.1").pathname;
        } catch {
          return request.url || "/";
        }
      })();
      if (
        request.method === "GET" &&
        ["/models", "/v1/models"].includes(pathname)
      ) {
        response.writeHead(200, { "content-type": "application/json" });
        response.end(JSON.stringify(openaiModelsBody()));
        return;
      }
      if (
        request.method !== "POST" ||
        !["/chat/completions", "/v1/chat/completions"].includes(pathname)
      ) {
        response.writeHead(404, { "content-type": "application/json" });
        response.end(JSON.stringify({ error: { message: "not found" } }));
        return;
      }

      let parsedBody = {};
      try {
        parsedBody = JSON.parse(body || "{}");
      } catch {
        parsedBody = {};
      }
      const content = fixtureTextForChatRequest(body);
      if (parsedBody.stream === false) {
        response.writeHead(200, { "content-type": "application/json" });
        response.end(JSON.stringify(openaiChatCompletionBody(content)));
        return;
      }

      if (
        activeSteerDelayMs > 0 &&
        body.includes(ACTIVE_STEER_INITIAL_PROMPT) &&
        !body.includes(ACTIVE_STEER_INPUT)
      ) {
        const firstChunk = JSON.stringify(
          openaiChatCompletionChunk({
            content: ACTIVE_STEER_FIRST_TEXT,
            finishReason: null,
          }),
        );
        const finalChunk = JSON.stringify(
          openaiChatCompletionChunk({ content: "", finishReason: "stop" }),
        );
        const usageChunk = JSON.stringify(openaiChatCompletionUsageChunk());
        response.writeHead(200, {
          "content-type": "text/event-stream; charset=utf-8",
          "cache-control": "no-cache",
          connection: "close",
        });
        response.write(`data: ${firstChunk}\n\n`);
        setTimeout(() => {
          response.end(
            `data: ${finalChunk}\n\ndata: ${usageChunk}\n\ndata: [DONE]\n\n`,
          );
        }, activeSteerDelayMs);
        return;
      }

      const responseBody = openaiChatCompletionSseBody(content);
      response.writeHead(200, {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache",
        connection: "close",
        "content-length": Buffer.byteLength(responseBody),
      });
      response.end(responseBody);
    });
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });

  const address = server.address();
  const port = address && typeof address === "object" ? address.port : null;
  if (!port) {
    await new Promise((resolve) => server.close(resolve));
    throw new Error("文本 Provider fixture server 未返回端口");
  }

  return {
    baseUrl: `http://127.0.0.1:${port}/v1`,
    host: "127.0.0.1",
    port,
    requestCount: () => requests.length,
    requests: () =>
      requests.map((entry) => ({
        method: entry.method,
        url: entry.url,
        authorization: entry.authorization,
        model: (() => {
          try {
            return JSON.parse(entry.body || "{}").model ?? null;
          } catch {
            return null;
          }
        })(),
        bodySummary: summarizeChatCompletionRequestBody(
          entry.body,
          soulStyleExpectation,
        ),
        bodyIncludesPresentationContract:
          entry.body.includes("image_task_presentation.v1") ||
          entry.body.includes(
            "Generate user-visible copy for one image generation turn.",
          ) ||
          entry.body.includes("image_command_presentation"),
      })),
    close: () =>
      new Promise((resolve) => {
        server.close(() => resolve());
      }),
  };
}
