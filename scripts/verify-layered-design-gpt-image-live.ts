#!/usr/bin/env tsx
/* global Buffer, process */
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import os from "node:os";
import path from "node:path";
import { parse as parseYaml } from "yaml";

import type { MediaTaskArtifactOutput } from "../src/lib/api/mediaTasks";
import {
  applyLayeredDesignImageTaskOutput,
  createLayeredDesignImageTaskRequest,
} from "../src/lib/layered-design/imageTasks";
import { createLayeredDesignAssetGenerationPlan } from "../src/lib/layered-design/generation";
import { createLayeredDesignSeedDocument } from "../src/lib/layered-design/planner";

const DEFAULT_OUTER_MODEL = "gpt-5.5";
const DEFAULT_IMAGE_MODEL = "gpt-images-2";
const DEFAULT_IMAGE_SIZE = "1024x1024";
const DEFAULT_DEV_BRIDGE_INVOKE_URL = "http://127.0.0.1:3030/invoke";
const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const APP_SERVER_METHOD_MODEL_PROVIDER_LIST = "modelProvider/list";
const TINY_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=";

type ImageGenerationTransport = "responses" | "lime_images_gateway";
type ExpectedExecutorMode = "images_api" | "responses_image_generation";

interface CliOptions {
  selfTest: boolean;
  localLimeImageGateway: boolean;
  dryRun: boolean;
  autoProvider: boolean;
  allowExternalImageGeneration: boolean;
  output?: string;
  imageOutput?: string;
  baseUrl?: string;
  providerId?: string;
  devBridgeInvokeUrl?: string;
  limeConfigPath?: string;
  imageModel: string;
  outerModel: string;
  imageSize: string;
  expectedExecutorMode?: ExpectedExecutorMode;
  evidenceSchema?: string;
  verificationLabel?: string;
}

interface ImageGenerationResponse {
  transport: ImageGenerationTransport;
  imageBase64: string;
  imageUrl?: string;
  imageItemId?: string;
  revisedPrompt?: string;
  eventCount: number;
  outputItemCount: number;
}

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    selfTest: false,
    localLimeImageGateway: false,
    dryRun: false,
    autoProvider: false,
    allowExternalImageGeneration: false,
    imageModel: DEFAULT_IMAGE_MODEL,
    outerModel: DEFAULT_OUTER_MODEL,
    imageSize: DEFAULT_IMAGE_SIZE,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--self-test") {
      options.selfTest = true;
      continue;
    }
    if (token === "--lime-local-image-gateway") {
      options.localLimeImageGateway = true;
      continue;
    }
    if (token === "--dry-run") {
      options.dryRun = true;
      continue;
    }
    if (token === "--auto-provider") {
      options.autoProvider = true;
      continue;
    }
    if (token === "--allow-external-image-generation") {
      options.allowExternalImageGeneration = true;
      continue;
    }

    if (!token.startsWith("--")) {
      throw new Error(`未知参数: ${token}`);
    }

    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`缺少 --${key} 的值`);
    }
    index += 1;

    switch (key) {
      case "output":
        options.output = value;
        break;
      case "image-output":
        options.imageOutput = value;
        break;
      case "base-url":
        options.baseUrl = value;
        break;
      case "provider-id":
        if (value.trim().toLowerCase() === "auto") {
          options.autoProvider = true;
        } else {
          options.providerId = value;
        }
        break;
      case "dev-bridge-invoke-url":
        options.devBridgeInvokeUrl = value;
        break;
      case "lime-config":
        options.limeConfigPath = value;
        break;
      case "image-model":
        options.imageModel = value;
        break;
      case "outer-model":
        options.outerModel = value;
        break;
      case "image-size":
        options.imageSize = value;
        break;
      case "expected-executor-mode":
        if (value !== "images_api" && value !== "responses_image_generation") {
          throw new Error(
            "--expected-executor-mode 只能是 images_api 或 responses_image_generation",
          );
        }
        options.expectedExecutorMode = value;
        break;
      case "evidence-schema":
        options.evidenceSchema = value;
        break;
      case "verification-label":
        options.verificationLabel = value;
        break;
      default:
        throw new Error(`未知参数: --${key}`);
    }
  }

  return options;
}

function usage(): string {
  return [
    "用法：",
    "  npm exec -- tsx scripts/verify-layered-design-gpt-image-live.ts --self-test --output /tmp/evidence.json",
    "",
    "真实网关验收：",
    "  IMAGE_API_KEY=... IMAGE_BASE_URL=... npm exec -- tsx scripts/verify-layered-design-gpt-image-live.ts --allow-external-image-generation --output internal/roadmap/ai-layered-design/evidence/gpt-image-live.json --image-output internal/roadmap/ai-layered-design/evidence/gpt-image-live.png",
    "",
    "复用 Lime 本地配图网关验收：",
    "  npm exec -- tsx scripts/verify-layered-design-gpt-image-live.ts --lime-local-image-gateway --provider-id <provider-id> --allow-external-image-generation --output internal/roadmap/ai-layered-design/evidence/gpt-image-live.json --image-output internal/roadmap/ai-layered-design/evidence/gpt-image-live.png",
    "  npm exec -- tsx scripts/verify-layered-design-gpt-image-live.ts --lime-local-image-gateway --auto-provider --allow-external-image-generation --output internal/roadmap/ai-layered-design/evidence/gpt-image-live.json --image-output internal/roadmap/ai-layered-design/evidence/gpt-image-live.png",
    "",
    "可选参数：",
    "  --base-url <Responses 网关基址，通常到 /v1>",
    "  --lime-local-image-gateway <复用本地 Lime /v1/images/generations 配图网关>",
    "  --provider-id <本地 Lime API Key Provider ID；也可传 auto>",
    "  --auto-provider <通过 DevBridge 自动选择 models 命中图片模型且有启用 key 的 provider>",
    "  --dev-bridge-invoke-url <DevBridge invoke URL，默认 http://127.0.0.1:3030/invoke>",
    "  --allow-external-image-generation <确认会调用真实图片接口并可能消耗额度>",
    "  --lime-config <本地 Lime config.yaml 路径>",
    "  --dry-run <只做本地网关/参数预检，不生成图片>",
    "  --image-model gpt-images-2",
    "  --outer-model gpt-5.5",
    "  --image-size 1024x1024",
    "  --expected-executor-mode responses_image_generation|images_api",
    "  --evidence-schema <证据 schema，默认 layered-design-gpt-image-live-evidence@1>",
    "  --verification-label <证据标签，默认 gpt-image-live>",
    "  --image-output <保存生成 PNG 的路径>",
  ].join("\n");
}

function buildResponsesUrl(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  if (!trimmed) {
    throw new Error("base URL 不能为空");
  }
  if (trimmed.endsWith("/responses")) {
    return trimmed;
  }
  return `${trimmed}/responses`;
}

function buildImagesGenerationUrl(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  if (!trimmed) {
    throw new Error("base URL 不能为空");
  }
  if (trimmed.endsWith("/images/generations")) {
    return trimmed;
  }
  if (trimmed.endsWith("/v1")) {
    return `${trimmed}/images/generations`;
  }
  return `${trimmed}/v1/images/generations`;
}

function buildLocalHealthUrl(baseUrl: string): string {
  const parsed = new URL(buildImagesGenerationUrl(baseUrl));
  return `${parsed.origin}/health`;
}

function buildLocalModelsUrl(baseUrl: string): string {
  const parsed = new URL(buildImagesGenerationUrl(baseUrl));
  return `${parsed.origin}/v1/models`;
}

function buildInput(prompt: string, useInputList: boolean): unknown {
  if (!useInputList) {
    return prompt;
  }

  return [
    {
      role: "user",
      content: [{ type: "input_text", text: prompt }],
    },
  ];
}

async function requestResponsesImageGeneration(params: {
  apiKey: string;
  baseUrl: string;
  outerModel: string;
  imageModel: string;
  prompt: string;
  useInputList?: boolean;
}): Promise<Response> {
  return fetch(buildResponsesUrl(params.baseUrl), {
    method: "POST",
    headers: {
      Authorization: `Bearer ${params.apiKey}`,
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      model: params.outerModel,
      input: buildInput(params.prompt, Boolean(params.useInputList)),
      tools: [{ type: "image_generation", model: params.imageModel }],
      stream: true,
    }),
  });
}

async function requestLimeImagesGatewayGeneration(params: {
  apiKey: string;
  baseUrl: string;
  providerId: string;
  imageModel: string;
  imageSize: string;
  prompt: string;
}): Promise<Response> {
  return fetch(buildImagesGenerationUrl(params.baseUrl), {
    method: "POST",
    headers: {
      Authorization: `Bearer ${params.apiKey}`,
      "Content-Type": "application/json",
      "X-Provider-Id": params.providerId,
    },
    body: JSON.stringify({
      prompt: params.prompt,
      model: params.imageModel,
      n: 1,
      size: params.imageSize,
      response_format: "b64_json",
      user: "layered-design-gpt-image-live",
    }),
  });
}

function shouldRetryWithInputList(status: number, text: string): boolean {
  return status === 400 && /input must be a list/i.test(text);
}

function parseSseEvent(rawEvent: string): { eventName?: string; dataText?: string } {
  const lines = rawEvent.split(/\r?\n/);
  const eventName = lines
    .find((line) => line.startsWith("event:"))
    ?.slice("event:".length)
    .trim();
  const dataText = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart())
    .join("\n");

  return { eventName, dataText };
}

async function extractImageGenerationResult(
  response: Response,
): Promise<ImageGenerationResponse> {
  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => "");
    throw new Error(`Responses 图片生成请求失败 ${response.status}: ${text.slice(0, 1000)}`);
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let eventCount = 0;
  let outputItemCount = 0;
  let imageBase64 = "";
  let imageItemId: string | undefined;
  let revisedPrompt: string | undefined;

  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
      const { eventName, dataText } = parseSseEvent(rawEvent);
      if (!eventName || !dataText || dataText.trim() === "[DONE]") {
        continue;
      }
      eventCount += 1;
      if (eventName !== "response.output_item.done") {
        continue;
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(dataText);
      } catch {
        continue;
      }
      const item = (parsed as { item?: Record<string, unknown> }).item;
      if (!item) {
        continue;
      }
      outputItemCount += 1;
      if (item.type !== "image_generation_call" || typeof item.result !== "string") {
        continue;
      }

      imageBase64 = item.result.trim();
      imageItemId = typeof item.id === "string" ? item.id : undefined;
      revisedPrompt =
        typeof item.revised_prompt === "string" ? item.revised_prompt : undefined;
    }
  }

  if (!imageBase64) {
    throw new Error("Responses SSE 流里没有 image_generation_call.result");
  }

  return {
    transport: "responses",
    imageBase64,
    imageItemId,
    revisedPrompt,
    eventCount,
    outputItemCount,
  };
}

function isHttpImageUrl(value: string): boolean {
  try {
    const url = new URL(value.trim());
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

async function downloadImageUrlAsBase64(imageUrl: string): Promise<string> {
  const response = await fetch(imageUrl, {
    headers: {
      Accept: "image/png,image/*",
      "Accept-Encoding": "identity",
    },
  });
  if (!response.ok) {
    throw new Error(`下载远程图片失败 ${response.status}: ${imageUrl}`);
  }

  const bytes = Buffer.from(await response.arrayBuffer());
  return bytes.toString("base64");
}

function extractBase64FromDataUrl(value: string): string | undefined {
  const match = value.trim().match(/^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$/);
  return match?.[1]?.trim();
}

async function extractImagesGatewayResult(response: Response): Promise<ImageGenerationResponse> {
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`Lime 本地配图网关请求失败 ${response.status}: ${text.slice(0, 1000)}`);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`解析 Lime 本地配图网关响应失败: ${String(error)}`);
  }

  const data = (parsed as { data?: Array<Record<string, unknown>> }).data;
  const firstImage = Array.isArray(data) ? data[0] : undefined;
  const rawBase64 =
    typeof firstImage?.b64_json === "string" ? firstImage.b64_json.trim() : undefined;
  const dataUrlBase64 =
    typeof firstImage?.url === "string" ? extractBase64FromDataUrl(firstImage.url) : undefined;
  const remoteImageUrl =
    typeof firstImage?.url === "string" && isHttpImageUrl(firstImage.url)
      ? firstImage.url.trim()
      : undefined;
  const imageBase64 =
    rawBase64 ||
    dataUrlBase64 ||
    (remoteImageUrl ? await downloadImageUrlAsBase64(remoteImageUrl) : undefined);
  if (!imageBase64) {
    throw new Error("Lime 本地配图网关未返回 b64_json、PNG data URL 或远程图片 URL");
  }

  return {
    transport: "lime_images_gateway",
    imageBase64,
    imageUrl: remoteImageUrl,
    revisedPrompt:
      typeof firstImage?.revised_prompt === "string" ? firstImage.revised_prompt : undefined,
    eventCount: 0,
    outputItemCount: 0,
  };
}

async function generateImage(params: {
  transport: ImageGenerationTransport;
  apiKey: string;
  baseUrl: string;
  providerId?: string;
  outerModel: string;
  imageModel: string;
  imageSize: string;
  prompt: string;
}): Promise<ImageGenerationResponse> {
  if (params.transport === "lime_images_gateway") {
    if (!params.providerId) {
      throw new Error("复用 Lime 本地配图网关时必须提供 providerId");
    }
    const response = await requestLimeImagesGatewayGeneration({
      apiKey: params.apiKey,
      baseUrl: params.baseUrl,
      providerId: params.providerId,
      imageModel: params.imageModel,
      imageSize: params.imageSize,
      prompt: params.prompt,
    });
    return extractImagesGatewayResult(response);
  }

  let response = await requestResponsesImageGeneration(params);
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    if (!shouldRetryWithInputList(response.status, text)) {
      throw new Error(`Responses 图片生成请求失败 ${response.status}: ${text.slice(0, 1000)}`);
    }
    response = await requestResponsesImageGeneration({
      ...params,
      useInputList: true,
    });
  }

  return extractImageGenerationResult(response);
}

function createTaskOutput(params: {
  projectRootPath: string;
  taskId: string;
  taskRequest: ReturnType<typeof createLayeredDesignImageTaskRequest>;
  result: ImageGenerationResponse;
}): MediaTaskArtifactOutput {
  const createdAt = new Date().toISOString();
  const imageUrl = `data:image/png;base64,${params.result.imageBase64}`;
  const imageSource =
    params.result.transport === "lime_images_gateway"
      ? "lime_images_gateway"
      : "responses_image_generation";
  return {
    success: true,
    task_id: params.taskId,
    task_type: "image_generate",
    task_family: "image",
    status: "succeeded",
    normalized_status: "succeeded",
    path: `.lime/tasks/image_generate/${params.taskId}.json`,
    absolute_path: path.join(
      params.projectRootPath,
      ".lime/tasks/image_generate",
      `${params.taskId}.json`,
    ),
    artifact_path: `.lime/tasks/image_generate/${params.taskId}.json`,
    absolute_artifact_path: path.join(
      params.projectRootPath,
      ".lime/tasks/image_generate",
      `${params.taskId}.json`,
    ),
    reused_existing: false,
    record: {
      task_id: params.taskId,
      task_type: "image_generate",
      task_family: "image",
      payload: {
        prompt: params.taskRequest.prompt,
        provider_id: params.taskRequest.providerId,
        model: params.taskRequest.model,
        executor_mode: params.taskRequest.executorMode,
        outer_model: params.taskRequest.outerModel,
      },
      status: "succeeded",
      normalized_status: "succeeded",
      created_at: createdAt,
      updated_at: createdAt,
      result: {
        executor_mode: params.taskRequest.executorMode,
        outer_model: params.taskRequest.outerModel,
        images: [
          {
            url: imageUrl,
            revised_prompt: params.result.revisedPrompt,
            source: imageSource,
          },
        ],
        responses: [
          {
            executor_mode: params.taskRequest.executorMode,
            transport: params.result.transport,
            event_count: params.result.eventCount,
            output_item_count: params.result.outputItemCount,
            image_item_id: params.result.imageItemId,
          },
        ],
      },
    },
  };
}

function sha256Short(value: string): string {
  return createHash("sha256").update(value).digest("hex").slice(0, 16);
}

function isPngBuffer(buffer: Buffer): boolean {
  return (
    buffer.length >= 8 &&
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47 &&
    buffer[4] === 0x0d &&
    buffer[5] === 0x0a &&
    buffer[6] === 0x1a &&
    buffer[7] === 0x0a
  );
}

function readEnvOrOption(options: CliOptions): { apiKey: string; baseUrl: string } {
  const apiKey = process.env.IMAGE_API_KEY?.trim();
  const baseUrl = options.baseUrl ?? process.env.IMAGE_BASE_URL?.trim();
  if (!apiKey || !baseUrl) {
    throw new Error(`${usage()}\n\n缺少 IMAGE_API_KEY 或 IMAGE_BASE_URL`);
  }
  return { apiKey, baseUrl };
}

function resolveDefaultLimeConfigPath(): string {
  if (process.env.LIME_CONFIG_PATH?.trim()) {
    return process.env.LIME_CONFIG_PATH.trim();
  }

  const homeDir = os.homedir();
  if (process.platform === "darwin") {
    return path.join(homeDir, "Library/Application Support/lime/config.yaml");
  }
  if (process.platform === "win32") {
    return path.join(process.env.APPDATA ?? path.join(homeDir, "AppData/Roaming"), "lime/config.yaml");
  }
  return path.join(process.env.XDG_CONFIG_HOME ?? path.join(homeDir, ".config"), "lime/config.yaml");
}

function readNestedRecord(value: unknown, key: string): Record<string, unknown> {
  const record = typeof value === "object" && value !== null ? value as Record<string, unknown> : {};
  const nested = record[key];
  return typeof nested === "object" && nested !== null
    ? nested as Record<string, unknown>
    : {};
}

function readOptionalNestedString(value: unknown, pathKeys: string[]): string | undefined {
  let current: unknown = value;
  for (const key of pathKeys) {
    if (typeof current !== "object" || current === null) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[key];
  }
  return typeof current === "string" && current.trim() ? current.trim() : undefined;
}

function normalizeServerHost(host: string | undefined): string {
  const trimmed = host?.trim() || "127.0.0.1";
  return trimmed === "0.0.0.0" || trimmed === "::" ? "127.0.0.1" : trimmed;
}

interface LocalProviderCandidate {
  id?: unknown;
  enabled?: unknown;
  api_key_count?: unknown;
  apiKeys?: unknown;
  api_keys?: unknown;
  models?: unknown;
}

function normalizeProviderModelToken(value: string): string {
  return value.trim().toLowerCase();
}

function readProviderModels(provider: LocalProviderCandidate): string[] {
  const raw = provider.models;
  return Array.isArray(raw)
    ? raw.flatMap((item) => {
        if (typeof item !== "object" || item === null) {
          return [];
        }
        const id = "id" in item && typeof item.id === "string" ? item.id.trim() : "";
        return id ? [id] : [];
      })
    : [];
}

function hasEnabledProviderKey(provider: LocalProviderCandidate): boolean {
  const rawKeys = provider.api_keys ?? provider.apiKeys;
  if (Array.isArray(rawKeys)) {
    return rawKeys.some((item) => {
      if (typeof item !== "object" || item === null) {
        return false;
      }
      return (item as { enabled?: unknown }).enabled !== false;
    });
  }

  return typeof provider.api_key_count === "number" && provider.api_key_count > 0;
}

function providerMatchesImageModel(
  provider: LocalProviderCandidate,
  imageModel: string,
): boolean {
  const target = normalizeProviderModelToken(imageModel);
  return readProviderModels(provider).some((model) => {
    const normalized = normalizeProviderModelToken(model);
    return normalized === target || normalized.endsWith(`/${target}`);
  });
}

async function invokeDevBridge(
  invokeUrl: string,
  cmd: string,
  args: Record<string, unknown> = {},
): Promise<unknown> {
  const response = await fetch(invokeUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ cmd, args }),
  });
  const text = await response.text();
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new Error(`DevBridge 非 JSON 响应: ${text.slice(0, 240)}`);
  }
  const record = typeof payload === "object" && payload !== null
    ? payload as { error?: unknown; result?: unknown }
    : {};
  if (record.error) {
    throw new Error(String(record.error));
  }
  return record.result;
}

let appServerRequestId = 1;

async function invokeAppServerMethod(
  invokeUrl: string,
  method: string,
  params: Record<string, unknown> = {},
): Promise<unknown> {
  const id = `layered-design-provider-${appServerRequestId++}`;
  const result = await invokeDevBridge(invokeUrl, APP_SERVER_HANDLE_JSON_LINES_COMMAND, {
    request: { lines: [`${JSON.stringify({ id, method, params })}\n`] },
  });
  const lines =
    typeof result === "object" &&
    result !== null &&
    Array.isArray((result as { lines?: unknown }).lines)
      ? ((result as { lines: unknown[] }).lines)
      : [];
  const messages = lines
    .map((line) => {
      try {
        return JSON.parse(String(line));
      } catch {
        return null;
      }
    })
    .filter((item): item is Record<string, unknown> => Boolean(item));
  const error = messages.find((message) => message.id === id && message.error);
  if (error) {
    const rpcError = error.error as { message?: unknown } | undefined;
    throw new Error(
      `${method} error: ${String(rpcError?.message || "App Server JSON-RPC error")}`,
    );
  }
  const response = messages.find(
    (message) => message.id === id && Object.hasOwn(message, "result"),
  );
  if (!response) {
    throw new Error(`${method} missing App Server response`);
  }
  return response.result;
}

async function resolveAutoLocalImageProviderId(options: CliOptions): Promise<string> {
  const invokeUrl =
    options.devBridgeInvokeUrl?.trim() ||
    process.env.LIME_DEV_BRIDGE_INVOKE_URL?.trim() ||
    DEFAULT_DEV_BRIDGE_INVOKE_URL;
  let providersRaw: unknown;
  try {
    const providerList = await invokeAppServerMethod(
      invokeUrl,
      APP_SERVER_METHOD_MODEL_PROVIDER_LIST,
    );
    providersRaw =
      typeof providerList === "object" &&
      providerList !== null &&
      Array.isArray((providerList as { providers?: unknown }).providers)
        ? (providerList as { providers: unknown[] }).providers
        : [];
  } catch (error) {
    throw new Error(
      `无法通过 App Server 自动选择图片 provider，请传 --provider-id；${String(error)}`,
    );
  }

  const providers = Array.isArray(providersRaw)
    ? providersRaw.filter(
        (item): item is LocalProviderCandidate =>
          typeof item === "object" && item !== null,
      )
    : [];
  const candidates = providers.filter((provider) => {
    const id = typeof provider.id === "string" ? provider.id.trim() : "";
    return (
      id &&
      provider.enabled !== false &&
      hasEnabledProviderKey(provider) &&
      providerMatchesImageModel(provider, options.imageModel)
    );
  });
  const selected = candidates[0];
  const selectedId = typeof selected?.id === "string" ? selected.id.trim() : "";
  if (!selectedId) {
    throw new Error(
      `未找到 models 包含 ${options.imageModel} 且有启用 API Key 的本地图片 provider，请传 --provider-id`,
    );
  }
  return selectedId;
}

async function readLocalLimeGatewayConfig(options: CliOptions): Promise<{
  apiKey: string;
  baseUrl: string;
  providerId: string;
}> {
  const configPath = path.resolve(options.limeConfigPath ?? resolveDefaultLimeConfigPath());
  const configText = await readFile(configPath, "utf8");
  const config = parseYaml(configText) as Record<string, unknown>;
  const server = readNestedRecord(config, "server");
  const apiKey = typeof server.api_key === "string" ? server.api_key.trim() : "";
  const host = normalizeServerHost(typeof server.host === "string" ? server.host : undefined);
  const port = Number(server.port ?? 8999);
  let providerId =
    options.providerId?.trim() ??
    process.env.IMAGE_PROVIDER_ID?.trim() ??
    readOptionalNestedString(config, [
      "workspace_preferences",
      "media_defaults",
      "image",
      "preferred_provider_id",
    ]);

  if (!apiKey) {
    throw new Error(`Lime config 缺少 server.api_key: ${configPath}`);
  }
  if (!providerId && options.autoProvider) {
    providerId = await resolveAutoLocalImageProviderId(options);
  }
  if (!providerId) {
    throw new Error("复用 Lime 本地配图网关时必须传 --provider-id / --auto-provider，或配置 media_defaults.image.preferred_provider_id");
  }
  if (!Number.isFinite(port) || port <= 0) {
    throw new Error(`Lime config server.port 无效: ${String(server.port)}`);
  }

  return {
    apiKey,
    baseUrl: `http://${host}:${port}/v1`,
    providerId,
  };
}

async function runDryRun(params: {
  transport: ImageGenerationTransport;
  baseUrl: string;
  providerId: string;
  imageModel: string;
}): Promise<Record<string, unknown>> {
  if (params.transport === "responses") {
    return {
      ok: true,
      dryRun: true,
      transport: params.transport,
      gateway: {
        baseUrlHash: sha256Short(params.baseUrl),
        responsesPath: new URL(buildResponsesUrl(params.baseUrl)).pathname,
      },
      checks: {
        credentialsProvided: true,
        networkProbeSkipped: true,
      },
    };
  }

  const healthUrl = buildLocalHealthUrl(params.baseUrl);
  const modelsUrl = buildLocalModelsUrl(params.baseUrl);
  let healthStatus = 0;
  let modelIds: string[] = [];
  try {
    const healthResponse = await fetch(healthUrl);
    healthStatus = healthResponse.status;
  } catch {
    healthStatus = 0;
  }

  try {
    const modelsResponse = await fetch(modelsUrl);
    const modelsBody = await modelsResponse.json() as { data?: Array<{ id?: string }> };
    modelIds = Array.isArray(modelsBody.data)
      ? modelsBody.data.map((item) => item.id).filter((item): item is string => Boolean(item))
      : [];
  } catch {
    modelIds = [];
  }

  const imageModelListed = modelIds.includes(params.imageModel);
  return {
    ok: healthStatus >= 200 && healthStatus < 300 && Boolean(params.providerId),
    dryRun: true,
    transport: params.transport,
    providerId: params.providerId,
    gateway: {
      baseUrlHash: sha256Short(params.baseUrl),
      healthPath: new URL(healthUrl).pathname,
      imagesPath: new URL(buildImagesGenerationUrl(params.baseUrl)).pathname,
      modelsPath: new URL(modelsUrl).pathname,
    },
    checks: {
      localServerHealthy: healthStatus >= 200 && healthStatus < 300,
      providerIdProvided: Boolean(params.providerId),
      imageModelListed,
      imageGenerationSkipped: true,
    },
    observed: {
      healthStatus,
      modelCount: modelIds.length,
      matchingModels: modelIds.filter((item) => item.includes("gpt-image") || item.includes("gpt-images")),
    },
  };
}

async function startSelfTestServer(): Promise<{
  baseUrl: string;
  close: () => Promise<void>;
}> {
  const server = createServer(async (request: IncomingMessage, response: ServerResponse) => {
    if (request.method === "GET" && request.url === "/health") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ status: "healthy", version: "self-test" }));
      return;
    }

    if (request.method === "GET" && request.url === "/v1/models") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(
        JSON.stringify({
          object: "list",
          data: [{ id: DEFAULT_IMAGE_MODEL, object: "model", owned_by: "self-test" }],
        }),
      );
      return;
    }

    if (request.method !== "POST") {
      response.writeHead(404).end("not found");
      return;
    }

    if (request.url === "/v1/images/generations") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(
        JSON.stringify({
          created: 1_717_200_000,
          data: [
            {
              b64_json: TINY_PNG_BASE64,
              revised_prompt: "本地配图网关自测青柠图层",
            },
          ],
        }),
      );
      return;
    }

    if (request.url !== "/v1/responses") {
      response.writeHead(404).end("not found");
      return;
    }

    response.writeHead(200, { "content-type": "text/event-stream" });
    response.write(
      [
        "event: response.output_item.done",
        `data: ${JSON.stringify({
          item: {
            id: "ig_self_test",
            type: "image_generation_call",
            result: TINY_PNG_BASE64,
            revised_prompt: "自测青柠图层",
          },
        })}`,
        "",
        "event: response.completed",
        'data: {"response":{"id":"resp_self_test"}}',
        "",
      ].join("\n"),
    );
    response.end();
  });

  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("无法启动 self-test server");
  }

  return {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
      }),
  };
}

async function writeJson(filePath: string, value: unknown): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

async function writeImageIfRequested(filePath: string | undefined, imageBase64: string) {
  if (!filePath) {
    return undefined;
  }
  const resolved = path.resolve(filePath);
  await mkdir(path.dirname(resolved), { recursive: true });
  await writeFile(resolved, Buffer.from(imageBase64, "base64"));
  return resolved;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  let selfTestServer: Awaited<ReturnType<typeof startSelfTestServer>> | null = null;
  let apiKey = "";
  let baseUrl = "";
  let providerId = options.providerId?.trim() || "openai";
  const transport: ImageGenerationTransport = options.localLimeImageGateway
    ? "lime_images_gateway"
    : "responses";
  const mode = options.selfTest ? "self_test" : "live";

  try {
    if (!options.selfTest && !options.dryRun && !options.allowExternalImageGeneration) {
      throw new Error(
        `${usage()}\n\n拒绝执行真实图片生成：请显式传 --allow-external-image-generation，表示已确认会调用真实图片接口并可能消耗额度。`,
      );
    }

    if (options.selfTest) {
      selfTestServer = await startSelfTestServer();
      apiKey = "self-test-key";
      baseUrl = selfTestServer.baseUrl;
      if (options.localLimeImageGateway) {
        providerId = options.providerId?.trim() || "self-test-provider";
      }
    } else if (options.localLimeImageGateway) {
      const localConfig = await readLocalLimeGatewayConfig(options);
      apiKey = localConfig.apiKey;
      baseUrl = localConfig.baseUrl;
      providerId = localConfig.providerId;
    } else {
      const env = readEnvOrOption(options);
      apiKey = env.apiKey;
      baseUrl = env.baseUrl;
    }

    if (options.dryRun) {
      const dryRun = await runDryRun({
        transport,
        baseUrl,
        providerId,
        imageModel: options.imageModel,
      });
      if (options.output) {
        await writeJson(path.resolve(options.output), dryRun);
      }
      console.log(JSON.stringify(dryRun, null, 2));
      return;
    }

    const projectRootPath = path.join(os.tmpdir(), "lime-layered-design-gpt-image-live");
    const verificationLabel = options.verificationLabel?.trim() || "gpt-image-live";
    const document = createLayeredDesignSeedDocument({
      prompt: "@配图 青柠汽水产品主视觉，透明主体图层",
      id: `${verificationLabel}-design`,
      title: `${verificationLabel} 图层生成验收`,
      createdAt: new Date().toISOString(),
    });
    const generationRequest =
      createLayeredDesignAssetGenerationPlan(document).find((request) => request.hasAlpha) ??
      createLayeredDesignAssetGenerationPlan(document)[0];
    if (!generationRequest) {
      throw new Error("没有可验收的图层生成请求");
    }

    const taskRequest = createLayeredDesignImageTaskRequest(document, generationRequest, {
      projectRootPath,
      providerId,
      model: options.imageModel,
      outerModel: options.outerModel,
      usage: "layered_design_gpt_image_live_verification",
    });

    const generated = await generateImage({
      transport,
      apiKey,
      baseUrl,
      providerId,
      outerModel: options.outerModel,
      imageModel: options.imageModel,
      imageSize: options.imageSize,
      prompt: taskRequest.prompt,
    });
    const taskOutput = createTaskOutput({
      projectRootPath,
      taskId: `${mode}-${verificationLabel}-task`,
      taskRequest,
      result: generated,
    });
    const appliedDocument = applyLayeredDesignImageTaskOutput(
      document,
      generationRequest,
      taskOutput,
    );
    if (!appliedDocument) {
      throw new Error("图片任务结果未能写回 LayeredDesignDocument");
    }

    const targetLayer = appliedDocument.layers.find(
      (layer) => layer.id === generationRequest.layerId,
    );
    const appliedAsset = appliedDocument.assets.find(
      (asset) => asset.id === targetLayer?.assetId,
    );
    const imageBuffer = Buffer.from(generated.imageBase64, "base64");
    const imageOutputPath = await writeImageIfRequested(options.imageOutput, generated.imageBase64);
    const expectedExecutorMode =
      options.expectedExecutorMode ?? "responses_image_generation";
    const evidence = {
      schema:
        options.evidenceSchema?.trim() || "layered-design-gpt-image-live-evidence@1",
      mode,
      generatedAt: new Date().toISOString(),
      verificationLabel,
      gateway: {
        baseUrlHash: sha256Short(baseUrl),
        transport,
        responsesPath: transport === "responses" ? new URL(buildResponsesUrl(baseUrl)).pathname : undefined,
        imagesPath:
          transport === "lime_images_gateway"
            ? new URL(buildImagesGenerationUrl(baseUrl)).pathname
            : undefined,
      },
      models: {
        imageModel: options.imageModel,
        outerModel: options.outerModel,
        executorMode: taskRequest.executorMode,
      },
      task: {
        entrySource: taskRequest.entrySource,
        providerId: taskRequest.providerId,
        model: taskRequest.model,
        executorMode: taskRequest.executorMode,
        outerModel: taskRequest.outerModel,
        routingSlot: taskRequest.routingSlot,
        modalityContractKey: taskRequest.modalityContractKey,
        targetOutputId: taskRequest.targetOutputId,
        targetOutputRefId: taskRequest.targetOutputRefId,
      },
      result: {
        transport: generated.transport,
        imageCount: 1,
        imageItemId: generated.imageItemId,
        eventCount: generated.eventCount,
        outputItemCount: generated.outputItemCount,
        imageBytes: imageBuffer.length,
        imageOutputPath,
        remoteImageUrl: generated.imageUrl,
      },
      document: {
        documentId: appliedDocument.id,
        targetLayerId: generationRequest.layerId,
        targetLayerAssetId: targetLayer?.assetId,
        generatedAssetId: appliedAsset?.id,
        generatedAssetSource: appliedAsset?.params?.generatedImageSource,
        generatedAssetExecutorMode: appliedAsset?.params?.executorMode,
      },
      checks: {
        noLegacyPosterRoute:
          !JSON.stringify(taskRequest).includes("poster_generate") &&
          !JSON.stringify(taskRequest).includes("canvas:poster"),
        executorModeMatchesExpected:
          taskRequest.executorMode === expectedExecutorMode,
        imageDataUrl: taskOutput.record.result?.images?.[0]?.url?.startsWith(
          "data:image/png;base64,",
        ),
        pngMagic: isPngBuffer(imageBuffer),
        generatedAssetApplied: Boolean(appliedAsset?.src),
        targetLayerUpdated: targetLayer?.assetId === appliedAsset?.id,
      },
    };

    const ok = Object.values(evidence.checks).every(Boolean);
    if (!ok) {
      throw new Error(`验收检查失败: ${JSON.stringify(evidence.checks)}`);
    }

    if (options.output) {
      await writeJson(path.resolve(options.output), evidence);
    }
    console.log(JSON.stringify({ ok: true, evidence }, null, 2));
  } finally {
    await selfTestServer?.close();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
