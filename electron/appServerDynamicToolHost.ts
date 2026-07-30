import {
  ERROR_CODES,
  isJsonRpcRequest,
  METHOD_ITEM_TOOL_CALL,
  type AppServerConnection,
  type DynamicToolCallParams,
  type DynamicToolCallResponse,
  type JsonRpcMessage,
  type JsonRpcRequest,
} from "@limecloud/app-server-client";
import { app } from "./electronRuntime";
import process from "node:process";

const THREAD_START_METHOD = "thread/start";
const THREAD_RESUME_METHOD = "thread/resume";
const DESKTOP_NAMESPACE = "desktop";
const APP_INFO_TOOL = "appInfo";
const APP_INFO_RUNTIME_NAME = `${DESKTOP_NAMESPACE}__${APP_INFO_TOOL}`;

type DynamicToolConnection = Pick<
  AppServerConnection,
  "respondServerRequest" | "rejectServerRequest"
>;

type AppInfo = {
  locale: string;
  name: string;
  platform: string;
  version: string;
};

type DynamicToolBinding = {
  capabilityId: string;
  deadlineMs: number;
  namespace: string;
  outputModalities: readonly ["text"];
  owner: "electron-desktop-host";
  runtimeName: string;
  schemaDigest: string;
  tool: string;
};

const APP_INFO_BINDING = Object.freeze<DynamicToolBinding>({
  capabilityId: "desktop.app-info.read",
  deadlineMs: 3_000,
  namespace: DESKTOP_NAMESPACE,
  outputModalities: ["text"],
  owner: "electron-desktop-host",
  runtimeName: APP_INFO_RUNTIME_NAME,
  schemaDigest: "desktop-app-info-input-v1",
  tool: APP_INFO_TOOL,
});

const DESKTOP_DYNAMIC_TOOLS = Object.freeze([
  Object.freeze({
    type: "namespace",
    name: DESKTOP_NAMESPACE,
    description: "Read information exposed by the Lime desktop host.",
    tools: Object.freeze([
      Object.freeze({
        type: "function",
        name: APP_INFO_TOOL,
        description:
          "Read the desktop application name, version, locale, and platform.",
        inputSchema: Object.freeze({
          type: "object",
          properties: Object.freeze({}),
          additionalProperties: false,
        }),
      }),
    ]),
  }),
]);

export class AppServerDynamicToolHost {
  readonly #bindingsByThread = new Map<string, DynamicToolBinding>();
  readonly #consumedCalls = new Set<string>();
  readonly #readAppInfo: () => AppInfo;

  constructor(
    readAppInfo: () => AppInfo = () => ({
      locale: app.getLocale(),
      name: app.getName(),
      platform: process.platform,
      version: app.getVersion(),
    }),
  ) {
    this.#readAppInfo = readAppInfo;
  }

  prepareClientRequest(message: JsonRpcRequest): JsonRpcRequest {
    if (message.method !== THREAD_START_METHOD) {
      return message;
    }
    const params = asRecord(message.params) ?? {};
    return {
      ...message,
      params: {
        ...params,
        dynamicTools: structuredClone(DESKTOP_DYNAMIC_TOOLS),
      },
    };
  }

  observeClientResult(method: string, result: unknown): void {
    if (method !== THREAD_START_METHOD && method !== THREAD_RESUME_METHOD) {
      return;
    }
    const thread = asRecord(asRecord(result)?.thread);
    const threadId = nonEmptyString(thread?.id);
    if (threadId) {
      this.#bindingsByThread.set(threadId, APP_INFO_BINDING);
    }
  }

  tryHandle(
    connection: DynamicToolConnection,
    message: JsonRpcMessage,
  ): boolean {
    if (
      !isJsonRpcRequest(message) ||
      message.method !== METHOD_ITEM_TOOL_CALL
    ) {
      return false;
    }
    const params = parseCallParams(message.params);
    if (!params.ok) {
      connection.rejectServerRequest(message.id, {
        code: ERROR_CODES.invalidParams,
        message: params.error,
      });
      return true;
    }
    const binding = this.#bindingsByThread.get(params.value.threadId);
    if (
      !binding ||
      params.value.namespace !== binding.namespace ||
      params.value.tool !== binding.tool
    ) {
      connection.rejectServerRequest(message.id, {
        code: ERROR_CODES.invalidParams,
        message:
          "item/tool/call does not match a frozen host capability binding",
      });
      return true;
    }
    if (Object.keys(params.value.arguments).length !== 0) {
      connection.rejectServerRequest(message.id, {
        code: ERROR_CODES.invalidParams,
        message: `${binding.runtimeName} does not accept arguments`,
      });
      return true;
    }
    const callKey = [
      params.value.threadId,
      params.value.turnId,
      params.value.callId,
    ].join("\u0000");
    if (this.#consumedCalls.has(callKey)) {
      connection.rejectServerRequest(message.id, {
        code: ERROR_CODES.invalidParams,
        message: "item/tool/call identity was already consumed",
      });
      return true;
    }
    this.#consumedCalls.add(callKey);
    try {
      const response: DynamicToolCallResponse = {
        contentItems: [
          {
            type: "inputText",
            text: JSON.stringify(this.#readAppInfo()),
          },
        ],
        success: true,
      };
      connection.respondServerRequest<DynamicToolCallResponse>(
        message.id,
        response,
      );
    } catch (error) {
      connection.respondServerRequest<DynamicToolCallResponse>(message.id, {
        contentItems: [
          {
            type: "inputText",
            text: `desktop app information is unavailable: ${errorMessage(error)}`,
          },
        ],
        success: false,
      });
    }
    return true;
  }

  reset(): void {
    this.#bindingsByThread.clear();
    this.#consumedCalls.clear();
  }
}

type ParsedCallParams =
  | {
      ok: true;
      value: DynamicToolCallParams & { arguments: Record<string, unknown> };
    }
  | { ok: false; error: string };

function parseCallParams(value: unknown): ParsedCallParams {
  const params = asRecord(value);
  const threadId = nonEmptyString(params?.threadId);
  const turnId = nonEmptyString(params?.turnId);
  const callId = nonEmptyString(params?.callId);
  const tool = nonEmptyString(params?.tool);
  const namespace = nonEmptyString(params?.namespace);
  const argumentsValue = asRecord(params?.arguments);
  if (
    !threadId ||
    !turnId ||
    !callId ||
    !tool ||
    !namespace ||
    !argumentsValue
  ) {
    return {
      ok: false,
      error:
        "item/tool/call requires threadId, turnId, callId, namespace, tool, and object arguments",
    };
  }
  return {
    ok: true,
    value: {
      threadId,
      turnId,
      callId,
      namespace,
      tool,
      arguments: argumentsValue,
    },
  };
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
