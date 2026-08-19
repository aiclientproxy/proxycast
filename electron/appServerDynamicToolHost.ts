import {
  ERROR_CODES,
  isJsonRpcNotification,
  isJsonRpcRequest,
  METHOD_ITEM_TOOL_CALL,
  METHOD_TURN_COMPLETED,
  type AppServerConnection,
  type DynamicToolCallParams,
  type DynamicToolCallResponse,
  type JsonRpcMessage,
  type JsonRpcRequest,
} from "@limecloud/app-server-client";
import { app } from "./electronRuntime";
import type {
  BrowserToolCall,
  BrowserToolResult,
  ElectronBrowserTabHost,
} from "./browserTabHost";
import process from "node:process";

const THREAD_START_METHOD = "thread/start";
const THREAD_RESUME_METHOD = "thread/resume";
const DESKTOP_NAMESPACE = "desktop";
const APP_INFO_TOOL = "appInfo";
const BROWSER_NAMESPACE = "browser";
const TERMINAL_TURN_STATUSES = new Set([
  "completed",
  "failed",
  "interrupted",
  "canceled",
  "cancelled",
]);

type DynamicToolConnection = Pick<
  AppServerConnection,
  "respondServerRequest" | "rejectServerRequest"
>;

export interface DynamicToolOwnerContext {
  ownerWebContentsId: number;
}

type AppInfo = {
  locale: string;
  name: string;
  platform: string;
  version: string;
};

type DynamicToolBinding = {
  namespace: string;
  runtimeName: string;
  tool: string;
};

type ThreadHostBinding = {
  ownerWebContentsId: number;
  tools: Map<string, DynamicToolBinding>;
};

const APP_INFO_BINDING = Object.freeze<DynamicToolBinding>({
  namespace: DESKTOP_NAMESPACE,
  runtimeName: `${DESKTOP_NAMESPACE}__${APP_INFO_TOOL}`,
  tool: APP_INFO_TOOL,
});

const BROWSER_TOOL_DEFINITIONS = Object.freeze([
  tool("openTabs", "List Browser tabs available to this conversation.", {}),
  tool(
    "newTab",
    "Open a new Agent-controlled tab in the visible Browser workspace.",
    { url: stringProperty("Initial http or https URL.") },
    ["url"],
  ),
  tool(
    "claimTab",
    "Claim an existing visible tab for the current turn.",
    {
      pageRevision: nonNegativeIntegerProperty(
        "Page revision returned by openTabs.",
      ),
      tabId: stringProperty("Tab id returned by openTabs."),
      title: stringProperty("Title returned by openTabs."),
      url: stringProperty("URL returned by openTabs."),
    },
    ["tabId", "title", "url", "pageRevision"],
  ),
  tool(
    "releaseTab",
    "Release Agent control while keeping a user tab visible.",
    { tabId: stringProperty("Tab id to release.") },
    ["tabId"],
  ),
  tool(
    "goto",
    "Navigate the claimed tab to an http or https URL.",
    {
      tabId: stringProperty("Claimed tab id; defaults to the selected tab."),
      url: stringProperty("Destination URL."),
    },
    ["url"],
  ),
  tool(
    "observe",
    "Read the claimed tab URL, title, navigation state, and accessibility tree.",
    { tabId: stringProperty("Claimed tab id; defaults to the selected tab.") },
  ),
  tool("screenshot", "Capture the claimed tab as a PNG image.", {
    tabId: stringProperty("Claimed tab id; defaults to the selected tab."),
  }),
  tool(
    "click",
    "Click an actionable node from the latest observe result. Sensitive targets hand control to the user.",
    {
      backendNodeId: integerProperty("backendNodeId from observe."),
      snapshotId: stringProperty("snapshotId from observe."),
      tabId: stringProperty("Claimed tab id; defaults to the selected tab."),
    },
    ["backendNodeId", "snapshotId"],
  ),
  tool(
    "fill",
    "Replace text in an actionable field. Password or secret fields hand control to the user.",
    {
      backendNodeId: integerProperty("backendNodeId from observe."),
      snapshotId: stringProperty("snapshotId from observe."),
      tabId: stringProperty("Claimed tab id; defaults to the selected tab."),
      text: stringProperty("Text to enter."),
    },
    ["backendNodeId", "snapshotId", "text"],
  ),
  tool(
    "press",
    "Press a non-submitting key in the claimed tab. Enter hands control to the user.",
    {
      key: stringProperty("DOM key value such as Escape or ArrowDown."),
      snapshotId: stringProperty("snapshotId from observe."),
      tabId: stringProperty("Claimed tab id; defaults to the selected tab."),
    },
    ["key", "snapshotId"],
  ),
  tool(
    "markHandoff",
    "Keep the claimed tab for a later turn and release control when this turn ends.",
    { tabId: stringProperty("Claimed tab id; defaults to the selected tab.") },
  ),
  tool(
    "markDeliverable",
    "Keep the claimed tab as a user-visible deliverable when this turn ends.",
    { tabId: stringProperty("Claimed tab id; defaults to the selected tab.") },
  ),
]);

const DESKTOP_DYNAMIC_TOOLS = Object.freeze([
  Object.freeze({
    type: "namespace",
    name: DESKTOP_NAMESPACE,
    description: "Read information exposed by the Lime desktop host.",
    tools: Object.freeze([
      tool(
        APP_INFO_TOOL,
        "Read the desktop application name, version, locale, and platform.",
        {},
      ),
    ]),
  }),
  Object.freeze({
    type: "namespace",
    name: BROWSER_NAMESPACE,
    description:
      "Operate the same Browser workspace tab that is visible to the user.",
    tools: BROWSER_TOOL_DEFINITIONS,
  }),
]);

const BROWSER_BINDINGS = new Map<string, DynamicToolBinding>(
  BROWSER_TOOL_DEFINITIONS.map((definition) => {
    const binding = {
      namespace: BROWSER_NAMESPACE,
      runtimeName: `${BROWSER_NAMESPACE}__${definition.name}`,
      tool: definition.name,
    };
    return [binding.runtimeName, binding];
  }),
);

export class AppServerDynamicToolHost {
  readonly #bindingsByThread = new Map<string, ThreadHostBinding>();
  readonly #browserHost: ElectronBrowserTabHost | null;
  readonly #consumedCalls = new Set<string>();
  readonly #readAppInfo: () => AppInfo;

  constructor(
    readAppInfo: () => AppInfo = () => ({
      locale: app.getLocale(),
      name: app.getName(),
      platform: process.platform,
      version: app.getVersion(),
    }),
    browserHost: ElectronBrowserTabHost | null = null,
  ) {
    this.#readAppInfo = readAppInfo;
    this.#browserHost = browserHost;
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

  observeClientResult(
    method: string,
    result: unknown,
    owner: DynamicToolOwnerContext | null = null,
  ): void {
    if (method !== THREAD_START_METHOD && method !== THREAD_RESUME_METHOD) {
      return;
    }
    const thread = asRecord(asRecord(result)?.thread);
    const threadId = nonEmptyString(thread?.id);
    if (!threadId) {
      return;
    }
    const tools = new Map<string, DynamicToolBinding>([
      [APP_INFO_BINDING.runtimeName, APP_INFO_BINDING],
      ...BROWSER_BINDINGS,
    ]);
    this.#bindingsByThread.set(threadId, {
      ownerWebContentsId: owner?.ownerWebContentsId ?? 0,
      tools,
    });
  }

  observeServerMessage(message: JsonRpcMessage): void {
    if (
      !this.#browserHost ||
      !isJsonRpcNotification(message) ||
      message.method !== METHOD_TURN_COMPLETED
    ) {
      return;
    }
    const params = asRecord(message.params);
    const threadId = nonEmptyString(params?.threadId);
    const turn = asRecord(params?.turn);
    const turnId = nonEmptyString(params?.turnId) ?? nonEmptyString(turn?.id);
    const status = nonEmptyString(turn?.status);
    if (threadId && turnId && status && TERMINAL_TURN_STATUSES.has(status)) {
      this.#browserHost.turnEnded(threadId, turnId);
    }
  }

  async tryHandle(
    connection: DynamicToolConnection,
    message: JsonRpcMessage,
  ): Promise<boolean> {
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
    const threadBinding = this.#bindingsByThread.get(params.value.threadId);
    const runtimeName = `${params.value.namespace}__${params.value.tool}`;
    const binding = threadBinding?.tools.get(runtimeName);
    if (
      !threadBinding ||
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
    const validationError = validateArguments(binding, params.value.arguments);
    if (validationError) {
      connection.rejectServerRequest(message.id, {
        code: ERROR_CODES.invalidParams,
        message: validationError,
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
      const response =
        binding.namespace === BROWSER_NAMESPACE
          ? await this.#executeBrowserTool(
              threadBinding.ownerWebContentsId,
              params.value,
            )
          : appInfoResponse(this.#readAppInfo());
      connection.respondServerRequest<DynamicToolCallResponse>(
        message.id,
        response,
      );
    } catch (error) {
      connection.respondServerRequest<DynamicToolCallResponse>(message.id, {
        contentItems: [
          {
            type: "inputText",
            text: `desktop host capability failed: ${errorMessage(error)}`,
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

  connectionLost(reason?: string): void {
    this.#browserHost?.connectionLost(reason);
  }

  async #executeBrowserTool(
    ownerWebContentsId: number,
    params: DynamicToolCallParams & { arguments: Record<string, unknown> },
  ): Promise<DynamicToolCallResponse> {
    if (!this.#browserHost || ownerWebContentsId <= 0) {
      throw new Error("Browser workspace is not bound to this desktop thread");
    }
    const call: BrowserToolCall = {
      arguments: params.arguments,
      callId: params.callId,
      ownerWebContentsId,
      threadId: params.threadId,
      tool: params.tool,
      turnId: params.turnId,
    };
    const result = await this.#browserHost.executeTool(call);
    return browserToolResponse(result);
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

function validateArguments(
  binding: DynamicToolBinding,
  args: Record<string, unknown>,
): string | null {
  if (binding.namespace === DESKTOP_NAMESPACE) {
    return Object.keys(args).length === 0
      ? null
      : `${binding.runtimeName} does not accept arguments`;
  }
  const definition = BROWSER_TOOL_DEFINITIONS.find(
    (candidate) => candidate.name === binding.tool,
  );
  if (!definition) {
    return `Unknown Browser tool binding: ${binding.tool}`;
  }
  const properties = asRecord(definition.inputSchema.properties) ?? {};
  for (const key of Object.keys(args)) {
    if (!(key in properties)) {
      return `${binding.runtimeName} does not accept argument: ${key}`;
    }
  }
  const required = Array.isArray(definition.inputSchema.required)
    ? definition.inputSchema.required
    : [];
  for (const key of required) {
    if (!(key in args)) {
      return `${binding.runtimeName} requires argument: ${key}`;
    }
  }
  for (const [key, value] of Object.entries(args)) {
    const schema = asRecord(properties[key]);
    if (schema?.type === "string" && nonEmptyString(value) === null) {
      return `${binding.runtimeName}.${key} must be a non-empty string`;
    }
    if (
      schema?.type === "integer" &&
      (!Number.isInteger(value) ||
        Number(value) <
          (typeof schema.minimum === "number" ? schema.minimum : 1))
    ) {
      return `${binding.runtimeName}.${key} must be an integer greater than or equal to ${typeof schema.minimum === "number" ? schema.minimum : 1}`;
    }
  }
  return null;
}

function appInfoResponse(info: AppInfo): DynamicToolCallResponse {
  return {
    contentItems: [{ type: "inputText", text: JSON.stringify(info) }],
    success: true,
  };
}

function browserToolResponse(
  result: BrowserToolResult,
): DynamicToolCallResponse {
  const contentItems: DynamicToolCallResponse["contentItems"] = [
    {
      type: "inputText",
      text: JSON.stringify({
        status: result.status,
        ...(result.state ? { tab: result.state } : {}),
        ...(result.data !== undefined ? { data: result.data } : {}),
      }),
    },
  ];
  if (result.imageBase64) {
    contentItems.push({
      type: "inputImage",
      imageUrl: `data:image/png;base64,${result.imageBase64}`,
    });
  }
  return {
    contentItems,
    success: result.status === "completed",
  };
}

function tool(
  name: string,
  description: string,
  properties: Record<string, unknown>,
  required: string[] = [],
) {
  return Object.freeze({
    type: "function" as const,
    name,
    description,
    inputSchema: Object.freeze({
      type: "object",
      properties: Object.freeze(properties),
      ...(required.length > 0 ? { required: Object.freeze(required) } : {}),
      additionalProperties: false,
    }),
  });
}

function stringProperty(description: string) {
  return Object.freeze({ type: "string", description });
}

function integerProperty(description: string) {
  return Object.freeze({ type: "integer", minimum: 1, description });
}

function nonNegativeIntegerProperty(description: string) {
  return Object.freeze({ type: "integer", minimum: 0, description });
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
