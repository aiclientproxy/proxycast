import {
  AppServerRequestError,
  AppServerSidecarLifecycle,
  cancelRequest,
  decodeMessage,
  defaultReleaseManifestPath,
  encodeMessage,
  isJsonRpcNotification,
  isJsonRpcResponse,
  isJsonRpcErrorResponse,
  METHOD_AGENT_MESSAGE_DELTA,
  METHOD_INITIALIZE,
  METHOD_INITIALIZED,
  METHOD_ITEM_COMPLETED,
  METHOD_ITEM_STARTED,
  METHOD_MODEL_LIST_UPDATED,
  METHOD_THREAD_STARTED,
  METHOD_THREAD_TOKEN_USAGE_UPDATED,
  METHOD_TURN_COMPLETED,
  METHOD_TURN_STARTED,
  METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED,
  readReleaseManifest,
  resolveSidecarFromReleaseManifest,
  stdioSidecar,
  type AppServerRequestOptions,
  type AppServerRequestResult,
  type ConnectedAppServerSidecar,
  type InitializeResponse,
  type InitializeParams,
  type JsonRpcRequest,
  type JsonRpcMessage,
  type RequestId,
  type SidecarLaunchConfig,
} from "@limecloud/app-server-client";
import { app, session } from "./electronRuntime";
import { resolveCurrentDesktopStorageRoots } from "./appDataPaths";
import { readFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { tryHandleCurrentTimeRead } from "./appServerCurrentTimeHost";
import {
  AppServerDynamicToolHost,
  type DynamicToolOwnerContext,
} from "./appServerDynamicToolHost";
import type { ElectronBrowserTabHost } from "./browserTabHost";

const DEFAULT_APP_SERVER_REQUEST_TIMEOUT_MS = 30_000;
const APP_SERVER_BACKEND_TIMEOUT_GRACE_MS = 30_000;
const APP_SERVER_TURN_START_METHOD = "turn/start";
const APP_SERVER_CONVERSATION_IMPORT_THREAD_COMMIT_METHOD =
  "conversationImport/thread/commit";
const APP_SERVER_CONVERSATION_IMPORT_JOB_READ_METHOD =
  "conversationImport/job/read";
const APP_SERVER_CONVERSATION_IMPORT_THREAD_COMMIT_TIMEOUT_MS = 180_000;
const APP_SERVER_CONVERSATION_IMPORT_SCAN_TIMEOUT_MS = 120_000;
const APP_SERVER_CONVERSATION_IMPORT_PREVIEW_TIMEOUT_MS = 120_000;
const APP_SERVER_REQUEST_TIMEOUT_OVERRIDE_CEILING_MS = 600_000;
const APP_SERVER_PROXY_REQUEST_ID_PREFIX = "electron-host";
const APP_SERVER_CANCEL_REQUEST_METHOD = "$/cancelRequest";
const APP_SERVER_CONFIG_FILE_NAME = "config.yaml";
const APP_SERVER_RECENT_NOTIFICATION_LIMIT = 500;
const APP_SERVER_SERVER_REQUEST_TOKEN_LIMIT = 500;
const APP_SERVER_SERVER_REQUEST_TOKEN_PREFIX = "electron-action:";
const APP_SERVER_DRAIN_FIRST_MESSAGE_WAIT_MS = 25;
const APP_SERVER_DRAIN_BUFFERED_MESSAGE_WAIT_MS = 0;
const APP_SERVER_RESTART_MAX_ATTEMPTS = 3;
const APP_SERVER_PROXY_PROBE_URL = "https://llm.limeai.run/v1/models";
const APP_SERVER_PROXY_ENV_KEYS = [
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "ALL_PROXY",
  "http_proxy",
  "https_proxy",
  "all_proxy",
] as const;
const APP_SERVER_NO_PROXY_ENV_KEYS = ["NO_PROXY", "no_proxy"] as const;
const APP_SERVER_LOOPBACK_NO_PROXY_HOSTS = ["127.0.0.1", "localhost", "::1"];

type ElectronAppServerLaunchConfig = {
  config: SidecarLaunchConfig;
  verifySha256?: boolean;
};

type HandleJsonLinesRequest = {
  lines: string[];
  timeoutMs?: number;
};

type DrainEventsRequest = {
  includeRecent?: boolean;
  limit?: number;
};

export type AppServerSidecarTermination = {
  pid: number | null;
  requested: boolean;
  signal: "SIGTERM";
};

type AppServerRestartWaiter = {
  lifecycle: AppServerSidecarLifecycle;
  promise: Promise<ConnectedAppServerSidecar>;
  reject: (error: Error) => void;
  resolve: (connected: ConnectedAppServerSidecar) => void;
};

export class ElectronAppServerHost {
  #lifecycle: AppServerSidecarLifecycle | null = null;
  #connected: ConnectedAppServerSidecar | null = null;
  #connectPromise: Promise<ConnectedAppServerSidecar> | null = null;
  #nextProxyRequestId = 1;
  #activeProxyRequestIds = new Map<RequestId, RequestId>();
  #consumedServerRequestTokens = new Set<string>();
  #recentNotifications: JsonRpcMessage[] = [];
  #restartWaiter: AppServerRestartWaiter | null = null;
  #serverRequestRawIdsByToken = new Map<string, RequestId>();
  #serverRequestTokensByRawId = new Map<string, string>();
  #stopping = false;
  readonly #dynamicToolHost: AppServerDynamicToolHost;

  constructor(browserHost: ElectronBrowserTabHost | null = null) {
    this.#dynamicToolHost = new AppServerDynamicToolHost(
      undefined,
      browserHost,
    );
  }

  async warmup(): Promise<InitializeResponse> {
    const connected = await this.#connect();
    return connected.initializeResponse;
  }

  async request<T>(method: string, params: unknown = {}): Promise<T> {
    const connected = await this.#connect();
    const request = this.#dynamicToolHost.prepareClientRequest(
      connected.client.request(method, params ?? {}),
    );
    const response = await this.#requestAppServer<T>(
      connected,
      request,
      method,
      {
        timeoutMs: resolveAppServerRequestTimeoutMs(method),
      },
    );
    this.#dynamicToolHost.observeClientResult(method, response.result);
    return response.result;
  }

  async handleJsonLines(
    request: HandleJsonLinesRequest,
    owner: DynamicToolOwnerContext | null = null,
  ): Promise<{ lines: string[] }> {
    const connected = await this.#connect();
    const messages = request.lines.map(decodeMessage);
    const responses: JsonRpcMessage[] = [];

    for (const message of messages) {
      if (isInitializedNotification(message)) {
        continue;
      }
      if (isCancelRequestNotification(message)) {
        this.#forwardCancelRequest(connected, message);
        continue;
      }
      if (
        isJsonRpcRequestLike(message) &&
        message.method === METHOD_INITIALIZE
      ) {
        responses.push(
          initializeResponseMessage(message, connected.initializeResponse),
        );
        continue;
      }
      if (isJsonRpcRequestLike(message)) {
        const preparedMessage =
          this.#dynamicToolHost.prepareClientRequest(message);
        const proxiedMessage = this.#proxyRequestMessage(preparedMessage);
        const timeoutMs = resolveAppServerRequestTimeoutMs(
          proxiedMessage.message.method,
          request.timeoutMs,
        );
        try {
          const result = await this.#withActiveProxyRequest(
            proxiedMessage.originalId,
            proxiedMessage.message.id,
            () =>
              this.#requestAppServer<unknown>(
                connected,
                proxiedMessage.message,
                proxiedMessage.message.method,
                { timeoutMs },
              ),
          );
          this.#dynamicToolHost.observeClientResult(
            proxiedMessage.message.method,
            result.result,
            owner,
          );
          responses.push(
            ...result.messages.map((response) =>
              restoreProxyResponseId(response, proxiedMessage.originalId),
            ),
          );
        } catch (error) {
          const errorMessages = restoreAppServerRequestError(
            error,
            proxiedMessage.originalId,
          );
          if (!errorMessages) {
            throw error;
          }
          responses.push(...errorMessages);
        }
        continue;
      }
      (await this.#connect()).connection.transport.send(
        this.#restoreServerRequestResponseId(message),
      );
    }

    this.#rememberRecentNotifications(responses);
    return {
      lines: responses.map(encodeMessage),
    };
  }

  async drainEvents(
    request: DrainEventsRequest = {},
  ): Promise<{ lines: string[] }> {
    const connected = await this.#connect();
    const limit = normalizeDrainEventsLimit(
      request.limit,
      request.includeRecent === true
        ? APP_SERVER_RECENT_NOTIFICATION_LIMIT
        : 100,
    );
    const drained: JsonRpcMessage[] = [];

    while (drained.length < limit) {
      let message: JsonRpcMessage;
      try {
        message = await connected.connection.nextServerMessage(
          drained.length === 0
            ? APP_SERVER_DRAIN_FIRST_MESSAGE_WAIT_MS
            : APP_SERVER_DRAIN_BUFFERED_MESSAGE_WAIT_MS,
        );
      } catch {
        break;
      }
      if (tryHandleCurrentTimeRead(connected.connection, message)) {
        continue;
      }
      this.#dynamicToolHost.observeServerMessage(message);
      if (
        await this.#dynamicToolHost.tryHandle(connected.connection, message)
      ) {
        continue;
      }
      const projected = this.#projectServerMessageForRenderer(message);
      if (projected) {
        drained.push(projected);
      }
    }

    const rendererMessages = drained;
    this.#rememberRecentNotifications(rendererMessages);
    const messages =
      request.includeRecent === true
        ? uniqueJsonRpcMessages([
            ...this.#recentNotifications,
            ...rendererMessages,
          ]).slice(-limit)
        : rendererMessages;

    return {
      lines: messages.map(encodeMessage),
    };
  }

  async stop(): Promise<void> {
    this.#stopping = true;
    this.#dynamicToolHost.connectionLost("app-server-stopped");
    this.#rejectRestartWaiter(this.#lifecycle, appServerHostStoppingError());
    await this.#lifecycle?.stop();
    this.#lifecycle = null;
    this.#connected = null;
    this.#connectPromise = null;
    this.#consumedServerRequestTokens.clear();
    this.#serverRequestRawIdsByToken.clear();
    this.#serverRequestTokensByRawId.clear();
    this.#dynamicToolHost.reset();
  }

  terminateSidecarForE2e(): AppServerSidecarTermination {
    if (process.env.LIME_ELECTRON_E2E !== "1") {
      throw new Error(
        "App Server sidecar termination is only available in E2E mode",
      );
    }
    const child = this.#lifecycle?.connected?.sidecar.child;
    const pid = child?.pid ?? null;
    const running =
      child && child.exitCode === null && child.signalCode === null;
    return {
      pid,
      requested: running ? child.kill("SIGTERM") : false,
      signal: "SIGTERM",
    };
  }

  #projectServerMessageForRenderer(
    message: JsonRpcMessage,
  ): JsonRpcMessage | null {
    if (isJsonRpcRequestLike(message)) {
      const token = this.#serverRequestToken(message.id);
      return { ...message, id: token };
    }
    if (!isJsonRpcNotification(message)) {
      return message;
    }
    if (message.method !== "serverRequest/resolved") {
      return message;
    }
    const params = asRecord(message.params);
    const rawRequestId = params?.requestId;
    if (typeof rawRequestId !== "string" && typeof rawRequestId !== "number") {
      return message;
    }
    const token = this.#serverRequestTokensByRawId.get(
      stableRequestIdKey(rawRequestId),
    );
    return token
      ? {
          ...message,
          params: { ...params, requestId: token },
        }
      : null;
  }

  #restoreServerRequestResponseId(message: JsonRpcMessage): JsonRpcMessage {
    if (!isJsonRpcResponse(message) && !isJsonRpcErrorResponse(message)) {
      return message;
    }
    if (
      typeof message.id !== "string" ||
      !message.id.startsWith(APP_SERVER_SERVER_REQUEST_TOKEN_PREFIX)
    ) {
      return message;
    }
    if (this.#consumedServerRequestTokens.has(message.id)) {
      throw new Error(
        "App Server server-request action token was already used",
      );
    }
    const rawId = this.#serverRequestRawIdsByToken.get(message.id);
    if (rawId === undefined) {
      throw new Error("Unknown App Server server-request action token");
    }
    this.#consumedServerRequestTokens.add(message.id);
    return { ...message, id: rawId };
  }

  #serverRequestToken(rawId: RequestId): string {
    const rawKey = stableRequestIdKey(rawId);
    const existing = this.#serverRequestTokensByRawId.get(rawKey);
    if (existing) {
      return existing;
    }
    const token = `${APP_SERVER_SERVER_REQUEST_TOKEN_PREFIX}${randomUUID()}`;
    this.#serverRequestTokensByRawId.set(rawKey, token);
    this.#serverRequestRawIdsByToken.set(token, rawId);
    while (
      this.#serverRequestRawIdsByToken.size >
      APP_SERVER_SERVER_REQUEST_TOKEN_LIMIT
    ) {
      const oldestToken = this.#serverRequestRawIdsByToken.keys().next().value;
      if (typeof oldestToken !== "string") {
        break;
      }
      const oldestRawId = this.#serverRequestRawIdsByToken.get(oldestToken);
      this.#serverRequestRawIdsByToken.delete(oldestToken);
      this.#consumedServerRequestTokens.delete(oldestToken);
      if (oldestRawId !== undefined) {
        this.#serverRequestTokensByRawId.delete(
          stableRequestIdKey(oldestRawId),
        );
      }
    }
    return token;
  }

  async #connect(): Promise<ConnectedAppServerSidecar> {
    if (this.#stopping) {
      throw appServerHostStoppingError();
    }
    if (this.#connected) {
      const lifecycleConnected = this.#lifecycle?.connected;
      if (lifecycleConnected && lifecycleConnected !== this.#connected) {
        this.#connected = lifecycleConnected;
        return lifecycleConnected;
      }
      if (lifecycleConnected) {
        return this.#connected;
      }
      this.#connected = null;
    }
    if (this.#restartWaiter) {
      this.#connected = await this.#restartWaiter.promise;
      return this.#connected;
    }
    if (!this.#connectPromise) {
      this.#connectPromise = this.#start();
    }
    const connectPromise = this.#connectPromise;
    try {
      this.#connected = await connectPromise;
      return this.#connected;
    } finally {
      if (this.#connectPromise === connectPromise) {
        this.#connectPromise = null;
      }
    }
  }

  #rememberRecentNotifications(messages: JsonRpcMessage[]): void {
    const notifications = messages.filter(isJsonRpcNotification);
    if (notifications.length === 0) {
      return;
    }
    this.#recentNotifications = [
      ...this.#recentNotifications,
      ...notifications,
    ].slice(-APP_SERVER_RECENT_NOTIFICATION_LIMIT);
  }

  async #start(): Promise<ConnectedAppServerSidecar> {
    const launchConfig = await resolveLaunchConfig();
    const sidecarEnv = await resolveAppServerSidecarEnv(
      launchConfig.config.binaryPath,
      launchConfig.config.dataDir ?? resolveAppServerDataDir(),
    );
    const initializeParams: InitializeParams = {
      clientInfo: {
        name: "lime_desktop_electron",
        title: "Lime Desktop Electron",
        version: app.getVersion(),
      },
      capabilities: {
        eventMethods: [
          "agentSession/event",
          METHOD_THREAD_STARTED,
          METHOD_TURN_STARTED,
          METHOD_TURN_COMPLETED,
          METHOD_ITEM_STARTED,
          METHOD_ITEM_COMPLETED,
          METHOD_AGENT_MESSAGE_DELTA,
          METHOD_MODEL_LIST_UPDATED,
          METHOD_THREAD_TOKEN_USAGE_UPDATED,
          METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CHANGED,
        ],
        experimentalApi: true,
      },
    };

    let lifecycle: AppServerSidecarLifecycle;
    lifecycle = new AppServerSidecarLifecycle(
      launchConfig.config,
      initializeParams,
      {
        verifySha256: launchConfig.verifySha256,
        ...(sidecarEnv ? { env: sidecarEnv } : {}),
        restartPolicy: {
          maxAttempts: APP_SERVER_RESTART_MAX_ATTEMPTS,
          initialDelayMs: 500,
          maxDelayMs: 5_000,
        },
        onExit: (event) => {
          if (this.#lifecycle === lifecycle) {
            this.#dynamicToolHost.connectionLost("app-server-disconnected");
            this.#connected = null;
            this.#waitForRestart(lifecycle);
          }
          console.warn("[electron-host] app-server exited", event);
        },
        onRestarted: (connected) => {
          if (this.#lifecycle === lifecycle) {
            this.#connected = connected;
            this.#installServerRequestHandler(connected);
            this.#resolveRestartWaiter(lifecycle, connected);
          }
        },
        onRestartFailed: (event) => {
          if (
            this.#lifecycle === lifecycle &&
            this.#restartWaiter?.lifecycle === lifecycle &&
            event.attempt >= APP_SERVER_RESTART_MAX_ATTEMPTS
          ) {
            this.#dynamicToolHost.connectionLost("app-server-restart-failed");
            this.#connected = null;
            this.#lifecycle = null;
            this.#rejectRestartWaiter(
              lifecycle,
              new Error("App Server sidecar restart attempts exhausted"),
            );
          }
          console.warn("[electron-host] app-server restart failed", event);
        },
      },
    );

    this.#lifecycle = lifecycle;
    const connected = await lifecycle.start();
    this.#installServerRequestHandler(connected);
    return connected;
  }

  #waitForRestart(
    lifecycle: AppServerSidecarLifecycle,
  ): Promise<ConnectedAppServerSidecar> {
    if (this.#restartWaiter?.lifecycle === lifecycle) {
      return this.#restartWaiter.promise;
    }
    let resolve!: (connected: ConnectedAppServerSidecar) => void;
    let reject!: (error: Error) => void;
    const promise = new Promise<ConnectedAppServerSidecar>(
      (resolvePromise, rejectPromise) => {
        resolve = resolvePromise;
        reject = rejectPromise;
      },
    );
    void promise.catch(() => undefined);
    this.#restartWaiter = { lifecycle, promise, reject, resolve };
    return promise;
  }

  #resolveRestartWaiter(
    lifecycle: AppServerSidecarLifecycle,
    connected: ConnectedAppServerSidecar,
  ): void {
    const waiter = this.#restartWaiter;
    if (!waiter || waiter.lifecycle !== lifecycle) {
      return;
    }
    this.#restartWaiter = null;
    waiter.resolve(connected);
  }

  #rejectRestartWaiter(
    lifecycle: AppServerSidecarLifecycle | null,
    error: Error,
  ): void {
    const waiter = this.#restartWaiter;
    if (!waiter || waiter.lifecycle !== lifecycle) {
      return;
    }
    this.#restartWaiter = null;
    waiter.reject(error);
  }

  #installServerRequestHandler(connected: ConnectedAppServerSidecar): void {
    if (typeof connected.connection.setServerRequestHandler !== "function") {
      return;
    }
    connected.connection.setServerRequestHandler(async (message) => {
      if (tryHandleCurrentTimeRead(connected.connection, message)) {
        return true;
      }
      this.#dynamicToolHost.observeServerMessage(message);
      return await this.#dynamicToolHost.tryHandle(
        connected.connection,
        message,
      );
    });
  }

  #proxyRequestMessage(message: JsonRpcRequest): {
    message: JsonRpcRequest;
    originalId: RequestId;
  } {
    const originalId = message.id;
    const id = `${APP_SERVER_PROXY_REQUEST_ID_PREFIX}:${this.#nextProxyRequestId}`;
    this.#nextProxyRequestId += 1;
    return {
      message: {
        ...message,
        id,
      },
      originalId,
    };
  }

  async #withActiveProxyRequest<T>(
    originalId: RequestId,
    proxiedId: RequestId,
    run: () => Promise<T>,
  ): Promise<T> {
    this.#activeProxyRequestIds.set(originalId, proxiedId);
    try {
      return await run();
    } finally {
      this.#activeProxyRequestIds.delete(originalId);
    }
  }

  #forwardCancelRequest(
    connected: ConnectedAppServerSidecar,
    message: JsonRpcMessage,
  ): void {
    const originalId = readCancelRequestId(message);
    if (originalId === null) {
      return;
    }
    const proxiedId = this.#activeProxyRequestIds.get(originalId);
    if (proxiedId === undefined) {
      return;
    }
    connected.connection.transport.send(cancelRequest(proxiedId));
  }

  async #requestAppServer<T>(
    connected: ConnectedAppServerSidecar,
    request: JsonRpcRequest,
    method: string,
    options: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<T>> {
    try {
      return await connected.connection.request<T>(request, method, options);
    } catch (error) {
      if (!isStaleSidecarConnectionError(error)) {
        throw error;
      }
      if (this.#stopping) {
        throw appServerHostStoppingError();
      }

      console.warn(
        "[electron-host] app-server stale connection detected; restarting sidecar",
        error,
      );
      await this.#discardStaleSidecar();
      const freshConnected = await this.#connect();
      return await freshConnected.connection.request<T>(
        request,
        method,
        options,
      );
    }
  }

  async #discardStaleSidecar(): Promise<void> {
    const lifecycle = this.#lifecycle;
    this.#rejectRestartWaiter(
      lifecycle,
      new Error("App Server stale sidecar lifecycle was discarded"),
    );
    this.#lifecycle = null;
    this.#connected = null;
    this.#connectPromise = null;
    this.#dynamicToolHost.connectionLost("app-server-stale-connection");
    try {
      await lifecycle?.stop();
    } catch (error) {
      console.warn(
        "[electron-host] app-server stale sidecar cleanup failed",
        error,
      );
    }
  }
}

function isStaleSidecarConnectionError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.message.includes("app-server sidecar stdin is closed") ||
      error.message.includes("app-server sidecar is closed") ||
      error.message.includes("app-server exited before next message"))
  );
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function stableRequestIdKey(id: RequestId): string {
  return JSON.stringify([typeof id, String(id)]);
}

function appServerHostStoppingError(): Error {
  return new Error("app-server host is stopping");
}

function isCancelRequestNotification(
  message: JsonRpcMessage,
): message is Extract<JsonRpcMessage, { method: string }> {
  return (
    isJsonRpcNotification(message) &&
    message.method === APP_SERVER_CANCEL_REQUEST_METHOD
  );
}

function readCancelRequestId(message: JsonRpcMessage): RequestId | null {
  if (!isCancelRequestNotification(message)) {
    return null;
  }
  const params = message.params;
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    return null;
  }
  const id = (params as { id?: unknown }).id;
  return typeof id === "string" || typeof id === "number" ? id : null;
}

function normalizeDrainEventsLimit(value: unknown, maxLimit: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return 20;
  }
  return Math.max(1, Math.min(maxLimit, Math.floor(value)));
}

function uniqueJsonRpcMessages(messages: JsonRpcMessage[]): JsonRpcMessage[] {
  const seen = new Set<string>();
  const uniqueMessages: JsonRpcMessage[] = [];

  for (const message of messages) {
    const key = jsonRpcMessageDedupKey(message);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    uniqueMessages.push(message);
  }

  return uniqueMessages;
}

function jsonRpcMessageDedupKey(message: JsonRpcMessage): string {
  const eventId = jsonRpcNotificationEventId(message);
  if (eventId) {
    return `event:${eventId}`;
  }
  return `message:${JSON.stringify(message)}`;
}

function jsonRpcNotificationEventId(
  message: JsonRpcMessage,
): string | undefined {
  if (!isJsonRpcNotification(message)) {
    return undefined;
  }
  const params = message.params;
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    return undefined;
  }
  const event = (params as { event?: unknown }).event;
  if (!event || typeof event !== "object" || Array.isArray(event)) {
    return undefined;
  }
  const eventId =
    (event as { eventId?: unknown; event_id?: unknown }).eventId ??
    (event as { eventId?: unknown; event_id?: unknown }).event_id;
  return typeof eventId === "string" && eventId.trim()
    ? eventId.trim()
    : undefined;
}

function restoreProxyResponseId(
  message: JsonRpcMessage,
  originalId: RequestId,
): JsonRpcMessage {
  if (isJsonRpcResponse(message) || isJsonRpcErrorResponse(message)) {
    return {
      ...message,
      id: originalId,
    };
  }
  return message;
}

function restoreAppServerRequestError(
  error: unknown,
  originalId: RequestId,
): JsonRpcMessage[] | null {
  if (!(error instanceof AppServerRequestError)) {
    return null;
  }
  const messages =
    error.messages.length > 0 ? error.messages : [error.response];
  return messages.map((message) => restoreProxyResponseId(message, originalId));
}

async function resolveLaunchConfig(): Promise<ElectronAppServerLaunchConfig> {
  const dataDir = resolveAppServerDataDir();
  const envBinary = process.env.APP_SERVER_BIN?.trim();
  if (envBinary) {
    return {
      config: stdioSidecarWithRuntimeBackend(
        envBinary,
        process.env.APP_SERVER_POLICY_PATH,
        "runtime",
        dataDir,
      ),
    };
  }

  const resourcesPath = process.resourcesPath;
  const resourceRoots = [
    resourcesPath,
    path.resolve(app.getAppPath(), "dist-electron"),
  ];
  for (const resourceRoot of resourceRoots) {
    const config = await resolveResourceLaunchConfig(resourceRoot);
    if (config) {
      return config;
    }
  }

  const devBinaryPath = resolveDevAppServerBinaryPath(app.getAppPath());
  return {
    config: stdioSidecarWithRuntimeBackend(
      devBinaryPath,
      process.env.APP_SERVER_POLICY_PATH,
      "runtime",
      dataDir,
    ),
  };
}

async function resolveAppServerSidecarEnv(
  binaryPath: string,
  agentRoot: string,
): Promise<NodeJS.ProcessEnv | undefined> {
  const env: NodeJS.ProcessEnv = {
    ...resolveAppServerRuntimeLibraryEnv(binaryPath),
    LIME_AGENT_RUNTIME_ROOT: agentRoot,
    // AppDataRoot 由 Host 显式解析后下发；App Server 的叶子 writer 不再自行猜平台根。
    APP_SERVER_APP_DATA_DIR: resolveAppServerAppDataRoot(),
  };
  const bundledMarketplacePath = resolveBundledPluginMarketplacePath();
  if (bundledMarketplacePath) {
    env.LIME_BUNDLED_PLUGIN_MARKETPLACE = bundledMarketplacePath;
  }
  const currentNoProxy = APP_SERVER_NO_PROXY_ENV_KEYS.map(
    (key) => process.env[key],
  ).find((value) => Boolean(value?.trim()));
  const noProxy = mergeLoopbackNoProxy(currentNoProxy);
  if (noProxy && noProxy !== process.env.NO_PROXY) {
    env.NO_PROXY = noProxy;
  }
  if (noProxy && noProxy !== process.env.no_proxy) {
    env.no_proxy = noProxy;
  }

  if (!hasExplicitProxyEnv(process.env)) {
    const proxyUrl = await resolveElectronSystemProxyUrl();
    if (proxyUrl) {
      for (const key of APP_SERVER_PROXY_ENV_KEYS) {
        env[key] = proxyUrl;
      }
    }
  }

  return Object.keys(env).length > 0 ? env : undefined;
}

function resolveBundledPluginMarketplacePath(): string | undefined {
  const resourceRoots = [
    process.resourcesPath,
    path.resolve(app.getAppPath(), "dist-electron"),
  ];
  for (const resourceRoot of resourceRoots) {
    const marketplacePath = path.join(
      resourceRoot,
      "plugins",
      "openai-bundled",
      ".agents",
      "plugins",
      "marketplace.json",
    );
    try {
      if (readFileSync(marketplacePath, "utf8")) {
        return marketplacePath;
      }
    } catch {
      // 开发态或旧包可能没有 bundled marketplace，App Server 继续发现其他来源。
    }
  }
  return undefined;
}

function resolveAppServerRuntimeLibraryEnv(
  binaryPath: string,
): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {
    LIME_CONFIG_PATH: resolveAppServerConfigPath(),
  };
  const binaryDir = path.dirname(binaryPath);
  if (!binaryDir || binaryDir === ".") {
    return env;
  }

  if (process.platform === "darwin") {
    return {
      ...env,
      DYLD_FALLBACK_LIBRARY_PATH: prependPathEnv(
        process.env.DYLD_FALLBACK_LIBRARY_PATH,
        [binaryDir],
      ),
      DYLD_LIBRARY_PATH: prependPathEnv(process.env.DYLD_LIBRARY_PATH, [
        binaryDir,
      ]),
    };
  }

  if (process.platform === "linux") {
    return {
      ...env,
      LD_LIBRARY_PATH: prependPathEnv(process.env.LD_LIBRARY_PATH, [binaryDir]),
    };
  }

  if (process.platform === "win32") {
    return {
      ...env,
      PATH: prependPathEnv(process.env.PATH, [binaryDir]),
    };
  }

  return env;
}

function prependPathEnv(
  currentValue: string | undefined,
  entries: string[],
): string {
  const result: string[] = [];
  const seen = new Set<string>();
  const remember = (entry: string | undefined) => {
    const trimmed = entry?.trim();
    if (!trimmed) {
      return;
    }
    const key = process.platform === "win32" ? trimmed.toLowerCase() : trimmed;
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    result.push(trimmed);
  };

  for (const entry of entries) {
    remember(entry);
  }
  for (const entry of currentValue?.split(path.delimiter) ?? []) {
    remember(entry);
  }
  return result.join(path.delimiter);
}

function hasExplicitProxyEnv(env: NodeJS.ProcessEnv): boolean {
  return APP_SERVER_PROXY_ENV_KEYS.some((key) => Boolean(env[key]?.trim()));
}

async function resolveElectronSystemProxyUrl(): Promise<string | undefined> {
  if (process.platform !== "darwin") {
    return undefined;
  }

  try {
    const rules = await session.defaultSession.resolveProxy(
      APP_SERVER_PROXY_PROBE_URL,
    );
    return firstProxyRuleToUrl(rules);
  } catch (error) {
    console.warn(
      "[electron-host] failed to resolve system proxy for app-server",
      error,
    );
    return undefined;
  }
}

function firstProxyRuleToUrl(rules: string): string | undefined {
  for (const rawRule of rules.split(";")) {
    const rule = rawRule.trim();
    if (!rule || rule.toUpperCase() === "DIRECT") {
      continue;
    }
    const [kind = "", address = ""] = rule.split(/\s+/, 2);
    const normalizedAddress = address.trim();
    if (!normalizedAddress || normalizedAddress.includes("://")) {
      continue;
    }
    switch (kind.toUpperCase()) {
      case "PROXY":
        return `http://${normalizedAddress}`;
      case "HTTPS":
        return `https://${normalizedAddress}`;
      case "SOCKS":
      case "SOCKS5":
        return `socks5://${normalizedAddress}`;
      default:
        continue;
    }
  }
  return undefined;
}

function mergeLoopbackNoProxy(value: string | undefined): string {
  const entries = (value ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  const normalized = new Set(entries.map((entry) => entry.toLowerCase()));
  for (const host of APP_SERVER_LOOPBACK_NO_PROXY_HOSTS) {
    if (!normalized.has(host.toLowerCase())) {
      entries.push(host);
      normalized.add(host.toLowerCase());
    }
  }
  return entries.join(",");
}

async function resolveResourceLaunchConfig(
  resourcesPath: string,
): Promise<ElectronAppServerLaunchConfig | null> {
  const dataDir = resolveAppServerDataDir();
  const manifestPath = defaultReleaseManifestPath(resourcesPath);
  try {
    const manifest = await readReleaseManifest(manifestPath);
    const resolved = resolveSidecarFromReleaseManifest(manifest, {
      allowEnvOverride: false,
      resourcesPath,
      appPolicyPath: process.env.APP_SERVER_POLICY_PATH,
      dataDir,
      ...resolveRuntimeBackendLaunchOptions("runtime"),
    });
    if (resolved) {
      return {
        config: resolved.config,
        verifySha256: shouldVerifyResourceSha256(resourcesPath),
      };
    }
  } catch {
    // 开发态或未执行资源准备时可以没有 packaged manifest。
  }
  return null;
}

function shouldVerifyResourceSha256(resourcesPath: string): boolean {
  if (process.platform !== "darwin" || !app.isPackaged) {
    return true;
  }

  return path.resolve(resourcesPath) !== path.resolve(process.resourcesPath);
}

function stdioSidecarWithRuntimeBackend(
  binaryPath: string,
  appPolicyPath: string | undefined,
  defaultBackendMode: NonNullable<SidecarLaunchConfig["backendMode"]>,
  dataDir: string,
): SidecarLaunchConfig {
  return {
    ...stdioSidecar(binaryPath, appPolicyPath, dataDir),
    ...resolveRuntimeBackendLaunchOptions(defaultBackendMode),
  };
}

function resolveAppServerDataDir(): string {
  return resolveCurrentDesktopStorageRoots(app.getPath("userData")).agentRoot;
}

function resolveAppServerAppDataRoot(): string {
  return resolveCurrentDesktopStorageRoots(app.getPath("userData")).appDataRoot;
}

function resolveAppServerConfigPath(): string {
  return path.join(app.getPath("userData"), APP_SERVER_CONFIG_FILE_NAME);
}

function resolveRuntimeBackendLaunchOptions(
  defaultBackendMode: NonNullable<SidecarLaunchConfig["backendMode"]>,
): Pick<
  SidecarLaunchConfig,
  "backendMode" | "backendCommand" | "backendArgs" | "backendTimeoutMs"
> {
  const backendMode = resolveBackendMode(
    process.env.APP_SERVER_BACKEND_MODE,
    defaultBackendMode,
  );
  const config: Pick<
    SidecarLaunchConfig,
    "backendMode" | "backendCommand" | "backendArgs" | "backendTimeoutMs"
  > = {
    backendMode,
  };

  if (backendMode === "external") {
    const backendCommand = process.env.APP_SERVER_BACKEND_COMMAND?.trim();
    if (backendCommand) {
      config.backendCommand = backendCommand;
    }
    const backendArgs = parseBackendArgs(process.env.APP_SERVER_BACKEND_ARGS);
    if (backendArgs.length > 0) {
      config.backendArgs = backendArgs;
    }
    const backendTimeoutMs = parsePositiveInteger(
      process.env.APP_SERVER_BACKEND_TIMEOUT_MS,
    );
    if (backendTimeoutMs !== undefined) {
      config.backendTimeoutMs = backendTimeoutMs;
    }
  }

  return config;
}

function resolveBackendMode(
  value: string | undefined,
  fallback: NonNullable<SidecarLaunchConfig["backendMode"]>,
): NonNullable<SidecarLaunchConfig["backendMode"]> {
  const normalized = value?.trim();
  if (normalized === "mock") {
    throw new Error(
      "Electron App Server host does not allow APP_SERVER_BACKEND_MODE=mock. Use APP_SERVER_BACKEND_MODE=runtime, APP_SERVER_BACKEND_MODE=external with APP_SERVER_BACKEND_COMMAND, or APP_SERVER_BACKEND_MODE=unavailable.",
    );
  }
  if (
    normalized === "runtime" ||
    normalized === "unavailable" ||
    normalized === "external"
  ) {
    return normalized;
  }
  return fallback;
}

function parseBackendArgs(value: string | undefined): string[] {
  const trimmed = value?.trim();
  if (!trimmed) {
    return [];
  }
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    return Array.isArray(parsed)
      ? parsed.filter((entry): entry is string => typeof entry === "string")
      : [];
  } catch {
    return [];
  }
}

function parsePositiveInteger(value: string | undefined): number | undefined {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function resolveAppServerRequestTimeoutMs(
  method: string,
  requestedTimeoutMs?: unknown,
): number {
  const defaultTimeoutMs = resolveDefaultAppServerRequestTimeoutMs(method);
  const overrideTimeoutMs = parsePositiveIntegerValue(requestedTimeoutMs);
  if (!overrideTimeoutMs) {
    return defaultTimeoutMs;
  }
  return Math.min(
    Math.max(defaultTimeoutMs, overrideTimeoutMs),
    APP_SERVER_REQUEST_TIMEOUT_OVERRIDE_CEILING_MS,
  );
}

function resolveDefaultAppServerRequestTimeoutMs(method: string): number {
  if (method === "command/exec") return 600_000;
  if (method === APP_SERVER_CONVERSATION_IMPORT_THREAD_COMMIT_METHOD) {
    return APP_SERVER_CONVERSATION_IMPORT_THREAD_COMMIT_TIMEOUT_MS;
  }
  if (method === APP_SERVER_CONVERSATION_IMPORT_JOB_READ_METHOD) {
    return APP_SERVER_CONVERSATION_IMPORT_SCAN_TIMEOUT_MS;
  }
  if (method === "conversationImport/source/scan") {
    return APP_SERVER_CONVERSATION_IMPORT_SCAN_TIMEOUT_MS;
  }
  if (method === "conversationImport/thread/preview") {
    return APP_SERVER_CONVERSATION_IMPORT_PREVIEW_TIMEOUT_MS;
  }
  if (method !== APP_SERVER_TURN_START_METHOD) {
    return DEFAULT_APP_SERVER_REQUEST_TIMEOUT_MS;
  }
  const backendTimeoutMs = parsePositiveInteger(
    process.env.APP_SERVER_BACKEND_TIMEOUT_MS,
  );
  return backendTimeoutMs
    ? backendTimeoutMs + APP_SERVER_BACKEND_TIMEOUT_GRACE_MS
    : DEFAULT_APP_SERVER_REQUEST_TIMEOUT_MS;
}

function parsePositiveIntegerValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? value
    : undefined;
}

function resolveDevAppServerBinaryPath(appPath: string): string {
  return path.join(
    resolveCargoTargetDirectory(appPath),
    "debug",
    process.platform === "win32" ? "app-server.exe" : "app-server",
  );
}

function resolveCargoTargetDirectory(appPath: string): string {
  const fallback = path.resolve(appPath, "lime-rs", "target");
  try {
    const config = readFileSync(
      path.resolve(appPath, ".cargo", "config.toml"),
      "utf8",
    );
    const match = config.match(/^\s*target-dir\s*=\s*["']([^"']+)["']/m);
    if (!match?.[1]?.trim()) {
      return fallback;
    }
    return path.resolve(appPath, match[1].trim());
  } catch {
    return fallback;
  }
}

function isJsonRpcRequestLike(
  message: JsonRpcMessage,
): message is JsonRpcRequest {
  return "id" in message && "method" in message;
}

function isInitializedNotification(message: JsonRpcMessage): boolean {
  return (
    isJsonRpcNotification(message) && message.method === METHOD_INITIALIZED
  );
}

function initializeResponseMessage(
  request: JsonRpcRequest,
  response: InitializeResponse,
): JsonRpcMessage {
  return {
    id: request.id,
    result: response,
  };
}
