import {
  ERROR_CODES,
  METHOD_SERVER_REQUEST_RESOLVED,
  isAppServerServerRequestMethod,
} from "../../../packages/app-server-client/src/protocol";
import { AppServerClient } from "./appServerClient";
import {
  getDefaultAppServerEventBus,
  type AppServerEventBus,
} from "./appServerEventBus";
import { resolveClawTraceEnabled } from "../developerFeatures";
import type {
  AppServerJsonRpcError,
  AppServerJsonRpcRequest,
  AppServerRequestId,
} from "./appServerTypes";

type ServerRequestResponder = Pick<
  AppServerClient,
  "respondServerRequest" | "rejectServerRequest"
>;

export type AppServerServerRequestHandler<TParams, TResult> = (
  params: TParams,
  request: AppServerJsonRpcRequest,
  signal: AbortSignal,
) => Promise<TResult> | TResult;

type RegisteredServerRequestHandler = AppServerServerRequestHandler<
  unknown,
  unknown
>;

type ResolvedServerRequestNotification = {
  requestId: AppServerRequestId;
  threadId: string;
};

type InFlightServerRequest = {
  controller: AbortController;
  threadId: string | null;
  turnId: string | null;
};

type TerminalServerRequestScope = {
  method: "thread/closed" | "turn/completed";
  threadId: string;
  turnId: string | null;
};

export const APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_KEY =
  "lime:debug:app-server-server-request-lifecycle:v1";
const APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_LIMIT = 100;

export class AppServerServerRequestDispatcher {
  readonly #eventBus: Pick<AppServerEventBus, "subscribe">;
  readonly #handlers = new Map<string, RegisteredServerRequestHandler>();
  readonly #inFlightRequests = new Map<string, InFlightServerRequest>();
  readonly #resolvedRequestKeys = new Set<string>();
  readonly #settledRequestKeys = new Set<string>();
  readonly #responder: ServerRequestResponder;
  #connectionGeneration = 0;
  #unsubscribe: (() => void) | null = null;

  constructor(
    responder: ServerRequestResponder = new AppServerClient(),
    eventBus: Pick<
      AppServerEventBus,
      "subscribe"
    > = getDefaultAppServerEventBus(),
  ) {
    this.#responder = responder;
    this.#eventBus = eventBus;
  }

  register<TParams, TResult>(
    method: string,
    handler: AppServerServerRequestHandler<TParams, TResult>,
  ): () => void {
    if (!isAppServerServerRequestMethod(method)) {
      throw new Error(`App Server method is not a server request: ${method}`);
    }
    if (this.#handlers.has(method)) {
      throw new Error(
        `App Server server request handler already registered: ${method}`,
      );
    }
    this.#handlers.set(method, handler as RegisteredServerRequestHandler);
    this.#ensureSubscribed();
    return () => {
      if (this.#handlers.get(method) === handler) {
        this.#handlers.delete(method);
      }
      this.#stopIfIdle();
    };
  }

  async dispatch(request: AppServerJsonRpcRequest): Promise<boolean> {
    const requestKey = stableRequestKey(request.id, requestThreadId(request));
    if (
      this.#inFlightRequests.has(requestKey) ||
      this.#settledRequestKeys.has(requestKey)
    ) {
      return false;
    }
    const identity = serverRequestIdentity(request);
    const controller = new AbortController();
    const connectionGeneration = this.#connectionGeneration;
    this.#inFlightRequests.set(requestKey, {
      controller,
      threadId: identity.threadId,
      turnId: identity.turnId,
    });
    appendServerRequestLifecycleTrace({
      kind: "request",
      ...identity,
    });
    try {
      const handler = this.#handlers.get(request.method);
      if (!handler) {
        await this.#reject(request.id, {
          code: ERROR_CODES.methodNotFound,
          message: `App Server server request method is not handled: ${request.method}`,
        });
        return false;
      }
      try {
        const result = await handler(
          request.params,
          request,
          controller.signal,
        );
        if (
          controller.signal.aborted ||
          this.#settledRequestKeys.has(requestKey)
        ) {
          return false;
        }
        await this.#responder.respondServerRequest(request.id, result);
        appendServerRequestLifecycleTrace({
          kind: "response",
          ...serverRequestIdentity(request),
          decision: readDecision(result),
        });
        return true;
      } catch (error) {
        if (
          controller.signal.aborted ||
          this.#settledRequestKeys.has(requestKey)
        ) {
          return false;
        }
        await this.#reject(request.id, {
          code: ERROR_CODES.runtimeError,
          message: serverRequestErrorMessage(error),
        });
        return false;
      }
    } finally {
      this.#inFlightRequests.delete(requestKey);
      if (connectionGeneration === this.#connectionGeneration) {
        this.#settledRequestKeys.add(requestKey);
      }
    }
  }

  reset(): void {
    this.#connectionGeneration += 1;
    for (const request of this.#inFlightRequests.values()) {
      request.controller.abort();
    }
    this.#handlers.clear();
    this.#inFlightRequests.clear();
    this.#resolvedRequestKeys.clear();
    this.#settledRequestKeys.clear();
    this.#unsubscribe?.();
    this.#unsubscribe = null;
  }

  #ensureSubscribed(): void {
    if (this.#unsubscribe) {
      return;
    }
    this.#unsubscribe = this.#eventBus.subscribe({
      onNotifications: (notifications) => {
        for (const notification of notifications) {
          const resolved = readResolvedNotification(notification);
          if (resolved) {
            this.#resolve(resolved);
          }
          const terminal = readTerminalServerRequestScope(notification);
          if (terminal) {
            this.#terminateScope(terminal);
          }
          recordRuntimeTerminalNotification(notification);
        }
      },
      onServerRequests: (requests) => {
        for (const request of requests) {
          void this.dispatch(request);
        }
      },
    });
  }

  #stopIfIdle(): void {
    if (this.#handlers.size > 0) {
      return;
    }
    this.#unsubscribe?.();
    this.#unsubscribe = null;
  }

  async #reject(id: AppServerRequestId, error: AppServerJsonRpcError) {
    await this.#responder.rejectServerRequest(id, error);
  }

  #resolve(resolved: ResolvedServerRequestNotification): void {
    const requestKey = stableRequestKey(resolved.requestId, resolved.threadId);
    if (this.#resolvedRequestKeys.has(requestKey)) {
      return;
    }
    this.#resolvedRequestKeys.add(requestKey);
    appendServerRequestLifecycleTrace({
      kind: "resolved",
      method: METHOD_SERVER_REQUEST_RESOLVED,
      id: resolved.requestId,
      threadId: resolved.threadId,
    });
    this.#settledRequestKeys.add(requestKey);
    this.#inFlightRequests.get(requestKey)?.controller.abort();
  }

  #terminateScope(scope: TerminalServerRequestScope): void {
    for (const request of this.#inFlightRequests.values()) {
      if (request.threadId !== scope.threadId) {
        continue;
      }
      if (scope.turnId && request.turnId !== scope.turnId) {
        continue;
      }
      request.controller.abort();
    }
  }
}

function serverRequestIdentity(request: AppServerJsonRpcRequest) {
  const params =
    request.params &&
    typeof request.params === "object" &&
    !Array.isArray(request.params)
      ? (request.params as Record<string, unknown>)
      : {};
  return {
    id: request.id,
    method: request.method,
    threadId: readString(params.threadId),
    turnId: readString(params.turnId),
    itemId: readString(params.itemId),
    approvalId: readString(params.approvalId),
  };
}

function readDecision(result: unknown): string | null {
  if (!result || typeof result !== "object" || Array.isArray(result)) {
    return null;
  }
  return readString((result as Record<string, unknown>).decision);
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function recordRuntimeTerminalNotification(notification: {
  method: string;
  params?: unknown;
}): void {
  if (
    notification.method !== "item/completed" &&
    notification.method !== "turn/completed" &&
    notification.method !== "thread/closed"
  ) {
    return;
  }
  const params =
    notification.params &&
    typeof notification.params === "object" &&
    !Array.isArray(notification.params)
      ? (notification.params as Record<string, unknown>)
      : {};
  appendServerRequestLifecycleTrace({
    kind: "terminal",
    method: notification.method,
    threadId: readString(params.threadId),
    turnId: readString(params.turnId),
  });
}

function readTerminalServerRequestScope(notification: {
  method: string;
  params?: unknown;
}): TerminalServerRequestScope | null {
  if (
    notification.method !== "turn/completed" &&
    notification.method !== "thread/closed"
  ) {
    return null;
  }
  const params =
    notification.params &&
    typeof notification.params === "object" &&
    !Array.isArray(notification.params)
      ? (notification.params as Record<string, unknown>)
      : null;
  const threadId = params ? readString(params.threadId) : null;
  if (!threadId) {
    return null;
  }
  if (notification.method === "thread/closed") {
    return { method: "thread/closed", threadId, turnId: null };
  }
  const turnId = readString(params?.turnId);
  return turnId
    ? { method: "turn/completed", threadId, turnId }
    : null;
}

function appendServerRequestLifecycleTrace(
  entry: Record<string, unknown>,
): void {
  if (typeof window === "undefined" || !resolveClawTraceEnabled()) {
    return;
  }
  try {
    const previous = window.localStorage.getItem(
      APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_KEY,
    );
    const parsed = previous ? JSON.parse(previous) : [];
    const entries = Array.isArray(parsed) ? parsed : [];
    const latestSequence = entries.at(-1)?.sequence;
    const sequence =
      typeof latestSequence === "number"
        ? latestSequence + 1
        : entries.length + 1;
    entries.push({ sequence, ...entry });
    window.localStorage.setItem(
      APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_KEY,
      JSON.stringify(
        entries.slice(-APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_LIMIT),
      ),
    );
  } catch {
    // Debug evidence must never affect request handling.
  }
}

let defaultAppServerServerRequestDispatcher: AppServerServerRequestDispatcher | null =
  null;

export function getDefaultAppServerServerRequestDispatcher(): AppServerServerRequestDispatcher {
  if (!defaultAppServerServerRequestDispatcher) {
    defaultAppServerServerRequestDispatcher =
      new AppServerServerRequestDispatcher();
  }
  return defaultAppServerServerRequestDispatcher;
}

export function resetDefaultAppServerServerRequestDispatcherForTests(): void {
  defaultAppServerServerRequestDispatcher?.reset();
  defaultAppServerServerRequestDispatcher = null;
}

function stableRequestKey(
  id: AppServerRequestId,
  threadId: string | null,
): string {
  return JSON.stringify([typeof id, String(id), threadId]);
}

function requestThreadId(request: AppServerJsonRpcRequest): string | null {
  const params =
    request.params &&
    typeof request.params === "object" &&
    !Array.isArray(request.params)
      ? (request.params as Record<string, unknown>)
      : {};
  return readString(params.threadId);
}

function serverRequestErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  return "App Server server request handler failed";
}

function readResolvedNotification(notification: {
  method: string;
  params?: unknown;
}): ResolvedServerRequestNotification | null {
  if (notification.method !== METHOD_SERVER_REQUEST_RESOLVED) {
    return null;
  }
  const params = notification.params;
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    return null;
  }
  const record = params as Record<string, unknown>;
  const requestId = record.requestId;
  if (typeof requestId !== "string" && typeof requestId !== "number") {
    return null;
  }
  const threadId = readString(record.threadId);
  return threadId ? { requestId, threadId } : null;
}
