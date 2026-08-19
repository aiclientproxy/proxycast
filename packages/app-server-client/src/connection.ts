import * as protocol from "./protocol.js";
import { installAppServerConnectionMethods } from "./connection-methods.js";
import { AppServerClient } from "./request-client.js";

export type AppServerMessageTransport = {
  send(message: protocol.JsonRpcMessage): void;
  nextMessage(timeoutMs?: number): Promise<protocol.JsonRpcMessage>;
};

export type AppServerServerMessage =
  | protocol.JsonRpcRequest
  | protocol.JsonRpcNotification;

export type AppServerServerRequestHandler = (
  message: protocol.JsonRpcRequest,
) => Promise<boolean> | boolean;

export type AppServerRequestOptions = {
  timeoutMs?: number;
  signal?: AbortSignal;
};

// The single read pump uses bounded reads so request timeouts and aborts remain responsive.
const APP_SERVER_TRANSPORT_READ_SLICE_MS = 25;

export type AppServerRequestResult<T> = {
  id: protocol.RequestId;
  result: T;
  response: protocol.JsonRpcResponse;
  notifications: protocol.JsonRpcNotification[];
  messages: protocol.JsonRpcMessage[];
};

export type AppServerRequestFirstMessageResult<T> =
  | (AppServerRequestResult<T> & { completed: true })
  | {
      id: protocol.RequestId;
      completed: false;
      notifications: protocol.JsonRpcNotification[];
      messages: protocol.JsonRpcMessage[];
    };

type PendingRequestMode = "response" | "first-message";

type PendingRequestResult =
  | {
      kind: "response";
      value: AppServerRequestResult<unknown>;
    }
  | {
      kind: "notification";
      notification: protocol.JsonRpcNotification;
    };

type PendingRequestRead = {
  abort?: () => void;
  id: protocol.RequestId;
  method: string;
  mode: PendingRequestMode;
  reject: (error: unknown) => void;
  resolve: (result: PendingRequestResult) => void;
  signal?: AbortSignal;
  timeout?: ReturnType<typeof setTimeout>;
};

type PendingMessageRead = {
  accept: (message: protocol.JsonRpcMessage) => boolean;
  reject: (error: unknown) => void;
  resolve: (message: protocol.JsonRpcMessage) => void;
  timeout?: ReturnType<typeof setTimeout>;
};

export class AppServerRequestError extends Error {
  readonly id: protocol.RequestId;
  readonly method: string;
  readonly response: protocol.JsonRpcErrorResponse;
  readonly notifications: protocol.JsonRpcNotification[];
  readonly messages: protocol.JsonRpcMessage[];

  constructor(
    method: string,
    response: protocol.JsonRpcErrorResponse,
    notifications: protocol.JsonRpcNotification[],
    messages: protocol.JsonRpcMessage[],
  ) {
    super(`${method} failed: ${response.error.message}`);
    this.name = "AppServerRequestError";
    this.id = response.id;
    this.method = method;
    this.response = response;
    this.notifications = notifications;
    this.messages = messages;
  }
}

export class AppServerRequestAbortedError extends Error {
  readonly id: protocol.RequestId;
  readonly method: string;
  readonly reason?: unknown;

  constructor(method: string, id: protocol.RequestId, reason?: unknown) {
    super("app-server request aborted");
    this.name = "AppServerRequestAbortedError";
    this.id = id;
    this.method = method;
    this.reason = reason;
  }
}

export class AppServerConnection {
  readonly client: AppServerClient;
  readonly transport: AppServerMessageTransport;

  #bufferedMessages: protocol.JsonRpcMessage[] = [];
  #detachedRequestIds = new Set<protocol.RequestId>();
  #messageReads: PendingMessageRead[] = [];
  #pendingRequests = new Map<protocol.RequestId, PendingRequestRead>();
  #pendingServerRequestIds = new Set<protocol.RequestId>();
  #readPump: Promise<void> | null = null;
  #resolvedServerRequestIds = new Set<protocol.RequestId>();
  #serverRequestHandler: AppServerServerRequestHandler | null = null;

  constructor(
    transport: AppServerMessageTransport,
    client: AppServerClient = new AppServerClient(),
  ) {
    this.transport = transport;
    this.client = client;
  }

  async request<T>(
    requestMessage: protocol.JsonRpcRequest,
    method = requestMessage.method,
    options: AppServerRequestOptions = {},
  ): Promise<AppServerRequestResult<T>> {
    throwIfRequestAborted(options.signal, method, requestMessage.id);
    this.transport.send(requestMessage);
    return await this.waitForResponse<T>(requestMessage.id, method, options);
  }

  async requestUntilFirstNotificationOrResponse<T>(
    requestMessage: protocol.JsonRpcRequest,
    method = requestMessage.method,
    options: AppServerRequestOptions = {},
  ): Promise<AppServerRequestFirstMessageResult<T>> {
    throwIfRequestAborted(options.signal, method, requestMessage.id);
    this.transport.send(requestMessage);
    const result = await this.#waitForRequest(
      requestMessage.id,
      method,
      "first-message",
      options,
    );
    if (result.kind === "notification") {
      return {
        id: requestMessage.id,
        completed: false,
        notifications: [result.notification],
        messages: [result.notification],
      };
    }
    return {
      ...(result.value as AppServerRequestResult<T>),
      completed: true,
    };
  }

  async waitForResponse<T>(
    id: protocol.RequestId,
    method: string,
    options: AppServerRequestOptions = {},
  ): Promise<AppServerRequestResult<T>> {
    const result = await this.#waitForRequest(id, method, "response", options);
    if (result.kind !== "response") {
      throw new Error(`unexpected notification while waiting for ${method}`);
    }
    return result.value as AppServerRequestResult<T>;
  }

  async nextNotification(
    timeoutMs?: number,
  ): Promise<protocol.JsonRpcNotification> {
    return (await this.#waitForMessage(
      protocol.isJsonRpcNotification,
      timeoutMs,
    )) as protocol.JsonRpcNotification;
  }

  async nextServerMessage(timeoutMs?: number): Promise<AppServerServerMessage> {
    return (await this.#waitForMessage(
      (message) =>
        protocol.isJsonRpcNotification(message) ||
        protocol.isJsonRpcRequest(message),
      timeoutMs,
    )) as AppServerServerMessage;
  }

  async nextMessage(timeoutMs?: number): Promise<protocol.JsonRpcMessage> {
    return await this.#waitForMessage(() => true, timeoutMs);
  }

  /**
   * Sends a response for an App Server initiated (typed server) request.
   * The request id is the only routing key; callers must not infer identity
   * from thread, turn, or action metadata.
   */
  respondServerRequest<T>(id: protocol.RequestId, result: T): void {
    this.#consumeServerRequestId(id);
    this.transport.send(protocol.response(id, result));
  }

  /**
   * Installs an optional host-side interceptor for reverse requests.
   * Returning true consumes the request; false keeps it available to drainEvents.
   */
  setServerRequestHandler(
    handler: AppServerServerRequestHandler | null,
  ): void {
    this.#serverRequestHandler = handler;
    if (handler) {
      this.#ensureReadPump();
    }
  }

  /** Sends a JSON-RPC error for an App Server initiated request. */
  rejectServerRequest(
    id: protocol.RequestId,
    error: protocol.JsonRpcError,
  ): void {
    this.#consumeServerRequestId(id);
    this.transport.send(protocol.errorResponse(id, error));
  }

  #waitForRequest(
    id: protocol.RequestId,
    method: string,
    mode: PendingRequestMode,
    options: AppServerRequestOptions,
  ): Promise<PendingRequestResult> {
    throwIfRequestAborted(options.signal, method, id);
    if (this.#pendingRequests.has(id)) {
      throw new Error(`duplicate pending app-server request id: ${String(id)}`);
    }

    return new Promise<PendingRequestResult>((resolve, reject) => {
      const pending: PendingRequestRead = {
        id,
        method,
        mode,
        reject,
        resolve,
        signal: options.signal,
      };
      if (options.timeoutMs !== undefined) {
        pending.timeout = setTimeout(() => {
          this.#detachedRequestIds.add(id);
          this.#rejectPendingRequest(
            id,
            new Error(
              `timed out waiting for app-server message after ${options.timeoutMs}ms`,
            ),
          );
        }, Math.max(0, options.timeoutMs));
      }
      if (options.signal) {
        pending.abort = () => {
          if (!this.#pendingRequests.has(id)) {
            return;
          }
          this.#sendCancelRequest(id);
          this.#detachedRequestIds.add(id);
          this.#rejectPendingRequest(
            id,
            new AppServerRequestAbortedError(
              method,
              id,
              options.signal?.reason,
            ),
          );
        };
        options.signal.addEventListener("abort", pending.abort, { once: true });
      }
      this.#pendingRequests.set(id, pending);
      if (options.signal?.aborted) {
        pending.abort?.();
        return;
      }
      const bufferedIndex = this.#bufferedMessages.findIndex((message) => {
        if (mode === "first-message" && protocol.isJsonRpcNotification(message)) {
          return true;
        }
        return (
          (protocol.isJsonRpcResponse(message) ||
            protocol.isJsonRpcErrorResponse(message)) &&
          message.id === id
        );
      });
      if (bufferedIndex >= 0) {
        const [buffered] = this.#bufferedMessages.splice(bufferedIndex, 1);
        this.#dispatchIncomingMessage(buffered);
      }
      this.#ensureReadPump();
    });
  }

  #waitForMessage(
    accept: (message: protocol.JsonRpcMessage) => boolean,
    timeoutMs?: number,
  ): Promise<protocol.JsonRpcMessage> {
    const buffered = this.#shiftBufferedMessage(accept);
    if (buffered) {
      return Promise.resolve(buffered);
    }
    if (timeoutMs !== undefined && timeoutMs <= 0) {
      return Promise.reject(
        new Error(
          `timed out waiting for app-server message after ${timeoutMs}ms`,
        ),
      );
    }

    return new Promise<protocol.JsonRpcMessage>((resolve, reject) => {
      const pending: PendingMessageRead = {
        accept,
        reject,
        resolve,
      };
      if (timeoutMs !== undefined) {
        pending.timeout = setTimeout(() => {
          this.#removeMessageRead(pending);
          reject(
            new Error(
              `timed out waiting for app-server message after ${timeoutMs}ms`,
            ),
          );
        }, timeoutMs);
      }
      this.#messageReads.push(pending);
      this.#ensureReadPump();
    });
  }

  #shiftBufferedMessage(
    accept: (message: protocol.JsonRpcMessage) => boolean,
  ): protocol.JsonRpcMessage | undefined {
    for (let index = 0; index < this.#bufferedMessages.length; index += 1) {
      const message = this.#bufferedMessages[index];
      if (this.#consumeDetachedRequestMessage(message)) {
        this.#bufferedMessages.splice(index, 1);
        index -= 1;
        continue;
      }
      if (!accept(message)) {
        continue;
      }
      this.#bufferedMessages.splice(index, 1);
      return message;
    }
    return undefined;
  }

  #ensureReadPump(): void {
    if (this.#readPump || !this.#hasReadDemand()) {
      return;
    }
    const pump = this.#runReadPump();
    this.#readPump = pump;
    void pump.finally(() => {
      if (this.#readPump === pump) {
        this.#readPump = null;
      }
      if (this.#hasReadDemand()) {
        this.#ensureReadPump();
      }
    });
  }

  async #runReadPump(): Promise<void> {
    while (this.#hasReadDemand()) {
      let message: protocol.JsonRpcMessage;
      try {
        message = await this.transport.nextMessage(
          APP_SERVER_TRANSPORT_READ_SLICE_MS,
        );
      } catch (error) {
        if (isAppServerTransportReadTimeoutError(error)) {
          continue;
        }
        this.#failPendingReads(error);
        return;
      }
      this.#dispatchIncomingMessage(message);
    }
  }

  #dispatchIncomingMessage(message: protocol.JsonRpcMessage): void {
    this.#observeServerMessage(message);
    if (this.#consumeDetachedRequestMessage(message)) {
      return;
    }

    if (protocol.isJsonRpcRequest(message) && this.#serverRequestHandler) {
      void Promise.resolve(this.#serverRequestHandler(message))
        .then((handled) => {
          if (!handled) {
            this.#dispatchBufferedMessage(message);
          }
        })
        .catch(() => {
          this.#dispatchBufferedMessage(message);
        });
      return;
    }

    if (
      protocol.isJsonRpcResponse(message) ||
      protocol.isJsonRpcErrorResponse(message)
    ) {
      const pending = this.#pendingRequests.get(message.id);
      if (pending) {
        if (protocol.isJsonRpcErrorResponse(message)) {
          this.#rejectPendingRequest(
            message.id,
            new AppServerRequestError(pending.method, message, [], [message]),
          );
        } else {
          this.#resolvePendingRequest(message.id, {
            kind: "response",
            value: {
              id: message.id,
              result: message.result,
              response: message,
              notifications: [],
              messages: [message],
            },
          });
        }
        return;
      }
    }

    if (protocol.isJsonRpcNotification(message)) {
      const firstMessagePending = Array.from(
        this.#pendingRequests.values(),
      ).find((pending) => pending.mode === "first-message");
      if (firstMessagePending) {
        this.#detachedRequestIds.add(firstMessagePending.id);
        this.#resolvePendingRequest(firstMessagePending.id, {
          kind: "notification",
          notification: message,
        });
      }
    }

    this.#dispatchBufferedMessage(message);
  }

  #dispatchBufferedMessage(message: protocol.JsonRpcMessage): void {
    const waiterIndex = this.#messageReads.findIndex((pending) =>
      pending.accept(message),
    );
    if (waiterIndex < 0) {
      this.#bufferedMessages.push(message);
      return;
    }
    const [pending] = this.#messageReads.splice(waiterIndex, 1);
    if (pending.timeout) {
      clearTimeout(pending.timeout);
    }
    pending.resolve(message);
  }

  #resolvePendingRequest(
    id: protocol.RequestId,
    result: PendingRequestResult,
  ): void {
    const pending = this.#takePendingRequest(id);
    pending?.resolve(result);
  }

  #rejectPendingRequest(id: protocol.RequestId, error: unknown): void {
    const pending = this.#takePendingRequest(id);
    pending?.reject(error);
  }

  #takePendingRequest(id: protocol.RequestId): PendingRequestRead | undefined {
    const pending = this.#pendingRequests.get(id);
    if (!pending) {
      return undefined;
    }
    this.#pendingRequests.delete(id);
    if (pending.timeout) {
      clearTimeout(pending.timeout);
    }
    if (pending.signal && pending.abort) {
      pending.signal.removeEventListener("abort", pending.abort);
    }
    return pending;
  }

  #removeMessageRead(pending: PendingMessageRead): void {
    const index = this.#messageReads.indexOf(pending);
    if (index >= 0) {
      this.#messageReads.splice(index, 1);
    }
    if (pending.timeout) {
      clearTimeout(pending.timeout);
    }
  }

  #failPendingReads(error: unknown): void {
    for (const id of Array.from(this.#pendingRequests.keys())) {
      this.#rejectPendingRequest(id, error);
    }
    const messageReads = this.#messageReads.splice(0);
    for (const pending of messageReads) {
      if (pending.timeout) {
        clearTimeout(pending.timeout);
      }
      pending.reject(error);
    }
  }

  #hasReadDemand(): boolean {
    return (
      this.#pendingRequests.size > 0 ||
      this.#messageReads.length > 0 ||
      this.#serverRequestHandler !== null
    );
  }

  #observeServerMessage(message: protocol.JsonRpcMessage): void {
    if (protocol.isJsonRpcRequest(message)) {
      if (this.#resolvedServerRequestIds.has(message.id)) {
        return;
      }
      this.#pendingServerRequestIds.add(message.id);
      return;
    }
    if (
      protocol.isJsonRpcNotification(message) &&
      message.method === protocol.METHOD_SERVER_REQUEST_RESOLVED
    ) {
      const params = message.params;
      if (
        params &&
        typeof params === "object" &&
        !Array.isArray(params) &&
        (typeof (params as { requestId?: unknown }).requestId === "string" ||
          typeof (params as { requestId?: unknown }).requestId === "number")
      ) {
        const requestId = (params as { requestId: protocol.RequestId })
          .requestId;
        this.#pendingServerRequestIds.delete(requestId);
        this.#resolvedServerRequestIds.add(requestId);
        while (this.#resolvedServerRequestIds.size > 2_048) {
          const oldest = this.#resolvedServerRequestIds.values().next().value;
          if (oldest === undefined) {
            break;
          }
          this.#resolvedServerRequestIds.delete(oldest);
        }
      }
    }
  }

  #consumeServerRequestId(id: protocol.RequestId): void {
    if (this.#pendingServerRequestIds.delete(id)) {
      return;
    }
    throw new Error(
      `unknown or already resolved server request id: ${String(id)}`,
    );
  }

  #consumeDetachedRequestMessage(message: protocol.JsonRpcMessage): boolean {
    if (
      (protocol.isJsonRpcResponse(message) ||
        protocol.isJsonRpcErrorResponse(message)) &&
      this.#detachedRequestIds.has(message.id)
    ) {
      this.#detachedRequestIds.delete(message.id);
      return true;
    }
    return false;
  }

  #sendCancelRequest(id: protocol.RequestId): void {
    this.transport.send(protocol.cancelRequest(id));
  }
}

installAppServerConnectionMethods(AppServerConnection.prototype);

function isAppServerTransportReadTimeoutError(error: unknown): boolean {
  return (
    error instanceof Error &&
    error.message.includes("timed out waiting for app-server message after")
  );
}

function throwIfRequestAborted(
  signal: AbortSignal | undefined,
  method: string,
  id: protocol.RequestId,
): void {
  if (signal?.aborted) {
    throw new AppServerRequestAbortedError(method, id, signal.reason);
  }
}
