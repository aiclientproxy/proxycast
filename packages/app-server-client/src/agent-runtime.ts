import {
  type AgentSessionActionRespondParams,
  type AgentSessionActionRespondResponse,
  type ThreadReadParams,
  type ThreadReadResponse,
  type ThreadMemoryModeSetParams,
  type ThreadMemoryModeSetResponse,
  type ThreadSettingsUpdateParams,
  type ThreadSettingsUpdateResponse,
  type AgentSessionToolInventoryReadParams,
  type AgentSessionToolInventoryReadResponse,
  type TurnInterruptParams,
  type TurnInterruptResponse,
  type TurnStartParams,
  type TurnStartResponse,
  type TurnSteerParams,
  type TurnSteerResponse,
  type ServerNotification,
  type JsonRpcMessage,
  type JsonRpcError,
  type RequestId,
} from "./protocol.js";
import { serverNotification } from "./server-notifications.js";
import {
  AppServerConnection,
  type AppServerRequestOptions,
  type AppServerRequestResult,
} from "./connection.js";

export type AgentRuntimeLifecycleNotification = Extract<
  ServerNotification,
  {
    method:
      | "thread/started"
      | "turn/started"
      | "turn/completed"
      | "item/started"
      | "item/completed"
      | "item/autoApprovalReview/started"
      | "item/autoApprovalReview/completed"
      | "item/agentMessage/delta"
      | "item/commandExecution/outputDelta"
      | "item/commandExecution/terminalInteraction"
      | "item/fileChange/patchUpdated"
      | "item/mcpToolCall/progress"
      | "item/plan/delta"
      | "item/reasoning/summaryTextDelta"
      | "item/reasoning/summaryPartAdded"
      | "item/reasoning/textDelta"
      | "thread/settings/updated";
  }
>;

export type AgentRuntimeSignalNotification = Extract<
  ServerNotification,
  {
    method:
      | "error"
      | "guardianWarning"
      | "turn/diff/updated"
      | "turn/moderationMetadata"
      | "turn/plan/updated";
  }
>;

export type AgentRuntimeNotification =
  | AgentRuntimeLifecycleNotification
  | AgentRuntimeSignalNotification;

export type AgentRuntimeLifecycleEventListener = (
  event: AgentRuntimeLifecycleNotification,
  notification: AgentRuntimeLifecycleNotification,
) => void | Promise<void>;

export type AgentRuntimeSignalEventListener = (
  event: AgentRuntimeSignalNotification,
  notification: AgentRuntimeSignalNotification,
) => void | Promise<void>;

export type AgentRuntimeClientOptions = {
  request?: AppServerRequestOptions;
};

export type AgentRuntimeClientSubscription = {
  unsubscribe(): void;
};

export interface AgentRuntimeClient {
  startTurn(
    params: TurnStartParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<TurnStartResponse>>;
  steerTurn(
    params: TurnSteerParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<TurnSteerResponse>>;
  cancelTurn(
    params: TurnInterruptParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<TurnInterruptResponse>>;
  respondAction(
    params: AgentSessionActionRespondParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<AgentSessionActionRespondResponse>>;
  /** Respond to a typed reverse request using its outer JSON-RPC id. */
  respondServerRequest?<T>(id: RequestId, result: T): void;
  /** Reject a typed reverse request using its outer JSON-RPC id. */
  rejectServerRequest?(id: RequestId, error: JsonRpcError): void;
  readThread(
    params: ThreadReadParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<ThreadReadResponse>>;
  updateThreadSettings(
    params: ThreadSettingsUpdateParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<ThreadSettingsUpdateResponse>>;
  setThreadMemoryMode(
    params: ThreadMemoryModeSetParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<ThreadMemoryModeSetResponse>>;
  readToolInventory(
    params?: AgentSessionToolInventoryReadParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<AgentSessionToolInventoryReadResponse>>;
  subscribeLifecycleEvents(
    listener: AgentRuntimeLifecycleEventListener,
  ): AgentRuntimeClientSubscription;
  subscribeSignalEvents(
    listener: AgentRuntimeSignalEventListener,
  ): AgentRuntimeClientSubscription;
  dispatchEvent(message: JsonRpcMessage): Promise<boolean>;
  nextEvent(timeoutMs?: number): Promise<AgentRuntimeNotification>;
}

export class AppServerAgentEventRouter {
  #lifecycleListeners = new Set<AgentRuntimeLifecycleEventListener>();
  #signalListeners = new Set<AgentRuntimeSignalEventListener>();

  subscribeLifecycle(listener: AgentRuntimeLifecycleEventListener): () => void {
    this.#lifecycleListeners.add(listener);
    return () => {
      this.#lifecycleListeners.delete(listener);
    };
  }

  subscribeSignal(listener: AgentRuntimeSignalEventListener): () => void {
    this.#signalListeners.add(listener);
    return () => {
      this.#signalListeners.delete(listener);
    };
  }

  async dispatch(message: JsonRpcMessage): Promise<boolean> {
    const signal = agentRuntimeSignalNotification(message);
    if (signal) {
      for (const listener of this.#signalListeners) {
        await listener(signal, signal);
      }
      return true;
    }
    const lifecycle = agentRuntimeLifecycleNotification(message);
    if (lifecycle) {
      for (const listener of this.#lifecycleListeners) {
        await listener(lifecycle, lifecycle);
      }
      return true;
    }
    return false;
  }
}

export class AppServerAgentRuntimeClient implements AgentRuntimeClient {
  readonly connection: AppServerConnection;
  readonly eventRouter: AppServerAgentEventRouter;
  readonly defaultRequestOptions: AppServerRequestOptions;

  constructor(
    connection: AppServerConnection,
    options: AgentRuntimeClientOptions = {},
  ) {
    this.connection = connection;
    this.eventRouter = new AppServerAgentEventRouter();
    this.defaultRequestOptions = options.request ?? {};
  }

  async startTurn(
    params: TurnStartParams,
    options: AppServerRequestOptions = {},
  ): Promise<AppServerRequestResult<TurnStartResponse>> {
    return await this.connection.startTurn(
      params,
      mergeRequestOptions(this.defaultRequestOptions, options),
    );
  }

  async steerTurn(
    params: TurnSteerParams,
    options: AppServerRequestOptions = {},
  ): Promise<AppServerRequestResult<TurnSteerResponse>> {
    return await this.connection.steerTurn(
      params,
      mergeRequestOptions(this.defaultRequestOptions, options),
    );
  }

  async cancelTurn(
    params: TurnInterruptParams,
    options: AppServerRequestOptions = {},
  ): Promise<AppServerRequestResult<TurnInterruptResponse>> {
    return await this.connection.cancelTurn(
      params,
      mergeRequestOptions(this.defaultRequestOptions, options),
    );
  }

  async respondAction(
    params: AgentSessionActionRespondParams,
    options: AppServerRequestOptions = {},
  ): Promise<AppServerRequestResult<AgentSessionActionRespondResponse>> {
    return await this.connection.respondAction(
      params,
      mergeRequestOptions(this.defaultRequestOptions, options),
    );
  }

  respondServerRequest<T>(id: RequestId, result: T): void {
    this.connection.respondServerRequest(id, result);
  }

  rejectServerRequest(id: RequestId, error: JsonRpcError): void {
    this.connection.rejectServerRequest(id, error);
  }

  async readThread(
    params: ThreadReadParams,
    options: AppServerRequestOptions = {},
  ): Promise<AppServerRequestResult<ThreadReadResponse>> {
    return await this.connection.readThread(
      params,
      mergeRequestOptions(this.defaultRequestOptions, options),
    );
  }

  async updateThreadSettings(
    params: ThreadSettingsUpdateParams,
    options: AppServerRequestOptions = {},
  ): Promise<AppServerRequestResult<ThreadSettingsUpdateResponse>> {
    return await this.connection.updateThreadSettings(
      params,
      mergeRequestOptions(this.defaultRequestOptions, options),
    );
  }

  async setThreadMemoryMode(
    params: ThreadMemoryModeSetParams,
    options: AppServerRequestOptions = {},
  ): Promise<AppServerRequestResult<ThreadMemoryModeSetResponse>> {
    return await this.connection.setThreadMemoryMode(
      params,
      mergeRequestOptions(this.defaultRequestOptions, options),
    );
  }

  async readToolInventory(
    params: AgentSessionToolInventoryReadParams = {},
    options: AppServerRequestOptions = {},
  ): Promise<AppServerRequestResult<AgentSessionToolInventoryReadResponse>> {
    return await this.connection.readAgentSessionToolInventory(
      params,
      mergeRequestOptions(this.defaultRequestOptions, options),
    );
  }

  subscribeLifecycleEvents(
    listener: AgentRuntimeLifecycleEventListener,
  ): AgentRuntimeClientSubscription {
    const unsubscribe = this.eventRouter.subscribeLifecycle(listener);
    return { unsubscribe };
  }

  subscribeSignalEvents(
    listener: AgentRuntimeSignalEventListener,
  ): AgentRuntimeClientSubscription {
    const unsubscribe = this.eventRouter.subscribeSignal(listener);
    return { unsubscribe };
  }

  async dispatchEvent(message: JsonRpcMessage): Promise<boolean> {
    return await this.eventRouter.dispatch(message);
  }

  async nextEvent(timeoutMs?: number): Promise<AgentRuntimeNotification> {
    for (;;) {
      const notification = await this.connection.nextNotification(timeoutMs);
      const signal = agentRuntimeSignalNotification(notification);
      if (signal) {
        await this.dispatchEvent(signal);
        return signal;
      }
      const lifecycle = agentRuntimeLifecycleNotification(notification);
      if (lifecycle) {
        await this.dispatchEvent(lifecycle);
        return lifecycle;
      }
    }
  }
}

export function agentRuntimeLifecycleNotification(
  message: JsonRpcMessage,
): AgentRuntimeLifecycleNotification | undefined {
  const notification = serverNotification(message);
  return notification && !isAgentRuntimeSignalNotification(notification)
    ? notification
    : undefined;
}

export function agentRuntimeSignalNotification(
  message: JsonRpcMessage,
): AgentRuntimeSignalNotification | undefined {
  const notification = serverNotification(message);
  return notification && isAgentRuntimeSignalNotification(notification)
    ? notification
    : undefined;
}

function isAgentRuntimeSignalNotification(
  notification: ServerNotification,
): notification is AgentRuntimeSignalNotification {
  return (
    notification.method === "error" ||
    notification.method === "turn/diff/updated" ||
    notification.method === "turn/moderationMetadata" ||
    notification.method === "turn/plan/updated"
  );
}

export function createAgentRuntimeClient(
  connection: AppServerConnection,
  options: AgentRuntimeClientOptions = {},
): AgentRuntimeClient {
  return new AppServerAgentRuntimeClient(connection, options);
}

function mergeRequestOptions(
  defaults: AppServerRequestOptions,
  overrides: AppServerRequestOptions,
): AppServerRequestOptions {
  return { ...defaults, ...overrides };
}
