import {
  agentSessionEventNotification,
  mediaReadEventNotification,
  serverNotification,
  AgentRuntimeClient,
  AgentRuntimeClientSubscription,
  AgentRuntimeEventListener,
  AgentRuntimeLifecycleEventListener,
  AgentRuntimeNotification,
  AgentSessionActionRespondParams,
  AgentSessionActionRespondResponse,
  AgentSessionEventNotification,
  ThreadReadParams,
  ThreadReadResponse,
  ThreadMemoryModeSetParams,
  ThreadMemoryModeSetResponse,
  ThreadSettingsUpdateParams,
  ThreadSettingsUpdateResponse,
  AgentSessionToolInventoryReadParams,
  AgentSessionToolInventoryReadResponse,
  TurnInterruptParams,
  TurnInterruptResponse,
  TurnStartParams,
  TurnStartResponse,
  TurnSteerParams,
  TurnSteerResponse,
  AppServerRequestOptions,
  AppServerRequestResult,
  EvidenceExportParams,
  EvidenceExportResponse,
  JsonRpcMessage,
  JsonRpcError,
  RequestId,
} from "@limecloud/app-server-client/browser";
import {
  AgentRuntimeEventPipeline,
  type AgentRuntimeEventAdapter,
  type AgentRuntimeEventPipelineResult,
  type AgentRuntimeEventPipelineMiddleware,
} from "./eventPipeline.js";
import {
  type AgentRuntimeSequenceViolationError,
  type AgentRuntimeSequenceVerifierLike,
  type AgentRuntimeSequenceVerifierMode,
} from "./eventVerifier.js";

export {
  AgentRuntimeEventSequenceGate,
  AgentRuntimeSequenceViolationError,
  runtimeExecutionEventFromLifecycleNotification,
  type AgentRuntimeSequenceVerifierLike,
  type AgentRuntimeSequenceVerifierMode,
} from "./eventVerifier.js";
export {
  AgentRuntimeEventPipeline,
  type AgentRuntimeEventAdapter,
  type AgentRuntimeEventMiddleware,
  type AgentRuntimeEventMiddlewareFunction,
  type AgentRuntimeEventPipelineContext,
  type AgentRuntimeEventPipelineMiddleware,
  type AgentRuntimeEventPipelineOptions,
} from "./eventPipeline.js";

export type AgentRuntimeLifecycleClient = Pick<
  AgentRuntimeClient,
  "startTurn" | "steerTurn" | "readThread" | "cancelTurn" | "respondAction"
>;

type AgentRuntimeGatewayMethod<Params, Result> = (
  params: Params,
  options?: AppServerRequestOptions,
) => Promise<AppServerRequestResult<Result>>;

export type AgentRuntimeSessionGateway = {
  startTurn: AgentRuntimeGatewayMethod<TurnStartParams, TurnStartResponse>;
  steerTurn: AgentRuntimeGatewayMethod<TurnSteerParams, TurnSteerResponse>;
  readThread: AgentRuntimeGatewayMethod<ThreadReadParams, ThreadReadResponse>;
  updateThreadSettings: AgentRuntimeGatewayMethod<
    ThreadSettingsUpdateParams,
    ThreadSettingsUpdateResponse
  >;
  setThreadMemoryMode: AgentRuntimeGatewayMethod<
    ThreadMemoryModeSetParams,
    ThreadMemoryModeSetResponse
  >;
  readToolInventory?: AgentRuntimeGatewayMethod<
    AgentSessionToolInventoryReadParams,
    AgentSessionToolInventoryReadResponse
  >;
  cancelTurn: AgentRuntimeGatewayMethod<
    TurnInterruptParams,
    TurnInterruptResponse
  >;
  respondAction: AgentRuntimeGatewayMethod<
    AgentSessionActionRespondParams,
    AgentSessionActionRespondResponse
  >;
  respondServerRequest?<T>(id: RequestId, result: T): void;
  rejectServerRequest?(id: RequestId, error: JsonRpcError): void;
  exportEvidence?: AgentRuntimeGatewayMethod<
    EvidenceExportParams,
    EvidenceExportResponse
  >;
  nextEvent?(timeoutMs?: number): Promise<AgentRuntimeNotification>;
  drainEvents?(limit?: number): Promise<JsonRpcMessage[]>;
};

export interface AgentRuntimeClientFromGatewayOptions {
  sequenceVerifier?: AgentRuntimeSequenceVerifierLike;
  sequenceVerifierMode?: AgentRuntimeSequenceVerifierMode;
  adapters?: readonly AgentRuntimeEventAdapter[];
  middlewares?: readonly AgentRuntimeEventPipelineMiddleware[];
}

export function createAgentRuntimeClientFromSessionGateway(
  gateway: AgentRuntimeSessionGateway,
  options: AgentRuntimeClientFromGatewayOptions = {},
): AgentRuntimeClient {
  const eventRouter = new AgentRuntimeGatewayEventRouter(options);
  return {
    startTurn: (params, options) =>
      callAgentRuntimeSessionGateway(gateway.startTurn, params, options),
    steerTurn: (params, options) =>
      callAgentRuntimeSessionGateway(gateway.steerTurn, params, options),
    readThread: (params, options) =>
      callAgentRuntimeSessionGateway(gateway.readThread, params, options),
    updateThreadSettings: (params, options) =>
      callAgentRuntimeSessionGateway(
        gateway.updateThreadSettings,
        params,
        options,
      ),
    setThreadMemoryMode: (params, options) =>
      callAgentRuntimeSessionGateway(
        gateway.setThreadMemoryMode,
        params,
        options,
      ),
    readToolInventory: (params = {}, options) =>
      callOptionalAgentRuntimeSessionGateway(
        gateway.readToolInventory,
        "readToolInventory",
        params,
        options,
      ),
    cancelTurn: (params, options) =>
      callAgentRuntimeSessionGateway(gateway.cancelTurn, params, options),
    respondAction: (params, options) =>
      callAgentRuntimeSessionGateway(gateway.respondAction, params, options),
    respondServerRequest: <T>(id: RequestId, result: T) => {
      if (!gateway.respondServerRequest) {
        throw new Error(
          "AgentRuntime session gateway does not expose respondServerRequest.",
        );
      }
      gateway.respondServerRequest(id, result);
    },
    rejectServerRequest: (id: RequestId, error: JsonRpcError) => {
      if (!gateway.rejectServerRequest) {
        throw new Error(
          "AgentRuntime session gateway does not expose rejectServerRequest.",
        );
      }
      gateway.rejectServerRequest(id, error);
    },
    exportEvidence: (params, options) =>
      callOptionalAgentRuntimeSessionGateway(
        gateway.exportEvidence,
        "exportEvidence",
        params,
        options,
      ),
    subscribeEvents(listener) {
      return eventRouter.subscribe(listener);
    },
    subscribeLifecycleEvents(listener) {
      return eventRouter.subscribeLifecycle(listener);
    },
    async dispatchEvent(message) {
      const result = await eventRouter.dispatch(message);
      return result.accepted;
    },
    async nextEvent(timeoutMs) {
      const pending = eventRouter.takePendingNextEvent();
      if (pending) {
        return pending;
      }
      if (gateway.nextEvent) {
        for (;;) {
          const notification = await gateway.nextEvent(timeoutMs);
          const result = await eventRouter.dispatch(notification);
          if (result.accepted) {
            const [next, ...rest] = result.notifications;
            eventRouter.queuePendingNextEvents(rest);
            return next;
          }
          if (result.reason === "sequence_violation") {
            throw eventRouter.sequenceViolationError();
          }
        }
      }
      if (gateway.drainEvents) {
        return await nextDrainedAgentRuntimeEvent(
          gateway,
          eventRouter,
          timeoutMs,
        );
      }
      throw new Error(
        "AgentRuntime session gateway does not expose direct lifecycle notifications.",
      );
    },
  };
}

function callAgentRuntimeSessionGateway<Params, Result>(
  method: (
    params: Params,
    options?: AppServerRequestOptions,
  ) => Promise<AppServerRequestResult<Result>>,
  params: Params,
  options?: AppServerRequestOptions,
): Promise<AppServerRequestResult<Result>> {
  if (options === undefined) {
    return method(params);
  }
  return method(params, options);
}

async function callOptionalAgentRuntimeSessionGateway<Params, Result>(
  method: AgentRuntimeGatewayMethod<Params, Result> | undefined,
  methodName: string,
  params: Params,
  options?: AppServerRequestOptions,
): Promise<AppServerRequestResult<Result>> {
  if (!method) {
    throw new Error(
      `AgentRuntime session gateway does not expose ${methodName}.`,
    );
  }
  return callAgentRuntimeSessionGateway(method, params, options);
}

async function nextDrainedAgentRuntimeEvent(
  gateway: AgentRuntimeSessionGateway,
  eventRouter: AgentRuntimeGatewayEventRouter,
  timeoutMs?: number,
): Promise<AgentRuntimeNotification> {
  const pending = eventRouter.takePendingNextEvent();
  if (pending) {
    return pending;
  }
  const messages = await gateway.drainEvents?.(1);
  for (const message of messages ?? []) {
    const result = await eventRouter.dispatch(message);
    if (result.accepted) {
      const [next, ...rest] = result.notifications;
      eventRouter.queuePendingNextEvents(rest);
      return next;
    }
    if (result.reason === "sequence_violation") {
      throw eventRouter.sequenceViolationError();
    }
  }
  throw new Error(
    timeoutMs === undefined
      ? "AgentRuntime session gateway drainEvents did not return a direct lifecycle notification."
      : `AgentRuntime session gateway drainEvents did not return a direct lifecycle notification within ${timeoutMs}ms.`,
  );
}

class AgentRuntimeGatewayEventRouter {
  readonly #listeners = new Set<AgentRuntimeEventListener>();
  readonly #lifecycleListeners = new Set<AgentRuntimeLifecycleEventListener>();
  readonly #eventPipeline: AgentRuntimeEventPipeline;
  readonly #pendingNextEvents: AgentRuntimeNotification[] = [];

  constructor(options: AgentRuntimeClientFromGatewayOptions = {}) {
    this.#eventPipeline = new AgentRuntimeEventPipeline({
      sequenceVerifier: options.sequenceVerifier,
      sequenceVerifierMode: options.sequenceVerifierMode,
      adapters: options.adapters,
      middlewares: options.middlewares,
    });
  }

  subscribe(
    listener: AgentRuntimeEventListener,
  ): AgentRuntimeClientSubscription {
    this.#listeners.add(listener);
    return {
      unsubscribe: () => {
        this.#listeners.delete(listener);
      },
    };
  }

  subscribeLifecycle(
    listener: AgentRuntimeLifecycleEventListener,
  ): AgentRuntimeClientSubscription {
    this.#lifecycleListeners.add(listener);
    return {
      unsubscribe: () => {
        this.#lifecycleListeners.delete(listener);
      },
    };
  }

  async dispatch(
    message: JsonRpcMessage,
  ): Promise<AgentRuntimeGatewayDispatchResult> {
    const lifecycle = serverNotification(message);
    if (!lifecycle) {
      const notification = agentSessionEventNotification(message);
      if (!notification || !mediaReadEventNotification(notification)) {
        return { accepted: false, reason: "dropped" };
      }
      for (const listener of this.#listeners) {
        await listener(notification.params.event, notification);
      }
      return {
        accepted: true,
        notification,
        notifications: [notification],
      };
    }
    const pipelineResult = await this.#eventPipeline.process(lifecycle);
    if (!pipelineResult.accepted) {
      return pipelineResult;
    }
    for (const notification of pipelineResult.notifications) {
      for (const listener of this.#lifecycleListeners) {
        await listener(notification, notification);
      }
    }
    return pipelineResult;
  }

  takePendingNextEvent(): AgentRuntimeNotification | undefined {
    return this.#pendingNextEvents.shift();
  }

  queuePendingNextEvents(
    notifications: readonly AgentRuntimeNotification[],
  ): void {
    this.#pendingNextEvents.push(...notifications);
  }

  sequenceViolationError(): AgentRuntimeSequenceViolationError {
    return this.#eventPipeline.sequenceViolationError();
  }
}

type AgentRuntimeGatewayDispatchResult =
  | AgentRuntimeEventPipelineResult
  | {
      accepted: true;
      notification: AgentSessionEventNotification;
      notifications: AgentSessionEventNotification[];
    };
