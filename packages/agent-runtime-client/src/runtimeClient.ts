import {
  AppServerAgentRuntimeClient as BaseAppServerAgentRuntimeClient,
  type AgentRuntimeClient,
  type AgentRuntimeClientOptions as BaseAgentRuntimeClientOptions,
  type AgentRuntimeClientSubscription,
  type AgentRuntimeLifecycleEventListener,
  type AgentRuntimeLifecycleNotification,
  type AgentRuntimeNotification,
  type JsonRpcError,
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
  type AppServerConnection,
  type AppServerRequestOptions,
  type AppServerRequestResult,
  type EvidenceExportParams,
  type EvidenceExportResponse,
  type JsonRpcMessage,
  type RequestId,
  type StructuredOutputContract,
  serverNotification,
} from "@limecloud/app-server-client";

export type { StructuredOutputContract };
import {
  AgentRuntimeEventPipeline,
  type AgentRuntimeEventPipelineResult,
  type AgentRuntimeEventAdapter,
  type AgentRuntimeEventPipelineMiddleware,
} from "./eventPipeline.js";
import type {
  AgentRuntimeSequenceVerifierLike,
  AgentRuntimeSequenceVerifierMode,
} from "./eventVerifier.js";

export interface AgentRuntimeClientOptions extends BaseAgentRuntimeClientOptions {
  sequenceVerifier?: AgentRuntimeSequenceVerifierLike;
  sequenceVerifierMode?: AgentRuntimeSequenceVerifierMode;
  adapters?: readonly AgentRuntimeEventAdapter[];
  middlewares?: readonly AgentRuntimeEventPipelineMiddleware[];
}

export class AppServerAgentRuntimeClient implements AgentRuntimeClient {
  readonly connection: AppServerConnection;
  readonly #base: BaseAppServerAgentRuntimeClient;
  readonly #lifecycleListeners = new Set<AgentRuntimeLifecycleEventListener>();
  readonly #eventPipeline: AgentRuntimeEventPipeline;
  readonly #pendingNextEvents: AgentRuntimeNotification[] = [];

  constructor(
    connection: AppServerConnection,
    options: AgentRuntimeClientOptions = {},
  ) {
    this.connection = connection;
    this.#base = new BaseAppServerAgentRuntimeClient(connection, {
      request: options.request,
    });
    this.#eventPipeline = new AgentRuntimeEventPipeline({
      sequenceVerifier: options.sequenceVerifier,
      sequenceVerifierMode: options.sequenceVerifierMode,
      adapters: options.adapters,
      middlewares: options.middlewares,
    });
  }

  async startTurn(
    params: TurnStartParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<TurnStartResponse>> {
    return await this.#base.startTurn(params, options);
  }

  async steerTurn(
    params: TurnSteerParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<TurnSteerResponse>> {
    return await this.#base.steerTurn(params, options);
  }

  async cancelTurn(
    params: TurnInterruptParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<TurnInterruptResponse>> {
    return await this.#base.cancelTurn(params, options);
  }

  async respondAction(
    params: AgentSessionActionRespondParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<AgentSessionActionRespondResponse>> {
    return await this.#base.respondAction(params, options);
  }

  respondServerRequest<T>(id: RequestId, result: T): void {
    this.#base.respondServerRequest(id, result);
  }

  rejectServerRequest(id: RequestId, error: JsonRpcError): void {
    this.#base.rejectServerRequest(id, error);
  }

  async readThread(
    params: ThreadReadParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<ThreadReadResponse>> {
    return await this.#base.readThread(params, options);
  }

  async updateThreadSettings(
    params: ThreadSettingsUpdateParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<ThreadSettingsUpdateResponse>> {
    return await this.#base.updateThreadSettings(params, options);
  }

  async setThreadMemoryMode(
    params: ThreadMemoryModeSetParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<ThreadMemoryModeSetResponse>> {
    return await this.#base.setThreadMemoryMode(params, options);
  }

  async readToolInventory(
    params: AgentSessionToolInventoryReadParams = {},
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<AgentSessionToolInventoryReadResponse>> {
    return await this.#base.readToolInventory(params, options);
  }

  async exportEvidence(
    params: EvidenceExportParams,
    options?: AppServerRequestOptions,
  ): Promise<AppServerRequestResult<EvidenceExportResponse>> {
    return await this.#base.exportEvidence(params, options);
  }

  subscribeLifecycleEvents(
    listener: AgentRuntimeLifecycleEventListener,
  ): AgentRuntimeClientSubscription {
    this.#lifecycleListeners.add(listener);
    return {
      unsubscribe: () => {
        this.#lifecycleListeners.delete(listener);
      },
    };
  }

  async dispatchEvent(message: JsonRpcMessage): Promise<boolean> {
    const lifecycle = serverNotification(message);
    if (lifecycle) {
      const result = await this.#dispatchLifecycle(lifecycle);
      return result.accepted;
    }
    return false;
  }

  async #dispatchLifecycle(
    notification: AgentRuntimeLifecycleNotification,
  ): Promise<AgentRuntimeEventPipelineResult> {
    const pipelineResult = await this.#eventPipeline.process(notification);
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

  async nextEvent(timeoutMs?: number): Promise<AgentRuntimeNotification> {
    for (;;) {
      const pending = this.#pendingNextEvents.shift();
      if (pending) {
        return pending;
      }
      const notification = await this.connection.nextNotification(timeoutMs);
      const lifecycle = serverNotification(notification);
      if (lifecycle) {
        const result = await this.#dispatchLifecycle(lifecycle);
        if (result.accepted) {
          const [next, ...rest] = result.notifications;
          this.#pendingNextEvents.push(...rest);
          return next;
        }
        if (result.reason === "dropped") {
          continue;
        }
        if (result.reason === "sequence_violation") {
          throw this.#eventPipeline.sequenceViolationError();
        }
      }
    }
  }
}

export function createAgentRuntimeClient(
  connection: AppServerConnection,
  options: AgentRuntimeClientOptions = {},
): AgentRuntimeClient {
  return new AppServerAgentRuntimeClient(connection, options);
}
