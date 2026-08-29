import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import {
  parseAgentEvent,
  type AgentEvent,
  type AgentThreadItem,
  type AgentThreadTurn,
} from "@/lib/api/agentProtocol";
import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntime,
} from "@/lib/api/agentExecutionRuntime";
import type {
  AgentTodoItem,
  AutoContinueRequestPayload,
} from "@/lib/api/agentRuntime/sessionTypes";
import { logAgentDebug } from "@/lib/agentDebug";
import type { ActionRequired, Message } from "../types";
import { handleTurnStreamEvent } from "./agentStreamRuntimeHandler";
import {
  AGENT_STREAM_FIRST_EVENT_TIMEOUT_MESSAGE,
  AGENT_STREAM_INACTIVITY_TIMEOUT_MESSAGE,
  buildAgentStreamFirstEventDeferredWarning,
  buildAgentStreamFirstEventSilentRecoveryWarning,
  buildAgentStreamInactivitySilentRecoveryWarning,
  resolveAgentStreamFirstEventTimeoutAction,
  resolveAgentStreamInactivityTimeoutAction,
} from "./agentStreamInactivityController";
import {
  buildAgentStreamFirstEventContext,
  buildAgentStreamFirstEventDeferredContext,
  buildAgentStreamListenerBoundContext,
  extractAgentStreamRuntimeEventSchemaVersion,
  extractAgentStreamRuntimeEventType,
  shouldDeferAgentStreamFirstEventTimeout,
  shouldIgnoreAgentStreamInactivityResult,
  shouldScheduleAgentStreamInactivityWatchdog,
} from "./agentStreamListenerReadinessController";
import { startAgentStreamRequest } from "./agentStreamRequestStartController";
import { buildAgentStreamProviderTraceMetricContext } from "./agentStreamRuntimeMetricsController";
import {
  rememberAgentStreamUnknownEventWarning,
  resolveAgentStreamUnknownEventPlan,
} from "./agentStreamUnknownEventController";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import type { StreamRequestState } from "./agentStreamSubmissionLifecycle";
import {
  recordAgentStreamPerformanceMetric,
  type AgentUiPerformanceTraceMetadata,
} from "./agentStreamPerformanceMetrics";
import type { SoulInteractionCopy } from "@/lib/soul/interactionCopy";
import { isAgentMessageFinalAnswerPhase } from "../utils/agentMessagePhase";

type MessageParts = NonNullable<Message["contentParts"]>;
const STREAM_FIRST_EVENT_TIMEOUT_MS = 12_000;
const STREAM_INACTIVITY_TIMEOUT_MS = 120_000; // 2 分钟，兼容推理模型长时间思考
const STREAM_DEFERRED_RECOVERY_POLL_MS = 5_000;
const STREAM_DEFERRED_RECOVERY_MAX_DURATION_MS = 120_000;

function normalizeEventNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parseEventTimestampMs(timestamp: string | undefined): number | null {
  if (!timestamp) {
    return null;
  }
  const parsed = Date.parse(timestamp);
  return Number.isFinite(parsed) ? parsed : null;
}

function buildAgentEventPerformanceTrace(params: {
  activeSessionId: string;
  event: AgentEvent;
  eventReceivedAt: number;
  previousTrace?: AgentUiPerformanceTraceMetadata | null;
  resolvedWorkspaceId: string;
}): AgentUiPerformanceTraceMetadata | null {
  const previousTrace = params.previousTrace ?? null;
  if (!previousTrace && !params.event.trace_id && !params.event.run_id) {
    return null;
  }

  return {
    ...previousTrace,
    requestId: params.event.request_id ?? previousTrace?.requestId ?? null,
    runId: params.event.run_id ?? previousTrace?.runId ?? null,
    sessionId:
      params.event.session_id ??
      previousTrace?.sessionId ??
      params.activeSessionId,
    traceId: params.event.trace_id ?? previousTrace?.traceId ?? null,
    turnId: params.event.turn_id ?? previousTrace?.turnId ?? null,
    workspaceId: previousTrace?.workspaceId ?? params.resolvedWorkspaceId,
    serverEventEmittedAt:
      normalizeEventNumber(params.event.server_event_emitted_at) ??
      parseEventTimestampMs(params.event.timestamp) ??
      previousTrace?.serverEventEmittedAt ??
      null,
    serverEventId:
      params.event.event_id ?? previousTrace?.serverEventId ?? null,
    serverEventSequence:
      normalizeEventNumber(params.event.sequence) ??
      previousTrace?.serverEventSequence ??
      null,
    serverEventType:
      params.event.type ?? previousTrace?.serverEventType ?? null,
    rendererEventReceivedAt:
      normalizeEventNumber(params.event.renderer_event_received_at) ??
      params.eventReceivedAt,
    providerWaitMs:
      params.event.type === "provider_trace" &&
      params.event.stage === "first_text_delta_received"
        ? (normalizeEventNumber(params.event.elapsed_ms) ??
          previousTrace?.providerWaitMs ??
          null)
        : (previousTrace?.providerWaitMs ?? null),
  };
}

function isFinalAgentMessageSnapshotEvent(event: AgentEvent): boolean {
  return (
    (event.type === "item_started" ||
      event.type === "item_updated" ||
      event.type === "item_completed") &&
    event.item.type === "agent_message" &&
    isAgentMessageFinalAnswerPhase(event.item.phase) &&
    Boolean(event.item.text.trim())
  );
}

interface StreamObserver {
  onTextDelta?: (delta: string, accumulated: string) => void;
  onComplete?: (content: string) => void;
  onError?: (message: string) => void;
}

interface RegisterAgentStreamTurnEventBindingOptions {
  runtime: AgentRuntimeAdapter;
  eventName: string;
  requestState: StreamRequestState;
  attemptSilentTurnRecovery?: (
    sessionId: string,
    requestStartedAt: number,
    promptText: string,
    options?: { requireTerminal?: boolean; turnId?: string | null },
  ) => Promise<boolean>;
  skipUserMessage: boolean;
  effectiveProviderType: string;
  effectiveModel: string;
  effectiveExecutionStrategy: AgentExecutionStrategy;
  systemPrompt?: string;
  content: string;
  webSearch?: boolean;
  autoContinue?: AutoContinueRequestPayload;
  activeSessionId: string;
  resolvedWorkspaceId: string;
  assistantMsgId: string;
  pendingTurnKey: string;
  pendingItemKey: string;
  effectiveWaitingRuntimeStatus: NonNullable<Message["runtimeStatus"]>;
  preserveAssistantContent?: string | null;
  assistantFallbackContent?: string | null;
  warnedKeysRef: MutableRefObject<Set<string>>;
  actionLoggedKeys: Set<string>;
  toolLogIdByToolId: Map<string, string>;
  toolStartedAtByToolId: Map<string, number>;
  toolNameByToolId: Map<string, string>;
  observer?: StreamObserver;
  onWriteFile?: (
    content: string,
    fileName: string,
    context?: import("../types").WriteArtifactContext,
  ) => void;
  callbacks: {
    activateStream: (
      activeSessionId: string,
      effectiveWaitingRuntimeStatus: NonNullable<Message["runtimeStatus"]>,
    ) => void;
    isStreamActivated: () => boolean;
    clearOptimisticItem: () => void;
    clearOptimisticTurn: () => void;
    disposeListener: () => void;
    clearActiveStreamIfMatch: (eventName: string) => boolean;
  };
  appendThinkingToParts: (
    parts: MessageParts,
    textDelta: string,
  ) => MessageParts;
  setMessages: Dispatch<SetStateAction<Message[]>>;
  setPendingActions: Dispatch<SetStateAction<ActionRequired[]>>;
  getThreadItems?: () => readonly AgentThreadItem[];
  setThreadItems: Dispatch<SetStateAction<AgentThreadItem[]>>;
  setTodoItems?: Dispatch<SetStateAction<AgentTodoItem[]>>;
  setThreadTurns: Dispatch<SetStateAction<AgentThreadTurn[]>>;
  setCurrentTurnId: Dispatch<SetStateAction<string | null>>;
  setExecutionRuntime: Dispatch<
    SetStateAction<AgentSessionExecutionRuntime | null>
  >;
  setIsSending: Dispatch<SetStateAction<boolean>>;
  soulCopy?: SoulInteractionCopy;
}

export async function registerAgentStreamTurnEventBinding(
  options: RegisterAgentStreamTurnEventBindingOptions,
) {
  const {
    runtime,
    eventName,
    requestState,
    attemptSilentTurnRecovery,
    skipUserMessage,
    effectiveProviderType,
    effectiveModel,
    effectiveExecutionStrategy,
    systemPrompt,
    content,
    webSearch,
    autoContinue,
    activeSessionId,
    resolvedWorkspaceId,
    assistantMsgId,
    pendingTurnKey,
    pendingItemKey,
    effectiveWaitingRuntimeStatus,
    preserveAssistantContent,
    assistantFallbackContent,
    warnedKeysRef,
    actionLoggedKeys,
    toolLogIdByToolId,
    toolStartedAtByToolId,
    toolNameByToolId,
    observer,
    onWriteFile,
    callbacks,
    appendThinkingToParts,
    setMessages,
    setPendingActions,
    getThreadItems,
    setThreadItems,
    setTodoItems,
    setThreadTurns,
    setCurrentTurnId,
    setExecutionRuntime,
    setIsSending,
    soulCopy,
  } = options;

  startAgentStreamRequest({
    activeSessionId,
    autoContinue,
    content,
    effectiveExecutionStrategy,
    effectiveModel,
    effectiveProviderType,
    eventName,
    requestState,
    resolvedWorkspaceId,
    skipUserMessage,
    systemPrompt,
  });

  let firstEventReceived = false;
  let lastEventReceivedAt = 0;
  let lastRuntimeEventType: string | null = null;
  const warnedUnknownEventTypes = new Set<string>();
  const surfaceThinkingDeltas = false;
  let terminalRecoveryPollStarted = false;
  let terminalRecoveryPollStartedAt: number | null = null;
  let terminalRecoveryPollAttempts = 0;
  let recoveryPollGeneration = 0;
  let disposed = false;
  const markFirstEventReceived = (params: {
    eventReceivedAt: number;
    eventType: string;
    recognized: boolean;
  }) => {
    if (firstEventReceived) {
      return;
    }

    firstEventReceived = true;
    requestState.firstEventReceivedAt = params.eventReceivedAt;
    const firstEventContext = buildAgentStreamFirstEventContext({
      activeSessionId,
      eventName,
      eventReceivedAt: params.eventReceivedAt,
      eventType: params.eventType,
      recognized: params.recognized,
      requestStartedAt: requestState.requestStartedAt,
      submissionDispatchedAt: requestState.submissionDispatchedAt,
    });
    recordAgentStreamPerformanceMetric(
      "agentStream.firstEvent",
      requestState.performanceTrace,
      firstEventContext,
    );
    logAgentDebug("AgentStream", "firstEvent", firstEventContext);
    clearFirstEventWatchdog();
  };
  let inactivityWatchdogId: ReturnType<typeof setTimeout> | null = null;
  const clearInactivityWatchdog = () => {
    if (inactivityWatchdogId) {
      clearTimeout(inactivityWatchdogId);
      inactivityWatchdogId = null;
    }
  };
  let deferredRecoveryPollId: ReturnType<typeof setTimeout> | null = null;
  const clearDeferredRecoveryPoll = () => {
    if (deferredRecoveryPollId) {
      clearTimeout(deferredRecoveryPollId);
      deferredRecoveryPollId = null;
    }
    recoveryPollGeneration += 1;
  };
  const readRecoveryTurnId = () =>
    requestState.activeTextSegmentTurnId ?? requestState.currentTurnId ?? null;
  function scheduleDeferredRecoveryPoll() {
    clearDeferredRecoveryPoll();
    if (
      disposed ||
      requestState.requestFinished ||
      !terminalRecoveryPollStarted
    ) {
      return;
    }
    const generation = recoveryPollGeneration;
    logAgentDebug(
      "AgentStream",
      "terminalRecoveryPoll.scheduled",
      {
        eventName,
        requireTerminal: terminalRecoveryPollStarted,
        sessionId: activeSessionId,
        turnId: readRecoveryTurnId(),
      },
      {
        dedupeKey: `${eventName}:terminalRecoveryPoll.scheduled`,
        throttleMs: 1000,
      },
    );
    deferredRecoveryPollId = globalThis.setTimeout(() => {
      deferredRecoveryPollId = null;
      if (
        disposed ||
        requestState.requestFinished ||
        generation !== recoveryPollGeneration
      ) {
        return;
      }
      terminalRecoveryPollAttempts += 1;
      void (async () => {
        logAgentDebug(
          "AgentStream",
          "terminalRecoveryPoll.attempt",
          {
            eventName,
            requireTerminal: terminalRecoveryPollStarted,
            sessionId: activeSessionId,
            turnId: readRecoveryTurnId(),
          },
          {
            dedupeKey: `${eventName}:terminalRecoveryPoll.attempt`,
            throttleMs: 1000,
          },
        );
        const recovered = await tryRecoverSilentTurn({
          requireTerminal: terminalRecoveryPollStarted,
        });
        if (
          disposed ||
          requestState.requestFinished ||
          generation !== recoveryPollGeneration
        ) {
          return;
        }
        if (recovered) {
          logAgentDebug("AgentStream", "terminalRecoveryPoll.recovered", {
            eventName,
            sessionId: activeSessionId,
            turnId: readRecoveryTurnId(),
          });
          console.warn(
            buildAgentStreamFirstEventSilentRecoveryWarning({ eventName }),
          );
          finalizeSilentTurnRecovery();
          return;
        }
        const startedAt = terminalRecoveryPollStartedAt ?? Date.now();
        if (
          Date.now() - startedAt >=
          STREAM_DEFERRED_RECOVERY_MAX_DURATION_MS
        ) {
          logAgentDebug("AgentStream", "terminalRecoveryPoll.exhausted", {
            attempts: terminalRecoveryPollAttempts,
            eventName,
            sessionId: activeSessionId,
            turnId: readRecoveryTurnId(),
          });
          return;
        }
        scheduleDeferredRecoveryPoll();
      })();
    }, STREAM_DEFERRED_RECOVERY_POLL_MS);
  }
  const startTerminalRecoveryPoll = () => {
    if (disposed || requestState.requestFinished) {
      return;
    }
    if (!terminalRecoveryPollStarted) {
      terminalRecoveryPollStarted = true;
      terminalRecoveryPollStartedAt = Date.now();
      terminalRecoveryPollAttempts = 0;
    }
    scheduleDeferredRecoveryPoll();
  };
  function deferFirstEventTimeoutAfterSubmission() {
    if (
      !shouldDeferAgentStreamFirstEventTimeout({
        firstEventReceived,
        requestFinished: requestState.requestFinished,
        submissionDispatchedAt: requestState.submissionDispatchedAt,
      })
    ) {
      return false;
    }

    firstEventReceived = true;
    lastEventReceivedAt = Date.now();
    lastRuntimeEventType = "deferred_submission";
    const deferredContext = buildAgentStreamFirstEventDeferredContext({
      activeSessionId,
      deferredAt: lastEventReceivedAt,
      eventName,
      requestStartedAt: requestState.requestStartedAt,
      submissionDispatchedAt: requestState.submissionDispatchedAt,
    });
    recordAgentStreamPerformanceMetric(
      "agentStream.firstEventDeferred",
      requestState.performanceTrace,
      deferredContext,
    );
    callbacks.activateStream(activeSessionId, effectiveWaitingRuntimeStatus);
    scheduleInactivityWatchdog();
    startTerminalRecoveryPoll();
    return true;
  }

  let firstEventWatchdogId: ReturnType<typeof setTimeout> | null =
    globalThis.setTimeout(() => {
      firstEventWatchdogId = null;
      if (firstEventReceived || requestState.requestFinished) {
        return;
      }
      void (async () => {
        const recovered = await tryRecoverSilentTurn();
        const timeoutAction = resolveAgentStreamFirstEventTimeoutAction({
          canDeferAfterSubmission: shouldDeferAgentStreamFirstEventTimeout({
            firstEventReceived,
            requestFinished: requestState.requestFinished,
            submissionDispatchedAt: requestState.submissionDispatchedAt,
          }),
          firstEventReceived,
          recovered,
          requestFinished: requestState.requestFinished,
        });
        switch (timeoutAction) {
          case "ignore":
            return;
          case "recover":
            console.warn(
              buildAgentStreamFirstEventSilentRecoveryWarning({ eventName }),
            );
            finalizeSilentTurnRecovery();
            return;
          case "defer":
            if (deferFirstEventTimeoutAfterSubmission()) {
              console.warn(
                buildAgentStreamFirstEventDeferredWarning({ eventName }),
              );
            }
            return;
          case "fail":
            firstEventReceived = true;
            dispatchSyntheticError(AGENT_STREAM_FIRST_EVENT_TIMEOUT_MESSAGE);
            return;
        }
      })();
    }, STREAM_FIRST_EVENT_TIMEOUT_MS);

  const clearFirstEventWatchdog = () => {
    if (firstEventWatchdogId) {
      clearTimeout(firstEventWatchdogId);
      firstEventWatchdogId = null;
    }
  };
  const disposeListenerWithWatchdogs = () => {
    disposed = true;
    clearFirstEventWatchdog();
    clearInactivityWatchdog();
    clearDeferredRecoveryPoll();
    callbacks.disposeListener();
  };
  const finalizeSilentTurnRecovery = () => {
    firstEventReceived = true;
    const activeStreamCleared =
      callbacks.clearActiveStreamIfMatch(eventName);
    disposeListenerWithWatchdogs();
    if (activeStreamCleared) {
      setIsSending(false);
    }
  };
  const tryRecoverSilentTurn = async (recoveryOptions?: {
    requireTerminal?: boolean;
  }) => {
    if (!attemptSilentTurnRecovery) {
      return false;
    }
    return await attemptSilentTurnRecovery(
      activeSessionId,
      requestState.requestStartedAt,
      content,
      {
        requireTerminal: recoveryOptions?.requireTerminal === true,
        turnId: readRecoveryTurnId(),
      },
    );
  };
  requestState.startTerminalRecoveryPoll = startTerminalRecoveryPoll;
  const dispatchSyntheticEvent = (data: AgentEvent) => {
    handleTurnStreamEvent({
      data,
      requestState,
      callbacks: {
        activateStream: () =>
          callbacks.activateStream(
            activeSessionId,
            effectiveWaitingRuntimeStatus,
          ),
        isStreamActivated: callbacks.isStreamActivated,
        clearOptimisticItem: callbacks.clearOptimisticItem,
        clearOptimisticTurn: callbacks.clearOptimisticTurn,
        disposeListener: disposeListenerWithWatchdogs,
        clearActiveStreamIfMatch: callbacks.clearActiveStreamIfMatch,
        appendThinkingToParts,
      },
      observer,
      eventName,
      pendingTurnKey,
      pendingItemKey,
      assistantMsgId,
      activeSessionId,
      resolvedWorkspaceId,
      effectiveExecutionStrategy,
      surfaceThinkingDeltas,
      preserveAssistantContent,
      assistantFallbackContent,
      content,
      runtime,
      _webSearch: webSearch,
      warnedKeysRef,
      actionLoggedKeys,
      toolLogIdByToolId,
      toolStartedAtByToolId,
      toolNameByToolId,
      onWriteFile,
      setMessages,
      setPendingActions,
      getThreadItems,
      setThreadItems,
      setTodoItems,
      setThreadTurns,
      setCurrentTurnId,
      setExecutionRuntime,
      setIsSending,
      soulCopy,
    });
  };
  const dispatchSyntheticError = (message: string) => {
    dispatchSyntheticEvent({
      type: "error",
      message,
    } as AgentEvent);
  };
  const scheduleInactivityWatchdog = () => {
    clearInactivityWatchdog();
    if (
      !shouldScheduleAgentStreamInactivityWatchdog({
        firstEventReceived,
        requestFinished: requestState.requestFinished,
        streamActivated: callbacks.isStreamActivated(),
      })
    ) {
      return;
    }

    inactivityWatchdogId = globalThis.setTimeout(() => {
      inactivityWatchdogId = null;
      if (requestState.requestFinished || !callbacks.isStreamActivated()) {
        return;
      }
      const timeoutStartedAt = Date.now();
      logAgentDebug(
        "AgentStream",
        "inactivityWatchdog.timeout",
        {
          eventName,
          sessionId: activeSessionId,
          elapsedSinceLastEventMs: lastEventReceivedAt
            ? timeoutStartedAt - lastEventReceivedAt
            : null,
          lastRuntimeEventType,
          requestElapsedMs: timeoutStartedAt - requestState.requestStartedAt,
        },
        { level: "warn" },
      );
      void (async () => {
        const recovered = await tryRecoverSilentTurn({
          requireTerminal: true,
        });
        const activeReadModelActivity = recovered
          ? false
          : await tryRecoverSilentTurn({
              requireTerminal: false,
            });
        const timeoutAction = resolveAgentStreamInactivityTimeoutAction({
          activeReadModelActivity,
          recovered,
          shouldIgnore: shouldIgnoreAgentStreamInactivityResult({
            lastEventReceivedAt,
            requestFinished: requestState.requestFinished,
            streamActivated: callbacks.isStreamActivated(),
            timeoutStartedAt,
          }),
        });
        switch (timeoutAction) {
          case "ignore":
            logAgentDebug("AgentStream", "inactivityWatchdog.ignored", {
              eventName,
              sessionId: activeSessionId,
              lastRuntimeEventType,
            });
            return;
          case "recover":
            console.warn(
              buildAgentStreamInactivitySilentRecoveryWarning({ eventName }),
            );
            logAgentDebug("AgentStream", "inactivityWatchdog.recovered", {
              eventName,
              sessionId: activeSessionId,
              lastRuntimeEventType,
            });
            finalizeSilentTurnRecovery();
            return;
          case "continue":
            logAgentDebug("AgentStream", "inactivityWatchdog.continue", {
              eventName,
              sessionId: activeSessionId,
              lastRuntimeEventType,
            });
            scheduleInactivityWatchdog();
            return;
          case "fail":
            logAgentDebug(
              "AgentStream",
              "inactivityWatchdog.failed",
              {
                eventName,
                sessionId: activeSessionId,
                lastRuntimeEventType,
              },
              { level: "warn" },
            );
            dispatchSyntheticError(AGENT_STREAM_INACTIVITY_TIMEOUT_MESSAGE);
            return;
        }
      })();
    }, STREAM_INACTIVITY_TIMEOUT_MS);
  };

  const unlisten = await runtime.listenToTurnEvents(
    eventName,
    (event: { payload: unknown }) => {
      const eventReceivedAt = Date.now();
      const data = parseAgentEvent(event.payload);
      const eventType = extractAgentStreamRuntimeEventType(event.payload);
      const schemaVersion = extractAgentStreamRuntimeEventSchemaVersion(
        event.payload,
      );
      if (data?.type === "text_delta" || data?.type === "text_delta_batch") {
        logAgentDebug(
          "AgentStream",
          "inboundTextDelta",
          {
            eventName,
            itemId: data.itemId ?? null,
            parsedType: data.type,
            phase: data.phase ?? null,
            rawEventType: eventType,
            sequence: data.sequence ?? null,
            sessionId: data.session_id ?? activeSessionId,
            turnId: data.turn_id ?? null,
          },
          {
            consoleOnly: true,
            dedupeKey: `${eventName}:inboundTextDelta:${data.itemId ?? "no-item"}:${data.phase ?? "no-phase"}`,
            throttleMs: 1000,
          },
        );
      }
      if (!data) {
        if (!terminalRecoveryPollStarted) {
          clearDeferredRecoveryPoll();
        }
        const unknownEventPlan = resolveAgentStreamUnknownEventPlan({
          eventName,
          eventType,
          schemaVersion,
          warnedEventTypes: warnedUnknownEventTypes,
        });
        if (!unknownEventPlan) {
          return;
        }
        if (!firstEventReceived) {
          markFirstEventReceived({
            eventReceivedAt,
            eventType: unknownEventPlan.eventType,
            recognized: false,
          });
        }
        lastEventReceivedAt = eventReceivedAt;
        lastRuntimeEventType = unknownEventPlan.eventType;
        callbacks.activateStream(
          activeSessionId,
          effectiveWaitingRuntimeStatus,
        );
        if (
          unknownEventPlan.shouldWarn &&
          unknownEventPlan.warningMessage &&
          rememberAgentStreamUnknownEventWarning({
            eventType: unknownEventPlan.eventType,
            warnedEventTypes: warnedUnknownEventTypes,
          })
        ) {
          console.warn(unknownEventPlan.warningMessage);
        }
        scheduleInactivityWatchdog();
        return;
      }
      if (!firstEventReceived) {
        markFirstEventReceived({
          eventReceivedAt,
          eventType: data.type,
          recognized: true,
        });
      }
      if (!terminalRecoveryPollStarted) {
        clearDeferredRecoveryPoll();
      }
      lastEventReceivedAt = eventReceivedAt;
      lastRuntimeEventType = data.type;
      requestState.performanceTrace = buildAgentEventPerformanceTrace({
        activeSessionId,
        event: data,
        eventReceivedAt,
        previousTrace: requestState.performanceTrace,
        resolvedWorkspaceId,
      });

      if (data.type === "provider_trace") {
        const providerTraceContext = buildAgentStreamProviderTraceMetricContext(
          {
            activeSessionId,
            attempt: data.attempt,
            cancelReason: data.cancel_reason,
            elapsedMs: data.elapsed_ms,
            eventName,
            failureCategory: data.failure_category,
            model: data.model,
            provider: data.provider,
            retryable: data.retryable,
            runtimeProviderActiveModel: data.runtime_provider_active_model,
            runtimeProviderBackend: data.runtime_provider_backend,
            runtimeProviderProtocol: data.runtime_provider_protocol,
            runtimeProviderSelector: data.runtime_provider_selector,
            runtimeEventType: data.runtime_event_type,
            stage: data.stage,
            status: data.status,
            textChars: data.text_chars,
          },
        );
        recordAgentStreamPerformanceMetric(
          "agentStream.providerTrace",
          requestState.performanceTrace,
          providerTraceContext,
        );
        logAgentDebug("AgentStream", "providerTrace", providerTraceContext, {
          consoleOnly: true,
          dedupeKey: `${eventName}:providerTrace:${data.stage ?? "no-stage"}:${data.status ?? "no-status"}`,
          throttleMs: 1000,
        });
        scheduleInactivityWatchdog();
        return;
      }

      handleTurnStreamEvent({
        data,
        requestState,
        callbacks: {
          activateStream: () =>
            callbacks.activateStream(
              activeSessionId,
              effectiveWaitingRuntimeStatus,
            ),
          isStreamActivated: callbacks.isStreamActivated,
          clearOptimisticItem: callbacks.clearOptimisticItem,
          clearOptimisticTurn: callbacks.clearOptimisticTurn,
          disposeListener: disposeListenerWithWatchdogs,
          clearActiveStreamIfMatch: callbacks.clearActiveStreamIfMatch,
          appendThinkingToParts,
        },
        observer,
        eventName,
        pendingTurnKey,
        pendingItemKey,
        assistantMsgId,
        activeSessionId,
        resolvedWorkspaceId,
        effectiveExecutionStrategy,
        surfaceThinkingDeltas,
        preserveAssistantContent,
        assistantFallbackContent,
        content,
        runtime,
        _webSearch: webSearch,
        warnedKeysRef,
        actionLoggedKeys,
        toolLogIdByToolId,
        toolStartedAtByToolId,
        toolNameByToolId,
        onWriteFile,
        setMessages,
        setPendingActions,
        getThreadItems,
        setThreadItems,
        setTodoItems,
        setThreadTurns,
        setCurrentTurnId,
        setExecutionRuntime,
        setIsSending,
        soulCopy,
      });
      if (
        data.type === "text_delta" ||
        data.type === "text_delta_batch" ||
        isFinalAgentMessageSnapshotEvent(data)
      ) {
        startTerminalRecoveryPoll();
      }
      scheduleInactivityWatchdog();
    },
  );

  requestState.listenerBoundAt = Date.now();
  const listenerBoundContext = buildAgentStreamListenerBoundContext({
    activeSessionId,
    eventName,
    listenerBoundAt: requestState.listenerBoundAt,
    requestStartedAt: requestState.requestStartedAt,
  });
  recordAgentStreamPerformanceMetric(
    "agentStream.listenerBound",
    requestState.performanceTrace,
    listenerBoundContext,
  );
  logAgentDebug("AgentStream", "listenerBound", listenerBoundContext);

  return () => {
    disposed = true;
    clearFirstEventWatchdog();
    clearInactivityWatchdog();
    clearDeferredRecoveryPoll();
    if (requestState.pendingTextRenderTimerId) {
      clearTimeout(requestState.pendingTextRenderTimerId);
      requestState.pendingTextRenderTimerId = null;
    }
    unlisten();
  };
}
