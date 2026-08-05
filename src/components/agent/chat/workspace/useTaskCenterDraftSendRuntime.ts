import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import { flushSync } from "react-dom";
import { recordAgentUiPerformanceMetric } from "@/lib/agentUiPerformanceMetrics";
import { mergeAgentUiPerformanceTraceMetadata } from "../hooks/agentStreamPerformanceMetrics";
import {
  buildHomePendingPreviewMessages,
  type TaskCenterDraftSendRequest,
} from "../homePendingPreview";
import type { Message } from "../types";
import type { InputbarSendHandler } from "../components/Inputbar/inputbarSendPayload";
import type { WorkspaceHandleSend } from "./useWorkspaceSendActions";
import {
  createTaskCenterDraftSendRequestId,
  resolveTaskCenterDraftSendTitle,
  scheduleAfterCommit,
  scheduleAfterNextPaint,
  type TaskCenterDraftTab,
} from "./agentChatWorkspaceHelpers";
import type { HandleSendOptions } from "../hooks/handleSendTypes";
import { markTaskCenterDraftTabRunning } from "./taskCenterDraftTabs";
import type { SoulInteractionCopy } from "@/lib/soul/interactionCopy";
import { logAgentDebug } from "@/lib/agentDebug";

type TaskCenterDraftSendExecutionStrategy = NonNullable<
  TaskCenterDraftSendRequest["sendExecutionStrategy"]
>;

function flushHomePendingPreview(apply: () => void): void {
  flushSync(apply);
}

interface UseTaskCenterHomePendingPreviewRuntimeParams {
  homePendingPreviewRequest: TaskCenterDraftSendRequest | null;
  displayMessagesLength: number;
  executionStrategy: TaskCenterDraftSendExecutionStrategy;
  workspaceId?: string | null;
  soulCopy?: SoulInteractionCopy;
}

interface UseTaskCenterEmptyStateSendRuntimeParams {
  agentEntry: string;
  input: string;
  setInput: (value: string) => void;
  activeSessionIdRef?: MutableRefObject<string | null>;
  activeDraftTabIdRef: MutableRefObject<string | null>;
  clearMessages: (options?: { showToast?: boolean }) => void;
  displayMessagesLength: number;
  turnsLength: number;
  threadItemsLength: number;
  hasDisplayMessages: boolean;
  handleSend: WorkspaceHandleSend;
  sessionId?: string | null;
  taskCenterWorkspaceId?: string | null;
  setTaskCenterDraftTabs: Dispatch<SetStateAction<TaskCenterDraftTab[]>>;
  setTaskCenterDraftSendRequest: Dispatch<
    SetStateAction<TaskCenterDraftSendRequest | null>
  >;
  taskCenterDraftSendRequest: TaskCenterDraftSendRequest | null;
  setHomePendingPreviewRequest: Dispatch<
    SetStateAction<TaskCenterDraftSendRequest | null>
  >;
  materializedSessionIdsRef?: MutableRefObject<Map<string, string>>;
  prewarmedDraftSessionIdsRef?: MutableRefObject<Set<string>>;
  onNonMaterializedSessionReady?: (
    sessionId: string,
    options?: { sourceDraftTabId?: string | null },
  ) => void;
}

export function useTaskCenterHomePendingPreviewRuntime({
  homePendingPreviewRequest,
  displayMessagesLength,
  executionStrategy,
  workspaceId,
  soulCopy,
}: UseTaskCenterHomePendingPreviewRuntimeParams): {
  homePendingPreviewMessages: Message[];
  isHomePendingPreviewActive: boolean;
} {
  const committedRequestIdsRef = useRef<Set<string>>(new Set());
  const paintedRequestIdsRef = useRef<Set<string>>(new Set());
  const shouldRenderPendingPreview =
    Boolean(homePendingPreviewRequest) &&
    (!homePendingPreviewRequest?.sessionReady || displayMessagesLength === 0);
  const homePendingPreviewMessages = useMemo(
    () =>
      homePendingPreviewRequest && shouldRenderPendingPreview
        ? buildHomePendingPreviewMessages(
            homePendingPreviewRequest,
            executionStrategy,
            soulCopy,
          )
        : [],
    [
      executionStrategy,
      homePendingPreviewRequest,
      shouldRenderPendingPreview,
      soulCopy,
    ],
  );

  useLayoutEffect(() => {
    if (
      !homePendingPreviewRequest ||
      homePendingPreviewMessages.length === 0 ||
      committedRequestIdsRef.current.has(homePendingPreviewRequest.id)
    ) {
      return;
    }

    const request = homePendingPreviewRequest;
    committedRequestIdsRef.current.add(request.id);
    recordAgentUiPerformanceMetric("homeInput.pendingPreviewCommitted", {
      durationMs: Date.now() - request.submittedAt,
      requestId: request.id,
      sessionId: request.draftTabId,
      source: request.source,
      workspaceId: workspaceId ?? null,
    });
  }, [
    homePendingPreviewMessages.length,
    homePendingPreviewRequest,
    workspaceId,
  ]);

  useEffect(() => {
    if (
      !homePendingPreviewRequest ||
      homePendingPreviewMessages.length === 0 ||
      paintedRequestIdsRef.current.has(homePendingPreviewRequest.id)
    ) {
      return;
    }

    const request = homePendingPreviewRequest;
    paintedRequestIdsRef.current.add(request.id);
    return scheduleAfterNextPaint(() => {
      recordAgentUiPerformanceMetric("homeInput.pendingPreviewPaint", {
        durationMs: Date.now() - request.submittedAt,
        requestId: request.id,
        sessionId: request.draftTabId,
        source: request.source,
        workspaceId: workspaceId ?? null,
      });
    });
  }, [
    homePendingPreviewMessages.length,
    homePendingPreviewRequest,
    workspaceId,
  ]);

  return {
    homePendingPreviewMessages,
    isHomePendingPreviewActive: homePendingPreviewMessages.length > 0,
  };
}

export function useTaskCenterEmptyStateSendRuntime({
  agentEntry,
  input,
  setInput,
  activeSessionIdRef,
  activeDraftTabIdRef,
  clearMessages,
  displayMessagesLength,
  turnsLength,
  threadItemsLength,
  hasDisplayMessages,
  handleSend,
  sessionId,
  taskCenterWorkspaceId,
  setTaskCenterDraftTabs,
  setTaskCenterDraftSendRequest,
  taskCenterDraftSendRequest,
  setHomePendingPreviewRequest,
  materializedSessionIdsRef,
  prewarmedDraftSessionIdsRef,
  onNonMaterializedSessionReady,
}: UseTaskCenterEmptyStateSendRuntimeParams): InputbarSendHandler {
  const homeSendInFlightRef = useRef(false);

  useEffect(() => {
    if (!taskCenterDraftSendRequest) {
      homeSendInFlightRef.current = false;
    }
  }, [taskCenterDraftSendRequest]);

  return useCallback<InputbarSendHandler>(
    (payload = {}) => {
      const handlerEnteredAt = Date.now();
      const triggeredAt =
        typeof payload.triggeredAt === "number" &&
        Number.isFinite(payload.triggeredAt)
          ? payload.triggeredAt
          : handlerEnteredAt;
      const triggerSource = payload.triggerSource ?? "adapter";
      if (homeSendInFlightRef.current || taskCenterDraftSendRequest) {
        logAgentDebug("AgentChatPage", "homeInput.submit.blocked", {
          hasPendingRequest: Boolean(taskCenterDraftSendRequest),
          homeSendInFlight: homeSendInFlightRef.current,
          source: "empty-state",
          triggerSource,
          workspaceId: taskCenterWorkspaceId,
        });
        return false;
      }

      const text = payload.textOverride ?? input;
      const images = payload.images ?? [];
      const sendOptions = payload.sendOptions;
      const normalizedText = text.trim();
      const activeDraftTabId = activeDraftTabIdRef.current;
      if (
        (agentEntry === "claw" || agentEntry === "new-task") &&
        activeDraftTabId
      ) {
        homeSendInFlightRef.current = true;
        const submittedAt = triggeredAt;
        const requestId = createTaskCenterDraftSendRequestId();
        const tracedSendOptions: HandleSendOptions = {
          ...(sendOptions || {}),
          skipWorkspaceCommandRouting: true,
          skipSessionRestore: true,
          skipSessionStartHooks: true,
          requestMetadata: mergeAgentUiPerformanceTraceMetadata(
            sendOptions?.requestMetadata,
            {
              requestId,
              sessionId: null,
              source: "task-center-empty-state",
              submittedAt,
              workspaceId: taskCenterWorkspaceId ?? null,
            },
          ),
        };
        const shouldUseMaterializedDraftDispatch =
          materializedSessionIdsRef?.current.has(activeDraftTabId) ||
          prewarmedDraftSessionIdsRef?.current.has(activeDraftTabId) ||
          false;
        recordAgentUiPerformanceMetric("homeInput.submit", {
          hasDraftTab: true,
          inputbarHandlerToHomeSubmitMs: Math.max(
            0,
            Date.now() - handlerEnteredAt,
          ),
          inputLength: normalizedText.length,
          requestId,
          sessionId: activeDraftTabId,
          source: "task-center-empty-state",
          triggerToHomeSubmitMs: Math.max(0, Date.now() - triggeredAt),
          triggerSource,
          workspaceId: taskCenterWorkspaceId,
        });
        const request: TaskCenterDraftSendRequest = {
          id: requestId,
          draftTabId: shouldUseMaterializedDraftDispatch
            ? activeDraftTabId
            : requestId,
          text,
          images,
          sendOptions: tracedSendOptions,
          submittedAt,
          materializeDraft: shouldUseMaterializedDraftDispatch,
          dispatchState: shouldUseMaterializedDraftDispatch
            ? undefined
            : "dispatched",
          source: "task-center-empty-state",
          triggerSource,
        };
        flushHomePendingPreview(() => {
          setInput("");
          if (
            displayMessagesLength > 0 ||
            turnsLength > 0 ||
            threadItemsLength > 0
          ) {
            clearMessages({ showToast: false });
          }
          setTaskCenterDraftSendRequest(request);
          setHomePendingPreviewRequest(request);
        });
        if (request.materializeDraft) {
          return;
        }

        scheduleAfterCommit(() => {
          setTaskCenterDraftTabs((current) =>
            markTaskCenterDraftTabRunning({
              current,
              draftTabId: activeDraftTabId,
              title: resolveTaskCenterDraftSendTitle(text),
            }),
          );
          recordAgentUiPerformanceMetric("homeInput.sendDispatch.start", {
            elapsedMs: Date.now() - submittedAt,
            requestId,
            sessionId: null,
            source: "task-center-empty-state",
            workspaceId: taskCenterWorkspaceId ?? null,
          });
          void handleSend(
            images,
            undefined,
            undefined,
            text,
            undefined,
            undefined,
            tracedSendOptions,
          ).then(
            (result) => {
              recordAgentUiPerformanceMetric("homeInput.sendDispatch.done", {
                durationMs: Date.now() - submittedAt,
                requestId,
                result,
                sessionId: null,
                source: "task-center-empty-state",
                workspaceId: taskCenterWorkspaceId ?? null,
              });
              if (result !== true) {
                setInput(text);
                setHomePendingPreviewRequest((current) =>
                  current?.id === requestId ? null : current,
                );
              } else {
                const readySessionId =
                  activeSessionIdRef?.current?.trim() || null;
                if (readySessionId) {
                  onNonMaterializedSessionReady?.(readySessionId, {
                    sourceDraftTabId: activeDraftTabId,
                  });
                  setHomePendingPreviewRequest((current) =>
                    current?.id === requestId
                      ? {
                          ...current,
                          draftTabId: readySessionId,
                          sessionReady: true,
                        }
                      : current,
                  );
                }
              }
              setTaskCenterDraftSendRequest((current) =>
                current?.id === requestId ? null : current,
              );
              homeSendInFlightRef.current = false;
            },
            (error) => {
              recordAgentUiPerformanceMetric("homeInput.sendDispatch.error", {
                durationMs: Date.now() - submittedAt,
                error: error instanceof Error ? error.message : String(error),
                requestId,
                sessionId: null,
                source: "task-center-empty-state",
                workspaceId: taskCenterWorkspaceId ?? null,
              });
              setInput(text);
              setTaskCenterDraftSendRequest((current) =>
                current?.id === requestId ? null : current,
              );
              setHomePendingPreviewRequest((current) =>
                current?.id === requestId ? null : current,
              );
              homeSendInFlightRef.current = false;
            },
          );
        });
        return;
      }

      const shouldQueueHomeSend =
        !hasDisplayMessages &&
        (agentEntry === "claw" || agentEntry === "new-task");
      if (shouldQueueHomeSend) {
        homeSendInFlightRef.current = true;
        const submittedAt = triggeredAt;
        const requestId = createTaskCenterDraftSendRequestId();
        const existingSessionId = sessionId?.trim() || null;
        recordAgentUiPerformanceMetric("homeInput.submit", {
          hasDraftTab: false,
          inputbarHandlerToHomeSubmitMs: Math.max(
            0,
            Date.now() - handlerEnteredAt,
          ),
          inputLength: normalizedText.length,
          requestId,
          sessionId: existingSessionId,
          source: "empty-state",
          triggerToHomeSubmitMs: Math.max(0, Date.now() - triggeredAt),
          triggerSource,
          workspaceId: taskCenterWorkspaceId,
        });
        if (!existingSessionId) {
          const tracedSendOptions: HandleSendOptions = {
            ...(sendOptions || {}),
            skipWorkspaceCommandRouting: true,
            skipSessionRestore: true,
            skipSessionStartHooks: true,
            requestMetadata: mergeAgentUiPerformanceTraceMetadata(
              sendOptions?.requestMetadata,
              {
                requestId,
                sessionId: null,
                source: "empty-state",
                submittedAt,
                workspaceId: taskCenterWorkspaceId ?? null,
              },
            ),
          };
          const previewRequest: TaskCenterDraftSendRequest = {
            id: requestId,
            draftTabId: requestId,
            text,
            images,
            sendOptions: tracedSendOptions,
            submittedAt,
            materializeDraft: false,
            dispatchState: "dispatched",
            source: "empty-state",
            triggerSource,
          };
          flushHomePendingPreview(() => {
            setInput("");
            setTaskCenterDraftSendRequest(previewRequest);
            setHomePendingPreviewRequest(previewRequest);
          });
          logAgentDebug("AgentChatPage", "homeInput.directDispatch.start", {
            requestId,
            source: "empty-state",
            workspaceId: taskCenterWorkspaceId,
          });
          scheduleAfterCommit(() => {
            recordAgentUiPerformanceMetric("homeInput.sendDispatch.start", {
              elapsedMs: Date.now() - submittedAt,
              requestId,
              sessionId: null,
              source: "empty-state",
              workspaceId: taskCenterWorkspaceId ?? null,
            });
            void handleSend(
              images,
              undefined,
              undefined,
              text,
              undefined,
              undefined,
              tracedSendOptions,
            ).then(
              (result) => {
                recordAgentUiPerformanceMetric("homeInput.sendDispatch.done", {
                  durationMs: Date.now() - submittedAt,
                  requestId,
                  result,
                  sessionId: null,
                  source: "empty-state",
                  workspaceId: taskCenterWorkspaceId ?? null,
                });
                logAgentDebug(
                  "AgentChatPage",
                  "homeInput.directDispatch.done",
                  {
                    requestId,
                    result,
                    source: "empty-state",
                    workspaceId: taskCenterWorkspaceId,
                  },
                );
                if (result !== true) {
                  setInput(text);
                  setHomePendingPreviewRequest((current) =>
                    current?.id === requestId ? null : current,
                  );
                } else {
                  const readySessionId =
                    activeSessionIdRef?.current?.trim() || null;
                  if (readySessionId) {
                    onNonMaterializedSessionReady?.(readySessionId);
                    setHomePendingPreviewRequest((current) =>
                      current?.id === requestId
                        ? {
                            ...current,
                            draftTabId: readySessionId,
                            sessionReady: true,
                          }
                        : current,
                    );
                  }
                }
                setTaskCenterDraftSendRequest((current) =>
                  current?.id === requestId ? null : current,
                );
                homeSendInFlightRef.current = false;
              },
              (error) => {
                recordAgentUiPerformanceMetric("homeInput.sendDispatch.error", {
                  durationMs: Date.now() - submittedAt,
                  error: error instanceof Error ? error.message : String(error),
                  requestId,
                  sessionId: null,
                  source: "empty-state",
                  workspaceId: taskCenterWorkspaceId ?? null,
                });
                logAgentDebug(
                  "AgentChatPage",
                  "homeInput.directDispatch.error",
                  {
                    error,
                    requestId,
                    source: "empty-state",
                    workspaceId: taskCenterWorkspaceId,
                  },
                  { level: "error" },
                );
                setInput(text);
                setTaskCenterDraftSendRequest((current) =>
                  current?.id === requestId ? null : current,
                );
                setHomePendingPreviewRequest((current) =>
                  current?.id === requestId ? null : current,
                );
                homeSendInFlightRef.current = false;
              },
            );
          });
          return;
        }
        const tracedSendOptions: HandleSendOptions = {
          ...(sendOptions || {}),
          skipWorkspaceCommandRouting: true,
          skipSessionRestore: true,
          skipSessionStartHooks: true,
          requestMetadata: mergeAgentUiPerformanceTraceMetadata(
            sendOptions?.requestMetadata,
            {
              requestId,
              sessionId: existingSessionId,
              source: "empty-state",
              submittedAt,
              workspaceId: taskCenterWorkspaceId ?? null,
            },
          ),
        };
        const previewRequest: TaskCenterDraftSendRequest = {
          id: requestId,
          draftTabId: existingSessionId,
          text,
          images,
          sendOptions: tracedSendOptions,
          submittedAt,
          materializeDraft: false,
          dispatchState: "dispatched",
          source: "empty-state",
          triggerSource,
        };
        flushHomePendingPreview(() => {
          setInput("");
          setTaskCenterDraftSendRequest(previewRequest);
          setHomePendingPreviewRequest(previewRequest);
        });
        scheduleAfterCommit(() => {
          recordAgentUiPerformanceMetric("homeInput.sendDispatch.start", {
            elapsedMs: Date.now() - submittedAt,
            requestId,
            sessionId: existingSessionId,
            source: "empty-state",
            workspaceId: taskCenterWorkspaceId ?? null,
          });
          void handleSend(
            images,
            undefined,
            undefined,
            text,
            undefined,
            undefined,
            tracedSendOptions,
          ).then(
            (result) => {
              recordAgentUiPerformanceMetric("homeInput.sendDispatch.done", {
                durationMs: Date.now() - submittedAt,
                requestId,
                result,
                sessionId: existingSessionId,
                source: "empty-state",
                workspaceId: taskCenterWorkspaceId ?? null,
              });
              if (result !== true) {
                setInput(text);
                setHomePendingPreviewRequest((current) =>
                  current?.id === requestId ? null : current,
                );
              }
              setTaskCenterDraftSendRequest((current) =>
                current?.id === requestId ? null : current,
              );
              homeSendInFlightRef.current = false;
            },
            (error) => {
              recordAgentUiPerformanceMetric("homeInput.sendDispatch.error", {
                durationMs: Date.now() - submittedAt,
                error: error instanceof Error ? error.message : String(error),
                requestId,
                sessionId: existingSessionId,
                source: "empty-state",
                workspaceId: taskCenterWorkspaceId ?? null,
              });
              setInput(text);
              setTaskCenterDraftSendRequest((current) =>
                current?.id === requestId ? null : current,
              );
              setHomePendingPreviewRequest((current) =>
                current?.id === requestId ? null : current,
              );
              homeSendInFlightRef.current = false;
            },
          );
        });
        return;
      }

      recordAgentUiPerformanceMetric("homeInput.submit", {
        hasDraftTab: false,
        inputbarHandlerToHomeSubmitMs: Math.max(
          0,
          Date.now() - handlerEnteredAt,
        ),
        inputLength: normalizedText.length,
        sessionId: sessionId ?? null,
        source: "empty-state",
        triggerToHomeSubmitMs: Math.max(0, Date.now() - triggeredAt),
        triggerSource,
        workspaceId: taskCenterWorkspaceId,
      });
      void handleSend(
        images,
        undefined,
        undefined,
        text,
        undefined,
        undefined,
        sendOptions,
      );
    },
    [
      activeDraftTabIdRef,
      activeSessionIdRef,
      agentEntry,
      clearMessages,
      displayMessagesLength,
      handleSend,
      hasDisplayMessages,
      input,
      materializedSessionIdsRef,
      sessionId,
      onNonMaterializedSessionReady,
      prewarmedDraftSessionIdsRef,
      setHomePendingPreviewRequest,
      setInput,
      setTaskCenterDraftSendRequest,
      setTaskCenterDraftTabs,
      taskCenterDraftSendRequest,
      taskCenterWorkspaceId,
      threadItemsLength,
      turnsLength,
    ],
  );
}

export { useTaskCenterDraftSendDispatchRuntime } from "./useTaskCenterDraftSendDispatchRuntime";
