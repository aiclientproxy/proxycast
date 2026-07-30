import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import { toast } from "sonner";
import type {
  ConfirmResponse,
  Message,
  ActionRequired,
  AgentThreadItem,
} from "../types";
import { resolveActionPromptKey } from "./agentChatCoreUtils";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import {
  applyAcknowledgedActionRequests,
  mapReplayedActionRequiredToAction,
  markQueuedFallbackActionInMessages,
  markQueuedFallbackActionInPendingActions,
  removeActionsByRequestIds,
  resolveFallbackActionResponsePlan,
  shouldPersistSubmittedActionForType,
  type QueuedFallbackActionResponse,
  upsertAssistantActionRequest,
  upsertSubmittedAction,
} from "./agentChatActionState";
import { markThreadActionItemSubmitted } from "./agentThreadState";
import { buildActionRequestSubmissionContext } from "../utils/actionRequestA2UI";
import { governActionRequest } from "../utils/actionRequestGovernance";
import {
  filterActionsForCurrentAssistantTail,
  findActionRequestSourceMessageId,
} from "../utils/currentTurnActionRequests";
import type { AgentSessionDetailRefreshRequest } from "./agentSessionRefresh";
import {
  actionFromPendingInteraction,
  findPendingActionInteraction,
} from "@/lib/api/agentRuntime/pendingInteractionController";
import { usePendingInteractions } from "./usePendingInteractions";

interface UseAgentToolsOptions {
  runtime: AgentRuntimeAdapter;
  sessionIdRef: MutableRefObject<string | null>;
  currentStreamingSessionIdRef: MutableRefObject<string | null>;
  currentStreamingEventNameRef: MutableRefObject<string | null>;
  messages: Message[];
  setMessages: Dispatch<SetStateAction<Message[]>>;
  setThreadItems: Dispatch<SetStateAction<AgentThreadItem[]>>;
  refreshSessionReadModel: (targetSessionId?: string) => Promise<boolean>;
  refreshSessionDetail?: (
    targetSessionId?: string,
    request?: AgentSessionDetailRefreshRequest,
  ) => Promise<boolean>;
}

function findActionRequestInMessages(
  messages: Message[],
  requestId: string,
): ActionRequired | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "assistant" || !message.actionRequests?.length) {
      continue;
    }

    const action = message.actionRequests.find(
      (item) => item.requestId === requestId,
    );
    if (action) {
      return {
        ...action,
        sourceMessageId: action.sourceMessageId || message.id,
      };
    }
  }

  return undefined;
}

function attachActionSourceFromMessages(
  action: ActionRequired,
  messages: Message[],
): ActionRequired {
  if (action.sourceMessageId) {
    return action;
  }

  const sourceMessageId = findActionRequestSourceMessageId(
    messages,
    action.requestId,
  );
  return sourceMessageId ? { ...action, sourceMessageId } : action;
}

export function useAgentTools(options: UseAgentToolsOptions) {
  const {
    runtime,
    sessionIdRef,
    currentStreamingSessionIdRef,
    currentStreamingEventNameRef,
    messages,
    setMessages,
    setThreadItems,
    refreshSessionReadModel,
    refreshSessionDetail,
  } = options;

  const [pendingActions, setPendingActions] = useState<ActionRequired[]>([]);
  const [submittedActionsInFlight, setSubmittedActionsInFlight] = useState<
    ActionRequired[]
  >([]);
  const {
    interactions: pendingInteractions,
    respond: respondPendingInteraction,
  } = usePendingInteractions();
  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  const projectedServerRequestIdsRef = useRef<Set<string>>(new Set());
  const warnedKeysRef = useRef<Set<string>>(new Set());
  const queuedFallbackResponsesRef = useRef<
    Map<string, QueuedFallbackActionResponse>
  >(new Map());

  useEffect(() => {
    const syncServerRequests = () => {
      const latestActions = pendingInteractions
        .map(actionFromPendingInteraction)
        .filter((action): action is ActionRequired => Boolean(action));
      const latestRequestIds = new Set(
        latestActions.map((action) => action.requestId),
      );
      const resolvedRequestIds = new Set(
        [...projectedServerRequestIdsRef.current].filter(
          (requestId) => !latestRequestIds.has(requestId),
        ),
      );
      if (resolvedRequestIds.size > 0) {
        setPendingActions((prev) =>
          removeActionsByRequestIds(prev, resolvedRequestIds),
        );
        setMessages((prev) =>
          applyAcknowledgedActionRequests({
            messages: prev,
            requestIds: resolvedRequestIds,
            shouldPersistSubmittedAction: false,
          }),
        );
      }
      const assistantMessage = [...messagesRef.current]
        .reverse()
        .find((message) => message.role === "assistant");
      if (!assistantMessage) {
        projectedServerRequestIdsRef.current = new Set();
        return;
      }
      for (const actionData of latestActions) {
        upsertAssistantActionRequest({
          assistantMsgId: assistantMessage.id,
          actionData,
          setPendingActions,
          setMessages,
        });
      }
      projectedServerRequestIdsRef.current = latestRequestIds;
    };
    syncServerRequests();
  }, [pendingInteractions, setMessages]);

  const refreshActionResponseSession = useCallback(
    async (targetSessionId: string) => {
      if (refreshSessionDetail) {
        const refreshedDetail = await refreshSessionDetail(targetSessionId, {
          source: "actionRespond",
          detailMergeMode: "runtime_sync",
        });
        if (refreshedDetail) {
          return true;
        }
      }
      return refreshSessionReadModel(targetSessionId);
    },
    [refreshSessionDetail, refreshSessionReadModel],
  );

  const confirmAction = useCallback(
    async (response: ConfirmResponse) => {
      const acknowledgedRequestIds = new Set<string>([response.requestId]);
      try {
        const pendingAction = pendingActions.find(
          (item) => item.requestId === response.requestId,
        );
        const persistedActionRaw = pendingAction
          ? attachActionSourceFromMessages(pendingAction, messages)
          : findActionRequestInMessages(messages, response.requestId);
        const persistedAction = persistedActionRaw
          ? governActionRequest(persistedActionRaw)
          : undefined;
        const actionType = response.actionType || persistedAction?.actionType;
        if (!actionType) {
          throw new Error("缺少 actionType，无法提交确认");
        }

        const normalizedResponse =
          typeof response.response === "string" ? response.response.trim() : "";
        let submittedUserData: unknown = response.userData;
        let effectiveRequestId = response.requestId;
        let metadataAction = persistedAction;
        let refreshSessionId: string | null = null;

        if (actionType === "elicitation" || actionType === "ask_user") {
          const activeSessionId =
            currentStreamingSessionIdRef.current || sessionIdRef.current;
          if (!activeSessionId) {
            throw new Error("缺少会话 ID，无法提交 elicitation 响应");
          }
          refreshSessionId = activeSessionId;

          let userData: unknown;
          if (!response.confirmed) {
            userData = "";
          } else if (response.userData !== undefined) {
            userData = response.userData;
          } else if (response.response !== undefined) {
            const rawResponse = response.response.trim();
            if (!rawResponse) {
              userData = "";
            } else {
              try {
                userData = JSON.parse(rawResponse);
              } catch {
                userData = rawResponse;
              }
            }
          } else {
            userData = "";
          }

          submittedUserData = userData;

          const userInputInteraction = findPendingActionInteraction(
            pendingInteractions,
            actionType,
            effectiveRequestId,
          );
          const handledByServerRequest = Boolean(
            userInputInteraction &&
            respondPendingInteraction({
              confirmed: response.confirmed !== false,
              interactionId: userInputInteraction.id,
              kind: "request_user_input",
              response: response.response,
              userData,
            }).accepted,
          );
          if (actionType === "ask_user" && !handledByServerRequest) {
            throw new Error("用户输入请求已失效，请等待当前 typed request");
          }

          if (!handledByServerRequest) {
            const fallbackPlan = resolveFallbackActionResponsePlan({
              actionType,
              pendingActions,
              persistedAction,
              response,
              userData,
            });

            if (fallbackPlan.kind === "queue") {
              queuedFallbackResponsesRef.current.set(
                fallbackPlan.promptKey,
                fallbackPlan.queuedResponse,
              );
              setPendingActions((prev) =>
                markQueuedFallbackActionInPendingActions(
                  prev,
                  fallbackPlan.queuedResponse.requestId,
                  normalizedResponse || undefined,
                  submittedUserData,
                ),
              );
              setMessages((prev) =>
                markQueuedFallbackActionInMessages({
                  messages: prev,
                  requestId: fallbackPlan.queuedResponse.requestId,
                  submittedResponse: normalizedResponse || undefined,
                  submittedUserData,
                }),
              );
              await refreshSessionReadModel(activeSessionId);
              toast.info("已记录你的回答，等待系统请求就绪后自动提交");
              return;
            }

            if (fallbackPlan.kind === "submit_resolved") {
              effectiveRequestId = fallbackPlan.resolvedAction.requestId;
              metadataAction = governActionRequest(fallbackPlan.resolvedAction);
              acknowledgedRequestIds.add(fallbackPlan.resolvedAction.requestId);
            }
          }

          setSubmittedActionsInFlight((prev) =>
            upsertSubmittedAction(prev, {
              ...(metadataAction ||
                persistedAction || {
                  requestId: effectiveRequestId,
                  actionType,
                }),
              requestId: effectiveRequestId,
              actionType,
              status: "submitted",
              submittedResponse: normalizedResponse || undefined,
              submittedUserData,
            }),
          );

          const submissionContext = metadataAction
            ? buildActionRequestSubmissionContext(metadataAction, userData)
            : null;
          const activeEventName =
            currentStreamingSessionIdRef.current === activeSessionId
              ? currentStreamingEventNameRef.current
              : null;
          const submissionEventName =
            activeEventName || metadataAction?.eventName;

          if (!handledByServerRequest) {
            await runtime.respondToAction({
              sessionId: activeSessionId,
              requestId: effectiveRequestId,
              actionType,
              confirmed: response.confirmed,
              decision: response.decision,
              response: response.response,
              userData,
              metadata: submissionContext?.requestMetadata,
              eventName: submissionEventName || undefined,
              actionScope: metadataAction?.scope,
            });
          }
        } else {
          const activeSessionId =
            currentStreamingSessionIdRef.current || sessionIdRef.current;
          refreshSessionId = activeSessionId;
          setSubmittedActionsInFlight((prev) =>
            upsertSubmittedAction(prev, {
              ...(metadataAction ||
                persistedAction || {
                  requestId: effectiveRequestId,
                  actionType,
                }),
              requestId: effectiveRequestId,
              actionType,
              status: "submitted",
              submittedResponse: normalizedResponse || undefined,
              submittedUserData,
            }),
          );
          const approvalInteraction = findPendingActionInteraction(
            pendingInteractions,
            actionType,
            effectiveRequestId,
          );
          const handledByServerRequest = Boolean(
            approvalInteraction &&
            respondPendingInteraction({
              interactionId: approvalInteraction.id,
              kind: "approval",
              response,
            }).accepted,
          );
          if (!handledByServerRequest) {
            throw new Error("审批请求已失效，请等待当前 typed request");
          }
        }

        setPendingActions((prev) =>
          removeActionsByRequestIds(prev, acknowledgedRequestIds),
        );
        const shouldPersistSubmittedAction =
          shouldPersistSubmittedActionForType(actionType);
        setMessages((prev) =>
          applyAcknowledgedActionRequests({
            messages: prev,
            requestIds: acknowledgedRequestIds,
            shouldPersistSubmittedAction,
            submittedResponse: normalizedResponse || undefined,
            submittedUserData,
          }),
        );
        setThreadItems((prev) =>
          markThreadActionItemSubmitted(
            prev,
            acknowledgedRequestIds,
            normalizedResponse || undefined,
            submittedUserData,
          ),
        );
        if (refreshSessionId) {
          await refreshActionResponseSession(refreshSessionId);
        }
        setSubmittedActionsInFlight((prev) =>
          removeActionsByRequestIds(prev, acknowledgedRequestIds),
        );
      } catch (error) {
        setSubmittedActionsInFlight((prev) =>
          removeActionsByRequestIds(prev, acknowledgedRequestIds),
        );
        console.error("[AgentChat] 确认失败:", error);
        toast.error(
          error instanceof Error && error.message
            ? error.message
            : "确认操作失败",
        );
      }
    },
    [
      currentStreamingSessionIdRef,
      currentStreamingEventNameRef,
      messages,
      pendingActions,
      refreshActionResponseSession,
      runtime,
      refreshSessionReadModel,
      sessionIdRef,
      setMessages,
      setThreadItems,
      pendingInteractions,
      respondPendingInteraction,
    ],
  );

  const visiblePendingActions = useMemo(
    () => filterActionsForCurrentAssistantTail(pendingActions, messages),
    [messages, pendingActions],
  );
  const visibleSubmittedActionsInFlight = useMemo(
    () =>
      filterActionsForCurrentAssistantTail(submittedActionsInFlight, messages, {
        keepUnscoped: true,
      }),
    [messages, submittedActionsInFlight],
  );

  useEffect(() => {
    for (const pendingAction of visiblePendingActions) {
      if (
        pendingAction.isFallback ||
        pendingAction.status === "submitted" ||
        (pendingAction.actionType !== "ask_user" &&
          pendingAction.actionType !== "elicitation")
      ) {
        continue;
      }

      const promptKey = resolveActionPromptKey(pendingAction);
      if (!promptKey) {
        continue;
      }

      const queuedResponse = queuedFallbackResponsesRef.current.get(promptKey);
      if (!queuedResponse) {
        continue;
      }
      if (
        queuedResponse.sourceMessageId &&
        pendingAction.sourceMessageId !== queuedResponse.sourceMessageId
      ) {
        continue;
      }

      queuedFallbackResponsesRef.current.delete(promptKey);
      void confirmAction({
        ...queuedResponse,
        requestId: pendingAction.requestId,
        actionType: pendingAction.actionType,
      });
      break;
    }
  }, [confirmAction, visiblePendingActions]);

  const handlePermissionResponse = useCallback(
    async (response: ConfirmResponse) => {
      await confirmAction(response);
    },
    [confirmAction],
  );

  const replayPendingAction = useCallback(
    async (requestId: string, assistantMessageId: string) => {
      const activeSessionId = sessionIdRef.current;
      if (!activeSessionId) {
        toast.error("当前没有激活会话，无法重新拉起请求");
        return false;
      }

      try {
        const replayedAction = await runtime.replayRequest(
          activeSessionId,
          requestId,
        );

        if (!replayedAction) {
          await refreshSessionReadModel(activeSessionId);
          toast.error("待处理请求已不存在，无法重新拉起");
          return false;
        }

        const actionData = mapReplayedActionRequiredToAction(replayedAction);

        upsertAssistantActionRequest({
          assistantMsgId: assistantMessageId,
          actionData,
          replaceByPrompt:
            actionData.actionType === "ask_user" ||
            actionData.actionType === "elicitation",
          setPendingActions,
          setMessages,
        });
        setSubmittedActionsInFlight((prev) =>
          prev.filter(
            (item) =>
              item.requestId !== requestId &&
              item.requestId !== actionData.requestId,
          ),
        );
        toast.success("已重新拉起待处理请求");
        return true;
      } catch (error) {
        console.error("[AgentChat] 重新拉起请求失败:", error);
        toast.error(
          error instanceof Error && error.message
            ? error.message
            : "重新拉起请求失败",
        );
        return false;
      }
    },
    [refreshSessionReadModel, runtime, sessionIdRef, setMessages],
  );

  return {
    pendingActions: visiblePendingActions,
    submittedActionsInFlight: visibleSubmittedActionsInFlight,
    setPendingActions,
    warnedKeysRef,
    confirmAction,
    handlePermissionResponse,
    replayPendingAction,
  };
}
