import {
  useEffect,
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
} from "react";
import type { AgentThreadTurn } from "@/lib/api/agentProtocol";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import type { AppServerJsonRpcNotification } from "@/lib/api/appServer";
import { subscribeAppServerNotifications } from "@/lib/api/appServerEventBus";
import {
  APP_SERVER_EVENT_DRAIN_ACTIVE_INTERVAL_MS,
  APP_SERVER_EVENT_DRAIN_INTERVAL_MS,
  APP_SERVER_EVENT_DRAIN_LIMIT,
} from "@/lib/api/agentRuntime/appServerEventStream";
import {
  projectAppServerV2NotificationPayload,
  readAppServerV2NotificationRoute,
} from "@/lib/api/agentRuntime/appServerV2Notification";
import { parseAgentRuntimeEvent } from "@/lib/api/agentProtocolRuntimeParsers";
import type { Message } from "../types";
import { applyAssistantTurnUsage } from "./agentMessageUsageMerge";

const MAX_SESSION_ASSISTANT_TURN_USAGES = 256;

type AssistantUsage = NonNullable<Message["usage"]>;

export function shouldReplayRecentAssistantTurnUsage(
  messages: readonly Message[],
  threadTurns: ReadonlyArray<Pick<AgentThreadTurn, "id" | "status">>,
): boolean {
  const terminalTurnIds = new Set(
    threadTurns
      .filter((turn) => turn.status !== "running")
      .map((turn) => turn.id.trim())
      .filter(Boolean),
  );
  return messages.some(
    (message) =>
      message.role === "assistant" &&
      Boolean(message.imageWorkbenchPreview) &&
      !message.usage &&
      Boolean(
        message.runtimeTurnId &&
        terminalTurnIds.has(message.runtimeTurnId.trim()),
      ),
  );
}

export function readScopedAssistantTurnUsage(params: {
  notification: AppServerJsonRpcNotification;
  currentThreadId?: string | null;
  messages: readonly Message[];
  threadTurns: readonly AgentThreadTurn[];
}): { runtimeTurnId: string; usage: AssistantUsage } | null {
  if (params.notification.method !== "thread/tokenUsage/updated") {
    return null;
  }
  const route = readAppServerV2NotificationRoute(params.notification);
  const currentThreadId = params.currentThreadId?.trim();
  if (!route || (currentThreadId && route.threadId !== currentThreadId)) {
    return null;
  }
  const projected = projectAppServerV2NotificationPayload(params.notification);
  const projectedType = projected?.type;
  if (typeof projectedType !== "string" || !projected) {
    return null;
  }
  const event = parseAgentRuntimeEvent(projectedType, projected);
  if (event?.type !== "token_usage_updated") {
    return null;
  }
  const runtimeTurnId = event.turn_id?.trim() || route.turnId?.trim();
  if (!runtimeTurnId) {
    return null;
  }
  const matchesCurrentSession =
    Boolean(currentThreadId) ||
    params.messages.some(
      (message) => message.runtimeTurnId === runtimeTurnId,
    ) ||
    params.threadTurns.some((turn) => turn.id === runtimeTurnId);
  return matchesCurrentSession ? { runtimeTurnId, usage: event.usage } : null;
}

export function rememberBoundedAssistantTurnUsage(
  usageByRuntimeTurnId: Map<string, AssistantUsage>,
  runtimeTurnId: string,
  usage: AssistantUsage,
  limit = MAX_SESSION_ASSISTANT_TURN_USAGES,
): void {
  usageByRuntimeTurnId.delete(runtimeTurnId);
  usageByRuntimeTurnId.set(runtimeTurnId, usage);
  while (usageByRuntimeTurnId.size > limit) {
    const oldestTurnId = usageByRuntimeTurnId.keys().next().value;
    if (!oldestTurnId) {
      return;
    }
    usageByRuntimeTurnId.delete(oldestTurnId);
  }
}

export function useAgentSessionTokenUsage(params: {
  sessionId?: string | null;
  sessionIdRef: MutableRefObject<string | null>;
  messagesRef: MutableRefObject<Message[]>;
  threadReadRef: MutableRefObject<AgentRuntimeThreadReadModel | null>;
  threadTurnsRef: MutableRefObject<AgentThreadTurn[]>;
  usageByRuntimeTurnIdRef: MutableRefObject<Map<string, AssistantUsage>>;
  setMessages: Dispatch<SetStateAction<Message[]>>;
}): void {
  const {
    messagesRef,
    sessionId,
    sessionIdRef,
    setMessages,
    threadReadRef,
    threadTurnsRef,
    usageByRuntimeTurnIdRef,
  } = params;
  useEffect(() => {
    const activeSessionId = sessionId?.trim();
    if (!activeSessionId) {
      usageByRuntimeTurnIdRef.current.clear();
      return;
    }

    usageByRuntimeTurnIdRef.current.clear();
    return subscribeAppServerNotifications({
      getDrainOptions: () => ({
        activeIntervalMs: APP_SERVER_EVENT_DRAIN_ACTIVE_INTERVAL_MS,
        includeRecent: shouldReplayRecentAssistantTurnUsage(
          messagesRef.current,
          threadTurnsRef.current,
        ),
        intervalMs: APP_SERVER_EVENT_DRAIN_INTERVAL_MS,
        limit: APP_SERVER_EVENT_DRAIN_LIMIT,
      }),
      onNotifications: (notifications) => {
        if (sessionIdRef.current !== activeSessionId) {
          return;
        }

        for (const notification of notifications) {
          const projected = readScopedAssistantTurnUsage({
            notification,
            currentThreadId: threadReadRef.current?.thread_id,
            messages: messagesRef.current,
            threadTurns: threadTurnsRef.current,
          });
          if (!projected) {
            continue;
          }
          rememberBoundedAssistantTurnUsage(
            usageByRuntimeTurnIdRef.current,
            projected.runtimeTurnId,
            projected.usage,
          );
          setMessages((previous) =>
            applyAssistantTurnUsage(
              previous,
              projected.runtimeTurnId,
              projected.usage,
            ),
          );
        }
      },
    });
  }, [
    messagesRef,
    sessionId,
    sessionIdRef,
    setMessages,
    threadReadRef,
    threadTurnsRef,
    usageByRuntimeTurnIdRef,
  ]);
}
