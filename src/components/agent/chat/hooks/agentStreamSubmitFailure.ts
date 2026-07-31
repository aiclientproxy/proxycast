import { toast } from "sonner";
import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import { activityLogger } from "@/lib/workspace/workbenchRuntime";
import { updateMessageArtifactsStatus } from "../utils/messageArtifacts";
import type { Message, MessageImage } from "../types";
import {
  buildFailedAgentMessageContent,
  buildFailedAgentRuntimeStatus,
} from "../utils/agentRuntimeStatus";
import { resolveAgentRuntimeSubmitErrorMessage } from "../utils/agentRuntimeErrorPresentation";
import { isWorkspacePathErrorMessage } from "./agentChatCoreUtils";
import type {
  ActiveStreamState,
  StreamRequestState,
} from "./agentStreamSubmissionLifecycle";
import type { WorkspacePathMissingState } from "./agentChatShared";
import type { SoulInteractionCopy } from "@/lib/soul/interactionCopy";

interface StreamObserver {
  onError?: (message: string) => void;
}

interface HandleSubmitFailureOptions {
  error: unknown;
  requestState: StreamRequestState;
  observer?: StreamObserver;
  content: string;
  images: MessageImage[];
  assistantMsgId: string;
  eventName: string;
  activeStreamRef: MutableRefObject<ActiveStreamState | null>;
  setMessages: Dispatch<SetStateAction<Message[]>>;
  setWorkspacePathMissing: Dispatch<
    SetStateAction<WorkspacePathMissingState | null>
  >;
  setIsSending: Dispatch<SetStateAction<boolean>>;
  clearActiveStreamIfMatch: (eventName: string) => boolean;
  disposeListener: () => void;
  markOptimisticFailure: (errorMessage: string) => void;
  soulCopy?: SoulInteractionCopy;
}

export function handleAgentStreamSubmitFailure(
  options: HandleSubmitFailureOptions,
) {
  const {
    error,
    requestState,
    observer,
    content,
    images,
    assistantMsgId,
    eventName,
    activeStreamRef,
    setMessages,
    setWorkspacePathMissing,
    setIsSending,
    clearActiveStreamIfMatch,
    disposeListener,
    markOptimisticFailure,
    soulCopy,
  } = options;

  if (requestState.requestLogId && !requestState.requestFinished) {
    requestState.requestFinished = true;
    activityLogger.updateLog(requestState.requestLogId, {
      eventType: "chat_request_error",
      status: "error",
      duration: Date.now() - requestState.requestStartedAt,
      error: error instanceof Error ? error.message : String(error),
    });
  }

  console.error("[AgentChat] 发送失败:", error);
  const errMsg = error instanceof Error ? error.message : String(error);
  const failedRuntimeStatus = buildFailedAgentRuntimeStatus(errMsg, soulCopy);
  observer?.onError?.(errMsg);

  if (errMsg.includes("429") || errMsg.toLowerCase().includes("rate limit")) {
    toast.warning("请求过于频繁，请稍后重试");
  } else if (isWorkspacePathErrorMessage(errMsg)) {
    setWorkspacePathMissing({ content, images });
  } else {
    toast.error(resolveAgentRuntimeSubmitErrorMessage(errMsg));
  }

  markOptimisticFailure(errMsg);
  setMessages((prev) =>
    prev.map((msg) =>
      msg.id === assistantMsgId
        ? {
            ...updateMessageArtifactsStatus(msg, "error"),
            isThinking: false,
            content: buildFailedAgentMessageContent(
              errMsg,
              msg.content,
              soulCopy,
            ),
            runtimeStatus: failedRuntimeStatus,
          }
        : msg,
    ),
  );
  clearActiveStreamIfMatch(eventName);
  disposeListener();
  if (!activeStreamRef.current) {
    setIsSending(false);
  }
}
