import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import type {
  AgentEvent,
  AgentThreadItem,
  AgentThreadTurn,
} from "@/lib/api/agentProtocol";
import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntime,
} from "@/lib/api/agentExecutionRuntime";
import type { AgentTodoItem } from "@/lib/api/agentRuntime/sessionTypes";
import type { ActionRequired, Message } from "../types";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import type { AgentUiPerformanceTraceMetadata } from "./agentStreamPerformanceMetrics";
import type { AgentStreamReasoningTimelineState } from "./agentStreamReasoningTimeline";
import type { TextSegmentFinalEligibility } from "./agentStreamTextDeltaLifecycle";
import type { SoulInteractionCopy } from "@/lib/soul/interactionCopy";
import type { AgentStreamConversationProjectionOwner } from "./agentStreamConversationProjection";

export type MessageParts = NonNullable<Message["contentParts"]>;

export interface StreamObserver {
  onTextDelta?: (delta: string, accumulated: string) => void;
  onComplete?: (content: string) => void;
  onError?: (message: string) => void;
}

export interface PendingImageTaskPresentation {
  assistantIntro: string;
  completionCaption: string;
  workflowRunId?: string | null;
  turnId?: string | null;
}

export interface StreamRequestState extends AgentStreamReasoningTimelineState {
  conversationProjectionOwner?: AgentStreamConversationProjectionOwner;
  accumulatedContent: string;
  hasMeaningfulCompletionSignal?: boolean;
  hasFinalAnswerRequiredProcessBoundary?: boolean;
  hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary?: boolean;
  requestLogId: string | null;
  requestStartedAt: number;
  submissionDispatchedAt?: number | null;
  listenerBoundAt?: number | null;
  firstEventReceivedAt?: number | null;
  firstRuntimeStatusAt?: number | null;
  firstThinkingDeltaAt?: number | null;
  firstTextDeltaAt?: number | null;
  firstTextPaintAt?: number | null;
  firstTextPaintScheduled?: boolean;
  firstTextRenderFlushAt?: number | null;
  lastTextRenderFlushAt?: number | null;
  textDeltaBufferedCount?: number;
  textDeltaFlushCount?: number;
  maxTextDeltaBacklogChars?: number;
  requestFinished: boolean;
  pendingTextRenderTimerId?: ReturnType<typeof setTimeout> | null;
  prefilledMessageSnapshotReplayOffset?: number;
  prefilledMessageSnapshotText?: string | null;
  renderedContent?: string;
  preservedAssistantContentInitialized?: boolean;
  hiddenThinkingPartsCleared?: boolean;
  shouldSurfaceVisibleProcessReasoning?: boolean;
  performanceTrace?: AgentUiPerformanceTraceMetadata | null;
  agentUiEventSequence?: number;
  currentTurnId?: string | null;
  streamedReasoningItemId?: string | null;
  streamedReasoningSourceItemId?: string | null;
  streamedReasoningSummaryIndex?: number | null;
  streamedReasoningText?: string;
  streamedReasoningStartedAt?: string | null;
  streamedReasoningSequence?: number | null;
  streamedReasoningSegmentCounter?: number;
  streamedAgentMessageTextByItemId?: Map<string, string>;
  activeTextSegmentItemId?: string | null;
  activeTextSegmentPhase?: string | null;
  activeTextSegmentSequence?: number | null;
  activeTextSegmentTurnId?: string | null;
  activeTextSegmentStartOffset?: number | null;
  activeTextSegmentFinalEligibility?: TextSegmentFinalEligibility | null;
  latestAssistantTextEventSequence?: number | null;
  maxProcessEventSequence?: number | null;
  maxFinalAnswerRequiredProcessEventSequence?: number | null;
  pendingImageTaskPresentation?: PendingImageTaskPresentation | null;
}

export interface StreamLifecycleCallbacks {
  activateStream: () => void;
  isStreamActivated: () => boolean;
  clearOptimisticItem: () => void;
  clearOptimisticTurn: () => void;
  disposeListener: () => void;
  clearActiveStreamIfMatch: (eventName: string) => boolean;
  appendThinkingToParts: (
    parts: MessageParts,
    textDelta: string,
  ) => MessageParts;
}

export interface HandleTurnStreamEventOptions {
  data: AgentEvent;
  requestState: StreamRequestState;
  callbacks: StreamLifecycleCallbacks;
  observer?: StreamObserver;
  eventName: string;
  pendingTurnKey: string;
  pendingItemKey: string;
  assistantMsgId: string;
  activeSessionId: string;
  resolvedWorkspaceId: string;
  effectiveExecutionStrategy: AgentExecutionStrategy;
  surfaceThinkingDeltas?: boolean;
  preserveAssistantContent?: string | null;
  assistantFallbackContent?: string | null;
  content: string;
  runtime: AgentRuntimeAdapter;
  _webSearch?: boolean;
  warnedKeysRef: MutableRefObject<Set<string>>;
  actionLoggedKeys: Set<string>;
  toolLogIdByToolId: Map<string, string>;
  toolStartedAtByToolId: Map<string, number>;
  toolNameByToolId: Map<string, string>;
  onWriteFile?: (
    content: string,
    fileName: string,
    context?: import("../types").WriteArtifactContext,
  ) => void;
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
