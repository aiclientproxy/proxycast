import { useMemo } from "react";
import type { WriteArtifactContext } from "../types";
import type { AgentSessionReadModelRefreshRequest } from "../hooks/agentSessionRefresh";
import type { WorkspaceConversationMessageListRuntime } from "./useWorkspaceConversationSceneRuntime";

type WorkspaceConversationMessageListProvider =
  WorkspaceConversationMessageListRuntime["providerType"];
type WorkspaceConversationMessageListAccessMode =
  WorkspaceConversationMessageListRuntime["accessMode"];
type WorkspaceConversationMessageListMessages =
  WorkspaceConversationMessageListRuntime["messages"];
type WorkspaceConversationMessageListTurns =
  WorkspaceConversationMessageListRuntime["turns"];
type WorkspaceConversationMessageListThreadItems =
  WorkspaceConversationMessageListRuntime["threadItems"];
type WorkspaceConversationMessageListThreadRead =
  WorkspaceConversationMessageListRuntime["threadRead"];
type WorkspaceConversationMessageListExecutionRuntime =
  WorkspaceConversationMessageListRuntime["executionRuntime"];
type WorkspaceConversationMessageListPendingActions =
  WorkspaceConversationMessageListRuntime["pendingActions"];
type WorkspaceConversationMessageListSubmittedActions =
  WorkspaceConversationMessageListRuntime["submittedActionsInFlight"];
type WorkspaceConversationMessageListSessionHistoryWindow =
  WorkspaceConversationMessageListRuntime["sessionHistoryWindow"];
type WorkspaceConversationMessageListPendingA2UIAction =
  WorkspaceConversationMessageListRuntime["pendingPromotedA2UIActionRequest"];

interface WorkspaceConversationMessageListProviderInput {
  accessMode: WorkspaceConversationMessageListAccessMode;
  model: string | null | undefined;
  providerType: WorkspaceConversationMessageListProvider;
  reasoningEffort?: string | null;
}

interface WorkspaceConversationMessageListProjectionInput {
  currentTurnId: WorkspaceConversationMessageListRuntime["currentTurnId"];
  executionRuntime: WorkspaceConversationMessageListExecutionRuntime;
  isSending: WorkspaceConversationMessageListRuntime["isSending"];
  messages: WorkspaceConversationMessageListMessages;
  pendingActions: WorkspaceConversationMessageListPendingActions;
  sessionHistoryWindow: WorkspaceConversationMessageListSessionHistoryWindow;
  submittedActionsInFlight: WorkspaceConversationMessageListSubmittedActions;
  threadItems: WorkspaceConversationMessageListThreadItems;
  threadRead: WorkspaceConversationMessageListThreadRead;
  todoItems: WorkspaceConversationMessageListRuntime["todoItems"];
  turns: WorkspaceConversationMessageListTurns;
}

interface WorkspaceConversationMessageListActionsInput {
  onA2UISubmit: WorkspaceConversationMessageListRuntime["onA2UISubmit"];
  onArtifactClick: WorkspaceConversationMessageListRuntime["onArtifactClick"];
  onCodeBlockClick: WorkspaceConversationMessageListRuntime["onCodeBlockClick"];
  onDeleteMessage: WorkspaceConversationMessageListRuntime["onDeleteMessage"];
  onEditMessage: WorkspaceConversationMessageListRuntime["onEditMessage"];
  onFileClick: WorkspaceConversationMessageListRuntime["onFileClick"];
  onInterruptCurrentTurn: WorkspaceConversationMessageListRuntime["onInterruptCurrentTurn"];
  onLoadFullHistory: () => void;
  onOpenArtifactFromTimeline: WorkspaceConversationMessageListRuntime["onOpenArtifactFromTimeline"];
  onOpenMessagePreview: WorkspaceConversationMessageListRuntime["onOpenMessagePreview"];
  onOpenSavedSiteContent: WorkspaceConversationMessageListRuntime["onOpenSavedSiteContent"];
  onOpenSubagentSession: WorkspaceConversationMessageListRuntime["onOpenSubagentSession"];
  onOpenUrlPreview: WorkspaceConversationMessageListRuntime["onOpenUrlPreview"];
  onPermissionResponse: WorkspaceConversationMessageListRuntime["onPermissionResponse"];
  onReplayPendingRequest: WorkspaceConversationMessageListRuntime["onReplayPendingRequest"];
  onSaveMessageAsKnowledge: WorkspaceConversationMessageListRuntime["onSaveMessageAsKnowledge"];
  onSaveMessageAsSkill: WorkspaceConversationMessageListRuntime["onSaveMessageAsSkill"];
  onWriteFile: (
    content: string,
    fileName: string,
    context?: WriteArtifactContext,
  ) => void | Promise<void>;
}

interface WorkspaceConversationMessageListInputBinding {
  onQuoteInputChange: WorkspaceConversationMessageListRuntime["onQuoteInputChange"];
  quoteInput: string;
}

interface WorkspaceConversationMessageListFocusInput {
  focusedTimelineItemId: string | null;
  timelineFocusRequestKey: number;
}

interface UseWorkspaceConversationMessageListRuntimeParams {
  actions: WorkspaceConversationMessageListActionsInput;
  collapseCodeBlocks: WorkspaceConversationMessageListRuntime["collapseCodeBlocks"];
  emptyStateVariant?: WorkspaceConversationMessageListRuntime["emptyStateVariant"];
  focus: WorkspaceConversationMessageListFocusInput;
  input: WorkspaceConversationMessageListInputBinding;
  pendingPromotedA2UIActionRequest: WorkspaceConversationMessageListPendingA2UIAction;
  projection: WorkspaceConversationMessageListProjectionInput;
  provider: WorkspaceConversationMessageListProviderInput;
  refreshSessionReadModel: (
    sessionId?: string,
    request?: AgentSessionReadModelRefreshRequest,
  ) => void | Promise<unknown>;
  sceneSessionId?: string | null;
  shouldCollapseCodeBlock: WorkspaceConversationMessageListRuntime["shouldCollapseCodeBlock"];
}

export function useWorkspaceConversationMessageListRuntime({
  actions,
  collapseCodeBlocks,
  emptyStateVariant,
  focus,
  input,
  pendingPromotedA2UIActionRequest,
  projection,
  provider,
  refreshSessionReadModel,
  sceneSessionId,
  shouldCollapseCodeBlock,
}: UseWorkspaceConversationMessageListRuntimeParams): WorkspaceConversationMessageListRuntime {
  return useMemo(
    () => ({
      emptyStateVariant,
      quoteInput: input.quoteInput,
      onQuoteInputChange: input.onQuoteInputChange,
      providerType: provider.providerType,
      model: provider.model,
      reasoningEffort: provider.reasoningEffort,
      accessMode: provider.accessMode,
      messages: projection.messages,
      turns: projection.turns,
      threadItems: projection.threadItems,
      todoItems: projection.todoItems,
      currentTurnId: projection.currentTurnId,
      threadRead: projection.threadRead,
      executionRuntime: projection.executionRuntime,
      pendingActions: projection.pendingActions,
      submittedActionsInFlight: projection.submittedActionsInFlight,
      sessionHistoryWindow: projection.sessionHistoryWindow,
      onLoadFullHistory: actions.onLoadFullHistory,
      isSending: projection.isSending,
      onInterruptCurrentTurn: actions.onInterruptCurrentTurn,
      onReplayPendingRequest: actions.onReplayPendingRequest,
      onDeleteMessage: actions.onDeleteMessage,
      onEditMessage: actions.onEditMessage,
      onA2UISubmit: actions.onA2UISubmit,
      onWriteFile: actions.onWriteFile,
      onFileClick: actions.onFileClick,
      onOpenArtifactFromTimeline: actions.onOpenArtifactFromTimeline,
      onOpenSavedSiteContent: actions.onOpenSavedSiteContent,
      onArtifactClick: actions.onArtifactClick,
      onOpenUrlPreview: actions.onOpenUrlPreview,
      onOpenMessagePreview: actions.onOpenMessagePreview,
      onSaveMessageAsSkill: actions.onSaveMessageAsSkill,
      onSaveMessageAsKnowledge: actions.onSaveMessageAsKnowledge,
      onOpenSubagentSession: actions.onOpenSubagentSession,
      onPermissionResponse: actions.onPermissionResponse,
      onRefreshSessionReadModel: (request) =>
        refreshSessionReadModel(sceneSessionId || undefined, request),
      pendingPromotedA2UIActionRequest,
      collapseCodeBlocks,
      shouldCollapseCodeBlock,
      onCodeBlockClick: actions.onCodeBlockClick,
      focusedTimelineItemId: focus.focusedTimelineItemId,
      timelineFocusRequestKey: focus.timelineFocusRequestKey,
    }),
    [
      actions,
      collapseCodeBlocks,
      emptyStateVariant,
      focus.focusedTimelineItemId,
      focus.timelineFocusRequestKey,
      input.onQuoteInputChange,
      input.quoteInput,
      pendingPromotedA2UIActionRequest,
      projection,
      provider.accessMode,
      provider.model,
      provider.providerType,
      provider.reasoningEffort,
      refreshSessionReadModel,
      sceneSessionId,
      shouldCollapseCodeBlock,
    ],
  );
}
