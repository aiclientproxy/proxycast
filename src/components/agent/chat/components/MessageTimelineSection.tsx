import { useState } from "react";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import { AgentThreadTimeline } from "./AgentThreadTimeline";
import { HistoricalTimelinePreview } from "./MessageListHistoricalPreviews";
import type { MessageListRenderGroup } from "./MessageList.types";
import type { ArtifactTimelineOpenTarget } from "../utils/artifactTimelineNavigation";
import type {
  ConfirmResponse,
  Message,
  SiteSavedContentTarget,
} from "../types";
import { isTerminalThreadTurnStatus } from "./messageListItemProjectionHelpers";
import { isActiveThreadTurnStatus } from "./messageListProjectionWebRetrieval";

type MessageTimelineProjection = NonNullable<
  MessageListRenderGroup["timeline"]
>;

interface MessageTimelineSectionProps {
  actionRequests: Message["actionRequests"] | undefined;
  activeCurrentTurnId: string | null;
  detailsDeferred?: boolean;
  expandCompletedProcessDetails?: boolean;
  focusedTimelineItemId?: string | null;
  focusRequestKey: number;
  isCurrentTurnSending: boolean;
  messageId: string;
  onFileClick?: (fileName: string, content: string) => void;
  onOpenArtifactFromTimeline?: (target: ArtifactTimelineOpenTarget) => void;
  onOpenSavedSiteContent?: (target: SiteSavedContentTarget) => void;
  onOpenSubagentSession?: (sessionId: string) => void;
  onPermissionResponse?: (response: ConfirmResponse) => void;
  onSaveMessageAsKnowledge?: (source: {
    messageId: string;
    content: string;
    sourceName?: string;
    description?: string | null;
  }) => void;
  placement: "leading" | "trailing";
  renderCompactPreview?: boolean;
  shouldDeferHistoricalTimelineDetails: boolean;
  threadRead: AgentRuntimeThreadReadModel | null;
  timeline: MessageTimelineProjection;
}

export function MessageTimelineSection({
  actionRequests,
  activeCurrentTurnId,
  detailsDeferred = false,
  expandCompletedProcessDetails = false,
  focusedTimelineItemId,
  focusRequestKey,
  isCurrentTurnSending,
  messageId,
  onFileClick,
  onOpenArtifactFromTimeline,
  onOpenSavedSiteContent,
  onOpenSubagentSession,
  onPermissionResponse,
  onSaveMessageAsKnowledge,
  placement,
  renderCompactPreview = false,
  shouldDeferHistoricalTimelineDetails,
  threadRead,
  timeline,
}: MessageTimelineSectionProps) {
  const [expandedTimelineKey, setExpandedTimelineKey] = useState<string | null>(
    null,
  );
  const timelineKey = `${placement}:${timeline.turn.id}`;
  const isActiveOperationalTurn =
    timeline.turn.id === activeCurrentTurnId &&
    isActiveThreadTurnStatus(timeline.turn.status) &&
    !isTerminalThreadTurnStatus(timeline.turn.status);
  const shouldRenderHistoricalCompactPreview =
    renderCompactPreview || !isActiveOperationalTurn;
  const shouldExpandHistoricalTimeline =
    shouldRenderHistoricalCompactPreview && expandedTimelineKey === timelineKey;

  if (shouldRenderHistoricalCompactPreview && !shouldExpandHistoricalTimeline) {
    return (
      <HistoricalTimelinePreview
        items={timeline.items}
        placement={placement}
        detailsDeferred={detailsDeferred}
        startedAt={timeline.turn.started_at}
        completedAt={timeline.turn.completed_at}
        onFileClick={onFileClick}
        onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
        onOpenSavedSiteContent={onOpenSavedSiteContent}
        onOpenSubagentSession={onOpenSubagentSession}
        onPermissionResponse={onPermissionResponse}
        onSaveFileArtifactAsKnowledge={onSaveMessageAsKnowledge}
        sourceMessageId={messageId}
        onExpand={() => setExpandedTimelineKey(timelineKey)}
      />
    );
  }

  return (
    <AgentThreadTimeline
      turn={timeline.turn}
      items={timeline.items}
      threadRead={threadRead}
      actionRequests={actionRequests}
      isCurrentTurn={timeline.turn.id === activeCurrentTurnId}
      collapseInactiveDetails={!isCurrentTurnSending}
      expandCompletedProcessDetails={expandCompletedProcessDetails}
      showOperationalDetails={true}
      deferCompletedSingleDetails={
        shouldDeferHistoricalTimelineDetails &&
        timeline.turn.id !== activeCurrentTurnId
      }
      placement={placement}
      onFileClick={onFileClick}
      onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
      sourceMessageId={messageId}
      onSaveFileArtifactAsKnowledge={onSaveMessageAsKnowledge}
      onOpenSavedSiteContent={onOpenSavedSiteContent}
      onOpenSubagentSession={onOpenSubagentSession}
      onPermissionResponse={onPermissionResponse}
      focusedItemId={focusedTimelineItemId}
      focusRequestKey={focusRequestKey}
    />
  );
}
