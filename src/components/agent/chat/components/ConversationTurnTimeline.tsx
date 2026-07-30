import React, { useMemo, useState } from "react";
import { Check, Copy } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { A2UIFormData } from "@/components/workspace/a2ui/types";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import { formatDate } from "@/i18n/format";
import type {
  ContentPart,
  Message,
  MessageImage,
  MessagePreviewTarget,
} from "../types";
import type {
  ActionRequired,
  ConfirmResponse,
  SiteSavedContentTarget,
  WriteArtifactContext,
} from "../types";
import type { ArtifactTimelineOpenTarget } from "../utils/artifactTimelineNavigation";
import type { SearchResultPreviewItem } from "../utils/searchResultPreview";
import type {
  CanonicalTurnMediaSegment,
  CanonicalTurnMessageSegment,
  CanonicalTurnRenderEntry,
} from "../projection/turnTimelineRenderProjection";
import {
  imageGenerationContentPartFromThreadItem,
  mediaReferenceContentPartFromThreadItem,
  messageContentPartsFromAgentThreadItem,
} from "../hooks/agentThreadMessageContentParts";
import { ContentColumn, MessageBubble, MessageWrapper } from "../styles";
import { AgentThreadTimeline } from "./AgentThreadTimeline";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { MessageActionButtons } from "./MessageActionButtons";
import { MessageImageAttachments } from "./MessageImageAttachments";
import {
  HistoricalAssistantMessagePreview,
  HistoricalTimelinePreview,
} from "./MessageListHistoricalPreviews";
import { UserInstalledSkillMessageContent } from "./MessageListUserContent";
import {
  MESSAGE_LIST_COMPACT_HISTORICAL_ASSISTANT_PREVIEW_CHARS,
  MESSAGE_LIST_COMPACT_HISTORICAL_ASSISTANT_THRESHOLD,
  MESSAGE_LIST_LONG_HISTORICAL_MESSAGE_PREVIEW_CHARS,
  MESSAGE_LIST_LONG_HISTORICAL_MESSAGE_THRESHOLD,
} from "./messageListConstants";
import { buildHistoricalMessagePreview } from "./messageListHistoricalPreviewText";
import { isTerminalThreadTurnStatus } from "./messageListItemProjectionHelpers";
import { isActiveThreadTurnStatus } from "./messageListProjectionWebRetrieval";
import { StreamingRenderer } from "./StreamingRenderer";

interface ConversationTurnTimelineProps {
  entry: CanonicalTurnRenderEntry;
  sessionId?: string | null;
  threadRead: AgentRuntimeThreadReadModel | null;
  pendingActions: readonly ActionRequired[];
  submittedActionsInFlight: readonly ActionRequired[];
  assistantLabel: string;
  copiedId: string | null;
  compactLeadingSpacing: boolean;
  isSending: boolean;
  renderA2UIInline: boolean;
  a2uiFormDataMap?: Record<string, { formId: string; formData: A2UIFormData }>;
  collapseCodeBlocks?: boolean;
  shouldCollapseCodeBlock?: (language: string, code: string) => boolean;
  focusedTimelineItemId?: string | null;
  timelineFocusRequestKey: number;
  shouldDeferHistoricalTimelineDetails: boolean;
  handleCopy: (content: string, id: string) => void | Promise<void>;
  onA2UISubmit?: (formData: A2UIFormData, messageId: string) => void;
  onA2UIFormChange?: (formId: string, formData: A2UIFormData) => void;
  onWriteFile?: (
    content: string,
    fileName: string,
    context?: WriteArtifactContext,
  ) => void;
  onFileClick?: (fileName: string, content: string) => void;
  onOpenArtifactFromTimeline?: (target: ArtifactTimelineOpenTarget) => void;
  onOpenUrlPreview?: (item: SearchResultPreviewItem) => void;
  onOpenSavedSiteContent?: (target: SiteSavedContentTarget) => void;
  onOpenMessagePreview?: (
    target: MessagePreviewTarget,
    message: Message,
  ) => void;
  onOpenSubagentSession?: (sessionId: string) => void;
  onPermissionResponse?: (response: ConfirmResponse) => void;
  onQuoteMessage?: (content: string, id: string) => void;
  onSaveMessageAsSkill?: (source: {
    messageId: string;
    content: string;
  }) => void;
  onSaveMessageAsKnowledge?: (source: {
    messageId: string;
    content: string;
    sourceName?: string;
    description?: string | null;
  }) => void;
  onCodeBlockClick?: (language: string, code: string) => void;
}

function resolveUserMessageContent(
  segment: CanonicalTurnMessageSegment,
): string {
  if (segment.item.type !== "user_message") return "";
  if (segment.item.content.trim()) return segment.item.content;
  return (segment.item.content_parts ?? [])
    .flatMap((part) => (part.type === "text" ? [part.text] : []))
    .join("\n");
}

function resolveUserMessageLabels(
  segment: CanonicalTurnMessageSegment,
): string[] {
  if (segment.item.type !== "user_message") return [];
  return (segment.item.content_parts ?? []).flatMap((part) =>
    part.type === "skill" || part.type === "mention" ? [part.name] : [],
  );
}

function normalizeInlineImageData(data: string): string {
  const normalized = data.trim();
  if (!normalized.toLowerCase().startsWith("data:")) return normalized;
  const commaIndex = normalized.indexOf(",");
  return commaIndex >= 0 ? normalized.slice(commaIndex + 1) : "";
}

function resolveUserMessageImages(
  segment: CanonicalTurnMessageSegment,
): MessageImage[] {
  if (segment.item.type !== "user_message") return [];
  return (segment.item.content_parts ?? []).flatMap((part, index) => {
    if (part.type !== "image") return [];
    return [
      {
        data: normalizeInlineImageData(part.data),
        mediaType: part.mime_type,
        sourceUri: part.uri,
        sourcePath: part.source_path,
        previewUrl: part.uri,
        index,
      },
    ];
  });
}

function resolveAgentMessageContentParts(
  segment: CanonicalTurnMessageSegment,
  canonicalMediaReferenceKeys: ReadonlySet<string>,
): ContentPart[] | undefined {
  if (segment.item.type !== "agent_message") return undefined;
  const parts = messageContentPartsFromAgentThreadItem(segment.item).filter(
    (part) =>
      part.type !== "media_reference" ||
      !mediaReferenceKeys(part).some((key) =>
        canonicalMediaReferenceKeys.has(key),
      ),
  );
  if (parts.length === 0) return undefined;
  if (segment.item.text.trim() && !parts.some((part) => part.type === "text")) {
    return [{ type: "text", text: segment.item.text }, ...parts];
  }
  return parts;
}

function mediaReferenceKeys(
  part: Extract<ContentPart, { type: "media_reference" }>,
): string[] {
  return [
    part.reference.refId,
    part.reference.uri,
    part.reference.sourcePath,
    part.reference.sourceUri,
  ].flatMap((value) => {
    const normalized = value?.trim();
    return normalized ? [normalized] : [];
  });
}

function resolveMediaContentPart(
  segment: CanonicalTurnMediaSegment,
): ContentPart | null {
  return segment.item.type === "media"
    ? mediaReferenceContentPartFromThreadItem(segment.item)
    : imageGenerationContentPartFromThreadItem(segment.item);
}

function buildCanonicalPreviewMessage(params: {
  entry: CanonicalTurnRenderEntry;
  id: string;
  role?: Message["role"];
  content?: string;
  images?: MessageImage[];
}): Message {
  return {
    id: params.id,
    role: params.role ?? "assistant",
    content: params.content ?? "",
    images: params.images,
    timestamp: new Date(params.entry.turn.started_at),
    runtimeTurnId: params.entry.turn.id,
  };
}

export function ConversationTurnTimeline({
  entry,
  sessionId,
  threadRead,
  pendingActions,
  submittedActionsInFlight,
  assistantLabel,
  copiedId,
  compactLeadingSpacing,
  isSending,
  renderA2UIInline,
  a2uiFormDataMap,
  collapseCodeBlocks,
  shouldCollapseCodeBlock,
  focusedTimelineItemId,
  timelineFocusRequestKey,
  shouldDeferHistoricalTimelineDetails,
  handleCopy,
  onA2UISubmit,
  onA2UIFormChange,
  onWriteFile,
  onFileClick,
  onOpenArtifactFromTimeline,
  onOpenUrlPreview,
  onOpenSavedSiteContent,
  onOpenMessagePreview,
  onOpenSubagentSession,
  onPermissionResponse,
  onQuoteMessage,
  onSaveMessageAsSkill,
  onSaveMessageAsKnowledge,
  onCodeBlockClick,
}: ConversationTurnTimelineProps) {
  const { i18n, t } = useTranslation("agent");
  const [expandedHistoricalProcessIds, setExpandedHistoricalProcessIds] =
    useState<Set<string>>(() => new Set());
  const [
    expandedHistoricalAssistantItemIds,
    setExpandedHistoricalAssistantItemIds,
  ] = useState<Set<string>>(() => new Set());
  const processSegmentIds = useMemo(
    () =>
      entry.segments
        .filter((segment) => segment.kind === "process")
        .map((segment) => segment.id),
    [entry.segments],
  );
  const canonicalMediaReferenceKeys = useMemo(
    () =>
      new Set(
        entry.segments.flatMap((segment) => {
          if (segment.kind !== "media") return [];
          const part = resolveMediaContentPart(segment);
          return part?.type === "media_reference"
            ? mediaReferenceKeys(part)
            : [];
        }),
      ),
    [entry.segments],
  );
  const isActiveOperationalTurn =
    entry.isActive &&
    isActiveThreadTurnStatus(entry.turn.status) &&
    !isTerminalThreadTurnStatus(entry.turn.status);
  const actionRequests = useMemo(() => {
    if (!isActiveOperationalTurn) return [];
    const scoped = [...pendingActions, ...submittedActionsInFlight].filter(
      (action) =>
        action.scope?.turnId === entry.turn.id || !action.scope?.turnId,
    );
    return [
      ...new Map(scoped.map((action) => [action.requestId, action])).values(),
    ];
  }, [
    entry.turn.id,
    isActiveOperationalTurn,
    pendingActions,
    submittedActionsInFlight,
  ]);
  const sourceMessageId = entry.segments.find(
    (segment) =>
      segment.kind === "message" && segment.item.type === "agent_message",
  );
  return (
    <div
      data-testid="conversation-turn-timeline"
      data-runtime-turn-id={entry.turn.id}
      data-runtime-turn-status={entry.turn.status}
    >
      {entry.segments.map((segment) => {
        if (segment.kind === "process") {
          const isFocusedProcess = segment.items.some(
            (item) => item.id === focusedTimelineItemId,
          );
          const showHistoricalDetails =
            expandedHistoricalProcessIds.has(segment.id) || isFocusedProcess;
          return (
            <MessageWrapper
              key={segment.id}
              $isUser={false}
              $compactLeadingSpacing={compactLeadingSpacing}
              data-testid="conversation-turn-process-segment"
              data-direct-segment-kind="process"
              data-segment-id={segment.id}
            >
              <ContentColumn $isUser={false}>
                {isActiveOperationalTurn || showHistoricalDetails ? (
                  <AgentThreadTimeline
                    turn={entry.turn}
                    items={segment.items}
                    threadRead={threadRead}
                    actionRequests={actionRequests}
                    isCurrentTurn={isActiveOperationalTurn}
                    collapseInactiveDetails={
                      isActiveOperationalTurn ? !isSending : false
                    }
                    expandCompletedProcessDetails={!isActiveOperationalTurn}
                    placement="leading"
                    showOperationalDetails={true}
                    showInlineStatusHint={processSegmentIds[0] === segment.id}
                    onFileClick={onFileClick}
                    onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
                    sourceMessageId={
                      sourceMessageId?.kind === "message"
                        ? sourceMessageId.item.id
                        : entry.turn.id
                    }
                    onSaveFileArtifactAsKnowledge={onSaveMessageAsKnowledge}
                    onOpenSavedSiteContent={onOpenSavedSiteContent}
                    onOpenSubagentSession={onOpenSubagentSession}
                    onPermissionResponse={onPermissionResponse}
                    focusedItemId={focusedTimelineItemId}
                    focusRequestKey={timelineFocusRequestKey}
                  />
                ) : (
                  <HistoricalTimelinePreview
                    items={segment.items}
                    placement="leading"
                    detailsDeferred={shouldDeferHistoricalTimelineDetails}
                    startedAt={entry.turn.started_at}
                    completedAt={entry.turn.completed_at}
                    onFileClick={onFileClick}
                    onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
                    onOpenSavedSiteContent={onOpenSavedSiteContent}
                    onOpenSubagentSession={onOpenSubagentSession}
                    onPermissionResponse={onPermissionResponse}
                    onSaveFileArtifactAsKnowledge={onSaveMessageAsKnowledge}
                    sourceMessageId={
                      sourceMessageId?.kind === "message"
                        ? sourceMessageId.item.id
                        : entry.turn.id
                    }
                    onExpand={() => {
                      setExpandedHistoricalProcessIds((current) => {
                        const next = new Set(current);
                        next.add(segment.id);
                        return next;
                      });
                    }}
                  />
                )}
              </ContentColumn>
            </MessageWrapper>
          );
        }

        if (segment.kind === "media") {
          const contentPart = resolveMediaContentPart(segment);
          const previewMessage = buildCanonicalPreviewMessage({
            entry,
            id: segment.item.id,
          });
          return (
            <MessageWrapper
              key={segment.id}
              $isUser={false}
              $compactLeadingSpacing={compactLeadingSpacing}
              data-direct-segment-kind="media"
            >
              <ContentColumn $isUser={false}>
                <MessageBubble
                  $isUser={false}
                  data-message-id={segment.item.id}
                  data-message-role="media"
                  data-runtime-turn-id={entry.turn.id}
                  data-thread-item-id={segment.item.id}
                  data-visual-tone="neutral-assistant"
                  aria-label={assistantLabel}
                >
                  {contentPart ? (
                    <StreamingRenderer
                      content=""
                      rawContent=""
                      contentParts={[contentPart]}
                      isStreaming={false}
                      suppressProcessFlow={true}
                      renderA2UIInline={false}
                      readOnlyA2UI={true}
                      readOnlyActionRequests={true}
                      onOpenMediaReference={
                        onOpenMessagePreview
                          ? (reference, index) =>
                              onOpenMessagePreview(
                                { kind: "media_reference", reference, index },
                                previewMessage,
                              )
                          : undefined
                      }
                    />
                  ) : (
                    <AgentThreadTimeline
                      turn={entry.turn}
                      items={[segment.item]}
                      threadRead={threadRead}
                      isCurrentTurn={false}
                      placement="leading"
                      showOperationalDetails={true}
                      showInlineStatusHint={false}
                      expandCompletedProcessDetails={true}
                      focusedItemId={focusedTimelineItemId}
                      focusRequestKey={timelineFocusRequestKey}
                    />
                  )}
                </MessageBubble>
              </ContentColumn>
            </MessageWrapper>
          );
        }

        const isUser = segment.item.type === "user_message";
        const content =
          segment.item.type === "user_message"
            ? resolveUserMessageContent(segment)
            : segment.item.text;
        const canSave = !isUser && content.trim().length >= 24;
        const isStreaming =
          !isUser &&
          entry.isActive &&
          entry.turn.status === "running" &&
          segment.item.status === "in_progress";
        const userLabels = isUser ? resolveUserMessageLabels(segment) : [];
        const userImages = isUser ? resolveUserMessageImages(segment) : [];
        const previewMessage = buildCanonicalPreviewMessage({
          entry,
          id: segment.item.id,
          role: isUser ? "user" : "assistant",
          content,
          images: userImages,
        });
        const agentContentParts = isUser
          ? undefined
          : resolveAgentMessageContentParts(
              segment,
              canonicalMediaReferenceKeys,
            );
        const hasNonTextAgentContent = Boolean(
          agentContentParts?.some((part) => part.type !== "text"),
        );
        const shouldPreviewHistoricalAssistant =
          !isUser &&
          !isStreaming &&
          shouldDeferHistoricalTimelineDetails &&
          content.length >
            MESSAGE_LIST_COMPACT_HISTORICAL_ASSISTANT_THRESHOLD &&
          !content.includes("```a2ui") &&
          !hasNonTextAgentContent &&
          !expandedHistoricalAssistantItemIds.has(segment.item.id);
        const isLongHistoricalAssistant =
          content.length > MESSAGE_LIST_LONG_HISTORICAL_MESSAGE_THRESHOLD;
        const historicalAssistantPreview = shouldPreviewHistoricalAssistant
          ? buildHistoricalMessagePreview(
              content,
              isLongHistoricalAssistant
                ? MESSAGE_LIST_LONG_HISTORICAL_MESSAGE_PREVIEW_CHARS
                : MESSAGE_LIST_COMPACT_HISTORICAL_ASSISTANT_PREVIEW_CHARS,
            )
          : "";

        return (
          <MessageWrapper
            key={segment.id}
            $isUser={isUser}
            $compactLeadingSpacing={compactLeadingSpacing}
          >
            <ContentColumn $isUser={isUser}>
              <MessageBubble
                $isUser={isUser}
                data-message-id={segment.item.id}
                data-message-role={isUser ? "user" : "assistant"}
                data-runtime-turn-id={entry.turn.id}
                data-thread-item-id={segment.item.id}
                data-direct-segment-kind="message"
                data-visual-tone={isUser ? "neutral-user" : "neutral-assistant"}
                aria-label={isUser ? undefined : assistantLabel}
              >
                {isUser ? (
                  <div className="space-y-2">
                    {userLabels.map((label, index) => (
                      <UserInstalledSkillMessageContent
                        key={`${label}:${index}`}
                        content=""
                        label={label}
                      />
                    ))}
                    {content ? (
                      <MarkdownRenderer
                        content={content}
                        renderA2UIInline={renderA2UIInline}
                        readOnlyA2UI={!isActiveOperationalTurn}
                      />
                    ) : null}
                    <MessageImageAttachments
                      images={userImages}
                      threadId={threadRead?.thread_id}
                      onOpenImage={
                        onOpenMessagePreview
                          ? (attachment, index) =>
                              onOpenMessagePreview(
                                {
                                  kind: "message_attachment",
                                  attachment,
                                  index,
                                },
                                previewMessage,
                              )
                          : undefined
                      }
                    />
                  </div>
                ) : shouldPreviewHistoricalAssistant ? (
                  <HistoricalAssistantMessagePreview
                    content={historicalAssistantPreview}
                    contentLength={content.length}
                    variant={isLongHistoricalAssistant ? "long" : "compact"}
                    onExpand={() => {
                      setExpandedHistoricalAssistantItemIds((current) => {
                        const next = new Set(current);
                        next.add(segment.item.id);
                        return next;
                      });
                    }}
                  />
                ) : (
                  <StreamingRenderer
                    content={content}
                    rawContent={content}
                    isStreaming={isStreaming}
                    contentParts={agentContentParts}
                    showCursor={isStreaming}
                    suppressProcessFlow={true}
                    renderA2UIInline={renderA2UIInline}
                    readOnlyA2UI={!isActiveOperationalTurn}
                    readOnlyActionRequests={!isActiveOperationalTurn}
                    onA2UISubmit={
                      onA2UISubmit
                        ? (formData) => onA2UISubmit(formData, segment.item.id)
                        : undefined
                    }
                    a2uiFormId={a2uiFormDataMap?.[segment.item.id]?.formId}
                    a2uiInitialFormData={
                      a2uiFormDataMap?.[segment.item.id]?.formData
                    }
                    onA2UIFormChange={onA2UIFormChange}
                    onWriteFile={
                      onWriteFile
                        ? (fileContent, fileName, context) =>
                            onWriteFile(fileContent, fileName, {
                              ...context,
                              sourceMessageId:
                                context?.sourceMessageId || segment.item.id,
                              source: context?.source || "message_content",
                            })
                        : undefined
                    }
                    onFileClick={onFileClick}
                    fileChangesUndoSessionId={sessionId}
                    onOpenSavedSiteContent={onOpenSavedSiteContent}
                    onOpenUrlPreview={onOpenUrlPreview}
                    onOpenMediaReference={
                      onOpenMessagePreview
                        ? (reference, index) =>
                            onOpenMessagePreview(
                              { kind: "media_reference", reference, index },
                              previewMessage,
                            )
                        : undefined
                    }
                    onPermissionResponse={onPermissionResponse}
                    collapseCodeBlocks={collapseCodeBlocks}
                    shouldCollapseCodeBlock={shouldCollapseCodeBlock}
                    onCodeBlockClick={onCodeBlockClick}
                  />
                )}

                {!isUser && content.trim() ? (
                  <MessageActionButtons
                    actionContent={content}
                    canCopyMessage={true}
                    canQuoteMessage={Boolean(onQuoteMessage)}
                    canSaveMessageAsKnowledge={
                      canSave && Boolean(onSaveMessageAsKnowledge)
                    }
                    canSaveMessageAsSkill={
                      canSave && Boolean(onSaveMessageAsSkill)
                    }
                    copied={copiedId === segment.item.id}
                    isImageWorkbenchMessage={false}
                    messageId={segment.item.id}
                    onCopy={handleCopy}
                    onQuoteMessage={onQuoteMessage}
                    onSaveMessageAsKnowledge={onSaveMessageAsKnowledge}
                    onSaveMessageAsSkill={onSaveMessageAsSkill}
                  />
                ) : null}
              </MessageBubble>

              {isUser ? (
                <div
                  className="user-message-footer flex items-center justify-end gap-2 pr-1 text-xs leading-5 text-slate-400"
                  data-testid="user-message-footer"
                >
                  <span data-testid="user-message-timestamp">
                    {formatDate(new Date(segment.item.started_at), {
                      locale: i18n.language,
                      weekday: "long",
                      hour: "2-digit",
                      minute: "2-digit",
                      hour12: false,
                    })}
                  </span>
                  {content.trim() ? (
                    <button
                      type="button"
                      className="inline-flex h-6 w-6 items-center justify-center rounded-md text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
                      onClick={() => void handleCopy(content, segment.item.id)}
                      aria-label={t("agentChat.messageList.actions.copy")}
                      title={t("agentChat.messageList.actions.copy")}
                    >
                      {copiedId === segment.item.id ? (
                        <Check size={13} />
                      ) : (
                        <Copy size={13} />
                      )}
                    </button>
                  ) : null}
                </div>
              ) : null}
            </ContentColumn>
          </MessageWrapper>
        );
      })}
    </div>
  );
}
