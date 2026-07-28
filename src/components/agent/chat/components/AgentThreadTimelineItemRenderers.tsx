import React, { useMemo } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import {
  AlertTriangle,
  Bot,
  Clock3,
  ListChecks,
  Loader2,
  ShieldAlert,
  Sparkles,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type {
  AgentThreadItem,
  ConfirmResponse,
  SiteSavedContentTarget,
} from "../types";
import type { AgentThreadOrderedBlock } from "../utils/agentThreadGrouping";
import type { ArtifactTimelineOpenTarget } from "../utils/artifactTimelineNavigation";
import { isActionRequestA2UICompatible } from "../utils/actionRequestA2UI";
import { resolveRuntimeAttachmentTaskDisplayName } from "../utils/runtimeAttachmentPlaceholder";
import { isHiddenConversationArtifactPath } from "../utils/internalArtifactVisibility";
import { shouldHideTurnSummaryFromConversation } from "../utils/turnSummaryPresentation";
import { parseAIResponse } from "@/components/workspace/a2ui/parser";
import type { A2UIResponse } from "@/components/workspace/a2ui/types";
import { TIMELINE_A2UI_TASK_CARD_PRESET } from "@/components/workspace/a2ui/taskCardPresets";
import { cn } from "@/lib/utils";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { ActionRequestA2UIPreviewCard } from "./ActionRequestA2UIPreviewCard";
import { A2UITaskCard, A2UITaskLoadingCard } from "./A2UITaskCard";
import { ToolCallItem } from "./ToolCallDisplay";
import { DecisionPanel } from "./DecisionPanel";
import { AgentThreadTimelineArtifactCard } from "./AgentThreadTimelineArtifactCard";
import { ApprovalRecordCard } from "./ApprovalRecordCard";
import {
  toActionRequired,
  toApprovalRecordFromThreadItem,
  toToolCallState,
  resolveStatusBadgeVariant,
  resolveItemStatusLabel,
  resolveSubagentStatusBadgeVariant,
  resolveSubagentStatusLabel,
  formatTimestamp,
  resolveUserFacingErrorMessage,
  resolveThinkingDisplayText,
  resolveTurnSummaryDisplayText,
  extractCompactThinkingParts,
  isThinkingTimelineItem,
  resolveThreadInlineStatusHint,
  resolveTimelineAlertFallback,
  resolveTimelineContextCompactionParts,
  resolveTimelinePlanTitle,
  resolveTimelineReasoningTitle,
  resolveTimelineSubagentDefaultTitle,
  resolveTimelineSubagentTitle,
  resolveTimelineTurnSummaryTitle,
} from "./timeline-utils";

type AgentTranslate = TFunction<"agent", undefined>;

function SurfaceCard({
  icon: Icon,
  title,
  badge,
  timestamp,
  children,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  badge?: React.ReactNode;
  timestamp?: string | null;
  children: React.ReactNode;
}) {
  return (
    <div className="py-1.5">
      <div className="flex items-start gap-2.5">
        <div className="flex h-5 w-5 shrink-0 items-center justify-center text-slate-400">
          <Icon className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <div className="text-sm leading-6 text-slate-900">{title}</div>
            {badge ? <div>{badge}</div> : null}
            {timestamp ? (
              <div className="text-xs text-slate-400">{timestamp}</div>
            ) : null}
          </div>
          <div className="ml-0 mt-1.5">{children}</div>
        </div>
      </div>
    </div>
  );
}

function ThinkingItemCard({
  item,
}: {
  item: Extract<AgentThreadItem, { type: "reasoning" | "turn_summary" }>;
}) {
  const { t } = useTranslation("agent");
  const displayText = useMemo(
    () =>
      item.type === "reasoning"
        ? resolveThinkingDisplayText(item)
        : resolveTurnSummaryDisplayText(item),
    [item],
  );
  const parsedContent = useMemo(
    () => parseAIResponse(displayText, false),
    [displayText],
  );
  const hasStructuredPreview =
    parsedContent.hasA2UI || parsedContent.hasPending;
  const shouldHideTurnSummaryContent =
    item.type === "turn_summary" &&
    !hasStructuredPreview &&
    shouldHideTurnSummaryFromConversation(item);

  const content = hasStructuredPreview ? (
    <div className="space-y-3">
      {parsedContent.parts.map((part, index) => {
        if (part.type === "a2ui" && typeof part.content !== "string") {
          const readonlyResponse: A2UIResponse = {
            ...part.content,
            submitAction: undefined,
          };

          return (
            <A2UITaskCard
              key={`timeline-a2ui-${index}`}
              response={readonlyResponse}
              compact={true}
              preview={true}
              preset={TIMELINE_A2UI_TASK_CARD_PRESET}
            />
          );
        }

        if (part.type === "pending_a2ui") {
          return (
            <A2UITaskLoadingCard
              key={`timeline-pending-a2ui-${index}`}
              compact={true}
              preset={TIMELINE_A2UI_TASK_CARD_PRESET}
              subtitle={t("agentChat.threadTimeline.pendingA2ui")}
            />
          );
        }

        const textContent =
          typeof part.content === "string" ? part.content.trim() : "";
        if (!textContent) {
          return null;
        }

        return (
          <MarkdownRenderer
            key={`timeline-text-${index}`}
            content={textContent}
          />
        );
      })}
    </div>
  ) : shouldHideTurnSummaryContent ? null : (
    <MarkdownRenderer content={displayText} />
  );
  const statusLabel =
    item.type === "turn_summary"
      ? resolveTimelineTurnSummaryTitle(item.status)
      : item.status === "in_progress"
        ? resolveTimelineReasoningTitle(item.status)
        : null;
  const ToneIcon =
    item.type === "turn_summary"
      ? item.status === "in_progress"
        ? Loader2
        : Clock3
      : item.status === "in_progress"
        ? Loader2
        : Sparkles;

  return (
    <div className="py-1.5">
      <div className="flex items-start gap-2.5">
        <div className="flex h-5 w-5 shrink-0 items-center justify-center text-slate-400">
          <ToneIcon
            className={cn(
              "h-4 w-4",
              item.status === "in_progress" && "animate-spin text-sky-600",
            )}
          />
        </div>
        <div className="min-w-0 flex-1">
          {statusLabel ? (
            <div className="mb-1 text-xs text-slate-500">{statusLabel}</div>
          ) : null}
          {content ? (
            <div className="text-sm leading-7 text-slate-800">{content}</div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ContextCompactionCard({
  item,
}: {
  item: Extract<AgentThreadItem, { type: "context_compaction" }>;
}) {
  const { t } = useTranslation("agent");
  const { detail, title, triggerLabel } =
    resolveTimelineContextCompactionParts(item);

  return (
    <SurfaceCard
      icon={Sparkles}
      title={title}
      badge={<Badge variant="outline">{triggerLabel}</Badge>}
    >
      <div className="text-sm text-slate-500">{detail}</div>
      {item.status === "in_progress" ? (
        <div className="mt-1 inline-flex items-center gap-1 text-xs text-slate-500">
          <Loader2 className="h-3 w-3 animate-spin" />
          <span>{t("agentChat.threadTimeline.compacting")}</span>
        </div>
      ) : null}
    </SurfaceCard>
  );
}

function InlinePlanBlock({
  content,
  isComplete,
}: {
  content: string;
  isComplete: boolean;
}) {
  if (!content.trim()) {
    return null;
  }

  return (
    <div className="py-1.5">
      <div className="flex items-start gap-2.5">
        <div className="flex h-5 w-5 shrink-0 items-center justify-center text-slate-400">
          {isComplete ? (
            <ListChecks className="h-4 w-4" />
          ) : (
            <Loader2 className="h-4 w-4 animate-spin text-sky-600" />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div className="mb-1 text-xs text-slate-500">
            {resolveTimelinePlanTitle(isComplete ? "completed" : "in_progress")}
          </div>
          <div className="text-sm leading-7 text-slate-800">
            <MarkdownRenderer content={content} />
          </div>
        </div>
      </div>
    </div>
  );
}

function renderThinkingItemDetails(item: AgentThreadItem) {
  if (item.type === "plan") {
    return (
      <InlinePlanBlock
        content={item.text}
        isComplete={item.status !== "in_progress"}
      />
    );
  }

  if (item.type === "reasoning" || item.type === "turn_summary") {
    return <ThinkingItemCard item={item} />;
  }

  if (item.type === "context_compaction") {
    return <ContextCompactionCard item={item} />;
  }

  return null;
}

function GroupedThinkingRow({
  item,
  groupMarker = "·",
}: {
  item: Extract<
    AgentThreadItem,
    { type: "plan" | "reasoning" | "turn_summary" | "context_compaction" }
  >;
  groupMarker?: string;
}) {
  const compact = extractCompactThinkingParts(item);

  if (!compact) {
    return renderThinkingItemDetails(item);
  }

  const ToneIcon =
    item.type === "plan"
      ? item.status === "in_progress"
        ? Loader2
        : ListChecks
      : item.type === "turn_summary"
        ? item.status === "in_progress"
          ? Loader2
          : Clock3
        : item.status === "in_progress"
          ? Loader2
          : Sparkles;

  return (
    <div className="flex items-start gap-2 py-1.5">
      <span className="pt-0.5 font-mono text-xs text-slate-400">
        {groupMarker}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-start gap-2">
          <ToneIcon
            className={cn(
              "mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-400",
              item.status === "in_progress" && "animate-spin text-sky-600",
            )}
          />
          <div className="min-w-0 flex-1">
            <div className="text-sm leading-6 text-slate-700">
              {compact.title}
            </div>
            {compact.detail ? (
              <div className="mt-0.5 text-sm leading-6 text-slate-500">
                <MarkdownRenderer content={compact.detail} />
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

function renderGroupItemDetails(
  item: AgentThreadItem,
  t: AgentTranslate,
  onFileClick?: (fileName: string, content: string) => void,
  onOpenArtifactFromTimeline?: (target: ArtifactTimelineOpenTarget) => void,
  onOpenSavedSiteContent?: (target: SiteSavedContentTarget) => void,
  onOpenSubagentSession?: (sessionId: string) => void,
  onPermissionResponse?: (response: ConfirmResponse) => void,
  options?: {
    defaultExpandCompletedToolResult?: boolean;
    groupedToolCall?: boolean;
    groupMarker?: string;
    openSubagentLabel?: string;
    sourceMessageId?: string;
    onSaveFileArtifactAsKnowledge?: (source: {
      messageId: string;
      content: string;
      sourceName?: string;
      description?: string | null;
    }) => void;
  },
) {
  const toolCall = toToolCallState(item);
  const actionRequest = toActionRequired(item);
  const timestamp = formatTimestamp(item.completed_at || item.updated_at);

  if (item.type === "approval_request") {
    if (item.status !== "completed" && item.status !== "failed") {
      return null;
    }
    const approvalRecord = toApprovalRecordFromThreadItem(item);
    return approvalRecord ? (
      <ApprovalRecordCard record={approvalRecord} />
    ) : null;
  }

  if (actionRequest) {
    if (
      actionRequest.actionType === "tool_confirmation" &&
      actionRequest.status !== "submitted"
    ) {
      return null;
    }

    if (isActionRequestA2UICompatible(actionRequest)) {
      return (
        <ActionRequestA2UIPreviewCard
          request={actionRequest}
          compact={true}
          context="timeline"
        />
      );
    }

    return (
      <DecisionPanel
        request={actionRequest}
        onSubmit={(response) => onPermissionResponse?.(response)}
      />
    );
  }

  if (toolCall) {
    return (
      <ToolCallItem
        toolCall={toolCall}
        defaultExpanded={
          item.status !== "completed" ||
          Boolean(options?.defaultExpandCompletedToolResult)
        }
        onFileClick={onFileClick}
        onOpenSavedSiteContent={onOpenSavedSiteContent}
        grouped={options?.groupedToolCall}
        groupMarker={options?.groupMarker}
      />
    );
  }

  if (item.type === "file_artifact") {
    if (isHiddenConversationArtifactPath(item.path)) {
      return null;
    }

    return (
      <AgentThreadTimelineArtifactCard
        item={item}
        timestamp={timestamp}
        onFileClick={onFileClick}
        onOpenArtifactFromTimeline={onOpenArtifactFromTimeline}
        sourceMessageId={options?.sourceMessageId}
        onSaveFileArtifactAsKnowledge={options?.onSaveFileArtifactAsKnowledge}
      />
    );
  }

  if (item.type === "subagent_activity") {
    const subagentThreadId = item.session_id?.trim();
    const displayTitle =
      resolveRuntimeAttachmentTaskDisplayName(item.title) ||
      resolveTimelineSubagentDefaultTitle();

    return (
      <div
        data-testid="subagent-activity-row"
        data-subagent-activity-item-id={item.id}
        data-subagent-activity-kind={item.status_label?.trim().toLowerCase()}
        data-subagent-thread-id={subagentThreadId || undefined}
      >
        <SurfaceCard
          icon={Bot}
          title={resolveTimelineSubagentTitle(displayTitle)}
          badge={
            <Badge
              variant={resolveSubagentStatusBadgeVariant(
                item.status_label,
                item.status,
              )}
            >
              {resolveSubagentStatusLabel(item.status_label, item.status)}
            </Badge>
          }
          timestamp={timestamp}
        >
          {item.summary ? (
            <div className="text-sm text-muted-foreground">{item.summary}</div>
          ) : null}
          {item.role || item.model ? (
            <div className="mt-2 flex flex-wrap gap-2">
              {item.role ? <Badge variant="outline">{item.role}</Badge> : null}
              {item.model ? (
                <Badge variant="outline">{item.model}</Badge>
              ) : null}
            </div>
          ) : null}
          {subagentThreadId && onOpenSubagentSession ? (
            <div className="mt-3">
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => onOpenSubagentSession(subagentThreadId)}
              >
                {options?.openSubagentLabel}
              </Button>
            </div>
          ) : null}
        </SurfaceCard>
      </div>
    );
  }

  if (item.type === "expert_profile_switch") {
    const previousExpertId = item.previous_expert_id?.trim();
    const nextExpertId = item.next_expert_id?.trim();
    const switchSummary =
      previousExpertId && nextExpertId
        ? t("agentChat.threadTimeline.expertProfileSwitch.fromTo", {
            previous: previousExpertId,
            next: nextExpertId,
          })
        : nextExpertId
          ? t("agentChat.threadTimeline.expertProfileSwitch.to", {
              next: nextExpertId,
            })
          : item.summary ||
            t("agentChat.threadTimeline.expertProfileSwitch.description");

    return (
      <SurfaceCard
        icon={Sparkles}
        title={t("agentChat.threadTimeline.expertProfileSwitch.title")}
        badge={
          <Badge variant={resolveStatusBadgeVariant(item.status)}>
            {t("agentChat.threadTimeline.expertProfileSwitch.badge")}
          </Badge>
        }
        timestamp={timestamp}
      >
        <div className="text-sm text-muted-foreground">{switchSummary}</div>
      </SurfaceCard>
    );
  }

  if (item.type === "warning" || item.type === "error") {
    const displayMessage =
      item.type === "error"
        ? resolveUserFacingErrorMessage(item.message)
        : item.message;
    return (
      <SurfaceCard
        icon={item.type === "warning" ? AlertTriangle : ShieldAlert}
        title={resolveTimelineAlertFallback(
          item.type === "warning" ? "completed" : "failed",
        )}
        badge={
          <Badge variant={resolveStatusBadgeVariant(item.status)}>
            {item.type === "warning"
              ? item.code || "warning"
              : resolveItemStatusLabel("failed")}
          </Badge>
        }
        timestamp={timestamp}
      >
        <div
          className={
            item.type === "error"
              ? "text-sm text-destructive"
              : "text-sm text-muted-foreground"
          }
        >
          {displayMessage}
        </div>
      </SurfaceCard>
    );
  }

  const unsupportedType =
    item.type === "unknown_item" ? item.upstream_type : item.type;
  const unknownFieldNames =
    item.type === "unknown_item" ? item.field_names : [];

  return (
    <div className="py-1.5" data-testid="timeline-unsupported-item">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-medium text-foreground">
          {t("agentChat.threadTimeline.unsupportedItem.title", {
            type: unsupportedType,
          })}
        </span>
        <Badge
          variant={resolveStatusBadgeVariant(item.status)}
          className="ml-auto"
        >
          {resolveItemStatusLabel(item.status)}
        </Badge>
      </div>
      <div className="text-sm leading-6 text-muted-foreground">
        {t("agentChat.threadTimeline.unsupportedItem.description")}
      </div>
      {unknownFieldNames.length > 0 ? (
        <div className="mt-1 text-xs leading-5 text-muted-foreground">
          {t("agentChat.threadTimeline.unsupportedItem.fields", {
            fields: unknownFieldNames.join(", "),
          })}
        </div>
      ) : null}
    </div>
  );
}

interface TimelineItemDetailsProps {
  item: AgentThreadItem;
  onFileClick?: (fileName: string, content: string) => void;
  onOpenArtifactFromTimeline?: (target: ArtifactTimelineOpenTarget) => void;
  onOpenSavedSiteContent?: (target: SiteSavedContentTarget) => void;
  onOpenSubagentSession?: (sessionId: string) => void;
  onPermissionResponse?: (response: ConfirmResponse) => void;
  defaultExpandCompletedToolResult?: boolean;
  groupedToolCall?: boolean;
  groupMarker?: string;
  openSubagentLabel?: string;
  sourceMessageId?: string;
  onSaveFileArtifactAsKnowledge?: (source: {
    messageId: string;
    content: string;
    sourceName?: string;
    description?: string | null;
  }) => void;
}

export function TimelineItemDetails({
  item,
  onFileClick,
  onOpenArtifactFromTimeline,
  onOpenSavedSiteContent,
  onOpenSubagentSession,
  onPermissionResponse,
  defaultExpandCompletedToolResult,
  groupedToolCall,
  groupMarker,
  openSubagentLabel,
  sourceMessageId,
  onSaveFileArtifactAsKnowledge,
}: TimelineItemDetailsProps) {
  const { t } = useTranslation("agent");

  if (isThinkingTimelineItem(item)) {
    if (groupedToolCall) {
      return <GroupedThinkingRow item={item} groupMarker={groupMarker} />;
    }
    return renderThinkingItemDetails(item);
  }

  return renderGroupItemDetails(
    item,
    t,
    onFileClick,
    onOpenArtifactFromTimeline,
    onOpenSavedSiteContent,
    onOpenSubagentSession,
    onPermissionResponse,
    {
      defaultExpandCompletedToolResult,
      groupedToolCall,
      groupMarker,
      openSubagentLabel,
      sourceMessageId,
      onSaveFileArtifactAsKnowledge,
    },
  );
}

export function ThreadInlineStatusHint({
  hint,
}: {
  hint: NonNullable<ReturnType<typeof resolveThreadInlineStatusHint>>;
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-2.5 py-1 text-sm",
        hint.tone === "warning" && "text-amber-900",
        hint.tone === "error" && "text-rose-900",
        hint.tone === "neutral" && "text-slate-700",
      )}
      data-testid="agent-thread-inline-status"
    >
      <div
        className={cn(
          "mt-2 h-1.5 w-1.5 shrink-0 rounded-full",
          hint.tone === "warning" && "bg-amber-500",
          hint.tone === "error" && "bg-rose-500",
          hint.tone === "neutral" && "bg-slate-400",
        )}
      />
      <div className="min-w-0 flex-1 leading-6">
        <span className="mr-2 text-xs font-medium">{hint.label}</span>
        <span>{hint.detail}</span>
      </div>
    </div>
  );
}

export function TimelineBlockStatusIndicator({
  block,
}: {
  block: AgentThreadOrderedBlock;
}) {
  if (block.kind === "approval" && block.status !== "completed") {
    return (
      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-amber-100 text-amber-700">
        <Clock3 className="h-2.5 w-2.5" />
      </span>
    );
  }

  if (block.status === "in_progress") {
    return (
      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-sky-100 text-sky-600">
        <Loader2 className="h-2.5 w-2.5 animate-spin" />
      </span>
    );
  }

  if (block.status === "failed" || block.kind === "alert") {
    return (
      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-rose-100 text-rose-700">
        <AlertTriangle className="h-2.5 w-2.5" />
      </span>
    );
  }

  return <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />;
}
