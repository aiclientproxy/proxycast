import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ExternalLink, FileText, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import { skillsApi } from "@/lib/api/skills";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { SearchResultPreviewList } from "./SearchResultPreviewList";
import { ToolSearchSummaryPanel } from "./ToolSearchSummaryPanel";
import {
  extractLimeToolMetadataBlock,
  normalizeToolResultImages,
} from "../hooks/agentChatToolResult";
import type { AgentToolCallState as ToolCallState } from "@/lib/api/agentProtocol";
import type { SiteSavedContentTarget } from "../types";
import type { SearchResultPreviewItem } from "../utils/searchResultPreview";
import {
  buildToolHeadline,
  getToolDisplayInfo,
  normalizeToolNameKey,
  parseToolCallArguments,
  resolveToolFilePath,
  resolveToolPrimarySubject,
} from "../utils/toolDisplayInfo";
import {
  isUnifiedWebSearchToolName,
  resolveSearchResultPreviewItemsFromText,
} from "../utils/searchResultPreview";
import { isUnifiedWebFetchToolName } from "../utils/toolNameFamily";
import { attachUrlPreviewSnapshotsToSearchResults } from "../utils/urlPreviewSnapshot";
import {
  resolveSiteSavedContentTargetDisplayName,
  resolveSiteSavedContentTargetRelativePath,
  resolveSiteSavedContentTargetFromMetadata,
} from "../utils/siteToolResultSummary";
import { normalizeToolSearchResultSummary } from "../utils/toolSearchResultSummary";
import { resolveTaskBoardResultDetailText } from "../utils/taskBoardToolResultDetail";
import {
  resolveToolErrorDetailText,
  resolveToolProcessNarrative,
  isLikelyWebRetrievalDiagnosticNoise,
} from "../utils/toolProcessSummary";
import { resolveTerminalInteractionSummaries } from "../utils/toolProcessSummaryMetadata";
import { resolveToolSoulMetadataDomAttributes } from "../utils/toolSoulLifecycleMetadata";
import { resolveMemoryToolEvidence } from "../utils/memoryToolEvidence";
import {
  resolveWorkspaceSkillRuntimeEnableResultDisplay,
  shouldHideProtocolToolResultEnvelope,
  shouldHideToolResultEnvelope,
} from "../utils/toolResultEnvelopeDisplay";
import {
  normalizeToolResultDetailText,
  resolveStructuredToolContentDetailText,
  resolveToolResultStructuredContent,
  sanitizeToolResultDetailMarkdown,
} from "../utils/toolResultDetailText";
import {
  isLimeTaskProtocolFailure,
  resolveLimeTaskProtocolFailureDisplayText,
} from "../utils/limeTaskProtocolNoise";
import { MemoryToolEvidencePanel } from "./MemoryToolEvidencePanel";
import {
  asRecord,
  buildSiteNoticeLines,
  normalizeSummaryLine,
  readBoolean,
  readNumber,
  readString,
  resolveWebFetchResultText,
  summarizeDiagnosticResultPreview,
  summarizeResultText,
  summarizeSearchResultPreview,
  summarizeToolSearchPreview,
} from "./InlineToolProcessStepViewModel";

interface InlineToolProcessStepProps {
  toolCall: ToolCallState;
  grouped?: boolean;
  groupMarker?: string;
  isActiveProcess?: boolean;
  isMessageStreaming?: boolean;
  onFileClick?: (fileName: string, content: string) => void;
  onOpenSavedSiteContent?: (target: SiteSavedContentTarget) => void;
  onOpenUrlPreview?: (item: SearchResultPreviewItem) => void;
  urlPreviewToolCalls?: ToolCallState[];
}

export const InlineToolProcessStep: React.FC<InlineToolProcessStepProps> = ({
  toolCall,
  grouped = false,
  groupMarker = "•",
  isActiveProcess,
  isMessageStreaming = false,
  onFileClick,
  onOpenSavedSiteContent,
  onOpenUrlPreview,
  urlPreviewToolCalls,
}) => {
  const { t } = useTranslation("agent");
  const shouldTreatAsActiveProcess = isActiveProcess ?? isMessageStreaming;
  const [expanded, setExpanded] = useState(false);
  const [skillContentExpanded, setSkillContentExpanded] = useState(false);
  const [fetchedSkillContent, setFetchedSkillContent] = useState<string | null>(
    null,
  );
  const [skillContentLoading, setSkillContentLoading] = useState(false);
  const [skillContentError, setSkillContentError] = useState<string | null>(
    null,
  );
  const [skillMarkdownBodyExpanded, setSkillMarkdownBodyExpanded] =
    useState(false);

  const parsedArgs = useMemo(
    () => parseToolCallArguments(toolCall.arguments),
    [toolCall.arguments],
  );
  const argsRecord = useMemo(() => asRecord(parsedArgs), [parsedArgs]);
  const toolDisplay = useMemo(
    () => getToolDisplayInfo(toolCall.name, toolCall.status),
    [toolCall.name, toolCall.status],
  );
  const ToolIcon = toolDisplay.icon;
  const metadata = useMemo(() => {
    const merged = {
      ...(asRecord(toolCall.metadata) || {}),
      ...(asRecord(toolCall.result?.metadata) || {}),
    };
    return Object.keys(merged).length > 0 ? merged : null;
  }, [toolCall.metadata, toolCall.result?.metadata]);
  const normalizedToolName = useMemo(
    () => normalizeToolNameKey(toolCall.name),
    [toolCall.name],
  );
  const isToolSearch = useMemo(
    () => normalizedToolName === "toolsearch",
    [normalizedToolName],
  );
  const isMcpStructuredToolResult = useMemo(
    () =>
      normalizedToolName === "mcp" ||
      toolCall.name.trim().toLowerCase().startsWith("mcp__"),
    [normalizedToolName, toolCall.name],
  );
  const isSkillLikeTool =
    toolDisplay.family === "skill" ||
    metadata?.tool_family === "skill" ||
    normalizedToolName === "skill" ||
    normalizedToolName === "skilltool" ||
    normalizedToolName === "limerunserviceskill";
  const shouldSuppressTransientResultText =
    (toolCall.status === "running" || isMessageStreaming) &&
    !isToolSearch &&
    !isUnifiedWebSearchToolName(toolCall.name) &&
    !isUnifiedWebFetchToolName(toolCall.name);
  const shouldSuppressResultText =
    shouldSuppressTransientResultText || toolDisplay.family === "vision";
  const filePath = useMemo(() => resolveToolFilePath(parsedArgs), [parsedArgs]);
  const fileContent = useMemo(() => {
    const content = parsedArgs.content || parsedArgs.text;
    if (content) {
      return String(content);
    }
    return toolCall.result?.output ? String(toolCall.result.output) : "";
  }, [parsedArgs.content, parsedArgs.text, toolCall.result?.output]);
  const subject = useMemo(
    () => resolveToolPrimarySubject(toolCall.name, parsedArgs, filePath),
    [filePath, parsedArgs, toolCall.name],
  );
  const rawResultText = useMemo(() => {
    const rawText = toolCall.result?.error || toolCall.result?.output || "";
    return extractLimeToolMetadataBlock(rawText).text.trim();
  }, [toolCall.result?.error, toolCall.result?.output]);
  const structuredContentValue = useMemo(
    () => resolveToolResultStructuredContent(toolCall.result),
    [toolCall.result],
  );
  const structuredContentDetail = useMemo(
    () => resolveStructuredToolContentDetailText(structuredContentValue),
    [structuredContentValue],
  );
  const shouldPreferStructuredProtocolResult = useMemo(
    () =>
      toolCall.status !== "failed" &&
      Boolean(structuredContentDetail) &&
      shouldHideProtocolToolResultEnvelope({
        toolName: toolCall.name,
        rawResultText,
        structuredContent: structuredContentValue,
      }),
    [
      rawResultText,
      structuredContentDetail,
      structuredContentValue,
      toolCall.name,
      toolCall.status,
    ],
  );
  const shouldHideResultEnvelope = useMemo(
    () =>
      shouldHideToolResultEnvelope({
        toolName: toolCall.name,
        rawResultText,
        metadata,
        result: toolCall.result,
      }),
    [metadata, rawResultText, toolCall.name, toolCall.result],
  );
  const workspaceSkillRuntimeEnableSummary = useMemo(
    () =>
      resolveWorkspaceSkillRuntimeEnableResultDisplay({
        toolName: toolCall.name,
        rawResultText,
        metadata,
        translate: (key, defaultValue, options) =>
          String(t(key, { defaultValue, ...options })),
      }),
    [metadata, rawResultText, t, toolCall.name],
  );
  const terminalInteractionSummary = useMemo(() => {
    const summaries = resolveTerminalInteractionSummaries(metadata);
    if (summaries.length === 0) {
      return null;
    }
    const separator = String(
      t("agentChat.processGroup.separator", { defaultValue: ", " }),
    );
    return String(
      t("agentChat.processGroup.terminalInteractions", {
        defaultValue: "Input sent: {{value}}",
        value: summaries.join(separator),
      }),
    );
  }, [metadata, t]);
  const limeTaskProtocolFailureText = useMemo(() => {
    if (toolCall.status !== "failed") {
      return null;
    }

    return isLimeTaskProtocolFailure({
      toolName: toolCall.name,
      text: rawResultText,
    })
      ? resolveLimeTaskProtocolFailureDisplayText({
          toolName: toolCall.name,
          text: rawResultText,
        })
      : null;
  }, [rawResultText, toolCall.name, toolCall.status]);
  const headline = useMemo(
    () =>
      limeTaskProtocolFailureText ||
      buildToolHeadline({
        toolDisplay,
        subject,
        toolName: toolCall.name,
      }),
    [limeTaskProtocolFailureText, subject, toolCall.name, toolDisplay],
  );
  const processNarrative = useMemo(
    () => resolveToolProcessNarrative(toolCall),
    [toolCall],
  );
  const soulLifecycleAttributes = useMemo(
    () => resolveToolSoulMetadataDomAttributes(processNarrative),
    [processNarrative],
  );
  const memoryToolEvidence = useMemo(
    () => resolveMemoryToolEvidence(toolCall),
    [toolCall],
  );
  const resultText = useMemo(() => {
    const fallbackSummary =
      processNarrative.postSummary ||
      processNarrative.summary ||
      processNarrative.preSummary ||
      "";

    if (shouldSuppressResultText) {
      return "";
    }

    if (shouldPreferStructuredProtocolResult) {
      return structuredContentDetail || "";
    }

    if (isMcpStructuredToolResult && structuredContentDetail) {
      return structuredContentDetail;
    }

    if (isUnifiedWebFetchToolName(toolCall.name)) {
      return resolveWebFetchResultText({
        rawResultText,
        fallbackSummary,
      });
    }

    if (shouldHideResultEnvelope) {
      return toolCall.status === "running"
        ? ""
        : structuredContentDetail || fallbackSummary || "";
    }

    if (toolCall.status !== "failed") {
      return (
        resolveTaskBoardResultDetailText({
          toolName: toolCall.name,
          outputText: rawResultText,
          metadata,
          fallbackSummary,
          copy: {
            taskNotFound: () =>
              t("agentChat.toolCall.taskBoard.notFound", "Task not found"),
            moreTasks: (count) =>
              t("agentChat.toolCall.taskBoard.moreTasks", {
                count,
                defaultValue: "{{count}} more tasks",
              }),
            emptyTaskList: () =>
              t(
                "agentChat.toolCall.taskBoard.emptyTaskList",
                "Task list is empty",
              ),
          },
        }) ||
        normalizeToolResultDetailText(rawResultText) ||
        structuredContentDetail ||
        ""
      );
    }

    return (
      resolveToolErrorDetailText(toolCall.name, rawResultText) ||
      rawResultText ||
      structuredContentDetail ||
      ""
    );
  }, [
    isMcpStructuredToolResult,
    metadata,
    processNarrative.postSummary,
    processNarrative.preSummary,
    processNarrative.summary,
    rawResultText,
    shouldHideResultEnvelope,
    shouldPreferStructuredProtocolResult,
    shouldSuppressResultText,
    structuredContentDetail,
    t,
    toolCall.name,
    toolCall.status,
  ]);
  const isRawDiagnosticDetail = useMemo(
    () =>
      processNarrative.postSource === "error" &&
      Boolean(rawResultText) &&
      isLikelyWebRetrievalDiagnosticNoise(rawResultText),
    [processNarrative.postSource, rawResultText],
  );
  const resultDetailMarkdown = useMemo(
    () => sanitizeToolResultDetailMarkdown(resultText),
    [resultText],
  );
  const resultPreview = useMemo(
    () => summarizeResultText(resultText),
    [resultText],
  );
  const liveResultPreview = useMemo(() => {
    if (
      toolCall.status !== "running" ||
      !rawResultText ||
      isSkillLikeTool ||
      toolDisplay.family === "vision" ||
      shouldHideResultEnvelope
    ) {
      return null;
    }

    return summarizeResultText(normalizeToolResultDetailText(rawResultText));
  }, [
    isSkillLikeTool,
    rawResultText,
    shouldHideResultEnvelope,
    toolCall.status,
    toolDisplay.family,
  ]);
  const resultImages = useMemo(
    () =>
      normalizeToolResultImages(
        toolCall.result?.images,
        rawResultText,
        metadata,
      ) || [],
    [metadata, rawResultText, toolCall.result?.images],
  );
  const toolSearchSummary = useMemo(
    () =>
      isToolSearch ? normalizeToolSearchResultSummary(rawResultText) : null,
    [isToolSearch, rawResultText],
  );
  const searchResultItems = useMemo(() => {
    if (!isUnifiedWebSearchToolName(toolCall.name)) {
      return [];
    }

    const items = resolveSearchResultPreviewItemsFromText(rawResultText);
    return attachUrlPreviewSnapshotsToSearchResults({
      items,
      toolCalls: urlPreviewToolCalls,
    });
  }, [rawResultText, toolCall.name, urlPreviewToolCalls]);
  const structuredResultPreview = useMemo(() => {
    if (shouldSuppressResultText) {
      return null;
    }
    if (toolSearchSummary) {
      return summarizeToolSearchPreview(toolSearchSummary);
    }
    if (memoryToolEvidence) {
      return memoryToolEvidence.summary;
    }
    if (searchResultItems.length > 0) {
      return summarizeSearchResultPreview(searchResultItems.length);
    }
    if (isRawDiagnosticDetail) {
      return summarizeDiagnosticResultPreview(rawResultText);
    }
    return resultPreview;
  }, [
    isRawDiagnosticDetail,
    memoryToolEvidence,
    rawResultText,
    resultPreview,
    searchResultItems.length,
    shouldSuppressResultText,
    toolSearchSummary,
  ]);
  const savedSiteContentTarget = useMemo(
    () => resolveSiteSavedContentTargetFromMetadata(toolCall.result?.metadata),
    [toolCall.result?.metadata],
  );
  const savedSiteContentDisplayName = useMemo(
    () =>
      resolveSiteSavedContentTargetDisplayName(savedSiteContentTarget) ||
      resolveSiteSavedContentTargetRelativePath(savedSiteContentTarget),
    [savedSiteContentTarget],
  );
  const siteNoticeLines = useMemo(
    () => buildSiteNoticeLines(toolCall),
    [toolCall],
  );
  const skillTitle =
    readString(argsRecord, ["skill_title", "skillTitle"]) ||
    readString(metadata, ["skill_title", "skillTitle"]);
  const isPreload =
    metadata?.execution_origin === "preload" || metadata?.preload === true;
  const hasOpenableFile = Boolean(filePath && onFileClick);
  const skillSource =
    readString(metadata, ["skill_source", "skillSource"]) ||
    readString(argsRecord, ["source"]);
  const isSkillInvocation = isSkillLikeTool || skillSource === "SKILL.md";
  const skillName =
    readString(metadata, ["skill_name", "skillName"]) ||
    readString(argsRecord, ["skill", "name"]);
  const skillDisplayName =
    readString(metadata, ["skill_display_name", "skillDisplayName"]) ||
    readString(argsRecord, ["display_name", "displayName"]) ||
    skillName;
  const skillSnapshotContent = readString(metadata, [
    "skill_markdown_content",
    "skillMarkdownContent",
    "markdown_content",
    "markdownContent",
  ]);
  const skillMarkdownContentBytes = readNumber(metadata, [
    "markdown_content_bytes",
    "markdownContentBytes",
  ]);
  const isStandardSkillSnapshot = readBoolean(metadata, [
    "agent_skills_standard",
    "agentSkillsStandard",
  ]);
  const hasSkillContentAccess =
    isSkillInvocation &&
    Boolean(skillSnapshotContent || fetchedSkillContent || skillName);
  const resolvedSkillContent =
    skillSnapshotContent || fetchedSkillContent || "";
  const skillContentSourceLabel = skillSnapshotContent
    ? t("agentChat.toolCall.skillContent.source.snapshot")
    : t("agentChat.toolCall.skillContent.source.current");
  const skillContentTitle = skillSnapshotContent
    ? t("agentChat.toolCall.skillContent.title.snapshot")
    : t("agentChat.toolCall.skillContent.title.current");
  const processSummary = useMemo(() => {
    const streamingResultPreview = liveResultPreview || structuredResultPreview;
    const streamingOutputSummary =
      toolCall.status === "running" && streamingResultPreview
        ? t("agentChat.toolCall.inline.streamingOutput", {
            value: streamingResultPreview,
          })
        : null;
    const progressSummary =
      toolCall.status === "running" && toolCall.progress?.message
        ? t("agentChat.toolCall.inline.progress", {
            message: toolCall.progress.message,
          })
        : null;
    const transientSummary = shouldSuppressResultText
      ? streamingOutputSummary || progressSummary || processNarrative.preSummary
      : null;
    const shouldPreferResultPreview =
      toolCall.status !== "running" &&
      Boolean(structuredResultPreview) &&
      (shouldPreferStructuredProtocolResult ||
        (isMcpStructuredToolResult && Boolean(structuredContentDetail)) ||
        (toolDisplay.family === "command" &&
          (processNarrative.postSource === "generic" ||
            processNarrative.postSource === "none")));
    const preferredSummary = transientSummary
      ? transientSummary
      : shouldPreferResultPreview
        ? structuredResultPreview
        : toolCall.status === "running"
          ? isSkillLikeTool
            ? progressSummary || processNarrative.preSummary
            : streamingOutputSummary ||
              progressSummary ||
              processNarrative.preSummary
          : processNarrative.postSummary ||
            (processNarrative.postSource === "generic"
              ? processNarrative.preSummary
              : structuredResultPreview) ||
            processNarrative.preSummary;

    return normalizeSummaryLine(preferredSummary, headline);
  }, [
    headline,
    isSkillLikeTool,
    isMcpStructuredToolResult,
    liveResultPreview,
    processNarrative.postSource,
    processNarrative.postSummary,
    processNarrative.preSummary,
    shouldSuppressResultText,
    shouldPreferStructuredProtocolResult,
    structuredContentDetail,
    structuredResultPreview,
    t,
    toolCall.progress?.message,
    toolCall.status,
    toolDisplay.family,
  ]);
  const hasDetails =
    (Boolean(resultText) &&
      (!isUnifiedWebSearchToolName(toolCall.name) ||
        toolCall.status === "failed" ||
        isRawDiagnosticDetail)) ||
    resultImages.length > 0 ||
    searchResultItems.length > 0 ||
    Boolean(toolSearchSummary) ||
    Boolean(memoryToolEvidence) ||
    siteNoticeLines.length > 0 ||
    Boolean(savedSiteContentTarget) ||
    Boolean(skillTitle && skillTitle !== subject);

  const handleOpenSearchResult = useCallback(
    (item: SearchResultPreviewItem) => {
      onOpenUrlPreview?.(item);
    },
    [onOpenUrlPreview],
  );

  useEffect(() => {
    if (
      (toolCall.status === "running" &&
        !isUnifiedWebSearchToolName(toolCall.name) &&
        !isUnifiedWebFetchToolName(toolCall.name)) ||
      siteNoticeLines.length > 0 ||
      (shouldTreatAsActiveProcess &&
        (isUnifiedWebSearchToolName(toolCall.name) ||
          isUnifiedWebFetchToolName(toolCall.name)))
    ) {
      setExpanded(true);
      return;
    }
  }, [
    shouldTreatAsActiveProcess,
    siteNoticeLines.length,
    toolCall.status,
    toolCall.name,
    toolSearchSummary,
    memoryToolEvidence,
  ]);

  useEffect(() => {
    setSkillContentExpanded(false);
    setFetchedSkillContent(null);
    setSkillContentLoading(false);
    setSkillContentError(null);
    setSkillMarkdownBodyExpanded(false);
  }, [toolCall.id]);

  const handleToggleSkillContent = useCallback(async () => {
    if (skillContentExpanded) {
      setSkillContentExpanded(false);
      setSkillMarkdownBodyExpanded(false);
      return;
    }

    setSkillContentExpanded(true);
    setSkillMarkdownBodyExpanded(false);
    if (skillSnapshotContent || fetchedSkillContent) {
      return;
    }

    const normalizedSkillName = skillName?.trim();
    if (!normalizedSkillName) {
      setSkillContentError(
        t("agentChat.toolCall.skillContent.error.unavailable"),
      );
      return;
    }

    setSkillContentLoading(true);
    setSkillContentError(null);
    try {
      const inspection = await skillsApi.inspectLocalSkill(normalizedSkillName);
      setFetchedSkillContent(inspection.content || "");
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : typeof error === "string"
            ? error
            : t("agentChat.toolCall.skillContent.error.unknown");
      setSkillContentError(
        t("agentChat.toolCall.skillContent.error.loadFailed", { message }),
      );
    } finally {
      setSkillContentLoading(false);
    }
  }, [
    fetchedSkillContent,
    skillContentExpanded,
    skillName,
    skillSnapshotContent,
    t,
  ]);

  const detailBadges = [
    isPreload ? t("agentChat.toolCall.inline.badge.preload") : null,
    skillTitle && skillTitle !== subject
      ? t("agentChat.toolCall.inline.badge.skill", { title: skillTitle })
      : null,
    workspaceSkillRuntimeEnableSummary,
    terminalInteractionSummary,
    toolCall.status === "running" || toolCall.status === "failed"
      ? toolDisplay.action
      : null,
  ].filter((value): value is string => Boolean(value));

  return (
    <div
      className="py-1"
      data-testid="inline-tool-process-step"
      data-grouped={grouped ? "yes" : "no"}
      {...soulLifecycleAttributes}
    >
      <div
        className="flex items-start gap-2"
        data-testid="tool-call-row"
        data-tool-call-id={toolCall.id}
        data-tool-name={toolCall.name}
        data-tool-status={toolCall.status}
      >
        {grouped ? (
          <span className="pt-0.5 font-mono text-xs text-slate-400">
            {groupMarker}
          </span>
        ) : null}
        <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center">
          {toolCall.status === "running" ? (
            <Loader2 className="h-4 w-4 animate-spin text-slate-500" />
          ) : (
            <ToolIcon
              className={cn(
                "h-4 w-4",
                toolCall.status === "completed" && "text-slate-400",
                toolCall.status === "failed" && "text-rose-600",
                toolCall.status !== "completed" &&
                  toolCall.status !== "failed" &&
                  "text-slate-500",
              )}
            />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start gap-2">
            <button
              type="button"
              className={cn(
                "min-w-0 flex-1 text-left",
                hasDetails && "cursor-pointer",
              )}
              onClick={() => {
                if (hasDetails) {
                  setExpanded((current) => !current);
                }
              }}
              aria-expanded={hasDetails ? expanded : undefined}
            >
              <div className="truncate text-[13px] font-normal leading-6 text-slate-700">
                {headline}
              </div>
              <div className="mt-0.5 flex flex-wrap gap-x-2 gap-y-1 text-xs leading-5 text-slate-500">
                {detailBadges.map((badge) => (
                  <span key={badge}>{badge}</span>
                ))}
              </div>
              {processSummary ? (
                <div className="mt-1 text-xs leading-5 text-slate-600">
                  {processSummary}
                </div>
              ) : !expanded && structuredResultPreview ? (
                <div className="mt-1 text-xs leading-5 text-slate-600">
                  {structuredResultPreview}
                </div>
              ) : null}
            </button>

            <div className="flex shrink-0 items-center gap-1 pt-0.5">
              {hasSkillContentAccess ? (
                <button
                  type="button"
                  className={cn(
                    "inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors",
                    skillContentExpanded
                      ? "bg-emerald-50 text-emerald-800"
                      : "text-slate-500 hover:bg-slate-100 hover:text-slate-800",
                  )}
                  title={
                    skillContentExpanded
                      ? t("agentChat.toolCall.skillContent.action.hide")
                      : t("agentChat.toolCall.skillContent.action.view")
                  }
                  aria-label={
                    skillContentExpanded
                      ? t("agentChat.toolCall.skillContent.action.hide")
                      : t("agentChat.toolCall.skillContent.action.view")
                  }
                  onClick={handleToggleSkillContent}
                >
                  <FileText className="h-3.5 w-3.5" />
                  <span>
                    {t("agentChat.toolCall.skillContent.action.viewShort")}
                  </span>
                </button>
              ) : null}
              {hasOpenableFile ? (
                <button
                  type="button"
                  data-testid="inline-tool-open-file"
                  data-file-path={filePath || undefined}
                  className="rounded-md p-1 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700"
                  title={t("agentChat.toolCall.openInCanvas")}
                  aria-label={t("agentChat.toolCall.openInCanvasWithTarget", {
                    target: filePath,
                  })}
                  onClick={() => {
                    if (filePath && onFileClick) {
                      onFileClick(filePath, fileContent);
                    }
                  }}
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                </button>
              ) : null}
              {hasDetails ? (
                <button
                  type="button"
                  className="rounded-md p-1 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700"
                  title={
                    expanded
                      ? t("agentChat.toolCall.inline.collapseDetails")
                      : t("agentChat.toolCall.inline.expandDetails")
                  }
                  aria-label={
                    expanded
                      ? t("agentChat.toolCall.inline.collapseDetails")
                      : t("agentChat.toolCall.inline.expandDetails")
                  }
                  onClick={() => setExpanded((current) => !current)}
                >
                  <ChevronDown
                    className={cn(
                      "h-3.5 w-3.5 transition-transform",
                      expanded && "rotate-180",
                    )}
                  />
                </button>
              ) : null}
            </div>
          </div>

          {skillContentExpanded ? (
            <div
              className="ml-1 mt-2 rounded-[14px] border border-emerald-100 bg-emerald-50/60 p-3"
              data-testid="inline-tool-skill-content-panel"
            >
              <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-900">
                <FileText className="h-3.5 w-3.5 shrink-0" />
                <span className="truncate">{skillContentTitle}</span>
              </div>
              <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-emerald-700">
                <span>{skillContentSourceLabel}</span>
                {skillDisplayName ? <span>{skillDisplayName}</span> : null}
                {skillMarkdownContentBytes !== null ? (
                  <span>
                    {t("agentChat.toolCall.skillContent.meta.bytes", {
                      count: skillMarkdownContentBytes,
                    })}
                  </span>
                ) : null}
                {isStandardSkillSnapshot === true ? (
                  <span>
                    {t("agentChat.toolCall.skillContent.meta.standard")}
                  </span>
                ) : null}
              </div>
              <button
                type="button"
                className="mt-3 flex w-full items-center justify-between rounded-[12px] border border-emerald-100 bg-white px-3 py-2 text-left text-xs font-medium text-emerald-900 transition-colors hover:border-emerald-200 hover:bg-emerald-50/60"
                aria-expanded={skillMarkdownBodyExpanded}
                aria-controls={`inline-skill-content-body-${toolCall.id}`}
                onClick={() =>
                  setSkillMarkdownBodyExpanded((current) => !current)
                }
              >
                <span>
                  {skillMarkdownBodyExpanded
                    ? t("agentChat.toolCall.skillContent.action.collapseBody")
                    : t("agentChat.toolCall.skillContent.action.expandBody")}
                </span>
                <ChevronDown
                  className={cn(
                    "h-3.5 w-3.5 shrink-0 transition-transform",
                    skillMarkdownBodyExpanded && "rotate-180",
                  )}
                />
              </button>
              {skillMarkdownBodyExpanded ? (
                <div
                  id={`inline-skill-content-body-${toolCall.id}`}
                  className="mt-2 max-h-80 overflow-y-auto rounded-[12px] border border-slate-200 bg-white p-3"
                  data-testid="inline-tool-skill-content-body"
                >
                  {skillContentLoading ? (
                    <div className="flex items-center gap-2 text-sm text-slate-500">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      {t("agentChat.toolCall.skillContent.loading")}
                    </div>
                  ) : skillContentError ? (
                    <div className="text-sm text-rose-700">
                      {skillContentError}
                    </div>
                  ) : resolvedSkillContent.trim() ? (
                    <MarkdownRenderer content={resolvedSkillContent} />
                  ) : (
                    <div className="text-sm text-slate-500">
                      {t("agentChat.toolCall.skillContent.empty")}
                    </div>
                  )}
                </div>
              ) : null}
            </div>
          ) : null}

          {memoryToolEvidence ? (
            <div className="ml-1 mt-2">
              <MemoryToolEvidencePanel evidence={memoryToolEvidence} />
            </div>
          ) : null}

          {expanded && hasDetails ? (
            <div className="ml-1 mt-2 space-y-2 border-l border-slate-200 pl-3">
              {siteNoticeLines.length > 0 ? (
                <div className="space-y-1 text-xs leading-5 text-slate-600">
                  {siteNoticeLines.map((line) => (
                    <div key={line}>{line}</div>
                  ))}
                </div>
              ) : null}

              {savedSiteContentTarget && onOpenSavedSiteContent ? (
                <div>
                  <button
                    type="button"
                    className="flex w-full items-center gap-3 rounded-xl border border-emerald-300 bg-emerald-50/70 px-3 py-2 text-left transition-colors hover:bg-emerald-100/70"
                    onClick={() =>
                      onOpenSavedSiteContent(savedSiteContentTarget)
                    }
                  >
                    <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-emerald-200 bg-white text-emerald-700">
                      <FileText className="h-4 w-4" />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block text-xs font-medium leading-5 text-emerald-900">
                        {savedSiteContentTarget.preferredTarget ===
                        "project_file"
                          ? t(
                              "agentChat.toolCall.siteResult.openMarkdownPreview",
                            )
                          : t("agentChat.toolCall.siteResult.openSavedContent")}
                      </span>
                      {savedSiteContentDisplayName ? (
                        <span className="block truncate text-[11px] leading-5 text-emerald-700/80">
                          {savedSiteContentDisplayName}
                        </span>
                      ) : null}
                    </span>
                    <ExternalLink className="h-3.5 w-3.5 shrink-0 text-emerald-700" />
                  </button>
                </div>
              ) : null}

              {toolSearchSummary ? (
                <ToolSearchSummaryPanel
                  summary={toolSearchSummary}
                  testId="inline-tool-process-tool-search-result"
                />
              ) : null}

              {!toolSearchSummary &&
              !memoryToolEvidence &&
              searchResultItems.length > 0 ? (
                <SearchResultPreviewList
                  items={searchResultItems}
                  onOpenItem={handleOpenSearchResult}
                  popoverSide="bottom"
                  popoverAlign="start"
                  className="max-w-2xl"
                />
              ) : null}

              {!toolSearchSummary &&
              !memoryToolEvidence &&
              searchResultItems.length === 0 &&
              resultText ? (
                <div className="text-sm leading-6 text-slate-700">
                  <MarkdownRenderer content={resultDetailMarkdown} />
                </div>
              ) : null}

              {resultImages.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {resultImages.map((image, index) => (
                    <img
                      key={`${image.src.slice(0, 48)}-${index}`}
                      src={image.src}
                      alt={t("agentChat.toolCall.resultImage.alt")}
                      className="h-20 w-20 rounded-lg border border-slate-200 object-cover"
                      loading="lazy"
                    />
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};
