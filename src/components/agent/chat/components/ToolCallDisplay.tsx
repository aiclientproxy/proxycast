/**
 * 工具调用显示组件
 *
 * 参考 agent UI 设计，显示工具执行状态、参数、日志和结果
 * Requirements: 9.1, 9.2 - 工具执行指示器和结果折叠面板
 */

import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
} from "react";
import { ChevronRight, ExternalLink } from "lucide-react";
import { useTranslation } from "react-i18next";
import { openExternalUrlWithSystemBrowser } from "@/lib/api/externalUrl";
import { cn } from "@/lib/utils";
import { skillsApi } from "@/lib/api/skills";
import type { AgentToolCallState as ToolCallState } from "@/lib/api/agentProtocol";
import type { SiteSavedContentTarget } from "../types";
import { SearchResultPreviewList } from "./SearchResultPreviewList";
import { ToolSearchSummaryPanel } from "./ToolSearchSummaryPanel";
import {
  isUnifiedWebSearchToolName,
  resolveSearchResultPreviewItemsFromText,
} from "../utils/searchResultPreview";
import {
  extractLimeToolMetadataBlock,
  isToolResultSuccessful,
} from "../hooks/agentChatToolResult";
import {
  normalizeSiteToolResultSummary,
  resolveSiteSavedContentTargetFromMetadata,
} from "../utils/siteToolResultSummary";
import { normalizeToolSearchResultSummary } from "../utils/toolSearchResultSummary";
import {
  buildDiffFileCanvasContent,
  buildDiffReviewScopeItems,
  resolveDiffReviewSummaryFromCandidates,
  type DiffReviewFile,
} from "../utils/diffReview";
import {
  buildGroupedChildLine as buildGroupedChildLineFromInfo,
  buildToolHeadline as buildToolHeadlineFromInfo,
  getToolDisplayInfo as getToolDisplayInfoFromInfo,
  parseToolCallArguments as parseToolCallArgumentsFromInfo,
  resolveToolFilePath as resolveToolFilePathFromInfo,
  resolveToolPrimarySubject as resolveToolPrimarySubjectFromInfo,
} from "../utils/toolDisplayInfo";
import {
  resolveToolErrorDetailText,
  resolveToolProcessNarrative,
} from "../utils/toolProcessSummary";
import { resolveTerminalInteractionSummaries } from "../utils/toolProcessSummaryMetadata";
import { resolveToolSoulMetadataDomAttributes } from "../utils/toolSoulLifecycleMetadata";
import {
  isLimeTaskProtocolFailure,
  resolveLimeTaskProtocolFailureDisplayText,
} from "../utils/limeTaskProtocolNoise";
import {
  buildRenderedToolResultContent,
  buildToolResultMetaNoticeKeys,
  formatCommandEncoding,
  isToolSearchToolName,
  normalizeToolResultImages,
  normalizeToolResultMetadata,
  readRecordString,
  resolveCommandOutputStreams,
  resolveCommandToolSummary,
  resolveSkillInvocationContentInfo,
  resolveToolResultPath,
  type ToolResultNotice,
} from "./ToolCallDisplayViewModel";
import { shouldHideToolResultEnvelope } from "../utils/toolResultEnvelopeDisplay";
import {
  resolveStructuredToolContentDetailText,
  resolveToolResultStructuredContent,
} from "../utils/toolResultDetailText";
import { ToolCallDisplayResultPanel } from "./ToolCallDisplayResultPanel";
import {
  ToolCallSkillContentButton,
  ToolCallSkillContentPanel,
} from "./ToolCallSkillContentPanel";

// ============ 主组件 ============

interface ToolCallDisplayProps {
  toolCall: ToolCallState;
  defaultExpanded?: boolean;
  /** 当前 assistant 消息是否仍在流式输出 */
  isMessageStreaming?: boolean;
  /** 文件点击回调 - 用于打开右边栏显示文件内容 */
  onFileClick?: (fileName: string, content: string) => void;
  onOpenSavedSiteContent?: (target: SiteSavedContentTarget) => void;
  grouped?: boolean;
  groupMarker?: string;
}

export const ToolCallDisplay: React.FC<ToolCallDisplayProps> = ({
  toolCall,
  defaultExpanded = false,
  isMessageStreaming = false,
  onFileClick,
  onOpenSavedSiteContent,
  grouped = false,
  groupMarker = "•",
}) => {
  const { t } = useTranslation("agent");
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [isSkillContentExpanded, setIsSkillContentExpanded] = useState(false);
  const [fetchedSkillContent, setFetchedSkillContent] = useState<string | null>(
    null,
  );
  const [skillContentLoading, setSkillContentLoading] = useState(false);
  const [skillContentError, setSkillContentError] = useState<string | null>(
    null,
  );
  const [skillMarkdownBodyExpanded, setSkillMarkdownBodyExpanded] =
    useState(false);
  const [showRawSearchResultOutput, setShowRawSearchResultOutput] =
    useState(false);
  const [expandedDiffFileIds, setExpandedDiffFileIds] = useState<
    Record<string, boolean>
  >({});
  const [previewImageSrc, setPreviewImageSrc] = useState<string | null>(null);
  const hasUserToggledExpandedRef = useRef(false);
  const emptyOutputLabel = t("agentChat.toolCall.result.empty");

  // 解析参数
  const parsedArgs = useMemo(
    () => parseToolCallArgumentsFromInfo(toolCall.arguments),
    [toolCall.arguments],
  );

  const toolDisplay = useMemo(
    () => getToolDisplayInfoFromInfo(toolCall.name, toolCall.status),
    [toolCall.name, toolCall.status],
  );

  // 获取文件路径
  const filePath = useMemo(
    () => resolveToolFilePathFromInfo(parsedArgs),
    [parsedArgs],
  );

  // 获取文件名
  const fileName = useMemo(
    () =>
      resolveToolPrimarySubjectFromInfo(toolCall.name, parsedArgs, filePath),
    [filePath, parsedArgs, toolCall.name],
  );

  // 获取文件内容（用于点击打开右边栏）
  const fileContent = useMemo(() => {
    const content = parsedArgs.content || parsedArgs.text;
    return content ? String(content) : null;
  }, [parsedArgs]);

  const isRunning = toolCall.status === "running";
  const isCompleted = toolCall.status === "completed";
  const isFailed = toolCall.status === "failed";
  const hasResult = !isRunning && toolCall.result;
  const resultImages = useMemo(
    () =>
      normalizeToolResultImages(
        toolCall.result?.images,
        toolCall.result?.output,
        toolCall.result?.metadata,
      ),
    [
      toolCall.result?.images,
      toolCall.result?.metadata,
      toolCall.result?.output,
    ],
  );
  const resultMetadata = useMemo(() => {
    const metadata = {
      ...(normalizeToolResultMetadata(toolCall.metadata) || {}),
      ...(normalizeToolResultMetadata(toolCall.result?.metadata) || {}),
    };
    return Object.keys(metadata).length > 0 ? metadata : undefined;
  }, [toolCall.metadata, toolCall.result?.metadata]);
  const terminalInteractionSummary = useMemo(() => {
    const summaries = resolveTerminalInteractionSummaries(resultMetadata);
    return summaries.length > 0
      ? t("agentChat.processGroup.terminalInteractions", {
          defaultValue: "Input sent: {{value}}",
          value: summaries.join(
            t("agentChat.processGroup.separator", { defaultValue: ", " }),
          ),
        })
      : null;
  }, [resultMetadata, t]);
  const skillInvocationContentInfo = useMemo(
    () =>
      resolveSkillInvocationContentInfo({
        toolCall,
        args: parsedArgs,
        metadata: resultMetadata,
      }),
    [parsedArgs, resultMetadata, toolCall],
  );
  const siteResultSummary = useMemo(
    () => normalizeSiteToolResultSummary(toolCall.result?.metadata),
    [toolCall.result?.metadata],
  );
  const savedSiteContentTarget = useMemo(
    () => resolveSiteSavedContentTargetFromMetadata(toolCall.result?.metadata),
    [toolCall.result?.metadata],
  );
  const rawResultText = useMemo(() => {
    const rawText = toolCall.result?.error || toolCall.result?.output || "";
    return extractLimeToolMetadataBlock(rawText).text;
  }, [toolCall.result?.error, toolCall.result?.output]);
  const processNarrative = useMemo(
    () => resolveToolProcessNarrative(toolCall),
    [toolCall],
  );
  const soulLifecycleAttributes = useMemo(
    () => resolveToolSoulMetadataDomAttributes(processNarrative),
    [processNarrative],
  );
  const shouldHideResultEnvelope = useMemo(
    () =>
      shouldHideToolResultEnvelope({
        toolName: toolCall.name,
        rawResultText,
        metadata: resultMetadata,
        result: toolCall.result,
      }),
    [rawResultText, resultMetadata, toolCall.name, toolCall.result],
  );
  const limeTaskProtocolFailureText = useMemo(() => {
    if (!isFailed) {
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
  }, [isFailed, rawResultText, toolCall.name]);
  const resultText = useMemo(() => {
    const normalized = rawResultText;
    const structuredContentDetail = resolveStructuredToolContentDetailText(
      resolveToolResultStructuredContent(toolCall.result),
    );
    const fallbackSummary =
      processNarrative.postSummary ||
      processNarrative.summary ||
      processNarrative.preSummary ||
      emptyOutputLabel;

    if (shouldHideResultEnvelope) {
      return isRunning ? "" : structuredContentDetail || fallbackSummary;
    }

    if (isFailed) {
      return (
        resolveToolErrorDetailText(toolCall.name, normalized) ||
        normalized ||
        structuredContentDetail ||
        emptyOutputLabel
      );
    }
    return normalized || structuredContentDetail || emptyOutputLabel;
  }, [
    emptyOutputLabel,
    isFailed,
    isRunning,
    processNarrative.postSummary,
    processNarrative.preSummary,
    processNarrative.summary,
    rawResultText,
    shouldHideResultEnvelope,
    toolCall.result,
    toolCall.name,
  ]);
  const isResultFailure = useMemo(() => {
    if (!toolCall.result) return isFailed;
    return !isToolResultSuccessful({
      success: toolCall.result.success,
      metadata: resultMetadata,
    });
  }, [isFailed, resultMetadata, toolCall.result]);
  const resolveSiteProjectTargetDisplay = useCallback(
    (params: { source?: string; projectId?: string }): string => {
      if (params.source === "context_project") {
        return t("agentChat.toolCall.siteResult.target.currentProject");
      }
      if (params.source === "explicit_project") {
        return t("agentChat.toolCall.siteResult.target.selectedProject");
      }
      const projectId = params.projectId?.trim();
      if (projectId) {
        return t("agentChat.toolCall.siteResult.target.project", {
          projectId,
        });
      }
      return t("agentChat.toolCall.siteResult.target.currentProject");
    },
    [t],
  );
  const resultMetaItems = useMemo(() => {
    return buildToolResultMetaNoticeKeys({
      metadata: resultMetadata,
      isResultFailure,
    }).map((key) => t(`agentChat.toolCall.resultNotice.${key}`));
  }, [isResultFailure, resultMetadata, t]);
  const commandSummary = useMemo(
    () =>
      resolveCommandToolSummary({
        toolName: toolCall.name,
        args: parsedArgs,
        metadata: resultMetadata,
      }),
    [parsedArgs, resultMetadata, toolCall.name],
  );
  const commandEncoding = useMemo(
    () => (commandSummary ? formatCommandEncoding(commandSummary) : null),
    [commandSummary],
  );
  const commandSurfaceLabel = useMemo(() => {
    if (!commandSummary?.executionSurface) {
      return null;
    }
    return commandSummary.executionSurface === "embedded"
      ? t("agentChat.toolCall.commandSummary.surfaceEmbedded")
      : t("agentChat.toolCall.commandSummary.surfaceExternal");
  }, [commandSummary, t]);
  const commandDecodedWithLabel = useMemo(() => {
    if (!commandSummary?.decodedWith) {
      return null;
    }
    return commandSummary.decodedWith === "strict"
      ? t("agentChat.toolCall.commandSummary.decodedWithStrict")
      : t("agentChat.toolCall.commandSummary.decodedWithLossy");
  }, [commandSummary, t]);
  const commandOutputStreams = useMemo(
    () =>
      commandSummary
        ? resolveCommandOutputStreams({
            output: toolCall.result?.output,
            error: toolCall.result?.error,
            metadata: resultMetadata,
          })
        : [],
    [
      commandSummary,
      resultMetadata,
      toolCall.result?.error,
      toolCall.result?.output,
    ],
  );
  const diffReviewSummary = useMemo(
    () =>
      resolveDiffReviewSummaryFromCandidates([
        readRecordString(parsedArgs, [
          "patch",
          "diff",
          "unified_diff",
          "unifiedDiff",
        ]),
        readRecordString(resultMetadata, [
          "patch",
          "diff",
          "unified_diff",
          "unifiedDiff",
        ]),
        ...commandOutputStreams.map((stream) => stream.content),
        resultText,
      ]),
    [commandOutputStreams, parsedArgs, resultMetadata, resultText],
  );
  const diffReviewScopeItems = useMemo(
    () =>
      diffReviewSummary
        ? buildDiffReviewScopeItems(diffReviewSummary.files)
        : [],
    [diffReviewSummary],
  );
  const siteResultNotices = useMemo(() => {
    if (!siteResultSummary) return [] as ToolResultNotice[];

    const notices: ToolResultNotice[] = [];
    const savedProjectId =
      siteResultSummary.savedProjectId ||
      siteResultSummary.savedContent?.projectId;
    const savedProjectTarget = resolveSiteProjectTargetDisplay({
      source: siteResultSummary.savedBy,
      projectId: savedProjectId,
    });

    if (siteResultSummary.savedContent?.title) {
      const text = t("agentChat.toolCall.siteResult.saved", {
        target: savedProjectTarget,
        title: siteResultSummary.savedContent.title,
      });
      notices.push({
        key: "site-save-success",
        text,
        tone: "success",
      });
    }

    if (siteResultSummary.savedContent?.markdownRelativePath) {
      notices.push({
        key: "site-save-markdown-path",
        text: t("agentChat.toolCall.siteResult.markdownExported"),
        tone: "neutral",
      });
    }

    if (typeof siteResultSummary.savedContent?.imageCount === "number") {
      notices.push({
        key: "site-save-images",
        text: t("agentChat.toolCall.siteResult.images", {
          count: siteResultSummary.savedContent.imageCount,
        }),
        tone: "neutral",
      });
    }

    if (siteResultSummary.saveSkippedProjectId) {
      const skippedProjectTarget = resolveSiteProjectTargetDisplay({
        source: siteResultSummary.saveSkippedBy,
        projectId: siteResultSummary.saveSkippedProjectId,
      });
      const text =
        toolCall.status === "failed"
          ? t("agentChat.toolCall.siteResult.saveSkippedAfterFailure", {
              target: skippedProjectTarget,
            })
          : t("agentChat.toolCall.siteResult.saveSkipped", {
              target: skippedProjectTarget,
            });
      notices.push({
        key: "site-save-skipped",
        text,
        tone: siteResultSummary.saveErrorMessage ? "warning" : "neutral",
      });
    }

    if (siteResultSummary.saveErrorMessage) {
      notices.push({
        key: "site-save-error",
        text: t("agentChat.toolCall.siteResult.saveError", {
          message: siteResultSummary.saveErrorMessage,
        }),
        tone: "error",
      });
    }
    return notices;
  }, [resolveSiteProjectTargetDisplay, siteResultSummary, t, toolCall.status]);
  const resultPath = useMemo(() => {
    const presentation = resolveToolResultPath(resultMetadata);
    return presentation
      ? {
          label: t("agentChat.toolCall.resultFile"),
          ...presentation,
        }
      : undefined;
  }, [resultMetadata, t]);
  const openableFilePath = useMemo(
    () => resultPath?.value || filePath,
    [filePath, resultPath?.value],
  );
  const renderedResultContent = useMemo(
    () =>
      buildRenderedToolResultContent({
        toolCall,
        content: resultText,
        filePath,
        resultPath: resultPath?.value,
        emptyOutputLabel,
      }),
    [emptyOutputLabel, filePath, resultPath?.value, resultText, toolCall],
  );
  const toolHeadline = useMemo(
    () =>
      limeTaskProtocolFailureText ||
      buildToolHeadlineFromInfo({
        toolDisplay,
        subject: fileName,
        toolName: toolCall.name,
      }),
    [fileName, limeTaskProtocolFailureText, toolCall.name, toolDisplay],
  );
  const searchResultItems = useMemo(() => {
    if (!isUnifiedWebSearchToolName(toolCall.name)) {
      return [];
    }

    return resolveSearchResultPreviewItemsFromText(toolCall.result?.output);
  }, [toolCall.name, toolCall.result?.output]);
  const hasResultImages = resultImages.length > 0;
  const hasSearchResults = searchResultItems.length > 0;
  const isToolSearch = useMemo(
    () => isToolSearchToolName(toolCall.name),
    [toolCall.name],
  );
  const toolSearchSummary = useMemo(
    () => (isToolSearch ? normalizeToolSearchResultSummary(resultText) : null),
    [isToolSearch, resultText],
  );
  const hasToolSearchSummary = Boolean(toolSearchSummary);
  const groupedChildLine = useMemo(() => {
    if (isToolSearch && processNarrative.postSource === "tool_search") {
      return processNarrative.summary;
    }
    return buildGroupedChildLineFromInfo(toolCall);
  }, [
    isToolSearch,
    processNarrative.postSource,
    processNarrative.summary,
    toolCall,
  ]);
  const shouldShowRawSearchResultToggle =
    hasSearchResults && resultText !== emptyOutputLabel;
  const shouldRenderResultPanel =
    isExpanded &&
    hasResult &&
    (!hasSearchResults || showRawSearchResultOutput) &&
    !hasToolSearchSummary;
  const resolvedSkillContent =
    skillInvocationContentInfo.snapshotContent || fetchedSkillContent || "";
  const hasSkillContentAccess =
    skillInvocationContentInfo.isSkillInvocation &&
    Boolean(
      skillInvocationContentInfo.snapshotContent ||
      fetchedSkillContent ||
      skillInvocationContentInfo.skillName,
    );
  const skillContentSourceLabel = skillInvocationContentInfo.snapshotContent
    ? t("agentChat.toolCall.skillContent.source.snapshot")
    : t("agentChat.toolCall.skillContent.source.current");
  const skillContentTitle = skillInvocationContentInfo.snapshotContent
    ? t("agentChat.toolCall.skillContent.title.snapshot")
    : t("agentChat.toolCall.skillContent.title.current");
  const openInCanvasTitle = t("agentChat.toolCall.openInCanvas");
  const openInCanvasAriaLabel = openableFilePath
    ? t("agentChat.toolCall.openInCanvasWithTarget", {
        target: openableFilePath,
      })
    : openInCanvasTitle;
  const resultImageAlt = t("agentChat.toolCall.resultImage.alt");
  const resultImageOpenTitle = t("agentChat.markdown.image.openTitle");
  const openExternalUnsupportedMessage = t(
    "agentChat.toolCall.error.openExternalUnsupported",
  );

  const handleOpenExternalUrl = useCallback(
    async (url: string) => {
      try {
        await openExternalUrlWithSystemBrowser(url);
      } catch (error) {
        throw error instanceof Error
          ? error
          : new Error(openExternalUnsupportedMessage);
      }
    },
    [openExternalUnsupportedMessage],
  );

  useEffect(() => {
    setIsSkillContentExpanded(false);
    setFetchedSkillContent(null);
    setSkillContentLoading(false);
    setSkillContentError(null);
    setSkillMarkdownBodyExpanded(false);
    setExpandedDiffFileIds({});
    hasUserToggledExpandedRef.current = false;
  }, [toolCall.id]);

  useEffect(() => {
    if (defaultExpanded && !hasUserToggledExpandedRef.current) {
      setIsExpanded(true);
    }
  }, [defaultExpanded]);

  useEffect(() => {
    if (
      isMessageStreaming &&
      !isToolSearch &&
      (isRunning || hasResult || hasResultImages || hasSearchResults)
    ) {
      setIsExpanded(true);
    }
  }, [
    isMessageStreaming,
    isToolSearch,
    isRunning,
    hasResult,
    hasResultImages,
    hasSearchResults,
  ]);

  useEffect(() => {
    if (hasSearchResults && !hasUserToggledExpandedRef.current) {
      setIsExpanded(true);
    }
  }, [hasSearchResults]);

  useEffect(() => {
    if (!hasSearchResults) {
      setShowRawSearchResultOutput(false);
    }
  }, [hasSearchResults]);

  // 处理点击事件 - 如果是文件写入工具，打开右边栏
  const handleOpenFile = useCallback(() => {
    if (openableFilePath && onFileClick) {
      onFileClick(openableFilePath, fileContent || "");
    }
  }, [fileContent, onFileClick, openableFilePath]);

  const handleOpenSavedSiteContent = useCallback(() => {
    if (savedSiteContentTarget && onOpenSavedSiteContent) {
      onOpenSavedSiteContent(savedSiteContentTarget);
    }
  }, [onOpenSavedSiteContent, savedSiteContentTarget]);
  const openSavedSiteContentLabel =
    savedSiteContentTarget?.preferredTarget === "project_file"
      ? t("agentChat.toolCall.siteResult.openMarkdownPreview")
      : t("agentChat.toolCall.siteResult.openSavedContent");
  const resultToggleTitle = isExpanded
    ? t("agentChat.toolCall.result.collapse")
    : t("agentChat.toolCall.result.expand");
  const searchRawToggleLabel = showRawSearchResultOutput
    ? t("agentChat.toolCall.searchRaw.collapse")
    : t("agentChat.toolCall.searchRaw.expand");
  const searchRawToggleAriaLabel = showRawSearchResultOutput
    ? t("agentChat.toolCall.searchRaw.collapseAria")
    : t("agentChat.toolCall.searchRaw.expandAria");

  const handleToggleExpanded = useCallback(() => {
    hasUserToggledExpandedRef.current = true;
    setIsExpanded((prev) => !prev);
  }, []);

  const handleToggleDiffFileExpanded = useCallback((fileId: string) => {
    setExpandedDiffFileIds((current) => ({
      ...current,
      [fileId]: !current[fileId],
    }));
  }, []);

  const handleOpenDiffFileInCanvas = useCallback(
    (file: DiffReviewFile) => {
      if (!onFileClick) return;
      const title = t("agentChat.toolCall.diffReview.canvasTitle", {
        path: file.path,
      });
      const content = buildDiffFileCanvasContent({
        file,
        title,
        statusLabel: t("agentChat.toolCall.diffReview.canvasStatus", {
          status: t(`agentChat.toolCall.diffReview.status.${file.status}`),
        }),
        additionsLabel: t("agentChat.toolCall.diffReview.additions", {
          count: file.additions,
        }),
        deletionsLabel: t("agentChat.toolCall.diffReview.deletions", {
          count: file.deletions,
        }),
        hunksLabel: t("agentChat.toolCall.diffReview.hunks", {
          count: file.hunks,
        }),
      });
      onFileClick(`${file.path}.diff.md`, content);
    },
    [onFileClick, t],
  );

  const handleToggleSkillContent = useCallback(async () => {
    if (isSkillContentExpanded) {
      setIsSkillContentExpanded(false);
      setSkillMarkdownBodyExpanded(false);
      return;
    }

    setIsSkillContentExpanded(true);
    setSkillMarkdownBodyExpanded(false);
    if (skillInvocationContentInfo.snapshotContent || fetchedSkillContent) {
      return;
    }

    const skillName = skillInvocationContentInfo.skillName?.trim();
    if (!skillName) {
      setSkillContentError(
        t("agentChat.toolCall.skillContent.error.unavailable"),
      );
      return;
    }

    setSkillContentLoading(true);
    setSkillContentError(null);
    try {
      const inspection = await skillsApi.inspectLocalSkill(skillName);
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
    isSkillContentExpanded,
    skillInvocationContentInfo.skillName,
    skillInvocationContentInfo.snapshotContent,
    t,
  ]);

  const skillContentButton = hasSkillContentAccess ? (
    <ToolCallSkillContentButton
      isExpanded={isSkillContentExpanded}
      onToggle={handleToggleSkillContent}
    />
  ) : null;

  return (
    <div className={cn("group", grouped && "pl-1")}>
      {grouped ? (
        <div
          className="flex items-start gap-2 py-1.5"
          data-testid="tool-call-row"
          data-tool-call-id={toolCall.id}
          data-tool-name={toolCall.name}
          data-tool-status={toolCall.status}
          {...soulLifecycleAttributes}
        >
          <span className="pt-0.5 font-mono text-xs text-slate-400">
            {groupMarker}
          </span>
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm text-slate-700">
              {groupedChildLine}
            </div>
            {terminalInteractionSummary ? (
              <div
                className="mt-0.5 text-xs leading-5 text-slate-500"
                data-testid="tool-call-terminal-interactions"
              >
                {terminalInteractionSummary}
              </div>
            ) : null}
          </div>
          <div className="ml-auto flex items-center gap-1 pt-0.5">
            {skillContentButton}
            {openableFilePath && onFileClick && (
              <button
                onClick={handleOpenFile}
                className="rounded-md p-1 transition-colors hover:bg-slate-100"
                title={openInCanvasTitle}
                aria-label={openInCanvasAriaLabel}
              >
                <ExternalLink className="h-3.5 w-3.5 text-slate-500 hover:text-slate-900" />
              </button>
            )}
            {(hasResult || hasSearchResults) && (
              <button
                onClick={handleToggleExpanded}
                className="rounded-md p-1 transition-colors hover:bg-slate-100"
                title={resultToggleTitle}
              >
                <ChevronRight
                  className={cn(
                    "h-3.5 w-3.5 text-slate-500 transition-transform",
                    isExpanded && "rotate-90",
                  )}
                />
              </button>
            )}
          </div>
        </div>
      ) : (
        <div
          className="flex items-start gap-2.5 py-1.5"
          data-testid="tool-call-row"
          data-tool-call-id={toolCall.id}
          data-tool-name={toolCall.name}
          data-tool-status={toolCall.status}
          {...soulLifecycleAttributes}
        >
          <span
            className={cn(
              "pt-0.5 text-sm font-medium",
              isCompleted && "text-emerald-600",
              isFailed && "text-rose-600",
              isRunning && "text-sky-600",
              !isCompleted && !isFailed && !isRunning && "text-slate-400",
            )}
            aria-hidden="true"
          >
            •
          </span>

          <div className="min-w-0 flex-1">
            <div className="truncate text-sm text-slate-900">
              {toolHeadline}
            </div>
            {terminalInteractionSummary ? (
              <div
                className="mt-0.5 text-xs leading-5 text-slate-500"
                data-testid="tool-call-terminal-interactions"
              >
                {terminalInteractionSummary}
              </div>
            ) : null}
          </div>

          <div className="ml-auto flex items-center gap-1 pt-0.5">
            {skillContentButton}
            {openableFilePath && onFileClick && (
              <button
                onClick={handleOpenFile}
                className="rounded-md p-1 transition-colors hover:bg-slate-100"
                title={openInCanvasTitle}
                aria-label={openInCanvasAriaLabel}
              >
                <ExternalLink className="h-3.5 w-3.5 text-slate-500 hover:text-slate-900" />
              </button>
            )}

            {(hasResult || hasSearchResults) && (
              <button
                onClick={handleToggleExpanded}
                className="rounded-md p-1 transition-colors hover:bg-slate-100"
                title={resultToggleTitle}
              >
                <ChevronRight
                  className={cn(
                    "h-3.5 w-3.5 text-slate-500 transition-transform",
                    isExpanded && "rotate-90",
                  )}
                />
              </button>
            )}
          </div>
        </div>
      )}

      {hasResultImages && (
        <div className="mb-2 ml-4 mt-2 flex flex-wrap gap-2">
          {resultImages.map((image, index) => (
            <button
              key={`${image.src.slice(0, 48)}-${index}`}
              className="overflow-hidden rounded-lg border border-slate-200 bg-white"
              onClick={() => setPreviewImageSrc(image.src)}
              title={resultImageOpenTitle}
            >
              <img
                src={image.src}
                alt={resultImageAlt}
                className="h-20 w-20 object-cover"
                loading="lazy"
              />
            </button>
          ))}
        </div>
      )}

      {hasSearchResults && isExpanded && (
        <div className="mb-2 ml-6 mt-1.5">
          <SearchResultPreviewList
            items={searchResultItems}
            onOpenUrl={handleOpenExternalUrl}
            popoverSide="bottom"
            popoverAlign="start"
          />
          {shouldShowRawSearchResultToggle ? (
            <div className="mt-2">
              <button
                type="button"
                className="rounded-md px-2 py-1 text-[11px] text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700"
                aria-label={searchRawToggleAriaLabel}
                onClick={() =>
                  setShowRawSearchResultOutput((current) => !current)
                }
              >
                {searchRawToggleLabel}
              </button>
            </div>
          ) : null}
        </div>
      )}

      {toolSearchSummary && isExpanded ? (
        <div
          className="mb-2 ml-6 mt-1.5"
          data-testid="tool-call-tool-search-result"
        >
          <ToolSearchSummaryPanel summary={toolSearchSummary} />
        </div>
      ) : null}

      {isSkillContentExpanded ? (
        <ToolCallSkillContentPanel
          toolCallId={toolCall.id}
          skillInvocationContentInfo={skillInvocationContentInfo}
          sourceLabel={skillContentSourceLabel}
          title={skillContentTitle}
          bodyExpanded={skillMarkdownBodyExpanded}
          onToggleBodyExpanded={() =>
            setSkillMarkdownBodyExpanded((current) => !current)
          }
          loading={skillContentLoading}
          error={skillContentError}
          content={resolvedSkillContent}
        />
      ) : null}

      {shouldRenderResultPanel && (
        <ToolCallDisplayResultPanel
          toolCallId={toolCall.id}
          commandSummary={commandSummary}
          commandOutputStreams={commandOutputStreams}
          commandSurfaceLabel={commandSurfaceLabel}
          commandEncoding={commandEncoding}
          commandDecodedWithLabel={commandDecodedWithLabel}
          diffReviewSummary={diffReviewSummary}
          diffReviewScopeItems={diffReviewScopeItems}
          expandedDiffFileIds={expandedDiffFileIds}
          onToggleDiffFileExpanded={handleToggleDiffFileExpanded}
          onOpenDiffFileInCanvas={handleOpenDiffFileInCanvas}
          canOpenDiffFileInCanvas={Boolean(onFileClick)}
          resultMetaItems={resultMetaItems}
          siteResultNotices={siteResultNotices}
          showOpenSavedSiteContent={Boolean(
            savedSiteContentTarget && onOpenSavedSiteContent,
          )}
          onOpenSavedSiteContent={handleOpenSavedSiteContent}
          openSavedSiteContentLabel={openSavedSiteContentLabel}
          resultPath={resultPath}
          isResultFailure={isResultFailure}
          renderedResultContent={renderedResultContent}
        />
      )}

      {previewImageSrc && (
        <button
          type="button"
          className="fixed inset-0 z-50 bg-[linear-gradient(180deg,rgba(240,249,255,0.88)_0%,rgba(236,253,245,0.8)_52%,rgba(255,255,255,0.92)_100%)] p-6 backdrop-blur-[2px]"
          onClick={() => setPreviewImageSrc(null)}
        >
          <img
            src={previewImageSrc}
            alt={resultImageAlt}
            className="mx-auto max-h-full max-w-full rounded-lg object-contain"
          />
        </button>
      )}
    </div>
  );
};

// 导出别名，用于交错显示模式
export const ToolCallItem = ToolCallDisplay;
