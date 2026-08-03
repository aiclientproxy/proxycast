import { useTranslation } from "react-i18next";
import { ArrowUpRight, FileStack, FileText } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  resolveArtifactDocumentCurrentVersion,
  resolveArtifactDocumentCurrentVersionDiff,
  type ArtifactDocumentBlock,
  type ArtifactDocumentKind,
  type ArtifactDocumentStatus,
  type ArtifactDocumentV1,
  type ArtifactDocumentVersionDiff,
} from "@/lib/artifact-document";
import {
  resolveArtifactProtocolDocumentPayload,
  resolveArtifactProtocolPreviewText,
} from "@/lib/artifact-protocol";
import {
  readAgentRuntimeTimelineArtifactContent,
  type AgentRuntimeTimelineArtifactContent,
} from "@/lib/api/agentRuntime/appServerArtifactClient";
import type { AgentThreadItem } from "../types";
import {
  resolveTimelineArtifactNavigation,
  type ArtifactTimelineOpenTarget,
} from "../utils/artifactTimelineNavigation";
import { isTimelineReadOnlyFileArtifact } from "../utils/timelineFileArtifactKind";
import { useLatestAgentUiProjectionEventForArtifact } from "../projection/useConversationProjectionStore";
import {
  formatAgentUiProjectionEventType,
  formatAgentUiProjectionPhase,
  formatAgentUiProjectionSourceType,
  type AgentUiProjectionTranslation,
} from "../projection/agentUiProjectionSummary";

interface AgentThreadTimelineArtifactCardProps {
  item: Extract<AgentThreadItem, { type: "file_artifact" }>;
  timestamp?: string | null;
  onFileClick?: (fileName: string, content: string) => void;
  onOpenArtifactFromTimeline?: (target: ArtifactTimelineOpenTarget) => void;
  readTimelineArtifactContent?: (
    item: Extract<AgentThreadItem, { type: "file_artifact" }>,
  ) => Promise<AgentRuntimeTimelineArtifactContent | null>;
  sourceMessageId?: string;
  onSaveFileArtifactAsKnowledge?: (source: {
    messageId: string;
    content: string;
    sourceName?: string;
    description?: string | null;
  }) => void;
  groupedAttachment?: boolean;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function normalizeText(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function readMetadataText(
  metadata: Record<string, unknown> | undefined,
  keys: string[],
): string | undefined {
  for (const key of keys) {
    const direct = normalizeText(metadata?.[key]);
    if (direct) {
      return direct;
    }
  }
  return undefined;
}

function readMetadataNumber(
  metadata: Record<string, unknown> | undefined,
  keys: string[],
): number | undefined {
  for (const key of keys) {
    const value = metadata?.[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === "string" && value.trim()) {
      const parsed = Number(value.trim());
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }
  return undefined;
}

function readMetadataRecord(
  metadata: Record<string, unknown> | undefined,
  keys: string[],
): Record<string, unknown> | undefined {
  for (const key of keys) {
    const direct = asRecord(metadata?.[key]);
    if (direct) {
      return direct;
    }
  }
  return undefined;
}

function readMetadataArray(
  metadata: Record<string, unknown> | undefined,
  keys: string[],
): unknown[] | undefined {
  for (const key of keys) {
    const direct = metadata?.[key];
    if (Array.isArray(direct)) {
      return direct;
    }
  }
  return undefined;
}

function resolveVersionDiffChangedBlockCount(
  diff:
    | ArtifactDocumentVersionDiff
    | Record<string, unknown>
    | null
    | undefined,
): number {
  const record = asRecord(diff);
  if (!record) {
    return 0;
  }

  const changedBlocks = Array.isArray(record.changedBlocks)
    ? record.changedBlocks
    : Array.isArray(record.changed_blocks)
      ? record.changed_blocks
      : [];
  return changedBlocks.length;
}

function resolveFileName(path: string): string {
  const normalized = path.replace(/\\/g, "/").trim();
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
}

function resolveFileExtension(path: string): string | null {
  const fileName = resolveFileName(path).split(/[?#]/)[0] || "";
  const match = /\.([a-z0-9]+)$/i.exec(fileName);
  return match?.[1]?.toUpperCase() || null;
}

function resolveFileKindKey(path: string): string {
  const extension = resolveFileExtension(path)?.toLowerCase();
  if (
    extension &&
    ["md", "markdown", "txt", "doc", "docx", "pdf"].includes(extension)
  ) {
    return "agentChat.messageList.fileArtifact.kind.document";
  }
  if (
    extension &&
    [
      "js",
      "jsx",
      "ts",
      "tsx",
      "json",
      "css",
      "html",
      "py",
      "rs",
      "go",
      "java",
      "c",
      "cpp",
      "h",
      "hpp",
      "toml",
      "yaml",
      "yml",
    ].includes(extension)
  ) {
    return "agentChat.messageList.fileArtifact.kind.code";
  }
  return "agentChat.messageList.fileArtifact.kind.file";
}

function truncateMiddle(value: string, maxLength = 72): string {
  const normalized = value.trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }

  const headLength = Math.max(20, Math.ceil((maxLength - 1) * 0.58));
  const tailLength = Math.max(14, maxLength - headLength - 1);
  return `${normalized.slice(0, headLength)}…${normalized.slice(-tailLength)}`;
}

function truncateInlineText(value: string, maxLength = 160): string {
  const normalized = value.trim().replace(/\s+/g, " ");
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, maxLength - 1).trimEnd()}…`;
}

function resolveArtifactDocumentKindLabel(
  kind?: ArtifactDocumentKind | string,
): string | null {
  switch (kind) {
    case "report":
      return "报告";
    case "roadmap":
      return "路线图";
    case "prd":
      return "PRD";
    case "brief":
      return "简报";
    case "analysis":
      return "分析";
    case "comparison":
      return "对比";
    case "plan":
      return "计划";
    case "table_report":
      return "表格报告";
    default:
      return kind || null;
  }
}

function resolveArtifactDocumentStatusLabel(
  status?: ArtifactDocumentStatus | string,
): string | null {
  switch (status) {
    case "draft":
      return "草稿";
    case "streaming":
      return "生成中";
    case "ready":
      return "可阅读";
    case "failed":
      return "失败";
    case "archived":
      return "已归档";
    default:
      return status || null;
  }
}

function resolveArtifactSourceLabel(source?: string): string | null {
  switch (source) {
    case "artifact_snapshot":
      return "已同步";
    case "artifact_document_service":
      return "文稿服务";
    case "tool_result":
      return "处理结果";
    case "tool_start":
      return "开始处理";
    case "message_content":
      return "消息内容";
    default:
      return source && !source.includes("_") ? source : null;
  }
}

function resolveBlockLabel(
  document: ArtifactDocumentV1 | null,
  blockId: string,
): string {
  const block = document?.blocks.find((entry) => entry.id === blockId);
  if (!block) {
    return blockId;
  }

  const record = block as ArtifactDocumentBlock & Record<string, unknown>;
  const fallbackByType: Record<string, string> = {
    hero_summary: "摘要",
    section_header: "章节",
    rich_text: "正文",
    callout: "提示",
    key_points: "要点",
  };
  const label =
    normalizeText(record.title) ||
    normalizeText(record.summary) ||
    normalizeText(record.description) ||
    normalizeText(record.label) ||
    normalizeText(record.text) ||
    normalizeText(record.markdown);

  return label
    ? truncateInlineText(label, 20)
    : fallbackByType[block.type] || blockId;
}

function resolveFallbackPreview(content: string | undefined): string | null {
  const normalized = normalizeText(content);
  if (!normalized) {
    return null;
  }

  if (/^[[{]/.test(normalized)) {
    return "包含结构化结果，点击在画布中查看完整内容。";
  }

  return truncateInlineText(normalized);
}

function hasDocumentFileExtension(path: string): boolean {
  const normalizedPath = path.replace(/\\/g, "/").split(/[?#]/)[0] || "";
  return /\.(md|markdown|txt)$/i.test(normalizedPath);
}

function looksLikeMarkdownDocument(content: string): boolean {
  return /^#\s+.+/m.test(content) || /\n#{2,6}\s+.+/m.test(content);
}

function resolveKnowledgeDocumentSource({
  item,
  metadata,
  displayTitle,
}: {
  item: Extract<AgentThreadItem, { type: "file_artifact" }>;
  metadata: Record<string, unknown> | undefined;
  displayTitle: string;
}): {
  content: string;
  sourceName: string;
  description: string | null;
} | null {
  const content = normalizeText(item.content);
  if (!content || content.length < 24 || /^[[{]/.test(content)) {
    return null;
  }

  if (
    !hasDocumentFileExtension(item.path) &&
    !looksLikeMarkdownDocument(content)
  ) {
    return null;
  }

  const sourceName =
    readMetadataText(metadata, [
      "fileName",
      "filename",
      "sourceName",
      "source_name",
      "artifactFileName",
      "artifact_file_name",
    ]) || resolveFileName(item.path);

  return {
    content,
    sourceName,
    description: displayTitle || sourceName || null,
  };
}

function resolveDocumentPreview(
  document: ArtifactDocumentV1 | null,
  displayTitle: string,
  syncedPreviewText: string,
): string | null {
  if (!document) {
    return null;
  }

  const preview = normalizeText(resolveArtifactProtocolPreviewText(document));
  if (!preview || preview === displayTitle) {
    return syncedPreviewText;
  }

  return truncateInlineText(preview);
}

// eslint-disable-next-line react-refresh/only-export-components
export function isPlainTimelineFileAttachment(
  item: AgentThreadItem,
): item is Extract<AgentThreadItem, { type: "file_artifact" }> {
  if (item.type !== "file_artifact") {
    return false;
  }
  if (isTimelineReadOnlyFileArtifact(item)) {
    return true;
  }
  const metadata = asRecord(item.metadata);
  const document = resolveArtifactProtocolDocumentPayload({
    content: item.content,
    metadata,
  });
  const metadataVersion = asRecord(metadata?.artifactVersion);
  const hasArtifactDocumentFacts = Boolean(
    normalizeText(document?.artifactId) ||
    readMetadataText(metadata, [
      "artifactId",
      "artifact_id",
      "artifactDocumentId",
      "artifact_document_id",
      "artifactKind",
      "artifact_kind",
      "artifactStatus",
      "artifact_status",
      "artifactVersionNo",
      "artifact_version_no",
      "artifactVersionId",
      "artifact_version_id",
      "snapshotPath",
      "snapshot_path",
    ]) ||
    readMetadataText(metadataVersion, [
      "id",
      "versionId",
      "version_id",
      "snapshotPath",
      "snapshot_path",
    ]) ||
    readMetadataNumber(metadata, [
      "artifactVersionNo",
      "artifact_version_no",
      "versionNo",
      "version_no",
    ]),
  );
  return !document && !hasArtifactDocumentFacts;
}

export function AgentThreadTimelineArtifactCard({
  item,
  timestamp,
  onFileClick,
  onOpenArtifactFromTimeline,
  readTimelineArtifactContent = readAgentRuntimeTimelineArtifactContent,
  sourceMessageId,
  onSaveFileArtifactAsKnowledge,
  groupedAttachment = false,
}: AgentThreadTimelineArtifactCardProps) {
  const { t } = useTranslation("agent");
  const translateProjection: AgentUiProjectionTranslation = (key, options) =>
    String(t(key as never, options as never));
  const metadata = asRecord(item.metadata);
  const navigation = resolveTimelineArtifactNavigation(item);
  const blockTargets = navigation?.blockTargets || [];
  const shouldOpenFocusedBlock =
    Boolean(onOpenArtifactFromTimeline) && blockTargets.length === 1;
  const document = resolveArtifactProtocolDocumentPayload({
    content: item.content,
    metadata,
  });
  const metadataVersion = asRecord(metadata?.artifactVersion);
  const currentVersion = document
    ? resolveArtifactDocumentCurrentVersion(document)
    : null;
  const documentMetadata = asRecord(document?.metadata);
  const metadataTitle = readMetadataText(metadata, [
    "artifactTitle",
    "artifact_title",
    "title",
  ]);
  const artifactProjectionId =
    normalizeText(document?.artifactId) ||
    readMetadataText(metadata, [
      "artifactId",
      "artifact_id",
      "artifactDocumentId",
      "artifact_document_id",
    ]);
  const latestArtifactProjection =
    useLatestAgentUiProjectionEventForArtifact(artifactProjectionId);
  const metadataKind = readMetadataText(metadata, [
    "artifactKind",
    "artifact_kind",
    "kind",
  ]);
  const metadataStatus =
    readMetadataText(metadata, [
      "artifactStatus",
      "artifact_status",
      "status",
    ]) || normalizeText(metadataVersion?.status);
  const metadataVersionNo =
    readMetadataNumber(metadata, [
      "artifactVersionNo",
      "artifact_version_no",
      "versionNo",
      "version_no",
    ]) || readMetadataNumber(metadataVersion, ["versionNo", "version_no"]);
  const metadataVersionId =
    readMetadataText(metadata, [
      "artifactVersionId",
      "artifact_version_id",
      "versionId",
      "version_id",
    ]) ||
    readMetadataText(metadataVersion, ["id", "versionId", "version_id"]) ||
    normalizeText(currentVersion?.id) ||
    readMetadataText(documentMetadata, [
      "currentVersionId",
      "current_version_id",
    ]);
  const snapshotPath =
    readMetadataText(metadata, ["snapshotPath", "snapshot_path"]) ||
    readMetadataText(metadataVersion, ["snapshotPath", "snapshot_path"]) ||
    normalizeText(currentVersion?.snapshotPath);
  const metadataVersionDiff =
    readMetadataRecord(metadata, [
      "currentVersionDiff",
      "current_version_diff",
      "artifactVersionDiff",
      "artifact_version_diff",
    ]) ||
    readMetadataRecord(metadataVersion, [
      "currentVersionDiff",
      "current_version_diff",
      "artifactVersionDiff",
      "artifact_version_diff",
    ]);
  const versionDiff =
    (document ? resolveArtifactDocumentCurrentVersionDiff(document) : null) ||
    metadataVersionDiff;
  const diffChangedBlockCount =
    resolveVersionDiffChangedBlockCount(versionDiff);
  const validationIssueCount =
    readMetadataNumber(metadata, [
      "validationIssueCount",
      "validation_issue_count",
    ]) ||
    readMetadataNumber(metadataVersion, [
      "validationIssueCount",
      "validation_issue_count",
    ]) ||
    readMetadataArray(metadata, ["validationIssues", "validation_issues"])
      ?.length ||
    0;
  const metadataPreview = readMetadataText(metadata, [
    "previewText",
    "preview_text",
    "artifactSummary",
    "artifact_summary",
    "summary",
  ]);
  const displayTitle =
    normalizeText(document?.title) ||
    metadataTitle ||
    resolveFileName(item.path);
  const displayPath = truncateMiddle(item.path, 84);
  const previewText =
    resolveDocumentPreview(
      document,
      displayTitle,
      t("agentChat.messageList.artifact.documentSyncedPreview"),
    ) ||
    (metadataPreview ? truncateInlineText(metadataPreview) : null) ||
    resolveFallbackPreview(item.content) ||
    "点击在画布中打开完整内容。";
  const sourceLabel = resolveArtifactSourceLabel(item.source);
  const kindLabel = resolveArtifactDocumentKindLabel(
    document?.kind || metadataKind,
  );
  const statusLabel = resolveArtifactDocumentStatusLabel(
    currentVersion?.status || document?.status || metadataStatus,
  );
  const versionNo = currentVersion?.versionNo || metadataVersionNo;
  const hasCheckpointFact = Boolean(
    snapshotPath || versionNo || metadataVersionId,
  );
  const blockCount = document?.blocks.length || 0;
  const sourceCount = document?.sources.length || 0;
  const knowledgeDocumentSource = resolveKnowledgeDocumentSource({
    item,
    metadata,
    displayTitle,
  });
  const canSaveAsKnowledge = Boolean(
    onSaveFileArtifactAsKnowledge && sourceMessageId && knowledgeDocumentSource,
  );
  const hasArtifactDocumentFacts = Boolean(
    artifactProjectionId ||
    metadataKind ||
    metadataStatus ||
    metadataVersionNo ||
    metadataVersionId ||
    snapshotPath ||
    metadataVersionDiff ||
    validationIssueCount > 0,
  );
  const resolveOpenTarget = async (
    target: ArtifactTimelineOpenTarget,
  ): Promise<ArtifactTimelineOpenTarget> => {
    if (target.content.trim()) {
      return target;
    }

    const artifactContent = await readTimelineArtifactContent(item);
    if (!artifactContent?.content.trim()) {
      return target;
    }

    return {
      ...target,
      artifactId: artifactContent.artifactId || target.artifactId,
      content: artifactContent.content,
      filePath: artifactContent.filePath || target.filePath,
    };
  };

  const openTimelineTarget = async (target: ArtifactTimelineOpenTarget) => {
    const resolvedTarget = await resolveOpenTarget(target).catch(() => target);
    onOpenArtifactFromTimeline?.(resolvedTarget);
  };
  const openFileAttachment = () => {
    if (onOpenArtifactFromTimeline && navigation) {
      void openTimelineTarget(navigation.rootTarget);
      return;
    }

    if (item.content?.trim()) {
      onFileClick?.(item.path, item.content);
      return;
    }

    void readTimelineArtifactContent(item)
      .then((artifactContent) => {
        onFileClick?.(
          artifactContent?.filePath || item.path,
          artifactContent?.content || "",
        );
      })
      .catch(() => {
        onFileClick?.(item.path, "");
      });
  };

  if (!document && !hasArtifactDocumentFacts) {
    const fileName = resolveFileName(item.path);
    const extension = resolveFileExtension(item.path);
    const fileKind = String(t(resolveFileKindKey(item.path) as never));
    const openAriaLabel = String(
      t(
        "agentChat.messageList.fileArtifact.openAria" as never,
        { name: fileName } as never,
      ),
    );

    return (
      <div className={groupedAttachment ? "" : "py-1.5"}>
        <div
          data-testid="timeline-file-attachment-card"
          className={
            groupedAttachment
              ? "bg-white text-left"
              : "overflow-hidden rounded-lg border border-slate-200 bg-white text-left"
          }
        >
          <div className="flex min-h-16 items-center gap-3 px-3 py-2.5">
            <button
              type="button"
              className="group flex min-w-0 flex-1 items-center gap-3 text-left"
              onClick={openFileAttachment}
              aria-label={openAriaLabel}
            >
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-slate-50 text-slate-500">
                <FileText className="h-[18px] w-[18px]" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-semibold leading-5 text-slate-900">
                  {fileName}
                </span>
                <span className="mt-0.5 block text-xs leading-5 text-slate-500">
                  {fileKind}
                  {extension ? ` · ${extension}` : ""}
                </span>
              </span>
            </button>

            {canSaveAsKnowledge && knowledgeDocumentSource ? (
              <button
                type="button"
                className="hidden shrink-0 rounded-lg border border-sky-200 bg-white px-2.5 py-1.5 text-xs font-semibold text-sky-700 transition hover:border-sky-300 hover:bg-sky-50 sm:inline-flex"
                onClick={() =>
                  onSaveFileArtifactAsKnowledge?.({
                    messageId: sourceMessageId!,
                    content: knowledgeDocumentSource.content,
                    sourceName: knowledgeDocumentSource.sourceName,
                    description: knowledgeDocumentSource.description,
                  })
                }
              >
                {t("agentChat.messageList.artifact.saveDocument")}
              </button>
            ) : null}

            <button
              type="button"
              className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
              onClick={openFileAttachment}
              aria-label={openAriaLabel}
            >
              <ArrowUpRight className="h-3.5 w-3.5" aria-hidden="true" />
              {t("agentChat.messageList.fileArtifact.openFile")}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="py-1.5">
      <div
        data-testid="timeline-file-artifact-card"
        className={
          knowledgeDocumentSource
            ? "flex w-full flex-col items-stretch gap-2 rounded-[18px] border border-sky-200/80 bg-sky-50 p-2 text-left shadow-sm shadow-sky-950/5 sm:flex-row"
            : "rounded-[18px] border border-slate-200 bg-white px-4 py-3 text-left shadow-sm shadow-slate-950/5 transition hover:border-sky-200 hover:bg-sky-50/40"
        }
      >
        <button
          type="button"
          className={
            knowledgeDocumentSource
              ? "group flex min-w-0 flex-1 items-start gap-3 rounded-[14px] px-2 py-2 text-left transition hover:bg-white"
              : "group flex w-full items-start gap-3 text-left"
          }
          onClick={() => {
            if (onOpenArtifactFromTimeline && navigation) {
              void openTimelineTarget(
                shouldOpenFocusedBlock
                  ? blockTargets[0]
                  : navigation.rootTarget,
              );
              return;
            }

            if (item.content?.trim()) {
              onFileClick?.(item.path, item.content);
              return;
            }

            void readTimelineArtifactContent(item)
              .then((artifactContent) => {
                onFileClick?.(
                  artifactContent?.filePath || item.path,
                  artifactContent?.content || "",
                );
              })
              .catch(() => {
                onFileClick?.(item.path, "");
              });
          }}
        >
          <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-slate-200 bg-slate-50 text-slate-600">
            {document ? (
              <FileStack className="h-[18px] w-[18px]" />
            ) : (
              <FileText className="h-[18px] w-[18px]" />
            )}
          </div>

          <div className="min-w-0 flex-1">
            {knowledgeDocumentSource ? (
              <div className="mb-1.5 flex flex-wrap items-center gap-1.5">
                <Badge
                  variant="outline"
                  className="border-sky-200 bg-white text-sky-700"
                >
                  {t("agentChat.messageList.artifact.documentBadge")}
                </Badge>
                <Badge
                  variant="outline"
                  className="border-emerald-200 bg-emerald-50 text-emerald-700"
                >
                  {t("agentChat.messageList.artifact.saveableBadge")}
                </Badge>
              </div>
            ) : null}
            <div className="flex flex-wrap items-center gap-2">
              <div className="min-w-0 flex-1 text-sm font-medium leading-6 text-slate-900">
                <span className="line-clamp-1 break-all">{displayTitle}</span>
              </div>
              {kindLabel ? (
                <Badge
                  variant="outline"
                  className="border-sky-200 bg-sky-50 text-sky-700"
                >
                  {kindLabel}
                </Badge>
              ) : null}
              {statusLabel ? (
                <Badge
                  variant="outline"
                  className="border-emerald-200 bg-emerald-50 text-emerald-700"
                >
                  {statusLabel}
                </Badge>
              ) : null}
              {sourceLabel ? (
                <Badge
                  variant="outline"
                  className="border-slate-200 bg-slate-50 text-slate-600"
                >
                  {sourceLabel}
                </Badge>
              ) : null}
              {latestArtifactProjection ? (
                <Badge
                  variant="outline"
                  className="border-sky-200 bg-white text-sky-700"
                  data-testid="timeline-file-artifact-agentui"
                  title={[
                    t("agentChat.messageList.artifact.agentUiProjectionTitle"),
                    formatAgentUiProjectionSourceType(
                      latestArtifactProjection.sourceType,
                      translateProjection,
                    ),
                    formatAgentUiProjectionPhase(
                      latestArtifactProjection.phase,
                      translateProjection,
                    ),
                  ].join(" · ")}
                >
                  {t("agentChat.messageList.artifact.agentUiBadgePrefix")}{" "}
                  {formatAgentUiProjectionEventType(
                    latestArtifactProjection.type,
                    translateProjection,
                  )}
                </Badge>
              ) : null}
              {timestamp ? (
                <span className="text-xs text-slate-400">{timestamp}</span>
              ) : null}
            </div>

            <div
              data-testid="timeline-file-artifact-preview"
              className="mt-2 text-sm leading-6 text-slate-600"
            >
              {previewText}
            </div>

            <div className="mt-2 flex flex-wrap items-center gap-2">
              <span
                title={item.path}
                className="inline-flex max-w-full rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 font-mono text-[11px] text-slate-500"
              >
                <span className="truncate">{displayPath}</span>
              </span>
              {versionNo ? (
                <span className="inline-flex rounded-full bg-slate-100 px-2 py-1 text-[11px] text-slate-500">
                  V{versionNo}
                </span>
              ) : null}
              {hasCheckpointFact ? (
                <span
                  title={snapshotPath || metadataVersionId || undefined}
                  className="inline-flex rounded-full bg-emerald-50 px-2 py-1 text-[11px] text-emerald-700"
                >
                  {t("agentChat.messageList.artifact.checkpointBadge")}
                </span>
              ) : null}
              {diffChangedBlockCount > 0 ? (
                <span className="inline-flex rounded-full bg-sky-50 px-2 py-1 text-[11px] text-sky-700">
                  {t("agentChat.messageList.artifact.diffBadge", {
                    count: diffChangedBlockCount,
                  })}
                </span>
              ) : null}
              {validationIssueCount > 0 ? (
                <span className="inline-flex rounded-full bg-amber-50 px-2 py-1 text-[11px] text-amber-700">
                  {t("agentChat.messageList.artifact.validationBadge", {
                    count: validationIssueCount,
                  })}
                </span>
              ) : null}
              {blockCount > 0 ? (
                <span className="inline-flex rounded-full bg-slate-100 px-2 py-1 text-[11px] text-slate-500">
                  {t("agentChat.messageList.artifact.blockCount", {
                    count: blockCount,
                  })}
                </span>
              ) : null}
              {sourceCount > 0 ? (
                <span className="inline-flex rounded-full bg-slate-100 px-2 py-1 text-[11px] text-slate-500">
                  {t("agentChat.messageList.artifact.sourceCount", {
                    count: sourceCount,
                  })}
                </span>
              ) : null}
              <span className="inline-flex items-center gap-1 text-xs text-slate-400 transition group-hover:text-sky-700">
                <span>{t("agentChat.toolCall.openInCanvas")}</span>
                <ArrowUpRight className="h-3.5 w-3.5" />
              </span>
            </div>
          </div>
        </button>

        {canSaveAsKnowledge && knowledgeDocumentSource ? (
          <button
            type="button"
            className="flex shrink-0 items-center justify-center rounded-xl border border-sky-200 bg-white px-3 py-2 text-xs font-semibold text-sky-700 transition hover:border-sky-300 hover:bg-sky-100 hover:text-sky-800 sm:py-0"
            onClick={() =>
              onSaveFileArtifactAsKnowledge?.({
                messageId: sourceMessageId!,
                content: knowledgeDocumentSource.content,
                sourceName: knowledgeDocumentSource.sourceName,
                description: knowledgeDocumentSource.description,
              })
            }
          >
            {t("agentChat.messageList.artifact.saveDocument")}
          </button>
        ) : null}
      </div>

      {onOpenArtifactFromTimeline && blockTargets.length > 1 ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {blockTargets.slice(0, 4).map((target) => (
            <button
              key={`${item.id}:${target.blockId}`}
              type="button"
              className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 transition hover:border-sky-200 hover:bg-sky-50 hover:text-sky-700"
              onClick={() => void openTimelineTarget(target)}
            >
              {t("agentChat.messageList.artifact.locateBlock", {
                label: resolveBlockLabel(document, target.blockId || ""),
              })}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
