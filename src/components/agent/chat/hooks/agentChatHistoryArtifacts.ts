import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import type {
  Artifact,
  ArtifactStatus,
  ArtifactType,
} from "@/lib/artifact/types";
import type { Message } from "../types";
import { mergeArtifacts } from "../utils/messageArtifacts";
import {
  asHistoryRecord,
  fileNameFromHistoryPath,
  parseHistoryTimestamp,
  readHistoryMetadataString,
  readHistoryNumber,
  readHistoryString,
} from "./agentChatHistoryPrimitives";
import {
  isWorkspaceArticlePatchArtifactPath,
  isWorkspaceArticlePatchArtifactKind,
  readWorkspaceArticlePatchRecordFromMetadata,
} from "../workspace/workspaceArticleWorkspaceMetadata";

type HistoryArtifactSummary = {
  artifactRef?: unknown;
  eventId?: unknown;
  sequence?: unknown;
  sessionId?: unknown;
  turnId?: unknown;
  artifactId?: unknown;
  path?: unknown;
  title?: unknown;
  kind?: unknown;
  status?: unknown;
  contentStatus?: unknown;
  metadata?: unknown;
};

const HISTORY_ARTIFACT_TYPES = new Set<ArtifactType>([
  "document",
  "code",
  "html",
  "svg",
  "mermaid",
  "react",
  "canvas:document",
  "canvas:video",
  "canvas:design",
]);

function historyArtifactTypeFromSummary(
  summary: HistoryArtifactSummary,
  path: string,
  metadata: Record<string, unknown> | null,
): ArtifactType {
  const explicit =
    readHistoryString(summary.kind) ||
    readHistoryMetadataString(metadata, ["artifact_type", "type", "kind"]);
  const normalizedExplicit = explicit.toLowerCase();
  if (HISTORY_ARTIFACT_TYPES.has(normalizedExplicit as ArtifactType)) {
    return normalizedExplicit as ArtifactType;
  }
  if (normalizedExplicit === "markdown" || normalizedExplicit === "text") {
    return "document";
  }
  if (normalizedExplicit === "artifact_document") {
    return "document";
  }
  const extension = fileNameFromHistoryPath(path)
    .split(".")
    .pop()
    ?.toLowerCase();
  if (extension === "html" || extension === "htm") {
    return "html";
  }
  if (extension === "svg") {
    return "svg";
  }
  if (extension === "mmd" || extension === "mermaid") {
    return "mermaid";
  }
  if (extension === "jsx" || extension === "tsx") {
    return "react";
  }
  if (extension === "md" || extension === "markdown" || extension === "txt") {
    return "document";
  }
  return "code";
}

function historyArtifactStatusFromSummary(
  summary: HistoryArtifactSummary,
): ArtifactStatus {
  const status = readHistoryString(summary.status).toLowerCase();
  if (status === "pending") {
    return "pending";
  }
  if (status === "streaming" || status === "running") {
    return "streaming";
  }
  if (status === "error" || status === "failed") {
    return "error";
  }
  return "complete";
}

function readHistoryArtifactSummaries(
  value: unknown,
): HistoryArtifactSummary[] {
  const record = asHistoryRecord(value);
  const artifacts = record?.artifacts;
  return Array.isArray(artifacts)
    ? artifacts.filter((artifact): artifact is HistoryArtifactSummary =>
        Boolean(asHistoryRecord(artifact)),
      )
    : [];
}

function collectHistoryArtifactSummaries(
  detail: AgentSessionDetail,
): HistoryArtifactSummary[] {
  const detailRecord = detail as AgentSessionDetail & {
    artifacts?: unknown;
    threadRead?: unknown;
  };
  return [
    ...readHistoryArtifactSummaries(detailRecord),
    ...readHistoryArtifactSummaries(detail.thread_read),
    ...readHistoryArtifactSummaries(detailRecord.threadRead),
  ];
}

function historyArtifactFromSummary(params: {
  sessionIdFallback: string;
  summary: HistoryArtifactSummary;
}): Artifact | null {
  const { sessionIdFallback, summary } = params;
  const metadata = asHistoryRecord(summary.metadata);
  if (isHiddenWorkspacePatchArtifactSummary(summary, metadata)) {
    return null;
  }
  const path =
    readHistoryString(summary.path) ||
    readHistoryMetadataString(metadata, [
      "filePath",
      "file_path",
      "path",
      "artifactPath",
      "artifact_path",
      "absolutePath",
      "absolute_path",
    ]);
  const id =
    readHistoryString(summary.artifactId) ||
    readHistoryString(summary.artifactRef) ||
    path ||
    readHistoryString(summary.eventId);
  if (!id || !path) {
    return null;
  }

  const title =
    readHistoryString(summary.title) ||
    readHistoryMetadataString(metadata, ["title", "filename", "fileName"]) ||
    fileNameFromHistoryPath(path);
  const previewText = readHistoryMetadataString(metadata, [
    "previewText",
    "preview_text",
    "contentPreview",
    "content_preview",
  ]);
  const now = Date.now();
  const timestamp =
    readHistoryNumber(metadata?.createdAt) ??
    readHistoryNumber(metadata?.updatedAt) ??
    now;
  const sessionId =
    readHistoryString(summary.sessionId) ||
    readHistoryMetadataString(metadata, [
      "sessionId",
      "session_id",
      "appServerSessionId",
      "app_server_session_id",
      "appServerArtifactSessionId",
      "app_server_artifact_session_id",
    ]) ||
    sessionIdFallback;
  const turnId = readHistoryString(summary.turnId) || undefined;
  const artifactRef = readHistoryString(summary.artifactRef) || undefined;

  return {
    id,
    type: historyArtifactTypeFromSummary(summary, path, metadata),
    title,
    content: previewText,
    status: historyArtifactStatusFromSummary(summary),
    meta: {
      ...(metadata ?? {}),
      artifactRef,
      eventId: readHistoryString(summary.eventId) || undefined,
      sequence: readHistoryNumber(summary.sequence),
      sessionId,
      turnId,
      appServerArtifactSessionId: sessionId,
      appServerArtifactTurnId: turnId,
      appServerArtifactRef: artifactRef,
      contentStatus: readHistoryString(summary.contentStatus) || undefined,
      filePath: path,
      artifactPath: path,
      path,
      previewText: previewText || undefined,
    },
    position: { start: 0, end: previewText.length },
    createdAt: timestamp,
    updatedAt: timestamp,
  };
}

function isHiddenWorkspacePatchArtifactSummary(
  summary: HistoryArtifactSummary,
  metadata: Record<string, unknown> | null,
): boolean {
  const summaryKind = readHistoryString(summary.kind);
  if (isWorkspaceArticlePatchArtifactKind(summaryKind)) {
    return true;
  }
  const path =
    readHistoryString(summary.path) ||
    readHistoryMetadataString(metadata, [
      "filePath",
      "file_path",
      "path",
      "artifactPath",
      "artifact_path",
    ]);
  if (isWorkspaceArticlePatchArtifactPath(path)) {
    return true;
  }
  if (!metadata) {
    return false;
  }
  const workspacePatch = readWorkspaceArticlePatchRecordFromMetadata(metadata);
  const artifactKind = readHistoryMetadataString(metadata, [
    "artifactKind",
    "artifact_kind",
    "kind",
  ]);
  return (
    Boolean(workspacePatch) || isWorkspaceArticlePatchArtifactKind(artifactKind)
  );
}

function isCodeHistoryArtifact(artifact: Artifact): boolean {
  const metadata = asHistoryRecord(artifact.meta);
  const language = readHistoryMetadataString(metadata, [
    "language",
    "fileLanguage",
    "file_language",
  ]);
  if (language) {
    return true;
  }
  if (artifact.type === "code" || artifact.type === "react") {
    return true;
  }
  const path = readHistoryMetadataString(metadata, [
    "filePath",
    "file_path",
    "artifactPath",
    "artifact_path",
    "path",
  ]);
  const extension = fileNameFromHistoryPath(path)
    .split(".")
    .pop()
    ?.toLowerCase();
  return Boolean(
    extension &&
    [
      "c",
      "cc",
      "cpp",
      "cs",
      "css",
      "go",
      "h",
      "hpp",
      "java",
      "js",
      "jsx",
      "kt",
      "mjs",
      "py",
      "rs",
      "sql",
      "swift",
      "ts",
      "tsx",
    ].includes(extension),
  );
}

function historyMessageTextFromArtifacts(artifacts: Artifact[]): string {
  const firstCompleteArtifact =
    artifacts.find((artifact) => artifact.status === "complete") ??
    artifacts[0];
  if (!firstCompleteArtifact) {
    return "";
  }

  const metadata = asHistoryRecord(firstCompleteArtifact.meta);
  const explicitText = readHistoryMetadataString(metadata, [
    "completionText",
    "completion_text",
    "messageText",
    "message_text",
    "summaryText",
    "summary_text",
    "statusMessage",
    "status_message",
  ]);
  if (explicitText) {
    return explicitText;
  }

  return isCodeHistoryArtifact(firstCompleteArtifact)
    ? "已生成代码产物，可在工作台查看。"
    : "已生成产物，可在工作台查看。";
}

export function hydrateSessionDetailMessagesFromArtifacts(
  detail: AgentSessionDetail,
  topicId: string,
): Message[] {
  const artifacts = mergeArtifacts(
    collectHistoryArtifactSummaries(detail)
      .map((summary) =>
        historyArtifactFromSummary({
          sessionIdFallback: detail.id,
          summary,
        }),
      )
      .filter((artifact): artifact is Artifact => artifact !== null),
  );
  if (artifacts.length === 0) {
    return [];
  }

  const content = historyMessageTextFromArtifacts(artifacts);
  const timestamp = parseHistoryTimestamp(
    detail.turns?.[0]?.completed_at ||
      detail.turns?.[0]?.updated_at ||
      detail.turns?.[0]?.started_at ||
      null,
  );
  return [
    {
      id: `${topicId}-app-server-artifacts`,
      role: "assistant",
      content,
      contentParts: content ? [{ type: "text", text: content }] : undefined,
      artifacts,
      timestamp,
      isThinking: false,
      runtimeTurnId: detail.turns?.[0]?.id,
    },
  ];
}
