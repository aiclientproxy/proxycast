import {
  AppServerClient,
  type AppServerArtifactReadResponse,
  type AppServerArtifactReadParams,
  type AppServerArtifactSummary,
  type AppServerArtifactWriteParams,
  type AppServerArtifactWriteResponse,
} from "@/lib/api/appServer";
import type { ArtifactDocumentV1 } from "@/lib/artifact-document";
import type { Artifact } from "@/lib/artifact/types";
import { resolveArtifactProtocolFilePath } from "@/lib/artifact-protocol";
import type { AgentThreadItem } from "../agentProtocol";

export type AgentRuntimeTimelineArtifactItem = Extract<
  AgentThreadItem,
  { type: "file_artifact" }
>;

export type AgentRuntimeTimelineArtifactContent = {
  artifactId?: string;
  artifactRef: string;
  content: string;
  filePath: string;
  metadata?: unknown;
  title?: string;
};

type AppServerArtifactReadRpcClient = Pick<AppServerClient, "readArtifacts">;

export type AppServerArtifactRpcClient = AppServerArtifactReadRpcClient &
  Pick<AppServerClient, "writeArtifact">;

export interface AppServerArtifactClientDeps {
  appServerClient?: AppServerArtifactRpcClient;
}

export type AgentRuntimeArtifactDocumentSnapshotSaveResult =
  | {
      status: "written";
      evidence: AgentRuntimeArtifactDocumentSnapshotSaveEvidence;
    }
  | {
      status: "skipped";
      reason: "missing_scope";
    };

export interface AgentRuntimeArtifactDocumentSnapshotSaveEvidence {
  artifactDocumentId: string;
  artifactRef: string;
  contentBytes?: number;
  contentSha256?: string;
  contentStatus?: string;
  eventId?: string;
  filePath?: string;
  lastPersistedAt?: string;
  sessionId: string;
  sidecarRelativePath?: string;
  sourceArtifactRef?: string;
  threadId: string;
  turnId?: string;
  versionId?: string;
  versionNo?: number;
}

export interface AgentRuntimeArtifactDocumentScope {
  artifactDocumentId?: string;
  artifactRef: string;
  lastPersistedAt?: string;
  sessionId: string;
  sidecarRelativePath?: string;
  sourceArtifactRef?: string;
  threadId?: string;
  turnId?: string;
  versionId?: string;
  versionNo?: number;
}

export function createAppServerArtifactClient({
  appServerClient = new AppServerClient(),
}: AppServerArtifactClientDeps = {}) {
  async function readAgentRuntimeTimelineArtifactContent(
    item: AgentRuntimeTimelineArtifactItem,
  ): Promise<AgentRuntimeTimelineArtifactContent | null> {
    const params = appServerArtifactReadParamsFromTimelineItem(item);
    if (!params) {
      return null;
    }

    const response = await appServerClient.readArtifacts(params);
    assertArtifactReadResponse(response.result);
    return projectTimelineArtifactContentFromAppServerSummaries({
      item,
      params,
      artifacts: response.result.artifacts,
    });
  }

  async function readAgentRuntimeArtifactPreviewContent(
    artifact: Artifact,
    artifactPath: string,
  ): Promise<AgentRuntimeTimelineArtifactContent | null> {
    const params = appServerArtifactReadParamsFromArtifactPreview(
      artifact,
      artifactPath,
    );
    if (!params) {
      return null;
    }

    const response = await appServerClient.readArtifacts(params);
    assertArtifactReadResponse(response.result);
    return projectArtifactPreviewContentFromAppServerSummaries({
      artifact,
      artifactPath,
      params,
      artifacts: response.result.artifacts,
    });
  }

  async function saveAgentRuntimeArtifactDocumentSnapshot(
    artifact: Artifact,
    document: ArtifactDocumentV1,
  ): Promise<AgentRuntimeArtifactDocumentSnapshotSaveResult> {
    const params = appServerArtifactWriteParamsFromArtifactDocument(
      artifact,
      document,
    );
    if (!params) {
      return {
        status: "skipped",
        reason: "missing_scope",
      };
    }
    const scope = resolveAgentRuntimeArtifactDocumentScope(artifact, {
      artifactPath: resolveArtifactProtocolFilePath(artifact),
      document,
    });
    if (!scope?.threadId || !scope.sessionId) {
      return {
        status: "skipped",
        reason: "missing_scope",
      };
    }

    const response = await appServerClient.writeArtifact(params);
    assertArtifactWriteResponse(response.result);
    const evidence = projectArtifactDocumentSnapshotSaveEvidence({
      document,
      params,
      response: response.result,
      sessionId: scope.sessionId,
    });
    return {
      status: "written",
      evidence,
    };
  }

  async function writeAgentRuntimeArtifactSnapshot(
    params: AppServerArtifactWriteParams,
  ): Promise<AppServerArtifactWriteResponse> {
    const response = await appServerClient.writeArtifact(params);
    assertArtifactWriteResponse(response.result);
    return response.result;
  }

  return {
    readAgentRuntimeArtifactPreviewContent,
    readAgentRuntimeTimelineArtifactContent,
    saveAgentRuntimeArtifactDocumentSnapshot,
    writeAgentRuntimeArtifactSnapshot,
  };
}

export function projectArtifactDocumentSnapshotSaveEvidence({
  document,
  params,
  response,
  sessionId,
}: {
  document: ArtifactDocumentV1;
  params: AppServerArtifactWriteParams;
  response: AppServerArtifactWriteResponse;
  sessionId: string;
}): AgentRuntimeArtifactDocumentSnapshotSaveEvidence {
  const metadata = asRecord(params.artifact.metadata);
  const versionId =
    readText(metadata, ["artifactVersionId", "artifact_version_id"]) ||
    normalizeText(document.metadata.currentVersionId);
  const versionNo =
    readFiniteNumber(metadata?.artifactVersionNo) ??
    readFiniteNumber(metadata?.artifact_version_no) ??
    document.metadata.currentVersionNo;
  const sourceArtifactRef = readText(metadata, [
    "sourceArtifactRef",
    "source_artifact_ref",
  ]);

  return omitUndefined({
    artifactDocumentId: response.artifactDocumentId || document.artifactId,
    artifactRef: response.artifactRef,
    contentBytes: response.sidecar.bytes,
    contentSha256: response.sidecar.sha256,
    contentStatus: response.sidecar.contentStatus,
    eventId: response.eventId,
    filePath: normalizeText(params.artifact.path),
    lastPersistedAt: response.persistedAt,
    sessionId,
    sidecarRelativePath: response.sidecar.relativePath,
    sourceArtifactRef,
    threadId: response.threadId,
    turnId: normalizeText(response.turnId) || normalizeText(params.turnId),
    versionId,
    versionNo,
  });
}

function assertArtifactReadResponse(
  value: unknown,
): asserts value is AppServerArtifactReadResponse {
  if (!isArtifactReadResponse(value)) {
    throw new Error("artifact/read did not return artifact summaries");
  }
}

function isArtifactReadResponse(
  value: unknown,
): value is AppServerArtifactReadResponse {
  return (
    Boolean(value) &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Array.isArray((value as { artifacts?: unknown }).artifacts) &&
    (value as { artifacts: unknown[] }).artifacts.every(isArtifactSummary) &&
    (typeof (value as { nextCursor?: unknown }).nextCursor === "undefined" ||
      typeof (value as { nextCursor?: unknown }).nextCursor === "string")
  );
}

function assertArtifactWriteResponse(
  value: unknown,
): asserts value is AppServerArtifactWriteResponse {
  if (!isArtifactWriteResponse(value)) {
    throw new Error("artifact/write did not return write evidence");
  }
}

function isArtifactWriteResponse(
  value: unknown,
): value is AppServerArtifactWriteResponse {
  return (
    Boolean(value) &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof (value as { threadId?: unknown }).threadId === "string" &&
    typeof (value as { artifactRef?: unknown }).artifactRef === "string" &&
    typeof (value as { eventId?: unknown }).eventId === "string" &&
    typeof (value as { sequence?: unknown }).sequence === "number" &&
    typeof (value as { persistedAt?: unknown }).persistedAt === "string" &&
    isArtifactWriteSidecar((value as { sidecar?: unknown }).sidecar)
  );
}

function isArtifactWriteSidecar(value: unknown): boolean {
  return (
    Boolean(value) &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof (value as { relativePath?: unknown }).relativePath === "string" &&
    typeof (value as { bytes?: unknown }).bytes === "number" &&
    typeof (value as { sha256?: unknown }).sha256 === "string" &&
    typeof (value as { contentStatus?: unknown }).contentStatus === "string"
  );
}

function isArtifactSummary(value: unknown): value is AppServerArtifactSummary {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }

  const artifact = value as Record<string, unknown>;
  return (
    typeof artifact.artifactRef === "string" &&
    artifact.artifactRef.length > 0 &&
    typeof artifact.eventId === "string" &&
    artifact.eventId.length > 0 &&
    typeof artifact.sequence === "number" &&
    Number.isFinite(artifact.sequence) &&
    isArtifactContentStatus(artifact.contentStatus) &&
    optionalString(artifact.turnId) &&
    optionalString(artifact.artifactId) &&
    optionalString(artifact.path) &&
    optionalString(artifact.title) &&
    optionalString(artifact.kind) &&
    optionalString(artifact.status) &&
    optionalString(artifact.content)
  );
}

function isArtifactContentStatus(value: unknown): boolean {
  return (
    value === "notRequested" || value === "available" || value === "unavailable"
  );
}

function optionalString(value: unknown): boolean {
  return typeof value === "undefined" || typeof value === "string";
}

export function appServerArtifactReadParamsFromTimelineItem(
  item: AgentRuntimeTimelineArtifactItem,
): AppServerArtifactReadParams | null {
  const metadata = asRecord(item.metadata);
  const sessionId = readText(metadata, [
    "sessionId",
    "session_id",
    "appServerSessionId",
    "app_server_session_id",
    "appServerArtifactSessionId",
    "app_server_artifact_session_id",
  ]);
  if (!sessionId) {
    return null;
  }

  const artifactRef = readText(metadata, [
    "artifactRef",
    "artifact_ref",
    "appServerArtifactRef",
    "app_server_artifact_ref",
    "artifactId",
    "artifact_id",
    "artifactDocumentId",
    "artifact_document_id",
  ]);
  const turnId =
    readText(metadata, [
      "turnId",
      "turn_id",
      "appServerTurnId",
      "app_server_turn_id",
      "appServerArtifactTurnId",
      "app_server_artifact_turn_id",
    ]) || normalizeText(item.turn_id);

  return omitUndefined({
    sessionId,
    turnId,
    artifactRef,
    includeContent: true,
    limit: artifactRef ? 1 : 20,
  });
}

export function appServerArtifactReadParamsFromArtifactPreview(
  artifact: Artifact,
  artifactPath: string,
): AppServerArtifactReadParams | null {
  const scope = resolveAgentRuntimeArtifactDocumentScope(artifact, {
    artifactPath,
  });
  if (!scope) {
    return null;
  }

  return omitUndefined({
    sessionId: scope.sessionId,
    turnId: scope.turnId,
    artifactRef: scope.artifactRef,
    includeContent: true,
    limit: 1,
  });
}

export function appServerArtifactWriteParamsFromArtifactDocument(
  artifact: Artifact,
  document: ArtifactDocumentV1,
): AppServerArtifactWriteParams | null {
  const metadata = asRecord(artifact.meta);
  const filePath = resolveArtifactProtocolFilePath(artifact);
  const scope = resolveAgentRuntimeArtifactDocumentScope(artifact, {
    artifactPath: filePath,
    document,
  });
  if (!scope) {
    return null;
  }

  const content = JSON.stringify(document, null, 2);
  const artifactMetadata = {
    ...(metadata || {}),
    artifactSchema: document.schemaVersion,
    artifactKind: document.kind,
    artifactDocument: document,
    artifactDocumentPersistence: scope,
    artifactTitle: document.title,
    artifactDocumentId: document.artifactId,
    artifactVersionId: normalizeText(document.metadata.currentVersionId),
    artifactVersionNo: document.metadata.currentVersionNo,
    artifactRef: scope.artifactRef,
    filePath,
    articleWorkspace:
      asRecord(document.metadata.articleWorkspace) ||
      asRecord(document.metadata.article_workspace) ||
      asRecord(metadata?.articleWorkspace) ||
      asRecord(metadata?.article_workspace),
  };

  if (!scope.threadId) {
    return null;
  }

  return omitUndefined({
    threadId: scope.threadId,
    turnId: scope.turnId,
    artifact: omitUndefined({
      artifactRef: scope.artifactRef,
      artifactDocumentId: document.artifactId,
      path: filePath,
      title: document.title || artifact.title,
      kind: "artifact_document",
      status: document.status,
      content,
      metadata: omitUndefined(artifactMetadata),
    }),
  });
}

export function resolveAgentRuntimeArtifactDocumentScope(
  artifact: Artifact,
  options: {
    artifactPath?: string;
    document?: ArtifactDocumentV1;
  } = {},
): AgentRuntimeArtifactDocumentScope | null {
  const metadata = asRecord(artifact.meta);
  const savedScope =
    asRecord(metadata?.artifactDocumentPersistence) ||
    asRecord(metadata?.artifactDocumentScope) ||
    asRecord(metadata?.artifactDocumentSaveEvidence);
  const document = options.document;
  const artifactPath =
    normalizeText(options.artifactPath) ||
    normalizeText(resolveArtifactProtocolFilePath(artifact));

  const sessionId =
    readText(savedScope, ["sessionId", "session_id"]) ||
    resolveArtifactSessionId(metadata);
  if (!sessionId) {
    return null;
  }

  const artifactRef =
    readText(savedScope, ["artifactRef", "artifact_ref"]) ||
    resolveArtifactRef(metadata) ||
    normalizeText(document?.artifactId) ||
    normalizeText(artifact.id) ||
    artifactPath;
  if (!artifactRef) {
    return null;
  }

  const artifactDocumentId =
    readText(savedScope, ["artifactDocumentId", "artifact_document_id"]) ||
    readText(metadata, [
      "artifactDocumentId",
      "artifact_document_id",
      "appServerArtifactDocumentId",
      "app_server_artifact_document_id",
    ]) ||
    normalizeText(document?.artifactId);

  return omitUndefined({
    artifactDocumentId,
    artifactRef,
    lastPersistedAt:
      readText(savedScope, ["lastPersistedAt", "last_persisted_at"]) ||
      readText(metadata, ["appServerLastPersistedAt"]),
    sessionId,
    sidecarRelativePath:
      readText(savedScope, ["sidecarRelativePath", "sidecar_relative_path"]) ||
      readText(metadata, [
        "appServerSidecarRelativePath",
        "app_server_sidecar_relative_path",
      ]),
    sourceArtifactRef:
      readText(savedScope, ["sourceArtifactRef", "source_artifact_ref"]) ||
      readText(metadata, ["sourceArtifactRef", "source_artifact_ref"]),
    threadId:
      readText(savedScope, ["threadId", "thread_id"]) ||
      readText(metadata, ["threadId", "thread_id"]) ||
      normalizeText(document?.threadId),
    turnId:
      readText(savedScope, ["turnId", "turn_id"]) ||
      resolveArtifactTurnId(metadata) ||
      normalizeText(document?.turnId),
    versionId:
      readText(savedScope, ["versionId", "version_id"]) ||
      readText(metadata, [
        "artifactVersionId",
        "artifact_version_id",
        "appServerArtifactVersionId",
      ]) ||
      normalizeText(document?.metadata.currentVersionId),
    versionNo:
      readFiniteNumber(savedScope?.versionNo) ??
      readFiniteNumber(savedScope?.version_no) ??
      readFiniteNumber(metadata?.artifactVersionNo) ??
      readFiniteNumber(metadata?.artifact_version_no) ??
      document?.metadata.currentVersionNo,
  });
}

export function agentRuntimeArtifactDocumentScopeFromSaveEvidence(
  evidence: AgentRuntimeArtifactDocumentSnapshotSaveEvidence,
): AgentRuntimeArtifactDocumentScope {
  return omitUndefined({
    artifactDocumentId: evidence.artifactDocumentId,
    artifactRef: evidence.artifactRef,
    lastPersistedAt: evidence.lastPersistedAt,
    sessionId: evidence.sessionId,
    sidecarRelativePath: evidence.sidecarRelativePath,
    sourceArtifactRef: evidence.sourceArtifactRef,
    threadId: evidence.threadId,
    turnId: evidence.turnId,
    versionId: evidence.versionId,
    versionNo: evidence.versionNo,
  });
}

export function hasAgentRuntimeArtifactPreviewScope(
  artifact: Artifact,
  artifactPath: string,
): boolean {
  return (
    appServerArtifactReadParamsFromArtifactPreview(artifact, artifactPath) !==
    null
  );
}

export function projectTimelineArtifactContentFromAppServerSummaries({
  item,
  params,
  artifacts,
}: {
  item: AgentRuntimeTimelineArtifactItem;
  params: AppServerArtifactReadParams;
  artifacts: AppServerArtifactSummary[];
}): AgentRuntimeTimelineArtifactContent | null {
  const metadata = asRecord(item.metadata);
  const expectedArtifactIds = new Set(
    [
      params.artifactRef,
      normalizeText(item.id),
      readText(metadata, ["artifactId", "artifact_id"]),
      readText(metadata, ["artifactDocumentId", "artifact_document_id"]),
    ].filter((value): value is string => Boolean(value)),
  );
  const expectedPath = normalizePath(item.path);
  const selected =
    artifacts.find((artifact) => artifact.artifactRef === params.artifactRef) ??
    artifacts.find(
      (artifact) =>
        (artifact.artifactId && expectedArtifactIds.has(artifact.artifactId)) ||
        expectedArtifactIds.has(artifact.artifactRef),
    ) ??
    artifacts.find((artifact) => normalizePath(artifact.path) === expectedPath);

  if (
    !selected ||
    selected.contentStatus !== "available" ||
    typeof selected.content !== "string"
  ) {
    return null;
  }

  return omitUndefined({
    artifactId: selected.artifactId,
    artifactRef: selected.artifactRef,
    content: selected.content,
    filePath: selected.path || item.path,
    metadata: selected.metadata,
    title: selected.title,
  });
}

export function projectArtifactPreviewContentFromAppServerSummaries({
  artifact,
  artifactPath,
  params,
  artifacts,
}: {
  artifact: Artifact;
  artifactPath: string;
  params: AppServerArtifactReadParams;
  artifacts: AppServerArtifactSummary[];
}): AgentRuntimeTimelineArtifactContent | null {
  const metadata = asRecord(artifact.meta);
  const expectedArtifactIds = new Set(
    [
      params.artifactRef,
      normalizeText(artifact.id),
      readText(metadata, ["artifactId", "artifact_id"]),
      readText(metadata, ["artifactDocumentId", "artifact_document_id"]),
    ].filter((value): value is string => Boolean(value)),
  );
  const expectedPath = normalizePath(artifactPath);
  const selected =
    artifacts.find((entry) => entry.artifactRef === params.artifactRef) ??
    artifacts.find(
      (entry) =>
        (entry.artifactId && expectedArtifactIds.has(entry.artifactId)) ||
        expectedArtifactIds.has(entry.artifactRef),
    ) ??
    artifacts.find((entry) => normalizePath(entry.path) === expectedPath);

  if (
    !selected ||
    selected.contentStatus !== "available" ||
    typeof selected.content !== "string"
  ) {
    return null;
  }

  return omitUndefined({
    artifactId: selected.artifactId,
    artifactRef: selected.artifactRef,
    content: selected.content,
    filePath: selected.path || artifactPath,
    metadata: selected.metadata,
    title: selected.title,
  });
}

const defaultAppServerArtifactClient = createAppServerArtifactClient();

export const readAgentRuntimeTimelineArtifactContent =
  defaultAppServerArtifactClient.readAgentRuntimeTimelineArtifactContent;

export const readAgentRuntimeArtifactPreviewContent =
  defaultAppServerArtifactClient.readAgentRuntimeArtifactPreviewContent;

export const saveAgentRuntimeArtifactDocumentSnapshot =
  defaultAppServerArtifactClient.saveAgentRuntimeArtifactDocumentSnapshot;

export const writeAgentRuntimeArtifactSnapshot =
  defaultAppServerArtifactClient.writeAgentRuntimeArtifactSnapshot;

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function normalizeText(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function readFiniteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function readText(
  record: Record<string, unknown> | undefined,
  keys: string[],
): string | undefined {
  for (const key of keys) {
    const direct = normalizeText(record?.[key]);
    if (direct) {
      return direct;
    }
  }
  return undefined;
}

function readNestedText(
  record: Record<string, unknown> | undefined,
  path: string[],
): string | undefined {
  let current: unknown = record;
  for (const key of path) {
    const next = asRecord(current)?.[key];
    if (typeof next === "undefined") {
      return undefined;
    }
    current = next;
  }
  return normalizeText(current);
}

function readStringArrayFirst(
  record: Record<string, unknown> | undefined,
  path: string[],
): string | undefined {
  let current: unknown = record;
  for (const key of path) {
    const next = asRecord(current)?.[key];
    if (typeof next === "undefined") {
      return undefined;
    }
    current = next;
  }
  if (!Array.isArray(current)) {
    return undefined;
  }
  for (const item of current) {
    const normalized = normalizeText(item);
    if (normalized) {
      return normalized;
    }
  }
  return undefined;
}

function resolveArtifactSessionId(
  metadata: Record<string, unknown> | undefined,
): string | undefined {
  return (
    readText(metadata, [
      "sessionId",
      "session_id",
      "appServerSessionId",
      "app_server_session_id",
      "appServerArtifactSessionId",
      "app_server_artifact_session_id",
    ]) ||
    readNestedText(metadata, ["articleWorkspace", "sessionId"]) ||
    readNestedText(metadata, ["articleWorkspace", "session_id"]) ||
    readNestedText(metadata, ["article_workspace", "sessionId"]) ||
    readNestedText(metadata, ["article_workspace", "session_id"]) ||
    readNestedText(metadata, ["sourceRunBinding", "sessionId"]) ||
    readNestedText(metadata, ["sourceRunBinding", "session_id"])
  );
}

function resolveArtifactTurnId(
  metadata: Record<string, unknown> | undefined,
): string | undefined {
  return (
    readText(metadata, [
      "turnId",
      "turn_id",
      "appServerTurnId",
      "app_server_turn_id",
      "appServerArtifactTurnId",
      "app_server_artifact_turn_id",
    ]) ||
    readNestedText(metadata, ["sourceRunBinding", "turnId"]) ||
    readNestedText(metadata, ["sourceRunBinding", "turn_id"])
  );
}

function resolveArtifactRef(
  metadata: Record<string, unknown> | undefined,
): string | undefined {
  return (
    readText(metadata, [
      "artifactRef",
      "artifact_ref",
      "appServerArtifactRef",
      "app_server_artifact_ref",
    ]) ||
    readStringArrayFirst(metadata, ["articleWorkspace", "artifactIds"]) ||
    readStringArrayFirst(metadata, ["articleWorkspace", "artifact_ids"]) ||
    readStringArrayFirst(metadata, ["article_workspace", "artifactIds"]) ||
    readStringArrayFirst(metadata, ["article_workspace", "artifact_ids"]) ||
    readText(metadata, [
      "sourceRef",
      "artifactId",
      "artifact_id",
      "artifactDocumentId",
      "artifact_document_id",
    ])
  );
}

function normalizePath(value: unknown): string | undefined {
  return typeof value === "string"
    ? value.replace(/\\/g, "/").trim()
    : undefined;
}

function omitUndefined<T extends Record<string, unknown>>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter(([, entry]) => entry !== undefined),
  ) as T;
}
