import { normalizeArtifactProtocolPath } from "@/lib/artifact-protocol";

export const WORKSPACE_PATCH_KIND = "workspace_patch";

export function asWorkspaceArticleMetadataRecord(
  value: unknown,
): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function readArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function pushRecord(
  candidates: Record<string, unknown>[],
  value: unknown,
): void {
  const record = asWorkspaceArticleMetadataRecord(value);
  if (record) {
    candidates.push(record);
  }
}

export function isWorkspaceArticlePatchRecord(
  value: unknown,
): value is Record<string, unknown> {
  const record = asWorkspaceArticleMetadataRecord(value);
  return Boolean(record && readArray(record.objects).length > 0);
}

function pushPatchRecord(
  candidates: Record<string, unknown>[],
  value: unknown,
): void {
  if (isWorkspaceArticlePatchRecord(value)) {
    candidates.push(value);
  }
}

export function readWorkspaceArticleRecordFromMetadata(
  record: Record<string, unknown> | null,
): Record<string, unknown> | null {
  if (!record) {
    return null;
  }
  return (
    asWorkspaceArticleMetadataRecord(record.articleWorkspace) ??
    asWorkspaceArticleMetadataRecord(record.article_workspace)
  );
}

export function readWorkspaceArticlePatchRecordFromMetadata(
  record: Record<string, unknown> | null,
): Record<string, unknown> | null {
  if (!record) {
    return null;
  }
  return (
    asWorkspaceArticleMetadataRecord(record.workspacePatch) ??
    asWorkspaceArticleMetadataRecord(record.workspace_patch)
  );
}

export function readWorkspaceArticlePatchLikeRecordFromMetadata(
  record: Record<string, unknown> | null,
): Record<string, unknown> | null {
  return (
    readWorkspaceArticleRecordFromMetadata(record) ??
    readWorkspaceArticlePatchRecordFromMetadata(record)
  );
}

export function hasWorkspaceArticlePatchMetadata(
  record: Record<string, unknown> | null,
): boolean {
  return Boolean(readWorkspaceArticlePatchLikeRecordFromMetadata(record));
}

export function isWorkspaceArticlePatchArtifactKind(value: unknown): boolean {
  const normalizedKind = readString(value)?.toLowerCase();
  return Boolean(normalizedKind && normalizedKind === WORKSPACE_PATCH_KIND);
}

export function isWorkspaceArticlePatchArtifactPath(value: unknown): boolean {
  const path = readString(value);
  if (!path) {
    return false;
  }
  const normalizedPath = normalizeArtifactProtocolPath(path);
  if (!normalizedPath.endsWith(".json")) {
    return false;
  }
  const inArtifactStore =
    normalizedPath.startsWith(".lime/artifacts/") ||
    normalizedPath.includes("/.lime/artifacts/");
  if (!inArtifactStore) {
    return false;
  }
  return normalizedPath.endsWith("/workspace-patch.json");
}

export function collectWorkspaceArticlePatchLikeMetadataRecords(
  record: Record<string, unknown> | null,
): Record<string, unknown>[] {
  if (!record) {
    return [];
  }
  const candidates: Record<string, unknown>[] = [];
  pushRecord(candidates, record.articleWorkspace);
  pushRecord(candidates, record.article_workspace);
  pushRecord(candidates, record.workspacePatch);
  pushRecord(candidates, record.workspace_patch);
  return candidates;
}

function pushKnownPatchMetadataRecords(
  candidates: Record<string, unknown>[],
  record: Record<string, unknown> | null,
): void {
  for (const candidate of collectWorkspaceArticlePatchLikeMetadataRecords(
    record,
  )) {
    pushPatchRecord(candidates, candidate);
  }
}

function pushContentPatchRecord(
  candidates: Record<string, unknown>[],
  record: Record<string, unknown> | null,
): void {
  const content = readString(record?.content);
  if (!content) {
    return;
  }
  try {
    pushPatchRecord(candidates, JSON.parse(content));
  } catch {
    return;
  }
}

export function collectWorkspaceArticlePatchRecordsFromArtifactLike(
  value: unknown,
): Record<string, unknown>[] {
  const record = asWorkspaceArticleMetadataRecord(value);
  if (!record) {
    return [];
  }

  const artifact = asWorkspaceArticleMetadataRecord(record.artifact);
  const meta = asWorkspaceArticleMetadataRecord(record.meta);
  const metadata = asWorkspaceArticleMetadataRecord(record.metadata);
  const artifactMetadata = asWorkspaceArticleMetadataRecord(artifact?.metadata);
  const candidates: Record<string, unknown>[] = [];

  pushPatchRecord(candidates, record);
  pushKnownPatchMetadataRecords(candidates, record);
  pushKnownPatchMetadataRecords(candidates, meta);
  pushKnownPatchMetadataRecords(candidates, metadata);
  pushKnownPatchMetadataRecords(candidates, artifact);
  pushKnownPatchMetadataRecords(candidates, artifactMetadata);
  pushContentPatchRecord(candidates, record);
  pushContentPatchRecord(candidates, artifact);

  return candidates;
}
