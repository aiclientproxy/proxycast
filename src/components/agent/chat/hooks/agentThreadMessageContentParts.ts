import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import type { ContentPart, MessageMediaReference } from "../types";
import { buildAgentTextDeltaContentPartMetadata } from "../utils/contentPartTimeline";

type AgentMessageItem = Extract<AgentThreadItem, { type: "agent_message" }>;
type MediaItem = Extract<AgentThreadItem, { type: "media" }>;
type ImageGenerationItem = Extract<
  AgentThreadItem,
  { type: "image_generation" }
>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function normalizeString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  return normalized ? normalized : null;
}

function isInlineMediaUri(uri: string): boolean {
  return uri.trimStart().toLowerCase().startsWith("data:");
}

function normalizeNonInlineSourceReference(value: unknown): string | null {
  const normalized = normalizeString(value);
  if (!normalized || isInlineMediaUri(normalized)) {
    return null;
  }
  return normalized;
}

function buildBaseMetadata(
  item: AgentMessageItem | MediaItem | ImageGenerationItem,
  index: number,
): Record<string, unknown> {
  return {
    itemId: item.id,
    threadItemId: item.id,
    turnId: item.turn_id,
    sequence: item.sequence,
    contentPartIndex: index,
    ...(item.type === "agent_message" && item.phase
      ? { phase: item.phase }
      : {}),
  };
}

function contentPartTextFallback(params: {
  caption: string | null;
  title: string | null;
  uri: string;
}): string {
  return params.caption ?? params.title ?? params.uri;
}

function mediaKindFromMimeType(mimeType: string | null): string {
  const [kind] = mimeType?.toLowerCase().split("/", 1) ?? [];
  return kind || "file";
}

function mediaTitleFromUri(uri: string): string {
  const withoutQuery = uri.split(/[?#]/u, 1)[0] ?? uri;
  const segments = withoutQuery.split(/[\\/]/u);
  const encodedTitle = segments.at(-1)?.trim();
  if (!encodedTitle) {
    return uri;
  }
  try {
    return decodeURIComponent(encodedTitle);
  } catch {
    return encodedTitle;
  }
}

function isAbsoluteLocalPath(uri: string): boolean {
  return (
    uri.startsWith("/") || /^[a-z]:[\\/]/iu.test(uri) || uri.startsWith("\\\\")
  );
}

function buildMediaReference(params: {
  caption: string | null;
  title: string | null;
  uri: string;
  refId: string | null;
  mimeType: string | null;
  mediaKind: string | null;
  sidecarRef?: unknown;
  sourceUri: string | null;
  sourcePath: string | null;
  previewUrl: string | null;
  sha256: string | null;
  byteSize?: number;
}): MessageMediaReference {
  return {
    uri: params.uri,
    ...(params.mediaKind ? { kind: params.mediaKind } : {}),
    ...(params.refId ? { refId: params.refId } : {}),
    ...(params.mimeType ? { mimeType: params.mimeType } : {}),
    ...(params.title ? { title: params.title } : {}),
    ...(params.caption ? { caption: params.caption } : {}),
    ...(params.sidecarRef !== undefined
      ? { sidecarRef: params.sidecarRef }
      : {}),
    ...(params.sourceUri ? { sourceUri: params.sourceUri } : {}),
    ...(params.sourcePath ? { sourcePath: params.sourcePath } : {}),
    ...(params.previewUrl ? { previewUrl: params.previewUrl } : {}),
    ...(params.sha256 ? { sha256: params.sha256 } : {}),
    ...(typeof params.byteSize === "number"
      ? { byteSize: params.byteSize }
      : {}),
  };
}

function buildTextContentPart(
  item: AgentMessageItem,
  part: Record<string, unknown>,
  index: number,
): ContentPart | null {
  const text = normalizeString(part.text);
  if (!text) {
    return null;
  }
  const metadata = {
    ...buildAgentTextDeltaContentPartMetadata({
      itemId: item.id,
      phase: item.phase,
      sequence: item.sequence,
      turnId: item.turn_id,
    }),
    contentPartIndex: index,
    threadItemId: item.id,
  };
  return { type: "text", text, metadata };
}

function buildMediaReferenceContentPart(
  item: AgentMessageItem,
  part: Record<string, unknown>,
  index: number,
): ContentPart | null {
  const sourceReference = isRecord(part.reference) ? part.reference : null;
  const uri = normalizeString(sourceReference?.uri);
  if (!uri || isInlineMediaUri(uri)) {
    return null;
  }

  const caption = normalizeString(part.caption);
  const title = normalizeString(sourceReference?.title);
  const mimeType =
    normalizeString(sourceReference?.mime_type) ??
    normalizeString(sourceReference?.mimeType);
  const sourceUri =
    normalizeNonInlineSourceReference(sourceReference?.source_uri) ??
    normalizeNonInlineSourceReference(sourceReference?.sourceUri);
  const sourcePath =
    normalizeString(sourceReference?.source_path) ??
    normalizeString(sourceReference?.sourcePath);
  const previewUrl =
    normalizeNonInlineSourceReference(sourceReference?.preview_url) ??
    normalizeNonInlineSourceReference(sourceReference?.previewUrl);
  const sidecarRef =
    sourceReference?.sidecar_ref !== undefined
      ? sourceReference.sidecar_ref
      : sourceReference?.sidecarRef;
  const sidecarRefRecord = isRecord(sidecarRef) ? sidecarRef : null;
  const refId =
    normalizeString(sourceReference?.ref_id) ??
    normalizeString(sourceReference?.refId) ??
    normalizeString(sidecarRefRecord?.ref);
  const sha256 = normalizeString(sourceReference?.sha256);
  const byteSize =
    typeof sourceReference?.byte_size === "number" &&
    Number.isFinite(sourceReference.byte_size)
      ? sourceReference.byte_size
      : typeof sourceReference?.byteSize === "number" &&
          Number.isFinite(sourceReference.byteSize)
        ? sourceReference.byteSize
        : undefined;
  const mediaKind = normalizeString(part.kind);

  const mediaReference = buildMediaReference({
    caption,
    title,
    uri,
    refId,
    mimeType,
    mediaKind,
    sidecarRef,
    sourceUri,
    sourcePath,
    previewUrl,
    sha256,
    byteSize,
  });
  if (!mediaReference.caption && !mediaReference.title) {
    mediaReference.caption = contentPartTextFallback({ caption, title, uri });
  }

  return {
    type: "media_reference",
    reference: mediaReference,
    metadata: {
      ...buildBaseMetadata(item, index),
      source: "agent_media_reference",
      referenceUri: uri,
      ...(mediaKind ? { mediaKind } : {}),
      ...(refId ? { refId } : {}),
      ...(mimeType ? { mimeType } : {}),
      ...(title ? { title } : {}),
      ...(caption ? { caption } : {}),
      ...(sidecarRef !== undefined ? { sidecarRef } : {}),
      ...(sourceUri ? { sourceUri } : {}),
      ...(sourcePath ? { sourcePath } : {}),
      ...(previewUrl ? { previewUrl } : {}),
      ...(sha256 ? { sha256 } : {}),
      ...(typeof byteSize === "number" ? { byteSize } : {}),
    },
  };
}

export function messageContentPartsFromAgentThreadItem(
  item: AgentMessageItem,
): ContentPart[] {
  if (!Array.isArray(item.contentParts)) {
    return [];
  }

  return item.contentParts.flatMap((part, index) => {
    if (!isRecord(part)) {
      return [];
    }
    const partType = normalizeString(part.type);
    if (partType === "text") {
      const textPart = buildTextContentPart(item, part, index);
      return textPart ? [textPart] : [];
    }
    if (partType === "media") {
      const mediaPart = buildMediaReferenceContentPart(item, part, index);
      return mediaPart ? [mediaPart] : [];
    }
    return [];
  });
}

export function mediaReferenceContentPartFromThreadItem(
  item: MediaItem,
): ContentPart | null {
  const uri = normalizeString(item.uri);
  if (!uri || isInlineMediaUri(uri)) {
    return null;
  }

  const mimeType = normalizeString(item.mime_type);
  const title = mediaTitleFromUri(uri);
  const sourcePath = isAbsoluteLocalPath(uri) ? uri : null;
  const mediaKind = mediaKindFromMimeType(mimeType);
  return {
    type: "media_reference",
    reference: buildMediaReference({
      caption: normalizeString(item.preview),
      title,
      uri,
      refId: null,
      mimeType,
      mediaKind,
      sourceUri: null,
      sourcePath,
      previewUrl: null,
      sha256: null,
    }),
    metadata: {
      ...buildBaseMetadata(item, 0),
      source: "agent_media_reference",
      referenceUri: uri,
      mediaKind,
      ...(mimeType ? { mimeType } : {}),
      ...(sourcePath ? { sourcePath } : {}),
    },
  };
}

export function imageGenerationContentPartFromThreadItem(
  item: ImageGenerationItem,
): ContentPart | null {
  const generationStatus = normalizeString(
    item.generation_status,
  )?.toLowerCase();
  const result = normalizeString(item.result);
  if (
    item.status !== "completed" ||
    generationStatus !== "completed" ||
    !result
  ) {
    return null;
  }

  const caption = normalizeString(item.revised_prompt);
  const sourcePath = normalizeString(item.saved_path);
  const title = sourcePath ? mediaTitleFromUri(sourcePath) : null;
  const uri = `data:image/png;base64,${result}`;
  return {
    type: "media_reference",
    reference: buildMediaReference({
      caption,
      title,
      uri,
      refId: null,
      mimeType: "image/png",
      mediaKind: "image",
      sourceUri: null,
      sourcePath,
      previewUrl: null,
      sha256: null,
    }),
    metadata: {
      ...buildBaseMetadata(item, 0),
      source: "hosted_image_generation",
      generationStatus,
      mediaKind: "image",
      mimeType: "image/png",
      ...(caption ? { caption } : {}),
      ...(sourcePath ? { sourcePath } : {}),
    },
  };
}
