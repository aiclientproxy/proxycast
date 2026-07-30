import type {
  AppServerMediaReadParams,
  AppServerMediaReadResponse,
} from "@/lib/api/appServer";
import type { Artifact } from "@/lib/artifact/types";
import type { Message, MessagePreviewTarget } from "../types";
import {
  buildMediaReadParams,
  createMediaReferencePreviewArtifact,
} from "./mediaReferencePreviewArtifacts";
import {
  MEDIA_REFERENCE_PREVIEW_CHUNK_BYTES,
  MEDIA_REFERENCE_PREVIEW_MAX_BYTES,
  normalizeMediaReadNumber,
} from "./mediaReferencePreviewPolicy";

type Translate = (key: string, options?: Record<string, unknown>) => string;

function decodeBase64Bytes(contentBase64: string): Uint8Array {
  const binary = globalThis.atob(contentBase64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function isMediaMimeType(mimeType?: string | null): boolean {
  const normalized = mimeType?.trim().toLowerCase() || "";
  return (
    normalized.startsWith("image/") ||
    normalized.startsWith("audio/") ||
    normalized.startsWith("video/")
  );
}

function mediaPageIndex(offset: number, pageBytes: number): number {
  if (pageBytes <= 0) {
    return 1;
  }
  return Math.floor(offset / pageBytes) + 1;
}

function createMediaReferencePageWindowPreviewArtifact(params: {
  maxBytes: number;
  media: AppServerMediaReadResponse;
  message: Message;
  pageBytes: number;
  target: Extract<MessagePreviewTarget, { kind: "media_reference" }>;
  t: Translate;
}): Artifact {
  const pageOffset = params.media.offset ?? 0;
  const pageLength = params.media.length ?? params.media.bytes;
  const loadedBytes = pageOffset + params.media.bytes;
  const previousOffset =
    pageOffset > 0 ? Math.max(0, pageOffset - params.pageBytes) : undefined;
  return createMediaReferencePreviewArtifact({
    fallbackStatus: "unavailable",
    message: params.message,
    policy: "sidecar_page_window",
    previewBudget: {
      canReadNextPage: params.media.hasMore === true,
      canReadPreviousPage: pageOffset > 0,
      chunkBytes: params.pageBytes,
      loadedBytes,
      maxBytes: params.maxBytes,
      nextOffset: params.media.hasMore === true ? loadedBytes : undefined,
      pageIndex: mediaPageIndex(pageOffset, params.pageBytes),
      pageLength,
      pageOffset,
      previousOffset,
      totalBytes: params.media.totalBytes,
    },
    progress: {
      contentRange: params.media.contentRange,
      hasMore: params.media.hasMore === true,
      loadedBytes,
      mimeType: params.media.mimeType ?? undefined,
      sha256: params.media.sha256,
      totalBytes: params.media.totalBytes,
    },
    target: params.target,
    t: params.t,
  });
}

export async function createMediaReferencePagedPreviewArtifact(params: {
  message: Message;
  target: Extract<MessagePreviewTarget, { kind: "media_reference" }>;
  threadId?: string | null;
  t: Translate;
  offset: number;
  length?: number;
  readMedia: (
    request: AppServerMediaReadParams,
  ) => Promise<AppServerMediaReadResponse>;
  maxBytes?: number;
  pageBytes?: number;
  shouldContinue?: () => boolean;
}): Promise<Artifact | null> {
  const canContinue = () => params.shouldContinue?.() !== false;
  const maxBytes = normalizeMediaReadNumber(
    params.maxBytes ?? MEDIA_REFERENCE_PREVIEW_MAX_BYTES,
  );
  const requestedPageBytes = normalizeMediaReadNumber(
    params.length ?? params.pageBytes ?? MEDIA_REFERENCE_PREVIEW_CHUNK_BYTES,
  );
  const offset = normalizeMediaReadNumber(params.offset);
  if (!maxBytes || !requestedPageBytes || offset === null) {
    return null;
  }

  const pageBytes = Math.min(requestedPageBytes, maxBytes);
  const request = buildMediaReadParams({
    threadId: params.threadId,
    target: params.target,
    maxBytes: pageBytes,
    offset,
    length: pageBytes,
  });
  if (!request || !canContinue()) {
    return null;
  }

  const media = await params.readMedia(request);
  if (!canContinue()) {
    return null;
  }

  const mediaOffset = normalizeMediaReadNumber(media.offset) ?? offset;
  const mediaLength = normalizeMediaReadNumber(media.length) ?? media.bytes;
  const totalBytes = normalizeMediaReadNumber(media.totalBytes) ?? media.bytes;
  const mimeType =
    media.mimeType?.trim() || params.target.reference.mimeType?.trim() || "";
  const bytes = decodeBase64Bytes(media.contentBase64.trim());
  if (
    mediaOffset !== offset ||
    mediaLength !== media.bytes ||
    bytes.byteLength !== media.bytes ||
    bytes.byteLength === 0 ||
    totalBytes <= 0 ||
    !isMediaMimeType(mimeType) ||
    !media.sha256.trim()
  ) {
    return null;
  }

  return createMediaReferencePageWindowPreviewArtifact({
    maxBytes,
    media: {
      ...media,
      mimeType,
      totalBytes,
      offset: mediaOffset,
      length: mediaLength,
    },
    message: params.message,
    pageBytes,
    target: params.target,
    t: params.t,
  });
}
