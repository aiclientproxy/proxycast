import type {
  AppServerMediaReadParams,
  AppServerMediaReadResponse,
  AppServerJsonRpcNotification,
} from "@/lib/api/appServer";
import { isAbsoluteLocalFilePath } from "@/lib/api/fileSystem";
import { createPreviewArtifact } from "@/lib/artifact/previewArtifact";
import type { Artifact } from "@/lib/artifact/types";
import type { Message, MessagePreviewTarget } from "../types";
import {
  MEDIA_REFERENCE_PREVIEW_CHUNK_BYTES,
  MEDIA_REFERENCE_PREVIEW_MAX_BYTES,
  mediaReferencePreviewPolicyMeta,
  normalizeMediaReadNumber,
  type MediaReferencePreviewBudgetFacts,
  type MediaReferencePreviewPolicy,
} from "./mediaReferencePreviewPolicy";
import {
  emitStreamingMediaReadProgress,
  type MediaReferencePreviewProgress,
} from "./mediaReferencePreviewStreamingProgress";

type Translate = (key: string, options?: Record<string, unknown>) => string;

export { emitStreamingMediaReadProgress };
export type { MediaReferencePreviewProgress };

type MediaReferencePreviewReadEnvelope = {
  media: AppServerMediaReadResponse;
  notifications?: readonly AppServerJsonRpcNotification[];
};

type MediaReferencePreviewReadResult =
  | AppServerMediaReadResponse
  | MediaReferencePreviewReadEnvelope;

type MediaReferencePreviewSource =
  | {
      kind: "direct_uri" | "preview_url" | "source_uri";
      value: string;
    }
  | {
      kind: "source_path";
      value: string;
    };

function isDirectPreviewMediaUri(uri: string): boolean {
  return /^(https?|file|asset):/iu.test(uri) || uri.startsWith("//");
}

function isSidecarMediaReadUri(uri?: string | null): boolean {
  return Boolean(uri?.trim().toLowerCase().startsWith("sidecar://"));
}

function isInlineMediaPayloadUri(uri?: string | null): boolean {
  return Boolean(uri?.trimStart().toLowerCase().startsWith("data:"));
}

function isMediaMimeType(mimeType?: string | null): boolean {
  const normalized = mimeType?.trim().toLowerCase() || "";
  return (
    normalized.startsWith("image/") ||
    normalized.startsWith("audio/") ||
    normalized.startsWith("video/")
  );
}

function isLikelyMediaReference(
  reference: Extract<
    MessagePreviewTarget,
    { kind: "media_reference" }
  >["reference"],
): boolean {
  const kind = reference.kind?.trim().toLowerCase();
  return (
    isMediaMimeType(reference.mimeType) ||
    kind === "image" ||
    kind === "audio" ||
    kind === "video"
  );
}

function extensionForMimeType(mimeType?: string | null): string {
  const normalized = mimeType?.trim().toLowerCase();
  switch (normalized) {
    case "image/png":
      return ".png";
    case "image/jpeg":
    case "image/jpg":
      return ".jpg";
    case "image/gif":
      return ".gif";
    case "image/webp":
      return ".webp";
    case "image/svg+xml":
      return ".svg";
    case "audio/mpeg":
      return ".mp3";
    case "audio/wav":
      return ".wav";
    case "audio/ogg":
      return ".ogg";
    case "video/mp4":
      return ".mp4";
    case "video/webm":
      return ".webm";
    default:
      return "";
  }
}

function resolveMediaReferencePreviewSource(
  reference: Extract<
    MessagePreviewTarget,
    { kind: "media_reference" }
  >["reference"],
): MediaReferencePreviewSource | null {
  const previewUrl = reference.previewUrl?.trim();
  if (previewUrl && isDirectPreviewMediaUri(previewUrl)) {
    return { kind: "preview_url", value: previewUrl };
  }

  const sourcePath = reference.sourcePath?.trim();
  if (sourcePath && isAbsoluteLocalFilePath(sourcePath)) {
    return { kind: "source_path", value: sourcePath };
  }

  const sourceUri = reference.sourceUri?.trim();
  if (sourceUri && isDirectPreviewMediaUri(sourceUri)) {
    return { kind: "source_uri", value: sourceUri };
  }

  const uri = reference.uri.trim();
  if (uri && isDirectPreviewMediaUri(uri)) {
    return { kind: "direct_uri", value: uri };
  }

  return null;
}

function resolveMediaReferenceReadOwner(
  reference: Extract<
    MessagePreviewTarget,
    { kind: "media_reference" }
  >["reference"],
): Pick<AppServerMediaReadParams, "refId" | "sidecarRef" | "uri"> {
  const sidecarRef = reference.sidecarRef;
  const refId = reference.refId?.trim();
  const uriCandidates = [reference.sourceUri, reference.uri];
  const uri = uriCandidates.find(isSidecarMediaReadUri)?.trim();
  return {
    ...(refId ? { refId } : {}),
    ...(sidecarRef !== undefined ? { sidecarRef } : {}),
    ...(uri ? { uri } : {}),
  };
}

export function buildMediaReadParams(params: {
  threadId?: string | null;
  target: Extract<MessagePreviewTarget, { kind: "media_reference" }>;
  maxBytes?: number;
  offset?: number;
  length?: number;
}): AppServerMediaReadParams | null {
  const threadId = params.threadId?.trim();
  if (!threadId) {
    return null;
  }

  const reference = params.target.reference;
  if (!isLikelyMediaReference(reference)) {
    return null;
  }
  if (resolveMediaReferencePreviewSource(reference)) {
    return null;
  }

  const readOwner = resolveMediaReferenceReadOwner(reference);
  if (
    !readOwner.uri &&
    !readOwner.refId &&
    readOwner.sidecarRef === undefined
  ) {
    return null;
  }

  const request: AppServerMediaReadParams = {
    threadId,
    maxBytes: params.maxBytes ?? MEDIA_REFERENCE_PREVIEW_MAX_BYTES,
    offset: params.offset ?? 0,
    length:
      params.length ?? params.maxBytes ?? MEDIA_REFERENCE_PREVIEW_MAX_BYTES,
    ...readOwner,
  };
  return request;
}

function isCompleteMediaReadResponse(
  media: AppServerMediaReadResponse,
): boolean {
  const offset = media.offset ?? 0;
  const totalBytes = media.totalBytes ?? media.bytes;
  return offset === 0 && media.hasMore !== true && media.bytes === totalBytes;
}

function decodeBase64Bytes(contentBase64: string): Uint8Array {
  const binary = globalThis.atob(contentBase64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function copyBytesToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

function formatMediaReadContentRange(totalBytes: number): string {
  if (totalBytes <= 0) {
    return "bytes */0";
  }
  return `bytes 0-${totalBytes - 1}/${totalBytes}`;
}

function normalizeMediaPreviewReadResult(
  result: MediaReferencePreviewReadResult,
): {
  media: AppServerMediaReadResponse;
  notifications: readonly AppServerJsonRpcNotification[];
} {
  if (isMediaReferencePreviewReadEnvelope(result)) {
    return {
      media: result.media,
      notifications: result.notifications ?? [],
    };
  }
  return { media: result, notifications: [] };
}

function isMediaReferencePreviewReadEnvelope(
  result: MediaReferencePreviewReadResult,
): result is MediaReferencePreviewReadEnvelope {
  return "media" in result;
}

function buildMediaReferenceFallbackMarkdown(params: {
  fallbackStatus?: "loading" | "unavailable";
  progress?: MediaReferencePreviewProgress;
  title: string;
  reference: Extract<
    MessagePreviewTarget,
    { kind: "media_reference" }
  >["reference"];
  t: Translate;
}): string {
  const {
    fallbackStatus = "unavailable",
    progress,
    reference,
    t,
    title,
  } = params;
  const lines = [
    `# ${title}`,
    "",
    fallbackStatus === "loading"
      ? t("agentChat.mediaReferencePreview.loading")
      : t("agentChat.mediaReferencePreview.previewUnavailable"),
    "",
  ];
  if (fallbackStatus === "loading" && progress) {
    lines.push(
      t("agentChat.mediaReferencePreview.loadingProgress", {
        loaded: progress.loadedBytes,
        total: progress.totalBytes,
      }),
      "",
    );
  }
  lines.push(
    t("agentChat.mediaReferencePreview.reference", {
      value: reference.uri,
    }),
  );
  if (reference.kind) {
    lines.push(
      t("agentChat.mediaReferencePreview.kind", {
        value: reference.kind,
      }),
    );
  }
  if (reference.mimeType) {
    lines.push(
      t("agentChat.mediaReferencePreview.mime", {
        value: reference.mimeType,
      }),
    );
  }
  if (typeof reference.byteSize === "number") {
    lines.push(
      t("agentChat.mediaReferencePreview.byteSize", {
        value: reference.byteSize,
      }),
    );
  }
  if (reference.sha256) {
    lines.push(
      t("agentChat.mediaReferencePreview.sha256", {
        value: reference.sha256,
      }),
    );
  }
  return lines.join("\n");
}

export function createMediaReferencePreviewArtifact(params: {
  fallbackStatus?: "loading" | "unavailable";
  message: Message;
  policy?: MediaReferencePreviewPolicy;
  progress?: MediaReferencePreviewProgress;
  previewBudget?: MediaReferencePreviewBudgetFacts;
  target: Extract<MessagePreviewTarget, { kind: "media_reference" }>;
  t: Translate;
}): Artifact {
  const {
    fallbackStatus = "unavailable",
    message,
    progress,
    target,
    t,
  } = params;
  const reference = target.reference;
  const sourceRef =
    reference.uri.trim() || `${message.id}:media:${target.index}`;
  const title =
    reference.title?.trim() ||
    reference.caption?.trim() ||
    t("agentChat.mediaReferencePreview.fallbackTitle", {
      index: target.index + 1,
    });
  const resolvedPreviewSource = resolveMediaReferencePreviewSource(reference);
  const previewSource = isMediaMimeType(reference.mimeType)
    ? resolvedPreviewSource
    : null;
  const previewPath =
    previewSource?.kind === "source_path" ? previewSource.value : sourceRef;
  const previewUrl =
    previewSource && previewSource.kind !== "source_path"
      ? previewSource.value
      : undefined;
  const policy =
    params.policy ??
    (previewSource?.kind === "source_path"
      ? "source_path_owner"
      : previewSource
        ? "direct_owner"
        : fallbackStatus === "loading"
          ? "sidecar_progress"
          : "sidecar_metadata_fallback");
  const projection = createPreviewArtifact({
    source: "session_file",
    sourceRef,
    path: previewSource
      ? previewPath
      : `${message.id}-media-${target.index + 1}.md`,
    title,
    content: previewSource
      ? ""
      : buildMediaReferenceFallbackMarkdown({
          fallbackStatus,
          progress,
          reference,
          t,
          title,
        }),
    isBinary: Boolean(previewSource),
    mimeType: previewSource ? reference.mimeType : "text/markdown",
    previewUrl,
    meta: {
      ...mediaReferencePreviewPolicyMeta({
        policy,
        facts: {
          ...params.previewBudget,
          loadedBytes:
            params.previewBudget?.loadedBytes ?? progress?.loadedBytes,
          totalBytes: params.previewBudget?.totalBytes ?? progress?.totalBytes,
        },
      }),
      openedFrom: "message-media-reference",
      messageId: message.id,
      contentPartIndex: target.index,
      mediaKind: reference.kind,
      mediaUri: reference.uri,
      mediaRefId: reference.refId,
      mediaSourceUri: isInlineMediaPayloadUri(reference.sourceUri)
        ? undefined
        : reference.sourceUri,
      mediaSourcePath: reference.sourcePath,
      mediaPreviewUrl:
        previewSource?.kind === "preview_url" ? previewSource.value : undefined,
      mediaPreviewSource:
        previewSource?.kind ??
        (fallbackStatus === "loading" ? "sidecar_progress" : undefined),
      mediaPreviewStatus: previewSource ? undefined : fallbackStatus,
      mediaReadContentRange: progress?.contentRange,
      mediaReadHasMore: progress?.hasMore,
      mediaReadLength: progress?.loadedBytes,
      mediaReadTotalBytes: progress?.totalBytes,
      mediaMimeType: reference.mimeType,
      mediaSha256: reference.sha256,
      mediaByteSize: reference.byteSize,
    },
  });
  return projection.artifact;
}

export function createMediaReferenceProgressPreviewArtifact(params: {
  message: Message;
  progress: MediaReferencePreviewProgress;
  target: Extract<MessagePreviewTarget, { kind: "media_reference" }>;
  t: Translate;
}): Artifact {
  return createMediaReferencePreviewArtifact({
    fallbackStatus: "loading",
    message: params.message,
    progress: params.progress,
    target: params.target,
    t: params.t,
  });
}

export function createMediaReferenceBinaryPreviewArtifact(params: {
  message: Message;
  target: Extract<MessagePreviewTarget, { kind: "media_reference" }>;
  media: AppServerMediaReadResponse;
  t: Translate;
  previewUrl?: string;
  previewSource?: "sidecar_read" | "sidecar_object_url";
  previewBudget?: MediaReferencePreviewBudgetFacts;
}): Artifact {
  const { message, target, media, t } = params;
  const reference = target.reference;
  const mimeType =
    media.mimeType?.trim() || reference.mimeType?.trim() || undefined;
  if (!isMediaMimeType(mimeType)) {
    return createMediaReferencePreviewArtifact({ message, target, t });
  }

  const title =
    reference.title?.trim() ||
    reference.caption?.trim() ||
    t("agentChat.mediaReferencePreview.fallbackTitle", {
      index: target.index + 1,
    });
  const contentBase64 = media.contentBase64.trim();
  const previewUrl =
    params.previewUrl?.trim() || `data:${mimeType};base64,${contentBase64}`;
  const sourceRef = media.uri.trim() || reference.uri.trim();
  const extension = extensionForMimeType(mimeType);
  const projection = createPreviewArtifact({
    source: "session_file",
    sourceRef,
    path: `${message.id}-media-${target.index + 1}${extension}`,
    title,
    content: "",
    isBinary: true,
    mimeType,
    size: media.bytes,
    previewUrl,
    meta: {
      ...mediaReferencePreviewPolicyMeta({
        policy:
          params.previewSource === "sidecar_object_url"
            ? "sidecar_object_url"
            : "sidecar_read",
        facts: {
          ...params.previewBudget,
          loadedBytes: params.previewBudget?.loadedBytes ?? media.length,
          totalBytes: params.previewBudget?.totalBytes ?? media.totalBytes,
        },
      }),
      openedFrom: "message-media-reference",
      messageId: message.id,
      contentPartIndex: target.index,
      mediaKind: reference.kind,
      mediaUri: reference.uri,
      mediaRefId: reference.refId,
      mediaReadUri: media.uri,
      mediaPreviewSource: params.previewSource ?? "sidecar_read",
      mediaPreviewObjectUrl:
        params.previewSource === "sidecar_object_url" ? previewUrl : undefined,
      mediaReadOffset: media.offset ?? 0,
      mediaReadLength: media.length ?? media.bytes,
      mediaReadTotalBytes: media.totalBytes ?? media.bytes,
      mediaReadContentRange: media.contentRange,
      mediaReadHasMore: media.hasMore === true,
      mediaMimeType: mimeType,
      mediaSha256: media.sha256,
      mediaByteSize: media.bytes,
      sidecarRef: media.sidecarRef ?? reference.sidecarRef,
      contentKind: reference.kind,
      renderMode: "media",
    },
  });
  return projection.artifact;
}

function createMediaReferenceBudgetExceededPreviewArtifact(params: {
  chunkBytes: number;
  maxBytes: number;
  media: AppServerMediaReadResponse;
  message: Message;
  target: Extract<MessagePreviewTarget, { kind: "media_reference" }>;
  t: Translate;
}): Artifact {
  const loadedBytes = (params.media.offset ?? 0) + params.media.bytes;
  return createMediaReferencePreviewArtifact({
    fallbackStatus: "unavailable",
    message: params.message,
    policy: "sidecar_preview_budget_exceeded",
    previewBudget: {
      chunkBytes: params.chunkBytes,
      loadedBytes,
      maxBytes: params.maxBytes,
      nextOffset: params.media.hasMore === true ? loadedBytes : undefined,
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

export function createMediaReferenceObjectUrlPreviewArtifact(params: {
  message: Message;
  target: Extract<MessagePreviewTarget, { kind: "media_reference" }>;
  media: AppServerMediaReadResponse;
  t: Translate;
  createObjectUrl?: (blob: Blob) => string;
}): Artifact | null {
  const { media } = params;
  const mimeType =
    media.mimeType?.trim() || params.target.reference.mimeType?.trim() || "";
  if (!isMediaMimeType(mimeType) || !isCompleteMediaReadResponse(media)) {
    return null;
  }

  const bytes = decodeBase64Bytes(media.contentBase64.trim());
  const blob = new Blob([copyBytesToArrayBuffer(bytes)], { type: mimeType });
  const previewUrl = (params.createObjectUrl ?? URL.createObjectURL)(blob);
  return createMediaReferenceBinaryPreviewArtifact({
    message: params.message,
    target: params.target,
    media,
    t: params.t,
    previewUrl,
    previewSource: "sidecar_object_url",
  });
}

export async function createMediaReferenceChunkedObjectUrlPreviewArtifact(params: {
  message: Message;
  target: Extract<MessagePreviewTarget, { kind: "media_reference" }>;
  threadId?: string | null;
  t: Translate;
  readMedia: (
    request: AppServerMediaReadParams,
  ) => Promise<MediaReferencePreviewReadResult>;
  createObjectUrl?: (blob: Blob) => string;
  maxBytes?: number;
  chunkBytes?: number;
  onProgress?: (progress: MediaReferencePreviewProgress) => void;
  shouldContinue?: () => boolean;
}): Promise<Artifact | null> {
  const canContinue = () => params.shouldContinue?.() !== false;
  const maxBytes = normalizeMediaReadNumber(
    params.maxBytes ?? MEDIA_REFERENCE_PREVIEW_MAX_BYTES,
  );
  const requestedChunkBytes = normalizeMediaReadNumber(
    params.chunkBytes ?? MEDIA_REFERENCE_PREVIEW_CHUNK_BYTES,
  );
  if (!maxBytes || !requestedChunkBytes) {
    return null;
  }

  const chunkBytes = Math.min(requestedChunkBytes, maxBytes);
  const firstRequest = buildMediaReadParams({
    threadId: params.threadId,
    target: params.target,
    maxBytes: chunkBytes,
    offset: 0,
    length: chunkBytes,
  });
  if (!firstRequest) {
    return null;
  }

  if (!canContinue()) {
    return null;
  }
  const firstResult = normalizeMediaPreviewReadResult(
    await params.readMedia({ ...firstRequest, stream: true }),
  );
  let didEmitStreamingProgress = emitStreamingMediaReadProgress({
    notifications: firstResult.notifications,
    onProgress: params.onProgress,
    threadId: firstRequest.threadId,
  }).emitted;
  const first = firstResult.media;
  if (!canContinue()) {
    return null;
  }
  const mimeType =
    first.mimeType?.trim() || params.target.reference.mimeType?.trim() || "";
  const totalBytes = normalizeMediaReadNumber(first.totalBytes) ?? first.bytes;
  const sha256 = first.sha256.trim();
  if (!isMediaMimeType(mimeType) || totalBytes <= 0) {
    return null;
  }
  if (totalBytes > maxBytes) {
    return createMediaReferenceBudgetExceededPreviewArtifact({
      chunkBytes,
      maxBytes,
      media: first,
      message: params.message,
      target: params.target,
      t: params.t,
    });
  }
  if ((first.offset ?? 0) !== 0 || !sha256) {
    return null;
  }

  const buffers: ArrayBuffer[] = [];
  let expectedOffset = 0;
  let latest = first;

  while (true) {
    const chunkOffset =
      normalizeMediaReadNumber(latest.offset) ?? expectedOffset;
    if (chunkOffset !== expectedOffset) {
      return null;
    }

    const chunkTotalBytes =
      normalizeMediaReadNumber(latest.totalBytes) ?? totalBytes;
    const chunkLength = normalizeMediaReadNumber(latest.length) ?? latest.bytes;
    const chunkMimeType = latest.mimeType?.trim() || mimeType;
    if (
      chunkTotalBytes !== totalBytes ||
      chunkLength !== latest.bytes ||
      chunkMimeType !== mimeType ||
      latest.sha256.trim() !== sha256
    ) {
      return null;
    }

    const bytes = decodeBase64Bytes(latest.contentBase64.trim());
    if (bytes.byteLength !== latest.bytes || bytes.byteLength === 0) {
      return null;
    }
    buffers.push(copyBytesToArrayBuffer(bytes));
    expectedOffset += bytes.byteLength;

    if (expectedOffset > totalBytes) {
      return null;
    }
    if (expectedOffset < totalBytes && !didEmitStreamingProgress) {
      params.onProgress?.({
        contentRange: latest.contentRange,
        hasMore: latest.hasMore === true,
        loadedBytes: expectedOffset,
        mimeType,
        sha256,
        totalBytes,
      });
    }
    if (expectedOffset === totalBytes) {
      if (latest.hasMore === true) {
        return null;
      }
      break;
    }
    if (latest.hasMore !== true) {
      return null;
    }

    const nextLength = Math.min(chunkBytes, totalBytes - expectedOffset);
    const nextRequest = buildMediaReadParams({
      threadId: params.threadId,
      target: params.target,
      maxBytes: chunkBytes,
      offset: expectedOffset,
      length: nextLength,
    });
    if (!nextRequest) {
      return null;
    }
    if (!canContinue()) {
      return null;
    }
    const latestResult = normalizeMediaPreviewReadResult(
      await params.readMedia({ ...nextRequest, stream: true }),
    );
    didEmitStreamingProgress = emitStreamingMediaReadProgress({
      notifications: latestResult.notifications,
      onProgress: params.onProgress,
      threadId: nextRequest.threadId,
    }).emitted;
    latest = latestResult.media;
    if (!canContinue()) {
      return null;
    }
  }

  if (!canContinue()) {
    return null;
  }
  const blob = new Blob(buffers, { type: mimeType });
  const previewUrl = (params.createObjectUrl ?? URL.createObjectURL)(blob);
  return createMediaReferenceBinaryPreviewArtifact({
    message: params.message,
    target: params.target,
    media: {
      ...first,
      bytes: totalBytes,
      totalBytes,
      offset: 0,
      length: totalBytes,
      contentRange: formatMediaReadContentRange(totalBytes),
      hasMore: false,
      contentBase64: "",
    },
    t: params.t,
    previewUrl,
    previewSource: "sidecar_object_url",
    previewBudget: {
      chunkBytes,
      maxBytes,
    },
  });
}
