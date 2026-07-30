import type { AppServerJsonRpcNotification } from "@/lib/api/appServer";
import { normalizeMediaReadNumber } from "./mediaReferencePreviewPolicy";

export interface MediaReferencePreviewProgress {
  contentRange?: string;
  hasMore: boolean;
  loadedBytes: number;
  mimeType?: string;
  sha256?: string;
  totalBytes: number;
}

export function emitStreamingMediaReadProgress(params: {
  expectedOffset?: number | null;
  expectedStreamId?: string | null;
  expectedUri?: string | null;
  notifications: readonly AppServerJsonRpcNotification[];
  onProgress?: (progress: MediaReferencePreviewProgress) => void;
  onStreamId?: (streamId: string) => void;
  seenEventIds?: Set<string>;
  threadId: string;
}): { emitted: boolean; streamId?: string } {
  if (!params.onProgress || params.notifications.length === 0) {
    return { emitted: false };
  }

  let emitted = false;
  let streamId = params.expectedStreamId?.trim() || undefined;
  const expectedUri = params.expectedUri?.trim();
  const expectedOffset =
    typeof params.expectedOffset === "number" &&
    Number.isFinite(params.expectedOffset)
      ? params.expectedOffset
      : undefined;
  for (const notification of params.notifications) {
    if (notification.method !== "agentSession/event") {
      continue;
    }
    const event = normalizeRecord(normalizeRecord(notification.params)?.event);
    if (
      readString(event, "type") !== "media.read.chunk" ||
      readString(event, "threadId") !== params.threadId
    ) {
      continue;
    }
    const payload = normalizeRecord(event?.payload);
    const payloadStreamId = readString(payload, "streamId");
    if (!payloadStreamId || readBoolean(payload, "done") !== false) {
      continue;
    }
    if (streamId && payloadStreamId !== streamId) {
      continue;
    }
    const chunk = normalizeRecord(payload?.chunk);
    const chunkThreadId = readString(chunk, "threadId");
    if (chunkThreadId && chunkThreadId !== params.threadId) {
      continue;
    }
    if (expectedUri && readString(chunk, "uri") !== expectedUri) {
      continue;
    }
    const offset = normalizeMediaReadNumber(readNumber(chunk, "offset")) ?? 0;
    if (!streamId) {
      if (expectedOffset !== undefined && offset !== expectedOffset) {
        continue;
      }
      streamId = payloadStreamId;
      params.onStreamId?.(payloadStreamId);
    }
    const eventId = readString(event, "eventId");
    if (eventId && params.seenEventIds?.has(eventId)) {
      continue;
    }
    if (eventId) {
      params.seenEventIds?.add(eventId);
    }
    const bytes = normalizeMediaReadNumber(readNumber(chunk, "bytes"));
    const totalBytes = normalizeMediaReadNumber(
      readNumber(chunk, "totalBytes"),
    );
    if (!bytes || !totalBytes) {
      continue;
    }
    const loadedBytes = offset + bytes;
    if (loadedBytes >= totalBytes) {
      continue;
    }
    params.onProgress({
      contentRange: readString(chunk, "contentRange"),
      hasMore: readBoolean(chunk, "hasMore") === true,
      loadedBytes,
      mimeType: readString(chunk, "mimeType"),
      sha256: readString(chunk, "sha256"),
      totalBytes,
    });
    emitted = true;
  }
  return { emitted, streamId };
}

function normalizeRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readString(record: Record<string, unknown> | null, key: string) {
  const value = record?.[key];
  return typeof value === "string" ? value.trim() : undefined;
}

function readNumber(record: Record<string, unknown> | null, key: string) {
  const value = record?.[key];
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function readBoolean(record: Record<string, unknown> | null, key: string) {
  const value = record?.[key];
  return typeof value === "boolean" ? value : undefined;
}
