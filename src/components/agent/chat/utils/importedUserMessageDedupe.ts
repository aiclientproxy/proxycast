import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import type { Message } from "../types";

type ImportedUserMessage = Extract<
  AgentThreadItem,
  { type: "user_message" }
>;

type ImportedImagePart = Extract<
  NonNullable<ImportedUserMessage["content_parts"]>[number],
  { type: "image" }
>;

interface ImportedUserMessageProvenance {
  sourceType: string;
  sourceSequence: number;
  sourceThreadId: string;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function isImportedHistoryMetadata(value: unknown): boolean {
  return asRecord(value)?.imported === true;
}

function readString(
  record: Record<string, unknown> | null,
  ...keys: string[]
): string | null {
  if (!record) return null;
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function readSequence(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.trunc(value);
  }
  if (typeof value === "string" && /^\d+$/u.test(value.trim())) {
    return Number(value);
  }
  return null;
}

function readImportedUserMessageProvenance(
  item: ImportedUserMessage,
): ImportedUserMessageProvenance | null {
  const metadata = asRecord(item.metadata);
  if (metadata?.imported !== true) return null;

  const provenance = asRecord(
    metadata.source_provenance ?? metadata.sourceProvenance,
  );
  const sourceType = readString(
    metadata,
    "source_event_type",
    "sourceEventType",
  ) || readString(
    provenance,
    "sourceEventType",
    "source_event_type",
    "sourcePayloadType",
    "source_payload_type",
  );
  const sourceSequence = readSequence(
    metadata.source_event_seq ??
      metadata.sourceEventSeq ??
      provenance?.sourceEventSeq ??
      provenance?.source_event_seq,
  );
  const sourceThreadId = readString(
    metadata,
    "source_thread_id",
    "sourceThreadId",
  ) || readString(provenance, "sourceThreadId", "source_thread_id");

  if (!sourceType || sourceSequence === null || !sourceThreadId) {
    return null;
  }

  return {
    sourceType: sourceType.toLowerCase(),
    sourceSequence,
    sourceThreadId: sourceThreadId.toLowerCase(),
  };
}

export function normalizeImportedUserMessageText(value: unknown): string {
  if (typeof value !== "string") return "";
  return value
    .replace(/<image\b[^>]*>/giu, " ")
    .replace(/<\/image>/giu, " ")
    .replace(/\[\s*image(?:\s*#\s*\d+)?\s*\]/giu, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function readImportedUserMessageText(item: ImportedUserMessage): string {
  const content = normalizeImportedUserMessageText(item.content);
  if (content) return content;

  return (item.content_parts || [])
    .flatMap((part) =>
      part.type === "text" ? [normalizeImportedUserMessageText(part.text)] : [],
    )
    .filter(Boolean)
    .join(" ");
}

function isResponseItemSource(sourceType: string): boolean {
  return sourceType === "message" || sourceType === "response_item";
}

function isEventMessageSource(sourceType: string): boolean {
  return sourceType === "user_message" || sourceType === "event_msg";
}

function areAdjacentSourceMessages(
  left: ImportedUserMessage,
  right: ImportedUserMessage,
): boolean {
  const leftProvenance = readImportedUserMessageProvenance(left);
  const rightProvenance = readImportedUserMessageProvenance(right);
  if (!leftProvenance || !rightProvenance) return false;

  return (
    leftProvenance.sourceThreadId === rightProvenance.sourceThreadId &&
    Math.abs(leftProvenance.sourceSequence - rightProvenance.sourceSequence) ===
      1 &&
    ((isResponseItemSource(leftProvenance.sourceType) &&
      isEventMessageSource(rightProvenance.sourceType)) ||
      (isEventMessageSource(leftProvenance.sourceType) &&
        isResponseItemSource(rightProvenance.sourceType)))
  );
}

function imageParts(item: ImportedUserMessage): ImportedImagePart[] {
  return (item.content_parts || []).filter(
    (part): part is ImportedImagePart => part.type === "image",
  );
}

function importedDuplicateKey(
  item: ImportedUserMessage,
): string | null {
  const provenance = readImportedUserMessageProvenance(item);
  if (!provenance) return null;

  const text = readImportedUserMessageText(item);
  if (text) return `text:${provenance.sourceThreadId}:${text}`;

  const count = imageParts(item).length;
  return count > 0
    ? `images:${provenance.sourceThreadId}:${count}`
    : null;
}

function inlineImageData(part: ImportedImagePart): string | null {
  const value = part.data.trim();
  if (!value) return null;
  if (!value.toLowerCase().startsWith("data:")) return value;
  const commaIndex = value.indexOf(",");
  return commaIndex >= 0 ? value.slice(commaIndex + 1).trim() || null : null;
}

function mergeImageParts(
  preferred: ImportedImagePart | undefined,
  duplicate: ImportedImagePart | undefined,
): ImportedImagePart | undefined {
  if (!preferred) return duplicate;
  if (!duplicate) return preferred;

  const merged = { ...preferred, ...duplicate } as ImportedImagePart;
  const data = inlineImageData(preferred) || inlineImageData(duplicate);
  if (data) {
    merged.data = data;
    delete merged.uri;
    delete merged.source_path;
  }
  return merged;
}

function mergeImportedUserMessage(
  preferred: ImportedUserMessage,
  duplicate: ImportedUserMessage,
): ImportedUserMessage {
  const preferredImages = imageParts(preferred);
  const duplicateImages = imageParts(duplicate);
  const images = Array.from(
    { length: Math.max(preferredImages.length, duplicateImages.length) },
    (_, index) => mergeImageParts(preferredImages[index], duplicateImages[index]),
  ).filter((part): part is ImportedImagePart => Boolean(part));
  const preferredNonImages = (preferred.content_parts || []).filter(
    (part) => part.type !== "image",
  );
  const duplicateNonImages = (duplicate.content_parts || []).filter(
    (part) => part.type !== "image",
  );

  return {
    ...preferred,
    content: preferred.content.trim() ? preferred.content : duplicate.content,
    content_parts:
      preferredNonImages.length > 0
        ? [...preferredNonImages, ...images]
        : [...duplicateNonImages, ...images],
  };
}

function preferImportedUserMessage(
  left: ImportedUserMessage,
  right: ImportedUserMessage,
): ImportedUserMessage {
  const leftSource = readImportedUserMessageProvenance(left)?.sourceType || "";
  const rightSource =
    readImportedUserMessageProvenance(right)?.sourceType || "";
  return isEventMessageSource(rightSource) && !isEventMessageSource(leftSource)
    ? right
    : left;
}

function areImportedUserMessageDuplicates(
  left: ImportedUserMessage,
  right: ImportedUserMessage,
): boolean {
  if (!areAdjacentSourceMessages(left, right)) return false;

  const leftText = readImportedUserMessageText(left);
  const rightText = readImportedUserMessageText(right);
  if (leftText || rightText) return leftText === rightText;

  return imagePartsRepresentSameImages(imageParts(left), imageParts(right));
}

/** 合并 Codex 导入把同一条用户输入写成 response_item + event_msg 的重复 Item。 */
export function dedupeImportedUserMessageItems(
  items: readonly AgentThreadItem[],
): AgentThreadItem[] {
  const result: AgentThreadItem[] = [];
  const candidatesByKey = new Map<string, number[]>();

  for (const item of items) {
    if (item.type !== "user_message") {
      result.push(item);
      continue;
    }

    const key = importedDuplicateKey(item);
    const candidates = key ? candidatesByKey.get(key) || [] : [];
    const candidateIndex = candidates.findIndex((index) => {
      const existing = result[index];
      return (
        existing?.type === "user_message" &&
        areImportedUserMessageDuplicates(existing, item)
      );
    });

    if (candidateIndex >= 0) {
      const resultIndex = candidates[candidateIndex]!;
      const existing = result[resultIndex] as ImportedUserMessage;
      const preferred = preferImportedUserMessage(existing, item);
      const duplicate = preferred.id === existing.id ? item : existing;
      result[resultIndex] = mergeImportedUserMessage(preferred, duplicate);
      // 一个 source event 只能参与一次配对，避免 9/10/11 三条相同消息串成一条。
      candidates.splice(candidateIndex, 1);
      continue;
    }

    if (key) {
      candidates.push(result.length);
      candidatesByKey.set(key, candidates);
    }
    result.push(item);
  }

  return result;
}

function normalizeImageDataIdentity(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const normalized = value.trim();
  if (!normalized) return null;
  if (normalized.toLowerCase().startsWith("data:")) {
    const commaIndex = normalized.indexOf(",");
    const payload = commaIndex >= 0 ? normalized.slice(commaIndex + 1) : "";
    const compactPayload = payload.replace(/\s+/g, "");
    return compactPayload ? `data:${compactPayload}` : null;
  }
  return `data:${normalized.replace(/\s+/g, "")}`;
}

function normalizeImageReferenceIdentity(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const normalized = value.trim();
  if (!normalized) return null;
  if (normalized.toLowerCase().startsWith("data:")) {
    return normalizeImageDataIdentity(normalized);
  }
  return `ref:${normalized.replace(/\\/g, "/").toLowerCase()}`;
}

function importedImageIdentities(part: ImportedImagePart): Set<string> {
  const identities = new Set<string>();
  const dataIdentity = normalizeImageDataIdentity(part.data);
  if (dataIdentity) identities.add(dataIdentity);
  for (const value of [part.uri, part.source_path]) {
    const identity = normalizeImageReferenceIdentity(value);
    if (identity) identities.add(identity);
  }
  return identities;
}

function imagePartsRepresentSameImages(
  left: readonly ImportedImagePart[],
  right: readonly ImportedImagePart[],
): boolean {
  if (left.length === 0 || left.length !== right.length) return false;

  const rightIdentities = right.map(importedImageIdentities);
  const matchedRightIndexes = new Set<number>();
  for (const leftPart of left) {
    const leftIdentities = importedImageIdentities(leftPart);
    if (leftIdentities.size === 0) return false;

    const matchIndex = rightIdentities.findIndex((identities, index) => {
      if (matchedRightIndexes.has(index)) return false;
      for (const identity of leftIdentities) {
        if (identities.has(identity)) return true;
      }
      return false;
    });
    if (matchIndex < 0) return false;
    matchedRightIndexes.add(matchIndex);
  }
  return true;
}

function canonicalImageIdentities(
  item: Extract<AgentThreadItem, { type: "user_message" }>,
): Set<string> {
  const identities = new Set<string>();
  for (const part of item.content_parts || []) {
    if (part.type !== "image") continue;
    for (const identity of importedImageIdentities(part)) {
      identities.add(identity);
    }
  }
  return identities;
}

function messageImageIdentities(message: Message): Set<string> {
  const identities = new Set<string>();
  for (const image of message.images || []) {
    const dataIdentity = normalizeImageDataIdentity(image.data);
    if (dataIdentity) identities.add(dataIdentity);
    for (const value of [image.sourceUri, image.sourcePath, image.previewUrl]) {
      const identity = normalizeImageReferenceIdentity(value);
      if (identity) {
        identities.add(identity);
      }
    }
  }
  return identities;
}

/** 判断旧 Message 是否就是同一条 canonical user_message，避免历史气泡重复。 */
export function messageMatchesCanonicalUserMessage(
  message: Message,
  item: Extract<AgentThreadItem, { type: "user_message" }>,
): boolean {
  if (message.role !== "user") return false;
  if (message.id === item.id || item.client_id === message.id) return true;

  // A persisted runtime turn is authoritative. Only pending turns are still
  // provisional and may fall back to content/image identity.
  const runtimeTurnId = message.runtimeTurnId?.trim();
  if (
    runtimeTurnId &&
    !runtimeTurnId.startsWith("pending-turn:") &&
    runtimeTurnId !== item.turn_id
  ) {
    return false;
  }

  const messageText = normalizeImportedUserMessageText(message.content);
  const itemText = normalizeImportedUserMessageText(
    item.content ||
      (item.content_parts || [])
        .flatMap((part) =>
          part.type === "text" ? [part.text] : [],
        )
        .join("\n"),
  );
  if (messageText && itemText && messageText === itemText) return true;

  const messageImages = messageImageIdentities(message);
  if (messageImages.size === 0) return false;
  for (const identity of canonicalImageIdentities(item)) {
    if (messageImages.has(identity)) return true;
  }
  return false;
}
