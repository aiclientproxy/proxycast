import type { AgentToolResultImage as ToolResultImage } from "@/lib/api/agentProtocol";
import type { MessageImage } from "../types";
import {
  LIME_TOOL_METADATA_BEGIN,
  LIME_TOOL_METADATA_END,
} from "./agentChatCoreUtils";

export const resolveHistoryUserDataText = (
  userData: unknown,
): string | undefined => {
  if (typeof userData === "string") {
    const value = userData.trim();
    return value || undefined;
  }

  if (userData && typeof userData === "object") {
    const record = userData as Record<string, unknown>;
    const answer = record.answer;
    if (typeof answer === "string" && answer.trim()) {
      return answer.trim();
    }
    const other = record.other;
    if (typeof other === "string" && other.trim()) {
      return other.trim();
    }
    try {
      const serialized = JSON.stringify(record);
      return serialized === "{}" ? undefined : serialized;
    } catch {
      return undefined;
    }
  }

  if (userData === null || userData === undefined) return undefined;
  return String(userData);
};

export const stringifyToolArguments = (
  argumentsValue: unknown,
): string | undefined => {
  if (argumentsValue === null || argumentsValue === undefined) return undefined;
  if (typeof argumentsValue === "string") {
    const value = argumentsValue.trim();
    return value || undefined;
  }
  try {
    return JSON.stringify(argumentsValue);
  } catch {
    return undefined;
  }
};

export const parseDataUrlToHistoryImage = (
  rawUrl: string,
): MessageImage | null => {
  const normalized = rawUrl.trim();
  if (!normalized.startsWith("data:")) return null;

  const commaIndex = normalized.indexOf(",");
  if (commaIndex <= 5) return null;

  const meta = normalized.slice(5, commaIndex);
  const payload = normalized.slice(commaIndex + 1).trim();
  if (!payload) return null;

  const metaSegments = meta.split(";").map((segment) => segment.trim());
  const mediaType = metaSegments[0] || "image/png";
  const hasBase64 = metaSegments.some(
    (segment) => segment.toLowerCase() === "base64",
  );
  if (!hasBase64) return null;

  return {
    mediaType,
    data: payload,
    sourceUri: normalized,
    previewUrl: normalized,
  };
};

function readStringField(
  record: Record<string, unknown>,
  ...keys: string[]
): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return undefined;
}

function readRecordField(
  record: Record<string, unknown>,
  ...keys: string[]
): Record<string, unknown> | undefined {
  for (const key of keys) {
    const value = record[key];
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return value as Record<string, unknown>;
    }
  }
  return undefined;
}

function readOptionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.trunc(value)
    : undefined;
}

function normalizeHistoryImageMetadata(
  value: unknown,
): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return Object.fromEntries(Object.entries(value));
}

function normalizeHistoryImageFromUri(params: {
  uri: string;
  metadata?: Record<string, unknown>;
  fallbackMediaType?: string;
}): MessageImage | null {
  const uri = params.uri.trim();
  if (!uri) {
    return null;
  }

  const dataUrlImage = parseDataUrlToHistoryImage(uri);
  const metadata = params.metadata;
  const mediaType =
    readStringField(metadata || {}, "mediaType", "media_type", "mimeType") ||
    params.fallbackMediaType ||
    dataUrlImage?.mediaType ||
    "image/png";
  const sourcePath =
    readStringField(metadata || {}, "localPath", "local_path", "path") ||
    (uri.startsWith("/") || /^[a-zA-Z]:[\\/]/.test(uri) ? uri : undefined);
  const index = readOptionalNumber(metadata?.index);

  if (dataUrlImage) {
    return {
      ...dataUrlImage,
      mediaType,
      metadata,
      index,
    };
  }

  return {
    mediaType,
    data: "",
    sourceUri: uri,
    sourcePath,
    previewUrl: sourcePath ? undefined : uri,
    metadata,
    index,
  };
}

export const normalizeHistoryImagePart = (
  rawPart: Record<string, unknown>,
): MessageImage | null => {
  const metadata = normalizeHistoryImageMetadata(rawPart.metadata);
  const sourcePath = readStringField(rawPart, "source_path", "sourcePath");
  const rawData = readStringField(rawPart, "data");

  // Inline data is the only self-contained preview. Prefer it over a stale
  // host path that may have been carried alongside the same imported image.
  if (rawData) {
    const dataUrlImage = parseDataUrlToHistoryImage(rawData);
    const mediaType =
      readStringField(rawPart, "mime_type", "media_type", "mediaType") ||
      dataUrlImage?.mediaType ||
      "image/png";
    if (dataUrlImage) {
      return {
        ...dataUrlImage,
        mediaType,
        metadata,
        index: readOptionalNumber(metadata?.index),
      };
    }
    return {
      data: rawData,
      mediaType,
      metadata,
      index: readOptionalNumber(metadata?.index),
      ...(sourcePath ? { sourcePath } : {}),
    };
  }

  const uri = readStringField(rawPart, "uri");
  if (uri) {
    const image = normalizeHistoryImageFromUri({
      uri,
      metadata,
      fallbackMediaType: readStringField(
        rawPart,
        "mime_type",
        "media_type",
        "mediaType",
      ),
    });
    if (image && sourcePath && !image.sourcePath) {
      image.sourcePath = sourcePath;
    }
    return image;
  }

  if (sourcePath) {
    return normalizeHistoryImageFromUri({
      uri: sourcePath,
      metadata,
      fallbackMediaType: readStringField(
        rawPart,
        "mime_type",
        "media_type",
        "mediaType",
      ),
    });
  }

  const imageUrlValue = rawPart.image_url ?? rawPart.url;
  if (typeof imageUrlValue === "string") {
    return normalizeHistoryImageFromUri({
      uri: imageUrlValue,
      metadata,
      fallbackMediaType: readStringField(
        rawPart,
        "mime_type",
        "media_type",
        "mediaType",
      ),
    });
  }

  if (imageUrlValue && typeof imageUrlValue === "object") {
    const imageUrlRecord =
      readRecordField(rawPart, "image_url", "url") ||
      (imageUrlValue as Record<string, unknown>);
    const nestedUrl = readStringField(imageUrlRecord, "url", "image_url") || "";
    if (nestedUrl) {
      return normalizeHistoryImageFromUri({
        uri: nestedUrl,
        metadata,
        fallbackMediaType: readStringField(
          imageUrlRecord,
          "mime_type",
          "media_type",
          "mediaType",
        ),
      });
    }
  }

  return null;
};

export const parseMimeTypeFromDataUrl = (
  rawUrl: string,
): string | undefined => {
  const normalized = rawUrl.trim();
  if (!normalized.startsWith("data:image/")) return undefined;
  const commaIndex = normalized.indexOf(",");
  if (commaIndex <= 5) return undefined;
  const meta = normalized.slice(5, commaIndex);
  const mimeType = meta.split(";")[0]?.trim();
  if (!mimeType || !mimeType.startsWith("image/")) return undefined;
  return mimeType;
};

export const extractDataImageUrlsFromText = (text: string): string[] => {
  if (!text.trim()) return [];
  const pattern = /data:image\/[\w.+-]+;base64,[A-Za-z0-9+/=]+/g;
  const matches = text.match(pattern);
  if (!matches) return [];
  const deduped = new Set<string>();
  for (const match of matches) {
    const value = match.trim();
    if (value) deduped.add(value);
  }
  return Array.from(deduped);
};

export const normalizeToolResultImages = (
  value: unknown,
  fallbackText?: string,
  metadata?: unknown,
): ToolResultImage[] | undefined => {
  const normalized: ToolResultImage[] = [];
  const seen = new Set<string>();

  const appendImage = (
    rawSrc: string,
    mimeType?: string,
    origin?: ToolResultImage["origin"],
  ) => {
    const src = rawSrc.trim();
    if (!src || seen.has(src)) return;
    seen.add(src);
    normalized.push({ src, mimeType, origin });
  };

  if (Array.isArray(value)) {
    for (const item of value) {
      if (typeof item === "string") {
        appendImage(item, parseMimeTypeFromDataUrl(item), "data_url");
        continue;
      }
      if (!item || typeof item !== "object") continue;
      const record = item as Record<string, unknown>;
      const src = typeof record.src === "string" ? record.src : "";
      if (!src.trim()) continue;
      const mimeType =
        (typeof record.mimeType === "string" && record.mimeType) ||
        (typeof record.mime_type === "string" && record.mime_type) ||
        parseMimeTypeFromDataUrl(src);
      const origin =
        record.origin === "data_url" ||
        record.origin === "tool_payload" ||
        record.origin === "file_path"
          ? record.origin
          : undefined;
      appendImage(src, mimeType, origin);
    }
  }

  if (normalized.length === 0 && metadata && typeof metadata === "object") {
    const record = metadata as Record<string, unknown>;
    const modelVisibleImage =
      record.model_visible_image === true || record.modelVisibleImage === true;
    const rawImageUrl = record.image_url ?? record.imageUrl;
    const imageUrl = typeof rawImageUrl === "string" ? rawImageUrl.trim() : "";

    if (modelVisibleImage && imageUrl.startsWith("data:image/")) {
      const mimeType =
        (typeof record.mime_type === "string" && record.mime_type.trim()) ||
        (typeof record.mimeType === "string" && record.mimeType.trim()) ||
        parseMimeTypeFromDataUrl(imageUrl);
      appendImage(imageUrl, mimeType, "tool_payload");
    }
  }

  if (normalized.length === 0 && typeof fallbackText === "string") {
    for (const dataUrl of extractDataImageUrlsFromText(fallbackText)) {
      appendImage(dataUrl, parseMimeTypeFromDataUrl(dataUrl), "data_url");
    }
  }

  return normalized.length > 0 ? normalized : undefined;
};

export const parseToolResultMetadataRecord = (
  value: unknown,
): Record<string, unknown> | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return Object.fromEntries(Object.entries(value));
};

export const extractLimeToolMetadataBlock = (
  text?: string,
): { text: string; metadata?: Record<string, unknown> } => {
  if (!text) {
    return { text: "" };
  }

  const beginIndex = text.lastIndexOf(LIME_TOOL_METADATA_BEGIN);
  const endIndex = text.lastIndexOf(LIME_TOOL_METADATA_END);
  if (beginIndex < 0 || endIndex < beginIndex) {
    return { text };
  }

  const metadataRaw = text
    .slice(beginIndex + LIME_TOOL_METADATA_BEGIN.length, endIndex)
    .trim();
  const parsedMetadata = (() => {
    if (!metadataRaw) return undefined;
    try {
      const parsed = JSON.parse(metadataRaw);
      return parseToolResultMetadataRecord(parsed) || undefined;
    } catch {
      return undefined;
    }
  })();

  const cleaned = text.slice(0, beginIndex).replace(/\s+$/, "");
  return {
    text: cleaned,
    metadata: parsedMetadata,
  };
};

export const parseLimeExecutionSummary = (
  text?: string,
): Record<string, unknown> | undefined => {
  if (!text) return undefined;
  const marker = "[Lime 执行摘要]";
  const markerIndex = text.lastIndexOf(marker);
  if (markerIndex < 0) return undefined;

  const raw = text.slice(markerIndex + marker.length).trim();
  if (!raw) return undefined;

  const metadata: Record<string, unknown> = {};
  for (const line of raw.split("\n")) {
    const separatorIndex = line.indexOf(":");
    if (separatorIndex <= 0) continue;
    const key = line.slice(0, separatorIndex).trim();
    const rawValue = line.slice(separatorIndex + 1).trim();
    if (!key || !rawValue) continue;

    if (rawValue === "true" || rawValue === "false") {
      metadata[key] = rawValue === "true";
      continue;
    }

    const numericValue = Number(rawValue);
    if (!Number.isNaN(numericValue) && rawValue === String(numericValue)) {
      metadata[key] = numericValue;
      continue;
    }

    metadata[key] = rawValue;
  }

  return Object.keys(metadata).length > 0 ? metadata : undefined;
};

export const normalizeToolResultMetadata = (
  value: unknown,
  ...fallbackTexts: Array<string | undefined>
): Record<string, unknown> | undefined => {
  const direct = parseToolResultMetadataRecord(value);
  const fromBlock = fallbackTexts.reduce<Record<string, unknown> | undefined>(
    (merged, text) => {
      const metadata = extractLimeToolMetadataBlock(text).metadata;
      if (!metadata) return merged;
      return {
        ...(merged || {}),
        ...metadata,
      };
    },
    undefined,
  );
  const fromSummary = fallbackTexts.reduce<Record<string, unknown> | undefined>(
    (merged, text) => {
      const metadata = parseLimeExecutionSummary(text);
      if (!metadata) return merged;
      return {
        ...(merged || {}),
        ...metadata,
      };
    },
    undefined,
  );

  if (!direct && !fromBlock && !fromSummary) return undefined;
  return {
    ...(direct || {}),
    ...(fromBlock || {}),
    ...(fromSummary || {}),
  };
};

export const normalizeIncomingToolResult = <
  T extends {
    success?: boolean;
    output?: string;
    error?: string;
    images?: unknown;
    metadata?: unknown;
    structuredContent?: unknown;
    structured_content?: unknown;
  },
>(
  result: T | null | undefined,
):
  | (T & {
      output: string;
      error?: string;
      images?: ToolResultImage[];
      metadata?: Record<string, unknown>;
      structuredContent?: unknown;
      structured_content?: unknown;
    })
  | undefined => {
  if (!result) return undefined;

  const normalizedOutput = extractLimeToolMetadataBlock(result.output);
  const normalizedError = extractLimeToolMetadataBlock(result.error);
  const metadata = normalizeToolResultMetadata(
    result.metadata,
    result.output,
    result.error,
  );

  return {
    ...result,
    output: normalizedOutput.text,
    error: normalizedError.text || undefined,
    images: normalizeToolResultImages(
      result.images,
      normalizedOutput.text,
      metadata,
    ),
    metadata,
  };
};

export const isToolResultSuccessful = (
  result:
    | {
        success?: boolean;
        metadata?: Record<string, unknown>;
      }
    | null
    | undefined,
): boolean => {
  if (!result) return false;

  const metadata = result.metadata;
  if (metadata?.reported_success === false) {
    return false;
  }
  if (metadata?.reported_success === true) {
    return true;
  }
  if (
    typeof metadata?.exit_code === "number" &&
    Number.isFinite(metadata.exit_code) &&
    metadata.exit_code !== 0
  ) {
    return false;
  }

  return result.success !== false;
};
