import type {
  ImageStoryboardSlot,
  MessageImageWorkbenchPreview,
} from "../types";
import {
  asRecord,
  parseJsonRecordString,
  readArrayRecords,
  readMetadataPositiveNumber,
  readMetadataString,
  resolveTaskPreviewPhase,
  resolveTaskPreviewStatus,
  type ToolResultPreviewParams,
} from "./taskPreviewToolResultShared";
import {
  resolveImageTaskFallbackPrompt,
  resolveImageTaskStatusMessage,
} from "./taskPreviewCopy";
import { findImageTaskRecord } from "./imageTaskToolResult";
import { sanitizeImageWorkbenchPresentationText } from "./imageWorkbenchPresentation";
import { readImageGenerationSoulMetadata } from "../workspace/imageTaskPreviewRuntimePayload";

function extractImageTaskPromptFromToolArguments(toolArguments: string | undefined): {
  prompt?: string;
  size?: string;
  imageCount?: number;
  layoutHint?: string;
  projectRootPath?: string;
} {
  if (!toolArguments) {
    return {};
  }

  try {
    const parsed = JSON.parse(toolArguments) as Record<string, unknown>;
    const prompt = readMetadataString([parsed], ["prompt"]);
    const size = readMetadataString([parsed], ["size", "resolution"]);
    const imageCount = readMetadataPositiveNumber(
      [parsed],
      ["count", "image_count", "imageCount"],
    );
    const layoutHint = readMetadataString(
      [parsed],
      ["layout_hint", "layoutHint"],
    );
    const projectRootPath = readMetadataString(
      [parsed],
      ["project_root_path", "projectRootPath"],
    );
    return { prompt, size, imageCount, layoutHint, projectRootPath };
  } catch {
    return {};
  }
}

function isAbsoluteTaskPath(value?: string | null): boolean {
  const normalized = value?.trim();
  if (!normalized) {
    return false;
  }
  return (
    normalized.startsWith("/") ||
    normalized.startsWith("\\\\") ||
    /^[A-Za-z]:[\\/]/.test(normalized)
  );
}

function joinProjectRelativeTaskPath(
  projectRootPath?: string | null,
  relativePath?: string | null,
): string | null {
  const root = projectRootPath?.trim().replace(/[\\/]+$/, "");
  const path = relativePath?.trim().replace(/^[\\/]+/, "");
  if (!root || !path) {
    return null;
  }
  if (isAbsoluteTaskPath(path)) {
    return path;
  }

  const separator = root.includes("\\") && !root.includes("/") ? "\\" : "/";
  const normalizedPath =
    separator === "\\" ? path.replace(/\//g, "\\") : path.replace(/\\/g, "/");
  return `${root}${separator}${normalizedPath}`;
}

function resolveImageTaskPreviewFilePath(params: {
  explicitTaskFilePath?: string | null;
  genericPath?: string | null;
  projectRootPath?: string | null;
}): string | null {
  const explicitTaskFilePath = params.explicitTaskFilePath?.trim();
  if (explicitTaskFilePath) {
    if (isAbsoluteTaskPath(explicitTaskFilePath)) {
      return explicitTaskFilePath;
    }
    return (
      joinProjectRelativeTaskPath(
        params.projectRootPath,
        explicitTaskFilePath,
      ) || explicitTaskFilePath
    );
  }

  const genericPath = params.genericPath?.trim();
  if (!genericPath) {
    return null;
  }
  if (isAbsoluteTaskPath(genericPath)) {
    return genericPath;
  }

  return (
    joinProjectRelativeTaskPath(params.projectRootPath, genericPath) ||
    genericPath
  );
}

function readImageStoryboardSlots(
  candidates: Array<Record<string, unknown> | null | undefined>,
  keys: string[],
): ImageStoryboardSlot[] {
  return readArrayRecords(candidates, keys)
    .map((record, index) => {
      const slotIndex =
        readMetadataPositiveNumber([record], ["slot_index", "slotIndex"]) ||
        index + 1;
      const slotId =
        readMetadataString([record], ["slot_id", "slotId"]) ||
        `storyboard-slot-${slotIndex}`;

      return {
        slotId,
        slotIndex,
        label:
          readMetadataString([record], ["label", "slot_label", "slotLabel"]) ||
          null,
        prompt:
          readMetadataString(
            [record],
            ["prompt", "slot_prompt", "slotPrompt", "revised_prompt"],
          ) || null,
        shotType:
          readMetadataString([record], ["shot_type", "shotType"]) || null,
        status: readMetadataString([record], ["status"]) || null,
      } satisfies ImageStoryboardSlot;
    })
    .sort((left, right) => left.slotIndex - right.slotIndex);
}

function isImageTaskRecord(params: {
  taskFamily?: string;
  taskType?: string;
}): boolean {
  const normalizedTaskType = params.taskType?.trim().toLowerCase() || "";
  const normalizedTaskFamily = params.taskFamily?.trim().toLowerCase() || "";
  return (
    normalizedTaskType.includes("image") ||
    normalizedTaskType.includes("cover") ||
    normalizedTaskFamily === "image" ||
    normalizedTaskFamily === "image_generation" ||
    normalizedTaskFamily.includes("image")
  );
}

function readImageTaskPresentationCaption(
  candidates: Array<Record<string, unknown> | null | undefined>,
  status: MessageImageWorkbenchPreview["status"],
  languageSource?: string | null,
): string | undefined {
  const statusKeys =
    status === "complete"
      ? ["completion_caption", "completionCaption", "complete"]
      : status === "partial"
        ? ["partial_caption", "partialCaption", "partial"]
        : status === "failed"
          ? [
              "failed_caption",
              "failedCaption",
              "failure_caption",
              "failureCaption",
              "failed",
              "failure",
            ]
          : status === "cancelled"
            ? ["cancelled_caption", "cancelledCaption", "cancelled"]
            : ["completion_caption", "completionCaption", "complete"];

  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    const presentation = asRecord(candidate.presentation) || candidate;
    const captions = asRecord(presentation.result_captions);
    const caption = readMetadataString(
      [presentation, captions],
      [...statusKeys, "result_caption", "resultCaption", "caption"],
    );
    if (caption) {
      return sanitizeImageWorkbenchPresentationText(caption, {
        languageSource,
      });
    }
  }

  return undefined;
}

function normalizeRenderableImageUrl(value?: string | null): string | null {
  const normalized = value?.trim();
  if (!normalized) {
    return null;
  }
  if (
    normalized.toLowerCase().startsWith("data:image/") ||
    normalized.startsWith("blob:") ||
    normalized.startsWith("file://") ||
    /^https?:\/\//i.test(normalized)
  ) {
    return normalized;
  }
  return null;
}

function collectImageTaskPreviewUrls(
  value: unknown,
  urls: string[],
  seenUrls: Set<string>,
  depth = 0,
): void {
  if (value === null || value === undefined || depth > 4) {
    return;
  }

  if (typeof value === "string") {
    const url = normalizeRenderableImageUrl(value);
    if (url && !seenUrls.has(url)) {
      seenUrls.add(url);
      urls.push(url);
    }
    return;
  }

  if (Array.isArray(value)) {
    value.forEach((item) =>
      collectImageTaskPreviewUrls(item, urls, seenUrls, depth + 1),
    );
    return;
  }

  const record = asRecord(value);
  if (!record) {
    return;
  }

  const directUrl =
    normalizeRenderableImageUrl(
      readMetadataString([record], ["url", "src", "imageUrl", "image_url"]),
    ) ||
    normalizeRenderableImageUrl(
      readMetadataString([record], ["b64_json", "b64Json"])
        ? `data:image/png;base64,${readMetadataString(
            [record],
            ["b64_json", "b64Json"],
          )}`
        : null,
    );
  if (directUrl && !seenUrls.has(directUrl)) {
    seenUrls.add(directUrl);
    urls.push(directUrl);
  }

  [
    record.images,
    record.outputs,
    record.results,
    record.items,
    record.data,
    record.output,
    record.result,
    record.image,
    record.asset,
    record.assets,
    record.response,
    record.responses,
  ].forEach((nested) =>
    collectImageTaskPreviewUrls(nested, urls, seenUrls, depth + 1),
  );
}

export function buildImageTaskPreviewFromToolResult(
  params: ToolResultPreviewParams,
): MessageImageWorkbenchPreview | null {
  const resultRecord = asRecord(params.toolResult);
  const detectedTaskRecord = findImageTaskRecord(params.toolResult);
  const metadata = asRecord(resultRecord?.metadata);
  const outputRecord = parseJsonRecordString(resultRecord?.output);
  const structuredContentRecord =
    asRecord(resultRecord?.structuredContent) ||
    asRecord(resultRecord?.structured_content);
  const nestedResultRecord = asRecord(resultRecord?.result);
  const responseRecord = asRecord(resultRecord?.response);
  const responseNestedRecord = asRecord(responseRecord?.record);
  const taskRecord =
    asRecord(resultRecord?.record) ||
    asRecord(metadata?.record) ||
    asRecord(outputRecord?.record) ||
    asRecord(structuredContentRecord?.record) ||
    asRecord(nestedResultRecord?.record) ||
    responseNestedRecord ||
    asRecord(detectedTaskRecord?.record) ||
    detectedTaskRecord;
  const payloadRecord =
    asRecord(taskRecord?.payload) ||
    asRecord(resultRecord?.payload) ||
    asRecord(metadata?.payload) ||
    asRecord(outputRecord?.payload) ||
    asRecord(structuredContentRecord?.payload) ||
    asRecord(nestedResultRecord?.payload) ||
    asRecord(responseRecord?.payload) ||
    asRecord(responseNestedRecord?.payload) ||
    asRecord(detectedTaskRecord?.payload);
  const progressRecord =
    asRecord(resultRecord?.progress) ||
    asRecord(metadata?.progress) ||
    asRecord(outputRecord?.progress) ||
    asRecord(structuredContentRecord?.progress) ||
    asRecord(nestedResultRecord?.progress) ||
    asRecord(taskRecord?.progress) ||
    asRecord(responseRecord?.progress) ||
    asRecord(responseNestedRecord?.progress) ||
    asRecord(detectedTaskRecord?.progress);
  const candidates = [
    resultRecord,
    detectedTaskRecord,
    metadata,
    outputRecord,
    structuredContentRecord,
    nestedResultRecord,
    responseRecord,
    responseNestedRecord,
    taskRecord,
    payloadRecord,
  ];
  const taskId = readMetadataString(candidates, ["task_id", "taskId"]);
  const taskType = readMetadataString(candidates, ["task_type", "taskType"]);
  const taskFamily = readMetadataString(candidates, [
    "task_family",
    "taskFamily",
  ]);
  if (!taskId || (!taskType && !taskFamily)) {
    return null;
  }

  if (!isImageTaskRecord({ taskFamily, taskType })) {
    return null;
  }

  const parsedArguments = extractImageTaskPromptFromToolArguments(
    params.toolArguments,
  );
  const projectRootPath =
    parsedArguments.projectRootPath ||
    readMetadataString(candidates, ["project_root_path", "projectRootPath"]);
  const status = readMetadataString(candidates, [
    "status",
    "normalized_status",
    "normalizedStatus",
  ]);
  const previewStatus = resolveTaskPreviewStatus(status);
  const requestedCount =
    parsedArguments.imageCount ||
    readMetadataPositiveNumber(candidates, [
      "requested_count",
      "requestedCount",
      "count",
      "image_count",
      "imageCount",
    ]);
  const receivedCount = readMetadataPositiveNumber(candidates, [
    "received_count",
    "receivedCount",
  ]);
  const layoutHint =
    parsedArguments.layoutHint ||
    readMetadataString(candidates, ["layout_hint", "layoutHint"]) ||
    null;
  const storyboardSlots = readImageStoryboardSlots(candidates, [
    "storyboard_slots",
    "storyboardSlots",
  ]);
  const expectedImageCount = Math.max(
    requestedCount || 0,
    storyboardSlots.length,
  );
  const resolvedImageCount =
    previewStatus === "running"
      ? expectedImageCount || requestedCount
      : receivedCount || expectedImageCount || requestedCount;
  const progressStatusMessage = readMetadataString(
    [progressRecord],
    ["message"],
  );
  const statusMessage =
    progressStatusMessage ||
    resolveImageTaskStatusMessage({
      status: previewStatus,
      layoutHint,
    });
  const soulMetadata = readImageGenerationSoulMetadata(candidates);
  const explicitTaskFilePath = readMetadataString(candidates, [
    "absolute_path",
    "absolutePath",
    "absolute_artifact_path",
    "absoluteArtifactPath",
    "task_file_path",
    "taskFilePath",
  ]);
  const genericTaskPath = readMetadataString(candidates, ["path"]);
  const artifactPath =
    readMetadataString(candidates, ["artifact_path", "artifactPath"]) ||
    genericTaskPath ||
    null;
  const taskFilePath = resolveImageTaskPreviewFilePath({
    explicitTaskFilePath,
    genericPath: genericTaskPath,
    projectRootPath,
  });
  const prompt =
    parsedArguments.prompt ||
    readMetadataString(candidates, ["prompt", "summary", "title"]) ||
    params.fallbackPrompt.trim() ||
    resolveImageTaskFallbackPrompt();
  const previewImages: string[] = [];
  const seenPreviewImageUrls = new Set<string>();
  [
    resultRecord,
    detectedTaskRecord,
    metadata,
    outputRecord,
    structuredContentRecord,
    nestedResultRecord,
    responseRecord,
    responseNestedRecord,
    taskRecord,
  ].forEach((candidate) =>
    collectImageTaskPreviewUrls(candidate, previewImages, seenPreviewImageUrls),
  );

  return {
    taskId,
    prompt,
    status: previewStatus,
    projectId:
      readMetadataString(candidates, ["project_id", "projectId"]) || null,
    contentId:
      readMetadataString(candidates, ["content_id", "contentId"]) || null,
    providerName:
      readMetadataString(candidates, [
        "provider_name",
        "providerName",
        "provider_id",
        "providerId",
        "provider",
      ]) || null,
    modelName:
      readMetadataString(candidates, [
        "model_name",
        "modelName",
        "model_id",
        "modelId",
        "model",
      ]) || null,
    caption:
      readImageTaskPresentationCaption(candidates, previewStatus, prompt) ||
      null,
    taskFilePath,
    artifactPath,
    imageUrl: previewImages[0] || null,
    previewImages: previewImages.slice(0, 9),
    imageCount: resolvedImageCount,
    expectedImageCount: expectedImageCount || requestedCount,
    layoutHint,
    storyboardSlots: storyboardSlots.length > 0 ? storyboardSlots : undefined,
    size:
      parsedArguments.size ||
      readMetadataString(candidates, ["size", "resolution"]),
    phase: resolveTaskPreviewPhase(status),
    statusMessage,
    soulMetadata,
  };
}
