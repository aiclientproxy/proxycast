import type { MessageVideoTaskPreview } from "../types";
import {
  asRecord,
  readFirstArrayRecord,
  readMetadataPositiveNumber,
  readMetadataString,
  resolveTaskPreviewPhase,
  resolveTaskPreviewStatus,
  type ToolResultPreviewParams,
} from "./taskPreviewToolResultShared";
import {
  resolveVideoTaskFallbackPrompt,
  resolveVideoTaskStatusMessage,
} from "./taskPreviewCopy";

function extractVideoTaskPromptFromToolArguments(toolArguments: string | undefined): {
  prompt?: string;
  durationSeconds?: number;
  aspectRatio?: string;
  resolution?: string;
  providerId?: string;
  model?: string;
} {
  if (!toolArguments) {
    return {};
  }

  try {
    const parsed = JSON.parse(toolArguments) as Record<string, unknown>;
    const prompt = readMetadataString([parsed], ["prompt"]);
    const durationSeconds = readMetadataPositiveNumber(
      [parsed],
      ["duration", "duration_seconds", "durationSeconds"],
    );
    const aspectRatio = readMetadataString(
      [parsed],
      ["aspect_ratio", "aspectRatio"],
    );
    const resolution = readMetadataString([parsed], ["resolution"]);
    const providerId = readMetadataString(
      [parsed],
      ["provider_id", "providerId"],
    );
    const model = readMetadataString([parsed], ["model"]);
    return {
      prompt,
      durationSeconds,
      aspectRatio,
      resolution,
      providerId,
      model,
    };
  } catch {
    return {};
  }
}

export function buildVideoTaskPreviewFromToolResult(
  params: ToolResultPreviewParams,
): MessageVideoTaskPreview | null {
  const resultRecord = asRecord(params.toolResult);
  const metadata = asRecord(resultRecord?.metadata);
  const taskResult =
    asRecord(resultRecord?.result) || asRecord(metadata?.result);
  const firstVideo = readFirstArrayRecord(
    [taskResult, metadata, resultRecord],
    ["videos", "results"],
  );
  const taskId = readMetadataString(
    [metadata, resultRecord, taskResult],
    ["task_id", "taskId", "id"],
  );
  const taskType = readMetadataString(
    [metadata, resultRecord, taskResult],
    ["task_type", "taskType"],
  );
  if (!taskId || !taskType) {
    return null;
  }

  const normalizedTaskType = taskType.trim().toLowerCase();
  if (!normalizedTaskType.includes("video")) {
    return null;
  }

  const parsedArguments = extractVideoTaskPromptFromToolArguments(
    params.toolArguments,
  );
  const status = readMetadataString(
    [metadata, resultRecord, taskResult],
    ["status"],
  );
  const videoUrl = readMetadataString(
    [firstVideo, taskResult, metadata, resultRecord],
    ["url", "result_url", "resultUrl"],
  );
  const thumbnailUrl = readMetadataString(
    [firstVideo, taskResult, metadata, resultRecord],
    ["thumbnail_url", "thumbnailUrl", "poster_url", "posterUrl"],
  );
  const durationMs = readMetadataPositiveNumber(
    [firstVideo],
    ["duration_ms", "durationMs"],
  );
  const durationSeconds =
    parsedArguments.durationSeconds ||
    (durationMs ? Math.max(1, Math.round(durationMs / 1000)) : undefined);
  const previewStatus = resolveTaskPreviewStatus(status);
  const phase = resolveTaskPreviewPhase(status);
  const errorMessage =
    previewStatus === "failed"
      ? readMetadataString(
          [metadata, resultRecord, taskResult],
          ["error", "error_message", "errorMessage"],
        )
      : undefined;

  return {
    kind: "video_generate",
    taskId,
    taskType: "video_generate",
    prompt:
      parsedArguments.prompt ||
      readMetadataString([metadata, resultRecord, taskResult], ["prompt"]) ||
      params.fallbackPrompt.trim() ||
      resolveVideoTaskFallbackPrompt(),
    status: previewStatus,
    projectId:
      readMetadataString(
        [metadata, resultRecord, taskResult],
        ["project_id", "projectId"],
      ) || null,
    contentId:
      readMetadataString(
        [metadata, resultRecord, taskResult],
        ["content_id", "contentId"],
      ) || null,
    videoUrl: videoUrl || null,
    thumbnailUrl: thumbnailUrl || null,
    durationSeconds,
    aspectRatio:
      parsedArguments.aspectRatio ||
      readMetadataString(
        [metadata, resultRecord, taskResult],
        ["aspect_ratio", "aspectRatio"],
      ),
    resolution:
      parsedArguments.resolution ||
      readMetadataString([metadata, resultRecord, taskResult], ["resolution"]),
    providerId:
      parsedArguments.providerId ||
      readMetadataString(
        [metadata, resultRecord, taskResult],
        ["provider_id", "providerId", "provider"],
      ) ||
      null,
    model:
      parsedArguments.model ||
      readMetadataString([metadata, resultRecord, taskResult], ["model"]) ||
      null,
    phase,
    statusMessage: resolveVideoTaskStatusMessage({
      status: previewStatus,
      phase,
      hasVideoUrl: Boolean(videoUrl),
      errorMessage,
    }),
  };
}
