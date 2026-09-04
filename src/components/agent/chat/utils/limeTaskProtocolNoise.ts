import { resolveAgentChatCopy } from "./agentChatCopy";
import { resolveContentWorkbenchToolCopy } from "./contentWorkbenchToolCopy";

const LIME_CREATE_TASK_RE = /\blime_create_([a-z0-9_]+?)_task\b/i;
const INTERNAL_RPC_ERROR_PAIR_RE =
  /(?:^|[^\d])-?\d{4,6}\s*:\s*-?\d{4,6}(?:\b|:)/;
const INTERNAL_TASK_FILE_RE =
  /(?:^|[\s"'`])\.lime[\\/]+tasks[\\/]+[a-z0-9_-]+[\\/]+/i;
const INTERNAL_TASK_ID_RE = /(?:任务\s*ID|task[_\s-]*id)\s*[:：]/i;
const FAILURE_CUE_RE =
  /(?:failed|failure|error|执行失败|开始失败|调用失败|生成失败|创建失败|未找到可执行|tool failed)/i;

const LIME_CREATE_TASK_FAILURE_LABELS: Record<
  string,
  { key: string; defaultValue: string }
> = {
  video_generation: {
    key: "failure.videoGeneration",
    defaultValue: "视频生成失败",
  },
  audio_generation: {
    key: "failure.audioGeneration",
    defaultValue: "配音生成失败",
  },
  transcription: {
    key: "failure.transcription",
    defaultValue: "转写失败",
  },
  broadcast_generation: {
    key: "failure.broadcastGeneration",
    defaultValue: "口播生成失败",
  },
  cover_generation: {
    key: "failure.coverGeneration",
    defaultValue: "封面生成失败",
  },
  resource_search: {
    key: "failure.resourceSearch",
    defaultValue: "素材检索失败",
  },
  modal_resource_search: {
    key: "failure.resourceSearch",
    defaultValue: "素材检索失败",
  },
  url_parse: {
    key: "failure.urlParse",
    defaultValue: "链接解析失败",
  },
  typesetting: {
    key: "failure.typesetting",
    defaultValue: "排版失败",
  },
};

const NORMALIZED_LIME_CREATE_TASK_KINDS: Record<string, string> = {
  limecreateaudiogenerationtask: "audio_generation",
  limecreatetranscriptiontask: "transcription",
  limecreatebroadcastgenerationtask: "broadcast_generation",
  limecreatecovergenerationtask: "cover_generation",
  limecreateresourcesearchtask: "resource_search",
  limecreatemodalresourcesearchtask: "modal_resource_search",
  limecreateimagegenerationtask: "image_generation",
  limecreateurlparsetask: "url_parse",
  limecreatetypesettingtask: "typesetting",
};

const DIRECT_CONTENT_FAILURE_LABELS: Record<
  string,
  { key: string; defaultValue: string }
> = {
  videogenerate: {
    key: "failure.videoGeneration",
    defaultValue: "视频生成失败",
  },
  generateimage: {
    key: "failure.imageGeneration",
    defaultValue: "生成失败",
  },
  socialgeneratecoverimage: {
    key: "failure.coverImageGeneration",
    defaultValue: "封面图生成失败",
  },
};

export function normalizeLimeProtocolName(value?: string | null): string {
  return (value || "").replace(/[^a-z0-9]+/gi, "").toLowerCase();
}

export function extractLimeCreateTaskKind(
  value?: string | null,
): string | null {
  const match = (value || "").match(LIME_CREATE_TASK_RE);
  if (match?.[1]) {
    return match[1].toLowerCase();
  }

  const normalized = normalizeLimeProtocolName(value);
  return NORMALIZED_LIME_CREATE_TASK_KINDS[normalized] || null;
}

function hasImageGenerationContext(value?: string | null): boolean {
  const normalized = normalizeLimeProtocolName(value);
  const taskKind = extractLimeCreateTaskKind(value);

  return Boolean(
    taskKind === "image_generation" ||
    normalized.includes("limecreateimagegenerationtask") ||
    normalized.includes("imagegeneration") ||
    normalized.includes("imagegenerate") ||
    normalized.includes("generateimage"),
  );
}

function resolveLimeTaskProtocolFailureKind(params: {
  toolName?: string | null;
  text?: string | null;
}): string | null {
  return (
    extractLimeCreateTaskKind(params.toolName) ||
    extractLimeCreateTaskKind(params.text)
  );
}

function resolveDirectContentFailureCopy(params: {
  toolName?: string | null;
  text?: string | null;
}): { key: string; defaultValue: string } | null {
  const normalizedToolName = normalizeLimeProtocolName(params.toolName);
  const normalizedText = normalizeLimeProtocolName(params.text);

  return (
    DIRECT_CONTENT_FAILURE_LABELS[normalizedToolName] ||
    Object.entries(DIRECT_CONTENT_FAILURE_LABELS).find(([key]) =>
      normalizedText.includes(key),
    )?.[1] ||
    null
  );
}

function hasLimeTaskProtocolContext(params: {
  toolName?: string | null;
  text?: string | null;
}): boolean {
  return Boolean(
    resolveLimeTaskProtocolFailureKind(params) ||
    resolveDirectContentFailureCopy(params),
  );
}

export function isImageGenerationToolContext(params: {
  toolName?: string | null;
  text?: string | null;
}): boolean {
  return (
    hasImageGenerationContext(params.toolName) ||
    hasImageGenerationContext(params.text)
  );
}

export function containsInternalLimeTaskProtocolNoise(
  value?: string | null,
): boolean {
  const text = value || "";
  if (!text.trim()) {
    return false;
  }

  return (
    INTERNAL_RPC_ERROR_PAIR_RE.test(text) ||
    LIME_CREATE_TASK_RE.test(text) ||
    INTERNAL_TASK_FILE_RE.test(text) ||
    INTERNAL_TASK_ID_RE.test(text)
  );
}

export function isImageGenerationProtocolFailure(params: {
  toolName?: string | null;
  text?: string | null;
}): boolean {
  if (!isImageGenerationToolContext(params)) {
    return false;
  }

  return isLimeTaskProtocolFailure(params);
}

export function isLimeTaskProtocolFailure(params: {
  toolName?: string | null;
  text?: string | null;
}): boolean {
  const text = params.text || "";
  if (!text.trim()) {
    return false;
  }
  if (!hasLimeTaskProtocolContext(params)) {
    return false;
  }

  return (
    containsInternalLimeTaskProtocolNoise(text) || FAILURE_CUE_RE.test(text)
  );
}

export function isImageGenerationProtocolFailureResidue(
  value?: string | null,
): boolean {
  return isImageGenerationProtocolFailure({ text: value });
}

export function isLimeTaskProtocolFailureResidue(
  value?: string | null,
): boolean {
  return isLimeTaskProtocolFailure({ text: value });
}

export function resolveImageGenerationFailureDisplayText(): string {
  return resolveAgentChatCopy(
    "imageWorkbenchPreview.placeholder.failed",
    "生成失败",
  );
}

export function resolveLimeTaskProtocolFailureDisplayText(params: {
  toolName?: string | null;
  text?: string | null;
}): string {
  const directCopy = resolveDirectContentFailureCopy(params);
  if (directCopy) {
    return resolveContentWorkbenchToolCopy(
      directCopy.key,
      directCopy.defaultValue,
    );
  }

  const taskKind = resolveLimeTaskProtocolFailureKind(params);
  if (taskKind === "image_generation") {
    return resolveImageGenerationFailureDisplayText();
  }

  const copy = taskKind ? LIME_CREATE_TASK_FAILURE_LABELS[taskKind] : null;
  return copy
    ? resolveContentWorkbenchToolCopy(copy.key, copy.defaultValue)
    : resolveContentWorkbenchToolCopy("failure.task", "任务发起失败");
}
