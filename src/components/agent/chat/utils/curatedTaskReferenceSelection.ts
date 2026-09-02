import type {
  MemoryCategory,
  MemoryReferenceRecord,
} from "@/lib/memory/memoryReferenceTypes";
import {
  getUserFacingMemoryCategoryLabel,
  getUserFacingMemoryFallbackTitle,
  normalizeUserFacingMemorySummary,
  normalizeUserFacingMemoryTags,
  normalizeUserFacingMemoryTitle,
} from "@/lib/memory/userFacingMemoryText";
import { buildMemoryEntryCreationReplayRequestMetadata } from "./creationReplayMetadata";
import type { CreationReplayMetadata } from "./creationReplayMetadata";
import type { CuratedTaskInputValues } from "./curatedTaskTemplates";

export type CuratedTaskReferenceSourceKind = "memory";

export interface CuratedTaskReferenceEntry {
  id: string;
  sourceKind?: CuratedTaskReferenceSourceKind;
  title: string;
  summary: string;
  category: MemoryCategory;
  categoryLabel: string;
  tags: string[];
  taskPrefillByTaskId?: Record<string, CuratedTaskInputValues>;
}

export interface CuratedTaskReferenceSelection {
  referenceMemoryIds: string[];
  referenceEntries: CuratedTaskReferenceEntry[];
}

const EXPERIENCE_MEMORY_REVIEW_TASK_ID = "account-project-review";
const EXPERIENCE_MEMORY_TREND_TASK_ID = "daily-trend-briefing";
const EXPERIENCE_MEMORY_SOCIAL_POST_TASK_ID = "social-post-starter";
const EXPERIENCE_MEMORY_GOAL_LABELS = ["场景", "项目", "目标"] as const;
const EXPERIENCE_MEMORY_PLATFORM_LABELS = [
  "平台",
  "目标平台",
  "发布平台",
  "渠道",
  "发布渠道",
] as const;
const EXPERIENCE_MEMORY_REGION_LABELS = [
  "地区",
  "地域",
  "区域",
  "语种",
] as const;
const EXPERIENCE_MEMORY_AUDIENCE_LABELS = [
  "目标受众",
  "受众",
  "目标用户",
  "用户",
] as const;
export function getCuratedTaskReferenceCategoryLabel(
  category: MemoryCategory,
): string {
  return getUserFacingMemoryCategoryLabel(category);
}

export function getCuratedTaskReferenceFallbackTitle(
  category: MemoryCategory,
): string {
  return getUserFacingMemoryFallbackTitle(category);
}

function normalizeOptionalText(value?: string | null): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized ? normalized : undefined;
}

function truncateText(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }

  return `${value.slice(0, maxLength - 1).trimEnd()}…`;
}

function dedupeNonEmptyText(
  values: Array<string | null | undefined>,
): string[] {
  return Array.from(
    new Set(
      values
        .map((value) => normalizeOptionalText(value))
        .filter((value): value is string => Boolean(value)),
    ),
  );
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function extractStructuredContentLine(
  source: string,
  label: string,
): string | undefined {
  const match = source.match(
    new RegExp(`(?:^|\\n)\\s*${escapeRegExp(label)}：\\s*(.+?)(?=\\n|$)`),
  );
  return normalizeOptionalText(match?.[1]);
}

function buildStructuredContentLine(
  source: string,
  label: string,
): string | undefined {
  const value = extractStructuredContentLine(source, label);
  return value ? `${label}：${value}` : undefined;
}

function extractFirstStructuredContentLine(
  source: string,
  labels: readonly string[],
): string | undefined {
  for (const label of labels) {
    const value = extractStructuredContentLine(source, label);
    if (value) {
      return value;
    }
  }

  return undefined;
}

function getNormalizedMemoryContent(memory: MemoryReferenceRecord): string {
  return typeof memory.content === "string"
    ? memory.content.replace(/\r\n/g, "\n")
    : "";
}

function normalizeExperienceMemoryProjectGoal(
  memory: MemoryReferenceRecord,
): string | undefined {
  const content = getNormalizedMemoryContent(memory);
  const title = normalizeOptionalText(memory.title);
  const structuredGoal = extractFirstStructuredContentLine(
    content,
    EXPERIENCE_MEMORY_GOAL_LABELS,
  );
  if (structuredGoal) {
    return structuredGoal;
  }
  if (!title) {
    return undefined;
  }

  const compactTitle = normalizeOptionalText(title.replace(/结果闭环$/, ""));
  const strippedTitle = compactTitle?.split(/\s*[·•｜|]\s*/)[0] ?? compactTitle;
  return normalizeOptionalText(strippedTitle) || compactTitle;
}

function buildExperienceMemoryExistingResults(
  memory: MemoryReferenceRecord,
): string | undefined {
  const content = getNormalizedMemoryContent(memory);
  const structuredSummary = dedupeNonEmptyText([
    memory.summary,
    buildStructuredContentLine(content, "结果摘要"),
    buildStructuredContentLine(content, "当前交付"),
    buildStructuredContentLine(content, "建议下一步"),
    buildStructuredContentLine(content, "当前信号"),
    buildStructuredContentLine(content, "最近反馈"),
    buildStructuredContentLine(content, "运行态回流"),
    buildStructuredContentLine(content, "当前判断"),
  ]).join(" ");

  if (structuredSummary) {
    return truncateText(structuredSummary, 320);
  }

  const fallbackSummary = dedupeNonEmptyText([
    memory.summary,
    memory.content,
  ]).join(" ");
  return fallbackSummary ? truncateText(fallbackSummary, 320) : undefined;
}

function buildExperienceMemoryPlatformRegion(
  memory: MemoryReferenceRecord,
): string | undefined {
  const content = getNormalizedMemoryContent(memory);
  const platform = extractFirstStructuredContentLine(
    content,
    EXPERIENCE_MEMORY_PLATFORM_LABELS,
  );
  const region = extractFirstStructuredContentLine(
    content,
    EXPERIENCE_MEMORY_REGION_LABELS,
  );
  if (platform && region) {
    if (platform.includes(region)) {
      return platform;
    }

    return `${platform}（${region}）`;
  }

  return platform || region;
}

function buildExperienceMemoryAudience(
  memory: MemoryReferenceRecord,
): string | undefined {
  return extractFirstStructuredContentLine(
    getNormalizedMemoryContent(memory),
    EXPERIENCE_MEMORY_AUDIENCE_LABELS,
  );
}

function buildExperienceMemorySocialPostSubject(
  memory: MemoryReferenceRecord,
): string | undefined {
  const subject = dedupeNonEmptyText([
    normalizeExperienceMemoryProjectGoal(memory)
      ? `当前主题：${normalizeExperienceMemoryProjectGoal(memory)}`
      : null,
    buildExperienceMemoryExistingResults(memory)
      ? `当前结果基线：${buildExperienceMemoryExistingResults(memory)}`
      : null,
  ]).join("\n");

  return subject ? truncateText(subject, 600) : undefined;
}

function buildCuratedTaskPrefillByTaskIdFromMemory(
  memory: MemoryReferenceRecord,
): Record<string, CuratedTaskInputValues> | undefined {
  if (memory.category !== "experience") {
    return undefined;
  }

  const projectGoal = normalizeExperienceMemoryProjectGoal(memory);
  const existingResults = buildExperienceMemoryExistingResults(memory);
  const platformRegion = buildExperienceMemoryPlatformRegion(memory);
  const targetAudience = buildExperienceMemoryAudience(memory);
  const subjectOrProduct = buildExperienceMemorySocialPostSubject(memory);
  const reviewPrefill = normalizeCuratedTaskLaunchInputValues({
    project_goal: projectGoal || "",
    existing_results: existingResults || "",
  });
  const dailyTrendPrefill = normalizeCuratedTaskLaunchInputValues({
    theme_target: projectGoal || "",
    platform_region: platformRegion || "",
  });
  const socialPostPrefill = normalizeCuratedTaskLaunchInputValues({
    subject_or_product: subjectOrProduct || "",
    target_audience: targetAudience || "",
  });

  const prefillByTaskId: Record<string, CuratedTaskInputValues> = {};
  if (reviewPrefill) {
    prefillByTaskId[EXPERIENCE_MEMORY_REVIEW_TASK_ID] = reviewPrefill;
  }
  if (dailyTrendPrefill) {
    prefillByTaskId[EXPERIENCE_MEMORY_TREND_TASK_ID] = dailyTrendPrefill;
  }
  if (socialPostPrefill) {
    prefillByTaskId[EXPERIENCE_MEMORY_SOCIAL_POST_TASK_ID] = socialPostPrefill;
  }

  if (Object.keys(prefillByTaskId).length === 0) {
    return undefined;
  }

  return prefillByTaskId;
}

function compactRecord<T extends Record<string, unknown>>(record: T): T {
  return Object.fromEntries(
    Object.entries(record).filter(([, value]) => value !== undefined),
  ) as T;
}

function normalizeCuratedTaskReferenceEntry(
  entry?: CuratedTaskReferenceEntry | null,
): CuratedTaskReferenceEntry | null {
  if (!entry) {
    return null;
  }

  const id = normalizeOptionalText(entry.id);
  if (!id) {
    return null;
  }

  const title = normalizeUserFacingMemoryTitle({
    value: entry.title,
    category: entry.category,
  });
  const summary = normalizeUserFacingMemorySummary(entry.summary);
  const tags = normalizeUserFacingMemoryTags(entry.tags, 6);
  const sourceKind = "memory";
  const taskPrefillByTaskId = Object.fromEntries(
    Object.entries(entry.taskPrefillByTaskId ?? {})
      .map(
        ([taskId, inputValues]) =>
          [
            normalizeOptionalText(taskId),
            normalizeCuratedTaskLaunchInputValues(inputValues),
          ] as const,
      )
      .filter(
        (item): item is [string, CuratedTaskInputValues] =>
          Boolean(item[0]) && Boolean(item[1]),
      ),
  );

  return {
    id,
    sourceKind,
    title,
    summary: truncateText(summary, 120),
    category: entry.category,
    categoryLabel: getUserFacingMemoryCategoryLabel(entry.category),
    tags,
    ...(Object.keys(taskPrefillByTaskId).length > 0
      ? {
          taskPrefillByTaskId,
        }
      : {}),
  };
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }

  return value as Record<string, unknown>;
}

export function normalizeCuratedTaskLaunchInputValues(
  inputValues?: CuratedTaskInputValues | null,
): CuratedTaskInputValues | undefined {
  if (!inputValues) {
    return undefined;
  }

  const normalizedEntries = Object.entries(inputValues)
    .map(([key, value]) => [key, String(value ?? "").trim()] as const)
    .filter(([, value]) => value.length > 0);

  if (normalizedEntries.length === 0) {
    return undefined;
  }

  return Object.fromEntries(normalizedEntries);
}

export function normalizeCuratedTaskReferenceMemoryIds(
  values?: Array<string | null | undefined> | null,
): string[] | undefined {
  if (!Array.isArray(values)) {
    return undefined;
  }

  const normalized = Array.from(
    new Set(
      values
        .map((value) => normalizeOptionalText(value))
        .filter((value): value is string => Boolean(value)),
    ),
  );

  return normalized.length > 0 ? normalized : undefined;
}

export function getCuratedTaskReferenceSourceKind(
  _entry?: Pick<CuratedTaskReferenceEntry, "sourceKind"> | null,
): CuratedTaskReferenceSourceKind {
  return "memory";
}

export function getCuratedTaskReferenceSourceLabel(
  _entry?: Pick<CuratedTaskReferenceEntry, "sourceKind"> | null,
): string {
  return "记忆参考";
}

export function getCuratedTaskReferenceMemoryId(
  entry?: Pick<CuratedTaskReferenceEntry, "id" | "sourceKind"> | null,
): string | undefined {
  if (!entry) {
    return undefined;
  }

  return normalizeOptionalText(entry.id);
}

export function extractCuratedTaskReferenceMemoryIds(
  entries?: Array<CuratedTaskReferenceEntry | null | undefined> | null,
): string[] | undefined {
  return normalizeCuratedTaskReferenceMemoryIds(
    (entries ?? []).map((entry) => getCuratedTaskReferenceMemoryId(entry)),
  );
}

export function buildCuratedTaskLaunchInputPrefillFromReferenceEntries(params: {
  taskId: string;
  inputValues?: CuratedTaskInputValues | null;
  referenceEntries?: CuratedTaskReferenceEntry[] | null;
}): CuratedTaskInputValues | undefined {
  const normalizedInputValues = normalizeCuratedTaskLaunchInputValues(
    params.inputValues,
  );
  const mergedInputValues: CuratedTaskInputValues = {
    ...(normalizedInputValues ?? {}),
  };

  for (const entry of params.referenceEntries ?? []) {
    const taskPrefill = entry.taskPrefillByTaskId?.[params.taskId];
    if (!taskPrefill) {
      continue;
    }

    for (const [key, value] of Object.entries(taskPrefill)) {
      if (normalizeOptionalText(mergedInputValues[key])) {
        continue;
      }

      mergedInputValues[key] = value;
    }
  }

  return normalizeCuratedTaskLaunchInputValues(mergedInputValues);
}

export function mergeCuratedTaskReferenceEntries(
  entries: Array<CuratedTaskReferenceEntry | null | undefined>,
): CuratedTaskReferenceEntry[] {
  const result: CuratedTaskReferenceEntry[] = [];
  const seen = new Set<string>();

  for (const entry of entries) {
    const normalized = normalizeCuratedTaskReferenceEntry(entry);
    if (!normalized || seen.has(normalized.id)) {
      continue;
    }

    seen.add(normalized.id);
    result.push(normalized);
  }

  return result;
}

export function buildCuratedTaskReferenceEntries(
  memories: MemoryReferenceRecord[],
): CuratedTaskReferenceEntry[] {
  return mergeCuratedTaskReferenceEntries(
    memories.map((memory) => {
      const summary =
        normalizeOptionalText(memory.summary) ||
        normalizeOptionalText(memory.content) ||
        "等待补充摘要";

      return {
        id: memory.id,
        sourceKind: "memory",
        title:
          normalizeOptionalText(memory.title) ||
          getCuratedTaskReferenceFallbackTitle(memory.category),
        summary,
        category: memory.category,
        categoryLabel: getUserFacingMemoryCategoryLabel(memory.category),
        tags: memory.tags,
        taskPrefillByTaskId: buildCuratedTaskPrefillByTaskIdFromMemory(memory),
      };
    }),
  );
}

export function buildCuratedTaskReferenceEntryFromCreationReplay(
  creationReplay?: CreationReplayMetadata,
): CuratedTaskReferenceEntry | null {
  if (!creationReplay || creationReplay.kind !== "memory_entry") {
    return null;
  }

  return normalizeCuratedTaskReferenceEntry({
    id: creationReplay.source.entry_id || "",
    sourceKind: "memory",
    title:
      normalizeOptionalText(creationReplay.data.title) ||
      getCuratedTaskReferenceFallbackTitle(creationReplay.data.category),
    summary:
      normalizeOptionalText(creationReplay.data.summary) ||
      normalizeOptionalText(creationReplay.data.content_excerpt) ||
      "等待补充摘要",
    category: creationReplay.data.category,
    categoryLabel: getUserFacingMemoryCategoryLabel(
      creationReplay.data.category,
    ),
    tags: creationReplay.data.tags || [],
  });
}

export function buildCuratedTaskReferenceSelectionFromCreationReplay(
  creationReplay?: CreationReplayMetadata,
): CuratedTaskReferenceSelection {
  const referenceEntry =
    buildCuratedTaskReferenceEntryFromCreationReplay(creationReplay);

  if (!referenceEntry) {
    return {
      referenceMemoryIds: [],
      referenceEntries: [],
    };
  }

  return {
    referenceMemoryIds:
      extractCuratedTaskReferenceMemoryIds([referenceEntry]) ?? [],
    referenceEntries: [referenceEntry],
  };
}

function buildReferencePromptLine(entry: CuratedTaskReferenceEntry): string {
  const parts = [`[${entry.categoryLabel}] ${entry.title}`];

  if (entry.summary) {
    parts.push(`摘要：${entry.summary}`);
  }

  if (entry.tags.length > 0) {
    parts.push(`标签：${entry.tags.slice(0, 3).join("、")}`);
  }

  return `- ${parts.join("；")}`;
}

export function buildCuratedTaskReferencePromptBlock(
  entries?: CuratedTaskReferenceEntry[],
): string | null {
  if (!entries || entries.length === 0) {
    return null;
  }

  return `本轮可优先参考这些参考对象：\n${entries
    .slice(0, 3)
    .map((entry) => buildReferencePromptLine(entry))
    .join("\n")}`;
}

export function buildCuratedTaskLaunchRequestMetadata(params: {
  taskId: string;
  taskTitle: string;
  inputValues?: CuratedTaskInputValues | null;
  referenceMemoryIds?: string[];
  referenceEntries?: CuratedTaskReferenceEntry[];
  baseRequestMetadata?: Record<string, unknown>;
}): Record<string, unknown> {
  const normalizedReferenceEntries = mergeCuratedTaskReferenceEntries(
    params.referenceEntries ?? [],
  ).slice(0, 3);
  const inferredReferenceIds = normalizeCuratedTaskReferenceMemoryIds([
    ...(params.referenceMemoryIds ?? []),
    ...(extractCuratedTaskReferenceMemoryIds(normalizedReferenceEntries) ?? []),
  ]);
  const primaryReference = normalizedReferenceEntries.find(
    (entry) => getCuratedTaskReferenceSourceKind(entry) === "memory",
  );
  const primaryCreationReplay = primaryReference
    ? buildMemoryEntryCreationReplayRequestMetadata({
        id: getCuratedTaskReferenceMemoryId(primaryReference),
        category: primaryReference.category,
        title: primaryReference.title,
        summary: primaryReference.summary,
        content: primaryReference.summary,
        tags: primaryReference.tags,
      }).harness.creation_replay
    : undefined;
  const existingHarness = asRecord(params.baseRequestMetadata?.harness) || {};

  return {
    ...(params.baseRequestMetadata || {}),
    harness: {
      ...existingHarness,
      ...(primaryCreationReplay
        ? {
            creation_replay: primaryCreationReplay,
          }
        : {}),
      curated_task: compactRecord({
        task_id: normalizeOptionalText(params.taskId),
        task_title: normalizeOptionalText(params.taskTitle),
        launch_input_values: normalizeCuratedTaskLaunchInputValues(
          params.inputValues,
        ),
        reference_memory_ids: inferredReferenceIds,
        reference_entries:
          normalizedReferenceEntries.length > 0
            ? normalizedReferenceEntries.map((entry) =>
                compactRecord({
                  id: entry.id,
                  source_kind: getCuratedTaskReferenceSourceKind(entry),
                  title: normalizeOptionalText(entry.title),
                  summary: normalizeOptionalText(entry.summary),
                  category: entry.category,
                  tags: entry.tags.slice(0, 4),
                  task_prefill_by_task_id:
                    entry.taskPrefillByTaskId &&
                    Object.keys(entry.taskPrefillByTaskId).length > 0
                      ? Object.fromEntries(
                          Object.entries(entry.taskPrefillByTaskId).map(
                            ([taskId, inputValues]) => [
                              taskId,
                              normalizeCuratedTaskLaunchInputValues(
                                inputValues,
                              ),
                            ],
                          ),
                        )
                      : undefined,
                }),
              )
            : undefined,
      }),
    },
  };
}
