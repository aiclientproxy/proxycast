import { agentZhCNResource as agentSourceResource } from "@/i18n/agentResources";
import type { InputCapabilitySendRoute } from "../skill-selection/inputCapabilitySelection";
import {
  buildCuratedTaskLaunchInputPrefillFromReferenceEntries,
  extractCuratedTaskReferenceMemoryIds,
  mergeCuratedTaskReferenceEntries,
  type CuratedTaskReferenceEntry,
} from "./curatedTaskReferenceSelection";
import {
  buildCuratedTaskLaunchPrompt,
  findCuratedTaskTemplateById,
  type CuratedTaskInputValues,
} from "./curatedTaskTemplates";

type AgentSourceResourceKey = keyof typeof agentSourceResource;

function interpolateSourceTemplate(
  template: string,
  values?: Record<string, number | string>,
): string {
  return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (match, name) => {
    const value = values?.[name];
    return value == null ? match : String(value);
  });
}

function translateSourceKey(
  key: AgentSourceResourceKey,
  values?: Record<string, number | string>,
): string {
  return interpolateSourceTemplate(agentSourceResource[key], values);
}

function normalizeOptionalText(value?: string | null): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized || undefined;
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

function extractMarkedSection(
  source: string,
  label: string,
  nextLabels: string[],
): string | undefined {
  const lookahead = nextLabels
    .map((item) => `${escapeRegExp(item)}：`)
    .join("|");
  const pattern =
    lookahead.length > 0
      ? `${escapeRegExp(label)}：\\s*(.+?)(?=\\s*(?:${lookahead})|$)`
      : `${escapeRegExp(label)}：\\s*(.+)$`;
  const match = source.match(new RegExp(pattern));
  return normalizeOptionalText(match?.[1]);
}

export interface CuratedTaskResultBaselineSnapshot {
  sourceTitle: string;
  projectGoal?: string;
  existingResults?: string;
  statusLabel?: string;
  failureSignalLabel?: string;
  nextAction?: string;
  operatingAction?: string;
  destinationsLabel?: string;
}

export interface CuratedTaskResultBaselineCopy {
  formatDestinationHighlight?: (value: string) => string;
  formatFailureSignalHighlight?: (value: string) => string;
  formatFollowUpBannerMessage?: (title: string) => string;
  formatOperatingActionHighlight?: (value: string) => string;
  formatStatusHighlight?: (value: string) => string;
}

interface ResolvedCuratedTaskResultBaselineCopy {
  formatDestinationHighlight: (value: string) => string;
  formatFailureSignalHighlight: (value: string) => string;
  formatFollowUpBannerMessage: (title: string) => string;
  formatOperatingActionHighlight: (value: string) => string;
  formatStatusHighlight: (value: string) => string;
}

const SOURCE_RESULT_BASELINE_COPY: ResolvedCuratedTaskResultBaselineCopy = {
  formatDestinationHighlight: (value) =>
    translateSourceKey("curatedTask.resultReference.highlight.destination", {
      value,
    }),
  formatFailureSignalHighlight: (value) =>
    translateSourceKey("curatedTask.resultReference.highlight.failureSignal", {
      value,
    }),
  formatFollowUpBannerMessage: (title) =>
    translateSourceKey("curatedTask.resultReference.followUpBanner", { title }),
  formatOperatingActionHighlight: (value) =>
    translateSourceKey(
      "curatedTask.resultReference.highlight.operatingAction",
      { value },
    ),
  formatStatusHighlight: (value) =>
    translateSourceKey("curatedTask.resultReference.highlight.status", {
      value,
    }),
};

function resolveResultBaselineCopy(
  copy?: CuratedTaskResultBaselineCopy,
): ResolvedCuratedTaskResultBaselineCopy {
  return {
    ...SOURCE_RESULT_BASELINE_COPY,
    ...(copy ?? {}),
  };
}

const REVIEW_BASELINE_FALLBACK_TASK_ID = "account-project-review";

function hasBaselinePrefillFields(
  prefill?: CuratedTaskInputValues | null,
): boolean {
  return Boolean(
    normalizeOptionalText(prefill?.project_goal) ||
    normalizeOptionalText(prefill?.existing_results),
  );
}

export function buildCuratedTaskResultBaselineSnapshot(params: {
  referenceEntries?: Array<CuratedTaskReferenceEntry | null | undefined> | null;
  taskId?: string | null;
}): CuratedTaskResultBaselineSnapshot | null {
  const taskId =
    normalizeOptionalText(params.taskId) || REVIEW_BASELINE_FALLBACK_TASK_ID;
  const candidateTaskIds = Array.from(
    new Set([taskId, REVIEW_BASELINE_FALLBACK_TASK_ID]),
  );
  const referenceEntry = mergeCuratedTaskReferenceEntries(
    params.referenceEntries ?? [],
  ).find((entry) =>
    candidateTaskIds.some((candidateTaskId) =>
      Boolean(entry.taskPrefillByTaskId?.[candidateTaskId]),
    ),
  );

  if (!referenceEntry) {
    return null;
  }

  const matchedTaskId = candidateTaskIds.find((candidateTaskId) =>
    hasBaselinePrefillFields(
      referenceEntry.taskPrefillByTaskId?.[candidateTaskId],
    ),
  );
  const prefill = matchedTaskId
    ? referenceEntry.taskPrefillByTaskId?.[matchedTaskId]
    : candidateTaskIds
        .map(
          (candidateTaskId) =>
            referenceEntry.taskPrefillByTaskId?.[candidateTaskId],
        )
        .find(Boolean);
  const existingResults = normalizeOptionalText(prefill?.existing_results);
  const normalizedExistingResults = existingResults || "";

  return {
    sourceTitle: referenceEntry.title,
    projectGoal: normalizeOptionalText(prefill?.project_goal),
    existingResults,
    statusLabel: extractMarkedSection(normalizedExistingResults, "当前判断", [
      "经营动作",
      "更适合去向",
      "当前卡点",
      "当前信号",
      "建议下一步",
    ]),
    failureSignalLabel:
      extractMarkedSection(normalizedExistingResults, "当前卡点", [
        "建议下一步",
        "当前判断",
        "经营动作",
        "更适合去向",
        "当前信号",
      ]) ||
      extractMarkedSection(normalizedExistingResults, "当前信号", [
        "建议下一步",
        "当前判断",
        "经营动作",
        "更适合去向",
      ]),
    nextAction: extractMarkedSection(normalizedExistingResults, "建议下一步", [
      "当前判断",
      "经营动作",
      "更适合去向",
    ]),
    operatingAction: extractMarkedSection(
      normalizedExistingResults,
      "经营动作",
      ["更适合去向"],
    ),
    destinationsLabel: extractMarkedSection(
      normalizedExistingResults,
      "更适合去向",
      [],
    ),
  };
}

export function buildCuratedTaskResultBaselineHighlights(
  snapshot?: CuratedTaskResultBaselineSnapshot | null,
  copyInput?: CuratedTaskResultBaselineCopy,
): string[] {
  if (!snapshot) {
    return [];
  }

  const copy = resolveResultBaselineCopy(copyInput);
  return dedupeNonEmptyText([
    snapshot.statusLabel
      ? copy.formatStatusHighlight(snapshot.statusLabel)
      : null,
    snapshot.failureSignalLabel
      ? copy.formatFailureSignalHighlight(snapshot.failureSignalLabel)
      : null,
    snapshot.operatingAction
      ? copy.formatOperatingActionHighlight(snapshot.operatingAction)
      : null,
    snapshot.destinationsLabel
      ? copy.formatDestinationHighlight(snapshot.destinationsLabel)
      : null,
  ]);
}

function buildResultBaselinePromptBlock(
  snapshot?: CuratedTaskResultBaselineSnapshot | null,
): string | null {
  if (!snapshot) {
    return null;
  }

  const lines = dedupeNonEmptyText([
    snapshot.sourceTitle ? `当前结果基线：${snapshot.sourceTitle}` : null,
    snapshot.projectGoal ? `当前项目目标：${snapshot.projectGoal}` : null,
    snapshot.existingResults
      ? `当前已有结果：${snapshot.existingResults}`
      : null,
  ]);
  return lines.length > 0
    ? `继续沿这轮项目结果基线推进：\n${lines
        .map((line) => `- ${line}`)
        .join("\n")}`
    : null;
}

export function buildCuratedTaskFollowUpActionFromReferences(params: {
  referenceEntries?: Array<CuratedTaskReferenceEntry | null | undefined> | null;
  taskId: string;
  inputValues?: CuratedTaskInputValues | null;
  copy?: CuratedTaskResultBaselineCopy;
}): {
  prompt: string;
  bannerMessage?: string;
  capabilityRoute: Extract<InputCapabilitySendRoute, { kind: "curated_task" }>;
} | null {
  const taskId = normalizeOptionalText(params.taskId);
  const task = taskId ? findCuratedTaskTemplateById(taskId) : null;
  if (!task) {
    return null;
  }

  const referenceEntries = mergeCuratedTaskReferenceEntries(
    params.referenceEntries ?? [],
  ).slice(0, 3);
  if (referenceEntries.length === 0) {
    return null;
  }

  const inputValues = buildCuratedTaskLaunchInputPrefillFromReferenceEntries({
    taskId: task.id,
    inputValues: params.inputValues,
    referenceEntries,
  });
  const baselinePromptBlock =
    task.id === REVIEW_BASELINE_FALLBACK_TASK_ID
      ? null
      : buildResultBaselinePromptBlock(
          buildCuratedTaskResultBaselineSnapshot({
            referenceEntries,
            taskId: task.id,
          }),
        );
  const prompt = [
    buildCuratedTaskLaunchPrompt({
      task,
      inputValues: inputValues ?? {},
      referenceEntries,
    }).trim(),
    baselinePromptBlock,
  ]
    .filter((section): section is string => Boolean(section))
    .join("\n\n")
    .trim();
  if (!prompt) {
    return null;
  }

  const referenceMemoryIds =
    extractCuratedTaskReferenceMemoryIds(referenceEntries) ?? [];
  const copy = resolveResultBaselineCopy(params.copy);

  return {
    prompt,
    bannerMessage: copy.formatFollowUpBannerMessage(task.title),
    capabilityRoute: {
      kind: "curated_task",
      taskId: task.id,
      taskTitle: task.title,
      prompt,
      ...(inputValues ? { launchInputValues: inputValues } : {}),
      ...(referenceMemoryIds.length > 0 ? { referenceMemoryIds } : {}),
      referenceEntries,
    },
  };
}
