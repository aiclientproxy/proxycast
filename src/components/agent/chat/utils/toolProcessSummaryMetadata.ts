import { resolveRequiredAgentChatCopy } from "./agentChatCopy";
import type {
  ToolProcessNarrativeSource,
  ToolProcessStatus,
} from "./toolProcessSummaryTypes";
import type { ToolOperationKind } from "./toolBatchGroupingTypes";
import type { ToolCallFamily } from "./toolDisplayTypes";
import { asRecord, readString } from "./toolProcessSummaryText";

export interface ResolvedToolProcessSummaryMetadata {
  preSummary: string | null;
  postSummary: string | null;
  postSource: ToolProcessNarrativeSource;
}

const SUMMARY_CONTAINER_KEYS = [
  "tool_process_summary",
  "toolProcessSummary",
  "process_summary",
  "processSummary",
] as const;

const ALLOWED_KEY_PREFIXES = [
  "toolCall.processSummary.",
  "toolCall.siteResult.",
] as const;

const FACTS_CONTAINER_KEYS = [
  "tool_process_facts",
  "toolProcessFacts",
] as const;

const TOOL_PROCESS_FACT_FAMILIES = new Set<ToolCallFamily>([
  "subagent",
  "task",
  "plan",
  "skill",
  "write",
  "read",
  "edit",
  "command",
  "search",
  "list",
  "browser",
  "fetch",
  "vision",
  "generic",
]);

const TOOL_PROCESS_OPERATION_KINDS = new Set<ToolOperationKind>([
  "read",
  "search",
  "web_search",
  "web_fetch",
  "list",
  "browser",
  "mutation",
  "absorbed",
  "other",
]);

const TERMINAL_INTERACTION_SUMMARY_RE = /^sent [0-9]+ chars$/u;

export function resolveTerminalInteractionSummaries(
  metadata: unknown,
): string[] {
  const interactions = asRecord(metadata)?.terminal_interactions;
  if (!Array.isArray(interactions)) {
    return [];
  }
  return interactions.flatMap((value) => {
    const stdin = asRecord(value)?.stdin;
    if (
      typeof stdin !== "string" ||
      (stdin !== "(poll)" &&
        stdin !== "(interrupt)" &&
        !TERMINAL_INTERACTION_SUMMARY_RE.test(stdin))
    ) {
      return [];
    }
    return [stdin];
  });
}

function readSummaryContainer(
  metadata: unknown,
): Record<string, unknown> | null {
  const record = asRecord(metadata);
  if (!record) {
    return null;
  }

  for (const key of SUMMARY_CONTAINER_KEYS) {
    const value = asRecord(record[key]);
    if (value) {
      return value;
    }
  }

  return null;
}

function readFactsContainer(metadata: unknown): Record<string, unknown> | null {
  const record = asRecord(metadata);
  if (!record) {
    return null;
  }

  for (const key of FACTS_CONTAINER_KEYS) {
    const value = asRecord(record[key]);
    if (value) {
      return value;
    }
  }

  return null;
}

export function resolveToolProcessFactsSubject(
  metadata: unknown,
): string | null {
  return readString(readFactsContainer(metadata), [
    "subject",
    "target",
    "query",
  ]);
}

export function resolveToolProcessFactsFamily(
  metadata: unknown,
): ToolCallFamily | null {
  const value = readString(readFactsContainer(metadata), [
    "toolFamily",
    "tool_family",
    "family",
  ]);
  if (!value) {
    return null;
  }
  const normalized = normalizeToolFamily(value);
  return TOOL_PROCESS_FACT_FAMILIES.has(normalized) ? normalized : null;
}

export function resolveToolProcessFactsOperationKind(
  metadata: unknown,
): ToolOperationKind | null {
  const value = readString(readFactsContainer(metadata), [
    "operationKind",
    "operation_kind",
    "operation",
  ]);
  if (!value) {
    return null;
  }
  const normalized = normalizeOperationKind(value);
  return TOOL_PROCESS_OPERATION_KINDS.has(normalized) ? normalized : null;
}

function normalizeToolFamily(value: string): ToolCallFamily {
  const normalized = value.trim();
  if (normalized === "webSearch") {
    return "search";
  }
  if (normalized === "webFetch" || normalized === "file") {
    return normalized === "webFetch" ? "fetch" : "read";
  }
  return normalized
    .replace(/[_\s-]+([a-z])/g, (_, character: string) =>
      character.toUpperCase(),
    )
    .replace(/^[A-Z]/, (character) =>
      character.toLowerCase(),
    ) as ToolCallFamily;
}

function normalizeOperationKind(value: string): ToolOperationKind {
  return value.trim().replace(/[\s-]+/g, "_") as ToolOperationKind;
}

function isAllowedSummaryKey(key: string): boolean {
  return ALLOWED_KEY_PREFIXES.some((prefix) => key.startsWith(prefix));
}

function normalizeValues(value: unknown): Record<string, unknown> {
  const record = asRecord(value);
  if (!record) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(record).filter(
      (entry): entry is [string, string | number | boolean] => {
        const nextValue = entry[1];
        return (
          typeof nextValue === "string" ||
          typeof nextValue === "number" ||
          typeof nextValue === "boolean"
        );
      },
    ),
  );
}

function resolveDescriptor(value: unknown): string | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }

  const key = typeof record.key === "string" ? record.key.trim() : "";
  if (!key || !isAllowedSummaryKey(key)) {
    return null;
  }

  return resolveRequiredAgentChatCopy(key, normalizeValues(record.values));
}

function pickDescriptor(
  container: Record<string, unknown>,
  keys: readonly string[],
): string | null {
  for (const key of keys) {
    const resolved = resolveDescriptor(container[key]);
    if (resolved) {
      return resolved;
    }
  }

  return null;
}

function statusPostKeys(status: ToolProcessStatus): string[] {
  if (status === "failed") {
    return ["failed", "failure", "postSummary", "post", "after", "summary"];
  }
  if (status === "completed") {
    return ["completed", "success", "postSummary", "post", "after", "summary"];
  }
  return ["postSummary", "post", "after"];
}

export function resolveToolProcessSummaryMetadata(
  metadata: unknown,
  status: ToolProcessStatus,
): ResolvedToolProcessSummaryMetadata | null {
  const container = readSummaryContainer(metadata);
  if (!container) {
    return null;
  }

  const isRunning = status === "running" || status === "in_progress";
  const preSummary = pickDescriptor(
    container,
    [
      isRunning ? "running" : "",
      "preSummary",
      "pre",
      "before",
      isRunning ? "summary" : "",
    ].filter(Boolean),
  );
  const postSummary = pickDescriptor(container, statusPostKeys(status));

  if (!preSummary && !postSummary) {
    return null;
  }

  return {
    preSummary,
    postSummary,
    postSource: postSummary ? "metadata" : "none",
  };
}
