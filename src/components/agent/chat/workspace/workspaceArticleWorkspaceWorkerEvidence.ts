import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";

export type WorkspaceArticleWorkspaceWorkerEvidenceStatus =
  | "completed"
  | "failed"
  | "skipped"
  | "unknown";

export interface WorkspaceArticleWorkspaceWorkerEvidenceItem {
  id: string;
  status: WorkspaceArticleWorkspaceWorkerEvidenceStatus;
  source: string;
  eventType: string | null;
  appId: string | null;
  taskId: string | null;
  taskKind: string | null;
  turnId: string | null;
  workerEntrypoint: string | null;
  inputSummary: string | null;
  outputSummary: string | null;
  outputObjectCount: number | null;
  artifactRef: string | null;
  artifactKind: string | null;
  errorCode: string | null;
  errorMessage: string | null;
  failureCategory: string | null;
  retryable: boolean | null;
  retryAdvice: string | null;
  retryAttempt: number | null;
  retryMaxAttempts: number | null;
  hookKey: string | null;
  hookEvent: string | null;
  hookScope: string | null;
  hookEntrypoint: string | null;
  hookRequired: boolean | null;
  reasonCode: string | null;
  resultSummary: string | null;
  workflowKey: string | null;
  subagents: string[];
  skillRefs: string[];
  cliRefs: string[];
  connectorRefs: string[];
  hookRefs: string[];
  hookPolicy: WorkspaceArticleWorkspaceWorkerEvidenceHookPolicy | null;
  runtimeRegistries: Record<string, unknown> | null;
  orchestration: WorkspaceArticleWorkspaceWorkerEvidenceOrchestrationStep[];
  updatedAt: string | null;
}

export type WorkspaceArticleWorkspaceWorkerEvidenceHookPolicy = Record<
  string,
  string[]
>;

export interface WorkspaceArticleWorkspaceWorkerEvidenceOrchestrationStep {
  id: string;
  title: string;
  subagent: string | null;
  skillRefs: string[];
  status: string | null;
  summary: string | null;
  expectedOutput: string | null;
}

export function readWorkspaceArticleWorkspaceWorkerEvidence(
  value: unknown,
): WorkspaceArticleWorkspaceWorkerEvidenceItem[] {
  return readArray(value)
    .map(readWorkerEvidenceItem)
    .filter((item): item is WorkspaceArticleWorkspaceWorkerEvidenceItem =>
      Boolean(item),
    )
    .sort(compareWorkerEvidenceDesc);
}

export function buildWorkspaceArticleWorkspaceWorkerEvidenceFromThreadRead(params: {
  articleWorkspace?: Record<string, unknown> | null;
  sourceArtifacts?: unknown;
  threadRead?: AgentRuntimeThreadReadModel | null;
}): WorkspaceArticleWorkspaceWorkerEvidenceItem[] {
  const articleWorkspace = params.articleWorkspace ?? null;
  const explicitEvidence = readWorkspaceArticleWorkspaceWorkerEvidence(
    firstArray(
      articleWorkspace?.workerEvidence,
      articleWorkspace?.worker_evidence,
    ),
  );
  if (explicitEvidence.length > 0) {
    return explicitEvidence;
  }

  const fallbackEvidence = readSourceArtifactWorkerEvidence(
    params.sourceArtifacts,
  );
  const diagnosticEvidence = readDiagnosticWorkerEvidence(params.threadRead);
  return [...diagnosticEvidence, ...fallbackEvidence].sort(
    compareWorkerEvidenceDesc,
  );
}

function readWorkerEvidenceItem(
  value: unknown,
): WorkspaceArticleWorkspaceWorkerEvidenceItem | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  const id = readString(record.id, record.eventId, record.event_id);
  const status = readWorkerEvidenceStatus(record.status);
  return {
    id:
      id ||
      [
        readString(record.turnId, record.turn_id),
        readString(record.taskId, record.task_id),
        status,
      ]
        .filter(Boolean)
        .join(":") ||
      "worker-evidence",
    status,
    source:
      readString(record.source, record.eventType, record.event_type) ||
      "workspace_patch",
    eventType: readString(record.eventType, record.event_type) || null,
    appId: readString(record.appId, record.app_id) || null,
    taskId: readString(record.taskId, record.task_id) || null,
    taskKind: readString(record.taskKind, record.task_kind) || null,
    turnId: readString(record.turnId, record.turn_id) || null,
    workerEntrypoint:
      readString(record.workerEntrypoint, record.worker_entrypoint) || null,
    inputSummary: readString(record.inputSummary, record.input_summary) || null,
    outputSummary:
      readString(record.outputSummary, record.output_summary) || null,
    outputObjectCount:
      readNumber(record.outputObjectCount, record.output_object_count) ?? null,
    artifactRef:
      readString(record.artifactRef, record.artifact_ref, record.path) || null,
    artifactKind:
      readString(record.artifactKind, record.artifact_kind, record.kind) ||
      null,
    errorCode: readString(record.errorCode, record.error_code) || null,
    errorMessage:
      readString(
        record.errorMessage,
        record.error_message,
        record.message,
        record.error,
      ) || null,
    failureCategory:
      readString(record.failureCategory, record.failure_category) || null,
    retryable: readBoolean(record.retryable) ?? null,
    retryAdvice: readString(record.retryAdvice, record.retry_advice) || null,
    retryAttempt: readNumber(record.retryAttempt, record.retry_attempt) ?? null,
    retryMaxAttempts:
      readNumber(record.retryMaxAttempts, record.retry_max_attempts) ?? null,
    hookKey: readString(record.hookKey, record.hook_key) || null,
    hookEvent: readString(record.hookEvent, record.hook_event) || null,
    hookScope: readString(record.hookScope, record.hook_scope) || null,
    hookEntrypoint:
      readString(record.hookEntrypoint, record.hook_entrypoint) || null,
    hookRequired:
      readBoolean(record.hookRequired, record.hook_required) ?? null,
    reasonCode: readString(record.reasonCode, record.reason_code) || null,
    resultSummary:
      readString(record.resultSummary, record.result_summary) || null,
    workflowKey: readString(record.workflowKey, record.workflow_key) || null,
    subagents: readStringList(record.subagents, record.sub_agents),
    skillRefs: readStringList(record.skillRefs, record.skill_refs),
    cliRefs: readStringList(record.cliRefs, record.cli_refs),
    connectorRefs: readStringList(record.connectorRefs, record.connector_refs),
    hookRefs: readStringList(record.hookRefs, record.hook_refs),
    hookPolicy:
      readWorkerEvidenceHookPolicy(record.hookPolicy, record.hook_policy) ??
      null,
    runtimeRegistries:
      firstRecord(record.runtimeRegistries, record.runtime_registries) ?? null,
    orchestration: readWorkerEvidenceOrchestration(
      record.orchestration,
      record.workflowSteps,
      record.workflow_steps,
    ),
    updatedAt: readString(record.updatedAt, record.updated_at) || null,
  };
}

function readSourceArtifactWorkerEvidence(
  value: unknown,
): WorkspaceArticleWorkspaceWorkerEvidenceItem[] {
  return readArray(value)
    .map((item, index): WorkspaceArticleWorkspaceWorkerEvidenceItem | null => {
      const record = asRecord(item);
      if (!record) {
        return null;
      }
      const artifactRef = readString(
        record.artifactRef,
        record.artifact_ref,
        record.path,
      );
      const artifactKind = readString(
        record.kind,
        record.artifactKind,
        record.artifact_kind,
      );
      if (!artifactRef && !artifactKind) {
        return null;
      }
      const id =
        readString(record.eventId, record.event_id, artifactRef) ||
        `source-artifact-${index + 1}`;
      return {
        id,
        status: "completed" as const,
        source: "artifact.snapshot",
        eventType: "artifact.snapshot",
        appId: null,
        taskId: null,
        taskKind: null,
        turnId: readString(record.turnId, record.turn_id) || null,
        workerEntrypoint: null,
        inputSummary: null,
        outputSummary: null,
        outputObjectCount: null,
        artifactRef: artifactRef || null,
        artifactKind: artifactKind || null,
        errorCode: null,
        errorMessage: null,
        failureCategory: null,
        retryable: null,
        retryAdvice: null,
        retryAttempt: null,
        retryMaxAttempts: null,
        hookKey: null,
        hookEvent: null,
        hookScope: null,
        hookEntrypoint: null,
        hookRequired: null,
        reasonCode: null,
        resultSummary: null,
        workflowKey: null,
        subagents: [],
        skillRefs: [],
        cliRefs: [],
        connectorRefs: [],
        hookRefs: [],
        hookPolicy: null,
        runtimeRegistries: null,
        orchestration: [],
        updatedAt: readString(record.updatedAt, record.updated_at) || null,
      };
    })
    .filter((item): item is WorkspaceArticleWorkspaceWorkerEvidenceItem =>
      Boolean(item),
    );
}

function readDiagnosticWorkerEvidence(
  threadRead?: AgentRuntimeThreadReadModel | null,
): WorkspaceArticleWorkspaceWorkerEvidenceItem[] {
  const runtimeSummary = asRecord(threadRead?.runtime_summary);
  const message = readString(
    threadRead?.diagnostics?.latest_turn_error_message,
    runtimeSummary?.latestTurnErrorMessage,
    runtimeSummary?.latest_turn_error_message,
  );
  if (!message) {
    return [];
  }
  const turns = threadRead?.turns ?? [];
  const turnId =
    readString(threadRead?.active_turn_id) ||
    readString(turns.length > 0 ? turns[turns.length - 1]?.turn_id : null);
  return [
    {
      id: `${turnId || threadRead?.thread_id || "thread"}:worker-error`,
      status: "failed",
      source: "diagnostics",
      eventType: "diagnostics",
      appId: null,
      taskId: null,
      taskKind: readString(threadRead?.task_kind) || null,
      turnId: turnId || null,
      workerEntrypoint: null,
      inputSummary: null,
      outputSummary: null,
      outputObjectCount: null,
      artifactRef: null,
      artifactKind: null,
      errorCode: null,
      errorMessage: message,
      failureCategory: null,
      retryable: null,
      retryAdvice: null,
      retryAttempt: null,
      retryMaxAttempts: null,
      hookKey: null,
      hookEvent: null,
      hookScope: null,
      hookEntrypoint: null,
      hookRequired: null,
      reasonCode: null,
      resultSummary: null,
      workflowKey: null,
      subagents: [],
      skillRefs: [],
      cliRefs: [],
      connectorRefs: [],
      hookRefs: [],
      hookPolicy: null,
      runtimeRegistries: null,
      orchestration: [],
      updatedAt:
        readString(
          threadRead?.diagnostics?.latest_turn_completed_at,
          threadRead?.diagnostics?.latest_turn_updated_at,
          threadRead?.updated_at,
        ) || null,
    },
  ];
}

function readWorkerEvidenceStatus(
  value: unknown,
): WorkspaceArticleWorkspaceWorkerEvidenceStatus {
  const status = readString(value).replace(/-/g, "_");
  if (status === "completed" || status === "failed" || status === "skipped") {
    return status;
  }
  return "unknown";
}

function compareWorkerEvidenceDesc(
  left: WorkspaceArticleWorkspaceWorkerEvidenceItem,
  right: WorkspaceArticleWorkspaceWorkerEvidenceItem,
): number {
  return (
    dateValue(right.updatedAt) - dateValue(left.updatedAt) ||
    right.id.localeCompare(left.id)
  );
}

function dateValue(value: string | null): number {
  if (!value) {
    return 0;
  }
  const time = Date.parse(value);
  return Number.isFinite(time) ? time : 0;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function firstArray(...values: unknown[]): unknown[] {
  for (const value of values) {
    if (Array.isArray(value)) {
      return value;
    }
  }
  return [];
}

function readString(...values: unknown[]): string {
  for (const value of values) {
    if (typeof value !== "string") {
      continue;
    }
    const normalized = value.trim();
    if (normalized) {
      return normalized;
    }
  }
  return "";
}

function readNumber(...values: unknown[]): number | undefined {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
}

function readBoolean(...values: unknown[]): boolean | undefined {
  for (const value of values) {
    if (typeof value === "boolean") {
      return value;
    }
  }
  return undefined;
}

function readArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function firstRecord(...values: unknown[]): Record<string, unknown> | null {
  for (const value of values) {
    const record = asRecord(value);
    if (record) {
      return record;
    }
  }
  return null;
}

function readStringList(...values: unknown[]): string[] {
  const items: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    for (const item of readArray(value)) {
      const text = readString(
        item,
        asRecord(item)?.id,
        asRecord(item)?.key,
        asRecord(item)?.name,
        asRecord(item)?.title,
      );
      if (!text || seen.has(text)) {
        continue;
      }
      seen.add(text);
      items.push(text);
    }
  }
  return items;
}

function readWorkerEvidenceHookPolicy(
  ...values: unknown[]
): WorkspaceArticleWorkspaceWorkerEvidenceHookPolicy | undefined {
  const policy = firstRecord(...values);
  if (!policy) {
    return undefined;
  }
  const normalized: WorkspaceArticleWorkspaceWorkerEvidenceHookPolicy = {};
  for (const [key, value] of Object.entries(policy)) {
    const hooks = readStringList(value);
    if (hooks.length > 0) {
      normalized[key] = hooks;
    }
  }
  return Object.keys(normalized).length > 0 ? normalized : undefined;
}

function readWorkerEvidenceOrchestration(
  ...values: unknown[]
): WorkspaceArticleWorkspaceWorkerEvidenceOrchestrationStep[] {
  return values.flatMap((value) =>
    readArray(value)
      .map(readWorkerEvidenceOrchestrationStep)
      .filter(
        (
          item,
        ): item is WorkspaceArticleWorkspaceWorkerEvidenceOrchestrationStep =>
          Boolean(item),
      ),
  );
}

function readWorkerEvidenceOrchestrationStep(
  value: unknown,
): WorkspaceArticleWorkspaceWorkerEvidenceOrchestrationStep | null {
  const record = asRecord(value);
  if (!record) {
    return null;
  }
  const id = readString(record.id, record.key);
  if (!id) {
    return null;
  }
  return {
    id,
    title: readString(record.title, record.name) || id,
    subagent:
      readString(record.subagent, record.sub_agent, record.owner) || null,
    skillRefs: readStringList(record.skillRefs, record.skill_refs),
    status: readString(record.status) || null,
    summary: readString(record.summary, record.description) || null,
    expectedOutput:
      readString(
        record.expectedOutput,
        record.expected_output,
        record.output,
      ) || null,
  };
}
