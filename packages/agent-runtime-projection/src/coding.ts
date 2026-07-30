import type {
  AgentRuntimeEventProjection,
  AgentRuntimeExecutionEvent,
  AgentRuntimeExecutionEventStatus,
  AgentRuntimeProjectionInput,
  AgentUiArtifactRefView,
  AgentUiDiagnosticView,
  AgentUiEvidenceRefView,
  AgentUiProjectionState,
  AgentUiRuntimeStatusView,
} from "@limecloud/agent-ui-contracts";
import {
  codingProcessFactsFromInput,
  mergeCodingProcessFacts,
  type CodingProcessFactInput,
} from "./codingProcessFacts.js";
import { projectAgentUiState } from "./uiState.js";

export interface CodingReadModelCommandFact extends CodingProcessFactInput {
  command_id?: string;
  commandId?: string;
  status?: string;
  command?: string;
  canonical_command?: string;
  canonicalCommand?: string;
  command_summary?: string;
  commandSummary?: string;
  command_argv?: string[];
  commandArgv?: string[];
  command_argv_source?: string;
  commandArgvSource?: string;
  cwd?: string;
  exit_code?: number;
  exitCode?: number;
  output_refs?: string[];
  outputRefs?: string[];
  output_preview?: string | null;
  outputPreview?: string | null;
  source_event_ids?: string[];
  sourceEventIds?: string[];
}

export interface CodingReadModelTestRunFact {
  test_run_id?: string;
  testRunId?: string;
  status?: string;
  command_id?: string;
  commandId?: string;
  suite?: string;
  canonical_command?: string;
  canonicalCommand?: string;
  command_summary?: string;
  commandSummary?: string;
  result?: string;
  passed?: number;
  failed?: number;
  output_refs?: string[];
  outputRefs?: string[];
  failure_category?: string;
  failureCategory?: string;
  source_event_ids?: string[];
  sourceEventIds?: string[];
}

export interface CodingReadModelPendingRequestFact {
  id?: string;
  turn_id?: string;
  turnId?: string;
  status?: string;
  request_type?: string;
  requestType?: string;
  title?: string;
  payload?: unknown;
}

export interface CodingReadModelArtifactFact {
  artifact_ref?: string;
  artifactRef?: string;
  event_id?: string;
  eventId?: string;
  sequence?: number;
  turn_id?: string;
  turnId?: string;
  artifact_id?: string;
  artifactId?: string;
  path?: string;
  title?: string;
  kind?: string;
  status?: string;
  content?: string;
  metadata?: unknown;
}

export interface CodingReadModelChangeSummaryFact {
  changed_file_count?: number;
  changedFileCount?: number;
  changed_files?: string[];
  changedFiles?: string[];
  patch_count?: number;
  patchCount?: number;
  applied_patch_count?: number;
  appliedPatchCount?: number;
  failed_patch_count?: number;
  failedPatchCount?: number;
  running_patch_count?: number;
  runningPatchCount?: number;
  source_event_ids?: string[];
  sourceEventIds?: string[];
  latest_sequence?: number;
  latestSequence?: number;
}

export interface CodingReadModelFacts {
  thread_id?: string;
  threadId?: string;
  active_turn_id?: string;
  activeTurnId?: string;
  commands?: CodingReadModelCommandFact[];
  tests?: CodingReadModelTestRunFact[];
  pending_requests?: CodingReadModelPendingRequestFact[];
  pendingRequests?: CodingReadModelPendingRequestFact[];
  active_command_id?: string | null;
  activeCommandId?: string | null;
  active_test_run_id?: string | null;
  activeTestRunId?: string | null;
  active_action_id?: string | null;
  activeActionId?: string | null;
  change_summary?: CodingReadModelChangeSummaryFact | null;
  changeSummary?: CodingReadModelChangeSummaryFact | null;
  artifacts?: unknown[];
  thread_items?: readonly unknown[];
  threadItems?: readonly unknown[];
}

export interface CodingWorkbenchProjectionInput<
  TEvent extends AgentRuntimeExecutionEvent = AgentRuntimeExecutionEvent,
> extends AgentRuntimeProjectionInput<TEvent> {
  codingReadModel?: CodingReadModelFacts | null;
  threadItems?: readonly unknown[];
}

export type CodingWorkbenchStatus =
  | "idle"
  | "running"
  | "blocked"
  | "completed"
  | "failed"
  | "degraded";

export interface CodingMainObjectView {
  id: string;
  title: string;
  status: CodingWorkbenchStatus;
  activeFilePath?: string;
  activePatchId?: string;
  activeCommandId?: string;
  activeTestRunId?: string;
  sourceEventId?: string;
}

export interface CodingFileView {
  id: string;
  path: string;
  status: string;
  title: string;
  artifactRefs: string[];
  checkpointRef?: string;
  contentRef?: string;
  latestEventId: string;
}

export interface FileChangeView {
  id: string;
  path: string;
  destinationPath?: string;
  status: string;
  changeKind?: string;
  artifactRefs: string[];
  checkpointRef?: string;
  diffRef?: string;
  preview?: string;
  sourceEventId: string;
}

export interface PatchView {
  patchId: string;
  status: string;
  title: string;
  path?: string;
  toolCallId?: string;
  diffRef?: string;
  failureCategory?: string;
  recoveryHintRef?: string;
  sourceEventIds: string[];
}

export interface CommandOutputView extends CodingProcessFactInput {
  commandId: string;
  status: string;
  title: string;
  command?: string;
  canonicalCommand?: string;
  commandSummary?: string;
  commandArgv?: string[];
  commandArgvSource?: string;
  cwd?: string;
  exitCode?: number;
  outputRefs: string[];
  preview?: string;
  sourceEventIds: string[];
}

export interface TestRunView {
  testRunId: string;
  status: string;
  title: string;
  commandId?: string;
  command?: string;
  canonicalCommand?: string;
  commandSummary?: string;
  suite?: string;
  result?: string;
  passed?: number;
  failed?: number;
  outputRefs: string[];
  failureCategory?: string;
  sourceEventIds: string[];
}

export interface ChangeSummaryView {
  changedFileCount: number;
  changedFiles: string[];
  patchCount: number;
  appliedPatchCount: number;
  failedPatchCount: number;
  runningPatchCount: number;
  sourceEventIds: string[];
}

export interface CodingLocalUiState {
  preferredTab: "preview" | "files" | "changes" | "outputs" | "logs";
  stale: boolean;
}

export interface CodingWorkbenchView<
  TEvent extends AgentRuntimeExecutionEvent = AgentRuntimeExecutionEvent,
> {
  runtime: AgentUiRuntimeStatusView;
  mainObject: CodingMainObjectView;
  files: CodingFileView[];
  changes: FileChangeView[];
  patches: PatchView[];
  changeSummary?: ChangeSummaryView;
  commands: CommandOutputView[];
  tests: TestRunView[];
  actions: AgentRuntimeEventProjection<TEvent>[];
  artifacts: AgentUiArtifactRefView[];
  evidence: AgentUiEvidenceRefView[];
  diagnostics: AgentUiDiagnosticView[];
  ui: CodingLocalUiState;
}

interface PatchAccumulator extends PatchView {
  sequence: number;
}

interface CommandAccumulator extends CommandOutputView {
  sequence: number;
}

interface TestAccumulator extends TestRunView {
  sequence: number;
}

interface ArtifactChangeAccumulator extends FileChangeView {
  sequence: number;
  fileId: string;
}

function isRecord(input: unknown): input is Record<string, unknown> {
  return typeof input === "object" && input !== null && !Array.isArray(input);
}

function payload(event: AgentRuntimeExecutionEvent): Record<string, unknown> {
  return isRecord(event.payload) ? event.payload : {};
}

function compact<T extends Record<string, unknown>>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter(([, field]) => field !== undefined),
  ) as T;
}

function payloadString(
  event: AgentRuntimeExecutionEvent,
  ...keys: string[]
): string | undefined {
  const eventPayload = payload(event);
  for (const key of keys) {
    const value = eventPayload[key];
    if (typeof value === "string" && value.trim()) {
      return value;
    }
  }
  return undefined;
}

function payloadNumber(
  event: AgentRuntimeExecutionEvent,
  key: string,
): number | undefined {
  const value = payload(event)[key];
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function valueString(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return undefined;
}

function valueNumber(...values: unknown[]): number | undefined {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
}

function valueStringArray(...values: unknown[]): string[] {
  for (const value of values) {
    if (Array.isArray(value)) {
      return value.filter(
        (item): item is string =>
          typeof item === "string" && Boolean(item.trim()),
      );
    }
  }
  return [];
}

function valueRecord(
  ...values: unknown[]
): Record<string, unknown> | undefined {
  for (const value of values) {
    if (isRecord(value)) return value;
  }
  return undefined;
}

function eventSequence(event: AgentRuntimeExecutionEvent): number {
  return typeof event.sequence === "number" ? event.sequence : 0;
}

function safeRelativePath(value: string | undefined): string | undefined {
  if (!value) return undefined;
  if (/^(?:[A-Za-z]:[\\/]|\/|file:)/.test(value)) return undefined;
  if (/(?:api[-_]?key|authorization|password|secret|token)=/i.test(value)) {
    return undefined;
  }
  return value;
}

function safePreview(value: string | undefined): string | undefined {
  if (!value) return undefined;
  return value.length > 320 ? `${value.slice(0, 317)}...` : value;
}

function safeCommand(value: string | undefined): string | undefined {
  if (!value) return undefined;
  if (
    /(?:api[-_]?key|authorization|password|secret|token)[=\s:]/i.test(value)
  ) {
    return undefined;
  }
  return safePreview(value);
}

function safeArtifactKind(value: string | undefined): string | undefined {
  return value?.trim().toLowerCase();
}

function refIds(event: AgentRuntimeExecutionEvent): string[] {
  return [
    ...(event.refIds ?? []),
    ...(event.artifactRefs ?? []),
    ...(event.evidenceRefs ?? []),
  ];
}

function outputRefs(event: AgentRuntimeExecutionEvent): string[] {
  return [
    ...refIds(event),
    payloadString(event, "outputRef"),
    payloadString(event, "contentRef"),
    payloadString(event, "diffRef"),
  ].filter((value): value is string => Boolean(value));
}

function eventScopeId(
  event: AgentRuntimeExecutionEvent,
  payloadKey: string,
): string | undefined {
  return payloadString(event, payloadKey) ?? event.toolCallId;
}

function isCodingEvent(event: AgentRuntimeExecutionEvent): boolean {
  const eventClass = event.eventClass ?? "";
  return (
    eventClass.startsWith("file.") ||
    eventClass.startsWith("patch.") ||
    eventClass.startsWith("command.") ||
    eventClass.startsWith("test.") ||
    eventClass === "sandbox.blocked" ||
    payloadString(event, "profileId") === "coding"
  );
}

function threadItemRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function threadItemStatus(
  item: Record<string, unknown>,
  exitCode?: number,
): AgentRuntimeExecutionEventStatus {
  if (typeof exitCode === "number" && exitCode !== 0) return "failed";
  const status = valueString(item.status)?.toLowerCase();
  if (status === "failed" || status === "error") return "failed";
  if (status === "in_progress" || status === "running") return "running";
  if (status === "canceled" || status === "cancelled") return "canceled";
  return "completed";
}

function threadItemTimestamp(item: Record<string, unknown>): string {
  return (
    valueString(
      item.updated_at,
      item.updatedAt,
      item.started_at,
      item.startedAt,
    ) ?? new Date(0).toISOString()
  );
}

function testSuiteFromCommand(command: string): string | undefined {
  const separator = command.split(/\s+--\s+/u)[1]?.trim();
  return separator || undefined;
}

function executionEventsFromThreadItems(
  threadItems: readonly unknown[] | undefined,
): AgentRuntimeExecutionEvent[] {
  if (!threadItems?.length) return [];
  const latestFileItems = new Map<
    string,
    {
      item: Record<string, unknown>;
      change?: Record<string, unknown>;
      path: string;
    }
  >();
  const commandItems: Record<string, unknown>[] = [];

  threadItems.forEach((value) => {
    const item = threadItemRecord(value);
    if (!item) return;
    const type = valueString(item.type);
    if (type === "command_execution" || type === "commandExecution") {
      commandItems.push(item);
      return;
    }
    const changes =
      (type === "fileChange" || type === "patch") && Array.isArray(item.changes)
        ? item.changes.filter(isRecord)
        : [];
    const fileEntries =
      changes.length > 0
        ? changes.flatMap((change) => {
            const path = valueString(change.path);
            return path ? [{ item, change, path }] : [];
          })
        : type === "patch"
          ? valueStringArray(item.paths, item.path).map((path) => ({
              item,
              path,
            }))
          : [];
    fileEntries.forEach((entry) => {
      const previous = latestFileItems.get(entry.path);
      const sequence = valueNumber(item.sequence) ?? 0;
      if (!previous || sequence >= (valueNumber(previous.item.sequence) ?? 0)) {
        latestFileItems.set(entry.path, entry);
      }
    });
  });

  const events: AgentRuntimeExecutionEvent[] = [];
  latestFileItems.forEach(({ item, change, path }) => {
    const id = valueString(item.id) ?? `thread-item:file:${path}`;
    const createdAt = threadItemTimestamp(item);
    const kind = valueRecord(change?.kind);
    const destinationPath = safeRelativePath(
      valueString(kind?.move_path, kind?.movePath),
    );
    events.push({
      id: `thread-item:${id}:file:${path}`,
      kind: "source",
      status: threadItemStatus(item),
      eventClass: "file.changed",
      title: path,
      sequence: valueNumber(item.sequence),
      threadId: valueString(item.thread_id, item.threadId),
      turnId: valueString(item.turn_id, item.turnId),
      artifactRefs: valueStringArray(item.artifactRefs, item.artifact_refs),
      createdAt,
      completedAt: valueString(item.completed_at, item.completedAt),
      payload: {
        path,
        destinationPath,
        changeKind: destinationPath
          ? "renamed"
          : (valueString(kind?.type) ?? "modified"),
        preview: valueString(change?.diff, item.text, item.summary),
      },
    });
  });

  commandItems.forEach((item) => {
    const id = valueString(item.id) ?? "thread-item:command";
    const command = valueString(item.command) ?? id;
    const exitCode = valueNumber(item.exit_code, item.exitCode);
    const status = threadItemStatus(item, exitCode);
    const createdAt = threadItemTimestamp(item);
    const base = {
      kind: "tool" as const,
      status,
      threadId: valueString(item.thread_id, item.threadId),
      turnId: valueString(item.turn_id, item.turnId),
      sequence: valueNumber(item.sequence),
      createdAt,
      completedAt: valueString(item.completed_at, item.completedAt),
    };
    events.push({
      ...base,
      id: `thread-item:${id}:command`,
      eventClass: "command.exited",
      title: command,
      payload: {
        commandId: id,
        command,
        cwd: valueString(item.cwd),
        preview: valueString(
          item.aggregated_output,
          item.aggregatedOutput,
          item.output,
        ),
        exitCode,
      },
    });

    const suite = testSuiteFromCommand(command);
    if (suite) {
      events.push({
        ...base,
        id: `thread-item:${id}:test`,
        eventClass: "test.completed",
        title: suite,
        sequence: (valueNumber(item.sequence) ?? 0) + 0.001,
        payload: {
          testRunId: `${id}:test`,
          commandId: id,
          suite,
          result: status === "failed" ? "failed" : "passed",
          passed: status === "failed" ? 0 : 1,
          failed: status === "failed" ? 1 : 0,
          outputRef: valueString(item.outputRef, item.output_ref),
        },
      });
    }
  });

  return events.sort(
    (left, right) => eventSequence(left) - eventSequence(right),
  );
}

function projectionThreadItems(
  input?: CodingWorkbenchProjectionInput,
): readonly unknown[] {
  const itemsById = new Map<string, unknown>();
  const anonymousItems: unknown[] = [];
  const readModelItems =
    input?.codingReadModel?.thread_items ??
    input?.codingReadModel?.threadItems ??
    [];
  [...readModelItems, ...(input?.threadItems ?? [])].forEach((item) => {
    const id = isRecord(item) ? valueString(item.id) : undefined;
    if (id) {
      itemsById.set(id, item);
    } else {
      anonymousItems.push(item);
    }
  });
  return [...itemsById.values(), ...anonymousItems];
}

function isTurnLifecycleEvent(event: AgentRuntimeExecutionEvent): boolean {
  return (event.eventClass ?? "").startsWith("turn.");
}

function eventMatchesAnyCodingScope(
  event: AgentRuntimeExecutionEvent,
  codingEvents: readonly AgentRuntimeExecutionEvent[],
): boolean {
  return codingEvents.some(
    (codingEvent) =>
      (event.turnId && codingEvent.turnId === event.turnId) ||
      (event.taskId && codingEvent.taskId === event.taskId),
  );
}

function collectCodingEvents(
  events: readonly AgentRuntimeExecutionEvent[],
): AgentRuntimeExecutionEvent[] {
  const directCodingEvents = events.filter(isCodingEvent);
  if (directCodingEvents.length === 0) {
    return [];
  }
  return events.filter(
    (event) =>
      isCodingEvent(event) ||
      (isTurnLifecycleEvent(event) &&
        eventMatchesAnyCodingScope(event, directCodingEvents)),
  );
}

function statusFromRuntime(
  state: AgentUiProjectionState,
  codingEvents: AgentRuntimeExecutionEvent[],
): CodingWorkbenchStatus {
  const latestTurnTerminal = [...codingEvents]
    .reverse()
    .find((event) => event.eventClass?.startsWith("turn."));
  if (latestTurnTerminal?.eventClass === "turn.completed") {
    return "completed";
  }
  if (latestTurnTerminal?.eventClass === "turn.failed") {
    return "failed";
  }
  if (latestTurnTerminal?.eventClass === "turn.canceled") {
    return "degraded";
  }
  if (codingEvents.some((event) => event.status === "blocked")) {
    return "blocked";
  }
  if (codingEvents.some((event) => event.status === "failed")) {
    return "failed";
  }
  if (
    state.runtime.status === "running" ||
    state.runtime.status === "waiting"
  ) {
    return "running";
  }
  if (state.runtime.status === "completed") {
    return "completed";
  }
  if (!codingEvents.length) {
    return "idle";
  }
  return "degraded";
}

function buildMainObject(
  state: AgentUiProjectionState,
  codingEvents: AgentRuntimeExecutionEvent[],
  files: CodingFileView[],
  patches: PatchView[],
  commands: CommandOutputView[],
  tests: TestRunView[],
  readModel?: CodingReadModelFacts | null,
): CodingMainObjectView {
  const latest = codingEvents.at(-1);
  const activePatch = [...patches]
    .reverse()
    .find((patch) => patch.status === "running");
  return {
    id:
      latest?.turnId ??
      state.runtime.activeTurnId ??
      valueString(readModel?.active_turn_id, readModel?.activeTurnId) ??
      valueString(readModel?.thread_id, readModel?.threadId) ??
      "coding-workbench",
    title: latest?.title ?? "Coding Workbench",
    status: statusFromRuntime(state, codingEvents),
    activeFilePath: files.at(-1)?.path,
    activePatchId: activePatch?.patchId,
    activeCommandId: activeCommandId(commands, readModel),
    activeTestRunId: activeTestRunId(tests, readModel),
    sourceEventId: latest?.id,
  };
}

function collectFiles(
  events: readonly AgentRuntimeExecutionEvent[],
): CodingFileView[] {
  const files = new Map<string, CodingFileView>();
  events.forEach((event) => {
    if (
      event.eventClass !== "file.read" &&
      event.eventClass !== "file.changed"
    ) {
      return;
    }
    const path = safeRelativePath(payloadString(event, "path", "relativePath"));
    if (!path) return;
    files.set(path, {
      id: path,
      path,
      status: event.status,
      title: event.title,
      artifactRefs: event.artifactRefs ?? [],
      checkpointRef: payloadString(event, "checkpointRef", "checkpointId"),
      contentRef: payloadString(event, "contentRef"),
      latestEventId: event.id,
    });
  });
  return Array.from(files.values());
}

function collectChanges(
  events: readonly AgentRuntimeExecutionEvent[],
): FileChangeView[] {
  return events.flatMap((event) => {
    if (event.eventClass !== "file.changed") return [];
    const path = safeRelativePath(payloadString(event, "path", "relativePath"));
    if (!path) return [];
    return [
      {
        id: event.id,
        path,
        destinationPath: safeRelativePath(
          payloadString(event, "destinationPath", "movePath"),
        ),
        status: event.status,
        changeKind: payloadString(event, "changeKind", "operation"),
        artifactRefs: event.artifactRefs ?? [],
        checkpointRef: payloadString(event, "checkpointRef", "checkpointId"),
        diffRef: payloadString(event, "diffRef"),
        preview: safePreview(payloadString(event, "preview", "summary")),
        sourceEventId: event.id,
      },
    ];
  });
}

function artifactCheckpointRef(
  artifact: CodingReadModelArtifactFact,
  metadata?: Record<string, unknown>,
): string | undefined {
  const artifactVersion = valueRecord(
    metadata?.artifactVersion,
    metadata?.artifact_version,
  );
  return valueString(
    metadata?.checkpointRef,
    metadata?.checkpoint_ref,
    metadata?.checkpointId,
    metadata?.checkpoint_id,
    metadata?.artifactVersionId,
    metadata?.artifact_version_id,
    artifactVersion?.id,
    artifactVersion?.checkpointRef,
    artifactVersion?.checkpoint_ref,
    artifact.artifact_ref,
    artifact.artifactRef,
  );
}

function artifactDiffRef(
  metadata?: Record<string, unknown>,
): string | undefined {
  const artifactVersion = valueRecord(
    metadata?.artifactVersion,
    metadata?.artifact_version,
  );
  const diff = valueRecord(
    metadata?.artifactVersionDiff,
    metadata?.artifact_version_diff,
  );
  return valueString(
    metadata?.diffRef,
    metadata?.diff_ref,
    artifactVersion?.diffRef,
    artifactVersion?.diff_ref,
    diff?.diffRef,
    diff?.diff_ref,
  );
}

function artifactPreview(
  artifact: CodingReadModelArtifactFact,
  metadata?: Record<string, unknown>,
): string | undefined {
  const artifactVersion = valueRecord(
    metadata?.artifactVersion,
    metadata?.artifact_version,
  );
  return safePreview(
    valueString(
      metadata?.previewText,
      metadata?.preview_text,
      metadata?.artifactSummary,
      metadata?.artifact_summary,
      metadata?.summary,
      artifactVersion?.title,
      artifact.title,
      artifact.path,
    ),
  );
}

function artifactChangeKind(
  metadata?: Record<string, unknown>,
): string | undefined {
  const change = valueRecord(metadata?.file_change, metadata?.fileChange);
  return valueString(
    metadata?.changeKind,
    metadata?.change_kind,
    metadata?.operation,
    change?.changeKind,
    change?.change_kind,
    change?.operation,
  );
}

function isCodingArtifact(artifact: CodingReadModelArtifactFact): boolean {
  const metadata = valueRecord(artifact.metadata);
  const kind = safeArtifactKind(
    valueString(
      artifact.kind,
      metadata?.artifactKind,
      metadata?.artifact_kind,
      metadata?.kind,
    ),
  );
  const path = safeRelativePath(valueString(artifact.path));
  if (!path) return false;
  if (!kind) return true;
  return (
    kind.includes("file") ||
    kind.includes("code") ||
    kind.includes("patch") ||
    kind.includes("diff")
  );
}

function readModelArtifact(value: unknown): CodingReadModelArtifactFact | null {
  return isRecord(value) ? (value as CodingReadModelArtifactFact) : null;
}

function changeFromReadModelArtifact(
  artifact: CodingReadModelArtifactFact,
): ArtifactChangeAccumulator | null {
  if (!isCodingArtifact(artifact)) return null;
  const path = safeRelativePath(valueString(artifact.path));
  if (!path) return null;
  const metadata = valueRecord(artifact.metadata);
  const artifactRef = valueString(artifact.artifact_ref, artifact.artifactRef);
  const sourceEventId =
    valueString(artifact.event_id, artifact.eventId) ??
    (artifactRef
      ? `read-model:artifact:${artifactRef}`
      : `read-model:path:${path}`);
  const checkpointRef = artifactCheckpointRef(artifact, metadata);
  return compact({
    id: sourceEventId,
    path,
    status: valueString(artifact.status) ?? "completed",
    changeKind: artifactChangeKind(metadata) ?? "modified",
    artifactRefs: artifactRef ? [artifactRef] : [],
    checkpointRef,
    diffRef: artifactDiffRef(metadata),
    preview: artifactPreview(artifact, metadata),
    sourceEventId,
    sequence: valueNumber(artifact.sequence) ?? 0,
    fileId: path.toLowerCase(),
  });
}

function mergeChanges(
  eventChanges: FileChangeView[],
  readModel?: CodingReadModelFacts | null,
): FileChangeView[] {
  const byFile = new Map<string, ArtifactChangeAccumulator>();
  eventChanges.forEach((change, index) => {
    byFile.set(change.path.toLowerCase(), {
      ...change,
      sequence: index + 1,
      fileId: change.path.toLowerCase(),
    });
  });
  readModel?.artifacts
    ?.map(readModelArtifact)
    .filter((artifact): artifact is CodingReadModelArtifactFact =>
      Boolean(artifact),
    )
    .map(changeFromReadModelArtifact)
    .filter((change): change is ArtifactChangeAccumulator => Boolean(change))
    .forEach((change) => {
      const existing = byFile.get(change.fileId);
      byFile.set(
        change.fileId,
        existing
          ? {
              ...change,
              ...existing,
              artifactRefs: [
                ...new Set([...change.artifactRefs, ...existing.artifactRefs]),
              ],
            }
          : change,
      );
    });
  return Array.from(byFile.values())
    .sort((left, right) => left.sequence - right.sequence)
    .map(({ sequence: _sequence, fileId: _fileId, ...change }) => change);
}

function filesFromChanges(
  changes: readonly FileChangeView[],
): CodingFileView[] {
  const files = new Map<string, CodingFileView>();
  changes.forEach((change) => {
    files.set(change.path, {
      id: change.path,
      path: change.path,
      status: change.status,
      title: change.path,
      artifactRefs: change.artifactRefs,
      checkpointRef: change.checkpointRef,
      contentRef: undefined,
      latestEventId: change.sourceEventId,
    });
  });
  return Array.from(files.values());
}

function fileFromReadModelArtifact(
  artifact: CodingReadModelArtifactFact,
): CodingFileView | null {
  if (!isCodingArtifact(artifact)) return null;
  const path = safeRelativePath(valueString(artifact.path));
  if (!path) return null;
  const metadata = valueRecord(artifact.metadata);
  const artifactRef = valueString(artifact.artifact_ref, artifact.artifactRef);
  const sourceEventId =
    valueString(artifact.event_id, artifact.eventId) ??
    (artifactRef
      ? `read-model:artifact:${artifactRef}`
      : `read-model:path:${path}`);
  return compact({
    id: artifactRef ?? path,
    path,
    status: valueString(artifact.status) ?? "completed",
    title: safeCommand(valueString(artifact.title)) ?? artifact.title ?? path,
    artifactRefs: artifactRef ? [artifactRef] : [],
    checkpointRef: artifactCheckpointRef(artifact, metadata),
    contentRef: valueString(
      metadata?.contentRef,
      metadata?.content_ref,
      metadata?.snapshotFile,
      metadata?.snapshot_file,
    ),
    latestEventId: sourceEventId,
  });
}

function mergeFiles(
  eventFiles: CodingFileView[],
  changes: readonly FileChangeView[],
  readModel?: CodingReadModelFacts | null,
): CodingFileView[] {
  const byPath = new Map<string, CodingFileView>();
  eventFiles.forEach((file) => {
    byPath.set(file.path.toLowerCase(), file);
  });
  filesFromChanges(changes).forEach((file) => {
    const key = file.path.toLowerCase();
    const existing = byPath.get(key);
    byPath.set(
      key,
      existing
        ? {
            ...existing,
            ...file,
            artifactRefs: [
              ...new Set([...existing.artifactRefs, ...file.artifactRefs]),
            ],
          }
        : file,
    );
  });
  readModel?.artifacts
    ?.map(readModelArtifact)
    .filter((artifact): artifact is CodingReadModelArtifactFact =>
      Boolean(artifact),
    )
    .map(fileFromReadModelArtifact)
    .filter((file): file is CodingFileView => Boolean(file))
    .forEach((file) => {
      const key = file.path.toLowerCase();
      const existing = byPath.get(key);
      byPath.set(
        key,
        existing
          ? {
              ...existing,
              ...file,
              artifactRefs: [
                ...new Set([...existing.artifactRefs, ...file.artifactRefs]),
              ],
            }
          : file,
      );
    });
  return Array.from(byPath.values());
}

function collectPatches(
  events: readonly AgentRuntimeExecutionEvent[],
): PatchView[] {
  const patches = new Map<string, PatchAccumulator>();
  events.forEach((event) => {
    if (!event.eventClass?.startsWith("patch.")) return;
    const patchId = eventScopeId(event, "patchId");
    if (!patchId) return;
    const previous = patches.get(patchId);
    patches.set(patchId, {
      patchId,
      status: event.status,
      title: event.title,
      path:
        safeRelativePath(payloadString(event, "path", "relativePath")) ??
        previous?.path,
      toolCallId: event.toolCallId ?? previous?.toolCallId,
      diffRef: payloadString(event, "diffRef") ?? previous?.diffRef,
      failureCategory:
        payloadString(event, "failureCategory") ?? previous?.failureCategory,
      recoveryHintRef:
        payloadString(event, "recoveryHintRef") ?? previous?.recoveryHintRef,
      sourceEventIds: [...(previous?.sourceEventIds ?? []), event.id],
      sequence: eventSequence(event),
    });
  });
  return Array.from(patches.values())
    .sort((left, right) => left.sequence - right.sequence)
    .map(({ sequence: _sequence, ...patch }) => patch);
}

function collectCommands(
  events: readonly AgentRuntimeExecutionEvent[],
): CommandOutputView[] {
  const commands = new Map<string, CommandAccumulator>();
  events.forEach((event) => {
    if (!event.eventClass?.startsWith("command.")) return;
    const commandId = eventScopeId(event, "commandId");
    if (!commandId) return;
    const previous = commands.get(commandId);
    commands.set(commandId, {
      commandId,
      status: event.status,
      title: event.title,
      command:
        safeCommand(payloadString(event, "command")) ?? previous?.command,
      canonicalCommand:
        safeCommand(
          payloadString(event, "canonicalCommand", "canonical_command"),
        ) ?? previous?.canonicalCommand,
      commandSummary:
        safeCommand(
          payloadString(event, "commandSummary", "command_summary"),
        ) ?? previous?.commandSummary,
      commandArgv: [
        ...new Set([
          ...(previous?.commandArgv ?? []),
          ...valueStringArray(
            payload(event).commandArgv,
            payload(event).command_argv,
          ),
        ]),
      ],
      commandArgvSource:
        payloadString(event, "commandArgvSource", "command_argv_source") ??
        previous?.commandArgvSource,
      cwd:
        safeRelativePath(payloadString(event, "cwd", "workingDirectory")) ??
        previous?.cwd,
      exitCode: payloadNumber(event, "exitCode") ?? previous?.exitCode,
      ...mergeCodingProcessFacts(
        previous ?? {},
        codingProcessFactsFromInput(payload(event)),
      ),
      outputRefs: [
        ...new Set([...(previous?.outputRefs ?? []), ...outputRefs(event)]),
      ],
      preview:
        safePreview(payloadString(event, "preview", "summary")) ??
        previous?.preview,
      sourceEventIds: [...(previous?.sourceEventIds ?? []), event.id],
      sequence: eventSequence(event),
    });
  });
  return Array.from(commands.values())
    .sort((left, right) => left.sequence - right.sequence)
    .map(({ sequence: _sequence, ...command }) => command);
}

function collectTests(
  events: readonly AgentRuntimeExecutionEvent[],
): TestRunView[] {
  const tests = new Map<string, TestAccumulator>();
  events.forEach((event) => {
    if (!event.eventClass?.startsWith("test.")) return;
    const testRunId = eventScopeId(event, "testRunId");
    if (!testRunId) return;
    const previous = tests.get(testRunId);
    tests.set(testRunId, {
      testRunId,
      status: event.status,
      title: event.title,
      commandId: payloadString(event, "commandId") ?? previous?.commandId,
      command:
        safeCommand(payloadString(event, "command")) ?? previous?.command,
      canonicalCommand:
        safeCommand(
          payloadString(event, "canonicalCommand", "canonical_command"),
        ) ?? previous?.canonicalCommand,
      commandSummary:
        safeCommand(
          payloadString(event, "commandSummary", "command_summary"),
        ) ?? previous?.commandSummary,
      suite: payloadString(event, "suite") ?? previous?.suite,
      result: payloadString(event, "result", "status") ?? previous?.result,
      passed: payloadNumber(event, "passed") ?? previous?.passed,
      failed: payloadNumber(event, "failed") ?? previous?.failed,
      outputRefs: [
        ...new Set([...(previous?.outputRefs ?? []), ...outputRefs(event)]),
      ],
      failureCategory:
        payloadString(event, "failureCategory") ?? previous?.failureCategory,
      sourceEventIds: [...(previous?.sourceEventIds ?? []), event.id],
      sequence: eventSequence(event),
    });
  });
  return Array.from(tests.values())
    .sort((left, right) => left.sequence - right.sequence)
    .map(({ sequence: _sequence, ...test }) => test);
}

function commandFromReadModel(
  command: CodingReadModelCommandFact,
): CommandOutputView | null {
  const commandId = valueString(command.command_id, command.commandId);
  if (!commandId) return null;
  const sourceEventIds = valueStringArray(
    command.source_event_ids,
    command.sourceEventIds,
  );
  return compact({
    commandId,
    status: valueString(command.status) ?? "running",
    title:
      safeCommand(
        valueString(command.command_summary, command.commandSummary),
      ) ??
      safeCommand(valueString(command.command)) ??
      commandId,
    command: safeCommand(valueString(command.command)),
    canonicalCommand: safeCommand(
      valueString(command.canonical_command, command.canonicalCommand),
    ),
    commandSummary: safeCommand(
      valueString(command.command_summary, command.commandSummary),
    ),
    commandArgv: valueStringArray(command.command_argv, command.commandArgv),
    commandArgvSource: valueString(
      command.command_argv_source,
      command.commandArgvSource,
    ),
    cwd: safeRelativePath(valueString(command.cwd)),
    exitCode: valueNumber(command.exit_code, command.exitCode),
    ...codingProcessFactsFromInput(command),
    outputRefs: valueStringArray(command.output_refs, command.outputRefs),
    preview: safePreview(
      valueString(command.output_preview, command.outputPreview),
    ),
    sourceEventIds: sourceEventIds.length
      ? sourceEventIds
      : [`read-model:command:${commandId}`],
  });
}

function testFromReadModel(
  test: CodingReadModelTestRunFact,
): TestRunView | null {
  const testRunId = valueString(test.test_run_id, test.testRunId);
  if (!testRunId) return null;
  const sourceEventIds = valueStringArray(
    test.source_event_ids,
    test.sourceEventIds,
  );
  return compact({
    testRunId,
    status: valueString(test.status) ?? "running",
    title:
      safeCommand(valueString(test.command_summary, test.commandSummary)) ??
      valueString(test.suite, test.command_id, test.commandId) ??
      testRunId,
    commandId: valueString(test.command_id, test.commandId),
    canonicalCommand: safeCommand(
      valueString(test.canonical_command, test.canonicalCommand),
    ),
    commandSummary: safeCommand(
      valueString(test.command_summary, test.commandSummary),
    ),
    suite: valueString(test.suite),
    result: valueString(test.result),
    passed: valueNumber(test.passed),
    failed: valueNumber(test.failed),
    outputRefs: valueStringArray(test.output_refs, test.outputRefs),
    failureCategory: valueString(test.failure_category, test.failureCategory),
    sourceEventIds: sourceEventIds.length
      ? sourceEventIds
      : [`read-model:test:${testRunId}`],
  });
}

function changeSummaryFromReadModel(
  readModel?: CodingReadModelFacts | null,
): ChangeSummaryView | undefined {
  const summary = readModel?.change_summary ?? readModel?.changeSummary;
  if (!summary) return undefined;
  const changedFiles = valueStringArray(
    summary.changed_files,
    summary.changedFiles,
  )
    .map((path) => safeRelativePath(path))
    .filter((path): path is string => Boolean(path));
  const patchCount = valueNumber(summary.patch_count, summary.patchCount) ?? 0;
  const changedFileCount =
    valueNumber(summary.changed_file_count, summary.changedFileCount) ??
    changedFiles.length;
  if (changedFileCount === 0 && patchCount === 0) return undefined;
  return {
    changedFileCount,
    changedFiles,
    patchCount,
    appliedPatchCount:
      valueNumber(summary.applied_patch_count, summary.appliedPatchCount) ?? 0,
    failedPatchCount:
      valueNumber(summary.failed_patch_count, summary.failedPatchCount) ?? 0,
    runningPatchCount:
      valueNumber(summary.running_patch_count, summary.runningPatchCount) ?? 0,
    sourceEventIds: valueStringArray(
      summary.source_event_ids,
      summary.sourceEventIds,
    ),
  };
}

function actionEventFromReadModel(
  request: CodingReadModelPendingRequestFact,
  readModel: CodingReadModelFacts,
): AgentRuntimeExecutionEvent | null {
  const actionId = valueString(request.id);
  if (!actionId) return null;
  const requestStatus = valueString(request.status);
  const requestType = valueString(request.request_type, request.requestType);
  return compact({
    id: `read-model:action:${actionId}`,
    schemaVersion: "lime-runtime-event/v0.1",
    runtimeId: "thread-read-model",
    threadId: valueString(readModel.thread_id, readModel.threadId),
    turnId:
      valueString(request.turn_id, request.turnId) ??
      valueString(readModel.active_turn_id, readModel.activeTurnId),
    actionId,
    eventClass: "action.required",
    kind: "action",
    status: requestStatus === "resolved" ? "completed" : "blocked",
    title: valueString(request.title, requestType) ?? "Action required",
    payload: {
      actionKind: requestType,
      targetModule: "coding-workbench",
      request: request.payload,
    },
    createdAt: new Date(0).toISOString(),
  }) as AgentRuntimeExecutionEvent;
}

function mergeCommands(
  eventCommands: CommandOutputView[],
  readModel?: CodingReadModelFacts | null,
): CommandOutputView[] {
  const byId = new Map(
    eventCommands.map((command) => [command.commandId, command]),
  );
  readModel?.commands
    ?.map(commandFromReadModel)
    .filter((command): command is CommandOutputView => Boolean(command))
    .forEach((command) => {
      const existing = byId.get(command.commandId);
      byId.set(
        command.commandId,
        existing ? { ...command, ...existing } : command,
      );
    });
  return Array.from(byId.values());
}

function mergeTests(
  eventTests: TestRunView[],
  readModel?: CodingReadModelFacts | null,
): TestRunView[] {
  const byId = new Map(eventTests.map((test) => [test.testRunId, test]));
  readModel?.tests
    ?.map(testFromReadModel)
    .filter((test): test is TestRunView => Boolean(test))
    .forEach((test) => {
      const existing = byId.get(test.testRunId);
      byId.set(test.testRunId, existing ? { ...test, ...existing } : test);
    });
  return Array.from(byId.values());
}

function mergeActions<TEvent extends AgentRuntimeExecutionEvent>(
  actions: AgentRuntimeEventProjection<TEvent>[],
  readModel?: CodingReadModelFacts | null,
): AgentRuntimeEventProjection<TEvent>[] {
  const requests =
    readModel?.pending_requests ?? readModel?.pendingRequests ?? [];
  if (!requests.length) return actions;
  const actionEvents = requests
    .map((request) => actionEventFromReadModel(request, readModel ?? {}))
    .filter((event): event is AgentRuntimeExecutionEvent => Boolean(event))
    .map(
      (event) =>
        projectAgentUiState({ executionEvents: [event as TEvent] }).actions,
    )
    .flat();
  if (!actionEvents.length) return actions;
  const byId = new Map(actions.map((action) => [action.id, action]));
  actionEvents.forEach((action) => {
    if (!byId.has(action.id)) byId.set(action.id, action);
  });
  return Array.from(byId.values());
}

function activeCommandId(
  commands: CommandOutputView[],
  readModel?: CodingReadModelFacts | null,
): string | undefined {
  return (
    valueString(readModel?.active_command_id, readModel?.activeCommandId) ??
    [...commands].reverse().find((command) => command.status === "running")
      ?.commandId
  );
}

function activeTestRunId(
  tests: TestRunView[],
  readModel?: CodingReadModelFacts | null,
): string | undefined {
  return (
    valueString(readModel?.active_test_run_id, readModel?.activeTestRunId) ??
    [...tests].reverse().find((test) => test.status === "running")?.testRunId
  );
}

function preferredTab(view: {
  changes: FileChangeView[];
  commands: CommandOutputView[];
  tests: TestRunView[];
}): CodingLocalUiState["preferredTab"] {
  if (view.tests.some((test) => test.status === "failed")) return "outputs";
  if (view.commands.length) return "outputs";
  if (view.changes.length) return "changes";
  return "preview";
}

export function projectCodingWorkbenchView<
  TEvent extends AgentRuntimeExecutionEvent,
>(
  state: AgentUiProjectionState<TEvent>,
  readModel?: CodingReadModelFacts | null,
): CodingWorkbenchView<TEvent> {
  const events = state.readModel.events.map((projection) => projection.source);
  const codingEvents = collectCodingEvents(events);
  const changes = mergeChanges(collectChanges(codingEvents), readModel);
  const files = mergeFiles(collectFiles(codingEvents), changes, readModel);
  const patches = collectPatches(codingEvents);
  const changeSummary = changeSummaryFromReadModel(readModel);
  const commands = mergeCommands(collectCommands(codingEvents), readModel);
  const tests = mergeTests(collectTests(codingEvents), readModel);
  const actions = mergeActions(state.actions, readModel);
  const diagnostics = state.diagnostics.filter((diagnostic) =>
    codingEvents.some((event) => event.id === diagnostic.sourceEventId),
  );
  return {
    runtime: state.runtime,
    mainObject: buildMainObject(
      state,
      codingEvents,
      files,
      patches,
      commands,
      tests,
      readModel,
    ),
    files,
    changes,
    patches,
    changeSummary,
    commands,
    tests,
    actions,
    artifacts: state.artifacts,
    evidence: state.evidence,
    diagnostics,
    ui: {
      preferredTab: preferredTab({ changes, commands, tests }),
      stale:
        state.hydration.status === "stale" ||
        state.hydration.status === "degraded",
    },
  };
}

export function projectCodingWorkbenchViewFromEvents<
  TEvent extends AgentRuntimeExecutionEvent,
>(input?: CodingWorkbenchProjectionInput<TEvent>): CodingWorkbenchView<TEvent> {
  const threadItemEvents = executionEventsFromThreadItems(
    projectionThreadItems(input),
  ) as TEvent[];
  const executionEvents = input?.executionEvents?.length
    ? [...input.executionEvents, ...threadItemEvents]
    : threadItemEvents;
  return projectCodingWorkbenchView(
    projectAgentUiState({
      ...input,
      executionEvents,
    }),
    input?.codingReadModel,
  );
}
