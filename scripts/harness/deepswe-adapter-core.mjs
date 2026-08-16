import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { parse as parseToml } from "smol-toml";

import {
  invokeAppServerMethod,
  readAgentRuntimeThreadCurrent,
  resolveProviderPreference,
  sleep,
  startAgentSessionTurnCurrent,
  updateAgentThreadSettingsCurrent,
  waitForHealth,
} from "../lib/agent-runtime-smoke-core.mjs";

export const DEEPSWE_MANIFEST_PATH =
  "internal/test/deepswe-coding-slice-v2.json";
export const DEEPSWE_SOURCE_COMMIT = "3cda4081fed96103a6395de39c85e9b20275e307";
export const DEFAULT_DEEPSWE_TASK = "happy-dom-abort-pending-body-reads";
export const REQUIRED_VERIFIER_FILES = [
  "reward.json",
  "ctrf.json",
  "test-stdout.txt",
];
const PATCH_CAPTURE_MAX_BYTES = 64 * 1024 * 1024;

const TERMINAL_TURN_STATUSES = new Set([
  "completed",
  "failed",
  "interrupted",
  "cancelled",
  "canceled",
  "aborted",
]);
const TOOL_ITEM_TYPES = new Set([
  "command",
  "command_execution",
  "file_artifact",
  "mcpToolCall",
  "mcp_tool_call",
  "tool",
  "toolCall",
  "tool_call",
]);

function isRecord(value) {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function normalizeString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function nonNegativeInteger(value) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : null;
}

function positiveInteger(value) {
  const number = nonNegativeInteger(value);
  return number != null && number > 0 ? number : null;
}

function commandOutput(command, args, options = {}) {
  return execFileSync(command, args, {
    cwd: options.cwd,
    encoding: "utf8",
    env: options.env,
    stdio: ["ignore", "pipe", "pipe"],
  }).trim();
}

function runGit(cwd, args) {
  return commandOutput("git", args, { cwd });
}

export function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

export function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export function timestampId(date = new Date()) {
  return date
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\.\d{3}Z$/, "Z");
}

export function loadSliceManifest(
  repoRoot,
  manifestPath = DEEPSWE_MANIFEST_PATH,
) {
  const absolutePath = path.resolve(repoRoot, manifestPath);
  const manifest = readJson(absolutePath);
  if (manifest.schemaVersion !== "lime-deepswe-coding-slice-v2") {
    throw new Error(`unsupported DeepSWE manifest: ${manifest.schemaVersion}`);
  }
  return { absolutePath, manifest };
}

export function taskIdsForSlice(manifest, sliceName) {
  const taskIds = manifest?.slices?.[sliceName];
  if (!Array.isArray(taskIds) || taskIds.length === 0) {
    throw new Error(`DeepSWE slice not found or empty: ${sliceName}`);
  }
  return taskIds;
}

function readTaskToml(taskTomlPath) {
  return parseToml(fs.readFileSync(taskTomlPath, "utf8"));
}

export function loadTaskDefinition({
  repoRoot,
  sourceRoot,
  taskId,
  manifestPath = DEEPSWE_MANIFEST_PATH,
}) {
  const { manifest } = loadSliceManifest(repoRoot, manifestPath);
  const taskMetadata = manifest.tasks.find((task) => task.id === taskId);
  if (!taskMetadata) {
    throw new Error(`task is not selected by DeepSWE v2 manifest: ${taskId}`);
  }
  const taskDir = path.resolve(sourceRoot, "tasks", taskId);
  const taskTomlPath = path.join(taskDir, "task.toml");
  const instructionPath = path.join(taskDir, "instruction.md");
  if (!fs.existsSync(taskTomlPath) || !fs.existsSync(instructionPath)) {
    throw new Error(`DeepSWE task files missing: ${taskDir}`);
  }
  const taskToml = readTaskToml(taskTomlPath);
  const metadata = taskToml.metadata || {};
  const environment = taskToml.environment || {};
  const verifier = taskToml.verifier || {};
  const repositoryUrl = normalizeString(metadata.repository_url);
  const baseCommit = normalizeString(metadata.base_commit_hash);
  if (!repositoryUrl || !baseCommit) {
    throw new Error(`DeepSWE task metadata incomplete: ${taskId}`);
  }
  return {
    id: taskId,
    taskDir,
    instruction: fs.readFileSync(instructionPath, "utf8").trim(),
    language: taskMetadata.language,
    repository: taskMetadata.repository,
    repositoryUrl,
    baseCommit,
    schemaVersion: normalizeString(taskToml.schema_version),
    environment: {
      dockerImage: normalizeString(environment.docker_image),
      allowInternet: environment.allow_internet === true,
      cpus: environment.cpus ?? null,
      memoryMb: environment.memory_mb ?? null,
      storageMb: environment.storage_mb ?? null,
    },
    verifier: {
      environmentMode: normalizeString(verifier.environment_mode),
      timeoutSec: verifier.timeout_sec ?? null,
    },
  };
}

export function preflightSelectedTasks({
  repoRoot,
  sourceRoot,
  sliceName = "release-20",
  manifestPath = DEEPSWE_MANIFEST_PATH,
}) {
  const checks = [];
  const add = (name, passed, detail) => checks.push({ name, passed, detail });
  const { manifest } = loadSliceManifest(repoRoot, manifestPath);
  const sourceHead = runGit(sourceRoot, ["rev-parse", "HEAD"]);
  add(
    "source-commit",
    sourceHead === manifest.source.commit &&
      sourceHead === DEEPSWE_SOURCE_COMMIT,
    sourceHead,
  );
  const taskIds = taskIdsForSlice(manifest, sliceName);
  for (const taskId of taskIds) {
    try {
      const task = loadTaskDefinition({
        repoRoot,
        sourceRoot,
        taskId,
        manifestPath,
      });
      add(`${taskId}:schema`, task.schemaVersion === "1.1", task.schemaVersion);
      add(
        `${taskId}:verifier`,
        task.verifier.environmentMode === "separate",
        task.verifier.environmentMode,
      );
      add(
        `${taskId}:image`,
        Boolean(task.environment.dockerImage),
        task.environment.dockerImage || "missing",
      );
    } catch (error) {
      add(
        `${taskId}:load`,
        false,
        error instanceof Error ? error.message : String(error),
      );
    }
  }
  return {
    schemaVersion: "deepswe-preflight-v1",
    generatedAt: new Date().toISOString(),
    sourceCommit: sourceHead,
    sliceName,
    taskCount: taskIds.length,
    status: checks.every((check) => check.passed) ? "pass" : "fail",
    checks,
  };
}

export function runtimePrerequisites({
  pierBin = "pier",
  containerBin = "docker",
} = {}) {
  const checks = [];
  const checkCommand = (name, command, args) => {
    const result = spawnSync(command, args, {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 15_000,
    });
    checks.push({
      name,
      passed: result.status === 0,
      detail:
        normalizeString(result.error?.message) ||
        normalizeString(result.stdout) ||
        normalizeString(result.stderr) ||
        `exit=${result.status}${result.signal ? ` signal=${result.signal}` : ""}`,
    });
  };
  checkCommand("pier", pierBin, ["--version"]);
  checkCommand("container", containerBin, ["info"]);
  return {
    status: checks.every((check) => check.passed) ? "pass" : "blocked",
    checks,
  };
}

function isPathInside(parentPath, candidatePath) {
  const relative = path.relative(
    path.resolve(parentPath),
    path.resolve(candidatePath),
  );
  return (
    relative === "" ||
    (!relative.startsWith(`..${path.sep}`) &&
      relative !== ".." &&
      !path.isAbsolute(relative))
  );
}

export function createTaskWorkspaceLocation({
  repoRoot,
  tempRoot = os.tmpdir(),
}) {
  const workspaceRoot = fs.mkdtempSync(
    path.join(path.resolve(tempRoot), "lime-deepswe-workspace-"),
  );
  const workspaceDir = path.join(workspaceRoot, "workspace");
  if (isPathInside(repoRoot, workspaceDir)) {
    fs.rmSync(workspaceRoot, { recursive: true, force: true });
    throw new Error(
      `DeepSWE workspace must be outside the Lime repository: ${workspaceDir}`,
    );
  }
  return { workspaceRoot, workspaceDir };
}

export function prepareTaskWorkspace({ task, workspaceDir, runId }) {
  if (fs.existsSync(workspaceDir)) {
    throw new Error(`DeepSWE workspace already exists: ${workspaceDir}`);
  }
  fs.mkdirSync(path.dirname(workspaceDir), { recursive: true });
  fs.mkdirSync(workspaceDir);
  runGit(workspaceDir, ["init"]);
  runGit(workspaceDir, [
    "fetch",
    "--depth",
    "1",
    task.repositoryUrl,
    task.baseCommit,
  ]);
  runGit(workspaceDir, ["checkout", "-b", "main", "FETCH_HEAD"]);
  runGit(workspaceDir, ["switch", "-c", `deepswe-${runId}`]);
  runGit(workspaceDir, ["config", "user.name", "Lime DeepSWE Adapter"]);
  runGit(workspaceDir, ["config", "user.email", "deepswe@localhost"]);
  fs.appendFileSync(path.join(workspaceDir, ".git/info/exclude"), "\n.lime/\n");
  return {
    workspaceDir,
    baseCommit: task.baseCommit,
    branch: runGit(workspaceDir, ["branch", "--show-current"]),
    head: runGit(workspaceDir, ["rev-parse", "HEAD"]),
  };
}

function workspaceIdentity(response) {
  const workspace = response?.workspace;
  const workspaceId = normalizeString(
    workspace?.id || workspace?.workspaceId || workspace?.workspace_id,
  );
  const rootPath = normalizeString(
    workspace?.rootPath || workspace?.root_path || workspace?.path,
  );
  if (!workspaceId || !rootPath) {
    throw new Error("workspace/ensure did not return workspace identity");
  }
  return { workspaceId, rootPath };
}

function turnFromSessionRead(sessionRead, turnId) {
  const turns = [
    ...(Array.isArray(sessionRead?.detail?.turns)
      ? sessionRead.detail.turns
      : []),
    ...(Array.isArray(sessionRead?.turns) ? sessionRead.turns : []),
  ];
  return (
    turns.find(
      (turn) =>
        normalizeString(turn?.id || turn?.turnId || turn?.turn_id) === turnId,
    ) || null
  );
}

function turnStatus(turn) {
  return normalizeString(turn?.status).toLowerCase();
}

function isAppServerMessageTimeout(error) {
  const message = error instanceof Error ? error.message : String(error || "");
  return /timed out waiting for app-server message after \d+ms/i.test(message);
}

function normalizedStringArray(value) {
  return Array.isArray(value)
    ? [...new Set(value.map(normalizeString).filter(Boolean))].sort()
    : [];
}

function providerStepUsage(payload) {
  const runtimeEvent = isRecord(payload?.runtimeEvent)
    ? payload.runtimeEvent
    : payload;
  const raw = isRecord(runtimeEvent?.usage)
    ? runtimeEvent.usage
    : isRecord(payload?.usage)
      ? payload.usage
      : null;
  if (!raw) {
    return null;
  }
  const inputTokens = nonNegativeInteger(
    raw.input_tokens ??
      raw.inputTokens ??
      raw.prompt_tokens ??
      raw.promptTokens,
  );
  const outputTokens = nonNegativeInteger(
    raw.output_tokens ??
      raw.outputTokens ??
      raw.completion_tokens ??
      raw.completionTokens,
  );
  if (inputTokens == null || outputTokens == null) {
    return null;
  }
  const cachedInputTokens =
    nonNegativeInteger(
      raw.cached_input_tokens ??
        raw.cachedInputTokens ??
        raw.cache_read_input_tokens ??
        raw.cacheReadInputTokens,
    ) ?? 0;
  const cacheCreationInputTokens =
    nonNegativeInteger(
      raw.cache_creation_input_tokens ??
        raw.cacheCreationInputTokens ??
        raw.cache_write_input_tokens ??
        raw.cacheWriteInputTokens,
    ) ?? 0;
  return {
    inputTokens,
    outputTokens,
    cachedInputTokens,
    cacheCreationInputTokens,
    budgetTokens: Math.max(0, inputTokens - cachedInputTokens) + outputTokens,
  };
}

function currentTurns(currentFacts) {
  const candidates = [
    currentFacts?.threadRead?.turns,
    currentFacts?.threadRead?.thread?.turns,
    currentFacts?.sessionRead?.detail?.thread_read?.turns,
    currentFacts?.sessionRead?.detail?.threadRead?.turns,
    currentFacts?.sessionRead?.detail?.turns,
    currentFacts?.sessionRead?.turns,
  ];
  return candidates.find(Array.isArray) || [];
}

function currentItems(currentFacts) {
  const turns = currentTurns(currentFacts);
  const turnItems = turns.flatMap((turn) =>
    Array.isArray(turn?.items) ? turn.items : [],
  );
  const candidates = [
    ...turnItems,
    ...(Array.isArray(currentFacts?.threadRead?.thread_items)
      ? currentFacts.threadRead.thread_items
      : []),
    ...(Array.isArray(currentFacts?.threadRead?.threadItems)
      ? currentFacts.threadRead.threadItems
      : []),
    ...(Array.isArray(currentFacts?.sessionRead?.detail?.thread_read?.thread_items)
      ? currentFacts.sessionRead.detail.thread_read.thread_items
      : []),
    ...(Array.isArray(currentFacts?.sessionRead?.detail?.items)
      ? currentFacts.sessionRead.detail.items
      : []),
    ...(Array.isArray(currentFacts?.sessionRead?.items)
      ? currentFacts.sessionRead.items
      : []),
  ];
  const seen = new Set();
  return candidates.filter((item) => {
    const id = normalizeString(item?.id || item?.item_id || item?.itemId);
    if (!id) return true;
    if (seen.has(id)) return false;
    seen.add(id);
    return true;
  });
}

function currentProviderStepRecords(currentFacts) {
  const records = [];
  const add = (value) => {
    if (Array.isArray(value)) records.push(...value.filter(isRecord));
    else if (isRecord(value)) records.push(value);
  };
  const threadRead = currentFacts?.threadRead;
  const detail = currentFacts?.sessionRead?.detail;
  const readModels = [
    threadRead,
    threadRead?.diagnostics,
    threadRead?.runtime_summary,
    threadRead?.runtimeSummary,
    detail?.thread_read,
    detail?.threadRead,
    detail?.diagnostics,
    detail?.runtime_summary,
    detail?.runtimeSummary,
  ];
  for (const record of readModels) {
    add(record?.provider_steps ?? record?.providerSteps);
  }
  for (const turn of currentTurns(currentFacts)) {
    add(turn?.provider_steps ?? turn?.providerSteps);
    add(turn?.metadata?.provider_steps ?? turn?.metadata?.providerSteps);
  }
  for (const item of currentItems(currentFacts)) {
    const metadata = isRecord(item?.metadata) ? item.metadata : {};
    add(metadata.provider_steps ?? metadata.providerSteps);
    add(metadata.provider_step ?? metadata.providerStep);
  }
  return records;
}

function currentUsage(currentFacts) {
  const detail = currentFacts?.sessionRead?.detail;
  const threadRead = currentFacts?.threadRead;
  const turn = currentTurns(currentFacts).find(
    (candidate) =>
      normalizeString(candidate?.id || candidate?.turn_id || candidate?.turnId) ===
      normalizeString(currentFacts?.turnId),
  );
  return (
    turn?.usage ||
    threadRead?.diagnostics?.latest_turn_usage ||
    threadRead?.diagnostics?.latestTurnUsage ||
    threadRead?.runtime_summary?.latest_turn_usage ||
    threadRead?.runtimeSummary?.latestTurnUsage ||
    detail?.thread_read?.diagnostics?.latest_turn_usage ||
    detail?.thread_read?.diagnostics?.latestTurnUsage ||
    detail?.runtime_summary?.latest_turn_usage ||
    detail?.runtime_summary?.latestTurnUsage ||
    null
  );
}

function providerStepRecord(record, index, toolNames) {
  const runtimeEvent = isRecord(record?.runtimeEvent)
    ? record.runtimeEvent
    : record;
  const usage = providerStepUsage(runtimeEvent);
  return {
    sequence: nonNegativeInteger(
      runtimeEvent?.sequence ?? runtimeEvent?.ordinal,
    ),
    timestamp: normalizeString(
      runtimeEvent?.timestamp ?? runtimeEvent?.updatedAt ?? runtimeEvent?.updated_at,
    ) || null,
    attempt: positiveInteger(runtimeEvent?.attempt) ?? index + 1,
    completed:
      runtimeEvent?.completed === true ||
      /^(completed|succeeded|success|stop)$/i.test(
        normalizeString(runtimeEvent?.status ?? runtimeEvent?.finishReason),
      ),
    finishReason:
      normalizeString(
        runtimeEvent?.finish_reason ?? runtimeEvent?.finishReason,
      ) || null,
    output: {
      textChars:
        nonNegativeInteger(
          runtimeEvent?.text_output_chars ?? runtimeEvent?.textOutputChars,
        ) ?? 0,
      reasoningChars:
        nonNegativeInteger(
          runtimeEvent?.reasoning_output_chars ??
            runtimeEvent?.reasoningOutputChars,
        ) ?? 0,
      toolCalls:
        nonNegativeInteger(
          runtimeEvent?.tool_call_count ?? runtimeEvent?.toolCallCount,
        ) ?? 0,
    },
    toolNames: normalizedStringArray(
      runtimeEvent?.tool_names ??
        runtimeEvent?.toolNames ??
        runtimeEvent?.tools ??
        toolNames,
    ),
    usage,
  };
}

export function providerStepsFromCurrentFacts(
  currentFacts,
  { maxProviderSteps = null, tokenBudget = null } = {},
) {
  const stepLimit = positiveInteger(maxProviderSteps);
  const tokenLimit = positiveInteger(tokenBudget);
  const itemToolNames = currentItems(currentFacts)
    .filter((item) => item?.kind === "tool" || TOOL_ITEM_TYPES.has(item?.type))
    .map((item) => item?.name || item?.tool_name || item?.payload?.name)
    .filter(Boolean);
  const steps = currentProviderStepRecords(currentFacts).map((record, index) =>
    providerStepRecord(record, index, itemToolNames),
  );
  const usage = steps.reduce(
    (total, step) => {
      if (!step.usage) {
        return total;
      }
      total.stepsWithUsage += 1;
      total.inputTokens += step.usage.inputTokens;
      total.outputTokens += step.usage.outputTokens;
      total.cachedInputTokens += step.usage.cachedInputTokens;
      total.cacheCreationInputTokens += step.usage.cacheCreationInputTokens;
      total.budgetTokens += step.usage.budgetTokens;
      return total;
    },
    {
      stepsWithUsage: 0,
      inputTokens: 0,
      outputTokens: 0,
      cachedInputTokens: 0,
      cacheCreationInputTokens: 0,
      budgetTokens: 0,
    },
  );
  const fallbackUsage = providerStepUsage({ usage: currentUsage(currentFacts) });
  if (steps.length === 0 && fallbackUsage) {
    usage.stepsWithUsage = 0;
    usage.inputTokens = fallbackUsage.inputTokens;
    usage.outputTokens = fallbackUsage.outputTokens;
    usage.cachedInputTokens = fallbackUsage.cachedInputTokens;
    usage.cacheCreationInputTokens = fallbackUsage.cacheCreationInputTokens;
    usage.budgetTokens = fallbackUsage.budgetTokens;
  }
  const reasons = [];
  if (stepLimit != null && steps.length >= stepLimit) {
    reasons.push("provider_steps");
  }
  if (tokenLimit != null && usage.budgetTokens >= tokenLimit) {
    reasons.push("token_budget");
  }
  const toolSnapshots = steps.map((step) => ({
    sequence: step.sequence,
    timestamp: step.timestamp,
    attempt: step.attempt,
    toolNames: step.toolNames,
  }));
  const snapshotsWithTools = toolSnapshots.filter(
    (snapshot) => snapshot.toolNames.length > 0,
  );
  const uniqueToolNames = [
    ...new Set(snapshotsWithTools.flatMap((snapshot) => snapshot.toolNames)),
  ].sort();
  return {
    schemaVersion: "deepswe-provider-steps-v1",
    generatedAt: new Date().toISOString(),
    budgets: {
      maxProviderSteps: stepLimit,
      tokenBudget: tokenLimit,
      exhausted: reasons.length > 0,
      reasons,
      remainingProviderSteps:
        stepLimit == null ? null : Math.max(0, stepLimit - steps.length),
      remainingTokens:
        tokenLimit == null
          ? null
          : Math.max(0, tokenLimit - usage.budgetTokens),
    },
    stepCount: steps.length,
    usageStatus:
      steps.length === 0
        ? "missing"
        : usage.stepsWithUsage === steps.length
          ? "complete"
          : "partial",
    usage,
    toolCatalog: {
      status:
        toolSnapshots.length === 0
          ? "missing"
          : snapshotsWithTools.length === toolSnapshots.length
            ? "complete"
            : "partial",
      requestCount: toolSnapshots.length,
      requestsWithTools: snapshotsWithTools.length,
      uniqueToolNames,
      applyPatchAvailableOnEveryRequest:
        snapshotsWithTools.length === 0
          ? null
          : snapshotsWithTools.length === toolSnapshots.length &&
            snapshotsWithTools.every((snapshot) =>
              snapshot.toolNames.includes("apply_patch"),
            ),
      requests: toolSnapshots,
    },
    steps,
  };
}

function toolLifecycleFromCurrentFacts(currentFacts) {
  const toolItems = currentItems(currentFacts).filter((item) => {
    const itemType = normalizeString(item?.type);
    const payloadType = normalizeString(item?.payload?.type);
    return (
      item?.kind === "tool" ||
      TOOL_ITEM_TYPES.has(itemType) ||
      TOOL_ITEM_TYPES.has(payloadType)
    );
  });
  return {
    schemaVersion: "deepswe-tool-lifecycle-v1",
    itemCount: toolItems.length,
    items: toolItems,
  };
}

function trajectoryFromCurrentFacts({
  sessionId,
  turnId,
  currentFacts,
  providerSteps,
}) {
  const turns = currentTurns(currentFacts);
  return {
    schemaVersion: "deepswe-current-chain-trajectory-v1",
    sessionId,
    turnId,
    generatedAt: new Date().toISOString(),
    turns,
    items: currentItems(currentFacts),
    usage:
      providerSteps?.stepCount > 0
        ? providerSteps.usage
        : currentUsage(currentFacts),
  };
}

async function writeCurrentChainFacts({
  runDir,
  sessionId,
  turnId,
  sessionRead,
  threadRead,
  captureStatus,
  startTurnError,
  budgets,
}) {
  const currentFacts = { sessionId, turnId, sessionRead, threadRead };
  const capture = {
    status: captureStatus,
    startTurnError:
      startTurnError instanceof Error
        ? startTurnError.message
        : startTurnError
          ? String(startTurnError)
          : null,
  };
  const providerSteps = providerStepsFromCurrentFacts(currentFacts, budgets);
  writeJson(path.join(runDir, "thread-turn-item.json"), {
    schemaVersion: "deepswe-thread-turn-item-v1",
    sessionId,
    turnId,
    capture,
    sessionRead,
    threadRead,
  });
  writeJson(
    path.join(runDir, "trajectory.json"),
    trajectoryFromCurrentFacts({
      sessionId,
      turnId,
      currentFacts,
      providerSteps,
    }),
  );
  writeJson(path.join(runDir, "provider-steps.json"), providerSteps);
  writeJson(
    path.join(runDir, "tool-lifecycle.json"),
    toolLifecycleFromCurrentFacts(currentFacts),
  );
  writeJson(path.join(runDir, "app-server-facts.json"), {
    schemaVersion: "deepswe-app-server-facts-v1",
    capture,
    facts: currentFacts,
  });
  return { currentFacts, providerSteps };
}

export function terminalMessageFromCurrentFacts(currentFacts, turnId) {
  const turn = currentTurns(currentFacts).find(
    (candidate) =>
      normalizeString(candidate?.id || candidate?.turn_id || candidate?.turnId) ===
      normalizeString(turnId),
  );
  const diagnostics =
    currentFacts?.threadRead?.diagnostics ||
    currentFacts?.sessionRead?.detail?.thread_read?.diagnostics ||
    currentFacts?.sessionRead?.detail?.diagnostics ||
    {};
  return normalizeString(
    turn?.error?.message ||
      turn?.error ||
      turn?.failure?.message ||
      turn?.failure ||
      diagnostics.latest_turn_error_message ||
      diagnostics.latestTurnErrorMessage,
  );
}

export function currentChainFromError(error) {
  return error instanceof Error && error.currentChain
    ? error.currentChain
    : null;
}

export function providerStepExhaustion(providerSteps) {
  const steps = Array.isArray(providerSteps?.steps) ? providerSteps.steps : [];
  const lastStep = steps.at(-1);
  if (
    !providerSteps?.budgets?.reasons?.includes("provider_steps") ||
    normalizeString(lastStep?.finishReason).toLowerCase() !== "tool_call"
  ) {
    return null;
  }
  return {
    reasons: [...providerSteps.budgets.reasons],
    stepCount: providerSteps.stepCount,
    usage: providerSteps.usage,
  };
}

export function createCurrentChainRpc({
  invoke = invokeAppServerMethod,
  waitForReady = waitForHealth,
} = {}) {
  return {
    waitForHealth: waitForReady,
    invoke,
    resolveProvider: (options) => resolveProviderPreference(options, invoke),
    readThread: (options, sessionId, readOptions) =>
      readAgentRuntimeThreadCurrent(options, sessionId, readOptions, invoke),
    startTurn: (options, params) =>
      startAgentSessionTurnCurrent(options, params, invoke),
    cancelTurn: (options, params) => invoke(options, "turn/interrupt", params),
    updateThreadSettings: (options, params) =>
      updateAgentThreadSettingsCurrent(options, params, invoke),
    sleep,
  };
}

export async function runCurrentChainTask({
  options,
  task,
  workspaceDir,
  runDir,
  runId,
  rpc = createCurrentChainRpc(),
}) {
  const budgets = {
    maxProviderSteps: positiveInteger(options.maxProviderSteps),
    tokenBudget: positiveInteger(options.tokenBudget),
  };
  const generation = {
    max_output_tokens: positiveInteger(options.maxOutputTokens),
    enable_thinking:
      typeof options.enableThinking === "boolean"
        ? options.enableThinking
        : undefined,
  };
  const hasGenerationOverrides =
    generation.max_output_tokens != null || generation.enable_thinking != null;
  await rpc.waitForHealth(options);
  const workspaceResponse = await rpc.invoke(options, "workspace/ensure", {
    name: `DeepSWE ${task.id}`,
    rootPath: workspaceDir,
    workspaceType: "temporary",
  });
  const workspace = workspaceIdentity(workspaceResponse);
  if (path.resolve(workspace.rootPath) !== path.resolve(workspaceDir)) {
    throw new Error(`workspace/ensure root mismatch: ${workspace.rootPath}`);
  }
  const provider = await rpc.resolveProvider(options);
  const sessionId = `deepswe-${runId}`;
  const turnId = `deepswe-turn-${runId}`;
  const sessionResponse = await rpc.invoke(options, "thread/start", {
    sessionId,
    threadId: sessionId,
    appId: "desktop",
    workspaceId: workspace.workspaceId,
    workingDir: workspaceDir,
    businessObjectRef: {
      kind: "agent.session",
      id: `agent-session:${workspace.workspaceId}:${sessionId}`,
      title: `DeepSWE ${task.id}`,
      metadata: {
        title: `DeepSWE ${task.id}`,
        workingDir: workspaceDir,
        working_dir: workspaceDir,
        executionStrategy: "react",
        runStartHooks: false,
        harness: {
          hiddenFromUserRecents: true,
          source: "harness:deepswe:run",
          scenarioId: "DSW-01",
          taskId: task.id,
        },
      },
    },
  });
  const actualSessionId = normalizeString(sessionResponse?.session?.sessionId);
  if (actualSessionId !== sessionId) {
    throw new Error("thread/start did not return requested sessionId");
  }
  await rpc.updateThreadSettings(options, {
    threadId: sessionId,
    provider,
  });
  const startedAt = new Date().toISOString();
  let startTurnSettled = false;
  let startTurnError = null;
  void rpc
    .startTurn(options, {
      sessionId,
      workspaceId: workspace.workspaceId,
      message: task.instruction,
      eventName: `deepswe_${runId}`,
      turnId,
      queueIfBusy: false,
      runtimeRequest: {
        providerPreference: provider.providerPreference,
        modelPreference: provider.modelPreference,
        approvalPolicy: "never",
        sandboxPolicy: "workspace-write",
        executionStrategy: "react",
        workingDir: workspaceDir,
        workspaceRoot: workspaceDir,
        projectRoot: workspaceDir,
        webSearch: false,
        searchMode: "disabled",
        metadata: {
          harness: {
            source: "harness:deepswe:run",
            scenarioId: "DSW-01",
            taskId: task.id,
            provider_budget:
              budgets.maxProviderSteps == null && budgets.tokenBudget == null
                ? undefined
                : {
                    max_provider_steps: budgets.maxProviderSteps,
                    token_budget: budgets.tokenBudget,
                  },
            ...(hasGenerationOverrides ? { generation } : {}),
          },
        },
      },
    })
    .then(
      () => {
        startTurnSettled = true;
      },
      (error) => {
        startTurnSettled = true;
        startTurnError = error;
      },
    );

  const pollStartedAt = Date.now();
  let sessionRead = null;
  let threadRead = null;
  let turn = null;
  let budgetCancellation = null;
  while (Date.now() - pollStartedAt < options.timeoutMs) {
    [sessionRead, threadRead] = await Promise.all([
      rpc.invoke(options, "thread/read", {
        sessionId,
        historyLimit: 500,
      }),
      rpc.readThread(options, sessionId, { historyLimit: 500 }),
    ]);
    turn = turnFromSessionRead(sessionRead, turnId);
    if (turn && TERMINAL_TURN_STATUSES.has(turnStatus(turn))) {
      break;
    }
    if (startTurnSettled && startTurnError) {
      break;
    }
    await rpc.sleep(options.intervalMs);
  }

  let status = turnStatus(turn);
  let terminal = Boolean(turn && TERMINAL_TURN_STATUSES.has(status));
  const timeoutReason =
    !terminal && !budgetCancellation
      ? Date.now() - pollStartedAt >= options.timeoutMs
        ? "wall_timeout"
        : isAppServerMessageTimeout(startTurnError)
          ? "turn_start_timeout"
          : null
      : null;
  let timeoutCancellation = null;
  if (timeoutReason) {
    const requestedAt = new Date().toISOString();
    let cancellationError = null;
    try {
      await rpc.cancelTurn(options, { sessionId, turnId });
      const cancelDeadline = Date.now() + 10_000;
      while (Date.now() < cancelDeadline) {
        [sessionRead, threadRead] = await Promise.all([
          rpc.invoke(options, "thread/read", {
            sessionId,
            historyLimit: 500,
          }),
          rpc.readThread(options, sessionId, { historyLimit: 500 }),
        ]);
        turn = turnFromSessionRead(sessionRead, turnId);
        status = turnStatus(turn);
        terminal = Boolean(turn && TERMINAL_TURN_STATUSES.has(status));
        if (terminal) break;
        await rpc.sleep(options.intervalMs);
      }
    } catch (error) {
      cancellationError =
        error instanceof Error ? error.message : String(error || "unknown");
    }
    timeoutCancellation = {
      requestedAt,
      reason: timeoutReason,
      terminalStatus: terminal ? status : null,
      settledAt: terminal ? new Date().toISOString() : null,
      error: cancellationError,
    };
  }
  const factsCapture = await writeCurrentChainFacts({
    runDir,
    sessionId,
    turnId,
    sessionRead,
    threadRead,
    captureStatus: terminal ? "terminal" : "partial",
    startTurnError,
    budgets,
  });
  const stepExhaustion = providerStepExhaustion(factsCapture.providerSteps);
  const finishedAt = new Date().toISOString();
  if (timeoutReason || !terminal) {
    let message;
    if (timeoutReason) {
      message = `DeepSWE turn timeout: session=${sessionId} turn=${turnId} status=${status || "missing"} cancelStatus=${timeoutCancellation?.terminalStatus || (timeoutCancellation?.error ? "failed" : "pending")}`;
    } else if (budgetCancellation) {
      message = `DeepSWE provider budget exhausted: reasons=${budgetCancellation.reasons.join(",")} steps=${budgetCancellation.stepCount} tokens=${budgetCancellation.usage.budgetTokens}`;
    } else if (startTurnError) {
      if (isAppServerMessageTimeout(startTurnError)) {
        message = `DeepSWE turn timeout: session=${sessionId} turn=${turnId} status=${status || "in_progress"}`;
      } else {
        message =
          startTurnError instanceof Error
            ? startTurnError.message
            : String(startTurnError);
      }
    } else {
      message = `DeepSWE turn timeout: session=${sessionId} turn=${turnId} status=${status || "missing"}`;
    }
    const error = new Error(message);
    error.currentChain = {
      status: timeoutReason
        ? "timeout"
        : startTurnError
          ? status || "failed"
          : "timeout",
      terminalStatus: terminal ? status : null,
      sessionId,
      turnId,
      workspace,
      provider: {
        providerPreference: provider.providerPreference,
        providerName: provider.providerName,
        modelPreference: provider.modelPreference,
        source: provider.source,
      },
      startedAt,
      finishedAt,
      terminalMessage: message,
      factsCapture: terminal ? "terminal" : "partial",
      providerSteps: factsCapture.providerSteps,
      budgetCancellation,
      timeoutCancellation,
    };
    throw error;
  }
  return {
    status,
    sessionId,
    turnId,
    workspace,
    provider: {
      providerPreference: provider.providerPreference,
      providerName: provider.providerName,
      modelPreference: provider.modelPreference,
      source: provider.source,
    },
    startedAt,
    finishedAt,
    terminalMessage: normalizeString(
      (budgetCancellation
        ? `DeepSWE provider budget exhausted: reasons=${budgetCancellation.reasons.join(",")} steps=${budgetCancellation.stepCount} tokens=${budgetCancellation.usage.budgetTokens}`
        : stepExhaustion
          ? `DeepSWE provider budget exhausted: reasons=${stepExhaustion.reasons.join(",")} steps=${stepExhaustion.stepCount}`
          : "") ||
        turn?.error?.message ||
        turn?.error ||
        turn?.failure?.message ||
        turn?.failure ||
        turn?.message ||
        terminalMessageFromCurrentFacts(factsCapture.currentFacts, turnId),
    ),
    factsCapture: "terminal",
    providerSteps: factsCapture.providerSteps,
    budgetCancellation,
    providerStepExhaustion: stepExhaustion,
  };
}

export function capturePatch({ workspaceDir, baseCommit, outputPath }) {
  const unexpectedCommitters = runGit(workspaceDir, [
    "log",
    "--format=%ce",
    `${baseCommit}..HEAD`,
  ])
    .split("\n")
    .map((value) => value.trim())
    .filter((value) => value && value !== "deepswe@localhost");
  if (unexpectedCommitters.length > 0) {
    throw new Error(
      `DeepSWE workspace HEAD contains non-candidate commits after base: ${[...new Set(unexpectedCommitters)].join(", ")}`,
    );
  }
  runGit(workspaceDir, ["add", "-A"]);
  const patch = execFileSync(
    "git",
    ["diff", "--binary", "--cached", baseCommit],
    { cwd: workspaceDir, maxBuffer: PATCH_CAPTURE_MAX_BYTES },
  );
  fs.writeFileSync(outputPath, patch);
  return {
    path: outputPath,
    bytes: patch.length,
    status: runGit(workspaceDir, ["status", "--short"]),
    head: runGit(workspaceDir, ["rev-parse", "HEAD"]),
  };
}

export function preparePierReplayTask({ task, runDir, patchPath }) {
  const replayTaskDir = path.join(runDir, "pier-task");
  fs.cpSync(task.taskDir, replayTaskDir, {
    recursive: true,
    filter: (source) => path.basename(source) !== "solution",
  });
  const solutionDir = path.join(replayTaskDir, "solution");
  fs.mkdirSync(solutionDir, { recursive: true });
  fs.copyFileSync(patchPath, path.join(solutionDir, "model.patch"));
  const solveScript = [
    "#!/bin/bash",
    "set -euo pipefail",
    "cd /app",
    "git config user.name 'DeepSWE patch replay'",
    "git config user.email 'deepswe@localhost'",
    "git apply --binary --index /solution/model.patch",
    "git commit -m 'Apply Lime App Server candidate patch'",
    "",
  ].join("\n");
  const solvePath = path.join(solutionDir, "solve.sh");
  fs.writeFileSync(solvePath, solveScript, { mode: 0o755 });
  return { replayTaskDir, solvePath };
}

function findFileRecursively(rootDir, fileName) {
  if (!fs.existsSync(rootDir)) {
    return "";
  }
  const stack = [rootDir];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const candidate = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(candidate);
      } else if (entry.name === fileName) {
        return candidate;
      }
    }
  }
  return "";
}

export function collectPierEvidence({ jobDir, runDir }) {
  const collected = {};
  for (const fileName of REQUIRED_VERIFIER_FILES) {
    const source = findFileRecursively(jobDir, fileName);
    if (!source) {
      throw new Error(`Pier verifier evidence missing: ${fileName}`);
    }
    const destination = path.join(runDir, fileName);
    fs.copyFileSync(source, destination);
    collected[fileName] = source;
  }
  return collected;
}

export function runPierVerifier({
  task,
  runDir,
  runId,
  patchPath,
  pierBin = "pier",
  containerBin = "docker",
  timeoutMs = 7_200_000,
}) {
  const prerequisites = runtimePrerequisites({ pierBin, containerBin });
  if (prerequisites.status !== "pass") {
    throw new Error(
      `Pier verifier prerequisites blocked: ${JSON.stringify(prerequisites.checks)}`,
    );
  }
  const { replayTaskDir } = preparePierReplayTask({ task, runDir, patchPath });
  const jobsDir = path.join(runDir, "pier-jobs");
  const jobName = `verify-${runId}`;
  const result = spawnSync(
    pierBin,
    [
      "run",
      "--path",
      replayTaskDir,
      "--agent",
      "oracle",
      "--env",
      "docker",
      "--job-name",
      jobName,
      "--jobs-dir",
      jobsDir,
      "--n-concurrent",
      "1",
      "--max-retries",
      "0",
      "--yes",
      "--quiet",
    ],
    {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: timeoutMs,
      env: { ...process.env, PIER_CONTAINER_BIN: containerBin },
    },
  );
  fs.writeFileSync(
    path.join(runDir, "pier-stdout.txt"),
    `${result.stdout || ""}${result.stderr || ""}`,
    "utf8",
  );
  if (result.error || result.status !== 0) {
    throw new Error(
      `Pier verifier failed: ${result.error?.message || `exit=${result.status}`}`,
    );
  }
  const jobDir = path.join(jobsDir, jobName);
  return {
    jobDir,
    evidence: collectPierEvidence({ jobDir, runDir }),
  };
}

export function classifyFailure(stage, error) {
  const message = error instanceof Error ? error.message : String(error);
  let owner = "environment";
  if (
    /unsupported workspace_type|spawnSync git ENOBUFS|workspace HEAD contains non-candidate commits/i.test(
      message,
    )
  ) {
    owner = "harness";
  } else if (
    /fetch failed|ECONNRESET|ECONNREFUSED|DevBridge health/i.test(message)
  ) {
    owner = "transport";
  } else if (
    /budget|token|cost|DeepSWE turn timeout|timed out waiting for app-server message/i.test(
      message,
    )
  ) {
    owner = "budget";
  } else if (
    /provider|model|api key|authentication|rate.limit/i.test(message)
  ) {
    owner = "model";
  } else if (/empty patch|produced no candidate|\bno[- ]op\b/i.test(message)) {
    owner = "model";
  } else if (/tool|sandbox|approval/i.test(message)) {
    owner = "tool-runtime";
  } else if (
    /app server|agentSession|thread\/(?:start|read)|turn\/(?:start|interrupt)|workspace\/ensure|evidence\/export|DevBridge/i.test(
      message,
    )
  ) {
    owner = "app-server";
  } else if (/Pier|verifier|reward\.json|ctrf\.json/i.test(message)) {
    owner = "verifier";
  } else if (stage === "transport") {
    owner = "transport";
  } else if (stage.startsWith("agent") || /terminal status/i.test(message)) {
    owner = "agent-runtime";
  }
  return {
    schemaVersion: "deepswe-failure-classification-v1",
    generatedAt: new Date().toISOString(),
    status: "failed",
    stage,
    owner,
    message,
  };
}

export function writeRunContext(runDir, context) {
  writeJson(path.join(runDir, "run-context.json"), {
    schemaVersion: "deepswe-run-context-v1",
    ...context,
  });
}
