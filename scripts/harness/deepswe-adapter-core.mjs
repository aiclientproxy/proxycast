import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { parse as parseToml } from "smol-toml";

import {
  invokeAppServerMethod,
  resolveProviderPreference,
  sleep,
  waitForHealth,
} from "../lib/agent-runtime-smoke-core.mjs";
import {
  currentItems,
  currentTurns,
  currentUsage,
  providerStepExhaustion,
  providerStepsFromCurrentFacts,
  terminalMessageFromCurrentFacts,
  toolLifecycleFromCurrentFacts,
} from "./deepswe-provider-evidence.mjs";
import {
  classifyFailure,
  currentChainFromError,
  verifierCompletionStatus,
} from "./deepswe-failure.mjs";
import {
  isRecord,
  nonNegativeInteger,
  normalizeString,
  positiveInteger,
} from "./deepswe-value-utils.mjs";

export {
  classifyFailure,
  currentChainFromError,
  providerStepExhaustion,
  providerStepsFromCurrentFacts,
  terminalMessageFromCurrentFacts,
  verifierCompletionStatus,
};
export const DEEPSWE_MANIFEST_PATH =
  "internal/test/deepswe-coding-slice-v2.json";
export const DEEPSWE_SOURCE_COMMIT = "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9";
export const DEEPSWE_TASK_SCHEMA_VERSION = "1.3";
export const DEEPSWE_PIER_VERSION = "0.3.1";
export const DEEPSWE_PIER_PACKAGE = `datacurve-pier==${DEEPSWE_PIER_VERSION}`;
export const DEEPSWE_ADAPTER_VERSION = "deepswe-current-chain-adapter-v7";
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
function positiveNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
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

export function verifierTaskIdForRun({ runDir, requestedTaskId = "" }) {
  const contextPath = path.join(path.resolve(runDir), "run-context.json");
  if (!fs.existsSync(contextPath)) {
    throw new Error(`DeepSWE verifier run context missing: ${contextPath}`);
  }
  const taskId = normalizeString(readJson(contextPath)?.task?.id);
  if (!taskId) {
    throw new Error(`DeepSWE verifier task id missing: ${contextPath}`);
  }
  const requested = normalizeString(requestedTaskId);
  if (requested && requested !== taskId) {
    throw new Error(
      `DeepSWE verifier task mismatch: expected=${taskId} actual=${requested}`,
    );
  }
  return taskId;
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
  const agent = taskToml.agent || {};
  const environment = taskToml.environment || {};
  const verifier = taskToml.verifier || {};
  const verifierCollect = Array.isArray(verifier.collect)
    ? verifier.collect.filter(isRecord)
    : [];
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
    artifacts: Array.isArray(taskToml.artifacts)
      ? taskToml.artifacts.map(normalizeString).filter(Boolean)
      : [],
    agent: {
      networkMode: normalizeString(agent.network_mode),
      timeoutSec: positiveNumber(agent.timeout_sec),
    },
    environment: {
      dockerImage: normalizeString(environment.docker_image),
      cpus: environment.cpus ?? null,
      memoryMb: environment.memory_mb ?? null,
      storageMb: environment.storage_mb ?? null,
    },
    verifier: {
      networkMode: normalizeString(verifier.network_mode),
      environmentMode: normalizeString(verifier.environment_mode),
      timeoutSec: positiveNumber(verifier.timeout_sec),
      collect: verifierCollect.map((collect) => ({
        command: normalizeString(collect.command),
        timeoutSec: positiveNumber(collect.timeout_sec),
      })),
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
  add(
    "source-license",
    fs.existsSync(path.join(sourceRoot, "LICENSE")),
    "LICENSE",
  );
  add(
    "source-provenance",
    fs.existsSync(path.join(sourceRoot, "PROVENANCE.md")),
    "PROVENANCE.md",
  );
  add(
    "manifest-task-schema",
    manifest.source.taskSchemaVersion === DEEPSWE_TASK_SCHEMA_VERSION,
    manifest.source.taskSchemaVersion,
  );
  add(
    "manifest-pier",
    manifest.source.runner === DEEPSWE_PIER_PACKAGE,
    manifest.source.runner,
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
      add(
        `${taskId}:schema`,
        task.schemaVersion === DEEPSWE_TASK_SCHEMA_VERSION,
        task.schemaVersion,
      );
      add(
        `${taskId}:agent-network`,
        task.agent.networkMode === "no-network",
        task.agent.networkMode,
      );
      add(
        `${taskId}:verifier-network`,
        task.verifier.networkMode === "no-network",
        task.verifier.networkMode,
      );
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
      add(
        `${taskId}:artifact`,
        task.artifacts.includes("/logs/artifacts/model.patch"),
        task.artifacts.join(",") || "missing",
      );
      add(
        `${taskId}:collect-count`,
        task.verifier.collect.length === 1,
        String(task.verifier.collect.length),
      );
      const collect = task.verifier.collect[0];
      const collectCommand = collect?.command || "";
      add(
        `${taskId}:collect-command`,
        collectCommand.includes(`git diff --binary ${task.baseCommit} HEAD`) &&
          collectCommand.includes("> /logs/artifacts/model.patch"),
        collectCommand || "missing",
      );
      add(
        `${taskId}:collect-timeout`,
        collect?.timeoutSec != null,
        collect?.timeoutSec ?? "missing",
      );
      add(
        `${taskId}:pre-artifacts-deleted`,
        !fs.existsSync(path.join(task.taskDir, "pre_artifacts.sh")),
        path.join(task.taskDir, "pre_artifacts.sh"),
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
  const checkCommand = (name, command, args, expectedOutput = null) => {
    const result = spawnSync(command, args, {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: name === "pier" ? 60_000 : 15_000,
    });
    const output =
      normalizeString(result.stdout) || normalizeString(result.stderr);
    const versionMatches = expectedOutput == null || output === expectedOutput;
    checks.push({
      name,
      passed: result.status === 0 && versionMatches,
      detail:
        normalizeString(result.error?.message) ||
        (result.status === 0 && !versionMatches
          ? `expected ${expectedOutput}; actual ${output || "empty"}`
          : output) ||
        `exit=${result.status}${result.signal ? ` signal=${result.signal}` : ""}`,
    });
  };
  checkCommand("pier", pierBin, ["--version"], DEEPSWE_PIER_VERSION);
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
    ...(Array.isArray(sessionRead?.thread?.turns)
      ? sessionRead.thread.turns
      : []),
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
  runtimeEvents = [],
}) {
  const currentFacts = {
    sessionId,
    turnId,
    sessionRead,
    threadRead,
    runtimeEvents,
  };
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

export function createCurrentChainRpc({
  invoke = invokeAppServerMethod,
  waitForReady = waitForHealth,
  readRuntimeEvents = async () => [],
} = {}) {
  return {
    waitForHealth: waitForReady,
    invoke,
    resolveProvider: (options) => resolveProviderPreference(options, invoke),
    readThread: (options, threadId) =>
      invoke(options, "thread/read", { threadId, includeTurns: true }),
    startTurn: (options, params) => invoke(options, "turn/start", params),
    cancelTurn: (options, params) => invoke(options, "turn/interrupt", params),
    readRuntimeEvents,
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
    first_visible_output_timeout_ms: positiveInteger(
      options.firstVisibleOutputTimeoutMs,
    ),
    max_output_tokens: positiveInteger(options.maxOutputTokens),
    enable_thinking:
      typeof options.enableThinking === "boolean"
        ? options.enableThinking
        : undefined,
  };
  const hasGenerationOverrides =
    generation.first_visible_output_timeout_ms != null ||
    generation.max_output_tokens != null ||
    generation.enable_thinking != null;
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
  const threadResponse = await rpc.invoke(options, "thread/start", {
    model: provider.modelPreference,
    modelProvider: provider.providerPreference,
    cwd: workspaceDir,
    runtimeWorkspaceRoots: [workspaceDir],
    approvalPolicy: "never",
    sandbox: "workspace-write",
    serviceName: `DeepSWE ${task.id}`,
    historyMode: "paginated",
    threadSource: "appServer",
  });
  const threadId = normalizeString(threadResponse?.thread?.id);
  const sessionId = normalizeString(threadResponse?.thread?.sessionId);
  if (!threadId || !sessionId) {
    throw new Error(
      "thread/start did not return canonical thread/session identity",
    );
  }
  const startedAt = new Date().toISOString();
  let turnId = "";
  let startTurnError = null;
  const harnessMetadata = {
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
  };
  try {
    const turnResponse = await rpc.startTurn(options, {
      threadId,
      clientUserMessageId: `deepswe-user-${runId}`,
      input: [{ type: "text", text: task.instruction }],
      cwd: workspaceDir,
      runtimeWorkspaceRoots: [workspaceDir],
      approvalPolicy: "never",
      sandboxPolicy: "workspace-write",
      model: provider.modelPreference,
      responsesapiClientMetadata: {
        source: "harness:deepswe:run",
        scenarioId: "DSW-01",
        taskId: task.id,
      },
      additionalContext: {
        metadata: {
          kind: "application",
          value: JSON.stringify(harnessMetadata),
        },
      },
    });
    turnId = normalizeString(turnResponse?.turn?.id);
    if (!turnId) {
      throw new Error("turn/start did not return canonical turn.id");
    }
  } catch (error) {
    startTurnError = error;
  }

  const pollStartedAt = Date.now();
  let sessionRead = null;
  let threadRead = null;
  let turn = null;
  let budgetCancellation = null;
  while (Date.now() - pollStartedAt < options.timeoutMs) {
    threadRead = await rpc.readThread(options, threadId);
    sessionRead = threadRead;
    turn = turnFromSessionRead(sessionRead, turnId);
    if (turn && TERMINAL_TURN_STATUSES.has(turnStatus(turn))) {
      break;
    }
    if (startTurnError) {
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
      if (!turnId) {
        throw new Error("turn/start did not return a turn to interrupt");
      }
      await rpc.cancelTurn(options, { threadId, turnId });
      const cancelDeadline = Date.now() + 10_000;
      while (Date.now() < cancelDeadline) {
        threadRead = await rpc.readThread(options, threadId);
        sessionRead = threadRead;
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
    runtimeEvents:
      typeof rpc.readRuntimeEvents === "function"
        ? await rpc.readRuntimeEvents({ sessionId, threadId, turnId })
        : [],
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
      threadId,
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
    threadId,
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
    "git apply --binary --whitespace=nowarn /solution/model.patch",
    "git add -A",
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

function pierRewardValue(reward) {
  if (typeof reward === "number") return reward;
  if (typeof reward?.reward === "number") return reward.reward;
  if (typeof reward?.score === "number") return reward.score;
  if (typeof reward?.passed === "boolean") return reward.passed ? 1 : 0;
  return null;
}

function availablePierJobName(jobsDir, runId) {
  const baseName = `verify-${runId}`;
  if (!fs.existsSync(path.join(jobsDir, baseName))) {
    return baseName;
  }
  let retry = 1;
  while (fs.existsSync(path.join(jobsDir, `${baseName}-retry-${retry}`))) {
    retry += 1;
  }
  return `${baseName}-retry-${retry}`;
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
  const pierTempDir = path.join(runDir, "pier-tmp");
  fs.mkdirSync(pierTempDir, { recursive: true });
  const jobName = availablePierJobName(jobsDir, runId);
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
      "--verifier-env",
      "NEXTEST_DOUBLE_SPAWN=0",
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
      env: {
        ...process.env,
        TMPDIR: pierTempDir,
        PIER_CONTAINER_BIN: containerBin,
      },
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
  const evidence = collectPierEvidence({ jobDir, runDir });
  const reward = pierRewardValue(readJson(path.join(runDir, "reward.json")));
  return {
    jobDir,
    evidence,
    reward,
  };
}

export function writeRunContext(runDir, context) {
  writeJson(path.join(runDir, "run-context.json"), {
    schemaVersion: "deepswe-run-context-v1",
    ...context,
  });
}

export function runContextBase(options, runId, task) {
  return {
    generatedAt: new Date().toISOString(),
    runId,
    scenarioId: "DSW-01",
    sourceCommit: DEEPSWE_SOURCE_COMMIT,
    task: {
      id: task.id,
      language: task.language,
      repository: task.repository,
      repositoryUrl: task.repositoryUrl,
      baseCommit: task.baseCommit,
      schemaVersion: task.schemaVersion,
      environment: task.environment,
      verifier: task.verifier,
    },
    executionContract: {
      adapterVersion: DEEPSWE_ADAPTER_VERSION,
      agentPath: "Lime App Server JSON-RPC current chain",
      appServerMethods: [
        "workspace/ensure",
        "thread/start",
        "turn/start",
        "thread/read",
        "turn/interrupt",
      ],
      verifier: "Pier separate verifier with patch replay",
      transport: options.transport,
      appServerDataIsolation:
        options.transport === "stdio" ? "sqlite-vacuum-snapshot" : null,
      taskWorkspaceIsolation: "system-temp-outside-repository",
      liveProviderExplicitlyAllowed: options.allowLiveProvider,
      providerBudget: {
        maxProviderSteps: options.maxProviderSteps,
        tokenBudget: options.tokenBudget,
        tokenFormula: "max(0,input_tokens-cached_input_tokens)+output_tokens",
        enforcementOwner:
          "agent-runtime reply loop before tool execution and next sampling",
      },
      generationControls: {
        firstVisibleOutputTimeoutMs: options.firstVisibleOutputTimeoutMs,
        maxOutputTokens: options.maxOutputTokens,
        enableThinking: options.enableThinking,
        projection:
          "additionalContext.metadata -> RuntimeRequest.metadata.harness.generation",
      },
    },
  };
}
