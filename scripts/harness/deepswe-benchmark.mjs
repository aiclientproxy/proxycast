#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  DEEPSWE_ADAPTER_VERSION,
  DEEPSWE_SOURCE_COMMIT,
  DEEPSWE_TASK_SCHEMA_VERSION,
  loadSliceManifest,
  preflightSelectedTasks,
  readJson,
  taskIdsForSlice,
  writeJson,
} from "./deepswe-adapter-core.mjs";

export const DEEPSWE_BATCH_SCHEMA = "deepswe-batch-summary-v1";
export const DEFAULT_BATCH_OUTPUT =
  ".lime/benchmark/v2/runs/batch-summary.json";
export const REQUIRED_VERIFIER_FILES = [
  "reward.json",
  "ctrf.json",
  "test-stdout.txt",
];

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const adapterPath = path.join(__dirname, "deepswe-adapter.mjs");

function normalizeString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function positiveInteger(value, name) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return number;
}

export function parseArgs(argv) {
  const options = {
    aggregate: false,
    allowLiveProvider: false,
    appServerBin: "",
    appServerDataDir: "",
    containerBin: "",
    enableThinking: null,
    help: false,
    manifestPath: "internal/test/deepswe-coding-slice-v2.json",
    maxOutputTokens: null,
    maxProviderSteps: null,
    mode: null,
    model: "",
    noWrite: false,
    output: path.resolve(repoRoot, DEFAULT_BATCH_OUTPUT),
    pierBin: "",
    plan: false,
    provider: "",
    runsRoot: path.resolve(repoRoot, ".lime/benchmark/v2/runs"),
    sliceName: "release-20",
    sourceRoot: path.resolve(repoRoot, ".lime/benchmark/sources/deep-swe"),
    timeoutMs: null,
    tokenBudget: null,
    trials: 1,
  };
  const valueOptions = new Map([
    ["--app-server-bin", "appServerBin"],
    ["--app-server-data-dir", "appServerDataDir"],
    ["--container-bin", "containerBin"],
    ["--manifest", "manifestPath"],
    ["--max-output-tokens", "maxOutputTokens"],
    ["--max-provider-steps", "maxProviderSteps"],
    ["--model", "model"],
    ["--output", "output"],
    ["--pier-bin", "pierBin"],
    ["--provider", "provider"],
    ["--runs-root", "runsRoot"],
    ["--slice", "sliceName"],
    ["--source-root", "sourceRoot"],
    ["--transport", "transport"],
    ["--timeout-ms", "timeoutMs"],
    ["--token-budget", "tokenBudget"],
    ["--trials", "trials"],
    ["--enable-thinking", "enableThinking"],
  ]);
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      options.help = true;
      continue;
    }
    if (arg === "--plan" || arg === "--aggregate" || arg === "--run") {
      const mode = arg.slice(2);
      if (options.mode && options.mode !== mode) {
        throw new Error("--plan, --aggregate and --run are mutually exclusive");
      }
      options.mode = mode;
      options[mode] = true;
      continue;
    }
    if (arg === "--allow-live-provider") {
      options.allowLiveProvider = true;
      continue;
    }
    if (arg === "--no-write") {
      options.noWrite = true;
      continue;
    }
    const key = valueOptions.get(arg);
    if (key && argv[index + 1]) {
      options[key] = argv[index + 1];
      index += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (options.help) return options;
  if (!options.mode) {
    throw new Error("one of --plan, --aggregate or --run is required");
  }
  options.trials = positiveInteger(options.trials, "--trials");
  if (![1, 3].includes(options.trials)) {
    throw new Error("--trials must be 1 or 3");
  }
  options.manifestPath = path.resolve(repoRoot, options.manifestPath);
  options.sourceRoot = path.resolve(repoRoot, options.sourceRoot);
  options.runsRoot = path.resolve(repoRoot, options.runsRoot);
  options.output = path.resolve(repoRoot, options.output);
  if (options.mode === "run" && !options.allowLiveProvider) {
    throw new Error("--run requires --allow-live-provider");
  }
  if (options.enableThinking != null) {
    if (!["true", "false"].includes(String(options.enableThinking))) {
      throw new Error("--enable-thinking must be true or false");
    }
  }
  return options;
}

function usage() {
  return `
DeepSWE Core batch benchmark

Usage:
  node scripts/harness/deepswe-benchmark.mjs --plan --slice smoke-10 --trials 1
  node scripts/harness/deepswe-benchmark.mjs --run --slice release-20 --trials 3 --allow-live-provider
  node scripts/harness/deepswe-benchmark.mjs --aggregate --slice release-20 --trials 3

The aggregate is score-eligible only when every trial has current source/adapter/schema
identity, a non-empty patch, completed current-chain evidence, and reward.json,
ctrf.json and test-stdout.txt from the same Pier run.
`;
}

export function createBatchPlan({
  repoRoot: root = repoRoot,
  manifestPath = "internal/test/deepswe-coding-slice-v2.json",
  sourceRoot = path.resolve(root, ".lime/benchmark/sources/deep-swe"),
  sliceName = "release-20",
  trials = 1,
} = {}) {
  const { manifest } = loadSliceManifest(root, manifestPath);
  const taskIds = taskIdsForSlice(manifest, sliceName);
  const preflight = preflightSelectedTasks({
    repoRoot: root,
    sourceRoot,
    sliceName,
    manifestPath,
  });
  const tasks = taskIds.flatMap((taskId) =>
    Array.from({ length: trials }, (_, index) => ({
      taskId,
      trialIndex: index + 1,
      trialKey: `${taskId}#${index + 1}`,
    })),
  );
  return {
    schemaVersion: "deepswe-batch-plan-v1",
    generatedAt: new Date().toISOString(),
    status: preflight.status === "pass" ? "ready" : "blocked",
    source: {
      commit: manifest.source.commit,
      taskSchemaVersion: manifest.source.taskSchemaVersion,
      runner: manifest.source.runner,
      adapterVersion: manifest.executionContract.adapterVersion,
    },
    sliceName,
    trialsPerTask: trials,
    taskCount: taskIds.length,
    expectedTrialCount: tasks.length,
    taskIds,
    tasks,
    preflight: {
      status: preflight.status,
      checks: preflight.checks.length,
      passed: preflight.checks.filter((check) => check.passed).length,
      failed: preflight.checks
        .filter((check) => !check.passed)
        .map((check) => check.name),
    },
  };
}

function walkFiles(root, fileName) {
  if (!fs.existsSync(root)) return [];
  const files = [];
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const candidate = path.join(current, entry.name);
      if (entry.isDirectory()) stack.push(candidate);
      else if (entry.isFile() && entry.name === fileName) files.push(candidate);
    }
  }
  return files.sort();
}

function readOptional(filePath) {
  try {
    return readJson(filePath);
  } catch {
    return null;
  }
}

function resolveEvidencePath(runDir, value) {
  const normalized = normalizeString(value);
  if (!normalized) return "";
  return path.isAbsolute(normalized)
    ? normalized
    : path.resolve(runDir, normalized);
}

function rewardValue(runDir, result) {
  const configured = result?.verification?.evidence?.["reward.json"];
  const rewardPath = resolveEvidencePath(runDir, configured);
  const reward = readOptional(rewardPath);
  if (typeof reward === "number") return reward;
  if (typeof reward?.reward === "number") return reward.reward;
  if (typeof reward?.score === "number") return reward.score;
  if (typeof reward?.passed === "boolean") return reward.passed ? 1 : 0;
  return null;
}

function trialIdentity({ context, manifest }) {
  const checks = [
    [
      "source-commit",
      context?.sourceCommit === manifest.source?.commit &&
        manifest.source?.commit === DEEPSWE_SOURCE_COMMIT,
    ],
    [
      "task-schema",
      context?.task?.schemaVersion === manifest.source?.taskSchemaVersion &&
        manifest.source?.taskSchemaVersion === DEEPSWE_TASK_SCHEMA_VERSION,
    ],
    [
      "adapter-version",
      context?.executionContract?.adapterVersion ===
        manifest.executionContract?.adapterVersion &&
        manifest.executionContract?.adapterVersion === DEEPSWE_ADAPTER_VERSION,
    ],
  ];
  return {
    passed: checks.every(([, passed]) => passed),
    failed: checks.filter(([, passed]) => !passed).map(([name]) => name),
  };
}

function classifyTrial({ runDir, result, context, manifest }) {
  const identity = trialIdentity({ context, manifest });
  const taskId = normalizeString(context?.task?.id);
  const patchBytes = Number(result?.patch?.bytes) || 0;
  const currentChainCompleted = result?.currentChain?.status === "completed";
  const evidence = result?.verification?.evidence || {};
  const missingArtifacts = REQUIRED_VERIFIER_FILES.filter(
    (name) => !fs.existsSync(resolveEvidencePath(runDir, evidence[name])),
  );
  const reward = rewardValue(runDir, result);
  const base = {
    runId: normalizeString(result?.runId) || path.basename(runDir),
    taskId,
    runDir,
    resultStatus: normalizeString(result?.status),
    patchBytes,
    currentChainCompleted,
    identity,
    missingArtifacts,
    reward,
    wallTimeMs: null,
    budgetTokens: null,
  };
  const startedAt = Date.parse(result?.currentChain?.startedAt || "");
  const finishedAt = Date.parse(result?.currentChain?.finishedAt || "");
  if (Number.isFinite(startedAt) && Number.isFinite(finishedAt)) {
    base.wallTimeMs = Math.max(0, finishedAt - startedAt);
  }
  base.budgetTokens =
    Number(result?.currentChain?.providerSteps?.usage?.budgetTokens) || null;
  if (!identity.passed) {
    return {
      ...base,
      validity: "invalid_identity",
      reason: identity.failed.join(","),
    };
  }
  if (
    base.resultStatus === "verified" &&
    patchBytes > 0 &&
    currentChainCompleted &&
    missingArtifacts.length === 0 &&
    reward != null
  ) {
    return { ...base, validity: "verified", pass: reward > 0 };
  }
  const owner = normalizeString(result?.failure?.owner);
  const blocked =
    result?.verifierPrerequisites?.status === "blocked" ||
    result?.verification?.status === "blocked" ||
    ["environment", "verifier", "transport", "harness"].includes(owner);
  return {
    ...base,
    validity: blocked ? "infra_failure" : "incomplete",
    reason:
      missingArtifacts.length > 0
        ? `missing:${missingArtifacts.join(",")}`
        : normalizeString(result?.failure?.message) ||
          "trial is not verifier-complete",
  };
}

function chooseCombination(n, k) {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let index = 1; index <= k; index += 1) {
    result = (result * (n - index + 1)) / index;
  }
  return result;
}

function passAtK(passCount, trialCount, k) {
  if (k > trialCount) return null;
  if (passCount === 0) return 0;
  if (passCount === trialCount || k === trialCount) return 1;
  return (
    1 -
    chooseCombination(trialCount - passCount, k) /
      chooseCombination(trialCount, k)
  );
}

export function aggregateBatch({
  repoRoot: root = repoRoot,
  manifestPath = "internal/test/deepswe-coding-slice-v2.json",
  runsRoot = path.resolve(root, ".lime/benchmark/v2/runs"),
  sliceName = "release-20",
  trials = 1,
} = {}) {
  const { manifest } = loadSliceManifest(root, manifestPath);
  const taskIds = taskIdsForSlice(manifest, sliceName);
  const records = walkFiles(runsRoot, "adapter-result.json").map(
    (resultPath) => {
      const runDir = path.dirname(resultPath);
      const result = readOptional(resultPath) || {};
      const context = readOptional(path.join(runDir, "run-context.json"));
      return classifyTrial({ runDir, result, context, manifest });
    },
  );
  const taskStats = taskIds.map((taskId) => {
    const trialsForTask = records
      .filter((record) => record.taskId === taskId)
      .sort((left, right) => left.runId.localeCompare(right.runId));
    const currentTrials = trialsForTask.filter(
      (record) => record.identity.passed,
    );
    const selected = currentTrials.slice(0, trials);
    const extraTrialCount = Math.max(0, currentTrials.length - trials);
    const staleIdentityCount = trialsForTask.filter(
      (record) => !record.identity.passed,
    ).length;
    const verified = selected.filter(
      (record) => record.validity === "verified",
    );
    const passCount = verified.filter((record) => record.pass).length;
    return {
      taskId,
      expectedTrials: trials,
      observedTrials: trialsForTask.length,
      selectedTrials: selected,
      extraTrialCount,
      staleIdentityCount,
      passCount,
      passAt1:
        selected.length === trials ? passAtK(passCount, trials, 1) : null,
      passAt3:
        selected.length === trials ? passAtK(passCount, trials, 3) : null,
      passPower3:
        selected.length === trials && trials === 3 ? passCount === 3 : null,
      status:
        selected.length !== trials || extraTrialCount > 0
          ? "missing_trials"
          : selected.every((record) => record.validity === "verified")
            ? "verified"
            : "blocked",
    };
  });
  const complete = taskStats.every((task) => task.status === "verified");
  const allTaskPassAt1 = taskStats
    .map((task) => task.passAt1)
    .filter((value) => value != null);
  const allTaskPassAt3 = taskStats
    .map((task) => task.passAt3)
    .filter((value) => value != null);
  const allPassPower3 = taskStats
    .map((task) => task.passPower3)
    .filter((value) => value != null);
  const allSelected = taskStats.flatMap((task) => task.selectedTrials);
  const taskIdSet = new Set(taskIds);
  const invalidIdentityCount = records.filter(
    (record) =>
      taskIdSet.has(record.taskId) && record.validity === "invalid_identity",
  ).length;
  const infraFailureCount = allSelected.filter(
    (record) => record.validity === "infra_failure",
  ).length;
  const incompleteTrialCount = allSelected.filter(
    (record) => record.validity === "incomplete",
  ).length;
  const wallTimes = allSelected
    .map((record) => record.wallTimeMs)
    .filter((value) => value != null);
  const budgets = allSelected
    .map((record) => record.budgetTokens)
    .filter((value) => value != null);
  return {
    schemaVersion: DEEPSWE_BATCH_SCHEMA,
    generatedAt: new Date().toISOString(),
    status: complete ? "complete" : "blocked",
    scoreEligible: complete,
    infraValid: complete && infraFailureCount === 0,
    source: {
      commit: manifest.source.commit,
      taskSchemaVersion: manifest.source.taskSchemaVersion,
      runner: manifest.source.runner,
      adapterVersion: manifest.executionContract.adapterVersion,
      expected: {
        commit: DEEPSWE_SOURCE_COMMIT,
        taskSchemaVersion: DEEPSWE_TASK_SCHEMA_VERSION,
        adapterVersion: DEEPSWE_ADAPTER_VERSION,
      },
    },
    sliceName,
    trialsPerTask: trials,
    taskCount: taskIds.length,
    observedRunCount: records.length,
    invalidIdentityCount,
    infraFailureCount,
    incompleteTrialCount,
    metrics: {
      passAt1:
        allTaskPassAt1.length === taskIds.length
          ? allTaskPassAt1.reduce((sum, value) => sum + value, 0) /
            taskIds.length
          : null,
      passAt3:
        allTaskPassAt3.length === taskIds.length
          ? allTaskPassAt3.reduce((sum, value) => sum + value, 0) /
            taskIds.length
          : null,
      passPower3:
        allPassPower3.length === taskIds.length
          ? allPassPower3.filter(Boolean).length / taskIds.length
          : null,
      wallTimeMs: wallTimes.length
        ? {
            count: wallTimes.length,
            total: wallTimes.reduce((sum, value) => sum + value, 0),
            mean:
              wallTimes.reduce((sum, value) => sum + value, 0) /
              wallTimes.length,
          }
        : null,
      budgetTokens: budgets.length
        ? {
            count: budgets.length,
            total: budgets.reduce((sum, value) => sum + value, 0),
            mean:
              budgets.reduce((sum, value) => sum + value, 0) / budgets.length,
          }
        : null,
    },
    taskStats,
  };
}

function adapterArgs(options, taskId) {
  const args = [
    adapterPath,
    "--task",
    taskId,
    "--slice",
    options.sliceName,
    "--runs-root",
    options.runsRoot,
    "--allow-live-provider",
  ];
  const values = [
    ["--provider", options.provider],
    ["--model", options.model],
    ["--transport", options.transport],
    ["--app-server-bin", options.appServerBin],
    ["--app-server-data-dir", options.appServerDataDir],
    ["--container-bin", options.containerBin],
    ["--pier-bin", options.pierBin],
    ["--max-output-tokens", options.maxOutputTokens],
    ["--max-provider-steps", options.maxProviderSteps],
    ["--timeout-ms", options.timeoutMs],
    ["--token-budget", options.tokenBudget],
    ["--enable-thinking", options.enableThinking],
  ];
  for (const [name, value] of values) {
    if (value !== "" && value != null) args.push(name, String(value));
  }
  return args;
}

export function runBatch(options, root = repoRoot) {
  const plan = createBatchPlan({
    repoRoot: root,
    manifestPath: options.manifestPath,
    sourceRoot: options.sourceRoot,
    sliceName: options.sliceName,
    trials: options.trials,
  });
  if (plan.status !== "ready")
    throw new Error("DeepSWE batch preflight is blocked");
  const runs = [];
  for (const item of plan.tasks) {
    const result = spawnSync(
      process.execPath,
      adapterArgs(options, item.taskId),
      {
        cwd: root,
        encoding: "utf8",
        stdio: "inherit",
      },
    );
    runs.push({
      ...item,
      exitCode: result.status,
      signal: result.signal || null,
    });
  }
  const summary = aggregateBatch({
    repoRoot: root,
    manifestPath: options.manifestPath,
    runsRoot: options.runsRoot,
    sliceName: options.sliceName,
    trials: options.trials,
  });
  return { ...summary, plan, runs };
}

export function runBenchmark(options, root = repoRoot) {
  if (options.mode === "plan") {
    const plan = createBatchPlan({
      repoRoot: root,
      manifestPath: options.manifestPath,
      sourceRoot: options.sourceRoot,
      sliceName: options.sliceName,
      trials: options.trials,
    });
    if (!options.noWrite) writeJson(options.output, plan);
    return plan;
  }
  if (options.mode === "aggregate") {
    const summary = aggregateBatch({
      repoRoot: root,
      manifestPath: options.manifestPath,
      runsRoot: options.runsRoot,
      sliceName: options.sliceName,
      trials: options.trials,
    });
    if (!options.noWrite) writeJson(options.output, summary);
    return summary;
  }
  const result = runBatch(options, root);
  if (!options.noWrite) writeJson(options.output, result);
  return result;
}

if (
  process.argv[1] &&
  pathToFileURL(process.argv[1]).href === import.meta.url
) {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) {
      console.log(usage());
    } else {
      const result = runBenchmark(options);
      console.log(JSON.stringify(result, null, 2));
      if (result.status === "blocked") process.exitCode = 2;
    }
  } catch (error) {
    console.error(
      error instanceof Error ? error.stack || error.message : error,
    );
    process.exitCode = 1;
  }
}
