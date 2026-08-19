import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import {
  DEEPSWE_PIER_PACKAGE,
  DEEPSWE_SOURCE_COMMIT,
  DEEPSWE_TASK_SCHEMA_VERSION,
  loadTaskDefinition,
} from "./deepswe-adapter-core.mjs";

export const DEEPSWE_DESKTOP_MANIFEST_PATH =
  "internal/test/deepswe-desktop-smoke-v1.json";
export const DEEPSWE_DESKTOP_TRIAL_SCHEMA = "deepswe-desktop-trial-v1";
export const DEEPSWE_DESKTOP_SUITE_SCHEMA = "deepswe-desktop-suite-v1";

const TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "interrupted",
  "cancelled",
  "canceled",
  "aborted",
]);
const REQUIRED_TOOL_PHASES = ["read", "search", "patch", "test"];
const TOOL_PHASES = new Map([
  ["Read", "read"],
  ["Glob", "search"],
  ["Grep", "search"],
  ["apply_patch", "patch"],
  ["exec_command", "test"],
]);
const REQUIRED_VERIFIER_ARTIFACTS = [
  "reward.json",
  "ctrf.json",
  "test-stdout.txt",
];

function isRecord(value) {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function normalizeString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeStatus(value) {
  return normalizeString(value).toLowerCase();
}

function isSha256(value) {
  return /^[a-f0-9]{64}$/u.test(normalizeString(value));
}

function uniqueStrings(value) {
  return Array.isArray(value)
    ? [...new Set(value.map(normalizeString).filter(Boolean))]
    : [];
}

export function sha256(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

export function isPathInside(parentPath, candidatePath) {
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

export function loadDesktopManifest(
  repoRoot,
  manifestPath = DEEPSWE_DESKTOP_MANIFEST_PATH,
) {
  const absolutePath = path.resolve(repoRoot, manifestPath);
  const manifest = JSON.parse(fs.readFileSync(absolutePath, "utf8"));
  validateDesktopManifest(manifest);
  return { absolutePath, manifest };
}

export function validateDesktopManifest(manifest) {
  if (manifest?.schemaVersion !== "lime-deepswe-desktop-smoke-v1") {
    throw new Error(
      `unsupported DeepSWE desktop manifest: ${manifest?.schemaVersion}`,
    );
  }
  const tasks = Array.isArray(manifest.tasks) ? manifest.tasks : [];
  if (tasks.length !== 5) {
    throw new Error(
      `Desktop Smoke 5 must contain 5 tasks, got ${tasks.length}`,
    );
  }
  const ids = uniqueStrings(tasks.map((task) => task?.id));
  const languages = uniqueStrings(tasks.map((task) => task?.language));
  if (ids.length !== tasks.length) {
    throw new Error("Desktop Smoke 5 task ids must be unique");
  }
  const expectedLanguages = uniqueStrings(
    manifest?.suiteRequirements?.languages,
  ).sort();
  if (JSON.stringify(languages.sort()) !== JSON.stringify(expectedLanguages)) {
    throw new Error(
      `Desktop Smoke 5 language coverage mismatch: ${languages.join(", ")}`,
    );
  }
  for (const task of tasks) {
    if (
      !normalizeString(task.repository) ||
      !/^[a-f0-9]{40}$/u.test(normalizeString(task.baseCommit)) ||
      !normalizeString(task.instructionPath) ||
      !normalizeString(task.taskTomlPath)
    ) {
      throw new Error(`Desktop task metadata incomplete: ${task.id}`);
    }
  }
  return manifest;
}

export function preflightDesktopManifest({ repoRoot, manifest }) {
  const checks = [];
  const add = (name, passed, detail = null) =>
    checks.push({ name, passed: passed === true, detail });
  const sourceRoot = path.resolve(
    repoRoot,
    path.dirname(manifest.taskSourceRoot),
  );
  let sourceHead = "unavailable";
  try {
    sourceHead = execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: sourceRoot,
      encoding: "utf8",
    }).trim();
  } catch {
    // The source identity check below remains fail-closed.
  }
  add(
    "source-commit",
    sourceHead === DEEPSWE_SOURCE_COMMIT &&
      manifest.sourceCommit === DEEPSWE_SOURCE_COMMIT,
    sourceHead,
  );
  add(
    "source-schema",
    manifest.sourceSchemaVersion === DEEPSWE_TASK_SCHEMA_VERSION,
    manifest.sourceSchemaVersion,
  );
  add("source-pier", manifest.runner === DEEPSWE_PIER_PACKAGE, manifest.runner);
  for (const task of manifest.tasks) {
    const instructionPath = path.resolve(repoRoot, task.instructionPath);
    const taskTomlPath = path.resolve(repoRoot, task.taskTomlPath);
    add(
      `${task.id}:instruction`,
      fs.existsSync(instructionPath),
      task.instructionPath,
    );
    add(`${task.id}:task-toml`, fs.existsSync(taskTomlPath), task.taskTomlPath);
    if (fs.existsSync(instructionPath)) {
      add(
        `${task.id}:instruction-nonempty`,
        fs.readFileSync(instructionPath, "utf8").trim().length > 0,
      );
    }
    if (fs.existsSync(taskTomlPath)) {
      let sourceTask = null;
      try {
        sourceTask = loadTaskDefinition({
          repoRoot,
          sourceRoot,
          taskId: task.id,
        });
      } catch {
        // The structured checks below remain fail-closed.
      }
      add(
        `${task.id}:base-commit`,
        sourceTask?.baseCommit === task.baseCommit,
        task.baseCommit,
      );
      add(
        `${task.id}:schema`,
        sourceTask?.schemaVersion === DEEPSWE_TASK_SCHEMA_VERSION,
        sourceTask?.schemaVersion || "missing",
      );
      add(
        `${task.id}:agent-network`,
        sourceTask?.agent.networkMode === "no-network",
        sourceTask?.agent.networkMode || "missing",
      );
      add(
        `${task.id}:verifier-network`,
        sourceTask?.verifier.networkMode === "no-network",
        sourceTask?.verifier.networkMode || "missing",
      );
      add(
        `${task.id}:separate-verifier`,
        sourceTask?.verifier.environmentMode === "separate",
        sourceTask?.verifier.environmentMode || "missing",
      );
      const collect = sourceTask?.verifier.collect || [];
      add(
        `${task.id}:collect`,
        collect.length === 1 &&
          collect[0].command.includes(
            `git diff --binary ${task.baseCommit} HEAD`,
          ) &&
          collect[0].command.includes("> /logs/artifacts/model.patch"),
        collect[0]?.command || "missing",
      );
      add(
        `${task.id}:pre-artifacts-deleted`,
        !fs.existsSync(
          path.join(path.dirname(taskTomlPath), "pre_artifacts.sh"),
        ),
      );
    }
  }
  return {
    schemaVersion: "deepswe-desktop-preflight-v1",
    taskCount: manifest.tasks.length,
    status: checks.every((check) => check.passed) ? "pass" : "fail",
    checks,
  };
}

function taskById(manifest, taskId) {
  return manifest.tasks.find((task) => task.id === taskId) ?? null;
}

function completedToolPhases(toolLifecycle) {
  const phases = new Set();
  for (const item of Array.isArray(toolLifecycle) ? toolLifecycle : []) {
    if (normalizeStatus(item?.status) !== "completed") continue;
    const phase = TOOL_PHASES.get(normalizeString(item?.name));
    if (phase) phases.add(phase);
  }
  return phases;
}

function verifierArtifactNames(verifier) {
  return uniqueStrings(
    Array.isArray(verifier?.artifacts)
      ? verifier.artifacts.map((artifact) =>
          typeof artifact === "string"
            ? path.basename(artifact)
            : artifact?.name,
        )
      : [],
  );
}

function buildTrialAssertions({ evidence, manifest, repoRoot }) {
  const task = taskById(manifest, normalizeString(evidence?.taskId));
  const toolPhases = completedToolPhases(evidence?.toolLifecycle);
  const changedFiles = uniqueStrings(evidence?.changedFiles);
  const workspace = normalizeString(evidence?.workspace);
  const patchSha256 = normalizeString(evidence?.patchSha256);
  const verifierStatus = normalizeStatus(evidence?.verifier?.status);
  const verifierPatchSha256 = normalizeString(evidence?.verifier?.patchSha256);
  const verifierArtifacts = verifierArtifactNames(evidence?.verifier);
  const terminalStatus = normalizeStatus(evidence?.readModel?.terminalStatus);
  const identity = evidence?.identity ?? {};
  const gateIdentity = evidence?.gui?.identity ?? {};
  const instructionPath = task
    ? path.resolve(repoRoot, task.instructionPath)
    : "";
  const instructionSha256 =
    instructionPath && fs.existsSync(instructionPath)
      ? sha256(fs.readFileSync(instructionPath))
      : "";

  return {
    schemaCurrent: evidence?.schemaVersion === DEEPSWE_DESKTOP_TRIAL_SCHEMA,
    taskSelected: Boolean(task),
    taskMetadataMatches:
      Boolean(task) &&
      evidence?.language === task.language &&
      evidence?.repository === task.repository &&
      evidence?.sourceCommit === task.baseCommit,
    originalInstructionMatches:
      Boolean(instructionSha256) &&
      normalizeString(evidence?.instructionSha256) === instructionSha256,
    workspaceOutsideLime:
      Boolean(workspace) && !isPathInside(repoRoot, workspace),
    canonicalIdentity:
      Boolean(
        normalizeString(identity.sessionId) &&
        normalizeString(identity.threadId) &&
        normalizeString(identity.turnId),
      ) &&
      normalizeString(gateIdentity.sessionId) ===
        normalizeString(identity.sessionId) &&
      normalizeString(gateIdentity.threadId) ===
        normalizeString(identity.threadId) &&
      normalizeString(gateIdentity.turnId) === normalizeString(identity.turnId),
    nonEmptyCandidatePatch:
      isSha256(patchSha256) &&
      Number(evidence?.patchBytes) > 0 &&
      changedFiles.length > 0,
    requiredToolPhasesCompleted: REQUIRED_TOOL_PHASES.every((phase) =>
      toolPhases.has(phase),
    ),
    testCommandPassed:
      evidence?.testResult?.status === "pass" &&
      evidence?.testResult?.exitCode === 0 &&
      evidence?.testResult?.outputVisible === true,
    realElectronHost: evidence?.gui?.electron === true,
    preloadInvokeBridge: evidence?.gui?.preloadInvokeBridge === true,
    appServerHandleJsonLines:
      evidence?.bridge?.appServerHandleJsonLinesSeen === true,
    terminalReadModel:
      TERMINAL_STATUSES.has(terminalStatus) &&
      normalizeString(evidence?.readModel?.threadId) ===
        normalizeString(identity.threadId) &&
      normalizeString(evidence?.readModel?.turnId) ===
        normalizeString(identity.turnId),
    visibleTerminal: evidence?.gui?.terminalVisible === true,
    visibleToolLifecycle: evidence?.gui?.toolLifecycleVisible === true,
    visibleDiffArtifact: evidence?.gui?.diffArtifactVisible === true,
    artifactContentAvailable:
      evidence?.gui?.artifactPreview?.status === "pass" &&
      evidence?.gui?.artifactPreview?.contentVisible === true &&
      evidence?.gui?.artifactPreview?.unavailableErrorVisible === false &&
      evidence?.gui?.artifactPreview?.artifactReadSeen === true,
    sessionReopen:
      evidence?.recovery?.sessionReopen?.status === "pass" &&
      normalizeString(evidence?.recovery?.sessionReopen?.sessionId) ===
        normalizeString(identity.sessionId),
    productionMockFallbackZero: evidence?.bridge?.mockFallbackHitCount === 0,
    invokeErrorsZero: evidence?.bridge?.invokeErrorCount === 0,
    consoleErrorsZero: evidence?.gui?.consoleErrorCount === 0,
    pageErrorsZero: evidence?.gui?.pageErrorCount === 0,
    verifierArtifactsComplete:
      verifierStatus === "pass" &&
      REQUIRED_VERIFIER_ARTIFACTS.every((name) =>
        verifierArtifacts.includes(name),
      ),
    verifierPatchMatches:
      verifierStatus === "pass" &&
      isSha256(verifierPatchSha256) &&
      verifierPatchSha256 === patchSha256,
  };
}

export function evaluateDesktopTrial({ evidence, manifest, repoRoot }) {
  const assertions = buildTrialAssertions({ evidence, manifest, repoRoot });
  const gateBAssertionNames = [
    "schemaCurrent",
    "taskSelected",
    "taskMetadataMatches",
    "originalInstructionMatches",
    "workspaceOutsideLime",
    "canonicalIdentity",
    "nonEmptyCandidatePatch",
    "requiredToolPhasesCompleted",
    "testCommandPassed",
    "realElectronHost",
    "preloadInvokeBridge",
    "appServerHandleJsonLines",
    "terminalReadModel",
    "visibleTerminal",
    "visibleToolLifecycle",
    "visibleDiffArtifact",
    "artifactContentAvailable",
    "sessionReopen",
    "productionMockFallbackZero",
    "invokeErrorsZero",
    "consoleErrorsZero",
    "pageErrorsZero",
  ];
  const failedGateBAssertions = gateBAssertionNames.filter(
    (name) => assertions[name] !== true,
  );
  const gateBPass = failedGateBAssertions.length === 0;
  const verifierPass =
    assertions.verifierArtifactsComplete && assertions.verifierPatchMatches;
  const liveTrial = evidence?.trialKind === "live_deepswe";
  const desktopCodingPass = liveTrial && gateBPass && verifierPass;
  const claimedGateBPass = evidence?.gateB?.pass;
  const claimedDesktopCodingPass = evidence?.desktopCodingPass;
  const claimConsistent =
    (claimedGateBPass == null || claimedGateBPass === gateBPass) &&
    (claimedDesktopCodingPass == null ||
      claimedDesktopCodingPass === desktopCodingPass);

  return {
    schemaVersion: "deepswe-desktop-trial-verdict-v1",
    taskId: normalizeString(evidence?.taskId) || null,
    trialId: normalizeString(evidence?.trialId) || null,
    trialKind: normalizeString(evidence?.trialKind) || null,
    gateBPass,
    verifierPass,
    desktopCodingPass,
    claimConsistent,
    valid: claimConsistent,
    assertions,
    failedGateBAssertions,
    failedVerifierAssertions: [
      ...(assertions.verifierArtifactsComplete
        ? []
        : ["verifierArtifactsComplete"]),
      ...(assertions.verifierPatchMatches ? [] : ["verifierPatchMatches"]),
    ],
  };
}

function recoveryCoverage(evidenceList) {
  const hasPass = (field) =>
    evidenceList.some(
      (evidence) => evidence?.recovery?.[field]?.status === "pass",
    );
  return {
    cancel_no_ghost_write: hasPass("cancelNoGhostWrite"),
    approval_resume: hasPass("approvalResume"),
    session_reopen: hasPass("sessionReopen"),
  };
}

export function evaluateDesktopSuite({ evidenceList, manifest, repoRoot }) {
  const trials = evidenceList.map((evidence) => ({
    evidence,
    verdict: evaluateDesktopTrial({ evidence, manifest, repoRoot }),
  }));
  const selectedTaskIds = new Set(manifest.tasks.map((task) => task.id));
  const observedTaskIds = new Set(
    trials.map(({ verdict }) => verdict.taskId).filter(Boolean),
  );
  const coverage = recoveryCoverage(evidenceList);
  const liveTrials = trials.filter(
    ({ verdict }) => verdict.trialKind === "live_deepswe",
  );
  const controlledTrials = trials.filter(
    ({ verdict }) => verdict.trialKind === "controlled_product_smoke",
  );
  const assertions = {
    allTasksObserved: [...selectedTaskIds].every((id) =>
      observedTaskIds.has(id),
    ),
    noUnknownTasks: [...observedTaskIds].every((id) => selectedTaskIds.has(id)),
    allClaimsConsistent: trials.every(({ verdict }) => verdict.claimConsistent),
    controlledGateBComplete:
      controlledTrials.length === manifest.tasks.length &&
      controlledTrials.every(({ verdict }) => verdict.gateBPass),
    liveTrialPerTask: manifest.tasks.every((task) =>
      liveTrials.some(({ verdict }) => verdict.taskId === task.id),
    ),
    recoveryCoverageComplete: Object.values(coverage).every(Boolean),
    allLiveDesktopCodingPass:
      liveTrials.length >= manifest.tasks.length &&
      liveTrials.every(({ verdict }) => verdict.desktopCodingPass),
  };
  const desktopCodingPass =
    assertions.allTasksObserved &&
    assertions.noUnknownTasks &&
    assertions.allClaimsConsistent &&
    assertions.liveTrialPerTask &&
    assertions.recoveryCoverageComplete &&
    assertions.allLiveDesktopCodingPass;
  return {
    schemaVersion: DEEPSWE_DESKTOP_SUITE_SCHEMA,
    status: desktopCodingPass
      ? "pass"
      : assertions.controlledGateBComplete
        ? "product_path_only"
        : "incomplete",
    desktopCodingPass,
    taskCount: manifest.tasks.length,
    trialCount: trials.length,
    controlledTrialCount: controlledTrials.length,
    liveTrialCount: liveTrials.length,
    recoveryCoverage: coverage,
    assertions,
    failedAssertions: Object.entries(assertions)
      .filter(([, passed]) => passed !== true)
      .map(([name]) => name),
    trials: trials.map(({ verdict }) => verdict),
  };
}
