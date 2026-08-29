import { execFileSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function sha256File(filePath) {
  return crypto
    .createHash("sha256")
    .update(fs.readFileSync(filePath))
    .digest("hex");
}

function normalizeString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function verifierFailureNames(result) {
  const evidencePath = result?.verification?.evidence?.["test-stdout.txt"];
  if (!evidencePath || !fs.existsSync(evidencePath)) {
    return [];
  }
  const failureNames = fs
    .readFileSync(evidencePath, "utf8")
    .split("\n")
    .map((line) => line.match(/^\s*FAIL\s+\[[^\]]+\]\s+(.+)$/)?.[1]?.trim())
    .filter(Boolean);
  return [...new Set(failureNames)].slice(0, 12);
}

export function loadCandidateContinuation({
  runDir,
  task,
  providerPreference,
  modelPreference,
}) {
  const absoluteRunDir = path.resolve(runDir);
  const contextPath = path.join(absoluteRunDir, "run-context.json");
  const resultPath = path.join(absoluteRunDir, "adapter-result.json");
  const patchPath = path.join(absoluteRunDir, "patch.diff");
  for (const requiredPath of [contextPath, resultPath, patchPath]) {
    if (!fs.existsSync(requiredPath)) {
      throw new Error(
        `DeepSWE candidate continuation evidence missing: ${requiredPath}`,
      );
    }
  }

  const context = readJson(contextPath);
  const result = readJson(resultPath);
  const sourceTaskId = normalizeString(context?.task?.id);
  if (sourceTaskId !== task.id) {
    throw new Error(
      `DeepSWE candidate continuation task mismatch: expected=${task.id} actual=${sourceTaskId || "missing"}`,
    );
  }
  const sourceBaseCommit = normalizeString(context?.task?.baseCommit);
  if (sourceBaseCommit !== task.baseCommit) {
    throw new Error(
      `DeepSWE candidate continuation base mismatch: expected=${task.baseCommit} actual=${sourceBaseCommit || "missing"}`,
    );
  }

  const patchBytes = fs.statSync(patchPath).size;
  if (patchBytes < 1 || Number(result?.patch?.bytes) < 1) {
    throw new Error("DeepSWE candidate continuation patch is empty");
  }
  const previousProvider =
    result?.currentChain?.provider || context?.continuation?.provider || {};
  const previousProviderId = normalizeString(
    previousProvider.providerPreference,
  );
  const previousModel = normalizeString(previousProvider.modelPreference);
  if (providerPreference && previousProviderId !== providerPreference) {
    throw new Error(
      `DeepSWE candidate continuation provider mismatch: expected=${providerPreference} actual=${previousProviderId || "missing"}`,
    );
  }
  if (modelPreference && previousModel !== modelPreference) {
    throw new Error(
      `DeepSWE candidate continuation model mismatch: expected=${modelPreference} actual=${previousModel || "missing"}`,
    );
  }

  const evidence = {
    schemaVersion: "deepswe-candidate-continuation-v1",
    sourceRunId: normalizeString(result.runId) || path.basename(absoluteRunDir),
    sourceRunDir: absoluteRunDir,
    sourceTaskId,
    sourceStatus: normalizeString(result.status),
    sourceCurrentChainStatus: normalizeString(result?.currentChain?.status),
    sourceTerminalStatus:
      normalizeString(result?.currentChain?.terminalStatus) || null,
    sourcePatch: {
      path: patchPath,
      bytes: patchBytes,
      sha256: sha256File(patchPath),
    },
    provider: {
      providerPreference: previousProviderId,
      modelPreference: previousModel,
    },
    verifier: {
      status: normalizeString(result?.verification?.status) || null,
      reward:
        typeof result?.verification?.reward === "number"
          ? result.verification.reward
          : null,
      failureNames: verifierFailureNames(result),
    },
  };
  return { patchPath, evidence };
}

export function applyCandidateContinuation({ workspaceDir, continuation }) {
  execFileSync(
    "git",
    ["apply", "--binary", "--check", continuation.patchPath],
    { cwd: workspaceDir, stdio: ["ignore", "pipe", "pipe"] },
  );
  execFileSync("git", ["apply", "--binary", continuation.patchPath], {
    cwd: workspaceDir,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const status = execFileSync("git", ["status", "--short"], {
    cwd: workspaceDir,
    encoding: "utf8",
  }).trim();
  if (!status) {
    throw new Error(
      "DeepSWE candidate continuation produced no workspace diff",
    );
  }
  return { ...continuation.evidence, workspaceStatus: status };
}

export function candidateContinuationInstruction(instruction, evidence) {
  const verifierFeedback = evidence.verifier?.failureNames?.length
    ? ` The previous Pier verifier reward was ${evidence.verifier.reward ?? "unknown"}; failing tests were: ${evidence.verifier.failureNames.join(", ")}.`
    : "";
  return `CONTINUATION: A previous live attempt for this same task left a candidate patch in the working tree (source run ${evidence.sourceRunId}, SHA-256 ${evidence.sourcePatch.sha256}).${verifierFeedback} Start from the current diff. Do not restart broad repository exploration and do not run unbounded scans of the home directory, dependency registries, or the whole filesystem. Inspect only files required by the diff or a concrete test failure, finish the implementation, run the narrow relevant tests, and end the turn as soon as the candidate is ready. Do not replace it with a reference solution.\n\nOriginal task:\n${instruction}`;
}
