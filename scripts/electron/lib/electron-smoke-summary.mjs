import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

export function readElectronSmokeSummary({
  evidenceDir,
  fileExists = existsSync,
  readFile = readFileSync,
  runId,
  summaryPath,
}) {
  let summary;
  try {
    summary = JSON.parse(readFile(summaryPath, "utf8"));
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : String(error),
      summary: null,
      valid: false,
    };
  }

  const failedAssertions = Array.isArray(summary.assertions?.failed)
    ? summary.assertions.failed
    : ["missing-assertions"];
  const tracePath = resolveEvidenceArtifact(
    summary.artifacts?.trace,
    evidenceDir,
  );
  const screenshotPath = resolveEvidenceArtifact(
    summary.artifacts?.screenshot,
    evidenceDir,
  );
  const checks = [
    [summary.candidateRunId === runId, "candidate-run-id"],
    [summary.result === "pass", "result"],
    [failedAssertions.length === 0, "failed-assertions"],
    [Boolean(tracePath && fileExists(tracePath)), "trace"],
    [Boolean(screenshotPath && fileExists(screenshotPath)), "screenshot"],
  ];
  const failedCheck = checks.find(([passed]) => !passed)?.[1] || null;

  return {
    error: failedCheck,
    summary,
    valid: failedCheck === null,
  };
}

export function resolveElectronSmokeExitCode({ childExitCode, summaryResult }) {
  if (summaryResult.valid) {
    return 0;
  }
  return Number.isInteger(childExitCode) && childExitCode > 0
    ? childExitCode
    : 1;
}

function resolveEvidenceArtifact(value, evidenceDir) {
  if (typeof value !== "string" || path.basename(value) !== value) {
    return null;
  }
  return path.join(evidenceDir, value);
}
