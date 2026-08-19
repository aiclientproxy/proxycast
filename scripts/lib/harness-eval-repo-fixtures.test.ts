import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { afterEach, describe, expect, it } from "vitest";

const repoRoot = process.cwd();
const tempRoots: string[] = [];

function createTempRoot() {
  const tempRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "lime-harness-repo-fixtures-"),
  );
  tempRoots.push(tempRoot);
  return tempRoot;
}

function runHarnessEval(args: string[]) {
  const output = execFileSync(
    process.execPath,
    [path.join(repoRoot, "scripts/harness/eval-runner.mjs"), ...args],
    {
      cwd: repoRoot,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  return JSON.parse(output);
}

afterEach(() => {
  while (tempRoots.length > 0) {
    const tempRoot = tempRoots.pop();
    if (tempRoot) {
      fs.rmSync(tempRoot, { recursive: true, force: true });
    }
  }
});

describe("Harness repo fixtures", () => {
  it("应固定 current 与 degraded observability 样本职责", () => {
    const workspaceRoot = createTempRoot();
    const summary = runHarnessEval([
      "--format",
      "json",
      "--manifest",
      "internal/test/harness-evals.manifest.json",
      "--workspace-root",
      workspaceRoot,
      "--no-strict",
    ]);

    const repoFixtureSuite = summary.suites.find(
      (entry: { id: string }) => entry.id === "repo-fixtures",
    );
    expect(repoFixtureSuite).toBeTruthy();
    expect(repoFixtureSuite.stats.caseCount).toBe(2);
    expect(repoFixtureSuite.stats.readyCount).toBe(2);
    expect(summary.totals.observabilityGapCaseCount).toBe(1);
    expect(summary.totals.currentObservabilityGapCaseCount).toBe(0);
    expect(summary.totals.degradedObservabilityGapCaseCount).toBe(1);
    expect(summary.totals.currentRecoveredVerificationCaseCount).toBe(3);

    const currentCase = repoFixtureSuite.cases.find(
      (entry: { caseId: string }) =>
        entry.caseId === "fixture-minimal-pending-request",
    );
    expect(currentCase).toBeTruthy();
    expect(currentCase.status).toBe("ready");
    expect(currentCase.observabilityGapCount).toBe(0);
    expect(currentCase.observabilitySignals).toContain(
      "requestTelemetry:exported",
    );
    expect(currentCase.observabilitySignals).toContain(
      "artifactValidator:exported",
    );
    expect(currentCase.observabilitySignals).toContain(
      "browserVerification:exported",
    );
    expect(currentCase.observabilitySignals).toContain("guiSmoke:exported");
    expect(currentCase.observabilityVerificationOutcomes).toContain(
      "artifactValidator:issues_present",
    );
    expect(currentCase.observabilityVerificationOutcomes).toContain(
      "artifactValidator:repaired",
    );
    expect(currentCase.observabilityVerificationOutcomes).toContain(
      "browserVerification:success",
    );
    expect(currentCase.observabilityVerificationOutcomes).toContain(
      "guiSmoke:passed",
    );

    const degradedCase = repoFixtureSuite.cases.find(
      (entry: { caseId: string }) =>
        entry.caseId === "fixture-minimal-observability-gap",
    );
    expect(degradedCase).toBeTruthy();
    expect(degradedCase.status).toBe("ready");
    expect(degradedCase.observabilityGapCount).toBe(2);
    expect(degradedCase.observabilitySignals).toContain(
      "requestTelemetry:known_gap",
    );
    expect(degradedCase.observabilitySignals).toContain(
      "artifactValidator:known_gap",
    );

    const gapCaseIds = repoFixtureSuite.cases
      .filter(
        (entry: { observabilityGapCount: number }) =>
          entry.observabilityGapCount > 0,
      )
      .map((entry: { caseId: string }) => entry.caseId);
    expect(gapCaseIds).toEqual(["fixture-minimal-observability-gap"]);

    const observabilityBreakdownNames =
      summary.breakdowns.observabilitySignals.map(
        (entry: { name: string }) => entry.name,
      );
    expect(observabilityBreakdownNames).toContain("requestTelemetry:exported");
    expect(observabilityBreakdownNames).toContain("requestTelemetry:known_gap");
    expect(observabilityBreakdownNames).toContain("artifactValidator:exported");
    expect(observabilityBreakdownNames).toContain(
      "artifactValidator:known_gap",
    );

    const verificationOutcomeBreakdownNames =
      summary.breakdowns.observabilityVerificationOutcomes.map(
        (entry: { name: string }) => entry.name,
      );
    expect(verificationOutcomeBreakdownNames).toContain(
      "artifactValidator:issues_present",
    );
    expect(verificationOutcomeBreakdownNames).toContain(
      "artifactValidator:repaired",
    );
    expect(verificationOutcomeBreakdownNames).toContain(
      "browserVerification:success",
    );
    expect(verificationOutcomeBreakdownNames).toContain("guiSmoke:passed");

    const currentVerificationOutcomeBreakdownNames =
      summary.breakdowns.currentObservabilityVerificationOutcomes.map(
        (entry: { name: string }) => entry.name,
      );
    expect(currentVerificationOutcomeBreakdownNames).toContain(
      "artifactValidator:issues_present",
    );
    expect(currentVerificationOutcomeBreakdownNames).toContain(
      "browserVerification:success",
    );

    const currentRecoveredVerificationOutcomeBreakdownNames =
      summary.breakdowns.currentRecoveredObservabilityVerificationOutcomes.map(
        (entry: { name: string }) => entry.name,
      );
    expect(currentRecoveredVerificationOutcomeBreakdownNames).toContain(
      "artifactValidator:repaired",
    );
    expect(currentRecoveredVerificationOutcomeBreakdownNames).toContain(
      "browserVerification:success",
    );
    expect(currentRecoveredVerificationOutcomeBreakdownNames).toContain(
      "guiSmoke:passed",
    );

    expect(
      summary.breakdowns.degradedObservabilityVerificationOutcomes,
    ).toEqual([]);
  }, 20_000);
});
