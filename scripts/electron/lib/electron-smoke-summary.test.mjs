import { describe, expect, it } from "vitest";

import {
  readElectronSmokeSummary,
  resolveElectronSmokeExitCode,
} from "./electron-smoke-summary.mjs";

const baseSummary = {
  candidateRunId: "run-1",
  result: "pass",
  assertions: { failed: [] },
  artifacts: { trace: "trace-summary.json", screenshot: "screen.png" },
};

function readSummary(summary, files = ["trace-summary.json", "screen.png"]) {
  return readElectronSmokeSummary({
    evidenceDir: "C:/evidence",
    fileExists: (filePath) => files.includes(filePath.split(/[\\/]/).pop()),
    readFile: () => JSON.stringify(summary),
    runId: "run-1",
    summaryPath: "C:/evidence/summary.json",
  });
}

describe("Electron smoke summary", () => {
  it("accepts a complete pass even when the child process exit code is unavailable", () => {
    const summaryResult = readSummary(baseSummary);

    expect(summaryResult).toMatchObject({ valid: true });
    expect(
      resolveElectronSmokeExitCode({ childExitCode: 1, summaryResult }),
    ).toBe(0);
  });

  it("rejects a summary for another run", () => {
    expect(
      readSummary({ ...baseSummary, candidateRunId: "stale-run" }),
    ).toMatchObject({
      valid: false,
      error: "candidate-run-id",
    });
  });

  it("rejects failed assertions and missing artifacts", () => {
    const failedSummary = readSummary({
      ...baseSummary,
      assertions: { failed: ["workbenchReady"] },
    });
    expect(failedSummary).toMatchObject({
      valid: false,
      error: "failed-assertions",
    });
    expect(
      resolveElectronSmokeExitCode({
        childExitCode: 0,
        summaryResult: failedSummary,
      }),
    ).toBe(1);
    expect(readSummary(baseSummary, ["trace-summary.json"])).toMatchObject({
      valid: false,
      error: "screenshot",
    });
  });
});
