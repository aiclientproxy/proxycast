import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  parseArgs,
  readTrialEvidence,
  runBenchmark,
} from "./deepswe-desktop-benchmark.mjs";

describe("DeepSWE desktop benchmark CLI", () => {
  it("parses preflight and evidence modes", () => {
    expect(parseArgs(["--preflight", "--no-write"])).toMatchObject({
      preflight: true,
      write: false,
    });
    expect(
      parseArgs(["--evidence", ".lime/trials", "--output", ".lime/out.json"]),
    ).toMatchObject({
      evidencePath: path.resolve(".lime/trials"),
      output: path.resolve(".lime/out.json"),
    });
    expect(() => parseArgs([])).toThrow(
      "either --preflight or --evidence is required",
    );
    expect(() => parseArgs(["--unknown"])).toThrow(
      "Unknown argument: --unknown",
    );
  });

  it("runs the source preflight from the repository fact source", () => {
    const result = runBenchmark(
      { preflight: true, evidencePath: null, write: false },
      process.cwd(),
    );
    expect(result.status).toBe("pass");
    expect(result.taskCount).toBe(5);
    expect(result.checks).toHaveLength(53);
  });

  it("discovers only desktop trial evidence recursively", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "desktop-evidence-"));
    try {
      fs.mkdirSync(path.join(root, "nested"));
      fs.writeFileSync(
        path.join(root, "nested", "trial.json"),
        JSON.stringify({
          schemaVersion: "deepswe-desktop-trial-v1",
          taskId: "task-1",
        }),
      );
      fs.writeFileSync(
        path.join(root, "summary.json"),
        JSON.stringify({ schemaVersion: "deepswe-desktop-suite-v1" }),
      );
      fs.writeFileSync(path.join(root, "ignored.txt"), "not json");
      expect(readTrialEvidence(root)).toMatchObject([
        {
          taskId: "task-1",
          evidencePath: path.join(root, "nested", "trial.json"),
        },
      ]);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});
