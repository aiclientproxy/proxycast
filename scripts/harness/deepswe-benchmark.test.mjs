import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { DEEPSWE_ADAPTER_VERSION } from "./deepswe-adapter-core.mjs";
import {
  aggregateBatch,
  createBatchPlan,
  parseArgs,
} from "./deepswe-benchmark.mjs";
import { createDeepSweSourceFixture } from "./fixtures/deepswe-source.mjs";

const repoRoot = process.cwd();

describe("DeepSWE Core batch benchmark", () => {
  it("plans Smoke 10 and Release 20 with explicit trial counts", () => {
    const fixture = createDeepSweSourceFixture({ repoRoot });
    const options = {
      repoRoot,
      sourceRoot: fixture.sourceRoot,
      resolveSourceCommit: () => fixture.sourceCommit,
    };
    try {
      const smoke = createBatchPlan({
        ...options,
        sliceName: "smoke-10",
        trials: 1,
      });
      const release = createBatchPlan({
        ...options,
        sliceName: "release-20",
        trials: 3,
      });

      expect(smoke).toMatchObject({
        status: "ready",
        taskCount: 10,
        expectedTrialCount: 10,
        trialsPerTask: 1,
      });
      expect(release).toMatchObject({
        status: "ready",
        taskCount: 20,
        expectedTrialCount: 60,
        trialsPerTask: 3,
      });
      expect(new Set(release.tasks.map((task) => task.trialKey)).size).toBe(60);
    } finally {
      fs.rmSync(fixture.sourceRoot, { recursive: true, force: true });
    }
  }, 60_000);

  it("rejects ambiguous modes and non-release trial counts", () => {
    expect(() => parseArgs(["--plan", "--trials", "2"])).toThrow(
      "--trials must be 1 or 3",
    );
    expect(() => parseArgs(["--plan", "--aggregate"])).toThrow(
      "mutually exclusive",
    );
    expect(() => parseArgs(["--run"])).toThrow(
      "requires --allow-live-provider",
    );
    expect(
      parseArgs(["--run", "--allow-live-provider", "--transport", "stdio"]),
    ).toMatchObject({
      mode: "run",
      allowLiveProvider: true,
      transport: "stdio",
    });
  });

  it("excludes historical runs whose source identity is stale", () => {
    const runsRoot = fs.mkdtempSync(
      path.join(os.tmpdir(), "deepswe-stale-run-test-"),
    );
    try {
      const runDir = path.join(runsRoot, "stale-run");
      fs.mkdirSync(runDir, { recursive: true });
      fs.writeFileSync(
        path.join(runDir, "run-context.json"),
        JSON.stringify({
          sourceCommit: "old-source",
          task: {
            id: "happy-dom-abort-pending-body-reads",
            schemaVersion: "1.1",
          },
          executionContract: { adapterVersion: "deepswe-adapter-v5" },
        }),
      );
      fs.writeFileSync(
        path.join(runDir, "adapter-result.json"),
        JSON.stringify({ runId: "stale-run", status: "verified" }),
      );

      const summary = aggregateBatch({
        repoRoot,
        sliceName: "smoke-10",
        runsRoot,
        trials: 1,
      });

      expect(summary.status).toBe("blocked");
      expect(summary.scoreEligible).toBe(false);
      expect(summary.invalidIdentityCount).toBe(1);
      expect(summary.metrics.passAt1).toBeNull();
    } finally {
      fs.rmSync(runsRoot, { recursive: true, force: true });
    }
  }, 60_000);

  it("calculates pass@1, pass@3, pass^3 only from verifier-complete trials", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "deepswe-batch-test-"));
    try {
      const manifestPath = path.join(root, "manifest.json");
      fs.writeFileSync(
        manifestPath,
        JSON.stringify({
          schemaVersion: "lime-deepswe-coding-slice-v2",
          source: {
            commit: "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9",
            taskSchemaVersion: "1.3",
            runner: "datacurve-pier==0.3.1",
          },
          executionContract: {
            adapterVersion: DEEPSWE_ADAPTER_VERSION,
          },
          slices: { "release-20": ["task-a", "task-b"] },
          tasks: [],
        }),
      );
      const runsRoot = path.join(root, "runs");
      for (const [taskId, rewards] of [
        ["task-a", [1, 1, 0]],
        ["task-b", [0, 0, 0]],
      ]) {
        rewards.forEach((reward, index) => {
          const runDir = path.join(runsRoot, `${taskId}-${index + 1}`);
          fs.mkdirSync(runDir, { recursive: true });
          for (const name of ["reward.json", "ctrf.json", "test-stdout.txt"]) {
            fs.writeFileSync(
              path.join(runDir, name),
              name === "reward.json" ? JSON.stringify({ reward }) : name,
            );
          }
          fs.writeFileSync(
            path.join(runDir, "run-context.json"),
            JSON.stringify({
              sourceCommit: "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9",
              task: { id: taskId, schemaVersion: "1.3" },
              executionContract: {
                adapterVersion: DEEPSWE_ADAPTER_VERSION,
              },
            }),
          );
          fs.writeFileSync(
            path.join(runDir, "adapter-result.json"),
            JSON.stringify({
              status: "verified",
              runId: `${taskId}-${index + 1}`,
              patch: { bytes: 100 },
              currentChain: {
                status: "completed",
                startedAt: "2026-08-19T00:00:00Z",
                finishedAt: "2026-08-19T00:00:10Z",
                providerSteps: { usage: { budgetTokens: 42 } },
              },
              verification: {
                evidence: {
                  "reward.json": path.join(runDir, "reward.json"),
                  "ctrf.json": path.join(runDir, "ctrf.json"),
                  "test-stdout.txt": path.join(runDir, "test-stdout.txt"),
                },
              },
            }),
          );
        });
      }
      const summary = aggregateBatch({
        repoRoot,
        manifestPath,
        runsRoot,
        sliceName: "release-20",
        trials: 3,
      });
      expect(summary).toMatchObject({
        status: "complete",
        scoreEligible: true,
        infraValid: true,
        metrics: {
          passAt3: 0.5,
          passPower3: 0,
        },
      });
      expect(summary.metrics.passAt1).toBeCloseTo(1 / 3, 10);
      expect(summary.metrics.budgetTokens.total).toBe(252);
      expect(summary.metrics.wallTimeMs.mean).toBe(10_000);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });
});
