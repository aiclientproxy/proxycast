import fs from "node:fs";
import { describe, expect, it } from "vitest";

const manifest = JSON.parse(
  fs.readFileSync("internal/test/deepswe-coding-slice-v2.json", "utf8"),
);

function languageCounts(taskIds) {
  const tasksById = new Map(manifest.tasks.map((task) => [task.id, task]));
  return taskIds.reduce((counts, id) => {
    const task = tasksById.get(id);
    expect(task, `missing task metadata for ${id}`).toBeDefined();
    counts[task.language] = (counts[task.language] ?? 0) + 1;
    return counts;
  }, {});
}

describe("DeepSWE coding slice v2", () => {
  it("pins the source and current-chain execution contract", () => {
    expect(manifest.schemaVersion).toBe("lime-deepswe-coding-slice-v2");
    expect(manifest.status).toBe("diagnostic_true_runs_blocked");
    expect(manifest.source).toMatchObject({
      commit: "3cda4081fed96103a6395de39c85e9b20275e307",
      taskSchemaVersion: "1.1",
      taskCount: 113,
    });
    expect(manifest.executionContract.agentPath).toBe(
      "Lime App Server JSON-RPC current chain",
    );
    expect(manifest.executionContract.adapterCommand).toBe(
      "npm run harness:deepswe:run",
    );
    expect(manifest.executionContract.workspace).toContain(
      "outside the Lime repository",
    );
    expect(manifest.executionContract).toMatchObject({
      adapterVersion: "deepswe-current-chain-adapter-v5",
      primaryModel:
        "custom-637ea2d5-e430-43de-86de-39c5f1735438 / agnes-2.0-flash",
      comparisonModel: "custom-1ae93b42-e57f-4a83-ac6e-3f5275a7b376 / gpt-5.5",
      providerBudget: {
        maxProviderSteps: 32,
        tokenBudget: 500_000,
        tokenFormula: "max(0,input_tokens-cached_input_tokens)+output_tokens",
        enforcementOwner:
          "agent-runtime reply loop before tool execution and next sampling",
      },
      requiredEvidence: {
        agent: [
          "run-context.json",
          "trajectory.json",
          "provider-steps.json",
          "thread-turn-item.json",
          "tool-lifecycle.json",
          "patch.diff",
          "failure-classification.json",
        ],
        verifier: ["reward.json", "ctrf.json", "test-stdout.txt"],
        blockedVerifier: [
          "verifier-prerequisites.json",
          "verifier-failure-classification.json",
        ],
      },
    });
    expect(manifest.executionContract.invalidTrialConditions).toContain(
      "task dependency resolution can reach Lime node_modules",
    );
    expect(manifest.executionContract.invalidTrialConditions).toContain(
      "task git repository exposes refs or commits after the pinned base",
    );
    expect(manifest.executionContract.currentBlockers).toEqual([
      "Agnes produced no candidate in TS/Go/Rust diagnostics, including runtime-capped runs with a complete provider tool catalog",
      "The local Pier wrapper cannot import its deleted editable package source",
      "No local Docker, Podman, nerdctl, or Colima container runtime is available",
    ]);
  });

  it("keeps Smoke 10 unique, balanced, and contained in Release 20", () => {
    const smoke = manifest.slices["smoke-10"];
    const release = manifest.slices["release-20"];

    expect(smoke).toHaveLength(10);
    expect(new Set(smoke).size).toBe(10);
    expect(release).toHaveLength(20);
    expect(new Set(release).size).toBe(20);
    expect(smoke.every((id) => release.includes(id))).toBe(true);
    expect(languageCounts(smoke)).toEqual({
      typescript: 2,
      go: 2,
      python: 2,
      rust: 2,
      javascript: 2,
    });
  });

  it("keeps Release 20 metadata complete and language-diverse", () => {
    const release = manifest.slices["release-20"];
    const metadataIds = manifest.tasks.map((task) => task.id);

    expect(manifest.tasks).toHaveLength(20);
    expect(new Set(metadataIds).size).toBe(20);
    expect(new Set(metadataIds)).toEqual(new Set(release));
    expect(
      manifest.tasks.every(
        (task) =>
          task.repository.length > 0 &&
          Array.isArray(task.focus) &&
          task.focus.length >= 2,
      ),
    ).toBe(true);
    expect(languageCounts(release)).toEqual({
      typescript: 6,
      go: 4,
      python: 5,
      rust: 3,
      javascript: 2,
    });
  });
});
