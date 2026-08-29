import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  applyCandidateContinuation,
  candidateContinuationInstruction,
  loadCandidateContinuation,
} from "./deepswe-candidate-continuation.mjs";

const temporaryRoots = [];

function temporaryRoot() {
  const root = fs.mkdtempSync(
    path.join(os.tmpdir(), "deepswe-continuation-test-"),
  );
  temporaryRoots.push(root);
  return root;
}

function git(cwd, args) {
  return execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
}

function createFixture() {
  const root = temporaryRoot();
  const workspaceDir = path.join(root, "workspace");
  const runDir = path.join(root, "run");
  fs.mkdirSync(workspaceDir);
  fs.mkdirSync(runDir);
  git(workspaceDir, ["init"]);
  git(workspaceDir, ["config", "user.name", "DeepSWE Test"]);
  git(workspaceDir, ["config", "user.email", "deepswe@localhost"]);
  fs.writeFileSync(path.join(workspaceDir, "file.txt"), "before\n");
  git(workspaceDir, ["add", "file.txt"]);
  git(workspaceDir, ["commit", "-m", "baseline"]);
  const baseCommit = git(workspaceDir, ["rev-parse", "HEAD"]);
  fs.writeFileSync(path.join(workspaceDir, "file.txt"), "after\n");
  const patch = execFileSync("git", ["diff", "--binary", baseCommit], {
    cwd: workspaceDir,
  });
  const patchPath = path.join(runDir, "patch.diff");
  fs.writeFileSync(patchPath, patch);
  fs.writeFileSync(
    path.join(runDir, "run-context.json"),
    `${JSON.stringify({ task: { id: "task-1", baseCommit } })}\n`,
  );
  fs.writeFileSync(
    path.join(runDir, "adapter-result.json"),
    `${JSON.stringify({
      runId: "source-run",
      status: "failed",
      currentChain: {
        status: "timeout",
        terminalStatus: "interrupted",
        provider: {
          providerPreference: "provider-1",
          modelPreference: "model-1",
        },
      },
      patch: { bytes: patch.length },
      verification: {
        status: "verified_with_product_failure",
        reward: 0,
        evidence: {
          "test-stdout.txt": path.join(runDir, "test-stdout.txt"),
        },
      },
    })}\n`,
  );
  fs.writeFileSync(
    path.join(runDir, "test-stdout.txt"),
    [
      "        FAIL [0.01s] package::test::preserves_selector_anchor",
      "        FAIL [0.02s] package::test::preserves_selector_anchor",
      "        FAIL [0.03s] package::test::preserves_child_anchor",
      "",
    ].join("\n"),
  );
  git(workspaceDir, ["restore", "file.txt"]);
  return { baseCommit, patchPath, root, runDir, workspaceDir };
}

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

describe("DeepSWE candidate continuation", () => {
  it("loads, fingerprints, and applies the previous live candidate patch", () => {
    const fixture = createFixture();
    const continuation = loadCandidateContinuation({
      runDir: fixture.runDir,
      task: { id: "task-1", baseCommit: fixture.baseCommit },
      providerPreference: "provider-1",
      modelPreference: "model-1",
    });

    expect(continuation.evidence).toMatchObject({
      sourceRunId: "source-run",
      sourceTaskId: "task-1",
      sourceCurrentChainStatus: "timeout",
      sourceTerminalStatus: "interrupted",
      sourcePatch: { path: fixture.patchPath, bytes: expect.any(Number) },
    });
    expect(continuation.evidence.sourcePatch.sha256).toMatch(/^[a-f0-9]{64}$/);
    const applied = applyCandidateContinuation({
      workspaceDir: fixture.workspaceDir,
      continuation,
    });
    expect(
      fs.readFileSync(path.join(fixture.workspaceDir, "file.txt"), "utf8"),
    ).toBe("after\n");
    expect(applied.workspaceStatus).toContain("file.txt");
  }, 15_000);

  it("rejects task and route drift", () => {
    const fixture = createFixture();
    const base = {
      runDir: fixture.runDir,
      task: { id: "task-1", baseCommit: fixture.baseCommit },
      providerPreference: "provider-1",
      modelPreference: "model-1",
    };
    expect(() =>
      loadCandidateContinuation({
        ...base,
        task: { ...base.task, id: "task-2" },
      }),
    ).toThrow("task mismatch");
    expect(() =>
      loadCandidateContinuation({ ...base, providerPreference: "provider-2" }),
    ).toThrow("provider mismatch");
    expect(() =>
      loadCandidateContinuation({ ...base, modelPreference: "model-2" }),
    ).toThrow("model mismatch");
  }, 15_000);

  it("reads the fixed route from context when the source chain failed before capture", () => {
    const fixture = createFixture();
    const resultPath = path.join(fixture.runDir, "adapter-result.json");
    const result = JSON.parse(fs.readFileSync(resultPath, "utf8"));
    delete result.currentChain.provider;
    fs.writeFileSync(resultPath, `${JSON.stringify(result)}\n`);
    const contextPath = path.join(fixture.runDir, "run-context.json");
    const context = JSON.parse(fs.readFileSync(contextPath, "utf8"));
    context.continuation = {
      provider: {
        providerPreference: "provider-1",
        modelPreference: "model-1",
      },
    };
    fs.writeFileSync(contextPath, `${JSON.stringify(context)}\n`);

    const continuation = loadCandidateContinuation({
      runDir: fixture.runDir,
      task: { id: "task-1", baseCommit: fixture.baseCommit },
      providerPreference: "provider-1",
      modelPreference: "model-1",
    });
    expect(continuation.evidence.provider).toEqual({
      providerPreference: "provider-1",
      modelPreference: "model-1",
    });
  }, 15_000);

  it("adds auditable continuation guidance without referencing the oracle", () => {
    const fixture = createFixture();
    const continuation = loadCandidateContinuation({
      runDir: fixture.runDir,
      task: { id: "task-1", baseCommit: fixture.baseCommit },
      providerPreference: "provider-1",
      modelPreference: "model-1",
    });
    const prompt = candidateContinuationInstruction(
      "Fix the task.",
      continuation.evidence,
    );
    expect(prompt).toContain("source run source-run");
    expect(prompt).toContain(continuation.evidence.sourcePatch.sha256);
    expect(prompt).toContain("Do not restart broad repository exploration");
    expect(prompt).toContain("do not run unbounded scans");
    expect(prompt.indexOf("CONTINUATION:")).toBeLessThan(
      prompt.indexOf("Original task:"),
    );
    expect(prompt).toContain("Do not replace it with a reference solution");
    expect(prompt).toContain("preserves_selector_anchor");
    expect(continuation.evidence.verifier.failureNames).toEqual([
      "package::test::preserves_selector_anchor",
      "package::test::preserves_child_anchor",
    ]);
    expect(prompt).toContain("previous Pier verifier reward was 0");
  }, 15_000);
});
