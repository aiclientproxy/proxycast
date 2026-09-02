import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  DEEPSWE_DESKTOP_TRIAL_SCHEMA,
  evaluateDesktopSuite,
  evaluateDesktopTrial,
  loadDesktopManifest,
  preflightDesktopManifest,
  sha256,
  validateDesktopManifest,
} from "./deepswe-desktop-contract.mjs";
import { createDeepSweSourceFixture } from "./fixtures/deepswe-source.mjs";

const repoRoot = process.cwd();
const { manifest } = loadDesktopManifest(repoRoot);
let sourceFixture;

beforeAll(() => {
  sourceFixture = createDeepSweSourceFixture({ repoRoot });
});

afterAll(() => {
  fs.rmSync(sourceFixture.sourceRoot, { recursive: true, force: true });
});

function sourceRoot() {
  return sourceFixture.sourceRoot;
}

function evaluateTrial(evidence) {
  return evaluateDesktopTrial({
    evidence,
    manifest,
    repoRoot,
    sourceRoot: sourceRoot(),
  });
}

function evaluateSuite(evidenceList) {
  return evaluateDesktopSuite({
    evidenceList,
    manifest,
    repoRoot,
    sourceRoot: sourceRoot(),
  });
}

function passingEvidence(task, overrides = {}) {
  const patchSha256 = "a".repeat(64);
  const sessionId = `desktop-${task.id}`;
  const threadId = `thread-${task.id}`;
  const turnId = `turn-${task.id}`;
  const instruction = fs.readFileSync(
    path.join(
      sourceRoot(),
      "tasks",
      task.id,
      path.basename(task.instructionPath),
    ),
  );
  const base = {
    schemaVersion: DEEPSWE_DESKTOP_TRIAL_SCHEMA,
    trialId: `trial-${task.id}`,
    trialKind: "live_deepswe",
    taskId: task.id,
    language: task.language,
    repository: task.repository,
    sourceCommit: task.baseCommit,
    instructionSha256: sha256(instruction),
    workspace: path.join(os.tmpdir(), "lime-desktop-smoke", task.id),
    identity: { sessionId, threadId, turnId },
    patchSha256,
    patchBytes: 128,
    changedFiles: ["src/change.txt"],
    toolLifecycle: [
      { name: "Read", status: "completed" },
      { name: "Grep", status: "completed" },
      { name: "apply_patch", status: "completed" },
      { name: "exec_command", status: "completed" },
    ],
    testResult: {
      command: "test",
      status: "pass",
      exitCode: 0,
      outputVisible: true,
    },
    gui: {
      electron: true,
      preloadInvokeBridge: true,
      identity: { sessionId, threadId, turnId },
      terminalVisible: true,
      toolLifecycleVisible: true,
      diffArtifactVisible: true,
      artifactPreview: {
        status: "pass",
        contentVisible: true,
        unavailableErrorVisible: false,
        artifactReadSeen: true,
      },
      consoleErrorCount: 0,
      pageErrorCount: 0,
    },
    readModel: {
      threadId,
      turnId,
      terminalStatus: "completed",
    },
    bridge: {
      appServerHandleJsonLinesSeen: true,
      mockFallbackHitCount: 0,
      invokeErrorCount: 0,
    },
    recovery: {
      cancelNoGhostWrite: { status: "not_run" },
      approvalResume: { status: "not_run" },
      coldRestart: {
        status: "pass",
        previousElectronPid: 100,
        restartedElectronPid: 200,
        electronProcessReplaced: true,
        appServerProcessReplaced: true,
        previousProcessTree: { appServerPids: [101] },
        restartedProcessTree: { appServerPids: [201] },
        previousProcessTreeExit: { exited: true },
        renderer: { electron: true, preloadInvokeBridge: true },
        identity: { sessionId, threadId, turnId },
        projection: {
          stable: true,
          beforeSha256: "c".repeat(64),
          afterSha256: "c".repeat(64),
        },
        toolLifecycleVisible: true,
        diffArtifactVisible: true,
        artifactPreview: {
          status: "pass",
          contentVisible: true,
          unavailableErrorVisible: false,
          artifactReadSeen: true,
        },
        approvalCompleted: true,
        cancelNoGhostWrite: true,
        patch: {
          stable: true,
          beforeSha256: patchSha256,
          afterSha256: patchSha256,
        },
        providerRequestCountStable: true,
        bridge: {
          appServerHandleJsonLinesSeen: true,
          mockFallbackHitCount: 0,
          invokeErrorCount: 0,
        },
        consoleErrorCount: 0,
        pageErrorCount: 0,
      },
    },
    verifier: {
      status: "pass",
      patchSha256,
      artifacts: ["reward.json", "ctrf.json", "test-stdout.txt"],
    },
    gateB: { pass: true },
    desktopCodingPass: true,
  };
  return {
    ...base,
    ...overrides,
    identity: { ...base.identity, ...overrides.identity },
    gui: {
      ...base.gui,
      ...overrides.gui,
      identity: { ...base.gui.identity, ...overrides.gui?.identity },
    },
    readModel: { ...base.readModel, ...overrides.readModel },
    bridge: { ...base.bridge, ...overrides.bridge },
    recovery: { ...base.recovery, ...overrides.recovery },
    verifier: { ...base.verifier, ...overrides.verifier },
  };
}

describe("DeepSWE Desktop Smoke 5 contract", () => {
  it("pins one task for each required language and keeps source identity complete", () => {
    expect(validateDesktopManifest(manifest)).toBe(manifest);
    expect(manifest.tasks).toHaveLength(5);
    expect(manifest.tasks.map((task) => task.id)).toEqual([
      "happy-dom-abort-pending-body-reads",
      "go-genai-streamed-function-args",
      "httpx-multipart-response-parsing",
      "fd-deterministic-multi-key-sorting",
      "yjs-map-conflict-detection",
    ]);
    expect(new Set(manifest.tasks.map((task) => task.language))).toEqual(
      new Set(["typescript", "go", "python", "rust", "javascript"]),
    );
    expect(
      manifest.tasks.every(
        (task) =>
          /^[a-f0-9]{40}$/u.test(task.baseCommit) &&
          task.desktopRisks.length >= 3,
      ),
    ).toBe(true);
    expect(manifest.suiteRequirements.recoveryCoverage).toEqual([
      "cancel_no_ghost_write",
      "approval_resume",
      "cold_restart",
    ]);
    expect(manifest.evidenceContract.requiredGateB).toEqual([
      "realElectronHost",
      "preloadInvokeBridge",
      "appServerHandleJsonLines",
      "canonicalIdentity",
      "terminalReadModel",
      "visibleTerminal",
      "visibleToolLifecycle",
      "visibleDiffArtifact",
      "artifactContentAvailable",
      "coldRestart",
      "productionMockFallbackZero",
      "invokeErrorsZero",
      "consoleErrorsZero",
      "pageErrorsZero",
    ]);
  });

  it("preflights original instructions, task metadata, and separate verifier mode", () => {
    const preflight = preflightDesktopManifest({
      repoRoot,
      manifest,
      sourceRoot: sourceRoot(),
      resolveSourceCommit: () => sourceFixture.sourceCommit,
    });
    expect(preflight.status).toBe("pass");
    expect(preflight.taskCount).toBe(5);
    expect(preflight.checks).toHaveLength(53);
    expect(preflight.checks.every((check) => check.passed)).toBe(true);
    expect(preflight.checks.map((check) => check.name)).toEqual(
      expect.arrayContaining([
        "source-commit",
        "source-schema",
        "source-pier",
        "happy-dom-abort-pending-body-reads:agent-network",
        "happy-dom-abort-pending-body-reads:verifier-network",
        "happy-dom-abort-pending-body-reads:collect",
        "happy-dom-abort-pending-body-reads:pre-artifacts-deleted",
      ]),
    );
  }, 20_000);

  it("accepts a live trial only when Gate B, verifier, and patch identity all match", () => {
    const evidence = passingEvidence(manifest.tasks[0]);
    const verdict = evaluateTrial(evidence);
    expect(verdict.gateBPass).toBe(true);
    expect(verdict.verifierPass).toBe(true);
    expect(verdict.desktopCodingPass).toBe(true);
    expect(verdict.claimConsistent).toBe(true);
    expect(verdict.failedGateBAssertions).toEqual([]);
  });

  it("never promotes a controlled product smoke to a DeepSWE pass", () => {
    const evidence = passingEvidence(manifest.tasks[0], {
      trialKind: "controlled_product_smoke",
      desktopCodingPass: false,
      verifier: {
        status: "not_run",
        patchSha256: null,
        artifacts: [],
      },
    });
    const verdict = evaluateTrial(evidence);
    expect(verdict.gateBPass).toBe(true);
    expect(verdict.verifierPass).toBe(false);
    expect(verdict.desktopCodingPass).toBe(false);
    expect(verdict.claimConsistent).toBe(true);
  });

  it.each([
    [
      "patch SHA mismatch",
      (evidence) => ({
        ...evidence,
        desktopCodingPass: false,
        verifier: { ...evidence.verifier, patchSha256: "b".repeat(64) },
      }),
      "verifierPatchMatches",
    ],
    [
      "workspace inside Lime",
      (evidence) => ({
        ...evidence,
        gateB: { pass: false },
        desktopCodingPass: false,
        workspace: path.join(repoRoot, ".lime", "bad-workspace"),
      }),
      "workspaceOutsideLime",
    ],
    [
      "missing terminal",
      (evidence) => ({
        ...evidence,
        gateB: { pass: false },
        desktopCodingPass: false,
        readModel: { ...evidence.readModel, terminalStatus: "in_progress" },
      }),
      "terminalReadModel",
    ],
    [
      "GUI identity mismatch",
      (evidence) => ({
        ...evidence,
        gateB: { pass: false },
        desktopCodingPass: false,
        gui: {
          ...evidence.gui,
          identity: { ...evidence.gui.identity, turnId: "other-turn" },
        },
      }),
      "canonicalIdentity",
    ],
    [
      "artifact content unavailable",
      (evidence) => ({
        ...evidence,
        gateB: { pass: false },
        desktopCodingPass: false,
        gui: {
          ...evidence.gui,
          artifactPreview: {
            ...evidence.gui.artifactPreview,
            contentVisible: false,
            unavailableErrorVisible: true,
          },
        },
      }),
      "artifactContentAvailable",
    ],
    [
      "mock fallback",
      (evidence) => ({
        ...evidence,
        gateB: { pass: false },
        desktopCodingPass: false,
        bridge: { ...evidence.bridge, mockFallbackHitCount: 1 },
      }),
      "productionMockFallbackZero",
    ],
    [
      "missing verifier artifacts",
      (evidence) => ({
        ...evidence,
        desktopCodingPass: false,
        verifier: { ...evidence.verifier, artifacts: ["reward.json"] },
      }),
      "verifierArtifactsComplete",
    ],
    [
      "cold restart without process-tree exit",
      (evidence) => ({
        ...evidence,
        gateB: { pass: false },
        desktopCodingPass: false,
        recovery: {
          ...evidence.recovery,
          coldRestart: {
            ...evidence.recovery.coldRestart,
            previousProcessTreeExit: { exited: false },
          },
        },
      }),
      "coldRestart",
    ],
  ])("rejects %s", (_label, mutate, assertionName) => {
    const evidence = mutate(passingEvidence(manifest.tasks[0]));
    const verdict = evaluateTrial(evidence);
    expect(verdict.assertions[assertionName]).toBe(false);
    expect(verdict.desktopCodingPass).toBe(false);
  });

  it("marks a contradictory claimed pass as invalid", () => {
    const evidence = passingEvidence(manifest.tasks[0], {
      workspace: path.join(repoRoot, "invalid"),
    });
    const verdict = evaluateTrial(evidence);
    expect(verdict.gateBPass).toBe(false);
    expect(verdict.claimConsistent).toBe(false);
    expect(verdict.valid).toBe(false);
  });

  it("requires all five live trials and suite-level recovery coverage", () => {
    const evidenceList = manifest.tasks.map((task, index) =>
      passingEvidence(task, {
        recovery: {
          cancelNoGhostWrite: {
            status: index === 0 ? "pass" : "not_run",
          },
          approvalResume: { status: index === 3 ? "pass" : "not_run" },
          coldRestart: passingEvidence(task).recovery.coldRestart,
        },
      }),
    );
    const suite = evaluateSuite(evidenceList);
    expect(suite.status).toBe("pass");
    expect(suite.desktopCodingPass).toBe(true);
    expect(suite.recoveryCoverage).toEqual({
      cancel_no_ghost_write: true,
      approval_resume: true,
      cold_restart: true,
    });

    evidenceList[0].recovery.cancelNoGhostWrite.status = "not_run";
    const incomplete = evaluateSuite(evidenceList);
    expect(incomplete.desktopCodingPass).toBe(false);
    expect(incomplete.failedAssertions).toContain("recoveryCoverageComplete");

    evidenceList[0].recovery.cancelNoGhostWrite.status = "pass";
    evidenceList.forEach((evidence) => {
      evidence.recovery.coldRestart.status = "not_run";
    });
    const restartIncomplete = evaluateSuite(evidenceList);
    expect(restartIncomplete.desktopCodingPass).toBe(false);
    expect(restartIncomplete.recoveryCoverage.cold_restart).toBe(false);
  });
});
