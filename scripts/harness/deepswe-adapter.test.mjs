import { execFileSync, spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  capturePatch,
  classifyFailure,
  collectPierEvidence,
  createTaskWorkspaceLocation,
  currentChainFromError,
  DEFAULT_DEEPSWE_TASK,
  loadTaskDefinition,
  preflightSelectedTasks,
  preparePierReplayTask,
  prepareTaskWorkspace,
  readJson,
  runContextBase,
  runCurrentChainTask,
  runPierVerifier,
  runtimePrerequisites,
  terminalMessageFromCurrentFacts,
  verifierCompletionStatus,
  verifierTaskIdForRun,
} from "./deepswe-adapter-core.mjs";
import { createDeepSweSourceFixture } from "./fixtures/deepswe-source.mjs";

const repoRoot = process.cwd();
const temporaryRoots = [];
let sourceFixture = null;

function temporaryRoot() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "deepswe-adapter-test-"));
  temporaryRoots.push(root);
  return root;
}

function deepSweSourceFixture() {
  if (!sourceFixture) {
    sourceFixture = createDeepSweSourceFixture({ repoRoot });
    temporaryRoots.push(sourceFixture.sourceRoot);
  }
  return sourceFixture;
}

function git(cwd, args) {
  return execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
}

function createLocalRepository(root) {
  const repository = path.join(root, "origin");
  fs.mkdirSync(repository, { recursive: true });
  git(repository, ["init"]);
  git(repository, ["config", "user.name", "DeepSWE Test"]);
  git(repository, ["config", "user.email", "deepswe-test@localhost"]);
  fs.writeFileSync(path.join(repository, "README.md"), "baseline\n", "utf8");
  git(repository, ["add", "README.md"]);
  git(repository, ["commit", "-m", "baseline"]);
  return { repository, baseCommit: git(repository, ["rev-parse", "HEAD"]) };
}

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true });
  }
  sourceFixture = null;
});

describe("DeepSWE current-chain adapter", () => {
  it("binds verifier-only runs to the task recorded in run context", () => {
    const runDir = temporaryRoot();
    fs.writeFileSync(
      path.join(runDir, "run-context.json"),
      `${JSON.stringify({ task: { id: "oxvg-task" } })}\n`,
    );

    expect(verifierTaskIdForRun({ runDir })).toBe("oxvg-task");
    expect(verifierTaskIdForRun({ runDir, requestedTaskId: "oxvg-task" })).toBe(
      "oxvg-task",
    );
    expect(() =>
      verifierTaskIdForRun({ runDir, requestedTaskId: "happy-dom-task" }),
    ).toThrow(
      "DeepSWE verifier task mismatch: expected=oxvg-task actual=happy-dom-task",
    );
  });

  it("only marks completed current chains as verified", () => {
    expect(verifierCompletionStatus("completed")).toBe("verified");
    for (const status of [
      "timeout",
      "failed",
      "interrupted",
      "cancelled",
      "canceled",
      "aborted",
      undefined,
    ]) {
      expect(verifierCompletionStatus(status)).toBe(
        "verified_with_product_failure",
      );
    }
  });

  it("requires a non-exhausted chain, non-empty patch, and Pier reward one", () => {
    const currentChain = { status: "completed", terminalMessage: "" };
    const patch = { bytes: 42 };
    expect(verifierCompletionStatus({ currentChain, patch, reward: 1 })).toBe(
      "verified",
    );
    for (const input of [
      {
        currentChain: {
          ...currentChain,
          providerStepExhaustion: { reasons: ["provider_steps"] },
        },
        patch,
        reward: 1,
      },
      {
        currentChain: {
          ...currentChain,
          terminalMessage: "DeepSWE provider budget exhausted: steps=48",
        },
        patch,
        reward: 1,
      },
      { currentChain, patch: { bytes: 0 }, reward: 1 },
      { currentChain, patch, reward: 0 },
      { currentChain, patch },
    ]) {
      expect(verifierCompletionStatus(input)).toBe(
        "verified_with_product_failure",
      );
    }
  });

  it("writes the current source and adapter identity into run context", () => {
    const context = runContextBase(
      {
        allowLiveProvider: false,
        firstVisibleOutputTimeoutMs: 300_000,
        transport: "stdio",
      },
      "run-identity",
      {
        id: "task-identity",
        language: "typescript",
        repository: "example/repository",
        repositoryUrl: "https://example.test/repository.git",
        baseCommit: "0123456789012345678901234567890123456789",
        schemaVersion: "1.3",
        environment: {},
        verifier: {},
      },
    );

    expect(context).toMatchObject({
      sourceCommit: "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9",
      executionContract: {
        adapterVersion: "deepswe-current-chain-adapter-v7",
        appServerMethods: [
          "workspace/ensure",
          "thread/start",
          "turn/start",
          "thread/read",
          "turn/interrupt",
        ],
        generationControls: {
          firstVisibleOutputTimeoutMs: 300_000,
          projection:
            "additionalContext.metadata -> RuntimeRequest.metadata.harness.generation",
        },
      },
    });
    expect(context.executionContract.appServerMethods).not.toContain(
      "thread/settings/update",
    );
  });

  it("validates all selected Release 20 tasks against the pinned source", () => {
    const fixture = deepSweSourceFixture();
    const result = preflightSelectedTasks({
      repoRoot,
      sourceRoot: fixture.sourceRoot,
      sliceName: "release-20",
      resolveSourceCommit: () => fixture.sourceCommit,
    });

    expect(result.status).toBe("pass");
    expect(result.taskCount).toBe(20);
    expect(result.checks).toHaveLength(205);
    expect(result.checks.every((check) => check.passed)).toBe(true);
    expect(result.checks.map((check) => check.name)).toEqual(
      expect.arrayContaining([
        "source-license",
        "source-provenance",
        "happy-dom-abort-pending-body-reads:agent-network",
        "happy-dom-abort-pending-body-reads:verifier-network",
        "happy-dom-abort-pending-body-reads:collect-command",
        "happy-dom-abort-pending-body-reads:pre-artifacts-deleted",
      ]),
    );
  }, 20_000);

  it("loads task metadata with TOML semantics instead of ad hoc line parsing", () => {
    const fixture = deepSweSourceFixture();
    const task = loadTaskDefinition({
      repoRoot,
      sourceRoot: fixture.sourceRoot,
      taskId: "happy-dom-abort-pending-body-reads",
    });
    const fixtureTask = fixture.tasks.get(
      "happy-dom-abort-pending-body-reads",
    );

    expect(task).toMatchObject({
      id: "happy-dom-abort-pending-body-reads",
      schemaVersion: "1.3",
      repository: "capricorn86/happy-dom",
      repositoryUrl: fixtureTask.repositoryUrl,
      baseCommit: fixtureTask.baseCommit,
      artifacts: ["/logs/artifacts/model.patch"],
      agent: { networkMode: "no-network", timeoutSec: 5400 },
      verifier: {
        networkMode: "no-network",
        environmentMode: "separate",
        collect: [
          {
            command: expect.stringContaining(
              `git diff --binary ${fixtureTask.baseCommit} HEAD`,
            ),
            timeoutSec: 300,
          },
        ],
      },
    });
    expect(task.instruction).toBe(fixtureTask.instruction);
  });

  it("prepares an isolated branch and captures committed plus uncommitted changes", () => {
    const root = temporaryRoot();
    const { repository, baseCommit } = createLocalRepository(root);
    const workspaceDir = path.join(root, "workspace");
    const workspace = prepareTaskWorkspace({
      task: { repositoryUrl: repository, baseCommit },
      workspaceDir,
      runId: "test-run",
    });
    fs.writeFileSync(path.join(workspaceDir, "README.md"), "changed\n", "utf8");
    fs.writeFileSync(path.join(workspaceDir, "new.txt"), "new\n", "utf8");
    const patchPath = path.join(root, "patch.diff");
    const patch = capturePatch({
      workspaceDir,
      baseCommit,
      outputPath: patchPath,
    });

    expect(workspace.branch).toBe("deepswe-test-run");
    expect(workspace.head).toBe(baseCommit);
    expect(git(workspaceDir, ["remote"])).toBe("");
    expect(git(workspaceDir, ["branch", "--format=%(refname:short)"])).toBe(
      "deepswe-test-run\nmain",
    );
    expect(patch.bytes).toBeGreaterThan(0);
    expect(fs.readFileSync(patchPath, "utf8")).toContain("new.txt");
  }, 20_000);

  it("places task workspaces outside the Lime repository and its node_modules lookup chain", () => {
    const root = temporaryRoot();
    const fakeRepoRoot = path.join(root, "lime");
    const hostOnlyModule = path.join(
      fakeRepoRoot,
      "node_modules",
      "host-only-module",
    );
    fs.mkdirSync(hostOnlyModule, { recursive: true });
    fs.writeFileSync(
      path.join(hostOnlyModule, "package.json"),
      JSON.stringify({ name: "host-only-module", main: "index.js" }),
      "utf8",
    );
    fs.writeFileSync(
      path.join(hostOnlyModule, "index.js"),
      "module.exports = 1;\n",
    );

    const location = createTaskWorkspaceLocation({
      repoRoot: fakeRepoRoot,
      tempRoot: root,
    });
    fs.mkdirSync(location.workspaceDir);
    const relative = path.relative(fakeRepoRoot, location.workspaceDir);
    const workspaceRequire = createRequire(
      path.join(location.workspaceDir, "resolution-probe.cjs"),
    );

    expect(relative === ".." || relative.startsWith(`..${path.sep}`)).toBe(
      true,
    );
    expect(() => workspaceRequire.resolve("host-only-module")).toThrow();
  });

  it("captures candidate patches larger than the child-process default buffer", () => {
    const root = temporaryRoot();
    const { repository } = createLocalRepository(root);
    const largePath = path.join(repository, "large.txt");
    fs.writeFileSync(largePath, `${"a".repeat(700_000)}\n`, "utf8");
    git(repository, ["add", "large.txt"]);
    git(repository, ["commit", "-m", "large baseline"]);
    const baseCommit = git(repository, ["rev-parse", "HEAD"]);
    fs.writeFileSync(largePath, `${"b".repeat(700_000)}\n`, "utf8");

    const patch = capturePatch({
      workspaceDir: repository,
      baseCommit,
      outputPath: path.join(root, "large.patch"),
    });

    expect(patch.bytes).toBeGreaterThan(1_000_000);
  }, 20_000);

  it("rejects upstream commits that moved HEAD beyond the task base", () => {
    const root = temporaryRoot();
    const { repository, baseCommit } = createLocalRepository(root);
    fs.writeFileSync(path.join(repository, "upstream.txt"), "future\n", "utf8");
    git(repository, ["add", "upstream.txt"]);
    git(repository, ["commit", "-m", "future upstream"]);
    const workspaceDir = path.join(root, "drifted-workspace");
    execFileSync("git", ["clone", repository, workspaceDir]);
    git(workspaceDir, ["config", "user.name", "Lime DeepSWE Adapter"]);
    git(workspaceDir, ["config", "user.email", "deepswe@localhost"]);

    expect(() =>
      capturePatch({
        workspaceDir,
        baseCommit,
        outputPath: path.join(root, "invalid.patch"),
      }),
    ).toThrow("workspace HEAD contains non-candidate commits");
  }, 20_000);

  it("runs the public current-chain contract and writes structured evidence", async () => {
    const root = temporaryRoot();
    const workspaceDir = path.join(root, "workspace");
    fs.mkdirSync(workspaceDir);
    const calls = [];
    let turnStartParams = null;
    const sessionRead = {
      thread: {
        id: "deepswe-thread-run-1",
        sessionId: "deepswe-session-run-1",
        turns: [
          {
            id: "deepswe-turn-run-1",
            status: "completed",
            items: [
              {
                kind: "tool",
                payload: { type: "tool", name: "Read" },
                status: "completed",
              },
              { type: "command_execution", status: "completed" },
              { type: "file_artifact", status: "completed" },
            ],
          },
        ],
      },
    };
    const rpc = {
      waitForHealth: async () => ({ status: "ok" }),
      invoke: async (_options, method, params) => {
        calls.push({ method, params });
        if (method === "workspace/ensure") {
          return { workspace: { id: "workspace-1", rootPath: workspaceDir } };
        }
        if (method === "thread/start") {
          return {
            thread: {
              id: "deepswe-thread-run-1",
              sessionId: "deepswe-session-run-1",
            },
          };
        }
        throw new Error(`unexpected method ${method}`);
      },
      resolveProvider: async () => ({
        providerPreference: "provider-1",
        providerName: "openai",
        modelPreference: "model-1",
        source: "test",
      }),
      startTurn: async (_options, params) => {
        turnStartParams = params;
        return { turn: { id: "deepswe-turn-run-1" } };
      },
      readThread: async (_options, threadId) => {
        calls.push({
          method: "thread/read",
          params: { threadId, includeTurns: true },
        });
        return sessionRead;
      },
      sleep: async () => {},
    };
    const result = await runCurrentChainTask({
      options: {
        healthUrl: "http://unused",
        invokeUrl: "http://unused",
        intervalMs: 1,
        timeoutMs: 30_000,
        maxProviderSteps: 2,
        tokenBudget: 1_000,
        firstVisibleOutputTimeoutMs: 300_000,
        maxOutputTokens: 4_096,
        enableThinking: false,
      },
      task: { id: "task-1", instruction: "Fix the task" },
      workspaceDir,
      runDir: root,
      runId: "run-1",
      rpc,
    });

    expect(result.status).toBe("completed");
    expect(result).toMatchObject({
      sessionId: "deepswe-session-run-1",
      threadId: "deepswe-thread-run-1",
      turnId: "deepswe-turn-run-1",
    });
    expect(calls.map((call) => call.method)).toEqual([
      "workspace/ensure",
      "thread/start",
      "thread/read",
    ]);
    const threadStartParams = calls.find(
      (call) => call.method === "thread/start",
    )?.params;
    expect(threadStartParams).toEqual({
      model: "model-1",
      modelProvider: "provider-1",
      cwd: workspaceDir,
      runtimeWorkspaceRoots: [workspaceDir],
      approvalPolicy: "never",
      sandbox: "workspace-write",
      serviceName: "DeepSWE task-1",
      historyMode: "paginated",
      threadSource: "appServer",
    });
    for (const retiredField of [
      "sessionId",
      "threadId",
      "appId",
      "workspaceId",
      "workingDir",
      "businessObjectRef",
    ]) {
      expect(threadStartParams).not.toHaveProperty(retiredField);
    }
    expect(turnStartParams).toMatchObject({
      threadId: "deepswe-thread-run-1",
      input: [{ type: "text", text: "Fix the task" }],
      cwd: workspaceDir,
      runtimeWorkspaceRoots: [workspaceDir],
      approvalPolicy: "never",
      sandboxPolicy: "workspace-write",
      model: "model-1",
    });
    expect(turnStartParams).not.toHaveProperty("runtimeRequest");
    expect(
      JSON.parse(turnStartParams.additionalContext.metadata.value),
    ).toEqual({
      harness: {
        source: "harness:deepswe:run",
        scenarioId: "DSW-01",
        taskId: "task-1",
        provider_budget: {
          max_provider_steps: 2,
          token_budget: 1_000,
        },
        generation: {
          first_visible_output_timeout_ms: 300_000,
          max_output_tokens: 4_096,
          enable_thinking: false,
        },
      },
    });
    expect(fs.existsSync(path.join(root, "thread-turn-item.json"))).toBe(true);
    expect(fs.existsSync(path.join(root, "trajectory.json"))).toBe(true);
    expect(fs.existsSync(path.join(root, "provider-steps.json"))).toBe(true);
    expect(fs.existsSync(path.join(root, "tool-lifecycle.json"))).toBe(true);
    expect(readJson(path.join(root, "tool-lifecycle.json")).itemCount).toBe(3);
  });

  it("keeps the App Server terminal failure message for owner classification", () => {
    expect(
      terminalMessageFromCurrentFacts(
        {
          turnId: "turn-1",
          threadRead: {
            turns: [
              {
                id: "turn-1",
                status: "failed",
                error:
                  "execution backend error: 读取 provider SSE 失败: error decoding response body",
              },
            ],
          },
        },
        "turn-1",
      ),
    ).toBe(
      "execution backend error: 读取 provider SSE 失败: error decoding response body",
    );
  });

  it("keeps partial current-chain evidence when turn start fails", async () => {
    const root = temporaryRoot();
    const workspaceDir = path.join(root, "workspace");
    fs.mkdirSync(workspaceDir);
    let readCount = 0;
    let turnStartParams = null;
    const rpc = {
      waitForHealth: async () => ({ status: "ok" }),
      invoke: async (_options, method) => {
        if (method === "workspace/ensure") {
          return { workspace: { id: "workspace-1", rootPath: workspaceDir } };
        }
        if (method === "thread/start") {
          return {
            thread: {
              id: "deepswe-thread-failed",
              sessionId: "deepswe-session-failed",
            },
          };
        }
        throw new Error(`unexpected method ${method}`);
      },
      resolveProvider: async () => ({
        providerPreference: "provider-1",
        providerName: "openai",
        modelPreference: "model-1",
        source: "test",
      }),
      startTurn: async (_options, params) => {
        turnStartParams = params;
        throw new Error("Provider tool call omitted tool name");
      },
      readThread: async () => {
        readCount += 1;
        return {
          thread: {
            id: "deepswe-thread-failed",
            sessionId: "deepswe-session-failed",
            turns: [],
          },
        };
      },
      sleep: async () => {},
    };

    const error = await runCurrentChainTask({
      options: { intervalMs: 1, timeoutMs: 30_000 },
      task: { id: "task-1", instruction: "Fix the task" },
      workspaceDir,
      runDir: root,
      runId: "run-failed",
      rpc,
    }).catch((caught) => caught);

    expect(error).toBeInstanceOf(Error);
    expect(error.message).toBe("Provider tool call omitted tool name");
    expect(
      JSON.parse(turnStartParams.additionalContext.metadata.value).harness,
    ).not.toHaveProperty("generation");
    expect(currentChainFromError(error)).toMatchObject({
      status: "failed",
      sessionId: "deepswe-session-failed",
      threadId: "deepswe-thread-failed",
      turnId: "",
      provider: {
        providerPreference: "provider-1",
        modelPreference: "model-1",
      },
      terminalMessage: "Provider tool call omitted tool name",
      factsCapture: "partial",
    });

    expect(readCount).toBeGreaterThan(0);
    expect(readJson(path.join(root, "thread-turn-item.json"))).toMatchObject({
      capture: {
        status: "partial",
        startTurnError: "Provider tool call omitted tool name",
      },
    });
    expect(readJson(path.join(root, "trajectory.json")).turns).toEqual([]);
    expect(fs.existsSync(path.join(root, "tool-lifecycle.json"))).toBe(true);
  });

  it("leaves provider step exhaustion to the runtime reply loop", async () => {
    const root = temporaryRoot();
    const workspaceDir = path.join(root, "workspace");
    fs.mkdirSync(workspaceDir);
    let readCount = 0;
    let canceled = false;
    const providerSteps = [1, 2].map((attempt) => ({
      sequence: attempt,
      attempt,
      completed: true,
      finish_reason: "tool_call",
      text_output_chars: 0,
      reasoning_output_chars: 10,
      tool_call_count: 1,
      usage: { input_tokens: 100, output_tokens: 10 },
    }));
    const rpc = {
      waitForHealth: async () => ({ status: "ok" }),
      invoke: async (_options, method) => {
        if (method === "workspace/ensure") {
          return { workspace: { id: "workspace-1", rootPath: workspaceDir } };
        }
        if (method === "thread/start") {
          return {
            thread: {
              id: "deepswe-thread-step-cap",
              sessionId: "deepswe-session-step-cap",
            },
          };
        }
        throw new Error(`unexpected method ${method}`);
      },
      resolveProvider: async () => ({
        providerPreference: "provider-1",
        providerName: "openai",
        modelPreference: "model-1",
        source: "test",
      }),
      startTurn: async () => ({ turn: { id: "deepswe-turn-step-cap" } }),
      cancelTurn: async () => {
        canceled = true;
      },
      readThread: async () => {
        readCount += 1;
        return {
          thread: {
            id: "deepswe-thread-step-cap",
            sessionId: "deepswe-session-step-cap",
            turns: [
              {
                id: "deepswe-turn-step-cap",
                status: readCount >= 2 ? "completed" : "inProgress",
                items: [],
              },
            ],
          },
          providerSteps,
        };
      },
      sleep: async () => {},
    };

    const result = await runCurrentChainTask({
      options: {
        intervalMs: 1,
        timeoutMs: 30_000,
        maxProviderSteps: 2,
        tokenBudget: 1_000,
      },
      task: { id: "task-1", instruction: "Fix the task" },
      workspaceDir,
      runDir: root,
      runId: "run-step-cap",
      rpc,
    });

    expect(canceled).toBe(false);
    expect(result).toMatchObject({
      status: "completed",
      budgetCancellation: null,
      providerStepExhaustion: {
        reasons: ["provider_steps"],
        stepCount: 2,
      },
      providerSteps: {
        stepCount: 2,
        budgets: {
          exhausted: true,
          reasons: ["provider_steps"],
        },
      },
    });
  });

  it("cancels a wall-timeout turn and captures its real terminal state", async () => {
    const root = temporaryRoot();
    const workspaceDir = path.join(root, "workspace");
    fs.mkdirSync(workspaceDir);
    let canceled = false;
    let cancelParams = null;
    const rpc = {
      waitForHealth: async () => ({ status: "ok" }),
      invoke: async (_options, method) => {
        if (method === "workspace/ensure") {
          return { workspace: { id: "workspace-1", rootPath: workspaceDir } };
        }
        if (method === "thread/start") {
          return {
            thread: {
              id: "deepswe-thread-wall-timeout",
              sessionId: "deepswe-session-wall-timeout",
            },
          };
        }
        throw new Error(`unexpected method ${method}`);
      },
      resolveProvider: async () => ({
        providerPreference: "provider-1",
        providerName: "openai",
        modelPreference: "model-1",
        source: "test",
      }),
      startTurn: async () => ({ turn: { id: "deepswe-turn-wall-timeout" } }),
      cancelTurn: async (_options, params) => {
        cancelParams = params;
        canceled = true;
      },
      readThread: async () => ({
        thread: {
          id: "deepswe-thread-wall-timeout",
          sessionId: "deepswe-session-wall-timeout",
          turns: [
            {
              id: "deepswe-turn-wall-timeout",
              status: canceled ? "interrupted" : "inProgress",
              items: [],
            },
          ],
        },
      }),
      sleep: async () => {},
    };

    const error = await runCurrentChainTask({
      options: { intervalMs: 1, timeoutMs: 1 },
      task: { id: "task-1", instruction: "Fix the task" },
      workspaceDir,
      runDir: root,
      runId: "run-wall-timeout",
      rpc,
    }).catch((caught) => caught);

    expect(canceled).toBe(true);
    expect(cancelParams).toEqual({
      threadId: "deepswe-thread-wall-timeout",
      turnId: "deepswe-turn-wall-timeout",
    });
    expect(error).toBeInstanceOf(Error);
    expect(error.message).toContain("cancelStatus=interrupted");
    expect(currentChainFromError(error)).toMatchObject({
      status: "timeout",
      terminalStatus: "interrupted",
      factsCapture: "terminal",
      timeoutCancellation: {
        reason: "wall_timeout",
        terminalStatus: "interrupted",
        error: null,
      },
    });
    expect(readJson(path.join(root, "thread-turn-item.json"))).toMatchObject({
      capture: { status: "terminal" },
    });
  });

  it("builds a Pier replay task without copying the reference solution", () => {
    const root = temporaryRoot();
    const taskDir = path.join(root, "task");
    fs.mkdirSync(path.join(taskDir, "solution"), { recursive: true });
    fs.mkdirSync(path.join(taskDir, "tests"), { recursive: true });
    fs.writeFileSync(
      path.join(taskDir, "task.toml"),
      [
        'schema_version = "1.3"',
        'artifacts = ["/logs/artifacts/model.patch"]',
        "[verifier]",
        'network_mode = "no-network"',
        'environment_mode = "separate"',
        "[[verifier.collect]]",
        'command = "cd /app && git diff --binary base HEAD > /logs/artifacts/model.patch"',
        "timeout_sec = 300.0",
        "",
      ].join("\n"),
    );
    fs.writeFileSync(
      path.join(taskDir, "solution", "reference.patch"),
      "secret\n",
    );
    fs.writeFileSync(path.join(taskDir, "tests", "test.sh"), "#!/bin/bash\n");
    const patchPath = path.join(root, "patch.diff");
    fs.writeFileSync(patchPath, "diff --git a/a b/a\n", "utf8");

    const replay = preparePierReplayTask({
      task: { taskDir },
      runDir: root,
      patchPath,
    });

    expect(
      fs.existsSync(
        path.join(replay.replayTaskDir, "solution", "reference.patch"),
      ),
    ).toBe(false);
    expect(
      fs.readFileSync(
        path.join(replay.replayTaskDir, "solution", "model.patch"),
        "utf8",
      ),
    ).toContain("diff --git");
    const solveScript = fs.readFileSync(replay.solvePath, "utf8");
    expect(solveScript).toContain(
      "git apply --binary --whitespace=nowarn /solution/model.patch",
    );
    expect(solveScript).toContain("git add -A");
    expect(solveScript).not.toContain("--index");
    expect(
      fs.readFileSync(path.join(replay.replayTaskDir, "task.toml"), "utf8"),
    ).toBe(fs.readFileSync(path.join(taskDir, "task.toml"), "utf8"));
  });

  it("collects the three verifier outputs required by the v2 contract", () => {
    const root = temporaryRoot();
    const jobDir = path.join(root, "jobs", "trial", "verifier");
    const runDir = path.join(root, "run");
    fs.mkdirSync(jobDir, { recursive: true });
    fs.mkdirSync(runDir, { recursive: true });
    for (const name of ["reward.json", "ctrf.json", "test-stdout.txt"]) {
      fs.writeFileSync(path.join(jobDir, name), `${name}\n`, "utf8");
    }

    const evidence = collectPierEvidence({
      jobDir: path.join(root, "jobs"),
      runDir,
    });

    expect(Object.keys(evidence).sort()).toEqual([
      "ctrf.json",
      "reward.json",
      "test-stdout.txt",
    ]);
    expect(fs.existsSync(path.join(runDir, "reward.json"))).toBe(true);
  });

  it("runs Pier with a Colima-visible temp directory inside the run", () => {
    const root = temporaryRoot();
    const runDir = path.join(root, "run");
    const taskDir = path.join(root, "task");
    const patchPath = path.join(root, "patch.diff");
    const pierBin = path.join(root, "pier");
    const containerBin = path.join(root, "docker");
    fs.mkdirSync(taskDir, { recursive: true });
    fs.mkdirSync(path.join(runDir, "pier-jobs", "verify-colima-temp"), {
      recursive: true,
    });
    fs.writeFileSync(path.join(taskDir, "task.toml"), "version = 1\n", "utf8");
    fs.writeFileSync(patchPath, "candidate patch\n", "utf8");
    fs.writeFileSync(
      pierBin,
      `#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
if (process.argv[2] === "--version") {
  process.stdout.write("0.3.1");
  process.exit(0);
}
const valueAfter = (name) => process.argv[process.argv.indexOf(name) + 1];
const jobsDir = valueAfter("--jobs-dir");
const jobName = valueAfter("--job-name");
const verifierDir = path.join(jobsDir, jobName, "verifier");
fs.mkdirSync(verifierDir, { recursive: true });
fs.writeFileSync(path.join(path.dirname(jobsDir), "pier-child-env.json"), JSON.stringify({
  tmpdir: process.env.TMPDIR,
  containerBin: process.env.PIER_CONTAINER_BIN,
  args: process.argv.slice(2),
}));
fs.writeFileSync(path.join(verifierDir, "reward.json"), JSON.stringify({ reward: 1 }));
fs.writeFileSync(path.join(verifierDir, "ctrf.json"), JSON.stringify({ results: {} }));
fs.writeFileSync(path.join(verifierDir, "test-stdout.txt"), "passed");
`,
      { mode: 0o755 },
    );
    fs.writeFileSync(containerBin, "#!/bin/sh\nexit 0\n", { mode: 0o755 });

    const result = runPierVerifier({
      task: { taskDir },
      runDir,
      runId: "colima-temp",
      patchPath,
      pierBin,
      containerBin,
    });

    const childEnv = readJson(path.join(runDir, "pier-child-env.json"));
    expect(childEnv).toMatchObject({
      tmpdir: path.join(runDir, "pier-tmp"),
      containerBin,
    });
    expect(childEnv.args).toEqual(
      expect.arrayContaining(["--verifier-env", "NEXTEST_DOUBLE_SPAWN=0"]),
    );
    expect(fs.statSync(childEnv.tmpdir).isDirectory()).toBe(true);
    expect(result.jobDir).toBe(
      path.join(runDir, "pier-jobs", "verify-colima-temp-retry-1"),
    );
    expect(result.reward).toBe(1);
  });

  it("reports missing Pier and container prerequisites without pretending to run", () => {
    const result = runtimePrerequisites({
      pierBin: "/path/that/does/not/exist/pier",
      containerBin: "/path/that/does/not/exist/docker",
    });
    expect(result.status).toBe("blocked");
    expect(result.checks.every((check) => check.passed === false)).toBe(true);
  });

  it("rejects a Pier binary that does not report the pinned 0.3.1 version", () => {
    const root = temporaryRoot();
    const pierBin = path.join(root, "pier");
    fs.writeFileSync(pierBin, "#!/bin/sh\necho 0.3.0\n", { mode: 0o755 });
    const result = runtimePrerequisites({
      pierBin,
      containerBin: "/path/that/does/not/exist/docker",
    });
    const pier = result.checks.find((check) => check.name === "pier");

    expect(result.status).toBe("blocked");
    expect(pier).toMatchObject({
      passed: false,
      detail: "expected 0.3.1; actual 0.3.0",
    });
  });

  it("classifies current-chain and verifier failures by owner", () => {
    expect(
      classifyFailure("agent", new Error("thread/read failed")).owner,
    ).toBe("app-server");
    expect(
      classifyFailure("verifier", new Error("Pier reward.json missing")).owner,
    ).toBe("verifier");
    expect(
      classifyFailure(
        "agent-terminal",
        new Error("读取 provider SSE 失败: error decoding response body"),
      ).owner,
    ).toBe("model");
    expect(
      classifyFailure("patch", new Error("spawnSync git ENOBUFS")).owner,
    ).toBe("harness");
    expect(
      classifyFailure(
        "patch",
        new Error("Lime agent produced an empty patch after a completed turn"),
      ).owner,
    ).toBe("model");
    expect(
      classifyFailure(
        "patch",
        new Error(
          "DeepSWE workspace HEAD contains non-candidate commits after base",
        ),
      ).owner,
    ).toBe("harness");
    expect(
      classifyFailure(
        "agent",
        new Error("DeepSWE turn timeout: session=s turn=t status=in_progress"),
      ).owner,
    ).toBe("budget");
    expect(
      classifyFailure(
        "agent",
        new Error("timed out waiting for app-server message after 37ms"),
      ).owner,
    ).toBe("budget");
  });

  it("keeps retired Benchmark runners and npm entries physically absent", () => {
    const retiredPaths = [
      "internal/test/benchmark-release.manifest.json",
      "internal/test/agent-qc-benchmark.manifest.json",
      "internal/roadmap/benchmark/dataset-selection.md",
      "internal/roadmap/benchmark/progress.md",
      "internal/roadmap/benchmark/version-test-plan.md",
      "internal/research/agent/lime-agent-verification-plan/07-flag-differential-harness.md",
    ];
    expect(
      retiredPaths.every((entry) => !fs.existsSync(path.join(repoRoot, entry))),
    ).toBe(true);
    const benchmarkScripts = fs
      .readdirSync(path.join(repoRoot, "scripts/agent-qc"))
      .filter(
        (entry) => entry.startsWith("benchmark") && entry.endsWith(".mjs"),
      );
    expect(benchmarkScripts).toEqual([]);
    const packageJson = JSON.parse(
      fs.readFileSync(path.join(repoRoot, "package.json"), "utf8"),
    );
    expect(
      Object.keys(packageJson.scripts).filter((name) =>
        name.startsWith("agent-qc:benchmark"),
      ),
    ).toEqual([]);
    const researchPaths = [
      "internal/research/agent/README.md",
      "internal/research/agent/lime-verifiable-agent-development-researched.md",
      "internal/research/agent/lime-agent-verification-plan/README.md",
      "internal/research/agent/lime-agent-verification-plan/08-30-60-90-roadmap.md",
      "internal/research/agent/lime-agent-verification-plan/09-progress-tracker.md",
    ];
    const researchText = researchPaths
      .map((entry) => fs.readFileSync(path.join(repoRoot, entry), "utf8"))
      .join("\n");
    expect(researchText).not.toContain("npm run agent-qc:benchmark");
    expect(researchText).not.toContain(
      "internal/test/agent-qc-benchmark.manifest.json",
    );
    expect(researchText).not.toContain("./07-flag-differential-harness.md");
  });

  it("fails closed before live execution without explicit authorization", () => {
    const result = spawnSync(
      process.execPath,
      [
        "scripts/harness/deepswe-adapter.mjs",
        "--task",
        "happy-dom-abort-pending-body-reads",
      ],
      { cwd: repoRoot, encoding: "utf8" },
    );

    expect(result.status).toBe(1);
    expect(`${result.stdout}${result.stderr}`).toContain(
      "--allow-live-provider",
    );
  });

  it("parses explicit generation controls and rejects invalid tri-state values", () => {
    const valid = spawnSync(
      process.execPath,
      [
        "scripts/harness/deepswe-adapter.mjs",
        "--help",
        "--max-output-tokens",
        "4096",
        "--enable-thinking",
        "false",
        "--first-visible-output-timeout-ms",
        "300000",
      ],
      { cwd: repoRoot, encoding: "utf8" },
    );
    expect(valid.status).toBe(0);

    const invalidTimeout = spawnSync(
      process.execPath,
      [
        "scripts/harness/deepswe-adapter.mjs",
        "--help",
        "--first-visible-output-timeout-ms",
        "0",
      ],
      { cwd: repoRoot, encoding: "utf8" },
    );
    expect(invalidTimeout.status).toBe(1);
    expect(`${invalidTimeout.stdout}${invalidTimeout.stderr}`).toContain(
      "--first-visible-output-timeout-ms must be a positive integer",
    );

    const invalid = spawnSync(
      process.execPath,
      [
        "scripts/harness/deepswe-adapter.mjs",
        "--help",
        "--enable-thinking",
        "auto",
      ],
      { cwd: repoRoot, encoding: "utf8" },
    );
    expect(invalid.status).toBe(1);
    expect(`${invalid.stdout}${invalid.stderr}`).toContain(
      "--enable-thinking must be true or false",
    );
  });

  it("records verifier prerequisite blockers without overwriting product failure evidence", () => {
    const root = temporaryRoot();
    const fixture = deepSweSourceFixture();
    const runDir = path.join(root, "existing-run");
    fs.mkdirSync(runDir);
    fs.writeFileSync(
      path.join(runDir, "patch.diff"),
      "candidate patch\n",
      "utf8",
    );
    fs.writeFileSync(
      path.join(runDir, "adapter-result.json"),
      JSON.stringify({
        schemaVersion: "deepswe-adapter-result-v1",
        status: "product_failed",
        failure: { owner: "model", message: "provider stream failed" },
      }),
      "utf8",
    );
    fs.writeFileSync(
      path.join(runDir, "run-context.json"),
      `${JSON.stringify({ task: { id: DEFAULT_DEEPSWE_TASK } })}\n`,
      "utf8",
    );

    const result = spawnSync(
      process.execPath,
      [
        "scripts/harness/deepswe-adapter.mjs",
        "--verifier-only",
        "--run-dir",
        runDir,
        "--source-root",
        fixture.sourceRoot,
        "--pier-bin",
        path.join(root, "missing-pier"),
        "--container-bin",
        path.join(root, "missing-container"),
      ],
      { cwd: repoRoot, encoding: "utf8" },
    );

    expect(result.status).toBe(1);
    const adapterResult = readJson(path.join(runDir, "adapter-result.json"));
    expect(adapterResult).toMatchObject({
      status: "product_failed",
      failure: { owner: "model" },
      verification: { status: "blocked", failure: { owner: "verifier" } },
    });
    expect(
      readJson(path.join(runDir, "verifier-prerequisites.json")).status,
    ).toBe("blocked");
  });
});
