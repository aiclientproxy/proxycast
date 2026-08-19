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
  loadTaskDefinition,
  preflightSelectedTasks,
  preparePierReplayTask,
  prepareTaskWorkspace,
  providerStepExhaustion,
  providerStepsFromCurrentFacts,
  readJson,
  runContextBase,
  runCurrentChainTask,
  runtimePrerequisites,
  terminalMessageFromCurrentFacts,
} from "./deepswe-adapter-core.mjs";

const repoRoot = process.cwd();
const sourceRoot = path.join(repoRoot, ".lime/benchmark/sources/deep-swe");
const temporaryRoots = [];

function temporaryRoot() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "deepswe-adapter-test-"));
  temporaryRoots.push(root);
  return root;
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
});

describe("DeepSWE current-chain adapter", () => {
  it("writes the current source and adapter identity into run context", () => {
    const context = runContextBase(
      { allowLiveProvider: false, transport: "stdio" },
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
        adapterVersion: "deepswe-current-chain-adapter-v6",
      },
    });
  });

  it("validates all selected Release 20 tasks against the pinned source", () => {
    const result = preflightSelectedTasks({
      repoRoot,
      sourceRoot,
      sliceName: "release-20",
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
    const task = loadTaskDefinition({
      repoRoot,
      sourceRoot,
      taskId: "happy-dom-abort-pending-body-reads",
    });

    expect(task).toMatchObject({
      id: "happy-dom-abort-pending-body-reads",
      schemaVersion: "1.3",
      repository: "capricorn86/happy-dom",
      baseCommit: "82a0888cb2c87a6123e05424b528f8e8c9b3e426",
      artifacts: ["/logs/artifacts/model.patch"],
      agent: { networkMode: "no-network", timeoutSec: 5400 },
      verifier: {
        networkMode: "no-network",
        environmentMode: "separate",
        collect: [
          {
            command: expect.stringContaining(
              "git diff --binary 82a0888cb2c87a6123e05424b528f8e8c9b3e426 HEAD",
            ),
            timeoutSec: 300,
          },
        ],
      },
    });
    expect(task.instruction).toContain("AbortError");
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
      detail: {
        turns: [{ id: "deepswe-turn-run-1", status: "completed" }],
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
    };
    const rpc = {
      waitForHealth: async () => ({ status: "ok" }),
      invoke: async (_options, method, params) => {
        calls.push({ method, params });
        if (method === "workspace/ensure") {
          return { workspace: { id: "workspace-1", rootPath: workspaceDir } };
        }
        if (method === "thread/start") {
          return { session: { sessionId: "deepswe-run-1" } };
        }
        if (method === "thread/read") {
          return sessionRead;
        }
        throw new Error(`unexpected method ${method}`);
      },
      resolveProvider: async () => ({
        providerPreference: "provider-1",
        providerName: "openai",
        modelPreference: "model-1",
        source: "test",
      }),
      updateThreadSettings: async () => {},
      startTurn: async (_options, params) => {
        turnStartParams = params;
      },
      readThread: async () => ({
        status: "completed",
        active_turn_id: null,
        turns: [{ id: "deepswe-turn-run-1", status: "completed" }],
      }),
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
    expect(calls.map((call) => call.method)).toEqual([
      "workspace/ensure",
      "thread/start",
      "thread/read",
    ]);
    expect(
      calls.find((call) => call.method === "thread/start")?.params,
    ).toMatchObject({ workingDir: workspaceDir, workspaceId: "workspace-1" });
    expect(turnStartParams).toMatchObject({
      runtimeRequest: {
        metadata: {
          harness: {
            provider_budget: {
              max_provider_steps: 2,
              token_budget: 1_000,
            },
            generation: {
              max_output_tokens: 4_096,
              enable_thinking: false,
            },
          },
        },
      },
    });
    expect(fs.existsSync(path.join(root, "thread-turn-item.json"))).toBe(true);
    expect(fs.existsSync(path.join(root, "trajectory.json"))).toBe(true);
    expect(fs.existsSync(path.join(root, "provider-steps.json"))).toBe(true);
    expect(fs.existsSync(path.join(root, "tool-lifecycle.json"))).toBe(true);
    expect(readJson(path.join(root, "tool-lifecycle.json")).itemCount).toBe(3);
  });

  it("summarizes provider step output and enforces comparable token accounting", () => {
    const summary = providerStepsFromCurrentFacts(
      {
        threadRead: {
          providerSteps: [
            {
              sequence: 10,
              timestamp: "2026-07-16T00:00:00Z",
              attempt: 1,
              completed: true,
              finish_reason: "tool_call",
              toolNames: ["Read", "apply_patch", "Read"],
              text_output_chars: 7,
              reasoning_output_chars: 40,
              tool_call_count: 1,
              usage: {
                input_tokens: 100,
                output_tokens: 20,
                cached_input_tokens: 40,
              },
            },
            {
              sequence: 20,
              attempt: 2,
              completed: true,
              finish_reason: "stop",
              toolNames: ["exec_command", "apply_patch"],
              text_output_chars: 12,
              reasoning_output_chars: 60,
              tool_call_count: 0,
              usage: {
                input_tokens: 200,
                output_tokens: 30,
                cached_input_tokens: 50,
              },
            },
          ],
        },
      },
      { maxProviderSteps: 2, tokenBudget: 250 },
    );

    expect(summary).toMatchObject({
      stepCount: 2,
      usageStatus: "complete",
      usage: {
        inputTokens: 300,
        outputTokens: 50,
        cachedInputTokens: 90,
        budgetTokens: 260,
      },
      toolCatalog: {
        status: "complete",
        requestCount: 2,
        requestsWithTools: 2,
        uniqueToolNames: ["Read", "apply_patch", "exec_command"],
        applyPatchAvailableOnEveryRequest: true,
      },
      budgets: {
        exhausted: true,
        reasons: ["provider_steps", "token_budget"],
        remainingProviderSteps: 0,
        remainingTokens: 0,
      },
    });
    expect(summary.steps[0].output).toEqual({
      textChars: 7,
      reasoningChars: 40,
      toolCalls: 1,
    });
    expect(summary.steps[0].toolNames).toEqual(["Read", "apply_patch"]);
    expect(summary.steps[1].toolNames).toEqual(["apply_patch", "exec_command"]);
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
          return { session: { sessionId: "deepswe-run-failed" } };
        }
        if (method === "thread/read") {
          readCount += 1;
          return { detail: { turns: [], items: [] } };
        }
        throw new Error(`unexpected method ${method}`);
      },
      resolveProvider: async () => ({
        providerPreference: "provider-1",
        providerName: "openai",
        modelPreference: "model-1",
        source: "test",
      }),
      updateThreadSettings: async () => {},
      startTurn: async (_options, params) => {
        turnStartParams = params;
        throw new Error("Provider tool call omitted tool name");
      },
      readThread: async () => ({ status: "in_progress", turns: [] }),
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
      turnStartParams?.runtimeRequest?.metadata?.harness,
    ).not.toHaveProperty("generation");
    expect(currentChainFromError(error)).toMatchObject({
      status: "failed",
      sessionId: "deepswe-run-failed",
      turnId: "deepswe-turn-run-failed",
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
          return { session: { sessionId: "deepswe-run-step-cap" } };
        }
        if (method === "thread/read") {
          readCount += 1;
          return {
            detail: {
              turns: [
                {
                  turnId: "deepswe-turn-run-step-cap",
                  status: readCount >= 2 ? "completed" : "accepted",
                },
              ],
              items: [],
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
      updateThreadSettings: async () => {},
      startTurn: async () => {},
      cancelTurn: async () => {
        canceled = true;
      },
      readThread: async () => ({
        status: readCount >= 2 ? "completed" : "running",
        turns: [],
        providerSteps,
      }),
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

  it("does not classify a natural final answer at the step limit as exhausted", () => {
    expect(
      providerStepExhaustion({
        stepCount: 2,
        budgets: { reasons: ["provider_steps"] },
        usage: { budgetTokens: 200 },
        steps: [{ finishReason: "tool_call" }, { finishReason: "stop" }],
      }),
    ).toBeNull();
  });

  it("cancels a wall-timeout turn and captures its real terminal state", async () => {
    const root = temporaryRoot();
    const workspaceDir = path.join(root, "workspace");
    fs.mkdirSync(workspaceDir);
    let canceled = false;
    const rpc = {
      waitForHealth: async () => ({ status: "ok" }),
      invoke: async (_options, method) => {
        if (method === "workspace/ensure") {
          return { workspace: { id: "workspace-1", rootPath: workspaceDir } };
        }
        if (method === "thread/start") {
          return { session: { sessionId: "deepswe-run-wall-timeout" } };
        }
        if (method === "thread/read") {
          return {
            detail: {
              turns: [
                {
                  turnId: "deepswe-turn-run-wall-timeout",
                  status: canceled ? "canceled" : "accepted",
                },
              ],
              items: [],
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
      updateThreadSettings: async () => {},
      startTurn: async () => {},
      cancelTurn: async () => {
        canceled = true;
      },
      readThread: async () => ({
        status: canceled ? "canceled" : "running",
        turns: [],
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
    expect(error).toBeInstanceOf(Error);
    expect(error.message).toContain("cancelStatus=canceled");
    expect(currentChainFromError(error)).toMatchObject({
      status: "timeout",
      terminalStatus: "canceled",
      factsCapture: "terminal",
      timeoutCancellation: {
        reason: "wall_timeout",
        terminalStatus: "canceled",
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
    expect(fs.readFileSync(replay.solvePath, "utf8")).toContain(
      "git apply --binary --index /solution/model.patch",
    );
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
      ],
      { cwd: repoRoot, encoding: "utf8" },
    );
    expect(valid.status).toBe(0);

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

    const result = spawnSync(
      process.execPath,
      [
        "scripts/harness/deepswe-adapter.mjs",
        "--verifier-only",
        "--run-dir",
        runDir,
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
