#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_DEEPSWE_TASK,
  capturePatch,
  classifyFailure,
  createCurrentChainRpc,
  createTaskWorkspaceLocation,
  currentChainFromError,
  loadTaskDefinition,
  preflightSelectedTasks,
  prepareTaskWorkspace,
  runCurrentChainTask,
  runContextBase,
  runPierVerifier,
  runtimePrerequisites,
  timestampId,
  writeJson,
  writeRunContext,
} from "./deepswe-adapter-core.mjs";
import { createAppServerStdioTransport } from "./app-server-stdio-transport.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const localPierBin = path.join(repoRoot, ".lime/benchmark/tools/bin/pier");

function parseArgs(argv) {
  const options = {
    allowLiveProvider: false,
    appServerBin:
      process.env.LIME_DEEPSWE_APP_SERVER_BIN ||
      path.join(repoRoot, "lime-rs/target/debug/app-server"),
    appServerDataDir: process.env.LIME_DEEPSWE_APP_SERVER_DATA_DIR || "",
    containerBin: "docker",
    healthUrl: "http://127.0.0.1:3030/health",
    intervalMs: 2_000,
    invokeUrl: "http://127.0.0.1:3030/invoke",
    logPrefix: "[harness:deepswe]",
    manifestPath: "internal/test/deepswe-coding-slice-v2.json",
    maxOutputTokens: null,
    modelPreference: "",
    maxProviderSteps: 32,
    pierBin: fs.existsSync(localPierBin) ? localPierBin : "pier",
    preflight: false,
    providerPreference: "",
    runDir: "",
    runsRoot: path.join(repoRoot, ".lime/benchmark/v2/runs"),
    sliceName: "release-20",
    sourceRoot: path.join(repoRoot, ".lime/benchmark/sources/deep-swe"),
    taskId: DEFAULT_DEEPSWE_TASK,
    tokenBudget: 500_000,
    timeoutMs: 5_400_000,
    transport: process.env.LIME_DEEPSWE_TRANSPORT || "dev-bridge",
    enableThinking: null,
    verifierOnly: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      options.help = true;
      continue;
    }
    if (arg === "--allow-live-provider") {
      options.allowLiveProvider = true;
      continue;
    }
    if (arg === "--preflight") {
      options.preflight = true;
      continue;
    }
    if (arg === "--verifier-only") {
      options.verifierOnly = true;
      continue;
    }
    const valueOptions = new Map([
      ["--container-bin", "containerBin"],
      ["--app-server-bin", "appServerBin"],
      ["--app-server-data-dir", "appServerDataDir"],
      ["--health-url", "healthUrl"],
      ["--interval-ms", "intervalMs"],
      ["--invoke-url", "invokeUrl"],
      ["--manifest", "manifestPath"],
      ["--max-output-tokens", "maxOutputTokens"],
      ["--model", "modelPreference"],
      ["--max-provider-steps", "maxProviderSteps"],
      ["--pier-bin", "pierBin"],
      ["--provider", "providerPreference"],
      ["--run-dir", "runDir"],
      ["--runs-root", "runsRoot"],
      ["--slice", "sliceName"],
      ["--source-root", "sourceRoot"],
      ["--task", "taskId"],
      ["--timeout-ms", "timeoutMs"],
      ["--token-budget", "tokenBudget"],
      ["--transport", "transport"],
      ["--enable-thinking", "enableThinking"],
    ]);
    const key = valueOptions.get(arg);
    if (key && argv[index + 1]) {
      options[key] = argv[index + 1];
      index += 1;
      continue;
    }
    throw new Error(`unknown argument: ${arg}`);
  }
  options.intervalMs = Number(options.intervalMs);
  options.maxOutputTokens =
    options.maxOutputTokens == null ? null : Number(options.maxOutputTokens);
  options.maxProviderSteps = Number(options.maxProviderSteps);
  options.tokenBudget = Number(options.tokenBudget);
  options.timeoutMs = Number(options.timeoutMs);
  options.sourceRoot = path.resolve(repoRoot, options.sourceRoot);
  options.runsRoot = path.resolve(repoRoot, options.runsRoot);
  options.runDir = options.runDir ? path.resolve(repoRoot, options.runDir) : "";
  options.appServerBin = path.resolve(repoRoot, options.appServerBin);
  options.appServerDataDir = options.appServerDataDir
    ? path.resolve(repoRoot, options.appServerDataDir)
    : "";
  if (options.enableThinking != null) {
    if (!new Set(["true", "false"]).has(options.enableThinking)) {
      throw new Error("--enable-thinking must be true or false");
    }
    options.enableThinking = options.enableThinking === "true";
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms must be >= 30000");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms must be >= 100");
  }
  if (
    !Number.isSafeInteger(options.maxProviderSteps) ||
    options.maxProviderSteps < 1
  ) {
    throw new Error("--max-provider-steps must be a positive integer");
  }
  if (
    options.maxOutputTokens != null &&
    (!Number.isSafeInteger(options.maxOutputTokens) ||
      options.maxOutputTokens < 1)
  ) {
    throw new Error("--max-output-tokens must be a positive integer");
  }
  if (!Number.isSafeInteger(options.tokenBudget) || options.tokenBudget < 1) {
    throw new Error("--token-budget must be a positive integer");
  }
  if (!new Set(["dev-bridge", "stdio"]).has(options.transport)) {
    throw new Error("--transport must be dev-bridge or stdio");
  }
  return options;
}

function printHelp() {
  console.log(`
DeepSWE current-chain adapter

Usage:
  npm run harness:deepswe:preflight
  npm run harness:deepswe:run -- --task ${DEFAULT_DEEPSWE_TASK} --allow-live-provider
  npm run harness:deepswe:run -- --verifier-only --run-dir .lime/benchmark/v2/runs/<run>

Options:
  --preflight              Validate the pinned source and selected task metadata
  --task ID                Run one selected task
  --provider ID            Select a configured Lime provider
  --model MODEL             Select a configured model
  --max-output-tokens N     Override provider output limit for this diagnostic run
  --enable-thinking BOOL    Override thinking for this run: true or false
  --max-provider-steps N    Stop after N completed provider steps, default: 32
  --token-budget N          Non-cached input plus output token budget, default: 500000
  --allow-live-provider    Required before a real model turn
  --transport MODE         App Server transport: dev-bridge or stdio
  --app-server-bin PATH    App Server binary for stdio transport
  --app-server-data-dir P  Existing App Server data dir for stdio transport
  --verifier-only          Resume Pier verification for an existing run directory
  --run-dir PATH            Existing run directory for --verifier-only
  --pier-bin PATH           Pier executable, default: isolated local install or PATH
  --container-bin PATH      Container executable, default: docker
  --timeout-ms N            Agent turn timeout, default: 5400000
`);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  if (options.preflight) {
    const result = preflightSelectedTasks({
      repoRoot,
      sourceRoot: options.sourceRoot,
      sliceName: options.sliceName,
      manifestPath: options.manifestPath,
    });
    console.log(JSON.stringify(result, null, 2));
    if (result.status !== "pass") {
      process.exitCode = 1;
    }
    return;
  }

  const task = loadTaskDefinition({
    repoRoot,
    sourceRoot: options.sourceRoot,
    taskId: options.taskId,
    manifestPath: options.manifestPath,
  });
  if (options.verifierOnly) {
    if (!options.runDir) {
      throw new Error("--verifier-only requires --run-dir");
    }
    const patchPath = path.join(options.runDir, "patch.diff");
    if (!fs.existsSync(patchPath)) {
      throw new Error(`patch.diff missing: ${patchPath}`);
    }
    const runId = path.basename(options.runDir);
    const resultPath = path.join(options.runDir, "adapter-result.json");
    const existingResult = fs.existsSync(resultPath)
      ? JSON.parse(fs.readFileSync(resultPath, "utf8"))
      : {
          schemaVersion: "deepswe-adapter-result-v1",
          status: "candidate_ready_verifier_blocked",
          runId,
          runDir: options.runDir,
        };
    const verifierPrerequisites = runtimePrerequisites({
      pierBin: options.pierBin,
      containerBin: options.containerBin,
    });
    writeJson(
      path.join(options.runDir, "verifier-prerequisites.json"),
      verifierPrerequisites,
    );
    if (verifierPrerequisites.status !== "pass") {
      const error = new Error(
        `Pier verifier prerequisites blocked: ${JSON.stringify(verifierPrerequisites.checks)}`,
      );
      const failure = classifyFailure("verifier-preflight", error);
      writeJson(
        path.join(options.runDir, "verifier-failure-classification.json"),
        failure,
      );
      writeJson(resultPath, {
        ...existingResult,
        patch: existingResult.patch || {
          path: patchPath,
          bytes: fs.statSync(patchPath).size,
        },
        verifierPrerequisites,
        verification: { status: "blocked", failure },
      });
      throw error;
    }
    const verification = runPierVerifier({
      task,
      runDir: options.runDir,
      runId,
      patchPath,
      pierBin: options.pierBin,
      containerBin: options.containerBin,
    });
    writeJson(resultPath, {
      ...existingResult,
      status:
        existingResult.currentChain?.status === "failed"
          ? "verified_with_product_failure"
          : "verified",
      patch: existingResult.patch || {
        path: patchPath,
        bytes: fs.statSync(patchPath).size,
      },
      verifierPrerequisites,
      verification,
    });
    console.log(`${options.logPrefix} verified runDir=${options.runDir}`);
    return;
  }

  if (!options.allowLiveProvider) {
    throw new Error(
      "real DeepSWE execution requires --allow-live-provider; preflight remains offline",
    );
  }
  const runId = `${timestampId()}-${task.id}`;
  const runDir = path.join(options.runsRoot, runId);
  const { workspaceRoot, workspaceDir } = createTaskWorkspaceLocation({
    repoRoot,
  });
  fs.mkdirSync(runDir, { recursive: true });
  let stage = "prepare";
  let workspace = null;
  let currentChain = null;
  let patch = null;
  let verifierPrerequisites = null;
  let appServerTransport = null;
  const context = runContextBase(options, runId, task);
  writeRunContext(runDir, context);
  try {
    stage = "prepare";
    workspace = prepareTaskWorkspace({ task, workspaceDir, runId });
    let rpc;
    if (options.transport === "stdio") {
      if (!options.appServerDataDir) {
        throw new Error(
          "--transport stdio requires --app-server-data-dir or LIME_DEEPSWE_APP_SERVER_DATA_DIR",
        );
      }
      stage = "transport";
      appServerTransport = await createAppServerStdioTransport({
        repoRoot,
        binaryPath: options.appServerBin,
        dataDir: options.appServerDataDir,
        timeoutMs: options.timeoutMs,
        logPrefix: options.logPrefix,
      });
      rpc = createCurrentChainRpc({
        invoke: appServerTransport.invoke,
        waitForReady: appServerTransport.waitForReady,
      });
    }
    stage = "agent";
    currentChain = await runCurrentChainTask({
      options,
      task,
      workspaceDir,
      runDir,
      runId,
      ...(rpc ? { rpc } : {}),
    });
    stage = "patch";
    patch = capturePatch({
      workspaceDir,
      baseCommit: task.baseCommit,
      outputPath: path.join(runDir, "patch.diff"),
    });
    if (currentChain.status !== "completed") {
      stage = "agent-terminal";
      throw new Error(
        `Lime agent turn reached terminal status=${currentChain.status}${
          currentChain.terminalMessage
            ? ` message=${currentChain.terminalMessage}`
            : ""
        }`,
      );
    }
    if (currentChain.providerStepExhaustion) {
      stage = "agent-terminal";
      throw new Error(
        `DeepSWE provider budget exhausted: reasons=${currentChain.providerStepExhaustion.reasons.join(",")} steps=${currentChain.providerStepExhaustion.stepCount}`,
      );
    }
    if (patch.bytes === 0) {
      throw new Error(
        "Lime agent produced an empty patch after a completed turn",
      );
    }
    stage = "verifier-preflight";
    verifierPrerequisites = runtimePrerequisites({
      pierBin: options.pierBin,
      containerBin: options.containerBin,
    });
    writeJson(
      path.join(runDir, "verifier-prerequisites.json"),
      verifierPrerequisites,
    );
    if (verifierPrerequisites.status !== "pass") {
      throw new Error(
        `Pier verifier prerequisites blocked after Lime candidate capture: ${JSON.stringify(verifierPrerequisites.checks)}`,
      );
    }
    stage = "verifier";
    const verification = runPierVerifier({
      task,
      runDir,
      runId,
      patchPath: patch.path,
      pierBin: options.pierBin,
      containerBin: options.containerBin,
    });
    const result = {
      schemaVersion: "deepswe-adapter-result-v1",
      status: "verified",
      runId,
      runDir,
      workspace,
      currentChain,
      patch,
      verification,
    };
    writeJson(path.join(runDir, "adapter-result.json"), result);
    writeJson(path.join(runDir, "failure-classification.json"), {
      schemaVersion: "deepswe-failure-classification-v1",
      generatedAt: new Date().toISOString(),
      status: "passed",
      stage: "complete",
      owner: null,
      message: null,
    });
    console.log(`${options.logPrefix} verified runDir=${runDir}`);
  } catch (error) {
    currentChain ||= currentChainFromError(error);
    if (!patch && workspace && fs.existsSync(workspaceDir)) {
      try {
        patch = capturePatch({
          workspaceDir,
          baseCommit: task.baseCommit,
          outputPath: path.join(runDir, "patch.diff"),
        });
      } catch (patchError) {
        console.warn(
          `${options.logPrefix} diagnostic patch capture failed: ${patchError instanceof Error ? patchError.message : String(patchError)}`,
        );
      }
    }
    const failure = classifyFailure(stage, error);
    const candidateReady = Boolean(patch?.bytes > 0);
    const status =
      candidateReady && failure.owner === "verifier"
        ? "candidate_ready_verifier_blocked"
        : failure.owner === "harness"
          ? "harness_failed"
          : failure.owner === "transport"
            ? "transport_failed"
            : ["agent-runtime", "app-server", "model", "tool-runtime"].includes(
                  failure.owner,
                )
              ? "product_failed"
              : "failed";
    writeJson(path.join(runDir, "failure-classification.json"), failure);
    writeJson(path.join(runDir, "adapter-result.json"), {
      schemaVersion: "deepswe-adapter-result-v1",
      status,
      runId,
      runDir,
      workspace,
      currentChain,
      patch,
      failure,
      verifierPrerequisites,
    });
    throw error;
  } finally {
    try {
      await appServerTransport?.close();
    } catch (error) {
      console.warn(
        `${options.logPrefix} App Server stdio close failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    fs.rmSync(workspaceRoot, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(
    `[harness:deepswe] failed: ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
});
