#!/usr/bin/env node

import fs from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import process from "node:process";
import {
  isLiveProviderTestPath,
  liveProviderSmokeAllowed,
} from "./lib/live-provider-smoke-gate.mjs";
import { isVitestRunnableTestFile } from "./lib/vitest-test-file-filter.mjs";

const repoRoot = process.cwd();
const scriptEntrypoint = path.resolve(repoRoot, "scripts/run-vitest-smart.mjs");
const vitestEntrypoint = path.resolve(
  repoRoot,
  "node_modules/vitest/vitest.mjs",
);
const SMART_STATE_SCHEMA_VERSION = 1;
const defaultStateFile = path.resolve(
  process.env.LIME_VITEST_SMART_STATE ||
    ".lime/test/vitest-smart-last-run.json",
);
const defaultBatchSize = Number.parseInt(
  process.env.LIME_VITEST_BATCH_SIZE || "16",
  10,
);
const batchSize =
  Number.isFinite(defaultBatchSize) && defaultBatchSize > 0
    ? defaultBatchSize
    : 16;
const serialTestFileOrder = [
  "scripts/lib/harness-eval-history-window.test.ts",
  "scripts/lib/harness-eval-history-record.test.ts",
  "src/components/agent/chat/index.test.tsx",
  "src/components/agent/chat/index.autoGuide01.test.tsx",
  "src/components/agent/chat/index.autoGuide02.test.tsx",
  "src/components/agent/chat/index.autoGuide03.test.tsx",
  "src/components/agent/chat/index.autoGuide04.test.tsx",
  "src/components/agent/chat/index.currentA2ui.test.tsx",
  "src/components/agent/chat/index.noVideo.test.tsx",
  "src/components/agent/chat/index.projectRestore.test.tsx",
  "src/components/agent/chat/index.serviceSkillA2ui.test.tsx",
  "src/components/agent/chat/index.shell-routing.test.tsx",
  "src/components/agent/chat/index.sidebar.test.tsx",
  "src/components/agent/chat/index.stopTeam.test.tsx",
  "src/components/agent/chat/index.taskTabs02.test.tsx",
  "src/components/agent/chat/index.workbench01.test.tsx",
  "src/components/agent/chat/index.workbench02.test.tsx",
  "src/components/agent/chat/index.workbench03.test.tsx",
  "src/components/agent/chat/index.workbench04.test.tsx",
  "src/components/agent/chat/index.workbench05.test.tsx",
  "src/components/agent/chat/hooks/useAgentChat.test.tsx",
  "src/components/workspace/WorkbenchPage.test.tsx",
  "src/components/agent/chat/components/HarnessStatusPanel.test.tsx",
  "src/components/agent/chat/components/ThemeWorkbenchSidebar.test.tsx",
  "src/components/settings-v2/system/automation/index.test.tsx",
];
const serialTestFiles = new Set(serialTestFileOrder);
const ignoredTestPathSegments = [
  "/node_modules/",
  "/tmp/lime-pnpm-frozen-node_modules/",
];
const includeLiveProviderTests = liveProviderSmokeAllowed();

let activeRunState = null;
let activeBatchIndex = null;
let activeRunStateShouldPersist = false;

function nowIso() {
  return new Date().toISOString();
}

function normalizeTestPath(file) {
  return path.resolve(file).replaceAll("\\", "/");
}

function displayPath(file) {
  return path.relative(repoRoot, file).replaceAll("\\", "/");
}

function shouldIgnoreCollectedTestFile(file) {
  const normalized = normalizeTestPath(file);
  if (ignoredTestPathSegments.some((segment) => normalized.includes(segment))) {
    return true;
  }
  return !includeLiveProviderTests && isLiveProviderTestPath(normalized);
}

function printUsage() {
  console.log(`Usage:
  npm test
  npm test -- --resume
  npm test -- --from-batch 60
  npm test -- --only-batch 60
  npm test -- --list-batches
  npm run test:related -- src/components/foo.ts
  npm run test:changed -- origin/main

Options:
  --resume          从上次 vitest-smart 状态文件中的失败、运行中或未完成批次继续
  --from-batch N    跳过前 N-1 个批次，按当前收集结果从第 N 批开始
  --only-batch N    只运行当前收集结果中的第 N 批
  --list-batches    只列出当前批次，不执行
  --json            与 --list-batches 搭配输出 JSON
  --related FILE... 使用 Vitest related，只跑与源码文件静态依赖相关的测试
  --changed [REF]   使用 Vitest --changed，只跑 Git 变更相关测试

Environment:
  LIME_VITEST_BATCH_SIZE=N         调整全量批次大小，默认 16
  LIME_VITEST_SMART_STATE=PATH     调整 --resume 状态文件路径
`);
}

export function parseSmartArgs(argv) {
  const options = {
    changed: false,
    changedRef: null,
    fromBatch: null,
    help: false,
    json: false,
    listBatches: false,
    onlyBatch: null,
    related: false,
    resume: false,
  };
  const passthroughArgs = [];

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--run") {
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      options.help = true;
      continue;
    }
    if (arg === "--resume") {
      options.resume = true;
      continue;
    }
    if (arg === "--list-batches") {
      options.listBatches = true;
      continue;
    }
    if (arg === "--json") {
      options.json = true;
      continue;
    }
    if (arg === "--related") {
      options.related = true;
      continue;
    }
    if (arg === "--changed") {
      options.changed = true;
      const nextArg = argv[index + 1];
      if (nextArg && !nextArg.startsWith("-")) {
        options.changedRef = nextArg;
        index += 1;
      }
      continue;
    }
    if (arg.startsWith("--from-batch=")) {
      options.fromBatch = parseBatchNumber(arg.slice("--from-batch=".length));
      continue;
    }
    if (arg === "--from-batch") {
      options.fromBatch = parseBatchNumber(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg.startsWith("--only-batch=")) {
      options.onlyBatch = parseBatchNumber(arg.slice("--only-batch=".length));
      continue;
    }
    if (arg === "--only-batch" || arg === "--batch") {
      options.onlyBatch = parseBatchNumber(argv[index + 1]);
      index += 1;
      continue;
    }

    passthroughArgs.push(arg);
  }

  return { options, passthroughArgs };
}

function parseBatchNumber(value) {
  const parsed = Number.parseInt(String(value || ""), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`[vitest-smart] 批次编号必须是正整数，收到：${value}`);
  }
  return parsed;
}

function assertNoFullSuiteOnlyOptions(options) {
  const hasFullSuiteOnlyOption =
    options.resume ||
    options.fromBatch !== null ||
    options.onlyBatch !== null ||
    options.listBatches;
  if ((options.related || options.changed) && hasFullSuiteOnlyOption) {
    throw new Error(
      "[vitest-smart] --resume/--from-batch/--only-batch/--list-batches 只能用于全量分批模式，不能与 --related 或 --changed 混用。",
    );
  }
}

function assertLiveProviderArgsAllowed(args) {
  if (includeLiveProviderTests) {
    return;
  }

  const liveTestArgs = args.filter((arg) => isLiveProviderTestPath(arg));
  if (liveTestArgs.length > 0) {
    throw new Error(
      `[vitest-smart] ${liveTestArgs.join(", ")} 会调用真实模型或多模态 Provider。为避免消耗额度，默认禁止执行；如确需运行，请设置 LIME_ALLOW_LIVE_PROVIDER_SMOKE=1 或 LIME_REAL_API_TEST=1。`,
    );
  }
}

function isElectronRelatedPath(arg) {
  const normalized = String(arg || "").replaceAll("\\", "/");
  return (
    normalized === "electron" ||
    normalized.startsWith("electron/") ||
    normalized.startsWith("./electron/") ||
    normalized.startsWith("scripts/electron/") ||
    normalized.startsWith("./scripts/electron/")
  );
}

function isOptionArg(arg) {
  return String(arg || "").startsWith("-");
}

function resolveExistingRelatedPath(arg) {
  if (isOptionArg(arg)) {
    return null;
  }

  const resolved = path.resolve(repoRoot, arg);
  return fs.existsSync(resolved) ? resolved : null;
}

function readRelatedSourceArgs(args) {
  return args
    .map((arg) => ({
      arg,
      resolved: resolveExistingRelatedPath(arg),
    }))
    .filter((entry) => entry.resolved !== null);
}

function resolveSiblingTestFileForSource(sourcePath) {
  const parsed = path.parse(sourcePath);
  if (/\.(test|spec)$/.test(parsed.name)) {
    return sourcePath;
  }

  const candidate = path.join(parsed.dir, `${parsed.name}.test${parsed.ext}`);
  return fs.existsSync(candidate) ? candidate : null;
}

function buildElectronRelatedRunArgs(args, sourceEntries) {
  const testFiles = [];
  const seen = new Set();
  for (const entry of sourceEntries) {
    const testFile = resolveSiblingTestFileForSource(entry.resolved);
    if (!testFile || seen.has(testFile)) {
      continue;
    }
    seen.add(testFile);
    testFiles.push(displayPath(testFile));
  }

  if (testFiles.length === 0) {
    throw new Error(
      "[vitest-smart] Electron related 模式未找到相邻 *.test.* 文件，请直接点名测试文件运行。",
    );
  }

  return [
    ...testFiles,
    ...args.filter(
      (arg) => isOptionArg(arg) || !resolveExistingRelatedPath(arg),
    ),
  ];
}

export function buildRelatedModeInvocation(args) {
  const sourceEntries = readRelatedSourceArgs(args);
  const hasSourceEntries = sourceEntries.length > 0;
  const allSourcesAreElectron =
    hasSourceEntries &&
    sourceEntries.every((entry) =>
      isElectronRelatedPath(displayPath(entry.resolved)),
    );

  if (allSourcesAreElectron) {
    return {
      command: "run",
      args: buildElectronRelatedRunArgs(args, sourceEntries),
    };
  }

  return {
    command: "related",
    args: ["--exclude", "electron/**", ...args],
  };
}

export function buildVitestCommandArgs(args, options = {}) {
  const command = options.command || "run";
  const baseArgs = [
    "--max-old-space-size=8192",
    vitestEntrypoint,
    ...(command === "related" ? ["related", "--run"] : ["--run"]),
    "--silent=passed-only",
    "--disableConsoleIntercept",
    "--poolOptions.forks.singleFork",
  ];

  return [...baseArgs, ...args];
}

function runVitest(args, label, options = {}) {
  assertLiveProviderArgsAllowed(args);

  if (label) {
    console.log(`[vitest-smart] ${label}`);
  }

  const result = spawnSync(
    process.execPath,
    buildVitestCommandArgs(args, options),
    {
      stdio: "inherit",
      env: process.env,
    },
  );

  if (result.error) {
    throw result.error;
  }

  return typeof result.status === "number" ? result.status : 1;
}

function collectTestFiles(listArgs = []) {
  const result = spawnSync(
    process.execPath,
    [vitestEntrypoint, "list", "--filesOnly", "--json", ...listArgs],
    {
      stdio: ["inherit", "pipe", "inherit"],
      env: process.env,
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    },
  );

  if (result.error) {
    throw result.error;
  }

  if (typeof result.status === "number" && result.status !== 0) {
    process.exit(result.status);
  }

  const parsed = JSON.parse(result.stdout || "[]");
  let skippedLiveTestCount = 0;
  const files = parsed
    .map((entry) => (typeof entry === "string" ? entry : entry?.file))
    .filter((entry) => typeof entry === "string" && entry.length > 0)
    .filter(isVitestRunnableTestFile)
    .filter((entry) => {
      if (!includeLiveProviderTests && isLiveProviderTestPath(entry)) {
        skippedLiveTestCount += 1;
      }
      return !shouldIgnoreCollectedTestFile(entry);
    });

  return { files, skippedLiveTestCount };
}

function chunkFiles(files, size) {
  const chunks = [];
  for (let index = 0; index < files.length; index += size) {
    chunks.push(files.slice(index, index + size));
  }
  return chunks;
}

export function buildBatches(files) {
  const serialBatches = [];
  const regularFiles = [];
  const fileByRelativePath = new Map();

  for (const file of files) {
    const relativePath = displayPath(file);
    fileByRelativePath.set(relativePath, file);
  }

  for (const relativePath of serialTestFileOrder) {
    const file = fileByRelativePath.get(relativePath);
    if (file) {
      serialBatches.push([file]);
    }
  }

  for (const file of files) {
    const relativePath = displayPath(file);
    if (serialTestFiles.has(relativePath)) {
      continue;
    }
    regularFiles.push(file);
  }

  return [...serialBatches, ...chunkFiles(regularFiles, batchSize)];
}

function createRunState({ batches, skippedLiveTestCount }) {
  const createdAt = nowIso();
  return {
    schema_version: SMART_STATE_SCHEMA_VERSION,
    runner: "vitest-smart",
    status: "running",
    repo_root: repoRoot,
    batch_size: batchSize,
    include_live_provider_tests: includeLiveProviderTests,
    skipped_live_provider_test_count: skippedLiveTestCount,
    started_at: createdAt,
    updated_at: createdAt,
    completed_at: null,
    failed_batch: null,
    batches: batches.map((files, index) => ({
      index: index + 1,
      status: "pending",
      started_at: null,
      completed_at: null,
      exit_status: null,
      files: files.map(displayPath),
    })),
  };
}

export function shouldPersistRunState(options) {
  return options.onlyBatch === null && !options.listBatches;
}

export function updateRunState(
  state,
  { persist = true, stateFile = defaultStateFile } = {},
) {
  const nextState = {
    ...state,
    updated_at: nowIso(),
  };
  if (!persist) {
    return nextState;
  }
  const directory = path.dirname(stateFile);
  fs.mkdirSync(directory, { recursive: true });
  const tempFile = `${stateFile}.tmp`;
  fs.writeFileSync(tempFile, `${JSON.stringify(nextState, null, 2)}\n`);
  fs.renameSync(tempFile, stateFile);
  return nextState;
}

function readRunState(stateFile = defaultStateFile) {
  if (!fs.existsSync(stateFile)) {
    throw new Error(
      `[vitest-smart] 找不到可恢复状态文件：${path.relative(repoRoot, stateFile)}`,
    );
  }

  const state = JSON.parse(fs.readFileSync(stateFile, "utf8"));
  if (
    state?.schema_version !== SMART_STATE_SCHEMA_VERSION ||
    state?.runner !== "vitest-smart" ||
    !Array.isArray(state?.batches)
  ) {
    throw new Error(
      `[vitest-smart] 状态文件格式不兼容：${path.relative(repoRoot, stateFile)}`,
    );
  }
  return state;
}

function hydrateBatchesFromState(state) {
  return state.batches.map((batch) =>
    batch.files.map((file) => path.resolve(repoRoot, file)),
  );
}

export function findFirstResumableBatchIndex(state) {
  const batch = state.batches.find((item) =>
    ["failed", "interrupted", "running", "pending"].includes(item.status),
  );
  return batch ? batch.index - 1 : -1;
}

export function selectBatchIndexesForRun({
  totalBatches,
  fromBatch,
  onlyBatch,
  resumeStartIndex = 0,
}) {
  if (onlyBatch !== null && onlyBatch !== undefined) {
    if (onlyBatch > totalBatches) {
      throw new Error(
        `[vitest-smart] --only-batch ${onlyBatch} 超出总批次数 ${totalBatches}`,
      );
    }
    return [onlyBatch - 1];
  }

  const startIndex =
    fromBatch !== null && fromBatch !== undefined
      ? fromBatch - 1
      : resumeStartIndex;
  if (startIndex >= totalBatches) {
    throw new Error(
      `[vitest-smart] 起始批次 ${startIndex + 1} 超出总批次数 ${totalBatches}`,
    );
  }
  return Array.from(
    { length: totalBatches - startIndex },
    (_value, index) => startIndex + index,
  );
}

function markBatch(state, batchIndex, patch) {
  const batches = state.batches.map((batch, index) =>
    index === batchIndex ? { ...batch, ...patch } : batch,
  );
  return {
    ...state,
    batches,
  };
}

export function markSkippedBatches(state, selectedIndexes) {
  const selected = new Set(selectedIndexes);
  return {
    ...state,
    batches: state.batches.map((batch, index) =>
      selected.has(index) || batch.status === "passed"
        ? batch
        : { ...batch, status: "skipped" },
    ),
  };
}

function printBatchList(state, options) {
  if (options.json) {
    console.log(JSON.stringify(state, null, 2));
    return;
  }

  console.log(
    `[vitest-smart] 共 ${state.batches.length} 批，状态文件：${path.relative(repoRoot, defaultStateFile)}`,
  );
  for (const batch of state.batches) {
    console.log(
      `${String(batch.index).padStart(3, " ")} ${batch.status.padEnd(11, " ")} ${batch.files.length} files ${batch.files[0] ?? ""}`,
    );
  }
}

function runFullSuite(options) {
  let state;
  let batches;
  let resumeStartIndex = 0;
  const persistState = shouldPersistRunState(options);
  const saveState = (nextState) =>
    updateRunState(nextState, { persist: persistState });

  if (options.resume) {
    state = readRunState();
    batches = hydrateBatchesFromState(state);
    resumeStartIndex = findFirstResumableBatchIndex(state);
    if (resumeStartIndex < 0) {
      console.log("[vitest-smart] 上次全量测试已完成，没有需要继续的批次。");
      return;
    }
  } else {
    const { files, skippedLiveTestCount } = collectTestFiles();
    if (!includeLiveProviderTests && skippedLiveTestCount > 0) {
      console.log(
        `[vitest-smart] 默认跳过 ${skippedLiveTestCount} 个 live Provider 测试；如确需运行，请设置 LIME_ALLOW_LIVE_PROVIDER_SMOKE=1 或 LIME_REAL_API_TEST=1。`,
      );
    }
    batches = buildBatches(files);
    state = createRunState({ batches, skippedLiveTestCount });
    state = saveState(state);
  }

  if (options.listBatches) {
    printBatchList(state, options);
    return;
  }

  const selectedIndexes = selectBatchIndexesForRun({
    totalBatches: batches.length,
    fromBatch: options.fromBatch,
    onlyBatch: options.onlyBatch,
    resumeStartIndex,
  });

  if (options.onlyBatch !== null || options.fromBatch !== null) {
    state = markSkippedBatches(state, selectedIndexes);
    state = saveState(state);
  }

  activeRunState = state;
  activeRunStateShouldPersist = persistState;

  for (const batchIndex of selectedIndexes) {
    activeBatchIndex = batchIndex;
    state = markBatch(state, batchIndex, {
      status: "running",
      started_at: nowIso(),
      completed_at: null,
      exit_status: null,
    });
    state = saveState(state);
    activeRunState = state;

    const status = runVitest(
      [
        "--maxWorkers",
        "1",
        "--minWorkers",
        "1",
        "--no-file-parallelism",
        ...batches[batchIndex],
      ],
      `运行批次 ${batchIndex + 1}/${batches.length}`,
    );

    if (status !== 0) {
      state = markBatch(state, batchIndex, {
        status: "failed",
        completed_at: nowIso(),
        exit_status: status,
      });
      state = {
        ...state,
        status: "failed",
        completed_at: nowIso(),
        failed_batch: batchIndex + 1,
      };
      state = saveState(state);
      activeRunState = null;
      activeBatchIndex = null;
      activeRunStateShouldPersist = false;
      console.error(
        persistState
          ? `[vitest-smart] 批次 ${batchIndex + 1}/${batches.length} 失败。修复后可执行：npm test -- --resume`
          : `[vitest-smart] 批次 ${batchIndex + 1}/${batches.length} 失败。targeted run 未改写 resume 状态。`,
      );
      process.exit(status);
    }

    state = markBatch(state, batchIndex, {
      status: "passed",
      completed_at: nowIso(),
      exit_status: 0,
    });
    state = saveState(state);
    activeRunState = state;
  }

  const hasRemaining = state.batches.some((batch) =>
    ["failed", "interrupted", "running", "pending"].includes(batch.status),
  );
  state = {
    ...state,
    status: hasRemaining ? "partial" : "passed",
    completed_at: hasRemaining ? null : nowIso(),
    failed_batch: null,
  };
  state = saveState(state);
  activeRunState = null;
  activeBatchIndex = null;
  activeRunStateShouldPersist = false;
}

function runRelatedMode(args) {
  if (args.length === 0) {
    throw new Error(
      "[vitest-smart] --related 需要至少一个源码文件，例如：npm run test:related -- src/foo.ts",
    );
  }
  const invocation = buildRelatedModeInvocation(args);
  const status = runVitest(invocation.args, "运行 related 测试", {
    command: invocation.command,
  });
  process.exit(status);
}

function runChangedMode(args, changedRef) {
  const changedArg = changedRef ? [`--changed=${changedRef}`] : ["--changed"];
  const { files, skippedLiveTestCount } = collectTestFiles([
    ...changedArg,
    ...args,
  ]);
  if (!includeLiveProviderTests && skippedLiveTestCount > 0) {
    console.log(
      `[vitest-smart] changed 模式默认跳过 ${skippedLiveTestCount} 个 live Provider 测试。`,
    );
  }
  if (files.length === 0) {
    console.log("[vitest-smart] changed 模式没有可运行的 Vitest 测试。");
    return;
  }
  const batches = buildBatches(files);
  for (const [batchIndex, batch] of batches.entries()) {
    const status = runVitest(
      [
        "--maxWorkers",
        "1",
        "--minWorkers",
        "1",
        "--no-file-parallelism",
        ...batch,
        ...args,
      ],
      `运行 changed 批次 ${batchIndex + 1}/${batches.length}`,
    );
    if (status !== 0) {
      process.exit(status);
    }
  }
}

function handleInterrupted(signal) {
  if (activeRunState && activeBatchIndex !== null) {
    activeRunState = markBatch(activeRunState, activeBatchIndex, {
      status: "interrupted",
      completed_at: nowIso(),
      exit_status: null,
    });
    activeRunState = {
      ...activeRunState,
      status: "interrupted",
      completed_at: nowIso(),
      failed_batch: activeBatchIndex + 1,
    };
    if (activeRunStateShouldPersist) {
      updateRunState(activeRunState);
    }
    console.error(
      activeRunStateShouldPersist
        ? `[vitest-smart] 收到 ${signal}，已记录中断批次。继续执行：npm test -- --resume`
        : `[vitest-smart] 收到 ${signal}，targeted run 未改写 resume 状态。`,
    );
  }
  process.exit(signal === "SIGINT" ? 130 : 143);
}

export function main(argv = process.argv.slice(2)) {
  const { options, passthroughArgs } = parseSmartArgs(argv);

  if (options.help) {
    printUsage();
    return;
  }

  assertNoFullSuiteOnlyOptions(options);

  if (options.related) {
    runRelatedMode(passthroughArgs);
    return;
  }

  if (options.changed) {
    runChangedMode(passthroughArgs, options.changedRef);
    return;
  }

  if (passthroughArgs.length > 0) {
    const status = runVitest(passthroughArgs);
    process.exit(status);
  }

  runFullSuite(options);
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptEntrypoint) {
  process.once("SIGINT", () => handleInterrupted("SIGINT"));
  process.once("SIGTERM", () => handleInterrupted("SIGTERM"));
  main();
}
