#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "../..");
const DEFAULT_OUTPUT = path.join(
  REPO_ROOT,
  ".lime/qc/windows-restricted-execution/summary.json",
);
const SCHEMA_VERSION = "windows-restricted-execution-evidence-v3";
export const RUNNER_TIMEOUT_MS = 15 * 60 * 1000;
export const SETUP_TIMEOUT_MS = 10 * 60 * 1000;
const TEST_COMMAND = [
  "cargo",
  "test",
  "--manifest-path",
  "lime-rs/Cargo.toml",
  "-p",
  "tool-runtime",
  "--test",
  "windows_restricted_execution",
  "--",
  "--test-threads=1",
];
export const REQUIRED_TESTS = Object.freeze([
  "workspace_write_allows_workspace_and_denies_metadata_and_external_paths",
  "restricted_execution_uses_offline_account_and_blocks_network",
  "restricted_execution_bounds_large_output",
  "restricted_execution_preserves_allowlisted_stdin_handle",
  "restricted_conpty_supports_stdin_resize_and_combined_output",
  "world_writable_audit_reports_everyone_write_acl",
  "terminate_ends_restricted_process_and_its_job",
]);

function parseArgs(argv) {
  const options = { output: DEFAULT_OUTPUT, provision: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      options.help = true;
      continue;
    }
    if (arg === "--output" && argv[index + 1]) {
      options.output = path.resolve(REPO_ROOT, argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--provision") {
      options.provision = true;
      continue;
    }
    throw new Error(`unknown argument: ${arg}`);
  }
  return options;
}

function stripAnsi(value) {
  return String(value || "").replace(/\u001b\[[0-?]*[ -/]*[@-~]/gu, "");
}

export function testCounts(stdout) {
  const match = String(stdout || "").match(
    /test result:\s+(?:ok|FAILED)\.\s+(\d+) passed;\s+(\d+) failed;[\s\S]*?(\d+) ignored/u,
  );
  if (!match) {
    return { passed: null, failed: null, ignored: null };
  }
  return {
    passed: Number(match[1]),
    failed: Number(match[2]),
    ignored: Number(match[3]),
  };
}

export function parseTestCases(stdout) {
  const cases = [];
  const normalized = stripAnsi(stdout);
  const pattern = /^test\s+(.+?)\s+\.\.\.\s+(ok|FAILED|ignored)\s*$/gmu;
  for (const match of normalized.matchAll(pattern)) {
    cases.push({ name: match[1], status: match[2] });
  }
  return cases;
}

export function buildMatrixResult(cases, requiredTests = REQUIRED_TESTS) {
  const required = [...requiredTests];
  const grouped = new Map();
  for (const testCase of cases) {
    const existing = grouped.get(testCase.name) || [];
    existing.push(testCase.status);
    grouped.set(testCase.name, existing);
  }
  const missing = required.filter((name) => !grouped.has(name));
  const duplicates = required.filter(
    (name) => (grouped.get(name) || []).length > 1,
  );
  const failed = required.filter((name) =>
    (grouped.get(name) || []).some((status) => status === "FAILED"),
  );
  const ignored = required.filter((name) =>
    (grouped.get(name) || []).some((status) => status === "ignored"),
  );
  const unexpected = cases
    .map((testCase) => testCase.name)
    .filter((name) => !required.includes(name));
  return {
    required,
    observed: cases,
    missing,
    duplicates,
    failed,
    ignored,
    unexpected,
    complete:
      missing.length === 0 &&
      duplicates.length === 0 &&
      failed.length === 0 &&
      ignored.length === 0 &&
      unexpected.length === 0 &&
      cases.length === required.length,
  };
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function writeText(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, String(value || ""), "utf8");
}

export function buildEvidenceSummary({
  platform,
  startedAt,
  completedAt,
  result,
  exitCode = null,
  error = null,
  failedStage = null,
  tests = { passed: null, failed: null, ignored: null },
  matrix = buildMatrixResult([]),
  setup = { requested: false, result: "not-requested", exitCode: null, error: null },
  stdoutPath = null,
  stderrPath = null,
  setupStdoutPath = null,
  setupStderrPath = null,
}) {
  const runnerUnavailable = platform !== "win32";
  const blockers = runnerUnavailable
    ? [
        "C-3 requires a real Windows runner; no restricted execution evidence was collected",
      ]
    : [];
  if (platform === "win32" && setup.result !== "pass") {
    blockers.push(
      setup.requested
        ? `Windows sandbox setup failed: ${setup.error || `exit=${setup.exitCode}`}`
        : "Windows sandbox setup was not provisioned; rerun with --provision",
    );
  }
  if (platform === "win32" && failedStage !== "setup" && !matrix.complete) {
    blockers.push(
      `Windows restricted execution matrix incomplete: missing=${matrix.missing.join(",") || "none"}; failed=${matrix.failed.join(",") || "none"}; ignored=${matrix.ignored.join(",") || "none"}; unexpected=${matrix.unexpected.join(",") || "none"}`,
    );
  }
  return {
    schemaVersion: SCHEMA_VERSION,
    platform,
    command: TEST_COMMAND,
    startedAt,
    completedAt,
    result,
    failedStage,
    exitCode,
    error,
    blockers,
    tests,
    matrix,
    setup,
    artifacts: {
      stdout: stdoutPath,
      stderr: stderrPath,
      setupStdout: setupStdoutPath,
      setupStderr: setupStderrPath,
    },
  };
}

export function runMatrix({
  platform = process.platform,
  outputPath = DEFAULT_OUTPUT,
  provision = false,
  runner = spawnSync,
  now = () => new Date().toISOString(),
  cwd = REPO_ROOT,
  environment = process.env,
} = {}) {
  const startedAt = now();
  const outputDir = path.dirname(outputPath);
  const stdoutPath = path.join(
    outputDir,
    "windows-restricted-execution.stdout.txt",
  );
  const stderrPath = path.join(
    outputDir,
    "windows-restricted-execution.stderr.txt",
  );
  const setupStdoutPath = path.join(
    outputDir,
    "windows-sandbox-setup.stdout.txt",
  );
  const setupStderrPath = path.join(
    outputDir,
    "windows-sandbox-setup.stderr.txt",
  );

  if (platform !== "win32") {
    writeText(stdoutPath, "");
    writeText(stderrPath, "Windows runner required\n");
    writeText(setupStdoutPath, "");
    writeText(setupStderrPath, "Windows runner required\n");
    const summary = buildEvidenceSummary({
      platform,
      startedAt,
      completedAt: now(),
      result: "evidence-pending",
      failedStage: "windows-runner",
      stdoutPath: path.basename(stdoutPath),
      stderrPath: path.basename(stderrPath),
      setupStdoutPath: path.basename(setupStdoutPath),
      setupStderrPath: path.basename(setupStderrPath),
    });
    writeJson(outputPath, summary);
    return summary;
  }

  // Keep DPAPI-protected setup artifacts outside the uploaded evidence directory.
  const agentRoot = path.join(
    path.dirname(outputDir),
    `${path.basename(outputDir)}-agent-root`,
  );
  const executionEnv = {
    ...environment,
    LIME_AGENT_RUNTIME_ROOT: agentRoot,
  };
  if (!provision) {
    const setup = {
      requested: false,
      result: "not-requested",
      exitCode: null,
      error: "explicit --provision is required on a clean Windows runner",
    };
    writeText(setupStdoutPath, "");
    writeText(setupStderrPath, `${setup.error}\n`);
    writeText(stdoutPath, "");
    writeText(stderrPath, "Windows sandbox setup was not requested\n");
    const summary = buildEvidenceSummary({
      platform,
      startedAt,
      completedAt: now(),
      result: "fail",
      failedStage: "setup",
      error: setup.error,
      setup,
      stdoutPath: path.basename(stdoutPath),
      stderrPath: path.basename(stderrPath),
      setupStdoutPath: path.basename(setupStdoutPath),
      setupStderrPath: path.basename(setupStderrPath),
    });
    writeJson(outputPath, summary);
    return summary;
  }
  let setup = {
    requested: provision,
    result: provision ? "pending" : "not-requested",
    exitCode: null,
    error: null,
  };
  if (provision) {
    const username = String(environment.USERNAME || "").trim();
    const domain = String(environment.USERDOMAIN || "").trim();
    if (!username) {
      setup = {
        requested: true,
        result: "fail",
        exitCode: null,
        error: "USERNAME is missing on the Windows runner",
      };
      writeText(setupStdoutPath, "");
      writeText(setupStderrPath, `${setup.error}\n`);
    } else {
      const owner = domain ? `${domain}\\${username}` : username;
      const setupResult = runner(
        "cargo",
        [
          "run",
          "--manifest-path",
          "lime-rs/Cargo.toml",
          "-p",
          "tool-runtime",
          "--bin",
          "windows-sandbox-setup",
          "--",
          "--agent-root",
          agentRoot,
          "--owner",
          owner,
        ],
        {
          cwd,
          encoding: "utf8",
          shell: false,
          stdio: ["ignore", "pipe", "pipe"],
          timeout: SETUP_TIMEOUT_MS,
          env: executionEnv,
        },
      );
      writeText(setupStdoutPath, setupResult.stdout || "");
      writeText(setupStderrPath, setupResult.stderr || "");
      setup = {
        requested: true,
        result:
          setupResult.status === 0 && !setupResult.error ? "pass" : "fail",
        exitCode: setupResult.status,
        error: setupResult.error?.message || null,
      };
    }

    if (setup.result !== "pass") {
      writeText(stdoutPath, "");
      writeText(stderrPath, "Windows sandbox setup failed before the matrix\n");
      const summary = buildEvidenceSummary({
        platform,
        startedAt,
        completedAt: now(),
        result: "fail",
        failedStage: "setup",
        exitCode: setup.exitCode,
        error: setup.error,
        setup,
        stdoutPath: path.basename(stdoutPath),
        stderrPath: path.basename(stderrPath),
        setupStdoutPath: path.basename(setupStdoutPath),
        setupStderrPath: path.basename(setupStderrPath),
      });
      writeJson(outputPath, summary);
      return summary;
    }
  }

  const commandResult = runner(TEST_COMMAND[0], TEST_COMMAND.slice(1), {
    cwd,
    encoding: "utf8",
    shell: false,
    stdio: ["ignore", "pipe", "pipe"],
    timeout: RUNNER_TIMEOUT_MS,
    env: executionEnv,
  });
  const stdout = commandResult.stdout || "";
  const stderr = commandResult.stderr || "";
  writeText(stdoutPath, stdout);
  writeText(stderrPath, stderr);

  const spawnError = commandResult.error?.message || null;
  const testCases = parseTestCases(stdout);
  const matrix = buildMatrixResult(testCases);
  const testSummary = testCounts(stripAnsi(stdout));
  const processPassed = commandResult.status === 0 && !spawnError;
  const summary = buildEvidenceSummary({
    platform,
    startedAt,
    completedAt: now(),
    result: processPassed && matrix.complete ? "pass" : "fail",
    failedStage:
      processPassed && matrix.complete
        ? null
        : matrix.complete
          ? "test"
          : "matrix",
    exitCode: commandResult.status,
    error: spawnError,
    tests: testSummary,
    matrix,
    setup,
    stdoutPath: path.basename(stdoutPath),
    stderrPath: path.basename(stderrPath),
    setupStdoutPath: path.basename(setupStdoutPath),
    setupStderrPath: path.basename(setupStderrPath),
  });
  writeJson(outputPath, summary);
  return summary;
}

function printHelp() {
  console.log(`
Windows restricted execution evidence runner

Usage:
  node scripts/lib/windows-restricted-execution-evidence.mjs [--output PATH] [--provision]

The command is fail-closed: non-Windows hosts write evidence-pending and exit non-zero.
--provision explicitly creates/updates the Windows sandbox accounts and network rules before testing.
`);
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) {
      printHelp();
    } else {
      const summary = runMatrix({
        outputPath: options.output,
        provision: options.provision,
      });
      console.log(JSON.stringify(summary, null, 2));
      if (summary.result !== "pass") {
        process.exitCode = 1;
      }
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
