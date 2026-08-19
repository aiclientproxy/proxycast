import electronPath from "electron";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnElectron } from "../lib/electron-launcher.mjs";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import {
  readElectronSmokeSummary,
  resolveElectronSmokeExitCode,
} from "./lib/electron-smoke-summary.mjs";

const runId = normalizeRunId(
  process.env.LIME_GATE_RUN_ID?.trim() || createStandaloneRunId(),
);
const evidenceDir = path.resolve(
  process.env.LIME_ELECTRON_SMOKE_EVIDENCE_DIR?.trim() ||
    path.join(
      process.cwd(),
      ".lime/qc/project-gates",
      runId,
      "shell-01-electron-smoke",
    ),
);
const summaryPath = path.join(evidenceDir, "summary.json");
const packagedExecutable = process.env.LIME_ELECTRON_SMOKE_EXECUTABLE?.trim();
const smokeExecutable = packagedExecutable
  ? path.resolve(packagedExecutable)
  : electronPath;
if (packagedExecutable && !existsSync(smokeExecutable)) {
  throw new Error(
    `LIME_ELECTRON_SMOKE_EXECUTABLE does not exist: ${smokeExecutable}`,
  );
}
const appServerEnv = packagedExecutable
  ? { APP_SERVER_BIN: "" }
  : resolveElectronAppServerRuntimeEnv();
const userDataDir =
  process.env.ELECTRON_E2E_USER_DATA_DIR?.trim() ||
  mkdtempSync(path.join(os.tmpdir(), "lime-electron-smoke-userdata-"));
const shouldRemoveUserDataDir = !process.env.ELECTRON_E2E_USER_DATA_DIR?.trim();
const smokeVisible = process.env.LIME_ELECTRON_SMOKE_VISIBLE?.trim() === "1";
let launcherFailureStage = null;
let finished = false;
let completionPoll;
let completionKillTimer;
let completionRequested = false;

function cleanupUserDataDir() {
  if (!shouldRemoveUserDataDir) {
    return;
  }
  rmSync(userDataDir, { recursive: true, force: true });
}

function finish(exitCode, failureStage = null) {
  if (finished) {
    return;
  }
  finished = true;
  clearInterval(completionPoll);
  clearTimeout(completionKillTimer);

  if (exitCode !== 0 && !existsSync(summaryPath)) {
    writeLauncherFailureSummary(failureStage || "electron-process-exit");
  }

  const summaryResult = readElectronSmokeSummary({
    evidenceDir,
    runId,
    summaryPath,
  });
  const effectiveExitCode = resolveElectronSmokeExitCode({
    childExitCode: exitCode,
    summaryResult,
  });
  if (summaryResult.valid) {
    console.log(
      `[electron-smoke] summary run_id=${runId} result=${String(summaryResult.summary.result)} path=${summaryPath}`,
    );
  } else if (summaryResult.summary) {
    console.error(
      `[electron-smoke] structured summary failed validation: ${summaryResult.error}`,
    );
  } else {
    console.error(
      `[electron-smoke] structured summary missing or invalid: ${summaryResult.error}`,
    );
  }

  cleanupUserDataDir();
  process.exit(effectiveExitCode);
}

function writeLauncherFailureSummary(failedStage) {
  mkdirSync(evidenceDir, { recursive: true });
  const temporaryPath = `${summaryPath}.${process.pid}.tmp`;
  writeFileSync(
    temporaryPath,
    `${JSON.stringify(
      {
        schemaVersion: 1,
        scenarioId: "SHELL-01-electron-smoke",
        priority: "P0",
        proofLevel: "Gate B-F",
        claimBoundary:
          "Electron launcher lifecycle only; product bridge assertions were not completed.",
        candidateRunId: runId,
        result: "fail",
        failedStage,
        failureClass: "harness",
        nextAction:
          "Fix the Electron launcher lifecycle failure and rerun SHELL-01 on the same candidate.",
        surfaceProof: {
          surfaceId: "SHELL-01",
          proof: "gate-b-f",
          complete: false,
        },
        assertions: {
          total: 1,
          passed: 0,
          failed: [failedStage],
        },
        completedAt: new Date().toISOString(),
        artifacts: { summary: "summary.json", trace: null, screenshot: null },
      },
      null,
      2,
    )}\n`,
    "utf8",
  );
  renameSync(temporaryPath, summaryPath);
}

function normalizeRunId(value) {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)) {
    throw new Error(
      "LIME_GATE_RUN_ID 只能包含字母、数字、点、下划线和连字符，且长度不超过 128",
    );
  }
  return value;
}

function createStandaloneRunId() {
  const timestamp = new Date()
    .toISOString()
    .replace(/[^0-9]/g, "")
    .slice(0, 14);
  return `standalone-shell-01-${timestamp}-${process.pid}`;
}

const child = spawnElectron({
  electronPath: smokeExecutable,
  args: packagedExecutable
    ? ["--use-mock-keychain"]
    : ["--use-mock-keychain", "."],
  env: {
    ...process.env,
    ...appServerEnv,
    ELECTRON_E2E_USER_DATA_DIR: userDataDir,
    LIME_ELECTRON_E2E: "1",
    ...(packagedExecutable ? { LIME_ELECTRON_BRAND_DEV_APP: "0" } : {}),
    LIME_ELECTRON_SMOKE: "1",
    LIME_ELECTRON_SMOKE_VISIBLE: smokeVisible ? "1" : "0",
    LIME_GATE_RUN_ID: runId,
    LIME_ELECTRON_SMOKE_EVIDENCE_DIR: evidenceDir,
  },
  shell: packagedExecutable ? false : undefined,
});

function requestExitFromCompleteSummary() {
  if (completionRequested) {
    return true;
  }
  const summaryResult = readElectronSmokeSummary({
    evidenceDir,
    runId,
    summaryPath,
  });
  if (!summaryResult.valid) {
    return false;
  }
  completionRequested = true;
  clearInterval(completionPoll);
  child.kill();
  completionKillTimer = setTimeout(() => {
    if (!finished) {
      child.kill("SIGKILL");
    }
  }, 5_000);
  return true;
}

completionPoll = setInterval(() => {
  if (!finished) {
    requestExitFromCompleteSummary();
  }
}, 250);

const timeout = setTimeout(() => {
  if (requestExitFromCompleteSummary()) {
    return;
  }
  launcherFailureStage = "launcher-timeout";
  child.kill();
  console.error("[electron-smoke] timed out waiting for renderer/workbench");
  setTimeout(() => {
    if (finished) {
      return;
    }
    child.kill("SIGKILL");
    finish(1, launcherFailureStage);
  }, 5_000);
}, 120_000);

child.once("exit", (code) => {
  clearTimeout(timeout);
  finish(code ?? (launcherFailureStage ? 1 : 0), launcherFailureStage);
});

child.once("error", (error) => {
  clearTimeout(timeout);
  console.error(error);
  finish(1, "electron-process-error");
});
