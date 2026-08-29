#!/usr/bin/env node

import { spawn } from "node:child_process";
import {
  accessSync,
  constants,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import {
  buildNMinusOneLaunchEnv,
  buildWaitForWindowsProcessExitScript,
  compareVersions,
  exerciseNMinusOneUpdate,
  findReadyElectronUpdaterPage,
  isFinalElectronRendererUrl,
  normalizeVersion,
  resolveInstalledSquirrelPaths,
  resolveSquirrelFeed,
  selectNMinusOneVersion,
  stopInstalledApp,
  waitForWindowsProcessExit,
} from "./lib/windows-squirrel-n-minus-one.mjs";

export {
  buildNMinusOneLaunchEnv,
  buildWaitForWindowsProcessExitScript,
  compareVersions,
  findReadyElectronUpdaterPage,
  isFinalElectronRendererUrl,
  normalizeVersion,
  resolveInstalledSquirrelPaths,
  resolveSquirrelFeed,
  selectNMinusOneVersion,
  waitForWindowsProcessExit,
};

const SCENARIO_ID = "PLT-02-windows-squirrel-rc";
const PRODUCT_NAME = "Lime";

export function selectSquirrelInstaller({ installerDir, version }) {
  const root = path.resolve(installerDir);
  const normalizedVersion = normalizeVersion(version);
  const expectedNames = [
    `${PRODUCT_NAME}-${normalizedVersion} Setup.exe`,
    `${PRODUCT_NAME}-${normalizedVersion}.Setup.exe`,
  ];
  const files = walkFiles(root);
  const exact = files.filter((filePath) =>
    expectedNames.some(
      (expectedName) =>
        path.basename(filePath).toLowerCase() === expectedName.toLowerCase(),
    ),
  );
  if (exact.length !== 1) {
    const setupCandidates = files
      .filter((filePath) => /setup\.exe$/i.test(path.basename(filePath)))
      .map((filePath) => path.relative(root, filePath));
    throw new Error(
      `expected exactly one ${expectedNames.join(" or ")} under ${root}; found ${exact.length}. ` +
        `Setup candidates: ${setupCandidates.join(", ") || "none"}`,
    );
  }
  return exact[0];
}

export function buildWindowsRcSummary({
  assertions,
  completedAt,
  error = null,
  evidence,
  failedStage = null,
  nMinusOneRequested = false,
  runId,
  startedAt,
  version,
}) {
  const failed = Object.entries(assertions)
    .filter(([, passed]) => passed !== true)
    .map(([name]) => name);
  const runnerUnavailable =
    assertions.windowsRunner === false &&
    error === `${SCENARIO_ID} requires a real Windows runner`;
  const result = runnerUnavailable
    ? "evidence-pending"
    : failed.length === 0 && !error
      ? "pass"
      : "fail";
  const nMinusOneAssertions = [
    "nMinusOneVersionOlder",
    "nMinusOneInstalled",
    "candidateFeedServed",
    "updateDownloaded",
    "updateInstallRequested",
    "candidateInstalledByUpdater",
  ];
  const nMinusOnePassed =
    nMinusOneRequested &&
    nMinusOneAssertions.every((name) => assertions[name] === true);
  return {
    schemaVersion: 1,
    scenarioId: SCENARIO_ID,
    proofLevel: "L8 platform/packaged",
    claimBoundary: nMinusOneRequested
      ? "Windows Squirrel N-1 Setup install, Electron autoUpdater download/install from an isolated candidate feed, candidate version path, shortcut/permissions, and installed candidate Lime.exe SHELL-01 Gate B. Long-duration soak is not exercised."
      : "Windows Squirrel Setup.exe install, installed application path/permissions/shortcut, and installed Lime.exe SHELL-01 Gate B. N-1 to candidate update and long-duration soak are not exercised.",
    candidateRunId: runId,
    platform: { os: process.platform, arch: process.arch, appVersion: version },
    startedAt,
    completedAt,
    result,
    failedStage:
      result === "fail"
        ? failedStage || "assertions"
        : runnerUnavailable
          ? "windows-runner"
          : null,
    error,
    blockers: runnerUnavailable
      ? ["PLT-02 requires a real Windows runner; no platform evidence was collected"]
      : [],
    assertions: {
      total: Object.keys(assertions).length,
      passed: Object.keys(assertions).length - failed.length,
      failed,
      details: assertions,
    },
    evidence,
    remainingClaims: {
      nMinusOneUpdate: nMinusOneRequested
        ? nMinusOnePassed
          ? "passed"
          : "failed"
        : "not-exercised",
      longDurationSoak: "not-exercised",
    },
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const version = normalizeVersion(
    args.version || JSON.parse(readFileSync("package.json", "utf8")).version,
  );
  const nMinusOneRequested = Boolean(
    args["n-minus-one-installer-dir"] || args["n-minus-one-version"],
  );
  const evidenceDir = path.resolve(
    args["evidence-dir"] ||
      path.join(".lime", "qc", "windows-squirrel-rc", version),
  );
  const summaryPath = path.join(evidenceDir, "summary.json");
  const runId = normalizeRunId(
    args["run-id"] || `windows-squirrel-rc-${version}-${Date.now()}`,
  );
  const startedAt = new Date().toISOString();
  const assertions = {
    windowsRunner: process.platform === "win32",
    candidateInstallerPresent: false,
    installerExitZero: false,
    installedExecutablePresent: false,
    updateExecutablePresent: false,
    installRootReadable: false,
    installRootWritable: false,
    shortcutCreated: false,
    shell01Passed: false,
    shell01VersionMatched: false,
    ...(nMinusOneRequested
      ? {
          nMinusOneVersionOlder: false,
          nMinusOneInstalled: false,
          candidateFeedServed: false,
          updateDownloaded: false,
          updateInstallRequested: false,
          candidateInstalledByUpdater: false,
        }
      : {}),
  };
  const evidence = {
    candidateInstaller: null,
    installer: null,
    installation: null,
    nMinusOneUpdate: null,
    shell01: null,
  };
  let failedStage = null;
  let errorMessage = null;

  mkdirSync(evidenceDir, { recursive: true });
  try {
    if (process.platform !== "win32") {
      throw new Error(`${SCENARIO_ID} requires a real Windows runner`);
    }

    failedStage = "candidate-installer-discovery";
    const candidateInstaller = selectSquirrelInstaller({
      installerDir: args["installer-dir"] || "release-electron",
      version,
    });
    assertions.candidateInstallerPresent = true;
    evidence.candidateInstaller = { path: candidateInstaller };

    let installVersion = version;
    let installer = candidateInstaller;
    if (nMinusOneRequested) {
      const nMinusOneVersion = normalizeVersion(args["n-minus-one-version"]);
      if (!args["n-minus-one-installer-dir"]) {
        throw new Error(
          "--n-minus-one-installer-dir is required with --n-minus-one-version",
        );
      }
      assertions.nMinusOneVersionOlder =
        compareVersions(nMinusOneVersion, version) < 0;
      if (!assertions.nMinusOneVersionOlder) {
        throw new Error(
          `N-1 version ${nMinusOneVersion} must be older than candidate ${version}`,
        );
      }
      installVersion = nMinusOneVersion;
      installer = selectSquirrelInstaller({
        installerDir: args["n-minus-one-installer-dir"],
        version: nMinusOneVersion,
      });
    }
    evidence.installer = {
      path: installer,
      version: installVersion,
      args: ["--silent"],
    };

    failedStage = "squirrel-install";
    const installResult = await runProcess(installer, ["--silent"], {
      env: process.env,
      timeoutMs: 180_000,
    });
    evidence.installer.exitCode = installResult.exitCode;
    assertions.installerExitZero = installResult.exitCode === 0;
    if (!assertions.installerExitZero) {
      throw new Error(
        `Squirrel installer exited with ${installResult.exitCode}`,
      );
    }

    failedStage = "installed-path";
    const localAppData = requiredEnv("LOCALAPPDATA");
    const baselineInstalled = resolveInstalledSquirrelPaths({
      localAppData,
      version: installVersion,
    });
    await waitFor(
      () =>
        existsSync(baselineInstalled.executable) &&
        existsSync(baselineInstalled.updateExecutable),
      { label: "installed Lime.exe and Update.exe", timeoutMs: 60_000 },
    );
    if (nMinusOneRequested) {
      assertions.nMinusOneInstalled = existsSync(baselineInstalled.executable);
    }

    failedStage = "stop-installer-launched-app";
    const baselineStop = await stopInstalledApp(baselineInstalled.executable);

    let installed = baselineInstalled;
    if (nMinusOneRequested) {
      failedStage = "n-minus-one-update";
      const updateEvidence = await exerciseNMinusOneUpdate({
        candidateFeedDir:
          args["candidate-feed-dir"] ||
          args["installer-dir"] ||
          "release-electron",
        candidateVersion: version,
        installed: baselineInstalled,
        nMinusOneVersion: installVersion,
        timeoutMs: 600_000,
      });
      evidence.nMinusOneUpdate = updateEvidence;
      assertions.candidateFeedServed = updateEvidence.candidateFeedServed;
      assertions.updateDownloaded = updateEvidence.updateDownloaded;
      assertions.updateInstallRequested = updateEvidence.updateInstallRequested;
      assertions.candidateInstalledByUpdater =
        updateEvidence.candidateInstalledByUpdater;
      installed = resolveInstalledSquirrelPaths({ localAppData, version });
    }

    failedStage = "candidate-installed-path";
    await waitFor(
      () =>
        existsSync(installed.executable) &&
        existsSync(installed.updateExecutable),
      { label: "candidate Lime.exe and Update.exe", timeoutMs: 60_000 },
    );
    assertions.installedExecutablePresent = existsSync(installed.executable);
    assertions.updateExecutablePresent = existsSync(installed.updateExecutable);

    accessSync(installed.executable, constants.R_OK);
    assertions.installRootReadable = true;
    const permissionProbe = path.join(
      installed.packageRoot,
      `.windows-rc-write-probe-${process.pid}`,
    );
    writeFileSync(permissionProbe, "ok\n", "utf8");
    rmSync(permissionProbe, { force: true });
    assertions.installRootWritable = true;

    failedStage = "squirrel-shortcut";
    const shortcutRoots = resolveShortcutRoots(process.env);
    const shortcuts = await waitFor(() => findProductShortcuts(shortcutRoots), {
      accept: (value) => value.length > 0,
      label: "Lime Squirrel shortcut",
      timeoutMs: 30_000,
    });
    assertions.shortcutCreated = shortcuts.length > 0;
    evidence.installation = {
      ...installed,
      baselineStop,
      shortcutRoots,
      shortcuts,
    };

    failedStage = "stop-updater-restarted-app";
    evidence.installation.candidateStop = await stopInstalledApp(
      installed.executable,
    );

    failedStage = "installed-shell-01";
    const shellEvidenceDir = path.join(evidenceDir, "shell-01");
    const shellSummaryPath = path.join(shellEvidenceDir, "summary.json");
    const shellResult = await runProcess(
      process.execPath,
      [path.resolve("scripts/electron/smoke.mjs")],
      {
        env: {
          ...process.env,
          APP_SERVER_BIN: "",
          LIME_ELECTRON_SMOKE_EXECUTABLE: installed.executable,
          LIME_ELECTRON_SMOKE_EVIDENCE_DIR: shellEvidenceDir,
          LIME_GATE_RUN_ID: runId,
        },
        timeoutMs: 180_000,
      },
    );
    const shellSummary = JSON.parse(readFileSync(shellSummaryPath, "utf8"));
    assertions.shell01Passed =
      shellResult.exitCode === 0 &&
      shellSummary.result === "pass" &&
      shellSummary.assertions?.failed?.length === 0;
    assertions.shell01VersionMatched =
      shellSummary.platform?.appVersion === version;
    evidence.shell01 = {
      exitCode: shellResult.exitCode,
      summaryPath: shellSummaryPath,
      result: shellSummary.result,
      scenarioId: shellSummary.scenarioId,
      proofLevel: shellSummary.proofLevel,
      appVersion: shellSummary.platform?.appVersion || null,
      bridge: shellSummary.bridge || null,
      artifacts: shellSummary.artifacts || null,
    };
    if (!assertions.shell01Passed || !assertions.shell01VersionMatched) {
      throw new Error(
        "installed Lime.exe did not pass version-matched SHELL-01",
      );
    }
  } catch (error) {
    errorMessage = error instanceof Error ? error.message : String(error);
  }

  const summary = buildWindowsRcSummary({
    assertions,
    completedAt: new Date().toISOString(),
    error: errorMessage,
    evidence,
    failedStage,
    nMinusOneRequested,
    runId,
    startedAt,
    version,
  });
  writeJsonAtomic(summaryPath, summary);
  console.log(
    `[windows-squirrel-rc] result=${summary.result} stage=${summary.failedStage || "complete"} summary=${summaryPath}`,
  );
  if (summary.result !== "pass") {
    process.exitCode = 1;
  }
}

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (!item.startsWith("--")) {
      throw new Error(`unexpected positional argument: ${item}`);
    }
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`${item} requires a value`);
    }
    args[item.slice(2)] = value;
    index += 1;
  }
  return args;
}

function walkFiles(root) {
  if (!existsSync(root)) {
    throw new Error(`installer directory does not exist: ${root}`);
  }
  const files = [];
  const pending = [root];
  while (pending.length > 0) {
    const current = pending.pop();
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const entryPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        pending.push(entryPath);
      } else if (entry.isFile()) {
        files.push(entryPath);
      }
    }
  }
  return files.sort();
}

function resolveShortcutRoots(env) {
  return [
    env.APPDATA
      ? path.join(env.APPDATA, "Microsoft", "Windows", "Start Menu", "Programs")
      : null,
    env.USERPROFILE ? path.join(env.USERPROFILE, "Desktop") : null,
  ].filter(Boolean);
}

function findProductShortcuts(roots) {
  return roots
    .filter((root) => existsSync(root))
    .flatMap((root) => walkFiles(root))
    .filter(
      (filePath) =>
        path.basename(filePath).toLowerCase() ===
        `${PRODUCT_NAME}.lnk`.toLowerCase(),
    )
    .sort();
}

async function waitFor(
  read,
  { accept = Boolean, label, timeoutMs, intervalMs = 250 },
) {
  const deadline = Date.now() + timeoutMs;
  let lastValue;
  while (Date.now() <= deadline) {
    lastValue = await read();
    if (accept(lastValue)) {
      return lastValue;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`timed out waiting for ${label}`);
}

function runProcess(command, args, { env, timeoutMs }) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      env,
      shell: false,
      stdio: "inherit",
      windowsHide: true,
    });
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill();
    }, timeoutMs);
    child.once("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.once("exit", (code, signal) => {
      clearTimeout(timeout);
      if (timedOut) {
        reject(
          new Error(`${path.basename(command)} timed out after ${timeoutMs}ms`),
        );
        return;
      }
      resolve({ exitCode: code ?? 1, signal });
    });
  });
}

function requiredEnv(name) {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`${name} is required on the Windows runner`);
  }
  return value;
}

function normalizeRunId(value) {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)) {
    throw new Error("run id contains unsupported characters or is too long");
  }
  return value;
}

function writeJsonAtomic(filePath, value) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  const temporaryDirectory = mkdtempSync(
    path.join(path.dirname(filePath), ".summary-"),
  );
  const temporaryPath = path.join(temporaryDirectory, path.basename(filePath));
  writeFileSync(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  renameSync(temporaryPath, filePath);
  rmSync(temporaryDirectory, { recursive: true, force: true });
}

const isCli =
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isCli) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
