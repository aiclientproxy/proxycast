#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { spawn } from "node:child_process";
import { pathToFileURL } from "node:url";

import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { writeJsonFile } from "../mcp/lib/current-smoke-transport.mjs";
import {
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  openSettings,
  parseInvokeTraceRaw,
} from "./mcp-config-fixture-smoke.mjs";
import {
  MEMORY_SOUL_PROFILE_ID,
  MEMORY_SOUL_REQUIRED_METHODS,
  MEMORY_SOUL_RUNTIME_MARKERS,
  applyFailedSettingsMemorySoulEvidence,
  applyPassingSettingsMemorySoulEvidence,
  createSettingsMemorySoulEvidence,
  parseSettingsMemorySoulFixtureArgs,
  summarizeSettingsMemorySoulTrace,
} from "./lib/settings-memory-soul-fixture-evidence.mjs";

const DEFAULTS = {
  runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
  evidenceDir: null,
  prefix: "settings-memory-soul-fixture",
  profileId: MEMORY_SOUL_PROFILE_ID,
  timeoutMs: 180_000,
  intervalMs: 250,
  keepTemp: false,
};

function printHelp() {
  console.log(`
Settings Memory Soul Electron Fixture

Usage:
  node scripts/electron/settings-memory-soul-fixture-smoke.mjs --run-id <id>

Options:
  --run-id <id> --evidence-dir <path> --prefix <name> --profile-id <id>
  --timeout-ms <ms> --interval-ms <ms> --keep-temp -h|--help

The fixture combines GUI save/restart evidence with the existing isolated
soul-style runtime fixture. Runtime prompt evidence is reduced to marker
booleans and never stores prompt or provider payloads.
`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function readInvokeBuffers(page) {
  return await page.evaluate(() => ({
    traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
    errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
  }));
}

async function waitForTraceCommand(page, command, options) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 45_000)) {
    const buffers = await readInvokeBuffers(page);
    const trace = summarizeSettingsMemorySoulTrace([buffers.traceRaw]);
    if (trace.configMethods.includes(command)) return trace;
    await sleep(options.intervalMs);
  }
  throw new Error(`Memory Soul GUI did not invoke ${command}`);
}

async function openMemorySoulSettings(page, options) {
  await openSettings(page, options);
  await page.locator('[data-testid="settings-sidebar-tab-memory"]').click();
  await page.locator('[data-testid="settings-memory-page"]').waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 45_000),
  });
  await page.locator('[data-testid="settings-memory-tab-soul"]').click();
  await page.locator('[data-testid="settings-memory-soul-panel"]').waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 30_000),
  });
}

async function ensureMemoryEnabled(page) {
  const toggle = page
    .locator('[data-testid="settings-memory-toggle"], [role="switch"]')
    .first();
  await toggle.waitFor({ state: "visible" });
  if ((await toggle.getAttribute("aria-checked")) !== "true") {
    await toggle.click();
  }
  if ((await toggle.getAttribute("aria-checked")) !== "true") {
    throw new Error("Memory global toggle was not enabled");
  }
}

async function assertMemorySoulEnabled(page) {
  const toggle = page
    .locator('[data-testid="settings-memory-toggle"], [role="switch"]')
    .first();
  const status = page.locator(
    '[data-testid="settings-memory-soul-current-status"]',
  );
  if ((await toggle.getAttribute("aria-checked")) !== "true") {
    throw new Error("Memory global toggle is not enabled");
  }
  const state = await status.getAttribute("data-state");
  if (state !== null) {
    if (state !== "enabled") {
      throw new Error("Soul style is not enabled");
    }
    return;
  }
  const visibleStatus = (await status.textContent()).trim().toLowerCase();
  if (
    !visibleStatus ||
    /尚未|未形成|未启用|未啟用|关闭|關閉|disabled|\boff\b/u.test(visibleStatus)
  ) {
    throw new Error("Soul style is not enabled");
  }
}

async function runSoulRuntimeFixture(options) {
  const runtimeEvidenceDir = fs.mkdtempSync(
    path.join(os.tmpdir(), "lime-settings-soul-runtime-"),
  );
  const runtimeScript = path.resolve(
    process.cwd(),
    "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
  );
  const runtimeSummaryPath = path.join(
    runtimeEvidenceDir,
    "settings-soul-runtime-summary.json",
  );
  try {
    const exitCode = await new Promise((resolve, reject) => {
      const child = spawn(
        process.execPath,
        [
          runtimeScript,
          "--run-id",
          options.runId,
          "--evidence-dir",
          runtimeEvidenceDir,
          "--prefix",
          "settings-soul-runtime",
          "--scenario",
          "soul-style",
          "--soul-style-profile",
          options.profileId,
          "--timeout-ms",
          String(options.timeoutMs),
          "--interval-ms",
          String(options.intervalMs),
        ],
        {
          cwd: process.cwd(),
          env: { ...process.env, LIME_GATE_RUN_ID: options.runId },
          stdio: ["ignore", "ignore", "ignore"],
        },
      );
      child.once("error", reject);
      child.once("exit", (code, signal) => {
        if (signal) {
          reject(new Error(`Soul runtime fixture stopped by ${signal}`));
          return;
        }
        resolve(code ?? 1);
      });
    });
    if (exitCode !== 0 || !fs.existsSync(runtimeSummaryPath)) {
      throw new Error("Soul runtime fixture did not produce a passing summary");
    }
    const runtime = JSON.parse(fs.readFileSync(runtimeSummaryPath, "utf8"));
    if (
      runtime.ok !== true ||
      runtime.scenario !== "soul-style" ||
      runtime.soulStyleExpectation?.profileId !== options.profileId
    ) {
      throw new Error("Soul runtime fixture profile or result did not pass");
    }
    return runtime;
  } finally {
    if (!options.keepTemp) {
      fs.rmSync(runtimeEvidenceDir, { recursive: true, force: true });
    }
  }
}

async function run() {
  const options = parseSettingsMemorySoulFixtureArgs(process.argv.slice(2), {
    defaults: DEFAULTS,
  });
  if (options.help) {
    printHelp();
    return;
  }
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const file = (suffix) =>
    path.join(options.evidenceDir, `${options.prefix}${suffix}`);
  const summaryPath = file("-summary.json");
  const rawEvidencePath = file("-raw.json");
  const savedScreenshotPath = file("-saved.png");
  const recoveredScreenshotPath = file("-recovered.png");
  const runtimeEnv = createTempRuntimeEnv();
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: false,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  const summary = createSettingsMemorySoulEvidence({
    candidateRunId: options.runId,
    startedAt: new Date().toISOString(),
    prefix: options.prefix,
    profileId: options.profileId,
  });
  const consoleErrors = [];
  const pageErrors = [];
  const rendererSnapshots = [];
  const traceRaws = [];
  const errorRaws = [];
  let electronHandle = null;
  let page = null;
  try {
    electronHandle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
      backendMode: "unavailable",
    });
    rendererSnapshots.push(electronHandle.rendererSnapshot);
    page = electronHandle.page;
    await openMemorySoulSettings(page, options);
    await ensureMemoryEnabled(page);
    const profileButton = page.locator(
      `[data-testid="settings-memory-soul-style-profile-${options.profileId}"]`,
    );
    await profileButton.waitFor({
      state: "visible",
      timeout: Math.min(options.timeoutMs, 45_000),
    });
    await profileButton.click();
    await page
      .locator('[data-testid="settings-memory-soul-template-direct"]')
      .click();
    await page.locator('[data-testid="settings-memory-save"]').click();
    await waitForTraceCommand(page, "config/batchWrite", options);
    await assertMemorySoulEnabled(page);
    await profileButton.waitFor({ state: "visible" });
    if ((await profileButton.getAttribute("aria-pressed")) !== "true") {
      throw new Error("Memory Soul profile was not selected after save");
    }
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
    await page.screenshot({ path: savedScreenshotPath, fullPage: true });
    const savedBuffers = await readInvokeBuffers(page);
    traceRaws.push(savedBuffers.traceRaw);
    errorRaws.push(savedBuffers.errorRaw);
    await closeElectronFixture(electronHandle);
    electronHandle = null;
    page = null;

    electronHandle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
      backendMode: "unavailable",
    });
    rendererSnapshots.push(electronHandle.rendererSnapshot);
    page = electronHandle.page;
    await openMemorySoulSettings(page, options);
    const recoveredProfileButton = page.locator(
      `[data-testid="settings-memory-soul-style-profile-${options.profileId}"]`,
    );
    await recoveredProfileButton.waitFor({
      state: "visible",
      timeout: Math.min(options.timeoutMs, 45_000),
    });
    await assertMemorySoulEnabled(page);
    if (
      (await recoveredProfileButton.getAttribute("aria-pressed")) !== "true"
    ) {
      throw new Error("Memory Soul profile did not recover after restart");
    }
    await page
      .locator('[data-testid="settings-memory-soul-current-status"]')
      .waitFor({ state: "visible" });
    await page.screenshot({ path: recoveredScreenshotPath, fullPage: true });
    const recoveredBuffers = await readInvokeBuffers(page);
    traceRaws.push(recoveredBuffers.traceRaw);
    errorRaws.push(recoveredBuffers.errorRaw);
    await closeElectronFixture(electronHandle);
    electronHandle = null;
    page = null;

    const runtime = await runSoulRuntimeFixture(options);
    const trace = summarizeSettingsMemorySoulTrace(traceRaws);
    applyPassingSettingsMemorySoulEvidence(summary, {
      completedAt: new Date().toISOString(),
      electronLaunchCount: rendererSnapshots.filter(
        (snapshot) => snapshot.electron,
      ).length,
      preloadLaunchCount: rendererSnapshots.filter(
        (snapshot) => snapshot.hasInvokeBridge,
      ).length,
      isolatedUserData: runtimeEnv.electronUserDataDir.startsWith(
        runtimeEnv.tempRoot,
      ),
      guiSaved: trace.configMethods.includes("config/batchWrite"),
      restartReadback: true,
      memoryEnabled: true,
      soulEnabled: true,
      profileSelected: true,
      runtime,
      trace,
      consoleErrors: [
        ...consoleErrors,
        ...(Array.isArray(runtime.consoleErrors) ? runtime.consoleErrors : []),
      ],
      pageErrors: [
        ...pageErrors,
        ...(Array.isArray(runtime.pageErrors) ? runtime.pageErrors : []),
      ],
      invokeErrorCount: errorRaws.reduce(
        (count, raw) => count + parseInvokeTraceRaw(raw).length,
        0,
      ),
      savedScreenshotWritten: fs.existsSync(savedScreenshotPath),
      recoveredScreenshotWritten: fs.existsSync(recoveredScreenshotPath),
    });
    writeJsonFile(rawEvidencePath, {
      lifecycle: summary.lifecycle,
      configMethods: trace.configMethods,
      appServerMethods: trace.methods.filter((method) =>
        MEMORY_SOUL_REQUIRED_METHODS.includes(method),
      ),
      runtime: {
        profileId: summary.runtime.profileId,
        markerKeys: MEMORY_SOUL_RUNTIME_MARKERS,
        markersComplete: summary.runtime.markersComplete,
        promptStored: false,
        providerRequestStored: false,
      },
    });
    writeJsonFile(summaryPath, summary);
    console.log(`[smoke:settings-memory-soul-fixture] summary=${summaryPath}`);
  } catch (error) {
    applyFailedSettingsMemorySoulEvidence(summary, error);
    summary.failureDiagnostics = {
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
    };
    if (page) {
      const buffers = await readInvokeBuffers(page).catch(() => null);
      summary.failureTrace = summarizeSettingsMemorySoulTrace([
        ...traceRaws,
        buffers?.traceRaw,
      ]);
      await page
        .screenshot({ path: file("-failure.png"), fullPage: true })
        .catch(() => undefined);
    }
    writeJsonFile(summaryPath, summary);
    throw error;
  } finally {
    if (electronHandle) await closeElectronFixture(electronHandle);
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  run().catch((error) => {
    console.error(
      `[smoke:settings-memory-soul-fixture] ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    process.exitCode = 1;
  });
}
