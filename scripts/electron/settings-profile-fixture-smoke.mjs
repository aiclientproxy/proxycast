#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { writeJsonFile } from "../mcp/lib/current-smoke-transport.mjs";
import {
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  openSettings,
  parseInvokeTraceRaw,
  sanitizeText,
  waitForPageCondition,
} from "./mcp-config-fixture-smoke.mjs";
import {
  applyFailedSettingsProfileEvidence,
  applyPassingSettingsProfileEvidence,
  createSettingsProfileEvidence,
  parseSettingsProfileFixtureArgs,
  summarizeSettingsProfileTrace,
} from "./lib/settings-profile-fixture-evidence.mjs";

const DEFAULTS = {
  runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
  evidenceDir: null,
  prefix: "settings-profile-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};

function printHelp() {
  console.log(`
Settings Profile Electron Fixture

Usage:
  node scripts/electron/settings-profile-fixture-smoke.mjs --run-id <id>

Options:
  --run-id <id> --evidence-dir <path> --prefix <name>
  --timeout-ms <ms> --interval-ms <ms> --keep-temp -h|--help
`);
}

async function openProfileSettings(page, options) {
  await page.evaluate(() => {
    window.__LIME_OEM_CLOUD__ = { enabled: false };
  });
  await openSettings(page, options);
  await page.evaluate(() => {
    window.localStorage.setItem(
      "lime.app-config.changed-at",
      `settings-profile-fixture-${Date.now()}`,
    );
  });
  await page.locator('[data-testid="settings-sidebar-tab-profile"]').click();
}

async function readProfileState(page, options, { requireSave = false } = {}) {
  return await waitForPageCondition(
    page,
    options,
    ({ requireSave }) => {
      const bodyText = document.body?.innerText ?? "";
      const active =
        document
          .querySelector('[data-testid="settings-sidebar-tab-profile"]')
          ?.getAttribute("data-active") === "true";
      const editButton = document.querySelector('[aria-label="编辑昵称"]');
      const loadingVisible =
        document.querySelectorAll(".animate-pulse").length > 0;
      const errorVisible =
        bodyText.includes("资料配置尚未加载完成") ||
        bodyText.includes("保存失败:");
      const saveConfirmed = bodyText.includes("资料已保存");
      const traceRaw = window.localStorage.getItem(
        "lime_invoke_trace_buffer_v1",
      );
      let commands = [];
      let appServerMethods = [];
      try {
        const entries = JSON.parse(traceRaw || "[]");
        const safeEntries = Array.isArray(entries) ? entries : [];
        commands = safeEntries
          .filter((entry) => entry?.transport === "electron-ipc")
          .map((entry) => entry?.command);
        appServerMethods = safeEntries.flatMap((entry) => {
          if (
            entry?.command !== "app_server_handle_json_lines" ||
            entry?.transport !== "electron-ipc"
          ) {
            return [];
          }
          return Array.isArray(entry?.args_preview?.request?.lines)
            ? entry.args_preview.request.lines
            : [];
        });
      } catch {
        commands = [];
        appServerMethods = [];
      }
      if (
        !active ||
        !bodyText.includes("个人资料") ||
        !(editButton instanceof HTMLButtonElement) ||
        loadingVisible ||
        errorVisible ||
        !appServerMethods.includes("config/read") ||
        (requireSave &&
          (!saveConfirmed || !appServerMethods.includes("config/batchWrite")))
      ) {
        return null;
      }
      return {
        profileTabActive: true,
        profileEditorReady: true,
        loadingVisible,
        errorVisible,
        saveConfirmed,
        traceRaw,
        errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      };
    },
    "Profile Settings did not reach the required current state",
    { requireSave },
  );
}

async function openNicknameEditor(page) {
  await page.getByRole("button", { name: "编辑昵称", exact: true }).click();
  const input = page.locator("#profile-field-nickname");
  await input.waitFor({ state: "visible" });
  return { input, value: await input.inputValue() };
}

async function saveNickname(page, value) {
  const editor = await openNicknameEditor(page);
  await editor.input.fill(value);
  await page.getByRole("button", { name: "保存昵称", exact: true }).click();
}

async function readNicknameViaEditor(page) {
  const editor = await openNicknameEditor(page);
  const value = editor.value;
  await page.getByRole("button", { name: "取消编辑昵称", exact: true }).click();
  return value;
}

async function run() {
  const options = parseSettingsProfileFixtureArgs(process.argv.slice(2), {
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
  const restartScreenshotPath = file("-restart.png");
  const restoredScreenshotPath = file("-restored.png");
  const failureScreenshotPath = file("-failure.png");
  const runtimeEnv = createTempRuntimeEnv();
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: false,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  const summary = createSettingsProfileEvidence({
    candidateRunId: options.runId,
    startedAt: new Date().toISOString(),
    prefix: options.prefix,
  });
  let handle = null;
  let page = null;
  const consoleErrors = [];
  const pageErrors = [];
  const traceRaws = [];
  const errorRaws = [];
  const rendererSnapshots = [];
  try {
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
    });
    rendererSnapshots.push(handle.rendererSnapshot);
    page = handle.page;
    await openProfileSettings(page, options);
    await readProfileState(page, options);
    const originalNickname = await readNicknameViaEditor(page);
    const fixtureNickname = `GateProfile${Date.now()}`;
    await saveNickname(page, fixtureNickname);
    const savedState = await readProfileState(page, options, {
      requireSave: true,
    });
    traceRaws.push(savedState.traceRaw);
    errorRaws.push(savedState.errorRaw);
    await page.screenshot({ path: savedScreenshotPath, fullPage: true });
    await closeElectronFixture(handle);
    handle = null;
    page = null;

    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
    });
    rendererSnapshots.push(handle.rendererSnapshot);
    page = handle.page;
    await openProfileSettings(page, options);
    const restartState = await readProfileState(page, options);
    const restartedNickname = await readNicknameViaEditor(page);
    await page.screenshot({ path: restartScreenshotPath, fullPage: true });
    await saveNickname(page, originalNickname);
    const restoredSaveState = await readProfileState(page, options, {
      requireSave: true,
    });
    traceRaws.push(restartState.traceRaw, restoredSaveState.traceRaw);
    errorRaws.push(restartState.errorRaw, restoredSaveState.errorRaw);
    await closeElectronFixture(handle);
    handle = null;
    page = null;

    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
    });
    rendererSnapshots.push(handle.rendererSnapshot);
    page = handle.page;
    await openProfileSettings(page, options);
    const finalState = await readProfileState(page, options);
    const finalNickname = await readNicknameViaEditor(page);
    traceRaws.push(finalState.traceRaw);
    errorRaws.push(finalState.errorRaw);
    await page.screenshot({ path: restoredScreenshotPath, fullPage: true });
    const trace = summarizeSettingsProfileTrace(traceRaws);
    applyPassingSettingsProfileEvidence(summary, {
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
      localProfileMode: true,
      profileTabActive: finalState.profileTabActive,
      profileEditorReady: finalState.profileEditorReady,
      profileChanged: fixtureNickname !== originalNickname,
      saveConfirmed: savedState.saveConfirmed,
      restartReadback: restartedNickname === fixtureNickname,
      restorationSaveConfirmed: restoredSaveState.saveConfirmed,
      restorationReadback: finalNickname === originalNickname,
      loadingVisible: finalState.loadingVisible,
      errorVisible: finalState.errorVisible,
      trace,
      consoleErrors,
      pageErrors,
      invokeErrorCount: errorRaws.reduce(
        (count, raw) => count + parseInvokeTraceRaw(raw).length,
        0,
      ),
      savedScreenshotWritten: fs.existsSync(savedScreenshotPath),
      restartScreenshotWritten: fs.existsSync(restartScreenshotPath),
      restoredScreenshotWritten: fs.existsSync(restoredScreenshotPath),
    });
    writeJsonFile(rawEvidencePath, {
      lifecycle: {
        isolatedUserData: true,
        localProfileMode: true,
        profileChanged: true,
        saveConfirmed: true,
        restartReadback: true,
        restorationSaveConfirmed: true,
        restorationReadback: true,
      },
      appServerMethods: trace.methods,
      hostCommands: trace.hostCommands,
    });
    writeJsonFile(summaryPath, summary);
    console.log(`[smoke:settings-profile-fixture] summary=${summaryPath}`);
  } catch (error) {
    applyFailedSettingsProfileEvidence(summary, error);
    summary.consoleErrors = consoleErrors.map(sanitizeText);
    summary.pageErrors = pageErrors.map(sanitizeText);
    summary.failureTrace = summarizeSettingsProfileTrace(traceRaws);
    writeJsonFile(summaryPath, summary);
    if (page) {
      await page
        .screenshot({ path: failureScreenshotPath, fullPage: true })
        .catch(() => undefined);
    }
    throw error;
  } finally {
    if (handle) await closeElectronFixture(handle);
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

run().catch((error) => {
  console.error(
    `[smoke:settings-profile-fixture] ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
