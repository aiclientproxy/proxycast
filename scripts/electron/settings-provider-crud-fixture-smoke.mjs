#!/usr/bin/env node

import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { writeJsonFile } from "../mcp/lib/current-smoke-transport.mjs";
import {
  appServerCallFromPage,
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  openSettings,
  parseInvokeTraceRaw,
} from "./mcp-config-fixture-smoke.mjs";
import {
  PROVIDER_CRUD_REQUIRED_METHODS,
  applyFailedSettingsProviderCrudEvidence,
  applyPassingSettingsProviderCrudEvidence,
  createSettingsProviderCrudEvidence,
  parseSettingsProviderCrudFixtureArgs,
  summarizeSettingsProviderCrudTrace,
} from "./lib/settings-provider-crud-fixture-evidence.mjs";

const DEFAULTS = {
  runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
  evidenceDir: null,
  prefix: "settings-provider-crud-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};
const FIXTURE_PROVIDER_NAME = "Settings Provider Fixture";
const FIXTURE_MODEL = "settings-provider-model";
const INVALID_API_KEY = "settings-provider-invalid-key";
const VALID_API_KEY = "settings-provider-valid-key";

function printHelp() {
  console.log(`
Settings Provider CRUD Electron Fixture

Usage:
  node scripts/electron/settings-provider-crud-fixture-smoke.mjs --run-id <id>

Options:
  --run-id <id> --evidence-dir <path> --prefix <name>
  --timeout-ms <ms> --interval-ms <ms> --keep-temp -h|--help
`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function jsonResponse(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "content-type": "application/json; charset=utf-8",
  });
  response.end(`${JSON.stringify(payload)}\n`);
}

async function startProviderAuthFixture() {
  const state = {
    unauthorizedRequestCount: 0,
    authorizedRequestCount: 0,
  };
  const server = http.createServer((request, response) => {
    const url = new URL(request.url || "/", "http://127.0.0.1");
    const isModelsRequest =
      request.method === "GET" && url.pathname === "/v1/models";
    const isChatRequest =
      request.method === "POST" && url.pathname === "/v1/chat/completions";
    if (!isModelsRequest && !isChatRequest) {
      jsonResponse(response, 404, {
        error: { type: "not_found", message: "fixture route not found" },
      });
      return;
    }

    if (request.headers.authorization !== `Bearer ${VALID_API_KEY}`) {
      state.unauthorizedRequestCount += 1;
      jsonResponse(response, 401, {
        error: { type: "authentication_error", message: "unauthorized" },
      });
      return;
    }

    state.authorizedRequestCount += 1;
    if (isChatRequest) {
      request.resume();
      jsonResponse(response, 200, {
        id: "chatcmpl-settings-fixture",
        object: "chat.completion",
        created: 1_770_000_000,
        model: FIXTURE_MODEL,
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "READY" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      });
      return;
    }
    jsonResponse(response, 200, {
      object: "list",
      data: [
        {
          id: FIXTURE_MODEL,
          object: "model",
          created: 1_770_000_000,
          owned_by: "settings-fixture",
        },
      ],
    });
  });
  server.keepAliveTimeout = 1_000;

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    server.close();
    throw new Error("Provider auth fixture failed to bind a local port");
  }

  return {
    apiHost: `http://127.0.0.1:${address.port}/v1`,
    state,
    close: async () => {
      await new Promise((resolve, reject) => {
        server.close((error) => {
          if (error) {
            reject(error);
            return;
          }
          resolve();
        });
        server.closeIdleConnections?.();
      });
    },
  };
}

async function readInvokeBuffers(page) {
  return await page.evaluate(() => ({
    traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
    errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
  }));
}

async function waitForFixtureRequests(state, key, minimum, options) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 45_000)) {
    if (state[key] >= minimum) return;
    await sleep(options.intervalMs);
  }
  throw new Error(`Provider auth fixture did not observe ${key}`);
}

async function openProviderSettings(page, options) {
  await openSettings(page, options);
  await page.locator('[data-testid="provider-settings-title"]').waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 45_000),
  });
  await page.locator('[data-testid="api-key-provider-section"]').waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 45_000),
  });
}

async function installEvidencePrivacyStyles(page) {
  await page.addStyleTag({
    content: `
      [data-testid="provider-name"],
      [data-testid="provider-api-host-input"],
      [data-testid="provider-api-key-input"],
      [data-testid="model-provider-name-input"],
      [data-testid="model-api-host-input"],
      [data-testid="model-api-key-input"],
      [data-testid="provider-icon"],
      [data-testid="provider-simple-card"] header p,
      [data-testid="enabled-model-item"] span,
      [data-testid="model-priority-item"] span,
      [data-testid="model-priority-list"] span.normal-case {
        color: transparent !important;
        text-shadow: none !important;
      }
      [data-testid="provider-icon"],
      [data-testid="provider-simple-card"] header > div:first-child,
      [data-testid="model-fetch-status"] span {
        visibility: hidden !important;
      }
    `,
  });
}

async function waitForFixtureProvider(page, options, predicate, message) {
  const startedAt = Date.now();
  let lastCall = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 45_000)) {
    lastCall = await appServerCallFromPage(page, "modelProvider/list", {});
    const providers = Array.isArray(lastCall.result?.providers)
      ? lastCall.result.providers
      : [];
    const provider = providers.find(
      (item) => item?.name === FIXTURE_PROVIDER_NAME,
    );
    if (predicate(provider)) return { provider, call: lastCall };
    await sleep(options.intervalMs);
  }
  throw new Error(message);
}

async function selectFixtureProvider(page, options) {
  const selectedName = page.locator('[data-testid="provider-name"]');
  if (
    (await selectedName.isVisible().catch(() => false)) &&
    (await selectedName.textContent())?.trim() === FIXTURE_PROVIDER_NAME
  ) {
    return;
  }

  const item = page
    .locator('[data-testid="enabled-model-item"]')
    .filter({ hasText: FIXTURE_MODEL });
  await item.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 45_000),
  });
  await item.click();
  await page.locator('[data-testid="provider-setting"]').waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 30_000),
  });
}

async function waitForSuccessTone(locator, options, message) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 45_000)) {
    if (await locator.isVisible().catch(() => false)) {
      const className = (await locator.getAttribute("class")) || "";
      if (className.includes("emerald")) return;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(message);
}

async function createProviderAndRecoverAuth(
  page,
  options,
  fixture,
  authFailureScreenshotPath,
) {
  await page.locator('[data-testid="add-model-button"]').click();
  await page.locator('[data-testid="model-add-catalog"]').waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 30_000),
  });
  await page.locator('[data-testid="custom-provider-template-card"]').click();
  await page.locator('[data-testid="model-add-configure"]').waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 30_000),
  });
  await page
    .locator('[data-testid="model-provider-name-input"]')
    .fill(FIXTURE_PROVIDER_NAME);
  await page
    .locator('[data-testid="model-api-host-input"]')
    .fill(fixture.apiHost);
  await page
    .locator('[data-testid="model-api-key-input"]')
    .fill(INVALID_API_KEY);
  await page.locator('[data-testid="fetch-models-button"]').click();
  await waitForFixtureRequests(
    fixture.state,
    "unauthorizedRequestCount",
    1,
    options,
  );
  const failureStatus = page.locator('[data-testid="model-fetch-status"]');
  await failureStatus.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 30_000),
  });
  const failureClass = (await failureStatus.getAttribute("class")) || "";
  if (!failureClass.includes("rose")) {
    throw new Error("Provider auth failure was not visible in the GUI");
  }
  await page.screenshot({ path: authFailureScreenshotPath, fullPage: true });

  await page.locator('[data-testid="model-api-key-input"]').fill(VALID_API_KEY);
  await page.locator('[data-testid="fetch-models-button"]').click();
  await waitForFixtureRequests(
    fixture.state,
    "authorizedRequestCount",
    1,
    options,
  );
  const modelPriority = page.locator('[data-testid="model-priority-list"]');
  await modelPriority.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 30_000),
  });
  await modelPriority.getByText(FIXTURE_MODEL, { exact: true }).waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 30_000),
  });
  await waitForSuccessTone(
    page.locator('[data-testid="model-fetch-status"]'),
    options,
    "Provider auth recovery did not reach a visible success state",
  );
  await page.locator('[data-testid="model-activate-button"]').click();
  await selectFixtureProvider(page, options);
}

async function run() {
  const options = parseSettingsProviderCrudFixtureArgs(process.argv.slice(2), {
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
  const authFailureScreenshotPath = file("-auth-failure.png");
  const configuredScreenshotPath = file("-configured.png");
  const recoveredScreenshotPath = file("-recovered.png");
  const finalScreenshotPath = file("-final.png");
  const runtimeEnv = createTempRuntimeEnv();
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: false,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  const summary = createSettingsProviderCrudEvidence({
    candidateRunId: options.runId,
    startedAt: new Date().toISOString(),
    prefix: options.prefix,
  });
  const consoleErrors = [];
  const pageErrors = [];
  const rendererSnapshots = [];
  const traceRaws = [];
  const errorRaws = [];
  let electronHandle = null;
  let page = null;
  let fixture = null;
  let authFailureVisible = false;
  let connectionReady = false;
  try {
    fixture = await startProviderAuthFixture();
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
    await openProviderSettings(page, options);
    await installEvidencePrivacyStyles(page);
    await createProviderAndRecoverAuth(
      page,
      options,
      fixture,
      authFailureScreenshotPath,
    );
    authFailureVisible = fixture.state.unauthorizedRequestCount > 0;

    const connectionStatus = page.locator('[data-testid="connection-status"]');
    await page
      .locator('[data-testid="provider-test-connection-button"]')
      .click();
    await waitForFixtureRequests(
      fixture.state,
      "authorizedRequestCount",
      2,
      options,
    );
    await waitForSuccessTone(
      connectionStatus,
      options,
      "Provider connection did not recover after the API key update",
    );
    connectionReady = true;

    const configured = await waitForFixtureProvider(
      page,
      options,
      (provider) =>
        Array.isArray(provider?.models) &&
        provider.models.some((model) => model?.id === FIXTURE_MODEL) &&
        provider.apiKeyCount >= 1,
      "Provider Settings did not persist the configured model and key",
    );
    await page.screenshot({ path: configuredScreenshotPath, fullPage: true });
    const configuredBuffers = await readInvokeBuffers(page);
    traceRaws.push(configuredBuffers.traceRaw);
    errorRaws.push(configuredBuffers.errorRaw);
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
    await openProviderSettings(page, options);
    await installEvidencePrivacyStyles(page);
    const recovered = await waitForFixtureProvider(
      page,
      options,
      (provider) =>
        Array.isArray(provider?.models) &&
        provider.models.some((model) => model?.id === FIXTURE_MODEL) &&
        provider.apiKeyCount >= 1,
      "Provider Settings restart did not recover the provider",
    );
    await selectFixtureProvider(page, options);
    await page
      .locator('[data-testid="model-priority-list"]')
      .getByText(FIXTURE_MODEL, { exact: true })
      .waitFor({
        state: "visible",
        timeout: Math.min(options.timeoutMs, 30_000),
      });
    await page.screenshot({ path: recoveredScreenshotPath, fullPage: true });

    page.once("dialog", async (dialog) => {
      await dialog.accept();
    });
    await page.locator('[data-testid="provider-delete-button"]').click();
    await waitForFixtureProvider(
      page,
      options,
      (provider) => !provider,
      "Provider Settings did not delete the fixture provider",
    );
    const deletedBuffers = await readInvokeBuffers(page);
    traceRaws.push(deletedBuffers.traceRaw);
    errorRaws.push(deletedBuffers.errorRaw);
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
    await openProviderSettings(page, options);
    await installEvidencePrivacyStyles(page);
    const finalList = await waitForFixtureProvider(
      page,
      options,
      (provider) => !provider,
      "Provider Settings final restart restored a deleted provider",
    );
    const leakedText = await page
      .locator("body")
      .evaluate(
        (body, values) =>
          values.some((value) => body.textContent?.includes(value) === true),
        [FIXTURE_PROVIDER_NAME, FIXTURE_MODEL],
      );
    if (leakedText) {
      throw new Error("Deleted Provider remained visible after final restart");
    }
    await page.screenshot({ path: finalScreenshotPath, fullPage: true });
    const finalBuffers = await readInvokeBuffers(page);
    traceRaws.push(finalBuffers.traceRaw);
    errorRaws.push(finalBuffers.errorRaw);

    const trace = summarizeSettingsProviderCrudTrace(traceRaws);
    applyPassingSettingsProviderCrudEvidence(summary, {
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
      guiCreated: trace.methods.includes("modelProvider/create"),
      authFailureVisible,
      authRecovered:
        fixture.state.unauthorizedRequestCount > 0 &&
        fixture.state.authorizedRequestCount > 0,
      modelSelected:
        configured.provider?.models?.some((model) => model?.id === FIXTURE_MODEL) === true &&
        recovered.provider?.models?.some((model) => model?.id === FIXTURE_MODEL) === true,
      connectionReady,
      restartReadback: Boolean(recovered.provider),
      guiDeleted: trace.methods.includes("modelProvider/delete"),
      finalRestartAbsent: !finalList.provider,
      unauthorizedRequestCount: fixture.state.unauthorizedRequestCount,
      authorizedRequestCount: fixture.state.authorizedRequestCount,
      trace,
      consoleErrors,
      pageErrors,
      invokeErrorCount: errorRaws.reduce(
        (count, raw) => count + parseInvokeTraceRaw(raw).length,
        0,
      ),
      authFailureScreenshotWritten: fs.existsSync(authFailureScreenshotPath),
      configuredScreenshotWritten: fs.existsSync(configuredScreenshotPath),
      recoveredScreenshotWritten: fs.existsSync(recoveredScreenshotPath),
      finalScreenshotWritten: fs.existsSync(finalScreenshotPath),
    });
    writeJsonFile(rawEvidencePath, {
      lifecycle: summary.lifecycle,
      localFixture: summary.localFixture,
      appServerMethods: trace.methods.filter((method) =>
        PROVIDER_CRUD_REQUIRED_METHODS.includes(method),
      ),
    });
    writeJsonFile(summaryPath, summary);
    console.log(
      `[smoke:settings-provider-crud-fixture] summary=${summaryPath}`,
    );
  } catch (error) {
    applyFailedSettingsProviderCrudEvidence(summary, error);
    summary.failureDiagnostics = {
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
      unauthorizedRequestCount: fixture?.state.unauthorizedRequestCount ?? 0,
      authorizedRequestCount: fixture?.state.authorizedRequestCount ?? 0,
    };
    if (page) {
      await installEvidencePrivacyStyles(page).catch(() => undefined);
      const buffers = await readInvokeBuffers(page).catch(() => null);
      summary.failureTrace = summarizeSettingsProviderCrudTrace([
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
    if (fixture) await fixture.close().catch(() => undefined);
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  run().catch((error) => {
    console.error(
      `[smoke:settings-provider-crud-fixture] ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    process.exitCode = 1;
  });
}
